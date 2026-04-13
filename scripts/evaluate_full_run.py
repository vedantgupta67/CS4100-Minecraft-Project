"""evaluate_full_run.py
====================
Combines wood gathering policy with crafting policy to create full pipeline
Wood policy is run until 1 log is obtained, then inventory is manually opened and switches to crafting policy

Phase 1  Wood-gathering policy runs (deterministically) until ≥ 1 log
         appears in inventory, or until --max-wood-steps is reached.
Phase 2  The inventory is opened programmatically; the crafting policy
         takes over in the *same* Minecraft world to craft a crafting table.

Usage
-----
  python scripts/evaluate_full_run.py \\
      --wood-policy   checkpoints/policy_wood.pth \\
      --craft-policy  checkpoints/policy_crafting.pth \\
      --vpt-model     scripts/foundation-model-1x.model \\
      --vpt-weights   scripts/foundation-model-1x.weights \\
      [--episodes 3] [--max-wood-steps 3000] [--max-craft-steps 500] [--render]
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import gym
import minerl  # noqa: F401 — registers MineRL envs
import numpy as np
import torch
import torch.nn.functional as F

# Pull shared constants, helpers, and PPOAgent directly from the training script
# so there is no duplication and policy loading is identical.
sys.path.insert(0, os.path.dirname(__file__))
from wood_crafting_agent import (
    DISCRETE_ACTIONS,
    LOG_TYPES,
    N_ACTIONS,
    N_RECIPE_ACTIONS,
    PLANK_TYPES,
    SLOTS,
    WORLD_SEED,
    PPOAgent,
    _patch_jvm_memory,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

POV_SIZE    = 64
PITCH_MIN   = -60.0
PITCH_MAX   =  60.0
ACTION_REPEAT = 4   # matches training env


# ──────────────────────────────────────────────────────────────────────────────
# Inventory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_logs(raw_obs: dict) -> int:
    inv = raw_obs.get("inventory", {})
    return sum(int(inv.get(t, 0)) for t in LOG_TYPES)


def _count_planks(raw_obs: dict) -> int:
    inv = raw_obs.get("inventory", {})
    return sum(int(inv.get(t, 0)) for t in PLANK_TYPES)


def _count_table(raw_obs: dict) -> int:
    inv = raw_obs.get("inventory", {})
    return int(inv.get("crafting_table", 0))


# ──────────────────────────────────────────────────────────────────────────────
# Observation builders — must match each policy's training env exactly
# ──────────────────────────────────────────────────────────────────────────────

def _resize_pov(raw_obs: dict) -> np.ndarray:
    """Resize raw 640×360 POV → (3, 64, 64) uint8, matching ObservationWrapper."""
    pov = raw_obs["pov"][:, :, :3]
    pov_t = (
        torch.as_tensor(pov.copy(), dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )
    pov_t = F.interpolate(
        pov_t, size=(POV_SIZE, POV_SIZE), mode="bilinear", align_corners=False
    )
    return (pov_t.squeeze(0) * 255).byte().numpy()   # (3, 64, 64) uint8


def make_wood_obs(raw_obs: dict, total_logs: int, n_inventory: int = 4) -> dict:
    """
    Obs format matching the wood-gathering training env (AutoCraftWrapper +
    ObservationWrapper).  inventory[0] is *accumulated* logs for the episode.
    n_inventory is trimmed to match the checkpoint's actual input size.
    """
    pov = _resize_pov(raw_obs)
    full_inv = [
        float(total_logs),               # accumulated logs this episode
        float(_count_planks(raw_obs)),
        float(_count_table(raw_obs)),
        0.0,                             # gui_open = 0 during wood phase
    ]
    inv = np.array(full_inv[:n_inventory], dtype=np.float32)
    return {"pov": pov, "inventory": inv}


def make_craft_obs(raw_obs: dict, n_inventory: int = 4) -> dict:
    """
    Obs format matching the crafting training env (RecipeBookWrapper +
    ObservationWrapper).  inventory[0] is current logs in inventory;
    gui_open is always 1 because inventory is pinned open.
    n_inventory is trimmed to match the checkpoint's actual input size.
    """
    pov = _resize_pov(raw_obs)
    full_inv = [
        float(_count_logs(raw_obs)),     # current logs in inventory
        float(_count_planks(raw_obs)),
        float(_count_table(raw_obs)),
        1.0,                             # gui_open = 1 (inventory pinned open)
    ]
    inv = np.array(full_inv[:n_inventory], dtype=np.float32)
    return {"pov": pov, "inventory": inv}


def _infer_n_inventory(ckpt_path: str, device: str) -> int:
    """
    Read n_inventory directly from the saved checkpoint without constructing
    a model first.  The inventory encoder's first weight matrix has shape
    [hidden, n_inventory], so index 1 gives the input size.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    weight = ckpt["policy"]["inv_enc.0.weight"]   # shape [hidden, n_inventory]
    return weight.shape[1]


def _load_agent(n_actions: int, ckpt_path: str,
                vpt_model: str, vpt_weights: str, device: str):
    """
    Construct a PPOAgent with n_inventory inferred from the checkpoint,
    then load its weights.  Returns (agent, n_inventory).
    """
    n_inventory = _infer_n_inventory(ckpt_path, device)
    print(f"[eval]   detected n_inventory={n_inventory} from {ckpt_path}")
    agent = PPOAgent(
        n_actions=n_actions,
        n_inventory=n_inventory,
        vpt_model=vpt_model,
        vpt_weights=vpt_weights,
        device=device,
    )
    agent.load(ckpt_path)
    agent.policy.eval()
    return agent, n_inventory


# ──────────────────────────────────────────────────────────────────────────────
# Dual-phase evaluation environment
# ──────────────────────────────────────────────────────────────────────────────

class EvalEnv:
    """
    Wraps a single MineRL instance and exposes two operational phases:

    Phase 1 (wood)   step_wood(action_idx)  — survival discrete actions with
                     action-repeat × 4 and pitch clamping, matching make_env().
    Phase 2 (craft)  step_craft(action_idx) — recipe-book slot-jump actions,
                     matching make_crafting_env() / RecipeBookWrapper.

    switch_to_crafting() opens the inventory and arms phase 2.
    """

    LOG_REWARD   = 1_000.0
    PLANK_REWARD =   500.0
    TABLE_REWARD = 10_000.0

    def __init__(self, wood_n_inv: int, craft_n_inv: int, render: bool = False):
        _patch_jvm_memory("6G")
        print("[EvalEnv] Launching Minecraft Java process …")
        self._raw          = gym.make("MineRLObtainDiamondShovel-v0")
        self._render       = render
        self._wood_n_inv   = wood_n_inv    # n_inventory used by the wood policy
        self._craft_n_inv  = craft_n_inv   # n_inventory used by the crafting policy

        # Phase 1 state
        self._phase      = 1
        self._pitch      = 0.0   # for pitch clamping
        self._total_logs = 0
        self._prev_logs  = 0

        # Phase 2 state
        self._cam_pitch       = 0.0
        self._cam_yaw         = 0.0
        self._planks_rewarded = False
        self._table_rewarded  = False

    # ── Episode reset ──────────────────────────────────────────────────────

    def reset(self) -> dict:
        self._raw.seed(WORLD_SEED)
        raw_obs = self._raw.reset()

        self._phase      = 1
        self._pitch      = 0.0
        self._total_logs = 0
        self._prev_logs  = _count_logs(raw_obs)

        self._cam_pitch       = 0.0
        self._cam_yaw         = 0.0
        self._planks_rewarded = False
        self._table_rewarded  = False

        if self._render:
            self._raw.render()
        return make_wood_obs(raw_obs, self._total_logs, self._wood_n_inv)

    # ── Phase 1: wood gathering ────────────────────────────────────────────

    def step_wood(self, action_idx: int):
        """
        Execute one wood-policy step.  Internally repeats the action 4 times
        (matching ActionRepeatWrapper) and clamps pitch (matching
        PitchClampWrapper).  Returns (obs, reward, done, info).
        """
        assert self._phase == 1, "step_wood called outside of phase 1"

        noop     = self._raw.action_space.no_op()
        template = DISCRETE_ACTIONS[int(action_idx)]
        act      = dict(noop)
        for key, val in template.items():
            act[key] = np.array(val, dtype=np.float32) if key == "camera" else val

        # Pitch clamp (mirrors PitchClampWrapper)
        delta     = float(act.get("camera", [0.0, 0.0])[0])
        new_pitch = float(np.clip(self._pitch + delta, PITCH_MIN, PITCH_MAX))
        cam       = act.get("camera", np.zeros(2, dtype=np.float32)).copy()
        cam[0]    = new_pitch - self._pitch
        act["camera"] = np.array(cam, dtype=np.float32)
        self._pitch = new_pitch

        total_reward = 0.0
        done         = False
        info         = {}
        raw_obs      = None

        for _ in range(ACTION_REPEAT):
            raw_obs, _, done, info = self._raw.step(act)
            if self._render:
                self._raw.render()

            curr_logs = _count_logs(raw_obs)
            new_logs  = max(0, curr_logs - self._prev_logs)
            self._prev_logs = curr_logs
            if new_logs > 0:
                total_reward    += self.LOG_REWARD * new_logs
                self._total_logs += new_logs

            if done:
                break

        obs = make_wood_obs(raw_obs, self._total_logs, self._wood_n_inv)
        return obs, total_reward, done, info

    # ── Phase transition ───────────────────────────────────────────────────

    def switch_to_crafting(self):
        """
        Send the inventory-open key, transition internal state to phase 2,
        and return (obs, done, info) after the GUI has opened.
        """
        assert self._phase == 1, "Already in crafting phase"

        inv_act              = self._raw.action_space.no_op()
        inv_act["inventory"] = 1
        _, _, done, info = self._raw.step(inv_act)
        if self._render:
            self._raw.render()

        time.sleep(0.3)   # let the GUI finish rendering

        self._phase           = 2
        self._cam_pitch       = 0.0   # cursor resets to centre on inventory open
        self._cam_yaw         = 0.0
        self._planks_rewarded = False
        self._table_rewarded  = False

        # Take a no-op step so the first obs the crafting policy sees is a
        # clean inventory-open frame, not the transition frame from the step
        # that triggered the GUI open.
        noop = self._raw.action_space.no_op()
        raw_obs, _, done2, info2 = self._raw.step(noop)
        if self._render:
            self._raw.render()
        done = done or done2
        if done2:
            info = info2

        obs = make_craft_obs(raw_obs, self._craft_n_inv)
        return obs, done, info

    # ── Phase 2: crafting ─────────────────────────────────────────────────

    def _warp_and_click(self, target_pitch, target_yaw):
        """
        Move cursor to (target_pitch, target_yaw) camera-degrees from centre,
        then left-click.  Mirrors RecipeBookWrapper._warp_and_click exactly.
        Returns (raw_obs, done, info).
        """
        noop = self._raw.action_space.no_op()
        if target_pitch is None or target_yaw is None:
            raw_obs, _, done, info = self._raw.step(noop)
            return raw_obs, done, info

        delta_pitch = target_pitch - self._cam_pitch
        delta_yaw   = target_yaw   - self._cam_yaw
        move_act             = dict(noop)
        move_act["camera"]   = np.array([delta_pitch, delta_yaw], dtype=np.float32)
        raw_obs, _, done, info = self._raw.step(move_act)
        self._cam_pitch = target_pitch
        self._cam_yaw   = target_yaw
        if done:
            return raw_obs, done, info

        time.sleep(0.5)   # one game-tick gap before clicking
        click_act          = dict(noop)
        click_act["attack"] = 1
        raw_obs, _, done, info = self._raw.step(click_act)
        time.sleep(0.2)
        return raw_obs, done, info

    def step_craft(self, action_idx: int):
        """
        Execute one crafting-policy step.  Action mapping and reward logic
        mirror RecipeBookWrapper.step() exactly.  Returns (obs, reward, done, info).
        """
        assert self._phase == 2, "step_craft called outside of phase 2"

        noop = self._raw.action_space.no_op()
        if action_idx == 0:
            raw_obs, _, done, info = self._raw.step(noop)
        elif action_idx == 1:
            raw_obs, done, info = self._warp_and_click(*SLOTS["recipe_book_btn"])
        elif action_idx == 2:
            raw_obs, done, info = self._warp_and_click(*SLOTS["recipe_planks"])
        elif action_idx == 3:
            raw_obs, done, info = self._warp_and_click(*SLOTS["recipe_table"])
        elif action_idx == 4:
            raw_obs, done, info = self._warp_and_click(*SLOTS["output_slot"])
        elif action_idx == 5:
            raw_obs, done, info = self._warp_and_click(*SLOTS["inv_slot"])
        else:
            raw_obs, _, done, info = self._raw.step(noop)

        time.sleep(0.5)   # throttle — matches RecipeBookWrapper training rate

        if self._render:
            self._raw.render()

        reward = 0.0
        if not self._planks_rewarded and _count_planks(raw_obs) > 0:
            reward += self.PLANK_REWARD
            self._planks_rewarded = True
        if not self._table_rewarded and _count_table(raw_obs) > 0:
            reward += self.TABLE_REWARD
            self._table_rewarded = True
            done = True

        obs = make_craft_obs(raw_obs, self._craft_n_inv)
        return obs, reward, done, info

    def close(self):
        self._raw.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    wood_ckpt:       str,
    craft_ckpt:      str,
    vpt_model:       str,
    vpt_weights:     str,
    n_episodes:      int,
    max_wood_steps:  int,
    max_craft_steps: int,
    render:          bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")

    # ── Load both policies (n_inventory inferred from each checkpoint) ────
    wood_agent, wood_n_inv = _load_agent(
        N_ACTIONS, wood_ckpt, vpt_model, vpt_weights, device)
    print(f"[eval] Loaded wood policy    : {wood_ckpt}  (n_inventory={wood_n_inv})")

    craft_agent, craft_n_inv = _load_agent(
        N_RECIPE_ACTIONS, craft_ckpt, vpt_model, vpt_weights, device)
    print(f"[eval] Loaded crafting policy: {craft_ckpt}  (n_inventory={craft_n_inv})")

    # ── Create environment ────────────────────────────────────────────────
    env = EvalEnv(wood_n_inv=wood_n_inv, craft_n_inv=craft_n_inv, render=render)

    ep_results = []

    for ep in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"[eval] Episode {ep + 1} / {n_episodes}")
        print(f"{'='*60}")

        try:
            obs = env.reset()
        except Exception as exc:
            print(f"[eval] Reset failed: {exc} — skipping episode")
            continue

        ep_reward   = 0.0
        success     = False
        wood_steps  = 0
        craft_steps = 0

        # ── Phase 1: wood gathering ────────────────────────────────────────
        print("[eval] Phase 1: wood gathering …")
        got_log = False

        while wood_steps < max_wood_steps:
            action, _, _ = wood_agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step_wood(action)
            ep_reward  += reward
            wood_steps += 1

            if "error" in info:
                print(f"[eval]   env error: {info['error']}")
                done = True

            # inventory[0] is total_logs accumulated this episode
            if obs["inventory"][0] >= 1.0:
                got_log = True
                print(
                    f"[eval]   Log obtained at wood step {wood_steps} "
                    f"(episode reward so far: {ep_reward:.0f})"
                )
                break

            if done:
                print(f"[eval]   Episode ended during wood phase (step {wood_steps})")
                break

        if not got_log:
            print(
                f"[eval] Phase 1 ended without a log after {wood_steps} steps — "
                "skipping crafting phase"
            )
            ep_results.append(
                {"episode": ep + 1, "success": False,
                 "reward": ep_reward, "wood_steps": wood_steps, "craft_steps": 0}
            )
            continue

        # ── Phase transition: open inventory ──────────────────────────────
        print("[eval] Opening inventory (switching to crafting policy) …")
        try:
            obs, done, info = env.switch_to_crafting()
        except Exception as exc:
            print(f"[eval]   switch_to_crafting failed: {exc}")
            ep_results.append(
                {"episode": ep + 1, "success": False,
                 "reward": ep_reward, "wood_steps": wood_steps, "craft_steps": 0}
            )
            continue

        if done:
            print("[eval]   Episode ended during inventory open")
            ep_results.append(
                {"episode": ep + 1, "success": False,
                 "reward": ep_reward, "wood_steps": wood_steps, "craft_steps": 0}
            )
            continue

        # ── Phase 2: crafting ─────────────────────────────────────────────
        ACTION_NAMES = {
            0: "no-op",
            1: "recipe_book_btn",
            2: "recipe_planks",
            3: "recipe_table",
            4: "output_slot",
            5: "inv_slot",
        }
        print("[eval] Phase 2: crafting …")
        action_tally = {k: 0 for k in ACTION_NAMES}

        while craft_steps < max_craft_steps:
            # Use stochastic sampling — the policy was trained stochastically, and
            # deterministic argmax on a partially-trained policy collapses to one
            # action every step, making the agent appear to do nothing.
            action, _, _ = craft_agent.select_action(obs, deterministic=False)
            action_tally[action] = action_tally.get(action, 0) + 1
            print(f"[craft] step {craft_steps:3d}: {ACTION_NAMES.get(action, action)}"
                  f"  inv={obs['inventory']}")
            obs, reward, done, info = env.step_craft(action)
            ep_reward   += reward
            craft_steps += 1

            if "error" in info:
                print(f"[eval]   env error: {info['error']}")
                break

            # inventory[2] is the crafting_table count
            if obs["inventory"][2] >= 1.0:
                success = True
                print(
                    f"[eval]   Crafting table obtained at craft step {craft_steps}!"
                )
                break

            if done:
                break

        print(f"[eval] Craft action tally: { {ACTION_NAMES[k]: v for k, v in action_tally.items()} }")

        result_str = "SUCCESS" if success else "FAILED"
        print(
            f"\n[eval] Episode {ep + 1} {result_str}: "
            f"total_reward={ep_reward:.0f}  "
            f"wood_steps={wood_steps}  craft_steps={craft_steps}"
        )
        ep_results.append(
            {"episode": ep + 1, "success": success,
             "reward": ep_reward, "wood_steps": wood_steps, "craft_steps": craft_steps}
        )

    # ── Summary ───────────────────────────────────────────────────────────
    env.close()

    print(f"\n{'='*60}")
    print("[eval] Summary")
    print(f"{'='*60}")
    if ep_results:
        successes = [r["success"] for r in ep_results]
        rewards   = [r["reward"]  for r in ep_results]
        print(f"  Episodes run   : {len(ep_results)}/{n_episodes}")
        print(f"  Success rate   : {sum(successes)}/{len(successes)}")
        print(f"  Mean reward    : {np.mean(rewards):.0f}")
        for r in ep_results:
            status = "OK" if r["success"] else "--"
            print(
                f"  [{status}] ep {r['episode']:2d}: "
                f"reward={r['reward']:.0f}  "
                f"wood={r['wood_steps']}  craft={r['craft_steps']}"
            )
    print("[eval] Done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--wood-policy", required=True,
        help="Path to wood-gathering policy checkpoint (.pth)",
    )
    ap.add_argument(
        "--craft-policy", required=True,
        help="Path to crafting policy checkpoint (.pth)",
    )
    ap.add_argument(
        "--vpt-model", default=None,
        help="VPT foundation model file (.model)",
    )
    ap.add_argument(
        "--vpt-weights", default=None,
        help="VPT foundation model weights file (.weights)",
    )
    ap.add_argument(
        "--episodes", type=int, default=3,
        help="Number of evaluation episodes (default: 3)",
    )
    ap.add_argument(
        "--max-wood-steps", type=int, default=3000,
        help="Max steps for the wood-gathering phase per episode (default: 3000)",
    )
    ap.add_argument(
        "--max-craft-steps", type=int, default=500,
        help="Max steps for the crafting phase per episode (default: 500)",
    )
    ap.add_argument(
        "--render", action="store_true",
        help="Render the Minecraft window after each step",
    )
    args = ap.parse_args()

    evaluate(
        wood_ckpt=args.wood_policy,
        craft_ckpt=args.craft_policy,
        vpt_model=args.vpt_model,
        vpt_weights=args.vpt_weights,
        n_episodes=args.episodes,
        max_wood_steps=args.max_wood_steps,
        max_craft_steps=args.max_craft_steps,
        render=args.render,
    )


if __name__ == "__main__":
    main()
