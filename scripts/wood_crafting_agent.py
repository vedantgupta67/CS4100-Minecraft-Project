"""
wood_crafting_agent.py
======================
PPO agent that learns to:
  1. Chop wood from trees (collect logs)
  2. Craft planks from logs
  3. Craft a crafting table from planks

Architecture
------------
- MineRLObtainDiamondShovel-v0 (survival world, has inventory obs, uses
    LowLevelInputsAgentStart required by MC 1.16.5) wrapped with:
    * AutoCraftWrapper  — reward shaping for log collection + crafting
    * DiscreteActionWrapper — 13 discrete actions (survival / gathering)
    * ObservationWrapper    — CHW image + virtual inventory vector
- Crafting env uses RecipeBookWrapper (6 discrete slot-jump actions):
    * Agent opens inventory, clicks recipe book, selects recipe, collects result
    * Cursor jumps directly to calibrated pixel positions — no incremental moves
- Actor-Critic CNN + Inventory encoder (PolicyNet)
- PPO with GAE for training

Usage
-----
  python scripts/wood_crafting_agent.py                                         # Defaults to wood gathering training
  pypython scripts/wood_crafting_agent.py --env-mode crafting                   # Train crafting
  python scripts/wood_crafting_agent.py --eval checkpoints/policy_final.pth     # Evaluate trained policy

TensorBoard
-----------
  # TensorBoard logging is ON by default. To view while training:
  #   Terminal 1 — start training:
  python scripts/wood_crafting_agent.py

  #   Terminal 2 — launch TensorBoard (logs go to <save-dir>/runs/ by default):
  tensorboard --logdir checkpoints/runs/

  # Then open http://localhost:6006 in your browser

  # Optional TensorBoard flags:
  #   --tensorboard-dir <path>    Override the log directory (default: tb_logs
  #   --tb-flush-every <N>        Flush writer every N PPO updates (default: 10)
  #   --tb-image-every <N>        Log a training frame every N completed episodes (default: 1)
  #   --tb-video-every <N>        Log an eval video every N eval cycles (default: 3)
  #   --tb-video-frames <N>       Max frames per logged eval video (default: 64)

"""

import argparse
import os
import threading
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import gym
import minerl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Environment
# ──────────────────────────────────────────────────────────────────────────────
#
# Root cause of previous timeouts: MineRLTreechop-v0 uses SimpleHumanEmbodiment-
# EnvSpec which omits LowLevelInputsAgentStart from the mission XML.
# MC 1.16.5 / MCP-Reborn requires that tag to initialise keyboard input; without
# it the server never acknowledges the mission → TimeoutError at env.reset().
#
# MineRLObtainDiamondShovel-v0 is based on HumanSurvival (→ HumanControlEnvSpec)
# which DOES include LowLevelInputsAgentStart.  It also:
#   • uses DefaultWorldGenerator(force_reset=True) — compatible with MC 1.16.5
#   • includes FlatInventoryObservation so we can read log counts directly
#   • spawns in a random survival world with trees nearby
#
# Log types in MC 1.16.5 inventory:
LOG_TYPES = [
    "oak_log", "spruce_log", "birch_log",
    "jungle_log", "acacia_log", "dark_oak_log",
]

PLANK_TYPES = [
    "oak_planks", "spruce_planks", "birch_planks",
    "jungle_planks", "acacia_planks", "dark_oak_planks",
]

# [logs, planks, crafting_table, gui_open] — all sourced from actual MC inventory
OBS_ITEMS = ["logs", "planks", "crafting_table", "gui_open"]

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Discrete-action wrapper
# ──────────────────────────────────────────────────────────────────────────────

# Each entry is a dict of action-key -> value overrides on top of a no-op base.
# 'camera' values are [pitch_delta, yaw_delta] in degrees.
# ── Survival / wood-gathering policy actions ──────────────────────────────────
DISCRETE_ACTIONS = [
    {},                                     # 0:  no-op
    {"forward": 1},                         # 1:  walk forward
    {"back": 1},                            # 2:  walk back
    {"left": 1},                            # 3:  strafe left
    {"right": 1},                           # 4:  strafe right
    {"jump": 1},                            # 5:  jump
    {"sprint": 1, "forward": 1},            # 6:  sprint forward
    {"attack": 1},                          # 7:  attack / mine
    {"attack": 1, "forward": 1},            # 8:  mine while moving forward
    {"camera": [0.0, -15.0]},              # 9:  turn left
    {"camera": [0.0,  15.0]},              # 10: turn right
    {"camera": [-15.0,  0.0]},             # 11: look up
    {"camera": [ 15.0,  0.0]},             # 12: look down
]
N_ACTIONS = len(DISCRETE_ACTIONS)


class DiscreteActionWrapper(gym.ActionWrapper):
    """Maps integer action indices to MineRL Dict actions.

    Defaults to DISCRETE_ACTIONS (survival / gathering policy).
    """

    def __init__(self, env, action_list=None):
        super().__init__(env)
        self._actions = action_list if action_list is not None else DISCRETE_ACTIONS
        self._base = None
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def _get_base(self):
        if self._base is None:
            self._base = self.env.action_space.no_op()
        return self._base

    def action(self, action_idx):
        act = dict(self._get_base())          # shallow copy
        for key, val in self._actions[int(action_idx)].items():
            if key == "camera":
                act["camera"] = np.array(val, dtype=np.float32)
            else:
                act[key] = val
        return act



class PitchClampWrapper(gym.Wrapper):
    """
    Tracks the agent's estimated pitch and clamps it to [PITCH_MIN, PITCH_MAX].

    When the agent would exceed a limit, the pitch delta is zeroed out so the
    action becomes a no-op in that axis.  This prevents the agent from spinning
    to look straight up or down and getting stuck there.

    Minecraft's default pitch range: -90 (straight up) to +90 (straight down).
    We restrict to [-60, 60] so the agent always sees the horizon.
    """

    PITCH_MIN = -60.0
    PITCH_MAX =  60.0

    def __init__(self, env):
        super().__init__(env)
        self._pitch = 0.0   # estimated current pitch in degrees

    def reset(self):
        self._pitch = 0.0
        return self.env.reset()

    def step(self, action):
        delta = float(action.get("camera", [0.0, 0.0])[0])
        new_pitch = self._pitch + delta
        if new_pitch < self.PITCH_MIN:
            action = dict(action)
            action["camera"] = np.array(
                [self.PITCH_MIN - self._pitch, action["camera"][1]], dtype=np.float32)
            new_pitch = self.PITCH_MIN
        elif new_pitch > self.PITCH_MAX:
            action = dict(action)
            action["camera"] = np.array(
                [self.PITCH_MAX - self._pitch, action["camera"][1]], dtype=np.float32)
            new_pitch = self.PITCH_MAX
        self._pitch = new_pitch
        return self.env.step(action)


class AutoCraftWrapper(gym.Wrapper):
    """
    Reward shaping for collecting wood:
      
    Small positive reward for mining, decays over time
        - This trains the agent to learn to mine for extended periods of time in order to mine long enough to gather a log
          while also giving a negative reward for mining for too long to prevent teaching the agent to mine and do nothing else

    Large reward for collecting a log

    Small pentalty for dying

    Pentalty for mining dirt to teach the agent what to mine
    """

    LOG_REWARD             = 1000.0   # per log collected
    DEATH_PENALTY          =   -5.0
    MINE_DIRT              =  -10.0
    ATTACK_REWARD_INITIAL  =    0.01
    ATTACK_REWARD_MIN      =   -0.05
    ATTACK_DECAY_AMOUNT    =    0.001

    MAX_CONSECUTIVE_RESETS = 5

    def __init__(self, env):
        super().__init__(env)
        self._prev_logs      = 0
        self._prev_planks    = 0
        self._prev_table     = 0
        self._prev_dirt      = 0
        self._total_logs     = 0
        self._got_inv_bonus  = False
        self._was_alive      = True
        self._consecutive_resets      = 0
        self._attack_reward_current   = self.ATTACK_REWARD_INITIAL

    @staticmethod
    def _count_logs(obs) -> int:
        inv = obs.get("inventory", {})
        return sum(int(inv.get(t, 0)) for t in LOG_TYPES)

    @staticmethod
    def _count_dirt(obs) -> int:
        inv = obs.get("inventory", {})
        return int(inv.get("dirt", 0))

    def _attach(self, obs):
        obs["_logs"]     = self._total_logs
        obs["_planks"]   = self._count_planks(obs)
        obs["_table"]    = self._count_table(obs)
        obs["_gui_open"] = int(obs.get("isGuiOpen", 0))
        return obs

    def reset(self):
        obs = self.env.reset()
        self._prev_logs               = self._count_logs(obs)
        self._prev_planks             = self._count_planks(obs)
        self._prev_table              = self._count_table(obs)
        self._prev_dirt               = 0
        self._total_logs              = 0
        self._got_inv_bonus           = False
        self._was_alive               = True
        self._consecutive_resets      = 0
        self._attack_reward_current   = self.ATTACK_REWARD_INITIAL
        return self._attach(obs)

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = 0.0

        # ── Death ──────────────────────────────────────────────────────────
        is_alive = bool(obs.get("life_stats", {}).get("is_alive", True))
        if self._was_alive and not is_alive:
            reward += self.DEATH_PENALTY
        self._was_alive = is_alive

        # ── Log collection ─────────────────────────────────────────────────
        curr_logs = self._count_logs(obs)
        new_logs  = max(0, curr_logs - self._prev_logs)
        self._prev_logs = curr_logs
        if new_logs > 0:
            reward += self.LOG_REWARD * new_logs
            self._total_logs += new_logs

        # ── Dirt penalty ───────────────────────────────────────────────────
        curr_dirt = self._count_dirt(obs)
        new_dirt  = max(0, curr_dirt - self._prev_dirt)
        self._prev_dirt = curr_dirt
        if new_dirt > 0:
            reward += self.MINE_DIRT * new_dirt

        # ── Attack decay (discourages fruitless mining) ────────────────────
        if action.get("attack", 0):
            reward += self._attack_reward_current
            self._attack_reward_current = max(
                self.ATTACK_REWARD_MIN,
                self._attack_reward_current - self.ATTACK_DECAY_AMOUNT,
            )
        else:
            self._attack_reward_current = self.ATTACK_REWARD_INITIAL

        # ── Episode-reset handling ─────────────────────────────────────────
        if done and curr_logs == 0:   # done without success → try resetting
            self._consecutive_resets += 1
            if self._consecutive_resets <= self.MAX_CONSECUTIVE_RESETS:
                try:
                    obs = self.env.reset()
                    self._prev_logs   = self._count_logs(obs)
                    self._prev_planks = self._count_planks(obs)
                    self._prev_table  = self._count_table(obs)
                    self._was_alive   = True
                    done = False
                except Exception:
                    pass
            else:
                print("[env] Too many resets → ending episode")
        else:
            self._consecutive_resets = 0

        return self._attach(obs), reward, done, info

class ObservationWrapper(gym.ObservationWrapper):
    """
    Converts the MineRL obs dict into:
      pov:       np.uint8  (3, 64, 64)  — CHW image, resized from 640×640
      inventory: np.float32 (4,)        — [logs, planks, crafting_table, gui_open]
    """

    POV_SIZE = 64  # target spatial resolution for the CNN

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "pov": gym.spaces.Box(
                low=0, high=255,
                shape=(3, self.POV_SIZE, self.POV_SIZE),
                dtype=np.uint8,
            ),
            "inventory": gym.spaces.Box(
                low=0, high=2304, shape=(len(OBS_ITEMS),), dtype=np.float32
            ),
        })

    def observation(self, obs):
        pov = obs["pov"][:, :, :3]           # drop alpha channel if present

        # Resize 640×640 → POV_SIZE × POV_SIZE using torch
        pov_t = torch.as_tensor(pov.copy(), dtype=torch.float32
                                ).permute(2, 0, 1).unsqueeze(0) / 255.0
        pov_t = torch.nn.functional.interpolate(
            pov_t, size=(self.POV_SIZE, self.POV_SIZE),
            mode="bilinear", align_corners=False,
        )
        pov = (pov_t.squeeze(0) * 255).byte().numpy()   # (3, 64, 64) uint8

        inv = np.array([
            float(obs.get("_logs",     0)),   # logs collected this episode
            float(obs.get("_planks",   0)),   # actual planks in inventory
            float(obs.get("_table",    0)),   # actual crafting_table count
            float(obs.get("_gui_open", 0)),   # 1 if inventory/GUI is open
        ], dtype=np.float32)
        return {"pov": pov, "inventory": inv}


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat each chosen action for `repeat` environment steps.

    Rewards are summed across repeated steps; the last observation is returned.
    Episode terminates early if `done` is True during a repeat.

    A repeat of 8 means one network decision = 8 game ticks, so a single
    "attack" choice sustains the swing long enough to noticeably damage a log
    and the agent gets a reward signal much sooner.
    """

    def __init__(self, env, repeat: int = 16):
        super().__init__(env)
        self._repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


WORLD_SEED = 175901257196164


class FixedSeedWrapper(gym.Wrapper):
    """Re-seeds the environment with a fixed value before every reset so the
    same world is generated each episode."""

    def reset(self):
        self.env.seed(WORLD_SEED)
        return self.env.reset()


# ──────────────────────────────────────────────────────────────────────────────
# Crafting-specific wrappers
# ──────────────────────────────────────────────────────────────────────────────

# GUI slot positions as camera-degree offsets from screen centre.
# Values are (pitch_delta, yaw_delta) — the exact values to send as the
# camera action from the centre position (where Minecraft places the cursor
# when the inventory opens).  No pixel-conversion factor needed.
SLOTS = {
    "recipe_book_btn": (-2, 4),   # (pitch, yaw) — green book icon
    "recipe_planks":   (-6, -20),   # (pitch, yaw) — planks recipe entry
    "recipe_table":    (-6, -12),   # (pitch, yaw) — crafting table entry
    "output_slot":     (-6, 23),   # (pitch, yaw) — output slot
    "inv_slot":        (0, 0),   # (pitch, yaw) — first inventory slot
}

N_RECIPE_ACTIONS = 6


class RecipeBookWrapper(gym.Wrapper):
    """
    Discrete crafting via the recipe book.

    The inventory is opened once at the start of each episode and stays pinned
    open — the agent has no way to close it.

    Action space (N_RECIPE_ACTIONS = 6):
      0  no-op
      1  click recipe book button
      2  click 'planks' recipe
      3  click 'crafting table' recipe
      4  click output slot (collect result)
      5  click first inventory slot (deposit held item)

    Click actions are wrapped to be one action with buttons. The agent will move to a button and then click it.
    The agent sees a single step for each. 

    Cursor position is tracked in camera-degree space from screen centre,
    which is where Minecraft places the cursor when the inventory opens.

    Reward:
      +PLANK_REWARD  per plank that appears in inventory
      +TABLE_REWARD  crafting table obtained → ends episode
    """

    PLANK_REWARD = 500.0
    TABLE_REWARD = 10_000.0

    def __init__(self, env):
        super().__init__(env)
        self._cam_pitch = 0.0   
        self._cam_yaw   = 0.0
        self._planks_rewarded = False
        self._table_rewarded  = False
        self.action_space  = gym.spaces.Discrete(N_RECIPE_ACTIONS)

    @staticmethod
    def _count_planks(obs):
        inv = obs.get("inventory", {})
        return sum(int(inv.get(t, 0)) for t in PLANK_TYPES)

    @staticmethod
    def _count_table(obs):
        inv = obs.get("inventory", {})
        return int(inv.get("crafting_table", 0))

    @staticmethod
    def _count_logs(obs):
        inv = obs.get("inventory", {})
        return sum(int(inv.get(t, 0)) for t in LOG_TYPES)

    def _attach(self, obs):
        obs["_logs"]     = self._count_logs(obs)
        obs["_planks"]   = self._count_planks(obs)
        obs["_table"]    = self._count_table(obs)
        obs["_gui_open"] = 1
        return obs

    def reset(self):
        obs = self.env.reset()
        # Open the inventory and leave it pinned — no toggle action exists.
        open_act = self.env.action_space.no_op()
        open_act["inventory"] = 1
        obs, _, _, _ = self.env.step(open_act)
        # Minecraft resets cursor to screen centre on GUI open → degrees = 0
        self._cam_pitch = 0.0
        self._cam_yaw   = 0.0
        self._planks_rewarded = False
        self._table_rewarded  = False
        return self._attach(obs)

    _warned_uncalibrated = False

    def _warp_and_click(self, target_pitch, target_yaw):
        """Move cursor to (target_pitch, target_yaw) degrees from centre, then left-click."""
        noop = self.env.action_space.no_op()
        if target_pitch is None or target_yaw is None:
            if not RecipeBookWrapper._warned_uncalibrated:
                print("[RecipeBookWrapper] SLOTS not calibrated — click actions are no-ops. "
                      "Run scripts/calibrate_gui.py to find the correct values.")
                RecipeBookWrapper._warned_uncalibrated = True
            obs, r, done, info = self.env.step(noop)
            return obs, r, done, info

        delta_pitch = target_pitch - self._cam_pitch
        delta_yaw   = target_yaw   - self._cam_yaw
        move_act = dict(noop)
        move_act["camera"] = np.array([delta_pitch, delta_yaw], dtype=np.float32)
        obs, r1, done, info = self.env.step(move_act)
        self._cam_pitch = target_pitch
        self._cam_yaw   = target_yaw
        if done:
            return obs, r1, done, info

        time.sleep(0.5)   # give Minecraft time to process
        click_act = dict(noop)
        click_act["attack"] = 1
        obs, r2, done, info = self.env.step(click_act)
        return obs, r1 + r2, done, info

    def step(self, action):
        noop = self.env.action_space.no_op()

        if action == 0:
            obs, r, done, info = self.env.step(noop)
        elif action == 1:
            obs, r, done, info = self._warp_and_click(*SLOTS["recipe_book_btn"])
        elif action == 2:
            obs, r, done, info = self._warp_and_click(*SLOTS["recipe_planks"])
        elif action == 3:
            obs, r, done, info = self._warp_and_click(*SLOTS["recipe_table"])
        elif action == 4:
            obs, r, done, info = self._warp_and_click(*SLOTS["output_slot"])
        elif action == 5:
            obs, r, done, info = self._warp_and_click(*SLOTS["inv_slot"])
        else:
            obs, r, done, info = self.env.step(noop)

        time.sleep(1)   # give minecraft time to process

        reward = 0.0

        if not self._planks_rewarded and self._count_planks(obs) > 0:
            reward += self.PLANK_REWARD
            self._planks_rewarded = True

        if not self._table_rewarded and self._count_table(obs) > 0:
            reward += self.TABLE_REWARD
            self._table_rewarded = True
            done = True

        return self._attach(obs), reward, done, info


class MaxStepsWrapper(gym.Wrapper):
    """Terminates the episode after ``max_steps`` agent steps."""

    def __init__(self, env, max_steps: int):
        super().__init__(env)
        self._max_steps = max_steps
        self._steps = 0

    def reset(self):
        self._steps = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps += 1
        if self._steps >= self._max_steps:
            done = True
        return obs, reward, done, info


def _patch_jvm_memory(max_mem: str = "6G"):
    """
    Monkey-patch MinecraftInstance to use a larger JVM heap before gym.make().
    The default is 4G which leaks and freezes after thousands of steps.
    This avoids modifying any minerl source files.
    """
    try:
        import minerl.env.malmo as _malmo
        _orig = _malmo.MinecraftInstance.__init__
        def _patched(self, port=None, existing=False, status_dir=None,
                     seed=None, instance_id=None, max_mem=None):
            _orig(self, port=port, existing=existing, status_dir=status_dir,
                  seed=seed, instance_id=instance_id, max_mem=max_mem or "6G")
        _malmo.MinecraftInstance.__init__ = _patched
        print(f"[make_env] JVM heap set to {max_mem}")
    except Exception as e:
        print(f"[make_env] Warning: could not patch JVM memory: {e}")


def make_env():
    """Survival env — agent must chop logs AND craft the table from scratch."""
    _patch_jvm_memory("6G")
    print("[make_env] Launching Minecraft Java process (can take 2–5 min on first run) …")
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env.seed(WORLD_SEED)
    print("[make_env] gym.make() done — wrapping env …")
    env = FixedSeedWrapper(env)
    env = AutoCraftWrapper(env)
    env = PitchClampWrapper(env)
    env = DiscreteActionWrapper(env)
    env = ActionRepeatWrapper(env, repeat=4)
    env = ObservationWrapper(env)
    return env


def make_crafting_env():
    """
    Crafting-only env — trains the crafting policy in isolation.

    Episode setup
    -------------
    - Agent spawns with 1 oak_log pre-loaded in slot 0 (InventoryAgentStart).
    - RecipeBookWrapper exposes 6 discrete actions: no-op, toggle inventory,
      click recipe book, click planks recipe, click crafting table recipe,
      click output slot.  Calibrate SLOTS before training.

    Action space (N_RECIPE_ACTIONS = 6)
    ------------------------------------
    Discrete slot-jump actions — the cursor warps directly to the target
    position in one tick then clicks.  No incremental camera movement.

    Reward
    ------
    RecipeBookWrapper: +500 per plank gained, +10 000 for crafting table.

    ----------------------------
    base → FixedSeed → RecipeBook → MaxSteps → Obs
    """
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from minerl.herobraine.hero.handlers.agent.start import InventoryAgentStart
    from minerl.env._singleagent import _SingleAgentEnv

    class CraftingEnvSpec(HumanSurvival):
        def __init__(self):
            super().__init__(name="CraftingTask-v0", max_episode_steps=2400)

        def create_agent_start(self):
            handlers = super().create_agent_start()
            handlers.append(InventoryAgentStart({
                0: {"type": "oak_log", "quantity": 1}
            }))
            return handlers

    _patch_jvm_memory("6G")
    print("[make_crafting_env] Launching Minecraft Java process …")
    spec = CraftingEnvSpec()
    env = _SingleAgentEnv(env_spec=spec)
    env.seed(WORLD_SEED)
    print("[make_crafting_env] env created — wrapping …")
    env = FixedSeedWrapper(env)
    env = RecipeBookWrapper(env)
    env = MaxStepsWrapper(env, max_steps=8000)
    env = ObservationWrapper(env)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PPO agent
# ──────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(
        self,
        n_actions:      int,
        n_inventory:    int,
        vpt_model:      str,
        vpt_weights:    str,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        clip_eps:       float = 0.2,
        ent_coef:       float = 0.01,
        vf_coef:        float = 0.5,
        max_grad_norm:  float = 0.5,
        n_epochs:       int   = 4,
        minibatch_size: int   = 64,
        device:         str   = "cpu",
    ):
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.clip_eps       = clip_eps
        self.ent_coef       = ent_coef
        self.vf_coef        = vf_coef
        self.max_grad_norm  = max_grad_norm
        self.n_epochs       = n_epochs
        self.minibatch_size = minibatch_size
        self.device         = device

        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(__file__))
        from vpt_policy import load_vpt_policy
        self.policy = load_vpt_policy(
            vpt_model_path=vpt_model,
            vpt_weights_path=vpt_weights,
            n_actions=n_actions,
            n_inventory=n_inventory,
            freeze_cnn=True,
            device=device,
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False):
        """Return (action_idx, log_prob, value) given a single obs dict."""
        pov = torch.as_tensor(obs["pov"], dtype=torch.float32, device=self.device
                              ).unsqueeze(0) / 255.0
        inv = torch.as_tensor(obs["inventory"], dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        if deterministic:
            logits, value = self.policy.forward(pov, inv)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action, log_prob, _, value = self.policy.get_action_and_value(pov, inv)
        return action.item(), log_prob.item(), value.item()

    # ------------------------------------------------------------------
    @staticmethod
    def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
        """Compute Generalised Advantage Estimation."""
        n          = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        for t in reversed(range(n)):
            next_val  = next_value if t == n - 1 else values[t + 1]
            next_done = dones[t]
            delta     = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
            gae       = delta + gamma * gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    def update(self, rollout: dict, next_obs: dict) -> dict:
        """Run PPO update epochs from one rollout."""
        # Bootstrap value for the last state
        with torch.no_grad():
            _, _, next_val = self.select_action(next_obs)

        advantages, returns = self.compute_gae(
            rollout["rewards"], rollout["values"],
            rollout["dones"],   next_val,
            self.gamma,         self.gae_lambda,
        )

        # Convert rollout to tensors
        povs     = (torch.as_tensor(np.array(rollout["povs"]),
                                    dtype=torch.float32, device=self.device) / 255.0)
        invs     = torch.as_tensor(np.array(rollout["invs"]),
                                   dtype=torch.float32, device=self.device)
        actions  = torch.as_tensor(rollout["actions"],
                                   dtype=torch.long,    device=self.device)
        old_lps  = torch.as_tensor(rollout["log_probs"],
                                   dtype=torch.float32, device=self.device)
        advs     = torch.as_tensor(advantages,
                                   dtype=torch.float32, device=self.device)
        rets     = torch.as_tensor(returns,
                                   dtype=torch.float32, device=self.device)

        # Normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        n       = len(actions)
        indices = np.arange(n)
        losses  = {"pg": 0.0, "vf": 0.0, "ent": 0.0}

        # When the CNN is frozen its output is constant — pre-compute features
        # once for the whole rollout instead of re-running it every minibatch
        # across every epoch. Cuts PPO update time by ~(n_epochs * n_minibatches)x.
        frozen_feats = None
        if hasattr(self.policy, "cnn_is_frozen") and self.policy.cnn_is_frozen():
            with torch.no_grad():
                frozen_feats = self.policy.cnn(povs)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start : start + self.minibatch_size]

                if frozen_feats is not None:
                    _, new_lps, entropy, new_vals = self.policy.forward_from_features(
                        frozen_feats[idx], invs[idx], actions[idx]
                    )
                else:
                    _, new_lps, entropy, new_vals = self.policy.get_action_and_value(
                        povs[idx], invs[idx], actions[idx]
                    )

                ratio = torch.exp(new_lps - old_lps[idx])
                adv_b = advs[idx]

                # Clipped policy gradient loss
                pg1  = -adv_b * ratio
                pg2  = -adv_b * torch.clamp(ratio, 1.0 - self.clip_eps,
                                                    1.0 + self.clip_eps)
                pg_loss  = torch.max(pg1, pg2).mean()
                vf_loss  = F.mse_loss(new_vals, rets[idx])
                ent_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                losses["pg"]  += pg_loss.item()
                losses["vf"]  += vf_loss.item()
                losses["ent"] += ent_loss.item()

        n_updates = self.n_epochs * max(1, n // self.minibatch_size)
        return {k: v / n_updates for k, v in losses.items()}

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def _reset_with_timeout(env, timeout_sec: int = 360):
    """
    Call env.reset() on a background thread.  If it does not return within
    timeout_sec seconds (Java process frozen), raise TimeoutError so the
    training loop can save a checkpoint and exit rather than hanging forever.
    """
    result, exc = [None], [None]
    def _worker():
        try:
            result[0] = env.reset()
        except Exception as e:
            exc[0] = e
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_sec)
    if t.is_alive():
        raise TimeoutError(
            f"env.reset() did not complete within {timeout_sec}s — "
            "Minecraft Java process likely frozen"
        )
    if exc[0]:
        raise exc[0]
    return result[0]


def _evaluate_policy(
    agent: PPOAgent,
    n_episodes: int,
    max_steps: int,
    video_frames: int,
    capture_video: bool,
    env_mode: str = "crafting",
):
    if n_episodes <= 0:
        return {
            "mean_reward": 0.0,
            "mean_steps": 0.0,
            "success_rate": 0.0,
            "sample_frame": None,
            "captured_frames": [],
        }

    rewards = []
    lengths = []
    successes = []
    sample_frame = None
    captured_frames = []

    _env_factory = make_crafting_env if env_mode == "crafting" else make_env
    for ep in range(n_episodes):
        env = _env_factory()
        try:
            obs = _reset_with_timeout(env)
            total_rew = 0.0
            steps = 0
            done = False
            crafted = False

            while not done and steps < max_steps:
                action, _, _ = agent.select_action(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_rew += reward
                steps += 1

                crafted = crafted or bool(obs["inventory"][2] >= 1.0)  # crafting_table slot

                if sample_frame is None and "pov" in obs:
                    sample_frame = obs["pov"]

                if capture_video and len(captured_frames) < video_frames and "pov" in obs:
                    captured_frames.append(obs["pov"])

            rewards.append(float(total_rew))
            lengths.append(int(steps))
            successes.append(float(crafted))
            print(
                f"[eval] Episode {ep + 1:2d}/{n_episodes}: "
                f"reward={total_rew:.2f} steps={steps} crafted={crafted}"
            )
        except Exception as e:
            print(f"[eval] Episode {ep + 1:2d}/{n_episodes}: env error ({e})")
        finally:
            env.close()

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_steps = float(np.mean(lengths)) if lengths else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0

    return {
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        "success_rate": success_rate,
        "sample_frame": sample_frame,
        "captured_frames": captured_frames,
    }


def train(
    total_timesteps: int = 500_000,
    rollout_steps:   int = 4096,
    save_dir:        str = "checkpoints",
    save_every:      int = 50_000,
    resume:          str = None,
    vpt_model:       str = None,
    vpt_weights:     str = None,
    print_rewards:   bool = False,
    tensorboard_dir: str = None,
    tb_flush_every:  int = 10,
    tb_image_every:  int = 1,
    tb_video_every:  int = 3,
    tb_video_frames: int = 64,
    eval_every:      int = 20_000,
    eval_episodes:   int = 2,
    eval_max_steps:  int = 800,
    no_tensorboard:  bool = False,
    env_mode:        str = "crafting",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  total_timesteps={total_timesteps:,}")

    os.makedirs(save_dir, exist_ok=True)

    writer = None
    tb_enabled = not no_tensorboard
    if tb_enabled:
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard is not available. Install with `pip install tensorboard`."
            )
        base_tb_dir = tensorboard_dir or os.path.join(save_dir, "runs")
        run_name = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(base_tb_dir, f"train-{run_name}")
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text(
            "run/config",
            (
                f"total_timesteps={total_timesteps}  rollout_steps={rollout_steps}  "
                f"save_every={save_every}  eval_every={eval_every}  "
                f"eval_episodes={eval_episodes}"
            ),
            global_step=0,
        )
        print(f"[train] TensorBoard enabled: {log_dir}")
    else:
        print("[train] TensorBoard disabled via --no-tensorboard")

    _env_factory = make_crafting_env if env_mode == "crafting" else make_env
    _n_actions   = N_RECIPE_ACTIONS if env_mode == "crafting" else N_ACTIONS
    print(f"[train] Creating environment (mode={env_mode}, n_actions={_n_actions}) …")
    env = _env_factory()
    print("[train] Environment ready.")
    print(f"        Action space   : {env.action_space}")
    print(f"        Obs pov shape  : {env.observation_space['pov'].shape}")
    print(f"        Obs inv shape  : {env.observation_space['inventory'].shape}")

    agent = PPOAgent(
        n_actions=_n_actions,
        n_inventory=len(OBS_ITEMS),
        vpt_model=vpt_model,
        vpt_weights=vpt_weights,
        device=device,
    )

    start_step = 0
    if resume:
        agent.load(resume)
        # Attempt to parse timestep from filename
        try:
            start_step = int(os.path.splitext(os.path.basename(resume))[0]
                             .split("_")[-1])
        except ValueError:
            pass
        print(f"[train] Resumed from {resume}  (step {start_step:,})")

    print("[train] Calling env.reset() — world generation can take ~1 min …")
    obs = env.reset()
    print("[train] env.reset() done.")
    ep_reward            = 0.0
    ep_count             = 0
    ep_rewards           = deque(maxlen=20)
    timestep             = start_step
    next_save            = start_step + save_every
    next_eval            = start_step + eval_every if eval_every > 0 else None
    resets_since_restart = 0
    RESTART_EVERY        = 2   # recreate the Java process every N full resets
    update_idx           = 0
    eval_idx             = 0

    rollout = {k: [] for k in
               ("povs", "invs", "actions", "log_probs", "rewards", "values", "dones")}

    print("[train] Starting collection …")
    t0 = time.time()

    try:
        while timestep < total_timesteps:
            # ── Collect one rollout ──────────────────────────────────────────
            for _ in range(rollout_steps):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)

                # MineRL signals a step-level timeout via info['error']; treat
                # it as a forced episode end so we reset rather than continuing
                # to step a broken env.
                if "error" in info:
                    print(f"[train] Step error from env: {info['error']}")
                    done = True

                rollout["povs"].append(obs["pov"])
                rollout["invs"].append(obs["inventory"])
                rollout["actions"].append(action)
                rollout["log_probs"].append(log_prob)
                rollout["rewards"].append(float(reward))
                rollout["values"].append(value)
                rollout["dones"].append(float(done))

                if print_rewards:
                    print(f"  step {timestep:>7,d} | reward {reward:+.4f}")

                ep_reward += reward
                obs        = next_obs
                timestep  += 1

                if done:
                    ended_episode_reward = ep_reward
                    ep_rewards.append(ep_reward)
                    ep_count  += 1
                    ep_reward  = 0.0
                    resets_since_restart += 1

                    if writer is not None and tb_image_every > 0 and ep_count % tb_image_every == 0:
                        writer.add_scalar("train/episode_reward", ended_episode_reward, timestep)
                        if "pov" in next_obs:
                            writer.add_image(
                                "media/train_episode_frame",
                                next_obs["pov"],
                                timestep,
                                dataformats="CHW",
                            )

                    # Periodically kill and recreate the Java process to
                    # prevent JVM heap accumulation from causing freezes.
                    if resets_since_restart >= RESTART_EVERY:
                        print("[train] Restarting Java process to clear JVM heap …")
                        env.close()
                        env = _env_factory()
                        resets_since_restart = 0

                    try:
                        obs = _reset_with_timeout(env)
                    except TimeoutError:
                        print("[train] Reset timed out — Java process frozen. "
                              "Recreating env and continuing…")
                        try:
                            env.close()
                        except Exception:
                            pass
                        env = _env_factory()
                        resets_since_restart = 0
                        obs = _reset_with_timeout(env)

            # ── PPO update ───────────────────────────────────────────────────
            metrics = agent.update(rollout, obs)

            for v in rollout.values():
                v.clear()

            # ── Logging ──────────────────────────────────────────────────────
            mean_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            fps      = rollout_steps / max(time.time() - t0, 1e-6)
            t0       = time.time()
            print(
                f"step {timestep:>7,d} | ep {ep_count:>4d} | "
                f"mean_rew {mean_rew:6.2f} | "
                f"pg {metrics['pg']:+.4f} | vf {metrics['vf']:.4f} | "
                f"ent {metrics['ent']:.4f} | fps {fps:.0f}"
            )

            if writer is not None:
                writer.add_scalar("train/mean_reward", mean_rew, timestep)
                writer.add_scalar("train/policy_loss", metrics["pg"], timestep)
                writer.add_scalar("train/value_loss", metrics["vf"], timestep)
                writer.add_scalar("train/entropy", metrics["ent"], timestep)
                writer.add_scalar("train/fps", fps, timestep)
                if ep_rewards:
                    writer.add_scalar("train/episode_reward_last20_std", float(np.std(ep_rewards)), timestep)

            update_idx += 1

            if writer is not None and tb_flush_every > 0 and update_idx % tb_flush_every == 0:
                writer.flush()

            if next_eval is not None and timestep >= next_eval:
                eval_idx += 1
                print(f"[train] Running deterministic eval ({eval_episodes} episodes) …")
                eval_out = _evaluate_policy(
                    agent=agent,
                    n_episodes=eval_episodes,
                    max_steps=eval_max_steps,
                    video_frames=tb_video_frames,
                    capture_video=(writer is not None and tb_video_every > 0 and eval_idx % tb_video_every == 0),
                    env_mode=env_mode,
                )
                print(
                    f"[eval] mean_reward={eval_out['mean_reward']:.2f} "
                    f"mean_steps={eval_out['mean_steps']:.1f} "
                    f"success_rate={eval_out['success_rate']:.2f}"
                )

                if writer is not None:
                    writer.add_scalar("eval/mean_reward", eval_out["mean_reward"], timestep)
                    writer.add_scalar("eval/mean_steps", eval_out["mean_steps"], timestep)
                    writer.add_scalar("eval/success_rate", eval_out["success_rate"], timestep)
                    if eval_out["sample_frame"] is not None:
                        writer.add_image(
                            "media/eval_frame",
                            eval_out["sample_frame"],
                            timestep,
                            dataformats="CHW",
                        )
                    if eval_out["captured_frames"]:
                        video = np.stack(eval_out["captured_frames"], axis=0).astype(np.float32) / 255.0
                        video = np.expand_dims(video, axis=0)
                        writer.add_video("media/eval_clip", video, timestep, fps=8)

                next_eval += eval_every

            # ── Checkpointing ────────────────────────────────────────────────
            if timestep >= next_save:
                path = os.path.join(save_dir, f"policy_{timestep}.pth")
                agent.save(path)
                print(f"[train] Checkpoint saved → {path}")
                next_save += save_every

    except (TimeoutError, KeyboardInterrupt) as e:
        print(f"[train] Stopping early: {e}")

    finally:
        final_path = os.path.join(save_dir, f"policy_{timestep}.pth")
        agent.save(final_path)
        print(f"[train] Checkpoint saved → {final_path}")
        if writer is not None:
            writer.flush()
            writer.close()
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint: str,
    n_episodes: int = 5,
    vpt_model:  str = "Video-Pre-Training/foundation-model-1x.model",
    vpt_weights:str = "Video-Pre-Training/foundation-model-1x.weights",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = PPOAgent(
        n_actions=N_ACTIONS,
        n_inventory=len(OBS_ITEMS),
        vpt_model=vpt_model,
        vpt_weights=vpt_weights,
        device=device,
    )
    agent.load(checkpoint)
    agent.policy.eval()
    print(f"[eval] device={device}")
    print(f"[eval] Loaded {checkpoint}")

    # ───────────── Metrics ─────────────
    successes = 0
    times_to_first_log = []
    steps_per_log = []

    ep = 0
    while ep < n_episodes:
        env = make_env()
        try:
            obs       = _reset_with_timeout(env)
            total_rew = 0.0
            done      = False
            steps     = 0

            # metrics per episode
            first_log_step = None
            logs_at_start = float(obs["inventory"][0])
            
            while not done:
                with torch.no_grad():
                    action, _, _ = agent.select_action(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_rew += reward
                steps     += 1

                # detect first log acquisition
                logs = float(obs["inventory"][0])
                if first_log_step is None and logs > 0:
                    first_log_step = steps

                env.render()

            # ───────────── Episode end metrics ─────────────
            logs = float(obs["inventory"][0])
            crafted = float(obs["inventory"][2])
            success = crafted > 0

            if success:
                successes += 1

            if first_log_step is not None:
                times_to_first_log.append(first_log_step)

            # steps per log (avoid divide by zero)
            log_gain = max(logs, 1e-8)
            steps_per_log.append(steps / log_gain)

            print(
                f"  Episode {ep + 1:2d}: "
                f"reward={total_rew:.2f}  steps={steps} "
                f"logs={int(logs)} success={success}"
            )

            ep += 1

        except Exception as e:
            print(f"  Episode {ep + 1:2d}: env error ({e}) — restarting Java process")
        finally:
            env.close()

    # ───────────── Final summary ─────────────
    success_rate = successes / max(ep, 1)

    avg_time_to_log = (
        np.mean(times_to_first_log) if times_to_first_log else float("inf")
    )

    avg_steps_per_log = (
        np.mean(steps_per_log) if steps_per_log else float("inf")
    )

    print("\n===== Evaluation Results =====")
    print(f"Success rate        : {success_rate * 100:.2f}%")
    print(f"Avg time to 1st log : {avg_time_to_log:.1f} steps")
    print(f"Avg steps per log   : {avg_steps_per_log:.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO agent: chop wood and craft a crafting table in Minecraft"
    )
    parser.add_argument(
        "--eval", metavar="CHECKPOINT",
        help="Path to a .pth checkpoint for evaluation (skips training)"
    )
    parser.add_argument(
        "--resume", metavar="CHECKPOINT",
        help="Path to a .pth checkpoint to resume training from"
    )
    parser.add_argument("--timesteps",    type=int, default=500_000)
    parser.add_argument("--rollout-steps",type=int, default=2048)
    parser.add_argument("--save-dir",     type=str, default="checkpoints")
    parser.add_argument("--save-every",   type=int, default=50_000)
    parser.add_argument("--episodes",     type=int, default=5,
                        help="Number of eval episodes (--eval only)")
    parser.add_argument("--vpt-model",   type=str,
                        default="./scripts/foundation-model-1x.model",
                        help="Path to VPT .model file")
    parser.add_argument("--vpt-weights", type=str,
                        default="./scripts/foundation-model-1x.weights",
                        help="Path to VPT .weights file")
    parser.add_argument("--print-rewards", action="store_true",
                        help="Print reward at every step during training")
    parser.add_argument("--tensorboard-dir", type=str, default="tb_logs",
                        help="Directory for TensorBoard event files (default: <save-dir>/runs)")
    parser.add_argument("--tb-flush-every", type=int, default=10,
                        help="Flush TensorBoard writer every N PPO updates")
    parser.add_argument("--tb-image-every", type=int, default=1,
                        help="Log one training wpoge every N completed episodes")
    parser.add_argument("--tb-video-every", type=int, default=3,
                        help="Log one eval video every N eval cycles")
    parser.add_argument("--tb-video-frames", type=int, default=64,
                        help="Maximum frames to keep in each logged eval video")
    parser.add_argument("--eval-every", type=int, default=20_000,
                        help="Run deterministic eval every N training timesteps (0 disables periodic eval)")
    parser.add_argument("--eval-episodes", type=int, default=2,
                        help="Number of episodes per periodic eval during training")
    parser.add_argument("--eval-max-steps", type=int, default=800,
                        help="Maximum steps per periodic eval episode")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging during training")
    parser.add_argument(
        "--env-mode", type=str, default="survival",
        choices=["crafting", "survival"],
        help=(
            "crafting: start each episode with 1 oak_log, train inventory GUI crafting. "
            "survival: standard survival world, agent must chop and craft from scratch."
        ),
    )
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, n_episodes=args.episodes,
                 vpt_model=args.vpt_model, vpt_weights=args.vpt_weights,
                 env_mode=args.env_mode)
    else:
        train(
            total_timesteps=args.timesteps,
            rollout_steps=args.rollout_steps,
            save_dir=args.save_dir,
            save_every=args.save_every,
            resume=args.resume,
            vpt_model=args.vpt_model,
            vpt_weights=args.vpt_weights,
            print_rewards=args.print_rewards,
            tensorboard_dir=args.tensorboard_dir,
            tb_flush_every=args.tb_flush_every,
            tb_image_every=args.tb_image_every,
            tb_video_every=args.tb_video_every,
            tb_video_frames=args.tb_video_frames,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            eval_max_steps=args.eval_max_steps,
            no_tensorboard=args.no_tensorboard,
            env_mode=args.env_mode,
        )
