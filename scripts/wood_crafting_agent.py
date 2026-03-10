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
    * AutoCraftWrapper  — simulates crafting from reward signal, ends episode
                          when crafting table is obtained
    * DiscreteActionWrapper — 13 discrete actions over the Dict action space
    * ObservationWrapper    — CHW image + virtual inventory vector
- Discrete action wrapper (15 actions) over the Dict action space
- Actor-Critic CNN + Inventory encoder (PolicyNet)
- PPO with GAE for training

Usage
-----
  python scripts/wood_crafting_agent.py            # train
  python scripts/wood_crafting_agent.py --eval checkpoints/policy_final.pth
"""

import argparse
import os
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

OBS_ITEMS = ["logs", "planks", "crafting_table"]   # names are only for reference

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Discrete-action wrapper
# ──────────────────────────────────────────────────────────────────────────────

# Each entry is a dict of action-key -> value overrides on top of a no-op base.
# 'camera' values are [pitch_delta, yaw_delta] in degrees.
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
    """Maps integer action indices to MineRL Dict actions."""

    def __init__(self, env):
        super().__init__(env)
        self._base = None
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def _get_base(self):
        if self._base is None:
            self._base = self.env.action_space.no_op()
        return self._base

    def action(self, action_idx):
        act = dict(self._get_base())          # shallow copy
        for key, val in DISCRETE_ACTIONS[int(action_idx)].items():
            if key == "camera":
                act["camera"] = np.array(val, dtype=np.float32)
            else:
                act[key] = val
        return act



class AutoCraftWrapper(gym.Wrapper):
    """
    Wraps MineRLObtainDiamondShovel-v0 to focus on the wood → crafting_table
    sub-task.

    Reads the real log count from obs['inventory'] each step (available because
    MineRLObtainDiamondShovel-v0 uses FlatInventoryObservation(ALL_ITEMS)).
    Replaces the env's sparse milestone rewards with dense per-log rewards and
    ends the episode once the agent has collected enough logs to craft a table
    (1 log → 4 planks → 1 crafting_table).

    Rewards:
      +1.0  per new log collected
      +0.5  per new log (planks bonus)
      +10.0 when virtual crafting_table is first obtained  (episode ends)
    """

    PLANKS_REWARD = 0.5
    TABLE_REWARD  = 10.0

    def __init__(self, env):
        super().__init__(env)
        self._prev_logs      = 0
        self._total_logs     = 0
        self._virtual_planks = 0
        self._virtual_table  = False

    @staticmethod
    def _count_logs(obs) -> int:
        inv = obs.get("inventory", {})
        return sum(int(inv.get(t, 0)) for t in LOG_TYPES)

    def _attach(self, obs):
        obs["_logs"]           = self._total_logs
        obs["_virtual_planks"] = self._virtual_planks
        obs["_virtual_table"]  = int(self._virtual_table)
        return obs

    def reset(self):
        obs = self.env.reset()
        self._prev_logs      = self._count_logs(obs)
        self._total_logs     = 0
        self._virtual_planks = 0
        self._virtual_table  = False
        return self._attach(obs)

    def step(self, action):
        obs, _env_reward, done, info = self.env.step(action)

        # Count newly collected logs from inventory delta
        curr_logs = self._count_logs(obs)
        new_logs  = max(0, curr_logs - self._prev_logs)
        self._prev_logs = curr_logs

        # Build our own reward signal from scratch
        reward = float(new_logs)                          # +1 per log
        if new_logs > 0:
            self._total_logs     += new_logs
            self._virtual_planks += new_logs * 4
            reward               += self.PLANKS_REWARD * new_logs   # +0.5 per log

        # First time planks ≥ 4 → crafting table obtained → episode ends
        if not self._virtual_table and self._virtual_planks >= 4:
            self._virtual_table = True
            reward += self.TABLE_REWARD
            done    = True

        return self._attach(obs), reward, done, info


class ObservationWrapper(gym.ObservationWrapper):
    """
    Converts the MineRL obs dict into:
      pov:       np.uint8  (3, 64, 64)  — CHW image, resized from 640×360
      inventory: np.float32 (3,)        — [logs, virtual_planks, virtual_table]
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

        # Resize 640×360 (or any size) → POV_SIZE × POV_SIZE using torch
        pov_t = torch.as_tensor(pov.copy(), dtype=torch.float32
                                ).permute(2, 0, 1).unsqueeze(0) / 255.0
        pov_t = torch.nn.functional.interpolate(
            pov_t, size=(self.POV_SIZE, self.POV_SIZE),
            mode="bilinear", align_corners=False,
        )
        pov = (pov_t.squeeze(0) * 255).byte().numpy()   # (3, 64, 64) uint8

        inv = np.array([
            float(obs.get("_logs",           0)),
            float(obs.get("_virtual_planks", 0)),
            float(obs.get("_virtual_table",  0)),
        ], dtype=np.float32)
        return {"pov": pov, "inventory": inv}


def make_env():
    print("[make_env] Launching Minecraft Java process (can take 2–5 min on first run) …")
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env.seed(175901257196164)
    print("[make_env] gym.make() done — wrapping env …")
    env = AutoCraftWrapper(env)
    env = DiscreteActionWrapper(env)
    env = ObservationWrapper(env)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Actor-Critic network
# ──────────────────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    """
    Actor-Critic network.

    Inputs
    ------
    pov       : (B, 3, 64, 64) float32 in [0, 1]
    inventory : (B, N_OBS_ITEMS) float32

    Outputs
    -------
    logits : (B, N_ACTIONS)
    value  : (B,)
    """

    def __init__(self, n_actions: int, n_inventory: int):
        super().__init__()

        # Visual encoder: 64×64 → 1024-d feature
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # → (32, 15, 15)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64,  6,  6)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64,  4,  4)
            nn.ReLU(),
            nn.Flatten(),                                  # → 1024
        )
        cnn_out = 64 * 4 * 4  # 1024

        # Inventory encoder
        self.inv_enc = nn.Sequential(
            nn.Linear(n_inventory, 64),
            nn.ReLU(),
        )

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(cnn_out + 64, 512),
            nn.ReLU(),
        )

        # Separate actor / critic heads
        self.actor  = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

        # Orthogonal weight initialisation (standard for PPO)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, pov, inventory):
        vis    = self.cnn(pov)
        inv    = self.inv_enc(inventory)
        shared = self.trunk(torch.cat([vis, inv], dim=-1))
        return self.actor(shared), self.critic(shared).squeeze(-1)

    def get_action_and_value(self, pov, inventory, action=None):
        logits, value = self.forward(pov, inventory)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PPO agent
# ──────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(
        self,
        n_actions:      int,
        n_inventory:    int,
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
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs      = n_epochs
        self.minibatch_size = minibatch_size
        self.device        = device
        self.policy    = PolicyNet(n_actions, n_inventory).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs: dict):
        """Return (action_idx, log_prob, value) given a single obs dict."""
        pov = torch.as_tensor(obs["pov"], dtype=torch.float32, device=self.device
                              ).unsqueeze(0) / 255.0
        inv = torch.as_tensor(obs["inventory"], dtype=torch.float32,
                              device=self.device).unsqueeze(0)
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

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start : start + self.minibatch_size]

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

def train(
    total_timesteps: int = 500_000,
    rollout_steps:   int = 2048,
    save_dir:        str = "checkpoints",
    save_every:      int = 50_000,
    resume:          str = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  total_timesteps={total_timesteps:,}")

    os.makedirs(save_dir, exist_ok=True)

    print("[train] Creating environment …")
    env = make_env()
    print("[train] Environment ready.")
    print(f"        Action space   : {env.action_space}")
    print(f"        Obs pov shape  : {env.observation_space['pov'].shape}")
    print(f"        Obs inv shape  : {env.observation_space['inventory'].shape}")

    agent = PPOAgent(
        n_actions=N_ACTIONS,
        n_inventory=len(OBS_ITEMS),
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
    ep_reward  = 0.0
    ep_count   = 0
    ep_rewards = deque(maxlen=20)
    timestep   = start_step
    next_save  = start_step + save_every

    rollout = {k: [] for k in
               ("povs", "invs", "actions", "log_probs", "rewards", "values", "dones")}

    print("[train] Starting collection …")
    t0 = time.time()

    while timestep < total_timesteps:
        # ── Collect one rollout ──────────────────────────────────────────────
        for _ in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            env.render()

            rollout["povs"].append(obs["pov"])
            rollout["invs"].append(obs["inventory"])
            rollout["actions"].append(action)
            rollout["log_probs"].append(log_prob)
            rollout["rewards"].append(float(reward))
            rollout["values"].append(value)
            rollout["dones"].append(float(done))

            ep_reward += reward
            obs        = next_obs
            timestep  += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_count  += 1
                ep_reward  = 0.0
                obs        = env.reset()

        # ── PPO update ───────────────────────────────────────────────────────
        metrics = agent.update(rollout, obs)

        # Clear rollout buffers
        for v in rollout.values():
            v.clear()

        # ── Logging ──────────────────────────────────────────────────────────
        mean_rew  = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        fps       = rollout_steps / max(time.time() - t0, 1e-6)
        t0        = time.time()
        print(
            f"step {timestep:>7,d} | ep {ep_count:>4d} | "
            f"mean_rew {mean_rew:6.2f} | "
            f"pg {metrics['pg']:+.4f} | vf {metrics['vf']:.4f} | "
            f"ent {metrics['ent']:.4f} | fps {fps:.0f}"
        )

        # ── Checkpointing ────────────────────────────────────────────────────
        if timestep >= next_save:
            path = os.path.join(save_dir, f"policy_{timestep}.pth")
            agent.save(path)
            print(f"[train] Checkpoint saved → {path}")
            next_save += save_every

    # Final save
    final_path = os.path.join(save_dir, "policy_final.pth")
    agent.save(final_path)
    print(f"[train] Training complete. Final model → {final_path}")
    env.close()


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(checkpoint: str, n_episodes: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env    = make_env()

    agent = PPOAgent(n_actions=N_ACTIONS, n_inventory=len(OBS_ITEMS), device=device)
    agent.load(checkpoint)
    agent.policy.eval()
    print(f"[eval] Loaded {checkpoint}")

    for ep in range(n_episodes):
        obs        = env.reset()
        total_rew  = 0.0
        done       = False
        steps      = 0

        while not done:
            with torch.no_grad():
                action, _, _ = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            total_rew += reward
            steps     += 1
            env.render()

        print(f"  Episode {ep + 1:2d}: reward={total_rew:.2f}  steps={steps}")

    env.close()


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
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, n_episodes=args.episodes)
    else:
        train(
            total_timesteps=args.timesteps,
            rollout_steps=args.rollout_steps,
            save_dir=args.save_dir,
            save_every=args.save_every,
            resume=args.resume,
        )
