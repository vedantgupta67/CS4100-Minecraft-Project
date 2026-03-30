"""
env.py
======
MineRL environment wrappers and factory for the wood-crafting PPO agent.

Exports
-------
make_env()            — build the fully-wrapped MineRL environment
_reset_with_timeout() — thread-guarded env.reset() used by wrappers and training
LOG_TYPES, OBS_ITEMS, DISCRETE_ACTIONS, N_ACTIONS, WORLD_SEED
"""

import logging
import threading
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import gym
import minerl  # noqa: F401  registers MineRL envs with gym
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Log types present in MC 1.16.5 inventory
LOG_TYPES = [
    "oak_log", "spruce_log", "birch_log",
    "jungle_log", "acacia_log", "dark_oak_log",
]

OBS_ITEMS = ["logs", "planks", "crafting_table"]

# Each entry is a dict of action-key -> value overrides on top of a no-op base.
# 'camera' values are [pitch_delta, yaw_delta] in degrees.
DISCRETE_ACTIONS = [
    {},                                    # 0:  no-op
    {"forward": 1},                        # 1:  walk forward
    {"back": 1},                           # 2:  walk back
    {"left": 1},                           # 3:  strafe left
    {"right": 1},                          # 4:  strafe right
    {"jump": 1},                           # 5:  jump
    {"sprint": 1, "forward": 1},           # 6:  sprint forward
    {"attack": 1},                         # 7:  attack / mine
    {"attack": 1, "forward": 1},           # 8:  mine while moving forward
    {"camera": [0.0, -15.0]},             # 9:  turn left
    {"camera": [0.0,  15.0]},             # 10: turn right
    {"camera": [-15.0,  0.0]},            # 11: look up
    {"camera": [ 15.0,  0.0]},            # 12: look down
]
N_ACTIONS = len(DISCRETE_ACTIONS)

WORLD_SEED = 175901257196164


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers
# ──────────────────────────────────────────────────────────────────────────────

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
        act = dict(self._get_base())
        for key, val in DISCRETE_ACTIONS[int(action_idx)].items():
            if key == "camera":
                act["camera"] = np.array(val, dtype=np.float32)
            else:
                act[key] = val
        return act


class PitchClampWrapper(gym.Wrapper):
    """
    Tracks the agent's estimated pitch and clamps it to [PITCH_MIN, PITCH_MAX].

    Minecraft's default pitch range: -90 (straight up) to +90 (straight down).
    We restrict to [-60, 60] so the agent always sees the horizon.
    """

    PITCH_MIN = -60.0
    PITCH_MAX =  60.0

    def __init__(self, env):
        super().__init__(env)
        self._pitch = 0.0

    def reset(self):
        self._pitch = 0.0
        return self.env.reset()

    def step(self, action):
        delta = float(action.get("camera", [0.0, 0.0])[0])
        new_pitch = float(np.clip(self._pitch + delta, self.PITCH_MIN, self.PITCH_MAX))
        clamped_delta = new_pitch - self._pitch
        if clamped_delta != delta:
            action = dict(action)
            action["camera"] = np.array([clamped_delta, action["camera"][1]], dtype=np.float32)
        self._pitch = new_pitch
        return self.env.step(action)


class AutoCraftWrapper(gym.Wrapper):
    """
    Wraps MineRLObtainDiamondShovel-v0 to focus on the wood → crafting_table
    sub-task.

    Reads the real log count from obs['inventory'] each step and replaces the
    env's sparse milestone rewards with dense per-log rewards.  Episode ends
    once the agent has collected enough logs to craft a table (1 log → 4 planks
    → 1 crafting_table).

    Rewards:
      +1.0   per new log collected
      +0.5   per new log (planks bonus)
      +10.0  when virtual crafting_table is first obtained  (episode ends)
      -5.0   when the agent dies (episode continues after respawn)
      +0.05  each step the center of view shows wood-colored pixels
      +0.08  when the tree fills a larger fraction of the view than last step
             (approach reward, gated on a minimum increase threshold)
      -0.04  downward look penalty (pitch > 45°)
    """

    PLANKS_REWARD        =  0.5
    TABLE_REWARD         = 10.0
    DEATH_REWARD         = -5.0
    LOOK_REWARD          =  0.05
    APPROACH_REWARD      =  0.08
    APPROACH_THRESH      =  0.02
    DOWNWARD_LOOK_PENALTY = -0.04

    MAX_CONSECUTIVE_RESETS = 5

    def __init__(self, env):
        super().__init__(env)
        self._prev_logs          = 0
        self._total_logs         = 0
        self._virtual_planks     = 0
        self._virtual_table      = False
        self._was_alive          = True
        self._prev_tree_frac     = 0.0
        self._consecutive_resets = 0
        self._pitch              = 0.0

    @staticmethod
    def _count_logs(obs) -> int:
        inv = obs.get("inventory", {})
        return sum(int(inv.get(t, 0)) for t in LOG_TYPES)

    @staticmethod
    def _wood_mask(pov_f) -> np.ndarray:
        """Boolean mask of wood-coloured pixels in a float32 HWC array."""
        r, g, b = pov_f[:, :, 0], pov_f[:, :, 1], pov_f[:, :, 2]
        return (r > g + 45) & (g > b + 5) & (r - b > 55) & (r > 80) & (r < 190)

    @staticmethod
    def _tree_pixel_fraction(pov) -> float:
        """Fraction of pixels that look like wood or leaves (HWC uint8)."""
        pov_f = pov.astype(np.float32)
        r, g, b = pov_f[:, :, 0], pov_f[:, :, 1], pov_f[:, :, 2]
        is_leaf = (g > r + 5) & (g > b + 5) & (g > 55) & (r < 160) & (b < 140)
        return float(np.mean(AutoCraftWrapper._wood_mask(pov_f) | is_leaf))

    @staticmethod
    def _looking_at_wood(pov) -> bool:
        """True when the central 15% of the view is mostly wood-coloured."""
        h, w = pov.shape[:2]
        mh, mw = max(1, h // 7), max(1, w // 7)
        cy, cx = h // 2, w // 2
        center = pov[cy - mh: cy + mh, cx - mw: cx + mw].astype(np.float32)
        return float(np.mean(AutoCraftWrapper._wood_mask(center))) > 0.25

    def _attach(self, obs):
        obs["_logs"]           = self._total_logs
        obs["_virtual_planks"] = self._virtual_planks
        obs["_virtual_table"]  = int(self._virtual_table)
        return obs

    def reset(self):
        obs = self.env.reset()
        self._prev_logs          = self._count_logs(obs)
        self._total_logs         = 0
        self._virtual_planks     = 0
        self._virtual_table      = False
        self._was_alive          = True
        self._prev_tree_frac     = 0.0
        self._consecutive_resets = 0
        self._pitch              = 0.0
        return self._attach(obs)

    def step(self, action):
        obs, _env_reward, done, info = self.env.step(action)

        is_alive = bool(obs.get("life_stats", {}).get("is_alive", True))
        died = self._was_alive and not is_alive
        self._was_alive = is_alive

        curr_logs = self._count_logs(obs)
        new_logs  = max(0, curr_logs - self._prev_logs)
        self._prev_logs = curr_logs

        reward = float(new_logs)
        if new_logs > 0:
            self._total_logs     += new_logs
            self._virtual_planks += new_logs * 4
            reward               += self.PLANKS_REWARD * new_logs

        if died:
            reward += self.DEATH_REWARD

        if not self._virtual_table and self._virtual_planks >= 4:
            self._virtual_table = True
            reward += self.TABLE_REWARD
            done    = True

        self._pitch += float(action.get("camera", [0.0, 0.0])[0])
        if self._pitch > 45.0:
            reward += self.DOWNWARD_LOOK_PENALTY

        pov = obs.get("pov")
        if pov is not None and pov.ndim == 3 and pov.shape[2] >= 3:
            if self._looking_at_wood(pov):
                reward += self.LOOK_REWARD
            tree_frac = self._tree_pixel_fraction(pov)
            if tree_frac > self._prev_tree_frac + self.APPROACH_THRESH:
                reward += self.APPROACH_REWARD
            self._prev_tree_frac = tree_frac

        if done and not self._virtual_table:
            self._consecutive_resets += 1
            if self._consecutive_resets <= self.MAX_CONSECUTIVE_RESETS:
                try:
                    obs = _reset_with_timeout(self.env)
                    self._prev_logs      = self._count_logs(obs)
                    self._prev_tree_frac = 0.0
                    self._was_alive      = True
                    done = False
                except Exception:
                    logger.warning("Internal reset timed out — propagating done=True")
            else:
                logger.warning(
                    "%d consecutive resets — ending episode to allow full reset",
                    self._consecutive_resets,
                )
        else:
            self._consecutive_resets = 0

        return self._attach(obs), reward, done, info


class ObservationWrapper(gym.ObservationWrapper):
    """
    Converts the MineRL obs dict into:
      pov:       np.uint8  (3, 64, 64)  — CHW image, resized from 640×640
      inventory: np.float32 (3,)        — [logs, virtual_planks, virtual_table]
    """

    POV_SIZE = 64

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
        pov = obs["pov"][:, :, :3]
        pov_t = torch.as_tensor(pov.copy(), dtype=torch.float32
                                ).permute(2, 0, 1).unsqueeze(0) / 255.0
        pov_t = torch.nn.functional.interpolate(
            pov_t, size=(self.POV_SIZE, self.POV_SIZE),
            mode="bilinear", align_corners=False,
        )
        pov = (pov_t.squeeze(0) * 255).byte().numpy()
        inv = np.array([
            float(obs.get("_logs",           0)),
            float(obs.get("_virtual_planks", 0)),
            float(obs.get("_virtual_table",  0)),
        ], dtype=np.float32)
        return {"pov": pov, "inventory": inv}


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat each chosen action for `repeat` environment steps.

    Rewards are summed; the last observation is returned.  Episode terminates
    early if `done` is True during a repeat.
    """

    def __init__(self, env, repeat: int = 16):
        super().__init__(env)
        self._repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class FixedSeedWrapper(gym.Wrapper):
    """Re-seeds the environment with a fixed value before every reset."""

    def reset(self):
        self.env.seed(WORLD_SEED)
        return self.env.reset()


# ──────────────────────────────────────────────────────────────────────────────
# Factory and utilities
# ──────────────────────────────────────────────────────────────────────────────

def _reset_with_timeout(env, timeout_sec: int = 180):
    """
    Call env.reset() on a background thread.  Raises TimeoutError if it does
    not return within timeout_sec seconds (Java process frozen).
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


def _patch_jvm_memory():
    """
    Monkey-patch MinecraftInstance to use an 8G JVM heap before gym.make().
    The default 4G leaks and freezes after thousands of steps.
    Guard prevents re-wrapping on repeated make_env() calls.
    """
    try:
        import minerl.env.malmo as _malmo
        if getattr(_malmo.MinecraftInstance, "_jvm_patched", False):
            return
        _orig = _malmo.MinecraftInstance.__init__
        def _patched(self, port=None, existing=False, status_dir=None,
                     seed=None, instance_id=None, max_mem=None):
            _orig(self, port=port, existing=existing, status_dir=status_dir,
                  seed=seed, instance_id=instance_id, max_mem=max_mem or "8G")
        _malmo.MinecraftInstance.__init__ = _patched
        _malmo.MinecraftInstance._jvm_patched = True
        logger.info("JVM heap set to 8G")
    except Exception as e:
        logger.warning("Could not patch JVM memory: %s", e)


def make_env():
    _patch_jvm_memory()
    logger.info("Launching Minecraft Java process (can take 2–5 min on first run) …")
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env.seed(WORLD_SEED)
    logger.info("gym.make() done — wrapping env …")
    env = FixedSeedWrapper(env)
    env = AutoCraftWrapper(env)
    env = PitchClampWrapper(env)
    env = DiscreteActionWrapper(env)
    env = ActionRepeatWrapper(env, repeat=4)
    env = ObservationWrapper(env)
    return env
