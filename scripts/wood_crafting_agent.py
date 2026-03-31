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
import copy
import json
import os
import random
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
from torch.utils.data import DataLoader, TensorDataset

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
    {"camera": [0.0, -5.0]},               # 9:  turn left
    {"camera": [0.0,  5.0]},               # 10: turn right
    {"camera": [-5.0,  0.0]},              # 11: look up
    {"camera": [ 5.0,  0.0]},              # 12: look down
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


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat each chosen action for `repeat` environment steps.

    Rewards are summed across repeated steps; the last observation is returned.
    Episode terminates early if `done` is True during a repeat.

    A repeat of 8 means one network decision = 8 game ticks, so a single
    "attack" choice sustains the swing long enough to noticeably damage a log
    and the agent gets a reward signal much sooner.
    """

    def __init__(self, env, repeat: int = 8):
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


def make_env():
    print("[make_env] Launching Minecraft Java process (can take 2–5 min on first run) …")
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env.seed(175901257196164)
    print("[make_env] gym.make() done — wrapping env …")
    env = AutoCraftWrapper(env)
    env = PitchClampWrapper(env)
    env = DiscreteActionWrapper(env)
    env = ActionRepeatWrapper(env, repeat=8)
    env = ObservationWrapper(env)
    return env


import signal

class ResetTimeoutError(BaseException):
    pass

def _timeout_handler(signum, frame):
    raise ResetTimeoutError("env.reset() timed out!")

def reset_env_with_retries(
    env,
    make_env_fn,
    max_retries: int = 5,
    retry_sleep: float = 5.0,
    reset_timeout: int = 90,
):
    """
    Attempt env.reset() with retries and a hard SIGALRM timeout.

    MineRL can intermittently fail mission reset/handshake (socket hanging); 
    when that happens we enforce a timeout, recreate the environment process, 
    and retry to keep long training runs alive.
    Returns a tuple of (obs, env), where env may be a recreated instance.
    """
    last_exc = None
    current_env = env
    
    # Register the signal handler for SIGALRM
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[train] Starting env.reset() ... (will timeout and restart after {reset_timeout}s if it hangs)")
            # Set the alarm
            signal.alarm(int(reset_timeout))
            
            obs = current_env.reset()
            
            # Disable the alarm on success
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return obs, current_env
            
        except ResetTimeoutError as exc:
            signal.alarm(0)
            last_exc = exc
            print(f"[train] env.reset() timed out (attempt {attempt}/{max_retries})")
        except Exception as exc:
            signal.alarm(0)
            last_exc = exc
            print(f"[train] env.reset() failed (attempt {attempt}/{max_retries}): {exc}")

        # If we reach here, it failed or timed out.
        try:
            # 1. Kill the malmo process aggressively FIRST to unblock any hanging sockets
            if hasattr(current_env, "unwrapped") and hasattr(current_env.unwrapped, "instances"):
                for inst in current_env.unwrapped.instances:
                    if hasattr(inst, "kill"):
                        try:
                            inst.kill()
                        except Exception:
                            pass
            
            # 2. Skip polite close() if we timed out because the socket is 100% stuck
            if not isinstance(last_exc, ResetTimeoutError):
                current_env.close()
        except Exception:
            pass

        if attempt < max_retries:
            print("[train] Recreating environment and retrying reset …")
            time.sleep(max(0.0, retry_sleep))
            current_env = make_env_fn()

    # Restore the original signal handler before raising
    signal.signal(signal.SIGALRM, old_handler)
    raise RuntimeError(
        f"env.reset() failed after {max_retries} attempts. Last error: {last_exc}"
    ) from last_exc


# ──────────────────────────────────────────────────────────────────────────────
# 2b. Imitation warm start (behavior cloning)
# ──────────────────────────────────────────────────────────────────────────────

CAMERA_DEADZONE = 3.0


def _to_float(x) -> float:
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0]) if arr.size > 0 else 0.0


def _to_int(x) -> int:
    return int(round(_to_float(x)))


def preprocess_pov(pov: np.ndarray, size: int = ObservationWrapper.POV_SIZE) -> np.ndarray:
    """Match ObservationWrapper's POV preprocessing for BC dataset samples."""
    pov = pov[:, :, :3]
    pov_t = torch.as_tensor(pov.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    pov_t = torch.nn.functional.interpolate(
        pov_t,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return (pov_t.squeeze(0) * 255).byte().numpy()


def obs_to_inventory_features(obs: dict) -> np.ndarray:
    """Build the inventory vector used by the online policy from raw MineRL obs."""
    logs = 0.0
    inv = obs.get("inventory", {})
    if isinstance(inv, dict):
        logs = float(sum(int(inv.get(t, 0)) for t in LOG_TYPES))
    return np.array([logs, 0.0, 0.0], dtype=np.float32)


def map_demo_action_to_discrete(action: dict) -> int:
    """Project a MineRL Dict action into this script's discrete action set."""
    attack = _to_int(action.get("attack", 0))
    forward = _to_int(action.get("forward", 0))
    back = _to_int(action.get("back", 0))
    left = _to_int(action.get("left", 0))
    right = _to_int(action.get("right", 0))
    jump = _to_int(action.get("jump", 0))
    sprint = _to_int(action.get("sprint", 0))

    cam = np.asarray(action.get("camera", np.array([0.0, 0.0], dtype=np.float32))).reshape(-1)
    pitch = float(cam[0]) if cam.size > 0 else 0.0
    yaw = float(cam[1]) if cam.size > 1 else 0.0

    if attack and forward:
        return 8
    if attack:
        return 7
    if sprint and forward:
        return 6
    if jump:
        return 5
    if forward:
        return 1
    if back:
        return 2
    if left:
        return 3
    if right:
        return 4

    abs_pitch = abs(pitch)
    abs_yaw = abs(yaw)
    if abs_pitch >= CAMERA_DEADZONE or abs_yaw >= CAMERA_DEADZONE:
        if abs_yaw >= abs_pitch:
            return 10 if yaw > 0 else 9
        return 12 if pitch > 0 else 11

    return 0


def _find_recording_dirs(root_dir: str):
    recording_dirs = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        has_video = "recording.mp4" in filenames
        has_univ = "univ.json" in filenames
        has_npz = "rendered.npz" in filenames
        if has_video and (has_univ or has_npz):
            recording_dirs.append(dirpath)
    return recording_dirs


def _extract_action_from_universal(frame_dict: dict) -> dict:
    if isinstance(frame_dict.get("action"), dict):
        return frame_dict["action"]

    action = {}
    for key in ("attack", "forward", "back", "left", "right", "jump", "sprint", "camera"):
        if key in frame_dict:
            action[key] = frame_dict[key]

    if "camera" not in action:
        pitch = frame_dict.get("cameraPitch", 0.0)
        yaw = frame_dict.get("cameraYaw", 0.0)
        action["camera"] = [pitch, yaw]
    return action


def _build_imitation_dataset_from_recordings(
    data_dir: str,
    max_trajectories: int,
    max_samples: int,
    seed: int,
):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "BC fallback requires OpenCV. Install opencv-python or use an environment "
            "with minerl.data available."
        ) from exc

    if not data_dir:
        raise RuntimeError(
            "BC fallback needs --bc-data-dir pointing to recordings containing "
            "recording.mp4 plus either univ.json or rendered.npz."
        )

    if not os.path.isdir(data_dir):
        raise RuntimeError(f"BC data dir does not exist: {data_dir}")

    random.seed(seed)
    recording_dirs = _find_recording_dirs(data_dir)
    if not recording_dirs:
        raise RuntimeError(
            "No recordings found under --bc-data-dir. Expected trajectory folders "
            "with recording.mp4 plus either univ.json or rendered.npz."
        )

    random.shuffle(recording_dirs)
    recording_dirs = recording_dirs[:max_trajectories]

    print(
        f"[bc] Using raw-recording fallback from {data_dir} "
        f"(trajectories={len(recording_dirs)}, max_samples={max_samples:,})"
    )

    povs, invs, actions = [], [], []
    for traj_i, rec_dir in enumerate(recording_dirs, start=1):
        video_path = os.path.join(rec_dir, "recording.mp4")
        univ_path = os.path.join(rec_dir, "univ.json")
        npz_path = os.path.join(rec_dir, "rendered.npz")

        raw_univ = None
        frame_keys = None
        act_npz = None
        if os.path.isfile(univ_path):
            with open(univ_path, "r", encoding="utf-8") as f:
                raw_univ = json.load(f)
            frame_keys = sorted((int(k) for k in raw_univ.keys()))
        elif os.path.isfile(npz_path):
            act_npz = np.load(npz_path)
        else:
            continue

        cap = cv2.VideoCapture(video_path)
        used = 0
        frame_i = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = frame_bgr[:, :, ::-1]

            if raw_univ is not None:
                if frame_i >= len(frame_keys):
                    break
                key = frame_keys[frame_i]
                frame_data = raw_univ.get(str(key), {})
                obs = {
                    "pov": frame_rgb,
                    "inventory": frame_data.get("inventory", {}),
                }
                act = _extract_action_from_universal(frame_data)
            else:
                # rendered.npz format stores action arrays aligned by frame index.
                n_steps = len(act_npz["action$forward"])
                if frame_i >= n_steps:
                    break
                obs = {
                    "pov": frame_rgb,
                    "inventory": {},
                }
                cam = act_npz["action$camera"][frame_i]
                act = {
                    "attack": int(act_npz["action$attack"][frame_i]),
                    "forward": int(act_npz["action$forward"][frame_i]),
                    "back": int(act_npz["action$back"][frame_i]),
                    "left": int(act_npz["action$left"][frame_i]),
                    "right": int(act_npz["action$right"][frame_i]),
                    "jump": int(act_npz["action$jump"][frame_i]),
                    "sprint": int(act_npz["action$sprint"][frame_i]),
                    "camera": [float(cam[0]), float(cam[1])],
                }

            povs.append(preprocess_pov(obs["pov"]))
            invs.append(obs_to_inventory_features(obs))
            actions.append(map_demo_action_to_discrete(act))
            used += 1
            frame_i += 1

            if len(actions) >= max_samples:
                break

        cap.release()
        if act_npz is not None:
            act_npz.close()
        print(f"[bc] Fallback trajectory {traj_i:>3d}: {rec_dir}  steps_used={used}")
        if len(actions) >= max_samples:
            break

    if not actions:
        raise RuntimeError("Fallback loader found recordings but produced 0 BC samples.")

    return (
        np.asarray(povs, dtype=np.uint8),
        np.asarray(invs, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def build_imitation_dataset(
    env_id: str = "MineRLObtainDiamondShovel-v0",
    data_dir: str = None,
    max_trajectories: int = 40,
    max_samples: int = 120_000,
    seed: int = 0,
):
    """Load MineRL demonstrations and convert them to supervised BC tensors."""
    random.seed(seed)

    # Preferred path: official MineRL dataset API.
    try:
        import importlib

        minerl_data = importlib.import_module("minerl.data")
        print(
            f"[bc] Loading MineRL data for {env_id} "
            f"(max_trajectories={max_trajectories}, max_samples={max_samples:,})"
        )
        data = minerl_data.make(env_id, data_dir=data_dir)
        trajectories = list(data.get_trajectory_names())
        if not trajectories:
            raise RuntimeError("No MineRL trajectories found via minerl.data.make(...).")
        random.shuffle(trajectories)

        povs, invs, actions = [], [], []
        for traj_i, traj_name in enumerate(trajectories[:max_trajectories], start=1):
            step_count = 0
            for obs, act, _r, _next_obs, _done in data.load_data(traj_name):
                povs.append(preprocess_pov(obs["pov"]))
                invs.append(obs_to_inventory_features(obs))
                actions.append(map_demo_action_to_discrete(act))
                step_count += 1

                if len(actions) >= max_samples:
                    break

            print(f"[bc] Trajectory {traj_i:>3d}: {traj_name}  steps_used={step_count}")
            if len(actions) >= max_samples:
                break

        if not actions:
            raise RuntimeError("MineRL dataset iterator produced 0 samples.")

        povs = np.asarray(povs, dtype=np.uint8)
        invs = np.asarray(invs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        print(f"[bc] Dataset ready: {len(actions):,} samples")
        return povs, invs, actions
    except Exception as exc:
        print(f"[bc] Official minerl.data loader unavailable ({exc}). Trying fallback loader...")
        return _build_imitation_dataset_from_recordings(
            data_dir=data_dir,
            max_trajectories=max_trajectories,
            max_samples=max_samples,
            seed=seed,
        )


def run_behavior_cloning(
    agent,
    povs: np.ndarray,
    invs: np.ndarray,
    actions: np.ndarray,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-4,
    val_frac: float = 0.05,
):
    """Pretrain actor with supervised behavior cloning from demonstrations."""
    n = len(actions)
    if n < 128:
        raise RuntimeError(f"Not enough BC samples ({n}) to run imitation warm start.")

    n_val = max(1, int(n * val_frac))
    perm = np.random.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = TensorDataset(
        torch.as_tensor(povs[train_idx], dtype=torch.float32) / 255.0,
        torch.as_tensor(invs[train_idx], dtype=torch.float32),
        torch.as_tensor(actions[train_idx], dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.as_tensor(povs[val_idx], dtype=torch.float32) / 255.0,
        torch.as_tensor(invs[val_idx], dtype=torch.float32),
        torch.as_tensor(actions[val_idx], dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(agent.policy.parameters(), lr=lr, eps=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(agent.policy.state_dict())

    print(
        f"[bc] Starting BC for {epochs} epochs "
        f"(train={len(train_idx):,}, val={len(val_idx):,}, batch={batch_size})"
    )
    for epoch in range(1, epochs + 1):
        agent.policy.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for pov_b, inv_b, act_b in train_loader:
            pov_b = pov_b.to(agent.device)
            inv_b = inv_b.to(agent.device)
            act_b = act_b.to(agent.device)

            logits, _ = agent.policy.forward(pov_b, inv_b)
            loss = criterion(logits, act_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item()) * act_b.size(0)
            preds = logits.argmax(dim=1)
            running_correct += int((preds == act_b).sum().item())
            running_total += int(act_b.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        agent.policy.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for pov_b, inv_b, act_b in val_loader:
                pov_b = pov_b.to(agent.device)
                inv_b = inv_b.to(agent.device)
                act_b = act_b.to(agent.device)
                logits, _ = agent.policy.forward(pov_b, inv_b)
                val_loss = criterion(logits, act_b)

                val_loss_sum += float(val_loss.item()) * act_b.size(0)
                val_correct += int((logits.argmax(dim=1) == act_b).sum().item())
                val_total += int(act_b.size(0))

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(
            f"[bc] epoch {epoch:>2d}/{epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(agent.policy.state_dict())

    agent.policy.load_state_dict(best_state)
    print(f"[bc] Warm start complete. Best val_loss={best_val_loss:.4f}")


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
        ent_coef:       float = 0.05,
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
        self.lr            = lr
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

    def reset_optimizer(self, lr: float = None):
        lr = self.lr if lr is None else lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 500_000,
    rollout_steps:   int = 2048,
    save_dir:        str = "checkpoints",
    save_every:      int = 10_000,
    resume:          str = None,
    imitation_warmstart: bool = False,
    bc_only: bool = False,
    bc_data_dir: str = None,
    bc_trajectories: int = 40,
    bc_samples: int = 120_000,
    bc_epochs: int = 5,
    bc_batch_size: int = 256,
    bc_lr: float = 1e-4,
    render_train: bool = False,
    reset_max_retries: int = 5,
    reset_retry_sleep: float = 5.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  total_timesteps={total_timesteps:,}")

    os.makedirs(save_dir, exist_ok=True)

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

    elif imitation_warmstart:
        povs, invs, actions = build_imitation_dataset(
            env_id="MineRLObtainDiamondShovel-v0",
            data_dir=bc_data_dir,
            max_trajectories=bc_trajectories,
            max_samples=bc_samples,
        )
        run_behavior_cloning(
            agent,
            povs,
            invs,
            actions,
            epochs=bc_epochs,
            batch_size=bc_batch_size,
            lr=bc_lr,
        )
        bc_path = os.path.join(save_dir, "policy_bc_init.pth")
        agent.save(bc_path)
        print(f"[train] BC warm-start checkpoint saved → {bc_path}")

        if bc_only:
            print("[train] --bc-only set. Skipping PPO fine-tuning.")
            return

        # Reset PPO optimizer state after supervised pretraining.
        agent.reset_optimizer()

    print("[train] Creating environment …")
    env = make_env()
    print("[train] Environment ready.")
    print(f"        Action space   : {env.action_space}")
    print(f"        Obs pov shape  : {env.observation_space['pov'].shape}")
    print(f"        Obs inv shape  : {env.observation_space['inventory'].shape}")

    print("[train] Calling env.reset() — world generation can take ~1 min …")
    obs, env = reset_env_with_retries(
        env,
        make_env,
        max_retries=reset_max_retries,
        retry_sleep=reset_retry_sleep,
    )
    print("[train] env.reset() done.")
    ep_reward  = 0.0
    ep_count   = 0
    ep_steps   = 0
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
            if render_train:
                env.render()

            rollout["povs"].append(obs["pov"])
            rollout["invs"].append(obs["inventory"])
            rollout["actions"].append(action)
            rollout["log_probs"].append(log_prob)
            rollout["rewards"].append(float(reward))
            rollout["values"].append(value)
            rollout["dones"].append(float(done))

            ep_reward += reward
            ep_steps  += 1
            obs        = next_obs
            timestep  += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_count  += 1
                ep_reward  = 0.0

                if ep_steps <= 2:
                    print("[train] Environment returned done instantly, likely crashed. Recreating...")
                    try:
                        env.close()
                    except:
                        pass
                    time.sleep(3)
                    env = make_env()

                ep_steps = 0

                obs, env = reset_env_with_retries(
                    env,
                    make_env,
                    max_retries=reset_max_retries,
                    retry_sleep=reset_retry_sleep,
                )

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
    parser.add_argument("--save-every",   type=int, default=10_000)
    parser.add_argument("--imitation-warmstart", action="store_true",
                        help="Pretrain policy with behavior cloning before PPO")
    parser.add_argument("--bc-only", action="store_true",
                        help="Run only behavior cloning and exit")
    parser.add_argument("--bc-data-dir", type=str, default=None,
                        help="Optional MineRL dataset directory for BC")
    parser.add_argument("--bc-trajectories", type=int, default=40,
                        help="Max demonstration trajectories used for BC")
    parser.add_argument("--bc-samples", type=int, default=120_000,
                        help="Max demonstration transitions used for BC")
    parser.add_argument("--bc-epochs", type=int, default=5,
                        help="Behavior cloning epochs")
    parser.add_argument("--bc-batch-size", type=int, default=256,
                        help="Behavior cloning batch size")
    parser.add_argument("--bc-lr", type=float, default=1e-4,
                        help="Behavior cloning learning rate")
    parser.add_argument("--render-train", action="store_true",
                        help="Render environment during PPO training")
    parser.add_argument("--reset-max-retries", type=int, default=5,
                        help="How many times to retry env.reset() before failing")
    parser.add_argument("--reset-retry-sleep", type=float, default=5.0,
                        help="Seconds to wait before recreating env after reset failure")
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
            imitation_warmstart=args.imitation_warmstart,
            bc_only=args.bc_only,
            bc_data_dir=args.bc_data_dir,
            bc_trajectories=args.bc_trajectories,
            bc_samples=args.bc_samples,
            bc_epochs=args.bc_epochs,
            bc_batch_size=args.bc_batch_size,
            bc_lr=args.bc_lr,
            render_train=args.render_train,
            reset_max_retries=args.reset_max_retries,
            reset_retry_sleep=args.reset_retry_sleep,
        )
