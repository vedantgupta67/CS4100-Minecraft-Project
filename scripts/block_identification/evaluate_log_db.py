#!/usr/bin/env python3
"""
Log-database evaluation agent.

The agent actively minimises the POV difference between the current view
and the nearest sample in the log observation database.  When that
difference falls below MINE_THRESH it mines; if it gets stuck in a
local minimum it restarts with a random wander.

States
------
SEEK   – scan/move to reduce diff toward the nearest log sample.
         Tracks diff history; deliberately steers toward lower diff.
         Stuck too long  → RESTART
         diff < thresh   → MINE
MINE   – hold attack until the block breaks or timeout.  → PICKUP
PICKUP – walk forward to collect the drop.             → SEEK
RESTART– random wander to escape a local minimum.      → SEEK

Usage
-----
    python evaluate_log_db.py
    python evaluate_log_db.py --episodes 5 --mine-thresh 0.04
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import gym
import minerl

sys.path.insert(0, os.path.dirname(__file__))
from KNN_model import KNN, pov_difference
from block_identification_train_no_empty import (
    ENV_NAME,
    noop,
    attack_action,
    walk_forward_action,
    snapshot_inventory,
    inventory_delta,
    PICKUP_STEPS,
    MINE_TIMEOUT,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT              = os.path.join(os.path.dirname(__file__), "..", "..")
LOG_OBS_PATH      = os.path.join(ROOT, "log_observations.npz")
LOG_LBL_PATH      = os.path.join(ROOT, "log_labels.json")

# ── Config ─────────────────────────────────────────────────────────────────────
MINE_THRESH       = 0.05   # mine when nearest-log diff falls below this
STUCK_STEPS       = 80     # steps with no improvement before triggering restart
RESTART_STEPS     = 120    # random-wander steps during a restart
SCAN_SPEED        = 20.0   # max horizontal camera degrees per step during scan
SEEK_SCAN_SPEED   = 8.0    # fine-grained scan speed while actively seeking
IMPROVE_MIN_DELTA = 0.0002 # minimum diff drop to count as "improving"
LOG_INTERVAL      = 20     # print status every N steps


# ── Load log database and build KNN ───────────────────────────────────────────

def load_log_knn() -> KNN:
    if not (os.path.exists(LOG_OBS_PATH) and os.path.exists(LOG_LBL_PATH)):
        raise FileNotFoundError(
            f"Log DB not found: {LOG_OBS_PATH} / {LOG_LBL_PATH}\n"
            "Run log_training.py first to collect log observations."
        )
    povs = np.load(LOG_OBS_PATH)["povs"]
    with open(LOG_LBL_PATH) as f:
        labels = json.load(f)
    print(f"[KNN] Building from {len(labels)} log samples...")
    knn = KNN()
    knn.fit(povs, labels)
    return knn


# ── Action helpers ─────────────────────────────────────────────────────────────

def _camera_action(env, pitch: float = 0.0, yaw: float = 0.0):
    a = noop(env)
    a["camera"] = np.array([pitch, yaw])
    return a


def random_wander_action(env, scan_speed: float = SCAN_SPEED):
    """Mostly forward movement with random horizontal camera scan."""
    a = noop(env)
    a["forward"] = int(np.random.random() > 0.2)
    a["back"]    = int(np.random.random() > 0.9)
    a["left"]    = int(np.random.random() > 0.75)
    a["right"]   = int(np.random.random() > 0.75)
    a["jump"]    = int(np.random.random() > 0.85)
    a["camera"]  = np.array([0.0, np.random.uniform(-scan_speed, scan_speed)])
    return a


# ── Agent ──────────────────────────────────────────────────────────────────────

class LogSeekAgent:
    """
    Actively seeks wood by minimising POV difference to the nearest log sample.

    Movement strategy in SEEK state
    --------------------------------
    Every step the agent:
      1. Measures current diff.
      2. If diff improved from the last step → continue in the same camera
         direction (exploit the gradient).
      3. If diff worsened or plateaued → try a random camera perturbation
         (escape the local plateau).
      4. After STUCK_STEPS consecutive steps without a meaningful improvement
         → trigger RESTART (escape the local minimum entirely).
    """

    SEEK    = "seek"
    MINE    = "mine"
    PICKUP  = "pickup"
    RESTART = "restart"

    def __init__(self, env: gym.Env, knn: KNN):
        self.env    = env
        self.knn    = knn

        # episode state
        self._state         = self.SEEK
        self._step          = 0
        self._mines         = 0
        self._successful_mines = 0

        # seek tracking
        self._best_diff     = float("inf")
        self._prev_diff     = float("inf")
        self._stuck_ctr     = 0
        self._last_camera_yaw = 0.0   # camera direction we're currently following

        # mine / pickup
        self._mine_start    = None
        self._pov_before    = None
        self._inv_before    = None
        self._pickup_ctr    = 0

        # restart
        self._restart_ctr   = 0

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self, obs: dict):
        self._state         = self.SEEK
        self._step          = 0
        self._best_diff     = float("inf")
        self._prev_diff     = float("inf")
        self._stuck_ctr     = 0
        self._last_camera_yaw = 0.0
        self._mine_start    = None
        self._pov_before    = None
        self._inv_before    = None
        self._pickup_ctr    = 0
        self._restart_ctr   = 0
        return self._compute_diff(obs)

    def step(self, obs: dict) -> dict:
        self._step += 1
        if self._state == self.SEEK:
            return self._seek_step(obs)
        elif self._state == self.MINE:
            return self._mine_step(obs)
        elif self._state == self.PICKUP:
            return self._pickup_step(obs)
        elif self._state == self.RESTART:
            return self._restart_step(obs)

    # ── KNN helpers ────────────────────────────────────────────────────────────

    def _compute_diff(self, obs: dict) -> float:
        _, nn_idx = self.knn.predict(obs["pov"])
        return pov_difference(obs["pov"], self.knn.povs[nn_idx])

    # ── SEEK ───────────────────────────────────────────────────────────────────

    def _seek_step(self, obs: dict) -> dict:
        diff = self._compute_diff(obs)

        improved = (self._prev_diff - diff) > IMPROVE_MIN_DELTA
        if diff < self._best_diff:
            self._best_diff = diff
            self._stuck_ctr = 0
        else:
            self._stuck_ctr += 1

        if self._step % LOG_INTERVAL == 0:
            print(
                f"[step {self._step:>5}] SEEK  diff={diff:.4f}"
                f"  best={self._best_diff:.4f}"
                f"  stuck={self._stuck_ctr}/{STUCK_STEPS}"
                f"  mines={self._mines}"
            )

        # Transition: found a log
        if diff < MINE_THRESH:
            print(f"[step {self._step:>5}] diff={diff:.4f} < {MINE_THRESH} → MINE")
            self._pov_before = obs["pov"].copy()
            self._inv_before = snapshot_inventory(obs)
            self._mine_start = time.time()
            self._mines     += 1
            self._state      = self.MINE
            return attack_action(self.env)

        # Transition: stuck in local minimum → restart
        if self._stuck_ctr >= STUCK_STEPS:
            print(
                f"[step {self._step:>5}] Stuck for {STUCK_STEPS} steps"
                f" (best diff={self._best_diff:.4f}) → RESTART"
            )
            self._stuck_ctr   = 0
            self._best_diff   = float("inf")
            self._prev_diff   = float("inf")
            self._restart_ctr = 0
            self._state       = self.RESTART
            return random_wander_action(self.env)

        # Exploit: diff improved last step → keep scanning same direction
        if improved:
            action = _camera_action(self.env, yaw=self._last_camera_yaw)
        else:
            # Explore: pick a new random camera direction
            self._last_camera_yaw = np.random.uniform(-SEEK_SCAN_SPEED, SEEK_SCAN_SPEED)
            # Also move forward most of the time to cover ground
            action = noop(self.env)
            action["forward"] = int(np.random.random() > 0.3)
            action["jump"]    = int(np.random.random() > 0.85)
            action["camera"]  = np.array([0.0, self._last_camera_yaw])

        self._prev_diff = diff
        return action

    # ── MINE ───────────────────────────────────────────────────────────────────

    def _mine_step(self, obs: dict) -> dict:
        elapsed = time.time() - self._mine_start
        if elapsed >= MINE_TIMEOUT:
            print(f"[step {self._step:>5}] Mine timeout ({elapsed:.1f}s) → PICKUP")
            self._pickup_ctr = 0
            self._state      = self.PICKUP
        return attack_action(self.env)

    # ── PICKUP ─────────────────────────────────────────────────────────────────

    def _pickup_step(self, obs: dict) -> dict:
        self._pickup_ctr += 1
        if self._pickup_ctr >= PICKUP_STEPS:
            inv_after = snapshot_inventory(obs)
            delta     = inventory_delta(self._inv_before, inv_after)
            log_items = {k: v for k, v in delta.items() if "log" in k.lower()}
            if log_items:
                label = max(log_items, key=log_items.get)
                print(f"[step {self._step:>5}] Mined '{label}' successfully!")
                self._successful_mines += 1
            else:
                print(f"[step {self._step:>5}] No log collected (may have missed).")

            # Reset seek state for the next target
            self._prev_diff  = float("inf")
            self._best_diff  = float("inf")
            self._stuck_ctr  = 0
            self._state      = self.SEEK
            print(f"[step {self._step:>5}] → SEEK")
        return walk_forward_action(self.env)

    # ── RESTART ────────────────────────────────────────────────────────────────

    def _restart_step(self, obs: dict) -> dict:
        self._restart_ctr += 1
        if self._restart_ctr >= RESTART_STEPS:
            print(f"[step {self._step:>5}] Restart complete → SEEK")
            self._state = self.SEEK
        return random_wander_action(self.env, scan_speed=SCAN_SPEED)


# ── Run ────────────────────────────────────────────────────────────────────────

def run(episodes: int = 3, max_steps: int = 3000):
    knn = load_log_knn()
    env = gym.make(ENV_NAME)
    env.seed(2147483637)
    agent = LogSeekAgent(env, knn)

    total_mines = 0
    total_successful = 0

    for ep in range(episodes):
        print(f"\n{'═' * 60}")
        print(f" Episode {ep + 1}/{episodes}")
        print(f"{'═' * 60}")

        obs  = env.reset()
        agent.reset(obs)
        done = False

        for _ in range(max_steps):
            action          = agent.step(obs)
            obs, _, done, _ = env.step(action)
            env.render()
            if done:
                break

        total_mines      += agent._mines
        total_successful += agent._successful_mines
        print(
            f"\n[EP {ep+1}] Mine attempts: {agent._mines}"
            f"  Successful: {agent._successful_mines}"
        )

    env.close()

    print(f"\n{'═' * 60}")
    print(f" EVALUATION SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Episodes         : {episodes}")
    print(f"  Total mine attempts  : {total_mines}")
    print(f"  Successful mines     : {total_successful}")
    if total_mines > 0:
        print(f"  Mining success rate  : {total_successful/total_mines:.1%}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate log DB by seeking and mining wood")
    p.add_argument("--episodes",    type=int,   default=3,    help="Number of episodes (default: 3)")
    p.add_argument("--max-steps",   type=int,   default=3000, help="Max steps per episode (default: 3000)")
    p.add_argument("--mine-thresh", type=float, default=MINE_THRESH,
                   help=f"Diff threshold to trigger mining (default: {MINE_THRESH})")
    p.add_argument("--stuck-steps", type=int,   default=STUCK_STEPS,
                   help=f"Steps without improvement before restart (default: {STUCK_STEPS})")
    args = p.parse_args()

    MINE_THRESH  = args.mine_thresh
    STUCK_STEPS  = args.stuck_steps
    run(episodes=args.episodes, max_steps=args.max_steps)
