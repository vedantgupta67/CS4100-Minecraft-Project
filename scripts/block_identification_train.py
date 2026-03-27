import os
import json
import time
import numpy as np
import gym
import minerl

# ── Config ─────────────────────────────────────────────────────────
ENV_NAME        = "MineRLObtainDiamondShovel-v0"
POV_DB_PATH     = "block_observations.npz"   # compressed POV arrays
LABEL_DB_PATH   = "block_labels.json"        # matching block labels
MINE_TIMEOUT    = 10.0   # seconds before giving up on a mine attempt
MIN_MINE_STEPS  = 20     # minimum attack steps before checking pov diff
POV_DIFF_THRESH = 0.15   # fraction of pixels that must change to count as "pov changed"
WANDER_STEPS    = 40     # steps to wander before attempting to mine
STOP_STEPS      = 10     # no-op steps to let the player stop moving
PICKUP_STEPS    = 20     # steps to walk forward and pick up the dropped block
SCAN_SPEED      = 25.0


# ── NPZ + JSON helpers ─────────────────────────────────────────────

def load_db() -> tuple[np.ndarray | None, list[str]]:
    """Load existing POV arrays and labels. Returns (povs, labels)."""
    povs   = None
    labels = []
    if os.path.exists(POV_DB_PATH) and os.path.exists(LABEL_DB_PATH):
        povs   = np.load(POV_DB_PATH)["povs"]
        with open(LABEL_DB_PATH) as f:
            labels = json.load(f)
        print(f"[DB] Loaded {len(labels)} existing observations")
    return povs, labels


def save_db(povs: np.ndarray, labels: list[str]):
    """Overwrite both DB files atomically."""
    np.savez_compressed(POV_DB_PATH, povs=povs)
    with open(LABEL_DB_PATH, "w") as f:
        json.dump(labels, f)


def append_to_db(existing_povs: np.ndarray | None,
                 existing_labels: list[str],
                 new_pov: np.ndarray,
                 new_label: str) -> tuple[np.ndarray, list[str]]:
    """Append one observation and persist immediately."""
    pov_4d = new_pov[np.newaxis]   # (1, H, W, 3)
    if existing_povs is None:
        povs = pov_4d
    else:
        povs = np.concatenate([existing_povs, pov_4d], axis=0)
    labels = existing_labels + [new_label]
    save_db(povs, labels)
    return povs, labels


# ── Inventory helpers ──────────────────────────────────────────────

def snapshot_inventory(obs: dict) -> dict:
    """Return a copy of the current inventory counts."""
    return {k: int(v) for k, v in obs["inventory"].items()}


def inventory_delta(before: dict, after: dict) -> dict:
    """Return items whose count increased."""
    return {
        k: after[k] - before[k]
        for k in before
        if after[k] > before[k]
    }


# ── POV helpers ────────────────────────────────────────────────────

def pov_changed(a: np.ndarray, b: np.ndarray, thresh=POV_DIFF_THRESH) -> bool:
    """True if enough pixels differ between two frames."""
    diff = np.abs(a.astype(int) - b.astype(int))
    changed_frac = (diff.mean(axis=2) > 20).mean()
    return changed_frac > thresh


# ── Action helpers ─────────────────────────────────────────────────

def noop(env):
    return env.action_space.noop()


def random_wander_action(env):
    a = noop(env)
    a["forward"] = int(np.random.random() > 0.3)
    a["back"]    = int(np.random.random() > 0.85)
    a["left"]    = int(np.random.random() > 0.7)
    a["right"]   = int(np.random.random() > 0.7)
    a["jump"]    = int(np.random.random() > 0.85)
    a["camera"]  = np.array([
        np.random.uniform(-SCAN_SPEED, SCAN_SPEED),
        np.random.uniform(-SCAN_SPEED, SCAN_SPEED),
    ])
    return a


def attack_action(env):
    a = noop(env)
    a["attack"] = 1
    return a


def walk_forward_action(env):
    a = noop(env)
    a["forward"] = 1
    return a


# ── Main collector ─────────────────────────────────────────────────

class BlockDataCollector:
    """
    State machine:
      WANDER  → randomly explore for WANDER_STEPS steps
      MINE    → hold attack, record pov before + watch for change / timeout
      PICKUP  → walk forward for PICKUP_STEPS to collect the dropped item
      LABEL   → diff inventory to label the observation, save to CSV
    """

    WANDER = "wander"
    STOP   = "stop"
    MINE   = "mine"
    PICKUP = "pickup"

    def __init__(self, env):
        self.env          = env
        self.povs, self.labels = load_db()
        self._state       = self.WANDER
        self._wander_ctr  = 0
        self._stop_ctr    = 0
        self._pickup_ctr  = 0
        self._pov_before  = None   # frame captured just before mining starts
        self._inv_before  = None   # inventory snapshot before mining
        self._mine_start  = None   # wall-clock time mining began
        self._mine_steps  = 0      # attack steps taken so far

    def reset(self):
        obs = self.env.reset()
        self._state      = self.WANDER
        self._wander_ctr = 0
        self._pickup_ctr = 0
        self._pov_before = None
        self._inv_before = None
        self._mine_start = None
        return obs

    def step(self, obs: dict):
        """
        Given the current observation, decide an action and advance
        the state machine. Returns (action, done_with_episode).
        """
        if self._state == self.WANDER:
            return self._wander_step(obs)
        elif self._state == self.STOP:
            return self._stop_step(obs)
        elif self._state == self.MINE:
            return self._mine_step(obs)
        elif self._state == self.PICKUP:
            return self._pickup_step(obs)

    # ── States ─────────────────────────────────────────────────────

    def _wander_step(self, obs):
        self._wander_ctr += 1
        if self._wander_ctr >= WANDER_STEPS:
            self._wander_ctr = 0
            self._stop_ctr   = 0
            self._state      = self.STOP
            print("[STATE] → STOP")
        return random_wander_action(self.env)

    def _stop_step(self, obs):
        self._stop_ctr += 1
        if self._stop_ctr >= STOP_STEPS:
            # Player has fully stopped — safe to snapshot and mine
            self._pov_before = obs["pov"].copy()
            self._inv_before = snapshot_inventory(obs)
            self._mine_start = time.time()
            self._mine_steps = 0
            self._state      = self.MINE
            print("[STATE] → MINE")
        return noop(self.env)

    def _mine_step(self, obs):
        elapsed   = time.time() - self._mine_start
        timed_out = elapsed >= MINE_TIMEOUT
        self._mine_steps += 1

        # Don't check pov diff until we've swung enough times for a
        # block to actually break — avoids tripping on crack animations
        pov_diff = (
            self._mine_steps >= MIN_MINE_STEPS and
            pov_changed(self._pov_before, obs["pov"])
        )

        if pov_diff or timed_out:
            reason = "pov changed" if pov_diff else "timeout"
            print(f"[MINE] Stopped ({reason}) after {self._mine_steps} steps / {elapsed:.1f}s")
            self._pickup_ctr = 0
            self._state      = self.PICKUP
            print("[STATE] → PICKUP")

        return attack_action(self.env)

    def _pickup_step(self, obs):
        self._pickup_ctr += 1
        if self._pickup_ctr >= PICKUP_STEPS:
            # Label and save
            inv_after = snapshot_inventory(obs)
            delta     = inventory_delta(self._inv_before, inv_after)
            self._label_and_save(delta, self._pov_before)
            # Back to wandering
            self._state = self.WANDER
            print("[STATE] → WANDER")
        return walk_forward_action(self.env)

    # ── Labelling ──────────────────────────────────────────────────

    def _label_and_save(self, delta: dict, pov: np.ndarray):
        if not delta:
            # Block broke but nothing was picked up (e.g. glass, leaves,
            # some ores without silk touch) — still save as "unknown"
            # so we have a record of the visual without a label.
            label = "unknown"
            print("[LABEL] No inventory change — saving as 'unknown'")
        else:
            # Pick the item with the largest count increase as the label
            label = max(delta, key=delta.get)
            print(f"[LABEL] Block identified as '{label}' (delta: {delta})")

        self.povs, self.labels = append_to_db(
            self.povs, self.labels, pov, label
        )
        print(f"[DB] Total observations: {len(self.labels)}")


# ── Run ────────────────────────────────────────────────────────────

def run(episodes=10, max_steps=1000):
    env       = gym.make(ENV_NAME)
    collector = BlockDataCollector(env)

    for ep in range(episodes):
        obs  = collector.reset()
        done = False
        print(f"\n═══ Episode {ep + 1} ═══")

        for _ in range(max_steps):
            action              = collector.step(obs)
            obs, _, done, _     = env.step(action)
            env.render()
            if done:
                break

    env.close()
    print(f"\n[DONE] Collected {len(collector.labels)} total observations")


if __name__ == "__main__":
    run()