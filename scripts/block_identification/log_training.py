import os
import time
import json
from pathlib import Path
import numpy as np
import gym
import minerl
from KNN_model import KNN, pov_difference
from block_identification_train import (
    ENV_NAME,
    random_wander_action,
    attack_action,
    walk_forward_action,
    noop,
    snapshot_inventory,
    inventory_delta,
)

# ── Config ─────────────────────────────────────────────────────────
PREDICT_EVERY   = 10    # run KNN every N wander steps
MINE_DURATION   = 10.0  # seconds to mine once a log is spotted
PICKUP_STEPS    = 50    # steps to walk forward after mining
LOG_DIFF_THRESH = 0.05  # mine if nearest log POV diff is below this

BASE_DIR = Path(__file__).resolve().parent
LOG_POV_DB_PATH = BASE_DIR / "log_observations.npz"
LOG_LABEL_DB_PATH = BASE_DIR / "log_labels.json"
STOP_FLAG = BASE_DIR / "STOP_REQUESTED"


# ── Log dataset helpers ────────────────────────────────────────────

def load_log_db() -> tuple[np.ndarray | None, list[str]]:
    if LOG_POV_DB_PATH.exists() and LOG_LABEL_DB_PATH.exists():
        povs = np.load(LOG_POV_DB_PATH)["povs"]
        with LOG_LABEL_DB_PATH.open() as f:
            labels = json.load(f)
        print(f"[LOG-DB] Loaded {len(labels)} existing log observations")
        return povs, labels
    return None, []


def save_log_db(povs: np.ndarray, labels: list[str]):
    np.savez_compressed(LOG_POV_DB_PATH, povs=povs)
    with LOG_LABEL_DB_PATH.open("w") as f:
        json.dump(labels, f)


def append_log_observation(existing_povs, existing_labels, pov, label):
    pov_4d = pov[np.newaxis]
    povs   = pov_4d if existing_povs is None else np.concatenate([existing_povs, pov_4d], axis=0)
    labels = existing_labels + [label]
    save_log_db(povs, labels)
    print(f"[LOG-DB] Saved '{label}' — total: {len(labels)}")
    return povs, labels


# ── Build log-only KNN ─────────────────────────────────────────────

def load_all_log_povs() -> tuple[np.ndarray, list[str]]:
    """
    Merge the dedicated log DB with any log-labelled rows from the main DB.
    """
    all_povs, all_labels = load_log_db()

    if all_povs is None or len(all_labels) == 0:
        raise RuntimeError("No log observations found — collect some log data first.")

    print(f"[LOG-DB] Total log observations available: {len(all_labels)}")
    return all_povs, all_labels


def build_log_knn() -> KNN:
    log_povs, log_labels = load_all_log_povs()
    knn = KNN()
    knn.fit(log_povs, log_labels)
    return knn


# ── Helpers ────────────────────────────────────────────────────────

def has_log(obs: dict) -> bool:
    return any("log" in k.lower() and v > 0 for k, v in obs["inventory"].items())


# ── Main loop ──────────────────────────────────────────────────────

def run(episodes=12, max_steps=1500):
    knn = build_log_knn()
    log_db_povs, log_db_labels = load_log_db()

    env = gym.make(ENV_NAME)
    env.seed(2147483637)

    for ep in range(episodes):
        if STOP_FLAG.exists():
            print("[RUN] Stop flag detected — exiting after current run.")
            break
        obs  = env.reset()
        done = False
        step = 0

        state      = "wander"
        mine_start = None
        pickup_ctr = 0
        pov_before = None
        inv_before = None

        print(f"\n═══ Episode {ep + 1} ═══")

        while not done and step < max_steps:
            #env.render()

            if has_log(obs):
                print(f"[step {step:>5}] Log in inventory — task complete!")
                if inv_before is not None:
                    delta     = inventory_delta(inv_before, snapshot_inventory(obs))
                    log_items = {k: v for k, v in delta.items() if "log" in k.lower()}
                    if log_items:
                        mined_label = max(log_items, key=log_items.get)
                        print(f"[LOG-DB] Mined '{mined_label}' — saving observation")
                        log_db_povs, log_db_labels = append_log_observation(
                            log_db_povs, log_db_labels, pov_before, mined_label
                        )
                break

            if state == "wander":
                if step % PREDICT_EVERY == 0:
                    pred_label, nn_idx = knn.predict(obs["pov"])
                    diff = pov_difference(obs["pov"], knn.povs[nn_idx])
                    print(f"[step {step:>5}] Nearest log: {pred_label:<20}  diff={diff:.4f}")

                    if diff < LOG_DIFF_THRESH:
                        print(f"[step {step:>5}] Close match (diff={diff:.4f} < {LOG_DIFF_THRESH}) → MINE")
                        pov_before = obs["pov"].copy()
                        inv_before = snapshot_inventory(obs)
                        state      = "mine"
                        mine_start = time.time()

                action = random_wander_action(env)

            elif state == "mine":
                elapsed = time.time() - mine_start
                if elapsed >= MINE_DURATION:
                    print(f"[MINE] Done after {elapsed:.1f}s → PICKUP")
                    state      = "pickup"
                    pickup_ctr = 0
                    action     = noop(env)
                else:
                    action = attack_action(env)

            elif state == "pickup":
                pickup_ctr += 1
                if pickup_ctr >= PICKUP_STEPS:
                    delta     = inventory_delta(inv_before, snapshot_inventory(obs))
                    log_items = {k: v for k, v in delta.items() if "log" in k.lower()}
                    if log_items:
                        mined_label = max(log_items, key=log_items.get)
                        print(f"[LOG-DB] Mined '{mined_label}' — saving observation")
                        log_db_povs, log_db_labels = append_log_observation(
                            log_db_povs, log_db_labels, pov_before, mined_label
                        )
                    else:
                        print("[PICKUP] No log gained — discarding observation")

                    state  = "wander"
                    action = noop(env)
                    print("[PICKUP] Done → WANDER")
                else:
                    action = walk_forward_action(env)

            obs, _, done, _ = env.step(action)
            step += 1

        print(f"[EVAL] Episode {ep + 1} finished after {step} steps")

    env.close()
    print(f"\n[DONE] Log DB now has {len(log_db_labels)} observations")


if __name__ == "__main__":
    run()
