import time
import joblib
import gym
import minerl
from utils import flatten_pov
from KNN_model import KNN
from block_identification_train import (
    ENV_NAME,
    KNN_MODEL_PATH,
    random_wander_action,
    attack_action,
    walk_forward_action,
    noop,
)

# ── Config ─────────────────────────────────────────────────────────
PREDICT_EVERY = 10      # run KNN every N wander steps
MINE_DURATION = 10.0   # seconds to mine once a log is spotted
PICKUP_STEPS  = 20     # steps to walk forward after mining


# ── Helpers ────────────────────────────────────────────────────────

def has_log(obs: dict) -> bool:
    """Return True if any log variant is in the player's inventory."""
    return any("log" in k.lower() and v > 0 for k, v in obs["inventory"].items())


# ── Main loop ──────────────────────────────────────────────────────

def run(episodes=1, max_steps=50000):
    knn = joblib.load(KNN_MODEL_PATH)
    print(f"[KNN] Loaded model from '{KNN_MODEL_PATH}'")

    env = gym.make(ENV_NAME)
    env.seed(2147483637)

    for ep in range(episodes):
        obs  = env.reset()
        done = False
        step = 0

        state       = "wander"
        mine_start  = None
        pickup_ctr  = 0

        print(f"\n═══ Episode {ep + 1} ═══")

        while not done and step < max_steps:
            env.render()

            if has_log(obs):
                print(f"[step {step:>5}] Log in inventory — task complete!")
                break

            if state == "wander":
                if step % PREDICT_EVERY == 0:
                    pred_label, _ = knn.predict(obs["pov"])
                    print(f"[step {step:>5}] Looking at: {pred_label}")

                    if "log" in pred_label.lower():
                        print(f"[step {step:>5}] Log detected! → MINE for {MINE_DURATION}s")
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
                    print(f"[PICKUP] Done → WANDER")
                    state  = "wander"
                    action = noop(env)
                else:
                    action = walk_forward_action(env)

            obs, _, done, _ = env.step(action)
            step += 1

        print(f"[EVAL] Episode {ep + 1} finished after {step} steps")

    env.close()
    print("\n[EVAL] Done")


if __name__ == "__main__":
    run()
