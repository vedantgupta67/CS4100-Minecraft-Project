import os
import json
import joblib
import numpy as np
from utils import flatten_pov

# ── Config ─────────────────────────────────────────────────────────
POV_DB_PATH    = "block_observations_2.npz"
LABEL_DB_PATH  = "block_labels_2.json"
KNN_MODEL_PATH = "knn_model.joblib"


# ── Data loading ───────────────────────────────────────────────────

def load_db() -> tuple[np.ndarray, list[str]]:
    if not (os.path.exists(POV_DB_PATH) and os.path.exists(LABEL_DB_PATH)):
        raise FileNotFoundError(f"DB files not found: {POV_DB_PATH}, {LABEL_DB_PATH}")
    povs = np.load(POV_DB_PATH)["povs"]
    with open(LABEL_DB_PATH) as f:
        labels = json.load(f)
    print(f"[DB] Loaded {len(labels)} observations")
    return povs, labels


# ── KNN (k=1, built from scratch) ─────────────────────────────────

class KNN:
    """k=1 nearest-neighbour classifier backed by raw numpy arrays."""

    def __init__(self):
        self.X: np.ndarray | None = None   # (n, d) flattened + normalised POVs
        self.y: list[str]         = []     # corresponding labels
        self.povs: np.ndarray | None = None  # original (n, H, W, 3) frames

    def fit(self, povs: np.ndarray, labels: list[str]):
        """Flatten and store all training POVs."""
        print(f"[KNN] Fitting on {len(labels)} observations...")
        self.X    = np.stack([flatten_pov(povs[i]) for i in range(len(povs))])
        self.y    = list(labels)
        self.povs = povs
        print("[KNN] Ready")

    def _nearest(self, x: np.ndarray) -> int:
        """Return the index of the training sample closest to x (euclidean)."""
        dists = np.linalg.norm(self.X - x, axis=1)  # (n,)
        return int(np.argmin(dists))

    def predict(self, pov: np.ndarray) -> tuple[str, int]:
        """
        Return (predicted_label, nearest_index) for a single raw POV frame.
        """
        x   = flatten_pov(pov)
        idx = self._nearest(x)
        return self.y[idx], idx

    def predict_batch(self, povs: np.ndarray) -> list[tuple[str, int]]:
        return [self.predict(povs[i]) for i in range(len(povs))]


# ── POV difference (accuracy metric) ──────────────────────────────

def pov_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference (0–1 scale) between two raw POV frames."""
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    return float(np.mean(np.abs(a - b)))


# ── Evaluation ─────────────────────────────────────────────────────

def evaluate_loo(povs: np.ndarray, labels: list[str]):
    """
    Leave-one-out evaluation using the from-scratch KNN.

    For each sample i, train on all others (k=1) and measure:
      * label accuracy  — did the nearest neighbour share the same label?
      * POV difference  — mean absolute pixel difference to the retrieved frame
    """
    n = len(labels)
    print(f"\n[EVAL] Leave-one-out over {n} samples...")

    X = np.stack([flatten_pov(povs[i]) for i in range(n)])

    label_correct = 0
    pov_diffs     = []

    for i in range(n):
        # LOO: exclude index i
        mask      = np.ones(n, dtype=bool)
        mask[i]   = False
        X_train   = X[mask]
        y_train   = [labels[j] for j in range(n) if j != i]
        povs_train = povs[mask]

        # Find nearest neighbour in reduced set
        dists  = np.linalg.norm(X_train - X[i], axis=1)
        nn_idx = int(np.argmin(dists))

        pred_label = y_train[nn_idx]
        nn_pov     = povs_train[nn_idx]
        diff       = pov_difference(povs[i], nn_pov)

        pov_diffs.append(diff)
        if pred_label == labels[i]:
            label_correct += 1

        if (i + 1) % max(1, n // 10) == 0:
            print(
                f"  [{i+1:>{len(str(n))}}/{n}]"
                f"  true={labels[i]:<20}"
                f"  pred={pred_label:<20}"
                f"  pov_diff={diff:.4f}"
            )

    label_accuracy = label_correct / n
    mean_pov_diff  = float(np.mean(pov_diffs))
    print(f"\n[EVAL] Label accuracy : {label_accuracy:.2%}  ({label_correct}/{n} correct)")
    print(f"[EVAL] Mean POV diff  : {mean_pov_diff:.4f}  (0 = identical, 1 = fully different)")
    return label_accuracy, mean_pov_diff


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    povs, labels = load_db()

    # Leave-one-out accuracy report
    evaluate_loo(povs, labels)

    # Train final model on all data and save
    knn = KNN()
    knn.fit(povs, labels)
    joblib.dump(knn, KNN_MODEL_PATH)
    print(f"\n[SAVE] Model saved to '{KNN_MODEL_PATH}'")
