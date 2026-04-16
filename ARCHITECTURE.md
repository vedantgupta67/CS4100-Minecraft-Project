# Architecture Overview

This repository has two project-owned pipelines plus a vendored dependency tree.

## 1. Wood/Crafting RL Pipeline

Primary files:

- `scripts/wood_crafting_agent.py`: PPO training and single-policy evaluation entry point.
- `scripts/evaluate_full_run.py`: two-phase evaluation (wood policy then craft policy in same world).
- `scripts/calibrate_gui.py`: calibration utility for crafting GUI click offsets.
- `scripts/shared_runtime.py`: shared runtime constants and JVM patch helper.

Flow:

1. `wood_crafting_agent.py` trains survival or crafting policy.
2. checkpoints are written to `checkpoints/`.
3. `evaluate_full_run.py` loads separate wood and craft checkpoints and runs both phases in one episode.

## 2. Block Identification/KNN Pipeline

Primary files:

- `scripts/block_identification/block_identification_train.py`: canonical data collector.
- `scripts/block_identification/block_identification_train_no_empty.py`: compatibility wrapper for no-empty-label mode.
- `scripts/block_identification/KNN_model.py`: KNN implementation.
- `scripts/block_identification/evaluate_block_identification.py`: basic evaluator.
- `scripts/block_identification/log_training.py`: log-focused data expansion.
- `scripts/block_identification/evaluate_log_db.py`: log-seeking evaluator.
- `scripts/block_identification/run_forever.py`: restart wrapper for long-running log training.

Artifacts (saved in `scripts/block_identification/`):

- `block_observations_2.npz`
- `block_labels_2.json`
- `knn_model.joblib`
- `log_observations.npz`
- `log_labels.json`
- `STOP_REQUESTED` (temporary control flag for wrapper runs)

## 3. Vendored Dependency

- `minerl/` is a vendored MineRL tree used via editable install (`pip install -e minerl/`).
- Project code in `scripts/` should avoid modifying vendored `minerl/` internals unless absolutely necessary.

## Entry Points

- Train RL policy: `python scripts/wood_crafting_agent.py`
- Full pipeline eval: `python scripts/evaluate_full_run.py --wood-policy ... --craft-policy ...`
- Calibrate crafting GUI: `python scripts/calibrate_gui.py`
- Collect block labels: `python scripts/block_identification/block_identification_train.py`
- Run no-empty variant: `python scripts/block_identification/block_identification_train_no_empty.py`
