# CS4100-Minecraft-Project

This project trains and evaluates Minecraft RL policies for:

1. Gathering wood in survival mode
2. Crafting planks and a crafting table using inventory GUI actions
3. Running a full pipeline evaluation (wood policy -> craft policy in same world)

## Installation Requirements for MineRL

### 1. Clone MineRL into this repository

```bash
git clone https://github.com/minerllabs/minerl.git minerl
```

### 2. Python 3.10

```bash
brew install python@3.10
python3.10 -m venv venv
source venv/bin/activate
python --version
```

Expected: `python --version` shows `3.10.x`.

### 3. Java 8

```bash
brew install --cask temurin@8
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
java --version
```

Expected: Java 8 / Temurin 8 output.

### 4. Install Python dependencies

From the project root:

```bash
pip install setuptools==65.5.0
pip install gym==0.23.1
pip install -r requirements.txt
pip install -e minerl/
```

MineRL install can take up to ~30 minutes.

### 5. Verify installation

```bash
python -c "import gym, minerl, torch; print('imports ok')"
pip list | grep -E "minerl|torch|gym"
```

## Windows Specific

Use WSL (Ubuntu 22+ recommended), install Java 8 and Python 3.10 in WSL, then follow the same steps above.

## VPT Model Files

The training/evaluation scripts require VPT model files.

```bash
git clone https://github.com/openai/Video-Pre-Training.git
```

Download `foundation-model-1x.model` and `foundation-model-1x.weights` from the VPT releases and place both into `scripts/`.

Defaults used by scripts:

```text
./scripts/foundation-model-1x.model
./scripts/foundation-model-1x.weights
```

Override with:

```text
--vpt-model PATH_TO_MODEL --vpt-weights PATH_TO_WEIGHTS
```

## Running

### A. Train Wood Gathering Policy (default)

```bash
python ./scripts/wood_crafting_agent.py
```

### B. Train Crafting Policy (GUI crafting mode)

```bash
python ./scripts/wood_crafting_agent.py --env-mode crafting
```

### C. Resume training from checkpoint

```bash
python ./scripts/wood_crafting_agent.py --resume checkpoints/policy_162312.pth
```

### D. Evaluate one checkpoint

```bash
python ./scripts/wood_crafting_agent.py --eval checkpoints/policy_162312.pth --episodes 5
```

### E. Full pipeline evaluation (wood -> craft)

```bash
python ./scripts/evaluate_full_run.py \
      --wood-policy checkpoints/policy_162312.pth \
      --craft-policy checkpoints/policy_31560.pth \
      --episodes 3
```

Optional: add `--render` to visualize the world.

## GUI Calibration (needed for crafting mode)

Crafting clicks use calibrated camera offsets stored in `SLOTS`.

Phase 1:

```bash
python ./scripts/calibrate_gui.py
```

Phase 2:

```bash
python ./scripts/calibrate_gui.py --phase 2 --btn-pitch P --btn-yaw Y
```

Phase 3:

```bash
python ./scripts/calibrate_gui.py --phase 3 \
      --btn-pitch P1 --btn-yaw Y1 \
      --planks-pitch P2 --planks-yaw Y2 \
      --output-pitch P3 --output-yaw Y3 \
      --inv-pitch P4 --inv-yaw Y4
```

After calibration, update slot values in `scripts/shared_runtime.py`.

## Checkpoints and Logging

- Default checkpoint directory: `checkpoints/`
- Default save frequency: every `50,000` steps
- Default rollout steps: `2048`
- TensorBoard is ON by default (disable with `--no-tensorboard`)
- TensorBoard logs are written under `<save-dir>/runs/`

Start TensorBoard:

```bash
tensorboard --logdir checkpoints/runs
```

## Common Useful Flags

Training (`wood_crafting_agent.py`):

- `--timesteps` (default `500000`)
- `--rollout-steps` (default `2048`)
- `--save-dir` / `--save-every`
- `--eval-every`, `--eval-episodes`, `--eval-max-steps`
- `--print-rewards`
- `--env-mode {survival,crafting}`

Full pipeline eval (`evaluate_full_run.py`):

- `--episodes` (default `3`)
- `--max-wood-steps` (default `3000`)
- `--max-craft-steps` (default `500`)
- `--render`

Block identification data collection (`block_identification_train.py`):

- `--episodes` (default `1`)
- `--max-steps` (default `50000`)
- `--skip-empty` (skip `far_away` labels)

## Runtime Notes and Troubleshooting

- Startup can take 2-5 minutes when launching Minecraft.
- If you see repeated reset messages, the script is attempting JVM/environment recovery.
- If training is interrupted, resume with `--resume` using the latest checkpoint.
- If VPT files are missing, pass explicit paths using `--vpt-model` and `--vpt-weights`.

## Block Identification Scripts

Location: `scripts/block_identification/`

Primary scripts:

- `block_identification_train.py`: canonical collector for KNN block labels.
- `block_identification_train_no_empty.py`: compatibility wrapper that runs canonical collector with empty-label saving disabled.
- `evaluate_block_identification.py`: basic KNN-driven online evaluation.
- `log_training.py`: log-focused dataset expansion and mining loop.
- `evaluate_log_db.py`: log-seeking evaluator using nearest-neighbor diff minimization.

Data artifacts are saved in `scripts/block_identification/` with deterministic paths:

- `block_observations_2.npz`
- `block_labels_2.json`
- `knn_model.joblib`
- `log_observations.npz`
- `log_labels.json`

## Current Reward System (wood and crafting wrappers)

- Wood policy shaping:
      - `+1000.0` per new log
      - `-10.0` per new dirt mined
      - `-5.0` on death
      - attack decay reward starts at `+0.01`, decays to `-0.05`
- Crafting wrapper shaping:
      - `+500.0` when planks are first obtained
      - `+10000.0` when crafting table is obtained (episode ends)

## PPO — How the Agent Learns

**Proximal Policy Optimization (PPO)** is the RL algorithm driving `wood_crafting_agent.py`. Here is how each piece maps to the theory:

### The Core Problem PPO Solves

Standard policy gradient methods are unstable — a large gradient step can destroy a policy. PPO constrains how much the policy changes per update using a **clipped objective**:

```
L = min(ratio * A,  clip(ratio, 1-ε, 1+ε) * A)
```

Where `ratio = π_new(a|s) / π_old(a|s)` and `ε = 0.2`. If the new policy deviates too far from the old one, the gradient is clipped — preventing catastrophically large updates.

### The Training Loop

**Phase 1 — Rollout collection**: The agent plays Minecraft for `rollout_steps` (default 4096) steps, recording `(obs, action, log_prob, reward, value, done)` tuples without updating weights.

**Phase 2 — PPO update**: After the rollout, the collected data is used for `n_epochs=4` passes, processing it in minibatches of 64.

### GAE — Advantage Estimation

Before updating, the agent computes **Generalized Advantage Estimation**:

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

- `gamma=0.99` — discounts future rewards
- `gae_lambda=0.95` — trades off bias vs variance in advantage estimates

The advantage `A_t` answers: *"was this action better or worse than expected?"* Positive advantages increase action probability; negative ones decrease it.

### The Full Loss

Three terms combined:

```
loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss
```

| Term | Purpose |
|------|---------|
| `pg_loss` | Clipped policy gradient — learn good actions |
| `vf_loss` (×0.5) | Value function MSE — improve reward prediction |
| `-ent_loss` (×0.01) | Entropy bonus — encourages exploration, prevents collapse to one action |

### The Policy Network

The policy is built on OpenAI's **VPT (Video Pre-Training)** model — a CNN pretrained on hours of human Minecraft gameplay. The CNN is **frozen** during training, so PPO only fine-tunes the head layers. This is a form of **transfer learning**: the VPT CNN already understands what trees, wood, and the Minecraft world look like, so PPO can focus on learning the decision-making policy rather than visual perception from scratch.

### Why PPO Fits This Task

Minecraft is a **sparse reward** problem — the agent might take thousands of steps before collecting a single log. The dense reward shaping (visual tree signals, approach rewards) makes the reward signal richer, but PPO still needs to handle:
- Long episodes (action repeat × rollout steps)
- A high-dimensional image observation (64×64 RGB)
- A discrete 13-action space

PPO's stability is critical here — training runs for 500k+ steps over hours, and an unstable algorithm would diverge long before converging.

### Additional Resources
https://www.geeksforgeeks.org/machine-learning/a-brief-introduction-to-proximal-policy-optimization/

https://spinningup.openai.com/en/latest/algorithms/ppo.html

https://en.wikipedia.org/wiki/Proximal_policy_optimization

https://openai.com/index/vpt/
