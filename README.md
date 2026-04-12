# CS4100-Minecraft-Project


# Installation requirements for MineRL


need to clone minerl into minerl folder:

git clone https://github.com/minerllabs/minerl.git minerl

python 3.10:

run:

brew install python@3.10

python3.10 -m venv venv

source venv/bin/activate

confirm with: python --version


java 8

run:

brew install --cask temurin@8

export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home

confirm with: java --version


Then from home directory run:
pip install setuptools==65.5.0
pip install gym==0.23.1
pip install -e minerl/

confirm installation with: pip list


## Windows Specific
Install WSL Ubuntu version >22 and Temurin Java 8 onto it

Should be able to run code after that!


## Installing MineRL
MineRL can take up to 30 minutes when installing, stay patient.

## VPT Model
To run the wood_crafting_agent.py file:

git clone https://github.com/openai/Video-Pre-Training.git

From https://github.com/openai/Video-Pre-Training

Download 1x Model and 1x Width Weights from their README.md

Place those files into /scripts

By default wood_crafting_agent.py uses those files, if you want to specify a different path use the flags `--vpt-model {path}` and `--vpt-weights {path}`

You can use different the 2x or 3x model and width weights, however, those are much larger and slower, but can produce better results than the 1x.

## Running the Agent

To run the agent: `python ./scripts/wood_crafting_agent.py`

The start up on the agent can take 2-5 minutes and by default, it will not render the environment. You can tell that the agent is running in the terminal if you see `[train] Starting collection …`. After some time, the stats will begin printing for each rollout of steps.

There is a known issue that the environment will freeze after some time, at that point, the program will try to restart the environment and continue the training. If you see multiple `[env] 6 consecutive resets — ending episode to allow full reset`, the program is attempting a restart.

If a training session is interrupted, the training data will be automatically saved to the checkpoints folder.

## Checkpoints

By default, wood_crafting_agent.py saves a .pth file to the /checkpoints folder every 50,000 steps

To change the save directory for checkpoints: `--save-dir {directory name}` (default: checkpoints)

You can change this using the `--save-every {# of steps}` flag

To run a checkpoint, use the `--resume {path}` flag

To evaluate a checkpoint, use the `--eval {path}` flag

## Other Useful Arguments

To change the total number of timesteps: `--timesteps {# of timesteps}`(default: 500,000)

To change the rollout of stats: `--rollout-steps {# of rollout-steps}` (default: 2048)

To number of episodes: `--episodes {# of episodes}` (default: 5)

To print the rewards gained or lost at every step: `--print-rewards` (default: false)

## Current Reward System
      +0.5   per new log
      +10.0  when virtual crafting_table is first obtained  (episode ends)
      -5.0   when the agent dies (episode continues after respawn)
      +0.05  each step the center of view shows wood-colored pixels
      +0.08  when the tree fills a larger fraction of the view than last step
             (approach reward, gated on a minimum increase threshold)

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
