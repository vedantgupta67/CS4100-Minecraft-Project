"""
wood_crafting_agent.py
======================
PPO agent that learns to:
  1. Chop wood from trees (collect logs)
  2. Craft planks from logs
  3. Craft a crafting table from planks

Architecture
------------
- MineRLObtainDiamondShovel-v0 wrapped with AutoCraftWrapper, PitchClampWrapper,
  DiscreteActionWrapper, ActionRepeatWrapper, ObservationWrapper  (see env.py)
- Actor-Critic backed by VPT's pretrained IMPALA CNN  (see vpt_policy.py)
- PPO with GAE  (see agent.py)

Usage
-----
  python scripts/wood_crafting_agent.py            # train
  python scripts/wood_crafting_agent.py --eval checkpoints/policy_final.pth
"""

import logging
import os
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from env import N_ACTIONS, OBS_ITEMS, make_env, _reset_with_timeout
from agent import PPOAgent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Training loop
# ──────────────────────────────────────────────────────────────────────────────


def train(
    total_timesteps: int = 500_000,
    rollout_steps: int = 4096,
    save_dir: str = "checkpoints",
    save_every: int = 50_000,
    resume: str = None,
    vpt_model: str = None,
    vpt_weights: str = None,
    print_rewards: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s  total_timesteps=%s", device, f"{total_timesteps:,}")

    os.makedirs(save_dir, exist_ok=True)

    env = make_env()
    agent = PPOAgent(
        n_actions=N_ACTIONS,
        n_inventory=len(OBS_ITEMS),
        vpt_model=vpt_model,
        vpt_weights=vpt_weights,
        device=device,
    )

    start_step = 0
    if resume:
        agent.load(resume)
        try:
            start_step = int(
                os.path.splitext(os.path.basename(resume))[0].split("_")[-1]
            )
        except ValueError:
            pass
        logger.info("Resumed from %s  (step %s)", resume, f"{start_step:,}")

    logger.info("Calling env.reset() — world generation can take ~1 min …")
    obs = env.reset()
    logger.info("env.reset() done.")

    ep_reward = 0.0
    ep_count = 0
    ep_rewards = deque(maxlen=20)
    timestep = start_step
    next_save = start_step + save_every
    resets_since_restart = 0
    RESTART_EVERY = 5

    rollout = {
        k: []
        for k in ("povs", "invs", "actions", "log_probs", "rewards", "values", "dones")
    }

    logger.info("Starting collection …")
    t0 = time.time()

    try:
        while timestep < total_timesteps:
            # ── Collect one rollout ──────────────────────────────────────────
            for _ in range(rollout_steps):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, _ = env.step(action)

                rollout["povs"].append(obs["pov"])
                rollout["invs"].append(obs["inventory"])
                rollout["actions"].append(action)
                rollout["log_probs"].append(log_prob)
                rollout["rewards"].append(float(reward))
                rollout["values"].append(value)
                rollout["dones"].append(float(done))

                if print_rewards:
                    logger.debug("step %7s | reward %+.4f", f"{timestep:,}", reward)

                ep_reward += reward
                obs = next_obs
                timestep += 1

                if done:
                    ep_rewards.append(ep_reward)
                    ep_count += 1
                    ep_reward = 0.0
                    resets_since_restart += 1

                    if resets_since_restart >= RESTART_EVERY:
                        logger.info("Restarting Java process to clear JVM heap …")
                        env.close()
                        env = make_env()
                        resets_since_restart = 0

                    obs = _reset_with_timeout(env)

            # ── PPO update ───────────────────────────────────────────────────
            metrics = agent.update(rollout, obs)
            for v in rollout.values():
                v.clear()

            # ── Logging ──────────────────────────────────────────────────────
            mean_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            fps = rollout_steps / max(time.time() - t0, 1e-6)
            t0 = time.time()
            logger.info(
                "step %s | ep %4d | mean_rew %6.2f | "
                "pg %+.4f | vf %.4f | ent %.4f | fps %.0f",
                f"{timestep:>7,}",
                ep_count,
                mean_rew,
                metrics["pg"],
                metrics["vf"],
                metrics["ent"],
                fps,
            )

            # ── Checkpointing ────────────────────────────────────────────────
            if timestep >= next_save:
                path = os.path.join(save_dir, f"policy_{timestep}.pth")
                agent.save(path)
                logger.info("Checkpoint saved → %s", path)
                next_save += save_every

    except (TimeoutError, KeyboardInterrupt) as e:
        logger.warning("Stopping early: %s", e)

    finally:
        final_path = os.path.join(save_dir, f"policy_{timestep}.pth")
        agent.save(final_path)
        logger.info("Checkpoint saved → %s", final_path)
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(
    checkpoint: str,
    n_episodes: int = 5,
    vpt_model: str = "./scripts/foundation-model-1x.model",
    vpt_weights: str = "./scripts/foundation-model-1x.weights",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = PPOAgent(
        n_actions=N_ACTIONS,
        n_inventory=len(OBS_ITEMS),
        vpt_model=vpt_model,
        vpt_weights=vpt_weights,
        device=device,
    )
    agent.load(checkpoint)
    agent.policy.eval()
    logger.info("device=%s", device)
    logger.info("Loaded %s", checkpoint)

    ep = 0
    while ep < n_episodes:
        env = make_env()
        try:
            obs = _reset_with_timeout(env)
            total_rew = 0.0
            done = False
            steps = 0

            while not done:
                action, _, _ = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                total_rew += reward
                steps += 1
                env.render()

            logger.info("Episode %2d: reward=%.2f  steps=%d", ep + 1, total_rew, steps)
            ep += 1
        except Exception as e:
            logger.warning(
                "Episode %2d: env error (%s) — restarting Java process", ep + 1, e
            )
        finally:
            env.close()
