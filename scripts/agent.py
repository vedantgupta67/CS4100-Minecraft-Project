"""
agent.py
========
PPOAgent — Actor-Critic PPO with GAE backed by VPTPolicyNet.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from vpt_policy import load_vpt_policy


class PPOAgent:
    def __init__(
        self,
        n_actions: int,
        n_inventory: int,
        vpt_model: str,
        vpt_weights: str,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.device = device

        self.policy = load_vpt_policy(
            vpt_model_path=vpt_model,
            vpt_weights_path=vpt_weights,
            n_actions=n_actions,
            n_inventory=n_inventory,
            freeze_cnn=True,
            device=device,
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs: dict):
        """Return (action_idx, log_prob, value) for a single obs dict."""
        actions, log_probs, values = self.select_action_batch(
            {
                "pov": obs["pov"][np.newaxis],
                "inventory": obs["inventory"][np.newaxis],
            }
        )
        return int(actions[0]), float(log_probs[0]), float(values[0])

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action_batch(self, obs: dict):
        """Return (actions, log_probs, values) as (N,) numpy arrays for batched obs.
        obs['pov'] is (N, 3, H, W) uint8; obs['inventory'] is (N, inv_size) float32.
        """
        pov = (
            torch.as_tensor(obs["pov"], dtype=torch.float32, device=self.device) / 255.0
        )
        inv = torch.as_tensor(obs["inventory"], dtype=torch.float32, device=self.device)
        actions, log_probs, _, values = self.policy.get_action_and_value(pov, inv)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    # ------------------------------------------------------------------
    @staticmethod
    def compute_gae(rewards, values, dones, next_values, gamma, gae_lambda):
        """Generalised Advantage Estimation for single-env (T,) or multi-env (T, N) arrays.

        next_values : scalar or (N,) bootstrap values for the state after the rollout.
        Returns advantages and returns flattened to (T,) or (T*N,).
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        next_values = np.atleast_1d(np.asarray(next_values, dtype=np.float32))

        if rewards.ndim == 1:
            rewards = rewards[:, None]
            values = values[:, None]
            dones = dones[:, None]

        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            nv = next_values if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * nv * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages.reshape(-1), returns.reshape(-1)

    # ------------------------------------------------------------------
    def update(self, rollout: dict, next_obs: dict) -> dict:
        """Run PPO update epochs from one rollout.

        Works for both single-env (rollout entries are scalars / 1-D arrays)
        and multi-env (rollout entries are (N,) arrays stacked to (T, N)).
        """

        def _flat(key, dtype, n_feat=0):
            """Flatten (T[, N], *feature) → (T[*N], *feature)."""
            arr = np.asarray(rollout[key], dtype=dtype)
            return arr.reshape(-1, *arr.shape[-n_feat:]) if n_feat else arr.reshape(-1)

        # Bootstrap value(s) for the state after the rollout
        pov = next_obs["pov"]
        inv = next_obs["inventory"]
        if pov.ndim == 3:  # single env: (C,H,W) → (1,C,H,W)
            pov, inv = pov[np.newaxis], inv[np.newaxis]
        _, _, next_vals = self.select_action_batch({"pov": pov, "inventory": inv})

        advantages, returns = self.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            next_vals,
            self.gamma,
            self.gae_lambda,
        )

        # Convert rollout to tensors — flatten (T[,N], …) → (T[*N], …)
        povs = (
            torch.as_tensor(
                _flat("povs", np.uint8, n_feat=3),
                dtype=torch.float32,
                device=self.device,
            )
            / 255.0
        )
        invs = torch.as_tensor(
            _flat("invs", np.float32, n_feat=1), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            _flat("actions", np.int64), dtype=torch.long, device=self.device
        )
        old_lps = torch.as_tensor(
            _flat("log_probs", np.float32), dtype=torch.float32, device=self.device
        )
        advs = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        rets = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        n = len(actions)
        indices = np.arange(n)
        losses = {"pg": 0.0, "vf": 0.0, "ent": 0.0}

        # When the CNN is frozen, pre-compute features once per rollout rather
        # than re-running the CNN every minibatch across all epochs.
        frozen_feats = None
        if hasattr(self.policy, "cnn_is_frozen") and self.policy.cnn_is_frozen():
            with torch.no_grad():
                frozen_feats = self.policy.cnn(povs)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start : start + self.minibatch_size]

                if frozen_feats is not None:
                    _, new_lps, entropy, new_vals = self.policy.forward_from_features(
                        frozen_feats[idx], invs[idx], actions[idx]
                    )
                else:
                    _, new_lps, entropy, new_vals = self.policy.get_action_and_value(
                        povs[idx], invs[idx], actions[idx]
                    )

                ratio = torch.exp(new_lps - old_lps[idx])
                adv_b = advs[idx]
                pg1 = -adv_b * ratio
                pg2 = -adv_b * torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                )
                pg_loss = torch.max(pg1, pg2).mean()
                vf_loss = F.mse_loss(new_vals, rets[idx])
                ent_loss = entropy.mean()
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses["pg"] += pg_loss.item()
                losses["vf"] += vf_loss.item()
                losses["ent"] += ent_loss.item()

        n_updates = self.n_epochs * max(1, n // self.minibatch_size)
        return {k: v / n_updates for k, v in losses.items()}

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
