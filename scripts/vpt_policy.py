"""
vpt_policy.py
=============
Swaps PolicyNet's shallow 3-layer CNN for VPT's pretrained IMPALA CNN.

Setup
-----
1. Clone the VPT repo alongside this project:
       git clone https://github.com/openai/Video-Pre-Training.git

2. Download a foundation model weights file from the VPT repo releases, e.g.:
       foundation-model-1x.model   (model architecture JSON)
       foundation-model-1x.weights (PyTorch weights)

3. Install VPT's dependencies (if not already):
       pip install -r Video-Pre-Training/requirements.txt

Usage
-----
    from scripts.vpt_policy import VPTPolicyNet, load_vpt_policy

    policy = load_vpt_policy(
        vpt_model_path  = "Video-Pre-Training/foundation-model-1x.model",
        vpt_weights_path= "Video-Pre-Training/foundation-model-1x.weights",
        n_actions       = N_ACTIONS,
        n_inventory     = len(OBS_ITEMS),
        freeze_cnn      = True,   # unfreeze after ~50k PPO steps
    )

Then pass `policy` to PPOAgent in place of the default PolicyNet.

Architecture
------------
VPT IMPALA CNN (pretrained, optionally frozen)
    ↓  1024-d features (same as original CNN — trunk unchanged)
Inventory encoder: Linear(N→64) + ReLU
Shared trunk:      Linear(1088→512) + ReLU
Actor head:        Linear(512→N_ACTIONS)
Critic head:       Linear(512→1)

The VPT CNN outputs exactly 1024-d for 64×64 input with the 1x foundation
model, so the trunk and heads are drop-in compatible with the original PolicyNet.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# IMPALA CNN definition
# ──────────────────────────────────────────────────────────────────────────────
# Reproduced here so you don't need to add the full VPT repo to sys.path.
# This matches the architecture used in the VPT foundation models exactly.


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = torch.relu(x)
        out = self.conv0(out)
        out = torch.relu(out)
        out = self.conv1(out)
        return out + x


class _ImpalaStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = _ResidualBlock(out_channels)
        self.res1 = _ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res0(x)
        x = self.res1(x)
        return x


class ImpalaCNN(nn.Module):
    """
    IMPALA-style CNN identical to VPT's visual encoder.

    Input : (B, 3, 64, 64)  float32 in [0, 1]
    Output: (B, 1024)        for the 1x foundation model
    """

    # Channel sizes for each stack — 1x, 2x, 3x foundation models multiply these
    BASE_CHANNELS = [16, 32, 32]

    def __init__(self, width_multiplier: int = 1, output_size: int = 1024):
        super().__init__()
        chans = [c * width_multiplier for c in self.BASE_CHANNELS]

        stacks = []
        in_ch = 3
        for out_ch in chans:
            stacks.append(_ImpalaStack(in_ch, out_ch))
            in_ch = out_ch
        self.stacks = nn.Sequential(*stacks)
        self.flatten = nn.Flatten()

        # Compute flattened size for 64×64 input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            flat = self.flatten(self.stacks(dummy))
            flat_size = flat.shape[1]

        self.linear = nn.Linear(flat_size, output_size)

    def forward(self, x):
        x = self.stacks(x)
        x = torch.relu(self.flatten(x))
        x = torch.relu(self.linear(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# VPT-backed PolicyNet
# ──────────────────────────────────────────────────────────────────────────────


class VPTPolicyNet(nn.Module):
    """
    Drop-in replacement for PolicyNet that uses VPT's IMPALA CNN as the
    visual encoder instead of the shallow 3-layer CNN.

    The trunk, actor, and critic are identical to the original PolicyNet,
    so checkpoints for those layers are interchangeable.
    """

    CNN_OUT = 1024  # output size of ImpalaCNN — matches original PolicyNet

    def __init__(
        self,
        n_actions: int,
        n_inventory: int,
        width_multiplier: int = 1,
        freeze_cnn: bool = True,
    ):
        super().__init__()

        self.cnn = ImpalaCNN(
            width_multiplier=width_multiplier,
            output_size=self.CNN_OUT,
        )

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad_(False)

        self.inv_enc = nn.Sequential(
            nn.Linear(n_inventory, 64),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.CNN_OUT + 64, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

        # Initialise everything except the pretrained CNN
        for m in [self.inv_enc, self.trunk, self.actor, self.critic]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def cnn_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.cnn.parameters())

    def unfreeze_cnn(self):
        """Call this after ~50k PPO steps to fine-tune the full network."""
        for p in self.cnn.parameters():
            p.requires_grad_(True)
        logger.info("CNN unfrozen — full network now trainable")

    def forward(self, pov, inventory):
        vis = self.cnn(pov)
        inv = self.inv_enc(inventory)
        shared = self.trunk(torch.cat([vis, inv], dim=-1))
        return self.actor(shared), self.critic(shared).squeeze(-1)

    @staticmethod
    def _act(logits, value, action):
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def forward_from_features(self, cnn_features, inventory, action=None):
        """Skip the CNN — use pre-computed features. Faster when CNN is frozen."""
        inv = self.inv_enc(inventory)
        shared = self.trunk(torch.cat([cnn_features, inv], dim=-1))
        return self._act(self.actor(shared), self.critic(shared).squeeze(-1), action)

    def get_action_and_value(self, pov, inventory, action=None):
        logits, value = self.forward(pov, inventory)
        return self._act(logits, value, action)


# ──────────────────────────────────────────────────────────────────────────────
# Weight loader
# ──────────────────────────────────────────────────────────────────────────────


def load_vpt_policy(
    vpt_model_path: str,
    vpt_weights_path: str,
    n_actions: int,
    n_inventory: int,
    freeze_cnn: bool = True,
    device: str = "cpu",
) -> VPTPolicyNet:
    """
    Build a VPTPolicyNet and load the pretrained IMPALA CNN weights from a
    VPT foundation model checkpoint.

    Parameters
    ----------
    vpt_model_path   : path to the .model file (architecture JSON)
    vpt_weights_path : path to the .weights file (PyTorch state dict)
    freeze_cnn       : if True, CNN gradients are disabled for the first phase
                       of PPO.  Call policy.unfreeze_cnn() to enable them later.
    """
    policy = VPTPolicyNet(
        n_actions=n_actions,
        n_inventory=n_inventory,
        freeze_cnn=freeze_cnn,
    ).to(device)

    logger.info("Loading weights from %s …", vpt_weights_path)
    raw = torch.load(vpt_weights_path, map_location=device, weights_only=False)

    # VPT state dicts nest the CNN under "net.img_process" or similar keys.
    # We extract only the keys that belong to the IMPALA CNN stacks + linear.
    cnn_state = {}
    for prefix in ("net.img_process.", "img_process."):
        cnn_state = {
            k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)
        }
        if cnn_state:
            break

    if not cnn_state:
        raise RuntimeError(
            "Could not find IMPALA CNN weights in the VPT checkpoint.\n"
            f"Available top-level keys: {list(raw.keys())[:10]}\n"
            "Check the key prefix and update cnn_prefix in load_vpt_policy()."
        )

    missing, unexpected = policy.cnn.load_state_dict(cnn_state, strict=False)
    if missing:
        logger.warning("Missing CNN keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected CNN keys: %s", unexpected)

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in policy.cnn.parameters())
    logger.info(
        "CNN loaded (%s params %s)",
        f"{frozen:,}",
        "frozen" if freeze_cnn else "trainable",
    )
    logger.info("Trainable params: %s", f"{trainable:,}")
    return policy
