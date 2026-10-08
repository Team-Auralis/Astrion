"""
Pokemon-specific neural network for ORION.

Architecture:
- Screen CNN branch: processes 80x72 grayscale frames
- State MLP branch: processes 20-dimensional game state features
- Fusion: concat + ResNet blocks
- Dual heads: policy (8 buttons) + value (expected return)
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCREEN_H = 72  # 144 / 2 (downsampled)
SCREEN_W = 80  # 160 / 2 (downsampled)
STATE_DIM = 20
NUM_BUTTONS = 8


class ConvBlock(nn.Module):
    """Conv + BN + ReLU with optional residual connection."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class PokemonResNet(nn.Module):
    """
    Dual-branch ResNet for Pokemon gameplay.

    Branch 1 (Screen): 80x72 grayscale -> CNN -> 256-dim feature vector
    Branch 2 (State): state_dim features -> MLP -> 64-dim feature vector
    Fusion: concat(256+64=320) -> ResNet blocks -> policy + value heads
    """

    def __init__(
        self,
        num_buttons: int = NUM_BUTTONS,
        in_channels: int = 4,
        state_dim: int = STATE_DIM,
    ):
        super().__init__()
        self.num_buttons = num_buttons
        self.state_dim = state_dim

        # --- Screen CNN branch ---
        self.screen_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 36x40
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 18x20
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.screen_res1 = ConvBlock(64, 64)
        self.screen_res2 = ConvBlock(64, 128, stride=2)  # 9x10
        self.screen_res3 = ConvBlock(128, 128)
        self.screen_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.screen_fc = nn.Linear(128 * 4 * 4, 256)

        # --- State MLP branch ---
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # --- Fusion + ResNet blocks ---
        fusion_dim = 256 + 64  # 320
        self.fusion_fc = nn.Linear(fusion_dim, 256)
        self.res1 = self._make_residual(256)
        self.res2 = self._make_residual(256)
        self.res3 = self._make_residual(256)

        # --- Policy head (buttons) ---
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_buttons),
        )

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self._init_weights()

    @staticmethod
    def _make_residual(channels):
        return nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        screen: torch.Tensor,
        state: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            screen: (B, 1, 72, 80) or (B, 72, 80) grayscale screen
            state: (B, state_dim) state feature vector (optional)

        Returns:
            policy_logits: (B, num_buttons)
            value: (B, 1) in [-1, 1]
        """
        if screen.dim() == 3:
            screen = screen.unsqueeze(1)
        if (
            screen.dim() == 4
            and screen.size(1) == 1
            and self.screen_conv[0].in_channels > 1
        ):
            # Repeat single channel to match expected input channels
            screen = screen.repeat(1, self.screen_conv[0].in_channels, 1, 1)
        screen = screen.float() / 255.0  # Normalize to [0, 1]

        # Screen branch
        x = self.screen_conv(screen)
        x = self.screen_res1(x)
        x = self.screen_res2(x)
        x = self.screen_res3(x)
        x = self.screen_pool(x)
        x = x.view(x.size(0), -1)
        screen_feat = F.relu(self.screen_fc(x))

        # State branch
        if state is not None:
            state_feat = self.state_fc(state.float())
        else:
            state_feat = torch.zeros(screen.size(0), 64, device=screen.device)

        # Fusion
        fused = torch.cat([screen_feat, state_feat], dim=1)
        fused = F.relu(self.fusion_fc(fused))
        fused = fused + self.res1(fused)  # Residual connection
        fused = fused + self.res2(fused)
        fused = fused + self.res3(fused)

        # Heads
        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)

        return policy_logits, value

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PokemonEncoder:
    """
    Converts a PokemonEnv observation dict into tensors for the network.
    Handles frame stacking for temporal context.
    """

    def __init__(
        self, frame_stack: int = 4, screen_h: int = SCREEN_H, screen_w: int = SCREEN_W
    ):
        self.frame_stack = frame_stack
        self.screen_h = screen_h
        self.screen_w = screen_w
        self._frames = []

    def reset(self):
        self._frames = []

    def encode(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single observation into (screen_tensor, state_tensor).

        screen_tensor: (frame_stack, H, W) stacked grayscale frames
        state_tensor: (STATE_DIM,) state features
        """
        screen = obs["screen"]  # (H, W) uint8

        # Temporal frame stacking
        self._frames.append(screen.copy())
        if len(self._frames) > self.frame_stack:
            self._frames.pop(0)

        # Pad with first frame if not enough frames yet
        while len(self._frames) < self.frame_stack:
            self._frames.insert(0, screen.copy())

        stacked = np.stack(self._frames, axis=0)  # (stack, H, W)
        screen_tensor = torch.from_numpy(stacked).float()

        state = obs["state"]  # (STATE_DIM,)
        state_tensor = torch.from_numpy(state).float()

        return screen_tensor, state_tensor

    def encode_batch(self, observations: list) -> Tuple[torch.Tensor, torch.Tensor]:
        screens, states = [], []
        for obs in observations:
            s, st = self.encode(obs)
            screens.append(s)
            states.append(st)
        return torch.stack(screens), torch.stack(states)
