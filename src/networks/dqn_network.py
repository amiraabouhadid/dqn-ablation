"""
Standard DQN CNN — Mnih et al. (2015).

Architecture: Conv(32,8,4) → Conv(64,4,2) → Conv(64,3,1) → FC(512) → FC(n_actions)
Input: (batch, 4, 84, 84) uint8 → normalised float32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Vanilla Q-network (used by Vanilla DQN, Double DQN)."""

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(in_channels)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        return int(self.conv(dummy).reshape(1, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
