"""
Dueling Network Architecture — Wang et al. (2016).

Shared CNN → two heads:
  V(s)        : scalar state value
  A(s, a)     : per-action advantage

Q(s, a) = V(s) + A(s,a) - mean_a[ A(s,a) ]

The mean subtraction removes the identifiability problem
(V and A are otherwise underdetermined individually).
"""

import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    """Dueling Q-network. Drop-in replacement for DQNNetwork."""

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(in_channels)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        return int(self.conv(dummy).reshape(1, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        features = self.conv(x).reshape(x.size(0), -1)

        value = self.value_stream(features)              # (B, 1)
        advantage = self.advantage_stream(features)      # (B, n_actions)

        # Combine: subtract mean advantage for identifiability
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
