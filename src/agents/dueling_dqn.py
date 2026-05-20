"""
Dueling DQN — Wang et al. (2016).

Adds the Dueling Network architecture on top of Double DQN.

Key change: swap DQNNetwork → DuelingNetwork.
  The network outputs Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

This separation is beneficial when:
  - Many actions have similar Q-values (common in Atari)
  - The agent needs to know state value without action effect

The training loop is identical to Double DQN.
"""

import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.networks.dueling_network import DuelingNetwork
from src.replay.replay_buffer import ReplayBuffer


class DuelingDQN(BaseAgent):
    """Wang et al. (2016) — dueling streams V(s) + A(s,a)."""

    name = "dueling_dqn"

    def _build_network(self):
        return DuelingNetwork(n_actions=self.n_actions, in_channels=self.obs_shape[0])

    def _build_buffer(self):
        return ReplayBuffer(
            capacity=self.config.get("buffer_size", 100_000),
            obs_shape=self.obs_shape,
            device=self.device,
        )

    def _compute_loss(self, batch) -> torch.Tensor:
        states, actions, rewards, next_states, dones, weights, _ = batch

        q_values = self.online_net(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Keep Double DQN target computation
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = (weights * F.mse_loss(q_pred, q_target, reduction="none")).mean()
        return loss
