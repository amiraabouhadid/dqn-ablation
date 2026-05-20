"""
Vanilla DQN — Mnih et al. (2015).

Key components:
  - Q-network (DQNNetwork)
  - Target network (frozen copy, synced every C steps)
  - Uniform experience replay
  - ε-greedy exploration

Loss: L = E[(r + γ max_a' Q̂(s',a') - Q(s,a))²]
where Q̂ is the target network.
"""

import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.networks.dqn_network import DQNNetwork
from src.replay.replay_buffer import ReplayBuffer


class VanillaDQN(BaseAgent):
    """Mnih et al. (2015) — baseline."""

    name = "vanilla_dqn"

    def _build_network(self):
        return DQNNetwork(n_actions=self.n_actions, in_channels=self.obs_shape[0])

    def _build_buffer(self):
        return ReplayBuffer(
            capacity=self.config.get("buffer_size", 100_000),
            obs_shape=self.obs_shape,
            device=self.device,
        )

    def _compute_loss(self, batch) -> torch.Tensor:
        states, actions, rewards, next_states, dones, weights, _ = batch

        # Current Q-values
        q_values = self.online_net(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: use target network, take max over actions
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1)[0]
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = (weights * F.mse_loss(q_pred, q_target, reduction="none")).mean()
        return loss
