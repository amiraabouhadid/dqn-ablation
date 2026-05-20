"""
Rainbow-lite: Double + Dueling + PER — Schaul et al. (2015).

Combines all three improvements:
  1. Double DQN target (decoupled selection/evaluation)
  2. Dueling Network (V + A streams)
  3. Prioritized Experience Replay (high-|δ| transitions sampled more)

Key additions over DuelingDQN:
  - PrioritizedReplayBuffer instead of ReplayBuffer
  - Loss weighted by importance-sampling weights w_i
  - _post_update() updates priorities after each gradient step
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.networks.dueling_network import DuelingNetwork
from src.replay.prioritized_replay import PrioritizedReplayBuffer


class PERDQN(BaseAgent):
    """Double + Dueling + PER (Schaul et al., 2015)."""

    name = "per_dqn"

    def _build_network(self):
        return DuelingNetwork(n_actions=self.n_actions, in_channels=self.obs_shape[0])

    def _build_buffer(self):
        return PrioritizedReplayBuffer(
            capacity=self.config.get("buffer_size", 100_000),
            obs_shape=self.obs_shape,
            device=self.device,
            alpha=self.config.get("per_alpha", 0.6),
            beta_start=self.config.get("per_beta_start", 0.4),
            beta_frames=self.config.get("per_beta_frames", 1_000_000),
        )

    def _compute_loss(self, batch) -> torch.Tensor:
        states, actions, rewards, next_states, dones, weights, _ = batch

        q_values = self.online_net(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        # IS-weighted loss — crucial for unbiased learning with PER
        element_loss = F.smooth_l1_loss(q_pred, q_target, reduction="none")
        loss = (weights * element_loss).mean()
        return loss

    def _post_update(self, batch, td_errors: np.ndarray) -> None:
        """Update SumTree priorities with new TD errors."""
        _, _, _, _, _, _, indices = batch
        self.buffer.update_priorities(indices, td_errors)
