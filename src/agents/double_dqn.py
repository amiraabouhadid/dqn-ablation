"""
Double DQN — van Hasselt et al. (2016).

Fixes the overestimation bias of Vanilla DQN.

Key change (one line different from Vanilla):
  - Action SELECTION: online network         → a* = argmax_a Q(s', a)
  - Action EVALUATION: target network        → Q̂(s', a*)

Loss: L = E[(r + γ Q̂(s', argmax_a Q(s',a)) - Q(s,a))²]

This decouples selection from evaluation, preventing
the maximisation bias that inflates Q-values.
"""

import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.networks.dqn_network import DQNNetwork
from src.replay.replay_buffer import ReplayBuffer


class DoubleDQN(BaseAgent):
    """van Hasselt et al. (2016) — fixes overestimation."""

    name = "double_dqn"

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

        q_values = self.online_net(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: select action with ONLINE net, evaluate with TARGET net
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = (weights * F.mse_loss(q_pred, q_target, reduction="none")).mean()
        return loss
