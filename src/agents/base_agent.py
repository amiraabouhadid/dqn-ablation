"""
Base DQN agent — shared training loop logic.

All variants (Vanilla, Double, Dueling, PER) inherit from here.
Subclasses only override:
  - _build_network()  → return the Q-network
  - _build_buffer()   → return the replay buffer
  - _compute_loss()   → TD loss with variant-specific target computation
  - _post_update()    → optional hook (e.g. priority update for PER)
"""

import numpy as np
import torch
import torch.optim as optim
import copy
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):

    def __init__(self, env, config: Dict[str, Any], device: torch.device):
        self.env = env
        self.config = config
        self.device = device

        self.n_actions = env.action_space.n
        self.obs_shape = env.observation_space.shape  # (C, H, W)

        # Networks
        self.online_net = self._build_network().to(device)
        self.target_net = copy.deepcopy(self.online_net).to(device)
        self.target_net.eval()

        # Optimiser
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config.get("lr", 1e-4),
        )

        # Replay buffer
        self.buffer = self._build_buffer()

        # Counters & hyperparams
        self.steps = 0
        self.episodes = 0
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.learning_starts = config.get("learning_starts", 50_000)
        self.train_freq = config.get("train_freq", 4)
        self.target_update_freq = config.get("target_update_freq", 10_000)
        self.eps_start = config.get("eps_start", 1.0)
        self.eps_end = config.get("eps_end", 0.1)
        self.eps_decay_steps = config.get("eps_decay_steps", 1_000_000)

        # Logging
        self.episode_rewards: list = []
        self.losses: list = []
        self.epsilons: list = []

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _build_network(self):
        """Return the Q-network (nn.Module)."""

    @abstractmethod
    def _build_buffer(self):
        """Return the replay buffer."""

    @abstractmethod
    def _compute_loss(self, batch) -> torch.Tensor:
        """Compute scalar loss from a sampled batch."""

    def _post_update(self, batch, td_errors: np.ndarray) -> None:
        """Called after each gradient step. Override for PER priority updates."""

    # ------------------------------------------------------------------ #
    # Shared logic                                                         #
    # ------------------------------------------------------------------ #

    def epsilon(self) -> float:
        """Linear epsilon schedule."""
        progress = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon():
            return self.env.action_space.sample()
        with torch.no_grad():
            s = torch.tensor(state[None], dtype=torch.uint8, device=self.device)
            q = self.online_net(s)
        return int(q.argmax(dim=1).item())

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self) -> float | None:
        if len(self.buffer) < self.learning_starts:
            return None
        if self.steps % self.train_freq != 0:
            return None

        batch = self.buffer.sample(self.batch_size)
        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # PER priority update hook (no-op for uniform buffers)
        with torch.no_grad():
            td_errors = self._td_errors(batch).cpu().numpy()
        self._post_update(batch, td_errors)

        if self.steps % self.target_update_freq == 0:
            self._sync_target()

        return loss.item()

    def _td_errors(self, batch) -> torch.Tensor:
        """Return per-sample |δ| (for PER; recomputed cheaply)."""
        states, actions, rewards, next_states, dones, weights, _ = batch
        with torch.no_grad():
            q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            q_next = self.target_net(next_states).max(1)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)
        return (q_pred - q_target).abs()

    def run_episode(self) -> float:
        state, _ = self.env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            self.steps += 1

            loss = self.train_step()
            if loss is not None:
                self.losses.append(loss)
                self.epsilons.append(self.epsilon())

        self.episodes += 1
        self.episode_rewards.append(episode_reward)
        return episode_reward

    def train(self, total_steps: int, log_interval: int = 10):
        """Train until total_steps environment steps."""
        while self.steps < total_steps:
            ep_reward = self.run_episode()
            if self.episodes % log_interval == 0:
                recent = self.episode_rewards[-log_interval:]
                print(
                    f"[{self.steps:>8d}] ep={self.episodes:>5d} "
                    f"reward={np.mean(recent):>7.2f} "
                    f"eps={self.epsilon():.3f}"
                )
