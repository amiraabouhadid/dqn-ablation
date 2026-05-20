"""
Prioritized Experience Replay (PER) — Schaul et al. (2015).

Priority of transition i: p_i = |δ_i| + ε
Sampling probability:     P(i) = p_i^α / Σ p_j^α
Importance-sampling weight: w_i = (N · P(i))^(-β) / max_j w_j

α controls how much prioritisation is used (0 = uniform).
β corrects the bias introduced by non-uniform sampling (annealed 0.4 → 1.0).

Uses a Sum Tree for O(log N) priority updates and sampling.
"""

import numpy as np
import torch
from typing import Tuple


class SumTree:
    """
    Binary tree where each leaf holds a priority.
    Internal nodes hold sum of children.
    Allows O(log N) update and O(log N) prefix-sum sample.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data_pointer = 0

    def update(self, index: int, priority: float) -> None:
        """Update leaf at `index` (leaf index, 0-based) with new priority."""
        tree_idx = index + self.capacity
        self.tree[tree_idx] = priority
        # Propagate up
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]

    def get(self, value: float) -> Tuple[int, float]:
        """
        Retrieve leaf index whose prefix sum covers `value`.
        Returns (leaf_index_0based, priority).
        """
        idx = 1  # start at root
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        leaf_idx = idx - self.capacity
        return leaf_idx, self.tree[idx]

    @property
    def total(self) -> float:
        return self.tree[1]

    @property
    def max_priority(self) -> float:
        return self.tree[self.capacity:self.capacity * 2].max()


class PrioritizedReplayBuffer:
    """PER buffer — same interface as ReplayBuffer but with priority-weighted sampling."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple,
        device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1  # for beta annealing

        self.tree = SumTree(capacity)
        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    @property
    def beta(self) -> float:
        """Linearly anneal beta from beta_start → 1.0."""
        fraction = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def push(self, state, action, reward, next_state, done) -> None:
        # New transitions get max priority so they are sampled at least once
        max_p = self.tree.max_priority
        priority = max_p if max_p > 0 else 1.0

        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        self.tree.update(self.pos, priority ** self.alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        self.frame += 1
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            val = np.random.uniform(segment * i, segment * (i + 1))
            idx, p = self.tree.get(val)
            indices[i] = idx
            priorities[i] = p

        # Importance-sampling weights
        probs = priorities / (self.tree.total + 1e-10)
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()

        states = torch.tensor(self.states[indices], dtype=torch.uint8, device=self.device)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.uint8, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.long, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones, weights_t, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities after computing new TD errors."""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, p in zip(indices, priorities):
            self.tree.update(int(idx), float(p))

    def __len__(self) -> int:
        return self.size
