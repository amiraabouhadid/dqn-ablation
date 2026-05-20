"""
Uniform Experience Replay Buffer.

Stores (s, a, r, s', done) tuples in a circular buffer.
Samples uniformly at random — used by Vanilla DQN, Double DQN, Dueling DQN.
"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""

    def __init__(self, capacity: int, obs_shape: Tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        # Pre-allocate all storage as numpy arrays for memory efficiency
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self._encode(indices)

    def _encode(self, indices: np.ndarray):
        states = torch.tensor(self.states[indices], dtype=torch.uint8, device=self.device)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.uint8, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.long, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        # weights = 1.0 for uniform buffer (PER overrides this)
        weights = torch.ones(len(indices), dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones, weights, indices

    def __len__(self) -> int:
        return self.size
