"""
Standard Atari preprocessing stack.

Following Mnih et al. (2015):
  - Grayscale + resize to 84x84
  - Frame skip = 4 (repeat action, max over last 2 frames)
  - Stack 4 consecutive frames as state
  - Clip rewards to [-1, +1]
  - Life-loss treated as episode end during training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import cv2
import ale_py

gym.register_envs(ale_py)


class NoopResetEnv(gym.Wrapper):
    """Sample random number of no-ops at episode start."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action for `skip` frames; return max-pooled obs."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodicLifeEnv(gym.Wrapper):
    """Treat loss of life as end of episode (training only)."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE to start episode (required by some Atari games)."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class WarpFrame(gym.ObservationWrapper):
    """Grayscale + resize to 84x84."""

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1}."""

    def reward(self, reward):
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    """Stack `k` last frames along channel dimension."""

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2)


class TransposeObs(gym.ObservationWrapper):
    """HWC → CHW for PyTorch."""

    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(shp[2], shp[0], shp[1]),
            dtype=np.uint8
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


def make_atari(env_id, seed=0, training=True, render_mode=None):
    """Build full Atari preprocessing stack."""
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if training:
        env = EpisodicLifeEnv(env)
    action_meanings = env.unwrapped.get_action_meanings()
    if "FIRE" in action_meanings:
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if training:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=4)
    env = TransposeObs(env)
    env.reset(seed=seed)
    return env
