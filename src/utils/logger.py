"""Lightweight CSV + console logger."""

import csv
import os
import time
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str, agent_name: str, env_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.agent_name = agent_name
        self.env_id = env_id
        self.start_time = time.time()

        csv_path = self.log_dir / f"{agent_name}_{env_id.replace('/', '_')}.csv"
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "episode", "reward", "avg_reward_100", "epsilon", "elapsed_s"])
        self._file.flush()

    def log(self, step: int, episode: int, reward: float, avg_reward: float, epsilon: float):
        elapsed = time.time() - self.start_time
        self._writer.writerow([step, episode, f"{reward:.2f}", f"{avg_reward:.2f}", f"{epsilon:.4f}", f"{elapsed:.1f}"])
        self._file.flush()

    def close(self):
        self._file.close()
