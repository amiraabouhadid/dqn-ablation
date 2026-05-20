"""
Ablation plotting utilities.

Produces publication-quality figures:
  1. Learning curves (smoothed reward vs. steps) for all agents
  2. Final performance bar chart
  3. Sample efficiency (steps to reach threshold reward)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List


AGENT_LABELS = {
    "vanilla_dqn": "Vanilla DQN",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "per_dqn": "Double + Dueling + PER",
}

COLORS = {
    "vanilla_dqn": "#e74c3c",
    "double_dqn": "#3498db",
    "dueling_dqn": "#2ecc71",
    "per_dqn": "#9b59b6",
}


def _smooth(values: List[float], window: int = 50) -> np.ndarray:
    """Exponential moving average."""
    if len(values) == 0:
        return np.array([])
    arr = np.array(values, dtype=float)
    smoothed = np.zeros_like(arr)
    alpha = 2.0 / (window + 1)
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def load_csv(csv_path: str) -> Dict[str, List]:
    data: Dict[str, List] = {"step": [], "reward": [], "avg_reward_100": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["step"].append(int(row["step"]))
            data["reward"].append(float(row["reward"]))
            data["avg_reward_100"].append(float(row["avg_reward_100"]))
    return data


def plot_learning_curves(
    results: Dict[str, Dict],
    save_path: str,
    env_id: str,
    smooth_window: int = 50,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent_name, data in results.items():
        label = AGENT_LABELS.get(agent_name, agent_name)
        color = COLORS.get(agent_name, "gray")
        steps = np.array(data["step"])
        rewards = _smooth(data["avg_reward_100"], window=smooth_window)
        ax.plot(steps, rewards, label=label, color=color, linewidth=2.0)

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Episode Reward (EMA)", fontsize=13)
    ax.set_title(f"DQN Ablation Study — {env_id}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_performance(
    results: Dict[str, Dict],
    save_path: str,
    env_id: str,
    last_n: int = 100,
):
    agents = list(results.keys())
    means = []
    stds = []
    for agent_name in agents:
        rewards = results[agent_name]["reward"][-last_n:]
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))

    labels = [AGENT_LABELS.get(a, a) for a in agents]
    colors = [COLORS.get(a, "gray") for a in agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_ylabel(f"Mean Reward (last {last_n} eps ± std)", fontsize=12)
    ax.set_title(f"Final Performance Comparison — {env_id}", fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_efficiency(
    results: Dict[str, Dict],
    save_path: str,
    env_id: str,
    threshold: float = 0.0,
):
    """Steps required to first exceed `threshold` reward (sustained over 10 eps)."""
    agents = list(results.keys())
    steps_to_solve = {}

    for agent_name in agents:
        data = results[agent_name]
        rewards = np.array(data["avg_reward_100"])
        steps = np.array(data["step"])
        solved = None
        for i in range(9, len(rewards)):
            if all(rewards[i-9:i+1] >= threshold):
                solved = steps[i]
                break
        steps_to_solve[agent_name] = solved

    labels = [AGENT_LABELS.get(a, a) for a in agents]
    values = [steps_to_solve[a] if steps_to_solve[a] else float("nan") for a in agents]
    colors = [COLORS.get(a, "gray") for a in agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, [v / 1e6 if not np.isnan(v) else 0 for v in values], color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Steps to Threshold (M)", fontsize=12)
    ax.set_title(f"Sample Efficiency (threshold={threshold}) — {env_id}", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        label = f"{val/1e6:.2f}M" if not np.isnan(val) else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                label, ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
