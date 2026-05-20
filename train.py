"""
Train a single DQN variant.

Usage:
  python train.py --agent vanilla --env PongNoFrameskip-v4
  python train.py --agent double  --env PongNoFrameskip-v4 --steps 2000000
  python train.py --agent dueling --env PongNoFrameskip-v4
  python train.py --agent per     --env PongNoFrameskip-v4
"""

import argparse
import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path

from src.env.atari_wrappers import make_atari
from src.agents import AGENTS
from src.utils.logger import Logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=list(AGENTS.keys()), default="vanilla")
    p.add_argument("--env", default="PongNoFrameskip-v4")
    p.add_argument("--steps", type=int, default=None, help="Override total_steps")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed
    if args.steps is not None:
        config["total_steps"] = args.steps

    set_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device} | Agent: {args.agent} | Env: {args.env}")

    env = make_atari(args.env, seed=args.seed, training=True)

    AgentClass = AGENTS[args.agent]
    agent = AgentClass(env, config, device)

    log_dir = Path(args.results_dir) / "logs"
    logger = Logger(str(log_dir), agent.name, args.env)

    total_steps = config["total_steps"]
    log_interval = config.get("log_interval", 10)

    print(f"Training for {total_steps:,} steps...")

    while agent.steps < total_steps:
        ep_reward = agent.run_episode()

        recent = agent.episode_rewards[-min(100, len(agent.episode_rewards)):]
        avg = np.mean(recent)
        logger.log(agent.steps, agent.episodes, ep_reward, avg, agent.epsilon())

        if agent.episodes % log_interval == 0:
            print(
                f"[{agent.steps:>8,d}] ep={agent.episodes:>5d} "
                f"reward={ep_reward:>7.2f}  avg100={avg:>7.2f}  "
                f"eps={agent.epsilon():.3f}"
            )

    logger.close()
    env.close()

    # Save model
    save_dir = Path(args.results_dir) / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{agent.name}_{args.env}.pt"
    torch.save(agent.online_net.state_dict(), model_path)
    print(f"Model saved → {model_path}")


if __name__ == "__main__":
    main()
