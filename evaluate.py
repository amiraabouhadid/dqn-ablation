"""
Evaluate a trained agent (no training, no epsilon).

Usage:
  python evaluate.py --agent vanilla --env PongNoFrameskip-v4 --episodes 30
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from src.env.atari_wrappers import make_atari
from src.agents import AGENTS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=list(AGENTS.keys()), required=True)
    p.add_argument("--env", default="PongNoFrameskip-v4")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--render", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_str)

    env = make_atari(
        args.env, seed=0, training=False,
        render_mode="human" if args.render else None,
    )

    AgentClass = AGENTS[args.agent]
    # Dummy config — we only need network architecture
    config = {"buffer_size": 1000, "learning_starts": 999_999}
    agent = AgentClass(env, config, device)

    # Load weights
    model_path = Path(args.results_dir) / "models" / f"{agent.name}_{args.env}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}. Run train.py first.")

    agent.online_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.online_net.eval()

    rewards = []
    for ep in range(args.episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                s = torch.tensor(state[None], dtype=torch.uint8, device=device)
                action = int(agent.online_net(s).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"  ep {ep+1:3d}: {total_reward:.1f}")

    env.close()
    print(f"\n{args.agent} | mean={np.mean(rewards):.2f}  std={np.std(rewards):.2f}  "
          f"min={np.min(rewards):.1f}  max={np.max(rewards):.1f}")


if __name__ == "__main__":
    main()
