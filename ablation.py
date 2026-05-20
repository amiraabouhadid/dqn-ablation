"""
Full ablation runner.

Trains all 4 agents sequentially (or in parallel with --parallel),
collects results, and generates all paper figures.

Usage:
  python ablation.py --env PongNoFrameskip-v4 --steps 1000000
  python ablation.py --env PongNoFrameskip-v4 --steps 1000000 --parallel
  python ablation.py --plot-only --env PongNoFrameskip-v4   # just re-plot
"""

import argparse
import subprocess
import sys
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.utils.plotting import (
    load_csv,
    plot_learning_curves,
    plot_final_performance,
    plot_sample_efficiency,
)

AGENT_ORDER = ["vanilla", "double", "dueling", "per"]
AGENT_FILE_NAMES = {
    "vanilla": "vanilla_dqn",
    "double": "double_dqn",
    "dueling": "dueling_dqn",
    "per": "per_dqn",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PongNoFrameskip-v4")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel", action="store_true", help="Train agents in parallel (GPU only)")
    p.add_argument("--plot-only", action="store_true", help="Skip training, just plot existing logs")
    p.add_argument("--threshold", type=float, default=0.0, help="Reward threshold for sample efficiency plot")
    return p.parse_args()


def train_agent(agent_key: str, env_id: str, steps: int, config: str, seed: int, results_dir: str):
    cmd = [
        sys.executable, "train.py",
        "--agent", agent_key,
        "--env", env_id,
        "--steps", str(steps),
        "--config", config,
        "--seed", str(seed),
        "--results-dir", results_dir,
    ]
    print(f"\n{'='*60}")
    print(f"  Starting: {agent_key.upper()} on {env_id}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"\n  Finished {agent_key} in {elapsed/60:.1f} min")
    return agent_key


def load_all_results(results_dir: str, env_id: str) -> dict:
    log_dir = Path(results_dir) / "logs"
    env_slug = env_id.replace("/", "_")
    results = {}
    for agent_key in AGENT_ORDER:
        file_name = AGENT_FILE_NAMES[agent_key]
        csv_path = log_dir / f"{file_name}_{env_slug}.csv"
        if csv_path.exists():
            results[file_name] = load_csv(str(csv_path))
            print(f"  Loaded {csv_path.name} ({len(results[file_name]['step'])} episodes)")
        else:
            print(f"  [WARN] Missing {csv_path} — skipping")
    return results


def print_ablation_table(results: dict):
    print("\n" + "="*65)
    print(f"{'Agent':<30} {'Mean(last100)':>14} {'Std':>8} {'Max':>8}")
    print("-"*65)
    for name, data in results.items():
        rewards = data["reward"][-100:]
        print(f"{name:<30} {np.mean(rewards):>14.2f} {np.std(rewards):>8.2f} {np.max(rewards):>8.2f}")
    print("="*65 + "\n")


def main():
    args = parse_args()
    plots_dir = Path(args.results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ────────────────────────────────────────────────────────
    if not args.plot_only:
        if args.parallel:
            with ProcessPoolExecutor(max_workers=len(AGENT_ORDER)) as ex:
                futures = {
                    ex.submit(train_agent, k, args.env, args.steps,
                              args.config, args.seed, args.results_dir): k
                    for k in AGENT_ORDER
                }
                for fut in as_completed(futures):
                    k = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[ERROR] {k}: {e}")
        else:
            for agent_key in AGENT_ORDER:
                train_agent(agent_key, args.env, args.steps,
                            args.config, args.seed, args.results_dir)

    # ── Load & Plot ─────────────────────────────────────────────────────
    print("\nLoading results...")
    results = load_all_results(args.results_dir, args.env)

    if not results:
        print("No results found. Run training first.")
        return

    print_ablation_table(results)

    env_slug = args.env.replace("/", "_")

    plot_learning_curves(
        results,
        save_path=str(plots_dir / f"learning_curves_{env_slug}.png"),
        env_id=args.env,
    )
    plot_final_performance(
        results,
        save_path=str(plots_dir / f"final_performance_{env_slug}.png"),
        env_id=args.env,
    )
    plot_sample_efficiency(
        results,
        save_path=str(plots_dir / f"sample_efficiency_{env_slug}.png"),
        env_id=args.env,
        threshold=args.threshold,
    )

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
