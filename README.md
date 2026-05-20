# DQN Ablation Study on Atari

Controlled ablation: Vanilla DQN → Double DQN → Dueling DQN → Double + Dueling + PER.  
Each improvement is a clean, swappable module. Output is a mini research paper with figures.

## Structure

```
src/agents/        # 4 agents, each overrides ~3 methods of BaseAgent
src/networks/      # DQNNetwork (standard) + DuelingNetwork (V+A streams)
src/replay/        # ReplayBuffer (uniform) + PrioritizedReplayBuffer (SumTree)
src/env/           # Full Atari preprocessing stack (Mnih et al. 2015)
train.py           # Train a single agent
evaluate.py        # Greedy evaluation of a trained model
ablation.py        # Run all 4 agents + generate all paper figures
paper/report.md    # Mini research paper 
```

## Setup

```bash
uv sync
uv run python -c "import ale_py; ale_py.install_roms()"
```

## Quick Start

```bash
# Train all agents and generate figures (sequential, ~4h on GPU)
uv run python ablation.py --env PongNoFrameskip-v4 --steps 1000000

# Train a single agent
uv run python train.py --agent vanilla --env PongNoFrameskip-v4 --steps 1000000
uv run python train.py --agent double  --env PongNoFrameskip-v4
uv run python train.py --agent dueling --env PongNoFrameskip-v4
uv run python train.py --agent per     --env PongNoFrameskip-v4

# Evaluate a trained model (greedy, 30 episodes)
uv run python evaluate.py --agent per --env PongNoFrameskip-v4

# Re-generate plots from existing logs
uv run python ablation.py --plot-only --env PongNoFrameskip-v4
```

## Key Design Decisions

| Component | Where | Why |
|-----------|-------|-----|
| `BaseAgent._compute_loss()` | `base_agent.py` | Only method that differs across agents |
| `SumTree` | `prioritized_replay.py` | O(log N) priority ops |
| `weights` in all batch tuples | `replay_buffer.py` | Uniform buffer returns `weights=1`; PER returns IS weights — same interface |
| `_post_update()` hook | `base_agent.py` | PER overrides to update priorities; others are no-ops |

## What Each Agent Adds

| Agent | vs. previous | Key line(s) |
|-------|-------------|-------------|
| Vanilla DQN | baseline | `q_next = target_net(s').max()` |
| Double DQN | fixes overestimation | `a* = online(s').argmax(); q_next = target(s')[a*]` |
| Dueling DQN | better V(s) generalisation | swap `DQNNetwork` → `DuelingNetwork` |
| PER | focus on surprising transitions | swap `ReplayBuffer` → `PrioritizedReplayBuffer` + IS weights |

## Outputs

After running `ablation.py`:

```
results/
  logs/        # CSV per agent: step, episode, reward, avg100, epsilon
  models/      # Saved .pt checkpoints
  plots/
    learning_curves_*.png      # Smoothed reward vs steps, all agents
    final_performance_*.png    # Bar chart: mean ± std (last 100 eps)
    sample_efficiency_*.png    # Steps to threshold reward
```

## Paper

See `paper/report.md` 

## References

- Mnih et al. (2015) — Human-level control through DRL
- van Hasselt et al. (2016) — Double Q-learning
- Wang et al. (2016) — Dueling Networks
- Schaul et al. (2015) — Prioritized Experience Replay
- Hessel et al. (2018) — Rainbow
