from src.agents.vanilla_dqn import VanillaDQN
from src.agents.double_dqn import DoubleDQN
from src.agents.dueling_dqn import DuelingDQN
from src.agents.per_dqn import PERDQN

AGENTS = {
    "vanilla": VanillaDQN,
    "double": DoubleDQN,
    "dueling": DuelingDQN,
    "per": PERDQN,
}
