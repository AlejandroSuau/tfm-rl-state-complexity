from typing import Tuple, Dict

# Grid cell types
EMPTY: int = 0
WALL: int = 1
COIN: int = 2
POWER: int = 3

# Action indexing: up, down, left, right, no-op
ACTIONS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0), # up
    1: (1, 0), # down
    2: (0, -1), # left
    3: (0, 1), # right
    4: (0, 0), # no-op
}

# Default rewards
#DEFAULT_REWARDS: Dict[str, float] = {
#    "coin": 1.0,
#    "power": 5.0,
#    "eat_ghost": 10.0,
#    "death": -20.0,
#    "step": -0.001,
#    "clear": 50.0,
#}

# DEFAULT_REWARDS: Dict[str, float] = {
#     "coin": 2.0,
#     "power": 5.0,
#     "eat_ghost": 10.0,
#     "death": -30.0,   # antes -10
#     "step": 0.0,      # mantenemos 0 para no incentivar acabar pronto
#     "clear": 50.0,
# }

DEFAULT_REWARDS = {
  "coin": 2.0,
  "power": 10.0,
  "eat_ghost": 20.0,
  "death": -30.0,
  "step": 0.0,
  "clear": 60.0,
  "power_tick": 0.05,
}
