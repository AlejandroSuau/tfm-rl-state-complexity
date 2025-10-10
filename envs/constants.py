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

DEFAULT_REWARDS = {
    "coin": 1.0,
    "power": 10.0,
    "eat_ghost": 5.0,
    "clear": 1000.0,
    "death": -100.0,
    "step": -0.01,
    "power_tick": 0.0,
    "coin_power_bonus": 0.0,
    "last_coin_bonus": 0.0,
}

