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
    # Moneda barata: farmear muchas monedas ya no gana a terminar
    "coin": 1.5,
    # Power y fantasma valen, pero no dominan
    "power": 3.0,
    "eat_ghost": 5.0,
    # Terminar tiene que ganar SIEMPRE
    "clear": 250.0,
    # Morir duele más para que no compense el “suicidio rápido”
    "death": -60.0,
    # Penalización por paso para empujar a terminar rápido (se normaliza con VecNorm)
    "step": -0.01,
    # Nada de goteo por estar en power
    "power_tick": 0.0,
    # Un pequeño empujón extra si pillas moneda en power (pero pequeño)
    "coin_power_bonus": 0.25,
}

