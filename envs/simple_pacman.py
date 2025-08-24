from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from .constants import EMPTY, WALL, COIN, POWER, ACTIONS, DEFAULT_REWARDS

@dataclass
class ObsConfig:
    """Configuration of environment observations."""
    mode: str = "minimal"  # ["minimal", "bool_power", "power_time", "coins_quadrants", "image"]
    grid_size: Tuple[int, int] = (15, 15)
    max_steps: int = 600
    power_duration: int = 40

class SimplePacmanEnv(gym.Env):
    """A simplified Pac-Man style environment for RL experiments."""

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 15}

    def __init__(self, obs_config: ObsConfig | None = None, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = obs_config or ObsConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self._build_layout()
        self._define_spaces()
        self.reset()

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options=None):
        """Reset the environment state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.power_timer = 0

        # Place player and ghost in random empty cells
        self.player_pos = self._find_empty()
        self.ghost_pos = self._find_empty()
        while self.ghost_pos == self.player_pos:
            self.ghost_pos = self._find_empty()

        # Place collectibles
        self._place_coins_and_powers()

        obs = self._get_obs()
        info = self._info()
        return obs, info

    def step(self, action: int):
        """Perform one environment step."""
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(np.asarray(action).squeeze()[0]) if np.asarray(action).ndim > 0 else int(np.asarray(action).item())
        elif hasattr(action, "item"):
            action = int(action.item())
        else:
            action = int(action)
        action = int(np.clip(action, 0, 4))

        self.steps += 1
        self._move_player(action)
        self._move_ghost()
        reward = 0.0
        terminated = False

        # --- Collect coin ---
        if self.grid[self.player_pos] == COIN:
            reward += self.rewards["coin"]
            self.grid[self.player_pos] = EMPTY
            self.coins_remaining -= 1

        # --- Collect power pellet ---
        if self.grid[self.player_pos] == POWER:
            reward += self.rewards["power"]
            self.grid[self.player_pos] = EMPTY
            self.power_timer = int(self.cfg.power_duration)
            self.power_pellets_remaining -= 1

        # --- Collision with ghost ---
        if self.player_pos == self.ghost_pos:
            if self.power_timer > 0:  # Ghost is vulnerable
                reward += self.rewards["eat_ghost"]
                self.ghost_pos = self._find_empty()
            else:  # Player dies
                reward += self.rewards["death"]
                terminated = True

        # --- Win condition: all coins collected ---
        if self.coins_remaining == 0:
            reward += self.rewards["clear"]
            terminated = True

        # --- Step penalty ---
        reward += self.rewards["step"]

        # --- Update timers ---
        if self.power_timer > 0:
            self.power_timer -= 1

        truncated = self.steps >= self.cfg.max_steps
        obs = self._get_obs()
        info = self._info()
        return obs, float(reward), terminated, truncated, info

    # ---------- Definitions ----------
    def _define_spaces(self):
        """Define action and observation spaces."""
        h, w = self.cfg.grid_size

        if self.cfg.mode == "image":
            # Observation as multi-channel grid
            self.observation_space = spaces.Box(low=0, high=1, shape=(h, w, 5), dtype=np.uint8)
        else:
            dim = 4  # player(x,y), ghost(x,y)
            if self.cfg.mode in {"bool_power", "power_time"}:
                dim += 1
            if self.cfg.mode == "power_time":
                dim += 1
            if self.cfg.mode == "coins_quadrants":
                dim += 4
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

        self.action_space = spaces.Discrete(5)  # up, down, left, right, no-op

        # Reward mapping
        self.rewards = DEFAULT_REWARDS

    def _build_layout(self):
        """Initialize grid with walls and empty cells."""
        h, w = self.cfg.grid_size
        self.grid = np.zeros((h, w), dtype=np.int8)

        # Borders as walls
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        # Internal simple walls
        for i in range(2, h - 2, 4):
            self.grid[i, 2:w - 2:4] = WALL

    def _place_coins_and_powers(self):
        """Randomly distribute coins and power pellets."""
        h, w = self.cfg.grid_size
        self.grid[self.grid == COIN] = EMPTY
        self.grid[self.grid == POWER] = EMPTY

        # Place coins
        empties = list(zip(*np.where(self.grid == EMPTY)))
        self.rng.shuffle(empties)
        n_coins = int(0.40 * len(empties))
        for pos in empties[:n_coins]:
            self.grid[pos] = COIN
        self.coins_remaining = n_coins
        self.coins_total = n_coins

        # Place power pellets
        leftover = empties[n_coins:]
        self.power_pellets_remaining = min(2, len(leftover))
        for pos in leftover[:self.power_pellets_remaining]:
            self.grid[pos] = POWER

    # ---------- Dynamics ----------
    def _find_empty(self) -> Tuple[int, int]:
        """Return a random empty position in the grid."""
        empties = np.argwhere(self.grid == EMPTY)
        idx = self.rng.integers(len(empties))
        return tuple(empties[idx])

    def _move_player(self, action: int):
        """Move player according to chosen action."""
        dy, dx = ACTIONS[action]
        ny = int(np.clip(self.player_pos[0] + dy, 0, self.grid.shape[0] - 1))
        nx = int(np.clip(self.player_pos[1] + dx, 0, self.grid.shape[1] - 1))
        if self.grid[ny, nx] != WALL:
            self.player_pos = (ny, nx)

    def _move_ghost(self):
        """Simple ghost policy: 50% chase, 50% random move."""
        if self.rng.random() < 0.5:
            dy = np.sign(self.player_pos[0] - self.ghost_pos[0])
            dx = np.sign(self.player_pos[1] - self.ghost_pos[1])
            candidates = [(dy, 0), (0, dx)]
        else:
            candidates = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.rng.shuffle(candidates)
        for dy, dx in candidates:
            ny = int(np.clip(self.ghost_pos[0] + dy, 0, self.grid.shape[0] - 1))
            nx = int(np.clip(self.ghost_pos[1] + dx, 0, self.grid.shape[1] - 1))
            if self.grid[ny, nx] != WALL:
                self.ghost_pos = (ny, nx)
                break

    # ---------- Observations ----------
    def _norm_pos(self, pos: Tuple[int, int]) -> List[float]:
        """Normalize position into [0,1] range."""
        h, w = self.cfg.grid_size
        return [pos[0] / (h - 1), pos[1] / (w - 1)]

    def _coins_by_quadrant(self) -> List[float]:
        """Return normalized coin distribution across 4 quadrants."""
        h, w = self.cfg.grid_size
        midy, midx = h // 2, w // 2
        q1 = np.sum(self.grid[:midy, :midx] == COIN)
        q2 = np.sum(self.grid[:midy, midx:] == COIN)
        q3 = np.sum(self.grid[midy:, :midx] == COIN)
        q4 = np.sum(self.grid[midy:, midx:] == COIN)
        total = max(1, q1 + q2 + q3 + q4)
        return [q1 / total, q2 / total, q3 / total, q4 / total]

    def _render_layers(self):
        """Return a multi-channel representation of the grid (walls, coins, powers, player, ghost)."""
        h, w = self.cfg.grid_size
        layers = np.zeros((h, w, 5), dtype=np.uint8)
        layers[:, :, 0] = (self.grid == WALL).astype(np.uint8)
        layers[:, :, 1] = (self.grid == COIN).astype(np.uint8)
        layers[:, :, 2] = (self.grid == POWER).astype(np.uint8)
        layers[self.player_pos][3] = 1
        layers[self.ghost_pos][4] = 1
        return layers
    
    def _get_obs(self):
        """Build observation according to the configured mode."""
        if self.cfg.mode == "image":
            return self._render_layers()

        v: List[float] = []
        v.extend(self._norm_pos(self.player_pos))
        v.extend(self._norm_pos(self.ghost_pos))

        if self.cfg.mode in {"bool_power", "power_time"}:
            v.append(float(self.power_timer > 0))
        if self.cfg.mode == "power_time":
            v.append(self.power_timer / max(1, self.cfg.power_duration))
        if self.cfg.mode == "coins_quadrants":
            v.extend(self._coins_by_quadrant())

        return np.asarray(v, dtype=np.float32)

    def _info(self) -> Dict[str, int]:
        """Extra diagnostic info returned with each step."""
        return {
            "steps": self.steps,
            "coins_remaining": int(self.coins_remaining),
            "coins_total": int(getattr(self, "coins_total", 0)), 
            "power_timer": int(self.power_timer),
        }