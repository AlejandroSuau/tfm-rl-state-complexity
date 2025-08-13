from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class ObsConfig:
    mode: str = "minimal"  # ["minimal", "bool_power", "power_time", "coins_quadrants", "image"]
    grid_size: Tuple[int, int] = (15, 15)
    max_steps: int = 600
    power_duration: int = 40

class SimplePacmanEnv(gym.Env):
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
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.power_timer = 0
        self.player_pos = self._find_empty()
        self.ghost_pos = self._find_empty()
        self._place_coins_and_powers()
        obs = self._get_obs()
        info = self._info()
        return obs, info

    def step(self, action: int):
        self.steps += 1
        self._move_player(action)
        self._move_ghost()
        reward = 0.0

        # Recoger moneda
        if self.grid[self.player_pos] == 2:
            reward += self.rewards["coin"]
            self.grid[self.player_pos] = 0
            self.coins_remaining -= 1

        # Recoger comodín
        if self.grid[self.player_pos] == 3:
            reward += self.rewards["power"]
            self.grid[self.player_pos] = 0
            self.power_timer = self.cfg.power_duration
            self.power_pellets_remaining -= 1

        # Colisión con fantasma
        terminated = False
        if self.player_pos == self.ghost_pos:
            if self.power_timer > 0:
                reward += self.rewards["eat_ghost"]
                self.ghost_pos = self._find_empty()
            else:
                reward += self.rewards["death"]
                terminated = True

        if self.coins_remaining == 0:
            reward += self.rewards["clear"]
            terminated = True

        truncated = self.steps >= self.cfg.max_steps
        reward += self.rewards["step"]

        if self.power_timer > 0:
            self.power_timer -= 1

        obs = self._get_obs()
        info = self._info()
        return obs, float(reward), terminated, truncated, info

    # ---------- Definiciones ----------
    def _define_spaces(self):
        h, w = self.cfg.grid_size
        if self.cfg.mode == "image":
            self.observation_space = spaces.Box(low=0, high=1, shape=(h, w, 5), dtype=np.uint8)
        else:
            dim = 4  # px, py, gx, gy
            if self.cfg.mode in {"bool_power", "power_time"}:
                dim += 1
            if self.cfg.mode in {"power_time"}:
                dim += 1
            if self.cfg.mode in {"coins_quadrants"}:
                dim += 4
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # up, down, left, right, no-op

        # Recompensas (constantes para todos los modos)
        self.rewards = {
            "coin": 1.0,
            "power": 5.0,
            "eat_ghost": 10.0,
            "death": -20.0,
            "step": -0.01,
            "clear": 50.0,
        }

    def _build_layout(self):
        h, w = self.cfg.grid_size
        # 0 vacío, 1 muro, 2 moneda, 3 comodín
        self.grid = np.zeros((h, w), dtype=np.int8)
        # bordes como muros
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        # algunos muros internos sencillos
        for i in range(2, h - 2, 4):
            self.grid[i, 2:w - 2:4] = 1

    def _place_coins_and_powers(self):
        h, w = self.cfg.grid_size
        # limpia monedas/comodines previos
        self.grid[self.grid == 2] = 0
        self.grid[self.grid == 3] = 0
        # coloca monedas en celdas vacías (no muro)
        empties = list(zip(*np.where(self.grid == 0)))
        self.rng.shuffle(empties)
        n_coins = int(0.25 * len(empties))
        for pos in empties[:n_coins]:
            self.grid[pos] = 2
        self.coins_remaining = n_coins
        # coloca 2 comodines
        leftover = empties[n_coins:]
        self.power_pellets_remaining = 2 if len(leftover) >= 2 else len(leftover)
        for pos in leftover[: self.power_pellets_remaining]:
            self.grid[pos] = 3

    # ---------- Dinámica ----------
    def _find_empty(self):
        empties = np.argwhere(self.grid == 0)
        idx = self.rng.integers(len(empties))
        return tuple(empties[idx])

    def _move_player(self, action: int):
        dyx = [(-1,0), (1,0), (0,-1), (0,1), (0,0)]
        dy, dx = dyx[action]
        ny = int(np.clip(self.player_pos[0] + dy, 0, self.grid.shape[0] - 1))
        nx = int(np.clip(self.player_pos[1] + dx, 0, self.grid.shape[1] - 1))
        if self.grid[ny, nx] != 1:  # no atraviesa muros
            self.player_pos = (ny, nx)

    def _move_ghost(self):
        # política simple: 70% moverse hacia el jugador, 30% aleatorio
        if self.rng.random() < 0.7:
            dy = np.sign(self.player_pos[0] - self.ghost_pos[0])
            dx = np.sign(self.player_pos[1] - self.ghost_pos[1])
            candidates = [(dy,0), (0,dx)]
        else:
            candidates = [(-1,0), (1,0), (0,-1), (0,1)]
        self.rng.shuffle(candidates)
        for dy, dx in candidates:
            ny = int(np.clip(self.ghost_pos[0] + dy, 0, self.grid.shape[0] - 1))
            nx = int(np.clip(self.ghost_pos[1] + dx, 0, self.grid.shape[1] - 1))
            if self.grid[ny, nx] != 1:
                self.ghost_pos = (ny, nx)
                break

    # ---------- Observaciones ----------
    def _norm_pos(self, pos):
        h, w = self.cfg.grid_size
        return [pos[0] / (h - 1), pos[1] / (w - 1)]

    def _coins_by_quadrant(self):
        h, w = self.cfg.grid_size
        midy, midx = h // 2, w // 2
        q1 = np.sum(self.grid[:midy, :midx] == 2)
        q2 = np.sum(self.grid[:midy, midx:] == 2)
        q3 = np.sum(self.grid[midy:, :midx] == 2)
        q4 = np.sum(self.grid[midy:, midx:] == 2)
        total = max(1, q1 + q2 + q3 + q4)
        return [q1/total, q2/total, q3/total, q4/total]

    def _render_layers(self):
        # 5 canales: muro, moneda, power, jugador, fantasma
        h, w = self.cfg.grid_size
        layers = np.zeros((h, w, 5), dtype=np.uint8)
        layers[:, :, 0] = (self.grid == 1).astype(np.uint8)
        layers[:, :, 1] = (self.grid == 2).astype(np.uint8)
        layers[:, :, 2] = (self.grid == 3).astype(np.uint8)
        layers[self.player_pos][3] = 1
        layers[self.ghost_pos][4] = 1
        return layers

    def _get_obs(self):
        if self.cfg.mode == "image":
            return self._render_layers()
        v = []
        v.extend(self._norm_pos(self.player_pos))
        v.extend(self._norm_pos(self.ghost_pos))
        if self.cfg.mode in {"bool_power", "power_time"}:
            v.append(float(self.power_pellets_remaining > 0))
        if self.cfg.mode in {"power_time"}:
            v.append(self.power_timer / max(1, self.cfg.power_duration))
        if self.cfg.mode in {"coins_quadrants"}:
            v.extend(self._coins_by_quadrant())
        return np.asarray(v, dtype=np.float32)

    def _info(self):
        return {
            "steps": self.steps,
            "coins_remaining": int(self.coins_remaining),
            "power_timer": int(self.power_timer),
        }