from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from .constants import COIN_POWER_TEMPLATE, WALL_TEMPLATE, PLAYER_START, GHOST_START, EMPTY, WALL, COIN, POWER, ACTIONS, DEFAULT_REWARDS

@dataclass
class ObsConfig:
    """Configuration of environment observations."""
    mode: str = "minimal"  # ["minimal", "bool_power", "power_time", "coins_quadrants", "image"]
    grid_size: Tuple[int, int] = (20, 17)
    max_steps: int = 1000
    power_duration: int = 40 # Steps that power pellet lasts
    ghost_respawn_delay: int = 15 # Steps to respawn ghost after being eaten

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

    def _build_layout(self):
        """Initialize grid with static walls, coins and power pellets."""
        # Forzamos que el grid_size concuerde con la plantilla
        h, w = COIN_POWER_TEMPLATE.shape
        self.cfg.grid_size = (h, w)

        self.grid = np.zeros((h, w), dtype=np.int8)

        for y in range(h):
            for x in range(w):
                if WALL_TEMPLATE[y, x] == 1:
                    self.grid[y, x] = WALL
                else:
                    val = COIN_POWER_TEMPLATE[y, x]
                    if val == 1:
                        self.grid[y, x] = COIN
                    elif val == 2:
                        self.grid[y, x] = POWER
                    else:
                        self.grid[y, x] = EMPTY

        # Keep a copy of the initial grid for reset
        self._initial_grid = self.grid.copy()

        # Initial statistics
        self.coins_total = int(np.sum(self.grid == COIN))
        self.coins_remaining = self.coins_total
        self.power_pellets_remaining = int(np.sum(self.grid == POWER))

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options=None):
        """Reset the environment state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.power_timer = 0
        self.powers_picked = 0
        self.ghosts_eaten = 0

        # Restaurar layout inicial estático (2)
        self.grid = self._initial_grid.copy()
        self.coins_remaining = int(np.sum(self.grid == COIN))
        self.coins_total = self.coins_remaining
        self.power_pellets_remaining = int(np.sum(self.grid == POWER))

        self.player_pos = PLAYER_START

        self.ghost_pos = GHOST_START
        self.ghost_dir: Tuple[int, int] | None = None
        self.ghost_alive: bool = True
        self.ghost_respawn_timer: int = 0

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
        action = int(np.clip(action, 0, self.action_space.n - 1))

        self.steps += 1
        self._move_player(action)
        
        # --- Ghost movement with speed reduction in power time ---
        self._update_ghost_state_before_move()
        if self.ghost_alive and self._ghost_should_move():
            self._move_ghost()

        reward = 0.0
        terminated = False

        # --- Collect coin ---
        if self.grid[self.player_pos] == COIN:
            reward += self.rewards["coin"]
            self.grid[self.player_pos] = EMPTY
            self.coins_remaining -= 1

            threshold = max(4, int(0.10 * self.coins_total))
            if self.coins_remaining <= threshold:
                reward += self.rewards.get("last_coin_bonus", 0.0)

        # --- Collect power pellet ---
        if self.grid[self.player_pos] == POWER:
            reward += self.rewards["power"]
            self.grid[self.player_pos] = EMPTY
            self.power_timer = int(self.cfg.power_duration)
            self.power_pellets_remaining -= 1
            self.powers_picked += 1

        # --- Collision with ghost (only if it is alive) ---
        if self.ghost_alive and self.player_pos == self.ghost_pos:
            if self.power_timer > 0:  # Ghost is vulnerable
                self.ghosts_eaten += 1
                reward += self.rewards["eat_ghost"]
                # Ghost dies and spawns after a delay
                self.ghost_alive = False
                self.ghost_respawn_timer = int(self.cfg.ghost_respawn_delay)
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
            reward += self.rewards.get("power_tick", 0.0)
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
            if self.cfg.mode in {"bool_power", "power_time", "coins_quadrants"}:
                dim += 1 # bool power actie
            
            if self.cfg.mode in {"power_time", "coins_quadrants"}:
                dim += 1 # power time (left time / duration)
            
            if self.cfg.mode == "coins_quadrants":
                dim += 4 # 4 dim for count of coins in each quadrant
                dim += 4 # 4 dim for pacman quadrant
            
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

        self.action_space = spaces.Discrete(4)  # up, down, left, right

        # Reward mapping
        self.rewards = DEFAULT_REWARDS

    # ---------- Dynamics ----------

    def _valid_directions(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Devuelve la lista de direcciones (dy, dx) posibles desde pos
        que no chocan con un muro.
        """
        y, x = pos
        h, w = self.grid.shape
        dirs = []
        for dy, dx in ACTIONS.values():  # ACTIONS: {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and self.grid[ny, nx] != WALL:
                dirs.append((dy, dx))
        return dirs

    def _ghost_should_move(self) -> bool:
        """Return True if the ghost should move this step, taking into account power-time speed."""
        if self.power_timer <= 0:
            return True
        
        # Reduce ghost speed by half during power time
        return (self.steps % 2) == 0

    def _update_ghost_state_before_move(self):
        """Handle ghost respawn timers and dead state (5)."""
        if not self.ghost_alive:
            if self.ghost_respawn_timer > 0:
                self.ghost_respawn_timer -= 1
            if self.ghost_respawn_timer == 0:
                # Respawn fixed pos
                self.ghost_pos = GHOST_START
                self.ghost_dir = None
                self.ghost_alive = True

    def _move_player(self, action: int):
        """
        Mueve a Pacman según la acción del agente:

        - Si la acción apunta a una celda libre, se mueve allí.
        - Si apunta a un muro o fuera del grid, se queda en la misma posición.
        """
        y, x = self.player_pos
        dy, dx = ACTIONS[action]

        ny, nx = y + dy, x + dx

        h, w = self.grid.shape
        if (0 <= ny < h
            and 0 <= nx < w
            and self.grid[ny, nx] != WALL):
            # Movimiento válido
            self.player_pos = (ny, nx)


    def _move_ghost(self):
        """
        Fantasma perseguidor:
        - Siempre intenta acercarse a Pacman (distancia Manhattan).
        - Evita giros de 180º salvo que no haya otra opción (evita bucles arriba/abajo).
        - Nunca atraviesa muros.
        """
        y, x = self.ghost_pos

        free_dirs = self._valid_directions(self.ghost_pos)
        if not free_dirs:
            return  # no hay movimiento posible (muy raro)

        # Intentamos evitar el giro de 180º si hay más de una opción
        if self.ghost_dir is not None and len(free_dirs) > 1:
            rev = (-self.ghost_dir[0], -self.ghost_dir[1])
            non_reverse = [d for d in free_dirs if d != rev]
            if non_reverse:
                free_dirs = non_reverse  # evitamos 180º si podemos

        py, px = self.player_pos

        # Elegimos las direcciones que minimizan la distancia Manhattan a Pacman
        def dist_after_move(d: Tuple[int, int]) -> int:
            ny, nx = y + d[0], x + d[1]
            return abs(py - ny) + abs(px - nx)

        dists = [dist_after_move(d) for d in free_dirs]
        min_dist = min(dists)

        # Ruptura de loops: 10% random alternativo
        if self.rng.random() < 0.1 and len(free_dirs) > 1:
            # evitar reverse si es posible
            if self.ghost_dir is not None:
                rev = (-self.ghost_dir[0], -self.ghost_dir[1])
                non_reverse = [d for d in free_dirs if d != rev]
                free_dirs = non_reverse or free_dirs
            dy, dx = free_dirs[self.rng.integers(len(free_dirs))]
        else:
            # política greedy perseguidora
            best_dirs = [d for d, dd in zip(free_dirs, dists) if dd == min_dist]
            dy, dx = best_dirs[self.rng.integers(len(best_dirs))]

        self.ghost_dir = (dy, dx)
        self.ghost_pos = (y + dy, x + dx)


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

     # ---------- Render ----------
    def render(self):
        """
        Renderiza el entorno según self.render_mode.
        - 'ansi'      -> devuelve un string con el mapa en ASCII
        - 'rgb_array' -> devuelve un array (H, W, 3) uint8 para usar con imshow
        """
        if self.render_mode is None:
            raise NotImplementedError(
                "SimplePacmanEnv: define render_mode='ansi' o 'rgb_array' al crear el entorno."
            )

        if self.render_mode == "ansi":
            # Construimos una cuadricula de caracteres
            h, w = self.grid.shape
            lines = []
            for y in range(h):
                row_chars = []
                for x in range(w):
                    pos = (y, x)
                    if pos == self.player_pos:
                        ch = "C"  # Pacman
                    elif pos == self.ghost_pos:
                        ch = "G"  # Fantasma
                    else:
                        cell = self.grid[y, x]
                        if cell == WALL:
                            ch = "#"
                        elif cell == COIN:
                            ch = "."
                        elif cell == POWER:
                            ch = "P"
                        else:
                            ch = " "
                    row_chars.append(ch)
                lines.append("".join(row_chars))
            return "\n".join(lines)

        if self.render_mode == "rgb_array":
            # Usamos las capas ya definidas para crear una imagen RGB sencilla
            layers = self._render_layers()  # (H, W, 5)
            h, w, _ = layers.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            
            # === PALETA ALTO CONTRASTE ===
            WALL_COLOR   = (100, 100, 100)   # Gris oscuro
            COIN_COLOR   = (255, 215, 0)     # Amarillo oro brillante
            POWER_COLOR  = (0, 200, 255)     # Cian brillante
            PLAYER_COLOR = (0, 255, 100)     # Verde lima
            GHOST_COLOR  = (255, 60, 60)     # Rojo intenso
            # ==============================

            img[layers[:, :, 0] == 1] = WALL_COLOR
            img[layers[:, :, 1] == 1] = COIN_COLOR
            img[layers[:, :, 2] == 1] = POWER_COLOR
            img[layers[:, :, 3] == 1] = PLAYER_COLOR
            img[layers[:, :, 4] == 1] = GHOST_COLOR

            return img

        # Si se pide un modo que no soportamos
        raise NotImplementedError(f"Render mode '{self.render_mode}' no soportado.")

    def _get_obs(self):
        """Build observation according to the configured mode."""
        if self.cfg.mode == "image":
            return self._render_layers()

        v: List[float] = []
        v.extend(self._norm_pos(self.player_pos))
        v.extend(self._norm_pos(self.ghost_pos))

        if self.cfg.mode in {"bool_power", "power_time", "coins_quadrants"}:
            v.append(float(self.power_timer > 0))
        
        if self.cfg.mode in {"power_time", "coins_quadrants"}:
            v.append(self.power_timer / max(1, self.cfg.power_duration))
        
        if self.cfg.mode == "coins_quadrants":
            v.extend(self._coins_by_quadrant())
            v.extend(self._pacman_quandrant())

        return np.asarray(v, dtype=np.float32)

    def _pacman_quandrant(self) -> List[float]:
        """One-hot quadrant where player is"""
        h, w = self.cfg.grid_size
        midy, midx = h // 2, w // 2
        y, x = self.player_pos
        if y < midy and x < midx:
            return [1.0, 0.0, 0.0, 0.0] # Q1
        elif y < midy and x >= midx:
            return [0.0, 1.0, 0.0, 0.0] # Q2
        elif y >= midy and x < midx:
            return [0.0, 0.0, 1.0, 0.0] # Q3
        else:
            return [0.0, 0.0, 0.0, 1.0] # Q4

    def _info(self) -> Dict[str, int]:
        """Extra diagnostic info returned with each step."""
        return {
            "steps": self.steps,
            "powers_picked": int(getattr(self, "powers_picked", 0)),
            "ghosts_eaten": int(getattr(self, "ghosts_eaten", 0)),
            "coins_remaining": int(self.coins_remaining),
            "coins_total": int(getattr(self, "coins_total", 0)), 
            "power_timer": int(self.power_timer),
        }