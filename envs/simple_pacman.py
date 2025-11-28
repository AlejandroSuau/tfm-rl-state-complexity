from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from .constants import COIN_POWER_TEMPLATE, WALL_TEMPLATE, PLAYER_START, GHOST_START, EMPTY, WALL, COIN, POWER, ACTIONS, DEFAULT_REWARDS
from collections import deque

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
        self.last_player_pos = self.player_pos

        self.ghost_pos = GHOST_START
        self.ghost_dir: Tuple[int, int] | None = None
        self.ghost_alive: bool = True
        self.ghost_respawn_timer: int = 0

        obs = self._get_obs()
        info = self._info()
        return obs, info

    def step(self, action: int):
        """Perform one environment step."""
        action = int(np.asarray(action).squeeze())
        action = int(np.clip(action, 0, self.action_space.n - 1))
        
        self.steps += 1
        prev_player_pos = self.player_pos

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
                self.ghost_pos = GHOST_START
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
        if self.player_pos == prev_player_pos:
            reward += self.rewards.get("idle", 0.0)

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

    def _is_ghost_intersection(self, pos: Tuple[int, int], current_dir: Tuple[int, int] | None) -> bool:
        """
        Devuelve True si el fantasma está en una intersección donde puede/conviene
        reconsiderar su dirección.

        Regla:
        - Si hay 3 o más salidas libres -> intersección.
        - Si hay exactamente 2 salidas:
            * Si son opuestas (pasillo recto)      -> NO intersección.
            * Si no son opuestas (curva/esquina)   -> SÍ intersección.
        """
        free_dirs = self._valid_directions(pos)

        if current_dir is None:
            # Justo respawn o sin dirección previa: lo tratamos como intersección
            return True

        if len(free_dirs) >= 3:
            return True

        if len(free_dirs) == 2:
            d1, d2 = free_dirs
            # ¿Son opuestas?
            if d1[0] == -d2[0] and d1[1] == -d2[1]:
                return False  # pasillo recto
            else:
                return True   # esquina

        return False

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
        Movimiento del fantasma:

        - Solo recalcula dirección cuando:
            * no tiene dirección previa (inicio / respawn), o
            * está en una intersección, o
            * la dirección actual está bloqueada.
        - Si power_time está activo (power_timer > 0):
            * la nueva dirección se elige SIEMPRE aleatoria entre las válidas.
        - Si NO está activo:
            * 90%: dirección que minimiza la distancia Manhattan al jugador.
            * 10%: dirección aleatoria entre las válidas.
        - Entre intersecciones, si puede seguir recto, mantiene la dirección.
        """
        y, x = self.ghost_pos

        free_dirs = self._valid_directions(self.ghost_pos)
        h, w = self.grid.shape

        # 1) ¿Puede seguir en la dirección actual?
        can_keep_dir = False
        if self.ghost_dir is not None:
            gy, gx = self.ghost_dir
            ny, nx = y + gy, x + gx
            if (
                0 <= ny < h
                and 0 <= nx < w
                and self.grid[ny, nx] != WALL
            ):
                can_keep_dir = True

        # 2) ¿Está en una intersección?
        is_intersection = self._is_ghost_intersection(self.ghost_pos, self.ghost_dir)

        # 3) ¿Necesita nueva dirección?
        need_new_dir = (
            self.ghost_dir is None  # inicio / respawn
            or not can_keep_dir     # muro delante
            or is_intersection      # cruce / esquina
        )

        if need_new_dir:
            # Partimos de las direcciones libres
            cand_dirs = list(free_dirs)

            # Evitar giro de 180º si hay más de una opción
            if self.ghost_dir is not None and len(cand_dirs) > 1:
                rev = (-self.ghost_dir[0], -self.ghost_dir[1])
                non_reverse = [d for d in cand_dirs if d != rev]
                if non_reverse:
                    cand_dirs = non_reverse

            # --- Caso power activo: siempre aleatorio ---
            if self.power_timer > 0:
                dy, dx = cand_dirs[self.rng.integers(len(cand_dirs))]
            else:
                # --- Caso normal: 90% seguir el camino más corto real, 10% aleatorio ---
                py, px = self.player_pos

                bfs_dir = self._bfs_next_step_towards((y, x), (py, px))

                # Aseguramos que la dirección BFS es candidata
                if bfs_dir is not None and bfs_dir in cand_dirs:
                    best_dirs = [bfs_dir]
                else:
                    # Si BFS falla (sin camino), caemos a greedy Manhattan sobre cand_dirs
                    def dist_after_move(d: Tuple[int, int]) -> int:
                        ny, nx = y + d[0], x + d[1]
                        return abs(py - ny) + abs(px - nx)
                    dists = [dist_after_move(d) for d in cand_dirs]
                    min_dist = min(dists)
                    best_dirs = [d for d, dd in zip(cand_dirs, dists) if dd == min_dist]

                if self.rng.random() < 0.9:
                    base = best_dirs
                else:
                    base = cand_dirs

                dy, dx = base[self.rng.integers(len(base))]

            self.ghost_dir = (dy, dx)

        # 4) Avanza en la dirección actual (ya sea la antigua o la recalculada)
        dy, dx = self.ghost_dir
        self.ghost_pos = (y + dy, x + dx)


    def _bfs_next_step_towards(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, int] | None:
        """
        Devuelve la primera dirección (dy, dx) del camino más corto desde `start` hasta `goal`
        respetando los muros. Si no hay camino, devuelve None.
        """
        if start == goal:
            return None

        h, w = self.grid.shape
        visited = [[False] * w for _ in range(h)]
        parent: dict[Tuple[int, int], Tuple[int, int]] = {}

        q = deque()
        q.append(start)
        visited[start[0]][start[1]] = True

        while q:
            cy, cx = q.popleft()
            for dy, dx in ACTIONS.values():
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if visited[ny][nx]:
                    continue
                if self.grid[ny, nx] == WALL:
                    continue

                visited[ny][nx] = True
                parent[(ny, nx)] = (cy, cx)

                if (ny, nx) == goal:
                    # Reconstruimos el camino al revés
                    path = [(ny, nx)]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    path.reverse()
                    first = path[1]
                    dy0 = first[0] - start[0]
                    dx0 = first[1] - start[1]
                    return (dy0, dx0)

                q.append((ny, nx))

        # No hay camino
        return None



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
        - 'rgb_array' -> devuelve un array (H, W, 3) uint8 para usar con imshow
        """
        if self.render_mode is None:
            raise NotImplementedError(
                "SimplePacmanEnv: define render_mode='rgb_array' al crear el entorno."
            )

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
            GHOST_NORMAL   = (255, 60, 60)     # Rojo intenso (persiguiendo)
            GHOST_SCARED   = (255, 140, 255)   # Rosa brillante (random por power pellet)
            GHOST_DEAD     = (180, 180, 180)   # Gris claro (respawn)
            # ==============================

            img[layers[:, :, 0] == 1] = WALL_COLOR
            img[layers[:, :, 1] == 1] = COIN_COLOR
            img[layers[:, :, 2] == 1] = POWER_COLOR
            img[layers[:, :, 3] == 1] = PLAYER_COLOR
            
            ghost_mask = (layers[:, :, 4] == 1)
            if self.ghost_respawn_timer > 0:
                img[ghost_mask] = GHOST_DEAD
            elif self.power_timer > 0:
                img[ghost_mask] = GHOST_SCARED
            else:
                img[ghost_mask] = GHOST_NORMAL

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