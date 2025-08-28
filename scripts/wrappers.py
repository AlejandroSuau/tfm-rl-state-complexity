from __future__ import annotations

import gymnasium as gym
import numpy as np


class NormalizeObs(gym.ObservationWrapper):
    """
    Normalize continuous Box observations into the [0, 1] range,
    unless they are already bounded that way.

    Notes:
        - Works element-wise based on observation_space.low/high.
        - Avoids division by zero if high == low for a dimension.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.low = self.observation_space.low
        self.high = self.observation_space.high

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        denom = self.high - self.low
        denom[denom == 0] = 1  # avoid division by zero
        return (obs - self.low) / denom
