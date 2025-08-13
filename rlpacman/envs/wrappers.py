from __future__ import annotations
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np

class NormalizeObs(gym.ObservationWrapper):
    """Normaliza observaciones Box a [0,1] si los límites no están ya en ese rango."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.low = self.observation_space.low
        self.high = self.observation_space.high

    def observation(self, obs):
        obs = obs.astype(np.float32)
        denom = (self.high - self.low)
        denom[denom == 0] = 1
        return (obs - self.low) / denom