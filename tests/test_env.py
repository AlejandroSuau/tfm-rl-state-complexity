from rlpacman.envs.simple_pacman import SimplePacmanEnv, ObsConfig


def test_reset_and_step_minimal():
    env = SimplePacmanEnv(ObsConfig(mode="minimal"), seed=0)
    obs, info = env.reset()
    assert env.action_space.n == 5
    assert env.observation_space.shape == (4,)
    # Step no-op
    obs, r, d, t, info = env.step(4)
    assert isinstance(r, float)
    assert obs.shape == (4,)
    env.close()