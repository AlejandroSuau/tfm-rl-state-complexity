from rlpacman.envs.simple_pacman import SimplePacmanEnv, ObsConfig


def test_obs_shapes():
    modes = ["minimal","bool_power","power_time","coins_quadrants","image"]
    expected_dims = {"minimal":4,"bool_power":5,"power_time":6,"coins_quadrants":8}
    for m in modes:
        env = SimplePacmanEnv(ObsConfig(mode=m), seed=0)
        obs, _ = env.reset()
        if m == "image":
            assert obs.ndim == 3
        else:
            assert obs.shape == (expected_dims[m],)
        env.close()