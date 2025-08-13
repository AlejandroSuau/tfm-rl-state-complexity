import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from rlpacman.envs.simple_pacman import SimplePacmanEnv, ObsConfig

ALGOS = {"ppo": PPO, "dqn": DQN}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS.keys(), required=True)
    parser.add_argument("--obs-mode", choices=["minimal","bool_power","power_time","coins_quadrants","image"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = SimplePacmanEnv(ObsConfig(mode=args.obs_mode), seed=42)
    model = ALGOS[args.algo].load(args.model)

    returns, successes = [], []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        ep_ret = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, trunc, info = env.step(int(action))
            ep_ret += float(rew)
        returns.append(ep_ret)
        successes.append(env.coins_remaining == 0)

    df = pd.DataFrame({"return": returns, "success": successes})
    print(df.describe())
    df.to_csv("experiments/eval_summary.csv", index=False)


if __name__ == "__main__":
    main()