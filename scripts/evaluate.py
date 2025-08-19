import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from rlpacman.envs.simple_pacman import SimplePacmanEnv, ObsConfig
from rlpacman.utils.metrics import auc_return, time_to_threshold

ALGOS = {"ppo": PPO, "dqn": DQN}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS.keys(), required=True)
    parser.add_argument("--obs-mode", choices=["minimal","bool_power","power_time","coins_quadrants","image"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=30.0, help="Umbral de retorno para time-to-threshold")
    args = parser.parse_args()

    env = SimplePacmanEnv(ObsConfig(mode=args.obs_mode), seed=42)
    model = ALGOS[args.algo].load(args.model)

    ep_returns, successes = [], []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        ep_ret = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, trunc, info = env.step(int(action))
            ep_ret += float(rew)
        ep_returns.append(ep_ret)
        successes.append(env.coins_remaining == 0)

    df = pd.DataFrame({"return": ep_returns, "success": successes})
    print("Resumen evaluación:\n", df.describe())

    steps = np.arange(1, len(ep_returns) + 1, dtype=float)
    # AUC del retorno por episodio (aprox - aquí no usamos steps_cum, sino episodios como x)
    auc = float(np.trapz(ep_returns, steps))
    ttt = time_to_threshold(
        pd.DataFrame({"reward": ep_returns, "length": np.ones_like(ep_returns), "steps_cum": steps}),
        args.threshold,
        window_episodes=5,
    )
    print({"AUC_return": auc, "time_to_threshold": ttt})

    df.to_csv("experiments/eval_summary.csv", index=False)

if __name__ == "__main__":
    main()
