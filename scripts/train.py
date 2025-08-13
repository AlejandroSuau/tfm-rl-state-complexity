import argparse
import os
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from rlpacman.envs.simple_pacman import SimplePacmanEnv, ObsConfig
from rlpacman.utils.callbacks import build_callbacks
from rlpacman.utils.seeding import set_global_seed

ALGOS = {"ppo": PPO, "dqn": DQN}

def make_env(obs_mode: str, seed: int):
    env = SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=seed)
    return Monitor(env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS.keys(), default="ppo")
    parser.add_argument("--obs-mode", choices=["minimal","bool_power","power_time","coins_quadrants","image"], default="minimal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--logdir", type=str, default="experiments/logs")
    parser.add_argument("--savedir", type=str, default="experiments/runs")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.savedir, exist_ok=True)

    set_global_seed(args.seed)

    train_env = make_env(args.obs_mode, args.seed)
    eval_env = make_env(args.obs_mode, args.seed + 123)

    model_cls = ALGOS[args.algo]
    policy = "MlpPolicy" if args.obs_mode != "image" else "CnnPolicy"
    model = model_cls(policy, train_env, verbose=1, tensorboard_log=args.logdir, seed=args.seed)

    callbacks = build_callbacks(
        savedir=f"{args.savedir}/{args.algo}_{args.obs_mode}",
        logdir=f"{args.logdir}/{args.algo}_{args.obs_mode}",
        eval_env=eval_env,
        eval_freq=10_000,
    )

    logger = configure(f"{args.logdir}/{args.algo}_{args.obs_mode}", ["stdout", "tensorboard"])
    model.set_logger(logger)

    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(f"{args.savedir}/{args.algo}_{args.obs_mode}/final_model")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()