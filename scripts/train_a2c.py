from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, List

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.simple_pacman import ObsConfig, SimplePacmanEnv


def make_env(obs_mode: str, seed: int, rank: int = 0) -> Callable[[], Monitor]:
    """
    Return a thunk that constructs a monitored SimplePacmanEnv with a deterministic seed.
    The seed is offset by `rank` for vectorized training.
    """
    return lambda: Monitor(SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=seed + rank))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument(
        "--obs-mode",
        default="minimal",
        choices=["minimal", "bool_power", "power_time", "coins_quadrants"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--net-arch", type=str, default="128,128")
    args = ap.parse_args()

    # Vectorized environment: Subproc for n_envs > 1, otherwise Dummy.
    if args.n_envs > 1:
        venv = SubprocVecEnv(
            [make_env(args.obs_mode, args.seed, i) for i in range(args.n_envs)]
        )
    else:
        venv = DummyVecEnv([make_env(args.obs_mode, args.seed, 0)])

    # Normalize observations and rewards (kept exactly as original).
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

    # Parse network architecture string into a list of ints (e.g., "128,128" -> [128, 128]).
    net_arch: List[int] = [int(x) for x in args.net_arch.split(",") if x.strip()]

    model = A2C(
        policy="MlpPolicy",
        env=venv,
        seed=args.seed,
        n_steps=args.n_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        verbose=1,
        policy_kwargs=dict(net_arch=net_arch),
    )

    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    # Persist model and VecNormalize state.
    os.makedirs("models", exist_ok=True)
    model_path = f"models/pacman_a2c_{args.obs_mode}_seed{args.seed}.zip"
    model.save(model_path)
    vecnorm_path = f"models/vecnorm_{args.obs_mode}_seed{args.seed}_a2c.pkl"
    venv.save(vecnorm_path)

    # Log the run in experiments/runs with a timestamp, preserving original schema.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("experiments/runs", exist_ok=True)
    run_rec = {
        "algo": "a2c",
        "obs_mode": args.obs_mode,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path,
    }
    with open(
        f"experiments/runs/a2c_{args.obs_mode}_seed{args.seed}_{stamp}.json", "w"
    ) as f:
        json.dump(run_rec, f, indent=2)

    print("✅ A2C listo:", model_path, "| vecnorm:", vecnorm_path)


if __name__ == "__main__":
    main()
