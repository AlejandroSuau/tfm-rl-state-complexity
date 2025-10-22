from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from envs.simple_pacman import ObsConfig, SimplePacmanEnv


def make_env(obs_mode: str, seed: int) -> Callable[[], Monitor]:
    """
    Return a thunk that creates a monitored SimplePacmanEnv with a deterministic seed.
    Used for constructing vectorized environments.
    """
    return lambda: Monitor(SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument(
        "--obs-mode",
        default="minimal",
        choices=["minimal", "bool_power", "power_time", "coins_quadrants"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--buffer-size", type=int, default=300_000)
    ap.add_argument("--learning-starts", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-freq", type=int, default=4)
    ap.add_argument("--gradient-steps", type=int, default=1)
    ap.add_argument("--target-update-interval", type=int, default=1000)
    ap.add_argument("--exploration-fraction", type=float, default=0.3)
    ap.add_argument("--exploration-final-eps", type=float, default=0.01)
    ap.add_argument("--net-arch", type=str, default="256,256")
    ap.add_argument("--vecnorm", type=int, default=1)
    ap.add_argument("--frame-stack", type=int, default=4)

    args = ap.parse_args()

    # DQN → use a single environment (most stable/robust setup).
    venv = DummyVecEnv([make_env(args.obs_mode, args.seed)])

    # Optionally stack frames (default = 4).
    if args.frame_stack and args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack)

    # Normalize ONLY observations (not rewards, since DQN is off-policy).
    if args.vecnorm:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    obs = venv.reset()
    print(f"[DEBUG] obs_space={venv.observation_space} | reset_shape={obs.shape}")

    # Parse architecture string into list of ints (e.g. "256,256" -> [256, 256]).
    net_arch = [int(x) for x in args.net_arch.split(",") if x]

    policy_kwargs = dict(net_arch=net_arch)
    model = DQN(
        policy="MlpPolicy",
        env=venv,
        seed=args.seed,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        gamma=args.gamma,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    # Save model and VecNormalize state.
    os.makedirs("models", exist_ok=True)
    model_path = f"models/pacman_dqn_{args.obs_mode}_seed{args.seed}.zip"
    model.save(model_path)

    if args.vecnorm:
        vecnorm_path = f"models/vecnorm_{args.obs_mode}_seed{args.seed}_dqn.pkl"
        venv.save(vecnorm_path)
    else:
        vecnorm_path = None

    # Log run metadata in experiments/runs.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("experiments/runs", exist_ok=True)
    run_rec = {
        "algo": "dqn",
        "obs_mode": args.obs_mode,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path,
        "frame_stack": args.frame_stack,
        "vecnorm": int(bool(args.vecnorm))
    }
    with open(
        f"experiments/runs/dqn_{args.obs_mode}_seed{args.seed}_{stamp}.json", "w"
    ) as f:
        json.dump(run_rec, f, indent=2)

    print("✅ DQN listo:", model_path, "| vecnorm:", vecnorm_path)


if __name__ == "__main__":
    main()
