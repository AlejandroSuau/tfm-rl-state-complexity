from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback

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
    ap.add_argument(
        "--early-stop",
        type=int,
        default=0,
        help="0 = sin early stopping; >0 = nº de evaluaciones sin mejora antes de parar",
    )
    ap.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Frecuencia de evaluación (en steps de entorno)",
    )
    ap.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Episodios por evaluación de early stopping",
    )

    args = ap.parse_args()

    # DQN → use a single environment (most stable/robust setup).
    venv = DummyVecEnv([make_env(args.obs_mode, args.seed)])

    # Entorno de evaluación (misma obs_mode, semilla distinta)
    eval_env = DummyVecEnv([make_env(args.obs_mode, args.seed + 100)])

    # Optionally stack frames (default = 4).
    if args.frame_stack and args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

    # Normalize ONLY observations (not rewards, since DQN is off-policy).
    if args.vecnorm:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

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
        device="auto",
        tensorboard_log="logs/tensorboard"
    )

    algo = "dqn"
    run_name = f"{algo}_{args.obs_mode}_seed{args.seed}"
    base_model_dir = os.path.join("models", algo)
    best_dir = os.path.join(base_model_dir, "best")
    checkpoints_dir = os.path.join(base_model_dir, "checkpoints")
    last_dir = os.path.join(base_model_dir, "last")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(last_dir, exist_ok=True)

    if args.early_stop > 0:
        # Callback que para si no hay mejora en la recompensa media
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.early_stop,
            min_evals=1,
            verbose=1,
        )

        eval_cb = EvalCallback(
            eval_env,
            callback_after_eval=stop_cb,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            best_model_save_path=os.path.join(best_dir, run_name),

            log_path="logs/eval_dqn",
            deterministic=True,
        )
        callback = eval_cb
    else:
        callback = None

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  # guardar cada 50k pasos, ajustable
        save_path=checkpoints_dir,
        name_prefix=run_name
    )

    model.learn(
        total_timesteps=int(args.timesteps),
        progress_bar=True,
        tb_log_name=run_name,
        callback=[callback, checkpoint_callback]
    )

    # Save model and VecNormalize state.
    model_path = os.path.join(last_dir, f"{run_name}.zip")
    model.save(model_path)

    if args.vecnorm:
        vecnorm_path = os.path.join(last_dir, f"vecnorm_{run_name}.pkl")
        venv.save(vecnorm_path)

        # Copia de vecnorm también junto al best model
        best_vecnorm_path = os.path.join(best_dir, f"vecnorm_{run_name}.pkl")
        venv.save(best_vecnorm_path)
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
        f"experiments/runs/{run_name}_{stamp}.json", "w"
    ) as f:
        json.dump(run_rec, f, indent=2)

    print("✅ DQN listo:", model_path, "| vecnorm:", vecnorm_path)


if __name__ == "__main__":
    main()
