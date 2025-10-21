# ============================================================
#  optuna_tune.py
#  ----------------
#  This script performs hyperparameter optimization for RL algorithms
#  (PPO, A2C, DQN) using Optuna. It automatically searches for the
#  best configuration that maximizes the average episodic reward
#  in the SimplePacmanEnv environment.
#
#  Each trial trains a fresh model for a given combination of
#  hyperparameters and evaluates it on a few episodes.
#
#  Master’s Thesis: Reinforcement Learning – Impact of State Complexity
# ============================================================

import argparse
import json
import os
import numpy as np
import optuna

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack

from envs.simple_pacman import SimplePacmanEnv, ObsConfig


# ============================================================
#  Environment creation helpers
# ============================================================

def make_env(obs_mode: str, seed: int, rank: int = 0):
    """
    Returns a function that creates a single instance of SimplePacmanEnv,
    wrapped with a Monitor to record statistics.

    Args:
        obs_mode: observation mode ('minimal', 'bool_power', etc.)
        seed: random seed for reproducibility.
        rank: offset to differentiate seeds across parallel environments.
    """
    def _init():
        env = SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=seed + rank)
        return Monitor(env)
    return _init


def build_vecenv(algo: str, obs_mode: str, seed: int, n_envs: int = 8, frame_stack: int = 4):
    """
    Creates a vectorized environment depending on the algorithm type.

    - PPO / A2C: use SubprocVecEnv (multi-process) or DummyVecEnv (single-thread)
      with normalization of both observations and rewards.
    - DQN: off-policy, therefore use a single environment, optional frame stacking,
      and normalize only observations.

    Returns:
        Vectorized environment compatible with Stable-Baselines3.
    """
    if algo in {"ppo", "a2c"}:
        # On-policy algorithms can run multiple envs in parallel.
        if n_envs > 1:
            venv = SubprocVecEnv([make_env(obs_mode, seed, i) for i in range(n_envs)])
        else:
            venv = DummyVecEnv([make_env(obs_mode, seed, 0)])
        # Normalize both observations and rewards for stability.
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=10.0)
        return venv
    else:
        # DQN typically runs on a single environment (off-policy learning).
        venv = DummyVecEnv([make_env(obs_mode, seed, 0)])
        # Optional frame stacking to provide temporal context to DQN.
        if frame_stack and frame_stack > 1:
            venv = VecFrameStack(venv, n_stack=frame_stack)
        # Normalize only observations (reward normalization not needed off-policy).
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        return venv


def evaluate(model, env, n_episodes: int = 5):
    """
    Runs several evaluation episodes using a trained model and returns
    the mean total reward. Compatible with vectorized environments.

    Args:
        model: trained RL model (PPO, A2C, DQN)
        env: environment to evaluate on
        n_episodes: number of evaluation episodes

    Returns:
        Mean episodic return across all evaluation episodes.
    """
    ep_returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # VecEnv returns (obs, rewards, dones, infos)
            obs, reward, dones, info = env.step(action)
            ep_ret += float(np.asarray(reward).mean())
            done = bool(np.asarray(dones).any())
        ep_returns.append(ep_ret)
    return float(np.mean(ep_returns))


# ============================================================
#  Hyperparameter search spaces for each algorithm
# ============================================================

def suggest_shared(trial):
    """
    Defines parameters shared across all algorithms:
    learning rate, gamma, entropy coefficient and network architecture.
    """
    return {
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
        "net_arch": trial.suggest_categorical("net_arch", ["64,64", "128,128", "256,128"]),
    }


def suggest_ppo(trial, n_envs: int):
    """
    Search space for PPO-specific hyperparameters.

    The batch size is derived dynamically from the rollout size (n_steps * n_envs)
    divided by a sampled batch_factor to ensure divisibility and avoid truncated minibatches.
    """
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
    rollout = n_steps * n_envs
    batch_factor = trial.suggest_categorical("batch_factor", [1, 2, 4, 8])

    batch_size = rollout // batch_factor
    if batch_size < 128:
        batch_size = 128  # Safety floor

    # Store derived value for logging/debugging
    trial.set_user_attr("derived_batch_size", batch_size)

    n_epochs = trial.suggest_categorical("n_epochs", [4, 8, 10])
    return {"n_steps": n_steps, "batch_size": batch_size, "n_epochs": n_epochs}


def suggest_a2c(trial):
    """
    Search space for A2C-specific hyperparameters.
    """
    return {
        "n_steps": trial.suggest_categorical("n_steps", [16, 32, 64, 128]),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.98),
    }


def suggest_dqn(trial):
    """
    Search space for DQN-specific hyperparameters.
    Includes replay buffer size, batch size, exploration schedule, and target update frequency.
    """
    return {
        "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 300_000, 500_000]),
        "learning_starts": trial.suggest_categorical("learning_starts", [5_000, 20_000, 50_000]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "target_update_interval": trial.suggest_categorical("target_update_interval", [500, 1000, 2000]),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.005, 0.05),
        "frame_stack": trial.suggest_categorical("frame_stack", [1, 4]),
    }


# ============================================================
#  Model builder for Stable-Baselines3 algorithms
# ============================================================

def build_model(algo: str, env, shared, specific, seed: int):
    """
    Instantiates and returns the SB3 model given algorithm type and hyperparameters.

    Args:
        algo: 'ppo', 'a2c', or 'dqn'
        env: VecEnv used for training
        shared: dictionary of common hyperparameters
        specific: dictionary of algorithm-specific hyperparameters
        seed: random seed

    Returns:
        Initialized (untrained) SB3 model ready for learning.
    """
    net_arch = [int(x) for x in shared["net_arch"].split(",")]

    if algo == "ppo":
        return PPO(
            "MlpPolicy", env, seed=seed,
            learning_rate=shared["learning_rate"],
            gamma=shared["gamma"],
            ent_coef=shared["ent_coef"],
            n_steps=specific["n_steps"],
            batch_size=specific["batch_size"],
            n_epochs=specific["n_epochs"],
            policy_kwargs=dict(net_arch=net_arch),
            verbose=0,
        )
    elif algo == "a2c":
        return A2C(
            "MlpPolicy", env, seed=seed,
            learning_rate=shared["learning_rate"],
            gamma=shared["gamma"],
            ent_coef=shared["ent_coef"],
            n_steps=specific["n_steps"],
            gae_lambda=specific["gae_lambda"],
            policy_kwargs=dict(net_arch=net_arch),
            verbose=0,
        )
    elif algo == "dqn":
        return DQN(
            "MlpPolicy", env, seed=seed,
            learning_rate=shared["learning_rate"],
            gamma=shared["gamma"],
            buffer_size=specific["buffer_size"],
            learning_starts=specific["learning_starts"],
            batch_size=specific["batch_size"],
            train_freq=specific["train_freq"],
            gradient_steps=specific["gradient_steps"],
            target_update_interval=specific["target_update_interval"],
            exploration_fraction=specific["exploration_fraction"],
            exploration_final_eps=specific["exploration_final_eps"],
            policy_kwargs=dict(net_arch=net_arch),
            verbose=0,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


# ============================================================
#  Optuna objective and main execution
# ============================================================

def main():
    """
    Entry point for Optuna-based hyperparameter optimization.

    CLI arguments:
      --algo: algorithm name ('ppo', 'a2c', or 'dqn')
      --obs-mode: observation configuration
      --timesteps: training steps per trial
      --trials: number of Optuna trials
      --seed: base random seed
      --n-envs: number of parallel envs for PPO/A2C
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "a2c", "dqn"], required=True)
    ap.add_argument("--obs-mode", default="coins_quadrants",
                    choices=["minimal", "bool_power", "power_time", "coins_quadrants"])
    ap.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps per trial (short runs).")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8, help="Only relevant for PPO/A2C.")
    args = ap.parse_args()

    # ---------- Optuna objective function ----------
    def objective(trial: optuna.Trial):
        # Shared and algorithm-specific parameters
        shared = suggest_shared(trial)
        if args.algo == "ppo":
            specific = suggest_ppo(trial, args.n_envs)
            env = build_vecenv("ppo", args.obs_mode, args.seed, n_envs=args.n_envs)
        elif args.algo == "a2c":
            specific = suggest_a2c(trial)
            env = build_vecenv("a2c", args.obs_mode, args.seed, n_envs=args.n_envs)
        elif args.algo == "dqn":
            specific = suggest_dqn(trial)
            env = build_vecenv("dqn", args.obs_mode, args.seed, n_envs=1, frame_stack=specific["frame_stack"])
        else:
            raise ValueError("Unsupported algorithm selected.")

        # Model creation and training
        model = build_model(args.algo, env, shared, specific, seed=args.seed)
        model.learn(total_timesteps=int(args.timesteps))

        # Evaluation and objective return
        mean_ret = evaluate(model, env, n_episodes=5)
        env.close()
        return mean_ret

    # ---------- Run the optimization ----------
    study = optuna.create_study(direction="maximize", study_name=f"{args.algo}_{args.obs_mode}")
    study.optimize(objective, n_trials=args.trials)

    # ---------- Save best results ----------
    os.makedirs("experiments/optuna", exist_ok=True)
    out = {
        "algo": args.algo,
        "obs_mode": args.obs_mode,
        "timesteps_per_trial": args.timesteps,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    out_path = f"experiments/optuna/best_{args.algo}_{args.obs_mode}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("- Best parameters found:", json.dumps(study.best_params, indent=2))
    print("- Saved to:", out_path)


if __name__ == "__main__":
    main()
