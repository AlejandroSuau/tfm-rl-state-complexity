from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, List

import torch
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CheckpointCallback,
)

from envs.simple_pacman import ObsConfig, SimplePacmanEnv


def make_env(obs_mode: str, base_seed: int, rank: int = 0) -> Callable[[], Monitor]:
    """
    Devuelve un thunk que crea un SimplePacmanEnv monitorizado con semilla determinista.
    Para vecenv con múltiples procesos usamos `rank` para desfasar la semilla base.
    """
    def _init():
        env = SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=base_seed + rank)
        return Monitor(env)

    return _init

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument(
        "--obs-mode",
        default="minimal",
        choices=["minimal", "bool_power", "power_time", "coins_quadrants"],
    )
    ap.add_argument("--seed", type=int, default=0)

    # Hiperparámetros típicos de PPO
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--net-arch", type=str, default="128,128")

    # Multi-entorno y tamaño de rollouts
    ap.add_argument("--n-envs", type=int, default=8, help="nº de entornos en paralelo")
    ap.add_argument("--n-steps", type=int, default=256, help="pasos por entorno antes de cada update")
    ap.add_argument("--batch-size", type=int, default=2048, help="tamaño de batch por update")
    ap.add_argument("--n-epochs", type=int, default=10, help="épocas por update PPO")

    # Opciones de normalización y frame stacking (paralelo a DQN)
    ap.add_argument("--vecnorm", type=int, default=1, help="1 = usar VecNormalize, 0 = no")
    ap.add_argument("--frame-stack", type=int, default=4, help="nº de frames apilados (>=1)")

    # gSDE (no se usa en acciones discretas, pero dejamos flags por compatibilidad)
    ap.add_argument("--use-sde", type=int, default=0)
    ap.add_argument("--sde-sample-freq", type=int, default=4)
    ap.add_argument("--ortho-init", type=int, default=0)

    # Early stopping + evaluación, igual estilo que tu DQN
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

    # NUEVO: flags para continuar entrenamiento
    ap.add_argument(
        "--continue-model",
        type=str,
        default="",
        help="Ruta al modelo PPO (.zip) desde el que continuar el entrenamiento",
    )
    ap.add_argument(
        "--continue-vecnorm",
        type=str,
        default="",
        help="Ruta al VecNormalize (.pkl) desde el que continuar (debe coincidir con el modelo)",
    )

    args = ap.parse_args()

    continuing = bool(args.continue_model)

    # ---------- Construcción del vecenv de entrenamiento ----------
    if args.n_envs > 1:
        env_fns: List[Callable] = [
            make_env(args.obs_mode, args.seed, rank=i) for i in range(args.n_envs)
        ]
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv([make_env(args.obs_mode, args.seed, rank=0)])

    # Entorno base de evaluación (1 solo env, semilla distinta)
    eval_base_env = DummyVecEnv([make_env(args.obs_mode, args.seed + 100, rank=0)])

    # Frame stacking (antes de normalizar, como en DQN)
    if args.frame_stack and args.frame_stack > 1:
        train_env = VecFrameStack(train_env, n_stack=args.frame_stack)
        eval_base_env = VecFrameStack(eval_base_env, n_stack=args.frame_stack)

    # ---------- Normalización (VecNormalize) ----------
    # Casos:
    #  - Si continue_vecnorm: cargar ese estado tanto en train como en eval.
    #  - Si no, y vecnorm=1: crear VecNormalize nuevo (como antes).
    #  - Si vecnorm=0 y sin continue_vecnorm: no normalizamos.
    if args.continue_vecnorm:
        # Cargar VecNormalize previo para entrenamiento
        train_env = VecNormalize.load(args.continue_vecnorm, train_env)
        train_env.training = True
        train_env.norm_reward = True  # lo usabas así en PPO

        # Cargar VecNormalize también para eval, pero en modo evaluación
        eval_venv = VecNormalize.load(args.continue_vecnorm, eval_base_env)
        eval_venv.training = False
        eval_venv.norm_reward = False
    else:
        if args.vecnorm:
            train_env = VecNormalize(
                train_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )
            eval_venv = VecNormalize(
                eval_base_env,
                norm_obs=True,
                norm_reward=False,  # típico: no normalizar reward en eval para ver reward “real”
                clip_obs=10.0,
            )
        else:
            eval_venv = eval_base_env

    obs = train_env.reset()
    print(f"[DEBUG] obs_space={train_env.observation_space} | reset_shape={obs.shape}")

    # ---------- Definición de la política PPO ----------
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]

    # gSDE no soportado para acciones discretas
    is_discrete = isinstance(train_env.action_space, Discrete)
    use_sde_flag = bool(args.use_sde)
    if is_discrete and use_sde_flag:
        print("[WARN] gSDE no soportado con acciones discretas. Desactivando use_sde.")
        use_sde_flag = False

    policy_kwargs = dict(net_arch=net_arch)
    if args.ortho_init == 0:
        policy_kwargs["ortho_init"] = False

    # Si se ha pasado --continue-model, cargamos el modelo desde disco.
    # Si no, creamos uno nuevo como antes.
    if continuing:
        if not args.continue_model:
            raise ValueError("Se indicó continuar entrenamiento pero falta --continue-model")
        print(f"[INFO] Continuando entrenamiento desde modelo: {args.continue_model}")
        model = PPO.load(args.continue_model, env=train_env, device="auto")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            seed=args.seed,
            learning_rate=args.lr,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            gae_lambda=args.gae_lambda,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs=policy_kwargs,
            use_sde=use_sde_flag,
            sde_sample_freq=args.sde_sample_freq,
            verbose=1,
            device="auto",  # usará CUDA si está disponible
            tensorboard_log="logs/tensorboard",
        )

    algo = "ppo"
    run_name = f"{algo}_{args.obs_mode}_seed{args.seed}"
    base_model_dir = os.path.join("models", algo)
    best_dir = os.path.join(base_model_dir, "best")
    checkpoints_dir = os.path.join(base_model_dir, "checkpoints")
    last_dir = os.path.join(base_model_dir, "last")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(last_dir, exist_ok=True)

    # ---------- Callbacks: early stopping + checkpoints ----------
    callbacks = []

    if args.early_stop > 0:
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.early_stop,
            min_evals=1,
            verbose=1,
        )

        eval_cb = EvalCallback(
            eval_venv,
            callback_after_eval=stop_cb,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            best_model_save_path=os.path.join(best_dir, run_name),
            log_path="logs/eval_ppo",
            deterministic=True,
        )
        callbacks.append(eval_cb)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoints_dir,
        name_prefix=run_name,
    )
    callbacks.append(checkpoint_callback)

    # ---------- Entrenamiento ----------
    model.learn(
        total_timesteps=int(args.timesteps),
        progress_bar=True,
        tb_log_name=run_name,
        callback=callbacks if callbacks else None,
        # CLAVE: si continuamos, no reseteamos el contador global
        reset_num_timesteps=not continuing,
    )

    # ---------- Guardar modelo y estado de VecNormalize ----------
    model_path = os.path.join(last_dir, f"{run_name}.zip")
    model.save(model_path)

    # train_env es VecNormalize solo si args.vecnorm=1 o si cargamos continue_vecnorm
    if isinstance(train_env, VecNormalize):
        # VecNormalize para el último modelo
        vecnorm_path = os.path.join(
            last_dir, f"vecnorm_{run_name}.pkl"
        )
        train_env.save(vecnorm_path)

        # Copia también a best/ para evaluar el best_model.zip
        best_vecnorm_path = os.path.join(
            best_dir, f"vecnorm_{run_name}.pkl"
        )
        train_env.save(best_vecnorm_path)
    else:
        vecnorm_path = None

    # ---------- Log de la ejecución en experiments/runs ----------
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("experiments/runs", exist_ok=True)
    run_rec = {
        "algo": "ppo",
        "obs_mode": args.obs_mode,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path,
        "frame_stack": args.frame_stack,
        "vecnorm": int(isinstance(train_env, VecNormalize)),
        "lr": args.lr,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
        "gae_lambda": args.gae_lambda,
        "net_arch": net_arch,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "continue_model": args.continue_model,
        "continue_vecnorm": args.continue_vecnorm,
    }
    with open(
        f"experiments/runs/{run_name}_{stamp}.json", "w"
    ) as f:
        json.dump(run_rec, f, indent=2)

    print("✅ PPO listo:", model_path, "| vecnorm:", vecnorm_path)


if __name__ == "__main__":
    main()
