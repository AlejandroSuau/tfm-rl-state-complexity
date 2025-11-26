from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, List

from stable_baselines3 import A2C
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

    # Hiperparámetros típicos de A2C
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--net-arch", type=str, default="128,128")

    # Multi-entorno y tamaño de rollouts
    ap.add_argument("--n-envs", type=int, default=8, help="nº de entornos en paralelo")
    ap.add_argument("--n-steps", type=int, default=32, help="pasos por entorno antes de cada update")

    # Normalización y frame stacking (alineado con PPO/DQN)
    ap.add_argument("--vecnorm", type=int, default=1, help="1 = usar VecNormalize, 0 = no")
    ap.add_argument("--frame-stack", type=int, default=4, help="nº de frames apilados (>=1)")

    # Early stopping + evaluación (igual estilo que DQN/PPO)
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

    # Flags para continuar entrenamiento (igual que en PPO)
    ap.add_argument(
        "--continue-model",
        type=str,
        default="",
        help="Ruta al modelo A2C (.zip) desde el que continuar el entrenamiento",
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

    # Frame stacking (antes de normalizar, como en DQN/PPO)
    if args.frame_stack and args.frame_stack > 1:
        train_env = VecFrameStack(train_env, n_stack=args.frame_stack)
        eval_base_env = VecFrameStack(eval_base_env, n_stack=args.frame_stack)

    # ---------- Normalización (VecNormalize) ----------
    # Casos:
    #  - Si continue_vecnorm: cargar ese estado tanto en train como en eval.
    #  - Si no, y vecnorm=1: crear VecNormalize nuevo.
    #  - Si vecnorm=0 y sin continue_vecnorm: no normalizamos.
    if args.continue_vecnorm:
        # Cargar VecNormalize previo para entrenamiento
        train_env = VecNormalize.load(args.continue_vecnorm, train_env)
        train_env.training = True
        train_env.norm_reward = True  # típico en on-policy

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
                norm_reward=False,  # no normalizar reward en eval para ver reward “real”
                clip_obs=10.0,
            )
        else:
            eval_venv = eval_base_env

    obs = train_env.reset()
    print(f"[DEBUG] obs_space={train_env.observation_space} | reset_shape={obs.shape}")

    # ---------- Definición de la política A2C ----------
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    policy_kwargs = dict(net_arch=net_arch)

    # Si se ha pasado --continue-model, cargamos el modelo desde disco.
    # Si no, creamos uno nuevo.
    if continuing:
        if not args.continue_model:
            raise ValueError("Se indicó continuar entrenamiento pero falta --continue-model")
        print(f"[INFO] Continuando entrenamiento A2C desde modelo: {args.continue_model}")
        model = A2C.load(args.continue_model, env=train_env, device="auto")
    else:
        model = A2C(
            policy="MlpPolicy",
            env=train_env,
            seed=args.seed,
            n_steps=args.n_steps,
            learning_rate=args.lr,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            gae_lambda=args.gae_lambda,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="auto",
            tensorboard_log="logs/tensorboard",
        )

    algo = "a2c"
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
            log_path="logs/eval_a2c",
            deterministic=True,
        )
        callbacks.append(eval_cb)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoints_dir,
        name_prefix=run_name,
    )
    callbacks.append(checkpoint_callback)

    # ========== Entrenamiento con manejo de interrupción ==========
    interrupted = False
    try:
        model.learn(
            total_timesteps=int(args.timesteps),
            progress_bar=True,
            callback=callbacks if callbacks else None,
            reset_num_timesteps=not continuing,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[WARN] Entrenamiento A2C interrumpido por el usuario (Ctrl+C). Guardando estado...")
    finally:
        # ========== Guardar modelo y VecNormalize ==========
        model_path = os.path.join(last_dir, f"{run_name}.zip")
        model.save(model_path)

        # Guardar VecNormalize si existe
        if isinstance(train_env, VecNormalize):
            vecnorm_path = os.path.join(last_dir, f"vecnorm_{run_name}.pkl")
            train_env.save(vecnorm_path)

            # Copia también a best/
            best_vecnorm_path = os.path.join(best_dir, f"vecnorm_{run_name}.pkl")
            train_env.save(best_vecnorm_path)
        else:
            vecnorm_path = None

        # ========== Log de ejecución ==========
        stamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("experiments/runs", exist_ok=True)

        run_rec = {
            "algo": "a2c",
            "obs_mode": args.obs_mode,
            "seed": args.seed,
            "timesteps": args.timesteps,
            "model_path": model_path,
            "vecnorm_path": vecnorm_path,
            "frame_stack": args.frame_stack,
            "vecnorm": int(isinstance(train_env, VecNormalize)),
            "lr": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "use_rms_prop": args.use_rms_prop,
            "net_arch": net_arch,
            "continue_model": args.continue_model,
            "continue_vecnorm": args.continue_vecnorm,
            "interrupted": int(interrupted),
        }

        json_path = f"experiments/runs/{run_name}_{stamp}.json"
        with open(json_path, "w") as f:
            json.dump(run_rec, f, indent=2)

        if interrupted:
            print(f"⏹️ Entrenamiento A2C interrumpido. Modelo guardado en: {model_path}")
        else:
            print(f"✅ A2C listo: {model_path} | vecnorm: {vecnorm_path}")



if __name__ == "__main__":
    main()
