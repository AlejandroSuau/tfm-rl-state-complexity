from __future__ import annotations
import argparse, time, os, json
from typing import Callable, List
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.simple_pacman import SimplePacmanEnv, ObsConfig

# --- helper para crear workers vecenv (compatible con Windows/subprocess) ---
def make_env(obs_mode: str, seed: int, rank: int) -> Callable[[], SimplePacmanEnv]:
    def _init():
        env = SimplePacmanEnv(obs_config=ObsConfig(mode=obs_mode), seed=seed + rank)
        return Monitor(env)
    return _init

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--obs-mode", default="minimal",
                    choices=["minimal","bool_power","power_time","coins_quadrants"])
    ap.add_argument("--seed", type=int, default=0)

    # Hiperparámetros PPO típicos
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--net-arch", type=str, default="128,128")  # "64,64" o "128,128,64", etc.

    # --- NUEVO: multi-entorno / tamaño de rollout ---
    ap.add_argument("--n-envs", type=int, default=8, help="nº de entornos en paralelo (8 recomendado)")
    ap.add_argument("--n-steps", type=int, default=256, help="pasos por entorno antes de cada update")
    ap.add_argument("--batch-size", type=int, default=2048, help="tamaño de batch por update")
    ap.add_argument("--n-epochs", type=int, default=10, help="épocas por update PPO")

    # ---- gSDE
    ap.add_argument("--use-sde", type=int, default=0)
    ap.add_argument("--sde-sample-freq", type=int, default=4)
    ap.add_argument("--ortho-init", type=int, default=0)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--gae-lambda", type=float, default=0.95)

    args = ap.parse_args()
    # --------- construir vecenv ----------
    if args.n_envs > 1:
        # SubprocVecEnv usa procesos → más estable/rápido para PPO
        env_fns: List[Callable] = [make_env(args.obs_mode, args.seed, i) for i in range(args.n_envs)]
        base_venv = SubprocVecEnv(env_fns)
    else:
        base_venv = DummyVecEnv([make_env(args.obs_mode, args.seed, 0)])

    # Normalización (obs y reward) durante el train
    vec = VecNormalize(base_venv, norm_obs=True, norm_reward=True, clip_reward=10.0)

    is_discrete = isinstance(base_venv.action_space, Discrete)
    use_sde_flag = bool(getattr(args, "use_sde", 0))
    if is_discrete and use_sde_flag:
        print("[WARN] gSDE no soportado en acciones discretas. Desactivando use_sde.")
        use_sde_flag = False


    # --------- modelo PPO ----------
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    model = PPO(
        policy="MlpPolicy",
        env=vec,
        seed=args.seed,
        verbose=1,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        # puedes dejar clip_range/gae_lambda en defaults si no los expones por CLI
        policy_kwargs=dict(net_arch=net_arch),
        use_sde=use_sde_flag,          # <- quedará False en discreto
        sde_sample_freq=getattr(args, "sde_sample_freq", -1),
    )

    # --------- entrenar ----------
    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    # --------- guardar ----------
    os.makedirs("models", exist_ok=True)
    model_path = f"models/pacman_ppo_{args.obs_mode}_seed{args.seed}.zip"
    model.save(model_path)

    # MUY IMPORTANTE: guardar estadísticas de normalización
    vecnorm_path = f"models/vecnorm_{args.obs_mode}_seed{args.seed}.pkl"
    vec.save(vecnorm_path)

    # guardamos metadatos del run
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("experiments/runs", exist_ok=True)
    info_path = f"experiments/runs/ppo_{args.obs_mode}_seed{args.seed}_{stamp}_run_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "algo": "ppo",
            "obs_mode": args.obs_mode,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "lr": args.lr,
            "net_arch": net_arch,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "model_path": model_path,
            "vecnorm_path": vecnorm_path,
            "run_info": info_path
        }, f, indent=2)

    print("✅ Entrenamiento listo:", model_path)
    print("   VecNormalize guardado en:", vecnorm_path)
    # cerrar procesos limpios
    vec.close()

if __name__ == "__main__":
    main()