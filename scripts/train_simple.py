from __future__ import annotations
import argparse, time, os, json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.simple_pacman import SimplePacmanEnv, ObsConfig

def make_env(obs_mode: str, seed: int):
    """Envolvemos el env en un thunk para DummyVecEnv (1 entorno)."""
    def _thunk():
        return SimplePacmanEnv(obs_config=ObsConfig(mode=obs_mode), seed=seed)
    return _thunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--obs-mode", default="minimal",
                    choices=["minimal","bool_power","power_time","coins_quadrants"], 
                    help="Mantén 'minimal' para empezar. 'image' se deja para más adelante.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--net-arch", type=str, default="128,128", help="Tamaños MLP separados por coma, p.ej. '64,64' o '128,128,64'")

     # Parámetros de VecNormalize
    ap.add_argument("--norm-obs", dest="norm_obs", action="store_true")
    ap.add_argument("--no-norm-obs", dest="norm_obs", action="store_false")
    ap.set_defaults(norm_obs=True)

    ap.add_argument("--norm-reward", dest="norm_reward", action="store_true")
    ap.add_argument("--no-norm-reward", dest="norm_reward", action="store_false")
    ap.set_defaults(norm_reward=True)

    ap.add_argument("--clip-obs", type=float, default=5.0)
    ap.add_argument("--clip-reward", type=float, default=10.0)

    args = ap.parse_args()

    # ---------- VecEnv + VecNormalize (1 entorno es suficiente para usar VN) ----------
    vec = DummyVecEnv([make_env(args.obs_mode, args.seed)])
    vec = VecNormalize(
        vec,
        norm_obs=args.norm_obs,
        norm_reward=args.norm_reward,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
        gamma=args.gamma,
    )

    # ---------- Modelo PPO ----------
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    model = PPO(
        policy="MlpPolicy",
        env=vec,
        seed=args.seed,
        verbose=1,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=net_arch))

    # ---------- Entrenar ----------
    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    # Guardar
    os.makedirs("models", exist_ok=True)
    model_path = f"models/pacman_ppo_{args.obs_mode}_seed{args.seed}.zip"
    vecnorm_path = f"models/vecnorm_{args.obs_mode}_seed{args.seed}.pkl"

    model.save(model_path)
    vec.save(vecnorm_path)

    # ---------- Guardar info del run ----------
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
            "net_arch": net_arch,
            "vecnormalize": {
                "norm_obs": args.norm_obs,
                "norm_reward": args.norm_reward,
                "clip_obs": args.clip_obs,
                "clip_reward": args.clip_reward,
                "stats_path": vecnorm_path,
            },
            "model_path": model_path,
            "run_info": info_path,
        }, f, indent=2)

    print("✅ Entrenamiento listo:", model_path)
    print("✅ VecNormalize guardado en:", vecnorm_path)
    print("ℹ️  run_info:", info_path)

if __name__ == "__main__":
    main()
