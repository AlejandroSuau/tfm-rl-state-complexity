from __future__ import annotations
import argparse, time, os, json
from stable_baselines3 import PPO
from envs.simple_pacman import SimplePacmanEnv, ObsConfig

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
    args = ap.parse_args()

    # Crear entorno
    env = SimplePacmanEnv(obs_config=ObsConfig(mode=args.obs_mode), seed=args.seed)

    # Modelo PPO con política MLP (tablas/vecinos)
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=args.seed,
        verbose=1,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=net_arch))

    # Entrenar
    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    # Guardar
    os.makedirs("models", exist_ok=True)
    model_path = f"models/pacman_ppo_{args.obs_mode}_seed{args.seed}.zip"
    model.save(model_path)

    # Guardar info del run
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("experiments/runs", exist_ok=True)
    info_path = f"experiments/runs/ppo_{args.obs_mode}_seed{args.seed}_{stamp}_run_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        info_json_to_dump = {
            "algo":"ppo",
            "obs_mode":args.obs_mode,
            "timesteps":args.timesteps,
            "seed":args.seed,
            "gamma":args.gamma,
            "ent_coef":args.ent_coef,
            "net_arch":net_arch,
            "model_path":model_path,
            "run_info": info_path
        }
        json.dump(info_json_to_dump, f, indent=2)

    print("✅ Entrenamiento listo:", model_path, " ", info_path)

if __name__ == "__main__":
    main()
