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
    args = ap.parse_args()

    # Crear entorno
    env = SimplePacmanEnv(obs_config=ObsConfig(mode=args.obs_mode), seed=args.seed)

    # Modelo PPO con política MLP (tablas/vecinos)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=args.seed,
        verbose=1,
        gamma=args.gamma,
        ent_coef=args.ent_coef)

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
            "model_path":model_path,
            "run_info": info_path
        }
        json.dump(info_json_to_dump, f, indent=2)

    print("✅ Entrenamiento listo:", model_path, " ", info_path)

if __name__ == "__main__":
    main()
