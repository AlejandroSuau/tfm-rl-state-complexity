from __future__ import annotations
import argparse, os, csv, re, glob
import numpy as np
from stable_baselines3 import PPO
from envs.simple_pacman import SimplePacmanEnv, ObsConfig

def evaluate_model(model_path: str, obs_mode: str, episodes: int, seed: int=123):
    env = SimplePacmanEnv(obs_config=ObsConfig(mode=obs_mode), seed=seed)
    model = PPO.load(model_path, env=env)
    rewards, lengths, success = [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed+ep)
        ep_r, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)  # el env ya normaliza la acción
            ep_r += float(r); steps += 1
            if terminated or truncated:
                rewards.append(ep_r)
                lengths.append(steps)
                success.append(1 if info.get("coins_remaining", 1) == 0 else 0)
                break
    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(lengths)), float(np.mean(success))

def parse_index_or_glob(index_csv: str):
    if os.path.exists(index_csv):
        with open(index_csv, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    rows = []
    pat = re.compile(r"pacman_ppo_(?P<obs>.+)_seed(?P<seed>\d+)\.zip$")
    for m in glob.glob(os.path.join("models","pacman_ppo_*_seed*.zip")):
        name = os.path.basename(m)
        m2 = pat.match(name)
        if not m2: continue
        rows.append({"algo":"ppo","obs_mode":m2.group("obs"),"seed":m2.group("seed"),"model_path":m})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="experiments/run_index.csv")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", type=str, default="experiments/metrics.csv")
    args = ap.parse_args()

    rows = parse_index_or_glob(args.index)
    if not rows:
        print("[WARN] No hay modelos. Ejecuta primero scripts/run_grid.py")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["algo","obs_mode","seed","episodes","mean_reward","std_reward","mean_len","success_rate","model_path"])
        for r in rows:
            algo, obs_mode, seed, model_path = r["algo"], r["obs_mode"], int(r["seed"]), r["model_path"]
            if not os.path.exists(model_path):
                print(f"[WARN] no existe {model_path}, salto")
                continue
            print(f"[EVAL] {algo} | {obs_mode} | seed={seed}")
            mean_r, std_r, mean_len, succ = evaluate_model(model_path, obs_mode, episodes=args.episodes, seed=123)
            w.writerow([algo, obs_mode, seed, args.episodes, mean_r, std_r, mean_len, succ, model_path])
    print(f"[OK] métricas en {args.out}")

if __name__ == "__main__":
    main()