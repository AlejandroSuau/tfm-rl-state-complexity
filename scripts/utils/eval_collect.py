from __future__ import annotations
import argparse, os, csv, re, glob
import numpy as np
from stable_baselines3 import PPO
from envs.simple_pacman import SimplePacmanEnv, ObsConfig

def evaluate_model(model_path: str, obs_mode: str, episodes: int, seed: int=123):
    env = SimplePacmanEnv(obs_config=ObsConfig(mode=obs_mode), seed=seed)
    model = PPO.load(model_path, env=env)
    rewards, lengths, success = [], [], []
    ratios, near = [], 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed+ep)
        ep_r, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)  # el env ya normaliza la acción
            ep_r += float(r); steps += 1
            
            if terminated or truncated:
                end_coins = info.get("coins_remaining", None)
                total = info.get("coins_total", None)
                if (total is not None) and total > 0 and (end_coins is not None):
                    r = 1.0 - float(end_coins)/float(total)
                    ratios.append(r)
                    if r >= 0.90:
                        near += 1

                rewards.append(ep_r)
                lengths.append(steps)
                success.append(1 if end_coins == 0 else 0)
                break
    
    mean_r = float(np.mean(rewards)) if rewards else 0.0
    std_r  = float(np.std(rewards)) if rewards else 0.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    succ_rate = float(np.mean(success)) if success else 0.0
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    near_rate = float(near/episodes)

    return mean_r, std_r, mean_len, succ_rate, mean_ratio, near_rate

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
    ap.add_argument("--episodes", type=int, default=50)
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
            w.writerow(["algo","obs_mode","seed","episodes","mean_reward","std_reward","mean_len","success_rate","completion_ratio","near_clear_rate","model_path"])
        for r in rows:
            algo, obs_mode, seed, model_path = r["algo"], r["obs_mode"], int(r["seed"]), r["model_path"]
            if not os.path.exists(model_path):
                print(f"[WARN] no existe {model_path}, salto")
                continue
            print(f"[EVAL] {algo} | {obs_mode} | seed={seed}")
            mean_r, std_r, mean_len, succ, mean_ratio, near_rate = evaluate_model(model_path, obs_mode, episodes=args.episodes, seed=123)
            w.writerow([algo, obs_mode, seed, args.episodes, mean_r, std_r, mean_len, succ, mean_ratio, near_rate, model_path])
    print(f"[OK] métricas en {args.out}")

if __name__ == "__main__":
    main()