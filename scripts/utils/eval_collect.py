from __future__ import annotations
import argparse, os, csv, re, glob
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.simple_pacman import SimplePacmanEnv, ObsConfig


def is_vecenv(env) -> bool:
    return hasattr(env, "num_envs")


def build_eval_env(obs_mode: str, seed: int, vecnorm_path: str | None):
    """
    Si vecnorm_path existe => crea DummyVecEnv(1) + carga VecNormalize y desactiva entrenamiento.
    (Obs normalizadas; reward SIN normalizar para reportar métricas comparables.)
    """
    if vecnorm_path and os.path.exists(vecnorm_path):
        base = DummyVecEnv([lambda: Monitor(SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=seed))])
        venv = VecNormalize.load(vecnorm_path, base)
        venv.training = False
        venv.norm_reward = False
        print(f"[INFO] VecNormalize cargado: {vecnorm_path}")
        return venv
    else:
        print("[WARN] Sin VecNormalize (.pkl no encontrado). Eval sin normalizar -> rendimiento menor.")
        return SimplePacmanEnv(obs_config=ObsConfig(mode=obs_mode), seed=seed)


def evaluate_model(model_path: str, obs_mode: str, episodes: int, seed: int = 123, vecnorm_path: str | None = None):
    env = build_eval_env(obs_mode, seed, vecnorm_path)
    model = PPO.load(model_path, env=env, device="cpu")

    use_vec = is_vecenv(env)

    rewards, lengths, success = [], [], []
    ratios, near = [], 0

    for ep in range(episodes):
        # reset + init info
        if use_vec:
            obs = env.reset()
            try:
                init_coins = int(env.get_attr("coins_remaining")[0])
                total_coins = int(env.get_attr("coins_total")[0])
            except Exception:
                init_coins, total_coins = None, None
        else:
            obs, info = env.reset(seed=seed + ep)
            init_coins = info.get("coins_remaining", None)
            total_coins = info.get("coins_total", None)

        ep_r, steps = 0.0, 0

        while True:
            action, _ = model.predict(obs, deterministic=True)

            if use_vec:
                # VecEnv: acción shape (1,)
                if not isinstance(action, np.ndarray):
                    action = np.array([int(action)], dtype=np.int64)
                obs, r, done, infos = env.step(action)
                r = float(np.asarray(r).squeeze()[()])
                ep_r += r; steps += 1
                if bool(np.asarray(done).squeeze()[()]):
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos
                    end_coins = info.get("coins_remaining", None)
                    if (total_coins is not None) and (end_coins is not None) and total_coins > 0:
                        comp = 1.0 - float(end_coins) / float(total_coins)
                        ratios.append(comp)
                        if comp >= 0.90:
                            near += 1
                    rewards.append(ep_r); lengths.append(steps)
                    success.append(1 if end_coins == 0 else 0)
                    break
            else:
                obs, r, terminated, truncated, info = env.step(int(action))
                ep_r += float(r); steps += 1
                if terminated or truncated:
                    end_coins = info.get("coins_remaining", None)
                    total = info.get("coins_total", None)
                    if (total is not None) and (end_coins is not None) and total > 0:
                        comp = 1.0 - float(end_coins) / float(total)
                        ratios.append(comp)
                        if comp >= 0.90:
                            near += 1
                    rewards.append(ep_r); lengths.append(steps)
                    success.append(1 if end_coins == 0 else 0)
                    break

    mean_r = float(np.mean(rewards)) if rewards else 0.0
    std_r  = float(np.std(rewards)) if rewards else 0.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    succ_rate = float(np.mean(success)) if success else 0.0
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    near_rate = float(near / episodes)
    return mean_r, std_r, mean_len, succ_rate, mean_ratio, near_rate


def parse_index_or_glob(index_csv: str):
    if os.path.exists(index_csv):
        with open(index_csv, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    rows = []
    pat = re.compile(r"pacman_ppo_(?P<obs>.+)_seed(?P<seed>\d+)\.zip$")
    for m in glob.glob(os.path.join("models", "pacman_ppo_*_seed*.zip")):
        name = os.path.basename(m)
        m2 = pat.match(name)
        if not m2:
            continue
        rows.append({"algo": "ppo", "obs_mode": m2.group("obs"), "seed": m2.group("seed"), "model_path": m})
    return rows


def maybe_vecnorm_path(row) -> str | None:
    """
    Si el índice incluye 'vecnorm_path' úsalo; si no, infiere el nombre estándar
    models/vecnorm_{obs_mode}_seed{seed}.pkl y úsalo si existe.
    """
    if "vecnorm_path" in row and row["vecnorm_path"]:
        return row["vecnorm_path"]
    # inferir por convención
    obs_mode = row["obs_mode"]; seed = int(row["seed"])
    cand = os.path.join("models", f"vecnorm_{obs_mode}_seed{seed}.pkl")
    return cand if os.path.exists(cand) else None


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
    # sobreescribimos para evitar duplicados de corridas anteriores
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "algo","obs_mode","seed","episodes",
            "mean_reward","std_reward","mean_len","success_rate",
            "completion_ratio","near_clear_rate","model_path","vecnorm_used"
        ])
        for r in rows:
            algo, obs_mode, seed, model_path = r["algo"], r["obs_mode"], int(r["seed"]), r["model_path"]
            if not os.path.exists(model_path):
                print(f"[WARN] no existe {model_path}, salto")
                continue
            vecnorm = maybe_vecnorm_path(r)
            print(f"[EVAL] {algo} | {obs_mode} | seed={seed} | vecnorm={'yes' if vecnorm else 'no'}")
            mean_r, std_r, mean_len, succ, mean_ratio, near_rate = evaluate_model(
                model_path, obs_mode, episodes=args.episodes, seed=123, vecnorm_path=vecnorm
            )
            w.writerow([algo, obs_mode, seed, args.episodes,
                        mean_r, std_r, mean_len, succ, mean_ratio, near_rate, model_path,
                        ("yes" if vecnorm else "no")])
    print(f"[OK] métricas en {args.out}")


if __name__ == "__main__":
    main()
