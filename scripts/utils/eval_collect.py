from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from envs.simple_pacman import ObsConfig, SimplePacmanEnv

# ----------------------------
# Supported algorithms
# ----------------------------
ALGOS: Dict[str, type] = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def is_vecenv(env) -> bool:
    """Return True if the environment is a VecEnv-like wrapper."""
    return hasattr(env, "num_envs")


def _calc_completion_ratio(
    end_coins: Optional[int], total_coins: Optional[int]
) -> Optional[float]:
    """
    Compute completion ratio in [0, 1] if both values are present and total > 0.
    Returns None if the ratio cannot be computed.
    """
    if total_coins is None or end_coins is None or total_coins <= 0:
        return None
    return 1.0 - float(end_coins) / float(total_coins)


def build_eval_env(algo: str, obs_mode: str, seed: int, vecnorm_path: str | None, frame_stack_dqn: int = 4):
    """
    If `vecnorm_path` exists => create DummyVecEnv(1) + load VecNormalize and disable training.
    (Observations are normalized; rewards are NOT normalized to keep comparable metrics.)
    """
    base = DummyVecEnv(
        [lambda: Monitor(SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=seed))]
    )

    if algo.lower() == "dqn" and frame_stack_dqn and frame_stack_dqn > 1:
        base = VecFrameStack(base, n_stack=frame_stack_dqn)

    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, base)
        venv.training = False
        venv.norm_reward = False
        print(f"[INFO] VecNormalize cargado: {vecnorm_path}")
        return venv

    print("[WARN] Sin VecNormalize (.pkl no encontrado). Eval sin normalizar -> rendimiento menor.")

    return base


def evaluate_model(
    algo: str,
    model_path: str,
    obs_mode: str,
    episodes: int,
    seed: int = 123,
    vecnorm_path: str | None = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate a saved model for a given number of episodes, computing aggregate metrics.

    Returns:
        mean_reward, std_reward, mean_len, success_rate, completion_ratio_mean, near_clear_rate
    """
    env = build_eval_env(algo, obs_mode, seed, vecnorm_path)
    ModelCls = ALGOS[algo]
    model = ModelCls.load(model_path, env=env, device="cpu")

    use_vec = is_vecenv(env)

    rewards: List[float] = []
    lengths: List[int] = []
    success: List[int] = []
    ratios: List[float] = []
    near = 0

    for ep in range(episodes):
        # Reset + init info
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

        ep_r = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)

            if use_vec:
                # VecEnv: action shape (1,)
                if not isinstance(action, np.ndarray):
                    action = np.array([int(action)], dtype=np.int64)
                obs, r, done, infos = env.step(action)
                r = float(np.asarray(r).squeeze()[()])  # robust scalar extraction
                ep_r += r
                steps += 1
                if bool(np.asarray(done).squeeze()[()]):
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos
                    end_coins = info.get("coins_remaining", None)

                    ratio = _calc_completion_ratio(end_coins, total_coins)
                    if ratio is not None:
                        ratios.append(ratio)
                        if ratio >= 0.90:
                            near += 1

                    rewards.append(ep_r)
                    lengths.append(steps)
                    success.append(1 if end_coins == 0 else 0)
                    break
            else:
                obs, r, terminated, truncated, info = env.step(int(action))
                ep_r += float(r)
                steps += 1
                if terminated or truncated:
                    end_coins = info.get("coins_remaining", None)
                    total = info.get("coins_total", None)

                    ratio = _calc_completion_ratio(end_coins, total)
                    if ratio is not None:
                        ratios.append(ratio)
                        if ratio >= 0.90:
                            near += 1

                    rewards.append(ep_r)
                    lengths.append(steps)
                    success.append(1 if end_coins == 0 else 0)
                    break

    mean_r = float(np.mean(rewards)) if rewards else 0.0
    std_r = float(np.std(rewards)) if rewards else 0.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    succ_rate = float(np.mean(success)) if success else 0.0
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    near_rate = float(near / episodes)

    return mean_r, std_r, mean_len, succ_rate, mean_ratio, near_rate


def parse_index_or_glob(index_csv: str) -> List[dict]:
    """
    Use the CSV index if present; otherwise, glob models for PPO/A2C/DQN.
    Returns a list of dicts with: algo, obs_mode, seed, model_path.
    """
    if os.path.exists(index_csv):
        with open(index_csv, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    rows: List[dict] = []
    patterns: Sequence[Tuple[str, str, str]] = [
        ("ppo", r"pacman_ppo_(?P<obs>.+)_seed(?P<seed>\d+)\.zip$", "pacman_ppo_*_seed*.zip"),
        ("a2c", r"pacman_a2c_(?P<obs>.+)_seed(?P<seed>\d+)\.zip$", "pacman_a2c_*_seed*.zip"),
        ("dqn", r"pacman_dqn_(?P<obs>.+)_seed(?P<seed>\d+)\.zip$", "pacman_dqn_*_seed*.zip"),
    ]
    for algo, regex, glob_pat in patterns:
        pat = re.compile(regex)
        for path in glob.glob(os.path.join("models", glob_pat)):
            name = os.path.basename(path)
            m = pat.match(name)
            if not m:
                continue

            rows.append(
                {
                    "algo": algo,
                    "obs_mode": m.group("obs"),
                    "seed": m.group("seed"),
                    "model_path": path,
                }
            )
    return rows


def maybe_vecnorm_path(row: dict) -> str | None:
    """
    If the index contains 'vecnorm_path', use it; otherwise, try common variants:
      models/vecnorm_{obs}_seed{seed}.pkl
      models/vecnorm_{obs}_seed{seed}_{algo}.pkl
    Return the first existing path, or None if none exist.
    """
    if "vecnorm_path" in row and row["vecnorm_path"]:
        return row["vecnorm_path"]

    obs_mode = row["obs_mode"]
    seed = int(row["seed"])
    algo = row.get("algo", "").lower()

    candidates = [
        os.path.join("models", f"vecnorm_{obs_mode}_seed{seed}.pkl"),
        os.path.join("models", f"vecnorm_{obs_mode}_seed{seed}_{algo}.pkl"),
        # In case it was saved with a different suffix:
        os.path.join("models", f"vecnorm_{obs_mode}_seed{seed}_ppo.pkl"),
        os.path.join("models", f"vecnorm_{obs_mode}_seed{seed}_a2c.pkl"),
        os.path.join("models", f"vecnorm_{obs_mode}_seed{seed}_dqn.pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _write_metrics_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    """Write a metrics CSV to `path`, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="experiments/run_index.csv")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", type=str, default="experiments/metrics.csv")
    args = ap.parse_args()

    rows = parse_index_or_glob(args.index)
    if not rows:
        print("[WARN] No hay modelos. Ejecuta train_* primero.")
        return

    header = [
        "algo",
        "obs_mode",
        "seed",
        "episodes",
        "mean_reward",
        "std_reward",
        "mean_len",
        "success_rate",
        "completion_ratio",
        "near_clear_rate",
        "model_path",
        "vecnorm_used",
    ]

    # Collect results to write once (global) and also per algorithm
    global_rows: List[List[object]] = []
    per_algo: Dict[str, List[List[object]]] = {}

    for r in rows:
        algo = r["algo"].lower()
        obs_mode = r["obs_mode"]
        seed = int(r["seed"])
        model_path = r["model_path"]

        if not os.path.exists(model_path):
            print(f"[WARN] no existe {model_path}, salto")
            continue

        vecnorm = maybe_vecnorm_path(r)
        print(f"[EVAL] {algo} | {obs_mode} | seed={seed} | vecnorm={'yes' if vecnorm else 'no'}")

        mean_r, std_r, mean_len, succ, mean_ratio, near_rate = evaluate_model(
            algo, model_path, obs_mode, episodes=args.episodes, seed=123, vecnorm_path=vecnorm
        )

        row_out = [
            algo,
            obs_mode,
            seed,
            args.episodes,
            mean_r,
            std_r,
            mean_len,
            succ,
            mean_ratio,
            near_rate,
            model_path,
            ("yes" if vecnorm else "no"),
        ]
        global_rows.append(row_out)
        per_algo.setdefault(algo, []).append(row_out)

    # (1) Write global metrics (backwards compatible)
    _write_metrics_csv(args.out, header, global_rows)
    print(f"[OK] métricas en {args.out}")

    # (2) Write per-algorithm metrics under experiments/<algo>/metrics.csv
    base_dir = os.path.dirname(args.out) or "."
    for algo, rows_list in per_algo.items():
        algo_dir = os.path.join(base_dir, algo)
        algo_out = os.path.join(algo_dir, os.path.basename(args.out))  # e.g., experiments/ppo/metrics.csv
        _write_metrics_csv(algo_out, header, rows_list)
        print(f"[OK] métricas por-algoritmo en {algo_out}")


if __name__ == "__main__":
    main()
