from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.simple_pacman import ObsConfig, SimplePacmanEnv


def as_scalar(x: Any) -> Any:
    """
    Convert common containers (list, tuple, ndarray, objects with .item()) to a scalar.
    Used to normalize actions or rewards to plain Python types.
    """
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.item() if x.ndim == 0 else x.squeeze()[0].item()
    if hasattr(x, "item"):
        return x.item()
    return x


def is_vecenv(env) -> bool:
    """Return True if the environment is a VecEnv-like wrapper (has 'num_envs')."""
    return hasattr(env, "num_envs")


def build_env(obs_mode: str, seed: int, vecnorm_path: str | None):
    """
    If vecnorm_path exists:
        - Create DummyVecEnv(1), load VecNormalize, disable training and reward normalization.
    Otherwise:
        - Return a raw SimplePacmanEnv.
    """
    if vecnorm_path and os.path.exists(vecnorm_path):
        base = DummyVecEnv(
            [lambda: Monitor(SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=seed))]
        )
        venv = VecNormalize.load(vecnorm_path, base)
        venv.training = False
        venv.norm_reward = False
        print(f"[INFO] VecNormalize cargado: {vecnorm_path}")
        return venv
    else:
        print(
            "[WARN] Sin VecNormalize (.pkl no encontrado). "
            "Eval sin normalizar -> rendimiento menor."
        )
        return SimplePacmanEnv(obs_config=ObsConfig(mode=obs_mode), seed=seed)


def get_init_info(env, seed_base: int, ep_idx: int, vec: bool) -> Tuple[Any, Dict, int | None]:
    """
    Reset the environment and return (obs, info_dict, init_coins).
    - With VecEnv: use get_attr to query state (no info dict from reset()).
    - With regular env: return the usual (obs, info).
    """
    if vec:
        obs = env.reset()
        try:
            init_coins = int(env.get_attr("coins_remaining")[0])
        except Exception:
            init_coins = None
        try:
            total_coins = int(env.get_attr("coins_total")[0])
        except Exception:
            total_coins = None
        info = {"coins_remaining": init_coins, "coins_total": total_coins}
        return obs, info, init_coins
    else:
        obs, info = env.reset(seed=seed_base + ep_idx)
        init_coins = info.get("coins_remaining", None)
        return obs, info, init_coins


def step_once(env, obs: Any, action: Any, vec: bool) -> Tuple[Any, float, bool, bool, Dict]:
    """
    Unified step function for Gymnasium envs and VecEnv.
    Returns: obs, reward (float), terminated (bool), truncated (bool), info (dict).
    """
    if vec:
        # With VecEnv, model.predict already returns action shape (1,)
        if not isinstance(action, np.ndarray):
            action = np.array([int(as_scalar(action))], dtype=np.int64)
        obs, r, done, infos = env.step(action)
        r = float(np.asarray(r).squeeze()[()])
        done = bool(np.asarray(done).squeeze()[()])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        terminated = bool(info.get("terminated", done))
        truncated = bool(info.get("TimeLimit.truncated", info.get("truncated", False)))
        return obs, r, terminated, truncated, info
    else:
        # Regular Gymnasium env (returns 5 values)
        action = int(as_scalar(action))
        return env.step(action)


def _infer_algo_from_model_path(model_path: str) -> str:
    """Infer algo name from filename (ppo/a2c/dqn). Return 'unknown' if not matched."""
    name = os.path.basename(model_path).lower()
    if "pacman_ppo_" in name:
        return "ppo"
    if "pacman_a2c_" in name:
        return "a2c"
    if "pacman_dqn_" in name:
        return "dqn"
    return "unknown"


def _save_eval_json(
    out_base: str,
    algo: str,
    payload: Dict[str, Any],
) -> str:
    """
    Save evaluation payload under experiments/<algo>/ with a timestamped filename.
    Returns the written path.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    algo_dir = os.path.join(out_base, algo)
    os.makedirs(algo_dir, exist_ok=True)
    out_path = os.path.join(algo_dir, f"eval_{algo}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument(
        "--obs-mode",
        default="minimal",
        choices=["minimal", "bool_power", "power_time", "coins_quadrants"],
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="Ruta al .pkl de VecNormalize (ej: models/vecnorm_minimal_seed0.pkl)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="experiments",
        help="Directorio base donde guardar el JSON de evaluación (por algoritmo).",
    )
    args = ap.parse_args()

    env = build_env(args.obs_mode, args.seed, args.vecnorm)
    model = PPO.load(args.model, env=env, device="cpu")

    use_vec = is_vecenv(env)

    rewards, lengths, success = [], [], []
    ratios, coins_list, powers, ghosts = [], [], [], []
    deaths = 0
    wins = 0
    trunc_count = 0
    near_clear = 0  # ≥90% coins collected

    for ep in range(args.episodes):
        # Reset (handles both VecEnv and native env)
        obs, info, init_coins = get_init_info(env, args.seed, ep, vec=use_vec)
        ep_r, steps = 0.0, 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = step_once(env, obs, action, vec=use_vec)

            ep_r += float(r)
            steps += 1

            if terminated or truncated:
                end_coins = info.get("coins_remaining", None)
                total_coins = info.get("coins_total", None)

                # Completion %
                if (
                    total_coins is not None
                    and end_coins is not None
                    and total_coins > 0
                ):
                    ratio = 1.0 - float(end_coins) / float(total_coins)
                    ratios.append(ratio)
                    if ratio >= 0.90:
                        near_clear += 1

                # Coins collected
                if init_coins is not None and end_coins is not None:
                    coins_list.append(init_coins - end_coins)

                # Cause of termination
                if terminated:
                    if end_coins == 0:
                        wins += 1
                    else:
                        deaths += 1
                elif truncated:
                    trunc_count += 1

                # Track powerups/ghosts if exposed
                powers.append(info.get("powers_picked", 0))
                ghosts.append(info.get("ghosts_eaten", 0))

                rewards.append(ep_r)
                lengths.append(steps)
                success.append(1 if (end_coins == 0) else 0)
                break

    mean_r = statistics.mean(rewards) if rewards else 0.0
    std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    mean_len = statistics.mean(lengths) if lengths else 0.0
    succ_rate = (sum(success) / len(success)) if success else 0.0

    # Console output (kept in Spanish)
    print(f"✅ Evaluación: {args.episodes} episodios")
    print(f"   Recompensa media = {mean_r:.2f} ± {std_r:.2f}")
    print(f"   Pasos medios     = {mean_len:.1f}")
    print(f"   Éxito (todas monedas) = {succ_rate:.2%}")
    if powers:
        print(f"   Powers recogidos (media) = {np.mean(powers):.2f}")
    if ghosts:
        print(f"   Fantasmas comidos (media) = {np.mean(ghosts):.2f}")
    if ratios:
        print(f"   % nivel completado (medio) = {100.0 * sum(ratios)/len(ratios):.1f}%")
        print(f"   Éxito 90%+ = {near_clear/args.episodes:.2%}")
    if coins_list:
        print(f"   Monedas recogidas (media) = {sum(coins_list)/len(coins_list):.1f}")
    print(f"   Terminados (muerte/clear): {deaths}/{wins} | Por tiempo: {trunc_count}")

    # -------- Persist evaluation summary under experiments/<algo>/ --------
    algo = _infer_algo_from_model_path(args.model)
    payload = {
        "algo": algo,
        "model_path": args.model,
        "obs_mode": args.obs_mode,
        "seed": args.seed,
        "episodes": args.episodes,
        "vecnorm_path": args.vecnorm,
        "metrics": {
            "mean_reward": mean_r,
            "std_reward": std_r,
            "mean_len": mean_len,
            "success_rate": succ_rate,
            "completion_ratio_mean": (float(np.mean(ratios)) if ratios else 0.0),
            "near_clear_rate": (float(near_clear / args.episodes) if args.episodes else 0.0),
            "deaths": deaths,
            "wins": wins,
            "truncations": trunc_count,
            "coins_collected_mean": (float(np.mean(coins_list)) if coins_list else 0.0),
            "powers_mean": (float(np.mean(powers)) if powers else 0.0),
            "ghosts_mean": (float(np.mean(ghosts)) if ghosts else 0.0),
        },
    }
    out_path = _save_eval_json(args.out_dir, algo, payload)
    print(f"[OK] Resumen de evaluación guardado en: {out_path}")


if __name__ == "__main__":
    main()
