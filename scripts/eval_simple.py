from __future__ import annotations
import argparse, statistics, os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.simple_pacman import SimplePacmanEnv, ObsConfig  # 👈 ojo con el import


def as_scalar(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.item() if x.ndim == 0 else x.squeeze()[0].item()
    if hasattr(x, "item"):
        return x.item()
    return x


def is_vecenv(env) -> bool:
    # VecNormalize y Dummy/SubprocVecEnv tienen atributo 'num_envs'
    return hasattr(env, "num_envs")


def build_env(obs_mode: str, seed: int, vecnorm_path: str | None):
    """
    Si vecnorm_path existe => crea DummyVecEnv(1) + carga VecNormalize y desactiva el
    entrenamiento (obs normalizadas, reward sin normalizar para reportar).
    Si no => devuelve el env nativo (no vectorizado).
    """
    if vecnorm_path and os.path.exists(vecnorm_path):
        base = DummyVecEnv([lambda: Monitor(SimplePacmanEnv(ObsConfig(mode=obs_mode), seed=seed))])
        venv = VecNormalize.load(vecnorm_path, base)
        venv.training = False
        venv.norm_reward = False
        return venv  # VecEnv
    else:
        return SimplePacmanEnv(obs_config=ObsConfig(mode=obs_mode), seed=seed)  # env normal


def get_init_info(env, seed_base: int, ep_idx: int, vec: bool):
    """
    Devuelve (obs, info_dict, init_coins).
    Con VecEnv no hay 'info' en reset; usamos get_attr para leer el estado.
    """
    if vec:
        obs = env.reset()  # np.array shape (1, obs_dim)
        try:
            init_coins = int(env.get_attr("coins_remaining")[0])
        except Exception:
            init_coins = None
        info = {"coins_remaining": init_coins,
                "coins_total": int(env.get_attr("coins_total")[0]) if init_coins is not None else None}
        return obs, info, init_coins
    else:
        obs, info = env.reset(seed=seed_base + ep_idx)
        init_coins = info.get("coins_remaining", None)
        return obs, info, init_coins


def step_once(env, obs, action, vec: bool):
    """
    Unifica la transición para env normal (Gymnasium) y VecEnv.
    Devuelve: obs, reward(float), terminated(bool), truncated(bool), info(dict)
    """
    if vec:
        # con VecEnv, model.predict ya devuelve acción shape (1,)
        if not isinstance(action, np.ndarray):
            action = np.array([int(as_scalar(action))], dtype=np.int64)
        obs, r, done, infos = env.step(action)
        r = float(np.asarray(r).squeeze()[()])
        done = bool(np.asarray(done).squeeze()[()])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        # si el entorno añade flags, los usamos; si no, distinguimos por coins
        terminated = bool(info.get("terminated", done))
        truncated = bool(info.get("truncated", False))
        return obs, r, terminated, truncated, info
    else:
        # entorno normal Gymnasium (devuelve 5 valores)
        action = int(as_scalar(action))
        return env.step(action)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--obs-mode", default="minimal",
                    choices=["minimal", "bool_power", "power_time", "coins_quadrants"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--vecnorm", type=str, default=None,
                    help="Ruta al .pkl de VecNormalize (ej: models/vecnorm_minimal_seed0.pkl)")
    args = ap.parse_args()

    env = build_env(args.obs_mode, args.seed, args.vecnorm)
    model = PPO.load(args.model, env=env, device="cpu")

    use_vec = is_vecenv(env)

    rewards, lengths, success = [], [], []
    ratios, coins_list, powers, ghosts = [], [], [], []
    deaths = 0
    wins = 0
    trunc_count = 0
    near_clear = 0  # ≥90% monedas

    for ep in range(args.episodes):
        # reset (maneja ambos tipos de env)
        obs, info, init_coins = get_init_info(env, args.seed, ep, vec=use_vec)
        ep_r, steps = 0.0, 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = step_once(env, obs, action, vec=use_vec)

            ep_r += float(r); steps += 1

            if terminated or truncated:
                end_coins = info.get("coins_remaining", None)
                total_coins = info.get("coins_total", None)

                # % completado
                if (total_coins is not None) and (end_coins is not None) and (total_coins > 0):
                    ratio = 1.0 - float(end_coins) / float(total_coins)
                    ratios.append(ratio)
                    if ratio >= 0.90:
                        near_clear += 1

                # monedas recogidas
                if init_coins is not None and end_coins is not None:
                    coins_list.append(init_coins - end_coins)

                # causa de finalización
                if terminated:
                    if end_coins == 0:
                        wins += 1
                    else:
                        deaths += 1
                elif truncated:
                    trunc_count += 1
                else:
                    # en VecEnv sin flags, contamos como "otros" (no debería pasar)
                    pass

                # contadores de power si el entorno los expone
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

    print(f"✅ Evaluación: {args.episodes} episodios")
    print(f"   Recompensa media = {mean_r:.2f} ± {std_r:.2f}")
    print(f"   Pasos medios     = {mean_len:.1f}")
    print(f"   Éxito (todas monedas) = {succ_rate:.2%}")
    if powers:  print(f"   Powers recogidos (media) = {np.mean(powers):.2f}")
    if ghosts:  print(f"   Fantasmas comidos (media) = {np.mean(ghosts):.2f}")
    if ratios:
        print(f"   % nivel completado (medio) = {100.0 * sum(ratios)/len(ratios):.1f}%")
        print(f"   Éxito 90%+ = {near_clear/args.episodes:.2%}")
    if coins_list:
        print(f"   Monedas recogidas (media) = {sum(coins_list)/len(coins_list):.1f}")
    print(f"   Terminados (muerte/clear): {deaths}/{wins} | Por tiempo: {trunc_count}")


if __name__ == "__main__":
    main()
