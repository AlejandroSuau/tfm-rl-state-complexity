from __future__ import annotations
import argparse, statistics
import numpy as np
from evs.simple_pacman import SimplePacmanEnv, ObsConfig
from stable_baselines3 import PPO

def as_scalar(x):
    """Convierte arrays/listas/np.scalars a escalar Python."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.item() if x.ndim == 0 else x.squeeze()[0].item()
    if hasattr(x, "item"):
        return x.item()
    return x

def first_info(info):
    """Extrae el primer diccionario si viene como lista (VecEnv 1×)."""
    if isinstance(info, (list, tuple)):
        return info[0]
    return info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--obs-mode", default="minimal",
                    choices=["minimal","bool_power","power_time","coins_quadrants"])
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    env = SimplePacmanEnv(obs_config=ObsConfig(mode=args.obs_mode), seed=args.seed)
    model = PPO.load(args.model, env=env)

    rewards, lengths, success = [], [], []
    ratios, coins_list = [], []
    deaths = 0
    wins = 0
    trunc_count = 0
    near_clear = 0  # >=90% monedas

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_r, steps = 0.0, 0
        init_coins = info.get("coins_remaining", None)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = int(as_scalar(action))

            obs, r, terminated, truncated, info = env.step(action)

            # normaliza tipos
            terminated = bool(as_scalar(terminated))
            truncated = bool(as_scalar(truncated))
            info = first_info(info)

            ep_r += float(as_scalar(r))
            steps += 1

            if terminated or truncated:
                end_coins = info.get("coins_remaining", None)
                total_coins = info.get("coins_total", None)

                # % nivel completado (float)
                if (total_coins is not None) and (end_coins is not None) and (total_coins > 0):
                    ratio = 1.0 - float(end_coins) / float(total_coins)
                    ratios.append(ratio)
                    if ratio >= 0.90:
                        near_clear += 1

                # monedas recogidas absolutas
                if init_coins is not None and end_coins is not None:
                    coins_collected = init_coins - end_coins
                    coins_list.append(coins_collected)

                # causa de fin (¡solo contamos una vez aquí!)
                if terminated:
                    if end_coins == 0:
                        wins += 1
                    else:
                        deaths += 1
                elif truncated:
                    trunc_count += 1

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
    if ratios:
        print(f"   % nivel completado (medio) = {100.0 * sum(ratios)/len(ratios):.1f}%")
        print(f"   Éxito 90%+ = {near_clear/args.episodes:.2%}")
    if coins_list:
        print(f"   Monedas recogidas (media) = {sum(coins_list)/len(coins_list):.1f}")
    print(f"   Terminados (muerte/clear): {deaths}/{wins} | Por tiempo: {trunc_count}")

if __name__ == "__main__":
    main()
