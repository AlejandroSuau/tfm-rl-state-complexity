from __future__ import annotations

import argparse

from envs.simple_pacman import ObsConfig, SimplePacmanEnv


def main() -> None:
    """
    Run a short random policy rollout in SimplePacmanEnv and print periodic stats.

    Behavior preserved:
      - Same CLI flags and defaults
      - Same console prints (in Spanish)
      - Random actions via env.action_space.sample()
      - Stops on termination/truncation or after N steps
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--obs-mode",
        default="minimal",
        choices=["minimal", "bool_power", "power_time", "coins_quadrants"],
    )
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = SimplePacmanEnv(obs_config=ObsConfig(mode=args.obs_mode), seed=args.seed)
    obs, info = env.reset(seed=args.seed)

    print("Obs inicial:", obs)
    print("Info inicial:", info)

    total_r = 0.0
    for t in range(args.steps):
        action = env.action_space.sample()  # acción aleatoria
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r

        if (t + 1) % 10 == 0:
            print(
                f"t={t+1} r_acum={total_r:.2f} "
                f"coins={info.get('coins_remaining')} "
                f"power={info.get('power_timer')}"
            )

        if terminated or truncated:
            print("Episodio terminó en t=", t + 1, " | reward_total=", total_r)
            break
    else:
        print("No terminó. reward_total=", total_r)


if __name__ == "__main__":
    main()
