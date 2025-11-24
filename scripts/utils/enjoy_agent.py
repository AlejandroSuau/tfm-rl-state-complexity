# scripts/enjoy_agent.py
from __future__ import annotations

import argparse
import time
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from envs.simple_pacman import SimplePacmanEnv, ObsConfig


ALGOS = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C
}


def make_env(obs_mode: str, render_mode: str):
    """
    Crea el entorno con el modo de observación y render deseado.
    render_mode: "ansi" (texto) o "rgb_array" (imagen)
    """
    def _init():
        cfg = ObsConfig(mode=obs_mode)
        env = SimplePacmanEnv(obs_config=cfg, render_mode=render_mode)
        return env
    return _init


def render_ansi(env):
    """
    Renderización simple en texto.
    """
    out = env.render()
    if isinstance(out, str):
        print(out)
    else:
        print("Render devolvió un objeto no textual:", type(out))


def run_ascii(venv, model, episodes: int, sleep: float):
    """
    Bucle de evaluación con render ascii.
    """
    for ep in range(episodes):
        obs = venv.reset()
        done = False
        ep_reward = 0.0
        step = 0

        print(f"\n=== Episodio {ep + 1} ===")
        render_ansi(venv.envs[0])

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            done = bool(done[0])

            ep_reward += float(reward[0])
            step += 1

            render_ansi(venv.envs[0])
            print(f"Step: {step} | Reward: {float(reward[0]):.2f} | Done: {done}")

            time.sleep(sleep)

        print(f"Recompensa total episodio {ep + 1}: {ep_reward:.2f}")


def run_rgb(venv, model, episodes: int, sleep: float):
    """
    Bucle de evaluación con render rgb usando matplotlib.
    """
    # Preparamos la ventana de matplotlib en el primer episodio
    obs = venv.reset()
    frame = venv.envs[0].render()
    if frame is None:
        raise RuntimeError(
            "render() devolvió None. Asegúrate de que el entorno usa render_mode='rgb_array'."
        )

    fig, ax = plt.subplots()
    im = ax.imshow(frame, interpolation="nearest")
    ax.set_title("SimplePacman - Episodio 1")
    plt.ion()
    plt.show()

    for ep in range(episodes):
        if ep > 0:
            # reset para episodios posteriores
            obs = venv.reset()
            frame = venv.envs[0].render()
            im.set_data(frame)
            ax.set_title(f"SimplePacman - Episodio {ep + 1}")
            fig.canvas.draw()
            fig.canvas.flush_events()

        done = False
        ep_reward = 0.0
        step = 0
        print(f"\n=== Episodio {ep + 1} ===")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            done = bool(done[0])
            ep_reward += float(reward[0])
            step += 1

            # Actualizar frame
            frame = venv.envs[0].render()
            im.set_data(frame)
            ax.set_title(
                f"Episodio {ep + 1} | Step {step} | Reward acum: {ep_reward:.1f}"
            )
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(sleep)

        print(f"Recompensa total episodio {ep + 1}: {ep_reward:.2f}")

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(ALGOS.keys()),
        help="Algoritmo con el que se entrenó el modelo.",
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Ruta al modelo .zip")
    parser.add_argument("--vecnorm-path", type=str, default=None,
                        help="Ruta al VecNormalize (.pkl) si se usó.")
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="minimal",
        choices=["minimal", "bool_power", "power_time", "coins_quadrants", "image"],
        help="Modo de observación con el que se entrenó el modelo.",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help="Número de frames apilados (igual que en entrenamiento, p.ej. 4).",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument(
        "--render",
        type=str,
        default="ascii",
        choices=["ascii", "rgb"],
        help="Tipo de renderizado: 'ascii' (texto) o 'rgb' (imagen con matplotlib).",
    )

    args = parser.parse_args()

    # Elegir clase de modelo
    ModelClass = ALGOS[args.algo.lower()]

    # Determinar render_mode para el entorno
    render_mode = "ansi" if args.render == "ascii" else "rgb_array"

    # 1) Crear entorno base vectorizado
    venv = DummyVecEnv([make_env(args.obs_mode, render_mode)])

    # 2) Aplicar frame stacking si procede (como en entrenamiento)
    if args.frame_stack is not None and args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack)

    # 3) Cargar VecNormalize si existe (mismo orden que en train_ppo)
    if args.vecnorm_path is not None:
        venv = VecNormalize.load(args.vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    # 4) Cargar modelo
    print(f"Cargando modelo {args.algo.upper()} desde: {args.model_path}")
    model = ModelClass.load(args.model_path, env=venv)

    # 5) Lanzar modo solicitado
    if args.render == "ascii":
        run_ascii(venv, model, args.episodes, args.sleep)
    else:
        run_rgb(venv, model, args.episodes, args.sleep)


if __name__ == "__main__":
    main()
