import subprocess
import itertools

algos = ["ppo", "dqn"]
modes = ["minimal", "bool_power", "power_time", "coins_quadrants"]
seeds = [0, 1, 2]

for algo, mode, seed in itertools.product(algos, modes, seeds):
    cmd = [
        "python", "scripts/train.py",
        "--algo", algo,
        "--obs-mode", mode,
        "--seed", str(seed),
        "--steps", str(200_000),
    ]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)