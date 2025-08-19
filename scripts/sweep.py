import subprocess
import itertools
import os

algos = ["ppo", "dqn"]
modes = ["minimal", "bool_power", "power_time", "coins_quadrants"]
seeds = [0, 1, 2]

steps = os.environ.get("STEPS", "200000")

for algo, mode, seed in itertools.product(algos, modes, seeds):
    cmd = [
        "python", "scripts/train.py",
        "--algo", algo,
        "--obs-mode", mode,
        "--seed", str(seed),
        "--steps", steps,
    ]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)
