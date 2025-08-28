from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import List


def main() -> None:
    """
    Launch multiple training runs for different observation modes and seeds,
    then append their model metadata to an index CSV.

    Notes:
      - Behavior is intentionally preserved:
        * The '--algo' flag is accepted but the index/model naming remains 'ppo'.
        * Training is invoked via 'scripts/train_simple.py' without passing '--algo'.
        * The index file is appended to; header is written only if the file does not exist.
      - Console messages remain in Spanish to match the original script.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--obs-modes",
        type=str,
        nargs="+",
        default=["minimal", "bool_power", "power_time", "coins_quadrants"],
    )
    ap.add_argument("--algo", type=str, default="ppo")  # kept for CLI compatibility
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--models-dir", type=str, default="models")
    ap.add_argument("--out-index", type=str, default="experiments/run_index.csv")
    args = ap.parse_args()

    # Preserve original directory creation behavior (including dirname of out_index).
    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    rows: List[List[object]] = []
    for mode in args.obs_modes:
        for seed in args.seeds:
            cmd = [
                args.python,
                os.path.join("scripts", "train_simple.py"),
                "--timesteps",
                str(args.timesteps),
                "--obs-mode",
                mode,
                "--seed",
                str(seed),
            ]
            print("[RUN]", " ".join(cmd), flush=True)
            ret = subprocess.call(cmd)
            if ret != 0:
                print(
                    f"[WARN] entrenamiento falló para {mode} seed {seed} (ret={ret})",
                    file=sys.stderr,
                )
                continue

            # Keep the original naming convention (always 'ppo' in the filename).
            model_path = os.path.join(args.models_dir, f"pacman_ppo_{mode}_seed{seed}.zip")
            rows.append(["ppo", mode, seed, model_path])

    write_header = not os.path.exists(args.out_index)
    with open(args.out_index, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
