# scripts/plot_ppo_modes.py
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rlpacman.utils.metrics import load_monitor_csv

PPO_DIR_RE = re.compile(r"ppo_(minimal|bool_power|power_time|coins_quadrants)_seed(\d+)")

def gather_runs(log_root: Path):
    runs = {}  # mode -> list[(seed, df)]
    for run_dir in log_root.glob("ppo_*"):
        m = PPO_DIR_RE.fullmatch(run_dir.name)
        if not m:
            continue
        mode, seed = m.group(1), int(m.group(2))
        # preferimos monitor_train.csv (curva de entrenamiento)
        csv_path = run_dir / "monitor_train.csv"
        if not csv_path.exists():
            # algunos setups añaden sufijo .monitor.csv
            alt = list(run_dir.glob("monitor_train*.csv"))
            if not alt:
                continue
            csv_path = alt[0]
        try:
            df = load_monitor_csv(csv_path)
        except Exception:
            continue
        runs.setdefault(mode, []).append((seed, df))
    return runs

def smooth(x: np.ndarray, k: int = 11):
    if k <= 1 or len(x) < k:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def plot_modes(runs: dict[str, list[tuple[int, pd.DataFrame]]], out_path: Path, smooth_k: int = 11):
    plt.figure(figsize=(10, 6))
    modes_order = ["minimal", "bool_power", "power_time", "coins_quadrants"]
    for mode in modes_order:
        if mode not in runs:
            continue
        seeds = sorted([s for s,_ in runs[mode]])
        # Alinear por número de episodios mínimo común entre seeds
        lens = [len(df) for _, df in runs[mode]]
        if not lens:
            continue
        n = min(lens)
        # Matriz [seeds, n] de returns por episodio
        mat = np.stack([df["reward"].to_numpy()[:n] for _, df in sorted(runs[mode], key=lambda x: x[0])], axis=0)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
        x = np.arange(1, n+1, dtype=float)
        mean_s = smooth(mean, smooth_k)
        std_s  = smooth(std, smooth_k)
        # líneas por seed (finas)
        for i in range(mat.shape[0]):
            plt.plot(x, smooth(mat[i], smooth_k), linewidth=0.8, alpha=0.35)
        # media con banda de desviación
        plt.plot(x, mean_s, linewidth=2.0, label=f"{mode} (n={len(seeds)})")
        plt.fill_between(x, mean_s-std_s, mean_s+std_s, alpha=0.15)
    plt.xlabel("Episodio")
    plt.ylabel("Return por episodio (entrenamiento)")
    plt.title("PPO — Return vs Episodio por modo de observación (media ± std)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[OK] Guardado gráfico: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=str, default="experiments/logs")
    ap.add_argument("--out", type=str, default="experiments/analysis/ppo_modes_curves.png")
    ap.add_argument("--smooth", type=int, default=11)
    args = ap.parse_args()

    runs = gather_runs(Path(args.logs))
    if not runs:
        print("No se encontraron runs de PPO en", args.logs)
        return
    plot_modes(runs, Path(args.out), smooth_k=args.smooth)

if __name__ == "__main__":
    main()
