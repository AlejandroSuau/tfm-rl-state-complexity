# scripts/plot_report.py
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rlpacman.utils.metrics import load_monitor_csv

PPO_RE = re.compile(r"ppo_(minimal|bool_power|power_time|coins_quadrants)_seed(\d+)")

def load_summary(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    df = pd.read_csv(summary_csv)
    # intenta extraer modo y seed desde 'run' si no están como columnas
    if "mode" not in df.columns:
        modes, seeds = [], []
        for r in df["run"].astype(str):
            m = PPO_RE.search(r)
            modes.append(m.group(1) if m else None)
            seeds.append(int(m.group(2)) if m else None)
        df["mode"] = modes
        df["seed"] = seeds
    return df

def find_monitor_csvs(log_root: Path, mode: str) -> list[Path]:
    out = []
    for d in log_root.glob(f"ppo_{mode}_seed*"):
        p = d / "monitor_train.csv"
        if not p.exists():
            alts = list(d.glob("monitor_train*.csv"))
            out += alts
        else:
            out.append(p)
    return out

def smooth(y, k=11):
    y = np.asarray(y)
    if k <= 1 or len(y) < k:
        return y
    w = np.ones(k)/k
    return np.convolve(y, w, mode="same")

def plot_all(summary_csv: Path, log_root: Path, out_path: Path, smooth_k=11):
    df_sum = load_summary(summary_csv)
    # Filtro PPO únicamente
    df_sum = df_sum[df_sum["run"].str.contains("ppo_")]
    modes = ["minimal", "bool_power", "power_time", "coins_quadrants"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

    # --- Panel superior: curvas mean±std por modo ---
    for mode in modes:
        csvs = find_monitor_csvs(log_root, mode)
        if not csvs:
            continue
        mats = []
        for csv in csvs:
            try:
                df = load_monitor_csv(csv)
                mats.append(df["reward"].to_numpy())
            except Exception:
                pass
        if not mats:
            continue
        n = min(len(x) for x in mats)
        M = np.stack([x[:n] for x in mats], axis=0)
        mean = M.mean(axis=0)
        std  = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean)
        x = np.arange(1, n+1, dtype=float)
        mean_s = smooth(mean, smooth_k)
        std_s  = smooth(std, smooth_k)
        # medias y banda
        ax1.plot(x, mean_s, linewidth=2.0, label=f"{mode} (n={M.shape[0]})")
        ax1.fill_between(x, mean_s-std_s, mean_s+std_s, alpha=0.15)
    ax1.set_title("PPO — Return por episodio (media ± std) en entrenamiento")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Return")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Panel inferior: barras con métricas agregadas (summary.csv) ---
    # Usamos return_mean y auc
    agg = (df_sum.dropna(subset=["mode"])
                 .groupby("mode", as_index=False)
                 .agg(mean_return=("return_mean","mean"),
                      auc=("auc","mean")))
    # ordenar por orden lógico de modos
    cat = pd.CategoricalDtype(categories=modes, ordered=True)
    agg["mode"] = agg["mode"].astype(cat)
    agg = agg.sort_values("mode")
    idx = np.arange(len(agg))
    width = 0.35
    ax2.bar(idx - width/2, agg["mean_return"].to_numpy(), width, label="mean_return")
    ax2.bar(idx + width/2, agg["auc"].to_numpy()/1000.0, width, label="auc/1000")  # escala para visualizar
    ax2.set_xticks(idx)
    ax2.set_xticklabels(agg["mode"])
    ax2.set_title("Métricas agregadas por modo (summary.csv)")
    ax2.set_ylabel("Valor (AUC reescalado /1000)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[OK] Guardado reporte: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default="experiments/analysis/summary.csv")
    ap.add_argument("--logs", type=str, default="experiments/logs")
    ap.add_argument("--out", type=str, default="experiments/analysis/ppo_report.png")
    ap.add_argument("--smooth", type=int, default=11)
    args = ap.parse_args()

    plot_all(Path(args.summary), Path(args.logs), Path(args.out), smooth_k=args.smooth)

if __name__ == "__main__":
    main()
