from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

METRICS_CSV = "experiments/metrics.csv"
OUT_DIR = "experiments"


def barplot(df: pd.DataFrame, metric: str, out_png: str, ylabel: str) -> None:
    """
    Plot the mean of a metric grouped by 'obs_mode' and save it as a PNG.

    Preserves the original behavior:
      - mean over groups
      - sort descending by the metric
      - default Matplotlib styling
      - same labels, title, rotation, and dpi
    """
    grouped = (
        df.groupby("obs_mode")[metric]
        .mean()
        .reset_index()
        .sort_values(by=metric, ascending=False)
    )

    plt.figure()
    plt.bar(grouped["obs_mode"], grouped[metric])
    plt.ylabel(ylabel)
    plt.xlabel("obs_mode")
    plt.title(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("[OK] guardado:", out_png)


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist (no-op if it does)."""
    os.makedirs(path, exist_ok=True)


def make_all_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Generate the standard set of plots into the given output directory."""
    ensure_dir(out_dir)
    barplot(df, "mean_reward",      os.path.join(out_dir, "plot_mean_reward.png"),      "Recompensa media")
    barplot(df, "success_rate",     os.path.join(out_dir, "plot_success_rate.png"),     "Tasa de éxito")
    barplot(df, "completion_ratio", os.path.join(out_dir, "plot_completion_ratio.png"), "Ratio de completado (medio)")
    barplot(df, "near_clear_rate",  os.path.join(out_dir, "plot_near_clear_rate.png"),  "Éxito ≥90%")


def main() -> None:
    """
    Read metrics CSV and generate bar plots:
      1) Overall plots in experiments/ (backwards compatible)
      2) Per-algorithm plots under experiments/<algo>/ if 'algo' column is present
    """
    if not os.path.exists(METRICS_CSV):
        raise SystemExit(f"No existe {METRICS_CSV}. Ejecuta primero eval_collect.py")

    ensure_dir(OUT_DIR)
    df = pd.read_csv(METRICS_CSV)

    make_all_plots(df, OUT_DIR)

    if "algo" in df.columns:
        algos = sorted(a for a in df["algo"].dropna().unique())
        for algo in algos:
            sub_df = df[df["algo"] == algo]
            algo_dir = os.path.join(OUT_DIR, str(algo))
            make_all_plots(sub_df, algo_dir)


if __name__ == "__main__":
    main()
