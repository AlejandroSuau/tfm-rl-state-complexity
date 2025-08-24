from __future__ import annotations
import os, pandas as pd, matplotlib.pyplot as plt

METRICS_CSV = "experiments/metrics.csv"
OUT_DIR = "experiments"

def barplot(df, metric: str, out_png: str, ylabel: str):
    g = df.groupby("obs_mode")[metric].mean().reset_index().sort_values(by=metric, ascending=False)
    plt.figure()
    plt.bar(g["obs_mode"], g[metric])
    plt.ylabel(ylabel); plt.xlabel("obs_mode"); plt.title(metric)
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    print("[OK] guardado:", out_png)

def main():
    if not os.path.exists(METRICS_CSV):
        raise SystemExit(f"No existe {METRICS_CSV}. Ejecuta primero eval_collect.py")
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(METRICS_CSV)
    barplot(df, "mean_reward",      os.path.join(OUT_DIR, "plot_mean_reward.png"),      "Recompensa media")
    barplot(df, "success_rate",     os.path.join(OUT_DIR, "plot_success_rate.png"),     "Tasa de éxito")
    barplot(df, "completion_ratio", os.path.join(OUT_DIR, "plot_completion_ratio.png"), "Ratio de completado (medio)")
    barplot(df, "near_clear_rate",  os.path.join(OUT_DIR, "plot_near_clear_rate.png"),  "Éxito ≥90%")
if __name__ == "__main__":
    main()
