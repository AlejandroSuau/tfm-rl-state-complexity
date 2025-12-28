#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Orden en el eje X (ajusta si quieres otro)
OBS_ORDER = ["minimal", "bool_power", "power_time", "coins_quadrants"]

# Paleta pastel (consistente y “bonita”)
PASTELS = [
    "#A3C4F3",  # azul suave
    "#B8E0D2",  # verde agua
    "#FFCFD2",  # rosa suave
    "#FDE2A7",  # amarillo suave
    "#CDB4DB",  # lila suave
    "#CDEAC0",  # verde suave
]

REQUIRED_COLS = ["algo", "obs_mode", "seed", "mean_reward", "std_reward", "completion_ratio"]

# ---------- Etiquetas “humanas” (Optuna budget) ----------
# A2C: seed10=40 trials 300k, seed20=40 trials 1M, seed30=200 trials 1M
A2C_LABELS = {
    10: "Optuna: 40 trials · 300k steps",
    20: "Optuna: 40 trials · 1M steps",
    30: "Optuna: 200 trials · 1M steps",
}

# PPO: seed10=40 trials 300k, seed30=200 trials 1M
PPO_LABELS = {
    10: "Optuna: 40 trials · 300k steps",
    30: "Optuna: 200 trials · 1M steps",
}

# “Experimentos finales” (5M steps) — selección fija por tu criterio
# A2C: seed20, excepto coins_quadrants seed10
FINAL_PICK = {
    ("a2c", "minimal"): 20,
    ("a2c", "bool_power"): 20,
    ("a2c", "power_time"): 20,
    ("a2c", "coins_quadrants"): 10,

    ("ppo", "minimal"): 30,
    ("ppo", "bool_power"): 30,
    ("ppo", "power_time"): 30,
    ("ppo", "coins_quadrants"): 30,

    ("dqn", "minimal"): 50,
    ("dqn", "bool_power"): 50,
    ("dqn", "power_time"): 50,
    ("dqn", "coins_quadrants"): 50,
}


def find_metrics_csvs(experiments_dir: Path) -> list[Path]:
    # experiments/<algo>/<algo>_all_metrics.csv  (también soporta *all_metrics.csv)
    return sorted(experiments_dir.glob("*/*_all_metrics.csv"))


def load_all_csvs(csv_paths: Iterable[Path]) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["__source_csv__"] = str(p)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No se encontraron CSVs tipo experiments/<algo>/*_all_metrics.csv")

    out = pd.concat(dfs, ignore_index=True)

    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    out["seed"] = out["seed"].astype(int)
    for c in ["mean_reward", "std_reward", "completion_ratio"]:
        out[c] = out[c].astype(float)

    out["obs_mode"] = pd.Categorical(out["obs_mode"], categories=OBS_ORDER, ordered=True)
    return out


def compute_text_offset(values: np.ndarray, metric: str) -> float:
    """Separación vertical de anotaciones para que se lean bien."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    if metric == "completion_ratio":
        return 0.02  # en [0,1], 0.02 se ve bien

    vmin, vmax = float(values.min()), float(values.max())
    span = max(vmax - vmin, 1e-6)
    return 0.03 * span  # 3% del rango


def grouped_bar(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    group_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    group_label_map: dict | None = None,
    ylim=None,
    note: str | None = None,
):
    # x_vals requiere categorical para mantener orden
    x_vals = list(data[x_col].cat.categories)
    groups = [g for g in sorted(data[group_col].dropna().unique())]

    x = np.arange(len(x_vals), dtype=float)
    width = 0.8 / max(len(groups), 1)

    # offsets para anotaciones
    all_y = data[y_col].to_numpy(dtype=float)
    text_off = compute_text_offset(all_y, y_col)

    for i, g in enumerate(groups):
        d = data[data[group_col] == g].set_index(x_col)

        y = np.array([d.loc[xv, y_col] if xv in d.index else np.nan for xv in x_vals], dtype=float)
        pos = x - 0.4 + width / 2 + i * width

        color = PASTELS[i % len(PASTELS)]
        bars = ax.bar(pos, y, width=width, label=str(g), color=color, edgecolor="white", linewidth=0.8)

        # Anotaciones
        for rect, py in zip(bars, y):
            if not np.isfinite(py):
                continue
            if y_col == "completion_ratio":
                txt = f"{py*100:.1f}%"
            else:
                txt = f"{py:.2f}"

            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                py + text_off,
                txt,
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Leyenda con nombres “humanos”
    handles, labels = ax.get_legend_handles_labels()
    if group_label_map:
        new_labels = []
        for l in labels:
            # Intenta mapear como int (seeds) si es posible; si no, como string (algos)
            try:
                key = int(l)
                new_labels.append(group_label_map.get(key, str(l)))
            except (ValueError, TypeError):
                new_labels.append(group_label_map.get(l, str(l)))
        labels = new_labels

    ax.legend(handles, labels, title=None, frameon=True)

    if ylim is not None:
        ax.set_ylim(*ylim)


def save(fig: plt.Figure, out_path: Path, dpi: int = 260):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def filter_optuna_variants(df_algo: pd.DataFrame, algo: str) -> pd.DataFrame:
    """Nos quedamos solo con las seeds que representan variantes Optuna, para no mezclar con seed=1 etc."""
    if algo == "a2c":
        keep = set(A2C_LABELS.keys())  # 10,20,30
    elif algo == "ppo":
        keep = set(PPO_LABELS.keys())  # 10,30
    else:
        keep = set(df_algo["seed"].unique())
    return df_algo[df_algo["seed"].isin(keep)].copy()


def build_final_df(df: pd.DataFrame) -> pd.DataFrame:
    """Construye DF con la selección FINAL_PICK (experimentos finales)."""
    rows = []
    for (algo, obs_mode), seed in FINAL_PICK.items():
        sub = df[(df["algo"] == algo) & (df["obs_mode"] == obs_mode) & (df["seed"] == seed)]
        if sub.shape[0] == 0:
            # si falta, lo dejamos como NaN para que quede “hueco”
            rows.append({"algo": algo, "obs_mode": obs_mode, "seed": seed,
                         "mean_reward": np.nan, "std_reward": np.nan, "completion_ratio": np.nan})
        else:
            rows.append(sub.iloc[0][["algo", "obs_mode", "seed", "mean_reward", "std_reward", "completion_ratio"]].to_dict())

    out = pd.DataFrame(rows)
    out["obs_mode"] = pd.Categorical(out["obs_mode"], categories=OBS_ORDER, ordered=True)
    out["algo"] = pd.Categorical(out["algo"], categories=["a2c", "ppo", "dqn"], ordered=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="experiments")
    parser.add_argument("--outdir", type=str, default="plots_tfm")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments)
    out_dir = Path(args.outdir)

    csvs = find_metrics_csvs(experiments_dir)
    df = load_all_csvs(csvs)

    # ---------- 1) A2C: Optuna budgets ----------
    a2c = df[df["algo"] == "a2c"].copy()
    a2c = filter_optuna_variants(a2c, "a2c")
    for metric, ylabel in [
        ("mean_reward", "Mean reward"),
        ("std_reward", "Std reward"),
        ("completion_ratio", "Completion ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        grouped_bar(
            ax=ax,
            data=a2c,
            x_col="obs_mode",
            group_col="seed",
            y_col=metric,
            title=f"A2C — Entrenamiento con 5M steps",
            ylabel=ylabel,
            group_label_map=A2C_LABELS,
            ylim=(0, 1.0) if metric == "completion_ratio" else None,
            note="",
        )
        save(fig, out_dir / f"a2c_optuna_{metric}.png")

    # ---------- 2) PPO: Optuna budgets ----------
    ppo = df[df["algo"] == "ppo"].copy()
    ppo = filter_optuna_variants(ppo, "ppo")
    for metric, ylabel in [
        ("mean_reward", "Mean reward"),
        ("std_reward", "Std reward"),
        ("completion_ratio", "Completion ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        grouped_bar(
            ax=ax,
            data=ppo,
            x_col="obs_mode",
            group_col="seed",
            y_col=metric,
            title=f"PPO — Entrenamiento con 5M steps",
            ylabel=ylabel,
            group_label_map=PPO_LABELS,
            ylim=(0, 1.0) if metric == "completion_ratio" else None,
        )
        save(fig, out_dir / f"ppo_optuna_{metric}.png")

    # ---------- 3) DQN: ----------
    dqn = df[(df["algo"] == "dqn") & (df["seed"] == 50)].copy()
    for metric, ylabel in [
        ("mean_reward", "Mean reward"),
        ("std_reward", "Std reward"),
        ("completion_ratio", "Completion ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        # aquí group_col puede ser algo constante; mejor hacemos “una sola serie”:
        dqn_plot = dqn.copy()
        dqn_plot["variant"] = "DQN"
        dqn_plot["variant"] = pd.Categorical(dqn_plot["variant"], categories=["DQN"], ordered=True)

        grouped_bar(
            ax=ax,
            data=dqn_plot,
            x_col="obs_mode",
            group_col="variant",
            y_col=metric,
            title="DQN — Entrenamiento con 5M steps",
            ylabel=ylabel,
            group_label_map=None,
            ylim=(0, 1.0) if metric == "completion_ratio" else None,
        )
        save(fig, out_dir / f"dqn_{metric}.png")

    # ---------- 4) Comparación FINAL----------
    final_df = build_final_df(df)

    for metric, ylabel in [
        ("mean_reward", "Mean reward"),
        ("std_reward", "Std reward"),
        ("completion_ratio", "Completion ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        grouped_bar(
            ax=ax,
            data=final_df,
            x_col="obs_mode",
            group_col="algo",
            y_col=metric,
            title="Comparación final — Entrenamiento con 5M steps",
            ylabel=ylabel,
            group_label_map={"a2c": "A2C", "ppo": "PPO", "dqn": "DQN"},
            ylim=(0, 1.0) if metric == "completion_ratio" else None,
        )
        save(fig, out_dir / f"final_algos_{metric}.png")

    print(f"OK. Figuras en: {out_dir.resolve()}")
    for p in csvs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
