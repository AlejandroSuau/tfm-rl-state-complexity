# rlpacman/utils/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

__all__ = [
    "load_monitor_csv",
    "auc_return",
    "time_to_threshold",
    "summarize_monitor",
]

MONITOR_COL_MAP = {
    # SB3 Monitor CSV columns (per episode)
    # r: episodic return, l: episode length (timesteps), t: time (seconds)
    "r": "reward",
    "l": "length",
    "t": "time_s",
}

def load_monitor_csv(path: Path | str) -> pd.DataFrame:
    """
    Carga un CSV de Stable-Baselines3 Monitor.
    Ignora líneas de cabecera que comienzan por '#'.
    Devuelve columnas: reward, length, time_s, ep, steps_cum.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el CSV: {path}")

    # Saltar líneas de comentarios
    with path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    if not lines:
        raise ValueError(f"CSV vacío tras eliminar comentarios: {path}")

    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines)))
    # Renombrar columnas a nombres claros
    df = df.rename(columns=MONITOR_COL_MAP)
    # Validaciones mínimas
    for col in ("reward", "length"):
        if col not in df.columns:
            raise ValueError(f"Falta columna '{col}' en {path.name}. Columnas: {list(df.columns)}")

    df["ep"] = np.arange(len(df), dtype=int)
    df["steps_cum"] = df["length"].cumsum()
    if "time_s" not in df.columns:
        df["time_s"] = np.nan
    return df

@dataclass
class AUCResult:
    auc: float
    auc_normalized: float
    steps_total: int

def auc_return(df: pd.DataFrame, normalize: bool = True) -> AUCResult:
    """
    AUC del retorno episodico respecto a los timesteps acumulados.
    Usa la regla trapezoidal entre puntos (steps_cum[i], reward[i]).
    """
    x = df["steps_cum"].to_numpy()
    y = df["reward"].to_numpy()
    if len(x) < 2:
        auc = float(np.nan)
        return AUCResult(auc=auc, auc_normalized=auc, steps_total=int(x[-1] if len(x) else 0))

    # AUC trapezoidal
    dx = np.diff(x)
    y_mid = 0.5 * (y[:-1] + y[1:])
    auc = float(np.sum(dx * y_mid))
    steps_total = int(x[-1])
    if normalize and steps_total > 0:
        # Normalización por el "área" máxima si el retorno fuese constante = max(y)
        # Alternativa: normalizar por steps_total * (max_reward - min_reward) si procede.
        max_y = np.nanmax(y) if len(y) else 1.0
        denom = steps_total * max(1.0, max_y)
        auc_norm = auc / denom if denom > 0 else np.nan
    else:
        auc_norm = auc
    return AUCResult(auc=auc, auc_normalized=auc_norm, steps_total=steps_total)

def time_to_threshold(
    df: pd.DataFrame,
    threshold: float,
    window_episodes: int = 5,
) -> float:
    """
    Devuelve el primer timestep acumulado donde la media móvil (por episodio) del retorno
    >= threshold. Si no se alcanza, devuelve np.inf.
    """
    if len(df) == 0:
        return np.inf
    rewards = df["reward"].rolling(window=window_episodes, min_periods=max(1, window_episodes)).mean()
    reached = np.where(rewards >= threshold)[0]
    if len(reached) == 0:
        return float(np.inf)
    idx = int(reached[0])
    return float(df.loc[idx, "steps_cum"])

def summarize_monitor(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    window_episodes: int = 5,
) -> Dict[str, Any]:
    """
    Resumen con métricas principales.
    """
    res_auc = auc_return(df, normalize=False)
    summary = {
        "episodes": int(len(df)),
        "steps_total": int(res_auc.steps_total),
        "return_mean": float(df["reward"].mean()) if len(df) else np.nan,
        "return_std": float(df["reward"].std(ddof=1)) if len(df) > 1 else np.nan,
        "auc": float(res_auc.auc),
    }
    if threshold is not None:
        ttt = time_to_threshold(df, threshold=threshold, window_episodes=window_episodes)
        summary["time_to_threshold"] = float(ttt)
    return summary
