from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd

from rlpacman.utils.metrics import load_monitor_csv, summarize_monitor

def find_monitor_csvs(log_root: Path) -> List[Path]:
    candidates = []
    for p in log_root.rglob("*.csv"):
        if "monitor" in p.name.lower():
            candidates.append(p)
    return sorted(candidates)

def parse_run_id(path: Path) -> str:
    parts = [p for p in path.parts if p]
    tail = parts[-4:]
    run_id = "/".join(tail)
    return run_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    log_root = Path(args.logs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    csvs = find_monitor_csvs(log_root)
    if not csvs:
        print(f"No se encontraron CSVs de Monitor en {log_root.resolve()}")
        return

    for csv_path in csvs:
        try:
            df = load_monitor_csv(csv_path)
            summary = summarize_monitor(df, threshold=args.threshold, window_episodes=args.window)
            row = {"run": parse_run_id(csv_path), "csv_path": str(csv_path), **summary}
            rows.append(row)
        except Exception as e:
            rows.append({"run": parse_run_id(csv_path), "csv_path": str(csv_path), "error": str(e)})

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Guardado resumen: {summary_csv}")
    if "auc" in summary_df.columns:
        print(summary_df.sort_values("auc", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
