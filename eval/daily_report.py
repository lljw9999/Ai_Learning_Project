#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.paths import EVAL_DIR, LOG_DIR, RANKING_DIR, ROOT_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily offline eval + drift report")
    parser.add_argument("--retrieval-mode", choices=["two_tower", "hybrid"], default="hybrid")
    parser.add_argument("--retrieval-k", type=int, default=200)
    parser.add_argument("--ranking-k", type=int, default=10)
    parser.add_argument("--itemcf-neighbors", type=int, default=200)
    parser.add_argument("--itemcf-min-pair", type=int, default=2)
    parser.add_argument("--two-tower-weight", type=float, default=1.0)
    parser.add_argument("--itemcf-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    if expected.size == 0 or actual.size == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 3:
        return 0.0

    expected_hist, _ = np.histogram(expected, bins=edges)
    actual_hist, _ = np.histogram(actual, bins=edges)

    expected_pct = np.clip(expected_hist / max(expected_hist.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_hist / max(actual_hist.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _run_offline_eval(args: argparse.Namespace) -> None:
    cmd = [
        "python",
        "eval/offline_eval.py",
        "--retrieval-mode",
        args.retrieval_mode,
        "--retrieval-k",
        str(args.retrieval_k),
        "--ranking-k",
        str(args.ranking_k),
        "--itemcf-neighbors",
        str(args.itemcf_neighbors),
        "--itemcf-min-pair",
        str(args.itemcf_min_pair),
        "--two-tower-weight",
        str(args.two_tower_weight),
        "--itemcf-weight",
        str(args.itemcf_weight),
        "--rrf-k",
        str(args.rrf_k),
        "--device",
        args.device,
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def _drift_value(train_frame: pd.DataFrame, test_frame: pd.DataFrame, col: str) -> float:
    if col not in train_frame.columns or col not in test_frame.columns:
        return 0.0
    return _psi(train_frame[col].to_numpy(dtype=float), test_frame[col].to_numpy(dtype=float))


def _append_csv(path: Path, row: dict, fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    _run_offline_eval(args)

    metrics = json.loads((EVAL_DIR / "offline_metrics.json").read_text())
    train_frame = pd.read_parquet(RANKING_DIR / "ranker_train_frame.parquet")
    test_frame = pd.read_parquet(EVAL_DIR / "ranking_test_frame.parquet")

    row = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "retrieval_recall100_test": metrics["retrieval"]["test"].get("recall@100", 0.0),
        "ranking_ndcg10_test": metrics["ranking"].get("ndcg@10", 0.0),
        "latency_p95_ms": metrics["latency"].get("p95_ms", 0.0),
        "drift_user_activity_psi": _drift_value(train_frame, test_frame, "user_activity"),
        "drift_item_popularity_psi": _drift_value(train_frame, test_frame, "item_popularity"),
        "drift_retrieval_score_psi": _drift_value(train_frame, test_frame, "retrieval_score"),
    }

    out_csv = LOG_DIR / "daily_metrics.csv"
    _append_csv(out_csv, row, fieldnames=list(row.keys()))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
