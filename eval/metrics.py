#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.paths import EVAL_DIR


def main() -> None:
    path = EVAL_DIR / "offline_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Run eval/offline_eval.py first. Missing {path}")

    metrics = json.loads(path.read_text())
    if "retrieval_mode" in metrics:
        print(f"Retrieval mode: {metrics['retrieval_mode']}")
    print("Retrieval (test):")
    for k, v in metrics["retrieval"]["test"].items():
        print(f"  {k}: {v:.4f}")

    print("Ranking (test):")
    for k, v in metrics["ranking"].items():
        print(f"  {k}: {v:.4f}")

    if "retrieval_order_ranking" in metrics:
        print("Retrieval-order ranking (test):")
        for k, v in metrics["retrieval_order_ranking"].items():
            print(f"  {k}: {v:.4f}")

    if "ranking_ablation" in metrics and "use_ranker_score" in metrics["ranking_ablation"]:
        print(f"Ranker enabled: {metrics['ranking_ablation']['use_ranker_score']}")

    print("Latency:")
    print(f"  p50_ms: {metrics['latency']['p50_ms']:.2f}")
    print(f"  p95_ms: {metrics['latency']['p95_ms']:.2f}")


if __name__ == "__main__":
    main()
