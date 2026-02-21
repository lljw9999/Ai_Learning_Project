#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.paths import EVAL_DIR, RANKING_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed end-to-end reproducibility sweep")
    parser.add_argument("--seeds", default="11,22,33,44,55", help="Comma-separated random seeds")
    parser.add_argument("--skip-retrieval", action="store_true", help="Reuse existing retrieval model for all seeds")

    # Retrieval config
    parser.add_argument("--retrieval-epochs", type=int, default=10)
    parser.add_argument("--retrieval-embedding-dim", type=int, default=96)
    parser.add_argument("--retrieval-num-negatives", type=int, default=20)
    parser.add_argument("--retrieval-learning-rate", type=float, default=8e-4)
    parser.add_argument("--retrieval-batch-size", type=int, default=512)
    parser.add_argument("--retrieval-max-history-len", type=int, default=80)

    # Ranking config
    parser.add_argument("--retrieval-mode", choices=["two_tower", "hybrid"], default="hybrid")
    parser.add_argument("--train-candidate-k", type=int, default=120)
    parser.add_argument("--eval-candidate-k", type=int, default=500)
    parser.add_argument("--future-positive-window", type=int, default=10)
    parser.add_argument("--max-queries-per-user", type=int, default=15)
    parser.add_argument("--num-boost-round", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--min-ranker-improve", type=float, default=0.005)
    parser.add_argument("--guardrail-confidence", type=float, default=0.95)
    parser.add_argument("--guardrail-bootstrap-samples", type=int, default=1000)

    # Offline eval config
    parser.add_argument("--retrieval-k", type=int, default=200)
    parser.add_argument("--ranking-k", type=int, default=10)
    parser.add_argument("--candidate-k-grid", default="100,200,500")
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _parse_seeds(seed_text: str) -> List[int]:
    seeds: List[int] = []
    for token in str(seed_text).split(","):
        tok = token.strip()
        if not tok:
            continue
        seeds.append(int(tok))
    if not seeds:
        raise ValueError("No seeds provided")
    return seeds


def _run(cmd: List[str]) -> None:
    print(f"[seed_sweep] run: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return json.loads(path.read_text())


def _summary_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seeds = _parse_seeds(args.seeds)

    runs: List[dict] = []
    for seed in seeds:
        if not args.skip_retrieval:
            _run(
                [
                    "python",
                    "retrieval/train.py",
                    "--epochs",
                    str(args.retrieval_epochs),
                    "--embedding-dim",
                    str(args.retrieval_embedding_dim),
                    "--num-negatives",
                    str(args.retrieval_num_negatives),
                    "--learning-rate",
                    str(args.retrieval_learning_rate),
                    "--batch-size",
                    str(args.retrieval_batch_size),
                    "--max-history-len",
                    str(args.retrieval_max_history_len),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ]
            )
            _run(["python", "retrieval/build_index.py"])

        _run(
            [
                "python",
                "ranking/train.py",
                "--retrieval-mode",
                args.retrieval_mode,
                "--train-candidate-k",
                str(args.train_candidate_k),
                "--eval-candidate-k",
                str(args.eval_candidate_k),
                "--future-positive-window",
                str(args.future_positive_window),
                "--max-queries-per-user",
                str(args.max_queries_per_user),
                "--num-boost-round",
                str(args.num_boost_round),
                "--learning-rate",
                str(args.learning_rate),
                "--num-leaves",
                str(args.num_leaves),
                "--early-stopping-rounds",
                str(args.early_stopping_rounds),
                "--seed",
                str(seed),
                "--min-ranker-improve",
                str(args.min_ranker_improve),
                "--guardrail-confidence",
                str(args.guardrail_confidence),
                "--guardrail-bootstrap-samples",
                str(args.guardrail_bootstrap_samples),
                "--guardrail-bootstrap-seed",
                str(seed),
                "--device",
                args.device,
            ]
        )

        _run(
            [
                "python",
                "eval/offline_eval.py",
                "--retrieval-mode",
                args.retrieval_mode,
                "--retrieval-k",
                str(args.retrieval_k),
                "--ranking-k",
                str(args.ranking_k),
                "--candidate-k-grid",
                args.candidate_k_grid,
                "--latency-warmup",
                str(args.latency_warmup),
                "--device",
                args.device,
            ]
        )

        train_metrics = _read_json(RANKING_DIR / "train_metrics.json")
        offline_metrics = _read_json(EVAL_DIR / "offline_metrics.json")
        retrieval_ndcg = float(offline_metrics["retrieval_order_ranking"].get(f"ndcg@{args.ranking_k}", 0.0))
        ranking_ndcg = float(offline_metrics["ranking"].get(f"ndcg@{args.ranking_k}", 0.0))
        lift_abs = ranking_ndcg - retrieval_ndcg
        lift_rel_pct = (lift_abs / retrieval_ndcg * 100.0) if retrieval_ndcg > 0 else 0.0
        runs.append(
            {
                "seed": int(seed),
                "retrieval_ndcg": retrieval_ndcg,
                "ranking_ndcg": ranking_ndcg,
                "lift_abs": float(lift_abs),
                "lift_rel_pct": float(lift_rel_pct),
                "use_ranker_score": bool(train_metrics.get("use_ranker_score", False)),
                "guardrail": train_metrics.get("guardrail", {}),
                "candidate_k_sweep": offline_metrics.get("candidate_k_sweep", {}),
            }
        )

    retrieval_vals = [float(r["retrieval_ndcg"]) for r in runs]
    ranking_vals = [float(r["ranking_ndcg"]) for r in runs]
    lift_rel_vals = [float(r["lift_rel_pct"]) for r in runs]
    pass_rate = float(np.mean([1.0 if bool(r["use_ranker_score"]) else 0.0 for r in runs])) if runs else 0.0

    candidate_keys = sorted({k for r in runs for k in r.get("candidate_k_sweep", {}).keys()}, key=lambda x: int(x))
    by_k: Dict[str, dict] = {}
    for key in candidate_keys:
        retr_k_vals: List[float] = []
        rank_k_vals: List[float] = []
        lift_k_vals: List[float] = []
        for run in runs:
            ks = run.get("candidate_k_sweep", {})
            if key not in ks:
                continue
            retr_k = float(ks[key]["retrieval_order"].get(f"ndcg@{args.ranking_k}", 0.0))
            rank_k = float(ks[key]["ranking"].get(f"ndcg@{args.ranking_k}", 0.0))
            lift_k = float(ks[key].get("relative_lift_pct", 0.0))
            retr_k_vals.append(retr_k)
            rank_k_vals.append(rank_k)
            lift_k_vals.append(lift_k)
        by_k[key] = {
            "retrieval_ndcg": _summary_stats(retr_k_vals),
            "ranking_ndcg": _summary_stats(rank_k_vals),
            "relative_lift_pct": _summary_stats(lift_k_vals),
        }

    out = {
        "config": {
            "seeds": seeds,
            "skip_retrieval": bool(args.skip_retrieval),
            "retrieval_mode": args.retrieval_mode,
            "ranking_k": int(args.ranking_k),
            "retrieval_k": int(args.retrieval_k),
            "candidate_k_grid": args.candidate_k_grid,
        },
        "runs": runs,
        "aggregate": {
            "retrieval_ndcg": _summary_stats(retrieval_vals),
            "ranking_ndcg": _summary_stats(ranking_vals),
            "relative_lift_pct": _summary_stats(lift_rel_vals),
            "guardrail_pass_rate": pass_rate,
            "by_candidate_k": by_k,
        },
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "seed_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
