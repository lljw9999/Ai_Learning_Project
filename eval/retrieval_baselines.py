#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.baselines import (
    build_item_cf_neighbors,
    build_item_popularity,
    popularity_ranking,
    recommend_item_cf,
    recommend_popularity,
)
from recsys.io import load_all_splits, split_ground_truth, train_and_eval_histories
from recsys.metrics import summarize_ranking_metrics, summarize_retrieval_metrics
from recsys.paths import EVAL_DIR, PROCESSED_DATA_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--itemcf-neighbors", type=int, default=200)
    parser.add_argument("--itemcf-min-pair", type=int, default=2)
    return parser.parse_args()


def evaluate(name: str, preds_val, preds_test, val_truth, test_truth, top_k: int) -> dict:
    return {
        "name": name,
        "val_retrieval": summarize_retrieval_metrics(preds_val, val_truth, ks=[50, 100, 200]),
        "test_retrieval": summarize_retrieval_metrics(preds_test, test_truth, ks=[50, 100, 200]),
        "val_ranking": summarize_ranking_metrics(preds_val, val_truth, ks=[10]),
        "test_ranking": summarize_ranking_metrics(preds_test, test_truth, ks=[10]),
        "top_k": top_k,
    }


def main() -> None:
    args = parse_args()
    ensure_dirs()

    splits = load_all_splits(PROCESSED_DATA_DIR)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    train_history, test_history = train_and_eval_histories(train_df, val_df)
    val_users = sorted(val_df["user_idx"].unique().tolist())
    test_users = sorted(test_df["user_idx"].unique().tolist())
    val_truth = split_ground_truth(val_df)
    test_truth = split_ground_truth(test_df)

    popularity = build_item_popularity(train_df)
    global_pop = popularity_ranking(popularity)

    pop_val = recommend_popularity(val_users, train_history, global_pop, top_k=args.top_k)
    pop_test = recommend_popularity(test_users, test_history, global_pop, top_k=args.top_k)

    item_cf = build_item_cf_neighbors(
        train_history,
        top_neighbors=args.itemcf_neighbors,
        min_pair_count=args.itemcf_min_pair,
    )
    cf_val = recommend_item_cf(val_users, train_history, item_cf, global_pop, top_k=args.top_k)
    cf_test = recommend_item_cf(test_users, test_history, item_cf, global_pop, top_k=args.top_k)

    results = {
        "popularity": evaluate("popularity", pop_val, pop_test, val_truth, test_truth, top_k=args.top_k),
        "item_cf": evaluate("item_cf", cf_val, cf_test, val_truth, test_truth, top_k=args.top_k),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "retrieval_baselines.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
