#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.baselines import build_item_cf_neighbors, build_item_popularity, popularity_ranking
from recsys.features import ItemFeatureStore, build_ranking_frame, build_user_context
from recsys.io import load_all_splits, load_items, load_users
from recsys.paths import PROCESSED_DATA_DIR, RANKING_DIR, RETRIEVAL_DIR
from recsys.pipeline import retrieve_candidates_for_users, retrieve_hybrid_candidates_for_users
from recsys.ranking import load_ranker, predict_scores
from recsys.retrieval import CandidateIndex, build_user_history, load_two_tower_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval + ranking for one user")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--retrieval-mode", choices=["two_tower", "hybrid"], default="hybrid")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _user_id_to_idx(users_df: pd.DataFrame) -> Dict[int, int]:
    return {int(row.user_id): int(row.user_idx) for row in users_df.itertuples(index=False)}


def _item_idx_to_id(items_df: pd.DataFrame) -> Dict[int, int]:
    return {int(row.item_idx): int(row.item_id) for row in items_df.itertuples(index=False)}


def main() -> None:
    args = parse_args()

    splits = load_all_splits(data_dir=PROCESSED_DATA_DIR)
    train_df, val_df = splits["train"], splits["val"]
    history_df = pd.concat([train_df, val_df], ignore_index=True)

    users_df = load_users(data_dir=PROCESSED_DATA_DIR)
    items_df = load_items(data_dir=PROCESSED_DATA_DIR)
    user_map = _user_id_to_idx(users_df)
    item_map = _item_idx_to_id(items_df)

    user_idx = user_map.get(args.user_id)
    item_store = ItemFeatureStore.load(RANKING_DIR / "item_feature_store.pkl")

    if user_idx is None:
        # Cold-start user fallback: global popularity.
        payload = [
            {"item_id": item_map.get(item_idx, -1), "item_idx": item_idx, "score": 0.0}
            for item_idx in item_store.global_popular_items[: args.top_n]
        ]
        print(json.dumps({"user_id": args.user_id, "cold_start": True, "recommendations": payload}, indent=2))
        return

    model_ret, ret_meta = load_two_tower_model(RETRIEVAL_DIR / "two_tower_model", device=args.device)
    index = CandidateIndex.load(RETRIEVAL_DIR / "candidate_index")

    user_histories = build_user_history(history_df)
    if args.retrieval_mode == "hybrid":
        popularity = build_item_popularity(train_df)
        global_pop = popularity_ranking(popularity)
        item_cf_neighbors = build_item_cf_neighbors(
            build_user_history(train_df),
            top_neighbors=200,
            min_pair_count=2,
        )
        candidates, scores = retrieve_hybrid_candidates_for_users(
            users=[user_idx],
            model=model_ret,
            index=index,
            user_histories=user_histories,
            max_history_len=int(ret_meta["max_history_len"]),
            top_k=args.candidate_k,
            item_cf_neighbors=item_cf_neighbors,
            global_pop_ranked_items=global_pop,
            two_tower_weight=1.0,
            item_cf_weight=1.0,
            rrf_k=60,
            device=args.device,
        )
    else:
        candidates, scores = retrieve_candidates_for_users(
            users=[user_idx],
            model=model_ret,
            index=index,
            user_histories=user_histories,
            max_history_len=int(ret_meta["max_history_len"]),
            top_k=args.candidate_k,
            device=args.device,
        )

    user_context = build_user_context(history_df, item_store.item_main_genre)
    context_ts = {user_idx: int(time.time())}

    frame = build_ranking_frame(
        candidates_by_user=candidates,
        scores_by_user=scores,
        item_store=item_store,
        user_context=user_context,
        context_timestamps=context_ts,
        ground_truth=None,
    )

    ranker, feature_cols, score_blend_alpha, use_ranker_score = load_ranker(RANKING_DIR / "lightgbm_ranker")
    if use_ranker_score:
        frame["rank_score"] = predict_scores(ranker, frame, feature_cols=feature_cols)
        frame["final_score"] = frame["rank_score"] + score_blend_alpha * frame["retrieval_score"]
    else:
        frame["final_score"] = frame["retrieval_score"]
    frame = frame.sort_values("final_score", ascending=False).head(args.top_n)

    payload = []
    for row in frame.itertuples(index=False):
        payload.append(
            {
                    "item_id": int(item_map.get(int(row.item_idx), -1)),
                    "item_idx": int(row.item_idx),
                    "score": float(row.final_score),
                }
            )

    print(
        json.dumps(
            {
                "user_id": args.user_id,
                "user_idx": int(user_idx),
                "cold_start": False,
                "recommendations": payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
