#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.baselines import build_item_cf_neighbors, build_item_popularity, popularity_ranking
from recsys.features import ItemFeatureStore, build_ranking_frame, build_user_context
from recsys.io import load_all_splits, split_ground_truth, train_and_eval_histories
from recsys.metrics import summarize_ranking_metrics, summarize_retrieval_metrics
from recsys.paths import EVAL_DIR, PROCESSED_DATA_DIR, RANKING_DIR, RETRIEVAL_DIR, ensure_dirs
from recsys.pipeline import context_timestamp_map, retrieve_candidates_for_users, retrieve_hybrid_candidates_for_users
from recsys.ranking import load_ranker, predict_scores, ranking_dict_from_frame
from recsys.retrieval import CandidateIndex, compute_user_embeddings_batch, load_two_tower_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for two-stage recommender")
    parser.add_argument("--retrieval-mode", choices=["two_tower", "hybrid"], default="hybrid")
    parser.add_argument("--retrieval-k", type=int, default=200)
    parser.add_argument("--ranking-k", type=int, default=10)
    parser.add_argument("--itemcf-neighbors", type=int, default=200)
    parser.add_argument("--itemcf-min-pair", type=int, default=2)
    parser.add_argument("--two-tower-weight", type=float, default=1.0)
    parser.add_argument("--itemcf-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--latency-warmup", type=int, default=50, help="Number of initial requests to skip in latency stats")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _latency_percentile(latencies_ms: List[float], pct: float) -> float:
    if not latencies_ms:
        return 0.0
    return float(np.percentile(np.asarray(latencies_ms, dtype=np.float32), pct))


def main() -> None:
    args = parse_args()
    ensure_dirs()

    splits = load_all_splits(data_dir=PROCESSED_DATA_DIR)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    model_ret, ret_meta = load_two_tower_model(RETRIEVAL_DIR / "two_tower_model", device=args.device)
    candidate_index = CandidateIndex.load(RETRIEVAL_DIR / "candidate_index")

    train_history, test_history = train_and_eval_histories(train_df, val_df)
    val_embedding_cache = compute_user_embeddings_batch(
        model=model_ret,
        user_histories=train_history,
        max_history_len=int(ret_meta["max_history_len"]),
        batch_size=1024,
        device=args.device,
    )
    test_embedding_cache = compute_user_embeddings_batch(
        model=model_ret,
        user_histories=test_history,
        max_history_len=int(ret_meta["max_history_len"]),
        batch_size=1024,
        device=args.device,
    )
    val_users = sorted(val_df["user_idx"].unique().tolist())
    test_users = sorted(test_df["user_idx"].unique().tolist())

    val_truth = split_ground_truth(val_df)
    test_truth = split_ground_truth(test_df)

    item_pop = build_item_popularity(train_df)
    global_pop = popularity_ranking(item_pop)
    item_cf_neighbors = build_item_cf_neighbors(
        train_history,
        top_neighbors=args.itemcf_neighbors,
        min_pair_count=args.itemcf_min_pair,
    )

    if args.retrieval_mode == "hybrid":
        val_candidates, _ = retrieve_hybrid_candidates_for_users(
            users=val_users,
            model=model_ret,
            index=candidate_index,
            user_histories=train_history,
            max_history_len=int(ret_meta["max_history_len"]),
            top_k=args.retrieval_k,
            item_cf_neighbors=item_cf_neighbors,
            global_pop_ranked_items=global_pop,
            two_tower_weight=args.two_tower_weight,
            item_cf_weight=args.itemcf_weight,
            rrf_k=args.rrf_k,
            device=args.device,
            user_embedding_cache=val_embedding_cache,
        )
        test_candidates, test_scores = retrieve_hybrid_candidates_for_users(
            users=test_users,
            model=model_ret,
            index=candidate_index,
            user_histories=test_history,
            max_history_len=int(ret_meta["max_history_len"]),
            top_k=args.retrieval_k,
            item_cf_neighbors=item_cf_neighbors,
            global_pop_ranked_items=global_pop,
            two_tower_weight=args.two_tower_weight,
            item_cf_weight=args.itemcf_weight,
            rrf_k=args.rrf_k,
            device=args.device,
            user_embedding_cache=test_embedding_cache,
        )
    else:
        val_candidates, _ = retrieve_candidates_for_users(
            users=val_users,
            model=model_ret,
            index=candidate_index,
            user_histories=train_history,
            max_history_len=int(ret_meta["max_history_len"]),
            top_k=args.retrieval_k,
            device=args.device,
            user_embedding_cache=val_embedding_cache,
        )
        test_candidates, test_scores = retrieve_candidates_for_users(
            users=test_users,
            model=model_ret,
            index=candidate_index,
            user_histories=test_history,
            max_history_len=int(ret_meta["max_history_len"]),
            top_k=args.retrieval_k,
            device=args.device,
            user_embedding_cache=test_embedding_cache,
        )

    retrieval_metrics = {
        "val": summarize_retrieval_metrics(val_candidates, val_truth, ks=[50, 100, 200]),
        "test": summarize_retrieval_metrics(test_candidates, test_truth, ks=[50, 100, 200]),
    }
    retrieval_order_ranking_metrics = summarize_ranking_metrics(test_candidates, test_truth, ks=[args.ranking_k])

    item_store = ItemFeatureStore.load(RANKING_DIR / "item_feature_store.pkl")
    ranker, feature_cols, score_blend_alpha, use_ranker_score = load_ranker(RANKING_DIR / "lightgbm_ranker")

    history_df_for_test = pd.concat([train_df, val_df], ignore_index=True)
    user_context_test = build_user_context(history_df_for_test, item_store.item_main_genre)
    test_context_ts = context_timestamp_map(test_df)

    rank_test_df = build_ranking_frame(
        candidates_by_user=test_candidates,
        scores_by_user=test_scores,
        item_store=item_store,
        user_context=user_context_test,
        context_timestamps=test_context_ts,
        ground_truth=test_truth,
        item_cf_neighbors=item_cf_neighbors if args.retrieval_mode == "hybrid" else None,
    )
    if use_ranker_score:
        rank_test_df["rank_score"] = predict_scores(ranker, rank_test_df, feature_cols=feature_cols)
        rank_test_df["final_score"] = rank_test_df["rank_score"] + score_blend_alpha * rank_test_df["retrieval_score"]
    else:
        rank_test_df["final_score"] = rank_test_df["retrieval_score"]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rank_test_df.to_parquet(EVAL_DIR / "ranking_test_frame.parquet", index=False)

    ranked_items = ranking_dict_from_frame(rank_test_df, score_col="final_score")
    ranking_metrics = summarize_ranking_metrics(ranked_items, test_truth, ks=[args.ranking_k])

    # Simple feature ablation: zero out retrieval_score and observe metric change.
    ablated_df = rank_test_df.copy()
    ablated_df["retrieval_score"] = 0.0
    if use_ranker_score:
        ablated_df["rank_score"] = predict_scores(ranker, ablated_df, feature_cols=feature_cols)
        ablated_df["final_score"] = ablated_df["rank_score"] + score_blend_alpha * ablated_df["retrieval_score"]
    else:
        ablated_df["final_score"] = ablated_df["retrieval_score"]
    ablated_ranked = ranking_dict_from_frame(ablated_df, score_col="final_score")
    ablation_metrics = summarize_ranking_metrics(ablated_ranked, test_truth, ks=[args.ranking_k])

    # End-to-end latency estimate over test users.
    latencies_ms: List[float] = []
    for idx, user in enumerate(test_users):
        start = time.perf_counter()
        if args.retrieval_mode == "hybrid":
            candidates_u, scores_u = retrieve_hybrid_candidates_for_users(
                users=[user],
                model=model_ret,
                index=candidate_index,
                user_histories=test_history,
                max_history_len=int(ret_meta["max_history_len"]),
                top_k=args.retrieval_k,
                item_cf_neighbors=item_cf_neighbors,
                global_pop_ranked_items=global_pop,
                two_tower_weight=args.two_tower_weight,
                item_cf_weight=args.itemcf_weight,
                rrf_k=args.rrf_k,
                device=args.device,
                user_embedding_cache=test_embedding_cache,
            )
        else:
            candidates_u, scores_u = retrieve_candidates_for_users(
                users=[user],
                model=model_ret,
                index=candidate_index,
                user_histories=test_history,
                max_history_len=int(ret_meta["max_history_len"]),
                top_k=args.retrieval_k,
                device=args.device,
                user_embedding_cache=test_embedding_cache,
            )
        frame_u = build_ranking_frame(
            candidates_by_user=candidates_u,
            scores_by_user=scores_u,
            item_store=item_store,
            user_context=user_context_test,
            context_timestamps={user: test_context_ts.get(user, 0)},
            ground_truth=None,
            item_cf_neighbors=item_cf_neighbors if args.retrieval_mode == "hybrid" else None,
        )
        if use_ranker_score:
            frame_u["rank_score"] = predict_scores(ranker, frame_u, feature_cols=feature_cols)
            frame_u["final_score"] = frame_u["rank_score"] + score_blend_alpha * frame_u["retrieval_score"]
        else:
            frame_u["final_score"] = frame_u["retrieval_score"]
        elapsed = (time.perf_counter() - start) * 1000.0
        if idx >= int(args.latency_warmup):
            latencies_ms.append(float(elapsed))

    latency_metrics = {
        "p50_ms": _latency_percentile(latencies_ms, 50),
        "p95_ms": _latency_percentile(latencies_ms, 95),
        "num_requests": len(latencies_ms),
    }

    summary = {
        "retrieval_mode": args.retrieval_mode,
        "retrieval": retrieval_metrics,
        "retrieval_order_ranking": retrieval_order_ranking_metrics,
        "ranking": ranking_metrics,
        "ranking_ablation": {
            "feature_removed": "retrieval_score",
            "score_blend_alpha": score_blend_alpha,
            "use_ranker_score": use_ranker_score,
            "metrics": ablation_metrics,
            "ndcg_drop": ranking_metrics.get(f"ndcg@{args.ranking_k}", 0.0)
            - ablation_metrics.get(f"ndcg@{args.ranking_k}", 0.0),
        },
        "latency": latency_metrics,
    }

    out_path = EVAL_DIR / "offline_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
