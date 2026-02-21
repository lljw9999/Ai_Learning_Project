#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.features import build_item_feature_store, build_ranking_frame, build_user_context, feature_columns
from recsys.io import load_all_splits, load_items, split_ground_truth
from recsys.metrics import bootstrap_lift_confidence, per_user_ndcg_at_k, summarize_ranking_metrics
from recsys.paths import PROCESSED_DATA_DIR, RANKING_DIR, RETRIEVAL_DIR, ensure_dirs
from recsys.baselines import build_item_cf_neighbors, build_item_popularity, popularity_ranking
from recsys.pipeline import context_timestamp_map, retrieve_candidates_for_users, retrieve_hybrid_candidates_for_users
from recsys.ranker_dataset import build_ranker_training_frame, build_train_queries
from recsys.ranking import predict_scores, ranking_dict_from_frame, save_ranker, train_ranker
from recsys.retrieval import CandidateIndex, build_user_history, load_two_tower_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM ranker with query-level train events")
    parser.add_argument("--retrieval-mode", choices=["two_tower", "hybrid"], default="hybrid")
    parser.add_argument("--train-candidate-k", type=int, default=120, help="Retrieved candidates per train query")
    parser.add_argument("--eval-candidate-k", type=int, default=500, help="Retrieved candidates per validation user")
    parser.add_argument("--itemcf-neighbors", type=int, default=200)
    parser.add_argument("--itemcf-min-pair", type=int, default=2)
    parser.add_argument("--two-tower-weight", type=float, default=1.0)
    parser.add_argument("--itemcf-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--min-history", type=int, default=5, help="Minimum history length to create a train query")
    parser.add_argument("--max-queries-per-user", type=int, default=15, help="Maximum train queries sampled per user")
    parser.add_argument(
        "--future-positive-window",
        type=int,
        default=10,
        help="Future interaction window size used for graded labels (label=1)",
    )
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1, help="LightGBM threads")
    parser.add_argument(
        "--min-ranker-improve",
        type=float,
        default=0.002,
        help="Minimum bootstrap median NDCG@10 lift over retrieval-order to enable ranker scores",
    )
    parser.add_argument(
        "--guardrail-confidence",
        type=float,
        default=0.95,
        help="Required bootstrap confidence P(lift > 0) to enable ranker scores",
    )
    parser.add_argument(
        "--guardrail-bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap iterations for ranker guardrail",
    )
    parser.add_argument("--guardrail-bootstrap-seed", type=int, default=42)
    parser.add_argument("--debug-sample-users", type=int, default=5)
    parser.add_argument(
        "--blend-grid",
        default="0.0,0.25,0.5,0.75,1.0,1.5,2.0",
        help="Comma-separated alpha grid for final_score = rank_score + alpha*retrieval_score",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    splits = load_all_splits(data_dir=PROCESSED_DATA_DIR)
    train_df = splits["train"]
    val_df = splits["val"]
    items_df = load_items(data_dir=PROCESSED_DATA_DIR)

    model, model_meta = load_two_tower_model(RETRIEVAL_DIR / "two_tower_model", device=args.device)
    index = CandidateIndex.load(RETRIEVAL_DIR / "candidate_index")

    item_store = build_item_feature_store(train_df=train_df, items_df=items_df)
    item_store.save(RANKING_DIR / "item_feature_store.pkl")
    item_pop = build_item_popularity(train_df)
    global_pop = popularity_ranking(item_pop)
    item_cf_neighbors = build_item_cf_neighbors(
        build_user_history(train_df),
        top_neighbors=args.itemcf_neighbors,
        min_pair_count=args.itemcf_min_pair,
    )

    train_queries = build_train_queries(
        interactions=train_df,
        min_history=args.min_history,
        max_queries_per_user=args.max_queries_per_user,
        future_positive_window=args.future_positive_window,
    )
    if not train_queries:
        raise ValueError("No train queries generated. Reduce --min-history or review preprocessing.")

    rank_train_df = build_ranker_training_frame(
        queries=train_queries,
        model=model,
        index=index,
        item_store=item_store,
        max_history_len=int(model_meta["max_history_len"]),
        candidate_k=args.train_candidate_k,
        include_missed_positive=False,
        item_cf_neighbors=item_cf_neighbors if args.retrieval_mode == "hybrid" else None,
        global_pop_ranked_items=global_pop,
        two_tower_weight=args.two_tower_weight,
        item_cf_weight=args.itemcf_weight,
        rrf_k=args.rrf_k,
        device=args.device,
    )
    if rank_train_df.empty:
        raise ValueError("Ranking training frame is empty")

    # Keep query groups with at least one positive item.
    query_pos = rank_train_df.groupby("query_id")["label"].sum()
    valid_queries = query_pos[query_pos > 0].index
    rank_train_df = rank_train_df.loc[rank_train_df["query_id"].isin(valid_queries)].copy()

    train_history = build_user_history(train_df)
    val_users = sorted(val_df["user_idx"].unique().tolist())
    val_truth = split_ground_truth(val_df)

    if args.retrieval_mode == "hybrid":
        val_candidates, val_scores = retrieve_hybrid_candidates_for_users(
            users=val_users,
            model=model,
            index=index,
            user_histories=train_history,
            max_history_len=int(model_meta["max_history_len"]),
            top_k=args.eval_candidate_k,
            item_cf_neighbors=item_cf_neighbors,
            global_pop_ranked_items=global_pop,
            two_tower_weight=args.two_tower_weight,
            item_cf_weight=args.itemcf_weight,
            rrf_k=args.rrf_k,
            device=args.device,
        )
    else:
        val_candidates, val_scores = retrieve_candidates_for_users(
            users=val_users,
            model=model,
            index=index,
            user_histories=train_history,
            max_history_len=int(model_meta["max_history_len"]),
            top_k=args.eval_candidate_k,
            device=args.device,
        )

    user_context = build_user_context(train_df, item_store.item_main_genre)
    val_context_ts = context_timestamp_map(val_df)
    rank_val_df = build_ranking_frame(
        candidates_by_user=val_candidates,
        scores_by_user=val_scores,
        item_store=item_store,
        user_context=user_context,
        context_timestamps=val_context_ts,
        ground_truth=val_truth,
        item_cf_neighbors=item_cf_neighbors if args.retrieval_mode == "hybrid" else None,
    )
    val_pos = rank_val_df.groupby("user_idx")["label"].sum()
    val_positive_users = set(int(u) for u in val_pos[val_pos > 0].index.tolist())
    rank_val_for_early_stopping = rank_val_df.loc[rank_val_df["user_idx"].isin(val_positive_users)].copy()

    feat_cols = feature_columns()
    model_ranker = train_ranker(
        rank_train_df,
        feature_cols=feat_cols,
        group_col="query_id",
        valid_df=rank_val_for_early_stopping,
        valid_group_col="user_idx",
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
    )

    rank_val_df["rank_score"] = predict_scores(model_ranker, rank_val_df, feature_cols=feat_cols)
    score_vals = rank_val_df["rank_score"].to_numpy(dtype=np.float64)
    label_vals = rank_val_df["label"].to_numpy(dtype=np.float64)
    if score_vals.size > 0 and float(np.std(score_vals)) > 0 and float(np.std(label_vals)) > 0:
        score_label_corr = float(np.corrcoef(score_vals, label_vals)[0, 1])
    else:
        score_label_corr = 0.0
    per_user_score_std = rank_val_df.groupby("user_idx")["rank_score"].std(ddof=0).fillna(0.0)

    retrieval_only_metrics = summarize_ranking_metrics(
        ranking_dict_from_frame(rank_val_df, score_col="retrieval_score"),
        val_truth,
        ks=[10],
    )
    retrieval_only_ndcg = float(retrieval_only_metrics.get("ndcg@10", 0.0))

    alpha_grid = [float(s.strip()) for s in args.blend_grid.split(",") if s.strip()]
    if not alpha_grid:
        alpha_grid = [0.0]

    best_alpha = alpha_grid[0]
    best_val_ndcg = -1.0
    tuned_metrics = {}
    for alpha in alpha_grid:
        score_col = f"score_alpha_{alpha:.4f}"
        rank_val_df[score_col] = rank_val_df["rank_score"] + alpha * rank_val_df["retrieval_score"]
        pred_dict = ranking_dict_from_frame(rank_val_df, score_col=score_col)
        metrics = summarize_ranking_metrics(pred_dict, val_truth, ks=[10])
        tuned_metrics[alpha] = metrics
        ndcg = float(metrics.get("ndcg@10", 0.0))
        if ndcg > best_val_ndcg:
            best_val_ndcg = ndcg
            best_alpha = alpha

    rank_val_df["final_score_model"] = rank_val_df["rank_score"] + best_alpha * rank_val_df["retrieval_score"]
    ranker_enabled_metrics = summarize_ranking_metrics(
        ranking_dict_from_frame(rank_val_df, score_col="final_score_model"),
        val_truth,
        ks=[10],
    )
    retrieval_user_ndcg = per_user_ndcg_at_k(
        ranking_dict_from_frame(rank_val_df, score_col="retrieval_score"),
        val_truth,
        k=10,
    )
    ranker_user_ndcg = per_user_ndcg_at_k(
        ranking_dict_from_frame(rank_val_df, score_col="final_score_model"),
        val_truth,
        k=10,
    )
    bootstrap_guardrail = bootstrap_lift_confidence(
        baseline_scores=retrieval_user_ndcg,
        candidate_scores=ranker_user_ndcg,
        n_bootstrap=args.guardrail_bootstrap_samples,
        random_state=args.guardrail_bootstrap_seed,
    )
    use_ranker_score = bool(
        float(bootstrap_guardrail.get("p_lift_gt_zero", 0.0)) >= float(args.guardrail_confidence)
        and float(bootstrap_guardrail.get("median_lift", 0.0)) >= float(args.min_ranker_improve)
    )
    if not use_ranker_score:
        rank_val_df["final_score"] = rank_val_df["retrieval_score"]
    else:
        rank_val_df["final_score"] = rank_val_df["final_score_model"]
    val_metrics = summarize_ranking_metrics(
        ranking_dict_from_frame(rank_val_df, score_col="final_score"),
        val_truth,
        ks=[10],
    )

    save_ranker(
        model_ranker,
        feature_cols=feat_cols,
        path_prefix=RANKING_DIR / "lightgbm_ranker",
        score_blend_alpha=best_alpha,
        use_ranker_score=use_ranker_score,
    )
    rank_train_df.to_parquet(RANKING_DIR / "ranker_train_frame.parquet", index=False)
    rank_val_df.to_parquet(RANKING_DIR / "ranker_val_frame.parquet", index=False)
    sample_users = sorted(val_positive_users)[: max(int(args.debug_sample_users), 0)]
    top10_examples = []
    for user in sample_users:
        frame_u = rank_val_df.loc[rank_val_df["user_idx"] == int(user)].copy()
        retrieval_top = frame_u.sort_values("retrieval_score", ascending=False).head(10)
        model_top = frame_u.sort_values("final_score_model", ascending=False).head(10)
        served_top = frame_u.sort_values("final_score", ascending=False).head(10)
        top10_examples.append(
            {
                "user_idx": int(user),
                "retrieval_top10": [
                    {"item_idx": int(r.item_idx), "score": float(r.retrieval_score), "label": int(r.label)}
                    for r in retrieval_top.itertuples(index=False)
                ],
                "ranker_top10": [
                    {"item_idx": int(r.item_idx), "score": float(r.final_score_model), "label": int(r.label)}
                    for r in model_top.itertuples(index=False)
                ],
                "served_top10": [
                    {"item_idx": int(r.item_idx), "score": float(r.final_score), "label": int(r.label)}
                    for r in served_top.itertuples(index=False)
                ],
            }
        )
    debug_report = {
        "score_distribution": {
            "min": float(np.min(score_vals)) if score_vals.size > 0 else 0.0,
            "mean": float(np.mean(score_vals)) if score_vals.size > 0 else 0.0,
            "max": float(np.max(score_vals)) if score_vals.size > 0 else 0.0,
            "std": float(np.std(score_vals)) if score_vals.size > 0 else 0.0,
        },
        "per_user_score_std": {
            "mean": float(per_user_score_std.mean()) if len(per_user_score_std) > 0 else 0.0,
            "median": float(per_user_score_std.median()) if len(per_user_score_std) > 0 else 0.0,
            "min": float(per_user_score_std.min()) if len(per_user_score_std) > 0 else 0.0,
            "max": float(per_user_score_std.max()) if len(per_user_score_std) > 0 else 0.0,
        },
        "score_label_corr": float(score_label_corr),
        "top10_examples": top10_examples,
    }
    (RANKING_DIR / "ranker_debug_report.json").write_text(json.dumps(debug_report, indent=2))

    summary = {
        "num_train_queries": int(rank_train_df["query_id"].nunique()),
        "num_train_rows": int(len(rank_train_df)),
        "num_val_users": int(rank_val_df["user_idx"].nunique()),
        "num_val_rows": int(len(rank_val_df)),
        "retrieval_mode": args.retrieval_mode,
        "train_candidate_k": args.train_candidate_k,
        "eval_candidate_k": args.eval_candidate_k,
        "best_iteration": int(getattr(model_ranker, "best_iteration_", 0) or 0),
        "score_blend_alpha": float(best_alpha),
        "use_ranker_score": bool(use_ranker_score),
        "val_retrieval_order_ndcg@10": retrieval_only_ndcg,
        "val_ranker_enabled_ndcg@10": float(ranker_enabled_metrics.get("ndcg@10", 0.0)),
        "val_users_with_retrieved_positive": int(len(val_positive_users)),
        "guardrail": {
            "p_lift_gt_zero_threshold": float(args.guardrail_confidence),
            "median_lift_threshold": float(args.min_ranker_improve),
            **bootstrap_guardrail,
        },
        "train_label_distribution": {
            str(int(k)): int(v) for k, v in rank_train_df["label"].value_counts().sort_index().items()
        },
        "score_diagnostics": {
            "score_label_corr": float(score_label_corr),
            "score_std": float(np.std(score_vals)) if score_vals.size > 0 else 0.0,
        },
        **{f"val_{k}": float(v) for k, v in val_metrics.items()},
    }
    (RANKING_DIR / "train_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
