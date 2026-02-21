#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Mapping, Sequence
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
    parser.add_argument(
        "--candidate-k-grid",
        default="100,200,500",
        help="Comma-separated candidate sizes used for @K robustness curves",
    )
    parser.add_argument("--ranking-k", type=int, default=10)
    parser.add_argument("--itemcf-neighbors", type=int, default=200)
    parser.add_argument("--itemcf-min-pair", type=int, default=2)
    parser.add_argument("--two-tower-weight", type=float, default=1.0)
    parser.add_argument("--itemcf-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--lgbm-baseline-rounds", type=int, default=150)
    parser.add_argument("--lgbm-baseline-leaves", type=int, default=31)
    parser.add_argument("--drift-psi-threshold", type=float, default=0.25)
    parser.add_argument("--drift-kl-threshold", type=float, default=0.10)
    parser.add_argument("--latency-warmup", type=int, default=50, help="Number of initial requests to skip in latency stats")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _latency_percentile(latencies_ms: List[float], pct: float) -> float:
    if not latencies_ms:
        return 0.0
    return float(np.percentile(np.asarray(latencies_ms, dtype=np.float32), pct))


def _parse_k_grid(grid_text: str, fallback_k: int) -> List[int]:
    out: List[int] = []
    for token in str(grid_text).split(","):
        tok = token.strip()
        if not tok:
            continue
        val = int(tok)
        if val > 0:
            out.append(val)
    if fallback_k > 0:
        out.append(int(fallback_k))
    if not out:
        out = [int(fallback_k)]
    return sorted(set(out))


def _truncate_candidates(
    candidates_by_user: Mapping[int, Sequence[int]],
    scores_by_user: Mapping[int, Sequence[float]],
    top_k: int,
) -> tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    items_out: Dict[int, List[int]] = {}
    scores_out: Dict[int, List[float]] = {}
    for user, items in candidates_by_user.items():
        user_i = int(user)
        items_list = [int(i) for i in list(items)[:top_k]]
        score_list = [float(s) for s in list(scores_by_user.get(user_i, []))[: len(items_list)]]
        if len(score_list) < len(items_list):
            score_list.extend([0.0] * (len(items_list) - len(score_list)))
        items_out[user_i] = items_list
        scores_out[user_i] = score_list
    return items_out, scores_out


def _score_frame(
    frame: pd.DataFrame,
    ranker,
    feature_cols: Sequence[str],
    score_blend_alpha: float,
    use_ranker_score: bool,
) -> pd.DataFrame:
    out = frame.copy()
    if use_ranker_score:
        out["rank_score"] = predict_scores(ranker, out, feature_cols=feature_cols)
        out["final_score"] = out["rank_score"] + score_blend_alpha * out["retrieval_score"]
    else:
        out["final_score"] = out["retrieval_score"]
    return out


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


def _kl(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
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
    return float(np.sum(actual_pct * np.log(actual_pct / expected_pct)))


def _drift_for_feature(train_arr: np.ndarray, eval_arr: np.ndarray) -> Dict[str, float]:
    return {
        "psi": float(_psi(train_arr, eval_arr)),
        "kl": float(_kl(train_arr, eval_arr)),
    }


def _drift_report(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Dict[str, dict]:
    features = sorted(set(str(c) for c in feature_cols if c in train_frame.columns))
    report: Dict[str, dict] = {
        "features": features,
        "pairs": {
            "train_val": {},
            "val_test": {},
            "train_test": {},
        },
    }
    for col in features:
        train_arr = train_frame[col].to_numpy(dtype=float)
        val_arr = val_frame[col].to_numpy(dtype=float)
        test_arr = test_frame[col].to_numpy(dtype=float)
        report["pairs"]["train_val"][col] = _drift_for_feature(train_arr, val_arr)
        report["pairs"]["val_test"][col] = _drift_for_feature(val_arr, test_arr)
        report["pairs"]["train_test"][col] = _drift_for_feature(train_arr, test_arr)
    return report


def _train_reranker_baselines(
    rank_train_df: pd.DataFrame,
    rank_val_df: pd.DataFrame,
    rank_test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    ranking_k: int,
    test_truth: Mapping[int, set[int]],
    lgbm_rounds: int,
    lgbm_leaves: int,
) -> Dict[str, dict]:
    baselines: Dict[str, dict] = {}
    if rank_train_df.empty or rank_val_df.empty or rank_test_df.empty:
        return baselines

    y_train = (rank_train_df["label"] > 0).astype(int)
    y_val = (rank_val_df["label"] > 0).astype(int)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return baselines

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
            ]
        )
        pipe.fit(rank_train_df[list(feature_cols)], y_train)
        logreg_df = rank_test_df.copy()
        logreg_df["baseline_score"] = pipe.predict_proba(logreg_df[list(feature_cols)])[:, 1]
        logreg_ranked = ranking_dict_from_frame(logreg_df, score_col="baseline_score")
        baselines["logistic_full_features"] = summarize_ranking_metrics(logreg_ranked, test_truth, ks=[ranking_k])
    except Exception as exc:  # pragma: no cover
        baselines["logistic_full_features_error"] = {"error": str(exc)}

    try:
        import lightgbm as lgb

        train_sorted = rank_train_df.sort_values(["query_id", "retrieval_rank"], ascending=[True, True]).reset_index(drop=True)
        val_sorted = rank_val_df.sort_values(["user_idx", "retrieval_rank"], ascending=[True, True]).reset_index(drop=True)
        lgbm_model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="gbdt",
            n_estimators=int(lgbm_rounds),
            learning_rate=0.05,
            num_leaves=int(lgbm_leaves),
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=42,
        )
        lgbm_model.fit(
            train_sorted[["retrieval_score"]],
            (train_sorted["label"] > 0).astype(int),
            group=train_sorted.groupby("query_id").size().tolist(),
            eval_set=[(val_sorted[["retrieval_score"]], (val_sorted["label"] > 0).astype(int))],
            eval_group=[val_sorted.groupby("user_idx").size().tolist()],
            eval_at=[ranking_k],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        lgbm_df = rank_test_df.copy()
        lgbm_df["baseline_score"] = lgbm_model.predict(lgbm_df[["retrieval_score"]], num_threads=1)
        lgbm_ranked = ranking_dict_from_frame(lgbm_df, score_col="baseline_score")
        baselines["lgbm_retrieval_score_only"] = summarize_ranking_metrics(lgbm_ranked, test_truth, ks=[ranking_k])
    except Exception as exc:  # pragma: no cover
        baselines["lgbm_retrieval_score_only_error"] = {"error": str(exc)}

    return baselines


def main() -> None:
    args = parse_args()
    ensure_dirs()
    k_grid = _parse_k_grid(args.candidate_k_grid, fallback_k=args.retrieval_k)
    max_retrieval_k = max(k_grid)

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
            top_k=max_retrieval_k,
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
            top_k=max_retrieval_k,
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
            top_k=max_retrieval_k,
            device=args.device,
            user_embedding_cache=val_embedding_cache,
        )
        test_candidates, test_scores = retrieve_candidates_for_users(
            users=test_users,
            model=model_ret,
            index=candidate_index,
            user_histories=test_history,
            max_history_len=int(ret_meta["max_history_len"]),
            top_k=max_retrieval_k,
            device=args.device,
            user_embedding_cache=test_embedding_cache,
        )

    val_candidates_main, _ = _truncate_candidates(val_candidates, {u: [] for u in val_candidates}, args.retrieval_k)
    test_candidates_main, test_scores_main = _truncate_candidates(test_candidates, test_scores, args.retrieval_k)
    retrieval_metrics = {
        "val": summarize_retrieval_metrics(val_candidates_main, val_truth, ks=[50, 100, 200]),
        "test": summarize_retrieval_metrics(test_candidates_main, test_truth, ks=[50, 100, 200]),
    }
    retrieval_order_ranking_metrics = summarize_ranking_metrics(test_candidates_main, test_truth, ks=[args.ranking_k])

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
    rank_test_df = _score_frame(
        rank_test_df,
        ranker=ranker,
        feature_cols=feature_cols,
        score_blend_alpha=score_blend_alpha,
        use_ranker_score=use_ranker_score,
    )
    rank_test_df_main = rank_test_df.loc[rank_test_df["retrieval_rank"] <= float(args.retrieval_k)].copy()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rank_test_df_main.to_parquet(EVAL_DIR / "ranking_test_frame.parquet", index=False)

    ranked_items = ranking_dict_from_frame(rank_test_df_main, score_col="final_score")
    ranking_metrics = summarize_ranking_metrics(ranked_items, test_truth, ks=[args.ranking_k])

    # Simple feature ablation: zero out retrieval_score and observe metric change.
    ablated_df = rank_test_df_main.copy()
    ablated_df["retrieval_score"] = 0.0
    ablated_df = _score_frame(
        ablated_df,
        ranker=ranker,
        feature_cols=feature_cols,
        score_blend_alpha=score_blend_alpha,
        use_ranker_score=use_ranker_score,
    )
    ablated_ranked = ranking_dict_from_frame(ablated_df, score_col="final_score")
    ablation_metrics = summarize_ranking_metrics(ablated_ranked, test_truth, ks=[args.ranking_k])

    # Candidate-size robustness sweep.
    k_sweep: Dict[str, dict] = {}
    for cand_k in k_grid:
        cand_items_k, cand_scores_k = _truncate_candidates(test_candidates, test_scores, cand_k)
        retrieval_order_k = summarize_ranking_metrics(cand_items_k, test_truth, ks=[args.ranking_k])
        frame_k = rank_test_df.loc[rank_test_df["retrieval_rank"] <= float(cand_k)].copy()
        frame_k = _score_frame(
            frame_k,
            ranker=ranker,
            feature_cols=feature_cols,
            score_blend_alpha=score_blend_alpha,
            use_ranker_score=use_ranker_score,
        )
        rank_k = summarize_ranking_metrics(ranking_dict_from_frame(frame_k, score_col="final_score"), test_truth, ks=[args.ranking_k])
        retrieval_ndcg = float(retrieval_order_k.get(f"ndcg@{args.ranking_k}", 0.0))
        rank_ndcg = float(rank_k.get(f"ndcg@{args.ranking_k}", 0.0))
        rel_lift = ((rank_ndcg - retrieval_ndcg) / retrieval_ndcg * 100.0) if retrieval_ndcg > 0 else 0.0
        k_sweep[str(int(cand_k))] = {
            "retrieval_order": retrieval_order_k,
            "ranking": rank_k,
            "relative_lift_pct": float(rel_lift),
        }

    # Reranker baseline comparisons on the same evaluation frame.
    rank_train_df = pd.read_parquet(RANKING_DIR / "ranker_train_frame.parquet")
    rank_val_df = pd.read_parquet(RANKING_DIR / "ranker_val_frame.parquet")
    rank_val_df_main = rank_val_df.loc[rank_val_df["retrieval_rank"] <= float(args.retrieval_k)].copy()
    reranker_baselines = _train_reranker_baselines(
        rank_train_df=rank_train_df,
        rank_val_df=rank_val_df_main,
        rank_test_df=rank_test_df_main,
        feature_cols=feature_cols,
        ranking_k=args.ranking_k,
        test_truth=test_truth,
        lgbm_rounds=args.lgbm_baseline_rounds,
        lgbm_leaves=args.lgbm_baseline_leaves,
    )

    drift = _drift_report(
        train_frame=rank_train_df,
        val_frame=rank_val_df_main,
        test_frame=rank_test_df_main,
        feature_cols=feature_cols,
    )
    high_drift_features = [
        feature
        for feature, vals in drift["pairs"]["val_test"].items()
        if float(vals.get("psi", 0.0)) >= float(args.drift_psi_threshold)
        or float(vals.get("kl", 0.0)) >= float(args.drift_kl_threshold)
    ]
    drift_warning = bool(use_ranker_score and len(high_drift_features) > 0)
    if drift_warning:
        print(
            "[offline_eval] WARNING guardrail passed while high val->test drift detected. "
            f"features={','.join(high_drift_features)}"
        )

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
        frame_u = _score_frame(
            frame_u,
            ranker=ranker,
            feature_cols=feature_cols,
            score_blend_alpha=score_blend_alpha,
            use_ranker_score=use_ranker_score,
        )
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
        "candidate_k_sweep": k_sweep,
        "reranker_baselines": reranker_baselines,
        "drift": {
            "thresholds": {
                "psi": float(args.drift_psi_threshold),
                "kl": float(args.drift_kl_threshold),
            },
            "high_drift_features_val_test": high_drift_features,
            "guardrail_drift_warning": drift_warning,
            **drift,
        },
        "latency": latency_metrics,
    }

    out_path = EVAL_DIR / "offline_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
