#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, TextIO

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.baselines import build_item_cf_neighbors, build_item_popularity, popularity_ranking
from recsys.features import ItemFeatureStore, build_ranking_frame, build_user_context
from recsys.io import load_all_splits, load_items, load_users
from recsys.paths import EVAL_DIR, LOG_DIR, PROCESSED_DATA_DIR, RANKING_DIR, RETRIEVAL_DIR, ensure_dirs
from recsys.pipeline import retrieve_candidates_for_users, retrieve_hybrid_candidates_for_users
from recsys.ranking import load_ranker, predict_scores
from recsys.retrieval import CandidateIndex, build_user_history, compute_user_embeddings_batch, load_two_tower_model

app = FastAPI(title="Two-Stage Recommender API", version="0.2.0")
RETRIEVAL_MODE_DEFAULT = os.getenv("RETRIEVAL_MODE", "hybrid")
DEFAULT_CANDIDATE_K = int(os.getenv("DEFAULT_CANDIDATE_K", "200"))


class MetricsState:
    def __init__(self, maxlen: int = 5000) -> None:
        self.latencies_ms: deque[float] = deque(maxlen=maxlen)
        self.error_flags: deque[int] = deque(maxlen=maxlen)
        self.lock = Lock()

    def record(self, latency_ms: float, is_error: bool) -> None:
        with self.lock:
            self.latencies_ms.append(float(latency_ms))
            self.error_flags.append(1 if is_error else 0)

    def summary(self) -> Dict[str, float | int]:
        with self.lock:
            latencies = list(self.latencies_ms)
            errors = list(self.error_flags)

        if not latencies:
            return {"requests": 0, "p50_ms": 0.0, "p95_ms": 0.0, "error_rate": 0.0}

        import numpy as np

        arr = np.asarray(latencies, dtype=float)
        err_rate = float(sum(errors) / len(errors)) if errors else 0.0
        return {
            "requests": len(latencies),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "error_rate": err_rate,
        }


class ServiceContext:
    def __init__(self) -> None:
        self.model_ret = None
        self.ret_meta = None
        self.index = None
        self.ranker = None
        self.feature_cols: List[str] = []
        self.score_blend_alpha: float = 0.0
        self.use_ranker_score: bool = True

        self.user_map: Dict[int, int] = {}
        self.user_idx_to_id: Dict[int, int] = {}
        self.item_idx_to_id: Dict[int, int] = {}
        self.item_id_to_idx: Dict[int, int] = {}
        self.item_meta: Dict[int, Dict[str, str]] = {}

        self.user_histories: Dict[int, List[int]] = {}
        self.user_embedding_cache: Dict[int, object] = {}
        self.item_cf_neighbors: Dict[int, List[tuple[int, float]]] = {}
        self.global_pop_ranked_items: List[int] = []
        self.item_store: ItemFeatureStore | None = None
        self.user_context = None

        self.model_version: str = "unknown"

        self.metrics = MetricsState()
        self.log_path = LOG_DIR / "service_requests.jsonl"
        self.log_file: TextIO | None = None
        self.log_lock = Lock()
        self.pending_log_lines = 0


CTX = ServiceContext()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _resolve_model_version() -> str:
    env_version = os.getenv("MODEL_VERSION")
    if env_version:
        return str(env_version)
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _log_request(payload: dict) -> None:
    if CTX.log_file is None:
        return
    line = json.dumps(payload) + "\n"
    with CTX.log_lock:
        CTX.log_file.write(line)
        CTX.pending_log_lines += 1
        if CTX.pending_log_lines >= 100:
            CTX.log_file.flush()
            CTX.pending_log_lines = 0


def _telemetry_snapshot() -> dict:
    offline_metrics = _load_json_if_exists(EVAL_DIR / "offline_metrics.json") or {}
    seed_sweep = _load_json_if_exists(EVAL_DIR / "seed_sweep.json") or {}
    train_metrics = _load_json_if_exists(RANKING_DIR / "train_metrics.json") or {}
    latency = CTX.metrics.summary()

    test_lift = offline_metrics.get("test_lift", {})
    lift_bootstrap = test_lift.get("bootstrap", {}) if isinstance(test_lift, dict) else {}
    aggregate = seed_sweep.get("aggregate", {}) if isinstance(seed_sweep, dict) else {}

    abs_lift_ci = aggregate.get("abs_lift_ndcg_mean_ci95", {}) if isinstance(aggregate, dict) else {}
    rel_lift_ci = aggregate.get("relative_lift_pct_mean_ci95", {}) if isinstance(aggregate, dict) else {}

    guardrail = train_metrics.get("guardrail", {}) if isinstance(train_metrics, dict) else {}
    drift = offline_metrics.get("drift", {}) if isinstance(offline_metrics, dict) else {}

    return {
        "api_status": "ok",
        "updated_utc": _now_utc_iso(),
        "model": {
            "label": "hybrid+ltr",
            "retrieval_default": RETRIEVAL_MODE_DEFAULT,
            "ranker_default_enabled": bool(CTX.use_ranker_score),
            "score_blend_alpha": float(CTX.score_blend_alpha),
            "model_version": CTX.model_version,
        },
        "defaults": {
            "candidate_k": int(DEFAULT_CANDIDATE_K),
            "top_n": 10,
        },
        "latency": latency,
        "last_eval": {
            "retrieval_ndcg10": float((offline_metrics.get("retrieval_order_ranking") or {}).get("ndcg@10", 0.0)),
            "ranking_ndcg10": float((offline_metrics.get("ranking") or {}).get("ndcg@10", 0.0)),
            "abs_lift_ndcg10": float(test_lift.get("abs_ndcg_lift", 0.0)) if isinstance(test_lift, dict) else 0.0,
            "relative_lift_pct": float(test_lift.get("relative_lift_pct", 0.0)) if isinstance(test_lift, dict) else 0.0,
            "ci95": [
                float(lift_bootstrap.get("ci95_low", 0.0)),
                float(lift_bootstrap.get("ci95_high", 0.0)),
            ],
            "ci95_positive": bool(test_lift.get("ci95_positive", False)) if isinstance(test_lift, dict) else False,
        },
        "seed_sweep": {
            "num_seeds": int(len(seed_sweep.get("runs", []))) if isinstance(seed_sweep, dict) else 0,
            "guardrail_pass_rate": float(aggregate.get("guardrail_pass_rate", 0.0)) if isinstance(aggregate, dict) else 0.0,
            "abs_lift_mean": float((aggregate.get("abs_lift_ndcg") or {}).get("mean", 0.0)),
            "abs_lift_ci95": [
                float(abs_lift_ci.get("ci95_low", 0.0)) if isinstance(abs_lift_ci, dict) else 0.0,
                float(abs_lift_ci.get("ci95_high", 0.0)) if isinstance(abs_lift_ci, dict) else 0.0,
            ],
            "relative_lift_mean": float((aggregate.get("relative_lift_pct") or {}).get("mean", 0.0)),
            "relative_lift_ci95": [
                float(rel_lift_ci.get("ci95_low", 0.0)) if isinstance(rel_lift_ci, dict) else 0.0,
                float(rel_lift_ci.get("ci95_high", 0.0)) if isinstance(rel_lift_ci, dict) else 0.0,
            ],
            "positive_abs_lift_seeds": (aggregate.get("positive_abs_lift_seeds") or {}).get("count", 0),
            "total_seeds": (aggregate.get("positive_abs_lift_seeds") or {}).get("total", 0),
        },
        "guardrail": {
            "enabled": bool(CTX.use_ranker_score),
            "p_lift_gt_zero": float(guardrail.get("p_lift_gt_zero", 0.0)) if isinstance(guardrail, dict) else 0.0,
            "median_lift": float(guardrail.get("median_lift", 0.0)) if isinstance(guardrail, dict) else 0.0,
            "ci95": [
                float(guardrail.get("ci95_low", 0.0)) if isinstance(guardrail, dict) else 0.0,
                float(guardrail.get("ci95_high", 0.0)) if isinstance(guardrail, dict) else 0.0,
            ],
            "reason": "guardrail_enabled" if CTX.use_ranker_score else "guardrail_disabled",
        },
        "drift": {
            "warning": bool((drift.get("guardrail_drift_warning", False)) if isinstance(drift, dict) else False),
            "high_features": list((drift.get("high_drift_features_val_test", [])) if isinstance(drift, dict) else []),
        },
    }


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()

    splits = load_all_splits(data_dir=PROCESSED_DATA_DIR)
    history_df = pd.concat([splits["train"], splits["val"]], ignore_index=True)

    users_df = load_users(data_dir=PROCESSED_DATA_DIR)
    items_df = load_items(data_dir=PROCESSED_DATA_DIR)

    CTX.user_map = {int(row.user_id): int(row.user_idx) for row in users_df.itertuples(index=False)}
    CTX.user_idx_to_id = {int(row.user_idx): int(row.user_id) for row in users_df.itertuples(index=False)}
    CTX.item_idx_to_id = {int(row.item_idx): int(row.item_id) for row in items_df.itertuples(index=False)}
    CTX.item_id_to_idx = {int(row.item_id): int(row.item_idx) for row in items_df.itertuples(index=False)}
    CTX.item_meta = {
        int(row.item_idx): {
            "title": str(row.title) if row.title is not None else f"Item {int(row.item_id)}",
            "genres": str(row.genres) if row.genres is not None else "unknown",
        }
        for row in items_df.itertuples(index=False)
    }

    CTX.model_ret, CTX.ret_meta = load_two_tower_model(RETRIEVAL_DIR / "two_tower_model", device="cpu")
    CTX.index = CandidateIndex.load(RETRIEVAL_DIR / "candidate_index")

    CTX.ranker, CTX.feature_cols, CTX.score_blend_alpha, CTX.use_ranker_score = load_ranker(
        RANKING_DIR / "lightgbm_ranker"
    )
    CTX.item_store = ItemFeatureStore.load(RANKING_DIR / "item_feature_store.pkl")

    CTX.user_histories = build_user_history(history_df)
    popularity = build_item_popularity(splits["train"])
    CTX.global_pop_ranked_items = popularity_ranking(popularity)
    CTX.item_cf_neighbors = build_item_cf_neighbors(
        build_user_history(splits["train"]),
        top_neighbors=200,
        min_pair_count=2,
    )
    CTX.user_embedding_cache = compute_user_embeddings_batch(
        model=CTX.model_ret,
        user_histories=CTX.user_histories,
        max_history_len=int(CTX.ret_meta["max_history_len"]),
        batch_size=1024,
        device="cpu",
    )
    CTX.user_context = build_user_context(history_df, CTX.item_store.item_main_genre)
    CTX.model_version = _resolve_model_version()

    CTX.log_path.parent.mkdir(parents=True, exist_ok=True)
    CTX.log_file = CTX.log_path.open("a")
    CTX.pending_log_lines = 0


@app.on_event("shutdown")
def shutdown() -> None:
    if CTX.log_file is not None:
        with CTX.log_lock:
            CTX.log_file.flush()
            CTX.log_file.close()
            CTX.log_file = None
            CTX.pending_log_lines = 0


@app.get("/")
def home() -> FileResponse:
    page = ROOT_DIR / "service" / "static" / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Frontend page not found")
    return FileResponse(page)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "utc": _now_utc_iso(), "model_version": CTX.model_version}


@app.get("/metrics")
def metrics() -> dict:
    return CTX.metrics.summary()


@app.get("/api/users")
def api_users(limit: int = Query(200, ge=1, le=2000)) -> dict:
    if CTX.user_context is None:
        raise HTTPException(status_code=500, detail="User context not loaded")

    users = []
    for user_id, user_idx in CTX.user_map.items():
        activity = int(CTX.user_context.user_activity.get(int(user_idx), 0))
        users.append(
            {
                "user_id": int(user_id),
                "user_idx": int(user_idx),
                "activity": activity,
                "top_genres": sorted(list(CTX.user_context.user_top_genres.get(int(user_idx), set()))),
            }
        )

    users.sort(key=lambda x: (-x["activity"], x["user_id"]))
    return {"count": int(len(users)), "users": users[: int(limit)]}


@app.get("/api/items")
def api_items(item_id: int | None = Query(None), limit: int = Query(20, ge=1, le=200)) -> dict:
    if CTX.item_store is None:
        raise HTTPException(status_code=500, detail="Item store is not loaded")

    if item_id is not None:
        item_idx = CTX.item_id_to_idx.get(int(item_id))
        if item_idx is None:
            raise HTTPException(status_code=404, detail=f"Unknown item_id={item_id}")
        meta = CTX.item_meta.get(int(item_idx), {"title": f"Item {item_id}", "genres": "unknown"})
        return {
            "item": {
                "item_id": int(item_id),
                "item_idx": int(item_idx),
                "title": meta["title"],
                "genres": meta["genres"],
            }
        }

    items = []
    for item_idx in CTX.item_store.global_popular_items[: int(limit)]:
        item_idx_i = int(item_idx)
        item_id_i = int(CTX.item_idx_to_id.get(item_idx_i, -1))
        meta = CTX.item_meta.get(item_idx_i, {"title": f"Item {item_id_i}", "genres": "unknown"})
        items.append(
            {
                "item_id": item_id_i,
                "item_idx": item_idx_i,
                "title": meta["title"],
                "genres": meta["genres"],
            }
        )
    return {"count": int(len(items)), "items": items}


@app.get("/api/telemetry")
def api_telemetry() -> dict:
    return _telemetry_snapshot()


@app.get("/recommend")
def recommend(
    user_id: int = Query(..., description="Original user_id"),
    n: int = Query(10, ge=1, le=100),
    candidate_k: int = Query(DEFAULT_CANDIDATE_K, ge=10, le=1000),
    retrieval_mode: str | None = Query(None, pattern="^(hybrid|two_tower)$"),
    use_ranker: bool | None = Query(None),
    explain: bool = Query(False),
) -> dict:
    request_id = uuid.uuid4().hex[:12]
    start_total = time.perf_counter()

    is_error = False
    retrieval_time_ms = 0.0
    ranking_time_ms = 0.0
    candidate_count = 0
    cache_hit = False

    effective_retrieval_mode = str(retrieval_mode or RETRIEVAL_MODE_DEFAULT)
    default_ranker_enabled = bool(CTX.use_ranker_score)
    if use_ranker is None:
        effective_ranker_enabled = bool(default_ranker_enabled)
    else:
        effective_ranker_enabled = bool(use_ranker)

    guardrail_reason = "guardrail_enabled" if default_ranker_enabled else "guardrail_disabled"
    if use_ranker is not None and bool(use_ranker) != default_ranker_enabled:
        if bool(use_ranker):
            guardrail_reason = "manual_override_guardrail_disabled"
        else:
            guardrail_reason = "manual_disable"

    try:
        if CTX.item_store is None or CTX.user_context is None:
            raise RuntimeError("Service artifacts are not loaded")

        user_idx = CTX.user_map.get(int(user_id))
        if user_idx is None:
            recs = []
            for pos, item_idx in enumerate(CTX.item_store.global_popular_items[: int(n)], start=1):
                item_idx_i = int(item_idx)
                item_id_i = int(CTX.item_idx_to_id.get(item_idx_i, -1))
                meta = CTX.item_meta.get(item_idx_i, {"title": f"Item {item_id_i}", "genres": "unknown"})
                rec = {
                    "rank": int(pos),
                    "item_id": item_id_i,
                    "item_idx": item_idx_i,
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "score": 0.0,
                    "score_type": "cold_start_popularity",
                    "retrieval_score": 0.0,
                    "ranker_score": None,
                    "retrieval_rank": float(pos),
                }
                if explain:
                    rec["explanation"] = {
                        "retrieval_score": 0.0,
                        "ranker_score": None,
                        "retrieval_rank": float(pos),
                        "rrf_score": None,
                        "feature_values": {
                            "cold_start": 1.0,
                            "item_popularity": float(CTX.item_store.item_popularity.get(item_idx_i, 0.0)),
                        },
                    }
                recs.append(rec)

            elapsed_ms = (time.perf_counter() - start_total) * 1000.0
            metric_snapshot = CTX.metrics.summary()
            return {
                "request_id": request_id,
                "timestamp_utc": _now_utc_iso(),
                "user_id": int(user_id),
                "user_idx": None,
                "cold_start": True,
                "candidate_count": len(recs),
                "params": {
                    "n": int(n),
                    "candidate_k": int(candidate_k),
                    "retrieval_mode": effective_retrieval_mode,
                    "use_ranker": bool(effective_ranker_enabled),
                    "explain": bool(explain),
                },
                "model": {
                    "model_version": CTX.model_version,
                    "retrieval_mode": effective_retrieval_mode,
                    "ranker_default_enabled": bool(default_ranker_enabled),
                    "ranker_used": False,
                    "score_blend_alpha": float(CTX.score_blend_alpha),
                },
                "guardrail": {
                    "enabled": bool(default_ranker_enabled),
                    "ranker_used": False,
                    "reason": "cold_start_popularity_fallback",
                },
                "metrics": {
                    "latency_ms": float(elapsed_ms),
                    "retrieval_time_ms": 0.0,
                    "ranking_time_ms": 0.0,
                    "candidate_count": int(len(recs)),
                    "cache_hit": False,
                    "p50_ms": float(metric_snapshot.get("p50_ms", 0.0)),
                    "p95_ms": float(metric_snapshot.get("p95_ms", 0.0)),
                },
                "recommendations": recs,
            }

        user_idx_i = int(user_idx)
        cache_hit = user_idx_i in CTX.user_embedding_cache

        retrieval_start = time.perf_counter()
        if effective_retrieval_mode == "hybrid":
            candidates, scores = retrieve_hybrid_candidates_for_users(
                users=[user_idx_i],
                model=CTX.model_ret,
                index=CTX.index,
                user_histories=CTX.user_histories,
                max_history_len=int(CTX.ret_meta["max_history_len"]),
                top_k=int(candidate_k),
                item_cf_neighbors=CTX.item_cf_neighbors,
                global_pop_ranked_items=CTX.global_pop_ranked_items,
                two_tower_weight=1.0,
                item_cf_weight=1.0,
                rrf_k=60,
                device="cpu",
                user_embedding_cache=CTX.user_embedding_cache,
            )
        else:
            candidates, scores = retrieve_candidates_for_users(
                users=[user_idx_i],
                model=CTX.model_ret,
                index=CTX.index,
                user_histories=CTX.user_histories,
                max_history_len=int(CTX.ret_meta["max_history_len"]),
                top_k=int(candidate_k),
                device="cpu",
                user_embedding_cache=CTX.user_embedding_cache,
            )
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000.0
        candidate_count = len(candidates.get(user_idx_i, []))

        frame = build_ranking_frame(
            candidates_by_user=candidates,
            scores_by_user=scores,
            item_store=CTX.item_store,
            user_context=CTX.user_context,
            context_timestamps={user_idx_i: int(time.time())},
            ground_truth=None,
            item_cf_neighbors=CTX.item_cf_neighbors if effective_retrieval_mode == "hybrid" else None,
        )
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"No candidates found for user_id={user_id}")

        if effective_ranker_enabled:
            ranking_start = time.perf_counter()
            frame["rank_score"] = predict_scores(CTX.ranker, frame, CTX.feature_cols)
            frame["final_score"] = frame["rank_score"] + CTX.score_blend_alpha * frame["retrieval_score"]
            ranking_time_ms = (time.perf_counter() - ranking_start) * 1000.0
        else:
            frame["final_score"] = frame["retrieval_score"]

        frame = frame.sort_values("final_score", ascending=False).head(int(n)).copy()

        recs = []
        explain_cols = [
            "user_genre_affinity",
            "cooccurrence_score",
            "item_popularity",
            "user_mean_rating",
            "item_mean_rating",
            "user_item_rating_delta",
            "days_since_similar",
            "category_match",
        ]
        for pos, row in enumerate(frame.itertuples(index=False), start=1):
            item_idx = int(row.item_idx)
            item_id = int(CTX.item_idx_to_id.get(item_idx, -1))
            meta = CTX.item_meta.get(item_idx, {"title": f"Item {item_id}", "genres": "unknown"})
            rec = {
                "rank": int(pos),
                "item_id": item_id,
                "item_idx": item_idx,
                "title": meta["title"],
                "genres": meta["genres"],
                "score": float(row.final_score),
                "score_type": "ranker" if effective_ranker_enabled else "retrieval",
                "retrieval_score": float(row.retrieval_score),
                "ranker_score": float(getattr(row, "rank_score")) if hasattr(row, "rank_score") else None,
                "retrieval_rank": float(row.retrieval_rank),
            }
            if effective_retrieval_mode == "hybrid":
                rec["rrf_score"] = float(row.retrieval_score)

            if explain:
                feat_values: Dict[str, float] = {}
                for col in explain_cols:
                    if hasattr(row, col):
                        feat_values[col] = float(getattr(row, col))
                rec["explanation"] = {
                    "retrieval_score": float(row.retrieval_score),
                    "ranker_score": float(getattr(row, "rank_score")) if hasattr(row, "rank_score") else None,
                    "retrieval_rank": float(row.retrieval_rank),
                    "rrf_score": float(row.retrieval_score) if effective_retrieval_mode == "hybrid" else None,
                    "feature_values": feat_values,
                }
            recs.append(rec)

        elapsed_ms = (time.perf_counter() - start_total) * 1000.0
        metric_snapshot = CTX.metrics.summary()

        train_metrics = _load_json_if_exists(RANKING_DIR / "train_metrics.json") or {}
        guardrail = train_metrics.get("guardrail", {}) if isinstance(train_metrics, dict) else {}

        return {
            "request_id": request_id,
            "timestamp_utc": _now_utc_iso(),
            "user_id": int(user_id),
            "user_idx": user_idx_i,
            "cold_start": False,
            "candidate_count": int(candidate_count),
            "params": {
                "n": int(n),
                "candidate_k": int(candidate_k),
                "retrieval_mode": effective_retrieval_mode,
                "use_ranker": bool(effective_ranker_enabled),
                "explain": bool(explain),
            },
            "model": {
                "model_version": CTX.model_version,
                "retrieval_mode": effective_retrieval_mode,
                "ranker_default_enabled": bool(default_ranker_enabled),
                "ranker_used": bool(effective_ranker_enabled),
                "score_blend_alpha": float(CTX.score_blend_alpha),
            },
            "guardrail": {
                "enabled": bool(default_ranker_enabled),
                "ranker_used": bool(effective_ranker_enabled),
                "reason": guardrail_reason,
                "p_lift_gt_zero": float(guardrail.get("p_lift_gt_zero", 0.0)) if isinstance(guardrail, dict) else 0.0,
                "median_lift": float(guardrail.get("median_lift", 0.0)) if isinstance(guardrail, dict) else 0.0,
                "ci95": [
                    float(guardrail.get("ci95_low", 0.0)) if isinstance(guardrail, dict) else 0.0,
                    float(guardrail.get("ci95_high", 0.0)) if isinstance(guardrail, dict) else 0.0,
                ],
            },
            "metrics": {
                "latency_ms": float(elapsed_ms),
                "retrieval_time_ms": float(retrieval_time_ms),
                "ranking_time_ms": float(ranking_time_ms),
                "candidate_count": int(candidate_count),
                "cache_hit": bool(cache_hit),
                "p50_ms": float(metric_snapshot.get("p50_ms", 0.0)),
                "p95_ms": float(metric_snapshot.get("p95_ms", 0.0)),
            },
            "recommendations": recs,
        }

    except HTTPException:
        is_error = True
        raise
    except Exception as exc:
        is_error = True
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        elapsed_ms = (time.perf_counter() - start_total) * 1000.0
        CTX.metrics.record(latency_ms=elapsed_ms, is_error=is_error)
        _log_request(
            {
                "timestamp_utc": _now_utc_iso(),
                "request_id": request_id,
                "user_id": int(user_id),
                "candidate_count": int(candidate_count),
                "retrieval_mode": effective_retrieval_mode,
                "use_ranker": bool(effective_ranker_enabled),
                "latency_ms": float(elapsed_ms),
                "retrieval_time_ms": float(retrieval_time_ms),
                "ranking_time_ms": float(ranking_time_ms),
                "cache_hit": bool(cache_hit),
                "error": bool(is_error),
            }
        )
