#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, TextIO

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.baselines import build_item_cf_neighbors, build_item_popularity, popularity_ranking
from recsys.features import ItemFeatureStore, build_ranking_frame, build_user_context
from recsys.io import load_all_splits, load_items, load_users
from recsys.paths import LOG_DIR, PROCESSED_DATA_DIR, RANKING_DIR, RETRIEVAL_DIR, ensure_dirs
from recsys.pipeline import retrieve_candidates_for_users, retrieve_hybrid_candidates_for_users
from recsys.ranking import load_ranker, predict_scores
from recsys.retrieval import CandidateIndex, build_user_history, compute_user_embeddings_batch, load_two_tower_model

app = FastAPI(title="Two-Stage Recommender API", version="0.1.0")
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")


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
        self.item_idx_to_id: Dict[int, int] = {}
        self.user_histories: Dict[int, List[int]] = {}
        self.user_embedding_cache: Dict[int, object] = {}
        self.item_cf_neighbors: Dict[int, List[tuple[int, float]]] = {}
        self.global_pop_ranked_items: List[int] = []
        self.item_store: ItemFeatureStore | None = None
        self.user_context = None
        self.metrics = MetricsState()
        self.log_path = LOG_DIR / "service_requests.jsonl"
        self.log_file: TextIO | None = None
        self.log_lock = Lock()
        self.pending_log_lines = 0


CTX = ServiceContext()


def _log_request(payload: dict) -> None:
    if CTX.log_file is None:
        return
    line = json.dumps(payload) + "\n"
    with CTX.log_lock:
        CTX.log_file.write(line)
        CTX.pending_log_lines += 1
        # Flush periodically to keep data visible without per-request fs overhead.
        if CTX.pending_log_lines >= 100:
            CTX.log_file.flush()
            CTX.pending_log_lines = 0


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()
    splits = load_all_splits(data_dir=PROCESSED_DATA_DIR)
    history_df = pd.concat([splits["train"], splits["val"]], ignore_index=True)

    users_df = load_users(data_dir=PROCESSED_DATA_DIR)
    items_df = load_items(data_dir=PROCESSED_DATA_DIR)

    CTX.user_map = {int(row.user_id): int(row.user_idx) for row in users_df.itertuples(index=False)}
    CTX.item_idx_to_id = {int(row.item_idx): int(row.item_id) for row in items_df.itertuples(index=False)}

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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    return CTX.metrics.summary()


@app.get("/recommend")
def recommend(
    user_id: int = Query(..., description="Original user_id"),
    n: int = Query(10, ge=1, le=100),
    candidate_k: int = Query(200, ge=10, le=1000),
) -> dict:
    start = time.perf_counter()
    is_error = False
    candidate_count = 0

    try:
        if CTX.item_store is None or CTX.user_context is None:
            raise RuntimeError("Service artifacts are not loaded")

        user_idx = CTX.user_map.get(user_id)
        if user_idx is None:
            recs = []
            for item_idx in CTX.item_store.global_popular_items[:n]:
                recs.append(
                    {
                        "item_id": int(CTX.item_idx_to_id.get(item_idx, -1)),
                        "item_idx": int(item_idx),
                        "score": 0.0,
                    }
                )
            response = {
                "user_id": user_id,
                "cold_start": True,
                "candidate_count": len(recs),
                "recommendations": recs,
            }
            return response

        if RETRIEVAL_MODE == "hybrid":
            candidates, scores = retrieve_hybrid_candidates_for_users(
                users=[user_idx],
                model=CTX.model_ret,
                index=CTX.index,
                user_histories=CTX.user_histories,
                max_history_len=int(CTX.ret_meta["max_history_len"]),
                top_k=candidate_k,
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
                users=[user_idx],
                model=CTX.model_ret,
                index=CTX.index,
                user_histories=CTX.user_histories,
                max_history_len=int(CTX.ret_meta["max_history_len"]),
                top_k=candidate_k,
                device="cpu",
                user_embedding_cache=CTX.user_embedding_cache,
            )
        candidate_count = len(candidates.get(user_idx, []))

        frame = build_ranking_frame(
            candidates_by_user=candidates,
            scores_by_user=scores,
            item_store=CTX.item_store,
            user_context=CTX.user_context,
            context_timestamps={user_idx: int(time.time())},
            ground_truth=None,
            item_cf_neighbors=CTX.item_cf_neighbors if RETRIEVAL_MODE == "hybrid" else None,
        )
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"No candidates found for user_id={user_id}")

        if CTX.use_ranker_score:
            frame["rank_score"] = predict_scores(CTX.ranker, frame, CTX.feature_cols)
            frame["final_score"] = frame["rank_score"] + CTX.score_blend_alpha * frame["retrieval_score"]
        else:
            frame["final_score"] = frame["retrieval_score"]
        frame = frame.sort_values("final_score", ascending=False).head(n)

        recs = []
        for row in frame.itertuples(index=False):
            recs.append(
                    {
                        "item_id": int(CTX.item_idx_to_id.get(int(row.item_idx), -1)),
                        "item_idx": int(row.item_idx),
                        "score": float(row.final_score),
                    }
                )

        response = {
            "user_id": user_id,
            "cold_start": False,
            "candidate_count": candidate_count,
            "recommendations": recs,
        }
        return response

    except HTTPException:
        is_error = True
        raise
    except Exception as exc:
        is_error = True
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        CTX.metrics.record(latency_ms=elapsed_ms, is_error=is_error)
        _log_request(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "candidate_count": int(candidate_count),
                "latency_ms": float(elapsed_ms),
                "error": bool(is_error),
            }
        )
