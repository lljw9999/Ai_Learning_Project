from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from recsys.baselines import item_cf_scored_candidates_for_history
from recsys.features import ItemFeatureStore, _safe_days
from recsys.retrieval import CandidateIndex, compute_user_embedding, recommend_from_index


@dataclass
class QueryExample:
    query_id: int
    user_idx: int
    history_items: List[int]
    history_timestamps: List[int]
    target_item: int
    target_timestamp: int


def build_train_queries(
    interactions: pd.DataFrame,
    min_history: int = 5,
    max_queries_per_user: int = 15,
) -> List[QueryExample]:
    if interactions.empty:
        return []

    queries: List[QueryExample] = []
    next_query_id = 0

    ordered = interactions.sort_values(["user_idx", "timestamp"])
    for user_idx, group in ordered.groupby("user_idx"):
        user = int(user_idx)
        items = [int(v) for v in group["item_idx"].tolist()]
        timestamps = [int(v) for v in group["timestamp"].tolist()]

        if len(items) <= min_history:
            continue

        candidate_positions = list(range(min_history, len(items)))
        if max_queries_per_user > 0 and len(candidate_positions) > max_queries_per_user:
            # Keep the latest queries to emphasize recent behavior.
            candidate_positions = candidate_positions[-max_queries_per_user:]

        for pos in candidate_positions:
            history_items = items[:pos]
            history_timestamps = timestamps[:pos]
            queries.append(
                QueryExample(
                    query_id=next_query_id,
                    user_idx=user,
                    history_items=history_items,
                    history_timestamps=history_timestamps,
                    target_item=items[pos],
                    target_timestamp=timestamps[pos],
                )
            )
            next_query_id += 1

    return queries


def _user_genre_profile(
    history_items: Sequence[int],
    history_timestamps: Sequence[int],
    item_main_genre: Mapping[int, str],
) -> tuple[set[str], Dict[str, int]]:
    genre_counts: Dict[str, int] = {}
    genre_last_ts: Dict[str, int] = {}

    for item, ts in zip(history_items, history_timestamps):
        genre = item_main_genre.get(int(item), "unknown")
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        genre_last_ts[genre] = int(ts)

    sorted_genres = sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))
    top_genres = {genre for genre, _ in sorted_genres[:3]}
    return top_genres, genre_last_ts


def _normalized_query(query_vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(query_vector))
    if norm <= 1e-12:
        return query_vector.astype(np.float32)
    return (query_vector / norm).astype(np.float32)


def _fuse_ranked_lists_rrf(
    ranked_sources: Sequence[Tuple[Sequence[int], float]],
    top_k: int,
    rrf_k: int = 60,
) -> Tuple[List[int], List[float]]:
    score_map: Dict[int, float] = defaultdict(float)
    for items, weight in ranked_sources:
        if weight <= 0:
            continue
        for rank, item in enumerate(items, start=1):
            score_map[int(item)] += float(weight / (rrf_k + rank))
    ranked = sorted(score_map.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [int(item) for item, _ in ranked], [float(score) for _, score in ranked]


def build_ranker_training_frame(
    queries: Sequence[QueryExample],
    model,
    index: CandidateIndex,
    item_store: ItemFeatureStore,
    max_history_len: int,
    candidate_k: int,
    include_missed_positive: bool = False,
    item_cf_neighbors: Mapping[int, Sequence[Tuple[int, float]]] | None = None,
    global_pop_ranked_items: Sequence[int] | None = None,
    two_tower_weight: float = 1.0,
    item_cf_weight: float = 1.0,
    rrf_k: int = 60,
    device: str = "cpu",
) -> pd.DataFrame:
    rows: List[dict] = []

    for query in queries:
        user_vec = compute_user_embedding(
            model=model,
            user_idx=query.user_idx,
            history=query.history_items,
            max_history_len=max_history_len,
            device=device,
        )

        tt_items, tt_scores = recommend_from_index(
            query_vector=user_vec,
            index=index,
            top_k=candidate_k,
            seen_items=query.history_items,
        )

        candidate_items = list(tt_items)
        retrieval_scores = list(tt_scores)
        if item_cf_neighbors is not None:
            cf_ranked = item_cf_scored_candidates_for_history(
                history=query.history_items,
                item_cf_neighbors=item_cf_neighbors,
                fallback_ranked_items=list(global_pop_ranked_items or []),
                top_k=candidate_k,
            )
            cf_items = [int(item) for item, _ in cf_ranked]
            merged_items, merged_scores = _fuse_ranked_lists_rrf(
                ranked_sources=[
                    (candidate_items, float(two_tower_weight)),
                    (cf_items, float(item_cf_weight)),
                ],
                top_k=candidate_k,
                rrf_k=rrf_k,
            )
            candidate_items = merged_items
            retrieval_scores = merged_scores

        item_to_score = {int(item): float(score) for item, score in zip(candidate_items, retrieval_scores)}

        if query.target_item not in item_to_score:
            if include_missed_positive:
                qvec = _normalized_query(user_vec)
                target_score = float(index.item_vectors[int(query.target_item)] @ qvec)
                item_to_score[int(query.target_item)] = target_score
            else:
                continue

        ranked_candidates = sorted(item_to_score.items(), key=lambda x: x[1], reverse=True)

        user_activity = len(query.history_items)
        user_last_ts = int(query.history_timestamps[-1]) if query.history_timestamps else None
        top_genres, genre_last_ts = _user_genre_profile(
            history_items=query.history_items,
            history_timestamps=query.history_timestamps,
            item_main_genre=item_store.item_main_genre,
        )

        for rank_pos, (item_idx, retrieval_score) in enumerate(ranked_candidates, start=1):
            item = int(item_idx)
            genre = item_store.item_main_genre.get(item, "unknown")
            similar_last_ts = genre_last_ts.get(genre)

            if user_last_ts is None:
                days_since_last = 999.0
            else:
                days_since_last = _safe_days(query.target_timestamp - user_last_ts)

            if similar_last_ts is None:
                days_since_similar = 999.0
            else:
                days_since_similar = _safe_days(query.target_timestamp - similar_last_ts)

            rows.append(
                {
                    "query_id": int(query.query_id),
                    "user_idx": int(query.user_idx),
                    "item_idx": int(item),
                    "retrieval_score": float(retrieval_score),
                    "retrieval_rank": float(rank_pos),
                    "user_activity": float(user_activity),
                    "user_days_since_last": float(days_since_last),
                    "item_popularity": float(item_store.item_popularity.get(item, 0.0)),
                    "item_year_bucket": float(item_store.item_year_bucket.get(item, 0)),
                    "item_genre_id": float(item_store.genre_to_id.get(genre, 0)),
                    "category_match": float(1.0 if genre in top_genres else 0.0),
                    "days_since_similar": float(days_since_similar),
                    "label": int(item == query.target_item),
                }
            )

    return pd.DataFrame(rows)
