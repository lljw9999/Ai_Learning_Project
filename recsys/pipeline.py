from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from recsys.baselines import recommend_item_cf_with_scores
from recsys.retrieval import CandidateIndex, compute_user_embedding, recommend_from_index


def retrieve_candidates_for_users(
    users: Iterable[int],
    model,
    index: CandidateIndex,
    user_histories: Mapping[int, Sequence[int]],
    max_history_len: int,
    top_k: int,
    device: str = "cpu",
    user_embedding_cache: Dict[int, np.ndarray] | None = None,
) -> Tuple[Dict[int, list[int]], Dict[int, list[float]]]:
    candidates_by_user: Dict[int, list[int]] = {}
    scores_by_user: Dict[int, list[float]] = {}

    for user in users:
        user_int = int(user)
        history = list(user_histories.get(user_int, []))
        if user_embedding_cache is not None and user_int in user_embedding_cache:
            user_vec = user_embedding_cache[user_int]
        else:
            user_vec = compute_user_embedding(
                model=model,
                user_idx=user_int,
                history=history,
                max_history_len=max_history_len,
                device=device,
            )
            if user_embedding_cache is not None:
                user_embedding_cache[user_int] = user_vec
        item_ids, scores = recommend_from_index(
            query_vector=user_vec,
            index=index,
            top_k=top_k,
            seen_items=history,
        )
        candidates_by_user[user_int] = item_ids
        scores_by_user[user_int] = scores

    return candidates_by_user, scores_by_user


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
            item_int = int(item)
            score_map[item_int] += float(weight / (rrf_k + rank))

    ranked = sorted(score_map.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [int(item) for item, _ in ranked], [float(score) for _, score in ranked]


def retrieve_hybrid_candidates_for_users(
    users: Iterable[int],
    model,
    index: CandidateIndex,
    user_histories: Mapping[int, Sequence[int]],
    max_history_len: int,
    top_k: int,
    item_cf_neighbors: Mapping[int, Sequence[Tuple[int, float]]],
    global_pop_ranked_items: Sequence[int],
    two_tower_weight: float = 1.0,
    item_cf_weight: float = 1.0,
    cf_top_k: int | None = None,
    rrf_k: int = 60,
    device: str = "cpu",
    user_embedding_cache: Dict[int, np.ndarray] | None = None,
) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    users_list = [int(u) for u in users]
    if not users_list:
        return {}, {}

    cf_k = int(cf_top_k or top_k)
    tt_items, _tt_scores = retrieve_candidates_for_users(
        users=users_list,
        model=model,
        index=index,
        user_histories=user_histories,
        max_history_len=max_history_len,
        top_k=top_k,
        device=device,
        user_embedding_cache=user_embedding_cache,
    )
    cf_items, _cf_scores = recommend_item_cf_with_scores(
        users=users_list,
        user_histories=user_histories,
        item_cf_neighbors=item_cf_neighbors,
        fallback_ranked_items=global_pop_ranked_items,
        top_k=cf_k,
    )

    fused_items: Dict[int, List[int]] = {}
    fused_scores: Dict[int, List[float]] = {}
    for user in users_list:
        items, scores = _fuse_ranked_lists_rrf(
            ranked_sources=[
                (tt_items.get(user, []), float(two_tower_weight)),
                (cf_items.get(user, []), float(item_cf_weight)),
            ],
            top_k=top_k,
            rrf_k=rrf_k,
        )
        fused_items[user] = items
        fused_scores[user] = scores

    return fused_items, fused_scores


def ensure_truth_in_candidates(
    candidates_by_user: Dict[int, list[int]],
    scores_by_user: Dict[int, list[float]],
    ground_truth: Mapping[int, set[int]],
) -> None:
    for user, truth_items in ground_truth.items():
        if user not in candidates_by_user:
            candidates_by_user[user] = []
            scores_by_user[user] = []
        existing = set(candidates_by_user[user])
        for item in truth_items:
            if item not in existing:
                candidates_by_user[user].append(int(item))
                scores_by_user[user].append(0.0)


def context_timestamp_map(split_df: pd.DataFrame) -> Dict[int, int]:
    if split_df.empty:
        return {}
    grouped = split_df.groupby("user_idx")["timestamp"].min()
    return {int(user): int(ts) for user, ts in grouped.items()}
