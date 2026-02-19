from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def build_item_popularity(train_df: pd.DataFrame) -> Dict[int, float]:
    counts = train_df.groupby("item_idx").size().sort_values(ascending=False)
    total = float(counts.sum())
    if total <= 0:
        return {}
    return {int(item): float(cnt / total) for item, cnt in counts.items()}


def popularity_ranking(popularity: Mapping[int, float]) -> List[int]:
    return [int(item) for item, _ in sorted(popularity.items(), key=lambda x: (-x[1], x[0]))]


def recommend_popularity(
    users: Iterable[int],
    user_histories: Mapping[int, Sequence[int]],
    global_pop_ranked_items: Sequence[int],
    top_k: int,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for user in users:
        seen = set(int(i) for i in user_histories.get(int(user), []))
        recs: List[int] = []
        for item in global_pop_ranked_items:
            if int(item) in seen:
                continue
            recs.append(int(item))
            if len(recs) >= top_k:
                break
        out[int(user)] = recs
    return out


def build_item_cf_neighbors(
    train_history: Mapping[int, Sequence[int]],
    top_neighbors: int = 200,
    min_pair_count: int = 2,
) -> Dict[int, List[Tuple[int, float]]]:
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    item_counts: Dict[int, int] = defaultdict(int)

    for items in train_history.values():
        unique_items = sorted(set(int(i) for i in items))
        for item in unique_items:
            item_counts[item] += 1
        for a, b in combinations(unique_items, 2):
            pair_counts[(a, b)] += 1

    neighbors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (a, b), c_ab in pair_counts.items():
        if c_ab < min_pair_count:
            continue
        c_a = item_counts[a]
        c_b = item_counts[b]
        if c_a <= 0 or c_b <= 0:
            continue
        # Cosine-like association from implicit co-occurrence.
        score = float(c_ab / np.sqrt(c_a * c_b))
        neighbors[a].append((b, score))
        neighbors[b].append((a, score))

    pruned: Dict[int, List[Tuple[int, float]]] = {}
    for item, nbrs in neighbors.items():
        ranked = sorted(nbrs, key=lambda x: (-x[1], x[0]))[:top_neighbors]
        pruned[int(item)] = [(int(i), float(s)) for i, s in ranked]
    return pruned


def recommend_item_cf(
    users: Iterable[int],
    user_histories: Mapping[int, Sequence[int]],
    item_cf_neighbors: Mapping[int, Sequence[Tuple[int, float]]],
    fallback_ranked_items: Sequence[int],
    top_k: int,
    max_history_for_scoring: int = 30,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}

    for user in users:
        user_int = int(user)
        history = [int(i) for i in user_histories.get(user_int, [])]
        ranked_scored = item_cf_scored_candidates_for_history(
            history=history,
            item_cf_neighbors=item_cf_neighbors,
            fallback_ranked_items=fallback_ranked_items,
            top_k=top_k,
            max_history_for_scoring=max_history_for_scoring,
        )
        ranked = [item for item, _ in ranked_scored]

        out[user_int] = ranked[:top_k]

    return out


def item_cf_scored_candidates_for_history(
    history: Sequence[int],
    item_cf_neighbors: Mapping[int, Sequence[Tuple[int, float]]],
    fallback_ranked_items: Sequence[int],
    top_k: int,
    max_history_for_scoring: int = 30,
) -> List[Tuple[int, float]]:
    seen = set(int(i) for i in history)
    scores: Dict[int, float] = defaultdict(float)

    hist_tail = [int(i) for i in history[-max_history_for_scoring:]]
    for pos, item in enumerate(reversed(hist_tail), start=1):
        weight = 1.0 / np.log2(pos + 1)
        for nbr, sim in item_cf_neighbors.get(item, []):
            nbr_i = int(nbr)
            if nbr_i in seen:
                continue
            scores[nbr_i] += float(weight * sim)

    ranked_scored = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    ranked_items = [int(item) for item, _ in ranked_scored]
    ranked_set = set(ranked_items)

    if len(ranked_scored) < top_k:
        min_score = float(ranked_scored[-1][1]) if ranked_scored else 0.0
        epsilon = 1e-9
        for offset, item in enumerate(fallback_ranked_items):
            item_i = int(item)
            if item_i in seen or item_i in ranked_set:
                continue
            # Keep fallback scores below learned CF scores while preserving order.
            fallback_score = min_score - (offset + 1) * epsilon
            ranked_scored.append((item_i, float(fallback_score)))
            if len(ranked_scored) >= top_k:
                break

    return ranked_scored[:top_k]


def recommend_item_cf_with_scores(
    users: Iterable[int],
    user_histories: Mapping[int, Sequence[int]],
    item_cf_neighbors: Mapping[int, Sequence[Tuple[int, float]]],
    fallback_ranked_items: Sequence[int],
    top_k: int,
    max_history_for_scoring: int = 30,
) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    items_by_user: Dict[int, List[int]] = {}
    scores_by_user: Dict[int, List[float]] = {}

    for user in users:
        user_int = int(user)
        history = [int(i) for i in user_histories.get(user_int, [])]
        ranked_scored = item_cf_scored_candidates_for_history(
            history=history,
            item_cf_neighbors=item_cf_neighbors,
            fallback_ranked_items=fallback_ranked_items,
            top_k=top_k,
            max_history_for_scoring=max_history_for_scoring,
        )
        items_by_user[user_int] = [int(item) for item, _ in ranked_scored]
        scores_by_user[user_int] = [float(score) for _, score in ranked_scored]

    return items_by_user, scores_by_user
