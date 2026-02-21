from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def recall_at_k(
    predictions: Dict[int, Sequence[int]],
    ground_truth: Dict[int, set[int]],
    k: int,
) -> float:
    recalls: List[float] = []
    for user, truth_items in ground_truth.items():
        if not truth_items:
            continue
        pred_items = list(predictions.get(user, []))[:k]
        hits = len(set(pred_items) & truth_items)
        recalls.append(hits / len(truth_items))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def dcg_at_k(pred_items: Sequence[int], truth_items: set[int], k: int) -> float:
    score = 0.0
    for rank, item in enumerate(pred_items[:k], start=1):
        if item in truth_items:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(
    predictions: Dict[int, Sequence[int]],
    ground_truth: Dict[int, set[int]],
    k: int,
) -> float:
    ndcgs: List[float] = []
    for user, truth_items in ground_truth.items():
        if not truth_items:
            continue
        pred_items = list(predictions.get(user, []))[:k]
        dcg = dcg_at_k(pred_items, truth_items, k)
        ideal_list = [1] * min(k, len(truth_items))
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_list, start=1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0


def per_user_ndcg_at_k(
    predictions: Dict[int, Sequence[int]],
    ground_truth: Dict[int, set[int]],
    k: int,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for user, truth_items in ground_truth.items():
        if not truth_items:
            continue
        pred_items = list(predictions.get(user, []))[:k]
        dcg = dcg_at_k(pred_items, truth_items, k)
        ideal_list = [1] * min(k, len(truth_items))
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_list, start=1))
        scores[int(user)] = float(dcg / idcg if idcg > 0 else 0.0)
    return scores


def average_precision(pred_items: Sequence[int], truth_items: set[int], k: int) -> float:
    if not truth_items:
        return 0.0
    hits = 0
    precisions: List[float] = []
    for idx, item in enumerate(pred_items[:k], start=1):
        if item in truth_items:
            hits += 1
            precisions.append(hits / idx)
    if not precisions:
        return 0.0
    return sum(precisions) / min(len(truth_items), k)


def map_at_k(
    predictions: Dict[int, Sequence[int]],
    ground_truth: Dict[int, set[int]],
    k: int,
) -> float:
    ap_scores: List[float] = []
    for user, truth_items in ground_truth.items():
        if not truth_items:
            continue
        pred_items = list(predictions.get(user, []))[:k]
        ap_scores.append(average_precision(pred_items, truth_items, k))
    return float(sum(ap_scores) / len(ap_scores)) if ap_scores else 0.0


def summarize_retrieval_metrics(
    predictions: Dict[int, Sequence[int]],
    ground_truth: Dict[int, set[int]],
    ks: Iterable[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(predictions, ground_truth, k)
    return out


def summarize_ranking_metrics(
    predictions: Dict[int, Sequence[int]],
    ground_truth: Dict[int, set[int]],
    ks: Iterable[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(predictions, ground_truth, k)
        out[f"map@{k}"] = map_at_k(predictions, ground_truth, k)
    return out


def bootstrap_lift_confidence(
    baseline_scores: Mapping[int, float],
    candidate_scores: Mapping[int, float],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, float | int]:
    users = sorted(set(int(u) for u in baseline_scores.keys()) & set(int(u) for u in candidate_scores.keys()))
    if not users:
        return {
            "n_users": 0,
            "n_bootstrap": int(max(n_bootstrap, 1)),
            "observed_mean_lift": 0.0,
            "p_lift_gt_zero": 0.0,
            "median_lift": 0.0,
            "mean_lift": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }

    base = np.asarray([float(baseline_scores[u]) for u in users], dtype=np.float64)
    cand = np.asarray([float(candidate_scores[u]) for u in users], dtype=np.float64)
    diff = cand - base
    n_users = int(diff.shape[0])
    n_boot = int(max(n_bootstrap, 1))

    rng = np.random.default_rng(random_state)
    sample_idx = rng.integers(0, n_users, size=(n_boot, n_users))
    sampled = diff[sample_idx].mean(axis=1)

    return {
        "n_users": n_users,
        "n_bootstrap": n_boot,
        "observed_mean_lift": float(diff.mean()),
        "p_lift_gt_zero": float(np.mean(sampled > 0.0)),
        "median_lift": float(np.median(sampled)),
        "mean_lift": float(np.mean(sampled)),
        "ci95_low": float(np.percentile(sampled, 2.5)),
        "ci95_high": float(np.percentile(sampled, 97.5)),
    }
