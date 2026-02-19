from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from recsys.retrieval import load_pickle, save_pickle

YEAR_RE = re.compile(r"\((\d{4})\)")


@dataclass
class ItemFeatureStore:
    item_popularity: Dict[int, float]
    item_main_genre: Dict[int, str]
    item_year_bucket: Dict[int, int]
    genre_to_id: Dict[str, int]
    global_popular_items: List[int]

    def save(self, path: Path) -> None:
        save_pickle(self, path)

    @classmethod
    def load(cls, path: Path) -> "ItemFeatureStore":
        obj = load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Unexpected object type in {path}: {type(obj)}")
        return obj


@dataclass
class UserContext:
    user_activity: Dict[int, int]
    user_last_ts: Dict[int, int]
    user_top_genres: Dict[int, set[str]]
    user_genre_last_ts: Dict[int, Dict[str, int]]



def _extract_year(title: str) -> int | None:
    if not isinstance(title, str):
        return None
    match = YEAR_RE.search(title)
    if not match:
        return None
    return int(match.group(1))


def _year_to_bucket(year: int | None) -> int:
    if year is None:
        return 0
    return max(1, (year // 10) * 10)


def _main_genre(genres: str) -> str:
    if not isinstance(genres, str) or not genres:
        return "unknown"
    first = genres.split("|")[0].strip().lower()
    return first if first else "unknown"


def build_item_feature_store(train_df: pd.DataFrame, items_df: pd.DataFrame) -> ItemFeatureStore:
    item_popularity_series = train_df.groupby("item_idx").size().sort_values(ascending=False)
    total_interactions = float(item_popularity_series.sum())
    item_popularity = {
        int(item): float(count / total_interactions) if total_interactions > 0 else 0.0
        for item, count in item_popularity_series.items()
    }

    item_main_genre = {}
    item_year_bucket = {}
    for row in items_df[["item_idx", "title", "genres"]].itertuples(index=False):
        item_idx = int(row[0])
        title = str(row[1]) if row[1] is not None else ""
        genres = str(row[2]) if row[2] is not None else ""
        main_genre = _main_genre(genres)
        item_main_genre[item_idx] = main_genre
        item_year_bucket[item_idx] = _year_to_bucket(_extract_year(title))

    genre_vocab = sorted(set(item_main_genre.values()) | {"unknown"})
    genre_to_id = {genre: idx for idx, genre in enumerate(genre_vocab)}

    global_popular_items = [int(item) for item in item_popularity_series.index.tolist()]
    return ItemFeatureStore(
        item_popularity=item_popularity,
        item_main_genre=item_main_genre,
        item_year_bucket=item_year_bucket,
        genre_to_id=genre_to_id,
        global_popular_items=global_popular_items,
    )


def build_user_context(history_df: pd.DataFrame, item_main_genre: Mapping[int, str]) -> UserContext:
    if history_df.empty:
        return UserContext(user_activity={}, user_last_ts={}, user_top_genres={}, user_genre_last_ts={})

    ordered = history_df.sort_values(["user_idx", "timestamp"])

    user_activity: Dict[int, int] = {}
    user_last_ts: Dict[int, int] = {}
    user_genre_counts: Dict[int, Dict[str, int]] = {}
    user_genre_last_ts: Dict[int, Dict[str, int]] = {}

    for row in ordered[["user_idx", "item_idx", "timestamp"]].itertuples(index=False):
        user = int(row[0])
        item = int(row[1])
        ts = int(row[2])
        genre = item_main_genre.get(item, "unknown")

        user_activity[user] = user_activity.get(user, 0) + 1
        user_last_ts[user] = ts

        genre_counts = user_genre_counts.setdefault(user, {})
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

        genre_last = user_genre_last_ts.setdefault(user, {})
        genre_last[genre] = ts

    user_top_genres: Dict[int, set[str]] = {}
    for user, counts in user_genre_counts.items():
        sorted_genres = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        user_top_genres[user] = {genre for genre, _ in sorted_genres[:3]}

    return UserContext(
        user_activity=user_activity,
        user_last_ts=user_last_ts,
        user_top_genres=user_top_genres,
        user_genre_last_ts=user_genre_last_ts,
    )


def _safe_days(delta_seconds: int | float) -> float:
    return max(float(delta_seconds) / 86400.0, 0.0)


def build_ranking_frame(
    candidates_by_user: Mapping[int, Sequence[int]],
    scores_by_user: Mapping[int, Sequence[float]],
    item_store: ItemFeatureStore,
    user_context: UserContext,
    context_timestamps: Mapping[int, int],
    ground_truth: Mapping[int, set[int]] | None = None,
) -> pd.DataFrame:
    rows: List[dict] = []

    for user, candidates in candidates_by_user.items():
        user = int(user)
        scores = list(scores_by_user.get(user, []))
        if len(scores) < len(candidates):
            scores = scores + [0.0] * (len(candidates) - len(scores))

        activity = user_context.user_activity.get(user, 0)
        user_last_ts = user_context.user_last_ts.get(user)
        context_ts = int(context_timestamps.get(user, user_last_ts or 0))
        user_top = user_context.user_top_genres.get(user, set())
        genre_last_ts = user_context.user_genre_last_ts.get(user, {})

        for rank_pos, (item, ret_score) in enumerate(zip(candidates, scores), start=1):
            item = int(item)
            genre = item_store.item_main_genre.get(item, "unknown")
            genre_id = item_store.genre_to_id.get(genre, 0)
            similar_last_ts = genre_last_ts.get(genre)

            label = 0
            if ground_truth is not None:
                label = int(item in ground_truth.get(user, set()))

            if user_last_ts is None:
                days_since_last = 999.0
            else:
                days_since_last = _safe_days(context_ts - user_last_ts)

            if similar_last_ts is None:
                days_since_similar = 999.0
            else:
                days_since_similar = _safe_days(context_ts - similar_last_ts)

            rows.append(
                {
                    "user_idx": user,
                    "item_idx": item,
                    "retrieval_score": float(ret_score),
                    "retrieval_rank": float(rank_pos),
                    "user_activity": float(activity),
                    "user_days_since_last": float(days_since_last),
                    "item_popularity": float(item_store.item_popularity.get(item, 0.0)),
                    "item_year_bucket": float(item_store.item_year_bucket.get(item, 0)),
                    "item_genre_id": float(genre_id),
                    "category_match": float(1.0 if genre in user_top else 0.0),
                    "days_since_similar": float(days_since_similar),
                    "label": int(label),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "user_idx",
                "item_idx",
                "retrieval_score",
                "retrieval_rank",
                "user_activity",
                "user_days_since_last",
                "item_popularity",
                "item_year_bucket",
                "item_genre_id",
                "category_match",
                "days_since_similar",
                "label",
            ]
        )

    return pd.DataFrame(rows)


def feature_columns() -> List[str]:
    return [
        "retrieval_score",
        "retrieval_rank",
        "user_activity",
        "user_days_since_last",
        "item_popularity",
        "item_year_bucket",
        "item_genre_id",
        "category_match",
        "days_since_similar",
    ]
