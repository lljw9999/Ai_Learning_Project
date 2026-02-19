#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.io import save_parquet
from recsys.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens into time-based splits")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=RAW_DATA_DIR / "ml-latest-small",
        help="Path to raw MovieLens directory containing ratings.csv and movies.csv",
    )
    parser.add_argument("--min-rating", type=float, default=4.0, help="Keep interactions with rating >= min-rating")
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=3,
        help="Users with fewer interactions are dropped",
    )
    return parser.parse_args()


def _time_based_leave_two_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        if len(group) < 3:
            continue
        train_parts.append(group.iloc[:-2])
        val_parts.append(group.iloc[[-2]])
        test_parts.append(group.iloc[[-1]])

    if not train_parts:
        raise ValueError("No users with enough interactions for leave-two-out split")

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, val_df, test_df


def _build_index_mapping(values: pd.Series, col_name: str) -> pd.DataFrame:
    unique = pd.Series(sorted(values.unique()))
    return pd.DataFrame({col_name: unique.values, f"{col_name.replace('_id', '_idx')}": range(len(unique))})


def main() -> None:
    args = parse_args()
    ensure_dirs()

    ratings_path = args.dataset_dir / "ratings.csv"
    movies_path = args.dataset_dir / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"Expected ratings.csv and movies.csv under {args.dataset_dir}. "
            "Run data/download_movielens.py first."
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    ratings = ratings.rename(
        columns={
            "userId": "user_id",
            "movieId": "item_id",
            "rating": "rating",
            "timestamp": "timestamp",
        }
    )

    ratings = ratings.loc[ratings["rating"] >= args.min_rating].copy()
    ratings["timestamp"] = ratings["timestamp"].astype("int64")

    user_counts = ratings.groupby("user_id").size()
    keep_users = user_counts[user_counts >= args.min_user_interactions].index
    ratings = ratings.loc[ratings["user_id"].isin(keep_users)].copy()

    if ratings.empty:
        raise ValueError("No interactions left after filtering")

    train_df, val_df, test_df = _time_based_leave_two_out(ratings)

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    users = _build_index_mapping(combined["user_id"], "user_id")
    items = _build_index_mapping(combined["item_id"], "item_id")

    def attach_indices(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.merge(users, on="user_id", how="inner")
        frame = frame.merge(items, on="item_id", how="inner")
        frame = frame.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
        return frame[["user_id", "user_idx", "item_id", "item_idx", "rating", "timestamp"]]

    train_df = attach_indices(train_df)
    val_df = attach_indices(val_df)
    test_df = attach_indices(test_df)

    items_df = items.merge(movies.rename(columns={"movieId": "item_id"}), on="item_id", how="left")
    items_df["title"] = items_df["title"].fillna("unknown")
    items_df["genres"] = items_df["genres"].fillna("unknown")
    items_df = items_df[["item_id", "item_idx", "title", "genres"]]

    users_df = users[["user_id", "user_idx"]]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet(train_df, PROCESSED_DATA_DIR / "interactions_train.parquet")
    save_parquet(val_df, PROCESSED_DATA_DIR / "interactions_val.parquet")
    save_parquet(test_df, PROCESSED_DATA_DIR / "interactions_test.parquet")
    save_parquet(items_df, PROCESSED_DATA_DIR / "items.parquet")
    save_parquet(users_df, PROCESSED_DATA_DIR / "users.parquet")

    meta = {
        "dataset_dir": str(args.dataset_dir),
        "num_users": int(users_df["user_idx"].nunique()),
        "num_items": int(items_df["item_idx"].nunique()),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "min_rating": float(args.min_rating),
        "split": "leave-two-out-per-user-time-based",
    }
    (PROCESSED_DATA_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
