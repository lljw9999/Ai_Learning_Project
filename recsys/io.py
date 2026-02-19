from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from recsys.paths import PROCESSED_DATA_DIR


SPLIT_FILES = {
    "train": "interactions_train.parquet",
    "val": "interactions_val.parquet",
    "test": "interactions_test.parquet",
}


def load_split(split: str, data_dir: Path | None = None) -> pd.DataFrame:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(SPLIT_FILES)}")
    target_dir = data_dir or PROCESSED_DATA_DIR
    path = target_dir / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def load_all_splits(data_dir: Path | None = None) -> Dict[str, pd.DataFrame]:
    return {split: load_split(split, data_dir=data_dir) for split in SPLIT_FILES}


def load_items(data_dir: Path | None = None) -> pd.DataFrame:
    target_dir = data_dir or PROCESSED_DATA_DIR
    path = target_dir / "items.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing items file: {path}")
    return pd.read_parquet(path)


def load_users(data_dir: Path | None = None) -> pd.DataFrame:
    target_dir = data_dir or PROCESSED_DATA_DIR
    path = target_dir / "users.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing users file: {path}")
    return pd.read_parquet(path)


def load_dataset_meta(data_dir: Path | None = None) -> Dict[str, int | str]:
    target_dir = data_dir or PROCESSED_DATA_DIR
    path = target_dir / "dataset_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    return json.loads(path.read_text())


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def interaction_frame_to_history(
    interactions: pd.DataFrame,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
) -> Dict[int, list[int]]:
    grouped = interactions.groupby(user_col)[item_col].agg(list)
    return {int(user): [int(item) for item in items] for user, items in grouped.items()}


def split_ground_truth(
    split_df: pd.DataFrame,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
) -> Dict[int, set[int]]:
    grouped = split_df.groupby(user_col)[item_col].agg(set)
    return {int(user): {int(item) for item in items} for user, items in grouped.items()}


def train_and_eval_histories(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
) -> Tuple[Dict[int, list[int]], Dict[int, list[int]]]:
    train_history = interaction_frame_to_history(train_df, user_col=user_col, item_col=item_col)

    val_history = {user: list(items) for user, items in train_history.items()}
    for row in val_df[[user_col, item_col]].itertuples(index=False):
        user, item = int(row[0]), int(row[1])
        val_history.setdefault(user, []).append(item)

    return train_history, val_history
