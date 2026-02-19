#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.io import load_items, load_users
from recsys.paths import PROCESSED_DATA_DIR, RETRIEVAL_DIR
from recsys.retrieval import (
    CandidateIndex,
    compute_user_embedding,
    load_pickle,
    load_two_tower_model,
    recommend_from_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-K candidate items for a user")
    parser.add_argument("--user-id", type=int, required=True, help="Original user_id value")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _build_user_mapping(users_df) -> Dict[int, int]:
    return {int(row.user_id): int(row.user_idx) for row in users_df.itertuples(index=False)}


def _build_item_mapping(items_df) -> Dict[int, int]:
    return {int(row.item_idx): int(row.item_id) for row in items_df.itertuples(index=False)}


def main() -> None:
    args = parse_args()
    users = load_users(data_dir=PROCESSED_DATA_DIR)
    items = load_items(data_dir=PROCESSED_DATA_DIR)

    user_to_idx = _build_user_mapping(users)
    item_to_id = _build_item_mapping(items)

    if args.user_id not in user_to_idx:
        raise KeyError(f"Unknown user_id={args.user_id}. Cold-start users are handled by service fallback.")

    user_idx = user_to_idx[args.user_id]

    model, model_meta = load_two_tower_model(RETRIEVAL_DIR / "two_tower_model", device=args.device)
    index = CandidateIndex.load(RETRIEVAL_DIR / "candidate_index")
    user_history = load_pickle(RETRIEVAL_DIR / "user_history_train.pkl")

    history = user_history.get(user_idx, [])
    user_vec = compute_user_embedding(
        model,
        user_idx=user_idx,
        history=history,
        max_history_len=int(model_meta["max_history_len"]),
        device=args.device,
    )
    item_idxs, scores = recommend_from_index(user_vec, index=index, top_k=args.top_k, seen_items=history)

    payload: List[dict] = []
    for item_idx, score in zip(item_idxs, scores):
        payload.append(
            {
                "item_id": int(item_to_id.get(int(item_idx), -1)),
                "item_idx": int(item_idx),
                "score": float(score),
            }
        )

    print(json.dumps({"user_id": args.user_id, "candidates": payload}, indent=2))


if __name__ == "__main__":
    main()
