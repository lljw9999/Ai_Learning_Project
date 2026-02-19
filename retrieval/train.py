#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.io import load_dataset_meta, load_split
from recsys.paths import PROCESSED_DATA_DIR, RETRIEVAL_DIR, ensure_dirs
from recsys.retrieval import (
    RetrievalTrainConfig,
    build_retrieval_train_events,
    build_user_history,
    save_pickle,
    save_two_tower_model,
    train_two_tower,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-tower retrieval model")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-negatives", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means auto from number of events")
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=1, help="Minimum prefix length for retrieval train events")
    parser.add_argument("--max-events-per-user", type=int, default=200, help="Max retrieval train events sampled per user")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cpu or cuda",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    train_df = load_split("train", data_dir=PROCESSED_DATA_DIR)
    meta = load_dataset_meta(data_dir=PROCESSED_DATA_DIR)

    user_history = build_user_history(train_df)
    train_events = build_retrieval_train_events(
        train_df,
        min_history=args.min_history,
        max_events_per_user=args.max_events_per_user,
    )
    config = RetrievalTrainConfig(
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        learning_rate=args.learning_rate,
        max_history_len=args.max_history_len,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    model, train_metrics = train_two_tower(
        train_events=train_events,
        user_seen_history=user_history,
        num_users=int(meta["num_users"]),
        num_items=int(meta["num_items"]),
        config=config,
        device=args.device,
    )

    model_prefix = RETRIEVAL_DIR / "two_tower_model"
    save_two_tower_model(
        model=model,
        path_prefix=model_prefix,
        num_users=int(meta["num_users"]),
        num_items=int(meta["num_items"]),
        embedding_dim=args.embedding_dim,
        max_history_len=args.max_history_len,
    )
    save_pickle(user_history, RETRIEVAL_DIR / "user_history_train.pkl")

    summary = {
        "device": args.device,
        "num_train_users": len(user_history),
        "num_train_events": len(train_events),
        **train_metrics,
    }
    (RETRIEVAL_DIR / "train_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
