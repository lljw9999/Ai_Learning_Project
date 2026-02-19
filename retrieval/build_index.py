#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.io import load_dataset_meta
from recsys.paths import PROCESSED_DATA_DIR, RETRIEVAL_DIR, ensure_dirs
from recsys.retrieval import CandidateIndex, compute_item_embeddings, load_two_tower_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS (or numpy) index from trained retrieval model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    dataset_meta = load_dataset_meta(data_dir=PROCESSED_DATA_DIR)
    num_items = int(dataset_meta["num_items"])

    model, model_meta = load_two_tower_model(RETRIEVAL_DIR / "two_tower_model", device=args.device)
    item_vectors = compute_item_embeddings(model, num_items=num_items, batch_size=args.batch_size, device=args.device)

    index = CandidateIndex.from_item_vectors(item_vectors)
    index.save(RETRIEVAL_DIR / "candidate_index")

    summary = {
        "num_items": num_items,
        "embedding_dim": int(model_meta["embedding_dim"]),
        "index_backend": "faiss" if index.faiss_index is not None else "numpy",
    }
    (RETRIEVAL_DIR / "index_meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
