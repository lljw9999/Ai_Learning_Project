#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.paths import RAW_DATA_DIR, ensure_dirs

DATASET_URLS = {
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MovieLens dataset")
    parser.add_argument(
        "--dataset",
        default="ml-latest-small",
        choices=sorted(DATASET_URLS),
        help="Dataset name to download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing dataset directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    dataset_dir = RAW_DATA_DIR / args.dataset
    zip_path = RAW_DATA_DIR / f"{args.dataset}.zip"
    url = DATASET_URLS[args.dataset]

    if dataset_dir.exists() and not args.force:
        print(f"Dataset already exists: {dataset_dir}")
        return

    if dataset_dir.exists() and args.force:
        shutil.rmtree(dataset_dir)

    print(f"Downloading {args.dataset} from {url}")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DATA_DIR)

    extracted_dir = RAW_DATA_DIR / args.dataset
    if not extracted_dir.exists():
        candidates = [p for p in RAW_DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("ml-")]
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            newest.rename(dataset_dir)
    print(f"Dataset ready at: {dataset_dir}")


if __name__ == "__main__":
    main()
