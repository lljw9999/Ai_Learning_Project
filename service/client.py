#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo client for recommendation API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = requests.get(
        f"{args.base_url}/recommend",
        params={"user_id": args.user_id, "n": args.n, "candidate_k": args.candidate_k},
        timeout=15,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
