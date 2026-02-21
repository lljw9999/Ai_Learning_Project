#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recsys.paths import RANKING_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ranker score usage on a saved ranking frame")
    parser.add_argument(
        "--frame-path",
        type=Path,
        default=RANKING_DIR / "ranker_val_frame.parquet",
        help="Path to ranking frame parquet (typically ranker_val_frame.parquet)",
    )
    parser.add_argument("--sample-users", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-path", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def _pick_col(frame: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError(f"None of expected columns found: {candidates}")


def _topn(frame_u: pd.DataFrame, score_col: str, top_n: int) -> list[dict]:
    rows = frame_u.sort_values(score_col, ascending=False).head(top_n)
    out = []
    for row in rows.itertuples(index=False):
        item = {"item_idx": int(row.item_idx), "score": float(getattr(row, score_col))}
        if hasattr(row, "label"):
            item["label"] = int(getattr(row, "label"))
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    if not args.frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {args.frame_path}")

    frame = pd.read_parquet(args.frame_path)
    if frame.empty:
        raise ValueError("Frame is empty")

    retrieval_col = _pick_col(frame, ["retrieval_score"])
    ranker_col = _pick_col(frame, ["rank_score", "final_score_model", "final_score"])
    served_col = _pick_col(frame, ["final_score", "final_score_model", "rank_score"])

    score_vals = frame[ranker_col].to_numpy(dtype=np.float64)
    label_corr = 0.0
    if "label" in frame.columns:
        label_vals = frame["label"].to_numpy(dtype=np.float64)
        if score_vals.size > 0 and float(np.std(score_vals)) > 0 and float(np.std(label_vals)) > 0:
            label_corr = float(np.corrcoef(score_vals, label_vals)[0, 1])

    if "label" in frame.columns:
        pos_users = frame.groupby("user_idx")["label"].sum()
        users = sorted(int(u) for u in pos_users[pos_users > 0].index.tolist())
    else:
        users = sorted(int(u) for u in frame["user_idx"].unique().tolist())
    users = users[: max(int(args.sample_users), 0)]

    samples: list[dict] = []
    for user in users:
        frame_u = frame.loc[frame["user_idx"] == int(user)].copy()
        samples.append(
            {
                "user_idx": int(user),
                "before_topn": _topn(frame_u, retrieval_col, args.top_n),
                "ranker_topn": _topn(frame_u, ranker_col, args.top_n),
                "served_topn": _topn(frame_u, served_col, args.top_n),
            }
        )

    report = {
        "frame_path": str(args.frame_path),
        "num_rows": int(len(frame)),
        "num_users": int(frame["user_idx"].nunique()),
        "score_distribution": {
            "col": ranker_col,
            "min": float(np.min(score_vals)) if score_vals.size > 0 else 0.0,
            "mean": float(np.mean(score_vals)) if score_vals.size > 0 else 0.0,
            "max": float(np.max(score_vals)) if score_vals.size > 0 else 0.0,
            "std": float(np.std(score_vals)) if score_vals.size > 0 else 0.0,
        },
        "score_label_corr": float(label_corr),
        "samples": samples,
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text)


if __name__ == "__main__":
    main()
