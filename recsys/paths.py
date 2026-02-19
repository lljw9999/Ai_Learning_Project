from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LOG_DIR = ROOT_DIR / "logs"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "data"
RETRIEVAL_DIR = ARTIFACTS_DIR / "retrieval"
RANKING_DIR = ARTIFACTS_DIR / "ranking"
EVAL_DIR = ARTIFACTS_DIR / "eval"


def ensure_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RETRIEVAL_DIR,
        RANKING_DIR,
        EVAL_DIR,
        LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
