from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    lgb = None
    _LIGHTGBM_IMPORT_ERROR = exc
else:
    _LIGHTGBM_IMPORT_ERROR = None


@dataclass
class RankerArtifacts:
    model_path: Path
    metadata_path: Path


def _ensure_lightgbm() -> None:
    if lgb is None:
        raise ImportError(
            "lightgbm is not installed. Install dependencies from requirements.txt"
        ) from _LIGHTGBM_IMPORT_ERROR


def train_ranker(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_col: str = "user_idx",
    label_col: str = "label",
    valid_df: pd.DataFrame | None = None,
    valid_group_col: str | None = None,
    num_boost_round: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    early_stopping_rounds: int = 50,
    n_jobs: int = 1,
    random_state: int = 42,
):
    _ensure_lightgbm()

    if train_df.empty:
        raise ValueError("Ranking training frame is empty")

    train_sorted = train_df.sort_values([group_col, "retrieval_rank"], ascending=[True, True]).reset_index(drop=True)
    X = train_sorted[list(feature_cols)]
    y = train_sorted[label_col]
    group_sizes = train_sorted.groupby(group_col).size().tolist()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=num_boost_round,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    fit_kwargs = {"group": group_sizes}
    if valid_df is not None and not valid_df.empty:
        eval_group_col = valid_group_col or group_col
        valid_sorted = valid_df.sort_values([eval_group_col, "retrieval_rank"], ascending=[True, True]).reset_index(drop=True)
        X_val = valid_sorted[list(feature_cols)]
        y_val = valid_sorted[label_col]
        group_val = valid_sorted.groupby(eval_group_col).size().tolist()

        fit_kwargs.update(
            {
                "eval_set": [(X_val, y_val)],
                "eval_group": [group_val],
                "eval_at": [10],
                "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
            }
        )

    model.fit(X, y, **fit_kwargs)
    return model


def predict_scores(model, frame: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=np.float32)
    return model.predict(frame[list(feature_cols)], num_threads=1)


def ranking_dict_from_frame(frame: pd.DataFrame, score_col: str = "rank_score") -> Dict[int, List[int]]:
    ranked: Dict[int, List[int]] = {}
    if frame.empty:
        return ranked

    sorted_df = frame.sort_values(["user_idx", score_col], ascending=[True, False])
    for user, group in sorted_df.groupby("user_idx"):
        ranked[int(user)] = [int(item) for item in group["item_idx"].tolist()]
    return ranked


def save_ranker(
    model,
    feature_cols: Sequence[str],
    path_prefix: Path,
    score_blend_alpha: float = 0.0,
    use_ranker_score: bool = True,
) -> RankerArtifacts:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = path_prefix.with_suffix(".joblib")
    meta_path = path_prefix.with_suffix(".json")

    joblib.dump(model, model_path)
    meta = {
        "feature_cols": list(feature_cols),
        "score_blend_alpha": float(score_blend_alpha),
        "use_ranker_score": bool(use_ranker_score),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return RankerArtifacts(model_path=model_path, metadata_path=meta_path)


def load_ranker(path_prefix: Path):
    model = joblib.load(path_prefix.with_suffix(".joblib"))
    meta = json.loads(path_prefix.with_suffix(".json").read_text())
    feature_cols = meta["feature_cols"]
    score_blend_alpha = float(meta.get("score_blend_alpha", 0.0))
    use_ranker_score = bool(meta.get("use_ranker_score", True))
    return model, feature_cols, score_blend_alpha, use_ranker_score
