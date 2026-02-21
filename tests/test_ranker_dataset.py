import pandas as pd
import pytest

from recsys.ranker_dataset import QueryExample, build_train_queries, validate_query_leakage


def test_validate_query_leakage_clean_queries() -> None:
    train_df = pd.DataFrame(
        {
            "user_idx": [1, 1, 1, 1, 2, 2, 2, 2],
            "item_idx": [10, 11, 12, 13, 20, 21, 22, 23],
            "rating": [5, 4, 5, 4, 4, 5, 4, 5],
            "timestamp": [1, 2, 3, 4, 10, 11, 12, 13],
        }
    )
    val_df = pd.DataFrame(
        {
            "user_idx": [1, 2],
            "item_idx": [14, 24],
            "timestamp": [5, 14],
        }
    )
    test_df = pd.DataFrame(
        {
            "user_idx": [1, 2],
            "item_idx": [15, 25],
            "timestamp": [6, 15],
        }
    )

    queries = build_train_queries(
        interactions=train_df,
        min_history=2,
        max_queries_per_user=10,
        future_positive_window=2,
    )
    result = validate_query_leakage(
        queries,
        val_interactions=val_df,
        test_interactions=test_df,
        strict=True,
    )
    assert result.num_queries > 0
    assert result.total_violations == 0


def test_validate_query_leakage_detects_boundary_crossing() -> None:
    queries = [
        QueryExample(
            query_id=0,
            user_idx=1,
            history_items=[10, 11],
            history_timestamps=[1, 2],
            history_ratings=[5.0, 4.0],
            target_item=12,
            target_timestamp=6,
            future_items=[13],
            future_timestamps=[7],
        )
    ]
    val_df = pd.DataFrame({"user_idx": [1], "item_idx": [14], "timestamp": [5]})
    result = validate_query_leakage(
        queries,
        val_interactions=val_df,
        test_interactions=None,
        strict=False,
    )
    assert result.target_crosses_val_boundary == 1
    assert result.future_crosses_val_boundary == 1
    assert result.total_violations == 2

    with pytest.raises(ValueError):
        validate_query_leakage(
            queries,
            val_interactions=val_df,
            test_interactions=None,
            strict=True,
        )
