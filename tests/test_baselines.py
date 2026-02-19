from recsys.baselines import item_cf_scored_candidates_for_history


def test_item_cf_candidates_exclude_seen_and_fill() -> None:
    history = [1, 2, 3]
    neighbors = {
        1: [(4, 0.9), (5, 0.8)],
        2: [(4, 0.7), (6, 0.4)],
        3: [(1, 0.5), (7, 0.3)],
    }
    fallback = [8, 9, 10, 11]

    ranked = item_cf_scored_candidates_for_history(
        history=history,
        item_cf_neighbors=neighbors,
        fallback_ranked_items=fallback,
        top_k=5,
    )

    items = [item for item, _ in ranked]
    assert len(items) == 5
    assert not any(item in set(history) for item in items)
    assert 4 in items
