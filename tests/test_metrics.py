from recsys.metrics import map_at_k, ndcg_at_k, recall_at_k


def test_metrics_perfect_ranking() -> None:
    preds = {1: [10, 11, 12], 2: [20, 21, 22]}
    truth = {1: {10}, 2: {20}}

    assert recall_at_k(preds, truth, 1) == 1.0
    assert ndcg_at_k(preds, truth, 10) == 1.0
    assert map_at_k(preds, truth, 10) == 1.0


def test_metrics_worse_ranking() -> None:
    preds = {1: [11, 12, 10], 2: [21, 20, 22]}
    truth = {1: {10}, 2: {20}}

    assert recall_at_k(preds, truth, 1) == 0.0
    assert 0.0 < ndcg_at_k(preds, truth, 10) < 1.0
    assert 0.0 < map_at_k(preds, truth, 10) < 1.0
