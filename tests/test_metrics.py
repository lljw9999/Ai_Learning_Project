from recsys.metrics import bootstrap_lift_confidence, map_at_k, ndcg_at_k, per_user_ndcg_at_k, recall_at_k


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


def test_per_user_ndcg_and_bootstrap_lift() -> None:
    baseline = {1: [11, 10], 2: [21, 20], 3: [30, 31]}
    improved = {1: [10, 11], 2: [20, 21], 3: [30, 31]}
    truth = {1: {10}, 2: {20}, 3: {30}}

    baseline_scores = per_user_ndcg_at_k(baseline, truth, 10)
    improved_scores = per_user_ndcg_at_k(improved, truth, 10)
    assert baseline_scores[1] < improved_scores[1]
    assert baseline_scores[2] < improved_scores[2]

    stats = bootstrap_lift_confidence(
        baseline_scores=baseline_scores,
        candidate_scores=improved_scores,
        n_bootstrap=400,
        random_state=7,
    )
    assert int(stats["n_users"]) == 3
    assert float(stats["median_lift"]) > 0.0
    assert float(stats["p_lift_gt_zero"]) > 0.9
