# Interview Stories (Ranking/RecSys)

## Story 1: "Reranker Was Doing Nothing, Then Added Robust Lift"

### Situation
- Initial pipeline had strong retrieval, but reranking produced no measurable gain.
- `use_ranker_score` was often false due to guardrail fallback.

### Task
- Diagnose why reranking failed and produce repeatable, statistically credible lift.

### Actions
- Added ranker diagnostics (`ranking/debug_ranker.py`) for score spread, score/label correlation, and before/after top-N comparisons.
- Upgraded labels to graded relevance (`2/1/0`) using future positives within strict time-safe windows.
- Added stronger ranking features (retrieval rank/reciprocal rank, affinity, co-occurrence, user-item rating delta, recency signals).
- Increased ranking evaluation hardness with candidate-size sweeps (`K=100/200/500`).
- Replaced threshold-only gate with bootstrap confidence guardrail.

### Result
- Across 5 seeds: retrieval-order NDCG@10 `0.0332 ± 0.0032` -> ranking NDCG@10 `0.0435 ± 0.0013`.
- Mean absolute lift `+0.0103` (95% CI `[0.0073, 0.0134]`).
- Mean relative lift `+32.16%`; positive absolute lift in `5/5` seeds.


## Story 2: "How I Prevented Fake Gains"

### Situation
- Typical recsys projects overstate improvements due to leakage, weak baselines, or unstable single-run results.

### Task
- Build an evaluation stack that makes it hard to fool myself.

### Actions
- Enforced temporal leakage checks with hard-fail behavior in query construction (`recsys/ranker_dataset.py`).
- Added leakage unit tests (`tests/test_ranker_dataset.py`).
- Added fair reranker baselines:
  - logistic regression on same feature set
  - LightGBM with retrieval score only
- Added drift report (PSI + KL across train/val/test) and warning path when guardrail passes under high drift.
- Added end-to-end multi-seed sweep (`eval/seed_sweep.py`) with aggregate confidence summaries.

### Result
- Leakage violations reported as zero in training artifacts.
- Full model outperformed fair reranker baselines on test.
- Guardrail pass rate remained `1.00` across seeds with documented confidence intervals.


## 30-Second Version

- "I built a two-stage recommender where retrieval was already strong, but reranking initially failed. I added graded labels, richer features, candidate-K stress tests, and bootstrap gating. Then I hardened against self-deception with leakage hard-fails, fair baselines, drift checks, and 5-seed sweeps. Final result was consistent positive lift with confidence intervals, not just a lucky run."


## Follow-Up Questions You Should Be Ready For

- Why graded labels instead of binary next-item labels?
- Why bootstrap over users for guardrails?
- Why does relative lift variance look high while absolute lift remains stable?
- What would change if you had impression/exposure logs?
- How would you move from this offline stack to online A/B testing?
