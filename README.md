# Two-Stage Recommender: Retrieval (Two-Tower + Item-CF Hybrid) + Ranking (LightGBM)

A production-style recommender pipeline with:

1. Time-based data split (leave-two-out per user)
2. Stage 1 retrieval (Two-Tower embedding model + Item-CF hybrid + FAISS/Numpy ANN index)
3. Stage 2 ranking (LightGBM ranker with user/item/interaction features)
4. Offline evaluation (retrieval + ranking metrics + ablation + latency)
5. FastAPI serving endpoint with request logging and basic monitoring
6. Daily report script with metric history and drift checks

## Architecture

`User history + item metadata -> Hybrid Retrieval top-K -> Feature generation -> LightGBM ranking (guarded) -> Top-N API response`

### Stage 1: Retrieval

- Model: `recsys/retrieval.py::TwoTowerModel`
- User tower: user embedding + mean history embedding
- Item tower: item embedding projection
- Training: leakage-safe prefix->next-item events
- Loss: pairwise ranking loss (`-log(sigmoid(pos-neg))`)
- Negatives: popularity-weighted sampling (`count^0.75`) with seen-item filtering
- Hybrid retriever: Two-Tower candidates + Item-CF candidates fused with Reciprocal Rank Fusion (RRF)
- Serving: ANN search over item embeddings via FAISS if installed, otherwise Numpy fallback
- Warm-user optimization: user embeddings are cached at service startup
- Output: `retrieve_candidates(user) -> [item_idx]`

### Stage 2: Ranking

- Model: `lightgbm.LGBMRanker` (LambdaRank objective)
- Train data: query-event samples from the train split (time-ordered histories), only when the next item is in retrieved candidates
- Validation: early stopping on the validation split
- Features:
  - user: activity, days since last interaction
  - item: popularity, year bucket, genre id
  - interaction: category match, days since similar genre, retrieval score/rank
- Final serving score: `rank_score + alpha * retrieval_score` (if ranker is enabled)
- Guardrail: if ranker does not beat retrieval-order by `min_ranker_improve` on validation, serving falls back to retrieval-order directly
- Output: robust ranked candidates with regression-safe fallback

## Repository Layout

- `data/` download + preprocess scripts
- `retrieval/` two-tower training, index build, retrieval CLI
- `ranking/` ranker training + inference CLI
- `eval/` offline metrics + daily drift report
- `service/` FastAPI app + demo client
- `recsys/` shared library code
- `artifacts/` model/data/eval outputs (generated)
- `logs/` service and daily metric logs (generated)

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Data + split (MovieLens)

```bash
python data/download_movielens.py --dataset ml-latest-small
python data/preprocess.py --dataset-dir data/raw/ml-latest-small
```

### 3) Train retrieval + index

```bash
python retrieval/train.py --epochs 10 --embedding-dim 96 --num-negatives 20 --learning-rate 8e-4 --batch-size 512 --max-history-len 80
python retrieval/build_index.py
```

### 4) Train ranking

```bash
python ranking/train.py --retrieval-mode hybrid --train-candidate-k 120 --eval-candidate-k 200 --num-boost-round 600 --learning-rate 0.04 --num-leaves 127 --early-stopping-rounds 80 --n-jobs 1 --min-ranker-improve 0.015
```

### 5) Offline evaluation

```bash
python eval/offline_eval.py --retrieval-mode hybrid --retrieval-k 200 --ranking-k 10
```

### 6) Serve API

```bash
uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Example request:

```bash
curl "http://127.0.0.1:8000/recommend?user_id=1&n=10&candidate_k=200"
```

Or demo client:

```bash
python service/client.py --user-id 1 --n 10
```

## Current Offline Results (MovieLens Small, Time Split)

From `artifacts/eval/offline_metrics.json`:

| Metric | Value |
|---|---:|
| Retrieval Recall@100 (test) | 0.3257 |
| Retrieval Recall@200 (test) | 0.4359 |
| Retrieval-order NDCG@10 (test) | 0.0356 |
| Final Ranking NDCG@10 (test) | 0.0356 |
| Final Ranking MAP@10 (test) | 0.0250 |
| Latency p95 (ms, offline simulation, warmup-skipped) | 50.31 |
| Ranker Guardrail (`use_ranker_score`) | false |

Current best configuration is hybrid retrieval with ranker guardrail fallback enabled; this avoids harmful reranking and keeps stronger retrieval ordering.

## Baseline Benchmarks

Run:

```bash
python eval/retrieval_baselines.py --top-k 200
```

From `artifacts/eval/retrieval_baselines.json`:

| Model | Recall@100 (test) | NDCG@10 (test) |
|---|---:|---:|
| Popularity baseline | 0.2072 | 0.0217 |
| Item-CF baseline | 0.3125 | 0.0338 |
| Hybrid retrieval (current) | 0.3257 | 0.0356 |

## Monitoring + Daily Reports

### Per-request logging (service)

- File: `logs/service_requests.jsonl`
- Fields: timestamp, user_id, candidate_count, latency_ms, error

### Runtime metrics endpoint

- `GET /metrics`
- Returns: request count, p50/p95 latency, error rate

### Daily evaluation + drift

```bash
python eval/daily_report.py --retrieval-mode hybrid --retrieval-k 200 --ranking-k 10
```

- Appends metrics to `logs/daily_metrics.csv`
- Includes PSI drift checks for:
  - `user_activity`
  - `item_popularity`
  - `retrieval_score`

## Time-Split Evaluation Design

The preprocessing split is per-user, timestamp ordered:

- train: all but each user's last two interactions
- val: second-to-last interaction
- test: last interaction

This avoids future leakage and keeps evaluation aligned with realistic recommendation timing.

## Failure Modes and Production Plan

### Known failure modes

- Cold-start users/items
- Popularity bias
- Concept/time drift
- Feedback loops from exposure bias

### Production upgrades

- A/B testing framework for online CTR/watch-time metrics
- Exploration policy (epsilon-greedy or contextual bandit)
- Guardrails: diversity, freshness, hard filters, business constraints
- Better retrieval (in-batch negatives, hard negatives, sequence towers)
- Better ranking labels (impressions/CTR logs instead of proxy labels)

## Notes

- If `faiss-cpu` is unavailable, retrieval uses a Numpy fallback index.
- The code is organized to make swapping datasets straightforward (`data/preprocess.py` is the integration point).

## Tests

```bash
python -m pytest
```

## Docker

Build and run API:

```bash
docker compose build
docker compose up
```

Service will be available at `http://127.0.0.1:8000`.
