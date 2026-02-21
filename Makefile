PYTHON ?= python

.PHONY: setup data preprocess retrieval ranking rank-debug eval seed-sweep baseline daily service test docker-build docker-up smoke github-harden

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) data/download_movielens.py --dataset ml-latest-small
	$(PYTHON) data/preprocess.py --dataset-dir data/raw/ml-latest-small

preprocess:
	$(PYTHON) data/preprocess.py --dataset-dir data/raw/ml-latest-small

retrieval:
	$(PYTHON) retrieval/train.py --epochs 10 --embedding-dim 96 --num-negatives 20 --learning-rate 8e-4 --batch-size 512 --max-history-len 80
	$(PYTHON) retrieval/build_index.py

ranking:
	$(PYTHON) ranking/train.py --retrieval-mode hybrid --train-candidate-k 120 --eval-candidate-k 500 --num-boost-round 600 --learning-rate 0.04 --num-leaves 127 --early-stopping-rounds 80 --n-jobs 1 --seed 42 --future-positive-window 10 --min-ranker-improve 0.005 --guardrail-confidence 0.95

rank-debug:
	$(PYTHON) ranking/debug_ranker.py --frame-path artifacts/ranking/ranker_val_frame.parquet --sample-users 5 --top-n 10 --output-path artifacts/ranking/ranker_debug_manual.json

eval:
	$(PYTHON) eval/offline_eval.py --retrieval-mode hybrid --candidate-k-grid 100,200,500

seed-sweep:
	$(PYTHON) eval/seed_sweep.py --seeds 11,22,33,44,55 --retrieval-mode hybrid --retrieval-k 200 --ranking-k 10 --candidate-k-grid 100,200,500

baseline:
	$(PYTHON) eval/retrieval_baselines.py --top-k 200

daily:
	$(PYTHON) eval/daily_report.py --retrieval-mode hybrid

service:
	uvicorn service.app:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) data/download_movielens.py --dataset ml-latest-small
	$(PYTHON) data/preprocess.py --dataset-dir data/raw/ml-latest-small
	$(PYTHON) retrieval/train.py --epochs 2 --embedding-dim 32 --num-negatives 8 --batch-size 512 --max-history-len 40 --learning-rate 1e-3
	$(PYTHON) retrieval/build_index.py
	$(PYTHON) ranking/train.py --retrieval-mode hybrid --train-candidate-k 80 --eval-candidate-k 120 --max-queries-per-user 8 --num-boost-round 120 --learning-rate 0.05 --num-leaves 63 --early-stopping-rounds 20 --n-jobs 1 --seed 42 --future-positive-window 8 --min-ranker-improve 0.005 --guardrail-confidence 0.9 --guardrail-bootstrap-samples 200
	$(PYTHON) eval/offline_eval.py --retrieval-mode hybrid --retrieval-k 120 --ranking-k 10 --candidate-k-grid 80,100,120 --latency-warmup 30

docker-build:
	docker compose build

docker-up:
	docker compose up

github-harden:
	./scripts/apply_github_hardening.sh
