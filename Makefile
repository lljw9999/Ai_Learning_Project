PYTHON ?= python

.PHONY: setup data preprocess retrieval ranking eval baseline daily service test docker-build docker-up

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
	$(PYTHON) ranking/train.py --retrieval-mode hybrid --train-candidate-k 120 --eval-candidate-k 200 --num-boost-round 600 --learning-rate 0.04 --num-leaves 127 --early-stopping-rounds 80 --n-jobs 1 --min-ranker-improve 0.015

eval:
	$(PYTHON) eval/offline_eval.py --retrieval-mode hybrid

baseline:
	$(PYTHON) eval/retrieval_baselines.py --top-k 200

daily:
	$(PYTHON) eval/daily_report.py --retrieval-mode hybrid

service:
	uvicorn service.app:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest

docker-build:
	docker compose build

docker-up:
	docker compose up
