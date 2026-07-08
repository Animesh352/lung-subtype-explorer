PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF   := .venv/bin/ruff

.PHONY: up down test lint migrate coverage download-data etl backfill train de load-de annotate pipeline

up:
	docker compose up --build -d

down:
	docker compose down

test:
	$(PYTEST) tests/ -v

lint:
	$(RUFF) check . && $(RUFF) format --check .

migrate:
	docker compose exec api alembic upgrade head

download-data:
	$(PYTHON) pipeline/download_data.py

etl:
	$(PYTHON) pipeline/etl.py

backfill:
	$(PYTHON) pipeline/etl.py backfill

train:
	$(PYTHON) pipeline/train.py

de:
	docker compose --profile analysis run --rm r-analysis
	$(PYTHON) pipeline/load_de_results.py

load-de:
	$(PYTHON) pipeline/load_de_results.py

annotate:
	$(PYTHON) pipeline/annotate.py

coverage:
	$(PYTEST) tests/ --cov=app --cov=pipeline --cov-report=term-missing --cov-report=html -v

pipeline:
	$(PYTHON) pipeline/download_data.py
	$(PYTHON) pipeline/etl.py
	docker compose --profile analysis run --rm r-analysis
	$(PYTHON) pipeline/load_de_results.py
	$(PYTHON) pipeline/train.py
	$(PYTHON) pipeline/annotate.py
