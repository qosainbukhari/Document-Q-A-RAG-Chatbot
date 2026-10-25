# Shortcuts. Run `make help` to list them.
PY := venv/bin/python

.PHONY: help install api ui test eval eval-all docker

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

install:  ## Create venv and install CPU-only dependencies
	python3.11 -m venv venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt
	@test -f .env || cp .env.example .env

api:  ## Run the FastAPI backend on :8000
	$(PY) -m uvicorn app.main:app --port 8000

ui:  ## Run the Streamlit frontend on :8501
	$(PY) -m streamlit run streamlit_app.py

test:  ## Run the test suite
	$(PY) -m pytest -q

eval:  ## Evaluate one config, e.g. make eval CONFIG=naive
	$(PY) -m evaluation.run_eval --config $(or $(CONFIG),advanced)

eval-all:  ## Evaluate naive, hybrid, rerank and advanced
	for c in naive hybrid rerank advanced; do $(PY) -m evaluation.run_eval --config $$c --sleep 10; done

docker:  ## Run backend + frontend with Docker Compose
	docker compose up --build
