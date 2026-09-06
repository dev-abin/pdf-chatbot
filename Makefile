.PHONY: help up down lint test api web evals

help:
	@echo "Targets: up, down, lint, test, api, web, evals"

up:
	docker compose up --build

down:
	docker compose down

lint:
	python -m ruff check apps/api/src apps/web/rag_ui tests

test:
	python -m pytest tests/unit

api:
	cd apps/api && python -m uvicorn documind.main:app --reload

web:
	cd apps/web && python -m streamlit run ui.py

evals:
	python scripts/run_evals.py --check
