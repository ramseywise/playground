.PHONY: help install install-eval dev test test-integration lint fmt ingest-clara eval

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install base dependencies
	uv sync

install-eval: ## Install with eval harness (ragas, deepeval)
	uv sync --extra eval

dev: ## Start FastAPI dev server (hot reload)
	uv run uvicorn main:app --reload --port 8002

test: ## Run all unit tests
	uv run pytest tests/unit -v

test-integration: ## Run integration tests (requires .env with real credentials)
	uv run pytest tests/ -m integration -v

lint: ## Lint with ruff
	uv run ruff check orchestrator/ rag/ core/ clients/ guardrails/ evals/ ingest/

fmt: ## Format with ruff
	uv run ruff format orchestrator/ rag/ core/ clients/ guardrails/ evals/ ingest/

ingest-clara: ## Re-index clara_raw → clara.duckdb (only needed after doc updates)
	uv run ingest-clara \
	  --source ../data/raw/clara_raw \
	  --vectordb ../data/stores/vectordb/clara.duckdb \
	  --metadb ../data/stores/metadb/clara_meta.db

eval: ## Run retrieval eval suite against clara corpus
	uv run python evals/runner.py run
