.PHONY: help install install-eval dev test test-unit test-integration lint fmt ingest-clara setup-data

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install base dependencies
	uv sync

install-eval: ## Install with eval harness (ragas, deepeval)
	uv sync --extra eval

dev: ## Start FastAPI dev server (hot reload)
	uv run uvicorn src.main:app --reload --port 8002

test: ## Run all unit tests
	uv run pytest tests/unit -v

test-integration: ## Run integration tests (requires .env with real credentials)
	uv run pytest tests/ -m integration -v

lint: ## Lint with ruff
	uv run ruff check src/ evals/ ingest/

fmt: ## Format with ruff
	uv run ruff format src/ evals/ ingest/

setup-data: ## Copy clara corpus from help-support-rag-agent into playground/data/ (run once)
	@echo "Copying DuckDB index..."
	cp ../../help-support-rag-agent/data/vectorstore/rag_index.duckdb \
	   ../data/stores/vectordb/clara.duckdb
	@echo "Copying raw JSONL sources..."
	cp ../../help-support-rag-agent/data/raptor_scraper/output/v2/*.jsonl \
	   ../data/raw/clara_raw/
	@echo "Copying eval datasets..."
	cp ../../help-support-rag-agent/data/eval_data/*.jsonl \
	   ../data/eval_data/ 2>/dev/null || true
	@echo "Done. Run 'make ingest-clara' only if re-indexing from scratch."

ingest-clara: ## Re-index clara_raw → clara.duckdb (only needed after doc updates)
	uv run ingest-clara \
	  --source ../data/raw/clara_raw \
	  --vectordb ../data/stores/vectordb/clara.duckdb \
	  --metadb ../data/stores/metadb/clara_meta.db

eval: ## Run capability eval suite against clara corpus
	uv run python evals/runner.py --suite capability
