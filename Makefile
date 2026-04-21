.PHONY: setup lint typecheck test test-librarian test-core test-eval eval-unit eval-regression eval-capability eval-compare eval-experiment \
        va-up va-up-ui va-down va-smoke

setup:
	bash setup.sh

# ── Lint & typecheck ─────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/agents/librarian src/core

# ── Test targets ─────────────────────────────────────────────────────────────

# All tests (excluding known-broken google_adk imports)
test:
	uv run pytest tests/librarian/ tests/core/ -v --ignore=tests/orchestration/google_adk/

# Librarian unit tests only — fast, no external deps.
test-librarian:
	uv run pytest tests/librarian/unit/ -v

# Core module tests.
test-core:
	uv run pytest tests/core/ -v

# Eval suite (unit-level — no API calls).
test-eval:
	uv run pytest tests/eval/ -v

# ── Eval tiers ────────────────────────────────────────────────────────────────

# Fast unit tests — no external deps, always safe to run.
eval-unit:
	uv run pytest tests/librarian/unit/ -v

# Retrieval metric regression — hit_rate@5 and MRR floor checks.
# Uses InMemoryRetriever + MockEmbedder; no API calls.
eval-regression:
	uv run pytest tests/librarian/evalsuite/regression/ -v

# Full capability suite — exercises graph assembly including cross-encoder
# and multilingual embedder (triggers 500 MB model download on cold cache).
# Gate: CONFIRM_EXPENSIVE_OPS=1 make eval-capability
eval-capability:
	@if [ "$$CONFIRM_EXPENSIVE_OPS" != "1" ]; then \
		echo "[gate] Capability tests are slow / download models on cold cache."; \
		echo "       Run with: CONFIRM_EXPENSIVE_OPS=1 make eval-capability"; \
		exit 1; \
	fi
	uv run pytest tests/librarian/evalsuite/capability/ -v

# Three-way variant comparison — prints hit_rate@k, MRR, failure clusters.
# Uses InMemoryRetriever + MockEmbedder; no API calls or model downloads.
eval-compare:
	uv run pytest tests/librarian/evalsuite/regression/test_variant_comparison.py -v -s

# LangFuse experiment runner — uploads golden dataset + runs all variants.
# Logs traces to LangFuse when LANGFUSE_ENABLED=true.
# Set EVAL_DATASET_PATH for the external golden JSONL; falls back to test samples.
eval-experiment:
	uv run python -m eval.experiment run

# ── VA agents (docker compose) ────────────────────────────────────────────────

COMPOSE := docker compose -f infrastructure/containers/docker-compose.va.yml

# Full stack: frontend + billy-mcp + both gateways + postgres
va-up:
	$(COMPOSE) up --build

# UI only: frontend + billy-mcp (fastest path to test the agent)
va-up-ui:
	$(COMPOSE) up --build frontend billy-mcp

va-down:
	$(COMPOSE) down

va-smoke:
	@echo "--- frontend ---"
	@curl -sf http://localhost:3000/health || echo "FAIL :3000"
	@echo "--- billy-mcp ---"
	@curl -sf http://localhost:8766/docs > /dev/null && echo "OK :8766" || echo "FAIL :8766"
	@echo "--- va-gateway-adk ---"
	@curl -sf http://localhost:8000/health || echo "FAIL :8000"
	@echo "--- va-gateway-lg ---"
	@curl -sf http://localhost:8001/health || echo "FAIL :8001"
