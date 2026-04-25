.PHONY: run gateway test install eval-unit eval-regression eval-capability

install:
	uv sync

gateway:
	uv run uvicorn gateway.main:app --host 0.0.0.0 --port 8001 --reload

test:
	uv run pytest tests/ -v

# ── Eval targets ─────────────────────────────────────────────────────────────

# Fast: grader unit tests (no LLM calls, no graph execution)
eval-unit:
	uv run pytest tests/unit/ -v

# Medium: safety regression floors (guardrail_node is deterministic, no LLM)
eval-regression:
	uv run pytest tests/evalsuite/regression/ -v

# Slow: routing accuracy + message quality (real LLM API calls)
# Gate: CONFIRM_EXPENSIVE_OPS=1 make eval-capability
eval-capability:
	@if [ "$$CONFIRM_EXPENSIVE_OPS" != "1" ]; then \
		echo "[gate] Capability tests make real LLM API calls."; \
		echo "       Run with: CONFIRM_EXPENSIVE_OPS=1 make eval-capability"; \
		exit 1; \
	fi
	uv run pytest tests/evalsuite/capability/ -v -s

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .
