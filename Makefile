.PHONY: setup typecheck eval-unit eval-regression eval-capability

setup:
	bash setup.sh

typecheck:
	uv run mypy src/agents/librarian src/core

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
