# syntax=docker/dockerfile:1.4
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app

# Copy only dependency files first (separate layer for better caching)
COPY va-langgraph/pyproject.toml va-langgraph/uv.lock ./

# Use BuildKit cache mount to persist uv cache between builds
# This avoids re-downloading packages on every build
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=180 uv sync --frozen --no-dev --no-install-project

# Now copy application code (only this layer rebuilds on code changes)
COPY va-langgraph/schema.py va-langgraph/memory.py va-langgraph/artefact_store.py va-langgraph/model_factory.py ./
COPY va-langgraph/graph ./graph
COPY va-langgraph/gateway ./gateway
COPY va-langgraph/prompts ./prompts

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8001"]
