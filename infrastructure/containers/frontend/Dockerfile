# syntax=docker/dockerfile:1.4
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv
WORKDIR /app

ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy only dependency files first (separate layer for better caching)
COPY va-google-adk/pyproject.toml va-google-adk/uv.lock ./va-google-adk/

# Use BuildKit cache mount to persist uv cache between builds
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=180 bash -c "cd va-google-adk && uv sync --frozen --no-dev --no-install-project"

# Copy agent source (only this layer rebuilds on code changes)
COPY va-google-adk/ ./va-google-adk/

# Copy static UI
COPY infrastructure/containers/frontend/static/ ./static/

# Copy server and config
COPY infrastructure/containers/frontend/server.py .
COPY infrastructure/containers/frontend/config.json .

ENV AGENTS_ROOT_DIR=/app/va-google-adk
ENV PATH="/app/.venv/bin:$PATH"
ENV FRONTEND_PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/health')"

CMD ["/app/.venv/bin/python", "server.py"]
