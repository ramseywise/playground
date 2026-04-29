# syntax=docker/dockerfile:1.4
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app

# Copy only dependency files first (separate layer for better caching)
COPY mcp_servers/billy/pyproject.toml mcp_servers/billy/uv.lock ./

# Use BuildKit cache mount to persist uv cache between builds
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=180 uv sync --frozen --no-dev --no-install-project

# Now copy application code (only this layer rebuilds on code changes)
COPY mcp_servers/billy/app ./app
COPY mcp_servers/billy/reset_db.py ./reset_db.py

# DB is created at runtime via reset_db.py; persist with a named volume
ENV BILLY_DB=/data/billy.db
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8765

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8765
EXPOSE 8766

# entrypoint seeds DB on first run, then starts both MCP (SSE) and REST servers
COPY infrastructure/containers/billy-mcp/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
