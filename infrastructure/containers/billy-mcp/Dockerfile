FROM python:3.12-slim

RUN pip install uv --no-cache-dir

WORKDIR /app

COPY mcp_servers/billy/pyproject.toml mcp_servers/billy/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

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
