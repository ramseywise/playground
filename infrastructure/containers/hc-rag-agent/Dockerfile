FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app

COPY hc-rag-agent/pyproject.toml hc-rag-agent/uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

COPY hc-rag-agent/orchestrator ./orchestrator
COPY hc-rag-agent/rag ./rag
COPY hc-rag-agent/core ./core
COPY hc-rag-agent/clients ./clients
COPY hc-rag-agent/guardrails ./guardrails
COPY hc-rag-agent/evals ./evals
COPY hc-rag-agent/ingest ./ingest
COPY hc-rag-agent/main.py .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
