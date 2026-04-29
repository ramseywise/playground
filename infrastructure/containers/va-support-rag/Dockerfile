FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app

COPY va-support-rag/pyproject.toml va-support-rag/uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

COPY va-support-rag/orchestrator ./orchestrator
COPY va-support-rag/rag ./rag
COPY va-support-rag/core ./core
COPY va-support-rag/clients ./clients
COPY va-support-rag/guardrails ./guardrails
COPY va-support-rag/evals ./evals
COPY va-support-rag/ingest ./ingest
COPY va-support-rag/main.py .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
