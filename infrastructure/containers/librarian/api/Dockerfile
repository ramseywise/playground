# Multi-stage build for the Librarian RAG Agent API
# Build: docker build -t librarian-api -f infrastructure/containers/api/Dockerfile .
# Run:   docker run -p 8000:8000 --env-file .env librarian-api

# ---------------------------------------------------------------------------
# Stage 1 — install dependencies with uv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra librarian --extra api --extra lambda

COPY src/ ./src/

# Pre-download the embedding model into the image so cold starts don't
# incur a 30–60s HuggingFace download.  The model is cached under
# /root/.cache/torch/sentence_transformers/ and reused by the running app.
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

# ---------------------------------------------------------------------------
# Stage 2 — lean runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual env and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY pyproject.toml /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "agents.librarian.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
