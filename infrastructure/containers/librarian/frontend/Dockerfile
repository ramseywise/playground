# Frontend Streamlit playground
# Build: docker build -t librarian-frontend -f infrastructure/containers/frontend/Dockerfile .
# Run:   docker run -p 8501:8501 -e LIBRARIAN_API_URL=http://... librarian-frontend

# ---------------------------------------------------------------------------
# Stage 1 — install dependencies with uv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra frontend

COPY frontend/ ./frontend/

# ---------------------------------------------------------------------------
# Stage 2 — lean runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/frontend /app/frontend
COPY pyproject.toml /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8501/_stcore/health').raise_for_status()"

CMD ["streamlit", "run", "frontend/streamlit/librarian_chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
