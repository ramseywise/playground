# Eval dashboard — Streamlit + Plotly + eval data layer
# Build: docker build -t eval-dashboard -f infrastructure/containers/eval-dashboard/Dockerfile .
# Run:   docker run -p 8502:8502 eval-dashboard

# ---------------------------------------------------------------------------
# Stage 1 — install dependencies with uv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
# Needs frontend (streamlit, plotly) + librarian (structlog, pydantic-settings,
# langfuse) since the dashboard imports eval.dashboard_data → librarian.config
RUN uv sync --frozen --no-dev --extra frontend --extra librarian

COPY src/ ./src/
COPY frontend/ ./frontend/

# ---------------------------------------------------------------------------
# Stage 2 — lean runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/frontend /app/frontend
COPY pyproject.toml /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8502/_stcore/health').raise_for_status()"

CMD ["streamlit", "run", "frontend/streamlit/eval_dashboard.py", "--server.port=8502", "--server.address=0.0.0.0"]
