FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app

COPY va-langgraph/pyproject.toml va-langgraph/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY va-langgraph/graph ./graph
COPY va-langgraph/gateway ./gateway
COPY va-langgraph/shared ./shared

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8001"]
