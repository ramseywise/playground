FROM python:3.12-slim

RUN pip install uv --no-cache-dir

WORKDIR /app

COPY va-langgraph/pyproject.toml va-langgraph/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY va-langgraph/graph ./graph
COPY va-langgraph/gateway ./gateway
COPY va-langgraph/shared ./shared

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8001"]
