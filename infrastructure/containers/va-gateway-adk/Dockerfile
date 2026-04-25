FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app

COPY va-google-adk/pyproject.toml va-google-adk/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY va-google-adk/agent.py va-google-adk/app.py ./
COPY va-google-adk/schema.py va-google-adk/memory.py va-google-adk/artefact_store.py va-google-adk/model_factory.py ./
COPY va-google-adk/sub_agents ./sub_agents
COPY va-google-adk/prompts ./prompts
COPY va-google-adk/gateway ./gateway

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
