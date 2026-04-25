FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir uv

# Install va-google-adk deps (cached layer)
COPY va-google-adk/pyproject.toml va-google-adk/uv.lock ./va-google-adk/
RUN cd va-google-adk && uv sync --frozen --no-dev

# Copy agent source
COPY va-google-adk/ ./va-google-adk/

# Copy static UI
COPY infrastructure/containers/frontend/static/ ./static/

# Copy server and config
COPY infrastructure/containers/frontend/server.py .
COPY infrastructure/containers/frontend/config.json .

ENV AGENTS_ROOT_DIR=/app/va-google-adk
ENV PATH="/app/va-google-adk/.venv/bin:$PATH"
ENV FRONTEND_PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/health')"

CMD ["python", "server.py"]
