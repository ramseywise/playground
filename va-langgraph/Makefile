.PHONY: run gateway test install

install:
	uv sync

gateway:
	uv run uvicorn gateway.main:app --host 0.0.0.0 --port 8001 --reload

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .
