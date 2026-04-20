.PHONY: run web gateway test eval install

install:
	uv sync

run:
	cd agents/va_assistant && uv run adk run .

web:
	cd agents/va_assistant && uv run adk web .

gateway:
	uv run uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest tests/ -v

eval:
	uv run adk eval agents/va_assistant/eval/evalset.json agents/va_assistant/eval/eval_config.json

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .
