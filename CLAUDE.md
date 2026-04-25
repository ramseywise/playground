# Billy VA

Two implementations of the same virtual assistant for Billy: `va-google-adk` (Google ADK multi-agent) and `va-langgraph` (LangGraph StateGraph). Shared Billy MCP backend in `mcp_servers/billy`. Docker Compose + Terraform in `infrastructure/`.

## Layout

```
va-google-adk/        Google ADK implementation (self-contained, own pyproject + uv.lock)
va-langgraph/         LangGraph implementation   (self-contained, own pyproject + uv.lock)
mcp_servers/billy/    Billy MCP + REST server    (self-contained, own pyproject + uv.lock)
infrastructure/
  containers/         Dockerfiles + docker-compose.va.yml
  terraform/          AWS ECS/Fargate + RDS + ALB
.claude/
  hooks/              Code quality enforcement (ruff, secrets scan, etc.)
  settings.json       Permissions + hook config
  skills/             Project-specific skills: adk-python, a2ui
```

## Commands

```bash
make va-up        # full stack (docker compose)
make va-up-ui     # frontend + billy-mcp only
make va-down
make va-smoke     # health-check all services

cd va-google-adk && uv run pytest tests/ -v
cd va-langgraph  && uv run pytest tests/ -v
cd mcp_servers/billy && uv run pytest tests/ -v
```

## Stack

- **va-google-adk** — Google ADK, Gemini 2.5 Flash, FastAPI gateway
- **va-langgraph** — LangGraph, Gemini 2.5 Flash, Postgres checkpointing, FastAPI gateway
- **mcp_servers/billy** — FastMCP + FastAPI, SQLite (billy.db), Bedrock KB for support knowledge
- Python 3.12, uv, pydantic v2, structlog

## Style

- `from __future__ import annotations` in all modules
- Type annotations on all signatures; f-strings only
- `httpx` not `requests`; async-first I/O
- Pydantic models at API boundaries
- No magic numbers — named constants or env vars
- Functions >40 lines → split; nesting >3 levels → early returns

## Hook-enforced standards

**PostToolUse (Write|Edit):** ruff format + check, no `print()` in src, no bare `except`, no stdlib `logging`, no hardcoded model strings, secrets scan, file size warning >400 lines.

**PreToolUse (Bash):** `git commit` blocked if tests fail or `uv.lock` out of sync, `pip install` blocked (use `uv add`), destructive commands blocked.

## Discipline

- Implement one plan step at a time — no skipping ahead
- Before multi-file changes: present numbered plan → wait for approval → execute
- Confirm before touching `pyproject.toml`, CI config, or Terraform
- Never commit `.env`, model weights, or large data files
- Ask before running costly API calls or anything >30s

## Commit convention

```
<type>(<ticket>): <description>
```

Types: `feat`, `fix`, `chore`, `refactor`, `docs`, `test`, `perf`, `ci`
Extract ticket from branch: `feature/LIN-123-desc` → `LIN-123`

## Issue tracking

Linear ↔ GitHub. Branch, commit, and PR names must include `LIN-{id}`.
Stack: Code → GitHub | Tasks → Linear | Knowledge → Notion
