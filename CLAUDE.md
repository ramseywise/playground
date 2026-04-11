# Agents — AI Research & Visualization Toolkit

Personal AI agent toolkit for processing research PDFs into an Obsidian knowledge base and generating presentations.

Stable user preferences and project facts live in `.claude/memory/`; keep this file focused on repo operating rules and workflow.

## Project Layout

```
src/agents/           # Python package (target structure — see plan)
  utils/              # Config, Claude client, shared utilities
  researcher/         # PDF → chunked notes → Obsidian vault
  presenter/          # Interactive presentation builder → PPTX
  cartographer/       # Session JSONL parser → workflow insights + cron
tests/                # Mirrors src/agents/ structure
obsidian/             # Curated knowledge corpus (vault output)
```

> **Current state**: Package restructure in progress. See `.claude/docs/plans/agent_platform_unification.md` for the active plan.

## Stack

- Python 3.12+, uv, ruff, pydantic v2, structlog
- Claude API (anthropic SDK) — all LLM calls
- pdftotext/pdfinfo for PDF extraction
- python-pptx for slide rendering
- Pollinations.ai for image generation (no API key)

## Style

- `from __future__ import annotations` in all modules
- Type annotations on all function signatures; docstrings on non-obvious functions
- f-strings over `.format()` or `%`
- `httpx` not `requests`; async-first for I/O; context managers for connections
- Pydantic models at API boundaries, not raw dicts
- Polars: lazy frames for large data, eager for small; DuckDB for local analytics
- Parquet for intermediate data; never CSV for processed outputs
- No magic numbers — named constants or config values
- Functions >40 lines → split; nesting >3 levels → early returns
- Every new function gets at least one test
- Seed all randomness (`random.seed()`, `np.random.seed()`, etc.)

## Tooling

- `uv run pytest tests/` — full test suite
- `uv run pytest tests/researcher/` — research agent tests only
- `uv run pytest tests/presenter/` — presenter tests only
- `.env` never committed; `.env.example` is the template

## Hook-enforced standards

All standards below are enforced via `settings.json` hooks where applicable — do not duplicate what hooks already enforce.

**PostToolUse (Write|Edit):**
- ruff format + check on every `.py` write
- `[no-print]` no `print()` in `src/` — use structlog
- `[bare-except]` no bare `except:` — catch specific exceptions
- `[use-structlog]` no stdlib `logging` — use structlog
- `[use-polars]` no pandas — use polars
- `[mutable-default]` no `def f(x=[])` — use `None` sentinel
- `[sdk-factory]` no bare `anthropic.Anthropic()` or `genai.Client()` — use factory
- `[sdk-model]` no hardcoded model strings — use settings
- Token usage logging advisory on files with API calls
- Test coverage warning on untested public functions
- Import cycle detection on `src/` edits
- Pyright type check (advisory)
- File size warning at >400 lines
- Phase artifact writes trigger a compact reminder on the next prompt

**PostToolUse (Bash):**
- Failed commands logged to `.claude/friction-log.jsonl`
- Desktop notification on long test runs (>30s)

**UserPromptSubmit:**
- Auto-injects compact reminder when a phase/step just completed

**PreToolUse (Write|Edit):**
- Secrets scan — blocks API keys, tokens in source files

**PreToolUse (Bash):**
- `git commit` blocked if tests fail
- `git commit` blocked if `uv.lock` out of sync
- Cost guard — warns on agent commands without `--dry-run`
- `pip install` blocked — use `uv add`
- Destructive commands (`rm -rf /`, `DROP TABLE`) blocked

## Discipline

- Implement one plan step at a time — do not skip ahead or refactor outside scope
- Before multi-file changes: present numbered plan → wait for approval → execute
- Never delete files, agents, or config without confirmation; re-present plan if scope expands
- Confirm before touching `pyproject.toml`, CI config, or infra files
- Never commit model weights, large data files, or notebooks with output cells
- Ask before running: costly API calls, large file/model loads, or anything >30s
- Prefer `--dry-run`, targeted `pytest -k`, and subsampled data for validation

### Review gate

A PreToolUse hook gates source file edits (outside `.claude/` and `tests/`). When blocked: show before/after code block → wait for user confirmation → `touch .claude/.edit_ok` → retry.

### Output conventions

- **Confidence**: High / Medium / Low on findings and assumptions
- **Severity**: [Blocking] / [Non-blocking] / [Nit] on review findings
- **Synthesis**: conclude first, then cite evidence — never list observations without a "so what"

## Workflow

Non-trivial tasks follow phases. Each writes an artifact the next reads. **Human reviews each artifact before the next phase.**

All phase artifacts live in `{project}/.claude/docs/` and are gitignored.

| Phase | Skill | Artifact |
|-------|-------|----------|
| 1. Research | `/research-review <name>` | `.claude/docs/research/<name>.md` |
| 1a. Iterate | `/research-review review\|refine\|argue` | (updates research file) |
| 2. Plan | `/plan-review <name>` | `.claude/docs/plans/<name>.md` |
| 2a. Iterate | `/plan-review review\|refine` | (updates plan file) |
| — | `/compact` | — ← **saves artifacts + checkpoint + commits before compacting** |
| 3. Execute | `/execute-plan` | `.claude/docs/CHANGELOG.md` (append per step when used) |
| 4. Review | `/code-review <name>` | `.claude/docs/reviews/<name>.md` + PR |

All phase artifacts live in `.claude/docs/` — do NOT create them at the project root. Use root-level `docs/` only for human-facing project documentation if the repo needs it.

All skills run **directly in the current conversation** — no subagents for pipeline phases.

Ad-hoc: `/code-debug`, `/plan-refactor`. Utilities: `/claude-insights`, `/compact`. Planning: `/design-sprint`, `/scope-initiative`. Git: `/quick-commit`, `/quick-pr`.

## Issue Tracking

Linear ↔ GitHub integration is active. See `~/.claude/CLAUDE.md` for full conventions.

- All non-trivial tasks get a Linear issue before implementation starts
- Branch, commit, and PR naming must include `LIN-{id}` for auto-linking
- Stack: Code → GitHub | Tasks → Linear | Knowledge → Notion

## TODO annotations

- `TODO(0)` — critical; do not merge
- `TODO(1)` — high (architecture, major bugs)
- `TODO(2)` — medium (bugs, missing features)
- `TODO(3)` — low (polish, tests, docs)
- `TODO(4)` — open questions / investigations
- `PERF` — performance follow-ups

## Context Management

- Run `/compact` when context is getting noisy (often around 40%) — do not rely on auto-compaction
  - `/compact` is a custom skill: it saves active artifacts, writes a mid-session checkpoint note, commits + pushes, then calls the built-in compact with a seed prompt
  - This means compaction is always safe — work is committed to git before context is discarded
- Run `/clear` when switching to an unrelated task (no checkpoint needed — no work to save)
- Between execute steps: `/compact` handles the seed prompt automatically from the checkpoint
- **Do not spawn subagents or use Skill tool for research/plan/execute phases** — do the work directly in the main context so the user can follow and interject. Use Write/WebSearch/Read tools directly.

### Session metadata convention

Per-session files live in `.claude/sessions/{YYYY-MM-DD}T{HHMM}.md`. Run `/compact` to write one — at end of session it stops; mid-session it checkpoints and continues.

Contents: position, metadata (duration, tools, files), gotchas, friction signals, attribution notes, open questions, skill candidates, session insights, next session prompt.

The cartographer agent reads these files for friction analysis (`uv run cartographer --cron`), and `/claude-insights` can summarize patterns for any Claude-managed repo.

`.claude/sessions/` is gitignored — local only.

## Path convention

All configurable paths (Dropbox readings, Obsidian vault, PDF binaries) are defined in `src/agents/utils/config.py` via `pydantic-settings` and loaded from `.env`. Never hardcode user-specific paths in source files.

```python
from agents.utils.config import settings
settings.readings_dir      # ~/Dropbox/ai_readings (default, overridable)
settings.obsidian_vault    # ~/workspace/obsidian (default, overridable)
```

## Memory

Project memory lives in `.claude/memory/` and should stay short: user preferences, durable repo facts, and only non-obvious lessons that do not fit here.
