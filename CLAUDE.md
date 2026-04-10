# Agents — AI Research & Visualization Toolkit

Personal AI agent toolkit for processing research PDFs into an Obsidian knowledge base and generating presentations.

## Project Layout

```
src/agents/           # Python package (target structure — see plan)
  utils/              # Config, Claude client, shared utilities
  researcher/         # PDF → chunked notes → Obsidian vault
  presenter/          # Interactive presentation builder → PPTX
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

@.claude/rules/style.md
@.claude/rules/logging.md
@.claude/rules/ml.md

## Tooling

- `uv run pytest tests/` — full test suite
- `uv run pytest tests/researcher/` — research agent tests only
- `uv run pytest tests/presenter/` — presenter tests only
- `.env` never committed; `.env.example` is the template
- Formatting and linting run automatically via hooks on every file write — do not run manually unless asked
  (Requires hooks in `.claude/settings.json` — see this project's config for the reference implementation)

## Workflow

Non-trivial tasks follow phases. Each writes an artifact the next reads. **Human reviews each artifact before the next phase.**

All phase artifacts live in `{project}/.claude/docs/` and are gitignored. Only `CLAUDE.md` and `SESSION.md` live at the project root.

| Phase | Command | Artifact |
|-------|---------|----------|
| 1. Research | `/research` | `.claude/docs/RESEARCH.md` |
| 2. Plan | `/plan` | `.claude/docs/PLAN.md` |
| 2.5. Plan Review | `/plan-review` | `.claude/docs/PLAN.md` (iterated) |
| — | `/compact` | — ← **required before execute** |
| 3. Execute | `/execute` | `.claude/docs/CHANGELOG.md` (append per step) |
| 4. Review | `/review` | `.claude/docs/EVAL.md` + PR |

Do NOT create `CHANGES.md`, `RESEARCH.md`, `PLAN.md`, or `EVAL.md` at the project root.

All commands run **directly in the current conversation** — do not spawn subagents or use the Skill/Agent tools for pipeline phases.

Ad-hoc (skip pipeline): `/debug`, `/refactor`.

Utilities: `/insights`, `/rag-research`, `/insights-analysis`.

Planning: `/design-sprint`, `/initiative-scoping`.

Each phase command suggests the next step when complete. All commands are self-contained — no separate skills directory.

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

- Run `/compact` at 40% — do not wait for auto-compaction
- Run `/clear` when switching to an unrelated task
- Between execute steps: `/compact keep current step N, test count, open gotchas, and next 2 actions`
- **Do not spawn subagents or use Skill tool for research/plan/execute phases** — do the work directly in the main context so the user can follow and interject. Use Write/WebSearch/Read tools directly.

### SESSION.md convention (all projects)

Each project keeps `.claude/docs/SESSION.md`. Update it at the end of every session before `/clear`.

Contents:
- **Current position**: step, test count, last updated
- **Token log**: start/end tokens from status bar, turn count, whether compacted
- **Active gotchas**: non-obvious bugs and decisions that will bite the next session
- **Open questions / blockers**
- **Next session prompt**: copy-paste starter for the next cold session

Start each session with `/start` (reads `.claude/docs/SESSION.md` + `CLAUDE.md`), then paste the next session prompt.
`.claude/` is gitignored — SESSION.md is local only.

## Path convention

All configurable paths (Dropbox readings, Obsidian vault, PDF binaries) are defined in `src/agents/utils/config.py` via `pydantic-settings` and loaded from `.env`. Never hardcode user-specific paths in source files.

```python
from agents.utils.config import settings
settings.readings_dir      # ~/Dropbox/ai_readings (default, overridable)
settings.obsidian_vault    # ~/workspace/obsidian (default, overridable)
```

## Memory

Memory has two locations only — nothing else:

- **Workspace-level** (`~/.claude/projects/-Users-wiseer-workspace/memory/`): feedback, global patterns, cross-project lessons
- **Project-level** (`{project}/.claude/memory/`): project-specific state, decisions, session notes — version-controlled

Never write memory to any Dropbox path or any other `~/.claude/projects/` bucket. The Dropbox-based projects no longer exist.
