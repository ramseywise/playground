# Session State

## Current position

**Active work**: Research agent refactor — note quality + batch automation
**Step**: Execute complete (all 6 plan steps done), pending `/review` + commit
**Tests**: 68/68 passing (4 deselected — pdftotext not in CI)
**Branch**: `cord/protestant-salamander-617e`
**Last updated**: 2026-04-06

## Agents overview

### Research agent (`src/agents/research/`)
**Status**: Active refactor in progress (uncommitted changes on branch)

Changes ready to commit:
- 6-section consolidated note template (Summary, Research Questions, Methodology, Key Techniques + failure modes, Critical Assessment, Open Questions)
- [[wikilinks]] throughout all sections + page references for citation
- Project context injection via `obsidian/project_context.md`
- Deprecation flagging (e.g. LangChain AgentExecutor → LangGraph)
- Batch processing: `research-agent --batch --max-pdfs 5 --max-tokens 4096`
- SHA-256 manifest for idempotent re-runs
- 4am cron script at `scripts/research-cron.sh`
- Token cost controls: `--max-pdfs` (default 5), `--max-tokens` (default 4096)

Docs:
- Research: `.claude/docs/research/research-agent-refactor.md`
- Plan: `.claude/docs/plans/research-agent-refactor.md`

### Visualizer agent (`src/agents/visualizer/`)
**Status**: Exists from prior work (commit `717da23`), ~994 LOC, 2 test files. No refactor planned yet.

Files: `__main__.py`, `image_fetcher.py`, `intake.py`, `outline.py`, `renderer.py`, `slide_writer.py`, `viz_classifier.py`
Stack: python-pptx, Pollinations.ai for images, rich for interactive CLI, yaml config
Tests: `test_outline.py`, `test_viz_classifier.py` (no batch/cron equivalent)
Research + plan complete. See Active docs section.

## Active gotchas

- `models.py` and `writer.py` show clean diff (dual-note code added then removed during iteration) — final state matches what was intended
- The existing ch13 note in `obsidian/topics/knowledge-graphs/` uses the OLD 8-section template. It won't be regenerated unless manually deleted and re-run.
- `obsidian/project_context.md` is committed to repo — update it as projects evolve or it becomes stale context for the agent
- Cron script paths are templated (`/path/to/repo`) — user must edit for their machine before `crontab -e`

## Active docs

- Research: `.claude/docs/research/visualizer-improvements.md`
- Plan: `.claude/docs/plans/visualizer_improvements.md`

## Open questions / blockers

- Research agent: `/review` + commit still pending on branch `cord/protestant-salamander-617e`
- Remote trigger setup: requires `/login` to claude.ai — deferred

## Next session prompt

```
/start

Continuing research agent refactor. Branch: cord/protestant-salamander-617e

State: All execute steps complete, 68/68 tests passing. Changes uncommitted.
Next: run /review, commit, push PR.

Docs: .claude/docs/plans/research-agent-refactor.md (plan),
      .claude/docs/research/research-agent-refactor.md (research)

Open: existing ch13 note uses old template, visualizer has no plans yet.
```

_compact: 2026-04-06 20:02_

_compact: 2026-04-07 23:54_
