---
name: end-session
description: "End-of-session checklist: write session metadata, decide what to save to memory, audit stale memory, and extract session insights."
---

Run the end-of-session checklist. Work through each section in order.

## 1. Session metadata

Write to `.claude/sessions/{YYYY-MM-DD}T{HHMM}.md`:

```markdown
# Session — {YYYY-MM-DD}T{HHMM}

## Position
- **Work**: [what was worked on]
- **Status**: [in-progress | complete | blocked]
- **Branch**: [branch name]
- **Tests**: [count passing / failing / skipped]

## Metadata
- **Compacted**: [yes/no]
- **Key tools**: [top 3-5 tools used]
- **Files touched**: [count or key files]

## Gotchas
[Non-obvious traps a cold agent must know.]

## Open questions
- [ ] [unresolved blockers]

## Skill candidates
[Multi-step workflows that recurred. 2-sentence description + trigger.]

## Next session prompt
[3-5 sentences. Where are we, first action, non-obvious context.]
```

Also mark completed steps as done in active plan files.

## 2. Memory decisions

Save only if non-obvious and will affect future behavior. Ask: "Would a cold session make a worse decision without this?"

| Type | Save when |
|------|-----------|
| **user** | Role, background, preferences |
| **project** | Non-obvious decisions, constraints not in code/git |
| **reference** | Pointers to external system locations |

**Feedback patterns go in enforcement** (hooks, CLAUDE.md, command checklists) — not memory.
**Do NOT save**: code patterns, git history, debugging solutions, anything in CLAUDE.md, ephemeral task details.

Check `MEMORY.md` for duplicates before writing.

## 3. Memory audit

| Check | Action |
|-------|--------|
| Stale — resolved phase/constraint | Compare against code + git log; flag |
| Redundant — duplicate of another | Merge into better one, delete weaker |
| Contradicts CLAUDE.md | Flag for removal |
| Completed `.claude/docs/` artifacts | Candidate for deletion — ask first |
| Sessions >30 days old, no open items | Candidate for deletion |

Present findings table, wait for confirmation before acting.

## 4. Session insights

Read `.claude/friction-log.jsonl` if it exists — surface patterns, then truncate.

Run `uv run cartographer --dry-run` — review for friction signals (high error rate, frequent compacts, high interruptions).

Friction worth preserving → add to relevant command or hook, not memory.
