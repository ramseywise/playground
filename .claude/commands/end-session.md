---
name: end
description: "End-of-session checklist: write session metadata, decide what to save to memory, audit stale memory, and extract session insights."
---

Run the end-of-session checklist. Work through each section in order.

---

## 1. Write session metadata

Write a per-session file to `.claude/sessions/{YYYY-MM-DD}T{HHMM}.md` (24h local time, e.g. `2026-04-10T1430.md`). This supports parallel sessions on the same day.

Write these fields:

```markdown
# Session — {YYYY-MM-DD}T{HHMM}

## Position
- **Work**: [what was worked on]
- **Status**: [in-progress | complete | blocked]
- **Branch**: [branch name]
- **Tests**: [count passing / failing / skipped]

## Metadata
- **Duration**: [approx from first to last message]
- **Compacted**: [yes/no, or "yes (xN)"]
- **Key tools**: [top 3-5 tools used]
- **Files touched**: [count or key files]

## Gotchas
[Non-obvious traps a cold agent must know. Remove resolved items from prior sessions.]

## Open questions
- [ ] [unresolved blockers]
- [x] [resolved — one-line note]

## Skill candidates
[Multi-step workflows that recurred or would recur. 2-sentence description + trigger.]

## Next session prompt
[3-5 sentences. Where are we, what is the first action, what context is not obvious from code?]
```

Also check active plan files: mark completed steps as done with today's date.

---

## 2. Memory: deciding what to save

A memory is worth writing if and only if it is **non-obvious and will affect future behavior**.

Ask: "Would a cold session, reading only MEMORY.md and the project code, make a worse decision without this?"

### Memory types for project-level (`.claude/memory/`)

| Type | Save when | Example |
|------|-----------|---------|
| **user** | Learn about the user's role, background, or preferences | "user is new to React but has 10yr Go" |
| **project** | Non-obvious decisions, constraints, or state not in code/git | "auth rewrite is compliance-driven" |
| **reference** | Pointer to external system locations | "pipeline bugs in Linear project INGEST" |

**Feedback patterns go in enforcement, not memory** — see CLAUDE.md § Memory for the full rule. Code quality patterns → `/code-review` checklist. Workflow conventions → CLAUDE.md. Only save feedback memories for cross-cutting behavioral guidance that doesn't fit elsewhere.

### What NOT to save
- Code patterns, conventions, architecture — derivable from the code
- Git history, recent changes — `git log` is authoritative
- Debugging solutions — the fix is in the code
- Anything already in CLAUDE.md
- Ephemeral task details or current conversation context

### Writing a memory file

```markdown
---
name: [short descriptive name]
description: [one-line — used to decide relevance in future sessions]
type: [user | project | reference]
---

[body — for project types: fact, then **Why:** and **How to apply:**]
```

After writing, add one line to `MEMORY.md`: `- [Title](filename.md) — one-line hook`

Before writing, check MEMORY.md for duplicates. Update existing files rather than creating new ones.

---

## 3. Memory audit

Check for stale, redundant, or contradictory memories.

**Project memory (`.claude/memory/`):**

| Check | Action |
|-------|--------|
| **Stale** — describes a phase, status, or constraint that resolved | Compare against code + git log; flag if outdated |
| **Redundant** — another memory says the same thing | Merge into the better one, delete the weaker |
| **Contradicts CLAUDE.md** | Flag for removal — CLAUDE.md is authoritative |

**Phase artifacts (`.claude/docs/`):**

| Check | Action |
|-------|--------|
| Completed artifacts (plans, research) where task is merged | Candidate for deletion — list them, ask before deleting |
| Plans where all steps are done | Candidate for deletion or archival |

**Session files (`.claude/sessions/`):**

| Check | Action |
|-------|--------|
| Sessions >30 days old with no open gotchas or questions | Candidate for deletion |
| Next session prompt describes completed work | Flag as stale |

Present a short table of findings and wait for user confirmation before taking action.

---

## 4. Session insights

**Friction log:** If `.claude/friction-log.jsonl` exists, read it. Surface any patterns. Then truncate: `> .claude/friction-log.jsonl`

**Cartographer:** Run the JSONL analysis:

```bash
uv run cartographer --dry-run
```

Review for friction signals:
- High tool error rate → plans with wrong file paths?
- Many user interruptions → steps too large or ambiguous?
- High compact frequency → documents too large?

Friction patterns worth preserving → add as a checklist item in the relevant command (e.g., `/code-review`, `/execute-plan`), or as a hook if it can be statically enforced. Do NOT save as memory.

---

Keep everything terse. No trailing summaries.
