---
name: end
description: "End-of-session checklist: update SESSION.md, decide what to save to memory, audit stale memory, and extract session insights from JSONL."
---

Run the end-of-session checklist. Work through each section in order.

---

## 1. Update SESSION.md

Read `.claude/docs/SESSION.md` and update these fields only:

**Current position:**
- Update the phase label and status
- Update `Last updated` to today's date

**Token log:**
- Add one row. Ask the user for start/end tokens from the status bar if you don't have them.
- Record whether `/compact` was run (yes/no, or "yes (xN)" if run multiple times)

**Active gotchas:**
- Add new non-obvious traps discovered this session
- Remove any gotchas that are now resolved
- Keep only what a cold agent must know to avoid a real mistake

**Open questions:**
- Add new blockers with `[ ]`
- Close resolved ones with `[x]` and a one-line resolution note

**Next session prompt:**
- Rewrite entirely — do not append to old text
- Must answer: where exactly are we, what is the first action, and what context would a cold agent need that is not obvious from the code?
- Target: 3-5 sentences. Longer is a sign of unresolved complexity.

**Active docs:**
- Update plan status if a phase completed this session
- Add new research files if created

Also check the active plan file: mark any completed steps as done with today's date.

---

## 2. Memory: deciding what to save

A memory is worth writing if and only if it is **non-obvious and will affect future behavior**.

Ask: "Would a cold session, reading only MEMORY.md and the project code, make a worse decision without this?"

### The 4 memory types

| Type | Save when | Example |
|------|-----------|---------|
| **user** | Learn something about the user's role, background, or preferences | "user is new to React but has 10yr Go experience" |
| **feedback** | User corrects your approach OR confirms a non-obvious choice | "don't mock the DB — prior incident showed mock/prod divergence" |
| **project** | Non-obvious decisions, constraints, or state not in code or git | "auth rewrite is compliance-driven, not tech-debt" |
| **reference** | Pointer to where something lives in an external system | "pipeline bugs tracked in Linear project INGEST" |

### What NOT to save
- Code patterns, conventions, architecture — derivable from reading the code
- Git history, recent changes — `git log` is authoritative
- Debugging solutions — the fix is in the code; the commit message has the context
- Anything already in CLAUDE.md
- Ephemeral task details or current conversation context

### Writing a memory file

```markdown
---
name: [short descriptive name]
description: [one-line — used to decide relevance in future sessions]
type: [user | feedback | project | reference]
---

[body — for feedback/project types: rule, then **Why:** and **How to apply:**]
```

After writing, add one line to `MEMORY.md`: `- [Title](filename.md) — one-line hook`

Before writing, check MEMORY.md for duplicates. Update existing files rather than creating new ones.

---

## 3. Memory audit

Check for stale, redundant, or contradictory memories.

**Global memory** (`~/.claude/projects/-Users-wiseer-workspace/memory/`):

For each `.md` file (skip MEMORY.md itself), apply:

| Check | Action |
|-------|--------|
| **Stale project state** — describes a phase, status, or TODO that may have resolved | Compare against current files and git log; flag if outdated |
| **Redundant** — another memory file says essentially the same thing | Merge into the more complete one, delete the weaker |
| **Contradicts CLAUDE.md** — recommends something CLAUDE.md now forbids | Flag for removal — CLAUDE.md is authoritative |

**Project `.claude/docs/`:**

| Check | Action |
|-------|--------|
| Completed task artifacts (RESEARCH.md, PLAN.md, CHANGELOG.md) where task is merged/DONE | Candidate for deletion — present list, ask before deleting |
| SESSION.md date >2 weeks old or next session prompt describes a completed step | Prompt to reset |
| Plans where all steps are marked done | Candidate for deletion or archival |

Present a short table of findings and wait for user confirmation before taking action.

---

## 4. Session insights

**Friction log:** If `.claude/friction-log.jsonl` exists, read it. Surface any patterns (repeated failures, common error types). If a pattern is worth remembering (e.g., "ruff always fails on X import pattern"), save it to memory. Then truncate the log: `> .claude/friction-log.jsonl`

**JSONL analysis:** If `src/agents/utils/session_insights.py` exists:

```bash
uv run python src/agents/utils/session_insights.py --session current --dry-run
```

Review the output for friction signals:
- High tool error rate → plans with wrong file paths?
- Many user interruptions → steps too large or ambiguous?
- High compact frequency → documents or skills too large?

If any pattern is worth preserving, save it as a feedback memory.

**Skill candidates:** Review what was done this session. If you notice a multi-step workflow that was repeated (or would likely recur in future sessions), add it to `## Skill candidates` in SESSION.md:

```
## Skill candidates
- [workflow name]: [2-sentence description of the repeated pattern and when it triggers]
```

Don't create the skill yet — just capture the signal. Run `/insights` to analyze accumulated candidates and generate skills.

---

Keep everything terse. No trailing summaries.
