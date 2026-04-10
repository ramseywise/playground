---
name: start
description: "Start a session. Reads SESSION.md and CLAUDE.md, outputs current position, active gotchas, and next action."
---

Read `.claude/docs/SESSION.md` and `CLAUDE.md` in the current project directory.

## Reading SESSION.md

Extract in this order:
1. **Active docs** — which plan and research files are live
2. **Current position** — what phase, what step, when last updated
3. **Active gotchas** — non-obvious traps to carry into this session
4. **Next session prompt** — the distilled context for starting immediately

Do NOT read plan or research files unless SESSION.md is blocked or the next session prompt says to.

## Output

1. **Current position** — step, test count, last updated
2. **Active gotchas** — list them concisely
3. **Next action** — one sentence: what we are doing first
4. **Token tip** — remind me to /compact at 40% and check the status bar

Keep the output short — this is a session kickoff, not a briefing.
