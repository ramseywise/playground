---
name: insights
description: "Parse Claude Code session JSONL files, surface workflow improvement signals, and suggest new skills. Project-agnostic across Claude repos; delegates heavy lifting to the cartographer agent."
disable-model-invocation: true
allowed-tools: Read Bash Write
---

Analyze workflow patterns and generate actionable improvements.

If `$ARGUMENTS` is `skills-only`, skip to section 2.

This skill is project-agnostic: use it in any Claude-managed repo that has session files and/or a friction log.

## 1. Session insights

```bash
uv run cartographer --dry-run          # stats only, no API call
uv run cartographer                    # full HTML report
uv run cartographer --cron             # SESSION.md + friction log analysis
```

Read `.claude/friction-log.jsonl` if it exists — surface repeated failure patterns.

### Key signals

| Signal | Indicates |
|--------|-----------|
| High tool error rate (`edit_failed`, `file_not_found`) | Plan steps with wrong file paths |
| Frequent user interruptions (<5s gap) | Plans ambiguous or steps too large |
| Compact frequency 3+ per session | Documents too large or phase boundaries too big |
| Bash dominates over Read/Edit | Over-reliance on shell |

For each pattern: **Signal** (numbers) → **Interpretation** → **Recommendation** (one concrete change) → **Attribution** (what likely caused it) → **Session note** (what to capture in the session file). Limit to 3-5 patterns.

## 2. Skill suggestions

Review `## Skill candidates` in session files and friction log patterns.

**Worth a skill if**: 3+ recurring steps, recognizable trigger, concrete enough to automate.
**Not worth a skill if**: one-off, already covered, too vague.

For approved candidates: generate `.claude/skills/<name>/SKILL.md`, remove from session file. For rejected: explain and remove.

If a session file exists, add a short `## Session insights` / `## Skill candidates` note so the next session can use the findings immediately.

## Output

Terse table: `| Area | Finding | Action taken |`
