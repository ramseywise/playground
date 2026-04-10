---
name: insights
description: "Parse Claude Code session JSONL files, surface workflow improvement signals, and suggest new skills. Delegates heavy lifting to the cartographer agent."
disable-model-invocation: true
allowed-tools: Read Bash Write
---

Analyze workflow patterns and generate actionable improvements.

If `$ARGUMENTS` is `skills-only`, skip to section 2.

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
| Compact frequency 3+ per session | Documents too large |
| Bash dominates over Read/Edit | Over-reliance on shell |

For each pattern: **Signal** (numbers) → **Interpretation** → **Recommendation** (one concrete change). Limit to 3-5 patterns.

## 2. Skill suggestions

Review `## Skill candidates` in session files and friction log patterns.

**Worth a skill if**: 3+ recurring steps, recognizable trigger, concrete enough to automate.
**Not worth a skill if**: one-off, already covered, too vague.

For approved candidates: generate `.claude/skills/<name>/SKILL.md`, remove from session file. For rejected: explain and remove.

## Output

Terse table: `| Area | Finding | Action taken |`
