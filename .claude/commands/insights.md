Analyze workflow patterns and generate actionable improvements. Runs two analyses in sequence:

1. **Session insights** — load the `insights_analysis` skill. Read `.claude/friction-log.jsonl` and recent session data. Produce the top 3-5 patterns with recommendations. Write findings to `.claude/docs/insights/[date].md`.

2. **Skill suggestions** — load the `insights_skill_suggest` skill. Review `## Skill candidates` in SESSION.md and friction log patterns. For each candidate: evaluate, generate skill files if warranted, clean up candidates list.

Output a summary table at the end:

```
| Area | Finding | Action taken |
|------|---------|-------------|
| [insight or skill candidate] | [what was found] | [recommendation or skill created] |
```

If run with an argument (e.g., `/insights skills-only`), skip the session insights and go straight to skill suggestions.

Keep output terse. This command is meant to run periodically — not produce a report to read.
