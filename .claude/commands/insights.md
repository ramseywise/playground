---
name: insights
description: "Parse Claude Code session JSONL files and surface workflow improvement signals. Optionally generates an HTML report via the Anthropic API."
tools: Read, Bash, Write
---

Analyze workflow patterns and generate actionable improvements. Runs two analyses in sequence.

If `$ARGUMENTS` is `skills-only`, skip to step 2.

---

## 1. Session insights

Run the JSONL parser:

```bash
# Dry run — extract stats only, no API call
uv run python src/agents/utils/session_insights.py --dry-run

# Full HTML report (requires ANTHROPIC_API_KEY)
uv run python src/agents/utils/session_insights.py

# Custom paths
uv run python src/agents/utils/session_insights.py --projects-dir ~/.claude/projects --output ~/.claude/usage-data/report.html
```

Also read `.claude/friction-log.jsonl` if it exists — surface any repeated failure patterns.

Write findings to `.claude/docs/insights/[date].md`.

Review the output for these signals:

| Signal | What to look for | Workflow implication |
|--------|-----------------|----------------------|
| **Tool error rate by type** | `edit_failed`, `file_not_found` spikes | Plan steps with wrong file paths; reading before editing |
| **User interruptions** (gap <5s) | High count = agent going off-track | Plans are ambiguous; steps too large |
| **Response time distribution** | Many >5m gaps = user reviewing, not waiting | Good: human-in-loop working. Many <10s = rubber-stamping |
| **Compact frequency** | Sessions compacted 3+ times | Context budget too tight; documents too large |
| **Top tools** | Bash dominates over Read/Edit | Over-reliance on shell; not using dedicated tools |
| **Error: `user_rejected`** | High count | Permission model too tight or commands surprising the user |
| **Session duration vs messages** | Long sessions, few messages | Steps sized too large |

For each pattern identified:

```
### [Pattern name]
**Signal**: [specific numbers from the data]
**Interpretation**: [what this likely means]
**Recommendation**: [one concrete change]
```

Limit to 3-5 patterns. End with a **Priority** section: which single change would have the highest impact.

---

## 2. Skill suggestions

Load the `insights_skill_suggest` skill. Review `## Skill candidates` in SESSION.md and friction log patterns. For each candidate: evaluate, generate skill files if warranted, clean up candidates list.

---

## Output

```
| Area | Finding | Action taken |
|------|---------|-------------|
| [insight or skill candidate] | [what was found] | [recommendation or skill created] |
```

Keep output terse. This command is meant to run periodically — not produce a report to read.

If any finding is worth preserving long-term, save it as a feedback memory via `/end`.
