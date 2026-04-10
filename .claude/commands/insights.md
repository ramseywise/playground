---
name: insights
description: "Parse Claude Code session JSONL files and surface workflow improvement signals. Optionally generates an HTML report via the Anthropic API."
tools: Read, Bash, Write
---

Run Claude Code session insights analysis.

## Usage

```bash
# Dry run — extract stats only, no API call
uv run python src/agents/utils/session_insights.py --dry-run

# Full HTML report (requires ANTHROPIC_API_KEY)
uv run python src/agents/utils/session_insights.py

# Custom paths
uv run python src/agents/utils/session_insights.py --projects-dir ~/.claude/projects --output ~/.claude/usage-data/report.html
```

If `$ARGUMENTS` is provided, pass it as additional CLI args.

## After running

Review the output for actionable patterns. Focus on the 3-5 strongest signals:

### Friction signals

| Signal | What to look for | Workflow implication |
|--------|-----------------|----------------------|
| **Tool error rate by type** | `edit_failed`, `file_not_found` spikes | Plan steps with wrong file paths; reading before editing |
| **User interruptions** (gap <5s) | High count = agent going off-track | Plans are ambiguous; steps too large |
| **Response time distribution** | Many >5m gaps = user reviewing, not waiting | Good: human-in-loop working. Many <10s = rubber-stamping |
| **Compact frequency** | Sessions compacted 3+ times | Context budget too tight; documents too large |
| **Top tools** | Bash dominates over Read/Edit | Over-reliance on shell; not using dedicated tools |
| **Error: `user_rejected`** | High count | Permission model too tight or commands surprising the user |
| **Session duration vs messages** | Long sessions, few messages | Steps sized too large |

### Context engineering problems

- **Frequent compacts**: always-loaded files too large, or plan/research docs bloating context
- **High `edit_failed` rate**: editing without reading first, or plan has wrong line numbers
- **High interruption rate**: agent making surprising decisions; need smaller steps or clearer scope
- **Bash dominates**: using shell for file reads instead of Read/Grep/Glob

### What the data cannot tell you

- Whether the work was correct (only whether tests passed)
- Whether a deviation from the plan was good or bad
- Why a user interrupted (reviewing vs. correcting)

## Output

For each pattern identified:

```
### [Pattern name]
**Signal**: [specific numbers from the data]
**Interpretation**: [what this likely means]
**Recommendation**: [one concrete change]
```

Limit to 3-5 patterns. End with a **Priority** section: which single change would have the highest impact.

If any finding is worth preserving long-term, save it as a feedback memory via `/end`.
