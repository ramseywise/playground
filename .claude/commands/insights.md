---
name: insights
description: "Parse Claude Code session JSONL files, surface workflow improvement signals, and suggest new skills. Delegates heavy lifting to the cartographer agent."
tools: Read, Bash, Write
---

Analyze workflow patterns and generate actionable improvements.

If `$ARGUMENTS` is `skills-only`, skip to step 2.

---

## 1. Session insights

Run the cartographer agent's JSONL parser:

```bash
# Dry run — extract stats only, no API call
uv run cartographer --dry-run

# Full HTML report (requires ANTHROPIC_API_KEY)
uv run cartographer

# Custom paths
uv run cartographer --projects-dir ~/.claude/projects --output ~/.claude/usage-data/report.html

# Cron-triggered analysis (SESSION.md + friction log)
uv run cartographer --cron
```

Also read `.claude/friction-log.jsonl` if it exists — surface any repeated failure patterns.

Write findings to `.claude/docs/insights/[date].md`.

### Interpretation framework

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
| **Parallel session overlap** | High = intentional multi-agent | Low = single-threaded; consider background agents |

### Patterns that indicate context engineering problems

- **Frequent compacts in same session** → always-loaded files too large, or plan docs bloating context. Fix: move `@`-includes to project-level; trim SESSION.md.
- **High `edit_failed` rate** → editing files without reading first, or plan has wrong paths. Fix: enforce read-before-edit; add plan_check before execute.
- **High interruption rate** → agent making surprising decisions. Fix: smaller plan steps; clearer "done when" conditions.
- **Bash dominates tool usage** → using shell for file reads/searches. Fix: reinforce Read/Grep/Glob preference in hooks.

### What the data cannot tell you

- Whether the work was correct (only whether tests passed)
- Whether a deviation from the plan was good or bad
- Why a user interrupted (reviewing vs. correcting)
- Whether token usage was efficient for the task complexity

Do not over-interpret low-signal metrics. A long session on a hard problem is not a problem.

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

Review `## Skill candidates` in SESSION.md and friction log patterns.

For each candidate, apply these filters:

**Worth a skill if:**
- The workflow has 3+ steps that recur across sessions
- The trigger is recognizable (user says a phrase, or a file pattern appears)
- The steps are concrete enough to automate

**NOT worth a skill if:**
- One-off workflow unlikely to recur
- Already covered by an existing command
- Too vague to define a clear trigger and steps

For approved candidates: generate a `.md` file in `.claude/commands/`, remove the candidate from SESSION.md. For rejected candidates: explain why and remove from list.

---

## Output

```
| Area | Finding | Action taken |
|------|---------|-------------|
| [insight or skill candidate] | [what was found] | [recommendation or skill created] |
```

Keep output terse. This command is meant to run periodically — not produce a report to read.

If any finding is worth preserving long-term, save it as a project memory.
