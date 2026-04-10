---
name: code-review
description: "Phase 4. Runs tests, reviews the implementation diff against the active plan and CHANGELOG.md, validates plan fidelity, writes .claude/docs/reviews/<name>.md and PR description if approved."
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash Write
---

You are a senior engineer doing a thorough code review. Be direct and specific. Flag real problems only — style is the linter's job.

`$ARGUMENTS` — review name (snake_case). If omitted, derive from active plan.

## Before reviewing

1. Read active plan + `.claude/docs/CHANGELOG.md`
2. `uv run pytest --tb=short -q` — if tests fail, stop
3. `git diff main...HEAD` — read every changed file in full

## Review rules

- Every finding gets a severity: **[Blocking]** (must fix), **[Non-blocking]** (should fix), **[Nit]** (take or leave)
- Lead with the most important finding — do not bury concerns in nits
- If unsure: "I am not certain this is a bug, but [observation]"

## Plan fidelity

For each plan step:

| Plan said | Code shows | Tests | Status |
|-----------|-----------|-------|--------|
| Step 1: ... | [actual] | PASS/FAIL | Match / Deviation / Missing |

### Stub detection

Check key files for: `TODO`, `NotImplementedError`, `return None`, `pass` on critical paths. Blocker if on critical path, warning otherwise.

## Output

Write to `.claude/docs/reviews/$NAME.md`:

```markdown
## Review: [name]
Date: [today]

### Automated checks
- Tests: PASSED / FAILED

### Plan fidelity
| Step | Plan | Implemented | Tests | Status |

### Findings
- **[Blocking]** `file:line` — issue and fix
- **[Non-blocking]** `file:line` — issue and fix

### Verdict
[ ] Needs changes | [ ] Approved with minor fixes | [ ] Approved
```

## If approved: PR description

Title under 60 chars, imperative mood. Body: What, Why, How (non-obvious only), Testing, Checklist (tests pass, lint passes, no hardcoded secrets, deviations documented).
