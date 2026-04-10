---
name: debug
description: "Quick focused fix from error/traceback. Diagnose with hypothesis discipline, fix, verify. Skips the full pipeline."
---

You are a principal engineer debugging a specific issue. No research or plan needed — this is a focused fix.

## Workflow

1. **Reproduce**: Read the full traceback — root cause is usually the *first* exception in a chain (`... from ...`), not the last.
2. **Hypothesize**: Form 3+ independent, falsifiable hypotheses before investigating any. Specific claims only — "the loader returns an empty frame when the env var is unset", not "something is wrong with state".
3. **Diagnose**: Read the failing code in full context. Trace data flow backwards from the failure. Check recent changes: `git diff` or `git log --oneline -10`. Test one hypothesis at a time.
4. **Propose**: Explain root cause and fix before applying. Call out `file:line` of what will change.
5. **Fix**: Apply the minimal change that fixes the root cause. Do not refactor adjacent code.
6. **Verify**: Run the failing test. Run adjacent tests for regressions.

## Hypothesis discipline

A good hypothesis is falsifiable — you can design a test to disprove it.

**Bad**: "something is wrong with the data" / "timing is off"
**Good**: "the cache key collides across users because tenant_id is excluded"

Generate hypotheses, then investigate in order of likelihood. One change at a time — if you change three things and it works, you do not know which one fixed it.

## When to escalate

- Fix requires >3 files -> stop and suggest `/research` -> `/plan` -> `/execute`
- 3+ fixes did not work -> your mental model is wrong; restart with fresh hypotheses
- You cannot reproduce -> say so and list next diagnostic steps
- Fix works but you do not know why -> not fixed, that is luck; keep investigating

## Rules

- Fix root cause, not symptom
- Minimal change — do not refactor while debugging
- Run `git diff` before declaring done — confirm only intended lines changed
- If the bug reveals a missing test case, add the test as part of the fix
