---
name: insights_skill_suggest
description: "Analyzes session workflow patterns and skill candidates to suggest, generate, and register new reusable skills. Used by /insights command."
---

You are analyzing accumulated workflow patterns to identify and create new reusable skills.

## Inputs

1. **Skill candidates** from `SESSION.md` `## Skill candidates` section
2. **Friction log** from `.claude/friction-log.jsonl` (repeated patterns, error types)
3. **Recent session context** (what was done, what tools were used repeatedly)
4. **Existing skills** in `.claude/skills/` (avoid duplicates)

## Analysis process

### Step 1: Gather signals

Read these files:
- `.claude/docs/SESSION.md` → `## Skill candidates` section
- `.claude/friction-log.jsonl` → look for repeated tool sequences
- `.claude/skills/` → list existing skills to avoid duplicates

### Step 2: Evaluate candidates

For each candidate, apply these filters:

**Worth a skill if:**
- The workflow has 3+ steps that recur across sessions
- The trigger is recognizable (user says a phrase, or a file pattern appears)
- The steps are concrete enough to automate (not vague like "think about architecture")

**NOT worth a skill if:**
- One-off workflow unlikely to recur
- Already covered by an existing skill or command
- Too vague to define a clear trigger and steps

### Step 3: Generate skill definitions

For each approved candidate, generate a skill `.md` file:

```markdown
---
name: [skill_name]
description: "[One-line description — specific enough to match in future sessions]"
---

[Skill prompt content — what Claude should do when this skill is invoked]

## Trigger
[When to use this skill — specific phrases or contexts]

## Steps
[Numbered steps the skill performs]

## Done when
[Clear completion criteria]
```

### Step 4: Register

- Write skill file to `.claude/skills/[name].md`
- If it should be user-invocable, also create `.claude/commands/[name].md` with the invocation prompt
- Remove the candidate from SESSION.md `## Skill candidates` after generating

## Output format

For each candidate analyzed:

```
### [Candidate name]
**Signal**: [where it was observed — session, friction log, or manual flag]
**Verdict**: GENERATE | SKIP | MERGE (with existing skill)
**Reason**: [one sentence]
```

If GENERATE: create the skill file and confirm.
If SKIP: explain why and remove from candidates list.
If MERGE: identify which existing skill to extend and make the edit.
