Run the `.claude/docs/SESSION.md` end-of-session checklist. Work through each item in order:

1. **Current position** — update step number, test count (run a quick count if unsure), and today's date
2. **Token log** — add a row. I'll tell you start/end tokens from the status bar; ask me if you don't have them
3. **Active docs** — update to reflect the current plan/research files (or clear if task is complete)
4. **Active gotchas** — add new non-obvious ones discovered this session; remove any that are now resolved
5. **Open questions** — add new blockers; close resolved ones
6. **Next session prompt** — rewrite it to reflect exactly where we are: current step + 3-5 lines of must-know context so a cold session can start immediately without reading the plan file
7. **Friction check** — read `.claude/friction-log.jsonl` if it exists. Surface any patterns (repeated failures, common error types). If a pattern is worth remembering (e.g., "ruff always fails on X import pattern"), save it to memory. Then truncate the log: `> .claude/friction-log.jsonl`

Then check the active plan file (from `## Active docs`): mark any completed steps as done with today's date.

8. **Skill candidates** — review what was done this session. If you notice a multi-step workflow that was repeated (or would likely recur in future sessions), add it to `## Skill candidates` in SESSION.md:
   ```
   ## Skill candidates
   - [workflow name]: [2-sentence description of the repeated pattern and when it triggers]
   ```
   Don't create the skill yet — just capture the signal. Run `/insights` to analyze accumulated candidates and generate skills.

Finally, check if anything from this session is worth saving to memory (non-obvious decisions, lessons that apply to future projects). If yes, write or update the relevant memory file.

Keep everything terse. No trailing summaries.
