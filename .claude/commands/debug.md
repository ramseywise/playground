---
name: debug
description: "Quick fix from error/traceback. Diagnose → fix → verify. Skips the full pipeline."
---

Read `.claude/skills/code_debug.md` and follow it end-to-end.

**Review protocol**: A PreToolUse hook gates source file edits. When blocked, show the proposed change as a before/after code block, wait for confirmation, then `touch .claude/.edit_ok` and retry.

<!-- NOTE: Delete this file if code_debug.md is generic.
     Vanilla Claude already does: read error → find file → fix → run tests.
     This command only earns its keep if code_debug.md encodes discipline
     like "reproduce first", "check git blame", "minimal failing test case",
     or project-specific debug patterns (e.g. "check the middleware chain first"). -->
