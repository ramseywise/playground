#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — enforces TS/Next.js standards
# Blocks (exit 2) on violations. Agent must fix before proceeding.

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '\.(tsx?|ts)$' || exit 0

issues=""

# [no-use-client-in-layout] "use client" disables SSR for the entire subtree
if echo "$path" | grep -qE 'layout\.(tsx?|ts)$'; then
  line=$(grep -n '"use client"' "$path" 2>/dev/null | head -1 || true)
  [ -n "$line" ] && issues="$issues  [no-use-client-in-layout] disables SSR for the whole subtree — remove: $line\n"
fi

# [no-console-log] No console.log in src/ — remove before committing
if echo "$path" | grep -qE '/src/'; then
  echo "$path" | grep -qE '\.(test|spec)\.(tsx?|ts)$' && exit 0
  line=$(grep -n 'console\.log' "$path" 2>/dev/null | grep -v '// noqa' | head -1 || true)
  [ -n "$line" ] && issues="$issues  [no-console-log] remove debug logging: $line\n"
fi

# [no-hardcoded-model] Model strings belong in the LlmAgent definition, not tools or lib
if echo "$path" | grep -qE 'src/agents/tools/|src/lib/'; then
  line=$(grep -nE '"gemini-[^"]+"|"claude-[^"]+"|"gpt-[^"]+"' "$path" 2>/dev/null | grep -v '// noqa' | head -1 || true)
  [ -n "$line" ] && issues="$issues  [no-hardcoded-model] move model string to LlmAgent definition: $line\n"
fi

# [no-any] Avoid 'as any' — use unknown and narrow, or fix the type
line=$(grep -nE '\bas any\b' "$path" 2>/dev/null | grep -v '// noqa' | head -1 || true)
[ -n "$line" ] && issues="$issues  [no-any] use unknown + type guard instead: $line\n"

if [ -n "$issues" ]; then
  printf "TS quality violations in %s:\n%b" "$path" "$issues" >&2
  exit 2
fi

# [file-size] Advisory only — does not block
lines=$(wc -l < "$path" 2>/dev/null || echo 0)
[ "$lines" -gt 400 ] && echo "Warning: $path is $lines lines (>400) — consider splitting" >&2

exit 0
