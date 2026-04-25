#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — runs tsc --noEmit on ts_google_adk/ after any TS file edit.
# Skips gracefully when node_modules/.bin/tsc is absent (deps not installed).
# Blocks (exit 2) on type errors so the agent must fix them before proceeding.

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '\.(tsx?|ts)$' || exit 0

# Derive the ts_google_adk project root from the file path
project_root=$(echo "$path" | grep -oE '.*/ts_google_adk' | head -1)
[ -z "$project_root" ] && exit 0

tsc_bin="$project_root/node_modules/.bin/tsc"
[ -x "$tsc_bin" ] || exit 0  # skip if deps not installed

result=$("$tsc_bin" --noEmit --project "$project_root/tsconfig.json" 2>&1)
rc=$?

if [ $rc -ne 0 ]; then
  # Show up to 10 errors so the agent has enough context to fix them
  printf "TypeScript errors in %s:\n%s\n" "$project_root" "$(echo "$result" | grep -E 'error TS' | head -10)" >&2
  exit 2
fi

exit 0
