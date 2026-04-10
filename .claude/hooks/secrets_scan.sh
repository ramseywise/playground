#!/usr/bin/env bash
# PreToolUse hook for Write|Edit — blocks files containing secret patterns
# Scoped to src/ and scripts/ — skips .claude/, tests/, .env files

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0

# Only scan source and script files
echo "$path" | grep -qE '/(src|scripts)/.*\.(py|sh|yaml|yml|toml|json)$' || exit 0

# For Edit, check the new_string; for Write, check content
content=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.new_string // .content // empty')
[ -z "$content" ] && exit 0

issues=""

# Anthropic API key pattern
echo "$content" | grep -qE 'sk-ant-[a-zA-Z0-9]{20,}' && issues="$issues  [secret] Anthropic API key (sk-ant-*)\n"

# Google AI API key pattern
echo "$content" | grep -qE 'AIza[a-zA-Z0-9_-]{35}' && issues="$issues  [secret] Google API key (AIza*)\n"

# AWS access key
echo "$content" | grep -qE 'AKIA[A-Z0-9]{16}' && issues="$issues  [secret] AWS access key (AKIA*)\n"

# GitHub tokens
echo "$content" | grep -qE 'gh[ps]_[a-zA-Z0-9]{36,}' && issues="$issues  [secret] GitHub token (ghp_/ghs_*)\n"

# Replicate API token
echo "$content" | grep -qE 'r8_[a-zA-Z0-9]{20,}' && issues="$issues  [secret] Replicate API token (r8_*)\n"

# Generic hardcoded api_key assignment with actual value (not empty or env ref)
echo "$content" | grep -qE 'api_key\s*=\s*"[a-zA-Z0-9_-]{20,}"' && issues="$issues  [secret] hardcoded api_key value\n"

# Langfuse / observability keys
echo "$content" | grep -qE 'pk-lf-[a-zA-Z0-9]{20,}|sk-lf-[a-zA-Z0-9]{20,}' && issues="$issues  [secret] Langfuse key\n"

if [ -n "$issues" ]; then
  printf "Secrets detected — do not write keys to source files:\n%b" "$issues" >&2
  exit 2
fi

exit 0
