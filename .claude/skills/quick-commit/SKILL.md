---
name: quick-commit
description: "Create a feature branch, stage changes, and commit — without opening a PR."
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash
---

Create a feature branch and commit current changes.

`$ARGUMENTS` — required. Format: `<slug> [commit message]`

## Branch naming

- If current branch matches `feature/lin-{id}-*` → carry ID: `feature/lin-{id}-<slug>`
- Otherwise → `feature/<slug>`

## Flow

1. `git status` — if tree is clean, stop
2. Create/switch to branch: `git checkout -b feature/<slug>`
3. List files to stage (`git diff --name-only` + `git diff --cached --name-only`)
4. Skip: `.env`, `*.pem`, `models/*.pkl`, files >10 MB
5. Show file list → ask **"Stage these files and commit? (y/n)"**
6. Commit message: `$ARGUMENTS` if provided, else derive from diff — imperative mood, under 72 chars, `type: description (LIN-{id})` if Linear ID present

## Safety

- Never force-push or amend published commits
- Never skip hooks (`--no-verify`)
- Never commit secrets or large binaries

To push and open a PR: `/quick-pr`.
