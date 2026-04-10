---
name: quick-pr
description: "All-in-one: stage, commit, push, open PR, resolve merge conflicts, and merge — single command end to end."
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash Write
---

End-to-end PR flow for the current working tree.

`$ARGUMENTS` — optional commit message. If omitted, derive from diff.

## Flow

1. **Check**: `git status` + `git diff main...HEAD --stat` — if clean and no unpushed commits, stop
2. **Commit** (if needed): list files, skip secrets/`.env`/large binaries, show list + message, confirm before `git commit`
3. **Push**: `git push -u origin HEAD` — if rejected, `git pull --rebase origin main`, resolve conflicts, retry
4. **Draft PR**: read active plan if present, write title + body per conventions below, show full content, confirm before `gh pr create`
5. **Merge** (optional): ask **"Merge now? (y/n/squash)"** → merge with `--delete-branch` or leave open

## PR conventions

- Title: under 60 chars, imperative mood. If branch is `feature/lin-{id}-*`, title is `LIN-{id} description`
- Body: What (user perspective), Why (problem solved), How (non-obvious decisions only), Testing (tests + manual validation), Checklist

## Conflict resolution

`git diff --name-only --diff-filter=U` → read each → resolve → `git add` → `git rebase --continue` → retry push.

## Safety

- Never force-push or skip hooks
- Never commit `.env`, `*.pem`, `models/*.pkl`, or files >10 MB
