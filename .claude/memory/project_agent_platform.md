---
name: Agent platform architecture decisions
description: Core architecture decisions for the agents monorepo — agents are independent processes, no LangGraph, obsidian is curated corpus output
type: project
---

Agents are **independent processes**, not integrated via a shared layer. Each agent is a standalone CLI tool that can be triggered manually or via cron.

**Why:** User views them as separate entities with different triggers and lifecycles — not nodes in an orchestration graph.

**How to apply:** No LangGraph, no shared state machine, no inter-agent communication. Shared infrastructure is limited to: Claude client, config loading, logging, paths. Each agent owns its own models, prompts, and orchestration logic.

Key facts:
- Obsidian vault = curated corpus (output of research agent, not integration layer)
- Raw PDFs (~1GB) stay in Dropbox, accessed via configurable path in `.env`
- Future agents: possibly RAG search, websearch — keep shared core extensible but lightweight
- No study guide maker planned
- Deployment: CLI tools, possibly cron-triggered — no web server or API gateway needed
