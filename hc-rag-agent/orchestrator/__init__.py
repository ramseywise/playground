"""Runtime-agnostic orchestrator.

Runtimes:
  langgraph/  — LangGraph StateGraph implementation (active)
  adk/        — Google ADK implementation (stub)

Shared:
  shared/schemas.py   — AgentInput / AgentOutput / StreamEvent
  shared/memory/      — episodic memory (LangGraph Store / ADK MemoryService)
  shared/protocols/   — MCP toolset + A2A AgentCard definitions
  shared/tools/       — RetrieverTool and future tool registry
"""
