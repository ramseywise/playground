"""Per-session LangGraph runner and SSE event queue.

Same contract as va-google-adk's session_manager:
  - get_or_create(session_id) → ensures queue exists before POST /chat is called
  - run_turn(session_id, message, page_url, trace_id, user_id) → executes graph, enqueues SSE events

Event types emitted:
  text        — streaming token chunk from the format node
  tool_result — debug: intermediate tool call result from a domain subgraph
  response    — final AssistantResponse (structured)
  error       — unhandled exception message
  done        — sentinel; emitted by the gateway stream endpoint
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

import memory as memory_store
from graph.builder import build_graph
from schema import AssistantResponse

logger = logging.getLogger(__name__)

_SENTINEL = object()

# Nodes that produce the final response directly (no format_node pass)
_TERMINAL_NODES = {"format", "direct", "blocked", "escalation", "memory"}


@dataclass
class _Session:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class GraphRunner:
    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._graph = None

    def init_graph(self, checkpointer) -> None:
        """Called from the FastAPI lifespan once the checkpointer is ready."""
        self._graph = build_graph(checkpointer=checkpointer)

    def get_or_create(self, session_id: str) -> _Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = _Session()
        return self._sessions[session_id]

    async def run_turn(
        self,
        session_id: str,
        message: str,
        page_url: str | None = None,
        trace_id: str | None = None,
        user_id: str = "default",
    ) -> None:
        """Execute one graph turn and push SSE events onto the session queue.

        Uses astream_events(v2) so:
          - text chunks stream in real-time from the format node
          - tool_results are forwarded as debug events from domain subgraphs
          - the final AssistantResponse is emitted when a terminal node completes
        """
        session = self.get_or_create(session_id)

        config = {"configurable": {"thread_id": session_id}}
        if trace_id:
            config["metadata"] = {"trace_id": trace_id}

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "user_id": user_id,
            "user_preferences": [],  # populated by memory_load_node
            "page_url": page_url,
            "intent": None,
            "routing_confidence": 0.0,
            "tool_results": [],
            "response": None,
            "blocked": False,
            "block_reason": None,
        }

        if self._graph is None:
            raise RuntimeError("GraphRunner.init_graph() has not been called")

        def _evt(type_: str, data) -> dict:
            e: dict = {"type": type_, "data": data}
            if trace_id:
                e["trace_id"] = trace_id
            return e

        _last_response_message: str | None = None

        try:
            async for event in self._graph.astream_events(
                initial_state,
                config=config,
                version="v2",
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                # Stream text tokens from the format node only
                if kind == "on_chat_model_stream" and node == "format":
                    chunk = event["data"].get("chunk")
                    if chunk is not None:
                        text = chunk.content if isinstance(chunk.content, str) else ""
                        if text:
                            await session.queue.put(_evt("text", text))

                # Forward tool results as debug events from domain subgraph completions
                elif kind == "on_chain_end" and node and node not in _TERMINAL_NODES | {"guardrail", "analyze", "memory_load"}:
                    output = event["data"].get("output", {})
                    if isinstance(output, dict):
                        for tr in output.get("tool_results", []):
                            await session.queue.put(_evt("tool_result", tr))

                # Capture final response when a terminal node completes
                elif kind == "on_chain_end" and node in _TERMINAL_NODES:
                    output = event["data"].get("output", {})
                    if isinstance(output, dict) and output.get("response") is not None:
                        raw = output["response"]
                        try:
                            resp = AssistantResponse(**raw) if isinstance(raw, dict) else raw
                            await session.queue.put(_evt("response", resp.model_dump()))
                            _last_response_message = resp.message
                        except Exception:
                            await session.queue.put(
                                _evt("response", AssistantResponse(message=str(raw)).model_dump())
                            )

        except Exception as e:
            logger.exception("Graph turn error for session %s", session_id)
            await session.queue.put(_evt("error", str(e)))
        finally:
            await session.queue.put(_SENTINEL)
            if _last_response_message:
                await _save_session_summary(user_id, session_id, message, _last_response_message)


async def _save_session_summary(
    user_id: str,
    session_id: str,
    user_message: str,
    agent_response: str,
) -> None:
    try:
        from model_factory import resolve_chat_model
        prompt = (
            f"In one sentence, summarise this interaction:\n"
            f"User: {user_message[:200]}\n"
            f"Agent: {agent_response[:200]}"
        )
        resp = await resolve_chat_model("small").ainvoke(prompt)
        summary = resp.content.strip()[:500]
        await memory_store.upsert(user_id, f"session:{session_id}", summary)
    except Exception as e:
        logger.warning("Failed to save session summary for %s: %s", session_id, e)


runner = GraphRunner()
