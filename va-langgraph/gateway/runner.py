"""Per-session LangGraph runner and SSE event queue.

Same contract as va-google-adk's session_manager:
  - get_or_create(session_id) → ensures queue exists before POST /chat is called
  - run_turn(session_id, message, page_url) → executes graph, enqueues SSE events
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from ..graph.builder import build_graph
from ..shared.schema import AssistantResponse

logger = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass
class _Session:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class GraphRunner:
    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._graph = build_graph()

    def get_or_create(self, session_id: str) -> _Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = _Session()
        return self._sessions[session_id]

    async def run_turn(
        self,
        session_id: str,
        message: str,
        page_url: str | None = None,
    ) -> None:
        """Execute one graph turn and push SSE events onto the session queue."""
        session = self.get_or_create(session_id)

        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "page_url": page_url,
            "intent": None,
            "routing_confidence": 0.0,
            "tool_results": [],
            "response": None,
            "blocked": False,
            "block_reason": None,
        }

        try:
            # Stream node-level updates so we can forward partial progress
            async for chunk in self._graph.astream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    # Forward tool_results additions as debug events
                    new_results = node_output.get("tool_results", [])
                    for tr in new_results:
                        await session.queue.put({"type": "tool_result", "data": tr})

                    # When the response is available, stream it
                    if node_output.get("response") is not None:
                        raw = node_output["response"]
                        try:
                            resp = AssistantResponse(**raw) if isinstance(raw, dict) else raw
                            await session.queue.put(
                                {"type": "response", "data": resp.model_dump()}
                            )
                        except Exception:
                            await session.queue.put(
                                {"type": "response", "data": AssistantResponse(message=str(raw)).model_dump()}
                            )

        except Exception as e:
            logger.exception("Graph turn error for session %s", session_id)
            await session.queue.put({"type": "error", "data": str(e)})
        finally:
            await session.queue.put(_SENTINEL)


runner = GraphRunner()
