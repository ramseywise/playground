"""LangGraph implementation of AgentRuntime."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.langgraph.schemas.state import GraphState
from orchestrator.pipeline_failure import (
    agent_output_for_pipeline_failure,
    stream_error_data,
)
from orchestrator.runtime_protocol import AgentRuntime  # noqa: F401 — Protocol, not base class
from orchestrator.schemas import AgentInput, AgentOutput, StreamEvent

log = logging.getLogger(__name__)

# Node names registered in the graph — used to filter astream_events noise.
_GRAPH_NODES = frozenset(
    [
        "planner",
        "retriever",
        "qa_policy_retrieval",
        "qa_retrieval_gate",
        "reranker",
        "qa_policy_rerank",
        "qa_rerank_gate",
        "escalation",
        "clarify",
        "scheduler",
        "confirm",
        "answer",
        "post_answer_evaluator",
        "summarizer",
    ]
)


def _extract_answer(result: dict) -> str:
    fa = result.get("final_answer")
    if fa is None:
        return ""
    if hasattr(fa, "content"):
        return str(fa.content)
    return str(fa)


def _extract_citations(result: dict) -> list[dict]:
    citations = result.get("citations") or []
    out = []
    for c in citations:
        if hasattr(c, "model_dump"):
            out.append(c.model_dump())
        elif isinstance(c, dict):
            out.append(c)
    return out


def _build_state(input: AgentInput) -> GraphState:
    return GraphState(
        query=input.query,
        messages=[HumanMessage(content=input.query)],
        locale=input.locale,
        market=input.market,
    )


class LangGraphRuntime:
    """Implements AgentRuntime over a compiled LangGraph graph.

    Args:
        graph: A compiled LangGraph graph (from ``build_graph(checkpointer)``).
               If omitted, falls back to the module-level ``poc_graph`` which
               uses an in-process ``MemorySaver`` — suitable for the CLI and tests.
    """

    def __init__(self, graph: Any = None) -> None:
        if graph is None:
            from orchestrator.langgraph.graph import (
                poc_graph,
            )  # lazy — CLI/test path

            self._graph = poc_graph
        else:
            self._graph = graph

    async def run(self, input: AgentInput) -> AgentOutput:
        config: RunnableConfig = {"configurable": {"thread_id": input.thread_id}}
        log.info(
            "LangGraphRuntime.run thread_id=%s query_len=%d",
            input.thread_id,
            len(input.query),
        )
        try:
            result = await self._graph.ainvoke(_build_state(input), config=config)
        except Exception:
            log.exception("LangGraphRuntime.run failed thread_id=%s", input.thread_id)
            return agent_output_for_pipeline_failure()
        return AgentOutput(
            answer=_extract_answer(result),
            citations=_extract_citations(result),
            mode=result.get("mode"),
            latency_ms=result.get("latency_ms") or {},
            escalated=result.get("qa_outcome") == "escalate",
        )

    async def stream(self, input: AgentInput) -> AsyncIterator[StreamEvent]:
        from langgraph.errors import GraphInterrupt

        config: RunnableConfig = {"configurable": {"thread_id": input.thread_id}}
        log.info(
            "LangGraphRuntime.stream thread_id=%s query_len=%d",
            input.thread_id,
            len(input.query),
        )
        try:
            async for event in self._graph.astream_events(
                _build_state(input), config=config, version="v2"
            ):
                ev_type = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                if ev_type == "on_chain_start" and name in _GRAPH_NODES:
                    yield StreamEvent(kind="node_start", node=name)

                elif ev_type == "on_chain_end" and name in _GRAPH_NODES:
                    # Attach final answer + citations on the answer node completion.
                    extra: dict = {}
                    if name == "answer":
                        output = data.get("output") or {}
                        fa = output.get("final_answer")
                        if fa is not None:
                            extra["answer"] = (
                                fa.content if hasattr(fa, "content") else str(fa)
                            )
                        cits = output.get("citations") or []
                        extra["citations"] = [
                            (c.model_dump() if hasattr(c, "model_dump") else c)
                            for c in cits
                        ]
                    yield StreamEvent(kind="node_end", node=name, data=extra)

                elif ev_type == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk is not None:
                        content = (
                            chunk.content if hasattr(chunk, "content") else str(chunk)
                        )
                        if content:
                            yield StreamEvent(
                                kind="token",
                                node=name or None,
                                data={"text": content},
                            )

        except GraphInterrupt as exc:
            # Graph paused at a HITL gate — surface interrupt payload to client.
            payload = exc.args[0] if exc.args else {}
            log.info(
                "LangGraphRuntime.stream interrupt thread_id=%s payload_kind=%s",
                input.thread_id,
                payload.get("kind")
                if isinstance(payload, dict)
                else type(payload).__name__,
            )
            yield StreamEvent(kind="interrupt", data={"payload": payload})
            return

        except Exception:
            log.exception("LangGraphRuntime.stream error thread_id=%s", input.thread_id)
            yield StreamEvent(kind="error", data=stream_error_data())
            return

        yield StreamEvent(kind="done")

    async def stream_resume(
        self, thread_id: str, value: object
    ) -> AsyncIterator[StreamEvent]:
        """Resume a paused HITL graph and stream remaining events."""
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        log.info("LangGraphRuntime.stream_resume thread_id=%s", thread_id)
        try:
            async for event in self._graph.astream_events(
                Command(resume=value), config=config, version="v2"
            ):
                ev_type = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                if ev_type == "on_chain_start" and name in _GRAPH_NODES:
                    yield StreamEvent(kind="node_start", node=name)

                elif ev_type == "on_chain_end" and name in _GRAPH_NODES:
                    extra: dict = {}
                    if name == "answer":
                        output = data.get("output") or {}
                        fa = output.get("final_answer")
                        if fa is not None:
                            extra["answer"] = (
                                fa.content if hasattr(fa, "content") else str(fa)
                            )
                        cits = output.get("citations") or []
                        extra["citations"] = [
                            (c.model_dump() if hasattr(c, "model_dump") else c)
                            for c in cits
                        ]
                    yield StreamEvent(kind="node_end", node=name, data=extra)

                elif ev_type == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk is not None:
                        content = (
                            chunk.content if hasattr(chunk, "content") else str(chunk)
                        )
                        if content:
                            yield StreamEvent(
                                kind="token",
                                node=name or None,
                                data={"text": content},
                            )

        except GraphInterrupt as exc:
            payload = exc.args[0] if exc.args else {}
            yield StreamEvent(kind="interrupt", data={"payload": payload})
            return

        except Exception:
            log.exception(
                "LangGraphRuntime.stream_resume error thread_id=%s", thread_id
            )
            yield StreamEvent(kind="error", data=stream_error_data())
            return

        yield StreamEvent(kind="done")

    async def resume(self, thread_id: str, value: object) -> AgentOutput:
        from langgraph.types import Command

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        try:
            result = await self._graph.ainvoke(Command(resume=value), config=config)
        except Exception:
            log.exception("LangGraphRuntime.resume failed thread_id=%s", thread_id)
            return agent_output_for_pipeline_failure()
        return AgentOutput(
            answer=_extract_answer(result),
            citations=_extract_citations(result),
            mode=result.get("mode"),
            latency_ms=result.get("latency_ms") or {},
            escalated=result.get("qa_outcome") == "escalate",
        )


assert isinstance(LangGraphRuntime(graph=None), AgentRuntime), (
    "LangGraphRuntime must satisfy AgentRuntime"
)

__all__ = ["LangGraphRuntime"]
