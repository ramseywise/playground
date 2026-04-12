from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from core.clients.llm import LLMClient
from librarian.generation.generator import (
    build_prompt,
    call_llm,
    extract_citations,
)
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

log = get_logger(__name__)

# Minimum confidence_score required to generate a response.
# Below this threshold the graph should retry retrieval (CRAG loop).
DEFAULT_CONFIDENCE_GATE = 0.3


class GeneratorAgent:
    """Prompt assembly, answer generation, and citation extraction.

    Stateless agent: assembles a retrieval-augmented prompt from reranked chunks,
    calls the LLM, and extracts chunk-level citations.

    Also exposes a ``confidence_gate`` method (used as a separate graph node)
    so the gate and generator share the same threshold without a second instance.
    """

    name = "generator"
    description = "Prompt assembly and answer generation with citation extraction"

    def __init__(
        self,
        llm: LLMClient,
        confidence_threshold: float = DEFAULT_CONFIDENCE_GATE,
    ) -> None:
        self._llm = llm
        self._threshold = confidence_threshold

    async def run(self, state: LibrarianState) -> dict[str, Any]:
        reranked = list(state.get("reranked_chunks") or [])

        system, messages = build_prompt(state, reranked)
        response_text = await call_llm(self._llm, system, messages)
        citations = extract_citations(reranked)

        log.info(
            "generation.subgraph.done",
            response_chars=len(response_text),
            citation_count=len(citations),
        )

        return {
            "response": response_text,
            "citations": citations,
        }

    async def run_stream(self, state: LibrarianState) -> AsyncIterator[dict[str, Any]]:
        """Stream generation: yield token chunks, then final metadata.

        Events emitted:
            {"event": "token", "data": "<text chunk>"}
            {"event": "done", "data": {"response": "...", "citations": [...]}}
        """
        reranked = list(state.get("reranked_chunks") or [])
        system, messages = build_prompt(state, reranked)
        citations = extract_citations(reranked)

        full_response: list[str] = []
        async for chunk in self._llm.stream(system, messages):
            full_response.append(chunk)
            yield {"event": "token", "data": chunk}

        response_text = "".join(full_response)
        log.info(
            "generation.subgraph.stream.done",
            response_chars=len(response_text),
            citation_count=len(citations),
        )
        yield {
            "event": "done",
            "data": {"response": response_text, "citations": citations},
        }

    def confidence_gate(self, state: LibrarianState) -> dict[str, Any]:
        """Evaluate confidence_score against the threshold.

        Returns state patch:
            confident=True  → proceed to generation
            confident=False → set fallback_requested=True for CRAG retry
        """
        score: float = state.get("confidence_score", 0.0)
        confident = score >= self._threshold

        log.info(
            "generation.confidence_gate",
            score=score,
            threshold=self._threshold,
            confident=confident,
        )

        return {
            "confident": confident,
            "fallback_requested": not confident,
        }

    def as_node(self) -> Any:
        """Return a LangGraph-compatible async generate node function."""
        async def generate(state: LibrarianState) -> dict[str, Any]:
            return await self.run(state)

        return generate

    def gate_as_node(self) -> Any:
        """Return a LangGraph-compatible sync gate node function."""
        def gate(state: LibrarianState) -> dict[str, Any]:
            result = self.confidence_gate(state)
            # Increment retry_count so the state update is persisted by LangGraph
            if result.get("fallback_requested"):
                result["retry_count"] = int(state.get("retry_count") or 0) + 1
            return result

        return gate
