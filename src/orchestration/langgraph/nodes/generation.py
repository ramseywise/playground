from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from core.clients.llm import LLMClient
from librarian.generation.generator import (
    build_prompt,
    call_llm,
    extract_citations,
)
from librarian.generation.prompts import get_system_prompt
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

log = get_logger(__name__)

# Minimum confidence_score required to generate a response.
# Below this threshold the graph should retry retrieval (CRAG loop).
DEFAULT_CONFIDENCE_GATE = 0.3


class GeneratorAgent:
    """Stateless node: build_prompt → call_llm → extract_citations.

    Also exposes ``confidence_gate`` as a separate callable so the supervisor
    can use it as a conditional edge without instantiating a second object.
    """

    name = "generator"
    description = "Builds prompt from context, calls LLM, extracts citations"
    instruction = (
        "You are a precise research assistant. "
        "Answer directly and concisely from the provided sources. "
        "Cite the source URL inline when referencing specific facts. "
        "If the sources do not contain the answer, say so — do not speculate."
    )

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

    def as_node(
        self,
    ) -> Callable[[LibrarianState], Coroutine[Any, Any, dict[str, Any]]]:
        """Return a LangGraph-compatible async node function for generation."""

        async def generate(state: LibrarianState) -> dict[str, Any]:
            return await self.run(state)

        return generate

    def as_gate_node(self) -> Callable[[LibrarianState], dict[str, Any]]:
        """Return a LangGraph-compatible sync node function for the confidence gate."""

        def gate(state: LibrarianState) -> dict[str, Any]:
            result = self.confidence_gate(state)
            if result.get("fallback_requested"):
                result["retry_count"] = int(state.get("retry_count") or 0) + 1
            return result

        return gate


# Backward-compatible alias
GenerationSubgraph = GeneratorAgent
