"""Query routing — select retrieval strategy based on analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from agents.librarian.pipeline.schemas.retrieval import Intent

if TYPE_CHECKING:
    from agents.librarian.pipeline.plan.analyzer import QueryAnalysis


class QueryRouter:
    """Routes a QueryAnalysis to a retrieval strategy."""

    _DIRECT_INTENTS = {Intent.CONVERSATIONAL, Intent.OUT_OF_SCOPE}

    def __init__(self, clarify_confidence_threshold: float = 0.5) -> None:
        self._threshold = clarify_confidence_threshold

    def route(
        self, analysis: QueryAnalysis
    ) -> Literal["retrieve", "direct", "clarify"]:
        if analysis.intent in self._DIRECT_INTENTS:
            return "direct"
        if analysis.confidence < self._threshold:
            return "clarify"
        return "retrieve"


def select_retrieval_mode(
    intent: Intent,
    complexity: Literal["simple", "moderate", "complex"],
) -> Literal["dense", "hybrid", "snippet"]:
    """Choose a retrieval strategy based on intent and query complexity.

    Rules (first match wins):
      - LOOKUP + simple   → snippet  (fast factual lookup)
      - LOOKUP + moderate → hybrid   (mix BM25 + vector)
      - COMPARE           → hybrid   (BM25 finds entities; vector finds context)
      - EXPLORE           → dense    (semantic similarity over broad topic)
      - default           → dense
    """
    if intent == Intent.LOOKUP:
        if complexity == "simple":
            return "snippet"
        return "hybrid"
    if intent == Intent.COMPARE:
        return "hybrid"
    return "dense"
