"""QueryAnalyzer — composes intent, entities, decomposition, expansion, routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agents.librarian.analysis.decomposition import decompose_query
from agents.librarian.analysis.entities import extract_entities
from agents.librarian.analysis.expansion import expand_terms
from agents.librarian.analysis.intent import classify_intent
from agents.librarian.analysis.routing import select_retrieval_mode
from agents.librarian.schemas.retrieval import Intent
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

# Complexity scoring thresholds
_COMPLEX_THRESHOLD = 3
_MODERATE_THRESHOLD = 2


@dataclass
class QueryAnalysis:
    intent: Intent
    confidence: float  # 0.0–1.0
    entities: dict[str, list[str]]  # type → matched strings
    sub_queries: list[str]
    complexity: Literal["simple", "moderate", "complex"]
    expanded_terms: list[str]
    retrieval_mode: Literal["dense", "hybrid", "snippet"] = "dense"


class QueryAnalyzer:
    """Rule-based query analysis: intent classification, entity extraction,
    sub-query decomposition, complexity scoring, and term expansion."""

    def analyze(self, query: str) -> QueryAnalysis:
        query_lower = query.lower().strip()

        intent, confidence = classify_intent(query_lower)
        entities = extract_entities(query)
        sub_queries = decompose_query(query)
        complexity = self._score_complexity(sub_queries, entities)
        expanded = expand_terms(query_lower)
        retrieval_mode = select_retrieval_mode(intent, complexity)

        log.debug(
            "query_understanding.analyze",
            intent=intent.value,
            confidence=confidence,
            complexity=complexity,
            retrieval_mode=retrieval_mode,
            sub_query_count=len(sub_queries),
        )
        return QueryAnalysis(
            intent=intent,
            confidence=confidence,
            entities=entities,
            sub_queries=sub_queries,
            complexity=complexity,
            expanded_terms=expanded,
            retrieval_mode=retrieval_mode,
        )

    def _score_complexity(
        self,
        sub_queries: list[str],
        entities: dict[str, list[str]],
    ) -> Literal["simple", "moderate", "complex"]:
        n = len(sub_queries)
        entity_bonus = 1 if len(entities) >= 2 else 0
        effective = n + entity_bonus

        if effective >= _COMPLEX_THRESHOLD:
            return "complex"
        if effective >= _MODERATE_THRESHOLD:
            return "moderate"
        return "simple"
