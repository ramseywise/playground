from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag_system.src.rag_core.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QueryAnalysis:
    """Result of query analysis."""

    original_query: str
    intent: str  # factual | procedural | exploratory | troubleshooting
    complexity: str  # simple | moderate | complex
    expanded_query: str
    entities: dict[str, list[str]]
    metadata_filters: dict[str, Any]
    sub_queries: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# Intent classification — keyword-based, English defaults
# ---------------------------------------------------------------------------

INTENT_PATTERNS: dict[str, dict[str, list[str]]] = {
    "factual": {
        "keywords": [
            "what is",
            "how much",
            "when",
            "where",
            "which",
            "can i",
            "is it possible",
            "cost",
            "price",
        ],
        "indicators": ["?"],
    },
    "procedural": {
        "keywords": [
            "how to",
            "how do i",
            "guide",
            "steps",
            "set up",
            "configure",
            "create",
            "export",
            "import",
        ],
        "indicators": ["step", "tutorial"],
    },
    "exploratory": {
        "keywords": [
            "why",
            "explain",
            "difference",
            "compare",
            "means",
            "don't understand",
            "relationship",
        ],
        "indicators": ["general", "overview"],
    },
    "troubleshooting": {
        "keywords": [
            "error",
            "problem",
            "not working",
            "doesn't work",
            "bug",
            "crash",
            "fails",
            "broken",
        ],
        "indicators": ["help", "urgent"],
    },
}


def classify_intent(query: str) -> dict[str, Any]:
    """Classify query intent using keyword matching."""
    query_lower = query.lower()
    scores: dict[str, dict[str, Any]] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        matches: list[str] = []
        for keyword in patterns["keywords"]:
            if keyword in query_lower:
                score += 1
                matches.append(keyword)
        for indicator in patterns["indicators"]:
            if indicator in query_lower:
                score += 0.5
        scores[intent] = {"score": score, "matches": matches}

    if not any(s["score"] > 0 for s in scores.values()):
        return {"intent": "factual", "confidence": 0.3, "matches": []}

    best_intent = max(scores.keys(), key=lambda k: scores[k]["score"])
    confidence = min(1.0, scores[best_intent]["score"] / 3)
    return {
        "intent": best_intent,
        "confidence": confidence,
        "matches": scores[best_intent]["matches"],
    }


# ---------------------------------------------------------------------------
# Query expansion — empty by default; consuming project can extend
# ---------------------------------------------------------------------------

TERM_EXPANSIONS: dict[str, list[str]] = {}


def expand_query(
    query: str, expansions: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    """Expand query with domain-specific terminology.

    Pass a custom ``expansions`` dict from the consuming project to add
    domain-specific term expansions without modifying rag_core.
    """
    mapping = expansions or TERM_EXPANSIONS
    query_lower = query.lower()
    expansions_applied: list[str] = []
    expanded_terms: list[str] = []

    for term, exps in mapping.items():
        if term in query_lower:
            expansions_applied.append(term)
            expanded_terms.extend(exps[:2])

    if expanded_terms:
        expanded_query = query + " " + " ".join(set(expanded_terms))
    else:
        expanded_query = query

    return {
        "original": query,
        "expanded": expanded_query,
        "terms_expanded": expansions_applied,
        "added_terms": list(set(expanded_terms)),
    }


# ---------------------------------------------------------------------------
# Entity extraction — generic patterns; consuming project can extend
# ---------------------------------------------------------------------------


def extract_entities(query: str) -> dict[str, list[str]]:
    """Extract entities for metadata filtering (generic patterns)."""
    entities: dict[str, list[str]] = {}

    dates = re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", query)
    if dates:
        entities["dates"] = dates

    amounts = re.findall(r"(\d+(?:[.,]\d{2})?)\s*(?:\$|€|£|USD|EUR|GBP)", query)
    if amounts:
        entities["amounts"] = amounts

    return entities


def generate_metadata_filters(
    intent: str, entities: dict[str, list[str]], query: str
) -> dict[str, Any]:
    """Generate metadata filters based on intent and entities."""
    filters: dict[str, Any] = {}

    if intent == "procedural":
        filters["source_types"] = ["help_center", "tutorial", "faq"]
    elif intent == "troubleshooting":
        filters["source_types"] = ["help_center", "faq", "community"]
    elif intent == "factual":
        filters["source_types"] = ["faq", "help_center"]

    return filters


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------


def decompose_query(query: str) -> list[str]:
    """Decompose complex queries into sub-queries on conjunction boundaries."""
    sub_queries: list[str] = []

    if " and " in query.lower() and ("how" in query.lower() or "what" in query.lower()):
        parts = re.split(r"\s+and\s+", query, flags=re.IGNORECASE)
        sub_queries.extend([p.strip() for p in parts if len(p.strip()) > 20])

    if query.count("?") > 1:
        parts = query.split("?")
        sub_queries.extend([p.strip() + "?" for p in parts if len(p.strip()) > 15])

    return sub_queries[:3] if sub_queries else [query]


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------


def score_complexity(query: str, entities: dict, sub_queries: list[str]) -> str:
    """Return complexity as 'simple', 'moderate', or 'complex'."""
    score = 0

    if len(query) > 200:
        score += 2
    elif len(query) > 100:
        score += 1

    if len(sub_queries) > 1:
        score += 2

    if len(entities) > 2:
        score += 1

    complex_terms = ["relationship", "difference", "compare", "when should", "best way"]
    if any(term in query.lower() for term in complex_terms):
        score += 2

    if score >= 4:
        return "complex"
    if score >= 2:
        return "moderate"
    return "simple"


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class QueryAnalyzer:
    """Full query analysis: intent, expansion, entity extraction, decomposition."""

    def __init__(
        self,
        expand_terms: bool = True,
        extract_entities_flag: bool = True,
        decompose: bool = True,
        term_expansions: dict[str, list[str]] | None = None,
    ) -> None:
        self.expand_terms = expand_terms
        self.extract_entities_flag = extract_entities_flag
        self.decompose = decompose
        self.term_expansions = term_expansions

    def analyze(self, query: str) -> QueryAnalysis:
        intent_result = classify_intent(query)

        expanded_query = query
        if self.expand_terms:
            expansion = expand_query(query, self.term_expansions)
            expanded_query = expansion["expanded"]

        entities: dict[str, list[str]] = {}
        if self.extract_entities_flag:
            entities = extract_entities(query)

        metadata_filters = generate_metadata_filters(
            intent_result["intent"], entities, query
        )

        sub_queries = decompose_query(query) if self.decompose else [query]
        complexity = score_complexity(query, entities, sub_queries)

        return QueryAnalysis(
            original_query=query,
            intent=intent_result["intent"],
            complexity=complexity,
            expanded_query=expanded_query,
            entities=entities,
            metadata_filters=metadata_filters,
            sub_queries=sub_queries,
            confidence=intent_result["confidence"],
        )
