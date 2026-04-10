from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from agents.librarian.schemas.retrieval import Intent
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Term expansion dictionary
# ---------------------------------------------------------------------------

TERM_EXPANSIONS: dict[str, list[str]] = {
    "auth": ["authentication", "authorization", "login", "token", "oauth"],
    "authn": ["authentication", "identity", "credential"],
    "authz": ["authorization", "permission", "access control", "rbac"],
    "api": ["endpoint", "rest", "http", "interface"],
    "config": ["configuration", "settings", "environment", "setup"],
    "deploy": ["deployment", "release", "rollout", "infrastructure"],
    "db": ["database", "storage", "persistence", "sql"],
    "perf": ["performance", "latency", "throughput", "benchmark"],
    "error": ["exception", "failure", "bug", "issue", "traceback"],
    "install": ["installation", "setup", "dependency", "package"],
    "k8s": ["kubernetes", "cluster", "pod", "container", "helm"],
    "ml": ["machine learning", "model", "training", "inference"],
    "llm": ["large language model", "gpt", "claude", "inference", "prompt"],
    # Music genre terms
    "hip-hop": ["rap", "hip hop", "emcee", "mc", "dj", "breakbeat"],
    "hiphop": ["rap", "hip hop", "emcee", "mc", "dj", "breakbeat"],
    "blues": ["delta blues", "chicago blues", "electric blues", "twelve-bar"],
    "jazz": ["bebop", "swing", "fusion", "improvisation", "big band"],
    "metal": ["heavy metal", "doom", "thrash", "black metal", "riff"],
    "punk": ["hardcore", "new wave", "post-punk", "diy", "three-chord"],
    "soul": ["rhythm and blues", "rnb", "gospel", "motown", "stax"],
    "electronic": ["synthesizer", "drum machine", "techno", "house", "edm"],
    "country": ["honky-tonk", "nashville", "bluegrass", "outlaw country"],
}

# ---------------------------------------------------------------------------
# Keyword → intent rules  (order matters — first match wins)
# ---------------------------------------------------------------------------

_INTENT_RULES: list[tuple[list[str], Intent]] = [
    (
        [
            "compare",
            "vs",
            "versus",
            "difference",
            "differences",
            "differ",
            "contrast",
            "which is better",
            "tradeoff",
            "pros and cons",
        ],
        Intent.COMPARE,
    ),
    (
        [
            "hello",
            "hi",
            "hey",
            "thanks",
            "thank you",
            "help me",
            "what can you do",
            "who are you",
            "good morning",
            "good afternoon",
        ],
        Intent.CONVERSATIONAL,
    ),
    (
        [
            "weather",
            "stock price",
            "sports score",
            "recipe",
            "movie",
        ],
        Intent.OUT_OF_SCOPE,
    ),
    (
        [
            "overview",
            "explore",
            "survey",
            "summarize",
            "summary",
            "explain",
            "how does",
            "what is",
            "walk me through",
            "tell me about",
        ],
        Intent.EXPLORE,
    ),
]

# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

_ENTITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("version", re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")),
    (
        "date",
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b",
            re.I,
        ),
    ),
    (
        "quantity",
        re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|kb|mb|gb|tb|req|rpm|qps|%)\b", re.I),
    ),
    ("identifier", re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")),
]

# ---------------------------------------------------------------------------
# Complexity scoring thresholds
# ---------------------------------------------------------------------------

_COMPLEX_THRESHOLD = 3  # sub-query count ≥ this → complex
_MODERATE_THRESHOLD = 2  # sub-query count ≥ this → moderate


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QueryAnalysis:
    intent: Intent
    confidence: float  # 0.0–1.0
    entities: dict[str, list[str]]  # type → matched strings
    sub_queries: list[str]
    complexity: Literal["simple", "moderate", "complex"]
    expanded_terms: list[str]
    retrieval_mode: Literal["dense", "hybrid", "snippet"] = "dense"


# ---------------------------------------------------------------------------
# QueryAnalyzer
# ---------------------------------------------------------------------------


class QueryAnalyzer:
    """Rule-based query analysis: intent classification, entity extraction,
    sub-query decomposition, complexity scoring, and term expansion."""

    def analyze(self, query: str) -> QueryAnalysis:
        query_lower = query.lower().strip()

        intent, confidence = self._classify_intent(query_lower)
        entities = self._extract_entities(query)
        sub_queries = self._decompose(query)
        complexity = self._score_complexity(sub_queries, entities)
        expanded_terms = self._expand_terms(query_lower)

        retrieval_mode = self._select_retrieval_mode(intent, complexity)

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
            expanded_terms=expanded_terms,
            retrieval_mode=retrieval_mode,
        )

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, query_lower: str) -> tuple[Intent, float]:
        for keywords, intent in _INTENT_RULES:
            for kw in keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, query_lower):
                    return intent, 0.9

        # Default: lookup with moderate confidence
        return Intent.LOOKUP, 0.6

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, query: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for label, pattern in _ENTITY_PATTERNS:
            matches = pattern.findall(query)
            if matches:
                result[label] = list(
                    dict.fromkeys(matches)
                )  # deduplicate, preserve order
        return result

    # ------------------------------------------------------------------
    # Sub-query decomposition
    # ------------------------------------------------------------------

    _CONJUNCTION_SPLIT = re.compile(
        r"\b(?:and|also|additionally|furthermore|moreover|plus)\b",
        re.I,
    )
    _QUESTION_SPLIT = re.compile(r"\?+")

    def _decompose(self, query: str) -> list[str]:
        # Split on question marks first
        parts = [p.strip() for p in self._QUESTION_SPLIT.split(query) if p.strip()]

        # Then split each part on conjunctions
        sub_queries: list[str] = []
        for part in parts:
            splits = [
                s.strip() for s in self._CONJUNCTION_SPLIT.split(part) if s.strip()
            ]
            sub_queries.extend(splits)

        # Always include the original query as a fallback
        if not sub_queries:
            return [query.strip()]

        return sub_queries

    # ------------------------------------------------------------------
    # Complexity scoring
    # ------------------------------------------------------------------

    def _score_complexity(
        self,
        sub_queries: list[str],
        entities: dict[str, list[str]],
    ) -> Literal["simple", "moderate", "complex"]:
        n = len(sub_queries)
        # Multiple entity types add complexity
        entity_bonus = 1 if len(entities) >= 2 else 0
        effective = n + entity_bonus

        if effective >= _COMPLEX_THRESHOLD:
            return "complex"
        if effective >= _MODERATE_THRESHOLD:
            return "moderate"
        return "simple"

    # ------------------------------------------------------------------
    # Retrieval mode selection
    # ------------------------------------------------------------------

    def _select_retrieval_mode(
        self,
        intent: Intent,
        complexity: Literal["simple", "moderate", "complex"],
    ) -> Literal["dense", "hybrid", "snippet"]:
        """Choose a retrieval strategy based on intent and query complexity.

        Rules (first match wins):
          - LOOKUP + simple   → snippet  (fast factual lookup from snippet DB)
          - LOOKUP + moderate → hybrid   (mix BM25 + vector for specific terms)
          - COMPARE           → hybrid   (BM25 finds named entities; vector finds context)
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

    # ------------------------------------------------------------------
    # Term expansion
    # ------------------------------------------------------------------

    def _expand_terms(self, query_lower: str) -> list[str]:
        words = re.findall(r"\w+", query_lower)
        expansions: list[str] = []
        seen: set[str] = set(words)
        for word in words:
            for term in TERM_EXPANSIONS.get(word, []):
                if term not in seen:
                    expansions.append(term)
                    seen.add(term)
        return expansions


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


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
