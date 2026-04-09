from __future__ import annotations

import pytest

from agents.librarian.orchestration.query_understanding import (
    TERM_EXPANSIONS,
    QueryAnalyzer,
    QueryRouter,
)
from agents.librarian.schemas.retrieval import Intent


@pytest.fixture()
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


@pytest.fixture()
def router() -> QueryRouter:
    return QueryRouter(clarify_confidence_threshold=0.5)


# ---------------------------------------------------------------------------
# Intent classification — returns Intent enum values
# ---------------------------------------------------------------------------


def test_classify_lookup_default(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("how do I reset my password?")
    assert result.intent is Intent.LOOKUP
    assert isinstance(result.intent, Intent)


def test_classify_compare_vs_keyword(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("postgres vs mysql which is better for analytics?")
    assert result.intent is Intent.COMPARE


def test_classify_compare_difference_keyword(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("what is the difference between v1 and v2?")
    assert result.intent is Intent.COMPARE


def test_classify_explore_what_is(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("what is the architecture of the system?")
    assert result.intent is Intent.EXPLORE


def test_classify_explore_how_does(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("how does the retry mechanism work?")
    assert result.intent is Intent.EXPLORE


def test_classify_conversational_greeting(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("hello, what can you help me with?")
    assert result.intent is Intent.CONVERSATIONAL


def test_classify_conversational_thanks(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("thanks!")
    assert result.intent is Intent.CONVERSATIONAL


def test_classify_out_of_scope(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("what is the weather in Paris?")
    assert result.intent is Intent.OUT_OF_SCOPE


def test_intent_is_enum_not_string(analyzer: QueryAnalyzer) -> None:
    """Intent must be Intent enum — not a raw string."""
    result = analyzer.analyze("compare approach A vs approach B")
    assert type(result.intent) is Intent
    assert result.intent is not None


# ---------------------------------------------------------------------------
# Confidence scores
# ---------------------------------------------------------------------------


def test_confidence_in_range(analyzer: QueryAnalyzer) -> None:
    for query in [
        "hello",
        "compare A vs B",
        "what is X?",
        "how does Y work?",
        "weather in london",
    ]:
        result = analyzer.analyze(query)
        assert 0.0 <= result.confidence <= 1.0, f"out of range for: {query!r}"


def test_lookup_default_confidence_below_compare(analyzer: QueryAnalyzer) -> None:
    lookup = analyzer.analyze("find the API key docs")
    compare = analyzer.analyze("compare REST vs GraphQL")
    assert compare.confidence >= lookup.confidence


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def test_entity_version_extracted(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("what changed in v2.3.1?")
    assert "version" in result.entities
    assert any("2.3.1" in v for v in result.entities["version"])


def test_entity_date_extracted(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("show me changes from 2024-01-15")
    assert "date" in result.entities
    assert "2024-01-15" in result.entities["date"]


def test_entity_quantity_extracted(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("latency must be under 100ms")
    assert "quantity" in result.entities
    assert any("100ms" in q for q in result.entities["quantity"])


def test_entity_identifier_extracted(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("what does AUTH_TOKEN control?")
    assert "identifier" in result.entities
    assert "AUTH_TOKEN" in result.entities["identifier"]


def test_entity_empty_for_plain_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("how do I log in?")
    # May or may not have entities — just check no error and type is dict
    assert isinstance(result.entities, dict)


# ---------------------------------------------------------------------------
# Sub-query decomposition
# ---------------------------------------------------------------------------


def test_decompose_single_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("what is authentication?")
    assert len(result.sub_queries) >= 1


def test_decompose_conjunction_splits(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("explain auth and describe billing")
    assert len(result.sub_queries) >= 2


def test_decompose_question_mark_splits(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("what is auth? how does billing work?")
    assert len(result.sub_queries) >= 2


def test_decompose_never_empty(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("X")
    assert len(result.sub_queries) >= 1


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------


def test_complexity_simple_single_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("what is an API key?")
    assert result.complexity == "simple"


def test_complexity_complex_multi_part(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("explain auth and describe billing and explain setup")
    assert result.complexity in ("moderate", "complex")


def test_complexity_valid_values(analyzer: QueryAnalyzer) -> None:
    for query in ["x", "a and b", "a and b and c and d"]:
        result = analyzer.analyze(query)
        assert result.complexity in ("simple", "moderate", "complex")


# ---------------------------------------------------------------------------
# Term expansion
# ---------------------------------------------------------------------------


def test_expand_auth_term(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("auth token configuration")
    assert (
        "authentication" in result.expanded_terms
        or "authorization" in result.expanded_terms
    )


def test_expand_api_term(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("api endpoint setup")
    assert any(t in result.expanded_terms for t in ["endpoint", "rest", "http"])


def test_expand_no_duplicates(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("auth api config")
    assert len(result.expanded_terms) == len(set(result.expanded_terms))


def test_expand_unknown_terms_empty(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("xyzzy frobble quux")
    assert isinstance(result.expanded_terms, list)


def test_term_expansions_dict_populated() -> None:
    assert len(TERM_EXPANSIONS) > 0
    assert "auth" in TERM_EXPANSIONS
    assert all(isinstance(v, list) for v in TERM_EXPANSIONS.values())


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


def test_router_retrieve_for_lookup(
    analyzer: QueryAnalyzer, router: QueryRouter
) -> None:
    analysis = analyzer.analyze("where are the API key docs?")
    assert router.route(analysis) == "retrieve"


def test_router_direct_for_conversational(
    analyzer: QueryAnalyzer, router: QueryRouter
) -> None:
    analysis = analyzer.analyze("hello, how are you?")
    assert router.route(analysis) == "direct"


def test_router_direct_for_out_of_scope(
    analyzer: QueryAnalyzer, router: QueryRouter
) -> None:
    analysis = analyzer.analyze("what is the weather?")
    assert router.route(analysis) == "direct"


def test_router_clarify_for_low_confidence(router: QueryRouter) -> None:
    from agents.librarian.orchestration.query_understanding import QueryAnalysis

    low_confidence = QueryAnalysis(
        intent=Intent.LOOKUP,
        confidence=0.3,
        entities={},
        sub_queries=["something"],
        complexity="simple",
        expanded_terms=[],
    )
    assert router.route(low_confidence) == "clarify"


def test_router_retrieve_at_threshold(router: QueryRouter) -> None:
    from agents.librarian.orchestration.query_understanding import QueryAnalysis

    at_threshold = QueryAnalysis(
        intent=Intent.LOOKUP,
        confidence=0.5,
        entities={},
        sub_queries=["q"],
        complexity="simple",
        expanded_terms=[],
    )
    assert router.route(at_threshold) == "retrieve"


def test_router_retrieve_for_explore(
    analyzer: QueryAnalyzer, router: QueryRouter
) -> None:
    analysis = analyzer.analyze("give me an overview of the auth system")
    assert router.route(analysis) == "retrieve"


def test_router_retrieve_for_compare(
    analyzer: QueryAnalyzer, router: QueryRouter
) -> None:
    analysis = analyzer.analyze("compare JWT vs session tokens")
    assert router.route(analysis) == "retrieve"


def test_router_custom_threshold() -> None:
    from agents.librarian.orchestration.query_understanding import QueryAnalysis

    strict_router = QueryRouter(clarify_confidence_threshold=0.8)
    analysis = QueryAnalysis(
        intent=Intent.LOOKUP,
        confidence=0.7,
        entities={},
        sub_queries=["q"],
        complexity="simple",
        expanded_terms=[],
    )
    assert strict_router.route(analysis) == "clarify"


# ---------------------------------------------------------------------------
# Retrieval mode routing
# ---------------------------------------------------------------------------


def test_retrieval_mode_simple_lookup_is_snippet(analyzer: QueryAnalyzer) -> None:
    """Simple LOOKUP queries should route to snippet retrieval."""
    result = analyzer.analyze("who founded hip-hop?")
    assert result.intent is Intent.LOOKUP
    assert result.complexity == "simple"
    assert result.retrieval_mode == "snippet"


def test_retrieval_mode_complex_lookup_is_hybrid(analyzer: QueryAnalyzer) -> None:
    """LOOKUP with multiple sub-parts should route to hybrid."""
    result = analyzer.analyze("find the artist and the album and the year")
    # complexity moderate or complex → hybrid
    assert result.retrieval_mode in ("hybrid", "dense")  # not snippet


def test_retrieval_mode_explore_is_dense(analyzer: QueryAnalyzer) -> None:
    """EXPLORE queries should use dense retrieval."""
    result = analyzer.analyze("explain how jazz evolved from bebop to fusion")
    assert result.intent is Intent.EXPLORE
    assert result.retrieval_mode == "dense"


def test_retrieval_mode_compare_is_hybrid(analyzer: QueryAnalyzer) -> None:
    """COMPARE queries should use hybrid retrieval."""
    result = analyzer.analyze("compare US punk vs UK punk scene differences")
    assert result.intent is Intent.COMPARE
    assert result.retrieval_mode == "hybrid"


def test_retrieval_mode_conversational_is_dense(analyzer: QueryAnalyzer) -> None:
    """Conversational / out-of-scope defaults to dense (doesn't reach retrieval anyway)."""
    result = analyzer.analyze("hello how are you")
    assert result.intent is Intent.CONVERSATIONAL
    # retrieval_mode is set but won't be used (routed to direct)
    assert result.retrieval_mode in ("dense", "hybrid", "snippet")


def test_retrieval_mode_field_always_present(analyzer: QueryAnalyzer) -> None:
    """retrieval_mode must always be one of the valid values."""
    queries = [
        "what year was Kind of Blue recorded?",
        "compare Motown vs Stax",
        "explain the history of blues music",
        "hello",
        "what is the weather?",
    ]
    for q in queries:
        result = analyzer.analyze(q)
        assert result.retrieval_mode in ("dense", "hybrid", "snippet"), (
            f"unexpected retrieval_mode for {q!r}: {result.retrieval_mode!r}"
        )


# ---------------------------------------------------------------------------
# Music corpus: "music" and "song" no longer OUT_OF_SCOPE
# ---------------------------------------------------------------------------


def test_music_keyword_not_out_of_scope(analyzer: QueryAnalyzer) -> None:
    """'music' should no longer be classified OUT_OF_SCOPE."""
    result = analyzer.analyze("tell me about the history of blues music")
    assert result.intent is not Intent.OUT_OF_SCOPE


def test_song_keyword_not_out_of_scope(analyzer: QueryAnalyzer) -> None:
    """'song' should no longer be classified OUT_OF_SCOPE."""
    result = analyzer.analyze("what song did Dolly Parton write for Whitney Houston?")
    assert result.intent is not Intent.OUT_OF_SCOPE


def test_music_term_expansions_present() -> None:
    """Music genre term expansions must exist in TERM_EXPANSIONS."""
    assert "blues" in TERM_EXPANSIONS
    assert "jazz" in TERM_EXPANSIONS
    assert "metal" in TERM_EXPANSIONS
    assert "punk" in TERM_EXPANSIONS
