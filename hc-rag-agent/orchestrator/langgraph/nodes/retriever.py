"""Candidate retrieval node + query expansion (LangGraph id: ``retriever``)."""

from __future__ import annotations

import logging
import time
from typing import Final

from core.config import RAG_ENSEMBLE_TOP_K, RAG_RETRIEVAL_QUERY_TRANSFORM
from orchestrator.langgraph.schemas import format_graded_context, locale_to_language
from orchestrator.langgraph.schemas.state import GraphState, _QA_CYCLE_RESET, qa_cycle_field_reset
from orchestrator.langgraph.utils import run_coro
from clients import llm as llm_mod
from rag.retrieval.pipeline import retrieve_graded_chunks

log = logging.getLogger(__name__)

_MAX_QUERIES: Final[int] = 3


def _dedupe_trim(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in queries:
        t = raw.strip()
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= _MAX_QUERIES:
            break
    return out


def expand_queries_for_retrieval(
    user_query: str, locale: str | None = None
) -> tuple[list[str], float]:
    """Return locale-aware search queries for the ensemble retriever and transform latency (ms).

    Generates queries in the language matching ``locale`` (e.g. ``"da"`` → Danish,
    ``"fr"`` → French). Falls back to English when ``locale`` is unrecognised.

    When ``RAG_RETRIEVAL_QUERY_TRANSFORM`` is false or the LLM is unavailable,
    returns ``([user_query], 0.0)``. On LLM failure, falls back to the original
    query (single-query retrieval).
    """
    q = user_query.strip()
    if not q:
        return [], 0.0

    if not RAG_RETRIEVAL_QUERY_TRANSFORM:
        return [q], 0.0

    if not llm_mod.llm_configured():
        log.info("retrieval_query_transform.skip reason=llm_not_configured")
        return [q], 0.0

    target_language = (
        locale_to_language(locale)
        if locale
        else "the same language as the user's question"
    )
    t0 = time.perf_counter()
    try:
        from orchestrator.langgraph.chains import (
            get_retrieval_query_transform_chain,
        )

        chain = get_retrieval_query_transform_chain()
        out = chain.invoke({"query": q, "target_language": target_language})
        candidates = _dedupe_trim(list(out.queries))
        if len(candidates) < 2:
            log.warning(
                "retrieval_query_transform.insufficient_queries count=%d",
                len(candidates),
            )
            return [q], (time.perf_counter() - t0) * 1000
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "retrieval_query_transform.done locale=%s language=%s queries=%d elapsed_ms=%.1f",
            locale,
            target_language,
            len(candidates),
            elapsed_ms,
        )
        return candidates, elapsed_ms
    except Exception:
        log.exception("retrieval_query_transform.failed")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return [q], elapsed_ms


def retriever_node(state: GraphState) -> dict:
    log.info(
        "retriever: start stage=candidate_retrieval query_len=%d",
        len(state.query or ""),
    )
    t0 = time.perf_counter()

    query = state.query
    if not query:
        return {
            **_QA_CYCLE_RESET,
            "error": "Missing query for retrieval",
            "graded_chunks": [],
            "retrieved_context": None,
            "retrieval_queries": None,
            "latency_ms": {
                **state.latency_ms,
                "query_transform_ms": 0.0,
                "retrieval_ms": 0.0,
            },
        }

    retrieval_queries, qtf_ms = expand_queries_for_retrieval(query, locale=state.locale)

    try:
        graded = run_coro(retrieve_graded_chunks(retrieval_queries, RAG_ENSEMBLE_TOP_K))
    except Exception:
        log.exception("retriever: ensemble failed")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            **_QA_CYCLE_RESET,
            "error": "Retrieval failed (see logs)",
            "graded_chunks": [],
            "retrieved_context": None,
            "retrieval_queries": retrieval_queries,
            "latency_ms": {
                **state.latency_ms,
                "query_transform_ms": qtf_ms,
                "retrieval_ms": elapsed_ms,
            },
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    ctx = format_graded_context(graded)
    log.info(
        "retriever: done stage=candidate_retrieval graded_chunks=%d elapsed_ms=%.1f",
        len(graded),
        elapsed_ms,
    )
    return {
        **_QA_CYCLE_RESET,
        "graded_chunks": graded,
        "retrieved_context": ctx,
        "retrieval_queries": retrieval_queries,
        "error": None,
        "latency_ms": {
            **state.latency_ms,
            "query_transform_ms": qtf_ms,
            "retrieval_ms": elapsed_ms,
        },
    }


__all__ = [
    "expand_queries_for_retrieval",
    "qa_cycle_field_reset",
    "retriever_node",
]
