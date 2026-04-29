"""Answer node (LangGraph id: ``answer``)."""

from __future__ import annotations

import logging
import time

from core.config import RAG_ANSWER_CONTEXT_MAX_CHUNKS, RAG_ANSWER_CONTEXT_MAX_TOKENS
from orchestrator.langgraph.schemas import (
    citations_from_ranked_ordered,
    locale_to_language,
)
from orchestrator.langgraph.schemas.context_builder import (
    ContextBuildConfig,
    build_context_from_ranked,
)
from orchestrator.langgraph.schemas.state import GraphState
from orchestrator.langgraph.utils import with_total_ms
from orchestrator.langgraph.chains import get_answer_chain

log = logging.getLogger(__name__)


async def answer_node(state: GraphState) -> dict:
    response_language = locale_to_language(state.locale)
    log.info(
        "answer: start mode=%s query_len=%d locale=%s",
        state.mode,
        len(state.query or ""),
        state.locale,
    )
    t0 = time.perf_counter()

    if state.mode == "q&a":
        if state.reranked_chunks:
            built = build_context_from_ranked(
                state.reranked_chunks,
                ContextBuildConfig(
                    max_tokens=RAG_ANSWER_CONTEXT_MAX_TOKENS,
                    max_chunks=RAG_ANSWER_CONTEXT_MAX_CHUNKS,
                ),
            )
            context_block = built.text
            if built.truncated:
                log.info(
                    "answer: context_truncated tokens=%d chunk_ids=%d",
                    built.tokens_used,
                    len(built.chunk_ids_in_order),
                )
            citations = citations_from_ranked_ordered(
                state.reranked_chunks,
                built.chunk_ids_in_order,
            )
        else:
            context_block = state.retrieved_context or "No relevant documents found."
            citations = []
        input_text = (
            f"\nUser question:\n{state.query}\n\nRetrieved context:\n{context_block}\n"
        )

    elif state.mode == "task_execution":
        steps_text = "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(state.action_steps)
        )
        input_text = (
            f"\nUser request:\n{state.query}\n\nConfirmed action plan:\n{steps_text}\n"
        )
        citations = []

    else:
        return {"final_answer": "I'm not sure how to handle this request."}

    final_answer = await get_answer_chain().ainvoke(
        {"input": input_text, "response_language": response_language}
    )
    llm_ms = (time.perf_counter() - t0) * 1000
    lat = with_total_ms({**state.latency_ms, "llm_ms": llm_ms})
    log.info(
        "answer: done elapsed_ms=%.1f total_ms=%.1f", llm_ms, lat.get("total_ms", 0.0)
    )
    out: dict = {
        "final_answer": final_answer,
        "latency_ms": lat,
        "qa_outcome": "answer",
    }
    if state.mode == "q&a":
        out["citations"] = citations
    return out


__all__ = ["answer_node"]
