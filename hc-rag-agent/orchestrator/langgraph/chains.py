"""LangGraph chain factories — LLM chains wired to prompts and structured outputs."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from clients.llm import get_chat_model, get_planner_chat_model


@lru_cache(maxsize=1)
def get_planner_agent() -> Any:
    from orchestrator.langgraph.prompts import PLANNER_PROMPT
    from orchestrator.langgraph.schemas import PlannerOutput

    return PLANNER_PROMPT | get_planner_chat_model().with_structured_output(
        PlannerOutput
    )


@lru_cache(maxsize=1)
def get_clarify_chain() -> Any:
    from orchestrator.langgraph.prompts import CLARIFY_PROMPT
    from orchestrator.langgraph.schemas import ClarifyOutput

    return CLARIFY_PROMPT | get_chat_model().with_structured_output(ClarifyOutput)


@lru_cache(maxsize=1)
def get_scheduler_chain() -> Any:
    from orchestrator.langgraph.prompts import SCHEDULER_PROMPT
    from orchestrator.langgraph.schemas import SchedulerOutput

    return SCHEDULER_PROMPT | get_chat_model().with_structured_output(SchedulerOutput)


@lru_cache(maxsize=1)
def get_answer_chain() -> Any:
    from orchestrator.langgraph.prompts import ANSWER_PROMPT

    return ANSWER_PROMPT | get_chat_model()


@lru_cache(maxsize=1)
def get_retrieval_query_transform_chain() -> Any:
    from orchestrator.langgraph.prompts import RETRIEVAL_QUERY_TRANSFORM_PROMPT
    from orchestrator.langgraph.schemas import RetrievalQueryTransformOutput

    return RETRIEVAL_QUERY_TRANSFORM_PROMPT | get_chat_model().with_structured_output(
        RetrievalQueryTransformOutput
    )


@lru_cache(maxsize=1)
def get_hybrid_retrieval_probe_chain() -> Any:
    from orchestrator.langgraph.prompts import HYBRID_RETRIEVAL_PROBE_PROMPT
    from orchestrator.langgraph.schemas import HybridRetrievalProbeOutput

    return (
        HYBRID_RETRIEVAL_PROBE_PROMPT
        | get_planner_chat_model().with_structured_output(HybridRetrievalProbeOutput)
    )


@lru_cache(maxsize=1)
def get_hybrid_rerank_probe_chain() -> Any:
    from orchestrator.langgraph.prompts import HYBRID_RERANK_PROBE_PROMPT
    from orchestrator.langgraph.schemas import HybridRerankProbeOutput

    return HYBRID_RERANK_PROBE_PROMPT | get_planner_chat_model().with_structured_output(
        HybridRerankProbeOutput
    )


@lru_cache(maxsize=1)
def get_summarizer_chain() -> Any:
    """Cheap model chain for conversation history summarization."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the following conversation concisely, preserving key facts, "
                "decisions made, and any outstanding questions. Write in third person. "
                "Maximum 200 words.",
            ),
            ("human", "{transcript}"),
        ]
    )
    return prompt | get_planner_chat_model()


@lru_cache(maxsize=1)
def get_post_answer_eval_chain() -> Any:
    from orchestrator.langgraph.prompts import POST_ANSWER_EVAL_PROMPT
    from orchestrator.langgraph.schemas import PostAnswerEvalOutput

    return POST_ANSWER_EVAL_PROMPT | get_chat_model().with_structured_output(
        PostAnswerEvalOutput
    )
