from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents.librarian.schemas.chunks import GradedChunk, RankedChunk
from agents.librarian.schemas.retrieval import QueryPlan, RetrievalResult


class LibrarianState(TypedDict, total=False):
    # Core
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    standalone_query: str
    trace_id: str

    # Planning output
    intent: str
    retrieval_mode: str  # "dense" | "hybrid" | "snippet"
    plan: QueryPlan
    skip_retrieval: bool

    # Retrieval output
    query_variants: list[str]
    retrieved_chunks: list[RetrievalResult]
    graded_chunks: list[GradedChunk]
    retry_count: int  # CRAG loop counter

    # Reranker output
    reranked_chunks: list[RankedChunk]
    confidence_score: float  # max relevance_score from reranker

    # Generation output
    response: str
    citations: list[dict]  # [{"url": ..., "title": ...}]
    confident: bool  # confidence_gate result
    fallback_requested: bool  # set True when confidence_gate fires
