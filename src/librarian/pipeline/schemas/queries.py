from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from agents.librarian.pipeline.schemas.chunks import Chunk


class Intent(str, Enum):
    LOOKUP = "lookup"  # find a specific fact or record
    EXPLORE = "explore"  # open-ended investigation across sources
    COMPARE = "compare"  # side-by-side of options or versions
    CONVERSATIONAL = "conversational"  # greetings, clarifications, chitchat
    OUT_OF_SCOPE = "out_of_scope"  # outside the corpus domain


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    source: Literal["vector", "bm25", "hybrid"]


class QueryPlan(BaseModel):
    intent: Intent
    routing: Literal["retrieve", "direct", "clarify"]
    query_variants: list[str]  # multi-query expansion
    needs_clarification: bool
    clarification_question: str | None = None
    retrieval_mode: Literal["dense", "hybrid", "snippet"] = "dense"
