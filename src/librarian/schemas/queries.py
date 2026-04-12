from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from core.schemas.chunks import Chunk
from core.schemas.retrieval import RetrievalResult  # noqa: F401


class Intent(str, Enum):
    LOOKUP = "lookup"  # find a specific fact or record
    EXPLORE = "explore"  # open-ended investigation across sources
    COMPARE = "compare"  # side-by-side of options or versions
    CONVERSATIONAL = "conversational"  # greetings, clarifications, chitchat
    OUT_OF_SCOPE = "out_of_scope"  # outside the corpus domain


class QueryPlan(BaseModel):
    intent: Intent
    routing: Literal["retrieve", "direct", "clarify"]
    query_variants: list[str]  # multi-query expansion
    needs_clarification: bool
    clarification_question: str | None = None
    retrieval_mode: Literal["dense", "hybrid", "snippet"] = "dense"


__all__ = ["Chunk", "Intent", "QueryPlan", "RetrievalResult"]
