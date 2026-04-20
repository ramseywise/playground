"""Backend triage — keyword-based routing before the librarian graph."""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel

from interfaces.api.backends import Route

from core.logging import get_logger
from librarian.plan.intent import classify_intent
from librarian.schemas.retrieval import Intent

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Static responses (no LLM — keep triage fast, cheap, testable)
# ---------------------------------------------------------------------------

_ESCALATION_RESPONSE = (
    "I'm a research assistant focused on the document corpus. "
    "I can't help with that topic, but I'm happy to answer questions "
    "about the available materials. What would you like to explore?"
)

_CONVERSATIONAL_RESPONSES: dict[str, str] = {
    "greeting": "Hello! I'm the librarian assistant. Ask me anything about the corpus.",
    "thanks": "You're welcome! Let me know if there's anything else I can look up.",
    "help": "I can search, summarise, and compare documents in the corpus. Just ask!",
    "default": "I'm here to help with research questions. What would you like to know?",
}

_GREETING_KEYWORDS = {"hello", "hi", "hey", "good morning", "good afternoon"}
_THANKS_KEYWORDS = {"thanks", "thank you"}
_HELP_KEYWORDS = {"help me", "what can you do", "who are you"}


def _build_conversational_reply(query_lower: str) -> str:
    """Pick the best canned conversational reply based on keyword match."""
    for kw in _GREETING_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
            return _CONVERSATIONAL_RESPONSES["greeting"]
    for kw in _THANKS_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
            return _CONVERSATIONAL_RESPONSES["thanks"]
    for kw in _HELP_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
            return _CONVERSATIONAL_RESPONSES["help"]
    return _CONVERSATIONAL_RESPONSES["default"]


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------

class TriageDecision(BaseModel):
    """Result of the triage classification."""

    route: Route
    intent: str
    confidence: float
    response: str | None = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

_DIRECT_INTENTS: frozenset[Intent] = frozenset({Intent.CONVERSATIONAL})
_ESCALATION_INTENTS: frozenset[Intent] = frozenset({Intent.OUT_OF_SCOPE})


class TriageService:
    """Keyword triage with optional backend fallback for cold starts.

    Parameters
    ----------
    graph_ready:
        Callable that returns ``True`` when the librarian graph is
        initialised and ready to serve.  When the graph is **not** ready
        and a fallback backend is available, librarian-bound queries are
        transparently rerouted so the user still gets an answer during
        cold start.
    bedrock_available:
        Callable that returns ``True`` when a bedrock KB client is
        configured and can serve as a fallback.
    google_adk_available:
        Callable that returns ``True`` when a Google RAG client is
        configured and can serve as a fallback.
    """

    def __init__(
        self,
        *,
        graph_ready: Callable[[], bool] | None = None,
        bedrock_available: Callable[[], bool] | None = None,
        google_adk_available: Callable[[], bool] | None = None,
    ) -> None:
        self._graph_ready = graph_ready or (lambda: True)
        self._bedrock_available = bedrock_available or (lambda: False)
        self._google_adk_available = google_adk_available or (lambda: False)

    def decide(self, query: str, backend: str = "librarian") -> TriageDecision:
        """Classify *query* and return a routing decision.

        If *backend* is ``"bedrock"`` or ``"google_adk"``, classification
        is skipped and the request is forwarded directly.
        """
        if backend == "bedrock":
            return TriageDecision(
                route="bedrock",
                intent="",
                confidence=1.0,
            )
        if backend == "google_adk":
            return TriageDecision(
                route="google_adk",
                intent="",
                confidence=1.0,
            )
        if backend == "adk_bedrock":
            return TriageDecision(
                route="adk_bedrock",
                intent="",
                confidence=1.0,
            )
        if backend == "adk_custom_rag":
            return TriageDecision(
                route="adk_custom_rag",
                intent="",
                confidence=1.0,
            )
        if backend == "adk_hybrid":
            return TriageDecision(
                route="adk_hybrid",
                intent="",
                confidence=1.0,
            )

        intent, confidence = classify_intent(query.lower())

        if intent in _ESCALATION_INTENTS:
            decision = TriageDecision(
                route="escalation",
                intent=intent.value,
                confidence=confidence,
                response=_ESCALATION_RESPONSE,
            )
        elif intent in _DIRECT_INTENTS:
            decision = TriageDecision(
                route="direct",
                intent=intent.value,
                confidence=confidence,
                response=_build_conversational_reply(query.lower()),
            )
        else:
            route: Route = "librarian"
            if not self._graph_ready():
                if self._bedrock_available():
                    log.warning(
                        "triage.fallback.bedrock",
                        reason="graph_not_ready",
                        intent=intent.value,
                    )
                    route = "bedrock"
                elif self._google_adk_available():
                    log.warning(
                        "triage.fallback.google_adk",
                        reason="graph_not_ready",
                        intent=intent.value,
                    )
                    route = "google_adk"
            decision = TriageDecision(
                route=route,
                intent=intent.value,
                confidence=confidence,
            )

        log.info(
            "triage.decision",
            route=decision.route,
            intent=decision.intent,
            confidence=decision.confidence,
        )
        return decision
