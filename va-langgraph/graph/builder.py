"""Graph builder — wires all nodes and edges into a compiled StateGraph."""

from __future__ import annotations

import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes.analyze import analyze_node
from .nodes.direct import direct_node
from .nodes.format import format_node
from .nodes.guardrail import guardrail_node
from .nodes.memory import memory_node
from .nodes.memory_load import memory_load_node
from .state import AgentState
from .subgraphs.domains import (
    accounting_subgraph,
    banking_subgraph,
    customer_subgraph,
    email_subgraph,
    expense_subgraph,
    insights_subgraph,
    invitation_subgraph,
    invoice_subgraph,
    product_subgraph,
    quote_subgraph,
    support_subgraph,
)

# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

_DOMAIN_NODES = {
    "invoice": "invoice",
    "quote": "quote",
    "customer": "customer",
    "product": "product",
    "email": "email",
    "invitation": "invitation",
    "insights": "insights",
    "expense": "expense",
    "banking": "banking",
    "accounting": "accounting",
    "support": "support",
    "direct": "direct",
    "escalation": "escalation",
    "memory": "memory",
}

# Route to direct (out-of-domain response) when classifier confidence is too low
_LOW_CONF_THRESHOLD = float(os.getenv("ROUTING_CONFIDENCE_THRESHOLD", "0.2"))


def _after_guardrail(state: AgentState) -> str:
    return "blocked" if state.get("blocked") else "analyze"


def _route_intent(state: AgentState) -> str:
    intent = state.get("intent", "support")
    confidence = state.get("routing_confidence", 1.0)
    if confidence < _LOW_CONF_THRESHOLD:
        # Too uncertain — let direct_node give a polite redirect
        return "direct"
    return _DOMAIN_NODES.get(intent, "support")


def _is_direct(state: AgentState) -> str:
    return "direct" if state.get("intent") == "direct" else "format"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None) -> "CompiledGraph":  # type: ignore[return]
    """Build and compile the VA LangGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver) for
                      session persistence and HITL support.
    """
    g = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────
    g.add_node("memory_load", memory_load_node)
    g.add_node("guardrail", guardrail_node)
    g.add_node("analyze", analyze_node)
    g.add_node("invoice", invoice_subgraph)
    g.add_node("quote", quote_subgraph)
    g.add_node("customer", customer_subgraph)
    g.add_node("product", product_subgraph)
    g.add_node("email", email_subgraph)
    g.add_node("invitation", invitation_subgraph)
    g.add_node("insights", insights_subgraph)
    g.add_node("expense", expense_subgraph)
    g.add_node("banking", banking_subgraph)
    g.add_node("accounting", accounting_subgraph)
    g.add_node("support", support_subgraph)
    g.add_node("direct", direct_node)
    g.add_node("escalation", _escalation_node)
    g.add_node("memory", memory_node)
    g.add_node("format", format_node)
    g.add_node("blocked", _blocked_node)

    # ── edges ──────────────────────────────────────────────────────────
    g.add_edge(START, "memory_load")
    g.add_edge("memory_load", "guardrail")

    g.add_conditional_edges(
        "guardrail",
        _after_guardrail,
        {"blocked": "blocked", "analyze": "analyze"},
    )

    g.add_conditional_edges(
        "analyze",
        _route_intent,
        {
            "invoice": "invoice",
            "quote": "quote",
            "customer": "customer",
            "product": "product",
            "email": "email",
            "invitation": "invitation",
            "insights": "insights",
            "expense": "expense",
            "banking": "banking",
            "accounting": "accounting",
            "support": "support",
            "direct": "direct",
            "escalation": "escalation",
            "memory": "memory",
        },
    )

    # Domain subgraphs → format (except direct/escalation/blocked/memory which set response directly)
    for domain in ("invoice", "quote", "customer", "product", "email", "invitation", "insights", "expense", "banking", "accounting", "support"):
        g.add_edge(domain, "format")

    g.add_edge("format", END)
    g.add_edge("direct", END)
    g.add_edge("escalation", END)
    g.add_edge("memory", END)
    g.add_edge("blocked", END)

    cp = checkpointer or MemorySaver()
    return g.compile(checkpointer=cp)


def _escalation_node(state: AgentState) -> AgentState:
    """Signal human-supporter handoff."""
    from schema import AssistantResponse

    return {
        **state,
        "response": AssistantResponse(
            message=(
                "I'll connect you with a human supporter right away. "
                "Please hold — someone from the Billy support team will be with you shortly."
            ),
            contact_support=True,
        ).model_dump(),
    }


def _blocked_node(state: AgentState) -> AgentState:
    """Produce a blocked AssistantResponse."""
    from schema import AssistantResponse

    reason = state.get("block_reason", "Your message could not be processed.")
    return {
        **state,
        "response": AssistantResponse(
            message=f"I'm unable to process this request. {reason}",
            contact_support=True,
        ).model_dump(),
    }
