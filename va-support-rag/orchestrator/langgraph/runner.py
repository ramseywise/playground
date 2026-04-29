"""CLI runner — prompts for input, runs the graph, handles clarify/confirm interrupts."""

import logging
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from clients.llm import require_llm_for_cli
from core.observability import configure_runtime
from orchestrator.langgraph.schemas.state import GraphState

log = logging.getLogger(__name__)


def run_graph_cli() -> None:
    # Observability before importing LangGraph / LangChain so LangSmith env is applied first.
    configure_runtime()
    require_llm_for_cli()
    from orchestrator.langgraph.graph import poc_graph

    print("=== Support RAG Agent CLI ===")
    print("Please enter your request:\n")

    user_query = input("> ")

    initial_state = GraphState(
        query=user_query,
        messages=[HumanMessage(content=user_query)],
    )

    config = cast(
        RunnableConfig,
        {
            "configurable": {"thread_id": "main-graph-cli"},
            "metadata": {
                "app": "support-rag-agent",
                "entrypoint": "cli",
            },
            "tags": ["langgraph", "cli", "support-rag"],
        },
    )

    thread_id = (config.get("configurable") or {}).get("thread_id", "main-graph-cli")
    log.info("Invoking graph (thread_id=%s)", thread_id)
    print("Invoking graph...\n")
    result = poc_graph.invoke(initial_state, config=config)

    print(f"  - Mode: {result.get('mode')}")
    print(f"  - Q&A outcome: {result.get('qa_outcome')}")
    lat = result.get("latency_ms") or {}
    if lat:
        print(f"  - Latency (ms): {lat}")
    retrieved = result.get("retrieved_context")
    if retrieved:
        preview = retrieved[:100] + ("…" if len(retrieved) > 100 else "")
    else:
        preview = None
    print(f"  - Retrieved Context: {preview}")
    print(f"  - Error: {result.get('error')}")
    print(f"  - Missing Fields: {result.get('missing_fields')}")
    print(f"  - Has Interrupt: {'__interrupt__' in result}\n")

    while "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0].value

        print("\n=== ACTION REQUIRED ===")

        if interrupt_info.get("kind") == "clarify":
            print("I need some additional information to continue.")
            for field in interrupt_info.get("missing_fields", []):
                print(f"- {field}")
            print("\nPlease provide the missing information:")

        elif interrupt_info.get("kind") == "confirm":
            print("Please review the proposed plan:\n")
            for idx, step in enumerate(interrupt_info.get("action_steps", []), start=1):
                print(f"{idx}. {step}")
            print("\nDo you confirm this plan? (yes/no)")

        else:
            print("Input required:")
            print(interrupt_info)

        user_response = input("> ")
        result = poc_graph.invoke(Command(resume=user_response), config=config)

    print("\n=== FINAL ANSWER ===")
    cites = result.get("citations")
    if cites:
        print(f"  (Citations: {len(cites)} chunk(s))")
    final = result.get("final_answer")
    if final is None:
        print("No answer generated")
    else:
        text = getattr(final, "content", None)
        print(text if text is not None else final)


if __name__ == "__main__":
    run_graph_cli()
