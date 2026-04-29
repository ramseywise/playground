"""Operational guardrails when the orchestrator raises (graph node bug, LLM timeout, etc.).

RAG quality paths already set ``GraphState.error`` and route to the escalation node; this
module covers *uncaught exceptions* at the runtime boundary.
"""

from __future__ import annotations

from orchestrator.schemas import AgentOutput

# User-safe text — do not echo exception strings to clients.
PIPELINE_FAILURE_ANSWER = (
    "We couldn't complete your request due to a technical issue. "
    "Please try again in a moment or contact support if the problem continues."
)


def agent_output_for_pipeline_failure() -> AgentOutput:
    """Structured response for FastAPI / clients when the pipeline throws."""
    return AgentOutput(
        answer=PIPELINE_FAILURE_ANSWER,
        citations=[],
        pipeline_error=True,
    )


def stream_error_data() -> dict[str, str]:
    """Payload for StreamEvent(kind='error') — safe for end users."""
    return {"message": PIPELINE_FAILURE_ANSWER}


__all__ = [
    "PIPELINE_FAILURE_ANSWER",
    "agent_output_for_pipeline_failure",
    "stream_error_data",
]
