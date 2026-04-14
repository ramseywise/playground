"""Escalation-appropriateness judge.

Evaluates whether the agent correctly escalated (or refrained from
escalating) based on query scope, context availability, and user signals.
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge
from eval.graders.metrics_registry import METRICS

_M = METRICS["escalation"]


class EscalationJudge(LLMJudge):
    """Evaluates whether escalation behaviour was appropriate."""

    system_prompt: str = _M.standalone_prompt
    grader_type: str = _M.grader_type

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        parts = [
            f"Query: {query}",
            f"Context available to agent: {context}" if context else "Context available to agent: [none]",
            f"Agent response: {response}",
        ]
        return "\n\n".join(parts)
