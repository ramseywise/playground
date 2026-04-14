"""EPA communication quality judge.

Scores three affective-communication dimensions specific to customer
support: Empathy (E), Professionalism (P), and Actionability (A).
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge
from eval.graders.metrics_registry import METRICS

_M = METRICS["epa"]


class EPAJudge(LLMJudge):
    """Scores Empathy, Professionalism, and Actionability."""

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
        return f"User query: {query}\n\nAgent response: {response}"
