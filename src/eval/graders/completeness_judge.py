"""Answer completeness judge.

Evaluates whether a multi-part question's answer covers all
sub-questions with adequate depth.
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge
from eval.graders.metrics_registry import METRICS

_M = METRICS["completeness"]


class CompletenessJudge(LLMJudge):
    """Evaluates answer completeness for multi-part questions."""

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
        parts = [f"Question: {query}", f"Answer: {response}"]
        if expected:
            parts.append(f"Expected answer (reference): {expected}")
        return "\n\n".join(parts)
