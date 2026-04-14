"""Answer completeness judge.

Evaluates whether a multi-part question's answer covers all
sub-questions with adequate depth.
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge

_SYSTEM = """\
You are a completeness evaluator for a question-answering system.

Your task is to determine whether the provided answer fully addresses ALL parts of a \
multi-part question. Some questions contain two or more distinct sub-questions; the \
answer must cover each one.

Steps:
1. Identify every distinct sub-question or information need within the user question.
2. For each sub-question, determine whether the answer addresses it (yes/no).
3. For each addressed sub-question, rate the depth of the answer (adequate/superficial).

Return ONLY a JSON object with these exact keys:
{
  "sub_questions_identified": <integer count of distinct sub-questions>,
  "sub_questions_answered": <integer count of sub-questions addressed in the answer>,
  "sub_question_coverage": <float 0.0-1.0, fraction answered>,
  "depth_adequacy": <float 0.0-1.0, average depth of answered sub-questions>,
  "overall_completeness": <float 0.0-1.0, weighted composite>,
  "is_correct": <true if overall_completeness >= 0.7, else false>,
  "score": <same value as overall_completeness>,
  "reasoning": <one sentence explaining which sub-questions were missed or were shallow>
}
No other text outside the JSON object."""


class CompletenessJudge(LLMJudge):
    """Evaluates answer completeness for multi-part questions."""

    system_prompt: str = _SYSTEM
    grader_type: str = "completeness_judge"

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
