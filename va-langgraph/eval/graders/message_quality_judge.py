"""LLM-as-judge for response clarity, tone, and actionability."""

from __future__ import annotations

from ..models import EvalTask, GraderResult
from .llm_judge import LLMJudge

_SYSTEM = """You are evaluating the quality of an AI accounting assistant's response.

Score the response on three dimensions (0.0–1.0 each):

- clarity: Is the message clear and easy to understand? Does it directly answer the question without jargon or unnecessary verbosity?
- tone: Is the tone professional, warm, and supportive? Appropriate for a customer-facing business app?
- actionability: Does the response give the user a clear next step or useful information they can act on?

Respond with JSON ONLY — no markdown fences:
{"clarity": <0.0-1.0>, "tone": <0.0-1.0>, "actionability": <0.0-1.0>, "reasoning": "<one sentence>"}"""

PASS_THRESHOLD = 0.7


class MessageQualityJudge(LLMJudge):
    """Judges response quality across clarity, tone, and actionability."""

    grader_type = "message_quality"
    system_prompt = _SYSTEM

    def _format_user_message(self, task: EvalTask) -> str:
        response = task.metadata.get("response", {})
        message = (
            response.get("message", "") if isinstance(response, dict) else str(response)
        )
        return f"User query: {task.query}\n\nAssistant response:\n{message}"

    def _parse_result(self, parsed: dict, task: EvalTask) -> GraderResult:
        clarity = float(parsed.get("clarity", 0.0))
        tone = float(parsed.get("tone", 0.0))
        actionability = float(parsed.get("actionability", 0.0))
        avg = (clarity + tone + actionability) / 3

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=avg >= PASS_THRESHOLD,
            score=round(avg, 3),
            reasoning=parsed.get("reasoning", ""),
            dimensions={
                "clarity": clarity,
                "tone": tone,
                "actionability": actionability,
            },
        )
