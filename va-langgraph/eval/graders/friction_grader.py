"""Friction and escalation judges — ported from help-support-rag-agent/evals/graders/llm_judge.py.

FrictionJudge:   rates UX friction in a VA response (0=low/good, 1=high/bad).
EscalationJudge: checks whether the VA's escalation decision matched the situation.
                 When a fixture has escalation_signal=True, the ground-truth label
                 is surfaced in dimensions so callers can compute per-tier accuracy.
"""

from __future__ import annotations

from typing import Any

from ..models import EvalTask, GraderResult
from .llm_judge import LLMJudge

# ---------------------------------------------------------------------------
# Friction
# ---------------------------------------------------------------------------

_FRICTION_SYSTEM = """\
You are a user-friction evaluator for a customer-facing support AI.

High friction means the response makes the user's job harder: unnecessary steps, \
jargon, evasion, asking for information the user already gave, burying the answer, \
or shifting work to the user when a direct answer was possible.

Low friction means: direct, concise, easy to follow, and respectful of the user's time.

Return ONLY a JSON object with these exact keys:
{
  "friction_score": <float 0.0-1.0 where 0.0 = low friction (good), 1.0 = high friction (bad)>,
  "is_correct": <true if friction_score <= 0.35, else false>,
  "score": <float 0.0-1.0, higher is better; use (1.0 - friction_score)>,
  "reasoning": <one sentence citing the main friction source or why friction is low>
}
No other text outside the JSON object."""

FRICTION_PASS_THRESHOLD = 0.35


class FrictionJudge(LLMJudge):
    """Rates UX friction in a VA response. Lower friction score = better."""

    system_prompt = _FRICTION_SYSTEM
    grader_type = "friction_judge"

    def _format_user_message(self, task: EvalTask) -> str:
        response = task.metadata.get("response", {})
        message = (
            response.get("message", "") if isinstance(response, dict) else str(response)
        )
        return f"User query: {task.query}\n\nAgent response:\n{message}"

    def _parse_result(self, parsed: dict[str, Any], task: EvalTask) -> GraderResult:
        friction_score = float(parsed.get("friction_score", 1.0))
        score = float(parsed.get("score", 1.0 - friction_score))
        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=friction_score <= FRICTION_PASS_THRESHOLD,
            score=round(score, 3),
            reasoning=parsed.get("reasoning", ""),
            dimensions={
                "friction_score": round(friction_score, 3),
            },
            details={
                "ces_rating": task.ces_rating,
                "test_type": task.test_type,
                "intent": task.expected_intent,
            },
        )


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

_ESCALATION_SYSTEM = """\
You are an escalation-appropriateness evaluator for a customer support AI.

Escalation is appropriate when:
- The user's request falls outside the system's documented scope.
- The user has expressed strong negative emotion or urgency requiring human empathy.
- The query requires a human decision (e.g. billing disputes, legal matters, locked accounts).
- The system could not retrieve any relevant context.

Escalation is inappropriate when:
- The query is a product how-to question the VA can answer from its knowledge base.
- Escalating would leave a resolvable question unanswered.

Steps:
1. Decide whether escalation was WARRANTED by the query.
2. Decide whether the agent ACTUALLY escalated (look for phrases like "I'll connect you", \
"let me transfer", "please contact support", "Bitte wenden Sie sich an", or hand-off language).
3. Score appropriateness as 1.0 if warranted and executed agree, else 0.0.

Return ONLY a JSON object with these exact keys:
{
  "escalation_warranted": <1.0 if warranted, 0.0 if not>,
  "escalation_executed": <1.0 if the agent did escalate, 0.0 if not>,
  "appropriateness": <1.0 if warranted == executed, 0.0 otherwise>,
  "is_correct": <true if appropriateness == 1.0, else false>,
  "score": <same value as appropriateness>,
  "reasoning": <one sentence explaining the decision>
}
No other text outside the JSON object."""


class EscalationJudge(LLMJudge):
    """Checks whether the VA's escalation decision matched the situation.

    When the task has escalation_signal=True (ground truth from the fixture),
    ground_truth_match is surfaced in dimensions for per-tier accuracy analysis.
    """

    system_prompt = _ESCALATION_SYSTEM
    grader_type = "escalation_judge"

    def _format_user_message(self, task: EvalTask) -> str:
        response = task.metadata.get("response", {})
        message = (
            response.get("message", "") if isinstance(response, dict) else str(response)
        )
        return f"User query: {task.query}\n\nAgent response:\n{message}"

    def _parse_result(self, parsed: dict[str, Any], task: EvalTask) -> GraderResult:
        warranted = float(parsed.get("escalation_warranted", 0.0))
        executed = float(parsed.get("escalation_executed", 0.0))
        appropriateness = float(parsed.get("appropriateness", 0.0))

        dimensions: dict[str, float] = {
            "escalation_warranted": warranted,
            "escalation_executed": executed,
            "appropriateness": appropriateness,
        }

        # Surface ground-truth signal for correlation analysis
        if task.escalation_signal is not None:
            gt = 1.0 if task.escalation_signal else 0.0
            dimensions["ground_truth_signal"] = gt
            dimensions["ground_truth_match"] = 1.0 if executed == gt else 0.0

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=appropriateness == 1.0,
            score=round(appropriateness, 3),
            reasoning=parsed.get("reasoning", ""),
            dimensions=dimensions,
            details={
                "ces_rating": task.ces_rating,
                "test_type": task.test_type,
                "intent": task.expected_intent,
                "escalation_signal": task.escalation_signal,
            },
        )
