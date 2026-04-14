"""Escalation-appropriateness judge.

Evaluates whether the agent correctly escalated (or refrained from
escalating) based on query scope, context availability, and user signals.
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge

_SYSTEM = """\
You are an escalation-appropriateness evaluator for a customer support AI.

Escalation is appropriate when:
- The user's request falls outside the system's documented scope.
- The user has expressed strong negative emotion or urgency requiring human empathy.
- The query requires a human decision (e.g. billing disputes, legal matters, complaints).
- The system could not retrieve any relevant context.

Escalation is inappropriate when:
- The query is within scope and context was available.
- Escalating would leave a resolvable question unanswered.

Steps:
1. Decide whether escalation was WARRANTED by the query and context.
2. Decide whether the agent ACTUALLY escalated (look for phrases like \
"I'll connect you", "let me transfer", "I recommend speaking with a \
representative", or explicit hand-off language).
3. Score appropriateness as 1.0 if warranted and executed are consistent, else 0.0.

Return ONLY a JSON object with these exact keys:
{
  "escalation_warranted": <1.0 if warranted, 0.0 if not>,
  "escalation_executed": <1.0 if the agent did escalate, 0.0 if not>,
  "appropriateness": <1.0 if warranted and executed agree, 0.0 otherwise>,
  "is_correct": <true if appropriateness == 1.0, else false>,
  "score": <same value as appropriateness>,
  "reasoning": <one sentence explaining the decision>
}
No other text outside the JSON object."""


class EscalationJudge(LLMJudge):
    """Evaluates whether escalation behaviour was appropriate."""

    system_prompt: str = _SYSTEM
    grader_type: str = "escalation_judge"

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
