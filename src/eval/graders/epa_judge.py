"""EPA communication quality judge.

Scores three affective-communication dimensions specific to customer
support: Empathy (E), Professionalism (P), and Actionability (A).
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge

_SYSTEM = """\
You are a communication quality evaluator for a customer-facing support AI.

Evaluate the agent's response on three dimensions:

EMPATHY (E): Does the response acknowledge the user's emotional state or difficulty? \
Does the tone feel warm and human? Does it avoid being robotic or dismissive?
0.0 = cold/dismissive, 1.0 = highly empathetic and validating.

PROFESSIONALISM (P): Is the language register appropriate for customer support? \
Is it free from slang, inappropriate content, excessive hedging, or harmful language? \
Does it reflect well on the brand?
0.0 = unprofessional or harmful, 1.0 = polished and brand-appropriate.

ACTIONABILITY (A): Does the response give the user a concrete, actionable next step? \
Is it clear what the user should do to resolve their issue?
0.0 = vague or unhelpful, 1.0 = clear actionable guidance with steps.

Return ONLY a JSON object with these exact keys:
{
  "empathy": <float 0.0-1.0>,
  "professionalism": <float 0.0-1.0>,
  "actionability": <float 0.0-1.0>,
  "epa_composite": <float 0.0-1.0, unweighted mean of the three>,
  "is_correct": <true if epa_composite >= 0.65, else false>,
  "score": <same as epa_composite>,
  "reasoning": <one sentence identifying the weakest dimension and why>
}
No other text outside the JSON object."""


class EPAJudge(LLMJudge):
    """Scores Empathy, Professionalism, and Actionability."""

    system_prompt: str = _SYSTEM
    grader_type: str = "epa_judge"

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        return f"User query: {query}\n\nAgent response: {response}"
