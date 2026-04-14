"""Knowledge-override detection judge.

Detects whether the model ignored freshly retrieved context in favour
of parametric (pre-trained) knowledge.  High ``parametric_override``
signals a knowledge-override failure — the RAG pipeline retrieved
relevant context but the generation step ignored it.
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge

_SYSTEM = """\
You are a knowledge-override detector for a RAG (retrieval-augmented generation) system.

Your task is to identify whether the agent's answer relied on its parametric (pre-trained) \
knowledge instead of — or in addition to — the retrieved context passages.

Knowledge override occurs when:
- The response contains specific facts, figures, or claims NOT found in the context.
- The response contradicts information in the context.
- The response adds information that could only come from training data.

Knowledge override does NOT occur when the agent:
- Synthesises or rephrases information that IS in the context.
- Adds only common-sense connectives or linguistic scaffolding.

Steps:
1. List up to 5 distinct factual claims in the response.
2. For each claim, state whether it is grounded in the provided context or appears \
to be from parametric knowledge.
3. Compute the fraction of claims that appear parametric.

Return ONLY a JSON object with these exact keys:
{
  "claims_inspected": <integer, number of claims you evaluated>,
  "parametric_claims": <integer, number of claims sourced from parametric knowledge>,
  "context_used": <float 0.0-1.0, fraction grounded in context>,
  "parametric_override": <float 0.0-1.0, fraction from parametric knowledge>,
  "override_score": <float 0.0-1.0, equals 1.0 - parametric_override>,
  "is_correct": <true if parametric_override <= 0.2, else false>,
  "score": <same as override_score>,
  "reasoning": <one sentence citing the most significant override if any>
}
No other text outside the JSON object."""


class KnowledgeOverrideJudge(LLMJudge):
    """Detects when the model ignores retrieved context for parametric knowledge."""

    system_prompt: str = _SYSTEM
    grader_type: str = "knowledge_override_judge"

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        parts = [
            f"Retrieved context:\n{context}" if context else "Retrieved context: [none]",
            f"Agent response:\n{response}",
            f"Original query (for reference): {query}",
        ]
        return "\n\n".join(parts)
