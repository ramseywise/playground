"""Claim-level grounding verification judge.

Verifies each factual claim in the agent's response against the
retrieved context passages.  A claim is *grounded* if directly supported,
*hallucinated* if contradicted, or *unverifiable* if neither.

Aligned with the RAPTOR Stage 3 Control requirement for claim grounding.
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge

_SYSTEM = """\
You are a claim-level grounding verifier for a retrieval-augmented generation system.

Your task is to verify each factual claim in the agent's response against the retrieved \
context passages. A claim is "grounded" if it is directly supported by text in the context. \
A claim is "hallucinated" if it directly contradicts the context. \
A claim is "unverifiable" if neither supported nor contradicted.

Steps:
1. Extract up to 7 distinct factual claims from the response.
2. For each claim, classify it as: grounded | hallucinated | unverifiable.
3. A response contains a hallucination if ANY claim is classified as hallucinated.

Return ONLY a JSON object with these exact keys:
{
  "claims_made": <integer, number of claims extracted>,
  "claims_grounded": <integer, number classified as grounded>,
  "claims_hallucinated": <integer, number classified as hallucinated>,
  "claims_unverifiable": <integer, number classified as unverifiable>,
  "grounding_ratio": <float 0.0-1.0, claims_grounded / claims_made>,
  "has_hallucination": <1.0 if any claim is hallucinated, else 0.0>,
  "is_correct": <true if grounding_ratio >= 0.8 AND has_hallucination == 0.0>,
  "score": <same as grounding_ratio>,
  "reasoning": <one sentence citing the most significant ungrounded or hallucinated claim>
}
No other text outside the JSON object."""


class GroundingJudge(LLMJudge):
    """Claim-level grounding verification against retrieved context."""

    system_prompt: str = _SYSTEM
    grader_type: str = "grounding_judge"

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        parts = [
            f"Retrieved context passages:\n{context}" if context else "Retrieved context passages: [none]",
            f"Agent response:\n{response}",
            f"User query (for reference): {query}",
        ]
        return "\n\n".join(parts)
