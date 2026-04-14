"""Claim-level grounding and knowledge-override verification judge.

Verifies each factual claim in the agent's response against the
retrieved context passages.  A claim is *grounded* if directly supported,
*hallucinated* if contradicted, *unverifiable* if neither, or *parametric*
if it appears to come from the model's pre-trained knowledge rather than
the retrieved context.

Combines two failure modes in a single evaluation pass:
- **Hallucination**: claims that contradict the retrieved context.
- **Knowledge override**: claims sourced from parametric knowledge instead
  of the freshly retrieved context.
"""

from __future__ import annotations

from eval.graders.llm_judge import LLMJudge
from eval.graders.metrics_registry import METRICS

_M = METRICS["grounding"]


class GroundingJudge(LLMJudge):
    """Claim-level grounding and knowledge-override verification.

    Pass criteria (all must hold):
    - ``grounding_ratio >= 0.8``
    - ``has_hallucination == 0.0``
    - ``parametric_override <= 0.2``
    """

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
            f"Retrieved context passages:\n{context}" if context else "Retrieved context passages: [none]",
            f"Agent response:\n{response}",
            f"User query (for reference): {query}",
        ]
        return "\n\n".join(parts)
