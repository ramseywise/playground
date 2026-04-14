"""Conciseness grader — hybrid deterministic + optional LLM check.

Stage 1 (deterministic): token-count ratio against a configurable budget.
Stage 2 (optional LLM): detects filler/padding in the response text.

Construct with ``llm=None`` for a free, deterministic-only mode.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from core.parsing.json import strip_json_fences
from eval.models import EvalTask, GraderResult

if TYPE_CHECKING:
    from playground.src.clients.llm import LLMClient

_PADDING_SYSTEM = """\
You are a conciseness evaluator for a customer support AI.

Assess whether the response is appropriately concise or contains unnecessary padding. \
Padding includes: excessive preamble, repetition of the question, hedging disclaimers \
that add no value, and filler phrases like "Certainly!", "Great question!", "Of course!".

Return ONLY a JSON object with these exact keys:
{
  "padding_score": <float 0.0-1.0, where 1.0 means no padding and 0.0 means severe padding>,
  "is_correct": <true if padding_score >= 0.7, else false>,
  "score": <same as padding_score>,
  "reasoning": <one sentence identifying the main source of padding if any>
}
No other text outside the JSON object."""


class ConcisenessGrader:
    """Hybrid conciseness grader: token budget + optional LLM padding check.

    Args:
        llm: Optional LLM client for the padding sub-evaluation.
             When ``None``, only the deterministic token-ratio stage runs.
        max_ratio: Maximum acceptable ``response_tokens / expected_tokens``.
        expected_tokens: Baseline expected token count for a concise answer.
    """

    grader_type: str = "conciseness"

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        max_ratio: float = 2.0,
        expected_tokens: int = 150,
    ) -> None:
        self._llm = llm
        self._max_ratio = max_ratio
        self._expected_tokens = expected_tokens

    async def grade(self, task: EvalTask) -> GraderResult:
        response = task.metadata.get("response", "")
        response_tokens = len(response.split())

        token_ratio = response_tokens / max(self._expected_tokens, 1)
        within_budget = 1.0 if token_ratio <= self._max_ratio else 0.0
        score = min(1.0, 1.0 / max(token_ratio, 0.01))

        dimensions: dict[str, float] = {
            "token_ratio": round(token_ratio, 4),
            "within_budget": within_budget,
        }

        padding_score: float | None = None
        if self._llm is not None:
            padding_score = await self._evaluate_padding(response)
            dimensions["padding_score"] = padding_score

        is_correct = within_budget == 1.0
        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=is_correct,
            score=score,
            reasoning=f"token_ratio={token_ratio:.2f}, within_budget={within_budget == 1.0}",
            dimensions=dimensions,
        )

    async def _evaluate_padding(self, response: str) -> float:
        """Run the LLM padding sub-evaluation. Returns padding_score 0.0-1.0."""
        assert self._llm is not None  # noqa: S101
        raw = await self._llm.generate(
            system=_PADDING_SYSTEM,
            messages=[{"role": "user", "content": f"Agent response:\n{response}"}],
            max_tokens=512,
        )
        try:
            data: dict[str, Any] = json.loads(strip_json_fences(raw))
            return float(data.get("padding_score", 0.0))
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.0
