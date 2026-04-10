"""Base LLM-as-judge class — shared across agents.

Provides the common pattern: format prompt → call LLM → parse JSON verdict.
Agent-specific judges subclass this and provide their own system prompts
and evaluation criteria.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from core.parsing.json import strip_json_fences
from eval.protocols import JudgeResult

if TYPE_CHECKING:
    from core.clients.llm import LLMClient


class LLMJudge:
    """Base LLM-as-judge.

    Subclasses must set ``system_prompt`` and optionally override
    ``_format_user_message`` and ``_parse_result``.
    """

    system_prompt: str = "You are an evaluation judge."

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def grade(
        self,
        *,
        query: str,
        response: str,
        context: str = "",
        expected: str = "",
    ) -> JudgeResult:
        """Grade a single (query, response) pair using the LLM."""
        user_msg = self._format_user_message(
            query=query, response=response, context=context, expected=expected
        )
        raw = await self._llm.generate(
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=1024,
        )
        return self._parse_result(raw)

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        """Build the user message for the judge LLM.  Override for custom format."""
        parts = [f"Question: {query}", f"Answer: {response}"]
        if context:
            parts.insert(1, f"Context: {context}")
        if expected:
            parts.append(f"Expected: {expected}")
        return "\n\n".join(parts)

    def _parse_result(self, raw: str) -> JudgeResult:
        """Parse the LLM's JSON verdict into a JudgeResult."""
        try:
            data: dict[str, Any] = json.loads(strip_json_fences(raw))
            return JudgeResult(
                is_correct=bool(data.get("is_correct", False)),
                score=float(data.get("score", 0.0)),
                reasoning=str(data.get("reasoning", "")),
                details={
                    k: v
                    for k, v in data.items()
                    if k not in {"is_correct", "score", "reasoning"}
                },
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return JudgeResult(
                is_correct=False,
                score=0.0,
                reasoning=f"Failed to parse judge response: {raw[:200]}",
            )
