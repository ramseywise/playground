"""Base LLM-as-judge class — shared across agents.

Provides the common pattern: format prompt -> call LLM -> parse JSON verdict.
Agent-specific judges subclass this and provide their own system prompts
and evaluation criteria.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from core.parsing.json import strip_json_fences
from eval.models import EvalTask, GraderResult

if TYPE_CHECKING:
    from clients.llm import LLMClient


class LLMJudge:
    """Base LLM-as-judge.

    Subclasses must set ``system_prompt`` and optionally override
    ``_format_user_message`` and ``_parse_result``.
    """

    system_prompt: str = "You are an evaluation judge."
    grader_type: str = "llm_judge"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def grade(self, task: EvalTask) -> GraderResult:
        """Grade a task using the LLM."""
        response = task.metadata.get("response", "")
        user_msg = self._format_user_message(
            query=task.query,
            response=response,
            context=task.context,
            expected=task.expected_answer,
        )
        raw = await self._llm.generate(
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=1024,
        )
        return self._parse_result(raw, task_id=task.id)

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

    def _parse_result(self, raw: str, task_id: str = "") -> GraderResult:
        """Parse the LLM's JSON verdict into a GraderResult."""
        try:
            data: dict[str, Any] = json.loads(strip_json_fences(raw))
            score = float(data.get("score", 0.0))
            return GraderResult(
                task_id=task_id,
                grader_type=self.grader_type,
                is_correct=bool(data.get("is_correct", False)),
                score=score,
                reasoning=str(data.get("reasoning", "")),
                dimensions={
                    k: float(v)
                    for k, v in data.items()
                    if k not in {"is_correct", "score", "reasoning"} and isinstance(v, (int, float))
                },
                details={
                    k: v
                    for k, v in data.items()
                    if k not in {"is_correct", "score", "reasoning"}
                },
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return GraderResult(
                task_id=task_id,
                grader_type=self.grader_type,
                is_correct=False,
                score=0.0,
                reasoning=f"Failed to parse judge response: {raw[:200]}",
            )
