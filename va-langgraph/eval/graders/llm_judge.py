"""Base LLM-as-judge: format prompt → call LLM → parse JSON → GraderResult."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..models import EvalTask, GraderResult

logger = logging.getLogger(__name__)


class LLMJudge:
    """Base class for all LLM-judged graders.

    Subclasses override system_prompt, grader_type, and _parse_result.
    """

    system_prompt: str = "You are an evaluator. Return JSON only."
    grader_type: str = "llm_judge"

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def grade(self, task: EvalTask) -> GraderResult:
        user_msg = self._format_user_message(task)

        try:
            resp = await self.llm.ainvoke([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            parsed = json.loads(raw)
            return self._parse_result(parsed, task)
        except Exception as e:
            logger.warning("%s failed for task %s: %s", self.grader_type, task.id, e)
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                is_correct=False,
                score=0.0,
                reasoning=f"Judge error: {e}",
            )

    def _format_user_message(self, task: EvalTask) -> str:
        response = task.metadata.get("response", {})
        message = response.get("message", "") if isinstance(response, dict) else ""
        return (
            f"User query: {task.query}\n\n"
            f"Assistant response:\n{message}"
        )

    def _parse_result(self, parsed: dict[str, Any], task: EvalTask) -> GraderResult:
        score = float(parsed.get("score", 0.0))
        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=bool(parsed.get("is_correct", score >= 0.7)),
            score=score,
            reasoning=parsed.get("reasoning", ""),
            dimensions={k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))},
        )
