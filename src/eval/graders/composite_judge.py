"""Multi-metric LLM judge — evaluates selected metrics in one call.

Combines multiple evaluation dimensions (grounding, completeness, EPA,
escalation) into a single LLM call.  Each metric produces its own
sub-object in the JSON response; pass/fail is determined by the
registry's ``passes`` predicate, not the LLM's self-assessed
``is_correct``.

Usage::

    judge = CompositeJudge(llm, metrics=["grounding", "epa"])
    result = await judge.grade(task)
    # result.grader_type == "composite:epa+grounding"
    # result.dimensions == {"grounding.grounding_ratio": 0.9, "epa.empathy": 0.8, ...}
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from core.parsing.json import strip_json_fences
from eval.graders.metrics_registry import METRICS, MetricDefinition
from eval.models import EvalTask, GraderResult

if TYPE_CHECKING:
    from playground.src.clients.llm import LLMClient

_PREAMBLE = """\
You are a multi-metric evaluation judge. Evaluate the provided input on each \
of the following dimensions. For EACH dimension return a JSON sub-object under \
its designated key. Return ONLY a single JSON object containing all the \
per-metric sub-objects — no other text.
"""

_FOOTER = """
Return ONLY a JSON object with one top-level key per metric, exactly as named above. \
No other text outside the JSON object."""


class CompositeJudge:
    """Multi-metric LLM judge — evaluates selected metrics in one call.

    Args:
        llm:     Async LLM client with ``generate(system, messages, max_tokens)``.
        metrics: List of metric names from the ``METRICS`` registry.
                 E.g. ``["grounding", "completeness", "epa"]``.

    Pass/fail: ALL selected metrics must pass their individual criteria.
    Score: mean of per-metric scores.
    Dimensions: namespaced as ``{metric}.{key}`` to avoid collision.
    """

    def __init__(self, llm: LLMClient, metrics: list[str]) -> None:
        unknown = set(metrics) - set(METRICS)
        if unknown:
            msg = f"Unknown metrics: {sorted(unknown)}. Available: {sorted(METRICS)}"
            raise ValueError(msg)
        if not metrics:
            msg = "At least one metric must be selected"
            raise ValueError(msg)

        self._llm = llm
        self._metric_names = list(metrics)
        self._metric_defs: list[MetricDefinition] = [METRICS[m] for m in metrics]
        self._system_prompt = self._build_system_prompt()

    @property
    def grader_type(self) -> str:
        return "composite:" + "+".join(sorted(self._metric_names))

    async def grade(self, task: EvalTask) -> GraderResult:
        """Grade a task across all selected metrics in one LLM call."""
        response = task.metadata.get("response", "")
        user_msg = self._format_user_message(
            query=task.query,
            response=response,
            context=task.context,
            expected=task.expected_answer,
        )
        raw = await self._llm.generate(
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=2048,
        )
        return self._parse_combined_result(raw, task_id=task.id)

    def _build_system_prompt(self) -> str:
        sections = [_PREAMBLE]
        for mdef in self._metric_defs:
            sections.append(mdef.composite_prompt_section)
        sections.append(_FOOTER)
        return "\n".join(sections)

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        """Build user message from the union of fields needed by selected metrics."""
        needed = frozenset().union(*(m.required_fields for m in self._metric_defs))

        parts: list[str] = []
        if "context" in needed:
            parts.append(
                f"Retrieved context passages:\n{context}" if context else "Retrieved context passages: [none]"
            )
        parts.append(f"Agent response:\n{response}")
        if "query" in needed:
            parts.append(f"User query: {query}")
        if "expected" in needed and expected:
            parts.append(f"Expected answer (reference): {expected}")
        return "\n\n".join(parts)

    def _parse_combined_result(self, raw: str, task_id: str) -> GraderResult:
        """Parse nested JSON with one sub-object per metric."""
        try:
            data: dict[str, Any] = json.loads(strip_json_fences(raw))
        except (json.JSONDecodeError, ValueError):
            return GraderResult(
                task_id=task_id,
                grader_type=self.grader_type,
                is_correct=False,
                score=0.0,
                reasoning=f"Failed to parse combined judge response: {raw[:200]}",
            )

        all_correct: list[bool] = []
        all_scores: list[float] = []
        all_reasonings: list[str] = []
        dimensions: dict[str, float] = {}
        details: dict[str, Any] = {}

        for mdef in self._metric_defs:
            sub = data.get(mdef.name, {})
            if not isinstance(sub, dict):
                sub = {}

            correct = mdef.passes(sub)
            score = float(sub.get("score", 0.0))
            reasoning = str(sub.get("reasoning", ""))

            all_correct.append(correct)
            all_scores.append(score)
            if reasoning:
                all_reasonings.append(f"[{mdef.name}] {reasoning}")

            for k, v in sub.items():
                if k not in {"is_correct", "score", "reasoning"} and isinstance(v, (int, float)):
                    dimensions[f"{mdef.name}.{k}"] = float(v)
                if k not in {"score", "reasoning"}:
                    details[f"{mdef.name}.{k}"] = v

        return GraderResult(
            task_id=task_id,
            grader_type=self.grader_type,
            is_correct=all(all_correct),
            score=sum(all_scores) / len(all_scores) if all_scores else 0.0,
            reasoning="; ".join(all_reasonings),
            dimensions=dimensions,
            details=details,
        )
