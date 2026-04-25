"""Capability harness: tasks × graders → EvalReport (any-pass logic)."""

from __future__ import annotations

from typing import Any

from ..models import EvalReport, EvalRunConfig, EvalTask
from ..runner import EvalRunner


async def run_capability_eval(
    tasks: list[EvalTask],
    graders: list[Any],
    config: EvalRunConfig | None = None,
) -> EvalReport:
    """Run capability evaluation: a task passes if ANY grader marks it correct."""
    runner = EvalRunner(graders=graders, config=config)
    return await runner.run_capability(tasks)
