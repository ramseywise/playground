from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval.models import EvalTask


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Mock LLM with ``generate(system, messages) -> str`` interface."""
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture()
def make_task() -> Any:
    """Factory for building EvalTask instances with sane defaults."""

    def _make(
        *,
        task_id: str = "t1",
        query: str = "How do I reset my password?",
        expected_answer: str = "Go to Settings > Security > Reset password.",
        context: str = "To reset your password, navigate to Settings, then Security.",
        response: str = "You can reset your password in Settings under Security.",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvalTask:
        meta: dict[str, Any] = {"response": response}
        if metadata:
            meta.update(metadata)
        return EvalTask(
            id=task_id,
            query=query,
            expected_answer=expected_answer,
            context=context,
            metadata=meta,
            tags=tags or [],
        )

    return _make
