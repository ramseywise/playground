"""Generic synthetic QA generation scaffold.

Generates (query, expected_answer) pairs from text passages using an LLM.
Agent-specific generators (e.g. librarian's generate_synthetic) can use
this as a base or call it directly.

Cost gate: ``confirm_expensive`` must be True before any LLM calls.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from eval.models import EvalTask
from librarian.tasks.extract import make_task_id

if TYPE_CHECKING:
    from clients.llm import LLMClientSync

log = structlog.get_logger(__name__)

DEFAULT_SYSTEM = """\
You are a question generation assistant.  Given a passage of text,
generate one realistic user question that the passage directly answers.

Rules:
- The question must be answerable ONLY from the passage.
- Write it as a natural user query.
- Do not reference the passage explicitly.
- Return ONLY a JSON object: {"query": "<question>", "answer": "<answer from passage>", "difficulty": "easy"|"medium"|"hard"}
No other text."""

DEFAULT_USER_TMPL = "Passage:\n{text}"


def generate_synthetic_tasks(
    passages: list[dict[str, Any]],
    llm: LLMClientSync,
    *,
    n: int | None = None,
    confirm_expensive: bool = False,
    text_field: str = "text",
    system_prompt: str = DEFAULT_SYSTEM,
    user_template: str = DEFAULT_USER_TMPL,
) -> list[EvalTask]:
    """Generate synthetic EvalTask objects from text passages.

    Args:
        passages: List of dicts with at least a ``text_field`` key.
        llm: Sync LLM client for generation.
        n: Max number of tasks to generate. Defaults to all passages.
        confirm_expensive: Cost gate — must be True to proceed.
        text_field: Key for the passage text.

    Returns:
        List of EvalTask with validation_level="synthetic".
    """
    if not confirm_expensive:
        msg = "Set confirm_expensive=True to run synthetic generation."
        raise RuntimeError(msg)

    target = passages[:n] if n is not None else passages
    tasks: list[EvalTask] = []

    for i, passage in enumerate(target):
        text = passage.get(text_field, "")
        if not text.strip():
            log.warning("synthetic.skip.empty", index=i)
            continue

        result = _generate_one(llm, text, system_prompt, user_template)
        if result is None:
            continue

        query = result.get("query", "").strip()
        answer = result.get("answer", "").strip()
        difficulty = result.get("difficulty", "medium")
        if not query:
            log.warning("synthetic.skip.empty_query", index=i)
            continue

        task_id = make_task_id(query, answer)
        tasks.append(
            EvalTask(
                id=task_id,
                query=query,
                expected_answer=answer,
                difficulty=difficulty,
                validation_level="synthetic",
                metadata={k: v for k, v in passage.items() if k != text_field},
            )
        )

    log.info("synthetic.done", n_passages=len(target), n_generated=len(tasks))
    return tasks


def _generate_one(
    llm: LLMClientSync,
    text: str,
    system_prompt: str,
    user_template: str,
) -> dict[str, Any] | None:
    """Call LLM for a single passage.  Returns parsed dict or None on error."""
    try:
        raw = llm.generate_sync(
            system=system_prompt,
            messages=[{"role": "user", "content": user_template.format(text=text[:2000])}],
            max_tokens=256,
        )
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("synthetic.parse_error", error=str(exc))
        return None
