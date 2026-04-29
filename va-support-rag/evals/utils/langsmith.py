"""LangSmith integration: fetch recent runs and push eval results.

Requires ``langsmith`` package and ``LANGCHAIN_API_KEY``.
Optional: ``LANGCHAIN_PROJECT`` to scope to a specific project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class LangSmithRunRow:
    """One root trace row for tabular display."""

    run_id: str
    name: str
    status: str
    latency_ms: float | None
    total_tokens: int | None
    app_path: str | None
    error: str | None


def fetch_run_table(
    *,
    project_name: str | None = None,
    limit: int = 40,
) -> tuple[list[LangSmithRunRow], str | None]:
    """Return recent root runs and an error string if the client cannot connect."""
    try:
        from langsmith import Client
    except ImportError:
        return [], "langsmith package not installed."

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return [], "LANGCHAIN_API_KEY is not set."

    client = Client(api_key=api_key)
    project = project_name or os.getenv("LANGCHAIN_PROJECT")

    try:
        gen = client.list_runs(project_name=project, is_root=True, limit=limit)
    except Exception as exc:
        return [], f"LangSmith API error: {exc}"

    rows: list[LangSmithRunRow] = []
    for run in gen:
        st = getattr(run, "start_time", None)
        et = getattr(run, "end_time", None)
        latency_ms = (et - st).total_seconds() * 1000.0 if st and et else None

        tokens: int | None = None
        tt = getattr(run, "total_tokens", None)
        if tt is not None:
            tokens = int(tt)
        elif (
            getattr(run, "prompt_tokens", None) is not None
            and getattr(run, "completion_tokens", None) is not None
        ):
            tokens = int(run.prompt_tokens or 0) + int(run.completion_tokens or 0)

        app_path = None
        if isinstance(getattr(run, "extra", None), dict):
            meta = run.extra.get("metadata")
            if isinstance(meta, dict):
                app_path = meta.get("app") or meta.get("entrypoint")

        rows.append(
            LangSmithRunRow(
                run_id=str(run.id),
                name=str(run.name or ""),
                status=str(run.status or ""),
                latency_ms=latency_ms,
                total_tokens=tokens,
                app_path=str(app_path) if app_path else None,
                error=str(run.error)[:500] if getattr(run, "error", None) else None,
            )
        )
    return rows, None


def push_eval_results(
    results: dict[str, Any],
    *,
    project_name: str | None = None,
    dataset_name: str | None = None,
) -> tuple[int, str | None]:
    """Push ExperimentResult scores to LangSmith as feedback on existing runs.

    Each QueryResult with a ``trace_id`` gets a feedback record for hit_rate
    and reciprocal_rank. Returns (n_pushed, error_string).

    This assumes traces were already created during the experiment run and
    their IDs were stored in QueryResult.trace_id. If you want to create new
    LangSmith experiments from scratch, use the LangSmith SDK evaluate() API.
    """
    try:
        from langsmith import Client
    except ImportError:
        return 0, "langsmith package not installed."

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return 0, "LANGCHAIN_API_KEY is not set."

    client = Client(api_key=api_key)
    n_pushed = 0

    for variant_name, result in results.items():
        for qr in result.query_results:
            if not getattr(qr, "trace_id", None):
                continue
            try:
                client.create_feedback(
                    run_id=qr.trace_id,
                    key="hit_rate",
                    score=1.0 if qr.hit else 0.0,
                    comment=f"variant={variant_name}",
                )
                client.create_feedback(
                    run_id=qr.trace_id,
                    key="reciprocal_rank",
                    score=qr.reciprocal_rank,
                    comment=f"variant={variant_name}",
                )
                n_pushed += 1
            except Exception:
                pass

    return n_pushed, None
