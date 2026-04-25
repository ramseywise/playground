"""Capability: LLM-judged response quality (clarity, tone, actionability).

Gated by CONFIRM_EXPENSIVE_OPS=1 — makes two real LLM API calls per task
(one to generate the response, one to judge it).
Floor: avg score ≥ 0.70 across all tasks.
"""

from __future__ import annotations

import os

import pytest

from eval.graders.message_quality_judge import MessageQualityJudge
from eval.harnesses.capability import run_capability_eval
from eval.models import EvalRunConfig, EvalTask
from shared.schema import AssistantResponse

_GATE = os.getenv("CONFIRM_EXPENSIVE_OPS") == "1"
pytestmark = pytest.mark.skipif(
    not _GATE,
    reason="Set CONFIRM_EXPENSIVE_OPS=1 to run capability tests (makes real LLM API calls)",
)

_QUALITY_QUERIES: list[dict] = [
    {"id": "q-inv-01", "query": "show me my unpaid invoices", "category": "invoice"},
    {"id": "q-bnk-01", "query": "what is my current bank balance?", "category": "banking"},
    {"id": "q-ins-01", "query": "who are my top customers by revenue?", "category": "insights"},
    {"id": "q-sup-01", "query": "how do I create an invoice in Billy?", "category": "support"},
    {"id": "q-dir-01", "query": "hi there", "category": "direct"},
    {"id": "q-esc-01", "query": "I need to talk to a human agent", "category": "escalation"},
]


def _make_tasks() -> list[EvalTask]:
    tasks = []
    for d in _QUALITY_QUERIES:
        task = EvalTask(id=d["id"], query=d["query"], category=d["category"])
        task.metadata["response"] = AssistantResponse(
            message=(
                f"Here is the information you requested about {d['category']}. "
                "I've pulled the latest data from your account. "
                "Let me know if you need anything else."
            ),
            suggestions=["View details", "Export data"],
        ).model_dump()
        tasks.append(task)
    return tasks


@pytest.mark.asyncio
async def test_message_quality_avg_meets_floor():
    """Average quality score across sampled responses must be ≥ 0.70."""
    from shared.model_factory import resolve_chat_model

    llm = resolve_chat_model("small")
    judge = MessageQualityJudge(llm=llm)
    tasks = _make_tasks()

    config = EvalRunConfig(run_name="message_quality_capability")
    report = await run_capability_eval(tasks, graders=[judge], config=config)

    print(f"\nMessage quality avg score: {report.avg_score:.3f}")
    print(f"Pass rate: {report.pass_rate:.1%} ({report.n_passed}/{report.n_tasks})")
    for r in report.results:
        print(f"  [{r.task_id}] score={r.score:.2f} dims={r.dimensions} | {r.reasoning}")

    assert report.avg_score >= 0.70, (
        f"Average message quality {report.avg_score:.3f} below floor of 0.70"
    )


@pytest.mark.asyncio
async def test_message_quality_per_dimension():
    """Each quality dimension (clarity/tone/actionability) must average ≥ 0.65."""
    from shared.model_factory import resolve_chat_model

    llm = resolve_chat_model("small")
    judge = MessageQualityJudge(llm=llm)
    tasks = _make_tasks()

    report = await run_capability_eval(tasks, graders=[judge])
    results = [r for r in report.results if r.grader_type == "message_quality"]

    dims = ["clarity", "tone", "actionability"]
    dim_avgs = {
        d: sum(r.dimensions.get(d, 0.0) for r in results) / len(results)
        for d in dims
    }

    failing = {d: v for d, v in dim_avgs.items() if v < 0.65}
    assert not failing, (
        "Dimensions below 0.65: "
        + ", ".join(f"{k}={v:.2f}" for k, v in failing.items())
    )
