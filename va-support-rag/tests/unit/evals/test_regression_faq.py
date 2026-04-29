"""Retrieval regression harness — FAQ CSV slice (default 50 rows) with a mock retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.schemas.chunks import Chunk, ChunkMetadata
from rag.schemas.retrieval import RetrievalResult
from evals.harnesses.regression import run_regression_eval
from evals.utils.loaders import golden_samples_to_eval_tasks, load_golden_from_faq_csv


def _write_faq_csv(path: Path, n_rows: int) -> None:
    lines = ["query_id,question,url,category\n"]
    for i in range(n_rows):
        lines.append(
            f"q{i},What is topic {i}?,https://example.com/doc{i},faq\n",
        )
    path.write_text("".join(lines), encoding="utf-8")


@pytest.mark.asyncio
async def test_regression_faq_fifty_rows_perfect_retriever(tmp_path: Path) -> None:
    """First-line regression: 50 FAQ labels; oracle retriever ⇒ hit_rate 1.0."""
    p = tmp_path / "golden.csv"
    _write_faq_csv(p, n_rows=60)
    gold = load_golden_from_faq_csv(p, limit=50)
    assert len(gold) == 50
    tasks = golden_samples_to_eval_tasks(gold)
    q_to_url = {s.query: s.expected_doc_url for s in gold}

    async def oracle_retrieve(query: str) -> list[RetrievalResult]:
        url = q_to_url[query]
        chunk = Chunk(
            id="c1",
            text="snippet",
            metadata=ChunkMetadata(url=url, title="t", doc_id="d1"),
        )
        return [RetrievalResult(chunk=chunk, score=1.0, source="test")]

    report = await run_regression_eval(tasks, oracle_retrieve, k=5)
    assert report.n_tasks == 50
    assert report.pass_rate == 1.0
    assert report.n_passed == 50


def test_golden_samples_to_eval_tasks_metadata(tmp_path: Path) -> None:
    p = tmp_path / "g.csv"
    p.write_text(
        "query_id,question,url\nx,q1,https://a\n",
        encoding="utf-8",
    )
    gold = load_golden_from_faq_csv(p)
    tasks = golden_samples_to_eval_tasks(gold)
    assert len(tasks) == 1
    assert tasks[0].metadata["expected_doc_url"] == "https://a"
