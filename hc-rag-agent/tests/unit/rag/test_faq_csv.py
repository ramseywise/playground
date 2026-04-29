"""FAQ CSV loaders for ingestion and golden eval rows."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from evals.utils.loaders import load_golden_from_faq_csv

from rag.preprocessing.faq_csv import (
    eval_query_and_doc_url,
    load_faq_csv_documents,
)


def test_load_faq_csv_documents_question_answer(tmp_path: Path) -> None:
    p = tmp_path / "faq.csv"
    p.write_text(
        "Question,Answer,URL\n"
        "How reset password?,Use the forgot link.,https://help.example.com/pw\n",
        encoding="utf-8",
    )
    docs = load_faq_csv_documents(p)
    assert len(docs) == 1
    d = docs[0]
    assert "Question: How reset password?" in d["text"]
    assert "Answer: Use the forgot link." in d["text"]
    assert d["url"] == "https://help.example.com/pw"
    assert "faq.csv:line_2" in d["source_file"]


def test_load_golden_from_faq_csv(tmp_path: Path) -> None:
    p = tmp_path / "golden.csv"
    p.write_text(
        "query_id,question,url,category,relevant_chunks\n"
        "q1,How reset password?,https://help.example.com/pw,auth,abc|def\n",
        encoding="utf-8",
    )
    samples = load_golden_from_faq_csv(p)
    assert len(samples) == 1
    s = samples[0]
    assert s.query_id == "q1"
    assert s.query == "How reset password?"
    assert s.expected_doc_url == "https://help.example.com/pw"
    assert s.category == "auth"
    assert s.relevant_chunk_ids == ["abc", "def"]


def test_eval_query_and_doc_url_aliases() -> None:
    q, u = eval_query_and_doc_url(
        {"question": "x", "source_doc_id": "https://x", "noise": ""}
    )
    assert q == "x"
    assert u == "https://x"


def test_load_golden_json_chunk_ids(tmp_path: Path) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["question", "url", "relevant_chunks"])
    w.writerow(["q", "https://u", '["a", "b"]'])
    p = tmp_path / "g.csv"
    p.write_text(buf.getvalue(), encoding="utf-8")
    samples = load_golden_from_faq_csv(p)
    assert samples[0].relevant_chunk_ids == ["a", "b"]


def test_load_golden_from_faq_csv_limit(tmp_path: Path) -> None:
    p = tmp_path / "many.csv"
    lines = ["question,url\n"] + [f"q{i},https://u{i}\n" for i in range(10)]
    p.write_text("".join(lines), encoding="utf-8")
    samples = load_golden_from_faq_csv(p, limit=3)
    assert len(samples) == 3


def test_parse_pipe_chunk_ids(tmp_path: Path) -> None:
    p = tmp_path / "g.csv"
    p.write_text(
        "question,url,relevant_chunks\nq,https://u,a1|a2\n",
        encoding="utf-8",
    )
    samples = load_golden_from_faq_csv(p)
    assert samples[0].relevant_chunk_ids == ["a1", "a2"]
