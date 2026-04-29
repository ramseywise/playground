"""Keyword planner routing — no LLM."""

import json

import pytest

from orchestrator.langgraph.nodes import planner as pk


def test_resolve_mode_task_phrase_longest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_PLANNER_KEYWORDS_JSON", raising=False)
    monkeypatch.delenv("RAG_PLANNER_KEYWORDS_PATH", raising=False)
    payload = {
        "task_execution": ["cancel", "cancel my order"],
        "q&a": {"misc": ["help"]},
    }
    monkeypatch.setenv("RAG_PLANNER_KEYWORDS_JSON", json.dumps(payload))
    pk.clear_planner_keyword_cache()

    assert pk.resolve_mode_keyword("please cancel my order today") == "task_execution"
    assert pk.resolve_mode_keyword("how do I cancel") == "task_execution"


def test_resolve_mode_defaults_to_qa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_PLANNER_KEYWORDS_JSON", raising=False)
    monkeypatch.delenv("RAG_PLANNER_KEYWORDS_PATH", raising=False)
    monkeypatch.setenv("RAG_PLANNER_KEYWORDS_JSON", json.dumps({"task_execution": []}))
    pk.clear_planner_keyword_cache()

    assert pk.resolve_mode_keyword("what is your return policy") == "q&a"


def test_qa_category_keywords_still_qa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "RAG_PLANNER_KEYWORDS_JSON",
        json.dumps(
            {
                "task_execution": [],
                "q&a": {"returns": ["return policy", "warranty"]},
            }
        ),
    )
    pk.clear_planner_keyword_cache()

    assert pk.resolve_mode_keyword("question about return policy") == "q&a"


def test_parse_accepts_qa_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "RAG_PLANNER_KEYWORDS_JSON",
        json.dumps({"task_execution": [], "qa": {"x": ["y"]}}),
    )
    pk.clear_planner_keyword_cache()
    cfg = pk.load_planner_keyword_config()
    assert "x" in cfg.qa_by_category


def test_corrupt_json_falls_back_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PLANNER_KEYWORDS_JSON", "not-json")
    pk.clear_planner_keyword_cache()
    assert pk.resolve_mode_keyword("anything") == "q&a"
