"""Planner node and keyword-based mode routing."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from core.config import RAG_LLM_PLANNER
from orchestrator.langgraph.schemas.state import GraphState
from orchestrator.langgraph.chains import get_planner_agent

log = logging.getLogger(__name__)

Mode = Literal["q&a", "task_execution"]


@dataclass(frozen=True)
class PlannerKeywordConfig:
    """Loaded keyword routing config."""

    task_phrases: tuple[str, ...]
    qa_by_category: dict[str, tuple[str, ...]]


def _project_data_dir() -> Path:
    env = os.getenv("RAG_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent.parent.parent / "data"


def _parse_json_payload(raw: dict[str, Any]) -> PlannerKeywordConfig:
    task_raw = raw.get("task_execution", [])
    if not isinstance(task_raw, list):
        raise ValueError("task_execution must be a list of strings")
    task_phrases = tuple(str(x).strip() for x in task_raw if str(x).strip())

    qa_raw = raw.get("q&a", raw.get("qa", {}))
    qa_by_category: dict[str, tuple[str, ...]] = {}
    if isinstance(qa_raw, dict):
        for cat, kws in qa_raw.items():
            if not isinstance(kws, list):
                continue
            cleaned = tuple(str(x).strip() for x in kws if str(x).strip())
            if cleaned:
                qa_by_category[str(cat)] = cleaned

    return PlannerKeywordConfig(
        task_phrases=task_phrases,
        qa_by_category=qa_by_category,
    )


def _load_raw_json() -> dict[str, Any] | None:
    inline = os.getenv("RAG_PLANNER_KEYWORDS_JSON")
    if inline and inline.strip():
        return cast(dict[str, Any], json.loads(inline))

    path_env = os.getenv("RAG_PLANNER_KEYWORDS_PATH")
    if path_env and path_env.strip():
        p = Path(path_env).expanduser()
        if not p.is_file():
            log.warning("planner_keywords.missing_file path=%s", p)
            return None
        return cast(dict[str, Any], json.loads(p.read_text(encoding="utf-8")))

    candidate = _project_data_dir() / "planner_keywords.json"
    if candidate.is_file():
        return cast(dict[str, Any], json.loads(candidate.read_text(encoding="utf-8")))

    return None


@lru_cache(maxsize=1)
def load_planner_keyword_config() -> PlannerKeywordConfig:
    """Load and cache keyword config (clear cache in tests after env changes)."""
    try:
        raw = _load_raw_json()
        if raw is None:
            return PlannerKeywordConfig(task_phrases=(), qa_by_category={})
        return _parse_json_payload(raw)
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
        log.warning("planner_keywords.load_failed error=%s defaulting=empty", e)
        return PlannerKeywordConfig(task_phrases=(), qa_by_category={})


def resolve_mode_keyword(user_text: str) -> Mode:
    """Return graph mode using substring rules; default ``q&a``."""
    cfg = load_planner_keyword_config()
    t = user_text.casefold()

    for phrase in sorted(cfg.task_phrases, key=len, reverse=True):
        if phrase.casefold() in t:
            return "task_execution"

    for kws in cfg.qa_by_category.values():
        for kw in kws:
            if kw.casefold() in t:
                return "q&a"

    return "q&a"


def clear_planner_keyword_cache() -> None:
    """Test helper: call after changing env-based JSON."""
    load_planner_keyword_config.cache_clear()


def _stringify_message_content(content: str | list[str | dict[str, Any]]) -> str:
    """LangChain message content may be str or multimodal blocks — planner needs text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            tx = block.get("text")
            if isinstance(tx, str):
                parts.append(tx)
    return "\n".join(parts).strip()


def planner_node(state: GraphState) -> dict:
    if state.messages:
        input_text = _stringify_message_content(state.messages[-1].content)
    else:
        input_text = state.query or ""
    log.info("planner: start query_len=%d llm=%s", len(input_text), RAG_LLM_PLANNER)
    t0 = time.perf_counter()

    intent: str | None = None
    hints: list[str] = []
    if RAG_LLM_PLANNER:
        try:
            result = get_planner_agent().invoke({"input": input_text})
            mode = result.mode
            intent = getattr(result, "intent", None) or None
            hints = list(getattr(result, "retrieval_hints", None) or [])
            log.info(
                "planner: mode=%s intent=%s source=llm elapsed=%.2fs",
                mode,
                intent,
                time.perf_counter() - t0,
            )
        except Exception:
            log.exception("planner: llm_failed falling_back=keyword")
            mode = resolve_mode_keyword(input_text)
            log.info(
                "planner: mode=%s source=keyword_fallback elapsed=%.2fs",
                mode,
                time.perf_counter() - t0,
            )
    else:
        mode = resolve_mode_keyword(input_text)
        log.info(
            "planner: mode=%s source=keyword elapsed=%.3fs",
            mode,
            time.perf_counter() - t0,
        )

    return {"mode": mode, "planner_intent": intent, "planner_retrieval_hints": hints}


__all__ = [
    "Mode",
    "PlannerKeywordConfig",
    "clear_planner_keyword_cache",
    "load_planner_keyword_config",
    "planner_node",
    "resolve_mode_keyword",
]
