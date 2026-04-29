"""Logging and LangSmith / LangChain tracing setup.

LangChain and LangGraph read tracing flags from the process environment.
Call :func:`configure_runtime` once at process start (e.g. CLI entrypoint)
before invoking the graph so traces and log levels apply consistently.
"""

from __future__ import annotations

import logging
import os
import sys

from core.config import (
    LANGCHAIN_API_KEY,
    LANGCHAIN_ENDPOINT,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
    LOG_LEVEL,
)

_configured = False


def _configure_root_logging(level: int, *, force: bool = False) -> None:
    """One stdout handler on the root logger; plain format (no request-id machinery)."""
    root = logging.getLogger()
    if not force and root.handlers:
        return
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    )
    root.addHandler(handler)


def configure_runtime() -> None:
    """Configure root logging and ensure LangSmith-related env vars are visible.

    Safe to call multiple times; only the first call has effect.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO

    _configure_root_logging(level, force=True)

    # Suppress HF Hub unauthenticated warnings and progress bars before any
    # lazy import of sentence-transformers / huggingface_hub loads the modules.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Reduce third-party noise unless debugging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    # LangSmith traces can fail with 403 when the key is invalid/expired; those
    # errors are already logged once at startup — suppress the per-run spam.
    logging.getLogger("langsmith").setLevel(logging.ERROR)

    log = logging.getLogger(__name__)

    # LangChain / LangSmith tracing (https://docs.smith.langchain.com/)
    if LANGCHAIN_TRACING_V2:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if LANGCHAIN_ENDPOINT:
            os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGCHAIN_ENDPOINT)
        if LANGCHAIN_API_KEY:
            os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
        if LANGCHAIN_PROJECT:
            os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)

        if LANGCHAIN_API_KEY:
            log.info(
                "LangSmith tracing enabled (project=%s)",
                LANGCHAIN_PROJECT or "(default project)",
            )
        else:
            log.warning(
                "LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is not set — "
                "LangSmith runs may fail. Set LANGCHAIN_API_KEY in your environment.",
            )
    else:
        log.debug(
            "LangSmith tracing off (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable)",
        )
