"""Structlog configuration for all agents.

Call ``configure_logging()`` once at startup.  All modules use
``get_logger(__name__)`` for structured logging.
"""

from __future__ import annotations

import structlog


def configure_logging(*, render_json: bool = False) -> None:
    """Call once at startup.  *render_json=True* for prod/CI (JSON lines)."""
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if render_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for *name*."""
    return structlog.get_logger(name)
