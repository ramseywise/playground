"""Re-export from canonical location: core.config.logging."""

from __future__ import annotations

from core.config.logging import configure_logging, get_logger  # noqa: F401

__all__ = ["configure_logging", "get_logger"]
