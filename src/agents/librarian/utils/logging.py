"""Re-export from canonical location: infra.config.logging.

Logging setup now lives in the shared infra layer.
This module re-exports for backward compatibility within the librarian.
"""

from __future__ import annotations

from infra.config.logging import configure_logging, get_logger  # noqa: F401

__all__ = ["configure_logging", "get_logger"]
