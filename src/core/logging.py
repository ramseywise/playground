"""Re-export from canonical location: core.config.logging.

This module re-exports ``get_logger`` for backward compatibility —
47+ files import ``from core.logging import get_logger``.
"""

from __future__ import annotations

from core.config.logging import get_logger

__all__ = ["get_logger"]
