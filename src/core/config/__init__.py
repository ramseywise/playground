"""Configuration and logging for the platform."""

from core.config.logging import configure_logging, get_logger
from core.config.settings import BaseSettings

__all__ = ["BaseSettings", "configure_logging", "get_logger"]
