from __future__ import annotations

from agents.librarian.utils.config import settings
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


def get_langfuse_handler(
    session_id: str,
    user_id: str | None = None,
) -> object | None:
    """Return a LangFuse CallbackHandler if LANGFUSE_ENABLED=true, else None.

    Pass the result as: `callbacks=[handler] if handler else []`
    """
    if not settings.langfuse_enabled:
        return None

    try:
        from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
    except ImportError:
        log.error(
            "tracing.langfuse.missing", msg="langfuse not installed; tracing disabled"
        )
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        log.error(
            "tracing.langfuse.missing_keys",
            msg="LANGFUSE_PUBLIC_KEY or SECRET_KEY not set",
        )
        return None

    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        session_id=session_id,
        user_id=user_id,
    )
