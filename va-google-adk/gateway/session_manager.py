"""Per-session ADK Runner management and SSE event queue.

Each session gets one Runner and one asyncio.Queue.  The POST /chat endpoint
fires a background task that runs the ADK turn and enqueues SSE events; the
GET /chat/stream endpoint dequeues and forwards them to the browser.

SSE/POST ordering: the client MUST open GET /chat/stream before calling
POST /chat.  session_manager guarantees the queue exists as soon as
get_or_create is called, which happens when the stream is opened.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

import structlog  # noqa: E402
import memory as memory_store  # noqa: E402
import observability  # noqa: E402
from app import app as va_app  # noqa: E402
from schema import AssistantResponse  # noqa: E402

log = structlog.get_logger(__name__)

_SENTINEL = object()  # signals stream end


@dataclass
class _Session:
    runner: Runner
    user_id: str = "default"
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._session_svc = InMemorySessionService()

    async def get_or_create(
        self, session_id: str, user_id: str = "default"
    ) -> _Session:
        if session_id not in self._sessions:
            # Load preferences from memory store to inject into initial session state
            prefs = await memory_store.get_top(user_id)
            initial_state: dict = {"user_id": user_id, "user_preferences": prefs}

            # Pre-create the ADK session so initial state (user_id, prefs) is available
            # to callbacks and tools from the very first turn.
            try:
                self._session_svc.create_session(
                    app_name=va_app.name,
                    user_id=user_id,
                    session_id=session_id,
                    state=initial_state,
                )
            except Exception as e:
                log.debug("adk-session-precreate-skipped", reason=str(e))

            runner = Runner(
                app_name=va_app.name,
                agent=va_app.root_agent,
                session_service=self._session_svc,
            )
            self._sessions[session_id] = _Session(runner=runner, user_id=user_id)
        return self._sessions[session_id]

    async def run_turn(
        self,
        session_id: str,
        message: str,
        page_url: str | None = None,
        trace_id: str | None = None,
        user_id: str = "default",
    ) -> None:
        """Execute one ADK turn and push SSE events onto the session queue."""
        session = await self.get_or_create(session_id, user_id)

        # Inject page context as a prefix when provided
        if page_url:
            user_text = f"[User is on page: {page_url}]\n{message}"
        else:
            user_text = message

        content = types.Content(role="user", parts=[types.Part(text=user_text)])

        def _evt(type_: str, data) -> dict:
            e: dict = {"type": type_, "data": data}
            if trace_id:
                e["trace_id"] = trace_id
            return e

        _last_response_message: str | None = None
        lf_trace = observability.start_trace(
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
            input=message,
        )

        try:
            async for event in session.runner.run_async(
                user_id=session.user_id,
                session_id=session_id,
                new_message=content,
            ):
                # Stream text chunks as they arrive
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text and not getattr(part, "thought", False):
                            await session.queue.put(_evt("text", part.text))

                # Final agent response — try to parse as AssistantResponse
                if event.is_final_response():
                    structured = _extract_response(event)
                    await session.queue.put(_evt("response", structured))
                    _last_response_message = structured.get("message")

            if lf_trace:
                lf_trace.finish(output=_last_response_message)

        except Exception as e:
            log.exception("adk-turn-error", session_id=session_id)
            await session.queue.put(_evt("error", str(e)))
            if lf_trace:
                lf_trace.finish(output=None, error=str(e))
        finally:
            await session.queue.put(_SENTINEL)
            if _last_response_message:
                await _save_session_summary(
                    session.user_id, session_id, message, _last_response_message
                )


async def _save_session_summary(
    user_id: str,
    session_id: str,
    user_message: str,
    agent_response: str,
) -> None:
    try:
        from model_factory import resolve_chat_model

        prompt = (
            f"In one sentence, summarise this interaction:\n"
            f"User: {user_message[:200]}\n"
            f"Agent: {agent_response[:200]}"
        )
        resp = await resolve_chat_model("small").ainvoke(prompt)
        summary = resp.content.strip()[:500]
        await memory_store.upsert(user_id, f"session:{session_id}", summary)
    except Exception as e:
        log.warning("session-summary-save-failed", session_id=session_id, error=str(e))


def _extract_response(event) -> dict:
    """Parse the final event into an AssistantResponse dict.

    When a sub-agent has output_schema=AssistantResponse, the model outputs
    JSON.  We try to parse it; if it fails we fall back to plain text wrapped
    in a minimal AssistantResponse.
    """
    text = ""
    if event.content and event.content.parts:
        text = "".join(
            p.text
            for p in event.content.parts
            if p.text and not getattr(p, "thought", False)
        )

    # Check session state for structured output written by output_schema
    structured_from_state: dict | None = None
    try:
        state = event._invocation_context.session.state  # type: ignore[attr-defined]
        if "response" in state:
            raw = state["response"]
            if isinstance(raw, dict):
                structured_from_state = raw
            elif isinstance(raw, str):
                structured_from_state = json.loads(raw)
    except Exception:
        pass

    if structured_from_state:
        try:
            return AssistantResponse(**structured_from_state).model_dump()
        except Exception:
            pass

    # Try parsing the text itself as JSON
    if text.strip().startswith("{"):
        try:
            parsed = json.loads(text.strip())
            return AssistantResponse(**parsed).model_dump()
        except Exception:
            pass

    # Fallback: wrap plain text
    return AssistantResponse(message=text or "(no response)").model_dump()


session_manager = SessionManager()
