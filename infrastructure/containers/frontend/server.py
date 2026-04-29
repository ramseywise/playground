"""Web client server for VA assistant — text (SSE) and voice (WebSocket)."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import pathlib
import re
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

load_dotenv()

from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# AGENTS_ROOT_DIR must contain an `agents/` package (e.g. va-google-adk/).
# Defaults to ../../va-google-adk relative to this file when running locally.
_AGENTS_ROOT = pathlib.Path(
    os.getenv(
        "AGENTS_ROOT_DIR",
        str(pathlib.Path(__file__).parent.parent.parent / "va-google-adk"),
    )
)
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))

from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_serialize(obj: Any) -> Any:
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        serialized = json.dumps(obj, default=str)
        return json.loads(serialized)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"


_AGENTS_DIR = _AGENTS_ROOT / "agents"
_SKIP_DIRS = {"__pycache__", "shared"}
_WEB_CLIENT_DIR = pathlib.Path(__file__).parent


def _load_web_client_config() -> dict:
    cfg_path = _WEB_CLIENT_DIR / "config.json"
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


_WEB_CLIENT_CONFIG = _load_web_client_config()


def _is_live_model(agent_py: pathlib.Path) -> bool:
    for line in agent_py.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"gemini-\S*live\S*", stripped, re.IGNORECASE):
            return True
    return False


def _supports_text_input(agent_py: pathlib.Path) -> bool:
    for line in agent_py.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"gemini-\S*live\S*preview", stripped, re.IGNORECASE):
            return False
    return True


def _discover_agents() -> list[dict[str, Any]]:
    excluded = set(_WEB_CLIENT_CONFIG.get("excluded_agents", []))
    agents = []
    for d in sorted(_AGENTS_DIR.iterdir()):
        if not d.is_dir() or d.name in _SKIP_DIRS or d.name.startswith("_"):
            continue
        if d.name in excluded:
            continue
        agent_py = d / "agent.py"
        if not agent_py.exists():
            continue
        agents.append(
            {
                "name": d.name,
                "is_live": _is_live_model(agent_py),
                "supports_text": _supports_text_input(agent_py),
            }
        )
    return agents


_AGENT_CACHE: dict[str, dict[str, Any]] = {}


def _load_agent(name: str) -> dict[str, Any]:
    if name in _AGENT_CACHE:
        return _AGENT_CACHE[name]

    root_agent = None
    try:
        mod = importlib.import_module(f"agents.{name}.agent")
        root_agent = mod.root_agent
        logger.info("Loaded %s via agent.py", name)
    except (ImportError, AttributeError):
        pass

    if root_agent is None:
        raise RuntimeError(f"Could not load agent '{name}' from {_AGENTS_DIR}")

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=name,
        session_service=session_service,
        auto_create_session=True,
    )

    is_live = _is_live_model(_AGENTS_DIR / name / "agent.py")

    entry = {"runner": runner, "is_live": is_live}
    _AGENT_CACHE[name] = entry
    return entry


@asynccontextmanager
async def _lifespan(_: FastAPI):
    warmup_cfg = _WEB_CLIENT_CONFIG.get("warmup", {})
    for name in warmup_cfg.get("agents", []):
        try:
            _load_agent(name)
            logger.info("Warmup: loaded agent '%s'", name)
        except Exception:
            logger.exception("Warmup: failed to load agent '%s'", name)
    yield


app = FastAPI(title="VA Assistant", lifespan=_lifespan)
_STATIC_DIR = _WEB_CLIENT_DIR / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    from fastapi.responses import FileResponse

    return FileResponse(
        str(_STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/log")
async def client_log(payload: dict):
    level = payload.get("level", "error")
    msg = payload.get("message", "")
    ctx = payload.get("context", "")
    line = f"[browser] {msg}" + (f" | {ctx}" if ctx else "")
    if level == "warn":
        logger.warning(line)
    elif level == "info":
        logger.info(line)
    else:
        logger.error(line)
    return {}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response

    return Response(status_code=204)


@app.get("/agents")
async def list_agents():
    return {
        "agents": _discover_agents(),
        "default_agent": _WEB_CLIENT_CONFIG.get("default_agent"),
    }


class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: str = "web-user"


@app.post("/chat/{agent_name}")
async def chat(agent_name: str, req: ChatRequest):
    entry = _load_agent(agent_name)
    runner: Runner = entry["runner"]

    new_message = types.Content(
        role="user",
        parts=[types.Part(text=req.message)],
    )
    stream_config = RunConfig(streaming_mode=StreamingMode.SSE)

    async def event_stream():
        try:
            async for event in runner.run_async(
                user_id=req.user_id,
                session_id=req.session_id,
                new_message=new_message,
                run_config=stream_config,
            ):
                if event.content and event.content.parts:
                    if not event.partial:
                        calls = [
                            {
                                "name": p.function_call.name,
                                "args": _safe_serialize(
                                    dict(p.function_call.args or {})
                                ),
                            }
                            for p in event.content.parts
                            if p.function_call
                        ]
                        responses = [
                            {
                                "name": p.function_response.name,
                                "response": _safe_serialize(
                                    p.function_response.response
                                ),
                            }
                            for p in event.content.parts
                            if p.function_response
                        ]
                        if calls:
                            yield f"data: {json.dumps({'type': 'tool_calls', 'calls': calls})}\n\n"
                        if responses:
                            yield f"data: {json.dumps({'type': 'tool_responses', 'responses': responses})}\n\n"
                    for part in event.content.parts:
                        if part.text:
                            data = json.dumps(
                                {
                                    "type": "text",
                                    "content": part.text,
                                    "partial": bool(event.partial),
                                }
                            )
                            yield f"data: {data}\n\n"
                if event.turn_complete:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.exception("Error in chat stream for %s", agent_name)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/{agent_name}")
async def live_chat(
    websocket: WebSocket,
    agent_name: str,
    session_id: str = "default",
    user_id: str = "web-user",
):
    await websocket.accept()
    entry = _load_agent(agent_name)
    runner: Runner = entry["runner"]

    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        max_llm_calls=20,
    )
    live_request_queue = LiveRequestQueue()

    async def upstream():
        try:
            while True:
                msg = await websocket.receive()
                if "bytes" in msg and msg["bytes"]:
                    audio_blob = types.Blob(
                        mime_type="audio/pcm;rate=16000",
                        data=msg["bytes"],
                    )
                    live_request_queue.send_realtime(audio_blob)
                elif "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("type") == "text":
                        content = types.Content(
                            role="user",
                            parts=[types.Part(text=data["content"])],
                        )
                        live_request_queue.send_content(content=content)
                    elif data.get("type") == "close":
                        break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            live_request_queue.close()

    async def downstream():
        _turn_calls: list[str] = []
        _turn_texts: list[str] = []
        try:
            async for event in runner.run_live(
                user_id=user_id,
                session_id=session_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                if event.content and event.content.parts:
                    for p in event.content.parts:
                        if p.function_call:
                            _turn_calls.append(p.function_call.name)
                        if p.text and not event.partial:
                            _turn_texts.append(p.text)

                if event.turn_complete:
                    _turn_calls.clear()
                    _turn_texts.clear()

                if event.input_transcription and event.input_transcription.text:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "transcription",
                                "role": "user",
                                "content": event.input_transcription.text,
                                "finished": bool(event.input_transcription.finished),
                            }
                        )
                    )

                if event.output_transcription and event.output_transcription.text:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "transcription",
                                "role": "model",
                                "content": event.output_transcription.text,
                                "finished": bool(event.output_transcription.finished),
                            }
                        )
                    )

                if event.content and event.content.parts:
                    if not event.partial:
                        calls = [
                            {
                                "name": p.function_call.name,
                                "args": _safe_serialize(
                                    dict(p.function_call.args or {})
                                ),
                            }
                            for p in event.content.parts
                            if p.function_call
                        ]
                        responses = [
                            {
                                "name": p.function_response.name,
                                "response": _safe_serialize(
                                    p.function_response.response
                                ),
                            }
                            for p in event.content.parts
                            if p.function_response
                        ]
                        if calls:
                            await websocket.send_text(
                                json.dumps({"type": "tool_calls", "calls": calls})
                            )
                        if responses:
                            await websocket.send_text(
                                json.dumps(
                                    {"type": "tool_responses", "responses": responses}
                                )
                            )
                    for part in event.content.parts:
                        if part.text:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "text",
                                        "content": part.text,
                                        "partial": bool(event.partial),
                                    }
                                )
                            )
                        if part.inline_data and part.inline_data.data:
                            await websocket.send_bytes(part.inline_data.data)

                if event.turn_complete:
                    await websocket.send_text(json.dumps({"type": "turn_complete"}))

        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as e:
            if "1000" in str(e):
                logger.debug("Live session closed normally for %s", agent_name)
            else:
                logger.exception("Error in live downstream for %s", agent_name)
                try:
                    await websocket.send_text(
                        json.dumps({"type": "error", "content": str(e)})
                    )
                except Exception:
                    pass
        finally:
            live_request_queue.close()

    try:
        await asyncio.gather(upstream(), downstream())
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        logger.exception("Live session error for %s", agent_name)
    finally:
        live_request_queue.close()


if __name__ == "__main__":
    port = int(os.getenv("FRONTEND_PORT", "3000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
