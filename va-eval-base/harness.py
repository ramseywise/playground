"""Shared HTTP harness: send Clara tickets to all 3 VA services concurrently."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import structlog

from .models import EvalTask, ServiceResponse

log = structlog.get_logger(__name__)

# Service endpoints — must match docker-compose.va.yml port mappings
_SERVICES = {
    "va-langgraph": "http://localhost:8000",
    "va-google-adk": "http://localhost:8001",
    "va-support-rag": "http://localhost:8002",
}


async def _call_va_google_adk(
    client: httpx.AsyncClient, task: EvalTask, session_id: str
) -> ServiceResponse:
    """Call va-google-adk via /chat endpoint (background task + SSE stream)."""
    base_url = _SERVICES["va-google-adk"]
    start = time.time()

    try:
        # Trigger the turn
        chat_response = await client.post(
            f"{base_url}/chat",
            json={
                "session_id": session_id,
                "request_id": task.id,
                "message": task.query,
                "page_url": None,
                "user_id": "eval",
            },
            timeout=5.0,
        )
        chat_response.raise_for_status()

        # Stream SSE events until we get the response
        response_data = None
        async with client.stream(
            "GET", f"{base_url}/chat/stream", params={"session_id": session_id}
        ) as stream:
            async for line in stream.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                    if event.get("type") == "response":
                        response_data = event.get("data")
                    elif event.get("type") == "done":
                        break
                except (json.JSONDecodeError, KeyError):
                    pass

        if not response_data:
            return ServiceResponse(
                service="va-google-adk",
                task_id=task.id,
                raw_response={},
                message="No response from ADK",
                latency_ms=(time.time() - start) * 1000,
                error="No response data received",
            )

        return ServiceResponse(
            service="va-google-adk",
            task_id=task.id,
            raw_response=response_data,
            message=response_data.get("message", ""),
            suggestions=response_data.get("suggestions", []),
            nav_buttons=response_data.get("nav_buttons", []),
            latency_ms=(time.time() - start) * 1000,
        )

    except Exception as e:
        log.error(
            "adk_call_failed",
            task_id=task.id,
            error=str(e),
            elapsed_ms=(time.time() - start) * 1000,
        )
        return ServiceResponse(
            service="va-google-adk",
            task_id=task.id,
            raw_response={},
            message="",
            latency_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def _call_va_langgraph(
    client: httpx.AsyncClient, task: EvalTask, session_id: str
) -> ServiceResponse:
    """Call va-langgraph via /chat endpoint (background task + SSE stream)."""
    base_url = _SERVICES["va-langgraph"]
    start = time.time()

    try:
        # Trigger the turn
        chat_response = await client.post(
            f"{base_url}/chat",
            json={
                "session_id": session_id,
                "request_id": task.id,
                "message": task.query,
                "page_url": None,
                "user_id": "eval",
            },
            timeout=5.0,
        )
        chat_response.raise_for_status()

        # Stream SSE events until we get the response
        response_data = None
        async with client.stream(
            "GET", f"{base_url}/chat/stream", params={"session_id": session_id}
        ) as stream:
            async for line in stream.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                    if event.get("type") == "response":
                        response_data = event.get("data")
                    elif event.get("type") == "done":
                        break
                except (json.JSONDecodeError, KeyError):
                    pass

        if not response_data:
            return ServiceResponse(
                service="va-langgraph",
                task_id=task.id,
                raw_response={},
                message="No response from LangGraph",
                latency_ms=(time.time() - start) * 1000,
                error="No response data received",
            )

        return ServiceResponse(
            service="va-langgraph",
            task_id=task.id,
            raw_response=response_data,
            message=response_data.get("message", ""),
            suggestions=response_data.get("suggestions", []),
            nav_buttons=response_data.get("nav_buttons", []),
            classified_intent=response_data.get("metadata", {}).get(
                "classified_intent"
            ),
            latency_ms=(time.time() - start) * 1000,
        )

    except Exception as e:
        log.error(
            "langgraph_call_failed",
            task_id=task.id,
            error=str(e),
            elapsed_ms=(time.time() - start) * 1000,
        )
        return ServiceResponse(
            service="va-langgraph",
            task_id=task.id,
            raw_response={},
            message="",
            latency_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def _call_va_support_rag(
    client: httpx.AsyncClient, task: EvalTask
) -> ServiceResponse:
    """Call va-support-rag via /api/v1/chat endpoint (synchronous)."""
    base_url = _SERVICES["va-support-rag"]
    start = time.time()

    try:
        response = await client.post(
            f"{base_url}/api/v1/chat",
            json={
                "query": task.query,
                "thread_id": task.id,
                "locale": "de",
                "metadata": {"source": "clara", "ces_rating": task.ces_rating},
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        return ServiceResponse(
            service="va-support-rag",
            task_id=task.id,
            raw_response=data,
            message=data.get("answer", ""),
            suggestions=[],
            nav_buttons=[],
            latency_ms=(time.time() - start) * 1000,
            metadata={
                "citations": data.get("citations", []),
                "mode": data.get("mode"),
                "escalated": data.get("escalated", False),
                "pipeline_error": data.get("pipeline_error", False),
            },
        )

    except Exception as e:
        log.error(
            "rag_call_failed",
            task_id=task.id,
            error=str(e),
            elapsed_ms=(time.time() - start) * 1000,
        )
        return ServiceResponse(
            service="va-support-rag",
            task_id=task.id,
            raw_response={},
            message="",
            latency_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def run_task_on_all_services(
    task: EvalTask, client: httpx.AsyncClient
) -> dict[str, ServiceResponse]:
    """Run a single task on all 3 services concurrently."""
    session_id = f"eval-{task.id}"

    results = await asyncio.gather(
        _call_va_langgraph(client, task, session_id),
        _call_va_google_adk(client, task, session_id),
        _call_va_support_rag(client, task),
        return_exceptions=False,
    )

    return {
        "va-langgraph": results[0],
        "va-google-adk": results[1],
        "va-support-rag": results[2],
    }


async def run_eval_suite(tasks: list[EvalTask], concurrency: int = 3) -> list[dict]:
    """Run all tasks against all services with concurrency control."""
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_semaphore(task: EvalTask, client: httpx.AsyncClient):
        async with semaphore:
            log.info("running_task", task_id=task.id, query=task.query[:50])
            return await run_task_on_all_services(task, client)

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks_coro = [_run_with_semaphore(task, client) for task in tasks]
        results = await asyncio.gather(*tasks_coro)

    return results
