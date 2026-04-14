"""Streamlit playground for testing the Librarian RAG agent.

Run:
    streamlit run frontend/librarian_chat.py

Requires the FastAPI server running at LIBRARIAN_API_URL (default http://localhost:8000).

Features:
    - Dynamic backend discovery via /api/v1/backends
    - Side-by-side comparison mode (any subset of backends)
    - SSE streaming for backends that support it
    - Per-backend session continuity
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import streamlit as st

API_URL = os.environ.get("LIBRARIAN_API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_URL}/api/v1/chat"
STREAM_ENDPOINT = f"{API_URL}/api/v1/chat/stream"
HEALTH_ENDPOINT = f"{API_URL}/health"
BACKENDS_ENDPOINT = f"{API_URL}/api/v1/backends"

st.set_page_config(
    page_title="Librarian RAG Playground", page_icon="\U0001f4da", layout="wide"
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "metadata" not in st.session_state:
    st.session_state.metadata: list[dict] = []
if "sessions" not in st.session_state:
    # Per-backend session IDs for conversation continuity
    st.session_state.sessions: dict[str, str | None] = {}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def _check_api_health() -> tuple[bool, int | None]:
    """Returns (ok, status_code). Cached for 30 s."""
    try:
        resp = httpx.get(HEALTH_ENDPOINT, timeout=3)
        return resp.status_code == 200, resp.status_code
    except httpx.ConnectError:
        return False, None


@st.cache_data(ttl=60)
def _fetch_backends() -> list[dict]:
    """Fetch available backends from the API. Cached for 60 s."""
    try:
        resp = httpx.get(BACKENDS_ENDPOINT, timeout=5)
        resp.raise_for_status()
        return resp.json().get("backends", [])
    except (httpx.HTTPError, httpx.ConnectError):
        return []


def _query_backend(query: str, backend_id: str) -> dict:
    """Send a sync chat request to a single backend. Returns the response dict."""
    session_id = st.session_state.sessions.get(backend_id)
    payload: dict = {"query": query, "backend": backend_id}
    if session_id:
        payload["session_id"] = session_id

    resp = httpx.post(CHAT_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Track session continuity
    if data.get("session_id"):
        st.session_state.sessions[backend_id] = data["session_id"]

    return data


def _stream_response(query: str, backend_id: str) -> tuple[str, dict]:
    """Stream SSE from /chat/stream and render tokens live. Returns (text, metadata)."""
    placeholder = st.empty()
    status_placeholder = st.empty()
    full_text = ""
    event_type = ""
    metadata: dict = {}

    session_id = st.session_state.sessions.get(backend_id)
    payload: dict = {"query": query, "backend": backend_id}
    if session_id:
        payload["session_id"] = session_id

    try:
        with httpx.stream(
            "POST", STREAM_ENDPOINT, json=payload, timeout=60,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    raw_data = line[6:]
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        data = raw_data

                    if event_type == "status":
                        stage = (
                            data.get("stage", "")
                            if isinstance(data, dict)
                            else str(data)
                        )
                        status_placeholder.caption(f"Stage: {stage}")
                    elif event_type == "token":
                        chunk = (
                            data.get("text", data)
                            if isinstance(data, dict)
                            else str(data)
                        )
                        full_text += chunk
                        placeholder.markdown(full_text + "\u258c")
                    elif event_type == "done":
                        if isinstance(data, dict):
                            full_text = data.get("response", full_text)
                            metadata = data
                    elif event_type == "error":
                        detail = (
                            data.get("detail", "Unknown error")
                            if isinstance(data, dict)
                            else str(data)
                        )
                        st.error(f"Stream error: {detail}")
    except httpx.HTTPError as e:
        st.error(f"Stream connection error: {e}")
        return "", {}

    status_placeholder.empty()
    placeholder.markdown(full_text)
    return full_text, metadata


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Librarian RAG Playground")
    st.divider()

    # Connection status
    _api_ok, _api_status = _check_api_health()
    if _api_ok:
        st.success("API connected")
    elif _api_status is not None:
        st.error(f"API returned {_api_status}")
    else:
        st.error(f"Cannot reach API at {API_URL}")

    st.caption(f"API: `{API_URL}`")

    # Fetch backend info
    all_backends = _fetch_backends()
    available_backends = [b for b in all_backends if b.get("available")]
    streaming_ids = {b["id"] for b in all_backends if b.get("streaming")}

    # Build label map for display
    backend_labels = {b["id"]: b["label"] for b in all_backends}

    # Mode selector
    compare_mode = st.toggle(
        "Compare backends",
        value=False,
        help="Run the same query against multiple backends side by side.",
    )

    if compare_mode:
        # Multi-select for comparison
        available_ids = [b["id"] for b in available_backends]
        selected_backends = st.multiselect(
            "Backends to compare",
            options=available_ids,
            default=available_ids[:2] if len(available_ids) >= 2 else available_ids,
            format_func=lambda x: backend_labels.get(x, x),
            help="Select 2 or more backends to compare responses.",
        )
        use_streaming = False  # No streaming in comparison mode
    else:
        # Single backend selector
        if available_backends:
            backend_id = st.radio(
                "Backend",
                options=[b["id"] for b in available_backends],
                format_func=lambda x: backend_labels.get(x, x),
                index=0,
            )
            selected_backends = [backend_id]
        else:
            st.warning("No backends available. Is the API running?")
            selected_backends = []
            backend_id = "librarian"

        # Streaming toggle — enabled only for streaming-capable backends
        can_stream = not compare_mode and backend_id in streaming_ids
        use_streaming = st.toggle(
            "Stream response",
            value=False,
            disabled=not can_stream,
            help="Streaming is available for: "
            + ", ".join(backend_labels.get(b, b) for b in streaming_ids)
            if streaming_ids
            else "No backends support streaming.",
        )

    # Unavailable backends info
    unavailable = [b for b in all_backends if not b.get("available")]
    if unavailable:
        with st.expander(f"{len(unavailable)} backend(s) not configured"):
            for b in unavailable:
                st.caption(f"**{b['label']}** (`{b['id']}`)")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.metadata = []
        st.session_state.sessions = {}
        st.rerun()

    st.divider()

    # Last response metadata
    st.subheader("Last response metadata")
    if st.session_state.get("metadata"):
        last = st.session_state.metadata[-1]
        # In comparison mode, metadata is a list of per-backend results
        results = last.get("results", [last])
        for result in results:
            result_backend = result.get("backend", "unknown")
            label = backend_labels.get(result_backend, result_backend)
            st.caption(f"Backend: **{label}**")
            st.metric("Confidence", f"{result.get('confidence_score', 0):.2f}")
            st.text(f"Intent: {result.get('intent', '\u2014')}")
            st.text(f"Latency: {result.get('latency_ms', '\u2014')}")
            citations = result.get("citations", [])
            if citations:
                st.markdown("**Citations:**")
                for c in citations:
                    _url = c.get("url", "#")
                    if not isinstance(_url, str) or not _url.startswith(
                        ("http://", "https://")
                    ):
                        _url = "#"
                    st.markdown(f"- [{c.get('title', 'source')}]({_url})")
            if len(results) > 1:
                st.divider()

    # Eval dashboard link
    st.divider()
    st.markdown(
        "[View Eval Dashboard \u2192](http://localhost:8502)",
        help="Open the eval metrics dashboard (run separately).",
    )


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("comparison"):
            # Render comparison columns
            results = msg["comparison"]
            cols = st.columns(len(results))
            for col, result in zip(cols, results):
                with col:
                    label = backend_labels.get(result["backend"], result["backend"])
                    st.caption(f"**{label}**")
                    st.markdown(result["response"])
        else:
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask the librarian..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if compare_mode and len(selected_backends) >= 2:
            # --- Comparison mode: parallel requests ---
            results: list[dict] = []
            errors: list[str] = []

            with st.spinner(f"Querying {len(selected_backends)} backends..."):
                with ThreadPoolExecutor(max_workers=len(selected_backends)) as pool:
                    futures = {
                        pool.submit(_query_backend, prompt, bid): bid
                        for bid in selected_backends
                    }
                    for future in as_completed(futures):
                        bid = futures[future]
                        try:
                            data = future.result()
                            results.append(
                                {
                                    "backend": data.get("backend", bid),
                                    "response": data.get("response", ""),
                                    "confidence_score": data.get(
                                        "confidence_score", 0
                                    ),
                                    "intent": data.get("intent", ""),
                                    "citations": data.get("citations", []),
                                }
                            )
                        except Exception as e:
                            errors.append(f"{backend_labels.get(bid, bid)}: {e}")

            # Sort results to match selection order
            order = {bid: i for i, bid in enumerate(selected_backends)}
            results.sort(key=lambda r: order.get(r["backend"], 99))

            if errors:
                for err in errors:
                    st.error(err)

            if results:
                cols = st.columns(len(results))
                for col, result in zip(cols, results):
                    with col:
                        label = backend_labels.get(
                            result["backend"], result["backend"]
                        )
                        st.caption(f"**{label}**")
                        st.markdown(result["response"])
                        if result["citations"]:
                            st.divider()
                            for c in result["citations"]:
                                _url = c.get("url", "#")
                                if not isinstance(_url, str) or not _url.startswith(
                                    ("http://", "https://")
                                ):
                                    _url = "#"
                                st.caption(f"[{c.get('title', 'source')}]({_url})")

                st.session_state.messages.append(
                    {"role": "assistant", "content": "", "comparison": results}
                )
                st.session_state.metadata.append({"results": results})
            else:
                st.error("All backends failed.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "All backends failed."}
                )

        elif selected_backends:
            # --- Single backend mode ---
            bid = selected_backends[0]

            if use_streaming and bid in streaming_ids:
                full_text, meta = _stream_response(prompt, bid)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_text}
                )
                st.session_state.metadata.append(
                    {
                        "confidence_score": meta.get("confidence_score", 0),
                        "intent": meta.get("intent", ""),
                        "citations": meta.get("citations", []),
                        "backend": meta.get("backend", bid),
                    }
                )
            else:
                try:
                    data = _query_backend(prompt, bid)
                    response_text = data.get("response", "")
                    st.markdown(response_text)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                    st.session_state.metadata.append(
                        {
                            "confidence_score": data.get("confidence_score", 0),
                            "intent": data.get("intent", ""),
                            "citations": data.get("citations", []),
                            "backend": data.get("backend", bid),
                        }
                    )
                except httpx.HTTPError as e:
                    st.error(f"API error: {e}")
        else:
            st.warning("No backend selected.")
