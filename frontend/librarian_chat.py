"""Streamlit playground for testing the Librarian RAG agent.

Run:
    streamlit run frontend/librarian_chat.py

Requires the FastAPI server running at LIBRARIAN_API_URL (default http://localhost:8000).
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_URL = os.environ.get("LIBRARIAN_API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_URL}/api/v1/chat"
STREAM_ENDPOINT = f"{API_URL}/api/v1/chat/stream"
HEALTH_ENDPOINT = f"{API_URL}/health"

st.set_page_config(
    page_title="Librarian RAG Playground", page_icon="📚", layout="wide"
)

# Initialise session state before any widget reads it
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata" not in st.session_state:
    st.session_state.metadata = []


@st.cache_data(ttl=30)
def _check_api_health() -> tuple[bool, int | None]:
    """Returns (ok, status_code). Cached for 30 s to avoid blocking on every render."""
    try:
        resp = httpx.get(HEALTH_ENDPOINT, timeout=3)
        return resp.status_code == 200, resp.status_code
    except httpx.ConnectError:
        return False, None


# ---------------------------------------------------------------------------
# Response helpers (must be defined before use)
# ---------------------------------------------------------------------------


def _sync_response(query: str, backend: str) -> str:
    """Non-streaming: POST to /chat and display the full response."""
    with st.spinner("Thinking..."):
        try:
            resp = httpx.post(
                CHAT_ENDPOINT,
                json={"query": query, "backend": backend},
                timeout=60,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            st.error(f"API error: {e}")
            return ""

    data = resp.json()
    response_text = data.get("response", "")
    st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.metadata.append(
        {
            "confidence_score": data.get("confidence_score", 0),
            "intent": data.get("intent", ""),
            "citations": data.get("citations", []),
            "backend": data.get("backend", backend),
        }
    )
    return response_text


def _stream_response(query: str, backend: str) -> str:
    """Streaming: consume SSE from /chat/stream and display tokens live."""
    placeholder = st.empty()
    status_placeholder = st.empty()
    full_text = ""
    event_type = ""
    metadata: dict = {}

    try:
        with httpx.stream(
            "POST",
            STREAM_ENDPOINT,
            json={"query": query, "backend": backend},
            timeout=60,
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
                        placeholder.markdown(full_text + "▌")
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
        return ""

    status_placeholder.empty()
    placeholder.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "content": full_text})
    st.session_state.metadata.append(
        {
            "confidence_score": metadata.get("confidence_score", 0),
            "intent": metadata.get("intent", ""),
            "citations": metadata.get("citations", []),
            "backend": metadata.get("backend", backend),
        }
    )
    return full_text


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Librarian RAG Playground")
    st.divider()

    # Connection status (cached 30 s — avoids blocking on every render)
    _api_ok, _api_status = _check_api_health()
    if _api_ok:
        st.success("API connected")
    elif _api_status is not None:
        st.error(f"API returned {_api_status}")
    else:
        st.error(f"Cannot reach API at {API_URL}")

    st.caption(f"API: `{API_URL}`")

    # Backend selector
    backend = st.radio(
        "Backend",
        options=["librarian", "bedrock"],
        format_func=lambda x: "Python RAG (Librarian)" if x == "librarian" else "AWS Bedrock KB",
        index=0,
        help="Switch between our custom RAG pipeline and AWS Bedrock Knowledge Bases.",
    )

    use_streaming = st.toggle(
        "Stream response",
        value=False,
        disabled=backend == "bedrock",
        help="Streaming is only available for the Librarian backend.",
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.metadata = []
        st.rerun()

    st.divider()
    st.subheader("Last response metadata")
    if st.session_state.get("metadata"):
        last = st.session_state.metadata[-1]
        last_backend = last.get("backend", "librarian")
        st.caption(
            f"Backend: **{'Python RAG' if last_backend == 'librarian' else 'AWS Bedrock KB'}**"
        )
        st.metric("Confidence", f"{last.get('confidence_score', 0):.2f}")
        st.text(f"Intent: {last.get('intent', '—')}")
        citations = last.get("citations", [])
        if citations:
            st.subheader("Citations")
            for c in citations:
                _url = c.get("url", "#")
                if not isinstance(_url, str) or not _url.startswith(("http://", "https://")):
                    _url = "#"
                st.markdown(f"- [{c.get('title', 'source')}]({_url})")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask the librarian..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Streaming only supported for librarian backend
        if use_streaming and backend == "librarian":
            _stream_response(prompt, backend)
        else:
            _sync_response(prompt, backend)
