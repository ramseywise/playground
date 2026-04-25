"""Observability config for va-langgraph.

LangSmith (active)
------------------
LangGraph auto-instruments when these env vars are set — no code needed here:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_pt_...
    LANGSMITH_PROJECT=billy-va
    LANGSMITH_ENDPOINT=https://api.smith.langchain.com

Every node, model call, and tool invocation is captured automatically.

Langfuse (reference — swap back by replacing .env vars and uncommenting below)
-------------------------------------------------------------------------------
# from langfuse import Langfuse
# from langfuse.callback import CallbackHandler
#
# _langfuse_client = None
#
# def init_langfuse() -> None:
#     global _langfuse_client
#     if os.getenv("LANGFUSE_ENABLED", "").lower() != "true":
#         return
#     public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
#     secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
#     if not public_key or not secret_key:
#         return
#     _langfuse_client = Langfuse(
#         public_key=public_key,
#         secret_key=secret_key,
#         host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
#     )
#
# def shutdown_langfuse() -> None:
#     if _langfuse_client is not None:
#         _langfuse_client.flush()
#
# def get_callback_handler(trace_id, user_id, session_id):
#     if _langfuse_client is None:
#         return None
#     return CallbackHandler(
#         public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
#         secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
#         host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
#         trace_id=trace_id, user_id=user_id, session_id=session_id,
#     )
#
# In gateway/main.py lifespan: init_langfuse() / shutdown_langfuse()
# In gateway/runner.py config:  "callbacks": [get_callback_handler(...)] or []
"""
