"""Google ADK orchestration — Variants 2 & 3.

Variant 2: ADK wrapping the LangGraph Librarian pipeline (polyglot multi-agent shell)
  - hybrid_agent.py   — ADK BaseAgent delegating to full CRAG pipeline
  - custom_rag_agent.py — ADK Agent with Gemini tool-calling RAG
  - coordinator.py    — multi-agent router between the two

Variant 3: ADK wrapping AWS Bedrock KB (managed RAG baseline)
  - bedrock_agent.py  — ADK BaseAgent delegating to Bedrock RetrieveAndGenerate

Shared:
  - tools.py      — FunctionTools for custom_rag_agent (search, rerank, analyze, condense, escalate)
  - callbacks.py  — observability (structlog + Langfuse)
  - utils.py      — ADK session/event helpers
"""
