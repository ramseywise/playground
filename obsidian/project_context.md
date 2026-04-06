# Active Project Context

Used by the research agent to connect PDF findings to your actual work.
Update this file as projects evolve.

## agents (this repo)

- **Research agent**: PDF -> chunked notes -> Obsidian vault. Uses Claude API (anthropic SDK), pdftotext, structlog.
- **Visualizer agent**: Interactive presentation builder -> PPTX via python-pptx.
- Stack: Python 3.12+, uv, ruff, pydantic v2, structlog, anthropic SDK.
- Tools pattern: `StructuredTool.from_function` used in `tools.py` for agent tool definitions.
- Eval: `tool_selection_eval.py` uses `return_intermediate_steps=True` to extract tool calls for evaluation.
- Vector store: ChromaDB with metadata filtering (`where={"key": ...}`) for entity-scoped retrieval.
- Orchestration: Moving from LangChain AgentExecutor to LangGraph for multi-agent workflows.

## Obsidian knowledge base

- Vault at `~/workspace/obsidian`, topics: rag, knowledge-graphs, llm-fundamentals, agentic-ai, ml-ops.
- Notes are used for: project planning reference, cross-document relationship mapping, future RAG retrieval.
- Wikilinks and tags drive graph view and sparse retrieval.

## Key patterns to watch for

- ReAct / tool-use agent patterns (directly applicable to this toolkit)
- Prompt engineering for tool selection and schema injection
- Knowledge graph construction and querying (Neo4j, Cypher)
- Embedding model selection and chunking strategies
- Deprecated API patterns in LangChain ecosystem (AgentExecutor -> LangGraph)
