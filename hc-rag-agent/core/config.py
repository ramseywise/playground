"""App settings — env loading only. Client instantiation lives in src/clients/."""

import os

from dotenv import load_dotenv

load_dotenv()


def _normalize_llm_provider(raw: str) -> str:
    """Map aliases to canonical provider names used in app/clients/llm.py."""
    x = raw.lower().strip().replace("-", "_")
    aliases = {
        "google": "gemini",
        "google_genai": "gemini",
        "genai": "gemini",
    }
    return aliases.get(x, x)


# Logging (see app/core/observability.py)
LOG_LEVEL: str = (os.getenv("LOG_LEVEL") or "INFO").upper()


# LangSmith / LangChain tracing (optional; LangChain reads env at runtime)
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


LANGCHAIN_TRACING_V2: bool = _env_bool("LANGCHAIN_TRACING_V2", default=False)
LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT: str | None = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT: str | None = os.getenv("LANGCHAIN_ENDPOINT")

# Chat LLM — gemini (default) | anthropic | openai; model resolution in src/clients/llm.py
LLM_PROVIDER: str = _normalize_llm_provider(os.getenv("LLM_PROVIDER") or "gemini")


def _embedding_model_and_revision() -> tuple[str, str | None]:
    """Resolve model id: explicit ``EMBEDDING_MODEL`` wins; else ``EMBEDDING_PROFILE`` preset."""
    raw = os.getenv("EMBEDDING_MODEL")
    revision = os.getenv("EMBEDDING_MODEL_REVISION")
    rev: str | None = revision.strip() if revision and revision.strip() else None
    presets: dict[str, str] = {
        "minilm": "all-MiniLM-L6-v2",
        "multilingual": "intfloat/multilingual-e5-large",
    }
    if raw is not None and str(raw).strip():
        return str(raw).strip(), rev
    profile = (os.getenv("EMBEDDING_PROFILE") or "").strip().lower()
    if profile in presets:
        return presets[profile], rev
    return "all-MiniLM-L6-v2", rev


EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION = _embedding_model_and_revision()
EMBEDDING_PROFILE: str = (os.getenv("EMBEDDING_PROFILE") or "").strip().lower()

# Three-store data paths (see playground/data/stores/)
VECTORDB_PATH: str = os.getenv("VECTORDB_PATH") or "../data/stores/vectordb/clara.duckdb"
METADB_PATH: str = os.getenv("METADB_PATH") or "../data/stores/metadb/clara_meta.db"

# Local vector store — duckdb (default) | chroma | opensearch | memory
VECTOR_STORE_BACKEND: str = (
    (os.getenv("VECTOR_STORE_BACKEND") or "duckdb").strip().lower()
)
VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR") or "../data/stores/vectordb"

# OpenSearch (VECTOR_STORE_BACKEND=opensearch) — kNN index for chunk embeddings
OPENSEARCH_HOSTS: str = os.getenv("OPENSEARCH_HOSTS", "http://127.0.0.1:9200")
OPENSEARCH_INDEX: str = os.getenv("OPENSEARCH_INDEX", "rag_chunks")
OPENSEARCH_USER: str | None = os.getenv("OPENSEARCH_USER") or None
OPENSEARCH_PASSWORD: str | None = os.getenv("OPENSEARCH_PASSWORD") or None
OPENSEARCH_USE_SSL: bool = _env_bool("OPENSEARCH_USE_SSL", default=False)
OPENSEARCH_VERIFY_CERTS: bool = _env_bool("OPENSEARCH_VERIFY_CERTS", default=True)
OPENSEARCH_TIMEOUT: int = int(os.getenv("OPENSEARCH_TIMEOUT") or "30")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# RAG pipeline — retriever → rerank → policy (see app/graph/)
# Defaults favor latency (<2s typical): smaller top-k, no extra LLM query rewrite.
RAG_CONFIDENCE_THRESHOLD: float = _env_float("RAG_CONFIDENCE_THRESHOLD", 0.25)
RAG_ENSEMBLE_SCORE_THRESHOLD: float = _env_float("RAG_ENSEMBLE_SCORE_THRESHOLD", 0.4)
# Answer context: cl100k token budget for retrieved passages (see app/graph/context_builder.py)
RAG_ANSWER_CONTEXT_MAX_TOKENS: int = int(
    os.getenv("RAG_ANSWER_CONTEXT_MAX_TOKENS") or "6000"
)
RAG_ANSWER_CONTEXT_MAX_CHUNKS: int = int(
    os.getenv("RAG_ANSWER_CONTEXT_MAX_CHUNKS") or "12"
)
RAG_ENSEMBLE_TOP_K: int = int(os.getenv("RAG_ENSEMBLE_TOP_K") or "8")
RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K") or "5")
# passthrough | cross_encoder | llm_listwise — passthrough is fast; cross_encoder is local;
# llm_listwise uses the chat LLM (extra latency + cost, best quality among rerank options).
RERANKER_BACKEND: str = (os.getenv("RERANKER_BACKEND") or "passthrough").strip().lower()
# Extra LLM call to rewrite into 2–3 locale-specific search queries (better recall, +latency).
RAG_RETRIEVAL_QUERY_TRANSFORM: bool = _env_bool(
    "RAG_RETRIEVAL_QUERY_TRANSFORM",
    default=False,
)

# Policy: scores_only (default) | hybrid — optional LLM probes on borderline scores
_rag_policy_mode = (os.getenv("RAG_POLICY_MODE") or "scores_only").strip().lower()
RAG_POLICY_MODE: str = (
    _rag_policy_mode if _rag_policy_mode in ("scores_only", "hybrid") else "scores_only"
)
# Borderline band lower bound as a fraction of the configured threshold (only hybrid)
RAG_POLICY_HYBRID_BORDER_LOW: float = _env_float("RAG_POLICY_HYBRID_BORDER_LOW", 0.85)

# Post-answer evaluator (extra LLM): off by default
RAG_POST_ANSWER_EVALUATOR: bool = _env_bool("RAG_POST_ANSWER_EVALUATOR", default=False)

# Conversation summarization — fires when messages >= threshold, keeps last N messages
RAG_SUMMARIZATION_ENABLED: bool = _env_bool("RAG_SUMMARIZATION_ENABLED", default=True)
RAG_SUMMARIZATION_THRESHOLD: int = int(os.getenv("RAG_SUMMARIZATION_THRESHOLD") or "8")
RAG_SUMMARIZATION_KEEP: int = int(os.getenv("RAG_SUMMARIZATION_KEEP") or "4")

# Checkpointer — memory (default/tests) | sqlite (local dev) | postgres (staging/prod)
# sqlite:  persists sessions to a local file; single-process only
# postgres: concurrent workers + multi-pod safe; requires DATABASE_URL
CHECKPOINTER_BACKEND: str = (
    (os.getenv("CHECKPOINTER_BACKEND") or "sqlite").strip().lower()
)
SQLITE_PATH: str = os.getenv("SQLITE_PATH") or "data/sessions.db"
DATABASE_URL: str | None = os.getenv("DATABASE_URL") or None

# Orchestrator backend — langgraph (default) | adk
# Switch to "adk" to route all requests through the Google ADK runtime.
ORCHESTRATOR_BACKEND: str = (
    (os.getenv("ORCHESTRATOR_BACKEND") or "langgraph").strip().lower()
)

# Guardrails — applied as FastAPI middleware before any runtime (both backends)
RAG_GUARDRAILS_ENABLED: bool = _env_bool("RAG_GUARDRAILS_ENABLED", default=False)

# API server
ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in (
        os.getenv("ALLOWED_ORIGINS") or "http://localhost:3000,http://localhost:8501"
    ).split(",")
    if o.strip()
]

# Planner — keyword routing by default; set true for LLM structured routing (+latency)
RAG_LLM_PLANNER: bool = _env_bool("RAG_LLM_PLANNER", default=False)
# Use small model for planner (RAG_PLANNER_LIGHT_MODEL=true) — see src/clients/llm.py
RAG_PLANNER_LIGHT_MODEL: bool = _env_bool("RAG_PLANNER_LIGHT_MODEL", default=False)
