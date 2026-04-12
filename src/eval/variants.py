"""Named retrieval configuration variants for eval comparison.

Presets capture the parameter sets of known pipelines so they can be
benchmarked side-by-side using the librarian eval harness without
duplicating retrieval code.

Three variants:
  librarian — current defaults: hybrid BM25+dense, multilingual-e5-large, CrossEncoder
  raptor    — cs_agent_assist_with_rag params: pure knn, minilm 384-dim, no reranker, k=5
  bedrock   — AWS out-of-the-box approximation: pure knn, no reranker, k=5

Note on bm25_weight / vector_weight:
  InMemoryRetriever (used in tests) always applies equal-weight RRF fusion —
  these fields have no effect in test mode.  They are stored here so that the
  same config objects drive real OpenSearch deployments with the correct weights.
"""

from __future__ import annotations

from librarian.config import LibrarySettings

# ---------------------------------------------------------------------------
# Librarian — current production defaults
# ---------------------------------------------------------------------------
LIBRARIAN = LibrarySettings(
    embedding_provider="multilingual",
    embedding_model="intfloat/multilingual-e5-large",
    retrieval_strategy="inmemory",
    reranker_strategy="cross_encoder",
    retrieval_k=10,
    reranker_top_k=3,
    bm25_weight=0.3,
    vector_weight=0.7,
    confidence_threshold=0.0,
    max_crag_retries=0,
    anthropic_api_key="test",
)

# ---------------------------------------------------------------------------
# Raptor — mirrors cs_agent_assist_with_rag pipeline parameters
#   - Pure knn (no BM25)
#   - all-MiniLM-L6-v2, 384-dim
#   - No reranker (passthrough)
#   - k=5, cosine similarity, OpenSearch HNSW (ef_construction=128, m=24)
# ---------------------------------------------------------------------------
RAPTOR = LibrarySettings(
    embedding_provider="minilm",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_strategy="inmemory",
    reranker_strategy="passthrough",
    retrieval_k=5,
    reranker_top_k=5,
    bm25_weight=0.0,
    vector_weight=1.0,
    confidence_threshold=0.0,
    max_crag_retries=0,
    anthropic_api_key="test",
)

# ---------------------------------------------------------------------------
# Bedrock — AWS Knowledge Bases out-of-the-box baseline
#   In production this routes through BedrockKBClient (RetrieveAndGenerate API).
#   For eval tests it is approximated as pure knn with no reranker —
#   AWS Titan embeddings become MockEmbedder in the test fixture.
#   To run against real Bedrock, set bedrock_knowledge_base_id / bedrock_model_arn.
# ---------------------------------------------------------------------------
BEDROCK = LibrarySettings(
    embedding_provider="minilm",  # approximation — production uses AWS Titan
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_strategy="inmemory",
    reranker_strategy="passthrough",
    retrieval_k=5,
    reranker_top_k=5,
    bm25_weight=0.0,
    vector_weight=1.0,
    confidence_threshold=0.0,
    max_crag_retries=0,
    anthropic_api_key="test",
)

# ---------------------------------------------------------------------------
# Bedrock-live — real AWS Bedrock KB API (RetrieveAndGenerate)
#   Requires BEDROCK_KNOWLEDGE_BASE_ID + BEDROCK_MODEL_ARN env vars.
#   Skipped automatically when not configured.
#   retrieval_strategy="bedrock" is the dispatch signal in run_variant_experiment.
# ---------------------------------------------------------------------------
BEDROCK_LIVE = LibrarySettings(
    embedding_provider="aws_titan",  # documentary — Bedrock handles embeddings
    embedding_model="",  # N/A — managed by Bedrock
    retrieval_strategy="bedrock",  # dispatch signal for experiment runner
    reranker_strategy="passthrough",
    retrieval_k=5,
    reranker_top_k=5,
    bm25_weight=0.0,
    vector_weight=1.0,
    confidence_threshold=0.0,
    max_crag_retries=0,
    anthropic_api_key="test",
    # bedrock_knowledge_base_id, bedrock_model_arn, bedrock_region
    # auto-populate from env vars via pydantic-settings.
)

# ---------------------------------------------------------------------------
# Google ADK-live — real Google Gemini + Vertex AI Search grounding
#   Requires GEMINI_API_KEY + GOOGLE_DATASTORE_ID env vars
#   (or GOOGLE_PROJECT_ID for Vertex AI mode).
#   Skipped automatically when not configured.
#   retrieval_strategy="google_adk" is the dispatch signal in run_variant_experiment.
# ---------------------------------------------------------------------------
GOOGLE_ADK_LIVE = LibrarySettings(
    embedding_provider="google",  # documentary — Google handles embeddings
    embedding_model="",  # N/A — managed by Vertex AI Search
    retrieval_strategy="google_adk",  # dispatch signal for experiment runner
    reranker_strategy="passthrough",
    retrieval_k=5,
    reranker_top_k=5,
    bm25_weight=0.0,
    vector_weight=1.0,
    confidence_threshold=0.0,
    max_crag_retries=0,
    anthropic_api_key="test",
    # google_project_id, google_datastore_id, gemini_api_key
    # auto-populate from env vars via pydantic-settings.
)

# ---------------------------------------------------------------------------
# ADK + Bedrock KB — Bedrock KB accessed via Google ADK agent wrapper
#   Same underlying Bedrock KB API, but routed through ADK's BaseAgent
#   session management.  Tests whether ADK's session/event model adds
#   value over the raw BedrockKBClient.
#   Requires BEDROCK_KNOWLEDGE_BASE_ID + BEDROCK_MODEL_ARN env vars.
#   retrieval_strategy="adk_bedrock" is the dispatch signal.
# ---------------------------------------------------------------------------
ADK_BEDROCK = LibrarySettings(
    embedding_provider="aws_titan",  # documentary — Bedrock handles embeddings
    embedding_model="",  # N/A — managed by Bedrock
    retrieval_strategy="adk_bedrock",  # dispatch signal for experiment runner
    reranker_strategy="passthrough",
    retrieval_k=5,
    reranker_top_k=5,
    bm25_weight=0.0,
    vector_weight=1.0,
    confidence_threshold=0.0,
    max_crag_retries=0,
    anthropic_api_key="test",
    # bedrock_knowledge_base_id, bedrock_model_arn, bedrock_region
    # auto-populate from env vars via pydantic-settings.
)

# ---------------------------------------------------------------------------
# ADK + Custom RAG — Gemini 2.0 Flash with custom retrieval tools
#   Uses the same Chroma/OpenSearch retriever and cross-encoder reranker
#   as the Librarian pipeline, but the LLM decides when to call them.
#   Requires GEMINI_API_KEY env var.
#   retrieval_strategy="adk_custom_rag" is the dispatch signal.
# ---------------------------------------------------------------------------
ADK_CUSTOM_RAG = LibrarySettings(
    embedding_provider="multilingual",
    embedding_model="intfloat/multilingual-e5-large",
    retrieval_strategy="adk_custom_rag",  # dispatch signal for experiment runner
    reranker_strategy="cross_encoder",
    retrieval_k=10,
    reranker_top_k=3,
    bm25_weight=0.3,
    vector_weight=0.7,
    confidence_threshold=0.3,
    max_crag_retries=0,  # LLM decides retries, not the pipeline
    anthropic_api_key="test",
    # gemini_api_key auto-populates from env vars via pydantic-settings.
)

# ---------------------------------------------------------------------------
# ADK + LangGraph Hybrid — full CRAG pipeline inside ADK BaseAgent
#   Uses the same LangGraph pipeline as Option 1 but wrapped in an ADK
#   agent for session management.  Tests whether the ADK wrapper adds
#   value (multi-agent routing, session persistence) over raw LangGraph.
#   retrieval_strategy="adk_hybrid" is the dispatch signal.
# ---------------------------------------------------------------------------
ADK_HYBRID = LibrarySettings(
    embedding_provider="multilingual",
    embedding_model="intfloat/multilingual-e5-large",
    retrieval_strategy="adk_hybrid",  # dispatch signal for experiment runner
    reranker_strategy="cross_encoder",
    retrieval_k=10,
    reranker_top_k=3,
    bm25_weight=0.3,
    vector_weight=0.7,
    confidence_threshold=0.4,
    max_crag_retries=1,
    anthropic_api_key="test",
)

# ---------------------------------------------------------------------------
# Registry — keyed by variant name used in pytest parametrize
# ---------------------------------------------------------------------------
VARIANTS: dict[str, LibrarySettings] = {
    "librarian": LIBRARIAN,
    "raptor": RAPTOR,
    "bedrock": BEDROCK,
    "bedrock-live": BEDROCK_LIVE,
    "google-adk": GOOGLE_ADK_LIVE,
    "adk-bedrock": ADK_BEDROCK,
    "adk-custom-rag": ADK_CUSTOM_RAG,
    "adk-hybrid": ADK_HYBRID,
}
