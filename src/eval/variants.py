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
# Registry — keyed by variant name used in pytest parametrize
# ---------------------------------------------------------------------------
VARIANTS: dict[str, LibrarySettings] = {
    "librarian": LIBRARIAN,
    "raptor": RAPTOR,
    "bedrock": BEDROCK,
}
