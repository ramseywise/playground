from __future__ import annotations

SYSTEM_PROMPT = """\
You are a research assistant building a personal AI engineering knowledge base in Obsidian.
Your notes are used for: (1) project planning reference, (2) cross-document relationship mapping,
(3) future RAG retrieval.

Active domains (use these to judge relevance):
- RAG systems and retrieval architecture
- LangGraph / multi-agent orchestration
- Prompt engineering and instruction tuning
- Knowledge graphs and graph-based retrieval

Rules:
- Use [[wikilinks]] for any concept, paper, tool, or technique that deserves its own note
  (e.g. [[RAG]], [[LangGraph]], [[knowledge graph]], [[HNSW]], [[chain-of-thought]])
- Add #tags inline for key terms to drive Obsidian tag search and sparse retrieval
  (e.g. #rag #knowledge-graph #langgraph #chunking #embeddings)
- Be critical: note what the work does NOT solve and where claims are weak or unsupported
- Rate relevance 1–5 against active domains (5 = directly applicable to current projects)
- For book chapters: note what this chapter adds beyond prior chapters (referenced in prior_summary)
- For multi-chunk documents: connections and open questions should build on prior chunks
- Output ONLY the markdown body — no preamble, no "Here is your note:" framing
"""

NOTE_TEMPLATE = """\
## Core Claim
{core_claim_placeholder}

## Research Questions / Hypotheses
{rq_placeholder}

## Methodology / Approach
{methodology_placeholder}

## Key Findings
{findings_placeholder}

## Tradeoffs & Limitations
{tradeoffs_placeholder}

## Critical Assessment
{critical_placeholder}

## Connections
{connections_placeholder}

## Open Questions
{open_questions_placeholder}
"""

_SECTION_INSTRUCTIONS = """\
Fill ALL sections below. Use this structure exactly — do not rename or skip sections.

## Core Claim
One or two sentences stating the central argument or contribution.

## Research Questions / Hypotheses
For papers: list the explicit research questions or hypotheses.
For book chapters / courses: list the chapter's stated learning goals or the questions it sets out to answer.

## Methodology / Approach
How was the work done? (experimental setup, algorithm design, dataset, system architecture, etc.)
Flag if methodology is weak or underspecified.

## Key Findings
Bullet list of the most important results, insights, or techniques introduced.
Use [[wikilinks]] for key concepts.

## Tradeoffs & Limitations
What does this approach trade off? What does it explicitly NOT solve?
Be specific — vague "future work" mentions don't count.

## Critical Assessment
Your critical read: where are the claims strongest? Where are they weakest or oversold?
Does this contradict anything in the prior summary? Flag contradictions explicitly.
Rate relevance: **Relevance: N/5** (final line of this section, N = 1–5).

## Connections
List related concepts, papers, and tools as [[wikilinks]].
Note HOW they connect (extends, contradicts, enables, is-enabled-by, alternative-to).

## Open Questions
Questions this work raises but does not answer. What would you want to investigate next?
"""


def build_note_prompt(
    chunk_text: str,
    source_title: str,
    doc_type: str,
    prior_summary: str = "",
    existing_vault_topics: list[str] | None = None,
) -> str:
    """Build the user-turn prompt for a single chunk.

    Args:
        chunk_text: Extracted text from this PDF chunk.
        source_title: Human-readable title of the source (filename or chapter title).
        doc_type: One of "paper", "book-chapter", "course", "article".
        prior_summary: Accumulated summary text from previous chunks (empty for first chunk).
        existing_vault_topics: List of topic slugs already in the vault, for grounding connections.
    """
    parts: list[str] = []

    parts.append(f"**Source**: {source_title}")
    parts.append(f"**Type**: {doc_type}")

    if existing_vault_topics:
        topics_str = ", ".join(existing_vault_topics)
        parts.append(f"**Existing vault topics** (ground [[wikilinks]] here where relevant): {topics_str}")

    if prior_summary:
        parts.append(
            f"**Prior chunks summary** (build on this — do not repeat, do reference it for connections and contradictions):\n{prior_summary}"
        )

    parts.append("\n**Text to analyse**:\n")
    parts.append(chunk_text)
    parts.append("\n---")
    parts.append(_SECTION_INSTRUCTIONS)

    return "\n\n".join(parts)


def build_merge_prompt(
    chunk_notes: list[str],
    source_title: str,
    doc_type: str,
) -> str:
    """Build a prompt to merge multiple chunk notes into one cohesive final note.

    Used when a PDF is split into >1 chunk and each chunk produces its own draft.
    """
    joined = "\n\n---CHUNK BOUNDARY---\n\n".join(chunk_notes)
    return (
        f"**Source**: {source_title}\n"
        f"**Type**: {doc_type}\n\n"
        "The following are draft notes from sequential chunks of the same document.\n"
        "Merge them into a single, coherent note. Remove redundancy. "
        "Preserve all [[wikilinks]], #tags, and critical assessments. "
        "Produce one Relevance score (average, rounded) in the Critical Assessment section.\n\n"
        f"{joined}\n\n---\n\n"
        + _SECTION_INSTRUCTIONS
    )
