# Agents

Personal AI agent toolkit for turning research PDFs into structured Obsidian notes and generating slide decks with AI-driven visuals.

## Agents

### Research Agent
Processes PDFs (books, papers, articles) into structured Obsidian notes with YAML frontmatter, wikilinks, tags, and relevance scoring. Handles multi-chapter documents via TOC-aware chunking.

```bash
uv run python -m agents.research path/to/paper.pdf
uv run python -m agents.research path/to/paper.pdf --dry-run  # preview without writing
uv run python -m agents.research path/to/paper.pdf --topic rag  # override topic folder
```

### Visualizer
Interactive presentation authoring agent. Takes a goal/audience/tone, generates an outline (with human approval checkpoint), writes slide content, generates images via Pollinations.ai, and renders a `.pptx` deck.

```bash
uv run python -m agents.visualizer             # full deck workflow
uv run python -m agents.visualizer --image-only # standalone image generation
```

## Setup

```bash
# Clone and install
git clone <repo-url> && cd agents
cp .env.example .env  # fill in your API key and paths
uv sync

# Run tests
uv run pytest tests/
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model for all LLM calls |
| `READINGS_DIR` | `~/Dropbox/ai_readings` | Source PDF directory |
| `OBSIDIAN_VAULT` | `~/workspace/obsidian` | Obsidian vault output path |
| `PDFTOTEXT_BIN` | `/opt/homebrew/bin/pdftotext` | Path to pdftotext binary |
| `PDFINFO_BIN` | `/opt/homebrew/bin/pdfinfo` | Path to pdfinfo binary |

## Project Structure

```
src/agents/
  shared/           Config, Claude client helpers
  research/         PDF → Obsidian notes pipeline
  visualizer/       Presentation authoring agent
tests/
  research/         Unit tests (mocked Claude calls)
  visualizer/       Unit tests
obsidian/           Curated knowledge corpus (vault)
```
