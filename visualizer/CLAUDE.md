# Visualizer — Presentation Authoring Agent

## Stack
- Python 3.11+, uv, ruff, pydantic v2, structlog
- Claude API (`claude-sonnet-4-20250514`) for all reasoning
- `python-pptx` for slide rendering
- Pollinations.ai for image generation (no API key)

## Commands
- `uv run python -m agent.main` — run full deck workflow
- `uv run python -m agent.main --image-only` — image-only workflow
- `uv run pytest tests/unit/` — unit tests

## Key files
- `config.yaml` — saved defaults (audience, template path, output dirs)
- `agent/viz_prompt_library.yaml` — image prompt templates by slide type
- `agent/template.pptx` — base slide template (do not modify programmatically)
- `agent/output/slides/` — rendered decks
- `agent/output/images/` — standalone generated images

## Workflow modes
1. **Deck**: intake → outline (approval checkpoint) → slide content → viz classification → image fetch → pptx render
2. **Image-only**: goal + description → Claude proposes 2-3 concepts → user picks → generate (with revision loop)

## Config defaults
Audience and template use are saved in `config.yaml` and can be overridden at runtime via CLI flags or interactive prompts.
