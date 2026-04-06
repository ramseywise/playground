# Changelog — Visualizer Improvements

## [Unreleased]

### Step 1 — Config & Models Foundation
- Created `src/agents/visualizer/models.py` — all Pydantic models (DeckIntake, ImageIntake, SlideOutline, DeckOutline, SlideContent, VizPrompt, ImageConcept) extracted here
- Added 11 visualizer fields to `src/agents/shared/config.py` Settings (image_provider, pollinations_model, pollinations_seed, pollinations_enhance, replicate_api_token, viz_output_dir, image_width, image_height, viz_audience, viz_model)
- Replaced `anthropic.Anthropic()` with `create_client()` in outline.py, slide_writer.py, viz_classifier.py
- Updated imports in intake.py, outline.py, slide_writer.py, viz_classifier.py, __main__.py to use models.py and settings
- Deleted `src/agents/visualizer/config.yaml` — all config now via shared Settings
- Updated test_outline.py and test_viz_classifier.py imports to use models.py
- Created `tests/visualizer/test_config.py` — 2 tests for Settings defaults and overrides
- Tests: `tests/visualizer/` — 10 tests (8 existing + 2 new), all passing
- Deviations: none

### Step 2 — Image Provider System
- Created `src/agents/visualizer/providers.py` — `ImageProvider` Protocol, `PollinationsProvider` (with model/seed/enhance params), `ReplicateProvider` (lazy import), `get_provider()` factory
- Rewrote `src/agents/visualizer/image_fetcher.py` — provider-based dispatch with `ThreadPoolExecutor` parallel fetching (max 4 workers)
- Updated `src/agents/visualizer/__main__.py` — image-only mode uses `filled_prompt` instead of `pollinations_url`
- Updated `tests/visualizer/test_viz_classifier.py` — `_build_url` test now tests `PollinationsProvider._build_url`
- Added `[project.optional-dependencies] replicate` to `pyproject.toml`
- Created `tests/visualizer/test_providers.py` — 9 tests (URL building, generate_image, import guard, factory)
- Created `tests/visualizer/test_image_fetcher.py` — 4 tests (skip, parallel, failure handling, single)
- Tests: `tests/visualizer/` — 23 tests, all passing
- Deviations: none

### Step 3 — Prompt Engineering Pipeline
- Rewrote `src/agents/visualizer/viz_classifier.py` — two-pass strategy: `_generate_scene_description()` (pass 1) + `_translate_to_image_prompt()` (pass 2)
- Added `deck_style_context` parameter to `classify_slides()` for visual coherence across slides
- Expanded `viz_prompt_library.yaml` — 3 new viz types (timeline, comparison, process_flow), richer templates with more variables, `scene_hint` field added to all types
- Enhanced `SLIDE_SYSTEM` in `slide_writer.py` — image_brief now requests 2-3 sentences with cinematic detail
- Updated `__main__.py` — passes deck style context (title, audience, tone, coherence instructions) to `classify_slides`
- Rewrote `tests/visualizer/test_viz_classifier.py` — 11 tests (5 model + 6 two-pass prompting)
- Created `tests/visualizer/test_slide_writer.py` — 3 tests (JSON parsing, system prompt check, null brief)
- Tests: `tests/visualizer/` — 31 tests, all passing
- Deviations: none
