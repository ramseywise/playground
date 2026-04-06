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

### Step 4 — Renderer Readability
- Updated `src/agents/presenter/renderer.py` — added dark scrim overlay (`_add_image_slide`): solid black rectangle at `Inches(4.5)–Inches(7.5)` with 40% opacity via `a:alpha val="40000"` XML manipulation; added `SCRIM_ALPHA`, `SCRIM_TOP`, `SCRIM_HEIGHT` constants; added `lxml` and `pptx.oxml.ns.qn` imports
- Updated `render_deck` — graceful degradation: image slides with no entry in `image_map` (failed fetch) fall back to `_add_text_slide` instead of rendering blank
- Created `tests/presenter/test_renderer.py` — 5 tests (slide count, scrim present, scrim alpha value, text-only no scrim, white headline text); uses Pillow for minimal test PNGs
- Tests: `tests/presenter/` — 36 tests, all passing
- Deviations: none

### Step 5 — Error Handling & CLI
- Updated `src/agents/utils/client.py` — added `parse_json_response(client, response_text, model, system)`: parses JSON with one retry on `JSONDecodeError`; added `json`, `Any`, `structlog` imports
- Updated `src/agents/presenter/providers.py` — added `_http_get_with_retry(url)` helper with exponential backoff (3 retries, 2s/4s/8s delays); `PollinationsProvider` and `ReplicateProvider` both use it; added `time`, `MAX_RETRIES`, `RETRY_DELAYS` constants
- Updated `src/agents/presenter/outline.py` — uses `parse_json_response` instead of `json.loads` in both `generate_outline` and `_apply_edit`; removed unused `import json`
- Updated `src/agents/presenter/slide_writer.py` — uses `parse_json_response`; removed unused `import json`
- Updated `src/agents/presenter/viz_classifier.py` — uses `parse_json_response` in `_generate_scene_description`, `_translate_to_image_prompt`, `propose_image_concepts`; removed unused `import json`
- Updated `src/agents/presenter/__main__.py` — added `--dry-run` flag (skips image fetch + render, prints slide content) and `--provider` flag (overrides `settings.image_provider`); added `dry_run: bool = False` param to `run_deck()`
- Created `tests/presenter/test_error_handling.py` — 11 tests (JSON valid/fenced/retry/system/double-fail/list, HTTP retry delays/no-sleep/error, renderer fallback/scrim-present)
- Tests: `tests/presenter/` — 47 tests, all passing
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
