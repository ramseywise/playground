from __future__ import annotations

from pathlib import Path

import httpx
import structlog

log = structlog.get_logger(__name__)

TIMEOUT = 60.0  # Pollinations can be slow on first request


def fetch_image(url: str, dest: Path) -> Path:
    """Fetch an image from Pollinations and write it to dest. Returns the path."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    log.info("image.fetch.start", url=url, dest=str(dest))
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    dest.write_bytes(response.content)
    log.info("image.fetch.done", dest=str(dest), size_kb=len(response.content) // 1024)
    return dest


def fetch_images_for_slides(
    viz_prompts: list,  # list[VizPrompt] — avoid circular import
    output_dir: Path,
    deck_slug: str,
) -> dict[int, Path]:
    """Fetch all non-skipped slide images. Returns {slide_number: image_path}."""
    results: dict[int, Path] = {}

    for vp in viz_prompts:
        if vp.skip_image or not vp.pollinations_url:
            continue
        dest = output_dir / f"{deck_slug}_slide_{vp.slide_number:02d}.png"
        try:
            path = fetch_image(vp.pollinations_url, dest)
            results[vp.slide_number] = path
        except httpx.HTTPError as exc:
            log.error("image.fetch.failed", slide=vp.slide_number, error=str(exc))

    return results


def fetch_single_image(url: str, output_dir: Path, filename: str) -> Path:
    """Fetch a single image for image-only mode."""
    dest = output_dir / filename
    return fetch_image(url, dest)
