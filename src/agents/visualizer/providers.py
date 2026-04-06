"""Image generation providers — Pollinations (free) and Replicate (paid, optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import httpx
import structlog

from agents.shared.config import settings

log = structlog.get_logger(__name__)

FETCH_TIMEOUT = 60.0


class ImageProvider(Protocol):
    """Interface for image generation providers."""

    def generate_image(
        self, prompt: str, dest: Path, width: int, height: int
    ) -> Path:
        """Generate an image from a text prompt and save to dest. Returns the path."""
        ...


class PollinationsProvider:
    """Free image generation via Pollinations.ai — no API key required."""

    def __init__(
        self,
        model: str = "flux",
        seed: int | None = None,
        enhance: bool = False,
    ) -> None:
        self.model = model
        self.seed = seed
        self.enhance = enhance

    def _build_url(self, prompt: str, width: int, height: int) -> str:
        encoded = quote(prompt)
        params = f"width={width}&height={height}&model={self.model}&nologo=true"
        if self.seed is not None:
            params += f"&seed={self.seed}"
        if self.enhance:
            params += "&enhance=true"
        return f"https://image.pollinations.ai/prompt/{encoded}?{params}"

    def generate_image(
        self, prompt: str, dest: Path, width: int, height: int
    ) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = self._build_url(prompt, width, height)

        log.info("pollinations.fetch.start", url=url[:120], dest=str(dest))
        with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

        dest.write_bytes(response.content)
        log.info(
            "pollinations.fetch.done",
            dest=str(dest),
            size_kb=len(response.content) // 1024,
        )
        return dest


class ReplicateProvider:
    """Paid image generation via Replicate — requires API token and `replicate` package."""

    def __init__(
        self,
        api_token: str,
        model: str = "black-forest-labs/flux-schnell",
    ) -> None:
        try:
            import replicate as _replicate
        except ImportError:
            raise RuntimeError(
                "Replicate provider requires the replicate package. "
                "Install it with: uv add replicate"
            )
        self._replicate = _replicate
        self._client = _replicate.Client(api_token=api_token)
        self.model = model

    def generate_image(
        self, prompt: str, dest: Path, width: int, height: int
    ) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)

        log.info("replicate.run.start", model=self.model, dest=str(dest))
        output = self._client.run(
            self.model,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
            },
        )

        # Replicate returns a list of FileOutput or URL strings
        image_url = output[0] if isinstance(output, list) else output
        url_str = str(image_url)

        with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            response = client.get(url_str)
            response.raise_for_status()

        dest.write_bytes(response.content)
        log.info(
            "replicate.fetch.done",
            dest=str(dest),
            size_kb=len(response.content) // 1024,
        )
        return dest


def get_provider() -> ImageProvider:
    """Create the image provider configured in settings."""
    provider_name = settings.image_provider

    if provider_name == "replicate":
        if not settings.replicate_api_token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN not set — add it to .env or use image_provider=pollinations"
            )
        return ReplicateProvider(api_token=settings.replicate_api_token)

    # Default: pollinations
    return PollinationsProvider(
        model=settings.pollinations_model,
        seed=settings.pollinations_seed,
        enhance=settings.pollinations_enhance,
    )
