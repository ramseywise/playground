from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import anthropic
import yaml
from pydantic import BaseModel

from agents.shared.client import strip_json_fences
from agents.visualizer.slide_writer import SlideContent

LIBRARY_PATH = Path(__file__).resolve().parent / "viz_prompt_library.yaml"

CLASSIFIER_SYSTEM = """You are a visual prompt engineer. Given a slide's content and type,
fill in the template variables for the matching viz type from the prompt library.
Return a JSON object with the variable names as keys and your filled values as strings.
Be specific and visual — these values become an image generation prompt."""

CONCEPT_SYSTEM = """You are a visual creative director. Given a goal, description, audience, and tone,
propose exactly {n} distinct image concepts. Each should be visually distinct in approach.
Return a JSON array of objects, each with:
- "label": short name (3-5 words)
- "viz_type": one of architecture|concept|narrative
- "description": one sentence of what the image depicts
- "rationale": one sentence on why this suits the goal"""

IMAGE_SYSTEM = """You are a visual prompt engineer. Given an image concept and viz type,
fill in the template variables for the matching viz type from the prompt library.
Return a JSON object with the variable names as keys and your filled values as strings."""


def _load_library() -> dict:
    with LIBRARY_PATH.open() as f:
        return yaml.safe_load(f)


class VizPrompt(BaseModel):
    slide_number: int
    viz_type: str
    skip_image: bool
    pollinations_url: str | None = None
    filled_prompt: str | None = None


class ImageConcept(BaseModel):
    label: str
    viz_type: str
    description: str
    rationale: str
    pollinations_url: str | None = None
    filled_prompt: str | None = None


def _build_url(prompt: str, width: int = 1280, height: int = 720) -> str:
    encoded = quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"


def _fill_template(
    client: anthropic.Anthropic,
    model: str,
    template: str,
    style: str,
    context: str,
) -> str:
    """Ask Claude to fill template variables; return the assembled prompt string."""
    import re

    variables = re.findall(r"\{(\w+)\}", template)
    if not variables:
        return f"{template}, {style}"

    user_msg = (
        f"Template: {template}\n"
        f"Variables to fill: {variables}\n\n"
        f"Context:\n{context}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=CLASSIFIER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = strip_json_fences(response.content[0].text)
    filled_vars = json.loads(raw)
    filled = template.format(**{k: filled_vars.get(k, k) for k in variables})
    return f"{filled}, {style}"


def classify_slides(
    slides: list[SlideContent],
    model: str,
    img_width: int = 1280,
    img_height: int = 720,
) -> list[VizPrompt]:
    client = anthropic.Anthropic()
    library = _load_library()
    results: list[VizPrompt] = []

    for slide in slides:
        viz_type = slide.slide_type
        entry = library.get(viz_type, library["concept"])

        if entry["skip_image"] or not slide.image_brief:
            results.append(
                VizPrompt(
                    slide_number=slide.slide_number,
                    viz_type=viz_type,
                    skip_image=True,
                )
            )
            continue

        context = (
            f"Slide headline: {slide.headline}\n"
            f"Bullet points: {', '.join(slide.body)}\n"
            f"Image brief: {slide.image_brief}"
        )

        filled = _fill_template(
            client, model, entry["template"], entry["style"], context
        )
        url = _build_url(filled, img_width, img_height)

        results.append(
            VizPrompt(
                slide_number=slide.slide_number,
                viz_type=viz_type,
                skip_image=False,
                pollinations_url=url,
                filled_prompt=filled,
            )
        )

    return results


def propose_image_concepts(
    goal: str,
    description: str,
    audience: str,
    tone: str,
    model: str,
    n: int = 3,
    img_width: int = 1280,
    img_height: int = 720,
) -> list[ImageConcept]:
    client = anthropic.Anthropic()
    library = _load_library()

    system = CONCEPT_SYSTEM.format(n=n)
    user_msg = (
        f"Goal: {goal}\n"
        f"Description: {description}\n"
        f"Audience: {audience}\n"
        f"Tone: {tone}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = strip_json_fences(response.content[0].text)
    concepts = [ImageConcept(**item) for item in json.loads(raw)]

    # Fill prompts for each concept
    for concept in concepts:
        entry = library.get(concept.viz_type, library["concept"])
        if entry["skip_image"]:
            continue
        context = (
            f"Goal: {goal}\nDescription: {description}\nConcept: {concept.description}"
        )
        filled = _fill_template(
            client, model, entry["template"], entry["style"], context
        )
        concept.filled_prompt = filled
        concept.pollinations_url = _build_url(filled, img_width, img_height)

    return concepts
