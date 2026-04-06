from __future__ import annotations

import json

import anthropic
from pydantic import BaseModel

from agents.visualizer.agent.intake import DeckIntake
from agents.visualizer.agent.outline import DeckOutline, SlideOutline

SLIDE_SYSTEM = """You are a slide content writer for technical presentations.
Given a slide's outline and the deck context, produce the final slide content.
Output a JSON object with:
- "headline": punchy, scannable title (max 8 words)
- "body": list of 2-4 bullet strings (concise, no filler)
- "speaker_note": 2-3 sentences the presenter would say
- "image_brief": one sentence describing the ideal supporting visual (or null if slide type is data/code_demo/team)

Tailor language to the audience. Avoid jargon unless the audience is technical."""


class SlideContent(BaseModel):
    slide_number: int
    slide_type: str
    headline: str
    body: list[str]
    speaker_note: str
    image_brief: str | None = None


def generate_slide_content(
    outline: DeckOutline,
    intake: DeckIntake,
    model: str,
) -> list[SlideContent]:
    client = anthropic.Anthropic()
    results: list[SlideContent] = []

    context_block = ""
    if intake.codebase_summary:
        context_block = f"\n\nCodebase context:\n{intake.codebase_summary}"

    deck_context = (
        f"Deck title: {outline.title}\n"
        f"Goal: {intake.goal}\n"
        f"Audience: {intake.audience}\n"
        f"Tone: {intake.tone}"
        f"{context_block}"
    )

    for slide in outline.slides:
        content = _generate_one_slide(client, slide, deck_context, model)
        results.append(content)

    return results


def _generate_one_slide(
    client: anthropic.Anthropic,
    slide: SlideOutline,
    deck_context: str,
    model: str,
) -> SlideContent:
    NO_IMAGE_TYPES = {"data", "code_demo", "team"}

    user_msg = (
        f"{deck_context}\n\n"
        f"Slide {slide.number}: {slide.title}\n"
        f"Type: {slide.type}\n"
        f"Talking points: {', '.join(slide.talking_points)}\n"
        f"Intent: {slide.speaker_note}\n"
    )
    if slide.type in NO_IMAGE_TYPES:
        user_msg += "\nNote: this slide type does not use a generated image — set image_brief to null."

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SLIDE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text
    if raw.strip().startswith("```"):
        raw = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

    data = json.loads(raw)
    return SlideContent(
        slide_number=slide.number,
        slide_type=slide.type,
        **data,
    )
