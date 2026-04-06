from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog
import yaml
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

console = Console()
log = structlog.get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
AGENT_DIR = Path(__file__).resolve().parent


def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["defaults"]


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


# ---------------------------------------------------------------------------
# Deck workflow
# ---------------------------------------------------------------------------


def run_deck(cfg: dict) -> None:
    from agents.visualizer.image_fetcher import fetch_images_for_slides
    from agents.visualizer.intake import collect_deck_intake
    from agents.visualizer.outline import approval_checkpoint, generate_outline
    from agents.visualizer.renderer import render_deck
    from agents.visualizer.slide_writer import generate_slide_content
    from agents.visualizer.viz_classifier import classify_slides

    model = cfg["model"]
    slides_dir = AGENT_DIR / cfg["output"]["slides_dir"]
    template_path = AGENT_DIR / cfg["template_path"]

    # 1. Intake
    intake = collect_deck_intake()

    # 2. Outline + approval checkpoint
    console.print("\n[bold yellow]Generating outline…[/bold yellow]")
    outline = generate_outline(intake, model)
    outline = approval_checkpoint(outline, model, intake)

    # 3. Slide content
    console.print("\n[bold yellow]Writing slide content…[/bold yellow]")
    slides = generate_slide_content(outline, intake, model)

    # 4. Viz classification + prompt building
    console.print("\n[bold yellow]Building image prompts…[/bold yellow]")
    img_w = cfg["image"]["width"]
    img_h = cfg["image"]["height"]
    viz_prompts = classify_slides(slides, model, img_w, img_h)

    skipped = sum(1 for vp in viz_prompts if vp.skip_image)
    fetching = len(viz_prompts) - skipped
    console.print(
        f"  {fetching} images to fetch, {skipped} slides use text-only layout"
    )

    # 5. Image fetch
    console.print("\n[bold yellow]Fetching images…[/bold yellow]")
    import re

    deck_slug = re.sub(r"[^\w]+", "_", outline.title.lower()).strip("_")[:40]
    image_map = fetch_images_for_slides(viz_prompts, slides_dir, deck_slug)

    # 6. Render
    console.print("\n[bold yellow]Rendering deck…[/bold yellow]")
    tpl = template_path if intake.use_template else None
    out_path = render_deck(outline, slides, image_map, tpl, slides_dir)

    console.print(
        f"\n[bold green]Done![/bold green] Deck saved to [cyan]{out_path}[/cyan]"
    )


# ---------------------------------------------------------------------------
# Image-only workflow
# ---------------------------------------------------------------------------


def run_image(cfg: dict) -> None:
    from agents.visualizer.image_fetcher import fetch_single_image
    from agents.visualizer.intake import collect_image_intake
    from agents.visualizer.viz_classifier import propose_image_concepts

    model = cfg["model"]
    images_dir = AGENT_DIR / cfg["output"]["images_dir"]
    img_w = cfg["image"]["width"]
    img_h = cfg["image"]["height"]

    # 1. Intake
    intake = collect_image_intake()

    # 2. Propose concepts
    console.print("\n[bold yellow]Generating image concepts…[/bold yellow]")
    concepts = propose_image_concepts(
        goal=intake.goal,
        description=intake.description,
        audience=intake.audience,
        tone=intake.tone,
        model=model,
        n=3,
        img_width=img_w,
        img_height=img_h,
    )

    # 3. Display concepts and let user pick
    console.print()
    for i, c in enumerate(concepts, 1):
        console.print(
            f"[bold cyan][{i}][/bold cyan] [bold]{c.label}[/bold] ({c.viz_type})"
        )
        console.print(f"    {c.description}")
        console.print(f"    [dim]Why: {c.rationale}[/dim]")
        console.print()

    while True:
        choice = IntPrompt.ask(
            "Pick a concept", choices=[str(i) for i in range(1, len(concepts) + 1)]
        )
        selected = concepts[choice - 1]

        confirm = Prompt.ask(
            f"Generate [bold]{selected.label}[/bold]? Or [bold]revise[/bold] the prompt first",
            choices=["generate", "revise"],
            default="generate",
        )

        if confirm == "revise":
            revision = Prompt.ask("Describe what to change")
            # Re-propose with the revision appended to description
            updated_description = f"{intake.description}. Revision: {revision}"
            concepts = propose_image_concepts(
                goal=intake.goal,
                description=updated_description,
                audience=intake.audience,
                tone=intake.tone,
                model=model,
                n=3,
                img_width=img_w,
                img_height=img_h,
            )
            console.print()
            for i, c in enumerate(concepts, 1):
                console.print(
                    f"[bold cyan][{i}][/bold cyan] [bold]{c.label}[/bold] ({c.viz_type})"
                )
                console.print(f"    {c.description}")
                console.print(f"    [dim]Why: {c.rationale}[/dim]")
                console.print()
            continue

        break

    if not selected.pollinations_url:
        console.print(
            "[red]Selected concept has no image URL — it may be a skip_image type.[/red]"
        )
        sys.exit(1)

    # 4. Fetch
    import re

    filename_slug = re.sub(r"[^\w]+", "_", selected.label.lower()).strip("_")[:40]
    filename = f"{filename_slug}.png"

    console.print(f"\n[bold yellow]Fetching image…[/bold yellow]")
    out_path = fetch_single_image(selected.pollinations_url, images_dir, filename)

    console.print(
        f"\n[bold green]Done![/bold green] Image saved to [cyan]{out_path}[/cyan]"
    )

    # Revision loop — offer another round
    while True:
        again = Prompt.ask(
            "Generate another variation?", choices=["yes", "no"], default="no"
        )
        if again == "no":
            break

        revision = Prompt.ask("What should change?")
        updated_description = f"{intake.description}. Revision: {revision}"
        new_concepts = propose_image_concepts(
            goal=intake.goal,
            description=updated_description,
            audience=intake.audience,
            tone=intake.tone,
            model=model,
            n=1,
            img_width=img_w,
            img_height=img_h,
        )
        if new_concepts and new_concepts[0].pollinations_url:
            slug = re.sub(r"[^\w]+", "_", new_concepts[0].label.lower()).strip("_")[:40]
            out_path = fetch_single_image(
                new_concepts[0].pollinations_url, images_dir, f"{slug}_v2.png"
            )
            console.print(f"[bold green]Saved:[/bold green] [cyan]{out_path}[/cyan]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _configure_logging()
    cfg = _load_config()

    parser = argparse.ArgumentParser(description="Presentation authoring agent")
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Run in image-only mode (no deck generation)",
    )
    args = parser.parse_args()

    if args.image_only:
        run_image(cfg)
    else:
        run_deck(cfg)


if __name__ == "__main__":
    main()
