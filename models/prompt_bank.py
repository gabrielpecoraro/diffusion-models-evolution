"""Curated prompts for benchmarking diffusion model capabilities.

Each prompt targets a specific capability that improved significantly
with the MMDiT architecture (SD3/FLUX) vs older UNet models.
"""

from dataclasses import dataclass


@dataclass
class PromptSet:
    name: str
    prompt: str
    category: str  # text_rendering | photorealism | spatial | artistic | complex


BENCHMARK_PROMPTS = [
    PromptSet(
        name="text_sign",
        prompt="A corgi holding a wooden sign that reads 'Hello World'",
        category="text_rendering",
    ),
    PromptSet(
        name="astronaut",
        prompt=(
            "A photorealistic astronaut riding a white horse on Mars, "
            "cinematic lighting, detailed spacesuit"
        ),
        category="photorealism",
    ),
    PromptSet(
        name="spatial",
        prompt="A red cube on top of a blue sphere, with a green cylinder behind them",
        category="spatial",
    ),
    PromptSet(
        name="artistic",
        prompt="An oil painting of a sunset over mountains in the style of Monet",
        category="artistic",
    ),
    PromptSet(
        name="complex",
        prompt=(
            "A steampunk clockwork owl perched on an ancient leather-bound book "
            "in a candlelit library"
        ),
        category="complex",
    ),
]


def get_prompts_by_category(category: str) -> list[PromptSet]:
    """Filter benchmark prompts by category."""
    return [p for p in BENCHMARK_PROMPTS if p.category == category]


def get_all_prompt_texts() -> list[str]:
    """Return just the prompt strings for quick iteration."""
    return [p.prompt for p in BENCHMARK_PROMPTS]
