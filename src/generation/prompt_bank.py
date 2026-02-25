"""PCB-defect-specific prompt bank for Stable Diffusion + ControlNet.

Provides carefully crafted positive and negative prompts for each of the
six DeepPCB defect categories.  The prompts describe the defects in the
visual language of close-up PCB inspection photography so that the
diffusion model generates realistic manufacturing-defect imagery.

DeepPCB class mapping (0-indexed):
    0 - open        Broken / interrupted copper trace
    1 - short       Unwanted copper bridge between adjacent traces
    2 - mousebite   Irregular nibble-shaped gap along a trace edge
    3 - spur        Small unwanted copper protrusion from a trace
    4 - copper      Residual copper in an area that should be etched away
    5 - pinhole     Tiny void / hole in an otherwise continuous copper area
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DefectPrompt:
    """A single defect-aware prompt entry.

    Attributes:
        defect_type: Human-readable defect category name.
        class_id: 0-indexed class label matching the DeepPCB convention.
        prompt: Positive text prompt describing the defect scene.
        negative_prompt: Negative text prompt suppressing unwanted artefacts.
    """

    defect_type: str
    class_id: int
    prompt: str
    negative_prompt: str


# ── Shared negative prompt fragment ──────────────────────────────────────
_NEGATIVE_COMMON: str = (
    "blurry, low resolution, watermark, text overlay, cartoon, painting, "
    "illustration, 3d render, out of focus, distorted, oversaturated, "
    "unrealistic colours, human, hand, fingers"
)

# ── Per-defect prompts ───────────────────────────────────────────────────
DEFECT_PROMPTS: List[DefectPrompt] = [
    # 0 — open
    DefectPrompt(
        defect_type="open",
        class_id=0,
        prompt=(
            "close-up macro photograph of a printed circuit board with an open "
            "circuit defect, a copper trace is visibly broken with a clear gap "
            "interrupting the conductive path, green solder mask, sharp detail, "
            "industrial inspection lighting, top-down view, high resolution"
        ),
        negative_prompt=_NEGATIVE_COMMON,
    ),
    # 1 — short
    DefectPrompt(
        defect_type="short",
        class_id=1,
        prompt=(
            "close-up macro photograph of a printed circuit board with a short "
            "circuit defect, an unwanted thin copper bridge connects two adjacent "
            "traces that should be electrically isolated, green solder mask, "
            "sharp detail, industrial inspection lighting, top-down view, "
            "high resolution"
        ),
        negative_prompt=_NEGATIVE_COMMON,
    ),
    # 2 — mousebite
    DefectPrompt(
        defect_type="mousebite",
        class_id=2,
        prompt=(
            "close-up macro photograph of a printed circuit board with a "
            "mousebite defect, an irregular nibble-shaped indentation along "
            "the edge of a copper trace as if a small piece was bitten away, "
            "green solder mask, sharp detail, industrial inspection lighting, "
            "top-down view, high resolution"
        ),
        negative_prompt=_NEGATIVE_COMMON,
    ),
    # 3 — spur
    DefectPrompt(
        defect_type="spur",
        class_id=3,
        prompt=(
            "close-up macro photograph of a printed circuit board with a spur "
            "defect, a small unwanted copper protrusion extends outward from "
            "the edge of a trace, green solder mask, sharp detail, industrial "
            "inspection lighting, top-down view, high resolution"
        ),
        negative_prompt=_NEGATIVE_COMMON,
    ),
    # 4 — copper (spurious copper)
    DefectPrompt(
        defect_type="copper",
        class_id=4,
        prompt=(
            "close-up macro photograph of a printed circuit board with a "
            "spurious copper defect, residual copper remains in an area that "
            "should have been fully etched away leaving exposed substrate, "
            "green solder mask, sharp detail, industrial inspection lighting, "
            "top-down view, high resolution"
        ),
        negative_prompt=_NEGATIVE_COMMON,
    ),
    # 5 — pinhole
    DefectPrompt(
        defect_type="pinhole",
        class_id=5,
        prompt=(
            "close-up macro photograph of a printed circuit board with a "
            "pinhole defect, a tiny circular void is visible in an otherwise "
            "continuous copper fill area, green solder mask, sharp detail, "
            "industrial inspection lighting, top-down view, high resolution"
        ),
        negative_prompt=_NEGATIVE_COMMON,
    ),
]

# ── Generic PCB prompt (useful as a fallback) ───────────────────────────
GENERIC_PCB_PROMPT: str = (
    "close-up macro photograph of a printed circuit board with visible "
    "manufacturing defects on copper traces, green solder mask background, "
    "sharp focus, industrial inspection lighting, top-down view, "
    "high resolution, professional quality"
)

GENERIC_PCB_NEGATIVE_PROMPT: str = _NEGATIVE_COMMON


def get_prompt_by_class_id(class_id: int) -> DefectPrompt:
    """Look up a defect prompt by its 0-indexed class ID.

    Args:
        class_id: Integer in ``[0, 5]``.

    Returns:
        The corresponding :class:`DefectPrompt`.

    Raises:
        ValueError: If *class_id* is out of range.
    """
    for prompt in DEFECT_PROMPTS:
        if prompt.class_id == class_id:
            return prompt
    raise ValueError(
        f"No prompt for class_id={class_id}. Valid range is 0-{len(DEFECT_PROMPTS) - 1}."
    )


def get_prompt_by_defect_type(defect_type: str) -> DefectPrompt:
    """Look up a defect prompt by its human-readable type name.

    Args:
        defect_type: One of ``"open"``, ``"short"``, ``"mousebite"``,
            ``"spur"``, ``"copper"``, ``"pinhole"``.

    Returns:
        The corresponding :class:`DefectPrompt`.

    Raises:
        ValueError: If *defect_type* is not recognised.
    """
    for prompt in DEFECT_PROMPTS:
        if prompt.defect_type == defect_type:
            return prompt
    valid = [p.defect_type for p in DEFECT_PROMPTS]
    raise ValueError(
        f"Unknown defect_type='{defect_type}'. Valid types: {valid}."
    )
