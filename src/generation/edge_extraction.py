"""Canny edge extraction for ControlNet conditioning.

Extracts structural edge maps from PCB template images so that
ControlNet can preserve the spatial layout of traces, pads, and vias
while the diffusion model synthesises realistic defect textures.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def extract_canny_edges(
    image: Image.Image,
    low: int = 100,
    high: int = 200,
) -> Image.Image:
    """Extract Canny edges from a PCB image.

    The returned image is a 3-channel (RGB) PIL Image where every channel
    is an identical copy of the binary edge map.  This is the format that
    the ``diffusers`` ControlNet pipeline expects as its conditioning
    input.

    Args:
        image: Source PCB image (any mode -- RGB, L, etc.).
        low: Lower hysteresis threshold for ``cv2.Canny``.
        high: Upper hysteresis threshold for ``cv2.Canny``.

    Returns:
        A 3-channel PIL Image of the same spatial dimensions as *image*,
        containing the Canny edge map replicated across R, G, and B.
    """
    arr: np.ndarray = np.array(image)

    # Convert to single-channel grayscale if needed.
    if len(arr.shape) == 3 and arr.shape[2] >= 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif len(arr.shape) == 3 and arr.shape[2] == 1:
        gray = arr[:, :, 0]
    else:
        gray = arr

    edges: np.ndarray = cv2.Canny(gray, low, high)

    # Stack to 3-channel so ControlNet receives an RGB-shaped tensor.
    edges_rgb = np.stack([edges, edges, edges], axis=2)

    logger.debug(
        "Canny edges extracted (low=%d, high=%d) — shape %s",
        low,
        high,
        edges_rgb.shape,
    )
    return Image.fromarray(edges_rgb)


def extract_canny_edges_batch(
    images: list[Image.Image],
    low: int = 100,
    high: int = 200,
) -> list[Image.Image]:
    """Apply :func:`extract_canny_edges` to a list of images.

    Args:
        images: List of PIL Images.
        low: Lower Canny threshold.
        high: Upper Canny threshold.

    Returns:
        List of 3-channel edge-map PIL Images.
    """
    return [extract_canny_edges(img, low, high) for img in images]


def auto_canny_thresholds(
    image: Image.Image,
    sigma: float = 0.33,
) -> Tuple[int, int]:
    """Compute adaptive Canny thresholds based on median pixel intensity.

    Uses the widely-adopted *sigma* heuristic:

        * low  = max(0,   (1 - sigma) * median)
        * high = min(255, (1 + sigma) * median)

    Args:
        image: Source image (converted to grayscale internally).
        sigma: Controls how far from the median the thresholds sit.

    Returns:
        ``(low, high)`` integer thresholds suitable for ``cv2.Canny``.
    """
    arr = np.array(image.convert("L"))
    median_val = float(np.median(arr))
    low = int(max(0, (1.0 - sigma) * median_val))
    high = int(min(255, (1.0 + sigma) * median_val))
    logger.debug(
        "Auto Canny thresholds: median=%.1f, sigma=%.2f -> low=%d, high=%d",
        median_val,
        sigma,
        low,
        high,
    )
    return low, high
