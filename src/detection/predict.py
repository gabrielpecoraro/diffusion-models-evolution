"""Single-image YOLOv8 inference with bounding-box visualisation.

Designed for interactive use in the Gradio demo: accepts a PIL Image,
runs defect detection, and returns an annotated image together with a
structured list of detections.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Module-level cache so we don't reload the model on every call in Gradio.
_MODEL_CACHE: Dict[str, YOLO] = {}


def _get_model(model_path: str) -> YOLO:
    """Return a cached YOLO model instance, loading it once on first access."""
    if model_path not in _MODEL_CACHE:
        logger.info("Loading model from %s", model_path)
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def detect_defects(
    model_path: Path,
    image: Image.Image,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str = "mps",
) -> Tuple[Image.Image, List[dict]]:
    """Run defect detection on a single PCB image.

    Parameters
    ----------
    model_path : Path
        Path to a trained YOLOv8 ``best.pt`` checkpoint.
    image : PIL.Image.Image
        Input image (RGB).
    confidence : float
        Minimum confidence score to keep a detection.
    iou_threshold : float
        IoU threshold for Non-Maximum Suppression.
    imgsz : int
        Inference resolution (images are resized internally).
    device : str
        Compute device (``"mps"``, ``"cuda"``, ``"cpu"``).

    Returns
    -------
    tuple[PIL.Image.Image, list[dict]]
        * ``annotated_image`` -- the input image with bounding boxes drawn.
        * ``detections`` -- list of dicts, each containing ``class``,
          ``confidence``, and ``bbox`` (xyxy format, pixel coords).

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist.
    ValueError
        If *image* is ``None``.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if image is None:
        raise ValueError("Input image must not be None.")

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    model = _get_model(str(model_path))

    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    result = results[0]

    # Ultralytics .plot() returns a BGR numpy array; convert to RGB PIL Image
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]  # BGR -> RGB
    annotated_image = Image.fromarray(annotated_rgb)

    # Build structured detection list
    detections: List[dict] = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append(
            {
                "class": model.names[cls_id],
                "class_id": cls_id,
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(c, 2) for c in box.xyxy[0].tolist()],
            }
        )

    logger.info(
        "Detected %d defect(s) at conf>=%.2f", len(detections), confidence
    )

    return annotated_image, detections


def detect_defects_batch(
    model_path: Path,
    images: List[Image.Image],
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str = "mps",
) -> List[Tuple[Image.Image, List[dict]]]:
    """Run defect detection on a batch of images.

    Parameters
    ----------
    model_path : Path
        Path to a trained YOLOv8 ``best.pt`` checkpoint.
    images : list[PIL.Image.Image]
        Batch of input images (RGB).
    confidence : float
        Minimum confidence score.
    iou_threshold : float
        NMS IoU threshold.
    imgsz : int
        Inference resolution.
    device : str
        Compute device.

    Returns
    -------
    list[tuple[PIL.Image.Image, list[dict]]]
        One ``(annotated_image, detections)`` tuple per input image.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not images:
        return []

    # Ensure all images are RGB
    rgb_images = [
        img.convert("RGB") if img.mode != "RGB" else img for img in images
    ]

    model = _get_model(str(model_path))

    results = model.predict(
        source=rgb_images,
        conf=confidence,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    outputs: List[Tuple[Image.Image, List[dict]]] = []
    for result in results:
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[..., ::-1]
        annotated_image = Image.fromarray(annotated_rgb)

        detections: List[dict] = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            detections.append(
                {
                    "class": model.names[cls_id],
                    "class_id": cls_id,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(c, 2) for c in box.xyxy[0].tolist()],
                }
            )
        outputs.append((annotated_image, detections))

    logger.info("Batch inference complete: %d images processed.", len(outputs))
    return outputs


def clear_model_cache() -> None:
    """Clear all cached model instances to free memory."""
    _MODEL_CACHE.clear()
    logger.info("Model cache cleared.")
