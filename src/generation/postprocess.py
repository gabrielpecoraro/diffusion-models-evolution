"""Quality filtering for synthetic PCB images using SSIM.

After the diffusion pipeline produces synthetic images, this module
compares each one against its source template using the Structural
Similarity Index Measure (SSIM).  Images that are *too similar* to the
original are likely near-copies with little augmentation value; images
that are *too different* probably failed to maintain the PCB layout and
are unusable.  Both extremes are filtered out.

The default acceptance window ``[0.3, 0.85]`` keeps images that are
recognisably the same board but with meaningful synthetic variation.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


def _load_and_prepare(
    image_path: Path,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Load an image and convert to a grayscale NumPy array.

    Args:
        image_path: Path to the image file.
        target_size: Optional ``(width, height)`` to resize to before
            comparison.  Both images in a pair must be the same size for
            SSIM.

    Returns:
        2-D ``np.ndarray`` of dtype ``uint8``.
    """
    img = Image.open(image_path).convert("L")
    if target_size is not None:
        img = img.resize(target_size)
    return np.array(img)


def compute_ssim(
    image_a: Path,
    image_b: Path,
    target_size: Optional[Tuple[int, int]] = None,
) -> float:
    """Compute SSIM between two images.

    Both images are converted to grayscale and optionally resized to
    *target_size* so that their dimensions match.

    Args:
        image_a: Path to the first image.
        image_b: Path to the second image.
        target_size: Optional ``(width, height)`` resize target.

    Returns:
        SSIM score in the range ``[-1, 1]`` (typically ``[0, 1]`` for
        natural images).
    """
    arr_a = _load_and_prepare(image_a, target_size)
    arr_b = _load_and_prepare(image_b, target_size)

    # Ensure both arrays share the same shape.
    if arr_a.shape != arr_b.shape:
        min_h = min(arr_a.shape[0], arr_b.shape[0])
        min_w = min(arr_a.shape[1], arr_b.shape[1])
        arr_a = arr_a[:min_h, :min_w]
        arr_b = arr_b[:min_h, :min_w]

    score: float = ssim(arr_a, arr_b, data_range=255)
    return score


def filter_synthetic_images(
    synthetic_dir: Path,
    original_dir: Path,
    min_ssim: float = 0.3,
    max_ssim: float = 0.85,
    metadata_path: Optional[Path] = None,
    target_size: Optional[Tuple[int, int]] = (640, 640),
) -> Dict[str, object]:
    """Filter generated images by SSIM against their source originals.

    Images with SSIM **below** *min_ssim* are too different from the
    original (garbage / failed generation) and are removed.  Images with
    SSIM **above** *max_ssim* are near-duplicates and offer little
    augmentation value, so they are also removed.

    Removed images (and their corresponding labels and edge maps) are
    moved to a ``rejected/`` sub-directory inside *synthetic_dir* for
    manual review rather than being permanently deleted.

    Args:
        synthetic_dir: Root synthetic data directory (must contain
            ``images/``, and optionally ``labels/`` and ``edge_maps/``
            sub-directories).
        original_dir: Directory containing the original template images
            used during generation.  File-name matching is performed via
            the metadata manifest.
        min_ssim: Lower SSIM bound.  Images below this are rejected.
        max_ssim: Upper SSIM bound.  Images above this are rejected.
        metadata_path: Explicit path to ``metadata.json``.  Defaults to
            ``synthetic_dir / "metadata.json"``.
        target_size: ``(width, height)`` to which both images are
            resized before SSIM computation.  Ensures shape parity.

    Returns:
        Summary dict with keys ``"total_evaluated"``, ``"accepted"``,
        ``"rejected_too_similar"``, ``"rejected_too_different"``,
        ``"ssim_scores"`` (list of per-image dicts).
    """
    images_dir = synthetic_dir / "images"
    labels_dir = synthetic_dir / "labels"
    edge_maps_dir = synthetic_dir / "edge_maps"

    rejected_dir = synthetic_dir / "rejected"
    (rejected_dir / "images").mkdir(parents=True, exist_ok=True)
    (rejected_dir / "labels").mkdir(parents=True, exist_ok=True)
    (rejected_dir / "edge_maps").mkdir(parents=True, exist_ok=True)

    # -- Load metadata to resolve original source for each synthetic image ---
    if metadata_path is None:
        metadata_path = synthetic_dir / "metadata.json"

    source_lookup: Dict[str, str] = {}
    if metadata_path.exists():
        with open(metadata_path) as fh:
            entries: List[Dict] = json.load(fh)
        for entry in entries:
            source_lookup[entry["filename"]] = entry.get(
                "source_template", entry.get("source_test", "")
            )

    # -- Iterate over synthetic images and compute SSIM ----------------------
    ssim_scores: List[Dict] = []
    accepted: int = 0
    rejected_similar: int = 0
    rejected_different: int = 0
    total: int = 0

    synthetic_images = sorted(images_dir.glob("*.jpg")) + sorted(
        images_dir.glob("*.png")
    )

    for syn_path in synthetic_images:
        total += 1
        stem = syn_path.stem

        # Resolve the corresponding original image.
        original_path: Optional[Path] = _resolve_original(
            stem, source_lookup, original_dir,
        )

        if original_path is None or not original_path.exists():
            logger.warning(
                "No matching original for '%s' — skipping SSIM check.", stem,
            )
            ssim_scores.append(
                {"filename": stem, "ssim": None, "status": "skipped"}
            )
            accepted += 1
            continue

        score = compute_ssim(syn_path, original_path, target_size)

        if score < min_ssim:
            status = "rejected_too_different"
            rejected_different += 1
            _move_to_rejected(stem, syn_path, labels_dir, edge_maps_dir, rejected_dir)
        elif score > max_ssim:
            status = "rejected_too_similar"
            rejected_similar += 1
            _move_to_rejected(stem, syn_path, labels_dir, edge_maps_dir, rejected_dir)
        else:
            status = "accepted"
            accepted += 1

        ssim_scores.append(
            {"filename": stem, "ssim": round(score, 4), "status": status}
        )
        logger.debug(
            "%s — SSIM=%.4f — %s", stem, score, status,
        )

    # -- Write per-image scores to disk for later analysis -------------------
    scores_path = synthetic_dir / "ssim_scores.json"
    with open(scores_path, "w") as fh:
        json.dump(ssim_scores, fh, indent=2)

    summary = {
        "total_evaluated": total,
        "accepted": accepted,
        "rejected_too_similar": rejected_similar,
        "rejected_too_different": rejected_different,
        "min_ssim_threshold": min_ssim,
        "max_ssim_threshold": max_ssim,
        "scores_path": str(scores_path),
    }
    logger.info(
        "SSIM filtering complete — %d accepted, %d rejected "
        "(too_similar=%d, too_different=%d) out of %d.",
        accepted,
        rejected_similar + rejected_different,
        rejected_similar,
        rejected_different,
        total,
    )
    return summary


# ── Private helpers ──────────────────────────────────────────────────────


def _resolve_original(
    synthetic_stem: str,
    source_lookup: Dict[str, str],
    original_dir: Path,
) -> Optional[Path]:
    """Attempt to locate the original image for a synthetic stem.

    Looks up the metadata manifest first; falls back to heuristic
    name-parsing (``syn_{original_stem}_{defect}_{variant}``).

    Args:
        synthetic_stem: Filename stem of the synthetic image.
        source_lookup: Mapping from synthetic filename to original path
            string (from ``metadata.json``).
        original_dir: Directory to search for originals.

    Returns:
        ``Path`` to the original image, or ``None`` if unresolvable.
    """
    # 1. Try metadata lookup.
    if synthetic_stem in source_lookup:
        candidate = Path(source_lookup[synthetic_stem])
        if candidate.exists():
            return candidate
        # Metadata may store a relative path; try resolving against original_dir.
        candidate = original_dir / candidate.name
        if candidate.exists():
            return candidate

    # 2. Heuristic: strip the synthetic prefix.
    #    Format: syn_{original_stem}_{defect_type}_v{variant}
    parts = synthetic_stem.split("_")
    if len(parts) >= 4 and parts[0] == "syn":
        # The original stem could contain underscores, so we need to
        # reconstruct it.  The last two tokens are {defect_type} and v{N}.
        original_stem = "_".join(parts[1:-2])
        for ext in (".jpg", ".png", ".bmp"):
            candidate = original_dir / f"{original_stem}{ext}"
            if candidate.exists():
                return candidate

    return None


def _move_to_rejected(
    stem: str,
    image_path: Path,
    labels_dir: Path,
    edge_maps_dir: Path,
    rejected_dir: Path,
) -> None:
    """Move a rejected synthetic image (and its companions) to the rejected folder.

    Args:
        stem: Filename stem shared by the image, label, and edge map.
        image_path: Path to the synthetic image file.
        labels_dir: Directory containing label ``.txt`` files.
        edge_maps_dir: Directory containing edge-map images.
        rejected_dir: Destination root (must already contain
            ``images/``, ``labels/``, ``edge_maps/`` sub-dirs).
    """
    # Move image.
    shutil.move(str(image_path), str(rejected_dir / "images" / image_path.name))

    # Move label if it exists.
    for ext in (".txt",):
        label_file = labels_dir / f"{stem}{ext}"
        if label_file.exists():
            shutil.move(
                str(label_file),
                str(rejected_dir / "labels" / label_file.name),
            )

    # Move edge map if it exists.
    for ext in (".jpg", ".png"):
        edge_file = edge_maps_dir / f"{stem}_edges{ext}"
        if edge_file.exists():
            shutil.move(
                str(edge_file),
                str(rejected_dir / "edge_maps" / edge_file.name),
            )
