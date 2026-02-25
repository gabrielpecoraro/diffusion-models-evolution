"""Batch synthetic PCB image generation with annotation transfer.

Generates synthetic defective PCB images by conditioning Stable Diffusion
on Canny edge maps extracted from real template images.  Because
ControlNet preserves the spatial structure of the conditioning image,
YOLO-format bounding-box annotations from the original test images can
be directly transferred to the synthetic variants.

Typical usage::

    from configs import ProjectConfig, GenerationConfig
    from src.generation.pipeline import load_generation_pipeline
    from src.generation.generate import generate_synthetic_batch

    project_cfg = ProjectConfig()
    gen_cfg = GenerationConfig()
    pipe = load_generation_pipeline(gen_cfg, project_cfg)

    results = generate_synthetic_batch(
        pipe, gen_cfg, project_cfg, template_images, annotation_map,
    )
    print(results)  # {"total_generated": 1000, "output_dir": "..."}
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from configs.base import ProjectConfig
from configs.generation import GenerationConfig
from src.generation.edge_extraction import extract_canny_edges
from src.generation.prompt_bank import DEFECT_PROMPTS
from src.utils.memory import clear_memory, log_memory_usage

logger = logging.getLogger(__name__)


def _create_output_dirs(synthetic_dir: Path) -> Dict[str, Path]:
    """Create the directory tree for synthetic outputs.

    Args:
        synthetic_dir: Root directory for all synthetic data.

    Returns:
        Dict mapping logical name to the created ``Path``.
    """
    dirs = {
        "images": synthetic_dir / "images",
        "labels": synthetic_dir / "labels",
        "edge_maps": synthetic_dir / "edge_maps",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _build_filename(
    test_stem: str,
    defect_type: str,
    variant: int,
) -> str:
    """Construct a deterministic, collision-free filename stem.

    Format: ``syn_{original_stem}_{defect_type}_v{variant}``

    Args:
        test_stem: Stem of the source test image (no extension).
        defect_type: Defect category name (e.g. ``"open"``).
        variant: Variant index for this (image, defect) pair.

    Returns:
        Filename stem string (no extension).
    """
    return f"syn_{test_stem}_{defect_type}_v{variant}"


def generate_synthetic_batch(
    pipe,
    gen_config: GenerationConfig,
    project_config: ProjectConfig,
    template_images: List[Tuple[Path, Path]],
    annotation_map: Dict[str, Path],
) -> Dict[str, object]:
    """Generate synthetic defective PCB images conditioned on Canny edges.

    For each template image the function:

    1. Extracts Canny edges at the configured thresholds.
    2. Iterates over every defect prompt and requested number of variants.
    3. Runs the ControlNet pipeline to produce a synthetic image.
    4. Transfers the YOLO-format label file from the original test image
       (spatial layout is preserved by ControlNet conditioning).
    5. Saves the synthetic image, edge map, and label.
    6. Writes a ``metadata.json`` manifest for reproducibility.

    Early-stops once *gen_config.target_total_synthetic* images have been
    produced.

    Args:
        pipe: A loaded ``StableDiffusionControlNetPipeline``.
        gen_config: Generation hyper-parameters.
        project_config: Project-wide settings (paths, seed, etc.).
        template_images: List of ``(template_path, test_path)`` tuples.
            *template_path* is used for edge extraction; *test_path* is
            the corresponding annotated test image whose labels we
            transfer.
        annotation_map: Dict mapping ``test_image_stem`` to the
            ``Path`` of the YOLO-format ``.txt`` label file.

    Returns:
        Summary dict with keys ``"total_generated"``,
        ``"output_dir"``, and ``"elapsed_seconds"``.
    """
    synthetic_dir: Path = project_config.resolve_path(
        project_config.synthetic_dir,
    )
    dirs = _create_output_dirs(synthetic_dir)

    metadata: List[Dict] = []
    count: int = 0
    target: int = gen_config.target_total_synthetic
    start_time: float = time.time()

    log_memory_usage("before_generation")
    logger.info(
        "Starting synthetic generation — target: %d images, %d templates.",
        target,
        len(template_images),
    )

    for template_path, test_path in tqdm(
        template_images, desc="Generating synthetic"
    ):
        # -- Load and resize template ----------------------------------------
        template_img: Image.Image = (
            Image.open(template_path)
            .convert("RGB")
            .resize((gen_config.width, gen_config.height))
        )

        # -- Extract conditioning edge map -----------------------------------
        edge_map: Image.Image = extract_canny_edges(
            template_img,
            gen_config.canny_low,
            gen_config.canny_high,
        )

        # -- Resolve annotation for the paired test image --------------------
        test_stem: str = test_path.stem
        label_path: Optional[Path] = annotation_map.get(test_stem)

        # -- Iterate over prompts x variants ---------------------------------
        for prompt in DEFECT_PROMPTS:
            for variant in range(gen_config.num_synthetic_per_image):
                if count >= target:
                    break

                seed: int = project_config.seed + count
                generator = torch.Generator().manual_seed(seed)

                # -- Run diffusion -------------------------------------------
                result = pipe(
                    prompt=prompt.prompt,
                    negative_prompt=prompt.negative_prompt,
                    image=edge_map,
                    num_inference_steps=gen_config.num_inference_steps,
                    guidance_scale=gen_config.guidance_scale,
                    controlnet_conditioning_scale=(
                        gen_config.controlnet_conditioning_scale
                    ),
                    generator=generator,
                )
                synth_image: Image.Image = result.images[0]

                # -- Save outputs --------------------------------------------
                filename: str = _build_filename(
                    test_stem, prompt.defect_type, variant,
                )
                synth_image.save(dirs["images"] / f"{filename}.jpg")
                edge_map.save(dirs["edge_maps"] / f"{filename}_edges.jpg")

                # -- Transfer annotations ------------------------------------
                if label_path is not None and label_path.exists():
                    shutil.copy(
                        label_path,
                        dirs["labels"] / f"{filename}.txt",
                    )

                # -- Record metadata -----------------------------------------
                metadata.append(
                    {
                        "filename": filename,
                        "source_template": str(template_path),
                        "source_test": str(test_path),
                        "defect_type": prompt.defect_type,
                        "class_id": prompt.class_id,
                        "variant": variant,
                        "seed": seed,
                        "num_inference_steps": gen_config.num_inference_steps,
                        "guidance_scale": gen_config.guidance_scale,
                        "controlnet_conditioning_scale": (
                            gen_config.controlnet_conditioning_scale
                        ),
                    }
                )

                count += 1

                # -- Periodic memory housekeeping ----------------------------
                clear_memory()
                if count % 50 == 0:
                    log_memory_usage(f"after_{count}_images")
                    logger.info("Generated %d / %d images.", count, target)

            if count >= target:
                break
        if count >= target:
            break

    # -- Write metadata manifest ---------------------------------------------
    metadata_path = synthetic_dir / "metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    elapsed: float = time.time() - start_time
    log_memory_usage("after_generation")

    summary = {
        "total_generated": count,
        "output_dir": str(synthetic_dir),
        "metadata_path": str(metadata_path),
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(
        "Synthetic generation complete: %d images in %.1f s.",
        count,
        elapsed,
    )
    return summary
