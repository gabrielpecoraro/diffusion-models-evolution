"""Generate synthetic PCB defect images using SD 1.5 + ControlNet Canny.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --num-images 500 --conditioning-scale 0.8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from configs.base import ProjectConfig
from configs.generation import GenerationConfig
from src.generation.generate import generate_synthetic_batch
from src.generation.pipeline import load_generation_pipeline
from src.utils.memory import clear_memory, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PCB defect images")
    parser.add_argument("--num-images", type=int, default=1000,
                        help="Total synthetic images to generate")
    parser.add_argument("--conditioning-scale", type=float, default=1.0,
                        help="ControlNet conditioning scale (0.5-1.5)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Diffusion inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--no-cpu-offload", action="store_true",
                        help="Load full pipeline on MPS/GPU (faster, uses ~5GB)")
    args = parser.parse_args()

    project_config = ProjectConfig(
        enable_cpu_offload=not args.no_cpu_offload,
    )
    gen_config = GenerationConfig(
        target_total_synthetic=args.num_images,
        controlnet_conditioning_scale=args.conditioning_scale,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )

    processed_dir = project_config.resolve_path(project_config.processed_dir)
    templates_dir = processed_dir / "templates"
    labels_dir = processed_dir / "labels" / "all"

    # Collect template + test image pairs and annotation map
    template_images = []
    annotation_map = {}

    for temp_img in sorted(templates_dir.glob("*_temp.jpg")):
        stem = temp_img.stem.replace("_temp", "")
        test_img = processed_dir / "images" / "all" / f"{stem}.jpg"
        if test_img.exists():
            template_images.append((temp_img, test_img))

        label = labels_dir / f"{stem}.txt"
        if label.exists():
            annotation_map[stem] = label

    if not template_images:
        print("ERROR: No template images found. Run 'python scripts/download_data.py' first.")
        sys.exit(1)

    print(f"Found {len(template_images)} template-test pairs")
    print(f"Generating {gen_config.target_total_synthetic} synthetic images...")

    log_memory_usage("before loading pipeline")
    pipe = load_generation_pipeline(gen_config, project_config)
    log_memory_usage("after loading pipeline")

    result = generate_synthetic_batch(
        pipe, gen_config, project_config, template_images, annotation_map,
    )

    del pipe
    clear_memory()

    print(f"\nGenerated {result['total_generated']} synthetic images")
    print(f"Saved to: {project_config.resolve_path(project_config.synthetic_dir)}")


if __name__ == "__main__":
    main()
