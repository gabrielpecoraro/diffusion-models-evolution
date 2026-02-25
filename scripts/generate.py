"""Generate a single image using SD3 Medium or FLUX.1-schnell.

Usage:
    python scripts/generate.py --model flux-schnell --prompt "A cat holding a sign"
    python scripts/generate.py --model sd3-medium --prompt "A sunset" --steps 28 --guidance 7.0
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default import DiffusionConfig
from models.memory_utils import log_memory_usage
from models.pipeline_factory import generate_image, load_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate images with SD3 or FLUX.1")
    parser.add_argument("--model", choices=["sd3-medium", "flux-schnell"],
                        default="flux-schnell", help="Model to use")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (default: 4 for FLUX, 28 for SD3)")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Guidance scale (default: 0.0 for FLUX, 7.0 for SD3)")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--quantization", type=str, default="Q4_K_S",
                        choices=["none", "Q4_K_S", "Q6_K", "Q8_0"],
                        help="GGUF quantization level for FLUX")
    args = parser.parse_args()

    # Set model-specific defaults
    if args.steps is None:
        args.steps = 4 if args.model == "flux-schnell" else 28
    if args.guidance is None:
        args.guidance = 0.0 if args.model == "flux-schnell" else 7.0

    config = DiffusionConfig(
        model_name=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        seed=args.seed,
        quantization=args.quantization if args.model == "flux-schnell" else "none",
        output_dir=args.output_dir,
    )

    print(f"Model:      {config.model_name}")
    print(f"Prompt:     {config.prompt}")
    print(f"Steps:      {config.num_inference_steps}")
    print(f"Guidance:   {config.guidance_scale}")
    print(f"Resolution: {config.width}x{config.height}")
    print(f"Seed:       {config.seed}")
    if config.model_name == "flux-schnell":
        print(f"Quant:      {config.quantization}")
    print()

    log_memory_usage("before loading")
    print("Loading pipeline...")
    pipe = load_pipeline(config)

    print("Generating image...")
    start = time.perf_counter()
    image = generate_image(pipe, config)
    elapsed = time.perf_counter() - start
    print(f"Generated in {elapsed:.1f}s")

    # Save
    model_dir = os.path.join(config.output_dir, config.model_name.replace("-", "_"))
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{config.seed}_{config.num_inference_steps}steps.png"
    output_path = os.path.join(model_dir, filename)
    image.save(output_path)
    print(f"Saved to {output_path}")

    log_memory_usage("final")


if __name__ == "__main__":
    main()
