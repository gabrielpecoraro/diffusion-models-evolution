"""Run all benchmark prompts through SD3 and FLUX, generate comparison grid.

Usage:
    python scripts/compare.py
    python scripts/compare.py --output assets/comparison_grid.png
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from config.default import DiffusionConfig
from models.memory_utils import clear_memory, log_memory_usage
from models.pipeline_factory import generate_image, load_pipeline
from models.prompt_bank import BENCHMARK_PROMPTS


def run_model(model_name, **kwargs):
    """Generate all benchmark images with a given model."""
    defaults = {
        "sd3-medium": {"num_inference_steps": 28, "guidance_scale": 7.0, "quantization": "none"},
        "flux-schnell": {"num_inference_steps": 4, "guidance_scale": 0.0, "quantization": "Q4_K_S"},
    }
    model_defaults = defaults[model_name]
    model_defaults.update(kwargs)

    config = DiffusionConfig(
        model_name=model_name,
        height=512, width=512, seed=42,
        dtype="float16", enable_cpu_offload=True,
        drop_t5_encoder=True,
        **model_defaults,
    )

    print(f"\nLoading {model_name}...")
    pipe = load_pipeline(config)

    images, times = [], []
    for i, ps in enumerate(BENCHMARK_PROMPTS):
        print(f"  [{i + 1}/{len(BENCHMARK_PROMPTS)}] {ps.prompt[:50]}...")
        config.prompt = ps.prompt
        start = time.perf_counter()
        img = generate_image(pipe, config)
        elapsed = time.perf_counter() - start
        images.append(img)
        times.append(elapsed)
        print(f"    {elapsed:.1f}s")

    del pipe
    clear_memory()
    return images, times


def build_comparison_grid(sd3_images, sd3_times, flux_images, flux_times, output_path):
    """Create a professional side-by-side comparison grid."""
    n = len(BENCHMARK_PROMPTS)
    fig = plt.figure(figsize=(14, 4 * n + 2))
    gs = gridspec.GridSpec(n + 1, 2, hspace=0.3, wspace=0.05,
                           height_ratios=[0.3] + [1] * n)

    # Headers
    for col, (label, color, bg) in enumerate([
        ("SD3 Medium\n28 steps | float16", "#1565C0", "#E3F2FD"),
        ("FLUX.1-schnell\n4 steps | GGUF Q4_K_S", "#C62828", "#FFEBEE"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=14,
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=bg, edgecolor=color))
        ax.axis("off")

    # Image rows
    for i in range(n):
        for col, (imgs, ts) in enumerate(
            [(sd3_images, sd3_times), (flux_images, flux_times)]
        ):
            ax = fig.add_subplot(gs[i + 1, col])
            ax.imshow(imgs[i])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    f"{BENCHMARK_PROMPTS[i].category.upper()}\n{ts[i]:.1f}s",
                    fontsize=10, fontweight="bold", rotation=0, labelpad=80, va="center",
                )

    fig.suptitle(
        "Diffusion Models Evolution: SD3 Medium vs FLUX.1-schnell\n"
        "Same prompts, same seed (42), 512x512",
        fontsize=16, fontweight="bold", y=0.98,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nComparison grid saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare SD3 vs FLUX.1 on benchmark prompts")
    parser.add_argument("--output", type=str, default="assets/comparison_grid.png",
                        help="Output path for comparison grid")
    args = parser.parse_args()

    print("=" * 60)
    print("COMPARISON: SD3 Medium vs FLUX.1-schnell")
    print("=" * 60)

    log_memory_usage("start")

    print("\n--- Phase 1: SD3 Medium ---")
    sd3_images, sd3_times = run_model("sd3-medium")

    print("\n--- Phase 2: FLUX.1-schnell ---")
    flux_images, flux_times = run_model("flux-schnell")

    # Build grid
    build_comparison_grid(sd3_images, sd3_times, flux_images, flux_times, args.output)

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Prompt':<16} {'Category':<16} {'SD3 (s)':<10} {'FLUX (s)':<10} {'Speedup':<10}")
    print("-" * 70)
    for ps, st, ft in zip(BENCHMARK_PROMPTS, sd3_times, flux_times):
        speedup = st / ft if ft > 0 else 0
        print(f"{ps.name:<16} {ps.category:<16} {st:<10.1f} {ft:<10.1f} {speedup:<10.1f}x")

    avg_sd3 = sum(sd3_times) / len(sd3_times)
    avg_flux = sum(flux_times) / len(flux_times)
    print("-" * 70)
    print(f"{'AVERAGE':<16} {'':<16} {avg_sd3:<10.1f} {avg_flux:<10.1f} {avg_sd3 / avg_flux:<10.1f}x")


if __name__ == "__main__":
    main()
