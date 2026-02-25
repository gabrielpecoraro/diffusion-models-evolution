"""Benchmark inference time and memory usage for each model.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --models flux-schnell
    python scripts/benchmark.py --runs 3
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil

from config.default import DiffusionConfig
from models.memory_utils import clear_memory, get_device, log_memory_usage
from models.pipeline_factory import generate_image, load_pipeline


MODEL_CONFIGS = {
    "sd3-medium": {
        "num_inference_steps": 28,
        "guidance_scale": 7.0,
        "quantization": "none",
    },
    "flux-schnell": {
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "quantization": "Q4_K_S",
    },
}

TEST_PROMPT = "A photorealistic astronaut riding a white horse on Mars, cinematic lighting"


def benchmark_model(model_name, config_overrides, num_runs=3):
    """Benchmark a model's load time, inference time, and peak memory."""
    config = DiffusionConfig(
        model_name=model_name,
        prompt=TEST_PROMPT,
        height=512, width=512, seed=42,
        dtype="float16", enable_cpu_offload=True,
        drop_t5_encoder=True,
        **config_overrides,
    )

    process = psutil.Process()

    # Measure load time
    mem_before_load = process.memory_info().rss / 1024**3
    load_start = time.perf_counter()
    pipe = load_pipeline(config)
    load_time = time.perf_counter() - load_start
    mem_after_load = process.memory_info().rss / 1024**3

    # Warm-up run
    print(f"  Warm-up run...")
    _ = generate_image(pipe, config)

    # Benchmark runs
    inference_times = []
    peak_mem = mem_after_load
    for i in range(num_runs):
        mem_pre = process.memory_info().rss / 1024**3
        start = time.perf_counter()
        _ = generate_image(pipe, config)
        elapsed = time.perf_counter() - start
        mem_post = process.memory_info().rss / 1024**3
        peak_mem = max(peak_mem, mem_post)
        inference_times.append(elapsed)
        print(f"  Run {i + 1}/{num_runs}: {elapsed:.2f}s")

    avg_time = sum(inference_times) / len(inference_times)

    del pipe
    clear_memory()

    return {
        "model": model_name,
        "load_time": load_time,
        "avg_inference": avg_time,
        "min_inference": min(inference_times),
        "max_inference": max(inference_times),
        "mem_model_gb": mem_after_load - mem_before_load,
        "peak_mem_gb": peak_mem,
        "steps": config.num_inference_steps,
        "quantization": config.quantization,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark diffusion models")
    parser.add_argument("--models", nargs="+",
                        default=["sd3-medium", "flux-schnell"],
                        choices=["sd3-medium", "flux-schnell"],
                        help="Models to benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of inference runs")
    args = parser.parse_args()

    print("=" * 70)
    print("DIFFUSION MODEL BENCHMARK")
    print(f"Device: {get_device()} | Prompt: {TEST_PROMPT[:50]}...")
    print(f"Resolution: 512x512 | Runs: {args.runs}")
    print("=" * 70)

    results = []
    for model_name in args.models:
        print(f"\nBenchmarking {model_name}...")
        config_overrides = MODEL_CONFIGS[model_name]
        result = benchmark_model(model_name, config_overrides, args.runs)
        results.append(result)

    # Print results table
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(
        f"{'Model':<18} {'Steps':<7} {'Quant':<8} {'Load (s)':<10} "
        f"{'Avg (s)':<9} {'Min (s)':<9} {'Max (s)':<9} {'Model (GB)':<11} {'Peak (GB)':<10}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['model']:<18} {r['steps']:<7} {r['quantization']:<8} "
            f"{r['load_time']:<10.1f} {r['avg_inference']:<9.2f} "
            f"{r['min_inference']:<9.2f} {r['max_inference']:<9.2f} "
            f"{r['mem_model_gb']:<11.2f} {r['peak_mem_gb']:<10.2f}"
        )

    print("-" * 90)
    print(f"\nTest prompt: {TEST_PROMPT}")
    print(f"Resolution: 512x512 | Seed: 42 | Device: {get_device()}")


if __name__ == "__main__":
    main()
