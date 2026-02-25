"""Memory management utilities for running large diffusion models on 16GB Apple Silicon."""

import gc
import os

import psutil
import torch


def get_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map string to torch dtype, handling MPS bfloat16 limitation."""
    device = get_device()
    # MPS does not fully support bfloat16 in all operations
    if dtype_str == "bfloat16" and device.type == "mps":
        return torch.float16
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.float16)


def setup_mps_environment(high_watermark_ratio: float = 0.0):
    """Configure MPS memory settings. Call before loading any pipeline."""
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(high_watermark_ratio)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def clear_memory():
    """Aggressive memory cleanup between model loads."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_memory_usage(label: str = "") -> dict:
    """Print and return current memory usage in GB."""
    process = psutil.Process()
    mem = process.memory_info()
    rss_gb = mem.rss / 1024**3
    vms_gb = mem.vms / 1024**3
    print(f"[Memory {label}] RSS: {rss_gb:.2f} GB | VMS: {vms_gb:.2f} GB")
    return {"rss_gb": rss_gb, "vms_gb": vms_gb}
