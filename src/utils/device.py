"""Device detection and dtype handling."""

import torch

from src.utils.memory import get_device, get_torch_dtype


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve 'auto' to the best available device."""
    if device_str == "auto":
        return get_device()
    return torch.device(device_str)


def resolve_dtype(dtype_str: str = "float16") -> torch.dtype:
    """Resolve dtype string, handling MPS bfloat16 limitation."""
    return get_torch_dtype(dtype_str)
