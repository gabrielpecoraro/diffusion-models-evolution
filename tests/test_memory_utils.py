"""Tests for memory management utilities."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.memory_utils import (
    get_device,
    get_torch_dtype,
    log_memory_usage,
    setup_mps_environment,
)


def test_get_device():
    """Should return a valid torch device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("mps", "cuda", "cpu")


def test_get_torch_dtype_float16():
    """float16 mapping should always work."""
    dtype = get_torch_dtype("float16")
    assert dtype == torch.float16


def test_get_torch_dtype_float32():
    """float32 mapping should work."""
    dtype = get_torch_dtype("float32")
    assert dtype == torch.float32


def test_get_torch_dtype_unknown():
    """Unknown dtype should fall back to float16."""
    dtype = get_torch_dtype("unknown")
    assert dtype == torch.float16


def test_setup_mps_environment():
    """MPS environment variables should be set."""
    setup_mps_environment(0.0)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_log_memory_usage():
    """Memory logging should return valid dict."""
    result = log_memory_usage("test")
    assert "rss_gb" in result
    assert "vms_gb" in result
    assert result["rss_gb"] > 0
