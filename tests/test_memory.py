"""Tests for memory utilities."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.memory import get_device, get_torch_dtype, log_memory_usage, setup_mps_environment


def test_get_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("mps", "cuda", "cpu")


def test_get_torch_dtype():
    assert get_torch_dtype("float16") == torch.float16
    assert get_torch_dtype("float32") == torch.float32
    assert get_torch_dtype("unknown") == torch.float16


def test_setup_mps_environment():
    setup_mps_environment(0.0)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_log_memory_usage():
    result = log_memory_usage("test")
    assert "rss_gb" in result
    assert result["rss_gb"] > 0
