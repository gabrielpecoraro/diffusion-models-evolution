"""Tests for DiffusionConfig dataclass."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default import DiffusionConfig


def test_default_config():
    """Default config targets FLUX.1-schnell with Q4_K_S."""
    config = DiffusionConfig()
    assert config.model_name == "flux-schnell"
    assert config.quantization == "Q4_K_S"
    assert config.num_inference_steps == 4
    assert config.guidance_scale == 0.0
    assert config.enable_cpu_offload is True


def test_sd3_config():
    """SD3 config with appropriate defaults."""
    config = DiffusionConfig(
        model_name="sd3-medium",
        num_inference_steps=28,
        guidance_scale=7.0,
    )
    assert config.model_name == "sd3-medium"
    assert config.num_inference_steps == 28
    assert config.guidance_scale == 7.0
    assert config.drop_t5_encoder is True


def test_resolution_defaults():
    """Default resolution should be 512x512 for 16GB safety."""
    config = DiffusionConfig()
    assert config.height == 512
    assert config.width == 512


def test_custom_config():
    """Custom config overrides work correctly."""
    config = DiffusionConfig(
        model_name="flux-schnell",
        prompt="test prompt",
        seed=123,
        height=256,
        width=256,
    )
    assert config.prompt == "test prompt"
    assert config.seed == 123
    assert config.height == 256


def test_dtype_default():
    """Default dtype should be float16."""
    config = DiffusionConfig()
    assert config.dtype == "float16"
