"""Tests for configuration dataclasses."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base import ProjectConfig
from configs.detection import DetectionConfig
from configs.generation import GenerationConfig


def test_project_config_defaults():
    config = ProjectConfig()
    assert config.seed == 42
    assert config.dtype == "float16"
    assert config.enable_cpu_offload is True
    assert config.mps_high_watermark_ratio == 0.0


def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.controlnet_model == "lllyasviel/sd-controlnet-canny"
    assert config.height == 640
    assert config.width == 640
    assert config.num_inference_steps == 20


def test_detection_config_defaults():
    config = DetectionConfig()
    assert config.model_variant == "yolov8s.pt"
    assert config.num_classes == 6
    assert len(config.class_names) == config.num_classes
    assert config.imgsz == 640


def test_detection_class_names():
    config = DetectionConfig()
    expected = ["open", "short", "mousebite", "spur", "copper", "pinhole"]
    assert config.class_names == expected


def test_project_config_resolve_path():
    config = ProjectConfig()
    resolved = config.resolve_path(config.data_dir)
    assert resolved.is_absolute()
    assert str(resolved).endswith("data")
