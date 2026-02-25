"""Tests for DeepPCB → YOLO format conversion."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.convert import deeppcb_to_yolo


def test_basic_conversion(sample_deeppcb_annotation):
    """Convert a sample annotation and verify YOLO format."""
    lines = deeppcb_to_yolo(sample_deeppcb_annotation, img_w=640, img_h=640)
    assert len(lines) == 2

    # First defect: class 1 → 0 (open)
    parts = lines[0].split()
    assert parts[0] == "0"  # class_id 0-indexed
    assert len(parts) == 5

    # All values should be normalized [0, 1]
    for val in parts[1:]:
        assert 0.0 <= float(val) <= 1.0


def test_class_mapping(sample_deeppcb_annotation):
    """Verify 1-indexed → 0-indexed class conversion."""
    lines = deeppcb_to_yolo(sample_deeppcb_annotation)
    # First line: class 1 → 0 (open)
    assert lines[0].startswith("0 ")
    # Second line: class 3 → 2 (mousebite)
    assert lines[1].startswith("2 ")


def test_center_coordinates(tmp_dir):
    """Verify center coordinate calculation."""
    # Create annotation: box from (0,0) to (640,640) — full image
    ann = tmp_dir / "test.txt"
    ann.write_text("0,0,640,640,1\n")
    lines = deeppcb_to_yolo(ann, img_w=640, img_h=640)
    parts = lines[0].split()
    # Center should be (0.5, 0.5), size should be (1.0, 1.0)
    assert abs(float(parts[1]) - 0.5) < 0.001
    assert abs(float(parts[2]) - 0.5) < 0.001
    assert abs(float(parts[3]) - 1.0) < 0.001
    assert abs(float(parts[4]) - 1.0) < 0.001


def test_empty_annotation(tmp_dir):
    """Empty annotation file should return empty list."""
    ann = tmp_dir / "empty.txt"
    ann.write_text("")
    lines = deeppcb_to_yolo(ann)
    assert lines == []


def test_normalization_bounds(tmp_dir):
    """Values should be clamped to [0, 1]."""
    ann = tmp_dir / "overflow.txt"
    ann.write_text("0,0,700,700,1\n")  # Exceeds 640x640
    lines = deeppcb_to_yolo(ann, img_w=640, img_h=640)
    for val in lines[0].split()[1:]:
        assert 0.0 <= float(val) <= 1.0
