"""Shared test fixtures."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_pcb_image():
    """Create a synthetic PCB-like image (green substrate + white traces)."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:, :] = [0, 100, 0]  # Green substrate
    # Horizontal traces
    for y in range(100, 600, 80):
        img[y:y + 3, 50:590] = [255, 255, 255]
    # Vertical traces
    for x in range(100, 600, 80):
        img[50:590, x:x + 3] = [255, 255, 255]
    return Image.fromarray(img)


@pytest.fixture
def sample_deeppcb_annotation(tmp_dir):
    """Create a sample DeepPCB annotation file."""
    content = "100,200,150,250,1\n300,400,350,420,3\n"
    path = tmp_dir / "00001.txt"
    path.write_text(content)
    return path


@pytest.fixture
def sample_yolo_label(tmp_dir):
    """Create a sample YOLO format label file."""
    content = "0 0.195312 0.351562 0.078125 0.078125\n2 0.507812 0.640625 0.078125 0.031250\n"
    path = tmp_dir / "00001.txt"
    path.write_text(content)
    return path
