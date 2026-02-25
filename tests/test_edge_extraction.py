"""Tests for Canny edge extraction."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.generation.edge_extraction import extract_canny_edges


def test_output_shape(sample_pcb_image):
    """Edge map should match input dimensions and be 3-channel."""
    edges = extract_canny_edges(sample_pcb_image)
    arr = np.array(edges)
    assert arr.shape == (640, 640, 3)


def test_output_is_binary(sample_pcb_image):
    """Edge pixels should be 0 or 255."""
    edges = extract_canny_edges(sample_pcb_image)
    arr = np.array(edges)
    unique = set(np.unique(arr))
    assert unique.issubset({0, 255})


def test_has_edges(sample_pcb_image):
    """PCB image with traces should produce non-trivial edges."""
    edges = extract_canny_edges(sample_pcb_image)
    arr = np.array(edges)
    assert arr.sum() > 0  # Not all black


def test_three_channels_identical(sample_pcb_image):
    """All 3 channels should be identical (grayscale stacked)."""
    edges = extract_canny_edges(sample_pcb_image)
    arr = np.array(edges)
    assert np.array_equal(arr[:, :, 0], arr[:, :, 1])
    assert np.array_equal(arr[:, :, 1], arr[:, :, 2])
