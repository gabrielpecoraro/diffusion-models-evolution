"""Tests for PCB defect prompt bank."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.prompt_bank import DEFECT_PROMPTS


def test_prompt_count():
    """Should have 6 prompts for 6 DeepPCB defect types."""
    assert len(DEFECT_PROMPTS) == 6


def test_all_defect_types_covered():
    """All 6 defect types should have prompts."""
    types = {p.defect_type for p in DEFECT_PROMPTS}
    expected = {"open", "short", "mousebite", "spur", "copper", "pinhole"}
    assert types == expected


def test_class_ids():
    """Class IDs should be 0-5."""
    ids = {p.class_id for p in DEFECT_PROMPTS}
    assert ids == {0, 1, 2, 3, 4, 5}


def test_prompts_non_empty():
    """All prompts and negative prompts should be non-empty."""
    for p in DEFECT_PROMPTS:
        assert len(p.prompt) > 10
        assert len(p.negative_prompt) > 5


def test_unique_prompts():
    """No duplicate prompts."""
    prompts = [p.prompt for p in DEFECT_PROMPTS]
    assert len(prompts) == len(set(prompts))
