"""Tests for the benchmark prompt bank."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prompt_bank import (
    BENCHMARK_PROMPTS,
    get_all_prompt_texts,
    get_prompts_by_category,
)


def test_benchmark_prompts_count():
    """Should have exactly 5 benchmark prompts."""
    assert len(BENCHMARK_PROMPTS) == 5


def test_all_prompts_have_required_fields():
    """Each prompt should have name, prompt text, and category."""
    for ps in BENCHMARK_PROMPTS:
        assert ps.name, f"Missing name in prompt: {ps}"
        assert ps.prompt, f"Missing prompt text in: {ps.name}"
        assert ps.category, f"Missing category in: {ps.name}"


def test_unique_names():
    """All prompt names should be unique."""
    names = [ps.name for ps in BENCHMARK_PROMPTS]
    assert len(names) == len(set(names)), "Duplicate prompt names found"


def test_valid_categories():
    """All categories should be from the expected set."""
    valid = {"text_rendering", "photorealism", "spatial", "artistic", "complex"}
    for ps in BENCHMARK_PROMPTS:
        assert ps.category in valid, f"Invalid category '{ps.category}' in {ps.name}"


def test_get_prompts_by_category():
    """Category filtering should work."""
    text_prompts = get_prompts_by_category("text_rendering")
    assert len(text_prompts) >= 1
    assert all(p.category == "text_rendering" for p in text_prompts)


def test_get_all_prompt_texts():
    """Should return list of strings."""
    texts = get_all_prompt_texts()
    assert len(texts) == 5
    assert all(isinstance(t, str) for t in texts)


def test_empty_category():
    """Non-existent category should return empty list."""
    result = get_prompts_by_category("nonexistent")
    assert result == []
