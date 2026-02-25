"""Shared plotting helpers for all pipeline stages."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(counts: dict, title: str, output_path: Path) -> plt.Figure:
    """Bar chart of class distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(counts.keys())
    values = list(counts.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_sample_detections(images: list, titles: list, output_path: Path, cols: int = 4) -> plt.Figure:
    """Grid of annotated detection images."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=9, fontweight="bold")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_metric_comparison(real_metrics: dict, aug_metrics: dict, metric_name: str,
                           class_names: list, output_path: Path) -> plt.Figure:
    """Side-by-side bar chart comparing a metric across classes for two models."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(class_names))
    width = 0.35

    real_vals = [real_metrics.get(c, 0) for c in class_names]
    aug_vals = [aug_metrics.get(c, 0) for c in class_names]

    ax.bar(x - width / 2, real_vals, width, label="Real Only", color="#1565C0", alpha=0.8)
    ax.bar(x + width / 2, aug_vals, width, label="Real + Synthetic", color="#C62828", alpha=0.8)

    ax.set_xlabel("Defect Class")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}: Real Only vs Real + Synthetic", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig
