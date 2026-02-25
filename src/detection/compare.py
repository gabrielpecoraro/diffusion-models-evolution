"""A/B comparison visualisation helpers.

Generates publication-ready plots that compare real-only vs real+synthetic
YOLOv8 training runs: per-class mAP bars, overlaid training curves, and a
full comparison report with a markdown summary table.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (consistent with evaluate.py)
# ---------------------------------------------------------------------------
COLOR_REAL = "#4C72B0"
COLOR_AUG = "#DD8452"

PLOT_DEFAULTS = {
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}
plt.rcParams.update(PLOT_DEFAULTS)


# ---------------------------------------------------------------------------
# Per-class mAP comparison bar chart
# ---------------------------------------------------------------------------

def plot_mAP_comparison(
    real_metrics: dict,
    aug_metrics: dict,
    class_names: List[str],
    output_path: Path,
    metric_key: str = "mAP50",
) -> plt.Figure:
    """Grouped bar chart comparing per-class mAP for real-only vs augmented.

    Parameters
    ----------
    real_metrics : dict
        Per-class metrics for the real-only model.  Expected structure::

            {"per_class": [{"class": "...", "mAP50": float, ...}, ...],
             "overall": {"mAP50": float, ...}}

    aug_metrics : dict
        Same structure for the real+synthetic model.
    class_names : list[str]
        Ordered class labels for the x-axis.
    output_path : Path
        File path to save the figure.
    metric_key : str
        Metric to compare (``"mAP50"`` or ``"mAP50_95"``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build value arrays aligned to class_names
    real_map = _extract_per_class(real_metrics, class_names, metric_key)
    aug_map = _extract_per_class(aug_metrics, class_names, metric_key)

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_r = ax.bar(x - width / 2, real_map, width, label="Real Only", color=COLOR_REAL)
    bars_a = ax.bar(x + width / 2, aug_map, width, label="Real + Synthetic", color=COLOR_AUG)

    ax.set_ylabel(metric_key)
    ax.set_title(f"Per-class {metric_key}: Real vs Augmented")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    _annotate_bars(ax, bars_r)
    _annotate_bars(ax, bars_a)

    # Overall annotation
    real_overall = real_metrics.get("overall", {}).get(metric_key, 0.0) or 0.0
    aug_overall = aug_metrics.get("overall", {}).get(metric_key, 0.0) or 0.0
    delta = aug_overall - real_overall
    sign = "+" if delta >= 0 else ""
    ax.text(
        0.98, 0.02,
        f"Overall: {real_overall:.3f} vs {aug_overall:.3f} ({sign}{delta:.3f})",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    fig.savefig(str(output_path))
    plt.close(fig)
    logger.info("mAP comparison chart saved to %s", output_path)
    return fig


# ---------------------------------------------------------------------------
# Training curves overlay
# ---------------------------------------------------------------------------

def plot_training_curves(
    real_csv: Path,
    aug_csv: Path,
    output_path: Path,
    metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """Overlay training loss and mAP curves from two experiments.

    Reads the ``results.csv`` files that Ultralytics writes during training
    and plots selected columns side-by-side.

    Parameters
    ----------
    real_csv : Path
        ``results.csv`` from the real-only run.
    aug_csv : Path
        ``results.csv`` from the real+synthetic run.
    output_path : Path
        Destination file for the saved figure.
    metrics : list[str], optional
        Column names to plot.  Defaults to a sensible set of loss and mAP
        columns if *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    real_csv = Path(real_csv)
    aug_csv = Path(aug_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not real_csv.exists():
        raise FileNotFoundError(f"Real training CSV not found: {real_csv}")
    if not aug_csv.exists():
        raise FileNotFoundError(f"Augmented training CSV not found: {aug_csv}")

    df_real = pd.read_csv(str(real_csv))
    df_aug = pd.read_csv(str(aug_csv))

    # Ultralytics pads column names with spaces; strip them
    df_real.columns = df_real.columns.str.strip()
    df_aug.columns = df_aug.columns.str.strip()

    if metrics is None:
        # Auto-detect common columns
        candidate_metrics = [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ]
        metrics = [m for m in candidate_metrics if m in df_real.columns and m in df_aug.columns]

    if not metrics:
        logger.warning("No common metrics found between the two CSVs. Returning empty figure.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No common metrics found", ha="center", va="center")
        fig.savefig(str(output_path))
        plt.close(fig)
        return fig

    n_metrics = len(metrics)
    cols = min(n_metrics, 3)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, metric_name in enumerate(metrics):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        if metric_name in df_real.columns:
            ax.plot(
                df_real["epoch"] if "epoch" in df_real.columns else range(len(df_real)),
                df_real[metric_name],
                label="Real Only",
                color=COLOR_REAL,
                linewidth=1.5,
            )
        if metric_name in df_aug.columns:
            ax.plot(
                df_aug["epoch"] if "epoch" in df_aug.columns else range(len(df_aug)),
                df_aug[metric_name],
                label="Real + Synthetic",
                color=COLOR_AUG,
                linewidth=1.5,
            )

        ax.set_title(metric_name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(n_metrics, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Training Curves: Real vs Augmented", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
    logger.info("Training curves saved to %s", output_path)
    return fig


# ---------------------------------------------------------------------------
# Full comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(
    real_results: dict,
    aug_results: dict,
    output_dir: Path,
    real_csv: Optional[Path] = None,
    aug_csv: Optional[Path] = None,
) -> Dict[str, str]:
    """Generate all comparison artefacts and a markdown summary.

    This is the high-level entry point that orchestrates the individual
    plotting functions and writes a summary table.

    Parameters
    ----------
    real_results : dict
        Output of ``evaluate.evaluate_model`` for the real-only model.
    aug_results : dict
        Same structure for the real+synthetic model.
    output_dir : Path
        Directory where all artefacts are saved.
    real_csv : Path, optional
        ``results.csv`` from the real-only training run (for training curves).
    aug_csv : Path, optional
        ``results.csv`` from the augmented training run.

    Returns
    -------
    dict[str, str]
        Mapping of artefact name to its file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artefacts: Dict[str, str] = {}

    class_names = real_results.get("class_names", [])

    # 1. Per-class mAP50 comparison
    mAP50_path = output_dir / "mAP50_comparison.png"
    plot_mAP_comparison(real_results, aug_results, class_names, mAP50_path, metric_key="mAP50")
    artefacts["mAP50_comparison"] = str(mAP50_path)

    # 2. Per-class mAP50-95 comparison
    mAP50_95_path = output_dir / "mAP50_95_comparison.png"
    plot_mAP_comparison(
        real_results, aug_results, class_names, mAP50_95_path, metric_key="mAP50_95"
    )
    artefacts["mAP50_95_comparison"] = str(mAP50_95_path)

    # 3. Training curves (if CSVs supplied)
    if real_csv is not None and aug_csv is not None:
        curves_path = output_dir / "training_curves.png"
        try:
            plot_training_curves(real_csv, aug_csv, curves_path)
            artefacts["training_curves"] = str(curves_path)
        except FileNotFoundError as exc:
            logger.warning("Skipping training curves: %s", exc)

    # 4. Summary markdown table
    summary_md_path = output_dir / "comparison_summary.md"
    summary_df = _build_comparison_dataframe(real_results, aug_results, class_names)
    summary_csv_path = output_dir / "comparison_summary.csv"
    summary_df.to_csv(str(summary_csv_path), index=False)
    artefacts["summary_csv"] = str(summary_csv_path)

    md_lines = [
        "# A/B Comparison Report",
        "",
        "## Per-class results",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Overall",
        "",
        _overall_summary_markdown(real_results, aug_results),
        "",
    ]
    summary_md_path.write_text("\n".join(md_lines))
    artefacts["summary_md"] = str(summary_md_path)

    logger.info("Comparison report written to %s (%d artefacts)", output_dir, len(artefacts))
    return artefacts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_per_class(
    metrics: dict,
    class_names: List[str],
    key: str,
) -> np.ndarray:
    """Extract a per-class metric array aligned to *class_names*."""
    lookup = {e["class"]: e.get(key, 0.0) or 0.0 for e in metrics.get("per_class", [])}
    return np.array([lookup.get(name, 0.0) for name in class_names])


def _annotate_bars(ax: plt.Axes, bars) -> None:
    """Add value labels on top of each bar."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def _build_comparison_dataframe(
    real_metrics: dict,
    aug_metrics: dict,
    class_names: List[str],
) -> pd.DataFrame:
    """Create a side-by-side DataFrame of per-class results."""
    real_map50 = _extract_per_class(real_metrics, class_names, "mAP50")
    aug_map50 = _extract_per_class(aug_metrics, class_names, "mAP50")
    real_map95 = _extract_per_class(real_metrics, class_names, "mAP50_95")
    aug_map95 = _extract_per_class(aug_metrics, class_names, "mAP50_95")

    df = pd.DataFrame(
        {
            "Class": class_names,
            "Real mAP50": np.round(real_map50, 4),
            "Aug mAP50": np.round(aug_map50, 4),
            "Delta mAP50": np.round(aug_map50 - real_map50, 4),
            "Real mAP50-95": np.round(real_map95, 4),
            "Aug mAP50-95": np.round(aug_map95, 4),
            "Delta mAP50-95": np.round(aug_map95 - real_map95, 4),
        }
    )
    return df


def _overall_summary_markdown(real_metrics: dict, aug_metrics: dict) -> str:
    """Return a markdown snippet with overall metric comparison."""
    ro = real_metrics.get("overall", {})
    ao = aug_metrics.get("overall", {})

    lines = [
        "| Metric | Real Only | Real+Synthetic | Delta |",
        "|--------|-----------|----------------|-------|",
    ]
    for key in ("mAP50", "mAP50_95", "precision", "recall"):
        rv = ro.get(key, 0.0) or 0.0
        av = ao.get(key, 0.0) or 0.0
        delta = av - rv
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {key} | {rv:.4f} | {av:.4f} | {sign}{delta:.4f} |")

    return "\n".join(lines)
