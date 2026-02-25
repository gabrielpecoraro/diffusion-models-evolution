"""Comprehensive YOLOv8 evaluation: mAP, per-class metrics, confusion matrix.

Evaluates trained models on the test split and produces structured metrics,
comparison artifacts (bar charts, confusion matrices, PR curves), and CSV
summaries suitable for downstream reporting.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consistent plot styling
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}
plt.rcParams.update(PLOT_STYLE)

PALETTE_REAL = "#4C72B0"
PALETTE_AUG = "#DD8452"


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: Path,
    dataset_yaml: Path,
    split: str = "test",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict:
    """Run evaluation on a dataset split and return structured metrics.

    Parameters
    ----------
    model_path : Path
        Path to a trained ``best.pt`` checkpoint.
    dataset_yaml : Path
        YOLO-format dataset YAML with train/val/test splits defined.
    split : str
        Which split to evaluate on (``"test"`` or ``"val"``).
    conf_threshold : float
        Confidence threshold used during NMS.
    iou_threshold : float
        IoU threshold for a true-positive match.

    Returns
    -------
    dict
        ``overall`` dict with mAP50, mAP50_95, precision, recall, and
        ``per_class`` list of dicts (one per class).
    """
    model_path = Path(model_path)
    dataset_yaml = Path(dataset_yaml)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    logger.info(
        "Evaluating %s on split='%s' (conf=%.2f, iou=%.2f)",
        model_path.name,
        split,
        conf_threshold,
        iou_threshold,
    )

    model = YOLO(str(model_path))
    results = model.val(
        data=str(dataset_yaml),
        split=split,
        conf=conf_threshold,
        iou=iou_threshold,
        device="mps",
        verbose=False,
    )

    # --- Overall metrics --------------------------------------------------
    overall = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
    }

    # --- Per-class metrics ------------------------------------------------
    class_names: List[str] = list(model.names.values())
    per_class: List[dict] = []

    # results.box contains per-class arrays when available
    box_results = results.box
    ap50_per_class = box_results.ap50 if box_results.ap50 is not None else []
    ap_per_class = box_results.ap if box_results.ap is not None else []

    for idx, name in enumerate(class_names):
        entry = {"class": name}
        if len(ap50_per_class) > idx:
            entry["mAP50"] = float(ap50_per_class[idx])
        else:
            entry["mAP50"] = None
        if len(ap_per_class) > idx:
            entry["mAP50_95"] = float(np.mean(ap_per_class[idx]))
        else:
            entry["mAP50_95"] = None
        per_class.append(entry)

    logger.info(
        "Evaluation done | mAP50=%.4f | mAP50-95=%.4f",
        overall["mAP50"],
        overall["mAP50_95"],
    )

    return {
        "overall": overall,
        "per_class": per_class,
        "class_names": class_names,
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model_path: Path,
    dataset_yaml: Path,
    output_path: Path,
    split: str = "test",
    normalize: bool = True,
) -> plt.Figure:
    """Generate and save a confusion matrix heatmap.

    Parameters
    ----------
    model_path : Path
        Trained model checkpoint.
    dataset_yaml : Path
        YOLO-format dataset YAML.
    output_path : Path
        File path to save the figure (e.g. ``confusion_matrix.png``).
    split : str
        Dataset split to use.
    normalize : bool
        Whether to row-normalize the matrix.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    model_path = Path(model_path)
    dataset_yaml = Path(dataset_yaml)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    results = model.val(data=str(dataset_yaml), split=split, device="mps", verbose=False)

    # The confusion matrix is stored in results.confusion_matrix
    cm = results.confusion_matrix
    matrix = cm.matrix  # numpy array (num_classes+1 x num_classes+1)
    class_names = list(model.names.values()) + ["background"]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalised)" if normalize else ""))

    fig.savefig(str(output_path))
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)
    return fig


# ---------------------------------------------------------------------------
# Per-class bar chart
# ---------------------------------------------------------------------------

def plot_per_class_metrics(
    metrics: dict,
    output_path: Path,
    metric_key: str = "mAP50",
    title: Optional[str] = None,
    color: str = PALETTE_REAL,
) -> plt.Figure:
    """Horizontal bar chart of a chosen metric broken down by class.

    Parameters
    ----------
    metrics : dict
        Output of :func:`evaluate_model`.
    output_path : Path
        Destination file path for the saved figure.
    metric_key : str
        Which per-class metric to plot (``"mAP50"`` or ``"mAP50_95"``).
    title : str, optional
        Plot title; auto-generated if *None*.
    color : str
        Bar colour.

    Returns
    -------
    matplotlib.figure.Figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = [entry["class"] for entry in metrics["per_class"]]
    values = [
        entry.get(metric_key, 0.0) or 0.0 for entry in metrics["per_class"]
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, len(classes) * 0.6)))
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, values, color=color, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel(metric_key)
    ax.set_xlim(0, 1.0)
    ax.set_title(title or f"Per-class {metric_key}")

    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    fig.savefig(str(output_path))
    plt.close(fig)
    logger.info("Per-class bar chart saved to %s", output_path)
    return fig


# ---------------------------------------------------------------------------
# Compare two models
# ---------------------------------------------------------------------------

def compare_models(
    real_model_path: Path,
    aug_model_path: Path,
    dataset_yaml: Path,
    output_dir: Path,
    split: str = "test",
) -> dict:
    """Evaluate both models and generate comparison artifacts.

    Produces:
    * Per-class mAP50 bar charts for each model and a grouped comparison.
    * Confusion matrices for each model.
    * A summary CSV and a summary markdown table.

    Parameters
    ----------
    real_model_path : Path
        Checkpoint trained on real-only data.
    aug_model_path : Path
        Checkpoint trained on real+synthetic data.
    dataset_yaml : Path
        Dataset YAML (both models are evaluated on the same test set).
    output_dir : Path
        Root directory for comparison outputs.
    split : str
        Dataset split.

    Returns
    -------
    dict
        ``real_metrics``, ``aug_metrics``, ``summary_df`` (as dict), and
        paths to all saved artifacts.
    """
    real_model_path = Path(real_model_path)
    aug_model_path = Path(aug_model_path)
    dataset_yaml = Path(dataset_yaml)
    output_dir = Path(output_dir) / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Comparing models: real=%s  aug=%s", real_model_path, aug_model_path)

    # --- Evaluate both ----------------------------------------------------
    real_metrics = evaluate_model(real_model_path, dataset_yaml, split=split)
    aug_metrics = evaluate_model(aug_model_path, dataset_yaml, split=split)

    # --- Per-class bar charts (individual) --------------------------------
    plot_per_class_metrics(
        real_metrics,
        output_dir / "per_class_mAP50_real.png",
        metric_key="mAP50",
        title="Per-class mAP50 -- Real Only",
        color=PALETTE_REAL,
    )
    plot_per_class_metrics(
        aug_metrics,
        output_dir / "per_class_mAP50_augmented.png",
        metric_key="mAP50",
        title="Per-class mAP50 -- Real + Synthetic",
        color=PALETTE_AUG,
    )

    # --- Grouped comparison bar chart -------------------------------------
    grouped_fig = _plot_grouped_comparison(
        real_metrics, aug_metrics, output_dir / "per_class_comparison.png"
    )

    # --- Confusion matrices -----------------------------------------------
    plot_confusion_matrix(
        real_model_path, dataset_yaml, output_dir / "confusion_matrix_real.png", split=split
    )
    plot_confusion_matrix(
        aug_model_path, dataset_yaml, output_dir / "confusion_matrix_augmented.png", split=split
    )

    # --- Summary table (CSV + Markdown) -----------------------------------
    summary_df = _build_summary_table(real_metrics, aug_metrics)
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    summary_df.to_csv(str(csv_path), index=False)
    md_path.write_text(summary_df.to_markdown(index=False))

    logger.info("Comparison artifacts written to %s", output_dir)

    return {
        "real_metrics": real_metrics,
        "aug_metrics": aug_metrics,
        "summary_df": summary_df.to_dict(orient="records"),
        "artifacts": {
            "csv": str(csv_path),
            "markdown": str(md_path),
            "comparison_chart": str(output_dir / "per_class_comparison.png"),
            "confusion_real": str(output_dir / "confusion_matrix_real.png"),
            "confusion_aug": str(output_dir / "confusion_matrix_augmented.png"),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plot_grouped_comparison(
    real_metrics: dict,
    aug_metrics: dict,
    output_path: Path,
) -> plt.Figure:
    """Side-by-side grouped bar chart comparing per-class mAP50."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = [e["class"] for e in real_metrics["per_class"]]
    real_vals = np.array([e.get("mAP50", 0.0) or 0.0 for e in real_metrics["per_class"]])
    aug_vals = np.array([e.get("mAP50", 0.0) or 0.0 for e in aug_metrics["per_class"]])

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_real = ax.bar(x - width / 2, real_vals, width, label="Real Only", color=PALETTE_REAL)
    bars_aug = ax.bar(x + width / 2, aug_vals, width, label="Real + Synthetic", color=PALETTE_AUG)

    ax.set_ylabel("mAP50")
    ax.set_title("Per-class mAP50 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Value labels
    for bar_group in (bars_real, bars_aug):
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.savefig(str(output_path))
    plt.close(fig)
    logger.info("Grouped comparison chart saved to %s", output_path)
    return fig


def _build_summary_table(real_metrics: dict, aug_metrics: dict) -> pd.DataFrame:
    """Build a DataFrame comparing real vs augmented metrics side-by-side."""
    rows = []
    class_names = [e["class"] for e in real_metrics["per_class"]]

    for idx, name in enumerate(class_names):
        real_map50 = real_metrics["per_class"][idx].get("mAP50", 0.0) or 0.0
        aug_map50 = aug_metrics["per_class"][idx].get("mAP50", 0.0) or 0.0
        rows.append(
            {
                "class": name,
                "real_mAP50": round(real_map50, 4),
                "aug_mAP50": round(aug_map50, 4),
                "delta_mAP50": round(aug_map50 - real_map50, 4),
            }
        )

    # Append overall row
    rows.append(
        {
            "class": "OVERALL",
            "real_mAP50": round(real_metrics["overall"]["mAP50"], 4),
            "aug_mAP50": round(aug_metrics["overall"]["mAP50"], 4),
            "delta_mAP50": round(
                aug_metrics["overall"]["mAP50"] - real_metrics["overall"]["mAP50"], 4
            ),
        }
    )

    return pd.DataFrame(rows)
