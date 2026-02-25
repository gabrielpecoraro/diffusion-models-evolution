"""Run evaluation and comparison of trained models.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --real-model outputs/real_only/weights/best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from configs.base import ProjectConfig
from src.detection.evaluate import evaluate_model, print_comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare detection models")
    parser.add_argument("--real-model", type=str, default=None,
                        help="Path to real-only model weights")
    parser.add_argument("--aug-model", type=str, default=None,
                        help="Path to augmented model weights")
    args = parser.parse_args()

    config = ProjectConfig()
    output_dir = config.resolve_path(config.output_dir)
    processed_dir = config.resolve_path(config.processed_dir)
    dataset_yaml = processed_dir / "dataset.yaml"

    # Auto-detect model paths
    real_model = Path(args.real_model) if args.real_model else _find_best_model(output_dir / "real_only")
    aug_model = Path(args.aug_model) if args.aug_model else _find_best_model(output_dir / "real_synthetic")

    if real_model and real_model.exists():
        print("Evaluating real-only model...")
        real_metrics = evaluate_model(real_model, dataset_yaml)
    else:
        print("Real-only model not found, skipping.")
        real_metrics = None

    if aug_model and aug_model.exists():
        print("Evaluating augmented model...")
        aug_metrics = evaluate_model(aug_model, dataset_yaml)
    else:
        print("Augmented model not found, skipping.")
        aug_metrics = None

    if real_metrics and aug_metrics:
        print_comparison(real_metrics, aug_metrics)


def _find_best_model(experiment_dir: Path) -> Path:
    """Search for best.pt in a YOLO experiment directory."""
    if not experiment_dir.exists():
        return None
    for best in experiment_dir.rglob("best.pt"):
        return best
    return None


if __name__ == "__main__":
    main()
