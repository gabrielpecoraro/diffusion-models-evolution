"""Train YOLOv8 detector on real and/or real+synthetic data.

Usage:
    python scripts/train_detector.py --experiment both
    python scripts/train_detector.py --experiment real_only --epochs 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from configs.base import ProjectConfig
from configs.detection import DetectionConfig
from src.data.split import create_augmented_dataset_yaml
from src.detection.trainer import run_comparison_experiment, train_detector


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 PCB defect detector")
    parser.add_argument("--experiment", choices=["real_only", "real_synthetic", "both"],
                        default="both", help="Which experiment to run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    args = parser.parse_args()

    project_config = ProjectConfig()
    det_config = DetectionConfig(
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    processed_dir = project_config.resolve_path(project_config.processed_dir)
    synthetic_dir = project_config.resolve_path(project_config.synthetic_dir)
    output_dir = project_config.resolve_path(project_config.output_dir)

    real_yaml = processed_dir / "dataset.yaml"
    if not real_yaml.exists():
        print("ERROR: dataset.yaml not found. Run 'python scripts/download_data.py' first.")
        sys.exit(1)

    if args.experiment in ("real_synthetic", "both"):
        # Create augmented dataset YAML
        aug_yaml = processed_dir / "dataset_augmented.yaml"
        create_augmented_dataset_yaml(processed_dir, synthetic_dir, aug_yaml)

    print("=" * 60)
    print(f"TRAINING: experiment={args.experiment}, model={args.model}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    print("=" * 60)

    if args.experiment == "both":
        results = run_comparison_experiment(det_config, real_yaml, aug_yaml, output_dir)
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"  Real-only mAP50:     {results['real_only']['best_mAP50']:.4f}")
        print(f"  Augmented mAP50:     {results['real_synthetic']['best_mAP50']:.4f}")
        print(f"  Improvement:         {results['mAP50_improvement']:+.4f}")
    elif args.experiment == "real_only":
        result = train_detector(det_config, real_yaml, output_dir, "real_only")
        print(f"\nmAP50: {result['best_mAP50']:.4f}")
        print(f"Model: {result['model_path']}")
    else:
        result = train_detector(det_config, aug_yaml, output_dir, "real_synthetic")
        print(f"\nmAP50: {result['best_mAP50']:.4f}")
        print(f"Model: {result['model_path']}")


if __name__ == "__main__":
    main()
