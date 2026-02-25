"""Download DeepPCB dataset, convert to YOLO format, and create splits.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base import ProjectConfig
from src.data.convert import convert_dataset
from src.data.download import download_deeppcb
from src.data.inspect import visualize_dataset
from src.data.split import create_splits


def main():
    parser = argparse.ArgumentParser(description="Download and prepare DeepPCB dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ProjectConfig()
    raw_dir = config.resolve_path(config.raw_dir)
    processed_dir = config.resolve_path(config.processed_dir)

    # Step 1: Download
    print("=" * 60)
    print("Step 1: Downloading DeepPCB dataset")
    print("=" * 60)
    summary = download_deeppcb(raw_dir)

    # Step 2: Convert to YOLO format
    print("\n" + "=" * 60)
    print("Step 2: Converting to YOLO format")
    print("=" * 60)
    conv_stats = convert_dataset(summary["pcb_data_dir"], processed_dir)

    # Step 3: Create splits
    print("\n" + "=" * 60)
    print("Step 3: Creating train/val/test splits")
    print("=" * 60)
    split_stats = create_splits(
        processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Step 4: Inspect
    print("\n" + "=" * 60)
    print("Step 4: Dataset inspection")
    print("=" * 60)
    output_dir = config.resolve_path(config.output_dir) / "data_inspection"
    visualize_dataset(processed_dir, output_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Dataset YAML: {split_stats['dataset_yaml']}")
    print(f"  Train: {split_stats['train']} images")
    print(f"  Val:   {split_stats['val']} images")
    print(f"  Test:  {split_stats['test']} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
