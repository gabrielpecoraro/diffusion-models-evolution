"""Create train/val/test splits and generate YOLO dataset.yaml."""

import random
import shutil
from pathlib import Path

import yaml

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]


def create_splits(
    processed_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split the converted dataset into train/val/test and write dataset.yaml.

    Args:
        processed_dir: Directory with images/all/ and labels/all/
        train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1.0)
        seed: Random seed for reproducibility

    Returns dict with image counts per split.
    """
    processed_dir = Path(processed_dir)
    all_images = sorted((processed_dir / "images" / "all").glob("*.jpg"))

    if not all_images:
        raise FileNotFoundError(f"No images found in {processed_dir / 'images' / 'all'}")

    random.seed(seed)
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    for split_name, image_list in splits.items():
        img_dir = processed_dir / "images" / split_name
        lbl_dir = processed_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_list:
            stem = img_path.stem
            shutil.copy2(img_path, img_dir / img_path.name)

            label_src = processed_dir / "labels" / "all" / f"{stem}.txt"
            if label_src.exists():
                shutil.copy2(label_src, lbl_dir / f"{stem}.txt")

    # Write dataset.yaml for YOLO
    dataset_yaml = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    yaml_path = processed_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    summary = {split: len(imgs) for split, imgs in splits.items()}
    summary["dataset_yaml"] = str(yaml_path)
    logger.info("Splits created: %s", summary)
    return summary


def create_augmented_dataset_yaml(
    processed_dir: Path,
    synthetic_dir: Path,
    output_yaml: Path,
) -> Path:
    """Create a YOLO dataset.yaml that includes synthetic images in the training set.

    This merges synthetic images into a combined training directory alongside real images.
    Val and test remain real-only for fair evaluation.
    """
    processed_dir = Path(processed_dir)
    synthetic_dir = Path(synthetic_dir)
    output_yaml = Path(output_yaml)

    # Create combined training directory
    combined_dir = processed_dir / "images" / "train_augmented"
    combined_labels = processed_dir / "labels" / "train_augmented"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_labels.mkdir(parents=True, exist_ok=True)

    # Copy real training images
    count_real = 0
    for img in (processed_dir / "images" / "train").glob("*.jpg"):
        shutil.copy2(img, combined_dir / img.name)
        label = processed_dir / "labels" / "train" / f"{img.stem}.txt"
        if label.exists():
            shutil.copy2(label, combined_labels / f"{img.stem}.txt")
        count_real += 1

    # Copy synthetic images
    count_syn = 0
    syn_images = synthetic_dir / "images"
    syn_labels = synthetic_dir / "labels"
    if syn_images.exists():
        for img in syn_images.glob("*.jpg"):
            shutil.copy2(img, combined_dir / img.name)
            label = syn_labels / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, combined_labels / f"{img.stem}.txt")
            count_syn += 1

    dataset_yaml = {
        "path": str(processed_dir.resolve()),
        "train": "images/train_augmented",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    logger.info("Augmented dataset: %d real + %d synthetic = %d total training images",
                count_real, count_syn, count_real + count_syn)
    return output_yaml
