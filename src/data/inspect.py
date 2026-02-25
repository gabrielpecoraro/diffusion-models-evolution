"""Dataset inspection and statistics."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils.logger import setup_logger
from src.utils.visualization import plot_class_distribution

logger = setup_logger(__name__)

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]


def compute_statistics(processed_dir: Path) -> dict:
    """Compute dataset statistics from YOLO-format labels."""
    processed_dir = Path(processed_dir)
    stats = {}

    for split in ["train", "val", "test"]:
        label_dir = processed_dir / "labels" / split
        if not label_dir.exists():
            continue

        class_counts = Counter()
        num_images = 0
        total_boxes = 0

        for label_file in label_dir.glob("*.txt"):
            num_images += 1
            text = label_file.read_text().strip()
            if not text:
                continue
            for line in text.split("\n"):
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] += 1
                total_boxes += 1

        named_counts = {CLASS_NAMES[k]: v for k, v in sorted(class_counts.items())}
        stats[split] = {
            "num_images": num_images,
            "total_boxes": total_boxes,
            "class_counts": named_counts,
        }

    return stats


def visualize_dataset(processed_dir: Path, output_dir: Path):
    """Generate all dataset inspection plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_statistics(processed_dir)

    # Class distribution for training set
    if "train" in stats:
        plot_class_distribution(
            stats["train"]["class_counts"],
            "Training Set Class Distribution",
            output_dir / "train_class_distribution.png",
        )

    # Overall summary
    logger.info("Dataset statistics:")
    for split, s in stats.items():
        logger.info("  %s: %d images, %d boxes", split, s["num_images"], s["total_boxes"])
        for cls, count in s["class_counts"].items():
            logger.info("    %s: %d", cls, count)

    return stats


def show_sample_images(processed_dir: Path, split: str = "train", n: int = 6):
    """Display sample images with bounding box overlays."""
    img_dir = Path(processed_dir) / "images" / split
    lbl_dir = Path(processed_dir) / "labels" / split

    images = sorted(img_dir.glob("*.jpg"))[:n]
    if not images:
        logger.warning("No images found in %s", img_dir)
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]

    for ax, img_path in zip(axes, images):
        img = np.array(Image.open(img_path))
        ax.imshow(img)

        label_path = lbl_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            h, w = img.shape[:2]
            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                rect_w = bw * w
                rect_h = bh * h

                rect = plt.Rectangle((x1, y1), rect_w, rect_h,
                                     linewidth=2, edgecolor=colors[cls_id % len(colors)],
                                     facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, y1 - 2, CLASS_NAMES[cls_id], fontsize=8,
                        color=colors[cls_id % len(colors)], fontweight="bold")

        ax.set_title(img_path.stem, fontsize=9)
        ax.axis("off")

    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Sample Images with Annotations ({split})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
