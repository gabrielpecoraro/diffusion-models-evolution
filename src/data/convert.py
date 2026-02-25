"""Convert DeepPCB annotations to YOLO format."""

import shutil
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# DeepPCB class mapping (1-indexed in original → 0-indexed for YOLO)
CLASS_MAP = {
    1: 0,  # open
    2: 1,  # short
    3: 2,  # mousebite
    4: 3,  # spur
    5: 4,  # copper (spurious copper)
    6: 5,  # pinhole
}


def deeppcb_to_yolo(annotation_path: Path, img_w: int = 640, img_h: int = 640) -> list[str]:
    """Convert a single DeepPCB annotation file to YOLO format lines.

    DeepPCB format: x1,y1,x2,y2,type (1-indexed class, pixel coords)
    YOLO format:    class_id x_center y_center width height (0-indexed, normalized)
    """
    lines = []
    text = annotation_path.read_text().strip()
    if not text:
        return lines

    for row in text.split("\n"):
        row = row.strip()
        if not row:
            continue
        parts = row.split(",")
        if len(parts) < 5:
            # Try space-separated
            parts = row.split()
        if len(parts) < 5:
            logger.warning("Skipping malformed line in %s: %s", annotation_path, row)
            continue

        x1, y1, x2, y2, cls = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

        class_id = CLASS_MAP.get(cls)
        if class_id is None:
            logger.warning("Unknown class %d in %s", cls, annotation_path)
            continue

        # Convert corner → center, normalize
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return lines


def convert_dataset(pcb_data_dir: Path, processed_dir: Path) -> dict:
    """Convert entire DeepPCB dataset to YOLO format.

    Copies test images to processed_dir/images/ and converted labels to processed_dir/labels/.
    Also copies template images to processed_dir/templates/ for ControlNet conditioning.

    Returns dict with conversion statistics.
    """
    pcb_data_dir = Path(pcb_data_dir)
    processed_dir = Path(processed_dir)

    images_dir = processed_dir / "images" / "all"
    labels_dir = processed_dir / "labels" / "all"
    templates_dir = processed_dir / "templates"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    total_defects = 0
    class_counts = {i: 0 for i in range(6)}

    for group_dir in sorted(pcb_data_dir.iterdir()):
        if not group_dir.is_dir():
            continue

        for annotation_file in sorted(group_dir.glob("*.txt")):
            stem = annotation_file.stem

            # Find corresponding images
            test_img = group_dir / f"{stem}_test.jpg"
            temp_img = group_dir / f"{stem}_temp.jpg"

            if not test_img.exists():
                skipped += 1
                continue

            # Convert annotations
            yolo_lines = deeppcb_to_yolo(annotation_file)
            if not yolo_lines:
                skipped += 1
                continue

            # Copy test image
            shutil.copy2(test_img, images_dir / f"{stem}.jpg")

            # Write YOLO label
            (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

            # Copy template if exists
            if temp_img.exists():
                shutil.copy2(temp_img, templates_dir / f"{stem}_temp.jpg")

            # Count classes
            for line in yolo_lines:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1
                total_defects += 1

            converted += 1

    class_names = ["open", "short", "mousebite", "spur", "copper", "pinhole"]
    named_counts = {class_names[k]: v for k, v in class_counts.items()}

    logger.info("Converted %d images (%d skipped), %d total defects", converted, skipped, total_defects)
    logger.info("Class distribution: %s", named_counts)

    return {
        "converted": converted,
        "skipped": skipped,
        "total_defects": total_defects,
        "class_counts": named_counts,
    }
