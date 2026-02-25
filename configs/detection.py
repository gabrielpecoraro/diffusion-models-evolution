"""Configuration for YOLOv8 training and evaluation."""

from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    model_variant: str = "yolov8s.pt"  # yolov8n.pt | yolov8s.pt

    # Training
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    patience: int = 20  # Early stopping

    # Optimizer
    lr0: float = 0.01
    lrf: float = 0.01

    # Augmentation (YOLO built-in)
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.0

    # DeepPCB classes
    class_names: list = field(default_factory=lambda: [
        "open", "short", "mousebite", "spur", "copper", "pinhole"
    ])
    num_classes: int = 6
