"""YOLOv8 training wrapper with A/B comparison experiment.

Provides functions to train YOLOv8 on PCB defect datasets and run
controlled A/B experiments comparing real-only vs real+synthetic data.
"""

import logging
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from configs.detection import DetectionConfig

logger = logging.getLogger(__name__)


def train_detector(
    config: DetectionConfig,
    dataset_yaml: Path,
    output_dir: Path,
    experiment_name: str,
    resume: bool = False,
) -> dict:
    """Train a YOLOv8 model on a dataset described by a YOLO-format YAML file.

    Parameters
    ----------
    config : DetectionConfig
        Hyper-parameters and model variant configuration.
    dataset_yaml : Path
        Path to the YOLO dataset YAML (must define train/val/test splits).
    output_dir : Path
        Root directory where training artifacts are written.
    experiment_name : str
        Sub-directory name under *output_dir* for this run.
    resume : bool, optional
        If True, resume training from the last checkpoint.

    Returns
    -------
    dict
        Contains ``best_mAP50``, ``best_mAP50_95``, and ``model_path``.
    """
    dataset_yaml = Path(dataset_yaml)
    output_dir = Path(output_dir)

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting training run '%s' | model=%s | epochs=%d | batch=%d | imgsz=%d",
        experiment_name,
        config.model_variant,
        config.epochs,
        config.batch_size,
        config.imgsz,
    )

    model = YOLO(config.model_variant)

    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.imgsz,
        patience=config.patience,
        lr0=config.lr0,
        lrf=config.lrf,
        augment=config.augment,
        mosaic=config.mosaic,
        mixup=config.mixup,
        project=str(output_dir),
        name=experiment_name,
        seed=42,
        device="mps",
        workers=0,
        verbose=True,
    )

    if resume:
        last_ckpt = output_dir / experiment_name / "weights" / "last.pt"
        if last_ckpt.exists():
            logger.info("Resuming from checkpoint: %s", last_ckpt)
            model = YOLO(str(last_ckpt))
            train_kwargs["resume"] = True
        else:
            logger.warning(
                "Resume requested but no checkpoint found at %s; starting fresh.",
                last_ckpt,
            )

    results = model.train(**train_kwargs)

    best_map50 = results.results_dict.get("metrics/mAP50(B)")
    best_map50_95 = results.results_dict.get("metrics/mAP50-95(B)")
    model_path = str(output_dir / experiment_name / "weights" / "best.pt")

    logger.info(
        "Training complete for '%s' | mAP50=%.4f | mAP50-95=%.4f | model=%s",
        experiment_name,
        best_map50 if best_map50 is not None else -1.0,
        best_map50_95 if best_map50_95 is not None else -1.0,
        model_path,
    )

    return {
        "best_mAP50": best_map50,
        "best_mAP50_95": best_map50_95,
        "model_path": model_path,
    }


def run_comparison_experiment(
    config: DetectionConfig,
    real_yaml: Path,
    augmented_yaml: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict:
    """Run an A/B experiment: real-only training vs real+synthetic training.

    Both models are trained with identical hyper-parameters; the only
    difference is the data they see.  This makes it possible to isolate the
    effect of synthetic augmentation.

    Parameters
    ----------
    config : DetectionConfig
        Shared hyper-parameters for both runs.
    real_yaml : Path
        Dataset YAML pointing to the real-only split.
    augmented_yaml : Path
        Dataset YAML pointing to the real+synthetic split.
    output_dir : Path
        Root directory for both experiment sub-folders.
    resume : bool, optional
        If True, resume each run from its last checkpoint (if present).

    Returns
    -------
    dict
        ``real_only`` and ``real_synthetic`` result dicts, plus
        ``mAP50_improvement`` and ``mAP50_95_improvement`` deltas.
    """
    real_yaml = Path(real_yaml)
    augmented_yaml = Path(augmented_yaml)
    output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info("A/B COMPARISON EXPERIMENT")
    logger.info("=" * 60)

    # --- Arm A: real-only ------------------------------------------------
    logger.info(">>> Arm A: Training on REAL data only")
    real_results = train_detector(
        config, real_yaml, output_dir, "real_only", resume=resume
    )

    # --- Arm B: real + synthetic -----------------------------------------
    logger.info(">>> Arm B: Training on REAL + SYNTHETIC data")
    aug_results = train_detector(
        config, augmented_yaml, output_dir, "real_synthetic", resume=resume
    )

    # --- Compute deltas --------------------------------------------------
    map50_real = real_results["best_mAP50"] or 0.0
    map50_aug = aug_results["best_mAP50"] or 0.0
    map50_95_real = real_results["best_mAP50_95"] or 0.0
    map50_95_aug = aug_results["best_mAP50_95"] or 0.0

    map50_improvement = map50_aug - map50_real
    map50_95_improvement = map50_95_aug - map50_95_real

    logger.info("-" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("-" * 60)
    logger.info("Real-only   mAP50=%.4f  mAP50-95=%.4f", map50_real, map50_95_real)
    logger.info("Real+Synth  mAP50=%.4f  mAP50-95=%.4f", map50_aug, map50_95_aug)
    logger.info(
        "Improvement mAP50=%+.4f  mAP50-95=%+.4f",
        map50_improvement,
        map50_95_improvement,
    )

    return {
        "real_only": real_results,
        "real_synthetic": aug_results,
        "mAP50_improvement": map50_improvement,
        "mAP50_95_improvement": map50_95_improvement,
    }
