"""Download the DeepPCB dataset."""

import shutil
import subprocess
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEEPPCB_REPO = "https://github.com/tangsanli5201/DeepPCB.git"


def download_deeppcb(raw_dir: Path) -> dict:
    """Clone the DeepPCB dataset from GitHub.

    The repo contains PCBData/ with groups of image pairs:
      - XXXXX_temp.jpg  (defect-free template)
      - XXXXX_test.jpg  (defective test image)
      - XXXXX.txt       (annotations: x1,y1,x2,y2,type)

    Returns summary dict with file counts.
    """
    raw_dir = Path(raw_dir)
    repo_dir = raw_dir / "DeepPCB"

    if repo_dir.exists() and (repo_dir / "PCBData").exists():
        logger.info("DeepPCB already downloaded at %s", repo_dir)
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning DeepPCB dataset...")
        subprocess.run(
            ["git", "clone", "--depth", "1", DEEPPCB_REPO, str(repo_dir)],
            check=True,
        )
        logger.info("Clone complete.")

    pcb_data = repo_dir / "PCBData"
    if not pcb_data.exists():
        raise FileNotFoundError(f"PCBData directory not found at {pcb_data}")

    # Count files
    templates = list(pcb_data.rglob("*_temp.jpg"))
    tests = list(pcb_data.rglob("*_test.jpg"))
    annotations = list(pcb_data.rglob("*.txt"))
    # Filter out non-annotation txt files (like README)
    annotations = [a for a in annotations if a.stem.isdigit() or a.stem.replace("_", "").isdigit()]

    summary = {
        "pcb_data_dir": str(pcb_data),
        "num_templates": len(templates),
        "num_tests": len(tests),
        "num_annotations": len(annotations),
    }
    logger.info("DeepPCB dataset: %d templates, %d tests, %d annotations",
                len(templates), len(tests), len(annotations))
    return summary
