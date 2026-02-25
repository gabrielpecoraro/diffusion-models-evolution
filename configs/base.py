"""Root project configuration — paths, seeds, device."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    synthetic_dir: Path = field(default_factory=lambda: Path("data/synthetic"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    seed: int = 42
    device: str = "auto"  # auto | mps | cuda | cpu
    dtype: str = "float16"

    # 16GB Apple Silicon memory settings
    mps_high_watermark_ratio: float = 0.0
    enable_cpu_offload: bool = True

    def resolve_path(self, relative: Path) -> Path:
        """Resolve a relative path against the project root."""
        return self.project_root / relative
