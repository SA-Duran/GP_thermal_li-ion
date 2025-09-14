from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Paths:
    root: Path
    artifacts_root: Path
    datasets_dir: Path
    reports_dir: Path
    figures_dir: Path
    metrics_file: Path

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_paths(cfg: dict, root: str | Path) -> Paths:
    root = Path(root)
    p = cfg["paths"]
    return Paths(
        root=root,
        artifacts_root=root / p["artifacts_root"],
        datasets_dir=root / p["datasets_dir"],
        reports_dir=root / p["reports_dir"],
        figures_dir=root / p["figures_dir"],
        metrics_file=root / p["metrics_file"],
    )
