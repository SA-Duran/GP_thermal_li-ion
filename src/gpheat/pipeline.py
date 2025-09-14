from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict
import json

from .data.loaders import load_training_data
from .simulate.pybamm_model import build_cell_model, simulate_cell
from .models.gp import GPRegressor
from .evaluation.metrics import rmse, mae

@dataclass
class Paths:
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    metrics_file: Path

def run_pipeline(cfg: Dict[str, Any], paths: Paths) -> None:
    # Ensure report dirs exist
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data
    X_train, y_train, X_test, y_test = load_training_data(paths, cfg)

    # 2) Simulation (optional step, placeholder)
    model = build_cell_model(cfg)
    _ = simulate_cell(model, cfg)

    # 3) Model
    gp = GPRegressor(cfg)
    gp.fit(X_train, y_train)
    preds = gp.predict(X_test)

    # 4) Metrics
    metrics = {
        "rmse": rmse(y_test, preds),
        "mae": mae(y_test, preds),
    }
    paths.metrics_file.write_text(json.dumps(metrics, indent=2))
    print("Metrics written to", paths.metrics_file)
