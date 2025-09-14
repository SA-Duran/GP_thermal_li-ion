from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict
import json

from .data.loaders import load_training_data
from .models.gp import GPRegressor
from .evaluation.metrics import rmse, mae
from .plot.series import plot_predictions

@dataclass
class Paths:
    reports_dir: Path
    figures_dir: Path
    metrics_file: Path

def run_pipeline(cfg: Dict[str, Any], paths: Paths) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_training_data(paths, cfg)

    gp = GPRegressor(cfg)
    gp.fit(X_train, y_train)
    preds = gp.predict(X_test)

    metrics = {
        "rmse": rmse(y_test, preds),
        "mae": mae(y_test, preds),
    }
    paths.metrics_file.write_text(json.dumps(metrics, indent=2))
    plot_predictions(y_test, preds, paths.figures_dir / "prediction_test.png")
    print("Metrics written to", paths.metrics_file)
