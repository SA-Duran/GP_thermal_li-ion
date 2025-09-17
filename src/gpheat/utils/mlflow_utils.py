from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

def try_mlflow_start_run(experiment: str, run_name: str, tags: Dict[str, str] | None = None):
    try:
        import mlflow
        mlflow.set_experiment(experiment)
        return mlflow.start_run(run_name=run_name, tags=tags)
    except Exception:
        return None  # silent fallback

def try_mlflow_log_params(params: Dict[str, Any]):
    try:
        import mlflow
        mlflow.log_params({k: (float(v) if isinstance(v, (int, float)) else v) for k,v in params.items()})
    except Exception:
        pass

def try_mlflow_log_metrics(metrics: Dict[str, float], step: int | None = None):
    try:
        import mlflow
        mlflow.log_metrics({k: float(v) for k,v in metrics.items()}, step=step)
    except Exception:
        pass

def try_mlflow_log_artifact(path: Path):
    try:
        import mlflow
        mlflow.log_artifact(str(path))
    except Exception:
        pass
