from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

def load_training_data(paths, cfg: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    # TODO: wire to artifacts/datasets CSV in next step
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    X = rng.normal(size=(200, 8))
    y = X @ rng.normal(size=(8,)) + rng.normal(scale=0.1, size=200)
    split = int(len(X)*0.8)
    return X[:split], y[:split], X[split:], y[split:]
