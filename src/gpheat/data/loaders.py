from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

def load_training_data(paths, cfg: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    # Placeholder: replace with real loading + preprocessing
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    X = rng.normal(size=(100, 5))
    y = X @ rng.normal(size=(5,)) + rng.normal(scale=0.1, size=100)
    split = int(len(X)*cfg["experiment"]["train_size"])
    return X[:split], y[:split], X[split:], y[split:]
