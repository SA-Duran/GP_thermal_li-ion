from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

def save_pickle(model: Any, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path

def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)
