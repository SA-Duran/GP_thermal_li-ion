from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable
import pandas as pd

from gpheat.utils.experiments import generate_dataframe_for_conditions
from gpheat.logger import get_logger

logger = get_logger(__name__)

def make_test_csvs(cfg: Dict[str, Any], paths, *, save: bool = True) -> dict[str, Path]:
    """Generate two CSVs: charge_test.csv and discharge_test.csv under artifacts/test/."""
    test_dir = Path(paths.__dict__.get("test_dir", paths.artifacts_root / "test"))
    test_dir.mkdir(parents=True, exist_ok=True)

    outs = {}

    def _one(name: str, items: Iterable[Dict[str, Any]]):
        df = generate_dataframe_for_conditions(cfg, items)
        out = test_dir / f"{name}_test.csv"
        if save:
            df.to_csv(out, index=False)
            logger.info(f"Saved {out} ({len(df)} rows)")
        return out

    outs["charge"]    = _one("charge", cfg["test_conditions"]["charge"])
    outs["discharge"] = _one("discharge", cfg["test_conditions"]["discharge"])
    return outs
