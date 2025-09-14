from __future__ import annotations
from typing import Dict, Any, Iterable, List
import pandas as pd

from gpheat.simulate.pybamm_model import run_experiment_grid

def expand_conditions_to_cfg_grid(cfg: Dict[str, Any], items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of {mode, c_rate, temperature_C}, produce a temporary cfg
    whose grid only contains those points. Reuses all other cfg fields.
    """
    modes = [d["mode"] for d in items]
    c_rates = [float(d["c_rate"]) for d in items]
    temps = [float(d["temperature_C"]) for d in items]
    tmp = {k: v for (k, v) in cfg.items()}
    tmp["grid"] = {
        "temperatures_C": sorted(set(temps)),
        "modes": sorted(set(modes)),
        "c_rates": sorted(set(c_rates)),
    }
    return tmp

def generate_dataframe_for_conditions(cfg: Dict[str, Any], items: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """
    Runs pybamm on the (tiny) grid induced by items and returns a DataFrame.
    Note: run_experiment_grid returns rows already consistent with your columns.
    """
    tmp_cfg = expand_conditions_to_cfg_grid(cfg, items)
    rows = run_experiment_grid(tmp_cfg)
    from gpheat.experiments.generate_data import COLUMNS
    return pd.DataFrame(rows, columns=COLUMNS)
