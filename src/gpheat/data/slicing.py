from __future__ import annotations
import numpy as np
import pandas as pd

def subset_df(
    df: pd.DataFrame,
    *,
    mode: str,
    c_rate: float,
    temperature_C: float,
    atol_rate: float = 1e-6,
    atol_tempK: float = 0.25,
) -> pd.DataFrame:
    """Return rows for a single run (mode, c_rate, tempC) with tolerant matching.

    PyBaMM convention:
      - Charge   → C-rate negative
      - Discharge→ C-rate positive
    We keep the config positive and flip internally for Charge.
    """
    is_charge = 1 if mode.lower() == "charge" else 0
    d = df[df["State (Charge 1, Discharge 0)"] == is_charge].copy()

    # temperature (K) match
    if "Initial temperature [K]" in d.columns:
        tempK = d["Initial temperature [K]"].to_numpy()
    elif "Ambient temperature [K]" in d.columns:
        tempK = d["Ambient temperature [K]"].to_numpy()
    else:
        raise KeyError("No temperature column found (Initial/Ambient temperature [K]).")
    sel_temp = np.isclose(tempK, temperature_C + 273.15, atol=atol_tempK, rtol=0.0)

    # C-rate sign-aware match
    desired = -abs(float(c_rate)) if is_charge == 1 else abs(float(c_rate))
    rate = d["C-rate"].to_numpy()
    sel_rate = np.isclose(rate, desired, atol=atol_rate, rtol=0.0)
    out=d[sel_temp & sel_rate]
    return out

