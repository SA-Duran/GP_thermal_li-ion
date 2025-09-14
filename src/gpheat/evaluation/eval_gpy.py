from __future__ import annotations
from typing import Dict, Any, Iterable
from pathlib import Path
import numpy as np
import pandas as pd

from gpheat.models.checkpoints import load_pickle
from gpheat.models.gpy_gp import metrics, predict
from gpheat.plot.gpy_results import plot_dt_dt, plot_qrev
from gpheat.data.slicing import subset_df
from gpheat.logger import get_logger

logger = get_logger(__name__)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def eval_runs_with_model(
    *,
    model_path: Path,
    csv_path: Path,
    features: list[str],
    target: str,
    runs: Iterable[Dict[str, Any]],
    out_dir: Path,
    prefix: str
) -> Path:
    """
    For each {mode, c_rate, temperature_C} in runs:
      - slice CSV to that run
      - predict with loaded model
      - compute metrics if target present
      - save per-run metrics and plots
    Returns path to a summary CSV.
    """
    gp = load_pickle(model_path)
    df_all = pd.read_csv(csv_path)
    ensure_dir(out_dir)
    per_run = []

    for i, r in enumerate(runs, start=1):
        mode = r["mode"]; c_rate = float(r["c_rate"]); tC = float(r["temperature_C"])
        
        df = subset_df(df_all, mode=mode, c_rate=c_rate, temperature_C=tC)
        
        if df.empty:
            # one-time diagnostics per run
            tmp = df_all[df_all["State (Charge 1, Discharge 0)"] == (1 if mode.lower()=="charge" else 0)]
            crates = np.unique(np.round(tmp["C-rate"].to_numpy(), 6))
            tcol = "Initial temperature [K]" if "Initial temperature [K]" in tmp.columns else "Ambient temperature [K]"
            temps = np.unique(np.round(tmp[tcol].to_numpy() - 273.15, 2))
            logger.warning(f"[{i}] No data for {mode} @ {c_rate}C, {tC}°C — skipping")
            logger.warning(f"    Available C-rates for state={mode}: {crates.tolist()}")
            expected = -abs(c_rate) if mode.lower()=="charge" else abs(c_rate)
            logger.warning(f"    Expected signed C-rate: {expected}")
            logger.warning(f"    Available temps(°C) for state={mode}: {temps.tolist()}")
            continue
        
        
        
        logger.info(f"eval run started for {mode} @ {c_rate}C, {tC}°C")
        X = df[features].to_numpy()
        mu, var = predict(gp, X)
        std = np.sqrt(var)

        run_id = f"{mode.lower()}_{c_rate:.3f}C_{int(tC)}C"
        # metrics (if target present)
        if target in df.columns:
            y_true = df[[target]].to_numpy()
            m = metrics(y_true, mu, gp)
        else:
            m = {"rmse": np.nan, "mae": np.nan, "neg_loglik": np.nan}

        per_run.append({
            "run": run_id,
            "mode": mode,
            "c_rate": c_rate,
            "temperature_C": tC,
            **m
        })

        # plots
        plot_dt_dt(df, mu, std, out_dir / f"{prefix}_{run_id}_dt_dt.png",
                   title=f"dT/dt — {mode} @ {c_rate:.3f}C, {tC}°C")
        plot_qrev(df, mu, std, out_dir / f"{prefix}_{run_id}_qrev.png",
                  title=f"Qrev — {mode} @ {c_rate:.3f}C, {tC}°C")

        logger.info(f"[{i}] {run_id}: {m}")

    summary = pd.DataFrame(per_run)
    out_csv = out_dir / f"{prefix}_summary.csv"
    summary.to_csv(out_csv, index=False)
    logger.info(f"Saved summary: {out_csv}")
    return out_csv
