# src/gpheat/plot/gpy_results.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gpheat.utils.physics import rho_eff_marquis2019
from gpheat.logger import get_logger

logger = get_logger(__name__)

def plot_dt_dt(df: pd.DataFrame, y_mean: np.ndarray, y_std: np.ndarray,
               out_png: Path, title: str,
               lim_dTdt: tuple[float,float] | None = (-1e-4, 2e-4)) -> None:
    t = df[["Time [s]"]].to_numpy().flatten()
    y_true = df[["Model temperature time derivative [K/s]"]].to_numpy().flatten()
    plt.figure(figsize=(8,5))
    plt.plot(t, y_mean, label="Predicted dT/dt")
    plt.fill_between(t, (y_mean-2*y_std).flatten(), (y_mean+2*y_std).flatten(), alpha=0.2, label="±2σ")
    plt.plot(t, y_true, label="Model dT/dt", color="red")
    if lim_dTdt:
        plt.ylim(lim_dTdt)
    plt.xlabel("Time [s]"); plt.ylabel("Temperature derivative [K/s]")
    plt.title(title); plt.grid(True); plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved {out_png}")

def plot_qrev(df: pd.DataFrame, y_mean: np.ndarray, y_std: np.ndarray,
              out_png: Path, title: str,
              lim_rev_heat: tuple[float,float] | None = (-1500, 1500)) -> None:
    rho_eff = rho_eff_marquis2019()
    q_total_pred = y_mean * rho_eff
    q_loss = df[["Heat loss [W.m-3]"]].to_numpy()
    q_irrev = df[["Irreversible heating [W.m-3]"]].to_numpy()
    q_ohm  = df[["Ohmic heating [W.m-3]"]].to_numpy()
    q_rev  = df[["Reversible heating [W.m-3]"]].to_numpy()
    q_rev_pred = q_total_pred - q_irrev - q_ohm - q_loss

    t = df[["Time [s]"]].to_numpy().flatten()
    band_lo = (y_mean - 2*y_std)*rho_eff - q_irrev - q_ohm - q_loss
    band_hi = (y_mean + 2*y_std)*rho_eff - q_irrev - q_ohm - q_loss

    plt.figure(figsize=(8,5))
    plt.plot(t, q_rev_pred, label="Predicted Qrev")
    plt.plot(t, q_rev, label="Model Qrev", color="red")
    plt.fill_between(t, band_lo.flatten(), band_hi.flatten(), alpha=0.2, label="±2σ")
    if lim_rev_heat:
        plt.ylim(lim_rev_heat)
    plt.xlabel("Time [s]"); plt.ylabel("Heat [W/m³]")
    plt.title(title); plt.grid(True); plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved {out_png}")
