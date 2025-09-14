# src/gpheat/utils/physics.py
from __future__ import annotations
import numpy as np
import pybamm

def rho_eff_marquis2019() -> float:
    param = pybamm.ParameterValues("Marquis2019")
    components = [
        "Negative current collector", "Negative electrode",
        "Separator", "Positive electrode", "Positive current collector"
    ]
    rho_k = np.array([param.get(f"{c} density [kg.m-3]") for c in components])
    cp_k  = np.array([param.get(f"{c} specific heat capacity [J.kg-1.K-1]") for c in components])
    L_k   = np.array([param.get(f"{c} thickness [m]") for c in components])
    return float(np.sum(rho_k * cp_k * L_k) / np.sum(L_k))
