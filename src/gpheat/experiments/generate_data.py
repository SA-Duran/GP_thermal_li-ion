from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import pandas as pd

from gpheat.simulate.pybamm_model import run_experiment_grid

COLUMNS = [
        "Terminal voltage [V]", "Time [s]", "Current [A]", "Charge %", "Cell temperature [K]", "Terminal power [W]", "Discharge capacity [A.h]",
        "Ohmic heating [W.m-3]", "Reversible heating [W.m-3]", "Irreversible heating [W.m-3]", 
        "Contact resistance heating [W.m-3]", "Heat loss [W.m-3]","Reversible / irreversible heat index",
        "C-rate", "Ambient temperature [K]", "State (Charge 1, Discharge 0)", "Total heating [W.m-3]", 
        "Approximate temperature derivative [K/s]", "Model temperature time derivative [K/s]", "Initial temperature [K]", 
        
        "Model open-circuit potential [V]", "Open-circuit potential [V]","Open-circuit potential time derivative [V/s]",
        
        "Electrode concentration overpotential [V]", "Electrode concentration overpotential time derivative [V/s]",      
        "Reaction overpotential [V]", "Reaction overpotential time derivative [V/s]",
        "Electrolyte concentration overpotential [V]", "Electrolyte concentration overpotential time derivative [V/s]",   
        "Electrolyte ohmic overpotential [V]", "Electrolyte ohmic overpotential time derivative [V/s]", 
        "Electrode ohmic overpotential [V]", "Electrode ohmic overpotential time derivative [V/s]" , 
        "Reaction / concentration overpotential index",
        
        "Electrode concentration overpotential difference [V]", "Electrode concentration overpotential difference time derivative [V/s]",
        "Reaction overpotential difference [V]","Reaction overpotential difference time derivative [V/s]",
        "Electrode ohmic overpotential difference [V]", "Electrode ohmic overpotential difference time derivative [V/s]",
        "Current Temperature [IT]", "Current Relative Temperature OVP time derivative[W/s]", "Relative Temperature [K/K]",
        
        "Differential capacity [Ah/V]", "Terminal voltage time derivative [V/s]",
        
        "Model temperature time derivative due to Ohmic Heat [K/s]","Model temperature time derivative due to Irreversible Heat [K/s]",
        "Model temperature time derivative due to Heat loss [K/s]", "Model temperature time derivative due to Ohmic, Irreversible and Heat loss [K/s]",
        "Model temperature time derivative due to internal heat [K/s]"
    ]

def generate_and_save(cfg: Dict[str, Any], paths) -> Path:
    paths.artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.datasets_dir.mkdir(parents=True, exist_ok=True)

    data_rows = run_experiment_grid(cfg)
    import pandas as pd
    df = pd.DataFrame(data_rows, columns=COLUMNS)

    out_csv = paths.datasets_dir / cfg["dataset"]["name"]
    df.to_csv(out_csv, index=False)
    return out_csv
