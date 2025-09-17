# src/gpheat/models/gpy_gp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import GPy
from typing import Optional
from gpheat.logger import get_logger

logger = get_logger(__name__)

@dataclass
class GPyDataset:
    X: np.ndarray
    y: np.ndarray
    df: pd.DataFrame  # keep the full DF for plotting/aux

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")
    return df

def prepare_xy(df: pd.DataFrame, features: list[str], target: str) -> GPyDataset:
    X = df[features].to_numpy()
    y = df[[target]].to_numpy()
    return GPyDataset(X=X, y=y, df=df)

def build_kernel_charge(cfg: Dict[str, Any], input_dims: int) -> GPy.kern.Kern:
    fx = cfg["gpy"]["charge_fixed"]
    # active_dims: 0 for Vdot, 1..4 for the other four
    kV = GPy.kern.RBF(input_dim=1, lengthscale=fx["kV_lengthscale"], variance=fx["kV_variance"],
                      ARD=False, active_dims=[0])
    k1 = GPy.kern.RBF(input_dim=input_dims-1, lengthscale=np.array(fx["k1_lengthscales"]),
                      variance=fx["k1_variance"], ARD=True, active_dims=[1,2,3,4])
    return kV * k1

def build_kernel_discharge(cfg: Dict[str, Any], input_dims: int) -> GPy.kern.Kern:
    fx = cfg["gpy"]["discharge_fixed"]
    kV = GPy.kern.RBF(input_dim=1, lengthscale=fx["kV_lengthscale"], variance=fx["kV_variance"],
                      ARD=False, active_dims=[0])
    k1 = GPy.kern.RBF(input_dim=input_dims-1, lengthscale=np.array(fx["k1_lengthscales"]),
                      variance=fx["k1_variance"], ARD=True, active_dims=[1,2,3,4])
    return kV * k1

def make_model(X: np.ndarray, y: np.ndarray, kernel: GPy.kern.Kern, noise_variance: float = 0.0) -> GPy.core.GP:
    like = GPy.likelihoods.Gaussian(variance=noise_variance)
    exact = GPy.inference.latent_function_inference.ExactGaussianInference()
    model = GPy.core.GP(X=X, Y=y, kernel=kernel, likelihood=like, inference_method=exact)
    return model

def set_bounds(model: GPy.core.GP, cfg: Dict[str, Any]) -> None:
    bnd = cfg["gpy"]["search_bounds"]
    # kV (assume product kernel[0] is RBF, kernel[1] is RBF)
    kV = model.kern.prod[0]
    k1 = model.kern.prod[1]

    kV.lengthscale.constrain_bounded(*bnd["kV_lengthscale"], warning=False)
    kV.variance.constrain_bounded(*bnd["kV_variance"], warning=False)
    k1.lengthscale.constrain_bounded(np.array([r[0] for r in bnd["k1_lengthscales"]]),
                                     np.array([r[1] for r in bnd["k1_lengthscales"]]), warning=False)
    k1.variance.constrain_bounded(*bnd["k1_variance"], warning=False)
    model.likelihood.variance.constrain_bounded(*bnd["noise_variance"], warning=False)

def fit(model: GPy.core.GP, iters: int = 200) -> None:
    logger.info(f"Optimizing GPy model for {iters} iters...")
    model.optimize(messages=True, max_iters=iters)

def metrics(y_true: np.ndarray, y_pred_mean: np.ndarray, model: GPy.core.GP) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred_mean)**2)))
    mae  = float(np.mean(np.abs(y_true - y_pred_mean)))
    nll  = float(-model.log_likelihood())
    return {"rmse": rmse, "mae": mae, "neg_loglik": nll}

def predict(model: GPy.core.GP, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu, var = model.predict(X)
    return mu, var

# === NEW: data-driven hyperparameter bounds ===
def hp_bounds_from_data(df: pd.DataFrame, features: list[str], target: str) -> dict:
    # feature scales from 1st–99th percentiles
    p01 = df[features].quantile(0.01)
    p99 = df[features].quantile(0.99)
    scale = (p99 - p01).replace(0, 1e-12).values  # avoid zeros
    # lengthscales in same units as inputs
    k1_ls_lo = scale * 0.1
    k1_ls_hi = scale * 10.0
    # first feature is Vdot
    kV_ls_lo = float(k1_ls_lo[0])
    kV_ls_hi = float(k1_ls_hi[0])

    # output variance heuristics
    y = df[[target]].to_numpy().flatten()
    vy = float(np.var(y))
    var_lo = max(vy * 1e-2, 1e-10)
    var_hi = max(vy * 10.0, 1e-5)

    bounds = {
        "kV_lengthscale": [kV_ls_lo, kV_ls_hi],
        "kV_variance":    [var_lo, var_hi],
        "k1_lengthscales":[[float(a), float(b)] for a,b in zip(k1_ls_lo[1:], k1_ls_hi[1:])],
        "k1_variance":    [var_lo, var_hi],
        "noise_variance": [1e-12, max(vy, 1e-9)]
    }
    return bounds

