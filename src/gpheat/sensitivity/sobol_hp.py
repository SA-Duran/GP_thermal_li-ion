from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from SALib.sample import saltelli
from SALib.analyze import sobol

from gpheat.models.gpy_gp import load_dataframe, prepare_xy, make_model, fit, predict, metrics, hp_bounds_from_data
from gpheat.models.kernels import build_kernel
from gpheat.logger import get_logger
from gpheat.utils.mlflow_utils import (
    try_mlflow_start_run, try_mlflow_log_params, try_mlflow_log_metrics, try_mlflow_log_artifact
)
from gpheat.plot.sobol_plots import plot_sobol_bars, plot_sobol_heatmap
from gpheat.utils.mem import hard_cleanup

# -> plot directly from DataFrame (no re-read)
from gpheat.plot.sobol_plots import plot_sobol_bars_df, plot_sobol_heatmap_df


logger = get_logger(__name__)

def _pack_bounds(bnd: dict, input_dims: int): #-> (List[str], np.ndarray, np.ndarray) :
    names = ["kV_lengthscale", "kV_variance",
             "k1_lengthscale_1","k1_lengthscale_2","k1_lengthscale_3","k1_lengthscale_4",
             "k1_variance"
             #,"noise_variance"
             ]
    lo = [bnd["kV_lengthscale"][0], bnd["kV_variance"][0]]
    lo += [r[0] for r in bnd["k1_lengthscales"]] + [bnd["k1_variance"][0]]
    #lo += [r[0] for r in bnd["k1_lengthscales"]] + [bnd["k1_variance"][0], bnd["noise_variance"][0]]
    hi = [bnd["kV_lengthscale"][1], bnd["kV_variance"][1]]
    hi += [r[1] for r in bnd["k1_lengthscales"]] + [bnd["k1_variance"][1]]
    
    
    #hi += [r[1] for r in bnd["k1_lengthscales"]] + [bnd["k1_variance"][1], bnd["noise_variance"][1]]
    return names, np.array(lo, float), np.array(hi, float)

def sobol_hyperparams(
    *,
    cfg: Dict[str, Any],
    mode: str,
    kernel_kind: str,
    csv_path: Path,       # TRAIN CSV
    out_dir: Path,
    n_samples: int,
    refit_iters: int,
    seed: int = 42,
    experiment_name: str = "gpheat_sobol_hp",
    eval_csv_path: Path | None = None  # <-- NEW
) -> Path:
    """Sample hyperparams in data-driven bounds, retrain GP per sample, evaluate on the SAME dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train = load_dataframe(str(csv_path))
    feats, target = cfg["gpy"]["features"], cfg["gpy"]["target"]
    ds_tr = prepare_xy(df_train, feats, target)

    # bounds from TRAIN
    bnd = hp_bounds_from_data(df_train, feats, target)
    # optional TEST for metrics
    if eval_csv_path and Path(eval_csv_path).exists():
        df_te = load_dataframe(str(eval_csv_path))
        ds_te = prepare_xy(df_te, feats, target)
    else:
        ds_te = ds_tr  # fallback to TRAIN if no test provided
        logger.warning("sobol-hp: no eval_csv_path provided or not found; using TRAIN for metric")

    
    names, lo, hi = _pack_bounds(bnd, len(feats))
    D = len(names)

    problem = {"num_vars": D, "names": names, "bounds": np.stack([lo, hi], axis=1)}
    Xs = saltelli.sample(problem, n_samples, calc_second_order=False)  # (N, D)

    y_metric = []
    rows = []
    pbar = tqdm(range(Xs.shape[0]), desc=f"Sobol-HP[{mode}/{kernel_kind}]")

    fixed_noise = float(cfg["gpy"].get("noise_variance", 0.0))
    for i in pbar:
        sample = Xs[i]
        fx = {
            "kV_lengthscale": float(sample[0]),
            "kV_variance":    float(sample[1]),
            "k1_lengthscales":[float(sample[2]), float(sample[3]), float(sample[4]), float(sample[5])],
            "k1_variance":    float(sample[6])
            #"noise_variance": float(sample[7]),
        }

        kern = build_kernel(kernel_kind, fx, input_dims=len(feats))
        gp = mu = var = None
        m = {"rmse": np.nan, "mae": np.nan, "neg_loglik": np.nan}

        def _train_and_eval():
            nonlocal gp, mu, var, m
            gp = make_model(ds_tr.X, ds_tr.y, kern, noise_variance=fixed_noise)
            #fit(gp, iters=refit_iters)
            mu, var = predict(gp, ds_te.X)
            m = metrics(ds_te.y, mu, gp)

        try:
            _train_and_eval()
        except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
            logger.warning(f"OOM at sample {i}: {e.__class__.__name__} — cleaning & retrying once")
            hard_cleanup(gp, mu, var, kern)
            kern = build_kernel(kernel_kind, fx, input_dims=len(feats))  # rebuild kernel (GC-ed)
            try:
                _train_and_eval()
            except (MemoryError, np.core._exceptions._ArrayMemoryError) as e2:
                logger.error(f"Sample {i}: OOM again — skipping this point.")
                # m se queda con NaN

        # Registra (aunque sean NaN; luego filtramos antes de SALib)
        y_metric.append(m["rmse"])
        rows.append({"i": i, **fx, **m})

        # MLflow (igual que antes)
        run = try_mlflow_start_run(experiment_name, run_name=f"{mode}-{kernel_kind}-i{i}",
                                   tags={"mode": mode, "kernel": kernel_kind, "type": "sobol_hp"})
        try_mlflow_log_params({**fx, "refit_iters": refit_iters, "N": len(ds_tr.X)})
        try_mlflow_log_metrics(m, step=i)
        if run:
            import mlflow; mlflow.end_run()

        # Limpieza dura SIEMPRE (éxito o fallo) para evitar fuga de arrays grandes
        hard_cleanup(gp, mu, var, kern)
    
    y_metric = np.array(y_metric, float)
    valid = np.isfinite(y_metric)
    if valid.sum() < len(y_metric) // 2:
        logger.warning(f"Muchos puntos OOM: válidos={int(valid.sum())}/{len(y_metric)}; "
                       "Sobol puede ser ruidoso.")
    # SALib no acepta NaN → filtra
    Xs_valid = Xs[valid]
    y_valid = y_metric[valid]
    
    Si = sobol.analyze(problem, y_metric, calc_second_order=False, print_to_console=False)
    # summary CSV
    summary = pd.DataFrame({
        "name": names,
        "S1": Si["S1"], "S1_conf": Si["S1_conf"],
        "ST": Si["ST"], "ST_conf": Si["ST_conf"],
    })
    out_csv = out_dir / f"sensitivity_{mode}_{kernel_kind}_hp.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    # extra safety
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected to write Sobol CSV but missing: {out_csv}")

    # ... dentro de sobol_hyperparams(), al final, después de escribir out_csv:
    bar_s1 = out_dir / f"sensitivity_{mode}_{kernel_kind}_hp_S1_bar.png"
    bar_st = out_dir / f"sensitivity_{mode}_{kernel_kind}_hp_ST_bar.png"
    heat   = out_dir / f"sensitivity_{mode}_{kernel_kind}_hp_heatmap.png"

    try:
        plot_sobol_bars(out_csv, bar_s1, metric="S1", title=f"Sobol S1 — {mode}/{kernel_kind} (HP)")
        plot_sobol_bars(out_csv, bar_st, metric="ST", title=f"Sobol ST — {mode}/{kernel_kind} (HP)")
        plot_sobol_heatmap(out_csv, heat, title=f"Sobol S1/ST — {mode}/{kernel_kind} (HP)")
    except:
        
        plot_sobol_bars_df(summary, bar_s1, metric="S1", title=f"Sobol S1 — {mode}/{kernel_kind} (HP)")
        plot_sobol_bars_df(summary, bar_st, metric="ST", title=f"Sobol ST — {mode}/{kernel_kind} (HP)")
        plot_sobol_heatmap_df(summary, heat, title=f"Sobol S1/ST — {mode}/{kernel_kind} (HP)")
    # Si MLflow está activo, sube las imágenes
    try_mlflow_log_artifact(bar_s1)
    try_mlflow_log_artifact(bar_st)
    try_mlflow_log_artifact(heat)    
    
    summary.to_csv(out_csv, index=False)
    logger.info(f"Sobol HP summary  {out_csv}")
    return out_csv
