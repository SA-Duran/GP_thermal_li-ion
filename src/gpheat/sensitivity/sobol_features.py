from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm

from gpheat.models.checkpoints import load_pickle
from gpheat.models.gpy_gp import load_dataframe
from gpheat.logger import get_logger
from gpheat.plot.sobol_plots import plot_sobol_bars, plot_sobol_heatmap
from gpheat.utils.mlflow_utils import try_mlflow_start_run, try_mlflow_log_artifact
from gpheat.plot.sobol_plots import plot_sobol_bars_df, plot_sobol_heatmap_df



logger = get_logger(__name__)

def feature_bounds_from_data(df: pd.DataFrame, features: list[str]): #-> (List[str], np.ndarray, np.ndarray):
    p01 = df[features].quantile(0.01)
    p99 = df[features].quantile(0.99)
    lo = p01.values.astype(float)
    hi = p99.values.astype(float)
    # ensure positive width
    hi = np.maximum(hi, lo + 1e-12)
    return features, lo, hi

def feature_groups(cfg: Dict[str, Any], features: list[str]) -> List[int] | None:
    """
    Devuelve una lista de enteros (1..G) del mismo largo que 'features'.
    Si una feature no está en ningún grupo del config, se le asigna un grupo nuevo.
    """
    groups_cfg = cfg["sobol"].get("groups", {})
    if not groups_cfg:
        return None

    # Asignar IDs a labels del config: 1..G
    label_to_id: Dict[str, int] = {}
    next_gid = 1
    for label in groups_cfg.keys():
        label_to_id[label] = next_gid
        next_gid += 1

    groups: List[int] = []
    for f in features:
        assigned = False
        for label, cols in groups_cfg.items():
            if f in cols:
                groups.append(label_to_id[label])
                assigned = True
                break
        if not assigned:
            groups.append(next_gid)
            next_gid += 1
    return groups

def sobol_features(
    *,
    cfg: Dict[str, Any],
    mode: str,
    model_path: Path,
    csv_path: Path,
    out_dir: Path,
    n_samples: int,
    seed: int = 42,
    experiment_name: str = "gpheat_sobol_features"
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataframe(str(csv_path))
    feats = cfg["gpy"]["features"]
    names, lo, hi = feature_bounds_from_data(df, feats)
    gp = load_pickle(model_path)
    
    groups = feature_groups(cfg, feats)  # ahora es List[int] o None
    if groups is not None and len(groups) != len(names):
        raise ValueError(f"groups length {len(groups)} != names length {len(names)}")

    # SALib espera 'bounds' como lista de [lo, hi]
    bounds = [[float(a), float(b)] for a, b in zip(lo, hi)]

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    if groups is not None:
        problem["groups"] = groups  # lista, no np.array

    Xs = saltelli.sample(problem, n_samples, calc_second_order=False)
    y = np.zeros(Xs.shape[0], dtype=float)


    for i in tqdm(range(Xs.shape[0]), desc=f"Sobol-X[{mode}]"):
        mu, _ = gp.predict(Xs[i:i+1, :])  # GPy model
        y[i] = float(mu[0,0])

    Si = sobol.analyze(problem, y, calc_second_order=False, print_to_console=False)

    # ---- Build labels aligned with SALib outputs ----
    if groups is None:
        names_out = names  # one index per feature
    else:
        # SALib aggregates by group id; order is sorted(unique(groups))
        uniq_ids = sorted(set(groups))

        # Build an id -> display label map:
        #   - If the id belongs to a configured group label (e.g. "voltage", "overpotentials"),
        #     use that label
        #   - Otherwise use the pretty label of the first feature in that group
        from gpheat.utils.labels import prettify
        # reconstruct config label -> id map like feature_groups() did
        cfg_groups = cfg["sobol"].get("groups", {})
        label_to_id = {}
        nid = 1
        for label in cfg_groups.keys():
            label_to_id[label] = nid
            nid += 1

        id_to_label = {}
        # prefer config labels
        for label, gid in label_to_id.items():
            if gid in uniq_ids:
                id_to_label[gid] = label

        # fill any remaining ids using the first feature carrying that id
        for gid in uniq_ids:
            if gid not in id_to_label:
                # pick first feature with this group id
                for f, g in zip(names, groups):
                    if g == gid:
                        id_to_label[gid] = prettify(f)
                        break

        names_out = [id_to_label[g] for g in uniq_ids]

    # ---- Assemble summary with matching lengths ----
    # SALib returns arrays sized to num groups (if groups) or num vars (if no groups)
    S1 = Si["S1"]
    S1c = Si.get("S1_conf", np.full_like(S1, np.nan))
    ST = Si["ST"]
    STc = Si.get("ST_conf", np.full_like(ST, np.nan))

    summary = pd.DataFrame({
        "name": names_out,
        "S1": S1, "S1_conf": S1c,
        "ST": ST, "ST_conf": STc,
    })

    out_csv = out_dir / f"sensitivity_{mode}_features.csv"
    summary.to_csv(out_csv, index=False)
    logger.info(f"Sobol features summary {out_csv}")


    
    bar_s1 = out_dir / f"sensitivity_{mode}_features_S1_bar.png"
    bar_st = out_dir / f"sensitivity_{mode}_features_ST_bar.png"
    heat   = out_dir / f"sensitivity_{mode}_features_heatmap.png"

    try:
        plot_sobol_bars(out_csv, bar_s1, metric="S1", title=f"Sobol S1 — {mode} (X)")
        plot_sobol_bars(out_csv, bar_st, metric="ST", title=f"Sobol ST — {mode} (X)")
        plot_sobol_heatmap(out_csv, heat, title=f"Sobol S1/ST — {mode} (X)")
    except:
        plot_sobol_bars_df(out_csv, bar_s1, metric="S1", title=f"Sobol S1 — {mode} (X)")
        plot_sobol_bars_df(out_csv, bar_st, metric="ST", title=f"Sobol ST — {mode} (X)")
        plot_sobol_heatmap_df(out_csv, heat, title=f"Sobol S1/ST — {mode} (X)")
    # Si MLflow está activo, sube las imágene
    
    # MLflow (opcional)
    run = try_mlflow_start_run("gpheat_sobol_features", run_name=f"{mode}-features",
                            tags={"mode": mode, "type": "sobol_features"})
    try_mlflow_log_artifact(bar_s1)
    try_mlflow_log_artifact(bar_st)
    try_mlflow_log_artifact(heat)
    
    if run:
        import mlflow; mlflow.end_run()    
    
    return out_csv
