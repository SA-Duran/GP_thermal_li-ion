import click
from gpheat.config import load_config, get_paths
from gpheat.pipeline import run_pipeline
from gpheat.experiments.generate_data import generate_and_save
from gpheat.config import load_config, get_paths
from gpheat.models.gpy_gp import (
    load_dataframe, prepare_xy, build_kernel_charge, build_kernel_discharge,
    make_model, set_bounds, fit, predict, metrics
)
from gpheat.models.checkpoints import save_pickle, load_pickle
from gpheat.plot.gpy_results import plot_dt_dt, plot_qrev
from gpheat.logger import get_logger
import numpy as np
from pathlib import Path
from gpheat.experiments.make_test_sets import make_test_csvs
from gpheat.evaluation.eval_gpy import eval_runs_with_model

from gpheat.models.kernels import build_kernel
from gpheat.evaluation.eval_gpy import eval_runs_with_model
from gpheat.sensitivity.sobol_hp import sobol_hyperparams
from gpheat.sensitivity.sobol_features import sobol_features
from gpheat.plot.sobol_plots import plot_sobol_bars, plot_sobol_heatmap

logger = get_logger("gpheat", log_file="gpheat.log")

@click.group()
def cli():
    """Command-line interface for gpheat."""

@cli.command()
@click.option("--config", default="config/config.yaml", help="Path to config YAML.")
@click.option("--root", default=".", help="Project root dir.")
def run(config: str, root: str):
    cfg = load_config(config)
    paths = get_paths(cfg, root)
    run_pipeline(cfg, paths)

@cli.command(name="gen-data")
@click.option("--config", default="config/config.yaml", help="Path to config YAML.")
@click.option("--root", default=".", help="Project root dir.")
def gen_data(config: str, root: str):
    """Generate dataset by running the SPMe thermal grid and save CSV under artifacts."""
    cfg = load_config(config)
    paths = get_paths(cfg, root)
    out_csv = generate_and_save(cfg, paths)
    click.echo(f"Dataset written to: {out_csv}")
    
    cfg["grid"]["modes"]=cfg["grid"]["charge_mode"]
    cfg["dataset"]["name"]=cfg["dataset"]["charge_name"]
    charge_csv = generate_and_save(cfg, paths)
    click.echo(f"Dataset written to: {charge_csv}")
    
    cfg["grid"]["modes"]=cfg["grid"]["discharge_mode"]
    cfg["dataset"]["name"]=cfg["dataset"]["discharge_name"]
    discharge_csv = generate_and_save(cfg, paths)
    click.echo(f"Dataset written to: {discharge_csv}")

@click.command("fit-gpy")
@click.option("--mode", type=click.Choice(["charge","discharge"]), required=True)
@click.option("--kernel", type=click.Choice(["mult","add","mixed"]), default=None, help="Kernel family")
@click.option("--optimize/--no-optimize", default=False)
@click.option("--iters", default=200, show_default=True)
@click.option("--config", default="config/config.yaml")
@click.option("--root", default=".")
@click.option("--out", default=None, help="Output model filename under artifacts/models/")
def fit_gpy(mode, kernel, optimize, iters, config, root, out):
    cfg = load_config(config); paths = get_paths(cfg, root)
    feats = cfg["gpy"]["features"]; target = cfg["gpy"]["target"]
    kernel_kind = kernel or cfg["gpy"]["kernels"]["default"]
    csv_path = cfg["gpy"]["dataset"]["charge_csv" if mode=="charge" else "discharge_csv"]

    df = load_dataframe(csv_path)
    ds = prepare_xy(df, feats, target)

    fixed = cfg["gpy"][f"{mode}_fixed"]
    kern = build_kernel(kernel_kind, fixed, input_dims=len(feats))

    gp = make_model(ds.X, ds.y, kern, noise_variance=fixed["noise_variance"])
    if optimize:
        fit(gp, iters=iters)

    mu, var = predict(gp, ds.X)
    m = metrics(ds.y, mu, gp)
    logger.info(f"{mode}/{kernel_kind} train metrics: {m}")

    name = out or f"{mode}_{kernel_kind}.pkl"
    save_path = paths.artifacts_root / "models" / name
    save_pickle(gp, save_path)
    logger.info(f"Saved model  {save_path}")
    

@click.command("make-test")
@click.option("--config", default="config/config.yaml")
@click.option("--root", default=".")
def make_test(config, root):
    """Generate fixed test CSVs under artifacts/test/ using test_conditions in config."""
    logger.info(f"test data making ")
    cfg = load_config(config); paths = get_paths(cfg, root)
    outs = make_test_csvs(cfg, paths)
    click.echo(f"Test CSVs: {outs}")


@click.command("eval-gpy")
@click.option("--mode", type=click.Choice(["charge","discharge"]), required=True)
@click.option("--model", default=None, help="Path to saved .pkl (defaults to config gpy.models.<mode>)")
@click.option("--csv", default=None, help="CSV to evaluate (defaults to artifacts/test/<mode>_test.csv)")
@click.option("--config", default="config/config.yaml")
@click.option("--root", default=".")
def eval_gpy(mode, model, csv, config, root):
    """Evaluate a saved model run-by-run over configured conditions; save per-run plots & summary."""
    cfg = load_config(config) 
    paths = get_paths(cfg, root)
    feats = cfg["gpy"]["features"]; target = cfg["gpy"]["target"]
    logger.info(f"dataset loaded from test")
    # model path default
    default_model = paths.artifacts_root / "models" / cfg["gpy"]["models"][mode]
    model_path = Path(model) if model else default_model
    logger.info(f"model loaded for predictions ")
    # CSV default: artifacts/test/<mode>_test.csv
    default_csv = Path(paths.__dict__.get("test_dir", paths.artifacts_root / "test")) / f"{mode}_test.csv"
    csv_path = Path(csv) if csv else default_csv

    runs = cfg["test_conditions"][mode]
    out_dir = Path(paths.reports_dir) / "eval" / mode
    logger.info(f"starting evaluating runs")
    eval_runs_with_model(
        model_path=model_path,
        csv_path=csv_path,
        features=feats,
        target=target,
        runs=runs,
        out_dir=out_dir,
        prefix=mode
    )


@click.command("predict-gpy")
@click.option("--mode", type=click.Choice(["charge","discharge"]), required=True)
@click.option("--model", required=True, help="Path to saved .pkl")
@click.option("--csv", required=True, help="CSV to predict on")
@click.option("--config", default="config/config.yaml")
@click.option("--root", default=".")
def predict_gpy(mode, model, csv, config, root):
    cfg = load_config(config); paths = get_paths(cfg, root)
    feats = cfg["gpy"]["features"]; target = cfg["gpy"]["target"]
    gp = load_pickle(model)
    df = load_dataframe(csv)
    X = df[feats].to_numpy()
    mu, var = gp.predict(X)
    m = {}
    if target in df.columns:
        y = df[[target]].to_numpy()
        from gpheat.models.gpy_gp import metrics as _met
        m = _met(y, mu, gp)
        logger.info(f"{mode} metrics on CSV: {m}")
    from gpheat.plot.gpy_results import plot_dt_dt, plot_qrev
    import numpy as np
    plot_dt_dt(df, mu, np.sqrt(var), paths.figures_dir / f"{mode}_predict_dt_dt.png", f"dT/dt ({mode})")
    plot_qrev(df, mu, np.sqrt(var), paths.figures_dir / f"{mode}_predict_qrev.png", f"Qrev ({mode})")



# === Sobol HP (retrain per sample) ===
@click.command("sobol-hp")
@click.option("--mode", type=click.Choice(["charge","discharge"]), required=True)
@click.option("--kernel", type=click.Choice(["mult","add","mixed"]), default=None)
@click.option("--config", default="config/config.yaml")
@click.option("--root", default=".")
def sobol_hp_cmd(mode, kernel, config, root):
    cfg = load_config(config); paths = get_paths(cfg, root)
    kernel_kind = kernel or cfg["gpy"]["kernels"]["default"]
    csv_train = Path(cfg["gpy"]["dataset"]["charge_csv" if mode=="charge" else "discharge_csv"])
    csv_test  = Path(paths.__dict__.get("test_dir", paths.artifacts_root / "test")) / f"{mode}_test.csv"

    out_dir = Path(cfg["paths"]["sobol_dir"]) / "hp" / mode / kernel_kind
    n = int(cfg["sobol"]["n_samples"]); iters = int(cfg["sobol"]["refit_iters"]); seed = int(cfg["sobol"]["random_seed"])
    sobol_hyperparams(
        cfg=cfg, mode=mode, kernel_kind=kernel_kind,
        csv_path=csv_train, out_dir=out_dir,
        n_samples=n, refit_iters=iters, seed=seed,
        eval_csv_path=csv_test  # <-- use TEST for metrics
    )

# === Sobol features (predict using saved model) ===
@click.command("sobol-x")
@click.option("--mode", type=click.Choice(["charge","discharge"]), required=True)
@click.option("--model", required=True, help="Path to saved .pkl to analyze")
@click.option("--config", default="config/config.yaml")
@click.option("--root", default=".")
def sobol_x_cmd(mode, model, config, root):
    cfg = load_config(config); paths = get_paths(cfg, root)
    csv_path = Path(cfg["gpy"]["dataset"]["charge_csv" if mode=="charge" else "discharge_csv"])
    out_dir = Path(cfg["paths"]["sobol_dir"]) / "features" / mode
    n = int(cfg["sobol"]["n_samples"])*2; seed = int(cfg["sobol"]["random_seed"])
    sobol_features(
        cfg=cfg, mode=mode, model_path=Path(model), csv_path=csv_path,
        out_dir=out_dir, n_samples=n, seed=seed
    )
    
@click.command("plot-sobol")
@click.option("--summary", required=True, help="Ruta al CSV de sensibilidad (S1/ST).")
def plot_sobol_cmd(summary):
    from pathlib import Path
    s = Path(summary)
    out_dir = s.parent
    # heurística para título
    title_base = s.stem.replace("_", " ")
    plot_sobol_bars(s, out_dir / (s.stem + "_S1_bar.png"), metric="S1", title=f"{title_base} — S1")
    plot_sobol_bars(s, out_dir / (s.stem + "_ST_bar.png"), metric="ST", title=f"{title_base} — ST")
    plot_sobol_heatmap(s, out_dir / (s.stem + "_heatmap.png"), title=f"{title_base} — S1/ST")
    print(f"Plots guardados en {out_dir}")
    
# register
# add to your click group
cli.add_command(fit_gpy)
cli.add_command(predict_gpy)
cli.add_command(make_test)
cli.add_command(eval_gpy)
cli.add_command(sobol_hp_cmd)
cli.add_command(sobol_x_cmd)
cli.add_command(plot_sobol_cmd)

if __name__ == "__main__":
    cli()



