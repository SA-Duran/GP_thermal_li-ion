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
@click.option("--optimize/--no-optimize", default=False, help="Optimize hyperparameters within bounds")
@click.option("--iters", default=200, show_default=True)
@click.option("--config", default="config/config.yaml")
@click.option("--root", default=".")
@click.option("--out", default=None, help="Override output model filename (under paths.models_dir)")
def fit_gpy(mode, optimize, iters, config, root, out):
    """Train a GPy model and save a pickle. (Prediction is done separately.)"""
    cfg = load_config(config); paths = get_paths(cfg, root)
    feats = cfg["gpy"]["features"] 
    target = cfg["gpy"]["target"]
    csv_path = cfg["gpy"]["dataset"]["charge_csv" if mode=="charge" else "discharge_csv"]

    df = load_dataframe(csv_path)
    ds = prepare_xy(df, feats, target)
    logger.info(f"{mode.capitalize()} data loaded, starting kernel and model making")
    kernel = build_kernel_charge(cfg, len(feats)) if mode=="charge" else build_kernel_discharge(cfg, len(feats))
    noise = cfg["gpy"][f"{mode}_fixed"]["noise_variance"]
    model = make_model(ds.X, ds.y, kernel, noise_variance=noise)
    logger.info(f"{mode.capitalize()} model ready starting predictions")
    if optimize:
        set_bounds(model, cfg)
        fit(model, iters=iters)

    mu, var = predict(model, ds.X)
    m = metrics(ds.y, mu, model)
    logger.info(f"{mode.capitalize()} train metrics: {m}")

    # Save model
    model_name = out or cfg["gpy"]["models"][mode]
    out_path = paths.root / paths.__dict__.get("artifacts_root").relative_to(paths.root) / "models" / model_name
    # The above ensures we save under artifacts/models even if models_dir changes later
    out_path = paths.artifacts_root / "models" / model_name if not model_name.startswith(str(paths.artifacts_root)) else Path(model_name)
    saved = save_pickle(model, out_path)
    logger.info(f"Saved model to {saved}")

    # Plots on training data (optional but useful sanity check)
    #from gpheat.plot.gpy_results import plot_dt_dt, plot_qrev
    #y_std = np.sqrt(var)
    #name = f"{mode}_gpy_train"
    #plot_dt_dt(df, mu, y_std, paths.figures_dir / f"{name}_dt_dt.png", title=f"dT/dt ({mode})")
    #plot_qrev(df, mu, y_std, paths.figures_dir / f"{name}_qrev.png", title=f"Qrev ({mode})")

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

# register
# add to your click group
cli.add_command(fit_gpy)
cli.add_command(make_test)
cli.add_command(eval_gpy)



if __name__ == "__main__":
    cli()


