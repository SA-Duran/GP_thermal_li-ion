import click
from gpheat.config import load_config, get_paths
from gpheat.pipeline import run_pipeline
from gpheat.experiments.generate_data import generate_and_save
from gpheat.logger import get_logger

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

if __name__ == "__main__":
    cli()
