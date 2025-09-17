from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gpheat.utils.labels import prettify_list  # <-- nuevo

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def plot_sobol_bars(summary_csv: Path, out_png: Path, metric: str, title: str, top_n: int | None = None) -> Path:
    df = pd.read_csv(summary_csv)
    if top_n:
        df = df.sort_values(metric, ascending=False).head(top_n)
    else:
        df = df.sort_values(metric, ascending=False)

    # Aplica etiquetas bonitas a la columna 'name'
    labels = prettify_list(df["name"].tolist())

    conf_col = f"{metric}_conf" if f"{metric}_conf" in df.columns else None
    x = np.arange(len(df))
    vals = df[metric].to_numpy()
    errs = df[conf_col].to_numpy() if conf_col else None

    plt.figure(figsize=(10, 5))
    plt.bar(x, vals, yerr=errs, capsize=3 if errs is not None else 0)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    _ensure_dir(out_png)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png

def plot_sobol_heatmap(summary_csv: Path, out_png: Path, title: str) -> Path:
    df = pd.read_csv(summary_csv).sort_values("ST", ascending=False)
    labels = prettify_list(df["name"].tolist())
    data = np.vstack([df["S1"].to_numpy(), df["ST"].to_numpy()])

    plt.figure(figsize=(max(8, 0.6*len(df)), 3.5))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks([0,1], ["S1","ST"])
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title)
    _ensure_dir(out_png)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png


def plot_sobol_bars_df(summary_df: pd.DataFrame, out_png: Path, metric: str, title: str, top_n: int | None = None) -> Path:
    df = summary_df.copy()
    if top_n:
        df = df.sort_values(metric, ascending=False).head(top_n)
    else:
        df = df.sort_values(metric, ascending=False)
    labels = prettify_list(df["name"].tolist())
    conf_col = f"{metric}_conf" if f"{metric}_conf" in df.columns else None
    x = np.arange(len(df))
    vals = df[metric].to_numpy()
    errs = df[conf_col].to_numpy() if conf_col else None

    plt.figure(figsize=(10, 5))
    plt.bar(x, vals, yerr=errs, capsize=3 if errs is not None else 0)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(metric); plt.title(title); plt.grid(axis="y", alpha=0.3)
    _ensure_dir(out_png); plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    return out_png

def plot_sobol_heatmap_df(summary_df: pd.DataFrame, out_png: Path, title: str) -> Path:
    df = summary_df.copy().sort_values("ST", ascending=False)
    labels = prettify_list(df["name"].tolist())
    data = np.vstack([df["S1"].to_numpy(), df["ST"].to_numpy()])
    plt.figure(figsize=(max(8, 0.6 * len(df)), 3.5))
    im = plt.imshow(data, aspect="auto"); plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks([0, 1], ["S1", "ST"]); plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title); _ensure_dir(out_png); plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    return out_png
