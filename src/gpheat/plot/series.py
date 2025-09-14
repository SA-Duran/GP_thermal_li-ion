from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, out_png: str | Path):
    out_png = Path(out_png)
    plt.figure()
    plt.plot(np.asarray(y_true), label="true")
    plt.plot(np.asarray(y_pred), label="pred")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
