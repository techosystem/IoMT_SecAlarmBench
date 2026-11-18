#!/usr/bin/env python3
"""
Fig. 2: host-matched AUROC by morphology and severity level, per attacker setting.

Values are AUROC x 100 (50 = chance), averaged over the six learned detectors and
the seeds in the grid file written by extras.run_all.

    python figures/make_fig2_grid.py
    python figures/make_fig2_grid.py --grid results/paper/grid_morph_level_v1.csv --out fig2_grid.pdf
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEARNED = ("IsolationForest", "USAD", "LSTM-AE", "GDN", "TranAD", "OmniAnomaly")
SETTINGS = ("C1-PPG", "C1-ECG", "C2", "C3")
MORPHS = (("bias", "bias"), ("stuck_at", "stuck-at"), ("drift", "drift"), ("spike", "spike"))
ROWS = ([("fdi", m, lab) for m, lab in MORPHS] + [("fdi", "rate_warp", "rate warp")]
        + [None] + [("spoofing", m, lab) for m, lab in MORPHS])

rcParams.update({"font.family": "serif",
                 "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                 "font.size": 8, "pdf.fonttype": 42})


def load_grid(path):
    g = pd.read_csv(path)
    g = g[g.detector.isin(LEARNED)]
    return g.groupby(["setting", "family", "morphology", "level"]).hmAUROC.mean()


def draw(grid, out):
    cmap = plt.get_cmap("Blues")
    n_rows, n_lv, gap = len(ROWS), 4, 0.5
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for si, setting in enumerate(SETTINGS):
        x0 = si * (n_lv + gap)
        ax.text(x0 + n_lv / 2, -0.25, setting, ha="center", va="bottom", fontsize=8)
        for ri, row in enumerate(ROWS):
            if row is None:
                continue
            fam, morph, _ = row
            for li in range(n_lv):
                key = (setting, fam, morph, li + 1)
                if key not in grid.index:
                    continue
                v = 100 * grid[key]
                shade = np.clip((v - 50) / 50, 0, 1)
                ax.add_patch(plt.Rectangle((x0 + li + 0.04, ri + 0.04), 0.92, 0.92,
                                           color=cmap(0.08 + 0.92 * shade), lw=0))
                ax.text(x0 + li + 0.5, ri + 0.52, f"{v:.0f}", ha="center", va="center",
                        fontsize=4.8 if v >= 99.5 else 5.5,
                        color="white" if shade > 0.6 else "black")
        for li in range(n_lv):
            ax.text(x0 + li + 0.5, n_rows + 0.3, f"L{li + 1}", ha="center", va="top", fontsize=6)
    for ri, row in enumerate(ROWS):
        if row is not None:
            ax.text(-0.2, ri + 0.5, row[2], ha="right", va="center", fontsize=7)
    ax.text(-4.6, 2.5, "FDI", rotation=90, ha="center", va="center", fontsize=8)
    ax.text(-4.6, 8.0, "Spoofing", rotation=90, ha="center", va="center", fontsize=8)
    ax.set_xlim(-5.0, len(SETTINGS) * (n_lv + gap) - gap)
    ax.set_ylim(n_rows + 0.9, -0.9)
    ax.axis("off")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"[saved] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Draw Fig. 2 from the morphology x level grid.")
    p.add_argument("--grid", default=str(ROOT / "results/paper/grid_morph_level_v1.csv"))
    p.add_argument("--out", default="fig2_grid.pdf")
    a = p.parse_args()
    draw(load_grid(a.grid), a.out)
