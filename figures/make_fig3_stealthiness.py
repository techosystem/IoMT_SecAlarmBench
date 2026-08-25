#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIGURE 3 (OPTIONAL) -- Per-morphology stealthiness curves.

Detection rate at the matched 10% false-alarm budget as severity decreases
L4 -> L1, one panel per morphology, one line per detector.  This is the figure
that would let Section V keep its original promise of "detection curves
against decreasing severity, one axis per morphology".  It does NOT fit in the
10-page limit; use it for the arXiv/extended version, or swap it in for
Table IV's left panel if you would rather show the gradient than the numbers.

>>> THE NUMBERS BELOW ARE PLACEHOLDERS. <<<
Replace DATA with your own per-(morphology, level) detection rates from
rq2_discrimination.py / eval_harness.py at the matched budget.  Every value
must come from the same run as Table IV (20260730-222727) or the figure will
silently disagree with the table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 6.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4, "pdf.fonttype": 42,
})

LEVELS = ["L4", "L3", "L2", "L1"]          # decreasing severity, left to right
MORPHS = ["Spike", "Stuck-at", "Drift", "Bias"]
DETECTORS = ["Isolation Forest", "OmniAnomaly", "GDN", "LSTM-AE", "TranAD", "USAD"]

STYLE = {
    "Isolation Forest": dict(color="#c0392b", marker="o", ls="-",  lw=0.9),
    "OmniAnomaly":      dict(color="#2c6fbb", marker="s", ls="-",  lw=0.7),
    "GDN":              dict(color="#2e8b57", marker="^", ls="--", lw=0.7),
    "LSTM-AE":          dict(color="#8e44ad", marker="v", ls="--", lw=0.7),
    "TranAD":           dict(color="#d68910", marker="D", ls=":",  lw=0.7),
    "USAD":             dict(color="#707070", marker="x", ls=":",  lw=0.7),
}

# ---------------------------------------------------------------------------
# PLACEHOLDER DATA -- DATA[morphology][detector] = [L4, L3, L2, L1]
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
DATA = {
    m: {d: np.clip(np.linspace(hi, lo, 4) + rng.normal(0, 0.015, 4), 0, 1)
        for d, hi, lo in zip(DETECTORS,
                             [0.95, 0.90, 0.88, 0.92, 0.93, 0.86],
                             [0.18, 0.15, 0.14, 0.13, 0.12, 0.11])}
    for m in MORPHS
}
CHANCE = 0.10       # matched false-alarm budget = chance detection rate


def main():
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.80), sharey=True)
    x = np.arange(len(LEVELS))

    for ax, m in zip(axes, MORPHS):
        for d in DETECTORS:
            ax.plot(x, DATA[m][d], markersize=2.1, markeredgewidth=0.6,
                    **STYLE[d], label=d)
        ax.axhline(CHANCE, color="k", lw=0.5, ls=(0, (4, 3)), alpha=0.6)
        ax.text(0.02, CHANCE + 0.025, "10% FAR budget", fontsize=4.8,
                color="k", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS)
        ax.set_xlim(-0.18, len(LEVELS) - 0.82)
        ax.set_ylim(0, 1.0)
        ax.set_title(m, fontsize=7.0, pad=2.5, fontweight="bold")
        ax.set_xlabel("severity (decreasing)", fontsize=5.8, labelpad=1.2)
        ax.tick_params(labelsize=5.6, length=1.8, pad=1.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", lw=0.3, alpha=0.25)

    axes[0].set_ylabel("detection rate", fontsize=6.0, labelpad=2.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.4, frameon=False, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.045), handlelength=1.8,
               handletextpad=0.4, columnspacing=1.4)

    fig.subplots_adjust(left=0.052, right=0.997, top=0.86, bottom=0.26,
                        wspace=0.10)
    fig.savefig("fig3_stealthiness.pdf", dpi=600, bbox_inches="tight",
                pad_inches=0.012)
    print("wrote fig3_stealthiness.pdf  [PLACEHOLDER DATA -- replace DATA dict]")


if __name__ == "__main__":
    main()
