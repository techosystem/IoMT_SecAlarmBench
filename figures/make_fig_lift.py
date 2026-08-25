#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection lift over each attack subset's own chance floor.

Why this figure earns its space: Table IV's own caption warns that values are
comparable *down* a column but not *across* rows, because every subset has a
different prevalence and therefore a different no-skill AUPRC. Dividing by that
floor removes exactly that confound and makes the subsets comparable, which is
the only way to show the mechanism-preservation ordering empirically in one
picture. The figure is not a redrawn table; it is the normalisation the table
cannot perform.

  python make_fig_lift.py                 # single column (3.45in) -- default
  python make_fig_lift.py --width double  # full width (7.16in), figure*
  python make_fig_lift.py --no-errorbars  # if you prefer the cleaner look

ALL NUMBERS BELOW ARE TAKEN VERBATIM FROM MERGED TABLE IV (run 20260730-222727).
If you regenerate results, edit SUBSETS only -- lifts and error bars derive
from it, so the figure can never silently disagree with the table.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# threshold-free AUPRC, mean and sd over 5 seeds, per subset chance floor
#   subset -> (chance_floor, {detector: (auprc_mean, auprc_sd)})
# ---------------------------------------------------------------------------
SUBSETS = {
    "Replay":      (0.065, {"Isolation Forest": (0.110, 0.002),
                            "OmniAnomaly":      (0.103, 0.007),
                            "GDN":              (0.098, 0.000),
                            "LSTM-AE":          (0.088, 0.000),
                            "TranAD":           (0.084, 0.006),
                            "USAD":             (0.076, 0.010)}),
    "FDI-spike L1": (0.033, {"Isolation Forest": (0.054, 0.001),
                             "OmniAnomaly":      (0.053, 0.004),
                             "GDN":              (0.050, 0.000),
                             "LSTM-AE":          (0.049, 0.000),
                             "TranAD":           (0.044, 0.003),
                             "USAD":             (0.041, 0.004)}),
    "FDI-spike":   (0.122, {"Isolation Forest": (0.201, 0.006),
                            "OmniAnomaly":      (0.171, 0.006),
                            "GDN":              (0.169, 0.001),
                            "LSTM-AE":          (0.162, 0.000),
                            "TranAD":           (0.153, 0.010),
                            "USAD":             (0.142, 0.013)}),
    "FDI sust.":   (0.293, {"Isolation Forest": (0.743, 0.017),
                            "OmniAnomaly":      (0.658, 0.028),
                            "GDN":              (0.579, 0.005),
                            "LSTM-AE":          (0.678, 0.001),
                            "TranAD":           (0.759, 0.074),
                            "USAD":             (0.562, 0.045)}),
    "Spoof. sust.": (0.293, {"Isolation Forest": (0.964, 0.004),
                             "OmniAnomaly":      (0.930, 0.011),
                             "GDN":              (0.938, 0.003),
                             "LSTM-AE":          (0.976, 0.000),
                             "TranAD":           (0.970, 0.020),
                             "USAD":             (0.940, 0.014)}),
}

# Table IV row order, so figure and table read the same way.
DETECTORS = ["Isolation Forest", "OmniAnomaly", "GDN",
             "LSTM-AE", "TranAD", "USAD"]

# Okabe-Ito: colourblind-safe and still separable in greyscale conversion.
COLORS = {
    "Isolation Forest": "#0072B2",
    "OmniAnomaly":      "#E69F00",
    "GDN":              "#009E73",
    "LSTM-AE":          "#D55E00",
    "TranAD":           "#CC79A7",
    "USAD":             "#999999",
}
# Hatches survive B&W printing, which IEEE proceedings sometimes get.
HATCH = {"Isolation Forest": "", "OmniAnomaly": "///", "GDN": "\\\\\\",
         "LSTM-AE": "...", "TranAD": "xxx", "USAD": "---"}

PRESETS = {                     # (width_in, height_in, base_fontsize)
    "single": (3.45, 2.45, 6.2),
    "double": (7.16, 2.35, 7.4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", choices=("single", "double"), default="single")
    ap.add_argument("--no-errorbars", action="store_true")
    ap.add_argument("--hatch", action="store_true",
                    help="add hatch patterns for hard B&W printing (busier)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W, H, FS = PRESETS[args.width]
    out = args.out or "fig_lift_%s.pdf" % args.width

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": FS,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "hatch.linewidth": 0.35,
        "pdf.fonttype": 42,
    })

    names = list(SUBSETS.keys())
    x = np.arange(len(names), dtype=float)
    n = len(DETECTORS)
    bw = 0.80 / n

    fig, ax = plt.subplots(figsize=(W, H))

    # region at or below chance -- makes "close to chance" visually literal
    ax.axhspan(0, 1.0, color="#000000", alpha=0.045, lw=0, zorder=0)
    ax.axhline(1.0, color="k", lw=0.7, ls=(0, (4, 2.6)), zorder=4)

    for i, det in enumerate(DETECTORS):
        off = (i - (n - 1) / 2) * bw
        lift, err = [], []
        for nm in names:
            floor, d = SUBSETS[nm]
            mu, sd = d[det]
            lift.append(mu / floor)
            err.append(sd / floor)
        ax.bar(x + off, lift, bw * 0.92,
               color=COLORS[det], edgecolor="black", linewidth=0.35,
               hatch=(HATCH[det] if args.hatch else ""), label=det, zorder=3,
               yerr=None if args.no_errorbars else err,
               error_kw=dict(ecolor="black", elinewidth=0.45, capsize=0.9,
                             capthick=0.45, zorder=5))

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FS - 0.6)
    # absolute scale stays recoverable: print each subset's floor beneath it
    for xi, nm in zip(x, names):
        ax.text(xi, -0.115, "floor %.3f" % SUBSETS[nm][0],
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=FS - 1.8, color="#555555")

    ax.set_ylabel(r"AUPRC $/$ chance floor", fontsize=FS - 0.2, labelpad=2.2)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1\n(chance)", "2", "3"])
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.tick_params(length=1.8, pad=1.6, labelsize=FS - 0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", lw=0.3, alpha=0.22, zorder=1)

    # row-major legend so its order matches the left-to-right order of the bars
    h, l = ax.get_legend_handles_labels()
    ncol = 3 if args.width == "single" else 6
    order = [j for c in range(ncol) for j in range(c, len(h), ncol)]
    h, l = [h[j] for j in order], [l[j] for j in order]
    ax.legend(h, l, fontsize=FS - 1.5, frameon=False, ncol=ncol,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.25, handleheight=0.85, handletextpad=0.42,
              columnspacing=1.0, labelspacing=0.28, borderpad=0.0)

    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.015)
    print("wrote", out)

    lo = min(m / f for f, d in SUBSETS.values() for m, _ in d.values()
             if f in (0.065, 0.033, 0.122))
    print("hard-subset lift spans %.2f-1.69x; spoofing 3.17-3.33x" % lo)


if __name__ == "__main__":
    main()
