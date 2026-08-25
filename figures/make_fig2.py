#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIGURE 2 -- Attack morphology gallery along the mechanism-preservation axis.

Emits fig2_morphologies.pdf, a full-width (figure*) strip of six panels.
Each panel shows the unattacked ECG channel (grey, top trace) above the
injected PPG channel (red, bottom trace) overlaid on the RETAINED
COUNTERFACTUAL clean signal (dashed grey) with the per-window plausibility
band [l_c, u_c] shaded.  That overlay is the paper's headline contribution
made visible: for every attacked window the clean signal is kept.

USAGE
  Synthetic illustration (runs anywhere, no data needed):
      python make_fig2.py

  Real benchmark windows (STRONGLY PREFERRED for the camera-ready) -- point
  it at the arrays your generator already writes:
      python make_fig2.py --npz /path/to/benchmark_windows.npz

  The .npz is expected to contain, at minimum:
      X_clean   [N, 2, T]   retained counterfactual clean windows (ch0=ECG, ch1=PPG)
      X_inj     [N, 2, T]   injected windows
      morph     [N]         morphology string per window
      regime    [N]         'in-bound' | 'out-of-bound' | 'replay'
      level     [N]         L1..L4
  Adjust PICK below to choose which window index illustrates each panel.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

FS = 125           # Hz, benchmark common base
WIN = 10           # s, window length
T = FS * WIN

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 6.2,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "pdf.fonttype": 42,
})

CLEAN = "#9a9a9a"
INJ = "#c0392b"
BAND = "#4a7ebb"


# ----------------------------------------------------------------- synthetic
def synth_ecg(t, hr=72.0, rng=None):
    """Crude but recognisable ECG: P-QRS-T as summed Gaussians per beat."""
    rng = rng or np.random.default_rng(0)
    x = np.zeros_like(t)
    rr = 60.0 / hr
    for beat in np.arange(0, t[-1] + rr, rr):
        jit = rng.normal(0, 0.006)
        b = beat + jit
        x += 0.10 * np.exp(-((t - (b - 0.16)) ** 2) / (2 * 0.022 ** 2))   # P
        x += -0.14 * np.exp(-((t - (b - 0.020)) ** 2) / (2 * 0.0085 ** 2))  # Q
        x += 1.00 * np.exp(-((t - b) ** 2) / (2 * 0.0090 ** 2))            # R
        x += -0.22 * np.exp(-((t - (b + 0.026)) ** 2) / (2 * 0.0100 ** 2))  # S
        x += 0.24 * np.exp(-((t - (b + 0.19)) ** 2) / (2 * 0.036 ** 2))    # T
    return x + rng.normal(0, 0.012, t.size)


def synth_ppg(t, hr=72.0, rng=None):
    """Pulsatile PPG: systolic upstroke + dicrotic notch, lagged behind ECG."""
    rng = rng or np.random.default_rng(1)
    x = np.zeros_like(t)
    rr = 60.0 / hr
    lag = 0.22                                   # pulse transit time
    for beat in np.arange(0, t[-1] + rr, rr):
        b = beat + lag + rng.normal(0, 0.006)
        x += 1.00 * np.exp(-((t - b) ** 2) / (2 * 0.055 ** 2))            # systolic
        x += 0.34 * np.exp(-((t - (b + 0.20)) ** 2) / (2 * 0.060 ** 2))   # dicrotic
    x += 0.05 * np.sin(2 * np.pi * 0.25 * t)      # respiratory baseline sway
    return x + rng.normal(0, 0.010, t.size)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


# ------------------------------------------------- morphologies (Table I forms)
def inject_spike(x, onset, dur_s, m, sign=+1):
    y = x.copy()
    i0 = int(onset * FS)
    n = max(3, int(round(dur_s * FS)))            # 3-5 sample support
    y[i0:i0 + n] += sign * m
    return y, (i0, i0 + n)


def inject_stuck(x, onset, dur_s, v):
    y = x.copy()
    i0, i1 = int(onset * FS), int((onset + dur_s) * FS)
    y[i0:i1] = v
    return y, (i0, i1)


def inject_drift(x, onset, dur_s, dmax):
    y = x.copy()
    i0, i1 = int(onset * FS), int((onset + dur_s) * FS)
    k = np.arange(i1 - i0)
    y[i0:i1] += (k / max(1, (i1 - i0 - 1))) * dmax
    return y, (i0, i1)


def inject_bias(x, onset, dur_s, beta):
    y = x.copy()
    i0, i1 = int(onset * FS), int((onset + dur_s) * FS)
    y[i0:i1] += beta
    return y, (i0, i1)


def inject_replay(x, onset, dur_s, stale_s):
    """Mechanism-preserving: paste a real earlier segment of the same channel."""
    y = x.copy()
    i0, i1 = int(onset * FS), int((onset + dur_s) * FS)
    s0 = i0 - int(stale_s * FS)
    y[i0:i1] = x[s0:s0 + (i1 - i0)]
    return y, (i0, i1)


# ------------------------------------------------------------------- plotting
def panel(axE, axP, t, ecg, clean, inj, span, band, title, sub, show_ylabel):
    lo, hi = band

    # --- unattacked ECG channel (still reflects S_t) ---
    axE.plot(t, ecg, color=CLEAN, lw=0.45)
    axE.set_xticks([])
    axE.set_yticks([])
    for s in axE.spines.values():
        s.set_visible(False)
    axE.set_xlim(t[0], t[-1])
    axE.text(0.012, 0.72, "ECG", transform=axE.transAxes,
             fontsize=5.0, color="#666666", va="center")

    # --- plausibility band ---
    axP.axhspan(lo, hi, color=BAND, alpha=0.10, lw=0, zorder=0)
    axP.axhline(lo, color=BAND, lw=0.4, ls=(0, (3, 2)), alpha=0.65, zorder=1)
    axP.axhline(hi, color=BAND, lw=0.4, ls=(0, (3, 2)), alpha=0.65, zorder=1)

    # --- retained counterfactual clean signal ---
    if span is None:                       # clean host: no injection to contrast
        axP.plot(t, clean, color=CLEAN, lw=0.6, zorder=3)
    else:
        axP.plot(t, clean, color=CLEAN, lw=0.75, ls=(0, (2.4, 1.6)), zorder=2,
                 label="retained clean")
        axP.plot(t, inj, color=INJ, lw=0.55, zorder=3, label="injected")

    if span is not None:
        axP.axvspan(span[0] / FS, span[1] / FS, color=INJ, alpha=0.09, lw=0, zorder=0)

    axP.set_xlim(t[0], t[-1])
    axP.set_ylim(-5.0, 8.3)
    axP.set_xticks([0, 5, 10])
    axP.set_yticks([])
    axP.tick_params(axis="x", length=1.6, pad=1.2, labelsize=5.2)
    for sp in ("top", "right", "left"):
        axP.spines[sp].set_visible(False)
    axP.text(0.012, 0.90, "PPG", transform=axP.transAxes,
             fontsize=5.0, color="#666666", va="center")
    axP.set_xlabel("time (s)", fontsize=5.4, labelpad=0.8)

    if span is not None and (span[1] - span[0]) <= 8:      # transient spike
        axP.annotate("", xy=((span[0] + span[1]) / 2 / FS, 4.3),
                     xytext=((span[0] + span[1]) / 2 / FS, 7.3),
                     arrowprops=dict(arrowstyle="-|>", color=INJ, lw=0.6,
                                     shrinkA=0, shrinkB=0, mutation_scale=4))
    axE.set_title(title, fontsize=6.4, pad=2.2, fontweight="bold")
    axP.text(0.5, -0.34, sub, transform=axP.transAxes, ha="center",
             va="top", fontsize=5.6)
    if show_ylabel:
        axP.set_ylabel(r"amplitude ($\bar\sigma_c$)", fontsize=5.4, labelpad=1.5)
        axP.plot([], [], color=INJ, lw=0.55, label="injected")
        axP.plot([], [], color=CLEAN, lw=0.75, ls=(0, (2.4, 1.6)),
                 label="retained clean")
        axP.axhspan(0, 0, color=BAND, alpha=0.10, lw=0, label=r"$[l_c,u_c]$")
        axP.legend(loc="lower left", fontsize=4.4, frameon=False,
                   handlelength=1.5, handletextpad=0.4, borderpad=0.0,
                   labelspacing=0.18, bbox_to_anchor=(-0.02, -0.03))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None,
                    help="path to real benchmark windows (see docstring)")
    ap.add_argument("--out", default="fig2_morphologies.pdf")
    args = ap.parse_args()

    t = np.arange(T) / FS

    if args.npz:
        d = np.load(args.npz, allow_pickle=True)
        # PICK: choose one window index per panel from your own arrays.
        PICK = dict(replay=None, spike=None, stuck=None, drift=None, spoof=None)
        raise SystemExit(
            "Fill in PICK with window indices from %s, then remove this guard. "
            "Arrays found: %s" % (args.npz, list(d.keys())))

    rng = np.random.default_rng(7)
    ecg = zscore(synth_ecg(t, rng=np.random.default_rng(3)))
    ppg = zscore(synth_ppg(t, rng=np.random.default_rng(4)))

    # plausibility bound  [l_c,u_c] = mu +- k*sigma, k=4.0 on PPG (paper Sec IV-C3)
    pre = ppg[: T // 4]
    mu, sd = pre.mean(), pre.std()
    band = (mu - 4.0 * sd, mu + 4.0 * sd)

    def clip(y):
        return np.clip(y, band[0], band[1])

    # ---- the six panels, ordered along the mechanism-preservation axis ----
    p_clean, s_clean = ppg.copy(), None
    p_rep, s_rep = inject_replay(ppg, onset=5.0, dur_s=4.0, stale_s=3.1)
    p_spk, s_spk = inject_spike(ppg, onset=5.4, dur_s=0.040, m=2.0)
    p_stk, s_stk = inject_stuck(ppg, onset=4.6, dur_s=4.0, v=1.6)
    p_drf, s_drf = inject_drift(ppg, onset=3.6, dur_s=4.0, dmax=2.0)
    p_spf, s_spf = inject_bias(ppg, onset=4.4, dur_s=4.0, beta=5.0)   # out-of-bound

    # in-bound cells are clipped to the band; spoofing is not
    p_spk, p_stk, p_drf = clip(p_spk), clip(p_stk), clip(p_drf)

    panels = [
        (p_clean, s_clean, "Clean host", "no intervention"),
        (p_rep,   s_rep,   "Replay",     r"mechanism-preserving"),
        (p_spk,   s_spk,   "FDI spike",  r"in-bound, $2.0\,\bar\sigma_c$ (L3)"),
        (p_stk,   s_stk,   "FDI stuck-at", r"in-bound, frozen 4 s"),
        (p_drf,   s_drf,   "FDI drift",  r"in-bound ramp to $2.0\,\bar\sigma_c$"),
        (p_spf,   s_spf,   "Spoofing bias", r"out-of-bound offset"),
    ]

    fig = plt.figure(figsize=(7.16, 1.86))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 2.5],
                          hspace=0.05, wspace=0.13,
                          left=0.045, right=0.995, top=0.885, bottom=0.265)

    for j, (inj, span, title, sub) in enumerate(panels):
        axE = fig.add_subplot(gs[0, j])
        axP = fig.add_subplot(gs[1, j])
        panel(axE, axP, t, ecg, ppg, inj, span, band, title, sub, show_ylabel=(j == 0))

    fig.savefig(args.out, dpi=600, bbox_inches="tight", pad_inches=0.012)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
