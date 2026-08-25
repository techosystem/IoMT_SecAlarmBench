"""
IoMT-SecAlarmBench — Attack Injection Generator (v2, patched)
=============================================================

Manufactures the integrity-attack windows (FDI / replay / value-spoofing) that no
public IoMT dataset carries natively, by injecting into clean, preprocessed
CinC-2015 / MIMIC-III waveform windows. Every attacked window ships with its clean
counterfactual and its true cause — the counterfactual ground truth.

v2 changes (paper §4.3 alignment):
  * FACTORIAL design: {replay} ∪ ({spike,stuck_at,drift,bias} × {FDI, spoofing}).
  * PER-MORPHOLOGY severity schedule (paper Table 4) — a shared grid is NOT comparable
    across morphologies, so each morphology has its own L1..L4 magnitude + duration.
    Levels are ORDINAL (1..4); the realised magnitude is recorded per window so RQ4
    curves are plotted PER MORPHOLOGY, never on a shared axis.
  * PLAUSIBILITY bounds are PER CHANNEL and DECOUPLED from the severity grid, so the
    top severity level no longer collides with the clip (old bug). Bounds are applied
    in the loader's within-patient normalized (z-scored) space; the per-channel
    sigma-multiples below are the normalized-space calibration of the physical ranges
    in paper Table 5. The loader stores zscore_mean/std per channel, so physical-unit
    bounds can be recovered by de-normalization if required.
  * CROSS-PATIENT replay in addition to same-patient replay (paper §4.3.3).

Durations are in SECONDS (waveform-appropriate at 125 Hz), adapting DSTAN-Med's
step-based durations. (Paper Table 4's "d (steps)" column should be restated in
seconds/window-fraction for the waveform setting — flagged for the revision.)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import pandas as pd

MECHANISMS = ("replay", "fdi", "spoofing")
FDI_MORPHOLOGIES = ("spike", "stuck_at", "drift", "bias")
LEVELS = (1, 2, 3, 4)                       # ordinal severity L1..L4
MECHANISM_PRESERVATION = {"replay": 1.0, "fdi": 0.5, "spoofing": 0.0}

# Per-morphology severity schedule (paper Table 4). Magnitudes in per-patient-sigma
# units; durations in seconds. Index by level-1.
SEVERITY_SCHEDULE = {
    "spike":    {"kind": "point",  "mag": [0.5, 1.0, 2.0, 4.0],
                 "dur_s": [0.024, 0.024, 0.040, 0.040]},
    "stuck_at": {"kind": "u_hi",   "hi":  [1.0, 3.0, 5.0, 10.0],
                 "dur_s": [1.0, 2.0, 4.0, 4.0]},
    "drift":    {"kind": "ramp",   "mag": [1.0, 2.0, 2.0, 4.0],
                 "dur_s": [2.0, 2.0, 4.0, 4.0]},
    "bias":     {"kind": "u_lohi", "lo":  [0.5, 1.0, 2.0, 4.0], "hi": [1.0, 2.0, 4.0, 8.0],
                 "dur_s": [2.0, 4.0, 4.0, 4.0]},
}

# Per-channel plausibility bound in normalized-sigma units (calibration of Table 5).
PLAUSIBILITY_SIGMA = {"ECG": 5.0, "PPG": 4.0, "ABP": 4.0, "RESP": 4.0}
DEFAULT_PLAUSIBILITY = 4.0


@dataclass
class GeneratorConfig:
    levels: Sequence[int] = LEVELS
    fdi_morphologies: Sequence[str] = FDI_MORPHOLOGIES
    preonset_frac: float = 0.25
    plausibility_sigma: dict = field(default_factory=lambda: dict(PLAUSIBILITY_SIGMA))
    default_plausibility: float = DEFAULT_PLAUSIBILITY
    cross_patient_replay: bool = True
    seed: int = 20260723


@dataclass
class Window:
    data: np.ndarray            # [C, T]
    channels: list
    patient_id: str
    fs: float = 125.0
    window_id: str = ""

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        assert self.data.ndim == 2 and self.data.shape[0] == len(self.channels)


@dataclass
class AttackedWindow:
    clean: np.ndarray
    attacked: np.ndarray
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def preonset_stats(window: Window, cfg: GeneratorConfig):
    T = window.data.shape[1]
    k = max(1, int(round(cfg.preonset_frac * T)))
    seg = window.data[:, :k]
    mu = seg.mean(axis=1)
    sd = seg.std(axis=1)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _onset(T, dur, cfg, rng):
    lo = max(1, int(round(cfg.preonset_frac * T)))
    hi = max(lo + 1, T - dur)
    return int(rng.integers(lo, hi)) if hi > lo else lo


def _apply_morphology(x, c, morph, sd_c, level, cfg, fs, rng):
    """Apply one morphology at its per-morphology scheduled magnitude+duration.
    Returns (t0, dur, realised_peak_abs_change)."""
    sched = SEVERITY_SCHEDULE[morph]
    i = level - 1
    T = x.shape[1]
    dur = max(1, int(round(sched["dur_s"][i] * fs)))
    dur = min(dur, T - max(1, int(round(cfg.preonset_frac * T))) - 1)
    dur = max(1, dur)
    t0 = _onset(T, dur, cfg, rng)
    sign = 1.0 if rng.random() < 0.5 else -1.0

    if sched["kind"] == "point":                         # spike
        mag = sched["mag"][i] * sd_c
        x[c, t0:t0 + dur] += sign * mag
        peak = mag
    elif sched["kind"] == "u_hi":                        # stuck_at (frozen constant)
        off = rng.uniform(0, sched["hi"][i]) * sd_c
        v = x[c, t0] + sign * off
        peak = abs(v - float(np.mean(x[c, t0:t0 + dur])))
        x[c, t0:t0 + dur] = v
    elif sched["kind"] == "ramp":                        # drift
        mag = sched["mag"][i] * sd_c
        k = np.arange(dur)
        x[c, t0:t0 + dur] += (k / max(1, dur - 1)) * sign * mag
        peak = mag
    elif sched["kind"] == "u_lohi":                      # bias
        beta = sign * rng.uniform(sched["lo"][i], sched["hi"][i]) * sd_c
        x[c, t0:t0 + dur] += beta
        peak = abs(beta)
    else:
        raise ValueError(morph)
    return t0, dur, float(peak)


def _bound(cfg, channel, mu_c, sd_c):
    k = cfg.plausibility_sigma.get(channel, cfg.default_plausibility)
    return mu_c - k * sd_c, mu_c + k * sd_c


def inject_fdi(window, target, morph, level, cfg, rng, spoof=False):
    x = window.data.copy()
    mu, sd = preonset_stats(window, cfg)
    c = window.channels.index(target)
    t0, dur, peak = _apply_morphology(x, c, morph, sd[c], level, cfg, window.fs, rng)
    lo, hi = _bound(cfg, target, mu[c], sd[c])
    seg = slice(t0, t0 + dur)

    if not spoof:                                        # FDI: keep in-bound (plausible)
        x[c, seg] = np.clip(x[c, seg], lo, hi)
        within = True
    else:                                                # spoofing: force out-of-bound
        direction = np.sign(np.mean(x[c, seg]) - mu[c]) or 1.0
        edge = hi if direction > 0 else lo
        x[c, seg] = edge + direction * max(1.0, level) * sd[c]
        within = False

    fam = "spoofing" if spoof else "fdi"
    meta = dict(family=fam, morphology=morph, target_channel=target,
                level=int(level), realised_mag_sigma=round(peak / sd[c], 3),
                onset=int(t0), duration=int(dur), within_plausibility=within,
                mechanism_preservation=MECHANISM_PRESERVATION[fam])
    return AttackedWindow(clean=window.data.copy(), attacked=x, meta=meta)


def inject_replay(window, target, bank, cfg, rng, source="same_patient"):
    x = window.data.copy()
    c = window.channels.index(target)
    T = window.data.shape[1]
    dur = min(int(round(2.0 * window.fs)),
              T - max(1, int(round(cfg.preonset_frac * T))) - 1)
    dur = max(1, dur)
    t0 = _onset(T, dur, cfg, rng)
    src_i = int(rng.integers(0, bank.shape[0]))
    src = bank[src_i]
    off = int(rng.integers(0, max(1, src.shape[0] - dur + 1)))
    x[c, t0:t0 + dur] = src[off:off + dur]

    meta = dict(family="replay", morphology="segment_sub", target_channel=target,
                level=0, realised_mag_sigma=float("nan"), onset=int(t0), duration=int(dur),
                replay_source=source, within_plausibility=True,
                mechanism_preservation=MECHANISM_PRESERVATION["replay"])
    return AttackedWindow(clean=window.data.copy(), attacked=x, meta=meta)


# --------------------------------------------------------------------------- #
def generate_benchmark(windows, target_channels, cfg=None, replay_banks=None,
                       include=("replay", "fdi", "spoofing")):
    """Sweep the factorial grid → (list[AttackedWindow], manifest DataFrame)."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    out, rows = [], []

    banks_by_channel = {}
    if replay_banks:
        for (pid, ch), arr in replay_banks.items():
            banks_by_channel.setdefault(ch, []).append((pid, arr))

    for w in windows:
        tchs = target_channels[w.patient_id] if isinstance(target_channels, dict) else target_channels
        for target in tchs:
            if target not in w.channels:
                continue

            if "replay" in include and replay_banks is not None:
                same = replay_banks.get((w.patient_id, target))
                if same is not None and same.shape[0] > 0:
                    aw = inject_replay(w, target, same, cfg, rng, "same_patient")
                    aw.meta.update(window_id=w.window_id, patient_id=w.patient_id)
                    out.append(aw); rows.append(aw.meta)
                if cfg.cross_patient_replay:
                    others = [(pid, a) for pid, a in banks_by_channel.get(target, [])
                              if pid != w.patient_id and a.shape[0] > 0]
                    if others:
                        pid_o, arr_o = others[int(rng.integers(0, len(others)))]
                        aw = inject_replay(w, target, arr_o, cfg, rng, "cross_patient")
                        aw.meta.update(window_id=w.window_id, patient_id=w.patient_id,
                                       replay_source_patient=pid_o)
                        out.append(aw); rows.append(aw.meta)

            for level in cfg.levels:
                for morph in cfg.fdi_morphologies:
                    if "fdi" in include:
                        aw = inject_fdi(w, target, morph, level, cfg, rng, spoof=False)
                        aw.meta.update(window_id=w.window_id, patient_id=w.patient_id)
                        out.append(aw); rows.append(aw.meta)
                    if "spoofing" in include:
                        aw = inject_fdi(w, target, morph, level, cfg, rng, spoof=True)
                        aw.meta.update(window_id=w.window_id, patient_id=w.patient_id)
                        out.append(aw); rows.append(aw.meta)

    return out, pd.DataFrame(rows)
