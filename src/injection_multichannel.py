"""
Multi-channel attack settings for IoMT-SecAlarmBench (ECG-only and joint ECG+PPG).

Extends injection_generator.py without modifying it, so the C1-PPG attacks are
reproduced exactly. Every attacked window keeps its clean counterfactual and its
true cause.

Attacker capability levels (paper Section III):

  C1  controls one device stream. ``C1_PPG`` is the base generator's output;
      ``C1_ECG`` applies the same 34 conditions to the ECG channel.
  C2  controls both channels but alters them independently
      (``C2_JOINT_INDEPENDENT``): independent onsets, signs and magnitudes per
      channel; replay draws a different source segment for each channel.
  C3  controls both channels and keeps them physiologically consistent
      (``C3_JOINT_COORDINATED``):
        * joint replay: the same earlier ECG+PPG stretch, copied into both channels
          at the same onset (same-patient and cross-patient);
        * coordinated FDI / spoofing: same morphology, level, sign and magnitude
          (each in its own channel's sigma units), with the PPG change delayed by
          the window's estimated pulse-transit lag;
        * rate warp: both channels time-scaled by the same factor, which fakes a
          heart-rate change while ECG heart rate and PPG pulse rate stay in step.

Seeds: each setting uses ``base.seed + SETTING_SEED_OFFSET[setting]``, so adding a
setting never changes the draws of another. ``C1_PPG`` uses offset 0.

Replay banks: ``joint_banks[pid]`` is an array [n, 2, L] whose two rows come from
the same source time (ECG, PPG). Build it with ``build_joint_banks`` from clean
windows that do not overlap the host windows, or with
``joint_banks_from_channel_banks`` when the per-channel banks were built from the
same source windows in the same order.

After generation, run the feature extractor on ``aw.attacked`` and keep
``aw.clean`` as the counterfactual.

Self-test on synthetic signals:  python src/injection_multichannel.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from injection_generator import (
    FDI_MORPHOLOGIES,
    MECHANISM_PRESERVATION,
    SEVERITY_SCHEDULE,
    AttackedWindow,
    GeneratorConfig,
    Window,
    _bound,
    _onset,
    generate_benchmark,
    inject_fdi,
    inject_replay,
    preonset_stats,
)

SETTINGS = ("C1_ECG", "C2_JOINT_INDEPENDENT", "C3_JOINT_COORDINATED")
ALL_SETTINGS = ("C1_PPG",) + SETTINGS
ATTACKER_LEVEL = {"C1_PPG": "C1", "C1_ECG": "C1",
                  "C2_JOINT_INDEPENDENT": "C2", "C3_JOINT_COORDINATED": "C3"}
SETTING_SEED_OFFSET = {"C1_PPG": 0, "C1_ECG": 101, "C2_JOINT_INDEPENDENT": 202,
                       "C3_JOINT_COORDINATED": 303}

# Rate warp (C3): time-scaling factor per ordinal level; direction (faster or
# slower) is drawn per window. 1.05-1.35 corresponds to roughly a 5-35% change in
# heart rate over the warped segment.
RATE_WARP_SCHEDULE = {"factor": [1.05, 1.10, 1.20, 1.35], "dur_s": 4.0}
REPLAY_DUR_S = 2.0          # same as inject_replay in the base generator


@dataclass
class MultiChannelConfig:
    base: GeneratorConfig = field(default_factory=GeneratorConfig)
    pair: tuple = ("ECG", "PPG")                 # (leading channel, lagging channel)
    ptt_search_s: tuple = (0.05, 0.50)           # pulse-transit lag search range
    ptt_default_s: float = 0.25                  # fallback when the estimate is weak
    ptt_min_corr: float = 0.10
    include_rate_warp: bool = True
    include_cross_patient_joint_replay: bool = True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def estimate_ptt_lag(window: Window, mc: MultiChannelConfig) -> tuple[int, bool]:
    """Lag (samples) by which the PPG pulse follows the ECG beat in this window.

    Correlates ECG slope energy with the positive PPG upstroke over the search
    range. Returns (lag_samples, estimated) where estimated=False means the
    default lag was used because the correlation was too weak.
    """
    ce, cp = (window.channels.index(ch) for ch in mc.pair)
    fs = window.fs
    e = np.abs(np.diff(window.data[ce]))
    p = np.clip(np.diff(window.data[cp]), 0, None)
    e = (e - e.mean()) / (e.std() + 1e-8)
    p = (p - p.mean()) / (p.std() + 1e-8)
    lo = max(1, int(round(mc.ptt_search_s[0] * fs)))
    hi = int(round(mc.ptt_search_s[1] * fs))
    best_lag, best_r = int(round(mc.ptt_default_s * fs)), -np.inf
    for lag in range(lo, hi + 1):
        a, b = e[:-lag], p[lag:]
        r = float(np.dot(a, b) / len(a))
        if r > best_r:
            best_lag, best_r = lag, r
    if best_r < mc.ptt_min_corr:
        return int(round(mc.ptt_default_s * fs)), False
    return best_lag, True


def build_joint_banks(windows_by_patient: Dict[str, Sequence[Window]],
                      pair: tuple = ("ECG", "PPG"),
                      exclude_window_ids: Iterable[str] = ()) -> Dict[str, np.ndarray]:
    """Joint replay banks from clean windows: pid -> array [n, 2, T].

    Pass only clean windows that do NOT overlap any host window (windows are 10 s
    with a 5 s stride, so neighbours overlap). Host ids in ``exclude_window_ids``
    are dropped as a safeguard.
    """
    excl = set(exclude_window_ids)
    banks = {}
    for pid, ws in windows_by_patient.items():
        rows = []
        for w in ws:
            if w.window_id in excl or not all(ch in w.channels for ch in pair):
                continue
            idx = [w.channels.index(ch) for ch in pair]
            rows.append(w.data[idx])
        if rows:
            banks[pid] = np.stack(rows, axis=0)
    return banks


def joint_banks_from_channel_banks(replay_banks: dict, pair: tuple = ("ECG", "PPG")):
    """Stack existing per-channel banks into joint banks.

    VALID ONLY IF row i of the ECG bank and row i of the PPG bank come from the
    same source window (check the bank builder). Shapes must match.
    """
    out = {}
    for pid in {p for p, _ in replay_banks}:
        a, b = replay_banks.get((pid, pair[0])), replay_banks.get((pid, pair[1]))
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            raise ValueError(f"bank shapes differ for {pid}: {a.shape} vs {b.shape}")
        out[pid] = np.stack([a, b], axis=1)
    return out


def _tag(meta: dict, setting: str, w: Window, **extra) -> dict:
    meta = dict(meta)
    meta.update(setting=setting, attacker_level=ATTACKER_LEVEL[setting],
                window_id=w.window_id, patient_id=w.patient_id, **extra)
    meta["condition_id"] = "|".join(str(meta.get(k)) for k in
                                    ("setting", "family", "morphology", "level",
                                     "replay_source"))
    return meta


# --------------------------------------------------------------------------- #
# C2: both channels, altered independently
# --------------------------------------------------------------------------- #
def inject_joint_independent_fdi(window, morph, level, mc, rng, spoof=False):
    ce, cp = mc.pair
    aw1 = inject_fdi(window, ce, morph, level, mc.base, rng, spoof=spoof)
    w2 = Window(aw1.attacked, window.channels, window.patient_id, window.fs,
                window.window_id)
    aw2 = inject_fdi(w2, cp, morph, level, mc.base, rng, spoof=spoof)
    m1, m2 = aw1.meta, aw2.meta
    meta = dict(family=m1["family"], morphology=morph, target_channel=f"{ce}+{cp}",
                level=int(level), coordination="independent",
                realised_mag_sigma=m1["realised_mag_sigma"],
                realised_mag_sigma_2=m2["realised_mag_sigma"],
                onset=m1["onset"], onset_2=m2["onset"],
                duration=m1["duration"], duration_2=m2["duration"],
                within_plausibility=m1["within_plausibility"],
                mechanism_preservation=m1["mechanism_preservation"])
    return AttackedWindow(clean=window.data.copy(), attacked=aw2.attacked, meta=meta)


def inject_joint_independent_replay(window, bank_a, bank_b, mc, rng, source):
    """Each channel replayed from a DIFFERENT, independently drawn source segment."""
    ce, cp = mc.pair
    aw1 = inject_replay(window, ce, bank_a, mc.base, rng, source)
    w2 = Window(aw1.attacked, window.channels, window.patient_id, window.fs,
                window.window_id)
    aw2 = inject_replay(w2, cp, bank_b, mc.base, rng, source)
    meta = dict(family="replay", morphology="segment_sub", target_channel=f"{ce}+{cp}",
                level=0, coordination="independent", realised_mag_sigma=float("nan"),
                onset=aw1.meta["onset"], onset_2=aw2.meta["onset"],
                duration=aw1.meta["duration"], duration_2=aw2.meta["duration"],
                replay_source=source, within_plausibility=True,
                mechanism_preservation=MECHANISM_PRESERVATION["replay"])
    return AttackedWindow(clean=window.data.copy(), attacked=aw2.attacked, meta=meta)


# --------------------------------------------------------------------------- #
# C3: both channels, physiologically consistent
# --------------------------------------------------------------------------- #
def inject_joint_replay(window, joint_bank, mc, rng, source="same_patient"):
    """Copy the SAME earlier ECG+PPG stretch into both channels at the same onset."""
    cfg = mc.base
    x = window.data.copy()
    idx = [window.channels.index(ch) for ch in mc.pair]
    T = x.shape[1]
    pre = max(1, int(round(cfg.preonset_frac * T)))
    dur = max(1, min(int(round(REPLAY_DUR_S * window.fs)), T - pre - 1))
    t0 = _onset(T, dur, cfg, rng)
    src_i = int(rng.integers(0, joint_bank.shape[0]))
    L = joint_bank.shape[2]
    off = int(rng.integers(0, max(1, L - dur + 1)))
    for k, c in enumerate(idx):
        x[c, t0:t0 + dur] = joint_bank[src_i, k, off:off + dur]
    meta = dict(family="replay", morphology="segment_sub_joint",
                target_channel="+".join(mc.pair), level=0, coordination="coordinated",
                realised_mag_sigma=float("nan"), onset=int(t0), onset_2=int(t0),
                duration=int(dur), duration_2=int(dur), replay_source=source,
                replay_bank_index=src_i, replay_offset=off,
                within_plausibility=True,
                mechanism_preservation=MECHANISM_PRESERVATION["replay"])
    return AttackedWindow(clean=window.data.copy(), attacked=x, meta=meta)


def inject_joint_coordinated_fdi(window, morph, level, mc, rng, spoof=False):
    """Same morphology, level, sign and sigma-magnitude on both channels; the PPG
    change starts one pulse-transit lag after the ECG change."""
    cfg = mc.base
    x = window.data.copy()
    mu, sd = preonset_stats(window, cfg)
    idx = [window.channels.index(ch) for ch in mc.pair]
    T = x.shape[1]
    fs = window.fs
    lag, lag_est = estimate_ptt_lag(window, mc)

    sched = SEVERITY_SCHEDULE[morph]
    i = level - 1
    pre = max(1, int(round(cfg.preonset_frac * T)))
    dur = max(1, int(round(sched["dur_s"][i] * fs)))
    dur = max(1, min(dur, T - pre - lag - 1))
    hi = T - dur - lag
    t0 = int(rng.integers(pre, hi)) if hi > pre else pre
    sign = 1.0 if rng.random() < 0.5 else -1.0

    kind = sched["kind"]
    if kind in ("point", "ramp"):
        u = float(sched["mag"][i])
    elif kind == "u_hi":
        u = float(rng.uniform(0, sched["hi"][i]))
    elif kind == "u_lohi":
        u = float(rng.uniform(sched["lo"][i], sched["hi"][i]))
    else:
        raise ValueError(morph)

    onsets = []
    for k, c in enumerate(idx):
        s = t0 + (lag if k == 1 else 0)
        seg = slice(s, s + dur)
        onsets.append(s)
        if kind == "point":
            x[c, seg] += sign * u * sd[c]
        elif kind == "u_hi":
            x[c, seg] = x[c, s] + sign * u * sd[c]
        elif kind == "ramp":
            r = np.arange(dur) / max(1, dur - 1)
            x[c, seg] += r * sign * u * sd[c]
        elif kind == "u_lohi":
            x[c, seg] += sign * u * sd[c]

        lo_b, hi_b = _bound(cfg, window.channels[c], mu[c], sd[c])
        if not spoof:
            x[c, seg] = np.clip(x[c, seg], lo_b, hi_b)
        else:
            edge = hi_b if sign > 0 else lo_b
            x[c, seg] = edge + sign * max(1.0, level) * sd[c]

    fam = "spoofing" if spoof else "fdi"
    meta = dict(family=fam, morphology=morph, target_channel="+".join(mc.pair),
                level=int(level), coordination="coordinated",
                realised_mag_sigma=round(u, 3), onset=int(onsets[0]),
                onset_2=int(onsets[1]), duration=int(dur), duration_2=int(dur),
                ptt_lag_s=round(lag / fs, 3), ptt_lag_estimated=bool(lag_est),
                within_plausibility=not spoof,
                mechanism_preservation=MECHANISM_PRESERVATION[fam])
    return AttackedWindow(clean=window.data.copy(), attacked=x, meta=meta)


def inject_rate_warp(window, level, mc, rng):
    """Time-scale the same segment of both channels by one factor (faster or
    slower), so ECG heart rate and PPG pulse rate change together."""
    cfg = mc.base
    x = window.data.copy()
    mu, sd = preonset_stats(window, cfg)
    idx = [window.channels.index(ch) for ch in mc.pair]
    T = x.shape[1]
    fs = window.fs
    r = RATE_WARP_SCHEDULE["factor"][level - 1]
    faster = bool(rng.random() < 0.5)
    factor = r if faster else 1.0 / r
    pre = max(1, int(round(cfg.preonset_frac * T)))
    dur = int(round(RATE_WARP_SCHEDULE["dur_s"] * fs))
    span = int(np.ceil(dur * max(factor, 1.0)))
    if pre + span >= T:
        span = T - pre - 1
        dur = int(span / max(factor, 1.0))
    hi = T - span
    t0 = int(rng.integers(pre, hi)) if hi > pre else pre
    src = t0 + np.arange(dur) * factor
    grid = np.arange(T)
    within = True
    for c in idx:
        x[c, t0:t0 + dur] = np.interp(src, grid, window.data[c])
        lo_b, hi_b = _bound(cfg, window.channels[c], mu[c], sd[c])
        clipped = np.clip(x[c, t0:t0 + dur], lo_b, hi_b)
        within &= bool(np.allclose(clipped, x[c, t0:t0 + dur]))
        x[c, t0:t0 + dur] = clipped
    meta = dict(family="fdi", morphology="rate_warp", target_channel="+".join(mc.pair),
                level=int(level), coordination="coordinated",
                realised_mag_sigma=float("nan"), warp_factor=round(factor, 4),
                onset=int(t0), onset_2=int(t0), duration=int(dur), duration_2=int(dur),
                within_plausibility=True, clipped_by_bound=not within,
                mechanism_preservation=MECHANISM_PRESERVATION["fdi"])
    return AttackedWindow(clean=window.data.copy(), attacked=x, meta=meta)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def generate_multichannel(windows: Sequence[Window],
                          settings: Sequence[str] = SETTINGS,
                          mc: Optional[MultiChannelConfig] = None,
                          replay_banks: Optional[dict] = None,
                          joint_banks: Optional[dict] = None,
                          include=("replay", "fdi", "spoofing")):
    """Generate the new settings → (list[AttackedWindow], manifest DataFrame).

    windows       clean host windows (the same 1,173 hosts as the PPG-only release)
    replay_banks  {(pid, channel): array [n, L]} as in the base generator
                  (needed for C1_ECG and C2 replay)
    joint_banks   {pid: array [n, 2, L]} aligned ECG+PPG banks (needed for C3 replay)
    """
    mc = mc or MultiChannelConfig()
    out, rows = [], []
    ce, cp = mc.pair

    for setting in settings:
        if setting not in SETTING_SEED_OFFSET:
            raise ValueError(f"unknown setting {setting}")
        seed = mc.base.seed + SETTING_SEED_OFFSET[setting]

        if setting in ("C1_PPG", "C1_ECG"):
            target = cp if setting == "C1_PPG" else ce
            cfg = replace(mc.base, seed=seed)
            aws, _ = generate_benchmark(windows, [target], cfg=cfg,
                                        replay_banks=replay_banks, include=include)
            by_id = {w.window_id: w for w in windows}
            for aw in aws:
                w = by_id[aw.meta["window_id"]]
                aw.meta = _tag(aw.meta, setting, w, coordination="single")
                out.append(aw); rows.append(aw.meta)
            continue

        rng = np.random.default_rng(seed)
        pool = {}
        if joint_banks:
            pool = {pid: b for pid, b in joint_banks.items() if b.shape[0] > 0}

        for w in windows:
            if not all(ch in w.channels for ch in mc.pair):
                continue

            if setting == "C2_JOINT_INDEPENDENT":
                if "replay" in include and replay_banks is not None:
                    a = replay_banks.get((w.patient_id, ce))
                    b = replay_banks.get((w.patient_id, cp))
                    if a is not None and b is not None and a.shape[0] and b.shape[0]:
                        aw = inject_joint_independent_replay(w, a, b, mc, rng, "same_patient")
                        aw.meta = _tag(aw.meta, setting, w)
                        out.append(aw); rows.append(aw.meta)
                for level in mc.base.levels:
                    for morph in mc.base.fdi_morphologies:
                        for fam, spoof in (("fdi", False), ("spoofing", True)):
                            if fam in include:
                                aw = inject_joint_independent_fdi(w, morph, level, mc, rng, spoof)
                                aw.meta = _tag(aw.meta, setting, w)
                                out.append(aw); rows.append(aw.meta)

            elif setting == "C3_JOINT_COORDINATED":
                if "replay" in include and pool:
                    same = pool.get(w.patient_id)
                    if same is not None:
                        aw = inject_joint_replay(w, same, mc, rng, "same_patient")
                        aw.meta = _tag(aw.meta, setting, w)
                        out.append(aw); rows.append(aw.meta)
                    if mc.include_cross_patient_joint_replay:
                        others = [pid for pid in pool if pid != w.patient_id]
                        if others:
                            pid_o = others[int(rng.integers(0, len(others)))]
                            aw = inject_joint_replay(w, pool[pid_o], mc, rng, "cross_patient")
                            aw.meta = _tag(aw.meta, setting, w, replay_source_patient=pid_o)
                            out.append(aw); rows.append(aw.meta)
                for level in mc.base.levels:
                    for morph in mc.base.fdi_morphologies:
                        for fam, spoof in (("fdi", False), ("spoofing", True)):
                            if fam in include:
                                aw = inject_joint_coordinated_fdi(w, morph, level, mc, rng, spoof)
                                aw.meta = _tag(aw.meta, setting, w)
                                out.append(aw); rows.append(aw.meta)
                    if mc.include_rate_warp and "fdi" in include:
                        aw = inject_rate_warp(w, level, mc, rng)
                        aw.meta = _tag(aw.meta, setting, w)
                        out.append(aw); rows.append(aw.meta)

    return out, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Self-test on synthetic ECG+PPG (python src/injection_multichannel.py)
# --------------------------------------------------------------------------- #
def _synthetic_window(pid, wid, hr_bpm, fs=125.0, T=1250, ptt_s=0.25, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fs
    period = 60.0 / hr_bpm
    beats = np.arange(0.1, T / fs, period)
    ecg = np.zeros(T)
    ppg = np.zeros(T)
    for b in beats:
        ecg += 1.2 * np.exp(-0.5 * ((t - b) / 0.012) ** 2)            # R wave
        ecg += 0.25 * np.exp(-0.5 * ((t - b - 0.25) / 0.04) ** 2)    # T wave
        tp = t - b - ptt_s
        ppg += np.where(tp > 0, (tp / 0.12) * np.exp(1 - tp / 0.12), 0.0)
    ecg += 0.03 * rng.standard_normal(T)
    ppg += 0.02 * rng.standard_normal(T)
    z = lambda v: (v - v.mean()) / v.std()
    return Window(np.vstack([z(ecg), z(ppg)]), ["ECG", "PPG"], pid, fs, wid)


def _rate_bpm(sig, fs):
    """Dominant beat rate from the autocorrelation peak in 40-200 bpm."""
    s = sig - sig.mean()
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    lo, hi = int(fs * 60 / 200), int(fs * 60 / 40)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return 60.0 * fs / lag


def _self_test():
    hosts = [_synthetic_window("p1", "p1_w3", 72, seed=1),
             _synthetic_window("p1", "p1_w9", 75, seed=2),
             _synthetic_window("p2", "p2_w4", 88, seed=3)]
    bank_src = {"p1": [_synthetic_window("p1", f"p1_b{i}", 70 + i, seed=10 + i) for i in range(3)],
                "p2": [_synthetic_window("p2", f"p2_b{i}", 85 + i, seed=20 + i) for i in range(3)]}
    joint = build_joint_banks(bank_src, exclude_window_ids=[h.window_id for h in hosts])
    per_ch = {}
    for pid, ws in bank_src.items():
        for k, ch in enumerate(("ECG", "PPG")):
            per_ch[(pid, ch)] = np.stack([w.data[k] for w in ws])

    aws, man = generate_multichannel(hosts, replay_banks=per_ch, joint_banks=joint)
    by_id = {h.window_id: h for h in hosts}
    fs = 125.0
    mc = MultiChannelConfig()
    failures = []

    for aw in aws:
        m = aw.meta
        h = by_id[m["window_id"]]
        if not np.array_equal(aw.clean, h.data):
            failures.append(("clean changed", m["condition_id"]))
        changed = [not np.allclose(aw.attacked[k], h.data[k]) for k in range(2)]
        if m["setting"] == "C1_ECG":
            if not changed[0] or changed[1]:
                failures.append(("C1_ECG channel scope", m["condition_id"], changed))
        else:
            if not all(changed):
                failures.append(("joint should change both", m["condition_id"], changed))
        if m["family"] == "fdi" and m["morphology"] != "rate_warp":
            mu, sd = preonset_stats(h, mc.base)
            for k, ch in enumerate(("ECG", "PPG")):
                lo, hi = _bound(mc.base, ch, mu[k], sd[k])
                if aw.attacked[k].max() > max(hi, h.data[k].max()) + 1e-9 or \
                   aw.attacked[k].min() < min(lo, h.data[k].min()) - 1e-9:
                    if changed[k]:
                        seg = aw.attacked[k] != h.data[k]
                        if (aw.attacked[k][seg] > hi + 1e-9).any() or (aw.attacked[k][seg] < lo - 1e-9).any():
                            failures.append(("FDI out of bound", m["condition_id"], ch))
        if m["family"] == "spoofing":
            mu, sd = preonset_stats(h, mc.base)
            for k, ch in enumerate(("ECG", "PPG")):
                if not changed[k]:
                    continue
                lo, hi = _bound(mc.base, ch, mu[k], sd[k])
                seg = aw.attacked[k] != h.data[k]
                v = aw.attacked[k][seg]
                if ((v <= hi) & (v >= lo)).any():
                    failures.append(("spoof inside bound", m["condition_id"], ch))
        if m.get("coordination") == "coordinated" and m["family"] in ("fdi", "spoofing") \
                and m["morphology"] != "rate_warp":
            lag = int(round(m["ptt_lag_s"] * fs))
            if m["onset_2"] - m["onset"] != lag:
                failures.append(("coordinated lag", m["condition_id"]))
        if m["morphology"] == "segment_sub_joint":
            src_pid = m.get("replay_source_patient", h.patient_id)
            b = joint[src_pid][m["replay_bank_index"]]
            s, d, off = m["onset"], m["duration"], m["replay_offset"]
            for k in range(2):
                if not np.array_equal(aw.attacked[k, s:s + d], b[k, off:off + d]):
                    failures.append(("joint replay alignment", m["condition_id"]))
        if m["morphology"] == "rate_warp":
            s, d = m["onset"], m["duration"]
            r_e = _rate_bpm(aw.attacked[0, s:s + d], fs)
            r_p = _rate_bpm(aw.attacked[1, s:s + d], fs)
            r_e0 = _rate_bpm(h.data[0, s:s + d], fs)
            if abs(r_e - r_p) / r_e > 0.06:
                failures.append(("warp ECG/PPG rate mismatch", m["condition_id"], r_e, r_p))
            ratio = r_e / r_e0
            if abs(ratio - m["warp_factor"]) / m["warp_factor"] > 0.08:
                failures.append(("warp ratio", m["condition_id"], ratio, m["warp_factor"]))

    lag, est = estimate_ptt_lag(hosts[0], mc)
    print(f"estimated pulse-transit lag on synthetic data: {lag / fs:.3f} s (true 0.250 s, estimated={est})")
    print(f"generated {len(aws)} attacked windows from {len(hosts)} hosts")
    print(man.groupby(["setting", "family"]).size().rename("windows").to_string())
    print("per-host counts:", man.groupby("setting").size().div(len(hosts)).to_dict())
    if failures:
        print(f"\n{len(failures)} CHECK FAILURES")
        for f in failures[:20]:
            print("  ", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _self_test()
