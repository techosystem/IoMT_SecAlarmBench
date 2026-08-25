"""
IoMT-SecAlarmBench — CinC-2015 loader (Pipeline A preprocessing)
=================================================================

Raw PhysioNet/CinC-2015 records -> the record contract `cinc_adapter.build_windows`
consumes. Implements Pipeline A:
  1. read record; parse alarm labels from WFDB header comments
     rec.comments == ['Asystole', 'False alarm']  ->  ('Asystole', False)
  2. map raw channel names to roles (II/V/... -> ECG, PLETH -> PPG, ABP/ART -> ABP, RESP)
  3. resample 250 -> 125 Hz (anti-aliased polyphase)
  4. zero-phase Butterworth band-pass: ECG 0.5-40, PPG/ABP 0.5-8, RESP 0.05-2 Hz
  5. artifact/fault masking (flat-line, rail saturation, non-finite) -- LABELLED, not deleted
  6. within-record z-score per channel; the per-channel (mean,std) used are STORED
     (zscore_mean/std) so the injection plausibility bounds can be de-normalized to
     physical units if required.

Alarm at the 300 s mark (native 250 Hz; 75000-82500 samples = 300-330 s), so
alarm_onset_sample = 300 * TARGET_FS = 37500 at 125 Hz.
"""

from __future__ import annotations
import os, glob
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.ndimage import uniform_filter1d

CINC_FS_NATIVE = 250.0
TARGET_FS = 125.0
ALARM_TIME_S = 300.0

ECG_NAMES = {"II", "V", "I", "III", "MCL", "MCL1", "aVR", "aVL", "aVF", "AVR", "AVL", "AVF"}
PPG_NAMES = {"PLETH"}
ABP_NAMES = {"ABP", "ART"}
RESP_NAMES = {"RESP"}
ROLE_BANDS = {"ECG": (0.5, 40.0), "PPG": (0.5, 8.0), "ABP": (0.5, 8.0), "RESP": (0.05, 2.0)}

ALARM_TYPE_CANON = {
    "asystole": "Asystole", "bradycardia": "Bradycardia", "tachycardia": "Tachycardia",
    "ventricular_tachycardia": "VentricularTachycardia",
    "ventricular_flutter_fib": "VentricularFlutterFib",
}


def parse_alarm_comments(comments):
    alarm_type, verdict = None, None
    for c in (comments or []):
        s = str(c).strip()
        low = s.lower().replace(" ", "_")
        if low in ALARM_TYPE_CANON:
            alarm_type = ALARM_TYPE_CANON[low]
        elif "true" in s.lower() and "alarm" in s.lower():
            verdict = True
        elif "false" in s.lower() and "alarm" in s.lower():
            verdict = False
    return alarm_type, verdict


def _bandpass(x, fs, lo, hi, order=4):
    nyq = fs / 2.0
    hi_ok = hi is not None and hi < nyq * 0.99
    if lo and hi_ok:
        sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    elif hi_ok:
        sos = butter(order, hi, btype="low", fs=fs, output="sos")
    elif lo:
        sos = butter(order, lo, btype="high", fs=fs, output="sos")
    else:
        return x
    if x.size <= 3 * (2 * order + 1):
        return x
    return sosfiltfilt(sos, x)


def _artifact_mask(x_raw, fs, flat_win_s=1.0, flat_eps=1e-6, rail_tol=1e-9):
    x = np.asarray(x_raw, dtype=float)
    bad = ~np.isfinite(x)
    xf = np.where(bad, 0.0, x)
    w = max(3, int(round(flat_win_s * fs)))
    m1 = uniform_filter1d(xf, w, mode="nearest")
    m2 = uniform_filter1d(xf * xf, w, mode="nearest")
    roll_std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    bad |= roll_std < flat_eps
    finite = xf[np.isfinite(x)]
    if finite.size:
        lo, hi = finite.min(), finite.max()
        if hi > lo:
            bad |= (np.abs(x - lo) <= rail_tol) | (np.abs(x - hi) <= rail_tol)
    return bad


def load_cinc_records(cinc_dir, target_fs=TARGET_FS, require=("ECG", "PPG"),
                      optional=("ABP", "RESP"), limit=None, verbose=True):
    """Load CinC-2015 into the adapter's record contract."""
    import wfdb
    heas = sorted(glob.glob(os.path.join(cinc_dir, "**", "*.hea"), recursive=True))
    records, skipped = [], {"no_label": 0, "missing_core": 0, "read_error": 0}

    for hea in heas:
        base = hea[:-4]
        try:
            rec = wfdb.rdrecord(base)
        except Exception:
            skipped["read_error"] += 1
            continue
        alarm_type, true_alarm = parse_alarm_comments(rec.comments)
        if alarm_type is None or true_alarm is None:
            skipped["no_label"] += 1
            continue

        names = list(rec.sig_name or [])
        role_idx = {}
        for i, n in enumerate(names):
            for role, pool in (("ECG", ECG_NAMES), ("PPG", PPG_NAMES),
                               ("ABP", ABP_NAMES), ("RESP", RESP_NAMES)):
                if n in pool and role not in role_idx:
                    role_idx[role] = i
        if any(r not in role_idx for r in require):
            skipped["missing_core"] += 1
            continue

        roles = list(require) + [r for r in optional if r in role_idx]
        fs_in = float(rec.fs)
        chans, masks = [], []
        for role in roles:
            raw = np.asarray(rec.p_signal[:, role_idx[role]], dtype=float)
            bad = _artifact_mask(raw, fs_in)
            x = np.where(np.isfinite(raw), raw, 0.0)
            if abs(fs_in - target_fs) > 1e-9:
                up, down = int(round(target_fs)), int(round(fs_in))
                g = np.gcd(up, down)
                x = resample_poly(x, up // g, down // g)
                idx = np.clip((np.arange(x.size) * fs_in / target_fs).astype(int),
                              0, bad.size - 1)
                bad = bad[idx]
            lo, hi = ROLE_BANDS[role]
            x = _bandpass(x, target_fs, lo, hi)
            chans.append(x)
            masks.append(bad)

        L = min(c.size for c in chans)
        sig = np.vstack([c[:L] for c in chans])
        mask = np.vstack([m[:L] for m in masks])
        mu = sig.mean(axis=1, keepdims=True)
        sd = sig.std(axis=1, keepdims=True)
        sd_safe = np.where(sd < 1e-12, 1.0, sd)
        sig = (sig - mu) / sd_safe

        records.append(dict(
            record_id=os.path.basename(base), signal=sig, channels=roles,
            alarm_type=alarm_type, true_alarm=bool(true_alarm),
            alarm_onset_sample=int(round(ALARM_TIME_S * target_fs)),
            artifact_mask=mask, fs=float(target_fs),
            zscore_mean=mu.ravel().tolist(), zscore_std=sd_safe.ravel().tolist(),
            enrichment=[r for r in optional if r in role_idx],
        ))
        if limit and len(records) >= limit:
            break

    if verbose:
        n_true = sum(r["true_alarm"] for r in records)
        print(f"[cinc] loaded {len(records)} records "
              f"({n_true} true / {len(records)-n_true} false alarms) | skipped: {skipped}")
        from collections import Counter
        types = Counter(r["alarm_type"] for r in records)
        print(f"[cinc] alarm types: {dict(types)}")
        tiers = Counter(tuple(r["enrichment"]) for r in records)
        print(f"[cinc] enrichment tiers: "
              f"{ {('core' if not k else '+'.join(k)): v for k, v in tiers.items()} }")
        if records:
            print(f"[cinc] example per-channel std after z-score (should be ~1.0): "
                  f"{np.round(records[0]['signal'].std(axis=1), 3)}")
    return records


# --------------------------------------------------------------------------- #
def _write_fake_cinc(dirpath, n_per_type=2):
    """Write CinC-shaped WFDB records spanning ALL FIVE alarm types for realistic demos."""
    import wfdb
    os.makedirs(dirpath, exist_ok=True)
    rng = np.random.default_rng(7)
    specs = [("Asystole", "False alarm"), ("Bradycardia", "True alarm"),
             ("Tachycardia", "True alarm"), ("Ventricular_Tachycardia", "False alarm"),
             ("Ventricular_Flutter_Fib", "False alarm")]
    idx = 0
    for atype, verdict in specs:
        for _ in range(n_per_type):
            T = int(CINC_FS_NATIVE * 330)
            t = np.arange(T) / CINC_FS_NATIVE
            ph = 2 * np.pi * 1.2 * t
            ecg = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.02 * rng.standard_normal(T)
            ppg = 0.9 * np.sin(ph - 0.6) + 0.15 * np.sin(2 * np.pi * 0.25 * t) \
                + 0.02 * rng.standard_normal(T) + 1.0
            if idx == 0:                                    # a flat-line artifact
                ppg[int(2 * CINC_FS_NATIVE):int(6 * CINC_FS_NATIVE)] = ppg[0]
            wfdb.wrsamp(f"rec{idx:03d}", fs=int(CINC_FS_NATIVE), units=["mV", "NU"],
                        sig_name=["II", "PLETH"], p_signal=np.vstack([ecg, ppg]).T,
                        comments=[atype, verdict], write_dir=dirpath)
            idx += 1


def _demo():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _write_fake_cinc(td, n_per_type=2)
        recs = load_cinc_records(td)
        r = recs[0]
        print(f"\nrecord {r['record_id']}: channels {r['channels']}, signal {r['signal'].shape}, "
              f"fs {r['fs']}, alarm {r['alarm_type']}/true={r['true_alarm']} "
              f"@ {r['alarm_onset_sample']} ({r['alarm_onset_sample']/r['fs']:.0f}s)")
        art = r["artifact_mask"].any(axis=0)
        print(f"artifact: {100*art.mean():.1f}% flagged (kept, not deleted) | "
              f"zscore_std stored: {np.round(r['zscore_std'],3)}")


if __name__ == "__main__":
    _demo()
