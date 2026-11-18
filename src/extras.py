"""
Analyses reported in the IoMT-SecAlarmBench paper.

Runs on the combined four-setting table and window tensor built by
scripts/reproduce_paper.py, the loaded CinC records, and a detector factory. It
does not change the benchmark: every step reads the same windows and writes CSV or
JSON files to cfg.out_dir. Steps are resumable; detector fits and beat detections
are cached.

Steps:
  0  label audit        alarm-zone windows (CinC event onset in [290 s, 300 s]),
                        artifact-confirmed vs monitor-false-alarm faults, host zones
  1  score cache        every detector x seed, fitted once on training normals
  2  beat pass          ECG beats and PPG pulses for every row (coherence features
                        and alarm flags), computed once and cached
  3  coherence baseline heart-rate, pulse-rate and pulse-transit coherence detector
  4  detection tables   Tables III-V and the morphology x level grid (Fig. 2), as
                        mean +/- sd over seeds with record-grouped bootstrap CIs
  5  label sensitivity  key numbers with alarm-zone windows removed from the normal
                        class, and host-matched AUROC by host zone
  6  position-matched   attacks injected into post-onset event and monitor-false-alarm
     hosts              windows (the same positions as the event and fault classes)
  7  discrimination     default forest, artifact-only, four-class, position-matched,
     variants           and event re-slicing at 1 s on the training folds (Table VII)
  8  alarm oracle       g(X_clean) vs g(X_attack) for the CinC 2015 asystole,
                        bradycardia and tachycardia definitions: validation against
                        CinC verdicts, alarm creation and suppression (Table VI)
  9  attribution        channel-attribution baseline on C1-PPG + C1-ECG
 10  WUSTL audit        identifiers, TCP flags and modality ablations over 5 split
                        seeds (Table VIII)
 11  hyperparameters    read from the detector objects (Table II)

Label modes: "v1" keeps every physiology-normal window in the normal class (the
default, ``primary_labels``); "corrected" removes the alarm-zone windows.

Usage:
    import extras
    cfg = extras.ExtrasConfig(out_dir="results/paper")
    res = extras.run_all(df, X, feat_cols, records, make_detectors, cfg)
"""

from __future__ import annotations

import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from injection_generator import GeneratorConfig, Window, generate_benchmark
from cinc_adapter import build_windows
from eval_harness import window_features

FS = 125.0
CORE = ("ECG", "PPG")
SETTING_SHORT = {"C1_PPG": "C1-PPG", "C1_ECG": "C1-ECG",
                 "C2_JOINT_INDEPENDENT": "C2", "C3_JOINT_COORDINATED": "C3"}
SUSTAINED = ("stuck_at", "drift", "bias")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ExtrasConfig:
    out_dir: str
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    single_seed_detectors: Sequence[str] = ("Coherence",)     # deterministic
    feature_detectors: Sequence[str] = ("IsolationForest",)   # take the 22 features
    far_main: float = 0.05            # Table V budget
    far_matched: float = 0.10         # Table IV budget
    n_boot: int = 1000
    boot_seed: int = 20261114
    primary_labels: str = "v1"        # "v1" (all normal windows) or "corrected" (alarm zone out)
    alarm_guard_s: float = 10.0       # CinC: event onset within 10 s before the alarm
    artifact_frac: float = 0.10
    rf_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, class_weight="balanced", random_state=20260723, n_jobs=-1))
    post_hosts_per_record: int = 3
    pm_settings: Sequence[str] = ("C1_PPG", "C1_ECG", "C2_JOINT_INDEPENDENT", "C3_JOINT_COORDINATED")
    pm_seed_offset: int = 404         # post-onset hosts get their own draws
    event_reslice_stride_s: float = 1.0
    base_generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    wustl_csv: Optional[str] = None
    wustl_label_col: str = "Label"
    wustl_dup_cols: Sequence[str] = ("Attack Category",)
    wustl_id_cols: Sequence[str] = ("SrcAddr", "DstAddr", "Sport", "Dport", "SrcMac", "DstMac")
    wustl_flag_cols: Sequence[str] = ("Flgs",)
    wustl_bio_cols: Sequence[str] = ("Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
                                     "Heart_rate", "Resp_Rate", "ST")
    wustl_counter_cols: Sequence[str] = ("Packet_num",)   # running packet counter
    wustl_rf_params: Optional[dict] = None  # defaults to rf_params
    n_jobs: int = -1
    run_attribution: bool = True
    beat_detector: str = "simple"     # "simple" (built-in) or "neurokit2" (NeuroKit2 ECG + Elgendi PPG)
    verbose: bool = True


def _log(cfg, *a):
    if cfg.verbose:
        print(*a, flush=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# --------------------------------------------------------------------------- #
# Fast (weighted) metrics -- weights implement the record-grouped bootstrap
# --------------------------------------------------------------------------- #
class PairMetric:
    """AUROC / AUPRC / detection rate for fixed positive and negative row sets.

    Row weights come from record multiplicities, so a bootstrap draw costs two
    bincounts. Ties are handled exactly (AUROC counts half; AUPRC uses distinct
    thresholds), matching sklearn on unweighted data.
    """

    def __init__(self, scores, pos_idx, neg_idx, rec_code):
        s = np.asarray(scores, dtype=np.float64)
        self.pos_idx = np.asarray(pos_idx)
        self.neg_idx = np.asarray(neg_idx)
        sp, sn = s[self.pos_idx], s[self.neg_idx]
        u, inv = np.unique(np.r_[sp, sn], return_inverse=True)
        self.nu = u.size
        self.inv_p = inv[:sp.size]
        self.inv_n = inv[sp.size:]
        self.rp = rec_code[self.pos_idx]
        self.rn = rec_code[self.neg_idx]
        self.sp = sp

    def _w(self, recw):
        wp = np.ones(self.rp.size) if recw is None else recw[self.rp]
        wn = np.ones(self.rn.size) if recw is None else recw[self.rn]
        return wp, wn

    def auroc(self, recw=None):
        wp, wn = self._w(recw)
        P = np.bincount(self.inv_p, weights=wp, minlength=self.nu)
        N = np.bincount(self.inv_n, weights=wn, minlength=self.nu)
        tp, tn = P.sum(), N.sum()
        if tp <= 0 or tn <= 0:
            return np.nan
        above = tp - np.cumsum(P)                 # positive weight strictly above value
        return float((N * (above + 0.5 * P)).sum() / (tp * tn))

    def auprc(self, recw=None):
        wp, wn = self._w(recw)
        P = np.bincount(self.inv_p, weights=wp, minlength=self.nu)[::-1]
        N = np.bincount(self.inv_n, weights=wn, minlength=self.nu)[::-1]
        tp_tot = P.sum()
        if tp_tot <= 0:
            return np.nan
        TP, FP = np.cumsum(P), np.cumsum(N)
        keep = (P + N) > 0
        TP, FP = TP[keep], FP[keep]
        prec = TP / np.maximum(TP + FP, 1e-12)
        rec = TP / tp_tot
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))

    def floor(self, recw=None):
        wp, wn = self._w(recw)
        return float(wp.sum() / (wp.sum() + wn.sum()))

    def det_rate(self, thr, recw=None):
        wp, _ = self._w(recw)
        return float((wp * (self.sp >= thr)).sum() / max(wp.sum(), 1e-12))


def rate_metric(flag, idx, rec_code):
    """Weighted mean of a boolean over rows idx (for FARs / alarm rates)."""
    f = np.asarray(flag, dtype=float)[idx]
    rc = rec_code[idx]

    def _f(recw=None):
        w = np.ones(rc.size) if recw is None else recw[rc]
        return float((w * f).sum() / max(w.sum(), 1e-12)) if w.sum() > 0 else np.nan
    _f.n = int(len(idx))
    return _f


def boot_ci(fn_by_seed, rec_ids, n_boot, seed, alpha=0.05):
    """Record-grouped bootstrap. fn_by_seed: list of callables f(recw)->float, one per
    seed; each draw averages the seeds (so the CI covers records and seeds together)."""
    rng = np.random.default_rng(seed)
    nrec = len(rec_ids)
    point = float(np.nanmean([f(None) for f in fn_by_seed]))
    if n_boot <= 0:
        return point, np.nan, np.nan
    vals = np.empty(n_boot)
    for b in range(n_boot):
        recw = rng.multinomial(nrec, np.full(nrec, 1.0 / nrec)).astype(float)
        vals[b] = np.nanmean([f(recw) for f in fn_by_seed])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Step 0 -- labels, zones, hosts
# --------------------------------------------------------------------------- #
def clean_window_meta(records, cfg: ExtrasConfig, win_sec=10.0, hop_sec=5.0):
    """Per clean window: zone (pre / alarm_zone / post), artifact fraction, start."""
    rows = []
    for r in records:
        C, T = r["signal"].shape
        fs = float(r.get("fs", FS))
        W, H = int(round(win_sec * fs)), int(round(hop_sec * fs))
        onset = int(r["alarm_onset_sample"])
        guard = int(round(cfg.alarm_guard_s * fs))
        amask = r.get("artifact_mask")
        for w_idx, start in enumerate(range(0, T - W + 1, H)):
            end = start + W
            art = float(amask[:, start:end].any(axis=0).mean()) if amask is not None else 0.0
            zone = "post" if end > onset else ("alarm_zone" if end > onset - guard else "pre")
            rows.append(dict(window_id=f"{r['record_id']}_w{w_idx:03d}",
                             record_id=r["record_id"], start=start, end=end, zone=zone,
                             artifact_frac=round(art, 4), ends_at_onset=(end == onset),
                             rec_true_alarm=bool(r["true_alarm"]),
                             rec_alarm_type=str(r["alarm_type"])))
    return pd.DataFrame(rows)


def prepare(df, records, cfg: ExtrasConfig):
    """Add zone / fine class / host links to a copy of df (row order kept)."""
    df = df.reset_index(drop=True).copy()
    if "split" not in df.columns:
        raise ValueError("df needs a 'split' column (run patient_wise_split first).")
    if "setting" not in df.columns:
        df["setting"] = np.where(df.cls == "attack", "C1_PPG", "none")
    meta = clean_window_meta(records, cfg).set_index("window_id")
    is_att = (df.cls == "attack").values

    wm = meta.reindex(df["window_id"])                 # attacks inherit their host's meta
    for c in ("zone", "artifact_frac", "start", "ends_at_onset",
              "rec_true_alarm", "rec_alarm_type"):
        df[c] = wm[c].values
    df["host_zone"] = np.where(is_att, df["zone"], "")
    cls2 = np.select(
        [df.cls == "physiology_normal", df.cls == "physiology_event",
         (df.cls == "fault_artifact") & (df.artifact_frac > cfg.artifact_frac),
         df.cls == "fault_artifact", df.cls == "attack"],
        ["normal", "event", "artifact", "monitor_fa", "attack"], default="other")
    df["cls2"] = cls2
    df["normal_v1"] = df.cls2 == "normal"
    df["normal_corrected"] = (df.cls2 == "normal") & (df.zone == "pre")

    clean_idx = pd.Series(np.flatnonzero(~is_att), index=df.loc[~is_att, "window_id"].values)
    df["host_row"] = -1
    df.loc[is_att, "host_row"] = clean_idx.reindex(df.loc[is_att, "window_id"]).fillna(-1).astype(int).values
    df["is_host"] = False
    host_rows = df.loc[is_att, "host_row"].values
    df.loc[np.unique(host_rows[host_rows >= 0]), "is_host"] = True
    missing = int((df.loc[is_att, "host_row"] < 0).sum())
    if missing:
        warnings.warn(f"{missing} attack rows have no clean host row in df")
    df["rec_code"] = pd.factorize(df["record_id"])[0]
    return df, meta.reset_index()


def label_audit(df, meta, cfg):
    out = {}
    clean = df[df.cls != "attack"]
    out["class_by_zone"] = pd.crosstab(clean.cls2, clean.zone)
    hosts = df[df.is_host]
    out["hosts_by_zone"] = pd.crosstab(hosts.zone, hosts.rec_true_alarm.map({True: "true alarm", False: "false alarm"}))
    out["n_normal_v1"] = int(df.normal_v1.sum())
    out["n_normal_in_alarm_zone"] = int((df.normal_v1 & (df.zone == "alarm_zone")).sum())
    out["n_hosts"] = int(df.is_host.sum())
    out["n_hosts_alarm_zone_true"] = int((df.is_host & (df.zone == "alarm_zone") & df.rec_true_alarm).sum())
    out["fault_split"] = clean.cls2.value_counts().reindex(["artifact", "monitor_fa"]).to_dict()
    out["records"] = int(meta.record_id.nunique())
    rec = meta.drop_duplicates("record_id")
    out["records_with_post_onset_windows"] = int(meta[meta.zone == "post"].record_id.nunique())
    out["records_true_alarm"] = int(rec.rec_true_alarm.sum())
    return out


# --------------------------------------------------------------------------- #
# Step 1 -- score cache
# --------------------------------------------------------------------------- #
def _fingerprint(df):
    ids = df["window_id"].astype(str).values
    return f"{len(ids)}|{ids[0]}|{ids[-1]}|{int((df.cls == 'attack').sum())}"


def describe_detector(det):
    out = {"class": type(det).__name__}
    for k, v in vars(det).items():
        if isinstance(v, (bool, int, float, str)) and not k.startswith("_"):
            out[k] = v
        elif isinstance(v, (tuple, list)) and len(v) <= 8 and all(isinstance(x, (int, float, str)) for x in v):
            out[k] = list(v)
        elif hasattr(v, "get_params") and hasattr(v, "fit"):
            for kk, vv in v.get_params().items():
                if isinstance(vv, (bool, int, float, str, type(None))):
                    out[f"{type(v).__name__}.{kk}"] = vv
    try:
        import torch
        n = 0
        for v in vars(det).values():
            if isinstance(v, torch.nn.Module):
                n += sum(p.numel() for p in v.parameters())
            if isinstance(v, torch.optim.Optimizer):
                g = v.param_groups[0]
                out["optimizer"] = type(v).__name__
                out.update({f"opt.{k}": g[k] for k in ("lr", "weight_decay") if k in g})
        if n:
            out["n_params"] = int(n)
    except ImportError:
        pass
    return out


def build_score_cache(df, X, F, make_detectors, cfg, train_mask):
    sdir = os.path.join(cfg.out_dir, "scores")
    os.makedirs(sdir, exist_ok=True)
    fp = _fingerprint(df)
    hp_path = os.path.join(cfg.out_dir, "hyperparameters.json")
    hparams = json.load(open(hp_path)) if os.path.exists(hp_path) else {}
    fit_times = {}
    for seed in cfg.seeds:
        set_all_seeds(seed)
        dets = make_detectors(seed)
        for name in list(dets):
            if name in cfg.single_seed_detectors and seed != cfg.seeds[0]:
                continue
            path = os.path.join(sdir, f"{name}__seed{seed}.npy")
            mpath = path.replace(".npy", ".json")
            if os.path.exists(path) and os.path.exists(mpath):
                if json.load(open(mpath)).get("fingerprint") == fp:
                    continue
            det = dets[name]
            Xin = F if name in cfg.feature_detectors else X
            set_all_seeds(seed)
            t0 = time.perf_counter()
            det.fit(Xin[train_mask])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            s = np.asarray(det.score(Xin), dtype=np.float32).ravel()
            lat_ms = 1000 * (time.perf_counter() - t0) / len(s)
            if s.size != len(df):
                raise ValueError(f"{name} returned {s.size} scores for {len(df)} rows")
            np.save(path, s)
            _save_json(dict(fingerprint=fp, fit_s=fit_s, latency_ms=lat_ms, seed=seed), mpath)
            fit_times.setdefault(name, []).append(fit_s)
            if name not in hparams:
                hparams[name] = describe_detector(det)
                _save_json(hparams, hp_path)
            _log(cfg, f"  [scores] {name} seed {seed}: fit {fit_s:.1f}s, {lat_ms:.4f} ms/window")
            del det
        del dets
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    return hparams


def load_scores(cfg, df):
    sdir = os.path.join(cfg.out_dir, "scores")
    fp = _fingerprint(df)
    scores = {}
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".npy"):
            continue
        name, seed = fn[:-4].split("__seed")
        meta = json.load(open(os.path.join(sdir, fn.replace(".npy", ".json"))))
        if meta.get("fingerprint") != fp:
            continue
        scores.setdefault(name, {})[int(seed)] = np.load(os.path.join(sdir, fn))
    return scores


# --------------------------------------------------------------------------- #
# Step 2 -- beat pass (ECG beats, PPG pulses, coherence features, alarm flags)
# --------------------------------------------------------------------------- #
BEAT_COLS = ["e_n", "p_n", "e_hr", "p_hr", "e_maxgap", "p_maxgap",
             "e_asys", "e_brady", "e_tachy", "p_asys", "p_brady", "p_tachy",
             "hr_diff", "unmatched", "ptt_sd", "ptt_maxdev", "cnt_mismatch"]
COH_FEATS = ["hr_diff", "unmatched", "ptt_sd", "ptt_maxdev", "cnt_mismatch"]


def _ecg_energy(x, fs):
    d = np.diff(x, prepend=x[0])
    k = max(1, int(round(0.15 * fs)))
    return np.convolve(d * d, np.ones(k) / k, mode="same")


def _ppg_slope(p, fs):
    d = np.diff(p, prepend=p[0])
    k = max(1, int(round(0.05 * fs)))
    return np.convolve(d, np.ones(k) / k, mode="same")


def _ref_height(sig, fs, min_dist_s):
    """Typical beat height: 90th percentile of candidate peak heights (beats make up
    15-70% of candidates between 40 and 160 bpm, so this lands on beats, not noise)."""
    cand, _ = find_peaks(sig, distance=max(1, int(round(min_dist_s * fs))))
    h = sig[cand]
    h = h[h > 0]
    if h.size == 0:
        return 0.0
    return float(np.percentile(h, 90))


def _detect(sig, fs, min_dist_s, floor, rel=0.30):
    """Peaks above rel x max(window reference, floor). No search-back: a monitor that
    fills long gaps with sub-threshold peaks would hide the asystole it must report."""
    cand, _ = find_peaks(sig, distance=max(1, int(round(min_dist_s * fs))))
    if cand.size == 0:
        return cand
    h = sig[cand]
    pos = h[h > 0]
    ref = max(float(np.percentile(pos, 90)) if pos.size else 0.0, floor)
    if ref <= 0:
        return cand[:0]
    return cand[h > rel * ref]


def _run_mean(rr, k, below=None, above=None):
    """True if the mean of some k consecutive RR intervals is below/above a bound,
    i.e. the heart rate computed over k+1 consecutive beats crosses the limit."""
    if rr.size < k:
        return False
    m = np.convolve(rr, np.ones(k) / k, mode="valid")
    return bool((m < below).any()) if below is not None else bool((m > above).any())


def _refine(x, peaks, fs, half_s=0.10):
    """Move each energy peak to the steepest point of the raw signal nearby."""
    if peaks.size == 0:
        return peaks
    d = np.abs(np.diff(x, prepend=x[0]))
    h = int(round(half_s * fs))
    out = np.array([max(0, p - h) + int(np.argmax(d[max(0, p - h):p + h + 1])) for p in peaks])
    out = np.unique(out)
    if out.size > 1:                       # merge refinements closer than 200 ms
        keep = np.r_[True, np.diff(out) > int(round(0.2 * fs))]
        out = out[keep]
    return out


def _rate_flags(b, T, fs):
    dur = T / fs
    t = b / fs
    if t.size == 0:
        return 0, 0.0, dur, 1.0, 0.0, 0.0
    gaps = np.diff(np.r_[0.0, t, dur])
    rr = np.diff(t)
    hr = 60.0 / np.median(rr) if rr.size else 0.0
    return (t.size, hr, float(gaps.max()), float(gaps.max() >= 4.0),
            float(_run_mean(rr, 4, above=1.5)), float(_run_mean(rr, 16, below=60.0 / 140.0)))


def _nk_peaks(ecg, ppg, fs):
    """NeuroKit2 R peaks (default 'neurokit' method) and Elgendi PPG systolic peaks."""
    import warnings as _w
    import neurokit2 as nk
    sr = int(round(fs))
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        try:
            _, info = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=sr, method="neurokit"),
                                   sampling_rate=sr, method="neurokit")
            be = np.asarray(info["ECG_R_Peaks"], dtype=int)
        except Exception:
            be = np.array([], dtype=int)
        try:
            pk = nk.ppg_findpeaks(nk.ppg_clean(ppg, sampling_rate=sr, method="elgendi"),
                                  sampling_rate=sr, method="elgendi")
            bp = np.asarray(pk["PPG_Peaks"], dtype=int)
        except Exception:
            bp = np.array([], dtype=int)
    return np.unique(be), np.unique(bp)


def _beat_row(ecg, ppg, fs, floors, detector="simple"):
    T = ecg.size
    if detector == "neurokit2":
        be, bp = _nk_peaks(ecg, ppg, fs)
    else:
        be = _refine(ecg, _detect(_ecg_energy(ecg, fs), fs, 0.25, floors[0]), fs)
        bp = _detect(_ppg_slope(ppg, fs), fs, 0.27, floors[1])
    en, ehr, egap, ea, eb, et = _rate_flags(be, T, fs)
    pn, phr, pgap, pa, pb, pt = _rate_flags(bp, T, fs)
    te, tp = be / fs, bp / fs
    dur = T / fs
    hr_diff = abs(ehr - phr) if (en >= 2 and pn >= 2) else np.nan
    # beat matching: each R peak should be followed by one PPG upstroke 0.08-0.6 s later
    r_in = te[te < dur - 0.6]
    p_in = tp[tp > 0.6]
    ptt = []
    mr = 0
    for r in r_in:
        m = tp[(tp > r + 0.08) & (tp < r + 0.6)]
        if m.size:
            mr += 1
            ptt.append(m[0] - r)
    mp = sum(bool(((te > p - 0.6) & (te < p - 0.08)).any()) for p in p_in)
    if r_in.size and p_in.size:
        unmatched = 1.0 - min(mr / r_in.size, mp / p_in.size)
    else:
        unmatched = 1.0
    ptt = np.array(ptt)
    if ptt.size >= 3:
        ptt_sd = float(ptt.std())
        ptt_maxdev = float(np.max(np.abs(ptt - np.median(ptt))))
    else:
        ptt_sd = ptt_maxdev = np.nan
    cm = 0.0
    for t0 in np.arange(0.0, max(dur - 3.0, 0.0) + 1e-9, 1.0):
        ce = ((te >= t0) & (te < t0 + 3.0)).sum()
        cp = ((tp >= t0 + 0.3) & (tp < t0 + 3.3)).sum()
        cm = max(cm, abs(float(ce) - float(cp)))
    return [en, pn, ehr, phr, egap, pgap, ea, eb, et, pa, pb, pt,
            hr_diff, unmatched, ptt_sd, ptt_maxdev, cm]


def _beat_chunk(Xc, fs, floors, detector="simple"):
    return np.array([_beat_row(Xc[i, 0].astype(np.float64), Xc[i, 1].astype(np.float64), fs, floors,
                               detector) for i in range(len(Xc))], dtype=np.float32)


def detector_floors(X, idx, fs=FS, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    re = [_ref_height(_ecg_energy(X[i, 0].astype(float), fs), fs, 0.25) for i in pick]
    rp = [_ref_height(_ppg_slope(X[i, 1].astype(float), fs), fs, 0.27) for i in pick]
    return (0.25 * float(np.median(re)), 0.25 * float(np.median(rp)))


def beat_pass(X, floors, fs=FS, n_jobs=-1, chunk=4000, detector="simple"):
    starts = list(range(0, len(X), chunk))
    try:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs)(delayed(_beat_chunk)(np.asarray(X[s:s + chunk]), fs, floors,
                                                             detector) for s in starts)
    except ImportError:
        parts = [_beat_chunk(np.asarray(X[s:s + chunk]), fs, floors, detector) for s in starts]
    return pd.DataFrame(np.vstack(parts), columns=BEAT_COLS)


def alarm_flags(B):
    gE = (B.e_asys + B.e_brady + B.e_tachy) > 0
    gP = (B.p_asys + B.p_brady + B.p_tachy) > 0
    gF = ((B.e_asys * B.p_asys) + (B.e_brady * B.p_brady) + (B.e_tachy * B.p_tachy)) > 0
    return pd.DataFrame({"g_ecg": gE.values, "g_ppg": gP.values, "g_fused": gF.values})


# --------------------------------------------------------------------------- #
# Step 3 -- coherence detector
# --------------------------------------------------------------------------- #
class CoherenceDetector:
    """Heart-rate / pulse-rate / pulse-transit coherence, fit on clean normals only.

    Features: |ECG HR - PPG pulse rate|, unmatched beat fraction, PTT SD, max PTT
    deviation, max beat-count mismatch in 3 s sub-windows. Score = sum of positive
    robust z-scores (median / IQR of training normals). Missing values (too few
    beats) take the training 99th percentile.
    """
    name = "Coherence"

    def __init__(self, feats=COH_FEATS):
        self.feats = list(feats)

    def fit(self, B):
        A = B[self.feats].astype(float)
        self.fill_ = A.quantile(0.99)
        A = A.fillna(self.fill_)
        self.med_ = A.median()
        iqr = A.quantile(0.75) - A.quantile(0.25)
        self.scale_ = (iqr / 1.349).where(iqr > 1e-9, A.std() + 1e-9)
        return self

    def score(self, B):
        A = B[self.feats].astype(float).fillna(self.fill_)
        z = (A - self.med_) / self.scale_
        return np.clip(z.values, 0, None).sum(axis=1)


# --------------------------------------------------------------------------- #
# Step 4 -- detection tables
# --------------------------------------------------------------------------- #
def _best_f1_threshold(y, s):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(s[order][int(np.argmax(f1))])


def _masks(df, mode):
    normal = df["normal_v1" if mode == "v1" else "normal_corrected"].values
    split = df.split.values
    return dict(
        train_norm=normal & (split == "train"),
        val_norm=normal & (split == "val"),
        test_norm=normal & (split == "test"),
        test=split == "test",
        val=split == "val",
        att=(df.cls == "attack").values,
        fault=df.cls2.isin(["artifact", "monitor_fa"]).values,
        artifact=(df.cls2 == "artifact").values,
        monitor=(df.cls2 == "monitor_fa").values,
        event=(df.cls2 == "event").values,
    )


def _subset_defs(df):
    """Named attack subsets (row masks) used in the tables."""
    st, fam, mor = df.setting.values, df.family.values, df.morphology.values
    lev = df.level.values
    att = (df.cls == "attack").values
    d = {}
    for s in SETTING_SHORT:
        a = att & (st == s)
        if not a.any():
            continue
        sh = SETTING_SHORT[s]
        d[f"{sh}|all"] = a
        d[f"{sh}|replay"] = a & (fam == "replay")
        d[f"{sh}|fdi"] = a & (fam == "fdi") & (mor != "rate_warp")
        d[f"{sh}|spoofing"] = a & (fam == "spoofing")
        d[f"{sh}|fdi_spike"] = a & (fam == "fdi") & (mor == "spike")
        d[f"{sh}|fdi_spike_L1"] = a & (fam == "fdi") & (mor == "spike") & (lev == 1)
        d[f"{sh}|fdi_sust"] = a & (fam == "fdi") & np.isin(mor, SUSTAINED)
        d[f"{sh}|spoof_sust"] = a & (fam == "spoofing") & np.isin(mor, SUSTAINED)
        if (a & (mor == "rate_warp")).any():
            d[f"{sh}|rate_warp"] = a & (mor == "rate_warp")
    return d


def _host_negs(df, pos_mask):
    hr = df.host_row.values[pos_mask]
    return np.unique(hr[hr >= 0])


def detection_tables(df, scores, cfg, mode):
    """Point estimates (mean +- sd over seeds) for Tables III, IV, V and the grid."""
    M = _masks(df, mode)
    rc = df.rec_code.values
    subsets = _subset_defs(df)
    rows, grid_rows = [], []
    for name, by_seed in scores.items():
        per_seed = []
        for seed, s in by_seed.items():
            r = {"detector": name, "seed": seed}
            te = M["test"]
            neg_n = np.flatnonzero(te & M["test_norm"])
            thr_far = {f: float(np.quantile(s[M["val_norm"]], 1 - f)) for f in (cfg.far_main, cfg.far_matched)}
            # equal-FAR thresholds on TEST normals (operating-point normalization)
            thr_eq = {f: float(np.quantile(s[neg_n], 1 - f)) for f in (cfg.far_main, cfg.far_matched)}
            c1 = subsets.get("C1-PPG|all")
            # Table III
            if c1 is not None:
                pos = np.flatnonzero(te & c1)
                r["T3_AUPRC"] = PairMetric(s, pos, neg_n, rc).auprc()
                negf = np.flatnonzero(te & (M["test_norm"] | M["fault"]))
                r["T3_AUPRC_fault"] = PairMetric(s, pos, negf, rc).auprc()
                vpos = np.flatnonzero(M["val"] & c1)
                vneg = np.flatnonzero(M["val_norm"])
                y = np.r_[np.ones(vpos.size), np.zeros(vneg.size)]
                thr_f1 = _best_f1_threshold(y, np.r_[s[vpos], s[vneg]])
                for k, m in (("normal", M["test_norm"]), ("fault", M["fault"]), ("event", M["event"])):
                    r[f"T3_FAR_{k}_maxF1"] = float((s[te & m] >= thr_f1).mean())
            # Table IV (C1-PPG subsets)
            for key in ("replay", "fdi_spike", "fdi_spike_L1", "fdi_sust", "spoof_sust"):
                m = subsets.get(f"C1-PPG|{key}")
                if m is None or not (te & m).any():
                    continue
                pm = PairMetric(s, np.flatnonzero(te & m), neg_n, rc)
                r[f"T4_AUPRC_{key}"] = pm.auprc()
                r[f"T4_floor_{key}"] = pm.floor()
                if key == "replay":
                    r["T4_AUROC_replay"] = pm.auroc()
                r[f"T4_det10_{key}"] = pm.det_rate(thr_far[cfg.far_matched])
            for k, m in (("fault", M["fault"]), ("artifact", M["artifact"]),
                         ("monitor", M["monitor"]), ("event", M["event"])):
                pos = np.flatnonzero(te & m)
                if pos.size:
                    r[f"T4_AUROC_{k}_vs_normal"] = PairMetric(s, pos, neg_n, rc).auroc()
                    r[f"T4_FAR_{k}_val10"] = float((s[pos] >= thr_far[cfg.far_matched]).mean())
                    for f in (cfg.far_main, cfg.far_matched):
                        r[f"T4_FAR_{k}_eq{int(f * 100)}"] = float((s[pos] >= thr_eq[f]).mean())
            r["T4_FAR_normal_val10"] = float((s[neg_n] >= thr_far[cfg.far_matched]).mean())
            r["T4_ratio_val10"] = r.get("T4_FAR_fault_val10", np.nan) / max(r["T4_FAR_normal_val10"], 1e-12)
            for k in ("fault", "artifact", "monitor", "event"):
                for f in (cfg.far_main, cfg.far_matched):
                    key = f"T4_FAR_{k}_eq{int(f * 100)}"
                    if key in r:
                        r[f"T4_ratio_{k}_eq{int(f * 100)}"] = r[key] / f
            # Table V (all settings, host-matched)
            r["T5_test_FAR"] = float((s[neg_n] >= thr_far[cfg.far_main]).mean())
            for key, m in subsets.items():
                pos = np.flatnonzero(te & m)
                if not pos.size:
                    continue
                hn = _host_negs(df, te & m)
                r[f"T5_hmAUROC_{key}"] = PairMetric(s, pos, hn, rc).auroc()
                pa = PairMetric(s, pos, neg_n, rc)
                r[f"T5_AUROC_{key}"] = pa.auroc()
                r[f"T5_AUPRC_{key}"] = pa.auprc()
                r[f"T5_det5_{key}"] = pa.det_rate(thr_far[cfg.far_main])
            per_seed.append(r)
            # morphology x level grid (test)
            g = df.loc[te & M["att"], ["setting", "family", "morphology", "level", "host_row"]].copy()
            g["score"] = s[te & M["att"]]
            g["det5"] = g.score >= thr_far[cfg.far_main]
            for (st, fam, mor, lev), gg in g.groupby(["setting", "family", "morphology", "level"]):
                hn = np.unique(gg.host_row.values[gg.host_row.values >= 0])
                grid_rows.append(dict(detector=name, seed=seed, setting=SETTING_SHORT.get(st, st),
                                      family=fam, morphology=mor, level=int(lev), n=len(gg),
                                      det5=float(gg.det5.mean()),
                                      hmAUROC=PairMetric(np.r_[gg.score.values, s[hn]],
                                                         np.arange(len(gg)),
                                                         np.arange(len(gg), len(gg) + hn.size),
                                                         np.r_[np.zeros(len(gg), int), np.zeros(hn.size, int)]).auroc()))
        rows.extend(per_seed)
    long = pd.DataFrame(rows)
    num = [c for c in long.columns if c not in ("detector", "seed")]
    summ = long.groupby("detector")[num].agg(["mean", "std"])
    grid = pd.DataFrame(grid_rows).groupby(["detector", "setting", "family", "morphology", "level"])[["det5", "hmAUROC", "n"]].mean().reset_index()
    return long, summ, grid


def bootstrap_table(df, scores, cfg, mode, metric_keys):
    """Record-grouped 95% CIs (test records) for the listed metric keys."""
    M = _masks(df, mode)
    rc = df.rec_code.values
    te = M["test"]
    subsets = _subset_defs(df)
    test_recs = np.unique(rc[te])
    # weights live on the global record code space; draws only touch test records
    recmap = np.zeros(rc.max() + 1, int)
    recmap[test_recs] = np.arange(test_recs.size)

    def wrap(f):
        def g(recw_test):
            if recw_test is None:
                return f(None)
            w = np.zeros(rc.max() + 1)
            w[test_recs] = recw_test
            return f(w)
        return g

    out = []
    for name, by_seed in scores.items():
        for key in metric_keys:
            fns = []
            for seed, s in by_seed.items():
                neg_n = np.flatnonzero(te & M["test_norm"])
                thr5 = float(np.quantile(s[M["val_norm"]], 1 - cfg.far_main))
                kind, sub = key.split(":", 1)
                if kind in ("hmAUROC", "AUROC", "AUPRC", "det5"):
                    m = subsets.get(sub)
                    if m is None or not (te & m).any():
                        continue
                    pos = np.flatnonzero(te & m)
                    neg = _host_negs(df, te & m) if kind == "hmAUROC" else neg_n
                    pm = PairMetric(s, pos, neg, rc)
                    f = {"hmAUROC": pm.auroc, "AUROC": pm.auroc, "AUPRC": pm.auprc,
                         "det5": (lambda w, pm=pm, t=thr5: pm.det_rate(t, w))}[kind]
                elif kind == "faultAUROC":
                    msk = {"fault": M["fault"], "artifact": M["artifact"],
                           "monitor": M["monitor"], "event": M["event"]}[sub]
                    pm = PairMetric(s, np.flatnonzero(te & msk), neg_n, rc)
                    f = pm.auroc
                else:
                    continue
                fns.append(wrap(f))
            if not fns:
                continue
            pt, lo, hi = boot_ci(fns, test_recs, cfg.n_boot, cfg.boot_seed)
            out.append(dict(detector=name, metric=key, point=pt, lo=lo, hi=hi, n_seeds=len(fns)))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Step 6 -- position-matched hosts (post-onset event / monitor-false-alarm windows)
# --------------------------------------------------------------------------- #
def generate_post_onset(df, records, cfg, B_floors, with_features=True):
    recs = {r["record_id"]: r for r in records}
    cand = df[(df.cls != "attack") & df.cls2.isin(["event", "monitor_fa"])].copy()
    cand = cand.sort_values(["record_id", "start"])
    hosts = []
    for rid, g in cand.groupby("record_id", sort=False):
        idx = g.index.values
        if len(idx) > cfg.post_hosts_per_record:
            pick = np.unique(np.round(np.linspace(0, len(idx) - 1, cfg.post_hosts_per_record)).astype(int))
            idx = idx[pick]
        r = recs[rid]
        fs = float(r.get("fs", FS))
        W = int(round(10 * fs))
        for i in idx:
            st = int(df.at[i, "start"])
            hosts.append(Window(r["signal"][:, st:st + W].copy(), list(r["channels"]), rid, fs,
                                df.at[i, "window_id"]))
    _, banks, _, _ = build_windows(records, max_hosts_per_record=2, artifact_frac=cfg.artifact_frac)
    from injection_multichannel import (MultiChannelConfig, generate_multichannel,
                                                  joint_banks_from_channel_banks)
    mc = MultiChannelConfig(base=replace(cfg.base_generator,
                                         seed=cfg.base_generator.seed + cfg.pm_seed_offset))
    joint = joint_banks_from_channel_banks(banks, mc.pair)
    rows, beat_rows = [], []
    for setting in cfg.pm_settings:
        aws, man = generate_multichannel(hosts, settings=(setting,), mc=mc,
                                         replay_banks=banks, joint_banks=joint)
        for aw, (_, m) in zip(aws, man.iterrows()):
            r = recs[m["patient_id"]]
            ch = list(r["channels"])
            feats = window_features(aw.attacked, ch, r.get("fs", FS), core=CORE) if with_features else {}
            feats.update(record_id=m["patient_id"], window_id=m["window_id"], cls="attack",
                         cls2="attack", setting=setting, family=m["family"],
                         morphology=m["morphology"], level=int(m["level"]),
                         replay_source=str(m.get("replay_source", "") if pd.notna(m.get("replay_source", "")) else ""))
            rows.append(feats)
            core = aw.attacked[[ch.index("ECG"), ch.index("PPG")]]
            beat_rows.append(_beat_row(core[0], core[1], float(r.get("fs", FS)), B_floors,
                                       cfg.beat_detector))
    pm = pd.DataFrame(rows)
    pmB = pd.DataFrame(np.array(beat_rows, dtype=np.float32), columns=BEAT_COLS)
    split_map = df.drop_duplicates("record_id").set_index("record_id").split
    pm["split"] = pm.record_id.map(split_map).values
    host_map = pd.Series(df.index[(df.cls != "attack")], index=df.loc[df.cls != "attack", "window_id"].values)
    pm["host_row"] = host_map.reindex(pm.window_id).values.astype(int)
    pm["rec_true_alarm"] = df.loc[pm.host_row.values, "rec_true_alarm"].values
    pm["host_cls2"] = df.loc[pm.host_row.values, "cls2"].values
    return pm, pmB, len(hosts)


# --------------------------------------------------------------------------- #
# Step 7 -- discrimination variants
# --------------------------------------------------------------------------- #
def _prf(cm):
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1e-12)
    rec = tp / np.maximum(cm.sum(1), 1e-12)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1


def _weighted_cm(y, p, w, K):
    return np.bincount(y * K + p, weights=w, minlength=K * K).reshape(K, K)


def rf_variant(name, Xtr, ytr, Xte, yte, rte, classes, cfg, extra_eval=None, wtr=None):
    rf = RandomForestClassifier(**cfg.rf_params)
    rf.fit(Xtr, ytr, sample_weight=wtr)
    p = rf.predict(Xte)
    K = len(classes)
    cm = _weighted_cm(yte, p, np.ones(len(yte)), K)
    prec, rec, f1 = _prf(cm)
    res = dict(variant=name, classes=classes, confusion=cm.astype(int).tolist(),
               precision=prec.round(3).tolist(), recall=rec.round(3).tolist(),
               f1=f1.round(3).tolist(), macro_f1=float(f1.mean()),
               mcc=float(matthews_corrcoef(yte, p)), kappa=float(cohen_kappa_score(yte, p)),
               support=cm.sum(1).astype(int).tolist(), n_train=int(len(ytr)))
    # record-grouped bootstrap of per-class F1 and macro-F1
    recs = np.unique(rte)
    code = np.searchsorted(recs, rte)
    rng = np.random.default_rng(cfg.boot_seed)
    vals = []
    for _ in range(cfg.n_boot):
        w = rng.multinomial(recs.size, np.full(recs.size, 1 / recs.size))[code].astype(float)
        _, _, fb = _prf(_weighted_cm(yte, p, w, K))
        vals.append(np.r_[fb, fb.mean()])
    if vals:
        v = np.array(vals)
        lo, hi = np.nanpercentile(v, [2.5, 97.5], axis=0)
        res["f1_ci"] = [[round(a, 3), round(b, 3)] for a, b in zip(lo[:-1], hi[:-1])]
        res["macro_f1_ci"] = [round(lo[-1], 3), round(hi[-1], 3)]
    if extra_eval is not None:
        lab, Xe = extra_eval
        pe = rf.predict(Xe)
        res[f"predictions_on_{lab}"] = {classes[k]: int((pe == k).sum()) for k in range(K)}
    return res


def reslice_events(records, df, cfg, splits=("train", "val")):
    """Extra training events: post-onset windows of true-alarm records at a 1 s stride."""
    split_map = df.drop_duplicates("record_id").set_index("record_id").split.to_dict()
    rows = []
    for r in records:
        if not r["true_alarm"] or split_map.get(r["record_id"]) not in splits:
            continue
        fs = float(r.get("fs", FS))
        W, H = int(round(10 * fs)), int(round(5 * fs))
        S = max(1, int(round(cfg.event_reslice_stride_s * fs)))
        T = r["signal"].shape[1]
        onset = int(r["alarm_onset_sample"])
        amask = r.get("artifact_mask")
        for st in range(0, T - W + 1, S):
            if st % H == 0 or st + W <= onset:
                continue                       # existing window, or not post-onset
            if amask is not None and amask[:, st:st + W].any(axis=0).mean() > cfg.artifact_frac:
                continue
            f = window_features(r["signal"][:, st:st + W], list(r["channels"]), fs, core=CORE)
            f.update(record_id=r["record_id"])
            rows.append(f)
    return pd.DataFrame(rows)


def rq4_variants(df, feat_cols, pm, records, cfg):
    out = []
    tr = df.split.isin(["train", "val"]).values
    te = (df.split == "test").values
    c1 = ((df.cls == "attack") & (df.setting == "C1_PPG")).values
    cls2 = df.cls2.values
    Xf = df[feat_cols].values

    def build(sel_tr, sel_te, lab_fn, classes):
        ytr = np.array([classes.index(lab_fn(i)) for i in np.flatnonzero(sel_tr)])
        yte = np.array([classes.index(lab_fn(i)) for i in np.flatnonzero(sel_te)])
        return (Xf[sel_tr], ytr, Xf[sel_te], yte, df.rec_code.values[sel_te])

    # A: default forest (event / fault pooled / attack)
    fault = np.isin(cls2, ["artifact", "monitor_fa"])
    ev = cls2 == "event"
    selA = ev | fault | c1
    labA = lambda i: "event" if ev[i] else ("fault" if fault[i] else "attack")
    A = build(selA & tr, selA & te, labA, ["event", "fault", "attack"])
    out.append(rf_variant("A_published_replicate", *A, ["event", "fault", "attack"], cfg))
    # B: artifact-only fault class; monitor false alarms scored separately
    art = cls2 == "artifact"
    mon = cls2 == "monitor_fa"
    selB = ev | art | c1
    labB = lambda i: "event" if ev[i] else ("artifact" if art[i] else "attack")
    B_ = build(selB & tr, selB & te, labB, ["event", "artifact", "attack"])
    out.append(rf_variant("B_artifact_only", *B_, ["event", "artifact", "attack"], cfg,
                          extra_eval=("monitor_fa_test", Xf[mon & te])))
    # C: four classes
    selC = ev | art | mon | c1
    labC = lambda i: "event" if ev[i] else ("artifact" if art[i] else ("monitor_fa" if mon[i] else "attack"))
    C_ = build(selC & tr, selC & te, labC, ["event", "artifact", "monitor_fa", "attack"])
    out.append(rf_variant("C_four_class", *C_, ["event", "artifact", "monitor_fa", "attack"], cfg))
    # D: position-matched -- attacks injected into post-onset event / monitor-FA hosts
    if pm is not None and len(pm):
        for setting in pm.setting.unique():
            p = pm[pm.setting == setting]
            post = df.zone.values == "post"
            sel = (ev | mon | (art & post)) & (df.cls != "attack").values
            Xtr = np.vstack([Xf[sel & tr], p.loc[p.split.isin(["train", "val"]), feat_cols].values])
            ytr = np.r_[[0 if ev[i] else 1 for i in np.flatnonzero(sel & tr)],
                        np.full(int(p.split.isin(["train", "val"]).sum()), 2)]
            Xte = np.vstack([Xf[sel & te], p.loc[p.split == "test", feat_cols].values])
            yte = np.r_[[0 if ev[i] else 1 for i in np.flatnonzero(sel & te)],
                        np.full(int((p.split == "test").sum()), 2)]
            pm_rec = pd.Series(pd.factorize(pd.concat([df.record_id, p.record_id]))[0][len(df):])
            rte = np.r_[df.rec_code.values[sel & te], pm_rec.values[(p.split == "test").values]]
            out.append(rf_variant(f"D_position_matched_{SETTING_SHORT[setting]}",
                                  Xtr, ytr, Xte, yte, rte,
                                  ["event", "fault (post-onset)", "attack (post-onset hosts)"], cfg))
    # F: corrected labels -- alarm-zone windows (CinC event onset 290-300 s) join the
    #    event class (true alarms) or the monitor-false-alarm class (false alarms)
    zone = df.zone.values == "alarm_zone"
    clean = (df.cls != "attack").values
    tru = df.rec_true_alarm.fillna(False).values.astype(bool)
    okart = df.artifact_frac.fillna(0).values <= cfg.artifact_frac
    evF = ev | (clean & zone & tru & okart)
    monF = mon | (clean & zone & ~tru & okart)
    faultF = art | monF
    selF = evF | faultF | c1
    labF = lambda i: "event" if evF[i] else ("fault" if faultF[i] else "attack")
    F_ = build(selF & tr, selF & te, labF, ["event", "fault", "attack"])
    out.append(rf_variant("F_alarm_zone_relabelled", *F_, ["event", "fault", "attack"], cfg))
    # E: event re-slicing at 1 s inside training folds (test distribution unchanged)
    ex = reslice_events(records, df, cfg)
    if len(ex):
        Xtr, ytr, Xte, yte, rte = A
        Xtr2 = np.vstack([Xtr, ex[feat_cols].values])
        ytr2 = np.r_[ytr, np.zeros(len(ex), int)]
        out.append(rf_variant("E_event_reslice_1s", Xtr2, ytr2, Xte, yte, rte,
                              ["event", "fault", "attack"], cfg))
        out[-1]["n_extra_events"] = int(len(ex))
        cfg_nw = replace(cfg, rf_params={k: v for k, v in cfg.rf_params.items() if k != "class_weight"})
        out.append(rf_variant("E2_event_reslice_no_class_weight", Xtr2, ytr2, Xte, yte, rte,
                              ["event", "fault", "attack"], cfg_nw))
        out[-1]["n_extra_events"] = int(len(ex))
    return out


# --------------------------------------------------------------------------- #
# Step 8 -- alarm-level counterfactual oracle
# --------------------------------------------------------------------------- #
RATE_TYPES = {"asys": "Asystole", "brady": "Bradycardia", "tachy": "Tachycardia"}


def _alarm_kind(t):
    t = str(t).lower()
    if "vent" in t or "vfib" in t or "vtach" in t or "flutter" in t:
        return "ventricular"
    for k, v in RATE_TYPES.items():
        if k in t:
            return v
    return "other"


def alarm_oracle(df, G, pm, pmG, cfg):
    res = {}
    clean = (df.cls != "attack").values
    kind = df.rec_alarm_type.map(_alarm_kind).values
    # validation: the window ending at the alarm (it contains the CinC event onset)
    v = clean & df.ends_at_onset.fillna(False).values.astype(bool)
    val = pd.DataFrame({"alarm_type": kind[v], "true_alarm": df.rec_true_alarm.values[v],
                        "g_ecg": G.g_ecg.values[v], "g_fused": G.g_fused.values[v],
                        "g_ppg": G.g_ppg.values[v]})
    res["validation"] = val.groupby(["alarm_type", "true_alarm"])[["g_ecg", "g_ppg", "g_fused"]].agg(["mean", "size"])
    pre = clean & (df.zone.values == "pre") & (df.cls2.values == "normal")
    res["fire_rate_pre_zone_normals"] = G.loc[pre, ["g_ecg", "g_ppg", "g_fused"]].mean().to_dict()

    def effects(att_rows_G, host_G, meta, tag):
        rows = []
        for g in ("g_ecg", "g_ppg", "g_fused"):
            a = att_rows_G[g].values.astype(bool)
            h = host_G[g].values.astype(bool)
            d = meta.copy()
            d["family"] = np.where(d.morphology.values == "rate_warp", "rate_warp", d.family.values)
            d["a"], d["h"] = a, h
            for keys, gg in d.groupby(["setting", "family"]):
                no = ~gg.h
                yes = gg.h
                rows.append(dict(hosts=tag, monitor=g, setting=SETTING_SHORT.get(keys[0], keys[0]),
                                 family=keys[1], n=len(gg),
                                 n_host_no_alarm=int(no.sum()),
                                 creation=float((gg.a & no).sum() / max(no.sum(), 1)),
                                 n_host_alarm=int(yes.sum()),
                                 suppression=float((~gg.a & yes).sum() / max(yes.sum(), 1)) if yes.sum() else np.nan,
                                 n_true_alarm_host_alarm=int((yes & gg.true_alarm).sum()),
                                 suppression_true_alarm=(float((~gg.a & yes & gg.true_alarm).sum() / max((yes & gg.true_alarm).sum(), 1))
                                                         if (yes & gg.true_alarm).sum() else np.nan)))
            # by morphology for the families with shapes
            for keys, gg in d.groupby(["setting", "family", "morphology"]):
                no = ~gg.h
                rows.append(dict(hosts=tag, monitor=g, setting=SETTING_SHORT.get(keys[0], keys[0]),
                                 family=f"{keys[1]}:{keys[2]}", n=len(gg),
                                 n_host_no_alarm=int(no.sum()),
                                 creation=float((gg.a & no).sum() / max(no.sum(), 1)),
                                 n_host_alarm=int(gg.h.sum()),
                                 suppression=float((~gg.a & gg.h).sum() / max(gg.h.sum(), 1)) if gg.h.sum() else np.nan))
        return pd.DataFrame(rows)

    att = (df.cls == "attack").values & (df.host_row.values >= 0)
    meta = df.loc[att, ["setting", "family", "morphology"]].reset_index(drop=True)
    meta["true_alarm"] = df.rec_true_alarm.values[att].astype(bool)
    eff = [effects(G[att].reset_index(drop=True), G.iloc[df.host_row.values[att]].reset_index(drop=True),
                   meta, "pre-onset hosts (benchmark)")]
    if pm is not None and len(pm):
        m2 = pm[["setting", "family", "morphology"]].reset_index(drop=True)
        m2["true_alarm"] = pm.rec_true_alarm.values.astype(bool)
        eff.append(effects(pmG.reset_index(drop=True), G.iloc[pm.host_row.values].reset_index(drop=True),
                           m2, "post-onset hosts (position-matched)"))
    res["effects"] = pd.concat(eff, ignore_index=True)
    return res


# --------------------------------------------------------------------------- #
# Step 9 -- channel attribution baseline
# --------------------------------------------------------------------------- #
def attribution_baseline(df, feat_cols, cfg, mode):
    M = _masks(df, mode)
    out = []
    chan_scores = {}
    for ch in CORE:
        cols = [c for c in feat_cols if c.startswith(f"{ch}_")]
        iso = IsolationForest(n_estimators=200, random_state=0).fit(df.loc[M["train_norm"], cols].values)
        s = -iso.decision_function(df[cols].values)
        ref = np.sort(s[M["val_norm"]])
        chan_scores[ch] = np.searchsorted(ref, s) / max(ref.size, 1)      # percentile vs val normals
    te_att = M["test"] & M["att"] & df.setting.isin(["C1_PPG", "C1_ECG"]).values
    truth = np.where(df.setting.values == "C1_ECG", "ECG", "PPG")
    pred = np.where(chan_scores["ECG"] > chan_scores["PPG"], "ECG", "PPG")
    d = pd.DataFrame({"truth": truth[te_att], "pred": pred[te_att], "family": df.family.values[te_att],
                      "setting": df.setting.values[te_att]})
    d["hit"] = d.truth == d.pred
    out.append(dict(subset="all C1", n=len(d), AC1=float(d.hit.mean())))
    for fam, g in d.groupby("family"):
        out.append(dict(subset=fam, n=len(g), AC1=float(g.hit.mean())))
    for st, g in d.groupby("setting"):
        out.append(dict(subset=f"target={st}", n=len(g), AC1=float(g.hit.mean())))
    out.append(dict(subset="always-PPG reference", n=len(d), AC1=float((d.truth == "PPG").mean())))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Step 10 -- WUSTL-EHMS-2020 audit
# --------------------------------------------------------------------------- #
def wustl_audit(cfg):
    if not cfg.wustl_csv or not os.path.exists(cfg.wustl_csv):
        return None
    d = pd.read_csv(cfg.wustl_csv)
    d.columns = [c.strip() for c in d.columns]
    low = {c.lower(): c for c in d.columns}

    def pick(names):
        return [low[n.lower()] for n in names if n.lower() in low]
    label = low.get(cfg.wustl_label_col.lower())
    if label is None:
        raise ValueError(f"label column {cfg.wustl_label_col} not in {list(d.columns)}")
    y = (pd.to_numeric(d[label], errors="coerce").fillna(0) > 0).astype(int).values
    dup, ids, flags, bio = (pick(cfg.wustl_dup_cols), pick(cfg.wustl_id_cols),
                            pick(cfg.wustl_flag_cols), pick(cfg.wustl_bio_cols))
    ctr = pick(cfg.wustl_counter_cols)
    rfp = cfg.wustl_rf_params or cfg.rf_params
    base = [c for c in d.columns if c not in [label] + dup]
    net = [c for c in base if c not in bio]
    variants = {
        "all features (identifiers kept)": base,
        "identifiers + flags removed (published)": [c for c in base if c not in ids + flags],
        "identifiers removed, flags kept": [c for c in base if c not in ids],
        "identifiers + flags + packet counter removed": [c for c in base if c not in ids + flags + ctr],
        "network only (no identifiers, no flags)": [c for c in net if c not in ids + flags],
        "biometrics only": bio,
    }
    X_all = d.copy()
    for c in X_all.columns:
        if not pd.api.types.is_numeric_dtype(X_all[c]):
            X_all[c] = pd.factorize(X_all[c].astype(str))[0]
    rows, imps = [], {}
    for vname, cols in variants.items():
        if not cols:
            continue
        for seed in range(5):
            Xtr, Xte, ytr, yte = train_test_split(X_all[cols].values, y, test_size=0.3,
                                                  stratify=y, random_state=seed)
            rf = RandomForestClassifier(**{**rfp, "random_state": seed}).fit(Xtr, ytr)
            p = rf.predict(Xte)
            pr = rf.predict_proba(Xte)[:, 1]
            tp = ((p == 1) & (yte == 1)).sum()
            prec = tp / max((p == 1).sum(), 1)
            rec = tp / max((yte == 1).sum(), 1)
            f1a = 2 * prec * rec / max(prec + rec, 1e-12)
            tn = ((p == 0) & (yte == 0)).sum()
            p0 = tn / max((p == 0).sum(), 1)
            r0 = tn / max((yte == 0).sum(), 1)
            f10 = 2 * p0 * r0 / max(p0 + r0, 1e-12)
            pm = PairMetric(pr, np.flatnonzero(yte == 1), np.flatnonzero(yte == 0), np.zeros(len(yte), int))
            rows.append(dict(variant=vname, seed=seed, n_features=len(cols),
                             accuracy=float((p == yte).mean()), attack_precision=float(prec),
                             attack_recall=float(rec), attack_F1=float(f1a),
                             macro_F1=float((f1a + f10) / 2), AUPRC=pm.auprc()))
            if seed == 0:
                imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
                imps[vname] = dict(top=imp.head(6).round(3).to_dict(),
                                   identifier_share=float(imp.reindex(ids + flags).fillna(0).sum()))
    res = pd.DataFrame(rows).groupby("variant").agg(["mean", "std"]).drop(columns=["seed"], level=0)
    return dict(table=res, importances=imps,
                columns_found=dict(ids=ids, flags=flags, dup=dup, bio=bio, counter=ctr),
                n=int(len(d)), attack_rate=float(y.mean()))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_all(df, X, feat_cols, records, make_detectors: Callable[[int], Dict[str, object]],
            cfg: ExtrasConfig, steps: Sequence[int] = tuple(range(12))):
    os.makedirs(cfg.out_dir, exist_ok=True)
    T0 = time.perf_counter()
    res = {}
    df, meta = prepare(df, records, cfg)
    F = df[feat_cols].values
    _log(cfg, f"[0] {len(df)} rows | settings: {sorted(df.setting.unique())}")

    # 0. label audit -----------------------------------------------------------
    aud = label_audit(df, meta, cfg)
    res["label_audit"] = aud
    _log(cfg, "[0] label audit\n", aud["class_by_zone"].to_string(), "\n", aud["hosts_by_zone"].to_string())
    _log(cfg, f"    normals in the alarm zone (CinC event onset 290-300 s): "
              f"{aud['n_normal_in_alarm_zone']} of {aud['n_normal_v1']}; "
              f"hosts in the alarm zone of TRUE-alarm records: {aud['n_hosts_alarm_zone_true']} of {aud['n_hosts']}")

    # 1. score cache -------------------------------------------------------------
    train_mask = df["normal_v1"].values & (df.split.values == "train")
    if 1 in steps:
        _log(cfg, "[1] fitting detectors (resumable)")
        res["hyperparameters"] = build_score_cache(df, X, F, make_detectors, cfg, train_mask)

    # 2. beat pass ---------------------------------------------------------------
    bd = cfg.beat_detector
    sfx = "" if bd == "simple" else f"_{bd}"
    bpath = os.path.join(cfg.out_dir, f"beats{sfx}.parquet")
    floors_path = os.path.join(cfg.out_dir, "beat_floors.json")
    if os.path.exists(floors_path) and json.load(open(floors_path)).get("fingerprint") == _fingerprint(df):
        floors = tuple(json.load(open(floors_path))["floors"])
    else:
        floors = detector_floors(X, np.flatnonzero(train_mask))
        _save_json(dict(fingerprint=_fingerprint(df), floors=floors), floors_path)
    if os.path.exists(bpath) and len(pd.read_parquet(bpath, columns=["e_n"])) == len(df):
        B = pd.read_parquet(bpath)
    else:
        _log(cfg, f"[2] beat pass over all rows (detector: {bd})")
        t0 = time.perf_counter()
        B = beat_pass(X, floors, n_jobs=cfg.n_jobs, detector=bd)
        B.to_parquet(bpath)
        _log(cfg, f"    done in {time.perf_counter() - t0:.0f}s")
    G = alarm_flags(B)

    # 3. coherence detector --------------------------------------------------------
    sdir = os.path.join(cfg.out_dir, "scores")
    os.makedirs(sdir, exist_ok=True)
    coh_name = "Coherence" if bd == "simple" else "Coherence-NK"
    cpath = os.path.join(sdir, f"{coh_name}__seed{cfg.seeds[0]}.npy")
    coh = CoherenceDetector().fit(B[train_mask])
    np.save(cpath, coh.score(B).astype(np.float32))
    _save_json(dict(fingerprint=_fingerprint(df), fit_s=0.0, seed=cfg.seeds[0],
                    features=COH_FEATS, median=coh.med_.to_dict(), scale=coh.scale_.to_dict()),
               cpath.replace(".npy", ".json"))
    scores = load_scores(cfg, df)
    _log(cfg, f"[3] detectors with scores: { {k: sorted(v) for k, v in scores.items()} }")

    # 4. detection tables + CIs ----------------------------------------------------
    for mode in ("v1", "corrected"):
        long, summ, grid = detection_tables(df, scores, cfg, mode)
        long.to_csv(os.path.join(cfg.out_dir, f"detection_per_seed_{mode}.csv"), index=False)
        summ.to_csv(os.path.join(cfg.out_dir, f"detection_summary_{mode}.csv"))
        grid.to_csv(os.path.join(cfg.out_dir, f"grid_morph_level_{mode}.csv"), index=False)
        res[f"detection_{mode}"] = summ
        res[f"grid_{mode}"] = grid
    keys = (["AUPRC:C1-PPG|all", "AUPRC:C1-PPG|replay", "AUROC:C1-PPG|replay",
             "AUPRC:C1-PPG|fdi_spike", "AUPRC:C1-PPG|fdi_spike_L1",
             "faultAUROC:fault", "faultAUROC:artifact", "faultAUROC:monitor", "faultAUROC:event"] +
            [f"hmAUROC:{s}|all" for s in ("C1-PPG", "C1-ECG", "C2", "C3")] +
            ["hmAUROC:C1-PPG|replay", "hmAUROC:C3|replay", "hmAUROC:C3|rate_warp", "det5:C3|rate_warp"])
    for mode in ("v1", "corrected"):
        _log(cfg, f"[4] bootstrap CIs ({cfg.n_boot} record-grouped draws, labels={mode})")
        ci = bootstrap_table(df, scores, cfg, mode, keys)
        ci.to_csv(os.path.join(cfg.out_dir, f"bootstrap_ci_{mode}.csv"), index=False)
        res[f"bootstrap_ci_{mode}"] = ci
    res["bootstrap_ci"] = res[f"bootstrap_ci_{cfg.primary_labels}"]
    mode = cfg.primary_labels

    # 5. label sensitivity: refit IF / coherence on pre-zone normals ----------------
    sens = []
    tm_c = df["normal_corrected"].values & (df.split.values == "train")
    iso = IsolationForest(n_estimators=200, random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(F[tm_c])
    iso.fit(sc.transform(F[tm_c]))
    s_if = -iso.decision_function(sc.transform(F))
    coh_c = CoherenceDetector().fit(B[tm_c])
    for nm, s in (("IsolationForest(refit pre-zone)", s_if), ("Coherence(refit pre-zone)", coh_c.score(B))):
        l2, _, _ = detection_tables(df, {nm: {0: s}}, cfg, "corrected")
        sens.append(l2)
    hz = []
    for name, by_seed in scores.items():
        for seed, s in by_seed.items():
            for z in ("pre", "alarm_zone"):
                for st in SETTING_SHORT:
                    m = (df.cls == "attack").values & (df.setting.values == st) & \
                        (df.split.values == "test") & (df.host_zone.values == z)
                    if m.any():
                        pm_ = PairMetric(s, np.flatnonzero(m), _host_negs(df, m), df.rec_code.values)
                        hz.append(dict(detector=name, seed=seed, host_zone=z,
                                       setting=SETTING_SHORT[st], hmAUROC=pm_.auroc()))
    hz = pd.DataFrame(hz).groupby(["detector", "setting", "host_zone"]).hmAUROC.mean().unstack()
    hz.to_csv(os.path.join(cfg.out_dir, "label_sensitivity_host_zone.csv"))
    pd.concat(sens).to_csv(os.path.join(cfg.out_dir, "label_sensitivity_refit.csv"), index=False)
    res["host_zone_hmAUROC"] = hz

    # 6. position-matched generation -------------------------------------------------
    pm_path = os.path.join(cfg.out_dir, "position_matched_attacks.parquet")
    pmb_path = pm_path.replace(".parquet", f"_beats{sfx}.parquet")
    if os.path.exists(pm_path) and os.path.exists(pmb_path):
        pm = pd.read_parquet(pm_path)
        pmB = pd.read_parquet(pmb_path)
    elif os.path.exists(pm_path):
        _log(cfg, f"[6] regenerating post-onset attacks for {bd} beats (features reused)")
        t0 = time.perf_counter()
        pm = pd.read_parquet(pm_path)
        pm2, pmB, nh = generate_post_onset(df, records, cfg, floors, with_features=False)
        if len(pm2) != len(pm) or not (pm2.window_id.values == pm.window_id.values).all():
            raise RuntimeError("post-onset regeneration did not reproduce the cached windows")
        pmB.to_parquet(pmb_path)
        _log(cfg, f"    {nh} hosts, {len(pm)} attacked windows in {time.perf_counter() - t0:.0f}s")
    else:
        _log(cfg, f"[6] generating {list(cfg.pm_settings)} attacks on post-onset hosts")
        t0 = time.perf_counter()
        pm, pmB, nh = generate_post_onset(df, records, cfg, floors)
        pm.to_parquet(pm_path)
        pmB.to_parquet(pmb_path)
        _log(cfg, f"    {nh} hosts, {len(pm)} attacked windows in {time.perf_counter() - t0:.0f}s")
    pmG = alarm_flags(pmB)

    # 7. discrimination variants -------------------------------------------------------
    _log(cfg, "[7] discrimination variants")
    rq4 = rq4_variants(df, feat_cols, pm, records, cfg)
    _save_json(rq4, os.path.join(cfg.out_dir, "rq4_variants.json"))
    res["rq4"] = rq4

    # 8. alarm oracle --------------------------------------------------------------------
    _log(cfg, "[8] alarm-level counterfactual oracle")
    orc = alarm_oracle(df, G, pm, pmG, cfg)
    orc["validation"].to_csv(os.path.join(cfg.out_dir, f"alarm_oracle_validation{sfx}.csv"))
    orc["effects"].to_csv(os.path.join(cfg.out_dir, f"alarm_oracle_effects{sfx}.csv"), index=False)
    res["alarm_oracle"] = orc

    # 9. attribution ---------------------------------------------------------------------
    if cfg.run_attribution:
        at = attribution_baseline(df, feat_cols, cfg, mode)
        at.to_csv(os.path.join(cfg.out_dir, "attribution_baseline.csv"), index=False)
        res["attribution"] = at

    # 10. WUSTL ---------------------------------------------------------------------------
    w = wustl_audit(cfg)
    if w is not None:
        w["table"].to_csv(os.path.join(cfg.out_dir, "wustl_audit.csv"))
        _save_json({k: v for k, v in w.items() if k != "table"}, os.path.join(cfg.out_dir, "wustl_meta.json"))
    res["wustl"] = w

    # 11. hyperparameters + timing ---------------------------------------------------------
    hp_path = os.path.join(cfg.out_dir, "hyperparameters.json")
    hp = json.load(open(hp_path)) if os.path.exists(hp_path) else {}
    hp["RandomForest (RQ4, WUSTL)"] = cfg.rf_params
    hp["Coherence"] = dict(features=COH_FEATS, score="sum of positive robust z (train normals)",
                           ecg_detector="squared slope, 150 ms integration, 0.30 x 90th-percentile peak (floor 0.25 x median of training normals)",
                           ppg_detector="max upslope, 50 ms smoothing, 0.30 x 90th-percentile peak (same floor rule)")
    times = {}
    for fn in os.listdir(sdir):
        if fn.endswith(".json"):
            m = json.load(open(os.path.join(sdir, fn)))
            times.setdefault(fn.split("__seed")[0], []).append(m.get("fit_s", np.nan))
    hp["_fit_seconds_mean"] = {k: float(np.nanmean(v)) for k, v in times.items()}
    _save_json(hp, hp_path)
    res["hyperparameters"] = hp

    report(res, cfg)
    _log(cfg, f"\n[done] {time.perf_counter() - T0:.0f}s. Files in {cfg.out_dir}")
    return res


def _fmt(summ, det, key, d=3):
    try:
        m, s = summ.loc[det, (key, "mean")], summ.loc[det, (key, "std")]
    except KeyError:
        return "--"
    if not np.isfinite(m):
        return "--"
    return f"{m:.{d}f}" + (f"+-{s:.{d}f}" if np.isfinite(s) and s > 0 else "")


def report(res, cfg):
    """Print the headline numbers of every step."""
    mode = cfg.primary_labels
    S = res[f"detection_{mode}"]
    dets = list(S.index)
    P = print
    P("\n" + "=" * 78 + f"\nHEADLINE NUMBERS (labels: {mode})\n" + "=" * 78)
    P("\nTable III  AUPRC | AUPRC+fault | FAR normal/fault/event at val max-F1")
    for d in dets:
        P(f"  {d:28s}", " | ".join(_fmt(S, d, k) for k in
          ("T3_AUPRC", "T3_AUPRC_fault", "T3_FAR_normal_maxF1", "T3_FAR_fault_maxF1", "T3_FAR_event_maxF1")))
    P("\nTable IV  replay AUPRC | replay AUROC | spike | spike L1 | FDI sust | spoof sust")
    for d in dets:
        P(f"  {d:28s}", " | ".join(_fmt(S, d, k) for k in
          ("T4_AUPRC_replay", "T4_AUROC_replay", "T4_AUPRC_fdi_spike", "T4_AUPRC_fdi_spike_L1",
           "T4_AUPRC_fdi_sust", "T4_AUPRC_spoof_sust")))
    P("\nTable IV new  fault-vs-normal AUROC: pooled | artifact | monitor-FA | event")
    for d in dets:
        P(f"  {d:28s}", " | ".join(_fmt(S, d, f"T4_AUROC_{k}_vs_normal") for k in
          ("fault", "artifact", "monitor", "event")))
    P("\nTable IV equal-FAR (threshold at 5% / 10% FAR on TEST normals): fault FAR artifact | monitor | event")
    for d in dets:
        P(f"  {d:28s}", " | ".join(_fmt(S, d, f"T4_FAR_{k}_eq{q}") for q in (5, 10)
                                 for k in ("artifact", "monitor", "event")))
    P("\nTable IV at 10% VAL FAR: FAR normal | FAR fault | ratio | det replay | det FDI sust | det spoof sust")
    for d in dets:
        P(f"  {d:28s}", " | ".join(_fmt(S, d, k) for k in
          ("T4_FAR_normal_val10", "T4_FAR_fault_val10", "T4_ratio_val10", "T4_det10_replay",
           "T4_det10_fdi_sust", "T4_det10_spoof_sust")))
    P("\nTable V  host-matched AUROC: C1-PPG | C1-ECG | C2 | C3 | C1 replay | C3 joint replay | C3 warp | det warp | test FAR")
    for d in dets:
        P(f"  {d:28s}", " | ".join(_fmt(S, d, k) for k in
          ("T5_hmAUROC_C1-PPG|all", "T5_hmAUROC_C1-ECG|all", "T5_hmAUROC_C2|all", "T5_hmAUROC_C3|all",
           "T5_hmAUROC_C1-PPG|replay", "T5_hmAUROC_C3|replay", "T5_hmAUROC_C3|rate_warp",
           "T5_det5_C3|rate_warp", "T5_test_FAR")))
    P("\nTable V  AUROC vs ALL test normals: C1-PPG | C1-ECG | C2 | C3")
    for d in dets:
        P(f"  {d:28s}", " | ".join(_fmt(S, d, f"T5_AUROC_{s}|all") for s in ("C1-PPG", "C1-ECG", "C2", "C3")))
    P("\nBootstrap 95% CIs (record-grouped, seeds averaged inside each draw)")
    ci = res["bootstrap_ci"]
    for _, r in ci.iterrows():
        P(f"  {r.detector:28s} {r.metric:28s} {r.point:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
    P("\nLabel audit")
    a = res["label_audit"]
    P(f"  normals in alarm zone: {a['n_normal_in_alarm_zone']} / {a['n_normal_v1']}; "
      f"hosts in alarm zone of true-alarm records: {a['n_hosts_alarm_zone_true']} / {a['n_hosts']}")
    P("  host-matched AUROC by host zone:\n" + res["host_zone_hmAUROC"].round(3).to_string())
    Sc = res["detection_corrected"]
    P("  v1 vs corrected labels (C1-PPG AUPRC | replay AUPRC | hm C3 warp):")
    for d in dets:
        P(f"  {d:28s}", " | ".join(f"{_fmt(S, d, k)} -> {_fmt(Sc, d, k)}" for k in
          ("T3_AUPRC", "T4_AUPRC_replay", "T5_hmAUROC_C3|rate_warp")))
    P("\nRQ4 variants (per-class F1 [95% CI]; macro-F1)")
    for v in res["rq4"]:
        f1s = ", ".join(f"{c}={f:.3f}" + (f" [{lo:.3f},{hi:.3f}]" if "f1_ci" in v else "")
                        for c, f, (lo, hi) in zip(v["classes"], v["f1"], v.get("f1_ci", [[np.nan, np.nan]] * len(v["f1"]))))
        P(f"  {v['variant']:36s} {f1s}; macro {v['macro_f1']:.3f}; MCC {v['mcc']:.3f}")
        P(f"      confusion {v['confusion']}")
        if any(k.startswith("predictions_on_") for k in v):
            P("      " + str({k: v[k] for k in v if k.startswith("predictions_on_")}))
    P("\nAlarm oracle validation (window ending at the alarm; fire rate, n):")
    P(res["alarm_oracle"]["validation"].round(3).to_string())
    P("  fire rate on pre-zone normal windows: " + str({k: round(v, 3) for k, v in res["alarm_oracle"]["fire_rate_pre_zone_normals"].items()}))
    e = res["alarm_oracle"]["effects"]
    main = e[~e.family.str.contains(":")]
    P("\nAlarm creation / suppression by setting and family:")
    P(main.round(3).to_string(index=False))
    if "attribution" in res:
        P("\nChannel attribution (C1-PPG + C1-ECG test attacks):")
        P(res["attribution"].round(3).to_string(index=False))
    if res.get("wustl") is not None:
        P("\nWUSTL-EHMS-2020 (mean, sd over 5 split seeds):")
        P(res["wustl"]["table"].round(3).to_string())
        P("  columns found: " + str(res["wustl"]["columns_found"]))
        P("  importances: " + json.dumps(res["wustl"]["importances"], indent=1)[:1500])
    P("\nHyperparameters (Table II):")
    P(json.dumps(res["hyperparameters"], indent=1, default=str)[:4000])
