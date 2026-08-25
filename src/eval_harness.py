"""
IoMT-SecAlarmBench — Evaluation Harness
=======================================

The shared substrate every baseline plugs into. It:
  1. rebuilds the windowed dataset (clean + attacked signal arrays) from the loaded
     records, deterministically matching secalarmbench_labels.csv;
  2. extracts per-window features, INCLUDING the ECG-PPG cross-channel coupling that
     integrity attacks break (the benchmark's discriminative signal);
  3. makes a LEAK-FREE patient-wise train/val/test split (whole records to one side,
     stratified by arrhythmia type);
  4. exposes a uniform Detector interface (fit / score) so any RQ1 detector is a thin
     adapter — Isolation Forest is the first;
  5. scores RQ1 (AUPRC / F1, un-adjusted), RQ4 (per-morphology stealthiness), and
     RQ5 (latency), and writes results to disk for the paper tables.

Usage (Colab):
    from cinc_loader import load_cinc_records
    from injection_generator import GeneratorConfig
    from eval_harness import build_dataset, patient_wise_split, IsolationForestDetector, run_rq1

    records = load_cinc_records(f"{REPO}/data/cinc2015/training")
    df, feat_cols = build_dataset(records, GeneratorConfig())
    df = patient_wise_split(df)
    res = run_rq1(df, df[feat_cols].values, IsolationForestDetector(), out_dir=REPO)
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

from cinc_adapter import build_windows
from injection_generator import GeneratorConfig, generate_benchmark

FS = 125.0


# --------------------------------------------------------------------------- #
# 1. Feature extraction (per window [C, T])
# --------------------------------------------------------------------------- #

def _channel_features(x, fs):
    """~10 robust features for one channel."""
    d = np.diff(x) if x.size > 1 else np.array([0.0])
    rfft = np.abs(np.fft.rfft(x - x.mean())) if x.size > 1 else np.array([0.0])
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs) if x.size > 1 else np.array([0.0])
    dom = float(freqs[int(np.argmax(rfft))]) if rfft.size and rfft.sum() > 0 else 0.0
    p = rfft / (rfft.sum() + 1e-12)
    spec_ent = float(-(p * np.log(p + 1e-12)).sum())
    with np.errstate(all="ignore"):
        sk = float(skew(x)) if np.std(x) > 1e-9 else 0.0
        ku = float(kurtosis(x)) if np.std(x) > 1e-9 else 0.0
    return {
        "mean": float(x.mean()), "std": float(x.std()), "ptp": float(np.ptp(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))), "madiff": float(np.mean(np.abs(d))),
        "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
        "skew": sk, "kurt": ku, "dom_freq": dom, "spec_ent": spec_ent,
    }


def window_features(win, channels, fs=FS, core=("ECG", "PPG")):
    """Per-channel features for the CORE cluster only (present in every record), plus
    the core cross-channel coupling. Restricting to core channels guarantees a fixed,
    rectangular, NaN-free feature vector across records with different enrichment tiers.
    """
    feats = {}
    idx = {ch: channels.index(ch) for ch in core if ch in channels}
    for ch in core:
        ci = idx.get(ch)
        x = win[ci] if ci is not None else np.zeros(win.shape[1])
        for k, v in _channel_features(x, fs).items():
            feats[f"{ch}_{k}"] = v
    # cross-channel coupling between the two core channels (the discriminator)
    if len(core) >= 2 and all(c in idx for c in core[:2]):
        a, b = win[idx[core[0]]], win[idx[core[1]]]
        if a.std() > 1e-9 and b.std() > 1e-9:
            feats["xcorr_01"] = float(np.corrcoef(a, b)[0, 1])
            L = int(0.2 * fs); best = 0.0
            for lag in range(-L, L + 1, max(1, L // 3)):
                c = abs(float(np.corrcoef(a, np.roll(b, lag))[0, 1]))
                best = max(best, c if np.isfinite(c) else 0.0)
            feats["xcorr_lag_best"] = best
        else:
            feats["xcorr_01"] = 0.0
            feats["xcorr_lag_best"] = 0.0
    return feats


# --------------------------------------------------------------------------- #
# 2. Dataset reconstruction (clean + attacked arrays -> feature table)
# --------------------------------------------------------------------------- #

def build_dataset(records, cfg=None, target_channels=("PPG",), win_sec=10.0,
                  hop_sec=5.0, max_hosts_per_record=2, artifact_frac=0.10,
                  core_channels=("ECG", "PPG"), return_windows=False, verbose=True):
    """Rebuild the benchmark windows (deterministic) and return (feature_df, feature_cols).

    If return_windows=True, also returns a raw core-channel tensor `windows` of shape
    [N, len(core_channels), T] aligned row-for-row with the feature_df — the input deep
    detectors (USAD, LSTM-AE, TranAD, ...) consume. Features are extracted from
    core_channels only, so both the matrix and the tensor are rectangular across
    enrichment tiers. Clean-window arrays are re-sliced from records via the window_id
    index; attack arrays come from the generator. Matches secalarmbench_labels.csv.
    """
    cfg = cfg or GeneratorConfig()
    W = int(round(win_sec * FS)); H = int(round(hop_sec * FS))
    recs_by_id = {r["record_id"]: r for r in records}
    wins = [] if return_windows else None

    def _core(win, chans):
        idx = [chans.index(c) for c in core_channels if c in chans]
        return win[idx]

    hosts, banks, clean_rows, host_labels = build_windows(
        records, win_sec=win_sec, hop_sec=hop_sec,
        max_hosts_per_record=max_hosts_per_record, artifact_frac=artifact_frac)
    attacked, manifest = generate_benchmark(
        hosts, target_channels=list(target_channels), cfg=cfg, replay_banks=banks)

    rows = []
    # --- clean windows: re-slice arrays from records by window_id ---
    for cr in clean_rows:
        rid, widx = cr["window_id"].rsplit("_w", 1)
        rec = recs_by_id[rid]
        start = int(widx) * H
        win = rec["signal"][:, start:start + W]
        if win.shape[1] < W:
            continue
        feats = window_features(win, rec["channels"], rec.get("fs", FS), core=core_channels)
        feats.update(record_id=rid, patient_id=rid, window_id=cr["window_id"],
                     cls=cr["cls"], family="none", morphology="none",
                     level=-1, target_channel="", alarm_type=cr["alarm_type"],
                     true_alarm=cr["true_alarm"])
        rows.append(feats)
        if return_windows:
            wins.append(_core(win, rec["channels"]))

    # --- attack windows: arrays from the generator (order matches manifest) ---
    for i, (_, m) in enumerate(manifest.iterrows()):
        aw = attacked[i]
        rid = m["patient_id"]
        rec = recs_by_id[rid]
        feats = window_features(aw.attacked, rec["channels"], rec.get("fs", FS), core=core_channels)
        hl = host_labels.get(m["window_id"], {})
        feats.update(record_id=rid, patient_id=rid, window_id=m["window_id"],
                     cls="attack", family=m["family"], morphology=m["morphology"],
                     level=int(m["level"]), target_channel=m["target_channel"],
                     replay_source=(m["replay_source"] if "replay_source" in m
                                    and pd.notna(m["replay_source"]) else ""),
                     alarm_type=hl.get("alarm_type", ""), true_alarm=hl.get("true_alarm", pd.NA))
        rows.append(feats)
        if return_windows:
            wins.append(_core(aw.attacked, rec["channels"]))

    df = pd.DataFrame(rows)
    meta_cols = ["record_id", "patient_id", "window_id", "cls", "family",
                 "morphology", "level", "target_channel", "replay_source",
                 "alarm_type", "true_alarm"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    n_nan = int(df[feat_cols].isna().sum().sum())
    if n_nan:
        bad = df[feat_cols].isna().sum()
        raise ValueError(f"Feature matrix has {n_nan} NaNs (ragged channels?). "
                         f"Offending columns:\n{bad[bad > 0].to_string()}")
    if verbose:
        print(f"[dataset] {len(df)} windows | {len(feat_cols)} features | NaNs: {n_nan}")
        print(df["cls"].value_counts().to_string())
    if return_windows:
        windows = np.stack(wins)                 # [N, C_core, T], aligned with df rows
        if verbose:
            print(f"[dataset] window tensor: {windows.shape} (aligned row-for-row with df)")
        return df, feat_cols, windows
    return df, feat_cols


# --------------------------------------------------------------------------- #
# 3. Leak-free patient-wise split (stratified by arrhythmia type)
# --------------------------------------------------------------------------- #

def patient_wise_split(df, ratios=(0.7, 0.15, 0.15), stratify="alarm_type", seed=20260723):
    rng = np.random.default_rng(seed)
    rec_df = df.drop_duplicates("record_id")[["record_id", stratify]]
    split_map = {}
    for _, grp in rec_df.groupby(stratify):
        recs = np.array(grp["record_id"].tolist(), dtype=object)
        rng.shuffle(recs)
        n = len(recs); n_tr = int(ratios[0] * n); n_va = int(ratios[1] * n)
        for r in recs[:n_tr]:            split_map[r] = "train"
        for r in recs[n_tr:n_tr + n_va]: split_map[r] = "val"
        for r in recs[n_tr + n_va:]:     split_map[r] = "test"
    df["split"] = df["record_id"].map(split_map)
    # leakage guard
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                ra = set(df[df.split == a]["record_id"]); rb = set(df[df.split == b]["record_id"])
                assert not (ra & rb), f"leakage between {a}/{b}"
    return df


# --------------------------------------------------------------------------- #
# 4. Detector interface + Isolation Forest adapter
# --------------------------------------------------------------------------- #

class Detector:
    name = "base"
    def fit(self, X): ...
    def score(self, X): ...   # higher = more anomalous


class IsolationForestDetector(Detector):
    name = "IsolationForest"
    def __init__(self, n_estimators=200, seed=20260723):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)
        self.scaler = StandardScaler()
    def fit(self, X):
        self.model.fit(self.scaler.fit_transform(X)); return self
    def score(self, X):
        return -self.model.decision_function(self.scaler.transform(X))


# --------------------------------------------------------------------------- #
# 5. RQ1 detection (+ RQ4 per-morphology, RQ5 latency)
# --------------------------------------------------------------------------- #

def _best_f1_threshold(y, s):
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(s[order][int(np.argmax(f1))])


def run_rq1(df, X, detector, out_dir=None):
    """Fit on TRAIN physiology_normal only (unsupervised, leak-free); evaluate on TEST.

    X is the input aligned row-for-row with df: the feature matrix (df[feat_cols].values)
    for classical detectors, or the [N, C, T] window tensor (from build_dataset with
    return_windows=True) for deep detectors. Positives = attack, negatives =
    physiology_normal. Reports AUPRC/AUROC + F1 at a val-selected operating point, plus
    per-morphology stealthiness (RQ4) and latency (RQ5).
    """
    df = df.reset_index(drop=True).copy()
    X = np.asarray(X)
    tr_mask = ((df.split == "train") & (df.cls == "physiology_normal")).values
    Xtr = X[tr_mask]
    if len(Xtr) == 0:
        raise ValueError("No TRAIN physiology_normal windows to fit on — check the split "
                         "(each arrhythmia-type stratum needs enough records).")
    t0 = time.perf_counter(); detector.fit(Xtr); fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    df["score"] = np.asarray(detector.score(X)).ravel()
    score_s = time.perf_counter() - t0
    lat_ms = 1000 * score_s / len(df)

    def _slice(split):
        d = df[(df.split == split) & (df.cls.isin(["attack", "physiology_normal"]))]
        return (d.cls == "attack").astype(int).values, d["score"].values, d

    yv, sv, _ = _slice("val")
    yte, ste, dte = _slice("test")
    train_normal_scores = df[(df.split == "train") & (df.cls == "physiology_normal")]["score"].values
    if len(yv) and yv.sum() and (yv == 0).any():
        thr = _best_f1_threshold(yv, sv)                       # val-selected operating point
        op_source = "val_maxF1"
    else:                                                       # fallback: 5% contamination on train normals
        thr = float(np.quantile(train_normal_scores, 0.95))
        op_source = "train_contamination_5pct"

    auprc = float(average_precision_score(yte, ste))
    auroc = float(roc_auc_score(yte, ste))
    pred = (ste >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)

    # RQ4: stealthiness curves. Split by regime so the stealthy (FDI) case isn't
    # averaged together with the easy (spoofing) case, and pull replay onto its own
    # line (it has no severity levels, so it never belonged in the level grid).
    dte = dte.copy(); dte["pred"] = pred
    atk = dte[dte.cls == "attack"].copy()

    def _grid(fam):
        g = atk[atk.family == fam]
        if not len(g):
            return pd.DataFrame()
        return (g.pivot_table(index="morphology", columns="level", values="pred",
                              aggfunc="mean").round(3))

    rq4 = {"fdi": _grid("fdi"), "spoofing": _grid("spoofing")}
    rep = atk[atk.family == "replay"]
    if len(rep):
        by_src = rep.groupby("replay_source")["pred"].mean().round(3)
        by_src["overall"] = round(float(rep["pred"].mean()), 3)
        rq4["replay"] = by_src
    else:
        rq4["replay"] = pd.Series(dtype=float)

    metrics = dict(detector=detector.name, n_test=int(len(yte)), test_attack=int(yte.sum()),
                   AUPRC=round(auprc, 4), AUROC=round(auroc, 4),
                   precision=round(float(prec), 4), recall=round(float(rec), 4),
                   F1=round(float(f1), 4), op_threshold=round(float(thr), 4), op_source=op_source,
                   fit_s=round(fit_s, 2), latency_ms_per_window=round(lat_ms, 4))

    print("\n=== RQ1 Detection (test, un-adjusted; attack vs physiology_normal) ===")
    for k, v in metrics.items():
        print(f"  {k:24s} {v}")
    print("\n=== RQ4 stealthiness — FDI detection rate by morphology × level ===")
    print(rq4["fdi"].to_string() if len(rq4["fdi"]) else "  (none)")
    print("\n=== RQ4 — spoofing detection rate by morphology × level (easy-case contrast) ===")
    print(rq4["spoofing"].to_string() if len(rq4["spoofing"]) else "  (none)")
    print("\n=== RQ4 — replay detection rate (mechanism-preserving; no severity levels) ===")
    print(rq4["replay"].to_string() if len(rq4["replay"]) else "  (none)")

    if out_dir:
        pd.DataFrame([metrics]).to_csv(f"{out_dir}/rq1_{detector.name}.csv", index=False)
        if len(rq4["fdi"]):
            rq4["fdi"].to_csv(f"{out_dir}/rq4_fdi_{detector.name}.csv")
        if len(rq4["spoofing"]):
            rq4["spoofing"].to_csv(f"{out_dir}/rq4_spoofing_{detector.name}.csv")
        if len(rq4["replay"]):
            rq4["replay"].to_csv(f"{out_dir}/rq4_replay_{detector.name}.csv", header=["detection_rate"])
        print(f"\n[saved] rq1_{detector.name}.csv + rq4_fdi/spoofing/replay_{detector.name}.csv")
    return metrics, rq4


# --------------------------------------------------------------------------- #
def _demo():
    # richer stub: several records per alarm type so the stratified split works
    rng = np.random.default_rng(0)
    specs = [("Asystole", False), ("Bradycardia", True), ("Tachycardia", True),
             ("VentricularTachycardia", False), ("VentricularFlutterFib", False)]
    recs = []
    for ti, (atype, is_true) in enumerate(specs):
        for k in range(6):                              # 6 records/type -> 30 records
            T = int(FS * 300); t = np.arange(T) / FS
            ph = 2 * np.pi * (70 + 3 * k) / 60.0 * t
            ecg = np.sin(ph) + 0.02 * rng.standard_normal(T)
            ppg = 0.9 * np.sin(ph - 0.6) + 0.02 * rng.standard_normal(T)
            sig = np.vstack([ecg, ppg])
            sig = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
            recs.append(dict(record_id=f"{atype[:3]}{k}", signal=sig, channels=["ECG", "PPG"],
                             alarm_type=atype, true_alarm=is_true,
                             alarm_onset_sample=int(T * 0.85), fs=FS))
    df, feat_cols = build_dataset(recs, GeneratorConfig(), verbose=True)
    df = patient_wise_split(df)
    print("\nsplit sizes (rows):", df["split"].value_counts().to_dict())
    print("split records:", df.drop_duplicates("record_id")["split"].value_counts().to_dict())
    run_rq1(df, df[feat_cols].values, IsolationForestDetector())


if __name__ == "__main__":
    _demo()
