#!/usr/bin/env python3
"""
Reproduce the four-setting results reported in the IoMT-SecAlarmBench paper.

Builds the benchmark for all four attacker settings (C1-PPG, C1-ECG, C2, C3), then
runs the analyses in src/extras.py. Results are written to --out.

Stages (default: all, in this order):
  labels      write the labels tables to --labels-dir:
              secalarmbench_labels.csv (C1-PPG, every class) and
              secalarmbench_labels_multichannel.csv (C1-ECG, C2 and C3 attacks)
  by_setting  each detector trained once, all four settings scored at the same 5%
              validation false-alarm budget (rq1_by_setting_<detector>.csv)
  analyses    extras.run_all with the built-in beat detectors (Tables III-VIII,
              Fig. 2, alarm oracle, label audit, WUSTL audit)
  neurokit2   extras.run_all again with NeuroKit2 ECG and Elgendi PPG beat
              detection; detector fits are read from the cache
  breakdown   per-alarm-type detection and discrimination, and alarm creation and
              suppression by morphology

Usage:
    python scripts/reproduce_paper.py
    python scripts/reproduce_paper.py --stages analyses --seeds 0     # quick pass

The analyses stage takes about 35-45 minutes on one A100 with five seeds. Every
stage is resumable: finished detector fits and beat detections are skipped.
"""
from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ALL_STAGES = ("labels", "by_setting", "analyses", "neurokit2", "breakdown")
NEW_SETTINGS = ("C1_ECG", "C2_JOINT_INDEPENDENT", "C3_JOINT_COORDINATED")
SETTINGS = ("C1_PPG",) + NEW_SETTINGS

# Random forests of Table II: 300 trees for discrimination, 200 for the WUSTL audit.
RF = dict(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=20260724)
RF_WUSTL = dict(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=20260724)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cinc", default=str(ROOT / "data/cinc2015/training"),
                   help="folder with the CinC 2015 training records")
    p.add_argument("--wustl", default=str(ROOT / "data/wustl/wustl-ehms-2020.csv"),
                   help="WUSTL-EHMS-2020 CSV (the WUSTL audit is skipped if missing)")
    p.add_argument("--out", default=str(ROOT / "results/paper"),
                   help="output folder for results and caches")
    p.add_argument("--labels-dir", default=str(ROOT / "data/benchmark"),
                   help="output folder for the labels tables")
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help=f"comma-separated subset of {','.join(ALL_STAGES)}")
    p.add_argument("--seeds", default="0,1,2,3,4", help="detector seeds, e.g. 0 or 0,1,2,3,4")
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap draws for the CIs")
    p.add_argument("--epochs", type=int, default=40, help="training epochs of the deep detectors")
    a = p.parse_args(argv)
    a.stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = set(a.stages) - set(ALL_STAGES)
    if bad:
        p.error(f"unknown stage(s): {sorted(bad)}")
    a.seeds = tuple(int(s) for s in a.seeds.split(","))
    return a


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def write_labels(records, labels_dir):
    """Labels tables for the C1-PPG release and for the three new settings."""
    from cinc_adapter import assemble_benchmark, table2_composition
    from injection_generator import GeneratorConfig

    os.makedirs(labels_dir, exist_ok=True)
    table, _ = assemble_benchmark(records=records, cfg=GeneratorConfig(), max_hosts_per_record=2)
    table2_composition(table)
    table.to_csv(f"{labels_dir}/secalarmbench_labels.csv", index=False)

    tabs = []
    for s in NEW_SETTINGS:
        t, _ = assemble_benchmark(records=records, cfg=GeneratorConfig(),
                                  max_hosts_per_record=2, settings=(s,))
        tabs.append(t[t.cls == "attack"])
        gc.collect()
    table_mc = pd.concat(tabs, ignore_index=True)
    table_mc.to_csv(f"{labels_dir}/secalarmbench_labels_multichannel.csv", index=False)
    print(table_mc.groupby(["setting", "family"]).size())


def build_all_settings(records):
    """Features and windows for all four settings, built one setting at a time.

    Clean rows are kept once (from C1-PPG); the other settings add their attack rows.
    """
    from eval_harness import build_dataset, patient_wise_split
    from injection_generator import GeneratorConfig

    parts, wparts, feat_cols = [], [], None
    for i, s in enumerate(SETTINGS):
        d, feat_cols, w = build_dataset(records, GeneratorConfig(), return_windows=True, settings=(s,))
        keep = np.ones(len(d), bool) if i == 0 else (d.cls == "attack").values
        parts.append(d[keep])
        wparts.append(w[keep].astype(np.float32))
        del d, w
        gc.collect()
    df = patient_wise_split(pd.concat(parts, ignore_index=True))
    windows = np.concatenate(wparts)
    print(df.groupby(["setting", "cls"]).size(), windows.shape)
    return df, feat_cols, windows


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #

def make_detector_factory(epochs):
    from deep_detectors import (GDNDetector, LSTMAEDetector, OmniAnomalyDetector,
                                TranADDetector, USADDetector)
    from eval_harness import IsolationForestDetector

    def _make(cls, seed, **kw):
        try:
            return cls(seed=seed, **kw)
        except TypeError:
            return cls(**kw)

    def make_detectors(seed):
        return {
            "IsolationForest": IsolationForestDetector(seed=seed),
            "USAD": _make(USADDetector, seed, epochs=epochs),
            "LSTM-AE": _make(LSTMAEDetector, seed, epochs=epochs),
            "GDN": _make(GDNDetector, seed, epochs=epochs),
            "TranAD": _make(TranADDetector, seed, epochs=epochs),
            "OmniAnomaly": _make(OmniAnomalyDetector, seed, epochs=epochs),
        }
    return make_detectors


def run_by_setting(df, feat_cols, windows, out_dir, epochs):
    """Each detector trained once; every setting scored at the same 5% validation FAR."""
    from deep_detectors import (GDNDetector, LSTMAEDetector, OmniAnomalyDetector,
                                TranADDetector, USADDetector)
    from eval_harness import IsolationForestDetector, run_rq1_by_setting

    os.makedirs(out_dir, exist_ok=True)
    run_rq1_by_setting(df, df[feat_cols].values, IsolationForestDetector(), out_dir=out_dir)
    for det in (USADDetector(epochs=epochs), LSTMAEDetector(epochs=epochs), GDNDetector(epochs=epochs),
                TranADDetector(epochs=epochs), OmniAnomalyDetector(epochs=epochs)):
        run_rq1_by_setting(df, windows, det, out_dir=out_dir)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def run_analyses(df, feat_cols, windows, records, args, beat_detector):
    import extras

    cfg = extras.ExtrasConfig(
        out_dir=args.out,
        seeds=args.seeds,
        n_boot=args.n_boot,
        beat_detector=beat_detector,
        rf_params=RF,
        wustl_rf_params=RF_WUSTL,
        wustl_csv=args.wustl if os.path.exists(args.wustl) else None,
    )
    if cfg.wustl_csv is None:
        print(f"[note] {args.wustl} not found; the WUSTL audit is skipped")
    return extras.run_all(df, windows, feat_cols, records, make_detector_factory(args.epochs), cfg)


def run_breakdown(df, feat_cols, records, out_dir):
    """Per-alarm-type results and alarm effects by morphology (cached scores; no new fits)."""
    import extras
    from sklearn.ensemble import RandomForestClassifier

    cfg = extras.ExtrasConfig(out_dir=out_dir)
    D, _ = extras.prepare(df, records, cfg)
    S = extras.load_scores(cfg, D)
    te = (D.split == "test").values
    va = (D.split == "val").values
    ev = (D.cls2 == "event").values
    norm = D.normal_v1.values
    kind = D.rec_alarm_type.astype(str).values

    # (1) detectors: genuine-event windows against normal windows, by alarm type
    rows = []
    for name, by_seed in S.items():
        for seed, s in by_seed.items():
            thr = np.quantile(s[norm & va], 0.95)            # 5% validation budget, as Table V
            neg = s[norm & te]
            for t in sorted(set(kind[ev & te])):
                pos = s[ev & te & (kind == t)]
                n = len(pos)
                pm = extras.PairMetric(np.r_[pos, neg], np.arange(n), np.arange(n, n + len(neg)),
                                       np.zeros(n + len(neg), int))
                rows.append(dict(alarm_type=t, detector=name, n=n,
                                 AUROC_vs_normal=pm.auroc(), flagged_at_5pct=float((pos >= thr).mean())))
    det = (pd.DataFrame(rows).groupby(["alarm_type", "detector"])
           [["n", "AUROC_vs_normal", "flagged_at_5pct"]].mean().round(3))
    det.to_csv(f"{out_dir}/alarm_type_detection.csv")
    print("=== Event windows vs normal windows, by alarm type (test, mean over seeds) ===")
    print(det.to_string())

    # (2) three-way forest: what the test event windows are called, by alarm type
    c1 = ((D.cls == "attack") & (D.setting == "C1_PPG")).values
    fault = D.cls2.isin(["artifact", "monitor_fa"]).values
    sel = ev | fault | c1
    tr = D.split.isin(["train", "val"]).values
    y = np.where(ev, 0, np.where(fault, 1, 2))
    rf = RandomForestClassifier(**RF).fit(D.loc[sel & tr, feat_cols].values, y[sel & tr])
    pred = np.array(["event", "fault", "attack"])[rf.predict(D.loc[ev & te, feat_cols].values)]
    ct = pd.crosstab(pd.Series(kind[ev & te], name="alarm_type"), pd.Series(pred, name="predicted"))
    ct = ct.reindex(columns=["event", "fault", "attack"], fill_value=0).assign(n=lambda t: t.sum(1))
    ct.to_csv(f"{out_dir}/alarm_type_predictions.csv")
    print("\n=== Test event windows by alarm type and predicted class (default forest) ===")
    print(ct.to_string())

    # (3) alarm creation and suppression by morphology (built-in beat detectors)
    E = pd.read_csv(f"{out_dir}/alarm_oracle_effects.csv")
    E = E[E.family.str.contains(":") & E.monitor.isin(["g_ecg", "g_fused"])].copy()
    E["hosts"] = np.where(E.hosts.str.startswith("pre"), "pre-alarm", "post-alarm")
    creation = E[E.hosts == "pre-alarm"].pivot_table(index=["setting", "family"], columns="monitor",
                                                     values="creation")
    suppression = E[E.hosts == "post-alarm"].pivot_table(index=["setting", "family"], columns="monitor",
                                                         values="suppression")
    out = pd.concat({"creation (pre-alarm hosts)": creation,
                     "suppression (post-alarm hosts)": suppression}, axis=1)
    out.round(3).to_csv(f"{out_dir}/alarm_effects_by_morphology.csv")
    print("\n=== Alarm creation and suppression by morphology ===")
    print(out.round(3).to_string())


# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    from cinc_loader import load_cinc_records

    records = load_cinc_records(args.cinc)
    if "labels" in args.stages:
        write_labels(records, args.labels_dir)
    if set(args.stages) - {"labels"}:
        df, feat_cols, windows = build_all_settings(records)
        if "by_setting" in args.stages:
            run_by_setting(df, feat_cols, windows, args.out, args.epochs)
        if "analyses" in args.stages:
            run_analyses(df, feat_cols, windows, records, args, beat_detector="simple")
        if "neurokit2" in args.stages:
            run_analyses(df, feat_cols, windows, records, args, beat_detector="neurokit2")
        if "breakdown" in args.stages:
            run_breakdown(df, feat_cols, records, args.out)


if __name__ == "__main__":
    main()
