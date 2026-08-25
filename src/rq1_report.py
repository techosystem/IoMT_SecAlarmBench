"""
IoMT-SecAlarmBench — rq1_report.py
==================================

Drop-in replacement for `eval_harness.run_rq1` that fixes three reporting defects in
the published Table 3-R. It does NOT touch eval_harness (build_dataset,
patient_wise_split and the Detector interface are unchanged and still authoritative),
so the old path stays runnable for comparison.

WHAT IT ADDS, and why each one matters
--------------------------------------

1. THE FAULT-ARTIFACT NEGATIVE CLASS.
   The published run scored `attack` against `physiology_normal` only, so the 651
   fault_artifact windows and 92 physiology_event windows in the test split were never
   scored. A detector that flags every sensor artifact as an attack was never charged
   for it -- the most clinically important false positive, and the benchmark's whole
   premise. We now report, at the SAME operating point:
     * AUPRC/F1 vs physiology_normal            (as published, for comparability)
     * AUPRC/F1 vs physiology_normal + fault    (the clinically honest version)
     * AUPRC/F1 vs all clean classes
     * per-clean-class false-alarm rate at the operating threshold
   No retraining is needed: the scores already exist for every row.

2. THE NO-SKILL FLOOR.
   AUPRC's chance baseline is the positive prevalence. Because the factorial grid emits
   34 attack variants per host window, attack is ~54% of the attack-vs-normal
   evaluation set, so an AUPRC of 0.90 is a lift of ~0.36 over chance -- not what the
   bare number suggests. Prevalence and floor are now reported beside every AUPRC, plus
   an optional prevalence-matched AUPRC (positives subsampled to a realistic base rate).

3. RECORD-LEVEL BOOTSTRAP CONFIDENCE INTERVALS.
   Every test attack window is one of 34 variants of one of ~188 distinct clean host
   windows drawn from ~100 records. Window-level point estimates therefore report false
   precision. All headline metrics get a percentile CI from resampling RECORDS (the
   independent unit, and the unit the split respects) with replacement. The effective
   sample sizes (records, host windows) are printed so the reader can see them.

Also reports the RQ4 frontier (replay overall, FDI-spike) with the same record-level CI,
and per-cell counts on the RQ4 grids so no cell's rate is read without its n.

Usage
-----
    from eval_harness import build_dataset, patient_wise_split, IsolationForestDetector
    from rq1_report import run_rq1

    df, feat_cols, windows = build_dataset(records, cfg, return_windows=True)
    df = patient_wise_split(df)
    res = run_rq1(df, df[feat_cols].values, IsolationForestDetector(), out_dir="results/")

`res` is a dict; `res["metrics"]` is one flat row ready to concatenate into Table 3-R.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CLEAN_CLASSES = ("physiology_normal", "fault_artifact", "physiology_event")

# Negative-class definitions, in increasing order of clinical honesty.
NEGATIVE_SETS = {
    "vs_normal": ("physiology_normal",),
    "vs_normal_fault": ("physiology_normal", "fault_artifact"),
    "vs_all_clean": ("physiology_normal", "fault_artifact", "physiology_event"),
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _best_f1_threshold(y, s):
    """Threshold maximising F1 on (y, s). Same rule as the published harness."""
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(s[order][int(np.argmax(f1))])


def _prf(y, pred):
    """precision, recall, F1 for the positive class, without sklearn's overhead."""
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _auprc(y, s):
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def _auroc(y, s):
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y, s))


# --------------------------------------------------------------------------- #
# record-level bootstrap
# --------------------------------------------------------------------------- #

def _record_index(sub, key="record_id"):
    """Map each record to its positional row indices within `sub` (0..len(sub)-1)."""
    codes, uniques = pd.factorize(sub[key].values)
    return {u: np.where(codes == i)[0] for i, u in enumerate(uniques)}


def bootstrap_ci(sub, stat_fn, n_boot=2000, ci=95, seed=0, key="record_id"):
    """Percentile CI for stat_fn(sub_rows) under resampling of RECORDS with replacement.

    stat_fn receives a positional index array into `sub` and returns a float (or nan to
    signal "undefined for this draw", which is dropped).
    """
    if n_boot is None or n_boot <= 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    by_rec = _record_index(sub, key)
    recs = np.array(list(by_rec.keys()), dtype=object)
    if len(recs) < 2:
        return float("nan"), float("nan"), 0
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(recs, size=len(recs), replace=True)
        idx = np.concatenate([by_rec[r] for r in pick])
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            draws.append(v)
    if len(draws) < max(20, n_boot // 20):
        return float("nan"), float("nan"), len(draws)
    lo, hi = np.percentile(draws, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi), len(draws)


def _fmt_ci(lo, hi):
    return "[nan, nan]" if not np.isfinite(lo) else f"[{lo:.3f}, {hi:.3f}]"


# --------------------------------------------------------------------------- #
# prevalence-matched AUPRC
# --------------------------------------------------------------------------- #

def prevalence_matched_auprc(y, s, target=0.05, n_rep=200, seed=0):
    """AUPRC with positives subsampled so prevalence == `target`, negatives kept whole.

    The released factorial grid puts attack at ~54% of the evaluation set, which is a
    generation artefact rather than any defensible deployment base rate. This reports
    what AUPRC looks like at a plausible one. Returns (mean, sd, achieved_prevalence).
    """
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan")
    # n_pos / (n_pos + n_neg) = target
    n_pos = int(round(target * len(neg) / (1.0 - target)))
    if n_pos < 10 or n_pos >= len(pos):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        take = rng.choice(pos, size=n_pos, replace=False)
        idx = np.concatenate([take, neg])
        vals.append(_auprc(y[idx], s[idx]))
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals)), n_pos / (n_pos + len(neg))


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #

def run_rq1(df, X, detector, out_dir=None, n_boot=2000, ci=95,
            target_prevalence=0.05, n_prev_rep=200, seed=20260729,
            verbose=True, save_prefix=None):
    """Fit on TRAIN physiology_normal only; evaluate on TEST with expanded negatives,
    prevalence floors and record-level CIs. Returns a dict of results."""
    df = df.reset_index(drop=True).copy()
    X = np.asarray(X)
    name = save_prefix or detector.name

    # ---------------- fit + score ----------------
    tr_mask = ((df.split == "train") & (df.cls == "physiology_normal")).values
    if tr_mask.sum() == 0:
        raise ValueError("No TRAIN physiology_normal windows to fit on -- check the split.")
    t0 = time.perf_counter()
    detector.fit(X[tr_mask])
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    df["score"] = np.asarray(detector.score(X)).ravel().astype(float)
    score_s = time.perf_counter() - t0
    lat_ms = 1000.0 * score_s / max(len(df), 1)

    # ---------------- operating point (val, attack vs normal -- as published) ----
    va = df[(df.split == "val") & (df.cls.isin(["attack", "physiology_normal"]))]
    yv = (va.cls == "attack").astype(int).values
    if len(yv) and yv.sum() and (yv == 0).any():
        thr = _best_f1_threshold(yv, va["score"].values)
        op_source = "val_maxF1"
    else:
        tr_scores = df.loc[tr_mask, "score"].values
        thr = float(np.quantile(tr_scores, 0.95))
        op_source = "train_contamination_5pct"

    te = df[df.split == "test"].reset_index(drop=True)
    te["pred"] = (te["score"].values >= thr).astype(int)

    n_records_test = int(te["record_id"].nunique())
    atk_te = te[te.cls == "attack"]
    n_hosts_test = int(atk_te["window_id"].nunique())

    # ---------------- metrics under each negative definition ----------------
    rows, metrics = [], dict(
        detector=name, op_threshold=round(float(thr), 6), op_source=op_source,
        fit_s=round(fit_s, 2), latency_ms_per_window=round(lat_ms, 4),
        n_records_test=n_records_test, n_attack_windows_test=int(len(atk_te)),
        n_distinct_host_windows_test=n_hosts_test,
    )

    for tag, negs in NEGATIVE_SETS.items():
        sub = te[te.cls.isin(("attack",) + tuple(negs))].reset_index(drop=True)
        y = (sub.cls == "attack").astype(int).values
        s = sub["score"].values
        pred = sub["pred"].values
        prev = float(y.mean())
        auprc = _auprc(y, s)
        auroc = _auroc(y, s)
        prec, rec, f1 = _prf(y, pred)

        lo_a, hi_a, nb = bootstrap_ci(
            sub, lambda i: _auprc(y[i], s[i]), n_boot=n_boot, ci=ci, seed=seed)
        lo_f, hi_f, _ = bootstrap_ci(
            sub, lambda i: _prf(y[i], pred[i])[2], n_boot=n_boot, ci=ci, seed=seed + 1)

        rows.append(dict(
            negatives="+".join(negs), n=len(sub), n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
            prevalence=round(prev, 4), no_skill_AUPRC=round(prev, 4),
            AUPRC=round(auprc, 4), AUPRC_lift=round(auprc - prev, 4),
            AUPRC_CI=_fmt_ci(lo_a, hi_a), AUROC=round(auroc, 4),
            precision=round(prec, 4), recall=round(rec, 4),
            F1=round(f1, 4), F1_CI=_fmt_ci(lo_f, hi_f), n_boot_ok=nb))

        metrics[f"AUPRC_{tag}"] = round(auprc, 4)
        metrics[f"AUPRC_CI_{tag}"] = _fmt_ci(lo_a, hi_a)
        metrics[f"no_skill_{tag}"] = round(prev, 4)
        metrics[f"F1_{tag}"] = round(f1, 4)
        metrics[f"recall_{tag}"] = round(rec, 4)
        metrics[f"precision_{tag}"] = round(prec, 4)

        if tag == "vs_normal" and target_prevalence:
            mu, sd, achieved = prevalence_matched_auprc(
                y, s, target=target_prevalence, n_rep=n_prev_rep, seed=seed)
            metrics["AUPRC_at_prev"] = round(mu, 4) if np.isfinite(mu) else float("nan")
            metrics["AUPRC_at_prev_sd"] = round(sd, 4) if np.isfinite(sd) else float("nan")
            metrics["prev_matched_target"] = round(achieved, 4) if np.isfinite(achieved) else float("nan")
            metrics["lift_x_released"] = round(auprc / prev, 2) if prev else float("nan")
            metrics["lift_x_matched"] = (round(mu / achieved, 2)
                                        if np.isfinite(mu) and achieved else float("nan"))

    neg_table = pd.DataFrame(rows)

    # ---------------- false-alarm rate per clean class at the operating point ----
    fpr_rows = []
    for c in CLEAN_CLASSES:
        d = te[te.cls == c]
        if not len(d):
            fpr_rows.append(dict(clean_class=c, n=0, n_records=0,
                                 false_alarm_rate=float("nan"), CI="[nan, nan]"))
            continue
        d = d.reset_index(drop=True)
        p = d["pred"].values
        lo, hi, _ = bootstrap_ci(d, lambda i: float(p[i].mean()),
                                 n_boot=n_boot, ci=ci, seed=seed + 2)
        fpr_rows.append(dict(clean_class=c, n=len(d), n_records=int(d["record_id"].nunique()),
                             false_alarm_rate=round(float(p.mean()), 4), CI=_fmt_ci(lo, hi)))
        metrics[f"FAR_{c}"] = round(float(p.mean()), 4)
    fpr_table = pd.DataFrame(fpr_rows)

    # ---------------- RQ4 grids with per-cell n ----------------
    def _grid(fam):
        g = atk_te[atk_te.family == fam]
        if not len(g):
            return pd.DataFrame(), pd.DataFrame()
        rate = g.pivot_table(index="morphology", columns="level", values="pred",
                             aggfunc="mean").round(3)
        n = g.pivot_table(index="morphology", columns="level", values="pred",
                          aggfunc="count")
        return rate, n

    rq4 = {}
    for fam in ("fdi", "spoofing"):
        rate, n = _grid(fam)
        rq4[fam] = rate
        rq4[f"{fam}_n"] = n

    rep = atk_te[atk_te.family == "replay"]
    if len(rep):
        by_src = rep.groupby("replay_source")["pred"].mean().round(3)
        by_src["overall"] = round(float(rep["pred"].mean()), 3)
        rq4["replay"] = by_src
        rq4["replay_n"] = rep.groupby("replay_source")["pred"].count()
    else:
        rq4["replay"] = pd.Series(dtype=float)
        rq4["replay_n"] = pd.Series(dtype=int)

    # ---------------- the frontier, with record-level CIs ----------------
    frontier = {"detector": name}
    for label, mask in (
        ("replay", atk_te.family == "replay"),
        ("fdi_spike", (atk_te.family == "fdi") & (atk_te.morphology == "spike")),
        ("fdi_spike_L1", (atk_te.family == "fdi") & (atk_te.morphology == "spike") & (atk_te.level == 1)),
        ("spoofing_sustained", (atk_te.family == "spoofing") & (atk_te.morphology != "spike")),
    ):
        d = atk_te[mask.values].reset_index(drop=True)
        if not len(d):
            frontier[label] = float("nan")
            frontier[f"{label}_CI"] = "[nan, nan]"
            frontier[f"{label}_n"] = 0
            continue
        p = d["pred"].values
        lo, hi, _ = bootstrap_ci(d, lambda i: float(p[i].mean()),
                                 n_boot=n_boot, ci=ci, seed=seed + 3)
        frontier[label] = round(float(p.mean()), 3)
        frontier[f"{label}_CI"] = _fmt_ci(lo, hi)
        frontier[f"{label}_n"] = int(len(d))

    # ---------------- report ----------------
    if verbose:
        print(f"\n{'='*78}\n RQ1 — {name}  (test; un-adjusted; operating point = {op_source})\n{'='*78}")
        print(f"  effective sample size: {n_records_test} test records | "
              f"{n_hosts_test} distinct host windows -> {len(atk_te)} attack windows "
              f"({len(atk_te)/max(n_hosts_test,1):.0f} variants per host)")
        print(f"  fit {fit_s:.2f}s | latency {lat_ms:.4f} ms/window | threshold {thr:.6g}")
        print("\n  --- detection under three negative-class definitions "
              "(CIs resample records) ---")
        print(neg_table.to_string(index=False))
        print("\n  --- false-alarm rate at the operating point, by clean class ---")
        print(fpr_table.to_string(index=False))
        if np.isfinite(metrics.get("AUPRC_at_prev", float("nan"))):
            tp = metrics["prev_matched_target"]
            rel, mp = metrics["no_skill_vs_normal"], metrics["AUPRC_at_prev"]
            print(f"\n  --- AUPRC vs its OWN chance floor at two base rates ---")
            print(f"    released prevalence {rel:.4f}: AUPRC {metrics['AUPRC_vs_normal']:.4f}"
                  f"  =  {metrics['AUPRC_vs_normal']/rel:.2f}x chance")
            print(f"    matched  prevalence {tp:.4f}: AUPRC {mp:.4f} +/- "
                  f"{metrics['AUPRC_at_prev_sd']:.4f}  =  {mp/tp:.2f}x chance")
            print("    (the released prevalence flatters the absolute AUPRC and "
                  "UNDERSTATES the lift)")
        print("\n  --- RQ4 FDI detection rate by morphology x level (n per cell below) ---")
        if len(rq4["fdi"]):
            print(rq4["fdi"].to_string())
            print("  n:"); print(rq4["fdi_n"].to_string())
        print("\n  --- RQ4 spoofing (manifold-breaking control) ---")
        print(rq4["spoofing"].to_string() if len(rq4["spoofing"]) else "  (none)")
        print("\n  --- RQ4 replay (mechanism-preserving; no severity levels) ---")
        print(rq4["replay"].to_string() if len(rq4["replay"]) else "  (none)")
        print("\n  --- frontier with record-level CIs ---")
        for label in ("replay", "fdi_spike", "fdi_spike_L1", "spoofing_sustained"):
            print(f"    {label:20s} {frontier[label]}  {frontier[f'{label}_CI']}  "
                  f"(n={frontier[f'{label}_n']})")

    # ---------------- save ----------------
    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(f"{out_dir}/rq1_{name}.csv", index=False)
        neg_table.to_csv(f"{out_dir}/rq1_negatives_{name}.csv", index=False)
        fpr_table.to_csv(f"{out_dir}/rq1_falsealarm_{name}.csv", index=False)
        pd.DataFrame([frontier]).to_csv(f"{out_dir}/rq4_frontier_{name}.csv", index=False)
        for fam in ("fdi", "spoofing"):
            if len(rq4[fam]):
                rq4[fam].to_csv(f"{out_dir}/rq4_{fam}_{name}.csv")
                rq4[f"{fam}_n"].to_csv(f"{out_dir}/rq4_{fam}_n_{name}.csv")
        if len(rq4["replay"]):
            rq4["replay"].to_csv(f"{out_dir}/rq4_replay_{name}.csv", header=["detection_rate"])
        if verbose:
            print(f"\n  [saved] rq1_{name}.csv, rq1_negatives_{name}.csv, "
                  f"rq1_falsealarm_{name}.csv, rq4_*_{name}.csv -> {out_dir}")

    return dict(metrics=metrics, negatives=neg_table, false_alarm=fpr_table,
                rq4=rq4, frontier=frontier, scores=df["score"].values)


# --------------------------------------------------------------------------- #
def consolidate(results, out_dir=None):
    """Stack per-detector results into the two paper tables."""
    t3 = pd.DataFrame([r["metrics"] for r in results])
    keep = ["detector", "no_skill_vs_normal", "AUPRC_vs_normal", "AUPRC_CI_vs_normal",
            "AUPRC_vs_normal_fault", "AUPRC_CI_vs_normal_fault", "AUPRC_vs_all_clean",
            "AUPRC_at_prev", "lift_x_released", "lift_x_matched",
            "precision_vs_normal", "recall_vs_normal", "F1_vs_normal",
            "F1_CI_vs_normal", "FAR_physiology_normal", "FAR_fault_artifact",
            "FAR_physiology_event", "latency_ms_per_window", "fit_s"]
    t3 = t3[[c for c in keep if c in t3.columns]]
    t6 = pd.DataFrame([r["frontier"] for r in results])
    if out_dir:
        t3.to_csv(f"{out_dir}/table3R_detection_all.csv", index=False)
        t6.to_csv(f"{out_dir}/table6R_frontier_all.csv", index=False)
    return t3, t6


# =========================================================================== #
#  PART 2 — matched-FAR comparison, threshold-free subproblems, multi-seed
# =========================================================================== #
#
# Why this section exists. Part 1 scores every detector at its OWN val-max-F1
# operating point, which is how the published Table 3-R was produced. That makes
# cross-detector detection rates incomparable: OmniAnomaly reaches replay 0.710 while
# flagging 58% of normal windows, 44% of artifacts and 74% of genuine physiological
# events, so its apparent advantage is partly a looser threshold rather than better
# discrimination. Two fixes, and they answer different questions:
#
#   subproblem_auprc()   THRESHOLD-FREE. AUPRC of one attack subset against the clean
#                        normals. No operating point, so no FAR confound at all. This is
#                        the analysis that decides whether the replay claim survives.
#
#   detection_at_far()   MATCHED OPERATING POINT. Every detector's threshold is set on
#                        the VALIDATION normals to hit a common false-alarm budget, then
#                        applied to test. Detection rates then mean the same thing across
#                        detectors, and stay in the units the paper already reports.
#
#   run_multiseed()      Deep detectors are not bit-reproducible: cuDNN's RNN and
#                        reduction kernels are nondeterministic and torch.manual_seed
#                        does not control them, so USAD moved AUPRC +0.038 and replay
#                        -0.085 between two identical runs. set_determinism() reduces but
#                        cannot eliminate this (cuDNN has no deterministic RNN backward),
#                        so the reportable answer is mean +/- sd over seeds.

import os
import warnings

# Attack subsets. Each is a predicate on the scored test frame.
ATTACK_GROUPS = {
    "all_attack":         lambda d: d["cls"] == "attack",
    "replay":             lambda d: d["family"] == "replay",
    "replay_same":        lambda d: (d["family"] == "replay") & (d["replay_source"] == "same_patient"),
    "replay_cross":       lambda d: (d["family"] == "replay") & (d["replay_source"] == "cross_patient"),
    "fdi_spike":          lambda d: (d["family"] == "fdi") & (d["morphology"] == "spike"),
    "fdi_spike_L1":       lambda d: (d["family"] == "fdi") & (d["morphology"] == "spike") & (d["level"] == 1),
    "fdi_sustained":      lambda d: (d["family"] == "fdi") & (d["morphology"] != "spike"),
    "fdi_sustained_L1":   lambda d: (d["family"] == "fdi") & (d["morphology"] != "spike") & (d["level"] == 1),
    "spoofing_spike":     lambda d: (d["family"] == "spoofing") & (d["morphology"] == "spike"),
    "spoofing_sustained": lambda d: (d["family"] == "spoofing") & (d["morphology"] != "spike"),
}

# The subsets that carry the paper's frontier argument.
FRONTIER_GROUPS = ("replay", "fdi_spike", "fdi_spike_L1", "fdi_sustained", "spoofing_sustained")

DEFAULT_FAR_TARGETS = (0.01, 0.05, 0.10)


def set_determinism(warn_only=True):
    """Best-effort torch determinism. Call BEFORE any CUDA work.

    Returns a short status string for the manifest. cuDNN provides no deterministic
    RNN backward, so LSTM-AE, OmniAnomaly (GRU) and GDN will still vary run to run
    even with this set -- which is why run_multiseed exists rather than a promise of
    bit-identical reruns.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        warnings.warn("CUDA already initialised; CUBLAS_WORKSPACE_CONFIG may be ignored. "
                      "Call set_determinism() in the first cell of a fresh runtime.")
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return f"deterministic_algorithms=True(warn_only={warn_only}), cudnn.deterministic=True"
    except Exception as e:                                   # pragma: no cover
        return f"could not enable determinism: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# fit + score once, then reuse the scored frame for every analysis
# --------------------------------------------------------------------------- #

def score_frame(df, X, detector, verbose=False):
    """Fit on TRAIN physiology_normal, score every row. Returns (scored_df, timing)."""
    df = df.reset_index(drop=True).copy()
    X = np.asarray(X)
    tr = ((df.split == "train") & (df.cls == "physiology_normal")).values
    if tr.sum() == 0:
        raise ValueError("No TRAIN physiology_normal windows to fit on.")
    t0 = time.perf_counter()
    detector.fit(X[tr])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    df["score"] = np.asarray(detector.score(X)).ravel().astype(float)
    score_s = time.perf_counter() - t0
    if verbose:
        print(f"  [{detector.name}] fit {fit_s:.1f}s, score {score_s:.1f}s")
    return df, dict(fit_s=round(fit_s, 2),
                    latency_ms_per_window=round(1000 * score_s / max(len(df), 1), 4))


# --------------------------------------------------------------------------- #
# (a) threshold-free: AUPRC per attack subset vs the clean normals
# --------------------------------------------------------------------------- #

def subproblem_auprc(df_scored, detector_name="detector", groups=None,
                     negatives=("physiology_normal",), split="test",
                     n_boot=2000, ci=95, seed=20260729, verbose=True):
    """AUPRC of each attack subset against `negatives`, with a record-level CI.

    Threshold-free, so a detector cannot buy a high number by flagging more. Each
    subproblem has its own prevalence (and therefore its own chance floor), which is
    reported beside it -- the numbers are comparable ACROSS DETECTORS within a row,
    not across rows.
    """
    groups = groups or ATTACK_GROUPS
    d = df_scored[df_scored.split == split]
    neg = d[d.cls.isin(negatives)]
    rows = []
    for name, pred in groups.items():
        pos = d[pred(d).values]
        if not len(pos) or not len(neg):
            continue
        sub = pd.concat([pos, neg], ignore_index=True)
        # positives are exactly the rows that came from `pos`
        y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
        s = sub["score"].values
        prev = float(y.mean())
        auprc = _auprc(y, s)
        auroc = _auroc(y, s)
        lo, hi, nb = bootstrap_ci(sub, lambda i: _auprc(y[i], s[i]),
                                 n_boot=n_boot, ci=ci, seed=seed)
        rows.append(dict(detector=detector_name, subproblem=name,
                         n_pos=len(pos), n_neg=len(neg),
                         n_records=int(sub["record_id"].nunique()),
                         prevalence=round(prev, 4), no_skill_AUPRC=round(prev, 4),
                         AUPRC=round(auprc, 4), AUPRC_CI=_fmt_ci(lo, hi),
                         lift_x=round(auprc / prev, 2) if prev else float("nan"),
                         AUROC=round(auroc, 4), n_boot_ok=nb))
    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f"\n  --- threshold-free AUPRC by attack subset vs {'+'.join(negatives)} "
              f"({detector_name}) ---")
        print(out.drop(columns=["detector"]).to_string(index=False))
    return out


# --------------------------------------------------------------------------- #
# (b) matched operating point: detection rate at a common false-alarm budget
# --------------------------------------------------------------------------- #

def detection_at_far(df_scored, detector_name="detector", far_targets=DEFAULT_FAR_TARGETS,
                     groups=None, calib_split="val", calib_class="physiology_normal",
                     n_boot=2000, ci=95, seed=20260729, verbose=True):
    """Detection rate per attack subset with the threshold set to a COMMON false-alarm
    budget on the validation normals, then applied to test.

    Calibrating on val (never test) keeps the comparison leak-free; the FAR actually
    achieved on the test normals is reported so the reader can see the calibration
    transfer. This is the comparison the published table should have made: at FAR 0.10
    every detector is allowed the same nuisance-alarm load, so the detection rates mean
    the same thing.
    """
    groups = groups or ATTACK_GROUPS
    cal = df_scored[(df_scored.split == calib_split) & (df_scored.cls == calib_class)]["score"].values
    if len(cal) < 50:
        warnings.warn(f"only {len(cal)} calibration windows in {calib_split}/{calib_class}; "
                      "falling back to TRAIN normals")
        cal = df_scored[(df_scored.split == "train") & (df_scored.cls == calib_class)]["score"].values
    te = df_scored[df_scored.split == "test"]

    rows = []
    for far in far_targets:
        thr = float(np.quantile(cal, 1.0 - far))
        te_p = te.assign(pred=(te["score"].values >= thr).astype(int))
        rec = dict(detector=detector_name, far_target=far, threshold=round(thr, 6))
        for c in CLEAN_CLASSES:
            dd = te_p[te_p.cls == c]
            rec[f"FAR_{c}"] = round(float(dd["pred"].mean()), 4) if len(dd) else float("nan")
        for name, pred in groups.items():
            dd = te_p[pred(te_p).values].reset_index(drop=True)
            if not len(dd):
                continue
            p = dd["pred"].values
            lo, hi, _ = bootstrap_ci(dd, lambda i: float(p[i].mean()),
                                     n_boot=n_boot, ci=ci, seed=seed)
            rec[name] = round(float(p.mean()), 3)
            rec[f"{name}_CI"] = _fmt_ci(lo, hi)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if verbose and len(out):
        cols = ["far_target", "FAR_physiology_normal", "FAR_fault_artifact",
                "FAR_physiology_event"] + [g for g in FRONTIER_GROUPS if g in out.columns]
        print(f"\n  --- detection at a MATCHED false-alarm budget ({detector_name}; "
              f"threshold calibrated on {calib_split} normals) ---")
        print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    return out


# --------------------------------------------------------------------------- #
# (c) multi-seed
# --------------------------------------------------------------------------- #

MULTISEED_METRICS = ("AUPRC_vs_normal", "AUPRC_vs_normal_fault", "F1_vs_normal",
                     "FAR_physiology_normal", "FAR_fault_artifact", "FAR_physiology_event",
                     "auprc_replay", "auprc_fdi_spike", "auprc_fdi_sustained",
                     "auprc_spoofing_sustained", "auroc_replay",
                     "far10_physiology_normal", "far10_fault_artifact", "far10_physiology_event",
                     "det10_replay", "det10_fdi_spike", "det10_fdi_sustained",
                     "det10_spoofing_sustained", "fit_s", "latency_ms_per_window")


def run_multiseed(df, X, factory, seeds=(0, 1, 2), name=None, far_target=0.10,
                  out_dir=None, verbose=True):
    """Repeat a detector across seeds and report mean +/- sd of the headline metrics.

    `factory` is a callable seed -> detector, e.g.
        lambda s: deep_detectors.USADDetector(epochs=40, seed=s)

    Deep detectors on cuDNN are not reproducible even at a fixed seed, so single-run
    point estimates in Table 3-R are not defensible. Report mean +/- sd over >=3 seeds
    and let the spread decide which cross-detector orderings you are entitled to claim.
    """
    per_seed = []
    for sd in seeds:
        det = factory(sd)
        nm = name or det.name
        if verbose:
            print(f"\n>>> {nm}  seed {sd}")
        scored, timing = score_frame(df, X, det, verbose=verbose)
        row = dict(detector=nm, seed=sd, **timing)

        # published-style metrics at this seed's own val-max-F1 point
        va = scored[(scored.split == "val") & (scored.cls.isin(["attack", "physiology_normal"]))]
        yv = (va.cls == "attack").astype(int).values
        thr = (_best_f1_threshold(yv, va["score"].values) if len(yv) and yv.sum() and (yv == 0).any()
               else float(np.quantile(scored[(scored.split == "train") &
                                             (scored.cls == "physiology_normal")]["score"], 0.95)))
        te = scored[scored.split == "test"].assign(pred=lambda d: (d["score"] >= thr).astype(int))
        for tag, negs in NEGATIVE_SETS.items():
            sub = te[te.cls.isin(("attack",) + tuple(negs))]
            y = (sub.cls == "attack").astype(int).values
            row[f"AUPRC_{tag}"] = round(_auprc(y, sub["score"].values), 4)
            row[f"F1_{tag}"] = round(_prf(y, sub["pred"].values)[2], 4)
        for c in CLEAN_CLASSES:
            dd = te[te.cls == c]
            row[f"FAR_{c}"] = round(float(dd["pred"].mean()), 4) if len(dd) else float("nan")

        # threshold-free + matched-FAR, no bootstrap (the seed spread is the uncertainty)
        sp = subproblem_auprc(scored, nm, groups={g: ATTACK_GROUPS[g] for g in FRONTIER_GROUPS},
                              n_boot=0, verbose=False).set_index("subproblem")
        for g in FRONTIER_GROUPS:
            if g in sp.index:
                row[f"auprc_{g}"] = sp.loc[g, "AUPRC"]
                row[f"auroc_{g}"] = sp.loc[g, "AUROC"]
        da = detection_at_far(scored, nm, far_targets=(far_target,),
                              groups={g: ATTACK_GROUPS[g] for g in FRONTIER_GROUPS},
                              n_boot=0, verbose=False)
        if len(da):
            tag = int(round(100 * far_target))
            # false-alarm rates AT THE MATCHED THRESHOLD -- distinct from the FAR_*
            # columns above, which come from this seed's own max-F1 point.
            for c in CLEAN_CLASSES:
                col = f"FAR_{c}"
                if col in da.columns:
                    row[f"far{tag}_{c}"] = da.iloc[0][col]
            for g in FRONTIER_GROUPS:
                if g in da.columns:
                    row[f"det{tag}_{g}"] = da.iloc[0][g]
        per_seed.append(row)

    ps = pd.DataFrame(per_seed)
    num = ps.select_dtypes("number").drop(columns=["seed"], errors="ignore")
    summ = pd.DataFrame({
        "mean": num.mean().round(4), "sd": num.std(ddof=1).round(4),
        "min": num.min().round(4), "max": num.max().round(4),
        "range": (num.max() - num.min()).round(4),
    })
    summ.insert(0, "detector", ps["detector"].iloc[0])
    summ["report_as"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(summ["mean"], summ["sd"].fillna(0))]

    if verbose:
        nm = ps["detector"].iloc[0]
        print(f"\n=== {nm}: {len(seeds)} seeds {list(seeds)} ===")
        show = [m for m in MULTISEED_METRICS if m in summ.index]
        print(summ.loc[show, ["mean", "sd", "min", "max", "range", "report_as"]].to_string())
        big = summ.loc[show][summ.loc[show, "range"] > 0.03]
        if len(big):
            print(f"\n  !! seed-to-seed range > 0.03 on: {', '.join(big.index)}")
            print("     single-run values for these are not reportable; use mean ± sd.")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        nm = ps["detector"].iloc[0]
        ps.to_csv(f"{out_dir}/multiseed_perseed_{nm}.csv", index=False)
        summ.to_csv(f"{out_dir}/multiseed_summary_{nm}.csv")
    return summ, ps


# --------------------------------------------------------------------------- #
def compare_matched(frames, out_dir=None, far_target=0.10, verbose=True):
    """Stack per-detector subproblem/matched-FAR frames into the two comparison tables.

    `frames` is a list of (subproblem_df, detection_at_far_df) pairs.
    """
    sp = pd.concat([a for a, _ in frames], ignore_index=True)
    da = pd.concat([b for _, b in frames], ignore_index=True)

    piv_auprc = sp.pivot(index="detector", columns="subproblem", values="AUPRC")
    # AUPRC_CI is a string column and is absent once seeds have been aggregated
    piv_ci = (sp.pivot(index="detector", columns="subproblem", values="AUPRC_CI")
              if "AUPRC_CI" in sp.columns else
              sp.pivot(index="detector", columns="subproblem", values="AUPRC_sd")
              if "AUPRC_sd" in sp.columns else None)
    at = da[da.far_target == far_target].set_index("detector")

    if verbose:
        print(f"\n=== threshold-free AUPRC by subproblem (all detectors) ===")
        print(piv_auprc[[c for c in FRONTIER_GROUPS if c in piv_auprc.columns]].to_string())
        print(f"\n  chance floors: " + ", ".join(
            f"{r.subproblem} {r.no_skill_AUPRC}" for r in
            sp.drop_duplicates("subproblem").itertuples() if r.subproblem in FRONTIER_GROUPS))
        cols = ["FAR_physiology_normal", "FAR_fault_artifact", "FAR_physiology_event"] + \
               [c for c in FRONTIER_GROUPS if c in at.columns]
        print(f"\n=== detection at matched FAR = {far_target} (all detectors) ===")
        print(at[[c for c in cols if c in at.columns]].to_string())
    if out_dir:
        sp.to_csv(f"{out_dir}/subproblem_auprc_all.csv", index=False)
        da.to_csv(f"{out_dir}/detection_at_far_all.csv", index=False)
        piv_auprc.to_csv(f"{out_dir}/table7R_subproblem_auprc.csv")
        at.to_csv(f"{out_dir}/table8R_detection_at_far{int(100*far_target)}.csv")
    return piv_auprc, piv_ci, at


def aggregate_seeds(frames, index_cols, seed_col="seed"):
    """Stack per-seed frames and return mean/sd of every numeric column.

    Used by the multi-seed matched-FAR stage so Tables 4-R and 5-R carry the same
    provenance as Table 3-R. String columns (bootstrap CI text) are dropped: with
    several seeds the across-seed sd is the interval to report.
    """
    d = pd.concat(frames, ignore_index=True)
    num = [c for c in d.columns
           if c not in index_cols + [seed_col] and pd.api.types.is_numeric_dtype(d[c])]
    g = d.groupby(index_cols, sort=False)[num]
    mean, sd = g.mean(), g.std(ddof=1)
    out = mean.round(4).copy()
    for c in num:
        out[c + "_sd"] = sd[c].round(4)
    out["n_seeds"] = g.size()
    return out.reset_index()
