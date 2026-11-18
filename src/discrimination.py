"""
Cause discrimination for IoMT-SecAlarmBench: attack vs fault vs physiology.

Given an anomalous window, is the cause a cyberattack, a device fault or a genuine
physiological event? This is supervised multi-class classification, separate from
unsupervised detection, and tests whether the causes are separable at all.

Two views:
  * 4-way: {physiology_normal, physiology_event, fault_artifact, attack}
  * 3-way: {physiology_event, fault_artifact, attack} (anomalies only)

Uses the harness feature matrix (df[feat_cols]) and the record-disjoint split. A
class-balanced random forest is trained on train+val records and evaluated on the
held-out test records. Reports the confusion matrix, per-class precision/recall/F1,
macro-F1, MCC, Cohen's kappa, and an expected cost under an asymmetric cost matrix
in which dismissing a genuine deterioration costs most. The cost matrix is
illustrative; it has not been set by a clinician.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             matthews_corrcoef, cohen_kappa_score, f1_score)

CLASSES_4 = ["physiology_normal", "physiology_event", "fault_artifact", "attack"]
CLASSES_3 = ["physiology_event", "fault_artifact", "attack"]
SHORT = {"physiology_normal": "normal", "physiology_event": "event",
         "fault_artifact": "fault", "attack": "attack"}

# Placeholder asymmetric clinician cost matrix (4-way). rows = TRUE, cols = PRED, in
# CLASSES_4 order. Dismissing a genuine physiological event (true=event predicted as
# normal/fault/attack) is most expensive. TO BE REPLACED with the clinician's ratio.
DEFAULT_COST_4 = np.array([
    #  pred:  normal  event  fault  attack
    [     0.0,   1.0,   1.0,    2.0],   # true normal  (spurious alarm — mild)
    [    10.0,   0.0,   8.0,    8.0],   # true event   (MISSED real deterioration — worst)
    [     2.0,   3.0,   0.0,    2.0],   # true fault   (artifact called real — moderate)
    [     3.0,   3.0,   2.0,    0.0],   # true attack  (missed/misattributed attack)
], dtype=float)


def _cost_submatrix(cost4, classes):
    idx = [CLASSES_4.index(c) for c in classes]
    return cost4[np.ix_(idx, idx)]


def run_rq3(df, feat_cols, mode="4way", classifier=None, cost_matrix=None,
            n_estimators=300, seed=20260724, out_dir=None, verbose=True):
    classes = CLASSES_4 if mode == "4way" else CLASSES_3
    d = df[df["cls"].isin(classes)].copy()
    tr = d[d["split"].isin(["train", "val"])]
    te = d[d["split"] == "test"]
    if len(tr) == 0 or len(te) == 0:
        raise ValueError("Empty train or test split for RQ3 — run patient_wise_split first.")

    clf = classifier or RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(tr[feat_cols].values, tr["cls"].values)
    pred = clf.predict(te[feat_cols].values)
    y = te["cls"].values

    cm = confusion_matrix(y, pred, labels=classes)
    P, R, F, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    macro = f1_score(y, pred, labels=classes, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y, pred)
    kappa = cohen_kappa_score(y, pred, labels=classes)

    cost4 = cost_matrix if cost_matrix is not None else DEFAULT_COST_4
    cost_m = _cost_submatrix(cost4, classes)
    exp_cost = float((cm * cost_m).sum() / cm.sum())

    short = [SHORT[c] for c in classes]
    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in short],
                         columns=[f"pred_{s}" for s in short])
    per_class = pd.DataFrame({"precision": P.round(3), "recall": R.round(3),
                              "F1": F.round(3), "support": sup}, index=short)
    summary = dict(mode=mode, macro_F1=round(macro, 4), MCC=round(mcc, 4),
                   cohen_kappa=round(kappa, 4), expected_cost=round(exp_cost, 4),
                   n_test=int(len(y)))

    if verbose:
        print(f"\n=== RQ3 {mode} discrimination (test; RandomForest, balanced) ===")
        print("confusion matrix (rows=true, cols=pred):")
        print(cm_df.to_string())
        print("\nper-class:")
        print(per_class.to_string())
        print(f"\nmacro-F1 {summary['macro_F1']} | MCC {summary['MCC']} | "
              f"kappa {summary['cohen_kappa']} | expected cost {summary['expected_cost']} "
              f"(placeholder cost matrix)")

        def _rate(true_c, pred_c):
            ti, pj = classes.index(true_c), classes.index(pred_c)
            row = cm[ti].sum()
            return cm[ti, pj] / row if row else float("nan")
        print("\nclinically critical confusion rates:")
        print(f"  true event -> attack : {_rate('physiology_event','attack'):.3f}  (dismiss deterioration as attack)")
        print(f"  true event -> fault  : {_rate('physiology_event','fault_artifact'):.3f}  (dismiss deterioration as artifact)")
        print(f"  true attack -> event : {_rate('attack','physiology_event'):.3f}  (mistake attack for real event)")
        print(f"  true fault  -> attack: {_rate('fault_artifact','attack'):.3f}  (mistake artifact for attack)")

    if out_dir:
        cm_df.to_csv(f"{out_dir}/rq3_{mode}_confusion.csv")
        per_class.to_csv(f"{out_dir}/rq3_{mode}_perclass.csv")
        pd.DataFrame([summary]).to_csv(f"{out_dir}/rq3_{mode}_summary.csv", index=False)
        if verbose:
            print(f"\n[saved] rq3_{mode}_confusion/perclass/summary.csv")
    return summary, cm_df, per_class


def _demo():
    from injection_generator import GeneratorConfig
    from eval_harness import build_dataset, patient_wise_split
    FS = 125.0; rng = np.random.default_rng(0); recs = []
    for i in range(6):
        for atype, tv in [("Asystole", False), ("Bradycardia", True), ("Tachycardia", True),
                          ("VentricularTachycardia", False), ("VentricularFlutterFib", False)]:
            T = int(FS * 300); t = np.arange(T) / FS; ph = 2 * np.pi * (70 + 3 * i) / 60.0 * t
            sig = np.vstack([np.sin(ph) + 0.02 * rng.standard_normal(T),
                             0.9 * np.sin(ph - 0.6) + 0.02 * rng.standard_normal(T)])
            sig = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
            recs.append(dict(record_id=f"{atype[:3]}{i}", signal=sig, channels=["ECG", "PPG"],
                             alarm_type=atype, true_alarm=tv, alarm_onset_sample=int(T * 0.85), fs=FS))
    df, feat_cols = build_dataset(recs, GeneratorConfig(), verbose=False)
    df = patient_wise_split(df)
    run_rq3(df, feat_cols, mode="4way")
    run_rq3(df, feat_cols, mode="3way")


if __name__ == "__main__":
    _demo()
