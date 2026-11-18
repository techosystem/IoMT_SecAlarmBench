"""
WUSTL-EHMS-2020 identifier-leakage audit for IoMT-SecAlarmBench.

Measures how much of the near-perfect accuracy obtained on WUSTL-EHMS-2020 under a
random split depends on identifier features rather than on detection.

The same classifier is trained twice:
  (A) with identifier features (source/destination addresses, MACs, ports,
      direction and flags);
  (B) without them (traffic-volume counters and biometrics kept);
and the drop in accuracy, macro-F1, attack recall and AUPRC is reported, together
with the feature importances of (A).

WUSTL-EHMS-2020 is a single capture session, so a stratified random split is used;
a whole-session hold-out is not possible. That is the setting in which identifiers
leak, and the limitation is reported with the result.

Usage:
    from leakage_audit import run_leakage_audit
    run_leakage_audit("data/wustl/wustl-ehms-2020.csv", out_dir="results")
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             average_precision_score, precision_score)

# identifier / session-artifact name fragments = the leakage features to drop in (B)
ID_FRAGMENTS = ("addr", "mac", "sport", "dport", "dir", "flgs", "sid",
                "stime", "ltime", "starttime", "lasttime", "timestamp", "seq")

# label-column candidates
LABEL_CANDIDATES = ("label", "attack", "class", "target", "y", "is_attack", "attack_category", "type")


def _find_label(df, label_col):
    if label_col and label_col in df.columns:
        return label_col
    for c in df.columns:
        if c.lower() in LABEL_CANDIDATES:
            return c
    return df.columns[-1]


def _is_identifier(col):
    c = col.lower()
    return any(frag in c for frag in ID_FRAGMENTS)


def _encode(df):
    """Factorize any non-numeric column (object / string / category) to integer codes."""
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.factorize(out[c])[0]
    return out.fillna(0)


def _metrics(y, pred, proba):
    return dict(
        accuracy=round(accuracy_score(y, pred), 4),
        macro_F1=round(f1_score(y, pred, average="macro", zero_division=0), 4),
        attack_precision=round(precision_score(y, pred, pos_label=1, zero_division=0), 4),
        attack_recall=round(recall_score(y, pred, pos_label=1, zero_division=0), 4),
        attack_F1=round(f1_score(y, pred, pos_label=1, zero_division=0), 4),
        AUPRC=round(average_precision_score(y, proba), 4),
    )


def run_leakage_audit(csv_path, label_col="Label", test_size=0.3, seed=20260724,
                      n_estimators=200, out_dir=None, verbose=True):
    df = pd.read_csv(csv_path)
    label = _find_label(df, label_col)
    y = df[label]
    # binarize label if needed (benign=0, anything else=1)
    if y.dtype == object or y.nunique() > 2:
        y = (y.astype(str).str.lower() != "benign").astype(int) if y.dtype == object else (y != y.mode()[0]).astype(int)
    y = y.astype(int).values

    X = df.drop(columns=[label])
    # drop ANY secondary label column (WUSTL ships both 'Label' and 'Attack Category',
    # which is a categorical copy of the answer and would leak perfectly if kept)
    SECONDARY_LABELS = ("attack category", "attack_category", "category", "attack",
                        "attack type", "attack_type", "label", "class", "type")
    sec = [c for c in X.columns if c.lower() in SECONDARY_LABELS]
    if sec:
        X = X.drop(columns=sec)
    # drop constant columns (no information; also removes constant identifiers)
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const)

    id_cols = [c for c in X.columns if _is_identifier(c)]
    keep_cols = [c for c in X.columns if c not in id_cols]

    if verbose:
        print(f"[wustl] rows {len(df)} | label '{label}' | attack rate {y.mean():.3f}")
        if sec:
            print(f"[wustl] dropped secondary label column(s) (would leak the answer): {sec}")
        print(f"[wustl] dropped {len(const)} constant cols: {const}")
        print(f"[wustl] identifier (leakage) cols dropped in condition B ({len(id_cols)}): {id_cols}")
        print(f"[wustl] retained cols in condition B ({len(keep_cols)}): {keep_cols}")

    Xe = _encode(X)
    Xtr, Xte, ytr, yte = train_test_split(Xe, y, test_size=test_size,
                                          random_state=seed, stratify=y)

    results = {}
    importances = {}
    for cond, cols in [("A_with_identifiers", list(Xe.columns)),
                       ("B_without_identifiers", keep_cols)]:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                     class_weight="balanced", n_jobs=-1)
        clf.fit(Xtr[cols], ytr)
        proba = clf.predict_proba(Xte[cols])[:, 1]
        pred = clf.predict(Xte[cols])
        results[cond] = _metrics(yte, pred, proba)
        imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
        importances[cond] = imp

    tbl = pd.DataFrame(results).T
    drop = (tbl.loc["A_with_identifiers"] - tbl.loc["B_without_identifiers"]).round(4)
    tbl.loc["DROP (A - B)"] = drop

    if verbose:
        print("\n=== Leakage audit: RandomForest with vs. without identifier features ===")
        print(tbl.to_string())
        print("\n=== Top-8 feature importances WITH identifiers (evidence of leakage) ===")
        top = importances["A_with_identifiers"].head(8)
        for name, val in top.items():
            flag = "  <-- IDENTIFIER" if _is_identifier(name) else ""
            print(f"  {name:16s} {val:.4f}{flag}")
        id_share = importances["A_with_identifiers"][id_cols].sum() if id_cols else 0.0
        print(f"\n  identifier features account for {id_share:.1%} of total RF importance")

    if out_dir:
        tbl.to_csv(f"{out_dir}/wustl_leakage_audit.csv")
        importances["A_with_identifiers"].to_csv(f"{out_dir}/wustl_importances.csv",
                                                 header=["importance"])
        if verbose:
            print(f"\n[saved] {out_dir}/wustl_leakage_audit.csv + wustl_importances.csv")
    return tbl, importances


# --------------------------------------------------------------------------- #
def _demo():
    """Synthetic WUSTL-shaped data: an identifier column that perfectly separates the
    label, plus noisy real features weakly predictive. Verifies the audit shows a large
    with-vs-without drop and that the identifier dominates importance."""
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.2).astype(int)                     # 20% attack
    attacker_ips = [f"10.0.0.{i}" for i in range(5)]
    benign_ips = [f"192.168.1.{i}" for i in range(20)]
    SrcAddr = np.where(y == 1, rng.choice(attacker_ips, n), rng.choice(benign_ips, n))
    SrcMac = np.where(y == 1, "AA:BB:CC:00:00:01", "11:22:33:44:55:66")
    Sport = np.where(y == 1, 6666, rng.integers(1024, 65535, n))
    # real-ish features weakly predictive
    SrcBytes = rng.normal(1000, 300, n) + y * 120
    SrcLoad = rng.normal(50, 20, n) + y * 6
    HR = rng.normal(75, 8, n) + y * 1.0
    SpO2 = rng.normal(97, 1.5, n) - y * 0.3
    Temp = rng.normal(36.8, 0.4, n)
    Dir = np.where(rng.random(n) < 0.5, "->", "<-")           # constant-ish artifact
    df = pd.DataFrame(dict(SrcAddr=SrcAddr, SrcMac=SrcMac, Sport=Sport, Dir=Dir,
                           SrcBytes=SrcBytes, SrcLoad=SrcLoad,
                           HR=HR, SpO2=SpO2, Temp=Temp, Label=y))
    import tempfile, os
    p = os.path.join(tempfile.gettempdir(), "wustl_demo.csv")
    df.to_csv(p, index=False)
    run_leakage_audit(p, out_dir=None)


if __name__ == "__main__":
    _demo()
