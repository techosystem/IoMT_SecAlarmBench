"""
Robustness checks for channel attribution (attribution.py).

  1. Permutation negative control: shuffling which channel is labeled as injected
     must drive AC@1 to the chance floor. The p-value tests whether the observed
     AC@1 is localization signal rather than an artifact of a two-way choice.
  2. Per-patient bootstrap 95% CIs for every AC@1.
  3. Baseline context: a random ranker (empirical chance) and the marginal-deviation
     baseline alongside the causal methods.
  4. Richer-channel candidates: the same attributors on windows with more than two
     real channels (ECG, PPG, ABP, RESP) give a genuinely lower chance floor (1/C),
     with the untampered channels as negative controls. This needs windows with
     those channels to be generated.

Padding with clean "decoy" channels does not lower the floor honestly: a deviation
ranker solves it trivially, as demo_decoy_is_trivial() shows.

All resampling runs on cached per-window scores, so the permutation test and the
bootstrap are fast once the attributors have scored the windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import attribution as V

CORE_CHANNELS = V.CORE_CHANNELS


# ---------------------------------------------------------------------------
# scoring: run an attributor once, cache per-window scores over channels
# ---------------------------------------------------------------------------


def score_windows(df, windows, attributor, channels=CORE_CHANNELS,
                  families=("fdi", "spoofing", "replay"), detect_quantile=0.95):
    """Fit on train-normal, score every test-attack window. Returns a dict of
    cached arrays so all downstream resampling needs no re-scoring."""
    windows = np.asarray(windows, float)
    N, C, T = windows.shape
    if len(channels) != C:
        channels = tuple(f"ch{i}" for i in range(C))

    split = df["split"].values if "split" in df.columns else np.array(["train"] * N)
    cls = df["cls"].astype(str).values
    fam = df["family"].astype(str).values if "family" in df.columns else np.array([""] * N)
    tgt = df["target_channel"].values if "target_channel" in df.columns else np.array([None] * N)
    pid_col = "patient_id" if "patient_id" in df.columns else ("record_id" if "record_id" in df.columns else None)
    pid = df[pid_col].astype(str).values if pid_col else np.array(["p0"] * N)

    is_norm = cls == "physiology_normal"
    train_norm = (split == "train") & is_norm
    if train_norm.sum() < 10:
        train_norm = is_norm
    test_attack = (split == "test") & np.isin(fam, list(families))
    if test_attack.sum() == 0:
        test_attack = np.isin(fam, list(families))

    attributor.fit(windows[train_norm])

    # cheap magnitude path for the detection threshold
    def _mags(W):
        if hasattr(attributor, "magnitudes"):
            return attributor.magnitudes(W)
        if hasattr(attributor, "attribute_batch"):
            return attributor.attribute_batch(W)[1]
        return np.array([attributor.attribute(w)[1] for w in W])

    thr = float(np.quantile(_mags(windows[train_norm]), detect_quantile))

    idx = np.where(test_attack)[0]
    if hasattr(attributor, "attribute_batch"):
        sc, mag = attributor.attribute_batch(windows[idx])
    else:
        sc = np.zeros((len(idx), C)); mag = np.zeros(len(idx))
        for k, i in enumerate(idx):
            sc[k], mag[k] = attributor.attribute(windows[i])

    true_idx = np.array([V._resolve_target_index(tgt[i], channels) for i in idx])
    keep = np.array([t is not None for t in true_idx])
    return {
        "method": attributor.name, "C": C, "channels": channels,
        "scores": sc[keep], "true": true_idx[keep].astype(int),
        "patient": pid[idx][keep], "family": fam[idx][keep],
        "detected": (mag[keep] > thr), "chance": 1.0 / C,
    }


# ---------------------------------------------------------------------------
# metrics on cached scores
# ---------------------------------------------------------------------------


def _hits(scores, true):
    """AC@1 hit per window using average-rank tie handling (ties count as miss)."""
    return np.array([V._rank_of(scores[i], true[i]) <= 1 for i in range(len(true))], dtype=float)


def _ac1_per_patient(hits, patients):
    return float(np.mean([hits[patients == p].mean() for p in pd.unique(patients)]))


def bootstrap_ci(hits, patients, n_boot=2000, ci=95, seed=0):
    """Per-patient bootstrap CI on AC@1 (resample patients with replacement)."""
    rng = np.random.default_rng(seed)
    pats = pd.unique(patients)
    per_pat = {p: hits[patients == p].mean() for p in pats}
    vals = np.array([per_pat[p] for p in pats])
    boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(vals.mean()), float(lo), float(hi)


def permutation_null(scores, true, patients, n_perm=2000, seed=0):
    """Shuffle the true-channel assignment across windows (breaks score<->truth
    association, preserves the channel marginal) and recompute per-patient AC@1.
    Returns (observed, null_mean, null_lo, null_hi, p_value)."""
    rng = np.random.default_rng(seed)
    obs_hits = _hits(scores, true)
    observed = _ac1_per_patient(obs_hits, patients)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(true)
        null[b] = _ac1_per_patient(_hits(scores, perm), patients)
    p = (1 + int(np.sum(null >= observed))) / (n_perm + 1)
    lo, hi = np.percentile(null, [2.5, 97.5])
    return observed, float(null.mean()), float(lo), float(hi), float(p)


def random_baseline_ac1(cached, n=2000, seed=0):
    """Empirical chance: rank channels uniformly at random."""
    rng = np.random.default_rng(seed)
    true, pats, C = cached["true"], cached["patient"], cached["C"]
    accs = []
    for _ in range(n):
        pred = rng.integers(0, C, size=len(true))
        accs.append(_ac1_per_patient((pred == true).astype(float), pats))
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# top-level report
# ---------------------------------------------------------------------------


def run_rigor(df, windows, attributors, channels=CORE_CHANNELS, n_boot=2000, n_perm=2000,
              verbose=True):
    """Score each attributor once, then report AC@1 [95% CI], the permutation-null
    AC@1, the lift above null, and a permutation p-value. Also detection-conditioned
    AC@1. Returns a tidy DataFrame."""
    rows = []
    rand_ac1 = None
    for A in attributors:
        c = score_windows(df, windows, A, channels=channels)
        if rand_ac1 is None:
            rand_ac1 = random_baseline_ac1(c)
        hits = _hits(c["scores"], c["true"])
        ac1, lo, hi = bootstrap_ci(hits, c["patient"], n_boot=n_boot)
        obs, null_mu, nlo, nhi, p = permutation_null(c["scores"], c["true"], c["patient"], n_perm=n_perm)
        det = c["detected"]
        cond_hits = hits[det]
        cond_pat = c["patient"][det]
        cond_ac1 = _ac1_per_patient(cond_hits, cond_pat) if det.sum() else float("nan")
        rows.append({
            "Method": c["method"], "AC@1": round(ac1, 3),
            "CI95": f"[{lo:.3f}, {hi:.3f}]",
            "perm_null_AC@1": round(null_mu, 3),
            "lift_over_null": round(ac1 - null_mu, 3),
            "perm_p": ("<0.001" if p < 1e-3 else round(p, 4)),
            "AC@1|detected": round(cond_ac1, 3),
            "chance(1/C)": round(c["chance"], 3),
            "n_windows": len(c["true"]),
        })
    out = pd.DataFrame(rows)
    if verbose:
        print(f"[rigor] empirical random-baseline AC@1 = {rand_ac1:.3f} "
              f"(theoretical chance 1/C = {out['chance(1/C)'].iloc[0]:.3f})")
        print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# enrichment-tier path (the HONEST floor-lowering) -- documentation + helper
# ---------------------------------------------------------------------------


def enrichment_note():
    return (
        "Enrichment-tier attribution (the only honest way to push chance below 0.50 "
        "on this data):\n"
        "  * Generate windows for the enrichment records (220 +ABP, 178 +RESP) with the "
        "extra channels retained, injecting into ECG/PPG as usual so ABP/RESP stay "
        "untampered.  The window tensor is then [N, C, T] with C=3 or 4.\n"
        "  * Run the SAME attributors + this rigor pack on those windows: candidate "
        "set = the C real channels, chance = 1/C (0.33 or 0.25), and the untampered "
        "real channels are automatic negative controls -- no decoy padding.\n"
        "  * The permutation control and CIs here apply unchanged.\n"
        "This requires one generation pass on the enrichment records; the code is "
        "ready (run_rigor is channel-count agnostic)."
    )


# ---------------------------------------------------------------------------
# demo: (a) rigor on the 2-channel core, (b) why decoy padding is trivial
# ---------------------------------------------------------------------------


def demo_decoy_is_trivial(df, windows, seed=0):
    """Show that padding the candidate set with clean-segment decoys lets a
    Deviation ranker 'solve' a fake low-chance task -- i.e. it is gamed, and the
    triviality context catches it. Educational; not a result to report."""
    rng = np.random.default_rng(seed)
    windows = np.asarray(windows, float)
    cls = df["cls"].astype(str).values
    clean_idx = np.where(cls == "physiology_normal")[0]
    A = V.MarginalDeviationBaseline().fit(windows[clean_idx])
    # score the 2 real channels of an attack window + K clean-decoy channels
    fam = df["family"].astype(str).values
    tgt = df["target_channel"].values
    atk = np.where(np.isin(fam, ["fdi", "spoofing", "replay"]))[0]
    K = 12  # pad to 14 candidates -> fake chance 1/14 = 0.071
    hits = []
    for i in atk[:500]:
        t = V._resolve_target_index(tgt[i], CORE_CHANNELS)
        if t is None:
            continue
        real, _ = A.attribute(windows[i])            # 2 real-channel deviation scores
        decoys = rng.choice(clean_idx, size=K)
        dsc = np.array([A.attribute(windows[d])[0].max() for d in decoys])  # clean -> low
        scores = np.concatenate([real, dsc])
        hits.append(int(np.argmax(scores) == t))
    ac1 = float(np.mean(hits))
    print(f"[decoy demo] padded to 14 candidates (fake chance 0.071); "
          f"Deviation ranker AC@1 = {ac1:.3f}")
    print("  -> a trivial ranker 'solves' the padded task because clean decoys never "
          "deviate; the low chance floor is FAKE. This is why decoy padding is rejected "
          "and only real extra channels (enrichment) count.")
    return ac1


def _demo():
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        df, windows, _ = V._demo(n_records=20, seed=1)
    # multi-patient test split so bootstrap CIs are non-degenerate (real data has 94)
    test_pat = sorted(df["patient_id"].unique())[-2:]
    df = df.copy()
    df["split"] = np.where(df["patient_id"].isin(test_pat), "test", "train")
    print("=== RQ2 rigor pack on the 2-channel core (chance 0.500) ===")
    run_rigor(df, windows,
              [V.MarginalDeviationBaseline(), V.VARResidualBaseline(), V.PCMCIAttributor()],
              n_boot=1000, n_perm=1000)
    print()
    demo_decoy_is_trivial(df, windows)
    print()
    print(enrichment_note())


if __name__ == "__main__":
    _demo()
