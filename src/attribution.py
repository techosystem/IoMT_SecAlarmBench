"""
Channel-level causal attribution for IoMT-SecAlarmBench.

Ranks the channels of each attacked window by how likely each is to be the injected
one. Two causal attributors:

  * PCMCIAttributor: discovers the lagged causal parents with tigramite PCMCI+
    (ParCorr CI tests), scores each channel by link-persistence residual on the
    discovered graph, then walks up to the most-upstream anomalous node.
  * GCMAttributor (default): fits a DoWhy-GCM structural causal model over the
    per-channel residuals on the discovered graph and calls
    ``gcm.attribute_anomalies`` (counterfactual Shapley attribution), with the same
    upstream root-cause step.

Two non-causal baselines:

  * MarginalDeviationBaseline: coupling-blind marginal-deviation ranking.
  * VARResidualBaseline: ridge-VAR residual ranking (Granger-style, no CI-test
    discovery; can blame a child of the tampered channel).

Metrics, per patient (primary) and pooled:
    AC@k  = (1/N) sum 1[rank(true) <= k]
    Avg@k = (1/N) sum (1/k) sum_{j<=k} AC@j
    MRR   = (1/N) sum 1/rank(true)
plus AC@1 conditioned on the window being detected.

Inputs match eval_harness: df (labels, row-aligned) and windows [N, C, T]. Every
attributor is fitted on training physiology-normal windows only.

Dependencies: numpy, pandas, scikit-learn; tigramite, dowhy and networkx for the two
causal methods (each falls back with a warning if missing). The names
MarginalDeviationAttributor, CausalResidualAttributor and GCMShapleyAttributor are
kept as aliases.

The paper reports a simpler channel-attribution baseline
(extras.attribution_baseline); these causal attributors are provided for follow-up
work.
"""

from __future__ import annotations

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

CORE_CHANNELS = ("ECG", "PPG")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pool(X, factor):
    if factor is None or factor <= 1:
        return X
    T = X.shape[-1]
    T2 = (T // factor) * factor
    X = X[..., :T2]
    return X.reshape(X.shape[:-1] + (T2 // factor, factor)).mean(axis=-1)


def _resolve_target_index(value, channels):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if 0 <= int(value) < len(channels) else None
    if isinstance(value, float):
        if np.isnan(value):
            return None
        v = int(value)
        return v if 0 <= v < len(channels) else None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    up = {c.upper(): i for i, c in enumerate(channels)}
    if s.upper() in up:
        return up[s.upper()]
    alias = {"PLETH": "PPG", "II": "ECG", "I": "ECG", "V": "ECG", "MCL": "ECG"}
    if s.upper() in alias and alias[s.upper()].upper() in up:
        return up[alias[s.upper()].upper()]
    return None


def _parent_design(X, parents, tau):
    """Design matrix predicting each channel from its (source, lag) parents.
    X:[C,T] -> dict c -> (features[len(tt), n_parents], target y, tt)."""
    C, T = X.shape
    tt = np.arange(tau, T)
    out = {}
    for c in range(C):
        par = parents[c]
        feats = [X[s, tt - l] for (s, l) in par] if par else []
        Xc = np.stack(feats, -1) if feats else np.zeros((len(tt), 0))
        out[c] = (Xc, X[c, tt], tt)
    return out


def _upstream_encode(residuals, thr, order, C):
    """Turn per-channel residuals into a final ranking score implementing the
    RCA 'most-upstream anomalous node first' rule.

    Channels whose residual exceeds their train threshold are 'anomalous'; among
    them the most upstream (smallest topological order) ranks first.  Non-
    anomalous channels rank below, by residual.  Encoded numerically so a plain
    descending argsort (and the rank_of() metric) reproduces this ordering.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    anom = residuals >= thr
    BIG = 1e6
    score = np.zeros(C)
    for c in range(C):
        if anom[c]:
            score[c] = BIG - order[c] * 10.0 + residuals[c] * 1e-3  # upstream first
        else:
            score[c] = residuals[c]  # below all anomalous
    return score


# ---------------------------------------------------------------------------
# attributor base
# ---------------------------------------------------------------------------


class Attributor:
    name = "base"
    is_causal = False

    def fit(self, W):  # pragma: no cover
        raise NotImplementedError

    def attribute(self, x):  # pragma: no cover
        raise NotImplementedError

    # optional: attribute_batch(W) -> (scores[N,C], anomaly[N]) for speed


# ---------------------------------------------------------------------------
# BASELINES (relabeled hand-derived methods; NOT the library estimators)
# ---------------------------------------------------------------------------


class MarginalDeviationBaseline(Attributor):
    """BASELINE (not causal): rank each channel by mean |z| from its own train-
    normal marginal.  Coupling-blind; localizes large out-of-range FDI/spoofing,
    blind to replay.  Stands in for the RootCLAM-style anomaly-magnitude idea.
    The causal methods are expected to beat this on coupling attacks."""

    name = "MarginalDeviation(baseline)"

    def __init__(self, pool=5):
        self.pool = pool
        self.mu_ = None
        self.sd_ = None

    def fit(self, W):
        Wp = _pool(np.asarray(W, float), self.pool)
        self.mu_ = Wp.mean(axis=(0, 2))
        self.sd_ = Wp.std(axis=(0, 2)) + 1e-8
        return self

    def attribute(self, x):
        xp = _pool(np.asarray(x, float), self.pool)
        z = np.abs((xp - self.mu_[:, None]) / self.sd_[:, None])
        s = z.mean(axis=1)
        return s, float(s.max())


class VARResidualBaseline(Attributor):
    """BASELINE (not PCMCI): ridge-VAR cross+within residual ranking, no CI-test
    discovery.  Ranks each channel by normalized residual of predicting it from
    the other channels (cross) and its own past (within).  Known weakness, kept
    on purpose: tampering an UPSTREAM channel inflates its child's residual, so
    raw residual ranking mis-attributes to the child (the propagation problem
    that PCMCI/GCM's graph-aware attribution is meant to fix)."""

    name = "VARResidual(baseline)"

    def __init__(self, pool=5, p=4, alpha=1.0):
        self.pool = pool
        self.p = p
        self.alpha = alpha
        self.win_ = {}
        self.cross_ = {}
        self.wv_ = None
        self.cv_ = None
        self.C_ = None

    def _design(self, W):
        N, C, T = W.shape
        tt = np.arange(self.p, T)
        d = {}
        for c in range(C):
            Xw = np.stack([W[:, c, tt - l] for l in range(1, self.p + 1)], -1).reshape(-1, self.p)
            cc = []
            for j in range(C):
                if j == c:
                    continue
                for l in range(0, self.p + 1):
                    cc.append(W[:, j, tt - l])
            Xc = np.stack(cc, -1).reshape(-1, len(cc)) if cc else np.zeros((Xw.shape[0], 0))
            d[c] = (Xw, Xc, W[:, c, tt].reshape(-1))
        return d

    def fit(self, W):
        Wp = _pool(np.asarray(W, float), self.pool)
        self.C_ = Wp.shape[1]
        d = self._design(Wp)
        self.wv_ = np.ones(self.C_)
        self.cv_ = np.ones(self.C_)
        for c in range(self.C_):
            Xw, Xc, y = d[c]
            mw = Ridge(self.alpha).fit(Xw, y)
            self.win_[c] = mw
            self.wv_[c] = np.mean((y - mw.predict(Xw)) ** 2) + 1e-8
            if Xc.shape[1]:
                mc = Ridge(self.alpha).fit(Xc, y)
                self.cross_[c] = mc
                self.cv_[c] = np.mean((y - mc.predict(Xc)) ** 2) + 1e-8
            else:
                self.cross_[c] = None
        return self

    def attribute(self, x):
        xp = _pool(np.asarray(x, float), self.pool)[None]
        d = self._design(xp)
        s = np.zeros(self.C_)
        cross = np.zeros(self.C_)
        for c in range(self.C_):
            Xw, Xc, y = d[c]
            wn = np.mean((y - self.win_[c].predict(Xw)) ** 2) / self.wv_[c]
            cn = 0.0
            if self.cross_[c] is not None and Xc.shape[1]:
                cn = np.mean((y - self.cross_[c].predict(Xc)) ** 2) / self.cv_[c]
            cross[c] = cn
            s[c] = cn + wn
        return s, float(cross.max() if self.C_ > 1 else s.max())


# ---------------------------------------------------------------------------
# PCMCI (tigramite) -- real discovery + link-persistence + upstream root cause
# ---------------------------------------------------------------------------


class PCMCIAttributor(Attributor):
    """Discover lagged parents with tigramite PCMCI+, then attribute by link-
    persistence residual on the DISCOVERED graph, resolving anomaly propagation
    with the 'most-upstream anomalous node' rule."""

    name = "PCMCI+(tigramite)"
    is_causal = True

    def __init__(self, pool=5, tau_max=3, pc_alpha=0.05, max_disc_windows=120, alpha=1.0):
        self.pool = pool
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.max_disc = max_disc_windows
        self.alpha = alpha
        self.parents_ = None
        self.order_ = None
        self.models_ = {}
        self.rvar_ = None
        self.thr_ = None
        self.C_ = None
        self.discovered_ = False

    def _discover(self, Wp):
        """Return parents{c:[(src,lag)]} and a channel topological order."""
        C = Wp.shape[1]
        try:
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
            import tigramite.data_processing as tp
            import networkx as nx

            sub = Wp[: self.max_disc]
            series = np.concatenate([w.T for w in sub], axis=0)  # [T_total, C]
            dataframe = tp.DataFrame(series, var_names=[f"c{c}" for c in range(C)])
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
            res = pcmci.run_pcmciplus(tau_min=1, tau_max=self.tau_max, pc_alpha=self.pc_alpha)
            graph = res["graph"]  # [C, C, tau+1], entries like '-->','o-o',''
            parents = {c: [] for c in range(C)}
            lagged_influence = np.zeros((C, C))  # src -> tgt strength (for direction)
            for tgt in range(C):
                for src in range(C):
                    for lag in range(1, graph.shape[2]):
                        g = str(graph[src, tgt, lag])
                        if g.endswith(">"):  # directed link src(t-lag) -> tgt(t)
                            parents[tgt].append((src, lag))
                            lagged_influence[src, tgt] += 1
            # ensure identifiability floor: each channel keeps its own lag-1
            for c in range(C):
                if (c, 1) not in parents[c]:
                    parents[c].append((c, 1))
            # channel-level DAG for topo order: src upstream of tgt if net lagged
            # influence src->tgt exceeds tgt->src
            g = nx.DiGraph()
            g.add_nodes_from(range(C))
            for a in range(C):
                for b in range(C):
                    if a == b:
                        continue
                    if lagged_influence[a, b] > lagged_influence[b, a]:
                        g.add_edge(a, b)
            try:
                topo = list(nx.topological_sort(g))
            except Exception:
                topo = list(range(C))
            order = {c: i for i, c in enumerate(topo)}
            self.discovered_ = True
            return parents, order
        except Exception as e:
            warnings.warn(f"[PCMCI] tigramite unavailable/failed ({e}); "
                          f"falling back to full-lag parents + index order.")
            parents = {c: [(j, l) for j in range(C) for l in range(1, self.tau_max + 1)]
                       for c in range(C)}
            order = {c: c for c in range(C)}
            self.discovered_ = False
            return parents, order

    def fit(self, W):
        Wp = _pool(np.asarray(W, float), self.pool)
        self.C_ = Wp.shape[1]
        self.parents_, self.order_ = self._discover(Wp)
        # fit link-persistence models on all train-normal windows
        self.rvar_ = np.ones(self.C_)
        per_ch_res = {c: [] for c in range(self.C_)}
        Xs = {c: [] for c in range(self.C_)}
        ys = {c: [] for c in range(self.C_)}
        for w in Wp:
            d = _parent_design(w, self.parents_, self.tau_max)
            for c in range(self.C_):
                Xc, y, _ = d[c]
                Xs[c].append(Xc)
                ys[c].append(y)
        for c in range(self.C_):
            X = np.concatenate(Xs[c]); y = np.concatenate(ys[c])
            m = Ridge(self.alpha).fit(X, y)
            self.models_[c] = m
            self.rvar_[c] = np.mean((y - m.predict(X)) ** 2) + 1e-8
        # per-channel normalized-residual threshold from train-normal windows
        norm_res = np.array([self._residuals(w) for w in Wp])  # [N, C]
        self.thr_ = np.quantile(norm_res, 0.95, axis=0)
        return self

    def _residuals(self, x):
        d = _parent_design(x, self.parents_, self.tau_max)
        r = np.zeros(self.C_)
        for c in range(self.C_):
            Xc, y, _ = d[c]
            if Xc.shape[1]:
                r[c] = np.mean((y - self.models_[c].predict(Xc)) ** 2) / self.rvar_[c]
            else:
                r[c] = 0.0
        return r

    def attribute(self, x):
        xp = _pool(np.asarray(x, float), self.pool)
        r = self._residuals(xp)
        score = _upstream_encode(r, self.thr_, self.order_, self.C_)
        return score, float(r.max())


# ---------------------------------------------------------------------------
# DoWhy-GCM -- real gcm.attribute_anomalies (Budhathoki et al.), batched
# ---------------------------------------------------------------------------


class GCMAttributor(Attributor):
    """Real DoWhy-GCM counterfactual anomaly attribution.

    Uses PCMCIAttributor internally to (a) discover the channel graph and (b)
    reduce each window to per-channel link-persistence residuals.  Fits a
    StructuralCausalModel over those residual nodes on the DISCOVERED contemporaneous
    DAG (train-normal), then calls `gcm.attribute_anomalies` on the topological
    sink and combines node contributions with the upstream root-cause rule.
    Batched: one gcm call scores all test windows.
    """

    name = "DoWhy-GCM(Shapley)"
    is_causal = True

    def __init__(self, pool=5, tau_max=3, pc_alpha=0.05, max_disc_windows=120,
                 num_dist_samples=800, chunk=512, verbose=True):
        # num_dist_samples: DoWhy's Monte-Carlo budget per anomaly sample.
        #   DoWhy defaults to 3000 (~260 ms/window -> ~28 min & silent on 6.4k
        #   windows, which reads as a freeze).  800 is ~2 min with no meaningful
        #   change in the ranking; 300 is ~1 min if you need it faster.
        # chunk: windows per attribute_anomalies call, with a heartbeat print so
        #   the run is never silent again (and memory stays bounded).
        self.pcmci = PCMCIAttributor(pool=pool, tau_max=tau_max, pc_alpha=pc_alpha,
                                     max_disc_windows=max_disc_windows)
        self.num_dist_samples = num_dist_samples
        self.chunk = chunk
        self.verbose = verbose
        self.C_ = None
        self.cols_ = None
        self.sink_ = None
        self.order_ = None
        self.thr_ = None
        self._scm = None
        self._ok = False

    def fit(self, W):
        self.pcmci.fit(W)
        self.C_ = self.pcmci.C_
        self.order_ = self.pcmci.order_
        self.thr_ = self.pcmci.thr_
        Wp = _pool(np.asarray(W, float), self.pcmci.pool)
        F = np.array([self.pcmci._residuals(w) for w in Wp])  # [N, C] residual nodes
        self.cols_ = [f"c{c}" for c in range(self.C_)]
        # sink = most-downstream channel in discovered order
        self.sink_ = max(range(self.C_), key=lambda c: self.order_[c])
        try:
            import networkx as nx
            from dowhy import gcm
            gcm.config.disable_progress_bars()
            # contemporaneous DAG from the discovered channel order (chain along order)
            topo = sorted(range(self.C_), key=lambda c: self.order_[c])
            g = nx.DiGraph()
            g.add_nodes_from(self.cols_)
            for a, b in zip(topo[:-1], topo[1:]):
                g.add_edge(self.cols_[a], self.cols_[b])
            data = pd.DataFrame(F, columns=self.cols_)
            scm = gcm.StructuralCausalModel(g)
            gcm.auto.assign_causal_mechanisms(scm, data)
            gcm.fit(scm, data)
            self._scm = scm
            self._ok = True
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[GCM] dowhy unavailable/failed ({e}); "
                          f"ranking by residual + upstream rule instead.")
            self._ok = False
        return self

    def _feats(self, W):
        Wp = _pool(np.asarray(W, float), self.pcmci.pool)
        return np.array([self.pcmci._residuals(w) for w in Wp])  # [N, C]

    def magnitudes(self, W):
        """Cheap anomaly magnitude (max residual) WITHOUT running gcm — used for
        the detection threshold so DoWhy is never called on the train-normal set."""
        return self._feats(W).max(axis=1)

    def attribute_batch(self, W):
        F = self._feats(W)                      # [N, C] residual nodes
        N = F.shape[0]
        anomaly = F.max(axis=1)
        if self._ok:
            from dowhy import gcm
            contrib = np.zeros((N, self.C_))
            t0 = time.time()
            for start in range(0, N, self.chunk):
                end = min(start + self.chunk, N)
                samples = pd.DataFrame(F[start:end], columns=self.cols_)
                attr = gcm.attribute_anomalies(
                    self._scm, target_node=self.cols_[self.sink_],
                    anomaly_samples=samples,
                    num_distribution_samples=self.num_dist_samples)
                contrib[start:end] = np.stack([np.asarray(attr[c]) for c in self.cols_], axis=1)
                if self.verbose:
                    el = time.time() - t0
                    eta = el / end * (N - end)
                    print(f"    [GCM] {end}/{N} windows  ({el:.0f}s elapsed, ~{eta:.0f}s left)",
                          flush=True)
        else:
            contrib = F
        # combine gcm contribution magnitude with the upstream root-cause rule,
        # using each window's residuals to decide the 'anomalous' set
        scores = np.zeros((N, self.C_))
        for i in range(N):
            base = _upstream_encode(F[i], self.thr_, self.order_, self.C_)
            # break ties inside the anomalous/non-anomalous tiers by gcm contribution
            scores[i] = base + 1e-6 * np.abs(contrib[i])
        return scores, anomaly

    def attribute(self, x):  # single-window fallback (slow path)
        s, a = self.attribute_batch(np.asarray(x, float)[None])
        return s[0], float(a[0])


def register_rcaeval(name="RCD"):  # pragma: no cover
    """Stretch hook: wrap RCAEval's RCD / CIRCA behind the Attributor interface
    (fit on train-normal, attribute per window).  Left unimplemented (heavy dep)."""
    raise NotImplementedError("Install `RCAEval` and wrap its ranker as an Attributor.")


# backward-compatible aliases (note: GCMShapley now maps to the REAL DoWhy GCM)
MarginalDeviationAttributor = MarginalDeviationBaseline
CausalResidualAttributor = VARResidualBaseline
GCMShapleyAttributor = GCMAttributor


# ---------------------------------------------------------------------------
# ranking + metric primitives (unchanged, tested)
# ---------------------------------------------------------------------------


def _rank_of(scores, true_idx):
    s = np.asarray(scores, float)
    t = s[true_idx]
    greater = int(np.sum(s > t))
    equal = int(np.sum(s == t))
    return greater + (equal + 1) / 2.0


def _ac_at_k(ranks, k):
    ranks = np.asarray(ranks, float)
    return float(np.mean(ranks <= k)) if ranks.size else float("nan")


def _avg_at_k(ranks, k):
    return float(np.mean([_ac_at_k(ranks, j) for j in range(1, k + 1)])) if len(ranks) else float("nan")


def _mrr(ranks):
    ranks = np.asarray(ranks, float)
    return float(np.mean(1.0 / ranks)) if ranks.size else float("nan")


def _metric_block(ranks, ncand):
    return {"AC@1": _ac_at_k(ranks, 1), "AC@3": _ac_at_k(ranks, min(3, ncand)),
            "Avg@5": _avg_at_k(ranks, min(5, ncand)), "MRR": _mrr(ranks), "n": int(len(ranks))}


def _per_patient(ranks, patients, ncand):
    ranks = np.asarray(ranks, float); patients = np.asarray(patients)
    a1, a3, a5, mr = [], [], [], []
    for pid in pd.unique(patients):
        r = ranks[patients == pid]
        if r.size == 0:
            continue
        a1.append(_ac_at_k(r, 1)); a3.append(_ac_at_k(r, min(3, ncand)))
        a5.append(_avg_at_k(r, min(5, ncand))); mr.append(_mrr(r))
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return {"AC@1": f(a1), "AC@3": f(a3), "Avg@5": f(a5), "MRR": f(mr), "n_patients": len(a1)}


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run_rq2(df, windows, attributor, out_dir=None, channels=CORE_CHANNELS,
            families=("fdi", "spoofing", "replay"), detect_quantile=0.95,
            extra_candidate_scores=None, verbose=True):
    """Run per-patient channel attribution for one attributor. See module docstring."""
    windows = np.asarray(windows, float)
    N, C, T = windows.shape
    if len(channels) != C:
        channels = tuple(f"ch{i}" for i in range(C))
    n_extra = 0 if extra_candidate_scores is None else int(np.asarray(extra_candidate_scores).shape[1])
    ncand = C + n_extra

    split = df["split"].values if "split" in df.columns else np.array(["train"] * N)
    cls = df["cls"].astype(str).values
    fam = df["family"].astype(str).values if "family" in df.columns else np.array([""] * N)
    morph = df["morphology"].astype(str).values if "morphology" in df.columns else np.array([""] * N)
    tgt = df["target_channel"].values if "target_channel" in df.columns else np.array([None] * N)
    pid_col = "patient_id" if "patient_id" in df.columns else ("record_id" if "record_id" in df.columns else None)
    pid = df[pid_col].astype(str).values if pid_col else np.array(["p0"] * N)

    is_norm = cls == "physiology_normal"
    train_norm = (split == "train") & is_norm
    if train_norm.sum() < 10:
        train_norm = is_norm
    is_attack = np.isin(fam, list(families))
    test_attack = (split == "test") & is_attack
    if test_attack.sum() == 0:
        test_attack = is_attack

    if verbose:
        print(f"[rq2/{attributor.name}] candidates={ncand} ({C} ch"
              f"{'' if n_extra == 0 else f' + {n_extra} distractors'}) | "
              f"train-normal={int(train_norm.sum())} | test-attack={int(test_attack.sum())} "
              f"| chance AC@1={1.0/ncand:.3f}")

    t0 = time.time()
    attributor.fit(windows[train_norm])
    fit_s = time.time() - t0
    if getattr(attributor, "is_causal", False) and hasattr(attributor, "pcmci"):
        disc = "discovered" if attributor.pcmci.discovered_ else "FALLBACK(no tigramite)"
        if verbose:
            print(f"    graph: {disc}  parents={ {c: attributor.pcmci.parents_[c] for c in range(C)} }")
    elif getattr(attributor, "is_causal", False) and hasattr(attributor, "discovered_"):
        if verbose:
            print(f"    graph: {'discovered' if attributor.discovered_ else 'FALLBACK'}  "
                  f"parents={attributor.parents_}")

    # detection threshold from train-normal magnitudes
    def _mag(W):
        if hasattr(attributor, "magnitudes"):          # cheap path (no gcm on train)
            return attributor.magnitudes(W)
        if hasattr(attributor, "attribute_batch"):
            return attributor.attribute_batch(W)[1]
        return np.array([attributor.attribute(w)[1] for w in W])

    tn_mag = _mag(windows[train_norm])
    thr = float(np.quantile(tn_mag, detect_quantile))

    idx = np.where(test_attack)[0]
    # score test-attack windows (batched if supported)
    t1 = time.time()
    if hasattr(attributor, "attribute_batch"):
        sc_all, mag_all = attributor.attribute_batch(windows[idx])
    else:
        sc_all = np.zeros((len(idx), C)); mag_all = np.zeros(len(idx))
        for k, i in enumerate(idx):
            sc_all[k], mag_all[k] = attributor.attribute(windows[i])
    attr_ms = 1000.0 * (time.time() - t1) / max(1, len(idx))

    ranks = np.full(len(idx), np.nan); flagged = np.zeros(len(idx), bool); skipped = 0
    for k, i in enumerate(idx):
        ti = _resolve_target_index(tgt[i], channels)
        if ti is None:
            skipped += 1; continue
        sc = sc_all[k]
        if n_extra:
            sc = np.concatenate([sc, np.asarray(extra_candidate_scores[i], float)])
        ranks[k] = _rank_of(sc, ti)
        flagged[k] = mag_all[k] > thr

    keep = ~np.isnan(ranks)
    rv, fv, mv, pv, flv = ranks[keep], fam[idx][keep], morph[idx][keep], pid[idx][keep], flagged[keep]

    overall_micro = _metric_block(rv, ncand)
    overall_pp = _per_patient(rv, pv, ncand)
    by_family = {f: {"micro": _metric_block(rv[fv == f], ncand),
                     "per_patient": _per_patient(rv[fv == f], pv[fv == f], ncand)}
                 for f in sorted(set(fv))}
    by_morph = {mm: _metric_block(rv[mv == mm], ncand) for mm in sorted(set(mv))
                if mm not in ("", "nan", "None")}
    det_rate = float(np.mean(flv)) if flv.size else float("nan")
    cond_ac1 = _ac_at_k(rv[flv], 1) if flv.sum() else float("nan")
    cond_mrr = _mrr(rv[flv]) if flv.sum() else float("nan")

    summary = {
        "method": attributor.name, "n_candidates": ncand, "n_channels": C,
        "n_distractors": n_extra, "chance_AC@1": 1.0 / ncand,
        "graph_discovered": bool(getattr(getattr(attributor, "pcmci", attributor), "discovered_", False)),
        "fit_seconds": round(fit_s, 3), "attr_ms_per_window": round(attr_ms, 4),
        "n_attack_windows": int(len(rv)), "skipped_no_target": int(skipped),
        "per_patient": overall_pp, "micro": overall_micro,
        f"detection_rate@q{detect_quantile:.2f}": det_rate,
        "attr_given_detected_AC@1": cond_ac1, "attr_given_detected_MRR": cond_mrr,
        "by_family": by_family, "by_morphology": by_morph,
    }

    if verbose:
        pp = overall_pp
        print(f"[rq2/{attributor.name}] per-patient AC@1={pp['AC@1']:.3f} AC@3={pp['AC@3']:.3f} "
              f"Avg@5={pp['Avg@5']:.3f} MRR={pp['MRR']:.3f}  (chance AC@1={1.0/ncand:.3f})")
        print(f"[rq2/{attributor.name}] detection={det_rate:.3f} | AC@1|detected={cond_ac1:.3f} "
              f"| fit={fit_s:.1f}s attr={attr_ms:.3f}ms/win")
        for f in sorted(by_family):
            b = by_family[f]["per_patient"]
            print(f"    {f:9s} AC@1={b['AC@1']:.3f} MRR={b['MRR']:.3f} (n_pat={b['n_patients']})")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        m = attributor.name.replace("/", "_").replace("(", "_").replace(")", "").replace("+", "p").replace(" ", "")
        row = {"Method": attributor.name, **{k: overall_pp[k] for k in ("AC@1", "AC@3", "Avg@5", "MRR")},
               "chance_AC@1": 1.0 / ncand, "attr_given_detected_AC@1": cond_ac1, "detection_rate": det_rate}
        pd.DataFrame([row]).to_csv(os.path.join(out_dir, f"rq2_{m}.csv"), index=False)
        pd.DataFrame([{"Method": attributor.name, "family": f,
                       **{k: by_family[f]["per_patient"][k] for k in ("AC@1", "AC@3", "Avg@5", "MRR")},
                       "n_patients": by_family[f]["per_patient"]["n_patients"]} for f in by_family]
                     ).to_csv(os.path.join(out_dir, f"rq2_byfamily_{m}.csv"), index=False)
        pd.DataFrame([{"Method": attributor.name, "morphology": k2, **v} for k2, v in by_morph.items()]
                     ).to_csv(os.path.join(out_dir, f"rq2_bymorph_{m}.csv"), index=False)
        with open(os.path.join(out_dir, f"rq2_{m}_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary


# ---------------------------------------------------------------------------
# demo / smoke test
# ---------------------------------------------------------------------------


def _demo(n_records=10, fs=125, seconds=10, seed=0):
    rng = np.random.default_rng(seed)
    T = fs * seconds
    rows, wins, wid = [], [], 0

    def coupled_pair():
        # PPG driven by lagged ECG -> a discoverable directed coupling
        ecg = np.zeros(T); ppg = np.zeros(T)
        a = rng.uniform(0.5, 0.7)
        for t in range(2, T):
            ecg[t] = 0.6 * ecg[t - 1] + rng.normal(0, 1)
            ppg[t] = 0.5 * ppg[t - 1] + a * ecg[t - 1] + rng.normal(0, 0.5)
        x = np.stack([ecg, ppg])
        return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)

    def add(cls, fam, mo, tgt, lv, x, rid, pid):
        nonlocal wid
        rows.append(dict(window_id=wid, record_id=rid, patient_id=pid, split="",
                         cls=cls, family=fam, morphology=mo, target_channel=tgt, level=lv))
        wins.append(x.copy()); wid += 1

    for r in range(n_records):
        rid, pid = f"rec{r:03d}", f"p{r % 5:03d}"
        for _ in range(8):
            add("physiology_normal", "", "", None, np.nan, coupled_pair(), rid, pid)
        for h in range(3):
            base = coupled_pair(); ti = h % 2; tn = CORE_CHANNELS[ti]
            sl = slice(T // 3, 2 * T // 3)
            for mo in ("bias", "spike", "drift", "stuck_at"):
                x = base.copy()
                if mo == "bias": x[ti, sl] += 1.5
                elif mo == "spike": x[ti, T // 2] += 4.0
                elif mo == "drift": x[ti, sl] += np.linspace(0, 2, sl.stop - sl.start)
                else: x[ti, sl] = x[ti, sl.start]
                add("attack", "fdi", mo, tn, 2, x, rid, pid)
            x = base.copy(); x[ti, sl] += 8.0
            add("attack", "spoofing", "spike", tn, 3, x, rid, pid)
            past = coupled_pair(); x = base.copy(); x[ti, sl] = past[ti, sl]
            add("attack", "replay", "segment_sub", tn, 0, x, rid, pid)

    df = pd.DataFrame(rows); windows = np.stack(wins)
    tp = sorted(df["patient_id"].unique())[-1:]
    df["split"] = np.where(df["patient_id"].isin(tp), "test", "train")
    print(f"[demo] windows={windows.shape} classes={df['cls'].value_counts().to_dict()}\n")
    outs = {}
    for A in (MarginalDeviationBaseline(), VARResidualBaseline(),
              PCMCIAttributor(), GCMAttributor()):
        outs[A.name] = run_rq2(df, windows, A, out_dir=None, verbose=True); print()
    return df, windows, outs


if __name__ == "__main__":
    _demo()
