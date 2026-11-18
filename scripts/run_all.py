#!/usr/bin/env python3
"""
Single-setting (C1-PPG) pipeline for IoMT-SecAlarmBench.

Regenerates the C1-PPG detection, matched-budget, multi-seed, discrimination and
leakage results into a timestamped results directory, with a manifest that records
the configuration, package versions and source-file hashes. The four-setting
analyses reported in the paper are run by scripts/reproduce_paper.py.

Usage:
    python scripts/run_all.py --self-test        # wiring check on synthetic records
    python scripts/run_all.py                    # default stages
    python scripts/run_all.py --stages rq1 --detectors IsolationForest,USAD
    python scripts/run_all.py --stages config    # design constants, no data needed

Or from Python:
    import run_all; res = run_all.main(["--stages", "rq1", "--detectors", "IsolationForest"])

Stages:
  config     design constants (severity schedule and plausibility bounds) as CSVs,
             read from injection_generator; no data, no GPU
  checks     invariant checks on the built table (classes, keys, counts, prevalence)
  table2     dataset composition
  rq1        detection and per-morphology results for every detector
             (detection_report)
  matched    threshold-free AUPRC per attack subset and detection at a matched
             false-alarm budget
  multiseed  each detector over several seeds, mean +/- sd
  rq3        3-way and 4-way discrimination
  wustl      identifier-leakage audit
  ablation   LSTM-AE without the pooling front-end (pool=1)
  summary    SUMMARY.md with all tables

Default: config, checks, table2, rq1, matched, rq3, wustl, summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import sys
import tempfile
import time
from dataclasses import MISSING, asdict, fields as dc_fields, is_dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

# --------------------------------------------------------------------------- #
# CONFIG — edit here, or override on the command line
# --------------------------------------------------------------------------- #

REPO = os.environ.get(
    "IOMT_REPO",
    str(pathlib.Path(__file__).resolve().parent.parent),
)

CONFIG = dict(
    repo=REPO,
    cinc_dir=f"{REPO}/data/cinc2015/training",
    wustl_csv=f"{REPO}/data/wustl/wustl-ehms-2020.csv",
    results_root=f"{REPO}/results",
    cache_dir=str(pathlib.Path(tempfile.gettempdir()) / "iomt_cache"),

    # --- generation (changing any of these changes every reported number) ---
    seed=20260723,
    target_channels=("PPG",),                 # ("ECG","PPG") makes channel attribution non-trivial
    win_sec=10.0,
    hop_sec=5.0,
    max_hosts_per_record=2,
    artifact_frac=0.10,
    core_channels=("ECG", "PPG"),
    limit_records=None,                       # int, for a fast partial run
    fs=125,                                   # core-cluster rate; seconds column of Table III

    # --- detectors: ONE uniform training budget for all of them ---
    epochs=40,
    detectors=("IsolationForest", "LSTM-AE", "USAD", "GDN", "TranAD", "OmniAnomaly"),

    # --- reporting ---
    n_boot=2000,
    ci=95,
    target_prevalence=0.05,
    n_prev_rep=200,

    # --- ablation ---
    ablation_pools=(1,),
    ablation_batch=64,

    # --- matched-FAR + multi-seed ---
    far_targets=(0.01, 0.05, 0.10),
    far_headline=0.10,
    seeds=(0, 1, 2),
)

ALL_STAGES = ["config", "checks", "table2", "rq1", "matched", "multiseed", "rq3",
              "wustl", "ablation", "summary"]
DEFAULT_STAGES = ["config", "checks", "table2", "rq1", "matched", "rq3", "wustl", "summary"]

# stages that need the CinC build; `--stages config` alone must not touch the data
NO_DATA_STAGES = {"config", "summary"}


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

SOURCE_FILES = ["cinc_loader.py", "cinc_adapter.py", "injection_generator.py",
                "eval_harness.py", "deep_detectors.py", "detection_report.py",
                "discrimination.py", "leakage_audit.py", "run_all.py"]


def _md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except OSError:
        return "MISSING"


def _versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "pandas", "scipy", "sklearn", "torch", "tigramite", "dowhy"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:
            out[m] = "not installed"
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        out["gpu"] = "unknown"
    return out


def write_manifest(cfg, run_dir, extra=None):
    man = {
        "run_id": os.path.basename(run_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "versions": _versions(),
        "source_md5": {f: _md5(os.path.join(cfg["repo"], f)) for f in SOURCE_FILES},
    }
    if extra:
        man.update(extra)
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    return man


# --------------------------------------------------------------------------- #
# dataset build (cached)
# --------------------------------------------------------------------------- #

def _cache_key(cfg, n_records):
    payload = json.dumps({
        "seed": cfg["seed"], "targets": list(cfg["target_channels"]),
        "win": cfg["win_sec"], "hop": cfg["hop_sec"],
        "hosts": cfg["max_hosts_per_record"], "artifact": cfg["artifact_frac"],
        "core": list(cfg["core_channels"]), "n_records": n_records,
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:10]


def load_records(cfg, self_test=False, verbose=True):
    if self_test:
        return _synthetic_records(cfg)
    from cinc_loader import load_cinc_records
    return load_cinc_records(cfg["cinc_dir"], limit=cfg["limit_records"], verbose=verbose)


def _synthetic_records(cfg, n_per_type=8):
    """CinC-shaped synthetic records spanning all five alarm types, for wiring checks."""
    FS = 125.0
    rng = np.random.default_rng(cfg["seed"])
    specs = [("Asystole", False), ("Bradycardia", True), ("Tachycardia", True),
             ("VentricularTachycardia", False), ("VentricularFlutterFib", False)]
    recs = []
    for ti, (atype, tv) in enumerate(specs):
        for k in range(n_per_type):
            T = int(FS * 320)
            t = np.arange(T) / FS
            ph = 2 * np.pi * (70 + 3 * k) / 60.0 * t
            sig = np.vstack([np.sin(ph) + 0.05 * rng.standard_normal(T),
                             0.9 * np.sin(ph - 0.6) + 0.05 * rng.standard_normal(T)])
            sig = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
            recs.append(dict(record_id=f"t{ti}r{k:02d}", signal=sig,
                             channels=["ECG", "PPG"], alarm_type=atype, true_alarm=tv,
                             alarm_onset_sample=int(FS * 300), fs=FS))
    print(f"[self-test] {len(recs)} synthetic records (5 alarm types x {n_per_type})")
    return recs


def build(cfg, records, need_windows=True, use_cache=True, verbose=True):
    """Build (df, feat_cols, windows) with the leak-free split applied. Cached."""
    from injection_generator import GeneratorConfig
    from eval_harness import build_dataset, patient_wise_split

    key = _cache_key(cfg, len(records))
    cdir = cfg["cache_dir"]
    os.makedirs(cdir, exist_ok=True)
    df_p = os.path.join(cdir, f"df_{key}.pkl")
    win_p = os.path.join(cdir, f"win_{key}.npy")
    feat_p = os.path.join(cdir, f"feat_{key}.json")

    have = os.path.exists(df_p) and os.path.exists(feat_p) and \
        (os.path.exists(win_p) or not need_windows)
    if use_cache and have:
        df = pd.read_pickle(df_p)
        feat_cols = json.load(open(feat_p))
        windows = np.load(win_p, mmap_mode="r") if need_windows else None
        if verbose:
            print(f"[cache] hit {key}: {len(df)} windows, {len(feat_cols)} features"
                  f"{'' if windows is None else f', tensor {tuple(windows.shape)}'}")
        return df, feat_cols, windows

    t0 = time.perf_counter()
    cfgen = GeneratorConfig(seed=cfg["seed"])
    out = build_dataset(records, cfgen,
                        target_channels=tuple(cfg["target_channels"]),
                        win_sec=cfg["win_sec"], hop_sec=cfg["hop_sec"],
                        max_hosts_per_record=cfg["max_hosts_per_record"],
                        artifact_frac=cfg["artifact_frac"],
                        core_channels=tuple(cfg["core_channels"]),
                        return_windows=need_windows, verbose=verbose)
    if need_windows:
        df, feat_cols, windows = out
    else:
        df, feat_cols = out
        windows = None
    df = patient_wise_split(df, seed=cfg["seed"])
    if verbose:
        print(f"[build] {time.perf_counter()-t0:.1f}s | records per split: "
              f"{df.drop_duplicates('record_id')['split'].value_counts().to_dict()}")

    df.to_pickle(df_p)
    json.dump(feat_cols, open(feat_p, "w"))
    if need_windows:
        np.save(win_p, np.asarray(windows, dtype=np.float32))
    return df, feat_cols, windows


# --------------------------------------------------------------------------- #
# stage: config — emit the hand-authored design constants as tables
# --------------------------------------------------------------------------- #
#
# The severity schedule (paper Table I) and the plausibility bounds are design
# INPUTS, not measured outputs: they transcribe constants that live in
# injection_generator.py. Typed by hand, they could drift from the code silently.
#
# This stage reads the constants out of the module and writes them, so the design
# tables get the same provenance as the results tables. It needs no data, no GPU and no
# dataset build: `--stages config` runs in about a second.
#
# It does not assume a shape. It searches by name, handles the shapes the generator
# actually uses (per-morphology dict of parallel level-indexed arrays; per-channel
# scalar sigma multiples) plus several it might, and PRINTS what it parsed so a
# mis-parse is visible rather than quietly wrong. The *_tidy.csv files keep the raw key
# path beside every value for that audit.

_SEVERITY_NAMES = ("SEVERITY_SCHEDULE", "SEVERITY_LEVELS", "SEVERITY_TABLE",
                   "SEVERITY", "SEVERITIES", "LEVEL_SCHEDULE", "SEVERITY_GRID")

# PLAUSIBILITY_SIGMA first: it is what the generator actually clips against.
_PLAUS_NAMES = ("PLAUSIBILITY_SIGMA", "PLAUSIBILITY", "PLAUSIBILITY_BOUNDS",
                "PLAUSIBLE_BOUNDS", "PLAUSIBLE_RANGES", "PHYSIOLOGICAL_BOUNDS",
                "PHYSIO_BOUNDS", "PHYS_BOUNDS", "CHANNEL_BOUNDS", "CHANNEL_RANGES",
                "CHANNEL_LIMITS", "IN_BOUND_LIMITS", "BOUNDS")

# A per-step (rate) limit, if one exists. Searched separately because Table IV reports a
# Delta_c column; if nothing is found, that column has no source in the code.
_DELTA_NAMES = ("DELTA_MAX", "MAX_STEP", "MAX_DELTA", "STEP_LIMITS", "DELTA_C",
                "MAX_ONE_STEP", "SLEW_LIMITS", "MAX_STEP_CHANGE", "PLAUSIBILITY_DELTA")

_DEFAULT_PLAUS_NAMES = ("DEFAULT_PLAUSIBILITY", "DEFAULT_PLAUSIBILITY_SIGMA",
                        "PLAUSIBILITY_DEFAULT")
_REPLAY_DUR_NAMES = ("REPLAY_DUR_S", "REPLAY_DURATION_S", "REPLAY_SEC", "REPLAY_DUR")
_FS_NAMES = ("TARGET_FS", "FS_TARGET", "CORE_FS", "FS", "SAMPLE_RATE", "TARGET_RATE", "SR")

_LOOKUP_MODULES = ("injection_generator", "cinc_adapter", "cinc_loader", "eval_harness")


def _find_const(names, modules=_LOOKUP_MODULES):
    """First (value, 'module.NAME') match: module globals, then a default
    GeneratorConfig() instance. (None, None) if nothing matches."""
    for mod_name in modules:
        try:
            mod = __import__(mod_name)
        except Exception:
            continue
        for n in names:
            v = getattr(mod, n, None)
            if v is not None and not callable(v):
                return v, f"{mod_name}.{n}"
        for dc_name in ("GeneratorConfig", "Config", "Window"):
            DC = getattr(mod, dc_name, None)
            if DC is None or not is_dataclass(DC):
                continue
            try:
                inst = DC()          # dataclasses needing args (Window) fall through
            except Exception:
                inst = None
            for n in names:
                ln = n.lower()
                v = getattr(inst, ln, None) if inst is not None else None
                if v is None:
                    v = _field_default(DC, ln)
                if v is not None and not callable(v):
                    return v, f"{mod_name}.{dc_name}.{ln}"
    return None, None


def _field_default(DC, name):
    """Declared default of a dataclass field, without instantiating the class."""
    for f in dc_fields(DC):
        if f.name != name:
            continue
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:
            try:
                return f.default_factory()
            except Exception:
                return None
    return None


def _candidate_names():
    """Module-level UPPERCASE names, for the error message when a lookup misses."""
    out = {}
    for mod_name in _LOOKUP_MODULES:
        try:
            mod = __import__(mod_name)
        except Exception:
            continue
        out[mod_name] = sorted(k for k in vars(mod)
                               if k.isupper() and not k.startswith("_")
                               and not callable(getattr(mod, k)))
    return out


def _flatten(obj, prefix=()):
    """Walk a nested dict / sequence / dataclass into (key_path, scalar) leaves."""
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = tuple(str(x) for x in k) if isinstance(k, tuple) else (str(k),)
            yield from _flatten(v, prefix + key)
    elif isinstance(obj, np.ndarray):
        for i, v in enumerate(obj.tolist()):
            yield from _flatten(v, prefix + (f"[{i}]",))
    elif isinstance(obj, (list, tuple, set)) and not isinstance(obj, (str, bytes)):
        for i, v in enumerate(obj):
            yield from _flatten(v, prefix + (f"[{i}]",))
    else:
        yield prefix, obj


_LEVEL_RE = re.compile(r"^(?:l|level)?[_ ]?([1-9])$", re.I)


def _as_level(tok):
    """'L3' / '3' / 'level_3' -> 'L3'; anything else -> None."""
    m = _LEVEL_RE.match(str(tok).strip())
    return f"L{m.group(1)}" if m else None


def _is_seq(v):
    return isinstance(v, (list, tuple, np.ndarray)) and not isinstance(v, (str, bytes))


def _severity_rows(sched):
    """-> tidy rows [morphology, level, param, value].

    Primary shape (what injection_generator uses): morphology -> {param: [v_L1..v_L4],
    scalar_param: v}, i.e. parallel arrays indexed by level-1, with scalars such as
    'kind' shared across levels. Also handles level-keyed dicts, 4-element sequences
    and (morphology, level) tuple keys."""
    def row(m, lv, k, v):
        return dict(morphology=str(m), level=lv, param=str(k), value=v)

    rows = []
    if not isinstance(sched, dict):
        return [row("?", "-", ".".join(p) or "value", v) for p, v in _flatten(sched)]

    for key, spec in sched.items():
        morph, lvl_from_key = str(key), None
        if isinstance(key, tuple) and len(key) == 2 and _as_level(key[1]):
            morph, lvl_from_key = str(key[0]), _as_level(key[1])

        if lvl_from_key:                                    # {("spike", 1): {...}}
            for p, v in _flatten(spec):
                rows.append(row(morph, lvl_from_key, ".".join(p) or "value", v))
            continue

        if isinstance(spec, dict) and spec and all(_as_level(k) for k in spec):
            for k, entry in spec.items():                   # {"spike": {"L1": {...}}}
                for p, v in _flatten(entry):
                    rows.append(row(morph, _as_level(k), ".".join(p) or "value", v))
            continue

        if isinstance(spec, dict):                          # parallel level-indexed arrays
            seqs = {k: list(v) for k, v in spec.items() if _is_seq(v)}
            scalars = {k: v for k, v in spec.items() if k not in seqs}
            if seqs:
                for i in range(max(len(v) for v in seqs.values())):
                    lvl = f"L{i + 1}"
                    for k, v in seqs.items():
                        if i < len(v):
                            rows.append(row(morph, lvl, k, v[i]))
                    for k, v in scalars.items():
                        rows.append(row(morph, lvl, k, v))
                continue
            for p, v in _flatten(spec):
                rows.append(row(morph, "-", ".".join(p) or "value", v))
            continue

        if _is_seq(spec) and len(spec) == 4:                # {"spike": [l1, l2, l3, l4]}
            for i, entry in enumerate(spec):
                for p, v in _flatten(entry):
                    rows.append(row(morph, f"L{i + 1}", ".".join(p) or "value", v))
            continue

        for p, v in _flatten(spec):
            rows.append(row(morph, "-", ".".join(p) or "value", v))
    return rows


_DUR_TOKENS = ("dur", "length", "len", "n_sample", "nsample", "samples", "width", "span")
_MAG_TOKENS = ("mag", "hi", "lo", "amp", "beta", "delta", "sigma", "offset", "gain")
_DELTA_TOKENS = ("delta", "step", "dmax", "d_max", "d_c", "max_change", "rate_limit", "slew")
_LO_TOKENS = ("lo", "low", "lower", "min", "l_c", "floor")
_HI_TOKENS = ("hi", "high", "upper", "max", "u_c", "ceil", "cap")


def _dur_unit(param):
    """None if not a duration; 's' | 'samples' | '?' otherwise."""
    p = str(param).lower()
    if not any(t in p for t in _DUR_TOKENS):
        return None
    if p.endswith("_s") or "sec" in p or p.endswith("_t"):
        return "s"
    if "sample" in p or "step" in p or "n_" in p:
        return "samples"
    return "?"


def _role(param):
    """Classify a plausibility parameter as lo / hi / delta. delta is tested first so
    'max_step' is not read as an upper bound."""
    p = str(param).lower()
    if p == "[2]" or any(t in p for t in _DELTA_TOKENS):
        return "delta"
    if p == "[0]" or any(t in p for t in _LO_TOKENS):
        return "lo"
    if p == "[1]" or any(t in p for t in _HI_TOKENS):
        return "hi"
    return param


def _wide(tidy, index, columns):
    """Safe pivot: keeps strings, never aggregates silently, and drops the index
    combinations pivot_table materialises but the constant does not contain."""
    if not len(tidy):
        return pd.DataFrame()
    w = tidy.pivot_table(index=index, columns=columns, values="value",
                         aggfunc="first", dropna=False)
    w.columns = [str(c) for c in w.columns]
    w = w.loc[~w.isna().all(axis=1)]
    return w.reset_index()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def stage_config(cfg, run_dir, verbose=True):
    """Emit paper Tables III and IV from the code that actually uses them."""
    prov, notes, ok = [], [], True
    tidy3 = tidy4 = pd.DataFrame()

    sched, sched_src = _find_const(_SEVERITY_NAMES)
    plaus, plaus_src = _find_const(_PLAUS_NAMES)
    fs_val, fs_src = _find_const(_FS_NAMES)
    if not isinstance(fs_val, (int, float)) or not (1 <= float(fs_val) <= 100000):
        fs_val, fs_src = cfg.get("fs", 125), "run_all CONFIG['fs'] (no module constant)"
    fs = float(fs_val)

    print("\n=== config — design constants behind paper Tables III and IV ===")
    print(f"sampling rate for the duration conversion: {fs:g} Hz  [{fs_src}]")

    # ---- Table III: per-morphology severity schedule -------------------------
    t3 = pd.DataFrame()
    if sched is None:
        ok = False
        print(f"\n[FAIL] no severity schedule found. Searched {list(_SEVERITY_NAMES)} "
              f"in {list(_LOOKUP_MODULES)}.")
        print("       Module-level constants that DO exist:")
        for m, names in _candidate_names().items():
            print(f"         {m}: {names}")
        print("       Nothing written for Table III — tell me the real identifier.")
    else:
        tidy3 = pd.DataFrame(_severity_rows(sched))
        tidy3["dur_unit"] = tidy3["param"].map(_dur_unit)
        v = tidy3["value"].map(_num)
        u = tidy3["dur_unit"]
        tidy3["duration_s"] = np.where(u == "s", v, np.where(u == "samples", v / fs, np.nan))
        tidy3["duration_samples"] = np.where(
            u == "samples", v, np.where(u == "s", (v * fs).round(0), np.nan))
        # only when the parameter name does not say which unit it is in
        tidy3["if_samples_then_s"] = np.where(u == "?", (v / fs).round(6), np.nan)
        tidy3["if_seconds_then_samples"] = np.where(u == "?", (v * fs).round(0), np.nan)

        order = tidy3["level"].map(lambda s: (0, s) if s == "-" else (1, s))
        tidy3 = (tidy3.assign(_o=order).sort_values(["morphology", "_o", "param"])
                 .drop(columns="_o"))
        t3 = _wide(tidy3, ["morphology", "level"], "param")

        n_morph = tidy3["morphology"].nunique()
        n_lvl = tidy3.loc[tidy3.level != "-", "level"].nunique()
        print(f"\nTable III — severity schedule   [{sched_src}]   "
              f"{n_morph} morphologies x {n_lvl} levels")
        print(t3.to_string(index=False))

        du = tidy3[tidy3["dur_unit"].notna()]
        if not len(du):
            notes.append("no parameter in the severity schedule looks like a duration, "
                         "so Table III's duration column has no source in this constant")
        elif (du["dur_unit"] == "?").any():
            amb = sorted(du.loc[du.dur_unit == "?", "param"].unique())
            notes.append(f"duration parameter(s) {amb} do not name their unit; the tidy "
                         "file gives both readings — pick the one matching Table III")
        else:
            dd = du.drop_duplicates(["morphology", "level"])
            print(f"\n   durations parsed as {sorted(du['dur_unit'].unique())}: "
                  f"{dd['duration_s'].min():g}–{dd['duration_s'].max():g} s "
                  f"({dd['duration_samples'].min():g}–{dd['duration_samples'].max():g} "
                  f"samples at {fs:g} Hz)")
        prov.append(dict(paper_table="III", quantity="severity schedule",
                         source=sched_src, n_values=len(tidy3)))

        # Replay carries a duration too, but Table III has no replay row.
        rep, rep_src = _find_const(_REPLAY_DUR_NAMES)
        if isinstance(rep, (int, float)):
            prov.append(dict(paper_table="III", quantity="replay duration",
                             source=rep_src, n_values=1))
        else:
            notes.append("replay duration is not a module constant (it is a literal "
                         "inside inject_replay), so no table can cite it; hoisting it to "
                         "REPLAY_DUR_S would let this stage emit it")

    # ---- Table IV: plausibility bounds ---------------------------------------
    t4 = pd.DataFrame()
    if plaus is None:
        ok = False
        print(f"\n[FAIL] no plausibility bounds found. Searched {list(_PLAUS_NAMES)} "
              f"in {list(_LOOKUP_MODULES)}.")
        if sched is not None:
            print("       Module-level constants that DO exist:")
            for m, names in _candidate_names().items():
                print(f"         {m}: {names}")
        print("       Nothing written for Table IV — tell me the real identifier.")
    else:
        tidy4 = pd.DataFrame([
            dict(channel=str(ch), param=".".join(path) or "value", value=val)
            for ch, entry in (plaus.items() if isinstance(plaus, dict) else [])
            for path, val in _flatten(entry)])
        if not len(tidy4):
            tidy4 = pd.DataFrame([dict(channel="?", param=".".join(p) or "value", value=v)
                                  for p, v in _flatten(plaus)])

        # Shape the generator uses: channel -> a single sigma multiple k, and the bound
        # is mu_c +/- k*sd_c computed per window from the pre-onset segment. There is no
        # fixed physical interval to report, so report k and the symbolic bound.
        sigma_form = (tidy4["param"] == "value").all() and tidy4["value"].map(
            lambda x: isinstance(x, (int, float))).all()
        if sigma_form:
            dflt, dflt_src = _find_const(_DEFAULT_PLAUS_NAMES)
            tidy4["param"] = "k_sigma"
            tidy4["role"] = "k_sigma"
            t4 = tidy4[["channel", "value"]].rename(columns={"value": "k_sigma"})
            t4["bound_lo"] = t4["k_sigma"].map(lambda k: f"mu_c - {k:g}*sd_c")
            t4["bound_hi"] = t4["k_sigma"].map(lambda k: f"mu_c + {k:g}*sd_c")
            t4["units"] = "per-patient sigma (within-patient z-scored space)"
            t4 = t4.sort_values("channel").reset_index(drop=True)
            notes.append("the bound is mu_c +/- k*sd_c on the pre-onset segment, "
                         "recomputed per window — it is NOT a fixed interval, so Table "
                         "IV's 'Range [l_c, u_c]' column describes a quantity the "
                         "generator does not use")
            if isinstance(dflt, (int, float)):
                notes.append(f"channels absent from the table fall back to k={dflt:g} "
                             f"[{dflt_src}]; Table IV must say so or a reader will think "
                             "unlisted channels are unclipped")
        else:
            tidy4["role"] = tidy4["param"].map(_role)
            if not (tidy4["role"] == "delta").any():
                dl, dl_src = _find_const(_DELTA_NAMES)
                if isinstance(dl, dict):
                    extra = [dict(channel=str(ch), param=".".join(pth) or "delta",
                                  value=vv, role="delta")
                             for ch, e in dl.items() for pth, vv in _flatten(e)]
                    if extra:
                        tidy4 = pd.concat([tidy4, pd.DataFrame(extra)], ignore_index=True)
                        plaus_src = f"{plaus_src} + {dl_src}"
            t4 = _wide(tidy4, "channel", "role")
            for c in ("lo", "hi", "delta"):
                if c not in t4.columns:
                    t4[c] = np.nan
            lead = ["channel", "lo", "hi", "delta"]
            t4 = t4[lead + [c for c in t4.columns if c not in lead]]

        print(f"\nTable IV — plausibility bounds   [{plaus_src}]")
        print(t4.to_string(index=False))
        prov.append(dict(paper_table="IV", quantity="plausibility bounds",
                         source=plaus_src, n_values=len(tidy4)))

        if not (tidy4["role"] == "delta").any():
            dl, _ = _find_const(_DELTA_NAMES)
            if not isinstance(dl, dict):
                ok = False
                notes.append("NO per-step / rate-limit constant exists anywhere in the "
                             "generator, so Table IV's Delta_c column reports a "
                             "constraint the code never applies — drop the column or "
                             "implement the limit")

        # Channels defined but never injected, and core channels with no bound at all.
        core = {c.upper() for c in cfg.get("core_channels", ())}
        defined = {str(c).upper() for c in t4["channel"]}
        match = lambda a, b: a.startswith(b) or b.startswith(a)  # noqa: E731
        unused = sorted(d for d in defined if core and not any(match(d, c) for c in core))
        if unused:
            notes.append(f"bounds defined for {unused} but core_channels="
                         f"{sorted(core)} — not exercised in this generation; say so in "
                         "the caption or move those rows to the supplement")
        missing = sorted(c for c in core if not any(match(d, c) for d in defined))
        if missing:
            ok = False
            notes.append(f"core channel(s) {missing} have NO bound — in-bound FDI "
                         "clipping is undefined for them")

    # ---- does the severity ladder collide with the bound? --------------------
    # The generator's docstring says the two are decoupled. They are only decoupled if
    # the top scheduled magnitude stays under k, otherwise in-bound FDI saturates at the
    # clip and the L3/L4 rungs stop being distinguishable.
    sat = []
    if sched is not None and len(tidy3) and isinstance(plaus, dict):
        dflt, _ = _find_const(_DEFAULT_PLAUS_NAMES)
        mag = tidy3[tidy3["param"].map(
            lambda p: any(t in str(p).lower() for t in _MAG_TOKENS)
            and _dur_unit(p) is None)]
        for ch in cfg.get("target_channels", ()):
            kv = plaus.get(ch, dflt)
            kv = _num(kv if not isinstance(kv, dict) else kv.get("hi", np.nan))
            if not np.isfinite(kv):
                continue
            for (m, lv), g in mag.groupby(["morphology", "level"]):
                vals = g["value"].map(_num)
                mx, mn = vals.max(), vals.min()
                if not (np.isfinite(mx) and mx >= kv):
                    continue
                sat.append(dict(
                    target_channel=ch, morphology=m, level=lv,
                    min_scheduled_sigma=mn, max_scheduled_sigma=mx, bound_k_sigma=kv,
                    verdict=("always clips (whole draw range is beyond the bound)"
                             if mn >= kv else
                             "can clip (part of the draw range is beyond the bound)")))
    if sat:
        sdf = pd.DataFrame(sat).sort_values(["target_channel", "morphology", "level"])
        sdf.to_csv(f"{run_dir}/config_severity_vs_bound.csv", index=False)
        print("\n!! severity/plausibility collision on the injected channel(s):")
        print(sdf.to_string(index=False))
        notes.append(f"{len(sdf)} scheduled (morphology, level) cell(s) meet or exceed "
                     "the plausibility bound on an injected channel, so in-bound FDI is "
                     "clipped there and the realised magnitude is below the schedule — "
                     "see config_severity_vs_bound.csv")

    # ---- write ---------------------------------------------------------------
    wrote = []
    if len(t3):
        t3.to_csv(f"{run_dir}/table3_severity.csv", index=False)
        tidy3.to_csv(f"{run_dir}/table3_severity_tidy.csv", index=False)
        wrote += ["table3_severity.csv", "table3_severity_tidy.csv"]
    if len(t4):
        t4.to_csv(f"{run_dir}/table4_plausibility.csv", index=False)
        tidy4.to_csv(f"{run_dir}/table4_plausibility_tidy.csv", index=False)
        wrote += ["table4_plausibility.csv", "table4_plausibility_tidy.csv"]
    if sat:
        wrote.append("config_severity_vs_bound.csv")
    if prov:
        pv = pd.DataFrame(prov)
        pv["source_file"] = pv["source"].str.split(".").str[0] + ".py"
        pv["source_md5"] = pv["source_file"].map(lambda f: _md5(os.path.join(cfg["repo"], f)))
        pv["fs_hz"] = fs
        pv["fs_source"] = fs_src
        pv.to_csv(f"{run_dir}/config_provenance.csv", index=False)
        wrote.append("config_provenance.csv")

    if verbose and notes:
        print("\n--- read before pasting into the paper ---")
        for i, n in enumerate(notes, 1):
            print(f"  {i}. {n}")
    if wrote:
        print("\n[saved] " + ", ".join(wrote))
    if not ok:
        print("\n  !! config stage incomplete — a paper table has no verified source")
    return dict(table3=t3, table4=t4, provenance=pd.DataFrame(prov),
                notes=notes, ok=ok,
                severity_vs_bound=pd.DataFrame(sat) if sat else None)


# --------------------------------------------------------------------------- #
# stage: invariant checks
# --------------------------------------------------------------------------- #

def stage_checks(cfg, df, feat_cols, run_dir):
    rows = []

    def chk(name, ok, detail="", warn_only=False):
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        rows.append(dict(check=name, status=status, detail=detail))

    n_nan = int(df[feat_cols].isna().sum().sum())
    chk("feature matrix NaN-free", n_nan == 0, f"{n_nan} NaNs, {len(feat_cols)} features")

    both = df["record_id"].notna() & df["patient_id"].notna()
    same = (df["record_id"].astype(str) == df["patient_id"].astype(str))
    chk("split keys complete and identical", bool(both.all() and same.all()),
        f"{int((~both).sum())} incomplete, {int((~same).sum())} mismatched")

    counts = df["cls"].value_counts().to_dict()
    chk("all four classes populated", len(counts) == 4 and all(v > 0 for v in counts.values()),
        str(counts))

    types = df["alarm_type"].replace("", np.nan).dropna().nunique()
    chk("all five alarm types present", types == 5, f"{types} types")

    atk = df[df.cls == "attack"]
    hosts = atk["window_id"].nunique()
    per_host = len(atk) / max(hosts, 1)
    nt = len(cfg["target_channels"])
    hi = nt * 34          # 2 replay + (4 morph x 4 level x 2 regime)
    lo = nt * 32          # replay is conditional on a non-empty bank
    chk("attacks per host window in range", lo - 0.01 <= per_host <= hi + 0.01,
        f"{per_host:.2f} observed, expected {lo}-{hi} "
        f"({hosts} host windows, {nt} target channel(s))")

    tgt = sorted(atk["target_channel"].unique())
    chk("attribution target is non-degenerate", len(tgt) > 1,
        f"injected channels = {tgt}" + ("  <-- RQ2 channel localisation is DEGENERATE; "
                                        "a constant predictor scores AC@1=1.0"
                                        if len(tgt) <= 1 else ""), warn_only=True)

    leak = []
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and set(df[df.split == a]["record_id"]) & set(df[df.split == b]["record_id"]):
                leak.append(f"{a}/{b}")
    chk("patient-wise split leak-free", not leak, ",".join(leak) or "no record in two splits")

    te = df[df.split == "test"]
    n_a = int((te.cls == "attack").sum())
    n_n = int((te.cls == "physiology_normal").sum())
    prev = n_a / max(n_a + n_n, 1)
    chk("test prevalence recorded", True,
        f"attack {n_a} / normal {n_n} -> prevalence {prev:.4f} "
        f"(= the no-skill AUPRC floor)")

    tbl = pd.DataFrame(rows)
    print("\n=== invariant checks ===")
    print(tbl.to_string(index=False))
    tbl.to_csv(f"{run_dir}/checks.csv", index=False)
    if (tbl.status == "FAIL").any():
        print("\n  !! one or more checks FAILED — read the detail column before trusting results")
    return tbl


# --------------------------------------------------------------------------- #
# stage: table 2 composition
# --------------------------------------------------------------------------- #

def stage_table2(cfg, df, run_dir):
    comp = df["cls"].value_counts().rename("windows").to_frame()
    comp["pct"] = (100 * comp["windows"] / len(df)).round(2)
    by_split = pd.crosstab(df["cls"], df["split"])
    atk = df[df.cls == "attack"]
    fam = atk["family"].value_counts().rename("windows").to_frame()
    lv = atk["level"].map(lambda v: "replay" if v == 0 else f"L{int(v)}")
    grid = pd.crosstab(atk["family"], lv)
    rep = (atk[atk.family == "replay"]["replay_source"].value_counts()
           if (atk.family == "replay").any() else pd.Series(dtype=int))
    clean_by_type = pd.crosstab(df[df.cls != "attack"]["alarm_type"],
                                df[df.cls != "attack"]["cls"])

    print("\n=== Table 2 — composition ===")
    print(comp.to_string())
    print("\nper split (records: "
          f"{df.drop_duplicates('record_id')['split'].value_counts().to_dict()})")
    print(by_split.to_string())
    print("\nattack families:"); print(fam.to_string())
    print("\nfamily x level:"); print(grid.to_string())
    if len(rep):
        print("\nreplay source:"); print(rep.to_string())
    print("\nclean classes by alarm type:"); print(clean_by_type.to_string())

    for obj, nm in ((comp, "composition"), (by_split, "by_split"), (grid, "family_by_level"),
                    (clean_by_type, "clean_by_alarmtype")):
        obj.to_csv(f"{run_dir}/table2_{nm}.csv")
    return dict(composition=comp, by_split=by_split, grid=grid, clean=clean_by_type)


# --------------------------------------------------------------------------- #
# stage: rq1 (+ rq4)
# --------------------------------------------------------------------------- #

def make_detector(name, cfg, **over):
    """Lazily construct a detector. torch is imported only when a deep model is asked
    for, so the classical path runs on a machine without torch."""
    seed = over.pop("seed", cfg["seed"])
    ep = over.pop("epochs", cfg["epochs"])
    if name == "IsolationForest":
        from eval_harness import IsolationForestDetector
        return IsolationForestDetector(seed=seed), "features"
    import deep_detectors as D
    table = {
        "LSTM-AE": D.LSTMAEDetector, "USAD": D.USADDetector, "GDN": D.GDNDetector,
        "TranAD": D.TranADDetector, "OmniAnomaly": D.OmniAnomalyDetector,
    }
    if name not in table:
        raise KeyError(f"unknown detector {name!r}; known: "
                       f"IsolationForest,{','.join(table)}")
    kw = dict(epochs=ep, seed=seed)
    kw.update(over)
    return table[name](**kw), "windows"


def stage_rq1(cfg, df, feat_cols, windows, run_dir, detectors=None):
    from detection_report import run_rq1, consolidate
    results = []
    Xfeat = df[feat_cols].values
    for name in (detectors or cfg["detectors"]):
        print(f"\n>>> RQ1: {name}")
        try:
            det, kind = make_detector(name, cfg)
        except Exception as e:
            print(f"    [skip] cannot construct {name}: {type(e).__name__}: {e}")
            continue
        X = Xfeat if kind == "features" else windows
        if X is None:
            print(f"    [skip] {name} needs the window tensor; rebuild with need_windows=True")
            continue
        try:
            res = run_rq1(df, X, det, out_dir=run_dir, n_boot=cfg["n_boot"], ci=cfg["ci"],
                          target_prevalence=cfg["target_prevalence"],
                          n_prev_rep=cfg["n_prev_rep"], seed=cfg["seed"])
            results.append(res)
        except Exception as e:
            import traceback
            print(f"    [FAILED] {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    if results:
        t3, t6 = consolidate(results, out_dir=run_dir)
        print("\n=== Table 3-R (consolidated) ===")
        print(t3.to_string(index=False))
        print("\n=== Table 6-R frontier (consolidated) ===")
        print(t6.to_string(index=False))
        return results, t3, t6
    return results, None, None


# --------------------------------------------------------------------------- #
# stage: ablation — does removing the shared pooling front-end rescue spikes?
# --------------------------------------------------------------------------- #

def stage_ablation(cfg, df, windows, run_dir):
    """Every deep detector shares an x10 average-pool front-end. At 125 Hz a level-1
    spike is 3 samples of 0.5 sigma, which that pool attenuates to ~0.15 sigma — so the
    paper's 'transient spikes evade all six detectors' may be one preprocessing choice
    failing five times rather than five independent method failures. This reruns LSTM-AE
    with the pool removed. If spike detection rises materially, the claim must be
    restated as conditional on the pooling front-end."""
    from detection_report import run_rq1
    if windows is None:
        print("[ablation] needs the window tensor — skipped")
        return []
    rows = []
    for pool in cfg["ablation_pools"]:
        tag = f"LSTM-AE-pool{pool}"
        print(f"\n>>> ablation: {tag}  (batch {cfg['ablation_batch']}; "
              f"pool=1 keeps full 125 Hz resolution and is slower)")
        try:
            det, _ = make_detector("LSTM-AE", cfg, pool=pool, batch=cfg["ablation_batch"])
            res = run_rq1(df, windows, det, out_dir=run_dir, n_boot=cfg["n_boot"],
                          ci=cfg["ci"], target_prevalence=cfg["target_prevalence"],
                          n_prev_rep=cfg["n_prev_rep"], seed=cfg["seed"], save_prefix=tag)
            r = dict(res["frontier"]); r["detector"] = tag; r["pool"] = pool
            rows.append(r)
        except Exception as e:
            import traceback
            print(f"    [FAILED] {tag}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    if rows:
        tbl = pd.DataFrame(rows)
        tbl.to_csv(f"{run_dir}/ablation_pooling.csv", index=False)
        print("\n=== pooling ablation (compare fdi_spike against the pool=10 rows) ===")
        print(tbl[[c for c in ("detector", "pool", "fdi_spike", "fdi_spike_CI",
                               "fdi_spike_L1", "replay", "replay_CI") if c in tbl]]
              .to_string(index=False))
    return rows


# --------------------------------------------------------------------------- #
# stage: matched-FAR + threshold-free subproblems
# --------------------------------------------------------------------------- #

def stage_matched(cfg, df, feat_cols, windows, run_dir, detectors=None):
    """Threshold-free results per attack subset, and detection at a common false-alarm
    budget calibrated on the validation normals. Run over the SAME seeds and the SAME
    deterministic configuration as the multi-seed detection stage, so all of its tables
    share one provenance.
    With a single seed the record-level bootstrap CIs are kept; with several, the
    across-seed sd replaces them."""
    import detection_report as R
    print(f"[determinism] {R.set_determinism()}")
    seeds = tuple(cfg["seeds"])
    multi = len(seeds) > 1
    n_boot = 0 if multi else cfg["n_boot"]
    Xfeat = df[feat_cols].values
    sp_all, da_all, frames = [], [], []

    for name in (detectors or cfg["detectors"]):
        try:
            _, kind = make_detector(name, cfg)
        except Exception as e:
            print(f"    [skip] {name}: {type(e).__name__}: {e}")
            continue
        X = Xfeat if kind == "features" else windows
        if X is None:
            print(f"    [skip] {name} needs the window tensor")
            continue
        sp_seeds, da_seeds = [], []
        for sd in seeds:
            print(f"\n>>> matched-FAR: {name}  seed {sd}")
            try:
                det, _ = make_detector(name, cfg, seed=sd)
                scored, _ = R.score_frame(df, X, det, verbose=True)
                sp = R.subproblem_auprc(scored, name, n_boot=n_boot, ci=cfg["ci"],
                                        seed=cfg["seed"], verbose=not multi)
                da = R.detection_at_far(scored, name, far_targets=tuple(cfg["far_targets"]),
                                        n_boot=n_boot, ci=cfg["ci"], seed=cfg["seed"],
                                        verbose=not multi)
                sp_seeds.append(sp.assign(seed=sd))
                da_seeds.append(da.assign(seed=sd))
            except Exception as e:
                import traceback
                print(f"    [FAILED] {name} seed {sd}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
        if not sp_seeds:
            continue
        if multi:
            sp_m = R.aggregate_seeds(sp_seeds, ["detector", "subproblem"])
            da_m = R.aggregate_seeds(da_seeds, ["detector", "far_target"])
        else:
            sp_m, da_m = sp_seeds[0], da_seeds[0]
        sp_all.append(pd.concat(sp_seeds, ignore_index=True))
        da_all.append(pd.concat(da_seeds, ignore_index=True))
        frames.append((sp_m, da_m))

    if not frames:
        return None, None, None
    pd.concat(sp_all, ignore_index=True).to_csv(f"{run_dir}/subproblem_perseed_all.csv", index=False)
    pd.concat(da_all, ignore_index=True).to_csv(f"{run_dir}/detection_at_far_perseed_all.csv", index=False)
    piv, piv_ci, at = R.compare_matched(frames, out_dir=run_dir, far_target=cfg["far_headline"])
    if multi:
        sp_m = pd.concat([a for a, _ in frames], ignore_index=True)
        rep = sp_m.pivot(index="detector", columns="subproblem", values="AUPRC")
        sd = sp_m.pivot(index="detector", columns="subproblem", values="AUPRC_sd")
        tbl = rep.round(3).astype(str) + " ± " + sd.round(3).astype(str)
        tbl.to_csv(f"{run_dir}/table4R_subproblem_auprc_meansd.csv")
        print(f"\n=== Table 4-R: threshold-free AUPRC, mean ± sd over seeds {list(seeds)} ===")
        print(tbl[[c for c in R.FRONTIER_GROUPS if c in tbl.columns]].to_string())
        da_m = pd.concat([b for _, b in frames], ignore_index=True)
        hd = da_m[da_m.far_target == cfg["far_headline"]].set_index("detector")
        cols = ["FAR_physiology_normal", "FAR_fault_artifact", "FAR_physiology_event"] + \
               [c for c in R.FRONTIER_GROUPS if c in hd.columns]
        hd[[c for c in cols if c in hd.columns]].round(3).to_csv(
            f"{run_dir}/table5R_detection_at_far{int(100*cfg['far_headline'])}_mean.csv")
        print(f"\n=== Table 5-R: detection at matched FAR {cfg['far_headline']}, "
              f"mean over seeds ===")
        print(hd[[c for c in cols if c in hd.columns]].round(3).to_string())
    return piv, piv_ci, at


# --------------------------------------------------------------------------- #
# stage: multi-seed
# --------------------------------------------------------------------------- #

def stage_multiseed(cfg, df, feat_cols, windows, run_dir, detectors=None):
    """Deep detectors are not bit-reproducible under cuDNN. Report mean +/- sd."""
    import detection_report as R
    print(f"[determinism] {R.set_determinism()}")
    Xfeat = df[feat_cols].values
    all_summ = []
    for name in (detectors or cfg["detectors"]):
        try:
            _, kind = make_detector(name, cfg)
        except Exception as e:
            print(f"    [skip] {name}: {type(e).__name__}: {e}")
            continue
        X = Xfeat if kind == "features" else windows
        if X is None:
            print(f"    [skip] {name} needs the window tensor")
            continue
        factory = lambda s, nm=name: make_detector(nm, cfg, seed=s)[0]
        try:
            summ, _ = R.run_multiseed(df, X, factory, seeds=tuple(cfg["seeds"]),
                                      name=name, far_target=cfg["far_headline"],
                                      out_dir=run_dir)
            all_summ.append(summ.assign(metric=summ.index))
        except Exception as e:
            import traceback
            print(f"    [FAILED] {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    if all_summ:
        tbl = pd.concat(all_summ, ignore_index=True)
        keep = tbl[tbl.metric.isin(R.MULTISEED_METRICS)]
        piv = keep.pivot(index="detector", columns="metric", values="report_as")
        piv.to_csv(f"{run_dir}/table3R_multiseed.csv")
        tbl.to_csv(f"{run_dir}/multiseed_all.csv", index=False)
        print("\n=== Table 3-R with seed spread (mean ± sd) ===")
        print(piv.to_string())
        risky = keep[keep["range"] > 0.03]
        if len(risky):
            print("\n  metrics whose seed-to-seed range exceeds 0.03 "
                  "(do NOT report single-run values for these):")
            print(risky[["detector", "metric", "mean", "sd", "range"]].to_string(index=False))
        return piv
    return None


# --------------------------------------------------------------------------- #
# stages: rq3, wustl
# --------------------------------------------------------------------------- #

def stage_rq3(cfg, df, feat_cols, run_dir):
    from discrimination import run_rq3
    out = {}
    for mode in ("4way", "3way"):
        try:
            out[mode] = run_rq3(df, feat_cols, mode=mode, seed=cfg["seed"], out_dir=run_dir)
        except Exception as e:
            print(f"[rq3 {mode}] FAILED: {type(e).__name__}: {e}")
    return out


def stage_wustl(cfg, run_dir):
    from leakage_audit import run_leakage_audit
    if not os.path.exists(cfg["wustl_csv"]):
        print(f"[wustl] CSV not found at {cfg['wustl_csv']} — skipped")
        return None
    try:
        return run_leakage_audit(cfg["wustl_csv"], seed=cfg["seed"], out_dir=run_dir)
    except Exception as e:
        print(f"[wustl] FAILED: {type(e).__name__}: {e}")
        return None


# --------------------------------------------------------------------------- #
# stage: summary
# --------------------------------------------------------------------------- #

def stage_summary(cfg, run_dir, man, t3, t6, ablation_rows, checks,
                  matched=None, multiseed=None, config_tables=None):
    L = [f"# IoMT-SecAlarmBench — run `{man['run_id']}`", "",
         f"- generated {man['timestamp_utc']}",
         f"- seed `{cfg['seed']}`, uniform epochs `{cfg['epochs']}`, "
         f"target channels `{list(cfg['target_channels'])}`, "
         f"hosts/record `{cfg['max_hosts_per_record']}`",
         f"- torch `{man['versions'].get('torch')}` on `{man['versions'].get('gpu')}`, "
         f"sklearn `{man['versions'].get('sklearn')}`",
         "- source hashes and full config: `manifest.json`", ""]

    if checks is not None:
        fails = checks[checks.status == "FAIL"]
        L += ["## Invariant checks", "",
              ("All checks passed." if not len(fails) else
               "**FAILED:** " + "; ".join(fails["check"] + " (" + fails["detail"] + ")")), ""]
        deg = checks[checks.check.str.contains("non-degenerate")]
        if len(deg) and deg.iloc[0].status == "FAIL":
            L += ["> RQ2 channel localisation is degenerate in this configuration "
                  "(single injection target). Re-run with `--target-channels ECG,PPG` "
                  "for a non-degenerate attribution evaluation.", ""]

    if t3 is not None:
        L += ["## Table 3-R — Detection", "",
              "`AUPRC_vs_normal` reproduces the published column. "
              "`no_skill_vs_normal` is its chance floor (the attack prevalence). "
              "`AUPRC_vs_normal_fault` adds the fault-artifact windows to the negative "
              "class — the clinically honest version. `FAR_fault_artifact` is the rate "
              "at which sensor artifacts are flagged as attacks. CIs resample records.",
              "", t3.to_markdown(index=False), ""]
    if t6 is not None:
        L += ["## Table 6-R — The frontier", "",
              "Detection rate on the mechanism-preserving and transient cases, with "
              "record-level 95% CIs and cell counts.", "",
              t6.to_markdown(index=False), ""]
    if matched is not None:
        L += ["## Table 7-R — threshold-free AUPRC by attack subset", "",
              "AUPRC of each attack subset against the clean normals, with no operating "
              "point — so a detector cannot buy a high number by flagging more. Compare "
              "DOWN a column (across detectors); rows have different chance floors.", "",
              matched.to_markdown(), "",
              "> This is the table that decides the replay claim. At its own max-F1 "
              "threshold OmniAnomaly reaches replay 0.71 while flagging 58% of normal "
              "windows; if its column here is not also highest, the advantage was the "
              "threshold, not the model.", ""]
    if multiseed is not None:
        L += ["## Table 3-R with seed spread", "",
              f"mean ± sd over seeds {list(cfg['seeds'])}. cuDNN has no deterministic RNN "
              "backward, so single-run deep-detector values are not reproducible; any "
              "cross-detector ordering smaller than the seed spread is not claimable.",
              "", multiseed.to_markdown(), ""]
    if ablation_rows:
        L += ["## Pooling ablation", "",
              "Every deep detector in Table 3-R shares an x10 average-pool front-end, "
              "which attenuates a 3-sample level-1 spike from 0.5σ to ~0.15σ. Compare "
              "`fdi_spike` here against the pooled LSTM-AE row above: if it rises "
              "materially, the spike finding is conditional on the pooling front-end and "
              "must be stated that way.", "",
              pd.DataFrame(ablation_rows).to_markdown(index=False), ""]

    if config_tables is not None:
        L += ["## Design constants (paper Tables III and IV)", "",
              "Emitted from the code that uses them, not transcribed. Provenance and "
              "source hashes: `config_provenance.csv`.", ""]
        for key, cap in (("table3", "Severity schedule"),
                         ("table4", "Plausibility bounds")):
            tb = config_tables.get(key)
            if tb is not None and len(tb):
                L += [f"**{cap}**", "", tb.to_markdown(index=False), ""]
        if config_tables.get("notes"):
            L += ["Caveats:", ""] + [f"- {n}" for n in config_tables["notes"]] + [""]

    L += ["## Files", ""] + [f"- `{f}`" for f in sorted(os.listdir(run_dir))]
    path = os.path.join(run_dir, "SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n[saved] {path}")
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate every IoMT-SecAlarmBench table.")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma list from {ALL_STAGES}, or 'all'")
    ap.add_argument("--detectors", default=None, help="comma list; default all six")
    ap.add_argument("--repo", default=CONFIG["repo"])
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--target-channels", default=",".join(CONFIG["target_channels"]))
    ap.add_argument("--max-hosts", type=int, default=CONFIG["max_hosts_per_record"])
    ap.add_argument("--limit-records", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=CONFIG["n_boot"])
    ap.add_argument("--ablation-pools", default=",".join(map(str, CONFIG["ablation_pools"])))
    ap.add_argument("--seeds", default=",".join(map(str, CONFIG["seeds"])),
                    help="comma list for the multiseed stage")
    ap.add_argument("--far-headline", type=float, default=CONFIG["far_headline"])
    ap.add_argument("--fs", type=float, default=CONFIG["fs"],
                    help="core-cluster sampling rate; only the config stage uses it")
    ap.add_argument("--cache-dir", default=CONFIG["cache_dir"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic records; verifies wiring without data or GPU")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print config and exit")
    a = ap.parse_args(argv)

    cfg = dict(CONFIG)
    cfg.update(repo=a.repo, epochs=a.epochs, seed=a.seed,
               target_channels=tuple(c.strip() for c in a.target_channels.split(",") if c.strip()),
               max_hosts_per_record=a.max_hosts, limit_records=a.limit_records,
               n_boot=a.n_boot, cache_dir=a.cache_dir,
               fs=a.fs,
               ablation_pools=tuple(int(p) for p in a.ablation_pools.split(",") if p.strip()),
               seeds=tuple(int(x) for x in a.seeds.split(",") if x.strip()),
               far_headline=a.far_headline)
    cfg["cinc_dir"] = f"{a.repo}/data/cinc2015/training"
    cfg["wustl_csv"] = f"{a.repo}/data/wustl/wustl-ehms-2020.csv"
    cfg["results_root"] = f"{a.repo}/results"
    if a.self_test:
        cfg["results_root"] = os.path.join(a.repo, "results_selftest")
        cfg["cache_dir"] = os.path.join(cfg["cache_dir"], "selftest")
        cfg["n_boot"] = min(cfg["n_boot"], 300)
        cfg["n_prev_rep"] = 50
        cfg["epochs"] = min(cfg["epochs"], 3)

    stages = ALL_STAGES if a.stages == "all" else [s.strip() for s in a.stages.split(",")]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        ap.error(f"unknown stage(s) {bad}; choose from {ALL_STAGES}")
    detectors = [d.strip() for d in a.detectors.split(",")] if a.detectors else None

    if a.repo not in sys.path:
        sys.path.insert(0, a.repo)

    run_id = a.run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + ("-selftest" if a.self_test else "")
    run_dir = os.path.join(cfg["results_root"], run_id)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 78)
    print(f" IoMT-SecAlarmBench  run_all  |  run_id {run_id}")
    print(f" repo    {cfg['repo']}")
    print(f" out     {run_dir}")
    print(f" stages  {stages}")
    print(f" seed {cfg['seed']} | epochs {cfg['epochs']} | targets "
          f"{list(cfg['target_channels'])} | hosts/record {cfg['max_hosts_per_record']}")
    print("=" * 78)
    man = write_manifest(cfg, run_dir)
    if a.dry_run:
        print(json.dumps(man, indent=2)[:4000])
        return dict(run_dir=run_dir, manifest=man)

    t_start = time.perf_counter()
    cfg_tables = stage_config(cfg, run_dir) if "config" in stages else None

    if not (set(stages) - NO_DATA_STAGES):
        # config (and/or summary) only: no records, no build, no GPU
        if "summary" in stages:
            stage_summary(cfg, run_dir, man, None, None, [], None, config_tables=cfg_tables)
        write_manifest(cfg, run_dir, extra=dict(
            elapsed_s=round(time.perf_counter() - t_start, 1), stages_run=stages))
        print(f"\ndone in {time.perf_counter()-t_start:.1f}s -> {run_dir}")
        return dict(run_dir=run_dir, config_tables=cfg_tables)

    deep_needed = (detectors or list(cfg["detectors"])) != ["IsolationForest"]
    need_windows = (deep_needed and any(st in stages for st in
                                        ("rq1", "matched", "multiseed"))) or "ablation" in stages

    records = load_records(cfg, self_test=a.self_test)
    if not records:
        print("[fatal] loader returned 0 records — check cinc_dir")
        return None
    df, feat_cols, windows = build(cfg, records, need_windows=need_windows,
                                  use_cache=not a.no_cache)

    checks = t3 = t6 = None
    ablation_rows = []
    if "checks" in stages:
        checks = stage_checks(cfg, df, feat_cols, run_dir)
    if "table2" in stages:
        stage_table2(cfg, df, run_dir)
    if "rq1" in stages:
        _, t3, t6 = stage_rq1(cfg, df, feat_cols, windows, run_dir, detectors)
    matched_tbl = multiseed_tbl = None
    if "matched" in stages:
        matched_tbl, _, _ = stage_matched(cfg, df, feat_cols, windows, run_dir, detectors)
    if "multiseed" in stages:
        multiseed_tbl = stage_multiseed(cfg, df, feat_cols, windows, run_dir, detectors)
    if "ablation" in stages:
        ablation_rows = stage_ablation(cfg, df, windows, run_dir)
    if "rq3" in stages:
        stage_rq3(cfg, df, feat_cols, run_dir)
    if "wustl" in stages:
        stage_wustl(cfg, run_dir)
    if "summary" in stages:
        stage_summary(cfg, run_dir, man, t3, t6, ablation_rows, checks,
                      matched=matched_tbl, multiseed=multiseed_tbl,
                      config_tables=cfg_tables)

    write_manifest(cfg, run_dir, extra=dict(
        elapsed_s=round(time.perf_counter() - t_start, 1),
        n_records=len(records), n_windows=int(len(df)),
        class_counts=df["cls"].value_counts().to_dict(),
        stages_run=stages))
    print(f"\ndone in {time.perf_counter()-t_start:.1f}s -> {run_dir}")
    return dict(run_dir=run_dir, df=df, feat_cols=feat_cols, table3=t3, table6=t6)


if __name__ == "__main__":
    main()
