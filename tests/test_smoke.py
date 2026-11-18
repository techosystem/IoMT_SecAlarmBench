"""Smoke tests: cheap, no data required, catch structural breakage."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_configs_parse():
    for name in ("benchmark.yaml", "detectors.yaml"):
        cfg = yaml.safe_load(open(ROOT / "configs" / name))
        assert isinstance(cfg, dict) and cfg


def test_severity_schedule_matches_table_one():
    """Table I has four morphologies x four levels; L1->L4 monotone within one."""
    sev = yaml.safe_load(open(ROOT / "configs" / "benchmark.yaml"))["severity"]
    assert set(sev) == {"spike", "stuck_at", "drift", "bias"}
    for morph, row in sev.items():
        assert len(row["duration_s"]) == 4, morph
        mags = []
        for lv in ("L1", "L2", "L3", "L4"):
            v = row[lv]
            mags.append(max(v) if isinstance(v, list) else v)
        assert mags == sorted(mags), f"{morph} severity not monotone: {mags}"


def test_dagger_cells_count_is_six():
    """Threats (ii): the bound is reached in 6 of 16 crafted cells."""
    sev = yaml.safe_load(open(ROOT / "configs" / "benchmark.yaml"))["severity"]
    n = sum(len(r.get("dagger", [])) for r in sev.values())
    assert n == 6, f"paper says 6 saturated cells, config says {n}"


def test_manifest_records_source_hashes():
    import manifest
    h = manifest.source_hashes()
    assert any(p.endswith("configs/benchmark.yaml") for p in h), \
        "Table I constants must be covered by the manifest"
    assert all(len(v) == 32 for v in h.values())


def test_no_waveform_files_committed():
    """The release must not redistribute raw or derived waveforms."""
    banned = {".dat", ".hea", ".npz", ".npy", ".h5", ".hdf5", ".mat"}
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    offenders = [f for f in tracked if Path(f).suffix in banned]
    assert not offenders, f"waveform data committed: {offenders}"


def test_multichannel_generator_self_test():
    """C1-ECG, C2 and C3 attacks: per-host counts, seeds and counterfactuals."""
    import injection_multichannel
    injection_multichannel._self_test()


def test_reported_results_present():
    """The paper's results ship with the repo and match Fig. 2."""
    import pandas as pd
    res = ROOT / "results" / "paper"
    for name in ("detection_summary_v1.csv", "grid_morph_level_v1.csv", "alarm_oracle_effects.csv",
                 "rq4_variants.json", "wustl_audit.csv", "hyperparameters.json"):
        assert (res / name).exists(), name
    g = pd.read_csv(res / "grid_morph_level_v1.csv")
    learned = ["IsolationForest", "USAD", "LSTM-AE", "GDN", "TranAD", "OmniAnomaly"]
    cell = g[g.detector.isin(learned) & (g.setting == "C1-PPG") & (g.family == "fdi")
             & (g.morphology == "bias") & (g.level == 1)]
    assert round(100 * cell.hmAUROC.mean()) == 60
