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
