#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run-provenance recorder.

The paper states that every results directory carries a manifest.json
recording "the configuration, package versions, GPU, and MD5 of every source
module, including the design constants of Table I". This module writes it.

That claim is checkable, so it needs to be true of every run. Call
`open_run()` at the top of any script that produces reported numbers:

    from manifest import open_run
    run = open_run(config_path="configs/benchmark.yaml", tag="rq2")
    ...
    run.finish(extra={"n_test_windows": 11825})

It creates results/<UTC timestamp>/ , writes manifest.json on entry, and
updates it with duration and any extras on exit. The timestamp format matches
the run IDs cited in the paper (20260730-205501).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Modules whose bytes must be pinned for a reported number to be meaningful.
# Table I's severity constants live in injection_generator.py, so it is here.
TRACKED_DIRS = ("src", "scripts", "configs", "figures")
TRACKED_SUFFIXES = (".py", ".yaml", ".yml")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes() -> dict:
    """MD5 of every tracked source file, keyed by repo-relative path."""
    out = {}
    for d in TRACKED_DIRS:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in TRACKED_SUFFIXES:
                out[str(p.relative_to(REPO_ROOT))] = _md5(p)
    return out


def package_versions() -> dict:
    pkgs = ["numpy", "pandas", "scipy", "sklearn", "torch", "wfdb",
            "matplotlib", "tigramite", "dowhy"]
    out = {}
    for name in pkgs:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def gpu_info() -> dict:
    info = {"available": False}
    try:
        import torch
        info["torch_cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())]
            info["deterministic"] = torch.backends.cudnn.deterministic
            info["benchmark"] = torch.backends.cudnn.benchmark
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    dirty = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "tag": run("git", "describe", "--tags", "--always"),
        # A dirty tree means the recorded hashes do not correspond to any
        # published commit. Reported runs should always be clean.
        "dirty": bool(dirty),
        "dirty_files": dirty.splitlines() if dirty else [],
    }


class Run:
    def __init__(self, directory: Path, manifest: dict):
        self.dir = directory
        self.manifest = manifest
        self._t0 = time.time()

    @property
    def run_id(self) -> str:
        return self.dir.name

    def path(self, *parts) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self) -> None:
        with open(self.dir / "manifest.json", "w") as fh:
            json.dump(self.manifest, fh, indent=2, sort_keys=True)

    def finish(self, extra: dict | None = None) -> Path:
        self.manifest["duration_s"] = round(time.time() - self._t0, 3)
        self.manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if extra:
            self.manifest.setdefault("results", {}).update(extra)
        self.write()
        return self.dir


def open_run(config_path: str | None = None, tag: str = "", seeds=None,
             results_root: str | Path = "results") -> Run:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{tag}" if tag else ts
    directory = Path(results_root) / name
    directory.mkdir(parents=True, exist_ok=True)

    cfg = None
    if config_path and Path(config_path).is_file():
        try:
            import yaml
            cfg = yaml.safe_load(open(config_path))
        except Exception:
            cfg = {"_unparsed": str(config_path)}

    manifest = {
        "run_id": name,
        "tag": tag,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "seeds": list(seeds) if seeds is not None else None,
        "config_path": config_path,
        "config": cfg,
        "git": git_state(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions(),
        "gpu": gpu_info(),
        "env": {k: os.environ[k] for k in
                ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")
                if k in os.environ},
        "source_md5": source_hashes(),
    }
    run = Run(directory, manifest)
    run.write()
    return run


def verify(manifest_path: str | Path) -> int:
    """Re-hash the working tree against a recorded manifest.

    Returns the number of mismatches; prints each. Use this to check that a
    results directory still corresponds to the code currently checked out.
    """
    recorded = json.load(open(manifest_path))["source_md5"]
    current = source_hashes()
    bad = 0
    for path, want in sorted(recorded.items()):
        got = current.get(path)
        if got is None:
            print(f"MISSING   {path}")
            bad += 1
        elif got != want:
            print(f"CHANGED   {path}\n          recorded {want}\n          current  {got}")
            bad += 1
    for path in sorted(set(current) - set(recorded)):
        print(f"UNTRACKED {path}  (added since the run)")
        bad += 1
    print(f"\n{len(recorded)} files recorded, {bad} discrepancies.")
    return bad


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "verify":
        raise SystemExit(1 if verify(sys.argv[2]) else 0)
    r = open_run(tag="selftest")
    print("wrote", r.dir / "manifest.json")
    print(json.dumps({k: r.manifest[k] for k in
                      ("run_id", "packages", "gpu")}, indent=2)[:600])
    r.finish()
