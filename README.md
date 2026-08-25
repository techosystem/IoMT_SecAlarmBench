# IoMT-SecAlarmBench

**A counterfactual benchmark for integrity attacks in the Internet of Medical Things.**

Companion code for *"IoMT-SecAlarmBench: A Counterfactual Benchmark for
Integrity Attacks in IoMT"* (IEEE BigData 2026).

When an ICU alarm fires, an anomaly detector can tell you the reading is
unusual. It cannot tell you *why* — genuine deterioration, a sensor fault, or
a spoofed measurement that sits inside the physiological range. Answering that
needs counterfactual ground truth: what the alarm would have done had the
network stayed healthy. No naturally collected recording contains it, because
you never observe both outcomes of the same event.

This benchmark supplies it by construction. Controlled integrity attacks are
injected into genuine coupled ECG+PPG recordings, and **for every attacked
window the clean signal that would have been observed is retained**.

---

## At a glance

| | |
|---|---|
| Windows | 78,729 labelled 10 s windows, 5 s stride, 2-channel ECG+PPG at 125 Hz |
| Classes | 39,882 attack · 32,875 physiology-normal · 5,350 fault/artifact · 622 physiology-event |
| Spine | 627 CinC 2015 records carrying the ECG+PPG core pair (231 true / 396 false alarms) |
| Injection space | {replay} ∪ (4 morphologies × 4 severity levels × 2 plausibility regimes) |
| Splits | patient-wise 436 / 91 / 100, stratified by arrhythmia type |
| Counterfactual | exact clean pre-injection signal retained per attacked window |

**Headline results.** No detector among five method families consistently
exceeds the performance limit: replay attacks and low-amplitude transient
spikes sit at 1.2–1.7× chance for all six baselines. The strongest detector on
hard cases flags fault/artifact windows at **5.3× its own false-alarm rate on
normal data** — the attack-versus-fault ambiguity, localised to a specific,
measurable signal relationship.

---

## Install

```bash
git clone https://github.com/techosystem/IoMT-SecAlarmBench.git
cd IoMT-SecAlarmBench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is pinned to the CUDA 12.8 build used for the reported runs; see
`docs/REPRODUCE.md` for the index URL. The classical baseline (Isolation
Forest), the generator, and the leakage audit run on CPU without PyTorch.

## Get the data

Nothing is redistributed here (see `LICENSE-DATA.md`). Fetch the sources:

```bash
bash scripts/get_data.sh          # downloads CinC 2015 into data/cinc2015/
```

WUSTL-EHMS-2020 must be requested from its authors and placed at
`data/wustl/wustl-ehms-2020.csv`.

## Build the benchmark

```bash
python src/injection_generator.py --config configs/benchmark.yaml --out data/benchmark/
```

Deterministic under the configured seed. Expect 78,729 windows; the generator
prints the class histogram so you can check it against the table above.

## Reproduce the paper

```bash
python scripts/run_all.py --config configs/benchmark.yaml --seeds 0 1 2 3 4 --deterministic
```

This writes a timestamped directory under `results/` containing a
`manifest.json` that records configuration, package versions, GPU, and the
MD5 of every source module — including the severity constants of Table I.
Reported runs are `20260730-205501` (Table III) and `20260730-222727`
(Table IV). Per-component entry points:

| Paper section | Command |
|---|---|
| RQ1 detection (Table III) | `python src/rq1_report.py` |
| RQ2 stealthiness (Table IV) | `python src/eval_harness.py --matched-far 0.10` |
| RQ3 discrimination (Table V) | `python src/rq3_discrimination.py` |
| RQ4 efficiency | `python src/eval_harness.py --profile` |
| Leakage audit | `python src/wustl_leakage.py` |
| Attribution (deferred; see §VIII) | `python src/rq2_attribution_v3.py` |
| Figures | `make figures` |

**Read `docs/REPRODUCE.md` before comparing numbers.** Two rules are not
optional: compare detectors threshold-free or at a matched false-alarm budget,
never at each method's own max-F₁ point (it moves by up to ±0.234 across seeds
while threshold-free AUPRC moves by ≤0.013), and report AUPRC against its
chance floor, which differs per subset.

## Layout

```
src/          generator, adapters, detectors, evaluation  (flat imports; run from repo root)
scripts/      get_data.sh, run_all.py
configs/      benchmark.yaml, detectors.yaml
figures/      figure generation scripts
docs/         REPRODUCE.md
tests/        smoke tests
results/      run outputs (gitignored except manifest.json)
DATASHEET.md  full datasheet, following Gebru et al.
```

## Known issues

Recorded openly, and discussed in the paper:

- **Single-target injection.** All attacks in this release target the
  pulsatile channel, so channel-level localisation is degenerate — a predictor
  that always answers "PPG" scores AC@1 = 1.000. No attribution baseline is
  reported for this reason. Non-degenerate attribution needs the target to
  vary across channels.
- **Severity saturation.** In 6 of 16 crafted cells the scheduled magnitude
  reaches the plausibility bound, so in-bound FDI is clipped and L3–L4 are
  ordinal rather than proportional.
- **Alarm-level counterfactual oracle** is specified but not implemented.
- **Single corpus.** All physiological results come from CinC's 5–5.5 min
  alarm-centred records.

## Citing

See `CITATION.cff`, and cite the source corpora as well.

## Not for clinical use

Recommended for benchmarking detection, discrimination, attribution and
leakage auditing. **Not** for clinical deployment decisions, and not for
training alarm systems without prospective validation.
