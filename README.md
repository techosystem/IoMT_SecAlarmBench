# IoMT-SecAlarmBench

**A semi-synthetic counterfactual benchmark for integrity attacks in the Internet of Medical Things.**

Companion code for the paper *"IoMT-SecAlarmBench: A Semi-Synthetic Counterfactual
Benchmark for Integrity Attacks in IoMT"* (Ugwuabonyi and Perkins, 2026).

When an ICU alarm fires, an anomaly detector can tell you the reading is unusual,
but not *why*: genuine deterioration, a drifting or detached sensor, or a malicious
value injected along the measurement path. Answering that needs counterfactual
ground truth, which no naturally collected recording contains.

This benchmark supplies it by construction. Controlled integrity attacks are
injected into authentic ECG+PPG recordings from PhysioNet/CinC 2015, and **every
attacked window keeps the clean waveform it replaced**. That enables evaluation at
two levels: the waveform itself, and the decision of a specified rate-alarm
algorithm applied to both the clean and the attacked window.

---

## At a glance

| | |
|---|---|
| Source | CinC 2015, 627-record ECG+PPG core cohort (231 true / 396 false alarms) |
| Windows | 10 s, 5 s stride, 2 × 1,250 samples at 125 Hz |
| Clean classes | physiology-normal 32,875 (1,110 in the alarm zone) · physiology-event 622 · artifact-confirmed fault 4,368 · monitor false alarm 982 |
| Attacks | 1,173 host windows, reused across every condition and setting |
| Attack grid | {replay} ∪ (4 morphologies × 4 severity levels × {FDI, spoofing}) |
| Attacker settings | C1-PPG 39,882 · C1-ECG 39,882 · C2 38,709 · C3 44,574 attack windows |
| Total | about 200,000 labeled windows across the four settings |
| Split | 436 / 91 / 100 records, record-disjoint, stratified by alarm type |
| Counterfactuals | clean waveform per attacked window; alarm decision g(clean) vs g(attacked) |

**Attacker settings**

| Setting | Attacker capability |
|---|---|
| C1-PPG, C1-ECG | controls one device stream |
| C2 | controls both channels, alters each independently |
| C3 | controls both channels and keeps them consistent: joint replay, coordinated FDI/spoofing (PPG delayed by the pulse-transit lag), and heart-rate warping |

**Headline results**

- Six unsupervised detectors struggle with replay attacks and short spikes, and a
  dedicated heart-rate and pulse-transit coherence baseline does not resolve these
  failures.
- Joint ECG+PPG replay and coordinated heart-rate warping perform at chance against
  their matched clean hosts (host-matched AUROC 0.49–0.55).
- A monitor that requires ECG and PPG agreement loses 38% of the rate alarms raised
  on true-alarm records when only the PPG stream is replayed, and 48% under
  coordinated replay.
- A supervised classifier separates attacks well (F1 0.97) but almost never
  identifies genuine physiological events, even when attacks and controls are
  matched by position (event F1 up to 0.02).

---

## Install

```bash
git clone https://github.com/techosystem/IoMT_SecAlarmBench.git
cd IoMT_SecAlarmBench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is pinned to the CUDA 12.8 build used for the reported runs; see
`docs/REPRODUCE.md` for the index URL. The generator, Isolation Forest, the
coherence baseline and the WUSTL audit run on CPU without PyTorch.

## Get the data

Nothing is redistributed here (see `LICENSE-DATA.md`).

```bash
bash scripts/get_data.sh          # downloads CinC 2015 into data/cinc2015/
```

WUSTL-EHMS-2020 is used only for the leakage audit. Place it at
`data/wustl/wustl-ehms-2020.csv`; the audit is skipped if the file is missing.

## Reproduce the paper

```bash
python scripts/reproduce_paper.py
```

This builds all four attacker settings, writes the labels tables to
`data/benchmark/`, and runs every analysis into `results/paper/`. It takes
about 35–45 minutes for the main analyses on one A100 with five seeds, and every
stage is resumable. Run single stages with `--stages`, for example
`--stages analyses --seeds 0` for a quick first pass.

| Paper | Produced by | Output in `results/paper/` |
|---|---|---|
| Table II (settings) | `extras` step 11 | `hyperparameters.json` |
| Table III (detection, C1-PPG) | `extras` step 4 | `detection_summary_v1.csv`, `bootstrap_ci_v1.csv` |
| Table IV (attack subsets) | `extras` step 4 | `detection_summary_v1.csv` |
| Fig. 2 (morphology × level) | `extras` step 4 | `grid_morph_level_v1.csv`; draw with `python figures/make_fig2_grid.py` |
| Table V (attacker settings) | `extras` step 4, `by_setting` stage | `detection_summary_v1.csv`, `rq1_by_setting_*.csv` |
| Table VI (alarm counterfactuals) | `extras` step 8 | `alarm_oracle_effects*.csv`, `alarm_oracle_validation*.csv` |
| Table VII (discrimination controls) | `extras` step 7 | `rq4_variants.json` |
| Sec. VI-F (channel attribution) | `extras` step 9 | `attribution_baseline.csv` |
| Table VIII (WUSTL audit) | `extras` step 10 | `wustl_audit.csv`, `wustl_meta.json` |
| Alarm-zone sensitivity | `extras` step 5 | `label_sensitivity_*.csv`, `*_corrected.csv` |
| Per-alarm-type results | `breakdown` stage | printed, and `alarm_type_*.csv` |

Files ending in `_v1` count every physiology-normal window as normal (the main
tables); `_corrected` removes the alarm-zone windows. Files ending in `_neurokit2`
repeat the alarm oracle with NeuroKit2 ECG and Elgendi PPG beat detection.

**Read `docs/REPRODUCE.md` before comparing numbers.** Compare detectors
threshold-free or at a matched false-alarm budget, never at each method's own
max-F1 point (across seeds it moves the false-alarm rate by up to ±0.234, while
threshold-free AUPRC moves by at most ±0.020), and report AUPRC against its chance
floor, which differs per subset.

## Layout

```
src/
  cinc_loader.py           CinC 2015 records -> filtered, resampled, z-scored channels
  cinc_adapter.py          windows, labels and the benchmark table (all settings)
  injection_generator.py   attack grid: replay, FDI, spoofing (C1)
  injection_multichannel.py  C1-ECG, C2 and C3 attack settings
  eval_harness.py          features, record-disjoint split, Detector interface, detection
  deep_detectors.py        LSTM-AE, USAD, GDN, TranAD, OmniAnomaly
  extras.py                every analysis in the paper (steps 0-11)
  detection_report.py      C1-PPG detection report with chance floors and record CIs
  discrimination.py        attack / fault / physiology classifier
  leakage_audit.py         WUSTL-EHMS-2020 identifier-leakage audit
  attribution.py           causal channel attribution (PCMCI, DoWhy-GCM), for follow-up work
  attribution_rigor.py     permutation control and CIs for attribution
  manifest.py              run provenance (config, versions, source hashes)
scripts/
  get_data.sh              download CinC 2015
  reproduce_paper.py       the four-setting pipeline behind the paper
  run_all.py               single-setting (C1-PPG) pipeline with multi-seed and ablation stages
figures/make_fig2_grid.py  Fig. 2
configs/                   design constants (Table I) and detector settings (Table II)
metadata/                  labels of the C1-PPG windows (secalarmbench_labels.csv)
results/paper/       the results reported in the paper
docs/REPRODUCE.md          environment, determinism, and rules for comparing numbers
DATASHEET.md               datasheet (Gebru et al.)
```

Run scripts from the repository root; modules in `src/` use flat imports.

## Limitations

Recorded openly, and discussed in the paper:

- Attacks are semi-synthetic and predefined, not collected from real deployments.
- The alarm oracle covers only the three rate-based CinC alarms (asystole, extreme
  bradycardia, extreme tachycardia) and is a reference implementation, not a
  commercial monitor.
- No synchronized network traffic.
- All physiological data come from 5–5.5 min alarm-centered CinC records. Splits
  are record-disjoint; CinC has no patient identifiers, so patient-disjointness is
  unverified.
- In six of the sixteen FDI cells the in-bound magnitude reaches the plausibility
  bound, so L3–L4 are ordinal rather than proportional.
- Detector hyperparameters were fixed, not tuned; among the pooled deep models only
  LSTM-AE was also evaluated without temporal pooling.
- The coherence baseline and the alarm oracle depend on beat detection within 10 s
  windows.

## Citing

```bibtex
@misc{ugwuabonyi2026iomtsecalarmbench,
  title         = {{IoMT-SecAlarmBench}: A Semi-Synthetic Counterfactual Benchmark for Integrity Attacks in {IoMT}},
  author        = {Ugwuabonyi, Emmanuel C. and Perkins, Dmitri},
  year          = {2026},
  eprint        = {2608.26416},
  archivePrefix = {arXiv}
}
```

Please also cite the source corpora (CinC 2015 and, if you use the audit,
WUSTL-EHMS-2020). See `CITATION.cff`.

## Licence

Code: MIT (`LICENSE`). Data: governed by the source corpora's terms
(`LICENSE-DATA.md`).

## Not for clinical use

Intended for benchmarking detection, discrimination, attribution and leakage
auditing. **Not** for clinical deployment decisions, and not for training alarm
systems without prospective validation.
