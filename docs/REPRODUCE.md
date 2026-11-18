# Reproducing the reported results

The results reported in the paper are in `results/paper/`. They
were produced by `scripts/reproduce_paper.py` with deterministic kernels and
seeds 0–4 on a single NVIDIA A100-SXM4-80GB.

---

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Reference environment: Ubuntu 22.04.5 LTS, Python 3.12.13, PyTorch 2.11.0 /
CUDA 12.8, NumPy 2.0.2, pandas 2.2.2, scikit-learn 1.6.1, SciPy 1.15.3,
wfdb 4.1.2.

Determinism requires this before launching anything that touches cuDNN:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
```

Deterministic kernels cost 30–70% extra training time for the recurrent and
attention-based models. **Timings from a non-deterministic run are not
comparable to the reported ones.**

## 2. Data

```bash
make data          # CinC 2015 into data/cinc2015/; WUSTL is a manual step, see the script output
```

## 3. Run

```bash
make paper         # = python scripts/reproduce_paper.py
```

The script runs five stages in order. Each can be run alone with `--stages`:

| Stage | What it does | Time (A100) |
|---|---|---|
| `labels` | writes `data/benchmark/secalarmbench_labels.csv` and `secalarmbench_labels_multichannel.csv` | minutes |
| `by_setting` | each detector trained once, all four settings at a 5% validation FAR | minutes |
| `analyses` | `extras.run_all` with the built-in beat detectors | 35–45 min |
| `neurokit2` | `extras.run_all` again with NeuroKit2 ECG / Elgendi PPG beats; fits are cached | ~25 min |
| `breakdown` | per-alarm-type results and alarm effects by morphology | ~2 min |

Detector scores, beat detections and the position-matched attacks are cached in
`results/paper/` (`scores/`, `*.parquet`), so an interrupted run resumes
where it stopped. These cache files are not committed.

Check the class counts the build prints against Table II of the paper:

| Class | Expected |
|---|---|
| physiology-normal | 32,875 |
| physiology-event | 622 |
| fault / artifact | 5,350 (4,368 artifact-confirmed, 982 monitor false alarm) |
| attack, C1-PPG / C1-ECG / C2 / C3 | 39,882 / 39,882 / 38,709 / 44,574 |

If these differ, stop: the splits and every chance floor downstream depend on
them. The held-out C1-PPG test split should be 100 records / 11,825 windows, of
which 6,392 are attacks (prevalence 0.541).

### The single-setting pipeline

`scripts/run_all.py` (`make all`) runs the C1-PPG detection, matched-budget,
multi-seed, discrimination, leakage and pooling-ablation stages into a
timestamped `results/<run-id>/` directory. Each run writes a `manifest.json`
recording configuration, package versions, GPU and the MD5 of every source file,
including `configs/benchmark.yaml`. To check that a results directory still
matches your working tree:

```bash
python src/manifest.py verify results/<run-id>/manifest.json
```

A non-zero exit means the code changed after the run.

---

## 4. Two rules for comparing numbers

These are findings of the paper, not style preferences.

### Never compare at each method's own max-F1 operating point

Across five seeds, the max-F1 threshold changes the false-alarm rate on normal
windows by up to **±0.234** and on event windows by up to **±0.222** (both USAD),
while threshold-free AUPRC moves by at most **±0.020**. Max-F1 also rewards
detectors that simply alarm more. **Compare threshold-free, or at a matched
false-alarm budget calibrated on validation normals and applied unchanged to
test.**

### Always report AUPRC against its chance floor

The no-skill AUPRC is the positive-class prevalence, and it differs for every
attack subset (C1-PPG test split):

| Subset | Chance floor |
|---|---|
| Aggregate (vs. physiology-normal) | 0.541 |
| Aggregate (fault added to negatives) | 0.512 |
| Replay | 0.065 |
| FDI-spike | 0.122 |
| FDI-spike L1 | 0.033 |
| FDI / spoofing sustained | 0.293 |

A bare "AUPRC 0.11 on replay" is uninterpretable; against a 0.065 floor it is
1.7× chance. Values are comparable down a column, not across subsets.

For attacker settings, use host-matched AUROC: each attacked window against its
own clean host window (chance 0.5).

## 5. Seed behaviour to expect

USAD is bimodal, not merely noisy: across five seeds its normal-window
false-alarm rate spans 0.311–0.954, so one run in five alarms on nearly
everything, while its AUPRC stays stable. **A single-seed evaluation will not
show it.** Report mean ± sd over at least three seeds; cuDNN offers no
deterministic RNN backward pass, so single-run values are not reproducible even
at a fixed seed without the flags in §1.

## 6. Notes

- **Label modes.** Result files ending in `_v1` count every physiology-normal
  window as normal; `_corrected` removes the 1,110 alarm-zone windows.
- **Discrimination run order.** Table VII reports a rerun of the default forest
  on the combined four-setting table; the event count it recognizes differs from
  the run reported in the text because of sample order (see the Table VII
  caption).
- **Channel attribution.** The paper reports the baseline in
  `extras.attribution_baseline`. `src/attribution.py` (PCMCI+, DoWhy-GCM) is
  provided for follow-up work and is not reported in the paper.
