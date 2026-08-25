# Reproducing the reported results

Reported runs: `20260730-205501` (Table III, detection and efficiency) and
`20260730-222727` (Table IV, threshold-free and matched-budget). Both used
deterministic kernels and seeds 0–4 on a single NVIDIA A100-SXM4-80GB.

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
attention-based models (USAD 26→44 s, GDN 141→186 s, TranAD 61→88 s).
**Timings from a non-deterministic run are not comparable to the reported
ones.**

## 2. Data, then benchmark

```bash
make data          # CinC 2015; WUSTL is a manual request, see the script output
make benchmark     # -> 78,729 windows
```

Check the class histogram the generator prints:

| Class | Expected |
|---|---|
| physiology-normal | 32,875 |
| physiology-event | 622 |
| fault / artifact | 5,350 |
| attack | 39,882 |
| **Total** | **78,729** |

If these differ, stop — the splits and every chance floor downstream depend on
them. The held-out test split should be 100 records / 11,825 windows, of which
6,392 are attack, giving an attack prevalence of 0.541.

## 3. Run

```bash
make all           # five seeds, deterministic
```

Each run writes `results/<run-id>/manifest.json` recording configuration,
package versions, GPU, and the MD5 of every source file — including
`configs/benchmark.yaml`, which holds the Table I severity constants. To check
that a results directory still matches your working tree:

```bash
python src/manifest.py verify results/<run-id>/manifest.json
```

A non-zero exit means the code moved after the run and the numbers no longer
correspond to what is checked out.

---

## 4. Two rules for comparing numbers

These are findings of the paper, not style preferences. Ignoring them will
produce numbers that look like ours but are not.

### Never compare at each method's own max-F₁ operating point

It is the single largest source of instability in the whole study. Across five
seeds:

| Quantity | Spread |
|---|---|
| False-alarm rate at max-F₁ | up to **±0.234** |
| Detection rate at matched 10% FAR | ≤ ±0.034 |
| Threshold-free AUPRC | ≤ **±0.013** |

An order of magnitude apart. Worse, max-F₁ rewards detectors that simply alarm
more: OmniAnomaly reaches a replay detection rate of 0.71 at its own operating
point while false-alarming on 51% of normal windows and 69% of genuine
physiological-event windows. At a matched budget the same detector gets 0.202
and ranks fourth of six threshold-free. **Compare threshold-free, or at a
matched false-alarm budget calibrated on validation normals and applied
unchanged to test.**

### Always report AUPRC against its chance floor

The no-skill AUPRC is the positive-class prevalence, and it is different for
every attack subset:

| Subset | Chance floor |
|---|---|
| Aggregate (vs. physiology-normal) | 0.541 |
| Aggregate (fault added to negatives) | 0.512 |
| Replay | 0.065 |
| FDI-spike | 0.122 |
| FDI-spike L1 | 0.033 |
| FDI / spoofing sustained | 0.293 |

A bare "AUPRC 0.11 on replay" is uninterpretable; against a 0.065 floor it is
1.7× chance. Values are comparable **down** a column, not across subsets —
which is exactly why the paper's lift figure divides by the floor.

## 5. Seed behaviour to expect

USAD is bimodal, not merely noisy: across five seeds its normal-window
false-alarm rate spans 0.311–0.954, so one run in five alarms on nearly
everything, while AUPRC stays inside 0.807–0.836. This is reported as a
property of the benchmark, not excluded. **A single-seed evaluation will not
show it.** Report mean ± sd over at least three seeds; cuDNN offers no
deterministic RNN backward pass, so single-run values are not reproducible
even at a fixed seed without the flags in §1.

## 6. Known non-reproducible items

- **Figure 2 morphology gallery** (if you regenerate it) samples specific
  windows; the panel indices are set in `figures/make_fig2.py`.
- **RQ2 attribution** (`src/rq2_attribution_v3.py`) is deferred in the paper,
  not reported as a baseline. All attacks target the pulsatile channel, so a
  constant predictor achieves AC@1 = 1.000 and the permutation control is
  uninformative by arithmetic. The script runs, but its output is not a result.
