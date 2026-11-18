# Datasheet for IoMT-SecAlarmBench

Following Gebru et al., *Datasheets for Datasets* (CACM 64(12), 2021), with
the medical extensions of DAIMS (arXiv:2501.14094). This datasheet describes
release v2.0.0.

---

## 1. Motivation

**For what purpose was the dataset created?**
To supply the counterfactual ground truth that root-cause attribution in IoMT
requires and no existing corpus provides. When an ICU alarm fires, the
clinically decisive question is counterfactual — *would this alarm have fired
had the network stayed healthy?* — and that answer is unobservable in any
naturally collected recording, because one never observes both the factual and
the counterfactual outcome of the same event. It can only be established by
deliberately introducing the attack into otherwise authentic data.

**What gap does it fill?**
Five requirements were used to assess existing corpora: (R1) dual-modality
cyber + physiological data; (R2) ground truth separating cyberattack, sensor
fault, and genuine physiology rather than merely attack vs. benign; (R3)
per-patient structure; (R4) integrity attacks on the measurement-to-record
path, not only network flooding; (R5) counterfactual ground truth. No existing
corpus satisfies all five. WUSTL-EHMS-2020 is closest but lacks per-patient
waveform structure and measurement-path attacks; PhysioNet corpora have rich
physiology but no attacks; SWaT/WADI and SMAP/MSL/SMD/PSM have neither
patients nor medical relevance.

**Who created it and who funded it?**
Department of Computer Science and Electrical Engineering, University of
Maryland, Baltimore County. *(Add grant numbers before release.)*

---

## 2. Composition

**What do instances represent?**
A 10-second window of two coupled physiological channels (ECG lead and
PPG/PLETH) at 125 Hz, from a single ICU record, with a 5-second stride between
consecutive windows.

**How many instances are there?**

| Class | Windows |
|---|---|
| physiology-normal | 32,875 (1,110 in the alarm zone) |
| physiology-event | 622 |
| fault: artifact-confirmed | 4,368 |
| fault: monitor false alarm | 982 |
| **attack, C1-PPG** | **39,882** |
| — replay (same / cross-patient) | 2,346 (1,173 / 1,173) |
| — FDI (4 morphologies × L1–L4) | 18,768 |
| — spoofing (4 morphologies × L1–L4) | 18,768 |
| **attack, C1-ECG** (same 34 conditions on ECG) | **39,882** |
| **attack, C2** (both channels, independent) | **38,709** |
| **attack, C3** (both channels, consistent) | **44,574** |
| **Total** | **201,894** |

All attacks use the same 1,173 host windows (up to two clean pre-alarm windows
per record), so attack windows are variants of these hosts, not independent
observations. C3 adds joint replay and a coordinated heart-rate warp to the
grid.

**What does each instance consist of?**
Raw 2-channel waveform; 22 engineered per-window features (ten per channel
plus ECG–PPG zero-lag cross-correlation and best-lag position); the four-way
cause label; attack family, morphology and regime; injected channel, onset,
duration and severity level; `artifact_frac`; the retained clean pre-injection
signal for attacked windows; the CinC-native alarm type and true/false
verdict; and unified record/patient identifiers.

**Is there a label or target?**
Yes — a four-way cause label in {physiology-normal, physiology-event,
fault/artifact, attack}, assigned deterministically:

- entirely before the 300 s alarm onset → **physiology-normal**. Windows ending
  in 290–300 s, where CinC annotates the event onset, are flagged as
  *alarm-zone* windows; the analyses report results with and without them.
- post-onset in a record with a native *true* alarm verdict → **physiology-event**
- post-onset in a *false*-alarm record, **or** artifact mask covering >10% of
  samples → **fault/artifact**
- injected → **attack**

The mask test is evaluated first, so `artifact_frac` partitions the fault
class exactly: >0.10 mask-driven (4,368 windows), ≤0.10 verdict-driven (982).

**Is any information missing?**
Yes, and by design. Exact counterfactual ground truth exists at the signal
level for injected attacks only; there is no equivalent counterfactual for
genuine physiological events, because no intervention was performed on them.
Alarm-level counterfactuals are computed with a specified reference monitor
for the three rate alarms (asystole, extreme bradycardia, extreme
tachycardia); ventricular alarms, which need beat-morphology analysis, are not
covered.

**Are there errors, noise, or redundancies?**
The fault class is heterogeneous *by design*, combining genuine sensor
corruption (mask-driven) with monitor false alarms (verdict-driven). These
behave measurably differently: for Isolation Forest, artifact-confirmed windows
reach AUROC 0.661 against normals, while monitor false alarms stay near chance
(0.48–0.56) for every detector. The two are separable via `artifact_frac`. Monitor false-alarm windows are not
necessarily clean signal; their false-alarm status selects for unusual
patterns.

**Is the dataset self-contained?**
No. It is generated from PhysioNet/CinC 2015 and WUSTL-EHMS-2020, which are
**not redistributed**. See `LICENSE-DATA.md`.

**Does it contain confidential or offensive content?**
No. Source recordings are de-identified by their providers. No new human data
was collected.

**Does it identify subpopulations?**
Not directly. CinC 2015 records carry arrhythmia type (asystole, extreme
bradycardia, extreme tachycardia, ventricular tachycardia, ventricular
flutter/fibrillation) and a true/false alarm verdict, but no demographic
attributes. **A consequence worth stating plainly: no demographic fairness
audit of detector performance is possible on this benchmark**, because the
required attributes do not exist in the source.

---

## 3. Collection process

**How was the data acquired?**
The physiological substrate is observed ICU data collected by the CinC 2015
challenge organisers; each alarm was reviewed by at least two expert
annotators who agreed on the true/false verdict. All attack and fault
*content* is generated by us through controlled injection.

**Why controlled injection rather than real attacks?**
Three reasons, all binding: no public IoMT physiological corpus carries native
integrity-attack labels; real incidents are legally reportable and therefore
not shareable; and red-teaming live clinical hardware is unacceptable.
Controlled injection into genuine recordings is the established standard for
this problem class.

**Over what timeframe?**
Source recordings predate this work. Benchmark generation and the reported
evaluation runs: July–October 2026.

**Was an ethical review conducted?**
No new human-subject data was collected; the work is secondary analysis of
publicly released de-identified corpora under their respective licences.
*(Insert IRB determination or exemption number if your institution issued one.)*

---

## 4. Preprocessing, cleaning, labelling

**What preprocessing was done?**
All channels resampled to a 125 Hz common base and cut into 10 s windows with
5 s stride, each carrying a validity flag. ECG band-pass filtered (zero-phase
4th-order Butterworth, 0.5–40 Hz); pulsatile channel 0.5–8 Hz. Channels
z-scored within record (verified σ̄ ≈ 1), with normalisation statistics
retained so bounds can be de-normalised to physical units.

**Were artifacts removed?**
**No — and this is a deliberate design choice.** Flat-line, saturation/clipping
and physiologically implausible segments (instantaneous HR ∉ [30, 200] bpm)
are *labelled* artifact/fault, never deleted, because they constitute the
fault class a detector must learn to separate from attacks. Windows exceeding
the artifact threshold are excluded from use as injection hosts.

**Was feature selection applied?**
No. Waveforms are preserved in full, because injection and cross-channel
coupling operate on the signal itself.

**Is the raw data available?**
The injection generator, preprocessing pipeline, evaluation harness and all
configuration are in this repository. Raw source waveforms come from the
providers under their own terms.

---

## 5. Uses

**What has the dataset been used for?**
Baseline evaluation of six unsupervised detectors (Isolation Forest, LSTM-AE,
USAD, OmniAnomaly, GDN, TranAD) and a heart-rate and pulse-transit coherence
baseline on detection, per-morphology and per-setting detection, alarm-level
counterfactuals, cause discrimination under position-matched controls, and a
channel-attribution baseline; plus an identifier-leakage audit of
WUSTL-EHMS-2020.

**What else could it be used for?**
Counterfactual root-cause attribution, signal-recovery evaluation against the
retained clean signal, causal discovery on coupled physiological channels, and
methodological work on operating-point selection in time-series anomaly
detection.

**Is there anything that could cause unfair treatment or harm?**
Three risks worth naming:

1. **Genuine deterioration is the failure mode.** In three-way discrimination,
   77.2% of genuine physiological deteriorations are misclassified as attacks
   and 20.7% as artifacts. A system trained naively on this benchmark would
   systematically mislabel the class where error is most clinically dangerous.
2. **No demographic attributes** means fairness cannot be audited here. Absence
   of evidence of disparity is not evidence of absence.
3. **Semi-synthetic attacks** are not observed attacks. The sim-to-real gap is
   unvalidated pending the planned blinded clinician study.

**Are there tasks for which it should not be used?**
**Clinical deployment decisions. Training alarm systems without prospective
validation. Any claim about real-world attack prevalence** — the class balance
here is a design parameter, not an epidemiological estimate.

---

## 6. Distribution

**How is it distributed?**
As generation code, not as data. Users regenerate locally from the source
corpora. Deterministic under a fixed seed.

**Under what licence?**
Code: MIT. Data: governed by the source corpora's terms; see `LICENSE-DATA.md`.

**Are there export controls or restrictions?**
None known beyond the source-dataset terms.

---

## 7. Maintenance

**Who maintains it, and how can they be contacted?**
Emmanuel C. Ugwuabonyi (ugwuabonyi@umbc.edu), University of Maryland,
Baltimore County. Issues and questions: GitHub issues on this repository.

**Will it be updated?**
Planned: extension to longer continuous ICU recordings (MIMIC-III Waveform),
synchronized network traffic, and dedicated causal attribution methods.

**How will updates be communicated?**
Through GitHub releases, each with a version tag. Archived releases carry a
Zenodo DOI so results remain pinned to an exact code state.

**Will older versions remain available?**
Yes — via git tags and archived DOIs. The results reported in the paper are in
`results/paper/` (release v2.0.0). The first release, with the C1-PPG setting
only, is tagged `v1.0.0`.

**Can others contribute?**
Yes, via pull request. Contributions that change generation behaviour must
update this datasheet and state the effect on the class histogram.
