# Results reported in the paper

Written by `scripts/reproduce_paper.py` (seeds 0–4, deterministic kernels, one
A100). Caches (`scores/`, `*.parquet`) are not committed.

| File | Contents | Paper |
|---|---|---|
| `detection_summary_{v1,corrected}.csv` | mean and sd over seeds of every detection metric, per detector | Tables III–V |
| `detection_per_seed_{v1,corrected}.csv` | the same metrics per seed | Tables III–V |
| `bootstrap_ci_{v1,corrected}.csv` | record-grouped bootstrap 95% CIs (1,000 draws) | Tables III–V, text |
| `grid_morph_level_{v1,corrected}.csv` | host-matched AUROC and detection rate per setting, morphology and level | Fig. 2 |
| `rq1_by_setting_<detector>.csv` | each detector at a 5% validation FAR, per attacker setting | Table V |
| `alarm_oracle_validation.csv` | reference monitors against the CinC verdicts | Sec. VI-D |
| `alarm_oracle_effects.csv` | alarm creation and suppression per setting and attack family | Table VI |
| `*_neurokit2.csv` | the alarm oracle with NeuroKit2 ECG / Elgendi PPG beats | Sec. VI-B check |
| `rq4_variants.json` | discrimination variants: confusion matrices, per-class F1, CIs | Table VII, Sec. VI-E |
| `attribution_baseline.csv` | channel-attribution baseline on C1-PPG + C1-ECG | Sec. VI-F |
| `wustl_audit.csv`, `wustl_meta.json` | WUSTL-EHMS-2020 leakage audit, five split seeds | Table VIII |
| `label_sensitivity_host_zone.csv`, `label_sensitivity_refit.csv` | alarm-zone sensitivity checks | Sec. VI-F |
| `hyperparameters.json` | settings read from the detector objects, and mean fit times | Table II |
| `beat_floors.json` | beat-detector amplitude floors from the training normals | Table II |

`_v1` files count every physiology-normal window as normal (the main tables);
`_corrected` files remove the 1,110 alarm-zone windows.
