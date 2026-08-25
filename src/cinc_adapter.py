"""
IoMT-SecAlarmBench — CinC/MIMIC adapter + benchmark assembler (v2)
==================================================================

Turns loaded records (from cinc_loader / mimic_loader) into generator inputs, runs the
factorial grid, and assembles ONE labeled table fusing the four classes:

    physiology_normal   clean windows away from the alarm region
    physiology_event    clean windows over a TRUE-alarm region
    fault_artifact      clean windows over a FALSE-alarm region OR high-artifact windows
    attack              injected windows (from injection_generator v2)

Replay banks are keyed (record_id, channel) and passed whole to generate_benchmark, so
both same-patient and cross-patient replay are available.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from injection_generator import Window, GeneratorConfig, generate_benchmark

FS = 125.0


# --------------------------------------------------------------------------- #
def load_cinc_records():
    """Demo stub (ALL FIVE alarm types). Real runs pass records= from cinc_loader."""
    rng = np.random.default_rng(0)
    specs = [("Asystole", False), ("Bradycardia", True), ("Tachycardia", True),
             ("VentricularTachycardia", False), ("VentricularFlutterFib", False)]
    recs = []
    for i, (atype, is_true) in enumerate(specs):
        T = int(FS * 300)
        t = np.arange(T) / FS
        ph = 2 * np.pi * (75 / 60.0) * t
        ecg = np.sin(ph) + 0.02 * rng.standard_normal(T)
        ppg = 0.9 * np.sin(ph - 0.6) + 0.02 * rng.standard_normal(T)
        sig = np.vstack([ecg, ppg])
        sig = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
        recs.append(dict(record_id=f"rec{i:03d}", signal=sig, channels=["ECG", "PPG"],
                         alarm_type=atype, true_alarm=is_true,
                         alarm_onset_sample=int(T * 0.85), fs=FS))
    return recs


# --------------------------------------------------------------------------- #
def build_windows(records, win_sec=10.0, hop_sec=5.0, fs=FS,
                  host_class="physiology_normal", artifact_frac=0.10,
                  max_hosts_per_record=None, seed=20260723):
    """Slice records into windows, label clean windows, build replay banks."""
    W = int(round(win_sec * fs))
    H = int(round(hop_sec * fs))
    rng = np.random.default_rng(seed)
    hosts, banks, clean_rows, host_labels = [], {}, [], {}

    for r in records:
        C, T = r["signal"].shape
        onset = r["alarm_onset_sample"]
        amask = r.get("artifact_mask")
        rec_fs = float(r.get("fs", fs))
        bank_segs = {ch: [] for ch in r["channels"]}
        candidates = []

        for w_idx, start in enumerate(range(0, T - W + 1, H)):
            seg = r["signal"][:, start:start + W]
            art_frac = 0.0
            if amask is not None:
                art_frac = float(amask[:, start:start + W].any(axis=0).mean())

            if art_frac > artifact_frac:
                cls = "fault_artifact"
            elif start + W > onset:
                cls = "physiology_event" if r["true_alarm"] else "fault_artifact"
            else:
                cls = "physiology_normal"

            wid = f"{r['record_id']}_w{w_idx:03d}"
            clean_rows.append(dict(
                record_id=r["record_id"], window_id=wid, cls=cls,
                alarm_type=r["alarm_type"], true_alarm=r["true_alarm"],
                artifact_frac=round(art_frac, 4),
                family="none", morphology="none", target_channel="",
                level=-1, realised_mag_sigma=float("nan"), onset=-1, duration=0,
                within_plausibility=True, mechanism_preservation=float("nan")))

            if cls == host_class:
                candidates.append((wid, seg))
                for ci, ch in enumerate(r["channels"]):
                    bank_segs[ch].append(seg[ci].copy())

        chosen = candidates
        if max_hosts_per_record is not None and len(candidates) > max_hosts_per_record:
            idx = np.unique(np.round(
                np.linspace(0, len(candidates) - 1, max_hosts_per_record)).astype(int))
            chosen = [candidates[i] for i in idx]

        for wid, seg in chosen:
            hosts.append(Window(seg.copy(), list(r["channels"]), r["record_id"], rec_fs, wid))
            host_labels[wid] = dict(alarm_type=r["alarm_type"], true_alarm=r["true_alarm"])

        for ch, segs in bank_segs.items():
            if segs:
                banks[(r["record_id"], ch)] = np.vstack(segs)

    return hosts, banks, clean_rows, host_labels


# --------------------------------------------------------------------------- #
def assemble_benchmark(records=None, cfg=None, target_channels=("PPG",),
                       win_sec=10.0, hop_sec=5.0, max_hosts_per_record=2,
                       artifact_frac=0.10):
    cfg = cfg or GeneratorConfig()
    records = records if records is not None else load_cinc_records()
    if not records:
        raise ValueError("assemble_benchmark received 0 records — fill your loader or call "
                         "with no args for the demo stub.")

    hosts, banks, clean_rows, host_labels = build_windows(
        records, win_sec=win_sec, hop_sec=hop_sec,
        max_hosts_per_record=max_hosts_per_record, artifact_frac=artifact_frac)

    if not hosts:
        avail = sorted({ch for r in records for ch in r["channels"]})
        raise ValueError(f"No host windows from {len(records)} records ({len(clean_rows)} "
                         f"clean). Check alarm_onset_sample and channels {avail}.")

    attacked, atk = generate_benchmark(
        hosts, target_channels=list(target_channels), cfg=cfg, replay_banks=banks)

    if atk.empty:
        avail = sorted({ch for w in hosts for ch in w.channels})
        raise ValueError(f"0 attacks generated. target_channels={list(target_channels)} "
                         f"not in data (available: {avail}).")

    atk = atk.copy()
    atk.insert(0, "cls", "attack")
    atk["alarm_type"] = atk["window_id"].map(lambda w: host_labels.get(w, {}).get("alarm_type", ""))
    atk["true_alarm"] = atk["window_id"].map(lambda w: host_labels.get(w, {}).get("true_alarm", pd.NA))

    table = pd.concat([pd.DataFrame(clean_rows), atk], ignore_index=True, sort=False)

    # Unify the grouping key: clean rows carry record_id, attack rows carry patient_id;
    # for CinC they are identical. Backfill both so EVERY row has a complete
    # record_id AND patient_id -> patient-wise splitting works across all classes.
    if "patient_id" not in table.columns:
        table["patient_id"] = pd.NA
    table["record_id"] = table["record_id"].fillna(table["patient_id"])
    table["patient_id"] = table["patient_id"].fillna(table["record_id"])
    # tidy empties: no target channel / not-an-attack rows
    table["target_channel"] = table["target_channel"].fillna("")
    return table, attacked


def table2_composition(table):
    print("=== Class composition ===")
    print(table["cls"].value_counts().to_string())
    atk = table[table["cls"] == "attack"]
    if len(atk):
        print("\n=== Attack breakdown: family x level ===")
        lv = atk["level"].map(lambda v: "replay" if v == 0 else f"L{int(v)}")
        print(pd.crosstab(atk["family"], lv).to_string())
        print("\n=== FDI/spoofing by morphology ===")
        fs_only = atk[atk["family"].isin(["fdi", "spoofing"])]
        print(pd.crosstab(fs_only["morphology"], fs_only["family"]).to_string())
        if (atk["family"] == "replay").any():
            print("\n=== Replay by source ===")
            print(atk[atk["family"] == "replay"]["replay_source"].value_counts().to_string())
        print("\n=== Clean classes by alarm type ===")
        clean = table[table["cls"] != "attack"]
        print(pd.crosstab(clean["alarm_type"], clean["cls"]).to_string())


if __name__ == "__main__":
    table, attacked = assemble_benchmark()
    print(f"total labeled windows: {len(table)}  "
          f"(clean {int((table['cls']!='attack').sum())}, attack {int((table['cls']=='attack').sum())})\n")
    table2_composition(table)
