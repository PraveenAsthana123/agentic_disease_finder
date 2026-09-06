#!/usr/bin/env python3
"""Generate a SYNTHETIC epilepsy clinical/governance dataset.

⚠️ SYNTHETIC DATA — generated for framework demonstration and pipeline testing
ONLY. Not real patients. Never use for clinical claims or as ground truth.

Covers every section of the EEG Technician / clinical data-collection form for
EPILEPSY, both modalities (EEG + video-EEG). Writes per-section CSVs + a
per-patient JSON bundle into a SEPARATE folder so it is never confused with
real data.

Output: clinical_data/synthetic/epilepsy/
Usage:  python scripts/generate_synthetic_epilepsy.py --n 30
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "clinical_data" / "synthetic" / "epilepsy"

RNG = random.Random(42)  # reproducible

AED = ["Levetiracetam", "Valproate", "Carbamazepine", "Lamotrigine", "Topiramate", "Oxcarbazepine", "Lacosamide"]
BZD = ["Clonazepam", "Diazepam", "Lorazepam", "Clobazam"]
SEIZURE_TYPES = ["Focal Aware", "Focal Impaired Awareness", "Focal to Bilateral Tonic-Clonic", "Generalized Tonic-Clonic"]
EPILEPSY_TYPES = ["Focal Epilepsy", "Generalized Epilepsy", "Combined", "Unknown"]
LOBES = ["Temporal", "Frontal", "Parietal", "Occipital", "Central"]
HEMI = ["Left", "Right", "Bilateral"]
ARTIFACTS = ["Eye Blink", "Muscle Artifact", "Movement Artifact", "ECG Artifact", "Electrode Pop", "Power Line Noise"]
MODALITIES = ["EEG", "video-EEG"]


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gen_patient(i: int) -> dict:
    pid = f"SYN-EPI-{i:04d}"
    onset = RNG.randint(2, 70)
    age = onset + RNG.randint(1, 30)
    gender = RNG.choice(["Male", "Female"])
    is_epilepsy = RNG.random() < 0.6  # ~60% epilepsy, 40% control
    modality = RNG.choice(MODALITIES)
    rec_start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i, hours=RNG.randint(8, 16))
    dur_min = RNG.choice([20, 30, 45, 60, 120, 240])

    # Event annotations (the AI training labels) — seizures only if epilepsy.
    events = []
    if is_epilepsy:
        for e in range(RNG.randint(1, 3)):
            start_s = RNG.randint(60, dur_min * 60 - 120)
            length = RNG.randint(20, 110)
            events.append({
                "event_id": f"{pid}-EVT{e+1:02d}",
                "event_type": "Seizure",
                "start_sec": start_s,
                "end_sec": start_s + length,
                "duration_sec": length,
                "channels_involved": ",".join(RNG.sample(["T3", "T4", "T5", "F7", "F8", "Fp1", "C3"], 2)),
                "clinical_correlation": RNG.choice(["Right arm jerk", "Staring", "Automatisms", "None observed"]),
                "video_available": "Yes" if modality == "video-EEG" else "No",
            })

    return {
        "_meta": {"synthetic": True, "disease": "epilepsy", "generated_at": _utc(),
                  "generator": "generate_synthetic_epilepsy.py", "seed": 42},
        "patient": {
            "patient_id": pid, "age": age, "gender": gender,
            "diagnosis": "Epilepsy" if is_epilepsy else "Non-epileptic",
            "label": 1 if is_epilepsy else 0,
        },
        "demographics": {
            "patient_id": pid, "age": age, "gender": gender, "age_of_onset": onset,
            "education": RNG.choice(["Primary", "Secondary", "Bachelor", "Master"]),
            "ethnicity": RNG.choice(["A", "B", "C"]), "occupation": RNG.choice(["Teacher", "Retired", "Student", "Engineer"]),
        },
        "eeg_acquisition": {
            "patient_id": pid, "study_id": f"EEG2026-{i:04d}", "modality": modality,
            "eeg_date": rec_start.date().isoformat(), "start_time": rec_start.strftime("%H:%M"),
            "duration_min": dur_min, "sampling_rate_hz": 256, "resolution_bit": 16,
            "device_vendor": RNG.choice(["Nicolet", "Natus", "Compumedics"]),
            "high_pass_hz": 1, "low_pass_hz": 70, "notch": "ON",
        },
        "channel_quality": {
            "patient_id": pid, "n_channels": RNG.choice([19, 23, 32]), "electrode_system": "10-20",
            "montage": RNG.choice(["Bipolar", "Referential"]),
            "avg_impedance_kohm": round(RNG.uniform(2, 8), 1),
            "missing_channels": RNG.choice([0, 0, 0, 1]), "signal_quality": RNG.choice(["Good", "Good", "Fair"]),
            "recording_usable": "Yes",
        },
        "clinical_history": {
            "patient_id": pid, "disease_duration_years": age - onset,
            "family_history_epilepsy": RNG.choice(["No", "No", "Yes"]),
            "previous_eeg": "Yes", "previous_mri": RNG.choice(["Yes", "No"]),
            "head_trauma": RNG.choice(["No", "Yes"]), "stroke_history": RNG.choice(["No", "Yes"]),
        },
        "recording_conditions": {
            "patient_id": pid, "awake": "Yes", "drowsy": RNG.choice(["No", "Yes"]),
            "sleep_recorded": RNG.choice(["No", "Yes"]), "sleep_deprived": RNG.choice(["No", "Yes"]),
            "hyperventilation": "Yes", "photic_stimulation": "Yes", "sedation": "No",
        },
        "medications": {
            "patient_id": pid, "drug_name": RNG.choice(AED), "drug_class": "Antiepileptic",
            "dose_mg": RNG.choice([250, 500, 750, 1000]), "frequency": RNG.choice(["BID", "TID"]),
            "benzodiazepine": RNG.choice([""] + BZD), "current": "Yes",
            "drug_resistance": RNG.choice(["No", "No", "Yes"]), "adherence": RNG.choice(["Good", "Fair"]),
        },
        "mri_findings": {
            "patient_id": pid, "mri_available": RNG.choice(["Yes", "No"]),
            "mri_normal": RNG.choice(["No", "Yes"]) if is_epilepsy else "Yes",
            "hippocampal_sclerosis": RNG.choice(["Yes", "No"]) if is_epilepsy else "No",
            "lesion_present": RNG.choice(["Yes", "No"]) if is_epilepsy else "No",
            "lesion_location": RNG.choice(LOBES) if is_epilepsy else "",
            "hemisphere": RNG.choice(HEMI), "structural_epilepsy": "Yes" if is_epilepsy else "No",
        },
        "neuropsych": {
            "patient_id": pid, "moca": RNG.randint(22, 30), "mmse": RNG.randint(24, 30),
            "phq9": RNG.randint(0, 12), "gad7": RNG.randint(0, 10),
        },
        "eeg_interpretation": {
            "patient_id": pid, "background_rhythm": f"Alpha {RNG.randint(8,12)} Hz",
            "epileptiform_discharge": "Yes" if is_epilepsy else "No",
            "focal_abnormality": (RNG.choice(HEMI) + " " + RNG.choice(LOBES)) if is_epilepsy else "None",
            "seizure_captured": "Yes" if events else "No",
            "epilepsy_type": RNG.choice(EPILEPSY_TYPES) if is_epilepsy else "Not Epilepsy",
            "seizure_type": RNG.choice(SEIZURE_TYPES) if is_epilepsy else "None",
            "impression": "Temporal Lobe Epilepsy" if is_epilepsy else "Normal EEG",
            "final_diagnosis": "Epilepsy" if is_epilepsy else "Non-epileptic",
        },
        "outcomes": {
            "patient_id": pid, "seizure_free": RNG.choice(["No", "Yes"]) if is_epilepsy else "Yes",
            "seizure_recurrence": RNG.choice(["Yes", "No"]) if is_epilepsy else "No",
            "seizure_count_monthly": RNG.randint(0, 8) if is_epilepsy else 0,
            "treatment_response": RNG.choice(["Improved", "Stable", "Worse"]),
            "er_visits": RNG.randint(0, 3), "hospital_admissions": RNG.randint(0, 2),
            "qolie31": RNG.randint(55, 90),
        },
        "hitl_reviews": {
            "patient_id": pid, "ai_prediction": "Epilepsy" if is_epilepsy else "Control",
            "ai_confidence": round(RNG.uniform(0.6, 0.97), 2),
            "reviewer_id": f"N{RNG.randint(1,5):03d}",
            "decision": RNG.choice(["accept", "accept", "override"]),
            "human_decision": "Epilepsy" if is_epilepsy else "Control",
            "reason_code": RNG.choice(["", "FP", "FN", "ART"]),
        },
        "seizure_metadata": {
            "patient_id": pid,
            "seizure_type": RNG.choice(SEIZURE_TYPES) if is_epilepsy else "None",
            "seizure_duration_sec": RNG.randint(20, 110) if is_epilepsy else 0,
            "seizure_frequency": RNG.choice(["Daily", "Weekly", "Monthly", "Yearly", "Rare"]) if is_epilepsy else "None",
            "trigger": RNG.choice(["Sleep deprivation", "Stress", "Missed medication", "None"]) if is_epilepsy else "None",
            "aura": RNG.choice(["Epigastric", "Deja vu", "Visual", "None"]) if is_epilepsy else "None",
            "postictal_symptoms": RNG.choice(["Confusion", "Fatigue", "Headache", "None"]) if is_epilepsy else "None",
            "epilepsy_type": RNG.choice(["Focal", "Generalized", "Combined"]) if is_epilepsy else "None",
            "status_epilepticus": RNG.choice(["No", "No", "Yes"]) if is_epilepsy else "No",
        },
        "event_annotations": events,
        "artifact_annotations": [
            {"patient_id": pid, "artifact_type": a, "severity": RNG.choice(["Low", "Medium", "High"])}
            for a in RNG.sample(ARTIFACTS, RNG.randint(1, 3))
        ],
    }


# Sections written as flat per-row CSVs (one row per patient, except list sections).
FLAT_SECTIONS = ["patient", "demographics", "eeg_acquisition", "channel_quality", "clinical_history",
                 "recording_conditions", "medications", "mri_findings", "neuropsych",
                 "eeg_interpretation", "outcomes", "hitl_reviews", "seizure_metadata"]
LIST_SECTIONS = ["event_annotations", "artifact_annotations"]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "patients").mkdir(exist_ok=True)

    patients = [gen_patient(i + 1) for i in range(args.n)]

    # Per-section CSVs.
    for sec in FLAT_SECTIONS:
        write_csv(OUT / f"{sec}.csv", [p[sec] for p in patients])
    for sec in LIST_SECTIONS:
        rows = [r for p in patients for r in p[sec]]
        write_csv(OUT / f"{sec}.csv", rows)

    # Per-patient JSON bundle.
    for p in patients:
        (OUT / "patients" / f"{p['patient']['patient_id']}.json").write_text(
            json.dumps(p, indent=2), encoding="utf-8")

    n_epi = sum(1 for p in patients if p["patient"]["label"] == 1)
    n_video = sum(1 for p in patients if p["eeg_acquisition"]["modality"] == "video-EEG")
    n_events = sum(len(p["event_annotations"]) for p in patients)

    manifest = {
        "synthetic": True, "disease": "epilepsy", "generated_at": _utc(), "seed": 42,
        "n_patients": len(patients), "n_epilepsy": n_epi, "n_control": len(patients) - n_epi,
        "n_video_eeg": n_video, "n_eeg": len(patients) - n_video, "n_seizure_events": n_events,
        "sections": FLAT_SECTIONS + LIST_SECTIONS,
        "warning": "SYNTHETIC — framework demonstration only. Not real patients.",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# SYNTHETIC Epilepsy Dataset\n\n"
        "⚠️ **SYNTHETIC — generated for framework demonstration only. Not real patients.**\n\n"
        f"- Generated: {manifest['generated_at']} (seed 42, reproducible)\n"
        f"- Patients: {manifest['n_patients']} ({n_epi} epilepsy, {len(patients)-n_epi} control)\n"
        f"- Modality: {len(patients)-n_video} EEG, {n_video} video-EEG\n"
        f"- Seizure events annotated: {n_events}\n\n"
        "Per-section CSVs + per-patient JSON in `patients/`. Import with "
        "`scripts/import_synthetic_epilepsy.py`.\n", encoding="utf-8")

    print(f"[generate_synthetic_epilepsy] {manifest['generated_at']}")
    print(f"  patients={len(patients)} (epilepsy={n_epi}, control={len(patients)-n_epi})")
    print(f"  modality: EEG={len(patients)-n_video}, video-EEG={n_video} | seizure events={n_events}")
    print(f"  output: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
