"""EEG Artifact Analysis Dashboard — artifact_annotations table (169 rows, 30 patients).

Artifact types: muscle, ECG, electrode_pop, movement, eye_blink, sweat (6 types)
Severities: mild (87), moderate (60), severe (22)
Clinically relevant: artifact burden affects EEG interpretation reliability,
seizure detection sensitivity, and signal quality scoring.

Sources:
  artifact_annotations (169 rows) — patient_id, fields_json (artifact_type, channel,
                                      start_time_min, duration_sec, severity)
  eeg_acquisition (30 rows)       — recording duration, signal quality
  patients (41 rows)              — age, gender, diagnosis
"""
import json
import sqlite3
import pathlib

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _connect():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _load():
    """Load and parse artifact_annotations with related context."""
    con = _connect()
    rows = con.execute(
        "SELECT patient_id, fields_json, created_at FROM artifact_annotations ORDER BY created_at"
    ).fetchall()
    con.close()
    artifacts = []
    for patient_id, fj, created_at in rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        artifacts.append({
            "patient_id": patient_id,
            "artifact_type": d.get("artifact_type", "unknown"),
            "channel": d.get("channel", "unknown"),
            "start_time_min": d.get("start_time_min", 0.0),
            "duration_sec": d.get("duration_sec", 0.0),
            "severity": d.get("severity", "unknown"),
            "created_at": created_at,
        })
    return artifacts


def overview():
    artifacts = _load()
    total = len(artifacts)
    patients = list({a["patient_id"] for a in artifacts})
    n_patients = len(patients)

    # Type distribution
    type_counts = {}
    for a in artifacts:
        t = a["artifact_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Severity distribution
    sev_counts = {}
    for a in artifacts:
        s = a["severity"]
        sev_counts[s] = sev_counts.get(s, 0) + 1

    # Avg duration per type
    type_durations = {}
    for a in artifacts:
        t = a["artifact_type"]
        type_durations.setdefault(t, []).append(a["duration_sec"])
    type_avg_dur = {t: round(sum(v) / len(v), 2) for t, v in type_durations.items()}

    # Avg artifacts per patient
    pat_counts = {}
    for a in artifacts:
        pat_counts[a["patient_id"]] = pat_counts.get(a["patient_id"], 0) + 1
    avg_per_patient = round(total / n_patients, 1) if n_patients else 0
    max_burden = max(pat_counts.values()) if pat_counts else 0

    # Severity mix
    mild_pct = round(sev_counts.get("mild", 0) / total * 100, 1) if total else 0
    severe_pct = round(sev_counts.get("severe", 0) / total * 100, 1) if total else 0

    return {
        "kpis": {
            "total_annotations": total,
            "patients_affected": n_patients,
            "artifact_types": len(type_counts),
            "avg_per_patient": avg_per_patient,
            "max_patient_burden": max_burden,
            "mild_pct": mild_pct,
            "severe_pct": severe_pct,
        },
        "type_distribution": [
            {"type": t, "count": c, "avg_duration_sec": type_avg_dur.get(t, 0)}
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
        ],
        "severity_distribution": [
            {"severity": s, "count": c,
             "pct": round(c / total * 100, 1) if total else 0}
            for s, c in sorted(
                sev_counts.items(),
                key=lambda x: {"mild": 0, "moderate": 1, "severe": 2}.get(x[0], 9)
            )
        ],
    }


def breakdown():
    artifacts = _load()

    # Channel distribution (top 10)
    channel_counts = {}
    for a in artifacts:
        ch = a["channel"]
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
    top_channels = sorted(channel_counts.items(), key=lambda x: -x[1])[:12]

    # Cross-tab: artifact type × severity
    cross = {}
    for a in artifacts:
        key = (a["artifact_type"], a["severity"])
        cross[key] = cross.get(key, 0) + 1
    types = sorted({a["artifact_type"] for a in artifacts})
    sevs = ["mild", "moderate", "severe"]
    type_sev_matrix = [
        {"type": t, **{s: cross.get((t, s), 0) for s in sevs}}
        for t in types
    ]

    # Per-patient table (sorted by burden desc)
    pat_data = {}
    for a in artifacts:
        pid = a["patient_id"]
        if pid not in pat_data:
            pat_data[pid] = {"count": 0, "severe": 0, "types": set(), "total_dur": 0}
        pat_data[pid]["count"] += 1
        pat_data[pid]["total_dur"] += a["duration_sec"]
        if a["severity"] == "severe":
            pat_data[pid]["severe"] += 1
        pat_data[pid]["types"].add(a["artifact_type"])

    per_patient = sorted(
        [
            {
                "patient_id": pid,
                "total_artifacts": d["count"],
                "severe_count": d["severe"],
                "unique_types": len(d["types"]),
                "total_duration_sec": round(d["total_dur"], 1),
                "burden": "High" if d["count"] >= 8 else "Moderate" if d["count"] >= 5 else "Low",
            }
            for pid, d in pat_data.items()
        ],
        key=lambda x: -x["total_artifacts"],
    )

    # Timeline: monthly annotation counts
    monthly = {}
    for a in artifacts:
        ym = a["created_at"][:7] if a["created_at"] else "unknown"
        monthly[ym] = monthly.get(ym, 0) + 1
    timeline = [{"month": m, "count": c} for m, c in sorted(monthly.items())]

    return {
        "channel_distribution": [{"channel": ch, "count": c} for ch, c in top_channels],
        "type_severity_matrix": type_sev_matrix,
        "per_patient": per_patient,
        "monthly_trend": timeline,
    }


def definitions():
    return {
        "title": "EEG Artifact Analysis — Clinical Definitions",
        "overview": (
            "EEG artifacts are non-cerebral signals that contaminate the recording. "
            "Detecting and classifying artifacts is essential before seizure detection "
            "models run — unmitigated artifacts inflate false-positive rates by 15–40% "
            "(Nunez & Srinivasan, 2006)."
        ),
        "artifact_types": [
            {
                "type": "muscle",
                "label": "Muscle (EMG)",
                "description": "High-frequency (20–500 Hz) contamination from scalp/jaw muscles. Most common artifact.",
                "channels_affected": "Temporal (T3, T4, T5, T6), Frontal",
                "clinical_impact": "Can mimic high-frequency oscillations; masks interictal spikes.",
                "mitigation": "ICA, frequency filtering (< 35 Hz LPF), relaxation instruction.",
            },
            {
                "type": "ECG",
                "label": "Cardiac (ECG)",
                "description": "Heartbeat-synchronous artifact (~1 Hz) from the pulse wave conducted to scalp.",
                "channels_affected": "Often bilateral temporal leads",
                "clinical_impact": "Can appear as periodic discharge; mistaken for subclinical seizure.",
                "mitigation": "ICA component rejection after ECG reference correlation.",
            },
            {
                "type": "electrode_pop",
                "label": "Electrode Pop",
                "description": "Sudden large-amplitude transient from poor electrode contact or gel drying.",
                "channels_affected": "Single channel (focal)",
                "clinical_impact": "High-amplitude spike mimics epileptiform discharge.",
                "mitigation": "Impedance check (< 5 kΩ), gel reapplication, re-recording flagged.",
            },
            {
                "type": "movement",
                "label": "Movement",
                "description": "Slow high-amplitude drift from patient movement or cable tug.",
                "channels_affected": "Diffuse or localized based on motion vector",
                "clinical_impact": "Baseline wander obscures slow-wave activity; masks post-ictal suppression.",
                "mitigation": "Highpass filter (> 0.5 Hz), epoch rejection, motion logging.",
            },
            {
                "type": "eye_blink",
                "label": "Eye Blink / Eye Movement",
                "description": "Corneoretinal dipole potential from blinks (slow) and saccades (stepped).",
                "channels_affected": "Frontal (Fp1, Fp2, F3, F4)",
                "clinical_impact": "Mimics frontal delta; EOG regression or ICA required.",
                "mitigation": "EOG reference channel + ICA / regression.",
            },
            {
                "type": "sweat",
                "label": "Sweat (Galvanic)",
                "description": "Very slow drift (< 0.5 Hz) from skin resistance change due to perspiration.",
                "channels_affected": "Diffuse, worse at temporal electrodes",
                "clinical_impact": "Baseline wander; accentuates slow-wave power artificially.",
                "mitigation": "Highpass filter (> 0.5 Hz), cool environment, dry gel.",
            },
        ],
        "severity_levels": [
            {"level": "mild", "label": "Mild", "description": "Short duration (<3 s) or isolated channel; automated rejection handles it.", "badge": "success"},
            {"level": "moderate", "label": "Moderate", "description": "Moderate duration (3–8 s) or affects adjacent channels; ICA recommended.", "badge": "warning"},
            {"level": "severe", "label": "Severe", "description": "Long duration (>8 s) or diffuse contamination; epoch must be excluded.", "badge": "danger"},
        ],
        "data_source": "artifact_annotations table — 169 rows, 30 patients, 6 artifact types",
        "references": [
            "Nunez & Srinivasan (2006) Electric Fields of the Brain — Oxford UP",
            "Urigüen & Garcia-Zapirain (2015) EEG artifact removal review — J Neural Eng",
            "Delorme et al. (2007) EEGLAB ICA — J Neurosci Methods",
        ],
    }
