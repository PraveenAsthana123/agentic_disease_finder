"""
Neuro AI Ecosystem -- Artifact Detection Dashboard
====================================================
Automated EEG artifact detection and channel quality analysis to
reduce manual artifact marking burden for EEG technicians.

Artifact Types Detected:
  eye_blink      -- Frontal (Fp1/Fp2) high-amplitude slow deflection
  muscle         -- High-frequency EMG contamination (temporal channels)
  movement       -- Broadband large-amplitude transient
  electrode_pop  -- Abrupt spike from electrode contact issue
  ECG            -- Cardiac QRS complex projected onto EEG
  sweat          -- Low-frequency slow drift from galvanic skin response

Data Sources:
  - artifact_annotations  (169 rows, 30 patients) -- auto-detected artifacts
  - channel_quality       (30 rows)  -- per-channel impedance + SNR
  - eeg_acquisition       (30 rows)  -- recording metadata
  - patients              -- demographics

All patient IDs sourced from real patients table in clinical.db.

Author: Research Team
"""

import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

ARTIFACT_LABELS = {
    "eye_blink": "Eye Blink",
    "muscle": "Muscle (EMG)",
    "movement": "Movement",
    "electrode_pop": "Electrode Pop",
    "ECG": "ECG (Cardiac)",
    "sweat": "Sweat (GSR)",
}

SEVERITY_ORDER = ["mild", "moderate", "severe"]
SEVERITY_COLORS = {"mild": "#22c55e", "moderate": "#f59e0b", "severe": "#ef4444"}

QUALITY_GRADES = {"Good": 3, "Fair": 2, "Poor": 1}


def _conn():
    return sqlite3.connect(DB_PATH)


def _load_artifacts():
    """Load all artifact annotations with parsed JSON fields."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json, created_at FROM artifact_annotations"
    ).fetchall()
    conn.close()
    artifacts = []
    for pid, fj, ts in rows:
        d = json.loads(fj)
        d["patient_id"] = pid
        d["created_at"] = ts
        artifacts.append(d)
    return artifacts


def _load_channel_quality():
    """Load per-patient channel quality data."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM channel_quality"
    ).fetchall()
    conn.close()
    result = {}
    for pid, fj in rows:
        result[pid] = json.loads(fj).get("channels", [])
    return result


def _load_patients():
    """Load patient demographics."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients"
    ).fetchall()
    conn.close()
    return {r[0]: {"name": r[1], "age": r[2], "gender": r[3], "disease": r[4]} for r in rows}


def _load_acquisitions():
    """Load EEG acquisition metadata."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM eeg_acquisition"
    ).fetchall()
    conn.close()
    result = {}
    for pid, fj in rows:
        result[pid] = json.loads(fj)
    return result


# ── Public API ────────────────────────────────────────────────────


def overview():
    """Artifact detection overview — KPIs, type distribution, severity breakdown,
    channel hotspots, and artifact rate per recording."""
    artifacts = _load_artifacts()
    cq = _load_channel_quality()
    acq = _load_acquisitions()

    total = len(artifacts)
    patient_ids = list({a["patient_id"] for a in artifacts})
    n_patients = len(patient_ids)

    # -- KPIs --
    severe_count = sum(1 for a in artifacts if a.get("severity") == "severe")
    avg_per_patient = round(total / n_patients, 1) if n_patients else 0

    # Average channel quality across all patients
    all_snr = []
    poor_channels = 0
    total_channels = 0
    for pid, channels in cq.items():
        for ch in channels:
            all_snr.append(ch.get("snr_db", 0))
            total_channels += 1
            if ch.get("quality_grade") == "Poor":
                poor_channels += 1
    avg_snr = round(sum(all_snr) / len(all_snr), 1) if all_snr else 0
    poor_pct = round(100 * poor_channels / total_channels, 1) if total_channels else 0

    # -- Artifact type distribution --
    type_counts = Counter(a.get("artifact_type", "unknown") for a in artifacts)
    type_dist = [
        {"type": t, "label": ARTIFACT_LABELS.get(t, t), "count": c}
        for t, c in type_counts.most_common()
    ]

    # -- Severity breakdown --
    sev_counts = Counter(a.get("severity", "unknown") for a in artifacts)
    severity_dist = [
        {"severity": s, "count": sev_counts.get(s, 0), "color": SEVERITY_COLORS.get(s, "#94a3b8")}
        for s in SEVERITY_ORDER
    ]

    # -- Channel hotspot (which channels have most artifacts) --
    ch_counts = Counter(a.get("channel", "?") for a in artifacts)
    channel_hotspots = [
        {"channel": ch, "count": c}
        for ch, c in ch_counts.most_common(10)
    ]

    # -- Artifact type × severity matrix --
    matrix = defaultdict(lambda: {"mild": 0, "moderate": 0, "severe": 0})
    for a in artifacts:
        matrix[a.get("artifact_type", "unknown")][a.get("severity", "mild")] += 1
    type_severity = [
        {"type": t, "label": ARTIFACT_LABELS.get(t, t), **counts}
        for t, counts in sorted(matrix.items())
    ]

    # -- Artifact rate per recording (artifacts per minute) --
    rate_data = []
    patient_artifact_counts = Counter(a["patient_id"] for a in artifacts)
    for pid in sorted(patient_ids):
        dur = acq.get(pid, {}).get("duration_min", 30)
        cnt = patient_artifact_counts.get(pid, 0)
        rate = round(cnt / dur, 3) if dur else 0
        rate_data.append({"patient_id": pid, "artifacts": cnt, "duration_min": dur, "rate_per_min": rate})
    rate_data.sort(key=lambda x: x["rate_per_min"], reverse=True)

    return {
        "kpis": {
            "total_artifacts": total,
            "patients_scanned": n_patients,
            "severe_artifacts": severe_count,
            "avg_per_patient": avg_per_patient,
            "avg_snr_db": avg_snr,
            "poor_channel_pct": poor_pct,
        },
        "type_distribution": type_dist,
        "severity_distribution": severity_dist,
        "channel_hotspots": channel_hotspots,
        "type_severity_matrix": type_severity,
        "artifact_rate": rate_data[:15],
    }


def breakdown():
    """Per-patient artifact breakdown — artifact list, channel quality,
    flagged patients, and artifact timeline."""
    artifacts = _load_artifacts()
    cq = _load_channel_quality()
    patients = _load_patients()
    acq = _load_acquisitions()

    # Group artifacts by patient
    by_patient = defaultdict(list)
    for a in artifacts:
        by_patient[a["patient_id"]].append(a)

    patient_profiles = []
    flagged = []
    for pid in sorted(by_patient.keys()):
        arts = by_patient[pid]
        demo = patients.get(pid, {})
        channels = cq.get(pid, [])
        recording = acq.get(pid, {})

        severe_count = sum(1 for a in arts if a.get("severity") == "severe")
        type_counts = Counter(a.get("artifact_type", "?") for a in arts)
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else "none"

        # Channel quality summary
        good_ch = sum(1 for c in channels if c.get("quality_grade") == "Good")
        fair_ch = sum(1 for c in channels if c.get("quality_grade") == "Fair")
        poor_ch = sum(1 for c in channels if c.get("quality_grade") == "Poor")
        avg_impedance = round(
            sum(c.get("impedance_kohm", 0) for c in channels) / len(channels), 1
        ) if channels else 0

        profile = {
            "patient_id": pid,
            "name": demo.get("name", pid),
            "age": demo.get("age"),
            "disease": demo.get("disease"),
            "total_artifacts": len(arts),
            "severe_count": severe_count,
            "dominant_artifact": ARTIFACT_LABELS.get(dominant_type, dominant_type),
            "type_breakdown": {
                ARTIFACT_LABELS.get(t, t): c for t, c in type_counts.most_common()
            },
            "channel_quality": {
                "good": good_ch, "fair": fair_ch, "poor": poor_ch,
                "avg_impedance_kohm": avg_impedance,
            },
            "recording": {
                "type": recording.get("recording_type", "unknown"),
                "duration_min": recording.get("duration_min", 0),
                "sampling_rate": recording.get("sampling_rate", 0),
                "montage": recording.get("montage", "unknown"),
            },
            "artifact_list": [
                {
                    "type": ARTIFACT_LABELS.get(a.get("artifact_type", "?"), a.get("artifact_type", "?")),
                    "channel": a.get("channel"),
                    "start_min": a.get("start_time_min"),
                    "duration_sec": a.get("duration_sec"),
                    "severity": a.get("severity"),
                }
                for a in sorted(arts, key=lambda x: x.get("start_time_min", 0))
            ],
        }
        patient_profiles.append(profile)

        # Flag patients with high artifact burden or poor channels
        if severe_count >= 2 or len(arts) >= 8 or poor_ch >= 5:
            flagged.append({
                "patient_id": pid,
                "name": demo.get("name", pid),
                "reason": (
                    f"{severe_count} severe" if severe_count >= 2
                    else f"{len(arts)} total artifacts" if len(arts) >= 8
                    else f"{poor_ch} poor channels"
                ),
                "total_artifacts": len(arts),
                "severe_count": severe_count,
                "poor_channels": poor_ch,
            })

    # Channel quality heatmap data (aggregate across all patients)
    channel_agg = defaultdict(lambda: {"snr_sum": 0, "imp_sum": 0, "n": 0, "poor": 0})
    for pid, channels in cq.items():
        for ch in channels:
            name = ch["channel"]
            channel_agg[name]["snr_sum"] += ch.get("snr_db", 0)
            channel_agg[name]["imp_sum"] += ch.get("impedance_kohm", 0)
            channel_agg[name]["n"] += 1
            if ch.get("quality_grade") == "Poor":
                channel_agg[name]["poor"] += 1

    channel_summary = [
        {
            "channel": ch,
            "avg_snr_db": round(v["snr_sum"] / v["n"], 1),
            "avg_impedance_kohm": round(v["imp_sum"] / v["n"], 1),
            "poor_rate_pct": round(100 * v["poor"] / v["n"], 1),
            "n_recordings": v["n"],
        }
        for ch, v in sorted(channel_agg.items())
    ]

    return {
        "patients": patient_profiles,
        "flagged_patients": flagged,
        "channel_summary": channel_summary,
    }


def definitions():
    """Metric definitions, artifact type descriptions, quality grading scales, glossary."""
    return {
        "artifact_types": [
            {
                "type": "eye_blink",
                "label": "Eye Blink",
                "description": "High-amplitude slow deflection primarily in frontal channels (Fp1, Fp2) caused by eyelid movement. Typically 200-400 ms duration.",
                "typical_channels": ["Fp1", "Fp2", "F3", "F4"],
                "frequency_range": "0.5-3 Hz",
            },
            {
                "type": "muscle",
                "label": "Muscle (EMG)",
                "description": "High-frequency electromyographic contamination from scalp or facial muscles. Most prominent in temporal and frontal channels.",
                "typical_channels": ["T3", "T4", "T5", "T6", "F7", "F8"],
                "frequency_range": ">20 Hz",
            },
            {
                "type": "movement",
                "label": "Movement",
                "description": "Broadband large-amplitude transient caused by patient head or body movement. Affects multiple channels simultaneously.",
                "typical_channels": ["all"],
                "frequency_range": "broadband",
            },
            {
                "type": "electrode_pop",
                "label": "Electrode Pop",
                "description": "Abrupt spike artifact from sudden change in electrode-skin contact impedance. Single-channel, very brief (<50 ms).",
                "typical_channels": ["any single channel"],
                "frequency_range": "broadband spike",
            },
            {
                "type": "ECG",
                "label": "ECG (Cardiac)",
                "description": "QRS complex from cardiac electrical activity projected onto EEG. Rhythmic at heart rate (~1 Hz). Most visible in ear and temporal references.",
                "typical_channels": ["T3", "T4", "A1", "A2"],
                "frequency_range": "~1 Hz (rhythmic)",
            },
            {
                "type": "sweat",
                "label": "Sweat (GSR)",
                "description": "Very low frequency slow drift from galvanic skin response / perspiration. Affects electrodes with poor contact.",
                "typical_channels": ["any (poor contact)"],
                "frequency_range": "<0.5 Hz",
            },
        ],
        "severity_scale": [
            {"level": "mild", "description": "Artifact present but does not obscure underlying EEG. No epoch rejection needed.", "color": "#22c55e"},
            {"level": "moderate", "description": "Artifact partially obscures EEG. Epoch may need rejection or ICA cleaning.", "color": "#f59e0b"},
            {"level": "severe", "description": "Artifact completely obscures EEG. Epoch must be rejected or channel excluded.", "color": "#ef4444"},
        ],
        "quality_grades": [
            {"grade": "Good", "impedance": "<5 kOhm", "snr": ">20 dB", "action": "No action needed"},
            {"grade": "Fair", "impedance": "5-10 kOhm", "snr": "10-20 dB", "action": "Consider re-gelling electrode"},
            {"grade": "Poor", "impedance": ">10 kOhm", "snr": "<10 dB", "action": "Reapply electrode immediately"},
        ],
        "glossary": [
            {"term": "SNR", "definition": "Signal-to-Noise Ratio in decibels. Higher is better. Clinical EEG typically requires >15 dB."},
            {"term": "Impedance", "definition": "Electrode-skin contact resistance in kilo-Ohms. Lower is better. Target <5 kOhm."},
            {"term": "ICA", "definition": "Independent Component Analysis — blind source separation method to remove artifact components without losing neural signal."},
            {"term": "Epoch", "definition": "A fixed-length segment of EEG data (typically 2-5 seconds) used as the unit of analysis."},
            {"term": "Montage", "definition": "Electrode arrangement and referencing scheme (e.g., average reference, bipolar, referential)."},
            {"term": "10-20 System", "definition": "International electrode placement system using 19 scalp electrodes at standardized positions."},
        ],
    }
