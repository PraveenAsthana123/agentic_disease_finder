"""
Neuro AI Ecosystem — Artifact Annotation Dashboard
====================================================
EEG artifact annotation analytics from artifact_annotations table.

Artifact types: muscle, ECG, electrode_pop, movement, eye_blink, sweat
Severity levels: mild, moderate, severe
Channels: standard 10-20 EEG montage (Fp1, F3, F7, T4, T6, O1, O2, etc.)

Real data: artifact_annotations (~169 rows, 30 patients)
in clinical.db.  fields_json contains: artifact_type, channel,
start_time_min, duration_sec, severity.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def _parse_fields(row_id, fields_json_str):
    """Parse fields_json safely, returning dict with defaults for missing keys."""
    try:
        d = json.loads(fields_json_str) if fields_json_str else {}
    except (json.JSONDecodeError, TypeError):
        d = {}
    return {
        "id": row_id,
        "artifact_type": d.get("artifact_type", "unknown"),
        "channel": d.get("channel", "unknown"),
        "start_time_min": d.get("start_time_min", 0),
        "duration_sec": d.get("duration_sec", 0),
        "severity": d.get("severity", "unknown"),
    }


def overview():
    """Artifact annotation overview — counts, type/severity/channel distributions,
    severity-by-type cross-tab, monthly trend, KPIs."""
    conn = _conn()
    cur = conn.cursor()

    # Fetch all rows
    cur.execute("SELECT id, patient_id, fields_json, created_at FROM artifact_annotations")
    raw_rows = cur.fetchall()

    parsed = []
    for row in raw_rows:
        p = _parse_fields(row[0], row[2])
        p["patient_id"] = row[1]
        p["created_at"] = row[3]
        parsed.append(p)

    total_annotations = len(parsed)
    unique_patients = len({r["patient_id"] for r in parsed})

    # Artifact type distribution
    type_dist = {}
    for r in parsed:
        t = r["artifact_type"]
        type_dist[t] = type_dist.get(t, 0) + 1

    # Severity distribution
    sev_dist = {}
    for r in parsed:
        s = r["severity"]
        sev_dist[s] = sev_dist.get(s, 0) + 1

    # Channel distribution (top 15)
    chan_dist_all = {}
    for r in parsed:
        c = r["channel"]
        chan_dist_all[c] = chan_dist_all.get(c, 0) + 1
    chan_sorted = sorted(chan_dist_all.items(), key=lambda x: x[1], reverse=True)[:15]
    channel_distribution = dict(chan_sorted)

    # Average duration
    durations = [r["duration_sec"] for r in parsed if r["duration_sec"]]
    avg_duration_sec = round(sum(durations) / max(len(durations), 1), 2)

    # Severity by type (stacked breakdown)
    type_sev = {}
    for r in parsed:
        t = r["artifact_type"]
        s = r["severity"]
        if t not in type_sev:
            type_sev[t] = {"type": t, "mild": 0, "moderate": 0, "severe": 0}
        if s in ("mild", "moderate", "severe"):
            type_sev[t][s] += 1
    severity_by_type = list(type_sev.values())

    # Monthly trend
    month_counts = {}
    for r in parsed:
        if r["created_at"]:
            month = r["created_at"][:7]
            month_counts[month] = month_counts.get(month, 0) + 1
    monthly_trend = [{"month": m, "count": c} for m, c in sorted(month_counts.items())]

    # KPIs
    severe_count = sev_dist.get("severe", 0)
    severe_pct = round(severe_count / max(total_annotations, 1) * 100, 1)

    conn.close()
    return {
        "total_annotations": total_annotations,
        "unique_patients": unique_patients,
        "artifact_type_distribution": type_dist,
        "severity_distribution": sev_dist,
        "channel_distribution": channel_distribution,
        "avg_duration_sec": avg_duration_sec,
        "severity_by_type": severity_by_type,
        "monthly_trend": monthly_trend,
        "kpis": {
            "total_annotations": total_annotations,
            "unique_patients": unique_patients,
            "artifact_types": len(type_dist),
            "avg_duration_sec": avg_duration_sec,
            "severe_pct": severe_pct,
        },
    }


def breakdown():
    """Artifact annotation breakdown — per-patient profiles, type-by-channel
    cross-tab, recent annotations, duration stats by type."""
    conn = _conn()
    cur = conn.cursor()

    # Fetch all rows
    cur.execute("SELECT id, patient_id, fields_json, created_at FROM artifact_annotations")
    raw_rows = cur.fetchall()

    parsed = []
    for row in raw_rows:
        p = _parse_fields(row[0], row[2])
        p["patient_id"] = row[1]
        p["created_at"] = row[3]
        parsed.append(p)

    # --- per_patient ---
    patient_data = {}
    for r in parsed:
        pid = r["patient_id"]
        if pid not in patient_data:
            patient_data[pid] = {"patient_id": pid, "rows": []}
        patient_data[pid]["rows"].append(r)

    per_patient = []
    for pid, info in patient_data.items():
        rows = info["rows"]
        types = {}
        severities = {}
        dur_sum = 0
        dur_count = 0
        for r in rows:
            t = r["artifact_type"]
            types[t] = types.get(t, 0) + 1
            s = r["severity"]
            severities[s] = severities.get(s, 0) + 1
            if r["duration_sec"]:
                dur_sum += r["duration_sec"]
                dur_count += 1
        per_patient.append({
            "patient_id": pid,
            "total": len(rows),
            "types": types,
            "severities": severities,
            "avg_duration": round(dur_sum / max(dur_count, 1), 2),
        })
    per_patient.sort(key=lambda x: x["total"], reverse=True)

    # --- type_by_channel cross-tab ---
    all_types = ["muscle", "ECG", "electrode_pop", "movement", "eye_blink", "sweat"]
    chan_type = {}
    for r in parsed:
        c = r["channel"]
        if c not in chan_type:
            chan_type[c] = {"channel": c}
            for at in all_types:
                chan_type[c][at] = 0
        at = r["artifact_type"]
        if at in all_types:
            chan_type[c][at] += 1
    type_by_channel = sorted(chan_type.values(), key=lambda x: x["channel"])

    # --- recent_annotations (last 20) ---
    sorted_by_date = sorted(parsed, key=lambda x: x.get("created_at") or "", reverse=True)
    recent_annotations = []
    for r in sorted_by_date[:20]:
        recent_annotations.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "artifact_type": r["artifact_type"],
            "channel": r["channel"],
            "start_time_min": r["start_time_min"],
            "duration_sec": r["duration_sec"],
            "severity": r["severity"],
            "created_at": r["created_at"],
        })

    # --- duration_by_type ---
    type_durations = {}
    for r in parsed:
        t = r["artifact_type"]
        d = r["duration_sec"]
        if d:
            if t not in type_durations:
                type_durations[t] = []
            type_durations[t].append(d)
    duration_by_type = []
    for t, durs in sorted(type_durations.items()):
        duration_by_type.append({
            "type": t,
            "avg_duration": round(sum(durs) / len(durs), 2),
            "min_duration": round(min(durs), 2),
            "max_duration": round(max(durs), 2),
        })

    conn.close()
    return {
        "per_patient": per_patient,
        "type_by_channel": type_by_channel,
        "recent_annotations": recent_annotations,
        "duration_by_type": duration_by_type,
    }


def definitions():
    """Artifact annotation definitions — artifact types, severity levels,
    EEG artifact glossary, references, clinical notes."""
    return {
        "artifact_types": [
            {"name": "muscle", "description": "High-frequency EMG contamination from scalp, facial, or cervical muscle activity; appears as broadband high-frequency noise, most prominent in temporal and frontal electrodes"},
            {"name": "ECG", "description": "Cardiac electrical activity (QRS complex) picked up by EEG electrodes, especially in referential montages; appears as periodic sharp deflections at heart rate frequency"},
            {"name": "electrode_pop", "description": "Sudden transient caused by momentary loss of electrode-skin contact; appears as a sharp, high-amplitude spike confined to a single channel"},
            {"name": "movement", "description": "Low-frequency, high-amplitude deflections caused by patient head or body movement; affects multiple channels simultaneously and can obscure underlying EEG activity"},
            {"name": "eye_blink", "description": "Frontally dominant slow deflections generated by vertical eye movements (Bell phenomenon); largest at Fp1/Fp2 with phase reversal, attenuating posteriorly"},
            {"name": "sweat", "description": "Very low-frequency baseline drift (< 1 Hz) caused by galvanic skin response from perspiration; creates slow undulating waveforms, worsened by warm environments"},
        ],
        "severity_levels": [
            {"level": "mild", "description": "Artifact is present but does not significantly obscure underlying EEG activity; signal remains interpretable for clinical review without additional processing"},
            {"level": "moderate", "description": "Artifact partially obscures EEG features; ICA or adaptive filtering may be needed to recover usable signal in affected epochs"},
            {"level": "severe", "description": "Artifact dominates the channel rendering the epoch uninterpretable; affected segments are typically excluded from clinical analysis and quantitative EEG measures"},
        ],
        "glossary": [
            {"term": "Artifact", "definition": "Any electrical signal recorded on EEG that does not originate from cerebral neuronal activity; may be physiological (patient-generated) or non-physiological (equipment-related)"},
            {"term": "ICA", "definition": "Independent Component Analysis — blind source separation technique used to decompose EEG into independent components for artifact removal while preserving neural signals"},
            {"term": "Epoch", "definition": "A fixed-duration segment of continuous EEG (typically 2-10 seconds) used as the unit of analysis for artifact marking, spectral analysis, and event detection"},
            {"term": "10-20 System", "definition": "International standard electrode placement system using proportional distances from skull landmarks; ensures reproducible electrode positions across sessions and patients"},
            {"term": "Montage", "definition": "Specific arrangement of EEG channel derivations (bipolar, referential, average reference) used for display and analysis; choice affects artifact appearance"},
            {"term": "Baseline Drift", "definition": "Slow deviation of the EEG signal from zero, often caused by electrode impedance changes, sweat artifacts, or patient movement"},
            {"term": "Impedance", "definition": "Resistance to current flow at the electrode-skin interface; high impedance increases susceptibility to noise and environmental interference artifacts"},
            {"term": "EMG Contamination", "definition": "Electromyographic activity from muscle contractions that overlaps with the EEG frequency range (especially beta/gamma bands), complicating spectral analysis"},
            {"term": "Adaptive Filtering", "definition": "Signal processing technique that uses a reference signal (e.g., EOG, ECG) to subtract correlated artifact from EEG channels in real time"},
            {"term": "Signal-to-Noise Ratio", "definition": "Ratio of desired cerebral signal power to artifact/noise power; higher SNR indicates cleaner recordings with better diagnostic utility"},
        ],
        "references": [
            "Tatum WO et al. Artifact and Recording Concepts in EEG. J Clin Neurophysiol. 2011;28(3):252-263",
            "Jiang X et al. Removal of Artifacts from EEG Signals: A Review. Sensors. 2019;19(5):987",
            "Urigüen JA, Garcia-Zapirain B. EEG artifact removal — state-of-the-art and guidelines. J Neural Eng. 2015;12(3):031001",
            "Delorme A, Makeig S. EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics. J Neurosci Methods. 2004;134(1):9-21",
            "IFCN Standards: Nuwer MR et al. IFCN standards for digital recording of clinical EEG. Electroencephalogr Clin Neurophysiol. 1998;106(3):259-261",
            "ACNS Guideline 1: Minimum Technical Requirements for Performing Clinical EEG. J Clin Neurophysiol. 2016;33(4):303-307",
        ],
        "clinical_notes": [
            "All artifact annotations should include onset time, duration, affected channel(s), and severity to enable automated artifact rejection pipeline calibration",
            "Muscle artifact is the most common EEG contaminant in ambulatory recordings; instruct patients to relax jaw and neck muscles during recording",
            "ECG artifact requires dedicated ECG reference channel for adaptive subtraction; amplitude increases with low electrode impedance at temporal sites",
            "Electrode pop artifacts indicate electrode maintenance issues; impedance should be checked and gel reapplied if pops recur within a session",
            "Eye blink artifacts can be removed using ICA or regression-based EOG subtraction without significant loss of frontal EEG information",
            "Sweat artifact prevalence increases in warm recording environments; maintain room temperature at 20-22°C and ensure adequate ventilation",
        ],
    }
