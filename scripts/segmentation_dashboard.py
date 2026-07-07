"""EEG Waveform Segmentation Dashboard — digitization pipeline analytics from clinical.db.

Tracks EEG waveform segmentation readiness: recording acquisition metadata,
channel quality for segmentation, artifact impact on trace extraction,
segmentation method specifications (classical vs U-Net), and per-patient
signal quality assessment for waveform digitization.

Sources:
- eeg_acquisition (30 recordings — type, duration, sampling rate, montage)
- channel_quality (30 records — per-channel impedance, SNR, quality grades)
- artifact_annotations (169 annotations — type, channel, severity, duration)
- config/agent_tasks.json (segmentation task definition + status)
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')
TASKS = os.path.join(os.path.dirname(__file__), '..', 'config', 'agent_tasks.json')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, default=0):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
#  /api/segmentation/overview
# ──────────────────────────────────────────────────────────────

def segmentation_overview():
    """Aggregate segmentation pipeline readiness: recording stats,
    channel quality distribution, artifact impact, method specs."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Recording inventory ──────────────────────────────
    total_recordings = _safe(cur, "SELECT count(*) FROM eeg_acquisition")
    distinct_patients = _safe(
        cur, "SELECT count(DISTINCT patient_id) FROM eeg_acquisition")

    # Parse fields_json for recording details
    rec_rows = _safe_rows(cur, "SELECT patient_id, fields_json FROM eeg_acquisition")
    durations = []
    sampling_rates = []
    montages = {}
    recording_types = {}
    for pid, fj in rec_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        dur = d.get("duration_min", 0)
        if dur:
            durations.append(dur)
        sr = d.get("sampling_rate", 0)
        if sr:
            sampling_rates.append(sr)
        mt = d.get("montage", "unknown")
        montages[mt] = montages.get(mt, 0) + 1
        rt = d.get("recording_type", "unknown")
        recording_types[rt] = recording_types.get(rt, 0) + 1

    avg_duration = round(sum(durations) / max(len(durations), 1), 1)
    avg_sr = round(sum(sampling_rates) / max(len(sampling_rates), 1), 0)

    # ── 2. Channel quality overview ─────────────────────────
    total_cq = _safe(cur, "SELECT count(*) FROM channel_quality")
    cq_rows = _safe_rows(cur, "SELECT patient_id, fields_json FROM channel_quality")

    quality_counts = {"Good": 0, "Fair": 0, "Poor": 0}
    impedance_grades = {"Good": 0, "Fair": 0, "Poor": 0}
    snr_values = []
    total_channels = 0
    for pid, fj in cq_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        for ch in d.get("channels", []):
            total_channels += 1
            qg = ch.get("quality_grade", "Unknown")
            if qg in quality_counts:
                quality_counts[qg] += 1
            ig = ch.get("impedance_grade", "Unknown")
            if ig in impedance_grades:
                impedance_grades[ig] += 1
            snr = ch.get("snr_db")
            if snr is not None:
                snr_values.append(snr)

    avg_snr = round(sum(snr_values) / max(len(snr_values), 1), 1)
    good_rate = round(quality_counts["Good"] / max(total_channels, 1) * 100, 1)

    # ── 3. Artifact impact on segmentation ──────────────────
    total_artifacts = _safe(cur, "SELECT count(*) FROM artifact_annotations")
    distinct_artifact_patients = _safe(
        cur, "SELECT count(DISTINCT patient_id) FROM artifact_annotations")

    art_rows = _safe_rows(
        cur, "SELECT patient_id, fields_json FROM artifact_annotations")
    artifact_types = {}
    severity_counts = {"mild": 0, "moderate": 0, "severe": 0}
    artifact_channels = {}
    total_artifact_dur = 0.0
    for pid, fj in art_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        at = d.get("artifact_type", "unknown")
        artifact_types[at] = artifact_types.get(at, 0) + 1
        sev = d.get("severity", "unknown")
        if sev in severity_counts:
            severity_counts[sev] += 1
        ch = d.get("channel", "unknown")
        artifact_channels[ch] = artifact_channels.get(ch, 0) + 1
        dur = d.get("duration_sec", 0)
        total_artifact_dur += dur

    avg_artifact_dur = round(
        total_artifact_dur / max(total_artifacts, 1), 2)

    # ── 4. Segmentation method specifications ───────────────
    methods = [
        {
            "name": "Classical Trace Extraction",
            "approach": "Thresholding + contour detection on binarized EEG report images",
            "input": "Scanned EEG report image (PNG/JPEG/TIFF)",
            "output": "Digitized waveform (time-series array)",
            "pros": "Fast, no training data needed, deterministic",
            "cons": "Sensitive to scan quality, rotation, noise",
            "status": "specification_ready",
        },
        {
            "name": "U-Net Semantic Segmentation",
            "approach": "Encoder-decoder CNN with skip connections for pixel-wise trace classification",
            "input": "Scanned EEG report image (any resolution)",
            "output": "Segmentation mask (trace vs background) + digitized waveform",
            "pros": "Robust to noise, handles complex layouts, state-of-art accuracy",
            "cons": "Requires labeled training data, GPU for inference",
            "status": "planned",
        },
        {
            "name": "FCN (Fully Convolutional Network)",
            "approach": "End-to-end pixel classification without fully connected layers",
            "input": "Scanned EEG report image",
            "output": "Dense prediction map of trace regions",
            "pros": "Arbitrary input size, efficient inference",
            "cons": "Lower accuracy than U-Net on fine structures",
            "status": "planned",
        },
        {
            "name": "Hough Line Transform",
            "approach": "Classical CV line detection for gridline removal + trace isolation",
            "input": "Preprocessed binary EEG image",
            "output": "Gridline-free trace image for downstream digitization",
            "pros": "Well-understood, fast preprocessing step",
            "cons": "Only handles gridlines, not full segmentation",
            "status": "specification_ready",
        },
    ]

    # ── 5. Segmentation readiness per patient ───────────────
    # Patients with both acquisition + channel quality = ready for segmentation
    acq_patients = set()
    for pid, _ in rec_rows:
        acq_patients.add(pid)
    cq_patients = set()
    for pid, _ in cq_rows:
        cq_patients.add(pid)
    ready_patients = acq_patients & cq_patients
    readiness_rate = round(
        len(ready_patients) / max(len(acq_patients), 1) * 100, 1)

    conn.close()

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_recordings": total_recordings,
            "distinct_patients": distinct_patients,
            "avg_duration_min": avg_duration,
            "avg_sampling_rate": int(avg_sr),
            "total_channels_assessed": total_channels,
            "good_quality_rate_pct": good_rate,
            "avg_snr_db": avg_snr,
            "total_artifacts": total_artifacts,
            "avg_artifact_duration_sec": avg_artifact_dur,
            "segmentation_ready_patients": len(ready_patients),
            "readiness_rate_pct": readiness_rate,
        },
        "recording_types": [{"type": k, "count": v}
                            for k, v in sorted(recording_types.items(),
                                               key=lambda x: -x[1])],
        "montages": [{"montage": k, "count": v}
                     for k, v in sorted(montages.items(),
                                        key=lambda x: -x[1])],
        "quality_distribution": quality_counts,
        "impedance_distribution": impedance_grades,
        "artifact_types": [{"type": k, "count": v}
                           for k, v in sorted(artifact_types.items(),
                                              key=lambda x: -x[1])],
        "artifact_severity": severity_counts,
        "methods": methods,
    }


# ──────────────────────────────────────────────────────────────
#  /api/segmentation/breakdown
# ──────────────────────────────────────────────────────────────

def segmentation_breakdown():
    """Per-patient segmentation breakdown: acquisition specs,
    channel quality per channel, artifact annotations, readiness."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Per-patient acquisition details ──────────────────
    acq_rows = _safe_rows(
        cur, "SELECT patient_id, fields_json, created_at FROM eeg_acquisition ORDER BY patient_id")
    patient_recordings = []
    for pid, fj, ts in acq_rows:
        try:
            d = json.loads(fj)
        except Exception:
            d = {}
        patient_recordings.append({
            "patient_id": pid,
            "recording_type": d.get("recording_type", "unknown"),
            "duration_min": d.get("duration_min", 0),
            "sampling_rate": d.get("sampling_rate", 0),
            "montage": d.get("montage", "unknown"),
            "electrode_system": d.get("electrode_system", "unknown"),
            "study_date": d.get("study_date", ""),
            "technician_notes": d.get("technician_notes", ""),
            "created_at": ts,
        })

    # ── 2. Per-channel quality (aggregated across patients) ─
    cq_rows = _safe_rows(
        cur, "SELECT patient_id, fields_json FROM channel_quality")
    channel_stats = {}
    for pid, fj in cq_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        for ch in d.get("channels", []):
            name = ch.get("channel", "?")
            if name not in channel_stats:
                channel_stats[name] = {
                    "channel": name, "impedances": [], "snrs": [],
                    "good": 0, "fair": 0, "poor": 0, "count": 0}
            channel_stats[name]["count"] += 1
            imp = ch.get("impedance_kohm")
            if imp is not None:
                channel_stats[name]["impedances"].append(imp)
            snr = ch.get("snr_db")
            if snr is not None:
                channel_stats[name]["snrs"].append(snr)
            qg = ch.get("quality_grade", "")
            if qg == "Good":
                channel_stats[name]["good"] += 1
            elif qg == "Fair":
                channel_stats[name]["fair"] += 1
            elif qg == "Poor":
                channel_stats[name]["poor"] += 1

    channel_summary = []
    for ch_name in sorted(channel_stats.keys()):
        s = channel_stats[ch_name]
        imps = s["impedances"]
        snrs = s["snrs"]
        channel_summary.append({
            "channel": ch_name,
            "recordings": s["count"],
            "avg_impedance_kohm": round(sum(imps) / max(len(imps), 1), 1),
            "avg_snr_db": round(sum(snrs) / max(len(snrs), 1), 1),
            "good_pct": round(s["good"] / max(s["count"], 1) * 100, 1),
            "fair_pct": round(s["fair"] / max(s["count"], 1) * 100, 1),
            "poor_pct": round(s["poor"] / max(s["count"], 1) * 100, 1),
        })

    # ── 3. Artifact details per channel ─────────────────────
    art_rows = _safe_rows(
        cur, "SELECT patient_id, fields_json FROM artifact_annotations")
    artifact_by_channel = {}
    artifact_detail = []
    for pid, fj in art_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        ch = d.get("channel", "unknown")
        if ch not in artifact_by_channel:
            artifact_by_channel[ch] = {"total": 0, "types": {}, "severities": {}}
        artifact_by_channel[ch]["total"] += 1
        at = d.get("artifact_type", "unknown")
        artifact_by_channel[ch]["types"][at] = \
            artifact_by_channel[ch]["types"].get(at, 0) + 1
        sev = d.get("severity", "unknown")
        artifact_by_channel[ch]["severities"][sev] = \
            artifact_by_channel[ch]["severities"].get(sev, 0) + 1

    artifact_channel_list = []
    for ch in sorted(artifact_by_channel.keys()):
        info = artifact_by_channel[ch]
        artifact_channel_list.append({
            "channel": ch,
            "total_artifacts": info["total"],
            "types": info["types"],
            "severities": info["severities"],
        })

    # ── 4. Per-patient segmentation readiness ───────────────
    acq_pids = set(r[0] for r in _safe_rows(
        cur, "SELECT DISTINCT patient_id FROM eeg_acquisition"))
    cq_pids = set(r[0] for r in _safe_rows(
        cur, "SELECT DISTINCT patient_id FROM channel_quality"))
    art_pids = set(r[0] for r in _safe_rows(
        cur, "SELECT DISTINCT patient_id FROM artifact_annotations"))

    readiness = []
    for pid in sorted(acq_pids):
        has_cq = pid in cq_pids
        has_art = pid in art_pids
        status = "ready" if has_cq else "needs_quality_check"
        readiness.append({
            "patient_id": pid,
            "has_acquisition": True,
            "has_channel_quality": has_cq,
            "has_artifact_annotations": has_art,
            "segmentation_status": status,
        })

    # ── 5. Sampling rate distribution ───────────────────────
    sr_dist = {}
    for rec in patient_recordings:
        sr = rec["sampling_rate"]
        if sr:
            sr_dist[sr] = sr_dist.get(sr, 0) + 1
    sampling_rate_dist = [{"rate": k, "count": v}
                          for k, v in sorted(sr_dist.items())]

    conn.close()

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_recordings": patient_recordings,
        "channel_summary": channel_summary,
        "artifact_by_channel": artifact_channel_list,
        "patient_readiness": readiness,
        "sampling_rate_distribution": sampling_rate_dist,
    }


# ──────────────────────────────────────────────────────────────
#  /api/segmentation/definitions
# ──────────────────────────────────────────────────────────────

def segmentation_definitions():
    """Metric definitions for the Segmentation Dashboard."""
    return {
        "available": True,
        "definitions": [
            {"metric": "Total Recordings",
             "definition": "Number of EEG acquisition records in clinical.db — each represents one recording session with metadata (type, duration, sampling rate, montage)"},
            {"metric": "Segmentation-Ready Patients",
             "definition": "Patients with both EEG acquisition data AND channel quality assessment — minimum requirement for waveform digitization"},
            {"metric": "Readiness Rate %",
             "definition": "Percentage of recorded patients that have complete channel quality data, indicating segmentation pipeline readiness"},
            {"metric": "Good Quality Rate %",
             "definition": "Percentage of all assessed channels with quality_grade = 'Good' — channels suitable for direct segmentation without denoising"},
            {"metric": "Avg SNR (dB)",
             "definition": "Average signal-to-noise ratio across all channels — higher SNR means cleaner traces for segmentation (>20 dB = Good, 10-20 = Fair, <10 = Poor)"},
            {"metric": "Avg Impedance (kohm)",
             "definition": "Average electrode impedance — lower is better for signal quality (<5 = Good, 5-10 = Fair, >10 = Poor)"},
            {"metric": "Artifact Impact",
             "definition": "Total artifact annotations that would affect segmentation accuracy — artifacts must be removed or masked before waveform digitization"},
            {"metric": "Classical Trace Extraction",
             "definition": "Image processing pipeline: binarize scanned EEG image → detect contours → extract pixel coordinates → convert to time-series waveform"},
            {"metric": "U-Net Segmentation",
             "definition": "Deep learning approach: encoder-decoder CNN with skip connections for pixel-wise classification of EEG trace vs background in scanned images"},
            {"metric": "Sampling Rate",
             "definition": "Recording sampling frequency in Hz — determines temporal resolution of digitized waveforms (256/512/1024 Hz common in clinical EEG)"},
            {"metric": "Montage",
             "definition": "Electrode referencing scheme (average, referential, bipolar) — affects how channels are displayed and segmented"},
            {"metric": "Channel Quality Grade",
             "definition": "Combined impedance + SNR assessment per channel: Good (low impedance, high SNR), Fair (marginal), Poor (needs re-application or exclusion)"},
        ],
    }
