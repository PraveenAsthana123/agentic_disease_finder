#!/usr/bin/env python3
"""EEG Channel Quality Map Dashboard
=====================================
Real data: channel_quality (30 patients × 19 channels each) + artifact_annotations
(169 rows) + eeg_acquisition (30 recordings).

Surfaces per-channel impedance grades, SNR distribution, artifact burden,
cross-patient quality heatmap, and EEG acquisition metadata.  All data is
read directly from clinical.db — never fabricated.
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "clinical.db")

# Standard 10-20 channels in this dataset
CHANNELS_10_20 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# Region groupings
CHANNEL_REGIONS: Dict[str, str] = {
    "Fp1": "Frontal", "Fp2": "Frontal",
    "F3": "Frontal", "F4": "Frontal",
    "F7": "Frontal", "F8": "Frontal",
    "Fz": "Frontal",
    "C3": "Central", "C4": "Central", "Cz": "Central",
    "T3": "Temporal", "T4": "Temporal", "T5": "Temporal", "T6": "Temporal",
    "P3": "Parietal", "P4": "Parietal", "Pz": "Parietal",
    "O1": "Occipital", "O2": "Occipital",
}

IMPEDANCE_GRADE_ORDER = {"Good": 0, "Fair": 1, "Poor": 2}
QUALITY_GRADE_ORDER = {"Good": 0, "Fair": 1, "Poor": 2}


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _load_channel_quality():
    """Return list of (patient_id, channels_list) from channel_quality table."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM channel_quality ORDER BY patient_id"
    ).fetchall()
    conn.close()
    result = []
    for patient_id, fjson in rows:
        try:
            data = json.loads(fjson)
            channels = data.get("channels", [])
            result.append((patient_id, channels))
        except Exception:
            pass
    return result


def _load_artifacts():
    """Return list of artifact dicts from artifact_annotations."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM artifact_annotations ORDER BY patient_id"
    ).fetchall()
    conn.close()
    result = []
    for patient_id, fjson in rows:
        try:
            data = json.loads(fjson)
            data["patient_id"] = patient_id
            result.append(data)
        except Exception:
            pass
    return result


def _load_acquisitions():
    """Return list of acquisition dicts from eeg_acquisition."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM eeg_acquisition ORDER BY patient_id"
    ).fetchall()
    conn.close()
    result = []
    for patient_id, fjson in rows:
        try:
            data = json.loads(fjson)
            data["patient_id"] = patient_id
            result.append(data)
        except Exception:
            pass
    return result


# ── overview ────────────────────────────────────────────────────────────────

def overview() -> Dict[str, Any]:
    """KPIs, per-channel aggregate stats, impedance grade distribution,
    quality grade distribution, region summary, top problematic channels."""
    cq = _load_channel_quality()
    arts = _load_artifacts()

    n_patients = len(cq)
    total_channel_records = sum(len(chs) for _, chs in cq)

    # Per-channel aggregates across all patients
    ch_imp: Dict[str, List[float]] = defaultdict(list)
    ch_snr: Dict[str, List[float]] = defaultdict(list)
    ch_imp_grade: Dict[str, List[str]] = defaultdict(list)
    ch_qual_grade: Dict[str, List[str]] = defaultdict(list)

    imp_grade_counts: Dict[str, int] = defaultdict(int)
    qual_grade_counts: Dict[str, int] = defaultdict(int)

    for _, channels in cq:
        for ch in channels:
            name = ch.get("channel", "")
            if not name:
                continue
            imp = ch.get("impedance_kohm")
            snr = ch.get("snr_db")
            ig = ch.get("impedance_grade", "Unknown")
            qg = ch.get("quality_grade", "Unknown")
            if imp is not None:
                ch_imp[name].append(float(imp))
            if snr is not None:
                ch_snr[name].append(float(snr))
            ch_imp_grade[name].append(ig)
            ch_qual_grade[name].append(qg)
            imp_grade_counts[ig] += 1
            qual_grade_counts[qg] += 1

    # Per-channel summary table (sorted by avg impedance descending)
    channel_summary = []
    for ch in CHANNELS_10_20:
        imps = ch_imp.get(ch, [])
        snrs = ch_snr.get(ch, [])
        grades = ch_imp_grade.get(ch, [])
        qgrades = ch_qual_grade.get(ch, [])
        poor_imp = grades.count("Poor")
        poor_qual = qgrades.count("Poor")
        channel_summary.append({
            "channel": ch,
            "region": CHANNEL_REGIONS.get(ch, "Other"),
            "avg_impedance_kohm": round(mean(imps), 1) if imps else None,
            "avg_snr_db": round(mean(snrs), 1) if snrs else None,
            "poor_impedance_count": poor_imp,
            "poor_quality_count": poor_qual,
            "n_patients": len(imps),
            "impedance_grade_dominant": max(
                ("Good", "Fair", "Poor"),
                key=lambda g: grades.count(g),
            ) if grades else "Unknown",
        })
    channel_summary.sort(key=lambda x: (x["avg_impedance_kohm"] or 0), reverse=True)

    # Artifact counts by channel
    art_by_ch: Dict[str, int] = defaultdict(int)
    art_by_type: Dict[str, int] = defaultdict(int)
    for a in arts:
        ch = a.get("channel")
        atype = a.get("artifact_type", "unknown")
        if ch:
            art_by_ch[ch] += 1
        art_by_type[atype] += 1

    # Region summary
    region_data: Dict[str, Dict[str, list]] = defaultdict(lambda: {"imp": [], "snr": []})
    for ch_stat in channel_summary:
        reg = ch_stat["region"]
        if ch_stat["avg_impedance_kohm"] is not None:
            region_data[reg]["imp"].append(ch_stat["avg_impedance_kohm"])
        if ch_stat["avg_snr_db"] is not None:
            region_data[reg]["snr"].append(ch_stat["avg_snr_db"])

    region_summary = []
    for reg in ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]:
        d = region_data.get(reg, {"imp": [], "snr": []})
        region_summary.append({
            "region": reg,
            "avg_impedance_kohm": round(mean(d["imp"]), 1) if d["imp"] else None,
            "avg_snr_db": round(mean(d["snr"]), 1) if d["snr"] else None,
            "n_channels": len(d["imp"]),
        })

    # Overall KPIs
    all_imps = [v for vs in ch_imp.values() for v in vs]
    all_snrs = [v for vs in ch_snr.values() for v in vs]
    poor_total = imp_grade_counts.get("Poor", 0)
    good_total = imp_grade_counts.get("Good", 0)

    return {
        "kpis": {
            "n_patients": n_patients,
            "total_channel_records": total_channel_records,
            "n_channels": len(CHANNELS_10_20),
            "avg_impedance_kohm": round(mean(all_imps), 1) if all_imps else None,
            "avg_snr_db": round(mean(all_snrs), 1) if all_snrs else None,
            "poor_impedance_pct": round(poor_total / (poor_total + good_total + imp_grade_counts.get("Fair", 0)) * 100, 1) if all_imps else 0,
            "total_artifacts": len(arts),
            "artifact_types": len(art_by_type),
        },
        "impedance_grade_distribution": [
            {"grade": g, "count": imp_grade_counts.get(g, 0)}
            for g in ["Good", "Fair", "Poor"]
        ],
        "quality_grade_distribution": [
            {"grade": g, "count": qual_grade_counts.get(g, 0)}
            for g in ["Good", "Fair", "Poor"]
        ],
        "channel_summary": channel_summary,
        "top_problematic_channels": sorted(
            channel_summary, key=lambda x: x["poor_impedance_count"], reverse=True
        )[:6],
        "region_summary": region_summary,
        "artifact_by_type": [
            {"artifact_type": k, "count": v}
            for k, v in sorted(art_by_type.items(), key=lambda x: -x[1])
        ],
        "top_artifact_channels": [
            {"channel": k, "artifact_count": v}
            for k, v in sorted(art_by_ch.items(), key=lambda x: -x[1])[:10]
        ],
    }


# ── breakdown ───────────────────────────────────────────────────────────────

def breakdown() -> Dict[str, Any]:
    """Per-patient channel quality cards, cross-patient heatmap data,
    artifact detail table, acquisition parameters."""
    cq = _load_channel_quality()
    arts = _load_artifacts()
    acqs = _load_acquisitions()

    # Per-patient summary cards
    patient_cards = []
    for patient_id, channels in cq:
        if not channels:
            continue
        imps = [c.get("impedance_kohm") for c in channels if c.get("impedance_kohm") is not None]
        snrs = [c.get("snr_db") for c in channels if c.get("snr_db") is not None]
        poor_count = sum(1 for c in channels if c.get("impedance_grade") == "Poor")
        good_count = sum(1 for c in channels if c.get("impedance_grade") == "Good")
        poor_q = sum(1 for c in channels if c.get("quality_grade") == "Poor")
        overall = "Good" if poor_count == 0 else ("Fair" if poor_count <= 3 else "Poor")
        patient_cards.append({
            "patient_id": patient_id,
            "n_channels": len(channels),
            "avg_impedance_kohm": round(mean(imps), 1) if imps else None,
            "avg_snr_db": round(mean(snrs), 1) if snrs else None,
            "good_channels": good_count,
            "poor_channels": poor_count,
            "poor_quality_channels": poor_q,
            "overall_grade": overall,
            "channels": [
                {
                    "channel": c.get("channel"),
                    "impedance_kohm": c.get("impedance_kohm"),
                    "impedance_grade": c.get("impedance_grade"),
                    "snr_db": c.get("snr_db"),
                    "quality_grade": c.get("quality_grade"),
                }
                for c in channels
            ],
        })

    # Cross-patient impedance heatmap (channel × patient, value = impedance_kohm)
    # Return channel list and per-patient row
    heatmap_patients = [pc["patient_id"] for pc in patient_cards]
    heatmap_rows = []
    for ch in CHANNELS_10_20:
        row_vals = []
        for _, channels in cq:
            match = next((c for c in channels if c.get("channel") == ch), None)
            row_vals.append(round(match["impedance_kohm"], 1) if match and match.get("impedance_kohm") is not None else None)
        heatmap_rows.append({"channel": ch, "region": CHANNEL_REGIONS.get(ch, "Other"), "values": row_vals})

    # Artifact detail table
    artifact_table = []
    for a in arts:
        artifact_table.append({
            "patient_id": a.get("patient_id"),
            "channel": a.get("channel"),
            "artifact_type": a.get("artifact_type"),
            "start_time_min": a.get("start_time_min"),
            "duration_sec": a.get("duration_sec"),
            "severity": a.get("severity"),
        })
    artifact_table.sort(key=lambda x: (x["artifact_type"] or "", x["channel"] or ""))

    # Artifact severity distribution
    sev_counts: Dict[str, int] = defaultdict(int)
    for a in arts:
        sev_counts[a.get("severity", "unknown")] += 1

    # Acquisition parameters summary
    rec_types: Dict[str, int] = defaultdict(int)
    durations = []
    sampling_rates: Dict[int, int] = defaultdict(int)
    for acq in acqs:
        rec_types[acq.get("recording_type", "unknown")] += 1
        dur = acq.get("duration_min")
        if dur is not None:
            durations.append(float(dur))
        sr = acq.get("sampling_rate")
        if sr is not None:
            sampling_rates[int(sr)] += 1

    return {
        "patient_cards": patient_cards,
        "heatmap": {
            "patients": heatmap_patients,
            "channels": heatmap_rows,
        },
        "artifact_table": artifact_table[:60],  # cap at 60 for payload size
        "artifact_severity_distribution": [
            {"severity": k, "count": v}
            for k, v in sorted(sev_counts.items(), key=lambda x: -x[1])
        ],
        "acquisition_summary": {
            "recording_types": [
                {"type": k, "count": v} for k, v in sorted(rec_types.items(), key=lambda x: -x[1])
            ],
            "avg_duration_min": round(mean(durations), 1) if durations else None,
            "sampling_rates": [
                {"rate_hz": k, "count": v} for k, v in sorted(sampling_rates.items())
            ],
        },
    }


# ── definitions ─────────────────────────────────────────────────────────────

def definitions() -> Dict[str, Any]:
    """Glossary of EEG channel quality terms, impedance standards, SNR thresholds."""
    return {
        "terms": [
            {
                "term": "Impedance (kΩ)",
                "definition": "Resistance at the electrode-skin interface. Lower is better. ACNS recommends < 5 kΩ for clinical recording; > 10 kΩ degrades signal quality.",
                "thresholds": {"Good": "< 5 kΩ", "Fair": "5–10 kΩ", "Poor": "> 10 kΩ"},
            },
            {
                "term": "SNR (Signal-to-Noise Ratio, dB)",
                "definition": "Ratio of neural signal power to background noise. Higher SNR indicates cleaner EEG. Clinical threshold: ≥ 20 dB acceptable, ≥ 30 dB excellent.",
                "thresholds": {"Good": "≥ 20 dB", "Fair": "10–20 dB", "Poor": "< 10 dB"},
            },
            {
                "term": "10-20 Electrode System",
                "definition": "International standard for EEG electrode placement. 19 electrodes placed at 10% and 20% intervals of the skull circumference, covering frontal, temporal, parietal, occipital, and central regions.",
            },
            {
                "term": "Artifact",
                "definition": "Non-cerebral signal contamination in EEG. Common types: muscle (EMG), eye movement (ocular), sweat (electrode drift), movement, cardiac (ECG), and line noise (60 Hz).",
                "types": ["muscle", "ocular", "sweat", "movement", "cardiac", "line_noise"],
            },
            {
                "term": "Quality Grade",
                "definition": "Composite channel quality assessment combining impedance, SNR, and artifact burden. Good = suitable for clinical interpretation; Fair = usable with caution; Poor = re-recording recommended.",
            },
            {
                "term": "Channel Re-record Request",
                "definition": "Clinical workflow action triggered when a channel's quality grade is Poor. The technician is alerted to reseat the electrode and re-check impedance before proceeding.",
            },
        ],
        "regions": [
            {"region": "Frontal", "channels": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
             "clinical_significance": "Executive function, motor planning, seizure onset zone in frontal lobe epilepsy"},
            {"region": "Central", "channels": ["C3", "C4", "Cz"],
             "clinical_significance": "Sensorimotor cortex, sleep spindles, mu rhythm (motor imagery BCI)"},
            {"region": "Temporal", "channels": ["T3", "T4", "T5", "T6"],
             "clinical_significance": "Most common seizure onset zone (TLE); memory, language, auditory processing"},
            {"region": "Parietal", "channels": ["P3", "P4", "Pz"],
             "clinical_significance": "Somatosensory processing, spatial awareness, P300 event-related potential"},
            {"region": "Occipital", "channels": ["O1", "O2"],
             "clinical_significance": "Visual cortex, alpha rhythm (8-13 Hz), photosensitivity assessment"},
        ],
        "data_sources": {
            "channel_quality": "30 patients × 19 channels — impedance, SNR, and quality grade per electrode",
            "artifact_annotations": "169 annotated artifact segments — type, channel, duration, severity",
            "eeg_acquisition": "30 recordings — type, duration, sampling rate, montage, electrode system",
        },
        "standards": [
            "ACNS (American Clinical Neurophysiology Society) electrode placement guidelines",
            "IEC 60601-2-26 Medical electrical equipment — EEG safety standards",
            "IFCN (International Federation of Clinical Neurophysiology) recording standards",
        ],
    }


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "overview"
    if target == "overview":
        print(json.dumps(overview(), indent=2))
    elif target == "breakdown":
        d = breakdown()
        # Truncate for display
        d["patient_cards"] = d["patient_cards"][:3]
        d["heatmap"]["channels"] = d["heatmap"]["channels"][:5]
        print(json.dumps(d, indent=2))
    elif target == "definitions":
        print(json.dumps(definitions(), indent=2))
