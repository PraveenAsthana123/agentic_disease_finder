"""
Neuro AI Ecosystem — EEG Acquisition Dashboard
================================================
EEG recording analytics from eeg_acquisition + channel_quality tables.

Recording types: routine, LTM, video_eeg, ambulatory
Montages: average, bipolar, referential
Electrode system: 10-20
Sampling rates: 256, 512, 1024 Hz

Real data: eeg_acquisition (30 rows) + channel_quality (30 rows, 570 channel readings)
in clinical.db.  fields_json stores structured data as JSON strings.
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


def overview():
    """EEG acquisition overview — total studies, recording type distribution,
    montage distribution, sampling rate distribution, duration stats,
    channel quality summary, monthly trend."""
    conn = _conn()
    cur = conn.cursor()

    # Total studies
    cur.execute("SELECT COUNT(*) FROM eeg_acquisition")
    total_studies = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM eeg_acquisition")
    total_patients = cur.fetchone()[0]

    # Parse all eeg_acquisition rows
    cur.execute("SELECT fields_json FROM eeg_acquisition")
    acq_rows = [json.loads(r[0]) for r in cur.fetchall()]

    # Recording type distribution
    recording_type_dist = {}
    for row in acq_rows:
        rt = row.get("recording_type", "unknown")
        recording_type_dist[rt] = recording_type_dist.get(rt, 0) + 1

    # Montage distribution
    montage_dist = {}
    for row in acq_rows:
        m = row.get("montage", "unknown")
        montage_dist[m] = montage_dist.get(m, 0) + 1

    # Sampling rate distribution
    sampling_rate_dist = {}
    for row in acq_rows:
        sr = str(row.get("sampling_rate", "unknown"))
        sampling_rate_dist[sr] = sampling_rate_dist.get(sr, 0) + 1

    # Duration stats
    durations = [row.get("duration_min", 0) for row in acq_rows if row.get("duration_min") is not None]
    duration_stats = {
        "avg": round(sum(durations) / max(len(durations), 1), 1),
        "min": min(durations) if durations else 0,
        "max": max(durations) if durations else 0
    }

    # Channel quality summary — parse all channel_quality rows
    cur.execute("SELECT fields_json FROM channel_quality")
    cq_rows = [json.loads(r[0]) for r in cur.fetchall()]

    all_channels = []
    for row in cq_rows:
        channels = row.get("channels", [])
        all_channels.extend(channels)

    total_channels = len(all_channels)

    impedance_grade_dist = {}
    quality_grade_dist = {}
    impedance_values = []
    snr_values = []

    for ch in all_channels:
        ig = ch.get("impedance_grade", "unknown")
        impedance_grade_dist[ig] = impedance_grade_dist.get(ig, 0) + 1

        qg = ch.get("quality_grade", "unknown")
        quality_grade_dist[qg] = quality_grade_dist.get(qg, 0) + 1

        if ch.get("impedance_kohm") is not None:
            impedance_values.append(ch["impedance_kohm"])
        if ch.get("snr_db") is not None:
            snr_values.append(ch["snr_db"])

    avg_impedance_kohm = round(sum(impedance_values) / max(len(impedance_values), 1), 2)
    avg_snr_db = round(sum(snr_values) / max(len(snr_values), 1), 2)

    pct_good_impedance = round(impedance_grade_dist.get("Good", 0) / max(total_channels, 1) * 100, 1)
    pct_good_quality = round(quality_grade_dist.get("Good", 0) / max(total_channels, 1) * 100, 1)

    # Monthly trend by study_date
    cur.execute("""
        SELECT id, fields_json FROM eeg_acquisition
    """)
    monthly_counts = {}
    for row in cur.fetchall():
        fields = json.loads(row[1])
        sd = fields.get("study_date", "")
        month = sd[:7] if sd and len(sd) >= 7 else "unknown"
        monthly_counts[month] = monthly_counts.get(month, 0) + 1

    monthly_trend = [{"month": m, "cnt": c} for m, c in sorted(monthly_counts.items())]

    conn.close()
    return {
        "total_studies": total_studies,
        "total_patients": total_patients,
        "recording_type_distribution": recording_type_dist,
        "montage_distribution": montage_dist,
        "sampling_rate_distribution": sampling_rate_dist,
        "duration_stats": duration_stats,
        "channel_quality_summary": {
            "total_channels": total_channels,
            "impedance_grade_distribution": impedance_grade_dist,
            "quality_grade_distribution": quality_grade_dist
        },
        "avg_impedance_kohm": avg_impedance_kohm,
        "avg_snr_db": avg_snr_db,
        "pct_good_impedance": pct_good_impedance,
        "pct_good_quality": pct_good_quality,
        "monthly_trend": monthly_trend
    }


def breakdown():
    """EEG acquisition breakdown — per-patient summary, per-channel stats,
    recent studies, poor quality channels, recording type detail."""
    conn = _conn()
    cur = conn.cursor()

    # Load all acquisition data with patient_id
    cur.execute("SELECT patient_id, fields_json FROM eeg_acquisition")
    acq_data = []
    for r in cur.fetchall():
        fields = json.loads(r[1])
        fields["patient_id"] = r[0]
        acq_data.append(fields)

    # Load all channel quality data with patient_id
    cur.execute("SELECT patient_id, fields_json FROM channel_quality")
    cq_data = {}
    for r in cur.fetchall():
        fields = json.loads(r[1])
        pid = r[0]
        channels = fields.get("channels", [])
        if pid not in cq_data:
            cq_data[pid] = []
        cq_data[pid].extend(channels)

    # Per-patient summary
    per_patient_summary = []
    patient_acq = {}
    for row in acq_data:
        pid = row["patient_id"]
        if pid not in patient_acq:
            patient_acq[pid] = []
        patient_acq[pid].append(row)

    for pid in sorted(patient_acq.keys()):
        rows = patient_acq[pid]
        rec = rows[0]  # use first recording for type/montage/rate
        channels = cq_data.get(pid, [])
        good_count = sum(1 for c in channels if c.get("quality_grade") == "Good")
        fair_count = sum(1 for c in channels if c.get("quality_grade") == "Fair")
        poor_count = sum(1 for c in channels if c.get("quality_grade") == "Poor")
        per_patient_summary.append({
            "patient_id": pid,
            "recording_type": rec.get("recording_type"),
            "duration_min": rec.get("duration_min"),
            "sampling_rate": rec.get("sampling_rate"),
            "montage": rec.get("montage"),
            "channels_good": good_count,
            "channels_fair": fair_count,
            "channels_poor": poor_count
        })

    # Per-channel stats across all patients
    all_channels = []
    for channels in cq_data.values():
        all_channels.extend(channels)

    channel_stats_map = {}
    for ch in all_channels:
        name = ch.get("channel", "unknown")
        if name not in channel_stats_map:
            channel_stats_map[name] = {
                "impedances": [],
                "snrs": [],
                "impedance_grades": {}
            }
        if ch.get("impedance_kohm") is not None:
            channel_stats_map[name]["impedances"].append(ch["impedance_kohm"])
        if ch.get("snr_db") is not None:
            channel_stats_map[name]["snrs"].append(ch["snr_db"])
        ig = ch.get("impedance_grade", "unknown")
        channel_stats_map[name]["impedance_grades"][ig] = channel_stats_map[name]["impedance_grades"].get(ig, 0) + 1

    per_channel_stats = []
    for ch_name in sorted(channel_stats_map.keys()):
        data = channel_stats_map[ch_name]
        imps = data["impedances"]
        snrs = data["snrs"]
        per_channel_stats.append({
            "channel": ch_name,
            "avg_impedance_kohm": round(sum(imps) / max(len(imps), 1), 2),
            "avg_snr_db": round(sum(snrs) / max(len(snrs), 1), 2),
            "impedance_grade_distribution": data["impedance_grades"]
        })

    # Recent studies — most recent 30 by study_date
    studies_with_dates = []
    cur.execute("SELECT id, patient_id, fields_json, created_at FROM eeg_acquisition")
    for r in cur.fetchall():
        fields = json.loads(r[2])
        studies_with_dates.append({
            "id": r[0],
            "patient_id": r[1],
            "study_date": fields.get("study_date"),
            "recording_type": fields.get("recording_type"),
            "duration_min": fields.get("duration_min"),
            "sampling_rate": fields.get("sampling_rate"),
            "montage": fields.get("montage"),
            "electrode_system": fields.get("electrode_system"),
            "technician_notes": fields.get("technician_notes"),
            "created_at": r[3]
        })
    studies_with_dates.sort(key=lambda x: x.get("study_date") or "", reverse=True)
    recent_studies = studies_with_dates[:30]

    # Poor quality channels grouped by patient
    poor_quality_channels = {}
    for pid, channels in cq_data.items():
        poor = [ch for ch in channels if ch.get("quality_grade") == "Poor"]
        if poor:
            poor_quality_channels[pid] = [
                {
                    "channel": ch.get("channel"),
                    "impedance_kohm": ch.get("impedance_kohm"),
                    "snr_db": ch.get("snr_db"),
                    "impedance_grade": ch.get("impedance_grade")
                }
                for ch in poor
            ]

    # Recording type detail
    type_groups = {}
    for row in acq_data:
        rt = row.get("recording_type", "unknown")
        if rt not in type_groups:
            type_groups[rt] = {"durations": [], "count": 0}
        type_groups[rt]["count"] += 1
        if row.get("duration_min") is not None:
            type_groups[rt]["durations"].append(row["duration_min"])

    # Get avg impedance per recording type by matching patients
    recording_type_detail = []
    for rt, info in sorted(type_groups.items()):
        # find patients with this recording type
        rt_patients = [r["patient_id"] for r in acq_data if r.get("recording_type") == rt]
        rt_impedances = []
        for pid in rt_patients:
            for ch in cq_data.get(pid, []):
                if ch.get("impedance_kohm") is not None:
                    rt_impedances.append(ch["impedance_kohm"])

        durs = info["durations"]
        recording_type_detail.append({
            "recording_type": rt,
            "count": info["count"],
            "avg_duration_min": round(sum(durs) / max(len(durs), 1), 1),
            "avg_impedance_kohm": round(sum(rt_impedances) / max(len(rt_impedances), 1), 2) if rt_impedances else None
        })

    conn.close()
    return {
        "per_patient_summary": per_patient_summary,
        "per_channel_stats": per_channel_stats,
        "recent_studies": recent_studies,
        "poor_quality_channels": poor_quality_channels,
        "recording_type_detail": recording_type_detail
    }


def definitions():
    """EEG acquisition definitions — clinical glossary, recording types,
    montage descriptions, channel regions, and quality references."""
    return {
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography — non-invasive recording of brain electrical activity via scalp electrodes. Measures voltage fluctuations from neuronal ionic currents."},
            {"term": "Impedance", "definition": "Resistance to current flow at the electrode-skin interface, measured in kilohms (kohm). Lower impedance = better signal quality. Target: < 5 kohm."},
            {"term": "SNR", "definition": "Signal-to-Noise Ratio — ratio of desired neural signal power to background noise power, measured in decibels (dB). Higher SNR = cleaner signal."},
            {"term": "10-20 System", "definition": "International standard for electrode placement based on skull landmarks (nasion, inion, preauricular points). Numbers indicate percentage distances; odd = left, even = right, z = midline."},
            {"term": "Montage (Average)", "definition": "Average reference montage — each channel referenced to the average of all electrodes. Reduces common-mode noise but can smear widespread activity."},
            {"term": "Montage (Bipolar)", "definition": "Bipolar montage — records voltage difference between adjacent electrode pairs in chains (anterior-posterior or transverse). Best for localizing focal activity."},
            {"term": "Montage (Referential)", "definition": "Referential montage — each channel referenced to a single common electrode (e.g., Cz, linked ears). Preserves amplitude but susceptible to reference contamination."},
            {"term": "Routine EEG", "definition": "Standard 20-30 minute recording with hyperventilation and photic stimulation activation procedures. First-line screening for epileptiform activity."},
            {"term": "LTM (Long-Term Monitoring)", "definition": "Continuous EEG recording over hours to days, typically in epilepsy monitoring unit. Used for seizure localization and presurgical evaluation."},
            {"term": "Video EEG", "definition": "Simultaneous EEG and video recording to correlate electrographic patterns with clinical behavior during events."},
            {"term": "Ambulatory EEG", "definition": "Portable EEG recording worn by patient at home for 24-72 hours. Captures events in natural environment with limited channels."},
            {"term": "Sampling Rate", "definition": "Number of data points captured per second per channel, in Hz. 256 Hz adequate for clinical EEG; 512-1024 Hz for high-frequency oscillation research."},
            {"term": "Artifact", "definition": "Non-cerebral signal contaminating EEG — sources include muscle (EMG), eye movement (EOG), electrode pop, 60 Hz line noise, and movement."},
            {"term": "Channel Quality Grade", "definition": "Overall channel signal quality rating (Good/Fair/Poor) based on impedance, SNR, artifact burden, and signal continuity."}
        ],
        "channel_regions": [
            {"region": "Frontal", "channels": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"], "description": "Frontopolar (Fp) and frontal (F) electrodes. Monitor executive function, eye movement artifacts, and frontal lobe epileptiform discharges."},
            {"region": "Central", "channels": ["C3", "C4", "Cz"], "description": "Central electrodes over motor/sensory cortex. Key for mu rhythm, rolandic spikes, and central sleep features (vertex waves, sleep spindles)."},
            {"region": "Parietal", "channels": ["P3", "P4", "Pz"], "description": "Parietal electrodes over somatosensory association cortex. Monitor alpha rhythm posterior spread and parietal epileptiform activity."},
            {"region": "Occipital", "channels": ["O1", "O2"], "description": "Occipital electrodes over visual cortex. Primary site for posterior dominant rhythm (alpha), photic driving response, and occipital seizures."},
            {"region": "Temporal", "channels": ["T3", "T4", "T5", "T6"], "description": "Temporal electrodes (T3/T4 anterior, T5/T6 posterior). Most common site for epileptiform discharges in temporal lobe epilepsy."}
        ],
        "quality_thresholds": [
            {"grade": "Good", "impedance_range": "< 5 kohm", "snr_range": "> 15 dB", "description": "Clean signal suitable for clinical interpretation and quantitative analysis."},
            {"grade": "Fair", "impedance_range": "5-10 kohm", "snr_range": "10-15 dB", "description": "Acceptable signal with minor artifacts; may require additional filtering."},
            {"grade": "Poor", "impedance_range": "> 10 kohm", "snr_range": "< 10 dB", "description": "Degraded signal with significant artifact contamination; channel may need re-application or exclusion."}
        ],
        "recording_protocols": [
            {"type": "routine", "typical_duration_min": "20-30", "channels": 19, "notes": "Standard clinical EEG with HV and photic activation"},
            {"type": "LTM", "typical_duration_min": "1440-10080", "channels": 19, "notes": "Continuous monitoring in EMU; medication tapering common"},
            {"type": "video_eeg", "typical_duration_min": "1440-10080", "channels": 19, "notes": "Synchronized video for behavioral correlation"},
            {"type": "ambulatory", "typical_duration_min": "1440-4320", "channels": "8-19", "notes": "Home-based recording; patient diary required"}
        ]
    }
