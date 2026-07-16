"""EEG Channel Quality Dashboard — channel impedance and signal quality analytics from clinical.db.

Tracks EEG channel quality across the standard 10-20 montage (19 channels):
impedance grades, SNR, per-channel averages, per-patient summaries,
poor-channel alerts, heatmap data, and scatter data.

Sources:
- channel_quality table (id, patient_id, fields_json, created_at)
  fields_json contains: {"channels": [{"channel": "Fp1", "impedance_kohm": ...,
  "impedance_grade": ..., "snr_db": ..., "quality_grade": ...}, ...]}
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

CHANNELS_10_20 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def _load_all_channels(conn):
    """Load and parse all channel_quality rows, returning a list of
    (patient_id, created_at, channels_list) tuples."""
    cur = conn.cursor()
    cur.execute("SELECT patient_id, fields_json, created_at FROM channel_quality ORDER BY created_at")
    rows = cur.fetchall()
    result = []
    for patient_id, fields_json, created_at in rows:
        try:
            data = json.loads(fields_json)
            channels = data.get("channels", [])
        except (json.JSONDecodeError, TypeError):
            channels = []
        result.append((patient_id, created_at, channels))
    return result


def overview():
    """High-level channel quality metrics: totals, averages, grade distributions,
    per-channel averages, monthly trend."""
    conn = _conn()
    all_data = _load_all_channels(conn)
    conn.close()

    total_recordings = len(all_data)
    patient_ids = set(r[0] for r in all_data)
    total_patients = len(patient_ids)
    total_channels = len(CHANNELS_10_20)

    # Flatten all channel entries
    all_channels = []
    for patient_id, created_at, channels in all_data:
        for ch in channels:
            ch["_patient_id"] = patient_id
            ch["_created_at"] = created_at
            all_channels.append(ch)

    total_channel_entries = len(all_channels)

    # Overall averages
    impedances = [ch["impedance_kohm"] for ch in all_channels if "impedance_kohm" in ch]
    snrs = [ch["snr_db"] for ch in all_channels if "snr_db" in ch]
    avg_impedance_kohm = round(sum(impedances) / len(impedances), 2) if impedances else 0
    avg_snr_db = round(sum(snrs) / len(snrs), 2) if snrs else 0

    # Impedance grade distribution
    imp_grades = {}
    for ch in all_channels:
        g = ch.get("impedance_grade", "Unknown")
        imp_grades[g] = imp_grades.get(g, 0) + 1
    impedance_grade_distribution = [
        {"grade": g, "count": c}
        for g, c in sorted(imp_grades.items(), key=lambda x: -x[1])
    ]

    # Quality grade distribution
    qual_grades = {}
    for ch in all_channels:
        g = ch.get("quality_grade", "Unknown")
        qual_grades[g] = qual_grades.get(g, 0) + 1
    quality_grade_distribution = [
        {"grade": g, "count": c}
        for g, c in sorted(qual_grades.items(), key=lambda x: -x[1])
    ]

    # Overall good percentages
    good_imp = sum(1 for ch in all_channels if ch.get("impedance_grade") == "Good")
    good_qual = sum(1 for ch in all_channels if ch.get("quality_grade") == "Good")
    overall_good_impedance_pct = round(100.0 * good_imp / total_channel_entries, 1) if total_channel_entries else 0
    overall_good_quality_pct = round(100.0 * good_qual / total_channel_entries, 1) if total_channel_entries else 0

    # Per-channel averages across all patients
    ch_data = {name: {"impedances": [], "snrs": [], "good_imp": 0, "good_qual": 0, "total": 0}
               for name in CHANNELS_10_20}
    for ch in all_channels:
        name = ch.get("channel")
        if name in ch_data:
            ch_data[name]["total"] += 1
            if "impedance_kohm" in ch:
                ch_data[name]["impedances"].append(ch["impedance_kohm"])
            if "snr_db" in ch:
                ch_data[name]["snrs"].append(ch["snr_db"])
            if ch.get("impedance_grade") == "Good":
                ch_data[name]["good_imp"] += 1
            if ch.get("quality_grade") == "Good":
                ch_data[name]["good_qual"] += 1

    per_channel_avg = []
    for name in CHANNELS_10_20:
        d = ch_data[name]
        t = d["total"] or 1
        per_channel_avg.append({
            "channel": name,
            "avg_impedance_kohm": round(sum(d["impedances"]) / len(d["impedances"]), 2) if d["impedances"] else 0,
            "avg_snr_db": round(sum(d["snrs"]) / len(d["snrs"]), 2) if d["snrs"] else 0,
            "pct_good_impedance": round(100.0 * d["good_imp"] / t, 1),
            "pct_good_quality": round(100.0 * d["good_qual"] / t, 1),
        })

    # Monthly trend
    monthly = {}
    for patient_id, created_at, channels in all_data:
        month = created_at[:7] if created_at and len(created_at) >= 7 else "Unknown"
        if month not in monthly:
            monthly[month] = {"recordings": 0, "impedances": [], "snrs": []}
        monthly[month]["recordings"] += 1
        for ch in channels:
            if "impedance_kohm" in ch:
                monthly[month]["impedances"].append(ch["impedance_kohm"])
            if "snr_db" in ch:
                monthly[month]["snrs"].append(ch["snr_db"])

    monthly_trend = []
    for month in sorted(monthly.keys()):
        m = monthly[month]
        monthly_trend.append({
            "month": month,
            "recordings": m["recordings"],
            "avg_impedance": round(sum(m["impedances"]) / len(m["impedances"]), 2) if m["impedances"] else 0,
            "avg_snr": round(sum(m["snrs"]) / len(m["snrs"]), 2) if m["snrs"] else 0,
        })

    return {
        "total_recordings": total_recordings,
        "total_patients": total_patients,
        "total_channels": total_channels,
        "avg_impedance_kohm": avg_impedance_kohm,
        "avg_snr_db": avg_snr_db,
        "overall_good_impedance_pct": overall_good_impedance_pct,
        "overall_good_quality_pct": overall_good_quality_pct,
        "impedance_grade_distribution": impedance_grade_distribution,
        "quality_grade_distribution": quality_grade_distribution,
        "per_channel_avg": per_channel_avg,
        "monthly_trend": monthly_trend,
    }


def breakdown():
    """Per-patient summaries, poor-channel alerts, heatmap data, scatter data."""
    conn = _conn()
    all_data = _load_all_channels(conn)
    conn.close()

    # Per-patient summary
    per_patient = []
    for patient_id, created_at, channels in all_data:
        if not channels:
            continue
        impedances = [ch["impedance_kohm"] for ch in channels if "impedance_kohm" in ch]
        snrs = [ch["snr_db"] for ch in channels if "snr_db" in ch]
        good_count = sum(1 for ch in channels if ch.get("impedance_grade") == "Good")
        fair_count = sum(1 for ch in channels if ch.get("impedance_grade") == "Fair")
        poor_count = sum(1 for ch in channels if ch.get("impedance_grade") == "Poor")
        total_ch = len(channels)
        avg_imp = round(sum(impedances) / len(impedances), 2) if impedances else 0
        avg_s = round(sum(snrs) / len(snrs), 2) if snrs else 0

        # Overall grade: Poor if >30% poor, Fair if >30% fair, else Good
        if total_ch > 0 and poor_count / total_ch > 0.3:
            overall_grade = "Poor"
        elif total_ch > 0 and (poor_count + fair_count) / total_ch > 0.3:
            overall_grade = "Fair"
        else:
            overall_grade = "Good"

        per_patient.append({
            "patient_id": patient_id,
            "avg_impedance": avg_imp,
            "avg_snr": avg_s,
            "good_channels": good_count,
            "fair_channels": fair_count,
            "poor_channels": poor_count,
            "overall_grade": overall_grade,
            "created_at": created_at,
        })

    per_patient.sort(key=lambda x: x["avg_impedance"])

    # Poor channels alert: patients with >30% poor impedance channels
    poor_channels_alert = []
    for p in per_patient:
        total = p["good_channels"] + p["fair_channels"] + p["poor_channels"]
        if total > 0 and p["poor_channels"] / total > 0.3:
            poor_channels_alert.append({
                "patient_id": p["patient_id"],
                "poor_count": p["poor_channels"],
                "total": total,
                "pct_poor": round(100.0 * p["poor_channels"] / total, 1),
            })

    # Channel impedance heatmap: one row per patient, columns = channel impedances
    channel_impedance_heatmap = []
    for patient_id, created_at, channels in all_data:
        row = {"patient_id": patient_id}
        ch_map = {ch["channel"]: ch.get("impedance_kohm", None) for ch in channels}
        for name in CHANNELS_10_20:
            row[name] = ch_map.get(name)
        channel_impedance_heatmap.append(row)

    # Channel quality heatmap: one row per patient, columns = quality grades
    channel_quality_heatmap = []
    for patient_id, created_at, channels in all_data:
        row = {"patient_id": patient_id}
        ch_map = {ch["channel"]: ch.get("quality_grade", None) for ch in channels}
        for name in CHANNELS_10_20:
            row[name] = ch_map.get(name)
        channel_quality_heatmap.append(row)

    # Impedance vs SNR scatter: flattened channel data
    impedance_vs_snr = []
    for patient_id, created_at, channels in all_data:
        for ch in channels:
            if "impedance_kohm" in ch and "snr_db" in ch:
                impedance_vs_snr.append({
                    "channel": ch["channel"],
                    "impedance_kohm": ch["impedance_kohm"],
                    "snr_db": ch["snr_db"],
                    "patient_id": patient_id,
                })

    return {
        "per_patient": per_patient,
        "poor_channels_alert": poor_channels_alert,
        "channel_impedance_heatmap": channel_impedance_heatmap,
        "channel_quality_heatmap": channel_quality_heatmap,
        "impedance_vs_snr": impedance_vs_snr,
    }


def definitions():
    """Clinical glossary, field definitions, grade meanings, channel positions."""
    return {
        "fields": {
            "patient_id": "Unique patient identifier linked to the recording",
            "fields_json": "JSON blob containing per-channel impedance and quality data",
            "created_at": "ISO timestamp when the channel quality assessment was recorded",
            "channel": "EEG electrode position name in the International 10-20 System",
            "impedance_kohm": "Electrode-skin contact impedance in kilo-ohms (lower is better)",
            "impedance_grade": "Categorical impedance rating: Good (<5 kohm), Fair (5-10 kohm), Poor (>10 kohm)",
            "snr_db": "Signal-to-noise ratio in decibels (higher is better)",
            "quality_grade": "Overall channel signal quality rating: Good, Fair, or Poor",
        },
        "impedance_grades": {
            "Good": "Impedance <5 kohm — optimal electrode contact, minimal signal loss, meets ACNS standards",
            "Fair": "Impedance 5-10 kohm — acceptable for most clinical recordings but may introduce noise in high-frequency bands",
            "Poor": "Impedance >10 kohm — likely to produce artifact, electrode should be re-gelled or replaced before proceeding",
        },
        "quality_grades": {
            "Good": "Channel signal is clean with minimal artifact — suitable for clinical interpretation and automated analysis",
            "Fair": "Channel has intermittent artifact or moderate noise — usable with caution, may require manual review",
            "Poor": "Channel is heavily contaminated by artifact or noise — unreliable for clinical use, should be excluded or re-acquired",
        },
        "channels": {
            "Fp1": "Left frontopolar — forehead region, sensitive to eye blink artifact",
            "Fp2": "Right frontopolar — forehead region, sensitive to eye blink artifact",
            "F3": "Left frontal — anterior brain activity, cognitive processing",
            "F4": "Right frontal — anterior brain activity, cognitive processing",
            "C3": "Left central — sensorimotor cortex, mu rhythm",
            "C4": "Right central — sensorimotor cortex, mu rhythm",
            "P3": "Left parietal — posterior association cortex, alpha rhythm",
            "P4": "Right parietal — posterior association cortex, alpha rhythm",
            "O1": "Left occipital — primary visual cortex, posterior alpha",
            "O2": "Right occipital — primary visual cortex, posterior alpha",
            "F7": "Left anterior temporal — inferior frontal, language areas",
            "F8": "Right anterior temporal — inferior frontal",
            "T3": "Left mid-temporal — auditory cortex, temporal lobe epilepsy focus",
            "T4": "Right mid-temporal — auditory cortex, temporal lobe epilepsy focus",
            "T5": "Left posterior temporal — posterior language areas",
            "T6": "Right posterior temporal — posterior association areas",
            "Fz": "Midline frontal — frontal midline theta, supplementary motor area",
            "Cz": "Midline central — vertex sharp waves, central reference",
            "Pz": "Midline parietal — P300 component, posterior midline alpha",
        },
        "glossary": {
            "Impedance": "Resistance to electrical current flow at the electrode-skin interface, measured in kilo-ohms (kohm). Lower impedance means better signal quality.",
            "SNR": "Signal-to-Noise Ratio — the ratio of desired EEG signal power to background noise power, expressed in decibels (dB). Higher SNR indicates cleaner signal.",
            "10-20 System": "International standard for scalp electrode placement. Numbers refer to proportional distances (10% and 20%) between skull landmarks (nasion, inion, preauricular points).",
            "Montage": "A specific arrangement and pairing of EEG channels for display. Common types: bipolar (sequential electrode pairs), referential (each electrode vs. common reference), average reference.",
            "Electrode": "A conductive sensor placed on the scalp to detect the brain's electrical activity. Typically made of silver/silver-chloride (Ag/AgCl) for clinical EEG.",
            "Artifact": "Non-cerebral electrical activity contaminating the EEG signal. Sources include eye movement (EOG), muscle activity (EMG), electrode pop, 50/60 Hz line noise, and sweat.",
            "Ground Electrode": "A reference point (usually at AFz or Fpz) that helps the amplifier reject common-mode noise. Essential for accurate differential recording.",
            "Reference Electrode": "The electrode against which all other channels are measured in a referential montage. Common choices: linked ears (A1+A2), Cz, or average reference.",
            "Gel Bridge": "Unintended conductive path between adjacent electrodes caused by excess conductive gel, artificially reducing apparent impedance and shorting channels together.",
            "ACNS": "American Clinical Neurophysiology Society — sets standards for clinical EEG recording, including impedance thresholds and minimum technical requirements.",
            "Amplifier Saturation": "When electrode impedance is too high or signal amplitude too large, the amplifier reaches its maximum output range, clipping the signal and producing flat-topped waveforms.",
            "Electrode Pop": "A transient high-amplitude artifact caused by sudden change in electrode impedance, often from loose contact or air bubble under the electrode.",
        },
        "clinical_notes": [
            "ACNS guidelines recommend electrode impedances below 5 kohm for clinical EEG recordings. Impedances above 10 kohm are unacceptable and require electrode re-application.",
            "SNR above 20 dB is generally preferred for reliable EEG interpretation. Channels with SNR below 10 dB may miss low-amplitude epileptiform discharges.",
            "Frontopolar channels (Fp1, Fp2) typically have higher impedance and more artifact due to forehead skin oils and proximity to eye movement sources.",
            "Impedance should be checked at the start of recording and periodically during long-term monitoring. Rising impedance often indicates drying gel or patient movement.",
            "Matched impedances across channels (<1 kohm difference between pairs) are as important as absolute values — mismatched impedances amplify common-mode noise rejection failure.",
        ],
    }
