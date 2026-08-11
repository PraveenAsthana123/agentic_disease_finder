"""Signal Quality Dashboard — EEG recording quality from clinical.db.

Provides channel-level impedance / SNR analysis, artifact burden by type /
channel / severity, recording parameter summary (duration, sampling rate,
montage), and per-patient quality scorecards — all drawn from:

  channel_quality      (30 rows, 19-channel JSON payload per patient)
  artifact_annotations (169 rows, per-event artifact metadata)
  eeg_acquisition      (30 rows, recording parameters)
  recording_conditions (30 rows, activation procedures and patient state)

Clinical rationale:
- Signal quality directly determines diagnostic validity. Poor impedance
  (> 10 kΩ) increases noise floor and can mask epileptiform discharges
  (ACNS guideline, 2016).
- SNR < 10 dB indicates channels unsuitable for quantitative analysis;
  the threshold rises to 20 dB for high-frequency oscillation detection.
- Artifact burden > 20 % of recording duration triggers re-recording per
  IFCN standards (Klem et al., Electroencephalogr Clin Neurophysiol, 1999).

Sources:
  channel_quality table      (clinical.db) — 30 rows
  artifact_annotations table (clinical.db) — 169 rows
  eeg_acquisition table      (clinical.db) — 30 rows
  recording_conditions table (clinical.db) — 30 rows
"""

import json
import pathlib
import sqlite3
import statistics
from collections import Counter, defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _load_channel_quality():
    """Return list of (patient_id, channel_dict) tuples — one per channel per patient."""
    con = _conn()
    rows = con.execute("SELECT patient_id, fields_json FROM channel_quality ORDER BY id").fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            fields = json.loads(r["fields_json"])
            for ch in fields.get("channels", []):
                records.append({"patient_id": r["patient_id"], **ch})
        except Exception:
            pass
    return records


def _load_artifacts():
    """Return list of artifact event dicts."""
    con = _conn()
    rows = con.execute("SELECT patient_id, fields_json FROM artifact_annotations ORDER BY id").fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            fields = json.loads(r["fields_json"])
            records.append({"patient_id": r["patient_id"], **fields})
        except Exception:
            pass
    return records


def _load_acquisitions():
    """Return list of recording parameter dicts."""
    con = _conn()
    rows = con.execute("SELECT patient_id, fields_json, created_at FROM eeg_acquisition ORDER BY id").fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            fields = json.loads(r["fields_json"])
            records.append({"patient_id": r["patient_id"], "created_at": r["created_at"], **fields})
        except Exception:
            pass
    return records


def _load_conditions():
    """Return list of recording condition dicts."""
    con = _conn()
    rows = con.execute("SELECT patient_id, fields_json FROM recording_conditions ORDER BY id").fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            fields = json.loads(r["fields_json"])
            records.append({"patient_id": r["patient_id"], **fields})
        except Exception:
            pass
    return records


def _avg(vals):
    valid = [v for v in vals if v is not None]
    return round(statistics.mean(valid), 1) if valid else None


def _impedance_grade(kohm):
    if kohm is None:
        return "Unknown"
    if kohm <= 5:
        return "Good (≤5 kΩ)"
    if kohm <= 10:
        return "Fair (5-10 kΩ)"
    return "Poor (>10 kΩ)"


def _snr_grade(db):
    if db is None:
        return "Unknown"
    if db >= 20:
        return "Excellent (≥20 dB)"
    if db >= 10:
        return "Acceptable (10-20 dB)"
    return "Poor (<10 dB)"


def overview():
    channels = _load_channel_quality()
    artifacts = _load_artifacts()
    acquisitions = _load_acquisitions()
    conditions = _load_conditions()

    n_patients = len({c["patient_id"] for c in channels})
    n_channels = len(channels)
    n_artifacts = len(artifacts)
    n_recordings = len(acquisitions)

    # Impedance distribution
    impedance_dist = Counter(_impedance_grade(c.get("impedance_kohm")) for c in channels)
    poor_impedance = sum(v for k, v in impedance_dist.items() if "Poor" in k)

    # SNR distribution
    snr_dist = Counter(_snr_grade(c.get("snr_db")) for c in channels)
    poor_snr = sum(v for k, v in snr_dist.items() if "Poor" in k)

    # Quality grade distribution
    quality_dist = Counter(c.get("quality_grade") for c in channels if c.get("quality_grade"))

    # Avg impedance and SNR
    avg_impedance = _avg([c.get("impedance_kohm") for c in channels])
    avg_snr = _avg([c.get("snr_db") for c in channels])

    # Artifact type distribution
    artifact_type_dist = Counter(a.get("artifact_type") for a in artifacts if a.get("artifact_type"))

    # Artifact severity distribution
    severity_dist = Counter(a.get("severity") for a in artifacts if a.get("severity"))
    severe_artifacts = severity_dist.get("severe", 0)
    moderate_artifacts = severity_dist.get("moderate", 0)

    # Artifact burden per patient (total duration)
    patient_artifact_duration = defaultdict(float)
    for a in artifacts:
        patient_artifact_duration[a["patient_id"]] += a.get("duration_sec", 0)

    # Recording parameters summary
    sampling_rates = Counter(a.get("sampling_rate") for a in acquisitions if a.get("sampling_rate"))
    montage_dist = Counter(a.get("montage") for a in acquisitions if a.get("montage"))
    recording_types = Counter(a.get("recording_type") for a in acquisitions if a.get("recording_type"))
    avg_duration = _avg([a.get("duration_min") for a in acquisitions])

    # Channel quality grade by channel name (across all patients)
    channel_poor = defaultdict(int)
    channel_total = defaultdict(int)
    for c in channels:
        ch = c.get("channel", "?")
        channel_total[ch] += 1
        if c.get("quality_grade") == "Poor":
            channel_poor[ch] += 1
    channel_poor_rate = [
        {"channel": ch, "poor_count": channel_poor[ch], "total": channel_total[ch],
         "poor_pct": round(channel_poor[ch] / channel_total[ch] * 100, 1)}
        for ch in sorted(channel_total)
    ]

    # Activation procedure coverage
    activation_counts = {
        "hyperventilation": sum(1 for c in conditions if c.get("hyperventilation")),
        "photic_stimulation": sum(1 for c in conditions if c.get("photic_stimulation")),
        "sleep_recorded": sum(1 for c in conditions if c.get("sleep_recorded")),
        "eyes_open": sum(1 for c in conditions if c.get("eyes_open")),
    }

    return {
        "total_patients": n_patients,
        "total_channels": n_channels,
        "channels_per_patient": round(n_channels / n_patients, 1) if n_patients else 0,
        "total_artifacts": n_artifacts,
        "total_recordings": n_recordings,
        "avg_impedance_kohm": avg_impedance,
        "avg_snr_db": avg_snr,
        "poor_impedance_channels": poor_impedance,
        "poor_snr_channels": poor_snr,
        "avg_recording_duration_min": avg_duration,
        "severe_artifacts": severe_artifacts,
        "moderate_artifacts": moderate_artifacts,
        "impedance_distribution": [
            {"grade": k, "count": v}
            for k, v in sorted(impedance_dist.items())
        ],
        "snr_distribution": [
            {"grade": k, "count": v}
            for k, v in sorted(snr_dist.items())
        ],
        "quality_grade_distribution": [
            {"grade": k, "count": v, "color": {"Good": "#22c55e", "Fair": "#f59e0b", "Poor": "#ef4444"}.get(k, "#94a3b8")}
            for k, v in quality_dist.most_common()
        ],
        "artifact_type_distribution": [
            {"type": k, "count": v}
            for k, v in artifact_type_dist.most_common()
        ],
        "artifact_severity_distribution": [
            {"severity": k, "count": v}
            for k, v in sorted(severity_dist.items())
        ],
        "sampling_rate_distribution": [
            {"rate_hz": k, "count": v}
            for k, v in sorted(sampling_rates.items())
        ],
        "montage_distribution": [
            {"montage": k, "count": v}
            for k, v in montage_dist.most_common()
        ],
        "recording_type_distribution": [
            {"type": k, "count": v}
            for k, v in recording_types.most_common()
        ],
        "channel_poor_rate": channel_poor_rate,
        "activation_procedures": [
            {"procedure": k.replace("_", " ").title(), "count": v, "total": len(conditions)}
            for k, v in activation_counts.items()
        ],
    }


def breakdown():
    channels = _load_channel_quality()
    artifacts = _load_artifacts()
    acquisitions = _load_acquisitions()

    # Per-patient channel quality scorecard
    patient_channels = defaultdict(list)
    for c in channels:
        patient_channels[c["patient_id"]].append(c)

    patient_scorecards = []
    for pid, chs in sorted(patient_channels.items()):
        n = len(chs)
        good_ch = sum(1 for c in chs if c.get("quality_grade") == "Good")
        poor_ch = sum(1 for c in chs if c.get("quality_grade") == "Poor")
        avg_imp = _avg([c.get("impedance_kohm") for c in chs])
        avg_snr = _avg([c.get("snr_db") for c in chs])
        # Artifact count for this patient
        pat_artifacts = [a for a in artifacts if a["patient_id"] == pid]
        pat_duration = sum(a.get("duration_sec", 0) for a in pat_artifacts)
        pat_severe = sum(1 for a in pat_artifacts if a.get("severity") == "severe")
        # Recording info
        acq = next((a for a in acquisitions if a["patient_id"] == pid), {})
        patient_scorecards.append({
            "patient_id": pid,
            "total_channels": n,
            "good_channels": good_ch,
            "poor_channels": poor_ch,
            "good_pct": round(good_ch / n * 100, 1) if n else 0,
            "avg_impedance_kohm": avg_imp,
            "avg_snr_db": avg_snr,
            "artifact_count": len(pat_artifacts),
            "artifact_duration_sec": round(pat_duration, 1),
            "severe_artifact_count": pat_severe,
            "recording_type": acq.get("recording_type", "—"),
            "duration_min": acq.get("duration_min"),
            "sampling_rate": acq.get("sampling_rate"),
            "montage": acq.get("montage", "—"),
            "study_date": acq.get("study_date", "—"),
        })

    # Artifact by channel (top channels affected)
    artifact_by_channel = Counter(a.get("channel") for a in artifacts if a.get("channel"))
    top_channels = [
        {"channel": ch, "artifact_count": cnt}
        for ch, cnt in artifact_by_channel.most_common(10)
    ]

    # SNR histogram buckets
    snr_buckets = {"<10 dB": 0, "10-15 dB": 0, "15-20 dB": 0, "20-25 dB": 0, "≥25 dB": 0}
    for c in channels:
        snr = c.get("snr_db")
        if snr is None:
            continue
        if snr < 10:
            snr_buckets["<10 dB"] += 1
        elif snr < 15:
            snr_buckets["10-15 dB"] += 1
        elif snr < 20:
            snr_buckets["15-20 dB"] += 1
        elif snr < 25:
            snr_buckets["20-25 dB"] += 1
        else:
            snr_buckets["≥25 dB"] += 1

    # Impedance histogram buckets
    imp_buckets = {"0-2 kΩ": 0, "2-5 kΩ": 0, "5-10 kΩ": 0, "10-20 kΩ": 0, ">20 kΩ": 0}
    for c in channels:
        imp = c.get("impedance_kohm")
        if imp is None:
            continue
        if imp <= 2:
            imp_buckets["0-2 kΩ"] += 1
        elif imp <= 5:
            imp_buckets["2-5 kΩ"] += 1
        elif imp <= 10:
            imp_buckets["5-10 kΩ"] += 1
        elif imp <= 20:
            imp_buckets["10-20 kΩ"] += 1
        else:
            imp_buckets[">20 kΩ"] += 1

    return {
        "patient_scorecards": patient_scorecards,
        "top_artifact_channels": top_channels,
        "snr_histogram": [{"bucket": k, "count": v} for k, v in snr_buckets.items()],
        "impedance_histogram": [{"bucket": k, "count": v} for k, v in imp_buckets.items()],
    }


def definitions():
    return {
        "dashboard": "Signal Quality Dashboard",
        "role": "Neurophysiologist",
        "data_sources": [
            "channel_quality (clinical.db) — 30 recordings, 19-channel impedance/SNR payloads",
            "artifact_annotations (clinical.db) — 169 artifact events",
            "eeg_acquisition (clinical.db) — 30 recording parameter records",
            "recording_conditions (clinical.db) — 30 activation procedure records",
        ],
        "terms": [
            {
                "term": "Electrode Impedance",
                "definition": (
                    "Resistance (kΩ) between electrode and scalp. ACNS guideline: "
                    "< 5 kΩ = Good; 5-10 kΩ = Fair (acceptable for clinical recording); "
                    "> 10 kΩ = Poor (increased noise, must document or re-apply electrode). "
                    "High impedance amplifies capacitive coupling and 50/60 Hz interference."
                ),
            },
            {
                "term": "Signal-to-Noise Ratio (SNR)",
                "definition": (
                    "Ratio of EEG signal power to background noise (dB). "
                    "≥ 20 dB = Excellent; 10-20 dB = Acceptable for visual interpretation; "
                    "< 10 dB = Poor — channel unsuitable for quantitative analysis or HFO detection "
                    "(Worrell et al., Epilepsia 2008)."
                ),
            },
            {
                "term": "Quality Grade",
                "definition": (
                    "Composite channel quality label: Good (impedance Good AND SNR ≥ 10 dB); "
                    "Fair (impedance Fair OR SNR 10-20 dB); Poor (impedance > 10 kΩ AND/OR SNR < 10 dB). "
                    "Poor channels are flagged for re-recording per IFCN standards."
                ),
            },
            {
                "term": "Artifact Types",
                "definition": (
                    "muscle — EMG contamination from jaw clenching or facial movement (broadband 20-500 Hz); "
                    "movement — electrode displacement artefact (large-amplitude, irregular); "
                    "sweat — slow-drift artefact from perspiration (< 0.5 Hz); "
                    "eye — eye blink (Fp1/2, large 50-100 µV) or lateral eye movement (F7/F8); "
                    "cardiac — ECG QRS contamination (regular rhythm); "
                    "electrode — impedance-related broadband noise."
                ),
            },
            {
                "term": "Artifact Severity",
                "definition": (
                    "mild — transient, < 5 s, localised to 1-2 channels, does not obscure "
                    "underlying EEG; moderate — 5-30 s or affecting 3-5 channels; "
                    "severe — > 30 s, > 5 channels, or obscures clinically relevant activity."
                ),
            },
            {
                "term": "Sampling Rate",
                "definition": (
                    "256 Hz — sufficient for clinical visual review (Nyquist 128 Hz); "
                    "512 Hz — standard for seizure onset zone analysis; "
                    "1024 Hz — required for high-frequency oscillations (HFO, 80-500 Hz). "
                    "IFCN minimum: 256 Hz for routine clinical EEG."
                ),
            },
            {
                "term": "Montage",
                "definition": (
                    "referential — each electrode referenced to a common point (Cz or average); "
                    "bipolar (longitudinal/transverse) — adjacent electrode pairs; "
                    "average reference — each electrode referenced to the mean of all electrodes. "
                    "Montage choice affects spike morphology and localisation."
                ),
            },
            {
                "term": "Activation Procedures",
                "definition": (
                    "Hyperventilation (3-5 min) — provokes absence and other seizure types via "
                    "hypocapnia-induced cerebral vasoconstriction; increases diagnostic yield ~20 %. "
                    "Photic stimulation (1-25 Hz) — elicits photoparoxysmal response in ~5 % of "
                    "epilepsy patients. Sleep recording — captures sleep-activated epileptiform "
                    "discharges (especially frontal-lobe and juvenile myoclonic epilepsy)."
                ),
            },
            {
                "term": "10-20 Electrode System",
                "definition": (
                    "International standard for scalp electrode placement (Jasper, 1958). "
                    "19 standard electrodes: Fp1/2, F3/4/7/8/z, C3/4/z, P3/4/z, O1/2, T3/4/5/6. "
                    "Distances between adjacent electrodes are 10 % or 20 % of nasion-inion "
                    "or pre-auricular distances."
                ),
            },
            {
                "term": "Standards",
                "definition": (
                    "ACNS (American Clinical Neurophysiology Society) guideline for EEG recording; "
                    "IFCN (International Federation of Clinical Neurophysiology) EEG standards; "
                    "Klem et al. 1999 electrode placement guidelines; "
                    "IFSECN Glossary of EEG terminology."
                ),
            },
        ],
        "abbreviations": {
            "SNR": "Signal-to-Noise Ratio",
            "HFO": "High-Frequency Oscillation",
            "ACNS": "American Clinical Neurophysiology Society",
            "IFCN": "International Federation of Clinical Neurophysiology",
            "EEG": "Electroencephalogram",
            "EMG": "Electromyography",
            "ECG": "Electrocardiography",
            "kΩ": "kilo-Ohm (impedance unit)",
            "dB": "decibel (SNR unit)",
        },
    }
