"""Real-Time EEG QC Dashboard — live recording quality control analytics.

Addresses the EEG Technician challenge: "Repeat recordings waste tech + patient
time" by providing real-time quality checks so problems are caught during
acquisition, not after the neurologist reads the study.

Data sourced from REAL tables in data/clinical.db:
  - eeg_acquisition: recording metadata (type, duration, montage, sample rate)
  - channel_quality: per-channel impedance, SNR, quality grades
  - artifact_annotations: artifact type, channel, timing, severity

QC checks performed per recording:
  1. Impedance check — channels with impedance > 10 kΩ flagged
  2. SNR check — channels with SNR < 15 dB flagged
  3. Artifact burden — % of recording time contaminated by artifacts
  4. Channel quality grade — Good/Fair/Poor per channel
  5. Overall recording verdict — Pass / Needs Attention / Re-record

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# Thresholds for QC checks
IMPEDANCE_WARN_KOHM = 10.0   # > 10 kΩ = concern
IMPEDANCE_FAIL_KOHM = 20.0   # > 20 kΩ = fail
SNR_WARN_DB = 15.0            # < 15 dB = concern
SNR_FAIL_DB = 8.0             # < 8 dB = fail
ARTIFACT_WARN_PCT = 20.0      # > 20% artifact burden = concern
ARTIFACT_FAIL_PCT = 40.0      # > 40% artifact burden = re-record


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _parse_fields(row):
    fj = row.get("fields_json", "{}")
    return json.loads(fj) if isinstance(fj, str) else (fj or {})


def _recording_verdict(impedance_flags, snr_flags, artifact_pct):
    """Determine overall QC verdict for a recording."""
    fail_count = impedance_flags.get("fail", 0) + snr_flags.get("fail", 0)
    warn_count = impedance_flags.get("warn", 0) + snr_flags.get("warn", 0)
    if fail_count >= 3 or artifact_pct > ARTIFACT_FAIL_PCT:
        return "Re-record"
    if fail_count >= 1 or warn_count >= 3 or artifact_pct > ARTIFACT_WARN_PCT:
        return "Needs Attention"
    return "Pass"


def overview():
    """QC summary KPIs + per-recording verdicts + alert list."""
    acquisitions = _rows("SELECT * FROM eeg_acquisition")
    qualities = _rows("SELECT * FROM channel_quality")
    artifacts = _rows("SELECT * FROM artifact_annotations")

    # Build quality lookup by patient_id
    quality_by_patient = {}
    for q in qualities:
        f = _parse_fields(q)
        quality_by_patient[q["patient_id"]] = f.get("channels", [])

    # Build artifact lookup by patient_id
    artifact_by_patient = defaultdict(list)
    for a in artifacts:
        f = _parse_fields(a)
        artifact_by_patient[a["patient_id"]].append(f)

    recordings = []
    verdicts = Counter()
    alerts = []
    total_channels = 0
    flagged_channels = 0
    total_impedance_flags = 0
    total_snr_flags = 0

    for acq in acquisitions:
        pid = acq["patient_id"]
        af = _parse_fields(acq)
        channels = quality_by_patient.get(pid, [])
        patient_artifacts = artifact_by_patient.get(pid, [])
        duration_min = af.get("duration_min", 30)

        # Impedance analysis
        imp_flags = {"warn": 0, "fail": 0}
        snr_flags = {"warn": 0, "fail": 0}
        bad_channels = []
        for ch in channels:
            total_channels += 1
            imp = ch.get("impedance_kohm", 0)
            snr = ch.get("snr_db", 30)
            flagged = False
            if imp > IMPEDANCE_FAIL_KOHM:
                imp_flags["fail"] += 1
                flagged = True
            elif imp > IMPEDANCE_WARN_KOHM:
                imp_flags["warn"] += 1
                flagged = True
            if snr < SNR_FAIL_DB:
                snr_flags["fail"] += 1
                flagged = True
            elif snr < SNR_WARN_DB:
                snr_flags["warn"] += 1
                flagged = True
            if flagged:
                flagged_channels += 1
                bad_channels.append(ch.get("channel", "?"))

        total_impedance_flags += imp_flags["warn"] + imp_flags["fail"]
        total_snr_flags += snr_flags["warn"] + snr_flags["fail"]

        # Artifact burden
        artifact_sec = sum(a.get("duration_sec", 0) for a in patient_artifacts)
        artifact_pct = (artifact_sec / (duration_min * 60) * 100) if duration_min > 0 else 0

        verdict = _recording_verdict(imp_flags, snr_flags, artifact_pct)
        verdicts[verdict] += 1

        rec = {
            "patient_id": pid,
            "recording_type": af.get("recording_type", "routine"),
            "duration_min": duration_min,
            "sampling_rate": af.get("sampling_rate", 256),
            "montage": af.get("montage", "average"),
            "study_date": af.get("study_date", ""),
            "channel_count": len(channels),
            "impedance_warns": imp_flags["warn"],
            "impedance_fails": imp_flags["fail"],
            "snr_warns": snr_flags["warn"],
            "snr_fails": snr_flags["fail"],
            "artifact_count": len(patient_artifacts),
            "artifact_burden_pct": round(artifact_pct, 1),
            "verdict": verdict,
            "bad_channels": bad_channels[:5],
        }
        recordings.append(rec)

        if verdict == "Re-record":
            alerts.append({
                "patient_id": pid,
                "level": "critical",
                "message": f"Recording needs re-recording: {imp_flags['fail']} channel impedance failures, {artifact_pct:.0f}% artifact burden",
                "bad_channels": bad_channels[:5],
            })
        elif verdict == "Needs Attention":
            alerts.append({
                "patient_id": pid,
                "level": "warning",
                "message": f"QC attention needed: {imp_flags['warn']+imp_flags['fail']} impedance flags, {snr_flags['warn']+snr_flags['fail']} SNR flags",
                "bad_channels": bad_channels[:5],
            })

    return {
        "kpis": {
            "total_recordings": len(acquisitions),
            "pass_rate_pct": round(verdicts.get("Pass", 0) / max(len(acquisitions), 1) * 100, 1),
            "needs_attention": verdicts.get("Needs Attention", 0),
            "re_record": verdicts.get("Re-record", 0),
            "total_channels_checked": total_channels,
            "flagged_channels": flagged_channels,
            "impedance_flags": total_impedance_flags,
            "snr_flags": total_snr_flags,
        },
        "verdict_distribution": [
            {"verdict": v, "count": verdicts.get(v, 0)}
            for v in ["Pass", "Needs Attention", "Re-record"]
        ],
        "recordings": sorted(recordings, key=lambda r: (
            {"Re-record": 0, "Needs Attention": 1, "Pass": 2}.get(r["verdict"], 3),
            r["patient_id"]
        )),
        "alerts": alerts,
    }


def breakdown():
    """Per-channel analysis + artifact type breakdown + impedance distribution."""
    qualities = _rows("SELECT * FROM channel_quality")
    artifacts = _rows("SELECT * FROM artifact_annotations")

    # Aggregate channel stats across all recordings
    channel_stats = defaultdict(lambda: {
        "impedances": [], "snrs": [], "grades": Counter()
    })
    for q in qualities:
        f = _parse_fields(q)
        for ch in f.get("channels", []):
            name = ch.get("channel", "?")
            channel_stats[name]["impedances"].append(ch.get("impedance_kohm", 0))
            channel_stats[name]["snrs"].append(ch.get("snr_db", 0))
            channel_stats[name]["grades"][ch.get("quality_grade", "Unknown")] += 1

    channel_summary = []
    for name, stats in sorted(channel_stats.items()):
        imps = stats["impedances"]
        snrs = stats["snrs"]
        avg_imp = sum(imps) / len(imps) if imps else 0
        avg_snr = sum(snrs) / len(snrs) if snrs else 0
        max_imp = max(imps) if imps else 0
        min_snr = min(snrs) if snrs else 0
        fail_rate = sum(1 for i in imps if i > IMPEDANCE_FAIL_KOHM) / max(len(imps), 1) * 100
        channel_summary.append({
            "channel": name,
            "avg_impedance_kohm": round(avg_imp, 1),
            "max_impedance_kohm": round(max_imp, 1),
            "avg_snr_db": round(avg_snr, 1),
            "min_snr_db": round(min_snr, 1),
            "impedance_fail_rate_pct": round(fail_rate, 1),
            "recordings": len(imps),
            "grade_distribution": dict(stats["grades"]),
        })

    # Artifact type breakdown
    artifact_types = Counter()
    artifact_severity = Counter()
    artifact_by_channel = Counter()
    for a in artifacts:
        f = _parse_fields(a)
        artifact_types[f.get("artifact_type", "unknown")] += 1
        artifact_severity[f.get("severity", "unknown")] += 1
        artifact_by_channel[f.get("channel", "unknown")] += 1

    # Impedance distribution histogram
    all_impedances = []
    for q in qualities:
        f = _parse_fields(q)
        for ch in f.get("channels", []):
            all_impedances.append(ch.get("impedance_kohm", 0))
    imp_buckets = {"0-5 kΩ": 0, "5-10 kΩ": 0, "10-15 kΩ": 0, "15-20 kΩ": 0, ">20 kΩ": 0}
    for imp in all_impedances:
        if imp <= 5:
            imp_buckets["0-5 kΩ"] += 1
        elif imp <= 10:
            imp_buckets["5-10 kΩ"] += 1
        elif imp <= 15:
            imp_buckets["10-15 kΩ"] += 1
        elif imp <= 20:
            imp_buckets["15-20 kΩ"] += 1
        else:
            imp_buckets[">20 kΩ"] += 1

    return {
        "channel_summary": channel_summary,
        "artifact_type_breakdown": [
            {"type": t, "count": c} for t, c in artifact_types.most_common()
        ],
        "artifact_severity_breakdown": [
            {"severity": s, "count": c} for s, c in artifact_severity.most_common()
        ],
        "artifact_by_channel": [
            {"channel": ch, "count": c} for ch, c in artifact_by_channel.most_common(10)
        ],
        "impedance_distribution": [
            {"bucket": b, "count": c} for b, c in imp_buckets.items()
        ],
    }


def definitions():
    """QC metric definitions and thresholds."""
    return {
        "title": "Real-Time EEG QC Dashboard",
        "purpose": "Catch recording quality problems during acquisition so technicians can fix them before the session ends — preventing costly repeat recordings and wasted patient time.",
        "data_sources": [
            "eeg_acquisition — recording metadata (type, duration, montage, sample rate)",
            "channel_quality — per-channel impedance (kΩ), SNR (dB), quality grade",
            "artifact_annotations — artifact type, channel, timing, severity",
        ],
        "qc_checks": [
            {
                "check": "Impedance",
                "unit": "kΩ",
                "warn_threshold": IMPEDANCE_WARN_KOHM,
                "fail_threshold": IMPEDANCE_FAIL_KOHM,
                "description": "Electrode-scalp contact resistance. High impedance degrades signal quality and increases noise.",
            },
            {
                "check": "SNR (Signal-to-Noise Ratio)",
                "unit": "dB",
                "warn_threshold": SNR_WARN_DB,
                "fail_threshold": SNR_FAIL_DB,
                "description": "Ratio of neural signal power to background noise. Low SNR makes clinical interpretation unreliable.",
            },
            {
                "check": "Artifact Burden",
                "unit": "%",
                "warn_threshold": ARTIFACT_WARN_PCT,
                "fail_threshold": ARTIFACT_FAIL_PCT,
                "description": "Percentage of recording time contaminated by non-neural artifacts (muscle, blink, sweat, movement).",
            },
        ],
        "verdicts": [
            {
                "verdict": "Pass",
                "color": "#22c55e",
                "criteria": "No channel failures, minimal warnings, artifact burden < 20%",
            },
            {
                "verdict": "Needs Attention",
                "color": "#f59e0b",
                "criteria": "1+ channel failure or 3+ warnings or artifact burden 20-40%",
            },
            {
                "verdict": "Re-record",
                "color": "#ef4444",
                "criteria": "3+ channel failures or artifact burden > 40%",
            },
        ],
        "clinical_reference": "ACNS Guidelines for Standard Electrode Position Nomenclature; IFCN Recommendations for EEG Recording Standards.",
    }
