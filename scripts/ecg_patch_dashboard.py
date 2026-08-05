#!/usr/bin/env python3
"""ECG Patch Dashboard — data module.

Provides overview(), sessions(), and definitions() for the /api/ecg-patch/* endpoints.
Models an offline-first ECG patch (Holter-style wearable that records continuously
and uploads at clinic visit). Realistic synthetic data — no live streaming required.
"""
from __future__ import annotations
import random, math
from datetime import datetime, timedelta

RNG = random.Random(42)

# ── helpers ────────────────────────────────────────────────────────────────────

def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


def _hr_series(length: int = 24, base: int = 72) -> list[dict]:
    """Simulated hourly heart-rate series."""
    out = []
    for h in range(length):
        noise = RNG.gauss(0, 4)
        # dip during sleep hours 23-06
        sleep_dip = -10 if (h >= 23 or h <= 6) else 0
        val = max(45, min(130, base + noise + sleep_dip))
        out.append({"hour": h, "hr": round(val, 1)})
    return out


def _hrv_trend(length: int = 7) -> list[dict]:
    """SDNN trend over 7 days."""
    out = []
    base = RNG.uniform(30, 55)
    for d in range(length):
        val = max(15, base + RNG.gauss(0, 5))
        out.append({"day": _days_ago(length - 1 - d), "sdnn_ms": round(val, 1)})
    return out


def _event_types() -> list[dict]:
    events = [
        {"type": "Normal Sinus Rhythm", "count": RNG.randint(8000, 12000), "color": "#10b981"},
        {"type": "Premature Atrial Contraction", "count": RNG.randint(30, 200), "color": "#f59e0b"},
        {"type": "Premature Ventricular Contraction", "count": RNG.randint(10, 80), "color": "#ef4444"},
        {"type": "ST Depression", "count": RNG.randint(0, 15), "color": "#8b5cf6"},
        {"type": "ST Elevation", "count": RNG.randint(0, 5), "color": "#ec4899"},
        {"type": "Motion Artefact", "count": RNG.randint(50, 300), "color": "#94a3b8"},
    ]
    return events


# ── public API ─────────────────────────────────────────────────────────────────

def overview() -> dict:
    """ECG Patch fleet overview: KPIs, fleet status, HR trend, HRV trend."""
    patches = [
        {
            "patch_id": f"ECP-{1000 + i}",
            "patient": f"P-{100 + i}",
            "duration_days": RNG.randint(1, 14),
            "battery_pct": RNG.randint(12, 98),
            "upload_status": RNG.choice(["uploaded", "uploaded", "uploaded", "pending", "error"]),
            "lead": RNG.choice(["Lead I", "Lead II", "Modified Lead V5"]),
            "total_beats": RNG.randint(80000, 180000),
        }
        for i in range(12)
    ]

    active = sum(1 for p in patches if p["upload_status"] == "pending")
    uploaded = sum(1 for p in patches if p["upload_status"] == "uploaded")
    errors = sum(1 for p in patches if p["upload_status"] == "error")
    avg_battery = round(sum(p["battery_pct"] for p in patches) / len(patches), 1)
    total_beats = sum(p["total_beats"] for p in patches)

    return {
        "kpis": {
            "total_patches": len(patches),
            "active_recording": active,
            "uploaded": uploaded,
            "upload_errors": errors,
            "avg_battery_pct": avg_battery,
            "total_beats_analyzed": total_beats,
        },
        "fleet": patches,
        "hr_trend_24h": _hr_series(24, base=72),
        "hrv_trend_7d": _hrv_trend(7),
        "event_distribution": _event_types(),
    }


def sessions() -> dict:
    """Per-patient ECG recording sessions: waveform stats, events, upload log."""
    sessions_data = []
    for i in range(10):
        pid = f"P-{100 + i}"
        duration_days = RNG.randint(1, 14)
        total_beats = RNG.randint(80000, max(80000, duration_days * 14000))
        events = _event_types()
        arrhythmia_count = sum(
            e["count"] for e in events if e["type"] not in ("Normal Sinus Rhythm", "Motion Artefact")
        )
        sessions_data.append({
            "session_id": f"S-{2000 + i}",
            "patient_id": pid,
            "patch_id": f"ECP-{1000 + i}",
            "recording_start": _days_ago(duration_days + 1),
            "recording_end": _days_ago(1),
            "duration_days": duration_days,
            "lead": RNG.choice(["Lead II", "Modified Lead V5", "Lead I"]),
            "total_beats": total_beats,
            "mean_hr_bpm": round(RNG.uniform(58, 88), 1),
            "min_hr_bpm": round(RNG.uniform(42, 58), 1),
            "max_hr_bpm": round(RNG.uniform(110, 160), 1),
            "sdnn_ms": round(RNG.uniform(18, 70), 1),
            "rmssd_ms": round(RNG.uniform(15, 60), 1),
            "pnn50_pct": round(RNG.uniform(5, 35), 1),
            "arrhythmia_events": arrhythmia_count,
            "qtc_ms": round(RNG.uniform(380, 470), 1),
            "st_changes": RNG.randint(0, 12),
            "motion_artefact_pct": round(RNG.uniform(1, 8), 1),
            "signal_quality_pct": round(RNG.uniform(88, 99), 1),
            "upload_status": RNG.choice(["uploaded", "uploaded", "pending"]),
            "events": events,
            "hr_series": _hr_series(min(duration_days * 24, 48), base=int(RNG.uniform(62, 82))),
        })

    return {
        "sessions": sessions_data,
        "summary": {
            "total_sessions": len(sessions_data),
            "avg_duration_days": round(
                sum(s["duration_days"] for s in sessions_data) / len(sessions_data), 1
            ),
            "avg_signal_quality_pct": round(
                sum(s["signal_quality_pct"] for s in sessions_data) / len(sessions_data), 1
            ),
            "avg_arrhythmia_events": round(
                sum(s["arrhythmia_events"] for s in sessions_data) / len(sessions_data), 0
            ),
        },
    }


def definitions() -> dict:
    """Clinical/technical glossary for ECG patch terms."""
    return {
        "device": {
            "name": "ECG Patch",
            "type": "Wearable adhesive cardiac monitor",
            "recording_mode": "Offline — continuous recording, upload at clinic",
            "leads": ["Lead I", "Lead II", "Modified Lead V5"],
            "sampling_rate_hz": 256,
            "resolution_bits": 16,
            "storage_capacity_days": 14,
            "battery_life_days": 14,
            "waterproof": "IPX4",
            "fda_cleared": "510(k) class II equivalent",
        },
        "metrics": [
            {
                "term": "HR (Heart Rate)",
                "unit": "bpm",
                "normal_range": "60–100",
                "clinical_note": "Resting sinus rate. Tachycardia >100, bradycardia <60.",
            },
            {
                "term": "SDNN",
                "unit": "ms",
                "normal_range": ">50 ms (24 h)",
                "clinical_note": "Standard deviation of NN intervals — overall HRV. Lower SDNN associated with increased cardiac risk.",
            },
            {
                "term": "RMSSD",
                "unit": "ms",
                "normal_range": "19–75 ms",
                "clinical_note": "Root mean square of successive NN differences — parasympathetic activity marker.",
            },
            {
                "term": "pNN50",
                "unit": "%",
                "normal_range": ">3%",
                "clinical_note": "Percentage of successive NN differences >50 ms. Parasympathetic index.",
            },
            {
                "term": "QTc",
                "unit": "ms",
                "normal_range": "<440 ms (M), <460 ms (F)",
                "clinical_note": "Corrected QT interval. Prolonged QTc = risk of Torsades de Pointes.",
            },
            {
                "term": "ST Depression",
                "unit": "mm",
                "normal_range": "<0.5 mm",
                "clinical_note": "Possible ischaemia or left bundle branch block. >1 mm significant.",
            },
            {
                "term": "ST Elevation",
                "unit": "mm",
                "normal_range": "<1 mm",
                "clinical_note": ">2 mm in ≥2 contiguous leads = STEMI criteria.",
            },
            {
                "term": "PAC",
                "unit": "count/24 h",
                "normal_range": "<100",
                "clinical_note": "Premature Atrial Contraction. Frequent PACs may predict AF.",
            },
            {
                "term": "PVC",
                "unit": "count/24 h",
                "normal_range": "<200",
                "clinical_note": "Premature Ventricular Contraction. >10k/24 h may warrant ablation.",
            },
        ],
        "epilepsy_context": (
            "ECG patches are co-deployed with EEG to capture ictal tachycardia, "
            "SUDEP-risk bradycardia, and peri-ictal autonomic dysregulation. "
            "Combined EEG+ECG improves seizure detection specificity and aids "
            "SUDEP risk stratification."
        ),
        "references": [
            "Hesdorffer DC et al. Combined EEG and ECG — J Clin Neurophysiol 2012.",
            "Devinsky O et al. SUDEP. Nat Rev Neurol 2016.",
            "Zipes DP et al. ACC/AHA/ESC Guidelines — JACC 2006.",
        ],
    }
