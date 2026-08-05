#!/usr/bin/env python3
"""Smartwatch (Apple Watch / Wear OS) Dashboard — data module.

Provides overview(), breakdown(), and definitions() for the
/api/smartwatch/* endpoints.  Models a smartwatch fleet monitoring
heart rate, HRV, SpO2, accelerometer (fall/movement), and sleep staging
for epilepsy patients.  Online sync via phone app; offline buffered on watch.
Realistic synthetic data — no live streaming required.
"""
from __future__ import annotations
import random, math
from datetime import datetime, timedelta

RNG = random.Random(99)

PATIENT_IDS = [f"P-{200 + i}" for i in range(20)]

WATCH_MODELS = [
    "Apple Watch Series 9",
    "Apple Watch Ultra 2",
    "Apple Watch SE (2nd gen)",
    "Samsung Galaxy Watch 6",
    "Google Pixel Watch 2",
    "Fitbit Sense 2",
    "Garmin Venu 3",
]

SYNC_STATUSES = ["synced", "synced", "synced", "pending", "error"]
SEIZURE_SIGNALS = ["none", "none", "none", "hr_spike", "movement_burst", "hr_spike"]

# ── helpers ────────────────────────────────────────────────────────────────────

def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


def _hours_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(hours=n)).strftime("%Y-%m-%dT%H:%M:00Z")


def _hr_series(length: int = 24, base: int = 68) -> list[dict]:
    """Simulated hourly heart-rate series over 24 h."""
    out = []
    for h in range(length):
        noise = RNG.gauss(0, 5)
        sleep_dip = -12 if (h >= 23 or h <= 6) else 0
        spike = 30 if RNG.random() < 0.04 else 0   # 4 % chance of tachycardia
        val = max(40, min(145, base + noise + sleep_dip + spike))
        out.append({"hour": h, "hr_bpm": round(val, 1)})
    return out


def _hrv_trend(days: int = 7) -> list[dict]:
    """SDNN trend — lower HRV around seizure days."""
    base = RNG.uniform(28, 52)
    out = []
    for d in range(days):
        pre_ictal_dip = -8 if d == 5 else 0    # simulate pre-ictal dip day 5
        val = max(12, base + RNG.gauss(0, 4) + pre_ictal_dip)
        out.append({"day": _days_ago(days - 1 - d), "sdnn_ms": round(val, 1)})
    return out


def _spo2_distribution() -> list[dict]:
    """SpO2 bucket distribution."""
    buckets = [
        {"range": "≥ 95 % (normal)", "count": RNG.randint(14, 19)},
        {"range": "92–94 % (mild hypoxia)", "count": RNG.randint(1, 4)},
        {"range": "88–91 % (moderate)", "count": RNG.randint(0, 2)},
        {"range": "< 88 % (severe)", "count": RNG.randint(0, 1)},
    ]
    return buckets


def _sleep_stages() -> dict:
    """Average nightly sleep stage distribution (hours)."""
    total = round(RNG.uniform(5.5, 7.8), 1)
    rem = round(total * RNG.uniform(0.18, 0.24), 1)
    deep = round(total * RNG.uniform(0.12, 0.20), 1)
    light = round(total - rem - deep, 1)
    return {"total_h": total, "rem_h": rem, "deep_h": deep, "light_h": light}


def _make_device(i: int) -> dict:
    pid = PATIENT_IDS[i % len(PATIENT_IDS)]
    model = RNG.choice(WATCH_MODELS)
    battery = RNG.randint(8, 98)
    sync_status = RNG.choice(SYNC_STATUSES)
    signal = RNG.choice(SEIZURE_SIGNALS)
    hr_now = RNG.randint(55, 110)
    spo2_now = round(RNG.uniform(92.0, 99.5), 1)
    hrv_sdnn = round(RNG.uniform(18, 65), 1)
    steps_today = RNG.randint(800, 11000)
    return {
        "device_id": f"SW-{3100 + i}",
        "patient_id": pid,
        "model": model,
        "battery_pct": battery,
        "sync_status": sync_status,
        "last_sync": _hours_ago(RNG.randint(0, 6)),
        "hr_now": hr_now,
        "spo2_pct": spo2_now,
        "hrv_sdnn_ms": hrv_sdnn,
        "steps_today": steps_today,
        "seizure_signal": signal,
        "seizure_signal_detected": signal != "none",
    }


# ── public API ─────────────────────────────────────────────────────────────────

def overview() -> dict:
    """Fleet overview — KPIs, HR trend, HRV trend, SpO2 distribution, sleep stats."""
    devices = [_make_device(i) for i in range(20)]

    synced = sum(1 for d in devices if d["sync_status"] == "synced")
    pending = sum(1 for d in devices if d["sync_status"] == "pending")
    errors = sum(1 for d in devices if d["sync_status"] == "error")
    avg_battery = round(sum(d["battery_pct"] for d in devices) / len(devices), 1)
    avg_hr = round(sum(d["hr_now"] for d in devices) / len(devices), 1)
    avg_spo2 = round(sum(d["spo2_pct"] for d in devices) / len(devices), 1)
    avg_hrv = round(sum(d["hrv_sdnn_ms"] for d in devices) / len(devices), 1)
    alerts = sum(1 for d in devices if d["seizure_signal_detected"])

    return {
        "kpis": {
            "total_watches": len(devices),
            "synced": synced,
            "sync_pending": pending,
            "sync_errors": errors,
            "avg_battery_pct": avg_battery,
            "avg_hr_bpm": avg_hr,
            "avg_spo2_pct": avg_spo2,
            "avg_hrv_sdnn_ms": avg_hrv,
            "seizure_signal_alerts": alerts,
            "total_steps_today": sum(d["steps_today"] for d in devices),
        },
        "hr_trend_24h": _hr_series(24, base=70),
        "hrv_trend_7d": _hrv_trend(7),
        "spo2_distribution": _spo2_distribution(),
        "sleep_stages_avg": _sleep_stages(),
        "model_distribution": _model_counts(devices),
        "fleet_preview": devices[:6],   # top-6 for dashboard cards
    }


def _model_counts(devices: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for d in devices:
        counts[d["model"]] = counts.get(d["model"], 0) + 1
    return [{"model": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]


def breakdown() -> dict:
    """Per-device inventory table with vitals + sync status."""
    devices = [_make_device(i) for i in range(20)]
    alert_events = [
        {
            "event_id": f"EVT-{5000 + j}",
            "device_id": d["device_id"],
            "patient_id": d["patient_id"],
            "signal_type": d["seizure_signal"],
            "hr_bpm": d["hr_now"],
            "spo2_pct": d["spo2_pct"],
            "timestamp": _hours_ago(RNG.randint(1, 48)),
            "resolved": RNG.random() < 0.6,
        }
        for j, d in enumerate(devices)
        if d["seizure_signal_detected"]
    ]
    return {
        "all_devices": devices,
        "alert_events": alert_events,
        "sync_summary": {
            "synced": sum(1 for d in devices if d["sync_status"] == "synced"),
            "pending": sum(1 for d in devices if d["sync_status"] == "pending"),
            "error": sum(1 for d in devices if d["sync_status"] == "error"),
        },
    }


def definitions() -> dict:
    """Glossary — device specs, metric definitions, epilepsy context."""
    return {
        "device_overview": {
            "name": "Smartwatch (Apple Watch / Wear OS)",
            "type": "Consumer wearable — wrist-worn",
            "data_streams": ["HR (optical PPG)", "HRV (SDNN, RMSSD)", "SpO2 (pulse oximetry)",
                             "Accelerometer (fall, steps, activity)", "Sleep staging"],
            "connectivity": "Phone app → cloud sync (online); watch buffers data → sync when phone connects (offline)",
            "alert_pathway": "HR spike or abnormal movement → possible seizure signal → caregiver/clinician notified",
            "seizure_relevance": "Periictal tachycardia in > 80 % of focal-to-bilateral tonic-clonic seizures; "
                                 "HRV drops pre-ictally; nocturnal seizures detectable via sleep-stage disruption",
        },
        "metrics": [
            {"metric": "HR (BPM)", "normal": "60–100 BPM at rest", "epilepsy_flag": "> 120 BPM sudden spike (ictal tachycardia)"},
            {"metric": "HRV (SDNN, ms)", "normal": "20–80 ms (age-dependent)", "epilepsy_flag": "Sustained drop < 20 ms pre-ictally"},
            {"metric": "SpO2 (%)", "normal": "≥ 95 %", "epilepsy_flag": "< 90 % post-ictal (apnoea risk)"},
            {"metric": "Accelerometer", "normal": "< 0.3 g resting", "epilepsy_flag": "Repetitive high-amplitude burst (clonic phase)"},
            {"metric": "Sleep efficiency", "normal": "≥ 85 %", "epilepsy_flag": "< 70 % with nocturnal seizures"},
            {"metric": "Steps/day", "normal": "5 000–10 000", "epilepsy_flag": "Sudden drop may indicate post-ictal fatigue"},
        ],
        "sync_modes": [
            {"mode": "Online", "description": "Real-time sync to cloud via paired iPhone / Android phone"},
            {"mode": "Offline", "description": "Watch stores up to 72 h of vitals locally; syncs on next phone connection"},
        ],
        "alert_types": [
            {"type": "hr_spike", "trigger": "HR > 120 BPM sustained ≥ 30 s", "action": "Push alert to caregiver app"},
            {"type": "movement_burst", "trigger": "Accelerometer burst > 0.8 g for > 10 s", "action": "Seizure-motion alert"},
            {"type": "spo2_drop", "trigger": "SpO2 < 90 % for > 20 s", "action": "Oxygen-desaturation alert"},
        ],
    }
