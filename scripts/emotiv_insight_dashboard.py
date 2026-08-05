#!/usr/bin/env python3
"""Emotiv Insight Dashboard — data module.

Provides overview(), sessions(), and definitions() for the
/api/emotiv-insight/* endpoints.  Models the Emotiv Insight 2+
5-channel consumer-research EEG headset (online BLE + offline buffer modes).
Realistic synthetic data — no PHI.
"""
from __future__ import annotations
import random
from datetime import datetime, timedelta

RNG = random.Random(42)

# Emotiv Insight 2+ channels (10-20 system)
CHANNELS = ["AF3", "AF4", "T7", "T8", "Pz"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_RANGES = {
    "delta": "0.5–4 Hz",
    "theta": "4–8 Hz",
    "alpha": "8–13 Hz",
    "beta":  "13–30 Hz",
    "gamma": "30–100 Hz",
}

PATIENT_IDS = [f"PT-{100 + i}" for i in range(20)]


# ── helpers ─────────────────────────────────────────────────────────────────

def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


def _hours_ago(h: int) -> str:
    return (datetime.utcnow() - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _contact_quality() -> str:
    return RNG.choice(["good", "good", "good", "marginal", "poor"])


def _band_power() -> dict:
    raw = {b: max(0.01, RNG.gauss(0.20, 0.07)) for b in BANDS}
    total = sum(raw.values())
    return {b: round(v / total, 3) for b, v in raw.items()}


def _battery() -> int:
    return RNG.randint(12, 100)


def _sync_status() -> str:
    return RNG.choices(["synced", "pending", "error"], weights=[70, 25, 5])[0]


def _seizure_risk() -> str:
    return RNG.choices(["none", "low", "moderate", "high"], weights=[60, 25, 10, 5])[0]


def _motion() -> dict:
    return {
        "accel_x": round(RNG.gauss(0, 0.3), 3),
        "accel_y": round(RNG.gauss(0, 0.3), 3),
        "accel_z": round(RNG.gauss(9.8, 0.2), 3),
        "gyro_pitch": round(RNG.gauss(0, 5), 1),
        "gyro_roll":  round(RNG.gauss(0, 5), 1),
        "motion_artifact": RNG.random() > 0.8,
    }


def _make_device(i: int) -> dict:
    battery = _battery()
    ch_quality = {ch: _contact_quality() for ch in CHANNELS}
    good_ch = sum(1 for v in ch_quality.values() if v == "good")
    bp = _band_power()
    risk = _seizure_risk()
    sync = _sync_status()
    return {
        "device_id": f"INS-{2000 + i}",
        "patient_id": PATIENT_IDS[i % len(PATIENT_IDS)],
        "firmware": f"2.{RNG.randint(1, 5)}.{RNG.randint(0, 9)}",
        "battery_pct": battery,
        "battery_status": "ok" if battery >= 30 else "low",
        "sync_status": sync,
        "channels_good": good_ch,
        "channel_quality": ch_quality,
        "band_power": bp,
        "dominant_band": max(bp, key=bp.get),
        "alpha_asymmetry": round(RNG.gauss(0.05, 0.15), 3),
        "seizure_risk": risk,
        "motion": _motion(),
        "last_seen": _hours_ago(RNG.randint(0, 48)),
        "mode": RNG.choice(["online", "offline"]),
    }


def _make_session(i: int) -> dict:
    duration = RNG.randint(10, 60)
    ch_quality = {ch: _contact_quality() for ch in CHANNELS}
    good_ch = sum(1 for v in ch_quality.values() if v == "good")
    return {
        "session_id": f"INS-S{4000 + i}",
        "device_id": f"INS-{2000 + (i % 20)}",
        "patient_id": PATIENT_IDS[i % len(PATIENT_IDS)],
        "date": _days_ago(RNG.randint(0, 30)),
        "duration_min": duration,
        "channels_good": good_ch,
        "avg_band_power": _band_power(),
        "seizure_risk_peak": _seizure_risk(),
        "motion_artifacts": RNG.randint(0, 12),
        "data_quality_pct": RNG.randint(72, 100),
        "mode": RNG.choice(["online", "offline"]),
        "uploaded": RNG.random() > 0.1,
    }


# ── public API ───────────────────────────────────────────────────────────────

def overview() -> dict:
    RNG.seed(42)
    devices = [_make_device(i) for i in range(20)]

    total = len(devices)
    synced = sum(1 for d in devices if d["sync_status"] == "synced")
    pending = sum(1 for d in devices if d["sync_status"] == "pending")
    errors = sum(1 for d in devices if d["sync_status"] == "error")
    avg_battery = round(sum(d["battery_pct"] for d in devices) / total, 1)
    low_battery = sum(1 for d in devices if d["battery_pct"] < 30)
    seizure_alerts = sum(1 for d in devices if d["seizure_risk"] in ("moderate", "high"))
    avg_good_ch = round(sum(d["channels_good"] for d in devices) / total, 1)
    online = sum(1 for d in devices if d["mode"] == "online")

    # Band power trend over 7 days
    band_trend_7d = []
    for day in range(6, -1, -1):
        bp = _band_power()
        band_trend_7d.append({"date": _days_ago(day), **bp})

    # Dominant band distribution
    from collections import Counter
    dom_counts = Counter(d["dominant_band"] for d in devices)
    dominant_band_dist = [{"band": b, "count": c} for b, c in sorted(dom_counts.items())]

    # Alpha asymmetry distribution (frontal EEG marker)
    asymmetries = [d["alpha_asymmetry"] for d in devices]
    aa_dist = [
        {"range": "< -0.2 (right-dominant)", "count": sum(1 for a in asymmetries if a < -0.2)},
        {"range": "-0.2–0 (slight right)",    "count": sum(1 for a in asymmetries if -0.2 <= a < 0)},
        {"range": "0–0.2 (slight left)",       "count": sum(1 for a in asymmetries if 0 <= a < 0.2)},
        {"range": "> 0.2 (left-dominant)",     "count": sum(1 for a in asymmetries if a >= 0.2)},
    ]

    # Seizure risk summary
    risk_summary = {}
    for d in devices:
        risk_summary[d["seizure_risk"]] = risk_summary.get(d["seizure_risk"], 0) + 1

    return {
        "kpis": {
            "total_devices": total,
            "synced": synced,
            "sync_pending": pending,
            "sync_errors": errors,
            "avg_battery_pct": avg_battery,
            "low_battery_count": low_battery,
            "online_devices": online,
            "avg_good_channels": avg_good_ch,
            "total_channels": 5,
            "seizure_risk_alerts": seizure_alerts,
        },
        "band_trend_7d": band_trend_7d,
        "dominant_band_distribution": dominant_band_dist,
        "alpha_asymmetry_distribution": aa_dist,
        "seizure_risk_summary": [
            {"risk": k, "count": v} for k, v in sorted(risk_summary.items())
        ],
    }


def sessions() -> dict:
    RNG.seed(42)
    devices = [_make_device(i) for i in range(20)]
    session_list = [_make_session(i) for i in range(40)]

    # Alert events from high/moderate risk devices
    alert_events = []
    for d in devices:
        if d["seizure_risk"] in ("moderate", "high"):
            alert_events.append({
                "event_id": f"ALERT-INS-{len(alert_events) + 1:04d}",
                "device_id": d["device_id"],
                "patient_id": d["patient_id"],
                "risk_level": d["seizure_risk"],
                "dominant_band": d["dominant_band"],
                "alpha_asymmetry": d["alpha_asymmetry"],
                "battery_pct": d["battery_pct"],
                "motion_artifact": d["motion"]["motion_artifact"],
                "timestamp": d["last_seen"],
                "resolved": RNG.random() > 0.4,
            })

    return {
        "all_devices": devices,
        "sessions": session_list,
        "alert_events": alert_events,
    }


def definitions() -> dict:
    return {
        "device_overview": {
            "name": "Emotiv Insight 2+",
            "channels": 5,
            "channel_positions": ["AF3", "AF4", "T7", "T8", "Pz"],
            "sampling_rate": "128 Hz (raw EEG)",
            "connectivity": "BLE 5.0",
            "battery_life": "~6 hours",
            "data": ["raw EEG", "band power", "motion (accelerometer + gyroscope)"],
            "modes": ["online (BLE → mobile/gateway → backend)", "offline (on-device buffer → sync on pairing)"],
            "use_case": "Epilepsy seizure-risk monitoring, ambulatory EEG, cognitive workload",
            "edge_model": "On-device seizure-risk classifier (TensorFlow Lite)",
        },
        "band_definitions": [
            {"band": b, "range": BAND_RANGES[b],
             "significance": {
                 "delta":  "Deep sleep, pathological slow waves (post-ictal slowing)",
                 "theta":  "Drowsiness, memory, mesial temporal lobe activity",
                 "alpha":  "Relaxed wakefulness; asymmetry → frontal activation marker",
                 "beta":   "Active cognition, anxiety; often suppressed post-seizure",
                 "gamma":  "High-frequency binding; HFO marker in epileptic zones",
             }[b]}
            for b in BANDS
        ],
        "channel_definitions": [
            {"channel": "AF3", "region": "Left prefrontal",  "role": "Frontal alpha asymmetry (L)"},
            {"channel": "AF4", "region": "Right prefrontal", "role": "Frontal alpha asymmetry (R)"},
            {"channel": "T7",  "region": "Left temporal",    "role": "Temporal lobe EEG (hippocampus)"},
            {"channel": "T8",  "region": "Right temporal",   "role": "Temporal lobe EEG (hippocampus)"},
            {"channel": "Pz",  "region": "Parietal midline", "role": "Generalized spike-wave reference"},
        ],
        "alpha_asymmetry": {
            "formula": "ln(AF4 alpha power) − ln(AF3 alpha power)",
            "positive": "Left frontal dominance (approach motivation; may reduce seizure threshold)",
            "negative": "Right frontal dominance (withdrawal/depression; seen in some TLE patients)",
            "clinical_note": "Asymmetry > ±0.2 flagged for neurologist review",
        },
        "seizure_risk_levels": [
            {"level": "none",     "description": "No EEG anomaly detected by edge model"},
            {"level": "low",      "description": "Minor spectral shift; monitor"},
            {"level": "moderate", "description": "Theta/delta surge or asymmetry spike; caregiver alert"},
            {"level": "high",     "description": "High-frequency burst or generalized slowing; immediate escalation"},
        ],
        "contact_quality": [
            {"quality": "good",     "description": "Impedance < 10 kΩ — reliable signal"},
            {"quality": "marginal", "description": "Impedance 10–25 kΩ — usable but refit recommended"},
            {"quality": "poor",     "description": "Impedance > 25 kΩ — re-seat electrode before recording"},
        ],
        "sync_modes": [
            {"mode": "online",  "description": "BLE stream → mobile app → backend in real-time; latency < 500 ms"},
            {"mode": "offline", "description": "On-device circular buffer (up to 4 h); auto-upload on BLE reconnect"},
        ],
    }
