#!/usr/bin/env python3
"""Emotiv EPOC X Dashboard — data module.

Provides overview(), breakdown(), and definitions() for the
/api/emotiv-epoc-x/* endpoints.  Models the Emotiv EPOC X 14-channel
research-grade EEG headset (BLE + saline felt pads).
Realistic synthetic data — no PHI.
"""
from __future__ import annotations
import random
from datetime import datetime, timedelta

RNG = random.Random(13)

# Emotiv EPOC X 14 channels (10-20 system)
CHANNELS = [
    "AF3", "F7", "F3", "FC5",
    "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4",
    "F8", "AF4",
]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_RANGES = {
    "delta": "0.5–4 Hz",
    "theta": "4–8 Hz",
    "alpha": "8–13 Hz",
    "beta":  "13–30 Hz",
    "gamma": "30–100 Hz",
}

PATIENT_IDS = [f"PT-{200 + i}" for i in range(25)]
LOCATIONS = ["EMU", "Lab A", "Lab B", "ICU", "Outpatient"]
SEIZURE_TYPES = ["focal", "generalised", "absence", "tonic-clonic", "unknown"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


def _hours_ago(h: int) -> str:
    return (datetime.utcnow() - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _impedance() -> float:
    """Electrode impedance kΩ — saline pads achieve lower impedance than dry."""
    if RNG.random() < 0.78:
        return round(RNG.uniform(1.5, 8.0), 1)   # good
    elif RNG.random() < 0.55:
        return round(RNG.uniform(8.0, 20.0), 1)   # marginal
    else:
        return round(RNG.uniform(20.0, 60.0), 1)  # poor


def _contact_quality(imp: float) -> str:
    if imp < 10:
        return "good"
    elif imp < 25:
        return "marginal"
    return "poor"


def _band_power() -> dict:
    raw = {b: max(0.01, RNG.gauss(0.20, 0.06)) for b in BANDS}
    total = sum(raw.values())
    return {b: round(v / total, 3) for b, v in raw.items()}


def _battery() -> int:
    return RNG.randint(15, 100)


def _make_device(i: int) -> dict:
    battery = _battery()
    imp_map = {ch: _impedance() for ch in CHANNELS}
    cq_map = {ch: _contact_quality(imp_map[ch]) for ch in CHANNELS}
    good_count = sum(1 for q in cq_map.values() if q == "good")
    bp = _band_power()
    dominant = max(bp, key=bp.get)
    alpha_asym = round(bp["alpha"] * (1 if RNG.random() > 0.5 else -1) * RNG.uniform(0.05, 0.25), 3)
    sync = RNG.choices(["synced", "pending", "error"], weights=[0.75, 0.18, 0.07])[0]
    risk = RNG.choices(["none", "low", "moderate", "high"], weights=[0.55, 0.25, 0.15, 0.05])[0]
    return {
        "device_id": f"EPOCX-{4000 + i}",
        "patient_id": PATIENT_IDS[i % len(PATIENT_IDS)],
        "battery_pct": battery,
        "location": RNG.choice(LOCATIONS),
        "connectivity": RNG.choice(["BLE 4.2", "BLE 5.0"]),
        "status": "online" if RNG.random() > 0.25 else "offline",
        "last_sync": _hours_ago(RNG.randint(0, 12)),
        "sync_status": sync,
        "channels_good": good_count,
        "channels_total": len(CHANNELS),
        "impedance_kOhm": imp_map,
        "channel_quality": cq_map,
        "band_power": bp,
        "dominant_band": dominant,
        "alpha_asymmetry": alpha_asym,
        "seizure_risk": risk,
        "pad_condition": RNG.choices(["fresh", "used", "dry"], weights=[0.5, 0.38, 0.12])[0],
        "firmware": RNG.choice(["3.7.0", "3.8.2", "4.0.1"]),
    }


DEVICES = [_make_device(i) for i in range(25)]


def _make_session(i: int) -> dict:
    dev = DEVICES[i % len(DEVICES)]
    duration = RNG.randint(20, 120)
    ch_good = RNG.randint(10, 14)
    return {
        "session_id": f"EPOCX-S{5000 + i}",
        "device_id": dev["device_id"],
        "patient_id": dev["patient_id"],
        "date": _days_ago(RNG.randint(0, 30)),
        "duration_min": duration,
        "channels_good": ch_good,
        "seizure_type": RNG.choice(SEIZURE_TYPES),
        "seizure_risk_peak": RNG.choices(
            ["none", "low", "moderate", "high"], weights=[0.45, 0.30, 0.18, 0.07]
        )[0],
        "motion_artifacts": RNG.randint(0, 8),
        "data_quality_pct": RNG.randint(72, 100),
        "pad_condition": RNG.choices(["fresh", "used", "dry"], weights=[0.5, 0.38, 0.12])[0],
        "uploaded": RNG.random() > 0.15,
        "location": RNG.choice(LOCATIONS),
    }


SESSIONS = [_make_session(i) for i in range(50)]


def _make_alert(i: int) -> dict:
    dev = DEVICES[i % len(DEVICES)]
    bp = _band_power()
    return {
        "event_id": f"EVT-X{6000 + i}",
        "device_id": dev["device_id"],
        "patient_id": dev["patient_id"],
        "risk_level": RNG.choices(["moderate", "high"], weights=[0.65, 0.35])[0],
        "dominant_band": max(bp, key=bp.get),
        "alpha_asymmetry": round(RNG.uniform(-0.25, 0.25), 3),
        "battery_pct": RNG.randint(15, 85),
        "motion_artifact": RNG.random() > 0.6,
        "timestamp": _hours_ago(RNG.randint(1, 72)),
        "resolved": RNG.random() > 0.35,
        "pad_condition": RNG.choice(["fresh", "used", "dry"]),
    }


ALERTS = [_make_alert(i) for i in range(18)]


# ── public functions ───────────────────────────────────────────────────────────

def overview() -> dict:
    """Fleet KPIs, 7-day band-power trend, dominant band distribution, seizure risk."""
    total = len(DEVICES)
    synced = sum(1 for d in DEVICES if d["sync_status"] == "synced")
    pending = sum(1 for d in DEVICES if d["sync_status"] == "pending")
    errors = sum(1 for d in DEVICES if d["sync_status"] == "error")
    avg_batt = round(sum(d["battery_pct"] for d in DEVICES) / total, 1)
    low_batt = sum(1 for d in DEVICES if d["battery_pct"] < 20)
    online = sum(1 for d in DEVICES if d["status"] == "online")
    avg_good = round(sum(d["channels_good"] for d in DEVICES) / total, 1)
    risk_alerts = sum(1 for d in DEVICES if d["seizure_risk"] in ["moderate", "high"])
    fresh_pads = sum(1 for d in DEVICES if d["pad_condition"] == "fresh")
    dry_pads = sum(1 for d in DEVICES if d["pad_condition"] == "dry")

    # 7-day band power trend
    band_trend = []
    for day in range(6, -1, -1):
        rng2 = random.Random(day * 7 + 42)
        raw = {b: max(0.01, rng2.gauss(0.20, 0.04)) for b in BANDS}
        total_bp = sum(raw.values())
        band_trend.append({
            "date": _days_ago(day),
            **{b: round(v / total_bp, 3) for b, v in raw.items()},
        })

    # Dominant band distribution
    dom_counts: dict[str, int] = {}
    for d in DEVICES:
        dom_counts[d["dominant_band"]] = dom_counts.get(d["dominant_band"], 0) + 1
    dom_dist = [{"band": b, "count": dom_counts.get(b, 0)} for b in BANDS]

    # Seizure risk summary
    risk_dist: dict[str, int] = {}
    for d in DEVICES:
        risk_dist[d["seizure_risk"]] = risk_dist.get(d["seizure_risk"], 0) + 1
    risk_summary = [{"risk": r, "count": risk_dist.get(r, 0)} for r in ["none", "low", "moderate", "high"]]

    # Pad condition breakdown
    pad_counts: dict[str, int] = {}
    for d in DEVICES:
        pad_counts[d["pad_condition"]] = pad_counts.get(d["pad_condition"], 0) + 1
    pad_dist = [{"condition": c, "count": pad_counts.get(c, 0)} for c in ["fresh", "used", "dry"]]

    # Location breakdown
    loc_counts: dict[str, int] = {}
    for d in DEVICES:
        loc_counts[d["location"]] = loc_counts.get(d["location"], 0) + 1
    loc_dist = [{"location": loc, "count": cnt} for loc, cnt in sorted(loc_counts.items())]

    return {
        "kpis": {
            "total_devices": total,
            "synced": synced,
            "sync_pending": pending,
            "sync_errors": errors,
            "avg_battery_pct": avg_batt,
            "low_battery_count": low_batt,
            "online_devices": online,
            "avg_good_channels": avg_good,
            "channels_per_device": len(CHANNELS),
            "seizure_risk_alerts": risk_alerts,
            "fresh_pads": fresh_pads,
            "dry_pads": dry_pads,
        },
        "band_trend_7d": band_trend,
        "dominant_band_distribution": dom_dist,
        "seizure_risk_summary": risk_summary,
        "pad_condition_distribution": pad_dist,
        "location_distribution": loc_dist,
    }


def breakdown() -> dict:
    """Per-device inventory, session log, seizure-risk alerts."""
    return {
        "all_devices": DEVICES,
        "sessions": SESSIONS,
        "alert_events": ALERTS,
    }


def definitions() -> dict:
    """Device specs, 14-channel map, band glossary, impedance grades, risk levels."""
    return {
        "device_overview": {
            "model": "Emotiv EPOC X",
            "channels": 14,
            "references": "CMS/DRL (P3/P4 positions)",
            "sampling_rate": "256 Hz (up to 2048 Hz via TestBench)",
            "resolution": "14-bit ADC, 0.51 µV LSB",
            "bandwidth": "0.16–43 Hz (3rd order Sinc filter)",
            "connectivity": "BLE 4.2 / BLE 5.0",
            "electrode_type": "Saline-soaked felt pads",
            "battery_life": "~6 hours",
            "weight": "310 g",
            "use_case": "Clinical research, epilepsy monitoring, BCI studies",
        },
        "channel_definitions": [
            {"channel": "AF3", "region": "Anterior Frontal (Left)",  "role": "Frontal asymmetry, emotional processing"},
            {"channel": "F7",  "region": "Frontal (Left)",           "role": "Language production (Broca's area proximity)"},
            {"channel": "F3",  "region": "Frontal (Left)",           "role": "Motor planning, executive function"},
            {"channel": "FC5", "region": "Fronto-Central (Left)",    "role": "Motor-frontal junction, seizure onset detection"},
            {"channel": "T7",  "region": "Temporal (Left)",          "role": "Auditory processing, hippocampal projection"},
            {"channel": "P7",  "region": "Parietal (Left)",          "role": "Somatosensory, spatial processing"},
            {"channel": "O1",  "region": "Occipital (Left)",         "role": "Visual cortex — photo-paroxysmal response"},
            {"channel": "O2",  "region": "Occipital (Right)",        "role": "Visual cortex — photo-paroxysmal response"},
            {"channel": "P8",  "region": "Parietal (Right)",         "role": "Somatosensory, spatial processing"},
            {"channel": "T8",  "region": "Temporal (Right)",         "role": "Emotional memory, amygdala proximity"},
            {"channel": "FC6", "region": "Fronto-Central (Right)",   "role": "Motor-frontal junction, seizure spread"},
            {"channel": "F4",  "region": "Frontal (Right)",          "role": "Working memory, inhibitory control"},
            {"channel": "F8",  "region": "Frontal (Right)",          "role": "Orbitofrontal cortex proximity"},
            {"channel": "AF4", "region": "Anterior Frontal (Right)", "role": "Frontal asymmetry, mood regulation"},
        ],
        "band_definitions": [
            {"band": "delta", "range": "0.5–4 Hz",  "significance": "Deep sleep, severe encephalopathy; inter-ictal slowing"},
            {"band": "theta", "range": "4–8 Hz",    "significance": "Drowsiness, memory encoding; focal seizure precursor"},
            {"band": "alpha", "range": "8–13 Hz",   "significance": "Relaxed wakefulness; suppressed post-ictally"},
            {"band": "beta",  "range": "13–30 Hz",  "significance": "Active cognition; prominent with benzodiazepines"},
            {"band": "gamma", "range": "30–100 Hz", "significance": "High-frequency oscillations; HFO marker for seizure onset"},
        ],
        "impedance_grades": [
            {"grade": "good",     "range_kOhm": "< 10",    "description": "Optimal signal quality — saline pad well saturated"},
            {"grade": "marginal", "range_kOhm": "10–25",   "description": "Acceptable — pad may need re-saturation"},
            {"grade": "poor",     "range_kOhm": "> 25",    "description": "High noise — re-seat electrode or refresh pad"},
        ],
        "seizure_risk_levels": [
            {"level": "none",     "description": "No epileptiform features detected in current window"},
            {"level": "low",      "description": "Mild theta excess or alpha suppression — monitor"},
            {"level": "moderate", "description": "Focal slowing or HFO-like bursts — escalate review"},
            {"level": "high",     "description": "Suspected ictal pattern — notify clinician immediately"},
        ],
        "pad_conditions": [
            {"condition": "fresh", "description": "Pad freshly saline-soaked — optimal impedance"},
            {"condition": "used",  "description": "Mid-session — impedance may drift, monitor"},
            {"condition": "dry",   "description": "Pad dried out — must refresh before continuing"},
        ],
        "sync_modes": [
            {"mode": "synced",  "description": "Session data uploaded to platform in real-time or post-session"},
            {"mode": "pending", "description": "Data queued locally; upload in progress"},
            {"mode": "error",   "description": "Upload failed — manual retry or gateway check required"},
        ],
        "alpha_asymmetry": {
            "formula": "AAI = ln(alpha_AF4) − ln(alpha_AF3)",
            "positive_value": "Right-hemisphere alpha dominance — left-hemisphere activation",
            "negative_value": "Left-hemisphere alpha dominance — right-hemisphere activation",
            "epilepsy_relevance": "Interictal asymmetry can lateralise seizure focus (esp. temporal lobe)",
        },
        "clinical_note": (
            "The EPOC X is validated for ambulatory and supervised clinical research. "
            "14-channel coverage spans temporal, frontal, parietal, and occipital lobes — "
            "adequate for broad epilepsy surveillance. Impedance < 10 kΩ mandatory for "
            "inter-ictal spike detection. Not cleared as a standalone diagnostic medical device."
        ),
    }
