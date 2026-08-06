#!/usr/bin/env python3
"""Patient Mobile App Dashboard — data module.

Provides overview(), diary(), and definitions() for the
/api/patient-mobile/* endpoints.  Simulates the patient-facing mobile
application: seizure diary, medication log, symptom tracker, device
pairing status, SOS alerts, and offline-sync queue.
Realistic synthetic data — no live device connection required.
"""
from __future__ import annotations
import random
from datetime import datetime, timedelta

RNG = random.Random(42)

PATIENT_IDS = [f"P-{100 + i}" for i in range(20)]

SEIZURE_TYPES = [
    "Focal aware",
    "Focal impaired awareness",
    "Focal to bilateral tonic-clonic",
    "Generalized tonic-clonic",
    "Absence",
    "Myoclonic",
]

TRIGGERS = [
    "Sleep deprivation",
    "Stress",
    "Missed medication",
    "Alcohol",
    "Photosensitivity",
    "Hormonal",
    "Unknown",
]

MEDICATIONS = [
    "Levetiracetam 500 mg",
    "Lamotrigine 100 mg",
    "Valproate 500 mg",
    "Carbamazepine 200 mg",
    "Oxcarbazepine 300 mg",
    "Lacosamide 100 mg",
    "Topiramate 50 mg",
]

SYMPTOMS = [
    "Aura (visual)",
    "Aura (sensory)",
    "Post-ictal fatigue",
    "Post-ictal confusion",
    "Headache",
    "Memory gap",
    "Nausea",
    "Anxiety",
]

PAIRED_DEVICES = [
    "Emotiv EPOC X",
    "Emotiv Insight 2+",
    "Apple Watch Series 9",
    "Empatica E4",
    "Seizure Band",
    "None",
]

SOS_STATUSES = ["sent", "acknowledged", "escalated", "resolved"]

# ── helpers ────────────────────────────────────────────────────────────────────

def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


def _hours_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(hours=n)).strftime("%Y-%m-%dT%H:%M:00Z")


def _minutes_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(minutes=n)).strftime("%Y-%m-%dT%H:%M:00Z")


def _make_diary_entry(i: int) -> dict:
    pid = PATIENT_IDS[i % len(PATIENT_IDS)]
    days_back = RNG.randint(0, 30)
    return {
        "entry_id": f"DRY-{3000 + i}",
        "patient_id": pid,
        "date": _days_ago(days_back),
        "seizure_type": RNG.choice(SEIZURE_TYPES),
        "duration_seconds": RNG.randint(15, 180),
        "trigger": RNG.choice(TRIGGERS),
        "severity": RNG.choice(["mild", "moderate", "severe"]),
        "witnessed": RNG.random() < 0.55,
        "post_ictal_duration_min": RNG.randint(5, 90),
        "notes": RNG.choice([
            "Occurred after poor sleep night",
            "Woke up feeling off beforehand",
            "No warning signs",
            "Aura preceded by 2 min",
            "",
        ]),
        "recorded_via": RNG.choice(["mobile_app", "caregiver_app", "clinician_entry"]),
    }


def _make_medication_log(pid: str, day_offset: int) -> dict:
    med = RNG.choice(MEDICATIONS)
    scheduled_time = f"{RNG.randint(7,22):02d}:00"
    taken = RNG.random() < 0.88
    late = taken and RNG.random() < 0.15
    return {
        "log_id": f"MED-{RNG.randint(5000, 9999)}",
        "patient_id": pid,
        "date": _days_ago(day_offset),
        "medication": med,
        "scheduled_time": scheduled_time,
        "taken": taken,
        "taken_late": late,
        "missed": not taken,
        "recorded_by": "patient",
    }


def _make_symptom_entry(pid: str, day_offset: int) -> dict:
    return {
        "entry_id": f"SYM-{RNG.randint(8000, 9999)}",
        "patient_id": pid,
        "date": _days_ago(day_offset),
        "symptom": RNG.choice(SYMPTOMS),
        "severity": RNG.choice(["mild", "moderate", "severe"]),
        "duration_min": RNG.randint(10, 240),
        "linked_seizure": RNG.random() < 0.6,
    }


def _make_sos_event(j: int) -> dict:
    pid = PATIENT_IDS[j % len(PATIENT_IDS)]
    return {
        "event_id": f"SOS-{7000 + j}",
        "patient_id": pid,
        "triggered_at": _hours_ago(RNG.randint(1, 120)),
        "trigger_source": RNG.choice(["manual_button", "auto_seizure_detection", "fall_detection"]),
        "location_shared": RNG.random() < 0.7,
        "caregiver_notified": True,
        "status": RNG.choice(SOS_STATUSES),
        "response_time_sec": RNG.randint(15, 300),
        "escalated_to_emergency": RNG.random() < 0.12,
    }


def _paired_device_info(pid: str) -> dict:
    device = RNG.choice(PAIRED_DEVICES)
    if device == "None":
        return {"device": "None", "paired": False, "battery_pct": None, "last_sync": None}
    return {
        "device": device,
        "paired": True,
        "battery_pct": RNG.randint(10, 100),
        "last_sync": _minutes_ago(RNG.randint(5, 180)),
        "firmware_up_to_date": RNG.random() < 0.8,
        "signal_quality": RNG.choice(["excellent", "good", "fair", "poor"]),
    }


def _offline_queue_depth(n_patients: int) -> list[dict]:
    """Simulate how many records are queued locally awaiting sync."""
    out = []
    for i in range(n_patients):
        pending = RNG.randint(0, 15)
        if pending > 0:
            out.append({
                "patient_id": PATIENT_IDS[i % len(PATIENT_IDS)],
                "queued_records": pending,
                "oldest_record_age_h": RNG.randint(1, 48),
                "type": RNG.choice(["seizure_diary", "medication_log", "symptom", "sos"]),
            })
    return out


def _adherence_trend(days: int = 14) -> list[dict]:
    """Daily medication adherence % over past n days."""
    out = []
    for d in range(days):
        pct = round(RNG.uniform(72, 98), 1)
        out.append({"date": _days_ago(days - 1 - d), "adherence_pct": pct})
    return out


# ── public API ─────────────────────────────────────────────────────────────────

def overview() -> dict:
    """App health KPIs: active users, seizure burden, medication adherence, SOS events, offline sync."""
    diary_entries = [_make_diary_entry(i) for i in range(80)]
    med_logs = [_make_medication_log(PATIENT_IDS[i % 20], i % 30) for i in range(120)]
    sos_events = [_make_sos_event(j) for j in range(18)]
    offline_queue = _offline_queue_depth(20)

    active_patients = 20
    seizures_7d = sum(1 for e in diary_entries if _days_ago(7) <= e["date"])
    missed_doses_7d = sum(1 for m in med_logs if not m["taken"] and _days_ago(7) <= m["date"])
    adherence_pct = round(
        100 * sum(1 for m in med_logs if m["taken"]) / max(len(med_logs), 1), 1
    )
    open_sos = sum(1 for s in sos_events if s["status"] not in ("resolved",))
    escalated_sos = sum(1 for s in sos_events if s["escalated_to_emergency"])
    offline_pending = sum(q["queued_records"] for q in offline_queue)

    paired_counts: dict[str, int] = {}
    for pid in PATIENT_IDS:
        dev = _paired_device_info(pid)["device"]
        paired_counts[dev] = paired_counts.get(dev, 0) + 1

    return {
        "kpis": {
            "active_patients": active_patients,
            "seizures_last_7d": seizures_7d,
            "medication_adherence_pct": adherence_pct,
            "missed_doses_7d": missed_doses_7d,
            "open_sos_events": open_sos,
            "escalated_sos_total": escalated_sos,
            "offline_records_pending_sync": offline_pending,
            "avg_diary_entries_per_patient": round(len(diary_entries) / active_patients, 1),
        },
        "adherence_trend_14d": _adherence_trend(14),
        "seizure_type_breakdown": _type_counts(diary_entries),
        "trigger_breakdown": _trigger_counts(diary_entries),
        "sos_recent": sos_events[:6],
        "offline_sync_queue": offline_queue[:8],
        "device_pairing_summary": [
            {"device": k, "count": v} for k, v in sorted(paired_counts.items(), key=lambda x: -x[1])
        ],
    }


def _type_counts(entries: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for e in entries:
        t = e["seizure_type"]
        counts[t] = counts.get(t, 0) + 1
    return [{"type": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]


def _trigger_counts(entries: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for e in entries:
        t = e["trigger"]
        counts[t] = counts.get(t, 0) + 1
    return [{"trigger": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]


def diary() -> dict:
    """Seizure diary entries, medication log, symptom tracker, and per-patient device pairing."""
    diary_entries = [_make_diary_entry(i) for i in range(30)]
    med_logs = [_make_medication_log(PATIENT_IDS[i % 20], i % 14) for i in range(50)]
    symptom_entries = [_make_symptom_entry(PATIENT_IDS[i % 20], i % 21) for i in range(40)]
    patient_devices = [
        {"patient_id": pid, **_paired_device_info(pid)} for pid in PATIENT_IDS
    ]
    sos_events = [_make_sos_event(j) for j in range(12)]

    return {
        "seizure_diary": diary_entries,
        "medication_log": med_logs,
        "symptom_tracker": symptom_entries,
        "patient_device_pairing": patient_devices,
        "sos_events": sos_events,
        "summary": {
            "diary_entries": len(diary_entries),
            "medication_records": len(med_logs),
            "symptom_entries": len(symptom_entries),
            "sos_events": len(sos_events),
        },
    }


def definitions() -> dict:
    """Glossary — app features, data fields, offline-first architecture, alert pathways."""
    return {
        "app_overview": {
            "name": "Patient Mobile App",
            "type": "Patient-facing mobile application (iOS + Android)",
            "modes": ["Online (live sync to backend)", "Offline (local SQLite → queued sync on reconnect)"],
            "core_features": [
                "Seizure diary — log seizure type, duration, trigger, severity",
                "Medication log — scheduled dose reminders + adherence tracking",
                "Symptom tracker — aura, post-ictal, mood, sleep",
                "Device pairing — pair EEG headset / smartwatch for auto-detection",
                "SOS — one-tap emergency alert to caregiver + location share",
                "Offline-first — all data stored locally, synced when connected",
            ],
            "data_sync": "SQLite local store → background sync queue → REST API → backend DB",
            "alert_pathway": "Manual SOS or auto-detection → push notification to caregiver app → escalation if no acknowledge within 60 s",
        },
        "fields": [
            {"field": "seizure_type", "description": "ILAE seizure type classification", "values": SEIZURE_TYPES},
            {"field": "trigger", "description": "Patient-reported seizure precipitant", "values": TRIGGERS},
            {"field": "severity", "description": "Patient-rated seizure severity", "values": ["mild", "moderate", "severe"]},
            {"field": "duration_seconds", "description": "Seizure duration in seconds (patient-estimated)"},
            {"field": "post_ictal_duration_min", "description": "Time to return to baseline (minutes)"},
            {"field": "medication_taken", "description": "Boolean — did patient take scheduled dose?"},
            {"field": "taken_late", "description": "Dose taken > 1 h after scheduled time"},
            {"field": "adherence_pct", "description": "% of scheduled doses taken over rolling 14-day window"},
            {"field": "offline_queued_records", "description": "Records stored locally awaiting connectivity for sync"},
            {"field": "sos_trigger_source", "description": "What triggered the SOS event", "values": ["manual_button", "auto_seizure_detection", "fall_detection"]},
            {"field": "sos_status", "description": "SOS lifecycle state", "values": SOS_STATUSES},
        ],
        "offline_architecture": {
            "local_store": "SQLite on device",
            "sync_strategy": "Write-ahead queue — records written locally, background worker retries until server confirms",
            "conflict_resolution": "Server-wins for medication schedule; client-wins for diary entries (last-write)",
            "max_offline_duration": "72 hours of full data capture without connectivity",
        },
        "adherence_thresholds": [
            {"level": "High", "range": "≥ 90 %", "action": "No intervention"},
            {"level": "Moderate", "range": "75–89 %", "action": "In-app reminder intensified"},
            {"level": "Low", "range": "< 75 %", "action": "Clinician alert + caregiver notification"},
        ],
        "sos_escalation": [
            {"step": 1, "trigger": "SOS button pressed or auto-detected", "action": "Push alert to primary caregiver"},
            {"step": 2, "trigger": "No acknowledge in 60 s", "action": "Alert secondary caregiver"},
            {"step": 3, "trigger": "No acknowledge in 120 s", "action": "Escalate to emergency services (112/911)"},
        ],
        "paired_devices": [
            {"device": "Emotiv EPOC X", "data": "14-ch EEG, auto seizure detection"},
            {"device": "Emotiv Insight 2+", "data": "5-ch EEG, alpha asymmetry, seizure risk"},
            {"device": "Apple Watch", "data": "HR, HRV, SpO2, accelerometer (fall)"},
            {"device": "Empatica E4", "data": "EDA, BVP, accelerometer, temperature"},
            {"device": "Seizure Band", "data": "Wrist-worn EEG + motion seizure band"},
        ],
    }
