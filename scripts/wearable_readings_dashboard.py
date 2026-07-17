"""Wearable Health Readings Dashboard — wearable device analytics from clinical.db.

Tracks wearable health readings across 30 patients and 30 devices:
heart rate, steps, sleep, SpO2, stress, seizure detection, fall detection,
health scores, seizure risk scores, and daily trends.

Sources:
- wearable_readings table (id, patient_id, device_id, fields_json, created_at)
  fields_json contains: {"patient_id": ..., "device_id": ..., "reading_date": ...,
  "heart_rate_avg": ..., "steps": ..., "sleep_duration_hours": ..., "spo2": ...,
  "health_score": ..., "stress_score": ..., "seizure_risk_score": ...,
  "seizure_detected": ..., "fall_detected": ..., ...}
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _load_all_readings(conn):
    """Load and parse all wearable_readings rows, returning a list of dicts
    with patient_id, device_id, created_at, and all parsed fields."""
    cur = conn.cursor()
    cur.execute(
        "SELECT patient_id, device_id, fields_json, created_at "
        "FROM wearable_readings ORDER BY created_at"
    )
    rows = cur.fetchall()
    result = []
    for patient_id, device_id, fields_json, created_at in rows:
        try:
            data = json.loads(fields_json)
        except (json.JSONDecodeError, TypeError):
            data = {}
        data["_patient_id"] = patient_id
        data["_device_id"] = device_id
        data["_created_at"] = created_at
        result.append(data)
    return result


def _safe_avg(values):
    return round(sum(values) / len(values), 1) if values else 0


def overview():
    """High-level wearable metrics: totals, averages, distributions, daily trend."""
    conn = _conn()
    readings = _load_all_readings(conn)
    conn.close()

    total_readings = len(readings)
    patient_ids = set(r["_patient_id"] for r in readings)
    device_ids = set(r["_device_id"] for r in readings)
    total_patients = len(patient_ids)
    total_devices = len(device_ids)

    # Averages
    hr_vals = [r["heart_rate_avg"] for r in readings if "heart_rate_avg" in r]
    steps_vals = [r["steps"] for r in readings if "steps" in r]
    sleep_vals = [r["sleep_duration_hours"] for r in readings if "sleep_duration_hours" in r]
    spo2_vals = [r["spo2"] for r in readings if "spo2" in r]
    health_vals = [r["health_score"] for r in readings if "health_score" in r]
    stress_vals = [r["stress_score"] for r in readings if "stress_score" in r]
    risk_vals = [r["seizure_risk_score"] for r in readings if "seizure_risk_score" in r]

    avg_heart_rate = _safe_avg(hr_vals)
    avg_steps = _safe_avg(steps_vals)
    avg_sleep_hours = _safe_avg(sleep_vals)
    avg_spo2 = _safe_avg(spo2_vals)
    avg_health_score = _safe_avg(health_vals)
    avg_stress_score = _safe_avg(stress_vals)
    avg_seizure_risk = _safe_avg(risk_vals)

    # Event counts
    seizure_event_count = sum(1 for r in readings if r.get("seizure_detected") is True)
    fall_event_count = sum(1 for r in readings if r.get("fall_detected") is True)
    seizure_detection_rate = round(100.0 * seizure_event_count / total_readings, 1) if total_readings else 0

    # Heart rate distribution (bins of 10)
    hr_buckets = {"<60": 0, "60-70": 0, "70-80": 0, "80-90": 0, "90+": 0}
    for hr in hr_vals:
        if hr < 60:
            hr_buckets["<60"] += 1
        elif hr < 70:
            hr_buckets["60-70"] += 1
        elif hr < 80:
            hr_buckets["70-80"] += 1
        elif hr < 90:
            hr_buckets["80-90"] += 1
        else:
            hr_buckets["90+"] += 1
    heart_rate_distribution = [
        {"bucket": k, "count": v} for k, v in hr_buckets.items()
    ]

    # Sleep quality distribution (scores 1-10)
    sq_dist = {i: 0 for i in range(1, 11)}
    for r in readings:
        sq = r.get("sleep_quality_score")
        if sq is not None and 1 <= sq <= 10:
            sq_dist[int(sq)] = sq_dist.get(int(sq), 0) + 1
    sleep_quality_distribution = [
        {"score": k, "count": v} for k, v in sorted(sq_dist.items())
    ]

    # Daily trend
    daily = {}
    for r in readings:
        date = r.get("reading_date", r.get("_created_at", "")[:10])
        if date not in daily:
            daily[date] = {"hr": [], "steps": [], "sleep": [], "seizures": 0}
        if "heart_rate_avg" in r:
            daily[date]["hr"].append(r["heart_rate_avg"])
        if "steps" in r:
            daily[date]["steps"].append(r["steps"])
        if "sleep_duration_hours" in r:
            daily[date]["sleep"].append(r["sleep_duration_hours"])
        if r.get("seizure_detected") is True:
            daily[date]["seizures"] += 1

    daily_trend = []
    for date in sorted(daily.keys()):
        d = daily[date]
        daily_trend.append({
            "date": date,
            "avg_heart_rate": _safe_avg(d["hr"]),
            "avg_steps": _safe_avg(d["steps"]),
            "avg_sleep": _safe_avg(d["sleep"]),
            "seizure_events": d["seizures"],
        })

    # Activity distribution based on active_minutes
    activity_dist = {"sedentary": 0, "light": 0, "moderate": 0, "active": 0}
    for r in readings:
        am = r.get("active_minutes")
        if am is not None:
            if am < 30:
                activity_dist["sedentary"] += 1
            elif am < 60:
                activity_dist["light"] += 1
            elif am < 120:
                activity_dist["moderate"] += 1
            else:
                activity_dist["active"] += 1
    activity_distribution = [
        {"category": k, "count": v} for k, v in activity_dist.items()
    ]

    return {
        "total_readings": total_readings,
        "total_patients": total_patients,
        "total_devices": total_devices,
        "avg_heart_rate": avg_heart_rate,
        "avg_steps": avg_steps,
        "avg_sleep_hours": avg_sleep_hours,
        "avg_spo2": avg_spo2,
        "avg_health_score": avg_health_score,
        "avg_stress_score": avg_stress_score,
        "avg_seizure_risk": avg_seizure_risk,
        "seizure_event_count": seizure_event_count,
        "fall_event_count": fall_event_count,
        "seizure_detection_rate": seizure_detection_rate,
        "heart_rate_distribution": heart_rate_distribution,
        "sleep_quality_distribution": sleep_quality_distribution,
        "daily_trend": daily_trend,
        "activity_distribution": activity_distribution,
    }


def breakdown():
    """Per-patient summaries, high-risk patients, recent readings,
    seizure events, daily patient readings."""
    conn = _conn()
    readings = _load_all_readings(conn)
    conn.close()

    # Per-patient aggregation
    patients = {}
    for r in readings:
        pid = r["_patient_id"]
        if pid not in patients:
            patients[pid] = {
                "patient_id": pid,
                "device_id": r["_device_id"],
                "hr": [], "steps": [], "sleep": [], "spo2": [],
                "health": [], "seizure_risk": [],
                "seizure_events": 0, "fall_events": 0, "count": 0,
            }
        p = patients[pid]
        p["count"] += 1
        if "heart_rate_avg" in r:
            p["hr"].append(r["heart_rate_avg"])
        if "steps" in r:
            p["steps"].append(r["steps"])
        if "sleep_duration_hours" in r:
            p["sleep"].append(r["sleep_duration_hours"])
        if "spo2" in r:
            p["spo2"].append(r["spo2"])
        if "health_score" in r:
            p["health"].append(r["health_score"])
        if "seizure_risk_score" in r:
            p["seizure_risk"].append(r["seizure_risk_score"])
        if r.get("seizure_detected") is True:
            p["seizure_events"] += 1
        if r.get("fall_detected") is True:
            p["fall_events"] += 1

    per_patient = []
    for pid in sorted(patients.keys()):
        p = patients[pid]
        per_patient.append({
            "patient_id": pid,
            "device_id": p["device_id"],
            "readings_count": p["count"],
            "avg_hr": _safe_avg(p["hr"]),
            "avg_steps": _safe_avg(p["steps"]),
            "avg_sleep": _safe_avg(p["sleep"]),
            "avg_spo2": _safe_avg(p["spo2"]),
            "avg_health": _safe_avg(p["health"]),
            "seizure_events": p["seizure_events"],
            "fall_events": p["fall_events"],
            "avg_seizure_risk": _safe_avg(p["seizure_risk"]),
        })

    # High-risk patients: avg seizure_risk_score > 50 or seizure events > 5
    high_risk_patients = [
        p for p in per_patient
        if p["avg_seizure_risk"] > 50 or p["seizure_events"] > 5
    ]

    # Recent readings: last 20 readings with all fields
    sorted_readings = sorted(readings, key=lambda r: r.get("reading_date", r.get("_created_at", "")), reverse=True)
    recent_readings = []
    for r in sorted_readings[:20]:
        entry = {k: v for k, v in r.items() if not k.startswith("_")}
        entry["patient_id"] = r["_patient_id"]
        entry["device_id"] = r["_device_id"]
        entry["created_at"] = r["_created_at"]
        recent_readings.append(entry)

    # Seizure events: all readings where seizure_detected=true
    seizure_events = []
    for r in readings:
        if r.get("seizure_detected") is True:
            seizure_events.append({
                "patient_id": r["_patient_id"],
                "date": r.get("reading_date", r.get("_created_at", "")[:10]),
                "confidence": r.get("seizure_detection_confidence", 0),
                "heart_rate_avg": r.get("heart_rate_avg", 0),
            })
    seizure_events.sort(key=lambda x: x["date"], reverse=True)

    # Daily patient readings: per patient, per date summary
    daily_patient = {}
    for r in readings:
        pid = r["_patient_id"]
        date = r.get("reading_date", r.get("_created_at", "")[:10])
        key = (pid, date)
        if key not in daily_patient:
            daily_patient[key] = {
                "patient_id": pid,
                "date": date,
                "hr": [], "steps": [], "sleep": [], "spo2": [],
                "seizures": 0, "falls": 0,
            }
        dp = daily_patient[key]
        if "heart_rate_avg" in r:
            dp["hr"].append(r["heart_rate_avg"])
        if "steps" in r:
            dp["steps"].append(r["steps"])
        if "sleep_duration_hours" in r:
            dp["sleep"].append(r["sleep_duration_hours"])
        if "spo2" in r:
            dp["spo2"].append(r["spo2"])
        if r.get("seizure_detected") is True:
            dp["seizures"] += 1
        if r.get("fall_detected") is True:
            dp["falls"] += 1

    daily_patient_readings = []
    for key in sorted(daily_patient.keys()):
        dp = daily_patient[key]
        daily_patient_readings.append({
            "patient_id": dp["patient_id"],
            "date": dp["date"],
            "avg_hr": _safe_avg(dp["hr"]),
            "avg_steps": _safe_avg(dp["steps"]),
            "avg_sleep": _safe_avg(dp["sleep"]),
            "avg_spo2": _safe_avg(dp["spo2"]),
            "seizure_events": dp["seizures"],
            "fall_events": dp["falls"],
        })

    return {
        "per_patient": per_patient,
        "high_risk_patients": high_risk_patients,
        "recent_readings": recent_readings,
        "seizure_events": seizure_events,
        "daily_patient_readings": daily_patient_readings,
    }


def definitions():
    """Clinical glossary, field definitions, clinical notes, thresholds."""
    return {
        "glossary": [
            {"term": "Heart Rate Variability", "definition": "The variation in time intervals between consecutive heartbeats (R-R intervals), measured in milliseconds. Higher HRV generally indicates better autonomic nervous system function and cardiovascular fitness. In epilepsy, reduced HRV may precede or accompany seizures."},
            {"term": "SpO2", "definition": "Peripheral oxygen saturation — the percentage of hemoglobin in the blood that is carrying oxygen, measured non-invasively via pulse oximetry. Normal range is 95-100%. Values below 90% indicate hypoxemia and may occur during or after seizures."},
            {"term": "Seizure Risk Score", "definition": "A composite score (0-100) derived from multiple wearable signals (heart rate variability, movement patterns, skin conductance, sleep quality) that estimates the relative likelihood of seizure occurrence. Higher scores indicate elevated risk."},
            {"term": "Health Score", "definition": "An aggregate wellness metric (0-100) combining heart rate, activity, sleep quality, stress, and SpO2 into a single daily health indicator. Scores above 70 suggest good overall health; below 50 may warrant clinical review."},
            {"term": "Deep Sleep", "definition": "The restorative phase of non-REM sleep (N3 stage) characterised by slow delta waves (0.5-2 Hz). Typically comprises 15-25% of total sleep. Important for physical recovery, immune function, and memory consolidation. Often disrupted in epilepsy patients."},
            {"term": "REM Sleep", "definition": "Rapid Eye Movement sleep — the dream stage characterised by fast desynchronised EEG activity, muscle atonia, and rapid eye movements. Typically 20-25% of total sleep. Seizures are less common during REM due to the desynchronised cortical state."},
            {"term": "Light Sleep", "definition": "The initial and transitional stages of non-REM sleep (N1 and N2). Characterised by sleep spindles and K-complexes. Typically comprises 50-60% of total sleep. Epileptiform discharges are often most prominent during light sleep transitions."},
            {"term": "Stress Score", "definition": "A wearable-derived metric (0-100) based on heart rate variability, skin conductance, and activity patterns that estimates physiological stress levels. Chronic elevated stress may lower seizure threshold in epilepsy patients."},
            {"term": "Resting Heart Rate", "definition": "The heart rate measured during periods of inactivity or rest, typically in beats per minute (bpm). Normal adult range is 60-100 bpm. Resting HR below 60 (bradycardia) or persistently above 100 (tachycardia) may warrant clinical evaluation."},
            {"term": "Active Minutes", "definition": "The total number of minutes per day during which the wearable detects moderate-to-vigorous physical activity based on accelerometer and heart rate data. Guidelines recommend at least 150 minutes per week for adults."},
            {"term": "Skin Temperature", "definition": "Peripheral skin surface temperature measured by the wearable sensor, typically in degrees Celsius. Normal range is 33-37C. Deviations may indicate fever, autonomic dysfunction, or environmental exposure. Skin temperature changes can precede seizure onset."},
            {"term": "Fall Detection", "definition": "An accelerometer-based algorithm that identifies sudden impact patterns consistent with a fall event. Important for epilepsy patients who may experience atonic seizures (drop attacks) or fall during tonic-clonic seizures."},
            {"term": "Seizure Detection Confidence", "definition": "A probability score (0.0-1.0) indicating the algorithm's confidence that a detected event is a true seizure rather than a false positive. Scores above 0.8 are considered high confidence; below 0.3 are likely artifacts or normal movement."},
            {"term": "Awakenings", "definition": "The number of times a patient transitions from sleep to wakefulness during a sleep period. Frequent awakenings (>5 per night) may indicate sleep fragmentation, which is common in epilepsy and can increase seizure susceptibility."},
            {"term": "Sleep Quality Score", "definition": "A composite score (1-10) derived from sleep duration, deep sleep percentage, REM percentage, awakenings count, and restlessness. Scores of 7-10 indicate good sleep quality; below 4 suggests poor sleep that may affect seizure control."},
        ],
        "field_definitions": {
            "patient_id": "Unique patient identifier (e.g., EPAT001)",
            "device_id": "Unique wearable device identifier (e.g., WD-0001)",
            "reading_date": "Date of the wearable reading in YYYY-MM-DD format",
            "heart_rate_avg": "Average heart rate in beats per minute over the reading period",
            "heart_rate_min": "Minimum heart rate recorded during the reading period (bpm)",
            "heart_rate_max": "Maximum heart rate recorded during the reading period (bpm)",
            "heart_rate_variability": "Heart rate variability in milliseconds (RMSSD method)",
            "resting_heart_rate": "Resting heart rate in beats per minute",
            "steps": "Total step count recorded during the day",
            "distance_km": "Total distance walked/run in kilometres",
            "calories_burned": "Estimated total calories burned including basal metabolic rate",
            "active_minutes": "Minutes of moderate-to-vigorous physical activity",
            "sleep_duration_hours": "Total sleep duration in hours",
            "sleep_quality_score": "Composite sleep quality rating from 1 (worst) to 10 (best)",
            "deep_sleep_pct": "Percentage of total sleep spent in deep sleep (N3 stage)",
            "rem_sleep_pct": "Percentage of total sleep spent in REM sleep",
            "light_sleep_pct": "Percentage of total sleep spent in light sleep (N1/N2 stages)",
            "awakenings": "Number of awakenings during the sleep period",
            "stress_score": "Physiological stress level from 0 (calm) to 100 (high stress)",
            "skin_temperature": "Skin surface temperature in degrees Celsius",
            "spo2": "Peripheral oxygen saturation percentage",
            "seizure_detected": "Boolean flag indicating whether the algorithm detected a seizure event",
            "seizure_detection_confidence": "Algorithm confidence score for seizure detection (0.0-1.0)",
            "fall_detected": "Boolean flag indicating whether a fall event was detected",
            "health_score": "Aggregate daily health score from 0 (poor) to 100 (excellent)",
            "seizure_risk_score": "Composite seizure risk estimate from 0 (low) to 100 (high)",
        },
        "clinical_notes": [
            "Wearable devices provide continuous, ambulatory physiological monitoring that complements clinical EEG. While not a replacement for gold-standard video-EEG monitoring, wearables enable long-term seizure tracking and early warning in the patient's natural environment.",
            "Seizure detection algorithms in consumer wearables primarily rely on heart rate acceleration, electrodermal activity changes, and abnormal movement patterns. Sensitivity for tonic-clonic seizures is typically 70-90%, but detection of focal seizures without motor manifestation remains poor (<30%).",
            "Sleep quality is a critical modifiable factor in epilepsy management. Poor sleep (fragmentation, insufficient deep sleep) lowers seizure threshold. Wearable sleep tracking helps clinicians identify sleep hygiene issues and adjust anti-seizure medication timing.",
            "Heart rate variability (HRV) is an emerging biomarker for seizure prediction. Pre-ictal HRV reduction has been observed 30-60 minutes before seizure onset in some patients, supporting the use of continuous HRV monitoring via wearables for early warning systems.",
            "Fall detection in epilepsy is particularly important for SUDEP (Sudden Unexpected Death in Epilepsy) prevention. Wearable-triggered alerts can notify caregivers within seconds of a fall or prolonged tonic-clonic seizure, enabling timely intervention and reducing SUDEP risk.",
        ],
        "thresholds": {
            "heart_rate": {
                "normal": "60-100 bpm",
                "bradycardia": "<60 bpm",
                "tachycardia": ">100 bpm",
                "note": "Resting HR consistently outside 50-90 bpm in epilepsy patients may reflect autonomic dysfunction or medication effects",
            },
            "spo2": {
                "normal": ">95%",
                "mild_hypoxemia": "90-95%",
                "moderate_hypoxemia": "85-90%",
                "severe_hypoxemia": "<85%",
                "note": "Post-ictal SpO2 drops below 90% are common after generalised tonic-clonic seizures and may last several minutes",
            },
            "sleep_duration": {
                "normal": "7-9 hours",
                "short_sleep": "<6 hours",
                "long_sleep": ">10 hours",
                "note": "Both short and excessively long sleep are associated with increased seizure frequency in epilepsy",
            },
            "stress_score": {
                "low": "<30",
                "moderate": "30-60",
                "high": ">60",
                "note": "Sustained high stress scores may correlate with increased seizure susceptibility and should trigger clinical review",
            },
            "seizure_risk_score": {
                "low": "<20",
                "moderate": "20-50",
                "high": ">50",
                "note": "Patients with consistently high seizure risk scores (>50) should be flagged for medication review and increased monitoring",
            },
        },
    }
