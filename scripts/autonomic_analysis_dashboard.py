"""
Neuro AI Ecosystem -- Autonomic Analysis Dashboard
====================================================
Unified autonomic analysis from wearable readings data: HRV trends,
autonomic dysfunction scoring, seizure-autonomic correlation, and
risk stratification using real clinical.db wearable_readings (900 rows,
30 patients) and wearable_devices (30 devices).

Autonomic Dysfunction Score (ADS):
  Composite 0-100 score based on:
    - HRV (heart_rate_variability): low HRV = higher dysfunction
    - Resting HR deviation from normal (60-80 bpm)
    - SpO2 deviation from normal (>95%)
    - Stress score elevation
    - Sleep quality deficit
  Each component contributes 0-20 points; summed and inverted so
  higher score = greater dysfunction.

Risk Stratification:
  high   -- ADS >= 60 OR (seizure_risk_score > 50 AND ADS >= 40)
  medium -- ADS >= 35 OR seizure_risk_score > 40
  low    -- otherwise

Data Sources:
  - wearable_readings  (900 rows, 30 patients) -- daily wearable metrics
  - wearable_devices   (30 rows) -- device registry
  - patients           -- demographics

Author: Research Team
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def _median(values):
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return round((s[n // 2 - 1] + s[n // 2]) / 2, 2)


def _real_patients(conn):
    """Load patient demographics — prefer patient_demographics (EPAT IDs)
    then fall back to patients table."""
    c = conn.cursor()
    result = {}
    # patient_demographics has EPAT* IDs matching wearable_readings
    try:
        c.execute("""
            SELECT patient_id, full_name, age, sex, epilepsy_type
            FROM patient_demographics WHERE patient_id IS NOT NULL
        """)
        for r in c.fetchall():
            result[r[0]] = {
                "patient_id": r[0], "name": r[1] or f"Patient {r[0]}",
                "age": r[2], "gender": r[3],
                "disease": r[4] or "Epilepsy", "department": "Neurology",
            }
    except Exception:
        pass
    # Fallback to patients table for any IDs not yet covered
    try:
        c.execute("""
            SELECT patient_id, name, age, gender, disease, department
            FROM patients WHERE patient_id IS NOT NULL
        """)
        for r in c.fetchall():
            if r[0] not in result:
                result[r[0]] = {
                    "patient_id": r[0], "name": r[1] or f"Patient {r[0]}",
                    "age": r[2], "gender": r[3],
                    "disease": r[4] or "unspecified", "department": r[5],
                }
    except Exception:
        pass
    return list(result.values())


def _load_readings(conn):
    c = conn.cursor()
    c.execute("SELECT patient_id, device_id, fields_json, created_at FROM wearable_readings ORDER BY created_at")
    results = []
    for row in c.fetchall():
        fields = _safe_json(row[2])
        fields["_patient_id"] = row[0]
        fields["_device_id"] = row[1]
        fields["_created_at"] = row[3]
        results.append(fields)
    return results


def _load_devices(conn):
    c = conn.cursor()
    c.execute("SELECT patient_id, fields_json, created_at FROM wearable_devices ORDER BY patient_id")
    results = []
    for row in c.fetchall():
        fields = _safe_json(row[1])
        fields["_patient_id"] = row[0]
        fields["_created_at"] = row[2]
        results.append(fields)
    return results


def _compute_ads(readings):
    """Compute Autonomic Dysfunction Score (0-100) from a list of readings."""
    if not readings:
        return 0

    hrv_vals = [r.get("heart_rate_variability", 40) for r in readings]
    rhr_vals = [r.get("resting_heart_rate", 70) for r in readings]
    spo2_vals = [r.get("spo2", 97) for r in readings]
    stress_vals = [r.get("stress_score", 30) for r in readings]
    sleep_vals = [r.get("sleep_quality_score", 5) for r in readings]

    avg_hrv = _avg(hrv_vals)
    avg_rhr = _avg(rhr_vals)
    avg_spo2 = _avg(spo2_vals)
    avg_stress = _avg(stress_vals)
    avg_sleep = _avg(sleep_vals)

    # HRV component: normal ~40-60ms; <20 = max dysfunction
    hrv_score = max(0, min(20, 20 - (avg_hrv / 3)))

    # Resting HR component: 60-80 normal; deviation penalized
    rhr_dev = abs(avg_rhr - 70)
    rhr_score = max(0, min(20, rhr_dev * 0.8))

    # SpO2 component: 97-100 normal; <94 = max dysfunction
    spo2_score = max(0, min(20, (97 - avg_spo2) * 5))

    # Stress component: 0-100 scale; >60 = high
    stress_score = max(0, min(20, avg_stress * 0.2))

    # Sleep quality component: 0-10 scale; <4 = poor
    sleep_score = max(0, min(20, (10 - avg_sleep) * 2))

    return round(hrv_score + rhr_score + spo2_score + stress_score + sleep_score, 1)


def _classify_risk(ads, seizure_risk):
    if ads >= 60 or (seizure_risk > 50 and ads >= 40):
        return "high"
    elif ads >= 35 or seizure_risk > 40:
        return "medium"
    return "low"


def _compute_slope(values):
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return round(num / den, 4) if den else 0.0


def overview():
    conn = _conn()
    patients = _real_patients(conn)
    readings = _load_readings(conn)
    devices = _load_devices(conn)
    patient_map = {p["patient_id"]: p for p in patients}

    # Group readings by patient
    by_patient = defaultdict(list)
    for r in readings:
        by_patient[r["_patient_id"]].append(r)

    # Per-patient ADS + risk
    patient_summaries = []
    all_ads = []
    all_hrv = []
    all_seizure_risk = []
    risk_dist = {"high": 0, "medium": 0, "low": 0}

    for pid, recs in by_patient.items():
        ads = _compute_ads(recs)
        avg_seizure_risk = _avg([r.get("seizure_risk_score", 0) for r in recs])
        risk = _classify_risk(ads, avg_seizure_risk)
        risk_dist[risk] += 1
        all_ads.append(ads)
        all_hrv.extend([r.get("heart_rate_variability", 0) for r in recs])
        all_seizure_risk.append(avg_seizure_risk)
        patient_summaries.append({
            "patient_id": pid,
            "ads": ads,
            "risk": risk,
            "avg_seizure_risk": round(avg_seizure_risk, 1),
            "readings_count": len(recs),
        })

    # HRV distribution buckets
    hrv_buckets = {"<20 (severe)": 0, "20-40 (low)": 0, "40-60 (normal)": 0, ">60 (high)": 0}
    for h in all_hrv:
        if h < 20:
            hrv_buckets["<20 (severe)"] += 1
        elif h < 40:
            hrv_buckets["20-40 (low)"] += 1
        elif h <= 60:
            hrv_buckets["40-60 (normal)"] += 1
        else:
            hrv_buckets[">60 (high)"] += 1

    hrv_distribution = [{"range": k, "count": v} for k, v in hrv_buckets.items()]

    # ADS distribution
    ads_buckets = {"0-20 (minimal)": 0, "20-35 (mild)": 0, "35-60 (moderate)": 0, "60-100 (severe)": 0}
    for a in all_ads:
        if a < 20:
            ads_buckets["0-20 (minimal)"] += 1
        elif a < 35:
            ads_buckets["20-35 (mild)"] += 1
        elif a < 60:
            ads_buckets["35-60 (moderate)"] += 1
        else:
            ads_buckets["60-100 (severe)"] += 1

    ads_distribution = [{"range": k, "count": v} for k, v in ads_buckets.items()]

    # Monthly HRV trend (aggregate)
    monthly_hrv = defaultdict(list)
    for r in readings:
        date_str = r.get("reading_date") or r.get("_created_at", "")
        if date_str:
            month = date_str[:7]
            hrv = r.get("heart_rate_variability")
            if hrv is not None:
                monthly_hrv[month].append(hrv)

    hrv_trend = sorted([
        {"month": m, "avg_hrv": _avg(vals), "count": len(vals)}
        for m, vals in monthly_hrv.items()
    ], key=lambda x: x["month"])

    # Device type breakdown
    device_types = defaultdict(int)
    for d in devices:
        dtype = d.get("device_type", "Unknown")
        device_types[dtype] += 1
    device_breakdown = [{"type": k, "count": v} for k, v in sorted(device_types.items())]

    # Seizure-autonomic correlation: patients with seizure detections
    seizure_events = [r for r in readings if r.get("seizure_detected")]
    non_seizure = [r for r in readings if not r.get("seizure_detected")]

    correlation = {
        "seizure_events": len(seizure_events),
        "non_seizure_events": len(non_seizure),
        "avg_hrv_during_seizure": _avg([r.get("heart_rate_variability", 0) for r in seizure_events]) if seizure_events else None,
        "avg_hrv_no_seizure": _avg([r.get("heart_rate_variability", 0) for r in non_seizure]),
        "avg_stress_during_seizure": _avg([r.get("stress_score", 0) for r in seizure_events]) if seizure_events else None,
        "avg_stress_no_seizure": _avg([r.get("stress_score", 0) for r in non_seizure]),
        "avg_hr_during_seizure": _avg([r.get("heart_rate_avg", 0) for r in seizure_events]) if seizure_events else None,
        "avg_hr_no_seizure": _avg([r.get("heart_rate_avg", 0) for r in non_seizure]),
    }

    conn.close()
    return {
        "kpis": {
            "total_patients": len(by_patient),
            "total_readings": len(readings),
            "total_devices": len(devices),
            "avg_ads": _avg(all_ads),
            "median_hrv": _median(all_hrv),
            "high_risk_patients": risk_dist["high"],
            "seizure_detections": len(seizure_events),
        },
        "risk_distribution": [{"risk": k, "count": v} for k, v in risk_dist.items()],
        "hrv_distribution": hrv_distribution,
        "ads_distribution": ads_distribution,
        "hrv_trend": hrv_trend,
        "device_breakdown": device_breakdown,
        "seizure_autonomic_correlation": correlation,
    }


def breakdown():
    conn = _conn()
    patients = _real_patients(conn)
    readings = _load_readings(conn)
    devices = _load_devices(conn)
    patient_map = {p["patient_id"]: p for p in patients}
    device_map = {}
    for d in devices:
        pid = d.get("_patient_id")
        if pid:
            device_map[pid] = d

    by_patient = defaultdict(list)
    for r in readings:
        by_patient[r["_patient_id"]].append(r)

    patient_profiles = []
    for pid, recs in sorted(by_patient.items()):
        info = patient_map.get(pid, {})
        dev = device_map.get(pid, {})
        ads = _compute_ads(recs)
        avg_seizure_risk = _avg([r.get("seizure_risk_score", 0) for r in recs])
        risk = _classify_risk(ads, avg_seizure_risk)

        hrv_vals = [r.get("heart_rate_variability", 0) for r in recs]
        hr_vals = [r.get("heart_rate_avg", 0) for r in recs]
        spo2_vals = [r.get("spo2", 97) for r in recs]
        sleep_vals = [r.get("sleep_duration_hours", 0) for r in recs]
        stress_vals = [r.get("stress_score", 0) for r in recs]
        steps_vals = [r.get("steps", 0) for r in recs]

        hrv_slope = _compute_slope(hrv_vals)

        # Daily readings for timeline
        daily = []
        for r in sorted(recs, key=lambda x: x.get("reading_date", "")):
            daily.append({
                "date": r.get("reading_date", ""),
                "hrv": r.get("heart_rate_variability"),
                "hr_avg": r.get("heart_rate_avg"),
                "spo2": r.get("spo2"),
                "stress": r.get("stress_score"),
                "sleep_hours": r.get("sleep_duration_hours"),
                "sleep_quality": r.get("sleep_quality_score"),
                "steps": r.get("steps"),
                "seizure_risk": r.get("seizure_risk_score"),
                "seizure_detected": r.get("seizure_detected", False),
            })

        patient_profiles.append({
            "patient_id": pid,
            "name": info.get("name", pid),
            "age": info.get("age"),
            "gender": info.get("gender"),
            "disease": info.get("disease"),
            "device_type": dev.get("device_type", "Unknown"),
            "device_brand": dev.get("brand", "Unknown"),
            "device_status": dev.get("status", "unknown"),
            "ads": ads,
            "risk": risk,
            "avg_seizure_risk": round(avg_seizure_risk, 1),
            "readings_count": len(recs),
            "hrv_trend_slope": hrv_slope,
            "hrv_trend_direction": "improving" if hrv_slope > 0.5 else ("declining" if hrv_slope < -0.5 else "stable"),
            "metrics": {
                "avg_hrv": _avg(hrv_vals),
                "min_hrv": min(hrv_vals) if hrv_vals else 0,
                "max_hrv": max(hrv_vals) if hrv_vals else 0,
                "avg_hr": _avg(hr_vals),
                "avg_spo2": _avg(spo2_vals),
                "avg_sleep": _avg(sleep_vals),
                "avg_stress": _avg(stress_vals),
                "avg_steps": round(_avg(steps_vals)),
                "seizure_events": sum(1 for r in recs if r.get("seizure_detected")),
                "fall_events": sum(1 for r in recs if r.get("fall_detected")),
            },
            "daily_readings": daily,
        })

    # High-risk patients list
    high_risk = [p for p in patient_profiles if p["risk"] == "high"]

    # Patients with declining HRV trend
    declining_hrv = [p for p in patient_profiles if p["hrv_trend_direction"] == "declining"]

    conn.close()
    return {
        "patient_profiles": patient_profiles,
        "high_risk_patients": high_risk,
        "declining_hrv_patients": declining_hrv,
        "total_patients": len(patient_profiles),
    }


def definitions():
    return {
        "title": "Autonomic Analysis Dashboard",
        "description": "Unified autonomic analysis from wearable biometric data — HRV trends, autonomic dysfunction scoring, seizure-autonomic correlation, and risk stratification.",
        "data_sources": [
            {"table": "wearable_readings", "rows": 900, "fields": "HR, HRV, SpO2, stress, sleep, steps, seizure_risk, skin_temp"},
            {"table": "wearable_devices", "rows": 30, "fields": "device_type, brand, firmware, status, capabilities"},
            {"table": "patients", "rows": 30, "fields": "demographics"},
        ],
        "metrics": {
            "ads": {
                "name": "Autonomic Dysfunction Score (ADS)",
                "range": "0-100",
                "components": [
                    "HRV deficit (0-20): low HRV = higher score",
                    "Resting HR deviation (0-20): distance from 60-80 bpm norm",
                    "SpO2 deficit (0-20): below 97% threshold",
                    "Stress elevation (0-20): high stress_score",
                    "Sleep quality deficit (0-20): poor sleep_quality_score",
                ],
                "interpretation": "Higher ADS = greater autonomic dysfunction",
            },
            "risk_levels": {
                "high": "ADS >= 60 OR (seizure_risk > 50 AND ADS >= 40)",
                "medium": "ADS >= 35 OR seizure_risk > 40",
                "low": "All others",
            },
            "hrv": {
                "name": "Heart Rate Variability",
                "unit": "milliseconds (SDNN approximation)",
                "ranges": {
                    "severe": "<20 ms — significant autonomic impairment",
                    "low": "20-40 ms — reduced autonomic function",
                    "normal": "40-60 ms — healthy range",
                    "high": ">60 ms — excellent autonomic function",
                },
            },
        },
        "glossary": [
            {"term": "HRV", "definition": "Heart Rate Variability — beat-to-beat interval variation reflecting autonomic nervous system balance"},
            {"term": "ADS", "definition": "Autonomic Dysfunction Score — composite 0-100 metric from 5 biometric domains"},
            {"term": "SpO2", "definition": "Peripheral oxygen saturation — normal ≥95%"},
            {"term": "SDNN", "definition": "Standard deviation of NN intervals — common HRV time-domain metric"},
            {"term": "Seizure-Autonomic Correlation", "definition": "Statistical comparison of autonomic metrics during seizure vs non-seizure periods"},
            {"term": "Resting Heart Rate", "definition": "Heart rate at rest — normal 60-100 bpm; athletes may be lower"},
        ],
    }
