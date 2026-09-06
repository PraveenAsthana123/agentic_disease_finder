"""Patient Safety Network Dashboard — cross-table composite safety score per patient.

Aggregates 5 safety dimensions per patient from real clinical.db data:
  1. Caregiver Coverage   (caregivers: trained, first-aid, rescue-med, burnout)
  2. Emergency Readiness  (emergency_contacts: primary, notify_on_seizure, verified)
  3. Medication Adherence (medication_adherence: miss rate last 90 days)
  4. Wearable Monitoring  (wearable_devices + wearable_readings: online, seizure detection)
  5. IoT Alert Burden     (iot_alerts: unresolved critical alerts)

Composite Safety Score = weighted average (0-100, higher = safer).
Real data: 30 caregivers, 30 emergency_contacts, 12600 medication_adherence,
30 wearable_devices, 900 wearable_readings, 50 iot_alerts, 40 patients.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _fetch(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _scalar(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None


def _build_patient_scores(conn):
    """Compute per-patient safety scores across all 5 dimensions."""
    patients = _fetch(conn, "SELECT DISTINCT patient_id FROM patients LIMIT 40")
    patient_ids = [p["patient_id"] for p in patients]

    # Dim 1: Caregiver (weight=25)
    cg_rows = _fetch(
        conn,
        "SELECT patient_id, first_aid_certified, rescue_med_trained, "
        "epilepsy_training_completed, burnout_score FROM caregivers",
    )
    cg_by_pid = {}
    for r in cg_rows:
        pid = r["patient_id"]
        score = 0
        if r["first_aid_certified"]:
            score += 30
        if r["rescue_med_trained"]:
            score += 30
        if r["epilepsy_training_completed"]:
            score += 20
        burnout = r.get("burnout_score") or 50
        score += max(0, 20 - int(burnout / 5))
        cg_by_pid[pid] = min(100, score)

    # Dim 2: Emergency Contacts (weight=20)
    ec_rows = _fetch(
        conn,
        "SELECT patient_id, is_primary, notify_on_seizure, last_verified FROM emergency_contacts",
    )
    ec_by_pid = {}
    for r in ec_rows:
        pid = r["patient_id"]
        score = ec_by_pid.get(pid, 0)
        if r["is_primary"] and r["notify_on_seizure"]:
            score += 60
        elif r["is_primary"]:
            score += 40
        elif r["notify_on_seizure"]:
            score += 20
        else:
            score += 10
        lv = r.get("last_verified") or ""
        if lv and lv > "2025-01-01":
            score += 20
        ec_by_pid[pid] = min(100, score)

    # Dim 3: Medication Adherence (weight=30) — last 90 days
    adh_rows = _fetch(
        conn,
        "SELECT patient_id, "
        "SUM(CASE WHEN taken='no' THEN 1 ELSE 0 END) as missed, "
        "COUNT(*) as total "
        "FROM medication_adherence "
        "WHERE log_date >= date('now','-90 days') "
        "GROUP BY patient_id",
    )
    adh_by_pid = {}
    for r in adh_rows:
        pid = r["patient_id"]
        total = r["total"] or 1
        missed = r["missed"] or 0
        miss_rate = missed / total
        adh_by_pid[pid] = round(max(0, 100 - miss_rate * 200), 1)

    # Dim 4: Wearable Monitoring (weight=15)
    wd_rows = _fetch(conn, "SELECT patient_id, fields_json FROM wearable_devices")
    wr_agg = _fetch(
        conn,
        "SELECT patient_id, "
        "AVG(json_extract(fields_json,'$.health_score')) as avg_health, "
        "MAX(json_extract(fields_json,'$.reading_date')) as last_reading "
        "FROM wearable_readings GROUP BY patient_id",
    )
    wr_by_pid = {r["patient_id"]: r for r in wr_agg}
    wd_by_pid = {}
    for r in wd_rows:
        pid = r["patient_id"]
        try:
            f = json.loads(r["fields_json"] or "{}")
        except Exception:
            f = {}
        score = 0
        status = f.get("status", "offline")
        if status == "active":
            score += 40
        elif status == "charging":
            score += 25
        else:
            score += 5
        if f.get("seizure_detection_enabled"):
            score += 30
        batt = f.get("battery_level", 0) or 0
        score += min(20, int(batt / 5))
        wr = wr_by_pid.get(pid, {})
        last = wr.get("last_reading") or ""
        if last and last > "2025-01-01":
            score += 10
        wd_by_pid[pid] = min(100, score)

    # Dim 5: IoT Alert Burden (weight=10) — unresolved critical penalizes
    ia_rows = _fetch(conn, "SELECT patient_id, fields_json FROM iot_alerts")
    ia_by_pid = {pid: 100 for pid in patient_ids}
    for r in ia_rows:
        pid = r["patient_id"]
        try:
            f = json.loads(r["fields_json"] or "{}")
        except Exception:
            f = {}
        sev = f.get("severity", "info")
        resolved = f.get("resolved", True)
        if not resolved:
            penalty = 30 if sev == "critical" else 15 if sev == "warning" else 5
            current = ia_by_pid.get(pid, 100)
            ia_by_pid[pid] = max(0, current - penalty)

    # Composite
    W = {"cg": 0.25, "ec": 0.20, "adh": 0.30, "wd": 0.15, "ia": 0.10}
    results = []
    for pid in patient_ids:
        cg = cg_by_pid.get(pid, 0)
        ec = ec_by_pid.get(pid, 0)
        adh = adh_by_pid.get(pid, 50)
        wd = wd_by_pid.get(pid, 0)
        ia = ia_by_pid.get(pid, 100)
        composite = round(
            cg * W["cg"] + ec * W["ec"] + adh * W["adh"] + wd * W["wd"] + ia * W["ia"], 1
        )
        tier = (
            "Critical" if composite < 40
            else "At Risk" if composite < 60
            else "Adequate" if composite < 80
            else "Strong"
        )
        results.append({
            "patient_id": pid,
            "composite_score": composite,
            "tier": tier,
            "caregiver_score": round(cg, 1),
            "emergency_score": round(ec, 1),
            "adherence_score": round(adh, 1),
            "wearable_score": round(wd, 1),
            "alert_score": round(ia, 1),
        })
    results.sort(key=lambda x: x["composite_score"])
    return results


def overview():
    conn = _conn()
    try:
        scores = _build_patient_scores(conn)
        total = len(scores)
        if total == 0:
            return {"available": False}

        tiers = {}
        for s in scores:
            tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1

        avg_score = round(sum(s["composite_score"] for s in scores) / total, 1)
        critical_count = tiers.get("Critical", 0)
        at_risk_count = tiers.get("At Risk", 0)
        adequate_count = tiers.get("Adequate", 0)
        strong_count = tiers.get("Strong", 0)

        dim_avgs = {
            "caregiver": round(sum(s["caregiver_score"] for s in scores) / total, 1),
            "emergency": round(sum(s["emergency_score"] for s in scores) / total, 1),
            "adherence": round(sum(s["adherence_score"] for s in scores) / total, 1),
            "wearable": round(sum(s["wearable_score"] for s in scores) / total, 1),
            "alerts": round(sum(s["alert_score"] for s in scores) / total, 1),
        }

        tier_distribution = [
            {"tier": t, "count": c}
            for t, c in [
                ("Critical", critical_count),
                ("At Risk", at_risk_count),
                ("Adequate", adequate_count),
                ("Strong", strong_count),
            ]
        ]

        buckets = {"0-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
        for s in scores:
            c = s["composite_score"]
            if c < 40:
                buckets["0-40"] += 1
            elif c < 60:
                buckets["40-60"] += 1
            elif c < 80:
                buckets["60-80"] += 1
            else:
                buckets["80-100"] += 1
        score_histogram = [{"range": k, "count": v} for k, v in buckets.items()]

        return {
            "kpis": {
                "total_patients": total,
                "avg_composite_score": avg_score,
                "critical_count": critical_count,
                "at_risk_count": at_risk_count,
                "adequate_count": adequate_count,
                "strong_count": strong_count,
                "patients_with_caregiver": _scalar(
                    conn, "SELECT COUNT(DISTINCT patient_id) FROM caregivers"
                ),
                "patients_with_emergency_contact": _scalar(
                    conn, "SELECT COUNT(DISTINCT patient_id) FROM emergency_contacts"
                ),
                "patients_with_wearable": _scalar(
                    conn, "SELECT COUNT(DISTINCT patient_id) FROM wearable_devices"
                ),
            },
            "tier_distribution": tier_distribution,
            "score_histogram": score_histogram,
            "dimension_averages": [
                {"dimension": "Medication Adherence", "score": dim_avgs["adherence"], "weight_pct": 30},
                {"dimension": "Caregiver Coverage", "score": dim_avgs["caregiver"], "weight_pct": 25},
                {"dimension": "Emergency Readiness", "score": dim_avgs["emergency"], "weight_pct": 20},
                {"dimension": "Wearable Monitoring", "score": dim_avgs["wearable"], "weight_pct": 15},
                {"dimension": "IoT Alert Health", "score": dim_avgs["alerts"], "weight_pct": 10},
            ],
            "critical_patients": [s for s in scores if s["tier"] == "Critical"][:8],
        }
    finally:
        conn.close()


def breakdown():
    conn = _conn()
    try:
        scores = _build_patient_scores(conn)

        adh_trend = _fetch(
            conn,
            "SELECT strftime('%Y-%m', log_date) as month, "
            "ROUND(100.0 * SUM(CASE WHEN taken='yes' OR taken='late' THEN 1 ELSE 0 END) / COUNT(*), 1) as adherence_rate "
            "FROM medication_adherence WHERE log_date >= date('now','-12 months') "
            "GROUP BY month ORDER BY month",
        )

        cg_training = _fetch(
            conn,
            "SELECT "
            "SUM(first_aid_certified) as first_aid, "
            "SUM(rescue_med_trained) as rescue_med, "
            "SUM(epilepsy_training_completed) as epilepsy_training, "
            "SUM(safety_plan_exists) as safety_plan, "
            "SUM(seizure_action_plan_exists) as seizure_action_plan, "
            "COUNT(*) as total FROM caregivers",
        )

        ec_coverage = _fetch(
            conn,
            "SELECT "
            "SUM(is_primary) as primary_contacts, "
            "SUM(notify_on_seizure) as notify_enabled, "
            "COUNT(DISTINCT patient_id) as patients_covered "
            "FROM emergency_contacts",
        )

        wd_rows = _fetch(conn, "SELECT patient_id, fields_json FROM wearable_devices")
        status_counts = {}
        seizure_enabled = 0
        for r in wd_rows:
            try:
                f = json.loads(r["fields_json"] or "{}")
            except Exception:
                f = {}
            s = f.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
            if f.get("seizure_detection_enabled"):
                seizure_enabled += 1
        wearable_status = [{"status": k, "count": v} for k, v in status_counts.items()]

        ia_sev = _fetch(
            conn,
            "SELECT json_extract(fields_json,'$.severity') as severity, "
            "COUNT(*) as total, "
            "SUM(CASE WHEN json_extract(fields_json,'$.resolved')=0 THEN 1 ELSE 0 END) as unresolved "
            "FROM iot_alerts GROUP BY severity",
        )

        return {
            "per_patient": scores,
            "adherence_monthly_trend": adh_trend,
            "caregiver_training": cg_training[0] if cg_training else {},
            "emergency_contact_coverage": ec_coverage[0] if ec_coverage else {},
            "wearable_status_distribution": wearable_status,
            "wearable_seizure_detection_count": seizure_enabled,
            "iot_alert_breakdown": ia_sev,
        }
    finally:
        conn.close()


def definitions():
    return {
        "dashboard": "Patient Safety Network Dashboard",
        "description": (
            "Composite patient safety score combining 5 monitoring and support dimensions. "
            "Identifies patients with weak safety networks so care teams can intervene proactively."
        ),
        "dimensions": [
            {
                "name": "Medication Adherence",
                "weight": "30%",
                "source": "medication_adherence (12600 rows)",
                "metric": "100 - (miss_rate x 200), capped 0-100. miss_rate = taken='no' / total",
            },
            {
                "name": "Caregiver Coverage",
                "weight": "25%",
                "source": "caregivers (30 rows)",
                "metric": "first_aid_certified(30) + rescue_med_trained(30) + epilepsy_training(20) + low_burnout(20)",
            },
            {
                "name": "Emergency Readiness",
                "weight": "20%",
                "source": "emergency_contacts (30 rows)",
                "metric": "primary+notify(60) + recently_verified_2025+(20), per patient",
            },
            {
                "name": "Wearable Monitoring",
                "weight": "15%",
                "source": "wearable_devices (30 rows) + wearable_readings (900 rows)",
                "metric": "active_status(40) + seizure_detection(30) + battery_level(20) + recent_reading(10)",
            },
            {
                "name": "IoT Alert Health",
                "weight": "10%",
                "source": "iot_alerts (50 rows)",
                "metric": "Start 100; deduct 30/15/5 per unresolved critical/warning/info alert",
            },
        ],
        "tiers": [
            {"tier": "Strong", "range": "80-100", "color": "green",
             "meaning": "Well-protected patient with robust support network"},
            {"tier": "Adequate", "range": "60-79", "color": "blue",
             "meaning": "Sufficient safety coverage; minor gaps to address"},
            {"tier": "At Risk", "range": "40-59", "color": "amber",
             "meaning": "Notable gaps in safety network; interventions recommended"},
            {"tier": "Critical", "range": "0-39", "color": "red",
             "meaning": "Significant safety deficiencies; immediate attention needed"},
        ],
        "data_sources": [
            "caregivers (30 rows, 30 patients)",
            "emergency_contacts (30 rows, 30 patients)",
            "medication_adherence (12600 rows, 30 patients)",
            "wearable_devices (30 rows, 30 patients)",
            "wearable_readings (900 rows, 30 patients)",
            "iot_alerts (50 rows, 26 patients)",
            "patients (40 rows)",
        ],
        "glossary": {
            "Composite Score": "Weighted average of 5 dimension scores (0-100, higher = safer)",
            "Caregiver Burnout": "burnout_score from caregivers table (0-100, lower = healthier caregiver)",
            "Miss Rate": "Fraction of medication_adherence records where taken='no'",
            "Seizure Detection": "wearable_devices.fields_json.seizure_detection_enabled flag",
            "Unresolved Alert": "iot_alerts where fields_json.resolved = false",
            "Safety Network": "The combined human and technological support system protecting a patient",
        },
    }
