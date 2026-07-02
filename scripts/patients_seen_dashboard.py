"""Patients Seen Dashboard — unique patients with completed appointments.

Tracks how many patients each provider/department actually saw (completed visits),
daily trends, per-patient visit history, appointment type distribution,
and recent completions.

Sources:
- appointments table (status='completed' → patient was seen)
- patients table (demographics for cross-referencing)
"""

import sqlite3
import os
from collections import defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


def patients_seen_overview():
    """KPI summary: total patients seen, completed appointments, providers,
    departments, avg patients per provider, avg duration, no-show rate,
    completion rate, status distribution."""
    conn = _conn()
    cur = conn.cursor()

    total_patients_seen = _safe(cur,
        "SELECT COUNT(DISTINCT patient_id) FROM appointments WHERE status='completed'")
    total_completed = _safe(cur,
        "SELECT COUNT(*) FROM appointments WHERE status='completed'")
    total_appointments = _safe(cur,
        "SELECT COUNT(*) FROM appointments")
    total_providers = _safe(cur,
        "SELECT COUNT(DISTINCT provider) FROM appointments WHERE status='completed'")
    total_departments = _safe(cur,
        "SELECT COUNT(DISTINCT department) FROM appointments WHERE status='completed'")
    avg_duration = _safe(cur,
        "SELECT AVG(duration_min) FROM appointments WHERE status='completed'")
    no_show_count = _safe(cur,
        "SELECT COUNT(*) FROM appointments WHERE status='no-show'")

    completion_rate = round(total_completed / total_appointments * 100, 1) if total_appointments > 0 else 0
    no_show_rate = round(no_show_count / total_appointments * 100, 1) if total_appointments > 0 else 0
    avg_per_provider = round(total_patients_seen / total_providers, 1) if total_providers > 0 else 0

    conn.close()
    return {
        "total_patients_seen": total_patients_seen,
        "total_completed_appointments": total_completed,
        "total_providers": total_providers,
        "total_departments": total_departments,
        "avg_patients_per_provider": avg_per_provider,
        "avg_duration_min": round(avg_duration, 1) if avg_duration else 0,
        "no_show_rate_pct": no_show_rate,
        "completion_rate_pct": completion_rate
    }


def patients_seen_breakdown():
    """Detailed breakdowns: by status, by appointment type, daily trend,
    by provider, by department, per-patient summary, recent completed."""
    conn = _conn()
    cur = conn.cursor()

    # Status distribution
    status_rows = _safe_rows(cur,
        "SELECT status, COUNT(*) FROM appointments GROUP BY status ORDER BY COUNT(*) DESC")
    by_status = [{"status": r[0], "count": r[1]} for r in status_rows]

    # By appointment type (completed only)
    type_rows = _safe_rows(cur, """
        SELECT appt_type, COUNT(*) FROM appointments
        WHERE status='completed' GROUP BY appt_type ORDER BY COUNT(*) DESC
    """)
    by_appt_type = [{"type": r[0], "count": r[1]} for r in type_rows]

    # Daily patients seen trend
    daily_rows = _safe_rows(cur, """
        SELECT date(scheduled_for) as d, COUNT(DISTINCT patient_id) as cnt
        FROM appointments WHERE status='completed'
        GROUP BY d ORDER BY d
    """)
    daily_trend = [{"date": r[0], "patients_seen": r[1]} for r in daily_rows]

    # By provider
    provider_rows = _safe_rows(cur, """
        SELECT provider, COUNT(DISTINCT patient_id) as ps,
               COUNT(*) as ca,
               AVG(duration_min) as ad,
               ROUND(COUNT(CASE WHEN status='completed' THEN 1 END) * 100.0 /
                     COUNT(*), 1) as cr,
               GROUP_CONCAT(DISTINCT department) as dept
        FROM appointments
        GROUP BY provider ORDER BY ps DESC
    """)
    by_provider = [{
        "provider": r[0],
        "patients_seen": r[1],
        "completed_appointments": r[2],
        "avg_duration_min": round(r[3], 1) if r[3] else 0,
        "completion_rate_pct": r[4] or 0,
        "department": r[5] or ""
    } for r in provider_rows]

    # By department
    dept_rows = _safe_rows(cur, """
        SELECT department, COUNT(DISTINCT patient_id) as ps,
               COUNT(*) as ca,
               AVG(duration_min) as ad,
               COUNT(DISTINCT provider) as prov_cnt
        FROM appointments WHERE status='completed'
        GROUP BY department ORDER BY ps DESC
    """)
    by_department = [{
        "department": r[0],
        "patients_seen": r[1],
        "completed_appointments": r[2],
        "avg_duration_min": round(r[3], 1) if r[3] else 0,
        "providers": r[4]
    } for r in dept_rows]

    # Per-patient summary
    patient_rows = _safe_rows(cur, """
        SELECT a.patient_id, p.name,
               COUNT(*) as vc,
               AVG(a.duration_min) as ad,
               MIN(a.scheduled_for) as fv,
               MAX(a.scheduled_for) as lv,
               GROUP_CONCAT(DISTINCT a.department) as depts
        FROM appointments a
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        WHERE a.status='completed'
        GROUP BY a.patient_id
        ORDER BY vc DESC
    """)
    per_patient = [{
        "patient_id": r[0],
        "name": r[1] or r[0],
        "visit_count": r[2],
        "avg_duration_min": round(r[3], 1) if r[3] else 0,
        "first_visit": r[4] or "",
        "last_visit": r[5] or "",
        "departments": r[6] or ""
    } for r in patient_rows]

    # Recent completed appointments (last 20)
    recent_rows = _safe_rows(cur, """
        SELECT a.patient_id, p.name, a.provider, a.department,
               a.appt_type, a.scheduled_for, a.completed_at, a.duration_min
        FROM appointments a
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        WHERE a.status='completed'
        ORDER BY a.completed_at DESC
        LIMIT 20
    """)
    recent_completed = [{
        "patient_id": r[0],
        "name": r[1] or r[0],
        "provider": r[2],
        "department": r[3],
        "type": r[4],
        "scheduled_for": r[5] or "",
        "completed_at": r[6] or "",
        "duration_min": r[7] or 0
    } for r in recent_rows]

    conn.close()
    return {
        "by_status": by_status,
        "by_appt_type": by_appt_type,
        "daily_trend": daily_trend,
        "by_provider": by_provider,
        "by_department": by_department,
        "per_patient": per_patient,
        "recent_completed": recent_completed
    }


def patients_seen_definitions():
    """Metric definitions and clinical relevance for the Patients Seen dashboard."""
    return {
        "concepts": {
            "Patient Seen": "A unique patient with at least one completed appointment. Distinct from bookings or no-shows — only completed visits count.",
            "Completed Appointment": "An appointment record with status='completed', indicating the patient attended and the visit occurred.",
            "No-Show": "A booked appointment where the patient did not attend (status='no-show'). High no-show rates waste clinical resources.",
            "Completion Rate": "Percentage of all appointments that reached 'completed' status. Target: >70%.",
            "Avg Duration": "Mean duration (minutes) of completed visits. Helps capacity planning and scheduling."
        },
        "quality_metrics": {
            "Coverage": "Fraction of registered patients who have at least one completed visit. Low coverage may indicate access barriers.",
            "Provider Load Balance": "Distribution of patients across providers. Large imbalances may indicate scheduling or referral bottlenecks.",
            "Department Utilization": "Patients seen per department. Neurology typically dominates in epilepsy clinics."
        },
        "clinical_relevance": {
            "ILAE": "International League Against Epilepsy recommends regular follow-up visits (every 3-6 months for stable patients, more frequently for newly diagnosed or refractory cases).",
            "IEC 62304": "Software as a Medical Device traceability — patient-seen counts must accurately reflect clinical records for regulatory audits.",
            "CMS / HEDIS": "Centers for Medicare & Medicaid Services track follow-up visit rates as quality measures for chronic condition management.",
            "HIPAA": "Patient visit records are Protected Health Information. Aggregated counts are de-identified; per-patient views require access controls."
        },
        "remediation_strategies": [
            "If completion rate < 60%: audit booking workflow, send appointment reminders, investigate no-show patterns.",
            "If provider load imbalance > 3x: redistribute patient panels, add providers to high-demand departments.",
            "If no-show rate > 15%: implement reminder calls, overbooking protocols, telehealth alternatives.",
            "If average visit too short (< 15 min): review clinical protocols, ensure adequate time per patient."
        ]
    }
