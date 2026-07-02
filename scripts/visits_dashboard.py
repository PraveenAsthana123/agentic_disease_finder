"""True Visits Dashboard — actual visits logged from clinical.db.

Distinguishes true visits (completed appointments) from bookings.
Tracks visit volume, completion rate, provider visit load, department
distribution, visit duration analysis, per-patient visit history,
and daily/monthly trends.

Sources:
- appointments table (status='completed' → true visit)
- patients table (demographics for cross-referencing)
"""

import sqlite3
import os
from datetime import datetime
from collections import Counter, defaultdict

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


def visits_overview():
    """KPI summary: total visits, unique patients, unique providers,
    completion rate, avg duration, visit-to-booking ratio, departments covered."""
    conn = _conn()
    cur = conn.cursor()

    total_visits = _safe(cur, "SELECT COUNT(*) FROM appointments WHERE status='completed'")
    total_appointments = _safe(cur, "SELECT COUNT(*) FROM appointments")
    unique_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM appointments WHERE status='completed'")
    unique_providers = _safe(cur, "SELECT COUNT(DISTINCT provider) FROM appointments WHERE status='completed'")
    unique_departments = _safe(cur, "SELECT COUNT(DISTINCT department) FROM appointments WHERE status='completed'")
    avg_duration = _safe(cur, "SELECT AVG(duration_min) FROM appointments WHERE status='completed'")
    no_show_count = _safe(cur, "SELECT COUNT(*) FROM appointments WHERE status='no-show'")

    completion_rate = round(total_visits / total_appointments * 100, 1) if total_appointments > 0 else 0
    no_show_rate = round(no_show_count / total_appointments * 100, 1) if total_appointments > 0 else 0

    # Status distribution
    status_rows = _safe_rows(cur, "SELECT status, COUNT(*) FROM appointments GROUP BY status ORDER BY COUNT(*) DESC")
    status_distribution = [{"status": r[0], "count": r[1]} for r in status_rows]

    # Provider visit counts
    provider_rows = _safe_rows(cur, """
        SELECT provider, COUNT(*) as cnt FROM appointments
        WHERE status='completed' GROUP BY provider ORDER BY cnt DESC
    """)
    provider_visits = [{"provider": r[0], "visits": r[1]} for r in provider_rows]

    # Department visit counts
    dept_rows = _safe_rows(cur, """
        SELECT department, COUNT(*) as cnt FROM appointments
        WHERE status='completed' GROUP BY department ORDER BY cnt DESC
    """)
    department_visits = [{"department": r[0], "visits": r[1]} for r in dept_rows]

    # Appointment type breakdown for visits
    type_rows = _safe_rows(cur, """
        SELECT appt_type, COUNT(*) as cnt FROM appointments
        WHERE status='completed' GROUP BY appt_type ORDER BY cnt DESC
    """)
    visit_types = [{"type": r[0], "count": r[1]} for r in type_rows]

    conn.close()
    return {
        "kpis": {
            "total_visits": total_visits,
            "total_appointments": total_appointments,
            "completion_rate_pct": completion_rate,
            "unique_patients": unique_patients,
            "unique_providers": unique_providers,
            "unique_departments": unique_departments,
            "avg_duration_min": round(avg_duration, 1) if avg_duration else 0,
            "no_show_rate_pct": no_show_rate
        },
        "status_distribution": status_distribution,
        "provider_visits": provider_visits,
        "department_visits": department_visits,
        "visit_types": visit_types
    }


def visits_breakdown():
    """Per-patient visit history, daily trend, monthly trend, duration
    distribution, provider-department cross-tab, recent visits."""
    conn = _conn()
    cur = conn.cursor()

    # Per-patient visit summary
    patient_rows = _safe_rows(cur, """
        SELECT a.patient_id,
               COALESCE(p.name, a.patient_id) as patient_name,
               COUNT(*) as visit_count,
               AVG(a.duration_min) as avg_duration,
               MIN(a.scheduled_for) as first_visit,
               MAX(a.scheduled_for) as last_visit,
               GROUP_CONCAT(DISTINCT a.department) as departments
        FROM appointments a
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        WHERE a.status = 'completed'
        GROUP BY a.patient_id
        ORDER BY visit_count DESC
    """)
    per_patient = [{
        "patient_id": r[0],
        "name": r[1],
        "visit_count": r[2],
        "avg_duration_min": round(r[3], 1) if r[3] else 0,
        "first_visit": r[4],
        "last_visit": r[5],
        "departments": r[6]
    } for r in patient_rows]

    # Daily visit trend
    daily_rows = _safe_rows(cur, """
        SELECT DATE(scheduled_for) as day, COUNT(*) as cnt
        FROM appointments WHERE status='completed'
        GROUP BY day ORDER BY day
    """)
    daily_trend = [{"date": r[0], "visits": r[1]} for r in daily_rows]

    # Monthly visit trend
    monthly_rows = _safe_rows(cur, """
        SELECT strftime('%Y-%m', scheduled_for) as month, COUNT(*) as cnt
        FROM appointments WHERE status='completed'
        GROUP BY month ORDER BY month
    """)
    monthly_trend = [{"month": r[0], "visits": r[1]} for r in monthly_rows]

    # Duration distribution (buckets: 0-15, 15-30, 30-45, 45-60, 60+)
    dur_rows = _safe_rows(cur, """
        SELECT
            CASE
                WHEN duration_min <= 15 THEN '0-15 min'
                WHEN duration_min <= 30 THEN '16-30 min'
                WHEN duration_min <= 45 THEN '31-45 min'
                WHEN duration_min <= 60 THEN '46-60 min'
                ELSE '60+ min'
            END as bucket,
            COUNT(*) as cnt
        FROM appointments WHERE status='completed'
        GROUP BY bucket ORDER BY MIN(duration_min)
    """)
    duration_distribution = [{"bucket": r[0], "count": r[1]} for r in dur_rows]

    # Provider-department cross-tab
    cross_rows = _safe_rows(cur, """
        SELECT provider, department, COUNT(*) as cnt
        FROM appointments WHERE status='completed'
        GROUP BY provider, department
        ORDER BY cnt DESC
    """)
    provider_dept = [{"provider": r[0], "department": r[1], "visits": r[2]} for r in cross_rows]

    # Recent visits (last 20)
    recent_rows = _safe_rows(cur, """
        SELECT a.patient_id,
               COALESCE(p.name, a.patient_id) as patient_name,
               a.provider, a.department, a.appt_type,
               a.scheduled_for, a.completed_at, a.duration_min, a.notes
        FROM appointments a
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        WHERE a.status = 'completed'
        ORDER BY a.scheduled_for DESC
        LIMIT 20
    """)
    recent_visits = [{
        "patient_id": r[0], "name": r[1], "provider": r[2],
        "department": r[3], "type": r[4], "scheduled_for": r[5],
        "completed_at": r[6], "duration_min": r[7], "notes": r[8]
    } for r in recent_rows]

    conn.close()
    return {
        "per_patient": per_patient,
        "daily_trend": daily_trend,
        "monthly_trend": monthly_trend,
        "duration_distribution": duration_distribution,
        "provider_department": provider_dept,
        "recent_visits": recent_visits
    }


def visits_definitions():
    """Metric definitions for the True Visits dashboard."""
    return {
        "sections": [
            {
                "title": "Visit Concepts",
                "items": [
                    {"term": "True Visit", "definition": "An appointment with status 'completed' — the patient was physically or virtually seen by a provider."},
                    {"term": "Completion Rate", "definition": "Percentage of all appointments that resulted in a completed visit (completed / total appointments)."},
                    {"term": "No-Show Rate", "definition": "Percentage of appointments where the patient did not attend (no-show / total appointments)."},
                    {"term": "Visit Duration", "definition": "Time in minutes recorded for the completed appointment, typically measured from check-in to check-out."}
                ]
            },
            {
                "title": "Quality Metrics",
                "items": [
                    {"term": "Provider Visit Load", "definition": "Number of completed visits per provider — indicates workload distribution across the care team."},
                    {"term": "Department Coverage", "definition": "Number of distinct departments with completed visits — shows breadth of care delivery."},
                    {"term": "Patient Visit Frequency", "definition": "Average number of completed visits per patient — reflects follow-up adherence and care continuity."},
                    {"term": "Duration Distribution", "definition": "Histogram of visit durations bucketed by 15-minute intervals — reveals appointment length patterns."}
                ]
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {"term": "IEC 62304", "definition": "Medical device software lifecycle standard — visit tracking supports clinical workflow validation."},
                    {"term": "HIPAA", "definition": "Visit logs are PHI — access controls and audit trails required per 45 CFR §164."},
                    {"term": "CMS Quality Measures", "definition": "Visit completion and no-show rates feed into quality reporting for value-based care programs."},
                    {"term": "Care Continuity", "definition": "Regular visit patterns indicate adherence to treatment plans — gaps may signal risk for epilepsy patients."}
                ]
            },
            {
                "title": "Remediation",
                "items": [
                    {"term": "High No-Show Rate", "definition": "Implement reminder workflows (SMS/email 24h before), review scheduling barriers, consider telehealth alternatives."},
                    {"term": "Provider Overload", "definition": "Redistribute appointments across providers, consider adding capacity in high-demand departments."},
                    {"term": "Short Visit Duration", "definition": "Review whether rushed visits correlate with missed assessments — may need to extend slot lengths."},
                    {"term": "Low Patient Coverage", "definition": "Identify patients with no completed visits — they may need outreach for appointment scheduling."}
                ]
            }
        ]
    }


if __name__ == '__main__':
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(visits_overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(visits_breakdown(), indent=2, default=str))
