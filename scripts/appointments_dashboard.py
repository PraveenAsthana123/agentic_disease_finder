"""Appointments Dashboard — appointment booking analytics from clinical.db.

Tracks appointment scheduling, completion rates, provider workload,
department distribution, no-show patterns, and booking trends over time.

Sources:
- appointments table (patient_id, provider, department, type, status, dates)
- patients table (demographics for cross-referencing)
- transaction_log (booking-related actions for correlation)
"""

import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta
import random

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


def _ensure_table():
    """Create appointments table and seed realistic data if it doesn't exist."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='appointments'
    """)
    if cur.fetchone():
        conn.close()
        return

    cur.execute("""
        CREATE TABLE appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            department TEXT NOT NULL,
            appt_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'booked',
            booked_at TEXT NOT NULL,
            scheduled_for TEXT NOT NULL,
            completed_at TEXT,
            duration_min INTEGER DEFAULT 30,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Seed realistic appointment data from existing patients
    cur.execute("SELECT patient_id, name, disease, department FROM patients")
    patients = cur.fetchall()
    if not patients:
        conn.close()
        return

    providers = [
        ('Dr. Sharma', 'Neurology'),
        ('Dr. Patel', 'Neurology'),
        ('Dr. Singh', 'Psychiatry'),
        ('Dr. Gupta', 'Neuropsychology'),
        ('Dr. Mehta', 'Neurology'),
        ('Dr. Kaur', 'EEG Lab'),
    ]
    appt_types = [
        'Initial Consultation', 'Follow-up', 'EEG Recording',
        'Medication Review', 'Neuropsych Assessment', 'Emergency',
        'Telemed Follow-up', 'Pre-surgical Evaluation'
    ]
    statuses_weighted = [
        ('completed', 55), ('booked', 15), ('confirmed', 10),
        ('no-show', 8), ('cancelled', 7), ('rescheduled', 5)
    ]
    status_pool = []
    for s, w in statuses_weighted:
        status_pool.extend([s] * w)

    rng = random.Random(42)
    base = datetime(2026, 6, 1, 8, 0, 0)
    rows = []
    for i in range(120):
        pt = rng.choice(patients)
        prov = rng.choice(providers)
        atype = rng.choice(appt_types)
        status = rng.choice(status_pool)

        day_offset = rng.randint(0, 29)
        hour = rng.choice([8, 9, 10, 11, 13, 14, 15, 16])
        minute = rng.choice([0, 15, 30, 45])
        sched = base + timedelta(days=day_offset, hours=hour - 8, minutes=minute)
        booked = sched - timedelta(days=rng.randint(1, 14))
        duration = rng.choice([15, 30, 30, 30, 45, 60, 60, 90])

        completed_at = None
        if status == 'completed':
            completed_at = (sched + timedelta(minutes=duration)).strftime('%Y-%m-%d %H:%M:%S')

        rows.append((
            pt[0], prov[0], prov[1], atype, status,
            booked.strftime('%Y-%m-%d %H:%M:%S'),
            sched.strftime('%Y-%m-%d %H:%M:%S'),
            completed_at, duration, ''
        ))

    cur.executemany("""
        INSERT INTO appointments
            (patient_id, provider, department, appt_type, status,
             booked_at, scheduled_for, completed_at, duration_min, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, rows)

    # Log a transaction for audit trail
    cur.execute("""
        INSERT INTO transaction_log (patient_id, component, action, actor, detail, ts_utc, ts_local)
        VALUES ('SYSTEM', 'appointments', 'create', 'system',
                'Seeded 120 appointment records', datetime('now'), datetime('now','localtime'))
    """)
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────────────────────
#  /api/appointments/overview
# ──────────────────────────────────────────────────────────────

def appointments_overview():
    """Aggregate appointment health: total bookings, completion rate,
    no-show rate, provider workload, department distribution, KPIs."""
    _ensure_table()
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total = _safe(cur, "SELECT COUNT(*) FROM appointments")
    completed = _safe(cur, "SELECT COUNT(*) FROM appointments WHERE status='completed'")
    booked = _safe(cur, "SELECT COUNT(*) FROM appointments WHERE status='booked'")
    confirmed = _safe(cur, "SELECT COUNT(*) FROM appointments WHERE status='confirmed'")
    no_show = _safe(cur, "SELECT COUNT(*) FROM appointments WHERE status='no-show'")
    cancelled = _safe(cur, "SELECT COUNT(*) FROM appointments WHERE status='cancelled'")
    rescheduled = _safe(cur, "SELECT COUNT(*) FROM appointments WHERE status='rescheduled'")

    completion_rate = round(completed / total * 100, 1) if total else 0
    no_show_rate = round(no_show / total * 100, 1) if total else 0

    avg_duration = _safe(cur, "SELECT AVG(duration_min) FROM appointments WHERE status='completed'")
    unique_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM appointments")

    # Status distribution
    status_dist = []
    for row in _safe_rows(cur,
            "SELECT status, COUNT(*) FROM appointments GROUP BY status ORDER BY COUNT(*) DESC"):
        status_dist.append({"status": row[0], "count": row[1]})

    # Provider workload
    provider_workload = []
    for row in _safe_rows(cur,
            """SELECT provider, COUNT(*) as total,
                      SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as done,
                      SUM(CASE WHEN status='no-show' THEN 1 ELSE 0 END) as noshow
               FROM appointments GROUP BY provider ORDER BY total DESC"""):
        provider_workload.append({
            "provider": row[0], "total": row[1],
            "completed": row[2], "no_show": row[3],
            "completion_pct": round(row[2] / row[1] * 100, 1) if row[1] else 0
        })

    # Department distribution
    dept_dist = []
    for row in _safe_rows(cur,
            "SELECT department, COUNT(*) FROM appointments GROUP BY department ORDER BY COUNT(*) DESC"):
        dept_dist.append({"department": row[0], "count": row[1]})

    # Appointment type breakdown
    type_dist = []
    for row in _safe_rows(cur,
            "SELECT appt_type, COUNT(*) FROM appointments GROUP BY appt_type ORDER BY COUNT(*) DESC"):
        type_dist.append({"type": row[0], "count": row[1]})

    conn.close()

    return {
        "available": True,
        "kpi": {
            "total_appointments": total,
            "completed": completed,
            "booked_upcoming": booked + confirmed,
            "no_shows": no_show,
            "cancelled": cancelled,
            "rescheduled": rescheduled,
            "completion_rate_pct": completion_rate,
            "no_show_rate_pct": no_show_rate,
            "avg_duration_min": round(avg_duration, 1) if avg_duration else 0,
            "unique_patients": unique_patients
        },
        "status_distribution": status_dist,
        "provider_workload": provider_workload,
        "department_distribution": dept_dist,
        "appointment_types": type_dist
    }


# ──────────────────────────────────────────────────────────────
#  /api/appointments/breakdown
# ──────────────────────────────────────────────────────────────

def appointments_breakdown():
    """Per-patient appointment matrix, daily trend, hourly heatmap,
    provider-department cross-tab, recent appointments list."""
    _ensure_table()
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # Daily appointment trend
    daily_trend = []
    for row in _safe_rows(cur,
            """SELECT DATE(scheduled_for) as day, COUNT(*) as total,
                      SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as done,
                      SUM(CASE WHEN status='no-show' THEN 1 ELSE 0 END) as noshow
               FROM appointments GROUP BY day ORDER BY day"""):
        daily_trend.append({
            "date": row[0], "total": row[1],
            "completed": row[2], "no_show": row[3]
        })

    # Hourly pattern
    hourly = []
    for row in _safe_rows(cur,
            """SELECT CAST(strftime('%H', scheduled_for) AS INTEGER) as hr, COUNT(*)
               FROM appointments GROUP BY hr ORDER BY hr"""):
        hourly.append({"hour": row[0], "count": row[1]})

    # Per-patient summary
    patient_appts = []
    for row in _safe_rows(cur,
            """SELECT a.patient_id, p.name, COUNT(*) as total,
                      SUM(CASE WHEN a.status='completed' THEN 1 ELSE 0 END) as done,
                      SUM(CASE WHEN a.status='no-show' THEN 1 ELSE 0 END) as noshow
               FROM appointments a LEFT JOIN patients p ON a.patient_id=p.patient_id
               GROUP BY a.patient_id ORDER BY total DESC LIMIT 20"""):
        patient_appts.append({
            "patient_id": row[0], "name": row[1] or row[0],
            "total": row[2], "completed": row[3], "no_show": row[4]
        })

    # Provider-department cross-tab
    prov_dept = []
    for row in _safe_rows(cur,
            """SELECT provider, department, COUNT(*),
                      SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)
               FROM appointments GROUP BY provider, department ORDER BY COUNT(*) DESC"""):
        prov_dept.append({
            "provider": row[0], "department": row[1],
            "total": row[2], "completed": row[3]
        })

    # Recent appointments
    recent = []
    for row in _safe_rows(cur,
            """SELECT a.patient_id, p.name, a.provider, a.appt_type, a.status,
                      a.scheduled_for, a.duration_min
               FROM appointments a LEFT JOIN patients p ON a.patient_id=p.patient_id
               ORDER BY a.scheduled_for DESC LIMIT 15"""):
        recent.append({
            "patient_id": row[0], "name": row[1] or row[0],
            "provider": row[2], "type": row[3], "status": row[4],
            "scheduled_for": row[5], "duration_min": row[6]
        })

    # No-show analysis by type
    noshow_by_type = []
    for row in _safe_rows(cur,
            """SELECT appt_type, COUNT(*) as total,
                      SUM(CASE WHEN status='no-show' THEN 1 ELSE 0 END) as noshow
               FROM appointments GROUP BY appt_type
               HAVING noshow > 0 ORDER BY noshow DESC"""):
        noshow_by_type.append({
            "type": row[0], "total": row[1], "no_show": row[2],
            "rate_pct": round(row[2] / row[1] * 100, 1) if row[1] else 0
        })

    conn.close()

    return {
        "available": True,
        "daily_trend": daily_trend,
        "hourly_pattern": hourly,
        "patient_appointments": patient_appts,
        "provider_department_matrix": prov_dept,
        "recent_appointments": recent,
        "noshow_by_type": noshow_by_type
    }


# ──────────────────────────────────────────────────────────────
#  /api/appointments/definitions
# ──────────────────────────────────────────────────────────────

def appointments_definitions():
    """Metric definitions for tooltip overlays."""
    return {
        "definitions": {
            "total_appointments": "Total number of appointment records across all statuses.",
            "completion_rate": "Percentage of appointments that reached 'completed' status. Target: >85%.",
            "no_show_rate": "Percentage of appointments where the patient did not attend. Target: <10%.",
            "booked_upcoming": "Appointments in 'booked' or 'confirmed' status, awaiting the scheduled date.",
            "avg_duration_min": "Mean duration (minutes) of completed appointments.",
            "unique_patients": "Distinct patients with at least one appointment record.",
            "provider_workload": "Appointment count per provider, with completion and no-show sub-totals.",
            "daily_trend": "Appointments per calendar day, split by completed vs no-show.",
            "hourly_pattern": "Distribution of appointments by hour of day (24h clock).",
            "noshow_by_type": "No-show counts and rates broken down by appointment type.",
            "status_distribution": "Count of appointments in each status: completed, booked, confirmed, no-show, cancelled, rescheduled.",
            "provider_department_matrix": "Cross-tabulation of providers and departments with appointment counts."
        }
    }


if __name__ == '__main__':
    import pprint
    print("=== Overview ===")
    pprint.pprint(appointments_overview())
    print("\n=== Breakdown ===")
    pprint.pprint(appointments_breakdown())
