"""
Patients Seen Dashboard — queries real data from clinical.db.

Provides overview(), breakdown(), and definitions() for the Patients Seen dashboard.
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# 1. overview() — summary KPIs
# ---------------------------------------------------------------------------

def overview():
    conn = _connect()
    cur = conn.cursor()

    # Total patients seen (unique patients with at least one completed appointment)
    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM appointments WHERE status = 'completed'")
    total_patients_seen = cur.fetchone()[0]

    # Total completed appointments
    cur.execute("SELECT COUNT(*) FROM appointments WHERE status = 'completed'")
    total_completed = cur.fetchone()[0]

    # Total appointments
    cur.execute("SELECT COUNT(*) FROM appointments")
    total_appts = cur.fetchone()[0]

    # Total providers
    cur.execute("SELECT COUNT(DISTINCT provider) FROM appointments")
    total_providers = cur.fetchone()[0]

    # Total departments
    cur.execute("SELECT COUNT(DISTINCT department) FROM appointments")
    total_departments = cur.fetchone()[0]

    # Avg patients per provider (unique patients with completed appts per provider)
    cur.execute("""
        SELECT AVG(cnt) FROM (
            SELECT COUNT(DISTINCT patient_id) AS cnt
            FROM appointments
            WHERE status = 'completed'
            GROUP BY provider
        )
    """)
    row = cur.fetchone()
    avg_patients_per_provider = round(row[0], 1) if row[0] is not None else 0

    # Avg duration of completed appointments
    cur.execute("SELECT AVG(duration_min) FROM appointments WHERE status = 'completed' AND duration_min IS NOT NULL")
    row = cur.fetchone()
    avg_duration = round(row[0], 1) if row[0] is not None else 0

    # No-show rate
    cur.execute("SELECT COUNT(*) FROM appointments WHERE status = 'no-show'")
    no_show_count = cur.fetchone()[0]
    no_show_rate = round(no_show_count / total_appts * 100, 1) if total_appts > 0 else 0

    # Completion rate
    completion_rate = round(total_completed / total_appts * 100, 1) if total_appts > 0 else 0

    conn.close()

    return {
        "total_patients_seen": total_patients_seen,
        "total_completed_appointments": total_completed,
        "total_providers": total_providers,
        "total_departments": total_departments,
        "avg_patients_per_provider": avg_patients_per_provider,
        "avg_duration_min": avg_duration,
        "no_show_rate_pct": no_show_rate,
        "completion_rate_pct": completion_rate,
    }


# ---------------------------------------------------------------------------
# 2. breakdown() — detailed data
# ---------------------------------------------------------------------------

def breakdown():
    conn = _connect()
    cur = conn.cursor()

    # --- by_provider ---
    cur.execute("""
        SELECT
            provider,
            COUNT(DISTINCT CASE WHEN status = 'completed' THEN patient_id END) AS patients_seen,
            COUNT(*) AS total_appts,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN status = 'no-show' THEN 1 ELSE 0 END) AS no_show,
            SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled,
            AVG(CASE WHEN status = 'completed' AND duration_min IS NOT NULL THEN duration_min END) AS avg_dur
        FROM appointments
        GROUP BY provider
        ORDER BY patients_seen DESC
    """)
    by_provider = []
    for r in cur.fetchall():
        total = r["total_appts"]
        completed = r["completed"]
        by_provider.append({
            "provider": r["provider"],
            "patients_seen": r["patients_seen"],
            "total_appts": total,
            "completed": completed,
            "no_show": r["no_show"],
            "cancelled": r["cancelled"],
            "completion_rate_pct": round(completed / total * 100, 1) if total > 0 else 0,
            "avg_duration_min": round(r["avg_dur"], 1) if r["avg_dur"] is not None else 0,
        })

    # --- by_department ---
    cur.execute("""
        SELECT
            department,
            COUNT(DISTINCT CASE WHEN status = 'completed' THEN patient_id END) AS patients_seen,
            COUNT(*) AS total_appts,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed
        FROM appointments
        GROUP BY department
        ORDER BY patients_seen DESC
    """)
    by_department = []
    for r in cur.fetchall():
        total = r["total_appts"]
        completed = r["completed"]
        by_department.append({
            "department": r["department"],
            "patients_seen": r["patients_seen"],
            "total_appts": total,
            "completed": completed,
            "completion_rate_pct": round(completed / total * 100, 1) if total > 0 else 0,
        })

    # --- by_appt_type ---
    cur.execute("""
        SELECT
            appt_type,
            COUNT(*) AS count,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed
        FROM appointments
        GROUP BY appt_type
        ORDER BY count DESC
    """)
    by_appt_type = []
    for r in cur.fetchall():
        total = r["count"]
        completed = r["completed"]
        by_appt_type.append({
            "appt_type": r["appt_type"],
            "count": total,
            "completed": completed,
            "completion_rate_pct": round(completed / total * 100, 1) if total > 0 else 0,
        })

    # --- by_status ---
    cur.execute("""
        SELECT status, COUNT(*) AS count
        FROM appointments
        GROUP BY status
        ORDER BY count DESC
    """)
    by_status = [{"status": r["status"], "count": r["count"]} for r in cur.fetchall()]

    # --- daily_trend (completed appointments grouped by date) ---
    cur.execute("""
        SELECT
            DATE(completed_at) AS date,
            COUNT(DISTINCT patient_id) AS patients_seen,
            COUNT(*) AS appointments
        FROM appointments
        WHERE status = 'completed' AND completed_at IS NOT NULL
        GROUP BY DATE(completed_at)
        ORDER BY date
    """)
    daily_trend = [
        {"date": r["date"], "patients_seen": r["patients_seen"], "appointments": r["appointments"]}
        for r in cur.fetchall()
    ]

    # --- per_patient (patients with completed appointments) ---
    cur.execute("""
        SELECT
            a.patient_id,
            p.name,
            p.age,
            p.gender,
            p.disease,
            COUNT(*) AS total_visits,
            COUNT(DISTINCT a.provider) AS providers_seen,
            GROUP_CONCAT(DISTINCT a.department) AS departments,
            MAX(a.completed_at) AS last_visit
        FROM appointments a
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        WHERE a.status = 'completed'
        GROUP BY a.patient_id
        ORDER BY total_visits DESC
    """)
    per_patient = []
    for r in cur.fetchall():
        per_patient.append({
            "patient_id": r["patient_id"],
            "name": r["name"],
            "age": r["age"],
            "gender": r["gender"],
            "disease": r["disease"],
            "total_visits": r["total_visits"],
            "providers_seen": r["providers_seen"],
            "departments": r["departments"],
            "last_visit": r["last_visit"],
        })

    # --- recent_completed (last 20 completed appointments) ---
    cur.execute("""
        SELECT patient_id, provider, department, appt_type, completed_at, duration_min
        FROM appointments
        WHERE status = 'completed' AND completed_at IS NOT NULL
        ORDER BY completed_at DESC
        LIMIT 20
    """)
    recent_completed = [
        {
            "patient_id": r["patient_id"],
            "provider": r["provider"],
            "department": r["department"],
            "appt_type": r["appt_type"],
            "completed_at": r["completed_at"],
            "duration_min": r["duration_min"],
        }
        for r in cur.fetchall()
    ]

    conn.close()

    return {
        "by_provider": by_provider,
        "by_department": by_department,
        "by_appt_type": by_appt_type,
        "by_status": by_status,
        "daily_trend": daily_trend,
        "per_patient": per_patient,
        "recent_completed": recent_completed,
    }


# ---------------------------------------------------------------------------
# 3. definitions() — clinical definitions
# ---------------------------------------------------------------------------

def definitions():
    return {
        "concepts": {
            "patients_seen": (
                "Unique patients who had at least one completed appointment "
                "within the reporting period. A single patient with multiple "
                "completed visits counts once."
            ),
            "completion_rate": (
                "Percentage of all booked appointments that reached 'completed' "
                "status: (completed / total_appointments) * 100."
            ),
            "no_show_rate": (
                "Percentage of appointments where the patient did not attend "
                "without prior cancellation: (no-show / total_appointments) * 100."
            ),
            "avg_duration": (
                "Mean duration in minutes of completed appointments, computed "
                "from the duration_min field recorded at checkout."
            ),
            "provider_load": (
                "Number of unique patients seen per provider, indicating "
                "workload distribution across the clinical team."
            ),
            "department_coverage": (
                "Distinct departments a patient has visited, reflecting "
                "multi-disciplinary care engagement."
            ),
        },
        "quality_metrics": {
            "completion_rate_pct": (
                "Target >= 85%. Below 80% triggers scheduling workflow review."
            ),
            "no_show_rate_pct": (
                "Target <= 10%. Above 15% triggers patient engagement intervention."
            ),
            "avg_duration_min": (
                "Benchmark 30-60 min for neurology consults. Outliers above 90 min "
                "or below 15 min flagged for review."
            ),
            "patients_per_provider": (
                "Balanced distribution expected. Variance > 50% from mean triggers "
                "workload rebalancing review."
            ),
        },
        "clinical_relevance": {
            "IEC_62304": (
                "Software lifecycle traceability: appointment status transitions "
                "(booked -> confirmed -> completed) are logged as auditable events, "
                "satisfying IEC 62304 Class B software documentation requirements."
            ),
            "HIPAA": (
                "Patient visit records are PHI. Access is logged in "
                "transaction_log with actor, component, and timestamp for "
                "HIPAA audit trail compliance (45 CFR 164.312(b))."
            ),
            "CMS": (
                "CMS Merit-based Incentive Payment System (MIPS) tracks "
                "patient encounter volume and no-show rates as quality "
                "measures. Dashboard data supports MIPS reporting."
            ),
        },
        "remediation_strategies": [
            {
                "issue": "High no-show rate (> 15%)",
                "strategies": [
                    "Implement automated SMS/email reminders 48h and 24h before appointment",
                    "Offer same-day rescheduling for cancellations",
                    "Track repeat no-show patients and flag for outreach",
                    "Analyze no-show patterns by day-of-week and time-of-day",
                ],
            },
            {
                "issue": "Low completion rate (< 80%)",
                "strategies": [
                    "Review cancellation reasons for systemic issues",
                    "Reduce time between booking and appointment date",
                    "Implement waitlist backfill for cancelled slots",
                    "Ensure appointment confirmation workflow is in place",
                ],
            },
            {
                "issue": "Uneven provider workload",
                "strategies": [
                    "Rebalance new-patient assignments across providers",
                    "Align scheduling templates with provider availability",
                    "Monitor provider burnout indicators alongside volume",
                ],
            },
            {
                "issue": "Long appointment durations (> 90 min)",
                "strategies": [
                    "Pre-visit questionnaire to streamline intake",
                    "Ensure prior records are available before the visit",
                    "Identify if documentation burden is extending visit time",
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== Overview ===")
    print(json.dumps(overview(), indent=2))

    print("\n=== Breakdown (summary) ===")
    bd = breakdown()
    for key, val in bd.items():
        if isinstance(val, list):
            print(f"  {key}: {len(val)} items")
        else:
            print(f"  {key}: {val}")

    print("\n=== Definitions (keys) ===")
    defs = definitions()
    for key in defs:
        print(f"  {key}: {list(defs[key].keys()) if isinstance(defs[key], dict) else f'{len(defs[key])} items'}")
