"""Patient Appointments Dashboard — appointment analytics from clinical.db.

Tracks appointment scheduling, completion rates, no-show patterns,
provider workload, location usage, and duration distribution for epilepsy patients.

Sources:
- patient_appointments table (patient_id, appointment_type, provider_name,
  appointment_date, appointment_time, duration_minutes, status, location,
  reminder_sent, notes, created_at)
- 191 records, 30 patients, 8 appointment types, 6 providers, 5 statuses
"""

import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta

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


# ──────────────────────────────────────────────────────────────
#  /api/patient-appointments/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Aggregate appointment health: totals, rates, distributions,
    monthly trends, KPIs."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total_appointments = _safe(cur, "SELECT COUNT(*) FROM patient_appointments")
    total_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM patient_appointments")

    completed_count = _safe(cur, "SELECT COUNT(*) FROM patient_appointments WHERE status = 'completed'")
    no_show_count = _safe(cur, "SELECT COUNT(*) FROM patient_appointments WHERE status = 'no-show'")
    cancelled_count = _safe(cur, "SELECT COUNT(*) FROM patient_appointments WHERE status = 'cancelled'")
    reminder_sent_count = _safe(cur, "SELECT COUNT(*) FROM patient_appointments WHERE reminder_sent = 1")

    completion_rate = round(completed_count / total_appointments * 100, 1) if total_appointments else 0
    no_show_rate = round(no_show_count / total_appointments * 100, 1) if total_appointments else 0
    cancellation_rate = round(cancelled_count / total_appointments * 100, 1) if total_appointments else 0
    reminder_sent_rate = round(reminder_sent_count / total_appointments * 100, 1) if total_appointments else 0

    # Status distribution
    status_distribution = {}
    for row in _safe_rows(cur,
            """SELECT status, COUNT(*) as cnt
               FROM patient_appointments GROUP BY status ORDER BY cnt DESC"""):
        status_distribution[row[0]] = row[1]

    # Type distribution
    type_distribution = {}
    for row in _safe_rows(cur,
            """SELECT appointment_type, COUNT(*) as cnt
               FROM patient_appointments GROUP BY appointment_type ORDER BY cnt DESC"""):
        type_distribution[row[0]] = row[1]

    # Provider distribution
    provider_distribution = {}
    for row in _safe_rows(cur,
            """SELECT provider_name, COUNT(*) as cnt
               FROM patient_appointments GROUP BY provider_name ORDER BY cnt DESC"""):
        provider_distribution[row[0]] = row[1]

    # Location distribution
    location_distribution = {}
    for row in _safe_rows(cur,
            """SELECT location, COUNT(*) as cnt
               FROM patient_appointments GROUP BY location ORDER BY cnt DESC"""):
        location_distribution[row[0]] = row[1]

    # Duration distribution
    duration_distribution = {}
    for row in _safe_rows(cur,
            """SELECT duration_minutes, COUNT(*) as cnt
               FROM patient_appointments GROUP BY duration_minutes ORDER BY cnt DESC"""):
        duration_distribution[str(row[0]) + "min"] = row[1]

    # Monthly trend
    monthly_trend = []
    for row in _safe_rows(cur,
            """SELECT strftime('%Y-%m', appointment_date) as month, COUNT(*) as cnt
               FROM patient_appointments GROUP BY month ORDER BY month"""):
        monthly_trend.append({
            "month": row[0], "count": row[1]
        })

    conn.close()

    return {
        "available": True,
        "title": "Patient Appointments Dashboard",
        "total_appointments": total_appointments,
        "total_patients": total_patients,
        "completion_rate": completion_rate,
        "no_show_rate": no_show_rate,
        "cancellation_rate": cancellation_rate,
        "reminder_sent_rate": reminder_sent_rate,
        "status_distribution": status_distribution,
        "type_distribution": type_distribution,
        "provider_distribution": provider_distribution,
        "location_distribution": location_distribution,
        "duration_distribution": duration_distribution,
        "monthly_trend": monthly_trend
    }


# ──────────────────────────────────────────────────────────────
#  /api/patient-appointments/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient appointment matrix, upcoming/recent/no-show lists,
    and provider performance stats."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary
    per_patient = []
    for row in _safe_rows(cur,
            """SELECT patient_id,
                      COUNT(*) as total,
                      SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                      SUM(CASE WHEN status = 'scheduled' THEN 1 ELSE 0 END) as scheduled,
                      SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
                      SUM(CASE WHEN status = 'no-show' THEN 1 ELSE 0 END) as no_show,
                      SUM(CASE WHEN status = 'rescheduled' THEN 1 ELSE 0 END) as rescheduled
               FROM patient_appointments GROUP BY patient_id
               ORDER BY total DESC"""):
        patient_id = row[0]
        total = row[1]
        completed = row[2]
        completion_rate = round(completed / total * 100, 1) if total else 0

        # Top appointment type for this patient
        top_type_row = _safe_rows(cur,
            """SELECT appointment_type, COUNT(*) as cnt
               FROM patient_appointments WHERE patient_id = ?
               GROUP BY appointment_type ORDER BY cnt DESC LIMIT 1""", (patient_id,))
        top_type = top_type_row[0][0] if top_type_row else "N/A"

        # Top provider for this patient
        top_provider_row = _safe_rows(cur,
            """SELECT provider_name, COUNT(*) as cnt
               FROM patient_appointments WHERE patient_id = ?
               GROUP BY provider_name ORDER BY cnt DESC LIMIT 1""", (patient_id,))
        top_provider = top_provider_row[0][0] if top_provider_row else "N/A"

        per_patient.append({
            "patient_id": patient_id,
            "total": total,
            "completed": completed,
            "scheduled": row[3],
            "cancelled": row[4],
            "no_show": row[5],
            "rescheduled": row[6],
            "completion_rate": completion_rate,
            "top_type": top_type,
            "top_provider": top_provider
        })

    # Upcoming scheduled appointments
    upcoming = []
    for row in _safe_rows(cur,
            """SELECT patient_id, appointment_type, provider_name,
                      appointment_date, appointment_time, duration_minutes,
                      location, reminder_sent, notes
               FROM patient_appointments WHERE status = 'scheduled'
               ORDER BY appointment_date ASC, appointment_time ASC"""):
        upcoming.append({
            "patient_id": row[0], "appointment_type": row[1],
            "provider_name": row[2], "appointment_date": row[3],
            "appointment_time": row[4], "duration_minutes": row[5],
            "location": row[6], "reminder_sent": row[7],
            "notes": row[8]
        })

    # Recent completed appointments (last 15)
    recent_completed = []
    for row in _safe_rows(cur,
            """SELECT patient_id, appointment_type, provider_name,
                      appointment_date, appointment_time, duration_minutes,
                      location, notes
               FROM patient_appointments WHERE status = 'completed'
               ORDER BY appointment_date DESC LIMIT 15"""):
        recent_completed.append({
            "patient_id": row[0], "appointment_type": row[1],
            "provider_name": row[2], "appointment_date": row[3],
            "appointment_time": row[4], "duration_minutes": row[5],
            "location": row[6], "notes": row[7]
        })

    # No-show records
    no_shows = []
    for row in _safe_rows(cur,
            """SELECT patient_id, appointment_type, provider_name,
                      appointment_date, appointment_time, duration_minutes,
                      location, reminder_sent, notes
               FROM patient_appointments WHERE status = 'no-show'
               ORDER BY appointment_date DESC"""):
        no_shows.append({
            "patient_id": row[0], "appointment_type": row[1],
            "provider_name": row[2], "appointment_date": row[3],
            "appointment_time": row[4], "duration_minutes": row[5],
            "location": row[6], "reminder_sent": row[7],
            "notes": row[8]
        })

    # Provider stats
    provider_stats = []
    for row in _safe_rows(cur,
            """SELECT provider_name,
                      COUNT(*) as total,
                      SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                      SUM(CASE WHEN status = 'no-show' THEN 1 ELSE 0 END) as no_show,
                      ROUND(AVG(duration_minutes), 1) as avg_duration
               FROM patient_appointments GROUP BY provider_name
               ORDER BY total DESC"""):
        no_show_rate = round(row[3] / row[1] * 100, 1) if row[1] else 0
        provider_stats.append({
            "provider": row[0],
            "total": row[1],
            "completed": row[2],
            "no_show_rate": no_show_rate,
            "avg_duration": row[4]
        })

    conn.close()

    return {
        "available": True,
        "per_patient": per_patient,
        "upcoming": upcoming,
        "recent_completed": recent_completed,
        "no_shows": no_shows,
        "provider_stats": provider_stats
    }


# ──────────────────────────────────────────────────────────────
#  /api/patient-appointments/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Metric definitions, glossary, and reference notes for tooltip overlays."""
    return {
        "glossary": [
            {"term": "EEG Review", "definition": "Appointment to review electroencephalogram results with a neurologist, assessing brain wave patterns for seizure activity."},
            {"term": "VNS Check", "definition": "Vagus Nerve Stimulator follow-up to verify device settings, battery status, and therapeutic efficacy."},
            {"term": "Neuropsychology", "definition": "Cognitive and psychological assessment to evaluate memory, attention, and mood impacts of epilepsy and AED therapy."},
            {"term": "Neurology Follow-Up", "definition": "Routine visit with the neurologist to monitor seizure control, adjust medications, and review side effects."},
            {"term": "Medication Review", "definition": "Focused appointment to assess current AED regimen, dosing, drug interactions, and adherence."},
            {"term": "Epilepsy Surgery Consult", "definition": "Pre-surgical evaluation to determine candidacy for resective surgery, laser ablation, or neuromodulation."},
            {"term": "Diet Therapy Review", "definition": "Follow-up for ketogenic or modified Atkins diet therapy, reviewing ketone levels, nutrition, and seizure response."},
            {"term": "Telehealth Follow-Up", "definition": "Remote video consultation for routine monitoring, reducing travel burden for stable patients."},
            {"term": "No-Show Rate", "definition": "Percentage of appointments where the patient did not attend without prior cancellation. High no-show rates increase seizure risk due to missed care."},
            {"term": "Completion Rate", "definition": "Percentage of all appointments that were successfully completed. A key indicator of patient engagement and care continuity."},
            {"term": "Cancellation Rate", "definition": "Percentage of appointments cancelled by the patient or provider. Distinct from no-shows, as cancellations allow rescheduling."},
            {"term": "Reminder Sent", "definition": "Whether an automated reminder (SMS, email, or phone) was sent before the appointment. Reminders reduce no-show rates by 25-30%."},
            {"term": "Rescheduled", "definition": "Appointment that was moved to a new date/time rather than cancelled outright. Indicates patient intent to maintain care."},
            {"term": "Duration Minutes", "definition": "Scheduled length of the appointment in minutes. Ranges from 15-minute check-ins to 60-minute comprehensive evaluations."}
        ],
        "appointment_types": {
            "EEG Review": "Review of EEG recordings with the epileptologist to assess seizure focus, interictal discharges, and treatment response.",
            "Neuropsychology": "Comprehensive neuropsychological testing battery evaluating cognitive domains affected by epilepsy and AED side effects.",
            "Diet Therapy Review": "Ketogenic or modified Atkins diet follow-up including ketone monitoring, growth assessment, and dietary compliance.",
            "Medication Review": "AED regimen evaluation covering drug levels, efficacy, side effects, and potential interactions.",
            "VNS Check": "Vagus Nerve Stimulator device interrogation, output current adjustment, and magnet use review.",
            "Epilepsy Surgery Consult": "Multidisciplinary pre-surgical evaluation including video-EEG, MRI review, and neuropsychological assessment.",
            "Neurology Follow-Up": "Standard neurology visit for seizure frequency tracking, medication titration, and comorbidity management.",
            "Telehealth Follow-Up": "Virtual visit via secure video platform for stable patients, medication refills, and routine check-ins."
        },
        "status_definitions": {
            "completed": "Patient attended the appointment and the visit was documented in the medical record.",
            "scheduled": "Appointment is confirmed and upcoming. Patient has been notified.",
            "rescheduled": "Original appointment was moved to a new date/time at the request of the patient or provider.",
            "cancelled": "Appointment was cancelled in advance. No clinical encounter occurred.",
            "no-show": "Patient did not attend the scheduled appointment without prior notification. Flagged for follow-up."
        },
        "location_notes": {
            "Epilepsy Center Main": "Primary epilepsy monitoring unit with video-EEG capability, located within the hospital neuroscience wing.",
            "Telehealth": "Secure HIPAA-compliant video platform for remote consultations. Reduces travel burden for rural patients.",
            "Outpatient Clinic B": "General neurology outpatient facility for routine follow-ups, medication reviews, and brief consultations.",
            "Home Video": "Home-based video EEG monitoring setup for extended ambulatory recordings in the patient's natural environment."
        }
    }


if __name__ == '__main__':
    import pprint
    print("=== Overview ===")
    pprint.pprint(overview())
    print("\n=== Breakdown ===")
    pprint.pprint(breakdown())
    print("\n=== Definitions ===")
    pprint.pprint(definitions())
