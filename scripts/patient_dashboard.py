"""Patient Dashboard — cross-domain KPIs + trends for the full patient cohort.

Aggregates data across patients, assessments, appointments, medications,
seizure diary, EEG analyses, and MRI findings to provide a single-pane
clinical overview with KPI cards, trends, and breakdowns.

Sources: clinical.db (patients, assessments, appointments, medications,
seizure_diary, analyses, mri_findings)
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


def patient_dashboard_overview():
    """KPI summary: patient count, gender/disease distribution, assessment
    coverage, appointment completion, medication/seizure/analysis counts."""
    conn = _conn()
    cur = conn.cursor()

    total_patients = _safe(cur, "SELECT COUNT(*) FROM patients")
    avg_age = _safe(cur, "SELECT AVG(age) FROM patients")
    gender_rows = _safe_rows(cur,
        "SELECT gender, COUNT(*) FROM patients GROUP BY gender ORDER BY COUNT(*) DESC")
    disease_rows = _safe_rows(cur,
        "SELECT disease, COUNT(*) FROM patients GROUP BY disease ORDER BY COUNT(*) DESC")

    # Assessment coverage
    total_assessments = _safe(cur, "SELECT COUNT(*) FROM assessments")
    patients_with_assessments = _safe(cur,
        "SELECT COUNT(DISTINCT patient_id) FROM assessments")
    unique_instruments = _safe(cur,
        "SELECT COUNT(DISTINCT instrument) FROM assessments")
    avg_score_pct = _safe(cur,
        "SELECT AVG(CAST(score AS REAL) / NULLIF(max_score, 0) * 100) FROM assessments "
        "WHERE max_score > 0")

    # Appointment stats
    total_appointments = _safe(cur, "SELECT COUNT(*) FROM appointments")
    completed_appointments = _safe(cur,
        "SELECT COUNT(*) FROM appointments WHERE status='completed'")
    completion_rate = round(completed_appointments / total_appointments * 100, 1) if total_appointments > 0 else 0

    # Medications
    total_prescriptions = _safe(cur, "SELECT COUNT(*) FROM medications")
    patients_on_meds = _safe(cur,
        "SELECT COUNT(DISTINCT patient_id) FROM medications")

    # Seizure diary
    total_seizure_events = _safe(cur, "SELECT COUNT(*) FROM seizure_diary")
    patients_with_seizures = _safe(cur,
        "SELECT COUNT(DISTINCT patient_id) FROM seizure_diary")

    # EEG analyses
    total_analyses = _safe(cur, "SELECT COUNT(*) FROM analyses")
    avg_confidence = _safe(cur,
        "SELECT AVG(confidence) FROM analyses WHERE confidence IS NOT NULL")

    # MRI findings
    total_mri = _safe(cur, "SELECT COUNT(*) FROM mri_findings")

    conn.close()
    return {
        "total_patients": total_patients,
        "avg_age": round(avg_age, 1) if avg_age else 0,
        "gender_distribution": [{"gender": g, "count": c} for g, c in gender_rows],
        "disease_distribution": [{"disease": d or "Unknown", "count": c} for d, c in disease_rows],
        "total_assessments": total_assessments,
        "patients_with_assessments": patients_with_assessments,
        "assessment_coverage_pct": round(patients_with_assessments / total_patients * 100, 1) if total_patients > 0 else 0,
        "unique_instruments": unique_instruments,
        "avg_score_pct": round(avg_score_pct, 1) if avg_score_pct else 0,
        "total_appointments": total_appointments,
        "completed_appointments": completed_appointments,
        "completion_rate_pct": completion_rate,
        "total_prescriptions": total_prescriptions,
        "patients_on_meds": patients_on_meds,
        "medication_coverage_pct": round(patients_on_meds / total_patients * 100, 1) if total_patients > 0 else 0,
        "total_seizure_events": total_seizure_events,
        "patients_with_seizures": patients_with_seizures,
        "total_analyses": total_analyses,
        "avg_confidence_pct": round(avg_confidence * 100, 1) if avg_confidence and avg_confidence <= 1 else (round(avg_confidence, 1) if avg_confidence else 0),
        "total_mri_findings": total_mri,
    }


def patient_dashboard_breakdown():
    """Detailed breakdowns: per-disease stats, severity distribution,
    appointment trends, assessment by instrument, per-patient summary,
    seizure severity, recent activity."""
    conn = _conn()
    cur = conn.cursor()

    # Per-disease patient counts + avg age
    disease_stats = []
    for row in _safe_rows(cur,
            "SELECT disease, COUNT(*), AVG(age) FROM patients GROUP BY disease ORDER BY COUNT(*) DESC"):
        disease, cnt, avg_a = row
        assessments_for = _safe(cur,
            "SELECT COUNT(*) FROM assessments WHERE patient_id IN "
            "(SELECT patient_id FROM patients WHERE disease=?)", (disease,))
        disease_stats.append({
            "disease": disease or "Unknown",
            "patients": cnt,
            "avg_age": round(avg_a, 1) if avg_a else 0,
            "assessments": assessments_for,
        })

    # Assessment by instrument
    instrument_stats = []
    for row in _safe_rows(cur,
            "SELECT instrument, COUNT(*), AVG(CAST(score AS REAL)/NULLIF(max_score,0)*100) "
            "FROM assessments WHERE max_score > 0 GROUP BY instrument ORDER BY COUNT(*) DESC"):
        instrument_stats.append({
            "instrument": row[0],
            "count": row[1],
            "avg_score_pct": round(row[2], 1) if row[2] else 0,
        })

    # Assessment severity distribution
    severity_dist = [{"level": r[0] or "unscored", "count": r[1]}
                     for r in _safe_rows(cur,
                         "SELECT level, COUNT(*) FROM assessments GROUP BY level ORDER BY COUNT(*) DESC")]

    # Appointment daily trend (last 30 days with data)
    daily_trend = [{"date": r[0], "total": r[1], "completed": r[2]}
                   for r in _safe_rows(cur,
                       "SELECT DATE(scheduled_for) as d, COUNT(*), "
                       "SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) "
                       "FROM appointments WHERE scheduled_for IS NOT NULL "
                       "GROUP BY d ORDER BY d DESC LIMIT 30")]
    daily_trend.reverse()

    # Seizure severity
    seizure_severity = [{"severity": r[0] or "unknown", "count": r[1]}
                        for r in _safe_rows(cur,
                            "SELECT severity, COUNT(*) FROM seizure_diary GROUP BY severity ORDER BY COUNT(*) DESC")]

    # Seizure triggers
    seizure_triggers = [{"trigger": r[0] or "unknown", "count": r[1]}
                        for r in _safe_rows(cur,
                            "SELECT trigger, COUNT(*) FROM seizure_diary GROUP BY trigger ORDER BY COUNT(*) DESC")]

    # Per-patient summary (top 30 by activity)
    patient_rows = _safe_rows(cur,
        "SELECT p.patient_id, p.name, p.age, p.gender, p.disease, "
        "(SELECT COUNT(*) FROM assessments a WHERE a.patient_id=p.patient_id), "
        "(SELECT COUNT(*) FROM appointments ap WHERE ap.patient_id=p.patient_id AND ap.status='completed'), "
        "(SELECT COUNT(*) FROM seizure_diary s WHERE s.patient_id=p.patient_id), "
        "(SELECT COUNT(*) FROM medications m WHERE m.patient_id=p.patient_id) "
        "FROM patients p ORDER BY "
        "((SELECT COUNT(*) FROM assessments a WHERE a.patient_id=p.patient_id) + "
        " (SELECT COUNT(*) FROM appointments ap WHERE ap.patient_id=p.patient_id) + "
        " (SELECT COUNT(*) FROM seizure_diary s WHERE s.patient_id=p.patient_id)) DESC "
        "LIMIT 30")
    patient_summary = [{
        "patient_id": r[0], "name": r[1], "age": r[2], "gender": r[3],
        "disease": r[4] or "Unknown", "assessments": r[5], "completed_visits": r[6],
        "seizure_events": r[7], "prescriptions": r[8],
    } for r in patient_rows]

    # EEG analysis results
    analysis_dist = [{"label": r[0] or "unknown", "count": r[1],
                      "avg_confidence": round(r[2] * 100, 1) if r[2] and r[2] <= 1 else (round(r[2], 1) if r[2] else 0)}
                     for r in _safe_rows(cur,
                         "SELECT predicted_label, COUNT(*), AVG(confidence) FROM analyses "
                         "GROUP BY predicted_label ORDER BY COUNT(*) DESC")]

    # Recent assessments (last 20)
    recent_assessments = [{
        "patient_id": r[0], "instrument": r[1], "score": r[2],
        "max_score": r[3], "level": r[4] or "—", "created_at": r[5],
    } for r in _safe_rows(cur,
        "SELECT patient_id, instrument, score, max_score, level, created_at "
        "FROM assessments ORDER BY created_at DESC LIMIT 20")]

    conn.close()
    return {
        "disease_stats": disease_stats,
        "instrument_stats": instrument_stats,
        "severity_distribution": severity_dist,
        "daily_appointment_trend": daily_trend,
        "seizure_severity": seizure_severity,
        "seizure_triggers": seizure_triggers,
        "patient_summary": patient_summary,
        "analysis_distribution": analysis_dist,
        "recent_assessments": recent_assessments,
    }


def patient_dashboard_definitions():
    """Clinical definitions and quality metrics for tooltip overlays."""
    return {
        "sections": [
            {
                "title": "Patient KPI Concepts",
                "items": [
                    {"term": "Assessment Coverage", "definition": "Percentage of patients who have at least one completed clinical assessment (e.g., GAD-7, PHQ-9, Barthel Index)."},
                    {"term": "Medication Coverage", "definition": "Percentage of patients with at least one active prescription in the system."},
                    {"term": "Completion Rate", "definition": "Percentage of scheduled appointments that reached 'completed' status (patient was seen)."},
                    {"term": "Seizure Event", "definition": "A recorded episode in the seizure diary, including severity, duration, triggers, and post-ictal state."},
                    {"term": "EEG Analysis Confidence", "definition": "Average model confidence score across all EEG classification analyses (ictal/interictal/normal)."},
                    {"term": "Severity Level", "definition": "Clinical severity assigned to an assessment score: normal, mild, moderate, moderately severe, or severe."},
                ]
            },
            {
                "title": "Quality Metrics",
                "items": [
                    {"term": "Average Score %", "definition": "Mean assessment score as a percentage of the instrument's maximum, across all scored assessments."},
                    {"term": "Polypharmacy", "definition": "Patients on 2+ concurrent medications — flagged for drug interaction review."},
                    {"term": "No-Show Rate", "definition": "Percentage of appointments where the patient did not attend."},
                    {"term": "Daily Trend", "definition": "Day-by-day count of total vs completed appointments, showing utilization patterns."},
                ]
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {"term": "ILAE Guidelines", "definition": "International League Against Epilepsy — classification, treatment, and monitoring standards for epilepsy care."},
                    {"term": "IEC 62304", "definition": "Medical device software lifecycle standard — requires traceability from clinical data to decision outputs."},
                    {"term": "CMS HEDIS", "definition": "Healthcare Effectiveness Data and Information Set — standardized measures for care quality including follow-up rates and medication adherence."},
                    {"term": "HIPAA", "definition": "Health Insurance Portability and Accountability Act — governs protected health information handling in all patient data views."},
                ]
            },
            {
                "title": "Remediation Strategies",
                "items": [
                    {"term": "Low Assessment Coverage", "definition": "Schedule intake assessments for unscreened patients. Set automated reminders for periodic re-assessment."},
                    {"term": "Low Completion Rate", "definition": "Review scheduling workflow, send appointment reminders, investigate no-show patterns by department/provider."},
                    {"term": "High Seizure Frequency", "definition": "Flag patients with >3 events/month for neurologist review. Check medication adherence and trigger avoidance."},
                    {"term": "Low Confidence Analyses", "definition": "Review signal quality of source EEG recordings. Consider re-recording or manual expert review for confidence <70%."},
                ]
            },
        ]
    }
