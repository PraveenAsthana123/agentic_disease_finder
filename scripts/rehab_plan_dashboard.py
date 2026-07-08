"""
Neuro AI Ecosystem — Rehab Plan Dashboard
==========================================
Tracks Occupational Therapist rehab plan progress across patients:
goal categories (ADL restoration, cognitive rehab, mobility training,
vocational rehab, social skills, fine motor), session adherence,
progress tracking, and target date management. Surfaces completion
rates, category distribution, session adherence, and upcoming targets
to support clinical oversight and rehabilitation outcomes.

Data Source:
  rehab_plans table in clinical.db — ~300 rows, 30 patients (P001-P030).

Goal Categories:
  adl_restoration   — Activities of Daily Living restoration
  cognitive_rehab   — Cognitive rehabilitation exercises
  mobility_training — Mobility and gait training
  vocational_rehab  — Vocational rehabilitation and work readiness
  social_skills     — Social skills development
  fine_motor        — Fine motor skills training

Status Values:
  active        — Currently being worked on
  completed     — Goal achieved
  on_hold       — Temporarily paused
  discontinued  — No longer pursued

Author: Research Team
"""

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

_CATEGORY_LABELS = {
    "adl_restoration": "ADL Restoration",
    "cognitive_rehab": "Cognitive Rehab",
    "mobility_training": "Mobility Training",
    "vocational_rehab": "Vocational Rehab",
    "social_skills": "Social Skills",
    "fine_motor": "Fine Motor",
}

_STATUS_LABELS = {
    "active": "Active",
    "completed": "Completed",
    "on_hold": "On Hold",
    "discontinued": "Discontinued",
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _patient_filter(patient_id: Optional[str]):
    """Return (where_clause, params) tuple for optional patient filtering."""
    if patient_id:
        return "WHERE rp.patient_id = ?", (patient_id,)
    return "", ()


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def overview(patient_id: Optional[str] = None) -> dict:
    """
    Rehab plan overview dashboard data.

    Returns:
        kpis                        — key performance indicators
        category_distribution       — pie chart: count per goal_category
        status_distribution         — pie chart: count per status
        progress_trend              — line chart: avg progress by month
        completion_rate_by_category — bar chart: completion rate per category
    """
    conn = _conn()
    c = conn.cursor()
    where, params = _patient_filter(patient_id)

    # ── KPIs ─────────────────────────────────────────────────────────
    c.execute(f"""
        SELECT
            COUNT(*)                                           AS total_plans,
            SUM(CASE WHEN rp.status = 'active' THEN 1 ELSE 0 END)      AS active_plans,
            SUM(CASE WHEN rp.status = 'completed' THEN 1 ELSE 0 END)   AS completed_plans,
            COALESCE(AVG(rp.progress_pct), 0)                  AS avg_progress_pct,
            COALESCE(
                CASE WHEN SUM(rp.sessions_planned) > 0
                     THEN CAST(SUM(rp.sessions_completed) AS REAL) / SUM(rp.sessions_planned) * 100
                     ELSE 0
                END, 0
            ) AS avg_sessions_completion_rate
        FROM rehab_plans rp
        {where}
    """, params)
    row = c.fetchone()
    total_plans = row[0] or 0
    active_plans = row[1] or 0
    completed_plans = row[2] or 0
    avg_progress_pct = round(float(row[3]), 1)
    avg_sessions_completion_rate = round(float(row[4]), 1)

    kpis = [
        {"label": "Total Plans",                 "value": total_plans,                          "sub": "rehab plans recorded"},
        {"label": "Active Plans",                "value": active_plans,                         "sub": "currently in progress"},
        {"label": "Completed Plans",             "value": completed_plans,                      "sub": "goals achieved"},
        {"label": "Avg Progress",                "value": f"{avg_progress_pct}%",               "sub": "mean progress_pct"},
        {"label": "Session Completion Rate",     "value": f"{avg_sessions_completion_rate}%",   "sub": "sessions_completed / sessions_planned"},
    ]

    # ── Category Distribution ────────────────────────────────────────
    c.execute(f"""
        SELECT rp.goal_category, COUNT(*) AS cnt
        FROM rehab_plans rp
        {where}
        GROUP BY rp.goal_category
        ORDER BY cnt DESC
    """, params)
    category_distribution = [
        {"category": _CATEGORY_LABELS.get(r[0], r[0]), "count": r[1]}
        for r in c.fetchall()
    ]

    # ── Status Distribution ──────────────────────────────────────────
    c.execute(f"""
        SELECT rp.status, COUNT(*) AS cnt
        FROM rehab_plans rp
        {where}
        GROUP BY rp.status
        ORDER BY cnt DESC
    """, params)
    status_distribution = [
        {"status": _STATUS_LABELS.get(r[0], r[0]), "count": r[1]}
        for r in c.fetchall()
    ]

    # ── Progress Trend (avg progress by month) ───────────────────────
    c.execute(f"""
        SELECT strftime('%Y-%m', rp.last_updated) AS month,
               AVG(rp.progress_pct) AS avg_progress
        FROM rehab_plans rp
        {where}
        GROUP BY month
        ORDER BY month
    """, params)
    progress_trend = [
        {"month": r[0], "avg_progress": round(float(r[1]), 1)}
        for r in c.fetchall()
    ]

    # ── Completion Rate by Category ──────────────────────────────────
    c.execute(f"""
        SELECT rp.goal_category,
               CASE WHEN COUNT(*) > 0
                    THEN ROUND(
                        CAST(SUM(CASE WHEN rp.status = 'completed' THEN 1 ELSE 0 END) AS REAL)
                        / COUNT(*) * 100, 1)
                    ELSE 0
               END AS rate
        FROM rehab_plans rp
        {where}
        GROUP BY rp.goal_category
        ORDER BY rate DESC
    """, params)
    completion_rate_by_category = [
        {"category": _CATEGORY_LABELS.get(r[0], r[0]), "rate": float(r[1])}
        for r in c.fetchall()
    ]

    conn.close()

    return {
        "kpis": kpis,
        "category_distribution": category_distribution,
        "status_distribution": status_distribution,
        "progress_trend": progress_trend,
        "completion_rate_by_category": completion_rate_by_category,
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed rehab plan breakdown.

    Returns:
        patient_summary    — per-patient stats table
        recent_updates     — last 20 updated plans
        session_adherence  — per-patient session adherence bar chart
        upcoming_targets   — active plans with upcoming target dates
    """
    conn = _conn()
    c = conn.cursor()
    where, params = _patient_filter(patient_id)

    # ── Patient Summary ──────────────────────────────────────────────
    c.execute(f"""
        SELECT
            rp.patient_id,
            COUNT(*)                                                    AS total_plans,
            SUM(CASE WHEN rp.status = 'active' THEN 1 ELSE 0 END)     AS active,
            SUM(CASE WHEN rp.status = 'completed' THEN 1 ELSE 0 END)  AS completed,
            ROUND(AVG(rp.progress_pct), 1)                             AS avg_progress
        FROM rehab_plans rp
        {where}
        GROUP BY rp.patient_id
        ORDER BY rp.patient_id
    """, params)
    patient_summary = [
        {
            "patient_id": r[0],
            "total_plans": r[1],
            "active": r[2] or 0,
            "completed": r[3] or 0,
            "avg_progress": float(r[4]) if r[4] is not None else 0.0,
        }
        for r in c.fetchall()
    ]

    # ── Recent Updates (last 20) ─────────────────────────────────────
    c.execute(f"""
        SELECT rp.patient_id, rp.goal_category, rp.goal_description,
               rp.status, rp.progress_pct, rp.last_updated
        FROM rehab_plans rp
        {where}
        ORDER BY rp.last_updated DESC
        LIMIT 20
    """, params)
    recent_updates = [
        {
            "patient_id": r[0],
            "goal_category": _CATEGORY_LABELS.get(r[1], r[1]),
            "goal_description": r[2],
            "status": _STATUS_LABELS.get(r[3], r[3]),
            "progress_pct": r[4],
            "last_updated": r[5],
        }
        for r in c.fetchall()
    ]

    # ── Session Adherence (per patient) ──────────────────────────────
    c.execute(f"""
        SELECT
            rp.patient_id,
            SUM(rp.sessions_planned)   AS planned,
            SUM(rp.sessions_completed) AS completed,
            CASE WHEN SUM(rp.sessions_planned) > 0
                 THEN ROUND(
                     CAST(SUM(rp.sessions_completed) AS REAL)
                     / SUM(rp.sessions_planned) * 100, 1)
                 ELSE 0
            END AS rate
        FROM rehab_plans rp
        {where}
        GROUP BY rp.patient_id
        ORDER BY rp.patient_id
    """, params)
    session_adherence = [
        {
            "patient_id": r[0],
            "planned": r[1] or 0,
            "completed": r[2] or 0,
            "rate": float(r[3]) if r[3] is not None else 0.0,
        }
        for r in c.fetchall()
    ]

    # ── Upcoming Targets (active plans with future target dates) ─────
    c.execute(f"""
        SELECT rp.patient_id, rp.goal_category, rp.goal_description,
               rp.target_date, rp.progress_pct, rp.sessions_planned,
               rp.sessions_completed
        FROM rehab_plans rp
        {where}
        {"AND" if where else "WHERE"} rp.status = 'active'
            AND rp.target_date >= date('now')
        ORDER BY rp.target_date ASC
    """, params)
    upcoming_targets = [
        {
            "patient_id": r[0],
            "goal_category": _CATEGORY_LABELS.get(r[1], r[1]),
            "goal_description": r[2],
            "target_date": r[3],
            "progress_pct": r[4],
            "sessions_planned": r[5] or 0,
            "sessions_completed": r[6] or 0,
        }
        for r in c.fetchall()
    ]

    conn.close()

    return {
        "patient_summary": patient_summary,
        "recent_updates": recent_updates,
        "session_adherence": session_adherence,
        "upcoming_targets": upcoming_targets,
    }


def definitions() -> dict:
    """
    Metric definitions, goal category descriptions, status meanings,
    and clinical glossary for the Rehab Plan Dashboard.
    """
    return {
        "metrics": [
            {
                "name": "Total Plans",
                "description": (
                    "Total number of rehab plan records across all patients. "
                    "Each row represents one rehabilitation goal with its associated "
                    "progress tracking and session data."
                ),
            },
            {
                "name": "Active Plans",
                "description": (
                    "Number of rehab plans with status = 'active', indicating goals "
                    "currently being worked on by the patient and therapist."
                ),
            },
            {
                "name": "Completed Plans",
                "description": (
                    "Number of rehab plans with status = 'completed', indicating "
                    "goals that have been fully achieved."
                ),
            },
            {
                "name": "Avg Progress",
                "description": (
                    "Mean progress_pct across all plans. Represents overall advancement "
                    "toward rehabilitation goals across the patient population."
                ),
            },
            {
                "name": "Session Completion Rate",
                "description": (
                    "Ratio of total sessions_completed to total sessions_planned across "
                    "all plans, expressed as a percentage. Indicates overall session "
                    "adherence for rehabilitation appointments."
                ),
            },
        ],
        "categories": {
            "adl_restoration": (
                "Activities of Daily Living (ADL) Restoration — Retraining in self-care "
                "tasks such as dressing, bathing, grooming, feeding, and toileting. "
                "Focuses on restoring independence in everyday functional activities."
            ),
            "cognitive_rehab": (
                "Cognitive Rehabilitation — Structured exercises targeting attention, "
                "memory, executive function, and problem-solving skills. Addresses "
                "cognitive deficits resulting from neurological conditions."
            ),
            "mobility_training": (
                "Mobility Training — Gait training, transfer skills, wheelchair mobility, "
                "and balance exercises. Aims to improve safe and independent movement "
                "within the patient's environment."
            ),
            "vocational_rehab": (
                "Vocational Rehabilitation — Work readiness assessment, job skills training, "
                "workplace adaptation, and return-to-work planning. Supports patients in "
                "achieving meaningful employment outcomes."
            ),
            "social_skills": (
                "Social Skills Development — Interventions targeting social interaction, "
                "communication, emotional regulation, and community participation. "
                "Addresses social functioning impacted by neurological conditions."
            ),
            "fine_motor": (
                "Fine Motor Skills Training — Hand dexterity, grip strength, hand-eye "
                "coordination, and precision manipulation exercises. Essential for "
                "writing, typing, buttoning, and other precise hand activities."
            ),
        },
        "statuses": {
            "active": (
                "The rehab goal is currently being actively worked on. The patient "
                "attends scheduled sessions and progress is being tracked."
            ),
            "completed": (
                "The rehabilitation goal has been fully achieved. The patient has "
                "met the target criteria defined in the goal description."
            ),
            "on_hold": (
                "The rehab goal is temporarily paused due to medical, logistical, "
                "or patient-related reasons. It may be resumed later."
            ),
            "discontinued": (
                "The rehab goal is no longer being pursued. This may be due to "
                "clinical reassessment, patient preference, or a change in the "
                "treatment plan."
            ),
        },
        "glossary": [
            {
                "term": "Rehab Plan",
                "definition": (
                    "A structured rehabilitation goal assigned to a patient by an "
                    "Occupational Therapist, including a target date, session schedule, "
                    "and progress tracking toward a specific functional outcome."
                ),
            },
            {
                "term": "Goal Category",
                "definition": (
                    "The clinical domain of the rehabilitation goal: ADL restoration, "
                    "cognitive rehab, mobility training, vocational rehab, social skills, "
                    "or fine motor training."
                ),
            },
            {
                "term": "Progress Percentage",
                "definition": (
                    "An integer 0-100 representing how far the patient has advanced "
                    "toward completing the rehabilitation goal. Updated by the therapist "
                    "at each session based on clinical assessment."
                ),
            },
            {
                "term": "Session Adherence",
                "definition": (
                    "The ratio of sessions_completed to sessions_planned for a given "
                    "rehab plan. Indicates how consistently the patient attends "
                    "scheduled therapy sessions."
                ),
            },
            {
                "term": "Target Date",
                "definition": (
                    "The projected date by which the rehabilitation goal should be "
                    "achieved, set collaboratively by the therapist and patient at "
                    "the start of the plan."
                ),
            },
            {
                "term": "Therapist Notes",
                "definition": (
                    "Free-text clinical notes recorded by the Occupational Therapist "
                    "at each session, documenting observations, progress, setbacks, "
                    "and plan adjustments."
                ),
            },
            {
                "term": "Occupational Therapy (OT)",
                "definition": (
                    "A healthcare profession focused on enabling individuals to "
                    "participate in meaningful daily activities (occupations) through "
                    "therapeutic intervention, adaptation, and environmental modification."
                ),
            },
            {
                "term": "Activities of Daily Living (ADL)",
                "definition": (
                    "Basic self-care tasks essential for independent living, including "
                    "bathing, dressing, grooming, feeding, toileting, and functional "
                    "mobility within the home."
                ),
            },
        ],
    }
