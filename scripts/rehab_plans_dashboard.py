"""
Neuro AI Ecosystem — Rehabilitation Plans Dashboard
====================================================
Rehabilitation goal-tracking analytics from rehab_plans table.

Goal categories: adl_restoration, cognitive_rehab, fine_motor,
    mobility_training, social_skills, vocational_rehab
Statuses: active, completed, on_hold, discontinued
Tracks: progress_pct, sessions_planned/completed, therapist notes

Real data: rehab_plans (311 rows, 30 patients) in clinical.db.
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """Rehab plans overview — totals, status distribution, goal category breakdown,
    average progress, session completion rate, monthly trend."""
    conn = _conn()
    cur = conn.cursor()

    # Totals
    cur.execute("SELECT COUNT(*) FROM rehab_plans")
    total_plans = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM rehab_plans")
    total_patients = cur.fetchone()[0]

    # Status distribution
    cur.execute("""
        SELECT status, COUNT(*) cnt
        FROM rehab_plans
        GROUP BY status
        ORDER BY cnt DESC
    """)
    status_dist = [{"name": r[0], "value": r[1]} for r in cur.fetchall()]

    # Goal category distribution
    cur.execute("""
        SELECT goal_category, COUNT(*) cnt
        FROM rehab_plans
        GROUP BY goal_category
        ORDER BY cnt DESC
    """)
    category_dist = [{"name": r[0], "value": r[1]} for r in cur.fetchall()]

    # Progress stats
    cur.execute("""
        SELECT
            ROUND(AVG(progress_pct), 1),
            ROUND(AVG(sessions_completed * 100.0 / NULLIF(sessions_planned, 0)), 1),
            SUM(sessions_planned),
            SUM(sessions_completed)
        FROM rehab_plans
    """)
    row = cur.fetchone()
    avg_progress = row[0]
    avg_session_rate = row[1]
    total_sessions_planned = row[2]
    total_sessions_completed = row[3]

    # Completion rate
    cur.execute("SELECT COUNT(*) FROM rehab_plans WHERE status = 'completed'")
    completed_count = cur.fetchone()[0]
    completion_rate = round(completed_count * 100.0 / total_plans, 1) if total_plans else 0

    # Progress distribution (buckets)
    cur.execute("""
        SELECT
            CASE
                WHEN progress_pct = 0 THEN '0%'
                WHEN progress_pct BETWEEN 1 AND 25 THEN '1-25%'
                WHEN progress_pct BETWEEN 26 AND 50 THEN '26-50%'
                WHEN progress_pct BETWEEN 51 AND 75 THEN '51-75%'
                WHEN progress_pct BETWEEN 76 AND 99 THEN '76-99%'
                ELSE '100%'
            END AS bucket,
            COUNT(*) cnt
        FROM rehab_plans
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '0%' THEN 1
                WHEN '1-25%' THEN 2
                WHEN '26-50%' THEN 3
                WHEN '51-75%' THEN 4
                WHEN '76-99%' THEN 5
                ELSE 6
            END
    """)
    progress_dist = [{"name": r[0], "value": r[1]} for r in cur.fetchall()]

    # Category × status cross-tab
    cur.execute("""
        SELECT goal_category, status, COUNT(*) cnt
        FROM rehab_plans
        GROUP BY goal_category, status
        ORDER BY goal_category, status
    """)
    cat_status = defaultdict(dict)
    for r in cur.fetchall():
        cat_status[r[0]][r[1]] = r[2]
    category_status = [
        {"category": cat, **statuses}
        for cat, statuses in sorted(cat_status.items())
    ]

    # Average progress by category
    cur.execute("""
        SELECT goal_category,
               ROUND(AVG(progress_pct), 1) avg_prog,
               COUNT(*) cnt
        FROM rehab_plans
        GROUP BY goal_category
        ORDER BY avg_prog DESC
    """)
    category_progress = [
        {"name": r[0], "avg_progress": r[1], "count": r[2]}
        for r in cur.fetchall()
    ]

    # Monthly trend (by start_date)
    cur.execute("""
        SELECT SUBSTR(start_date, 1, 7) AS month,
               COUNT(*) new_plans,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) completed,
               ROUND(AVG(progress_pct), 1) avg_progress
        FROM rehab_plans
        WHERE start_date IS NOT NULL
        GROUP BY month
        ORDER BY month
    """)
    monthly_trend = [
        {"month": r[0], "new_plans": r[1], "completed": r[2], "avg_progress": r[3]}
        for r in cur.fetchall()
    ]

    conn.close()
    return {
        "total_plans": total_plans,
        "total_patients": total_patients,
        "avg_progress": avg_progress,
        "avg_session_rate": avg_session_rate,
        "total_sessions_planned": total_sessions_planned,
        "total_sessions_completed": total_sessions_completed,
        "completion_rate": completion_rate,
        "status_dist": status_dist,
        "category_dist": category_dist,
        "progress_dist": progress_dist,
        "category_status": category_status,
        "category_progress": category_progress,
        "monthly_trend": monthly_trend,
    }


def breakdown():
    """Rehab plans breakdown — per-patient summary, per-category detail,
    on-hold/discontinued plans, high-performers, recent updates."""
    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary
    cur.execute("""
        SELECT patient_id,
               COUNT(*) total,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) completed,
               SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) active,
               SUM(CASE WHEN status = 'on_hold' THEN 1 ELSE 0 END) on_hold,
               SUM(CASE WHEN status = 'discontinued' THEN 1 ELSE 0 END) discontinued,
               ROUND(AVG(progress_pct), 1) avg_progress,
               ROUND(AVG(sessions_completed * 100.0 / NULLIF(sessions_planned, 0)), 1) session_rate
        FROM rehab_plans
        GROUP BY patient_id
        ORDER BY patient_id
    """)
    per_patient = _dict_rows(cur)

    # Per-category detail
    cur.execute("""
        SELECT goal_category,
               COUNT(*) total,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) completed,
               SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) active,
               ROUND(AVG(progress_pct), 1) avg_progress,
               ROUND(AVG(sessions_completed), 1) avg_sessions_done,
               ROUND(AVG(sessions_planned), 1) avg_sessions_planned
        FROM rehab_plans
        GROUP BY goal_category
        ORDER BY goal_category
    """)
    per_category = _dict_rows(cur)

    # On-hold / discontinued plans (attention needed)
    cur.execute("""
        SELECT id, patient_id, goal_category, goal_description, status,
               progress_pct, sessions_completed, sessions_planned, therapist_notes,
               last_updated
        FROM rehab_plans
        WHERE status IN ('on_hold', 'discontinued')
        ORDER BY last_updated DESC
    """)
    attention_plans = _dict_rows(cur)

    # High performers (progress >= 80, still active)
    cur.execute("""
        SELECT id, patient_id, goal_category, goal_description,
               progress_pct, sessions_completed, sessions_planned,
               therapist_notes, last_updated
        FROM rehab_plans
        WHERE status = 'active' AND progress_pct >= 80
        ORDER BY progress_pct DESC
    """)
    high_performers = _dict_rows(cur)

    # Low progress active plans (progress < 25, still active)
    cur.execute("""
        SELECT id, patient_id, goal_category, goal_description,
               progress_pct, sessions_completed, sessions_planned,
               therapist_notes, last_updated
        FROM rehab_plans
        WHERE status = 'active' AND progress_pct < 25
        ORDER BY progress_pct ASC
    """)
    low_progress = _dict_rows(cur)

    # Recently updated plans
    cur.execute("""
        SELECT id, patient_id, goal_category, goal_description, status,
               progress_pct, sessions_completed, sessions_planned,
               therapist_notes, last_updated
        FROM rehab_plans
        ORDER BY last_updated DESC
        LIMIT 20
    """)
    recent_updates = _dict_rows(cur)

    conn.close()
    return {
        "per_patient": per_patient,
        "per_category": per_category,
        "attention_plans": attention_plans,
        "high_performers": high_performers,
        "low_progress": low_progress,
        "recent_updates": recent_updates,
    }


def definitions():
    """Rehab plans definitions — goal category descriptions, status definitions,
    progress milestones, clinical glossary."""
    return {
        "goal_categories": {
            "adl_restoration": "Activities of Daily Living — restoring independence in self-care tasks (hygiene, dressing, feeding, toileting) that may be impaired by seizure-related injuries or post-ictal states.",
            "cognitive_rehab": "Cognitive Rehabilitation — targeted interventions for memory, attention, executive function, and processing speed deficits common in epilepsy patients, especially after temporal lobe resection.",
            "fine_motor": "Fine Motor Rehabilitation — restoring dexterity, hand-eye coordination, and manual precision impaired by seizure activity, AED side effects, or surgical intervention.",
            "mobility_training": "Mobility Training — gait retraining, balance exercises, fall prevention, and progressive ambulation programs for patients with seizure-related musculoskeletal injuries.",
            "social_skills": "Social Skills Training — rebuilding social confidence, communication skills, and community participation affected by seizure-related stigma, anxiety, or cognitive changes.",
            "vocational_rehab": "Vocational Rehabilitation — graduated return-to-work programs, workplace accommodation planning, and functional capacity evaluations for epilepsy patients.",
        },
        "statuses": {
            "active": "Plan is currently being executed with ongoing sessions and progress tracking.",
            "completed": "All goals achieved or target reached — plan closed successfully.",
            "on_hold": "Plan temporarily paused due to medical events (e.g., seizure cluster, medication change, hospitalization) — requires reassessment before resuming.",
            "discontinued": "Plan stopped before completion — patient declined, transferred, or goal deemed inappropriate after reassessment.",
        },
        "progress_milestones": {
            "0%": "Not started — plan created but no sessions conducted.",
            "1-25%": "Early phase — initial assessments and baseline established.",
            "26-50%": "Building phase — active skill acquisition and practice.",
            "51-75%": "Consolidation phase — generalizing skills to daily contexts.",
            "76-99%": "Near completion — final adjustments and independence verification.",
            "100%": "Goal achieved — ready for discharge from this rehab stream.",
        },
        "glossary": {
            "ADL": "Activities of Daily Living — basic self-care tasks (bathing, dressing, eating, toileting, transferring).",
            "IADL": "Instrumental Activities of Daily Living — complex tasks (cooking, managing finances, medication management, transportation).",
            "FIM": "Functional Independence Measure — 18-item assessment of disability severity and rehabilitation outcomes.",
            "Session Completion Rate": "Percentage of planned therapy sessions actually attended and completed by the patient.",
            "Progress Percentage": "Therapist-rated estimate of how close the patient is to achieving the stated rehabilitation goal.",
            "Post-ictal": "Period following a seizure characterized by confusion, fatigue, and temporary neurological deficits that may impact rehab participation.",
            "AED Side Effects": "Antiepileptic drug effects (drowsiness, tremor, cognitive slowing) that may complicate rehabilitation progress.",
            "Graduated Return": "Stepwise increase in activity level or work hours to safely transition from rehabilitation to full function.",
            "Goal Category": "Clinical domain of the rehabilitation plan — determines which therapy team leads and which outcome measures apply.",
            "Therapist Notes": "Free-text clinical observations recorded at each session — tracks qualitative progress, barriers, and plan modifications.",
        },
        "session_guidelines": {
            "typical_frequency": "2-5 sessions per week depending on goal category and patient tolerance.",
            "session_duration": "30-60 minutes per session; cognitive rehab sessions may be shorter due to fatigue.",
            "reassessment_interval": "Every 4-6 weeks or after any significant medical event (seizure cluster, medication change).",
        },
    }
