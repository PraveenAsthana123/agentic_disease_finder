"""
Neuro AI Ecosystem — Caregivers Dashboard
==========================================
Caregiver registry and wellness analytics from caregivers table.

Roles: spouse, parent, sibling, child, friend, professional
Availability: full-time, part-time, on-call, weekends
Training: epilepsy training, first-aid certification, rescue-med trained
Wellness: stress, sleep quality, burnout score, work impact
Safety: safety plan, seizure action plan, emergency protocol

Real data: caregivers (30 rows, 30 patients) in clinical.db.
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """Caregivers overview — KPIs, role/availability distributions,
    training rates, wellness averages, burnout distribution."""
    conn = _conn()
    cur = conn.cursor()

    # Total caregivers
    cur.execute("SELECT COUNT(*) FROM caregivers")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM caregivers")
    patients_covered = cur.fetchone()[0]

    # Average experience
    cur.execute("SELECT ROUND(AVG(experience_years), 1) FROM caregivers")
    avg_experience = cur.fetchone()[0]

    # Training rates
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN epilepsy_training_completed = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) epilepsy_trained_pct,
               ROUND(AVG(CASE WHEN first_aid_certified = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) first_aid_pct,
               ROUND(AVG(CASE WHEN rescue_med_trained = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) rescue_med_pct,
               ROUND(AVG(CASE WHEN safety_plan_exists = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) safety_plan_pct,
               ROUND(AVG(CASE WHEN seizure_action_plan_exists = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) action_plan_pct
        FROM caregivers
    """)
    training = _dict_rows(cur)[0]

    # Wellness averages (scales 1-10)
    cur.execute("""
        SELECT ROUND(AVG(seizure_first_aid_confidence), 1) avg_confidence,
               ROUND(AVG(caregiver_stress), 1) avg_stress,
               ROUND(AVG(caregiver_sleep_quality), 1) avg_sleep,
               ROUND(AVG(work_impact), 1) avg_work_impact,
               ROUND(AVG(burnout_score), 1) avg_burnout
        FROM caregivers
    """)
    wellness = _dict_rows(cur)[0]

    # Role distribution
    cur.execute("""
        SELECT role, COUNT(*) cnt
        FROM caregivers
        GROUP BY role
        ORDER BY cnt DESC
    """)
    role_dist = _dict_rows(cur)

    # Availability distribution
    cur.execute("""
        SELECT availability, COUNT(*) cnt
        FROM caregivers
        GROUP BY availability
        ORDER BY cnt DESC
    """)
    availability_dist = _dict_rows(cur)

    # Burnout tiers
    cur.execute("""
        SELECT CASE
                 WHEN burnout_score <= 25 THEN 'Low (0-25)'
                 WHEN burnout_score <= 50 THEN 'Moderate (26-50)'
                 WHEN burnout_score <= 75 THEN 'High (51-75)'
                 ELSE 'Critical (76-100)'
               END AS tier,
               COUNT(*) cnt
        FROM caregivers
        GROUP BY tier
        ORDER BY MIN(burnout_score)
    """)
    burnout_dist = _dict_rows(cur)

    # Training bar: each certification count
    training_counts = [
        {"name": "Epilepsy Training", "count": 0},
        {"name": "First Aid", "count": 0},
        {"name": "Rescue Med", "count": 0},
        {"name": "Safety Plan", "count": 0},
        {"name": "Action Plan", "count": 0},
    ]
    cur.execute("""
        SELECT SUM(epilepsy_training_completed), SUM(first_aid_certified),
               SUM(rescue_med_trained), SUM(safety_plan_exists),
               SUM(seizure_action_plan_exists)
        FROM caregivers
    """)
    row = cur.fetchone()
    for i, v in enumerate(row):
        training_counts[i]["count"] = v or 0
        training_counts[i]["total"] = total

    # Stress vs burnout by role
    cur.execute("""
        SELECT role,
               ROUND(AVG(caregiver_stress), 1) avg_stress,
               ROUND(AVG(burnout_score), 1) avg_burnout,
               ROUND(AVG(seizure_first_aid_confidence), 1) avg_confidence,
               COUNT(*) cnt
        FROM caregivers
        GROUP BY role
        ORDER BY AVG(burnout_score) DESC
    """)
    role_wellness = _dict_rows(cur)

    conn.close()
    return {
        "kpis": {
            "total_caregivers": total,
            "patients_covered": patients_covered,
            "avg_experience_years": avg_experience,
            "epilepsy_trained_pct": training["epilepsy_trained_pct"],
            "first_aid_pct": training["first_aid_pct"],
            "rescue_med_pct": training["rescue_med_pct"],
            "avg_burnout": wellness["avg_burnout"],
            "avg_stress": wellness["avg_stress"],
            "avg_confidence": wellness["avg_confidence"],
        },
        "role_distribution": role_dist,
        "availability_distribution": availability_dist,
        "burnout_distribution": burnout_dist,
        "training_counts": training_counts,
        "role_wellness": role_wellness,
    }


def breakdown():
    """Caregivers breakdown — all caregivers table, by role, by availability,
    high-burnout caregivers needing intervention."""
    conn = _conn()
    cur = conn.cursor()

    # All caregivers
    cur.execute("""
        SELECT id, patient_id, name, role, availability, experience_years,
               epilepsy_training_completed, first_aid_certified, rescue_med_trained,
               seizure_first_aid_confidence, caregiver_stress, caregiver_sleep_quality,
               work_impact, burnout_score, safety_plan_exists, seizure_action_plan_exists,
               last_respite_date, notes, created_at
        FROM caregivers
        ORDER BY patient_id
    """)
    all_caregivers = _dict_rows(cur)

    # By role summary
    cur.execute("""
        SELECT role,
               COUNT(*) total,
               ROUND(AVG(experience_years), 1) avg_experience,
               ROUND(AVG(burnout_score), 1) avg_burnout,
               ROUND(AVG(caregiver_stress), 1) avg_stress,
               ROUND(AVG(seizure_first_aid_confidence), 1) avg_confidence,
               SUM(CASE WHEN epilepsy_training_completed = 1 THEN 1 ELSE 0 END) trained_count,
               SUM(CASE WHEN first_aid_certified = 1 THEN 1 ELSE 0 END) first_aid_count
        FROM caregivers
        GROUP BY role
        ORDER BY total DESC
    """)
    by_role = _dict_rows(cur)

    # By availability summary
    cur.execute("""
        SELECT availability,
               COUNT(*) total,
               ROUND(AVG(experience_years), 1) avg_experience,
               ROUND(AVG(burnout_score), 1) avg_burnout,
               ROUND(AVG(caregiver_stress), 1) avg_stress,
               SUM(CASE WHEN epilepsy_training_completed = 1 THEN 1 ELSE 0 END) trained_count
        FROM caregivers
        GROUP BY availability
        ORDER BY total DESC
    """)
    by_availability = _dict_rows(cur)

    # High-burnout caregivers (burnout > 60)
    cur.execute("""
        SELECT patient_id, name, role, availability, burnout_score,
               caregiver_stress, caregiver_sleep_quality, work_impact,
               last_respite_date,
               CAST(JULIANDAY('now') - JULIANDAY(last_respite_date) AS INTEGER) days_since_respite
        FROM caregivers
        WHERE burnout_score > 60
        ORDER BY burnout_score DESC
    """)
    high_burnout = _dict_rows(cur)

    conn.close()
    return {
        "all_caregivers": all_caregivers,
        "by_role": by_role,
        "by_availability": by_availability,
        "high_burnout": high_burnout,
    }


def definitions():
    """Caregivers definitions — field glossary, role descriptions,
    wellness scales, training requirements."""
    return {
        "glossary": [
            {"term": "Caregiver", "definition": "Person providing regular care and support to an epilepsy patient, either professionally or as family/friend"},
            {"term": "Epilepsy Training", "definition": "Formal training on seizure recognition, triggers, medication management, and emergency response specific to epilepsy"},
            {"term": "First Aid Certified", "definition": "Current certification in general first aid (CPR, wound care, emergency response)"},
            {"term": "Rescue Medication Trained", "definition": "Trained to administer emergency seizure rescue medications (e.g., nasal midazolam, buccal diazepam)"},
            {"term": "Seizure First-Aid Confidence", "definition": "Self-rated confidence (1-10) in ability to manage a seizure event safely"},
            {"term": "Caregiver Stress", "definition": "Self-rated stress level (1-10) related to caregiving responsibilities"},
            {"term": "Sleep Quality", "definition": "Self-rated sleep quality (1-10); lower scores indicate sleep disruption from caregiving duties"},
            {"term": "Work Impact", "definition": "Self-rated impact on professional work (1-10); higher scores indicate greater career disruption"},
            {"term": "Burnout Score", "definition": "Composite burnout index (0-100) based on stress, sleep quality, work impact, and respite frequency; >60 flags need for intervention"},
            {"term": "Safety Plan", "definition": "Written plan identifying home hazards and mitigations for seizure-related injuries"},
            {"term": "Seizure Action Plan", "definition": "Step-by-step protocol for caregiver to follow during and after a seizure event"},
            {"term": "Respite Date", "definition": "Date of last respite break; regular respite reduces burnout and improves care quality"},
        ],
        "roles": [
            {"role": "spouse", "description": "Married or domestic partner providing daily care"},
            {"role": "parent", "description": "Biological or adoptive parent of the patient"},
            {"role": "sibling", "description": "Brother or sister providing regular caregiving support"},
            {"role": "child", "description": "Adult child providing care to parent with epilepsy"},
            {"role": "friend", "description": "Close friend taking on caregiving responsibilities"},
            {"role": "professional", "description": "Paid caregiver, nurse, or home health aide with formal training"},
        ],
        "wellness_thresholds": [
            {"metric": "Burnout Score", "low": "0-25", "moderate": "26-50", "high": "51-75", "critical": "76-100"},
            {"metric": "Stress Level", "low": "1-3", "moderate": "4-6", "high": "7-8", "critical": "9-10"},
            {"metric": "Confidence", "low": "1-3 (needs training)", "moderate": "4-6", "high": "7-8", "critical": "9-10 (expert)"},
        ],
    }
