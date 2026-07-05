"""Patient Education Dashboard — module completion, quiz performance, engagement analytics.

All data from REAL education_modules table in data/clinical.db.
Columns: id, patient_id, module_name, completion_pct, quiz_score, time_spent_minutes,
         started_at, completed_at, format, created_at.
Joined with patients table (patient_id, name, age, sex, diagnosis) for patient context.
"""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _scalar(query, params=()):
    with _conn() as c:
        row = c.execute(query, params).fetchone()
        return row[0] if row else None


def overview():
    """KPIs + topic/format breakdown + completion distribution + monthly trend."""
    total = _scalar("SELECT COUNT(*) FROM education_modules")
    if not total:
        return {"total_modules": 0, "unique_patients": 0, "message": "No education data yet"}

    unique_patients = _scalar("SELECT COUNT(DISTINCT patient_id) FROM education_modules")
    unique_topics = _scalar("SELECT COUNT(DISTINCT module_name) FROM education_modules")
    avg_completion = _scalar("SELECT AVG(completion_pct) FROM education_modules")
    avg_quiz = _scalar("SELECT AVG(quiz_score) FROM education_modules WHERE quiz_score IS NOT NULL")
    avg_time = _scalar("SELECT AVG(time_spent_minutes) FROM education_modules WHERE time_spent_minutes IS NOT NULL")
    completed_count = _scalar("SELECT COUNT(*) FROM education_modules WHERE completion_pct = 100")

    by_topic = _rows("""
        SELECT module_name AS topic,
               COUNT(*) AS count,
               ROUND(AVG(completion_pct), 1) AS avg_completion,
               ROUND(AVG(quiz_score), 1) AS avg_quiz_score,
               ROUND(AVG(time_spent_minutes), 1) AS avg_time
        FROM education_modules
        GROUP BY module_name
        ORDER BY count DESC
    """)

    by_format = _rows("""
        SELECT format,
               COUNT(*) AS count,
               ROUND(AVG(completion_pct), 1) AS avg_completion,
               ROUND(AVG(quiz_score), 1) AS avg_quiz_score
        FROM education_modules
        GROUP BY format
        ORDER BY count DESC
    """)

    completion_distribution = _rows("""
        SELECT
            CASE
                WHEN completion_pct BETWEEN 0 AND 25 THEN '0-25'
                WHEN completion_pct BETWEEN 26 AND 50 THEN '26-50'
                WHEN completion_pct BETWEEN 51 AND 75 THEN '51-75'
                WHEN completion_pct BETWEEN 76 AND 100 THEN '76-100'
            END AS range,
            COUNT(*) AS count
        FROM education_modules
        WHERE completion_pct IS NOT NULL
        GROUP BY range
        ORDER BY range
    """)

    monthly_trend = _rows("""
        SELECT month, new_starts, COALESCE(completions, 0) AS completions
        FROM (
            SELECT SUBSTR(started_at, 1, 7) AS month,
                   COUNT(*) AS new_starts
            FROM education_modules
            WHERE started_at IS NOT NULL
            GROUP BY SUBSTR(started_at, 1, 7)
        ) s
        LEFT JOIN (
            SELECT SUBSTR(completed_at, 1, 7) AS cmonth,
                   COUNT(*) AS completions
            FROM education_modules
            WHERE completed_at IS NOT NULL
            GROUP BY SUBSTR(completed_at, 1, 7)
        ) c ON s.month = c.cmonth
        ORDER BY month
    """)

    return {
        "total_modules": total,
        "unique_patients": unique_patients,
        "unique_topics": unique_topics,
        "avg_completion_pct": round(avg_completion, 1) if avg_completion is not None else 0,
        "avg_quiz_score": round(avg_quiz, 1) if avg_quiz is not None else 0,
        "avg_time_minutes": round(avg_time, 1) if avg_time is not None else 0,
        "completion_rate": round(completed_count / total, 3) if total else 0,
        "by_topic": by_topic,
        "by_format": by_format,
        "completion_distribution": completion_distribution,
        "monthly_trend": monthly_trend,
    }


def breakdown():
    """Per-patient, per-topic-format, quiz performance, engagement, at-risk patients."""
    total = _scalar("SELECT COUNT(*) FROM education_modules")
    if not total:
        return {"per_patient": [], "per_topic_format": [], "quiz_performance": [],
                "engagement_by_format": [], "at_risk_patients": []}

    per_patient = _rows("""
        SELECT e.patient_id,
               COALESCE(p.name, e.patient_id) AS name,
               COUNT(*) AS modules_started,
               SUM(CASE WHEN e.completion_pct = 100 THEN 1 ELSE 0 END) AS modules_completed,
               ROUND(AVG(e.quiz_score), 1) AS avg_quiz_score,
               ROUND(AVG(e.completion_pct), 1) AS avg_completion,
               COALESCE(SUM(e.time_spent_minutes), 0) AS total_time
        FROM education_modules e
        LEFT JOIN patients p ON e.patient_id = p.patient_id
        GROUP BY e.patient_id
        ORDER BY avg_completion DESC
    """)

    per_topic_format = _rows("""
        SELECT module_name AS topic,
               format,
               COUNT(*) AS count,
               ROUND(AVG(completion_pct), 1) AS avg_completion,
               ROUND(AVG(quiz_score), 1) AS avg_quiz_score
        FROM education_modules
        GROUP BY module_name, format
        ORDER BY module_name, format
    """)

    quiz_performance = _rows("""
        SELECT module_name AS topic,
               ROUND(AVG(quiz_score), 1) AS avg_score,
               ROUND(MIN(quiz_score), 1) AS min_score,
               ROUND(MAX(quiz_score), 1) AS max_score,
               SUM(CASE WHEN quiz_score >= 70 THEN 1 ELSE 0 END) AS pass_count,
               SUM(CASE WHEN quiz_score < 70 THEN 1 ELSE 0 END) AS fail_count
        FROM education_modules
        WHERE quiz_score IS NOT NULL
        GROUP BY module_name
        ORDER BY avg_score DESC
    """)

    engagement_by_format = _rows("""
        SELECT format,
               ROUND(AVG(time_spent_minutes), 1) AS avg_time,
               ROUND(AVG(completion_pct), 1) AS avg_completion,
               COUNT(*) AS count
        FROM education_modules
        GROUP BY format
        ORDER BY avg_time DESC
    """)

    at_risk_patients = _rows("""
        SELECT e.patient_id,
               COALESCE(p.name, e.patient_id) AS name,
               ROUND(AVG(e.completion_pct), 1) AS avg_completion,
               COUNT(*) AS modules_started,
               SUM(CASE WHEN e.completion_pct = 100 THEN 1 ELSE 0 END) AS modules_completed
        FROM education_modules e
        LEFT JOIN patients p ON e.patient_id = p.patient_id
        GROUP BY e.patient_id
        HAVING AVG(e.completion_pct) < 40
           OR SUM(CASE WHEN e.completion_pct = 100 THEN 1 ELSE 0 END) = 0
        ORDER BY avg_completion ASC
    """)

    return {
        "per_patient": per_patient,
        "per_topic_format": per_topic_format,
        "quiz_performance": quiz_performance,
        "engagement_by_format": engagement_by_format,
        "at_risk_patients": at_risk_patients,
    }


def definitions():
    """Education-related terminology and module descriptions."""
    terms = [
        {"term": "Completion Rate", "definition": "Fraction of education modules where a patient reached 100% completion."},
        {"term": "Quiz Score", "definition": "Score (0-100) on the knowledge-check quiz at the end of an education module."},
        {"term": "Pass Threshold", "definition": "A quiz score of 70 or above is considered a passing grade."},
        {"term": "Time Spent", "definition": "Total minutes a patient actively engaged with a given education module."},
        {"term": "At-Risk Patient", "definition": "A patient whose average completion is below 40% or who has zero completed modules."},
        {"term": "Module Format", "definition": "Delivery method of educational content: article (text-based), interactive (hands-on exercises), quiz (assessment-focused), or video (audio-visual)."},
        {"term": "Completion Percentage", "definition": "How far through a module a patient has progressed, from 0% (not started) to 100% (finished)."},
        {"term": "Engagement", "definition": "A composite measure of how actively a patient participates, combining time spent, completion, and quiz performance."},
        {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy — a critical topic in patient education for seizure disorder management."},
        {"term": "AED", "definition": "Anti-Epileptic Drug — medication used to control seizures; a core education module topic."},
        {"term": "Ketogenic Diet", "definition": "A high-fat, low-carbohydrate diet sometimes used as adjunct therapy for drug-resistant epilepsy."},
        {"term": "Seizure Diary", "definition": "A patient-maintained log of seizure events used for tracking patterns and treatment efficacy."},
    ]

    modules = _rows("SELECT DISTINCT module_name FROM education_modules ORDER BY module_name")

    descriptions_map = {
        "AED Basics": "Covers anti-epileptic drug classes, mechanisms of action, common side effects, and adherence strategies.",
        "Driving & Legal Rights": "Explains seizure-free driving requirements, state reporting laws, and legal protections for epilepsy patients.",
        "Emergency Preparedness": "Teaches patients and caregivers how to prepare for seizure emergencies including rescue medication use.",
        "Epilepsy Surgery Overview": "Introduces surgical options for drug-resistant epilepsy including resection, VNS, and RNS procedures.",
        "Ketogenic Diet": "Details the ketogenic and modified Atkins diets as therapeutic interventions for seizure reduction.",
        "Lifestyle & Triggers": "Identifies common seizure triggers such as sleep deprivation, stress, and alcohol, with mitigation strategies.",
        "Mental Health & Epilepsy": "Addresses comorbid depression and anxiety in epilepsy, screening tools, and treatment approaches.",
        "SUDEP Awareness": "Educates on Sudden Unexpected Death in Epilepsy risk factors, prevention strategies, and monitoring devices.",
        "Seizure Diary Training": "Guides patients on maintaining accurate seizure logs including timing, duration, type, and triggers.",
        "Seizure First Aid": "Teaches bystanders and caregivers proper seizure response including positioning, timing, and when to call 911.",
    }

    module_descriptions = []
    for m in modules:
        name = m["module_name"]
        desc = descriptions_map.get(name, f"Educational module covering {name.lower()} for epilepsy patients.")
        module_descriptions.append({"module_name": name, "description": desc})

    return {
        "terms": terms,
        "module_descriptions": module_descriptions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2))
    print("\n=== DEFINITIONS ===")
    print(json.dumps(definitions(), indent=2))
