"""Daily Plans Dashboard — patient daily plan analytics from clinical.db.

Tracks daily plan adherence: activity completion rates (medication, meals,
exercise, sleep, mood, seizure logging), plan completion trends, AI suggestion
distribution, per-patient engagement scores, and date-range analysis.

Sources:
- daily_plans table (patient_id, plan_date, activity booleans, plan_completion_pct, ai_suggestion)
"""

import sqlite3
import datetime
from pathlib import Path
from collections import Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """High-level daily plan metrics: totals, activity rates, completion distribution,
    daily trend, activity breakdown."""
    conn = _conn()
    cur = conn.cursor()

    # totals
    cur.execute("SELECT COUNT(*) FROM daily_plans")
    total_plans = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM daily_plans")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT plan_date) FROM daily_plans")
    total_days = cur.fetchone()[0]

    # overall completion
    cur.execute("SELECT AVG(plan_completion_pct), MIN(plan_completion_pct), MAX(plan_completion_pct) FROM daily_plans")
    avg_completion, min_completion, max_completion = cur.fetchone()
    avg_completion = round(avg_completion, 1) if avg_completion else 0

    # activity totals
    cur.execute("""
        SELECT
            SUM(medication_reminders_set) as medication,
            SUM(meals_logged) as meals,
            SUM(exercise_logged) as exercise,
            SUM(sleep_logged) as sleep,
            SUM(mood_logged) as mood,
            SUM(seizure_logged) as seizure
        FROM daily_plans
    """)
    row = cur.fetchone()
    activity_totals = {
        "medication_reminders": row[0] or 0,
        "meals_logged": row[1] or 0,
        "exercise_logged": row[2] or 0,
        "sleep_logged": row[3] or 0,
        "mood_logged": row[4] or 0,
        "seizure_logged": row[5] or 0,
    }

    # activity rates (% of plans where each activity > 0)
    cur.execute("""
        SELECT
            ROUND(100.0 * SUM(CASE WHEN medication_reminders_set > 0 THEN 1 ELSE 0 END) / COUNT(*), 1),
            ROUND(100.0 * SUM(CASE WHEN meals_logged > 0 THEN 1 ELSE 0 END) / COUNT(*), 1),
            ROUND(100.0 * SUM(CASE WHEN exercise_logged > 0 THEN 1 ELSE 0 END) / COUNT(*), 1),
            ROUND(100.0 * SUM(CASE WHEN sleep_logged > 0 THEN 1 ELSE 0 END) / COUNT(*), 1),
            ROUND(100.0 * SUM(CASE WHEN mood_logged > 0 THEN 1 ELSE 0 END) / COUNT(*), 1),
            ROUND(100.0 * SUM(CASE WHEN seizure_logged > 0 THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM daily_plans
    """)
    rates = cur.fetchone()
    activity_rates = {
        "medication_reminders": rates[0] or 0,
        "meals_logged": rates[1] or 0,
        "exercise_logged": rates[2] or 0,
        "sleep_logged": rates[3] or 0,
        "mood_logged": rates[4] or 0,
        "seizure_logged": rates[5] or 0,
    }

    # completion distribution (buckets: 0-25, 26-50, 51-75, 76-100)
    cur.execute("""
        SELECT
            SUM(CASE WHEN plan_completion_pct <= 25 THEN 1 ELSE 0 END) as low,
            SUM(CASE WHEN plan_completion_pct > 25 AND plan_completion_pct <= 50 THEN 1 ELSE 0 END) as medium,
            SUM(CASE WHEN plan_completion_pct > 50 AND plan_completion_pct <= 75 THEN 1 ELSE 0 END) as high,
            SUM(CASE WHEN plan_completion_pct > 75 THEN 1 ELSE 0 END) as excellent
        FROM daily_plans
    """)
    dist = cur.fetchone()
    completion_distribution = {
        "0-25%": dist[0] or 0,
        "26-50%": dist[1] or 0,
        "51-75%": dist[2] or 0,
        "76-100%": dist[3] or 0,
    }

    # daily completion trend (avg completion per date)
    cur.execute("""
        SELECT plan_date, ROUND(AVG(plan_completion_pct), 1) as avg_pct, COUNT(*) as plan_count
        FROM daily_plans
        GROUP BY plan_date
        ORDER BY plan_date
    """)
    daily_trend = [{"date": r[0], "avg_completion": r[1], "plan_count": r[2]} for r in cur.fetchall()]

    # date range
    cur.execute("SELECT MIN(plan_date), MAX(plan_date) FROM daily_plans")
    date_range = cur.fetchone()

    conn.close()
    return {
        "total_plans": total_plans,
        "total_patients": total_patients,
        "total_days": total_days,
        "avg_completion_pct": avg_completion,
        "min_completion_pct": min_completion or 0,
        "max_completion_pct": max_completion or 0,
        "activity_totals": activity_totals,
        "activity_rates": activity_rates,
        "completion_distribution": completion_distribution,
        "daily_trend": daily_trend,
        "date_range": {"start": date_range[0], "end": date_range[1]},
    }


def breakdown():
    """Per-patient engagement, recent plans, low-adherence alerts, activity heatmap."""
    conn = _conn()
    cur = conn.cursor()

    # per-patient summary
    cur.execute("""
        SELECT
            patient_id,
            COUNT(*) as total_plans,
            ROUND(AVG(plan_completion_pct), 1) as avg_completion,
            SUM(medication_reminders_set) as total_med_reminders,
            SUM(meals_logged) as total_meals,
            SUM(exercise_logged) as total_exercise,
            SUM(sleep_logged) as total_sleep,
            SUM(mood_logged) as total_mood,
            SUM(seizure_logged) as total_seizure,
            MIN(plan_date) as first_plan,
            MAX(plan_date) as last_plan
        FROM daily_plans
        GROUP BY patient_id
        ORDER BY avg_completion DESC
    """)
    per_patient = _dict_rows(cur)

    # recent plans (last 30)
    cur.execute("""
        SELECT patient_id, plan_date, medication_reminders_set, meals_logged,
               exercise_logged, sleep_logged, mood_logged, seizure_logged,
               plan_completion_pct, ai_suggestion
        FROM daily_plans
        ORDER BY plan_date DESC, patient_id
        LIMIT 30
    """)
    recent_plans = _dict_rows(cur)

    # low-adherence patients (bottom quartile — avg below overall avg - 5)
    cur.execute("SELECT AVG(plan_completion_pct) FROM daily_plans")
    overall_avg = cur.fetchone()[0] or 50
    low_thresh = round(overall_avg - 3, 1)
    high_thresh = round(overall_avg + 3, 1)

    cur.execute("""
        SELECT patient_id, ROUND(AVG(plan_completion_pct), 1) as avg_completion, COUNT(*) as plan_count
        FROM daily_plans
        GROUP BY patient_id
        HAVING AVG(plan_completion_pct) < ?
        ORDER BY avg_completion ASC
    """, (low_thresh,))
    low_adherence = _dict_rows(cur)

    # high-adherence patients (above average + 3)
    cur.execute("""
        SELECT patient_id, ROUND(AVG(plan_completion_pct), 1) as avg_completion, COUNT(*) as plan_count
        FROM daily_plans
        GROUP BY patient_id
        HAVING AVG(plan_completion_pct) >= ?
        ORDER BY avg_completion DESC
    """, (high_thresh,))
    high_adherence = _dict_rows(cur)

    # activity heatmap (day-of-week × activity)
    cur.execute("""
        SELECT
            CASE CAST(strftime('%w', plan_date) AS INTEGER)
                WHEN 0 THEN 'Sun' WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue'
                WHEN 3 THEN 'Wed' WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri' WHEN 6 THEN 'Sat'
            END as day_of_week,
            CAST(strftime('%w', plan_date) AS INTEGER) as dow_num,
            ROUND(AVG(plan_completion_pct), 1) as avg_completion,
            ROUND(100.0 * SUM(CASE WHEN medication_reminders_set > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as med_rate,
            ROUND(100.0 * SUM(CASE WHEN meals_logged > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as meal_rate,
            ROUND(100.0 * SUM(CASE WHEN exercise_logged > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as exercise_rate,
            ROUND(100.0 * SUM(CASE WHEN sleep_logged > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as sleep_rate,
            COUNT(*) as plan_count
        FROM daily_plans
        GROUP BY strftime('%w', plan_date)
        ORDER BY dow_num
    """)
    weekly_pattern = _dict_rows(cur)

    # AI suggestion frequency (top keywords from ai_suggestion)
    cur.execute("SELECT ai_suggestion FROM daily_plans WHERE ai_suggestion IS NOT NULL AND ai_suggestion != ''")
    suggestions = [r[0] for r in cur.fetchall()]
    suggestion_count = len(suggestions)

    conn.close()
    return {
        "per_patient": per_patient,
        "recent_plans": recent_plans,
        "low_adherence": low_adherence,
        "high_adherence": high_adherence,
        "weekly_pattern": weekly_pattern,
        "ai_suggestion_count": suggestion_count,
    }


def definitions():
    """Daily plan field definitions, activity descriptions, completion tiers, glossary."""
    return {
        "fields": {
            "patient_id": "Unique patient identifier (e.g. EPAT001)",
            "plan_date": "Date of the daily plan (YYYY-MM-DD)",
            "medication_reminders_set": "Number of medication reminders configured for the day",
            "meals_logged": "Number of meals logged by the patient",
            "exercise_logged": "Whether exercise was logged (0/1)",
            "sleep_logged": "Whether sleep data was logged (0/1)",
            "mood_logged": "Whether mood was logged (0/1)",
            "seizure_logged": "Whether a seizure event was logged (0/1)",
            "plan_completion_pct": "Overall plan completion percentage (0-100)",
            "ai_suggestion": "AI-generated personalized suggestion for the patient",
        },
        "completion_tiers": {
            "Low (0-25%)": "Patient completed very few planned activities",
            "Medium (26-50%)": "Patient completed some planned activities",
            "High (51-75%)": "Patient completed most planned activities",
            "Excellent (76-100%)": "Patient completed nearly all planned activities",
        },
        "activity_types": {
            "Medication Reminders": "Tracks medication reminder configuration and adherence",
            "Meal Logging": "Tracks dietary intake documentation",
            "Exercise Logging": "Tracks physical activity recording",
            "Sleep Logging": "Tracks sleep pattern documentation",
            "Mood Logging": "Tracks emotional/psychological state recording",
            "Seizure Logging": "Tracks seizure event documentation",
        },
        "glossary": {
            "Daily Plan": "A structured set of health activities assigned to a patient for a specific day",
            "Plan Completion": "Percentage of assigned activities completed by the patient",
            "Adherence Rate": "How consistently a patient follows their daily plan over time",
            "AI Suggestion": "Machine-learning-generated personalized recommendation based on patient history",
            "Activity Rate": "Percentage of plans where a specific activity was logged at least once",
            "Low Adherence": "Patients averaging below 30% plan completion",
            "High Adherence": "Patients averaging 70% or above plan completion",
            "Medication Reminder": "Scheduled alert for medication intake",
            "Weekly Pattern": "Day-of-week aggregation showing activity and completion trends",
            "Engagement Score": "Composite measure of patient interaction with their daily plan",
            "SUDEP": "Sudden Unexpected Death in Epilepsy — a risk factor tracked via plan adherence",
            "Self-Management": "Patient-driven health tracking and activity logging",
        },
    }
