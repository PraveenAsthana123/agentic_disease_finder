"""
Neuro AI Ecosystem — Daily Care Plan Dashboard
================================================
Tracks daily care plan adherence across patients: medication reminders,
meals, exercise, sleep, mood, and seizure logging. Surfaces completion
trends, activity logging rates, weekly patterns, and AI suggestion
frequency to support clinical oversight and patient engagement.

Data Source:
  daily_plans table in clinical.db — 900 rows, 30 patients,
  date range 2025-05-16 to 2025-06-14.

Activity Columns (integer counts per day):
  medication_reminders_set — number of medication reminders configured
  meals_logged             — meal entries recorded
  exercise_logged          — exercise sessions recorded
  sleep_logged             — sleep entries recorded
  mood_logged              — mood check-ins recorded
  seizure_logged           — seizure events recorded

Completion:
  plan_completion_pct — 0-100 integer representing overall daily plan
                        completion percentage computed from activity counts.

Author: Research Team
"""

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# Activity columns used for logging-rate calculations
_ACTIVITY_COLS = [
    "medication_reminders_set",
    "meals_logged",
    "exercise_logged",
    "sleep_logged",
    "mood_logged",
    "seizure_logged",
]

_ACTIVITY_LABELS = {
    "medication_reminders_set": "Medication Reminders",
    "meals_logged": "Meals",
    "exercise_logged": "Exercise",
    "sleep_logged": "Sleep",
    "mood_logged": "Mood",
    "seizure_logged": "Seizure",
}

_WEEKDAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def _conn():
    return sqlite3.connect(DB_PATH)


def _patient_filter(patient_id: Optional[str]):
    """Return (where_clause, params) tuple for optional patient filtering."""
    if patient_id:
        return "WHERE dp.patient_id = ?", (patient_id,)
    return "", ()


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def overview(patient_id: Optional[str] = None) -> dict:
    """
    Daily care plan overview dashboard data.

    Returns:
        kpis                   — key performance indicators
        completion_distribution — bar chart buckets (0-25, 26-50, 51-75, 76-100)
        activity_trend         — line chart: avg completion per day
        logging_rates          — bar chart: % of plans where each activity > 0
    """
    conn = _conn()
    c = conn.cursor()
    where, params = _patient_filter(patient_id)

    # ── KPIs ─────────────────────────────────────────────────────────
    c.execute(f"""
        SELECT
            COUNT(*)                                    AS total_plans,
            COUNT(DISTINCT dp.patient_id)               AS total_patients,
            COALESCE(AVG(dp.plan_completion_pct), 0)    AS avg_completion,
            SUM(CASE WHEN dp.seizure_logged > 0 THEN 1 ELSE 0 END) AS plans_with_seizure,
            SUM(CASE WHEN dp.medication_reminders_set >= 3 THEN 1 ELSE 0 END) AS med_adherent,
            COUNT(DISTINCT dp.plan_date)                AS active_days
        FROM daily_plans dp
        {where}
    """, params)
    row = c.fetchone()
    total_plans = row[0] or 0
    total_patients = row[1] or 0
    avg_completion = round(float(row[2]), 1)
    plans_with_seizure = row[3] or 0
    med_adherent = row[4] or 0
    active_days = row[5] or 0
    medication_adherence_rate = round(med_adherent / total_plans * 100, 1) if total_plans else 0.0

    kpis = [
        {"label": "Total Plans",              "value": total_plans,               "sub": "daily care plans recorded"},
        {"label": "Total Patients",           "value": total_patients,            "sub": "distinct patients"},
        {"label": "Avg Completion",           "value": f"{avg_completion}%",      "sub": "mean plan_completion_pct"},
        {"label": "Plans with Seizure",       "value": plans_with_seizure,        "sub": "seizure_logged > 0"},
        {"label": "Medication Adherence",     "value": f"{medication_adherence_rate}%", "sub": "plans with >= 3 reminders set"},
        {"label": "Active Days",             "value": active_days,               "sub": "distinct plan dates"},
    ]

    # ── Completion Distribution ──────────────────────────────────────
    c.execute(f"""
        SELECT
            SUM(CASE WHEN dp.plan_completion_pct BETWEEN 0  AND 25  THEN 1 ELSE 0 END),
            SUM(CASE WHEN dp.plan_completion_pct BETWEEN 26 AND 50  THEN 1 ELSE 0 END),
            SUM(CASE WHEN dp.plan_completion_pct BETWEEN 51 AND 75  THEN 1 ELSE 0 END),
            SUM(CASE WHEN dp.plan_completion_pct BETWEEN 76 AND 100 THEN 1 ELSE 0 END)
        FROM daily_plans dp
        {where}
    """, params)
    dist = c.fetchone()
    completion_distribution = [
        {"bucket": "0-25%",   "count": dist[0] or 0},
        {"bucket": "26-50%",  "count": dist[1] or 0},
        {"bucket": "51-75%",  "count": dist[2] or 0},
        {"bucket": "76-100%", "count": dist[3] or 0},
    ]

    # ── Activity Trend (avg completion per day) ──────────────────────
    c.execute(f"""
        SELECT dp.plan_date, AVG(dp.plan_completion_pct)
        FROM daily_plans dp
        {where}
        GROUP BY dp.plan_date
        ORDER BY dp.plan_date
    """, params)
    activity_trend = [
        {"date": r[0], "avg_completion_pct": round(float(r[1]), 1)}
        for r in c.fetchall()
    ]

    # ── Logging Rates ────────────────────────────────────────────────
    if total_plans > 0:
        rate_cases = ", ".join(
            f"SUM(CASE WHEN dp.{col} > 0 THEN 1 ELSE 0 END) AS {col}_logged_count"
            for col in _ACTIVITY_COLS
        )
        c.execute(f"SELECT {rate_cases} FROM daily_plans dp {where}", params)
        counts = c.fetchone()
        logging_rates = [
            {
                "activity": _ACTIVITY_LABELS[col],
                "column": col,
                "logged_count": counts[i] or 0,
                "rate_pct": round((counts[i] or 0) / total_plans * 100, 1),
            }
            for i, col in enumerate(_ACTIVITY_COLS)
        ]
    else:
        logging_rates = [
            {"activity": _ACTIVITY_LABELS[col], "column": col, "logged_count": 0, "rate_pct": 0.0}
            for col in _ACTIVITY_COLS
        ]

    conn.close()

    return {
        "kpis": kpis,
        "completion_distribution": completion_distribution,
        "activity_trend": activity_trend,
        "logging_rates": logging_rates,
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed daily care plan breakdown.

    Returns:
        patient_summary — per-patient stats table
        weekly_heatmap  — avg completion by day-of-week
        ai_suggestions  — top 10 most frequent AI suggestions
        recent_plans    — last 20 plans with all fields
    """
    conn = _conn()
    c = conn.cursor()
    where, params = _patient_filter(patient_id)

    # ── Patient Summary ──────────────────────────────────────────────
    c.execute(f"""
        SELECT
            dp.patient_id,
            COUNT(*)                                          AS total_plans,
            ROUND(AVG(dp.plan_completion_pct), 1)             AS avg_completion,
            SUM(CASE WHEN dp.seizure_logged > 0 THEN 1 ELSE 0 END) AS days_with_seizure,
            SUM(CASE WHEN dp.medication_reminders_set >= 3 THEN 1 ELSE 0 END) AS med_adherent_plans,
            dp.ai_suggestion
        FROM daily_plans dp
        {where}
        GROUP BY dp.patient_id
        ORDER BY dp.patient_id
    """, params)
    patient_rows = c.fetchall()

    # Get top AI suggestion per patient (most frequent)
    top_suggestions: dict = {}
    for pid_group in {r[0] for r in patient_rows}:
        c.execute("""
            SELECT ai_suggestion, COUNT(*) AS cnt
            FROM daily_plans
            WHERE patient_id = ? AND ai_suggestion IS NOT NULL AND ai_suggestion != ''
            GROUP BY ai_suggestion
            ORDER BY cnt DESC
            LIMIT 1
        """, (pid_group,))
        sug_row = c.fetchone()
        top_suggestions[pid_group] = sug_row[0] if sug_row else None

    # Join patient demographics for name/age/gender
    patient_info: dict = {}
    try:
        c.execute("SELECT patient_id, age, gender FROM patients")
        for r in c.fetchall():
            patient_info[r[0]] = {"age": r[1], "gender": r[2]}
    except Exception:
        pass

    patient_demo: dict = {}
    try:
        c.execute("SELECT patient_id, fields_json FROM patient_demographics")
        import json as _json
        for r in c.fetchall():
            try:
                fields = _json.loads(r[1]) if r[1] else {}
                patient_demo[r[0]] = fields.get("name", None)
            except (ValueError, TypeError):
                pass
    except Exception:
        pass

    patient_summary = []
    for row in patient_rows:
        pid = row[0]
        total = row[1]
        avg_comp = float(row[2]) if row[2] is not None else 0.0
        days_seizure = row[3] or 0
        med_adherent = row[4] or 0
        med_rate = round(med_adherent / total * 100, 1) if total else 0.0
        info = patient_info.get(pid, {})
        patient_summary.append({
            "patient_id": pid,
            "name": patient_demo.get(pid),
            "age": info.get("age"),
            "gender": info.get("gender"),
            "total_plans": total,
            "avg_completion": avg_comp,
            "days_with_seizure": days_seizure,
            "medication_adherence_rate": med_rate,
            "top_ai_suggestion": top_suggestions.get(pid),
        })

    # ── Weekly Heatmap (avg completion by day-of-week) ───────────────
    # SQLite strftime('%w', date) returns 0=Sunday .. 6=Saturday
    c.execute(f"""
        SELECT
            CAST(strftime('%w', dp.plan_date) AS INTEGER) AS dow,
            AVG(dp.plan_completion_pct) AS avg_comp
        FROM daily_plans dp
        {where}
        GROUP BY dow
        ORDER BY dow
    """, params)
    dow_data = {r[0]: round(float(r[1]), 1) for r in c.fetchall()}
    # Map SQLite dow (0=Sun) to Monday-first ordering
    _sqlite_dow_to_name = {
        1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
        5: "Friday", 6: "Saturday", 0: "Sunday",
    }
    weekly_heatmap = [
        {
            "day": _sqlite_dow_to_name.get(dow, f"Day{dow}"),
            "avg_completion_pct": dow_data.get(dow, 0.0),
        }
        for dow in [1, 2, 3, 4, 5, 6, 0]
    ]

    # ── AI Suggestions (top 10 most frequent) ────────────────────────
    c.execute(f"""
        SELECT dp.ai_suggestion, COUNT(*) AS cnt
        FROM daily_plans dp
        {where}
        {"AND" if where else "WHERE"} dp.ai_suggestion IS NOT NULL AND dp.ai_suggestion != ''
        GROUP BY dp.ai_suggestion
        ORDER BY cnt DESC
        LIMIT 10
    """, params)
    ai_suggestions = [
        {"suggestion": r[0], "count": r[1]}
        for r in c.fetchall()
    ]

    # ── Recent Plans (last 20) ───────────────────────────────────────
    c.execute(f"""
        SELECT dp.id, dp.patient_id, dp.plan_date,
               dp.medication_reminders_set, dp.meals_logged,
               dp.exercise_logged, dp.sleep_logged, dp.mood_logged,
               dp.seizure_logged, dp.plan_completion_pct,
               dp.ai_suggestion, dp.created_at
        FROM daily_plans dp
        {where}
        ORDER BY dp.plan_date DESC, dp.id DESC
        LIMIT 20
    """, params)
    recent_cols = [
        "id", "patient_id", "plan_date", "medication_reminders_set",
        "meals_logged", "exercise_logged", "sleep_logged", "mood_logged",
        "seizure_logged", "plan_completion_pct", "ai_suggestion", "created_at",
    ]
    recent_plans = [dict(zip(recent_cols, r)) for r in c.fetchall()]

    conn.close()

    return {
        "patient_summary": patient_summary,
        "weekly_heatmap": weekly_heatmap,
        "ai_suggestions": ai_suggestions,
        "recent_plans": recent_plans,
    }


def definitions() -> dict:
    """
    Metric definitions, activity type descriptions, completion scoring
    methodology, and clinical glossary for the Daily Care Plan Dashboard.
    """
    return {
        "metrics": [
            {
                "name": "Total Plans",
                "description": (
                    "Total number of daily care plan records across all patients and dates. "
                    "Each row represents one patient-day with logged activities."
                ),
            },
            {
                "name": "Total Patients",
                "description": (
                    "Count of distinct patient_id values with at least one daily care plan."
                ),
            },
            {
                "name": "Avg Completion",
                "description": (
                    "Mean plan_completion_pct across all plans. Represents overall adherence "
                    "to the daily care plan across the patient population."
                ),
            },
            {
                "name": "Plans with Seizure",
                "description": (
                    "Number of daily plans where seizure_logged > 0, indicating at least one "
                    "seizure event was recorded that day."
                ),
            },
            {
                "name": "Medication Adherence Rate",
                "description": (
                    "Percentage of daily plans where medication_reminders_set >= 3, used as a "
                    "proxy for adequate medication reminder configuration and adherence."
                ),
            },
            {
                "name": "Active Days",
                "description": (
                    "Count of distinct plan_date values, representing the calendar span of "
                    "recorded activity in the dataset."
                ),
            },
        ],
        "activity_types": {
            "medication_reminders_set": (
                "Number of medication reminders configured for the day. A value >= 3 is "
                "considered adherent, reflecting morning, midday, and evening reminders."
            ),
            "meals_logged": (
                "Count of meal entries recorded. Tracks nutritional adherence to the care "
                "plan, including breakfast, lunch, dinner, and snacks."
            ),
            "exercise_logged": (
                "Number of exercise or physical activity sessions recorded. Includes "
                "prescribed rehabilitation exercises, walks, and structured fitness."
            ),
            "sleep_logged": (
                "Sleep tracking entries for the day. Captures sleep onset, duration, "
                "quality, and nocturnal events relevant to seizure risk."
            ),
            "mood_logged": (
                "Mood check-in entries. Captures emotional state using validated scales "
                "(e.g., PHQ-2, GAD-2) to monitor comorbid depression/anxiety."
            ),
            "seizure_logged": (
                "Number of seizure events recorded for the day. Zero indicates a "
                "seizure-free day; values > 0 are clinically significant."
            ),
        },
        "completion_scoring": {
            "method": (
                "plan_completion_pct is an integer 0-100 representing the proportion of "
                "planned daily care activities that were actually completed and logged. "
                "Each activity category contributes equally to the total score. A plan "
                "with all six activities logged at target levels scores 100%."
            ),
            "thresholds": {
                "excellent": "76-100% — Patient completed most or all daily activities.",
                "good": "51-75% — Majority of activities logged; minor gaps.",
                "partial": "26-50% — Roughly half of planned activities completed.",
                "low": "0-25% — Minimal engagement with the daily care plan.",
            },
        },
        "glossary": [
            {
                "term": "Daily Care Plan",
                "definition": (
                    "A structured set of activities and check-ins assigned to a patient for "
                    "each calendar day, covering medication, nutrition, exercise, sleep, mood, "
                    "and seizure monitoring."
                ),
            },
            {
                "term": "Medication Adherence",
                "definition": (
                    "The degree to which a patient follows prescribed medication schedules. "
                    "Measured here by the number of medication reminders set (>= 3 = adherent)."
                ),
            },
            {
                "term": "Seizure Logging",
                "definition": (
                    "Patient or caregiver recording of seizure events including type, duration, "
                    "and triggers. Critical for adjusting antiepileptic drug regimens and "
                    "identifying patterns."
                ),
            },
            {
                "term": "AI Suggestion",
                "definition": (
                    "An algorithmically generated recommendation provided to the patient or "
                    "clinician based on recent activity patterns, adherence trends, and clinical "
                    "risk factors. Examples include medication timing adjustments, exercise "
                    "prompts, and sleep hygiene tips."
                ),
            },
            {
                "term": "Plan Completion Percentage",
                "definition": (
                    "A composite score (0-100) reflecting the proportion of daily care plan "
                    "components that were completed. Used as the primary adherence KPI."
                ),
            },
            {
                "term": "Activity Logging Rate",
                "definition": (
                    "The percentage of daily plans in which a given activity type was logged "
                    "at least once (value > 0). Indicates engagement with each care plan component."
                ),
            },
            {
                "term": "Weekly Heatmap",
                "definition": (
                    "A visualization of average plan completion grouped by day of the week "
                    "(Monday-Sunday). Helps identify weekday vs weekend adherence patterns."
                ),
            },
            {
                "term": "Comorbid Monitoring",
                "definition": (
                    "Tracking of conditions commonly co-occurring with epilepsy, such as "
                    "depression and anxiety, through mood logging and validated screening scales."
                ),
            },
        ],
    }
