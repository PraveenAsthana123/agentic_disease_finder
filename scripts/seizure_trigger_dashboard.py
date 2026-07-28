"""Seizure Trigger Logs Dashboard — daily seizure diary analytics from clinical.db.

Tracks patient-reported seizure triggers, lifestyle factors, medication adherence,
seizure occurrence rates, and trigger-specific risk profiles.

Sources:
- seizure_trigger_logs table (203 rows, 9 trigger types, 4 sleep quality levels,
  6 seizure types, lifestyle metrics)
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
    """Return high-level seizure trigger metrics, distributions, trends, risk comparison."""
    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM seizure_trigger_logs")
    total_logs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM seizure_trigger_logs")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM seizure_trigger_logs WHERE seizure_occurred = 1")
    seizure_count = cur.fetchone()[0]

    seizure_rate = round((seizure_count / total_logs) * 100, 1) if total_logs else 0.0

    cur.execute("SELECT ROUND(AVG(sleep_hours), 1) FROM seizure_trigger_logs")
    avg_sleep_hours = cur.fetchone()[0] or 0

    cur.execute(
        "SELECT ROUND(100.0 * SUM(medication_adherence) / COUNT(*), 1) "
        "FROM seizure_trigger_logs"
    )
    medication_adherence_rate = cur.fetchone()[0] or 0

    # trigger distribution — array of {trigger, count}
    cur.execute(
        "SELECT primary_trigger, COUNT(*) AS cnt "
        "FROM seizure_trigger_logs GROUP BY primary_trigger ORDER BY cnt DESC"
    )
    trigger_distribution = [{"trigger": r[0], "count": r[1]} for r in cur.fetchall()]

    # sleep quality distribution — array of {quality, count}
    cur.execute(
        "SELECT sleep_quality, COUNT(*) AS cnt "
        "FROM seizure_trigger_logs GROUP BY sleep_quality ORDER BY cnt DESC"
    )
    sleep_quality_distribution = [{"quality": r[0], "count": r[1]} for r in cur.fetchall()]

    # seizure type distribution — array of {type, count}
    cur.execute(
        "SELECT seizure_type, COUNT(*) AS cnt "
        "FROM seizure_trigger_logs "
        "WHERE seizure_occurred = 1 AND seizure_type IS NOT NULL "
        "GROUP BY seizure_type ORDER BY cnt DESC"
    )
    seizure_type_distribution = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]

    # monthly trend — array of {month, total_logs, seizures}
    cur.execute(
        "SELECT SUBSTR(log_date, 1, 7) AS month, "
        "COUNT(*) AS total_logs, "
        "SUM(CASE WHEN seizure_occurred = 1 THEN 1 ELSE 0 END) AS seizures "
        "FROM seizure_trigger_logs GROUP BY month ORDER BY month"
    )
    monthly_trend = [{"month": r[0], "total_logs": r[1], "seizures": r[2]} for r in cur.fetchall()]

    # trigger seizure rate — array of {trigger, rate}
    cur.execute(
        "SELECT primary_trigger, "
        "ROUND(100.0 * SUM(CASE WHEN seizure_occurred = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS rate "
        "FROM seizure_trigger_logs GROUP BY primary_trigger ORDER BY rate DESC"
    )
    trigger_seizure_rate = [{"trigger": r[0], "rate": r[1]} for r in cur.fetchall()]

    # risk comparison: avg lifestyle factors for seizure vs non-seizure days
    factors = [
        ("sleep_hours", "Sleep Hours"),
        ("stress_level", "Stress Level"),
        ("fatigue_level", "Fatigue Level"),
        ("mood_score", "Mood Score"),
        ("exercise_minutes", "Exercise (min)"),
        ("caffeine_mg", "Caffeine (mg)"),
        ("alcohol_units", "Alcohol Units"),
        ("screen_time_hours", "Screen Time (h)"),
    ]
    risk_comparison = []
    for col, label in factors:
        cur.execute(
            "SELECT ROUND(AVG(CASE WHEN seizure_occurred = 1 THEN [{}] END), 1), "
            "ROUND(AVG(CASE WHEN seizure_occurred = 0 THEN [{}] END), 1) "
            "FROM seizure_trigger_logs".format(col, col)
        )
        row = cur.fetchone()
        risk_comparison.append({
            "factor": label,
            "with_seizure": row[0],
            "without_seizure": row[1],
        })

    # sleep quality vs seizure rate
    cur.execute(
        "SELECT sleep_quality, "
        "ROUND(100.0 * SUM(CASE WHEN seizure_occurred = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS rate "
        "FROM seizure_trigger_logs GROUP BY sleep_quality ORDER BY rate DESC"
    )
    sleep_vs_seizure = [{"quality": r[0], "rate": r[1]} for r in cur.fetchall()]

    conn.close()
    return {
        "total_logs": total_logs,
        "total_patients": total_patients,
        "seizure_count": seizure_count,
        "seizure_rate": seizure_rate,
        "avg_sleep_hours": avg_sleep_hours,
        "medication_adherence_rate": medication_adherence_rate,
        "trigger_distribution": trigger_distribution,
        "sleep_quality_distribution": sleep_quality_distribution,
        "seizure_type_distribution": seizure_type_distribution,
        "monthly_trend": monthly_trend,
        "trigger_seizure_rate": trigger_seizure_rate,
        "risk_comparison": risk_comparison,
        "sleep_vs_seizure": sleep_vs_seizure,
    }


def breakdown():
    """Return all logs, per-patient summary, top patient-trigger combos."""
    conn = _conn()
    cur = conn.cursor()

    # all logs
    cur.execute("SELECT * FROM seizure_trigger_logs ORDER BY log_date DESC")
    logs = _dict_rows(cur)

    # per-patient summary
    cur.execute(
        "SELECT patient_id, "
        "COUNT(*) AS total_logs, "
        "SUM(CASE WHEN seizure_occurred = 1 THEN 1 ELSE 0 END) AS seizures, "
        "ROUND(AVG(sleep_hours), 1) AS avg_sleep, "
        "ROUND(AVG(stress_level), 1) AS avg_stress, "
        "ROUND(100.0 * SUM(medication_adherence) / COUNT(*), 1) AS adherence_pct, "
        "MAX(log_date) AS latest_log "
        "FROM seizure_trigger_logs GROUP BY patient_id "
        "ORDER BY seizures DESC, total_logs DESC"
    )
    patient_summary = _dict_rows(cur)

    # top patient-trigger combos (seizure events)
    cur.execute(
        "SELECT patient_id, primary_trigger, COUNT(*) AS count "
        "FROM seizure_trigger_logs WHERE seizure_occurred = 1 "
        "GROUP BY patient_id, primary_trigger "
        "ORDER BY count DESC LIMIT 20"
    )
    top_patient_triggers = _dict_rows(cur)

    conn.close()
    return {
        "logs": logs,
        "patient_summary": patient_summary,
        "top_patient_triggers": top_patient_triggers,
    }


def definitions():
    """Return concepts, seizure types, and data sources for the definitions tab."""
    return {
        "title": "Seizure Trigger Logs — Definitions & Glossary",
        "concepts": [
            {"name": "Primary Trigger", "description": "The main factor identified as a potential seizure trigger for the day. Patients self-report from a standardised list of 9 categories: illness, sleep deprivation, photosensitivity, dehydration, hormonal changes, stress, alcohol, missed medication, and fatigue."},
            {"name": "Seizure Occurred", "description": "Whether a seizure event was recorded on that calendar day (yes/no). Used to compute seizure rate and correlate with trigger and lifestyle factors."},
            {"name": "Sleep Quality", "description": "Patient-reported sleep quality on a 4-level scale (good, fair, poor, very poor). Poor sleep is a well-established seizure precipitant."},
            {"name": "Stress Level", "description": "Self-reported daily stress on a 1\u201310 scale. Higher stress is associated with increased seizure frequency in many epilepsy patients."},
            {"name": "Medication Adherence", "description": "Whether the patient took all prescribed anti-epileptic drugs (AEDs) that day. Non-adherence is one of the strongest modifiable seizure risk factors."},
            {"name": "Seizure Rate", "description": "Percentage of logged days on which a seizure occurred, calculated as (seizure days / total logged days) \u00d7 100."},
            {"name": "Risk Comparison", "description": "Side-by-side average of lifestyle factors on seizure days vs non-seizure days. Highlights which modifiable factors differ most between the two groups."},
            {"name": "Fatigue Level", "description": "Self-reported fatigue on a 1\u201310 scale. Fatigue can lower seizure threshold, particularly in combination with sleep deprivation."},
            {"name": "Mood Score", "description": "Self-reported mood on a 1\u201310 scale. Mood disturbances (anxiety, depression) are common epilepsy comorbidities and may influence seizure frequency."},
        ],
        "seizure_types": [
            {"type": "focal aware", "description": "Seizure originating in one brain area; patient remains conscious. May include motor, sensory, or autonomic symptoms."},
            {"type": "focal impaired awareness", "description": "Seizure originating in one brain area with altered consciousness. Previously called complex partial seizure."},
            {"type": "focal to bilateral tonic-clonic", "description": "Starts as a focal seizure then spreads to both hemispheres, causing full-body convulsions."},
            {"type": "generalized tonic-clonic", "description": "Seizure involving both hemispheres from onset with stiffening (tonic) then jerking (clonic) phases."},
            {"type": "absence", "description": "Brief generalized seizure with momentary loss of awareness, often mistaken for daydreaming. Common in children."},
            {"type": "myoclonic", "description": "Brief, shock-like jerks of a muscle or group of muscles. Often occurs shortly after waking."},
        ],
        "data_sources": [
            "seizure_trigger_logs table in clinical.db \u2014 203 patient-reported daily diary entries",
            "Covers lifestyle factors (sleep, stress, fatigue, mood, exercise, caffeine, alcohol, screen time), medication adherence, primary trigger, and seizure outcomes",
            "9 trigger categories, 4 sleep quality levels, 6 ILAE seizure types",
        ],
    }
