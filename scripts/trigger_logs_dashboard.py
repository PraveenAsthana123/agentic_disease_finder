"""
Trigger Logs & Lifestyle Diary Dashboard
=========================================
Daily lifestyle diary with seizure trigger analysis.

Real data: seizure_trigger_logs (203 rows, 40 patients) in clinical.db.
Columns: id, patient_id, log_date, sleep_hours, sleep_quality, stress_level,
         fatigue_level, mood_score, exercise_minutes, caffeine_mg, alcohol_units,
         screen_time_hours, medication_adherence, missed_doses, primary_trigger,
         seizure_occurred, seizure_type, seizure_duration_sec, notes
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
    """Trigger logs overview — KPIs, trigger distribution, sleep quality,
    monthly seizure trend, lifestyle averages with/without seizure, stress vs seizure rate."""
    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM seizure_trigger_logs")
    total_logs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM seizure_trigger_logs")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM seizure_trigger_logs WHERE seizure_occurred = 1")
    total_seizure_events = cur.fetchone()[0]

    seizure_rate_pct = round(total_seizure_events / max(total_logs, 1) * 100, 1)

    cur.execute("SELECT ROUND(AVG(sleep_hours), 1) FROM seizure_trigger_logs")
    avg_sleep_hours = cur.fetchone()[0] or 0.0

    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN medication_adherence = 1 THEN 100.0 ELSE 0.0 END), 1)
        FROM seizure_trigger_logs
    """)
    medication_adherence_pct = cur.fetchone()[0] or 0.0

    # Primary trigger distribution
    cur.execute("""
        SELECT primary_trigger, COUNT(*) cnt
        FROM seizure_trigger_logs
        WHERE primary_trigger IS NOT NULL
        GROUP BY primary_trigger
        ORDER BY cnt DESC
    """)
    trigger_rows = cur.fetchall()
    trigger_total = sum(r[1] for r in trigger_rows)
    primary_trigger_distribution = [
        {"trigger": r[0], "count": r[1], "pct": round(r[1] / max(trigger_total, 1) * 100, 1)}
        for r in trigger_rows
    ]

    # Sleep quality distribution
    cur.execute("""
        SELECT sleep_quality, COUNT(*) cnt
        FROM seizure_trigger_logs
        WHERE sleep_quality IS NOT NULL
        GROUP BY sleep_quality
        ORDER BY cnt DESC
    """)
    sleep_quality_distribution = [{"quality": r[0], "count": r[1]} for r in cur.fetchall()]

    # Monthly seizure trend
    cur.execute("""
        SELECT SUBSTR(log_date, 1, 7) month,
               COUNT(*) total_logs,
               SUM(seizure_occurred) seizures
        FROM seizure_trigger_logs
        GROUP BY month
        ORDER BY month
    """)
    seizure_by_month = [
        {"month": r[0], "total_logs": r[1], "seizures": r[2]}
        for r in cur.fetchall()
    ]

    # Lifestyle averages: with seizure vs without
    lifestyle_factors = [
        ("Sleep Hours",    "sleep_hours"),
        ("Stress Level",   "stress_level"),
        ("Fatigue Level",  "fatigue_level"),
        ("Mood Score",     "mood_score"),
        ("Exercise (min)", "exercise_minutes"),
        ("Caffeine (mg)",  "caffeine_mg"),
        ("Alcohol Units",  "alcohol_units"),
        ("Missed Doses",   "missed_doses"),
    ]
    lifestyle_averages = []
    for label, col in lifestyle_factors:
        cur.execute(f"""
            SELECT
              ROUND(AVG(CASE WHEN seizure_occurred = 1 THEN {col} END), 2),
              ROUND(AVG(CASE WHEN seizure_occurred = 0 THEN {col} END), 2)
            FROM seizure_trigger_logs
        """)
        r = cur.fetchone()
        lifestyle_averages.append({
            "factor":          label,
            "with_seizure":    r[0] or 0.0,
            "without_seizure": r[1] or 0.0,
        })

    # Stress level vs seizure rate
    cur.execute("""
        SELECT stress_level,
               COUNT(*) cnt,
               ROUND(AVG(seizure_occurred) * 100, 1) seizure_pct
        FROM seizure_trigger_logs
        GROUP BY stress_level
        ORDER BY stress_level
    """)
    stress_vs_seizure = [
        {"stress_level": r[0], "count": r[1], "seizure_pct": r[2]}
        for r in cur.fetchall()
    ]

    conn.close()
    return {
        "kpis": {
            "total_logs":               total_logs,
            "total_patients":           total_patients,
            "total_seizure_events":     total_seizure_events,
            "seizure_rate_pct":         seizure_rate_pct,
            "avg_sleep_hours":          avg_sleep_hours,
            "medication_adherence_pct": medication_adherence_pct,
        },
        "primary_trigger_distribution": primary_trigger_distribution,
        "sleep_quality_distribution":   sleep_quality_distribution,
        "seizure_by_month":             seizure_by_month,
        "lifestyle_averages":           lifestyle_averages,
        "stress_vs_seizure":            stress_vs_seizure,
    }


def breakdown():
    """Trigger logs breakdown — per-patient summary, high-risk days, recent diary logs."""
    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary
    cur.execute("""
        SELECT patient_id,
               COUNT(*) total_logs,
               SUM(seizure_occurred) seizures,
               ROUND(AVG(seizure_occurred) * 100, 1) seizure_rate,
               ROUND(AVG(sleep_hours), 1) avg_sleep,
               ROUND(AVG(stress_level), 1) avg_stress,
               ROUND(AVG(mood_score), 1) avg_mood,
               ROUND(AVG(CASE WHEN medication_adherence = 1 THEN 100.0 ELSE 0.0 END), 1) adherence_pct
        FROM seizure_trigger_logs
        GROUP BY patient_id
        ORDER BY seizure_rate DESC
    """)
    per_patient = _dict_rows(cur)

    # High-risk days: stress >= 7 OR sleep < 5 OR missed_doses > 0
    cur.execute("""
        SELECT patient_id, log_date, primary_trigger,
               sleep_hours, stress_level, fatigue_level, missed_doses
        FROM seizure_trigger_logs
        WHERE stress_level >= 7 OR sleep_hours < 5 OR missed_doses > 0
        ORDER BY log_date DESC
        LIMIT 50
    """)
    high_risk_days = _dict_rows(cur)

    # Recent logs (last 30 entries)
    cur.execute("""
        SELECT patient_id, log_date, sleep_hours, sleep_quality, stress_level,
               mood_score, seizure_occurred, primary_trigger, medication_adherence, notes
        FROM seizure_trigger_logs
        ORDER BY log_date DESC
        LIMIT 30
    """)
    recent_logs = _dict_rows(cur)

    conn.close()
    return {
        "per_patient":    per_patient,
        "high_risk_days": high_risk_days,
        "recent_logs":    recent_logs,
    }


def definitions():
    """Trigger and lifestyle metric definitions."""
    return {
        "trigger_descriptions": {
            "sleep_deprivation":  "Less than 6 hours of sleep or severely disrupted sleep preceding the event.",
            "photosensitivity":   "Exposure to flickering lights, screens, or high-contrast patterns.",
            "missed_medication":  "Missed or delayed antiepileptic drug dose(s) within 24 hours.",
            "stress":             "Emotional or psychological stress exceeding the patient's normal baseline.",
            "hormonal_changes":   "Menstrual-cycle-related fluctuations (catamenial pattern).",
            "alcohol":            "Alcohol intake or withdrawal after significant consumption.",
            "illness":            "Fever, infection, or systemic illness disrupting seizure threshold.",
            "fatigue":            "Cumulative physical fatigue without primary sleep deprivation.",
            "dehydration":        "Inadequate fluid intake reducing electrolyte balance.",
        },
        "lifestyle_definitions": {
            "sleep_hours":          "Hours of sleep recorded for that diary day (float).",
            "sleep_quality":        "Self-rated sleep quality: good / fair / poor / very_poor.",
            "stress_level":         "Self-rated stress on a 1–10 scale (10 = maximum).",
            "fatigue_level":        "Self-rated fatigue on a 1–10 scale.",
            "mood_score":           "Self-rated mood on a 1–10 scale (10 = best).",
            "medication_adherence": "1 = all prescribed doses taken on time; 0 = at least one missed.",
            "missed_doses":         "Number of prescribed doses skipped that day.",
            "exercise_minutes":     "Minutes of physical activity recorded.",
            "caffeine_mg":          "Estimated caffeine intake in milligrams.",
            "alcohol_units":        "Standard alcohol units consumed (1 unit ≈ 10 g ethanol).",
            "screen_time_hours":    "Hours of screen use (phones/TV/computer).",
            "seizure_rate":         "Percentage of diary days on which at least one seizure occurred.",
            "high_risk_day":        "Day flagged by stress ≥ 7, sleep < 5 h, or missed doses > 0.",
        },
        "table": "seizure_trigger_logs",
        "rows":  203,
        "patients": 40,
        "date_range": "2026-01 to 2026-09",
        "source": "Clinical daily diary — self-reported by patients, EEG-technician verified.",
    }
