"""Seizure Trigger Log Dashboard — trigger diary analytics from clinical.db.

Tracks patient seizure trigger logs including sleep, stress, medication
adherence, lifestyle factors, and seizure occurrence patterns.

Sources:
- trigger_logs table (patient_id, fields_json containing log_date,
  sleep_hours, sleep_quality, stress_level, medication_adherence,
  missed_doses, exercise_minutes, caffeine_cups, alcohol_units,
  menstrual_phase, illness, photo_exposure, weather_change, mood_score,
  fatigue_level, seizure_occurred, seizure_count, primary_trigger, notes)
- 300 records, 30 patients, 44 seizure events, 7 primary trigger types
"""

import sqlite3
import os

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
#  /api/trigger-logs/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Aggregate trigger-log health: total logs, seizure rate, lifestyle
    averages, trigger distribution, and seizure-vs-no-seizure comparisons."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total_logs = _safe(cur,
        "SELECT COUNT(*) FROM trigger_logs")
    total_patients = _safe(cur,
        "SELECT COUNT(DISTINCT patient_id) FROM trigger_logs")
    total_seizure_events = _safe(cur,
        "SELECT COUNT(*) FROM trigger_logs "
        "WHERE json_extract(fields_json, '$.seizure_occurred') = 1")
    seizure_rate_pct = round(total_seizure_events / total_logs * 100, 1) if total_logs else 0
    avg_sleep_hours = _safe(cur,
        "SELECT ROUND(AVG(json_extract(fields_json, '$.sleep_hours')), 1) "
        "FROM trigger_logs")
    avg_stress_level = _safe(cur,
        "SELECT ROUND(AVG(json_extract(fields_json, '$.stress_level')), 1) "
        "FROM trigger_logs")
    avg_mood_score = _safe(cur,
        "SELECT ROUND(AVG(json_extract(fields_json, '$.mood_score')), 1) "
        "FROM trigger_logs")
    medication_adherence_pct = 0
    adherent_count = _safe(cur,
        "SELECT COUNT(*) FROM trigger_logs "
        "WHERE json_extract(fields_json, '$.medication_adherence') = 'yes'")
    if total_logs:
        medication_adherence_pct = round(adherent_count / total_logs * 100, 1)
    avg_exercise_minutes = _safe(cur,
        "SELECT ROUND(AVG(json_extract(fields_json, '$.exercise_minutes')), 1) "
        "FROM trigger_logs")
    avg_caffeine_cups = _safe(cur,
        "SELECT ROUND(AVG(json_extract(fields_json, '$.caffeine_cups')), 1) "
        "FROM trigger_logs")
    avg_fatigue_level = _safe(cur,
        "SELECT ROUND(AVG(json_extract(fields_json, '$.fatigue_level')), 1) "
        "FROM trigger_logs")

    # Primary trigger distribution
    primary_trigger_distribution = []
    for row in _safe_rows(cur,
            """SELECT json_extract(fields_json, '$.primary_trigger') as trigger_name,
                      COUNT(*) as cnt
               FROM trigger_logs
               WHERE json_extract(fields_json, '$.seizure_occurred') = 1
                 AND json_extract(fields_json, '$.primary_trigger') IS NOT NULL
                 AND json_extract(fields_json, '$.primary_trigger') != ''
               GROUP BY trigger_name ORDER BY cnt DESC"""):
        primary_trigger_distribution.append({
            "trigger": row[0], "count": row[1],
            "pct": round(row[1] / total_seizure_events * 100, 1) if total_seizure_events else 0
        })

    # Seizure by month
    seizure_by_month = []
    for row in _safe_rows(cur,
            """SELECT strftime('%Y-%m', json_extract(fields_json, '$.log_date')) as month,
                      SUM(CASE WHEN json_extract(fields_json, '$.seizure_occurred') = 1
                               THEN 1 ELSE 0 END) as seizures,
                      COUNT(*) as total_logs
               FROM trigger_logs
               GROUP BY month ORDER BY month"""):
        seizure_by_month.append({
            "month": row[0], "seizures": row[1], "total_logs": row[2]
        })

    # Sleep quality distribution (1-5 histogram)
    sleep_quality_distribution = []
    for row in _safe_rows(cur,
            """SELECT json_extract(fields_json, '$.sleep_quality') as quality,
                      COUNT(*) as cnt
               FROM trigger_logs
               WHERE json_extract(fields_json, '$.sleep_quality') IS NOT NULL
               GROUP BY quality ORDER BY quality"""):
        sleep_quality_distribution.append({
            "quality": row[0], "count": row[1]
        })

    # Stress vs seizure correlation
    stress_vs_seizure = []
    for row in _safe_rows(cur,
            """SELECT json_extract(fields_json, '$.stress_level') as stress_level,
                      ROUND(SUM(CASE WHEN json_extract(fields_json, '$.seizure_occurred') = 1
                                     THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as seizure_pct,
                      COUNT(*) as cnt
               FROM trigger_logs
               WHERE json_extract(fields_json, '$.stress_level') IS NOT NULL
               GROUP BY stress_level ORDER BY stress_level"""):
        stress_vs_seizure.append({
            "stress_level": row[0], "seizure_pct": row[1], "count": row[2]
        })

    # Lifestyle averages: seizure days vs non-seizure days
    lifestyle_averages = []
    for factor in ['sleep_hours', 'stress_level', 'caffeine_cups', 'fatigue_level']:
        with_sz = _safe(cur,
            f"SELECT ROUND(AVG(json_extract(fields_json, '$.{factor}')), 1) "
            f"FROM trigger_logs "
            f"WHERE json_extract(fields_json, '$.seizure_occurred') = 1")
        without_sz = _safe(cur,
            f"SELECT ROUND(AVG(json_extract(fields_json, '$.{factor}')), 1) "
            f"FROM trigger_logs "
            f"WHERE json_extract(fields_json, '$.seizure_occurred') != 1")
        lifestyle_averages.append({
            "factor": factor, "with_seizure": with_sz, "without_seizure": without_sz
        })

    conn.close()

    return {
        "available": True,
        "kpis": {
            "total_logs": total_logs,
            "total_patients": total_patients,
            "seizure_rate_pct": seizure_rate_pct,
            "total_seizure_events": total_seizure_events,
            "avg_sleep_hours": avg_sleep_hours,
            "avg_stress_level": avg_stress_level,
            "avg_mood_score": avg_mood_score,
            "medication_adherence_pct": medication_adherence_pct,
            "avg_exercise_minutes": avg_exercise_minutes,
            "avg_caffeine_cups": avg_caffeine_cups,
            "avg_fatigue_level": avg_fatigue_level
        },
        "primary_trigger_distribution": primary_trigger_distribution,
        "seizure_by_month": seizure_by_month,
        "sleep_quality_distribution": sleep_quality_distribution,
        "stress_vs_seizure": stress_vs_seizure,
        "lifestyle_averages": lifestyle_averages
    }


# ──────────────────────────────────────────────────────────────
#  /api/trigger-logs/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient trigger summary, recent log entries, high-risk days,
    and medication adherence issues."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary (top 20 by log count)
    per_patient = []
    for row in _safe_rows(cur,
            """SELECT patient_id,
                      COUNT(*) as total_logs,
                      SUM(CASE WHEN json_extract(fields_json, '$.seizure_occurred') = 1
                               THEN 1 ELSE 0 END) as seizures,
                      ROUND(SUM(CASE WHEN json_extract(fields_json, '$.seizure_occurred') = 1
                                     THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as seizure_rate,
                      ROUND(AVG(json_extract(fields_json, '$.sleep_hours')), 1) as avg_sleep,
                      ROUND(AVG(json_extract(fields_json, '$.stress_level')), 1) as avg_stress,
                      ROUND(AVG(json_extract(fields_json, '$.mood_score')), 1) as avg_mood,
                      ROUND(SUM(CASE WHEN json_extract(fields_json, '$.medication_adherence') = 'yes'
                                     THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as adherence_pct
               FROM trigger_logs GROUP BY patient_id
               ORDER BY total_logs DESC LIMIT 20"""):
        per_patient.append({
            "patient_id": row[0], "total_logs": row[1],
            "seizures": row[2], "seizure_rate": row[3],
            "avg_sleep": row[4], "avg_stress": row[5],
            "avg_mood": row[6], "adherence_pct": row[7]
        })

    # Recent logs (last 20)
    recent_logs = []
    for row in _safe_rows(cur,
            """SELECT patient_id,
                      json_extract(fields_json, '$.log_date') as log_date,
                      json_extract(fields_json, '$.sleep_hours') as sleep_hours,
                      json_extract(fields_json, '$.sleep_quality') as sleep_quality,
                      json_extract(fields_json, '$.stress_level') as stress_level,
                      json_extract(fields_json, '$.mood_score') as mood_score,
                      json_extract(fields_json, '$.seizure_occurred') as seizure_occurred,
                      json_extract(fields_json, '$.seizure_count') as seizure_count,
                      json_extract(fields_json, '$.primary_trigger') as primary_trigger,
                      json_extract(fields_json, '$.medication_adherence') as medication_adherence,
                      json_extract(fields_json, '$.notes') as notes
               FROM trigger_logs
               ORDER BY json_extract(fields_json, '$.log_date') DESC LIMIT 20"""):
        recent_logs.append({
            "patient_id": row[0], "log_date": row[1],
            "sleep_hours": row[2], "sleep_quality": row[3],
            "stress_level": row[4], "mood_score": row[5],
            "seizure_occurred": bool(row[6]), "seizure_count": row[7],
            "primary_trigger": row[8], "medication_adherence": row[9],
            "notes": row[10]
        })

    # High-risk days (seizure_occurred = true)
    high_risk_days = []
    for row in _safe_rows(cur,
            """SELECT patient_id,
                      json_extract(fields_json, '$.log_date') as log_date,
                      json_extract(fields_json, '$.primary_trigger') as primary_trigger,
                      json_extract(fields_json, '$.sleep_hours') as sleep_hours,
                      json_extract(fields_json, '$.stress_level') as stress_level,
                      json_extract(fields_json, '$.fatigue_level') as fatigue_level,
                      json_extract(fields_json, '$.missed_doses') as missed_doses
               FROM trigger_logs
               WHERE json_extract(fields_json, '$.seizure_occurred') = 1
               ORDER BY json_extract(fields_json, '$.log_date') DESC"""):
        high_risk_days.append({
            "patient_id": row[0], "log_date": row[1],
            "primary_trigger": row[2], "sleep_hours": row[3],
            "stress_level": row[4], "fatigue_level": row[5],
            "missed_doses": row[6]
        })

    # Adherence issues (medication_adherence='no' or missed_doses > 0)
    adherence_issues = []
    for row in _safe_rows(cur,
            """SELECT patient_id,
                      json_extract(fields_json, '$.log_date') as log_date,
                      json_extract(fields_json, '$.medication_adherence') as medication_adherence,
                      json_extract(fields_json, '$.missed_doses') as missed_doses,
                      json_extract(fields_json, '$.seizure_occurred') as seizure_occurred,
                      json_extract(fields_json, '$.primary_trigger') as primary_trigger,
                      json_extract(fields_json, '$.notes') as notes
               FROM trigger_logs
               WHERE json_extract(fields_json, '$.medication_adherence') = 'no'
                  OR json_extract(fields_json, '$.missed_doses') > 0
               ORDER BY json_extract(fields_json, '$.log_date') DESC"""):
        adherence_issues.append({
            "patient_id": row[0], "log_date": row[1],
            "medication_adherence": row[2], "missed_doses": row[3],
            "seizure_occurred": bool(row[4]), "primary_trigger": row[5],
            "notes": row[6]
        })

    conn.close()

    return {
        "available": True,
        "per_patient": per_patient,
        "recent_logs": recent_logs,
        "high_risk_days": high_risk_days,
        "adherence_issues": adherence_issues
    }


# ──────────────────────────────────────────────────────────────
#  /api/trigger-logs/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Metric definitions for tooltip overlays."""
    return {
        "trigger_descriptions": {
            "Stress": "Elevated psychological or emotional stress reported by the patient, "
                      "a well-established seizure precipitant in epilepsy.",
            "Sleep deprivation": "Insufficient sleep duration or poor sleep quality. Sleep "
                                 "deprivation is one of the most common seizure triggers.",
            "Medication missed": "One or more anti-seizure medication doses were missed or "
                                 "taken late, reducing therapeutic drug levels.",
            "Fatigue": "Physical or mental exhaustion beyond normal tiredness, often linked "
                       "to increased seizure susceptibility.",
            "Alcohol": "Alcohol consumption, which can lower seizure threshold and interact "
                       "with anti-seizure medications.",
            "Photosensitivity": "Exposure to flickering or flashing lights, patterns, or "
                                "screens that can provoke seizures in photosensitive individuals.",
            "Illness": "Acute illness (fever, infection) that may lower seizure threshold "
                       "through metabolic disruption or medication malabsorption."
        },
        "field_descriptions": {
            "sleep_hours": "Total hours of sleep the night before the log entry.",
            "sleep_quality": "Self-rated sleep quality on a 1-5 scale (1 = very poor, 5 = excellent).",
            "stress_level": "Self-rated stress level on a 1-5 scale (1 = minimal, 5 = extreme).",
            "medication_adherence": "Whether all prescribed anti-seizure medications were taken as directed (yes/no).",
            "missed_doses": "Number of medication doses missed in the reporting period.",
            "exercise_minutes": "Total minutes of physical exercise performed during the day.",
            "caffeine_cups": "Number of caffeinated beverage servings consumed.",
            "alcohol_units": "Standard alcohol units consumed (1 unit = ~14g pure alcohol).",
            "menstrual_phase": "Current phase of menstrual cycle, relevant for catamenial epilepsy.",
            "illness": "Whether the patient is currently experiencing acute illness.",
            "photo_exposure": "Whether the patient was exposed to flickering lights, screens, or visual patterns.",
            "weather_change": "Whether significant weather changes (pressure, temperature) occurred.",
            "mood_score": "Self-rated mood on a 1-10 scale (1 = very low, 10 = excellent).",
            "fatigue_level": "Self-rated fatigue on a 1-5 scale (1 = alert, 5 = severely fatigued).",
            "seizure_occurred": "Whether one or more seizures occurred during the reporting period.",
            "seizure_count": "Total number of seizures recorded in the reporting period.",
            "primary_trigger": "The most likely trigger identified for the seizure event.",
            "notes": "Free-text clinical or patient notes about the day's events."
        },
        "clinical_notes": [
            "Seizure trigger diaries are a cornerstone of self-management in epilepsy, "
            "helping patients and clinicians identify modifiable risk factors.",
            "Consistent diary completion improves seizure forecasting accuracy and enables "
            "personalized trigger avoidance strategies.",
            "Multi-factor trigger analysis (sleep + stress + adherence) provides better "
            "predictive value than any single factor alone.",
            "Catamenial epilepsy affects up to 40% of women with epilepsy; menstrual cycle "
            "tracking is essential for this subgroup.",
            "Medication adherence is the single most modifiable factor in seizure control — "
            "even one missed dose can precipitate breakthrough seizures."
        ],
        "glossary": {
            "Seizure Threshold": "The level of neuronal excitability required to trigger a "
                                 "seizure. Various factors (sleep, stress, medication) raise "
                                 "or lower this threshold.",
            "Breakthrough Seizure": "A seizure occurring despite ongoing anti-seizure medication, "
                                    "often triggered by modifiable risk factors.",
            "Catamenial Epilepsy": "Seizure pattern linked to the menstrual cycle, with increased "
                                   "seizure frequency during specific hormonal phases.",
            "Photosensitivity": "Abnormal brain response to visual stimuli (flashing lights, "
                                "patterns) that can provoke seizures in susceptible individuals.",
            "Medication Adherence": "The extent to which a patient takes medications as prescribed. "
                                    "Non-adherence is the leading cause of preventable seizures.",
            "Seizure Diary": "A systematic daily log of seizure events, potential triggers, and "
                             "lifestyle factors used for clinical management.",
            "Prodrome": "Warning signs (mood changes, fatigue, irritability) that may precede a "
                        "seizure by hours or days.",
            "Trigger Stacking": "The cumulative effect of multiple sub-threshold triggers (e.g., "
                                "poor sleep + missed medication + stress) that together precipitate "
                                "a seizure.",
            "Self-Management": "Patient-directed strategies (trigger avoidance, adherence, lifestyle "
                               "modification) that complement medical treatment.",
            "Seizure Precipitant": "Any factor that acutely lowers seizure threshold and increases "
                                   "the likelihood of a seizure in a person with epilepsy."
        }
    }
