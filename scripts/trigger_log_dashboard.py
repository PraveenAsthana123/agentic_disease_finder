"""Seizure Trigger Log Dashboard — trigger diary analytics from clinical.db.

Tracks patient seizure trigger logs including sleep, stress, medication
adherence, lifestyle factors, and seizure occurrence patterns.

Sources:
- seizure_trigger_logs table (patient_id, log_date, sleep_hours, sleep_quality,
  stress_level, fatigue_level, mood_score, exercise_minutes, caffeine_mg,
  alcohol_units, screen_time_hours, medication_adherence, missed_doses,
  primary_trigger, seizure_occurred, seizure_type, seizure_duration_sec, notes)
"""

import sqlite3
import datetime
import random
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

PRIMARY_TRIGGERS = [
    "sleep_deprivation", "stress", "missed_medication", "alcohol",
    "photosensitivity", "hormonal_changes", "illness", "fatigue", "dehydration",
]

SLEEP_QUALITIES = ["good", "fair", "poor", "very_poor"]

SEIZURE_TYPES = [
    "focal aware", "focal impaired awareness", "focal to bilateral tonic-clonic",
    "generalized tonic-clonic", "absence", "myoclonic",
]

NOTES_OPTIONS = [
    None, None, None, None, None,
    "Patient reported aura before event",
    "Occurred during sleep",
    "Stressful work week reported",
    "Missed evening dose",
    "Poor sleep previous night",
    "High caffeine intake noted",
    "Alcohol consumption evening prior",
    "Screen exposure before bed",
    "Hormonal cycle noted",
    "Patient recovering from cold",
    "Dehydration during exercise",
    "Skipped breakfast",
    "Routine follow-up log",
    "Patient feeling well overall",
    "Increased fatigue reported",
]


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def ensure_table():
    """Create seizure_trigger_logs table and seed data if not exists."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS seizure_trigger_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            log_date TEXT NOT NULL,
            sleep_hours REAL,
            sleep_quality TEXT,
            stress_level INTEGER,
            fatigue_level INTEGER,
            mood_score INTEGER,
            exercise_minutes INTEGER,
            caffeine_mg INTEGER,
            alcohol_units REAL,
            screen_time_hours REAL,
            medication_adherence INTEGER,
            missed_doses INTEGER,
            primary_trigger TEXT,
            seizure_occurred INTEGER,
            seizure_type TEXT,
            seizure_duration_sec INTEGER,
            notes TEXT
        )
    """)
    cur.execute("SELECT COUNT(*) FROM seizure_trigger_logs")
    if cur.fetchone()[0] == 0:
        _seed(conn)
    conn.commit()
    conn.close()


def _seed(conn):
    """Seed realistic seizure trigger log data for 40 patients over 6 months."""
    cur = conn.cursor()
    rng = random.Random(42)
    patients = [f"PAT-{str(i).zfill(3)}" for i in range(1, 41)]
    base_date = datetime.date(2026, 1, 1)
    rows = []

    for pid in patients:
        # Each patient logs 3-7 entries over the 6-month window
        n_logs = rng.randint(3, 7)
        log_days = sorted(rng.sample(range(0, 181), n_logs))

        # Per-patient baseline traits
        base_sleep = rng.gauss(7.0, 1.0)
        base_stress = rng.randint(3, 6)
        base_adherence_prob = rng.uniform(0.65, 0.95)

        for day_offset in log_days:
            log_date = base_date + datetime.timedelta(days=day_offset)

            # Sleep
            sleep_hours = round(max(2.0, min(12.0, rng.gauss(base_sleep, 1.5))), 1)
            if sleep_hours < 5:
                sq = rng.choice(["poor", "very_poor"])
            elif sleep_hours < 6.5:
                sq = rng.choice(["poor", "fair"])
            elif sleep_hours < 8:
                sq = rng.choice(["fair", "good"])
            else:
                sq = rng.choice(["good", "good", "fair"])

            # Stress and fatigue
            stress = max(1, min(10, int(rng.gauss(base_stress, 2))))
            fatigue = max(1, min(10, int(rng.gauss(stress * 0.7 + 1.5, 1.5))))
            mood = max(1, min(10, int(rng.gauss(8 - stress * 0.5, 1.5))))

            # Lifestyle
            exercise = max(0, int(rng.gauss(30, 20)))
            caffeine = max(0, int(rng.gauss(150, 80)))
            alcohol = round(max(0.0, rng.gauss(0.5, 1.0)), 1)
            screen_time = round(max(0.5, rng.gauss(4.0, 2.0)), 1)

            # Medication adherence
            adherent = 1 if rng.random() < base_adherence_prob else 0
            missed = 0 if adherent else rng.randint(1, 3)

            # Primary trigger selection — weighted by current state
            trigger_weights = {
                "sleep_deprivation": 3.0 if sleep_hours < 5 else (1.5 if sleep_hours < 6 else 0.5),
                "stress": 3.0 if stress > 7 else (1.5 if stress > 5 else 0.5),
                "missed_medication": 4.0 if missed > 0 else 0.3,
                "alcohol": 3.0 if alcohol > 2.0 else (1.0 if alcohol > 0.5 else 0.2),
                "photosensitivity": 0.8,
                "hormonal_changes": 0.6,
                "illness": 0.4,
                "fatigue": 2.5 if fatigue > 7 else (1.0 if fatigue > 5 else 0.3),
                "dehydration": 0.5,
            }
            triggers = list(trigger_weights.keys())
            weights = [trigger_weights[t] for t in triggers]
            primary_trigger = rng.choices(triggers, weights=weights, k=1)[0]

            # Seizure occurrence — clinically realistic probabilities
            seizure_prob = 0.12  # baseline ~12%
            if sleep_hours < 5:
                seizure_prob += 0.20
            elif sleep_hours < 6:
                seizure_prob += 0.10
            if stress > 7:
                seizure_prob += 0.18
            elif stress > 5:
                seizure_prob += 0.08
            if missed > 0:
                seizure_prob += 0.22
            if alcohol > 2.0:
                seizure_prob += 0.12
            if fatigue > 7:
                seizure_prob += 0.10
            if primary_trigger == "photosensitivity":
                seizure_prob += 0.10
            seizure_prob = min(seizure_prob, 0.85)

            seizure_occurred = 1 if rng.random() < seizure_prob else 0

            seizure_type = None
            seizure_duration = None
            if seizure_occurred:
                seizure_type = rng.choice(SEIZURE_TYPES)
                seizure_duration = rng.randint(15, 300)

            note = rng.choice(NOTES_OPTIONS)

            rows.append((
                pid, log_date.isoformat(), sleep_hours, sq,
                stress, fatigue, mood, exercise, caffeine, alcohol,
                screen_time, adherent, missed, primary_trigger,
                seizure_occurred, seizure_type, seizure_duration, note,
            ))

    cur.executemany(
        "INSERT INTO seizure_trigger_logs "
        "(patient_id, log_date, sleep_hours, sleep_quality, stress_level, "
        "fatigue_level, mood_score, exercise_minutes, caffeine_mg, alcohol_units, "
        "screen_time_hours, medication_adherence, missed_doses, primary_trigger, "
        "seizure_occurred, seizure_type, seizure_duration_sec, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ──────────────────────────────────────────────────────────────
#  /api/trigger-logs/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """High-level seizure trigger log metrics."""
    ensure_table()
    conn = _conn()
    cur = conn.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*) FROM seizure_trigger_logs")
    total_logs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM seizure_trigger_logs")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT SUM(seizure_occurred) FROM seizure_trigger_logs")
    total_seizure_events = cur.fetchone()[0] or 0

    seizure_rate_pct = round(total_seizure_events / total_logs * 100, 1) if total_logs else 0.0

    cur.execute("SELECT ROUND(AVG(sleep_hours), 1) FROM seizure_trigger_logs")
    avg_sleep_hours = cur.fetchone()[0] or 0.0

    cur.execute(
        "SELECT ROUND(AVG(medication_adherence) * 100, 1) FROM seizure_trigger_logs"
    )
    medication_adherence_pct = cur.fetchone()[0] or 0.0

    kpis = {
        "total_logs": total_logs,
        "total_patients": total_patients,
        "seizure_rate_pct": seizure_rate_pct,
        "total_seizure_events": total_seizure_events,
        "avg_sleep_hours": avg_sleep_hours,
        "medication_adherence_pct": medication_adherence_pct,
    }

    # Primary trigger distribution
    cur.execute(
        "SELECT primary_trigger, COUNT(*) AS cnt FROM seizure_trigger_logs "
        "GROUP BY primary_trigger ORDER BY cnt DESC"
    )
    trigger_rows = cur.fetchall()
    primary_trigger_distribution = [
        {
            "trigger": r[0],
            "count": r[1],
            "pct": round(r[1] / total_logs * 100, 1) if total_logs else 0.0,
        }
        for r in trigger_rows
    ]

    # Seizure by month
    cur.execute(
        "SELECT SUBSTR(log_date, 1, 7) AS month, "
        "SUM(seizure_occurred) AS seizures, COUNT(*) AS total_logs "
        "FROM seizure_trigger_logs GROUP BY month ORDER BY month"
    )
    seizure_by_month = [
        {"month": r[0], "seizures": r[1], "total_logs": r[2]}
        for r in cur.fetchall()
    ]

    # Sleep quality distribution
    cur.execute(
        "SELECT sleep_quality, COUNT(*) FROM seizure_trigger_logs "
        "GROUP BY sleep_quality ORDER BY COUNT(*) DESC"
    )
    sleep_quality_distribution = [
        {"quality": r[0], "count": r[1]} for r in cur.fetchall()
    ]

    # Lifestyle averages: with seizure vs without
    lifestyle_factors = [
        ("sleep_hours", "Sleep Hours"),
        ("stress_level", "Stress Level"),
        ("fatigue_level", "Fatigue Level"),
        ("mood_score", "Mood Score"),
        ("exercise_minutes", "Exercise (min)"),
        ("caffeine_mg", "Caffeine (mg)"),
        ("alcohol_units", "Alcohol (units)"),
        ("screen_time_hours", "Screen Time (hrs)"),
    ]
    lifestyle_averages = []
    for col, label in lifestyle_factors:
        cur.execute(
            f"SELECT ROUND(AVG({col}), 2) FROM seizure_trigger_logs "
            f"WHERE seizure_occurred = 1"
        )
        with_sz = cur.fetchone()[0] or 0.0
        cur.execute(
            f"SELECT ROUND(AVG({col}), 2) FROM seizure_trigger_logs "
            f"WHERE seizure_occurred = 0"
        )
        without_sz = cur.fetchone()[0] or 0.0
        lifestyle_averages.append({
            "factor": label,
            "with_seizure": with_sz,
            "without_seizure": without_sz,
        })

    # Stress vs seizure
    cur.execute(
        "SELECT stress_level, "
        "ROUND(AVG(seizure_occurred) * 100, 1) AS seizure_pct, "
        "COUNT(*) AS count "
        "FROM seizure_trigger_logs GROUP BY stress_level ORDER BY stress_level"
    )
    stress_vs_seizure = [
        {"stress_level": r[0], "seizure_pct": r[1], "count": r[2]}
        for r in cur.fetchall()
    ]

    conn.close()
    return {
        "available": True,
        "kpis": kpis,
        "primary_trigger_distribution": primary_trigger_distribution,
        "seizure_by_month": seizure_by_month,
        "sleep_quality_distribution": sleep_quality_distribution,
        "lifestyle_averages": lifestyle_averages,
        "stress_vs_seizure": stress_vs_seizure,
    }


# ──────────────────────────────────────────────────────────────
#  /api/trigger-logs/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient detail, high-risk days, adherence issues, recent logs."""
    ensure_table()
    conn = _conn()
    cur = conn.cursor()

    # High-risk days: stress > 7 OR sleep < 5 OR missed_doses > 0
    cur.execute(
        "SELECT patient_id, log_date, primary_trigger, sleep_hours, "
        "stress_level, fatigue_level, missed_doses "
        "FROM seizure_trigger_logs "
        "WHERE stress_level > 7 OR sleep_hours < 5 OR missed_doses > 0 "
        "ORDER BY log_date DESC"
    )
    high_risk_days = _dict_rows(cur)

    # Adherence issues: medication_adherence = 0
    cur.execute(
        "SELECT patient_id, log_date, primary_trigger, seizure_occurred, "
        "medication_adherence, notes "
        "FROM seizure_trigger_logs "
        "WHERE medication_adherence = 0 "
        "ORDER BY log_date DESC"
    )
    adherence_issues = _dict_rows(cur)

    # Per-patient summary
    cur.execute(
        "SELECT patient_id, COUNT(*) AS total_logs, "
        "SUM(seizure_occurred) AS seizures, "
        "ROUND(AVG(seizure_occurred) * 100, 1) AS seizure_rate, "
        "ROUND(AVG(sleep_hours), 1) AS avg_sleep, "
        "ROUND(AVG(stress_level), 1) AS avg_stress, "
        "ROUND(AVG(mood_score), 1) AS avg_mood, "
        "ROUND(AVG(medication_adherence) * 100, 1) AS adherence_pct "
        "FROM seizure_trigger_logs GROUP BY patient_id ORDER BY patient_id"
    )
    per_patient = _dict_rows(cur)

    # Recent logs (last 30)
    cur.execute(
        "SELECT patient_id, log_date, sleep_hours, sleep_quality, "
        "stress_level, mood_score, seizure_occurred, primary_trigger, "
        "medication_adherence, notes "
        "FROM seizure_trigger_logs ORDER BY log_date DESC LIMIT 30"
    )
    recent_logs = _dict_rows(cur)

    conn.close()
    return {
        "high_risk_days": high_risk_days,
        "adherence_issues": adherence_issues,
        "per_patient": per_patient,
        "recent_logs": recent_logs,
    }


# ──────────────────────────────────────────────────────────────
#  /api/trigger-logs/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Trigger definitions, field descriptions, clinical notes, glossary."""
    return {
        "trigger_descriptions": {
            "sleep_deprivation": "Inadequate sleep duration or quality. Sleep deprivation is one "
                                 "of the most common and potent seizure triggers, lowering the "
                                 "seizure threshold significantly.",
            "stress": "Psychological or physiological stress. Chronic or acute stress elevates "
                      "cortisol levels, which can increase neuronal excitability and seizure "
                      "susceptibility.",
            "missed_medication": "Failure to take prescribed anti-epileptic drugs (AEDs) on "
                                 "schedule. Even a single missed dose can cause breakthrough "
                                 "seizures in many patients.",
            "alcohol": "Alcohol consumption, particularly binge drinking or withdrawal. Alcohol "
                       "affects GABA receptors and can lower seizure threshold, especially during "
                       "withdrawal.",
            "photosensitivity": "Exposure to flashing or flickering lights, bright patterns, or "
                                "screen flicker. Affects approximately 3-5% of epilepsy patients "
                                "(photosensitive epilepsy).",
            "hormonal_changes": "Hormonal fluctuations, particularly during menstrual cycle, "
                                "pregnancy, or menopause. Catamenial epilepsy affects up to 40% "
                                "of women with epilepsy.",
            "illness": "Acute illness including fever, infection, or systemic inflammation. Fever "
                       "is a particularly potent trigger, lowering the seizure threshold "
                       "substantially.",
            "fatigue": "Physical or mental exhaustion. Fatigue often co-occurs with sleep "
                       "deprivation and stress, creating a compounding effect on seizure risk.",
            "dehydration": "Insufficient fluid intake or excessive fluid loss. Dehydration can "
                           "alter electrolyte balance, particularly sodium, affecting neuronal "
                           "function.",
        },
        "field_descriptions": {
            "sleep_hours": "Total hours of sleep in the preceding night (0-12). Below 5 hours "
                           "is clinically significant for seizure risk.",
            "sleep_quality": "Subjective sleep quality rating: good, fair, poor, very_poor. "
                             "Correlates with sleep architecture disruption.",
            "stress_level": "Self-reported stress on a 1-10 scale. Levels above 7 are considered "
                            "high-risk for seizure provocation.",
            "fatigue_level": "Self-reported fatigue on a 1-10 scale. Levels above 7 indicate "
                             "significant exhaustion.",
            "mood_score": "Self-reported mood on a 1-10 scale (10 = best). Low mood may correlate "
                          "with increased seizure frequency.",
            "exercise_minutes": "Minutes of physical activity in the preceding 24 hours. Moderate "
                                "exercise is generally protective.",
            "caffeine_mg": "Estimated caffeine intake in milligrams. Excessive caffeine (>400mg) "
                           "may lower seizure threshold.",
            "alcohol_units": "Standard alcohol units consumed. Even moderate consumption can be a "
                             "trigger for some patients.",
            "screen_time_hours": "Hours of screen exposure. Relevant for photosensitive patients "
                                 "and sleep hygiene.",
            "medication_adherence": "Whether all prescribed AED doses were taken (1 = yes, 0 = no). "
                                    "Binary adherence flag.",
            "missed_doses": "Number of AED doses missed in the preceding 24 hours. Even one missed "
                            "dose is clinically significant.",
            "primary_trigger": "The dominant suspected trigger for the log entry, as identified by "
                               "clinical assessment or patient report.",
            "seizure_occurred": "Whether a seizure event was recorded during this period "
                                "(1 = yes, 0 = no).",
            "seizure_type": "ILAE classification of the seizure type if one occurred.",
            "seizure_duration_sec": "Duration of the seizure event in seconds, if one occurred.",
        },
        "clinical_notes": [
            "Sleep deprivation and missed medication are the two most modifiable seizure "
            "triggers. Patient education should prioritize these.",
            "Stress management techniques (CBT, mindfulness) have been shown to reduce "
            "seizure frequency by 20-50% in some studies.",
            "Alcohol is both a direct trigger and an indirect one (via sleep disruption and "
            "medication non-adherence).",
            "Trigger logs should be maintained for a minimum of 3 months to identify "
            "reliable patterns.",
            "Multiple triggers often co-occur; a single seizure event may have multiple "
            "contributing factors.",
            "Catamenial patterns require at least 3 menstrual cycles of data to confirm.",
            "Photosensitive patients should be counselled on screen brightness, viewing "
            "distance, and blue-light filters.",
            "AED blood levels should be checked when adherence issues are identified in "
            "trigger logs.",
        ],
        "glossary": {
            "Seizure Trigger": "An identifiable factor that increases the probability of a "
                               "seizure occurring in a susceptible individual.",
            "Seizure Threshold": "The level of stimulation required to induce a seizure. Lower "
                                 "thresholds mean higher susceptibility.",
            "AED": "Anti-Epileptic Drug — medication prescribed to prevent or reduce seizure "
                   "frequency.",
            "Catamenial Epilepsy": "Epilepsy in which seizure frequency is linked to the "
                                   "menstrual cycle, often worsening perimenstrually.",
            "Photosensitive Epilepsy": "A form of epilepsy where seizures are triggered by "
                                       "visual stimuli such as flashing lights.",
            "ILAE": "International League Against Epilepsy — the global body that classifies "
                    "seizure types and epilepsy syndromes.",
            "Breakthrough Seizure": "A seizure occurring despite ongoing AED therapy, often due "
                                    "to a trigger or subtherapeutic drug levels.",
            "Seizure Diary": "A patient-maintained log of seizure events, triggers, and "
                             "associated factors — the clinical basis for trigger log dashboards.",
            "Medication Adherence": "The degree to which a patient takes medications as "
                                    "prescribed. Non-adherence is the leading cause of "
                                    "breakthrough seizures.",
            "Prodrome": "Warning symptoms occurring hours to days before a seizure, such as "
                        "mood changes, irritability, or headache.",
        },
    }


if __name__ == "__main__":
    import json
    ensure_table()
    print("=== Overview ===")
    print(json.dumps(overview(), indent=2))
    print("\n=== Breakdown (summary) ===")
    b = breakdown()
    print(f"Patients: {len(b['per_patient'])}, High-risk days: {len(b['high_risk_days'])}")
    print(f"Adherence issues: {len(b['adherence_issues'])}, Recent: {len(b['recent_logs'])}")
    print("\n=== Definitions ===")
    d = definitions()
    print(f"Triggers defined: {len(d['trigger_descriptions'])}, "
          f"Fields: {len(d['field_descriptions'])}")
    print(f"Clinical notes: {len(d['clinical_notes'])}, Glossary: {len(d['glossary'])}")
