"""Seed clinical_decisions table with ~75 realistic HITL decision records."""

import sqlite3
import random
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

PATIENTS = [f"P{str(i).zfill(4)}" for i in range(1, 46)]  # P0001-P0045

AI_PREDICTIONS = [
    "Epilepsy", "Normal", "PNES", "Focal Seizure", "Generalized Seizure"
]

EEG_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "T5", "T6",
                "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]

ARTIFACT_RISKS = ["None", "Low", "Medium", "High"]

TIME_WINDOWS = ["0-30s", "30-60s", "60-90s", "90-120s", "full recording",
                "2:15-2:45", "3:00-3:30", "1:30-2:00", "4:00-4:30", "5:00-5:30"]

AGREEMENTS = ["Agree", "Disagree", "Partial"]

DECISIONS = ["Confirm", "Override", "Defer", "Escalate"]

REVIEWERS = ["Dr. Patel", "Dr. Singh", "Dr. Kumar", "Dr. Sharma", "Dr. Gupta"]

NOTES = [
    "Clear epileptiform discharges in temporal lobe",
    "AI confidence too low for automated diagnosis",
    "Artifact contamination requires repeat recording",
    "Patient history supports AI prediction",
    "Disagreement on lateralization — need video correlation",
    "Focal onset confirmed with clinical semiology",
    "PNES suspected — refer for psychiatric evaluation",
    "Normal variant — benign temporal sharp waves",
    "Generalized spike-wave consistent with JME",
    "Override: artifact mimicking spikes in frontal channels",
    "High confidence prediction aligns with clinical impression",
    "Escalate to epilepsy conference for surgical planning",
    "Defer pending MRI correlation",
    "Partial agreement — seizure type correct but localization uncertain",
    "Channel selection optimal — AI identified correct focus",
    "Low artifact risk — clean recording supports diagnosis",
    "Repeat EEG recommended for confirmation",
    "Interictal pattern — clinical correlation needed",
    "AI missed subtle temporal slowing",
    "Override: non-epileptic myoclonus misclassified",
]


def seed():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Clear existing data
    cur.execute("DELETE FROM clinical_decisions")

    start_date = datetime(2026, 1, 5)
    end_date = datetime(2026, 7, 15)
    date_range = (end_date - start_date).days

    rows = []
    for i in range(75):
        patient_id = random.choice(PATIENTS)
        analysis_id = random.randint(100, 999)
        ai_prediction = random.choice(AI_PREDICTIONS)

        # Confidence varies by prediction type
        if ai_prediction == "Normal":
            ai_confidence = round(random.uniform(0.70, 0.98), 2)
        elif ai_prediction in ("Focal Seizure", "Generalized Seizure"):
            ai_confidence = round(random.uniform(0.55, 0.95), 2)
        else:
            ai_confidence = round(random.uniform(0.45, 0.92), 2)

        # Top channels (2-4 channels)
        n_channels = random.randint(2, 4)
        top_channels = ", ".join(random.sample(EEG_CHANNELS, n_channels))

        artifact_risk = random.choices(
            ARTIFACT_RISKS, weights=[30, 35, 25, 10]
        )[0]

        time_window = random.choice(TIME_WINDOWS)

        # Agreement correlates with confidence
        if ai_confidence >= 0.85:
            agreement = random.choices(AGREEMENTS, weights=[70, 10, 20])[0]
        elif ai_confidence >= 0.65:
            agreement = random.choices(AGREEMENTS, weights=[45, 25, 30])[0]
        else:
            agreement = random.choices(AGREEMENTS, weights=[20, 50, 30])[0]

        # Decision correlates with agreement
        if agreement == "Agree":
            decision = random.choices(DECISIONS, weights=[80, 5, 10, 5])[0]
        elif agreement == "Disagree":
            decision = random.choices(DECISIONS, weights=[5, 70, 10, 15])[0]
        else:
            decision = random.choices(DECISIONS, weights=[30, 20, 35, 15])[0]

        reviewer = random.choice(REVIEWERS)
        note = random.choice(NOTES)

        # Spread dates across the range
        days_offset = random.randint(0, date_range)
        created_at = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d %H:%M:%S")

        rows.append((
            patient_id, analysis_id, ai_prediction, ai_confidence,
            top_channels, artifact_risk, time_window,
            agreement, decision, reviewer, note, created_at
        ))

    cur.executemany("""
        INSERT INTO clinical_decisions
        (patient_id, analysis_id, ai_prediction, ai_confidence,
         top_channels, artifact_risk, time_window,
         neurologist_agreement, final_decision, reviewer, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    print(f"Inserted {len(rows)} clinical decision records")
    cur.execute("SELECT COUNT(*) FROM clinical_decisions")
    print(f"Total rows in clinical_decisions: {cur.fetchone()[0]}")
    conn.close()


if __name__ == "__main__":
    seed()
