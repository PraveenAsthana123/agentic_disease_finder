"""Clinical Decisions Dashboard — Human-in-the-Loop AI oversight analytics from clinical.db.

Tracks how neurologists confirm/override AI predictions, agreement rates,
reviewer workload, and confidence distribution across clinical decisions.

Sources:
- clinical_decisions table (id, patient_id, analysis_id, ai_prediction, ai_confidence,
  top_channels, artifact_risk, time_window, neurologist_agreement, final_decision,
  reviewer, note, created_at)
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


# ──────────────────────────────────────────────────────────────
#  /api/clinical-decisions/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """High-level HITL decision KPIs, agreement/decision distributions, timeline."""
    conn = _conn()
    cur = conn.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*) FROM clinical_decisions")
    total_decisions = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM clinical_decisions WHERE neurologist_agreement = 'Agree'")
    agree_count = cur.fetchone()[0] or 0
    agreement_rate_pct = round(agree_count / total_decisions * 100, 1) if total_decisions else 0.0

    cur.execute("SELECT COUNT(*) FROM clinical_decisions WHERE final_decision = 'Override'")
    override_count = cur.fetchone()[0] or 0
    override_rate_pct = round(override_count / total_decisions * 100, 1) if total_decisions else 0.0

    cur.execute("SELECT ROUND(AVG(ai_confidence), 3) FROM clinical_decisions")
    avg_confidence = cur.fetchone()[0] or 0.0

    cur.execute("SELECT COUNT(DISTINCT reviewer) FROM clinical_decisions")
    unique_reviewers = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM clinical_decisions")
    unique_patients = cur.fetchone()[0] or 0

    # Agreement distribution
    cur.execute("SELECT neurologist_agreement, COUNT(*) FROM clinical_decisions GROUP BY neurologist_agreement")
    agreement_distribution = dict(cur.fetchall())

    # Decision distribution
    cur.execute("SELECT final_decision, COUNT(*) FROM clinical_decisions GROUP BY final_decision")
    decision_distribution = dict(cur.fetchall())

    # Monthly timeline
    cur.execute("""
        SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS decisions
        FROM clinical_decisions
        WHERE created_at IS NOT NULL
        GROUP BY month ORDER BY month
    """)
    monthly_decisions = [{"date": r[0], "decisions": r[1]} for r in cur.fetchall()]

    # Confidence distribution (buckets)
    cur.execute("""
        SELECT
            CASE
                WHEN ai_confidence < 0.5 THEN '0.0-0.5'
                WHEN ai_confidence < 0.6 THEN '0.5-0.6'
                WHEN ai_confidence < 0.7 THEN '0.6-0.7'
                WHEN ai_confidence < 0.8 THEN '0.7-0.8'
                WHEN ai_confidence < 0.9 THEN '0.8-0.9'
                ELSE '0.9-1.0'
            END AS bucket,
            COUNT(*) AS count
        FROM clinical_decisions
        GROUP BY bucket ORDER BY bucket
    """)
    confidence_distribution = [{"bucket": r[0], "count": r[1]} for r in cur.fetchall()]

    conn.close()
    return {
        "total_decisions": total_decisions,
        "agreement_rate_pct": agreement_rate_pct,
        "override_rate_pct": override_rate_pct,
        "avg_confidence": round(avg_confidence, 3),
        "unique_reviewers": unique_reviewers,
        "unique_patients": unique_patients,
        "agreement_distribution": agreement_distribution,
        "decision_distribution": decision_distribution,
        "monthly_decisions": monthly_decisions,
        "confidence_distribution": confidence_distribution,
    }


# ──────────────────────────────────────────────────────────────
#  /api/clinical-decisions/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Detailed breakdown — reviewer workload, prediction cross-tab, disagreement analysis."""
    conn = _conn()
    cur = conn.cursor()

    # Reviewer workload
    cur.execute("""
        SELECT reviewer, COUNT(*) AS total,
               SUM(CASE WHEN neurologist_agreement = 'Agree' THEN 1 ELSE 0 END) AS agrees,
               SUM(CASE WHEN neurologist_agreement = 'Disagree' THEN 1 ELSE 0 END) AS disagrees,
               SUM(CASE WHEN final_decision = 'Override' THEN 1 ELSE 0 END) AS overrides,
               ROUND(AVG(ai_confidence), 3) AS avg_confidence
        FROM clinical_decisions
        GROUP BY reviewer ORDER BY total DESC
    """)
    reviewer_workload = _dict_rows(cur)

    # Prediction by decision cross-tab
    cur.execute("""
        SELECT ai_prediction, final_decision, COUNT(*) AS count
        FROM clinical_decisions
        GROUP BY ai_prediction, final_decision
    """)
    prediction_decision_cross = defaultdict(dict)
    for row in cur.fetchall():
        prediction_decision_cross[row[0]][row[1]] = row[2]
    prediction_decision_cross = dict(prediction_decision_cross)

    # High-disagreement cases (Disagree with details)
    cur.execute("""
        SELECT patient_id, ai_prediction, ai_confidence, top_channels,
               artifact_risk, final_decision, reviewer, note, created_at
        FROM clinical_decisions
        WHERE neurologist_agreement = 'Disagree'
        ORDER BY created_at DESC
    """)
    disagreement_cases = _dict_rows(cur)

    # Per-reviewer performance
    cur.execute("""
        SELECT reviewer,
               COUNT(*) AS total,
               ROUND(100.0 * SUM(CASE WHEN neurologist_agreement = 'Agree' THEN 1 ELSE 0 END) / COUNT(*), 1) AS agree_rate,
               ROUND(100.0 * SUM(CASE WHEN final_decision = 'Override' THEN 1 ELSE 0 END) / COUNT(*), 1) AS override_rate,
               ROUND(100.0 * SUM(CASE WHEN final_decision = 'Escalate' THEN 1 ELSE 0 END) / COUNT(*), 1) AS escalate_rate
        FROM clinical_decisions
        GROUP BY reviewer ORDER BY total DESC
    """)
    reviewer_performance = _dict_rows(cur)

    # Artifact risk vs disagreement
    cur.execute("""
        SELECT artifact_risk,
               COUNT(*) AS total,
               SUM(CASE WHEN neurologist_agreement = 'Disagree' THEN 1 ELSE 0 END) AS disagree_count,
               ROUND(100.0 * SUM(CASE WHEN neurologist_agreement = 'Disagree' THEN 1 ELSE 0 END) / COUNT(*), 1) AS disagree_rate
        FROM clinical_decisions
        GROUP BY artifact_risk ORDER BY disagree_rate DESC
    """)
    artifact_vs_disagreement = _dict_rows(cur)

    # Recent decisions (last 30)
    cur.execute("""
        SELECT patient_id, ai_prediction, ai_confidence, top_channels,
               artifact_risk, time_window, neurologist_agreement, final_decision,
               reviewer, note, created_at
        FROM clinical_decisions
        ORDER BY created_at DESC LIMIT 30
    """)
    recent_decisions = _dict_rows(cur)

    conn.close()
    return {
        "reviewer_workload": reviewer_workload,
        "prediction_decision_cross": prediction_decision_cross,
        "disagreement_cases": disagreement_cases,
        "reviewer_performance": reviewer_performance,
        "artifact_vs_disagreement": artifact_vs_disagreement,
        "recent_decisions": recent_decisions,
    }


# ──────────────────────────────────────────────────────────────
#  /api/clinical-decisions/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Definitions and glossary for clinical decisions HITL domain."""
    return {
        "decision_types": [
            {"type": "Confirm", "description": "Neurologist confirms AI prediction as clinically accurate — no change to diagnosis"},
            {"type": "Override", "description": "Neurologist disagrees with AI and provides alternative diagnosis or classification"},
            {"type": "Defer", "description": "Decision deferred pending additional data (repeat EEG, MRI, clinical correlation)"},
            {"type": "Escalate", "description": "Case escalated to epilepsy board or senior neurologist for multidisciplinary review"},
        ],
        "agreement_levels": [
            {"level": "Agree", "description": "Full agreement with AI prediction — prediction accepted as-is"},
            {"level": "Partial", "description": "Partial agreement — correct category but disagreement on specifics (lateralization, subtype)"},
            {"level": "Disagree", "description": "Full disagreement — AI prediction considered incorrect based on clinical judgment"},
        ],
        "artifact_risk_levels": [
            {"level": "None", "description": "No artifact contamination detected — clean EEG recording"},
            {"level": "Low", "description": "Minor artifact present (eye blinks, muscle) but not affecting interpretation"},
            {"level": "Medium", "description": "Moderate artifact that may influence AI confidence; manual review recommended"},
            {"level": "High", "description": "Significant artifact contamination — AI prediction unreliable, repeat recording advised"},
        ],
        "ai_prediction_categories": [
            {"category": "Epilepsy", "description": "General epilepsy classification based on interictal/ictal EEG patterns"},
            {"category": "Normal", "description": "No epileptiform abnormalities detected — normal EEG within age-appropriate limits"},
            {"category": "PNES", "description": "Psychogenic Non-Epileptic Seizures — events without epileptic EEG correlate"},
            {"category": "Focal Seizure", "description": "Seizure originating from a localized brain region (temporal, frontal, parietal, occipital)"},
            {"category": "Generalized Seizure", "description": "Seizure involving bilateral hemispheric networks from onset (absence, tonic-clonic, myoclonic)"},
        ],
        "eeg_channel_descriptions": [
            {"channel": "Fp1/Fp2", "description": "Frontopolar — eye movement artifact common, prefrontal activity"},
            {"channel": "F3/F4/F7/F8", "description": "Frontal — motor planning, frontal lobe seizures, muscle artifact zone"},
            {"channel": "T3/T4/T5/T6", "description": "Temporal — most common epilepsy focus (mesial temporal sclerosis)"},
            {"channel": "C3/C4", "description": "Central — sensorimotor cortex, Rolandic epilepsy focus"},
            {"channel": "P3/P4", "description": "Parietal — sensory processing, posterior seizure propagation"},
            {"channel": "O1/O2", "description": "Occipital — visual cortex, occipital lobe epilepsy, photic driving"},
            {"channel": "Fz/Cz/Pz", "description": "Midline — generalized discharges, vertex waves, reference montage"},
        ],
        "clinical_notes": [
            "AI confidence below 0.60 should always trigger manual review",
            "High artifact risk cases require repeat recording before clinical decision",
            "Override decisions must include documented clinical rationale",
            "Escalated cases enter the weekly epilepsy multidisciplinary conference",
            "Agreement rate below 70% for a prediction category triggers model retraining review",
        ],
        "glossary": [
            {"term": "HITL", "definition": "Human-in-the-Loop — clinical AI paradigm where human experts validate/override AI outputs"},
            {"term": "Override Rate", "definition": "Percentage of AI predictions that neurologists override with alternative diagnosis"},
            {"term": "Agreement Rate", "definition": "Percentage of cases where neurologist fully agrees with AI prediction"},
            {"term": "AI Confidence", "definition": "Model output probability (0-1) representing certainty in the predicted class"},
            {"term": "Epileptiform", "definition": "EEG patterns characteristic of epilepsy (spikes, sharp waves, spike-wave complexes)"},
            {"term": "Interictal", "definition": "EEG activity between seizures — used for epilepsy diagnosis and localization"},
            {"term": "Ictal", "definition": "EEG activity during a seizure — rhythmic evolution with clinical correlate"},
            {"term": "Lateralization", "definition": "Determining which hemisphere or brain region a seizure originates from"},
            {"term": "Semiology", "definition": "Clinical signs and symptoms during a seizure used for localization"},
            {"term": "Montage", "definition": "EEG display configuration showing voltage differences between electrode pairs"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
    print("\n=== DEFINITIONS ===")
    print(json.dumps(definitions(), indent=2, default=str))
