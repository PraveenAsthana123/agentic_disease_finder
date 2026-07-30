"""Clinical Workbench Dashboard — unified Patient → EEG → AI → Explainability → Human → Audit view.

Assembles the full clinical decision pipeline from real DB tables:
- patients / analyses: patient demographics + AI analysis results
- clinical_decisions: neurologist confirm/override of AI predictions
- expert_reviews: specialist findings and agreement
- hitl_reviews: human-in-the-loop decision overrides
- explainability_gt: ground-truth explainability features per disease
- transaction_log: full audit trail of every action
"""

import json
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
#  /api/workbench/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Pipeline-wide KPIs: patients analysed, AI predictions, human reviews,
    agreement rates, audit completeness, explainability coverage."""
    conn = _conn()
    cur = conn.cursor()

    # Patient counts
    cur.execute("SELECT COUNT(*) FROM patients")
    total_patients = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM analyses")
    patients_analysed = cur.fetchone()[0] or 0

    # Analysis counts
    cur.execute("SELECT COUNT(*) FROM analyses")
    total_analyses = cur.fetchone()[0] or 0

    cur.execute("SELECT disease, COUNT(*) FROM analyses GROUP BY disease ORDER BY COUNT(*) DESC")
    analyses_by_disease = dict(cur.fetchall())

    cur.execute("SELECT signal_quality, COUNT(*) FROM analyses GROUP BY signal_quality")
    quality_distribution = dict(cur.fetchall())

    # AI prediction summary
    cur.execute("SELECT predicted_label, COUNT(*) FROM analyses GROUP BY predicted_label ORDER BY COUNT(*) DESC")
    prediction_distribution = dict(cur.fetchall())

    cur.execute("SELECT ROUND(AVG(confidence), 3) FROM analyses")
    avg_ai_confidence = cur.fetchone()[0] or 0.0

    # Clinical decisions (human oversight)
    cur.execute("SELECT COUNT(*) FROM clinical_decisions")
    total_decisions = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM clinical_decisions WHERE neurologist_agreement = 'Agree'")
    agree_count = cur.fetchone()[0] or 0
    agreement_rate_pct = round(agree_count / total_decisions * 100, 1) if total_decisions else 0.0

    cur.execute("SELECT final_decision, COUNT(*) FROM clinical_decisions GROUP BY final_decision")
    decision_distribution = dict(cur.fetchall())

    # Expert reviews
    cur.execute("SELECT COUNT(*) FROM expert_reviews")
    total_expert_reviews = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM expert_reviews WHERE agree_with_ai = 'agree'")
    expert_agree = cur.fetchone()[0] or 0

    # HITL reviews
    cur.execute("SELECT COUNT(*) FROM hitl_reviews")
    total_hitl_reviews = cur.fetchone()[0] or 0

    # Explainability coverage
    cur.execute("SELECT COUNT(*) FROM explainability_gt")
    explainability_entries = cur.fetchone()[0] or 0

    # Audit trail
    cur.execute("SELECT COUNT(*) FROM transaction_log")
    total_audit_events = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM transaction_log WHERE patient_id IS NOT NULL")
    audited_patients = cur.fetchone()[0] or 0

    # Pipeline completion: how many patients have gone through all stages
    cur.execute("""
        SELECT COUNT(DISTINCT p.patient_id) FROM patients p
        JOIN analyses a ON a.patient_id = p.patient_id
        JOIN clinical_decisions cd ON cd.patient_id = p.patient_id
    """)
    full_pipeline_patients = cur.fetchone()[0] or 0

    conn.close()

    return {
        "total_patients": total_patients,
        "patients_analysed": patients_analysed,
        "total_analyses": total_analyses,
        "analyses_by_disease": analyses_by_disease,
        "quality_distribution": quality_distribution,
        "prediction_distribution": prediction_distribution,
        "avg_ai_confidence": avg_ai_confidence,
        "total_decisions": total_decisions,
        "agreement_rate_pct": agreement_rate_pct,
        "decision_distribution": decision_distribution,
        "total_expert_reviews": total_expert_reviews,
        "expert_agreement_count": expert_agree,
        "total_hitl_reviews": total_hitl_reviews,
        "explainability_entries": explainability_entries,
        "total_audit_events": total_audit_events,
        "audited_patients": audited_patients,
        "full_pipeline_patients": full_pipeline_patients,
        "pipeline_stages": [
            {"stage": "Patient", "icon": "👤", "count": total_patients},
            {"stage": "EEG Analysis", "icon": "🧠", "count": total_analyses},
            {"stage": "AI Prediction", "icon": "🤖", "count": total_analyses},
            {"stage": "Explainability", "icon": "🔍", "count": explainability_entries},
            {"stage": "Human Review", "icon": "👨‍⚕️", "count": total_decisions + total_expert_reviews + total_hitl_reviews},
            {"stage": "Audit", "icon": "📋", "count": total_audit_events},
        ],
    }


# ──────────────────────────────────────────────────────────────
#  /api/workbench/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient pipeline status: which patients have completed each stage,
    recent analyses with decisions, and reviewer workload."""
    conn = _conn()
    cur = conn.cursor()

    # Per-patient pipeline tracker
    cur.execute("""
        SELECT
            p.patient_id,
            p.name,
            p.age,
            p.gender AS sex,
            COUNT(DISTINCT a.id) AS analysis_count,
            COUNT(DISTINCT cd.id) AS decision_count,
            COUNT(DISTINCT er.id) AS expert_review_count
        FROM patients p
        LEFT JOIN analyses a ON a.patient_id = p.patient_id
        LEFT JOIN clinical_decisions cd ON cd.patient_id = p.patient_id
        LEFT JOIN expert_reviews er ON er.patient_id = p.patient_id
        GROUP BY p.patient_id
        ORDER BY p.patient_id
    """)
    patient_pipeline = _dict_rows(cur)

    # Recent analyses with their decisions
    cur.execute("""
        SELECT
            a.id AS analysis_id,
            a.patient_id,
            a.disease,
            a.predicted_label,
            a.confidence,
            a.signal_quality,
            a.created_at AS analysis_date,
            cd.ai_prediction AS cd_ai_prediction,
            cd.neurologist_agreement,
            cd.final_decision,
            cd.reviewer,
            cd.note AS decision_note,
            cd.created_at AS decision_date
        FROM analyses a
        LEFT JOIN clinical_decisions cd ON cd.analysis_id = a.id
        ORDER BY a.created_at DESC
        LIMIT 50
    """)
    recent_cases = _dict_rows(cur)

    # Reviewer workload
    cur.execute("""
        SELECT reviewer, COUNT(*) AS cases,
               SUM(CASE WHEN final_decision = 'Confirm' THEN 1 ELSE 0 END) AS confirms,
               SUM(CASE WHEN final_decision = 'Override' THEN 1 ELSE 0 END) AS overrides,
               SUM(CASE WHEN final_decision = 'Escalate' THEN 1 ELSE 0 END) AS escalations
        FROM clinical_decisions
        GROUP BY reviewer
        ORDER BY cases DESC
    """)
    reviewer_workload = _dict_rows(cur)

    # Explainability ground truth per disease
    cur.execute("SELECT patient_id, fields_json, created_at FROM explainability_gt ORDER BY created_at DESC")
    xai_rows = _dict_rows(cur)
    explainability = []
    for row in xai_rows:
        fields = {}
        if row.get("fields_json"):
            try:
                fields = json.loads(row["fields_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        explainability.append({
            "id": row["patient_id"],
            "features": fields.get("Key_EEG_Features_Used", []),
            "channels": fields.get("Most_Important_Channels", []),
            "rationale": fields.get("Clinical_Rationale", ""),
            "created_at": row["created_at"],
        })

    # Audit trail summary by action type
    cur.execute("""
        SELECT
            CASE
                WHEN action LIKE '%predict%' OR action LIKE '%analys%' THEN 'AI Analysis'
                WHEN action LIKE '%override%' OR action LIKE '%review%' OR action LIKE '%decision%' THEN 'Human Review'
                WHEN action LIKE '%upload%' THEN 'Data Upload'
                WHEN action LIKE '%export%' THEN 'Export'
                ELSE 'Other'
            END AS category,
            COUNT(*) AS event_count
        FROM transaction_log
        GROUP BY category
        ORDER BY event_count DESC
    """)
    audit_by_category = _dict_rows(cur)

    conn.close()

    return {
        "patient_pipeline": patient_pipeline,
        "recent_cases": recent_cases,
        "reviewer_workload": reviewer_workload,
        "explainability": explainability,
        "audit_by_category": audit_by_category,
    }


# ──────────────────────────────────────────────────────────────
#  /api/workbench/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Workbench pipeline definitions, stage descriptions, and glossary."""
    return {
        "title": "Clinical Workbench — Pipeline Definitions",
        "pipeline_stages": {
            "Patient": "Demographics, clinical history, and referral information for each patient in the system.",
            "EEG Analysis": "Raw EEG signal acquisition, preprocessing, artifact rejection, and feature extraction (band powers, entropy, Hjorth parameters).",
            "AI Prediction": "Machine learning model inference producing disease classification (e.g., Epilepsy vs Control) with confidence scores.",
            "Explainability": "Ground-truth feature attribution — which EEG features and channels drove the AI prediction, with clinical rationale.",
            "Human Review": "Neurologist confirms, overrides, or escalates the AI prediction. Includes expert specialist reviews and HITL override decisions.",
            "Audit": "Immutable transaction log recording every action (upload, analysis, decision, export) with timestamps and actor.",
        },
        "decision_types": {
            "Confirm": "Neurologist agrees with the AI prediction — no change to the clinical label.",
            "Override": "Neurologist disagrees and replaces the AI prediction with a different clinical label.",
            "Escalate": "Case flagged for additional specialist review or multidisciplinary team discussion.",
            "Defer": "Decision postponed pending additional data (e.g., repeat EEG, MRI correlation).",
        },
        "agreement_levels": {
            "Agree": "Full agreement between AI prediction and neurologist assessment.",
            "Partial": "Partial agreement — neurologist agrees with the disease category but modifies severity, onset zone, or subtype.",
            "Disagree": "Neurologist disagrees with the AI prediction entirely.",
        },
        "quality_grades": {
            "Good": "Signal-to-noise ratio adequate for reliable analysis; minimal artifacts.",
            "Fair": "Some artifact contamination; analysis possible with caution.",
            "Poor": "Significant artifacts; results should be interpreted with caution or EEG repeated.",
        },
        "glossary": {
            "HITL": "Human-in-the-Loop — a human expert validates or overrides AI outputs before clinical use.",
            "XAI": "Explainable AI — techniques that make AI predictions interpretable to clinicians.",
            "Artifact Risk": "Likelihood that EEG artifacts (muscle, eye movement, electrode) are affecting the AI prediction.",
            "Confidence": "AI model's self-assessed probability that its prediction is correct (0.0–1.0).",
            "Transaction Log": "Immutable audit trail recording every clinical action with timestamp and actor identity.",
        },
    }
