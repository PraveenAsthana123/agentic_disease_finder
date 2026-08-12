"""Post-Market Surveillance (PMS) AI Performance Dashboard.

Tracks real-world AI-vs-clinician agreement, override rates, confidence
distributions, and vigilance indicators from the clinical_decisions table
(75 records, 30 patients, 5 prediction classes).

Regulatory context: EU MDR 2017/745 Art.83, FDA 21 CFR Part 803,
ISO 14971:2019 risk management, IEC 62304 software lifecycle,
NICE ECD6, IMDRF AI/ML SaMD good practices.
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


# ──────────────────────────────────────────────────────────────
#  /api/pms/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Top-level PMS KPIs and agreement heatmap."""
    conn = _conn()
    cur = conn.cursor()

    # Core counts
    cur.execute("SELECT COUNT(*) FROM clinical_decisions")
    total_decisions = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM clinical_decisions")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM clinical_decisions WHERE neurologist_agreement = 'Agree'")
    agree_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM clinical_decisions WHERE neurologist_agreement = 'Disagree'")
    disagree_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM clinical_decisions WHERE neurologist_agreement = 'Partial'")
    partial_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM clinical_decisions WHERE final_decision = 'Override'")
    override_count = cur.fetchone()[0]

    cur.execute("SELECT AVG(ai_confidence) FROM clinical_decisions")
    avg_conf = round(cur.fetchone()[0] or 0, 3)

    cur.execute(
        "SELECT AVG(ai_confidence) FROM clinical_decisions "
        "WHERE neurologist_agreement = 'Agree'"
    )
    avg_conf_agree = round(cur.fetchone()[0] or 0, 3)

    cur.execute(
        "SELECT AVG(ai_confidence) FROM clinical_decisions "
        "WHERE neurologist_agreement = 'Disagree'"
    )
    avg_conf_disagree = round(cur.fetchone()[0] or 0, 3)

    agree_rate = round(agree_count / total_decisions * 100, 1) if total_decisions else 0
    override_rate = round(override_count / total_decisions * 100, 1) if total_decisions else 0

    kpis = {
        "total_decisions": total_decisions,
        "total_patients": total_patients,
        "agree_count": agree_count,
        "disagree_count": disagree_count,
        "partial_count": partial_count,
        "override_count": override_count,
        "agree_rate_pct": agree_rate,
        "override_rate_pct": override_rate,
        "avg_confidence": avg_conf,
        "avg_conf_when_agree": avg_conf_agree,
        "avg_conf_when_disagree": avg_conf_disagree,
    }

    # Agreement distribution (donut data)
    agreement_distribution = [
        {"label": "Agree", "count": agree_count,
         "pct": round(agree_count / total_decisions * 100, 1)},
        {"label": "Partial", "count": partial_count,
         "pct": round(partial_count / total_decisions * 100, 1)},
        {"label": "Disagree", "count": disagree_count,
         "pct": round(disagree_count / total_decisions * 100, 1)},
    ]

    # Final decision distribution
    cur.execute(
        "SELECT final_decision, COUNT(*) AS cnt "
        "FROM clinical_decisions GROUP BY final_decision ORDER BY cnt DESC"
    )
    final_decision_dist = [
        {"decision": r[0], "count": r[1],
         "pct": round(r[1] / total_decisions * 100, 1)}
        for r in cur.fetchall()
    ]

    # Confidence bucket distribution
    cur.execute("""
        SELECT
          CASE
            WHEN ai_confidence >= 0.9 THEN '≥0.90'
            WHEN ai_confidence >= 0.8 THEN '0.80–0.89'
            WHEN ai_confidence >= 0.7 THEN '0.70–0.79'
            WHEN ai_confidence >= 0.6 THEN '0.60–0.69'
            ELSE '<0.60'
          END AS bucket,
          COUNT(*) AS cnt
        FROM clinical_decisions
        GROUP BY bucket
        ORDER BY MIN(ai_confidence) DESC
    """)
    confidence_buckets = [
        {"bucket": r[0], "count": r[1]} for r in cur.fetchall()
    ]

    # Reviewer workload
    cur.execute(
        "SELECT reviewer, COUNT(*) AS cnt, "
        "SUM(CASE WHEN final_decision='Override' THEN 1 ELSE 0 END) AS overrides "
        "FROM clinical_decisions GROUP BY reviewer ORDER BY cnt DESC"
    )
    reviewer_workload = [
        {"reviewer": r[0], "decisions": r[1], "overrides": r[2],
         "override_rate_pct": round(r[2] / r[1] * 100, 1) if r[1] else 0}
        for r in cur.fetchall()
    ]

    conn.close()
    return {
        "kpis": kpis,
        "agreement_distribution": agreement_distribution,
        "final_decision_distribution": final_decision_dist,
        "confidence_buckets": confidence_buckets,
        "reviewer_workload": reviewer_workload,
    }


# ──────────────────────────────────────────────────────────────
#  /api/pms/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-class performance, per-patient drill, and HITL vigilance."""
    conn = _conn()
    cur = conn.cursor()

    # Per-prediction-class performance
    cur.execute("""
        SELECT
            ai_prediction,
            COUNT(*) AS total,
            AVG(ai_confidence) AS avg_conf,
            SUM(CASE WHEN neurologist_agreement = 'Agree' THEN 1 ELSE 0 END) AS agree,
            SUM(CASE WHEN neurologist_agreement = 'Disagree' THEN 1 ELSE 0 END) AS disagree,
            SUM(CASE WHEN neurologist_agreement = 'Partial' THEN 1 ELSE 0 END) AS partial,
            SUM(CASE WHEN final_decision = 'Override' THEN 1 ELSE 0 END) AS overrides
        FROM clinical_decisions
        GROUP BY ai_prediction
        ORDER BY total DESC
    """)
    per_class = [
        {
            "ai_prediction": r[0],
            "total": r[1],
            "avg_confidence": round(r[2], 3),
            "agree": r[3],
            "disagree": r[4],
            "partial": r[5],
            "overrides": r[6],
            "agree_rate_pct": round(r[3] / r[1] * 100, 1) if r[1] else 0,
            "override_rate_pct": round(r[6] / r[1] * 100, 1) if r[1] else 0,
        }
        for r in cur.fetchall()
    ]

    # Per-patient summary
    cur.execute("""
        SELECT
            patient_id,
            COUNT(*) AS total_decisions,
            AVG(ai_confidence) AS avg_conf,
            SUM(CASE WHEN neurologist_agreement = 'Agree' THEN 1 ELSE 0 END) AS agree,
            SUM(CASE WHEN final_decision = 'Override' THEN 1 ELSE 0 END) AS overrides
        FROM clinical_decisions
        GROUP BY patient_id
        ORDER BY total_decisions DESC
    """)
    per_patient = [
        {
            "patient_id": r[0],
            "total_decisions": r[1],
            "avg_confidence": round(r[2], 3),
            "agree": r[3],
            "overrides": r[4],
            "agree_rate_pct": round(r[3] / r[1] * 100, 1) if r[1] else 0,
            "override_rate_pct": round(r[4] / r[1] * 100, 1) if r[1] else 0,
        }
        for r in cur.fetchall()
    ]

    # Artifact risk distribution
    cur.execute("""
        SELECT artifact_risk, COUNT(*) AS cnt,
               AVG(ai_confidence) AS avg_conf
        FROM clinical_decisions
        WHERE artifact_risk IS NOT NULL AND artifact_risk != ''
        GROUP BY artifact_risk ORDER BY cnt DESC
    """)
    artifact_risk_dist = [
        {"artifact_risk": r[0], "count": r[1], "avg_confidence": round(r[2], 3)}
        for r in cur.fetchall()
    ]

    # Confidence vs agreement cross-tab
    cur.execute("""
        SELECT
            CASE WHEN ai_confidence >= 0.8 THEN 'High (≥0.80)' ELSE 'Low (<0.80)' END AS conf_tier,
            neurologist_agreement,
            COUNT(*) AS cnt
        FROM clinical_decisions
        GROUP BY conf_tier, neurologist_agreement
        ORDER BY conf_tier, neurologist_agreement
    """)
    conf_agreement_crosstab = [
        {"conf_tier": r[0], "agreement": r[1], "count": r[2]}
        for r in cur.fetchall()
    ]

    # HITL reviews
    cur.execute("""
        SELECT id, patient_id, analysis_id, fields_json, created_at
        FROM hitl_reviews
        ORDER BY created_at DESC
    """)
    import json as _json
    hitl_reviews = []
    for r in cur.fetchall():
        fields = {}
        try:
            fields = _json.loads(r[3] or "{}")
        except Exception:
            pass
        hitl_reviews.append({
            "id": r[0], "patient_id": r[1], "analysis_id": r[2],
            "ai_prediction": fields.get("ai_prediction"),
            "decision": fields.get("decision"),
            "human_decision": fields.get("human_decision"),
            "reason_code": fields.get("reason_code"),
            "reviewer_id": fields.get("reviewer_id"),
            "created_at": r[4],
        })

    conn.close()
    return {
        "per_class_performance": per_class,
        "per_patient": per_patient,
        "artifact_risk_distribution": artifact_risk_dist,
        "confidence_agreement_crosstab": conf_agreement_crosstab,
        "hitl_reviews": hitl_reviews,
    }


# ──────────────────────────────────────────────────────────────
#  /api/pms/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    return {
        "title": "Post-Market Surveillance — AI Performance Monitoring",
        "regulatory_context": {
            "EU_MDR_Art83": (
                "EU MDR 2017/745 Article 83 mandates a proactive Post-Market Surveillance "
                "system for all CE-marked medical devices, including AI/ML-based SaMD. "
                "Manufacturers must collect and analyse real-world performance data "
                "continuously and update the risk/benefit assessment accordingly."
            ),
            "FDA_21CFR803": (
                "FDA 21 CFR Part 803 (Medical Device Reporting) requires manufacturers to "
                "report device malfunctions and serious injuries/deaths. For AI-based SaMD, "
                "significant performance degradation that could cause or contribute to patient "
                "harm must be reported as a malfunction event."
            ),
            "IMDRF_AI_ML": (
                "The IMDRF AI/ML SaMD working group recommends continuous performance "
                "monitoring via agreed-upon 'SaMD Pre-Specifications' (SPS) and an "
                "'Algorithm Change Protocol' (ACP) to manage model updates while maintaining "
                "regulatory compliance."
            ),
            "ISO_14971": (
                "ISO 14971:2019 (Risk Management for Medical Devices) requires manufacturers "
                "to evaluate the acceptability of residual risks against the overall benefit. "
                "PMS data feeds back into the risk management file to trigger design changes "
                "when new hazards emerge post-deployment."
            ),
            "IEC_62304": (
                "IEC 62304:2006+AMD1:2015 (Medical Device Software Lifecycle) applies to "
                "AI/ML model software. Class C software (most AI diagnostic tools) requires "
                "full traceability from requirements to testing. Model updates require a "
                "formal change management process."
            ),
        },
        "key_metrics": {
            "Agreement Rate": (
                "Percentage of AI predictions where the reviewing neurologist fully agreed "
                "with the AI conclusion (neurologist_agreement = 'Agree'). Target ≥75% for "
                "Class B SaMD; deviations trigger root-cause analysis."
            ),
            "Override Rate": (
                "Percentage of AI predictions where a clinician reversed the AI's "
                "recommendation (final_decision = 'Override'). Override rates >20% indicate "
                "systematic model errors and may trigger retraining or withdrawal."
            ),
            "Average Confidence": (
                "Mean AI posterior probability assigned to the predicted class. Confidence "
                "calibration is assessed via the Brier Score and reliability diagrams; "
                "over-confident models (high conf + low agreement) are a PMS red flag."
            ),
            "HITL Override": (
                "Human-In-The-Loop (HITL) override: a formal record where a clinician "
                "substitutes their own judgment for the AI prediction. HITL overrides are "
                "the primary vigilance signal in an AI PMS system."
            ),
        },
        "vigilance_thresholds": [
            {
                "signal": "Override rate >20%",
                "action": "Mandatory root-cause analysis; notify regulatory body if causal "
                          "to patient harm; consider model re-validation.",
            },
            {
                "signal": "Agreement rate <65% for any prediction class",
                "action": "Class-specific performance review; audit training data quality "
                          "for the affected class; update algorithm if needed.",
            },
            {
                "signal": "Average confidence >0.85 but agreement rate <70%",
                "action": "Confidence miscalibration detected — initiate recalibration "
                          "study; update SPS/ACP; notify QA.",
            },
            {
                "signal": ">3 HITL ART (artifact) overrides in 30 days",
                "action": "Signal quality degradation or new artifact type — update "
                          "preprocessing pipeline; review IEC 62304 change control.",
            },
        ],
        "references": [
            "EU MDR 2017/745, Article 83 — Post-Market Surveillance System (2017)",
            "FDA 21 CFR Part 803 — Medical Device Reporting (2024)",
            "IMDRF/SaMD N41 — Software as a Medical Device: Possible Framework (2014)",
            "IMDRF AI/ML-Based SaMD Working Group Report (2022)",
            "ISO 14971:2019 — Application of Risk Management to Medical Devices",
            "IEC 62304:2006+AMD1:2015 — Medical Device Software Lifecycle Processes",
            "NICE ECD6 — Evidence Standards for AI/Digital Health Technologies (2023)",
            "Benjamens S et al., 'The state of artificial intelligence-based FDA-approved "
            "algorithms in medicine', npj Digital Medicine 2020;3:118",
            "Muehlematter UJ et al., 'Approval of artificial intelligence and machine "
            "learning-based medical devices', Lancet Digital Health 2021;3:e195-e203",
        ],
        "glossary": {
            "PMS": (
                "Post-Market Surveillance — the systematic process of proactively collecting "
                "and reviewing real-world performance data after a device is placed on the market."
            ),
            "SaMD": (
                "Software as a Medical Device — software intended to be used for one or more "
                "medical purposes without being part of a hardware medical device."
            ),
            "HITL": (
                "Human-In-The-Loop — an AI deployment pattern where a human clinician reviews, "
                "accepts, or overrides AI outputs before they affect clinical decisions."
            ),
            "SPS": (
                "SaMD Pre-Specifications — a description of planned modifications to an AI/ML "
                "model that the manufacturer anticipates making based on real-world learning."
            ),
            "ACP": (
                "Algorithm Change Protocol — a predetermined plan describing how modifications "
                "to an AI/ML algorithm will be verified, validated, and (if required) "
                "re-submitted to regulators."
            ),
            "Brier Score": (
                "A proper scoring rule for probabilistic predictions. For binary outcomes, "
                "Brier Score = mean squared error of predicted probability vs actual label. "
                "Lower is better; 0 = perfect, 0.25 = uninformative (50/50) classifier."
            ),
            "Override": (
                "A clinical decision where the reviewing clinician substitutes their own "
                "judgment for the AI's recommendation, typically documented with a reason code."
            ),
            "Artifact Risk": (
                "The AI model's internal estimate of the probability that an EEG segment "
                "contains a technical artifact (electrode pop, motion, EMG contamination) "
                "that could degrade classification reliability."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== Overview ===")
    o = overview()
    print(json.dumps(o["kpis"], indent=2))
    print(f"\nReviewers: {len(o['reviewer_workload'])}")
    print("\n=== Breakdown ===")
    b = breakdown()
    print(f"Per-class: {len(b['per_class_performance'])}, "
          f"Per-patient: {len(b['per_patient'])}, "
          f"HITL: {len(b['hitl_reviews'])}")
    print("\n=== Definitions ===")
    d = definitions()
    print(f"Regulatory refs: {len(d['regulatory_context'])}, "
          f"Thresholds: {len(d['vigilance_thresholds'])}, "
          f"Glossary: {len(d['glossary'])}")
