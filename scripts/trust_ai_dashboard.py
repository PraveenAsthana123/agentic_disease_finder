"""Trust AI Dashboard — AI confidence scoring, concordance (AI-human agreement),
HITL oversight tracking, and clinical decision audit from clinical.db.

Covers:
- Confidence distribution: model self-assessed certainty across analyses
- Concordance: agreement rate between AI predictions and human expert judgment
- HITL oversight: accept/override ratio from human-in-the-loop reviews
- Clinical decision audit: final clinician decisions vs AI recommendations
- Trust score: composite metric across confidence, agreement, oversight, coverage
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql):
    try:
        cur.execute(sql)
        r = cur.fetchone()
        return r[0] if r else 0
    except Exception:
        return 0


def _safe_rows(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


def _parse_fields_json(raw):
    """Safely parse fields_json from hitl_reviews."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


# ── Overview ──────────────────────────────────────────────────

def overview():
    """Trust AI overview: confidence stats, concordance rate, HITL accept/override,
    expert agreement, clinical decision coverage, composite trust score."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc)
    result = {"available": True, "generated_at": now.isoformat()}

    # ── Analyses confidence ──
    total_analyses = _safe(cur,
        "SELECT count(*) FROM analyses WHERE confidence IS NOT NULL")
    mean_confidence = _safe(cur,
        "SELECT avg(confidence) FROM analyses WHERE confidence IS NOT NULL")
    min_confidence = _safe(cur,
        "SELECT min(confidence) FROM analyses WHERE confidence IS NOT NULL")
    max_confidence = _safe(cur,
        "SELECT max(confidence) FROM analyses WHERE confidence IS NOT NULL")

    if mean_confidence is not None:
        mean_confidence = round(float(mean_confidence), 4)
    else:
        mean_confidence = None

    # ── HITL reviews ──
    total_hitl = _safe(cur, "SELECT count(*) FROM hitl_reviews")
    hitl_rows = _safe_rows(cur, "SELECT fields_json FROM hitl_reviews")
    hitl_accept = 0
    hitl_override = 0
    for (fj,) in hitl_rows:
        parsed = _parse_fields_json(fj)
        decision = parsed.get("decision", "").lower()
        if decision == "accept":
            hitl_accept += 1
        elif decision == "override":
            hitl_override += 1
    hitl_accept_rate = round(hitl_accept / max(total_hitl, 1), 4)

    # ── Expert reviews ──
    total_expert = _safe(cur, "SELECT count(*) FROM expert_reviews")
    expert_agree = _safe(cur,
        "SELECT count(*) FROM expert_reviews WHERE agree_with_ai='agree'")
    expert_disagree = _safe(cur,
        "SELECT count(*) FROM expert_reviews WHERE agree_with_ai='disagree'")
    expert_agree_rate = round(expert_agree / max(total_expert, 1), 4)

    # ── Clinical decisions ──
    total_clinical = _safe(cur, "SELECT count(*) FROM clinical_decisions")

    # ── Concordance rate (from expert_reviews) ──
    concordance_rate = expert_agree_rate

    # ── Trust score (composite 0-100) ──
    components = []
    weights = []
    if total_analyses > 0 and mean_confidence is not None:
        components.append(float(mean_confidence) * 100)
        weights.append(25)
    if total_expert > 0:
        components.append(expert_agree_rate * 100)
        weights.append(25)
    if total_hitl > 0:
        components.append(hitl_accept_rate * 100)
        weights.append(25)
    # Clinical decision coverage: fraction of analyses with a clinical decision
    if total_analyses > 0:
        cov = min(total_clinical / max(total_analyses, 1), 1.0) * 100
        components.append(cov)
        weights.append(25)

    if weights:
        trust_score = round(sum(c * w for c, w in zip(components, weights))
                            / sum(weights), 1)
    else:
        trust_score = None

    result["kpis"] = {
        "total_analyses": total_analyses,
        "mean_confidence": mean_confidence,
        "confidence_range": {
            "min": round(float(min_confidence), 4) if min_confidence is not None else None,
            "max": round(float(max_confidence), 4) if max_confidence is not None else None,
        },
        "total_hitl_reviews": total_hitl,
        "hitl_accept_rate": hitl_accept_rate,
        "expert_reviews": total_expert,
        "expert_agree_rate": expert_agree_rate,
        "clinical_decisions": total_clinical,
        "concordance_rate": concordance_rate,
        "override_count": hitl_override,
        "trust_score": trust_score,
    }

    conn.close()
    return result


# ── Breakdown ─────────────────────────────────────────────────

def breakdown():
    """Drill-down: confidence distribution, per-label confidence, expert reviews
    by role, HITL decision log, clinical decision log, concordance by confidence
    band, trust trend."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    result = {"available": True}

    # ── Confidence distribution (buckets) ──
    buckets = [
        ("<0.5", "confidence < 0.5"),
        ("0.5-0.6", "confidence >= 0.5 AND confidence < 0.6"),
        ("0.6-0.7", "confidence >= 0.6 AND confidence < 0.7"),
        ("0.7-0.8", "confidence >= 0.7 AND confidence < 0.8"),
        ("0.8-0.9", "confidence >= 0.8 AND confidence < 0.9"),
        (">=0.9", "confidence >= 0.9"),
    ]
    conf_dist = []
    for label, cond in buckets:
        cnt = _safe(cur,
            f"SELECT count(*) FROM analyses WHERE confidence IS NOT NULL AND {cond}")
        conf_dist.append({"bucket": label, "count": cnt})
    result["confidence_distribution"] = conf_dist

    # ── Confidence by predicted label ──
    label_rows = _safe_rows(cur,
        "SELECT predicted_label, count(*) as cnt, avg(confidence) as avg_conf "
        "FROM analyses WHERE confidence IS NOT NULL AND predicted_label IS NOT NULL "
        "GROUP BY predicted_label ORDER BY avg_conf DESC")
    result["confidence_by_label"] = [
        {"label": r[0], "count": r[1],
         "mean_confidence": round(float(r[2]), 4) if r[2] is not None else None}
        for r in label_rows
    ]

    # ── Expert reviews by role ──
    role_rows = _safe_rows(cur,
        "SELECT role, count(*) as total, "
        "sum(CASE WHEN agree_with_ai='agree' THEN 1 ELSE 0 END) as agree, "
        "sum(CASE WHEN agree_with_ai='disagree' THEN 1 ELSE 0 END) as disagree "
        "FROM expert_reviews GROUP BY role ORDER BY total DESC")
    result["expert_reviews_by_role"] = [
        {"role": r[0], "total": r[1], "agree": r[2], "disagree": r[3],
         "agree_rate": round(r[2] / max(r[1], 1), 4)}
        for r in role_rows
    ]

    # ── HITL decision log ──
    hitl_rows = _safe_rows(cur,
        "SELECT patient_id, fields_json FROM hitl_reviews")
    hitl_decisions = []
    for pid, fj in hitl_rows:
        parsed = _parse_fields_json(fj)
        hitl_decisions.append({
            "patient_id": pid,
            "decision": parsed.get("decision"),
            "ai_prediction": parsed.get("ai_prediction"),
            "human_decision": parsed.get("human_decision"),
            "reason_code": parsed.get("reason_code"),
            "reviewer_id": parsed.get("reviewer_id"),
        })
    result["hitl_decisions"] = hitl_decisions

    # ── Clinical decision log ──
    cd_rows = _safe_rows(cur,
        "SELECT patient_id, ai_prediction, ai_confidence, final_decision, "
        "neurologist_agreement, reviewer FROM clinical_decisions "
        "ORDER BY created_at DESC")
    result["clinical_decision_log"] = [
        {"patient_id": r[0], "ai_prediction": r[1],
         "ai_confidence": r[2], "final_decision": r[3],
         "neurologist_agreement": r[4], "reviewer": r[5]}
        for r in cd_rows
    ]

    # ── Concordance by confidence band ──
    # Join expert_reviews with analyses on analysis_id to get confidence
    band_rows = _safe_rows(cur,
        "SELECT "
        "  CASE "
        "    WHEN a.confidence >= 0.7 THEN 'high' "
        "    WHEN a.confidence >= 0.5 THEN 'mid' "
        "    ELSE 'low' "
        "  END AS band, "
        "  count(*) as total, "
        "  sum(CASE WHEN er.agree_with_ai='agree' THEN 1 ELSE 0 END) as agree, "
        "  sum(CASE WHEN er.agree_with_ai='disagree' THEN 1 ELSE 0 END) as disagree "
        "FROM expert_reviews er "
        "JOIN analyses a ON er.analysis_id = a.id "
        "WHERE a.confidence IS NOT NULL "
        "GROUP BY band ORDER BY band")
    result["concordance_by_confidence_band"] = [
        {"band": r[0], "total": r[1], "agree": r[2], "disagree": r[3],
         "agree_rate": round(r[2] / max(r[1], 1), 4)}
        for r in band_rows
    ]

    # ── Trust trend (daily) ──
    # Try to build daily metrics from expert_reviews + analyses created_at
    trend_rows = _safe_rows(cur,
        "SELECT substr(er.created_at, 1, 10) as day, "
        "  count(*) as reviews, "
        "  sum(CASE WHEN er.agree_with_ai='agree' THEN 1 ELSE 0 END) as agree, "
        "  avg(a.confidence) as avg_conf "
        "FROM expert_reviews er "
        "LEFT JOIN analyses a ON er.analysis_id = a.id "
        "WHERE er.created_at IS NOT NULL "
        "GROUP BY day ORDER BY day")
    if trend_rows:
        result["trust_trend"] = [
            {"date": r[0], "reviews": r[1], "agree": r[2],
             "avg_confidence": round(float(r[3]), 4) if r[3] is not None else None,
             "agree_rate": round(r[2] / max(r[1], 1), 4)}
            for r in trend_rows
        ]
    else:
        result["trust_trend"] = "insufficient data for trend"

    conn.close()
    return result


# ── Definitions ───────────────────────────────────────────────

def definitions():
    """Metric definitions for the Trust AI dashboard."""
    return {
        "available": True,
        "sections": [
            {
                "title": "Trust Score",
                "items": [
                    {"term": "Trust Score", "definition": "Composite metric (0-100) measuring overall AI system trustworthiness. Weighted average of: mean confidence * 100 (25%), expert agreement rate * 100 (25%), HITL accept rate * 100 (25%), clinical decision coverage (25%). If a component has no data, its weight redistributes to components with data."},
                    {"term": "Trust Dimensions", "definition": "Four trust dimensions: Accuracy trust (confidence + prediction quality), Oversight trust (HITL accept/override balance), Transparency trust (explainability, SHAP, confidence calibration), Consistency trust (concordance stability over time)."},
                ],
            },
            {
                "title": "Confidence",
                "items": [
                    {"term": "Confidence Score", "definition": "Model's self-assessed prediction certainty, range 0-1. Derived from the classifier's posterior probability for the predicted class. Higher confidence generally correlates with higher accuracy but must be calibrated."},
                    {"term": "Confidence Distribution", "definition": "Histogram of confidence scores across all analyses, bucketed into 6 ranges: <0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, >=0.9. Healthy distributions cluster in higher buckets."},
                    {"term": "Confidence by Label", "definition": "Mean confidence grouped by predicted label (e.g., epilepsy, depression, normal). Reveals which conditions the model is most/least certain about."},
                ],
            },
            {
                "title": "Concordance",
                "items": [
                    {"term": "Concordance Rate", "definition": "Overall agreement rate between AI predictions and human expert judgment, computed from expert_reviews table. Fraction of reviews where agree_with_ai = 'agree'."},
                    {"term": "Concordance by Confidence Band", "definition": "Agreement rate stratified by the AI's confidence level: high (>=0.7), mid (0.5-0.7), low (<0.5). Tests whether high-confidence predictions align better with expert judgment."},
                    {"term": "Expert Agreement by Role", "definition": "Per-role breakdown of expert concurrence with AI (e.g., neurologist, EEG technician, psychiatrist). Different roles may have different agreement patterns."},
                ],
            },
            {
                "title": "HITL Oversight",
                "items": [
                    {"term": "HITL Review", "definition": "Human-in-the-Loop review where a clinician evaluates the AI's prediction and decides to accept or override it. Stored in hitl_reviews with fields_json containing decision, reason_code, and reviewer_id."},
                    {"term": "Accept Rate", "definition": "Fraction of HITL reviews where the human accepted the AI prediction without modification. High accept rates suggest good AI-human alignment."},
                    {"term": "Override Count", "definition": "Number of HITL reviews where the human overrode the AI's prediction. Overrides are critical safety signals — each should be analyzed for model improvement."},
                    {"term": "Reason Codes", "definition": "Structured codes explaining why a reviewer overrode the AI: clinical_context, artifact_concern, medication_effect, patient_history, borderline_case, etc."},
                ],
            },
            {
                "title": "Clinical Decision Audit",
                "items": [
                    {"term": "Clinical Decision", "definition": "Final clinician decision recorded after reviewing AI prediction, confidence, top EEG channels, artifact risk, and time window. Stored in clinical_decisions table."},
                    {"term": "Neurologist Agreement", "definition": "Whether the neurologist agreed with the AI prediction (agree/disagree/partial). Recorded in clinical_decisions.neurologist_agreement."},
                    {"term": "Final Decision", "definition": "The clinician's final diagnostic or treatment decision, which may differ from the AI prediction. This is the ground truth for clinical outcome tracking."},
                    {"term": "Decision Coverage", "definition": "Fraction of AI analyses that have a corresponding clinical decision record. Low coverage means many AI predictions are not formally reviewed."},
                ],
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {"term": "IEC 62304 Software Lifecycle", "definition": "Medical device software standard requiring traceability of AI decisions, human oversight documentation, and risk management. Trust AI dashboard provides continuous monitoring of AI-human alignment."},
                    {"term": "FDA AI/ML Transparency", "definition": "FDA guidance on artificial intelligence and machine learning in medical devices emphasizes transparency, including confidence reporting, human oversight rates, and decision audit trails."},
                    {"term": "Calibration", "definition": "A well-calibrated model's confidence scores match empirical accuracy: 80% confidence predictions should be correct ~80% of the time. Concordance by confidence band helps assess calibration."},
                ],
            },
        ],
    }
