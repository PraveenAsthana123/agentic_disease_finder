"""AI Governance Dashboard — real governance metrics from clinical.db + config registries.

Sources:
- clinical_decisions (AI→clinician decision flow, agreement/override)
- expert_reviews (specialist panel reviews, agree/disagree)
- hitl_reviews (human-in-the-loop override/accept)
- feedback (clinician ratings of AI output)
- transaction_log (governance events: sign-off, blocked, override)
- config/consultant_matrix.json (role-based oversight registry)
- model_governance table (governance records)
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')
CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')


def _conn():
    return sqlite3.connect(DB)


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def _safe_query(cur, sql):
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


def governance_overview():
    """Aggregate governance posture: decision audit trail, expert review panel,
    HITL oversight, feedback loop, consultant coverage — all from real data."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # 1. Clinical decisions
    decisions = _safe_query(cur, "SELECT * FROM clinical_decisions ORDER BY created_at DESC")
    total_decisions = len(decisions)
    agreed = sum(1 for d in decisions if (d.get('neurologist_agreement') or '').lower() == 'yes')
    overridden = sum(1 for d in decisions if (d.get('final_decision') or '').lower() != 'confirm')
    agreement_rate = round(agreed / total_decisions * 100, 1) if total_decisions else 0

    # 2. Expert reviews
    reviews = _safe_query(cur, "SELECT * FROM expert_reviews ORDER BY created_at DESC")
    total_reviews = len(reviews)
    expert_agree = sum(1 for r in reviews if (r.get('agree_with_ai') or '').lower() == 'agree')
    expert_disagree = sum(1 for r in reviews if (r.get('agree_with_ai') or '').lower() == 'disagree')
    expert_agreement_pct = round(expert_agree / total_reviews * 100, 1) if total_reviews else 0

    # 3. HITL reviews
    hitl = _safe_query(cur, "SELECT * FROM hitl_reviews ORDER BY created_at DESC")
    total_hitl = len(hitl)
    hitl_overrides = 0
    hitl_accepts = 0
    for h in hitl:
        fj = {}
        try:
            fj = json.loads(h.get('fields_json', '{}'))
        except Exception:
            pass
        dec = fj.get('decision', '').lower()
        if dec == 'override':
            hitl_overrides += 1
        elif dec == 'accept':
            hitl_accepts += 1
    override_rate = round(hitl_overrides / total_hitl * 100, 1) if total_hitl else 0

    # 4. Feedback
    feedback = _safe_query(cur, "SELECT * FROM feedback ORDER BY created_at DESC")
    total_feedback = len(feedback)
    avg_rating = 0
    if feedback:
        ratings = [f['rating'] for f in feedback if f.get('rating')]
        avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else 0

    # 5. Governance log entries from transaction_log
    gov_events = _safe_count(cur,
        "SELECT count(*) FROM transaction_log WHERE action IN ('sign-off','blocked','human_decision','rating','submit')")

    # 6. Consultant matrix
    consultant_count = 0
    mandatory_count = 0
    try:
        with open(os.path.join(CONFIG, 'consultant_matrix.json')) as f:
            cm = json.load(f)
        consultants = cm.get('consultants', [])
        consultant_count = len(consultants)
        mandatory_count = sum(1 for c in consultants if c.get('mandatory'))
    except Exception:
        pass

    # 7. Model governance records
    gov_records = _safe_count(cur, "SELECT count(*) FROM model_governance")

    conn.close()

    summary = {
        "total_decisions": total_decisions,
        "agreement_rate": agreement_rate,
        "expert_reviews": total_reviews,
        "expert_agreement_pct": expert_agreement_pct,
        "hitl_reviews": total_hitl,
        "hitl_overrides": hitl_overrides,
        "override_rate": override_rate,
        "avg_feedback_rating": avg_rating,
        "governance_events": gov_events,
        "consultant_roles": consultant_count,
        "mandatory_consultants": mandatory_count,
        "model_governance_records": gov_records,
    }

    # Decision audit trail (last 10)
    decision_trail = []
    for d in decisions[:10]:
        decision_trail.append({
            "id": d.get("id"),
            "patient_id": d.get("patient_id"),
            "ai_prediction": d.get("ai_prediction"),
            "confidence": d.get("ai_confidence"),
            "agreement": d.get("neurologist_agreement"),
            "final_decision": d.get("final_decision"),
            "reviewer": d.get("reviewer"),
            "created_at": d.get("created_at"),
        })

    # Expert review panel
    review_panel = []
    for r in reviews[:10]:
        review_panel.append({
            "id": r.get("id"),
            "patient_id": r.get("patient_id"),
            "role": r.get("role"),
            "expert": r.get("expert"),
            "finding": r.get("finding"),
            "agree_with_ai": r.get("agree_with_ai"),
            "created_at": r.get("created_at"),
        })

    # HITL detail
    hitl_detail = []
    for h in hitl[:10]:
        fj = {}
        try:
            fj = json.loads(h.get('fields_json', '{}'))
        except Exception:
            pass
        hitl_detail.append({
            "id": h.get("id"),
            "patient_id": h.get("patient_id"),
            "ai_prediction": fj.get("ai_prediction"),
            "decision": fj.get("decision"),
            "human_decision": fj.get("human_decision"),
            "reason_code": fj.get("reason_code"),
            "created_at": h.get("created_at"),
        })

    # Governance event breakdown
    conn2 = _conn()
    cur2 = conn2.cursor()
    gov_event_breakdown = _safe_query(cur2,
        "SELECT action, count(*) as cnt FROM transaction_log "
        "WHERE action IN ('sign-off','blocked','human_decision','rating','submit') "
        "GROUP BY action ORDER BY cnt DESC")

    # Daily governance events
    daily_events = _safe_query(cur2,
        "SELECT date(created_at) as day, count(*) as cnt FROM transaction_log "
        "WHERE action IN ('sign-off','blocked','human_decision','rating','submit') "
        "GROUP BY day ORDER BY day")
    conn2.close()

    return {
        "available": True,
        "summary": summary,
        "decision_trail": decision_trail,
        "review_panel": review_panel,
        "hitl_detail": hitl_detail,
        "governance_event_breakdown": gov_event_breakdown,
        "daily_governance_events": daily_events,
    }


def governance_breakdown():
    """Detailed breakdowns: consultant matrix, per-role coverage,
    governance health scores, use-case risk classification."""
    result = {}

    # Consultant matrix
    try:
        with open(os.path.join(CONFIG, 'consultant_matrix.json')) as f:
            cm = json.load(f)
        consultants = cm.get('consultants', [])
        result["consultant_matrix"] = []
        for c in consultants:
            result["consultant_matrix"].append({
                "id": c.get("id"),
                "name": c.get("name"),
                "tier": c.get("tier"),
                "mandatory": c.get("mandatory", False),
                "role": c.get("role"),
                "objective": c.get("objective"),
                "task_count": len(c.get("tasks", [])),
                "compliance_doc_count": len(c.get("compliance_docs", [])),
                "challenge_count": len(c.get("challenges", [])),
            })
        result["engagement_model"] = cm.get("engagement_model", "")
        result["matrix_updated"] = cm.get("updated_at", "")
    except Exception:
        result["consultant_matrix"] = []

    # Governance health scores (derived from real data)
    conn = _conn()
    cur = conn.cursor()

    total_decisions = _safe_count(cur, "SELECT count(*) FROM clinical_decisions")
    agreed_decisions = _safe_count(cur,
        "SELECT count(*) FROM clinical_decisions WHERE neurologist_agreement = 'Yes'")
    total_reviews = _safe_count(cur, "SELECT count(*) FROM expert_reviews")
    expert_agrees = _safe_count(cur,
        "SELECT count(*) FROM expert_reviews WHERE agree_with_ai = 'agree'")
    total_hitl = _safe_count(cur, "SELECT count(*) FROM hitl_reviews")
    total_feedback = _safe_count(cur, "SELECT count(*) FROM feedback")

    # Per-role expert breakdown
    role_breakdown = _safe_query(cur,
        "SELECT role, count(*) as cnt, "
        "sum(CASE WHEN agree_with_ai='agree' THEN 1 ELSE 0 END) as agreed "
        "FROM expert_reviews GROUP BY role")

    # Decision confidence distribution
    confidence_dist = _safe_query(cur,
        "SELECT CASE "
        "WHEN ai_confidence < 0.5 THEN 'Low (<0.5)' "
        "WHEN ai_confidence < 0.7 THEN 'Medium (0.5-0.7)' "
        "WHEN ai_confidence < 0.9 THEN 'High (0.7-0.9)' "
        "ELSE 'Very High (>=0.9)' END as band, count(*) as cnt "
        "FROM clinical_decisions GROUP BY band ORDER BY band")

    # Governance transaction timeline
    gov_timeline = _safe_query(cur,
        "SELECT date(created_at) as day, action, count(*) as cnt "
        "FROM transaction_log "
        "WHERE action IN ('sign-off','blocked','human_decision','rating','submit') "
        "GROUP BY day, action ORDER BY day")

    # Use-case risk classification from consultant matrix
    use_case_register = []
    for c in result.get("consultant_matrix", []):
        risk_level = "high" if c.get("mandatory") else ("medium" if c.get("tier", 99) <= 2 else "low")
        use_case_register.append({
            "role": c.get("name"),
            "risk_class": risk_level,
            "tier": c.get("tier"),
            "mandatory": c.get("mandatory"),
            "tasks": c.get("task_count", 0),
            "compliance_docs": c.get("compliance_doc_count", 0),
        })

    conn.close()

    health_scores = {
        "clinical_agreement": round(agreed_decisions / total_decisions * 100, 1) if total_decisions else 0,
        "expert_consensus": round(expert_agrees / total_reviews * 100, 1) if total_reviews else 0,
        "hitl_coverage": total_hitl,
        "feedback_coverage": total_feedback,
        "consultant_coverage": len(result.get("consultant_matrix", [])),
        "decision_audit_completeness": round(total_decisions / max(total_decisions, 1) * 100, 1),
    }

    result["health_scores"] = health_scores
    result["role_breakdown"] = role_breakdown
    result["confidence_distribution"] = confidence_dist
    result["governance_timeline"] = gov_timeline
    result["use_case_register"] = use_case_register

    return result


def governance_definitions():
    """AI Governance concepts, metrics, clinical relevance, remediation."""
    return {
        "sections": [
            {
                "title": "AI Governance Concepts",
                "items": [
                    {"term": "Clinical Decision Audit", "definition": "Complete trail of AI prediction → clinician review → final decision, ensuring every AI output is reviewed before clinical action."},
                    {"term": "Expert Review Panel", "definition": "Multi-disciplinary specialist review (Neurologist, EEG Technician, etc.) providing independent validation of AI outputs."},
                    {"term": "Human-in-the-Loop (HITL)", "definition": "Mandatory human oversight where clinicians can accept or override AI predictions with documented rationale."},
                    {"term": "Consultant Matrix", "definition": "Role-based oversight registry mapping each clinical role to tasks, compliance docs, and mandatory review requirements."},
                    {"term": "Use-Case Risk Classification", "definition": "Risk stratification of AI use cases (high/medium/low) based on clinical impact and regulatory requirements."},
                    {"term": "Override Rate", "definition": "Percentage of AI predictions overridden by human reviewers — high rates indicate model calibration issues."},
                    {"term": "Governance Event Log", "definition": "Audit trail of sign-offs, blocked actions, overrides, and feedback submitted through the governance pipeline."},
                ]
            },
            {
                "title": "Governance Metrics",
                "items": [
                    {"term": "Agreement Rate", "definition": "Percentage of AI predictions confirmed by neurologist review (target: >80%)."},
                    {"term": "Expert Consensus", "definition": "Proportion of expert reviews agreeing with AI output across all specialist roles."},
                    {"term": "Override Rate", "definition": "Fraction of HITL reviews resulting in override rather than accept (lower is better for model confidence)."},
                    {"term": "Feedback Rating", "definition": "Average clinician rating of AI output quality (1-5 scale, target: >=4.0)."},
                    {"term": "Consultant Coverage", "definition": "Number of mandatory consultant roles actively engaged in governance oversight."},
                    {"term": "Decision Audit Completeness", "definition": "Percentage of AI predictions with full audit trail (prediction → review → decision)."},
                ]
            },
            {
                "title": "Clinical & Regulatory Relevance",
                "items": [
                    {"term": "EU AI Act (Article 14)", "definition": "High-risk AI systems must have human oversight measures proportionate to risk, including ability to override AI decisions."},
                    {"term": "FDA AI/ML SaMD", "definition": "Good Machine Learning Practice requires predetermined change control plans and clinical validation with human oversight."},
                    {"term": "IEC 62304", "definition": "Medical device software lifecycle standard requiring documented decision processes and traceability."},
                    {"term": "ISO 14971", "definition": "Risk management for medical devices — requires use-case risk classification and mitigation documentation."},
                    {"term": "ILAE Guidelines", "definition": "International League Against Epilepsy standards for seizure classification requiring expert neurologist validation."},
                    {"term": "HIPAA", "definition": "Patient data governance in clinical AI systems — audit trails must protect PHI while maintaining decision transparency."},
                ]
            },
            {
                "title": "Remediation Strategies",
                "items": [
                    {"term": "Low Agreement Rate", "definition": "Retrain model with updated ground-truth labels, increase calibration, add uncertainty quantification."},
                    {"term": "High Override Rate", "definition": "Investigate systematic prediction errors, recalibrate confidence thresholds, expand training data for failure modes."},
                    {"term": "Low Feedback Rating", "definition": "Review AI output format/presentation, add contextual explanations (XAI), reduce false positive rate."},
                    {"term": "Missing Consultant Coverage", "definition": "Engage missing mandatory roles per consultant matrix, document gaps in compliance report."},
                    {"term": "Incomplete Audit Trail", "definition": "Enforce governance pipeline — block clinical action until decision + review + sign-off recorded."},
                ]
            }
        ]
    }
