"""Agent Evaluation Dashboard — real clinical.db analytics.

Aggregates expert reviews, HITL reviews, clinical decisions, AI analysis
confidence, consensus, and decision routing into a single evaluation view.
"""

import sqlite3
import json
from pathlib import Path
from collections import Counter

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


# ── overview ────────────────────────────────────────────────────────────
def agent_eval_overview():
    if not DB.exists():
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    c = conn.cursor()

    # ── analyses summary ──
    c.execute("SELECT COUNT(*) FROM analyses")
    total_analyses = c.fetchone()[0]
    c.execute("SELECT AVG(confidence) FROM analyses")
    avg_confidence = c.fetchone()[0]
    c.execute("SELECT disease, COUNT(*) as cnt FROM analyses GROUP BY disease ORDER BY cnt DESC")
    disease_dist = [{"disease": r["disease"], "count": r["cnt"]} for r in c.fetchall()]
    c.execute("SELECT predicted_label, COUNT(*) as cnt FROM analyses GROUP BY predicted_label ORDER BY cnt DESC")
    label_dist = [{"label": r["predicted_label"], "count": r["cnt"]} for r in c.fetchall()]

    # confidence distribution buckets
    c.execute("SELECT confidence FROM analyses")
    confs = [r["confidence"] for r in c.fetchall()]
    buckets = {"<0.5": 0, "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0}
    for cv in confs:
        if cv < 0.5:
            buckets["<0.5"] += 1
        elif cv < 0.6:
            buckets["0.5-0.6"] += 1
        elif cv < 0.7:
            buckets["0.6-0.7"] += 1
        elif cv < 0.8:
            buckets["0.7-0.8"] += 1
        elif cv < 0.9:
            buckets["0.8-0.9"] += 1
        else:
            buckets["0.9-1.0"] += 1
    confidence_dist = [{"bucket": k, "count": v} for k, v in buckets.items()]

    # signal quality distribution
    c.execute("SELECT signal_quality, COUNT(*) as cnt FROM analyses GROUP BY signal_quality ORDER BY cnt DESC")
    quality_dist = [{"quality": r["signal_quality"], "count": r["cnt"]} for r in c.fetchall()]

    # ── expert reviews ──
    c.execute("SELECT COUNT(*) FROM expert_reviews")
    total_expert = c.fetchone()[0]
    c.execute("SELECT agree_with_ai, COUNT(*) as cnt FROM expert_reviews GROUP BY agree_with_ai")
    agree_dist = {r["agree_with_ai"]: r["cnt"] for r in c.fetchall()}
    expert_agree_rate = agree_dist.get("agree", 0) / total_expert if total_expert else None
    c.execute("SELECT role, COUNT(*) as cnt FROM expert_reviews GROUP BY role ORDER BY cnt DESC")
    expert_by_role = [{"role": r["role"], "count": r["cnt"]} for r in c.fetchall()]
    c.execute("SELECT id, patient_id, role, expert, finding, agree_with_ai, note, created_at FROM expert_reviews ORDER BY created_at DESC")
    expert_log = [dict(r) for r in c.fetchall()]

    # ── HITL reviews ──
    c.execute("SELECT COUNT(*) FROM hitl_reviews")
    total_hitl = c.fetchone()[0]
    hitl_rows = []
    c.execute("SELECT id, patient_id, analysis_id, fields_json, created_at FROM hitl_reviews ORDER BY created_at DESC")
    accept_cnt = 0
    override_cnt = 0
    for r in c.fetchall():
        row = dict(r)
        try:
            fields = json.loads(row.get("fields_json", "{}"))
        except Exception:
            fields = {}
        row["decision"] = fields.get("decision", "unknown")
        row["ai_prediction"] = fields.get("ai_prediction", "")
        row["human_decision"] = fields.get("human_decision", "")
        row["reason_code"] = fields.get("reason_code", "")
        if row["decision"] == "accept":
            accept_cnt += 1
        elif row["decision"] == "override":
            override_cnt += 1
        hitl_rows.append(row)
    hitl_accept_rate = accept_cnt / total_hitl if total_hitl else None
    hitl_override_rate = override_cnt / total_hitl if total_hitl else None

    # ── clinical decisions ──
    c.execute("SELECT COUNT(*) FROM clinical_decisions")
    total_decisions = c.fetchone()[0]
    c.execute("SELECT final_decision, COUNT(*) as cnt FROM clinical_decisions GROUP BY final_decision")
    decision_dist = [{"decision": r["final_decision"], "count": r["cnt"]} for r in c.fetchall()]
    c.execute("SELECT neurologist_agreement, COUNT(*) as cnt FROM clinical_decisions GROUP BY neurologist_agreement")
    neuro_agree = {r["neurologist_agreement"]: r["cnt"] for r in c.fetchall()}
    c.execute("SELECT id, patient_id, ai_prediction, ai_confidence, final_decision, reviewer, neurologist_agreement, note, created_at FROM clinical_decisions ORDER BY created_at DESC")
    decision_log = [dict(r) for r in c.fetchall()]

    # ── decision routing simulation ──
    routing_results = []
    c.execute("SELECT confidence FROM analyses")
    for r in c.fetchall():
        conf = r["confidence"]
        if conf >= 0.8:
            routing_results.append("auto")
        elif conf >= 0.5:
            routing_results.append("review")
        else:
            routing_results.append("escalate")
    route_dist = Counter(routing_results)
    routing_summary = [{"route": k, "count": v} for k, v in sorted(route_dist.items())]

    # ── per-patient evaluation depth ──
    c.execute("""
        SELECT a.patient_id,
               COUNT(DISTINCT a.id) as analyses,
               COUNT(DISTINCT er.id) as expert_reviews,
               COUNT(DISTINCT hr.id) as hitl_reviews,
               COUNT(DISTINCT cd.id) as clinical_decisions
        FROM analyses a
        LEFT JOIN expert_reviews er ON er.patient_id = a.patient_id
        LEFT JOIN hitl_reviews hr ON hr.patient_id = a.patient_id
        LEFT JOIN clinical_decisions cd ON cd.patient_id = a.patient_id
        GROUP BY a.patient_id
        ORDER BY analyses DESC
    """)
    per_patient = [dict(r) for r in c.fetchall()]
    patients_with_review = sum(1 for p in per_patient if p["expert_reviews"] > 0 or p["hitl_reviews"] > 0 or p["clinical_decisions"] > 0)
    review_coverage = patients_with_review / len(per_patient) if per_patient else None

    # ── transaction log: agent-related actions ──
    agent_actions = []
    try:
        c.execute("""
            SELECT component, action, COUNT(*) as cnt
            FROM transaction_log
            WHERE component IN ('expert_review','clinical_trust','council','feedback','consultant:neurologist')
            GROUP BY component, action
            ORDER BY cnt DESC
        """)
        agent_actions = [dict(r) for r in c.fetchall()]
    except Exception:
        pass

    # daily trend (transaction_log)
    daily_trend = []
    try:
        c.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as cnt
            FROM transaction_log
            WHERE component IN ('expert_review','clinical_trust','council','feedback','consultant:neurologist')
            GROUP BY day ORDER BY day
        """)
        daily_trend = [dict(r) for r in c.fetchall()]
    except Exception:
        pass

    conn.close()

    return {
        "available": True,
        "summary": {
            "total_analyses": total_analyses,
            "avg_confidence": round(avg_confidence, 4) if avg_confidence else None,
            "total_expert_reviews": total_expert,
            "expert_agree_rate": round(expert_agree_rate, 4) if expert_agree_rate is not None else None,
            "total_hitl_reviews": total_hitl,
            "hitl_accept_rate": round(hitl_accept_rate, 4) if hitl_accept_rate is not None else None,
            "hitl_override_rate": round(hitl_override_rate, 4) if hitl_override_rate is not None else None,
            "total_clinical_decisions": total_decisions,
            "review_coverage": round(review_coverage, 4) if review_coverage is not None else None,
            "patients_with_review": patients_with_review,
            "total_patients_analyzed": len(per_patient),
        },
        "confidence_distribution": confidence_dist,
        "disease_distribution": disease_dist,
        "label_distribution": label_dist,
        "quality_distribution": quality_dist,
        "expert_agreement": {
            "agree": agree_dist.get("agree", 0),
            "disagree": agree_dist.get("disagree", 0),
        },
        "expert_by_role": expert_by_role,
        "decision_distribution": decision_dist,
        "neurologist_agreement": neuro_agree,
        "routing_summary": routing_summary,
        "per_patient": per_patient,
        "agent_actions": agent_actions,
        "daily_trend": daily_trend,
    }


# ── breakdown (logs) ────────────────────────────────────────────────────
def agent_eval_breakdown():
    if not DB.exists():
        return {"available": False}

    conn = _conn()
    c = conn.cursor()

    c.execute("SELECT id, patient_id, role, expert, finding, agree_with_ai, note, created_at FROM expert_reviews ORDER BY created_at DESC")
    expert_log = [dict(r) for r in c.fetchall()]

    hitl_log = []
    c.execute("SELECT id, patient_id, analysis_id, fields_json, created_at FROM hitl_reviews ORDER BY created_at DESC")
    for r in c.fetchall():
        row = dict(r)
        try:
            fields = json.loads(row.pop("fields_json", "{}"))
        except Exception:
            fields = {}
        row.update(fields)
        hitl_log.append(row)

    c.execute("SELECT id, patient_id, ai_prediction, ai_confidence, final_decision, reviewer, neurologist_agreement, note, created_at FROM clinical_decisions ORDER BY created_at DESC")
    decision_log = [dict(r) for r in c.fetchall()]

    # recent agent-related transaction events
    event_log = []
    try:
        c.execute("""
            SELECT id, patient_id, component, action, actor, detail, created_at
            FROM transaction_log
            WHERE component IN ('expert_review','clinical_trust','council','feedback','consultant:neurologist')
            ORDER BY created_at DESC LIMIT 50
        """)
        event_log = [dict(r) for r in c.fetchall()]
    except Exception:
        pass

    conn.close()
    return {
        "available": True,
        "expert_log": expert_log,
        "hitl_log": hitl_log,
        "decision_log": decision_log,
        "event_log": event_log,
    }


# ── definitions ─────────────────────────────────────────────────────────
def agent_eval_definitions():
    return {
        "title": "Agent Evaluation Dashboard — Metric Definitions",
        "metrics": [
            {"name": "Total Analyses", "definition": "Count of all AI analyses run on patient EEG data in clinical.db."},
            {"name": "Avg Confidence", "definition": "Mean confidence score across all AI predictions (0-1 scale)."},
            {"name": "Expert Agree Rate", "definition": "Fraction of expert reviews where the reviewer agreed with the AI prediction."},
            {"name": "HITL Accept Rate", "definition": "Fraction of human-in-the-loop reviews where the human accepted the AI prediction."},
            {"name": "HITL Override Rate", "definition": "Fraction of HITL reviews where the human overrode the AI prediction."},
            {"name": "Review Coverage", "definition": "Fraction of analyzed patients that received at least one expert, HITL, or clinical decision review."},
            {"name": "Confidence Distribution", "definition": "Histogram of AI confidence scores across analyses, bucketed into 0.1-wide ranges."},
            {"name": "Decision Routing", "definition": "How analyses would be routed by confidence: auto (>=0.8), review (0.5-0.8), escalate (<0.5)."},
            {"name": "Expert Agreement", "definition": "Agree vs disagree counts across all expert reviews of AI predictions."},
            {"name": "Neurologist Agreement", "definition": "Whether the neurologist agreed with the AI in clinical-trust sign-off decisions."},
            {"name": "Per-Patient Depth", "definition": "Number of analyses, expert reviews, HITL reviews, and clinical decisions per patient."},
            {"name": "Agent Actions", "definition": "Transaction log entries for agent-related components (expert_review, clinical_trust, council, feedback)."},
        ]
    }
