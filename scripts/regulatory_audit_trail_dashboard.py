"""Regulatory Audit Trail Dashboard — compliance audit analytics from clinical.db.

Tracks regulatory submissions, audit actions, actor activity, category distribution,
document references, and timeline analysis for FDA/CE compliance workflows.

Sources:
- regulatory_audit_trail table (submission_id, action, actor, timestamp, details,
  document_ref, category)
- 102 records, 16 submissions, 11 actors, 9 action types, 5 categories
"""

import sqlite3
import os
from collections import Counter

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
#  /api/regulatory-audit-trail/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Aggregate audit trail health: total actions, submissions, actors,
    category distribution, action type breakdown, monthly timeline."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total_actions = _safe(cur, "SELECT COUNT(*) FROM regulatory_audit_trail")
    if total_actions == 0:
        conn.close()
        return {"available": False, "note": "No regulatory audit trail data"}

    total_submissions = _safe(cur, "SELECT COUNT(DISTINCT submission_id) FROM regulatory_audit_trail")
    total_actors = _safe(cur, "SELECT COUNT(DISTINCT actor) FROM regulatory_audit_trail")
    total_categories = _safe(cur, "SELECT COUNT(DISTINCT category) FROM regulatory_audit_trail")
    total_documents = _safe(cur, "SELECT COUNT(DISTINCT document_ref) FROM regulatory_audit_trail")

    # Category distribution
    category_rows = _safe_rows(cur, """
        SELECT category, COUNT(*) as cnt
        FROM regulatory_audit_trail
        GROUP BY category ORDER BY cnt DESC
    """)
    category_distribution = [{"category": r[0], "count": r[1]} for r in category_rows]

    # Action type breakdown
    action_rows = _safe_rows(cur, """
        SELECT action, COUNT(*) as cnt
        FROM regulatory_audit_trail
        GROUP BY action ORDER BY cnt DESC
    """)
    action_breakdown = [{"action": r[0], "count": r[1]} for r in action_rows]

    # Actor activity
    actor_rows = _safe_rows(cur, """
        SELECT actor, COUNT(*) as cnt
        FROM regulatory_audit_trail
        GROUP BY actor ORDER BY cnt DESC
    """)
    actor_activity = [{"actor": r[0], "count": r[1]} for r in actor_rows]

    # Monthly timeline
    monthly_rows = _safe_rows(cur, """
        SELECT substr(timestamp, 1, 7) as month, COUNT(*) as cnt
        FROM regulatory_audit_trail
        GROUP BY month ORDER BY month
    """)
    monthly_timeline = [{"month": r[0], "count": r[1]} for r in monthly_rows]

    # Most active submission
    top_submission = _safe_rows(cur, """
        SELECT submission_id, COUNT(*) as cnt
        FROM regulatory_audit_trail
        GROUP BY submission_id ORDER BY cnt DESC LIMIT 1
    """)

    conn.close()

    return {
        "available": True,
        "kpis": {
            "total_actions": total_actions,
            "total_submissions": total_submissions,
            "total_actors": total_actors,
            "total_categories": total_categories,
            "total_documents": total_documents,
            "most_active_submission": top_submission[0][0] if top_submission else "--",
            "most_active_submission_count": top_submission[0][1] if top_submission else 0,
        },
        "category_distribution": category_distribution,
        "action_breakdown": action_breakdown,
        "actor_activity": actor_activity,
        "monthly_timeline": monthly_timeline,
    }


# ──────────────────────────────────────────────────────────────
#  /api/regulatory-audit-trail/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-submission breakdown, recent actions, CAPA/deviation alerts."""
    if not os.path.exists(DB):
        return {"available": False}

    conn = _conn()
    cur = conn.cursor()

    # Per-submission summary
    sub_rows = _safe_rows(cur, """
        SELECT submission_id, COUNT(*) as action_count,
               COUNT(DISTINCT actor) as actor_count,
               COUNT(DISTINCT category) as cat_count,
               MIN(timestamp) as first_action,
               MAX(timestamp) as last_action
        FROM regulatory_audit_trail
        GROUP BY submission_id ORDER BY submission_id
    """)
    per_submission = [{
        "submission_id": r[0],
        "action_count": r[1],
        "actor_count": r[2],
        "category_count": r[3],
        "first_action": r[4],
        "last_action": r[5],
    } for r in sub_rows]

    # Recent actions (last 20)
    recent_rows = _safe_rows(cur, """
        SELECT submission_id, action, actor, timestamp, details, document_ref, category
        FROM regulatory_audit_trail
        ORDER BY timestamp DESC LIMIT 20
    """)
    recent_actions = [{
        "submission_id": r[0], "action": r[1], "actor": r[2],
        "timestamp": r[3], "details": r[4], "document_ref": r[5], "category": r[6]
    } for r in recent_rows]

    # CAPA and deviation alerts
    alert_rows = _safe_rows(cur, """
        SELECT submission_id, action, actor, timestamp, document_ref, category
        FROM regulatory_audit_trail
        WHERE action IN ('CAPA opened', 'Deviation logged')
        ORDER BY timestamp DESC
    """)
    alerts = [{
        "submission_id": r[0], "action": r[1], "actor": r[2],
        "timestamp": r[3], "document_ref": r[4], "category": r[5]
    } for r in alert_rows]

    # Per-actor summary
    actor_summary_rows = _safe_rows(cur, """
        SELECT actor, COUNT(*) as action_count,
               COUNT(DISTINCT submission_id) as submission_count,
               COUNT(DISTINCT action) as action_types,
               MAX(timestamp) as last_activity
        FROM regulatory_audit_trail
        GROUP BY actor ORDER BY action_count DESC
    """)
    per_actor = [{
        "actor": r[0], "action_count": r[1], "submission_count": r[2],
        "action_types": r[3], "last_activity": r[4]
    } for r in actor_summary_rows]

    conn.close()

    return {
        "available": True,
        "per_submission": per_submission,
        "recent_actions": recent_actions,
        "alerts": alerts,
        "per_actor": per_actor,
    }


# ──────────────────────────────────────────────────────────────
#  /api/regulatory-audit-trail/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Regulatory audit trail definitions — action types, categories, glossary."""
    return {
        "available": True,
        "action_types": [
            {"action": "Risk assessment updated", "description": "Revision to the product/process risk assessment documentation"},
            {"action": "Document uploaded", "description": "New regulatory document added to the submission file"},
            {"action": "Design review completed", "description": "Formal design review milestone completed"},
            {"action": "Deviation logged", "description": "Non-conformance or protocol deviation recorded"},
            {"action": "Comment added", "description": "Reviewer comment or annotation on submission"},
            {"action": "Signature obtained", "description": "Required approval signature captured electronically"},
            {"action": "Review initiated", "description": "Formal review cycle started for a submission"},
            {"action": "CAPA opened", "description": "Corrective and Preventive Action opened for a finding"},
            {"action": "Status changed", "description": "Submission status transitioned to a new state"},
        ],
        "categories": [
            {"category": "Clinical", "description": "Actions related to clinical evidence, trials, or patient safety data"},
            {"category": "Quality", "description": "Quality management system actions — audits, CAPA, deviations"},
            {"category": "Administrative", "description": "Administrative actions — scheduling, assignments, logistics"},
            {"category": "Regulatory", "description": "Direct regulatory authority interactions and compliance actions"},
            {"category": "Technical", "description": "Technical documentation — software validation, design specs"},
        ],
        "glossary": [
            {"term": "CAPA", "definition": "Corrective and Preventive Action — systematic approach to identifying root causes and preventing recurrence"},
            {"term": "Deviation", "definition": "Departure from an approved procedure, specification, or established standard"},
            {"term": "Submission", "definition": "A regulatory filing (e.g., 510(k), PMA, CE Technical File) sent to authorities"},
            {"term": "Audit Trail", "definition": "Chronological record showing who did what, when, and why — required by 21 CFR Part 11"},
            {"term": "Design Review", "definition": "Systematic examination of a design to evaluate its adequacy and identify problems"},
            {"term": "Risk Assessment", "definition": "Analysis of potential hazards and their severity/probability per ISO 14971"},
            {"term": "21 CFR Part 11", "definition": "FDA regulation governing electronic records and electronic signatures"},
            {"term": "ISO 13485", "definition": "Quality management system standard for medical device manufacturers"},
            {"term": "Document Control", "definition": "Process ensuring documents are reviewed, approved, distributed, and maintained"},
            {"term": "Electronic Signature", "definition": "Legally binding electronic equivalent of a handwritten signature per 21 CFR 11"},
        ],
        "clinical_notes": [
            "All audit trail entries are immutable and timestamped per 21 CFR Part 11 requirements",
            "CAPA findings must be resolved before submission advancement",
            "Deviation logs trigger automatic review escalation",
            "Document uploads require version control and approval workflows",
            "Signature actions are electronically verified and non-repudiable",
        ],
    }
