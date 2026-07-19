"""Regulatory Submissions Dashboard — FDA/CE pathway tracking from clinical.db.

Tracks regulatory submission lifecycle: pathways (FDA 510(k), De Novo, PMA, CE Mark),
product classifications (SaMD Class I/II/III), submission statuses, reviewer workload,
risk classes, validation scores, and phase progression.

Sources:
- regulatory_submissions table (16 rows, 8 products, 5 pathways, 5 statuses,
  5 reviewers, 5 phases, 4 risk classes)
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
#  /api/regulatory-submissions/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Aggregate submission health: total, pathways, statuses, products,
    risk distribution, validation scores, timeline."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total = _safe(cur, "SELECT COUNT(*) FROM regulatory_submissions")
    if total == 0:
        conn.close()
        return {"available": False, "note": "No regulatory submissions data"}

    total_products = _safe(cur, "SELECT COUNT(DISTINCT product_name) FROM regulatory_submissions")
    total_pathways = _safe(cur, "SELECT COUNT(DISTINCT pathway) FROM regulatory_submissions")
    total_reviewers = _safe(cur, "SELECT COUNT(DISTINCT reviewer) FROM regulatory_submissions")
    approved_count = _safe(cur, "SELECT COUNT(*) FROM regulatory_submissions WHERE status='Approved'")
    avg_validation = _safe(cur, "SELECT AVG(validation_score) FROM regulatory_submissions WHERE validation_score IS NOT NULL", default=None)

    # Status distribution
    status_rows = _safe_rows(cur, """
        SELECT status, COUNT(*) as cnt
        FROM regulatory_submissions
        GROUP BY status ORDER BY cnt DESC
    """)
    status_distribution = [{"status": r[0], "count": r[1]} for r in status_rows]

    # Pathway distribution
    pathway_rows = _safe_rows(cur, """
        SELECT pathway, COUNT(*) as cnt
        FROM regulatory_submissions
        GROUP BY pathway ORDER BY cnt DESC
    """)
    pathway_distribution = [{"pathway": r[0], "count": r[1]} for r in pathway_rows]

    # Risk class distribution
    risk_rows = _safe_rows(cur, """
        SELECT risk_class, COUNT(*) as cnt
        FROM regulatory_submissions
        GROUP BY risk_class ORDER BY cnt DESC
    """)
    risk_distribution = [{"risk_class": r[0], "count": r[1]} for r in risk_rows]

    # Product breakdown
    product_rows = _safe_rows(cur, """
        SELECT product_name, COUNT(*) as cnt
        FROM regulatory_submissions
        GROUP BY product_name ORDER BY cnt DESC
    """)
    product_breakdown = [{"product": r[0], "count": r[1]} for r in product_rows]

    # Phase distribution
    phase_rows = _safe_rows(cur, """
        SELECT phase, COUNT(*) as cnt
        FROM regulatory_submissions
        GROUP BY phase ORDER BY cnt DESC
    """)
    phase_distribution = [{"phase": r[0], "count": r[1]} for r in phase_rows]

    # Monthly submission timeline
    monthly_rows = _safe_rows(cur, """
        SELECT strftime('%Y-%m', submitted_date) as month, COUNT(*) as cnt
        FROM regulatory_submissions
        WHERE submitted_date IS NOT NULL
        GROUP BY month ORDER BY month
    """)
    monthly_timeline = [{"month": r[0], "submissions": r[1]} for r in monthly_rows]

    conn.close()
    return {
        "available": True,
        "kpis": {
            "total_submissions": total,
            "total_products": total_products,
            "total_pathways": total_pathways,
            "total_reviewers": total_reviewers,
            "approved_count": approved_count,
            "approval_rate": round(approved_count / total * 100, 1) if total else 0,
            "avg_validation_score": round(avg_validation, 3) if avg_validation else None
        },
        "status_distribution": status_distribution,
        "pathway_distribution": pathway_distribution,
        "risk_distribution": risk_distribution,
        "product_breakdown": product_breakdown,
        "phase_distribution": phase_distribution,
        "monthly_timeline": monthly_timeline
    }


# ──────────────────────────────────────────────────────────────
#  /api/regulatory-submissions/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Detailed breakdown: per-product submissions, reviewer workload,
    overdue submissions, validation scores by product, recent submissions."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total = _safe(cur, "SELECT COUNT(*) FROM regulatory_submissions")
    if total == 0:
        conn.close()
        return {"available": False, "note": "No regulatory submissions data"}

    # Reviewer workload
    reviewer_rows = _safe_rows(cur, """
        SELECT reviewer, COUNT(*) as cnt,
               SUM(CASE WHEN status='Approved' THEN 1 ELSE 0 END) as approved,
               AVG(validation_score) as avg_score
        FROM regulatory_submissions
        GROUP BY reviewer ORDER BY cnt DESC
    """)
    reviewer_workload = [{
        "reviewer": r[0],
        "total": r[1],
        "approved": r[2],
        "avg_validation_score": round(r[3], 3) if r[3] else None
    } for r in reviewer_rows]

    # Per-product submission details
    product_rows = _safe_rows(cur, """
        SELECT product_name,
               GROUP_CONCAT(DISTINCT pathway) as pathways,
               GROUP_CONCAT(DISTINCT status) as statuses,
               COUNT(*) as cnt,
               AVG(validation_score) as avg_score
        FROM regulatory_submissions
        GROUP BY product_name ORDER BY cnt DESC
    """)
    per_product = [{
        "product": r[0],
        "pathways": r[1],
        "statuses": r[2],
        "submissions": r[3],
        "avg_validation_score": round(r[4], 3) if r[4] else None
    } for r in product_rows]

    # Overdue / at-risk submissions (target_date in the past, not approved)
    overdue_rows = _safe_rows(cur, """
        SELECT submission_id, product_name, pathway, status, target_date, reviewer
        FROM regulatory_submissions
        WHERE target_date < date('now') AND status != 'Approved'
        ORDER BY target_date
    """)
    overdue = [{
        "submission_id": r[0], "product": r[1], "pathway": r[2],
        "status": r[3], "target_date": r[4], "reviewer": r[5]
    } for r in overdue_rows]

    # Validation scores by product
    score_rows = _safe_rows(cur, """
        SELECT product_name, validation_score, pathway, status
        FROM regulatory_submissions
        WHERE validation_score IS NOT NULL
        ORDER BY validation_score DESC
    """)
    validation_scores = [{
        "product": r[0], "score": r[1], "pathway": r[2], "status": r[3]
    } for r in score_rows]

    # Pathway-status cross-tab
    cross_rows = _safe_rows(cur, """
        SELECT pathway, status, COUNT(*) as cnt
        FROM regulatory_submissions
        GROUP BY pathway, status ORDER BY pathway, status
    """)
    pathway_status = [{
        "pathway": r[0], "status": r[1], "count": r[2]
    } for r in cross_rows]

    # Recent submissions
    recent_rows = _safe_rows(cur, """
        SELECT submission_id, product_name, pathway, classification, status,
               submitted_date, target_date, reviewer, phase, risk_class, validation_score
        FROM regulatory_submissions
        ORDER BY submitted_date DESC LIMIT 16
    """)
    recent = [{
        "submission_id": r[0], "product": r[1], "pathway": r[2],
        "classification": r[3], "status": r[4], "submitted_date": r[5],
        "target_date": r[6], "reviewer": r[7], "phase": r[8],
        "risk_class": r[9], "validation_score": r[10]
    } for r in recent_rows]

    conn.close()
    return {
        "available": True,
        "reviewer_workload": reviewer_workload,
        "per_product": per_product,
        "overdue_submissions": overdue,
        "validation_scores": validation_scores,
        "pathway_status_crosstab": pathway_status,
        "recent_submissions": recent
    }


# ──────────────────────────────────────────────────────────────
#  /api/regulatory-submissions/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Reference definitions: pathways, statuses, risk classes, phases, glossary."""
    return {
        "available": True,
        "pathways": [
            {"id": "fda_denovo", "name": "FDA De Novo", "description": "Novel device classification for low-to-moderate risk devices without a predicate"},
            {"id": "fda_510k", "name": "FDA 510(k)", "description": "Premarket notification demonstrating substantial equivalence to a predicate device"},
            {"id": "fda_pma", "name": "FDA PMA", "description": "Premarket Approval for Class III high-risk devices requiring clinical evidence"},
            {"id": "ce_mdr", "name": "CE Mark (MDR)", "description": "EU Medical Device Regulation conformity assessment for medical devices"},
            {"id": "ce_ivdr", "name": "CE Mark (IVDR)", "description": "EU In Vitro Diagnostic Regulation conformity assessment for IVD software"}
        ],
        "statuses": [
            {"id": "pre_submission", "name": "Pre-submission", "description": "Preparing submission package; pre-sub meeting with regulatory body"},
            {"id": "submitted", "name": "Submitted", "description": "Submission filed and under initial administrative review"},
            {"id": "under_review", "name": "Under Review", "description": "Active substantive review by regulatory body"},
            {"id": "additional_info", "name": "Additional Info Requested", "description": "Regulatory body requested supplemental data or clarification"},
            {"id": "approved", "name": "Approved", "description": "Regulatory clearance/approval granted; device may be marketed"}
        ],
        "risk_classes": [
            {"id": "class_i", "name": "Class I", "description": "Low risk — general controls sufficient (EU: non-invasive, simple)"},
            {"id": "class_iia", "name": "Class IIa", "description": "Low-to-moderate risk — special controls (EU: short-term invasive)"},
            {"id": "class_iib", "name": "Class IIb", "description": "Moderate-to-high risk — additional performance data (EU: active implants)"},
            {"id": "class_iii", "name": "Class III", "description": "High risk — full PMA clinical evidence required (EU: long-term implants)"}
        ],
        "phases": [
            {"id": "design_controls", "name": "Design Controls", "description": "Design input/output, V&V planning, DHF documentation"},
            {"id": "verification", "name": "Verification", "description": "Technical verification — software testing, bench testing, biocompat"},
            {"id": "validation", "name": "Validation", "description": "Clinical validation — usability, simulated use, clinical trials"},
            {"id": "clinical_evaluation", "name": "Clinical Evaluation", "description": "Clinical evidence compilation per MEDDEV 2.7/1 or FDA guidance"},
            {"id": "post_market", "name": "Post-Market", "description": "Post-market surveillance, PMCF/PMS plan execution, vigilance"}
        ],
        "classifications": [
            {"id": "samd_i", "name": "SaMD Class I", "description": "Software as Medical Device — informs clinical management, non-serious condition"},
            {"id": "samd_ii", "name": "SaMD Class II", "description": "Software as Medical Device — drives clinical management, serious condition"},
            {"id": "samd_iii", "name": "SaMD Class III", "description": "Software as Medical Device — treats/diagnoses, critical condition"}
        ],
        "field_descriptions": [
            {"field": "submission_id", "description": "Unique regulatory submission tracking identifier (REG-YYYY-NNNN)"},
            {"field": "pathway", "description": "Regulatory pathway selected (FDA 510(k), De Novo, PMA, CE MDR/IVDR)"},
            {"field": "validation_score", "description": "Algorithm validation performance score (0-1 scale, higher=better)"},
            {"field": "target_date", "description": "Expected regulatory decision/clearance target date"},
            {"field": "risk_class", "description": "EU MDR/IVDR risk classification (Class I, IIa, IIb, III)"}
        ],
        "clinical_notes": [
            "SaMD classification follows IMDRF framework (state of healthcare situation × significance of information)",
            "FDA 510(k) requires predicate device comparison; De Novo is for truly novel low/moderate-risk devices",
            "CE Mark MDR applies from May 2021; IVDR from May 2022 — both replace previous directives",
            "Validation scores reflect clinical performance metrics (sensitivity, specificity, AUC) from V&V studies",
            "Post-market surveillance is mandatory for all classes; PMCF studies required for Class IIb/III"
        ],
        "glossary": [
            {"term": "SaMD", "definition": "Software as a Medical Device — software intended to be used for medical purposes without being part of a hardware device"},
            {"term": "510(k)", "definition": "FDA premarket notification pathway demonstrating substantial equivalence to a legally marketed predicate"},
            {"term": "De Novo", "definition": "FDA pathway for novel, low-to-moderate risk devices with no predicate; establishes a new classification"},
            {"term": "PMA", "definition": "Premarket Approval — most rigorous FDA pathway for Class III devices requiring clinical trials"},
            {"term": "MDR", "definition": "Medical Device Regulation (EU 2017/745) — replaced MDD, stricter classification and surveillance"},
            {"term": "IVDR", "definition": "In Vitro Diagnostic Regulation (EU 2017/746) — new risk-based classification for IVD software"},
            {"term": "DHF", "definition": "Design History File — complete record of design controls per 21 CFR 820.30"},
            {"term": "PMCF", "definition": "Post-Market Clinical Follow-up — ongoing clinical data collection post-CE marking"},
            {"term": "V&V", "definition": "Verification & Validation — systematic confirmation that design outputs meet inputs (V) and user needs (V)"},
            {"term": "Notified Body", "definition": "EU-designated organization that performs conformity assessment for CE marking (e.g., BSI, TÜV)"}
        ]
    }
