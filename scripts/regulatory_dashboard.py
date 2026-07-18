"""Clinical Validation & Regulatory Dashboard — FDA/CE/MDR pathway tracking,
validation study progress, regulatory submission status, and audit trail
from clinical.db regulatory_submissions, validation_studies, regulatory_audit_trail tables.

Covers:
- Submission pipeline: pathway distribution, status breakdown, risk classification
- Validation studies: performance metrics (sensitivity/specificity/AUC), study status
- Audit trail: document activity, reviewer actions, compliance tracking
- Regulatory KPIs: approval rate, mean review time, validation pass rate
"""

import os
import sqlite3
from collections import defaultdict
from datetime import datetime

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql):
    """Execute SQL safely, return [] on failure."""
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


def _cols(cur):
    return [d[0] for d in cur.description] if cur.description else []


def overview():
    """Regulatory overview — submission pipeline KPIs, pathway distribution,
    status breakdown, risk class distribution, validation summary."""
    conn = _conn()
    cur = conn.cursor()

    # Submission KPIs
    total = _safe(cur, "SELECT COUNT(*) FROM regulatory_submissions")[0][0]
    approved = _safe(cur, "SELECT COUNT(*) FROM regulatory_submissions WHERE status IN ('Approved','Conditionally Approved')")[0][0]
    under_review = _safe(cur, "SELECT COUNT(*) FROM regulatory_submissions WHERE status='Under Review'")[0][0]
    submitted = _safe(cur, "SELECT COUNT(*) FROM regulatory_submissions WHERE status='Submitted'")[0][0]

    # Unique products
    products = _safe(cur, "SELECT COUNT(DISTINCT product_name) FROM regulatory_submissions")[0][0]

    # Validation study KPIs
    total_studies = _safe(cur, "SELECT COUNT(*) FROM validation_studies")[0][0]
    passed_studies = _safe(cur, "SELECT COUNT(*) FROM validation_studies WHERE status='Passed'")[0][0]
    avg_sens = _safe(cur, "SELECT AVG(sensitivity) FROM validation_studies WHERE sensitivity IS NOT NULL")
    avg_sens = round(avg_sens[0][0], 3) if avg_sens and avg_sens[0][0] else None
    avg_spec = _safe(cur, "SELECT AVG(specificity) FROM validation_studies WHERE specificity IS NOT NULL")
    avg_spec = round(avg_spec[0][0], 3) if avg_spec and avg_spec[0][0] else None
    avg_auc = _safe(cur, "SELECT AVG(auc_roc) FROM validation_studies WHERE auc_roc IS NOT NULL")
    avg_auc = round(avg_auc[0][0], 3) if avg_auc and avg_auc[0][0] else None

    # Pathway distribution
    pathway_dist = _safe(cur, "SELECT pathway, COUNT(*) as cnt FROM regulatory_submissions GROUP BY pathway ORDER BY cnt DESC")

    # Status distribution
    status_dist = _safe(cur, "SELECT status, COUNT(*) as cnt FROM regulatory_submissions GROUP BY status ORDER BY cnt DESC")

    # Risk class distribution
    risk_dist = _safe(cur, "SELECT risk_class, COUNT(*) as cnt FROM regulatory_submissions GROUP BY risk_class ORDER BY cnt DESC")

    # Phase distribution
    phase_dist = _safe(cur, "SELECT phase, COUNT(*) as cnt FROM regulatory_submissions GROUP BY phase ORDER BY cnt DESC")

    # Validation study status
    study_status = _safe(cur, "SELECT status, COUNT(*) as cnt FROM validation_studies GROUP BY status ORDER BY cnt DESC")

    # Avg validation score
    avg_val_score = _safe(cur, "SELECT AVG(validation_score) FROM regulatory_submissions WHERE validation_score IS NOT NULL")
    avg_val_score = round(avg_val_score[0][0], 3) if avg_val_score and avg_val_score[0][0] else None

    approval_rate = round(approved / total * 100, 1) if total > 0 else 0
    pass_rate = round(passed_studies / total_studies * 100, 1) if total_studies > 0 else 0

    conn.close()
    return {
        "kpis": {
            "total_submissions": total,
            "approved": approved,
            "under_review": under_review,
            "submitted": submitted,
            "products_tracked": products,
            "approval_rate_pct": approval_rate,
            "total_studies": total_studies,
            "passed_studies": passed_studies,
            "study_pass_rate_pct": pass_rate,
            "avg_sensitivity": avg_sens,
            "avg_specificity": avg_spec,
            "avg_auc_roc": avg_auc,
            "avg_validation_score": avg_val_score,
        },
        "pathway_distribution": [{"pathway": r[0], "count": r[1]} for r in pathway_dist],
        "status_distribution": [{"status": r[0], "count": r[1]} for r in status_dist],
        "risk_class_distribution": [{"risk_class": r[0], "count": r[1]} for r in risk_dist],
        "phase_distribution": [{"phase": r[0], "count": r[1]} for r in phase_dist],
        "study_status_distribution": [{"status": r[0], "count": r[1]} for r in study_status],
    }


def breakdown():
    """Regulatory breakdown — per-product submission table, validation study details,
    performance metrics, audit timeline, reviewer workload."""
    conn = _conn()
    cur = conn.cursor()

    # Full submission table
    rows = _safe(cur, """SELECT submission_id, pathway, product_name, classification,
        status, submitted_date, target_date, reviewer, phase, risk_class, validation_score
        FROM regulatory_submissions ORDER BY submitted_date DESC""")
    submissions = [
        {"submission_id": r[0], "pathway": r[1], "product_name": r[2],
         "classification": r[3], "status": r[4], "submitted_date": r[5],
         "target_date": r[6], "reviewer": r[7], "phase": r[8],
         "risk_class": r[9], "validation_score": r[10]}
        for r in rows
    ]

    # Validation studies with metrics
    rows = _safe(cur, """SELECT study_id, submission_id, study_type, title, status,
        sample_size, sensitivity, specificity, auc_roc, start_date, end_date,
        principal_investigator, site
        FROM validation_studies ORDER BY start_date DESC""")
    studies = [
        {"study_id": r[0], "submission_id": r[1], "study_type": r[2],
         "title": r[3], "status": r[4], "sample_size": r[5],
         "sensitivity": r[6], "specificity": r[7], "auc_roc": r[8],
         "start_date": r[9], "end_date": r[10],
         "principal_investigator": r[11], "site": r[12]}
        for r in rows
    ]

    # Reviewer workload
    rows = _safe(cur, """SELECT reviewer, COUNT(*) as cnt,
        SUM(CASE WHEN status='Approved' THEN 1 ELSE 0 END) as approved
        FROM regulatory_submissions GROUP BY reviewer ORDER BY cnt DESC""")
    reviewer_workload = [{"reviewer": r[0], "submissions": r[1], "approved": r[2]} for r in rows]

    # Per-product summary
    rows = _safe(cur, """SELECT product_name, COUNT(*) as submissions,
        SUM(CASE WHEN status IN ('Approved','Conditionally Approved') THEN 1 ELSE 0 END) as approved,
        AVG(validation_score) as avg_score
        FROM regulatory_submissions GROUP BY product_name ORDER BY submissions DESC""")
    product_summary = [
        {"product": r[0], "submissions": r[1], "approved": r[2],
         "avg_validation_score": round(r[3], 3) if r[3] else None}
        for r in rows
    ]

    # Study type performance
    rows = _safe(cur, """SELECT study_type, COUNT(*) as cnt,
        AVG(sensitivity) as avg_sens, AVG(specificity) as avg_spec, AVG(auc_roc) as avg_auc
        FROM validation_studies WHERE sensitivity IS NOT NULL
        GROUP BY study_type ORDER BY cnt DESC""")
    study_type_perf = [
        {"study_type": r[0], "count": r[1],
         "avg_sensitivity": round(r[2], 3) if r[2] else None,
         "avg_specificity": round(r[3], 3) if r[3] else None,
         "avg_auc_roc": round(r[4], 3) if r[4] else None}
        for r in rows
    ]

    # Audit trail (recent 50)
    rows = _safe(cur, """SELECT submission_id, action, actor, timestamp, details, document_ref, category
        FROM regulatory_audit_trail ORDER BY timestamp DESC LIMIT 50""")
    audit_trail = [
        {"submission_id": r[0], "action": r[1], "actor": r[2],
         "timestamp": r[3], "details": r[4], "document_ref": r[5], "category": r[6]}
        for r in rows
    ]

    # Audit category distribution
    rows = _safe(cur, "SELECT category, COUNT(*) as cnt FROM regulatory_audit_trail GROUP BY category ORDER BY cnt DESC")
    audit_categories = [{"category": r[0], "count": r[1]} for r in rows]

    conn.close()
    return {
        "submissions": submissions,
        "validation_studies": studies,
        "reviewer_workload": reviewer_workload,
        "product_summary": product_summary,
        "study_type_performance": study_type_perf,
        "audit_trail": audit_trail,
        "audit_categories": audit_categories,
    }


def definitions():
    """Regulatory definitions — pathway descriptions, risk classifications,
    validation criteria, regulatory standards, glossary."""
    return {
        "regulatory_pathways": [
            {"pathway": "FDA 510(k)", "description": "Premarket notification for devices substantially equivalent to a legally marketed predicate device. Most common pathway for SaMD Class II.", "timeline": "3-12 months", "evidence": "Analytical + clinical performance data"},
            {"pathway": "FDA De Novo", "description": "Classification pathway for novel low-to-moderate risk devices without a predicate. Creates a new device classification.", "timeline": "6-12 months", "evidence": "Clinical validation + risk/benefit analysis"},
            {"pathway": "FDA PMA", "description": "Premarket Approval for Class III high-risk devices. Most stringent pathway requiring clinical trials.", "timeline": "12-24 months", "evidence": "Prospective clinical trials + manufacturing QMS"},
            {"pathway": "CE Mark (MDR)", "description": "EU Medical Device Regulation conformity assessment via Notified Body. Required for EU market access.", "timeline": "6-18 months", "evidence": "Clinical evaluation report + technical documentation"},
            {"pathway": "CE Mark (IVDR)", "description": "EU In Vitro Diagnostic Regulation pathway for diagnostic AI/ML algorithms.", "timeline": "6-18 months", "evidence": "Analytical + clinical performance study"},
        ],
        "risk_classifications": [
            {"class": "Class I", "description": "Low risk — general controls only (registration, labeling, GMP)", "examples": "EEG artifact rejection, signal filtering tools", "regulatory_controls": "General controls"},
            {"class": "Class IIa", "description": "Low-moderate risk — requires conformity assessment by Notified Body (EU MDR)", "examples": "Sleep staging AI, non-critical decision support", "regulatory_controls": "General + special controls"},
            {"class": "Class IIb", "description": "Moderate-high risk — requires clinical evaluation and NB audit (EU MDR)", "examples": "Seizure detection, diagnostic algorithms", "regulatory_controls": "General + special controls + clinical evidence"},
            {"class": "Class III", "description": "High risk — most stringent controls, PMA required (FDA) or full conformity (MDR)", "examples": "Treatment recommendations, closed-loop stimulation control", "regulatory_controls": "General + special + premarket approval"},
        ],
        "validation_criteria": [
            {"metric": "Sensitivity", "description": "True positive rate — proportion of actual positives correctly identified", "threshold": "≥ 0.90 for seizure detection", "standard": "IEC 62304, FDA guidance"},
            {"metric": "Specificity", "description": "True negative rate — proportion of actual negatives correctly identified", "threshold": "≥ 0.85 for clinical SaMD", "standard": "IEC 62304, FDA guidance"},
            {"metric": "AUC-ROC", "description": "Area under receiver operating characteristic curve — overall discriminative ability", "threshold": "≥ 0.90 for Class II devices", "standard": "FDA AI/ML guidance 2021"},
            {"metric": "Validation Score", "description": "Composite score combining analytical validity, clinical validity, and clinical utility", "threshold": "≥ 0.80 for regulatory submission readiness", "standard": "Internal quality gate"},
            {"metric": "Sample Size", "description": "Number of independent cases in the validation cohort", "threshold": "≥ 200 for pivotal study (FDA guidance)", "standard": "FDA statistical guidance"},
        ],
        "regulatory_standards": [
            {"standard": "IEC 62304", "title": "Medical device software lifecycle processes", "scope": "Software development, maintenance, risk management"},
            {"standard": "ISO 14971", "title": "Application of risk management to medical devices", "scope": "Risk analysis, evaluation, control, residual risk"},
            {"standard": "ISO 13485", "title": "Medical devices — Quality management systems", "scope": "Design controls, CAPA, document control, traceability"},
            {"standard": "FDA 21 CFR Part 820", "title": "Quality System Regulation (QSR)", "scope": "Design controls, production controls, CAPA, records"},
            {"standard": "EU MDR 2017/745", "title": "Medical Device Regulation", "scope": "Classification, conformity, clinical evaluation, PMS"},
            {"standard": "IMDRF SaMD N12", "title": "Software as Medical Device — Clinical Evaluation", "scope": "Valid clinical association, analytical + clinical validation"},
            {"standard": "FDA AI/ML Action Plan", "title": "Artificial Intelligence/Machine Learning Action Plan (2021)", "scope": "GMLP, predetermined change control, transparency, real-world performance"},
        ],
        "glossary": [
            {"term": "SaMD", "definition": "Software as a Medical Device — software intended for medical purposes without being part of a hardware device"},
            {"term": "GMLP", "definition": "Good Machine Learning Practice — FDA/Health Canada/MHRA guiding principles for AI/ML-based SaMD"},
            {"term": "Predetermined Change Control Plan", "definition": "FDA framework allowing pre-specified modifications to AI/ML algorithms without new submission"},
            {"term": "Clinical Evaluation Report (CER)", "definition": "EU MDR requirement documenting clinical evidence supporting safety and performance claims"},
            {"term": "Design Controls", "definition": "FDA QSR requirement for systematic design input/output, verification, validation, and review"},
            {"term": "CAPA", "definition": "Corrective and Preventive Action — systematic process to identify, investigate, and resolve quality issues"},
            {"term": "Post-Market Surveillance (PMS)", "definition": "Ongoing monitoring of device safety and performance after regulatory approval and market launch"},
            {"term": "Notified Body", "definition": "EU-designated organization that performs conformity assessment for medical devices under MDR/IVDR"},
            {"term": "510(k) Substantial Equivalence", "definition": "FDA determination that a new device is as safe and effective as a legally marketed predicate device"},
            {"term": "Real-World Evidence (RWE)", "definition": "Clinical evidence derived from real-world data outside of traditional clinical trials"},
        ],
        "references": [
            "FDA Guidance: Clinical Decision Support Software (2022)",
            "FDA Guidance: Artificial Intelligence/Machine Learning-Based SaMD Action Plan (2021)",
            "EU MDR 2017/745 — Regulation on Medical Devices",
            "IMDRF SaMD N12: Software as Medical Device — Clinical Evaluation",
            "IEC 62304:2006+AMD1:2015 — Medical device software lifecycle",
            "ISO 14971:2019 — Risk management for medical devices",
        ],
    }
