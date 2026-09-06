"""
Regulatory Compliance Dashboard
================================
FDA/CE pathway tracking, validation studies, and audit trail analytics.

Data sources:
- regulatory_submissions (16 rows): FDA De Novo / 510(k) / CE Mark pathways
- validation_studies (42 rows): clinical validation, software verification, usability
- regulatory_audit_trail (102 rows): timestamped compliance actions
"""

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")


def _conn():
    return sqlite3.connect(_DB_PATH)


def overview():
    """KPIs, submission pipeline, validation pass rates, audit activity."""
    db = _conn()
    db.row_factory = sqlite3.Row

    # --- Submissions ---
    subs = [dict(r) for r in db.execute("SELECT * FROM regulatory_submissions ORDER BY submitted_date DESC").fetchall()]
    status_counts = Counter(s["status"] for s in subs)
    pathway_counts = Counter(s["pathway"] for s in subs)
    risk_counts = Counter(s["risk_class"] for s in subs)
    avg_val_score = None
    scored = [s["validation_score"] for s in subs if s.get("validation_score") is not None]
    if scored:
        avg_val_score = round(sum(scored) / len(scored), 3)

    # --- Validation studies ---
    vals = [dict(r) for r in db.execute("SELECT * FROM validation_studies ORDER BY start_date DESC").fetchall()]
    study_type_counts = Counter(v["study_type"] for v in vals)
    val_status_counts = Counter(v["status"] for v in vals)
    completed_with_metrics = [v for v in vals if v.get("sensitivity") is not None]
    avg_sensitivity = round(sum(v["sensitivity"] for v in completed_with_metrics) / len(completed_with_metrics), 3) if completed_with_metrics else None
    avg_specificity = round(sum(v["specificity"] for v in completed_with_metrics) / len(completed_with_metrics), 3) if completed_with_metrics else None
    avg_auc = round(sum(v["auc_roc"] for v in completed_with_metrics) / len(completed_with_metrics), 3) if completed_with_metrics else None

    # --- Audit trail ---
    audits = [dict(r) for r in db.execute("SELECT * FROM regulatory_audit_trail ORDER BY timestamp DESC").fetchall()]
    action_counts = Counter(a["action"] for a in audits)
    actor_counts = Counter(a["actor"] for a in audits)
    category_counts = Counter(a["category"] for a in audits)

    # Monthly audit volume
    monthly = Counter()
    for a in audits:
        ts = a.get("timestamp", "")
        if ts and len(ts) >= 7:
            monthly[ts[:7]] += 1
    monthly_trend = [{"month": k, "count": v} for k, v in sorted(monthly.items())]

    db.close()

    return {
        "kpis": {
            "total_submissions": len(subs),
            "approved": status_counts.get("Approved", 0),
            "in_review": sum(v for k, v in status_counts.items() if k not in ("Approved", "Withdrawn")),
            "validation_studies": len(vals),
            "avg_validation_score": avg_val_score,
            "avg_sensitivity": avg_sensitivity,
            "avg_specificity": avg_specificity,
            "avg_auc_roc": avg_auc,
            "audit_events": len(audits),
        },
        "submissions_by_status": [{"status": k, "count": v} for k, v in sorted(status_counts.items(), key=lambda x: -x[1])],
        "submissions_by_pathway": [{"pathway": k, "count": v} for k, v in sorted(pathway_counts.items(), key=lambda x: -x[1])],
        "risk_class_distribution": [{"risk_class": k, "count": v} for k, v in sorted(risk_counts.items(), key=lambda x: -x[1])],
        "validation_by_type": [{"study_type": k, "count": v} for k, v in sorted(study_type_counts.items(), key=lambda x: -x[1])],
        "validation_by_status": [{"status": k, "count": v} for k, v in sorted(val_status_counts.items(), key=lambda x: -x[1])],
        "audit_by_action": [{"action": k, "count": v} for k, v in sorted(action_counts.items(), key=lambda x: -x[1])],
        "audit_by_category": [{"category": k, "count": v} for k, v in sorted(category_counts.items(), key=lambda x: -x[1])],
        "monthly_audit_trend": monthly_trend,
    }


def breakdown():
    """Per-submission detail, validation study results, audit trail per submission."""
    db = _conn()
    db.row_factory = sqlite3.Row

    subs = [dict(r) for r in db.execute("SELECT * FROM regulatory_submissions ORDER BY submitted_date DESC").fetchall()]
    vals = [dict(r) for r in db.execute("SELECT * FROM validation_studies ORDER BY start_date DESC").fetchall()]
    audits = [dict(r) for r in db.execute("SELECT * FROM regulatory_audit_trail ORDER BY timestamp DESC").fetchall()]

    # Group validation studies by submission
    val_by_sub = {}
    for v in vals:
        sid = v.get("submission_id", "")
        val_by_sub.setdefault(sid, []).append(v)

    # Group audits by submission
    audit_by_sub = {}
    for a in audits:
        sid = a.get("submission_id", "")
        audit_by_sub.setdefault(sid, []).append(a)

    # Per-submission dossier
    submission_details = []
    for s in subs:
        sid = s["submission_id"]
        s_vals = val_by_sub.get(sid, [])
        s_audits = audit_by_sub.get(sid, [])
        completed_vals = [v for v in s_vals if v.get("sensitivity") is not None]
        submission_details.append({
            "submission_id": sid,
            "pathway": s["pathway"],
            "product_name": s["product_name"],
            "classification": s["classification"],
            "status": s["status"],
            "submitted_date": s["submitted_date"],
            "target_date": s["target_date"],
            "reviewer": s["reviewer"],
            "phase": s["phase"],
            "risk_class": s["risk_class"],
            "validation_score": s["validation_score"],
            "validation_studies": len(s_vals),
            "completed_validations": len(completed_vals),
            "avg_sensitivity": round(sum(v["sensitivity"] for v in completed_vals) / len(completed_vals), 3) if completed_vals else None,
            "avg_specificity": round(sum(v["specificity"] for v in completed_vals) / len(completed_vals), 3) if completed_vals else None,
            "avg_auc": round(sum(v["auc_roc"] for v in completed_vals) / len(completed_vals), 3) if completed_vals else None,
            "audit_events": len(s_audits),
        })

    # Validation study details
    validation_details = []
    for v in vals:
        validation_details.append({
            "study_id": v["study_id"],
            "submission_id": v["submission_id"],
            "study_type": v["study_type"],
            "title": v["title"],
            "status": v["status"],
            "sample_size": v["sample_size"],
            "sensitivity": v["sensitivity"],
            "specificity": v["specificity"],
            "auc_roc": v["auc_roc"],
            "start_date": v["start_date"],
            "end_date": v["end_date"],
            "principal_investigator": v["principal_investigator"],
            "site": v["site"],
            "findings": v["findings"],
        })

    # Recent audit trail (top 30)
    recent_audits = []
    for a in audits[:30]:
        recent_audits.append({
            "submission_id": a["submission_id"],
            "action": a["action"],
            "actor": a["actor"],
            "timestamp": a["timestamp"],
            "category": a["category"],
            "document_ref": a["document_ref"],
            "details": a["details"],
        })

    # Top actors
    actor_counts = Counter(a["actor"] for a in audits)
    top_actors = [{"actor": k, "events": v} for k, v in sorted(actor_counts.items(), key=lambda x: -x[1])]

    db.close()

    return {
        "submission_details": submission_details,
        "validation_details": validation_details,
        "recent_audit_trail": recent_audits,
        "top_actors": top_actors,
    }


def definitions():
    """Regulatory terms, pathway descriptions, risk class definitions."""
    return {
        "terms": [
            {"term": "SaMD", "definition": "Software as a Medical Device — software intended for medical purposes that is not part of a hardware medical device (IEC 62304, FDA guidance)."},
            {"term": "510(k)", "definition": "FDA premarket notification — demonstrate substantial equivalence to a legally marketed predicate device."},
            {"term": "De Novo", "definition": "FDA pathway for novel low-to-moderate risk devices without a predicate. Results in a new regulatory classification."},
            {"term": "CE Mark", "definition": "European conformity marking under MDR 2017/745 — required for medical devices marketed in the EU/EEA."},
            {"term": "Clinical Validation", "definition": "Study demonstrating that a device achieves its intended clinical benefit in the target patient population."},
            {"term": "Software Verification", "definition": "Confirmation through objective evidence that software specifications are correctly implemented (IEC 62304)."},
            {"term": "Usability Study", "definition": "Formative or summative evaluation of user interface design per IEC 62366-1 to ensure safe and effective use."},
            {"term": "Sensitivity", "definition": "True positive rate — proportion of actual positives correctly identified. Critical for seizure detection (miss rate = 1 − sensitivity)."},
            {"term": "Specificity", "definition": "True negative rate — proportion of actual negatives correctly identified. High specificity reduces false alarm burden."},
            {"term": "AUC-ROC", "definition": "Area Under the Receiver Operating Characteristic curve — aggregate measure of discriminative ability across all thresholds (0.5 = random, 1.0 = perfect)."},
            {"term": "Risk Class", "definition": "Regulatory risk classification (EU MDR: Class I / IIa / IIb / III; FDA: Class I / II / III) based on intended use, duration of contact, and invasiveness."},
            {"term": "Pre-submission", "definition": "Voluntary meeting/correspondence with FDA to discuss device classification, testing strategy, and submission pathway before formal filing."},
            {"term": "Phase", "definition": "Current regulatory phase — Pre-submission, Clinical Evaluation, Post-market Surveillance, or Approved/Cleared."},
            {"term": "Validation Score", "definition": "Composite score (0–1) summarizing the strength of the validation evidence package for a submission."},
        ],
        "pathways": [
            {"name": "FDA 510(k)", "description": "Most common FDA pathway. Requires demonstration of substantial equivalence to a predicate device. Typical review: 90–180 days."},
            {"name": "FDA De Novo", "description": "For novel devices without a predicate. Requires risk-based classification and special controls. Typical review: 150–300 days."},
            {"name": "CE Mark (MDR)", "description": "European conformity via Notified Body. Requires clinical evaluation, post-market surveillance plan, and QMS (ISO 13485)."},
        ],
        "risk_classes": [
            {"class": "Class I", "description": "Lowest risk. General controls only (e.g., tongue depressors, elastic bandages)."},
            {"class": "Class IIa", "description": "Low-to-moderate risk. Requires conformity assessment (e.g., diagnostic EEG software with clinician review)."},
            {"class": "Class IIb", "description": "Moderate-to-high risk. Requires notified body involvement (e.g., AI-assisted seizure detection triggering clinical action)."},
            {"class": "Class III", "description": "Highest risk. Full conformity assessment (e.g., implantable neurostimulators)."},
        ],
    }
