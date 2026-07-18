"""Validation Studies Dashboard — clinical validation analytics from clinical.db.

Tracks AI/SaMD validation studies including clinical validation, software
verification, analytical validation, prospective trials, retrospective cohorts,
cross-validation, and usability studies across multiple international sites.

Sources:
- validation_studies table (study_id, submission_id, study_type, title, status,
  sample_size, sensitivity, specificity, auc_roc, start_date, end_date,
  principal_investigator, site, findings)
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

STUDY_TYPES = [
    "Software Verification", "Clinical Validation", "Analytical Validation",
    "Prospective Trial", "Retrospective Cohort", "Cross-validation", "Usability Study",
]

STATUS_ORDER = ["Completed", "Passed", "In Progress", "Planned", "Failed - Remediation"]


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


# ──────────────────────────────────────────────────────────────
#  /api/validation-studies/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """High-level validation study metrics, charts, and distributions."""
    conn = _conn()
    cur = conn.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*) FROM validation_studies")
    total_studies = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT submission_id) FROM validation_studies")
    total_submissions = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT site) FROM validation_studies")
    total_sites = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT principal_investigator) FROM validation_studies")
    total_pis = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM validation_studies WHERE status IN ('Completed','Passed')")
    completed_passed = cur.fetchone()[0]

    pass_rate = round(completed_passed / total_studies * 100, 1) if total_studies else 0.0

    cur.execute("SELECT ROUND(AVG(sensitivity), 3) FROM validation_studies WHERE sensitivity IS NOT NULL")
    avg_sensitivity = cur.fetchone()[0] or 0.0

    cur.execute("SELECT ROUND(AVG(specificity), 3) FROM validation_studies WHERE specificity IS NOT NULL")
    avg_specificity = cur.fetchone()[0] or 0.0

    cur.execute("SELECT ROUND(AVG(auc_roc), 3) FROM validation_studies WHERE auc_roc IS NOT NULL")
    avg_auc = cur.fetchone()[0] or 0.0

    cur.execute("SELECT ROUND(AVG(sample_size), 0) FROM validation_studies")
    avg_sample_size = int(cur.fetchone()[0] or 0)

    kpis = {
        "total_studies": total_studies,
        "total_submissions": total_submissions,
        "total_sites": total_sites,
        "total_pis": total_pis,
        "pass_rate_pct": pass_rate,
        "avg_sensitivity": avg_sensitivity,
        "avg_specificity": avg_specificity,
        "avg_auc_roc": avg_auc,
        "avg_sample_size": avg_sample_size,
    }

    # Study type distribution
    cur.execute(
        "SELECT study_type, COUNT(*) AS cnt FROM validation_studies "
        "GROUP BY study_type ORDER BY cnt DESC"
    )
    study_type_distribution = [
        {"type": r[0], "count": r[1], "pct": round(r[1] / total_studies * 100, 1)}
        for r in cur.fetchall()
    ]

    # Status distribution
    cur.execute(
        "SELECT status, COUNT(*) AS cnt FROM validation_studies "
        "GROUP BY status ORDER BY cnt DESC"
    )
    status_distribution = [
        {"status": r[0], "count": r[1], "pct": round(r[1] / total_studies * 100, 1)}
        for r in cur.fetchall()
    ]

    # Site distribution
    cur.execute(
        "SELECT site, COUNT(*) AS cnt FROM validation_studies "
        "GROUP BY site ORDER BY cnt DESC"
    )
    site_distribution = [
        {"site": r[0], "count": r[1]} for r in cur.fetchall()
    ]

    # Performance by study type (avg sensitivity, specificity, AUC for each type)
    cur.execute(
        "SELECT study_type, "
        "ROUND(AVG(sensitivity), 3), ROUND(AVG(specificity), 3), ROUND(AVG(auc_roc), 3), "
        "ROUND(AVG(sample_size), 0), COUNT(*) "
        "FROM validation_studies "
        "WHERE sensitivity IS NOT NULL "
        "GROUP BY study_type ORDER BY AVG(auc_roc) DESC"
    )
    perf_by_type = [
        {
            "type": r[0], "avg_sensitivity": r[1], "avg_specificity": r[2],
            "avg_auc_roc": r[3], "avg_sample_size": int(r[4]) if r[4] else 0, "studies": r[5],
        }
        for r in cur.fetchall()
    ]

    # Performance by site
    cur.execute(
        "SELECT site, "
        "ROUND(AVG(sensitivity), 3), ROUND(AVG(specificity), 3), ROUND(AVG(auc_roc), 3), "
        "COUNT(*) "
        "FROM validation_studies "
        "WHERE sensitivity IS NOT NULL "
        "GROUP BY site ORDER BY AVG(auc_roc) DESC"
    )
    perf_by_site = [
        {
            "site": r[0], "avg_sensitivity": r[1], "avg_specificity": r[2],
            "avg_auc_roc": r[3], "studies": r[4],
        }
        for r in cur.fetchall()
    ]

    conn.close()
    return {
        "kpis": kpis,
        "study_type_distribution": study_type_distribution,
        "status_distribution": status_distribution,
        "site_distribution": site_distribution,
        "performance_by_type": perf_by_type,
        "performance_by_site": perf_by_site,
    }


# ──────────────────────────────────────────────────────────────
#  /api/validation-studies/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Detailed study records, failed studies, per-submission summary, PI workload."""
    conn = _conn()
    cur = conn.cursor()

    # Failed / remediation studies (alert table)
    cur.execute(
        "SELECT study_id, submission_id, study_type, title, site, "
        "principal_investigator, sample_size, start_date, end_date, findings "
        "FROM validation_studies WHERE status = 'Failed - Remediation' "
        "ORDER BY end_date DESC"
    )
    failed_studies = _dict_rows(cur)

    # In-progress studies
    cur.execute(
        "SELECT study_id, submission_id, study_type, title, site, "
        "principal_investigator, sample_size, start_date, status "
        "FROM validation_studies WHERE status = 'In Progress' "
        "ORDER BY start_date DESC"
    )
    in_progress_studies = _dict_rows(cur)

    # Per-submission summary
    cur.execute(
        "SELECT submission_id, COUNT(*) AS total_studies, "
        "SUM(CASE WHEN status IN ('Completed','Passed') THEN 1 ELSE 0 END) AS passed, "
        "SUM(CASE WHEN status = 'Failed - Remediation' THEN 1 ELSE 0 END) AS failed, "
        "SUM(CASE WHEN status = 'In Progress' THEN 1 ELSE 0 END) AS in_progress, "
        "SUM(CASE WHEN status = 'Planned' THEN 1 ELSE 0 END) AS planned, "
        "ROUND(AVG(CASE WHEN sensitivity IS NOT NULL THEN sensitivity END), 3) AS avg_sensitivity, "
        "ROUND(AVG(CASE WHEN auc_roc IS NOT NULL THEN auc_roc END), 3) AS avg_auc "
        "FROM validation_studies GROUP BY submission_id ORDER BY submission_id"
    )
    per_submission = _dict_rows(cur)

    # PI workload
    cur.execute(
        "SELECT principal_investigator, COUNT(*) AS studies, "
        "SUM(CASE WHEN status IN ('Completed','Passed') THEN 1 ELSE 0 END) AS passed, "
        "SUM(CASE WHEN status = 'Failed - Remediation' THEN 1 ELSE 0 END) AS failed, "
        "ROUND(AVG(sample_size), 0) AS avg_sample_size "
        "FROM validation_studies GROUP BY principal_investigator "
        "ORDER BY studies DESC"
    )
    pi_workload = _dict_rows(cur)

    # Recent / all studies table
    cur.execute(
        "SELECT study_id, submission_id, study_type, title, status, "
        "sample_size, sensitivity, specificity, auc_roc, "
        "start_date, end_date, principal_investigator, site "
        "FROM validation_studies ORDER BY start_date DESC"
    )
    all_studies = _dict_rows(cur)

    # Top performing studies (by AUC)
    cur.execute(
        "SELECT study_id, study_type, title, status, "
        "sensitivity, specificity, auc_roc, sample_size, site, principal_investigator "
        "FROM validation_studies WHERE auc_roc IS NOT NULL "
        "ORDER BY auc_roc DESC LIMIT 10"
    )
    top_performing = _dict_rows(cur)

    conn.close()
    return {
        "failed_studies": failed_studies,
        "in_progress_studies": in_progress_studies,
        "per_submission": per_submission,
        "pi_workload": pi_workload,
        "all_studies": all_studies,
        "top_performing": top_performing,
    }


# ──────────────────────────────────────────────────────────────
#  /api/validation-studies/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Study type descriptions, metric definitions, status definitions, glossary."""
    return {
        "study_types": [
            {"type": "Software Verification", "description": "Confirms the software was built correctly per design specifications. Tests functional requirements, edge cases, and integration points."},
            {"type": "Clinical Validation", "description": "Demonstrates the device meets user needs and intended use in real clinical settings. Gold-standard comparison with patient data."},
            {"type": "Analytical Validation", "description": "Verifies accuracy, precision, and reproducibility of the analytical/measurement component under controlled conditions."},
            {"type": "Prospective Trial", "description": "Forward-looking study enrolling patients to evaluate real-time performance. Strongest evidence for clinical claims."},
            {"type": "Retrospective Cohort", "description": "Analysis of previously collected patient data to assess algorithm performance on historical cases."},
            {"type": "Cross-validation", "description": "K-fold or leave-one-out cross-validation to evaluate model generalization without data leakage."},
            {"type": "Usability Study", "description": "Human-factors evaluation of the user interface and clinical workflow integration with intended users."},
        ],
        "metrics": [
            {"metric": "Sensitivity (Recall)", "description": "True positive rate — proportion of actual positives correctly identified. Critical for ruling out disease (high sensitivity = few missed cases)."},
            {"metric": "Specificity", "description": "True negative rate — proportion of actual negatives correctly identified. Critical for ruling in disease (high specificity = few false alarms)."},
            {"metric": "AUC-ROC", "description": "Area Under the Receiver Operating Characteristic curve. Overall discriminative ability: 0.5 = random, 0.7-0.8 = acceptable, 0.8-0.9 = excellent, >0.9 = outstanding."},
            {"metric": "Sample Size", "description": "Number of subjects/recordings included in the study. Larger samples increase statistical power and generalizability."},
        ],
        "statuses": [
            {"status": "Completed", "description": "Study finished and results accepted — met all acceptance criteria."},
            {"status": "Passed", "description": "Study passed all predefined acceptance thresholds (sensitivity, specificity, AUC targets)."},
            {"status": "In Progress", "description": "Study currently enrolling, collecting data, or under analysis."},
            {"status": "Planned", "description": "Study protocol approved but not yet started — awaiting site readiness or enrollment."},
            {"status": "Failed - Remediation", "description": "Study did not meet acceptance criteria. Root cause analysis and corrective actions underway."},
        ],
        "regulatory_context": [
            {"item": "IEC 62304", "description": "Software lifecycle standard — requires software verification for all SaMD."},
            {"item": "ISO 13485", "description": "Quality management system — mandates design validation."},
            {"item": "FDA 21 CFR 820.30", "description": "Design controls — requires design verification and validation."},
            {"item": "EU MDR 2017/745", "description": "Medical Device Regulation — requires clinical evaluation and performance studies."},
            {"item": "IMDRF SaMD N41", "description": "International framework for SaMD clinical evaluation — risk-based evidence requirements."},
        ],
        "glossary": [
            {"term": "SaMD", "definition": "Software as a Medical Device — software intended for medical purposes without being part of a hardware device."},
            {"term": "V&V", "definition": "Verification & Validation — verification confirms correct build; validation confirms correct product."},
            {"term": "Acceptance Criteria", "definition": "Predefined thresholds (e.g., sensitivity > 0.85, AUC > 0.90) that a study must meet to pass."},
            {"term": "PI", "definition": "Principal Investigator — the lead researcher responsible for a validation study."},
            {"term": "Cross-validation", "definition": "Statistical method that partitions data into training/test folds to estimate model generalization."},
            {"term": "Remediation", "definition": "Corrective actions taken when a study fails — may include retraining, data augmentation, or protocol changes."},
            {"term": "Design Freeze", "definition": "Point at which the software design is locked for validation — no further changes allowed without re-validation."},
            {"term": "Predicate Device", "definition": "An existing legally marketed device to which a new device is compared for regulatory equivalence."},
            {"term": "Clinical Evidence", "definition": "Data and analysis demonstrating clinical performance and safety of a medical device."},
            {"term": "Post-Market Surveillance", "definition": "Ongoing monitoring of device performance after regulatory clearance/approval."},
        ],
    }
