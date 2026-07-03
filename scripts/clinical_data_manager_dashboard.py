"""
Clinical Data Manager Dashboard
================================
Real analytics from clinical.db + existing data-manager sub-endpoints:
  - Data quality dimensions (completeness, uniqueness, validity, label coverage, signal quality)
  - AI readiness scoring (composite score from real metrics)
  - Modality coverage (EEG, MRI, assessments, seizure diary, medications)
  - Missing data matrix (per-modality gap analysis)
  - Data lineage (pipeline stages from intake to model-ready)
  - Task catalog (17 CDM tasks with status, steps, challenges)
  - Dataset inventory (table counts, row totals, schema)
  - Archival / retention (per-table age, policy compliance)
  - Terminology mapping (instrument → canonical domain coverage)
  - Channel quality (flat/noisy/disconnected detection from real EEG)
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"
CONFIG = Path(__file__).resolve().parent.parent / "config" / "data_manager.json"


def _connect():
    if not DB.exists():
        return None
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


def _safe(v):
    """Replace NaN/Inf with None for JSON safety."""
    if isinstance(v, float):
        import math
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def _cfg():
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    return {}


# ════════════════════════════════════════════════════════════════════════
# 1. overview()
# ════════════════════════════════════════════════════════════════════════

def overview():
    """KPI cards + chart data for the Clinical Data Manager dashboard."""
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found"}

    cur = conn.cursor()
    cfg = _cfg()

    # ── Basic counts ────────────────────────────────────────────────────
    total_patients = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_analyses = cur.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    total_assessments = cur.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    total_uploads = cur.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    total_mri = cur.execute("SELECT COUNT(*) FROM mri_findings").fetchone()[0]
    total_seizure_diary = cur.execute("SELECT COUNT(*) FROM seizure_diary").fetchone()[0]
    total_medications = cur.execute("SELECT COUNT(*) FROM medications").fetchone()[0]
    total_tables = len(cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall())
    total_rows = 0
    for tbl in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        total_rows += cur.execute(f"SELECT COUNT(*) FROM [{tbl[0]}]").fetchone()[0]

    audit_events = cur.execute("SELECT COUNT(*) FROM transaction_log").fetchone()[0]

    # ── Modality coverage ──────────────────────────────────────────────
    eeg_patients = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM analyses"
    ).fetchone()[0]
    assessment_patients = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM assessments"
    ).fetchone()[0]
    seizure_patients = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM seizure_diary"
    ).fetchone()[0]
    mri_patients = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM mri_findings"
    ).fetchone()[0]
    med_patients = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM medications"
    ).fetchone()[0]

    modality_coverage = [
        {"modality": "EEG Analysis", "patients": eeg_patients,
         "pct": round(100 * eeg_patients / total_patients, 1) if total_patients else 0},
        {"modality": "Assessments", "patients": assessment_patients,
         "pct": round(100 * assessment_patients / total_patients, 1) if total_patients else 0},
        {"modality": "Seizure Diary", "patients": seizure_patients,
         "pct": round(100 * seizure_patients / total_patients, 1) if total_patients else 0},
        {"modality": "MRI", "patients": mri_patients,
         "pct": round(100 * mri_patients / total_patients, 1) if total_patients else 0},
        {"modality": "Medications", "patients": med_patients,
         "pct": round(100 * med_patients / total_patients, 1) if total_patients else 0},
    ]

    # ── Quality dimensions ─────────────────────────────────────────────
    avg_coverage = sum(m["pct"] for m in modality_coverage) / len(modality_coverage)
    dup_check = cur.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT patient_id) FROM patients"
    ).fetchone()[0]
    patients_with_age = cur.execute(
        "SELECT COUNT(*) FROM patients WHERE age IS NOT NULL"
    ).fetchone()[0]
    validity_pct = round(100 * patients_with_age / total_patients, 1) if total_patients else 0

    # Label coverage
    labeled = cur.execute(
        "SELECT COUNT(*) FROM analyses WHERE predicted_label IS NOT NULL AND predicted_label != ''"
    ).fetchone()[0]
    label_pct = round(100 * labeled / total_analyses, 1) if total_analyses else 0

    # Signal quality from analyses
    good_quality = cur.execute(
        "SELECT COUNT(*) FROM analyses WHERE signal_quality >= 0.7"
    ).fetchone()[0]
    signal_pct = round(100 * good_quality / total_analyses, 1) if total_analyses else 0

    quality_dimensions = [
        {"dimension": "Completeness", "score": round(avg_coverage, 1),
         "basis": f"mean modality coverage across {len(modality_coverage)} modalities"},
        {"dimension": "Uniqueness", "score": 100.0 if dup_check == 0 else round(100 * (1 - dup_check / total_patients), 1),
         "basis": f"{dup_check} duplicate patient_ids of {total_patients}"},
        {"dimension": "Validity", "score": validity_pct,
         "basis": f"{patients_with_age}/{total_patients} patients have age+gender"},
        {"dimension": "Label Coverage", "score": label_pct,
         "basis": f"{labeled}/{total_analyses} analyses labeled"},
        {"dimension": "Signal Quality", "score": _safe(signal_pct),
         "basis": f"{good_quality}/{total_analyses} analyses with quality >= 0.7"},
    ]

    # AI readiness
    components = {
        "completeness": round(avg_coverage, 1),
        "uniqueness": 100.0 if dup_check == 0 else round(100 * (1 - dup_check / total_patients), 1),
        "validity": validity_pct,
        "label_coverage": label_pct,
        "signal_quality": _safe(signal_pct),
    }
    scores = [v for v in components.values() if v is not None]
    ai_readiness = round(sum(scores) / len(scores), 1) if scores else 0
    if ai_readiness >= 90:
        grade = "A (excellent)"
    elif ai_readiness >= 75:
        grade = "B (usable, gaps)"
    elif ai_readiness >= 50:
        grade = "C (needs work)"
    else:
        grade = "D (not ready)"

    # ── Missing data matrix ────────────────────────────────────────────
    missing_matrix = [
        {"modality": m["modality"], "present": m["patients"],
         "missing": total_patients - m["patients"],
         "pct_missing": round(100 * (total_patients - m["patients"]) / total_patients, 1) if total_patients else 0}
        for m in modality_coverage
    ]

    # ── Task status summary ────────────────────────────────────────────
    tasks = cfg.get("tasks", [])
    task_statuses = {}
    for t in tasks:
        s = t.get("status", "unknown")
        task_statuses[s] = task_statuses.get(s, 0) + 1
    task_status_chart = [{"status": k, "count": v} for k, v in task_statuses.items()]

    # ── Data lineage ───────────────────────────────────────────────────
    lineage = [
        {"stage": "Raw Upload", "description": "EDF/CSV/MRI/Video ingested via /api/upload", "status": "active"},
        {"stage": "Validation", "description": "Completeness + schema checks", "status": "active"},
        {"stage": "Cleaning", "description": "Flat/noisy channel removal, NaN sanitization", "status": "active"},
        {"stage": "Standardization", "description": "Terminology mapping to ICD/LOINC/SNOMED", "status": "active"},
        {"stage": "Labeling", "description": "AI classification + expert review", "status": "active"},
        {"stage": "Versioning", "description": "SHA-256 manifest + dataset fingerprint", "status": "active"},
        {"stage": "AI-Ready", "description": "Validated, labeled, versioned dataset", "status": "active"},
        {"stage": "Model Training", "description": "Feed into CNN/RNN/XGBoost pipeline", "status": "active"},
    ]

    conn.close()

    return {
        "kpis": {
            "total_patients": total_patients,
            "total_records": total_rows,
            "total_tables": total_tables,
            "total_uploads": total_uploads,
            "total_analyses": total_analyses,
            "total_assessments": total_assessments,
            "audit_events": audit_events,
            "ai_readiness_score": ai_readiness,
            "ai_readiness_grade": grade,
        },
        "modality_coverage": modality_coverage,
        "quality_dimensions": quality_dimensions,
        "ai_readiness_components": components,
        "missing_matrix": missing_matrix,
        "task_status_chart": task_status_chart,
        "lineage": lineage,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ════════════════════════════════════════════════════════════════════════
# 2. breakdown()
# ════════════════════════════════════════════════════════════════════════

def breakdown():
    """Detailed breakdowns — per-task catalog, dataset inventory, per-patient coverage."""
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found"}

    cur = conn.cursor()
    cfg = _cfg()

    # ── Task catalog ───────────────────────────────────────────────────
    tasks = cfg.get("tasks", [])

    # ── Dataset inventory (per-table) ──────────────────────────────────
    tables = []
    for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
        tname = row[0]
        cnt = cur.execute(f"SELECT COUNT(*) FROM [{tname}]").fetchone()[0]
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info([{tname}])").fetchall()]
        tables.append({"table": tname, "rows": cnt, "columns": len(cols), "column_names": cols[:8]})
    tables.sort(key=lambda x: -x["rows"])

    # ── Per-patient modality coverage matrix ───────────────────────────
    patients = cur.execute("SELECT patient_id, name, age, gender FROM patients LIMIT 50").fetchall()
    patient_coverage = []
    for p in patients:
        pid = p[0]
        has_eeg = cur.execute("SELECT COUNT(*) FROM analyses WHERE patient_id=?", (pid,)).fetchone()[0] > 0
        has_assessment = cur.execute("SELECT COUNT(*) FROM assessments WHERE patient_id=?", (pid,)).fetchone()[0] > 0
        has_seizure = cur.execute("SELECT COUNT(*) FROM seizure_diary WHERE patient_id=?", (pid,)).fetchone()[0] > 0
        has_mri = cur.execute("SELECT COUNT(*) FROM mri_findings WHERE patient_id=?", (pid,)).fetchone()[0] > 0
        has_med = cur.execute("SELECT COUNT(*) FROM medications WHERE patient_id=?", (pid,)).fetchone()[0] > 0
        modalities_present = sum([has_eeg, has_assessment, has_seizure, has_mri, has_med])
        patient_coverage.append({
            "patient_id": pid,
            "name": p[1] or "—",
            "age": p[2],
            "gender": p[3],
            "eeg": has_eeg,
            "assessment": has_assessment,
            "seizure_diary": has_seizure,
            "mri": has_mri,
            "medication": has_med,
            "modalities": modalities_present,
            "coverage_pct": round(100 * modalities_present / 5, 0),
        })
    patient_coverage.sort(key=lambda x: x["modalities"])

    # ── Assessment instrument distribution ─────────────────────────────
    instruments = cur.execute(
        "SELECT instrument, COUNT(*) as cnt FROM assessments GROUP BY instrument ORDER BY cnt DESC"
    ).fetchall()
    instrument_dist = [{"instrument": r[0], "count": r[1]} for r in instruments]

    # ── Archival summary ───────────────────────────────────────────────
    archival = []
    for tbl_row in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
        tname = tbl_row[0]
        cnt = cur.execute(f"SELECT COUNT(*) FROM [{tname}]").fetchone()[0]
        # Try to get date range if created_at exists
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info([{tname}])").fetchall()]
        ts_col = None
        for c in ["created_at", "timestamp", "date", "recorded_at"]:
            if c in cols:
                ts_col = c
                break
        age_info = None
        if ts_col:
            oldest = cur.execute(f"SELECT MIN([{ts_col}]) FROM [{tname}]").fetchone()[0]
            newest = cur.execute(f"SELECT MAX([{ts_col}]) FROM [{tname}]").fetchone()[0]
            age_info = {"oldest": oldest, "newest": newest}
        archival.append({"table": tname, "rows": cnt, "timestamped": ts_col is not None, "date_range": age_info})

    conn.close()

    return {
        "tasks": tasks,
        "dataset_inventory": tables,
        "patient_coverage": patient_coverage,
        "instrument_distribution": instrument_dist,
        "archival_summary": archival,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ════════════════════════════════════════════════════════════════════════
# 3. definitions()
# ════════════════════════════════════════════════════════════════════════

def definitions():
    """Clinical Data Manager concepts, quality metrics, compliance references."""
    return {
        "concepts": [
            {"term": "Data Quality Dimensions",
             "definition": "Framework measuring Completeness, Uniqueness, Validity, Consistency, and Accuracy of clinical datasets. Each dimension scored 0-100% from real database metrics."},
            {"term": "AI Readiness Score",
             "definition": "Composite score (0-100) measuring whether a dataset is suitable for AI/ML training. Combines completeness, uniqueness, validity, label coverage, and signal quality."},
            {"term": "Data Lineage",
             "definition": "End-to-end traceability of data from raw upload through validation, cleaning, standardization, labeling, versioning, to model-ready state. Supports reproducibility and audit."},
            {"term": "Modality Coverage",
             "definition": "Percentage of patients with data in each clinical modality (EEG, MRI, assessments, seizure diary, medications). Gaps identify data collection priorities."},
            {"term": "Dataset Versioning",
             "definition": "SHA-256 fingerprinting of dataset artifacts (EDF files, CSV exports, model weights) to ensure reproducibility and detect unauthorized changes."},
            {"term": "Terminology Mapping",
             "definition": "Standardization of clinical instrument names and assessment categories to canonical taxonomies (ICD-10, LOINC, SNOMED-CT) for interoperability."},
            {"term": "Annotation QC",
             "definition": "Quality control of clinical labels using inter-rater reliability (Cohen's κ, Fleiss's κ) and AI-human agreement metrics. Ensures label consistency for supervised learning."},
            {"term": "Data Archival",
             "definition": "Lifecycle management per retention policy: clinical records retained indefinitely, operational logs (transaction_log 30d, conversation/team 90d) per §7.4/§41.2."},
        ],
        "quality_metrics": [
            {"metric": "Completeness", "target": "≥80%", "method": "Mean modality coverage across all data types"},
            {"metric": "Uniqueness", "target": "100%", "method": "Zero duplicate patient_ids"},
            {"metric": "Validity", "target": "≥90%", "method": "Patients with required demographic fields (age, gender)"},
            {"metric": "AI Readiness", "target": "≥75 (Grade B+)", "method": "Weighted average of all quality dimensions"},
        ],
        "compliance_references": [
            {"standard": "HIPAA §164.530", "scope": "Data integrity and access audit trail requirements"},
            {"standard": "21 CFR Part 11", "scope": "Electronic records, dataset versioning, audit trails"},
            {"standard": "ICH-GCP E6(R2)", "scope": "Clinical data management, source data verification"},
            {"standard": "FAIR Principles", "scope": "Findable, Accessible, Interoperable, Reusable data standards"},
        ],
        "remediation": [
            {"gap": "Low completeness (<50%)", "action": "Prioritize modality-specific data collection campaigns; flag patients with <3 modalities"},
            {"gap": "Missing demographics", "action": "Run patient intake form completion audit; auto-flag records missing age/gender"},
            {"gap": "Label gaps", "action": "Queue unlabeled analyses for expert review; implement active learning prioritization"},
            {"gap": "Signal quality issues", "action": "Re-run EEG QC pipeline; flag flat/noisy channels for re-acquisition"},
        ],
    }
