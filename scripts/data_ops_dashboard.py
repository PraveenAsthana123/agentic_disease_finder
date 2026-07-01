"""
DataOps Dashboard
Data pipeline monitoring: ingestion metrics, data quality scores, storage stats,
vector ingest status, pipeline lineage, and modality coverage.

Registry item: dataops (admin_module.ops_dashboards)
Purpose: ingestion, data quality, lineage, vector ingest
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")
_DQ_PATH = os.path.join(_BASE_DIR, "jobs", "reports", "data_quality_latest.json")
_VECTOR_LOG = os.path.join(_BASE_DIR, "jobs", "logs", "vector_ingest.log")
_VECTOR_DB = os.path.join(_BASE_DIR, "data", "vector_db")


def _db():
    """Return a read-only connection to clinical.db."""
    conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default or {}


def _dir_size_mb(path):
    """Total size in MB of all files under *path*."""
    total = 0
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
    elif os.path.isfile(path):
        total = os.path.getsize(path)
    return round(total / (1024 * 1024), 2)


def _file_size_mb(path):
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    except OSError:
        return 0.0


def _table_count(conn, table):
    try:
        return conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    except Exception:
        return 0


# ── overview() ────────────────────────────────────────────────────────

def overview():
    dq = _load_json(_DQ_PATH)
    conn = _db()

    # Row counts
    patients   = _table_count(conn, "patients")
    uploads    = _table_count(conn, "uploads")
    analyses   = _table_count(conn, "analyses")
    assessments = _table_count(conn, "assessments")
    txn_total  = _table_count(conn, "transaction_log")

    # Ingestion pipeline counts from transaction_log
    rows = conn.execute(
        "SELECT component, action, COUNT(*) as cnt "
        "FROM transaction_log GROUP BY component, action ORDER BY cnt DESC"
    ).fetchall()
    pipeline_activity = [{"component": r["component"], "action": r["action"],
                          "count": r["cnt"]} for r in rows]

    # Signal quality from analyses
    sq_rows = conn.execute(
        "SELECT signal_quality, COUNT(*) as cnt FROM analyses GROUP BY signal_quality"
    ).fetchall()
    signal_quality = {r["signal_quality"]: r["cnt"] for r in sq_rows}
    total_analyses = sum(signal_quality.values())
    good_pct = round(signal_quality.get("Good", 0) / max(total_analyses, 1) * 100, 1)

    # Storage stats
    db_size_mb = _file_size_mb(_DB_PATH)
    vector_size_mb = _dir_size_mb(_VECTOR_DB)

    # Data quality dimensions
    quality_dims = dq.get("quality_dimensions", {})
    scored_dims = {k: v["score"] for k, v in quality_dims.items()
                   if v.get("score") is not None}
    avg_quality = round(sum(scored_dims.values()) / max(len(scored_dims), 1), 1)

    # Modality coverage
    modality_coverage = dq.get("modality_coverage_pct", {})
    avg_coverage = round(sum(modality_coverage.values()) / max(len(modality_coverage), 1), 1)

    # Vector ingest status
    vector_status = _parse_vector_log()

    conn.close()

    kpis = {
        "total_patients":   patients,
        "total_uploads":    uploads,
        "total_analyses":   analyses,
        "total_txn_events": txn_total,
        "ai_readiness":     dq.get("ai_readiness_score", 0),
        "ai_grade":         dq.get("ai_readiness_grade", "N/A"),
        "avg_quality":      avg_quality,
        "signal_good_pct":  good_pct,
        "avg_coverage_pct": avg_coverage,
        "db_size_mb":       db_size_mb,
        "vector_size_mb":   vector_size_mb,
    }

    return {
        "kpis": kpis,
        "signal_quality_distribution": signal_quality,
        "modality_coverage": modality_coverage,
        "quality_dimensions_summary": scored_dims,
        "pipeline_top5": pipeline_activity[:5],
        "vector_ingest": vector_status,
    }


# ── breakdown() ───────────────────────────────────────────────────────

def breakdown():
    dq = _load_json(_DQ_PATH)
    conn = _db()

    # Full pipeline activity
    rows = conn.execute(
        "SELECT component, action, COUNT(*) as cnt, "
        "MIN(ts_utc) as first_ts, MAX(ts_utc) as last_ts "
        "FROM transaction_log GROUP BY component, action ORDER BY cnt DESC"
    ).fetchall()
    pipeline_activity = [{
        "component": r["component"], "action": r["action"],
        "count": r["cnt"], "first_seen": r["first_ts"], "last_seen": r["last_ts"],
    } for r in rows]

    # Daily ingestion volume (last 14 days)
    daily_rows = conn.execute(
        "SELECT DATE(ts_utc) as day, COUNT(*) as cnt "
        "FROM transaction_log GROUP BY DATE(ts_utc) ORDER BY day"
    ).fetchall()
    daily_volume = [{"date": r["day"], "count": r["cnt"]} for r in daily_rows]

    # Component-level counts (ingestion components only)
    ingestion_components = ["eeg_upload", "patient_master", "cv_pipeline",
                            "video_frames", "assessment", "seizure_diary",
                            "medications"]
    ingestion_rows = conn.execute(
        "SELECT component, COUNT(*) as cnt FROM transaction_log "
        "WHERE component IN ({}) GROUP BY component ORDER BY cnt DESC"
        .format(",".join(f"'{c}'" for c in ingestion_components))
    ).fetchall()
    ingestion_breakdown = [{"component": r["component"], "count": r["cnt"]}
                           for r in ingestion_rows]

    # Table row counts (storage inventory)
    tables = ["patients", "uploads", "analyses", "assessments",
              "seizure_diary", "medications", "mri_findings",
              "eeg_acquisition", "channel_quality", "transaction_log",
              "clinical_decisions", "finops_costs"]
    storage_inventory = []
    for t in tables:
        cnt = _table_count(conn, t)
        if cnt > 0:
            storage_inventory.append({"table": t, "rows": cnt})
    storage_inventory.sort(key=lambda x: -x["rows"])

    # Data quality full dimensions
    quality_dimensions = []
    for dim_name, dim_val in dq.get("quality_dimensions", {}).items():
        quality_dimensions.append({
            "dimension": dim_name,
            "score": dim_val.get("score"),
            "basis": dim_val.get("basis", ""),
            "measured": dim_val.get("real", False),
        })

    # Missing matrix
    missing_matrix = dq.get("missing_matrix", [])

    # Data lineage
    data_lineage = dq.get("data_lineage", [])

    # AI readiness components
    ai_components = dq.get("ai_readiness_components", {})

    conn.close()

    return {
        "pipeline_activity":   pipeline_activity,
        "daily_volume":        daily_volume,
        "ingestion_breakdown": ingestion_breakdown,
        "storage_inventory":   storage_inventory,
        "quality_dimensions":  quality_dimensions,
        "missing_matrix":      missing_matrix,
        "data_lineage":        data_lineage,
        "ai_readiness_components": ai_components,
        "dq_run_at":           dq.get("run_at", "unknown"),
    }


# ── definitions() ─────────────────────────────────────────────────────

def definitions():
    return {
        "sections": [
            {
                "title": "Data Pipelines",
                "items": [
                    {"term": "EEG Upload", "definition": "Raw EDF/CSV files uploaded per patient, parsed and validated before analysis."},
                    {"term": "CV Pipeline", "definition": "Computer-vision pipeline that extracts frames from video recordings (hourly cron)."},
                    {"term": "Video Frames", "definition": "Individual frames extracted from patient video recordings for movement analysis."},
                    {"term": "Patient Master", "definition": "Canonical patient demographic ingest (age, gender, disease, department)."},
                    {"term": "Assessment", "definition": "Clinical assessment forms (PHQ-9, GAD-7, MMSE, etc.) submitted per patient."},
                    {"term": "Seizure Diary", "definition": "Patient-reported seizure events with type, duration, and triggers."},
                    {"term": "Vector Ingest", "definition": "Twice-daily (07:00/19:00) embedding of clinical records into ChromaDB for RAG retrieval."},
                ],
            },
            {
                "title": "Data Quality Dimensions (ISO 25012)",
                "items": [
                    {"term": "Completeness", "definition": "Mean modality coverage across 5 clinical data modalities (EEG, Assessment, Seizure diary, MRI, Medication)."},
                    {"term": "Uniqueness", "definition": "Absence of duplicate patient_id records in the patients table."},
                    {"term": "Validity", "definition": "Proportion of patients with all required fields (age, gender, disease) populated."},
                    {"term": "Timeliness", "definition": "Freshness of analysis timestamps relative to upload time."},
                    {"term": "Consistency", "definition": "Cross-system agreement (requires EMR integration — not yet wired)."},
                    {"term": "Accuracy", "definition": "Value correctness against source-of-truth (requires external validation — not yet wired)."},
                ],
            },
            {
                "title": "AI Readiness Score",
                "items": [
                    {"term": "Formula", "definition": "Weighted average: completeness (20%) + uniqueness (20%) + validity (20%) + label_coverage (20%) + signal_quality (20%)."},
                    {"term": "Grades", "definition": "A (≥85): production-ready, B (70–84): usable with gaps, C (50–69): needs work, D (<50): not ready."},
                    {"term": "Signal Quality", "definition": "Percentage of EEG analyses graded 'Good' by AutoReject/PyPREP QC pipeline."},
                    {"term": "Label Coverage", "definition": "Percentage of patients with a confirmed disease classification label."},
                ],
            },
            {
                "title": "Storage",
                "items": [
                    {"term": "clinical.db", "definition": "SQLite database holding all structured clinical data (patients, analyses, assessments, transaction log)."},
                    {"term": "Vector DB (ChromaDB)", "definition": "Embedding store at data/vector_db/ used by RAG pipeline for semantic search over clinical records."},
                    {"term": "EDF/Model Files", "definition": "Raw EEG recordings (EDF format) and trained model artifacts stored in data/ directory."},
                ],
            },
            {
                "title": "Data Lineage Steps",
                "items": [
                    {"term": "Step 1: Raw Upload", "definition": "EDF/CSV files uploaded via /api/upload endpoint."},
                    {"term": "Step 2: Parse + Validate", "definition": "File format validation, header parsing, channel mapping."},
                    {"term": "Step 3: Signal QC", "definition": "Automated quality check via AutoReject/PyPREP for artifact detection."},
                    {"term": "Step 4: Feature Extraction", "definition": "47 features extracted (spectral power, asymmetry, connectivity, entropy)."},
                    {"term": "Step 5: Model Classify", "definition": "Disease classification via trained ML model with scaled features."},
                    {"term": "Step 6: SHAP Explanation", "definition": "Per-prediction SHAP values generated for explainability."},
                    {"term": "Step 7: Decision Audit", "definition": "Human-in-the-loop oversight review and approval."},
                    {"term": "Step 8: Vector Ingest", "definition": "Embedding into ChromaDB for RAG-powered clinical queries."},
                ],
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {"term": "Why DataOps matters", "definition": "EEG-based diagnosis depends on data quality — poor signal, missing modalities, or stale data can lead to misclassification. DataOps monitors the full pipeline from raw upload to AI prediction."},
                    {"term": "Regulatory", "definition": "IEC 62304 (medical device software) requires documented data lineage and quality controls for clinical AI systems."},
                ],
            },
        ],
    }


# ── Vector ingest log parser ──────────────────────────────────────────

def _parse_vector_log():
    """Parse the vector_ingest.log for the latest run info."""
    result = {
        "status": "unknown",
        "last_run": None,
        "records_embedded": 0,
        "records_failed": 0,
        "collection": "clinical",
        "db_path": "data/vector_db",
        "db_size_mb": _dir_size_mb(_VECTOR_DB),
    }
    try:
        with open(_VECTOR_LOG) as f:
            lines = f.readlines()
    except FileNotFoundError:
        result["status"] = "no_log_found"
        return result

    # Parse last run
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("[vector_ingest]"):
            ts = line.replace("[vector_ingest]", "").strip()
            result["last_run"] = ts
            result["status"] = "ok"
            break
        if "embedded" in line and "/" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "embedded" and i + 3 < len(parts):
                    try:
                        result["records_embedded"] = int(parts[i + 1])
                        result["records_failed"] = int(
                            parts[i + 5].rstrip(")") if i + 5 < len(parts) else "0"
                        )
                    except (ValueError, IndexError):
                        pass
    return result
