#!/usr/bin/env python3
"""
Integration Dashboard — multi-format ingest analytics from clinical.db
======================================================================

Provides visibility into data ingestion across formats:
  - Upload volume & format distribution (EDF, CSV, BDF, etc.)
  - Per-patient upload coverage
  - Ingest pipeline activity (eeg_upload + patient_master ingest transactions)
  - Daily ingest trends
  - Department/actor distribution for uploads
  - Format-specific success metrics

Functions:
  - integration_overview    — top-level ingest KPIs
  - integration_breakdown   — per-format, per-patient, per-day drill-down
  - integration_definitions — metric definitions for tooltip overlays
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "clinical.db"


def _query(sql, params=()):
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _scalar(sql, params=()):
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def integration_overview() -> Dict[str, Any]:
    """Top-level multi-format ingest KPIs."""
    if not DB_PATH.exists():
        return {"available": False, "note": "clinical.db not found"}

    total_uploads = _scalar("SELECT COUNT(*) FROM uploads")
    unique_patients = _scalar("SELECT COUNT(DISTINCT patient_id) FROM uploads")
    unique_files = _scalar("SELECT COUNT(DISTINCT file_name) FROM uploads")
    total_patients = _scalar("SELECT COUNT(*) FROM patients")

    # Format distribution from uploads
    format_rows = _query(
        "SELECT CASE "
        "  WHEN file_name LIKE '%.edf' THEN 'EDF' "
        "  WHEN file_name LIKE '%.csv' THEN 'CSV' "
        "  WHEN file_name LIKE '%.bdf' THEN 'BDF' "
        "  WHEN file_name LIKE '%.set' THEN 'SET' "
        "  WHEN file_name LIKE '%.fif' THEN 'FIF' "
        "  ELSE 'Other' "
        "END AS format, COUNT(*) AS count "
        "FROM uploads GROUP BY format ORDER BY count DESC"
    )

    # Ingest transactions (eeg_upload analyze + patient_master ingest)
    ingest_txns = _scalar(
        "SELECT COUNT(*) FROM transaction_log "
        "WHERE (component = 'eeg_upload' AND action = 'analyze') "
        "OR (component = 'patient_master' AND action = 'ingest')"
    )

    # Unique ingest actors
    ingest_actors = _query(
        "SELECT actor, COUNT(*) AS count FROM transaction_log "
        "WHERE (component = 'eeg_upload' AND action = 'analyze') "
        "OR (component = 'patient_master' AND action = 'ingest') "
        "GROUP BY actor ORDER BY count DESC"
    )

    # Patient upload coverage
    upload_coverage = round(unique_patients / max(total_patients, 1) * 100, 1)

    # Department distribution
    dept_rows = _query(
        "SELECT CASE WHEN department = '' OR department IS NULL THEN 'Unassigned' "
        "ELSE department END AS dept, COUNT(*) AS count "
        "FROM uploads GROUP BY dept ORDER BY count DESC"
    )

    # Date range
    first_upload = _scalar("SELECT MIN(created_at) FROM uploads") or "N/A"
    last_upload = _scalar("SELECT MAX(created_at) FROM uploads") or "N/A"

    # Disease distribution from uploads
    disease_rows = _query(
        "SELECT COALESCE(disease, 'unknown') AS disease, COUNT(*) AS count "
        "FROM uploads GROUP BY disease ORDER BY count DESC"
    )

    return {
        "available": True,
        "total_uploads": total_uploads,
        "unique_patients": unique_patients,
        "unique_files": unique_files,
        "total_patients": total_patients,
        "upload_coverage_pct": upload_coverage,
        "format_distribution": format_rows,
        "ingest_transactions": ingest_txns,
        "ingest_actors": ingest_actors,
        "department_distribution": dept_rows,
        "disease_distribution": disease_rows,
        "first_upload": first_upload,
        "last_upload": last_upload,
    }


def integration_breakdown() -> Dict[str, Any]:
    """Per-format, per-patient, per-day drill-down."""
    if not DB_PATH.exists():
        return {"available": False, "note": "clinical.db not found"}

    # Per-patient upload counts
    per_patient = _query(
        "SELECT patient_id, COUNT(*) AS uploads, "
        "COUNT(DISTINCT file_name) AS unique_files, "
        "MIN(created_at) AS first_upload, MAX(created_at) AS last_upload "
        "FROM uploads GROUP BY patient_id ORDER BY uploads DESC"
    )

    # Daily upload trend
    daily_trend = _query(
        "SELECT DATE(created_at) AS day, COUNT(*) AS uploads "
        "FROM uploads GROUP BY day ORDER BY day"
    )

    # Daily ingest transaction trend
    daily_ingest = _query(
        "SELECT DATE(ts_local) AS day, COUNT(*) AS transactions "
        "FROM transaction_log "
        "WHERE (component = 'eeg_upload' AND action = 'analyze') "
        "OR (component = 'patient_master' AND action = 'ingest') "
        "GROUP BY day ORDER BY day"
    )

    # Per-format, per-patient matrix
    format_patient = _query(
        "SELECT patient_id, "
        "  CASE "
        "    WHEN file_name LIKE '%.edf' THEN 'EDF' "
        "    WHEN file_name LIKE '%.csv' THEN 'CSV' "
        "    WHEN file_name LIKE '%.bdf' THEN 'BDF' "
        "    ELSE 'Other' "
        "  END AS format, "
        "  COUNT(*) AS count "
        "FROM uploads GROUP BY patient_id, format ORDER BY patient_id, count DESC"
    )

    # Ingest pipeline detail (recent 50 transactions)
    recent_ingest = _query(
        "SELECT patient_id, component, action, actor, detail, ts_local "
        "FROM transaction_log "
        "WHERE (component = 'eeg_upload' AND action = 'analyze') "
        "OR (component = 'patient_master' AND action = 'ingest') "
        "ORDER BY ts_utc DESC LIMIT 50"
    )

    # Hourly upload pattern
    hourly = _query(
        "SELECT CAST(strftime('%H', created_at) AS INTEGER) AS hour, COUNT(*) AS count "
        "FROM uploads GROUP BY hour ORDER BY hour"
    )

    # File name frequency (top repeated files)
    top_files = _query(
        "SELECT file_name, COUNT(*) AS uploads, "
        "COUNT(DISTINCT patient_id) AS patients "
        "FROM uploads GROUP BY file_name ORDER BY uploads DESC LIMIT 15"
    )

    return {
        "available": True,
        "per_patient": per_patient,
        "daily_upload_trend": daily_trend,
        "daily_ingest_trend": daily_ingest,
        "format_patient_matrix": format_patient,
        "recent_ingest_events": recent_ingest,
        "hourly_pattern": hourly,
        "top_files": top_files,
    }


def integration_definitions() -> Dict[str, Any]:
    """Metric definitions for tooltip overlays."""
    return {
        "definitions": {
            "total_uploads": "Total number of file upload records in the uploads table.",
            "unique_patients": "Count of distinct patient IDs that have at least one upload.",
            "unique_files": "Count of distinct file names across all uploads.",
            "upload_coverage_pct": "Percentage of registered patients with at least one upload. (unique uploading patients / total patients) * 100.",
            "format_distribution": "Breakdown of uploads by file extension (EDF, CSV, BDF, etc.).",
            "ingest_transactions": "Count of transaction_log entries where component=eeg_upload+action=analyze or component=patient_master+action=ingest.",
            "ingest_actors": "Who triggered ingest operations (system, neurologist, etc.).",
            "department_distribution": "Which departments uploaded data.",
            "disease_distribution": "Disease label distribution across uploads.",
            "per_patient": "Per-patient upload count, unique files, first/last upload timestamps.",
            "daily_upload_trend": "Number of uploads per calendar day.",
            "daily_ingest_trend": "Number of ingest transactions per calendar day.",
            "format_patient_matrix": "Which patients uploaded which formats.",
            "recent_ingest_events": "Most recent 50 ingest pipeline transactions with details.",
            "hourly_pattern": "Upload volume by hour of day (0-23).",
            "top_files": "Most frequently uploaded file names and how many patients uploaded each.",
        }
    }


if __name__ == "__main__":
    import json
    print("=== Overview ===")
    print(json.dumps(integration_overview(), indent=2, default=str))
    print("\n=== Breakdown ===")
    print(json.dumps(integration_breakdown(), indent=2, default=str))
