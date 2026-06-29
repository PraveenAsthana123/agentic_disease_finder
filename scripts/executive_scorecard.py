#!/usr/bin/env python3
"""
Executive Scorecard — aggregated top-level KPIs from clinical.db
================================================================

Provides a single-screen executive view that aggregates:
  - Patient census (total, by department)
  - Clinical activity (assessments, seizure events, medications)
  - AI operations (transaction volume, component breakdown)
  - Quality indicators (assessment completion, alert rates)
  - Trend data (operations over time)

Functions:
  - scorecard_overview   — all KPIs in one payload
  - scorecard_breakdown  — per-department and per-instrument drill-down
  - scorecard_definitions — metric definitions for tooltip overlays
"""

import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clinical_db as cdb

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "clinical.db"


def _query(sql: str, params: tuple = ()) -> list:
    """Run a read-only query against clinical.db."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _scalar(sql: str, params: tuple = ()):
    """Return a single scalar value."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def scorecard_overview() -> Dict[str, Any]:
    """All executive KPIs in one payload."""
    if not DB_PATH.exists():
        return {"available": False, "note": "clinical.db not found"}

    total_patients = _scalar("SELECT COUNT(*) FROM patients")
    total_assessments = _scalar("SELECT COUNT(*) FROM assessments")
    total_seizures = _scalar("SELECT COUNT(*) FROM seizure_diary")
    total_medications = _scalar("SELECT COUNT(*) FROM medications")
    total_transactions = _scalar("SELECT COUNT(*) FROM transaction_log")
    total_expert_reviews = _scalar("SELECT COUNT(*) FROM expert_reviews")
    total_hitl = _scalar("SELECT COUNT(*) FROM hitl_reviews")

    # Alert rate: assessments where alert = 1
    alert_count = _scalar("SELECT COUNT(*) FROM assessments WHERE alert = 1")
    alert_rate = round(alert_count / max(total_assessments, 1) * 100, 1)

    # Seizure severity breakdown
    severity_rows = _query(
        "SELECT severity, COUNT(*) as cnt FROM seizure_diary GROUP BY severity"
    )
    severity_dist = {r["severity"]: r["cnt"] for r in severity_rows}

    # Department census
    dept_rows = _query(
        "SELECT COALESCE(NULLIF(department,''), 'Unassigned') as dept, "
        "COUNT(*) as cnt FROM patients GROUP BY dept ORDER BY cnt DESC"
    )

    # Assessment instruments used
    instrument_rows = _query(
        "SELECT instrument, COUNT(*) as cnt FROM assessments "
        "GROUP BY instrument ORDER BY cnt DESC"
    )

    # Patients per unique instrument (breadth of assessment)
    instruments_used = len(instrument_rows)
    avg_assessments_per_patient = round(total_assessments / max(total_patients, 1), 1)

    # Transaction trend: last 7 days
    daily_trend = _query(
        "SELECT DATE(ts_utc) as day, COUNT(*) as ops "
        "FROM transaction_log "
        "WHERE ts_utc >= DATE('now', '-7 days') "
        "GROUP BY day ORDER BY day"
    )

    # Top transaction components
    component_rows = _query(
        "SELECT component, COUNT(*) as ops FROM transaction_log "
        "GROUP BY component ORDER BY ops DESC LIMIT 8"
    )

    return {
        "available": True,
        "summary": {
            "total_patients": total_patients,
            "total_assessments": total_assessments,
            "total_seizure_events": total_seizures,
            "total_medications": total_medications,
            "total_ai_operations": total_transactions,
            "total_expert_reviews": total_expert_reviews,
            "total_hitl_reviews": total_hitl,
            "alert_rate_pct": alert_rate,
            "alerts_triggered": alert_count,
            "instruments_used": instruments_used,
            "avg_assessments_per_patient": avg_assessments_per_patient,
        },
        "severity_distribution": severity_dist,
        "department_census": [
            {"department": r["dept"], "count": r["cnt"]} for r in dept_rows
        ],
        "instrument_usage": [
            {"instrument": r["instrument"], "count": r["cnt"]}
            for r in instrument_rows
        ],
        "daily_operations": [
            {"date": r["day"], "operations": r["ops"]} for r in daily_trend
        ],
        "top_components": [
            {"component": r["component"], "operations": r["ops"]}
            for r in component_rows
        ],
    }


def scorecard_breakdown() -> Dict[str, Any]:
    """Per-department and per-instrument drill-down."""
    if not DB_PATH.exists():
        return {"available": False}

    # Per-department: patients + assessments
    dept_detail = _query(
        "SELECT COALESCE(NULLIF(p.department,''), 'Unassigned') as dept, "
        "COUNT(DISTINCT p.patient_id) as patients, "
        "COUNT(a.id) as assessments "
        "FROM patients p LEFT JOIN assessments a ON p.patient_id = a.patient_id "
        "GROUP BY dept ORDER BY patients DESC"
    )

    # Per-instrument: score stats
    instrument_stats = _query(
        "SELECT instrument, COUNT(*) as total, "
        "ROUND(AVG(CAST(score AS REAL)), 1) as avg_score, "
        "ROUND(AVG(CAST(max_score AS REAL)), 1) as avg_max, "
        "SUM(CASE WHEN alert=1 THEN 1 ELSE 0 END) as alerts "
        "FROM assessments GROUP BY instrument ORDER BY total DESC"
    )

    # Seizure timeline (monthly)
    monthly_seizures = _query(
        "SELECT SUBSTR(event_date, 1, 7) as month, COUNT(*) as events, "
        "SUM(CASE WHEN severity='Severe' THEN 1 ELSE 0 END) as severe "
        "FROM seizure_diary GROUP BY month ORDER BY month"
    )

    return {
        "available": True,
        "department_detail": dept_detail,
        "instrument_stats": instrument_stats,
        "monthly_seizures": monthly_seizures,
    }


def scorecard_definitions() -> Dict[str, Any]:
    """Metric definitions for tooltip overlays."""
    return {
        "available": True,
        "definitions": [
            {
                "metric": "Total Patients",
                "description": "Number of patients registered in the clinical database",
                "source": "patients table",
            },
            {
                "metric": "Total Assessments",
                "description": "Completed clinical assessments across all instruments (PHQ-9, GAD-7, MMSE, etc.)",
                "source": "assessments table",
            },
            {
                "metric": "Alert Rate",
                "description": "Percentage of assessments that triggered a clinical alert (elevated scores requiring attention)",
                "source": "assessments.alert field",
            },
            {
                "metric": "Seizure Events",
                "description": "Total seizure diary entries logged by patients or clinicians",
                "source": "seizure_diary table",
            },
            {
                "metric": "AI Operations",
                "description": "Total transactions processed by the AI system (classifications, reports, assessments, etc.)",
                "source": "transaction_log table",
            },
            {
                "metric": "Expert Reviews",
                "description": "Human expert reviews completed on AI decisions (governance oversight)",
                "source": "expert_reviews table",
            },
            {
                "metric": "HITL Reviews",
                "description": "Human-in-the-loop reviews for high-risk decisions",
                "source": "hitl_reviews table",
            },
            {
                "metric": "Instruments Used",
                "description": "Number of distinct clinical assessment instruments deployed (PHQ-9, GAD-7, MOCA, BNT, WAB, etc.)",
                "source": "assessments.instrument distinct count",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(scorecard_overview(), indent=2, default=str))
