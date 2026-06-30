#!/usr/bin/env python3
"""
Executive AI Dashboard — AI-centric executive KPIs from clinical.db
===================================================================

Provides an executive-level view of AI adoption, utilization, and governance:
  - AI adoption metrics (operations per department, actor distribution)
  - AI component utilization (which AI components are most active)
  - AI oversight coverage (expert reviews + HITL vs total AI operations)
  - AI conversation engagement (chat volume, source distribution)
  - AI throughput trends (daily action volume by type)
  - Department-level AI penetration (patients touched by AI per department)

Functions:
  - executive_ai_overview    — top-level AI KPIs
  - executive_ai_breakdown   — per-department and per-component drill-down
  - executive_ai_definitions — metric definitions for tooltip overlays
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


def executive_ai_overview() -> Dict[str, Any]:
    """Top-level AI executive KPIs."""
    if not DB_PATH.exists():
        return {"available": False, "note": "clinical.db not found"}

    total_ai_ops = _scalar("SELECT COUNT(*) FROM transaction_log")
    total_patients = _scalar("SELECT COUNT(*) FROM patients")
    total_conversations = _scalar("SELECT COUNT(*) FROM conversation_log")
    total_expert_reviews = _scalar("SELECT COUNT(*) FROM expert_reviews")
    total_hitl = _scalar("SELECT COUNT(*) FROM hitl_reviews")
    total_assessments = _scalar("SELECT COUNT(*) FROM assessments")
    total_feedback = _scalar("SELECT COUNT(*) FROM feedback")

    # AI oversight rate: (expert + HITL) / total ops
    oversight_total = total_expert_reviews + total_hitl
    oversight_rate = round(oversight_total / max(total_ai_ops, 1) * 100, 1)

    # Automation rate: system-actor ops / total ops
    system_ops = _scalar(
        "SELECT COUNT(*) FROM transaction_log WHERE actor = 'system'"
    )
    automation_rate = round(system_ops / max(total_ai_ops, 1) * 100, 1)

    # AI-touched patients (patients with at least one transaction)
    ai_touched = _scalar(
        "SELECT COUNT(DISTINCT patient_id) FROM transaction_log "
        "WHERE patient_id IS NOT NULL AND patient_id != ''"
    )
    ai_penetration = round(ai_touched / max(total_patients, 1) * 100, 1)

    # Actor distribution
    actor_rows = _query(
        "SELECT actor, COUNT(*) as ops FROM transaction_log "
        "GROUP BY actor ORDER BY ops DESC"
    )

    # Top AI components by volume
    component_rows = _query(
        "SELECT component, COUNT(*) as ops FROM transaction_log "
        "GROUP BY component ORDER BY ops DESC LIMIT 10"
    )

    # Action type distribution
    action_rows = _query(
        "SELECT action, COUNT(*) as ops FROM transaction_log "
        "GROUP BY action ORDER BY ops DESC LIMIT 8"
    )

    # Daily AI throughput (last 14 days)
    daily_throughput = _query(
        "SELECT DATE(ts_utc) as day, COUNT(*) as ops, "
        "COUNT(DISTINCT component) as components_active "
        "FROM transaction_log "
        "WHERE ts_utc >= DATE('now', '-14 days') "
        "GROUP BY day ORDER BY day"
    )

    # Department AI utilization
    dept_ai = _query(
        "SELECT COALESCE(NULLIF(p.department,''), 'Unassigned') as dept, "
        "COUNT(DISTINCT p.patient_id) as patients, "
        "COUNT(t.id) as ai_ops "
        "FROM patients p LEFT JOIN transaction_log t "
        "ON p.patient_id = t.patient_id "
        "GROUP BY dept ORDER BY ai_ops DESC"
    )

    # Conversation role distribution
    conv_sources = _query(
        "SELECT role, COUNT(*) as count FROM conversation_log "
        "GROUP BY role ORDER BY count DESC"
    )

    return {
        "available": True,
        "summary": {
            "total_ai_operations": total_ai_ops,
            "total_patients": total_patients,
            "ai_touched_patients": ai_touched,
            "ai_penetration_pct": ai_penetration,
            "total_conversations": total_conversations,
            "total_assessments": total_assessments,
            "expert_reviews": total_expert_reviews,
            "hitl_reviews": total_hitl,
            "oversight_rate_pct": oversight_rate,
            "automation_rate_pct": automation_rate,
            "system_ops": system_ops,
            "human_ops": total_ai_ops - system_ops,
            "feedback_count": total_feedback,
        },
        "actor_distribution": [
            {"actor": r["actor"], "operations": r["ops"]} for r in actor_rows
        ],
        "top_components": [
            {"component": r["component"], "operations": r["ops"]}
            for r in component_rows
        ],
        "action_distribution": [
            {"action": r["action"], "operations": r["ops"]}
            for r in action_rows
        ],
        "daily_throughput": [
            {"date": r["day"], "operations": r["ops"],
             "components_active": r["components_active"]}
            for r in daily_throughput
        ],
        "department_ai_utilization": [
            {"department": r["dept"], "patients": r["patients"],
             "ai_operations": r["ai_ops"]}
            for r in dept_ai
        ],
        "conversation_roles": [
            {"role": r["role"], "count": r["count"]}
            for r in conv_sources
        ],
    }


def executive_ai_breakdown() -> Dict[str, Any]:
    """Per-component and per-department AI drill-down."""
    if not DB_PATH.exists():
        return {"available": False}

    # Component detail: ops, unique patients, date range
    component_detail = _query(
        "SELECT component, COUNT(*) as ops, "
        "COUNT(DISTINCT patient_id) as patients_touched, "
        "MIN(ts_utc) as first_seen, MAX(ts_utc) as last_seen "
        "FROM transaction_log GROUP BY component ORDER BY ops DESC"
    )

    # Department + component cross-tab
    dept_component = _query(
        "SELECT COALESCE(NULLIF(p.department,''), 'Unassigned') as dept, "
        "t.component, COUNT(*) as ops "
        "FROM transaction_log t "
        "LEFT JOIN patients p ON t.patient_id = p.patient_id "
        "GROUP BY dept, t.component ORDER BY ops DESC LIMIT 30"
    )

    # Weekly AI volume (last 8 weeks)
    weekly_volume = _query(
        "SELECT strftime('%Y-W%W', ts_utc) as week, "
        "COUNT(*) as ops, "
        "COUNT(DISTINCT component) as components "
        "FROM transaction_log "
        "WHERE ts_utc >= DATE('now', '-56 days') "
        "GROUP BY week ORDER BY week"
    )

    # Oversight detail
    expert_detail = _query(
        "SELECT * FROM expert_reviews ORDER BY created_at DESC LIMIT 10"
    )
    hitl_detail = _query(
        "SELECT * FROM hitl_reviews ORDER BY created_at DESC LIMIT 10"
    )

    return {
        "available": True,
        "component_detail": component_detail,
        "department_component_cross": dept_component,
        "weekly_volume": weekly_volume,
        "recent_expert_reviews": expert_detail,
        "recent_hitl_reviews": hitl_detail,
    }


def executive_ai_definitions() -> Dict[str, Any]:
    """Metric definitions for Executive AI Dashboard."""
    return {
        "available": True,
        "definitions": [
            {
                "metric": "Total AI Operations",
                "description": "Total transactions processed by AI components (assessments, analyses, ingestion, chat, etc.)",
                "source": "transaction_log table",
            },
            {
                "metric": "AI Penetration",
                "description": "Percentage of registered patients who have been touched by at least one AI operation",
                "source": "transaction_log.patient_id vs patients",
            },
            {
                "metric": "Automation Rate",
                "description": "Percentage of AI operations executed by the system actor (fully automated, no human trigger)",
                "source": "transaction_log WHERE actor='system'",
            },
            {
                "metric": "Oversight Rate",
                "description": "Ratio of human oversight actions (expert reviews + HITL reviews) to total AI operations",
                "source": "expert_reviews + hitl_reviews vs transaction_log",
            },
            {
                "metric": "AI Component Utilization",
                "description": "Distribution of AI operations across system components (assessment, CV pipeline, chat, etc.)",
                "source": "transaction_log.component",
            },
            {
                "metric": "Actor Distribution",
                "description": "Breakdown of who initiated AI operations — system (automated), clinicians, compliance agents",
                "source": "transaction_log.actor",
            },
            {
                "metric": "Daily AI Throughput",
                "description": "Number of AI operations per day with count of distinct active components",
                "source": "transaction_log grouped by DATE(ts_utc)",
            },
            {
                "metric": "Department AI Utilization",
                "description": "AI operations per department, showing which departments use AI most heavily",
                "source": "transaction_log joined with patients.department",
            },
            {
                "metric": "Conversation Volume",
                "description": "Total AI-powered conversations (patient chat, team chat, consultant interactions)",
                "source": "conversation_log table",
            },
            {
                "metric": "Feedback Count",
                "description": "User feedback submissions on AI outputs — a proxy for engagement and satisfaction tracking",
                "source": "feedback table",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(executive_ai_overview(), indent=2, default=str))
