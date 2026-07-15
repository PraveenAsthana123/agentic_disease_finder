"""
Paperclip Business Orchestration Dashboard
Business workflow orchestration analytics: patient intake, referral processing,
report generation, appointment scheduling, compliance review, data export,
EEG upload pipeline, medication reconciliation, insurance pre-auth, discharge planning.
Real business_workflows table in clinical.db.
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB)


def _rows(cur, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def overview():
    """Workflow counts, completion rate, category/status/priority/trigger distribution, daily volume."""
    con = _conn()
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) FROM business_workflows")
    total = cur.fetchone()[0]

    status_counts = {}
    for row in _rows(cur, "SELECT status, COUNT(*) as cnt FROM business_workflows GROUP BY status"):
        status_counts[row["status"]] = row["cnt"]

    active = status_counts.get("active", 0)
    completed = status_counts.get("completed", 0)
    failed = status_counts.get("failed", 0)
    paused = status_counts.get("paused", 0)
    pending = status_counts.get("pending", 0)

    completion_rate = round(completed / total * 100, 1) if total else 0.0

    cur.execute("SELECT AVG(duration_seconds) FROM business_workflows WHERE status='completed' AND duration_seconds IS NOT NULL")
    avg_dur = cur.fetchone()[0]
    avg_duration_seconds = round(avg_dur, 1) if avg_dur else 0

    category_distribution = _rows(cur, "SELECT category, COUNT(*) as count FROM business_workflows GROUP BY category ORDER BY count DESC")
    status_distribution = _rows(cur, "SELECT status, COUNT(*) as count FROM business_workflows GROUP BY status ORDER BY count DESC")
    priority_distribution = _rows(cur, "SELECT priority, COUNT(*) as count FROM business_workflows GROUP BY priority ORDER BY count DESC")
    trigger_distribution = _rows(cur, "SELECT trigger_type, COUNT(*) as count FROM business_workflows GROUP BY trigger_type ORDER BY count DESC")

    # Daily volume last 14 days
    cutoff = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")
    daily_volume = _rows(
        cur,
        "SELECT DATE(created_at) as date, COUNT(*) as count "
        "FROM business_workflows WHERE DATE(created_at) >= ? "
        "GROUP BY DATE(created_at) ORDER BY date",
        (cutoff,),
    )

    # Top failing workflow types
    top_failing = _rows(
        cur,
        "SELECT workflow_name, COUNT(*) as failure_count "
        "FROM business_workflows WHERE status='failed' "
        "GROUP BY workflow_name ORDER BY failure_count DESC LIMIT 5",
    )

    # Avg steps completion pct for active workflows
    cur.execute(
        "SELECT AVG(CAST(steps_completed AS REAL) / CAST(steps_total AS REAL) * 100) "
        "FROM business_workflows WHERE status='active' AND steps_total > 0"
    )
    avg_steps = cur.fetchone()[0]
    avg_steps_completion_pct = round(avg_steps, 1) if avg_steps else 0.0

    con.close()
    return {
        "total_workflows": total,
        "active": active,
        "completed": completed,
        "failed": failed,
        "paused": paused,
        "pending": pending,
        "completion_rate": completion_rate,
        "avg_duration_seconds": avg_duration_seconds,
        "category_distribution": category_distribution,
        "status_distribution": status_distribution,
        "priority_distribution": priority_distribution,
        "trigger_distribution": trigger_distribution,
        "daily_volume": daily_volume,
        "top_failing": top_failing,
        "avg_steps_completion_pct": avg_steps_completion_pct,
    }


def breakdown():
    """Per-category detail, recent workflows, owner workload, stalled detection, per-workflow-type stats."""
    con = _conn()
    cur = con.cursor()

    # Per-category detail
    all_rows = _rows(
        cur,
        "SELECT workflow_id, workflow_name, category, status, priority, trigger_type, "
        "steps_total, steps_completed, owner, patient_id, created_at, completed_at, "
        "duration_seconds, error_message, retry_count "
        "FROM business_workflows ORDER BY created_at DESC",
    )

    per_category = {}
    for r in all_rows:
        cat = r["category"]
        per_category.setdefault(cat, []).append({
            "workflow_id": r["workflow_id"],
            "workflow_name": r["workflow_name"],
            "status": r["status"],
            "priority": r["priority"],
            "steps_progress": f"{r['steps_completed']}/{r['steps_total']}",
            "duration_seconds": r["duration_seconds"],
            "owner": r["owner"],
        })

    # Recent 20 workflows
    recent_workflows = all_rows[:20]

    # Owner workload
    owner_workload = _rows(
        cur,
        "SELECT owner, "
        "SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active_count, "
        "SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed_count, "
        "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed_count "
        "FROM business_workflows GROUP BY owner ORDER BY active_count DESC",
    )

    # Stalled workflows: active, <50% steps, created >3 days ago
    cutoff_3d = (datetime.utcnow() - timedelta(days=3)).isoformat()
    stalled_workflows = _rows(
        cur,
        "SELECT workflow_id, workflow_name, category, steps_completed, steps_total, "
        "owner, created_at "
        "FROM business_workflows "
        "WHERE status='active' "
        "AND CAST(steps_completed AS REAL) / CAST(steps_total AS REAL) < 0.5 "
        "AND created_at < ? "
        "ORDER BY created_at",
        (cutoff_3d,),
    )

    # Per workflow type stats
    per_workflow_type = _rows(
        cur,
        "SELECT workflow_name, "
        "COUNT(*) as total_count, "
        "AVG(CASE WHEN duration_seconds IS NOT NULL THEN duration_seconds END) as avg_duration, "
        "ROUND(SUM(CASE WHEN status='completed' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as success_rate "
        "FROM business_workflows GROUP BY workflow_name ORDER BY total_count DESC",
    )
    # Round avg_duration
    for row in per_workflow_type:
        if row.get("avg_duration") is not None:
            row["avg_duration"] = round(row["avg_duration"], 1)

    con.close()
    return {
        "per_category": per_category,
        "recent_workflows": recent_workflows,
        "owner_workload": owner_workload,
        "stalled_workflows": stalled_workflows,
        "per_workflow_type": per_workflow_type,
    }


def definitions():
    """Statuses, categories, triggers, priorities, orchestration glossary."""
    return {
        "workflow_statuses": {
            "active": "Workflow is currently executing steps",
            "paused": "Workflow execution temporarily halted by operator or system",
            "completed": "All steps finished successfully",
            "failed": "Workflow encountered an unrecoverable error and stopped",
            "pending": "Workflow created but not yet started (queued)",
        },
        "categories": {
            "Clinical": "Patient-facing workflows: intake, medication reconciliation, discharge",
            "Administrative": "Back-office workflows: scheduling, referrals, insurance pre-auth",
            "Technical": "System workflows: data export, EEG upload, report generation",
            "Compliance": "Regulatory workflows: HIPAA review, audit trails, consent verification",
        },
        "trigger_types": {
            "manual": "Initiated by a human operator (clinician, admin, scheduler)",
            "scheduled": "Triggered on a recurring schedule (cron-based)",
            "event-driven": "Triggered by a system event (new patient, EEG upload, alert)",
            "api": "Triggered by an external API call or integration webhook",
        },
        "priority_levels": {
            "critical": "Must complete within 1 hour — patient safety or regulatory deadline",
            "high": "Must complete within same business day",
            "medium": "Standard priority — complete within 48 hours",
            "low": "Best-effort — no strict deadline",
        },
        "glossary": {
            "workflow": "A sequence of automated and manual steps that accomplish a business objective",
            "orchestration": "Coordination of multiple services and actions into a coherent workflow",
            "step": "A single atomic unit of work within a workflow",
            "stalled workflow": "An active workflow that has not progressed past 50% after 3+ days",
            "retry": "Automatic re-execution of a failed step or workflow",
            "trigger": "The event or condition that initiates workflow execution",
            "SLA": "Service Level Agreement — the maximum acceptable time for workflow completion",
            "idempotency": "Property ensuring re-running a workflow produces the same result without duplication",
            "compensation": "Rollback action taken when a workflow step fails after partial completion",
            "circuit breaker": "Pattern that stops retrying a failing step after a threshold, preventing cascade failures",
            "dead letter queue": "Holding area for workflows that fail repeatedly and need manual intervention",
            "backpressure": "Mechanism to slow intake when downstream services are overloaded",
        },
    }


if __name__ == "__main__":
    print(json.dumps({"overview": overview(), "breakdown": breakdown(), "definitions": definitions()}, indent=2, default=str))
