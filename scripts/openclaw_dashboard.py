"""
OpenClaw Execution Orchestration Dashboard
Agent execution orchestration analytics: task scheduling, agent workload,
token consumption, step completion, chained executions, failure tracking.
Real openclaw_executions table in clinical.db.
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
    """Execution counts, completion rate, token totals, agent/status/mode/trigger/priority distribution, daily volume."""
    con = _conn()
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) FROM openclaw_executions")
    total = cur.fetchone()[0]

    status_counts = {}
    for row in _rows(cur, "SELECT status, COUNT(*) as cnt FROM openclaw_executions GROUP BY status"):
        status_counts[row["status"]] = row["cnt"]

    completed = status_counts.get("completed", 0)
    running = status_counts.get("running", 0)
    failed = status_counts.get("failed", 0)
    queued = status_counts.get("queued", 0)
    cancelled = status_counts.get("cancelled", 0)

    completion_rate = round(completed / total * 100, 1) if total else 0.0

    cur.execute("SELECT AVG(duration_seconds) FROM openclaw_executions WHERE status='completed' AND duration_seconds IS NOT NULL")
    avg_dur = cur.fetchone()[0]
    avg_duration_seconds = round(avg_dur, 1) if avg_dur else 0

    cur.execute("SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0) FROM openclaw_executions")
    total_tokens = cur.fetchone()[0]

    agent_distribution = _rows(cur, "SELECT agent_name, COUNT(*) as count FROM openclaw_executions GROUP BY agent_name ORDER BY count DESC")
    status_distribution = _rows(cur, "SELECT status, COUNT(*) as count FROM openclaw_executions GROUP BY status ORDER BY count DESC")
    mode_distribution = _rows(cur, "SELECT execution_mode, COUNT(*) as count FROM openclaw_executions GROUP BY execution_mode ORDER BY count DESC")
    trigger_distribution = _rows(cur, "SELECT triggered_by, COUNT(*) as count FROM openclaw_executions GROUP BY triggered_by ORDER BY count DESC")
    priority_distribution = _rows(cur, "SELECT priority, COUNT(*) as count FROM openclaw_executions GROUP BY priority ORDER BY count DESC")

    # Daily volume last 21 days
    cutoff = (datetime.utcnow() - timedelta(days=21)).strftime("%Y-%m-%d")
    daily_volume = _rows(
        cur,
        "SELECT DATE(created_at) as date, COUNT(*) as count "
        "FROM openclaw_executions WHERE DATE(created_at) >= ? "
        "GROUP BY DATE(created_at) ORDER BY date",
        (cutoff,),
    )

    # Top failing agents
    top_failing_agents = _rows(
        cur,
        "SELECT agent_name, COUNT(*) as failure_count "
        "FROM openclaw_executions WHERE status='failed' "
        "GROUP BY agent_name ORDER BY failure_count DESC LIMIT 5",
    )

    # Avg steps completion pct for running executions
    cur.execute(
        "SELECT AVG(CAST(steps_completed AS REAL) / CAST(steps_total AS REAL) * 100) "
        "FROM openclaw_executions WHERE status='running' AND steps_total > 0"
    )
    avg_steps = cur.fetchone()[0]
    avg_steps_completion_pct = round(avg_steps, 1) if avg_steps else 0.0

    con.close()
    return {
        "total_executions": total,
        "completed": completed,
        "running": running,
        "failed": failed,
        "queued": queued,
        "cancelled": cancelled,
        "completion_rate": completion_rate,
        "avg_duration_seconds": avg_duration_seconds,
        "total_tokens": total_tokens,
        "agent_distribution": agent_distribution,
        "status_distribution": status_distribution,
        "mode_distribution": mode_distribution,
        "trigger_distribution": trigger_distribution,
        "priority_distribution": priority_distribution,
        "daily_volume": daily_volume,
        "top_failing_agents": top_failing_agents,
        "avg_steps_completion_pct": avg_steps_completion_pct,
    }


def breakdown():
    """Per-agent detail, recent executions, agent workload, chained executions, failed executions, per-agent stats."""
    con = _conn()
    cur = con.cursor()

    # All executions for per-agent grouping
    all_rows = _rows(
        cur,
        "SELECT execution_id, agent_id, agent_name, task_description, "
        "execution_mode, status, priority, input_tokens, output_tokens, "
        "duration_seconds, steps_total, steps_completed, parent_execution_id, "
        "triggered_by, patient_id, error_message, retry_count, "
        "created_at, completed_at, metadata_json "
        "FROM openclaw_executions ORDER BY created_at DESC",
    )

    per_agent = {}
    for r in all_rows:
        agent = r["agent_id"]
        per_agent.setdefault(agent, []).append({
            "execution_id": r["execution_id"],
            "agent_name": r["agent_name"],
            "task_description": r["task_description"],
            "status": r["status"],
            "priority": r["priority"],
            "steps_progress": f"{r['steps_completed']}/{r['steps_total']}",
            "duration_seconds": r["duration_seconds"],
            "triggered_by": r["triggered_by"],
        })

    # Recent 25 executions
    recent_executions = all_rows[:25]

    # Agent workload
    agent_workload = _rows(
        cur,
        "SELECT agent_name, "
        "SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed_count, "
        "SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) as running_count, "
        "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed_count, "
        "AVG(CASE WHEN duration_seconds IS NOT NULL THEN duration_seconds END) as avg_duration "
        "FROM openclaw_executions GROUP BY agent_name ORDER BY running_count DESC",
    )
    for row in agent_workload:
        if row.get("avg_duration") is not None:
            row["avg_duration"] = round(row["avg_duration"], 1)

    # Chained executions
    chained_executions = _rows(
        cur,
        "SELECT execution_id, agent_name, task_description, status, "
        "parent_execution_id, triggered_by, created_at "
        "FROM openclaw_executions WHERE parent_execution_id IS NOT NULL "
        "ORDER BY created_at DESC",
    )

    # Failed executions
    failed_executions = _rows(
        cur,
        "SELECT execution_id, agent_name, task_description, priority, "
        "error_message, retry_count, created_at "
        "FROM openclaw_executions WHERE status='failed' "
        "ORDER BY created_at DESC",
    )

    # Per-agent stats
    per_agent_stats = _rows(
        cur,
        "SELECT agent_name, "
        "COUNT(*) as total_count, "
        "ROUND(SUM(CASE WHEN status='completed' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as success_rate, "
        "AVG(CASE WHEN duration_seconds IS NOT NULL THEN duration_seconds END) as avg_duration, "
        "AVG(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) as avg_tokens "
        "FROM openclaw_executions GROUP BY agent_name ORDER BY total_count DESC",
    )
    for row in per_agent_stats:
        if row.get("avg_duration") is not None:
            row["avg_duration"] = round(row["avg_duration"], 1)
        if row.get("avg_tokens") is not None:
            row["avg_tokens"] = round(row["avg_tokens"], 1)

    con.close()
    return {
        "per_agent": per_agent,
        "recent_executions": recent_executions,
        "agent_workload": agent_workload,
        "chained_executions": chained_executions,
        "failed_executions": failed_executions,
        "per_agent_stats": per_agent_stats,
    }


def definitions():
    """Execution statuses, modes, trigger types, priority levels, orchestration glossary."""
    return {
        "execution_statuses": {
            "completed": "Execution finished all steps successfully and produced output",
            "running": "Execution is currently in progress, actively processing steps",
            "failed": "Execution encountered an unrecoverable error and terminated",
            "queued": "Execution is waiting in the queue to be picked up by an available agent",
            "cancelled": "Execution was manually or automatically cancelled before completion",
        },
        "execution_modes": {
            "autonomous": "Agent operates independently without human oversight for each step",
            "supervised": "Agent executes but pauses at checkpoints for human approval",
            "manual": "Human-driven execution with agent providing assistance and suggestions",
        },
        "trigger_types": {
            "cron": "Triggered on a recurring schedule (time-based automation)",
            "api": "Triggered by an external API call or integration webhook",
            "user": "Initiated manually by a human operator or clinician",
            "event": "Triggered by a system event (new data, alert, threshold breach)",
            "chain": "Triggered by completion of a parent execution (chained workflow)",
        },
        "priority_levels": {
            "critical": "Must complete immediately — patient safety or system integrity at risk",
            "high": "Must complete within 1 hour — time-sensitive clinical or operational task",
            "medium": "Standard priority — complete within same business day",
            "low": "Best-effort — no strict deadline, can be deferred if resources are constrained",
        },
        "glossary": {
            "execution": "A single run of an agent performing a defined task with tracked inputs and outputs",
            "agent": "An autonomous or semi-autonomous software entity that performs tasks on behalf of the system",
            "orchestrator": "The central coordinator that schedules, dispatches, and monitors agent executions",
            "DAG": "Directed Acyclic Graph — a dependency structure ensuring tasks execute in correct order without cycles",
            "pipeline": "A sequence of chained executions where each step feeds into the next",
            "idempotent": "Property ensuring re-running an execution produces the same result without side-effect duplication",
            "circuit breaker": "Pattern that stops retrying a failing agent after a threshold, preventing cascade failures",
            "backpressure": "Mechanism to slow intake of new executions when downstream agents are overloaded",
            "fan-out": "Pattern where a single execution spawns multiple parallel child executions",
            "fan-in": "Pattern where multiple parallel executions must all complete before a downstream step proceeds",
            "saga": "Long-running distributed transaction with compensating actions for rollback on failure",
            "dead letter queue": "Holding area for executions that fail repeatedly and require manual intervention",
        },
    }


if __name__ == "__main__":
    print(json.dumps({"overview": overview(), "breakdown": breakdown(), "definitions": definitions()}, indent=2, default=str))
