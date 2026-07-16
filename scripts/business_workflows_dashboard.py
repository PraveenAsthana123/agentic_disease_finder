"""Business Workflows Dashboard — workflow automation analytics from clinical.db.

Tracks clinical and administrative workflow execution: status distribution,
category breakdown, trigger types, step completion rates, owner workload,
priority mix, failure/retry analysis, and per-patient workflow history.

Sources:
- business_workflows table (workflow_id, workflow_name, category, status,
  priority, trigger_type, steps_total, steps_completed, owner, patient_id,
  created_at, completed_at, duration_seconds, error_message, retry_count)
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """High-level workflow metrics: totals, status distribution, category
    breakdown, priority mix, trigger types, completion rate, avg duration,
    monthly trend."""
    conn = _conn()
    cur = conn.cursor()

    # totals
    cur.execute("SELECT COUNT(*) FROM business_workflows")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM business_workflows WHERE patient_id IS NOT NULL")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT workflow_name) FROM business_workflows")
    total_types = cur.fetchone()[0]

    # status distribution
    cur.execute("SELECT status, COUNT(*) FROM business_workflows GROUP BY status ORDER BY COUNT(*) DESC")
    status_dist = [{"status": r[0], "count": r[1]} for r in cur.fetchall()]

    # completion rate
    cur.execute("SELECT COUNT(*) FROM business_workflows WHERE status='completed'")
    completed = cur.fetchone()[0]
    completion_rate = round(100.0 * completed / total, 1) if total else 0

    # failure rate
    cur.execute("SELECT COUNT(*) FROM business_workflows WHERE status='failed'")
    failed = cur.fetchone()[0]
    failure_rate = round(100.0 * failed / total, 1) if total else 0

    # active count
    cur.execute("SELECT COUNT(*) FROM business_workflows WHERE status='active'")
    active = cur.fetchone()[0]

    # avg duration (completed only)
    cur.execute("SELECT AVG(duration_seconds), MIN(duration_seconds), MAX(duration_seconds) FROM business_workflows WHERE status='completed' AND duration_seconds IS NOT NULL AND duration_seconds > 0")
    dur = cur.fetchone()
    avg_duration_min = round(dur[0] / 60, 1) if dur[0] else 0
    min_duration_min = round(dur[1] / 60, 1) if dur[1] else 0
    max_duration_min = round(dur[2] / 60, 1) if dur[2] else 0

    # avg step completion %
    cur.execute("SELECT AVG(100.0 * steps_completed / steps_total) FROM business_workflows WHERE steps_total > 0")
    avg_step_pct = round(cur.fetchone()[0] or 0, 1)

    # category breakdown
    cur.execute("SELECT category, COUNT(*) FROM business_workflows GROUP BY category ORDER BY COUNT(*) DESC")
    category_dist = [{"category": r[0], "count": r[1]} for r in cur.fetchall()]

    # priority distribution
    cur.execute("SELECT priority, COUNT(*) FROM business_workflows GROUP BY priority ORDER BY COUNT(*) DESC")
    priority_dist = [{"priority": r[0], "count": r[1]} for r in cur.fetchall()]

    # trigger type distribution
    cur.execute("SELECT trigger_type, COUNT(*) FROM business_workflows GROUP BY trigger_type ORDER BY COUNT(*) DESC")
    trigger_dist = [{"trigger_type": r[0], "count": r[1]} for r in cur.fetchall()]

    # workflow name distribution
    cur.execute("SELECT workflow_name, COUNT(*) FROM business_workflows GROUP BY workflow_name ORDER BY COUNT(*) DESC")
    workflow_dist = [{"workflow_name": r[0], "count": r[1]} for r in cur.fetchall()]

    # monthly trend
    cur.execute("""
        SELECT strftime('%Y-%m', created_at) as month,
               COUNT(*) as total,
               SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed
        FROM business_workflows
        GROUP BY strftime('%Y-%m', created_at)
        ORDER BY month
    """)
    monthly_trend = [{"month": r[0], "total": r[1], "completed": r[2], "failed": r[3]} for r in cur.fetchall()]

    # retry stats
    cur.execute("SELECT SUM(retry_count), COUNT(*) FROM business_workflows WHERE retry_count > 0")
    retry_row = cur.fetchone()
    total_retries = retry_row[0] or 0
    workflows_with_retries = retry_row[1] or 0

    conn.close()
    return {
        "total_workflows": total,
        "total_patients": total_patients,
        "total_workflow_types": total_types,
        "active_workflows": active,
        "completion_rate": completion_rate,
        "failure_rate": failure_rate,
        "avg_duration_min": avg_duration_min,
        "min_duration_min": min_duration_min,
        "max_duration_min": max_duration_min,
        "avg_step_completion_pct": avg_step_pct,
        "total_retries": total_retries,
        "workflows_with_retries": workflows_with_retries,
        "status_distribution": status_dist,
        "category_distribution": category_dist,
        "priority_distribution": priority_dist,
        "trigger_distribution": trigger_dist,
        "workflow_distribution": workflow_dist,
        "monthly_trend": monthly_trend,
    }


def breakdown():
    """Per-owner workload, per-workflow-type stats, active/failed workflows,
    recent workflows, category x status cross-tab, per-patient history."""
    conn = _conn()
    cur = conn.cursor()

    # owner workload
    cur.execute("""
        SELECT owner,
               COUNT(*) as total,
               SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed,
               ROUND(AVG(CASE WHEN status='completed' AND duration_seconds > 0 THEN duration_seconds / 60.0 END), 1) as avg_duration_min
        FROM business_workflows
        GROUP BY owner
        ORDER BY total DESC
    """)
    owner_workload = _dict_rows(cur)

    # per-workflow-type stats
    cur.execute("""
        SELECT workflow_name,
               COUNT(*) as total,
               SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed,
               ROUND(AVG(100.0 * steps_completed / steps_total), 1) as avg_step_pct,
               ROUND(AVG(CASE WHEN status='completed' AND duration_seconds > 0 THEN duration_seconds / 60.0 END), 1) as avg_duration_min
        FROM business_workflows
        WHERE steps_total > 0
        GROUP BY workflow_name
        ORDER BY total DESC
    """)
    workflow_stats = _dict_rows(cur)

    # active workflows (currently running)
    cur.execute("""
        SELECT workflow_id, workflow_name, category, priority, owner, patient_id,
               steps_completed, steps_total, created_at
        FROM business_workflows
        WHERE status = 'active'
        ORDER BY priority DESC, created_at
    """)
    active_workflows = _dict_rows(cur)

    # failed workflows
    cur.execute("""
        SELECT workflow_id, workflow_name, category, priority, owner, patient_id,
               steps_completed, steps_total, error_message, retry_count, created_at
        FROM business_workflows
        WHERE status = 'failed'
        ORDER BY created_at DESC
    """)
    failed_workflows = _dict_rows(cur)

    # recent workflows (last 20)
    cur.execute("""
        SELECT workflow_id, workflow_name, category, status, priority, trigger_type,
               owner, patient_id, steps_completed, steps_total,
               created_at, completed_at, duration_seconds, retry_count
        FROM business_workflows
        ORDER BY created_at DESC
        LIMIT 20
    """)
    recent = _dict_rows(cur)

    # category x status cross-tab
    cur.execute("""
        SELECT category, status, COUNT(*) as cnt
        FROM business_workflows
        GROUP BY category, status
        ORDER BY category, status
    """)
    cross_rows = cur.fetchall()
    cat_status = {}
    for cat, st, cnt in cross_rows:
        if cat not in cat_status:
            cat_status[cat] = {"category": cat}
        cat_status[cat][st] = cnt
    category_status_crosstab = list(cat_status.values())

    # per-patient workflow count
    cur.execute("""
        SELECT patient_id, COUNT(*) as total,
               SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed
        FROM business_workflows
        WHERE patient_id IS NOT NULL
        GROUP BY patient_id
        ORDER BY total DESC
        LIMIT 20
    """)
    per_patient = _dict_rows(cur)

    conn.close()
    return {
        "owner_workload": owner_workload,
        "workflow_stats": workflow_stats,
        "active_workflows": active_workflows,
        "failed_workflows": failed_workflows,
        "recent_workflows": recent,
        "category_status_crosstab": category_status_crosstab,
        "per_patient": per_patient,
    }


def definitions():
    """Workflow field definitions, status meanings, category descriptions,
    priority levels, trigger types, clinical glossary."""
    return {
        "fields": {
            "workflow_id": "Unique workflow instance identifier (e.g. WF-001)",
            "workflow_name": "Type of workflow (e.g. Patient Intake, Referral Processing)",
            "category": "Workflow domain — Administrative, Clinical, Compliance, or Technical",
            "status": "Current workflow state (active, completed, failed, paused, pending)",
            "priority": "Workflow urgency level (critical, high, medium, low)",
            "trigger_type": "How the workflow was initiated (manual, event-driven, scheduled, api)",
            "steps_total": "Total number of steps in the workflow definition",
            "steps_completed": "Number of steps that have been executed",
            "owner": "Role/user responsible for the workflow (admin, clinician, scheduler, system)",
            "patient_id": "Patient linked to this workflow instance (may be null for system workflows)",
            "created_at": "ISO timestamp when the workflow was created",
            "completed_at": "ISO timestamp when the workflow finished (null if not completed)",
            "duration_seconds": "Total execution time in seconds (completed workflows only)",
            "error_message": "Error details if the workflow failed",
            "retry_count": "Number of times the workflow was retried after failure",
        },
        "statuses": {
            "active": "Workflow is currently executing — steps are being processed",
            "completed": "All steps finished successfully",
            "failed": "Workflow encountered an error and stopped — may be retried",
            "paused": "Workflow was manually or automatically paused — waiting for intervention",
            "pending": "Workflow is queued but has not started executing yet",
        },
        "categories": {
            "Administrative": "Business operations — scheduling, billing, referrals, intake",
            "Clinical": "Patient care workflows — treatment plans, medication reconciliation, discharge",
            "Compliance": "Regulatory and audit workflows — consent review, HIPAA checks, quality assurance",
            "Technical": "System operations — EEG upload pipelines, data export, report generation",
        },
        "priorities": {
            "critical": "Must be resolved immediately — patient safety or system-down",
            "high": "Should be addressed within hours — clinical workflows, active referrals",
            "medium": "Standard priority — routine operations, scheduled tasks",
            "low": "Can be deferred — background tasks, non-urgent administrative work",
        },
        "trigger_types": {
            "manual": "Started by a user action (button click, form submission)",
            "event-driven": "Triggered automatically by a system event (new patient, EEG upload, alert)",
            "scheduled": "Runs on a time-based schedule (daily, weekly, cron)",
            "api": "Initiated via API call from an external system or integration",
        },
        "workflow_types": {
            "Patient Intake": "New patient registration, demographics, consent, initial assessment",
            "Referral Processing": "Incoming referral triage, scheduling, specialist assignment",
            "Appointment Scheduling": "Patient appointment creation, confirmation, reminders",
            "EEG Upload Pipeline": "EEG file upload, format validation, artifact detection, AI pre-read",
            "Compliance Review": "HIPAA audit, consent verification, data retention checks",
            "Report Generation": "Clinical report creation, review, sign-off, distribution",
            "Insurance Pre-Auth": "Insurance pre-authorization request, follow-up, approval tracking",
            "Medication Reconciliation": "Cross-check current medications, interactions, dosage verification",
            "Data Export": "Patient data export for research, transfer, or backup",
            "Discharge Planning": "Discharge checklist, follow-up scheduling, care instructions",
        },
        "glossary": {
            "Workflow": "A defined sequence of steps that automates a clinical or administrative process",
            "Step Completion": "Percentage of workflow steps executed out of total defined steps",
            "Workflow Duration": "Time elapsed from workflow creation to completion",
            "Retry": "Re-execution of a failed workflow, usually after fixing the underlying issue",
            "Owner": "The role or user accountable for a workflow's progress and resolution",
            "Trigger": "The event or mechanism that initiates a workflow instance",
            "SLA": "Service Level Agreement — target time for workflow completion",
            "Bottleneck": "A workflow step or category that causes delays in overall throughput",
            "Throughput": "Number of workflows completed per unit of time",
            "Error Rate": "Percentage of workflows that end in failure",
            "Queue Depth": "Number of pending workflows waiting to start",
            "Escalation": "Transferring a stalled or failed workflow to a higher-priority handler",
        },
    }
