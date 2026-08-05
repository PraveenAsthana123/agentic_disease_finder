"""Temporal Approval Workflow Dashboard — AI Dark Factory Stage 9
Durable approval queues, workflow states, deployment pipeline (stage 10).
Data sourced from config/ai_dark_factory.json + in-memory approval queue."""

import json
import time
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "ai_dark_factory.json"

# ---------------------------------------------------------------------------
# Static approval queue data (representative; no live Temporal server needed)
# ---------------------------------------------------------------------------
_APPROVAL_QUEUE = [
    {
        "id": "wf-001",
        "workflow": "EEG Model Retrain v2.4",
        "requested_by": "AutoBuild Agent",
        "stage": "Human Approval Gate",
        "status": "pending",
        "created_at": "2026-08-04T14:00:00Z",
        "sla_hours": 4,
        "elapsed_hours": 2.1,
        "priority": "high",
        "artifact": "train_90_plus.py → model_v2.4.pt",
        "eval_score": 0.913,
        "risk": "medium",
    },
    {
        "id": "wf-002",
        "workflow": "Fairness Gate — Depression Model",
        "requested_by": "Responsible AI Agent",
        "stage": "Bias Review",
        "status": "pending",
        "created_at": "2026-08-04T12:30:00Z",
        "sla_hours": 6,
        "elapsed_hours": 5.2,
        "priority": "high",
        "artifact": "depression_model_v3.1.onnx",
        "eval_score": 0.887,
        "risk": "high",
    },
    {
        "id": "wf-003",
        "workflow": "Deploy Seizure Forecasting API",
        "requested_by": "DeepEval Validation",
        "stage": "Deployment Approval",
        "status": "approved",
        "created_at": "2026-08-04T08:00:00Z",
        "sla_hours": 8,
        "elapsed_hours": 6.0,
        "priority": "medium",
        "artifact": "seizure_forecasting_v1.2",
        "eval_score": 0.925,
        "risk": "low",
        "approved_by": "Clinical AI Lead",
        "approved_at": "2026-08-04T14:00:00Z",
    },
    {
        "id": "wf-004",
        "workflow": "OTel Trace Config Update",
        "requested_by": "Infra Agent",
        "stage": "Change Review",
        "status": "approved",
        "created_at": "2026-08-03T18:00:00Z",
        "sla_hours": 12,
        "elapsed_hours": 10.0,
        "priority": "low",
        "artifact": "otel_config_v2.yaml",
        "eval_score": None,
        "risk": "low",
        "approved_by": "DevOps Lead",
        "approved_at": "2026-08-04T04:00:00Z",
    },
    {
        "id": "wf-005",
        "workflow": "ICA Noise Cleaning Pipeline",
        "requested_by": "Data Quality Agent",
        "stage": "Data Governance Review",
        "status": "rejected",
        "created_at": "2026-08-03T10:00:00Z",
        "sla_hours": 4,
        "elapsed_hours": 8.0,
        "priority": "medium",
        "artifact": "ica_noise_cleaning.py",
        "eval_score": 0.741,
        "risk": "medium",
        "rejected_by": "Data Governance Board",
        "rejected_at": "2026-08-03T18:00:00Z",
        "rejection_reason": "Insufficient artifact label coverage (< 80%)",
    },
]

_DEPLOYMENT_STAGES = [
    {"n": 1, "name": "Plan", "tool": "BMAD", "status": "complete", "icon": "📋"},
    {"n": 2, "name": "Code", "tool": "OpenHands", "status": "complete", "icon": "💻"},
    {"n": 3, "name": "Test", "tool": "Playwright + DeepEval", "status": "complete", "icon": "🧪"},
    {"n": 4, "name": "Approval", "tool": "Temporal (this dashboard)", "status": "active", "icon": "✅"},
    {"n": 5, "name": "Deploy", "tool": "Harness/CI", "status": "pending", "icon": "🚀"},
    {"n": 6, "name": "Monitor", "tool": "OTel + OpenLIT", "status": "pending", "icon": "📡"},
]

_WORKFLOW_STATES = [
    {"state": "PENDING", "desc": "Waiting for human reviewer", "color": "warning"},
    {"state": "RUNNING", "desc": "Reviewer actively working", "color": "info"},
    {"state": "APPROVED", "desc": "Approval granted — ready to deploy", "color": "success"},
    {"state": "REJECTED", "desc": "Rejected — workflow paused", "color": "danger"},
    {"state": "TIMED_OUT", "desc": "SLA breached — escalated", "color": "secondary"},
    {"state": "CANCELLED", "desc": "Operator cancelled", "color": "dark"},
]


def _load_cfg():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# ── overview ────────────────────────────────────────────────────────────────
def overview():
    """KPIs: queue depth, pending/approved/rejected, SLA stats, deployment pipeline."""
    cfg = _load_cfg()
    temporal_stage = next(
        (s for s in (cfg or {}).get("full_flow", []) if s.get("n") == 9), {}
    )

    total = len(_APPROVAL_QUEUE)
    pending = sum(1 for w in _APPROVAL_QUEUE if w["status"] == "pending")
    approved = sum(1 for w in _APPROVAL_QUEUE if w["status"] == "approved")
    rejected = sum(1 for w in _APPROVAL_QUEUE if w["status"] == "rejected")
    sla_breached = sum(
        1 for w in _APPROVAL_QUEUE
        if w["status"] == "pending" and w["elapsed_hours"] >= w["sla_hours"]
    )
    avg_elapsed = (
        sum(w["elapsed_hours"] for w in _APPROVAL_QUEUE) / total if total else 0
    )

    # Status distribution for chart
    status_dist = [
        {"name": "Pending", "value": pending, "color": "#ffc107"},
        {"name": "Approved", "value": approved, "color": "#198754"},
        {"name": "Rejected", "value": rejected, "color": "#dc3545"},
    ]

    # Priority breakdown
    from collections import Counter
    prio_counts = Counter(w["priority"] for w in _APPROVAL_QUEUE)
    priority_dist = [
        {"name": p.capitalize(), "value": c}
        for p, c in sorted(prio_counts.items())
    ]

    # Risk breakdown
    risk_counts = Counter(
        w.get("risk", "unknown") for w in _APPROVAL_QUEUE
    )
    risk_dist = [
        {"name": r.capitalize(), "value": c}
        for r, c in sorted(risk_counts.items())
    ]

    return {
        "available": True,
        "title": "Temporal Approval Workflow",
        "description": (
            "AI Dark Factory Stage 9 — Durable human-in-the-loop approval gateway. "
            "Workflows pause here until an authorized reviewer approves, rejects, or "
            "escalates. Uses Temporal-style durable execution: retries, SLA timers, "
            "audit trail."
        ),
        "temporal_stage": temporal_stage,
        "kpis": {
            "queue_depth": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "sla_breached": sla_breached,
            "avg_elapsed_hours": round(avg_elapsed, 1),
        },
        "status_distribution": status_dist,
        "priority_distribution": priority_dist,
        "risk_distribution": risk_dist,
        "deployment_pipeline": _DEPLOYMENT_STAGES,
        "workflow_states": _WORKFLOW_STATES,
        "queue": _APPROVAL_QUEUE,
    }


# ── breakdown ────────────────────────────────────────────────────────────────
def breakdown():
    """Per-workflow detail: SLA status, approval chain, risk matrix."""
    rows = []
    for w in _APPROVAL_QUEUE:
        sla_pct = round((w["elapsed_hours"] / w["sla_hours"]) * 100, 0) if w["sla_hours"] else 0
        sla_status = (
            "ok" if sla_pct < 75
            else "warning" if sla_pct < 100
            else "breached"
        )
        rows.append({
            **w,
            "sla_pct": sla_pct,
            "sla_status": sla_status,
        })

    # SLA summary
    ok = sum(1 for r in rows if r["sla_status"] == "ok")
    warn = sum(1 for r in rows if r["sla_status"] == "warning")
    breach = sum(1 for r in rows if r["sla_status"] == "breached")

    # Harness CI deployment pipeline detail
    harness_pipeline = [
        {
            "stage": "Source",
            "action": "git checkout + version tag",
            "tool": "GitHub Actions",
            "status": "complete",
            "duration_s": 12,
        },
        {
            "stage": "Build",
            "action": "docker build + push",
            "tool": "Docker / ECR",
            "status": "complete",
            "duration_s": 94,
        },
        {
            "stage": "Test",
            "action": "pytest + DeepEval + Playwright",
            "tool": "Harness CI",
            "status": "complete",
            "duration_s": 210,
        },
        {
            "stage": "Approval Gate",
            "action": "Temporal pause — human sign-off",
            "tool": "This dashboard",
            "status": "active",
            "duration_s": None,
        },
        {
            "stage": "Canary Deploy",
            "action": "10% traffic shift",
            "tool": "Harness CD",
            "status": "pending",
            "duration_s": None,
        },
        {
            "stage": "Full Deploy",
            "action": "100% cutover",
            "tool": "Harness CD",
            "status": "pending",
            "duration_s": None,
        },
        {
            "stage": "OTel Verify",
            "action": "Error rate + latency check",
            "tool": "OpenLIT",
            "status": "pending",
            "duration_s": None,
        },
    ]

    return {
        "available": True,
        "approval_rows": rows,
        "sla_summary": {"ok": ok, "warning": warn, "breached": breach},
        "harness_pipeline": harness_pipeline,
    }


# ── definitions ──────────────────────────────────────────────────────────────
def definitions():
    """Glossary: Temporal concepts, approval flow, SLA policy, integration notes."""
    return {
        "available": True,
        "glossary": [
            {
                "term": "Temporal",
                "def": (
                    "Open-source durable workflow engine. Workflows survive "
                    "process restarts; activities are retried automatically. "
                    "Used here as the approval-pause mechanism."
                ),
            },
            {
                "term": "Durable Execution",
                "def": (
                    "Execution state is checkpointed in Temporal's history so "
                    "long-running approvals (hours/days) are never lost."
                ),
            },
            {
                "term": "Human Signal (HITL)",
                "def": (
                    "Temporal.signal() unblocks a paused workflow when a "
                    "reviewer approves. Rejection triggers compensating actions."
                ),
            },
            {
                "term": "SLA Timer",
                "def": (
                    "A Temporal timer fires after N hours; if no signal "
                    "arrives, the workflow auto-escalates to a senior reviewer."
                ),
            },
            {
                "term": "Harness/CI",
                "def": (
                    "CI/CD platform (stage 10). Harness pipelines consume the "
                    "approval result from Temporal and trigger canary → full deploy."
                ),
            },
            {
                "term": "Approval Gate",
                "def": (
                    "A required checkpoint in the AI lifecycle where a human "
                    "must explicitly approve before the workflow can continue."
                ),
            },
            {
                "term": "Canary Deploy",
                "def": (
                    "Route a small percentage (10%) of traffic to the new "
                    "model/service while monitoring error rate and latency."
                ),
            },
        ],
        "sla_policy": {
            "high_priority_hours": 4,
            "medium_priority_hours": 8,
            "low_priority_hours": 24,
            "escalation": "Auto-escalate to Senior Clinical AI Lead on SLA breach",
        },
        "integration_note": (
            "This dashboard simulates the Temporal approval queue. "
            "Production integration: Temporal SDK (Go/Python) + Harness webhook. "
            "Adopting Temporal requires §56 6-gate review before install."
        ),
        "references": [
            {"name": "Temporal.io docs", "url": "https://docs.temporal.io"},
            {"name": "Harness CI/CD", "url": "https://developer.harness.io"},
            {"name": "AI Dark Factory overview", "url": "/ai-dark-factory"},
            {"name": "HITL Human Evaluation", "url": "/human-evaluation"},
        ],
    }
