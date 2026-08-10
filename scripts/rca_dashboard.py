"""
Root Cause Analysis (RCA) Center Dashboard
Surfaces open issues, component failures, SLA breaches, and fix status
for the NeuroLab AI epilepsy platform.

Reads: advisor_issues, system_health_log (clinical.db)
Endpoints: /api/rca/overview | breakdown | definitions
"""
import os
import sqlite3
from datetime import datetime, timezone

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH  = os.path.join(_BASE_DIR, "data", "clinical.db")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SEV_ORDER  = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
_SEV_LABELS = {
    "P1": "Critical — system-down / data-loss risk",
    "P2": "High — significant degradation",
    "P3": "Medium — limited impact",
    "P4": "Low — cosmetic / informational",
}
_LAYER_ICON = {
    "model":    "🤖",
    "frontend": "🖥️",
    "data":     "🗄️",
    "security": "🔒",
    "git":      "📦",
    "backend":  "⚙️",
    "infra":    "🏗️",
    "api":      "🔌",
    "vector":   "🧮",
    "graph":    "🕸️",
    "other":    "📋",
}
_STATUS_ICON = {"open": "🔴", "running": "🟡", "fixed": "✅", "closed": "✅"}


def _conn():
    return sqlite3.connect(_DB_PATH)


def _load_issues():
    """Load advisor_issues rows."""
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT id, severity, surface, issue, guidance, status, scanned_at "
                "FROM advisor_issues ORDER BY id DESC"
            ).fetchall()
    except Exception:
        rows = []
    issues = []
    for r in rows:
        iid, sev, surface, issue_text, guidance, status, scanned_at = r
        issues.append({
            "id":        f"RCA-{iid:03d}",
            "severity":  sev or "P3",
            "layer":     (surface or "other").lower(),
            "symptom":   issue_text or "",
            "fix":       guidance or "",
            "status":    (status or "open").lower(),
            "scanned_at": scanned_at or "",
        })
    return issues


def _load_component_health():
    """Latest health status per component from system_health_log."""
    try:
        with _conn() as con:
            rows = con.execute(
                """SELECT component, status, response_time_ms, cpu_pct, memory_pct,
                          error_count, MAX(timestamp) as ts
                   FROM system_health_log
                   GROUP BY component"""
            ).fetchall()
    except Exception:
        rows = []
    components = []
    for r in rows:
        comp, status, rt, cpu, mem, errs, ts = r
        components.append({
            "component":       comp,
            "status":          status or "unknown",
            "response_time_ms": rt or 0,
            "cpu_pct":         round(cpu or 0, 1),
            "memory_pct":      round(mem or 0, 1),
            "error_count":     errs or 0,
            "last_check":      ts or "",
        })
    return sorted(components, key=lambda x: 0 if x["status"] == "down" else (1 if x["status"] == "degraded" else 2))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def overview():
    issues    = _load_issues()
    components = _load_component_health()

    open_issues   = [i for i in issues if i["status"] == "open"]
    fixed_issues  = [i for i in issues if i["status"] in ("fixed", "closed")]
    running_issues = [i for i in issues if i["status"] == "running"]
    p1_open = sum(1 for i in open_issues if i["severity"] == "P1")
    p2_open = sum(1 for i in open_issues if i["severity"] == "P2")

    down_components    = [c for c in components if c["status"] == "down"]
    degraded_components = [c for c in components if c["status"] == "degraded"]
    healthy_components = [c for c in components if c["status"] == "healthy"]

    # SLA breaches: open P1 = breach
    sla_breached = p1_open

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "kpis": {
            "open_issues":       len(open_issues),
            "p1_critical":       p1_open,
            "p2_high":           p2_open,
            "sla_breached":      sla_breached,
            "fixed_issues":      len(fixed_issues),
            "in_progress":       len(running_issues),
            "components_total":  len(components),
            "components_down":   len(down_components),
            "components_degraded": len(degraded_components),
            "components_healthy": len(healthy_components),
        },
        "severity_breakdown": {
            sev: sum(1 for i in issues if i["severity"] == sev)
            for sev in ("P1", "P2", "P3", "P4")
        },
        "layer_breakdown": _layer_breakdown(issues),
        "status_summary": {
            "open":    len(open_issues),
            "running": len(running_issues),
            "fixed":   len(fixed_issues),
        },
        "top_open": sorted(open_issues, key=lambda x: _SEV_ORDER.get(x["severity"], 9))[:5],
        "component_summary": {
            "down":    [c["component"] for c in down_components],
            "degraded":[c["component"] for c in degraded_components],
            "healthy": [c["component"] for c in healthy_components],
        },
    }


def _layer_breakdown(issues):
    layers = {}
    for i in issues:
        layer = i["layer"] or "other"
        if layer not in layers:
            layers[layer] = {"open": 0, "fixed": 0, "total": 0, "icon": _LAYER_ICON.get(layer, "📋")}
        layers[layer]["total"] += 1
        if i["status"] in ("fixed", "closed"):
            layers[layer]["fixed"] += 1
        elif i["status"] == "open":
            layers[layer]["open"] += 1
    return layers


# ---------------------------------------------------------------------------
# Breakdown
# ---------------------------------------------------------------------------
def breakdown():
    issues     = _load_issues()
    components = _load_component_health()

    enriched = []
    for i in issues:
        enriched.append({
            **i,
            "layer_icon": _LAYER_ICON.get(i["layer"], "📋"),
            "status_icon": _STATUS_ICON.get(i["status"], "🔴"),
            "sev_order":   _SEV_ORDER.get(i["severity"], 9),
        })
    enriched.sort(key=lambda x: (x["sev_order"], x["id"]))

    return {
        "generated":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "issues":     enriched,
        "components": components,
        "sla_rules": [
            {"event": "NEW → PLANNED",         "sla": "2 min",  "description": "Task must be planned within 2 minutes of creation"},
            {"event": "PLANNED → JOB_CREATED", "sla": "2 min",  "description": "Job must be created within 2 minutes of planning"},
            {"event": "JOB_CREATED → RUNNING", "sla": "1 min",  "description": "Job must start within 1 minute of creation"},
            {"event": "RUNNING heartbeat gap",  "sla": "30 sec", "description": "Heartbeat must be received every 30 seconds"},
            {"event": "RUNNING no update",      "sla": "2 min",  "description": "If no update for 2 min, mark STALE"},
            {"event": "BLOCKED without RCA",    "sla": "5 min",  "description": "Every blocked task must have an RCA within 5 minutes"},
            {"event": "DONE without evidence",  "sla": "N/A",    "description": "DONE status requires evidence — not allowed without it"},
        ],
        "metrics": {
            "total_issues": len(issues),
            "open":   sum(1 for i in issues if i["status"] == "open"),
            "fixed":  sum(1 for i in issues if i["status"] in ("fixed", "closed")),
            "p1_open": sum(1 for i in issues if i["severity"] == "P1" and i["status"] == "open"),
        },
    }


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
def definitions():
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "terms": [
            {
                "term": "Root Cause Analysis (RCA)",
                "definition": "Structured method to identify the fundamental reason a failure occurred, not just its symptoms.",
                "example": "UI not showing latest job status → root cause: frontend cache not refreshed, not API failure.",
            },
            {
                "term": "Severity (P1–P4)",
                "definition": "Priority classification of an issue based on impact.",
                "levels": _SEV_LABELS,
            },
            {
                "term": "Layer",
                "definition": "The architectural layer where the failure originated.",
                "layers": {k: v for k, v in _LAYER_ICON.items()},
            },
            {
                "term": "SLA (Service Level Agreement)",
                "definition": "Maximum acceptable time between system events. Breaching an SLA triggers an alert and RCA requirement.",
            },
            {
                "term": "STALE",
                "definition": "A running job that has not sent a heartbeat for more than 2 minutes. Triggers automatic Recovery Agent.",
            },
            {
                "term": "BLOCKED",
                "definition": "A task that cannot proceed without external input (credentials, approval, data). Requires RCA within 5 minutes.",
            },
            {
                "term": "Evidence",
                "definition": "Proof that a fix actually resolved the issue. Required before marking status DONE/FIXED. Includes API 200s, test pass, screenshot, or DB record.",
            },
            {
                "term": "Recovery Agent",
                "definition": "Automated agent triggered when a job goes STALE. Attempts to restart the job and log the RCA.",
            },
        ],
        "fields": [
            {"field": "Issue ID",   "description": "Unique identifier for the RCA record (e.g., RCA-001)"},
            {"field": "Severity",   "description": "P1 (critical) → P4 (low)"},
            {"field": "Layer",      "description": "Where the failure originated (model, frontend, data, etc.)"},
            {"field": "Symptom",    "description": "Observable manifestation of the problem"},
            {"field": "Root Cause", "description": "Underlying reason the symptom occurred"},
            {"field": "Fix",        "description": "Recommended or applied corrective action"},
            {"field": "Evidence",   "description": "Verification that the fix worked"},
            {"field": "Status",     "description": "open / running / fixed"},
            {"field": "Owner",      "description": "Agent or operator responsible for resolution"},
        ],
    }
