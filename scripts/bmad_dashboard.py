"""
BMAD Dashboard — Breakthrough Method for Agile ai-agent Development
Spec-driven agent methodology analysis: registry coverage, category breakdown,
dependency tracking, and compliance metrics from config/agent_tasks.json + clinical.db.
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, 'data', 'clinical.db')
AGENT_TASKS = os.path.join(BASE, 'config', 'agent_tasks.json')


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _conn():
    return sqlite3.connect(DB)


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def _categorize(task_text):
    """Classify an agent into a category based on task text keywords."""
    t = (task_text or "").lower()
    if "ingest" in t:
        return "Data Ingestion"
    if any(k in t for k in ("train", "model", "classifier", "xgboost", "lightgbm", "cnn", "resnet", "lstm")):
        return "ML/Training"
    if any(k in t for k in ("dashboard", "report", "analytics")):
        return "Analytics"
    if any(k in t for k in ("review", "audit", "check", "verify")):
        return "Quality/Review"
    if any(k in t for k in ("ai", "agent", "orchestrat")):
        return "AI/Agent"
    return "Infrastructure"


def overview():
    """Registry-level BMAD compliance overview + clinical.db counts."""
    data = _load_json(AGENT_TASKS)
    agents = data.get("agents", []) if data else []

    total = len(agents)
    built = sum(1 for a in agents if a.get("status") == "built")
    planned = sum(1 for a in agents if a.get("status") == "planned")
    spec_coverage_pct = round(built / total * 100, 1) if total else 0.0

    agents_with_modules = sum(
        1 for a in agents
        if a.get("module") and not a["module"].lower().startswith("(planned")
    )
    agents_with_needs = sum(1 for a in agents if a.get("needs"))

    # Status distribution
    status_counts = {}
    for a in agents:
        s = a.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    status_distribution = [{"status": k, "count": v} for k, v in sorted(status_counts.items())]

    # Category distribution
    cat_counts = {}
    for a in agents:
        cat = _categorize(a.get("task", ""))
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    category_distribution = [{"category": k, "count": v} for k, v in sorted(cat_counts.items())]

    # Implementation completeness
    implementation_completeness = [
        {
            "agent_id": a["id"],
            "task": (a.get("task", "") or "")[:60],
            "status": a.get("status", "unknown"),
            "has_module": bool(a.get("module") and not a["module"].lower().startswith("(planned")),
            "has_needs": bool(a.get("needs")),
        }
        for a in agents
    ]

    # Needs summary
    needs_counter = {}
    for a in agents:
        n = a.get("needs")
        if n:
            needs_counter[n] = needs_counter.get(n, 0) + 1
    needs_summary = [{"need": k, "count": v} for k, v in sorted(needs_counter.items(), key=lambda x: -x[1])]

    # Clinical DB counts
    total_decisions = 0
    total_transactions = 0
    try:
        con = _conn()
        cur = con.cursor()
        total_decisions = _safe_count(cur, "SELECT COUNT(*) FROM clinical_decisions")
        total_transactions = _safe_count(cur, "SELECT COUNT(*) FROM transaction_log")
        con.close()
    except Exception:
        pass

    return {
        "total_agents": total,
        "built_agents": built,
        "planned_agents": planned,
        "spec_coverage_pct": spec_coverage_pct,
        "agents_with_modules": agents_with_modules,
        "agents_with_needs": agents_with_needs,
        "status_distribution": status_distribution,
        "category_distribution": category_distribution,
        "implementation_completeness": implementation_completeness,
        "needs_summary": needs_summary,
        "total_decisions": total_decisions,
        "total_transactions": total_transactions,
        "spec_to_production_rate": spec_coverage_pct,
    }


def breakdown():
    """Per-category and per-status agent detail + clinical.db activity."""
    data = _load_json(AGENT_TASKS)
    agents = data.get("agents", []) if data else []

    # Per category
    per_category = {}
    for a in agents:
        cat = _categorize(a.get("task", ""))
        per_category.setdefault(cat, []).append({
            "id": a["id"],
            "task": a.get("task", ""),
            "status": a.get("status", "unknown"),
            "module": a.get("module", ""),
            "needs": a.get("needs"),
        })

    built_agents_detail = [
        {
            "id": a["id"],
            "task": a.get("task", ""),
            "module": a.get("module", ""),
            "needs": a.get("needs"),
        }
        for a in agents if a.get("status") == "built"
    ]

    planned_agents_detail = [
        {
            "id": a["id"],
            "task": a.get("task", ""),
            "needs": a.get("needs"),
        }
        for a in agents if a.get("status") == "planned"
    ]

    # Module coverage
    with_real = sum(
        1 for a in agents
        if a.get("module") and not a["module"].lower().startswith("(planned")
    )
    with_planned = sum(
        1 for a in agents
        if a.get("module") and a["module"].lower().startswith("(planned")
    )
    module_coverage = {
        "with_real_module": with_real,
        "with_planned_module": with_planned,
        "total": len(agents),
    }

    # Dependency map
    dependency_map = [
        {"agent_id": a["id"], "needs": a["needs"]}
        for a in agents if a.get("needs")
    ]

    # Recent activity from transaction_log
    recent_activity = []
    try:
        con = _conn()
        cur = con.cursor()
        cur.execute(
            "SELECT action, COUNT(*) as cnt FROM transaction_log "
            "GROUP BY action ORDER BY cnt DESC LIMIT 10"
        )
        recent_activity = [{"action": r[0], "count": r[1]} for r in cur.fetchall()]
        con.close()
    except Exception:
        pass

    return {
        "per_category": per_category,
        "built_agents_detail": built_agents_detail,
        "planned_agents_detail": planned_agents_detail,
        "module_coverage": module_coverage,
        "dependency_map": dependency_map,
        "recent_activity": recent_activity,
    }


def definitions():
    """BMAD methodology definitions, status semantics, and category descriptions."""
    return {
        "bmad_method": (
            "Breakthrough Method for Agile ai-agent Development — spec-first methodology "
            "where every agent is formally specified (id, task, module, status, dependencies) "
            "before implementation begins. Ensures traceability from requirement to deployed "
            "capability."
        ),
        "statuses": {
            "built": "Agent is implemented, tested, and verified",
            "planned": "Agent is specified but not yet implemented",
            "partial": "Agent has scaffolding but is not complete",
        },
        "categories": {
            "Data Ingestion": "Agents that ingest, parse, or extract data from files and devices",
            "ML/Training": "Agents that train, evaluate, or serve machine learning models",
            "Analytics": "Agents that produce dashboards, reports, or KPI analytics",
            "Quality/Review": "Agents that audit, review, check, or verify outputs",
            "AI/Agent": "AI assistants, orchestrators, and autonomous agent capabilities",
            "Infrastructure": "Supporting infrastructure, gateways, and system-level services",
        },
        "spec_fields": {
            "id": "Unique agent identifier",
            "task": "What the agent does",
            "module": "Python module/script implementing the agent",
            "status": "Current implementation status",
            "needs": "External dependencies or prerequisites",
        },
        "compliance_note": (
            "All agents follow the BMAD spec-first pattern: specify before building, "
            "track dependencies, measure coverage."
        ),
    }


if __name__ == "__main__":
    print(json.dumps({"overview": overview(), "breakdown": breakdown(), "definitions": definitions()}, indent=2, default=str))
