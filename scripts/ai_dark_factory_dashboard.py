"""AI Dark Factory Dashboard — Autonomous Software Factory Reference
visualization from config/ai_dark_factory.json.
11 flow stages, 18 tools, 3 planes, 2 patterns."""

import json
from pathlib import Path
from collections import Counter

_CFG = Path(__file__).resolve().parent.parent / "config" / "ai_dark_factory.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: flow stages, built/cataloged/planned counts, tools,
    categories, planes, patterns, status distributions, charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "ai_dark_factory.json missing"}

    flow = cfg.get("full_flow", [])
    tools = cfg.get("tool_catalog", [])
    planes = cfg.get("planes", [])
    patterns = cfg.get("patterns", {})

    # Flow stage status counts
    flow_status = Counter(s.get("status", "unknown") for s in flow)
    total_stages = len(flow)
    built = flow_status.get("built", 0)
    cataloged = flow_status.get("cataloged", 0)
    planned = flow_status.get("planned", 0)

    # Tool stats
    total_tools = len(tools)
    tool_categories = sorted(set(t.get("category", "unknown") for t in tools))
    tool_status = Counter(t.get("status", "unknown") for t in tools)

    # Tools per category
    cat_counter = Counter(t.get("category", "unknown") for t in tools)
    tools_per_category = [
        {"name": cat, "value": cnt}
        for cat, cnt in sorted(cat_counter.items(), key=lambda x: -x[1])
    ]

    # Status distributions (pie chart data)
    flow_status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(flow_status.items(), key=lambda x: -x[1])
    ]
    tool_status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(tool_status.items(), key=lambda x: -x[1])
    ]

    # Flow table (all 11 stages)
    flow_table = []
    for s in flow:
        flow_table.append({
            "n": s.get("n"),
            "stage": s.get("stage", ""),
            "tool": s.get("tool", ""),
            "produces": s.get("produces", ""),
            "status": s.get("status", ""),
            "note": s.get("note", ""),
        })

    return {
        "available": True,
        "title": cfg.get("title", "AI Dark Factory"),
        "description": cfg.get("note", ""),
        "kpis": {
            "total_stages": total_stages,
            "built": built,
            "cataloged": cataloged,
            "planned": planned,
            "total_tools": total_tools,
            "tool_categories": len(tool_categories),
            "planes": len(planes),
            "patterns": len(patterns),
        },
        "flow_status_distribution": flow_status_distribution,
        "tool_status_distribution": tool_status_distribution,
        "tools_per_category": tools_per_category,
        "flow_table": flow_table,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-category tool catalog, planes, and patterns."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "ai_dark_factory.json missing"}

    tools = cfg.get("tool_catalog", [])
    planes_raw = cfg.get("planes", [])
    patterns_raw = cfg.get("patterns", {})

    # Group tools by category
    cat_map = {}
    for t in tools:
        cat = t.get("category", "unknown")
        if cat not in cat_map:
            cat_map[cat] = []
        cat_map[cat].append({
            "tool": t.get("tool", ""),
            "for": t.get("for", ""),
            "status": t.get("status", ""),
            "note": t.get("note", ""),
        })

    per_category = [
        {"category": cat, "tools": tlist}
        for cat, tlist in cat_map.items()
    ]

    # Planes
    planes_out = []
    for p in planes_raw:
        planes_out.append({
            "plane": p.get("plane", ""),
            "components": p.get("components", []),
            "status": p.get("status", ""),
            "note": p.get("note", ""),
        })

    # Patterns
    patterns_out = []
    for pname, pinfo in patterns_raw.items():
        patterns_out.append({
            "id": pname,
            "desc": pinfo.get("desc", ""),
            "best_for": pinfo.get("best_for", ""),
            "failure_mode": pinfo.get("failure_mode", ""),
            "status": pinfo.get("status", ""),
            "note": pinfo.get("note", ""),
        })

    return {
        "available": True,
        "per_category": per_category,
        "planes": planes_out,
        "patterns": patterns_out,
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "description": "Implemented and verified in this project — live code, tested endpoints, working dashboards"},
            {"status": "cataloged", "description": "Tool or stage evaluated and documented in the registry — not yet installed or integrated"},
            {"status": "planned", "description": "Architecture slot reserved — implementation pending 6-gate review"},
            {"status": "adapter", "description": "Integration adapter layer available — connects to external tool via API or SDK"},
        ],
        "glossary": [
            {"term": "BMAD", "definition": "Business/Market Analyst & Designer — multi-agent framework producing PRD, System Design, and Stories from a single idea prompt"},
            {"term": "Archon", "definition": "Deterministic AI workflow controller — orchestrates coding agents with structured, reproducible task pipelines"},
            {"term": "OpenHands", "definition": "Autonomous coding agent — generates code from specifications with built-in sandboxed execution"},
            {"term": "Playwright", "definition": "End-to-end browser testing framework — headless Chrome/Firefox/WebKit automation for UI validation"},
            {"term": "DeepEval", "definition": "LLM and agent evaluation framework — scores model outputs on faithfulness, relevance, hallucination, and custom metrics"},
            {"term": "Temporal", "definition": "Durable workflow execution engine — retries, pause/resume, approval gates with persistent state"},
            {"term": "OTel", "definition": "OpenTelemetry — vendor-neutral observability framework for traces, metrics, and logs across distributed systems"},
            {"term": "OpenLIT", "definition": "OpenTelemetry-native LLM observability platform — tracing, token counting, cost monitoring for AI workloads"},
            {"term": "Harness", "definition": "CI/CD platform — deployment pipeline orchestration with canary, blue-green, and feature-flag release strategies"},
            {"term": "Prefect", "definition": "Python-native workflow orchestration — DAG-free data/AI pipelines with built-in monitoring UI"},
            {"term": "Hub-and-Spoke", "definition": "Architecture pattern where a central orchestrator dispatches work to peripheral worker agents and aggregates results"},
            {"term": "Council", "definition": "Council of Agents pattern — author, reviewer, and chair agents reach consensus through structured multi-round deliberation"},
        ],
        "clinical_notes": [
            "The AI Dark Factory is a REFERENCE architecture — not all stages are installed in this project.",
            "Built components include Playwright testing, Council pattern, Hub-and-Spoke fleet, and Governance plane (guardrails + HITL override).",
            "Adopting any cataloged/planned stage requires a 6-gate review per project governance policy.",
            "The full flow (Idea to Monitoring) represents the target autonomous software factory pipeline for clinical AI systems.",
        ],
        "references": [
            {"ref": "BMAD Framework", "detail": "Multi-agent business analysis framework — Analyst, Architect, Scrum agents for structured requirements engineering"},
            {"ref": "Archon Workflows", "detail": "Deterministic AI coding workflow control — reproducible agent task pipelines with rollback support"},
            {"ref": "OpenTelemetry", "detail": "CNCF observability standard — unified traces, metrics, and logs for distributed systems and LLM workloads"},
            {"ref": "Temporal.io", "detail": "Durable execution engine — fault-tolerant workflows with built-in retry, timeout, and human-approval gates"},
        ],
    }
