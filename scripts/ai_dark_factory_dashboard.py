"""AI Dark Factory Dashboard — autonomous software factory reference architecture.
Surfaces flow stages, tool catalog, architectural patterns, and planes from
config/ai_dark_factory.json."""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: flow stages, built/cataloged/planned counts, tool catalog size,
    patterns implemented, planes active."""
    data = _load('ai_dark_factory.json')
    if not data:
        return {"available": False, "note": "ai_dark_factory.json missing"}

    flow = data.get('full_flow', [])
    flow_status = Counter(s.get('status', '?') for s in flow)

    tools = data.get('tool_catalog', [])
    tool_status = Counter(t.get('status', '?') for t in tools)
    tool_categories = Counter(t.get('category', '?') for t in tools)

    patterns = data.get('patterns', {})
    patterns_built = sum(1 for p in patterns.values() if p.get('status') == 'built')

    planes = data.get('planes', [])
    planes_built = sum(1 for p in planes if p.get('status') == 'built')

    total_stages = len(flow)
    stages_built = flow_status.get('built', 0)
    stages_cataloged = flow_status.get('cataloged', 0)
    stages_planned = flow_status.get('planned', 0)

    kpis = {
        "total_flow_stages": total_stages,
        "stages_built": stages_built,
        "stages_cataloged": stages_cataloged,
        "stages_planned": stages_planned,
        "flow_completion_pct": round(stages_built / total_stages * 100, 1) if total_stages else 0,
        "total_tools": len(tools),
        "tools_planned": tool_status.get('planned', 0),
        "tools_cataloged": tool_status.get('cataloged', 0),
        "patterns_built": patterns_built,
        "total_patterns": len(patterns),
        "planes_built": planes_built,
        "total_planes": len(planes),
    }

    charts = {
        "flow_status_pie": [{"name": k, "value": v} for k, v in flow_status.items()],
        "tool_status_pie": [{"name": k, "value": v} for k, v in tool_status.items()],
        "tool_category_bar": [{"name": k, "value": v} for k, v in tool_categories.most_common()],
    }

    return {"kpis": kpis, "charts": charts}


def breakdown():
    """Full detail: flow stages list, tool catalog, patterns, planes."""
    data = _load('ai_dark_factory.json')
    if not data:
        return {"available": False, "note": "ai_dark_factory.json missing"}

    flow = data.get('full_flow', [])
    tools = data.get('tool_catalog', [])
    patterns = data.get('patterns', {})
    planes = data.get('planes', [])

    patterns_list = []
    for name, info in patterns.items():
        patterns_list.append({
            "name": name,
            "description": info.get('desc', ''),
            "best_for": info.get('best_for', ''),
            "failure_mode": info.get('failure_mode', ''),
            "status": info.get('status', '?'),
            "note": info.get('note', ''),
        })

    return {
        "flow_stages": flow,
        "tool_catalog": tools,
        "patterns": patterns_list,
        "planes": planes,
        "summary": data.get('summary', {}),
    }


def definitions():
    """Terminology for AI Dark Factory concepts."""
    return [
        {"term": "Dark Factory", "definition": "Fully autonomous software factory that runs without human intervention — lights-off manufacturing applied to code."},
        {"term": "Flow Stage", "definition": "One step in the end-to-end autonomous pipeline from idea to deployed, monitored software."},
        {"term": "Built", "definition": "Implemented and running in this project today."},
        {"term": "Cataloged", "definition": "Tool/stage evaluated and documented; not yet integrated into the running system."},
        {"term": "Planned", "definition": "Identified as needed but no integration work started."},
        {"term": "BMAD", "definition": "Business Model Analyst/Designer — AI agents that produce PRDs, system designs, and user stories."},
        {"term": "Archon", "definition": "Deterministic workflow controller that sequences coding tasks for AI agents."},
        {"term": "OpenHands", "definition": "Autonomous coding agent that writes, tests, and iterates on code."},
        {"term": "Temporal", "definition": "Durable workflow engine for long-running processes with retries, pause/resume, and human approval gates."},
        {"term": "Plane", "definition": "An architectural layer grouping related components (e.g., Governance Plane, DevOps Plane)."},
        {"term": "Hub-and-Spoke", "definition": "Pattern where a central orchestrator dispatches work to specialized worker agents."},
        {"term": "Council of Agents", "definition": "Multi-agent consensus pattern: author proposes, reviewer critiques, chair decides."},
        {"term": "6-Gate Review", "definition": "Adoption protocol requiring security, compliance, cost, integration, observability, and rollback gates before adding new tools."},
    ]
