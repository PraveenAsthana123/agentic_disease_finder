"""Production Issues Dashboard — enterprise agentic AI issue catalog analytics
from config/production_issues.json (16 layers, ~37 issues, severity + detection status)."""

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


def _det_status(raw):
    """Normalize detected_in_project to built/partial/planned."""
    if not raw:
        return 'planned'
    r = raw.strip().lower()
    if r.startswith('built'):
        return 'built'
    elif r.startswith('partial'):
        return 'partial'
    return 'planned'


def overview():
    """Summary KPIs: issue counts by severity, detection status, layer distribution,
    top Pareto issues, internal flow."""
    cfg = _load('production_issues.json')
    if not cfg:
        return {"available": False, "note": "production_issues.json missing"}

    layers = cfg.get('layers', [])
    total_layers = len(layers)
    all_issues = []
    for l in layers:
        for iss in l.get('issues', []):
            iss['_layer'] = l['layer']
            all_issues.append(iss)
    total_issues = len(all_issues)

    # Severity counts
    sev_cnt = Counter(i.get('severity', '?') for i in all_issues)
    p1 = sev_cnt.get('P1', 0)
    p2 = sev_cnt.get('P2', 0)

    # Detection status counts
    det_cnt = Counter(_det_status(i.get('detected_in_project', '')) for i in all_issues)
    built = det_cnt.get('built', 0)
    partial = det_cnt.get('partial', 0)
    planned = det_cnt.get('planned', 0)
    coverage_pct = round((built + partial) / total_issues * 100, 1) if total_issues else 0

    # Issues per layer
    layer_dist = []
    for l in layers:
        issues = l.get('issues', [])
        lp1 = sum(1 for i in issues if i.get('severity') == 'P1')
        lp2 = sum(1 for i in issues if i.get('severity') == 'P2')
        lb = sum(1 for i in issues if _det_status(i.get('detected_in_project', '')) == 'built')
        lpa = sum(1 for i in issues if _det_status(i.get('detected_in_project', '')) == 'partial')
        lpl = sum(1 for i in issues if _det_status(i.get('detected_in_project', '')) == 'planned')
        layer_dist.append({
            "layer": l['layer'],
            "total": len(issues),
            "p1": lp1,
            "p2": lp2,
            "built": lb,
            "partial": lpa,
            "planned": lpl,
        })

    # Severity distribution for pie chart
    severity_distribution = [{"name": k, "value": v} for k, v in sev_cnt.items()]

    # Detection distribution for pie chart
    detection_distribution = [
        {"name": "Built", "value": built},
        {"name": "Partial", "value": partial},
        {"name": "Planned", "value": planned},
    ]

    # Top Pareto issues
    pareto = cfg.get('top_20_pct_cause_80_pct', [])

    return {
        "available": True,
        "summary": {
            "total_issues": total_issues,
            "total_layers": total_layers,
            "p1": p1,
            "p2": p2,
            "built": built,
            "partial": partial,
            "planned": planned,
            "coverage_pct": coverage_pct,
        },
        "severity_distribution": severity_distribution,
        "detection_distribution": detection_distribution,
        "layer_distribution": layer_dist,
        "pareto_issues": pareto,
        "internal_flow": cfg.get('internal_flow', ''),
    }


def breakdown():
    """Detailed view: all issues per layer with severity, root cause, detection,
    solution, project detection status."""
    cfg = _load('production_issues.json')
    if not cfg:
        return {"available": False}

    layers = cfg.get('layers', [])
    layer_details = []
    for l in layers:
        issues = []
        for iss in l.get('issues', []):
            issues.append({
                "issue": iss.get('issue', '?'),
                "severity": iss.get('severity', '?'),
                "root_cause": iss.get('root_cause', ''),
                "detection": iss.get('detection', ''),
                "solution": iss.get('solution', ''),
                "detected_in_project": iss.get('detected_in_project', ''),
                "det_status": _det_status(iss.get('detected_in_project', '')),
            })
        layer_details.append({
            "layer": l['layer'],
            "issues": issues,
        })

    return {
        "available": True,
        "layers": layer_details,
        "meta": {
            "title": cfg.get('title', ''),
            "note": cfg.get('note', ''),
            "updated_at": cfg.get('updated_at', ''),
        },
    }


def definitions():
    """Production issues terminology and definitions."""
    return {
        "available": True,
        "definitions": [
            {"term": "P1 (Critical)", "definition": "Highest severity — issue can cause data loss, security breach, patient safety risk, or complete system failure. Requires immediate mitigation."},
            {"term": "P2 (Major)", "definition": "Significant issue that degrades quality, reliability, or performance but does not cause immediate safety or data risks. Should be addressed in current sprint."},
            {"term": "Built", "definition": "Detection/mitigation for this issue is fully implemented and operational in the project (has live endpoint, dashboard, or automated check)."},
            {"term": "Partial", "definition": "Some detection or mitigation exists but is incomplete — may cover only a subset of scenarios or lack automation."},
            {"term": "Planned", "definition": "Issue is cataloged and acknowledged but no detection or mitigation is implemented yet."},
            {"term": "Layer", "definition": "Architectural tier where the issue manifests — e.g., Agent, Security, MCP, Planner, Retrieval, Memory, Reasoning, Workflow, LLM, Governance, etc."},
            {"term": "Root Cause", "definition": "The underlying technical reason the issue occurs — helps prioritize which fix approach to take."},
            {"term": "Detection", "definition": "The method or metric used to identify the issue at runtime — e.g., health checks, grounding scores, audit logs."},
            {"term": "Solution", "definition": "The recommended mitigation strategy — e.g., guardrails, consensus voting, circuit breakers, HITL gates."},
            {"term": "Pareto Issues (top 20%)", "definition": "The ~10 issues that cause ~80% of production incidents in agentic AI systems. Addressing these first yields the highest reliability gain."},
            {"term": "Internal Flow", "definition": "The request processing path through all layers: User → Gateway → Router → Planner → Security → Memory → RAG → Specialist → Evaluation → Reviewer → Council → Compliance → Audit → Response."},
            {"term": "Coverage %", "definition": "Percentage of cataloged issues that have at least partial detection/mitigation implemented (built + partial out of total)."},
        ],
        "clinical_notes": [
            "In clinical AI systems, P1 issues (hallucination, PII leakage, unauthorized actions) can directly impact patient safety.",
            "The Pareto principle applies: a small number of issue types cause the majority of production incidents.",
            "Coverage percentage tracks readiness — 100% means every known issue has at least partial detection.",
            "Layer-based categorization helps assign ownership: Security team owns injection/PII, ML team owns drift/hallucination, Platform team owns infra/deployment.",
        ],
        "references": [
            "OWASP Top 10 for LLM Applications (2025) — security risk taxonomy for LLM-based systems.",
            "Anthropic (2024). Challenges in evaluating AI systems. Technical report.",
            "NIST AI Risk Management Framework (AI RMF 1.0) — governance and risk categories.",
            "Microsoft (2024). Failure Modes in Machine Learning. Azure AI documentation.",
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
