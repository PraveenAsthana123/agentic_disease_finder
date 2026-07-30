"""Role IPO (Input → Process → Output) Pipeline Dashboard.

Shows each clinical role's data pipeline as staged flows with quality gates,
drawn from config/role_process_flows.json and config/role_specs.json.
"""

import json
import pathlib

_CFG = pathlib.Path(__file__).resolve().parent.parent / "config"


def _load(name):
    with open(_CFG / name) as f:
        return json.load(f)


def _role_flows():
    return _load("role_process_flows.json").get("roles", {})


def _role_specs():
    specs = _load("role_specs.json").get("roles", [])
    return {r["role"]: r for r in specs if isinstance(r, dict)}


def overview():
    flows = _role_flows()
    specs = _role_specs()

    roles = []
    total_steps = 0
    for rname, rdata in flows.items():
        steps = rdata.get("steps", [])
        total_steps += len(steps)
        spec = None
        for sname, sdata in specs.items():
            if rname.lower() in sname.lower() or sname.lower() in rname.lower():
                spec = sdata
                break
        roles.append({
            "name": rname,
            "step_count": len(steps),
            "sections": spec.get("sections", []) if spec else [],
            "status": spec.get("status", "planned") if spec else "planned",
            "priority": spec.get("priority", "—") if spec else "—",
            "has_mermaid": bool(rdata.get("mermaid")),
        })

    statuses = {}
    for r in roles:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    return {
        "kpis": [
            {"label": "Total Roles", "value": len(roles), "color": "blue"},
            {"label": "Total Pipeline Steps", "value": total_steps, "color": "blue"},
            {"label": "Built", "value": statuses.get("built", 0), "color": "green"},
            {"label": "Partial", "value": statuses.get("partial", 0), "color": "yellow"},
            {"label": "Planned", "value": statuses.get("planned", 0), "color": "gray"},
            {"label": "Avg Steps/Role", "value": round(total_steps / max(len(roles), 1), 1), "color": "blue"},
        ],
        "roles": roles,
        "status_distribution": [{"name": k, "value": v} for k, v in statuses.items()],
        "honest_note": "Pipeline stages from config/role_process_flows.json; status from config/role_specs.json.",
    }


def breakdown():
    flows = _role_flows()

    pipelines = {}
    for rname, rdata in flows.items():
        steps = rdata.get("steps", [])
        stages = []
        for i, step in enumerate(steps):
            stages.append({
                "index": i,
                "label": step,
                "phase": "input" if i == 0 else ("output" if i == len(steps) - 1 else "process"),
            })
        pipelines[rname] = {
            "stages": stages,
            "mermaid": rdata.get("mermaid", ""),
            "total": len(steps),
        }

    # Cross-role comparison matrix
    all_step_labels = sorted({s for rd in flows.values() for s in rd.get("steps", [])})
    matrix_rows = []
    for rname, rdata in flows.items():
        row = {"role": rname}
        rsteps = set(rdata.get("steps", []))
        for sl in all_step_labels:
            row[sl] = sl in rsteps
        matrix_rows.append(row)

    return {
        "pipelines": pipelines,
        "cross_matrix": {
            "step_labels": all_step_labels,
            "rows": matrix_rows,
        },
    }


def definitions():
    return {
        "phases": [
            {"name": "Input", "description": "Data sources and patient information entering the pipeline."},
            {"name": "Process", "description": "Analytical, AI, and clinical review stages that transform input into actionable findings."},
            {"name": "Output", "description": "Final deliverables — reports, scores, sign-offs, and audit records."},
        ],
        "quality_gates": [
            {"gate": "Data Completeness", "description": "All required input fields present and validated."},
            {"gate": "AI Confidence", "description": "Model prediction meets confidence threshold before proceeding."},
            {"gate": "Human Validation", "description": "Clinician reviews and approves AI-generated findings."},
            {"gate": "Audit Trail", "description": "Every decision logged with timestamp, actor, and rationale."},
        ],
        "status_legend": [
            {"status": "built", "description": "Pipeline fully implemented and verified."},
            {"status": "partial", "description": "Some stages implemented; others pending."},
            {"status": "planned", "description": "Designed but not yet implemented."},
        ],
    }
