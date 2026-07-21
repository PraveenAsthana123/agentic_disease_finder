"""Admin Module Dashboard — Team Roles + Ops Dashboards + Access Control + Integrations
from config/admin_module.json.
7 team roles, 10 ops dashboards, 7 access control items, 9 integrations."""

import json
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "admin_module.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        raw = json.load(f)
    return raw[0] if isinstance(raw, list) else raw


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: roles, dashboards, access control, integrations, charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "admin_module.json missing"}

    roles = cfg.get("team_roles", [])
    ops = cfg.get("ops_dashboards", [])
    acl = cfg.get("access_control", [])
    integ = cfg.get("integrations", [])

    # Count statuses
    def status_counts(items):
        counts = {}
        for it in items:
            s = it.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    role_counts = status_counts(roles)
    ops_counts = status_counts(ops)
    acl_counts = status_counts(acl)
    integ_counts = status_counts(integ)

    # KPIs
    total_roles = len(roles)
    total_ops = len(ops)
    total_acl = len(acl)
    total_integ = len(integ)
    built_roles = role_counts.get("built", 0)
    built_ops = ops_counts.get("built", 0)
    built_acl = acl_counts.get("built", 0)
    planned_integ = integ_counts.get("planned", 0)

    # Role status pie chart
    role_status_dist = [{"name": k.title(), "value": v} for k, v in role_counts.items() if v > 0]

    # Ops dashboard status pie chart
    ops_status_dist = [{"name": k.title(), "value": v} for k, v in ops_counts.items() if v > 0]

    # Responsibilities per role bar chart
    resp_per_role = []
    for r in roles:
        resp_per_role.append({
            "name": r.get("role", ""),
            "value": len(r.get("owns", [])),
            "icon": r.get("icon", "")
        })

    # Integration status pie chart
    integ_status_dist = [{"name": k.title(), "value": v} for k, v in integ_counts.items() if v > 0]

    # Access control status pie chart
    acl_status_dist = [{"name": k.title(), "value": v} for k, v in acl_counts.items() if v > 0]

    return {
        "available": True,
        "title": cfg.get("title", "Admin Module"),
        "note": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "kpis": {
            "total_roles": total_roles,
            "built_roles": built_roles,
            "total_ops_dashboards": total_ops,
            "built_ops": built_ops,
            "total_access_control": total_acl,
            "built_acl": built_acl,
            "total_integrations": total_integ,
            "planned_integrations": planned_integ,
        },
        "charts": {
            "role_status_distribution": role_status_dist,
            "ops_status_distribution": ops_status_dist,
            "responsibilities_per_role": resp_per_role,
            "integration_status_distribution": integ_status_dist,
            "access_control_status_distribution": acl_status_dist,
        },
        "summary_table": [
            {
                "role": r.get("role", ""),
                "icon": r.get("icon", ""),
                "owns": r.get("owns", []),
                "status": r.get("status", ""),
                "maps_to": r.get("maps_to", ""),
            }
            for r in roles
        ],
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-section breakdown: roles, ops dashboards, access control, integrations."""
    cfg = _load()
    if not cfg:
        return {"available": False}

    return {
        "available": True,
        "team_roles": cfg.get("team_roles", []),
        "ops_dashboards": cfg.get("ops_dashboards", []),
        "access_control": cfg.get("access_control", []),
        "integrations": cfg.get("integrations", []),
        "integration_note": cfg.get("integration_note", ""),
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "status_legend": [
            {"status": "built", "meaning": "Live — code, endpoints, and UI exist in this project"},
            {"status": "partial", "meaning": "Data present but visualization or full workflow incomplete"},
            {"status": "planned", "meaning": "Designed but not yet implemented"},
            {"status": "n/a", "meaning": "Not applicable for current project scope"},
        ],
        "glossary": [
            {"term": "RBAC", "definition": "Role-Based Access Control — permissions assigned by role (Neurologist, Tech, Admin, Reviewer)"},
            {"term": "ABAC", "definition": "Attribute-Based Access Control — permissions based on attributes (tenant, department, clearance level)"},
            {"term": "DevOps", "definition": "Development + Operations — CI/CD, deploy frequency, change-fail rate, MTTR"},
            {"term": "SecOps", "definition": "Security Operations — threat detection, injection/jailbreak prevention, access audit"},
            {"term": "FinOps", "definition": "Financial Operations — token/GPU/cloud cost tracking per request/user/model"},
            {"term": "MLOps", "definition": "Machine Learning Operations — training pipeline, experiment tracking, feature store"},
            {"term": "LLMOps", "definition": "Large Language Model Operations — prompt versions, token cost, hallucination detection"},
            {"term": "DataOps", "definition": "Data Operations — ingestion, quality, lineage, vector ingest pipelines"},
            {"term": "MCP", "definition": "Model Context Protocol — standardized tool protocol for external integrations"},
            {"term": "PII", "definition": "Personally Identifiable Information — detection, masking, redaction"},
            {"term": "MTTR", "definition": "Mean Time To Recovery — average time to restore service after failure"},
            {"term": "WAL", "definition": "Write-Ahead Log — database durability mechanism for crash recovery"},
        ],
        "clinical_notes": [
            "All team roles are operational for a single-study epilepsy EEG research project",
            "External integrations (Slack, Gmail, Drive) are designed but need OAuth credentials",
            "ABAC is not applicable for single-tenant; RBAC covers all access control needs",
            "Each ops dashboard maps to real endpoints verified in this project",
        ],
        "references": [
            "config/admin_module.json — source registry",
            "RBAC — Role-Based Access Control (NIST SP 800-162)",
            "MCP — Model Context Protocol (Anthropic)",
            "OWASP — Open Web Application Security Project",
        ],
    }
