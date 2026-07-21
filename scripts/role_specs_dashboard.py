"""Role Specs Dashboard — 17-role specification registry visualization
from config/role_specs.json.
17 roles, priority ratings, field counts, section inventories, build status."""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, "..", "config")


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: role count, field totals, status distribution, priority chart, sections-per-role bar."""
    cfg = _load("role_specs.json")
    if not cfg:
        return {"available": False, "note": "role_specs.json missing"}

    roles = cfg.get("roles", [])
    summary = cfg.get("summary", {})

    status_counts = Counter(r.get("status", "unknown") for r in roles)
    status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    # Priority distribution
    priority_counts = Counter(r.get("priority", "?") for r in roles)
    priority_distribution = [
        {"name": p, "value": c}
        for p, c in sorted(priority_counts.items(), key=lambda x: -x[1])
    ]

    # Sections per role (bar chart)
    sections_per_role = [
        {"name": r.get("role", ""), "value": len(r.get("sections", []))}
        for r in roles
    ]
    sections_per_role.sort(key=lambda x: -x["value"])

    # Field count per role (bar chart) — parse "300-350" → midpoint
    field_counts = []
    for r in roles:
        f = r.get("fields", "0")
        if "-" in str(f):
            parts = str(f).split("-")
            try:
                mid = (int(parts[0]) + int(parts[1])) / 2
            except ValueError:
                mid = 0
        else:
            try:
                mid = float(f)
            except ValueError:
                mid = 0
        field_counts.append({"name": r.get("role", ""), "value": mid, "range": str(f)})
    field_counts.sort(key=lambda x: -x["value"])

    total_sections = sum(len(r.get("sections", [])) for r in roles)

    # Roles table
    roles_table = [
        {
            "role": r.get("role", ""),
            "priority": r.get("priority", ""),
            "fields": r.get("fields", ""),
            "sections": len(r.get("sections", [])),
            "status": r.get("status", ""),
        }
        for r in roles
    ]

    return {
        "available": True,
        "title": cfg.get("title", ""),
        "note": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "summary": {
            "total_roles": len(roles),
            "total_fields_estimate": summary.get("total_fields_estimate", ""),
            "total_sections": total_sections,
            "built": status_counts.get("built", 0),
            "partial": status_counts.get("partial", 0),
            "planned": status_counts.get("planned", 0),
        },
        "status_distribution": status_distribution,
        "priority_distribution": priority_distribution,
        "sections_per_role": sections_per_role,
        "field_counts": field_counts,
        "roles_table": roles_table,
    }


def breakdown():
    """Per-role detail cards: sections list, endpoints, frontend component, notes."""
    cfg = _load("role_specs.json")
    if not cfg:
        return {"available": False, "note": "role_specs.json missing"}

    roles = cfg.get("roles", [])

    per_role = []
    for r in roles:
        endpoints = r.get("endpoints", [])
        api_prefix = r.get("api_prefix", "")
        if api_prefix and not endpoints:
            endpoints = [api_prefix]

        per_role.append({
            "role": r.get("role", ""),
            "priority": r.get("priority", ""),
            "fields": r.get("fields", ""),
            "status": r.get("status", ""),
            "sections": r.get("sections", []),
            "section_count": len(r.get("sections", [])),
            "endpoints": endpoints,
            "frontend": r.get("frontend", r.get("component", r.get("dashboard", ""))),
            "note": r.get("note", ""),
        })

    # Group by priority tier
    high_priority = [r for r in per_role if r["priority"] in ("10/10",)]
    medium_priority = [r for r in per_role if r["priority"] in ("9/10",)]
    other_priority = [r for r in per_role if r["priority"] not in ("10/10", "9/10")]

    return {
        "available": True,
        "per_role": per_role,
        "by_priority": {
            "critical_10": high_priority,
            "high_9": medium_priority,
            "standard": other_priority,
        },
    }


def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "description": "Role workbench live — forms, dashboard, and API endpoints verified"},
            {"status": "partial", "description": "Role data exists but workbench incomplete — some sections pending"},
            {"status": "planned", "description": "Role cataloged — specification defined, implementation not started"},
        ],
        "priority_legend": [
            {"priority": "10/10", "description": "Critical — core clinical/governance role, must be fully operational"},
            {"priority": "9/10", "description": "High — important specialist role with direct patient impact"},
            {"priority": "8/10", "description": "Standard — supporting specialist role for comprehensive care"},
        ],
        "glossary": [
            {"term": "Role Specification", "definition": "Complete definition of a clinical/research role including assessment fields, workflow sections, and AI features"},
            {"term": "Workbench", "definition": "Per-role operational dashboard with patient-centric data, forms, and AI-augmented views"},
            {"term": "HITL", "definition": "Human-in-the-Loop — clinician review/override of AI predictions with audit trail"},
            {"term": "ADL", "definition": "Activities of Daily Living — functional independence measures (Barthel, Katz, Lawton, FIM)"},
            {"term": "AED/ASM", "definition": "Anti-Epileptic Drug / Anti-Seizure Medication — pharmacological seizure management"},
            {"term": "ILAE", "definition": "International League Against Epilepsy — standard seizure classification authority"},
            {"term": "DRE", "definition": "Drug-Resistant Epilepsy — failure of 2+ tolerated ASMs to achieve seizure freedom"},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy — leading cause of epilepsy-related mortality"},
            {"term": "IRB", "definition": "Institutional Review Board — ethics committee for human subjects research oversight"},
            {"term": "RBAC", "definition": "Role-Based Access Control — permission scoping by clinical role"},
            {"term": "XAI", "definition": "Explainable AI — SHAP/Grad-CAM/LIME methods for model transparency"},
            {"term": "PNES", "definition": "Psychogenic Non-Epileptic Seizures — functional seizures requiring psychiatric differential"},
        ],
        "clinical_notes": [
            "The 17-role registry covers the complete multidisciplinary epilepsy care team from diagnosis through long-term management",
            "Field counts (3200-3900 total) represent the structured data capture capacity across all role workbenches",
            "Each role workbench includes AI augmentation under governance: explainability, human override, and audit trail",
            "Priority ratings reflect clinical criticality — 10/10 roles are essential for safe AI-assisted epilepsy care",
        ],
        "references": [
            "ILAE Commission on Classification and Terminology, 2017 — Operational classification of seizure types",
            "NICE Guideline CG137, 2022 — Epilepsies in children, young people and adults: diagnosis and management",
            "Devinsky et al., 2018 — Epilepsy (Nature Reviews Disease Primers)",
            "Kwan et al., 2010 — Definition of drug resistant epilepsy: consensus proposal by the ILAE",
            "FDA, 2021 — Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan",
        ],
    }
