"""Expert Roles Dashboard — 8-role multidisciplinary care team visualization
from config/expert_roles.json.
8 roles, 35 tasks, all built, AI features, steps, challenges, endpoints."""

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
    """Summary KPIs: role count, task totals, status distribution, tasks-per-role bar,
    dashboard-status distribution, roles table, steps/challenges/endpoints totals."""
    cfg = _load("expert_roles.json")
    if not cfg:
        return {"available": False, "note": "expert_roles.json missing"}

    roles = cfg.get("roles", [])
    summary = cfg.get("summary", {})

    # Collect all tasks across roles
    all_tasks = []
    for r in roles:
        all_tasks.extend(r.get("tasks", []))

    # Task status distribution (pie chart)
    status_counts = Counter(t.get("status", "unknown") for t in all_tasks)
    status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    # Tasks per role (bar chart)
    tasks_per_role = [
        {"name": r.get("role", ""), "value": len(r.get("tasks", []))}
        for r in roles
    ]
    tasks_per_role.sort(key=lambda x: -x["value"])

    # Dashboard status distribution across roles (pie chart)
    dash_counts = Counter(r.get("dashboard_status", "not set") for r in roles)
    dashboard_status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(dash_counts.items(), key=lambda x: -x[1])
    ]

    # Aggregate step / challenge / endpoint counts
    total_steps = sum(
        len(t.get("steps", []))
        for r in roles
        for t in r.get("tasks", [])
    )
    total_challenges = sum(
        len(t.get("challenges", []))
        for r in roles
        for t in r.get("tasks", [])
    )
    total_endpoints = sum(
        len(t.get("endpoints", []))
        for r in roles
        for t in r.get("tasks", [])
    )

    # Roles summary table
    roles_table = []
    for r in roles:
        tasks = r.get("tasks", [])
        built_count = sum(1 for t in tasks if t.get("status") == "built")
        has_endpoints = any(t.get("endpoints") for t in tasks)
        roles_table.append({
            "role": r.get("role", ""),
            "icon": r.get("icon", ""),
            "mission_short": r.get("mission", "")[:60],
            "task_count": len(tasks),
            "built_count": built_count,
            "dashboard_status": r.get("dashboard_status", ""),
            "has_endpoints": has_endpoints,
        })

    return {
        "available": True,
        "title": cfg.get("title", ""),
        "note": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "summary": {
            "total_roles": len(roles),
            "total_tasks": len(all_tasks),
            "built_tasks": status_counts.get("built", 0),
            "partial_tasks": status_counts.get("partial", 0),
            "planned_tasks": status_counts.get("planned", 0),
            "roles_with_dashboards": dash_counts.get("built", 0),
        },
        "status_distribution": status_distribution,
        "tasks_per_role": tasks_per_role,
        "dashboard_status_distribution": dashboard_status_distribution,
        "roles_table": roles_table,
        "total_steps": total_steps,
        "total_challenges": total_challenges,
        "total_endpoints": total_endpoints,
    }


def breakdown():
    """Per-role detail: mission, data_hook, dashboard, all tasks with steps/challenges/endpoints."""
    cfg = _load("expert_roles.json")
    if not cfg:
        return {"available": False, "note": "expert_roles.json missing"}

    roles = cfg.get("roles", [])

    per_role = []
    for r in roles:
        tasks_raw = r.get("tasks", [])
        tasks = []
        role_endpoint_count = 0
        for t in tasks_raw:
            endpoints = t.get("endpoints", [])
            role_endpoint_count += len(endpoints)
            tasks.append({
                "name": t.get("name", ""),
                "ai_feature": t.get("ai_feature", ""),
                "status": t.get("status", ""),
                "steps": t.get("steps", []),
                "challenges": t.get("challenges", []),
                "endpoints": endpoints,
                "step_count": len(t.get("steps", [])),
                "challenge_count": len(t.get("challenges", [])),
            })

        per_role.append({
            "role": r.get("role", ""),
            "icon": r.get("icon", ""),
            "mission": r.get("mission", ""),
            "data_hook": r.get("data_hook", ""),
            "dashboard_status": r.get("dashboard_status", ""),
            "dashboard_component": r.get("dashboard_component", ""),
            "tasks": tasks,
            "task_count": len(tasks),
            "endpoint_count": role_endpoint_count,
        })

    return {
        "available": True,
        "per_role": per_role,
    }


def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "description": "Task live — real data path, AI feature active, endpoint verified"},
            {"status": "partial", "description": "Task partially live — some data exists but pipeline incomplete"},
            {"status": "planned", "description": "Task cataloged — specification defined, implementation not started"},
        ],
        "glossary": [
            {"term": "ADL", "definition": "Activities of Daily Living — functional independence measures (Barthel, Katz, Lawton, FIM)"},
            {"term": "AED/ASM", "definition": "Anti-Epileptic Drug / Anti-Seizure Medication — pharmacological seizure management"},
            {"term": "ILAE", "definition": "International League Against Epilepsy — standard seizure classification and terminology authority"},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy — leading cause of epilepsy-related mortality; requires proactive counseling"},
            {"term": "CBT", "definition": "Cognitive Behavioral Therapy — evidence-based psychotherapy for depression, anxiety, and coping in epilepsy"},
            {"term": "SDOH", "definition": "Social Determinants of Health — non-medical factors (housing, finance, transport) affecting health outcomes"},
            {"term": "MDT", "definition": "Multidisciplinary Team — coordinated care model bringing all expert roles together for case decisions"},
            {"term": "MMAS-8", "definition": "Morisky Medication Adherence Scale (8-item) — validated self-report tool for medication adherence"},
            {"term": "ZBI", "definition": "Zarit Burden Interview — standard caregiver burden assessment (22 items, 0-88 score)"},
            {"term": "BNT", "definition": "Boston Naming Test — standardized confrontation naming test for language/aphasia assessment"},
            {"term": "QOLIE", "definition": "Quality of Life in Epilepsy — disease-specific QoL instrument (QOLIE-31 / QOLIE-89)"},
            {"term": "MAD", "definition": "Modified Atkins Diet — less restrictive ketogenic variant for seizure control in adults and adolescents"},
        ],
        "clinical_notes": [
            "The 8-role expert team extends beyond the Neurologist/CDM to provide comprehensive biopsychosocial epilepsy care",
            "All 35 tasks are fully built with real data paths — AI features operate under HITL governance with human override",
            "Multidisciplinary coordination (MDT/Coordinator role) is essential for DRE surgical candidacy evaluation and long-term follow-up",
            "Social determinants (MSW), caregiver burden (ZBI), and quality of life (QOLIE) are systematically captured alongside clinical metrics",
        ],
        "references": [
            "ILAE Commission on Classification and Terminology, 2017 — Operational classification of seizure types",
            "NICE Guideline CG137, 2022 — Epilepsies in children, young people and adults: diagnosis and management",
            "Devinsky et al., 2018 — Epilepsy (Nature Reviews Disease Primers)",
            "WHO, 2022 — Epilepsy fact sheet — burden, treatment gap, and social consequences",
            "AAN, 2018 — Practice guideline update: Efficacy and tolerability of the new antiepileptic drugs I & II",
        ],
    }
