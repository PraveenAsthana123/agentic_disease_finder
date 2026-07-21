"""Expert Dashboards Catalog Dashboard — 36-dashboard, multi-role expert
dashboard catalog visualization from config/expert_dashboards.json.
Roles: Neurologist, Epileptologist, Neurophysiologist, Neuropsychologist,
Psychiatrist, Researcher, AI Team, AI Governance, MLOps, Admin, Caregiver."""

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
    """Summary KPIs: total dashboards, status, roles, priorities, charts."""
    cfg = _load('expert_dashboards.json')
    if not cfg:
        return {"available": False, "note": "expert_dashboards.json missing"}

    dashboards = cfg.get('dashboards', [])
    summary = cfg.get('summary', {})
    total = summary.get('total', len(dashboards))
    built = summary.get('built', 0)
    partial = summary.get('partial', 0)
    planned = summary.get('planned', 0)
    must_have = cfg.get('must_have_p0', [])
    libraries = cfg.get('libraries', {})

    # Role distribution
    role_counter = Counter()
    priority_counter = Counter()
    status_counter = Counter()
    for d in dashboards:
        role = d.get('role', d.get('label', 'Unknown'))
        role_counter[role] += 1
        priority_counter[d.get('priority', 'N/A')] += 1
        status_counter[d.get('status', 'unknown')] += 1

    role_distribution = [
        {"name": r, "value": c}
        for r, c in role_counter.most_common()
    ]
    priority_distribution = [
        {"name": p, "value": c}
        for p, c in sorted(priority_counter.items())
    ]
    status_distribution = [
        {"name": s, "value": c}
        for s, c in status_counter.most_common()
    ]

    unique_roles = len(role_counter)

    # Dashboards per role bar chart
    dashboards_per_role = [
        {"name": r, "value": c}
        for r, c in role_counter.most_common()
    ]

    # Endpoint coverage
    total_endpoints = 0
    dashboards_with_endpoints = 0
    for d in dashboards:
        eps = d.get('endpoints', [])
        if eps:
            dashboards_with_endpoints += 1
            total_endpoints += len(eps)

    # Summary table
    dashboards_table = []
    for d in dashboards:
        dashboards_table.append({
            "name": d.get('name', d.get('title', d.get('label', ''))),
            "role": d.get('role', ''),
            "viz": d.get('viz', ''),
            "priority": d.get('priority', 'N/A'),
            "status": d.get('status', ''),
            "endpoints": len(d.get('endpoints', [])),
        })

    # Libraries list
    lib_list = [
        {"purpose": k, "library": v}
        for k, v in libraries.items()
    ]

    return {
        "available": True,
        "title": cfg.get('title', 'Expert Dashboards Catalog'),
        "note": cfg.get('note', ''),
        "updated_at": cfg.get('updated_at', ''),
        "kpis": {
            "total_dashboards": total,
            "built": built,
            "partial": partial,
            "planned": planned,
            "unique_roles": unique_roles,
            "total_endpoints": total_endpoints,
            "dashboards_with_endpoints": dashboards_with_endpoints,
            "libraries_count": len(libraries),
        },
        "charts": {
            "status_distribution": status_distribution,
            "role_distribution": role_distribution,
            "priority_distribution": priority_distribution,
            "dashboards_per_role": dashboards_per_role,
        },
        "dashboards_table": dashboards_table,
        "libraries": lib_list,
    }


def breakdown():
    """Per-role dashboard grouping with details."""
    cfg = _load('expert_dashboards.json')
    if not cfg:
        return {"available": False, "note": "expert_dashboards.json missing"}

    dashboards = cfg.get('dashboards', [])

    # Group by role
    by_role = {}
    for d in dashboards:
        role = d.get('role', d.get('label', 'Unknown'))
        by_role.setdefault(role, []).append({
            "name": d.get('name', d.get('title', d.get('label', ''))),
            "viz": d.get('viz', ''),
            "feature": d.get('feature', ''),
            "why": d.get('why', ''),
            "priority": d.get('priority', 'N/A'),
            "status": d.get('status', ''),
            "endpoints": d.get('endpoints', []),
            "component": d.get('component', ''),
            "tabs": d.get('tabs', []),
        })

    per_role = [
        {"role": role, "count": len(items), "dashboards": items}
        for role, items in sorted(by_role.items(), key=lambda x: -len(x[1]))
    ]

    # Must-have P0 items
    must_have = cfg.get('must_have_p0', [])
    must_have_list = []
    for m in must_have:
        if isinstance(m, str):
            must_have_list.append({"name": m, "type": "visualization"})
        elif isinstance(m, dict):
            must_have_list.append({
                "name": m.get('name', ''),
                "status": m.get('status', ''),
                "note": m.get('note', ''),
                "type": "dashboard",
            })

    return {
        "available": True,
        "per_role": per_role,
        "must_have_p0": must_have_list,
        "libraries": cfg.get('libraries', {}),
    }


def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "color": "#22c55e",
             "meaning": "Dashboard is live with verified API endpoints and real data"},
            {"status": "partial", "color": "#f97316",
             "meaning": "Data is present but visualization is basic or incomplete"},
            {"status": "planned", "color": "#3b82f6",
             "meaning": "Dashboard is on the roadmap but not yet implemented"},
        ],
        "glossary": [
            {"term": "Expert Dashboard", "definition": "A role-specific clinical visualization panel tailored to a specialist's workflow needs"},
            {"term": "P0", "definition": "Highest clinical priority — must-have for safe EEG interpretation"},
            {"term": "P1", "definition": "High priority — important for comprehensive clinical workflow"},
            {"term": "P2", "definition": "Medium priority — enhances clinical depth but not critical for basic workflow"},
            {"term": "Topomap", "definition": "Topographic scalp map showing spatial distribution of EEG power across the 10-20 electrode system"},
            {"term": "Spectrogram", "definition": "Time-frequency representation showing how spectral content of EEG evolves over time"},
            {"term": "PSD", "definition": "Power Spectral Density — frequency decomposition of EEG signal power"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — game-theoretic approach to explain individual AI predictions"},
            {"term": "Montage", "definition": "Electrode referencing scheme (bipolar, referential, average) affecting EEG display"},
            {"term": "Lateralization", "definition": "Determining which brain hemisphere shows predominant abnormal activity"},
            {"term": "HITL", "definition": "Human-In-The-Loop — clinician review before AI decisions are finalized"},
            {"term": "ROC/PR", "definition": "Receiver Operating Characteristic / Precision-Recall curves for model evaluation"},
        ],
        "clinical_notes": [
            "All 36 dashboards are built and functional — covering the full EEG-AI clinical lifecycle",
            "P0 dashboards (waveform, timeline, artifact, PSD, spectrogram, topomap, spike overlay) are critical for safe interpretation",
            "MNE-Python provides the scientific EEG processing backbone; Recharts/Plotly provide the visualization layer",
            "HITL (Human-In-The-Loop) panels ensure all AI predictions receive clinical review before action",
        ],
        "references": [
            {"label": "expert_dashboards.json", "detail": "Master catalog of all expert/clinical dashboards and their build status"},
            {"label": "MNE-Python", "detail": "Open-source Python package for EEG/MEG data analysis and visualization"},
            {"label": "ILAE", "detail": "International League Against Epilepsy — classification and diagnosis standards"},
            {"label": "ACNS", "detail": "American Clinical Neurophysiology Society — EEG reporting guidelines"},
        ],
    }
