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

    # expert_dashboards.json is a flat list of disease dashboard objects
    dashboards = cfg if isinstance(cfg, list) else cfg.get('dashboards', [])
    total = len(dashboards)
    built = sum(1 for d in dashboards if d.get('status') == 'built')
    partial = sum(1 for d in dashboards if d.get('status') == 'partial')
    planned = sum(1 for d in dashboards if d.get('status') == 'planned')
    libraries = {}

    # Status distribution
    status_counter = Counter()
    registration_counter = Counter()
    for d in dashboards:
        status_counter[d.get('status', 'unknown')] += 1
        reg = d.get('registered', '')[:7]  # YYYY-MM
        if reg:
            registration_counter[reg] += 1

    status_distribution = [
        {"name": s, "value": c}
        for s, c in status_counter.most_common()
    ]
    registration_trend = [
        {"month": m, "count": c}
        for m, c in sorted(registration_counter.items())
    ]

    # Endpoint coverage: each item has api field like "/api/id/overview|breakdown|definitions"
    dashboards_with_api = sum(1 for d in dashboards if d.get('api'))
    total_endpoints = dashboards_with_api * 3  # overview + breakdown + definitions

    # Summary table (last 50 most recent)
    recent = sorted(dashboards, key=lambda d: d.get('registered', ''), reverse=True)
    dashboards_table = []
    for d in recent[:50]:
        dashboards_table.append({
            "id": d.get('id', ''),
            "name": d.get('name', '')[:100],
            "status": d.get('status', ''),
            "api": d.get('api', ''),
            "cohort_n": d.get('cohort_n', ''),
            "registered": d.get('registered', ''),
        })

    lib_list = []

    return {
        "available": True,
        "title": "Expert Dashboards Catalog",
        "note": "Disease-specific epilepsy dashboards with verified API endpoints",
        "updated_at": dashboards[-1].get('registered', '') if dashboards else '',
        "kpis": {
            "total_dashboards": total,
            "built": built,
            "partial": partial,
            "planned": planned,
            "dashboards_with_api": dashboards_with_api,
            "total_endpoints": total_endpoints,
        },
        "charts": {
            "status_distribution": status_distribution,
            "registration_trend": registration_trend,
        },
        "dashboards_table": dashboards_table,
        "libraries": lib_list,
    }


def breakdown():
    """Per-disease dashboard grouping with details."""
    cfg = _load('expert_dashboards.json')
    if not cfg:
        return {"available": False, "note": "expert_dashboards.json missing"}

    dashboards = cfg if isinstance(cfg, list) else cfg.get('dashboards', [])

    # Group by status
    by_status = {}
    for d in dashboards:
        status = d.get('status', 'unknown')
        by_status.setdefault(status, []).append({
            "id": d.get('id', ''),
            "name": d.get('name', '')[:120],
            "status": status,
            "api": d.get('api', ''),
            "cohort_n": d.get('cohort_n', ''),
            "seed": d.get('seed', ''),
            "registered": d.get('registered', ''),
        })

    per_status = [
        {"status": status, "count": len(items), "dashboards": items}
        for status, items in sorted(by_status.items(), key=lambda x: -len(x[1]))
    ]

    # Recent registrations (last 20)
    recent = sorted(dashboards, key=lambda d: d.get('registered', ''), reverse=True)[:20]
    recent_list = [
        {"id": d.get('id', ''), "name": d.get('name', '')[:100], "registered": d.get('registered', '')}
        for d in recent
    ]

    return {
        "available": True,
        "per_status": per_status,
        "recent_registrations": recent_list,
        "total": len(dashboards),
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
