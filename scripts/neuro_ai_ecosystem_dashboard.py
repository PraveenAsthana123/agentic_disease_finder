"""Neuro AI Ecosystem Dashboard — open-source tool catalog visualization
from config/neuro_ai_ecosystem.json.
64 tools, 10 categories, 4 status levels (built/installed/external/commercial)."""

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
    """Summary KPIs: tool counts, status distribution, category breakdown, recommended stack."""
    cfg = _load('neuro_ai_ecosystem.json')
    if not cfg:
        return {"available": False, "note": "neuro_ai_ecosystem.json missing"}

    categories = cfg.get('categories', [])
    summary = cfg.get('summary', {})
    rec_stack = cfg.get('recommended_stack', {})

    # Flatten all tools
    all_tools = []
    for cat in categories:
        for t in cat.get('tools', []):
            t['_category'] = cat.get('category', '')
            all_tools.append(t)

    status_counts = Counter(t.get('status', 'unknown') for t in all_tools)
    status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    cat_counts = [
        {"name": cat.get('category', ''), "value": len(cat.get('tools', []))}
        for cat in categories
    ]
    cat_counts.sort(key=lambda x: -x['value'])

    # Rating distribution (only tools with ratings)
    rated = [t for t in all_tools if t.get('rating')]
    rating_counts = Counter(t.get('rating') for t in rated)
    rating_distribution = [
        {"name": f"{r} stars", "value": c}
        for r, c in sorted(rating_counts.items(), key=lambda x: -x[0])
    ]

    tools_table = [
        {
            "name": t.get('name', ''),
            "category": t.get('_category', ''),
            "purpose": t.get('purpose', t.get('disease', '')),
            "status": t.get('status', ''),
            "rating": t.get('rating'),
        }
        for t in all_tools
    ]

    return {
        "available": True,
        "title": cfg.get('title', ''),
        "note": cfg.get('note', ''),
        "updated_at": cfg.get('updated_at', ''),
        "summary": {
            "total_tools": len(all_tools),
            "categories": len(categories),
            "built": status_counts.get('built', 0),
            "installed": status_counts.get('installed', 0),
            "external": status_counts.get('external', 0),
            "commercial": status_counts.get('commercial', 0),
        },
        "status_distribution": status_distribution,
        "category_distribution": cat_counts,
        "rating_distribution": rating_distribution,
        "recommended_stack": rec_stack,
        "tools_table": tools_table,
    }


def breakdown():
    """Per-category tool details, tools with endpoints, recommended stack mapping."""
    cfg = _load('neuro_ai_ecosystem.json')
    if not cfg:
        return {"available": False, "note": "neuro_ai_ecosystem.json missing"}

    categories = cfg.get('categories', [])

    per_category = []
    for cat in categories:
        tools = cat.get('tools', [])
        statuses = Counter(t.get('status', 'unknown') for t in tools)
        per_category.append({
            "category": cat.get('category', ''),
            "note": cat.get('note', ''),
            "tool_count": len(tools),
            "status_counts": dict(statuses),
            "tools": [
                {
                    "name": t.get('name', ''),
                    "purpose": t.get('purpose', t.get('disease', '')),
                    "status": t.get('status', ''),
                    "rating": t.get('rating'),
                    "domain": t.get('domain', ''),
                    "endpoints": t.get('endpoints', t.get('endpoint', '')),
                }
                for t in tools
            ],
        })

    # Tools with live endpoints
    tools_with_endpoints = []
    for cat in categories:
        for t in cat.get('tools', []):
            eps = t.get('endpoints', [])
            if not eps:
                ep = t.get('endpoint', '')
                if ep:
                    eps = [ep]
            if eps:
                tools_with_endpoints.append({
                    "name": t.get('name', ''),
                    "category": cat.get('category', ''),
                    "endpoints": eps,
                    "status": t.get('status', ''),
                })

    return {
        "available": True,
        "per_category": per_category,
        "tools_with_endpoints": tools_with_endpoints,
    }


def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "description": "Live in this platform — endpoints verified, dashboard functional"},
            {"status": "installed", "description": "Python package installed — import verified, not yet dashboard-wired"},
            {"status": "external", "description": "Separate server/desktop/MATLAB tool — recommended, not embedded"},
            {"status": "commercial", "description": "Commercial license required — cataloged for reference only"},
        ],
        "glossary": [
            {"term": "eCRF", "definition": "Electronic Case Report Form — structured clinical data capture"},
            {"term": "EDC", "definition": "Electronic Data Capture — digital clinical trial data collection"},
            {"term": "REDCap", "definition": "Research Electronic Data Capture — NIH-funded clinical assessment platform"},
            {"term": "MNE-Python", "definition": "Open-source EEG/MEG analysis library for Python"},
            {"term": "ICLabel", "definition": "ICA component classifier — brain/muscle/eye/heart/line/channel/other"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — model-agnostic feature importance"},
            {"term": "Captum", "definition": "PyTorch model interpretability library (Meta)"},
            {"term": "AIF360", "definition": "AI Fairness 360 — IBM bias detection and mitigation toolkit"},
            {"term": "Great Expectations", "definition": "Data validation framework — schema + statistical quality checks"},
            {"term": "Evidently AI", "definition": "ML model monitoring — data drift, target drift, model performance"},
            {"term": "Label Studio", "definition": "Open-source data labeling platform — signal, image, text annotation"},
            {"term": "CVAT", "definition": "Computer Vision Annotation Tool — video/image labeling (Intel)"},
            {"term": "Grad-CAM", "definition": "Gradient-weighted Class Activation Mapping — CNN visualization"},
        ],
        "clinical_notes": [
            "All 'built' tools have verified API endpoints in this platform",
            "'Installed' tools are importable but not yet exposed via dashboard endpoints",
            "'External' tools run on separate infrastructure (REDCap server, MATLAB, etc.)",
            "The recommended stack represents the minimum viable open-source neuro AI platform",
        ],
        "references": [
            "MNE-Python: Gramfort et al., 2013 — MEG and EEG data analysis with MNE-Python",
            "REDCap: Harris et al., 2009 — Research electronic data capture (REDCap)",
            "SHAP: Lundberg & Lee, 2017 — A Unified Approach to Interpreting Model Predictions",
            "AIF360: Bellamy et al., 2019 — AI Fairness 360: An Extensible Toolkit",
            "Great Expectations: https://greatexpectations.io — open-source data quality",
        ],
    }
