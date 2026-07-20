"""Feature Gaps Dashboard — Epilepsy DL review (50 papers) vs project gap
analysis from config/feature_gaps.json. Categories: functional, technology,
data, gap, architecture, decision_ai."""

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
    """Summary KPIs: total gaps, category distribution, built coverage."""
    cfg = _load('feature_gaps.json')
    if not cfg:
        return {"available": False, "note": "feature_gaps.json missing"}

    gaps = cfg.get('gaps', [])
    total = len(gaps)

    cat_counts = Counter(g.get('category', 'unknown') for g in gaps)
    priority_counts = Counter(g.get('priority', 'unknown') for g in gaps)
    status_counts = Counter(g.get('in_project', 'unknown') for g in gaps)

    built = status_counts.get('built', 0)
    built_pct = round(built / total * 100, 1) if total else 0

    category_distribution = [
        {"name": cat, "value": cnt}
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])
    ]

    priority_distribution = [
        {"name": p, "value": priority_counts.get(p, 0)}
        for p in ['high', 'medium', 'low']
        if priority_counts.get(p, 0) > 0
    ]

    review_covers = cfg.get('review_covers', [])
    top_recommendations = cfg.get('top_recommendations', [])

    return {
        "available": True,
        "summary": {
            "total_gaps": total,
            "built": built,
            "built_pct": built_pct,
            "categories": len(cat_counts),
            "high_priority": priority_counts.get('high', 0),
            "medium_priority": priority_counts.get('medium', 0),
            "low_priority": priority_counts.get('low', 0),
            "review_papers": 50,
            "review_covers_count": len(review_covers),
            "top_recommendations_count": len(top_recommendations),
        },
        "category_distribution": category_distribution,
        "priority_distribution": priority_distribution,
        "gap_table": [
            {
                "category": g.get('category', ''),
                "feature": g.get('feature', ''),
                "in_project": g.get('in_project', ''),
                "priority": g.get('priority', ''),
            }
            for g in gaps
        ],
    }


def breakdown():
    """Per-category breakdown with full detail including why/dashboard."""
    cfg = _load('feature_gaps.json')
    if not cfg:
        return {"available": False, "note": "feature_gaps.json missing"}

    gaps = cfg.get('gaps', [])

    categories = {}
    for g in gaps:
        cat = g.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "feature": g.get('feature', ''),
            "in_project": g.get('in_project', ''),
            "priority": g.get('priority', ''),
            "why": g.get('why', ''),
            "dashboard": g.get('dashboard', ''),
        })

    per_category = []
    for cat, items in categories.items():
        built = sum(1 for i in items if i['in_project'] == 'built')
        per_category.append({
            "category": cat,
            "total": len(items),
            "built": built,
            "items": items,
        })

    review_covers = cfg.get('review_covers', [])
    top_recommendations = cfg.get('top_recommendations', [])

    return {
        "available": True,
        "per_category": per_category,
        "review_covers": review_covers,
        "top_recommendations": top_recommendations,
    }


def definitions():
    """Category descriptions, priority legend, glossary, references."""
    return {
        "available": True,
        "categories": [
            {"name": "functional", "description": "Clinical features: seizure detection, prediction, closed-loop response, time-frequency analysis"},
            {"name": "technology", "description": "Model architectures: CNN, RNN/LSTM, GNN, SNN, Transformer, transfer learning, federated learning"},
            {"name": "data", "description": "Datasets, augmentation, neonatal EEG, benchmark data sources"},
            {"name": "gap", "description": "Missing capabilities: cross-patient benchmarks, foundation models, prediction-horizon metrics"},
            {"name": "architecture", "description": "System architecture: hybrid pipelines, edge deployment, quantization"},
            {"name": "decision_ai", "description": "Decision layer: risk forecasting, closed-loop decisions with audit trails"},
        ],
        "priority_legend": [
            {"level": "high", "description": "Core gap — reviewers/examiners expect this. Must address for thesis defense."},
            {"level": "medium", "description": "Important gap — strengthens the contribution. Should address if time permits."},
            {"level": "low", "description": "Nice-to-have — differentiator but not expected for a DBA thesis."},
        ],
        "status_legend": [
            {"status": "built", "description": "Live in platform — dashboard + API endpoints verified 200"},
            {"status": "partial", "description": "Backend exists but frontend incomplete or endpoints not verified"},
            {"status": "planned", "description": "In roadmap, not yet implemented"},
        ],
        "glossary": [
            {"term": "EEGNet", "definition": "Compact CNN architecture for EEG decoding (Lawhern et al. 2018)"},
            {"term": "CHB-MIT", "definition": "Children's Hospital Boston–MIT scalp EEG dataset (23 patients, 844 hrs)"},
            {"term": "TUH-TUSZ", "definition": "Temple University Hospital Seizure Corpus — largest open EEG seizure dataset"},
            {"term": "Bonn", "definition": "University of Bonn EEG dataset — 5 classes (A–E), 100 segments each"},
            {"term": "STFT", "definition": "Short-Time Fourier Transform — time-frequency representation of EEG"},
            {"term": "GNN", "definition": "Graph Neural Network — models electrode-to-electrode connectivity"},
            {"term": "SNN", "definition": "Spiking Neural Network — neuromorphic, ultra-low-power neural computing"},
            {"term": "FAR", "definition": "False Alarm Rate — false positives per hour, critical for seizure prediction"},
            {"term": "LOSO", "definition": "Leave-One-Subject-Out — cross-patient validation preventing data leakage"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — model-agnostic feature importance"},
            {"term": "ONNX", "definition": "Open Neural Network Exchange — portable model format for edge deployment"},
            {"term": "FedAvg", "definition": "Federated Averaging — privacy-preserving distributed training algorithm"},
        ],
        "clinical_notes": [
            "The 50-paper review covers 2018–2025 epilepsy deep learning literature.",
            "Gap analysis maps each review finding to project implementation status.",
            "Priority levels reflect thesis defense expectations, not clinical urgency.",
            "All 'built' items have verified API endpoints and frontend dashboards.",
        ],
        "references": [
            "Epilepsy DL Review (50 papers) — project internal literature review",
            "CHB-MIT Scalp EEG Database — PhysioNet",
            "TUH EEG Corpus — Temple University Hospital",
            "Lawhern et al. (2018) — EEGNet: A Compact CNN for EEG-based BCI",
            "Bonn University EEG Dataset — Andrzejak et al. (2001)",
        ],
    }
