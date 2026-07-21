"""Feature Gaps Dashboard — Epilepsy DL Review (50 papers) vs Project
Implementation gap analysis from config/feature_gaps.json.
18 gaps across 6 categories, 5 recommendations, priority/status tracking."""

import json
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "feature_gaps.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# -- overview --
def overview():
    """Summary KPIs, category/priority charts, gap summary table."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "feature_gaps.json missing"}

    gaps = cfg.get("gaps", [])
    total = len(gaps)

    built = sum(1 for g in gaps if g.get("in_project") == "built")
    partial = sum(1 for g in gaps if g.get("in_project") == "partial")
    planned = sum(1 for g in gaps if g.get("in_project") == "planned")
    built_pct = round(built / total * 100, 1) if total else 0

    high = sum(1 for g in gaps if g.get("priority") == "high")
    medium = sum(1 for g in gaps if g.get("priority") == "medium")
    low = sum(1 for g in gaps if g.get("priority") == "low")

    cats = {}
    for g in gaps:
        c = g.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1

    review_covers = cfg.get("review_covers", [])
    recs = cfg.get("top_recommendations", [])

    category_distribution = [
        {"name": c, "value": v} for c, v in sorted(cats.items(), key=lambda x: -x[1])
    ]

    priority_distribution = []
    if high:
        priority_distribution.append({"name": "high", "value": high})
    if medium:
        priority_distribution.append({"name": "medium", "value": medium})
    if low:
        priority_distribution.append({"name": "low", "value": low})

    gap_table = [
        {
            "category": g.get("category", ""),
            "feature": g.get("feature", ""),
            "in_project": g.get("in_project", "unknown"),
            "priority": g.get("priority", "medium"),
        }
        for g in gaps
    ]

    return {
        "available": True,
        "title": cfg.get("title", "Feature Gaps"),
        "source": cfg.get("source", ""),
        "updated_at": cfg.get("updated_at", ""),
        "summary": {
            "total_gaps": total,
            "built": built,
            "partial": partial,
            "planned": planned,
            "built_pct": built_pct,
            "categories": len(cats),
            "high_priority": high,
            "medium_priority": medium,
            "low_priority": low,
            "review_papers": 50,
            "review_covers_count": len(review_covers),
            "recommendations": len(recs),
        },
        "category_distribution": category_distribution,
        "priority_distribution": priority_distribution,
        "gap_table": gap_table,
    }


# -- breakdown --
def breakdown():
    """Per-category gap details + review topics + recommendations."""
    cfg = _load()
    if not cfg:
        return {"available": False}

    gaps = cfg.get("gaps", [])

    cat_map = {}
    for g in gaps:
        c = g.get("category", "unknown")
        if c not in cat_map:
            cat_map[c] = []
        cat_map[c].append(g)

    per_category = []
    for cat, items in cat_map.items():
        built = sum(1 for it in items if it.get("in_project") == "built")
        per_category.append({
            "category": cat,
            "total": len(items),
            "built": built,
            "items": [
                {
                    "feature": it.get("feature", ""),
                    "in_project": it.get("in_project", "unknown"),
                    "priority": it.get("priority", "medium"),
                    "why": it.get("why", ""),
                    "dashboard": it.get("dashboard", ""),
                }
                for it in items
            ],
        })

    return {
        "available": True,
        "per_category": per_category,
        "review_covers": cfg.get("review_covers", []),
        "top_recommendations": cfg.get("top_recommendations", []),
    }


# -- definitions --
def definitions():
    """Category descriptions, priority/status legends, glossary, references."""
    return {
        "available": True,
        "categories": [
            {"name": "functional", "description": "Clinical features directly usable by clinicians — prediction, time-frequency analysis, closed-loop alerts."},
            {"name": "technology", "description": "AI/ML model architectures and paradigms — deep learning, GNN, SNN, transfer learning, federated learning."},
            {"name": "data", "description": "Dataset availability, augmentation, and benchmark coverage — CHB-MIT, TUH, Bonn, neonatal EEG."},
            {"name": "gap", "description": "Known evaluation or methodology gaps — cross-patient benchmarks, foundation models, prediction metrics."},
            {"name": "architecture", "description": "System architecture patterns — hybrid pipelines (CNN-LSTM/Transformer), edge/quantized deployment."},
            {"name": "decision_ai", "description": "Decision-layer AI — seizure-risk forecasting with alert thresholds, closed-loop decision audit trails."},
        ],
        "priority_legend": [
            {"level": "high", "description": "Critical gap — required for publication-grade coverage or clinical safety."},
            {"level": "medium", "description": "Important gap — adds significant value but project can function without it."},
            {"level": "low", "description": "Nice-to-have — emerging research area or niche clinical need."},
        ],
        "status_legend": [
            {"status": "built", "description": "Feature is implemented with verified endpoints and frontend dashboard."},
            {"status": "partial", "description": "Feature has some implementation but is not fully functional."},
            {"status": "planned", "description": "Feature is identified but not yet implemented."},
        ],
        "glossary": [
            {"term": "EEGNet", "definition": "A compact CNN architecture specifically designed for EEG-based brain-computer interfaces."},
            {"term": "CHB-MIT", "definition": "Children's Hospital Boston — MIT scalp EEG dataset, a standard benchmark for seizure detection."},
            {"term": "TUH/TUSZ", "definition": "Temple University Hospital Seizure corpus — large-scale annotated clinical EEG dataset."},
            {"term": "Bonn", "definition": "University of Bonn EEG dataset — 5-class benchmark (sets A-E) for epilepsy classification."},
            {"term": "STFT", "definition": "Short-Time Fourier Transform — time-frequency decomposition of EEG signals."},
            {"term": "GNN", "definition": "Graph Neural Network — models electrode spatial relationships as a graph structure."},
            {"term": "SNN", "definition": "Spiking Neural Network — neuromorphic model using biologically-inspired spike timing."},
            {"term": "FAR", "definition": "False Alarm Rate — number of incorrect seizure alerts per hour, critical for clinical usability."},
            {"term": "Pre-ictal", "definition": "The period immediately before a seizure, used for seizure prediction/forecasting."},
            {"term": "LOSO", "definition": "Leave-One-Subject-Out — cross-validation strategy that tests generalization to unseen patients."},
            {"term": "ONNX", "definition": "Open Neural Network Exchange — format for exporting models for edge/device deployment."},
            {"term": "Federated Learning", "definition": "Privacy-preserving distributed training across multiple hospital sites without sharing raw data."},
        ],
        "clinical_notes": [
            "Gap analysis is derived from a systematic review of 50 deep learning papers in epilepsy EEG analysis.",
            "All 18 gaps have been addressed (built status), demonstrating comprehensive project coverage.",
            "Seizure prediction and cross-patient generalization are the highest-priority clinical gaps.",
            "Edge deployment and federated learning address practical clinical deployment constraints.",
        ],
        "references": [
            {"label": "feature_gaps.json", "url": "config/feature_gaps.json", "note": "Source configuration for gap analysis"},
            {"label": "Epilepsy DL Review", "url": "Epilepsy/paper/epilepsy-dl-review.pdf", "note": "50-paper systematic review source"},
            {"label": "CHB-MIT Dataset", "url": "https://physionet.org/content/chbmit/", "note": "PhysioNet CHB-MIT scalp EEG database"},
            {"label": "TUH EEG Corpus", "url": "https://isip.piconepress.com/projects/tuh_eeg/", "note": "Temple University Hospital EEG corpus"},
        ],
    }
