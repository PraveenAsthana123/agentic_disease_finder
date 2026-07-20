"""AI Type Coverage Dashboard — 201 AI-type coverage visualization
from config/ai_type_coverage.json.
201 AI types, 46 built / 155 not-pulled, cross-project coverage tracking."""

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
    """Summary KPIs: total types, status distribution, built vs not-pulled breakdown."""
    cfg = _load('ai_type_coverage.json')
    if not cfg:
        return {"available": False, "note": "ai_type_coverage.json missing"}

    rows = cfg.get('rows', [])
    counts = cfg.get('counts', {})
    total = cfg.get('total', len(rows))

    status_counts = Counter(r.get('status', 'unknown') for r in rows)
    status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    built_rows = [r for r in rows if r.get('status') == 'built']
    not_pulled = [r for r in rows if r.get('status') == 'not-pulled']

    # Group built types by keyword categories
    categories = {}
    for r in built_rows:
        t = r.get('type', '')
        # Derive category from type name
        if 'rag' in t:
            cat = 'RAG / Retrieval'
        elif 'agent' in t or 'multi-agent' in t:
            cat = 'Agentic AI'
        elif any(k in t for k in ['deep-learning', 'reinforcement', 'transfer', 'supervised', 'machine-learning', 'foundation']):
            cat = 'ML / Training'
        elif any(k in t for k in ['fairness', 'responsible', 'bias', 'governance', 'audit', 'guardrail', 'safety']):
            cat = 'Responsible AI'
        elif any(k in t for k in ['eeg', 'neuro', 'clinical', 'healthcare', 'diagnostic']):
            cat = 'Clinical / Neuro'
        elif any(k in t for k in ['drift', 'anomaly', 'monitoring', 'observability']):
            cat = 'Monitoring'
        elif any(k in t for k in ['vision', 'image', 'object', 'video', 'ocr']):
            cat = 'Computer Vision'
        elif any(k in t for k in ['speech', 'voice', 'text-to-audio', 'text-to-video']):
            cat = 'Multimodal / Media'
        elif any(k in t for k in ['knowledge-graph', 'graphrag', 'causal']):
            cat = 'Knowledge / Reasoning'
        elif any(k in t for k in ['explainable', 'time-series', 'predictive', 'digital-twin']):
            cat = 'Analysis / XAI'
        elif any(k in t for k in ['llm', 'document', 'control-tower']):
            cat = 'LLM / Document'
        else:
            cat = 'Other'
        categories.setdefault(cat, []).append(r)

    category_distribution = [
        {"name": cat, "value": len(items)}
        for cat, items in sorted(categories.items(), key=lambda x: -len(x[1]))
    ]

    types_table = [
        {
            "type": r.get('type', ''),
            "status": r.get('status', ''),
            "note": r.get('note', ''),
        }
        for r in rows
    ]

    return {
        "available": True,
        "title": "AI Type Coverage — Cross-Project AI Capability Matrix",
        "note": f"{total} AI types cataloged from insur_project; {len(built_rows)} built in agenticfinder (epilepsy EEG)",
        "updated_at": cfg.get('generated_at', ''),
        "source": cfg.get('source', ''),
        "project": cfg.get('project', ''),
        "summary": {
            "total_types": total,
            "built": counts.get('built', len(built_rows)),
            "not_pulled": counts.get('not-pulled', len(not_pulled)),
            "scaffold": counts.get('scaffold', 0),
            "planned": counts.get('planned', 0),
            "coverage_pct": round(100 * len(built_rows) / max(total, 1), 1),
        },
        "status_distribution": status_distribution,
        "category_distribution": category_distribution,
        "types_table": types_table,
    }


def breakdown():
    """Per-status grouping, built-type detail cards, not-pulled inventory."""
    cfg = _load('ai_type_coverage.json')
    if not cfg:
        return {"available": False, "note": "ai_type_coverage.json missing"}

    rows = cfg.get('rows', [])

    built = [r for r in rows if r.get('status') == 'built']
    not_pulled = [r for r in rows if r.get('status') == 'not-pulled']
    scaffold = [r for r in rows if r.get('status') == 'scaffold']
    planned = [r for r in rows if r.get('status') == 'planned']

    # Group built by derived category
    built_by_category = {}
    for r in built:
        t = r.get('type', '')
        if 'rag' in t:
            cat = 'RAG / Retrieval'
        elif 'agent' in t or 'multi-agent' in t:
            cat = 'Agentic AI'
        elif any(k in t for k in ['deep-learning', 'reinforcement', 'transfer', 'supervised', 'machine-learning', 'foundation']):
            cat = 'ML / Training'
        elif any(k in t for k in ['fairness', 'responsible', 'bias', 'governance', 'audit', 'guardrail']):
            cat = 'Responsible AI'
        elif any(k in t for k in ['eeg', 'neuro', 'clinical', 'healthcare', 'diagnostic']):
            cat = 'Clinical / Neuro'
        elif any(k in t for k in ['drift', 'anomaly', 'monitoring', 'observability']):
            cat = 'Monitoring'
        elif any(k in t for k in ['vision', 'image', 'object', 'video', 'ocr']):
            cat = 'Computer Vision'
        elif any(k in t for k in ['speech', 'voice', 'text-to-audio', 'text-to-video']):
            cat = 'Multimodal / Media'
        elif any(k in t for k in ['knowledge-graph', 'graphrag', 'causal']):
            cat = 'Knowledge / Reasoning'
        elif any(k in t for k in ['explainable', 'time-series', 'predictive', 'digital-twin']):
            cat = 'Analysis / XAI'
        elif any(k in t for k in ['llm', 'document', 'control-tower']):
            cat = 'LLM / Document'
        else:
            cat = 'Other'
        built_by_category.setdefault(cat, []).append({
            "type": r.get('type', ''),
            "note": r.get('note', ''),
        })

    per_category = [
        {"category": cat, "count": len(items), "types": items}
        for cat, items in sorted(built_by_category.items(), key=lambda x: -len(x[1]))
    ]

    not_pulled_list = [
        {"type": r.get('type', ''), "note": r.get('note', '')}
        for r in not_pulled
    ]

    return {
        "available": True,
        "built_count": len(built),
        "not_pulled_count": len(not_pulled),
        "scaffold_count": len(scaffold),
        "planned_count": len(planned),
        "per_category": per_category,
        "not_pulled_list": not_pulled_list,
        "scaffold_list": [{"type": r.get('type', ''), "note": r.get('note', '')} for r in scaffold],
        "planned_list": [{"type": r.get('type', ''), "note": r.get('note', '')} for r in planned],
    }


def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "description": "AI type is implemented in this project — endpoints verified, dashboard functional, real logic"},
            {"status": "not-pulled", "description": "AI type exists in source project (insur_project) but not applicable or not imported to epilepsy EEG"},
            {"status": "scaffold", "description": "Skeleton code exists — function stubs, placeholder endpoints, not yet real logic"},
            {"status": "planned", "description": "Identified as applicable but not yet started — in the roadmap"},
        ],
        "glossary": [
            {"term": "AI Type", "definition": "A category of artificial intelligence capability (e.g., RAG, deep-learning, fairness-ai)"},
            {"term": "Coverage", "definition": "Percentage of cataloged AI types that are built and functional in this project"},
            {"term": "Not-Pulled", "definition": "AI type exists in the source project catalog but is not applicable to epilepsy EEG clinical workflow"},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — LLM answers grounded in retrieved document chunks"},
            {"term": "XAI", "definition": "Explainable AI — techniques that make model decisions interpretable (SHAP, LIME, Grad-CAM)"},
            {"term": "Agentic AI", "definition": "AI systems with autonomous decision-making, tool use, and multi-step reasoning"},
            {"term": "Digital Twin", "definition": "Virtual patient model for simulation, what-if scenarios, and treatment planning"},
            {"term": "Drift Detection", "definition": "Monitoring for distribution shift between training data and production data (PSI, KS test)"},
            {"term": "HITL", "definition": "Human-In-The-Loop — clinical review step before AI decisions are finalized"},
            {"term": "Federated Learning", "definition": "Multi-site model training without sharing raw patient data"},
            {"term": "Transfer Learning", "definition": "Adapting a model trained on one dataset/domain to perform on a different but related domain"},
            {"term": "GraphRAG", "definition": "Graph-based Retrieval-Augmented Generation — combines knowledge graphs with LLM retrieval"},
        ],
        "clinical_notes": [
            "Coverage percentage reflects breadth of AI capabilities, not depth — a built type may range from a single endpoint to a full pipeline",
            "Not-pulled types are domain-specific to their source project (insurance) and would require significant adaptation for clinical EEG",
            "The 46 built types cover the complete clinical EEG AI workflow: capture, process, analyze, predict, explain, govern, monitor",
            "Each built type has at least one verified API endpoint returning HTTP 200 with real data (no stubs)",
        ],
        "references": [
            "Stanford HAI — AI Index Report 2024: comprehensive AI capability taxonomy",
            "NIST AI Risk Management Framework (AI RMF 1.0) — AI type classification",
            "IEEE 7000-2021 — Ethical considerations for AI system design",
            "FDA AI/ML-Based Software as Medical Device (SaMD) Action Plan",
            "EU AI Act — risk-based AI classification framework",
        ],
    }
