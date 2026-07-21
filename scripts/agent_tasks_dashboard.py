"""Agent Tasks Dashboard — 60-agent pipeline registry visualization
from config/agent_tasks.json.
60 agents, 55 built / 5 planned, modules, statuses, categories."""

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


def _categorize(agent):
    """Assign a functional category based on agent id/task keywords."""
    aid = agent.get("id", "").lower()
    task = agent.get("task", "").lower()
    combined = aid + " " + task
    if any(k in combined for k in ["ingest", "extract", "router", "dat_", "edf_", "pdf_", "docx_", "text_", "image_"]):
        return "Ingestion"
    if any(k in combined for k in ["feature", "noise", "ica", "artifact", "cleaning"]):
        return "Signal Processing"
    if any(k in combined for k in ["classif", "predict", "model", "xgboost", "lightgbm", "cnn", "resnet", "rnn", "lstm", "yolo", "snn", "ensemble", "dim_reduction", "pca", "smote", "balancing"]):
        return "ML / Models"
    if any(k in combined for k in ["xai", "explainab", "interpretab", "shap", "saliency"]):
        return "Explainability"
    if any(k in combined for k in ["responsible", "fairness", "guardrail", "nemo", "governance", "accountability", "hitl"]):
        return "Governance / Safety"
    if any(k in combined for k in ["iot", "wearable", "emotiv", "device", "camera", "microphone", "continuous_monitoring"]):
        return "IoT / Devices"
    if any(k in combined for k in ["voice", "conversational", "assistant", "decision_ai", "analytics", "feedback", "correction", "consensus", "rlhf"]):
        return "AI Services"
    if any(k in combined for k in ["patient", "reporting", "dashboard"]):
        return "Patient / Reporting"
    if any(k in combined for k in ["video", "audio", "body_movement", "object_detect", "segmentation"]):
        return "Multimedia"
    if any(k in combined for k in ["council", "bmad", "paperclip", "openclaw", "mcp"]):
        return "Orchestration"
    if any(k in combined for k in ["synthetic", "master_data", "transaction"]):
        return "Data Pipeline"
    return "Other"


def overview():
    """Summary KPIs: agent count, status distribution, category breakdown,
    agents-per-category bar, modules table."""
    cfg = _load("agent_tasks.json")
    if not cfg:
        return {"available": False, "note": "agent_tasks.json missing"}

    agents = cfg.get("agents", [])
    total = len(agents)

    # Status distribution (pie)
    status_counts = Counter(a.get("status", "unknown") for a in agents)
    status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    # Category distribution (pie + bar)
    cats = [_categorize(a) for a in agents]
    cat_counts = Counter(cats)
    category_distribution = [
        {"name": c, "value": v}
        for c, v in sorted(cat_counts.items(), key=lambda x: -x[1])
    ]

    # Agents with notes (have richer metadata)
    agents_with_notes = sum(1 for a in agents if a.get("note"))
    agents_with_needs = sum(1 for a in agents if a.get("needs"))

    # Unique modules
    modules = set()
    for a in agents:
        m = a.get("module", "")
        if m and m != "(planned)":
            modules.add(m.split("+")[0].strip().split("/")[0].strip())

    # Agents summary table
    agents_table = []
    for a in agents:
        agents_table.append({
            "id": a.get("id", ""),
            "task": a.get("task", ""),
            "module": a.get("module", ""),
            "status": a.get("status", ""),
            "category": _categorize(a),
            "has_note": bool(a.get("note")),
            "has_needs": bool(a.get("needs")),
        })

    return {
        "available": True,
        "title": "Agent Tasks Registry",
        "description": cfg.get("description", ""),
        "domain": cfg.get("domain", ""),
        "updated_at": cfg.get("updated_at", ""),
        "kpis": {
            "total_agents": total,
            "built": status_counts.get("built", 0),
            "planned": status_counts.get("planned", 0),
            "categories": len(cat_counts),
            "with_notes": agents_with_notes,
            "with_needs": agents_with_needs,
            "unique_modules": len(modules),
        },
        "status_distribution": status_distribution,
        "category_distribution": category_distribution,
        "agents_table": agents_table,
    }


def breakdown():
    """Per-category detail: grouped agents with full metadata."""
    cfg = _load("agent_tasks.json")
    if not cfg:
        return {"available": False, "note": "agent_tasks.json missing"}

    agents = cfg.get("agents", [])

    # Group by category
    by_cat = {}
    for a in agents:
        cat = _categorize(a)
        by_cat.setdefault(cat, []).append({
            "id": a.get("id", ""),
            "task": a.get("task", ""),
            "module": a.get("module", ""),
            "status": a.get("status", ""),
            "needs": a.get("needs", ""),
            "note": a.get("note", ""),
        })

    per_category = []
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        built = sum(1 for i in items if i["status"] == "built")
        per_category.append({
            "category": cat,
            "agents": items,
            "total": len(items),
            "built": built,
            "planned": len(items) - built,
        })

    per_category.sort(key=lambda x: -x["total"])

    return {
        "available": True,
        "per_category": per_category,
    }


def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "description": "Agent live — code exists, verified, producing real outputs"},
            {"status": "planned", "description": "Agent cataloged — specification defined, implementation not started"},
        ],
        "glossary": [
            {"term": "MoA", "definition": "Mixture of Agents — ensemble architecture with multiple specialist agents and a meta-orchestrator"},
            {"term": "MCP", "definition": "Model Context Protocol — standardized tool/context interface for AI agents"},
            {"term": "HITL", "definition": "Human-in-the-Loop — mandatory expert review for high-stakes AI decisions"},
            {"term": "ICA", "definition": "Independent Component Analysis — artifact removal method for EEG signals"},
            {"term": "RLHF", "definition": "Reinforcement Learning from Human Feedback — training from expert corrections and preferences"},
            {"term": "XAI", "definition": "Explainable AI — SHAP/Grad-CAM methods making model predictions interpretable"},
            {"term": "BMAD", "definition": "BMAD Spec-Driven Agent Development — specification-first methodology for building AI agents"},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — grounding LLM responses with retrieved clinical documents"},
            {"term": "SMOTE", "definition": "Synthetic Minority Oversampling Technique — class balancing for imbalanced EEG datasets"},
            {"term": "EDF", "definition": "European Data Format — standard format for EEG/polysomnography signal storage"},
            {"term": "SNN", "definition": "Spiking Neural Network — neuromorphic computing model mimicking biological neural signaling"},
            {"term": "GNN", "definition": "Graph Neural Network — network modeling electrode spatial relationships for connectivity analysis"},
        ],
        "clinical_notes": [
            "The 60-agent registry covers the full EEG epilepsy AI pipeline from file ingestion through diagnosis to patient reporting.",
            "55 of 60 agents (92%) are fully built with real data paths — 5 remaining are IoT/device agents needing hardware SDKs.",
            "Agent statuses are honest: built = code exists and is verified; planned = specification only, not yet implemented.",
            "Governance agents (guardrails, HITL, responsible AI) ensure all clinical decisions pass safety and fairness gates.",
        ],
        "references": [
            {"ref": "ILAE (2017)", "detail": "Operational classification of seizure types — agent task alignment"},
            {"ref": "MNE-Python", "detail": "Open-source Python library for EEG/MEG signal processing used by feature/noise agents"},
            {"ref": "Anthropic MCP (2024)", "detail": "Model Context Protocol specification for agent tool interfaces"},
            {"ref": "OWASP AI Security (2023)", "detail": "AI application security guidelines informing guardrail agent design"},
        ],
    }
