"""EEG AI RAG Pipeline Dashboard — 23-step end-to-end pipeline visualization
from config/eeg_ai_rag_pipeline.json.
23 steps (Research Objective → Governance+Monitoring), all built,
6 pipeline phases, real implementation locations."""

import json
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "eeg_ai_rag_pipeline.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


_PHASES = [
    {"phase": "Data Acquisition", "steps": [1, 2, 3, 4], "color": "#3b82f6"},
    {"phase": "Signal Processing", "steps": [5, 6, 7, 8, 9, 10], "color": "#22c55e"},
    {"phase": "Feature Engineering", "steps": [11, 12, 13], "color": "#f97316"},
    {"phase": "Model Training & Validation", "steps": [14, 15, 16], "color": "#ef4444"},
    {"phase": "Explainability & RAG", "steps": [17, 18, 19, 20], "color": "#8b5cf6"},
    {"phase": "Clinical Output & Governance", "steps": [21, 22, 23], "color": "#14b8a6"},
]


def _phase_for_step(n):
    for p in _PHASES:
        if n in p["steps"]:
            return p["phase"]
    return "Other"


def overview():
    """Summary KPIs: step count, phase count, status distribution, charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "eeg_ai_rag_pipeline.json missing"}

    steps = cfg.get("steps", [])
    summary = cfg.get("summary", {})

    status_counts = {}
    for s in steps:
        st = s.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    phase_step_counts = []
    for p in _PHASES:
        matching = [s for s in steps if s.get("n") in p["steps"]]
        built = sum(1 for s in matching if s.get("status") == "built")
        phase_step_counts.append({
            "name": p["phase"],
            "total": len(p["steps"]),
            "built": built,
            "color": p["color"],
        })

    status_dist = [
        {"name": k.title(), "value": v}
        for k, v in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    steps_per_phase = [
        {"name": p["name"], "value": p["total"]} for p in phase_step_counts
    ]

    step_table = []
    for s in steps:
        step_table.append({
            "n": s.get("n"),
            "step": s.get("step", ""),
            "detail": s.get("detail", ""),
            "status": s.get("status", ""),
            "phase": _phase_for_step(s.get("n", 0)),
        })

    return {
        "available": True,
        "title": cfg.get("title", "EEG → AI → RAG Pipeline"),
        "note": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "kpis": {
            "total_steps": len(steps),
            "built": summary.get("built", 0),
            "partial": summary.get("partial", 0),
            "planned": summary.get("planned", 0),
            "total_phases": len(_PHASES),
            "completion_pct": round(summary.get("built", 0) / max(len(steps), 1) * 100),
        },
        "charts": {
            "status_distribution": status_dist,
            "steps_per_phase": steps_per_phase,
            "phase_completion": phase_step_counts,
        },
        "step_table": step_table,
    }


def breakdown():
    """Per-phase step details with implementation locations."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "eeg_ai_rag_pipeline.json missing"}

    steps = cfg.get("steps", [])
    phases = []
    for p in _PHASES:
        matching = [s for s in steps if s.get("n") in p["steps"]]
        phase_steps = []
        for s in matching:
            phase_steps.append({
                "n": s.get("n"),
                "step": s.get("step", ""),
                "detail": s.get("detail", ""),
                "status": s.get("status", ""),
                "where": s.get("where", ""),
            })
        built = sum(1 for s in matching if s.get("status") == "built")
        phases.append({
            "phase": p["phase"],
            "color": p["color"],
            "total": len(p["steps"]),
            "built": built,
            "steps": phase_steps,
        })

    eeg_linkage = cfg.get("eeg_linkage_summary", [])

    return {
        "available": True,
        "phases": phases,
        "eeg_linkage_summary": eeg_linkage,
    }


def definitions():
    """Glossary, clinical notes, references for the EEG AI RAG pipeline."""
    return {
        "available": True,
        "phase_legend": [
            {"phase": p["phase"], "steps": f"Steps {p['steps'][0]}-{p['steps'][-1]}",
             "color": p["color"]}
            for p in _PHASES
        ],
        "status_legend": [
            {"status": "Built", "meaning": "Step fully implemented with working code, endpoints, or dashboards"},
            {"status": "Partial", "meaning": "Core functionality exists but some sub-features pending"},
            {"status": "Planned", "meaning": "Designed and documented but not yet implemented"},
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography — non-invasive recording of brain electrical activity via scalp electrodes"},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — combining LLM generation with document retrieval for evidence-based output"},
            {"term": "ICA", "definition": "Independent Component Analysis — artifact removal technique separating neural from non-neural signals"},
            {"term": "STFT", "definition": "Short-Time Fourier Transform — time-frequency decomposition for spectral analysis of EEG segments"},
            {"term": "CWT", "definition": "Continuous Wavelet Transform — multi-resolution time-frequency analysis using mother wavelets"},
            {"term": "SPWVD", "definition": "Smoothed Pseudo Wigner-Ville Distribution — high-resolution time-frequency representation"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — game-theoretic approach to explain individual predictions"},
            {"term": "LASSO", "definition": "Least Absolute Shrinkage and Selection Operator — L1-regularized feature selection"},
            {"term": "RFE", "definition": "Recursive Feature Elimination — iterative feature removal based on model importance"},
            {"term": "PLV", "definition": "Phase Locking Value — measure of phase synchronization between EEG channels"},
            {"term": "HITL", "definition": "Human-In-The-Loop — requiring clinician review before finalizing AI predictions"},
            {"term": "ChromaDB", "definition": "Open-source vector database for embedding storage and similarity search"},
        ],
        "clinical_notes": [
            "The 23-step pipeline is disease-agnostic — originally derived from schizophrenia EEG but applies to all 7 supported diseases.",
            "Subject-level cross-validation (GroupKFold) prevents data leakage between train/test splits.",
            "RAG layer indexes clinical guidelines, SOPs, and published literature for evidence-based report generation.",
            "Human review (Step 21) is mandatory before any clinical output reaches patients or referrers.",
        ],
        "references": [
            {"label": "eeg_ai_rag_pipeline.json", "note": "Source config — 23-step pipeline definition with per-step status and implementation locations"},
            {"label": "MNE-Python", "note": "Open-source Python package for EEG/MEG signal processing (mne.tools)"},
            {"label": "ILAE Guidelines", "note": "International League Against Epilepsy — seizure classification and EEG standards"},
            {"label": "SHAP (Lundberg & Lee 2017)", "note": "A unified approach to interpreting model predictions"},
        ],
    }
