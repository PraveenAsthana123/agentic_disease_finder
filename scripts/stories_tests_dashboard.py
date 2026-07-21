"""Stories & Tests Dashboard — User Stories, Demo Stories, and
9-Dimension Testing Matrix from config/stories_and_tests.json.
5 user stories, 3 demo stories, 9 test dimensions."""

import json
from pathlib import Path
from collections import Counter

_CFG = Path(__file__).resolve().parent.parent / "config" / "stories_and_tests.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: story counts, test dimension counts, status
    distribution, persona distribution, demo time estimate."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "stories_and_tests.json missing"}

    user_stories = cfg.get("user_stories", [])
    demo_stories = cfg.get("demo_stories", [])
    testing = cfg.get("testing", [])

    total_user_stories = len(user_stories)
    total_demo_stories = len(demo_stories)
    total_test_dims = len(testing)

    # Status counts
    status_counter = Counter(t.get("status", "unknown") for t in testing)
    built_dims = status_counter.get("built", 0)
    partial_dims = status_counter.get("partial", 0)

    # Personas
    personas = sorted(set(s.get("persona", "") for s in user_stories))

    # Demo time estimate (count of demo stories × ~30s each)
    total_demo_time_estimate = f"{total_demo_stories * 30}s"

    # Status distribution for pie chart
    status_distribution = [
        {"name": status, "value": count}
        for status, count in sorted(status_counter.items(), key=lambda x: -x[1])
    ]

    # Persona distribution for bar chart
    persona_counter = Counter(s.get("persona", "") for s in user_stories)
    persona_distribution = [
        {"name": persona, "value": count}
        for persona, count in sorted(persona_counter.items(), key=lambda x: -x[1])
    ]

    return {
        "available": True,
        "title": cfg.get("title", "User Stories & Testing Matrix"),
        "kpis": {
            "total_user_stories": total_user_stories,
            "total_demo_stories": total_demo_stories,
            "total_test_dims": total_test_dims,
            "built_dims": built_dims,
            "partial_dims": partial_dims,
            "personas": personas,
            "total_demo_time_estimate": total_demo_time_estimate,
        },
        "user_stories": user_stories,
        "demo_stories": demo_stories,
        "status_distribution": status_distribution,
        "persona_distribution": persona_distribution,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Full detail for user stories, demo stories, and per-dimension
    testing matrix."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "stories_and_tests.json missing"}

    return {
        "available": True,
        "user_stories": cfg.get("user_stories", []),
        "demo_stories": cfg.get("demo_stories", []),
        "testing": cfg.get("testing", []),
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "color": "#22c55e", "label": "Test dimension fully implemented and passing"},
            {"status": "partial", "color": "#f59e0b", "label": "Test dimension partially implemented — some checks pending"},
        ],
        "glossary": [
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — game-theoretic approach to explain ML model predictions by assigning feature importance values"},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — LLM architecture that retrieves relevant documents before generating a response"},
            {"term": "CRUD", "definition": "Create, Read, Update, Delete — the four basic operations for persistent storage"},
            {"term": "HITL", "definition": "Human-In-The-Loop — workflow pattern requiring clinician review before AI outputs are actioned"},
            {"term": "STT", "definition": "Speech-To-Text — automatic transcription of spoken language into written text for clinical note capture"},
            {"term": "EEG", "definition": "Electroencephalography — recording of electrical activity of the brain via scalp electrodes"},
            {"term": "AED", "definition": "Anti-Epileptic Drug — medication prescribed to control or prevent seizures in epilepsy patients"},
            {"term": "PRO", "definition": "Patient-Reported Outcome — health outcome measures derived directly from patient self-reports"},
            {"term": "SOS", "definition": "System-Of-Systems — architecture where independently managed subsystems interoperate to deliver end-to-end capability"},
        ],
        "clinical_notes": [
            "User stories are mapped to real API endpoints — each persona flow is executable end-to-end.",
            "Demo stories are time-boxed to ~30 seconds each, showing the full Responsible-AI-under-oversight loop.",
            "The 9-dimension testing matrix covers API, Process, ML, XAI, Clinical, Security, Data, UI, and Integration layers.",
            "All 'built' dimensions have automated checks; 'partial' dimensions have manual verification pending automation.",
        ],
        "references": [
            {"ref": "ILAE", "detail": "International League Against Epilepsy — clinical classification and testing standards for epilepsy AI tools"},
            {"ref": "OWASP", "detail": "Open Worldwide Application Security Project — security testing guidelines for healthcare AI applications"},
            {"ref": "ISO 62304", "detail": "Medical device software lifecycle standard — defines verification and validation requirements for clinical AI"},
            {"ref": "FDA AI/ML", "detail": "FDA framework for AI/ML-based Software as a Medical Device — testing and monitoring requirements"},
        ],
    }
