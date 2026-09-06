"""Tab Taxonomy Dashboard — Patient Master + Role Ops + AI Capability tab
taxonomy visualization from config/tab_taxonomy.json.
3 categories, 35 total tabs, all mapped to real endpoints/components."""

import json
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "tab_taxonomy.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: categories, tabs, statuses, charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "tab_taxonomy.json missing"}

    pm_tabs = cfg.get("patient_master_tabs", [])
    role_tabs = cfg.get("role_operational_tabs", [])
    ai_tabs = cfg.get("ai_capability_tabs", [])
    all_tabs = pm_tabs + role_tabs + ai_tabs

    total_tabs = len(all_tabs)
    built = sum(1 for t in all_tabs if t.get("status") == "built")
    partial = sum(1 for t in all_tabs if t.get("status") == "partial")
    planned = sum(1 for t in all_tabs if t.get("status") == "planned")
    built_pct = round(built / total_tabs * 100, 1) if total_tabs else 0

    # Tabs per category chart
    tabs_per_category = [
        {"name": "Patient Master", "value": len(pm_tabs)},
        {"name": "Role Operations", "value": len(role_tabs)},
        {"name": "AI Capabilities", "value": len(ai_tabs)},
    ]

    # Status distribution chart
    status_distribution = []
    if built:
        status_distribution.append({"name": "Built", "value": built})
    if partial:
        status_distribution.append({"name": "Partial", "value": partial})
    if planned:
        status_distribution.append({"name": "Planned", "value": planned})

    # Tabs with maps_to (have real endpoints/components)
    mapped = sum(1 for t in all_tabs if t.get("maps_to"))
    unmapped = total_tabs - mapped

    mapping_coverage = [
        {"name": "Mapped", "value": mapped},
        {"name": "Unmapped", "value": unmapped},
    ]

    # All tabs summary table
    tabs_summary = []
    for cat_name, cat_tabs in [
        ("Patient Master", pm_tabs),
        ("Role Operations", role_tabs),
        ("AI Capabilities", ai_tabs),
    ]:
        for t in cat_tabs:
            tabs_summary.append({
                "category": cat_name,
                "id": t.get("id", ""),
                "label": t.get("label", ""),
                "status": t.get("status", "unknown"),
                "has_mapping": bool(t.get("maps_to")),
            })

    as_is_to_be = cfg.get("as_is_to_be", {})

    return {
        "available": True,
        "title": cfg.get("title", "Tab Taxonomy"),
        "note": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "kpis": {
            "total_tabs": total_tabs,
            "categories": 3,
            "built": built,
            "partial": partial,
            "planned": planned,
            "built_pct": built_pct,
            "mapped": mapped,
            "patient_master_count": len(pm_tabs),
            "role_ops_count": len(role_tabs),
            "ai_caps_count": len(ai_tabs),
        },
        "tabs_per_category": tabs_per_category,
        "status_distribution": status_distribution,
        "mapping_coverage": mapping_coverage,
        "tabs_summary": tabs_summary,
        "as_is_to_be": as_is_to_be,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-category tab details with captures/metrics and mappings."""
    cfg = _load()
    if not cfg:
        return {"available": False}

    result = {"available": True, "categories": []}

    for cat_key, cat_name, desc_field, metric_field in [
        ("patient_master_tabs", "Patient Master", "captures", None),
        ("role_operational_tabs", "Role Operations", None, "metric"),
        ("ai_capability_tabs", "AI Capabilities", None, None),
    ]:
        tabs = cfg.get(cat_key, [])
        cat_data = {
            "key": cat_key,
            "name": cat_name,
            "total": len(tabs),
            "built": sum(1 for t in tabs if t.get("status") == "built"),
            "tabs": [],
        }
        for t in tabs:
            tab_info = {
                "id": t.get("id", ""),
                "label": t.get("label", ""),
                "status": t.get("status", "unknown"),
                "maps_to": t.get("maps_to", ""),
            }
            if desc_field and t.get(desc_field):
                tab_info["description"] = t[desc_field]
            if metric_field and t.get(metric_field):
                tab_info["metric"] = t[metric_field]
            cat_data["tabs"].append(tab_info)
        result["categories"].append(cat_data)

    result["as_is_to_be"] = cfg.get("as_is_to_be", {})
    return result


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "color": "#4caf50",
             "meaning": "Tab is implemented with real endpoints and UI components"},
            {"status": "partial", "color": "#ff9800",
             "meaning": "Tab has some functionality but is not fully implemented"},
            {"status": "planned", "color": "#f44336",
             "meaning": "Tab is designed but not yet built"},
        ],
        "glossary": [
            {"term": "Patient Master", "definition": "Unified self-service portal where a patient sees all their data — assessments, medications, alerts, reports, AI chat, and transaction history."},
            {"term": "Role Operational Tabs", "definition": "Per-role workspace tabs that track each expert's work — patients seen, appointments, prescriptions, ADL scores, tasks, and reporting."},
            {"term": "AI Capability Tabs", "definition": "Tabs exposing AI services — conversational (RAG), generative (summaries), decision (routing), responsible, ethical, trust, explainable, fairness, and governance AI."},
            {"term": "MoCA", "definition": "Montreal Cognitive Assessment — a 30-point cognitive screening instrument for mild cognitive dysfunction."},
            {"term": "PHQ-9", "definition": "Patient Health Questionnaire-9 — a 9-item depression severity screening tool."},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — a game-theoretic approach to explaining model predictions."},
            {"term": "RAG", "definition": "Retrieval Augmented Generation — grounding AI responses in patient-specific retrieved documents."},
            {"term": "HITL", "definition": "Human-In-The-Loop — requiring clinician validation before AI findings are finalized."},
            {"term": "ADL", "definition": "Activities of Daily Living — functional assessment of a patient's ability to perform everyday tasks."},
            {"term": "AED", "definition": "Anti-Epileptic Drug — medication used to prevent or reduce seizure frequency."},
            {"term": "SOS", "definition": "Emergency alert triggered by patient or device when a seizure or critical event is detected."},
            {"term": "As-Is / To-Be", "definition": "Gap analysis comparing current siloed workflows (As-Is) to the unified, AI-augmented target state (To-Be)."},
        ],
        "clinical_notes": [
            "Patient Master tabs give patients self-service access to their own clinical data under governance.",
            "Role operational tabs ensure every expert's work is captured, measured, and auditable.",
            "AI capability tabs are governed: every AI output goes through explainability, fairness, and human review.",
            "The As-Is → To-Be transformation unifies siloed expert workflows into a single auditable patient-centric system.",
        ],
        "references": [
            {"label": "tab_taxonomy.json", "url": "config/tab_taxonomy.json", "note": "Source configuration for tab taxonomy"},
            {"label": "ILAE Classification", "url": "https://www.ilae.org/guidelines", "note": "International League Against Epilepsy guidelines"},
            {"label": "MoCA", "url": "https://mocatest.org", "note": "Montreal Cognitive Assessment official site"},
            {"label": "SHAP", "url": "https://shap.readthedocs.io", "note": "SHAP explainability framework"},
        ],
    }
