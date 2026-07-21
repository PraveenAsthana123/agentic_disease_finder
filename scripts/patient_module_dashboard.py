"""Patient Module Dashboard — 8-section, ~1,250-field patient module overview
from config/patient_module.json.
8 sections, ~1250 fields, seizure diary + trigger tracking + medication + PRO + wearables."""

import json
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "patient_module.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: sections, fields, status, tiers, control groups, charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "patient_module.json missing"}

    sections = cfg.get("sections", [])
    summary = cfg.get("summary", {})
    tiers = cfg.get("tiers", {})
    control_groups = cfg.get("control_groups", {})

    total_sections = summary.get("total_sections", len(sections))
    total_fields = summary.get("total_fields", "~1250")
    built_count = summary.get("built", 0)
    partial_count = summary.get("partial", 0)
    planned_count = summary.get("planned", 0)

    # Status distribution for pie chart
    status_dist = []
    if built_count > 0:
        status_dist.append({"name": "Built", "value": built_count})
    if partial_count > 0:
        status_dist.append({"name": "Partial", "value": partial_count})
    if planned_count > 0:
        status_dist.append({"name": "Planned", "value": planned_count})

    # Fields per section bar chart
    fields_per_section = []
    for s in sections:
        field_str = s.get("fields", "0")
        # Parse range like "30-40" → take midpoint
        if "-" in str(field_str):
            parts = str(field_str).split("-")
            try:
                avg = (int(parts[0]) + int(parts[1])) // 2
            except (ValueError, IndexError):
                avg = 0
        else:
            try:
                avg = int(field_str)
            except (ValueError, TypeError):
                avg = 0
        fields_per_section.append({
            "name": s.get("section", ""),
            "value": avg,
            "range": str(field_str)
        })

    # Items per section bar chart
    items_per_section = []
    for s in sections:
        items_per_section.append({
            "name": s.get("section", ""),
            "value": len(s.get("items", []))
        })

    # Tier counts
    tier1 = tiers.get("tier1_mandatory", [])
    tier2 = tiers.get("tier2_recommended", [])
    tier3 = tiers.get("tier3_dba_excellent", [])

    # Control group minimum dataset
    min_dataset = control_groups.get("minimum_dataset", [])
    ideal_dataset = control_groups.get("ideal_dataset", [])
    min_total = sum(c.get("n", 0) for c in min_dataset)
    ideal_total = sum(c.get("n", 0) for c in ideal_dataset)

    # Sections summary table
    sections_table = []
    for s in sections:
        sections_table.append({
            "n": s.get("n"),
            "section": s.get("section", ""),
            "fields": str(s.get("fields", "")),
            "status": s.get("status", ""),
            "items_count": len(s.get("items", [])),
            "note": s.get("note", "")
        })

    return {
        "available": True,
        "title": cfg.get("title", ""),
        "kpis": {
            "total_sections": total_sections,
            "total_fields": total_fields,
            "built": built_count,
            "partial": partial_count,
            "planned": planned_count,
            "tier1_count": len(tier1),
            "tier2_count": len(tier2),
            "tier3_count": len(tier3),
            "min_cohort_total": min_total,
            "ideal_cohort_total": ideal_total,
            "control_groups": len(control_groups.get("most_valuable", []))
        },
        "charts": {
            "status_distribution": status_dist,
            "fields_per_section": fields_per_section,
            "items_per_section": items_per_section,
            "minimum_dataset": [{"name": c["cohort"], "value": c["n"]} for c in min_dataset],
            "ideal_dataset": [{"name": c["cohort"], "value": c["n"]} for c in ideal_dataset]
        },
        "sections_table": sections_table
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-section details: items, notes, field ranges."""
    cfg = _load()
    if not cfg:
        return {"available": False}

    sections = cfg.get("sections", [])
    tiers = cfg.get("tiers", {})
    control_groups = cfg.get("control_groups", {})
    artifact_template = cfg.get("artifact_template", [])
    top10_artifacts = cfg.get("top10_artifacts", [])
    technician_deliverables = cfg.get("technician_deliverables", [])
    single_most_important = cfg.get("single_most_important", "")

    sections_detail = []
    for s in sections:
        sections_detail.append({
            "n": s.get("n"),
            "section": s.get("section", ""),
            "fields": str(s.get("fields", "")),
            "status": s.get("status", ""),
            "items": s.get("items", []),
            "note": s.get("note", "")
        })

    return {
        "available": True,
        "sections": sections_detail,
        "tiers": {
            "tier1_mandatory": tiers.get("tier1_mandatory", []),
            "tier2_recommended": tiers.get("tier2_recommended", []),
            "tier3_dba_excellent": tiers.get("tier3_dba_excellent", [])
        },
        "control_groups": {
            "note": control_groups.get("note", ""),
            "most_valuable": control_groups.get("most_valuable", []),
            "minimum_dataset": control_groups.get("minimum_dataset", []),
            "ideal_dataset": control_groups.get("ideal_dataset", [])
        },
        "artifact_template": artifact_template,
        "top10_artifacts": top10_artifacts,
        "technician_deliverables": technician_deliverables,
        "single_most_important": single_most_important
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "color": "#22c55e", "meaning": "Section fully built with real CRUD endpoints, database tables, and dashboard UI"},
            {"status": "partial", "color": "#f97316", "meaning": "Schema/table exists but needs real labeled clinical data"},
            {"status": "planned", "color": "#94a3b8", "meaning": "Spec defined, implementation not started"}
        ],
        "glossary": [
            {"term": "Patient Module", "definition": "The 8-section core data model covering demographics through wearables/digital twin for each epilepsy patient"},
            {"term": "Seizure Diary", "definition": "Patient-reported seizure event log with auto-severity scoring, triggers, post-ictal state, and monthly trends"},
            {"term": "Trigger Tracking", "definition": "Daily logging of sleep, medication, stress, illness, hormonal, environmental, and lifestyle factors correlated with seizure risk"},
            {"term": "PRO (Patient-Reported Outcomes)", "definition": "Validated instruments (PSQI/ESS/PHQ-9/GAD-7/QOLIE-31/MoCA/NDDI-E/WPAI) capturing sleep, mood, cognition, and quality of life"},
            {"term": "Digital Twin", "definition": "Physiological baseline model per patient with sleep/activity/risk profiles and 1yr/5yr health trajectory projections"},
            {"term": "AED", "definition": "Anti-Epileptic Drug — first-line seizure medications tracked in medication self-management section"},
            {"term": "DBA", "definition": "Database Administrator / Data-Based Assessment — the dataset requirements and governance framework"},
            {"term": "Tier 1 (Mandatory)", "definition": "Minimum required data: EDF files, EEG reports, diagnosis, age, gender, medication history, seizure type, MRI"},
            {"term": "Tier 2 (Recommended)", "definition": "Recommended enrichment: video EEG, clinical notes, follow-up outcome, hospitalization, treatment response"},
            {"term": "Tier 3 (DBA Excellent)", "definition": "Gold-standard governance: neurologist feedback, clinician review, second opinion, audit trail, decision logs"},
            {"term": "Control Group", "definition": "Non-epilepsy cohorts (healthy, PNES, syncope, migraine, stroke) proving the AI detects epilepsy specifically"},
            {"term": "ILAE", "definition": "International League Against Epilepsy — standard seizure classification system used in the clinical section"}
        ],
        "clinical_notes": [
            "The patient module captures ~1,250 fields across 8 sections — from demographics to digital twin projections",
            "Seizure diary auto-calculates severity scores and correlates events with tracked triggers",
            "All 8 validated PRO instruments are auto-scored on submission with published interpretation bands",
            "Wearable data feeds into digital twin models for longitudinal seizure risk forecasting"
        ],
        "references": [
            {"label": "patient_module.json", "description": "Source config — 8-section patient module spec with field counts and build status"},
            {"label": "ILAE Classification", "description": "International League Against Epilepsy seizure classification standard"},
            {"label": "QOLIE-31", "description": "Quality of Life in Epilepsy Inventory — 31-item validated instrument"},
            {"label": "MoCA", "description": "Montreal Cognitive Assessment — cognitive screening used in neuropsychological section"}
        ]
    }
