"""Data Manager Dashboard — Clinical Data Manager (CDM) role overview.
17 tasks (intake→archival), 8 sub-dashboards, 10 quality assessments,
per-task steps/challenges/AI features from config/data_manager.json."""

import json
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "data_manager.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# -- overview --
def overview():
    """Summary KPIs, task status distribution, quality assessments, sub-dashboard list."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "data_manager.json missing"}

    tasks = cfg.get("tasks", [])
    total = len(tasks)
    built = sum(1 for t in tasks if t.get("status") == "built")
    partial = sum(1 for t in tasks if t.get("status") == "partial")
    planned = sum(1 for t in tasks if t.get("status") == "planned")
    built_pct = round(built / total * 100, 1) if total else 0

    dashboards = cfg.get("dashboards", [])
    dash_built = sum(1 for d in dashboards if d.get("status") == "built")

    qa = cfg.get("quality_assessments", [])

    # count total steps and challenges across all tasks
    total_steps = sum(len(t.get("steps", [])) for t in tasks)
    total_challenges = sum(len(t.get("challenges", [])) for t in tasks)

    status_distribution = []
    if built:
        status_distribution.append({"name": "built", "value": built})
    if partial:
        status_distribution.append({"name": "partial", "value": partial})
    if planned:
        status_distribution.append({"name": "planned", "value": planned})

    task_table = [
        {
            "name": t.get("name", ""),
            "ai_feature": t.get("ai_feature", ""),
            "deliverable": t.get("deliverable", ""),
            "status": t.get("status", "unknown"),
            "steps_count": len(t.get("steps", [])),
            "challenges_count": len(t.get("challenges", [])),
        }
        for t in tasks
    ]

    dashboard_table = [
        {
            "name": d.get("name", ""),
            "shows": d.get("shows", ""),
            "status": d.get("status", "unknown"),
            "endpoint": d.get("endpoint", ""),
        }
        for d in dashboards
    ]

    return {
        "available": True,
        "role": cfg.get("role", "Clinical Data Manager"),
        "mission": cfg.get("mission", ""),
        "updated_at": cfg.get("updated_at", ""),
        "summary": {
            "total_tasks": total,
            "built": built,
            "partial": partial,
            "planned": planned,
            "built_pct": built_pct,
            "total_steps": total_steps,
            "total_challenges": total_challenges,
            "dashboards": len(dashboards),
            "dashboards_built": dash_built,
            "quality_assessments": len(qa),
        },
        "status_distribution": status_distribution,
        "task_table": task_table,
        "dashboard_table": dashboard_table,
        "quality_assessments": qa,
    }


# -- breakdown --
def breakdown():
    """Per-task detail: steps, challenges, AI features, endpoints."""
    cfg = _load()
    if not cfg:
        return {"available": False}

    tasks = cfg.get("tasks", [])
    per_task = []
    for t in tasks:
        per_task.append({
            "name": t.get("name", ""),
            "ai_feature": t.get("ai_feature", ""),
            "deliverable": t.get("deliverable", ""),
            "status": t.get("status", "unknown"),
            "steps": t.get("steps", []),
            "challenges": t.get("challenges", []),
            "endpoints": t.get("endpoints", t.get("endpoint", [])),
        })

    dashboards = cfg.get("dashboards", [])

    return {
        "available": True,
        "per_task": per_task,
        "dashboards": dashboards,
    }


# -- definitions --
def definitions():
    """CDM role description, status legend, glossary, references."""
    return {
        "available": True,
        "role_description": (
            "The Clinical Data Manager (CDM) ensures EEG/MRI/Video-EEG/EMR/assessment/AI datasets "
            "are complete, clean, standardized, versioned, traceable, and AI-ready. "
            "Garbage in = garbage out — the data-governance backbone of Responsible AI."
        ),
        "task_categories": [
            {"name": "Intake & Validation", "tasks": ["Data Intake", "Data Validation", "Data Cleaning", "Missing Data", "Duplicate Detection"]},
            {"name": "Standardization", "tasks": ["Data Standardization", "Terminology Mapping"]},
            {"name": "Modality QC", "tasks": ["EEG Validation", "MRI Validation", "Video Validation"]},
            {"name": "Labeling & Annotation", "tasks": ["Label Validation", "Annotation QC"]},
            {"name": "Governance & Lifecycle", "tasks": ["Dataset Versioning", "Data Lineage", "Dataset Approval", "Data Sharing", "Data Archival"]},
        ],
        "status_legend": [
            {"status": "built", "description": "Task is implemented with live engine, verified endpoints, and frontend dashboard."},
            {"status": "partial", "description": "Task has basic implementation but some workflows are incomplete."},
            {"status": "planned", "description": "Task is catalogued but not yet implemented."},
        ],
        "glossary": [
            {"term": "CDM", "definition": "Clinical Data Manager — role responsible for data quality throughout the clinical data lifecycle."},
            {"term": "EDF/BDF", "definition": "European Data Format / BioSemi Data Format — standard file formats for physiological signals."},
            {"term": "DICOM", "definition": "Digital Imaging and Communications in Medicine — standard for MRI/CT imaging data."},
            {"term": "ICD-10", "definition": "International Classification of Diseases, 10th Revision — WHO standard for diagnosis coding."},
            {"term": "SNOMED-CT", "definition": "Systematized Nomenclature of Medicine — Clinical Terms, a comprehensive clinical terminology."},
            {"term": "LOINC", "definition": "Logical Observation Identifiers Names and Codes — standard for laboratory and clinical observations."},
            {"term": "Cohen Kappa", "definition": "Statistical measure of inter-rater agreement for 2 raters, correcting for chance agreement."},
            {"term": "Fleiss Kappa", "definition": "Extension of Cohen's kappa to 3+ raters, measuring inter-rater reliability."},
            {"term": "FAIR", "definition": "Findable, Accessible, Interoperable, Reusable — data management principles."},
            {"term": "DVC", "definition": "Data Version Control — Git-based versioning system for datasets and ML models."},
            {"term": "PHI", "definition": "Protected Health Information — any individually identifiable health information under HIPAA."},
            {"term": "AI Readiness", "definition": "Composite score measuring whether a dataset is complete, labeled, and quality-checked enough for model training."},
        ],
        "clinical_notes": [
            "All 17 CDM tasks are implemented (built), covering the full data lifecycle from intake to archival.",
            "10 quality assessment dimensions provide comprehensive data quality monitoring.",
            "Inter-rater agreement (Cohen/Fleiss kappa) validates annotation consistency across expert reviewers.",
            "Dataset versioning and lineage tracking enable full reproducibility of model training runs.",
        ],
        "references": [
            {"label": "data_manager.json", "url": "config/data_manager.json", "note": "Source configuration for CDM task registry"},
            {"label": "FAIR Principles", "url": "https://www.go-fair.org/fair-principles/", "note": "FAIR data management principles"},
            {"label": "CDISC Standards", "url": "https://www.cdisc.org/", "note": "Clinical Data Interchange Standards Consortium"},
            {"label": "HIPAA", "url": "https://www.hhs.gov/hipaa/", "note": "Health Insurance Portability and Accountability Act"},
        ],
    }
