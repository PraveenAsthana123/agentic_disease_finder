"""Report Layout Dashboard -- EEG / Video-EEG Summary Report Layout
visualization from config/report_layout.json.
Report types, components with AI sources, sections, editability status."""

import json
from pathlib import Path
from collections import Counter

_CFG = Path(__file__).resolve().parent.parent / "config" / "report_layout.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# -- overview ----------------------------------------------------------------
def overview():
    """Summary KPIs: report types, components, sections, editable count,
    AI sources, status, charts, tables."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "report_layout.json missing"}

    report_types = cfg.get("report_types", [])
    components = cfg.get("components", [])
    sections = cfg.get("sections", [])
    status = cfg.get("status", {})

    # Counts
    total_report_types = len(report_types)
    total_components = len(components)
    total_sections = len(sections)
    editable_sections = sum(1 for s in sections if s.get("editable"))
    read_only_sections = total_sections - editable_sections

    # Unique AI sources
    ai_sources = sorted(set(c.get("ai_source", "unknown") for c in components))
    total_ai_sources = len(ai_sources)

    # Component by AI source bar chart
    source_counter = Counter(c.get("ai_source", "unknown") for c in components)
    components_by_source = [
        {"name": src, "value": cnt}
        for src, cnt in sorted(source_counter.items(), key=lambda x: -x[1])
    ]

    # Section type classification
    def classify_section(s):
        sid = s.get("id", "")
        if sid.startswith("ai_"):
            return "AI"
        elif sid.startswith("expert_"):
            return "Expert"
        elif sid == "final_summary":
            return "Final"
        elif sid == "audit":
            return "Audit"
        return "Other"

    section_type_counter = Counter(classify_section(s) for s in sections)
    section_type_distribution = [
        {"name": t, "value": c}
        for t, c in sorted(section_type_counter.items(), key=lambda x: -x[1])
    ]

    # Editable vs read-only pie chart
    editability_distribution = [
        {"name": "Editable", "value": editable_sections},
        {"name": "Read-only", "value": read_only_sections},
    ]

    # Components table
    components_table = []
    for c in components:
        components_table.append({
            "id": c.get("id", ""),
            "label": c.get("label", ""),
            "ai_source": c.get("ai_source", ""),
            "ai_finding": c.get("ai_finding", ""),
            "ai_recommendation": c.get("ai_recommendation", ""),
        })

    # Sections table
    sections_table = []
    for s in sections:
        sections_table.append({
            "id": s.get("id", ""),
            "label": s.get("label", ""),
            "source": s.get("source", ""),
            "editable": s.get("editable", False),
            "type": classify_section(s),
        })

    return {
        "available": True,
        "title": cfg.get("title", "Report Layout"),
        "description": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "kpis": {
            "total_report_types": total_report_types,
            "total_components": total_components,
            "total_sections": total_sections,
            "editable_sections": editable_sections,
            "total_ai_sources": total_ai_sources,
            "status_summary": status,
        },
        "components_by_source": components_by_source,
        "section_type_distribution": section_type_distribution,
        "editability_distribution": editability_distribution,
        "components_table": components_table,
        "sections_table": sections_table,
        "report_types": report_types,
    }


# -- breakdown ---------------------------------------------------------------
def breakdown():
    """Per-component AI sources, per-section editability, status details."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "report_layout.json missing"}

    components = cfg.get("components", [])
    sections = cfg.get("sections", [])
    status = cfg.get("status", {})
    report_types = cfg.get("report_types", [])

    # Per-component detail
    per_component = []
    for c in components:
        per_component.append({
            "id": c.get("id", ""),
            "label": c.get("label", ""),
            "ai_source": c.get("ai_source", ""),
            "ai_finding": c.get("ai_finding", ""),
            "ai_recommendation": c.get("ai_recommendation", ""),
        })

    # Sections grouped by type
    def classify_section(s):
        sid = s.get("id", "")
        if sid.startswith("ai_"):
            return "AI"
        elif sid.startswith("expert_"):
            return "Expert"
        elif sid == "final_summary":
            return "Final"
        elif sid == "audit":
            return "Audit"
        return "Other"

    sections_by_type = {}
    for s in sections:
        t = classify_section(s)
        if t not in sections_by_type:
            sections_by_type[t] = []
        sections_by_type[t].append({
            "id": s.get("id", ""),
            "label": s.get("label", ""),
            "source": s.get("source", ""),
            "editable": s.get("editable", False),
        })

    per_section_type = [
        {"type": t, "sections": slist}
        for t, slist in sections_by_type.items()
    ]

    # Report type details
    report_type_details = []
    for rt in report_types:
        report_type_details.append({
            "name": rt,
            "description": _report_type_description(rt),
        })

    # Status breakdown
    status_items = [
        {"key": k, "value": v}
        for k, v in status.items()
    ]

    return {
        "available": True,
        "per_component": per_component,
        "per_section_type": per_section_type,
        "report_type_details": report_type_details,
        "status_items": status_items,
    }


def _report_type_description(rt):
    """Return a short description for each report type."""
    descs = {
        "routine EEG": "Standard 20-30 minute scalp EEG recording with hyperventilation and photic stimulation activation procedures.",
        "video-EEG": "Simultaneous EEG and video recording for correlating electrographic events with clinical semiology.",
        "ambulatory EEG": "Portable 24-72 hour EEG recording in the patient's home environment for capturing events during daily activities.",
        "long-term monitoring (LTM)": "Continuous inpatient EEG monitoring (days to weeks) in an epilepsy monitoring unit for presurgical evaluation.",
    }
    return descs.get(rt, "EEG recording type.")


# -- definitions -------------------------------------------------------------
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "description": "Implemented and verified in this project -- live code, tested endpoints, working pipeline"},
            {"status": "planned", "description": "Architecture slot reserved -- implementation pending (e.g., video correlation module)"},
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography -- recording of brain electrical activity via scalp electrodes, primary diagnostic tool for epilepsy"},
            {"term": "Video-EEG", "definition": "Simultaneous EEG and video recording correlating electrographic patterns with observable clinical behaviors (semiology)"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations -- game-theory feature attribution method showing which EEG features drive the AI prediction"},
            {"term": "ICA", "definition": "Independent Component Analysis -- blind source separation technique for removing artifacts (eye blinks, muscle) from EEG signals"},
            {"term": "HITL", "definition": "Human-In-The-Loop -- expert neurologist review and override gate ensuring AI findings are clinically validated before final report"},
            {"term": "Ambulatory EEG", "definition": "Portable EEG recording worn by the patient at home for 24-72 hours, capturing events in daily-life conditions"},
            {"term": "LTM", "definition": "Long-Term Monitoring -- continuous inpatient EEG recording over days to weeks in an epilepsy monitoring unit"},
            {"term": "Signal Quality", "definition": "Assessment of electrode impedance, channel integrity, and noise levels determining whether the recording is diagnostically usable"},
            {"term": "Impedance", "definition": "Electrical resistance at the electrode-skin interface; high impedance causes noisy signals and must be flagged for re-recording"},
            {"term": "Band Power", "definition": "Spectral power in standard EEG frequency bands: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30+ Hz)"},
            {"term": "Semiology", "definition": "Clinical description of seizure signs and symptoms as observed on video, correlated with EEG onset and propagation patterns"},
            {"term": "Artifact", "definition": "Non-cerebral signal contamination in EEG (eye movement, muscle, electrode, cardiac) that must be identified and removed or flagged"},
        ],
        "clinical_notes": [
            "All AI findings are preliminary and require expert neurologist review before clinical use.",
            "The video correlation component is planned but not yet implemented -- video-EEG reports currently rely on manual semiology annotation.",
            "The audit trail records every AI prediction and expert override with UTC timestamps for regulatory compliance.",
            "Final reports are only valid after expert sign-off through the HITL review gate.",
        ],
        "references": [
            {"ref": "ILAE Classification", "detail": "International League Against Epilepsy -- standardized seizure and epilepsy classification framework used in report categorization"},
            {"ref": "ACNS Guidelines", "detail": "American Clinical Neurophysiology Society -- EEG recording and reporting standards defining minimum technical requirements"},
            {"ref": "MNE-Python", "detail": "Open-source Python library for EEG signal processing, filtering, ICA artifact removal, and spectral analysis"},
            {"ref": "SHAP (Lundberg 2017)", "detail": "Unified framework for interpreting ML predictions using Shapley values from cooperative game theory"},
        ],
    }
