"""Onboarding Intake Dashboard — Patient intake-vs-deferred field classification,
3-step workflow, upload auto-extraction, and time-savings from config/onboarding_intake.json.
~80 intake fields, ~1170 deferred, 15x reduction, 8-10 min active intake."""

import json
import os

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: field counts, time savings, step breakdown, field distribution."""
    cfg = _load('onboarding_intake.json')
    if not cfg:
        return {"available": False, "note": "onboarding_intake.json missing"}

    steps = cfg.get('steps', [])
    summary = cfg.get('summary', {})

    # Step 1 group analysis
    step1 = next((s for s in steps if s.get('step') == 1), {})
    groups = step1.get('groups', [])
    total_intake_fields = step1.get('total_intake_fields', sum(g.get('n', 0) for g in groups))
    group_distribution = [
        {"name": g.get('group', ''), "value": g.get('n', 0)}
        for g in groups
    ]

    # Step 2 extraction sources
    step2 = next((s for s in steps if s.get('step') == 2), {})
    extracts = step2.get('extracts', [])
    extraction_sources = [
        {"doc": e.get('doc', ''), "fills_count": len(e.get('fills', [])), "fills": e.get('fills', [])}
        for e in extracts
    ]

    # Step 3 deferred sections
    step3 = next((s for s in steps if s.get('step') == 3), {})
    deferred_sections = step3.get('deferred_sections', [])
    deferred_field_estimate = step3.get('deferred_field_estimate', '~1170')

    # Pie chart: intake vs deferred
    intake_vs_deferred = [
        {"name": "Intake (active)", "value": total_intake_fields},
        {"name": "Deferred (longitudinal)", "value": 1170},
    ]

    # Steps summary table
    steps_table = []
    for s in steps:
        steps_table.append({
            "step": s.get('step', ''),
            "title": s.get('title', ''),
            "approach": s.get('approach', ''),
        })

    return {
        "available": True,
        "title": cfg.get('title', ''),
        "note": cfg.get('note', ''),
        "goal": cfg.get('goal', ''),
        "summary": {
            "true_intake_fields": summary.get('true_intake_fields', '~80'),
            "deferred_fields": summary.get('deferred_fields', '~1170'),
            "reduction": summary.get('reduction', '15x'),
            "time_saved": summary.get('time_saved', '2-3 hrs -> 8-10 min'),
            "total_groups": len(groups),
            "extraction_sources": len(extracts),
            "deferred_sections_count": len(deferred_sections),
        },
        "intake_vs_deferred": intake_vs_deferred,
        "group_distribution": group_distribution,
        "steps_table": steps_table,
    }


def breakdown():
    """Per-step details: intake groups with fields, extraction sources, deferred sections."""
    cfg = _load('onboarding_intake.json')
    if not cfg:
        return {"available": False, "note": "onboarding_intake.json missing"}

    steps = cfg.get('steps', [])

    # Step 1: intake groups with field lists
    step1 = next((s for s in steps if s.get('step') == 1), {})
    intake_groups = [
        {
            "group": g.get('group', ''),
            "n": g.get('n', 0),
            "fields": g.get('fields', []),
        }
        for g in step1.get('groups', [])
    ]

    # Step 2: extraction detail
    step2 = next((s for s in steps if s.get('step') == 2), {})
    extraction_detail = [
        {
            "doc": e.get('doc', ''),
            "fills": e.get('fills', []),
        }
        for e in step2.get('extracts', [])
    ]
    extraction_note = step2.get('note', '')

    # Step 3: deferred sections
    step3 = next((s for s in steps if s.get('step') == 3), {})
    deferred_detail = [
        {
            "section": d.get('section', ''),
            "capture": d.get('capture', ''),
        }
        for d in step3.get('deferred_sections', [])
    ]
    deferred_note = step3.get('note', '')
    deferred_field_estimate = step3.get('deferred_field_estimate', '~1170')

    return {
        "available": True,
        "step1": {
            "title": step1.get('title', ''),
            "approach": step1.get('approach', ''),
            "total_intake_fields": step1.get('total_intake_fields', 0),
            "groups": intake_groups,
        },
        "step2": {
            "title": step2.get('title', ''),
            "approach": step2.get('approach', ''),
            "extracts": extraction_detail,
            "note": extraction_note,
        },
        "step3": {
            "title": step3.get('title', ''),
            "approach": step3.get('approach', ''),
            "deferred_sections": deferred_detail,
            "deferred_field_estimate": deferred_field_estimate,
            "note": deferred_note,
        },
    }


def definitions():
    """Onboarding definitions: step descriptions, field classification legend, glossary, references."""
    return {
        "available": True,
        "step_descriptions": [
            {"step": 1, "title": "Required Intake (5 min)", "description": "Core patient demographics, chief complaint, seizure history, current medications, emergency contact, and key risk factors. Captured once at registration — required-first approach."},
            {"step": 2, "title": "Upload Reports (2 min)", "description": "Multi-format document upload with AI auto-extraction. EEG reports (PDF), MRI reports, EMR exports, and prior neurology notes are parsed to auto-fill acquisition params, findings, demographics, and treatment history."},
            {"step": 3, "title": "Deferred to Portal", "description": "Longitudinal data captured over time through patient self-service: seizure diary, trigger tracking, medication adherence, PRO questionnaires, wearable data, and caregiver assessments. Not captured at intake."},
        ],
        "field_classification_legend": [
            {"type": "Intake", "color": "#3b82f6", "description": "TRUE intake field — captured once at registration. ~80 fields across 6 groups."},
            {"type": "Auto-extracted", "color": "#22c55e", "description": "Filled automatically from uploaded documents (EEG PDF, MRI report, EMR export, neurology notes)."},
            {"type": "Deferred", "color": "#8b5cf6", "description": "Longitudinal field — captured over time through portal use, self-service, or continuous monitoring."},
        ],
        "glossary": [
            {"term": "Intake", "definition": "The initial patient registration process — fields captured at the first visit or enrollment."},
            {"term": "Deferred", "definition": "Fields NOT captured at intake; instead populated over time through portal use, questionnaires, or device data."},
            {"term": "DRE", "definition": "Drug-Resistant Epilepsy — epilepsy that has not responded to 2+ adequate antiseizure medication trials."},
            {"term": "EMR", "definition": "Electronic Medical Record — digital version of a patient's paper chart."},
            {"term": "PRO", "definition": "Patient-Reported Outcome — health data reported directly by the patient (e.g., PHQ-9, QOLIE-31)."},
            {"term": "Auto-extract", "definition": "AI-powered parsing of uploaded documents to automatically fill structured fields."},
            {"term": "Longitudinal", "definition": "Data collected over time (multiple visits, daily logs, continuous monitoring) rather than at a single point."},
            {"term": "MTS", "definition": "Mesial Temporal Sclerosis — a common finding on MRI associated with temporal lobe epilepsy."},
            {"term": "Status Epilepticus", "definition": "A medical emergency — prolonged seizure lasting >5 minutes or repeated seizures without recovery."},
            {"term": "Semiology", "definition": "Clinical signs and symptoms observed during a seizure — motor patterns, automatisms, aura."},
        ],
        "clinical_notes": [
            "The 15x field reduction (1250 → 80 intake) is the key insight: most 'patient fields' are longitudinal, not intake.",
            "Upload-based auto-extraction (Step 2) saves ~40% of manual data entry by parsing existing clinical documents.",
            "EMR/FHIR pre-fill and voice AI intake are planned extensions that would further reduce active intake time.",
            "The wizard-based intake flow (built) guides clinical staff through the 80 fields in logical order.",
        ],
        "references": [
            "HL7 FHIR R4 — Fast Healthcare Interoperability Resources for structured clinical data exchange.",
            "IHE Patient Administration Management (PAM) — integration profiles for patient registration workflows.",
            "ILAE Classification of Epilepsies (2017) — international standard for seizure and epilepsy classification.",
            "FDA 21 CFR Part 11 — electronic records and electronic signatures in clinical systems.",
            "HIPAA Privacy Rule (45 CFR 164) — protected health information handling requirements.",
        ],
    }
