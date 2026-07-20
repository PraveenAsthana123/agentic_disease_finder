"""Consultant Matrix Dashboard — 10 clinical consultant roles with tasks,
challenges, AI solutions, data requirements, and compliance docs, from
config/consultant_matrix.json."""

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
    """Summary KPIs: total consultants, tier distribution, task/challenge counts, AI coverage."""
    cfg = _load('consultant_matrix.json')
    if not cfg:
        return {"available": False, "note": "consultant_matrix.json missing"}

    consultants = cfg.get('consultants', [])
    total = len(consultants)

    tier_counts = Counter(c.get('tier', 0) for c in consultants)
    mandatory_count = sum(1 for c in consultants if c.get('mandatory'))

    total_tasks = sum(len(c.get('tasks', [])) for c in consultants)
    total_challenges = sum(len(c.get('challenges', [])) for c in consultants)
    total_ai_solutions = sum(len(c.get('ai_solutions', [])) for c in consultants)
    total_docs = sum(len(c.get('documents', [])) for c in consultants)
    total_compliance = sum(len(c.get('compliance_docs', [])) for c in consultants)
    total_assessments = sum(len(c.get('assessment', [])) for c in consultants)
    total_tools = sum(len(c.get('tools', [])) for c in consultants)

    # Data requirement matrix: count yes/optional/no across all consultants
    data_fields = ['raw_eeg', 'eeg_report', 'diagnosis', 'clinical_notes', 'moca', 'adl', 'medication']
    data_coverage = {}
    for field in data_fields:
        vals = [c.get('data', {}).get(field, 'no') for c in consultants]
        data_coverage[field] = {
            'yes': sum(1 for v in vals if v == 'yes'),
            'optional': sum(1 for v in vals if v == 'optional'),
            'no': sum(1 for v in vals if v in ('no', 'metadata', 'aggregated')),
        }

    tier_distribution = [
        {"name": f"Tier {t}", "value": tier_counts.get(t, 0)}
        for t in sorted(tier_counts.keys())
    ]

    role_summary = []
    for c in consultants:
        role_summary.append({
            "id": c.get('id'),
            "name": c.get('name'),
            "tier": c.get('tier'),
            "mandatory": c.get('mandatory', False),
            "role": c.get('role'),
            "tasks": len(c.get('tasks', [])),
            "challenges": len(c.get('challenges', [])),
            "ai_solutions": len(c.get('ai_solutions', [])),
            "tools": len(c.get('tools', [])),
            "assessments": len(c.get('assessment', [])),
        })

    return {
        "available": True,
        "summary": {
            "total_consultants": total,
            "mandatory": mandatory_count,
            "optional": total - mandatory_count,
            "tier_1": tier_counts.get(1, 0),
            "tier_2": tier_counts.get(2, 0),
            "total_tasks": total_tasks,
            "total_challenges": total_challenges,
            "total_ai_solutions": total_ai_solutions,
            "total_documents": total_docs,
            "total_compliance_docs": total_compliance,
            "total_assessments": total_assessments,
            "total_tools": total_tools,
            "ai_coverage_pct": round(total_ai_solutions / total_challenges * 100, 1) if total_challenges else 0,
        },
        "tier_distribution": tier_distribution,
        "data_coverage": data_coverage,
        "role_summary": role_summary,
    }


def breakdown():
    """Per-consultant detail: tasks, challenges, AI solutions, data, tools, docs."""
    cfg = _load('consultant_matrix.json')
    if not cfg:
        return {"available": False}

    consultants = cfg.get('consultants', [])
    roles = []
    ai_solution_list = []

    for c in consultants:
        roles.append({
            "id": c.get('id'),
            "name": c.get('name'),
            "tier": c.get('tier'),
            "mandatory": c.get('mandatory', False),
            "role": c.get('role'),
            "objective": c.get('objective'),
            "tasks": c.get('tasks', []),
            "challenges": c.get('challenges', []),
            "documents": c.get('documents', []),
            "compliance_docs": c.get('compliance_docs', []),
            "internal_tasks": c.get('internal_tasks', []),
            "patient_documents": c.get('patient_documents', []),
            "patient_questionnaire": c.get('patient_questionnaire', []),
            "assessment": c.get('assessment', []),
            "data": c.get('data', {}),
            "tools": c.get('tools', []),
            "ai_solutions": c.get('ai_solutions', []),
        })

        for ai in c.get('ai_solutions', []):
            ai_solution_list.append({
                "consultant": c.get('name'),
                "challenge": ai.get('challenge'),
                "ai": ai.get('ai'),
            })

    # Build data requirements matrix
    data_fields = ['raw_eeg', 'eeg_report', 'diagnosis', 'clinical_notes', 'moca', 'adl', 'medication']
    data_matrix = []
    for c in consultants:
        row = {"consultant": c.get('name'), "id": c.get('id')}
        for field in data_fields:
            row[field] = c.get('data', {}).get(field, 'no')
        data_matrix.append(row)

    return {
        "available": True,
        "roles": roles,
        "ai_solutions": ai_solution_list,
        "data_matrix": data_matrix,
        "data_fields": data_fields,
        "core_team": cfg.get('core_team_mandatory', []),
        "recommended_addons": cfg.get('recommended_addons', []),
    }


def definitions():
    """Glossary, tier legend, and references."""
    return {
        "available": True,
        "tiers": [
            {"tier": 1, "label": "Core / Mandatory", "description": "Essential for clinical validity and regulatory compliance. Must be engaged."},
            {"tier": 2, "label": "Recommended", "description": "Strongly recommended for comprehensive coverage; engagement depends on scope."},
        ],
        "data_requirement_legend": [
            {"value": "yes", "meaning": "Required — consultant needs this data to perform their role"},
            {"value": "optional", "meaning": "Helpful but not essential — used if available"},
            {"value": "no", "meaning": "Not needed for this role"},
            {"value": "metadata", "meaning": "Only metadata/summary, not raw data"},
            {"value": "aggregated", "meaning": "Aggregated/statistical view, not individual records"},
        ],
        "glossary": [
            {"term": "ILAE", "definition": "International League Against Epilepsy — gold-standard seizure/epilepsy classification system"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — model interpretability via feature attribution"},
            {"term": "ICA", "definition": "Independent Component Analysis — EEG artifact separation technique"},
            {"term": "MoCA", "definition": "Montreal Cognitive Assessment — cognitive screening tool (30-point scale)"},
            {"term": "MMSE", "definition": "Mini-Mental State Examination — cognitive screening for orientation, memory, attention"},
            {"term": "ADL", "definition": "Activities of Daily Living — functional independence measures"},
            {"term": "QOLIE-31", "definition": "Quality of Life in Epilepsy Inventory — 31-item patient-reported QoL measure"},
            {"term": "PHQ-9", "definition": "Patient Health Questionnaire-9 — depression severity screening tool"},
            {"term": "GAD-7", "definition": "Generalized Anxiety Disorder 7-item scale — anxiety severity screening"},
            {"term": "DPIA", "definition": "Data Protection Impact Assessment — GDPR-mandated privacy risk assessment"},
            {"term": "IED", "definition": "Interictal Epileptiform Discharge — spike/sharp-wave EEG abnormality between seizures"},
            {"term": "HFO", "definition": "High-Frequency Oscillation — 80-500 Hz EEG biomarker linked to seizure onset zone"},
            {"term": "AIF360", "definition": "IBM AI Fairness 360 — toolkit for detecting/mitigating algorithmic bias"},
            {"term": "IRB", "definition": "Institutional Review Board — ethics committee overseeing human-subjects research"},
        ],
        "clinical_notes": [
            "Consultant engagement is advisory; all decisions require clinical sign-off",
            "Tier 1 (core) consultants must be engaged for any clinical validation claim",
            "AI solutions augment, never replace, human clinical judgment (HITL principle)",
            "Data access follows least-privilege: each role sees only what they need",
        ],
        "references": [
            "ILAE 2017 Classification of Seizure Types (Fisher et al.)",
            "IEC 62304 — Medical device software lifecycle",
            "EU AI Act — High-risk AI system requirements",
            "ISO 14971 — Application of risk management to medical devices",
            "NIST AI RMF 1.0 — AI Risk Management Framework",
        ],
    }
