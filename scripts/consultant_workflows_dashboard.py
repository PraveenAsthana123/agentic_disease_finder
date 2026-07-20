"""Consultant Workflows Dashboard — per-role step-by-step workflow (phases,
steps with input/task/output), sign-off gates, and role summary from
config/consultant_workflows.json."""

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
    """Summary KPIs: role counts, phase/step distribution, signoff gates."""
    cfg = _load('consultant_workflows.json')
    if not cfg:
        return {"available": False, "note": "consultant_workflows.json missing"}

    workflows = cfg.get('workflows', {})
    total_roles = len(workflows)

    # Count mandatory roles from consultant_matrix.json if available
    matrix = _load('consultant_matrix.json')
    mandatory_count = 0
    if matrix:
        for c in matrix.get('consultants', []):
            if c.get('mandatory'):
                mandatory_count += 1

    role_summary = []
    total_phases = 0
    total_steps = 0
    total_signoffs = 0

    phase_distribution = []
    signoff_distribution = []

    for role_id, role in workflows.items():
        phases = role.get('phases', [])
        signoffs = role.get('signoffs', [])
        phase_count = len(phases)
        step_count = sum(len(p.get('steps', [])) for p in phases)

        role_summary.append({
            "role_id": role_id,
            "name": role.get('name', role_id),
            "phases": phase_count,
            "steps": step_count,
            "signoffs": len(signoffs),
        })

        phase_distribution.append({
            "name": role.get('name', role_id),
            "value": phase_count,
        })
        signoff_distribution.append({
            "name": role.get('name', role_id),
            "value": len(signoffs),
        })

        total_phases += phase_count
        total_steps += step_count
        total_signoffs += len(signoffs)

    avg_phases = round(total_phases / total_roles, 1) if total_roles else 0
    avg_steps = round(total_steps / total_phases, 1) if total_phases else 0

    return {
        "available": True,
        "summary": {
            "total_roles": total_roles,
            "total_phases": total_phases,
            "total_steps": total_steps,
            "total_signoffs": total_signoffs,
            "avg_phases_per_role": avg_phases,
            "avg_steps_per_phase": avg_steps,
            "mandatory_roles": mandatory_count,
        },
        "role_summary": role_summary,
        "phase_distribution": phase_distribution,
        "signoff_distribution": signoff_distribution,
    }


def breakdown():
    """Detailed view: all roles with phases expanded, each step with
    input/task/output, plus a flat list of all steps."""
    cfg = _load('consultant_workflows.json')
    if not cfg:
        return {"available": False}

    workflows = cfg.get('workflows', {})

    roles = []
    all_steps = []

    for role_id, role in workflows.items():
        role_name = role.get('name', role_id)
        phases = role.get('phases', [])
        signoffs = role.get('signoffs', [])

        phase_detail = []
        for phase in phases:
            phase_name = phase.get('name', '?')
            steps = phase.get('steps', [])
            step_detail = []
            for s in steps:
                step_obj = {
                    "step": s.get('step', '?'),
                    "input": s.get('input', ''),
                    "task": s.get('task', ''),
                    "output": s.get('output', ''),
                }
                step_detail.append(step_obj)
                all_steps.append({
                    "role_id": role_id,
                    "role_name": role_name,
                    "phase_name": phase_name,
                    **step_obj,
                })
            phase_detail.append({
                "name": phase_name,
                "steps": step_detail,
                "step_count": len(step_detail),
            })

        roles.append({
            "role_id": role_id,
            "name": role_name,
            "summary": role.get('summary', ''),
            "phases": phase_detail,
            "signoffs": signoffs,
            "phase_count": len(phase_detail),
            "step_count": sum(p['step_count'] for p in phase_detail),
        })

    return {
        "available": True,
        "roles": roles,
        "all_steps": all_steps,
        "meta": {
            "title": cfg.get('title', ''),
            "purpose": cfg.get('purpose', ''),
            "updated_at": cfg.get('updated_at', ''),
        },
    }


def definitions():
    """Consultant workflow terminology, role descriptions, glossary,
    clinical notes, and references."""
    cfg = _load('consultant_workflows.json')
    workflows = cfg.get('workflows', {}) if cfg else {}

    roles = []
    for role_id, role in workflows.items():
        roles.append({
            "id": role_id,
            "name": role.get('name', role_id),
            "summary": role.get('summary', ''),
        })

    return {
        "available": True,
        "roles": roles,
        "glossary": [
            {"term": "ILAE", "definition": "International League Against Epilepsy — the global authority on epilepsy classification, diagnosis standards, and treatment guidelines."},
            {"term": "PNES", "definition": "Psychogenic Non-Epileptic Seizures — seizure-like events not caused by abnormal electrical brain activity; require psychiatric evaluation to differentiate from epileptic seizures."},
            {"term": "EEG", "definition": "Electroencephalogram — a recording of electrical activity along the scalp, used to detect abnormal brain patterns such as spikes, sharp waves, and seizure discharges."},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — a game-theoretic approach to explain AI model predictions by attributing importance to each input feature."},
            {"term": "MoCA", "definition": "Montreal Cognitive Assessment — a 30-point screening tool for detecting mild cognitive impairment across multiple domains (memory, attention, executive, language, visuospatial)."},
            {"term": "PHQ-9", "definition": "Patient Health Questionnaire-9 — a validated 9-item self-report measure of depression severity scored 0-27."},
            {"term": "GAD-7", "definition": "Generalized Anxiety Disorder 7-item scale — a brief self-report questionnaire for screening and measuring anxiety severity scored 0-21."},
            {"term": "ICA", "definition": "Independent Component Analysis — a computational method for separating EEG signals into independent source components, commonly used for artifact removal (eye blinks, muscle, heart)."},
            {"term": "FCD", "definition": "Focal Cortical Dysplasia — a malformation of cortical development that is a common cause of drug-resistant epilepsy, identifiable on MRI."},
            {"term": "Biomarker", "definition": "A measurable indicator of a biological state or condition. In EEG epilepsy research: spike frequency, band power ratios, connectivity metrics, and entropy measures."},
            {"term": "Sign-off Gate", "definition": "A mandatory approval checkpoint where a qualified clinical consultant reviews and validates a specific deliverable before the workflow proceeds to the next phase."},
            {"term": "Ground Truth", "definition": "The verified, clinician-confirmed correct labels (diagnosis, seizure type, annotations) used as the reference standard for training and evaluating AI models."},
        ],
        "clinical_notes": [
            "Human clinical oversight is mandatory at every phase of an AI-assisted epilepsy research workflow. No AI output should be treated as a final clinical decision without consultant review.",
            "Each consultant role addresses a distinct clinical domain (neurology, neurophysiology, psychiatry, psychology, radiology, EEG technology). Overlap is intentional — cross-validation between roles strengthens clinical validity.",
            "Sign-off gates are not administrative checkboxes; they represent genuine expert validation that the data, labels, features, or outputs meet clinical standards before downstream use.",
            "PNES differentiation requires psychiatrist involvement — EEG and neurological exam alone cannot reliably distinguish PNES from epileptic seizures in all cases.",
            "Cognitive assessment (MoCA/MMSE) should be correlated with EEG findings (temporal spikes -> memory, frontal -> executive) to validate AI cognitive-risk predictions.",
        ],
        "references": [
            "Fisher RS, et al. ILAE official report: a practical clinical definition of epilepsy. Epilepsia. 2014;55(4):475-482.",
            "Scheffer IE, et al. ILAE classification of the epilepsies: Position paper of the ILAE Commission for Classification and Terminology. Epilepsia. 2017;58(4):512-521.",
            "Nasreddine ZS, et al. The Montreal Cognitive Assessment, MoCA: a brief screening tool for mild cognitive impairment. J Am Geriatr Soc. 2005;53(4):695-699.",
            "Kroenke K, Spitzer RL, Williams JB. The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med. 2001;16(9):606-613.",
            "Spitzer RL, Kroenke K, Williams JBW, Lowe B. A brief measure for assessing generalized anxiety disorder: the GAD-7. Arch Intern Med. 2006;166(10):1092-1097.",
            "Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.",
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
    print("\n=== DEFINITIONS ===")
    print(json.dumps(definitions(), indent=2, default=str))
