"""Role Process Flows Dashboard — per-role end-to-end clinical workflow
visualization from config/role_process_flows.json.

Each role has a linear process flow (steps) and a mermaid flowchart.
Provides overview KPIs, per-role step breakdowns, and definitions."""

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
    """Summary KPIs: total roles, total steps, avg steps per role, step distribution."""
    cfg = _load('role_process_flows.json')
    if not cfg:
        return {"available": False, "note": "role_process_flows.json missing"}

    roles = cfg.get('roles', {})
    default_flow = cfg.get('default', {})
    default_steps = default_flow.get('steps', [])

    total_roles = len(roles)
    steps_per_role = {name: len(r.get('steps', [])) for name, r in roles.items()}
    total_steps = sum(steps_per_role.values())
    avg_steps = round(total_steps / total_roles, 1) if total_roles else 0
    max_role = max(steps_per_role, key=steps_per_role.get) if steps_per_role else '--'
    min_role = min(steps_per_role, key=steps_per_role.get) if steps_per_role else '--'

    # Has mermaid chart?
    has_mermaid = sum(1 for r in roles.values() if r.get('mermaid'))

    # Steps distribution for chart
    steps_distribution = [
        {"role": name, "steps": count}
        for name, count in sorted(steps_per_role.items(), key=lambda x: -x[1])
    ]

    # Role summary table
    role_table = []
    for name, r in sorted(roles.items()):
        steps = r.get('steps', [])
        role_table.append({
            "role": name,
            "num_steps": len(steps),
            "first_step": steps[0] if steps else '--',
            "last_step": steps[-1] if steps else '--',
            "has_mermaid": bool(r.get('mermaid')),
        })

    return {
        "available": True,
        "summary": {
            "total_roles": total_roles,
            "total_steps": total_steps,
            "avg_steps_per_role": avg_steps,
            "max_steps_role": max_role,
            "max_steps_count": steps_per_role.get(max_role, 0),
            "min_steps_role": min_role,
            "min_steps_count": steps_per_role.get(min_role, 0),
            "with_mermaid": has_mermaid,
            "default_steps": len(default_steps),
        },
        "steps_distribution": steps_distribution,
        "role_table": role_table,
    }


def breakdown():
    """Per-role detailed step sequences + mermaid definitions."""
    cfg = _load('role_process_flows.json')
    if not cfg:
        return {"available": False}

    roles = cfg.get('roles', {})
    default_flow = cfg.get('default', {})

    role_details = []
    for name in sorted(roles.keys()):
        r = roles[name]
        steps = r.get('steps', [])
        step_rows = [
            {"n": i + 1, "step": s}
            for i, s in enumerate(steps)
        ]
        role_details.append({
            "role": name,
            "num_steps": len(steps),
            "steps": step_rows,
            "mermaid": r.get('mermaid', ''),
        })

    # All steps flat table
    all_steps = []
    for name in sorted(roles.keys()):
        r = roles[name]
        for i, s in enumerate(r.get('steps', [])):
            all_steps.append({
                "role": name,
                "step_num": i + 1,
                "step": s,
            })

    return {
        "available": True,
        "role_details": role_details,
        "all_steps": all_steps,
        "default_flow": {
            "steps": [{"n": i + 1, "step": s} for i, s in enumerate(default_flow.get('steps', []))],
            "mermaid": default_flow.get('mermaid', ''),
        },
    }


def definitions():
    """Glossary, role descriptions, and references for process flows."""
    return {
        "available": True,
        "role_descriptions": [
            {"role": "Neurologist", "description": "Physician specializing in disorders of the nervous system; reviews EEG + clinical data, makes seizure classification and treatment decisions."},
            {"role": "EEG Technician", "description": "Trained technologist who performs EEG recordings — electrode placement, impedance checks, activation procedures, artifact annotation, and signal QC."},
            {"role": "Clinical Neurophysiologist", "description": "Specialist in interpreting neurophysiological studies — background rhythm, epileptiform discharges, seizure evolution, and artifact review."},
            {"role": "Patient", "description": "The person with epilepsy — logs seizure diary, tracks triggers/medications, syncs wearables, views risk reports, and engages in telehealth."},
            {"role": "AI Governance", "description": "Oversight role ensuring AI models meet explainability, fairness, safety, and clinical validation standards before deployment."},
            {"role": "Pharmacist", "description": "Medication specialist — reviews drug interactions, therapeutic drug monitoring (TDM), adherence, side effects, and counsels patients."},
        ],
        "glossary": [
            {"term": "10-20 System", "definition": "International standard for EEG electrode placement using 10% and 20% distances between anatomical landmarks."},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — model-agnostic explainability method assigning feature importance to each prediction."},
            {"term": "ICA", "definition": "Independent Component Analysis — artifact removal technique separating EEG into independent sources."},
            {"term": "ILAE", "definition": "International League Against Epilepsy — global authority on epilepsy classification and treatment guidelines."},
            {"term": "TDM", "definition": "Therapeutic Drug Monitoring — measuring drug levels in blood to optimize dosage and minimize toxicity."},
            {"term": "EDF", "definition": "European Data Format — standard file format for storing EEG and polysomnography recordings."},
            {"term": "HITL", "definition": "Human-In-The-Loop — process requiring human review/override of AI-generated decisions."},
            {"term": "Mermaid", "definition": "Text-based diagramming syntax for rendering flowcharts, sequence diagrams, and other visualizations."},
            {"term": "MRI/EEG Concordance", "definition": "Agreement between structural MRI lesions and EEG-identified seizure focus for surgical planning."},
            {"term": "Activation Procedure", "definition": "Clinical provocations (hyperventilation, photic stimulation, sleep deprivation) used during EEG to elicit epileptiform activity."},
        ],
        "clinical_notes": [
            "Each role's process flow represents the end-to-end workflow from case receipt to audit sign-off.",
            "The default flow applies to roles without a specific process defined.",
            "Mermaid flowcharts encode decision points (diamond nodes) and feedback loops.",
            "Human validation/override gates are embedded in clinical roles to ensure AI outputs are reviewed before final decisions.",
        ],
        "references": [
            "ILAE Classification of Epilepsies (2017) — Scheffer et al., Epilepsia 58(4):512-521",
            "ACNS Guidelines for EEG Recording — American Clinical Neurophysiology Society",
            "EU AI Act (2024) — Risk-based AI governance framework for clinical AI systems",
            "WHO Epilepsy Guidelines — World Health Organization, 2023",
        ],
    }
