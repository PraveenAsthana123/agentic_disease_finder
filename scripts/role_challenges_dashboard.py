"""Role Challenges Dashboard — per-role clinical workflow challenges and AI mitigations
by build status (built/partial/planned), from config/role_challenges.json."""

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
    """Summary KPIs: total roles, total challenges, status distribution, per-role counts."""
    cfg = _load('role_challenges.json')
    if not cfg:
        return {"available": False, "note": "role_challenges.json missing"}

    roles = cfg.get('roles', [])
    total_roles = len(roles)

    all_items = [item for role in roles for item in role.get('items', [])]
    total_challenges = len(all_items)

    status_counts = Counter(item.get('status', 'unknown') for item in all_items)
    built = status_counts.get('built', 0)
    partial = status_counts.get('partial', 0)
    planned = status_counts.get('planned', 0)
    built_pct = round(built / total_challenges * 100, 1) if total_challenges else 0

    roles_fully_built = sum(
        1 for role in roles
        if all(item.get('status', 'unknown') == 'built' for item in role.get('items', []))
    )

    status_distribution = [
        {"name": s, "value": status_counts.get(s, 0)}
        for s in ['built', 'partial', 'planned']
        if status_counts.get(s, 0) > 0
    ]

    challenges_per_role = []
    for role in roles:
        items = role.get('items', [])
        role_status = Counter(item.get('status', 'unknown') for item in items)
        challenges_per_role.append({
            "name": role.get('role', ''),
            "value": len(items),
            "built": role_status.get('built', 0),
            "partial": role_status.get('partial', 0),
            "planned": role_status.get('planned', 0),
        })

    role_summary = []
    for role in roles:
        items = role.get('items', [])
        role_status = Counter(item.get('status', 'unknown') for item in items)
        role_summary.append({
            "role": role.get('role', ''),
            "icon": role.get('icon', ''),
            "total": len(items),
            "built": role_status.get('built', 0),
            "partial": role_status.get('partial', 0),
            "planned": role_status.get('planned', 0),
        })

    return {
        "available": True,
        "summary": {
            "total_roles": total_roles,
            "total_challenges": total_challenges,
            "built": built,
            "partial": partial,
            "planned": planned,
            "built_pct": built_pct,
            "roles_fully_built": roles_fully_built,
        },
        "status_distribution": status_distribution,
        "challenges_per_role": challenges_per_role,
        "role_summary": role_summary,
    }


def breakdown():
    """Per-role full item list with challenge, AI mitigation, status, dashboard, and solution."""
    cfg = _load('role_challenges.json')
    if not cfg:
        return {"available": False, "note": "role_challenges.json missing"}

    roles = cfg.get('roles', [])

    roles_out = []
    for role in roles:
        items = []
        for item in role.get('items', []):
            items.append({
                "challenge": item.get('challenge', ''),
                "ai": item.get('ai', ''),
                "status": item.get('status', ''),
                "dashboard": item.get('dashboard', ''),
                "solution": item.get('solution', ''),
                "note": item.get('note', ''),
                "endpoints": item.get('endpoints', []),
            })
        roles_out.append({
            "role": role.get('role', ''),
            "icon": role.get('icon', ''),
            "items": items,
        })

    return {
        "available": True,
        "roles": roles_out,
    }


def definitions():
    """Glossary, role descriptions, status legend, clinical notes, and references."""
    return {
        "available": True,
        "role_descriptions": [
            {"role": "Neurologist", "description": "Diagnoses and manages epilepsy; interprets EEG studies, makes treatment decisions, and carries medico-legal responsibility for the clinical record."},
            {"role": "EEG Technician", "description": "Acquires EEG recordings; responsible for electrode placement, impedance quality, artifact avoidance, and session documentation."},
            {"role": "Psychiatrist", "description": "Evaluates psychiatric comorbidities in epilepsy (depression, anxiety, PNES) and manages psychotropic co-medication risk."},
            {"role": "Occupational Therapist", "description": "Assesses functional recovery and daily-living capacity; tracks goal attainment and determines rehabilitation intensity."},
            {"role": "Clinical Psychologist", "description": "Administers and scores neuropsychological batteries (MoCA, WAIS); monitors cognitive trajectory and psychological wellbeing."},
            {"role": "Radiologist", "description": "Interprets structural MRI; identifies lesions, focal cortical dysplasia, and correlates imaging findings with EEG focus localisation."},
            {"role": "IoT Engineer", "description": "Maintains wearable EEG devices (Emotiv, etc.), edge gateways, and real-time data pipelines for ambulatory monitoring."},
            {"role": "IRB / Governance Reviewer", "description": "Ensures AI decisions are governed, auditable, fair, and compliant with ethics board requirements and regulatory standards."},
            {"role": "Patient", "description": "The end beneficiary: receives plain-language summaries, faster turnarounds, continuous monitoring, and a safety alert channel."},
        ],
        "status_legend": [
            {"status": "built", "description": "Feature is live and verified — endpoint returns 200, dashboard is wired in the frontend, and real data flows through."},
            {"status": "partial", "description": "Partially implemented — core logic exists but integration, edge cases, or UI wiring is incomplete."},
            {"status": "planned", "description": "Planned for a future sprint — design is specified but no code has been written yet."},
        ],
        "glossary": [
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — assigns each input feature a contribution score to explain individual AI predictions."},
            {"term": "ICA", "definition": "Independent Component Analysis — blind source separation technique used to isolate and remove EEG artifacts (eye blink, muscle, ECG)."},
            {"term": "HITL", "definition": "Human-In-The-Loop — mandatory clinician oversight gate where the AI surfaces a decision and a human must confirm or override."},
            {"term": "PNES", "definition": "Psychogenic Non-Epileptic Seizures — seizure-like episodes without ictal EEG correlate; require psychiatric differential from epilepsy."},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — LLM answers grounded in retrieved documents (patient records, clinical notes) to prevent hallucination."},
            {"term": "IED", "definition": "Interictal Epileptiform Discharge — spike or sharp wave between clinical seizures; key EEG marker for epilepsy diagnosis."},
            {"term": "COPM", "definition": "Canadian Occupational Performance Measure — standardised self-report instrument scoring occupational performance and satisfaction."},
            {"term": "MoCA", "definition": "Montreal Cognitive Assessment — 30-point screening tool for mild cognitive impairment; auto-scored in this system."},
            {"term": "FIM", "definition": "Functional Independence Measure — 18-item scale rating motor and cognitive independence for rehabilitation tracking."},
            {"term": "ILAE", "definition": "International League Against Epilepsy — sets global classification and reporting standards for epilepsy diagnosis and EEG findings."},
            {"term": "DI", "definition": "Disparate Impact — fairness metric measuring outcome rate ratio across protected groups; gate threshold ≥ 0.80."},
            {"term": "EO", "definition": "Equal Opportunity — fairness metric checking true-positive-rate parity across demographic groups."},
        ],
        "clinical_notes": [
            "Challenges are mapped to the 9 clinical roles involved in the epilepsy AI governance system — every role with a real workflow touchpoint is covered.",
            "Status 'built' means at least one verified API endpoint (200 OK) and a frontend dashboard component wired in the nav.",
            "IoT Engineer challenges are the primary partial-build area: real-time wearable streaming requires hardware integration beyond the simulator.",
            "The Patient role challenges drive the plain-language reporting and caregiver-readiness dashboards, closing the loop from diagnosis to self-management.",
        ],
        "references": [
            "ILAE Classification of Epilepsies (2017) — Fisher et al., Epilepsia 58(4):512-521",
            "ACNS Guidelines for EEG Recording — American Clinical Neurophysiology Society (2016)",
            "EU AI Act Article 13 — Transparency and Provision of Information to Deployers (2024)",
            "COPM Manual (5th ed.) — Law et al., CAOT Publications ACE",
            "MoCA Administration and Scoring Guide — Nasreddine et al. (2005)",
        ],
    }


if __name__ == '__main__':
    import sys
    fn = sys.argv[1] if len(sys.argv) > 1 else 'overview'
    dispatch = {'overview': overview, 'breakdown': breakdown, 'definitions': definitions}
    if fn not in dispatch:
        print(f"Unknown function '{fn}'. Choose: overview, breakdown, definitions")
        sys.exit(1)
    print(json.dumps(dispatch[fn](), indent=2))
