"""Process Simulations Dashboard — per-role end-to-end process simulations,
step layers, auto vs manual distribution, endpoint mapping
from config/simulations.json.
7 roles, ordered pipeline steps per role."""

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
    """Summary KPIs: role count, total steps, layer distribution, auto/manual ratio."""
    cfg = _load('simulations.json')
    if not cfg:
        return {"available": False, "note": "simulations.json missing"}

    roles = cfg.get('roles', [])
    total_roles = len(roles)
    total_steps = sum(len(r.get('steps', [])) for r in roles)

    # Layer distribution across all steps
    all_layers = []
    all_modes = []
    all_actors = []
    for r in roles:
        for s in r.get('steps', []):
            all_layers.append(s.get('layer', 'unknown'))
            all_modes.append(s.get('mode', 'unknown'))
            all_actors.append(s.get('actor', 'unknown'))

    layer_counts = Counter(all_layers)
    layer_distribution = [
        {"name": layer.capitalize(), "value": count}
        for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1])
    ]

    mode_counts = Counter(all_modes)
    mode_distribution = [
        {"name": mode.capitalize(), "value": count}
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1])
    ]

    actor_counts = Counter(all_actors)
    actor_distribution = [
        {"name": actor, "value": count}
        for actor, count in sorted(actor_counts.items(), key=lambda x: -x[1])
    ]

    auto_count = mode_counts.get('auto', 0)
    manual_count = mode_counts.get('manual', 0)

    # Steps per role for bar chart
    steps_per_role = [
        {"name": r.get('role', ''), "value": len(r.get('steps', []))}
        for r in roles
    ]

    # Summary table
    roles_table = [
        {
            "role": r.get('role', ''),
            "icon": r.get('icon', ''),
            "process": r.get('process', ''),
            "steps": len(r.get('steps', [])),
            "auto": sum(1 for s in r.get('steps', []) if s.get('mode') == 'auto'),
            "manual": sum(1 for s in r.get('steps', []) if s.get('mode') == 'manual'),
            "layers": ', '.join(sorted(set(s.get('layer', '') for s in r.get('steps', [])))),
        }
        for r in roles
    ]

    return {
        "available": True,
        "kpis": {
            "total_roles": total_roles,
            "total_steps": total_steps,
            "auto_steps": auto_count,
            "manual_steps": manual_count,
            "unique_layers": len(layer_counts),
            "unique_actors": len(actor_counts),
        },
        "layer_distribution": layer_distribution,
        "mode_distribution": mode_distribution,
        "actor_distribution": actor_distribution,
        "steps_per_role": steps_per_role,
        "roles_table": roles_table,
    }


def breakdown():
    """Per-role step details, endpoint mapping, layer/mode breakdown."""
    cfg = _load('simulations.json')
    if not cfg:
        return {"available": False}

    roles = cfg.get('roles', [])

    roles_detail = []
    for r in roles:
        steps = r.get('steps', [])
        step_items = []
        for i, s in enumerate(steps):
            step_items.append({
                "step": i + 1,
                "layer": s.get('layer', ''),
                "mode": s.get('mode', ''),
                "actor": s.get('actor', ''),
                "input": s.get('input', ''),
                "process": s.get('process', ''),
                "output": s.get('output', ''),
                "maps_to": s.get('maps_to', ''),
            })
        layer_counts = Counter(s.get('layer', '') for s in steps)
        mode_counts = Counter(s.get('mode', '') for s in steps)
        roles_detail.append({
            "role": r.get('role', ''),
            "icon": r.get('icon', ''),
            "process": r.get('process', ''),
            "steps": step_items,
            "layer_breakdown": [{"name": k.capitalize(), "value": v} for k, v in layer_counts.items()],
            "mode_breakdown": [{"name": k.capitalize(), "value": v} for k, v in mode_counts.items()],
        })

    # Endpoint mapping — all maps_to values
    endpoint_map = []
    for r in roles:
        for s in r.get('steps', []):
            mt = s.get('maps_to', '')
            if mt:
                endpoint_map.append({
                    "role": r.get('role', ''),
                    "step": s.get('process', ''),
                    "maps_to": mt,
                    "mode": s.get('mode', ''),
                })

    return {
        "available": True,
        "roles": roles_detail,
        "endpoint_map": endpoint_map,
    }


def definitions():
    """Definitions: layer types, mode types, glossary, references."""
    return {
        "available": True,
        "layer_legend": [
            {"layer": "Data", "description": "Data acquisition, upload, parsing, feature extraction"},
            {"layer": "Process", "description": "Signal processing, artifact detection, clinical judgment, quality checks"},
            {"layer": "Accuracy", "description": "ML model inference, XAI explanations, differential analysis"},
            {"layer": "Reporting", "description": "Report generation, audit trail, PDF output, transaction logging"},
            {"layer": "Backend", "description": "System-level orchestration and API routing"},
        ],
        "mode_legend": [
            {"mode": "Auto", "description": "Fully automated by AI pipeline, model, or system — no human intervention"},
            {"mode": "Manual", "description": "Requires human action — clinician judgment, data entry, override decision"},
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography — recording of brain electrical activity via scalp electrodes"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — game-theoretic feature importance for ML models"},
            {"term": "XAI", "definition": "Explainable AI — methods that make model predictions interpretable to humans"},
            {"term": "HITL", "definition": "Human-In-The-Loop — human oversight/override of AI decisions"},
            {"term": "ADL", "definition": "Activities of Daily Living — functional independence measure used by OTs"},
            {"term": "QC", "definition": "Quality Control — systematic process to ensure data/signal meets standards"},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — LLM pattern combining retrieval with generation"},
            {"term": "RAI", "definition": "Responsible AI — fairness, transparency, privacy, and safety practices"},
            {"term": "PII", "definition": "Personally Identifiable Information — data that could identify an individual"},
            {"term": "IRB", "definition": "Institutional Review Board — ethics committee overseeing research involving humans"},
        ],
        "clinical_notes": [
            "Each role simulation reflects a real clinical workflow with ordered dependencies.",
            "Auto steps map to built or planned API endpoints; manual steps require clinician interaction.",
            "The maps_to field links each step to the actual system component that implements it.",
            "Override/HITL steps ensure AI never makes unsupervised clinical decisions.",
        ],
        "references": [
            {"id": "ACNS", "title": "American Clinical Neurophysiology Society — EEG Guidelines"},
            {"id": "ILAE", "title": "International League Against Epilepsy — Classification & Terminology"},
            {"id": "FDA-SaMD", "title": "FDA Software as Medical Device — Clinical Decision Support Guidance"},
            {"id": "IEC-62304", "title": "IEC 62304 — Medical Device Software Lifecycle Processes"},
            {"id": "HIPAA", "title": "Health Insurance Portability and Accountability Act — Privacy Rule"},
        ],
    }
