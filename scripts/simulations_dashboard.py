"""Per-Role End-to-End Process Simulations Dashboard.

Visualizes config/simulations.json — 7 roles, each with ordered steps
describing the full human+AI pipeline (layer, mode, actor, input, process,
output, maps_to). Provides overview KPIs, per-role step breakdowns, and
definitions."""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load():
    path = os.path.join(_CFG, 'simulations.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: roles, steps, mode/layer distribution, actor breakdown."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "simulations.json missing"}

    roles = cfg.get('roles', [])
    all_steps = [s for r in roles for s in r.get('steps', [])]

    total_roles = len(roles)
    total_steps = len(all_steps)
    avg_steps = round(total_steps / total_roles, 1) if total_roles else 0

    mode_counts = Counter(s.get('mode', 'unknown') for s in all_steps)
    layer_counts = Counter(s.get('layer', 'unknown') for s in all_steps)
    actor_counts = Counter(s.get('actor', 'unknown') for s in all_steps)

    auto_pct = round(mode_counts.get('auto', 0) / total_steps * 100, 1) if total_steps else 0

    steps_per_role = [
        {"role": r.get('role', '?'), "icon": r.get('icon', ''), "count": len(r.get('steps', []))}
        for r in sorted(roles, key=lambda x: -len(x.get('steps', [])))
    ]

    role_table = []
    for r in roles:
        steps = r.get('steps', [])
        modes = Counter(s.get('mode', '') for s in steps)
        role_table.append({
            "role": r.get('role', '?'),
            "icon": r.get('icon', ''),
            "process": r.get('process', ''),
            "total_steps": len(steps),
            "auto_steps": modes.get('auto', 0),
            "manual_steps": modes.get('manual', 0),
        })

    return {
        "available": True,
        "generated_at": "2026-08-06",
        "kpis": [
            {"label": "Roles", "value": total_roles, "color": "primary"},
            {"label": "Total Steps", "value": total_steps, "color": "info"},
            {"label": "Avg Steps/Role", "value": avg_steps, "color": "secondary"},
            {"label": "AI-Automated %", "value": f"{auto_pct}%", "color": "success"},
        ],
        "summary": {
            "total_roles": total_roles,
            "total_steps": total_steps,
            "avg_steps_per_role": avg_steps,
            "auto_pct": auto_pct,
            "manual_pct": round(100 - auto_pct, 1),
        },
        "mode_distribution": [
            {"name": k, "value": v}
            for k, v in sorted(mode_counts.items(), key=lambda x: -x[1])
        ],
        "layer_distribution": [
            {"name": k, "value": v}
            for k, v in sorted(layer_counts.items(), key=lambda x: -x[1])
        ],
        "actor_distribution": [
            {"name": k, "value": v}
            for k, v in sorted(actor_counts.items(), key=lambda x: -x[1])
        ],
        "steps_per_role": steps_per_role,
        "role_table": role_table,
    }


def breakdown():
    """Per-role step tables with layer/mode/actor/input/process/output/maps_to."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "simulations.json missing"}

    roles = cfg.get('roles', [])
    role_details = []
    for r in roles:
        steps = r.get('steps', [])
        enriched = []
        for i, s in enumerate(steps, 1):
            enriched.append({
                "step": i,
                "layer": s.get('layer', ''),
                "mode": s.get('mode', ''),
                "actor": s.get('actor', ''),
                "input": s.get('input', ''),
                "process": s.get('process', ''),
                "output": s.get('output', ''),
                "maps_to": s.get('maps_to', ''),
            })
        role_details.append({
            "role": r.get('role', '?'),
            "icon": r.get('icon', ''),
            "process": r.get('process', ''),
            "steps": enriched,
        })

    return {
        "available": True,
        "role_details": role_details,
    }


def definitions():
    """Layer definitions, mode glossary, actor types, and references."""
    return {
        "available": True,
        "layers": [
            {"layer": "data", "description": "Data ingestion, storage, parsing, extraction steps"},
            {"layer": "process", "description": "Clinical or pipeline processing — QC, analysis, decision routing"},
            {"layer": "accuracy", "description": "AI/ML inference — model prediction, XAI explanation, evaluation"},
            {"layer": "reporting", "description": "Report generation, audit trail, notification delivery"},
            {"layer": "backend", "description": "Infrastructure — API calls, database writes, scheduling"},
        ],
        "modes": [
            {"mode": "auto", "description": "Executed autonomously by AI pipeline or system — no human input required"},
            {"mode": "manual", "description": "Requires direct human action — data entry, review, override, approval"},
        ],
        "actors": [
            {"actor": "Neurologist", "role": "Clinical lead — EEG read, override, sign-off"},
            {"actor": "EEG Technician", "role": "Signal acquisition, impedance check, QC re-recording"},
            {"actor": "Psychiatrist", "role": "Comorbidity screening, differential diagnosis"},
            {"actor": "OT", "role": "Functional outcome tracking, rehab plan adjustment"},
            {"actor": "Reviewer", "role": "IRB/Governance — audit review, approval gating"},
            {"actor": "IoT Engineer", "role": "Device fleet ops, gateway health, alert chain"},
            {"actor": "Patient", "role": "Consent, profile fill, mobile diary interaction"},
            {"actor": "Pipeline", "role": "Automated backend — feature extraction, signal QC"},
            {"actor": "Model", "role": "Trained ML model — seizure-risk inference"},
            {"actor": "XAI", "role": "SHAP/Captum — feature attribution for explainability"},
            {"actor": "Council", "role": "Multi-agent council — security gate, RAG orchestration"},
            {"actor": "RAG+Eval", "role": "Retrieve-Augment-Generate with grounding evaluation"},
            {"actor": "Compliance", "role": "Fairness + PII check automation"},
            {"actor": "Alert", "role": "SOS escalation chain — caregiver notification"},
            {"actor": "System", "role": "Audit trail writer, PDF report generator"},
            {"actor": "Gateway", "role": "IoT gateway — packet validation, buffering, heartbeat"},
            {"actor": "Edge", "role": "Edge compute — local feature extraction, PII scrub"},
            {"actor": "Edge model", "role": "Lightweight on-device inference model"},
            {"actor": "Decision", "role": "Confidence gate — route alert vs silence"},
            {"actor": "Wearable/Emotiv", "role": "Hardware sensor — EEG/bio signal source"},
        ],
        "glossary": {
            "maps_to": "Real backend endpoint or DB table this step corresponds to",
            "layer": "Functional layer of the system where this step executes",
            "mode": "Whether step is automated (auto) or requires human action (manual)",
            "process": "What happens at this step — the transformation or action",
        },
        "note": "Simulations are per-role e2e walkthroughs for training, audit, and governance demonstration.",
    }
