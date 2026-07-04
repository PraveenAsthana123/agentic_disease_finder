"""Dataset Coverage Dashboard — modality map, AI stream status,
implementation phases, target scale, and research context from config/dataset_coverage.json."""

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
    """Summary KPIs: modality counts by status/tier, AI stream status,
    phase progress, target scale."""
    dc = _load('dataset_coverage.json')
    if not dc:
        return {"available": False, "note": "dataset_coverage.json missing"}

    modalities = dc.get('modalities', [])
    ai_streams = dc.get('ai_streams', [])
    phases = dc.get('phases', [])
    target_scale = dc.get('target_scale', {})

    # Modality status counts
    mod_status = Counter(m.get('status', '?') for m in modalities)
    # Modality tier distribution
    mod_tier = Counter(m.get('tier', 0) for m in modalities)
    # Phase status
    phase_status = Counter(p.get('status', '?') for p in phases)

    # AI stream summary
    ai_built = sum(1 for s in ai_streams if 'built' in s.get('status', ''))
    ai_planned = sum(1 for s in ai_streams if 'planned' in s.get('status', ''))
    ai_scaffold = sum(1 for s in ai_streams if 'scaffold' in s.get('status', ''))

    total_modalities = len(modalities)
    built_modalities = mod_status.get('built', 0)
    coverage_pct = round(built_modalities / total_modalities * 100, 1) if total_modalities else 0

    return {
        "available": True,
        "summary": {
            "total_modalities": total_modalities,
            "modalities_built": built_modalities,
            "modalities_scaffold": mod_status.get('scaffold', 0),
            "modalities_planned": mod_status.get('planned', 0),
            "coverage_pct": coverage_pct,
            "ai_streams_total": len(ai_streams),
            "ai_streams_built": ai_built,
            "ai_streams_scaffold": ai_scaffold,
            "ai_streams_planned": ai_planned,
            "phases_total": len(phases),
            "phases_built": phase_status.get('built', 0),
            "phases_partial": phase_status.get('partial', 0),
            "phases_planned": phase_status.get('planned', 0),
        },
        "target_scale": target_scale,
        "research_question": dc.get('novel_research_question', ''),
        "focus": dc.get('focus', ''),
        "modality_status_distribution": {
            "built": mod_status.get('built', 0),
            "scaffold": mod_status.get('scaffold', 0),
            "planned": mod_status.get('planned', 0),
        },
        "modality_tier_distribution": {
            f"tier_{k}": v for k, v in sorted(mod_tier.items())
        },
        "phase_status_distribution": {
            "built": phase_status.get('built', 0),
            "partial": phase_status.get('partial', 0),
            "planned": phase_status.get('planned', 0),
        },
        "ai_stream_distribution": {
            "built": ai_built,
            "scaffold": ai_scaffold,
            "planned": ai_planned,
        },
    }


def breakdown():
    """Detailed view: all modalities, AI streams, phases, provider questions."""
    dc = _load('dataset_coverage.json')
    if not dc:
        return {"available": False}

    modalities = dc.get('modalities', [])
    ai_streams = dc.get('ai_streams', [])
    phases = dc.get('phases', [])

    modality_detail = []
    for m in modalities:
        modality_detail.append({
            "code": m.get('code', '?'),
            "name": m.get('name', '?'),
            "measures": m.get('measures', ''),
            "ai_potential": m.get('ai_potential', ''),
            "tier": m.get('tier', 0),
            "phase": m.get('phase', 0),
            "status": m.get('status', '?'),
            "where": m.get('where', ''),
        })

    ai_detail = []
    for s in ai_streams:
        ai_detail.append({
            "id": s.get('id', '?'),
            "input": s.get('input', ''),
            "output": s.get('output', ''),
            "models": s.get('models', []),
            "status": s.get('status', '?'),
        })

    phase_detail = []
    for p in phases:
        phase_detail.append({
            "phase": p.get('phase', 0),
            "name": p.get('name', '?'),
            "status": p.get('status', '?'),
        })

    return {
        "available": True,
        "modalities": modality_detail,
        "ai_streams": ai_detail,
        "phases": phase_detail,
        "provider_questions": dc.get('provider_questions', []),
    }


def definitions():
    """Dataset coverage terminology and definitions."""
    return {
        "available": True,
        "definitions": [
            {"term": "Modality", "definition": "A distinct type of neurophysiology data (e.g., EEG, EMG, Video) collected as part of the dataset. Each modality has a tier (priority) and phase (planned introduction)."},
            {"term": "Tier", "definition": "Priority classification for data modalities. Tier 1 = core (must-have for MVP), Tier 2 = enrichment (adds multimodal value), Tier 3 = extended (future/research scope)."},
            {"term": "Phase", "definition": "Implementation stage when a modality becomes active. Phase 1 = Core (EEG + demographics), Phase 2 = Multimodal, Phase 3 = Autonomic, Phase 4 = Neuromuscular/EP."},
            {"term": "AI Stream", "definition": "A machine-learning pipeline consuming one or more modalities. Each stream has defined inputs, outputs, candidate models, and a development status."},
            {"term": "Coverage Percentage", "definition": "Ratio of built modalities to total modalities. Indicates how much of the target data ecosystem is actively collecting and processing data."},
            {"term": "Target Scale", "definition": "The planned dataset size in patients, EEG studies, video-EEG sessions, and retrospective years — sets feasibility expectations for statistical power."},
            {"term": "Provider Questions", "definition": "Due-diligence questions asked to data-providing hospitals/sites to assess whether their archives meet the required modalities and quality."},
            {"term": "Scaffold", "definition": "A modality whose data schema and ingestion path exist but full AI processing is not yet built. Data can be stored but not yet analyzed end-to-end."},
            {"term": "Built", "definition": "A modality or stream that is fully operational — data flows in, processing runs, and outputs (features, predictions, reports) are available."},
            {"term": "Planned", "definition": "A modality or stream identified in the roadmap but not yet implemented. No data collection, schema, or processing exists yet."},
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
