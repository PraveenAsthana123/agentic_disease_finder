"""Neuro Advancements Dashboard — per-modality AI advancement opportunities,
AI model coverage, biomarker inventory, cross-modal fusion ideas
from config/neuro_advancements.json.
12 modalities, all built, 5 cross-modal advancements."""

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
    """Summary KPIs: modality counts, status distribution, AI model coverage, biomarker inventory."""
    cfg = _load('neuro_advancements.json')
    if not cfg:
        return {"available": False, "note": "neuro_advancements.json missing"}

    modalities = cfg.get('modalities', [])
    cross_modal = cfg.get('cross_modal_advancements', [])

    total = len(modalities)
    built = sum(1 for m in modalities if m.get('status') == 'built')
    planned = sum(1 for m in modalities if m.get('status') == 'planned')
    partial = sum(1 for m in modalities if m.get('status') == 'partial')

    # Unique AI models across all modalities
    all_models = []
    for m in modalities:
        all_models.extend(m.get('ai_models', []))
    unique_models = sorted(set(all_models))

    # Model frequency
    model_counts = Counter(all_models)
    model_distribution = [
        {"name": model, "value": count}
        for model, count in sorted(model_counts.items(), key=lambda x: -x[1])
    ]

    # Status distribution
    status_distribution = []
    if built:
        status_distribution.append({"name": "Built", "value": built})
    if partial:
        status_distribution.append({"name": "Partial", "value": partial})
    if planned:
        status_distribution.append({"name": "Planned", "value": planned})

    # Biomarkers
    biomarkers = [m.get('biomarker', '') for m in modalities if m.get('biomarker')]

    # Models per modality for bar chart
    models_per_modality = [
        {"name": m.get('code', ''), "value": len(m.get('ai_models', []))}
        for m in modalities
    ]

    # Summary table
    modalities_table = [
        {
            "code": m.get('code', ''),
            "name": m.get('name', ''),
            "advancement": m.get('advancement', ''),
            "ai_models": ', '.join(m.get('ai_models', [])),
            "biomarker": m.get('biomarker', ''),
            "status": m.get('status', 'planned'),
        }
        for m in modalities
    ]

    return {
        "available": True,
        "title": cfg.get('title', ''),
        "updated_at": cfg.get('updated_at', ''),
        "summary": {
            "total_modalities": total,
            "built": built,
            "partial": partial,
            "planned": planned,
            "unique_ai_models": len(unique_models),
            "total_biomarkers": len(biomarkers),
            "cross_modal_count": len(cross_modal),
        },
        "status_distribution": status_distribution,
        "model_distribution": model_distribution,
        "models_per_modality": models_per_modality,
        "modalities_table": modalities_table,
    }


def breakdown():
    """Per-modality details, AI model inventory, biomarker mapping, cross-modal advancements."""
    cfg = _load('neuro_advancements.json')
    if not cfg:
        return {"available": False, "note": "neuro_advancements.json missing"}

    modalities = cfg.get('modalities', [])
    cross_modal = cfg.get('cross_modal_advancements', [])

    per_modality = [
        {
            "code": m.get('code', ''),
            "name": m.get('name', ''),
            "advancement": m.get('advancement', ''),
            "ai_models": m.get('ai_models', []),
            "biomarker": m.get('biomarker', ''),
            "status": m.get('status', 'planned'),
            "note": m.get('note', ''),
        }
        for m in modalities
    ]

    # Group by model type category
    model_to_modalities = {}
    for m in modalities:
        for model in m.get('ai_models', []):
            if model not in model_to_modalities:
                model_to_modalities[model] = []
            model_to_modalities[model].append(m.get('code', ''))
    ai_model_index = [
        {"model": model, "used_by": codes, "count": len(codes)}
        for model, codes in sorted(model_to_modalities.items(), key=lambda x: -len(x[1]))
    ]

    # Biomarker index
    biomarker_index = [
        {"modality": m.get('code', ''), "name": m.get('name', ''), "biomarker": m.get('biomarker', '')}
        for m in modalities
        if m.get('biomarker')
    ]

    return {
        "available": True,
        "per_modality": per_modality,
        "ai_model_index": ai_model_index,
        "biomarker_index": biomarker_index,
        "cross_modal_advancements": cross_modal,
    }


def definitions():
    """Modality descriptions, status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "color": "#22c55e", "description": "Built — modality has a working AI pipeline with endpoint, dashboard, and real data processing."},
            {"status": "partial", "color": "#f97316", "description": "Partial — modality has some components built but pipeline is incomplete or unverified."},
            {"status": "planned", "color": "#94a3b8", "description": "Planned — modality is documented but no AI pipeline code has been written yet."},
        ],
        "modality_categories": [
            {"category": "Electrophysiology", "modalities": ["EEG", "EMG", "NCV", "SSEP", "VEP", "BERA", "RNST", "SSR", "BLINK"], "description": "Electrical signal recording from brain, nerves, or muscles — the core of neurophysiology AI."},
            {"category": "Autonomic / Cardiac", "modalities": ["RR_VAR", "ABPM_HOLTER"], "description": "Heart rate variability and blood pressure monitoring for autonomic dysfunction assessment."},
            {"category": "Video / Imaging", "modalities": ["VIDEO"], "description": "Patient video analysis for seizure semiology, fall detection, and behavioral patterns."},
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography — recording of electrical brain activity via scalp electrodes for seizure detection and brain function assessment."},
            {"term": "EMG", "definition": "Electromyography — recording of electrical activity in skeletal muscles for neuromuscular disorder diagnosis."},
            {"term": "NCV", "definition": "Nerve Conduction Velocity — measurement of electrical impulse speed through peripheral nerves for neuropathy grading."},
            {"term": "SSEP", "definition": "Somatosensory Evoked Potentials — measurement of sensory pathway integrity from peripheral nerve to cortex."},
            {"term": "VEP", "definition": "Visual Evoked Potential — measurement of visual pathway function via P100 latency for optic neuritis/MS screening."},
            {"term": "BERA", "definition": "Brainstem Evoked Response Audiometry — auditory brainstem response for hearing and brainstem function assessment."},
            {"term": "RNST", "definition": "Repetitive Nerve Stimulation Test — assessment of neuromuscular junction function for myasthenia gravis screening."},
            {"term": "HRV", "definition": "Heart Rate Variability — time/frequency domain analysis of RR intervals for autonomic nervous system assessment."},
            {"term": "ABPM", "definition": "Ambulatory Blood Pressure Monitoring — 24-hour blood pressure recording for dipping status and hypertension assessment."},
            {"term": "SSR", "definition": "Sympathetic Skin Response — measurement of sudomotor function for autonomic neuropathy screening."},
            {"term": "HFO", "definition": "High-Frequency Oscillation — biomarker for epileptogenic zones requiring high sampling rate EEG."},
            {"term": "MUAP", "definition": "Motor Unit Action Potential — the electrical signal from a single motor unit, analyzed for neuromuscular disease classification."},
            {"term": "3D-CNN", "definition": "3D Convolutional Neural Network — deep learning architecture for spatiotemporal video analysis of seizure movements."},
            {"term": "EEGNet", "definition": "Compact CNN architecture specifically designed for EEG-based brain-computer interfaces and seizure detection."},
            {"term": "Federated Learning", "definition": "Distributed machine learning that trains across multiple sites without sharing raw patient data."},
        ],
        "clinical_notes": [
            "All 12 modalities are built — this platform covers the full spectrum of clinical neurophysiology testing.",
            "Cross-modal fusion (EEG + video + autonomic) is the novel advancement frontier, combining electrographic and behavioral seizure markers.",
            "Foundation model pretraining on unlabeled neurophysiology data can dramatically reduce per-test labeled data requirements.",
            "Federated learning enables multi-site model training without HIPAA-violating data transfers between institutions.",
        ],
        "references": [
            "Schirrmeister RT et al. Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping, 2017.",
            "Lawhern VJ et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of Neural Engineering, 2018.",
            "Devinsky O et al. Epilepsy. Nature Reviews Disease Primers, 2018.",
            "ACNS Guidelines for Standard Electrode Position Nomenclature. American Clinical Neurophysiology Society.",
            "ILAE Classification of Epilepsies. International League Against Epilepsy, 2017.",
        ],
    }
