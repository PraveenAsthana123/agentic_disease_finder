"""EEG Data Formats Dashboard — AI-readiness, routing, star ratings,
and data-request guidance from config/eeg_data_formats.json.
12 formats, 5 routes, 3 readiness levels, data-request rules."""

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
    """Summary KPIs: format counts, AI readiness, routing, star distribution."""
    cfg = _load('eeg_data_formats.json')
    if not cfg:
        return {"available": False, "note": "eeg_data_formats.json missing"}

    formats = cfg.get('formats', [])
    routing = cfg.get('routing', {})

    ai_ready_count = sum(1 for f in formats if f.get('ai_ready') is True)
    partially_ready = sum(1 for f in formats if f.get('ai_ready') == 'partial')
    not_ready = sum(1 for f in formats if f.get('ai_ready') is False)
    supported_count = sum(1 for f in formats if f.get('supported') is True)
    unsupported_count = sum(1 for f in formats if not f.get('supported'))

    route_counts = Counter(f.get('route', 'unknown') for f in formats)
    route_distribution = [
        {"name": r, "value": c}
        for r, c in sorted(route_counts.items(), key=lambda x: -x[1])
    ]

    star_counts = Counter(f.get('stars', 0) for f in formats)
    star_distribution = [
        {"name": f"{s} star{'s' if s != 1 else ''}", "value": c}
        for s, c in sorted(star_counts.items(), key=lambda x: -x[0])
    ]

    total = len(formats)
    avg_stars = round(sum(f.get('stars', 0) for f in formats) / total, 1) if total else 0

    ai_readiness_distribution = [
        {"name": "AI Ready", "value": ai_ready_count},
        {"name": "Partial", "value": partially_ready},
        {"name": "Not Ready", "value": not_ready},
    ]

    formats_table = [
        {
            "ext": f.get('ext', ''),
            "name": f.get('name', ''),
            "ai_ready": f.get('ai_ready', False),
            "stars": f.get('stars', 0),
            "route": f.get('route', ''),
            "supported": f.get('supported', False),
        }
        for f in formats
    ]

    return {
        "available": True,
        "title": cfg.get('title', ''),
        "note": cfg.get('note', ''),
        "updated_at": cfg.get('updated_at', ''),
        "summary": {
            "total_formats": total,
            "ai_ready_count": ai_ready_count,
            "partially_ready": partially_ready,
            "not_ready": not_ready,
            "supported_count": supported_count,
            "unsupported_count": unsupported_count,
            "route_types": len(route_counts),
            "avg_stars": avg_stars,
        },
        "ai_readiness_distribution": ai_readiness_distribution,
        "route_distribution": route_distribution,
        "star_distribution": star_distribution,
        "formats_table": formats_table,
    }


def breakdown():
    """Per-format details, routing groups, data request guidance."""
    cfg = _load('eeg_data_formats.json')
    if not cfg:
        return {"available": False, "note": "eeg_data_formats.json missing"}

    formats = cfg.get('formats', [])
    routing = cfg.get('routing', {})
    data_request = cfg.get('data_request', {})

    per_format = [
        {
            "ext": f.get('ext', ''),
            "name": f.get('name', ''),
            "ai_ready": f.get('ai_ready', False),
            "stars": f.get('stars', 0),
            "route": f.get('route', ''),
            "contains": f.get('contains', ''),
            "supported": f.get('supported', False),
            "good_for": f.get('good_for', ''),
            "bad_for": f.get('bad_for', ''),
        }
        for f in formats
    ]

    # Group formats by route
    per_route = {}
    for f in formats:
        route = f.get('route', 'unknown')
        if route not in per_route:
            per_route[route] = {
                "route": route,
                "description": routing.get(route, ''),
                "formats": [],
            }
        per_route[route]["formats"].append({
            "ext": f.get('ext', ''),
            "name": f.get('name', ''),
            "ai_ready": f.get('ai_ready', False),
            "stars": f.get('stars', 0),
        })
    per_route_list = list(per_route.values())

    return {
        "available": True,
        "per_format": per_format,
        "per_route": per_route_list,
        "data_request": {
            "must_have": data_request.get('must_have', []),
            "good_to_have": data_request.get('good_to_have', []),
            "minimum_dbA": data_request.get('minimum_dbA', ''),
            "rule": data_request.get('rule', ''),
        },
    }


def definitions():
    """Route descriptions, readiness legend, star ratings, glossary, references."""
    return {
        "available": True,
        "route_descriptions": [
            {"route": "signal", "description": "Raw EEG signal pipeline — band power extraction, biomarker computation, seizure classification, deep learning models."},
            {"route": "rag", "description": "Retrieval-Augmented Generation — text extraction from PDFs/reports, vectorization, semantic search for clinical ground-truth labels."},
            {"route": "cv", "description": "Computer Vision pipeline — EEG screenshots processed via OCR, image classification, clinician validation overlays."},
            {"route": "extract", "description": "Metadata extraction — structured data parsed from XML/other formats for enrichment and cross-referencing."},
            {"route": "video", "description": "Video pipeline — seizure video analysis, behavioral semiology, video-EEG concordance, caregiver-recorded seizure capture."},
        ],
        "ai_readiness_legend": [
            {"value": "true", "color": "#22c55e", "description": "AI Ready — format contains raw signal data suitable for direct model training and inference."},
            {"value": "partial", "color": "#f97316", "description": "Partially Ready — contains structured data but may lack signal fidelity or require preprocessing."},
            {"value": "false", "color": "#ef4444", "description": "Not AI Ready — does not contain raw signal; useful for RAG, CV, or validation only."},
        ],
        "star_rating_legend": [
            {"stars": 5, "description": "Gold standard — native EEG format with full metadata, channels, and sampling frequency (EDF, BDF)."},
            {"stars": 4, "description": "High quality — rich metadata and signal data, minor format conversion may be needed (FIF, MAT)."},
            {"stars": 3, "description": "Usable — numeric data present but metadata (sfreq, channel names) may be assumed or missing (CSV, NPZ)."},
            {"stars": 2, "description": "Limited — structured metadata or video content, not suitable for signal AI (XML, MP4, video variants)."},
            {"stars": 1, "description": "Minimal — no signal data; useful only for RAG ground-truth or CV validation (PDF, PNG, JPG)."},
        ],
        "glossary": [
            {"term": "EDF", "definition": "European Data Format — the most widely used standard for storing EEG recordings with header metadata."},
            {"term": "BDF", "definition": "BioSemi Data Format — 24-bit extension of EDF used by BioSemi amplifiers for higher resolution."},
            {"term": "FIF", "definition": "Functional Imaging File — MNE/Elekta native format storing processed EEG/MEG data with full metadata."},
            {"term": "EEG", "definition": "Electroencephalography — measurement of electrical brain activity via scalp electrodes."},
            {"term": "sfreq", "definition": "Sampling frequency — the number of data points recorded per second (Hz), critical for signal fidelity."},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — AI technique combining document retrieval with language model generation."},
            {"term": "CV", "definition": "Computer Vision — AI field for image/video analysis, used for EEG screenshot interpretation."},
            {"term": "OCR", "definition": "Optical Character Recognition — extracting text from images, used on EEG screenshot annotations."},
            {"term": "HFO", "definition": "High-Frequency Oscillation — biomarker for epileptogenic zones, requires high sfreq signal data."},
            {"term": "ICA", "definition": "Independent Component Analysis — artifact removal technique applied to raw EEG signals."},
            {"term": "MNE", "definition": "MNE-Python — open-source Python library for EEG/MEG data analysis and visualization."},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — explainable AI method for feature importance in EEG models."},
        ],
        "clinical_notes": [
            "Always request EDF/BDF from the EEG machine vendor — PDF reports alone cannot power signal-based AI models.",
            "Video-EEG concordance requires both signal files (EDF/BDF) and seizure video (MP4) to correlate behavioral and electrographic patterns.",
            "EEG screenshots (PNG/JPG) lose sampling frequency and resolution — never substitute for raw signal files in AI training.",
            "A minimum dataset of 200-300 patients with EDF + PDF + diagnosis labels enables Explainable AI, Responsible-AI governance, and HITL validation.",
        ],
        "references": [
            "Kemp B, Olivan J. European Data Format 'plus' (EDF+), an EDF alike standard format for the exchange of physiological data. Clinical Neurophysiology, 2003.",
            "BioSemi Data Format (BDF) Specification — 24-bit extension of EDF for high-resolution biosignal recording.",
            "Gramfort A et al. MNE software for processing MEG and EEG data. NeuroImage, 2014.",
            "Pernet CR et al. EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. Scientific Data, 2019.",
            "IEEE 11073 — Health informatics standards for point-of-care medical device communication.",
        ],
    }
