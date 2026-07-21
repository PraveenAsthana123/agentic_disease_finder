"""EEG AI Stack Dashboard â 16-layer tool stack visualization
from config/eeg_ai_stack.json.
16 layers, 38 tools, 4 status levels (installed/built/external/cataloged),
EDC assessment tools, recommended pipeline."""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, "..", "config")


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: layer count, tool counts, status distribution, tools-per-layer bar."""
    cfg = _load("eeg_ai_stack.json")
    if not cfg:
        return {"available": False, "note": "eeg_ai_stack.json missing"}

    layers = cfg.get("layers", [])
    edc_tools = cfg.get("edc_assessment_tools", [])
    summary = cfg.get("summary", {})

    all_tools = []
    for layer in layers:
        for t in layer.get("tools", []):
            t["_layer"] = layer.get("layer", "")
            all_tools.append(t)

    for t in edc_tools:
        t["_layer"] = "EDC Assessment"
        all_tools.append(t)

    status_counts = Counter(t.get("status", "unknown") for t in all_tools)
    status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    layer_counts = [
        {"name": layer.get("layer", ""), "value": len(layer.get("tools", []))}
        for layer in layers
    ]

    tools_with_endpoints = []
    for t in all_tools:
        eps = t.get("endpoints", [])
        if not eps:
            ep = t.get("endpoint", "")
            if ep:
                eps = [ep]
        dash = t.get("dashboard", "")
        if dash:
            eps = eps or [dash]
        if eps:
            tools_with_endpoints.append({
                "name": t.get("name", ""),
                "layer": t.get("_layer", ""),
                "endpoints": eps if isinstance(eps, list) else [eps],
                "status": t.get("status", ""),
            })

    tools_table = [
        {
            "name": t.get("name", ""),
            "layer": t.get("_layer", ""),
            "use": t.get("use", ""),
            "status": t.get("status", ""),
        }
        for t in all_tools
    ]

    return {
        "available": True,
        "title": cfg.get("title", ""),
        "note": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "summary": {
            "total_layers": len(layers),
            "total_tools": len(all_tools),
            "installed": status_counts.get("installed", 0),
            "built": status_counts.get("built", 0),
            "external": status_counts.get("external", 0),
            "cataloged": status_counts.get("cataloged", 0),
            "tools_with_endpoints": len(tools_with_endpoints),
            "edc_tools": len(edc_tools),
        },
        "status_distribution": status_distribution,
        "layer_distribution": layer_counts,
        "tools_table": tools_table,
        "recommended_pipeline": cfg.get("recommended_pipeline", ""),
    }


def breakdown():
    """Per-layer tool details, tools with endpoints, EDC assessment tools."""
    cfg = _load("eeg_ai_stack.json")
    if not cfg:
        return {"available": False, "note": "eeg_ai_stack.json missing"}

    layers = cfg.get("layers", [])
    edc_tools = cfg.get("edc_assessment_tools", [])

    per_layer = []
    for layer in layers:
        tools = layer.get("tools", [])
        statuses = Counter(t.get("status", "unknown") for t in tools)
        per_layer.append({
            "layer": layer.get("layer", ""),
            "tool_count": len(tools),
            "status_counts": dict(statuses),
            "tools": [
                {
                    "name": t.get("name", ""),
                    "use": t.get("use", ""),
                    "status": t.get("status", ""),
                    "endpoints": t.get("endpoints", []),
                    "endpoint": t.get("endpoint", ""),
                    "dashboard": t.get("dashboard", ""),
                    "note": t.get("note", ""),
                }
                for t in tools
            ],
        })

    tools_with_endpoints = []
    for layer in layers:
        for t in layer.get("tools", []):
            eps = t.get("endpoints", [])
            if not eps:
                ep = t.get("endpoint", "")
                if ep:
                    eps = [ep]
            if eps:
                tools_with_endpoints.append({
                    "name": t.get("name", ""),
                    "layer": layer.get("layer", ""),
                    "endpoints": eps,
                    "status": t.get("status", ""),
                })

    edc = [
        {
            "name": t.get("name", ""),
            "use": t.get("use", ""),
            "status": t.get("status", ""),
            "endpoints": t.get("endpoints", []),
        }
        for t in edc_tools
    ]

    return {
        "available": True,
        "per_layer": per_layer,
        "tools_with_endpoints": tools_with_endpoints,
        "edc_assessment_tools": edc,
    }


def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "installed", "description": "Python package installed and import verified in this environment"},
            {"status": "built", "description": "Live in platform â API endpoints verified, dashboard functional"},
            {"status": "external", "description": "MATLAB/desktop/server tool â recommended, runs on separate infrastructure"},
            {"status": "cataloged", "description": "Recommended tool â not yet installed, cataloged for future integration"},
        ],
        "glossary": [
            {"term": "MNE-Python", "definition": "Open-source Python library for EEG/MEG analysis â the central platform"},
            {"term": "EDF/BDF", "definition": "European Data Format / BioSemi Data Format â standard EEG file formats"},
            {"term": "ICA", "definition": "Independent Component Analysis â artifact separation technique"},
            {"term": "CSP", "definition": "Common Spatial Patterns â spatial filter for BCI/classification"},
            {"term": "PSD", "definition": "Power Spectral Density â frequency-domain signal representation"},
            {"term": "CWT/DWT", "definition": "Continuous/Discrete Wavelet Transform â time-frequency decomposition"},
            {"term": "STFT", "definition": "Short-Time Fourier Transform â sliding-window spectral analysis"},
            {"term": "EEGNet", "definition": "Compact CNN architecture for EEG-based BCI (Lawhern et al. 2018)"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations â model-agnostic feature importance"},
            {"term": "Grad-CAM", "definition": "Gradient-weighted Class Activation Mapping â CNN visualization"},
            {"term": "LIME", "definition": "Local Interpretable Model-agnostic Explanations â local surrogate models"},
            {"term": "AutoReject", "definition": "Automated bad epoch/channel rejection for EEG preprocessing"},
            {"term": "PyPREP", "definition": "Python PREP pipeline â robust EEG referencing"},
            {"term": "ICLabel", "definition": "Neural-net ICA component classifier (brain/eye/muscle/heart/noise)"},
            {"term": "PLV", "definition": "Phase Locking Value â functional connectivity metric"},
        ],
        "clinical_notes": [
            "The 16-layer stack covers the complete EEG AI pipeline from raw signal to responsible deployment",
            "Installed tools are verified importable in the project Python environment",
            "Built tools have live API endpoints and functional dashboard panels in the platform",
            "The recommended pipeline represents the minimum viable EEG AI processing chain",
        ],
        "references": [
            "Gramfort et al., 2013 â MEG and EEG data analysis with MNE-Python (Frontiers in Neuroscience)",
            "Lawhern et al., 2018 â EEGNet: A compact CNN for EEG-based BCI (Journal of Neural Engineering)",
            "Lundberg & Lee, 2017 â A Unified Approach to Interpreting Model Predictions (NIPS)",
            "Jas et al., 2017 â Autoreject: Automated artifact rejection for MEG and EEG (NeuroImage)",
            "Bigdely-Shamlo et al., 2015 â PREP pipeline: standardized preprocessing for large-scale EEG (Frontiers)",
        ],
    }
