"""Time-Frequency Representations (TFR) Dashboard.

Serves overview, breakdown, and definitions for STFT / Wavelet / Spectrogram
methods used in EEG seizure analysis. Backed by config/time_frequency.json.
"""

import json
import os

_CFG = os.path.join(os.path.dirname(__file__), '..', 'config', 'time_frequency.json')


def _load():
    if not os.path.exists(_CFG):
        return None
    with open(_CFG) as f:
        return json.load(f)


def overview():
    """KPIs + pipeline + band coverage summary."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "time_frequency.json missing"}

    methods = cfg.get('methods', [])
    bands = cfg.get('frequency_bands', [])
    stats = cfg.get('coverage_stats', {})
    pipeline = cfg.get('pipeline_integration', {}).get('stages', [])

    family_counts = {}
    for m in methods:
        fam = m.get('family', 'other')
        family_counts[fam] = family_counts.get(fam, 0) + 1

    return {
        "available": True,
        "generated_at": "2026-08-06",
        "kpis": [
            {"label": "TFR Methods", "value": len(methods), "color": "primary", "sub": "STFT · CWT · DWT · Spec · Mel · WVD"},
            {"label": "Freq Bands", "value": len(bands), "color": "info", "sub": "Delta → HFO (0.5–500 Hz)"},
            {"label": "Pipeline Stages", "value": len(pipeline), "color": "secondary", "sub": "raw EEG → model input"},
            {"label": "Datasets Tested", "value": len(stats.get('datasets_tested', [])), "color": "success", "sub": "Bonn · CHB-MIT · TUH"},
        ],
        "summary": {
            "methods_total": len(methods),
            "fourier_methods": family_counts.get('fourier', 0),
            "wavelet_methods": family_counts.get('wavelet', 0),
            "quadratic_methods": family_counts.get('quadratic', 0),
            "pipeline_stages": len(pipeline),
            "backend_endpoints": stats.get('backend_endpoints', []),
            "datasets_tested": stats.get('datasets_tested', []),
        },
        "family_distribution": [
            {"name": k.title(), "value": v}
            for k, v in sorted(family_counts.items(), key=lambda x: -x[1])
        ],
        "pipeline_stages": pipeline,
        "frequency_bands": bands,
        "methods_table": [
            {
                "name": m['name'],
                "family": m.get('family', '').title(),
                "resolution": m.get('resolution', ''),
                "output": m.get('output', ''),
                "pipeline_stage": m.get('pipeline_stage', ''),
            }
            for m in methods
        ],
    }


def breakdown():
    """Per-method detailed breakdown: params, bands, use-cases, notes."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "time_frequency.json missing"}

    methods = cfg.get('methods', [])
    bands = cfg.get('frequency_bands', [])

    method_cards = []
    for m in methods:
        method_cards.append({
            "id": m['id'],
            "name": m['name'],
            "family": m.get('family', '').title(),
            "resolution": m.get('resolution', ''),
            "time_resolution": m.get('time_resolution', ''),
            "freq_resolution": m.get('freq_resolution', ''),
            "output": m.get('output', ''),
            "used_for": m.get('used_for', []),
            "params": m.get('params', {}),
            "epilepsy_bands": m.get('epilepsy_bands', []),
            "pipeline_stage": m.get('pipeline_stage', ''),
            "notes": m.get('notes', ''),
        })

    resolution_matrix = [
        {
            "method": m['name'].split('(')[0].strip(),
            "time_res": m.get('time_resolution', 'moderate'),
            "freq_res": m.get('freq_resolution', 'moderate'),
            "complexity": "O(n log n)" if m.get('family') == 'fourier' else ("O(n)" if m.get('id') == 'dwt' else "O(n²)"),
        }
        for m in methods
    ]

    use_case_map = {}
    for m in methods:
        for uc in m.get('used_for', []):
            use_case_map.setdefault(uc, []).append(m['name'].split('(')[0].strip())

    return {
        "available": True,
        "method_cards": method_cards,
        "resolution_matrix": resolution_matrix,
        "use_case_map": [{"use_case": k, "methods": v} for k, v in sorted(use_case_map.items())],
        "band_coverage": bands,
    }


def definitions():
    """Glossary of TFR terms + methodology notes."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "time_frequency.json missing"}

    defs = cfg.get('definitions', [])
    methods = cfg.get('methods', [])

    method_glossary = [
        {
            "term": m['name'].split('(')[0].strip(),
            "abbrev": m['id'].upper(),
            "family": m.get('family', '').title(),
            "one_line": m.get('output', '') + " — " + m.get('notes', '')[:80],
        }
        for m in methods
    ]

    return {
        "available": True,
        "definitions": defs,
        "method_glossary": method_glossary,
        "references": [
            {"title": "Mallat (2009): A Wavelet Tour of Signal Processing", "type": "textbook"},
            {"title": "Gabor (1946): Theory of Communication — STFT origin", "type": "paper"},
            {"title": "Morlet wavelet in EEG: Tallon-Baudry & Bertrand (1999)", "type": "paper"},
            {"title": "HFO review: Jacobs et al. (2012) — Epilepsia", "type": "paper"},
            {"title": "PyWavelets documentation — pywt.wavedec()", "type": "library"},
            {"title": "librosa.feature.melspectrogram()", "type": "library"},
            {"title": "SciPy signal.spectrogram()", "type": "library"},
        ],
        "uncertainty_principle": {
            "statement": "Δt · Δf ≥ 1/(4π) — cannot achieve arbitrary precision in both time and frequency simultaneously.",
            "implication": "STFT: fixed window trades time vs freq. CWT: adaptive — short windows at high freq, long at low freq.",
        },
    }
