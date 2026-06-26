#!/usr/bin/env python3
"""
Entropy & Complexity Dashboard — AntroPy + SciPy
=================================================

Computes real entropy/complexity features from EEG recordings on disk
using the AntroPy library (Vallat, 2023).  Every value comes from actual
EDF data — nothing fabricated.

Metrics computed per channel:
  - Sample Entropy (SampEn)   — regularity/predictability
  - Permutation Entropy (PE)  — ordinal-pattern complexity
  - Spectral Entropy (SpEn)   — frequency-domain uncertainty
  - Approximate Entropy (ApEn)— similar to SampEn, faster
  - Higuchi Fractal Dimension (HFD) — signal complexity
  - Detrended Fluctuation Analysis (DFA) — long-range correlation

Clinical relevance:
  Seizure epochs show LOWER entropy (more regular/rhythmic) compared
  to interictal background.  Entropy features are top-ranked in many
  EEG seizure-detection pipelines (Acharya et al., 2018).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).parent.parent

# ── data discovery ──────────────────────────────────────────────────

def _find_edf() -> Optional[Path]:
    """Return the first available real EDF file."""
    dirs = [
        ROOT / "data/real_eeg/epilepsy_physionet",
        ROOT / "data/real_eeg/depression_figshare",
        ROOT / "data/real_eeg",
    ]
    for d in dirs:
        if d.is_dir():
            edfs = sorted(d.glob("*.edf"))
            if edfs:
                return edfs[0]
    return None


def _load_raw(file: Optional[str] = None, seconds: float = 30.0):
    """Load raw EDF via MNE, return (data_array, sfreq, ch_names)."""
    import mne
    mne.set_log_level("ERROR")

    if file:
        p = ROOT / file if not Path(file).is_absolute() else Path(file)
    else:
        p = _find_edf()
    if not p or not p.exists():
        return None, None, None

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)
    sfreq = raw.info["sfreq"]
    n_samples = min(int(seconds * sfreq), raw.n_times)
    data = raw.get_data(start=0, stop=n_samples)  # (n_ch, n_times)
    ch_names = raw.ch_names
    return data, sfreq, ch_names


# ── entropy computation ─────────────────────────────────────────────

def _compute_channel_entropy(signal: np.ndarray) -> Dict[str, Any]:
    """Compute all entropy/complexity metrics for a single channel."""
    import antropy as ant

    # Downsample if very long (keep ≤4096 points for speed)
    if len(signal) > 4096:
        step = len(signal) // 4096
        signal = signal[::step]

    result = {}

    try:
        result["sample_entropy"] = round(float(ant.sample_entropy(signal)), 4)
    except Exception:
        result["sample_entropy"] = None

    try:
        result["permutation_entropy"] = round(
            float(ant.perm_entropy(signal, normalize=True)), 4
        )
    except Exception:
        result["permutation_entropy"] = None

    try:
        result["spectral_entropy"] = round(
            float(ant.spectral_entropy(signal, sf=256, normalize=True)), 4
        )
    except Exception:
        result["spectral_entropy"] = None

    try:
        result["approximate_entropy"] = round(
            float(ant.app_entropy(signal)), 4
        )
    except Exception:
        result["approximate_entropy"] = None

    try:
        result["higuchi_fd"] = round(float(ant.higuchi_fd(signal)), 4)
    except Exception:
        result["higuchi_fd"] = None

    try:
        result["dfa"] = round(float(ant.detrended_fluctuation(signal)), 4)
    except Exception:
        result["dfa"] = None

    return result


# ── public API ──────────────────────────────────────────────────────

def overview(file: str = None, seconds: float = 30.0) -> Dict[str, Any]:
    """Full entropy dashboard: per-channel metrics + summary statistics."""
    data, sfreq, ch_names = _load_raw(file, seconds)
    if data is None:
        return {
            "available": False,
            "note": "No EDF files found. Place .edf files under data/real_eeg/",
        }

    n_ch = min(data.shape[0], 19)  # cap at 19 channels (10-20 system)
    per_channel = []
    metric_keys = [
        "sample_entropy", "permutation_entropy", "spectral_entropy",
        "approximate_entropy", "higuchi_fd", "dfa",
    ]

    for i in range(n_ch):
        ent = _compute_channel_entropy(data[i])
        ent["channel"] = ch_names[i]
        per_channel.append(ent)

    # Summary statistics across channels
    summary = {}
    for key in metric_keys:
        vals = [ch[key] for ch in per_channel if ch[key] is not None]
        if vals:
            summary[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "median": round(float(np.median(vals)), 4),
            }
        else:
            summary[key] = None

    return {
        "available": True,
        "file": str(Path(file).name) if file else str(_find_edf().name),
        "sfreq": sfreq,
        "seconds_analyzed": seconds,
        "n_channels": n_ch,
        "per_channel": per_channel,
        "summary": summary,
        "source": "Real EDF via MNE-Python + AntroPy (Vallat 2023). No synthetic data.",
    }


def heatmap(file: str = None, seconds: float = 30.0) -> Dict[str, Any]:
    """Entropy heatmap data: channels × metrics matrix for visualization."""
    data, sfreq, ch_names = _load_raw(file, seconds)
    if data is None:
        return {"available": False, "note": "No EDF files found."}

    n_ch = min(data.shape[0], 19)
    metric_keys = [
        "sample_entropy", "permutation_entropy", "spectral_entropy",
        "approximate_entropy", "higuchi_fd", "dfa",
    ]
    matrix = []
    channels = []

    for i in range(n_ch):
        ent = _compute_channel_entropy(data[i])
        channels.append(ch_names[i])
        row = [ent.get(k) for k in metric_keys]
        matrix.append(row)

    return {
        "available": True,
        "channels": channels,
        "metrics": metric_keys,
        "matrix": matrix,
        "interpretation": {
            "lower_entropy": "More regular/rhythmic — may indicate seizure activity",
            "higher_entropy": "More complex/irregular — typical of normal background EEG",
            "hfd_range": "Typically 1.0–2.0 for EEG; lower = smoother signal",
            "dfa_range": "~0.5 = white noise, ~1.0 = 1/f noise (healthy), >1.0 = long-range correlations",
        },
        "source": "Real EDF via AntroPy. Channels × metrics matrix.",
    }


def definitions() -> Dict[str, Any]:
    """Entropy metric definitions, clinical interpretation, and references."""
    return {
        "metrics": [
            {
                "name": "Sample Entropy (SampEn)",
                "description": "Measures the regularity and unpredictability of a time series. "
                "Lower values indicate more self-similar, regular patterns.",
                "range": "0 to ∞ (typical EEG: 0.1–2.5)",
                "clinical": "Seizure epochs typically show SampEn 0.2–0.8 vs interictal 1.0–2.0. "
                "One of the most discriminative features for seizure detection.",
                "reference": "Richman & Moorman (2000). Am J Physiol Heart Circ Physiol.",
            },
            {
                "name": "Permutation Entropy (PE)",
                "description": "Ordinal-pattern complexity measure. Normalized 0–1. "
                "Robust to noise, computationally efficient.",
                "range": "0 (deterministic) to 1 (random)",
                "clinical": "PE drops significantly during seizures (~0.3–0.6) vs background (~0.8–0.95). "
                "Effective even at short window lengths (3–5 seconds).",
                "reference": "Bandt & Pompe (2002). Phys Rev Lett.",
            },
            {
                "name": "Spectral Entropy (SpEn)",
                "description": "Shannon entropy of the normalized power spectral density. "
                "Measures how uniformly power is distributed across frequencies.",
                "range": "0 (single frequency) to 1 (uniform/white noise)",
                "clinical": "Rhythmic seizure activity concentrates power → lower SpEn. "
                "Useful for distinguishing focal vs generalized patterns.",
                "reference": "Inouye et al. (1991). Electroencephalogr Clin Neurophysiol.",
            },
            {
                "name": "Approximate Entropy (ApEn)",
                "description": "Similar to SampEn but includes self-matches. "
                "Faster to compute, slightly biased for short data.",
                "range": "0 to ∞ (typical EEG: 0.1–2.0)",
                "clinical": "Predates SampEn in seizure detection literature. "
                "Sensitive to epileptiform discharges and burst-suppression.",
                "reference": "Pincus (1991). Proc Natl Acad Sci USA.",
            },
            {
                "name": "Higuchi Fractal Dimension (HFD)",
                "description": "Fractal (self-similarity) dimension of the time series. "
                "Higher values = more complex waveform.",
                "range": "1.0 (smooth) to 2.0 (space-filling)",
                "clinical": "HFD ~1.3–1.6 for healthy EEG; drops to ~1.1–1.3 during seizures. "
                "Sensitive to transient EEG changes.",
                "reference": "Higuchi (1988). Physica D.",
            },
            {
                "name": "Detrended Fluctuation Analysis (DFA)",
                "description": "Quantifies long-range temporal correlations. "
                "The scaling exponent α characterizes the signal's memory.",
                "range": "~0.5 (white noise) to ~1.5 (strong long-range correlation)",
                "clinical": "α ~ 0.7–1.0 for healthy EEG. Seizures may alter scaling, "
                "suggesting disrupted cortical dynamics.",
                "reference": "Peng et al. (1994). Phys Rev E.",
            },
        ],
        "clinical_summary": (
            "Entropy and complexity features are among the top-ranked biomarkers "
            "for automated seizure detection (Acharya et al., 2018). Seizure "
            "epochs show LOWER entropy (more regular, rhythmic oscillations) "
            "compared to interictal background. Combining multiple entropy "
            "metrics with ML classifiers achieves >95% accuracy in several "
            "benchmark datasets."
        ),
        "library": {
            "name": "AntroPy",
            "version": "0.2.2",
            "author": "Raphael Vallat",
            "url": "https://github.com/raphaelvallat/antropy",
            "license": "BSD-3-Clause",
        },
    }


if __name__ == "__main__":
    r = overview(seconds=10)
    print(json.dumps({k: v for k, v in r.items() if k != "per_channel"}, indent=2))
    if r.get("per_channel"):
        print(f"\nChannels computed: {len(r['per_channel'])}")
        print(json.dumps(r["per_channel"][0], indent=2))
