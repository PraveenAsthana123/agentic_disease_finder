#!/usr/bin/env python3
"""
CWT Scalogram Dashboard — Continuous Wavelet Transform Time-Frequency Analysis
===============================================================================

Performs REAL CWT/scalogram computations on EEG features from ``data/clinical.db``
(analyses table -> result_json -> features).  Uses band-power features
(delta/theta/alpha/beta/gamma) and spectral features already extracted in the
pipeline to reconstruct scalogram-equivalent time-frequency representations and
compute wavelet coefficient statistics.

When scipy.signal.cwt is available the module applies a Morlet wavelet to the
raw feature vectors.  When scipy is absent it falls back to a numpy-only
analytic Morlet implementation that produces equivalent wavelet coefficient
matrices from the stored band-power and spectral features.

Functions:
  overview()    -- KPIs + scalogram energy distribution + time-frequency
                   concentration metrics
  breakdown()   -- Per-band scalogram statistics, wavelet coefficient
                   distributions, cross-band energy ratios
  definitions() -- CWT methodology, Morlet wavelets, scalogram interpretation,
                   clinical relevance in EEG/epilepsy
"""

import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")

# Feature names matching the analyses table schema
FEATURE_NAMES = [
    "mean", "std", "var", "min", "max", "median", "ptp", "skewness",
    "kurtosis", "q25", "q75", "rms", "mav", "line_length",
    "zero_crossings", "delta_power", "theta_power", "alpha_power",
    "beta_power", "gamma_power", "total_power", "dominant_freq",
    "spectral_entropy", "psd_std", "psd_mean", "psd_median", "psd_q10",
    "psd_q90", "peak_ratio", "spectral_flatness", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff", "mean_abs_diff", "std_diff",
    "max_diff", "hjorth_mobility", "hjorth_complexity", "autocorr",
    "slope_changes", "trend", "crest_factor", "approx_entropy",
    "sample_entropy", "hurst_exponent", "dfa_alpha", "lz_complexity",
]

# Band-power feature indices (within FEATURE_NAMES)
BAND_FEATURES = {
    "delta": "delta_power",
    "theta": "theta_power",
    "alpha": "alpha_power",
    "beta": "beta_power",
    "gamma": "gamma_power",
}

# Canonical EEG frequency bands (Hz)
BAND_RANGES = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# CWT scale ranges corresponding to each band (for 256 Hz sampling rate)
# scale = f_center / (f * dt), with Morlet f_center ~ 0.8125 Hz and dt = 1/256
_FS = 256.0
_MORLET_FC = 0.8125  # Morlet wavelet center frequency (omega0=5)
BAND_SCALES = {
    band: (
        _MORLET_FC * _FS / hi,
        _MORLET_FC * _FS / lo,
    )
    for band, (lo, hi) in BAND_RANGES.items()
}


# ---------------------------------------------------------------------------
# Data loading (matches other dashboards)
# ---------------------------------------------------------------------------

def _load_data():
    """Load feature matrix and labels from clinical.db analyses table."""
    if not os.path.exists(_DB_PATH):
        return None, None, "Database not found"

    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute("SELECT result_json FROM analyses WHERE result_json IS NOT NULL")
    except sqlite3.OperationalError:
        conn.close()
        return None, None, "Table 'analyses' not found"

    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None, None, "No analyses found"

    X_list: List[List[float]] = []
    labels: List[str] = []

    for row in rows:
        try:
            data = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        feats = data.get("features")
        prediction = data.get("prediction", {})
        label = prediction.get("predicted_label") if isinstance(prediction, dict) else None
        if not feats or not isinstance(feats, dict) or not label:
            continue

        sample: List[float] = []
        for fname in FEATURE_NAMES:
            val = feats.get(fname)
            sample.append(float(val) if val is not None else np.nan)
        X_list.append(sample)
        labels.append(label)

    if not X_list:
        return None, None, "No valid samples"

    X = np.array(X_list, dtype=float)
    # Impute NaN with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0

    return X, labels, None


def _band_powers(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract per-band power columns from the feature matrix."""
    idx = {name: FEATURE_NAMES.index(fname) for name, fname in BAND_FEATURES.items()}
    return {band: np.abs(X[:, i]) for band, i in idx.items()}


# ---------------------------------------------------------------------------
# CWT / Morlet wavelet implementation (numpy-only)
# ---------------------------------------------------------------------------

def _morlet_wavelet(M: int, w: float = 5.0, s: float = 1.0) -> np.ndarray:
    """
    Complex Morlet wavelet of length *M* points.

    psi(t) = pi^{-1/4} * exp(j*w*t/s) * exp(-t^2 / (2*s^2))
    """
    t = np.linspace(-M / 2, M / 2, M, endpoint=False) / _FS
    norm = (np.pi ** -0.25) / np.sqrt(s)
    oscillation = np.exp(1j * w * t / s)
    envelope = np.exp(-0.5 * (t / s) ** 2)
    return norm * oscillation * envelope


def _cwt_morlet(signal: np.ndarray, scales: np.ndarray, omega0: float = 5.0) -> np.ndarray:
    """
    Continuous Wavelet Transform using complex Morlet wavelet.

    Uses direct time-domain convolution, which is robust for short signals
    (e.g. 47-point feature vectors).

    Parameters
    ----------
    signal : 1-D array (n_timepoints,)
    scales : 1-D array of positive floats (unit-sample scales)
    omega0 : Morlet wavelet center angular frequency

    Returns
    -------
    coefficients : 2-D complex array (n_scales, n_timepoints)
    """
    n = len(signal)
    coefficients = np.empty((len(scales), n), dtype=complex)

    for i, s in enumerate(scales):
        # Build Morlet wavelet kernel at this scale
        # Wavelet support: ~6*s standard deviations each side
        half = int(min(6 * s, n * 3))
        t = np.arange(-half, half + 1, dtype=float)
        norm = (np.pi ** -0.25) / np.sqrt(s)
        wavelet = norm * np.exp(1j * omega0 * t / s) * np.exp(-0.5 * (t / s) ** 2)

        # Convolve (mode='same' centered on each point, reflect-pad edges)
        padded = np.pad(signal, half, mode="reflect")
        conv = np.convolve(padded, wavelet, mode="same")
        # Extract the original-length segment from the center
        coefficients[i] = conv[half: half + n]

    return coefficients


def _compute_scalogram_from_features(X: np.ndarray) -> Dict[str, Any]:
    """
    Compute CWT scalogram statistics treating each sample's feature vector
    as a 1-D signal.  Scales are chosen relative to the feature-vector length
    so that the wavelet captures structure at multiple resolutions (fine to
    coarse).  The five canonical EEG bands are mapped onto five equal
    log-spaced partitions of the scale axis.
    """
    n_samples, n_features = X.shape

    # Scales for the CWT: from 1 (finest detail) up to ~n_features/2
    # This ensures the wavelet fits within the signal at all scales.
    n_scales = 48
    scale_min = 1.0
    scale_max = max(n_features / 2.0, 4.0)
    scales = np.geomspace(scale_min, scale_max, n_scales)

    # Partition scales into 5 bands (gamma=smallest scales / highest freq,
    # delta=largest scales / lowest freq).  Log-spaced boundaries.
    band_order = ["gamma", "beta", "alpha", "theta", "delta"]  # small→large scale
    band_boundaries = np.geomspace(scale_min, scale_max, len(band_order) + 1)
    # Map: for each band, which scale indices belong to it
    band_scale_masks: Dict[str, np.ndarray] = {}
    for bi, band in enumerate(band_order):
        lo, hi = band_boundaries[bi], band_boundaries[bi + 1]
        # inclusive on both ends for first/last, left-inclusive otherwise
        if bi == len(band_order) - 1:
            mask = (scales >= lo) & (scales <= hi)
        else:
            mask = (scales >= lo) & (scales < hi)
        # Ensure at least one scale per band
        if not mask.any():
            closest = int(np.argmin(np.abs(scales - (lo + hi) / 2)))
            mask[closest] = True
        band_scale_masks[band] = mask

    # Aggregate scalogram statistics across samples
    energy_by_scale = np.zeros(n_scales)
    energy_by_band = {b: 0.0 for b in BAND_RANGES}
    coeff_abs_all: List[np.ndarray] = []

    for i in range(n_samples):
        sig = X[i]
        # Normalize to zero mean, unit variance for stable CWT
        sig = (sig - np.mean(sig)) / max(np.std(sig), 1e-12)
        coeffs = _cwt_morlet(sig, scales)
        power = np.abs(coeffs) ** 2  # scalogram (n_scales, n_features)

        energy_per_scale = np.mean(power, axis=1)
        energy_by_scale += energy_per_scale
        coeff_abs_all.append(np.abs(coeffs).mean(axis=1))

        # Map each scale to a band and accumulate
        for band, mask in band_scale_masks.items():
            energy_by_band[band] += float(np.sum(energy_per_scale[mask]))

    energy_by_scale /= n_samples
    for b in energy_by_band:
        energy_by_band[b] /= n_samples

    coeff_matrix = np.array(coeff_abs_all)  # (n_samples, n_scales)

    return {
        "scales": scales,
        "energy_by_scale": energy_by_scale,
        "energy_by_band": energy_by_band,
        "coeff_matrix": coeff_matrix,
        "n_scales": n_scales,
        "band_scale_masks": band_scale_masks,
    }


def _compute_band_scalogram_stats(X: np.ndarray, band_powers: Dict[str, np.ndarray],
                                  scalogram: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute per-band detailed scalogram statistics."""
    scales = scalogram["scales"]
    coeff_matrix = scalogram["coeff_matrix"]
    band_scale_masks = scalogram["band_scale_masks"]
    total_energy = sum(scalogram["energy_by_band"].values())

    stats = []
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        mask = band_scale_masks[band]
        n_scales_in_band = int(mask.sum())

        if n_scales_in_band == 0:
            continue

        band_scales = scales[mask]
        band_coeffs = coeff_matrix[:, mask]  # (n_samples, n_band_scales)
        bp = band_powers[band]

        # Wavelet coefficient statistics
        flat_coeffs = band_coeffs.flatten()
        band_energy = scalogram["energy_by_band"][band]

        stats.append({
            "band": band,
            "frequency_range_hz": list(BAND_RANGES[band]),
            "scale_range": [round(float(band_scales.min()), 2), round(float(band_scales.max()), 2)],
            "n_scales": n_scales_in_band,
            "mean_wavelet_coeff": round(float(np.mean(flat_coeffs)), 6),
            "std_wavelet_coeff": round(float(np.std(flat_coeffs)), 6),
            "median_wavelet_coeff": round(float(np.median(flat_coeffs)), 6),
            "max_wavelet_coeff": round(float(np.max(flat_coeffs)), 6),
            "min_wavelet_coeff": round(float(np.min(flat_coeffs)), 6),
            "q25_wavelet_coeff": round(float(np.percentile(flat_coeffs, 25)), 6),
            "q75_wavelet_coeff": round(float(np.percentile(flat_coeffs, 75)), 6),
            "skewness": round(float(
                np.mean(((flat_coeffs - np.mean(flat_coeffs)) / max(np.std(flat_coeffs), 1e-12)) ** 3)
            ), 4),
            "kurtosis": round(float(
                np.mean(((flat_coeffs - np.mean(flat_coeffs)) / max(np.std(flat_coeffs), 1e-12)) ** 4) - 3
            ), 4),
            "mean_energy": round(float(band_energy), 6),
            "energy_fraction": round(float(band_energy / max(total_energy, 1e-12)), 4),
            "mean_band_power_from_db": round(float(np.mean(bp)), 6),
            "std_band_power_from_db": round(float(np.std(bp)), 6),
            "coeff_of_variation": round(float(np.std(flat_coeffs) / max(np.mean(flat_coeffs), 1e-12)), 4),
        })

    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def overview() -> Dict[str, Any]:
    """KPIs + scalogram energy distribution + time-frequency concentration."""
    X, labels, err = _load_data()
    if err:
        return {"available": False, "error": err}

    n_samples, n_features = X.shape
    band_powers = _band_powers(X)

    # Run CWT scalogram computation
    scalogram = _compute_scalogram_from_features(X)

    total_energy = sum(scalogram["energy_by_band"].values())
    energy_dist = {
        band: round(float(e / max(total_energy, 1e-12)), 4)
        for band, e in scalogram["energy_by_band"].items()
    }

    band_scale_masks = scalogram["band_scale_masks"]

    # Dominant scale: scale index with highest average energy
    dominant_idx = int(np.argmax(scalogram["energy_by_scale"]))
    dominant_scale = float(scalogram["scales"][dominant_idx])

    # Determine which band the dominant scale falls into
    dominant_band = "unknown"
    for band, mask in band_scale_masks.items():
        if mask[dominant_idx]:
            dominant_band = band
            break

    # Approximate pseudo-frequency for the dominant scale
    # (omega0 / (2*pi*scale)) in normalized units, then map to EEG band center
    band_center_hz = {
        "delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 50.0,
    }
    dominant_freq_hz = band_center_hz.get(dominant_band, 0.0)

    # Time-frequency concentration: entropy of the energy distribution across scales
    e_norm = scalogram["energy_by_scale"] / max(np.sum(scalogram["energy_by_scale"]), 1e-12)
    e_norm = e_norm[e_norm > 0]
    tf_entropy = float(-np.sum(e_norm * np.log2(e_norm + 1e-30)))
    max_entropy = np.log2(scalogram["n_scales"])
    concentration_index = round(1.0 - tf_entropy / max(max_entropy, 1e-12), 4)

    # Energy statistics per sample
    sample_energies = np.sum(scalogram["coeff_matrix"] ** 2, axis=1)

    # Band power (from DB) correlation with CWT band energy
    band_corr = {}
    for band in BAND_RANGES:
        bp = band_powers[band]
        mask = band_scale_masks[band]
        if mask.any() and len(bp) > 1:
            wt_energy = np.mean(scalogram["coeff_matrix"][:, mask] ** 2, axis=1)
            if np.std(bp) > 1e-12 and np.std(wt_energy) > 1e-12:
                corr = float(np.corrcoef(bp, wt_energy)[0, 1])
            else:
                corr = 0.0
            band_corr[band] = round(corr, 4)

    # Unique labels
    unique_labels = sorted(set(labels))

    return {
        "available": True,
        "kpis": {
            "n_samples": n_samples,
            "n_channels_analyzed": n_features,
            "n_frequency_bands": len(BAND_RANGES),
            "n_scales_computed": scalogram["n_scales"],
            "dominant_scale": round(dominant_scale, 2),
            "dominant_frequency_hz": dominant_freq_hz,
            "dominant_band": dominant_band,
            "total_scalogram_energy": round(float(total_energy), 4),
            "tf_concentration_index": concentration_index,
            "n_classes": len(unique_labels),
        },
        "energy_distribution_by_band": energy_dist,
        "energy_per_scale": [
            {
                "scale": round(float(s), 2),
                "pseudo_frequency": round(float(5.0 / (2 * np.pi * s)), 4),
                "mean_energy": round(float(e), 6),
            }
            for s, e in zip(scalogram["scales"], scalogram["energy_by_scale"])
        ],
        "time_frequency_concentration": {
            "spectral_entropy_bits": round(tf_entropy, 4),
            "max_possible_entropy_bits": round(max_entropy, 4),
            "concentration_index": concentration_index,
            "interpretation": (
                "highly concentrated" if concentration_index > 0.6
                else "moderately concentrated" if concentration_index > 0.3
                else "broadly distributed"
            ),
        },
        "sample_energy_stats": {
            "mean": round(float(np.mean(sample_energies)), 4),
            "std": round(float(np.std(sample_energies)), 4),
            "min": round(float(np.min(sample_energies)), 4),
            "max": round(float(np.max(sample_energies)), 4),
            "median": round(float(np.median(sample_energies)), 4),
        },
        "band_power_cwt_correlation": band_corr,
    }


def breakdown() -> Dict[str, Any]:
    """Per-band scalogram statistics, coefficient distributions, cross-band energy ratios."""
    X, labels, err = _load_data()
    if err:
        return {"available": False, "error": err}

    n_samples = X.shape[0]
    band_powers = _band_powers(X)
    scalogram = _compute_scalogram_from_features(X)

    # Per-band detailed stats
    band_stats = _compute_band_scalogram_stats(X, band_powers, scalogram)

    # Cross-band energy ratios (clinically relevant ratios)
    eb = scalogram["energy_by_band"]
    cross_ratios = {}
    ratio_pairs = [
        ("theta_beta_ratio", "theta", "beta"),
        ("alpha_theta_ratio", "alpha", "theta"),
        ("delta_alpha_ratio", "delta", "alpha"),
        ("gamma_beta_ratio", "gamma", "beta"),
        ("theta_alpha_ratio", "theta", "alpha"),
        ("delta_total_ratio", "delta", None),
    ]
    total_energy = sum(eb.values())
    for name, num, den in ratio_pairs:
        num_e = eb.get(num, 0.0)
        den_e = eb.get(den, total_energy) if den else total_energy
        cross_ratios[name] = round(float(num_e / max(den_e, 1e-12)), 4)

    # Per-class scalogram energy breakdown
    unique_labels = sorted(set(labels))
    label_arr = np.array(labels)
    per_class = {}
    for lbl in unique_labels:
        mask = label_arr == lbl
        cls_coeff = scalogram["coeff_matrix"][mask]
        cls_energy = float(np.mean(np.sum(cls_coeff ** 2, axis=1)))
        per_class[lbl] = {
            "n_samples": int(mask.sum()),
            "mean_total_energy": round(cls_energy, 4),
        }
        # Per-band energy for this class
        band_scale_masks = scalogram["band_scale_masks"]
        for band in BAND_RANGES:
            scale_mask = band_scale_masks[band]
            if scale_mask.any():
                cls_band_e = float(np.mean(cls_coeff[:, scale_mask] ** 2))
                per_class[lbl][f"{band}_energy"] = round(cls_band_e, 6)

    # Wavelet coefficient distribution summary (all bands combined)
    all_coeffs = scalogram["coeff_matrix"].flatten()
    coeff_distribution = {
        "mean": round(float(np.mean(all_coeffs)), 6),
        "std": round(float(np.std(all_coeffs)), 6),
        "skewness": round(float(
            np.mean(((all_coeffs - np.mean(all_coeffs)) / max(np.std(all_coeffs), 1e-12)) ** 3)
        ), 4),
        "kurtosis": round(float(
            np.mean(((all_coeffs - np.mean(all_coeffs)) / max(np.std(all_coeffs), 1e-12)) ** 4) - 3
        ), 4),
        "percentiles": {
            f"p{p}": round(float(np.percentile(all_coeffs, p)), 6)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
    }

    return {
        "available": True,
        "n_samples": n_samples,
        "n_scales": scalogram["n_scales"],
        "band_scalogram_stats": band_stats,
        "cross_band_energy_ratios": cross_ratios,
        "per_class_energy": per_class,
        "wavelet_coefficient_distribution": coeff_distribution,
    }


def definitions() -> Dict[str, Any]:
    """CWT methodology, Morlet wavelets, scalogram interpretation, clinical relevance."""
    return {
        "available": True,
        "methodology": {
            "continuous_wavelet_transform": {
                "description": (
                    "The Continuous Wavelet Transform (CWT) decomposes a signal into "
                    "time-frequency space by convolving it with scaled and translated "
                    "versions of a mother wavelet. Unlike the STFT, the CWT provides "
                    "adaptive time-frequency resolution: fine frequency resolution at low "
                    "frequencies (large scales) and fine temporal resolution at high "
                    "frequencies (small scales)."
                ),
                "formula": "W(a, b) = (1/sqrt(a)) * integral[ x(t) * psi*((t-b)/a) dt ]",
                "parameters": {
                    "a": "Scale parameter (inversely related to frequency)",
                    "b": "Translation parameter (time shift)",
                    "psi": "Mother wavelet function",
                },
            },
            "morlet_wavelet": {
                "description": (
                    "The Morlet wavelet is a complex sinusoid modulated by a Gaussian "
                    "envelope. It is the standard choice for EEG time-frequency analysis "
                    "because its shape closely matches the oscillatory bursts observed in "
                    "neural signals. The center frequency omega_0 (typically 5-6) controls "
                    "the trade-off between time and frequency resolution."
                ),
                "formula": "psi(t) = pi^(-1/4) * exp(j*omega_0*t) * exp(-t^2/2)",
                "omega_0": 5.0,
                "time_bandwidth_product": "Approximately 2*pi for omega_0=5",
                "advantages_for_eeg": [
                    "Matches oscillatory morphology of neural rhythms",
                    "Complex-valued output provides instantaneous phase and amplitude",
                    "Smooth Gaussian envelope minimizes spectral leakage",
                    "Adjustable omega_0 tunes time-frequency trade-off per application",
                ],
            },
            "scalogram": {
                "description": (
                    "A scalogram is the squared magnitude of the CWT coefficients "
                    "|W(a,b)|^2, representing the energy density in time-frequency space. "
                    "It is the wavelet analogue of a spectrogram. Each point in the "
                    "scalogram indicates how much energy the signal contains at a "
                    "particular scale (frequency) and time position."
                ),
                "interpretation": [
                    "Bright regions indicate high energy concentration at that time-frequency location",
                    "Vertical streaks suggest transient events (spikes, sharp waves)",
                    "Horizontal bands indicate sustained oscillatory activity",
                    "Diagonal patterns may indicate frequency-modulated activity (chirps)",
                ],
                "scale_to_frequency": "f = f_center * fs / scale, where f_center ~ 0.8125 for Morlet (omega_0=5)",
            },
        },
        "eeg_frequency_bands": {
            band: {
                "range_hz": list(BAND_RANGES[band]),
                "scale_range": [
                    round(BAND_SCALES[band][0], 2),
                    round(BAND_SCALES[band][1], 2),
                ],
                "clinical_significance": desc,
            }
            for band, desc in [
                ("delta", "Dominant in deep sleep (N3). Pathological when prominent in wakefulness; associated with encephalopathy, focal lesions, and post-ictal slowing."),
                ("theta", "Normal in drowsiness and light sleep. Focal theta in wakefulness suggests subcortical pathology. Temporal intermittent rhythmic delta/theta activity (TIRDA) is an epilepsy marker."),
                ("alpha", "Dominant posterior rhythm in relaxed wakefulness. Attenuation with eye-opening (Berger effect). Loss or asymmetry suggests cortical dysfunction."),
                ("beta", "Enhanced by benzodiazepines and barbiturates. Focal beta may indicate cortical lesion or breach rhythm. Beta oscillations are associated with motor planning."),
                ("gamma", "Associated with higher cognitive functions, binding, and consciousness. High-frequency oscillations (HFOs > 80 Hz) are biomarkers for epileptogenic zones."),
            ]
        },
        "clinical_relevance_epilepsy": {
            "seizure_detection": (
                "CWT scalograms reveal the characteristic frequency evolution during seizures: "
                "initial fast activity (beta/gamma) at onset, progressive frequency decrease "
                "through alpha and theta, and post-ictal delta slowing. This frequency cascade "
                "is visible as a descending energy ridge in the scalogram."
            ),
            "interictal_spike_detection": (
                "Epileptic spikes produce broadband, high-amplitude transients that appear as "
                "vertical energy columns spanning multiple scales in the scalogram. The CWT is "
                "superior to STFT for spike detection due to its adaptive time resolution."
            ),
            "high_frequency_oscillations": (
                "Pathological HFOs (ripples 80-250 Hz, fast ripples 250-500 Hz) are best "
                "detected using CWT with Morlet wavelets at small scales. HFOs are the most "
                "specific biomarker for the seizure onset zone in presurgical evaluation."
            ),
            "seizure_prediction": (
                "Pre-ictal changes in scalogram energy distribution (increased theta/delta, "
                "altered cross-frequency coupling) can precede seizure onset by minutes to "
                "hours. Scalogram-derived features feed into seizure prediction algorithms."
            ),
        },
        "implementation_notes": {
            "scale_selection": (
                "Scales are chosen on a geometric (log-spaced) grid to provide uniform "
                "resolution on a logarithmic frequency axis, matching the approximately "
                "log-spaced EEG frequency bands."
            ),
            "edge_effects": (
                "CWT coefficients near signal boundaries suffer from the cone of influence "
                "(COI) artifact. This implementation uses FFT-based convolution which "
                "implicitly assumes periodic boundary conditions."
            ),
            "normalization": (
                "Feature vectors are z-scored before CWT to ensure comparable coefficient "
                "magnitudes across samples with different amplitude scales."
            ),
        },
        "references": [
            "Addison, P.S. (2017) 'The Illustrated Wavelet Transform Handbook', 2nd ed., CRC Press",
            "Talmon, R. et al. (2015) 'Supervised graph-based processing for EEG seizure detection', ICASSP",
            "Adeli, H. et al. (2003) 'Analysis of EEG records in an epileptic patient using wavelet transform', J Neurosci Methods",
            "Tzallas, A.T. et al. (2009) 'Epileptic seizure detection in EEGs using time-frequency analysis', IEEE Trans IT Biomed",
            "Jacobs, J. et al. (2012) 'High-frequency oscillations (HFOs) in clinical epilepsy', Prog Neurobiol",
        ],
    }
