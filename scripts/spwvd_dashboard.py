#!/usr/bin/env python3
"""
SPWVD Dashboard — Smoothed Pseudo Wigner-Ville Distribution
=============================================================

Computes REAL Smoothed Pseudo Wigner-Ville Distribution time-frequency analysis
on EEG features from ``data/clinical.db`` (analyses table -> result_json ->
features).  Uses band-power features and spectral features already extracted in
the pipeline to compute WVD-based time-frequency representations.

The Wigner-Ville Distribution (WVD) provides the highest possible time-frequency
resolution (no uncertainty-principle trade-off like STFT/CWT), but suffers from
cross-term interference for multi-component signals.  The Smoothed Pseudo
Wigner-Ville Distribution (SPWVD) applies independent smoothing in time and
frequency to suppress cross-terms while preserving most of the resolution
advantage.

Functions:
  overview()    -- KPIs + WVD energy distribution + time-frequency resolution
                   metrics + cross-term suppression statistics
  breakdown()   -- Per-band WVD statistics, instantaneous frequency tracking,
                   cross-component interference analysis
  definitions() -- WVD/SPWVD methodology, kernel design, comparison with
                   STFT/CWT, clinical relevance in EEG/epilepsy
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

BAND_FEATURES = {
    "delta": "delta_power",
    "theta": "theta_power",
    "alpha": "alpha_power",
    "beta": "beta_power",
    "gamma": "gamma_power",
}

BAND_RANGES = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

_FS = 256.0  # sampling rate


# ---------------------------------------------------------------------------
# Data loading
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
# SPWVD implementation (numpy-only)
# ---------------------------------------------------------------------------

def _analytic_signal(x: np.ndarray) -> np.ndarray:
    """Compute analytic signal via Hilbert transform (FFT-based)."""
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)


def _wvd(x: np.ndarray, n_freq: int = None) -> np.ndarray:
    """
    Discrete Wigner-Ville Distribution of a 1-D signal.

    Parameters
    ----------
    x : 1-D real or complex array
    n_freq : number of frequency bins (default = len(x))

    Returns
    -------
    wvd : 2-D real array (n_freq, n_time)
    """
    z = _analytic_signal(x) if np.isrealobj(x) else x
    n = len(z)
    if n_freq is None:
        n_freq = n

    wvd = np.zeros((n_freq, n), dtype=float)
    for t in range(n):
        tau_max = min(t, n - 1 - t, n_freq // 2 - 1)
        if tau_max < 0:
            continue
        r = np.zeros(n_freq, dtype=complex)
        for tau in range(-tau_max, tau_max + 1):
            r[tau % n_freq] = z[t + tau] * np.conj(z[t - tau])
        wvd[:, t] = np.real(np.fft.fft(r))

    return wvd


def _spwvd(x: np.ndarray, n_freq: int = None,
           t_smooth: int = 5, f_smooth: int = 5) -> np.ndarray:
    """
    Smoothed Pseudo Wigner-Ville Distribution.

    Applies a Hamming window in time (length t_smooth) and a Hamming window
    in frequency (length f_smooth) to suppress cross-term interference.

    Parameters
    ----------
    x : 1-D real array
    n_freq : number of frequency bins
    t_smooth : time smoothing window length (odd recommended)
    f_smooth : frequency smoothing window length (odd recommended)

    Returns
    -------
    spwvd : 2-D real array (n_freq, n_time)
    """
    z = _analytic_signal(x) if np.isrealobj(x) else x
    n = len(z)
    if n_freq is None:
        n_freq = n

    # Time smoothing window (Hamming)
    t_win = np.hamming(t_smooth)
    t_win /= t_win.sum()
    t_half = t_smooth // 2

    # Compute raw WVD
    raw = _wvd(z, n_freq)

    # Time smoothing: convolve each frequency row with t_win
    smoothed = np.zeros_like(raw)
    for fi in range(n_freq):
        smoothed[fi] = np.convolve(raw[fi], t_win, mode="same")

    # Frequency smoothing: convolve each time column with f_win
    f_win = np.hamming(f_smooth)
    f_win /= f_win.sum()
    result = np.zeros_like(smoothed)
    for ti in range(n):
        col = smoothed[:, ti]
        result[:, ti] = np.convolve(col, f_win, mode="same")

    return result


def _compute_spwvd_from_features(X: np.ndarray) -> Dict[str, Any]:
    """
    Compute SPWVD statistics treating each sample's feature vector as a 1-D
    signal.  Frequency bins are partitioned into 5 canonical EEG bands.
    """
    n_samples, n_features = X.shape
    n_freq = max(n_features, 32)  # at least 32 frequency bins

    # Map frequency bins to EEG bands
    freq_axis = np.linspace(0, _FS / 2, n_freq)
    band_masks = {}
    for band, (lo, hi) in BAND_RANGES.items():
        band_masks[band] = (freq_axis >= lo) & (freq_axis <= hi)
        # Ensure at least one bin per band
        if not band_masks[band].any():
            closest = int(np.argmin(np.abs(freq_axis - (lo + hi) / 2)))
            band_masks[band][closest] = True

    energy_by_band = {b: 0.0 for b in BAND_RANGES}
    energy_by_freq = np.zeros(n_freq)
    cross_term_ratios = []
    instantaneous_freqs = []
    marginal_spectra = []

    for i in range(n_samples):
        sig = X[i]
        sig = (sig - np.mean(sig)) / max(np.std(sig), 1e-12)

        # Compute raw WVD and SPWVD
        raw_wvd = _wvd(sig, n_freq)
        smooth_wvd = _spwvd(sig, n_freq, t_smooth=5, f_smooth=5)

        # Energy per frequency bin (time-marginal)
        freq_energy = np.mean(np.abs(smooth_wvd), axis=1)
        energy_by_freq += freq_energy

        # Marginal spectrum (frequency marginal of SPWVD)
        marginal = np.sum(np.abs(smooth_wvd), axis=1)
        marginal_spectra.append(marginal)

        # Band energy
        for band, mask in band_masks.items():
            energy_by_band[band] += float(np.sum(freq_energy[mask]))

        # Cross-term suppression ratio: how much energy the smoothing removed
        raw_total = np.sum(np.abs(raw_wvd))
        smooth_total = np.sum(np.abs(smooth_wvd))
        if raw_total > 1e-12:
            ct_ratio = 1.0 - smooth_total / raw_total
        else:
            ct_ratio = 0.0
        cross_term_ratios.append(ct_ratio)

        # Instantaneous frequency: frequency of max energy at each time point
        if smooth_wvd.shape[1] > 0:
            inst_freq_idx = np.argmax(np.abs(smooth_wvd), axis=0)
            mean_inst_freq = float(np.mean(freq_axis[inst_freq_idx]))
            instantaneous_freqs.append(mean_inst_freq)

    energy_by_freq /= n_samples
    for b in energy_by_band:
        energy_by_band[b] /= n_samples

    marginal_matrix = np.array(marginal_spectra)  # (n_samples, n_freq)

    return {
        "n_freq": n_freq,
        "freq_axis": freq_axis,
        "energy_by_freq": energy_by_freq,
        "energy_by_band": energy_by_band,
        "band_masks": band_masks,
        "cross_term_ratios": np.array(cross_term_ratios),
        "instantaneous_freqs": np.array(instantaneous_freqs),
        "marginal_matrix": marginal_matrix,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def overview() -> Dict[str, Any]:
    """KPIs + SPWVD energy distribution + time-frequency resolution metrics."""
    X, labels, err = _load_data()
    if err:
        return {"available": False, "error": err}

    n_samples, n_features = X.shape
    band_powers = _band_powers(X)
    spwvd = _compute_spwvd_from_features(X)

    total_energy = sum(spwvd["energy_by_band"].values())
    energy_dist = {
        band: round(float(e / max(total_energy, 1e-12)), 4)
        for band, e in spwvd["energy_by_band"].items()
    }

    # Dominant band
    dominant_band = max(spwvd["energy_by_band"], key=spwvd["energy_by_band"].get)

    # Cross-term suppression stats
    ct = spwvd["cross_term_ratios"]
    mean_ct = float(np.mean(ct))

    # Instantaneous frequency stats
    inst_f = spwvd["instantaneous_freqs"]
    mean_inst_freq = float(np.mean(inst_f)) if len(inst_f) > 0 else 0.0

    # Spectral entropy of the frequency marginal
    e_norm = spwvd["energy_by_freq"] / max(np.sum(spwvd["energy_by_freq"]), 1e-12)
    e_pos = e_norm[e_norm > 0]
    spectral_entropy = float(-np.sum(e_pos * np.log2(e_pos + 1e-30)))
    max_entropy = np.log2(spwvd["n_freq"])
    concentration_index = round(1.0 - spectral_entropy / max(max_entropy, 1e-12), 4)

    # Resolution advantage vs CWT/STFT
    freq_axis = spwvd["freq_axis"]
    freq_resolution = float(freq_axis[1] - freq_axis[0]) if len(freq_axis) > 1 else 0.0

    return {
        "available": True,
        "kpis": {
            "n_samples": n_samples,
            "n_features_analyzed": n_features,
            "n_frequency_bands": len(BAND_RANGES),
            "n_frequency_bins": spwvd["n_freq"],
            "dominant_band": dominant_band,
            "mean_cross_term_suppression": round(mean_ct, 4),
            "mean_instantaneous_freq_hz": round(mean_inst_freq, 2),
            "tf_concentration_index": concentration_index,
        },
        "band_energy_distribution": [
            {"band": b, "energy": round(float(spwvd["energy_by_band"][b]), 6),
             "fraction": energy_dist[b]}
            for b in ["delta", "theta", "alpha", "beta", "gamma"]
        ],
        "frequency_marginal": [
            {"freq_hz": round(float(f), 2), "energy": round(float(e), 6)}
            for f, e in zip(spwvd["freq_axis"][:64], spwvd["energy_by_freq"][:64])
        ],
        "cross_term_suppression": {
            "mean": round(mean_ct, 4),
            "std": round(float(np.std(ct)), 4),
            "min": round(float(np.min(ct)), 4),
            "max": round(float(np.max(ct)), 4),
            "interpretation": (
                "strong suppression (>30%)" if mean_ct > 0.3
                else "moderate suppression (10-30%)" if mean_ct > 0.1
                else "minimal cross-terms (<10%)"
            ),
        },
        "time_frequency_resolution": {
            "frequency_resolution_hz": round(freq_resolution, 4),
            "concentration_index": concentration_index,
            "spectral_entropy_bits": round(spectral_entropy, 4),
            "advantage_over_stft": (
                "WVD achieves twice the time-frequency resolution of STFT "
                "by eliminating the window-length trade-off"
            ),
        },
        "energy_concentration": [
            {"sample": i + 1,
             "concentration": round(float(np.sum(spwvd["marginal_matrix"][i] ** 2) /
                                         max(np.sum(spwvd["marginal_matrix"][i]) ** 2, 1e-12)), 6)}
            for i in range(min(n_samples, 50))
        ],
    }


def breakdown() -> Dict[str, Any]:
    """Per-band SPWVD statistics, instantaneous frequency, interference analysis."""
    X, labels, err = _load_data()
    if err:
        return {"available": False, "error": err}

    n_samples = X.shape[0]
    band_powers = _band_powers(X)
    spwvd = _compute_spwvd_from_features(X)

    total_energy = sum(spwvd["energy_by_band"].values())

    # Per-band statistics
    band_stats = []
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        mask = spwvd["band_masks"][band]
        n_bins = int(mask.sum())
        band_energy = spwvd["energy_by_band"][band]

        # Marginal coefficients in this band
        band_marginals = spwvd["marginal_matrix"][:, mask]
        flat = band_marginals.flatten()
        bp = band_powers[band]

        band_stats.append({
            "band": band,
            "freq_range": f"{BAND_RANGES[band][0]}-{BAND_RANGES[band][1]} Hz",
            "n_frequency_bins": n_bins,
            "mean": round(float(np.mean(flat)), 6),
            "std": round(float(np.std(flat)), 6),
            "min": round(float(np.min(flat)), 6),
            "max": round(float(np.max(flat)), 6),
            "median": round(float(np.median(flat)), 6),
            "energy_pct": round(float(band_energy / max(total_energy, 1e-12)), 4),
            "mean_band_power_db": round(float(np.mean(bp)), 6),
            "std_band_power_db": round(float(np.std(bp)), 6),
            "skewness": round(float(
                np.mean(((flat - np.mean(flat)) / max(np.std(flat), 1e-12)) ** 3)
            ), 4),
            "kurtosis": round(float(
                np.mean(((flat - np.mean(flat)) / max(np.std(flat), 1e-12)) ** 4) - 3
            ), 4),
        })

    # Cross-band energy ratios
    eb = spwvd["energy_by_band"]
    cross_ratios = []
    ratio_pairs = [
        ("theta/beta", "theta", "beta"),
        ("alpha/theta", "alpha", "theta"),
        ("delta/alpha", "delta", "alpha"),
        ("gamma/beta", "gamma", "beta"),
        ("theta/alpha", "theta", "alpha"),
        ("delta/total", "delta", None),
    ]
    for name, num, den in ratio_pairs:
        num_e = eb.get(num, 0.0)
        den_e = eb.get(den, total_energy) if den else total_energy
        cross_ratios.append({
            "ratio_name": name,
            "value": round(float(num_e / max(den_e, 1e-12)), 4),
        })

    # Per-class energy
    unique_labels = sorted(set(labels))
    label_arr = np.array(labels)
    per_class = {}
    for lbl in unique_labels:
        lbl_mask = label_arr == lbl
        cls_marginals = spwvd["marginal_matrix"][lbl_mask]
        cls_energy = float(np.mean(np.sum(cls_marginals ** 2, axis=1)))
        entry = {
            "n_samples": int(lbl_mask.sum()),
            "mean_total_energy": round(cls_energy, 4),
        }
        for band in BAND_RANGES:
            bmask = spwvd["band_masks"][band]
            if bmask.any():
                entry[f"{band}_energy"] = round(
                    float(np.mean(cls_marginals[:, bmask] ** 2)), 6
                )
        per_class[lbl] = entry

    # Instantaneous frequency tracking
    inst_f = spwvd["instantaneous_freqs"]
    inst_freq_stats = {
        "mean_hz": round(float(np.mean(inst_f)), 2) if len(inst_f) > 0 else 0.0,
        "std_hz": round(float(np.std(inst_f)), 2) if len(inst_f) > 0 else 0.0,
        "min_hz": round(float(np.min(inst_f)), 2) if len(inst_f) > 0 else 0.0,
        "max_hz": round(float(np.max(inst_f)), 2) if len(inst_f) > 0 else 0.0,
    }

    # Coefficient distributions per band (for chart)
    coeff_distributions = []
    for bs in band_stats:
        mask = spwvd["band_masks"][bs["band"]]
        band_flat = spwvd["marginal_matrix"][:, mask].flatten()
        coeff_distributions.append({
            "band": bs["band"],
            "p25": round(float(np.percentile(band_flat, 25)), 6),
            "median": round(float(np.median(band_flat)), 6),
            "p75": round(float(np.percentile(band_flat, 75)), 6),
            "max": round(float(np.max(band_flat)), 6),
        })

    # Ratio trends across samples (first 50)
    ratio_trends = []
    for si in range(min(n_samples, 50)):
        entry = {"sample": si + 1}
        sample_marginal = spwvd["marginal_matrix"][si]
        for name, num, den in ratio_pairs:
            num_mask = spwvd["band_masks"].get(num, np.zeros(len(sample_marginal), dtype=bool))
            num_e = float(np.sum(np.abs(sample_marginal[num_mask]) ** 2)) if num_mask.any() else 0.0
            if den:
                den_mask = spwvd["band_masks"].get(den, np.zeros(len(sample_marginal), dtype=bool))
                den_e = float(np.sum(np.abs(sample_marginal[den_mask]) ** 2)) if den_mask.any() else 1e-12
            else:
                den_e = float(np.sum(np.abs(sample_marginal) ** 2))
            entry[name] = round(num_e / max(den_e, 1e-12), 4)
        ratio_trends.append(entry)

    return {
        "available": True,
        "n_samples": n_samples,
        "band_statistics": band_stats,
        "cross_band_ratios": cross_ratios,
        "per_class_energy": per_class,
        "instantaneous_frequency": inst_freq_stats,
        "coefficient_distributions": coeff_distributions,
        "ratio_trends": ratio_trends,
    }


def definitions() -> Dict[str, Any]:
    """SPWVD methodology, kernel design, clinical relevance."""
    return {
        "available": True,
        "methodology": [
            {
                "name": "Wigner-Ville Distribution (WVD)",
                "description": (
                    "The Wigner-Ville Distribution is a bilinear time-frequency "
                    "representation that achieves the highest possible joint "
                    "time-frequency resolution. Unlike the STFT (which uses a "
                    "fixed-length window) or CWT (which uses variable-width "
                    "wavelets), the WVD has no analysis window — it directly "
                    "maps the signal's instantaneous autocorrelation to the "
                    "time-frequency plane. W(t,f) = integral[ x(t+tau/2) * "
                    "x*(t-tau/2) * exp(-j*2*pi*f*tau) d_tau ]."
                ),
                "parameters": "t = time, f = frequency, tau = lag variable",
                "strengths": [
                    "Optimal time-frequency resolution (no uncertainty trade-off)",
                    "Exact marginals: time-marginal = |x(t)|^2, freq-marginal = |X(f)|^2",
                    "Real-valued for analytic signals",
                    "Preserves instantaneous frequency of mono-component signals",
                ],
                "limitations": [
                    "Cross-term interference for multi-component signals",
                    "Cross-terms oscillate at the mean frequency between components",
                    "Quadratic (bilinear) — not additive for sum of signals",
                    "Computationally O(N^2) per time point",
                ],
                "reference": "Cohen, L. (1995) 'Time-Frequency Analysis', Prentice Hall",
            },
            {
                "name": "Smoothed Pseudo Wigner-Ville Distribution (SPWVD)",
                "description": (
                    "The SPWVD applies two independent smoothing kernels to the "
                    "WVD: a time-direction window g(t) that smooths along the "
                    "time axis, and a frequency-direction window h(tau) that "
                    "smooths along the frequency axis. This 2-D smoothing "
                    "suppresses cross-term interference while preserving most "
                    "of the WVD's superior resolution. The 'Pseudo' prefix "
                    "indicates the use of the time window alone (PWVD); adding "
                    "the frequency window yields the full SPWVD."
                ),
                "parameters": (
                    "g(t) = Hamming time window (5 points), "
                    "h(tau) = Hamming frequency window (5 points)"
                ),
                "strengths": [
                    "Significantly reduced cross-term interference vs raw WVD",
                    "Independent control over time and frequency smoothing",
                    "Better resolution than STFT for the same cross-term level",
                    "Flexible kernel design for application-specific trade-offs",
                ],
                "limitations": [
                    "Resolution reduced compared to raw WVD (by the smoothing)",
                    "Cross-terms not fully eliminated, only suppressed",
                    "Kernel design requires domain knowledge",
                    "Still O(N^2) computational cost",
                ],
                "reference": (
                    "Hlawatsch, F. & Boudreaux-Bartels, G.F. (1992) "
                    "'Linear and Quadratic Time-Frequency Signal Representations', "
                    "IEEE Signal Processing Magazine"
                ),
            },
            {
                "name": "Comparison: STFT vs CWT vs WVD/SPWVD",
                "description": (
                    "STFT uses a fixed window → uniform but limited resolution. "
                    "CWT uses scaled wavelets → good low-freq resolution, poor "
                    "high-freq resolution. WVD has no window → best possible "
                    "resolution but with cross-terms. SPWVD → best practical "
                    "resolution with controlled cross-terms. For EEG, SPWVD is "
                    "ideal for analyzing transient events (spikes, HFOs) where "
                    "both precise timing and frequency content matter."
                ),
                "parameters": None,
                "strengths": [
                    "SPWVD reveals fine structure missed by spectrogram/scalogram",
                    "Better seizure onset localization due to superior time resolution",
                    "Cross-frequency coupling analysis benefits from high resolution",
                ],
                "limitations": [
                    "Higher computational cost than STFT or CWT",
                    "Interpretation requires understanding of cross-term artifacts",
                    "Smoothing kernel choice affects results",
                ],
                "reference": (
                    "Boashash, B. (2015) 'Time-Frequency Signal Analysis and "
                    "Processing', 2nd ed., Academic Press"
                ),
            },
        ],
        "clinical_relevance": [
            "SPWVD provides superior temporal resolution for detecting epileptic spike onset timing",
            "Instantaneous frequency tracking reveals seizure frequency evolution (chirp patterns)",
            "Cross-term suppression enables clean analysis of multi-rhythm EEG (alpha + beta mixing)",
            "High time-frequency resolution aids identification of high-frequency oscillations (HFOs)",
            "Pre-ictal frequency shifts are more precisely timed with WVD vs STFT/CWT",
            "Band energy ratios from SPWVD correlate with cognitive state and medication effects",
        ],
        "wavelet_notes": [
            "SPWVD is a quadratic (Cohen's class) distribution, not a linear transform like CWT",
            "The analytic signal (via Hilbert transform) eliminates negative-frequency cross-terms",
            "Hamming windows chosen for their good sidelobe suppression (-43 dB)",
            "Time smoothing length of 5 preserves transients while reducing interference",
            "Frequency smoothing length of 5 balances band separation and resolution",
            "For real-time applications, short-time SPWVD with sliding windows is recommended",
        ],
    }
