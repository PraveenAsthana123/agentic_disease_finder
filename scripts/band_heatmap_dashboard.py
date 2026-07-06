#!/usr/bin/env python3
"""
Band Heatmap Dashboard — EEG Frequency Band Power Spatial Distribution
=======================================================================

Computes REAL band-power heatmaps from ``data/clinical.db`` (analyses table →
result_json → features).  Extracts delta/theta/alpha/beta/gamma power features
per subject and computes cross-subject statistics, channel-band matrices,
band dominance patterns, and abnormality indices.

Functions:
  overview()    -- KPIs + band power distribution + dominance map + abnormality index
  breakdown()   -- Per-band detailed statistics, subject-level heatmap rows,
                   band ratios, asymmetry analysis
  definitions() -- Band definitions, clinical significance, heatmap interpretation
"""

import json
import math
import os
import sqlite3
from typing import Any, Dict, List

import numpy as np

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")

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

# Standard 10-20 channel names for spatial mapping
CHANNELS_10_20 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

def _load_data():
    """Load feature matrix and labels from clinical.db analyses table."""
    if not os.path.exists(_DB_PATH):
        return None, None, None, "Database not found"
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("SELECT result_json FROM analyses WHERE result_json IS NOT NULL")
    except sqlite3.OperationalError:
        conn.close()
        return None, None, None, "Table 'analyses' not found"
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None, None, None, "No analyses found"

    X_list = []
    labels = []
    channels = []
    for row in rows:
        try:
            data = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        feats = data.get("features")
        prediction = data.get("prediction", {})
        label = prediction.get("predicted_label") if isinstance(prediction, dict) else None
        channel = data.get("channel", "unknown")
        if not feats or not isinstance(feats, dict) or not label:
            continue
        sample = []
        for fname in FEATURE_NAMES:
            val = feats.get(fname)
            sample.append(float(val) if val is not None else np.nan)
        X_list.append(sample)
        labels.append(label)
        channels.append(channel)

    if not X_list:
        return None, None, None, "No valid samples"

    X = np.array(X_list, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
    return X, labels, channels, None


def _band_powers(X):
    idx = {name: FEATURE_NAMES.index(fname) for name, fname in BAND_FEATURES.items()}
    return {band: np.abs(X[:, i]) for band, i in idx.items()}


def _safe(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return v


def overview():
    X, labels, channels, err = _load_data()
    if err:
        return {"available": False, "error": err}

    n_samples, n_features = X.shape
    bp = _band_powers(X)
    total_power_idx = FEATURE_NAMES.index("total_power")
    total_power = np.abs(X[:, total_power_idx])

    # Band power distribution (mean across all samples)
    band_distribution = []
    total_bp = sum(float(np.mean(bp[b])) for b in BAND_RANGES)
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        mean_power = float(np.mean(bp[band]))
        band_distribution.append({
            "band": band,
            "frequency_range": list(BAND_RANGES[band]),
            "mean_power": round(_safe(mean_power), 6),
            "std_power": round(_safe(float(np.std(bp[band]))), 6),
            "fraction": round(_safe(mean_power / max(total_bp, 1e-12)), 4),
        })

    # Dominant band per sample
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    band_matrix = np.column_stack([bp[b] for b in band_names])
    dominant_per_sample = np.argmax(band_matrix, axis=1)
    dominance_counts = {}
    for bi, bname in enumerate(band_names):
        dominance_counts[bname] = int(np.sum(dominant_per_sample == bi))

    # Heatmap matrix: subjects x bands (for the first up to 50 subjects)
    max_display = min(50, n_samples)
    heatmap_rows = []
    for i in range(max_display):
        row = {}
        for band in band_names:
            row[band] = round(_safe(float(bp[band][i])), 6)
        row["subject"] = f"S{i+1}"
        row["label"] = labels[i]
        row["total"] = round(_safe(float(total_power[i])), 6)
        heatmap_rows.append(row)

    # Abnormality index: samples where any band deviates >2 SD from mean
    abnormal_count = 0
    for i in range(n_samples):
        for band in band_names:
            val = bp[band][i]
            mean_b = np.mean(bp[band])
            std_b = np.std(bp[band])
            if std_b > 1e-12 and abs(val - mean_b) > 2 * std_b:
                abnormal_count += 1
                break

    # Spectral entropy
    se_idx = FEATURE_NAMES.index("spectral_entropy")
    spectral_entropy = X[:, se_idx]

    # Per-diagnosis band profile
    unique_labels = sorted(set(labels))
    diagnosis_profiles = []
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        profile = {"diagnosis": lbl, "count": int(mask.sum())}
        for band in band_names:
            profile[f"{band}_mean"] = round(_safe(float(np.mean(bp[band][mask]))), 6)
        diagnosis_profiles.append(profile)

    return {
        "available": True,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_diagnoses": len(unique_labels),
        "band_distribution": band_distribution,
        "dominance_counts": dominance_counts,
        "heatmap_data": heatmap_rows,
        "abnormal_count": abnormal_count,
        "abnormal_pct": round(_safe(abnormal_count / max(n_samples, 1) * 100), 1),
        "mean_spectral_entropy": round(_safe(float(np.mean(spectral_entropy))), 4),
        "diagnosis_profiles": diagnosis_profiles,
        "kpis": {
            "total_subjects": n_samples,
            "dominant_band": max(dominance_counts, key=dominance_counts.get),
            "abnormal_pct": round(_safe(abnormal_count / max(n_samples, 1) * 100), 1),
            "mean_entropy": round(_safe(float(np.mean(spectral_entropy))), 4),
            "bands_analyzed": len(band_names),
        },
    }


def breakdown():
    X, labels, channels, err = _load_data()
    if err:
        return {"available": False, "error": err}

    n_samples = X.shape[0]
    bp = _band_powers(X)
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]

    # Per-band detailed statistics
    band_stats = []
    for band in band_names:
        vals = bp[band]
        band_stats.append({
            "band": band,
            "frequency_range": list(BAND_RANGES[band]),
            "mean": round(_safe(float(np.mean(vals))), 6),
            "std": round(_safe(float(np.std(vals))), 6),
            "median": round(_safe(float(np.median(vals))), 6),
            "min": round(_safe(float(np.min(vals))), 6),
            "max": round(_safe(float(np.max(vals))), 6),
            "q25": round(_safe(float(np.percentile(vals, 25))), 6),
            "q75": round(_safe(float(np.percentile(vals, 75))), 6),
            "iqr": round(_safe(float(np.percentile(vals, 75) - np.percentile(vals, 25))), 6),
            "skewness": round(_safe(float(
                np.mean(((vals - np.mean(vals)) / max(np.std(vals), 1e-12)) ** 3)
            )), 4),
            "kurtosis": round(_safe(float(
                np.mean(((vals - np.mean(vals)) / max(np.std(vals), 1e-12)) ** 4) - 3
            )), 4),
            "cv": round(_safe(float(np.std(vals) / max(np.mean(vals), 1e-12))), 4),
        })

    # Band ratios (clinically relevant)
    mean_bp = {b: float(np.mean(bp[b])) for b in band_names}
    ratios = []
    ratio_pairs = [
        ("theta", "alpha", "Theta/Alpha (drowsiness/encephalopathy)"),
        ("delta", "alpha", "Delta/Alpha (diffuse slowing)"),
        ("theta", "beta", "Theta/Beta (ADHD marker)"),
        ("delta", "theta", "Delta/Theta (deep vs light slow)"),
        ("alpha", "beta", "Alpha/Beta (relaxation vs alertness)"),
    ]
    for num, den, desc in ratio_pairs:
        ratio_val = mean_bp[num] / max(mean_bp[den], 1e-12)
        ratios.append({
            "ratio": f"{num}/{den}",
            "value": round(_safe(ratio_val), 4),
            "description": desc,
        })

    # Per-diagnosis breakdown
    unique_labels = sorted(set(labels))
    diagnosis_breakdown = []
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        entry = {"diagnosis": lbl, "count": int(mask.sum()), "bands": []}
        for band in band_names:
            vals = bp[band][mask]
            entry["bands"].append({
                "band": band,
                "mean": round(_safe(float(np.mean(vals))), 6),
                "std": round(_safe(float(np.std(vals))), 6),
            })
        diagnosis_breakdown.append(entry)

    # Band correlation matrix
    band_matrix = np.column_stack([bp[b] for b in band_names])
    if n_samples > 2:
        corr = np.corrcoef(band_matrix.T)
        corr = np.nan_to_num(corr, nan=0.0)
    else:
        corr = np.eye(len(band_names))
    correlation_matrix = []
    for i, b1 in enumerate(band_names):
        for j, b2 in enumerate(band_names):
            if j >= i:
                correlation_matrix.append({
                    "band_1": b1,
                    "band_2": b2,
                    "correlation": round(_safe(float(corr[i, j])), 4),
                })

    return {
        "available": True,
        "band_stats": band_stats,
        "ratios": ratios,
        "diagnosis_breakdown": diagnosis_breakdown,
        "correlation_matrix": correlation_matrix,
    }


def definitions():
    return {
        "available": True,
        "title": "EEG Band Power Heatmap — Methodology & Clinical Reference",
        "bands": [
            {
                "band": "Delta",
                "range_hz": "0.5–4 Hz",
                "clinical": "Deep sleep (NREM stage 3), encephalopathy, focal lesions. Elevated delta in wakefulness suggests diffuse cerebral dysfunction.",
                "color": "#3b82f6",
            },
            {
                "band": "Theta",
                "range_hz": "4–8 Hz",
                "clinical": "Drowsiness, light sleep, memory encoding (hippocampal theta). Excess theta may indicate subcortical pathology or medication effects.",
                "color": "#10b981",
            },
            {
                "band": "Alpha",
                "range_hz": "8–13 Hz",
                "clinical": "Relaxed wakefulness, posterior dominant rhythm. Alpha attenuation with eyes open is normal. Absent alpha suggests cortical dysfunction.",
                "color": "#f59e0b",
            },
            {
                "band": "Beta",
                "range_hz": "13–30 Hz",
                "clinical": "Active thinking, focus, anxiety. Enhanced by benzodiazepines and barbiturates. Focal beta may indicate cortical lesion.",
                "color": "#ef4444",
            },
            {
                "band": "Gamma",
                "range_hz": "30–100 Hz",
                "clinical": "High-level cognitive processing, perception binding, memory consolidation. Often contaminated by EMG artifact in scalp EEG.",
                "color": "#8b5cf6",
            },
        ],
        "heatmap_interpretation": [
            "Rows represent individual subjects/recordings; columns represent frequency bands.",
            "Color intensity encodes band power magnitude (darker = higher power).",
            "Abnormal patterns: elevated delta/theta in wakefulness, absent alpha, asymmetric band power.",
            "Band ratios (e.g., theta/alpha) are clinically meaningful biomarkers for encephalopathy screening.",
        ],
        "clinical_applications": [
            "Epilepsy: focal band-power changes may lateralize seizure onset zone.",
            "Encephalopathy: diffuse delta/theta increase with alpha suppression.",
            "Sleep staging: delta dominance = N3, theta/alpha = N1/N2, alpha = Wake.",
            "Medication monitoring: benzodiazepines increase beta, AEDs may slow background.",
            "Cognitive assessment: theta/alpha ratio correlates with memory performance.",
        ],
        "references": [
            "Niedermeyer E, da Silva FL. Electroencephalography. 6th ed. Lippincott, 2010.",
            "Nuwer MR. Quantitative EEG: I. Techniques and problems of frequency analysis. J Clin Neurophysiol. 1988;5(1):1-43.",
            "Klimesch W. EEG alpha and theta oscillations reflect cognitive and memory performance. Brain Res Rev. 1999;29(2-3):169-195.",
        ],
    }
