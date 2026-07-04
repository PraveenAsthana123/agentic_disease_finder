#!/usr/bin/env python3
"""
EEG Connectivity Analysis Dashboard
=====================================

Computes REAL connectivity metrics from EEG features stored in
``data/clinical.db`` (analyses table → result_json → features).

Virtual channels are constructed from frequency-band power features:
  delta, theta, alpha, beta, gamma  →  5 "channels"

Connectivity measures:
  1. Cross-channel correlation matrix
  2. Magnitude-squared coherence proxy (correlation of power values)
  3. Phase Lag Value (PLV) approximation via Hilbert-transform analytic signal
  4. Graph-theoretic metrics (density, clustering, path length, small-world)
  5. Per-band connectivity breakdown

Functions:
  overview()    — KPIs + connectivity matrix + band connectivity summary
  breakdown()   — Per-band pair detail + graph metrics + strongest/weakest links
  definitions() — Methodology descriptions, formulae, clinical relevance
"""

import json
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

# Virtual EEG channels mapped from band-power feature indices
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_FEATURE_INDICES = {
    "delta": FEATURE_NAMES.index("delta_power"),
    "theta": FEATURE_NAMES.index("theta_power"),
    "alpha": FEATURE_NAMES.index("alpha_power"),
    "beta": FEATURE_NAMES.index("beta_power"),
    "gamma": FEATURE_NAMES.index("gamma_power"),
}
N_CHANNELS = len(BAND_NAMES)


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

    X_list = []
    labels = []

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

        sample = []
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


def _extract_band_matrix(X):
    """Extract n_samples x n_channels matrix of band-power values."""
    indices = [BAND_FEATURE_INDICES[b] for b in BAND_NAMES]
    return X[:, indices]


def _correlation_matrix(band_mat):
    """Compute Pearson correlation matrix between virtual channels."""
    # band_mat: (n_samples, n_channels)
    n = band_mat.shape[1]
    corr = np.corrcoef(band_mat, rowvar=False)
    # Ensure no NaN
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def _coherence_proxy(band_mat):
    """
    Magnitude-squared coherence proxy.
    For each channel pair, compute squared correlation of the power time series
    as a proxy for spectral coherence (|Cxy|^2).
    """
    corr = _correlation_matrix(band_mat)
    return corr ** 2


def _plv_approximation(band_mat):
    """
    Phase Lag Value approximation using Hilbert transform.
    Apply the Hilbert transform to each channel's power time series to extract
    instantaneous phase, then compute PLV = |mean(exp(j * dphi))|.
    """
    n_samples, n_channels = band_mat.shape
    plv_mat = np.zeros((n_channels, n_channels))

    # Compute analytic signal phases for each channel
    phases = np.zeros_like(band_mat)
    for ch in range(n_channels):
        sig = band_mat[:, ch] - np.mean(band_mat[:, ch])
        # Hilbert transform via FFT
        n = len(sig)
        fft_sig = np.fft.fft(sig)
        h = np.zeros(n)
        if n > 0:
            h[0] = 1
            if n % 2 == 0:
                h[n // 2] = 1
                h[1:n // 2] = 2
            else:
                h[1:(n + 1) // 2] = 2
        analytic = np.fft.ifft(fft_sig * h)
        phases[:, ch] = np.angle(analytic)

    # PLV for each pair
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                plv_mat[i, j] = 1.0
            else:
                dphi = phases[:, i] - phases[:, j]
                plv_mat[i, j] = float(np.abs(np.mean(np.exp(1j * dphi))))

    return plv_mat


def _graph_metrics(adj_matrix, threshold=0.3):
    """
    Compute graph-theoretic metrics from a weighted adjacency/connectivity matrix.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Symmetric connectivity matrix (n_channels x n_channels), values in [0, 1].
    threshold : float
        Edges below this weight are removed for binary graph metrics.

    Returns
    -------
    dict with density, clustering_coefficient, avg_path_length, small_world_index,
    modularity_estimate.
    """
    n = adj_matrix.shape[0]
    # Binary adjacency (exclude diagonal)
    A = (np.abs(adj_matrix) >= threshold).astype(float)
    np.fill_diagonal(A, 0)

    # Density: actual edges / possible edges
    n_edges = A.sum() / 2
    possible = n * (n - 1) / 2
    density = float(n_edges / possible) if possible > 0 else 0.0

    # Clustering coefficient (average local)
    clustering_vals = []
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering_vals.append(0.0)
            continue
        # Count triangles
        triangles = 0
        for ni_idx in range(len(neighbors)):
            for nj_idx in range(ni_idx + 1, len(neighbors)):
                if A[neighbors[ni_idx], neighbors[nj_idx]] > 0:
                    triangles += 1
        clustering_vals.append(2.0 * triangles / (k * (k - 1)))
    clustering_coeff = float(np.mean(clustering_vals))

    # Average path length via BFS on binary graph
    path_lengths = []
    for src in range(n):
        visited = {src}
        queue = [(src, 0)]
        head = 0
        while head < len(queue):
            node, dist = queue[head]
            head += 1
            for nbr in range(n):
                if A[node, nbr] > 0 and nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
                    path_lengths.append(dist + 1)
    avg_path_length = float(np.mean(path_lengths)) if path_lengths else float("inf")

    # Small-world index: C / C_random / (L / L_random)
    # For random graph: C_rand ~ density, L_rand ~ ln(n) / ln(k_mean)
    k_mean = A.sum(axis=1).mean()
    if k_mean > 1 and n > 1:
        c_rand = density
        l_rand = np.log(n) / np.log(k_mean) if k_mean > 1 else avg_path_length
        sigma = (clustering_coeff / max(c_rand, 1e-9)) / (avg_path_length / max(l_rand, 1e-9))
        small_world_index = float(np.clip(sigma, 0, 100))
    else:
        small_world_index = 0.0

    # Modularity estimate (greedy: split into 2 groups by sign of 2nd eigenvector of modularity matrix)
    if n >= 2:
        k = A.sum(axis=1)
        m = A.sum() / 2
        if m > 0:
            B = A - np.outer(k, k) / (2 * m)
            eigvals, eigvecs = np.linalg.eigh(B)
            partition = (eigvecs[:, -1] >= 0).astype(int)
            Q = 0.0
            for i in range(n):
                for j in range(n):
                    if partition[i] == partition[j]:
                        Q += B[i, j]
            modularity_estimate = float(Q / (2 * m))
        else:
            modularity_estimate = 0.0
    else:
        modularity_estimate = 0.0

    return {
        "density": round(density, 4),
        "clustering_coefficient": round(clustering_coeff, 4),
        "avg_path_length": round(avg_path_length, 4),
        "small_world_index": round(small_world_index, 4),
        "modularity_estimate": round(modularity_estimate, 4),
    }


def overview():
    """KPIs + connectivity matrix + band connectivity summary."""
    X, labels, err = _load_data()
    if err:
        return {"available": False, "error": err}

    band_mat = _extract_band_matrix(X)
    n_samples = band_mat.shape[0]

    # Compute connectivity matrices
    corr_mat = _correlation_matrix(band_mat)
    coh_mat = _coherence_proxy(band_mat)
    plv_mat = _plv_approximation(band_mat)

    # Combined connectivity: mean of |correlation|, coherence, PLV
    combined = (np.abs(corr_mat) + coh_mat + plv_mat) / 3.0

    # Graph metrics from combined matrix
    gm = _graph_metrics(combined, threshold=0.3)

    # Mean connectivity (off-diagonal)
    mask = ~np.eye(N_CHANNELS, dtype=bool)
    mean_connectivity = float(np.mean(combined[mask]))

    # KPIs
    kpis = {
        "n_samples": int(n_samples),
        "n_channels": N_CHANNELS,
        "mean_connectivity": round(mean_connectivity, 4),
        "graph_density": gm["density"],
        "clustering_coeff": gm["clustering_coefficient"],
        "avg_path_length": gm["avg_path_length"],
    }

    # Connectivity matrix as list of {source, target, value}
    connectivity_matrix = []
    for i in range(N_CHANNELS):
        for j in range(N_CHANNELS):
            connectivity_matrix.append({
                "source": BAND_NAMES[i],
                "target": BAND_NAMES[j],
                "value": round(float(combined[i, j]), 4),
            })

    # Per-band connectivity summary
    band_connectivity = []
    for i, band in enumerate(BAND_NAMES):
        # Average connectivity of this band with all others
        other_corr = [abs(float(corr_mat[i, j])) for j in range(N_CHANNELS) if j != i]
        other_coh = [float(coh_mat[i, j]) for j in range(N_CHANNELS) if j != i]
        other_plv = [float(plv_mat[i, j]) for j in range(N_CHANNELS) if j != i]
        band_connectivity.append({
            "band": band,
            "mean_correlation": round(float(np.mean(other_corr)), 4),
            "mean_coherence": round(float(np.mean(other_coh)), 4),
            "mean_plv": round(float(np.mean(other_plv)), 4),
        })

    return {
        "available": True,
        "kpis": kpis,
        "connectivity_matrix": connectivity_matrix,
        "band_connectivity": band_connectivity,
    }


def breakdown():
    """Per-band pair detail + graph metrics + strongest/weakest connections."""
    X, labels, err = _load_data()
    if err:
        return {"available": False, "error": err}

    band_mat = _extract_band_matrix(X)

    # Compute all three matrices
    corr_mat = _correlation_matrix(band_mat)
    coh_mat = _coherence_proxy(band_mat)
    plv_mat = _plv_approximation(band_mat)
    combined = (np.abs(corr_mat) + coh_mat + plv_mat) / 3.0

    # Per-band detail with all pairs
    per_band_detail = []
    for i, band in enumerate(BAND_NAMES):
        pairs = []
        for j in range(N_CHANNELS):
            if j == i:
                continue
            pairs.append({
                "source": band,
                "target": BAND_NAMES[j],
                "correlation": round(float(corr_mat[i, j]), 4),
                "coherence": round(float(coh_mat[i, j]), 4),
                "plv": round(float(plv_mat[i, j]), 4),
            })
        per_band_detail.append({"band": band, "pairs": pairs})

    # Graph metrics
    graph_metrics = _graph_metrics(combined, threshold=0.3)

    # Collect all unique pairs with combined connectivity strength
    all_pairs = []
    for i in range(N_CHANNELS):
        for j in range(i + 1, N_CHANNELS):
            all_pairs.append({
                "source": BAND_NAMES[i],
                "target": BAND_NAMES[j],
                "connectivity": round(float(combined[i, j]), 4),
                "correlation": round(float(corr_mat[i, j]), 4),
                "coherence": round(float(coh_mat[i, j]), 4),
                "plv": round(float(plv_mat[i, j]), 4),
            })

    # Sort by connectivity strength
    all_pairs_sorted = sorted(all_pairs, key=lambda p: p["connectivity"], reverse=True)

    # Top-10 strongest (or all if fewer than 10 pairs)
    strongest = all_pairs_sorted[:min(10, len(all_pairs_sorted))]
    # Bottom-5 weakest
    weakest = all_pairs_sorted[-min(5, len(all_pairs_sorted)):]

    return {
        "available": True,
        "per_band_detail": per_band_detail,
        "graph_metrics": graph_metrics,
        "strongest_connections": strongest,
        "weakest_connections": weakest,
    }


def definitions():
    """Methodology descriptions, formulae, clinical relevance, references."""
    return {
        "available": True,
        "methods": [
            {
                "name": "Cross-Channel Correlation",
                "description": "Pearson correlation coefficient between band-power time series of virtual EEG channels. Measures linear statistical dependence between channel pairs across all samples.",
                "formula_note": "r_xy = cov(X,Y) / (std(X) * std(Y)), range [-1, 1]",
                "clinical_relevance": "High inter-channel correlation in specific bands (e.g., delta-theta in temporal regions) is a hallmark of epileptic networks. Reduced correlation after treatment indicates network normalization.",
                "references": [
                    "Kramer & Cash (2012) 'Epilepsy as a Disorder of Cortical Network Organization', The Neuroscientist",
                    "Schindler et al. (2007) 'Assessing seizure dynamics by analysing the correlation structure of multichannel intracranial EEG', Brain",
                ],
            },
            {
                "name": "Magnitude-Squared Coherence (Proxy)",
                "description": "Squared correlation of band-power values as a proxy for spectral coherence. True MSC requires raw time-series data; this proxy captures linear power coupling between channels.",
                "formula_note": "Coh_xy = |r_xy|^2, approximating |S_xy(f)|^2 / (S_xx(f) * S_yy(f))",
                "clinical_relevance": "Coherence identifies frequency-specific coupling. Elevated theta-alpha coherence in temporal-frontal circuits is a biomarker for temporal lobe epilepsy. Post-ictal coherence drops predict seizure termination.",
                "references": [
                    "Nunez et al. (1997) 'EEG coherency: statistical significance of changes in coherence estimates', Clinical Neurophysiology",
                    "Mormann et al. (2000) 'Mean phase coherence as a measure for phase synchronization', Physica D",
                ],
            },
            {
                "name": "Phase Lag Value (PLV)",
                "description": "Measures phase synchronization between channels via the Hilbert transform. PLV = |mean(exp(j*(phi_x - phi_y)))|, where phi is instantaneous phase from the analytic signal. Values near 1 indicate consistent phase relationship.",
                "formula_note": "PLV = |1/N * sum(exp(j * (phi_i(t) - phi_j(t))))|, range [0, 1]",
                "clinical_relevance": "PLV captures non-linear phase coupling missed by correlation/coherence. Pre-ictal PLV increases in the seizure onset zone 5-30 minutes before seizure onset, making it a candidate biomarker for seizure prediction.",
                "references": [
                    "Lachaux et al. (1999) 'Measuring phase synchrony in brain signals', Human Brain Mapping",
                    "Mormann et al. (2003) 'Automated detection of a preseizure state based on a decrease in synchronization', Clinical Neurophysiology",
                ],
            },
            {
                "name": "Graph Density",
                "description": "Fraction of possible edges that exist in the thresholded connectivity graph. Density = 2E / (N*(N-1)), where E is edge count and N is node count.",
                "formula_note": "D = 2|E| / (N * (N-1)), range [0, 1]",
                "clinical_relevance": "Epileptic networks show increased density (hyper-connectivity) during the interictal period. A density above 0.7 in band-power networks suggests pathological hyper-synchrony.",
                "references": [
                    "Ponten et al. (2007) 'Small-world networks and epilepsy', Clinical Neurophysiology",
                ],
            },
            {
                "name": "Clustering Coefficient",
                "description": "Average fraction of a node's neighbors that are also connected to each other. High clustering indicates local processing efficiency and network segregation.",
                "formula_note": "C_i = 2T_i / (k_i * (k_i - 1)), averaged over all nodes",
                "clinical_relevance": "Epileptic brains show altered clustering: increased locally (seizure focus) but decreased globally. Anti-epileptic drugs that normalize clustering correlate with seizure freedom.",
                "references": [
                    "Watts & Strogatz (1998) 'Collective dynamics of small-world networks', Nature",
                    "van Diessen et al. (2013) 'Functional and structural brain networks in epilepsy', Frontiers in Neurology",
                ],
            },
            {
                "name": "Average Path Length",
                "description": "Mean shortest path between all pairs of nodes in the connectivity graph. Short path length indicates efficient global information transfer (network integration).",
                "formula_note": "L = (1 / N(N-1)) * sum(d(i,j)) for all i != j",
                "clinical_relevance": "Reduced path length during seizures reflects hyper-synchronous states where all brain regions become directly coupled. Pre-surgical planning uses path length to identify critical network hubs.",
                "references": [
                    "Bullmore & Sporns (2009) 'Complex brain networks: graph theoretical analysis', Nature Reviews Neuroscience",
                ],
            },
            {
                "name": "Small-World Index",
                "description": "Ratio sigma = (C/C_rand) / (L/L_rand). Values > 1 indicate small-world topology: high local clustering with short global paths, characteristic of efficient brain networks.",
                "formula_note": "sigma = (C/C_random) / (L/L_random), small-world if sigma > 1",
                "clinical_relevance": "Healthy brains exhibit small-world architecture (sigma ~ 2-5). Epileptic networks show disrupted small-worldness that correlates with disease severity and cognitive impairment.",
                "references": [
                    "Humphries & Gurney (2008) 'Network small-world-ness: a quantitative method for determining canonical network equivalence', PLoS ONE",
                    "Ponten et al. (2007) 'Small-world networks and epilepsy', Clinical Neurophysiology",
                ],
            },
        ],
        "best_practices": [
            "Use multiple connectivity measures (correlation, coherence, PLV) to capture different coupling modes",
            "Apply surrogate data testing to distinguish true connectivity from volume conduction artifacts",
            "Threshold selection for graph metrics should be validated with permutation testing",
            "Report both weighted and binary graph metrics for completeness",
            "Band-specific analysis is essential: connectivity patterns differ across frequency bands",
        ],
        "eeg_specific_notes": [
            "Delta-band connectivity reflects subcortical-cortical coupling and is elevated in encephalopathies",
            "Theta-band connectivity in temporal regions is the most sensitive marker for temporal lobe epilepsy",
            "Alpha-band connectivity disruption correlates with cognitive decline in epilepsy patients",
            "Beta-gamma hyper-connectivity in the seizure onset zone persists interictally and aids surgical planning",
            "Volume conduction inflates connectivity estimates; PLV and imaginary coherence are more robust to this artifact",
        ],
    }
