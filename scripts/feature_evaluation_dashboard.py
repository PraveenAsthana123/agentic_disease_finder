#!/usr/bin/env python3
"""
Feature Evaluation Dashboard — ANOVA F-test & Feature Importance
================================================================

Performs real ANOVA F-test and feature importance ranking on EEG features
extracted from the ``analyses`` table in ``data/clinical.db``.

Each row's ``result_json`` column contains a JSON object with a ``features``
key (47 named EEG features) and a ``prediction`` key with ``predicted_label``
(Control or Epilepsy).

Functions:
  overview()    — KPIs + chart data (top features, categories, class dist)
  breakdown()   — Full feature table, top features, correlation pairs
  definitions() — Methods, categories, clinical relevance explanations

All data is REAL — queried from the database, never fabricated.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")

# ── Feature names (47) ────────────────────────────────────────────────
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

# ── Feature categories ────────────────────────────────────────────────
FEATURE_CATEGORIES = {
    "Spectral": [
        "delta_power", "theta_power", "alpha_power", "beta_power",
        "gamma_power", "total_power", "dominant_freq", "spectral_entropy",
        "psd_std", "psd_mean", "psd_median", "psd_q10", "psd_q90",
        "peak_ratio", "spectral_flatness", "spectral_centroid",
        "spectral_bandwidth", "spectral_rolloff",
    ],
    "Statistical": [
        "mean", "std", "var", "min", "max", "median", "ptp", "skewness",
        "kurtosis", "q25", "q75", "rms", "mav",
    ],
    "Complexity": [
        "approx_entropy", "sample_entropy", "hurst_exponent", "dfa_alpha",
        "lz_complexity",
    ],
    "Time-domain": [
        "line_length", "zero_crossings", "mean_abs_diff", "std_diff",
        "max_diff", "slope_changes", "trend", "crest_factor", "autocorr",
    ],
    "Hjorth": [
        "hjorth_mobility", "hjorth_complexity",
    ],
}

# Reverse lookup: feature -> category
_FEATURE_TO_CATEGORY = {}
for cat, feats in FEATURE_CATEGORIES.items():
    for f in feats:
        _FEATURE_TO_CATEGORY[f] = cat


# ── Helpers ───────────────────────────────────────────────────────────

def _get_category(feature: str) -> str:
    return _FEATURE_TO_CATEGORY.get(feature, "Unknown")


def _load_feature_data() -> Dict[str, Any]:
    """
    Query analyses table, parse result_json, return structured data.
    Returns dict with keys: features_matrix (dict of feature->list of values),
    labels (list of predicted_label strings), diseases (set), count.
    """
    if not os.path.exists(_DB_PATH):
        return {"available": False, "error": "Database not found at " + _DB_PATH}

    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute("SELECT result_json FROM analyses WHERE result_json IS NOT NULL")
    except sqlite3.OperationalError:
        conn.close()
        return {"available": False, "error": "Table 'analyses' not found in database"}

    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"available": False, "error": "No analyses with result_json found"}

    features_matrix: Dict[str, List[float]] = {f: [] for f in FEATURE_NAMES}
    labels: List[str] = []
    diseases: set = set()

    for row in rows:
        try:
            data = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        feats = data.get("features")
        prediction = data.get("prediction", {})
        label = prediction.get("predicted_label") if isinstance(prediction, dict) else None

        if not feats or not isinstance(feats, dict):
            continue
        if not label:
            continue

        labels.append(label)

        # Track disease from prediction
        diseases.add(label)

        for fname in FEATURE_NAMES:
            val = feats.get(fname)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                features_matrix[fname].append(float(val))
            else:
                features_matrix[fname].append(np.nan)

    return {
        "available": True,
        "features_matrix": features_matrix,
        "labels": labels,
        "diseases": diseases,
        "count": len(labels),
    }


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return 0.0
    mean_a, mean_b = np.nanmean(group_a), np.nanmean(group_b)
    var_a, var_b = np.nanvar(group_a, ddof=1), np.nanvar(group_b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


def _run_anova(features_matrix: Dict[str, List[float]], labels: List[str]) -> List[Dict[str, Any]]:
    """
    Run one-way ANOVA F-test for each feature across classes.
    Returns list of dicts with feature, f_score, p_value, significant, category,
    mean per class, effect_size.
    """
    from scipy.stats import f_oneway

    unique_labels = sorted(set(labels))
    labels_arr = np.array(labels)
    results = []

    for fname in FEATURE_NAMES:
        vals = np.array(features_matrix[fname], dtype=float)

        # Split by class
        groups = {}
        for lbl in unique_labels:
            mask = labels_arr == lbl
            group_vals = vals[mask]
            # Remove NaNs
            group_vals = group_vals[~np.isnan(group_vals)]
            groups[lbl] = group_vals

        # Compute means per class
        means = {}
        for lbl, gv in groups.items():
            means[lbl] = float(np.nanmean(gv)) if len(gv) > 0 else 0.0

        # ANOVA requires 2+ groups with 2+ samples each
        valid_groups = [g for g in groups.values() if len(g) >= 2]

        if len(valid_groups) >= 2:
            f_stat, p_val = f_oneway(*valid_groups)
            if np.isnan(f_stat):
                f_stat = 0.0
            if np.isnan(p_val):
                p_val = 1.0
        else:
            f_stat = 0.0
            p_val = 1.0

        # Effect size (Cohen's d for two-class case)
        effect_size = 0.0
        if len(unique_labels) == 2 and all(len(groups[l]) >= 2 for l in unique_labels):
            effect_size = _cohens_d(groups[unique_labels[0]], groups[unique_labels[1]])

        results.append({
            "feature": fname,
            "category": _get_category(fname),
            "f_score": round(float(f_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": p_val < 0.05,
            "mean_control": round(means.get("Control", 0.0), 6),
            "mean_epilepsy": round(means.get("Epilepsy", 0.0), 6),
            "effect_size": round(float(abs(effect_size)), 4),
        })

    # Rank by f_score descending
    results.sort(key=lambda x: x["f_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


# ── Public API ────────────────────────────────────────────────────────

def overview() -> Dict[str, Any]:
    """
    Return KPIs + chart data for the Feature Evaluation Dashboard.

    KPIs: total_analyses, total_features, diseases_covered,
          top_discriminative_feature, mean_f_score, features_above_significance
    Charts: anova_chart (top 15), category_chart, class_distribution
    """
    data = _load_feature_data()
    if not data.get("available"):
        return {"available": False, "error": data.get("error", "No data available")}

    features_matrix = data["features_matrix"]
    labels = data["labels"]
    diseases = data["diseases"]

    # Run ANOVA
    anova_results = _run_anova(features_matrix, labels)

    # KPIs
    significant_count = sum(1 for r in anova_results if r["significant"])
    mean_f = float(np.mean([r["f_score"] for r in anova_results])) if anova_results else 0.0
    top_feature = anova_results[0]["feature"] if anova_results else "N/A"

    kpis = {
        "total_analyses": data["count"],
        "total_features": len(FEATURE_NAMES),
        "diseases_covered": len(diseases),
        "top_discriminative_feature": top_feature,
        "mean_f_score": round(mean_f, 4),
        "features_above_significance": significant_count,
    }

    # ANOVA chart: top 15
    anova_chart = [
        {
            "feature": r["feature"],
            "f_score": r["f_score"],
            "p_value": r["p_value"],
            "significant": r["significant"],
        }
        for r in anova_results[:15]
    ]

    # Category chart
    cat_scores: Dict[str, List[float]] = {}
    cat_counts: Dict[str, int] = {}
    for r in anova_results:
        cat = r["category"]
        cat_scores.setdefault(cat, []).append(r["f_score"])
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    category_chart = [
        {
            "category": cat,
            "count": cat_counts[cat],
            "avg_f_score": round(float(np.mean(cat_scores[cat])), 4),
        }
        for cat in sorted(cat_counts.keys())
    ]

    # Class distribution
    from collections import Counter
    label_counts = Counter(labels)
    class_distribution = [
        {"label": lbl, "count": cnt}
        for lbl, cnt in sorted(label_counts.items())
    ]

    return {
        "available": True,
        "kpis": kpis,
        "anova_chart": anova_chart,
        "category_chart": category_chart,
        "class_distribution": class_distribution,
    }


def breakdown() -> Dict[str, Any]:
    """
    Return detailed tables for the Feature Evaluation Dashboard.

    - feature_table: all 47 features with full stats
    - top_features: top 10 by f_score
    - correlation_pairs: top 10 most correlated feature pairs
    """
    data = _load_feature_data()
    if not data.get("available"):
        return {"available": False, "error": data.get("error", "No data available")}

    features_matrix = data["features_matrix"]
    labels = data["labels"]

    # Run ANOVA
    anova_results = _run_anova(features_matrix, labels)

    # Feature table (all 47)
    feature_table = anova_results  # already has all needed fields

    # Top 10
    top_features = anova_results[:10]

    # Correlation pairs — compute pairwise Pearson correlations
    # Build matrix of features (samples x features), skip NaN rows
    n_samples = len(labels)
    mat = np.zeros((n_samples, len(FEATURE_NAMES)))
    for j, fname in enumerate(FEATURE_NAMES):
        mat[:, j] = np.array(features_matrix[fname], dtype=float)

    # Remove rows with any NaN
    valid_mask = ~np.isnan(mat).any(axis=1)
    mat_clean = mat[valid_mask]

    correlation_pairs: List[Dict[str, Any]] = []
    if mat_clean.shape[0] >= 3:
        # Compute correlation matrix
        corr_matrix = np.corrcoef(mat_clean.T)  # (47 x 47)

        # Extract top 10 pairs (upper triangle, exclude diagonal)
        pairs = []
        n_feats = len(FEATURE_NAMES)
        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                c = corr_matrix[i, j]
                if not np.isnan(c):
                    pairs.append((i, j, abs(c)))

        pairs.sort(key=lambda x: x[2], reverse=True)
        for i, j, c in pairs[:10]:
            correlation_pairs.append({
                "feature_a": FEATURE_NAMES[i],
                "feature_b": FEATURE_NAMES[j],
                "correlation": round(float(c), 4),
            })

    return {
        "available": True,
        "feature_table": feature_table,
        "top_features": top_features,
        "correlation_pairs": correlation_pairs,
    }


def definitions() -> Dict[str, Any]:
    """
    Return explanations of methods, feature categories, and clinical relevance.
    """
    feature_categories = {
        "Spectral": (
            "Features derived from the frequency domain via power spectral density (PSD). "
            "Includes band powers (delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, "
            "beta 13-30 Hz, gamma 30-100 Hz), spectral shape descriptors, and "
            "PSD distribution statistics."
        ),
        "Statistical": (
            "Basic descriptive statistics of the time-domain EEG signal amplitude. "
            "Captures central tendency, spread, shape of the distribution, and "
            "signal magnitude."
        ),
        "Complexity": (
            "Nonlinear complexity and regularity measures. These capture the "
            "predictability and information content of the EEG signal, related to "
            "underlying neural dynamics and cortical information processing."
        ),
        "Time-domain": (
            "Temporal structure features capturing signal morphology, rate of change, "
            "zero-crossings, and trend. Sensitive to transient events such as spikes "
            "and sharp waves."
        ),
        "Hjorth": (
            "Hjorth parameters (activity, mobility, complexity) characterize "
            "signal variance, mean frequency, and frequency spread respectively. "
            "Computationally efficient descriptors of EEG dynamics."
        ),
    }

    methods = {
        "ANOVA_F_test": (
            "One-way Analysis of Variance (ANOVA) F-test (scipy.stats.f_oneway). "
            "Tests whether the means of a feature differ significantly across "
            "diagnostic groups (Control vs Epilepsy). A higher F-score indicates "
            "greater between-group variance relative to within-group variance."
        ),
        "Cohens_d": (
            "Cohen's d effect size measures the standardized difference between "
            "two group means. Values: |d| < 0.2 negligible, 0.2-0.5 small, "
            "0.5-0.8 medium, > 0.8 large."
        ),
        "significance_threshold": (
            "Alpha = 0.05. Features with p-value < 0.05 are considered "
            "statistically significant discriminators between groups. "
            "No multiple-comparison correction is applied in this exploratory analysis."
        ),
    }

    clinical_relevance = {
        "spectral_entropy": (
            "Lower spectral entropy during seizures reflects rhythmic, "
            "narrowband activity. Biomarker for seizure detection."
        ),
        "delta_power": (
            "Elevated delta power indicates cortical slowing, seen post-ictally "
            "and in encephalopathy. Key marker for focal abnormalities."
        ),
        "theta_power": (
            "Increased theta correlates with temporal lobe epilepsy and "
            "drowsiness. Useful for lateralization."
        ),
        "alpha_power": (
            "Alpha suppression (low alpha) is seen in epileptic foci. "
            "Posterior dominant rhythm attenuation is a clinical sign."
        ),
        "beta_power": (
            "Beta enhancement can indicate cortical irritability. "
            "Medication effects (benzodiazepines) also increase beta."
        ),
        "gamma_power": (
            "High-frequency oscillations in gamma range are associated with "
            "seizure onset zones. Important for surgical planning."
        ),
        "hjorth_complexity": (
            "Increased Hjorth complexity during seizures reflects frequency "
            "content changes. Used in real-time seizure detectors."
        ),
        "approx_entropy": (
            "Reduced approximate entropy indicates increased signal regularity, "
            "characteristic of seizure activity."
        ),
        "sample_entropy": (
            "Similar to ApEn but less dependent on data length. Decreased "
            "SampEn is a reliable ictal biomarker."
        ),
        "line_length": (
            "Increased line length captures high-frequency content and amplitude "
            "changes during seizures. One of the earliest automated detection features."
        ),
        "hurst_exponent": (
            "H > 0.5 indicates persistent (long-memory) signal; H < 0.5 anti-persistent. "
            "Epileptic EEG tends toward lower Hurst values."
        ),
        "dfa_alpha": (
            "Detrended Fluctuation Analysis scaling exponent. Changes in DFA-alpha "
            "track transitions between interictal and ictal states."
        ),
        "lz_complexity": (
            "Lempel-Ziv complexity quantifies signal randomness. Seizures produce "
            "more repetitive patterns, lowering LZ complexity."
        ),
    }

    return {
        "available": True,
        "feature_categories": feature_categories,
        "methods": methods,
        "clinical_relevance": clinical_relevance,
    }


# ── CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    print("=== Feature Evaluation Dashboard ===\n")

    print("--- overview() ---")
    ov = overview()
    if ov.get("available"):
        print(f"  Analyses: {ov['kpis']['total_analyses']}")
        print(f"  Features: {ov['kpis']['total_features']}")
        print(f"  Top feature: {ov['kpis']['top_discriminative_feature']}")
        print(f"  Mean F-score: {ov['kpis']['mean_f_score']}")
        print(f"  Significant: {ov['kpis']['features_above_significance']}/47")
        print(f"  Classes: {ov['class_distribution']}")
    else:
        print(f"  Not available: {ov.get('error')}")

    print("\n--- breakdown() ---")
    bd = breakdown()
    if bd.get("available"):
        print(f"  Feature table rows: {len(bd['feature_table'])}")
        print(f"  Top features: {[f['feature'] for f in bd['top_features']]}")
        print(f"  Correlation pairs: {len(bd['correlation_pairs'])}")
    else:
        print(f"  Not available: {bd.get('error')}")

    print("\n--- definitions() ---")
    defs = definitions()
    print(f"  Categories: {list(defs['feature_categories'].keys())}")
    print(f"  Methods: {list(defs['methods'].keys())}")
    print(f"  Clinical biomarkers: {len(defs['clinical_relevance'])} features mapped")
