#!/usr/bin/env python3
"""
Feature Selection Dashboard — LASSO, RFE, PCA, SelectKBest, Boruta
===================================================================

Performs real feature selection on EEG features extracted from the
``analyses`` table in ``data/clinical.db``.

Each row's ``result_json`` column contains a JSON object with a ``features``
key (47 named EEG features) and a ``prediction`` key with ``predicted_label``
(Control or Epilepsy).

Functions:
  overview()    — KPIs + consensus selection, method summary, category rates
  breakdown()   — Full feature table with per-method selection status
  definitions() — Methods, strengths/weaknesses, clinical relevance

All data is REAL — queried from the database, never fabricated.
"""

import json
import os
import sqlite3
from itertools import combinations
from typing import Any, Dict, List

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


def _build_matrix(features_matrix: Dict[str, List[float]], labels: List[str]):
    """
    Build a clean numpy matrix (samples x features) and encoded labels.
    Imputes NaN with column median.
    Returns X (ndarray), y (ndarray), success (bool).
    """
    from sklearn.preprocessing import LabelEncoder

    n_samples = len(labels)
    X = np.zeros((n_samples, len(FEATURE_NAMES)))
    for j, fname in enumerate(FEATURE_NAMES):
        X[:, j] = np.array(features_matrix[fname], dtype=float)

    # Impute NaN with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            col[nan_mask] = median_val

    le = LabelEncoder()
    y = le.fit_transform(labels)

    return X, y


def _run_lasso(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Run LassoCV, return selected features and coefficients."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=min(5, len(y)), random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)

    coefs = lasso.coef_
    selected_mask = coefs != 0.0

    return {
        "alpha": float(lasso.alpha_),
        "coefs": coefs.tolist(),
        "selected_mask": selected_mask.tolist(),
        "n_selected": int(selected_mask.sum()),
    }


def _run_rfe(X: np.ndarray, y: np.ndarray, n_select: int = 15) -> Dict[str, Any]:
    """Run RFE with RandomForest, return rankings."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_select, step=1)
    rfe.fit(X, y)

    return {
        "selected_mask": rfe.support_.tolist(),
        "ranking": rfe.ranking_.tolist(),
        "n_selected": int(rfe.support_.sum()),
    }


def _run_selectkbest(X: np.ndarray, y: np.ndarray, n_select: int = 15) -> Dict[str, Any]:
    """Run SelectKBest with f_classif."""
    from sklearn.feature_selection import SelectKBest, f_classif

    skb = SelectKBest(score_func=f_classif, k=n_select)
    skb.fit(X, y)

    return {
        "selected_mask": skb.get_support().tolist(),
        "scores": skb.scores_.tolist(),
        "n_selected": n_select,
    }


def _run_pca(X: np.ndarray, n_select: int = 15) -> Dict[str, Any]:
    """Run PCA, select top features by abs loading on PC1."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(3, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    # Loadings: components_ shape is (n_components, n_features)
    pc1_loadings = np.abs(pca.components_[0])
    top_indices = np.argsort(pc1_loadings)[::-1][:n_select]
    selected_mask = np.zeros(X.shape[1], dtype=bool)
    selected_mask[top_indices] = True

    return {
        "selected_mask": selected_mask.tolist(),
        "pc1_loadings": pca.components_[0].tolist(),
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "n_components": n_components,
        "n_selected": n_select,
    }


def _run_boruta(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Simplified Boruta: create shadow features (shuffled), fit RF,
    compare real importances to max shadow importance.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(42)
    n_samples, n_features = X.shape

    # Create shadow features by shuffling each column independently
    X_shadow = X.copy()
    for j in range(n_features):
        rng.shuffle(X_shadow[:, j])

    X_combined = np.hstack([X, X_shadow])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_combined, y)

    importances = rf.feature_importances_
    real_importances = importances[:n_features]
    shadow_importances = importances[n_features:]

    max_shadow = shadow_importances.max()
    selected_mask = real_importances > max_shadow

    return {
        "selected_mask": selected_mask.tolist(),
        "importances": real_importances.tolist(),
        "max_shadow_importance": float(max_shadow),
        "n_selected": int(selected_mask.sum()),
    }


# ── Public API ────────────────────────────────────────────────────────

def overview() -> Dict[str, Any]:
    """
    Return KPIs + chart data for the Feature Selection Dashboard.

    KPIs: total_features, total_samples, methods_applied, selected_consensus
    Charts: top_selected, method_summary, category_selection_rate, consensus_distribution
    """
    data = _load_feature_data()
    if not data.get("available"):
        return {"available": False, "error": data.get("error", "No data available")}

    if data["count"] < 5:
        return {"available": False, "error": "Fewer than 5 samples available"}

    features_matrix = data["features_matrix"]
    labels = data["labels"]

    X, y = _build_matrix(features_matrix, labels)

    # Run all 5 methods
    lasso_res = _run_lasso(X, y)
    rfe_res = _run_rfe(X, y)
    skb_res = _run_selectkbest(X, y)
    pca_res = _run_pca(X)
    boruta_res = _run_boruta(X, y)

    # Compute consensus votes per feature
    votes = np.zeros(len(FEATURE_NAMES), dtype=int)
    for res in [lasso_res, rfe_res, skb_res, pca_res, boruta_res]:
        mask = np.array(res["selected_mask"], dtype=bool)
        votes += mask.astype(int)

    # Consensus: selected by >= 3/5 methods
    consensus_mask = votes >= 3
    selected_consensus = int(consensus_mask.sum())

    # Top selected (sorted by votes desc, then feature name)
    feature_votes = [(FEATURE_NAMES[i], int(votes[i])) for i in range(len(FEATURE_NAMES))]
    feature_votes.sort(key=lambda x: (-x[1], x[0]))
    top_selected = [
        {"feature": fv[0], "selection_count": fv[1], "category": _get_category(fv[0])}
        for fv in feature_votes[:10]
    ]

    # Method summary
    method_summary = [
        {"method": "LASSO", "n_selected": lasso_res["n_selected"], "type": "embedded"},
        {"method": "RFE", "n_selected": rfe_res["n_selected"], "type": "wrapper"},
        {"method": "SelectKBest", "n_selected": skb_res["n_selected"], "type": "filter"},
        {"method": "PCA", "n_selected": pca_res["n_selected"], "type": "dimensionality"},
        {"method": "Boruta", "n_selected": boruta_res["n_selected"], "type": "wrapper"},
    ]

    # Category selection rate
    category_selection_rate = []
    for cat, cat_feats in FEATURE_CATEGORIES.items():
        indices = [FEATURE_NAMES.index(f) for f in cat_feats]
        n_selected = int(sum(1 for i in indices if consensus_mask[i]))
        total = len(cat_feats)
        rate = round(n_selected / total, 4) if total > 0 else 0.0
        category_selection_rate.append({
            "category": cat,
            "total": total,
            "selected": n_selected,
            "rate": rate,
        })

    # Consensus distribution
    consensus_distribution = []
    for v in range(6):
        count = int((votes == v).sum())
        consensus_distribution.append({"votes": v, "count": count})

    return {
        "available": True,
        "total_features": len(FEATURE_NAMES),
        "total_samples": data["count"],
        "methods_applied": 5,
        "selected_consensus": selected_consensus,
        "top_selected": top_selected,
        "method_summary": method_summary,
        "category_selection_rate": category_selection_rate,
        "consensus_distribution": consensus_distribution,
    }


def breakdown() -> Dict[str, Any]:
    """
    Return detailed per-feature selection status across all 5 methods.

    - feature_table: all 47 features with per-method selection + scores
    - lasso_details, rfe_details, pca_details
    - method_agreement: pairwise Jaccard similarity
    """
    data = _load_feature_data()
    if not data.get("available"):
        return {"available": False, "error": data.get("error", "No data available")}

    if data["count"] < 5:
        return {"available": False, "error": "Fewer than 5 samples available"}

    features_matrix = data["features_matrix"]
    labels = data["labels"]

    X, y = _build_matrix(features_matrix, labels)

    # Run all 5 methods
    lasso_res = _run_lasso(X, y)
    rfe_res = _run_rfe(X, y)
    skb_res = _run_selectkbest(X, y)
    pca_res = _run_pca(X)
    boruta_res = _run_boruta(X, y)

    # Build feature table
    feature_table = []
    votes = np.zeros(len(FEATURE_NAMES), dtype=int)
    for res in [lasso_res, rfe_res, skb_res, pca_res, boruta_res]:
        mask = np.array(res["selected_mask"], dtype=bool)
        votes += mask.astype(int)

    for i, fname in enumerate(FEATURE_NAMES):
        row = {
            "feature": fname,
            "category": _get_category(fname),
            "lasso_selected": bool(lasso_res["selected_mask"][i]),
            "lasso_coef": round(float(lasso_res["coefs"][i]), 6),
            "rfe_selected": bool(rfe_res["selected_mask"][i]),
            "rfe_rank": int(rfe_res["ranking"][i]),
            "selectkbest_selected": bool(skb_res["selected_mask"][i]),
            "selectkbest_score": round(float(skb_res["scores"][i]), 4),
            "pca_loading": round(float(abs(pca_res["pc1_loadings"][i])), 6),
            "boruta_selected": bool(boruta_res["selected_mask"][i]),
            "boruta_importance": round(float(boruta_res["importances"][i]), 6),
            "consensus_votes": int(votes[i]),
            "consensus_selected": bool(votes[i] >= 3),
        }
        feature_table.append(row)

    # Sort by consensus_votes descending
    feature_table.sort(key=lambda x: (-x["consensus_votes"], x["feature"]))

    # LASSO details
    lasso_top = sorted(
        [(FEATURE_NAMES[i], abs(lasso_res["coefs"][i])) for i in range(len(FEATURE_NAMES)) if lasso_res["selected_mask"][i]],
        key=lambda x: -x[1]
    )
    lasso_details = {
        "alpha": round(lasso_res["alpha"], 6),
        "n_selected": lasso_res["n_selected"],
        "top_features": [{"feature": f, "abs_coef": round(c, 6)} for f, c in lasso_top[:15]],
    }

    # RFE details
    rfe_ranking = sorted(
        [(FEATURE_NAMES[i], int(rfe_res["ranking"][i])) for i in range(len(FEATURE_NAMES))],
        key=lambda x: x[1]
    )
    rfe_details = {
        "n_selected": rfe_res["n_selected"],
        "ranking": [{"feature": f, "rank": r} for f, r in rfe_ranking],
    }

    # PCA details
    pc1_loadings_sorted = sorted(
        [(FEATURE_NAMES[i], float(pca_res["pc1_loadings"][i])) for i in range(len(FEATURE_NAMES))],
        key=lambda x: -abs(x[1])
    )
    pca_details = {
        "n_components": pca_res["n_components"],
        "variance_explained": [round(v, 6) for v in pca_res["variance_explained"]],
        "top_loadings": [{"feature": f, "loading": round(l, 6)} for f, l in pc1_loadings_sorted[:15]],
    }

    # Method agreement: pairwise Jaccard similarity
    method_names = ["LASSO", "RFE", "SelectKBest", "PCA", "Boruta"]
    method_masks = [
        np.array(lasso_res["selected_mask"], dtype=bool),
        np.array(rfe_res["selected_mask"], dtype=bool),
        np.array(skb_res["selected_mask"], dtype=bool),
        np.array(pca_res["selected_mask"], dtype=bool),
        np.array(boruta_res["selected_mask"], dtype=bool),
    ]

    method_agreement = []
    for (i, name_a), (j, name_b) in combinations(enumerate(method_names), 2):
        intersection = (method_masks[i] & method_masks[j]).sum()
        union = (method_masks[i] | method_masks[j]).sum()
        jaccard = float(intersection / union) if union > 0 else 0.0
        method_agreement.append({
            "method_a": name_a,
            "method_b": name_b,
            "jaccard": round(jaccard, 4),
        })

    return {
        "available": True,
        "feature_table": feature_table,
        "lasso_details": lasso_details,
        "rfe_details": rfe_details,
        "pca_details": pca_details,
        "method_agreement": method_agreement,
    }


def definitions() -> Dict[str, Any]:
    """
    Return explanations of feature selection methods, their strengths/weaknesses,
    clinical relevance for EEG, and interpretation guidelines.
    """
    methods = {
        "LASSO": {
            "full_name": "Least Absolute Shrinkage and Selection Operator",
            "type": "embedded",
            "description": (
                "L1-regularized linear model that shrinks coefficients of irrelevant "
                "features to exactly zero. LassoCV selects the optimal regularization "
                "strength via cross-validation."
            ),
            "strengths": [
                "Simultaneous feature selection and model fitting",
                "Handles multicollinearity by selecting one from correlated groups",
                "Computationally efficient for high-dimensional data",
            ],
            "weaknesses": [
                "Assumes linear relationship between features and target",
                "May underperform with highly correlated features (selects one arbitrarily)",
                "Sensitive to feature scaling (requires standardization)",
            ],
            "eeg_relevance": (
                "Effective for identifying the minimal set of EEG features that linearly "
                "discriminate between diagnostic groups. Tends to select one representative "
                "from each correlated spectral band."
            ),
        },
        "RFE": {
            "full_name": "Recursive Feature Elimination",
            "type": "wrapper",
            "description": (
                "Iteratively removes the least important feature according to a base "
                "estimator (RandomForest) until the desired number of features is reached. "
                "Features are ranked by elimination order."
            ),
            "strengths": [
                "Model-agnostic wrapper approach",
                "Captures non-linear feature interactions via RandomForest",
                "Provides a complete ranking of all features",
            ],
            "weaknesses": [
                "Computationally expensive (refits model at each step)",
                "Greedy algorithm — may miss globally optimal subsets",
                "Results depend on the choice of base estimator",
            ],
            "eeg_relevance": (
                "Captures complex non-linear interactions between EEG features "
                "(e.g., spectral-complexity interactions during seizures). The RF base "
                "estimator naturally handles the heterogeneous scales of EEG features."
            ),
        },
        "SelectKBest": {
            "full_name": "Univariate Feature Selection (ANOVA F-test)",
            "type": "filter",
            "description": (
                "Selects the top K features based on univariate ANOVA F-statistics. "
                "Each feature is scored independently by its between-class variance "
                "relative to within-class variance."
            ),
            "strengths": [
                "Fast and computationally cheap",
                "Model-independent — no assumptions about classifier",
                "Easy to interpret (F-score = discriminative power)",
            ],
            "weaknesses": [
                "Ignores feature interactions and redundancy",
                "May select highly correlated features",
                "Assumes features are independent (rarely true for EEG)",
            ],
            "eeg_relevance": (
                "Identifies individual EEG features with the strongest univariate "
                "separation between Control and Epilepsy groups. Good first pass but "
                "may over-select correlated spectral features."
            ),
        },
        "PCA": {
            "full_name": "Principal Component Analysis",
            "type": "dimensionality",
            "description": (
                "Unsupervised dimensionality reduction that finds orthogonal directions "
                "of maximum variance. Features are ranked by their absolute loading on "
                "the first principal component (PC1)."
            ),
            "strengths": [
                "Unsupervised — not biased by labels",
                "Identifies latent structure in the feature space",
                "Decorrelates features naturally",
            ],
            "weaknesses": [
                "Directions of max variance may not be discriminative",
                "Linear method — misses non-linear structure",
                "Selects features contributing to variance, not classification",
            ],
            "eeg_relevance": (
                "Reveals the dominant axes of variation in EEG features. High PC1 "
                "loading features often reflect overall signal power (total_power, rms) "
                "which may or may not align with diagnostic differences."
            ),
        },
        "Boruta": {
            "full_name": "Boruta Feature Selection (simplified shadow-feature method)",
            "type": "wrapper",
            "description": (
                "Creates 'shadow' features by shuffling each original feature, fits a "
                "RandomForest on combined real + shadow features, and selects real "
                "features whose importance exceeds the maximum shadow importance. "
                "This is a simplified single-iteration version of the full Boruta algorithm."
            ),
            "strengths": [
                "All-relevant selection (not minimal-optimal)",
                "Automatic threshold via shadow features — no k to set",
                "Captures non-linear relationships",
            ],
            "weaknesses": [
                "Computationally expensive (double the features)",
                "Single iteration may miss borderline features",
                "Sensitive to RandomForest hyperparameters",
            ],
            "eeg_relevance": (
                "Identifies ALL features that carry genuine signal above noise level. "
                "Particularly valuable for EEG where many features are informative "
                "and we want to avoid discarding borderline biomarkers."
            ),
        },
    }

    interpretation_guidelines = {
        "consensus_voting": (
            "A feature selected by >= 3 out of 5 methods is considered a consensus "
            "selection. This multi-method agreement increases confidence that the "
            "feature carries genuine discriminative information rather than being "
            "an artifact of a single method's assumptions."
        ),
        "method_agreement_jaccard": (
            "Jaccard similarity between methods measures the overlap of their selected "
            "feature sets. High agreement (> 0.5) suggests methods converge on the same "
            "features. Low agreement (< 0.2) indicates different methods capture "
            "complementary aspects of the data."
        ),
        "category_selection_rate": (
            "The fraction of features from each category that pass consensus selection. "
            "High rates in a category suggest that category is broadly informative. "
            "For EEG epilepsy classification, Spectral and Complexity categories "
            "typically show high selection rates."
        ),
        "clinical_decision": (
            "Consensus-selected features form the recommended minimal feature set for "
            "clinical deployment. Fewer features reduce overfitting risk, improve "
            "interpretability, and decrease computation for real-time BCI/monitoring."
        ),
    }

    clinical_relevance = {
        "spectral_features": (
            "Spectral features (band powers, entropy) are the gold standard in clinical "
            "EEG. They directly map to clinical concepts: delta slowing, alpha suppression, "
            "beta activation, gamma high-frequency oscillations."
        ),
        "complexity_features": (
            "Complexity measures (entropy, Hurst, DFA, LZ) capture the non-linear "
            "dynamics of neural networks. Reduced complexity during seizures is a "
            "well-established biomarker."
        ),
        "time_domain_features": (
            "Time-domain features detect transient morphological changes: spikes, "
            "sharp waves, amplitude changes. Line length and zero-crossings are among "
            "the earliest automated seizure detection features."
        ),
        "minimal_feature_set": (
            "For real-time seizure detection, a minimal set of 10-15 consensus features "
            "typically achieves > 90%% of full-feature-set performance while enabling "
            "sub-second classification on embedded hardware."
        ),
    }

    return {
        "available": True,
        "methods": methods,
        "interpretation_guidelines": interpretation_guidelines,
        "clinical_relevance": clinical_relevance,
    }


# ── CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    print("=== Feature Selection Dashboard ===\n")

    print("--- overview() ---")
    ov = overview()
    if ov.get("available"):
        print(f"  Samples: {ov['total_samples']}")
        print(f"  Features: {ov['total_features']}")
        print(f"  Methods applied: {ov['methods_applied']}")
        print(f"  Consensus selected (>=3/5): {ov['selected_consensus']}")
        print(f"  Top selected: {[f['feature'] for f in ov['top_selected'][:5]]}")
        print(f"  Method summary:")
        for ms in ov["method_summary"]:
            print(f"    {ms['method']}: {ms['n_selected']} selected ({ms['type']})")
        print(f"  Consensus distribution: {ov['consensus_distribution']}")
    else:
        print(f"  Not available: {ov.get('error')}")

    print("\n--- breakdown() ---")
    bd = breakdown()
    if bd.get("available"):
        print(f"  Feature table rows: {len(bd['feature_table'])}")
        print(f"  LASSO alpha: {bd['lasso_details']['alpha']}, selected: {bd['lasso_details']['n_selected']}")
        print(f"  RFE selected: {bd['rfe_details']['n_selected']}")
        print(f"  PCA components: {bd['pca_details']['n_components']}, variance: {bd['pca_details']['variance_explained']}")
        print(f"  Method agreement pairs: {len(bd['method_agreement'])}")
        top3 = bd["feature_table"][:3]
        for t in top3:
            print(f"    {t['feature']}: {t['consensus_votes']}/5 votes")
    else:
        print(f"  Not available: {bd.get('error')}")

    print("\n--- definitions() ---")
    defs = definitions()
    print(f"  Methods: {list(defs['methods'].keys())}")
    print(f"  Guidelines: {list(defs['interpretation_guidelines'].keys())}")
    print(f"  Clinical: {list(defs['clinical_relevance'].keys())}")
