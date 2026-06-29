#!/usr/bin/env python3
"""
AIF360 Bias Detection Dashboard — real IBM AI Fairness 360 on EEG classification
=================================================================================

Extracts real EEG spectral/entropy/Hjorth features from CHB-MIT EDF recordings
via MNE, trains a small sklearn RandomForestClassifier, then runs comprehensive
AIF360 bias detection and mitigation analysis.

Functions provided:
  - aif360_overview       — dataset bias metrics, classification fairness metrics,
                            Reweighing mitigation before/after comparison
  - aif360_groups         — per-group (privileged vs unprivileged) breakdown with
                            base rates, positive prediction rates, confusion matrices
  - aif360_definitions    — metric definitions and clinical relevance (static)

Clinical relevance:
  Fairness in EEG-AI systems is critical because patient demographics (age, sex,
  ethnicity) can influence baseline EEG characteristics — alpha peak frequency
  decreases with age, skull thickness varies across populations affecting signal
  amplitude, and medication regimens differ by demographic group.  If a seizure
  detection model is trained predominantly on one demographic, it may exhibit
  disparate performance on others, leading to missed seizures in under-represented
  groups or excessive false alarms in others.  AIF360 provides mathematically
  rigorous bias metrics (disparate impact, equal opportunity, average odds) and
  mitigation algorithms (Reweighing, prejudice remover, calibrated equalized odds)
  that align with FDA guidance on algorithmic fairness in SaMD, the EU AI Act's
  non-discrimination requirements for high-risk AI, and NICE evidence standards
  for equitable digital health technologies.  Detecting and mitigating bias before
  clinical deployment ensures that EEG-AI seizure detection provides equitable
  diagnostic accuracy across all patient populations.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure user-local site-packages (where pip --break-system-packages installs)
# is reachable even inside a venv that doesn't inherit system packages.
_USER_SP = Path.home() / ".local/lib/python3.12/site-packages"
if _USER_SP.is_dir() and str(_USER_SP) not in sys.path:
    sys.path.insert(0, str(_USER_SP))

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

# EEG band definitions
_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


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


# ── feature extraction ──────────────────────────────────────────────

def _extract_features(file=None, seconds=10.0):
    """Extract real EEG features from EDF for AIF360 fairness analysis.

    Returns (X, y, feature_names, metadata) where X is a feature matrix,
    y is synthetic binary labels derived from theta/alpha power ratio,
    feature_names lists the columns, and metadata holds file info.
    """
    import mne
    mne.set_log_level("ERROR")

    if file:
        p = ROOT / file if not Path(file).is_absolute() else Path(file)
    else:
        p = _find_edf()
    if not p or not p.exists():
        return None, None, None, None

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names[:8]  # limit to first 8 channels
    raw.pick(ch_names)

    epoch_len = 2.0
    n_samples_epoch = int(epoch_len * sfreq)
    total_samples = min(int(seconds * sfreq), raw.n_times)
    data = raw.get_data(start=0, stop=total_samples)

    n_epochs = max(total_samples // n_samples_epoch, 1)

    from scipy.signal import welch

    # Feature names: per channel — 5 band powers + entropy + 3 Hjorth params
    feature_names = []
    for ch in ch_names:
        for band in _BANDS:
            feature_names.append(f"{ch}_{band}_power")
        feature_names.append(f"{ch}_shannon_entropy")
        feature_names.append(f"{ch}_hjorth_activity")
        feature_names.append(f"{ch}_hjorth_mobility")
        feature_names.append(f"{ch}_hjorth_complexity")

    n_features = len(feature_names)
    X = np.zeros((n_epochs, n_features), dtype=np.float32)

    for ep in range(n_epochs):
        start = ep * n_samples_epoch
        end = start + n_samples_epoch
        seg = data[:, start:end]

        idx = 0
        for ci in range(len(ch_names)):
            ch_data = seg[ci]
            freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(256, n_samples_epoch))

            # Band powers
            for band, (lo, hi) in _BANDS.items():
                mask = (freqs >= lo) & (freqs <= hi)
                X[ep, idx] = np.log1p(np.sum(psd[mask]))
                idx += 1

            # Shannon entropy on normalised PSD
            psd_norm = psd / (np.sum(psd) + 1e-12)
            X[ep, idx] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            idx += 1

            # Hjorth activity (variance)
            activity = np.var(ch_data)
            X[ep, idx] = activity
            idx += 1

            # Hjorth mobility
            diff1 = np.diff(ch_data)
            mobility = np.sqrt(np.var(diff1) / (activity + 1e-12))
            X[ep, idx] = mobility
            idx += 1

            # Hjorth complexity
            diff2 = np.diff(diff1)
            mobility_diff = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12))
            X[ep, idx] = mobility_diff / (mobility + 1e-12)
            idx += 1

    # Synthetic binary labels: theta/alpha ratio above median → abnormal (1)
    theta_cols = [i for i, n in enumerate(feature_names) if "theta_power" in n]
    alpha_cols = [i for i, n in enumerate(feature_names) if "alpha_power" in n]
    theta_mean = np.mean(X[:, theta_cols], axis=1)
    alpha_mean = np.mean(X[:, alpha_cols], axis=1)
    ratio = theta_mean / (alpha_mean + 1e-12)
    threshold = np.median(ratio)
    y = (ratio > threshold).astype(int)

    meta = {
        "file": str(p.name),
        "sfreq": float(sfreq),
        "n_channels": len(ch_names),
        "n_epochs": int(n_epochs),
        "n_features": n_features,
        "epoch_duration_s": epoch_len,
        "seconds_analyzed": float(n_epochs * epoch_len),
    }
    return X, y, feature_names, meta


# ── JSON helper ─────────────────────────────────────────────────────

def _json_clean(obj):
    """Make numpy/pandas types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return round(v, 6)
    if isinstance(obj, np.ndarray):
        return [_json_clean(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    # Handle pandas types
    try:
        import pandas as pd
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, (pd.Series,)):
            return obj.tolist()
        if isinstance(obj, (pd.DataFrame,)):
            return obj.to_dict(orient="records")
    except ImportError:
        pass
    return obj


# ── protected attribute creation ────────────────────────────────────

def _create_protected_attribute(X, feature_names):
    """Create a synthetic 'age_group' protected attribute from EEG alpha power.

    Uses the first channel's alpha power to split the population into two
    groups at the median.  This simulates demographic stratification (e.g.,
    young=0 vs old=1), since alpha peak frequency and amplitude naturally
    decrease with age — a well-documented EEG phenomenon.  In a real clinical
    setting, actual patient demographics would replace this synthetic proxy.

    Returns:
        protected (np.ndarray): Binary array of shape (n_samples,), 0 or 1.
        attr_name (str): Name of the protected attribute column.
    """
    # Find first channel's alpha power column
    alpha_col_idx = None
    for i, name in enumerate(feature_names):
        if "alpha_power" in name:
            alpha_col_idx = i
            break

    if alpha_col_idx is None:
        # Fallback: use first feature
        alpha_col_idx = 0

    alpha_vals = X[:, alpha_col_idx]
    median_val = np.median(alpha_vals)
    protected = (alpha_vals >= median_val).astype(int)
    return protected, "age_group"


# ── model training ──────────────────────────────────────────────────

def _train_rf(X, y):
    """Train a RandomForestClassifier and return the model + predictions.

    Returns (model, y_pred, y_pred_proba).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    return model, scaler, y_pred


def _train_rf_weighted(X, y, sample_weights, scaler):
    """Retrain RF with sample weights from Reweighing mitigation.

    Returns (model, y_pred).
    """
    from sklearn.ensemble import RandomForestClassifier

    X_scaled = scaler.transform(X)

    model = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(X_scaled, y, sample_weight=sample_weights)

    y_pred = model.predict(X_scaled)
    return model, y_pred


# ── AIF360 helpers ──────────────────────────────────────────────────

def _build_aif360_dataset(X, y, protected, feature_names, attr_name="age_group"):
    """Wrap data into an AIF360 BinaryLabelDataset.

    Returns the BinaryLabelDataset instance.
    """
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y.astype(float)
    df[attr_name] = protected.astype(float)

    dataset = BinaryLabelDataset(
        favorable_label=1.0,
        unfavorable_label=0.0,
        df=df,
        label_names=["label"],
        protected_attribute_names=[attr_name],
        privileged_protected_attributes=[[1.0]],
        unprivileged_protected_attributes=[[0.0]],
    )
    return dataset


def _compute_dataset_metrics(dataset, attr_name="age_group"):
    """Compute dataset-level bias metrics using BinaryLabelDatasetMetric."""
    from aif360.metrics import BinaryLabelDatasetMetric

    privileged = [{attr_name: 1.0}]
    unprivileged = [{attr_name: 0.0}]

    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=unprivileged,
        privileged_groups=privileged,
    )

    consistency_vals = metric.consistency()
    mean_consistency = float(np.mean(consistency_vals)) if len(consistency_vals) > 0 else 0.0

    return {
        "disparate_impact": float(metric.disparate_impact()),
        "statistical_parity_difference": float(metric.statistical_parity_difference()),
        "consistency": mean_consistency,
        "num_positives_privileged": int(metric.num_positives(privileged=True)),
        "num_positives_unprivileged": int(metric.num_positives(privileged=False)),
        "num_negatives_privileged": int(metric.num_negatives(privileged=True)),
        "num_negatives_unprivileged": int(metric.num_negatives(privileged=False)),
        "base_rate_privileged": float(metric.base_rate(privileged=True)),
        "base_rate_unprivileged": float(metric.base_rate(privileged=False)),
    }


def _compute_classification_metrics(dataset_true, dataset_pred, attr_name="age_group"):
    """Compute classification-level fairness metrics using ClassificationMetric."""
    from aif360.metrics import ClassificationMetric

    privileged = [{attr_name: 1.0}]
    unprivileged = [{attr_name: 0.0}]

    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged,
        privileged_groups=privileged,
    )

    return {
        "equal_opportunity_difference": float(metric.equal_opportunity_difference()),
        "average_odds_difference": float(metric.average_odds_difference()),
        "theil_index": float(metric.theil_index()),
        "disparate_impact": float(metric.disparate_impact()),
        "false_positive_rate_difference": float(metric.false_positive_rate_difference()),
        "false_negative_rate_difference": float(metric.false_negative_rate_difference()),
        "accuracy": float(metric.accuracy()),
        "precision": float(metric.precision()),
        "recall": float(metric.recall()),
    }


def _apply_reweighing(dataset, attr_name="age_group"):
    """Apply AIF360 Reweighing preprocessing mitigation.

    Returns (reweighed_dataset, sample_weights).
    """
    from aif360.algorithms.preprocessing import Reweighing

    privileged = [{attr_name: 1.0}]
    unprivileged = [{attr_name: 0.0}]

    rw = Reweighing(
        unprivileged_groups=unprivileged,
        privileged_groups=privileged,
    )
    dataset_rw = rw.fit_transform(dataset)
    sample_weights = dataset_rw.instance_weights

    return dataset_rw, sample_weights


# ── public API ──────────────────────────────────────────────────────

def aif360_overview(file=None, seconds=10.0) -> Dict[str, Any]:
    """Run real AIF360 bias detection and mitigation on EEG data.

    Returns a dict with:
      - dataset_metrics: disparate impact, statistical parity, consistency
      - classification_metrics: equal opportunity, average odds, Theil index, etc.
      - mitigation_results: before/after Reweighing comparison
      - metadata: file info, library version, model details
    """
    try:
        import aif360
        aif360_version = str(aif360.__version__) if hasattr(aif360, "__version__") else "installed"
    except ImportError:
        return {"available": False, "error": "aif360 not installed. Install with: pip install aif360"}

    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False, "error": "pandas or sklearn not installed."}

    try:
        result = _extract_features(file, seconds)
        if result[0] is None:
            return {"available": False, "error": "No EDF file found. Place .edf files in data/real_eeg/"}

        X, y, feature_names, meta = result

        # Create protected attribute from first channel alpha power
        protected, attr_name = _create_protected_attribute(X, feature_names)

        # Build AIF360 dataset
        dataset = _build_aif360_dataset(X, y, protected, feature_names, attr_name)

        # ── Dataset-level bias metrics ──
        dataset_metrics = _compute_dataset_metrics(dataset, attr_name)

        # ── Train model and get predictions ──
        model, scaler, y_pred = _train_rf(X, y)

        # Build predicted dataset for ClassificationMetric
        dataset_pred = dataset.copy(deepcopy=True)
        dataset_pred.labels = y_pred.reshape(-1, 1).astype(float)

        # ── Classification-level fairness metrics (BEFORE mitigation) ──
        classification_metrics_before = _compute_classification_metrics(
            dataset, dataset_pred, attr_name
        )

        # ── Apply Reweighing mitigation ──
        dataset_rw, sample_weights = _apply_reweighing(dataset, attr_name)

        # Retrain with sample weights
        model_rw, y_pred_rw = _train_rf_weighted(X, y, sample_weights, scaler)

        # Build predicted dataset after mitigation
        dataset_pred_rw = dataset.copy(deepcopy=True)
        dataset_pred_rw.labels = y_pred_rw.reshape(-1, 1).astype(float)

        # ── Classification-level fairness metrics (AFTER mitigation) ──
        classification_metrics_after = _compute_classification_metrics(
            dataset, dataset_pred_rw, attr_name
        )

        # ── Dataset metrics after Reweighing ──
        dataset_metrics_after = _compute_dataset_metrics(dataset_rw, attr_name)

        # Feature importance from original model
        feat_importance = {}
        try:
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            for i in top_idx:
                feat_importance[feature_names[i]] = round(float(importances[i]), 6)
        except Exception:
            pass

        out = {
            "available": True,
            "library": "aif360",
            "library_version": aif360_version,
            "model_type": "RandomForestClassifier (n_estimators=50, max_depth=5)",
            "protected_attribute": attr_name,
            "protected_attribute_source": "First EEG channel alpha power (median split)",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_privileged": int(np.sum(protected == 1)),
            "n_unprivileged": int(np.sum(protected == 0)),
            "label_distribution": {
                "positive (abnormal)": int(np.sum(y == 1)),
                "negative (normal)": int(np.sum(y == 0)),
            },
            "dataset_metrics": dataset_metrics,
            "classification_metrics": classification_metrics_before,
            "mitigation_results": {
                "method": "Reweighing (preprocessing)",
                "description": (
                    "Reweighing assigns sample weights to counteract bias in the "
                    "training data without modifying feature values. It adjusts the "
                    "weight of each (group, label) combination to equalize the "
                    "probability of positive outcomes across groups."
                ),
                "before": {
                    "dataset_metrics": dataset_metrics,
                    "classification_metrics": classification_metrics_before,
                },
                "after": {
                    "dataset_metrics": dataset_metrics_after,
                    "classification_metrics": classification_metrics_after,
                },
                "improvement": {
                    "disparate_impact_change": round(
                        dataset_metrics_after["disparate_impact"] - dataset_metrics["disparate_impact"], 6
                    ),
                    "statistical_parity_change": round(
                        dataset_metrics_after["statistical_parity_difference"]
                        - dataset_metrics["statistical_parity_difference"], 6
                    ),
                    "equal_opportunity_change": round(
                        classification_metrics_after["equal_opportunity_difference"]
                        - classification_metrics_before["equal_opportunity_difference"], 6
                    ),
                    "average_odds_change": round(
                        classification_metrics_after["average_odds_difference"]
                        - classification_metrics_before["average_odds_difference"], 6
                    ),
                },
            },
            "top_feature_importance": feat_importance,
            "data_info": meta,
        }
        return _json_clean(out)

    except Exception as exc:
        return {"available": False, "error": f"AIF360 analysis failed: {exc}"}


def aif360_groups(file=None, seconds=10.0) -> Dict[str, Any]:
    """Per-group fairness breakdown: privileged vs unprivileged statistics.

    Returns a dict with per-group base rates, positive prediction rates,
    confusion matrix components, and demographic parity information.
    """
    try:
        import aif360
        aif360_version = str(aif360.__version__) if hasattr(aif360, "__version__") else "installed"
    except ImportError:
        return {"available": False, "error": "aif360 not installed. Install with: pip install aif360"}

    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False, "error": "pandas or sklearn not installed."}

    try:
        result = _extract_features(file, seconds)
        if result[0] is None:
            return {"available": False, "error": "No EDF file found. Place .edf files in data/real_eeg/"}

        X, y, feature_names, meta = result

        # Create protected attribute
        protected, attr_name = _create_protected_attribute(X, feature_names)

        # Train model
        model, scaler, y_pred = _train_rf(X, y)

        # ── Per-group statistics ──
        groups = {}
        for group_val, group_name in [(1, "privileged"), (0, "unprivileged")]:
            mask = protected == group_val
            group_y_true = y[mask]
            group_y_pred = y_pred[mask]
            n_group = int(np.sum(mask))

            if n_group == 0:
                groups[group_name] = {
                    "group_value": int(group_val),
                    "n_samples": 0,
                    "note": "No samples in this group",
                }
                continue

            # Base rate (proportion of positive labels)
            base_rate = float(np.mean(group_y_true))

            # Positive prediction rate
            positive_pred_rate = float(np.mean(group_y_pred))

            # Confusion matrix components
            tp = int(np.sum((group_y_true == 1) & (group_y_pred == 1)))
            tn = int(np.sum((group_y_true == 0) & (group_y_pred == 0)))
            fp = int(np.sum((group_y_true == 0) & (group_y_pred == 1)))
            fn = int(np.sum((group_y_true == 1) & (group_y_pred == 0)))

            # Derived rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            accuracy = (tp + tn) / n_group if n_group > 0 else 0.0
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val = tpr

            groups[group_name] = {
                "group_value": int(group_val),
                "label": f"{attr_name}={group_val}",
                "description": "High alpha power (older)" if group_val == 1 else "Low alpha power (younger)",
                "n_samples": n_group,
                "base_rate": round(base_rate, 6),
                "positive_prediction_rate": round(positive_pred_rate, 6),
                "confusion_matrix": {
                    "true_positives": tp,
                    "true_negatives": tn,
                    "false_positives": fp,
                    "false_negatives": fn,
                },
                "rates": {
                    "true_positive_rate": round(tpr, 6),
                    "false_positive_rate": round(fpr, 6),
                    "true_negative_rate": round(tnr, 6),
                    "false_negative_rate": round(fnr, 6),
                    "accuracy": round(accuracy, 6),
                    "precision": round(precision_val, 6),
                    "recall": round(recall_val, 6),
                },
            }

        # ── Group comparison metrics ──
        priv = groups.get("privileged", {})
        unpriv = groups.get("unprivileged", {})

        comparison = {}
        if priv.get("n_samples", 0) > 0 and unpriv.get("n_samples", 0) > 0:
            comparison = {
                "base_rate_difference": round(
                    priv.get("base_rate", 0) - unpriv.get("base_rate", 0), 6
                ),
                "positive_prediction_rate_difference": round(
                    priv.get("positive_prediction_rate", 0)
                    - unpriv.get("positive_prediction_rate", 0), 6
                ),
                "tpr_difference": round(
                    priv.get("rates", {}).get("true_positive_rate", 0)
                    - unpriv.get("rates", {}).get("true_positive_rate", 0), 6
                ),
                "fpr_difference": round(
                    priv.get("rates", {}).get("false_positive_rate", 0)
                    - unpriv.get("rates", {}).get("false_positive_rate", 0), 6
                ),
                "accuracy_difference": round(
                    priv.get("rates", {}).get("accuracy", 0)
                    - unpriv.get("rates", {}).get("accuracy", 0), 6
                ),
            }

        out = {
            "available": True,
            "library": "aif360",
            "library_version": aif360_version,
            "protected_attribute": attr_name,
            "protected_attribute_source": "First EEG channel alpha power (median split)",
            "groups": groups,
            "group_comparison": comparison,
            "data_info": meta,
            "clinical_note": (
                "Group differences in EEG-AI performance may reflect genuine "
                "neurophysiological variation (e.g., age-related alpha slowing) "
                "or biased training data.  Clinical deployment requires ensuring "
                "equitable sensitivity/specificity across all patient subgroups."
            ),
        }
        return _json_clean(out)

    except Exception as exc:
        return {"available": False, "error": f"AIF360 group analysis failed: {exc}"}


def aif360_definitions() -> Dict[str, Any]:
    """Metric definitions, library info, and clinical relevance for EEG-AI fairness."""
    return {
        "available": True,
        "library_info": {
            "name": "IBM AI Fairness 360 (AIF360)",
            "url": "https://aif360.readthedocs.io/",
            "github": "https://github.com/Trusted-AI/AIF360",
            "citation": (
                "Bellamy, R.K.E., et al. (2019). AI Fairness 360: An extensible "
                "toolkit for detecting and mitigating algorithmic bias. IBM Journal "
                "of Research and Development, 63(4/5), 4:1-4:15. "
                "DOI: 10.1147/JRD.2019.2942287"
            ),
        },
        "dataset_metrics": [
            {
                "name": "Disparate Impact",
                "method": "BinaryLabelDatasetMetric.disparate_impact()",
                "formula": "P(Y=1 | unprivileged) / P(Y=1 | privileged)",
                "ideal_value": 1.0,
                "acceptable_range": "[0.8, 1.25] (80% rule from US EEOC guidelines)",
                "description": (
                    "Ratio of positive outcome rates between unprivileged and "
                    "privileged groups.  Values below 0.8 indicate potential "
                    "adverse impact against the unprivileged group."
                ),
                "clinical_relevance": (
                    "If seizure detection labels (positive=abnormal) are "
                    "disproportionately assigned to one age group in the training "
                    "data, the model may learn to associate demographic features "
                    "with pathology, leading to biased diagnoses."
                ),
            },
            {
                "name": "Statistical Parity Difference",
                "method": "BinaryLabelDatasetMetric.statistical_parity_difference()",
                "formula": "P(Y=1 | unprivileged) - P(Y=1 | privileged)",
                "ideal_value": 0.0,
                "description": (
                    "Difference in positive outcome rates between groups.  "
                    "A value of 0 means both groups have equal probability of "
                    "receiving a positive label."
                ),
                "clinical_relevance": (
                    "Unequal seizure detection rates across demographic groups "
                    "may indicate systematic labeling bias in the training data "
                    "or inherent data collection disparities."
                ),
            },
            {
                "name": "Consistency",
                "method": "BinaryLabelDatasetMetric.consistency()",
                "formula": "1 - mean(|y_i - mean(y_kNN(i))|) for k-nearest neighbors",
                "ideal_value": 1.0,
                "description": (
                    "Measures whether similar individuals receive similar labels.  "
                    "High consistency means the labeling is smooth with respect to "
                    "the feature space (individual fairness)."
                ),
                "clinical_relevance": (
                    "Patients with similar EEG patterns should receive similar "
                    "diagnostic outcomes regardless of demographic group membership."
                ),
            },
        ],
        "classification_metrics": [
            {
                "name": "Equal Opportunity Difference",
                "method": "ClassificationMetric.equal_opportunity_difference()",
                "formula": "TPR(unprivileged) - TPR(privileged)",
                "ideal_value": 0.0,
                "description": (
                    "Difference in true positive rates between groups.  A value "
                    "of 0 means both groups have equal probability of a positive "
                    "instance being correctly classified."
                ),
                "clinical_relevance": (
                    "Ensures seizure events are detected at equal rates across "
                    "patient demographics.  A negative value means the unprivileged "
                    "group's seizures are more likely to be missed."
                ),
            },
            {
                "name": "Average Odds Difference",
                "method": "ClassificationMetric.average_odds_difference()",
                "formula": "0.5 * [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]",
                "ideal_value": 0.0,
                "description": (
                    "Average of the difference in false positive rates and true "
                    "positive rates between groups.  Captures both types of "
                    "classification error disparity."
                ),
                "clinical_relevance": (
                    "Balances both missed seizures (TPR gap) and false alarms "
                    "(FPR gap) across groups.  Critical for ICU settings where "
                    "alarm fatigue from one group's false positives can delay "
                    "response to another group's true events."
                ),
            },
            {
                "name": "Theil Index",
                "method": "ClassificationMetric.theil_index()",
                "formula": "Generalized entropy index with alpha=1",
                "ideal_value": 0.0,
                "description": (
                    "Measures inequality in benefit (correct classification) "
                    "across individuals.  A value of 0 means all individuals "
                    "receive equal benefit from the classifier."
                ),
                "clinical_relevance": (
                    "Quantifies overall inequality in diagnostic accuracy.  "
                    "High Theil index suggests the model serves some patients "
                    "much better than others — unacceptable for a clinical tool "
                    "where equitable care is a regulatory requirement."
                ),
            },
            {
                "name": "Disparate Impact (Classification)",
                "method": "ClassificationMetric.disparate_impact()",
                "formula": "P(Y_pred=1 | unprivileged) / P(Y_pred=1 | privileged)",
                "ideal_value": 1.0,
                "acceptable_range": "[0.8, 1.25]",
                "description": (
                    "Ratio of positive prediction rates between groups.  Unlike "
                    "the dataset metric, this measures model output bias rather "
                    "than data bias."
                ),
                "clinical_relevance": (
                    "Even if training data is balanced, the model may amplify "
                    "small biases.  This metric catches prediction-level "
                    "disparities that dataset metrics miss."
                ),
            },
            {
                "name": "False Positive Rate Difference",
                "method": "ClassificationMetric.false_positive_rate_difference()",
                "formula": "FPR(unprivileged) - FPR(privileged)",
                "ideal_value": 0.0,
                "description": (
                    "Difference in false positive rates between groups.  "
                    "Positive values mean the unprivileged group experiences "
                    "more false alarms."
                ),
                "clinical_relevance": (
                    "Excessive false seizure alarms in one demographic group "
                    "leads to unnecessary interventions, alarm fatigue, and "
                    "erosion of clinical trust in the AI system for that group."
                ),
            },
            {
                "name": "False Negative Rate Difference",
                "method": "ClassificationMetric.false_negative_rate_difference()",
                "formula": "FNR(unprivileged) - FNR(privileged)",
                "ideal_value": 0.0,
                "description": (
                    "Difference in false negative rates between groups.  "
                    "Positive values mean the unprivileged group's positive "
                    "instances are more likely to be missed."
                ),
                "clinical_relevance": (
                    "Missed seizures in one demographic group can have "
                    "life-threatening consequences.  This is arguably the most "
                    "safety-critical fairness metric for seizure detection AI."
                ),
            },
        ],
        "mitigation_algorithms": [
            {
                "name": "Reweighing",
                "class": "aif360.algorithms.preprocessing.Reweighing",
                "type": "preprocessing",
                "description": (
                    "Assigns sample weights to training instances to equalize "
                    "the probability of positive outcomes across protected groups, "
                    "without modifying the feature values or labels themselves."
                ),
                "clinical_relevance": (
                    "Reweighing is the least invasive mitigation technique — it "
                    "does not alter clinical data, only the emphasis placed on each "
                    "training example.  This makes it suitable for regulated medical "
                    "AI where data provenance and integrity must be maintained."
                ),
            },
        ],
        "regulatory_context": {
            "FDA_SaMD": (
                "FDA guidance on AI/ML-based SaMD emphasizes the need for "
                "performance evaluation across relevant patient subgroups "
                "(age, sex, race, comorbidities).  AIF360 metrics directly "
                "support this requirement by quantifying subgroup performance "
                "disparities."
            ),
            "EU_AI_Act": (
                "The EU AI Act (Article 10) requires high-risk AI systems to "
                "use training data that is sufficiently representative and free "
                "from bias.  AIF360 provides the mathematical framework to "
                "verify compliance."
            ),
            "NICE_EVF": (
                "NICE Evidence Standards Framework for digital health "
                "technologies requires demonstration that AI tools perform "
                "equitably across the intended patient population.  AIF360 "
                "metrics provide the quantitative evidence base."
            ),
        },
        "clinical_relevance": (
            "Fairness in EEG-AI is essential because neurophysiological signals "
            "vary systematically with age (alpha slowing), sex (hormonal effects on "
            "cortical excitability), ethnicity (skull thickness affecting amplitude), "
            "and medication status.  A seizure detection model trained predominantly "
            "on young adult males may exhibit degraded sensitivity for elderly females "
            "or pediatric patients.  AIF360 provides a rigorous framework to detect, "
            "quantify, and mitigate such biases before clinical deployment, ensuring "
            "equitable diagnostic accuracy across the full patient population."
        ),
    }


# ── CLI entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    fn_map = {
        "overview": aif360_overview,
        "groups": aif360_groups,
        "definitions": aif360_definitions,
    }
    target = sys.argv[1] if len(sys.argv) > 1 else "overview"
    func = fn_map.get(target, aif360_overview)

    if target == "definitions":
        result = func()
    else:
        edf = sys.argv[2] if len(sys.argv) > 2 else None
        secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
        result = func(edf, secs)

    print(json.dumps(result, indent=2, default=str))
