#!/usr/bin/env python3
"""
Deepchecks Dashboard — real deepchecks data/model validation on EEG classification
===================================================================================

Extracts real EEG spectral/entropy/Hjorth features from CHB-MIT EDF recordings
via MNE, trains a small sklearn RandomForestClassifier, then runs comprehensive
deepchecks validation suites for data integrity and model performance.

Functions provided:
  - deepchecks_overview      — data integrity + model performance checks
  - deepchecks_suites        — detailed suite results with per-condition pass/fail
  - deepchecks_definitions   — check definitions and clinical relevance

Clinical relevance:
  Automated data and model validation is critical for EEG-AI systems destined
  for clinical deployment.  Deepchecks detects data integrity issues (nulls,
  duplicates, type mismatches), distribution drift between train/test splits,
  feature-label leakage, and model performance degradation — all of which can
  silently degrade seizure detection accuracy.  These checks align with FDA
  SaMD continuous monitoring requirements and IEC 62304 software lifecycle
  standards for AI-based medical devices.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")

# ── NumPy 2.x compat patches (deepchecks 0.19 uses removed aliases) ──
# In NumPy 2.x, removed aliases raise AttributeError via __getattr__,
# so we must force-set them regardless of hasattr.
try:
    np.Inf = np.inf
    np.NaN = np.nan
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float64
    np.object = np.object_
    np.str = np.str_
except Exception:
    pass

ROOT = Path(__file__).parent.parent

# EEG band definitions
_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def _patch_sklearn_scorers():
    """Patch missing 'max_error' scorer in newer sklearn (>=1.4) so deepchecks can import.
    Also patches np.Inf removed in NumPy 2.0."""
    try:
        if not hasattr(np, "Inf"):
            np.Inf = np.inf
        if not hasattr(np, "NaN"):
            np.NaN = np.nan
        if not hasattr(np, "bool"):
            np.bool = np.bool_
        if not hasattr(np, "int"):
            np.int = np.int_
        if not hasattr(np, "float"):
            np.float = np.float64
        if not hasattr(np, "object"):
            np.object = np.object_
        if not hasattr(np, "str"):
            np.str = np.str_
    except Exception:
        pass
    try:
        from sklearn.metrics._scorer import _SCORERS
        if "max_error" not in _SCORERS:
            from sklearn.metrics import make_scorer, max_error
            _SCORERS["max_error"] = make_scorer(max_error, greater_is_better=False)
    except Exception:
        pass


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
    """Extract real EEG features from EDF for deepchecks validation.

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


# ── model training ──────────────────────────────────────────────────

def _train_rf(X, y, feature_names):
    """Train a RandomForestClassifier and return train/test DataFrames + model.

    Returns (model, df_train, df_test, label_col, feature_names).
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build DataFrame for deepchecks
    df = pd.DataFrame(X_scaled, columns=feature_names)
    label_col = "label"
    df[label_col] = y

    # Train/test split (stratified when possible)
    try:
        df_train, df_test = train_test_split(
            df, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        df_train, df_test = train_test_split(
            df, test_size=0.3, random_state=42
        )

    # Ensure at least 2 test samples
    if len(df_test) < 2:
        df_test = df.copy()
        df_train = df.copy()

    model = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(df_train[feature_names], df_train[label_col])

    return model, df_train, df_test, label_col, feature_names


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


def _extract_check_result(check_result):
    """Extract meaningful data from a single deepchecks CheckResult."""
    info = {
        "check_name": str(check_result.header) if hasattr(check_result, "header") else str(type(check_result).__name__),
    }

    # Extract conditions results
    if hasattr(check_result, "conditions_results") and check_result.conditions_results:
        conditions = []
        for cond in check_result.conditions_results:
            cond_info = {
                "status": str(cond.category.value) if hasattr(cond.category, "value") else str(cond.category),
                "name": str(cond.name) if hasattr(cond, "name") else "",
                "details": str(cond.details) if hasattr(cond, "details") else "",
            }
            conditions.append(cond_info)
        info["conditions"] = conditions
        # Determine overall status from conditions
        statuses = [c["status"].upper() for c in conditions]
        if any("FAIL" in s for s in statuses):
            info["status"] = "FAIL"
        elif any("WARN" in s for s in statuses):
            info["status"] = "WARNING"
        else:
            info["status"] = "PASS"
    else:
        info["status"] = "PASS"
        info["conditions"] = []

    # Extract value
    if hasattr(check_result, "value") and check_result.value is not None:
        val = check_result.value
        try:
            import pandas as pd
            if isinstance(val, pd.DataFrame):
                if val.shape[0] <= 20:
                    info["value"] = val.to_dict(orient="records")
                else:
                    info["value"] = f"DataFrame with {val.shape[0]} rows x {val.shape[1]} cols"
                    info["value_sample"] = val.head(5).to_dict(orient="records")
            elif isinstance(val, pd.Series):
                info["value"] = val.to_dict()
            elif isinstance(val, dict):
                info["value"] = val
            elif isinstance(val, (int, float, str, bool)):
                info["value"] = val
            elif isinstance(val, np.ndarray):
                info["value"] = val.tolist()
            else:
                info["value"] = str(val)[:500]
        except Exception:
            try:
                info["value"] = str(val)[:500]
            except Exception:
                info["value"] = "unable to serialize"

    return info


# ── public API ──────────────────────────────────────────────────────

def deepchecks_overview(file=None, seconds=10.0) -> Dict[str, Any]:
    """Run real deepchecks data integrity + model performance checks on EEG data."""
    # Ensure numpy compat patches are applied (deepchecks 0.19 + numpy 2.x)
    import sys
    _np = sys.modules.get("numpy")
    if _np is not None:
        _np.Inf = _np.inf
        _np.NaN = _np.nan
        for _alias, _real in [("bool", "bool_"), ("int", "int_"), ("float", "float64"),
                               ("object", "object_"), ("str", "str_")]:
            if hasattr(_np, _real):
                setattr(_np, _alias, getattr(_np, _real))

    try:
        _patch_sklearn_scorers()
        import deepchecks
        from deepchecks.tabular import Dataset
        dc_version = str(deepchecks.__version__)
    except ImportError:
        return {"available": False, "note": "deepchecks not installed. Install with: pip install deepchecks"}

    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return {"available": False, "note": "pandas or sklearn not installed."}

    result = _extract_features(file, seconds)
    if result[0] is None:
        return {"available": False, "note": "No EDF data found for deepchecks validation."}

    X, y, feature_names, meta = result

    try:
        model, df_train, df_test, label_col, feat_names = _train_rf(X, y, feature_names)
    except Exception as exc:
        return {"available": False, "note": f"Model training failed: {exc}"}

    # Build deepchecks Datasets
    try:
        ds_train = Dataset(df_train, label=label_col, features=feat_names)
        ds_test = Dataset(df_test, label=label_col, features=feat_names)
    except Exception as exc:
        return {"available": False, "note": f"Dataset creation failed: {exc}"}

    check_results = []
    summary = {"passed": 0, "failed": 0, "warnings": 0, "errors": 0, "total": 0}

    # ── Data Integrity Checks ──
    data_integrity_checks = []

    # MixedNulls
    try:
        from deepchecks.tabular.checks import MixedNulls
        cr = MixedNulls().run(ds_train)
        info = _extract_check_result(cr)
        info["category"] = "data_integrity"
        data_integrity_checks.append(info)
    except Exception as exc:
        data_integrity_checks.append({"check_name": "MixedNulls", "status": "ERROR", "error": str(exc), "category": "data_integrity"})

    # MixedDataTypes
    try:
        from deepchecks.tabular.checks import MixedDataTypes
        cr = MixedDataTypes().run(ds_train)
        info = _extract_check_result(cr)
        info["category"] = "data_integrity"
        data_integrity_checks.append(info)
    except Exception as exc:
        data_integrity_checks.append({"check_name": "MixedDataTypes", "status": "ERROR", "error": str(exc), "category": "data_integrity"})

    # StringMismatch
    try:
        from deepchecks.tabular.checks import StringMismatch
        cr = StringMismatch().run(ds_train)
        info = _extract_check_result(cr)
        info["category"] = "data_integrity"
        data_integrity_checks.append(info)
    except Exception as exc:
        data_integrity_checks.append({"check_name": "StringMismatch", "status": "ERROR", "error": str(exc), "category": "data_integrity"})

    # DataDuplicates
    try:
        from deepchecks.tabular.checks import DataDuplicates
        cr = DataDuplicates().run(ds_train)
        info = _extract_check_result(cr)
        info["category"] = "data_integrity"
        data_integrity_checks.append(info)
    except Exception as exc:
        data_integrity_checks.append({"check_name": "DataDuplicates", "status": "ERROR", "error": str(exc), "category": "data_integrity"})

    # FeatureFeatureCorrelation
    try:
        from deepchecks.tabular.checks import FeatureFeatureCorrelation
        cr = FeatureFeatureCorrelation().run(ds_train)
        info = _extract_check_result(cr)
        info["category"] = "data_integrity"
        data_integrity_checks.append(info)
    except Exception as exc:
        data_integrity_checks.append({"check_name": "FeatureFeatureCorrelation", "status": "ERROR", "error": str(exc), "category": "data_integrity"})

    # FeatureLabelCorrelation
    try:
        from deepchecks.tabular.checks import FeatureLabelCorrelation
        cr = FeatureLabelCorrelation().run(ds_train)
        info = _extract_check_result(cr)
        info["category"] = "data_integrity"
        data_integrity_checks.append(info)
    except Exception as exc:
        data_integrity_checks.append({"check_name": "FeatureLabelCorrelation", "status": "ERROR", "error": str(exc), "category": "data_integrity"})

    # OutlierSampleDetection
    try:
        from deepchecks.tabular.checks import OutlierSampleDetection
        cr = OutlierSampleDetection().run(ds_train)
        info = _extract_check_result(cr)
        info["category"] = "data_integrity"
        data_integrity_checks.append(info)
    except Exception as exc:
        data_integrity_checks.append({"check_name": "OutlierSampleDetection", "status": "ERROR", "error": str(exc), "category": "data_integrity"})

    check_results.extend(data_integrity_checks)

    # ── Model Performance Checks ──
    model_perf_checks = []

    # TrainTestPerformance
    try:
        from deepchecks.tabular.checks import TrainTestPerformance
        cr = TrainTestPerformance().run(ds_train, ds_test, model)
        info = _extract_check_result(cr)
        info["category"] = "model_performance"
        model_perf_checks.append(info)
    except Exception as exc:
        model_perf_checks.append({"check_name": "TrainTestPerformance", "status": "ERROR", "error": str(exc), "category": "model_performance"})

    # ConfusionMatrixReport
    try:
        from deepchecks.tabular.checks import ConfusionMatrixReport
        cr = ConfusionMatrixReport().run(ds_train, ds_test, model)
        info = _extract_check_result(cr)
        info["category"] = "model_performance"
        model_perf_checks.append(info)
    except Exception as exc:
        model_perf_checks.append({"check_name": "ConfusionMatrixReport", "status": "ERROR", "error": str(exc), "category": "model_performance"})

    # RocReport (may not be available in all versions)
    try:
        from deepchecks.tabular.checks import RocReport
        cr = RocReport().run(ds_train, ds_test, model)
        info = _extract_check_result(cr)
        info["category"] = "model_performance"
        model_perf_checks.append(info)
    except ImportError:
        model_perf_checks.append({"check_name": "RocReport", "status": "SKIPPED", "note": "Not available in this deepchecks version", "category": "model_performance"})
    except Exception as exc:
        model_perf_checks.append({"check_name": "RocReport", "status": "ERROR", "error": str(exc), "category": "model_performance"})

    # SimpleModelComparison
    try:
        from deepchecks.tabular.checks import SimpleModelComparison
        cr = SimpleModelComparison().run(ds_train, ds_test, model)
        info = _extract_check_result(cr)
        info["category"] = "model_performance"
        model_perf_checks.append(info)
    except Exception as exc:
        model_perf_checks.append({"check_name": "SimpleModelComparison", "status": "ERROR", "error": str(exc), "category": "model_performance"})

    check_results.extend(model_perf_checks)

    # Tally summary
    for cr in check_results:
        summary["total"] += 1
        status = cr.get("status", "PASS").upper()
        if "FAIL" in status:
            summary["failed"] += 1
        elif "WARN" in status:
            summary["warnings"] += 1
        elif "ERROR" in status:
            summary["errors"] += 1
        else:
            summary["passed"] += 1

    # Feature importance from the RF model
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
        "library": "deepchecks",
        "library_version": dc_version,
        "model_type": "RandomForestClassifier (n_estimators=50, max_depth=5)",
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "n_train_samples": len(df_train),
        "n_test_samples": len(df_test),
        "n_classes": 2,
        "class_names": ["Normal (0)", "Abnormal (1)"],
        "file": meta["file"],
        "data_info": meta,
        "checks": check_results,
        "summary": summary,
        "top_feature_importance": feat_importance,
    }
    return _json_clean(out)


def deepchecks_suites(file=None, seconds=10.0) -> Dict[str, Any]:
    """Run full deepchecks validation suites with detailed per-condition results."""
    try:
        _patch_sklearn_scorers()
        import deepchecks
        from deepchecks.tabular import Dataset
        dc_version = str(deepchecks.__version__)
    except ImportError:
        return {"available": False, "note": "deepchecks not installed. Install with: pip install deepchecks"}

    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return {"available": False, "note": "pandas or sklearn not installed."}

    result = _extract_features(file, seconds)
    if result[0] is None:
        return {"available": False, "note": "No EDF data found for deepchecks suite validation."}

    X, y, feature_names, meta = result

    try:
        model, df_train, df_test, label_col, feat_names = _train_rf(X, y, feature_names)
    except Exception as exc:
        return {"available": False, "note": f"Model training failed: {exc}"}

    try:
        ds_train = Dataset(df_train, label=label_col, features=feat_names)
        ds_test = Dataset(df_test, label=label_col, features=feat_names)
    except Exception as exc:
        return {"available": False, "note": f"Dataset creation failed: {exc}"}

    suites_output = {
        "available": True,
        "library": "deepchecks",
        "library_version": dc_version,
        "file": meta["file"],
        "data_info": meta,
        "suites": [],
    }

    # ── Data Integrity Suite ──
    try:
        from deepchecks.tabular.suites import data_integrity
        integrity_suite = data_integrity()
        suite_result = integrity_suite.run(ds_train)

        suite_checks = []
        passed = 0
        failed = 0
        warned = 0

        for check_result in suite_result.results:
            info = _extract_check_result(check_result)
            suite_checks.append(info)
            status = info.get("status", "PASS").upper()
            if "FAIL" in status:
                failed += 1
            elif "WARN" in status:
                warned += 1
            else:
                passed += 1

        suites_output["suites"].append({
            "suite_name": "Data Integrity",
            "description": "Validates data quality: nulls, types, duplicates, outliers, correlations",
            "total_checks": len(suite_checks),
            "passed": passed,
            "failed": failed,
            "warnings": warned,
            "checks": suite_checks,
        })
    except Exception as exc:
        suites_output["suites"].append({
            "suite_name": "Data Integrity",
            "error": str(exc),
            "checks": [],
        })

    # ── Train-Test Validation Suite ──
    try:
        from deepchecks.tabular.suites import train_test_validation
        tt_suite = train_test_validation()
        suite_result = tt_suite.run(ds_train, ds_test, model)

        suite_checks = []
        passed = 0
        failed = 0
        warned = 0

        for check_result in suite_result.results:
            info = _extract_check_result(check_result)
            suite_checks.append(info)
            status = info.get("status", "PASS").upper()
            if "FAIL" in status:
                failed += 1
            elif "WARN" in status:
                warned += 1
            else:
                passed += 1

        suites_output["suites"].append({
            "suite_name": "Train-Test Validation",
            "description": "Validates train-test split: drift, performance, leakage detection",
            "total_checks": len(suite_checks),
            "passed": passed,
            "failed": failed,
            "warnings": warned,
            "checks": suite_checks,
        })
    except Exception as exc:
        suites_output["suites"].append({
            "suite_name": "Train-Test Validation",
            "error": str(exc),
            "checks": [],
        })

    # Feature importance
    try:
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        suites_output["top_feature_importance"] = {
            feature_names[i]: round(float(importances[i]), 6) for i in top_idx
        }
    except Exception:
        suites_output["top_feature_importance"] = {}

    # Overall summary across all suites
    total_p = sum(s.get("passed", 0) for s in suites_output["suites"])
    total_f = sum(s.get("failed", 0) for s in suites_output["suites"])
    total_w = sum(s.get("warnings", 0) for s in suites_output["suites"])
    total_c = sum(s.get("total_checks", 0) for s in suites_output["suites"])
    suites_output["overall_summary"] = {
        "total_checks": total_c,
        "passed": total_p,
        "failed": total_f,
        "warnings": total_w,
    }

    return _json_clean(suites_output)


def deepchecks_definitions() -> Dict[str, Any]:
    """Check definitions, library info, and clinical relevance for EEG-AI."""
    try:
        _patch_sklearn_scorers()
        import deepchecks
        version = str(deepchecks.__version__)
    except ImportError:
        return {"available": False, "note": "deepchecks not installed."}

    return {
        "available": True,
        "library_info": {
            "name": "Deepchecks",
            "version": version,
            "url": "https://docs.deepchecks.com/stable/",
            "citation": (
                "Deepchecks (2022). Deepchecks: A Library for Testing and "
                "Validating Machine Learning Models and Data. "
                "https://github.com/deepchecks/deepchecks"
            ),
        },
        "data_integrity_checks": [
            {
                "name": "MixedNulls",
                "class": "deepchecks.tabular.checks.MixedNulls",
                "description": (
                    "Detects columns with multiple representations of null values "
                    "(e.g., None, NaN, empty string, 'NA'). In EEG pipelines, mixed "
                    "nulls can indicate corrupted electrode channels or dropped packets."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_mixed_nulls.html",
                "clinical_relevance": (
                    "Corrupted EEG channels with inconsistent null representations "
                    "can silently propagate through feature extraction, producing "
                    "misleading band power or entropy values that degrade seizure "
                    "detection accuracy."
                ),
            },
            {
                "name": "MixedDataTypes",
                "class": "deepchecks.tabular.checks.MixedDataTypes",
                "description": (
                    "Identifies columns containing a mix of data types (e.g., strings "
                    "and numbers in the same column). Can reveal parsing errors in "
                    "EEG metadata or feature pipelines."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_mixed_data_types.html",
                "clinical_relevance": (
                    "Type inconsistencies in EEG feature matrices can cause silent "
                    "dtype coercions during model training, especially when aggregating "
                    "data from heterogeneous recording systems (e.g., different EDF "
                    "exporters)."
                ),
            },
            {
                "name": "StringMismatch",
                "class": "deepchecks.tabular.checks.StringMismatch",
                "description": (
                    "Detects string columns with near-duplicate values due to "
                    "capitalization or whitespace differences."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_string_mismatch.html",
                "clinical_relevance": (
                    "In EEG datasets, channel name mismatches (e.g., 'FP1' vs 'Fp1' "
                    "vs 'fp1') can cause incorrect channel-to-region mapping, leading "
                    "to erroneous lateralization or localization conclusions."
                ),
            },
            {
                "name": "DataDuplicates",
                "class": "deepchecks.tabular.checks.DataDuplicates",
                "description": (
                    "Detects exact duplicate samples in the dataset. Duplicates "
                    "inflate apparent model performance and violate train/test "
                    "independence."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_data_duplicates.html",
                "clinical_relevance": (
                    "Duplicate EEG epochs can occur from overlapping windowing or "
                    "data pipeline bugs. They artificially inflate accuracy metrics, "
                    "giving false confidence in seizure detection performance."
                ),
            },
            {
                "name": "FeatureFeatureCorrelation",
                "class": "deepchecks.tabular.checks.FeatureFeatureCorrelation",
                "description": (
                    "Identifies highly correlated feature pairs. High inter-feature "
                    "correlation indicates redundancy and can cause instability in "
                    "linear models."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_feature_feature_correlation.html",
                "clinical_relevance": (
                    "EEG band powers from adjacent frequency bands (e.g., alpha/beta) "
                    "and from spatially nearby electrodes are often highly correlated. "
                    "Identifying and managing these correlations prevents "
                    "multicollinearity artifacts in classification."
                ),
            },
            {
                "name": "FeatureLabelCorrelation",
                "class": "deepchecks.tabular.checks.FeatureLabelCorrelation",
                "description": (
                    "Measures correlation between each feature and the target label. "
                    "Extremely high correlation may indicate label leakage."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_feature_label_correlation.html",
                "clinical_relevance": (
                    "In EEG-AI, leakage often occurs when the theta/alpha ratio "
                    "(used to derive synthetic labels) remains as a feature. "
                    "This check catches such circular dependencies before deployment."
                ),
            },
            {
                "name": "OutlierSampleDetection",
                "class": "deepchecks.tabular.checks.OutlierSampleDetection",
                "description": (
                    "Identifies outlier samples using LOF (Local Outlier Factor) or "
                    "isolation forest. Outlier epochs may indicate artifacts."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/data_integrity/plot_outlier_sample_detection.html",
                "clinical_relevance": (
                    "EEG artifact-contaminated epochs (eye blinks, muscle activity, "
                    "electrode pops) produce outlier feature vectors. Detecting these "
                    "before model training prevents artifact-driven false seizure "
                    "detections."
                ),
            },
        ],
        "model_validation_checks": [
            {
                "name": "TrainTestPerformance",
                "class": "deepchecks.tabular.checks.TrainTestPerformance",
                "description": (
                    "Compares model performance metrics (accuracy, precision, recall, "
                    "F1) between train and test sets. Large gaps indicate overfitting."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_train_test_performance.html",
                "clinical_relevance": (
                    "Overfitting on patient-specific EEG patterns produces models "
                    "that fail on new patients. This check quantifies the "
                    "generalization gap critical for cross-patient seizure detection."
                ),
            },
            {
                "name": "ConfusionMatrixReport",
                "class": "deepchecks.tabular.checks.ConfusionMatrixReport",
                "description": (
                    "Generates a detailed confusion matrix showing true/false "
                    "positives and negatives for each class."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_confusion_matrix_report.html",
                "clinical_relevance": (
                    "For seizure detection, false negatives (missed seizures) are "
                    "clinically dangerous, while false positives cause alarm fatigue "
                    "in ICU settings. The confusion matrix makes these trade-offs "
                    "explicit."
                ),
            },
            {
                "name": "RocReport",
                "class": "deepchecks.tabular.checks.RocReport",
                "description": (
                    "Plots the ROC curve and computes AUC. Provides "
                    "threshold-independent assessment of classifier discriminative "
                    "ability."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_roc_report.html",
                "clinical_relevance": (
                    "AUROC is a standard metric for evaluating EEG-based diagnostic "
                    "classifiers, required by FDA SaMD guidance for AI medical "
                    "devices. Values below 0.7 are generally considered clinically "
                    "inadequate."
                ),
            },
            {
                "name": "SimpleModelComparison",
                "class": "deepchecks.tabular.checks.SimpleModelComparison",
                "description": (
                    "Compares the trained model against simple baseline models "
                    "(random, constant, tree). Ensures the model meaningfully "
                    "outperforms trivial strategies."
                ),
                "docs": "https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_simple_model_comparison.html",
                "clinical_relevance": (
                    "An EEG classifier that does not significantly outperform a "
                    "simple baseline (e.g., always predicting 'no seizure') provides "
                    "no clinical value. This check prevents deployment of trivially "
                    "performing models."
                ),
            },
        ],
        "validation_suites": [
            {
                "name": "data_integrity",
                "class": "deepchecks.tabular.suites.data_integrity",
                "description": (
                    "Full data integrity suite: runs all data quality checks "
                    "including nulls, types, duplicates, outliers, and correlations."
                ),
            },
            {
                "name": "train_test_validation",
                "class": "deepchecks.tabular.suites.train_test_validation",
                "description": (
                    "Full train-test validation suite: runs drift detection, "
                    "performance comparison, feature importance, and leakage checks."
                ),
            },
        ],
        "clinical_relevance": (
            "Automated data and model validation is mandatory for EEG-AI systems "
            "intended for clinical use.  Deepchecks provides a systematic framework "
            "that catches data quality issues (corrupted channels, duplicate epochs, "
            "type mismatches), distribution shifts between training and deployment "
            "data, feature leakage, and model degradation — all of which can silently "
            "undermine seizure detection reliability.  These checks align with FDA "
            "SaMD predetermined change control plans, IEC 62304 software verification "
            "requirements, and the EU AI Act's continuous monitoring mandates for "
            "high-risk medical AI applications."
        ),
    }


# ── CLI entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    fn_map = {
        "overview": deepchecks_overview,
        "suites": deepchecks_suites,
        "definitions": deepchecks_definitions,
    }
    target = sys.argv[1] if len(sys.argv) > 1 else "overview"
    func = fn_map.get(target, deepchecks_overview)

    if target == "definitions":
        result = func()
    else:
        edf = sys.argv[2] if len(sys.argv) > 2 else None
        secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
        result = func(edf, secs)

    print(json.dumps(result, indent=2, default=str))
