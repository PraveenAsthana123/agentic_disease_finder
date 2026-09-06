"""Training Configuration Dashboard — 7-disease ML training pipeline
configuration from config/training_config.yaml.
Covers data, features, models, training, validation, and performance targets."""

import yaml
from pathlib import Path
from collections import Counter

_CFG = Path(__file__).resolve().parent.parent / "config" / "training_config.yaml"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return yaml.safe_load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: disease count, feature count, model count,
    sample sizes, CV folds, performance targets."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "training_config.yaml missing"}

    data_cfg = cfg.get("data", {})
    features_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    validation_cfg = cfg.get("validation", {})
    targets_cfg = cfg.get("targets", {})

    diseases = data_cfg.get("diseases", {})
    disease_count = len(diseases)

    # Sample size info
    total_original = sum(d.get("original_samples", 0) for d in diseases.values())
    total_augmented = sum(d.get("augmented_samples", 0) for d in diseases.values())

    # Feature counts by domain
    time_feats = features_cfg.get("time_domain", [])
    freq_feats = features_cfg.get("frequency_domain", [])
    nonlinear_feats = features_cfg.get("nonlinear", [])
    total_features = features_cfg.get("total_features", len(time_feats) + len(freq_feats) + len(nonlinear_feats))
    selected_features = features_cfg.get("selected_features", total_features)

    # Model info
    model_type = model_cfg.get("type", "unknown")
    n_models = model_cfg.get("n_models", 0)
    model_names = [k for k in model_cfg if k not in ("type", "n_models")]

    # CV info
    cv = training_cfg.get("cross_validation", {})
    n_splits = cv.get("n_splits", 0)

    # Feature domain distribution for pie chart
    feature_distribution = [
        {"name": "Time Domain", "value": len(time_feats)},
        {"name": "Frequency Domain", "value": len(freq_feats)},
        {"name": "Nonlinear", "value": len(nonlinear_feats)},
    ]

    # Samples per disease for bar chart
    samples_per_disease = [
        {"name": d.get("name", k), "original": d.get("original_samples", 0),
         "augmented": d.get("augmented_samples", 0)}
        for k, d in diseases.items()
    ]

    # Models table
    models_table = []
    for mname in model_names:
        mconf = model_cfg.get(mname, {})
        if isinstance(mconf, dict):
            key_params = ", ".join(f"{pk}={pv}" for pk, pv in list(mconf.items())[:3])
            models_table.append({"model": mname.replace("_", " ").title(), "params": key_params})

    return {
        "available": True,
        "title": cfg.get("project", {}).get("name", "Training Configuration"),
        "version": cfg.get("project", {}).get("version", ""),
        "summary": {
            "disease_count": disease_count,
            "total_original_samples": total_original,
            "total_augmented_samples": total_augmented,
            "total_features": total_features,
            "selected_features": selected_features,
            "feature_selection_method": features_cfg.get("selection_method", ""),
            "n_models": n_models,
            "model_type": model_type,
            "cv_folds": n_splits,
            "stratified": cv.get("stratified", False),
            "min_accuracy_target": targets_cfg.get("min_accuracy", 0),
            "min_f1_target": targets_cfg.get("min_f1_score", 0),
            "max_train_test_gap": targets_cfg.get("max_train_test_gap", 0),
        },
        "feature_distribution": feature_distribution,
        "samples_per_disease": samples_per_disease,
        "models_table": models_table,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Full detail for diseases, features, models, training, validation."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "training_config.yaml missing"}

    data_cfg = cfg.get("data", {})
    features_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    validation_cfg = cfg.get("validation", {})
    output_cfg = cfg.get("output", {})
    targets_cfg = cfg.get("targets", {})

    # Disease cards
    diseases = []
    for k, d in data_cfg.get("diseases", {}).items():
        diseases.append({
            "key": k,
            "name": d.get("name", k),
            "path": d.get("path", ""),
            "original_samples": d.get("original_samples", 0),
            "augmented_samples": d.get("augmented_samples", 0),
        })

    # Feature lists by domain
    feature_domains = {
        "time_domain": features_cfg.get("time_domain", []),
        "frequency_domain": features_cfg.get("frequency_domain", []),
        "nonlinear": features_cfg.get("nonlinear", []),
    }

    # Model configs
    models = []
    for mname in [k for k in model_cfg if k not in ("type", "n_models")]:
        mconf = model_cfg.get(mname, {})
        if isinstance(mconf, dict):
            models.append({
                "name": mname.replace("_", " ").title(),
                "key": mname,
                "params": {pk: pv for pk, pv in mconf.items()},
            })

    # Training settings
    training_detail = {
        "cross_validation": training_cfg.get("cross_validation", {}),
        "regularization": training_cfg.get("regularization", {}),
        "data_augmentation": training_cfg.get("data_augmentation", {}),
        "oversampling": training_cfg.get("oversampling", {}),
    }

    # Validation settings
    validation_detail = {
        "external_validation": validation_cfg.get("external_validation", False),
        "holdout_ratio": validation_cfg.get("holdout_ratio", 0),
        "bootstrap_ci": validation_cfg.get("bootstrap_ci", False),
        "n_bootstrap": validation_cfg.get("n_bootstrap", 0),
        "confidence_level": validation_cfg.get("confidence_level", 0),
    }

    # Output settings
    output_detail = {
        "models_dir": output_cfg.get("models_dir", ""),
        "results_dir": output_cfg.get("results_dir", ""),
        "reports_dir": output_cfg.get("reports_dir", ""),
        "save_models": output_cfg.get("save_models", False),
        "save_metrics": output_cfg.get("save_metrics", False),
        "generate_reports": output_cfg.get("generate_reports", False),
    }

    # Performance targets
    perf_targets = {
        "min_accuracy": targets_cfg.get("min_accuracy", 0),
        "min_f1_score": targets_cfg.get("min_f1_score", 0),
        "max_train_test_gap": targets_cfg.get("max_train_test_gap", 0),
        "max_cv_std": targets_cfg.get("max_cv_std", 0),
        "overfitting_threshold": targets_cfg.get("overfitting_threshold", 0),
    }

    return {
        "available": True,
        "diseases": diseases,
        "feature_domains": feature_domains,
        "feature_selection": {
            "total": features_cfg.get("total_features", 0),
            "selected": features_cfg.get("selected_features", 0),
            "method": features_cfg.get("selection_method", ""),
        },
        "models": models,
        "training": training_detail,
        "validation": validation_detail,
        "output": output_detail,
        "targets": perf_targets,
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Glossary, clinical notes, references for training config."""
    return {
        "available": True,
        "glossary": [
            {"term": "Stratified K-Fold", "definition": "Cross-validation that preserves class proportion in each fold, critical for imbalanced disease datasets"},
            {"term": "SMOTE", "definition": "Synthetic Minority Over-sampling Technique — generates synthetic samples for minority classes to address class imbalance"},
            {"term": "Mutual Information", "definition": "Feature selection method measuring statistical dependency between features and target; selects most informative EEG features"},
            {"term": "Random Forest", "definition": "Ensemble of decision trees using bagging with random feature subsets; robust to overfitting with interpretable feature importances"},
            {"term": "Extra Trees", "definition": "Extremely Randomized Trees — similar to Random Forest but with random split thresholds for faster training and reduced variance"},
            {"term": "Gradient Boosting", "definition": "Sequential ensemble that fits trees to residual errors; achieves high accuracy with controlled learning rate"},
            {"term": "SVM (RBF)", "definition": "Support Vector Machine with Radial Basis Function kernel — maps features to high-dimensional space for nonlinear classification"},
            {"term": "MLP", "definition": "Multi-Layer Perceptron — feedforward neural network with hidden layers for learning nonlinear decision boundaries"},
            {"term": "L2 Regularization", "definition": "Weight decay penalty that shrinks model coefficients toward zero, reducing overfitting risk"},
            {"term": "Bootstrap CI", "definition": "Confidence Interval estimated by resampling with replacement; provides uncertainty bounds for performance metrics"},
            {"term": "DFA", "definition": "Detrended Fluctuation Analysis — measures long-range temporal correlations in EEG signals"},
            {"term": "Hjorth Parameters", "definition": "Time-domain EEG descriptors: Activity (variance), Mobility (mean frequency), Complexity (bandwidth)"},
        ],
        "clinical_notes": [
            "All 7 diseases use augmented datasets (SMOTE + noise injection) to reach 200 samples per class for balanced training.",
            "Feature selection reduces 47 raw features to 25 using mutual information, preventing curse of dimensionality.",
            "The 6-model ensemble (RF, ET, GB, SVM, LR, MLP) uses regularization to prevent overfitting on small clinical datasets.",
            "External validation on 20% holdout ensures generalization beyond cross-validation splits.",
        ],
        "references": [
            {"ref": "training_config.yaml", "detail": "Primary training configuration file — all hyperparameters, disease paths, feature lists, and performance targets"},
            {"ref": "scikit-learn", "detail": "Machine learning library providing all 6 ensemble models, cross-validation, and SMOTE oversampling"},
            {"ref": "ILAE Classification", "detail": "International League Against Epilepsy — defines seizure types and epilepsy syndromes used as classification targets"},
            {"ref": "FDA AI/ML SaMD", "detail": "FDA Good Machine Learning Practice — requires documented training config, validation splits, and performance targets for clinical AI"},
        ],
    }
