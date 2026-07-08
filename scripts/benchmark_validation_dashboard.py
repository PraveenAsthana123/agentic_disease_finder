"""
Benchmark Validation Dashboard
===============================
External dataset validation results for the neuro-AI epilepsy platform.
Shows Bonn University epilepsy EEG dataset cross-validation performance
to demonstrate model generalization beyond CHB-MIT.

Data Source:
  - jobs/reports/bonn_external_validation.json

Author: Research Team
"""

import json
from pathlib import Path

REPORT_PATH = str(Path(__file__).parent.parent / "jobs" / "reports" / "bonn_external_validation.json")


def _load():
    try:
        with open(REPORT_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def overview():
    """Dataset KPIs + per-model performance summary."""
    d = _load()
    if not d:
        return {"dataset": None, "n_samples": 0, "n_features": 0,
                "balance": "", "cv": "", "purpose": "", "generated_at": "",
                "models": {}}

    results = d.get("results", {})
    models = {}
    for model_id, metrics in results.items():
        models[model_id] = {
            "accuracy_mean": metrics.get("accuracy_mean", 0),
            "f1_mean": metrics.get("f1_mean", 0),
            "auc_mean": metrics.get("auc_mean", 0),
        }

    return {
        "dataset": d.get("dataset", ""),
        "n_samples": d.get("n_samples", 0),
        "n_features": d.get("n_features", 0),
        "balance": d.get("balance", ""),
        "cv": d.get("cv", ""),
        "purpose": d.get("purpose", ""),
        "generated_at": d.get("generated_at", ""),
        "models": models,
    }


def breakdown():
    """Fold-level results + model comparison table."""
    d = _load()
    if not d:
        return {"models": {}, "comparison": []}

    results = d.get("results", {})

    models = {}
    comparison = []
    for model_id, metrics in results.items():
        models[model_id] = {
            "fold_acc": metrics.get("fold_acc", []),
            "accuracy_mean": metrics.get("accuracy_mean", 0),
            "accuracy_std": metrics.get("accuracy_std", 0),
            "f1_mean": metrics.get("f1_mean", 0),
            "auc_mean": metrics.get("auc_mean", 0),
        }
        comparison.append({
            "model": model_id,
            "accuracy": metrics.get("accuracy_mean", 0),
            "f1": metrics.get("f1_mean", 0),
            "auc": metrics.get("auc_mean", 0),
        })

    return {
        "models": models,
        "comparison": comparison,
    }


def definitions():
    """Benchmark validation glossary."""
    return {
        "terms": [
            {"term": "External Validation", "definition": "Testing a model on a completely independent dataset that was not used during training or hyperparameter tuning. The gold standard for demonstrating model generalization."},
            {"term": "Bonn Dataset", "definition": "University of Bonn epilepsy EEG dataset — 5 classes (A-E) of EEG segments from healthy volunteers and epilepsy patients. Widely used benchmark in epilepsy AI research."},
            {"term": "Cross-Validation", "definition": "Splitting data into K folds, training on K-1 and testing on 1, rotating through all folds. Reduces overfitting risk and gives confidence intervals."},
            {"term": "Stratified K-Fold", "definition": "Cross-validation that preserves class proportions in each fold, critical for imbalanced medical datasets."},
            {"term": "AUC (Area Under ROC Curve)", "definition": "Probability that the model ranks a random positive higher than a random negative. 1.0 = perfect, 0.5 = random. Threshold-independent metric."},
            {"term": "F1 Score", "definition": "Harmonic mean of precision and recall. Balances false positives and false negatives — critical in clinical settings where both matter."},
            {"term": "Random Forest", "definition": "Ensemble of decision trees trained on random subsets. Robust, interpretable, handles mixed features well. The project's primary classifier."},
            {"term": "Ensemble", "definition": "Combining multiple models (e.g., RF + SVM + MLP) via voting or stacking. Typically more robust than any single model."},
            {"term": "Generalization", "definition": "A model's ability to perform well on unseen data from different sources, populations, or recording setups. The #1 concern in clinical AI."},
            {"term": "Dataset Confound", "definition": "When a model learns dataset-specific artifacts (recording equipment, population demographics) instead of the actual signal. External validation detects this."},
            {"term": "Accuracy", "definition": "Proportion of correct predictions out of total predictions. Simple but can be misleading with imbalanced classes."},
            {"term": "Balance", "definition": "Ratio of positive to negative samples. 100/100 means perfectly balanced — no class imbalance bias."},
            {"term": "CHB-MIT", "definition": "Children's Hospital Boston / MIT scalp EEG dataset. The primary training dataset for this platform's seizure models."},
        ]
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(overview(), indent=2, default=str))
