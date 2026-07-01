"""Model Drift Dashboard — model performance degradation monitoring, accuracy/sensitivity/F1
trends across training runs, evaluation strategy comparison, external validation.

Aggregates data from:
- jobs/reports/training_*.json (dated training runs with per-script results)
- jobs/reports/accuracy_patient_specific.json (per-subject accuracy/sensitivity/F1)
- jobs/reports/accuracy_all_options.json (4 evaluation strategies with fold-level data)
- jobs/reports/bootstrap_ci_baselines.json (bootstrap CIs + literature comparison)
- jobs/reports/bonn_external_validation.json (Bonn University external validation)
- data/clinical.db transaction_log (training + cv_pipeline events)
- models/*.joblib (trained model files)
"""

import sqlite3
import json
import os
import glob
import re
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')
REPORTS_DIR = os.path.join(BASE, 'jobs', 'reports')
MODELS_DIR = os.path.join(BASE, 'models')

BASELINE_ACCURACY = 0.95  # literature benchmark for patient-specific epilepsy detection


def _load_json(filename):
    """Safely load a JSON file from REPORTS_DIR."""
    path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _load_training_files():
    """Load all dated training_*.json files, sorted by date."""
    pattern = os.path.join(REPORTS_DIR, 'training_[0-9]*.json')
    files = sorted(glob.glob(pattern))
    runs = []
    for fpath in files:
        # Extract date from filename: training_YYYYMMDD_HHMMSS.json
        basename = os.path.basename(fpath)
        match = re.search(r'training_(\d{8})_(\d{6})\.json', basename)
        if not match:
            continue
        date_str = match.group(1)
        try:
            date_parsed = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        except ValueError:
            date_parsed = date_str
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        results = data.get('results', [])
        n_runs = len(results)
        all_ok = all(r.get('ok', False) for r in results)
        runs.append({
            'date': date_parsed,
            'run_at': data.get('run_at_local', data.get('run_at_utc', '')),
            'success': all_ok,
            'n_runs': n_runs,
            'summary': data.get('summary', ''),
            'results': results,
        })
    return runs


def _load_db_events(component):
    """Load events from transaction_log for a given component."""
    if not os.path.exists(DB):
        return []
    try:
        conn = sqlite3.connect(DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, component, action, actor, detail, ts_utc, ts_local "
            "FROM transaction_log WHERE component = ? "
            "ORDER BY ts_utc DESC",
            (component,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except (sqlite3.Error, IOError):
        return []


def _load_model_inventory():
    """Load model file metadata from models/*.joblib."""
    pattern = os.path.join(MODELS_DIR, '*.joblib')
    files = sorted(glob.glob(pattern))
    inventory = []
    for fpath in files:
        basename = os.path.basename(fpath)
        name = basename.replace('_model.joblib', '').replace('.joblib', '')
        try:
            size_bytes = os.path.getsize(fpath)
            mtime = os.path.getmtime(fpath)
            modified = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        except OSError:
            size_bytes = 0
            modified = ''
        inventory.append({
            'name': name,
            'path': fpath,
            'size_mb': round(size_bytes / (1024 * 1024), 2),
            'modified': modified,
        })
    return inventory


def _performance_verdict(current_accuracy):
    """Compare current accuracy against literature baseline."""
    if current_accuracy is None:
        return 'UNKNOWN'
    diff = current_accuracy - BASELINE_ACCURACY
    if diff > 0.01:
        return 'IMPROVED'
    elif diff < -0.05:
        return 'DEGRADED'
    return 'STABLE'


def _drift_score(per_subject):
    """Calculate drift score from coefficient of variation of per-subject accuracies.

    drift_score = 100 * (1 - CV), clamped to [0, 100].
    CV = std / mean of per-subject accuracies.
    """
    if not per_subject:
        return 0
    accs = [s.get('accuracy', 0) for s in per_subject]
    if not accs:
        return 0
    mean_acc = sum(accs) / len(accs)
    if mean_acc == 0:
        return 0
    variance = sum((a - mean_acc) ** 2 for a in accs) / len(accs)
    std_acc = variance ** 0.5
    cv = std_acc / mean_acc
    score = 100.0 * (1.0 - cv)
    return round(max(0.0, min(100.0, score)), 1)


def overview():
    """KPI-level summary for the Model Drift dashboard."""
    ps = _load_json('accuracy_patient_specific.json')
    ao = _load_json('accuracy_all_options.json')
    bootstrap = _load_json('bootstrap_ci_baselines.json')
    bonn = _load_json('bonn_external_validation.json')
    training_runs = _load_training_files()
    models = _load_model_inventory()
    training_events = _load_db_events('training')
    cv_events = _load_db_events('cv_pipeline')

    if not ps and not ao:
        return {
            'available': False,
            'message': 'No model performance data available. Run training pipeline first.'
        }

    # Patient-specific metrics
    per_subject = ps.get('per_subject', []) if ps else []
    mean_accuracy = ps.get('mean_accuracy', 0) if ps else 0
    mean_sensitivity = ps.get('mean_sensitivity', 0) if ps else 0

    # Cross-patient accuracy
    cross_patient_acc = 0
    if ao:
        options = ao.get('options', {})
        cp_rf = options.get('2_cross_patient_rf', {})
        cross_patient_acc = cp_rf.get('mean_accuracy', 0)

    # Bonn external validation
    bonn_acc = 0
    if bonn:
        rf_results = bonn.get('results', {}).get('rf', {})
        bonn_acc = rf_results.get('accuracy_mean', 0)

    # Bootstrap CI
    bootstrap_ci = None
    if bootstrap:
        ci_data = bootstrap.get('patient_specific_accuracy_ci', {})
        bootstrap_ci = {
            'mean': ci_data.get('mean', 0),
            'ci95_low': ci_data.get('ci95_low', 0),
            'ci95_high': ci_data.get('ci95_high', 0),
            'n_subjects': ci_data.get('n_subjects', 0),
            'n_boot': ci_data.get('n_boot', 0),
        }

    # Latest training run timestamp
    run_at = ''
    if training_runs:
        run_at = training_runs[-1].get('run_at', '')

    verdict = _performance_verdict(mean_accuracy)
    score = _drift_score(per_subject)

    # KPI colors
    def acc_color(v):
        if v >= 0.95:
            return '#10b981'
        if v >= 0.80:
            return '#f59e0b'
        return '#ef4444'

    kpis = [
        {'label': 'Drift Score', 'value': f'{score}%',
         'color': '#10b981' if score >= 90 else '#f59e0b' if score >= 70 else '#ef4444'},
        {'label': 'Verdict', 'value': verdict,
         'color': '#10b981' if verdict == 'IMPROVED' else '#f59e0b' if verdict == 'STABLE' else '#ef4444'},
        {'label': 'Patient-Specific Acc', 'value': f'{mean_accuracy:.4f}',
         'color': acc_color(mean_accuracy)},
        {'label': 'Mean Sensitivity', 'value': f'{mean_sensitivity:.4f}',
         'color': acc_color(mean_sensitivity)},
        {'label': 'Cross-Patient Acc', 'value': f'{cross_patient_acc:.4f}',
         'color': acc_color(cross_patient_acc)},
        {'label': 'Bonn External Acc', 'value': f'{bonn_acc:.4f}',
         'color': acc_color(bonn_acc)},
        {'label': 'Training Runs', 'value': str(len(training_runs))},
        {'label': 'Models', 'value': str(len(models))},
    ]

    return {
        'available': True,
        'run_at': run_at,
        'n_training_runs': len(training_runs),
        'n_models': len(models),
        'patient_specific_accuracy': mean_accuracy,
        'patient_specific_sensitivity': mean_sensitivity,
        'cross_patient_accuracy': cross_patient_acc,
        'bonn_external_accuracy': bonn_acc,
        'bootstrap_ci': bootstrap_ci,
        'performance_verdict': verdict,
        'drift_score': score,
        'kpis': kpis,
        'n_cv_events': len(cv_events),
        'n_training_events': len(training_events),
    }


def breakdown():
    """Detailed per-subject, per-strategy, per-run data for the Model Drift dashboard."""
    ps = _load_json('accuracy_patient_specific.json')
    ao = _load_json('accuracy_all_options.json')
    bootstrap = _load_json('bootstrap_ci_baselines.json')
    training_runs = _load_training_files()
    models = _load_model_inventory()
    training_events = _load_db_events('training')

    if not ps and not ao:
        return {'available': False}

    # Per-subject data
    per_subject = []
    if ps:
        for s in ps.get('per_subject', []):
            per_subject.append({
                'subject': s.get('subject', ''),
                'accuracy': s.get('accuracy', 0),
                'sensitivity': s.get('sensitivity', 0),
                'f1': s.get('f1', 0),
                'n_total': s.get('n_total', 0),
                'n_seizure': s.get('n_seizure', 0),
            })

    # Evaluation strategies
    evaluation_strategies = []
    cross_validation_folds = {}
    if ao:
        options = ao.get('options', {})

        # 1. Patient-specific
        ps_opt = options.get('1_patient_specific', {})
        evaluation_strategies.append({
            'method': 'Patient-Specific (temporal split, ensemble)',
            'accuracy': ps_opt.get('mean_accuracy', 0),
            'details': 'Per-subject temporal split; train on early windows, test on late windows.',
        })
        cross_validation_folds['patient_specific'] = ps_opt.get('per_subject', [])

        # 2. Cross-patient RF
        cp_rf = options.get('2_cross_patient_rf', {})
        evaluation_strategies.append({
            'method': 'Cross-Patient RF (leave-one-subject-out)',
            'accuracy': cp_rf.get('mean_accuracy', 0),
            'details': 'Random Forest with leave-one-subject-out cross-validation.',
        })
        cross_validation_folds['cross_patient_rf'] = cp_rf.get('folds', [])

        # 3. Cross-patient ensemble
        cp_ens = options.get('3_cross_patient_ensemble', {})
        evaluation_strategies.append({
            'method': 'Cross-Patient Ensemble',
            'accuracy': cp_ens.get('mean_accuracy', 0),
            'details': 'Ensemble classifier with leave-one-subject-out cross-validation.',
        })
        cross_validation_folds['cross_patient_ensemble'] = cp_ens.get('folds', [])

        # 4. Cross-patient ensemble + normalization
        cp_norm = options.get('4_cross_patient_ensemble_normed', {})
        evaluation_strategies.append({
            'method': 'Cross-Patient Ensemble + Per-Subject Normalization',
            'accuracy': cp_norm.get('mean_accuracy', 0),
            'details': 'Ensemble with per-subject feature normalization.',
        })
        cross_validation_folds['cross_patient_ensemble_normed'] = cp_norm.get('folds', [])

    # Training timeline
    training_timeline = []
    for run in training_runs:
        training_timeline.append({
            'date': run['date'],
            'success': run['success'],
            'n_runs': run['n_runs'],
            'summary': run.get('summary', ''),
        })

    # Bootstrap comparison
    bootstrap_comparison = []
    if bootstrap:
        methods = bootstrap.get('baseline_comparison', {}).get('methods', [])
        for m in methods:
            bootstrap_comparison.append({
                'method': m.get('method', ''),
                'setting': m.get('setting', ''),
                'reported': m.get('reported', ''),
                'source': m.get('source', ''),
            })

    # Model inventory
    model_inventory = models

    # Training events from transaction_log
    event_list = []
    for ev in training_events:
        event_list.append({
            'ts': ev.get('ts_local', ev.get('ts_utc', '')),
            'action': ev.get('action', ''),
            'actor': ev.get('actor', ''),
            'detail': ev.get('detail', ''),
        })

    return {
        'available': True,
        'per_subject': per_subject,
        'evaluation_strategies': evaluation_strategies,
        'training_timeline': training_timeline,
        'cross_validation_folds': cross_validation_folds,
        'bootstrap_comparison': bootstrap_comparison,
        'model_inventory': model_inventory,
        'training_events': event_list,
    }


def definitions():
    """Definitions tab for the Model Drift dashboard."""
    return {
        'sections': [
            {
                'title': 'Model Drift',
                'items': [
                    {'term': 'Model Drift',
                     'definition': 'Degradation of model performance over time, measured by drops in accuracy, sensitivity, F1, or precision. Unlike data drift (which monitors input feature distributions via PSI/KS), model drift monitors output quality — whether the model still predicts correctly as conditions change.'},
                    {'term': 'Performance Baseline',
                     'definition': f'The literature benchmark for patient-specific epilepsy detection is {BASELINE_ACCURACY:.0%} accuracy. Current performance is compared against this baseline to determine drift verdict.'},
                    {'term': 'Drift Score',
                     'definition': 'A 0–100 score measuring consistency of model performance across subjects. Calculated as 100 * (1 - coefficient of variation of per-subject accuracies). 100 = perfectly uniform performance, 0 = extreme variation.'},
                    {'term': 'Performance Verdict',
                     'definition': 'IMPROVED: accuracy exceeds baseline by > 1%. STABLE: accuracy within 5% below baseline. DEGRADED: accuracy dropped > 5% below baseline. Triggers retraining alerts.'},
                ]
            },
            {
                'title': 'Performance Metrics',
                'items': [
                    {'term': 'Accuracy',
                     'definition': 'Fraction of all predictions (seizure and non-seizure) that are correct. (TP + TN) / (TP + TN + FP + FN).'},
                    {'term': 'Sensitivity (Recall)',
                     'definition': 'Fraction of actual seizure events correctly detected. TP / (TP + FN). Critical in clinical settings — a missed seizure is more dangerous than a false alarm.'},
                    {'term': 'Specificity',
                     'definition': 'Fraction of non-seizure windows correctly identified. TN / (TN + FP). Important for reducing alarm fatigue.'},
                    {'term': 'F1 Score',
                     'definition': 'Harmonic mean of precision and recall. 2 * (precision * recall) / (precision + recall). Balances false positives and false negatives.'},
                    {'term': 'AUC',
                     'definition': 'Area Under the ROC Curve. Probability that the model ranks a random seizure window higher than a random non-seizure window. 1.0 = perfect, 0.5 = random.'},
                ]
            },
            {
                'title': 'Evaluation Strategies',
                'items': [
                    {'term': 'Patient-Specific (temporal split)',
                     'definition': 'Train on early EEG windows, test on late windows for each subject independently. Simulates clinical deployment where a model is calibrated to one patient. Most realistic evaluation.'},
                    {'term': 'Cross-Patient (leave-one-subject-out)',
                     'definition': 'Train on N-1 subjects, test on the held-out subject. Tests generalizability across patients. Typically yields lower accuracy due to inter-patient variability.'},
                    {'term': 'Ensemble vs Single Model',
                     'definition': 'Ensemble combines multiple classifiers (Random Forest, Gradient Boosting, etc.) via voting. Single model uses one classifier. Ensemble reduces variance but may not improve cross-patient generalization.'},
                    {'term': 'Per-Subject Normalization',
                     'definition': 'Z-score normalization of features within each subject before cross-patient evaluation. Aims to reduce inter-subject variability but can remove discriminative information.'},
                ]
            },
            {
                'title': 'Bootstrap Confidence Intervals',
                'items': [
                    {'term': 'Subject-Level Bootstrap',
                     'definition': 'Resamples subjects (not individual windows) to compute confidence intervals. Correct for non-independent overlapping windows within each subject.'},
                    {'term': '95% CI',
                     'definition': 'The interval within which the true population metric falls with 95% probability, estimated from 2000 bootstrap iterations over 4 subjects.'},
                    {'term': 'Literature Comparison',
                     'definition': 'Bootstrap CIs are compared against published benchmarks (Shoeb 2010, Truong et al. 2018) to assess whether our model performance is competitive.'},
                ]
            },
            {
                'title': 'External Validation',
                'items': [
                    {'term': 'Bonn University Dataset',
                     'definition': 'An independent epilepsy EEG dataset (200 samples, balanced 100/100) used to validate that the model generalizes beyond the CHB-MIT training data.'},
                    {'term': 'Generalizability',
                     'definition': 'A model that achieves high accuracy on both CHB-MIT and Bonn datasets demonstrates robustness across recording equipment, patient populations, and clinical settings.'},
                    {'term': 'Stratified 5-Fold CV',
                     'definition': 'Cross-validation on the Bonn dataset using 5 folds with preserved class balance. Provides an unbiased estimate of external validation performance.'},
                ]
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'IEC 62304',
                     'definition': 'Medical device software lifecycle standard. Requires ongoing performance monitoring and documented evidence that model accuracy has not degraded post-deployment.'},
                    {'term': 'FDA AI/ML Framework',
                     'definition': 'The Predetermined Change Control Plan (PCCP) requires continuous monitoring of model performance metrics. Drift beyond acceptable thresholds triggers review and potential retraining.'},
                    {'term': 'EU AI Act',
                     'definition': 'High-risk AI systems (including medical diagnostics) must implement post-market monitoring systems that detect performance degradation and trigger corrective action.'},
                    {'term': 'Clinical Impact',
                     'definition': 'Undetected model drift in seizure detection can lead to missed seizures (reduced sensitivity) or alarm fatigue (reduced specificity). Both have direct patient safety implications.'},
                ]
            },
        ]
    }
