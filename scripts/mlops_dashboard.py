"""MLOps Dashboard — training pipeline monitoring, experiment tracking, feature store,
cross-validation metrics.

Aggregates data from:
- jobs/reports/training_*.json (7 dated training run files + training_latest.json)
- jobs/reports/accuracy_patient_specific.json (per-subject CV results, CHB-MIT)
- jobs/reports/accuracy_all_options.json (4 evaluation strategies)
- jobs/reports/multi_disease_accuracy.json (7 disease models, in-sample)
- jobs/reports/drift_latest.json (feature drift monitoring)
- jobs/reports/cv_pipeline_latest.json (CV pipeline latest run)
- data/clinical.db transaction_log (training + cv_pipeline components)
- models/*.joblib (7 trained model files)
"""

import sqlite3
import json
import os
import glob
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')
MODELS_DIR = os.path.join(BASE, 'models')
REPORTS_DIR = os.path.join(BASE, 'jobs', 'reports')

# The 15 EEG features used in the patient-specific pipeline (deterministic list)
EEG_FEATURES = [
    'delta_power',
    'theta_power',
    'alpha_power',
    'beta_power',
    'gamma_power',
    'line_length',
    'hjorth_activity',
    'hjorth_mobility',
    'hjorth_complexity',
    'spectral_entropy',
    'zero_crossings',
    'peak_frequency',
    'band_ratio_theta_alpha',
    'band_ratio_delta_theta',
    'variance',
]


def _conn():
    return sqlite3.connect(DB)


def _read_json(name):
    path = os.path.join(REPORTS_DIR, name)
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_training_runs():
    """Load all dated training_*.json files (excluding training_latest.json)."""
    pattern = os.path.join(REPORTS_DIR, 'training_????????_??????.json')
    files = sorted(glob.glob(pattern))
    runs = []
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            # Extract date from filename: training_YYYYMMDD_HHMMSS.json
            fname = os.path.basename(fpath)
            date_part = fname.replace('training_', '').replace('.json', '')
            runs.append({
                'filename': fname,
                'date_key': date_part,
                'run_at_utc': data.get('run_at_utc'),
                'run_at_local': data.get('run_at_local'),
                'dataset': data.get('dataset'),
                'results': data.get('results', []),
                'summary': data.get('summary', ''),
            })
        except Exception:
            continue
    return runs


def _pipeline_health():
    """Get most recent training/cv_pipeline event from transaction_log."""
    conn = _conn()
    cur = conn.cursor()
    latest = None
    try:
        cur.execute("""
            SELECT ts_utc, component, action, actor, detail
            FROM transaction_log
            WHERE component IN ('training', 'cv_pipeline')
            ORDER BY ts_utc DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            latest = {
                'ts_utc': row[0],
                'component': row[1],
                'action': row[2],
                'actor': row[3],
                'detail': row[4],
            }
    except Exception:
        pass
    finally:
        conn.close()
    return latest


def overview():
    """MLOps overview: KPIs, training history, CV summary, evaluation strategies."""
    runs = _load_training_runs()
    patient_specific = _read_json('accuracy_patient_specific.json') or {}
    all_options = _read_json('accuracy_all_options.json') or {}
    multi_disease = _read_json('multi_disease_accuracy.json') or {}
    pipeline_health = _pipeline_health()

    # ---------- KPIs ----------
    total_experiments = len(runs)
    total_models = 7  # 7 .joblib files in models/

    # Mean accuracy + best/worst from multi_disease (in-sample, optimistic)
    disease_results = multi_disease.get('results', [])
    accuracies = [r['accuracy'] for r in disease_results if r.get('accuracy') is not None]
    mean_accuracy = round(sum(accuracies) / len(accuracies), 4) if accuracies else None
    best_model = None
    worst_model = None
    if disease_results:
        best = max(disease_results, key=lambda r: r.get('accuracy', 0))
        worst = min(disease_results, key=lambda r: r.get('accuracy', 1))
        best_model = {'disease': best['disease'], 'accuracy': best['accuracy']}
        worst_model = {'disease': worst['disease'], 'accuracy': worst['accuracy']}

    # Patient-specific CV subjects
    per_subject = patient_specific.get('per_subject', [])
    total_cv_subjects = len(per_subject)
    mean_sensitivity = patient_specific.get('mean_sensitivity')

    # Pipeline health: determine status from last event
    if pipeline_health:
        ph_status = 'healthy'
        ph_last_seen = pipeline_health['ts_utc']
        ph_component = pipeline_health['component']
    else:
        ph_status = 'unknown'
        ph_last_seen = None
        ph_component = None

    kpis = {
        'total_experiments': total_experiments,
        'total_models': total_models,
        'mean_accuracy': mean_accuracy,
        'mean_accuracy_note': 'in-sample (optimistic) — multi-disease in-sample evaluation',
        'best_model': best_model,
        'worst_model': worst_model,
        'total_cv_subjects': total_cv_subjects,
        'mean_sensitivity': mean_sensitivity,
        'mean_sensitivity_source': 'CHB-MIT patient-specific CV (accuracy_patient_specific.json)',
        'pipeline_health': {
            'status': ph_status,
            'last_event_utc': ph_last_seen,
            'component': ph_component,
        },
    }

    # ---------- Training History ----------
    training_history = []
    for run in runs:
        results = run['results']
        scripts_run = len(results)
        all_ok = all(r.get('ok', False) for r in results)
        total_seconds = round(sum(r.get('seconds', 0) for r in results), 1)
        # Parse date from run_at_local or run_at_utc
        date_str = None
        for ts_field in ('run_at_local', 'run_at_utc'):
            ts = run.get(ts_field)
            if ts:
                try:
                    date_str = ts[:10]
                    break
                except Exception:
                    pass
        training_history.append({
            'date': date_str,
            'run_at_utc': run['run_at_utc'],
            'run_at_local': run['run_at_local'],
            'dataset': run['dataset'],
            'scripts_run': scripts_run,
            'all_ok': all_ok,
            'total_seconds': total_seconds,
            'summary': run['summary'],
            'script_details': [
                {
                    'script': r.get('script'),
                    'exit_code': r.get('exit_code'),
                    'ok': r.get('ok'),
                    'seconds': r.get('seconds'),
                }
                for r in results
            ],
        })

    # ---------- CV Summary ----------
    cv_summary = {
        'benchmark': patient_specific.get('benchmark'),
        'no_leakage': patient_specific.get('no_leakage'),
        'window_seconds': patient_specific.get('window_seconds'),
        'stride_seconds': patient_specific.get('stride_seconds'),
        'features': patient_specific.get('features'),
        'mean_accuracy': patient_specific.get('mean_accuracy'),
        'mean_sensitivity': patient_specific.get('mean_sensitivity'),
        'generated_at': patient_specific.get('generated_at'),
        'per_subject': per_subject,
    }

    # ---------- Evaluation Strategies ----------
    options = all_options.get('options', {})
    evaluation_strategies = []
    strategy_labels = {
        '1_patient_specific': 'Patient-Specific (per-subject temporal split)',
        '2_cross_patient_rf': 'Cross-Patient RandomForest (leave-one-out)',
        '3_cross_patient_ensemble': 'Cross-Patient Ensemble (leave-one-out)',
        '4_cross_patient_ensemble_normed': 'Cross-Patient Ensemble + Per-Subject Norm',
    }
    for key, label in strategy_labels.items():
        opt = options.get(key, {})
        evaluation_strategies.append({
            'strategy': key,
            'label': label,
            'mean_accuracy': opt.get('mean_accuracy'),
            'method': opt.get('method'),
        })

    return {
        'kpis': kpis,
        'training_history': training_history,
        'cv_summary': cv_summary,
        'evaluation_strategies': evaluation_strategies,
    }


def breakdown():
    """MLOps breakdown: disease model accuracy, full CV detail, pipeline events,
    feature inventory, model file inventory, daily training activity."""
    multi_disease = _read_json('multi_disease_accuracy.json') or {}
    all_options = _read_json('accuracy_all_options.json') or {}

    # ---------- Multi-Disease Accuracy ----------
    multi_disease_accuracy = []
    for r in multi_disease.get('results', []):
        multi_disease_accuracy.append({
            'disease': r.get('disease'),
            'status': r.get('status'),
            'n_samples': r.get('n_samples'),
            'n_features': r.get('n_features'),
            'accuracy': r.get('accuracy'),
            'precision': r.get('precision'),
            'recall': r.get('recall'),
            'f1': r.get('f1'),
            'evaluation': r.get('evaluation'),
            'class_names': r.get('class_names'),
            'confusion_matrix': r.get('confusion_matrix'),
            'training_metrics': r.get('training_metrics'),
        })
    multi_disease_note = {
        'evaluation_type': multi_disease.get('evaluation_type'),
        'caveat': multi_disease.get('caveat'),
        'mean_accuracy': multi_disease.get('mean_accuracy'),
        'generated_at': multi_disease.get('generated_at'),
    }

    # ---------- Cross-Validation Detail ----------
    options = all_options.get('options', {})
    cross_validation_detail = {}
    for key, opt in options.items():
        cross_validation_detail[key] = {
            'method': opt.get('method'),
            'mean_accuracy': opt.get('mean_accuracy'),
            'folds': opt.get('folds') or opt.get('per_subject'),
        }

    # ---------- Training Pipeline Events ----------
    training_pipeline_events = []
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT ts_utc, ts_local, component, action, actor, detail
            FROM transaction_log
            WHERE component IN ('training', 'cv_pipeline')
            ORDER BY ts_utc DESC
            LIMIT 50
        """)
        for row in cur.fetchall():
            training_pipeline_events.append({
                'ts_utc': row[0],
                'ts_local': row[1],
                'component': row[2],
                'action': row[3],
                'actor': row[4],
                'detail': row[5],
            })
    except Exception:
        pass

    # ---------- Feature Inventory ----------
    patient_specific = _read_json('accuracy_patient_specific.json') or {}
    feature_count = patient_specific.get('features', 15)
    feature_descriptions = {
        'delta_power': 'Power in 0.5–4 Hz band; associated with deep sleep and pathological states',
        'theta_power': 'Power in 4–8 Hz band; linked to drowsiness and memory encoding',
        'alpha_power': 'Power in 8–13 Hz band; dominant in relaxed wakefulness',
        'beta_power': 'Power in 13–30 Hz band; associated with active thinking and alertness',
        'gamma_power': 'Power in 30–100 Hz band; linked to high-level cognitive processing',
        'line_length': 'Sum of absolute first differences; sensitive to seizure onset amplitude',
        'hjorth_activity': 'Variance of the signal; measures signal power/amplitude',
        'hjorth_mobility': 'Square root of variance ratio of 1st derivative to signal',
        'hjorth_complexity': 'Ratio of mobility of 1st derivative to signal mobility',
        'spectral_entropy': 'Entropy of normalized power spectrum; measures signal complexity',
        'zero_crossings': 'Rate of sign changes; sensitive to oscillation frequency',
        'peak_frequency': 'Frequency bin with maximum power in the spectrum',
        'band_ratio_theta_alpha': 'Ratio of theta to alpha power; tracks arousal shifts',
        'band_ratio_delta_theta': 'Ratio of delta to theta power; distinguishes sleep stages',
        'variance': 'Statistical variance of the EEG window; amplitude variability measure',
    }
    feature_inventory = [
        {
            'index': i + 1,
            'name': feat,
            'description': feature_descriptions.get(feat, ''),
        }
        for i, feat in enumerate(EEG_FEATURES[:feature_count])
    ]

    # ---------- Model File Inventory ----------
    model_file_inventory = []
    disease_names = [
        'epilepsy', 'depression', 'alzheimer', 'parkinson',
        'autism', 'schizophrenia', 'stress',
    ]
    for disease in disease_names:
        fpath = os.path.join(MODELS_DIR, f'{disease}_model.joblib')
        if os.path.exists(fpath):
            stat = os.stat(fpath)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            model_file_inventory.append({
                'disease': disease,
                'filename': f'{disease}_model.joblib',
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 3),
                'modified': mtime.strftime('%Y-%m-%d %H:%M:%S'),
            })

    # ---------- Daily Training Activity ----------
    daily_training_activity = []
    try:
        cur.execute("""
            SELECT DATE(ts_utc) as d, component, COUNT(*) as cnt
            FROM transaction_log
            WHERE component IN ('training', 'cv_pipeline')
            GROUP BY d, component
            ORDER BY d
        """)
        for row in cur.fetchall():
            daily_training_activity.append({
                'date': row[0],
                'component': row[1],
                'count': row[2],
            })
    except Exception:
        pass
    finally:
        conn.close()

    return {
        'multi_disease_accuracy': multi_disease_accuracy,
        'multi_disease_note': multi_disease_note,
        'cross_validation_detail': cross_validation_detail,
        'training_pipeline_events': training_pipeline_events,
        'feature_inventory': feature_inventory,
        'model_file_inventory': model_file_inventory,
        'daily_training_activity': daily_training_activity,
    }


def definitions():
    """Definitions for the MLOps dashboard."""
    return {
        'title': 'MLOps Dashboard — Definitions',
        'sections': [
            {
                'name': 'Training Pipeline',
                'description': (
                    'Automated training pipeline that runs on a daily cron schedule. '
                    'Two scripts are executed per run: accuracy_patient_specific.py '
                    '(per-subject temporal split with ensemble) and accuracy_all_options.py '
                    '(four evaluation strategies). Dataset: CHB-MIT PhysioNet scalp EEG '
                    '(chb01–chb04), a public benchmark for seizure detection.'
                ),
                'fields': [
                    {
                        'name': 'accuracy_patient_specific.py',
                        'description': (
                            'Trains one model per CHB-MIT subject on early time windows, '
                            'tests on late windows (no data leakage). Uses a 4-second window '
                            'with 2-second stride. Computes accuracy, F1, and sensitivity per subject.'
                        ),
                    },
                    {
                        'name': 'accuracy_all_options.py',
                        'description': (
                            'Evaluates four evaluation regimes (patient-specific, cross-patient RF, '
                            'cross-patient ensemble, cross-patient ensemble + per-subject normalisation) '
                            'to give an honest accuracy summary across strategies.'
                        ),
                    },
                    {
                        'name': 'Dataset',
                        'description': (
                            'CHB-MIT Scalp EEG Database (PhysioNet). Pediatric patients with '
                            'intractable seizures. 4 subjects (chb01–chb04) used in current pipeline.'
                        ),
                    },
                    {
                        'name': 'Run Schedule',
                        'description': (
                            'Daily cron at 02:30 local (08:30 UTC). Results written to '
                            'jobs/reports/training_YYYYMMDD_HHMMSS.json and training_latest.json.'
                        ),
                    },
                ],
            },
            {
                'name': 'Evaluation Types',
                'description': (
                    'Multiple evaluation strategies are used with different '
                    'generalization assumptions. Honest reporting requires '
                    'distinguishing in-sample estimates from genuine out-of-sample results.'
                ),
                'items': [
                    (
                        'In-Sample (optimistic) — multi-disease models trained and evaluated on '
                        'the same dataset. Accuracy is inflated and does NOT represent real-world '
                        'generalization. Source: multi_disease_accuracy.json.'
                    ),
                    (
                        'Patient-Specific (clinical use case) — each subject\'s model is trained '
                        'on early windows and tested on late windows of the same subject. '
                        'Temporal split prevents leakage. Accuracy ~0.98. '
                        'Source: accuracy_patient_specific.json.'
                    ),
                    (
                        'Cross-Patient (leave-one-out) — model trained on all subjects except '
                        'one, tested on the held-out subject. Tests true cross-subject '
                        'generalization. Accuracy ~0.66–0.73. '
                        'Source: accuracy_all_options.json strategies 2–4.'
                    ),
                    (
                        'Honest reporting: clinical AI claims should cite cross-patient or '
                        'external validation accuracy, not in-sample figures. '
                        'Reference: IEC 62304 §5.1 — software safety classification.'
                    ),
                ],
            },
            {
                'name': 'CV Strategies',
                'description': 'The four evaluation options computed by accuracy_all_options.py.',
                'fields': [
                    {
                        'name': '1 — Patient-Specific',
                        'description': (
                            'Per-subject temporal split with ensemble. Train on first 2/3 '
                            'of each subject\'s windows, test on last 1/3. Mean accuracy: 0.8934.'
                        ),
                    },
                    {
                        'name': '2 — Cross-Patient RandomForest',
                        'description': (
                            'Leave-one-subject-out CV using a RandomForest classifier. '
                            'Train on 3 subjects, test on the 4th. Mean accuracy: 0.7277.'
                        ),
                    },
                    {
                        'name': '3 — Cross-Patient Ensemble',
                        'description': (
                            'Leave-one-subject-out CV using an ensemble of classifiers. '
                            'Tests whether ensemble diversity helps cross-subject transfer. '
                            'Mean accuracy: 0.661.'
                        ),
                    },
                    {
                        'name': '4 — Cross-Patient Ensemble + Per-Subject Norm',
                        'description': (
                            'Same as strategy 3, but with per-subject feature normalisation '
                            'applied at test time. Intended to reduce amplitude variability '
                            'across subjects. Mean accuracy: 0.6314.'
                        ),
                    },
                ],
            },
            {
                'name': 'EEG Features',
                'description': (
                    '15 time- and frequency-domain features extracted from each 4-second '
                    'EEG window (2-second stride). Computed per channel, then aggregated.'
                ),
                'fields': [
                    {'name': 'delta_power', 'description': 'Power in 0.5–4 Hz band'},
                    {'name': 'theta_power', 'description': 'Power in 4–8 Hz band'},
                    {'name': 'alpha_power', 'description': 'Power in 8–13 Hz band'},
                    {'name': 'beta_power', 'description': 'Power in 13–30 Hz band'},
                    {'name': 'gamma_power', 'description': 'Power in 30–100 Hz band'},
                    {'name': 'line_length', 'description': 'Sum of absolute first differences; seizure-sensitive'},
                    {'name': 'hjorth_activity', 'description': 'Signal variance (amplitude)'},
                    {'name': 'hjorth_mobility', 'description': 'Frequency of dominant oscillation'},
                    {'name': 'hjorth_complexity', 'description': 'Waveform complexity ratio'},
                    {'name': 'spectral_entropy', 'description': 'Power spectrum entropy; complexity measure'},
                    {'name': 'zero_crossings', 'description': 'Sign-change rate; oscillation proxy'},
                    {'name': 'peak_frequency', 'description': 'Dominant frequency bin in spectrum'},
                    {'name': 'band_ratio_theta_alpha', 'description': 'Theta/alpha ratio; arousal indicator'},
                    {'name': 'band_ratio_delta_theta', 'description': 'Delta/theta ratio; sleep/wake proxy'},
                    {'name': 'variance', 'description': 'Window amplitude variability'},
                ],
            },
            {
                'name': 'Metrics',
                'description': 'Standard classification performance metrics used across all evaluations.',
                'fields': [
                    {
                        'name': 'Accuracy',
                        'description': 'Fraction of all samples correctly classified: (TP+TN)/(TP+TN+FP+FN)',
                    },
                    {
                        'name': 'Precision',
                        'description': 'Fraction of positive predictions that are correct: TP/(TP+FP). '
                                       'High precision = low false alarm rate.',
                    },
                    {
                        'name': 'Recall (Sensitivity)',
                        'description': 'Fraction of actual positives correctly detected: TP/(TP+FN). '
                                       'High recall = low missed-seizure rate. Clinically critical.',
                    },
                    {
                        'name': 'F1',
                        'description': 'Harmonic mean of precision and recall: 2*(P*R)/(P+R). '
                                       'Balanced metric for imbalanced classes.',
                    },
                    {
                        'name': 'Specificity',
                        'description': 'Fraction of actual negatives correctly identified: TN/(TN+FP). '
                                       'Measures false positive suppression.',
                    },
                    {
                        'name': 'AUC',
                        'description': 'Area Under the ROC Curve. Threshold-independent measure of '
                                       'discriminative ability (1.0 = perfect, 0.5 = random).',
                    },
                ],
            },
            {
                'name': 'Clinical Relevance',
                'description': (
                    'MLOps practices directly support IEC 62304 (medical device software lifecycle) '
                    'and ISO 13485 (quality management for medical devices).'
                ),
                'items': [
                    (
                        'IEC 62304 §5.1 — Software development planning requires documented '
                        'training and validation procedures. Experiment tracking fulfills this.'
                    ),
                    (
                        'IEC 62304 §5.7 — Software release must include verification that '
                        'performance meets clinical requirements. CV metrics provide evidence.'
                    ),
                    (
                        'Honest evaluation — in-sample scores must be clearly labeled (optimistic). '
                        'Cross-patient or external validation accuracy must be cited in clinical claims.'
                    ),
                    (
                        'Sensitivity is the primary clinical metric for seizure detection: '
                        'a missed seizure (false negative) carries higher clinical risk than '
                        'a false alarm (false positive).'
                    ),
                    (
                        'Feature reproducibility — the 15 EEG features are deterministic given '
                        'a fixed window/stride. Documented for audit trail and reproducibility (IEC 62304 §5.5).'
                    ),
                ],
            },
        ],
    }
