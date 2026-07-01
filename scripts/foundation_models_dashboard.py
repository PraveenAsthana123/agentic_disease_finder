"""Foundation Models Dashboard — catalog of all trained clinical AI models,
architectures, training metadata, performance metrics, and disease coverage.

Scans real model files from:
- models/*.joblib (7 production disease classifiers)
- saved_models/*.joblib + *.pt (ensemble/deep/stacking variants)
- models/deep_learning_models.py (neural architecture definitions)
- data/clinical.db analyses (21 EEG analyses with model predictions)
- data/clinical.db patients (40 patients with disease labels)
"""

import sqlite3
import json
import os
import re
from collections import defaultdict, Counter
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')
MODELS_DIR = os.path.join(BASE, 'models')
SAVED_DIR = os.path.join(BASE, 'saved_models')


def _conn():
    return sqlite3.connect(DB)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _file_size_human(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _scan_production_models():
    """Scan models/ for production .joblib classifiers."""
    models = []
    if not os.path.isdir(MODELS_DIR):
        return models
    for f in sorted(os.listdir(MODELS_DIR)):
        if not f.endswith('.joblib'):
            continue
        path = os.path.join(MODELS_DIR, f)
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        disease = f.replace('_model.joblib', '').replace('.joblib', '')
        models.append({
            'filename': f,
            'disease': disease.title(),
            'size_bytes': size,
            'size_human': _file_size_human(size),
            'modified': datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
            'framework': 'scikit-learn',
            'format': 'joblib',
            'tier': 'production',
        })
    return models


def _scan_saved_models():
    """Scan saved_models/ for all model variants."""
    models = []
    if not os.path.isdir(SAVED_DIR):
        return models
    for f in sorted(os.listdir(SAVED_DIR)):
        if not (f.endswith('.joblib') or f.endswith('.pt')):
            continue
        path = os.path.join(SAVED_DIR, f)
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        fmt = 'pytorch' if f.endswith('.pt') else 'joblib'
        framework = 'PyTorch' if f.endswith('.pt') else 'scikit-learn'

        parts = f.replace('.joblib', '').replace('.pt', '').split('_')
        disease = parts[0].title() if parts else 'Unknown'

        algo = 'Unknown'
        tier = 'experimental'
        if 'ultra_stacking' in f:
            algo = 'Ultra Stacking Ensemble'
            tier = 'advanced'
        elif 'Stacking_real' in f:
            algo = 'Stacking Ensemble'
            tier = 'advanced'
        elif 'deep_model' in f:
            algo = 'Deep Learning Pipeline'
            tier = 'advanced'
        elif 'robust_model' in f:
            algo = 'Robust Ensemble'
            tier = 'advanced'
        elif 'improved_model' in f:
            algo = 'Improved Ensemble'
            tier = 'advanced'
        elif '_model.' in f:
            algo = 'Standard Pipeline'
            tier = 'baseline'
        else:
            algo_match = re.search(
                r'(AdaBoost|Bagging|DecisionTree|ExtraTrees|GradientBoosting|'
                r'KNN|LogisticRegression|MLP|NaiveBayes|RandomForest|SVM)', f)
            if algo_match:
                algo = algo_match.group(1)
                tier = 'component'

        acc_match = re.search(r'_(\d{2,3})\.joblib$', f.replace('.pt', '.joblib'))
        accuracy = int(acc_match.group(1)) if acc_match else None

        models.append({
            'filename': f,
            'disease': disease,
            'algorithm': algo,
            'size_bytes': size,
            'size_human': _file_size_human(size),
            'modified': datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
            'framework': framework,
            'format': fmt,
            'tier': tier,
            'accuracy_pct': accuracy,
        })
    return models


def _scan_architectures():
    """Parse deep_learning_models.py for neural architecture class definitions."""
    arch_file = os.path.join(MODELS_DIR, 'deep_learning_models.py')
    architectures = []
    if not os.path.isfile(arch_file):
        return architectures
    try:
        with open(arch_file, 'r') as fh:
            content = fh.read()
        classes = re.findall(r'class\s+(\w+)\s*\(', content)
        for cls in classes:
            desc = ''
            if 'CNN' in cls or 'Conv' in cls:
                desc = 'Convolutional neural network for spatial EEG feature extraction'
            elif 'Transformer' in cls or 'Attention' in cls:
                desc = 'Attention-based architecture for temporal EEG sequence modeling'
            elif 'LSTM' in cls or 'GRU' in cls or 'RNN' in cls:
                desc = 'Recurrent architecture for sequential EEG signal processing'
            elif 'Net' in cls:
                desc = 'Specialized neural network architecture for EEG classification'
            elif 'Ensemble' in cls:
                desc = 'Multi-model ensemble combining diverse architectures'
            else:
                desc = 'Custom neural architecture for clinical EEG analysis'
            architectures.append({
                'class_name': cls,
                'framework': 'PyTorch',
                'description': desc,
            })
    except Exception:
        pass
    return architectures


def _load_model_performance(cur):
    """Load prediction performance from analyses table."""
    rows = cur.execute(
        'SELECT patient_id, disease, predicted_label, confidence, '
        'result_json FROM analyses'
    ).fetchall()
    perf = []
    for r in rows:
        result = {}
        try:
            result = json.loads(r[4]) if r[4] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        metrics = result.get('prediction', {}).get('model_metrics', {})
        perf.append({
            'patient_id': r[0],
            'disease': r[1],
            'predicted_label': r[2],
            'confidence': r[3],
            'accuracy': metrics.get('accuracy_mean'),
            'f1': metrics.get('f1_mean'),
            'auc': metrics.get('auc_mean'),
            'sensitivity': metrics.get('sensitivity_mean'),
            'specificity': metrics.get('specificity_mean'),
        })
    return perf


def _load_disease_distribution(cur):
    """Load patient disease distribution."""
    rows = cur.execute(
        'SELECT disease, COUNT(*) as n FROM patients GROUP BY disease'
    ).fetchall()
    return [{'disease': r[0] or 'Unknown', 'count': r[1]} for r in rows]


# ── Public API ────────────────────────────────────────────────────────────

def foundation_models_overview():
    con = _conn()
    cur = con.cursor()
    perf = _load_model_performance(cur)
    disease_dist = _load_disease_distribution(cur)
    con.close()

    prod_models = _scan_production_models()
    saved_models = _scan_saved_models()
    architectures = _scan_architectures()

    all_models = prod_models + saved_models
    total_size = sum(m['size_bytes'] for m in all_models)
    diseases_covered = sorted(set(m['disease'] for m in all_models))
    frameworks = dict(Counter(m['framework'] for m in all_models))
    tiers = dict(Counter(m.get('tier', 'unknown') for m in all_models))
    algorithms = dict(Counter(m.get('algorithm', 'Unknown')
                              for m in saved_models if m.get('algorithm')))

    avg_conf = (sum(p['confidence'] for p in perf if p['confidence'])
                / len(perf)) if perf else 0

    return {
        'generated_at': _now_iso(),
        'kpis': {
            'total_models': len(all_models),
            'production_models': len(prod_models),
            'experimental_models': len(saved_models),
            'neural_architectures': len(architectures),
            'diseases_covered': len(diseases_covered),
            'total_model_size': _file_size_human(total_size),
            'total_predictions': len(perf),
            'avg_prediction_confidence': round(avg_conf, 3),
        },
        'disease_coverage': diseases_covered,
        'disease_patient_distribution': disease_dist,
        'framework_distribution': [
            {'framework': k, 'count': v} for k, v in sorted(frameworks.items())
        ],
        'tier_distribution': [
            {'tier': k, 'count': v} for k, v in sorted(tiers.items())
        ],
        'algorithm_distribution': sorted(
            [{'algorithm': k, 'count': v} for k, v in algorithms.items()],
            key=lambda x: -x['count']
        )[:12],
        'production_models': prod_models,
        'architectures': architectures,
    }


def foundation_models_breakdown():
    con = _conn()
    cur = con.cursor()
    perf = _load_model_performance(cur)
    con.close()

    prod_models = _scan_production_models()
    saved_models = _scan_saved_models()

    disease_models = defaultdict(list)
    for m in saved_models:
        disease_models[m['disease']].append(m)

    disease_summaries = []
    for disease in sorted(disease_models.keys()):
        models = disease_models[disease]
        prod = [m for m in prod_models if m['disease'] == disease]
        accs = [m['accuracy_pct'] for m in models if m.get('accuracy_pct')]
        best_acc = max(accs) if accs else None
        total_size = sum(m['size_bytes'] for m in models)
        algos = sorted(set(m['algorithm'] for m in models))
        tiers_d = dict(Counter(m['tier'] for m in models))
        disease_summaries.append({
            'disease': disease,
            'total_variants': len(models),
            'production_model': bool(prod),
            'best_accuracy_pct': best_acc,
            'algorithms': algos,
            'total_size': _file_size_human(total_size),
            'tier_breakdown': tiers_d,
        })

    disease_perf = defaultdict(list)
    for p in perf:
        disease_perf[p['disease']].append(p)

    prediction_by_disease = []
    for disease, preds in sorted(disease_perf.items()):
        confs = [p['confidence'] for p in preds if p['confidence']]
        avg_c = sum(confs) / len(confs) if confs else 0
        prediction_by_disease.append({
            'disease': disease,
            'total_predictions': len(preds),
            'avg_confidence': round(avg_c, 3),
            'patients': sorted(set(p['patient_id'] for p in preds)),
        })

    top_models = sorted(
        [m for m in saved_models if m.get('accuracy_pct')],
        key=lambda x: -(x.get('accuracy_pct') or 0)
    )[:20]

    largest_models = sorted(saved_models, key=lambda x: -x['size_bytes'])[:10]

    return {
        'generated_at': _now_iso(),
        'disease_model_summaries': disease_summaries,
        'prediction_performance_by_disease': prediction_by_disease,
        'top_models_by_accuracy': top_models,
        'largest_models': largest_models,
        'all_saved_models': saved_models,
    }


def definitions():
    return {
        'sections': [
            {
                'title': 'Foundation Model Concepts',
                'items': [
                    {'term': 'Foundation Model',
                     'definition': 'A pre-trained model that serves as the base for clinical decision support. In this project, disease-specific classifiers trained on EEG features form the foundation for seizure detection, disease classification, and clinical prediction.'},
                    {'term': 'Production Model',
                     'definition': 'A validated, deployed model in models/ used for live patient predictions. These are the canonical classifiers loaded by the prediction pipeline.'},
                    {'term': 'Experimental Model',
                     'definition': 'A model variant in saved_models/ produced during training experiments. Includes diverse algorithms (RandomForest, SVM, stacking ensembles) tested for each disease.'},
                    {'term': 'Model Tier',
                     'definition': 'Classification of model maturity: production (deployed), advanced (stacking/deep), component (single algorithm), baseline (initial pipeline), experimental (research).'},
                    {'term': 'Disease Coverage',
                     'definition': 'The set of neurological/psychiatric conditions for which trained classifiers exist: epilepsy, alzheimer, depression, parkinson, schizophrenia, stress, autism.'},
                ],
            },
            {
                'title': 'Model Architectures',
                'items': [
                    {'term': 'Random Forest',
                     'definition': 'Ensemble of decision trees using bagging. Strong baseline for EEG feature classification with built-in feature importance.'},
                    {'term': 'Stacking Ensemble',
                     'definition': 'Meta-learner combining predictions from multiple base models (RF, SVM, GBM, etc.) for improved accuracy and robustness.'},
                    {'term': 'Ultra Stacking',
                     'definition': 'Extended stacking with more base models and cross-validated meta-learning. Largest model files but highest potential accuracy.'},
                    {'term': 'Deep Learning Pipeline',
                     'definition': 'Neural network models (CNN, Transformer, LSTM) for direct EEG signal processing, defined in deep_learning_models.py.'},
                    {'term': 'Gradient Boosting',
                     'definition': 'Sequential ensemble that builds trees to correct residual errors. Effective for structured EEG feature data with complex interactions.'},
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'IEC 62304 (Software Lifecycle)',
                     'definition': 'Foundation model catalog supports software configuration management by documenting all model versions, their provenance, and deployment status.'},
                    {'term': 'FDA AI/ML PCCP',
                     'definition': 'Model registry enables predetermined change control by tracking which models are in production vs experimental, supporting modification impact assessment.'},
                    {'term': 'ILAE Classification',
                     'definition': 'Disease-specific models aligned with ILAE seizure/epilepsy classification standards for clinically meaningful predictions.'},
                    {'term': 'ISO 14971 (Risk Management)',
                     'definition': 'Model tier classification supports risk analysis by distinguishing validated production models from experimental variants.'},
                    {'term': 'EU AI Act (Transparency)',
                     'definition': 'Complete model catalog with architecture details, training dates, and performance metrics supports transparency requirements for high-risk AI systems.'},
                ],
            },
            {
                'title': 'Model Governance',
                'items': [
                    {'term': 'Model Versioning',
                     'definition': 'Each model file includes a timestamp in its filename, enabling version tracking and rollback. Production models are the latest validated versions.'},
                    {'term': 'Model Provenance',
                     'definition': 'Training metadata (date, algorithm, accuracy) embedded in filenames and tracked through the model registry for full audit trail.'},
                    {'term': 'Model Retirement',
                     'definition': 'Process for deprecating and removing models that are superseded by better-performing variants or no longer clinically relevant.'},
                    {'term': 'Validation Protocol',
                     'definition': 'Models must pass cross-validation accuracy thresholds and clinical review before promotion from experimental to production tier.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Retrain on Updated Data',
                     'definition': 'When model drift is detected or new patient data is available, retrain foundation models to maintain accuracy and relevance.'},
                    {'term': 'Ensemble Diversification',
                     'definition': 'Add new base model architectures to stacking ensembles to improve robustness against distribution shift.'},
                    {'term': 'Cross-Site Validation',
                     'definition': 'Validate foundation models against data from independent clinical sites (e.g., Bonn dataset) to confirm generalizability.'},
                    {'term': 'Automated Monitoring',
                     'definition': 'Integrate with model monitoring dashboard to automatically flag performance degradation and trigger retraining workflows.'},
                ],
            },
        ],
    }
