"""Deep Learning Dashboard — model architectures, training history, and
accuracy metrics for epilepsy EEG seizure detection.

Aggregates real data from:
- data/clinical.db patients + analyses tables
- jobs/reports/training_latest.json (training run history)
- jobs/reports/accuracy_patient_specific.json (per-subject metrics)
- jobs/reports/accuracy_all_options.json (cross-patient accuracy methods)
- models/deep_learning_models.py (architecture class info)
- models/*.joblib (trained model file sizes)
"""

import sqlite3
import json
import os
import math
import re
from collections import defaultdict
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 2) if vals else 0


def _load_json(relpath):
    fp = os.path.join(BASE, relpath)
    if not os.path.exists(fp):
        return {}
    with open(fp) as f:
        return json.load(f)


def _load_model_architectures():
    """Parse deep_learning_models.py to extract class names and descriptions."""
    src_path = os.path.join(BASE, 'models', 'deep_learning_models.py')
    archs = []
    if not os.path.exists(src_path):
        return archs
    with open(src_path) as f:
        content = f.read()

    # Find all nn.Module subclasses
    pattern = r'class\s+(\w+)\((?:nn\.Module|ABC)\):\s*\n\s*"""(.*?)"""'
    matches = re.findall(pattern, content, re.DOTALL)
    for name, docstring in matches:
        # Extract first line of docstring as title
        lines = [l.strip() for l in docstring.strip().split('\n') if l.strip()]
        title = lines[0] if lines else name
        desc = ' '.join(lines[1:]) if len(lines) > 1 else ''
        # Determine architecture type
        if name in ('AttentionBlock',):
            arch_type = 'Utility'
        elif 'CNN3D' in name or 'Conv3d' in name:
            arch_type = '3D CNN'
        elif 'Transformer' in name:
            arch_type = 'Transformer'
        elif 'LSTM' in name:
            arch_type = 'LSTM'
        elif 'CNN' in name or 'Conv1d' in name:
            arch_type = '1D CNN'
        elif 'EEGNet' in name:
            arch_type = 'EEGNet'
        elif 'Graph' in name:
            arch_type = 'Graph Neural Network'
        elif 'Ensemble' in name:
            arch_type = 'Ensemble'
        else:
            arch_type = 'Neural Network'
        # Determine target disease
        if arch_type == 'Utility':
            disease = 'N/A'
        elif 'Alzheimer' in name:
            disease = 'Alzheimer\'s'
        elif 'Parkinson' in name:
            disease = 'Parkinson\'s'
        elif 'Schizophrenia' in name:
            disease = 'Schizophrenia'
        elif 'MultiDisease' in name or 'Ensemble' in name:
            disease = 'Multi-Disease'
        else:
            disease = 'General'

        archs.append({
            'class_name': name,
            'title': title,
            'description': desc.strip(),
            'architecture_type': arch_type,
            'target_disease': disease,
        })

    # Also grab utility classes (AttentionBlock, Dataset, Factory, Trainer)
    util_pattern = r'class\s+(\w+)\((?:nn\.Module|Dataset|ABC)\):\s*\n\s*"""(.*?)"""'
    util_matches = re.findall(util_pattern, content, re.DOTALL)
    seen = {a['class_name'] for a in archs}
    for name, docstring in util_matches:
        if name not in seen:
            lines = [l.strip() for l in docstring.strip().split('\n') if l.strip()]
            title = lines[0] if lines else name
            archs.append({
                'class_name': name,
                'title': title,
                'description': ' '.join(lines[1:]).strip() if len(lines) > 1 else '',
                'architecture_type': 'Utility',
                'target_disease': 'N/A',
            })

    return archs


def _load_model_files():
    """Scan models/ for .joblib files and report sizes."""
    models_dir = os.path.join(BASE, 'models')
    files = []
    if not os.path.isdir(models_dir):
        return files
    for fn in sorted(os.listdir(models_dir)):
        if fn.endswith('.joblib'):
            fp = os.path.join(models_dir, fn)
            size_bytes = os.path.getsize(fp)
            size_kb = round(size_bytes / 1024, 1)
            size_mb = round(size_bytes / (1024 * 1024), 2)
            disease = fn.replace('_model.joblib', '').replace('_', ' ').title()
            files.append({
                'filename': fn,
                'disease': disease,
                'size_bytes': size_bytes,
                'size_kb': size_kb,
                'size_mb': size_mb,
            })
    return files


# ===================================================================
# 1. OVERVIEW
# ===================================================================

def deep_learning_overview():
    conn = _conn()
    cur = conn.cursor()

    # Load patients and analyses from DB
    patients = cur.execute(
        'SELECT patient_id, name, age, gender, disease FROM patients'
    ).fetchall()
    analyses = cur.execute(
        'SELECT patient_id, predicted_label, confidence, signal_quality FROM analyses'
    ).fetchall()
    conn.close()

    total_patients = len(patients)
    total_analyses = len(analyses)

    # Confidence stats from analyses
    confidences = [a[2] for a in analyses if a[2] is not None]
    avg_confidence = _safe_mean(confidences)

    # Load training reports
    training = _load_json('jobs/reports/training_latest.json')
    acc_ps = _load_json('jobs/reports/accuracy_patient_specific.json')
    acc_all = _load_json('jobs/reports/accuracy_all_options.json')

    # Training run history
    training_runs = training.get('results', [])
    total_training_runs = len(training_runs)
    total_training_time = round(sum(r.get('seconds', 0) for r in training_runs), 1)
    successful_runs = sum(1 for r in training_runs if r.get('ok'))

    # Accuracy metrics from patient-specific
    ps_subjects = acc_ps.get('per_subject', [])
    ps_mean_acc = acc_ps.get('mean_accuracy', 0)
    ps_mean_sens = acc_ps.get('mean_sensitivity', 0)
    best_accuracy = max((s.get('accuracy', 0) for s in ps_subjects), default=0)

    # Accuracy methods comparison
    options = acc_all.get('options', {})
    methods_comparison = []
    for key, val in options.items():
        label = key.replace('_', ' ').lstrip('0123456789 ')
        mean_acc = val.get('mean_accuracy', 0)
        methods_comparison.append({
            'method': label,
            'key': key,
            'mean_accuracy': round(mean_acc, 4),
            'mean_accuracy_pct': round(mean_acc * 100, 2),
        })

    # Per-patient accuracy bars
    per_patient_accuracy = []
    for s in ps_subjects:
        per_patient_accuracy.append({
            'subject': s.get('subject', ''),
            'accuracy': round(s.get('accuracy', 0), 4),
            'accuracy_pct': round(s.get('accuracy', 0) * 100, 2),
            'sensitivity': round(s.get('sensitivity', 0), 4),
            'sensitivity_pct': round(s.get('sensitivity', 0) * 100, 2),
            'f1': round(s.get('f1', 0), 4),
            'f1_pct': round(s.get('f1', 0) * 100, 2),
            'n_total': s.get('n_total', 0),
            'n_seizure': s.get('n_seizure', 0),
            'n_test': s.get('n_test', 0),
        })

    # Model architectures
    architectures = _load_model_architectures()
    model_classes = [a for a in architectures if a['architecture_type'] != 'Utility']
    arch_type_counts = {}
    for a in model_classes:
        at = a['architecture_type']
        arch_type_counts[at] = arch_type_counts.get(at, 0) + 1

    arch_type_chart = [{'name': k, 'value': v} for k, v in sorted(arch_type_counts.items())]

    # Model files
    model_files = _load_model_files()
    total_model_size_mb = round(sum(m['size_mb'] for m in model_files), 2)

    # Training timeline
    training_timeline = []
    for r in training_runs:
        training_timeline.append({
            'script': r.get('script', ''),
            'ok': r.get('ok', False),
            'seconds': r.get('seconds', 0),
            'exit_code': r.get('exit_code', -1),
        })

    # Disease distribution from analyses
    disease_dist = {}
    for a in analyses:
        lbl = a[1] or 'Unknown'
        disease_dist[lbl] = disease_dist.get(lbl, 0) + 1
    disease_chart = [{'name': k, 'value': v} for k, v in
                     sorted(disease_dist.items(), key=lambda x: x[1], reverse=True)]

    kpis = {
        'total_models': len(model_files),
        'total_training_runs': total_training_runs,
        'best_accuracy': round(best_accuracy, 4),
        'best_accuracy_pct': round(best_accuracy * 100, 2),
        'mean_accuracy': round(ps_mean_acc, 4),
        'mean_accuracy_pct': round(ps_mean_acc * 100, 2),
        'mean_sensitivity': round(ps_mean_sens, 4),
        'mean_sensitivity_pct': round(ps_mean_sens * 100, 2),
        'total_patients_trained': len(ps_subjects),
        'architecture_count': len(model_classes),
        'total_analyses': total_analyses,
        'avg_confidence': avg_confidence,
        'total_training_time_sec': total_training_time,
        'successful_runs': successful_runs,
        'total_model_size_mb': total_model_size_mb,
        'window_seconds': acc_ps.get('window_seconds', 4),
        'features_count': acc_ps.get('features', 15),
    }

    return {
        'kpis': kpis,
        'methods_comparison': methods_comparison,
        'per_patient_accuracy': per_patient_accuracy,
        'arch_type_chart': arch_type_chart,
        'model_files': model_files,
        'training_timeline': training_timeline,
        'disease_chart': disease_chart,
        'training_meta': {
            'run_at': training.get('run_at_local', training.get('run_at_utc', '')),
            'dataset': training.get('dataset', ''),
            'summary': training.get('summary', ''),
        },
    }


# ===================================================================
# 2. BREAKDOWN
# ===================================================================

def deep_learning_breakdown():
    conn = _conn()
    cur = conn.cursor()

    patients = cur.execute(
        'SELECT patient_id, name, age, gender, disease FROM patients'
    ).fetchall()
    analyses = cur.execute(
        'SELECT patient_id, predicted_label, confidence, signal_quality FROM analyses'
    ).fetchall()
    conn.close()

    acc_ps = _load_json('jobs/reports/accuracy_patient_specific.json')
    acc_all = _load_json('jobs/reports/accuracy_all_options.json')
    training = _load_json('jobs/reports/training_latest.json')

    # Per-patient detailed metrics
    ps_subjects = acc_ps.get('per_subject', [])
    patient_details = []
    for s in ps_subjects:
        patient_details.append({
            'subject': s.get('subject', ''),
            'n_total': s.get('n_total', 0),
            'n_seizure': s.get('n_seizure', 0),
            'n_test': s.get('n_test', 0),
            'accuracy': round(s.get('accuracy', 0), 4),
            'accuracy_pct': round(s.get('accuracy', 0) * 100, 2),
            'f1': round(s.get('f1', 0), 4),
            'f1_pct': round(s.get('f1', 0) * 100, 2),
            'sensitivity': round(s.get('sensitivity', 0), 4),
            'sensitivity_pct': round(s.get('sensitivity', 0) * 100, 2),
            'seizure_ratio': round(s.get('n_seizure', 0) / max(s.get('n_total', 1), 1), 3),
        })

    # Per-model comparison (cross-patient methods)
    options = acc_all.get('options', {})
    model_comparison = []
    for key, val in options.items():
        label = key.replace('_', ' ').lstrip('0123456789 ')
        method_data = val.get('method', label)
        folds = val.get('folds', val.get('per_subject', []))
        per_fold = []
        for fold in folds:
            subj = fold.get('held_out', fold.get('subject', ''))
            per_fold.append({
                'subject': subj,
                'accuracy': round(fold.get('accuracy', 0), 4),
                'accuracy_pct': round(fold.get('accuracy', 0) * 100, 2),
                'f1': round(fold.get('f1', fold.get('accuracy', 0)), 4),
            })
        accs = [f.get('accuracy', 0) for f in folds]
        model_comparison.append({
            'method_key': key,
            'method_label': label,
            'method_description': method_data,
            'mean_accuracy': round(val.get('mean_accuracy', _safe_mean(accs)), 4),
            'mean_accuracy_pct': round(val.get('mean_accuracy', _safe_mean(accs)) * 100, 2),
            'min_accuracy': round(min(accs) if accs else 0, 4),
            'max_accuracy': round(max(accs) if accs else 0, 4),
            'folds': per_fold,
        })

    # Training run detail
    training_runs = training.get('results', [])
    training_detail = []
    for r in training_runs:
        training_detail.append({
            'script': r.get('script', ''),
            'ok': r.get('ok', False),
            'exit_code': r.get('exit_code', -1),
            'seconds': r.get('seconds', 0),
            'tail': r.get('tail', '')[:500],
        })

    # Architecture details
    architectures = _load_model_architectures()

    # Model file details
    model_files = _load_model_files()

    # Per-patient analysis from DB
    analyses_by_patient = defaultdict(list)
    for a in analyses:
        analyses_by_patient[a[0]].append({
            'predicted_label': a[1],
            'confidence': a[2],
            'signal_quality': a[3],
        })

    db_patient_profiles = []
    for p in patients:
        pid = p[0]
        p_analyses = analyses_by_patient.get(pid, [])
        confs = [a['confidence'] for a in p_analyses if a['confidence'] is not None]
        diseases = [a['predicted_label'] for a in p_analyses if a['predicted_label']]
        db_patient_profiles.append({
            'patient_id': pid,
            'name': p[1],
            'age': p[2],
            'gender': p[3],
            'disease': p[4],
            'analysis_count': len(p_analyses),
            'avg_confidence': _safe_mean(confs),
            'predicted_diseases': list(set(diseases)),
        })
    db_patient_profiles.sort(key=lambda x: x['analysis_count'], reverse=True)

    return {
        'patient_details': patient_details,
        'model_comparison': model_comparison,
        'training_detail': training_detail,
        'architectures': architectures,
        'model_files': model_files,
        'db_patient_profiles': db_patient_profiles[:20],
        'benchmark_info': {
            'benchmark': acc_ps.get('benchmark', ''),
            'no_leakage': acc_ps.get('no_leakage', ''),
            'window_seconds': acc_ps.get('window_seconds', 4),
            'stride_seconds': acc_ps.get('stride_seconds', 2),
            'features': acc_ps.get('features', 15),
        },
    }


# ===================================================================
# 3. DEFINITIONS
# ===================================================================

def definitions():
    return {
        'sections': [
            {
                'title': 'Deep Learning Concepts',
                'items': [
                    {'term': 'Convolutional Neural Network (CNN)',
                     'definition': 'A deep learning architecture using learnable convolutional filters to automatically extract spatial and temporal features from input data. 1D CNNs process time-series (EEG, gait sensors), while 3D CNNs process volumetric data (MRI).'},
                    {'term': 'Recurrent Neural Network (LSTM)',
                     'definition': 'Long Short-Term Memory networks capture temporal dependencies in sequential data. Bidirectional LSTMs process sequences in both forward and backward directions, capturing context from past and future time steps.'},
                    {'term': 'Transformer / Self-Attention',
                     'definition': 'Attention-based architecture that weighs the importance of different parts of the input. Multi-head self-attention enables the model to focus on multiple relevant features simultaneously, excelling at capturing long-range dependencies in EEG signals.'},
                    {'term': 'Transfer Learning',
                     'definition': 'Reusing a pre-trained model on a new but related task. In neurological AI, models pre-trained on large EEG corpora can be fine-tuned for patient-specific seizure detection with limited labelled data.'},
                    {'term': 'Patient-Specific vs Cross-Patient Models',
                     'definition': 'Patient-specific models are trained and evaluated on data from a single subject (temporal split), yielding higher accuracy but requiring per-patient calibration. Cross-patient models generalise across subjects but may sacrifice per-individual performance.'},
                ],
            },
            {
                'title': 'Model Architectures',
                'items': [
                    {'term': 'AlzheimerCNN3D',
                     'definition': 'VGG-style 3D CNN with 4 convolutional blocks (32→64→128→256 channels), BatchNorm, ReLU, MaxPool3d, and adaptive average pooling. Processes volumetric brain MRI for Alzheimer\'s classification (3 classes).'},
                    {'term': 'AlzheimerTransformer',
                     'definition': 'Vision Transformer for 3D MRI. Splits volume into patches via Conv3d, adds positional embeddings and CLS token, then passes through 6 transformer blocks (256-dim, 8 heads). Classification via CLS token output.'},
                    {'term': 'ParkinsonVoiceLSTM',
                     'definition': 'Bidirectional 2-layer LSTM (128 hidden units) with learned attention pooling over the temporal dimension. Processes 26-dimensional voice features for binary Parkinson\'s classification.'},
                    {'term': 'ParkinsonGaitCNN',
                     'definition': '1D CNN with 4 blocks (64→128→256→256 channels) processing 6-channel accelerometer/gyroscope gait time-series. Uses adaptive average pooling and fully connected classifier for PD detection.'},
                    {'term': 'SchizophreniaEEGNet',
                     'definition': 'EEGNet-variant with temporal convolution, depthwise spatial convolution, and separable convolution. Processes multi-channel (64-ch) EEG for schizophrenia detection with built-in spatial filtering.'},
                    {'term': 'SchizophreniaGraphNet',
                     'definition': 'Graph neural network treating EEG channels as graph nodes with functional connectivity edges. Applies graph convolution layers to capture inter-channel relationships for schizophrenia classification.'},
                    {'term': 'MultiDiseaseEnsemble',
                     'definition': 'Meta-ensemble combining disease-specific model outputs via learned weighting. Supports multi-task classification across Alzheimer\'s, Parkinson\'s, and Schizophrenia simultaneously.'},
                ],
            },
            {
                'title': 'Training Metrics',
                'items': [
                    {'term': 'Accuracy',
                     'definition': 'Proportion of correctly classified windows (seizure and non-seizure). Reported per-patient for patient-specific models and as leave-one-subject-out for cross-patient models.'},
                    {'term': 'Sensitivity (Recall)',
                     'definition': 'Proportion of actual seizure windows correctly identified (true positives / (true positives + false negatives)). Critical for seizure detection where missing a seizure event has high clinical cost.'},
                    {'term': 'F1 Score',
                     'definition': 'Harmonic mean of precision and sensitivity. Balances false positives (false alarms) against false negatives (missed seizures), especially important with class-imbalanced EEG data.'},
                    {'term': 'Temporal Split (No Leakage)',
                     'definition': 'Training uses early windows, testing uses later windows from the same recording. Prevents data leakage from overlapping windows, ensuring honest generalisation estimates.'},
                    {'term': 'Window Parameters',
                     'definition': 'EEG is segmented into fixed-length windows (default 4 seconds, stride 2 seconds). Each window is independently classified, with overlapping windows providing temporal smoothing.'},
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'IEC 62304',
                     'definition': 'Medical device software lifecycle standard. Deep learning seizure detection software is classified as Class C (could cause serious injury) requiring full traceability from architecture design through validation.'},
                    {'term': 'FDA AI/ML PCCP',
                     'definition': 'Pre-determined Change Control Plan for AI/ML-based Software as a Medical Device. Model retraining, architecture changes, and new patient calibration must follow the locked PCCP protocol with pre-specified performance bounds.'},
                    {'term': 'ILAE Classification',
                     'definition': 'International League Against Epilepsy classification of seizure types (focal, generalised, unknown onset). Deep learning models must align with ILAE 2017 operational classification for clinical reporting.'},
                    {'term': 'ISO 14971 Risk Management',
                     'definition': 'Risk analysis for AI-based seizure detection: false negatives (missed seizures) are severity-critical, false positives (false alarms) affect quality of life. Risk controls include confidence thresholds and human-in-the-loop review.'},
                    {'term': 'EU AI Act (High-Risk)',
                     'definition': 'AI systems for medical diagnosis are classified as high-risk under the EU AI Act, requiring transparency, human oversight, robustness testing, and documentation of training data and model architecture.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Model Recalibration',
                     'definition': 'When patient-specific accuracy drops below threshold (e.g. <90%), recalibrate the model using the latest EEG recordings from that patient. Temperature scaling or Platt scaling can improve probability calibration.'},
                    {'term': 'Architecture Search',
                     'definition': 'If a model architecture underperforms on certain patients, evaluate alternative architectures (CNN vs LSTM vs Transformer) for that patient\'s EEG characteristics. Some patients\' signals favour temporal models over spatial.'},
                    {'term': 'Data Augmentation',
                     'definition': 'Address class imbalance (seizure windows are rare) through time-domain augmentation (jitter, scaling, cropping), frequency-domain augmentation (spectral perturbation), and synthetic seizure generation.'},
                    {'term': 'Ensemble Voting',
                     'definition': 'Combine predictions from multiple model architectures via majority voting or weighted averaging. Cross-validate ensemble weights to prevent overfitting to the calibration set.'},
                    {'term': 'Continuous Monitoring',
                     'definition': 'Deploy drift detection on model inputs (EEG feature distributions) and outputs (confidence scores). Trigger automated retraining when PSI exceeds threshold or accuracy drops on held-out validation windows.'},
                ],
            },
        ],
    }


if __name__ == '__main__':
    import json as _json
    ov = deep_learning_overview()
    print(_json.dumps(ov, indent=2, default=str)[:3000])
