"""Interpretable AI Dashboard — intrinsically interpretable models (decision trees,
logistic regression, rule lists) trained on real EEG feature data from clinical.db.

Distinct from Explainable AI (post-hoc SHAP/LIME): this module builds models whose
internal logic is directly readable — decision rules, regression coefficients, tree
structures — providing glass-box transparency for clinical decision support.

Data sources:
- data/clinical.db analyses table (21 analyses, 47 EEG features per result_json)
- data/clinical.db patients table (40 patients with age/gender/disease)
- models/ directory (.joblib files — production model metadata)
"""

import sqlite3
import json
import os
import math
import warnings
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore', category=UserWarning)

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')
MODELS_DIR = os.path.join(BASE, 'models')

# ── Feature category mapping (47 EEG features) ─────────────────────────────
FEATURE_CATEGORIES = {
    # Time-Domain (19)
    'mean': 'Time-Domain', 'std': 'Time-Domain', 'rms': 'Time-Domain',
    'max': 'Time-Domain', 'min': 'Time-Domain', 'line_length': 'Time-Domain',
    'zero_crossings': 'Time-Domain', 'var': 'Time-Domain', 'ptp': 'Time-Domain',
    'median': 'Time-Domain', 'skewness': 'Time-Domain', 'kurtosis': 'Time-Domain',
    'q25': 'Time-Domain', 'q75': 'Time-Domain', 'mav': 'Time-Domain',
    'crest_factor': 'Time-Domain', 'max_diff': 'Time-Domain',
    'mean_abs_diff': 'Time-Domain', 'std_diff': 'Time-Domain',
    # Spectral (17)
    'delta_power': 'Spectral', 'theta_power': 'Spectral',
    'alpha_power': 'Spectral', 'beta_power': 'Spectral',
    'gamma_power': 'Spectral', 'total_power': 'Spectral',
    'dominant_freq': 'Spectral', 'spectral_entropy': 'Spectral',
    'psd_mean': 'Spectral', 'psd_std': 'Spectral',
    'psd_median': 'Spectral', 'psd_q10': 'Spectral', 'psd_q90': 'Spectral',
    'spectral_bandwidth': 'Spectral', 'spectral_centroid': 'Spectral',
    'spectral_flatness': 'Spectral', 'spectral_rolloff': 'Spectral',
    # Complexity (8)
    'lz_complexity': 'Complexity', 'hurst_exponent': 'Complexity',
    'dfa_alpha': 'Complexity', 'sample_entropy': 'Complexity',
    'permutation_entropy': 'Complexity', 'correlation_dim': 'Complexity',
    'approx_entropy': 'Complexity', 'autocorr': 'Complexity',
    # Hjorth (2)
    'hjorth_mobility': 'Hjorth', 'hjorth_complexity': 'Hjorth',
    # Time-Domain extras
    'slope_changes': 'Time-Domain', 'peak_ratio': 'Time-Domain',
    'trend': 'Time-Domain',
    # Connectivity (4)
    'plv_mean': 'Connectivity', 'plv_std': 'Connectivity',
    'coherence_mean': 'Connectivity', 'coherence_std': 'Connectivity',
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _connect():
    if not os.path.exists(DB):
        return None
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn, name):
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row[0] > 0


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        if math.isfinite(f):
            return f
        return None
    except (ValueError, TypeError):
        return None


def _safe_mean(values):
    vals = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _json_safe(val):
    """Ensure value is JSON-serialisable (no NaN/Infinity)."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def _get_category(feature_name):
    return FEATURE_CATEGORIES.get(feature_name, 'Uncategorised')


def _load_analyses(conn):
    if not _table_exists(conn, 'analyses'):
        return []
    rows = conn.execute(
        "SELECT id, upload_id, patient_id, disease, predicted_label, confidence, "
        "signal_quality, result_json, created_at FROM analyses ORDER BY id"
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        rj = {}
        if d.get('result_json'):
            try:
                rj = json.loads(d['result_json'])
            except (json.JSONDecodeError, TypeError):
                pass
        d['_parsed'] = rj
        d['_features'] = rj.get('features', {})
        d['_prediction'] = rj.get('prediction', {})
        results.append(d)
    return results


def _load_patients(conn):
    if not _table_exists(conn, 'patients'):
        return {}
    rows = conn.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients"
    ).fetchall()
    return {r['patient_id']: dict(r) for r in rows}


def _build_feature_matrix(analyses):
    """Build X (feature matrix) and metadata from analyses with parsed features.
    Returns (X, feature_names, confidences, patient_ids, analysis_ids)."""
    # Collect all feature names consistently
    all_feature_names = set()
    for a in analyses:
        all_feature_names.update(a['_features'].keys())
    feature_names = sorted(all_feature_names)

    X_rows = []
    confidences = []
    patient_ids = []
    analysis_ids = []
    for a in analyses:
        feats = a['_features']
        row = []
        valid = True
        for fname in feature_names:
            v = _safe_float(feats.get(fname))
            if v is None:
                v = 0.0
            row.append(v)
        X_rows.append(row)
        conf = _safe_float(a.get('confidence'))
        confidences.append(conf if conf is not None else 0.0)
        patient_ids.append(a['patient_id'])
        analysis_ids.append(a['id'])

    X = np.array(X_rows, dtype=np.float64)
    return X, feature_names, confidences, patient_ids, analysis_ids


def _build_labels(confidences):
    """Create binary labels from confidence: high (>=median) vs low (<median).
    Used when all analyses share the same predicted class (single-disease dataset)."""
    median_conf = float(np.median(confidences))
    labels = ['high_confidence' if c >= median_conf else 'low_confidence' for c in confidences]
    return labels, median_conf


def _model_files():
    """List .joblib model files with sizes."""
    if not os.path.isdir(MODELS_DIR):
        return []
    results = []
    for fn in sorted(os.listdir(MODELS_DIR)):
        if fn.endswith('.joblib'):
            fp = os.path.join(MODELS_DIR, fn)
            sz = os.path.getsize(fp)
            disease = fn.replace('_model.joblib', '').replace('_', ' ').title()
            results.append({
                'filename': fn,
                'disease': disease,
                'size_bytes': sz,
                'size_kb': round(sz / 1024, 1),
            })
    return results


# ── Decision Tree Surrogate ──────────────────────────────────────────────────

def _build_decision_tree(X, y, feature_names, max_depth=4):
    """Build a shallow decision tree and extract its structure."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42, min_samples_leaf=2)
    dt.fit(X, y_enc)

    # Cross-validation accuracy (limited data, use 3-fold)
    n_splits = min(3, len(set(y_enc)))
    if n_splits >= 2 and len(y_enc) >= 6:
        cv_scores = cross_val_score(dt, X, y_enc, cv=min(3, len(y_enc) // 2), scoring='accuracy')
        cv_accuracy = float(np.mean(cv_scores))
    else:
        cv_accuracy = float(dt.score(X, y_enc))

    # Tree structure
    tree = dt.tree_
    n_nodes = int(tree.node_count)
    depth = int(tree.max_depth)
    n_leaves = int(tree.n_leaves) if hasattr(tree, 'n_leaves') else sum(
        1 for i in range(n_nodes) if tree.children_left[i] == -1
    )

    # Feature importance from the tree
    importances = dt.feature_importances_
    feat_importance = sorted(
        [{'feature': feature_names[i], 'importance': round(float(importances[i]), 4),
          'category': _get_category(feature_names[i])}
         for i in range(len(feature_names)) if importances[i] > 0],
        key=lambda x: -x['importance']
    )

    # Extract decision rules as text
    tree_text = export_text(dt, feature_names=feature_names, max_depth=max_depth)

    # Extract individual rules (paths from root to leaves)
    rules = _extract_rules(dt, feature_names, le.classes_)

    return {
        'model': dt,
        'label_encoder': le,
        'n_nodes': n_nodes,
        'depth': depth,
        'n_leaves': n_leaves,
        'cv_accuracy': round(cv_accuracy, 4),
        'train_accuracy': round(float(dt.score(X, y_enc)), 4),
        'feature_importance': feat_importance,
        'tree_text': tree_text,
        'rules': rules,
        'classes': le.classes_.tolist(),
    }


def _extract_rules(dt, feature_names, class_names):
    """Extract decision rules as human-readable if-then statements."""
    tree = dt.tree_
    rules = []

    def _recurse(node, conditions):
        if tree.children_left[node] == -1:
            # Leaf node
            class_idx = int(np.argmax(tree.value[node][0]))
            class_name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
            samples = int(tree.n_node_samples[node])
            confidence = round(float(tree.value[node][0][class_idx] / tree.n_node_samples[node]), 4)
            rule_str = ' AND '.join(conditions) if conditions else 'DEFAULT'
            rules.append({
                'rule': rule_str,
                'prediction': str(class_name),
                'samples': samples,
                'confidence': confidence,
                'conditions_count': len(conditions),
            })
            return

        feat_idx = tree.feature[node]
        threshold = round(float(tree.threshold[node]), 4)
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feature_{feat_idx}'

        # Left child: feature <= threshold
        left_cond = f'{feat_name} <= {threshold}'
        _recurse(tree.children_left[node], conditions + [left_cond])

        # Right child: feature > threshold
        right_cond = f'{feat_name} > {threshold}'
        _recurse(tree.children_right[node], conditions + [right_cond])

    _recurse(0, [])
    return rules


# ── Logistic Regression ──────────────────────────────────────────────────────

def _build_logistic_regression(X, y, feature_names):
    """Build a logistic regression model and extract coefficients."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_scaled, y_enc)

    # Cross-validation
    if len(set(y_enc)) >= 2 and len(y_enc) >= 6:
        cv_scores = cross_val_score(lr, X_scaled, y_enc, cv=min(3, len(y_enc) // 2), scoring='accuracy')
        cv_accuracy = float(np.mean(cv_scores))
    else:
        cv_accuracy = float(lr.score(X_scaled, y_enc))

    # Coefficients
    coefs = lr.coef_[0] if lr.coef_.ndim > 1 else lr.coef_
    coefficients = sorted(
        [{'feature': feature_names[i], 'coefficient': round(float(coefs[i]), 4),
          'abs_coefficient': round(float(abs(coefs[i])), 4),
          'category': _get_category(feature_names[i]),
          'direction': 'positive' if coefs[i] > 0 else 'negative'}
         for i in range(len(feature_names))],
        key=lambda x: -x['abs_coefficient']
    )

    # Intercept
    intercept = round(float(lr.intercept_[0] if hasattr(lr.intercept_, '__len__') else lr.intercept_), 4)

    return {
        'model': lr,
        'scaler': scaler,
        'label_encoder': le,
        'cv_accuracy': round(cv_accuracy, 4),
        'train_accuracy': round(float(lr.score(X_scaled, y_enc)), 4),
        'intercept': intercept,
        'coefficients': coefficients,
        'top_positive': [c for c in coefficients if c['direction'] == 'positive'][:10],
        'top_negative': [c for c in coefficients if c['direction'] == 'negative'][:10],
        'classes': le.classes_.tolist(),
    }


# ── Public API ───────────────────────────────────────────────────────────────

def interpretable_overview():
    """KPI-level summary: decision tree structure, logistic regression coefficients,
    top decision rules, accuracy comparison (interpretable vs black-box)."""
    conn = _connect()
    if conn is None:
        return {'available': False, 'message': 'Database not found.'}

    try:
        analyses = _load_analyses(conn)
        patients = _load_patients(conn)
    except Exception:
        conn.close()
        return {'available': False, 'message': 'Error reading database.'}
    finally:
        conn.close()

    if not analyses:
        return {'available': False, 'message': 'No analyses found in database.'}

    # Build feature matrix
    X, feature_names, confidences, patient_ids, analysis_ids = _build_feature_matrix(analyses)

    # Build labels (confidence-based binary for single-class dataset)
    labels, median_conf = _build_labels(confidences)

    # Build interpretable models
    dt_result = _build_decision_tree(X, labels, feature_names, max_depth=4)
    lr_result = _build_logistic_regression(X, labels, feature_names)

    # Model files (production black-box models for comparison)
    model_files = _model_files()

    # Accuracy comparison: interpretable vs production models
    # Production models use the stored confidence as proxy for performance
    avg_prod_confidence = round(_safe_mean(confidences), 4)

    # Count total rules
    total_rules = len(dt_result['rules'])

    # Unique features used in decision tree
    dt_features_used = set()
    for rule in dt_result['rules']:
        for cond in rule['rule'].split(' AND '):
            parts = cond.strip().split(' ')
            if len(parts) >= 3:
                dt_features_used.add(parts[0])
    dt_features_used.discard('DEFAULT')

    # KPI cards
    kpis = {
        'total_interpretable_models': 2,
        'decision_tree_depth': dt_result['depth'],
        'decision_tree_accuracy': _json_safe(dt_result['cv_accuracy']),
        'logistic_regression_accuracy': _json_safe(lr_result['cv_accuracy']),
        'total_rules': total_rules,
        'total_features_used': len(dt_features_used),
        'total_analyses': len(analyses),
        'total_patients': len(set(patient_ids)),
    }

    # Top decision rules (sorted by confidence)
    top_rules = sorted(dt_result['rules'], key=lambda r: -r['confidence'])[:10]

    # Accuracy comparison
    accuracy_comparison = [
        {
            'model': 'Decision Tree (depth-4 surrogate)',
            'type': 'interpretable',
            'train_accuracy': _json_safe(dt_result['train_accuracy']),
            'cv_accuracy': _json_safe(dt_result['cv_accuracy']),
            'n_parameters': dt_result['n_nodes'],
        },
        {
            'model': 'Logistic Regression (L2)',
            'type': 'interpretable',
            'train_accuracy': _json_safe(lr_result['train_accuracy']),
            'cv_accuracy': _json_safe(lr_result['cv_accuracy']),
            'n_parameters': len(feature_names) + 1,
        },
    ]

    # Add production model info if available
    for mf in model_files:
        accuracy_comparison.append({
            'model': f'{mf["disease"]} (production .joblib)',
            'type': 'black-box',
            'train_accuracy': None,
            'cv_accuracy': None,
            'n_parameters': None,
            'file_size_kb': mf['size_kb'],
        })

    return {
        'available': True,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'kpis': kpis,
        'decision_tree': {
            'n_nodes': dt_result['n_nodes'],
            'depth': dt_result['depth'],
            'n_leaves': dt_result['n_leaves'],
            'train_accuracy': _json_safe(dt_result['train_accuracy']),
            'cv_accuracy': _json_safe(dt_result['cv_accuracy']),
            'feature_importance': dt_result['feature_importance'][:15],
            'tree_text': dt_result['tree_text'],
            'classes': dt_result['classes'],
        },
        'logistic_regression': {
            'intercept': _json_safe(lr_result['intercept']),
            'train_accuracy': _json_safe(lr_result['train_accuracy']),
            'cv_accuracy': _json_safe(lr_result['cv_accuracy']),
            'top_positive_coefficients': lr_result['top_positive'],
            'top_negative_coefficients': lr_result['top_negative'],
            'classes': lr_result['classes'],
        },
        'top_decision_rules': top_rules,
        'accuracy_comparison': accuracy_comparison,
        'model_files': model_files,
        'label_strategy': 'confidence_median_split',
        'median_confidence': _json_safe(median_conf),
    }


def interpretable_breakdown():
    """Per-disease models, per-patient decision paths, full coefficient list,
    rule extraction, feature importance comparison between models."""
    conn = _connect()
    if conn is None:
        return {'available': False, 'message': 'Database not found.'}

    try:
        analyses = _load_analyses(conn)
        patients = _load_patients(conn)
    except Exception:
        conn.close()
        return {'available': False, 'message': 'Error reading database.'}
    finally:
        conn.close()

    if not analyses:
        return {'available': False, 'message': 'No analyses found.'}

    X, feature_names, confidences, patient_ids, analysis_ids = _build_feature_matrix(analyses)
    labels, median_conf = _build_labels(confidences)

    dt_result = _build_decision_tree(X, labels, feature_names, max_depth=4)
    lr_result = _build_logistic_regression(X, labels, feature_names)

    # ── Per-patient decision paths ─────────────────────────────────────
    dt_model = dt_result['model']
    le = dt_result['label_encoder']
    patient_paths = []

    for idx in range(len(analyses)):
        a = analyses[idx]
        pat_info = patients.get(a['patient_id'], {})
        x_row = X[idx:idx + 1]

        # Decision tree prediction and path
        dt_pred_idx = dt_model.predict(x_row)[0]
        dt_pred_proba = dt_model.predict_proba(x_row)[0]
        dt_pred_class = le.classes_[dt_pred_idx]

        # Get the decision path (node indices traversed)
        node_indicator = dt_model.decision_path(x_row)
        node_indices = node_indicator.indices.tolist()

        # Build human-readable path
        tree = dt_model.tree_
        path_steps = []
        for ni in node_indices:
            if tree.children_left[ni] == -1:
                # Leaf
                leaf_class_idx = int(np.argmax(tree.value[ni][0]))
                leaf_class = le.classes_[leaf_class_idx]
                path_steps.append({
                    'type': 'leaf',
                    'prediction': str(leaf_class),
                    'samples': int(tree.n_node_samples[ni]),
                })
            else:
                feat_idx = tree.feature[ni]
                threshold = round(float(tree.threshold[ni]), 4)
                feat_name = feature_names[feat_idx]
                feat_val = round(float(X[idx, feat_idx]), 4)
                goes_left = feat_val <= threshold
                path_steps.append({
                    'type': 'decision',
                    'feature': feat_name,
                    'threshold': threshold,
                    'value': feat_val,
                    'direction': 'left (<= threshold)' if goes_left else 'right (> threshold)',
                })

        # Logistic regression prediction
        scaler = lr_result['scaler']
        lr_model = lr_result['model']
        x_scaled = scaler.transform(x_row)
        lr_pred_idx = lr_model.predict(x_scaled)[0]
        lr_pred_proba = lr_model.predict_proba(x_scaled)[0]
        lr_pred_class = lr_result['label_encoder'].classes_[lr_pred_idx]

        # Top contributing features for LR (|coefficient * scaled_value|)
        coefs = lr_model.coef_[0] if lr_model.coef_.ndim > 1 else lr_model.coef_
        contributions = []
        for fi in range(len(feature_names)):
            contrib = float(coefs[fi] * x_scaled[0, fi])
            contributions.append({
                'feature': feature_names[fi],
                'contribution': round(contrib, 4),
                'abs_contribution': round(abs(contrib), 4),
            })
        contributions.sort(key=lambda c: -c['abs_contribution'])
        top_contributions = contributions[:5]
        # Clean up helper field
        for c in top_contributions:
            del c['abs_contribution']

        patient_paths.append({
            'analysis_id': a['id'],
            'patient_id': a['patient_id'],
            'name': pat_info.get('name', ''),
            'disease': a.get('disease') or pat_info.get('disease', ''),
            'actual_confidence': round(confidences[idx], 4),
            'actual_label': labels[idx],
            'dt_prediction': str(dt_pred_class),
            'dt_confidence': round(float(max(dt_pred_proba)), 4),
            'dt_path_length': len(node_indices),
            'dt_path': path_steps,
            'lr_prediction': str(lr_pred_class),
            'lr_confidence': round(float(max(lr_pred_proba)), 4),
            'lr_top_contributions': top_contributions,
        })

    # ── All decision rules ─────────────────────────────────────────────
    all_rules = dt_result['rules']

    # ── Full coefficient table ─────────────────────────────────────────
    full_coefficients = lr_result['coefficients']

    # ── Feature importance comparison (DT vs LR) ──────────────────────
    dt_imp = {f['feature']: f['importance'] for f in dt_result['feature_importance']}
    lr_coef_abs = {c['feature']: c['abs_coefficient'] for c in lr_result['coefficients']}

    # Normalise LR coefficients to [0,1] for comparison
    max_lr = max(lr_coef_abs.values()) if lr_coef_abs else 1.0
    if max_lr == 0:
        max_lr = 1.0

    importance_comparison = []
    for fname in feature_names:
        importance_comparison.append({
            'feature': fname,
            'category': _get_category(fname),
            'dt_importance': _json_safe(round(dt_imp.get(fname, 0.0), 4)),
            'lr_importance': _json_safe(round(lr_coef_abs.get(fname, 0.0) / max_lr, 4)),
            'lr_coefficient': _json_safe(round(
                next((c['coefficient'] for c in lr_result['coefficients'] if c['feature'] == fname), 0.0), 4
            )),
            'agreement': 'both_important' if dt_imp.get(fname, 0) > 0.05 and lr_coef_abs.get(fname, 0) / max_lr > 0.1
                         else 'dt_only' if dt_imp.get(fname, 0) > 0.05
                         else 'lr_only' if lr_coef_abs.get(fname, 0) / max_lr > 0.1
                         else 'neither',
        })
    importance_comparison.sort(key=lambda x: -(x['dt_importance'] + x['lr_importance']))

    # ── Per-disease interpretable model summaries ─────────────────────
    disease_groups = defaultdict(list)
    for idx, a in enumerate(analyses):
        dis = a.get('disease', 'Unknown') or 'Unknown'
        disease_groups[dis].append(idx)

    per_disease_models = {}
    for dis, indices in disease_groups.items():
        X_dis = X[indices]
        labels_dis = [labels[i] for i in indices]
        confs_dis = [confidences[i] for i in indices]

        if len(set(labels_dis)) >= 2 and len(indices) >= 4:
            dt_dis = _build_decision_tree(X_dis, labels_dis, feature_names, max_depth=3)
            lr_dis = _build_logistic_regression(X_dis, labels_dis, feature_names)
            per_disease_models[dis] = {
                'n_samples': len(indices),
                'dt_accuracy': _json_safe(dt_dis['cv_accuracy']),
                'dt_depth': dt_dis['depth'],
                'dt_n_rules': len(dt_dis['rules']),
                'dt_top_features': dt_dis['feature_importance'][:5],
                'lr_accuracy': _json_safe(lr_dis['cv_accuracy']),
                'lr_top_positive': lr_dis['top_positive'][:3],
                'lr_top_negative': lr_dis['top_negative'][:3],
                'avg_confidence': _json_safe(round(_safe_mean(confs_dis), 4)),
            }
        else:
            per_disease_models[dis] = {
                'n_samples': len(indices),
                'message': 'Insufficient class diversity for per-disease model',
                'avg_confidence': _json_safe(round(_safe_mean(confs_dis), 4)),
            }

    return {
        'available': True,
        'per_patient_paths': patient_paths,
        'all_decision_rules': all_rules,
        'full_coefficients': full_coefficients,
        'importance_comparison': importance_comparison[:20],
        'per_disease_models': per_disease_models,
    }


def definitions():
    """Definitions tab for the Interpretable AI dashboard."""
    return {
        'sections': [
            {
                'title': 'Interpretable AI Concepts',
                'items': [
                    {
                        'term': 'Intrinsic Interpretability',
                        'definition': (
                            'A model is intrinsically interpretable when its internal decision '
                            'logic can be directly inspected and understood by a human without '
                            'additional explanation tools. Unlike post-hoc explainability (SHAP, '
                            'LIME), the model itself IS the explanation. Examples: decision trees, '
                            'logistic regression, rule lists, scoring systems.'
                        ),
                    },
                    {
                        'term': 'Surrogate Model',
                        'definition': (
                            'A simpler, interpretable model trained to approximate the behaviour '
                            'of a complex black-box model. Here, a shallow decision tree and '
                            'logistic regression are trained on the same EEG features as the '
                            'production model, using confidence-based labels as the target. '
                            'The surrogate reveals which features and thresholds the complex '
                            'model is implicitly relying on.'
                        ),
                    },
                    {
                        'term': 'Decision Rule',
                        'definition': (
                            'A human-readable if-then statement extracted from a decision tree. '
                            'Each rule represents a path from the root to a leaf, composed of '
                            'feature threshold conditions joined by AND. Rules are directly '
                            'auditable by clinicians and can be validated against domain knowledge.'
                        ),
                    },
                    {
                        'term': 'Confidence-Median Split',
                        'definition': (
                            'When all analyses share the same predicted class (e.g., all Epilepsy), '
                            'the dataset is split into high_confidence and low_confidence groups '
                            'based on the median confidence score. This creates a binary '
                            'classification target that reveals which features distinguish '
                            'certain predictions from uncertain ones — critical for clinical trust.'
                        ),
                    },
                ],
            },
            {
                'title': 'Model Types',
                'items': [
                    {
                        'term': 'Decision Tree (Depth-4 Surrogate)',
                        'definition': (
                            'A tree-structured classifier with maximum depth 4, producing at most '
                            '16 leaf nodes. Each internal node tests a single EEG feature against '
                            'a threshold. The tree is trained using the CART algorithm (Gini '
                            'impurity). Depth is limited to ensure the entire tree can be printed '
                            'and reviewed by a clinician in under 2 minutes.'
                        ),
                    },
                    {
                        'term': 'Logistic Regression (L2 Regularised)',
                        'definition': (
                            'A linear classifier that assigns a coefficient to each of the 47 EEG '
                            'features. The sign indicates direction (positive coefficient pushes '
                            'toward high_confidence), and magnitude indicates strength. L2 '
                            'regularisation prevents overfitting on the small dataset. Features '
                            'are standardised (z-scored) before fitting so coefficients are '
                            'comparable across features with different scales.'
                        ),
                    },
                    {
                        'term': 'Feature Importance (Tree-Based)',
                        'definition': (
                            'In a decision tree, feature importance is computed as the total '
                            'reduction in Gini impurity contributed by splits on that feature, '
                            'normalised to sum to 1.0. A feature with importance 0.0 is never '
                            'used in any split — the tree ignores it entirely.'
                        ),
                    },
                    {
                        'term': 'Logistic Regression Coefficients',
                        'definition': (
                            'Each coefficient represents the change in log-odds of the positive '
                            'class per one-standard-deviation increase in that feature (after '
                            'standardisation). Positive coefficients push toward high_confidence; '
                            'negative push toward low_confidence. The intercept is the base '
                            'log-odds when all features are at their population mean.'
                        ),
                    },
                ],
            },
            {
                'title': 'Decision Path Interpretation',
                'items': [
                    {
                        'term': 'Decision Path',
                        'definition': (
                            'The sequence of feature threshold tests a specific patient sample '
                            'traverses from the root of the decision tree to its predicted leaf. '
                            'Each step shows which feature was tested, the threshold, the actual '
                            'feature value, and which branch was taken. This provides a complete, '
                            'auditable trace of WHY the tree made its prediction.'
                        ),
                    },
                    {
                        'term': 'LR Top Contributions',
                        'definition': (
                            'For logistic regression, each feature contributes coefficient * '
                            'standardised_value to the total log-odds. The top contributions '
                            'show which features most influenced the prediction for a specific '
                            'patient. Unlike tree paths (binary splits), LR contributions are '
                            'continuous and additive.'
                        ),
                    },
                    {
                        'term': 'Model Agreement',
                        'definition': (
                            'When both the decision tree and logistic regression predict the same '
                            'class for a patient, confidence in the prediction is higher. '
                            'Disagreement flags cases where interpretability method matters — '
                            'the tree and LR may capture different aspects of the feature space.'
                        ),
                    },
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {
                        'term': 'IEC 62304',
                        'definition': (
                            'Medical device software lifecycle standard. Interpretable models '
                            'satisfy the traceability requirement (clause 5.3) by providing '
                            'directly readable decision logic. For Class C safety classification, '
                            'regulators require that each algorithmic decision can be traced to '
                            'specific feature conditions — decision trees provide this natively.'
                        ),
                    },
                    {
                        'term': 'FDA AI/ML PCCP',
                        'definition': (
                            'The Predetermined Change Control Plan requires documentation of '
                            'model decision boundaries. Interpretable models make this trivial: '
                            'the tree structure IS the decision boundary documentation. When the '
                            'model is updated, the old and new trees can be diff-compared to show '
                            'exactly what changed — a key FDA expectation for SaMD updates.'
                        ),
                    },
                    {
                        'term': 'ILAE Clinical Relevance',
                        'definition': (
                            'International League Against Epilepsy guidelines favour models whose '
                            'features map to known EEG biomarkers. Decision tree splits on '
                            'spectral power bands (delta, theta, alpha) or complexity measures '
                            '(entropy, Hurst exponent) can be validated against published seizure '
                            'signatures, building clinician trust in the AI system.'
                        ),
                    },
                    {
                        'term': 'ISO 14971 Risk Management',
                        'definition': (
                            'Risk management for medical devices requires hazard identification '
                            'at the algorithmic level. Decision rules with small sample counts '
                            'or low confidence identify high-risk prediction regions. Logistic '
                            'regression coefficients close to zero reveal features that add '
                            'complexity without clinical value — candidates for removal to '
                            'reduce risk.'
                        ),
                    },
                    {
                        'term': 'EU AI Act',
                        'definition': (
                            'The EU AI Act classifies medical AI as high-risk (Annex III) and '
                            'mandates transparency and human oversight. Interpretable models '
                            'directly satisfy Article 13 (transparency) by providing readable '
                            'decision logic, and Article 14 (human oversight) by enabling '
                            'clinicians to verify and override specific decision rules.'
                        ),
                    },
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {
                        'term': 'Low Tree Accuracy',
                        'definition': (
                            'If the decision tree surrogate has much lower accuracy than the '
                            'black-box model, the production model relies on complex feature '
                            'interactions that a shallow tree cannot capture. Consider: (1) '
                            'increasing tree depth cautiously, (2) using a rule-list (RuleFit) '
                            'instead, (3) accepting partial interpretability and supplementing '
                            'with post-hoc SHAP explanations for complex cases.'
                        ),
                    },
                    {
                        'term': 'Unstable Decision Rules',
                        'definition': (
                            'If decision rules change significantly with small data perturbations, '
                            'the tree is not stable. Bootstrap the tree construction (build 100 '
                            'trees on resampled data) and report only rules that appear in >50% '
                            'of bootstraps. Unstable rules should not be presented to clinicians '
                            'as reliable decision criteria.'
                        ),
                    },
                    {
                        'term': 'Large Coefficients',
                        'definition': (
                            'Logistic regression coefficients with very large absolute values '
                            '(|coef| > 5.0 after standardisation) suggest the model is overfitting '
                            'to noise in the small dataset. Increase L2 regularisation (decrease C '
                            'parameter) or remove highly correlated features. Cross-reference with '
                            'the Explainable AI dashboard\'s feature correlation matrix.'
                        ),
                    },
                    {
                        'term': 'Feature Disagreement Between Models',
                        'definition': (
                            'When the decision tree and logistic regression rank features very '
                            'differently, it indicates the classification boundary has both '
                            'linear and nonlinear components. Features important only in the tree '
                            'contribute through threshold effects; features important only in LR '
                            'contribute through gradual linear trends. Both perspectives should '
                            'be presented to clinicians for a complete picture.'
                        ),
                    },
                    {
                        'term': 'Clinical Validation of Rules',
                        'definition': (
                            'Every decision rule should be reviewed by a neurologist for clinical '
                            'plausibility. Rules involving spectral power thresholds can be '
                            'validated against published EEG norms (e.g., elevated delta power '
                            'in epileptic foci). Rules involving statistical features (kurtosis, '
                            'skewness) should be checked for artefact sensitivity. Document '
                            'clinical validation status per rule.'
                        ),
                    },
                ],
            },
        ],
    }


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== INTERPRETABLE AI OVERVIEW ===')
    ov = interpretable_overview()
    if ov.get('available'):
        print(f"KPIs: {json.dumps(ov['kpis'], indent=2)}")
        print(f"DT depth: {ov['decision_tree']['depth']}, nodes: {ov['decision_tree']['n_nodes']}")
        print(f"DT accuracy: {ov['decision_tree']['cv_accuracy']}")
        print(f"DT top features: {[f['feature'] for f in ov['decision_tree']['feature_importance'][:5]]}")
        print(f"LR accuracy: {ov['logistic_regression']['cv_accuracy']}")
        print(f"LR intercept: {ov['logistic_regression']['intercept']}")
        print(f"Rules: {len(ov['top_decision_rules'])}")
        print(f"Model files: {len(ov['model_files'])}")
        print(f"\nTree text:\n{ov['decision_tree']['tree_text'][:500]}")
    else:
        print(ov)

    print('\n=== INTERPRETABLE AI BREAKDOWN ===')
    bd = interpretable_breakdown()
    if bd.get('available'):
        print(f"Patient paths: {len(bd.get('per_patient_paths', []))}")
        print(f"All rules: {len(bd.get('all_decision_rules', []))}")
        print(f"Full coefficients: {len(bd.get('full_coefficients', []))}")
        print(f"Importance comparison: {len(bd.get('importance_comparison', []))}")
        print(f"Per-disease models: {list(bd.get('per_disease_models', {}).keys())}")
        if bd.get('per_patient_paths'):
            p = bd['per_patient_paths'][0]
            print(f"\nFirst patient path: {p['patient_id']}")
            print(f"  DT: {p['dt_prediction']} (conf={p['dt_confidence']})")
            print(f"  LR: {p['lr_prediction']} (conf={p['lr_confidence']})")
            print(f"  Path steps: {len(p['dt_path'])}")
    else:
        print(bd)

    print('\n=== DEFINITIONS ===')
    defs = definitions()
    for sec in defs['sections']:
        print(f"  {sec['title']}: {len(sec['items'])} items")
