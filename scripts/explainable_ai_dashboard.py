"""Explainable AI (XAI) Dashboard — feature importance, counterfactual analysis,
SHAP-like direction analysis, and per-patient explainability profiles from real
EEG analysis data.

Aggregates data from:
- data/clinical.db analyses table (21 analyses, 47 EEG features per result_json)
- data/clinical.db patients table (40 patients with age/gender/disease)
"""

import sqlite3
import json
import os
import math
from datetime import datetime, timezone
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

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
    """Return a DB connection with Row factory, or None if DB missing."""
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
    """Convert value to float, returning None if not possible."""
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
    """Compute mean of a list of numbers, or 0.0 if empty."""
    vals = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _safe_std(values):
    """Compute population std of a list of numbers, or 0.0 if empty."""
    vals = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / max(n - 1, 1)
    return math.sqrt(var) if var > 0 else 0.0


def _pearson(x, y):
    """Compute Pearson correlation between two equal-length lists."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _json_safe(val):
    """Ensure value is JSON-serialisable (no NaN/Infinity)."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return 0.0
    return val


def _get_category(feature_name):
    """Map feature name to its category."""
    return FEATURE_CATEGORIES.get(feature_name, 'Uncategorised')


def _load_analyses(conn):
    """Load all analyses with parsed result_json."""
    if not _table_exists(conn, 'analyses'):
        return []
    rows = conn.execute(
        "SELECT id, patient_id, disease, predicted_label, confidence, "
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
        d['_analysis'] = rj.get('analysis', {})
        d['_prediction'] = rj.get('prediction', {})
        results.append(d)
    return results


def _load_patients(conn):
    """Load patients as dict keyed by patient_id."""
    if not _table_exists(conn, 'patients'):
        return {}
    rows = conn.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients"
    ).fetchall()
    return {r['patient_id']: dict(r) for r in rows}


def _collect_feature_vectors(analyses):
    """Build {feature_name: [values]} and parallel confidence list."""
    feature_vectors = {}
    confidences = []
    for a in analyses:
        feats = a['_features']
        conf = _safe_float(a.get('confidence')) or _safe_float(
            a['_prediction'].get('confidence') if a['_prediction'] else None
        )
        confidences.append(conf if conf is not None else 0.0)
        for fname, val in feats.items():
            f = _safe_float(val)
            if f is not None:
                feature_vectors.setdefault(fname, []).append(f)
    return feature_vectors, confidences


# ── Public API ───────────────────────────────────────────────────────────────

def xai_overview():
    """KPI-level summary: feature importance, category importance, band power,
    confidence distribution, disease-wise importance."""
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

    feature_vectors, confidences = _collect_feature_vectors(analyses)

    total_analyses = len(analyses)
    total_patients = len({a['patient_id'] for a in analyses})
    total_features = len(feature_vectors)
    avg_confidence = round(_safe_mean(confidences), 4)

    # Distinct diseases
    diseases = {a.get('disease', 'Unknown') or 'Unknown' for a in analyses}

    # Signal quality
    sq_vals = []
    for a in analyses:
        sq = _safe_float(a.get('signal_quality')) or _safe_float(
            a['_analysis'].get('signal_quality') if a['_analysis'] else None
        )
        if sq is not None:
            sq_vals.append(sq)
    avg_signal_quality = round(_safe_mean(sq_vals), 4) if sq_vals else 0.0

    # ── Global feature importance ───────────────────────────────────────
    # Importance = |correlation(feature values across analyses, confidence)|
    feature_importance = {}
    for fname, vals in feature_vectors.items():
        if len(vals) == len(confidences) and len(vals) >= 3:
            corr = abs(_pearson(vals, confidences))
            feature_importance[fname] = round(corr, 4)
        else:
            feature_importance[fname] = 0.0

    importance_mean = _safe_mean(list(feature_importance.values()))
    features_above_threshold = sum(
        1 for v in feature_importance.values() if v > importance_mean
    )

    # Top feature
    top_feature = 'N/A'
    if feature_importance:
        top_feature = max(feature_importance, key=feature_importance.get)

    # Sort top-20
    top_20_features = sorted(
        [
            {
                'feature': fname,
                'importance': _json_safe(imp),
                'category': _get_category(fname),
            }
            for fname, imp in feature_importance.items()
        ],
        key=lambda x: -(x['importance'] or 0),
    )[:20]

    # ── Category importance ─────────────────────────────────────────────
    cat_importance = defaultdict(float)
    cat_count = defaultdict(int)
    for fname, imp in feature_importance.items():
        cat = _get_category(fname)
        cat_importance[cat] += imp
        cat_count[cat] += 1

    category_importance = sorted(
        [
            {
                'category': cat,
                'total_importance': round(cat_importance[cat], 4),
                'feature_count': cat_count[cat],
                'avg_importance': round(
                    cat_importance[cat] / max(cat_count[cat], 1), 4
                ),
            }
            for cat in cat_importance
        ],
        key=lambda x: -x['total_importance'],
    )

    # ── Band power contribution ─────────────────────────────────────────
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_power_data = {b: [] for b in band_names}
    for a in analyses:
        bp = a['_analysis'].get('band_power_relative', {})
        if not bp:
            bp = a['_parsed'].get('analysis', {}).get('band_power_relative', {})
        for band in band_names:
            v = _safe_float(bp.get(band))
            if v is not None:
                band_power_data[band].append(v)

    band_power_contribution = [
        {
            'band': band,
            'mean_contribution': round(_safe_mean(band_power_data[band]), 4),
            'sample_count': len(band_power_data[band]),
        }
        for band in band_names
    ]

    # ── Confidence distribution (5 bins) ────────────────────────────────
    valid_confs = [c for c in confidences if c > 0]
    if valid_confs:
        mn, mx = min(valid_confs), max(valid_confs)
        rng = mx - mn if mx > mn else 1.0
        bin_width = rng / 5
        bins = []
        for i in range(5):
            lo = mn + i * bin_width
            hi = mn + (i + 1) * bin_width
            count = sum(1 for c in valid_confs if lo <= c < hi or (i == 4 and c == hi))
            bins.append({
                'bin': f'{round(lo, 2)}-{round(hi, 2)}',
                'count': count,
            })
    else:
        bins = [{'bin': '0-1', 'count': 0}]

    confidence_distribution = bins

    # ── Disease-wise importance ─────────────────────────────────────────
    disease_groups = defaultdict(list)
    for idx, a in enumerate(analyses):
        dis = a.get('disease', 'Unknown') or 'Unknown'
        disease_groups[dis].append(idx)

    disease_wise_importance = {}
    for dis, indices in disease_groups.items():
        # Gather feature values and confidences for this disease subset
        dis_confs = [confidences[i] for i in indices]
        dis_feat_vals = defaultdict(list)
        for i in indices:
            feats = analyses[i]['_features']
            for fname, val in feats.items():
                f = _safe_float(val)
                if f is not None:
                    dis_feat_vals[fname].append(f)

        dis_importance = {}
        for fname, vals in dis_feat_vals.items():
            if len(vals) == len(dis_confs) and len(vals) >= 3:
                corr = abs(_pearson(vals, dis_confs))
                dis_importance[fname] = round(corr, 4)
            else:
                dis_importance[fname] = 0.0

        top5 = sorted(dis_importance.items(), key=lambda x: -x[1])[:5]
        disease_wise_importance[dis] = [
            {'feature': f, 'importance': _json_safe(imp), 'category': _get_category(f)}
            for f, imp in top5
        ]

    # ── KPI cards ───────────────────────────────────────────────────────
    kpis = {
        'total_analyses': total_analyses,
        'total_patients': total_patients,
        'total_features': total_features,
        'avg_confidence': _json_safe(avg_confidence),
        'features_above_threshold': features_above_threshold,
        'top_feature': top_feature,
        'model_diseases': len(diseases),
        'avg_signal_quality': _json_safe(avg_signal_quality),
    }

    return {
        'available': True,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'kpis': kpis,
        'global_feature_importance': top_20_features,
        'category_importance': category_importance,
        'band_power_contribution': band_power_contribution,
        'confidence_distribution': confidence_distribution,
        'disease_wise_importance': disease_wise_importance,
        'feature_categories': FEATURE_CATEGORIES,
    }


def xai_breakdown():
    """Detailed per-patient profiles, per-feature stats, counterfactual analysis,
    feature correlations, and SHAP-like direction analysis."""
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

    feature_vectors, confidences = _collect_feature_vectors(analyses)

    # Pre-compute population stats per feature
    pop_stats = {}
    for fname, vals in feature_vectors.items():
        m = _safe_mean(vals)
        s = _safe_std(vals)
        pop_stats[fname] = {'mean': m, 'std': s, 'min': min(vals), 'max': max(vals)}

    # Feature importance (same as overview)
    feature_importance = {}
    for fname, vals in feature_vectors.items():
        if len(vals) == len(confidences) and len(vals) >= 3:
            feature_importance[fname] = abs(_pearson(vals, confidences))
        else:
            feature_importance[fname] = 0.0
    importance_rank = {}
    for rank, (fname, _) in enumerate(
        sorted(feature_importance.items(), key=lambda x: -x[1]), start=1
    ):
        importance_rank[fname] = rank

    # ── Per-patient explainability profiles ─────────────────────────────
    patient_analyses = defaultdict(list)
    for idx, a in enumerate(analyses):
        patient_analyses[a['patient_id']].append((idx, a))

    per_patient_profiles = []
    for pid, pa_list in sorted(patient_analyses.items()):
        pat_info = patients.get(pid, {})
        # Use the latest analysis for this patient
        _, latest = pa_list[-1]
        feats = latest['_features']
        conf = _safe_float(latest.get('confidence')) or _safe_float(
            latest['_prediction'].get('confidence') if latest['_prediction'] else None
        )
        predicted = latest.get('predicted_label') or latest['_prediction'].get(
            'predicted_label', ''
        )
        sq = latest.get('signal_quality') or (
            latest['_analysis'].get('signal_quality') if latest['_analysis'] else None
        )

        # Top 3 features by absolute z-score from population mean
        zscore_list = []
        for fname, val in feats.items():
            f = _safe_float(val)
            if f is not None and fname in pop_stats:
                st = pop_stats[fname]
                if st['std'] > 0:
                    z = abs(f - st['mean']) / st['std']
                else:
                    z = 0.0
                zscore_list.append({'feature': fname, 'zscore': round(z, 3), 'value': round(f, 4)})

        zscore_list.sort(key=lambda x: -x['zscore'])
        top_3 = zscore_list[:3]

        per_patient_profiles.append({
            'patient_id': pid,
            'name': pat_info.get('name', ''),
            'disease': latest.get('disease') or pat_info.get('disease', ''),
            'predicted_label': predicted,
            'confidence': round(conf, 4) if conf is not None else None,
            'top_3_features': top_3,
            'signal_quality': sq,
        })

    # ── Per-feature statistics ──────────────────────────────────────────
    per_feature_stats = []
    for fname in sorted(pop_stats.keys()):
        st = pop_stats[fname]
        vals = feature_vectors[fname]
        # Patients with extreme values (|z| >= 2.5)
        extreme_patients = []
        for idx, a in enumerate(analyses):
            f = _safe_float(a['_features'].get(fname))
            if f is not None and st['std'] > 0:
                z = abs(f - st['mean']) / st['std']
                if z >= 2.5:
                    extreme_patients.append(a['patient_id'])

        per_feature_stats.append({
            'feature': fname,
            'category': _get_category(fname),
            'mean': round(st['mean'], 4),
            'std': round(st['std'], 4),
            'min': round(st['min'], 4),
            'max': round(st['max'], 4),
            'importance_rank': importance_rank.get(fname, 0),
            'importance': round(feature_importance.get(fname, 0.0), 4),
            'extreme_patients': sorted(set(extreme_patients)),
        })

    # ── Counterfactual analysis ─────────────────────────────────────────
    # For each analysis: find the feature with highest class-separation
    # (difference in means between predicted class and other class),
    # report that feature and the delta needed to flip prediction.
    class_feature_means = defaultdict(lambda: defaultdict(list))
    for a in analyses:
        label = a.get('predicted_label') or a['_prediction'].get('predicted_label', '')
        if not label:
            continue
        for fname, val in a['_features'].items():
            f = _safe_float(val)
            if f is not None:
                class_feature_means[label][fname].append(f)

    # Compute per-class means
    class_means = {}
    for label, feat_vals in class_feature_means.items():
        class_means[label] = {
            fname: _safe_mean(vals) for fname, vals in feat_vals.items()
        }

    labels = list(class_means.keys())

    # If only one observed class, use confidence-based split (high/low) as
    # proxy classes to still produce meaningful counterfactual explanations.
    single_class_mode = len(labels) <= 1
    if single_class_mode:
        median_conf = sorted(confidences)[len(confidences) // 2] if confidences else 0.5
        high_feat = defaultdict(list)
        low_feat = defaultdict(list)
        for idx, a in enumerate(analyses):
            c = confidences[idx]
            target = high_feat if c >= median_conf else low_feat
            for fname, val in a['_features'].items():
                f = _safe_float(val)
                if f is not None:
                    target[fname].append(f)
        class_means['high_confidence'] = {
            fname: _safe_mean(vals) for fname, vals in high_feat.items()
        }
        class_means['low_confidence'] = {
            fname: _safe_mean(vals) for fname, vals in low_feat.items()
        }

    counterfactual_analysis = []
    for idx, a in enumerate(analyses):
        pred_label = a.get('predicted_label') or a['_prediction'].get('predicted_label', '')

        if single_class_mode:
            # Use confidence-based proxy classes
            my_class = 'high_confidence' if confidences[idx] >= median_conf else 'low_confidence'
            other_class = 'low_confidence' if my_class == 'high_confidence' else 'high_confidence'
        else:
            my_class = pred_label
            other_classes = [l for l in labels if l != pred_label]
            if not other_classes or my_class not in class_means:
                continue
            other_class = other_classes[0]

        if my_class not in class_means or other_class not in class_means:
            continue

        best_feature = None
        best_delta = None
        best_separation = 0.0

        for fname, val in a['_features'].items():
            f = _safe_float(val)
            if f is None or fname not in class_means.get(my_class, {}):
                continue
            if fname not in class_means.get(other_class, {}):
                continue

            my_mean = class_means[my_class][fname]
            other_mean = class_means[other_class][fname]
            separation = abs(my_mean - other_mean)

            if separation > best_separation:
                best_separation = separation
                best_feature = fname
                best_delta = round(other_mean - f, 4)

        if best_feature:
            counterfactual_analysis.append({
                'analysis_id': a['id'],
                'patient_id': a['patient_id'],
                'predicted_label': pred_label or my_class,
                'counterfactual_class': other_class,
                'flip_feature': best_feature,
                'flip_feature_category': _get_category(best_feature),
                'current_value': round(_safe_float(a['_features'].get(best_feature)) or 0, 4),
                'delta_needed': _json_safe(best_delta),
                'class_separation': round(best_separation, 4),
            })

    # ── Feature correlation matrix (top-15 pairs) ──────────────────────
    # Compute Pearson correlation between all feature pairs
    feat_names = sorted(feature_vectors.keys())
    correlation_pairs = []
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):
            f1, f2 = feat_names[i], feat_names[j]
            v1 = feature_vectors[f1]
            v2 = feature_vectors[f2]
            if len(v1) == len(v2) and len(v1) >= 3:
                corr = _pearson(v1, v2)
                correlation_pairs.append({
                    'feature_1': f1,
                    'feature_2': f2,
                    'correlation': round(_json_safe(corr) or 0, 4),
                    'abs_correlation': round(abs(corr), 4),
                })

    correlation_pairs.sort(key=lambda x: -x['abs_correlation'])
    top_15_correlations = correlation_pairs[:15]
    # Remove helper field
    for p in top_15_correlations:
        del p['abs_correlation']

    # ── SHAP-like direction analysis ────────────────────────────────────
    # For each top feature, whether higher values push toward Epilepsy or Control
    # When only one observed class, use confidence-based proxy classes
    shap_labels = labels if not single_class_mode else ['high_confidence', 'low_confidence']
    shap_direction = []
    importance_sorted = sorted(feature_importance.items(), key=lambda x: -x[1])
    for fname, imp in importance_sorted[:20]:
        directions = {}
        for label in shap_labels:
            m = class_means.get(label, {}).get(fname)
            if m is not None:
                directions[label] = round(m, 4)

        # Determine direction: which class has higher mean?
        if len(directions) >= 2:
            sorted_labels = sorted(directions.items(), key=lambda x: -x[1])
            push_direction = f'Higher values push toward {sorted_labels[0][0]}'
        elif len(directions) == 1:
            push_direction = f'Only class: {list(directions.keys())[0]}'
        else:
            push_direction = 'Insufficient data'

        shap_direction.append({
            'feature': fname,
            'category': _get_category(fname),
            'importance': round(imp, 4),
            'class_means': directions,
            'direction': push_direction,
        })

    return {
        'available': True,
        'per_patient_profiles': per_patient_profiles,
        'per_feature_stats': per_feature_stats,
        'counterfactual_analysis': counterfactual_analysis,
        'feature_correlation_matrix': top_15_correlations,
        'shap_direction_analysis': shap_direction,
    }


def definitions():
    """Definitions tab for the Explainable AI dashboard."""
    return {
        'sections': [
            {
                'title': 'Explainability Methods',
                'items': [
                    {
                        'term': 'Feature Importance (Correlation-Based)',
                        'definition': (
                            'Each feature\'s importance is computed as the absolute Pearson '
                            'correlation between that feature\'s values across all analyses '
                            'and the model\'s confidence score. Higher correlation indicates '
                            'the feature has a stronger linear relationship with prediction '
                            'confidence, making it more "important" to the model\'s output.'
                        ),
                    },
                    {
                        'term': 'SHAP-Like Direction Analysis',
                        'definition': (
                            'Inspired by SHapley Additive exPlanations (SHAP), this analysis '
                            'determines whether higher or lower values of each feature push '
                            'the prediction toward a specific class (e.g., Epilepsy vs Control). '
                            'Computed by comparing per-class mean feature values — the class '
                            'with the higher mean is the direction that feature "pushes toward".'
                        ),
                    },
                    {
                        'term': 'Counterfactual Analysis',
                        'definition': (
                            'For each analysis, identifies the smallest feature change needed '
                            'to flip the prediction from the current class to the alternative. '
                            'Approximated by finding the feature with the highest class-separation '
                            '(difference in means between predicted and other class) and computing '
                            'the delta from the current value to the other class mean. Helps '
                            'clinicians understand what would need to change for a different '
                            'prediction.'
                        ),
                    },
                    {
                        'term': 'Feature Correlation Analysis',
                        'definition': (
                            'Pairwise Pearson correlations between features across all analyses. '
                            'Highly correlated features may be redundant — removing one could '
                            'simplify the model without losing predictive power. Also reveals '
                            'feature clusters that represent underlying neurophysiological '
                            'dimensions.'
                        ),
                    },
                ],
            },
            {
                'title': 'EEG Feature Categories',
                'items': [
                    {
                        'term': 'Time-Domain',
                        'definition': (
                            'Statistical descriptors of the EEG signal amplitude over time: '
                            'mean, std, rms, max, min, line_length, zero_crossings, variance, '
                            'peak-to-peak, median, skewness, kurtosis, quartiles, mean absolute '
                            'value, crest factor, and differential statistics. These capture '
                            'signal morphology, distribution shape, and amplitude dynamics.'
                        ),
                    },
                    {
                        'term': 'Spectral',
                        'definition': (
                            'Frequency-domain features from FFT/PSD decomposition: band powers '
                            '(delta 0.5-4Hz, theta 4-8Hz, alpha 8-13Hz, beta 13-30Hz, gamma '
                            '30-100Hz), total power, dominant frequency, spectral entropy, PSD '
                            'statistics, bandwidth, centroid, flatness, and rolloff. These reflect '
                            'brain oscillation states and frequency composition.'
                        ),
                    },
                    {
                        'term': 'Complexity',
                        'definition': (
                            'Nonlinear dynamics measures: Lempel-Ziv complexity, Hurst exponent, '
                            'detrended fluctuation analysis (DFA) alpha, sample entropy, '
                            'permutation entropy, correlation dimension, approximate entropy, '
                            'and autocorrelation. These capture signal regularity, self-similarity, '
                            'and information content — often elevated in epileptic EEG.'
                        ),
                    },
                    {
                        'term': 'Hjorth',
                        'definition': (
                            'Hjorth parameters — mobility (mean frequency, proportional to std '
                            'of first derivative) and complexity (bandwidth/frequency spread, '
                            'ratio of mobility of first derivative to signal mobility). Compact '
                            'time-domain descriptors widely used in sleep staging and seizure '
                            'detection.'
                        ),
                    },
                    {
                        'term': 'Connectivity',
                        'definition': (
                            'Inter-channel synchronisation measures: Phase Locking Value (PLV) '
                            'mean/std and coherence mean/std. These capture functional brain '
                            'connectivity patterns — abnormal synchronisation is a hallmark of '
                            'epileptic networks.'
                        ),
                    },
                ],
            },
            {
                'title': 'Interpretation Guide',
                'items': [
                    {
                        'term': 'Reading Feature Importance',
                        'definition': (
                            'Features are ranked by their absolute correlation with prediction '
                            'confidence. An importance of 1.0 means perfect linear correlation; '
                            '0.0 means no linear relationship. Values above the population mean '
                            'importance are considered "above threshold." The top feature is the '
                            'single most influential feature across all analyses.'
                        ),
                    },
                    {
                        'term': 'Confidence Score',
                        'definition': (
                            'The model\'s self-reported probability that its prediction is correct, '
                            'ranging from 0 to 1. A confidence of 0.95 means the model assigns '
                            '95% probability to its predicted class. Low confidence (<0.6) suggests '
                            'the model is uncertain and the prediction should be reviewed by a '
                            'clinician.'
                        ),
                    },
                    {
                        'term': 'Counterfactual Interpretation',
                        'definition': (
                            'The counterfactual delta shows how much a feature would need to change '
                            'to flip the prediction. A small delta means the prediction is sensitive '
                            'to that feature — a slight data quality issue could change the result. '
                            'A large delta means the prediction is robust with respect to that '
                            'feature. Use counterfactuals to identify fragile predictions.'
                        ),
                    },
                    {
                        'term': 'Per-Patient Profiles',
                        'definition': (
                            'Each patient\'s top-3 features (by z-score deviation from population '
                            'mean) show which features are most unusual for that individual. '
                            'High z-scores indicate the patient is an outlier on that feature, '
                            'which may explain unusual predictions or low confidence.'
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
                            'Medical device software lifecycle standard requires traceability '
                            'from clinical requirements to algorithmic decisions. Explainability '
                            'modules provide the mandatory audit trail showing WHY a prediction '
                            'was made, supporting Class B/C software safety classification and '
                            'risk management documentation.'
                        ),
                    },
                    {
                        'term': 'FDA AI/ML PCCP',
                        'definition': (
                            'The Predetermined Change Control Plan requires transparency in '
                            'model decision-making. Feature importance and counterfactual '
                            'explanations constitute the "algorithmic transparency" evidence '
                            'the FDA expects for AI-based Software as a Medical Device (SaMD). '
                            'Changes in top features across model versions must be documented.'
                        ),
                    },
                    {
                        'term': 'ILAE Clinical Relevance',
                        'definition': (
                            'International League Against Epilepsy guidelines emphasise '
                            'interpretable biomarkers for clinical adoption. Mapping AI '
                            'feature importance to known EEG biomarkers (e.g., spike-wave '
                            'complexes via kurtosis, focal slowing via spectral power) builds '
                            'clinician trust and validates that the model captures clinically '
                            'meaningful patterns.'
                        ),
                    },
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {
                        'term': 'Low-Confidence Predictions',
                        'definition': (
                            'When model confidence falls below 0.6, route the prediction to '
                            'human-in-the-loop review. Examine the patient\'s top-3 features '
                            'to understand why the model is uncertain. If key features have '
                            'poor signal quality, re-record the EEG before acting on the '
                            'prediction.'
                        ),
                    },
                    {
                        'term': 'Feature Quality Issues',
                        'definition': (
                            'If an important feature has extreme values (|z| > 3.0) for a '
                            'patient, verify the raw EEG data quality before trusting the '
                            'prediction. Artefacts (muscle, eye movement, electrode noise) '
                            'can distort features and produce misleading explanations. '
                            'Cross-reference with signal quality metrics.'
                        ),
                    },
                    {
                        'term': 'Model Improvement',
                        'definition': (
                            'Use feature importance rankings to guide feature engineering. '
                            'Low-importance features may be candidates for removal to reduce '
                            'model complexity. Features with high importance but low clinical '
                            'interpretability may indicate the model is learning artefacts '
                            'rather than genuine biomarkers — investigate with domain experts.'
                        ),
                    },
                    {
                        'term': 'Counterfactual-Driven Review',
                        'definition': (
                            'When a counterfactual delta is very small (prediction easily '
                            'flipped), flag the case for clinical review. These "borderline" '
                            'cases are where AI assistance is least reliable and human '
                            'expertise is most needed. Document these cases for model '
                            'retraining prioritisation.'
                        ),
                    },
                    {
                        'term': 'Bias Detection via Explanations',
                        'definition': (
                            'If feature importance differs dramatically between demographic '
                            'groups (disease-wise importance), investigate potential bias. '
                            'The model may be relying on different features for different '
                            'populations, which could indicate inequitable performance. '
                            'Cross-reference with the Bias Detection dashboard.'
                        ),
                    },
                ],
            },
        ],
    }


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pprint
    print('=== XAI OVERVIEW ===')
    ov = xai_overview()
    if ov.get('available'):
        print(f"KPIs: {json.dumps(ov['kpis'], indent=2)}")
        print(f"Top-5 features: {ov['global_feature_importance'][:5]}")
        print(f"Categories: {len(ov['category_importance'])}")
        print(f"Band powers: {ov['band_power_contribution']}")
        print(f"Confidence bins: {len(ov['confidence_distribution'])}")
        print(f"Diseases: {list(ov['disease_wise_importance'].keys())}")
    else:
        pprint.pprint(ov)

    print('\n=== XAI BREAKDOWN ===')
    bd = xai_breakdown()
    if bd.get('available'):
        print(f"Patient profiles: {len(bd.get('per_patient_profiles', []))}")
        print(f"Feature stats: {len(bd.get('per_feature_stats', []))}")
        print(f"Counterfactuals: {len(bd.get('counterfactual_analysis', []))}")
        print(f"Correlation pairs: {len(bd.get('feature_correlation_matrix', []))}")
        print(f"SHAP directions: {len(bd.get('shap_direction_analysis', []))}")
        if bd.get('per_patient_profiles'):
            print('\nTop 3 patients:')
            for p in bd['per_patient_profiles'][:3]:
                print(f"  {p['patient_id']}: {p['disease']}, conf={p['confidence']}, "
                      f"top_feats={[f['feature'] for f in p['top_3_features']]}")
    else:
        pprint.pprint(bd)

    print('\n=== DEFINITIONS ===')
    defs = definitions()
    for sec in defs['sections']:
        print(f"  {sec['title']}: {len(sec['items'])} items")
