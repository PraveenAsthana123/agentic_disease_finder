"""Anomaly Detection Dashboard — z-score and IQR-based EEG feature anomaly
detection across patient analyses, with seizure diary cross-referencing and
signal quality analysis.

Aggregates data from:
- data/clinical.db analyses table (21 analyses, 47 EEG features per result_json)
- data/clinical.db seizure_diary table (seizure events for correlation)
"""

import sqlite3
import json
import os
import math
from datetime import datetime, timezone

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
    # Spectral (16)
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
    # Hjorth (2) — named hjorth_* in DB
    'hjorth_mobility': 'Hjorth', 'hjorth_complexity': 'Hjorth',
    # Time-Domain extras
    'slope_changes': 'Time-Domain', 'peak_ratio': 'Time-Domain',
    'trend': 'Time-Domain',
    # Connectivity (4) — may not be present in current data
    'plv_mean': 'Connectivity', 'plv_std': 'Connectivity',
    'coherence_mean': 'Connectivity', 'coherence_std': 'Connectivity',
}

ZSCORE_ANOMALY = 2.5
ZSCORE_SEVERE = 3.0


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


def _load_seizure_diary(conn):
    """Load seizure diary grouped by patient_id."""
    if not _table_exists(conn, 'seizure_diary'):
        return {}
    rows = conn.execute(
        "SELECT patient_id, event_date, duration_sec, severity, trigger, injury "
        "FROM seizure_diary ORDER BY event_date"
    ).fetchall()
    by_patient = {}
    for r in rows:
        pid = r['patient_id']
        by_patient.setdefault(pid, []).append(dict(r))
    return by_patient


def _get_category(feature_name):
    """Map feature name to its category."""
    return FEATURE_CATEGORIES.get(feature_name, 'Uncategorised')


def _collect_feature_vectors(analyses):
    """Build {feature_name: [values]} from all analyses.

    Returns (feature_vectors, feature_by_analysis) where feature_by_analysis
    is a list parallel to analyses with each entry being {feature: value}.
    """
    feature_vectors = {}
    feature_by_analysis = []
    for a in analyses:
        feats = a['_features']
        fba = {}
        for fname, val in feats.items():
            if isinstance(val, (int, float)) and math.isfinite(val):
                feature_vectors.setdefault(fname, []).append(val)
                fba[fname] = val
        feature_by_analysis.append(fba)
    return feature_vectors, feature_by_analysis


def _compute_stats(values):
    """Compute mean, std, Q1, Q3, IQR from a list of numbers."""
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    s = sorted(values)
    q1 = s[n // 4] if n >= 4 else s[0]
    q3 = s[(3 * n) // 4] if n >= 4 else s[-1]
    median = s[n // 2]
    iqr = q3 - q1
    return {
        'mean': mean, 'std': std, 'min': s[0], 'max': s[-1],
        'median': median, 'q1': q1, 'q3': q3, 'iqr': iqr, 'n': n,
    }


def _zscore(value, mean, std):
    """Compute absolute z-score."""
    if std == 0:
        return 0.0
    return abs(value - mean) / std


def _severity_label(z):
    """Map z-score to severity."""
    if z >= ZSCORE_SEVERE:
        return 'Severe'
    if z >= ZSCORE_ANOMALY:
        return 'Moderate'
    if z >= 2.0:
        return 'Mild'
    return 'Normal'


def _is_iqr_outlier(value, q1, q3, iqr):
    """Check if value is outside 1.5*IQR fences."""
    if iqr == 0:
        return False
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return value < lower or value > upper


def _detect_anomalies(analyses, feature_vectors, feature_by_analysis):
    """Run z-score + IQR anomaly detection on every analysis.

    Returns:
        per_analysis: list of dicts with anomaly info per analysis
        per_feature: dict {feature: stats + anomaly list}
    """
    # Pre-compute stats per feature
    feature_stats = {}
    for fname, vals in feature_vectors.items():
        st = _compute_stats(vals)
        if st:
            feature_stats[fname] = st

    per_analysis = []
    per_feature = {fname: {'stats': feature_stats[fname], 'anomalies': [], 'zscores': []}
                   for fname in feature_stats}

    for idx, a in enumerate(analyses):
        fba = feature_by_analysis[idx]
        anomalies = []
        severe_count = 0
        for fname, val in fba.items():
            if fname not in feature_stats:
                continue
            st = feature_stats[fname]
            z = _zscore(val, st['mean'], st['std'])
            iqr_out = _is_iqr_outlier(val, st['q1'], st['q3'], st['iqr'])
            sev = _severity_label(z)
            per_feature[fname]['zscores'].append(z)
            if z >= ZSCORE_ANOMALY or iqr_out:
                anomalies.append({
                    'feature': fname,
                    'value': round(val, 6),
                    'zscore': round(z, 3),
                    'iqr_outlier': iqr_out,
                    'severity': sev,
                    'category': _get_category(fname),
                })
                per_feature[fname]['anomalies'].append({
                    'analysis_id': a['id'],
                    'patient_id': a['patient_id'],
                    'value': round(val, 6),
                    'zscore': round(z, 3),
                })
                if sev == 'Severe':
                    severe_count += 1

        per_analysis.append({
            'analysis_id': a['id'],
            'patient_id': a['patient_id'],
            'created_at': a.get('created_at', ''),
            'confidence': a.get('confidence') or (
                a['_prediction'].get('confidence') if a['_prediction'] else None
            ),
            'signal_quality': a.get('signal_quality') or (
                a['_analysis'].get('signal_quality') if a['_analysis'] else None
            ),
            'disease': a.get('disease', ''),
            'predicted_label': a.get('predicted_label', ''),
            'anomaly_count': len(anomalies),
            'severe_count': severe_count,
            'anomalies': anomalies,
        })

    return per_analysis, per_feature, feature_stats


# ── Public API ───────────────────────────────────────────────────────────────

def anomaly_detection_overview():
    """KPI-level summary: anomaly rates, severity distribution, top features."""
    conn = _connect()
    if conn is None:
        return {'available': False, 'message': 'Database not found.'}

    try:
        analyses = _load_analyses(conn)
        seizure_diary = _load_seizure_diary(conn)
    finally:
        conn.close()

    if not analyses:
        return {'available': False, 'message': 'No analyses found in database.'}

    feature_vectors, feature_by_analysis = _collect_feature_vectors(analyses)
    per_analysis, per_feature, feature_stats = _detect_anomalies(
        analyses, feature_vectors, feature_by_analysis
    )

    total_analyses = len(analyses)
    total_patients = len({a['patient_id'] for a in analyses})
    total_features = len(feature_stats)
    anomalous_analyses = sum(1 for a in per_analysis if a['anomaly_count'] > 0)
    severe_total = sum(a['severe_count'] for a in per_analysis)
    total_anomaly_count = sum(a['anomaly_count'] for a in per_analysis)
    mean_anomalies = round(total_anomaly_count / max(total_analyses, 1), 1)
    anomaly_rate = round(anomalous_analyses / max(total_analyses, 1) * 100, 1)

    # Most anomalous feature
    feat_anomaly_counts = {
        fname: len(info['anomalies']) for fname, info in per_feature.items()
    }
    most_anomalous_feature = (
        max(feat_anomaly_counts, key=feat_anomaly_counts.get)
        if feat_anomaly_counts else 'N/A'
    )

    # Anomaly by category
    cat_counts = {}
    cat_features = {}
    for fname, info in per_feature.items():
        cat = _get_category(fname)
        cat_counts.setdefault(cat, 0)
        cat_features.setdefault(cat, set())
        cat_counts[cat] += len(info['anomalies'])
        cat_features[cat].add(fname)

    anomaly_by_category = sorted([
        {
            'category': cat,
            'anomaly_count': cat_counts.get(cat, 0),
            'feature_count': len(cat_features.get(cat, set())),
            'anomaly_rate': f"{round(cat_counts.get(cat, 0) / max(len(cat_features.get(cat, set())) * total_analyses, 1) * 100, 1)}%",
        }
        for cat in sorted(set(list(cat_counts.keys()) + list(cat_features.keys())))
    ], key=lambda x: -x['anomaly_count'])

    # Severity distribution across all individual anomaly flags
    sev_dist = {'Normal': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0}
    for a in per_analysis:
        for anom in a['anomalies']:
            sev_dist[anom['severity']] = sev_dist.get(anom['severity'], 0) + 1
    # Also count non-anomalous as Normal for the overall distribution
    total_feature_checks = sum(len(fba) for fba in feature_by_analysis)
    sev_dist['Normal'] = total_feature_checks - total_anomaly_count

    anomaly_severity_distribution = [
        {'severity': sev, 'count': cnt}
        for sev, cnt in [('Normal', sev_dist['Normal']),
                         ('Mild', sev_dist['Mild']),
                         ('Moderate', sev_dist['Moderate']),
                         ('Severe', sev_dist['Severe'])]
    ]

    # Top anomalous features
    top_anomalous_features = []
    for fname, info in sorted(per_feature.items(),
                              key=lambda x: -len(x[1]['anomalies'])):
        if not info['anomalies']:
            continue
        zscores = info['zscores']
        mean_z = round(sum(zscores) / max(len(zscores), 1), 2)
        top_anomalous_features.append({
            'feature': fname,
            'category': _get_category(fname),
            'anomaly_count': len(info['anomalies']),
            'mean_zscore': mean_z,
        })
    top_anomalous_features = top_anomalous_features[:10]

    return {
        'available': True,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'kpis': {
            'total_analyses': total_analyses,
            'total_patients': total_patients,
            'total_features_monitored': total_features,
            'anomalous_analyses': anomalous_analyses,
            'anomaly_rate': f'{anomaly_rate}%',
            'severe_anomalies': severe_total,
            'mean_anomalies_per_analysis': mean_anomalies,
            'most_anomalous_feature': most_anomalous_feature,
        },
        'anomaly_by_category': anomaly_by_category,
        'anomaly_severity_distribution': anomaly_severity_distribution,
        'top_anomalous_features': top_anomalous_features,
    }


def anomaly_detection_breakdown():
    """Detailed per-patient, per-feature, timeline, correlation, and signal quality."""
    conn = _connect()
    if conn is None:
        return {'available': False, 'message': 'Database not found.'}

    try:
        analyses = _load_analyses(conn)
        seizure_diary = _load_seizure_diary(conn)
    finally:
        conn.close()

    if not analyses:
        return {'available': False, 'message': 'No analyses found.'}

    feature_vectors, feature_by_analysis = _collect_feature_vectors(analyses)
    per_analysis, per_feature, feature_stats = _detect_anomalies(
        analyses, feature_vectors, feature_by_analysis
    )

    # ── Per-patient anomalies ────────────────────────────────────────────
    patient_agg = {}
    for a in per_analysis:
        pid = a['patient_id']
        if pid not in patient_agg:
            patient_agg[pid] = {
                'patient_id': pid,
                'total_anomalies': 0,
                'severe': 0,
                'anomalous_features': set(),
                'confidences': [],
                'has_seizures': pid in seizure_diary,
            }
        patient_agg[pid]['total_anomalies'] += a['anomaly_count']
        patient_agg[pid]['severe'] += a['severe_count']
        for anom in a['anomalies']:
            patient_agg[pid]['anomalous_features'].add(anom['feature'])
        if a['confidence'] is not None:
            try:
                patient_agg[pid]['confidences'].append(float(a['confidence']))
            except (ValueError, TypeError):
                pass

    per_patient_anomalies = []
    for pid, info in sorted(patient_agg.items(),
                            key=lambda x: -x[1]['total_anomalies']):
        confs = info['confidences']
        avg_conf = round(sum(confs) / len(confs), 3) if confs else None
        per_patient_anomalies.append({
            'patient_id': pid,
            'total_anomalies': info['total_anomalies'],
            'severe': info['severe'],
            'anomalous_features': sorted(info['anomalous_features']),
            'confidence': avg_conf,
            'has_seizures': info['has_seizures'],
        })

    # ── Per-feature stats ────────────────────────────────────────────────
    per_feature_stats = []
    for fname in sorted(per_feature.keys()):
        info = per_feature[fname]
        st = info['stats']
        outlier_patients = sorted({a['patient_id'] for a in info['anomalies']})
        per_feature_stats.append({
            'feature': fname,
            'category': _get_category(fname),
            'mean': round(st['mean'], 6),
            'std': round(st['std'], 6),
            'min': round(st['min'], 6),
            'max': round(st['max'], 6),
            'anomaly_count': len(info['anomalies']),
            'zscore_threshold': ZSCORE_ANOMALY,
            'outlier_patients': outlier_patients,
        })

    # ── Anomaly timeline ─────────────────────────────────────────────────
    anomaly_timeline = []
    for a in per_analysis:
        anomaly_timeline.append({
            'analysis_id': a['analysis_id'],
            'patient_id': a['patient_id'],
            'created_at': a['created_at'],
            'anomaly_count': a['anomaly_count'],
            'severe_count': a['severe_count'],
            'confidence': a['confidence'],
        })

    # ── Feature correlation anomalies ────────────────────────────────────
    # Find pairs of features that tend to be anomalous together
    # Build co-occurrence matrix from analyses
    feature_anomaly_sets = {}  # {analysis_idx: set of anomalous features}
    for idx, a in enumerate(per_analysis):
        anom_feats = {anom['feature'] for anom in a['anomalies']}
        if anom_feats:
            feature_anomaly_sets[idx] = anom_feats

    # Compute pairwise overlap for features with anomalies
    anomalous_feat_names = sorted({
        fname for info in per_feature.values()
        for a in info['anomalies']
        for fname in [a.get('patient_id')]  # just to iterate
    } | {fname for fname, info in per_feature.items() if info['anomalies']})

    # Simple pairwise co-occurrence
    pair_overlap = {}
    feat_list = [f for f in anomalous_feat_names if f in per_feature and per_feature[f]['anomalies']]
    for i in range(len(feat_list)):
        for j in range(i + 1, len(feat_list)):
            f1, f2 = feat_list[i], feat_list[j]
            # Count analyses where both features are anomalous
            overlap = 0
            for idx, anom_set in feature_anomaly_sets.items():
                if f1 in anom_set and f2 in anom_set:
                    overlap += 1
            if overlap > 0:
                # Compute correlation of the raw feature values
                vals1 = feature_vectors.get(f1, [])
                vals2 = feature_vectors.get(f2, [])
                corr = _pearson(vals1, vals2) if len(vals1) == len(vals2) and len(vals1) >= 3 else 0.0
                pair_overlap[(f1, f2)] = {'overlap': overlap, 'correlation': corr}

    feature_correlation_anomalies = sorted([
        {
            'feature_pair': f'{f1} vs {f2}',
            'correlation': round(info['correlation'], 3),
            'anomaly_overlap': info['overlap'],
        }
        for (f1, f2), info in pair_overlap.items()
    ], key=lambda x: -x['anomaly_overlap'])[:15]

    # ── Signal quality analysis ──────────────────────────────────────────
    sq_agg = {}
    for a in per_analysis:
        sq = a.get('signal_quality') or 'unknown'
        if isinstance(sq, (int, float)):
            sq = f'{sq:.1f}' if sq < 10 else str(int(sq))
        sq_agg.setdefault(sq, {'count': 0, 'anomalies': 0, 'patients': set()})
        sq_agg[sq]['count'] += 1
        sq_agg[sq]['anomalies'] += a['anomaly_count']
        sq_agg[sq]['patients'].add(a['patient_id'])

    signal_quality_analysis = sorted([
        {
            'signal_quality': sq,
            'analysis_count': info['count'],
            'anomaly_count': info['anomalies'],
            'patient_ids': sorted(info['patients']),
        }
        for sq, info in sq_agg.items()
    ], key=lambda x: -x['anomaly_count'])

    return {
        'available': True,
        'per_patient_anomalies': per_patient_anomalies,
        'per_feature_stats': per_feature_stats,
        'anomaly_timeline': anomaly_timeline,
        'feature_correlation_anomalies': feature_correlation_anomalies,
        'signal_quality_analysis': signal_quality_analysis,
    }


def anomaly_detection_definitions():
    """Definitions tab for the Anomaly Detection dashboard."""
    return {
        'sections': [
            {
                'title': 'Anomaly Detection Methods',
                'items': [
                    {
                        'term': 'Z-Score',
                        'definition': (
                            'Measures how many standard deviations a value is from the '
                            'population mean. Values with |z| > 2.5 are flagged as anomalies. '
                            'Assumes approximate normality; robust for EEG features with '
                            'sufficient sample sizes.'
                        ),
                    },
                    {
                        'term': 'IQR Method',
                        'definition': (
                            'Interquartile Range method flags values below Q1 - 1.5*IQR or '
                            'above Q3 + 1.5*IQR as outliers. Non-parametric and robust to '
                            'skewed distributions common in spectral power features.'
                        ),
                    },
                    {
                        'term': 'Severity Levels',
                        'definition': (
                            'Normal (|z| < 2.0): within expected range. '
                            'Mild (2.0 <= |z| < 2.5): borderline, monitor. '
                            'Moderate (2.5 <= |z| < 3.0): anomalous, investigate. '
                            'Severe (|z| >= 3.0): extreme outlier, immediate review recommended.'
                        ),
                    },
                    {
                        'term': 'Dual-Method Detection',
                        'definition': (
                            'Anomalies are flagged if EITHER the z-score exceeds 2.5 OR the '
                            'value falls outside IQR fences. This union approach maximises '
                            'sensitivity for clinical safety.'
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
                            'mean, std, rms, max, min, line_length, zero_crossings, var, ptp, '
                            'median, skewness, kurtosis, q25, q75, mav — amplitude-based '
                            'statistics capturing signal morphology and distribution shape.'
                        ),
                    },
                    {
                        'term': 'Spectral',
                        'definition': (
                            'delta_power, theta_power, alpha_power, beta_power, gamma_power, '
                            'total_power, dominant_freq, spectral_entropy, psd_mean, psd_std, '
                            'psd_max, psd_min — frequency-domain features from FFT/PSD '
                            'decomposition reflecting brain oscillation states.'
                        ),
                    },
                    {
                        'term': 'Complexity',
                        'definition': (
                            'lz_complexity, hurst_exponent, dfa_alpha, sample_entropy, '
                            'permutation_entropy, correlation_dim — nonlinear dynamics '
                            'measures capturing signal regularity, self-similarity, and '
                            'information content.'
                        ),
                    },
                    {
                        'term': 'Hjorth',
                        'definition': (
                            'activity (signal variance/power), mobility (mean frequency), '
                            'complexity (bandwidth/frequency spread) — compact time-domain '
                            'descriptors of EEG waveform morphology.'
                        ),
                    },
                    {
                        'term': 'Connectivity',
                        'definition': (
                            'plv_mean, plv_std (Phase Locking Value), coherence_mean, '
                            'coherence_std — inter-channel synchronisation measures '
                            'reflecting functional brain connectivity patterns.'
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
                            'Medical device software lifecycle standard requires continuous '
                            'monitoring of input data quality. Feature-level anomaly detection '
                            'provides granular post-market surveillance evidence for Class C '
                            'software safety classification.'
                        ),
                    },
                    {
                        'term': 'FDA AI/ML PCCP',
                        'definition': (
                            'Predetermined Change Control Plan requires documenting input '
                            'monitoring thresholds. Anomaly detection z-score and IQR '
                            'thresholds constitute pre-specified performance boundaries '
                            'for triggered model review.'
                        ),
                    },
                    {
                        'term': 'Anomaly & Seizure Correlation',
                        'definition': (
                            'Cross-referencing EEG feature anomalies with seizure diary '
                            'entries identifies whether anomalous patterns correspond to '
                            'genuine clinical events (true positives) or data quality '
                            'issues (artefacts). Patients with both anomalies and seizure '
                            'events warrant prioritised clinical review.'
                        ),
                    },
                    {
                        'term': 'Signal Quality Impact',
                        'definition': (
                            'Low signal quality (noise, artefact, poor electrode contact) '
                            'inflates anomaly counts. Stratifying anomalies by signal '
                            'quality separates clinical anomalies from technical artefacts.'
                        ),
                    },
                ],
            },
            {
                'title': 'Remediation',
                'items': [
                    {
                        'term': 'Severe Anomaly Response',
                        'definition': (
                            'Analyses with severe anomalies (|z| >= 3.0) should trigger '
                            'human-in-the-loop review. Flag the patient record for clinician '
                            'inspection and pause automated treatment recommendations.'
                        ),
                    },
                    {
                        'term': 'Recurrent Feature Anomalies',
                        'definition': (
                            'Features that are anomalous across multiple patients may indicate '
                            'a systematic issue (equipment calibration drift, preprocessing '
                            'bug, population shift). Investigate the data pipeline before '
                            'the model.'
                        ),
                    },
                    {
                        'term': 'Patient-Specific Patterns',
                        'definition': (
                            'A single patient with many anomalous features may represent a '
                            'rare phenotype, recording artefact, or data entry error. '
                            'Cross-reference with clinical notes and seizure diary.'
                        ),
                    },
                    {
                        'term': 'Retraining Trigger',
                        'definition': (
                            'When the overall anomaly rate exceeds 20% or severe anomalies '
                            'exceed 5%, evaluate whether the training data distribution '
                            'still represents the production population. Consider '
                            'retraining with updated data.'
                        ),
                    },
                ],
            },
        ],
    }


# ── Utility ──────────────────────────────────────────────────────────────────

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


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(anomaly_detection_overview())
    print('\n=== BREAKDOWN ===')
    bd = anomaly_detection_breakdown()
    # Print summary only
    if bd.get('available'):
        print(f"Patients: {len(bd.get('per_patient_anomalies', []))}")
        print(f"Features: {len(bd.get('per_feature_stats', []))}")
        print(f"Timeline entries: {len(bd.get('anomaly_timeline', []))}")
        print(f"Correlation pairs: {len(bd.get('feature_correlation_anomalies', []))}")
        print(f"Signal quality groups: {len(bd.get('signal_quality_analysis', []))}")
        print('\nTop 5 patients by anomalies:')
        for p in bd['per_patient_anomalies'][:5]:
            print(f"  {p['patient_id']}: {p['total_anomalies']} anomalies, "
                  f"severe={p['severe']}, seizures={p['has_seizures']}")
    else:
        pprint.pprint(bd)
    print('\n=== DEFINITIONS ===')
    defs = anomaly_detection_definitions()
    for sec in defs['sections']:
        print(f"  {sec['title']}: {len(sec['items'])} items")
