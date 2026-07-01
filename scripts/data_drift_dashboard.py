"""Data Drift Dashboard — feature drift monitoring, PSI/KS analysis, schema drift,
threshold alerts, historical drift trends.

Aggregates data from:
- jobs/reports/drift_latest.json (latest drift run: PSI + KS per feature)
- data/clinical.db transaction_log (drift monitor events over time)
- scripts/drift_monitor.py thresholds for display
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')
REPORTS_DIR = os.path.join(BASE, 'jobs', 'reports')

PSI_HIGH = 0.25
PSI_MODERATE = 0.1
KS_ALPHA = 0.05


def _load_drift_latest():
    """Load the most recent drift report."""
    path = os.path.join(REPORTS_DIR, 'drift_latest.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_drift_events():
    """Load drift monitoring events from transaction_log."""
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log WHERE component = 'drift' "
        "ORDER BY ts_utc DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _severity_color(severity):
    if severity == 'high':
        return '#ef4444'
    if severity == 'moderate':
        return '#f59e0b'
    return '#10b981'


def _psi_severity(psi):
    if psi >= PSI_HIGH:
        return 'high'
    if psi >= PSI_MODERATE:
        return 'moderate'
    return 'low'


def overview():
    """KPI-level summary for the Data Drift dashboard."""
    drift = _load_drift_latest()
    events = _load_drift_events()

    if not drift or not drift.get('available'):
        return {
            'available': False,
            'message': 'No drift data available. Run scripts/drift_monitor.py first.'
        }

    top = drift.get('top_drift', [])
    n_features = drift.get('n_features', 0)
    n_high = drift.get('n_high_drift', 0)
    frac = drift.get('frac_drifted', 0)
    verdict = drift.get('verdict', 'unknown')

    # Severity distribution from top_drift
    severity_counts = {'high': 0, 'moderate': 0, 'low': 0}
    psi_values = []
    ks_values = []
    for feat in top:
        sev = feat.get('severity', _psi_severity(feat.get('psi', 0)))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        psi_values.append(feat.get('psi', 0))
        ks_values.append(feat.get('ks_stat', 0))

    avg_psi = sum(psi_values) / len(psi_values) if psi_values else 0
    avg_ks = sum(ks_values) / len(ks_values) if ks_values else 0
    max_psi = max(psi_values) if psi_values else 0
    max_ks = max(ks_values) if ks_values else 0

    # Worst feature
    worst = top[0] if top else None

    # Health score: 100 - (frac_drifted * 100)
    health_score = round((1 - frac) * 100, 1)

    return {
        'available': True,
        'run_at': drift.get('run_at_local', ''),
        'disease': drift.get('disease', ''),
        'verdict': verdict,
        'health_score': health_score,
        'n_features': n_features,
        'n_reference': drift.get('n_reference', 0),
        'n_live': drift.get('n_live', 0),
        'n_high_drift': n_high,
        'frac_drifted': frac,
        'severity_counts': severity_counts,
        'avg_psi': round(avg_psi, 4),
        'avg_ks': round(avg_ks, 4),
        'max_psi': round(max_psi, 4),
        'max_ks': round(max_ks, 4),
        'worst_feature': worst.get('feature', '') if worst else '',
        'worst_psi': worst.get('psi', 0) if worst else 0,
        'n_drift_events': len(events),
        'method': drift.get('method', ''),
        'interpretation': drift.get('interpretation', ''),
        'thresholds': drift.get('thresholds', {'psi_high': PSI_HIGH, 'psi_moderate': PSI_MODERATE}),
        'kpis': [
            {'label': 'Health Score', 'value': f'{health_score}%',
             'color': '#10b981' if health_score >= 80 else '#f59e0b' if health_score >= 50 else '#ef4444'},
            {'label': 'Verdict', 'value': verdict,
             'color': '#ef4444' if 'SEVERE' in verdict.upper() else '#f59e0b' if 'MODERATE' in verdict.upper() else '#10b981'},
            {'label': 'Features Monitored', 'value': str(n_features)},
            {'label': 'High Drift', 'value': str(n_high),
             'color': '#ef4444' if n_high > 0 else '#10b981'},
            {'label': 'Drift Fraction', 'value': f'{frac*100:.1f}%',
             'color': '#ef4444' if frac > 0.5 else '#f59e0b' if frac > 0.2 else '#10b981'},
            {'label': 'Mean PSI', 'value': f'{avg_psi:.2f}',
             'color': '#ef4444' if avg_psi >= PSI_HIGH else '#f59e0b' if avg_psi >= PSI_MODERATE else '#10b981'},
            {'label': 'Reference Samples', 'value': str(drift.get('n_reference', 0))},
            {'label': 'Live Samples', 'value': str(drift.get('n_live', 0))},
        ]
    }


def breakdown():
    """Detailed feature-level drift data, event history, severity distribution."""
    drift = _load_drift_latest()
    events = _load_drift_events()

    if not drift or not drift.get('available'):
        return {'available': False}

    top = drift.get('top_drift', [])

    # Feature-level PSI chart data (sorted by PSI descending)
    feature_psi = []
    for feat in top:
        psi = feat.get('psi', 0)
        sev = feat.get('severity', _psi_severity(psi))
        feature_psi.append({
            'feature': feat.get('feature', ''),
            'psi': round(psi, 4),
            'ks_stat': round(feat.get('ks_stat', 0), 4),
            'ks_p': feat.get('ks_p', 1),
            'severity': sev,
            'color': _severity_color(sev),
            'ks_significant': feat.get('ks_p', 1) < KS_ALPHA,
        })

    # Severity distribution for pie chart
    sev_dist = {'high': 0, 'moderate': 0, 'low': 0}
    for f in feature_psi:
        sev_dist[f['severity']] = sev_dist.get(f['severity'], 0) + 1
    severity_chart = [
        {'name': 'High', 'value': sev_dist['high'], 'color': '#ef4444'},
        {'name': 'Moderate', 'value': sev_dist['moderate'], 'color': '#f59e0b'},
        {'name': 'Low', 'value': sev_dist['low'], 'color': '#10b981'},
    ]

    # Top-10 worst features for bar chart
    top10 = feature_psi[:10]

    # KS statistic chart (top 10)
    ks_chart = [{'feature': f['feature'], 'ks_stat': f['ks_stat'], 'color': f['color']}
                for f in top10]

    # Drift event timeline from transaction_log
    event_timeline = []
    for ev in events:
        detail = ev.get('detail', '')
        # Parse frac from detail string like "SEVERE drift frac=1.0 high=47/47"
        frac_val = None
        high_val = None
        if 'frac=' in detail:
            try:
                frac_val = float(detail.split('frac=')[1].split(' ')[0])
            except (ValueError, IndexError):
                pass
        if 'high=' in detail:
            try:
                high_str = detail.split('high=')[1].split('/')[0]
                high_val = int(high_str)
            except (ValueError, IndexError):
                pass
        event_timeline.append({
            'ts': ev.get('ts_local', ev.get('ts_utc', '')),
            'detail': detail,
            'frac_drifted': frac_val,
            'n_high': high_val,
        })

    # PSI threshold reference lines
    thresholds = [
        {'name': 'High (PSI >= 0.25)', 'value': PSI_HIGH, 'color': '#ef4444'},
        {'name': 'Moderate (PSI >= 0.1)', 'value': PSI_MODERATE, 'color': '#f59e0b'},
    ]

    return {
        'available': True,
        'feature_psi': feature_psi,
        'top10_psi': top10,
        'ks_chart': ks_chart,
        'severity_chart': severity_chart,
        'event_timeline': event_timeline,
        'thresholds': thresholds,
        'n_features_total': len(feature_psi),
        'n_ks_significant': sum(1 for f in feature_psi if f['ks_significant']),
    }


def definitions():
    """Definitions tab for Data Drift dashboard."""
    return {
        'sections': [
            {
                'title': 'Population Stability Index (PSI)',
                'items': [
                    {'term': 'PSI', 'definition': 'Measures how much a feature distribution has shifted between a reference (training) dataset and a live (inference) dataset. Calculated by binning the reference and live distributions and summing (actual% - expected%) * ln(actual%/expected%).'},
                    {'term': 'PSI < 0.1', 'definition': 'No significant drift. Model predictions are stable.'},
                    {'term': '0.1 <= PSI < 0.25', 'definition': 'Moderate drift. Monitor closely; consider retraining.'},
                    {'term': 'PSI >= 0.25', 'definition': 'Severe drift. Feature distribution has shifted significantly. Model reliability is compromised.'},
                ]
            },
            {
                'title': 'Kolmogorov-Smirnov (KS) Test',
                'items': [
                    {'term': 'KS Statistic', 'definition': 'Maximum absolute difference between the empirical CDFs of two distributions. Ranges from 0 (identical) to 1 (completely different).'},
                    {'term': 'KS p-value', 'definition': 'Probability that the observed KS statistic would occur if the two distributions were identical. p < 0.05 indicates statistically significant drift.'},
                    {'term': 'Significance threshold', 'definition': 'alpha = 0.05. Features with p < 0.05 are flagged as statistically drifted.'},
                ]
            },
            {
                'title': 'Drift Severity',
                'items': [
                    {'term': 'High', 'definition': 'PSI >= 0.25. Major distributional shift detected. Immediate action recommended.'},
                    {'term': 'Moderate', 'definition': '0.1 <= PSI < 0.25. Noticeable shift. Schedule investigation.'},
                    {'term': 'Low', 'definition': 'PSI < 0.1. Stable. No action needed.'},
                ]
            },
            {
                'title': 'Reference vs Live Data',
                'items': [
                    {'term': 'Reference dataset', 'definition': 'The training data distribution used when the model was built. Serves as the baseline.'},
                    {'term': 'Live dataset', 'definition': 'The current inference/production data. Compared against the reference to detect drift.'},
                    {'term': 'Train/serve skew', 'definition': 'When live data differs from training data, model confidence may not reflect true accuracy.'},
                ]
            },
            {
                'title': 'Feature Categories',
                'items': [
                    {'term': 'EEG Power Bands', 'definition': 'delta, theta, alpha, beta, gamma — spectral power in standard frequency bands.'},
                    {'term': 'Time-domain', 'definition': 'mean, std, rms, max, min, line_length, zero_crossings, slope_changes — amplitude statistics.'},
                    {'term': 'Complexity', 'definition': 'spectral_entropy, lz_complexity, hurst_exponent, dfa_alpha, peak_ratio — nonlinear dynamics.'},
                    {'term': 'Hjorth Parameters', 'definition': 'activity (variance), mobility (mean frequency), complexity (bandwidth) — compact signal descriptors.'},
                ]
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'IEC 62304', 'definition': 'Medical device software lifecycle standard. Drift monitoring is part of post-market surveillance.'},
                    {'term': 'FDA AI/ML Framework', 'definition': 'Requires monitoring for dataset drift as part of the Predetermined Change Control Plan (PCCP).'},
                    {'term': 'EU AI Act', 'definition': 'High-risk AI systems must implement post-deployment monitoring including input data drift detection.'},
                    {'term': 'Clinical impact', 'definition': 'Undetected drift can cause silent model degradation — predictions become unreliable without visible errors.'},
                ]
            },
        ]
    }
