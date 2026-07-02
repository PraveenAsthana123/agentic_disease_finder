"""Time-Series AI Dashboard — EEG time-series analysis, spectral decomposition,
stationarity metrics, complexity measures, and temporal feature extraction.

Maps clinical.db tables to time-series AI concepts:
- analyses            -> 47+ statistical/spectral/complexity features per EEG recording
- uploads             -> source EEG files with sampling rates and durations
- seizure_diary       -> temporal seizure event sequences
- assessments         -> longitudinal clinical assessment time-series
- transaction_log     -> pipeline event time-series
"""

import json
import sqlite3
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Feature categories for EEG time-series analysis
FEATURE_CATEGORIES = {
    'Statistical': ['mean', 'std', 'var', 'min', 'max', 'median', 'ptp',
                    'skewness', 'kurtosis', 'q25', 'q75', 'rms', 'mav'],
    'Spectral': ['delta_power', 'theta_power', 'alpha_power', 'beta_power',
                 'gamma_power', 'total_power', 'dominant_freq', 'spectral_entropy',
                 'spectral_flatness', 'spectral_centroid', 'spectral_bandwidth',
                 'spectral_rolloff', 'psd_mean', 'psd_std', 'psd_median',
                 'psd_q10', 'psd_q90', 'peak_ratio'],
    'Complexity': ['approx_entropy', 'sample_entropy', 'hurst_exponent',
                   'dfa_alpha', 'lz_complexity'],
    'Temporal': ['line_length', 'zero_crossings', 'mean_abs_diff', 'std_diff',
                 'max_diff', 'hjorth_mobility', 'hjorth_complexity', 'autocorr',
                 'slope_changes', 'trend', 'crest_factor'],
}

BAND_LABELS = {
    'delta_power': 'Delta (0.5-4 Hz)',
    'theta_power': 'Theta (4-8 Hz)',
    'alpha_power': 'Alpha (8-13 Hz)',
    'beta_power': 'Beta (13-30 Hz)',
    'gamma_power': 'Gamma (30-100 Hz)',
}

QUALITY_COLORS = {
    'Good': '#10b981',
    'Fair': '#f59e0b',
    'Poor': '#ef4444',
}


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _parse_features(result_json_str):
    """Extract the features dict from an analysis result_json."""
    if not result_json_str:
        return {}
    try:
        data = json.loads(result_json_str)
        return data.get('features', {})
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_analysis_meta(result_json_str):
    """Extract analysis metadata (channels, sampling_rate, duration, band_power)."""
    if not result_json_str:
        return {}
    try:
        data = json.loads(result_json_str)
        analysis = data.get('analysis', {})
        return {
            'n_channels': analysis.get('n_channels', 0),
            'n_samples': analysis.get('n_samples', 0),
            'sampling_rate': analysis.get('sampling_rate', 0),
            'duration_seconds': analysis.get('duration_seconds', 0),
            'flat_channels': analysis.get('flat_channels', 0),
            'band_power_relative': analysis.get('band_power_relative', {}),
        }
    except (json.JSONDecodeError, TypeError):
        return {}


def _load_analyses():
    return _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, "
        "confidence, signal_quality, result_json, created_at "
        "FROM analyses ORDER BY created_at"
    )


def _load_uploads():
    return _db_query(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at"
    )


def _load_seizure_events():
    return _db_query(
        "SELECT id, patient_id, event_date, event_time, duration_sec, "
        "severity, trigger, created_at "
        "FROM seizure_diary ORDER BY event_date"
    )


def _load_assessments():
    return _db_query(
        "SELECT id, patient_id, instrument, score, max_score, level, "
        "created_at FROM assessments ORDER BY created_at"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log ORDER BY ts_utc"
    )


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Time-Series AI dashboard."""
    analyses = _load_analyses()
    uploads = _load_uploads()
    seizures = _load_seizure_events()
    assessments = _load_assessments()
    pipeline = _load_pipeline_events()

    if not analyses:
        return {
            'available': False,
            'message': 'No EEG analysis data available. '
                       'Upload and analyze EEG files first.',
        }

    total_analyses = len(analyses)

    # Extract features from all analyses
    all_features = []
    all_meta = []
    for a in analyses:
        feats = _parse_features(a.get('result_json'))
        meta = _parse_analysis_meta(a.get('result_json'))
        if feats:
            all_features.append(feats)
        if meta:
            all_meta.append(meta)

    # Unique patients
    patient_ids = set(a['patient_id'] for a in analyses if a.get('patient_id'))
    patients_analyzed = len(patient_ids)

    # Total EEG duration (hours)
    total_duration_hrs = round(
        sum(m.get('duration_seconds', 0) for m in all_meta) / 3600, 1
    )

    # Total samples processed
    total_samples = sum(m.get('n_samples', 0) for m in all_meta)

    # Signal quality distribution
    quality_counts = Counter(
        a.get('signal_quality', 'Unknown') for a in analyses
    )

    # Mean confidence
    confidences = [a['confidence'] for a in analyses if a.get('confidence')]
    mean_confidence = _avg(confidences)

    # Feature count (unique features extracted)
    feature_names = set()
    for feats in all_features:
        feature_names.update(feats.keys())
    total_features = len(feature_names)

    # Mean sampling rate
    sampling_rates = [m['sampling_rate'] for m in all_meta if m.get('sampling_rate')]
    mean_sr = _avg(sampling_rates)

    # --- Chart data ---

    # Band power distribution (from analyses' band_power_relative)
    band_totals = {b: [] for b in BAND_LABELS}
    for m in all_meta:
        bp = m.get('band_power_relative', {})
        for band_key in BAND_LABELS:
            raw_key = band_key.replace('_power', '')
            if raw_key in bp:
                band_totals[band_key].append(bp[raw_key])
    band_power_chart = [
        {
            'band': BAND_LABELS[bk],
            'key': bk,
            'mean_power': _avg(vals),
            'n': len(vals),
        }
        for bk, vals in band_totals.items()
        if vals
    ]

    # Signal quality distribution (pie)
    quality_distribution = [
        {
            'quality': q,
            'count': cnt,
            'color': QUALITY_COLORS.get(q, '#6b7280'),
        }
        for q, cnt in sorted(quality_counts.items())
        if cnt > 0
    ]

    # Feature category breakdown (bar)
    category_feature_counts = []
    for cat_name, cat_features in FEATURE_CATEGORIES.items():
        present = sum(1 for f in cat_features if f in feature_names)
        category_feature_counts.append({
            'category': cat_name,
            'features_extracted': present,
            'features_possible': len(cat_features),
        })

    # Complexity metrics summary (radar-like)
    complexity_summary = []
    for feat_name in FEATURE_CATEGORIES['Complexity']:
        vals = [f.get(feat_name) for f in all_features if f.get(feat_name) is not None]
        if vals:
            complexity_summary.append({
                'metric': feat_name.replace('_', ' ').title(),
                'key': feat_name,
                'mean': _avg(vals),
                'min': round(min(vals), 4),
                'max': round(max(vals), 4),
                'n': len(vals),
            })

    # Daily analysis activity (line)
    daily_counts = Counter()
    for a in analyses:
        day = (a.get('created_at') or '')[:10]
        if day:
            daily_counts[day] += 1
    daily_activity = [
        {'date': day, 'count': cnt}
        for day, cnt in sorted(daily_counts.items())
    ]

    # Seizure temporal distribution
    seizure_daily = Counter()
    for s in seizures:
        day = s.get('event_date', '')
        if day:
            seizure_daily[day] += 1
    seizure_timeline = [
        {'date': day, 'events': cnt}
        for day, cnt in sorted(seizure_daily.items())
    ]

    return {
        'available': True,
        'total_analyses': total_analyses,
        'patients_analyzed': patients_analyzed,
        'total_features': total_features,
        'total_duration_hrs': total_duration_hrs,
        'total_samples': total_samples,
        'mean_confidence': mean_confidence,
        'mean_sampling_rate': mean_sr,
        'seizure_events': len(seizures),
        'assessment_count': len(assessments),
        'pipeline_events': len(pipeline),
        'band_power_chart': band_power_chart,
        'quality_distribution': quality_distribution,
        'category_feature_counts': category_feature_counts,
        'complexity_summary': complexity_summary,
        'daily_activity': daily_activity,
        'seizure_timeline': seizure_timeline,
        'kpis': [
            {'label': 'EEG Analyses', 'value': str(total_analyses)},
            {'label': 'Patients Analyzed', 'value': str(patients_analyzed)},
            {'label': 'Features Extracted', 'value': str(total_features),
             'sub': f'{len(all_features)} recordings'},
            {'label': 'Total Duration', 'value': f'{total_duration_hrs}h',
             'sub': f'{total_samples:,} samples'},
            {'label': 'Mean Confidence', 'value': f'{mean_confidence:.1%}',
             'color': '#10b981' if mean_confidence >= 0.8 else '#f59e0b' if mean_confidence >= 0.6 else '#ef4444'},
            {'label': 'Sampling Rate', 'value': f'{mean_sr:.0f} Hz'},
            {'label': 'Seizure Events', 'value': str(len(seizures))},
            {'label': 'Pipeline Events', 'value': str(len(pipeline))},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed time-series breakdown — per-recording features, per-patient profiles,
    band power analysis, seizure temporal patterns, pipeline events."""
    analyses = _load_analyses()
    uploads = _load_uploads()
    seizures = _load_seizure_events()
    pipeline = _load_pipeline_events()

    if not analyses:
        return {'available': False}

    upload_map = {u['id']: u for u in uploads}

    # --- Recording inventory ---
    recording_inventory = []
    for a in analyses:
        feats = _parse_features(a.get('result_json'))
        meta = _parse_analysis_meta(a.get('result_json'))
        upload = upload_map.get(a.get('upload_id'), {})

        recording_inventory.append({
            'id': a['id'],
            'patient_id': a['patient_id'],
            'file_name': upload.get('file_name', ''),
            'disease': a.get('disease', ''),
            'predicted_label': a.get('predicted_label', ''),
            'confidence': a.get('confidence'),
            'signal_quality': a.get('signal_quality', ''),
            'n_channels': meta.get('n_channels', 0),
            'sampling_rate': meta.get('sampling_rate', 0),
            'duration_seconds': meta.get('duration_seconds', 0),
            'duration_display': _format_duration(meta.get('duration_seconds', 0)),
            'n_features': len(feats),
            'spectral_entropy': feats.get('spectral_entropy'),
            'hurst_exponent': feats.get('hurst_exponent'),
            'sample_entropy': feats.get('sample_entropy'),
            'dominant_freq': feats.get('dominant_freq'),
            'created_at': a.get('created_at'),
        })

    # --- Feature matrix (per recording, per category) ---
    feature_matrix = []
    for a in analyses:
        feats = _parse_features(a.get('result_json'))
        if not feats:
            continue
        row = {
            'id': a['id'],
            'patient_id': a['patient_id'],
        }
        for cat_name, cat_keys in FEATURE_CATEGORIES.items():
            cat_vals = {}
            for k in cat_keys:
                if k in feats:
                    cat_vals[k] = round(feats[k], 6) if isinstance(feats[k], float) else feats[k]
            row[cat_name.lower()] = cat_vals
        feature_matrix.append(row)

    # --- Band power per recording ---
    band_power_details = []
    for a in analyses:
        meta = _parse_analysis_meta(a.get('result_json'))
        bp = meta.get('band_power_relative', {})
        if bp:
            band_power_details.append({
                'id': a['id'],
                'patient_id': a['patient_id'],
                'delta': bp.get('delta'),
                'theta': bp.get('theta'),
                'alpha': bp.get('alpha'),
                'beta': bp.get('beta'),
                'gamma': bp.get('gamma'),
            })

    # --- Patient profiles ---
    patient_analyses = {}
    for a in analyses:
        pid = a['patient_id']
        patient_analyses.setdefault(pid, []).append(a)

    patient_profiles = []
    for pid in sorted(patient_analyses.keys()):
        pa = patient_analyses[pid]
        n_recordings = len(pa)
        qualities = [a.get('signal_quality', 'Unknown') for a in pa]
        confs = [a['confidence'] for a in pa if a.get('confidence')]

        all_feats = [_parse_features(a.get('result_json')) for a in pa]
        all_feats = [f for f in all_feats if f]

        # Aggregate key metrics
        spectral_entropies = [f['spectral_entropy'] for f in all_feats if 'spectral_entropy' in f]
        hurst_vals = [f['hurst_exponent'] for f in all_feats if 'hurst_exponent' in f]
        sample_entropies = [f['sample_entropy'] for f in all_feats if 'sample_entropy' in f]

        total_dur = sum(
            _parse_analysis_meta(a.get('result_json')).get('duration_seconds', 0)
            for a in pa
        )

        diseases = list(set(a.get('disease', '') for a in pa if a.get('disease')))

        patient_profiles.append({
            'patient_id': pid,
            'n_recordings': n_recordings,
            'diseases': diseases,
            'mean_confidence': _avg(confs),
            'total_duration': _format_duration(total_dur),
            'total_duration_sec': total_dur,
            'quality_summary': dict(Counter(qualities)),
            'mean_spectral_entropy': _avg(spectral_entropies),
            'mean_hurst': _avg(hurst_vals),
            'mean_sample_entropy': _avg(sample_entropies),
        })

    # --- Seizure inventory ---
    seizure_inventory = [
        {
            'id': s['id'],
            'patient_id': s['patient_id'],
            'event_date': s.get('event_date'),
            'event_time': s.get('event_time'),
            'duration_sec': s.get('duration_sec'),
            'severity': s.get('severity'),
            'trigger': s.get('trigger'),
        }
        for s in seizures
    ]

    # --- Pipeline events ---
    pipeline_log = [
        {
            'id': e['id'],
            'component': e.get('component'),
            'action': e.get('action'),
            'actor': e.get('actor'),
            'detail': (e.get('detail') or '')[:120],
            'ts_utc': e.get('ts_utc'),
        }
        for e in pipeline[-50:]
    ]

    return {
        'available': True,
        'recording_inventory': recording_inventory,
        'feature_matrix': feature_matrix,
        'band_power_details': band_power_details,
        'patient_profiles': patient_profiles,
        'seizure_inventory': seizure_inventory,
        'pipeline_log': pipeline_log,
    }


def _format_duration(seconds):
    """Format seconds as human-readable duration."""
    if not seconds:
        return '0s'
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f'{hours}h {minutes}m'
    if minutes > 0:
        return f'{minutes}m {secs}s'
    return f'{secs}s'


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Time-Series AI concepts, quality metrics, compliance, remediation."""
    return {
        'concepts': [
            {
                'term': 'Time-Series Analysis',
                'definition': 'Statistical and spectral decomposition of temporally ordered '
                              'EEG signals to extract clinically meaningful features for '
                              'seizure detection, brain state classification, and trend monitoring.',
            },
            {
                'term': 'Spectral Decomposition',
                'definition': 'Decomposing EEG signals into frequency bands (delta, theta, alpha, '
                              'beta, gamma) via FFT/Welch method to quantify oscillatory brain activity '
                              'and identify pathological patterns.',
            },
            {
                'term': 'Band Power Analysis',
                'definition': 'Relative and absolute power in canonical EEG frequency bands. '
                              'Delta (0.5-4 Hz) dominates deep sleep; theta (4-8 Hz) relates to '
                              'drowsiness; alpha (8-13 Hz) to relaxed wakefulness; beta (13-30 Hz) '
                              'to active cognition; gamma (30-100 Hz) to binding/perception.',
            },
            {
                'term': 'Spectral Entropy',
                'definition': 'Information-theoretic measure of spectral complexity. Low entropy '
                              'indicates concentrated power (e.g., epileptiform spike); high entropy '
                              'indicates broad, uniform spectral distribution.',
            },
            {
                'term': 'Hurst Exponent',
                'definition': 'Measure of long-range temporal dependence. H > 0.5 indicates '
                              'persistent (trending) behavior; H = 0.5 is random walk; H < 0.5 '
                              'is anti-persistent. Epileptic EEG often shows H > 0.7.',
            },
            {
                'term': 'Sample Entropy',
                'definition': 'Regularity statistic quantifying signal complexity without self-matching. '
                              'Lower values indicate more regular, predictable signals — often seen '
                              'during ictal (seizure) epochs.',
            },
            {
                'term': 'Hjorth Parameters',
                'definition': 'Time-domain descriptors: Activity (signal variance), Mobility '
                              '(mean frequency), and Complexity (bandwidth). Used for real-time '
                              'EEG monitoring and seizure detection.',
            },
            {
                'term': 'Detrended Fluctuation Analysis (DFA)',
                'definition': 'Scaling analysis method that detects long-range correlations in '
                              'nonstationary time series. DFA alpha > 1 indicates strong correlations; '
                              'alpha = 0.5 indicates uncorrelated noise.',
            },
        ],
        'quality_metrics': [
            {
                'metric': 'Signal-to-Noise Ratio',
                'target': '> 20 dB',
                'description': 'Ratio of clean EEG signal power to artifact/noise power.',
            },
            {
                'metric': 'Feature Extraction Coverage',
                'target': '> 95% features per recording',
                'description': 'Percentage of defined features successfully computed per EEG recording.',
            },
            {
                'metric': 'Spectral Resolution',
                'target': '< 0.5 Hz',
                'description': 'Frequency resolution of spectral decomposition (depends on epoch length).',
            },
            {
                'metric': 'Temporal Stationarity',
                'target': 'ADF p < 0.05 per epoch',
                'description': 'Augmented Dickey-Fuller test for stationarity within analysis windows.',
            },
        ],
        'feature_categories': [
            {'name': cat, 'features': feats, 'count': len(feats)}
            for cat, feats in FEATURE_CATEGORIES.items()
        ],
        'compliance': [
            {
                'standard': 'FDA AI/ML SaMD',
                'reference': 'Software as a Medical Device framework for AI-based EEG analysis',
            },
            {
                'standard': 'EU AI Act Article 6',
                'reference': 'High-risk AI system requirements for medical signal processing',
            },
            {
                'standard': 'ISO 14971:2019',
                'reference': 'Risk management for medical devices — time-series feature validation',
            },
            {
                'standard': 'IEC 62304',
                'reference': 'Software life cycle for medical signal analysis pipelines',
            },
            {
                'standard': 'HIPAA',
                'reference': 'Protected health information in EEG recordings and derived features',
            },
        ],
        'remediation': [
            {
                'issue': 'High artifact contamination',
                'strategy': 'Apply ICA-based artifact removal (EOG, EMG, ECG) before feature extraction',
            },
            {
                'issue': 'Insufficient spectral resolution',
                'strategy': 'Increase epoch length or use multitaper spectral estimation',
            },
            {
                'issue': 'Non-stationary segments',
                'strategy': 'Apply windowed analysis with overlap; flag non-stationary epochs',
            },
            {
                'issue': 'Feature drift across recordings',
                'strategy': 'Implement z-score normalization per-patient or per-session baseline',
            },
        ],
    }


if __name__ == '__main__':
    import json as _json
    print('=== OVERVIEW ===')
    ov = overview()
    print(_json.dumps(ov, indent=2, default=str)[:2000])
    print('\n=== BREAKDOWN (summary) ===')
    bd = breakdown()
    print(f"Recordings: {len(bd.get('recording_inventory', []))}")
    print(f"Feature matrix rows: {len(bd.get('feature_matrix', []))}")
    print(f"Patient profiles: {len(bd.get('patient_profiles', []))}")
    print(f"Seizure events: {len(bd.get('seizure_inventory', []))}")
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f"Concepts: {len(df.get('concepts', []))}")
    print(f"Quality metrics: {len(df.get('quality_metrics', []))}")
