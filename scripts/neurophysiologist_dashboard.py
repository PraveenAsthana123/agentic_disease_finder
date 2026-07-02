"""Clinical Neurophysiologist / EEG Reviewer Dashboard — EEG recording
inventory, background rhythm analysis, band power distribution, signal
quality assessment, AI label validation, spectral features from real
clinical.db data.

Maps clinical.db tables to neurophysiology concepts:
- analyses          -> EEG analysis results (band power, signal quality, predictions)
- uploads           -> EEG file uploads (file name, patient, disease)
- seizure_diary     -> Seizure events (ictal correlates)
- transaction_log   -> Pipeline activity
- patients          -> Demographics
"""

import json
import sqlite3
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _pct(part, total):
    if not total:
        return '0%'
    return f'{round(100 * part / total, 1)}%'


# EEG background rhythm classification based on dominant frequency
def _classify_rhythm(dom_freq):
    if dom_freq is None:
        return 'Unknown'
    if dom_freq <= 4:
        return 'Delta (<4 Hz)'
    elif dom_freq <= 8:
        return 'Theta (4-8 Hz)'
    elif dom_freq <= 13:
        return 'Alpha (8-13 Hz)'
    elif dom_freq <= 30:
        return 'Beta (13-30 Hz)'
    else:
        return 'Gamma (>30 Hz)'


# Signal quality grading
QUALITY_COLORS = {
    'Good': '#10b981',
    'Fair': '#f59e0b',
    'Poor': '#ef4444',
    'Unknown': '#94a3b8',
}


def _load_analyses():
    rows = _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, confidence, "
        "signal_quality, result_json, created_at FROM analyses ORDER BY id"
    )
    results = []
    for r in rows:
        rj = _safe_json(r.get('result_json'))
        analysis = rj.get('analysis', {})
        features = rj.get('features', {})
        prediction = rj.get('prediction', {})
        channels = rj.get('channels', [])
        results.append({
            'id': r['id'],
            'upload_id': r.get('upload_id'),
            'patient_id': r['patient_id'],
            'disease': r.get('disease', ''),
            'predicted_label': r.get('predicted_label', ''),
            'confidence': r.get('confidence'),
            'signal_quality': r.get('signal_quality', 'Unknown'),
            'created_at': r.get('created_at', ''),
            'file': rj.get('file', ''),
            'channels': channels,
            'n_channels': analysis.get('n_channels', len(channels)),
            'sampling_rate': analysis.get('sampling_rate'),
            'duration_seconds': analysis.get('duration_seconds'),
            'band_power': analysis.get('band_power_relative', {}),
            'flat_channels': analysis.get('flat_channels', 0),
            'per_channel': analysis.get('per_channel', []),
            'dominant_freq': features.get('dominant_freq'),
            'spectral_entropy': features.get('spectral_entropy'),
            'hjorth_mobility': features.get('hjorth_mobility'),
            'hjorth_complexity': features.get('hjorth_complexity'),
            'approx_entropy': features.get('approx_entropy'),
            'sample_entropy': features.get('sample_entropy'),
            'hurst_exponent': features.get('hurst_exponent'),
            'dfa_alpha': features.get('dfa_alpha'),
            'lz_complexity': features.get('lz_complexity'),
            'kurtosis': features.get('kurtosis'),
            'skewness': features.get('skewness'),
            'autocorr': features.get('autocorr'),
            'class_probabilities': prediction.get('class_probabilities', {}),
        })
    return results


def _load_uploads():
    return _db_query(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY id"
    )


def _load_seizures():
    return _db_query(
        "SELECT patient_id, event_date, duration_sec, severity, "
        "location, motor_signs, injury, aura, trigger, created_at "
        "FROM seizure_diary ORDER BY event_date DESC"
    )


def _load_pipeline_events(limit=50):
    return _db_query(
        "SELECT action, detail, ts_utc as created_at FROM transaction_log "
        "ORDER BY ts_utc DESC LIMIT ?", (limit,)
    )


# ── Overview (summary KPIs + charts) ─────────────────────────────────────

def overview():
    analyses = _load_analyses()
    uploads = _load_uploads()
    seizures = _load_seizures()
    pipeline = _load_pipeline_events()

    total_recordings = len(analyses)
    unique_patients = len(set(a['patient_id'] for a in analyses))

    # Signal quality distribution
    quality_counts = Counter(a['signal_quality'] for a in analyses)
    good_quality = quality_counts.get('Good', 0)
    quality_rate = _pct(good_quality, total_recordings)

    # Band power averages across all recordings
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_avgs = {}
    for band in band_names:
        vals = [a['band_power'].get(band, 0) for a in analyses if a['band_power']]
        band_avgs[band] = _avg(vals)

    # Background rhythm distribution (dominant frequency classification)
    rhythm_counts = Counter(_classify_rhythm(a['dominant_freq']) for a in analyses)

    # AI prediction distribution
    prediction_counts = Counter(a['predicted_label'] for a in analyses if a['predicted_label'])
    avg_confidence = _avg([a['confidence'] for a in analyses if a.get('confidence') is not None])

    # Mean spectral entropy
    entropy_vals = [a['spectral_entropy'] for a in analyses if a.get('spectral_entropy') is not None]
    mean_entropy = _avg(entropy_vals)

    # Flat channel rate
    flat_total = sum(a['flat_channels'] for a in analyses)
    channel_total = sum(a['n_channels'] for a in analyses)

    # Mean duration
    durations = [a['duration_seconds'] for a in analyses if a.get('duration_seconds')]
    mean_duration_hrs = round(_avg(durations) / 3600, 1) if durations else 0

    # Disease distribution
    disease_counts = Counter(a['disease'] for a in analyses if a['disease'])

    # Daily activity (from pipeline events)
    day_counts = Counter()
    for ev in pipeline:
        dt = ev.get('created_at', '')[:10]
        if dt:
            day_counts[dt] += 1
    daily_activity = [{'date': d, 'events': c} for d, c in sorted(day_counts.items())[-14:]]

    return {
        'available': True,
        'title': 'Clinical Neurophysiologist / EEG Reviewer Dashboard',
        'subtitle': (
            f'{total_recordings} recordings \u00b7 {unique_patients} patients \u00b7 '
            f'Signal quality {quality_rate} Good \u00b7 '
            f'Mean confidence {avg_confidence} \u00b7 '
            f'Mean entropy {mean_entropy}'
        ),
        'kpis': [
            {'label': 'EEG Recordings', 'value': total_recordings},
            {'label': 'Unique Patients', 'value': unique_patients},
            {'label': 'Good Signal Quality', 'value': quality_rate},
            {'label': 'Mean AI Confidence', 'value': f'{avg_confidence}'},
            {'label': 'Mean Spectral Entropy', 'value': f'{mean_entropy}'},
            {'label': 'Mean Duration', 'value': f'{mean_duration_hrs}h'},
            {'label': 'Seizure Events', 'value': len(seizures)},
            {'label': 'Flat Channel Rate', 'value': _pct(flat_total, channel_total)},
        ],
        'band_power_distribution': [
            {'band': band.capitalize(), 'power': band_avgs[band]}
            for band in band_names
        ],
        'signal_quality_distribution': [
            {'name': q, 'value': quality_counts.get(q, 0)}
            for q in ['Good', 'Fair', 'Poor', 'Unknown']
            if quality_counts.get(q, 0) > 0
        ],
        'background_rhythm_distribution': [
            {'name': rhythm, 'value': count}
            for rhythm, count in sorted(rhythm_counts.items())
        ],
        'prediction_distribution': [
            {'name': label, 'value': count}
            for label, count in prediction_counts.most_common()
        ],
        'disease_distribution': [
            {'name': d, 'value': c}
            for d, c in disease_counts.most_common()
        ],
        'daily_activity': daily_activity,
    }


# ── Breakdown (detailed tables + chart data) ──────────────────────────────

def breakdown():
    analyses = _load_analyses()
    uploads = _load_uploads()
    seizures = _load_seizures()

    # Upload-file map for cross-reference
    upload_map = {u['id']: u for u in uploads}

    # EEG Recording Inventory
    recording_inventory = []
    for a in analyses:
        upload = upload_map.get(a['upload_id'], {})
        duration_hrs = round(a['duration_seconds'] / 3600, 1) if a.get('duration_seconds') else None
        recording_inventory.append({
            'id': a['id'],
            'patient_id': a['patient_id'],
            'file': a['file'] or upload.get('file_name', ''),
            'disease': a['disease'],
            'n_channels': a['n_channels'],
            'sampling_rate': a['sampling_rate'],
            'duration_hrs': duration_hrs,
            'signal_quality': a['signal_quality'],
            'flat_channels': a['flat_channels'],
            'dominant_freq': a['dominant_freq'],
            'background_rhythm': _classify_rhythm(a['dominant_freq']),
            'predicted_label': a['predicted_label'],
            'confidence': a['confidence'],
            'created_at': a['created_at'],
        })

    # Band Power per Recording (for stacked bar chart)
    band_power_table = []
    for a in analyses:
        bp = a['band_power']
        if bp:
            band_power_table.append({
                'patient_id': a['patient_id'],
                'recording_id': a['id'],
                'delta': round(bp.get('delta', 0), 4),
                'theta': round(bp.get('theta', 0), 4),
                'alpha': round(bp.get('alpha', 0), 4),
                'beta': round(bp.get('beta', 0), 4),
                'gamma': round(bp.get('gamma', 0), 4),
            })

    # Spectral Features Table
    spectral_features = []
    for a in analyses:
        spectral_features.append({
            'patient_id': a['patient_id'],
            'recording_id': a['id'],
            'spectral_entropy': a.get('spectral_entropy'),
            'hjorth_mobility': a.get('hjorth_mobility'),
            'hjorth_complexity': a.get('hjorth_complexity'),
            'approx_entropy': a.get('approx_entropy'),
            'sample_entropy': a.get('sample_entropy'),
            'hurst_exponent': a.get('hurst_exponent'),
            'dfa_alpha': a.get('dfa_alpha'),
            'lz_complexity': a.get('lz_complexity'),
            'kurtosis': a.get('kurtosis'),
            'skewness': a.get('skewness'),
            'autocorr': a.get('autocorr'),
        })

    # AI Label Validation Table
    ai_validation = []
    for a in analyses:
        probs = a.get('class_probabilities', {})
        ai_validation.append({
            'patient_id': a['patient_id'],
            'recording_id': a['id'],
            'predicted_label': a['predicted_label'],
            'confidence': a['confidence'],
            'class_probabilities': probs,
            'signal_quality': a['signal_quality'],
            'review_status': 'Pending',
        })

    # Channel Statistics (aggregated per recording)
    channel_stats = []
    for a in analyses:
        per_ch = a.get('per_channel', [])
        if per_ch:
            ch_stds = [ch.get('std', 0) for ch in per_ch]
            channel_stats.append({
                'patient_id': a['patient_id'],
                'recording_id': a['id'],
                'n_channels': a['n_channels'],
                'channels_listed': len(per_ch),
                'mean_channel_std': round(_avg(ch_stds), 6),
                'max_channel_std': round(max(ch_stds), 6) if ch_stds else 0,
                'flat_channels': a['flat_channels'],
                'channel_names': a['channels'][:8],
            })

    # Seizure correlates
    seizure_log = []
    for s in seizures:
        seizure_log.append({
            'patient_id': s['patient_id'],
            'date': s.get('event_date', ''),
            'duration_sec': s.get('duration_sec'),
            'severity': s.get('severity', ''),
            'location': s.get('location', ''),
            'motor_signs': s.get('motor_signs'),
            'aura': s.get('aura'),
            'trigger': s.get('trigger', ''),
        })

    return {
        'recording_inventory': recording_inventory,
        'band_power_table': band_power_table,
        'spectral_features': spectral_features,
        'ai_validation': ai_validation,
        'channel_stats': channel_stats,
        'seizure_log': seizure_log,
    }


# ── Definitions ───────────────────────────────────────────────────────────

def definitions():
    return [
        {'term': 'Background Rhythm',
         'definition': 'The dominant EEG activity when a patient is awake and relaxed. Normal adult posterior dominant rhythm is 8-13 Hz (alpha band). Slowing suggests encephalopathy or structural lesion.'},
        {'term': 'Interictal Epileptiform Discharges (IEDs)',
         'definition': 'Spikes, sharp waves, or spike-wave complexes occurring between seizures. Their location helps lateralize/localize the epileptogenic zone. Prevalence varies by epilepsy syndrome.'},
        {'term': 'Band Power (Relative)',
         'definition': 'Proportion of total EEG power in each frequency band: Delta (<4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (>30 Hz). Pathological excess in specific bands suggests different conditions.'},
        {'term': 'Signal Quality',
         'definition': 'Assessment of EEG recording fidelity. Good = minimal artifact, adequate impedance. Fair = some muscle/movement artifact. Poor = significant contamination requiring review. Based on flat channels, impedance, and artifact density.'},
        {'term': 'Spectral Entropy',
         'definition': 'Measure of signal complexity/irregularity from the power spectrum. Higher entropy = more complex/irregular activity. Used for anesthesia depth monitoring and seizure detection.'},
        {'term': 'Hjorth Parameters',
         'definition': 'Three signal descriptors: Activity (variance/power), Mobility (mean frequency), Complexity (bandwidth). Computed in time domain, useful for real-time EEG classification.'},
        {'term': 'Hurst Exponent',
         'definition': 'Measure of long-range temporal dependence in EEG. H > 0.5 = persistent (positively correlated). H < 0.5 = anti-persistent. Changes during seizures and in different brain states.'},
        {'term': 'DFA Alpha (Detrended Fluctuation Analysis)',
         'definition': 'Scaling exponent measuring fractal-like temporal correlations. Alpha > 1 indicates non-stationary long-range correlations. Useful for distinguishing normal vs pathological EEG dynamics.'},
        {'term': 'Lempel-Ziv Complexity',
         'definition': 'Measure of algorithmic complexity of the EEG signal. Higher values indicate more complex/random patterns. Decreases during seizures and under anesthesia.'},
        {'term': 'Approximate/Sample Entropy',
         'definition': 'Non-linear measures of signal regularity. Lower entropy = more regular/predictable patterns. Sample entropy is less biased for short data. Both decrease during seizures.'},
        {'term': 'Seizure Semiology',
         'definition': 'Clinical signs and symptoms during a seizure (motor signs, aura type, autonomic features). Combined with EEG patterns, semiology helps localize the seizure onset zone.'},
        {'term': 'AI Label Validation',
         'definition': 'Process where a clinical neurophysiologist reviews AI-generated EEG classifications (e.g., epilepsy vs. control) and confirms, rejects, or modifies the automated label. Essential for quality assurance.'},
    ]


# ── Combined endpoint ─────────────────────────────────────────────────────

def combined():
    ov = overview()
    bd = breakdown()
    defs = definitions()
    return {**ov, **bd, 'definitions': defs}


if __name__ == '__main__':
    import pprint
    result = combined()
    pprint.pprint({k: type(v).__name__ for k, v in result.items()})
    s = result.get('kpis', [])
    print(f"\nKPIs: {len(s)}")
    print(f"Recordings: {len(result.get('recording_inventory', []))}")
    print(f"Band power entries: {len(result.get('band_power_table', []))}")
    print(f"Spectral features: {len(result.get('spectral_features', []))}")
    print(f"AI validation: {len(result.get('ai_validation', []))}")
    print(f"Channel stats: {len(result.get('channel_stats', []))}")
    print(f"Seizure log: {len(result.get('seizure_log', []))}")
    print(f"Definitions: {len(result.get('definitions', []))}")
