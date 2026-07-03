"""RNN/LSTM Temporal Model Dashboard — recurrent neural network architectures
applied to EEG temporal sequences for epilepsy classification using vanilla RNN,
LSTM, GRU, Bidirectional LSTM, and attention-based Temporal models.

Maps clinical.db tables to RNN/LSTM temporal concepts:
- analyses            -> temporal sequence features, model confidence, signal quality
- uploads             -> source EEG files feeding sequence extraction
- seizure_diary       -> ground-truth seizure labels for temporal pattern validation
- assessments         -> longitudinal clinical data for sequence context
- transaction_log     -> pipeline events for sequence + inference jobs
- patients            -> per-patient temporal model inference profiles
- medications         -> AED context for temporal subgroup analysis
"""

import json
import sqlite3
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

EEG_BANDS = {
    'delta': {'label': 'Delta', 'range': '0.5-4 Hz', 'color': '#6366f1'},
    'theta': {'label': 'Theta', 'range': '4-8 Hz',   'color': '#8b5cf6'},
    'alpha': {'label': 'Alpha', 'range': '8-13 Hz',  'color': '#10b981'},
    'beta':  {'label': 'Beta',  'range': '13-30 Hz', 'color': '#f59e0b'},
    'gamma': {'label': 'Gamma', 'range': '30-100 Hz','color': '#ef4444'},
}

QUALITY_COLORS = {
    'Good': '#10b981',
    'Fair': '#f59e0b',
    'Poor': '#ef4444',
}

# Realistic architecture specs for EEG-adapted RNN/LSTM models
MODEL_ARCHITECTURES = {
    'Vanilla-RNN': {
        'name': 'Vanilla Recurrent Neural Network',
        'input_shape': '(batch, seq_len, n_features)',
        'description': 'Basic recurrent network with simple hidden state transition; '
                       'each time step computes h_t = tanh(W_hh * h_{t-1} + W_xh * x_t). '
                       'Prone to vanishing gradients on long EEG sequences (>100 steps).',
        'layers': [
            {'type': 'RNN', 'hidden_size': 128, 'num_layers': 2, 'bidirectional': False},
            {'type': 'Dropout', 'p': 0.3},
            {'type': 'LayerNorm', 'features': 128},
            {'type': 'Linear', 'in': 128, 'out': 64},
            {'type': 'ReLU'},
            {'type': 'Dropout', 'p': 0.2},
            {'type': 'Linear', 'in': 64, 'out': 5},
        ],
        'total_params': 165_893,
        'trainable_params': 165_893,
        'hidden_size': 128,
        'num_layers': 2,
        'output_classes': 5,
        'optimizer': 'Adam (lr=1e-3)',
        'loss': 'CrossEntropyLoss',
        'batch_size': 64,
        'max_seq_len': 256,
    },
    'LSTM': {
        'name': 'Long Short-Term Memory Network',
        'input_shape': '(batch, seq_len, n_features)',
        'description': 'Gated recurrent architecture with forget, input, and output gates '
                       'plus a cell state for long-range dependency capture. The cell state '
                       'highway preserves gradients across hundreds of EEG time steps, making '
                       'LSTM the standard baseline for temporal EEG classification.',
        'layers': [
            {'type': 'LSTM', 'hidden_size': 256, 'num_layers': 3, 'bidirectional': False},
            {'type': 'Dropout', 'p': 0.4},
            {'type': 'LayerNorm', 'features': 256},
            {'type': 'Linear', 'in': 256, 'out': 128},
            {'type': 'ReLU'},
            {'type': 'Dropout', 'p': 0.3},
            {'type': 'Linear', 'in': 128, 'out': 5},
        ],
        'total_params': 1_580_805,
        'trainable_params': 1_580_805,
        'hidden_size': 256,
        'num_layers': 3,
        'output_classes': 5,
        'optimizer': 'Adam (lr=5e-4)',
        'loss': 'CrossEntropyLoss',
        'batch_size': 32,
        'max_seq_len': 512,
    },
    'GRU': {
        'name': 'Gated Recurrent Unit',
        'input_shape': '(batch, seq_len, n_features)',
        'description': 'Simplified gating mechanism combining forget and input gates into '
                       'a single update gate, with a reset gate controlling the hidden state '
                       'exposure. ~33% fewer parameters than LSTM with comparable accuracy '
                       'on EEG seizure detection tasks.',
        'layers': [
            {'type': 'GRU', 'hidden_size': 256, 'num_layers': 3, 'bidirectional': False},
            {'type': 'Dropout', 'p': 0.4},
            {'type': 'LayerNorm', 'features': 256},
            {'type': 'Linear', 'in': 256, 'out': 128},
            {'type': 'ReLU'},
            {'type': 'Dropout', 'p': 0.3},
            {'type': 'Linear', 'in': 128, 'out': 5},
        ],
        'total_params': 1_188_101,
        'trainable_params': 1_188_101,
        'hidden_size': 256,
        'num_layers': 3,
        'output_classes': 5,
        'optimizer': 'Adam (lr=5e-4)',
        'loss': 'CrossEntropyLoss',
        'batch_size': 32,
        'max_seq_len': 512,
    },
    'BiLSTM': {
        'name': 'Bidirectional LSTM',
        'input_shape': '(batch, seq_len, n_features)',
        'description': 'Processes EEG sequences in both forward and backward directions, '
                       'capturing past and future temporal context at each time step. '
                       'Concatenated hidden states (2x width) improve seizure onset detection '
                       'by leveraging post-ictal patterns alongside pre-ictal buildup.',
        'layers': [
            {'type': 'LSTM', 'hidden_size': 256, 'num_layers': 2, 'bidirectional': True},
            {'type': 'Dropout', 'p': 0.4},
            {'type': 'LayerNorm', 'features': 512},
            {'type': 'Linear', 'in': 512, 'out': 256},
            {'type': 'ReLU'},
            {'type': 'Dropout', 'p': 0.3},
            {'type': 'Linear', 'in': 256, 'out': 5},
        ],
        'total_params': 2_634_757,
        'trainable_params': 2_634_757,
        'hidden_size': 256,
        'num_layers': 2,
        'output_classes': 5,
        'optimizer': 'Adam (lr=3e-4)',
        'loss': 'CrossEntropyLoss',
        'batch_size': 32,
        'max_seq_len': 1024,
    },
    'Attention-LSTM': {
        'name': 'LSTM with Temporal Attention',
        'input_shape': '(batch, seq_len, n_features)',
        'description': 'LSTM backbone augmented with a learned attention layer that computes '
                       'soft alignment weights over all time steps. The context vector '
                       'highlights clinically relevant EEG segments (spike-wave complexes, '
                       'ictal onsets) while suppressing background activity, producing '
                       'interpretable temporal saliency maps.',
        'layers': [
            {'type': 'LSTM', 'hidden_size': 256, 'num_layers': 2, 'bidirectional': True},
            {'type': 'TemporalAttention', 'input_dim': 512, 'attn_dim': 128},
            {'type': 'Dropout', 'p': 0.3},
            {'type': 'LayerNorm', 'features': 512},
            {'type': 'Linear', 'in': 512, 'out': 256},
            {'type': 'GELU'},
            {'type': 'Dropout', 'p': 0.2},
            {'type': 'Linear', 'in': 256, 'out': 5},
        ],
        'total_params': 2_832_389,
        'trainable_params': 2_832_389,
        'hidden_size': 256,
        'num_layers': 2,
        'output_classes': 5,
        'optimizer': 'AdamW (lr=2e-4, wd=1e-5)',
        'loss': 'FocalLoss (gamma=2)',
        'batch_size': 16,
        'max_seq_len': 2048,
    },
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


def _parse_analysis_meta(result_json_str):
    """Extract analysis block from result_json (n_channels, sampling_rate, etc.)."""
    if not result_json_str:
        return {}
    try:
        data = json.loads(result_json_str)
        analysis = data.get('analysis', {})
        return {
            'n_channels':         analysis.get('n_channels', 0),
            'n_samples':          analysis.get('n_samples', 0),
            'sampling_rate':      analysis.get('sampling_rate', 0),
            'duration_seconds':   analysis.get('duration_seconds', 0),
            'flat_channels':      analysis.get('flat_channels', 0),
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


def _load_patients():
    return _db_query(
        "SELECT patient_id, age, gender, disease FROM patients ORDER BY patient_id"
    )


def _load_medications():
    return _db_query(
        "SELECT id, patient_id, fields_json, created_at "
        "FROM medications ORDER BY patient_id"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log ORDER BY ts_utc DESC LIMIT 100"
    )


def _sequence_length_label(sampling_rate, duration_seconds):
    """Derive temporal sequence length from signal parameters."""
    if not sampling_rate or not duration_seconds:
        return 'N/A'
    total_samples = int(sampling_rate * duration_seconds)
    return f'{total_samples:,} samples ({duration_seconds:.0f}s @ {sampling_rate:.0f} Hz)'


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the RNN/LSTM Temporal Model dashboard."""
    analyses  = _load_analyses()
    seizures  = _load_seizure_events()
    pipeline  = _load_pipeline_events()

    if not analyses:
        return {
            'available': False,
            'message': 'No EEG analysis data available. '
                       'Upload and analyze EEG files first.',
        }

    total_analyses = len(analyses)

    # Parse metadata from all analyses
    all_meta = []
    for a in analyses:
        meta = _parse_analysis_meta(a.get('result_json'))
        if meta:
            all_meta.append({**meta, 'analysis': a})

    # Unique patients
    patient_ids = {a['patient_id'] for a in analyses if a.get('patient_id')}
    patients_analyzed = len(patient_ids)

    # Temporal sequences extracted — 1 per analysis
    total_sequences = total_analyses

    # Mean model confidence
    confidences = [a['confidence'] for a in analyses if a.get('confidence') is not None]
    mean_confidence = _avg(confidences)

    # Signal quality distribution
    quality_counts = Counter(
        a.get('signal_quality', 'Unknown') for a in analyses
    )

    # Sequence length from first analysis with metadata
    seq_length_label = 'N/A'
    for m in all_meta:
        sr = m.get('sampling_rate', 0)
        dur = m.get('duration_seconds', 0)
        if sr and dur:
            seq_length_label = _sequence_length_label(sr, dur)
            break

    # Total EEG frequency bands
    total_freq_bands = len(EEG_BANDS)

    # Mean sampling rate and duration
    sampling_rates = [m.get('sampling_rate', 0) for m in all_meta if m.get('sampling_rate')]
    mean_sr = _avg(sampling_rates)
    durations = [m.get('duration_seconds', 0) for m in all_meta if m.get('duration_seconds')]
    mean_dur = _avg(durations)

    # Model architectures count
    n_architectures = len(MODEL_ARCHITECTURES)

    # --- Chart: Temporal band power distribution (bar) ---
    band_totals = {b: [] for b in EEG_BANDS}
    for m in all_meta:
        bp = m.get('band_power_relative', {})
        for band_key in EEG_BANDS:
            if band_key in bp:
                band_totals[band_key].append(bp[band_key])

    band_power_chart = [
        {
            'band':       EEG_BANDS[bk]['label'],
            'range':      EEG_BANDS[bk]['range'],
            'color':      EEG_BANDS[bk]['color'],
            'mean_power': _avg(vals),
            'n':          len(vals),
        }
        for bk, vals in band_totals.items()
        if vals
    ]

    # --- Chart: Signal quality distribution (pie) ---
    quality_distribution = [
        {
            'quality': q,
            'count':   cnt,
            'color':   QUALITY_COLORS.get(q, '#6b7280'),
        }
        for q, cnt in sorted(quality_counts.items())
        if cnt > 0
    ]

    # --- Chart: Classification by predicted label (bar) ---
    label_counts = Counter(
        a.get('predicted_label', 'Unknown') for a in analyses
    )
    label_conf = {}
    for a in analyses:
        lbl = a.get('predicted_label', 'Unknown')
        label_conf.setdefault(lbl, [])
        if a.get('confidence') is not None:
            label_conf[lbl].append(a['confidence'])

    classification_chart = [
        {
            'predicted_label': lbl,
            'count':           cnt,
            'mean_confidence': _avg(label_conf.get(lbl, [])),
        }
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])
    ]

    # --- Chart: Daily temporal sequence generation activity (line) ---
    daily_counts = Counter()
    for a in analyses:
        day = (a.get('created_at') or '')[:10]
        if day:
            daily_counts[day] += 1
    daily_activity = [
        {'date': day, 'sequences': cnt}
        for day, cnt in sorted(daily_counts.items())
    ]

    # --- Chart: Seizure temporal patterns (duration distribution) ---
    seizure_durations = []
    for s in seizures:
        dur = s.get('duration_sec')
        if dur is not None:
            seizure_durations.append({
                'patient_id':  s.get('patient_id'),
                'duration_sec': dur,
                'severity':    s.get('severity', 'Unknown'),
                'event_date':  s.get('event_date'),
            })

    # Bin seizure durations for histogram
    duration_bins = {'<30s': 0, '30-60s': 0, '60-120s': 0, '120-300s': 0, '>300s': 0}
    for sd in seizure_durations:
        d = sd['duration_sec']
        if d < 30:
            duration_bins['<30s'] += 1
        elif d < 60:
            duration_bins['30-60s'] += 1
        elif d < 120:
            duration_bins['60-120s'] += 1
        elif d < 300:
            duration_bins['120-300s'] += 1
        else:
            duration_bins['>300s'] += 1

    duration_histogram = [
        {'bin': k, 'count': v}
        for k, v in duration_bins.items()
    ]

    return {
        'available':            True,
        'total_analyses':       total_analyses,
        'patients_analyzed':    patients_analyzed,
        'total_sequences':      total_sequences,
        'mean_confidence':      mean_confidence,
        'mean_sampling_rate':   mean_sr,
        'mean_duration':        mean_dur,
        'seq_length_label':     seq_length_label,
        'total_freq_bands':     total_freq_bands,
        'n_architectures':      n_architectures,
        'seizure_events':       len(seizures),
        'pipeline_events':      len(pipeline),
        'band_power_chart':     band_power_chart,
        'quality_distribution': quality_distribution,
        'classification_chart': classification_chart,
        'daily_activity':       daily_activity,
        'duration_histogram':   duration_histogram,
        'kpis': [
            {'label': 'EEG Sequences',          'value': str(total_sequences)},
            {'label': 'Patients Analyzed',      'value': str(patients_analyzed)},
            {'label': 'RNN/LSTM Architectures', 'value': str(n_architectures),
             'sub': 'RNN / LSTM / GRU / BiLSTM / Attn'},
            {'label': 'Mean Confidence',
             'value': f'{mean_confidence:.1%}',
             'color': '#10b981' if mean_confidence >= 0.8
                      else '#f59e0b' if mean_confidence >= 0.6
                      else '#ef4444'},
            {'label': 'Signal Quality (Good)',
             'value': str(quality_counts.get('Good', 0)),
             'sub': f'of {total_analyses} analyses'},
            {'label': 'Sequence Length',        'value': seq_length_label},
            {'label': 'Seizure Events',         'value': str(len(seizures)),
             'sub': 'temporal labels'},
            {'label': 'Mean Duration',          'value': f'{mean_dur:.0f}s'},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed RNN/LSTM breakdown — sequence inventory, patient profiles,
    model architecture specs, temporal training readiness, pipeline events."""
    analyses   = _load_analyses()
    uploads    = _load_uploads()
    seizures   = _load_seizure_events()
    patients   = _load_patients()
    medications = _load_medications()
    pipeline   = _load_pipeline_events()

    if not analyses:
        return {'available': False}

    upload_map  = {u['id']: u for u in uploads}
    patient_map = {p['patient_id']: p for p in patients}

    # --- Sequence inventory (per-analysis detail) ---
    sequence_inventory = []
    for a in analyses:
        meta   = _parse_analysis_meta(a.get('result_json'))
        upload = upload_map.get(a.get('upload_id'), {})
        sr     = meta.get('sampling_rate', 0)
        dur    = meta.get('duration_seconds', 0)
        n_samples = int(sr * dur) if sr and dur else 0
        sequence_inventory.append({
            'id':               a['id'],
            'patient_id':       a['patient_id'],
            'file_name':        upload.get('file_name', ''),
            'disease':          a.get('disease', ''),
            'predicted_label':  a.get('predicted_label', ''),
            'confidence':       a.get('confidence'),
            'signal_quality':   a.get('signal_quality', ''),
            'n_channels':       meta.get('n_channels', 0),
            'sampling_rate':    sr,
            'duration_seconds': dur,
            'total_samples':    n_samples,
            'sequence_label':   _sequence_length_label(sr, dur),
            'band_power':       meta.get('band_power_relative', {}),
            'created_at':       a.get('created_at'),
        })

    # --- Patient temporal profiles (per-patient summary) ---
    patient_analyses = {}
    for a in analyses:
        pid = a['patient_id']
        patient_analyses.setdefault(pid, []).append(a)

    # Seizure events by patient
    seizures_by_patient = {}
    for s in seizures:
        pid = s.get('patient_id')
        if pid:
            seizures_by_patient.setdefault(pid, []).append(s)

    med_by_patient = {}
    for m in medications:
        pid = m.get('patient_id')
        if pid:
            try:
                fj = json.loads(m.get('fields_json') or '{}')
                drug = fj.get('drug_name', '')
            except (json.JSONDecodeError, TypeError):
                drug = ''
            if drug:
                med_by_patient.setdefault(pid, []).append(drug)

    patient_profiles = []
    for pid in sorted(patient_analyses.keys()):
        pa = patient_analyses[pid]
        confs = [a['confidence'] for a in pa if a.get('confidence') is not None]
        qualities = [a.get('signal_quality', 'Unknown') for a in pa]
        diseases  = list({a.get('disease', '') for a in pa if a.get('disease')})

        # Aggregate band powers for temporal features
        band_avgs = {}
        for bk in EEG_BANDS:
            vals = []
            for a in pa:
                bp = _parse_analysis_meta(a.get('result_json')).get('band_power_relative', {})
                if bk in bp:
                    vals.append(bp[bk])
            if vals:
                band_avgs[bk] = _avg(vals)

        # Seizure statistics for this patient
        patient_seizures = seizures_by_patient.get(pid, [])
        seizure_durs = [s.get('duration_sec', 0) for s in patient_seizures if s.get('duration_sec')]

        pinfo = patient_map.get(pid, {})
        patient_profiles.append({
            'patient_id':          pid,
            'age':                 pinfo.get('age'),
            'sex':                 pinfo.get('gender'),
            'total_sequences':     len(pa),
            'diseases':            diseases,
            'mean_confidence':     _avg(confs),
            'quality_distribution': dict(Counter(qualities)),
            'band_power_averages':  band_avgs,
            'seizure_count':       len(patient_seizures),
            'mean_seizure_dur':    _avg(seizure_durs) if seizure_durs else None,
            'medications':         list(set(med_by_patient.get(pid, []))),
        })

    # --- Model architecture specs ---
    model_architecture = []
    for arch_key, arch in MODEL_ARCHITECTURES.items():
        model_architecture.append({
            'key':               arch_key,
            'name':              arch['name'],
            'input_shape':       arch['input_shape'],
            'description':       arch['description'],
            'n_layers':          len(arch['layers']),
            'total_params':      arch['total_params'],
            'trainable_params':  arch['trainable_params'],
            'hidden_size':       arch['hidden_size'],
            'num_layers':        arch['num_layers'],
            'output_classes':    arch['output_classes'],
            'optimizer':         arch['optimizer'],
            'loss':              arch['loss'],
            'batch_size':        arch['batch_size'],
            'max_seq_len':       arch['max_seq_len'],
            'layers':            arch['layers'],
        })

    # --- Temporal training readiness ---
    disease_counts = Counter(
        a.get('disease', 'Unknown') for a in analyses
    )
    quality_counts = Counter(
        a.get('signal_quality', 'Unknown') for a in analyses
    )
    good_sequences = quality_counts.get('Good', 0)
    total_sequences = len(analyses)
    usable_ratio = round(good_sequences / total_sequences, 3) if total_sequences else 0.0

    n_classes = len(disease_counts)
    counts_list = list(disease_counts.values())
    max_c = max(counts_list) if counts_list else 1
    min_c = min(counts_list) if counts_list else 0
    class_balance_ratio = round(min_c / max_c, 3) if max_c else 0.0

    # Temporal-specific readiness: check mean duration for sequence adequacy
    durations = []
    for a in analyses:
        meta = _parse_analysis_meta(a.get('result_json'))
        dur = meta.get('duration_seconds', 0)
        if dur:
            durations.append(dur)
    mean_dur = _avg(durations)

    training_readiness = {
        'total_sequences':     total_sequences,
        'usable_sequences':    good_sequences,
        'usable_ratio':        usable_ratio,
        'n_classes':           n_classes,
        'class_balance_ratio': class_balance_ratio,
        'balance_status':      'Balanced' if class_balance_ratio >= 0.8
                               else 'Moderate imbalance' if class_balance_ratio >= 0.5
                               else 'Severe imbalance',
        'mean_duration_sec':   mean_dur,
        'per_disease': [
            {
                'disease':    disease,
                'sequences':  cnt,
                'pct':        round(cnt / total_sequences * 100, 1) if total_sequences else 0,
            }
            for disease, cnt in sorted(disease_counts.items(), key=lambda x: -x[1])
        ],
        'seizure_labeled':     len(seizures),
        'readiness_flags':     _training_readiness_flags(
            total_sequences, good_sequences, class_balance_ratio, mean_dur
        ),
    }

    # --- Seizure temporal patterns ---
    seizure_temporal = []
    for s in seizures:
        seizure_temporal.append({
            'id':          s['id'],
            'patient_id':  s.get('patient_id'),
            'event_date':  s.get('event_date'),
            'event_time':  s.get('event_time'),
            'duration_sec': s.get('duration_sec'),
            'severity':    s.get('severity'),
            'trigger':     s.get('trigger'),
        })

    # --- Pipeline events ---
    pipeline_log = [
        {
            'id':        e['id'],
            'component': e.get('component'),
            'action':    e.get('action'),
            'actor':     e.get('actor'),
            'detail':    (e.get('detail') or '')[:120],
            'ts_utc':    e.get('ts_utc'),
        }
        for e in pipeline[:50]
    ]

    return {
        'available':            True,
        'sequence_inventory':   sequence_inventory,
        'patient_profiles':     patient_profiles,
        'model_architecture':   model_architecture,
        'training_readiness':   training_readiness,
        'seizure_temporal':     seizure_temporal,
        'pipeline_log':         pipeline_log,
    }


def _training_readiness_flags(total, usable, balance_ratio, mean_dur):
    """Return list of readiness flag dicts for temporal training readiness."""
    flags = []
    if total < 50:
        flags.append({
            'flag':    'Insufficient data',
            'detail':  f'Only {total} sequences available; recommend >= 50 per class.',
            'severity': 'warning',
        })
    else:
        flags.append({
            'flag':    'Data volume OK',
            'detail':  f'{total} sequences available.',
            'severity': 'ok',
        })

    if usable < total * 0.8:
        flags.append({
            'flag':    'Signal quality concern',
            'detail':  f'Only {usable}/{total} sequences are Good quality.',
            'severity': 'warning',
        })
    else:
        flags.append({
            'flag':    'Signal quality OK',
            'detail':  f'{usable}/{total} sequences are Good quality.',
            'severity': 'ok',
        })

    if balance_ratio < 0.5:
        flags.append({
            'flag':    'Class imbalance',
            'detail':  f'Imbalance ratio {balance_ratio:.2f}; consider oversampling / class weights.',
            'severity': 'error',
        })
    elif balance_ratio < 0.8:
        flags.append({
            'flag':    'Mild class imbalance',
            'detail':  f'Imbalance ratio {balance_ratio:.2f}; monitor during training.',
            'severity': 'warning',
        })
    else:
        flags.append({
            'flag':    'Class balance OK',
            'detail':  f'Imbalance ratio {balance_ratio:.2f}.',
            'severity': 'ok',
        })

    # Temporal-specific: check if sequences are long enough for RNN training
    if mean_dur < 10:
        flags.append({
            'flag':    'Short sequences',
            'detail':  f'Mean duration {mean_dur:.0f}s; RNN/LSTM needs >= 10s for temporal patterns.',
            'severity': 'warning',
        })
    else:
        flags.append({
            'flag':    'Sequence length OK',
            'detail':  f'Mean duration {mean_dur:.0f}s; adequate for temporal modeling.',
            'severity': 'ok',
        })

    return flags


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """RNN/LSTM Temporal Model AI concepts, quality metrics, model variants, compliance."""
    return {
        'concepts': [
            {
                'term': 'RNN (Recurrent Neural Network)',
                'definition': 'Neural network with feedback connections that maintain a hidden state '
                              'across time steps, enabling sequential processing of EEG signals. '
                              'Each time step updates the hidden state based on current input and '
                              'previous state, creating a temporal memory of past signal patterns.',
            },
            {
                'term': 'LSTM (Long Short-Term Memory)',
                'definition': 'Gated recurrent architecture that solves the vanishing gradient problem '
                              'through three gates (forget, input, output) and a cell state highway. '
                              'The cell state can carry information unchanged across hundreds of time steps, '
                              'making LSTM ideal for detecting long-range temporal dependencies in EEG '
                              'such as pre-ictal buildup lasting minutes before seizure onset.',
            },
            {
                'term': 'GRU (Gated Recurrent Unit)',
                'definition': 'Simplified gating variant of LSTM that merges the forget and input gates '
                              'into a single update gate and uses a reset gate to control hidden state exposure. '
                              'Approximately 33% fewer parameters than LSTM with comparable performance on '
                              'most EEG temporal classification benchmarks.',
            },
            {
                'term': 'Bidirectional RNN/LSTM',
                'definition': 'Architecture that processes EEG sequences in both forward (past-to-future) '
                              'and backward (future-to-past) directions simultaneously. The concatenated '
                              'bidirectional hidden states capture both pre-ictal buildup and post-ictal '
                              'suppression patterns, improving seizure onset and offset detection.',
            },
            {
                'term': 'Temporal Attention Mechanism',
                'definition': 'Learnable soft-alignment layer applied over RNN/LSTM hidden states that '
                              'computes importance weights for each time step. Produces interpretable '
                              'temporal saliency maps highlighting clinically relevant EEG segments '
                              '(spike-wave complexes, ictal discharges) while suppressing background.',
            },
            {
                'term': 'Hidden State',
                'definition': 'Internal vector representation that the RNN/LSTM maintains and updates '
                              'at each time step. Encodes a compressed summary of the EEG signal history '
                              'up to that point. Hidden state dimensionality (typically 128-512) controls '
                              'the model capacity for capturing temporal patterns.',
            },
            {
                'term': 'Sequence-to-Label Classification',
                'definition': 'Temporal classification paradigm where a variable-length EEG sequence '
                              '(seconds to minutes) is mapped to a single diagnostic label using the '
                              'final hidden state or an attention-weighted context vector. Contrasts with '
                              'sequence-to-sequence (per-sample) annotation used in seizure localization.',
            },
            {
                'term': 'Vanishing/Exploding Gradients',
                'definition': 'Fundamental training instability in deep recurrent networks where '
                              'gradient magnitudes shrink exponentially (vanishing) or grow unboundedly '
                              '(exploding) through long sequences. LSTM cell states and gradient clipping '
                              'are the primary mitigations for EEG signals spanning thousands of samples.',
            },
        ],
        'quality_metrics': [
            {
                'metric':      'Sequence Length',
                'target':      '>= 256 time steps (2-10s EEG)',
                'description': 'Number of temporal samples per input sequence. Longer sequences capture '
                               'more temporal context but increase training time quadratically for attention.',
            },
            {
                'metric':      'Temporal Resolution',
                'target':      '>= 256 Hz sampling rate',
                'description': 'Sampling frequency of the input EEG signal. Higher resolution preserves '
                               'fast transients (spikes, sharp waves) critical for seizure classification.',
            },
            {
                'metric':      'Class Balance',
                'target':      'Imbalance ratio >= 0.80',
                'description': 'Ratio of minority to majority class sequences. Temporal models are '
                               'sensitive to imbalance; use weighted sampling or focal loss for correction.',
            },
            {
                'metric':      'Temporal Stationarity',
                'target':      'Segment-level stationarity check',
                'description': 'Statistical consistency of EEG signal properties within each sequence. '
                               'Non-stationary segments (electrode drift, movement artifacts) degrade '
                               'RNN/LSTM hidden state stability.',
            },
        ],
        'model_variants': [
            {
                'name':        'Vanilla RNN',
                'params':      '~166 K',
                'description': 'Basic recurrent architecture; fast training, limited long-range memory '
                               '(effective for sequences < 100 steps).',
            },
            {
                'name':        'LSTM',
                'params':      '~1.6 M',
                'description': 'Standard gated architecture; strong baseline for EEG temporal tasks; '
                               'cell state preserves gradients across 500+ time steps.',
            },
            {
                'name':        'GRU',
                'params':      '~1.2 M',
                'description': 'Lightweight gated variant; faster convergence than LSTM; competitive '
                               'accuracy with fewer parameters on moderate-length EEG sequences.',
            },
            {
                'name':        'Bidirectional LSTM',
                'params':      '~2.6 M',
                'description': 'Processes sequences forward and backward; best for seizure onset '
                               'detection where future context (post-ictal pattern) aids classification.',
            },
            {
                'name':        'Attention-LSTM',
                'params':      '~2.8 M',
                'description': 'BiLSTM with temporal attention; highest accuracy and interpretability; '
                               'attention weights produce clinically meaningful saliency maps.',
            },
        ],
        'compliance': [
            {
                'standard':   'FDA AI/ML SaMD',
                'reference':  'Software as a Medical Device framework for RNN/LSTM-based EEG temporal '
                              'classifiers; requires predetermined change control plan and locked model '
                              'weights for clinical deployment.',
            },
            {
                'standard':   'IEC 62304',
                'reference':  'Medical device software lifecycle for RNN/LSTM training, validation, '
                              'and deployment pipelines including sequence preprocessing modules.',
            },
            {
                'standard':   'ISO 14971:2019',
                'reference':  'Risk management — quantify and mitigate risks from temporal model '
                              'misclassification including false-negative seizure detection.',
            },
            {
                'standard':   'HIPAA',
                'reference':  'Protected health information in EEG temporal sequences; de-identify '
                              'patient data before model training or federated learning.',
            },
            {
                'standard':   'EU AI Act Article 6',
                'reference':  'High-risk AI system obligations for RNN/LSTM diagnostic tools: '
                              'temporal model transparency, human oversight, and conformity assessment.',
            },
        ],
        'remediation': [
            {
                'issue':    'Vanishing gradients on long EEG sequences',
                'strategy': 'Switch from vanilla RNN to LSTM/GRU; apply gradient clipping (max_norm=1.0); '
                            'reduce sequence length with sliding window or learned downsampling.',
            },
            {
                'issue':    'Poor temporal generalization across patients',
                'strategy': 'Use leave-one-subject-out cross-validation; apply patient-level normalization; '
                            'add domain adaptation layers between LSTM encoder and classifier head.',
            },
            {
                'issue':    'Low seizure detection sensitivity (high false negatives)',
                'strategy': 'Use focal loss (gamma=2) to upweight rare seizure samples; train with '
                            'attention-LSTM to focus on ictal onset segments; ensemble with CNN baseline.',
            },
            {
                'issue':    'Overfitting on small temporal datasets',
                'strategy': 'Apply variational dropout (shared across time steps); use temporal data '
                            'augmentation (time-warp, jitter, magnitude-warp); reduce hidden_size.',
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
    print(f"Sequence inventory: {len(bd.get('sequence_inventory', []))}")
    print(f"Patient profiles: {len(bd.get('patient_profiles', []))}")
    print(f"Model architectures: {len(bd.get('model_architecture', []))}")
    tr = bd.get('training_readiness', {})
    print(f"Training readiness: {tr.get('total_sequences')} sequences, "
          f"balance ratio {tr.get('class_balance_ratio')}")
    print(f"Seizure temporal: {len(bd.get('seizure_temporal', []))}")
    print(f"Pipeline events: {len(bd.get('pipeline_log', []))}")
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f"Concepts: {len(df.get('concepts', []))}")
    print(f"Quality metrics: {len(df.get('quality_metrics', []))}")
    print(f"Model variants: {len(df.get('model_variants', []))}")
    print(f"Compliance refs: {len(df.get('compliance', []))}")
    print(f"Remediation strategies: {len(df.get('remediation', []))}")
