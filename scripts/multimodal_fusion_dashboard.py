"""Multimodal Fusion Dashboard — AI Module.
Tracks multimodal data integration (EEG + video + clinical notes + medication + MRI),
fusion analysis status, risk scoring, and subtype prediction across patients.

Reads from:
  - multimodal_fusion  (session-level fusion records with modality availability,
                         risk scores, predicted subtypes, concordance)

Uses real patient data from clinical.db (EPAT001-EPAT030).
Fusion methods: early, late, attention, cross-modal transformer, tensor fusion.
"""

import json
import os
import sqlite3
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _db_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = _db_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def _pct(n, total):
    return round(n / total * 100, 1) if total else 0


def overview():
    """Return KPI cards + chart data for the Multimodal Fusion overview tab."""
    rows = _db_query('SELECT * FROM multimodal_fusion')
    total = len(rows)
    if total == 0:
        return {'available': False, 'reason': 'No multimodal fusion data found'}

    patient_ids = list(set(r['patient_id'] for r in rows))
    total_patients = len(patient_ids)

    # --- Status breakdown ---
    completed = [r for r in rows if r['status'] == 'completed']
    partial = [r for r in rows if r['status'] == 'partial']
    failed = [r for r in rows if r['status'] == 'failed']
    insufficient = [r for r in rows if r['status'] == 'insufficient']

    completion_rate = _pct(len(completed), total)

    # --- Risk score stats ---
    risk_scores = [r['risk_score'] for r in completed if r['risk_score'] is not None]
    avg_risk = _avg(risk_scores)
    high_risk = sum(1 for s in risk_scores if s >= 0.7)
    moderate_risk = sum(1 for s in risk_scores if 0.3 <= s < 0.7)
    low_risk = sum(1 for s in risk_scores if s < 0.3)

    # --- Confidence ---
    conf_vals = [r['confidence'] for r in completed if r['confidence'] is not None]
    avg_confidence = _avg(conf_vals)

    # --- Concordance ---
    conc_vals = [r['concordance_score'] for r in completed if r['concordance_score'] is not None]
    avg_concordance = _avg(conc_vals)

    # --- Processing time ---
    proc_times = [r['processing_time_sec'] for r in rows if r['processing_time_sec'] is not None]
    avg_proc_time = _avg(proc_times)

    # --- Modality availability ---
    modality_keys = ['eeg_available', 'video_available', 'clinical_notes_available',
                     'medication_data_available', 'mri_available']
    modality_labels = ['EEG', 'Video', 'Clinical Notes', 'Medication', 'MRI']
    modality_rates = []
    for key, label in zip(modality_keys, modality_labels):
        avail = sum(1 for r in rows if r[key])
        modality_rates.append({'modality': label, 'available': avail,
                               'missing': total - avail, 'rate': _pct(avail, total)})

    # --- Avg modalities per session ---
    avg_modalities = _avg([r['modalities_count'] for r in rows])

    # --- Fusion method distribution ---
    method_counts = Counter(r['fusion_method'] for r in completed if r['fusion_method'])
    fusion_methods = [{'method': m, 'count': c} for m, c in method_counts.most_common()]

    # --- Predicted subtype distribution ---
    subtype_counts = Counter(r['predicted_subtype'] for r in completed if r['predicted_subtype'])
    subtypes = [{'subtype': s, 'count': c} for s, c in subtype_counts.most_common()]

    # --- Risk distribution chart ---
    risk_distribution = [
        {'category': 'High (≥0.7)', 'count': high_risk},
        {'category': 'Moderate (0.3–0.7)', 'count': moderate_risk},
        {'category': 'Low (<0.3)', 'count': low_risk},
    ]

    # --- Status pie ---
    status_dist = [
        {'status': 'Completed', 'count': len(completed)},
        {'status': 'Partial', 'count': len(partial)},
        {'status': 'Failed', 'count': len(failed)},
        {'status': 'Insufficient', 'count': len(insufficient)},
    ]

    # --- Monthly trend ---
    monthly = Counter()
    for r in rows:
        month = r['session_date'][:7] if r['session_date'] else 'unknown'
        monthly[month] += 1
    monthly_trend = [{'month': m, 'sessions': c} for m, c in sorted(monthly.items())]

    return {
        'available': True,
        'total_sessions': total,
        'total_patients': total_patients,
        'completion_rate': completion_rate,
        'avg_risk_score': avg_risk,
        'avg_confidence': avg_confidence,
        'avg_concordance': avg_concordance,
        'avg_processing_time_sec': avg_proc_time,
        'avg_modalities_per_session': avg_modalities,
        'risk_distribution': risk_distribution,
        'status_distribution': status_dist,
        'modality_availability': modality_rates,
        'fusion_methods': fusion_methods,
        'predicted_subtypes': subtypes,
        'monthly_trend': monthly_trend,
    }


def breakdown():
    """Per-patient and per-session fusion details, recent sessions, method comparison."""
    rows = _db_query('SELECT * FROM multimodal_fusion ORDER BY session_date DESC')
    if not rows:
        return {'available': False}

    # --- Per-patient summary ---
    patient_map = {}
    for r in rows:
        pid = r['patient_id']
        if pid not in patient_map:
            patient_map[pid] = {'patient_id': pid, 'sessions': 0, 'completed': 0,
                                'risk_scores': [], 'confidences': [], 'modalities': []}
        pm = patient_map[pid]
        pm['sessions'] += 1
        if r['status'] == 'completed':
            pm['completed'] += 1
            if r['risk_score'] is not None:
                pm['risk_scores'].append(r['risk_score'])
            if r['confidence'] is not None:
                pm['confidences'].append(r['confidence'])
        pm['modalities'].append(r['modalities_count'])

    per_patient = []
    for pid, pm in sorted(patient_map.items()):
        per_patient.append({
            'patient_id': pid,
            'total_sessions': pm['sessions'],
            'completed': pm['completed'],
            'completion_rate': _pct(pm['completed'], pm['sessions']),
            'avg_risk': _avg(pm['risk_scores']),
            'avg_confidence': _avg(pm['confidences']),
            'avg_modalities': _avg(pm['modalities']),
        })

    # --- Recent sessions (last 20) ---
    recent = []
    for r in rows[:20]:
        recent.append({
            'session_id': r['session_id'],
            'patient_id': r['patient_id'],
            'date': r['session_date'],
            'modalities': r['modalities_count'],
            'fusion_method': r['fusion_method'] or '--',
            'risk_score': r['risk_score'],
            'subtype': r['predicted_subtype'] or '--',
            'confidence': r['confidence'],
            'concordance': r['concordance_score'],
            'status': r['status'],
            'explanation': r['explanation'] or '--',
        })

    # --- Method comparison ---
    method_stats = {}
    completed = [r for r in rows if r['status'] == 'completed']
    for r in completed:
        m = r['fusion_method']
        if m not in method_stats:
            method_stats[m] = {'risks': [], 'confs': [], 'concs': [], 'times': [], 'count': 0}
        ms = method_stats[m]
        ms['count'] += 1
        if r['risk_score'] is not None:
            ms['risks'].append(r['risk_score'])
        if r['confidence'] is not None:
            ms['confs'].append(r['confidence'])
        if r['concordance_score'] is not None:
            ms['concs'].append(r['concordance_score'])
        if r['processing_time_sec'] is not None:
            ms['times'].append(r['processing_time_sec'])

    method_comparison = []
    for m, ms in sorted(method_stats.items()):
        method_comparison.append({
            'method': m,
            'sessions': ms['count'],
            'avg_risk': _avg(ms['risks']),
            'avg_confidence': _avg(ms['confs']),
            'avg_concordance': _avg(ms['concs']),
            'avg_processing_sec': _avg(ms['times']),
        })

    # --- Modality co-occurrence matrix ---
    modality_keys = ['eeg_available', 'video_available', 'clinical_notes_available',
                     'medication_data_available', 'mri_available']
    modality_labels = ['EEG', 'Video', 'Notes', 'Meds', 'MRI']
    cooccurrence = []
    for i, (ki, li) in enumerate(zip(modality_keys, modality_labels)):
        row_data = {'modality': li}
        for j, (kj, lj) in enumerate(zip(modality_keys, modality_labels)):
            both = sum(1 for r in rows if r[ki] and r[kj])
            row_data[lj] = both
        cooccurrence.append(row_data)

    return {
        'available': True,
        'per_patient': per_patient,
        'recent_sessions': recent,
        'method_comparison': method_comparison,
        'modality_cooccurrence': cooccurrence,
    }


def definitions():
    """Multimodal fusion glossary — modalities, fusion methods, metrics, risk tiers."""
    return {
        'glossary': [
            {'term': 'Multimodal Fusion', 'definition': 'Combining data from multiple modalities (EEG, video, clinical notes, medication, MRI) into a unified representation for joint analysis and prediction.'},
            {'term': 'Early Fusion', 'definition': 'Concatenating raw features from all modalities before model input. Simple but may lose modality-specific structure.'},
            {'term': 'Late Fusion', 'definition': 'Training separate models per modality, then combining their outputs (e.g., averaging, voting). Preserves modality-specific patterns.'},
            {'term': 'Attention Fusion', 'definition': 'Using attention mechanisms to dynamically weight contributions from each modality based on relevance and quality.'},
            {'term': 'Cross-Modal Transformer', 'definition': 'Transformer architecture with cross-attention between modality encoders, enabling rich inter-modal interaction.'},
            {'term': 'Tensor Fusion', 'definition': 'Computing outer products of modality representations to capture higher-order interactions between modalities.'},
            {'term': 'Risk Score', 'definition': 'Predicted probability (0–1) of seizure recurrence or adverse outcome from the fused multimodal analysis. ≥0.7 = high, 0.3–0.7 = moderate, <0.3 = low.'},
            {'term': 'Confidence', 'definition': 'Model certainty in its prediction (0–1). Higher values indicate more reliable predictions.'},
            {'term': 'Concordance Score', 'definition': 'Agreement between modality-specific predictions (0–1). High concordance = modalities agree; low = conflicting signals warranting clinical review.'},
            {'term': 'Modalities Count', 'definition': 'Number of data modalities available for a given session (max 5: EEG, video, notes, medication, MRI).'},
            {'term': 'Predicted Subtype', 'definition': 'Epilepsy classification predicted by the fusion model — focal, generalized, focal-to-bilateral, absence, myoclonic, or tonic-clonic (ILAE taxonomy).'},
        ],
        'risk_tiers': [
            {'tier': 'High', 'range': '≥ 0.7', 'action': 'Urgent clinical review, medication adjustment consideration'},
            {'tier': 'Moderate', 'range': '0.3 – 0.7', 'action': 'Scheduled follow-up, monitor trends'},
            {'tier': 'Low', 'range': '< 0.3', 'action': 'Routine monitoring, standard follow-up interval'},
        ],
        'modalities': [
            {'name': 'EEG', 'description': 'Electroencephalography — primary neural signal for seizure detection and classification'},
            {'name': 'Video', 'description': 'Video monitoring — captures semiology (movement patterns) during seizure events'},
            {'name': 'Clinical Notes', 'description': 'Free-text clinical documentation — history, observations, assessments'},
            {'name': 'Medication', 'description': 'Antiseizure medication (ASM) data — drugs, doses, adherence, side effects'},
            {'name': 'MRI', 'description': 'Magnetic resonance imaging — structural brain data, lesion detection, volumetrics'},
        ],
    }
