"""PNES Screening Dashboard — AI Module.
Tracks psychogenic non-epileptic seizures (PNES) screening assessments,
semiological scoring, EEG findings, psychiatric comorbidity, and
differential classification (PNES vs epilepsy) across patients.

Reads from:
  - pnes_screening  (per-assessment records with semiological scores,
                      EEG findings, psychiatric factors, classification)

Uses real patient data from clinical.db (EPAT001-EPAT030).
Classifications: pnes_likely, epilepsy_likely, mixed, indeterminate.
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
    """Return KPI cards + chart data for the PNES Screening overview tab."""
    rows = _db_query('SELECT * FROM pnes_screening')
    total = len(rows)
    if total == 0:
        return {'available': False, 'reason': 'No PNES screening data found'}

    patient_ids = list(set(r['patient_id'] for r in rows))
    total_patients = len(patient_ids)

    # --- Classification distribution ---
    cls_counts = Counter(r['classification'] for r in rows)
    classification_dist = [
        {'classification': c, 'count': n}
        for c, n in cls_counts.most_common()
    ]

    # --- PNES probability distribution ---
    pnes_probs = [r['pnes_probability'] for r in rows if r['pnes_probability'] is not None]
    high_prob = sum(1 for p in pnes_probs if p >= 0.65)
    moderate_prob = sum(1 for p in pnes_probs if 0.35 <= p < 0.65)
    low_prob = sum(1 for p in pnes_probs if p < 0.35)
    probability_dist = [
        {'category': 'High (>=0.65)', 'count': high_prob},
        {'category': 'Moderate (0.35-0.65)', 'count': moderate_prob},
        {'category': 'Low (<0.35)', 'count': low_prob},
    ]

    avg_pnes_prob = _avg(pnes_probs)

    # --- Confidence ---
    conf_vals = [r['confidence'] for r in rows if r['confidence'] is not None]
    avg_confidence = _avg(conf_vals)

    # --- Review status breakdown ---
    status_counts = Counter(r['status'] for r in rows)
    status_dist = [
        {'status': s, 'count': n}
        for s, n in status_counts.most_common()
    ]

    # --- Top semiological features (average scores) ---
    feature_keys = [
        ('eye_closure_score', 'Eye Closure'),
        ('pelvic_thrusting_score', 'Pelvic Thrusting'),
        ('side_to_side_head_score', 'Side-to-Side Head'),
        ('ictal_crying_score', 'Ictal Crying'),
        ('memory_recall_score', 'Memory Recall'),
        ('gradual_onset_score', 'Gradual Onset'),
    ]
    semiological_avgs = []
    for key, label in feature_keys:
        vals = [r[key] for r in rows if r[key] is not None]
        semiological_avgs.append({'feature': label, 'avg_score': _avg(vals)})

    # --- Psychiatric comorbidity distribution ---
    comorbidity_counts = Counter(r['psychiatric_comorbidity'] for r in rows if r['psychiatric_comorbidity'])
    comorbidity_dist = [
        {'comorbidity': c or 'None', 'count': n}
        for c, n in comorbidity_counts.most_common()
    ]

    # --- Monthly screening trend ---
    monthly = Counter()
    for r in rows:
        month = r['screening_date'][:7] if r['screening_date'] else 'unknown'
        monthly[month] += 1
    monthly_trend = [{'month': m, 'screenings': c} for m, c in sorted(monthly.items())]

    # --- Video-EEG recommendation rate ---
    veeg_recommended = sum(1 for r in rows if r['video_eeg_recommended'])
    veeg_rate = _pct(veeg_recommended, total)

    # --- Psychiatry referral rate ---
    psych_referred = sum(1 for r in rows if r['psychiatry_referral'])
    psych_rate = _pct(psych_referred, total)

    # --- Referral reason distribution ---
    reason_counts = Counter(r['referral_reason'] for r in rows if r['referral_reason'])
    referral_reasons = [
        {'reason': r, 'count': n}
        for r, n in reason_counts.most_common()
    ]

    # --- Duration > 2 min rate ---
    duration_gt2 = sum(1 for r in rows if r['duration_gt_2min'])
    duration_rate = _pct(duration_gt2, total)

    # --- EEG finding rates ---
    ictal_normal = sum(1 for r in rows if r['eeg_ictal_normal'])
    interictal_normal = sum(1 for r in rows if r['eeg_interictal_normal'])

    return {
        'available': True,
        'total_screenings': total,
        'total_patients': total_patients,
        'avg_pnes_probability': avg_pnes_prob,
        'avg_confidence': avg_confidence,
        'classification_distribution': classification_dist,
        'probability_distribution': probability_dist,
        'status_distribution': status_dist,
        'semiological_averages': semiological_avgs,
        'comorbidity_distribution': comorbidity_dist,
        'monthly_trend': monthly_trend,
        'video_eeg_recommendation_rate': veeg_rate,
        'psychiatry_referral_rate': psych_rate,
        'referral_reason_distribution': referral_reasons,
        'duration_gt2min_rate': duration_rate,
        'eeg_ictal_normal_count': ictal_normal,
        'eeg_interictal_normal_count': interictal_normal,
    }


def breakdown():
    """Per-patient and per-screening details, semiological feature comparison."""
    rows = _db_query('SELECT * FROM pnes_screening ORDER BY screening_date DESC')
    if not rows:
        return {'available': False}

    # --- Per-patient summary ---
    patient_map = {}
    for r in rows:
        pid = r['patient_id']
        if pid not in patient_map:
            patient_map[pid] = {
                'patient_id': pid, 'count': 0,
                'pnes_probs': [], 'classifications': [],
                'confidences': []
            }
        pm = patient_map[pid]
        pm['count'] += 1
        if r['pnes_probability'] is not None:
            pm['pnes_probs'].append(r['pnes_probability'])
        if r['confidence'] is not None:
            pm['confidences'].append(r['confidence'])
        pm['classifications'].append(r['classification'])

    per_patient = []
    for pid, pm in sorted(patient_map.items()):
        cls_counter = Counter(pm['classifications'])
        latest_cls = pm['classifications'][0] if pm['classifications'] else '--'
        per_patient.append({
            'patient_id': pid,
            'screenings': pm['count'],
            'latest_classification': latest_cls,
            'avg_pnes_probability': _avg(pm['pnes_probs']),
            'avg_confidence': _avg(pm['confidences']),
            'dominant_classification': cls_counter.most_common(1)[0][0] if cls_counter else '--',
        })

    # --- Recent screenings (last 20) ---
    recent = []
    for r in rows[:20]:
        recent.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'date': r['screening_date'],
            'referral_reason': r['referral_reason'] or '--',
            'classification': r['classification'],
            'pnes_probability': r['pnes_probability'],
            'epilepsy_probability': r['epilepsy_probability'],
            'confidence': r['confidence'],
            'eye_closure': r['eye_closure_score'],
            'pelvic_thrusting': r['pelvic_thrusting_score'],
            'side_to_side_head': r['side_to_side_head_score'],
            'ictal_crying': r['ictal_crying_score'],
            'memory_recall': r['memory_recall_score'],
            'gradual_onset': r['gradual_onset_score'],
            'duration_gt_2min': r['duration_gt_2min'],
            'eeg_ictal_normal': r['eeg_ictal_normal'],
            'eeg_interictal_normal': r['eeg_interictal_normal'],
            'trauma_history': r['trauma_history'],
            'conversion_features': r['conversion_features'],
            'psychiatric_comorbidity': r['psychiatric_comorbidity'] or '--',
            'video_eeg_recommended': r['video_eeg_recommended'],
            'psychiatry_referral': r['psychiatry_referral'],
            'status': r['status'],
            'reviewed_by': r['reviewed_by'] or '--',
        })

    # --- Semiological feature comparison across classifications ---
    feature_keys = [
        ('eye_closure_score', 'Eye Closure'),
        ('pelvic_thrusting_score', 'Pelvic Thrusting'),
        ('side_to_side_head_score', 'Side-to-Side Head'),
        ('ictal_crying_score', 'Ictal Crying'),
        ('memory_recall_score', 'Memory Recall'),
        ('gradual_onset_score', 'Gradual Onset'),
    ]
    classifications = ['pnes_likely', 'epilepsy_likely', 'mixed', 'indeterminate']
    feature_comparison = []
    for key, label in feature_keys:
        entry = {'feature': label}
        for cls in classifications:
            cls_rows = [r for r in rows if r['classification'] == cls]
            vals = [r[key] for r in cls_rows if r[key] is not None]
            entry[cls] = _avg(vals)
        feature_comparison.append(entry)

    # --- EEG finding patterns ---
    eeg_patterns = []
    for cls in classifications:
        cls_rows = [r for r in rows if r['classification'] == cls]
        n = len(cls_rows)
        if n == 0:
            continue
        ictal_normal = sum(1 for r in cls_rows if r['eeg_ictal_normal'])
        interictal_normal = sum(1 for r in cls_rows if r['eeg_interictal_normal'])
        eeg_patterns.append({
            'classification': cls,
            'total': n,
            'ictal_normal_pct': _pct(ictal_normal, n),
            'interictal_normal_pct': _pct(interictal_normal, n),
        })

    return {
        'available': True,
        'per_patient': per_patient,
        'recent_screenings': recent,
        'feature_comparison': feature_comparison,
        'eeg_patterns': eeg_patterns,
    }


def definitions():
    """PNES screening glossary — semiological signs, classification tiers,
    EEG interpretation, scoring."""
    return {
        'glossary': [
            {'term': 'PNES', 'definition': 'Psychogenic Non-Epileptic Seizures — events that resemble epileptic seizures but are not caused by abnormal electrical brain activity. They are associated with psychological factors such as stress, trauma, or conversion disorder.'},
            {'term': 'Semiological Signs', 'definition': 'Observable clinical features of seizure-like events used to differentiate PNES from epileptic seizures. Key indicators include eye closure, pelvic thrusting, side-to-side head movements, and ictal crying.'},
            {'term': 'Eye Closure Score', 'definition': 'Forced eye closure during an event (0-3). Prominent eye closure during a seizure is strongly suggestive of PNES — epileptic seizures typically present with eyes open.'},
            {'term': 'Pelvic Thrusting Score', 'definition': 'Rhythmic pelvic thrusting during an event (0-3). While not exclusive to PNES, it is significantly more common in psychogenic events than in epileptic seizures.'},
            {'term': 'Side-to-Side Head Score', 'definition': 'Repetitive side-to-side head movements during an event (0-3). This pattern is more characteristic of PNES than epileptic seizures, which tend to show versive head turning.'},
            {'term': 'Ictal Crying Score', 'definition': 'Crying or emotional vocalization during or immediately after an event (0-3). Ictal crying is a strong PNES indicator, rarely seen in epileptic seizures.'},
            {'term': 'Memory Recall Score', 'definition': 'Ability to recall events during the episode (0-3). Preserved recall or detailed memory of the event may suggest PNES, though some PNES patients also report amnesia.'},
            {'term': 'Gradual Onset Score', 'definition': 'Whether the event begins gradually rather than abruptly (0-3). Gradual onset is more typical of PNES; epileptic seizures usually begin suddenly.'},
            {'term': 'Duration > 2 min', 'definition': 'Whether the event lasted longer than 2 minutes (binary). Prolonged duration is suggestive of PNES — most epileptic seizures resolve within 1-2 minutes.'},
            {'term': 'EEG Ictal Normal', 'definition': 'Normal EEG recording during the clinical event (binary). A normal ictal EEG during a seizure-like event is the gold-standard indicator for PNES.'},
            {'term': 'EEG Interictal Normal', 'definition': 'Normal EEG between events (binary). Normal interictal EEG is common in PNES but does not rule out epilepsy, as up to 50% of epilepsy patients may have normal interictal EEGs.'},
            {'term': 'Trauma History', 'definition': 'History of psychological trauma such as abuse, neglect, or PTSD (binary). Trauma is present in 40-100% of PNES patients and is a significant risk factor.'},
            {'term': 'Conversion Features', 'definition': 'Presence of conversion disorder symptoms such as non-epileptic weakness, sensory loss, or psychogenic movement disorders (binary).'},
        ],
        'classification_tiers': [
            {'classification': 'pnes_likely', 'description': 'High semiological scores + normal ictal EEG + psychological risk factors. PNES probability >= 0.65.', 'action': 'Psychiatry referral, video-EEG confirmation, psychotherapy evaluation'},
            {'classification': 'epilepsy_likely', 'description': 'Abnormal EEG, low semiological PNES scores, typical epileptic semiology. Epilepsy probability >= 0.65.', 'action': 'Continue ASM management, standard epilepsy follow-up'},
            {'classification': 'mixed', 'description': 'Features of both PNES and epilepsy present. Both probabilities in moderate range. Possible concurrent PNES + epilepsy.', 'action': 'Prolonged video-EEG monitoring, dual management pathway'},
            {'classification': 'indeterminate', 'description': 'Insufficient evidence for clear classification. Low confidence in either direction.', 'action': 'Additional monitoring, repeat assessment, gather more history'},
        ],
        'scoring_guide': [
            {'score': 0, 'label': 'Absent', 'description': 'Feature not observed during the event'},
            {'score': 1, 'label': 'Mild', 'description': 'Feature possibly present or subtle'},
            {'score': 2, 'label': 'Moderate', 'description': 'Feature clearly present'},
            {'score': 3, 'label': 'Severe', 'description': 'Feature prominent and sustained'},
        ],
    }
