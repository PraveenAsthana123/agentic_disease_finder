"""Voice AI Dashboard — vocal biomarker extraction, prosody analysis,
articulation/fluency/swallowing assessment from clinical voice evaluations.

Maps clinical.db tables to voice AI concepts:
- assessments (WAB)            -> aphasia severity via Western Aphasia Battery
- assessments (VERBAL_FLUENCY) -> phonemic/semantic fluency scores
- assessments (MASA)           -> swallowing/articulation (Mann Assessment)
- assessments (BNT)            -> naming ability (Boston Naming Test)
- assessments (DIGIT_SPAN)     -> auditory working memory
- patients                     -> per-patient vocal profiles
- conversation_log             -> voice interaction transcripts
- transaction_log              -> voice pipeline events
"""

import sqlite3
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Voice-relevant assessment instruments
VOICE_INSTRUMENTS = ['WAB', 'VERBAL_FLUENCY', 'MASA', 'BNT', 'DIGIT_SPAN']

INSTRUMENT_LABELS = {
    'WAB': 'Western Aphasia Battery',
    'VERBAL_FLUENCY': 'Verbal Fluency',
    'MASA': 'Mann Assessment of Swallowing Ability',
    'BNT': 'Boston Naming Test',
    'DIGIT_SPAN': 'Digit Span (Auditory Memory)',
}

INSTRUMENT_DOMAIN = {
    'WAB': 'Aphasia / Language',
    'VERBAL_FLUENCY': 'Fluency / Executive',
    'MASA': 'Swallowing / Articulation',
    'BNT': 'Naming / Word Retrieval',
    'DIGIT_SPAN': 'Auditory Working Memory',
}

SEVERITY_COLORS = {
    'normal': '#10b981',
    'mild': '#f59e0b',
    'moderate': '#f97316',
    'severe': '#ef4444',
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


def _load_voice_assessments():
    placeholders = ','.join('?' for _ in VOICE_INSTRUMENTS)
    return _db_query(
        f"SELECT id, patient_id, instrument, score, max_score, "
        f"interpretation, level, alert, examiner, created_at "
        f"FROM assessments WHERE instrument IN ({placeholders}) "
        f"ORDER BY created_at",
        VOICE_INSTRUMENTS
    )


def _load_patients():
    return _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients ORDER BY patient_id"
    )


def _load_conversations():
    return _db_query(
        "SELECT id, role, text, ts_utc, ts_local "
        "FROM conversation_log ORDER BY ts_utc"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component IN ('patient_chat', 'genai_bot', 'eeg_upload') "
        "ORDER BY ts_utc"
    )


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Voice AI dashboard."""
    assessments = _load_voice_assessments()
    patients = _load_patients()
    conversations = _load_conversations()
    pipeline = _load_pipeline_events()

    if not assessments:
        return {
            'available': False,
            'message': 'No voice assessment data available. '
                       'Run WAB, VERBAL_FLUENCY, MASA, BNT, or DIGIT_SPAN assessments first.',
        }

    total_assessments = len(assessments)

    # Instrument counts
    instrument_counts = Counter(a['instrument'] for a in assessments)

    # Unique patients assessed
    patient_ids = set(a['patient_id'] for a in assessments if a.get('patient_id'))
    patients_assessed = len(patient_ids)

    # Severity distribution
    level_counts = Counter(a.get('level', 'unknown') for a in assessments)

    # Mean normalized score (score / max_score)
    norm_scores = []
    for a in assessments:
        if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
            norm_scores.append(a['score'] / a['max_score'])
    mean_norm_score = _avg(norm_scores)

    # Alerts count
    alerts = [a for a in assessments if a.get('alert')]
    alert_count = len(alerts)

    # Abnormal rate (non-normal level)
    abnormal = sum(1 for a in assessments if a.get('level') and a['level'] != 'normal')
    abnormal_rate = round(abnormal / total_assessments, 4) if total_assessments else 0

    # Voice interactions (conversations)
    total_interactions = len(conversations)

    # Pipeline events
    pipeline_count = len(pipeline)

    # --- Chart data ---

    # Instrument distribution (pie)
    instrument_distribution = [
        {
            'instrument': inst,
            'label': INSTRUMENT_LABELS.get(inst, inst),
            'count': instrument_counts.get(inst, 0),
        }
        for inst in VOICE_INSTRUMENTS
        if instrument_counts.get(inst, 0) > 0
    ]

    # Severity distribution (bar)
    severity_order = ['normal', 'mild', 'moderate', 'severe']
    severity_distribution = [
        {
            'level': lvl,
            'count': level_counts.get(lvl, 0),
            'color': SEVERITY_COLORS.get(lvl, '#6b7280'),
        }
        for lvl in severity_order
        if level_counts.get(lvl, 0) > 0
    ]

    # Per-instrument mean score (bar)
    inst_scores = {}
    for a in assessments:
        inst = a['instrument']
        if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
            inst_scores.setdefault(inst, []).append(a['score'] / a['max_score'])
    instrument_scores = [
        {
            'instrument': inst,
            'label': INSTRUMENT_LABELS.get(inst, inst),
            'mean_score': _avg(scores),
            'n': len(scores),
        }
        for inst, scores in sorted(inst_scores.items())
    ]

    # Daily assessment activity (line)
    daily_counts = Counter()
    for a in assessments:
        day = (a.get('created_at') or '')[:10]
        if day:
            daily_counts[day] += 1
    daily_activity = [
        {'date': day, 'count': cnt}
        for day, cnt in sorted(daily_counts.items())
    ]

    return {
        'available': True,
        'total_assessments': total_assessments,
        'instruments_used': len(instrument_counts),
        'patients_assessed': patients_assessed,
        'mean_normalized_score': mean_norm_score,
        'abnormal_rate': abnormal_rate,
        'alert_count': alert_count,
        'total_interactions': total_interactions,
        'pipeline_events': pipeline_count,
        'instrument_distribution': instrument_distribution,
        'severity_distribution': severity_distribution,
        'instrument_scores': instrument_scores,
        'daily_activity': daily_activity,
        'kpis': [
            {'label': 'Voice Assessments', 'value': str(total_assessments)},
            {'label': 'Instruments Used', 'value': str(len(instrument_counts))},
            {'label': 'Patients Assessed', 'value': str(patients_assessed)},
            {'label': 'Mean Score', 'value': f'{mean_norm_score:.1%}',
             'color': '#10b981' if mean_norm_score >= 0.8 else '#f59e0b' if mean_norm_score >= 0.6 else '#ef4444'},
            {'label': 'Abnormal Rate', 'value': f'{abnormal_rate:.1%}',
             'color': '#ef4444' if abnormal_rate > 0.3 else '#f59e0b' if abnormal_rate > 0.15 else '#10b981'},
            {'label': 'Clinical Alerts', 'value': str(alert_count),
             'color': '#ef4444' if alert_count > 5 else '#f59e0b' if alert_count > 0 else '#10b981'},
            {'label': 'Voice Interactions', 'value': str(total_interactions)},
            {'label': 'Pipeline Events', 'value': str(pipeline_count)},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed voice AI breakdown — assessment inventory, per-patient profiles,
    instrument statistics, clinical alerts, pipeline events."""
    assessments = _load_voice_assessments()
    patients = _load_patients()
    pipeline = _load_pipeline_events()

    if not assessments:
        return {'available': False}

    patient_map = {p['patient_id']: p for p in patients}

    # --- Assessment inventory ---
    assessment_inventory = []
    for a in assessments:
        pct = None
        if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
            pct = round(a['score'] / a['max_score'] * 100, 1)
        assessment_inventory.append({
            'id': a['id'],
            'patient_id': a['patient_id'],
            'instrument': a['instrument'],
            'instrument_label': INSTRUMENT_LABELS.get(a['instrument'], a['instrument']),
            'domain': INSTRUMENT_DOMAIN.get(a['instrument'], ''),
            'score': a['score'],
            'max_score': a['max_score'],
            'pct': pct,
            'interpretation': a.get('interpretation'),
            'level': a.get('level'),
            'alert': a.get('alert'),
            'examiner': a.get('examiner'),
            'created_at': a.get('created_at'),
        })

    # --- Patient profiles ---
    patient_assessments = {}
    for a in assessments:
        pid = a['patient_id']
        patient_assessments.setdefault(pid, []).append(a)

    patient_profiles = []
    for pid in sorted(patient_assessments.keys()):
        pa = patient_assessments[pid]
        instruments = sorted(set(a['instrument'] for a in pa))
        norm_scores = []
        levels = []
        alerts = []
        for a in pa:
            if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
                norm_scores.append(a['score'] / a['max_score'])
            if a.get('level'):
                levels.append(a['level'])
            if a.get('alert'):
                alerts.append(a['alert'])

        worst_level = 'normal'
        for sev in ['severe', 'moderate', 'mild']:
            if sev in levels:
                worst_level = sev
                break

        pinfo = patient_map.get(pid, {})
        patient_profiles.append({
            'patient_id': pid,
            'name': pinfo.get('name', ''),
            'age': pinfo.get('age'),
            'gender': pinfo.get('gender'),
            'disease': pinfo.get('disease'),
            'n_assessments': len(pa),
            'instruments': instruments,
            'instrument_labels': [INSTRUMENT_LABELS.get(i, i) for i in instruments],
            'mean_score': _avg(norm_scores),
            'worst_level': worst_level,
            'alert_count': len(alerts),
        })

    # --- Instrument stats ---
    inst_groups = {}
    for a in assessments:
        inst = a['instrument']
        inst_groups.setdefault(inst, {'scores': [], 'levels': [], 'alerts': 0})
        if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
            inst_groups[inst]['scores'].append(a['score'] / a['max_score'])
        if a.get('level'):
            inst_groups[inst]['levels'].append(a['level'])
        if a.get('alert'):
            inst_groups[inst]['alerts'] += 1

    instrument_stats = []
    for inst in VOICE_INSTRUMENTS:
        if inst not in inst_groups:
            continue
        g = inst_groups[inst]
        lc = Counter(g['levels'])
        instrument_stats.append({
            'instrument': inst,
            'label': INSTRUMENT_LABELS.get(inst, inst),
            'domain': INSTRUMENT_DOMAIN.get(inst, ''),
            'count': len(g['scores']),
            'mean_score': _avg(g['scores']),
            'normal': lc.get('normal', 0),
            'mild': lc.get('mild', 0),
            'moderate': lc.get('moderate', 0),
            'severe': lc.get('severe', 0),
            'alerts': g['alerts'],
        })

    # --- Clinical alerts ---
    clinical_alerts = [
        {
            'id': a['id'],
            'patient_id': a['patient_id'],
            'instrument': a['instrument'],
            'level': a.get('level'),
            'alert': a.get('alert'),
            'score': a.get('score'),
            'max_score': a.get('max_score'),
            'created_at': a.get('created_at'),
        }
        for a in assessments if a.get('alert')
    ]

    # --- Pipeline events ---
    pipeline_events = [
        {
            'id': ev['id'],
            'component': ev.get('component'),
            'action': ev.get('action'),
            'actor': ev.get('actor'),
            'detail': ev.get('detail'),
            'ts_utc': ev.get('ts_utc'),
            'ts_local': ev.get('ts_local'),
        }
        for ev in pipeline
    ]

    return {
        'available': True,
        'assessment_inventory': assessment_inventory,
        'patient_profiles': patient_profiles,
        'instrument_stats': instrument_stats,
        'clinical_alerts': clinical_alerts,
        'pipeline_events': pipeline_events,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Voice AI dashboard."""
    return {
        'concepts': [
            {
                'name': 'Voice Biomarker',
                'description': 'Measurable vocal characteristics (pitch, jitter, shimmer, '
                               'harmonic-to-noise ratio, formant frequencies) that correlate '
                               'with neurological states. In epilepsy care, voice biomarkers '
                               'detect post-ictal dysarthria, medication-induced speech changes, '
                               'and cognitive decline trajectories.',
            },
            {
                'name': 'Prosody Analysis',
                'description': 'Quantitative examination of speech rhythm, stress, intonation, '
                               'and timing patterns. Prosodic features reveal neurological '
                               'impairment severity, fatigue levels, and emotional distress. '
                               'Measured via F0 contour, speech rate, and pause duration.',
            },
            {
                'name': 'Western Aphasia Battery (WAB)',
                'description': 'Standardized clinical assessment of language function measuring '
                               'spontaneous speech, auditory comprehension, repetition, and '
                               'naming. Yields an Aphasia Quotient (AQ) score out of 100. '
                               'Scores below 93.8 indicate aphasia.',
            },
            {
                'name': 'Verbal Fluency Test',
                'description': 'Neuropsychological assessment where patients generate words '
                               'beginning with a specific letter (phonemic) or from a category '
                               '(semantic) within a time limit. Measures executive function, '
                               'language retrieval speed, and frontal lobe integrity.',
            },
            {
                'name': 'MASA (Mann Assessment of Swallowing Ability)',
                'description': 'Bedside clinical assessment of swallowing function evaluating '
                               'oral motor control, pharyngeal response, and aspiration risk. '
                               'Scores out of 200; higher scores indicate better function. '
                               'Critical for post-ictal and post-surgical patients.',
            },
            {
                'name': 'Boston Naming Test (BNT)',
                'description': 'Visual confrontation naming test where patients name 60 drawn '
                               'objects of increasing difficulty. Measures word retrieval and '
                               'lexical access. Sensitive to anomia in temporal lobe epilepsy.',
            },
            {
                'name': 'Digit Span',
                'description': 'Auditory working memory test with forward (immediate recall) '
                               'and backward (manipulation) components. Assesses phonological '
                               'loop capacity and central executive function. Sensitive to '
                               'anti-epileptic drug effects on cognitive processing.',
            },
            {
                'name': 'Acoustic Feature Extraction',
                'description': 'Computational extraction of vocal signal properties including '
                               'fundamental frequency (F0), mel-frequency cepstral coefficients '
                               '(MFCCs), spectral flux, zero-crossing rate, and energy contour. '
                               'These features feed into voice biomarker classification models.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Assessment Score (Normalized)',
                'description': 'Score expressed as percentage of maximum possible score. '
                               'Enables cross-instrument comparison. ≥80% typically normal, '
                               '60–80% mild impairment, <60% moderate-severe impairment.',
            },
            {
                'name': 'Inter-rater Reliability',
                'description': 'Degree of agreement between independent examiners scoring '
                               'the same voice assessment. Measured via Cohen\'s kappa or '
                               'intraclass correlation coefficient (ICC). Target: ICC > 0.85.',
            },
            {
                'name': 'Test-Retest Reliability',
                'description': 'Consistency of assessment scores when administered to the '
                               'same patient at different time points under similar conditions. '
                               'Important for tracking longitudinal voice changes.',
            },
            {
                'name': 'Clinical Alert Sensitivity',
                'description': 'Proportion of clinically significant voice impairments that '
                               'trigger an alert. Targets ≥95% sensitivity for moderate-severe '
                               'findings to avoid missed referrals.',
            },
        ],
        'assessment_instruments': [
            {
                'instrument': inst,
                'label': INSTRUMENT_LABELS.get(inst, inst),
                'domain': INSTRUMENT_DOMAIN.get(inst, ''),
            }
            for inst in VOICE_INSTRUMENTS
        ],
        'compliance': [
            {
                'ref': 'FDA AI/ML Framework',
                'note': 'Voice biomarker models used for clinical decision support must '
                        'follow the Predetermined Change Control Plan (PCCP). Acoustic '
                        'feature extraction and classification accuracy must be validated '
                        'against expert clinician assessment scores.',
            },
            {
                'ref': 'EU AI Act Art. 6',
                'note': 'Voice AI systems used in medical diagnosis are classified as '
                        'high-risk. Require conformity assessment, human oversight, and '
                        'transparency obligations including disclosure of AI involvement '
                        'in voice-based clinical assessments.',
            },
            {
                'ref': 'ISO 14971',
                'note': 'Risk management must address misclassification of voice '
                        'impairment severity, false-negative aphasia screening, and '
                        'failure to detect aspiration risk from MASA-equivalent analysis.',
            },
            {
                'ref': 'IEC 62304',
                'note': 'Voice processing software components (acoustic extraction, '
                        'biomarker classification, severity scoring) must follow medical '
                        'device software lifecycle processes with documented V&V.',
            },
            {
                'ref': 'HIPAA',
                'note': 'Voice recordings and assessment transcripts contain PHI. Systems '
                        'must encrypt audio at rest and in transit, enforce role-based access, '
                        'and maintain audit trails for all voice data access.',
            },
        ],
        'remediation': [
            {
                'strategy': 'Biomarker Model Recalibration',
                'description': 'When voice biomarker classification accuracy drops below '
                               'threshold, recalibrate acoustic feature thresholds using '
                               'recent clinician-scored assessments as ground truth.',
            },
            {
                'strategy': 'Assessment Protocol Standardization',
                'description': 'Ensure consistent administration conditions (quiet room, '
                               'calibrated microphone, standardized prompts) across all '
                               'voice assessment sessions to minimize environmental variance.',
            },
            {
                'strategy': 'Longitudinal Baseline Update',
                'description': 'Update patient-specific voice baselines after significant '
                               'clinical events (surgery, medication change, seizure cluster) '
                               'to prevent false-positive drift alerts.',
            },
            {
                'strategy': 'Cross-Instrument Correlation Audit',
                'description': 'Periodically verify that WAB, BNT, Verbal Fluency, and MASA '
                               'scores remain internally consistent for each patient. Flag '
                               'discordant profiles for clinician review.',
            },
        ],
    }
