"""Voice Assessment Mapping Dashboard — maps STT (voice) intake to structured
clinical assessment forms.

Reads from clinical.db tables:
- guided_assessment_sessions  -> voice_ai channel sessions (STT -> form mapping)
- assessments                 -> completed scored assessments
- patients                    -> patient demographics

Shows: mapping pipeline status, instrument coverage, completion rates,
score distributions by channel, per-patient mapping profiles, quality metrics.
"""

import sqlite3
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

INSTRUMENT_LABELS = {
    'PHQ9': 'Patient Health Questionnaire-9',
    'GAD7': 'Generalized Anxiety Disorder-7',
    'NDDIE': 'Neurological Disorders Depression Inventory (Epilepsy)',
    'MOCA': 'Montreal Cognitive Assessment',
    'QOLIE31': 'Quality of Life in Epilepsy-31',
    'MMSE': 'Mini-Mental State Examination',
    'BARTHEL': 'Barthel Index (Activities of Daily Living)',
    'EPWORTH': 'Epworth Sleepiness Scale',
    'BNT': 'Boston Naming Test',
    'WAB': 'Western Aphasia Battery',
    'VERBAL_FLUENCY': 'Verbal Fluency Test',
    'MASA': 'Mann Assessment of Swallowing Ability',
    'LSSS': 'Liverpool Seizure Severity Scale',
    'CSSRS': 'Columbia Suicide Severity Rating Scale',
    'WAIS': 'Wechsler Adult Intelligence Scale',
    'DIGIT_SPAN': 'Digit Span (Auditory Memory)',
}

INSTRUMENT_DOMAIN = {
    'PHQ9': 'Mood / Depression',
    'GAD7': 'Anxiety',
    'NDDIE': 'Epilepsy-specific Depression',
    'MOCA': 'Cognitive Screening',
    'QOLIE31': 'Quality of Life',
    'MMSE': 'Cognitive Screening',
    'BARTHEL': 'Functional Status',
    'EPWORTH': 'Sleepiness / Fatigue',
    'BNT': 'Language / Naming',
    'WAB': 'Aphasia / Language',
    'VERBAL_FLUENCY': 'Fluency / Executive',
    'MASA': 'Swallowing / Articulation',
    'LSSS': 'Seizure Severity',
    'CSSRS': 'Suicide Risk',
    'WAIS': 'Intelligence / Cognition',
    'DIGIT_SPAN': 'Auditory Working Memory',
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


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary of voice-to-assessment mapping pipeline."""
    sessions = _db_query(
        "SELECT * FROM guided_assessment_sessions ORDER BY started_at"
    )
    if not sessions:
        return {
            'available': False,
            'message': 'No guided assessment sessions found. '
                       'Run voice or conversational assessments first.',
        }

    voice_sessions = [s for s in sessions if s.get('channel') == 'voice_ai']
    chat_sessions = [s for s in sessions if s.get('channel') == 'conversational_ai']

    total = len(sessions)
    voice_count = len(voice_sessions)
    chat_count = len(chat_sessions)

    # Completion rates
    completed = [s for s in sessions if s.get('status') == 'completed']
    abandoned = [s for s in sessions if s.get('status') == 'abandoned']
    in_progress = [s for s in sessions if s.get('status') == 'in_progress']
    completion_rate = round(len(completed) / total, 4) if total else 0

    voice_completed = [s for s in voice_sessions if s.get('status') == 'completed']
    voice_completion_rate = round(len(voice_completed) / voice_count, 4) if voice_count else 0

    # Instruments covered by voice
    voice_instruments = sorted(set(s['instrument'] for s in voice_sessions))
    all_instruments = sorted(set(s['instrument'] for s in sessions))

    # Unique patients with voice assessments
    voice_patients = set(s['patient_id'] for s in voice_sessions if s.get('patient_id'))

    # Mean duration
    voice_durations = [s['duration_seconds'] for s in voice_completed if s.get('duration_seconds')]
    chat_durations = [s['duration_seconds'] for s in chat_sessions
                      if s.get('status') == 'completed' and s.get('duration_seconds')]
    mean_voice_duration = _avg(voice_durations)
    mean_chat_duration = _avg(chat_durations)

    # Mean scores (voice vs chat completed)
    voice_scores = [s['score'] / s['max_score'] for s in voice_completed
                    if s.get('score') is not None and s.get('max_score') and s['max_score'] > 0]
    chat_completed = [s for s in chat_sessions if s.get('status') == 'completed']
    chat_scores = [s['score'] / s['max_score'] for s in chat_completed
                   if s.get('score') is not None and s.get('max_score') and s['max_score'] > 0]

    # Item completion (current_item / total_items) for voice
    voice_item_rates = []
    for s in voice_sessions:
        if s.get('current_item') and s.get('total_items') and s['total_items'] > 0:
            voice_item_rates.append(s['current_item'] / s['total_items'])

    # Channel distribution (pie)
    channel_distribution = [
        {'channel': 'Voice AI (STT)', 'count': voice_count},
        {'channel': 'Conversational AI', 'count': chat_count},
    ]

    # Status distribution (bar)
    status_distribution = [
        {'status': 'Completed', 'count': len(completed), 'color': '#10b981'},
        {'status': 'In Progress', 'count': len(in_progress), 'color': '#3b82f6'},
        {'status': 'Abandoned', 'count': len(abandoned), 'color': '#ef4444'},
    ]

    # Instrument coverage by channel (grouped bar)
    inst_by_channel = {}
    for s in sessions:
        inst = s['instrument']
        ch = 'voice' if s.get('channel') == 'voice_ai' else 'chat'
        inst_by_channel.setdefault(inst, {'voice': 0, 'chat': 0})
        inst_by_channel[inst][ch] += 1

    instrument_coverage = [
        {
            'instrument': inst,
            'label': INSTRUMENT_LABELS.get(inst, inst),
            'voice': counts['voice'],
            'chat': counts['chat'],
        }
        for inst, counts in sorted(inst_by_channel.items())
    ]

    # Daily mapping activity (line)
    daily = Counter()
    for s in voice_sessions:
        day = (s.get('started_at') or '')[:10]
        if day:
            daily[day] += 1
    daily_activity = [{'date': d, 'count': c} for d, c in sorted(daily.items())]

    return {
        'available': True,
        'total_sessions': total,
        'voice_sessions': voice_count,
        'chat_sessions': chat_count,
        'voice_completion_rate': voice_completion_rate,
        'overall_completion_rate': completion_rate,
        'voice_instruments': voice_instruments,
        'voice_patients': len(voice_patients),
        'mean_voice_duration_s': mean_voice_duration,
        'mean_chat_duration_s': mean_chat_duration,
        'mean_voice_score': _avg(voice_scores),
        'mean_chat_score': _avg(chat_scores),
        'mean_item_completion': _avg(voice_item_rates),
        'channel_distribution': channel_distribution,
        'status_distribution': status_distribution,
        'instrument_coverage': instrument_coverage,
        'daily_activity': daily_activity,
        'kpis': [
            {'label': 'Voice Mappings', 'value': str(voice_count)},
            {'label': 'Instruments (Voice)', 'value': str(len(voice_instruments))},
            {'label': 'Patients (Voice)', 'value': str(len(voice_patients))},
            {'label': 'Voice Completion',
             'value': f'{voice_completion_rate:.0%}',
             'color': '#10b981' if voice_completion_rate >= 0.8 else '#f59e0b' if voice_completion_rate >= 0.5 else '#ef4444'},
            {'label': 'Avg Duration (Voice)',
             'value': f'{mean_voice_duration:.0f}s' if mean_voice_duration else 'N/A'},
            {'label': 'Avg Score (Voice)',
             'value': f'{_avg(voice_scores):.0%}' if voice_scores else 'N/A',
             'color': '#10b981' if _avg(voice_scores) >= 0.7 else '#f59e0b'},
            {'label': 'Item Fill Rate',
             'value': f'{_avg(voice_item_rates):.0%}' if voice_item_rates else 'N/A'},
            {'label': 'Total Sessions', 'value': str(total)},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed voice assessment mapping breakdown — session inventory,
    per-patient mapping profiles, instrument stats, channel comparison."""
    sessions = _db_query(
        "SELECT * FROM guided_assessment_sessions ORDER BY started_at"
    )
    patients = _db_query(
        "SELECT patient_id, name, age, gender, disease FROM patients ORDER BY patient_id"
    )
    if not sessions:
        return {'available': False}

    patient_map = {p['patient_id']: p for p in patients}
    voice_sessions = [s for s in sessions if s.get('channel') == 'voice_ai']

    # --- Session inventory (voice only) ---
    session_inventory = []
    for s in voice_sessions:
        pct = None
        if s.get('score') is not None and s.get('max_score') and s['max_score'] > 0:
            pct = round(s['score'] / s['max_score'] * 100, 1)
        item_pct = None
        if s.get('current_item') and s.get('total_items') and s['total_items'] > 0:
            item_pct = round(s['current_item'] / s['total_items'] * 100, 1)
        pinfo = patient_map.get(s.get('patient_id'), {})
        session_inventory.append({
            'session_id': s['session_id'],
            'patient_id': s.get('patient_id'),
            'patient_name': pinfo.get('name', ''),
            'instrument': s['instrument'],
            'instrument_label': INSTRUMENT_LABELS.get(s['instrument'], s['instrument']),
            'domain': INSTRUMENT_DOMAIN.get(s['instrument'], ''),
            'status': s.get('status'),
            'items_completed': s.get('current_item'),
            'total_items': s.get('total_items'),
            'item_completion_pct': item_pct,
            'score': s.get('score'),
            'max_score': s.get('max_score'),
            'score_pct': pct,
            'interpretation': s.get('interpretation'),
            'duration_seconds': s.get('duration_seconds'),
            'started_at': s.get('started_at'),
            'completed_at': s.get('completed_at'),
        })

    # --- Per-patient mapping profiles ---
    patient_sessions = {}
    for s in voice_sessions:
        pid = s.get('patient_id')
        if pid:
            patient_sessions.setdefault(pid, []).append(s)

    patient_profiles = []
    for pid in sorted(patient_sessions.keys()):
        ps = patient_sessions[pid]
        instruments = sorted(set(s['instrument'] for s in ps))
        completed = sum(1 for s in ps if s.get('status') == 'completed')
        scores = [s['score'] / s['max_score'] for s in ps
                  if s.get('status') == 'completed' and s.get('score') is not None
                  and s.get('max_score') and s['max_score'] > 0]
        durations = [s['duration_seconds'] for s in ps
                     if s.get('status') == 'completed' and s.get('duration_seconds')]
        pinfo = patient_map.get(pid, {})
        patient_profiles.append({
            'patient_id': pid,
            'name': pinfo.get('name', ''),
            'age': pinfo.get('age'),
            'gender': pinfo.get('gender'),
            'disease': pinfo.get('disease'),
            'total_voice_sessions': len(ps),
            'completed': completed,
            'instruments': instruments,
            'instrument_labels': [INSTRUMENT_LABELS.get(i, i) for i in instruments],
            'mean_score': _avg(scores),
            'mean_duration': _avg(durations),
        })

    # --- Instrument mapping stats ---
    inst_groups = {}
    for s in voice_sessions:
        inst = s['instrument']
        inst_groups.setdefault(inst, {'completed': 0, 'abandoned': 0, 'in_progress': 0,
                                      'scores': [], 'durations': []})
        g = inst_groups[inst]
        st = s.get('status', '')
        if st == 'completed':
            g['completed'] += 1
        elif st == 'abandoned':
            g['abandoned'] += 1
        elif st == 'in_progress':
            g['in_progress'] += 1
        if st == 'completed' and s.get('score') is not None and s.get('max_score') and s['max_score'] > 0:
            g['scores'].append(s['score'] / s['max_score'])
        if st == 'completed' and s.get('duration_seconds'):
            g['durations'].append(s['duration_seconds'])

    instrument_stats = []
    for inst in sorted(inst_groups.keys()):
        g = inst_groups[inst]
        total = g['completed'] + g['abandoned'] + g['in_progress']
        instrument_stats.append({
            'instrument': inst,
            'label': INSTRUMENT_LABELS.get(inst, inst),
            'domain': INSTRUMENT_DOMAIN.get(inst, ''),
            'total': total,
            'completed': g['completed'],
            'abandoned': g['abandoned'],
            'in_progress': g['in_progress'],
            'completion_rate': round(g['completed'] / total, 4) if total else 0,
            'mean_score': _avg(g['scores']),
            'mean_duration': _avg(g['durations']),
        })

    # --- Channel comparison ---
    chat_sessions = [s for s in sessions if s.get('channel') == 'conversational_ai']
    chat_completed = [s for s in chat_sessions if s.get('status') == 'completed']
    voice_completed = [s for s in voice_sessions if s.get('status') == 'completed']

    channel_comparison = {
        'voice': {
            'total': len(voice_sessions),
            'completed': len(voice_completed),
            'completion_rate': round(len(voice_completed) / len(voice_sessions), 4) if voice_sessions else 0,
            'mean_score': _avg([s['score'] / s['max_score'] for s in voice_completed
                                if s.get('score') is not None and s.get('max_score') and s['max_score'] > 0]),
            'mean_duration': _avg([s['duration_seconds'] for s in voice_completed
                                   if s.get('duration_seconds')]),
        },
        'chat': {
            'total': len(chat_sessions),
            'completed': len(chat_completed),
            'completion_rate': round(len(chat_completed) / len(chat_sessions), 4) if chat_sessions else 0,
            'mean_score': _avg([s['score'] / s['max_score'] for s in chat_completed
                                if s.get('score') is not None and s.get('max_score') and s['max_score'] > 0]),
            'mean_duration': _avg([s['duration_seconds'] for s in chat_completed
                                   if s.get('duration_seconds')]),
        },
    }

    return {
        'available': True,
        'session_inventory': session_inventory,
        'patient_profiles': patient_profiles,
        'instrument_stats': instrument_stats,
        'channel_comparison': channel_comparison,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Voice Assessment Mapping dashboard."""
    return {
        'concepts': [
            {
                'name': 'Voice Assessment Mapping',
                'description': 'The process of converting speech-to-text (STT) voice '
                               'input into structured clinical assessment form responses. '
                               'A patient speaks their answers to standardized instruments '
                               '(PHQ-9, GAD-7, MoCA, etc.) via voice, and the system maps '
                               'spoken responses to scored items automatically.',
            },
            {
                'name': 'STT-to-Form Pipeline',
                'description': 'End-to-end pipeline: (1) Audio capture, (2) Speech-to-text '
                               'transcription (Whisper), (3) Intent/item extraction via NLP, '
                               '(4) Answer normalization to Likert/numeric scale, (5) Form '
                               'population and scoring, (6) Clinical interpretation generation.',
            },
            {
                'name': 'Guided Assessment Session',
                'description': 'An interactive session where the system prompts the patient '
                               'through each item of a clinical instrument one at a time, '
                               'accepts voice or text responses, validates and scores them. '
                               'Tracks progress (current_item / total_items) and completion status.',
            },
            {
                'name': 'Channel Comparison',
                'description': 'Voice AI vs Conversational AI channel analytics. Voice AI '
                               'uses speech-to-text for input; Conversational AI uses typed text. '
                               'Comparing completion rates, scores, and durations across channels '
                               'validates that voice mapping produces equivalent clinical results.',
            },
            {
                'name': 'Item Fill Rate',
                'description': 'Percentage of assessment items successfully mapped from voice '
                               'input to structured responses. A fill rate below 100% indicates '
                               'items where the STT pipeline could not confidently extract a '
                               'valid response from the spoken input.',
            },
            {
                'name': 'Mapping Equivalence',
                'description': 'Statistical verification that voice-mapped assessment scores '
                               'are clinically equivalent to manually administered scores. '
                               'Measured via Bland-Altman analysis, ICC, and clinical threshold '
                               'agreement rates.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Voice Completion Rate',
                'description': 'Proportion of voice-initiated assessment sessions that reach '
                               'fully-completed status with all items answered and scored. '
                               'Target: >=80%. Low rates indicate STT recognition issues or '
                               'patient discomfort with voice interface.',
            },
            {
                'name': 'Mean Mapping Duration',
                'description': 'Average time from session start to completion for voice '
                               'channel assessments. Compared against chat channel to '
                               'ensure voice mapping does not add excessive overhead.',
            },
            {
                'name': 'Score Concordance',
                'description': 'Agreement between voice-mapped scores and reference scores '
                               '(clinician-administered or chat-administered). High concordance '
                               '(>0.90 ICC) validates the voice mapping pipeline accuracy.',
            },
            {
                'name': 'Abandonment Rate',
                'description': 'Proportion of voice sessions abandoned before completion. '
                               'High abandonment suggests usability issues with the voice '
                               'interface or STT accuracy problems for specific instruments.',
            },
        ],
        'supported_instruments': [
            {
                'instrument': inst,
                'label': INSTRUMENT_LABELS.get(inst, inst),
                'domain': INSTRUMENT_DOMAIN.get(inst, ''),
            }
            for inst in sorted(INSTRUMENT_LABELS.keys())
        ],
        'compliance': [
            {
                'ref': 'FDA AI/ML Guidance',
                'note': 'Voice-to-assessment mapping systems used for clinical scoring '
                        'must demonstrate analytical validation equivalent to manual '
                        'administration. STT accuracy and response mapping must be '
                        'validated against clinician-scored gold standard.',
            },
            {
                'ref': 'APA Testing Standards',
                'note': 'Automated administration of standardized psychological instruments '
                        'via voice must maintain standardized conditions (consistent prompts, '
                        'timing, scoring) to preserve psychometric validity.',
            },
            {
                'ref': 'HIPAA',
                'note': 'Voice recordings used for assessment mapping contain PHI. Audio '
                        'must be encrypted, access-controlled, and retention-limited. '
                        'Transcripts inherit PHI classification.',
            },
        ],
        'remediation': [
            {
                'strategy': 'STT Confidence Thresholding',
                'description': 'When STT confidence for a response falls below threshold '
                               '(e.g., <0.85), re-prompt the patient or flag the item for '
                               'manual review rather than mapping a low-confidence answer.',
            },
            {
                'strategy': 'Instrument-Specific Vocabularies',
                'description': 'Maintain per-instrument custom vocabularies for the STT '
                               'engine to improve recognition accuracy for clinical terms '
                               '(e.g., "not at all" -> 0, "several days" -> 1 for PHQ-9).',
            },
            {
                'strategy': 'Fallback to Text Input',
                'description': 'If voice mapping fails for 2+ consecutive items, offer '
                               'automatic fallback to conversational AI (text) channel '
                               'to prevent session abandonment.',
            },
        ],
    }
