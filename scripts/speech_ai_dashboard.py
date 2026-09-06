"""Speech AI Dashboard — speech transcription, NLP analysis, voice biomarker
extraction from clinical conversations and patient interactions.

Maps clinical.db tables to speech AI concepts:
- conversation_log  -> speech transcriptions (role=operator -> clinician speech,
                      role=assistant -> AI speech response)
- uploads           -> audio/EEG files submitted for speech analysis
- analyses          -> speech-derived predictions (confidence, signal quality)
- transaction_log   -> speech pipeline events (patient_chat, genai_bot)
"""

import sqlite3
import json
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Speech interaction type labels
INTERACTION_TYPES = {
    'operator': 'Clinician Speech',
    'assistant': 'AI Response',
}

# Transcript length tiers (character count)
LENGTH_TIERS = {
    'brief': (0, 200),
    'moderate': (200, 1000),
    'detailed': (1000, 5000),
    'extensive': (5000, float('inf')),
}

# Signal quality mapping for speech clarity
SPEECH_CLARITY_MAP = {
    'good': 0.95,
    'acceptable': 0.80,
    'poor': 0.50,
    'unknown': 0.70,
}


def _db_query(sql, params=()):
    """Execute a read-only query against clinical.db and return list of dicts."""
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _avg(values):
    """Safe average that returns 0.0 for empty sequences."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _length_tier(char_count):
    """Classify transcript length into a tier."""
    for tier, (lo, hi) in LENGTH_TIERS.items():
        if lo <= char_count < hi:
            return tier
    return 'extensive'


def _load_conversations():
    """Load all conversation_log entries."""
    return _db_query(
        "SELECT id, role, text, ts_utc, ts_local "
        "FROM conversation_log ORDER BY ts_utc"
    )


def _load_uploads():
    """Load all upload records."""
    return _db_query(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at"
    )


def _load_analyses():
    """Load all analysis records."""
    return _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, "
        "confidence, signal_quality, result_json, created_at "
        "FROM analyses ORDER BY created_at"
    )


def _load_pipeline_events():
    """Load speech-relevant pipeline events from transaction_log."""
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component IN ('patient_chat', 'genai_bot', 'eeg_upload') "
        "ORDER BY ts_utc"
    )


def _load_patients():
    """Load all patient records."""
    return _db_query(
        "SELECT id, name, age, gender, disease, department, diagnosis, created_at "
        "FROM patients ORDER BY id"
    )


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Speech AI dashboard."""
    conversations = _load_conversations()
    uploads = _load_uploads()
    analyses = _load_analyses()
    pipeline = _load_pipeline_events()

    if not conversations and not uploads:
        return {
            'available': False,
            'message': 'No speech or conversation data available. '
                       'Run patient interactions or upload audio files first.',
        }

    # --- KPI counts ---
    total_transcriptions = len(conversations)
    clinician_utterances = sum(1 for c in conversations if c.get('role') == 'operator')
    ai_responses = sum(1 for c in conversations if c.get('role') == 'assistant')
    total_audio_files = len(uploads)

    # Patients covered (union of upload patient_ids and analysis patient_ids)
    patient_ids = set()
    for u in uploads:
        if u.get('patient_id'):
            patient_ids.add(u['patient_id'])
    for a in analyses:
        if a.get('patient_id'):
            patient_ids.add(a['patient_id'])
    patients_covered = len(patient_ids)

    # Mean confidence from analyses
    confidences = [a['confidence'] for a in analyses if a.get('confidence') is not None]
    mean_confidence = _avg(confidences)

    pipeline_events = len(pipeline)

    # Mean transcript length
    text_lengths = [len(c.get('text', '') or '') for c in conversations]
    mean_transcript_length = _avg(text_lengths)

    # --- Chart data ---

    # Interaction type distribution (pie chart)
    role_counts = Counter(c.get('role', 'unknown') for c in conversations)
    interaction_type_distribution = [
        {
            'role': role,
            'label': INTERACTION_TYPES.get(role, role),
            'count': count,
        }
        for role, count in role_counts.most_common()
    ]

    # Transcript length distribution (bar chart)
    tier_counts = Counter()
    for length in text_lengths:
        tier_counts[_length_tier(length)] += 1
    transcript_length_distribution = [
        {'tier': tier, 'count': tier_counts.get(tier, 0)}
        for tier in LENGTH_TIERS.keys()
    ]

    # Daily activity (line chart — transcriptions per day)
    daily_counts = Counter()
    for c in conversations:
        ts = c.get('ts_local') or c.get('ts_utc') or ''
        day = ts[:10] if len(ts) >= 10 else ts
        if day:
            daily_counts[day] += 1
    daily_activity = [
        {'date': day, 'count': count}
        for day, count in sorted(daily_counts.items())
    ]

    # Speech pipeline health (per-component counts)
    comp_counts = Counter(ev.get('component', 'unknown') for ev in pipeline)
    speech_pipeline_health = [
        {'component': comp, 'count': cnt}
        for comp, cnt in comp_counts.most_common()
    ]

    return {
        'available': True,
        'total_transcriptions': total_transcriptions,
        'clinician_utterances': clinician_utterances,
        'ai_responses': ai_responses,
        'total_audio_files': total_audio_files,
        'patients_covered': patients_covered,
        'mean_confidence': mean_confidence,
        'pipeline_events': pipeline_events,
        'mean_transcript_length': round(mean_transcript_length, 1),
        'interaction_type_distribution': interaction_type_distribution,
        'transcript_length_distribution': transcript_length_distribution,
        'daily_activity': daily_activity,
        'speech_pipeline_health': speech_pipeline_health,
        'kpis': [
            {'label': 'Total Transcriptions', 'value': str(total_transcriptions)},
            {'label': 'Clinician Utterances', 'value': str(clinician_utterances)},
            {'label': 'AI Responses', 'value': str(ai_responses)},
            {'label': 'Audio Files', 'value': str(total_audio_files)},
            {'label': 'Patients Covered', 'value': str(patients_covered)},
            {'label': 'Mean Confidence', 'value': f'{mean_confidence:.3f}',
             'color': '#10b981' if mean_confidence >= 0.7 else '#f59e0b' if mean_confidence >= 0.5 else '#ef4444'},
            {'label': 'Pipeline Events', 'value': str(pipeline_events)},
            {'label': 'Avg Transcript Length', 'value': f'{mean_transcript_length:.0f} chars'},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed speech AI breakdown — transcription inventory, audio files,
    patient profiles, pipeline events, and per-role statistics."""
    conversations = _load_conversations()
    uploads = _load_uploads()
    analyses = _load_analyses()
    pipeline = _load_pipeline_events()

    if not conversations and not uploads:
        return {'available': False}

    # --- Transcription inventory ---
    transcription_inventory = []
    for c in conversations:
        text = c.get('text', '') or ''
        length = len(text)
        role = c.get('role', 'unknown')
        transcription_inventory.append({
            'id': c.get('id'),
            'role': role,
            'interaction_type': INTERACTION_TYPES.get(role, role),
            'text': text[:200],
            'length': length,
            'length_tier': _length_tier(length),
            'ts_utc': c.get('ts_utc'),
            'ts_local': c.get('ts_local'),
        })

    # --- Audio files ---
    audio_files = [
        {
            'id': u.get('id'),
            'patient_id': u.get('patient_id'),
            'file_name': u.get('file_name'),
            'disease': u.get('disease'),
            'created_at': u.get('created_at'),
        }
        for u in uploads
    ]

    # --- Patient profiles ---
    # Build per-patient aggregates from uploads and analyses
    patient_uploads = Counter()
    patient_diseases = {}
    for u in uploads:
        pid = u.get('patient_id')
        if pid is None:
            continue
        patient_uploads[pid] += 1
        patient_diseases.setdefault(pid, set()).add(u.get('disease', 'unknown'))

    patient_analyses_count = Counter()
    patient_confidences = {}
    for a in analyses:
        pid = a.get('patient_id')
        if pid is None:
            continue
        patient_analyses_count[pid] += 1
        patient_confidences.setdefault(pid, []).append(a.get('confidence', 0))
        patient_diseases.setdefault(pid, set()).add(a.get('disease', 'unknown'))

    all_patient_ids = set(patient_uploads.keys()) | set(patient_analyses_count.keys())
    patient_profiles = []
    for pid in sorted(all_patient_ids):
        confs = patient_confidences.get(pid, [])
        patient_profiles.append({
            'patient_id': pid,
            'n_uploads': patient_uploads.get(pid, 0),
            'n_analyses': patient_analyses_count.get(pid, 0),
            'diseases': sorted(patient_diseases.get(pid, set())),
            'mean_confidence': _avg(confs),
        })

    # --- Pipeline events ---
    pipeline_events = [
        {
            'id': ev.get('id'),
            'component': ev.get('component'),
            'action': ev.get('action'),
            'actor': ev.get('actor'),
            'detail': ev.get('detail'),
            'ts_utc': ev.get('ts_utc'),
            'ts_local': ev.get('ts_local'),
        }
        for ev in pipeline
    ]

    # --- Role stats ---
    role_groups = {}
    for c in conversations:
        role = c.get('role', 'unknown')
        text = c.get('text', '') or ''
        if role not in role_groups:
            role_groups[role] = {'lengths': [], 'count': 0, 'total_chars': 0}
        role_groups[role]['count'] += 1
        role_groups[role]['lengths'].append(len(text))
        role_groups[role]['total_chars'] += len(text)

    role_stats = []
    for role, info in sorted(role_groups.items()):
        role_stats.append({
            'role': role,
            'label': INTERACTION_TYPES.get(role, role),
            'count': info['count'],
            'mean_length': _avg(info['lengths']),
            'total_chars': info['total_chars'],
        })

    return {
        'available': True,
        'transcription_inventory': transcription_inventory,
        'audio_files': audio_files,
        'patient_profiles': patient_profiles,
        'pipeline_events': pipeline_events,
        'role_stats': role_stats,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Speech AI dashboard."""
    return {
        'concepts': [
            {
                'name': 'ASR (Automatic Speech Recognition)',
                'description': 'Technology that converts spoken language into written '
                               'text. In clinical settings, ASR transcribes clinician-'
                               'patient conversations, dictated notes, and audio from '
                               'video-EEG monitoring sessions for downstream NLP analysis.',
            },
            {
                'name': 'NLP (Natural Language Processing)',
                'description': 'Computational analysis of human language to extract '
                               'meaning, entities, and intent. Applied to clinical '
                               'transcripts to identify seizure descriptions, medication '
                               'names, symptom timelines, and treatment responses.',
            },
            {
                'name': 'Voice Biomarker',
                'description': 'Measurable vocal characteristics (pitch, cadence, '
                               'articulation, pause patterns) that correlate with '
                               'neurological states. In epilepsy care, voice biomarkers '
                               'can indicate post-ictal confusion, medication side effects, '
                               'or cognitive decline.',
            },
            {
                'name': 'Diarization',
                'description': 'The process of partitioning an audio stream into segments '
                               'according to speaker identity ("who spoke when"). Essential '
                               'for multi-party clinical conversations to attribute '
                               'statements to clinician vs. patient vs. caregiver.',
            },
            {
                'name': 'Sentiment Analysis',
                'description': 'Detection of emotional tone and affect in speech or text. '
                               'Used to monitor patient mood, anxiety levels around seizure '
                               'events, and clinician-patient rapport over longitudinal visits.',
            },
            {
                'name': 'Speech-to-Text',
                'description': 'End-to-end pipeline that captures audio input, applies '
                               'noise reduction, performs acoustic modeling, and produces '
                               'a text transcript. Includes punctuation restoration and '
                               'medical vocabulary adaptation for clinical accuracy.',
            },
            {
                'name': 'Prosody Analysis',
                'description': 'Examination of speech rhythm, stress, and intonation '
                               'patterns. Prosodic features can reveal neurological '
                               'impairment, fatigue, or emotional distress in epilepsy '
                               'patients, complementing semantic content analysis.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Transcript Accuracy',
                'description': 'Percentage of correctly transcribed words compared to '
                               'ground-truth reference. Clinical ASR targets > 95% accuracy '
                               'for medical terminology.',
            },
            {
                'name': 'Word Error Rate (WER)',
                'description': 'Standard ASR metric: (substitutions + insertions + deletions) '
                               '/ total reference words. Lower is better. Clinical-grade '
                               'systems aim for WER < 5%.',
            },
            {
                'name': 'Speaker Identification Accuracy',
                'description': 'Percentage of speech segments correctly attributed to the '
                               'right speaker during diarization. Critical for distinguishing '
                               'clinician orders from patient self-reports.',
            },
            {
                'name': 'Response Latency',
                'description': 'Time between end of user speech and start of AI response. '
                               'Measured in milliseconds. Real-time clinical systems target '
                               '< 500ms for conversational flow.',
            },
            {
                'name': 'Signal-to-Noise Ratio (SNR)',
                'description': 'Ratio of speech signal power to background noise power in '
                               'decibels. Clinical environments (ICU, EMU) often have low '
                               'SNR, requiring robust noise-cancellation preprocessing.',
            },
        ],
        'clinical_relevance': (
            'Speech AI in epilepsy care enables automated transcription of seizure '
            'semiology descriptions, extraction of medication and dosage entities from '
            'clinical conversations, longitudinal tracking of patient-reported outcomes '
            'via voice, detection of post-ictal speech impairment as a biomarker, and '
            'real-time clinical decision support during patient consultations. Voice '
            'biomarkers derived from prosody and articulation patterns can complement '
            'EEG and imaging data for comprehensive neurological assessment.'
        ),
        'compliance': [
            {
                'ref': 'FDA AI/ML Framework',
                'note': 'Speech AI used for clinical decision support must follow the '
                        'Predetermined Change Control Plan (PCCP) for model updates. '
                        'ASR accuracy and NLP extraction precision must be validated '
                        'against clinical ground truth.',
            },
            {
                'ref': 'EU AI Act Art. 10',
                'note': 'Training data for speech models must be representative of '
                        'clinical populations, including accented speech, pediatric '
                        'voices, and patients with speech impairments.',
            },
            {
                'ref': 'ISO 14971',
                'note': 'Risk management for speech AI must address misrecognition of '
                        'critical medical terms (e.g., drug names, dosages), speaker '
                        'misattribution, and failure-to-transcribe scenarios.',
            },
            {
                'ref': 'IEC 62304',
                'note': 'Speech processing software components must follow medical device '
                        'software lifecycle processes, including unit-level validation of '
                        'ASR, NLP, and diarization modules.',
            },
            {
                'ref': 'HIPAA',
                'note': 'Audio recordings and transcripts contain PHI. Speech AI systems '
                        'must encrypt audio at rest and in transit, enforce access controls, '
                        'and maintain audit trails for all transcript access.',
            },
        ],
        'remediation': [
            {
                'strategy': 'ASR Model Retraining',
                'description': 'When transcript accuracy drops below threshold, retrain '
                               'the ASR model with recent clinical audio samples, focusing '
                               'on medical vocabulary and accented speech patterns.',
            },
            {
                'strategy': 'Noise Environment Calibration',
                'description': 'Periodically recalibrate noise-cancellation filters for '
                               'each clinical environment (EMU, outpatient clinic, ICU) to '
                               'maintain acceptable SNR levels.',
            },
            {
                'strategy': 'Speaker Model Update',
                'description': 'Update diarization speaker embeddings when new clinicians '
                               'join the care team or when patient voice characteristics '
                               'change (e.g., post-surgical recovery).',
            },
            {
                'strategy': 'NLP Entity Dictionary Refresh',
                'description': 'Synchronize the medical entity extraction dictionary with '
                               'current formulary, ICD codes, and epilepsy-specific '
                               'terminology updates to prevent extraction drift.',
            },
        ],
    }
