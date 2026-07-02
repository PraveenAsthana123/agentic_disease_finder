"""Text-to-Audio AI Dashboard — clinical text-to-speech synthesis pipeline
monitoring, audio report generation, patient instruction audio, accessibility
audio for neurological care documentation.

Maps clinical.db tables to text-to-audio AI concepts:
- conversation_log        -> synthesizable clinical text (patient interactions)
- transaction_log         -> TTS pipeline events (genai_bot actions)
- assessments             -> clinical report text for audio conversion
- patients                -> per-patient audio synthesis profiles
- finops_costs            -> TTS model cost tracking
"""

import sqlite3
import os
from collections import Counter
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# TTS-relevant transaction_log components
TTS_COMPONENTS = ['genai_bot', 'patient_chat', 'clinical_report']

# Audio output categories
AUDIO_CATEGORIES = {
    'clinical_report': 'Clinical Report Audio',
    'patient_instruction': 'Patient Instruction',
    'assessment_summary': 'Assessment Summary',
    'discharge_note': 'Discharge Note Audio',
    'medication_guide': 'Medication Guide',
}

# Text length tiers for synthesis complexity
LENGTH_TIERS = [
    ('short', 0, 100, '#10b981'),
    ('medium', 100, 500, '#3b82f6'),
    ('long', 500, 2000, '#f59e0b'),
    ('extended', 2000, 999999, '#ef4444'),
]


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


def _text_length_tier(text_len):
    for label, lo, hi, color in LENGTH_TIERS:
        if lo <= text_len < hi:
            return label
    return 'extended'


def _estimate_audio_duration(char_count):
    """Estimate TTS audio duration in seconds (~150 words/min, ~5 chars/word)."""
    words = char_count / 5.0
    return round(words / 150.0 * 60.0, 1)


def _load_conversations():
    return _db_query(
        "SELECT id, role, text, ts_utc, ts_local "
        "FROM conversation_log ORDER BY ts_utc"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, patient_id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component IN ('genai_bot', 'patient_chat', 'clinical_report') "
        "ORDER BY ts_utc"
    )


def _load_assessments():
    return _db_query(
        "SELECT id, patient_id, instrument, score, max_score, "
        "interpretation, level, examiner, created_at "
        "FROM assessments ORDER BY created_at"
    )


def _load_patients():
    return _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients ORDER BY patient_id"
    )


def _load_finops():
    return _db_query(
        "SELECT id, cost_date, category, sub_category, model_or_service, "
        "component, requests, tokens_in, tokens_out, cost_usd "
        "FROM finops_costs WHERE category = 'inference' "
        "ORDER BY cost_date"
    )


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Text-to-Audio AI dashboard."""
    conversations = _load_conversations()
    pipeline = _load_pipeline_events()
    assessments = _load_assessments()
    patients = _load_patients()
    finops = _load_finops()

    if not conversations and not assessments:
        return {
            'available': False,
            'message': 'No synthesizable text data available. '
                       'Add conversation logs or clinical assessments first.',
        }

    # --- Synthesizable text corpus ---
    # Conversations with non-empty text
    synth_texts = [c for c in conversations if c.get('text') and len(c['text'].strip()) > 0]
    total_texts = len(synth_texts)

    # Character and word counts
    total_chars = sum(len(c['text']) for c in synth_texts)
    total_words = sum(len(c['text'].split()) for c in synth_texts)

    # Estimated total audio duration
    total_est_duration = _estimate_audio_duration(total_chars)
    total_est_minutes = round(total_est_duration / 60.0, 1)

    # Text length distribution
    tier_counts = Counter()
    for c in synth_texts:
        tier = _text_length_tier(len(c['text']))
        tier_counts[tier] += 1

    text_length_distribution = [
        {
            'tier': label,
            'count': tier_counts.get(label, 0),
            'color': color,
        }
        for label, lo, hi, color in LENGTH_TIERS
        if tier_counts.get(label, 0) > 0
    ]

    # Role distribution (who generates text for synthesis)
    role_counts = Counter(c.get('role', 'unknown') for c in synth_texts)
    role_distribution = [
        {'role': role, 'count': cnt}
        for role, cnt in sorted(role_counts.items(), key=lambda x: -x[1])
    ]

    # Assessment reports as synthesizable content
    assessment_texts = [a for a in assessments if a.get('interpretation')]
    total_report_texts = len(assessment_texts)

    # Daily synthesis activity
    daily_counts = Counter()
    for c in synth_texts:
        day = (c.get('ts_utc') or '')[:10]
        if day:
            daily_counts[day] += 1
    daily_activity = [
        {'date': day, 'count': cnt}
        for day, cnt in sorted(daily_counts.items())
    ]

    # Pipeline events
    pipeline_count = len(pipeline)

    # Unique patients with synthesizable content
    patient_ids_pipeline = set(e['patient_id'] for e in pipeline if e.get('patient_id'))
    patients_with_audio = len(patient_ids_pipeline)

    # FinOps — TTS-related costs
    tts_costs = [f for f in finops if 'tts' in (f.get('model_or_service') or '').lower()
                 or 'speech' in (f.get('sub_category') or '').lower()
                 or 'audio' in (f.get('sub_category') or '').lower()]
    total_tts_cost = round(sum(f.get('cost_usd', 0) for f in tts_costs), 2)
    # If no specific TTS costs, use inference costs as proxy
    if not tts_costs:
        total_tts_cost = round(sum(f.get('cost_usd', 0) for f in finops) * 0.05, 2)

    # Mean text length
    mean_text_len = round(_avg([len(c['text']) for c in synth_texts]), 0)

    return {
        'available': True,
        'total_texts': total_texts,
        'total_report_texts': total_report_texts,
        'total_words': total_words,
        'total_chars': total_chars,
        'mean_text_length': int(mean_text_len),
        'est_audio_minutes': total_est_minutes,
        'patients_with_audio': patients_with_audio,
        'pipeline_events': pipeline_count,
        'tts_cost_usd': total_tts_cost,
        'text_length_distribution': text_length_distribution,
        'role_distribution': role_distribution,
        'daily_activity': daily_activity,
        'kpis': [
            {'label': 'Synthesizable Texts', 'value': str(total_texts)},
            {'label': 'Clinical Reports', 'value': str(total_report_texts)},
            {'label': 'Total Words', 'value': f'{total_words:,}'},
            {'label': 'Est. Audio (min)', 'value': str(total_est_minutes),
             'color': '#3b82f6'},
            {'label': 'Mean Text Length', 'value': f'{int(mean_text_len)} chars'},
            {'label': 'Patients Covered', 'value': str(patients_with_audio)},
            {'label': 'Pipeline Events', 'value': str(pipeline_count)},
            {'label': 'TTS Cost (USD)', 'value': f'${total_tts_cost:.2f}',
             'color': '#f59e0b' if total_tts_cost > 50 else '#10b981'},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed text-to-audio breakdown — text inventory, per-patient profiles,
    synthesis queue, pipeline events."""
    conversations = _load_conversations()
    assessments = _load_assessments()
    patients = _load_patients()
    pipeline = _load_pipeline_events()

    synth_texts = [c for c in conversations if c.get('text') and len(c['text'].strip()) > 0]

    if not synth_texts and not assessments:
        return {'available': False}

    patient_map = {p['patient_id']: p for p in patients}

    # --- Text inventory (synthesizable conversation texts) ---
    text_inventory = []
    for c in synth_texts:
        text = c['text']
        char_count = len(text)
        tier = _text_length_tier(char_count)
        est_sec = _estimate_audio_duration(char_count)
        text_inventory.append({
            'id': c['id'],
            'role': c.get('role', 'unknown'),
            'text_preview': text[:120] + ('...' if len(text) > 120 else ''),
            'char_count': char_count,
            'word_count': len(text.split()),
            'length_tier': tier,
            'est_duration_sec': est_sec,
            'ts_utc': c.get('ts_utc'),
        })

    # --- Assessment report texts (clinical reports for audio) ---
    report_inventory = []
    for a in assessments:
        interp = a.get('interpretation') or ''
        if not interp:
            continue
        char_count = len(interp)
        est_sec = _estimate_audio_duration(char_count)
        report_inventory.append({
            'id': a['id'],
            'patient_id': a['patient_id'],
            'instrument': a['instrument'],
            'interpretation': interp[:150] + ('...' if len(interp) > 150 else ''),
            'char_count': char_count,
            'est_duration_sec': est_sec,
            'level': a.get('level'),
            'created_at': a.get('created_at'),
        })

    # --- Patient audio profiles ---
    patient_pipeline = {}
    for e in pipeline:
        pid = e.get('patient_id')
        if pid:
            patient_pipeline.setdefault(pid, []).append(e)

    patient_assessments = {}
    for a in assessments:
        pid = a.get('patient_id')
        if pid:
            patient_assessments.setdefault(pid, []).append(a)

    all_patient_ids = set(patient_pipeline.keys()) | set(patient_assessments.keys())
    patient_profiles = []
    for pid in sorted(all_patient_ids):
        pa = patient_assessments.get(pid, [])
        pe = patient_pipeline.get(pid, [])
        pinfo = patient_map.get(pid, {})

        report_count = sum(1 for a in pa if a.get('interpretation'))
        total_report_chars = sum(len(a.get('interpretation', '')) for a in pa if a.get('interpretation'))
        est_total_audio = _estimate_audio_duration(total_report_chars)

        instruments = sorted(set(a['instrument'] for a in pa))

        patient_profiles.append({
            'patient_id': pid,
            'name': pinfo.get('name', ''),
            'age': pinfo.get('age'),
            'disease': pinfo.get('disease'),
            'n_reports': report_count,
            'n_pipeline_events': len(pe),
            'instruments': instruments,
            'total_report_chars': total_report_chars,
            'est_audio_sec': est_total_audio,
            'actions': sorted(set(e.get('action', '') for e in pe)),
        })

    # --- Pipeline events ---
    pipeline_events = [
        {
            'id': ev['id'],
            'patient_id': ev.get('patient_id'),
            'component': ev.get('component'),
            'action': ev.get('action'),
            'actor': ev.get('actor'),
            'detail': (ev.get('detail') or '')[:120],
            'ts_utc': ev.get('ts_utc'),
        }
        for ev in pipeline
    ]

    # --- Synthesis queue summary ---
    action_counts = Counter(e.get('action', 'unknown') for e in pipeline)
    action_distribution = [
        {'action': act, 'count': cnt}
        for act, cnt in sorted(action_counts.items(), key=lambda x: -x[1])
    ]

    return {
        'available': True,
        'text_inventory': text_inventory[:100],  # cap for performance
        'report_inventory': report_inventory[:100],
        'patient_profiles': patient_profiles,
        'pipeline_events': pipeline_events[:200],
        'action_distribution': action_distribution,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Text-to-Audio AI dashboard."""
    return {
        'concepts': [
            {
                'name': 'Text-to-Speech (TTS)',
                'description': 'AI technology that converts written clinical text into '
                               'natural-sounding speech audio. Modern neural TTS models '
                               '(e.g., VITS, Tacotron 2, FastSpeech 2) produce prosodically '
                               'rich audio with appropriate intonation for medical terminology.',
            },
            {
                'name': 'Neural Vocoder',
                'description': 'Deep learning model that generates raw audio waveforms from '
                               'mel-spectrograms. Examples include WaveNet, WaveGlow, and '
                               'HiFi-GAN. Vocoders determine audio fidelity and naturalness '
                               'of the synthesized clinical audio.',
            },
            {
                'name': 'Mel-Spectrogram',
                'description': 'Time-frequency representation of audio using mel-scaled '
                               'frequency bins that approximate human auditory perception. '
                               'TTS models first predict mel-spectrograms from text, then '
                               'a vocoder converts them to waveforms.',
            },
            {
                'name': 'Prosody Control',
                'description': 'Manipulation of speech rhythm, stress, intonation, and '
                               'pacing in synthesized audio. Critical for clinical audio to '
                               'convey urgency (alerts), emphasis (key findings), and '
                               'appropriate tone (patient-facing instructions).',
            },
            {
                'name': 'Medical Text Normalization',
                'description': 'Pre-processing step that converts medical abbreviations, '
                               'dosage numbers, units, and clinical shorthand into '
                               'pronounceable text. E.g., "100mg BID" → "one hundred '
                               'milligrams twice daily". Essential for accurate TTS output.',
            },
            {
                'name': 'Audio Accessibility',
                'description': 'Provision of clinical information in audio format for '
                               'patients with visual impairments, low literacy, or cognitive '
                               'difficulties. TTS enables equitable access to discharge '
                               'instructions, medication guides, and assessment summaries.',
            },
            {
                'name': 'Speaker Embedding',
                'description': 'Learned vector representation of a speaker\'s vocal '
                               'characteristics, enabling multi-speaker TTS with consistent '
                               'voice identity. Clinical systems may use distinct voices for '
                               'different roles (clinician, automated alert, patient-facing).',
            },
            {
                'name': 'Streaming TTS',
                'description': 'Real-time text-to-speech synthesis that begins audio '
                               'playback before the full text is processed. Reduces latency '
                               'for interactive clinical applications like voice-based '
                               'patient portals and real-time report dictation playback.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Mean Opinion Score (MOS)',
                'description': 'Subjective audio quality rating on a 1–5 scale assessed '
                               'by human listeners. Medical TTS should target MOS ≥ 4.0 '
                               'for patient-facing audio and ≥ 3.5 for internal use.',
            },
            {
                'name': 'Word Error Rate (WER)',
                'description': 'Percentage of words in synthesized audio that are '
                               'misrecognized when fed back through ASR. Lower WER indicates '
                               'clearer pronunciation. Medical terms require WER < 5%.',
            },
            {
                'name': 'Synthesis Latency',
                'description': 'Time from text submission to audio output availability. '
                               'Streaming TTS targets < 200ms to first audio chunk; '
                               'batch synthesis targets < 2x real-time factor.',
            },
            {
                'name': 'Character Coverage',
                'description': 'Percentage of input text characters successfully converted '
                               'to audio without fallback or silence. Target ≥ 99.5% for '
                               'clinical text with medical terminology handling.',
            },
        ],
        'audio_categories': [
            {'category': k, 'label': v}
            for k, v in AUDIO_CATEGORIES.items()
        ],
        'compliance': [
            {
                'ref': 'FDA AI/ML Framework',
                'note': 'TTS systems generating clinical audio content must ensure '
                        'accuracy of medical terminology pronunciation. Mispronunciation '
                        'of drug names or dosages could constitute a safety risk. Systems '
                        'must validate against pharmacological pronunciation databases.',
            },
            {
                'ref': 'EU AI Act Art. 6',
                'note': 'TTS systems in healthcare must disclose AI-generated audio '
                        'content to patients. Audio outputs must be clearly identified '
                        'as machine-generated when used for clinical communication.',
            },
            {
                'ref': 'ISO 14971',
                'note': 'Risk analysis must address TTS failure modes: incorrect '
                        'pronunciation of medication names, missing critical text '
                        'segments, inappropriate prosody for urgent alerts, and '
                        'audio quality degradation in noisy clinical environments.',
            },
            {
                'ref': 'ADA / Section 508',
                'note': 'Audio accessibility outputs must meet WCAG 2.1 AA standards. '
                        'TTS-generated audio must be available in formats compatible '
                        'with assistive technologies and screen readers.',
            },
            {
                'ref': 'HIPAA',
                'note': 'Synthesized audio containing patient health information is PHI. '
                        'Audio files must be encrypted at rest and in transit, access '
                        'must be logged, and audio must be purged per retention policy.',
            },
        ],
        'remediation': [
            {
                'strategy': 'Medical Lexicon Update',
                'description': 'Maintain a curated pronunciation dictionary for medical '
                               'terms, drug names, and clinical abbreviations. Update '
                               'quarterly from pharmacological databases and clinical '
                               'terminology standards (SNOMED CT, RxNorm).',
            },
            {
                'strategy': 'Audio Quality Monitoring',
                'description': 'Run automated MOS estimation on synthesized audio samples. '
                               'Flag outputs below quality threshold for re-synthesis with '
                               'alternative TTS model or manual review.',
            },
            {
                'strategy': 'Synthesis Fallback Pipeline',
                'description': 'When primary TTS model fails or produces low-quality output, '
                               'automatically route to backup model. Maintain at least two '
                               'TTS providers for clinical audio continuity.',
            },
            {
                'strategy': 'Patient Comprehension Validation',
                'description': 'Periodically verify that TTS-generated patient instructions '
                               'are comprehensible via read-back tests or comprehension '
                               'surveys. Adjust speaking rate and vocabulary level based '
                               'on patient population literacy assessments.',
            },
        ],
    }
