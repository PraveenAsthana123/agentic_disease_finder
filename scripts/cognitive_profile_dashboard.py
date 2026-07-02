"""Cognitive Profile Summary Dashboard — neuropsychological assessment,
cognitive domain scoring, impairment detection from clinical evaluations.

Maps clinical.db tables to cognitive profile concepts:
- assessments (MOCA)        -> cognitive screening (MCI threshold ~26)
- assessments (MMSE)        -> cognitive screening (dementia threshold ~24)
- assessments (WAIS)        -> IQ / cognitive ability
- assessments (DIGIT_SPAN)  -> working memory
- assessments (PHQ9)        -> depression comorbidity
- assessments (GAD7)        -> anxiety comorbidity
- assessments (QOLIE31)     -> epilepsy quality of life
- assessments (NDDIE)       -> neurological disorders depression
- patients                  -> per-patient cognitive profiles
- transaction_log           -> pipeline events
"""

import sqlite3
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Cognitive-relevant assessment instruments
COG_INSTRUMENTS = ['MOCA', 'MMSE', 'WAIS', 'DIGIT_SPAN', 'PHQ9', 'GAD7', 'QOLIE31', 'NDDIE']

INSTRUMENT_LABELS = {
    'MOCA': 'Montreal Cognitive Assessment',
    'MMSE': 'Mini-Mental State Examination',
    'WAIS': 'Wechsler Adult Intelligence Scale',
    'DIGIT_SPAN': 'Digit Span (Working Memory)',
    'PHQ9': 'Patient Health Questionnaire-9',
    'GAD7': 'Generalized Anxiety Disorder-7',
    'QOLIE31': 'Quality of Life in Epilepsy-31',
    'NDDIE': 'Neurological Disorders Depression Inventory',
}

INSTRUMENT_DOMAIN = {
    'MOCA': 'Global Cognition',
    'MMSE': 'Global Cognition',
    'WAIS': 'Intelligence / Ability',
    'DIGIT_SPAN': 'Working Memory',
    'PHQ9': 'Mood (Depression)',
    'GAD7': 'Mood (Anxiety)',
    'QOLIE31': 'Epilepsy Quality of Life',
    'NDDIE': 'Mood (Neuro Depression)',
}

# Domain groupings for summary
DOMAIN_ORDER = [
    'Global Cognition',
    'Intelligence / Ability',
    'Working Memory',
    'Mood (Depression)',
    'Mood (Anxiety)',
    'Mood (Neuro Depression)',
    'Epilepsy Quality of Life',
]

SEVERITY_COLORS = {
    'normal': '#10b981',
    'mild': '#f59e0b',
    'moderate': '#f97316',
    'severe': '#ef4444',
}

# WAIS uses IQ-norm labels — map to severity tiers for unified reporting
WAIS_LEVEL_MAP = {
    'Extremely High': 'normal',
    'High Average': 'normal',
    'Average': 'normal',
    'Low Average': 'mild',
    'Borderline': 'moderate',
    'Extremely Low': 'severe',
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


def _normalize_level(level, instrument=''):
    """Normalize WAIS IQ-norm labels to standard severity tiers."""
    if not level:
        return 'unknown'
    if instrument == 'WAIS' and level in WAIS_LEVEL_MAP:
        return WAIS_LEVEL_MAP[level]
    return level


def _load_cognitive_assessments():
    placeholders = ','.join('?' for _ in COG_INSTRUMENTS)
    return _db_query(
        f"SELECT id, patient_id, instrument, score, max_score, "
        f"interpretation, level, alert, examiner, created_at "
        f"FROM assessments WHERE instrument IN ({placeholders}) "
        f"ORDER BY created_at",
        COG_INSTRUMENTS
    )


def _load_patients():
    return _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients ORDER BY patient_id"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component IN ('assessment', 'neuropsych', 'genai_bot') "
        "ORDER BY ts_utc"
    )


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Cognitive Profile dashboard."""
    assessments = _load_cognitive_assessments()
    patients = _load_patients()
    pipeline = _load_pipeline_events()

    if not assessments:
        return {
            'available': False,
            'message': 'No cognitive assessment data available. '
                       'Run MOCA, MMSE, WAIS, DIGIT_SPAN, or mood instruments first.',
        }

    total_assessments = len(assessments)

    # Instrument counts
    instrument_counts = Counter(a['instrument'] for a in assessments)

    # Unique patients assessed
    patient_ids = set(a['patient_id'] for a in assessments if a.get('patient_id'))
    patients_assessed = len(patient_ids)

    # Severity distribution (normalize WAIS labels)
    level_counts = Counter()
    for a in assessments:
        nlvl = _normalize_level(a.get('level', 'unknown'), a.get('instrument', ''))
        level_counts[nlvl] += 1

    # Mean normalized score (score / max_score)
    norm_scores = []
    for a in assessments:
        if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
            norm_scores.append(a['score'] / a['max_score'])
    mean_norm_score = _avg(norm_scores)

    # Cognitive impairment flags: MOCA < 26 or MMSE < 24
    impairment_flags = 0
    for a in assessments:
        if a.get('score') is not None:
            if a['instrument'] == 'MOCA' and a['score'] < 26:
                impairment_flags += 1
            elif a['instrument'] == 'MMSE' and a['score'] < 24:
                impairment_flags += 1

    # Alerts count
    alerts = [a for a in assessments if a.get('alert')]
    alert_count = len(alerts)

    # Abnormal rate (non-normal level)
    abnormal = 0
    for a in assessments:
        nlvl = _normalize_level(a.get('level'), a.get('instrument', ''))
        if nlvl and nlvl not in ('normal', 'unknown'):
            abnormal += 1
    abnormal_rate = round(abnormal / total_assessments, 4) if total_assessments else 0

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
        for inst in COG_INSTRUMENTS
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

    # Domain summary (grouped)
    domain_scores = {}
    for a in assessments:
        domain = INSTRUMENT_DOMAIN.get(a['instrument'], 'Other')
        if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
            domain_scores.setdefault(domain, []).append(a['score'] / a['max_score'])
    domain_summary = [
        {
            'domain': d,
            'mean_score': _avg(domain_scores.get(d, [])),
            'n': len(domain_scores.get(d, [])),
        }
        for d in DOMAIN_ORDER
        if domain_scores.get(d)
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
        'impairment_flags': impairment_flags,
        'abnormal_rate': abnormal_rate,
        'alert_count': alert_count,
        'pipeline_events': pipeline_count,
        'instrument_distribution': instrument_distribution,
        'severity_distribution': severity_distribution,
        'instrument_scores': instrument_scores,
        'domain_summary': domain_summary,
        'daily_activity': daily_activity,
        'kpis': [
            {'label': 'Cognitive Assessments', 'value': str(total_assessments)},
            {'label': 'Instruments Used', 'value': str(len(instrument_counts))},
            {'label': 'Patients Profiled', 'value': str(patients_assessed)},
            {'label': 'Mean Score', 'value': f'{mean_norm_score:.1%}',
             'color': '#10b981' if mean_norm_score >= 0.8 else '#f59e0b' if mean_norm_score >= 0.6 else '#ef4444'},
            {'label': 'Impairment Flags', 'value': str(impairment_flags),
             'color': '#ef4444' if impairment_flags > 5 else '#f59e0b' if impairment_flags > 0 else '#10b981'},
            {'label': 'Abnormal Rate', 'value': f'{abnormal_rate:.1%}',
             'color': '#ef4444' if abnormal_rate > 0.3 else '#f59e0b' if abnormal_rate > 0.15 else '#10b981'},
            {'label': 'Clinical Alerts', 'value': str(alert_count),
             'color': '#ef4444' if alert_count > 5 else '#f59e0b' if alert_count > 0 else '#10b981'},
            {'label': 'Pipeline Events', 'value': str(pipeline_count)},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed cognitive profile breakdown — assessment inventory, per-patient
    cognitive profiles, domain scores, instrument stats, clinical alerts."""
    assessments = _load_cognitive_assessments()
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
        nlvl = _normalize_level(a.get('level'), a.get('instrument', ''))
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
            'normalized_level': nlvl,
            'alert': a.get('alert'),
            'examiner': a.get('examiner'),
            'created_at': a.get('created_at'),
        })

    # --- Patient cognitive profiles ---
    patient_assessments = {}
    for a in assessments:
        pid = a['patient_id']
        patient_assessments.setdefault(pid, []).append(a)

    patient_profiles = []
    for pid in sorted(patient_assessments.keys()):
        pa = patient_assessments[pid]
        instruments = sorted(set(a['instrument'] for a in pa))
        domains = sorted(set(INSTRUMENT_DOMAIN.get(a['instrument'], '') for a in pa))
        norm_scores = []
        levels = []
        alerts = []
        impaired = False
        for a in pa:
            if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
                norm_scores.append(a['score'] / a['max_score'])
            nlvl = _normalize_level(a.get('level'), a.get('instrument', ''))
            if nlvl and nlvl != 'unknown':
                levels.append(nlvl)
            if a.get('alert'):
                alerts.append(a['alert'])
            # Check MOCA/MMSE impairment
            if a.get('score') is not None:
                if a['instrument'] == 'MOCA' and a['score'] < 26:
                    impaired = True
                elif a['instrument'] == 'MMSE' and a['score'] < 24:
                    impaired = True

        worst_level = 'normal'
        for sev in ['severe', 'moderate', 'mild']:
            if sev in levels:
                worst_level = sev
                break

        # Per-domain scores for this patient
        domain_profile = {}
        for a in pa:
            domain = INSTRUMENT_DOMAIN.get(a['instrument'], 'Other')
            if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
                domain_profile.setdefault(domain, []).append(a['score'] / a['max_score'])
        domain_scores = [
            {'domain': d, 'mean_score': _avg(scores)}
            for d, scores in sorted(domain_profile.items())
        ]

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
            'domains': domains,
            'domain_scores': domain_scores,
            'mean_score': _avg(norm_scores),
            'worst_level': worst_level,
            'impaired': impaired,
            'alert_count': len(alerts),
        })

    # --- Instrument stats ---
    inst_groups = {}
    for a in assessments:
        inst = a['instrument']
        inst_groups.setdefault(inst, {'scores': [], 'raw_scores': [], 'levels': [], 'alerts': 0})
        if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
            inst_groups[inst]['scores'].append(a['score'] / a['max_score'])
            inst_groups[inst]['raw_scores'].append(a['score'])
        nlvl = _normalize_level(a.get('level'), inst)
        if nlvl and nlvl != 'unknown':
            inst_groups[inst]['levels'].append(nlvl)
        if a.get('alert'):
            inst_groups[inst]['alerts'] += 1

    instrument_stats = []
    for inst in COG_INSTRUMENTS:
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
            'mean_raw': _avg(g['raw_scores']),
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
            'domain': INSTRUMENT_DOMAIN.get(a['instrument'], ''),
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
    """Definitions tab for the Cognitive Profile dashboard."""
    return {
        'concepts': [
            {
                'name': 'Montreal Cognitive Assessment (MoCA)',
                'description': 'Brief 30-point screening instrument for mild cognitive '
                               'impairment (MCI). Assesses visuospatial/executive function, '
                               'naming, memory, attention, language, abstraction, delayed '
                               'recall, and orientation. Scores <26 suggest MCI; sensitivity '
                               '90% for MCI vs 18% for MMSE.',
            },
            {
                'name': 'Mini-Mental State Examination (MMSE)',
                'description': 'Widely used 30-point cognitive screening tool measuring '
                               'orientation, registration, attention/calculation, recall, '
                               'language, and visuospatial ability. Scores <24 suggest '
                               'dementia; 24-26 suggest MCI. Less sensitive to early '
                               'cognitive decline than MoCA.',
            },
            {
                'name': 'Wechsler Adult Intelligence Scale (WAIS)',
                'description': 'Gold-standard IQ assessment yielding Full Scale IQ (FSIQ) '
                               'from four indices: Verbal Comprehension, Perceptual Reasoning, '
                               'Working Memory, and Processing Speed. Mean 100, SD 15. '
                               'Critical for pre-surgical neuropsychological evaluation in '
                               'epilepsy patients.',
            },
            {
                'name': 'Digit Span',
                'description': 'Working memory subtest with forward (immediate recall) and '
                               'backward (manipulation) components. Assesses phonological '
                               'loop capacity and central executive function. Sensitive to '
                               'anti-epileptic drug effects on cognitive processing speed.',
            },
            {
                'name': 'PHQ-9 (Patient Health Questionnaire-9)',
                'description': 'Nine-item depression screening tool scoring 0-27. Thresholds: '
                               '5 mild, 10 moderate, 15 moderately severe, 20 severe depression. '
                               'Depression is the most common psychiatric comorbidity in epilepsy '
                               '(prevalence 20-55%).',
            },
            {
                'name': 'GAD-7 (Generalized Anxiety Disorder-7)',
                'description': 'Seven-item anxiety screening scale scoring 0-21. Thresholds: '
                               '5 mild, 10 moderate, 15 severe anxiety. Anxiety disorders '
                               'affect 10-25% of epilepsy patients and are associated with '
                               'worse seizure control.',
            },
            {
                'name': 'QOLIE-31 (Quality of Life in Epilepsy-31)',
                'description': 'Epilepsy-specific quality of life instrument with 31 items '
                               'across 7 subscales: seizure worry, overall QoL, emotional '
                               'well-being, energy/fatigue, cognitive function, medication '
                               'effects, and social function. Higher scores = better QoL.',
            },
            {
                'name': 'Cognitive Domain Profiling',
                'description': 'Systematic mapping of assessment scores to cognitive domains '
                               '(global cognition, working memory, intelligence, mood) to '
                               'create patient-specific cognitive profiles. Enables detection '
                               'of domain-specific impairments and longitudinal tracking.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Assessment Score (Normalized)',
                'description': 'Score expressed as percentage of maximum possible score. '
                               'Enables cross-instrument comparison. Thresholds vary by '
                               'instrument (MOCA <87% = MCI, MMSE <80% = dementia).',
            },
            {
                'name': 'Cognitive Impairment Flag Rate',
                'description': 'Percentage of MOCA/MMSE assessments falling below clinical '
                               'impairment thresholds (MOCA <26, MMSE <24). Higher rates '
                               'indicate population-level cognitive burden.',
            },
            {
                'name': 'Domain Coverage',
                'description': 'Number of distinct cognitive domains assessed per patient. '
                               'Complete neuropsychological profiles require coverage of '
                               'global cognition, working memory, mood, and quality of life.',
            },
            {
                'name': 'Longitudinal Consistency',
                'description': 'Stability of assessment scores across time points for the '
                               'same patient. Significant decline (>2 SD change) triggers '
                               'clinical review for progressive cognitive impairment.',
            },
        ],
        'assessment_instruments': [
            {
                'instrument': inst,
                'label': INSTRUMENT_LABELS.get(inst, inst),
                'domain': INSTRUMENT_DOMAIN.get(inst, ''),
            }
            for inst in COG_INSTRUMENTS
        ],
        'compliance': [
            {
                'ref': 'FDA AI/ML Framework',
                'note': 'AI-derived cognitive impairment flags used for clinical decision '
                        'support must follow the Predetermined Change Control Plan (PCCP). '
                        'Score normalization and impairment thresholds must be validated '
                        'against expert neuropsychological assessment.',
            },
            {
                'ref': 'EU AI Act Art. 6',
                'note': 'Cognitive profiling systems used in medical diagnosis are classified '
                        'as high-risk. Require conformity assessment, human oversight, and '
                        'transparency obligations for AI-assisted cognitive screening.',
            },
            {
                'ref': 'ISO 14971',
                'note': 'Risk management must address misclassification of cognitive '
                        'impairment severity, false-negative MCI screening, and failure '
                        'to detect progressive cognitive decline in epilepsy patients.',
            },
            {
                'ref': 'APA Ethical Standards',
                'note': 'Neuropsychological assessments must be administered and interpreted '
                        'by qualified professionals. AI-augmented scoring must maintain '
                        'clinician oversight and informed consent for automated profiling.',
            },
            {
                'ref': 'HIPAA',
                'note': 'Cognitive assessment scores and neuropsychological profiles contain '
                        'PHI. Systems must encrypt data at rest and in transit, enforce '
                        'role-based access, and maintain audit trails.',
            },
        ],
        'remediation': [
            {
                'strategy': 'Impairment Threshold Calibration',
                'description': 'Recalibrate MOCA/MMSE impairment cut-offs using site-specific '
                               'normative data adjusted for age, education, and language. '
                               'Reduces false-positive rates in diverse populations.',
            },
            {
                'strategy': 'Domain Gap Analysis',
                'description': 'Identify patients with incomplete cognitive domain coverage '
                               'and schedule targeted assessments. Prioritize WAIS for '
                               'pre-surgical candidates lacking IQ baseline.',
            },
            {
                'strategy': 'Longitudinal Tracking Protocol',
                'description': 'Implement automated alerts when sequential assessments show '
                               'clinically significant decline (>1.5 SD). Triggers '
                               'neuropsychologist review and medication evaluation.',
            },
            {
                'strategy': 'Mood-Cognition Interaction Review',
                'description': 'Cross-reference PHQ9/GAD7 elevations with MOCA/MMSE declines. '
                               'Untreated depression/anxiety can mimic or exacerbate cognitive '
                               'impairment — requires differentiation before diagnosis.',
            },
        ],
    }
