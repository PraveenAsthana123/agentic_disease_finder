"""Neuropsychologist Dashboard — cognitive assessment battery, memory profiles,
executive function, attention, processing speed, and IQ analysis from real
clinical evaluations.

Maps clinical.db tables to neuropsychological concepts:
- assessments (WAIS)       -> Full-scale IQ, index scores, subtest profiles
- assessments (DIGIT_SPAN) -> Working memory, forward/backward/sequencing spans
- assessments (MOCA)       -> Cognitive screening (visuospatial, naming, attention, etc.)
- assessments (MMSE)       -> Mini mental state screening
- assessments (PHQ9)       -> Depression comorbidity (cognitive impact)
- assessments (GAD7)       -> Anxiety comorbidity (cognitive impact)
- assessments (NDDIE)      -> Epilepsy-specific depression screen
- neuropsych               -> Legacy neuropsych records
- patients                 -> Demographics, disease, lateralization
- seizure_diary            -> Seizure burden (cognitive correlate)
- medications              -> AED cognitive side-effect profiling
"""

import json
import sqlite3
import os
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# WAIS-IV Index → subtest mapping
WAIS_INDICES = {
    'VCI': {'label': 'Verbal Comprehension', 'subtests': ['similarities', 'vocabulary', 'information']},
    'PRI': {'label': 'Perceptual Reasoning', 'subtests': ['block_design', 'matrix_reasoning', 'visual_puzzles']},
    'WMI': {'label': 'Working Memory', 'subtests': ['digit_span', 'arithmetic']},
    'PSI': {'label': 'Processing Speed', 'subtests': ['symbol_search', 'coding']},
}

# Cognitive level classification
LEVEL_ORDER = ['Extremely Low', 'Very Low', 'Low', 'Low Average', 'Average',
               'High Average', 'Superior', 'Very Superior']

# AEDs known to affect cognition (evidence-based)
AED_COGNITIVE_EFFECTS = {
    'Topiramate':     {'severity': 'high',   'effects': ['word-finding difficulty', 'psychomotor slowing', 'memory impairment']},
    'Zonisamide':     {'severity': 'moderate', 'effects': ['cognitive slowing', 'word-finding difficulty']},
    'Phenobarbital':  {'severity': 'high',   'effects': ['sedation', 'attention impairment', 'memory dysfunction']},
    'Phenytoin':      {'severity': 'moderate', 'effects': ['cerebellar dysfunction', 'attention impairment']},
    'Carbamazepine':  {'severity': 'moderate', 'effects': ['psychomotor slowing', 'attention effects']},
    'Valproate':      {'severity': 'low',    'effects': ['tremor', 'mild cognitive effects at high doses']},
    'Levetiracetam':  {'severity': 'low',    'effects': ['irritability (indirect cognitive impact)']},
    'Lamotrigine':    {'severity': 'minimal', 'effects': ['generally cognitive-neutral or mildly beneficial']},
    'Lacosamide':     {'severity': 'low',    'effects': ['dizziness', 'mild attention effects']},
    'Oxcarbazepine':  {'severity': 'low',    'effects': ['mild psychomotor slowing']},
    'Brivaracetam':   {'severity': 'minimal', 'effects': ['generally well-tolerated cognitively']},
    'Perampanel':     {'severity': 'moderate', 'effects': ['dizziness', 'irritability', 'attention impairment']},
    'Clobazam':       {'severity': 'moderate', 'effects': ['sedation', 'attention effects']},
    'Gabapentin':     {'severity': 'low',    'effects': ['sedation at higher doses']},
    'Pregabalin':     {'severity': 'low',    'effects': ['sedation', 'mild attention effects']},
}


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _avg(values):
    return round(sum(values) / len(values), 1) if values else None


def _classify_iq(score):
    if score is None:
        return 'Unknown'
    if score >= 130:
        return 'Very Superior'
    if score >= 120:
        return 'Superior'
    if score >= 110:
        return 'High Average'
    if score >= 90:
        return 'Average'
    if score >= 80:
        return 'Low Average'
    if score >= 70:
        return 'Low'
    if score >= 55:
        return 'Very Low'
    return 'Extremely Low'


def _load_assessments(instrument):
    return _db_query(
        "SELECT patient_id, score, max_score, interpretation, level, "
        "answers_json, examiner, created_at FROM assessments "
        "WHERE instrument = ? ORDER BY created_at DESC", (instrument,)
    )


def _load_patients():
    return _db_query("SELECT id, age, gender, disease FROM patients")


def _load_medications():
    return _db_query("SELECT patient_id, drug, dose_mg, frequency FROM medications")


def _load_seizure_counts():
    rows = _db_query(
        "SELECT patient_id, COUNT(*) as cnt FROM seizure_diary GROUP BY patient_id"
    )
    return {r['patient_id']: r['cnt'] for r in rows}


def _load_daily_activity():
    rows = _db_query(
        "SELECT DATE(timestamp) as day, COUNT(*) as cnt "
        "FROM transaction_log GROUP BY DATE(timestamp) ORDER BY day DESC LIMIT 30"
    )
    rows.reverse()
    return [{'date': r['day'], 'events': r['cnt']} for r in rows]


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Neuropsychologist dashboard."""
    wais = _load_assessments('WAIS')
    digit = _load_assessments('DIGIT_SPAN')
    moca = _load_assessments('MOCA')
    mmse = _load_assessments('MMSE')
    phq9 = _load_assessments('PHQ9')
    gad7 = _load_assessments('GAD7')
    patients = _load_patients()
    meds = _load_medications()
    seizure_counts = _load_seizure_counts()
    daily = _load_daily_activity()

    all_neuropsych = wais + digit + moca + mmse
    if not all_neuropsych:
        return {
            'available': False,
            'message': 'No neuropsychological assessment data available. '
                       'Import WAIS, MOCA, MMSE, or Digit Span assessments first.',
        }

    neuropsych_patient_ids = set()
    for a in all_neuropsych:
        if a.get('patient_id'):
            neuropsych_patient_ids.add(a['patient_id'])

    total_patients_assessed = len(neuropsych_patient_ids)
    total_assessments = len(wais) + len(digit) + len(moca) + len(mmse)

    # WAIS IQ stats
    iq_scores = [a['score'] for a in wais if a.get('score') is not None]
    avg_fsiq = _avg(iq_scores)
    iq_below_avg = sum(1 for s in iq_scores if s < 90)

    # MOCA stats
    moca_scores = [a['score'] for a in moca if a.get('score') is not None]
    avg_moca = _avg(moca_scores)
    moca_impaired = sum(1 for s in moca_scores if s < 26)

    # MMSE stats
    mmse_scores = [a['score'] for a in mmse if a.get('score') is not None]
    avg_mmse = _avg(mmse_scores)

    # Digit span stats
    digit_scores = [a['score'] for a in digit if a.get('score') is not None]
    avg_digit = _avg(digit_scores)

    # Depression/anxiety comorbidity
    phq9_elevated = sum(1 for a in phq9 if a.get('score') is not None and a['score'] >= 10)
    gad7_elevated = sum(1 for a in gad7 if a.get('score') is not None and a['score'] >= 10)

    # --- IQ distribution ---
    iq_distribution = []
    iq_level_counts = Counter(_classify_iq(s) for s in iq_scores)
    for level in LEVEL_ORDER:
        cnt = iq_level_counts.get(level, 0)
        if cnt > 0:
            iq_distribution.append({'level': level, 'count': cnt})

    # --- WAIS index profile (aggregate) ---
    index_profile = []
    for idx_key, idx_info in WAIS_INDICES.items():
        subtest_means = []
        for a in wais:
            answers = _safe_json(a.get('answers_json'))
            vals = [answers.get(st) for st in idx_info['subtests'] if answers.get(st) is not None]
            if vals:
                subtest_means.append(sum(vals) / len(vals))
        index_profile.append({
            'index': idx_key,
            'label': idx_info['label'],
            'mean_score': _avg(subtest_means),
            'n': len(subtest_means),
        })

    # --- MOCA score distribution ---
    moca_distribution = []
    moca_bins = {'Normal (≥26)': 0, 'Mild Impairment (18-25)': 0, 'Moderate Impairment (<18)': 0}
    for s in moca_scores:
        if s >= 26:
            moca_bins['Normal (≥26)'] += 1
        elif s >= 18:
            moca_bins['Mild Impairment (18-25)'] += 1
        else:
            moca_bins['Moderate Impairment (<18)'] += 1
    for label, cnt in moca_bins.items():
        if cnt > 0:
            moca_distribution.append({'category': label, 'count': cnt})

    # --- Digit span sub-scores ---
    forward_scores = []
    backward_scores = []
    sequencing_scores = []
    for a in digit:
        answers = _safe_json(a.get('answers_json'))
        if answers.get('forward_span') is not None:
            forward_scores.append(answers['forward_span'])
        if answers.get('backward_span') is not None:
            backward_scores.append(answers['backward_span'])
        if answers.get('sequencing_span') is not None:
            sequencing_scores.append(answers['sequencing_span'])

    digit_span_profile = [
        {'type': 'Forward', 'mean_span': _avg(forward_scores), 'n': len(forward_scores)},
        {'type': 'Backward', 'mean_span': _avg(backward_scores), 'n': len(backward_scores)},
        {'type': 'Sequencing', 'mean_span': _avg(sequencing_scores), 'n': len(sequencing_scores)},
    ]

    # --- AED cognitive impact ---
    patient_meds = defaultdict(list)
    for m in meds:
        if m.get('patient_id') and m.get('drug'):
            patient_meds[m['patient_id']].append(m['drug'])

    aed_cog_counts = Counter()
    for pid in neuropsych_patient_ids:
        for drug in patient_meds.get(pid, []):
            normalized = drug.strip().title()
            if normalized in AED_COGNITIVE_EFFECTS:
                info = AED_COGNITIVE_EFFECTS[normalized]
                aed_cog_counts[normalized] += 1

    aed_cognitive_risk = []
    for drug, count in aed_cog_counts.most_common(10):
        info = AED_COGNITIVE_EFFECTS[drug]
        aed_cognitive_risk.append({
            'drug': drug,
            'severity': info['severity'],
            'effects': info['effects'],
            'patients_on_drug': count,
        })

    # --- Mood comorbidity summary ---
    mood_summary = {
        'phq9_assessed': len(phq9),
        'phq9_elevated': phq9_elevated,
        'gad7_assessed': len(gad7),
        'gad7_elevated': gad7_elevated,
    }

    return {
        'available': True,
        'total_patients_assessed': total_patients_assessed,
        'total_assessments': total_assessments,
        'wais_count': len(wais),
        'moca_count': len(moca),
        'mmse_count': len(mmse),
        'digit_span_count': len(digit),
        'avg_fsiq': avg_fsiq,
        'iq_below_average': iq_below_avg,
        'avg_moca': avg_moca,
        'moca_impaired': moca_impaired,
        'avg_mmse': avg_mmse,
        'avg_digit_span': avg_digit,
        'iq_distribution': iq_distribution,
        'index_profile': index_profile,
        'moca_distribution': moca_distribution,
        'digit_span_profile': digit_span_profile,
        'aed_cognitive_risk': aed_cognitive_risk,
        'mood_comorbidity': mood_summary,
        'daily_activity': daily,
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed per-patient neuropsychological profiles."""
    wais = _load_assessments('WAIS')
    digit = _load_assessments('DIGIT_SPAN')
    moca = _load_assessments('MOCA')
    mmse = _load_assessments('MMSE')
    phq9 = _load_assessments('PHQ9')
    gad7 = _load_assessments('GAD7')
    patients_raw = _load_patients()
    meds = _load_medications()
    seizure_counts = _load_seizure_counts()

    patient_map = {p['id']: p for p in patients_raw}
    patient_meds = defaultdict(list)
    for m in meds:
        if m.get('patient_id') and m.get('drug'):
            patient_meds[m['patient_id']].append({
                'drug': m['drug'],
                'dose_mg': m.get('dose_mg'),
                'frequency': m.get('frequency'),
            })

    # Group all assessments by patient
    all_assessments = (
        [{'instrument': 'WAIS', **a} for a in wais] +
        [{'instrument': 'DIGIT_SPAN', **a} for a in digit] +
        [{'instrument': 'MOCA', **a} for a in moca] +
        [{'instrument': 'MMSE', **a} for a in mmse] +
        [{'instrument': 'PHQ9', **a} for a in phq9] +
        [{'instrument': 'GAD7', **a} for a in gad7]
    )
    by_patient = defaultdict(list)
    for a in all_assessments:
        if a.get('patient_id'):
            by_patient[a['patient_id']].append(a)

    # WAIS subtest inventory (per-patient latest)
    wais_profiles = []
    for a in wais:
        answers = _safe_json(a.get('answers_json'))
        profile = {
            'patient_id': a['patient_id'],
            'fsiq': a.get('score'),
            'level': a.get('level', _classify_iq(a.get('score'))),
            'interpretation': a.get('interpretation'),
            'date': a.get('created_at'),
        }
        for idx_key, idx_info in WAIS_INDICES.items():
            vals = [answers.get(st) for st in idx_info['subtests'] if answers.get(st) is not None]
            profile[idx_key] = _avg(vals) if vals else None
        wais_profiles.append(profile)

    # Digit span detail
    digit_profiles = []
    for a in digit:
        answers = _safe_json(a.get('answers_json'))
        digit_profiles.append({
            'patient_id': a['patient_id'],
            'scaled_score': a.get('score'),
            'level': a.get('level'),
            'forward_span': answers.get('forward_span'),
            'backward_span': answers.get('backward_span'),
            'sequencing_span': answers.get('sequencing_span'),
            'forward_raw': answers.get('forward_raw'),
            'backward_raw': answers.get('backward_raw'),
            'total_raw': answers.get('total_raw'),
            'date': a.get('created_at'),
        })

    # Cognitive screening (MOCA + MMSE)
    screening = []
    for a in moca + mmse:
        screening.append({
            'patient_id': a['patient_id'],
            'instrument': 'MOCA' if a in moca else 'MMSE',
            'score': a.get('score'),
            'max_score': a.get('max_score'),
            'level': a.get('level'),
            'interpretation': a.get('interpretation'),
            'date': a.get('created_at'),
        })

    # Patient cognitive profiles (summary per patient)
    patient_profiles = []
    for pid in sorted(by_patient.keys()):
        pinfo = patient_map.get(pid, {})
        p_assessments = by_patient[pid]
        instruments = list(set(a['instrument'] for a in p_assessments))

        latest_wais = next((a for a in p_assessments if a['instrument'] == 'WAIS'), None)
        latest_moca = next((a for a in p_assessments if a['instrument'] == 'MOCA'), None)
        latest_phq9 = next((a for a in p_assessments if a['instrument'] == 'PHQ9'), None)
        latest_gad7 = next((a for a in p_assessments if a['instrument'] == 'GAD7'), None)

        # Cognitive risk factors
        risk_factors = []
        if latest_wais and latest_wais.get('score') is not None and latest_wais['score'] < 85:
            risk_factors.append('Below-average IQ')
        if latest_moca and latest_moca.get('score') is not None and latest_moca['score'] < 26:
            risk_factors.append('MoCA impaired')
        if latest_phq9 and latest_phq9.get('score') is not None and latest_phq9['score'] >= 10:
            risk_factors.append('Depression (PHQ-9 elevated)')
        if latest_gad7 and latest_gad7.get('score') is not None and latest_gad7['score'] >= 10:
            risk_factors.append('Anxiety (GAD-7 elevated)')

        # Check for cognitively risky AEDs
        cog_risk_drugs = []
        for med in patient_meds.get(pid, []):
            normalized = med['drug'].strip().title()
            if normalized in AED_COGNITIVE_EFFECTS:
                info = AED_COGNITIVE_EFFECTS[normalized]
                if info['severity'] in ('high', 'moderate'):
                    cog_risk_drugs.append(normalized)
        if cog_risk_drugs:
            risk_factors.append(f"Cognitively risky AED(s): {', '.join(cog_risk_drugs)}")

        seizure_count = seizure_counts.get(pid, 0)
        if seizure_count > 5:
            risk_factors.append(f'High seizure burden ({seizure_count} events)')

        patient_profiles.append({
            'patient_id': pid,
            'age': pinfo.get('age'),
            'gender': pinfo.get('gender'),
            'disease': pinfo.get('disease'),
            'instruments_completed': instruments,
            'assessment_count': len(p_assessments),
            'fsiq': latest_wais['score'] if latest_wais else None,
            'iq_level': latest_wais.get('level') if latest_wais else None,
            'moca_score': latest_moca['score'] if latest_moca else None,
            'phq9_score': latest_phq9['score'] if latest_phq9 else None,
            'gad7_score': latest_gad7['score'] if latest_gad7 else None,
            'seizure_count': seizure_count,
            'risk_factors': risk_factors,
            'medications': [m['drug'] for m in patient_meds.get(pid, [])],
        })

    return {
        'wais_profiles': wais_profiles,
        'digit_span_profiles': digit_profiles,
        'cognitive_screening': screening,
        'patient_profiles': patient_profiles,
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Neuropsychologist dashboard."""
    return {
        'concepts': [
            {
                'name': 'WAIS-IV (Wechsler Adult Intelligence Scale)',
                'description': 'Gold-standard measure of adult intellectual functioning. '
                               'Yields a Full-Scale IQ (FSIQ) and four index scores: '
                               'Verbal Comprehension (VCI), Perceptual Reasoning (PRI), '
                               'Working Memory (WMI), and Processing Speed (PSI). '
                               'Mean = 100, SD = 15. Scores below 85 indicate below-average '
                               'functioning; below 70 indicates intellectual disability.',
            },
            {
                'name': 'MoCA (Montreal Cognitive Assessment)',
                'description': 'Brief cognitive screening tool (30 points) assessing '
                               'visuospatial/executive, naming, memory, attention, language, '
                               'abstraction, delayed recall, and orientation. Score ≥ 26 = normal; '
                               '18-25 = mild cognitive impairment; < 18 = moderate impairment. '
                               'More sensitive than MMSE for detecting mild cognitive impairment.',
            },
            {
                'name': 'MMSE (Mini-Mental State Examination)',
                'description': 'Classic 30-point screening for orientation, registration, '
                               'attention/calculation, recall, and language. Score ≥ 24 = normal; '
                               '18-23 = mild impairment; 10-17 = moderate; < 10 = severe. '
                               'Less sensitive than MoCA for executive dysfunction and MCI.',
            },
            {
                'name': 'Digit Span (WAIS-IV subtest)',
                'description': 'Measures working memory and attention. Three conditions: '
                               'Forward (auditory attention), Backward (working memory manipulation), '
                               'and Sequencing (mental flexibility). Scaled scores: mean = 10, SD = 3. '
                               'Scores ≤ 7 suggest working memory weakness; ≤ 4 = impaired.',
            },
            {
                'name': 'Executive Function',
                'description': 'Higher-order cognitive processes including planning, cognitive '
                               'flexibility, inhibition, working memory, and abstract reasoning. '
                               'Mediated primarily by prefrontal cortex. Often impaired in frontal '
                               'lobe epilepsy. Assessed by WAIS PRI, Digit Span Sequencing, '
                               'and MoCA visuospatial/executive domains.',
            },
            {
                'name': 'Processing Speed',
                'description': 'Rate at which cognitive operations are performed. '
                               'Measured by WAIS-IV PSI (Symbol Search + Coding subtests). '
                               'Commonly reduced in epilepsy due to seizure burden, AED effects '
                               '(especially topiramate, phenobarbital), and subclinical discharges.',
            },
            {
                'name': 'AED Cognitive Side Effects',
                'description': 'Anti-epileptic drugs vary in cognitive impact. Topiramate and '
                               'phenobarbital have the highest cognitive burden (word-finding, '
                               'psychomotor slowing). Lamotrigine and brivaracetam are relatively '
                               'cognitive-neutral. Neuropsychological monitoring is recommended '
                               'when initiating or titrating high-risk AEDs.',
            },
            {
                'name': 'Cognitive Comorbidity in Epilepsy',
                'description': 'Up to 50% of epilepsy patients have cognitive impairment. '
                               'Contributing factors: seizure frequency/severity, epileptiform '
                               'discharges, AED side effects, underlying etiology, depression, '
                               'and anxiety. Pre-surgical neuropsychological evaluation predicts '
                               'post-operative cognitive outcome and guides surgical planning.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Battery Completion Rate',
                'description': 'Percentage of referred patients who complete the full '
                               'neuropsychological battery (WAIS + Digit Span + MoCA/MMSE + mood). '
                               'Target: ≥ 90%. Incomplete batteries limit interpretation.',
            },
            {
                'name': 'Test-Retest Reliability',
                'description': 'Consistency of scores on repeated administration. WAIS FSIQ '
                               'r = .96; MoCA r = .92; Digit Span r = .83. Practice effects '
                               'should be considered for serial assessments (use alternate forms).',
            },
            {
                'name': 'Clinically Significant Change',
                'description': 'Reliable Change Index (RCI) thresholds for detecting true change '
                               'vs. measurement error. FSIQ: ± 8 points; MoCA: ± 3 points. '
                               'Changes exceeding RCI at 90% CI are clinically meaningful.',
            },
            {
                'name': 'Normative Comparison',
                'description': 'All scores interpreted against age-corrected norms. WAIS: '
                               'age-banded standard scores; MoCA: +1 point for ≤ 12 years '
                               'education. Ethnic and educational demographic adjustments apply.',
            },
        ],
        'compliance': [
            {
                'name': 'APA Ethical Standards',
                'description': 'American Psychological Association ethics code for psychological '
                               'testing: informed consent, test security, competent administration, '
                               'culturally sensitive interpretation, and appropriate reporting.',
            },
            {
                'name': 'ILAE Pre-Surgical Guidelines',
                'description': 'International League Against Epilepsy recommends comprehensive '
                               'neuropsychological evaluation for all epilepsy surgery candidates '
                               'to establish cognitive baseline and predict post-operative risk.',
            },
            {
                'name': 'HIPAA / Data Protection',
                'description': 'Neuropsychological test data is Protected Health Information. '
                               'Raw test data and scoring sheets require secure storage. '
                               'Reports should use standard scores, not raw data, in shared records.',
            },
        ],
        'remediation_strategies': [
            {
                'name': 'AED Optimization',
                'description': 'Switch from cognitively burdensome AEDs (topiramate, phenobarbital) '
                               'to cognitive-neutral alternatives (lamotrigine, brivaracetam) when '
                               'seizure control permits. Monitor with serial neuropsych testing.',
            },
            {
                'name': 'Cognitive Rehabilitation',
                'description': 'Structured programs targeting specific deficits: attention training, '
                               'memory strategies (spaced retrieval, errorless learning), '
                               'compensatory aids (calendars, alarms). Evidence supports '
                               'attention training in epilepsy (Level B recommendation).',
            },
            {
                'name': 'Mood Treatment',
                'description': 'Treat comorbid depression/anxiety, which compound cognitive '
                               'dysfunction. SSRIs are generally safe in epilepsy. CBT has '
                               'evidence for both mood and subjective cognitive improvement.',
            },
            {
                'name': 'Seizure Control',
                'description': 'Reducing seizure frequency improves cognition. Each seizure causes '
                               'transient postictal cognitive suppression (hours to days). '
                               'Subclinical EEG discharges also impair attention and memory. '
                               'Consider surgery evaluation for drug-resistant cases.',
            },
        ],
    }
