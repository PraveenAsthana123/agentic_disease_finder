"""Clinical Psychologist Dashboard — neuropsychological assessment analysis,
cognitive profiling, mood/anxiety screening, impairment classification,
lateralization hypothesis, and pre-surgical cognitive risk evaluation
for epilepsy patients.

Maps clinical.db tables to neuropsychology concepts:
- neuropsych       -> PHQ-9, GAD-7, MoCA, MMSE, cognitive indices,
                      trail making, digit span, impairment classification,
                      lateralization hypothesis, referral reason
- patients         -> Demographics, disease, department
"""

import json
import sqlite3
import os
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# PHQ-9 severity thresholds
PHQ9_LEVELS = [
    ('minimal',           0,  4),
    ('mild',              5,  9),
    ('moderate',         10, 14),
    ('moderately_severe', 15, 19),
    ('severe',           20, 27),
]

# GAD-7 severity thresholds
GAD7_LEVELS = [
    ('minimal',  0,  4),
    ('mild',     5,  9),
    ('moderate', 10, 14),
    ('severe',   15, 21),
]


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
    return round(sum(values) / len(values), 2) if values else None


def _classify_phq9(score):
    """Classify PHQ-9 score into severity level."""
    if score is None:
        return 'unknown'
    for label, lo, hi in PHQ9_LEVELS:
        if lo <= score <= hi:
            return label
    return 'unknown'


def _classify_gad7(score):
    """Classify GAD-7 score into severity level."""
    if score is None:
        return 'unknown'
    for label, lo, hi in GAD7_LEVELS:
        if lo <= score <= hi:
            return label
    return 'unknown'


def _load_neuropsych():
    """Load all neuropsych records with parsed JSON fields."""
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at FROM neuropsych "
        "ORDER BY created_at DESC"
    )
    results = []
    for r in rows:
        fields = _safe_json(r.get('fields_json'))
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'created_at': r.get('created_at'),
            **fields,
        })
    return results


def _load_patients():
    """Load all patients."""
    return _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients"
    )


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Clinical Psychologist dashboard."""
    np_list = _load_neuropsych()
    patients = _load_patients()

    if not np_list:
        return {
            'available': False,
            'message': 'No neuropsychological assessment data available. '
                       'Import records into the neuropsych table first.',
        }

    total_assessments = len(np_list)
    patient_ids_assessed = set(r['patient_id'] for r in np_list if r.get('patient_id'))
    total_patients_assessed = len(patient_ids_assessed)

    # Battery type distribution
    battery_counts = Counter(r.get('battery_type', 'Unknown') for r in np_list)
    battery_type_distribution = [
        {'battery_type': bt, 'count': c}
        for bt, c in battery_counts.most_common()
    ]

    # MoCA and MMSE
    moca_values = [r['moca'] for r in np_list if r.get('moca') is not None]
    mmse_values = [r['mmse'] for r in np_list if r.get('mmse') is not None]
    avg_moca = _avg(moca_values)
    avg_mmse = _avg(mmse_values)
    moca_impaired_rate = round(
        sum(1 for v in moca_values if v < 26) / len(moca_values) * 100, 1
    ) if moca_values else None
    mmse_impaired_rate = round(
        sum(1 for v in mmse_values if v < 26) / len(mmse_values) * 100, 1
    ) if mmse_values else None

    # PHQ-9 and GAD-7 averages
    phq9_values = [r['phq9'] for r in np_list if r.get('phq9') is not None]
    gad7_values = [r['gad7'] for r in np_list if r.get('gad7') is not None]
    avg_phq9 = _avg(phq9_values)
    avg_gad7 = _avg(gad7_values)

    # Depression distribution (PHQ-9 levels)
    dep_counts = Counter(_classify_phq9(r.get('phq9')) for r in np_list)
    depression_distribution = [
        {'level': label, 'count': dep_counts.get(label, 0)}
        for label, _, _ in PHQ9_LEVELS
    ]

    # Anxiety distribution (GAD-7 levels)
    anx_counts = Counter(_classify_gad7(r.get('gad7')) for r in np_list)
    anxiety_distribution = [
        {'level': label, 'count': anx_counts.get(label, 0)}
        for label, _, _ in GAD7_LEVELS
    ]

    # Impairment distribution
    imp_counts = Counter(r.get('impairment_flag', 'unknown') for r in np_list)
    impairment_distribution = [
        {'level': level, 'count': imp_counts.get(level, 0)}
        for level in ('none', 'mild', 'moderate', 'severe')
    ]

    # Cognitive index means
    index_names = [
        ('memory_index', 'Memory'),
        ('attention_index', 'Attention'),
        ('executive_index', 'Executive'),
        ('language_index', 'Language'),
        ('processing_speed_index', 'Processing Speed'),
    ]
    cognitive_index_means = {}
    for key, label in index_names:
        vals = [r[key] for r in np_list if r.get(key) is not None]
        cognitive_index_means[label] = _avg(vals)

    # Lateralization distribution
    lat_counts = Counter(
        r.get('lateralization_hypothesis', 'unknown') for r in np_list
    )
    lateralization_distribution = [
        {'lateralization': lat, 'count': c}
        for lat, c in lat_counts.most_common()
    ]

    # Referral reason distribution
    ref_counts = Counter(
        r.get('referral_reason', 'unknown') for r in np_list
    )
    referral_reason_distribution = [
        {'reason': reason, 'count': c}
        for reason, c in ref_counts.most_common()
    ]

    return {
        'available': True,
        'total_assessments': total_assessments,
        'total_patients_assessed': total_patients_assessed,
        'battery_type_distribution': battery_type_distribution,
        'avg_moca': avg_moca,
        'avg_mmse': avg_mmse,
        'moca_impaired_rate': moca_impaired_rate,
        'mmse_impaired_rate': mmse_impaired_rate,
        'avg_phq9': avg_phq9,
        'avg_gad7': avg_gad7,
        'depression_distribution': depression_distribution,
        'anxiety_distribution': anxiety_distribution,
        'impairment_distribution': impairment_distribution,
        'cognitive_index_means': cognitive_index_means,
        'lateralization_distribution': lateralization_distribution,
        'referral_reason_distribution': referral_reason_distribution,
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed per-patient neuropsych data with cognitive profiles."""
    np_list = _load_neuropsych()
    patients = _load_patients()

    patient_map = {}
    for p in patients:
        pid = p.get('patient_id')
        if pid:
            patient_map[pid] = p

    # Group all assessments by patient (include multiple assessments)
    assessments_by_patient = defaultdict(list)
    for r in np_list:
        pid = r.get('patient_id')
        if pid:
            assessments_by_patient[pid].append(r)

    patient_list = []
    for pid in sorted(assessments_by_patient.keys()):
        pinfo = patient_map.get(pid, {})
        records = assessments_by_patient[pid]
        # Sort by date ascending so follow-ups come after baseline
        records.sort(key=lambda x: x.get('created_at', ''))

        assessments = []
        for rec in records:
            assessments.append({
                'id': rec.get('id'),
                'battery_type': rec.get('battery_type'),
                'created_at': rec.get('created_at'),
                'phq9': rec.get('phq9'),
                'gad7': rec.get('gad7'),
                'moca': rec.get('moca'),
                'mmse': rec.get('mmse'),
                'memory_index': rec.get('memory_index'),
                'attention_index': rec.get('attention_index'),
                'executive_index': rec.get('executive_index'),
                'language_index': rec.get('language_index'),
                'processing_speed_index': rec.get('processing_speed_index'),
                'verbal_memory_raw': rec.get('verbal_memory_raw'),
                'visual_memory_raw': rec.get('visual_memory_raw'),
                'digit_span_forward': rec.get('digit_span_forward'),
                'digit_span_backward': rec.get('digit_span_backward'),
                'trail_a_seconds': rec.get('trail_a_seconds'),
                'trail_b_seconds': rec.get('trail_b_seconds'),
                'impairment_flag': rec.get('impairment_flag'),
                'lateralization_hypothesis': rec.get('lateralization_hypothesis'),
                'referral_reason': rec.get('referral_reason'),
                'assessor': rec.get('assessor'),
            })

        patient_list.append({
            'patient_id': pid,
            'name': pinfo.get('name'),
            'age': pinfo.get('age'),
            'gender': pinfo.get('gender'),
            'disease': pinfo.get('disease'),
            'assessment_count': len(assessments),
            'assessments': assessments,
        })

    # Cognitive profile chart: mean score per index across all assessments
    index_keys = [
        ('memory_index', 'Memory'),
        ('attention_index', 'Attention'),
        ('executive_index', 'Executive'),
        ('language_index', 'Language'),
        ('processing_speed_index', 'Processing Speed'),
    ]
    cognitive_profile_chart = []
    for key, label in index_keys:
        vals = [r[key] for r in np_list if r.get(key) is not None]
        cognitive_profile_chart.append({
            'index': label,
            'mean_score': _avg(vals),
            'n': len(vals),
        })

    # Trail making stats
    trail_a_vals = [r['trail_a_seconds'] for r in np_list if r.get('trail_a_seconds') is not None]
    trail_b_vals = [r['trail_b_seconds'] for r in np_list if r.get('trail_b_seconds') is not None]
    avg_trail_a = _avg(trail_a_vals)
    avg_trail_b = _avg(trail_b_vals)
    trail_b_a_ratio = round(avg_trail_b / avg_trail_a, 2) if avg_trail_a and avg_trail_b else None
    trail_making_stats = {
        'avg_trail_a_seconds': avg_trail_a,
        'avg_trail_b_seconds': avg_trail_b,
        'trail_b_a_ratio': trail_b_a_ratio,
    }

    # Memory lateralization cross-tab
    lat_memory = defaultdict(list)
    for r in np_list:
        lat = r.get('lateralization_hypothesis', 'unknown')
        mem = r.get('memory_index')
        if mem is not None:
            lat_memory[lat].append(mem)
    memory_lateralization = [
        {
            'lateralization': lat,
            'mean_memory_index': _avg(vals),
            'n': len(vals),
        }
        for lat, vals in sorted(lat_memory.items())
    ]

    # Mood comorbidity
    phq9_elevated = sum(1 for r in np_list if (r.get('phq9') or 0) >= 10)
    gad7_elevated = sum(1 for r in np_list if (r.get('gad7') or 0) >= 10)
    both_elevated = sum(
        1 for r in np_list
        if (r.get('phq9') or 0) >= 10 and (r.get('gad7') or 0) >= 10
    )
    mood_comorbidity = {
        'phq9_elevated_count': phq9_elevated,
        'gad7_elevated_count': gad7_elevated,
        'both_elevated_count': both_elevated,
        'total_assessments': len(np_list),
    }

    return {
        'patients': patient_list,
        'cognitive_profile_chart': cognitive_profile_chart,
        'trail_making_stats': trail_making_stats,
        'memory_lateralization': memory_lateralization,
        'mood_comorbidity': mood_comorbidity,
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Clinical Psychologist dashboard —
    neuropsychological assessment concepts, scoring, and ILAE references."""
    return {
        'concepts': [
            {
                'name': 'PHQ-9 (Patient Health Questionnaire-9)',
                'description': 'A 9-item self-report screening tool for depression severity. '
                               'Each item scored 0-3 (not at all to nearly every day), total 0-27. '
                               'Thresholds: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately '
                               'severe, 20-27 severe depression. Depression is highly prevalent in '
                               'epilepsy (20-30% lifetime), particularly temporal lobe epilepsy. '
                               'PHQ-9 >= 10 has 88% sensitivity and 88% specificity for major '
                               'depression (Kroenke et al., 2001).',
            },
            {
                'name': 'GAD-7 (Generalized Anxiety Disorder-7)',
                'description': 'A 7-item self-report measure of anxiety severity. Each item scored '
                               '0-3, total 0-21. Thresholds: 0-4 minimal, 5-9 mild, 10-14 moderate, '
                               '15-21 severe anxiety. Anxiety disorders affect up to 20-25% of '
                               'epilepsy patients, often comorbid with depression. Seizure '
                               'anticipatory anxiety and ictal fear are unique to epilepsy. '
                               'GAD-7 >= 10 indicates clinically significant anxiety (Spitzer et al., '
                               '2006).',
            },
            {
                'name': 'MoCA (Montreal Cognitive Assessment)',
                'description': 'A 30-point screening tool for mild cognitive impairment covering '
                               'visuospatial/executive, naming, attention, language, abstraction, '
                               'delayed recall, and orientation. Score >= 26 is normal; 18-25 suggests '
                               'mild cognitive impairment; < 18 suggests moderate impairment. Widely '
                               'used in epilepsy clinics for rapid cognitive screening. More sensitive '
                               'than MMSE for executive and visuospatial deficits common in epilepsy '
                               '(Nasreddine et al., 2005).',
            },
            {
                'name': 'MMSE (Mini-Mental State Examination)',
                'description': 'A 30-point cognitive screening tool assessing orientation, registration, '
                               'attention/calculation, recall, language, and visuospatial skills. '
                               'Score >= 26 normal, 20-25 mild impairment, 10-19 moderate, < 10 severe. '
                               'Less sensitive than MoCA for subtle deficits but widely used as a '
                               'baseline measure. Ceiling effects limit utility in younger, higher-'
                               'functioning epilepsy patients (Folstein et al., 1975).',
            },
            {
                'name': 'Cognitive Indices (Standardized Scores)',
                'description': 'Neuropsychological test scores are typically converted to standardized '
                               'scores with mean = 100 and SD = 15 (like IQ). Scores are classified as: '
                               'Average (90-109), Low Average (80-89), Borderline (70-79), Impaired '
                               '(< 70), High Average (110-119), Superior (120-129), Very Superior '
                               '(130+). In epilepsy, Memory and Processing Speed indices are most '
                               'commonly affected. The five core indices are: Memory, Attention, '
                               'Executive Function, Language, and Processing Speed.',
            },
            {
                'name': 'Trail Making Test (TMT)',
                'description': 'Part A: Connect numbered circles in sequence (measures processing speed '
                               'and visual scanning). Part B: Alternate between numbers and letters '
                               '(measures cognitive flexibility/executive function). Trail B:A ratio '
                               '> 2.5 suggests executive dysfunction beyond simple slowing. Epilepsy '
                               'patients, especially frontal lobe epilepsy, commonly show elevated '
                               'Trail B times. Normative data: Trail A < 29s, Trail B < 75s for ages '
                               '18-39 (Tombaugh, 2004).',
            },
            {
                'name': 'Verbal vs Visual Memory Lateralization',
                'description': 'Material-specific memory lateralization is a key neuropsychological '
                               'finding in epilepsy. Left temporal lobe epilepsy (TLE) typically shows '
                               'verbal memory deficits (word lists, stories), while right TLE shows '
                               'visual/spatial memory deficits (figures, faces, spatial locations). '
                               'This pattern helps lateralize the seizure focus and predict post-surgical '
                               'memory outcomes. Assessed via RAVLT/CVLT (verbal) and RCFT/BVMT-R '
                               '(visual). Wada test may supplement neuropsychological lateralization.',
            },
            {
                'name': 'Impairment Classification',
                'description': 'Overall cognitive impairment is classified based on the average of '
                               'core cognitive indices: None (mean >= 90, within normal limits), '
                               'Mild (80-89, low average with isolated deficits), Moderate (70-79, '
                               'borderline with multiple deficits affecting daily function), Severe '
                               '(< 70, significant global impairment). Pre-surgical patients with '
                               'moderate-to-severe impairment warrant careful discussion of cognitive '
                               'risks of surgery.',
            },
            {
                'name': 'Pre-Surgical Neuropsychological Evaluation',
                'description': 'ILAE recommends comprehensive neuropsychological assessment before '
                               'epilepsy surgery to: (1) establish a cognitive baseline, (2) lateralize '
                               'and localize cognitive dysfunction to support the seizure focus hypothesis, '
                               '(3) predict cognitive risks of surgery (especially memory decline after '
                               'temporal lobectomy), (4) inform surgical counseling and consent. The '
                               'evaluation should include measures of memory, attention, executive '
                               'function, language, and mood (Baxendale et al., Epilepsia 2019).',
            },
            {
                'name': 'Digit Span (Forward and Backward)',
                'description': 'Forward digit span measures basic auditory attention and short-term '
                               'memory (normal: 6-7 digits). Backward digit span additionally measures '
                               'working memory and mental manipulation (normal: 4-5 digits). The '
                               'forward-backward difference can indicate specific working memory '
                               'deficits. Commonly impaired in frontal and temporal lobe epilepsy.',
            },
            {
                'name': 'ILAE Neuropsychology References',
                'description': 'Key references: Baxendale et al., Epilepsia 2019 (Neuropsychological '
                               'assessment in epilepsy surgery — ILAE Neuropsychology Task Force '
                               'recommendations); Helmstaedter et al., Epilepsia 2003 (Cognitive '
                               'outcomes of epilepsy surgery); Loring et al., Neuropsychology Review '
                               '2010 (When should memory be tested in epilepsy surgery candidates?); '
                               'Jones-Gotman et al., Epilepsia 2010 (ILAE Neuropsychology Task Force '
                               'diagnostic methods commission).',
            },
        ],
    }
