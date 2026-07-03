"""Occupational Therapist Dashboard — functional outcome assessment, ADL
independence (Barthel Index), quality of life (QOLIE-31), sleepiness impact
(Epworth), seizure severity (LSSS), rehabilitation planning, and goal
attainment tracking from real clinical evaluations.

Maps clinical.db tables to OT concepts:
- assessments (BARTHEL) -> ADL independence (Barthel Index, 0-100)
- assessments (QOLIE31) -> Quality of Life in Epilepsy (7 domains)
- assessments (EPWORTH) -> Daytime sleepiness (Epworth Sleepiness Scale)
- assessments (LSSS)    -> Seizure severity (Liverpool Seizure Severity Scale)
- assessments (PHQ9)    -> Depression screening (mood impacts function)
- assessments (GAD7)    -> Anxiety screening (anxiety impacts function)
- patients              -> Demographics, disease
- seizure_diary         -> Seizure burden (functional impact)
- medications           -> AED side-effect profiling (function-relevant)
"""

import json
import sqlite3
import os
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Barthel Index interpretation (0-100)
BARTHEL_LEVELS = [
    (0, 20, 'Total Dependence'),
    (21, 60, 'Severe Dependence'),
    (61, 90, 'Moderate Dependence'),
    (91, 99, 'Slight Dependence'),
    (100, 100, 'Independent'),
]

# Barthel Index items (10 items, standard scoring)
BARTHEL_ITEMS = [
    'Feeding',
    'Bathing',
    'Grooming',
    'Dressing',
    'Bowel Control',
    'Bladder Control',
    'Toilet Use',
    'Transfers',
    'Mobility',
    'Stairs',
]

# QOLIE-31 domains (7 subscales)
QOLIE_DOMAINS = [
    'Seizure Worry',
    'Overall QoL',
    'Emotional Well-being',
    'Energy/Fatigue',
    'Cognitive Function',
    'Medication Effects',
    'Social Function',
]

# Epworth Sleepiness Scale thresholds
ESS_LEVELS = [
    (0, 5, 'Lower Normal'),
    (6, 10, 'Higher Normal'),
    (11, 12, 'Mild Sleepiness'),
    (13, 15, 'Moderate Sleepiness'),
    (16, 24, 'Severe Sleepiness'),
]

# LSSS severity thresholds (adapted)
LSSS_LEVELS = [
    (0, 20, 'Mild'),
    (21, 40, 'Moderate'),
    (41, 60, 'Severe'),
    (61, 100, 'Very Severe'),
]

# AEDs known to impact daily function (evidence-based)
AED_FUNCTIONAL_EFFECTS = {
    'Levetiracetam':  {'severity': 'moderate', 'effects': ['fatigue', 'irritability', 'coordination issues']},
    'Topiramate':     {'severity': 'high',     'effects': ['cognitive slowing', 'word-finding difficulty', 'weight loss', 'fatigue']},
    'Phenobarbital':  {'severity': 'high',     'effects': ['sedation', 'motor slowing', 'cognitive impairment', 'falls risk']},
    'Zonisamide':     {'severity': 'moderate', 'effects': ['drowsiness', 'cognitive slowing', 'appetite loss']},
    'Perampanel':     {'severity': 'moderate', 'effects': ['dizziness', 'gait instability', 'falls risk', 'fatigue']},
    'Vigabatrin':     {'severity': 'moderate', 'effects': ['visual field defects', 'drowsiness', 'weight gain']},
    'Felbamate':      {'severity': 'moderate', 'effects': ['insomnia', 'appetite loss', 'nausea']},
    'Clobazam':       {'severity': 'moderate', 'effects': ['sedation', 'drooling', 'motor coordination issues']},
    'Valproate':      {'severity': 'low',      'effects': ['tremor', 'weight gain', 'hair thinning']},
    'Carbamazepine':  {'severity': 'low',      'effects': ['dizziness', 'diplopia', 'mild sedation']},
    'Lamotrigine':    {'severity': 'low',      'effects': ['generally well-tolerated', 'occasional dizziness']},
    'Oxcarbazepine':  {'severity': 'low',      'effects': ['dizziness', 'hyponatremia risk', 'mild fatigue']},
    'Lacosamide':     {'severity': 'low',      'effects': ['dizziness', 'generally well-tolerated']},
    'Brivaracetam':   {'severity': 'low',      'effects': ['mild fatigue', 'generally well-tolerated']},
    'Gabapentin':     {'severity': 'low',      'effects': ['sedation at high doses', 'weight gain']},
    'Pregabalin':     {'severity': 'low',      'effects': ['dizziness', 'weight gain', 'peripheral edema']},
}

# OT goal domains for epilepsy patients
OT_GOAL_DOMAINS = [
    'Self-Care Independence',
    'Home Safety / Falls Prevention',
    'Community Participation',
    'Driving / Transportation',
    'Vocational Rehabilitation',
    'Fatigue Management',
    'Seizure First Aid Training (Family)',
    'Cognitive Compensation Strategies',
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
    return round(sum(values) / len(values), 1) if values else 0


def _classify_barthel(score):
    for lo, hi, label in BARTHEL_LEVELS:
        if lo <= score <= hi:
            return label
    return 'Independent' if score >= 100 else 'Total Dependence'


def _classify_ess(score):
    for lo, hi, label in ESS_LEVELS:
        if lo <= score <= hi:
            return label
    return 'Severe Sleepiness' if score > 15 else 'Lower Normal'


def _classify_lsss(score):
    for lo, hi, label in LSSS_LEVELS:
        if lo <= score <= hi:
            return label
    return 'Very Severe' if score > 60 else 'Mild'


def _classify_qolie(score):
    if score >= 70:
        return 'Good QoL'
    elif score >= 50:
        return 'Moderate QoL'
    elif score >= 30:
        return 'Poor QoL'
    return 'Very Poor QoL'


def _load_assessments(instrument):
    return _db_query(
        'SELECT patient_id, score, max_score, interpretation, level, answers_json, '
        'examiner, created_at FROM assessments WHERE instrument = ? ORDER BY created_at DESC',
        (instrument,)
    )


def _load_patients():
    return {r['patient_id']: r for r in _db_query('SELECT * FROM patients')}


def _load_medications():
    meds = defaultdict(list)
    for r in _db_query('SELECT patient_id, fields_json FROM medications'):
        data = _safe_json(r['fields_json'])
        drug = data.get('drug_name', data.get('medication', ''))
        if drug:
            meds[r['patient_id']].append(drug)
    return dict(meds)


def _load_seizure_counts():
    counts = {}
    for r in _db_query('SELECT patient_id, COUNT(*) as cnt FROM seizure_diary GROUP BY patient_id'):
        counts[r['patient_id']] = r['cnt']
    return counts


def _load_daily_activity():
    rows = _db_query(
        "SELECT DATE(created_at) as day, instrument, COUNT(*) as cnt "
        "FROM assessments WHERE instrument IN ('BARTHEL','QOLIE31','EPWORTH','LSSS') "
        "GROUP BY day, instrument ORDER BY day"
    )
    timeline = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r['day']:
            timeline[r['day']][r['instrument']] = r['cnt']
    return [{'date': d, **dict(v)} for d, v in sorted(timeline.items())]


def overview():
    """Return KPI cards + chart data for the OT overview tab."""
    barthel = _load_assessments('BARTHEL')
    qolie = _load_assessments('QOLIE31')
    epworth = _load_assessments('EPWORTH')
    lsss = _load_assessments('LSSS')
    patients = _load_patients()
    meds = _load_medications()
    seizure_counts = _load_seizure_counts()

    # Unique patients across all OT-relevant instruments
    ot_patients = set()
    for a_list in [barthel, qolie, epworth, lsss]:
        for a in a_list:
            ot_patients.add(a['patient_id'])

    # Barthel Index distribution
    barthel_scores = [a['score'] for a in barthel if a['score'] is not None]
    barthel_severity = Counter(_classify_barthel(s) for s in barthel_scores)
    barthel_dist = [{'level': lbl, 'count': barthel_severity.get(lbl, 0)}
                    for _, _, lbl in BARTHEL_LEVELS]

    # QOLIE-31 distribution
    qolie_scores = [a['score'] for a in qolie if a['score'] is not None]
    qolie_dist_data = Counter(_classify_qolie(s) for s in qolie_scores)
    qolie_dist = [{'level': lbl, 'count': qolie_dist_data.get(lbl, 0)}
                  for lbl in ['Very Poor QoL', 'Poor QoL', 'Moderate QoL', 'Good QoL']]

    # Epworth distribution
    ess_scores = [a['score'] for a in epworth if a['score'] is not None]
    ess_severity = Counter(_classify_ess(s) for s in ess_scores)
    ess_dist = [{'level': lbl, 'count': ess_severity.get(lbl, 0)}
                for _, _, lbl in ESS_LEVELS]

    # LSSS distribution
    lsss_scores = [a['score'] for a in lsss if a['score'] is not None]
    lsss_severity = Counter(_classify_lsss(s) for s in lsss_scores)
    lsss_dist = [{'level': lbl, 'count': lsss_severity.get(lbl, 0)}
                 for _, _, lbl in LSSS_LEVELS]

    # Functional impairment counts
    barthel_impaired = sum(1 for s in barthel_scores if s < 91)
    ess_elevated = sum(1 for s in ess_scores if s >= 11)
    qolie_poor = sum(1 for s in qolie_scores if s < 50)
    lsss_severe = sum(1 for s in lsss_scores if s >= 41)

    # AED functional impact profiling
    aed_risk_summary = []
    all_drugs_seen = set()
    for pid, drug_list in meds.items():
        for drug in drug_list:
            all_drugs_seen.add(drug)
    for drug in sorted(all_drugs_seen):
        info = AED_FUNCTIONAL_EFFECTS.get(drug)
        if info:
            aed_risk_summary.append({
                'drug': drug,
                'severity': info['severity'],
                'effects': info['effects'],
            })

    # Barthel item-level analysis (average per item across all patients)
    barthel_item_avgs = []
    for idx, item_name in enumerate(BARTHEL_ITEMS):
        key = f'item{idx + 1}'
        vals = []
        for a in barthel:
            ans = _safe_json(a['answers_json'])
            v = ans.get(key)
            if v is not None:
                vals.append(v)
        barthel_item_avgs.append({
            'item': item_name,
            'avg_score': _avg(vals),
            'n': len(vals),
        })

    # QOLIE domain-level analysis
    qolie_domain_avgs = []
    for idx, domain_name in enumerate(QOLIE_DOMAINS):
        key = f'item{idx + 1}'
        vals = []
        for a in qolie:
            ans = _safe_json(a['answers_json'])
            v = ans.get(key)
            if v is not None:
                vals.append(v)
        qolie_domain_avgs.append({
            'domain': domain_name,
            'avg_score': _avg(vals),
            'n': len(vals),
        })

    timeline = _load_daily_activity()

    return {
        'kpis': {
            'total_patients': len(ot_patients),
            'barthel_assessments': len(barthel),
            'qolie_assessments': len(qolie),
            'epworth_assessments': len(epworth),
            'lsss_assessments': len(lsss),
            'barthel_impaired': barthel_impaired,
            'ess_elevated': ess_elevated,
            'qolie_poor': qolie_poor,
            'avg_barthel': _avg(barthel_scores),
            'avg_qolie': _avg(qolie_scores),
        },
        'barthel_distribution': barthel_dist,
        'qolie_distribution': qolie_dist,
        'ess_distribution': ess_dist,
        'lsss_distribution': lsss_dist,
        'barthel_items': barthel_item_avgs,
        'qolie_domains': qolie_domain_avgs,
        'aed_functional_risk': aed_risk_summary,
        'timeline': timeline,
    }


def breakdown():
    """Return per-patient functional profiles + detailed breakdowns."""
    barthel = _load_assessments('BARTHEL')
    qolie = _load_assessments('QOLIE31')
    epworth = _load_assessments('EPWORTH')
    lsss = _load_assessments('LSSS')
    phq9 = _load_assessments('PHQ9')
    gad7 = _load_assessments('GAD7')
    patients = _load_patients()
    meds = _load_medications()
    seizure_counts = _load_seizure_counts()

    # Index by patient
    barthel_by_pt = defaultdict(list)
    for a in barthel:
        barthel_by_pt[a['patient_id']].append(a)
    qolie_by_pt = defaultdict(list)
    for a in qolie:
        qolie_by_pt[a['patient_id']].append(a)
    epworth_by_pt = defaultdict(list)
    for a in epworth:
        epworth_by_pt[a['patient_id']].append(a)
    lsss_by_pt = defaultdict(list)
    for a in lsss:
        lsss_by_pt[a['patient_id']].append(a)
    phq9_by_pt = defaultdict(list)
    for a in phq9:
        phq9_by_pt[a['patient_id']].append(a)
    gad7_by_pt = defaultdict(list)
    for a in gad7:
        gad7_by_pt[a['patient_id']].append(a)

    all_pids = set(barthel_by_pt) | set(qolie_by_pt) | set(epworth_by_pt) | set(lsss_by_pt)

    profiles = []
    for pid in sorted(all_pids):
        pt = patients.get(pid, {})
        pt_meds = meds.get(pid, [])

        # Latest scores
        b_score = barthel_by_pt[pid][0]['score'] if barthel_by_pt[pid] else None
        q_score = qolie_by_pt[pid][0]['score'] if qolie_by_pt[pid] else None
        e_score = epworth_by_pt[pid][0]['score'] if epworth_by_pt[pid] else None
        l_score = lsss_by_pt[pid][0]['score'] if lsss_by_pt[pid] else None
        p_score = phq9_by_pt[pid][0]['score'] if phq9_by_pt[pid] else None
        g_score = gad7_by_pt[pid][0]['score'] if gad7_by_pt[pid] else None

        # Risk factors
        risk_factors = []
        if b_score is not None and b_score < 91:
            risk_factors.append(f'ADL impaired (Barthel {b_score})')
        if q_score is not None and q_score < 50:
            risk_factors.append(f'Poor QoL (QOLIE {q_score})')
        if e_score is not None and e_score >= 11:
            risk_factors.append(f'Excessive sleepiness (ESS {e_score})')
        if l_score is not None and l_score >= 41:
            risk_factors.append(f'Severe seizures (LSSS {l_score})')
        if p_score is not None and p_score >= 10:
            risk_factors.append(f'Depression (PHQ-9 {p_score})')
        if g_score is not None and g_score >= 10:
            risk_factors.append(f'Anxiety (GAD-7 {g_score})')
        # AED functional risk
        risky_aeds = [d for d in pt_meds if AED_FUNCTIONAL_EFFECTS.get(d, {}).get('severity') in ('moderate', 'high')]
        if risky_aeds:
            risk_factors.append(f'Functionally risky AEDs: {", ".join(risky_aeds)}')
        sz_count = seizure_counts.get(pid, 0)
        if sz_count >= 3:
            risk_factors.append(f'High seizure burden ({sz_count} events)')

        # Barthel item breakdown for this patient
        barthel_items = []
        if barthel_by_pt[pid]:
            ans = _safe_json(barthel_by_pt[pid][0]['answers_json'])
            for idx, item_name in enumerate(BARTHEL_ITEMS):
                key = f'item{idx + 1}'
                barthel_items.append({
                    'item': item_name,
                    'score': ans.get(key, 0),
                })

        # QOLIE domain breakdown
        qolie_domains = []
        if qolie_by_pt[pid]:
            ans = _safe_json(qolie_by_pt[pid][0]['answers_json'])
            for idx, domain_name in enumerate(QOLIE_DOMAINS):
                key = f'item{idx + 1}'
                qolie_domains.append({
                    'domain': domain_name,
                    'score': ans.get(key, 0),
                })

        # Suggested OT goals based on risk factors
        suggested_goals = []
        if b_score is not None and b_score < 91:
            suggested_goals.append('Self-Care Independence')
        if l_score is not None and l_score >= 41 or sz_count >= 3:
            suggested_goals.append('Home Safety / Falls Prevention')
            suggested_goals.append('Seizure First Aid Training (Family)')
        if q_score is not None and q_score < 50:
            suggested_goals.append('Community Participation')
        if e_score is not None and e_score >= 11:
            suggested_goals.append('Fatigue Management')
        if p_score is not None and p_score >= 10 or g_score is not None and g_score >= 10:
            suggested_goals.append('Cognitive Compensation Strategies')
        if not suggested_goals:
            suggested_goals.append('Vocational Rehabilitation')
            suggested_goals.append('Driving / Transportation')

        profiles.append({
            'patient_id': pid,
            'name': pt.get('name', pid),
            'age': pt.get('age'),
            'gender': pt.get('gender'),
            'disease': pt.get('disease'),
            'barthel_score': b_score,
            'barthel_level': _classify_barthel(b_score) if b_score is not None else None,
            'qolie_score': q_score,
            'qolie_level': _classify_qolie(q_score) if q_score is not None else None,
            'epworth_score': e_score,
            'epworth_level': _classify_ess(e_score) if e_score is not None else None,
            'lsss_score': l_score,
            'lsss_level': _classify_lsss(l_score) if l_score is not None else None,
            'phq9_score': p_score,
            'gad7_score': g_score,
            'medications': pt_meds,
            'seizure_count': sz_count,
            'risk_factors': risk_factors,
            'risk_count': len(risk_factors),
            'barthel_items': barthel_items,
            'qolie_domains': qolie_domains,
            'suggested_goals': suggested_goals,
        })

    # Rehab candidates: patients with >= 2 risk factors
    rehab_candidates = [p for p in profiles if p['risk_count'] >= 2]

    return {
        'profiles': profiles,
        'rehab_candidates': rehab_candidates,
        'goal_domains': OT_GOAL_DOMAINS,
    }


def definitions():
    """Return OT-relevant definitions, quality metrics, and compliance refs."""
    return {
        'concepts': [
            {
                'term': 'Barthel Index',
                'definition': 'A 10-item ordinal scale (0-100) measuring functional independence in activities of daily living (ADLs): feeding, bathing, grooming, dressing, bowel/bladder control, toilet use, transfers, mobility, and stairs. Higher scores indicate greater independence.',
            },
            {
                'term': 'QOLIE-31',
                'definition': 'Quality of Life in Epilepsy — a 31-item self-report instrument measuring 7 domains: seizure worry, overall QoL, emotional well-being, energy/fatigue, cognitive function, medication effects, and social function. Scores range 0-100 (higher = better QoL).',
            },
            {
                'term': 'Epworth Sleepiness Scale (ESS)',
                'definition': 'An 8-item questionnaire measuring general daytime sleepiness. Scores range 0-24; >=11 indicates excessive daytime sleepiness (EDS). Important in epilepsy due to AED side effects and seizure-related sleep disruption.',
            },
            {
                'term': 'Liverpool Seizure Severity Scale (LSSS)',
                'definition': 'A patient-reported outcome measure quantifying the perceived severity of seizure episodes, incorporating ictal and postictal phenomena. Higher scores indicate more severe seizures with greater functional impact.',
            },
            {
                'term': 'Activities of Daily Living (ADLs)',
                'definition': 'Basic self-care tasks essential for independent living: eating, bathing, dressing, toileting, transferring, and continence. Occupational therapists assess ADL capacity to plan rehabilitation goals.',
            },
            {
                'term': 'Instrumental ADLs (IADLs)',
                'definition': 'Higher-level activities required for community living: managing finances, medication management, meal preparation, housekeeping, transportation, and communication. Often impaired in poorly controlled epilepsy.',
            },
            {
                'term': 'Functional Rehabilitation',
                'definition': 'Goal-directed interventions to restore, maintain, or improve functional ability and quality of life. In epilepsy, focuses on seizure safety, ADL training, cognitive compensation, fatigue management, and vocational support.',
            },
            {
                'term': 'AED Functional Side Effects',
                'definition': 'Antiepileptic drugs can impair daily functioning through sedation, cognitive slowing, dizziness, motor coordination issues, and fatigue. OTs must account for medication-related functional limitations in rehab planning.',
            },
        ],
        'quality_metrics': [
            {'metric': 'Barthel Index completeness', 'target': '>= 80% of patients assessed', 'unit': '%'},
            {'metric': 'QOLIE-31 follow-up rate', 'target': '>= 70% at 3-month review', 'unit': '%'},
            {'metric': 'Goal attainment scaling', 'target': '>= 60% goals met at discharge', 'unit': '%'},
            {'metric': 'ADL improvement rate', 'target': '>= 5-point Barthel gain post-rehab', 'unit': 'points'},
        ],
        'compliance': [
            {'standard': 'AOTA Practice Framework (4th ed.)', 'scope': 'OT evaluation domains, intervention approaches'},
            {'standard': 'ILAE Epilepsy Rehabilitation Guidelines', 'scope': 'Seizure-specific functional rehab'},
            {'standard': 'HIPAA', 'scope': 'Patient functional data privacy'},
            {'standard': 'ICF (WHO)', 'scope': 'International Classification of Functioning, Disability and Health'},
        ],
        'remediation': [
            'Refer for OT evaluation if Barthel < 91 or QOLIE < 50',
            'Implement falls prevention protocol if LSSS >= 41 or seizure count >= 3',
            'Screen for excessive daytime sleepiness if ESS >= 11; review AED regimen',
            'Address vocational rehabilitation if seizure-free >= 6 months and Barthel >= 91',
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN (first 2 profiles) ===')
    bd = breakdown()
    for p in bd['profiles'][:2]:
        pprint.pprint(p)
    print(f'\nRehab candidates: {len(bd["rehab_candidates"])}')
    print('\n=== DEFINITIONS ===')
    pprint.pprint(definitions())
