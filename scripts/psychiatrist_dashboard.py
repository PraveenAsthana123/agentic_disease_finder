"""Psychiatrist Dashboard — psychiatric comorbidity assessment, mood/anxiety
screening, PNES differential, suicidality risk, medication psychiatric effects,
and patient psychiatric profiles from real clinical evaluations.

Maps clinical.db tables to psychiatric concepts:
- assessments (PHQ9)    -> Depression severity (PHQ-9 screening)
- assessments (GAD7)    -> Anxiety severity (GAD-7 screening)
- assessments (CSSRS)   -> Suicidality risk (C-SSRS screening)
- assessments (NDDIE)   -> Epilepsy-specific depression (NDDI-E)
- patients              -> Demographics, disease, lateralization
- seizure_diary         -> Seizure burden (psychiatric correlate)
- medications           -> AED psychiatric side-effect profiling
"""

import json
import sqlite3
import os
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# PHQ-9 severity thresholds
PHQ9_LEVELS = [
    (0, 4, 'Minimal'),
    (5, 9, 'Mild'),
    (10, 14, 'Moderate'),
    (15, 19, 'Moderately Severe'),
    (20, 27, 'Severe'),
]

# GAD-7 severity thresholds
GAD7_LEVELS = [
    (0, 4, 'Minimal'),
    (5, 9, 'Mild'),
    (10, 14, 'Moderate'),
    (15, 21, 'Severe'),
]

# C-SSRS risk levels
CSSRS_RISK = {
    'none': 'No Risk',
    'low': 'Low Risk',
    'moderate': 'Moderate Risk',
    'high': 'High Risk',
}

# AEDs known to have psychiatric side effects (evidence-based)
AED_PSYCHIATRIC_EFFECTS = {
    'Levetiracetam':  {'severity': 'moderate', 'effects': ['irritability', 'aggression', 'depression', 'psychosis (rare)']},
    'Topiramate':     {'severity': 'moderate', 'effects': ['depression', 'anxiety', 'psychomotor slowing', 'word-finding difficulty']},
    'Phenobarbital':  {'severity': 'high',     'effects': ['depression', 'sedation', 'behavioral changes', 'suicidality risk']},
    'Zonisamide':     {'severity': 'moderate', 'effects': ['depression', 'psychosis (rare)', 'irritability']},
    'Perampanel':     {'severity': 'moderate', 'effects': ['irritability', 'aggression', 'hostility', 'homicidal ideation (rare)']},
    'Vigabatrin':     {'severity': 'moderate', 'effects': ['depression', 'psychosis', 'behavioral changes']},
    'Felbamate':      {'severity': 'moderate', 'effects': ['insomnia', 'anxiety', 'depression']},
    'Clobazam':       {'severity': 'low',      'effects': ['sedation', 'behavioral disinhibition (children)']},
    'Valproate':      {'severity': 'low',      'effects': ['weight gain (mood impact)', 'tremor']},
    'Carbamazepine':  {'severity': 'low',      'effects': ['sedation', 'rare hyponatremia (confusion)']},
    'Lamotrigine':    {'severity': 'beneficial','effects': ['mood stabilization', 'antidepressant effect', 'anxiety reduction']},
    'Oxcarbazepine':  {'severity': 'low',      'effects': ['mild sedation', 'hyponatremia (confusion)']},
    'Lacosamide':     {'severity': 'low',      'effects': ['dizziness', 'generally psychiatrically neutral']},
    'Brivaracetam':   {'severity': 'low',      'effects': ['less irritability than levetiracetam', 'generally well-tolerated']},
    'Gabapentin':     {'severity': 'beneficial','effects': ['anxiolytic properties', 'sleep improvement']},
    'Pregabalin':     {'severity': 'beneficial','effects': ['anxiolytic properties', 'sleep improvement', 'GAD indication']},
}

# PNES vs epileptic seizure differentiating features
PNES_FEATURES = [
    'Duration > 2 minutes',
    'Asynchronous limb movements',
    'Pelvic thrusting',
    'Side-to-side head movement',
    'Eye closure during event',
    'Preserved awareness with bilateral motor',
    'Gradual onset',
    'Waxing-waning intensity',
    'No postictal confusion',
    'Crying/emotional expression during event',
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


def _classify_phq9(score):
    for lo, hi, label in PHQ9_LEVELS:
        if lo <= score <= hi:
            return label
    return 'Severe' if score > 20 else 'Minimal'


def _classify_gad7(score):
    for lo, hi, label in GAD7_LEVELS:
        if lo <= score <= hi:
            return label
    return 'Severe' if score > 14 else 'Minimal'


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
        "FROM assessments WHERE instrument IN ('PHQ9','GAD7','CSSRS','NDDIE') "
        "GROUP BY day, instrument ORDER BY day"
    )
    timeline = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r['day']:
            timeline[r['day']][r['instrument']] = r['cnt']
    return [{'date': d, **dict(v)} for d, v in sorted(timeline.items())]


def _load_comorbidities():
    """Load comorbidity screening records keyed by patient_id."""
    rows = _db_query('SELECT patient_id, fields_json, created_at FROM comorbidities ORDER BY created_at DESC')
    by_pt = {}
    for r in rows:
        pid = r['patient_id']
        if pid not in by_pt:
            by_pt[pid] = _safe_json(r['fields_json'])
            by_pt[pid]['created_at'] = r['created_at']
    return by_pt


def _load_referrals():
    """Load referral entries from transaction_log."""
    rows = _db_query(
        "SELECT patient_id, action, actor, detail, ts_local "
        "FROM transaction_log WHERE component = 'referral' ORDER BY ts_utc DESC"
    )
    return rows


def overview():
    """Return KPI cards + chart data for the psychiatrist overview tab."""
    phq9 = _load_assessments('PHQ9')
    gad7 = _load_assessments('GAD7')
    cssrs = _load_assessments('CSSRS')
    nddie = _load_assessments('NDDIE')
    patients = _load_patients()
    meds = _load_medications()
    seizure_counts = _load_seizure_counts()

    # Unique patients across all psychiatric instruments
    psych_patients = set()
    for a_list in [phq9, gad7, cssrs, nddie]:
        for a in a_list:
            psych_patients.add(a['patient_id'])

    # PHQ-9 severity distribution
    phq9_scores = [a['score'] for a in phq9 if a['score'] is not None]
    phq9_severity = Counter(_classify_phq9(s) for s in phq9_scores)
    phq9_dist = [{'level': lbl, 'count': phq9_severity.get(lbl, 0)}
                 for _, _, lbl in PHQ9_LEVELS]

    # GAD-7 severity distribution
    gad7_scores = [a['score'] for a in gad7 if a['score'] is not None]
    gad7_severity = Counter(_classify_gad7(s) for s in gad7_scores)
    gad7_dist = [{'level': lbl, 'count': gad7_severity.get(lbl, 0)}
                 for _, _, lbl in GAD7_LEVELS]

    # C-SSRS risk distribution
    cssrs_levels = Counter()
    for a in cssrs:
        lvl = (a.get('level') or 'none').lower()
        cssrs_levels[CSSRS_RISK.get(lvl, 'No Risk')] += 1
    cssrs_dist = [{'level': k, 'count': v} for k, v in cssrs_levels.items()]

    # Elevated counts (clinically significant)
    phq9_elevated = sum(1 for s in phq9_scores if s >= 10)  # Moderate+
    gad7_elevated = sum(1 for s in gad7_scores if s >= 10)
    cssrs_positive = sum(1 for a in cssrs if (a.get('score') or 0) > 0)

    # NDDI-E above threshold (>=15 = major depression likely in epilepsy)
    nddie_scores = [a['score'] for a in nddie if a['score'] is not None]
    nddie_elevated = sum(1 for s in nddie_scores if s >= 15)

    # AED psychiatric risk profiling
    aed_risk_summary = []
    all_drugs_seen = set()
    for pid, drug_list in meds.items():
        for drug in drug_list:
            all_drugs_seen.add(drug)
    for drug in sorted(all_drugs_seen):
        info = AED_PSYCHIATRIC_EFFECTS.get(drug)
        if info:
            aed_risk_summary.append({
                'drug': drug,
                'severity': info['severity'],
                'effects': info['effects'],
            })

    # Timeline
    timeline = _load_daily_activity()

    # Comorbidity screening KPIs
    comorb = _load_comorbidities()
    screened_count = sum(1 for c in comorb.values() if c.get('screened'))
    screen_rate = round(screened_count / len(psych_patients) * 100, 1) if psych_patients else 0
    all_conditions = []
    for c in comorb.values():
        all_conditions.extend(c.get('comorbidities', []))
    condition_dist = Counter(all_conditions)
    condition_chart = [{'condition': k, 'count': v} for k, v in
                       sorted(condition_dist.items(), key=lambda x: -x[1])]

    # Behavioral risk score KPIs
    risk_scores = [c.get('behavioral_risk_score', 0) for c in comorb.values()
                   if c.get('behavioral_risk_score') is not None]
    avg_risk = _avg(risk_scores)
    risk_severity_dist = Counter(c.get('risk_severity', 'minimal') for c in comorb.values())
    risk_chart = [{'level': lbl, 'count': risk_severity_dist.get(lbl, 0)}
                  for lbl in ['minimal', 'mild', 'moderate', 'severe']]
    high_risk_count = sum(1 for s in risk_scores if s >= 60)

    # Referrals KPIs
    referrals = _load_referrals()
    refs_to_neuro = [r for r in referrals if r['action'] == 'refer_to_neurology']
    refs_to_psych = [r for r in referrals if r['action'] == 'refer_to_psychiatry']

    # Treatment status distribution
    treatment_dist = Counter(c.get('treatment_status', 'none') for c in comorb.values()
                             if c.get('comorbidity_count', 0) > 0)
    treatment_chart = [{'status': k, 'count': v} for k, v in treatment_dist.items()]

    return {
        'kpis': {
            'total_patients': len(psych_patients),
            'phq9_assessments': len(phq9),
            'gad7_assessments': len(gad7),
            'cssrs_assessments': len(cssrs),
            'nddie_assessments': len(nddie),
            'phq9_elevated': phq9_elevated,
            'gad7_elevated': gad7_elevated,
            'cssrs_positive': cssrs_positive,
            'avg_phq9': _avg(phq9_scores),
            'avg_gad7': _avg(gad7_scores),
            'comorbidity_screen_rate': screen_rate,
            'screened_patients': screened_count,
            'avg_behavioral_risk': avg_risk,
            'high_risk_patients': high_risk_count,
            'referrals_to_neurology': len(refs_to_neuro),
            'referrals_from_neurology': len(refs_to_psych),
            'total_comorbidities': len(all_conditions),
            'unique_conditions': len(condition_dist),
        },
        'phq9_severity': phq9_dist,
        'gad7_severity': gad7_dist,
        'cssrs_risk': cssrs_dist,
        'aed_psychiatric_risk': aed_risk_summary,
        'timeline': timeline,
        'comorbidity_distribution': condition_chart,
        'risk_severity_distribution': risk_chart,
        'treatment_status': treatment_chart,
        'referral_summary': {
            'to_neurology': len(refs_to_neuro),
            'from_neurology': len(refs_to_psych),
            'total': len(referrals),
        },
    }


def breakdown():
    """Return per-patient psychiatric profiles + detailed breakdowns."""
    phq9 = _load_assessments('PHQ9')
    gad7 = _load_assessments('GAD7')
    cssrs = _load_assessments('CSSRS')
    nddie = _load_assessments('NDDIE')
    patients = _load_patients()
    meds = _load_medications()
    seizure_counts = _load_seizure_counts()

    comorb = _load_comorbidities()
    referrals = _load_referrals()
    refs_by_pt = defaultdict(list)
    for r in referrals:
        refs_by_pt[r['patient_id']].append(r)

    # Index by patient
    phq9_by_pt = defaultdict(list)
    for a in phq9:
        phq9_by_pt[a['patient_id']].append(a)
    gad7_by_pt = defaultdict(list)
    for a in gad7:
        gad7_by_pt[a['patient_id']].append(a)
    cssrs_by_pt = defaultdict(list)
    for a in cssrs:
        cssrs_by_pt[a['patient_id']].append(a)
    nddie_by_pt = defaultdict(list)
    for a in nddie:
        nddie_by_pt[a['patient_id']].append(a)

    # All patients with any psychiatric assessment
    all_pids = set(phq9_by_pt) | set(gad7_by_pt) | set(cssrs_by_pt) | set(nddie_by_pt)

    profiles = []
    for pid in sorted(all_pids):
        pt = patients.get(pid, {})
        pt_meds = meds.get(pid, [])
        seizures = seizure_counts.get(pid, 0)

        # Latest scores
        latest_phq9 = phq9_by_pt[pid][0]['score'] if phq9_by_pt[pid] else None
        latest_gad7 = gad7_by_pt[pid][0]['score'] if gad7_by_pt[pid] else None
        latest_cssrs = cssrs_by_pt[pid][0]['score'] if cssrs_by_pt[pid] else None
        latest_nddie = nddie_by_pt[pid][0]['score'] if nddie_by_pt[pid] else None

        # Risk flags
        flags = []
        if latest_phq9 is not None and latest_phq9 >= 10:
            flags.append(f'Depression ({_classify_phq9(latest_phq9)})')
        if latest_gad7 is not None and latest_gad7 >= 10:
            flags.append(f'Anxiety ({_classify_gad7(latest_gad7)})')
        if latest_cssrs is not None and latest_cssrs > 0:
            flags.append('Suicidality risk (C-SSRS positive)')
        if latest_nddie is not None and latest_nddie >= 15:
            flags.append('NDDI-E elevated (epilepsy depression)')

        # AED psychiatric risk
        risky_aeds = []
        for drug in pt_meds:
            info = AED_PSYCHIATRIC_EFFECTS.get(drug)
            if info and info['severity'] in ('moderate', 'high'):
                risky_aeds.append(drug)
        if risky_aeds:
            flags.append(f'Psychiatric-risk AEDs: {", ".join(risky_aeds)}')

        if seizures >= 5:
            flags.append(f'High seizure burden ({seizures} events)')

        # PHQ-9 item breakdown (most recent)
        phq9_items = []
        if phq9_by_pt[pid]:
            answers = _safe_json(phq9_by_pt[pid][0].get('answers_json'))
            item_labels = [
                'Anhedonia', 'Depressed mood', 'Sleep disturbance',
                'Fatigue', 'Appetite changes', 'Guilt/worthlessness',
                'Concentration difficulty', 'Psychomotor changes', 'Suicidal ideation'
            ]
            for i, label in enumerate(item_labels, 1):
                val = answers.get(f'item{i}')
                if val is not None:
                    phq9_items.append({'item': label, 'score': val})

        # GAD-7 item breakdown
        gad7_items = []
        if gad7_by_pt[pid]:
            answers = _safe_json(gad7_by_pt[pid][0].get('answers_json'))
            item_labels = [
                'Nervous/anxious', 'Uncontrollable worry', 'Excessive worry',
                'Trouble relaxing', 'Restlessness', 'Irritability', 'Feeling afraid'
            ]
            for i, label in enumerate(item_labels, 1):
                val = answers.get(f'item{i}')
                if val is not None:
                    gad7_items.append({'item': label, 'score': val})

        # Comorbidity data
        pt_comorb = comorb.get(pid, {})
        pt_refs = refs_by_pt.get(pid, [])

        profiles.append({
            'patient_id': pid,
            'name': pt.get('name', ''),
            'age': pt.get('age'),
            'gender': pt.get('gender', ''),
            'disease': pt.get('disease', ''),
            'medications': pt_meds,
            'seizure_count': seizures,
            'latest_phq9': latest_phq9,
            'phq9_level': _classify_phq9(latest_phq9) if latest_phq9 is not None else None,
            'latest_gad7': latest_gad7,
            'gad7_level': _classify_gad7(latest_gad7) if latest_gad7 is not None else None,
            'latest_cssrs': latest_cssrs,
            'latest_nddie': latest_nddie,
            'risk_flags': flags,
            'phq9_items': phq9_items,
            'gad7_items': gad7_items,
            'assessments_count': (len(phq9_by_pt[pid]) + len(gad7_by_pt[pid])
                                  + len(cssrs_by_pt[pid]) + len(nddie_by_pt[pid])),
            'comorbidities': pt_comorb.get('comorbidities', []),
            'comorbidity_count': pt_comorb.get('comorbidity_count', 0),
            'behavioral_risk_score': pt_comorb.get('behavioral_risk_score'),
            'risk_severity': pt_comorb.get('risk_severity'),
            'screening_instruments': pt_comorb.get('screening_instruments', []),
            'screened': pt_comorb.get('screened', False),
            'screening_date': pt_comorb.get('screening_date'),
            'treatment_status': pt_comorb.get('treatment_status'),
            'functional_impact': pt_comorb.get('functional_impact'),
            'referrals': [{'action': r['action'], 'actor': r['actor'],
                           'detail': r['detail'], 'date': r['ts_local']}
                          for r in pt_refs],
        })

    # PNES screening candidates: patients with high seizure burden + psychiatric flags
    pnes_candidates = []
    for p in profiles:
        pnes_score = 0
        if p['latest_phq9'] is not None and p['latest_phq9'] >= 10:
            pnes_score += 1
        if p['latest_gad7'] is not None and p['latest_gad7'] >= 10:
            pnes_score += 1
        if p['latest_cssrs'] is not None and p['latest_cssrs'] > 0:
            pnes_score += 1
        if p['seizure_count'] >= 3:
            pnes_score += 1
        if pnes_score >= 2:
            pnes_candidates.append({
                'patient_id': p['patient_id'],
                'name': p['name'],
                'pnes_risk_factors': pnes_score,
                'flags': p['risk_flags'],
            })

    # Mood-seizure correlation: patients with both mood data and seizure diary
    mood_seizure = []
    for p in profiles:
        if p['latest_phq9'] is not None and p['seizure_count'] > 0:
            mood_seizure.append({
                'patient_id': p['patient_id'],
                'phq9': p['latest_phq9'],
                'gad7': p['latest_gad7'],
                'seizures': p['seizure_count'],
            })

    return {
        'patient_profiles': profiles,
        'pnes_candidates': pnes_candidates,
        'mood_seizure_correlation': mood_seizure,
        'pnes_features': PNES_FEATURES,
    }


def threshold_flags():
    """Consolidated threshold-based alert system for mood/anxiety screening.

    Returns patients who exceed clinical thresholds across any instrument
    (PHQ-9 >= 10, GAD-7 >= 10, C-SSRS > 0, NDDI-E >= 15), with severity,
    recommended actions, and priority ranking.
    """
    phq9 = _load_assessments('PHQ9')
    gad7 = _load_assessments('GAD7')
    cssrs = _load_assessments('CSSRS')
    nddie = _load_assessments('NDDIE')
    patients = _load_patients()
    meds = _load_medications()
    seizure_counts = _load_seizure_counts()

    # Index by patient (latest score per instrument)
    phq9_by_pt = {}
    for a in phq9:
        if a['patient_id'] not in phq9_by_pt:
            phq9_by_pt[a['patient_id']] = a
    gad7_by_pt = {}
    for a in gad7:
        if a['patient_id'] not in gad7_by_pt:
            gad7_by_pt[a['patient_id']] = a
    cssrs_by_pt = {}
    for a in cssrs:
        if a['patient_id'] not in cssrs_by_pt:
            cssrs_by_pt[a['patient_id']] = a
    nddie_by_pt = {}
    for a in nddie:
        if a['patient_id'] not in nddie_by_pt:
            nddie_by_pt[a['patient_id']] = a

    all_pids = set(phq9_by_pt) | set(gad7_by_pt) | set(cssrs_by_pt) | set(nddie_by_pt)

    # Clinical action thresholds
    THRESHOLDS = {
        'PHQ-9':  {'cutoff': 10, 'action': 'Initiate/adjust antidepressant; refer for CBT; rescreen in 4 weeks'},
        'GAD-7':  {'cutoff': 10, 'action': 'Consider SSRI/SNRI or pregabalin; CBT referral; rescreen in 4 weeks'},
        'C-SSRS': {'cutoff': 1,  'action': 'Same-day safety assessment; Stanley-Brown safety plan; means restriction'},
        'NDDI-E': {'cutoff': 15, 'action': 'Evaluate for major depression in epilepsy context; optimize AED choice'},
    }

    flagged_patients = []
    total_alerts = 0
    severity_counts = Counter()

    for pid in sorted(all_pids):
        pt = patients.get(pid, {})
        pt_meds = meds.get(pid, [])
        seizures = seizure_counts.get(pid, 0)
        alerts = []

        # PHQ-9 check
        phq9_score = phq9_by_pt[pid]['score'] if pid in phq9_by_pt and phq9_by_pt[pid]['score'] is not None else None
        if phq9_score is not None and phq9_score >= 10:
            level = _classify_phq9(phq9_score)
            alerts.append({
                'instrument': 'PHQ-9',
                'score': phq9_score,
                'threshold': 10,
                'severity': level,
                'action': THRESHOLDS['PHQ-9']['action'],
            })

        # GAD-7 check
        gad7_score = gad7_by_pt[pid]['score'] if pid in gad7_by_pt and gad7_by_pt[pid]['score'] is not None else None
        if gad7_score is not None and gad7_score >= 10:
            level = _classify_gad7(gad7_score)
            alerts.append({
                'instrument': 'GAD-7',
                'score': gad7_score,
                'threshold': 10,
                'severity': level,
                'action': THRESHOLDS['GAD-7']['action'],
            })

        # C-SSRS check
        cssrs_score = cssrs_by_pt[pid]['score'] if pid in cssrs_by_pt and cssrs_by_pt[pid]['score'] is not None else None
        if cssrs_score is not None and cssrs_score > 0:
            cssrs_level = (cssrs_by_pt[pid].get('level') or 'low').lower()
            risk = CSSRS_RISK.get(cssrs_level, 'Low Risk')
            alerts.append({
                'instrument': 'C-SSRS',
                'score': cssrs_score,
                'threshold': 1,
                'severity': risk,
                'action': THRESHOLDS['C-SSRS']['action'],
            })

        # NDDI-E check
        nddie_score = nddie_by_pt[pid]['score'] if pid in nddie_by_pt and nddie_by_pt[pid]['score'] is not None else None
        if nddie_score is not None and nddie_score >= 15:
            alerts.append({
                'instrument': 'NDDI-E',
                'score': nddie_score,
                'threshold': 15,
                'severity': 'Elevated',
                'action': THRESHOLDS['NDDI-E']['action'],
            })

        if not alerts:
            continue

        # Priority: C-SSRS positive = critical, 3+ flags = high, 2 = moderate, 1 = standard
        has_cssrs = any(a['instrument'] == 'C-SSRS' for a in alerts)
        if has_cssrs:
            priority = 'critical'
        elif len(alerts) >= 3:
            priority = 'high'
        elif len(alerts) >= 2:
            priority = 'moderate'
        else:
            priority = 'standard'

        severity_counts[priority] += 1
        total_alerts += len(alerts)

        # AED risk context
        risky_aeds = []
        for drug in pt_meds:
            info = AED_PSYCHIATRIC_EFFECTS.get(drug)
            if info and info['severity'] in ('moderate', 'high'):
                risky_aeds.append({'drug': drug, 'severity': info['severity'],
                                   'effects': info['effects']})

        flagged_patients.append({
            'patient_id': pid,
            'name': pt.get('name', ''),
            'age': pt.get('age'),
            'gender': pt.get('gender', ''),
            'disease': pt.get('disease', ''),
            'priority': priority,
            'alert_count': len(alerts),
            'alerts': alerts,
            'seizure_count': seizures,
            'medications': pt_meds,
            'risky_aeds': risky_aeds,
        })

    # Sort: critical first, then high, moderate, standard
    priority_order = {'critical': 0, 'high': 1, 'moderate': 2, 'standard': 3}
    flagged_patients.sort(key=lambda p: (priority_order.get(p['priority'], 9), -p['alert_count']))

    # Per-instrument summary
    instrument_summary = []
    for inst, info in THRESHOLDS.items():
        flagged_for_inst = sum(1 for p in flagged_patients
                               if any(a['instrument'] == inst for a in p['alerts']))
        instrument_summary.append({
            'instrument': inst,
            'cutoff': info['cutoff'],
            'flagged_count': flagged_for_inst,
            'action': info['action'],
        })

    return {
        'total_flagged': len(flagged_patients),
        'total_alerts': total_alerts,
        'severity_distribution': [
            {'priority': p, 'count': severity_counts.get(p, 0)}
            for p in ['critical', 'high', 'moderate', 'standard']
        ],
        'instrument_summary': instrument_summary,
        'flagged_patients': flagged_patients,
        'thresholds': {k: v['cutoff'] for k, v in THRESHOLDS.items()},
    }


def definitions():
    """Psychiatry definitions, quality metrics, compliance, remediation."""
    return {
        'concepts': [
            {
                'name': 'PHQ-9 (Patient Health Questionnaire-9)',
                'description': 'Nine-item depression screening tool. Each item scored 0-3 '
                               '(not at all to nearly every day). Total 0-27. Cutoffs: '
                               '5 mild, 10 moderate, 15 moderately severe, 20 severe. '
                               'Item 9 screens for suicidal ideation.',
            },
            {
                'name': 'GAD-7 (Generalized Anxiety Disorder-7)',
                'description': 'Seven-item anxiety screening tool. Each item scored 0-3. '
                               'Total 0-21. Cutoffs: 5 mild, 10 moderate, 15 severe. '
                               'Validated in epilepsy populations. Sensitivity 89%, '
                               'specificity 82% for GAD at cutoff of 10.',
            },
            {
                'name': 'C-SSRS (Columbia Suicide Severity Rating Scale)',
                'description': 'Structured interview for suicidal ideation and behavior. '
                               'Screens ideation (wish to be dead, active ideation with/without '
                               'plan/intent) and behavior (preparatory, aborted, interrupted, '
                               'actual attempts). FDA-recommended for clinical trials.',
            },
            {
                'name': 'NDDI-E (Neurological Disorders Depression Inventory for Epilepsy)',
                'description': 'Six-item epilepsy-specific depression screen. Avoids somatic '
                               'items that overlap with AED side effects (fatigue, cognitive '
                               'slowing). Score >= 15 suggests major depressive episode. '
                               'Validated specifically for people with epilepsy.',
            },
            {
                'name': 'PNES (Psychogenic Non-Epileptic Seizures)',
                'description': 'Seizure-like events not caused by abnormal electrical activity. '
                               'Diagnosed by video-EEG monitoring showing no ictal correlate. '
                               'Associated with psychological trauma, conversion disorder, '
                               'dissociative disorders. Prevalence: 20-30% of epilepsy '
                               'monitoring unit admissions.',
            },
            {
                'name': 'Psychiatric Comorbidity in Epilepsy',
                'description': 'Depression (23-35%), anxiety (11-25%), psychosis (2-7%), '
                               'ADHD (12-17%) are significantly elevated in epilepsy vs. '
                               'general population. Bidirectional relationship: psychiatric '
                               'disorders increase seizure risk and vice versa.',
            },
            {
                'name': 'AED Psychiatric Effects',
                'description': 'Antiepileptic drugs can cause or worsen psychiatric symptoms. '
                               'Levetiracetam: irritability/aggression (10-15%). '
                               'Phenobarbital: depression/suicidality. Topiramate: depression, '
                               'cognitive effects. Lamotrigine: mood-stabilizing (beneficial). '
                               'FDA black-box warning: all AEDs carry suicidality risk.',
            },
            {
                'name': 'Interictal Dysphoric Disorder',
                'description': 'Unique mood disorder in epilepsy with pleomorphic symptoms: '
                               'depressive mood, irritability, euphoria, anxiety, anergia, '
                               'insomnia, atypical pain. Intermittent course linked to seizure '
                               'timing. May not meet criteria for standard DSM categories.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Screening Coverage',
                'description': 'Proportion of epilepsy patients screened with validated '
                               'psychiatric instruments (PHQ-9, GAD-7). ILAE recommends '
                               'routine screening at diagnosis and annually.',
            },
            {
                'name': 'Treatment Response Monitoring',
                'description': 'Serial PHQ-9/GAD-7 scores to track psychiatric treatment '
                               'response. Clinically significant change: PHQ-9 drop >= 5 '
                               'points or >= 50% reduction. Remission: PHQ-9 < 5.',
            },
            {
                'name': 'Safety Monitoring',
                'description': 'C-SSRS administered at AED initiation/change and during '
                               'follow-up. PHQ-9 item 9 checked at every visit. Positive '
                               'screens require same-day safety assessment.',
            },
            {
                'name': 'PNES Diagnostic Accuracy',
                'description': 'Gold standard: video-EEG capture of habitual event with no '
                               'ictal correlate. Serum prolactin (drawn <20 min post-event) '
                               'elevated in GTCS but not PNES. Average diagnostic delay: '
                               '7-10 years. Misdiagnosis rate: up to 25%.',
            },
        ],
        'compliance': [
            {
                'name': 'APA Practice Guidelines',
                'description': 'American Psychiatric Association guidelines for assessment '
                               'and treatment of depression, anxiety, and suicidality. '
                               'Standardized instruments recommended for screening and monitoring.',
            },
            {
                'name': 'ILAE Psychiatric Screening',
                'description': 'International League Against Epilepsy recommends systematic '
                               'psychiatric screening in all epilepsy patients using validated '
                               'instruments. Annual rescreening recommended.',
            },
            {
                'name': 'FDA AED Suicidality Warning',
                'description': 'FDA class-wide warning (2008): all AEDs associated with '
                               'increased suicidality risk (OR 1.8). Monitor psychiatric '
                               'symptoms at AED initiation and dose changes.',
            },
            {
                'name': 'HIPAA / Mental Health',
                'description': 'Psychiatric assessment data requires heightened privacy '
                               'protections. 42 CFR Part 2 for substance use records. '
                               'State laws may impose stricter mental health data protections.',
            },
        ],
        'remediation_strategies': [
            {
                'name': 'AED Optimization for Psychiatric Comorbidity',
                'description': 'Avoid psychiatrically adverse AEDs (levetiracetam, phenobarbital, '
                               'topiramate) in patients with depression/anxiety history. Prefer '
                               'lamotrigine (mood-stabilizing), pregabalin/gabapentin (anxiolytic). '
                               'Consider brivaracetam over levetiracetam for less behavioral effects.',
            },
            {
                'name': 'Integrated Treatment',
                'description': 'SSRIs (sertraline, citalopram) are safe and effective in epilepsy. '
                               'Seizure threshold not significantly lowered at therapeutic doses. '
                               'CBT has Level A evidence for depression in epilepsy. Combine with '
                               'psychiatric follow-up every 4-6 weeks during titration.',
            },
            {
                'name': 'PNES Management',
                'description': 'After diagnosis communication: validate symptoms, explain mechanism '
                               '(functional neurological disorder). CBT is first-line treatment '
                               '(NES Treatment Trial, Level B). Gradually taper unnecessary AEDs. '
                               'Coordinate with neurology. Avoid reinforcing sick role.',
            },
            {
                'name': 'Crisis Intervention',
                'description': 'C-SSRS positive screens: same-day psychiatric assessment, safety '
                               'planning (Stanley-Brown), means restriction counseling. PHQ-9 '
                               'item 9 > 0: assess acuity, document safety plan. Hospitalize '
                               'if imminent risk. Coordinate with emergency services.',
            },
        ],
    }
