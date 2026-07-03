"""Patient-Reported Outcomes (PRO) Dashboard — Patient Module Section 6.
Tracks validated PRO instruments for Sleep, Mood, Cognition, and Quality of
Life across epilepsy patients: PSQI, ESS, PHQ-9, GAD-7, QOLIE-31, MoCA,
NDDI-E, WPAI, plus subjective ratings (fatigue, mood, memory, concentration,
social function, daily function, seizure worry).

Populates and reads from:
  - pro_outcomes  (monthly patient-reported outcome assessments)

Uses real patient_ids from the patients table (first 30).
Validated clinical instruments and ILAE-recommended PRO measures applied throughout.
"""

import json
import math
import os
import random
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

NOTES_POOL = [
    '', '', '', '', '',  # most entries have no notes
    'Feeling well-rested today',
    'Poor sleep due to seizure worry',
    'Noticed memory lapses this week',
    'Mood improved after exercise',
    'Side effects from medication change',
    'Difficulty concentrating at work',
    'Anxious about upcoming appointment',
    'Fatigue worse than usual',
    'Social event went well despite anxiety',
    'Missed work due to post-ictal fatigue',
    'Started mindfulness practice',
    'Seizure worry affecting daily activities',
    'Cognitive fog in the mornings',
    'Sleep disrupted by nocturnal seizure',
    'Feeling more confident this month',
    'Struggling with medication side effects',
]

random.seed(99)


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


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _generate_pro_record(patient_id, assessment_date, rng):
    """Generate a single monthly PRO assessment record."""
    psqi_score = _clamp(round(rng.gauss(8, 4)), 0, 21)
    ess_score = _clamp(round(rng.gauss(9, 4)), 0, 24)
    phq9_score = _clamp(round(rng.gauss(8, 5)), 0, 27)
    gad7_score = _clamp(round(rng.gauss(7, 4)), 0, 21)
    qolie31_score = _clamp(round(rng.gauss(62, 15)), 0, 100)
    moca_score = _clamp(round(rng.gauss(25, 4)), 0, 30)
    nddi_e_score = _clamp(round(rng.gauss(12, 4)), 6, 24)
    wpai_percent = _clamp(round(rng.gauss(35, 20)), 0, 100)
    sleep_hours = round(_clamp(rng.gauss(6.5, 1.5), 2.0, 12.0), 1)
    fatigue_level = _clamp(round(rng.gauss(5, 2)), 1, 10)
    mood_rating = _clamp(round(rng.gauss(6, 2)), 1, 10)
    memory_complaints = rng.random() < 0.30
    concentration_difficulty = rng.random() < 0.35
    social_function_rating = _clamp(round(rng.gauss(6, 2)), 1, 10)
    daily_function_rating = _clamp(round(rng.gauss(6.5, 2)), 1, 10)
    seizure_worry_score = _clamp(round(rng.gauss(5.5, 2)), 1, 10)
    notes = rng.choice(NOTES_POOL)

    return {
        'patient_id': patient_id,
        'assessment_date': assessment_date.strftime('%Y-%m-%d'),
        'psqi_score': psqi_score,
        'ess_score': ess_score,
        'phq9_score': phq9_score,
        'gad7_score': gad7_score,
        'qolie31_score': qolie31_score,
        'moca_score': moca_score,
        'nddi_e_score': nddi_e_score,
        'wpai_percent': wpai_percent,
        'sleep_hours': sleep_hours,
        'fatigue_level': fatigue_level,
        'mood_rating': mood_rating,
        'memory_complaints': memory_complaints,
        'concentration_difficulty': concentration_difficulty,
        'social_function_rating': social_function_rating,
        'daily_function_rating': daily_function_rating,
        'seizure_worry_score': seizure_worry_score,
        'notes': notes,
    }


def _populate_if_empty():
    """Populate pro_outcomes table with realistic data if empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS pro_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            fields_json TEXT,
            created_at TEXT
        )''')
        conn.commit()

        count = conn.execute('SELECT COUNT(*) FROM pro_outcomes').fetchone()[0]
        if count > 0:
            return  # already populated

        # Get real patient IDs
        patients = conn.execute(
            'SELECT patient_id, name, age, gender FROM patients'
        ).fetchall()
        patients = [dict(p) for p in patients]

        epat = [p for p in patients if p['patient_id'].startswith('EPAT')]
        others = [p for p in patients if not p['patient_id'].startswith('EPAT')]
        ordered = epat + others
        target_patients = ordered[:30]
        patient_ids = [p['patient_id'] for p in target_patients]

        rng = random.Random(99)

        # Generate ~6 assessments per patient over 6 months (~180 total)
        base_date = datetime(2025, 6, 15)

        for pid in patient_ids:
            num_assessments = 6
            for month_offset in range(num_assessments):
                # Assessment date: base_date minus month_offset months, with slight jitter
                assessment_date = base_date - timedelta(days=month_offset * 30 + rng.randint(-5, 5))
                entry = _generate_pro_record(pid, assessment_date, rng)
                conn.execute(
                    'INSERT INTO pro_outcomes (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                    (pid, json.dumps(entry))
                )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def _load_all_data():
    """Load all pro_outcomes and return structured list."""
    _populate_if_empty()

    records = []
    for r in _db_query('SELECT patient_id, fields_json FROM pro_outcomes'):
        entry = _safe_json(r['fields_json'])
        if entry:
            records.append(entry)

    return records


def _compute_correlation(records, x_fn, y_fn):
    """Compute Pearson correlation between two numeric fields.
    Returns a value in [-1, 1]."""
    n = len(records)
    if n < 3:
        return 0.0

    xs = [x_fn(r) for r in records]
    ys = [y_fn(r) for r in records]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n) if n > 0 else 0
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n) if n > 0 else 0

    if std_x == 0 or std_y == 0:
        return 0.0

    return round(cov / (std_x * std_y), 3)


def overview():
    """Return KPI cards + chart data for the PRO Outcomes overview tab."""
    records = _load_all_data()

    total_assessments = len(records)
    patient_ids = list(set(r.get('patient_id', '') for r in records))
    total_patients = len(patient_ids)

    avg_psqi = _avg([r.get('psqi_score', 0) for r in records])
    avg_ess = _avg([r.get('ess_score', 0) for r in records])
    avg_phq9 = _avg([r.get('phq9_score', 0) for r in records])
    avg_gad7 = _avg([r.get('gad7_score', 0) for r in records])
    avg_qolie31 = _avg([r.get('qolie31_score', 0) for r in records])
    avg_moca = _avg([r.get('moca_score', 0) for r in records])
    avg_nddi_e = _avg([r.get('nddi_e_score', 0) for r in records])
    avg_wpai = _avg([r.get('wpai_percent', 0) for r in records])

    # Sleep quality distribution based on PSQI cutoffs
    sleep_quality_distribution = []
    psqi_good = sum(1 for r in records if r.get('psqi_score', 0) <= 5)
    psqi_fair = sum(1 for r in records if 5 < r.get('psqi_score', 0) <= 10)
    psqi_poor = sum(1 for r in records if r.get('psqi_score', 0) > 10)
    sleep_quality_distribution = [
        {'category': 'Good', 'count': psqi_good},
        {'category': 'Fair', 'count': psqi_fair},
        {'category': 'Poor', 'count': psqi_poor},
    ]

    # Depression severity based on PHQ-9 cutoffs
    dep_minimal = sum(1 for r in records if r.get('phq9_score', 0) <= 4)
    dep_mild = sum(1 for r in records if 5 <= r.get('phq9_score', 0) <= 9)
    dep_moderate = sum(1 for r in records if 10 <= r.get('phq9_score', 0) <= 14)
    dep_mod_severe = sum(1 for r in records if 15 <= r.get('phq9_score', 0) <= 19)
    dep_severe = sum(1 for r in records if r.get('phq9_score', 0) >= 20)
    depression_severity = [
        {'severity': 'Minimal', 'count': dep_minimal},
        {'severity': 'Mild', 'count': dep_mild},
        {'severity': 'Moderate', 'count': dep_moderate},
        {'severity': 'Mod-Severe', 'count': dep_mod_severe},
        {'severity': 'Severe', 'count': dep_severe},
    ]

    # Anxiety severity based on GAD-7 cutoffs
    anx_minimal = sum(1 for r in records if r.get('gad7_score', 0) <= 4)
    anx_mild = sum(1 for r in records if 5 <= r.get('gad7_score', 0) <= 9)
    anx_moderate = sum(1 for r in records if 10 <= r.get('gad7_score', 0) <= 14)
    anx_severe = sum(1 for r in records if r.get('gad7_score', 0) >= 15)
    anxiety_severity = [
        {'severity': 'Minimal', 'count': anx_minimal},
        {'severity': 'Mild', 'count': anx_mild},
        {'severity': 'Moderate', 'count': anx_moderate},
        {'severity': 'Severe', 'count': anx_severe},
    ]

    # Cognitive status based on MoCA cutoffs
    cog_normal = sum(1 for r in records if r.get('moca_score', 0) >= 26)
    cog_mild = sum(1 for r in records if 22 <= r.get('moca_score', 0) <= 25)
    cog_moderate = sum(1 for r in records if r.get('moca_score', 0) < 22)
    cognitive_status = [
        {'status': 'Normal', 'count': cog_normal},
        {'status': 'Mild Impairment', 'count': cog_mild},
        {'status': 'Moderate Impairment', 'count': cog_moderate},
    ]

    # QoL trend: monthly average QOLIE-31 over 6 months
    month_scores = {}
    for r in records:
        date_str = r.get('assessment_date', '')
        if date_str:
            month_key = date_str[:7]  # YYYY-MM
            month_scores.setdefault(month_key, []).append(r.get('qolie31_score', 0))

    qol_trend = sorted([
        {'month': m, 'avg_score': _avg(scores)}
        for m, scores in month_scores.items()
    ], key=lambda x: x['month'])

    # Mood vs seizure worry correlation
    mood_vs_seizure_worry = _compute_correlation(
        records,
        lambda r: r.get('mood_rating', 5),
        lambda r: r.get('seizure_worry_score', 5),
    )

    # Domain averages (normalized 0-100)
    # Sleep: invert PSQI (0-21, lower=better) -> (21-score)/21*100
    sleep_norm = _avg([(21 - r.get('psqi_score', 0)) / 21 * 100 for r in records])
    # Mood: invert PHQ-9 (0-27, lower=better) -> (27-score)/27*100
    mood_norm = _avg([(27 - r.get('phq9_score', 0)) / 27 * 100 for r in records])
    # Cognition: MoCA (0-30, higher=better) -> score/30*100
    cognition_norm = _avg([r.get('moca_score', 0) / 30 * 100 for r in records])
    # QoL: QOLIE-31 already 0-100
    qol_norm = _avg([r.get('qolie31_score', 0) for r in records])
    # Function: avg of daily_function_rating and social_function_rating (1-10) -> /10*100
    function_norm = _avg([
        (r.get('daily_function_rating', 5) + r.get('social_function_rating', 5)) / 2 / 10 * 100
        for r in records
    ])

    domain_averages = [
        {'domain': 'Sleep', 'score': round(sleep_norm, 1)},
        {'domain': 'Mood', 'score': round(mood_norm, 1)},
        {'domain': 'Cognition', 'score': round(cognition_norm, 1)},
        {'domain': 'QoL', 'score': round(qol_norm, 1)},
        {'domain': 'Function', 'score': round(function_norm, 1)},
    ]

    return {
        'available': True,
        'total_assessments': total_assessments,
        'total_patients': total_patients,
        'avg_psqi': avg_psqi,
        'avg_ess': avg_ess,
        'avg_phq9': avg_phq9,
        'avg_gad7': avg_gad7,
        'avg_qolie31': avg_qolie31,
        'avg_moca': avg_moca,
        'avg_nddi_e': avg_nddi_e,
        'avg_wpai': avg_wpai,
        'sleep_quality_distribution': sleep_quality_distribution,
        'depression_severity': depression_severity,
        'anxiety_severity': anxiety_severity,
        'cognitive_status': cognitive_status,
        'qol_trend': qol_trend,
        'mood_vs_seizure_worry': mood_vs_seizure_worry,
        'domain_averages': domain_averages,
    }


def _trend_direction(values):
    """Return 'improving', 'worsening', or 'stable' based on simple linear trend."""
    if len(values) < 2:
        return 'stable'
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 'stable'
    slope = num / den
    if abs(slope) < 0.3:
        return 'stable'
    return 'improving' if slope < 0 else 'worsening'


def _trend_direction_higher_better(values):
    """Trend for scales where higher = better (e.g., QOLIE-31)."""
    if len(values) < 2:
        return 'stable'
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 'stable'
    slope = num / den
    if abs(slope) < 0.3:
        return 'stable'
    return 'improving' if slope > 0 else 'worsening'


def breakdown():
    """Return per-patient summary and all records for the PRO Outcomes dashboard."""
    records = _load_all_data()

    # Group records by patient
    patient_records_map = {}
    for r in records:
        pid = r.get('patient_id', '')
        patient_records_map.setdefault(pid, []).append(r)

    patients = []
    for pid in sorted(patient_records_map.keys()):
        precs = patient_records_map[pid]
        total = len(precs)

        # Sort by assessment_date for trend analysis
        sorted_recs = sorted(precs, key=lambda x: x.get('assessment_date', ''))

        # Latest scores
        latest = sorted_recs[-1] if sorted_recs else {}

        # Trend direction for PHQ-9 (lower=better)
        phq9_values = [r.get('phq9_score', 0) for r in sorted_recs]
        phq9_trend = _trend_direction(phq9_values)

        # Trend direction for QOLIE-31 (higher=better)
        qolie_values = [r.get('qolie31_score', 0) for r in sorted_recs]
        qolie_trend = _trend_direction_higher_better(qolie_values)

        # Recent assessments (last 3 by date)
        recent = sorted(precs, key=lambda x: x.get('assessment_date', ''), reverse=True)[:3]

        patients.append({
            'patient_id': pid,
            'total_assessments': total,
            'latest_psqi': latest.get('psqi_score'),
            'latest_ess': latest.get('ess_score'),
            'latest_phq9': latest.get('phq9_score'),
            'latest_gad7': latest.get('gad7_score'),
            'latest_qolie31': latest.get('qolie31_score'),
            'latest_moca': latest.get('moca_score'),
            'latest_nddi_e': latest.get('nddi_e_score'),
            'latest_wpai': latest.get('wpai_percent'),
            'phq9_trend': phq9_trend,
            'qolie31_trend': qolie_trend,
            'recent_assessments': recent,
        })

    # All assessments sorted by date desc
    all_assessments = sorted(records, key=lambda x: x.get('assessment_date', ''), reverse=True)

    return {
        'patients': patients,
        'all_assessments': all_assessments,
    }


def definitions():
    """Return clinical PRO instrument definitions and terminology."""
    return {
        'concepts': [
            {
                'name': 'PSQI (Pittsburgh Sleep Quality Index)',
                'description': 'A 19-item self-report questionnaire that assesses sleep quality and disturbances over a one-month period. The global score ranges from 0 to 21, where lower scores indicate better sleep quality. A score of 5 or less is classified as good sleep quality, 6-10 as fair, and above 10 as poor. The PSQI evaluates seven domains: subjective sleep quality, sleep latency, sleep duration, habitual sleep efficiency, sleep disturbances, use of sleep medication, and daytime dysfunction. Sleep disorders are highly prevalent in epilepsy (30-50% of patients) and poor sleep quality is a recognized seizure trigger.',
            },
            {
                'name': 'ESS (Epworth Sleepiness Scale)',
                'description': 'An 8-item questionnaire measuring the general level of daytime sleepiness by asking subjects to rate their likelihood of dozing in eight common situations. Scores range from 0 to 24: below 10 is normal, 10-15 indicates moderate excessive daytime sleepiness, and above 15 indicates severe sleepiness. In epilepsy populations, elevated ESS scores may reflect antiepileptic drug side effects, nocturnal seizures disrupting sleep architecture, or comorbid sleep disorders such as obstructive sleep apnea, which has a higher prevalence in people with epilepsy.',
            },
            {
                'name': 'PHQ-9 (Patient Health Questionnaire-9)',
                'description': 'A 9-item depression screening instrument based on DSM criteria, scoring 0-27. Severity thresholds: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe depression. Depression is the most common psychiatric comorbidity in epilepsy, affecting 20-55% of patients. The PHQ-9 is validated for use in neurological populations and is recommended by the ILAE for routine screening. Depression in epilepsy is bidirectional — it increases seizure frequency and reduces medication adherence, while seizures and AED side effects can worsen depressive symptoms.',
            },
            {
                'name': 'GAD-7 (Generalized Anxiety Disorder-7)',
                'description': 'A 7-item anxiety screening tool scoring 0-21, with severity bands: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe anxiety. Anxiety disorders affect 10-25% of people with epilepsy, often co-occurring with depression. Seizure-related anxiety includes ictal anxiety (anxiety as a seizure manifestation), anticipatory anxiety (fear of future seizures), and phobic avoidance behaviors. The GAD-7 captures generalized anxiety and is recommended alongside the PHQ-9 for comprehensive mental health screening in epilepsy clinics.',
            },
            {
                'name': 'QOLIE-31 (Quality of Life in Epilepsy-31)',
                'description': 'A 31-item epilepsy-specific quality of life measure scoring 0-100, where higher scores indicate better QoL. It covers seven domains: seizure worry, overall QoL, emotional well-being, energy/fatigue, cognitive function, medication effects, and social function. The QOLIE-31 is the most widely used epilepsy-specific QoL instrument internationally and is recommended by the ILAE for clinical trials and routine care. It captures the unique burden of living with epilepsy beyond seizure frequency, including driving restrictions, employment limitations, and stigma.',
            },
            {
                'name': 'MoCA (Montreal Cognitive Assessment)',
                'description': 'A 30-point screening tool for mild cognitive impairment assessing visuospatial/executive function, naming, memory, attention, language, abstraction, delayed recall, and orientation. Scores of 26 or above are considered normal, 22-25 indicate mild cognitive impairment, and below 22 suggests moderate impairment. Cognitive dysfunction is common in epilepsy due to seizure-related neuronal injury, interictal epileptiform discharges, and AED cognitive side effects (particularly topiramate, zonisamide, and phenobarbital). Serial MoCA assessments help track cognitive trajectories over time.',
            },
            {
                'name': 'NDDI-E (Neurological Disorders Depression Inventory for Epilepsy)',
                'description': 'A 6-item screening tool specifically designed and validated for detecting major depressive disorder in people with epilepsy. Scores range from 6 to 24, with a score above 15 indicating major depression risk. Unlike the PHQ-9, the NDDI-E was developed to distinguish depression symptoms from AED side effects and seizure-related symptoms, reducing false positives. It is the only depression screening tool specifically validated in epilepsy populations and is recommended by the AAN/AES for rapid clinical screening.',
            },
            {
                'name': 'WPAI (Work Productivity and Activity Impairment)',
                'description': 'A 6-item questionnaire that measures the impact of health problems on work productivity and regular daily activities over the past seven days. The overall work impairment score ranges from 0% to 100%, where higher percentages indicate greater impairment. In epilepsy, work productivity is affected by seizure unpredictability, post-ictal recovery periods, AED side effects (sedation, cognitive slowing), driving restrictions limiting employment options, and workplace discrimination. WPAI data helps quantify the socioeconomic burden of epilepsy beyond clinical outcomes.',
            },
            {
                'name': 'Patient-Reported Outcome (PRO)',
                'description': 'A measurement of any aspect of a patient\'s health status that comes directly from the patient, without interpretation by clinicians or others. PROs capture the patient\'s subjective experience of their disease and treatment. In epilepsy care, PROs are essential because seizure frequency alone does not capture the full disease burden — medication side effects, cognitive impairment, mood disorders, social limitations, and stigma significantly affect patients\' lives. The FDA and EMA increasingly require PRO data in clinical trials, and the ILAE recommends systematic PRO collection in routine epilepsy care.',
            },
            {
                'name': 'Seizure-Related Quality of Life',
                'description': 'The multidimensional impact of epilepsy on a patient\'s well-being beyond seizure control. Key domains include: seizure worry (anticipatory anxiety about when/where seizures will occur), social function (relationships, social participation, perceived stigma), daily function (self-care, household tasks, driving), vocational function (employment, education, career limitations), and emotional well-being (depression, anxiety, self-esteem). Research consistently shows that seizure worry and perceived stigma have a greater impact on QoL than seizure frequency alone, underscoring the importance of PRO measurement.',
            },
            {
                'name': 'Cognitive Comorbidity',
                'description': 'Cognitive dysfunction that co-occurs with epilepsy, affecting 20-50% of patients across the lifespan. Contributing factors include: the underlying brain pathology causing epilepsy, cumulative effects of recurrent seizures (particularly status epilepticus), interictal epileptiform discharges disrupting cognitive processing, AED-related cognitive side effects, and comorbid mood disorders affecting attention and executive function. Common domains affected include memory (most frequently reported), attention, processing speed, executive function, and language. Serial cognitive screening with instruments like the MoCA enables early detection of decline and guides AED selection toward cognitively favorable agents.',
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN (first 3 patients) ===')
    bd = breakdown()
    for p in bd['patients'][:3]:
        pprint.pprint({k: v for k, v in p.items() if k != 'recent_assessments'})
        print(f'  recent_assessments: {len(p["recent_assessments"])} entries')
    print(f'\nTotal patients: {len(bd["patients"])}')
    print(f'Total assessments: {len(bd["all_assessments"])}')
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Concepts: {len(df["concepts"])}')
    for c in df['concepts']:
        print(f'  {c["name"]}')
