"""Trigger Tracking & Forecasting Dashboard — Patient Module Section 4.
Tracks patient-reported seizure triggers, daily factor logging (sleep, stress,
medication adherence, etc.), and trigger-seizure correlation analysis.

Populates and reads from:
  - trigger_logs  (daily patient factor logs with seizure occurrence tracking)

Uses real patient_ids from the patients table (first 30).
ILAE Trigger Classification, seizure diary best practices applied throughout.
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

TRIGGER_CATEGORIES = [
    'Sleep deprivation',
    'Stress',
    'Medication missed',
    'Illness',
    'Photosensitivity',
    'Alcohol',
    'Hormonal changes',
    'Weather change',
    'Fatigue',
    'Unknown',
]

MENSTRUAL_PHASES = ['follicular', 'luteal', 'ovulation', 'none', 'n_a']
ILLNESS_OPTIONS = ['none', 'cold', 'flu', 'fever', 'infection']
PHOTO_EXPOSURE_OPTIONS = ['none', 'low', 'moderate', 'high']
WEATHER_CHANGE_OPTIONS = ['none', 'mild', 'moderate', 'severe']
MED_ADHERENCE_OPTIONS = ['yes', 'no', 'partial']

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


def _generate_trigger_log(patient_id, log_date, rng, force_seizure=False):
    """Generate a single daily trigger log entry."""
    # Decide seizure occurrence first (or force it)
    seizure_occurred = force_seizure or (rng.random() < 0.17)

    if seizure_occurred:
        # Seizure days: correlate with risk factors
        sleep_hours = round(rng.uniform(3.0, 6.5), 1)
        sleep_quality = rng.randint(1, 3)
        stress_level = rng.randint(6, 10)
        med_adherence = rng.choices(MED_ADHERENCE_OPTIONS, weights=[0.25, 0.45, 0.30])[0]
        missed_doses = rng.choices([0, 1, 2, 3], weights=[0.15, 0.35, 0.35, 0.15])[0]
        fatigue_level = rng.randint(6, 10)
        mood_score = rng.randint(1, 5)
        illness = rng.choices(ILLNESS_OPTIONS, weights=[0.40, 0.20, 0.15, 0.15, 0.10])[0]
        photo_exposure = rng.choices(PHOTO_EXPOSURE_OPTIONS, weights=[0.30, 0.20, 0.25, 0.25])[0]
        caffeine_cups = rng.randint(2, 6)
        alcohol_units = rng.choices([0, 1, 2, 3, 4], weights=[0.30, 0.20, 0.20, 0.15, 0.15])[0]
        weather_change = rng.choices(WEATHER_CHANGE_OPTIONS, weights=[0.20, 0.25, 0.30, 0.25])[0]
        seizure_count = rng.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]

        # Determine the primary trigger
        trigger_weights = []
        if sleep_hours < 5:
            trigger_weights.append(('Sleep deprivation', 3))
        if stress_level >= 8:
            trigger_weights.append(('Stress', 3))
        if med_adherence != 'yes':
            trigger_weights.append(('Medication missed', 3))
        if illness != 'none':
            trigger_weights.append(('Illness', 2))
        if photo_exposure in ('moderate', 'high'):
            trigger_weights.append(('Photosensitivity', 2))
        if alcohol_units >= 2:
            trigger_weights.append(('Alcohol', 2))
        if weather_change in ('moderate', 'severe'):
            trigger_weights.append(('Weather change', 1))
        if fatigue_level >= 8:
            trigger_weights.append(('Fatigue', 2))

        if trigger_weights:
            triggers_pool, weights = zip(*trigger_weights)
            primary_trigger = rng.choices(list(triggers_pool), weights=list(weights))[0]
        else:
            primary_trigger = rng.choice(TRIGGER_CATEGORIES)
    else:
        # Non-seizure days: generally healthier values
        sleep_hours = round(rng.uniform(6.0, 9.5), 1)
        sleep_quality = rng.randint(3, 5)
        stress_level = rng.randint(1, 6)
        med_adherence = rng.choices(MED_ADHERENCE_OPTIONS, weights=[0.80, 0.05, 0.15])[0]
        missed_doses = rng.choices([0, 1, 2, 3], weights=[0.75, 0.20, 0.04, 0.01])[0]
        fatigue_level = rng.randint(1, 5)
        mood_score = rng.randint(5, 10)
        illness = rng.choices(ILLNESS_OPTIONS, weights=[0.85, 0.05, 0.04, 0.03, 0.03])[0]
        photo_exposure = rng.choices(PHOTO_EXPOSURE_OPTIONS, weights=[0.50, 0.30, 0.15, 0.05])[0]
        caffeine_cups = rng.randint(0, 3)
        alcohol_units = rng.choices([0, 1, 2, 3, 4], weights=[0.55, 0.25, 0.15, 0.04, 0.01])[0]
        weather_change = rng.choices(WEATHER_CHANGE_OPTIONS, weights=[0.55, 0.25, 0.15, 0.05])[0]
        seizure_count = 0
        primary_trigger = None

    exercise_minutes = rng.randint(0, 60)

    # Menstrual phase — roughly half patients are female
    menstrual_phase = rng.choices(
        MENSTRUAL_PHASES, weights=[0.15, 0.15, 0.08, 0.12, 0.50]
    )[0]

    # Notes
    notes_pool = [
        '', '', '', '',  # most entries have no notes
        'Felt aura before event',
        'Woke up feeling groggy',
        'Stressful day at work',
        'Forgot evening dose',
        'Bright flickering lights at store',
        'Coming down with a cold',
        'Menstrual cramps today',
        'Thunderstorm overnight',
        'Very tired after poor sleep',
        'Drank coffee late at night',
        'Had wine at dinner',
        'Exercised intensely in morning',
    ]
    notes = rng.choice(notes_pool)

    return {
        'patient_id': patient_id,
        'log_date': log_date.strftime('%Y-%m-%d'),
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality,
        'stress_level': stress_level,
        'medication_adherence': med_adherence,
        'missed_doses': missed_doses,
        'exercise_minutes': exercise_minutes,
        'caffeine_cups': caffeine_cups,
        'alcohol_units': alcohol_units,
        'menstrual_phase': menstrual_phase,
        'illness': illness,
        'photo_exposure': photo_exposure,
        'weather_change': weather_change,
        'mood_score': mood_score,
        'fatigue_level': fatigue_level,
        'seizure_occurred': seizure_occurred,
        'seizure_count': seizure_count,
        'primary_trigger': primary_trigger,
        'notes': notes,
    }


def _populate_if_empty():
    """Populate trigger_logs table with realistic data if empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS trigger_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            fields_json TEXT,
            created_at TEXT
        )''')
        conn.commit()

        count = conn.execute('SELECT COUNT(*) FROM trigger_logs').fetchone()[0]
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

        # Generate 300 trigger_logs across 30 patients (~10 per patient)
        # spanning the past 90 days
        base_date = datetime(2025, 6, 15)
        records_generated = 0

        for pid in patient_ids:
            # Each patient gets ~10 log entries spread over 90 days
            num_logs = rng.randint(8, 12)
            # Pick random days within the past 90 days (no duplicates per patient)
            available_days = list(range(1, 91))
            rng.shuffle(available_days)
            log_days = sorted(available_days[:num_logs])

            for day_offset in log_days:
                log_date = base_date - timedelta(days=day_offset)
                entry = _generate_trigger_log(pid, log_date, rng)
                conn.execute(
                    'INSERT INTO trigger_logs (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                    (pid, json.dumps(entry))
                )
                records_generated += 1

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def _load_all_data():
    """Load all trigger_logs and return structured list."""
    _populate_if_empty()

    logs = []
    for r in _db_query('SELECT patient_id, fields_json FROM trigger_logs'):
        entry = _safe_json(r['fields_json'])
        if entry:
            logs.append(entry)

    return logs


def _compute_correlation(logs, factor_fn, seizure_fn):
    """Compute a simple point-biserial-style correlation between a
    continuous factor and binary seizure occurrence.
    Returns a value in [-1, 1]."""
    n = len(logs)
    if n < 3:
        return 0.0

    xs = [factor_fn(l) for l in logs]
    ys = [seizure_fn(l) for l in logs]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n) if n > 0 else 0
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n) if n > 0 else 0

    if std_x == 0 or std_y == 0:
        return 0.0

    return round(cov / (std_x * std_y), 3)


def _compute_risk_level(patient_logs):
    """Compute risk level for a patient based on their logs."""
    if not patient_logs:
        return 'low'

    seizure_rate = sum(1 for l in patient_logs if l.get('seizure_occurred')) / len(patient_logs)
    avg_stress = _avg([l.get('stress_level', 5) for l in patient_logs])
    avg_sleep = _avg([l.get('sleep_hours', 7) for l in patient_logs])
    adherence_rate = sum(1 for l in patient_logs if l.get('medication_adherence') == 'yes') / len(patient_logs)

    # Composite score: higher = more risk
    score = 0
    score += seizure_rate * 40
    score += (avg_stress / 10) * 20
    score += max(0, (7 - avg_sleep) / 7) * 20
    score += (1 - adherence_rate) * 20

    if score >= 60:
        return 'critical'
    elif score >= 40:
        return 'high'
    elif score >= 20:
        return 'moderate'
    else:
        return 'low'


def overview():
    """Return KPI cards + chart data for the Trigger Tracking overview tab."""
    logs = _load_all_data()

    total_logs = len(logs)
    patient_ids = list(set(l.get('patient_id', '') for l in logs))
    total_patients = len(patient_ids)

    seizure_logs = [l for l in logs if l.get('seizure_occurred')]
    seizure_count = len(seizure_logs)
    avg_seizure_rate = round(seizure_count / total_logs * 100, 1) if total_logs else 0

    avg_sleep_hours = _avg([l.get('sleep_hours', 0) for l in logs])
    avg_stress_level = _avg([l.get('stress_level', 0) for l in logs])

    adherent_count = sum(1 for l in logs if l.get('medication_adherence') == 'yes')
    medication_adherence_rate = round(adherent_count / total_logs * 100, 1) if total_logs else 0

    # Unique dates tracked
    all_dates = set(l.get('log_date', '') for l in logs if l.get('log_date'))
    days_tracked = len(all_dates)

    # Top trigger: most common primary_trigger when seizure occurs
    trigger_counts = Counter(
        l.get('primary_trigger', 'Unknown')
        for l in seizure_logs
        if l.get('primary_trigger')
    )
    top_trigger = trigger_counts.most_common(1)[0][0] if trigger_counts else 'Unknown'

    # trigger_distribution: count of each trigger category when seizures occur
    trigger_distribution = []
    for trig in TRIGGER_CATEGORIES:
        cnt = trigger_counts.get(trig, 0)
        trigger_distribution.append({'trigger': trig, 'count': cnt})

    # temporal_trend: daily seizure count over the past 90 days
    base_date = datetime(2025, 6, 15)
    temporal_trend = []
    for day_offset in range(90, 0, -1):
        d = (base_date - timedelta(days=day_offset)).strftime('%Y-%m-%d')
        day_seizures = sum(
            l.get('seizure_count', 0) for l in logs
            if l.get('log_date') == d and l.get('seizure_occurred')
        )
        temporal_trend.append({'date': d, 'seizure_count': day_seizures})

    # factor_correlation
    seizure_fn = lambda l: 1 if l.get('seizure_occurred') else 0
    factor_correlation = [
        {
            'factor': 'sleep_hours',
            'correlation': _compute_correlation(
                logs, lambda l: l.get('sleep_hours', 7), seizure_fn
            ),
        },
        {
            'factor': 'stress_level',
            'correlation': _compute_correlation(
                logs, lambda l: l.get('stress_level', 5), seizure_fn
            ),
        },
        {
            'factor': 'med_adherence',
            'correlation': _compute_correlation(
                logs, lambda l: 1 if l.get('medication_adherence') == 'yes' else 0, seizure_fn
            ),
        },
        {
            'factor': 'caffeine_cups',
            'correlation': _compute_correlation(
                logs, lambda l: l.get('caffeine_cups', 0), seizure_fn
            ),
        },
        {
            'factor': 'alcohol_units',
            'correlation': _compute_correlation(
                logs, lambda l: l.get('alcohol_units', 0), seizure_fn
            ),
        },
        {
            'factor': 'fatigue_level',
            'correlation': _compute_correlation(
                logs, lambda l: l.get('fatigue_level', 5), seizure_fn
            ),
        },
    ]

    # risk_level_distribution: per-patient risk
    patient_logs_map = {}
    for l in logs:
        pid = l.get('patient_id', '')
        patient_logs_map.setdefault(pid, []).append(l)

    risk_counts = Counter()
    for pid, plogs in patient_logs_map.items():
        risk = _compute_risk_level(plogs)
        risk_counts[risk] += 1

    risk_level_distribution = [
        {'risk_level': rl, 'count': risk_counts.get(rl, 0)}
        for rl in ['low', 'moderate', 'high', 'critical']
    ]

    # sleep_vs_seizure: bucketed sleep hours vs seizure rate
    sleep_buckets = {
        '<4h': (0, 4),
        '4-5h': (4, 5),
        '5-6h': (5, 6),
        '6-7h': (6, 7),
        '7-8h': (7, 8),
        '8+h': (8, 100),
    }
    sleep_vs_seizure = []
    for label, (lo, hi) in sleep_buckets.items():
        bucket_logs = [l for l in logs if lo <= l.get('sleep_hours', 7) < hi]
        if bucket_logs:
            rate = round(
                sum(1 for l in bucket_logs if l.get('seizure_occurred')) / len(bucket_logs) * 100, 1
            )
        else:
            rate = 0
        sleep_vs_seizure.append({
            'bucket': label,
            'total_entries': len(bucket_logs),
            'seizure_rate': rate,
        })

    # adherence_impact: seizure rate for adherent vs non-adherent days
    adherent_logs = [l for l in logs if l.get('medication_adherence') == 'yes']
    nonadherent_logs = [l for l in logs if l.get('medication_adherence') != 'yes']
    adherence_impact = [
        {
            'group': 'adherent',
            'total_entries': len(adherent_logs),
            'seizure_rate': round(
                sum(1 for l in adherent_logs if l.get('seizure_occurred')) / len(adherent_logs) * 100, 1
            ) if adherent_logs else 0,
        },
        {
            'group': 'non_adherent',
            'total_entries': len(nonadherent_logs),
            'seizure_rate': round(
                sum(1 for l in nonadherent_logs if l.get('seizure_occurred')) / len(nonadherent_logs) * 100, 1
            ) if nonadherent_logs else 0,
        },
    ]

    return {
        'available': True,
        'total_logs': total_logs,
        'total_patients': total_patients,
        'avg_seizure_rate': avg_seizure_rate,
        'avg_sleep_hours': avg_sleep_hours,
        'avg_stress_level': avg_stress_level,
        'medication_adherence_rate': medication_adherence_rate,
        'top_trigger': top_trigger,
        'days_tracked': days_tracked,
        'trigger_distribution': trigger_distribution,
        'temporal_trend': temporal_trend,
        'factor_correlation': factor_correlation,
        'risk_level_distribution': risk_level_distribution,
        'sleep_vs_seizure': sleep_vs_seizure,
        'adherence_impact': adherence_impact,
    }


def breakdown():
    """Return per-patient summary and all logs for the Trigger Tracking dashboard."""
    logs = _load_all_data()

    # Group logs by patient
    patient_logs_map = {}
    for l in logs:
        pid = l.get('patient_id', '')
        patient_logs_map.setdefault(pid, []).append(l)

    patients = []
    for pid in sorted(patient_logs_map.keys()):
        plogs = patient_logs_map[pid]
        total = len(plogs)
        sz_count = sum(1 for l in plogs if l.get('seizure_occurred'))
        sz_rate = round(sz_count / total * 100, 1) if total else 0
        avg_sleep = _avg([l.get('sleep_hours', 0) for l in plogs])
        avg_stress = _avg([l.get('stress_level', 0) for l in plogs])
        adherent = sum(1 for l in plogs if l.get('medication_adherence') == 'yes')
        adherence_rate = round(adherent / total * 100, 1) if total else 0

        # Top triggers for this patient
        pt_triggers = Counter(
            l.get('primary_trigger', 'Unknown')
            for l in plogs
            if l.get('seizure_occurred') and l.get('primary_trigger')
        )
        top_triggers = [t for t, _ in pt_triggers.most_common(3)]

        risk_level = _compute_risk_level(plogs)

        # Recent logs (last 7 by date)
        sorted_plogs = sorted(plogs, key=lambda x: x.get('log_date', ''), reverse=True)
        recent_logs = sorted_plogs[:7]

        patients.append({
            'patient_id': pid,
            'total_logs': total,
            'seizure_count': sz_count,
            'seizure_rate': sz_rate,
            'avg_sleep': avg_sleep,
            'avg_stress': avg_stress,
            'adherence_rate': adherence_rate,
            'top_triggers': top_triggers,
            'risk_level': risk_level,
            'recent_logs': recent_logs,
        })

    # All logs sorted by date desc
    all_logs = sorted(logs, key=lambda x: x.get('log_date', ''), reverse=True)

    return {
        'patients': patients,
        'all_logs': all_logs,
    }


def definitions():
    """Return seizure trigger terminology, clinical references, and ILAE standards."""
    return {
        'concepts': [
            {
                'name': 'Seizure Trigger',
                'description': 'An identifiable external or internal factor that increases the likelihood of a seizure occurring in a person with epilepsy. Common triggers include sleep deprivation, psychological stress, missed antiepileptic medication, alcohol consumption, photosensitivity (flashing lights), illness/fever, hormonal fluctuations (catamenial epilepsy), and weather changes. Trigger identification is a cornerstone of self-management in epilepsy care. The ILAE recognizes that triggers lower the seizure threshold rather than directly causing seizures, meaning individual susceptibility varies.',
            },
            {
                'name': 'Trigger Diary',
                'description': 'A structured daily log where patients record potential seizure-precipitating factors alongside seizure occurrence. Modern electronic trigger diaries capture: sleep duration and quality, stress levels, medication adherence, caffeine and alcohol intake, exercise, illness, photosensitivity exposure, menstrual phase, weather conditions, mood, and fatigue. Consistent diary keeping (minimum 3 months) enables statistical analysis of individualized trigger-seizure associations. Evidence shows that patients who maintain trigger diaries achieve better seizure control through proactive avoidance strategies.',
            },
            {
                'name': 'Risk Score',
                'description': 'A composite numerical score (0-100) estimating a patient\'s daily seizure risk based on their logged factors. Components include: seizure frequency rate (40% weight), average stress level (20%), sleep deficit relative to recommended 7-8 hours (20%), and medication non-adherence rate (20%). Risk levels: Low (score < 20) indicates well-controlled epilepsy with good self-management; Moderate (20-39) suggests some modifiable risk factors; High (40-59) indicates multiple concurrent risk factors requiring intervention; Critical (60+) suggests immediate clinical review needed. The score is recalculated with each new diary entry.',
            },
            {
                'name': 'Medication Adherence',
                'description': 'The degree to which a patient takes antiepileptic drugs (AEDs) as prescribed. Tracked as: Yes (all doses taken on time), Partial (some doses taken or taken late), No (one or more doses missed entirely). Non-adherence is the single most modifiable risk factor for breakthrough seizures. Studies show that missing even one dose of a short-half-life AED (e.g., levetiracetam) can significantly increase seizure risk within 24-48 hours. The dashboard tracks missed doses (0-3 per day) and correlates adherence patterns with seizure occurrence to provide personalized feedback.',
            },
            {
                'name': 'Sleep Hygiene',
                'description': 'The relationship between sleep quality/quantity and seizure susceptibility. Sleep deprivation is one of the most potent and universal seizure triggers, affecting up to 90% of people with epilepsy. The dashboard tracks both sleep duration (hours) and subjective sleep quality (1-5 scale). Clinical thresholds: <5 hours = high risk, 5-6 hours = moderate risk, 6-7 hours = mild risk, 7-9 hours = optimal. Sleep quality factors include: sleep fragmentation, REM disruption (common with certain AEDs), sleep apnea comorbidity, and circadian rhythm disturbances. NREM sleep stages are particularly relevant as many epilepsies show increased interictal discharges during slow-wave sleep.',
            },
            {
                'name': 'Stress Management',
                'description': 'Psychological and physiological stress is a frequently reported seizure trigger, mediated through cortisol elevation, sympathetic nervous system activation, and hyperventilation. The dashboard tracks perceived stress on a 1-10 scale and correlates with seizure occurrence. Stress-related seizure mechanisms include: cortisol-mediated reduction in GABA-ergic inhibition, stress-induced sleep disruption (indirect trigger), hyperventilation-induced alkalosis (lowers seizure threshold), and autonomic dysregulation. Behavioral interventions (CBT, mindfulness, biofeedback) have Level B evidence for reducing stress-related seizures.',
            },
            {
                'name': 'Photosensitivity',
                'description': 'An abnormal cortical response to visual stimuli (flickering lights, patterns, screen use) that can provoke seizures in susceptible individuals. Affects approximately 3-5% of all people with epilepsy, but up to 30% of those with juvenile myoclonic epilepsy or childhood absence epilepsy. The dashboard tracks daily photo exposure levels (none/low/moderate/high) and correlates with seizure events. Clinical photosensitivity is confirmed by photic stimulation during routine EEG. Management includes polarized lenses, screen filters, adequate ambient lighting, and avoidance of known visual triggers.',
            },
            {
                'name': 'Hormonal Triggers',
                'description': 'Seizure susceptibility fluctuations linked to the menstrual cycle (catamenial epilepsy). Three recognized patterns: C1 — perimenstrual (days -3 to +3, estrogen withdrawal), most common; C2 — periovulatory (days 10-13, estrogen surge); C3 — inadequate luteal phase (days 10-3 before menses, low progesterone). The dashboard tracks menstrual phase (follicular/luteal/ovulation/none/n_a) and correlates with seizure occurrence. Affects approximately 40% of women with epilepsy. Treatment adjuncts include progesterone supplementation, clobazam cycling, and acetazolamide perimenstrually.',
            },
            {
                'name': 'Trigger Correlation Engine',
                'description': 'A statistical analysis module that computes point-biserial correlations between continuous daily factors (sleep hours, stress level, caffeine intake, alcohol units, fatigue level) and the binary seizure occurrence variable. Correlations range from -1 to +1: negative values (e.g., sleep hours) indicate the factor is protective when higher; positive values (e.g., stress level) indicate the factor is provocative when higher. Correlations are computed across all patients and per-patient for individualized insights. Clinical significance threshold: |r| > 0.15. Results inform personalized trigger avoidance recommendations.',
            },
            {
                'name': 'Daily Risk Score',
                'description': 'A real-time seizure risk estimate updated with each diary entry. Incorporates current-day factors (sleep, stress, medication, illness) weighted by the patient\'s individualized trigger profile. Unlike the cumulative risk score, the daily score reflects acute risk based on the most recent 24-hour window. Thresholds: Green (low risk, score < 30) — maintain current routines; Yellow (moderate, 30-50) — increase vigilance, ensure medication adherence; Orange (high, 50-70) — avoid known triggers, consider rescue medication access; Red (critical, 70+) — contact clinical team, avoid driving/swimming/heights.',
            },
            {
                'name': 'ILAE Trigger Classification',
                'description': 'The International League Against Epilepsy (ILAE) classifies seizure precipitants into: Reflex triggers (specific sensory stimuli reproducibly provoking seizures, e.g., photosensitivity, startle, reading, hot water); Facilitating factors (conditions lowering seizure threshold without directly causing seizures, e.g., sleep deprivation, stress, alcohol, fever); and Provoking factors (acute conditions causing seizures in otherwise non-epileptic individuals, e.g., acute head trauma, drug intoxication, metabolic derangement). This dashboard primarily tracks facilitating factors and reflex triggers as reported in patient diaries.',
            },
            {
                'name': 'Prodromes vs Triggers',
                'description': 'A critical clinical distinction: Prodromes are subjective warning symptoms occurring hours to days before a seizure (irritability, headache, mood change, cognitive fogging) that signal an impending seizure but do not cause it. Triggers are external or internal factors that actively lower the seizure threshold and increase seizure likelihood. Patients often confuse the two — for example, fatigue may be a trigger (sleep deprivation lowering threshold) or a prodrome (pre-seizure state manifesting as fatigue). The diary\'s notes field allows patients to flag prodromal symptoms, and clinicians can retrospectively differentiate prodromes from triggers through pattern analysis across multiple diary entries.',
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
        pprint.pprint({k: v for k, v in p.items() if k != 'recent_logs'})
        print(f'  recent_logs: {len(p["recent_logs"])} entries')
    print(f'\nTotal patients: {len(bd["patients"])}')
    print(f'Total logs: {len(bd["all_logs"])}')
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Concepts: {len(df["concepts"])}')
    for c in df['concepts']:
        print(f'  {c["name"]}')
