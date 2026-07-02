"""ADL (Activities of Daily Living) Dashboard — functional assessment analytics from clinical.db.

Tracks Barthel Index (ADL independence), QOLIE-31 (quality of life),
Epworth Sleepiness Scale, per-patient functional profiles, severity
distributions, and longitudinal trends.

Sources:
- assessments table (BARTHEL, QOLIE31, EPWORTH instruments)
- patients table (40 rows for coverage calculations)
"""

import sqlite3
import os
from collections import defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


# --- Barthel Index interpretation ---
def _barthel_category(score):
    if score is None:
        return 'Unknown'
    if score >= 80:
        return 'Independent'
    if score >= 60:
        return 'Mildly dependent'
    if score >= 40:
        return 'Moderately dependent'
    if score >= 20:
        return 'Severely dependent'
    return 'Totally dependent'


def _epworth_category(score):
    if score is None:
        return 'Unknown'
    if score <= 5:
        return 'Lower normal'
    if score <= 10:
        return 'Higher normal'
    if score <= 12:
        return 'Mild sleepiness'
    if score <= 15:
        return 'Moderate sleepiness'
    return 'Severe sleepiness'


def _qolie_category(score):
    if score is None:
        return 'Unknown'
    if score >= 70:
        return 'Good QoL'
    if score >= 50:
        return 'Moderate QoL'
    if score >= 30:
        return 'Poor QoL'
    return 'Very poor QoL'


def adl_overview():
    """KPI summary: total functional assessments, coverage, severity distribution, instrument stats."""
    conn = _conn()
    cur = conn.cursor()

    instruments = ['BARTHEL', 'QOLIE31', 'EPWORTH']
    total_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM patients", default=0)

    # Per-instrument stats
    instrument_stats = []
    all_rows = []
    for inst in instruments:
        rows = _safe_rows(cur, """
            SELECT patient_id, score, max_score, interpretation, level, examiner, created_at
            FROM assessments WHERE instrument=? ORDER BY created_at
        """, (inst,))
        all_rows.extend([(inst, *r) for r in rows])
        count = len(rows)
        scores = [r[1] for r in rows if r[1] is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        unique_patients = len(set(r[0] for r in rows))
        levels = defaultdict(int)
        for r in rows:
            levels[r[4] or 'unknown'] += 1
        instrument_stats.append({
            'instrument': inst,
            'label': {'BARTHEL': 'Barthel Index (ADL)', 'QOLIE31': 'QOLIE-31 (Quality of Life)',
                       'EPWORTH': 'Epworth Sleepiness Scale'}[inst],
            'count': count,
            'unique_patients': unique_patients,
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'level_distribution': dict(levels)
        })

    total_assessments = sum(s['count'] for s in instrument_stats)
    patients_assessed = len(set(r[1] for r in all_rows))
    coverage_pct = round(patients_assessed / total_patients * 100, 1) if total_patients else 0

    # Severity distribution across all instruments
    severity_counts = defaultdict(int)
    for r in all_rows:
        severity_counts[r[5] or 'unknown'] += 1

    # Independence rate (Barthel >= 80)
    barthel_scores = [r[2] for r in all_rows if r[0] == 'BARTHEL' and r[2] is not None]
    independent_count = sum(1 for s in barthel_scores if s >= 80)
    independence_rate = round(independent_count / len(barthel_scores) * 100, 1) if barthel_scores else 0

    # Avg QOLIE
    qolie_scores = [r[2] for r in all_rows if r[0] == 'QOLIE31' and r[2] is not None]
    avg_qolie = round(sum(qolie_scores) / len(qolie_scores), 1) if qolie_scores else 0

    # Avg Epworth
    epworth_scores = [r[2] for r in all_rows if r[0] == 'EPWORTH' and r[2] is not None]
    avg_epworth = round(sum(epworth_scores) / len(epworth_scores), 1) if epworth_scores else 0

    # Patients needing intervention (severe level in any instrument)
    severe_patients = set()
    for r in all_rows:
        if r[5] == 'severe':
            severe_patients.add(r[1])

    conn.close()
    return {
        'total_assessments': total_assessments,
        'total_patients': total_patients,
        'patients_assessed': patients_assessed,
        'coverage_pct': coverage_pct,
        'instruments_tracked': len(instruments),
        'instrument_stats': instrument_stats,
        'severity_distribution': dict(severity_counts),
        'independence_rate': independence_rate,
        'avg_qolie': avg_qolie,
        'avg_epworth': avg_epworth,
        'severe_patients_count': len(severe_patients)
    }


def adl_breakdown():
    """Per-patient functional profiles and score distributions."""
    conn = _conn()
    cur = conn.cursor()

    instruments = ['BARTHEL', 'QOLIE31', 'EPWORTH']
    categorizers = {
        'BARTHEL': _barthel_category,
        'QOLIE31': _qolie_category,
        'EPWORTH': _epworth_category
    }

    # Per-patient profiles
    patient_data = defaultdict(lambda: {'instruments': {}, 'overall_level': 'unknown'})
    for inst in instruments:
        rows = _safe_rows(cur, """
            SELECT patient_id, score, max_score, interpretation, level, created_at
            FROM assessments WHERE instrument=? ORDER BY created_at
        """, (inst,))
        for r in rows:
            pid = r[0]
            cat = categorizers[inst](r[1])
            patient_data[pid]['instruments'][inst] = {
                'score': r[1],
                'max_score': r[2],
                'interpretation': r[3],
                'level': r[4],
                'category': cat,
                'assessed_at': r[5]
            }

    # Compute overall functional level per patient
    level_priority = {'severe': 3, 'moderate': 2, 'normal': 1, 'unknown': 0}
    patient_profiles = []
    for pid in sorted(patient_data.keys()):
        p = patient_data[pid]
        instruments_done = list(p['instruments'].keys())
        worst_level = max((p['instruments'].get(i, {}).get('level', 'unknown') for i in instruments),
                         key=lambda l: level_priority.get(l, 0))
        patient_profiles.append({
            'patient_id': pid,
            'instruments_completed': len(instruments_done),
            'instruments_list': instruments_done,
            'overall_level': worst_level,
            'scores': p['instruments']
        })

    # Score distribution buckets for each instrument
    score_distributions = {}
    for inst in instruments:
        rows = _safe_rows(cur, "SELECT score FROM assessments WHERE instrument=? AND score IS NOT NULL", (inst,))
        scores = [r[0] for r in rows]
        if inst == 'BARTHEL':
            buckets = {'0-20': 0, '21-40': 0, '41-60': 0, '61-80': 0, '81-100': 0}
            for s in scores:
                if s <= 20: buckets['0-20'] += 1
                elif s <= 40: buckets['21-40'] += 1
                elif s <= 60: buckets['41-60'] += 1
                elif s <= 80: buckets['61-80'] += 1
                else: buckets['81-100'] += 1
        elif inst == 'QOLIE31':
            buckets = {'0-25': 0, '26-50': 0, '51-75': 0, '76-100': 0}
            for s in scores:
                if s <= 25: buckets['0-25'] += 1
                elif s <= 50: buckets['26-50'] += 1
                elif s <= 75: buckets['51-75'] += 1
                else: buckets['76-100'] += 1
        else:  # EPWORTH
            buckets = {'0-5 Normal': 0, '6-10 High-Normal': 0, '11-12 Mild': 0, '13-15 Moderate': 0, '16-24 Severe': 0}
            for s in scores:
                if s <= 5: buckets['0-5 Normal'] += 1
                elif s <= 10: buckets['6-10 High-Normal'] += 1
                elif s <= 12: buckets['11-12 Mild'] += 1
                elif s <= 15: buckets['13-15 Moderate'] += 1
                else: buckets['16-24 Severe'] += 1
        score_distributions[inst] = [{'bucket': k, 'count': v} for k, v in buckets.items()]

    # Recent assessments
    recent = _safe_rows(cur, """
        SELECT patient_id, instrument, score, max_score, interpretation, level, created_at
        FROM assessments WHERE instrument IN ('BARTHEL','QOLIE31','EPWORTH')
        ORDER BY created_at DESC LIMIT 20
    """)
    recent_list = [{'patient_id': r[0], 'instrument': r[1], 'score': r[2],
                     'max_score': r[3], 'interpretation': r[4], 'level': r[5],
                     'assessed_at': r[6]} for r in recent]

    conn.close()
    return {
        'patient_profiles': patient_profiles,
        'score_distributions': score_distributions,
        'recent_assessments': recent_list
    }


def adl_definitions():
    """ADL metric definitions and clinical relevance."""
    return {
        'sections': [
            {
                'title': 'ADL Assessment Instruments',
                'items': [
                    {'term': 'Barthel Index', 'definition': 'Measures functional independence in 10 activities of daily living (feeding, bathing, grooming, dressing, bowel/bladder control, toilet use, transfers, mobility, stairs). Score 0-100; higher = more independent.'},
                    {'term': 'QOLIE-31', 'definition': 'Quality of Life in Epilepsy — 31-item patient-reported instrument covering seizure worry, overall QoL, emotional well-being, energy/fatigue, cognitive functioning, medication effects, social function. Score 0-100; higher = better QoL.'},
                    {'term': 'Epworth Sleepiness Scale', 'definition': 'Self-administered questionnaire measuring daytime sleepiness in 8 situations. Score 0-24; higher = more sleepy. Scores > 10 indicate excessive daytime sleepiness, common in epilepsy due to AED side effects and nocturnal seizures.'}
                ]
            },
            {
                'title': 'Scoring & Severity Levels',
                'items': [
                    {'term': 'Barthel 81-100', 'definition': 'Independent — patient can perform ADLs with minimal or no assistance.'},
                    {'term': 'Barthel 61-80', 'definition': 'Mildly dependent — needs occasional help with some ADLs.'},
                    {'term': 'Barthel 41-60', 'definition': 'Moderately dependent — requires regular assistance with several ADLs.'},
                    {'term': 'Barthel 0-40', 'definition': 'Severely to totally dependent — needs extensive help or full care.'},
                    {'term': 'QOLIE-31 >= 70', 'definition': 'Good quality of life — seizures and medications have limited impact on daily functioning.'},
                    {'term': 'QOLIE-31 < 50', 'definition': 'Poor quality of life — significant seizure burden, medication side effects, or psychosocial impact.'},
                    {'term': 'Epworth <= 10', 'definition': 'Normal daytime sleepiness.'},
                    {'term': 'Epworth > 10', 'definition': 'Excessive daytime sleepiness — warrants clinical investigation for sleep disorders or AED-related sedation.'}
                ]
            },
            {
                'title': 'Quality Metrics',
                'items': [
                    {'term': 'ADL Coverage', 'definition': 'Percentage of patients with at least one functional assessment. Target: 100% of epilepsy patients should have a baseline Barthel assessment.'},
                    {'term': 'Independence Rate', 'definition': 'Percentage of patients scoring Barthel >= 80 (independent). Tracks functional outcomes across the cohort.'},
                    {'term': 'Severe Patients', 'definition': 'Patients with severe-level scores on any functional instrument — candidates for OT/rehab intervention referral.'}
                ]
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'ILAE Functional Outcomes', 'definition': 'International League Against Epilepsy recommends tracking ADL and QoL outcomes alongside seizure freedom as measures of treatment success.'},
                    {'term': 'IEC 62304 (SaMD)', 'definition': 'ADL tracking software must maintain data integrity and traceability per medical device software standards.'},
                    {'term': 'FDA PRO Guidance', 'definition': 'Patient-Reported Outcomes (QOLIE-31, Epworth) should be collected with validated instruments and standardized scoring.'},
                    {'term': 'OT/Rehab Referral', 'definition': 'Patients with Barthel < 60 should be flagged for occupational therapy assessment and rehabilitation goal-setting.'}
                ]
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Low ADL coverage', 'definition': 'Ensure all patients receive baseline Barthel assessment at intake. Add automated reminders for missing assessments.'},
                    {'term': 'High dependency rate', 'definition': 'Review seizure control (frequency, severity), AED side effects, and comorbidities. Consider OT referral for functional rehabilitation.'},
                    {'term': 'Poor QoL scores', 'definition': 'Investigate seizure worry, medication side effects, mood disorders (PHQ-9/GAD-7 cross-reference). Multidisciplinary care plan adjustment.'},
                    {'term': 'Excessive sleepiness', 'definition': 'Review AED regimen for sedating drugs (phenobarbital, benzodiazepines). Screen for sleep apnea or nocturnal seizures. Consider sleep study referral.'}
                ]
            }
        ]
    }
