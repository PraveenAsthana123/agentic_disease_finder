"""Functional Recovery Over Time Dashboard — Occupational Therapist tool.

Tracks longitudinal functional recovery for epilepsy/neuro patients using
validated PRO instruments stored in the pro_outcomes table (180 rows, 30 patients).

Domains: Daily Function, Social Function, Cognition (MoCA), Quality of Life
(QOLIE-31), Work Productivity (WPAI), Fatigue, Mood, Sleep, Seizure Worry.

Sources:
  - pro_outcomes table (fields_json column with monthly assessments)
"""

import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, default=0):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


def _avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def _parse_fields(rows):
    """Parse list of (id, patient_id, fields_json, created_at) into dicts."""
    results = []
    for row in rows:
        try:
            fields = json.loads(row[2]) if row[2] else {}
        except Exception:
            fields = {}
        fields['_row_id'] = row[0]
        fields['_patient_id'] = row[1]
        fields['_created_at'] = row[3]
        results.append(fields)
    return results


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI summary: functional recovery metrics, trends, distributions."""
    conn = _conn()
    cur = conn.cursor()

    # Load all records
    raw = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM pro_outcomes ORDER BY created_at")
    records = _parse_fields(raw)

    total_assessments = len(records)
    patient_ids = list({r.get('patient_id', r.get('_patient_id')) for r in records})
    total_patients = len(patient_ids)

    # Averages
    daily_vals = [r['daily_function_rating'] for r in records if r.get('daily_function_rating') is not None]
    social_vals = [r['social_function_rating'] for r in records if r.get('social_function_rating') is not None]
    qolie_vals = [r['qolie31_score'] for r in records if r.get('qolie31_score') is not None]
    wpai_vals = [r['wpai_percent'] for r in records if r.get('wpai_percent') is not None]

    # Per-patient first/last daily_function_rating for trajectory
    patient_records = defaultdict(list)
    for r in records:
        pid = r.get('patient_id', r.get('_patient_id'))
        patient_records[pid].append(r)

    recovery_summary = []
    improving = 0
    declining = 0
    for pid in sorted(patient_records.keys()):
        recs = sorted(patient_records[pid], key=lambda x: x.get('assessment_date', x.get('_created_at', '')))
        first_daily = recs[0].get('daily_function_rating')
        last_daily = recs[-1].get('daily_function_rating')
        if first_daily is not None and last_daily is not None:
            change = round(last_daily - first_daily, 2)
            if change > 0:
                trajectory = 'improving'
                improving += 1
            elif change < 0:
                trajectory = 'declining'
                declining += 1
            else:
                trajectory = 'stable'
        else:
            change = 0
            trajectory = 'stable'
        recovery_summary.append({
            'patient_id': pid,
            'first_daily': first_daily,
            'last_daily': last_daily,
            'change': change,
            'trajectory': trajectory,
        })

    # Function trend by month
    monthly_agg = defaultdict(lambda: {'daily': [], 'social': []})
    for r in records:
        date_str = r.get('assessment_date', r.get('_created_at', ''))
        if date_str:
            month_key = date_str[:7]  # YYYY-MM
        else:
            continue
        if r.get('daily_function_rating') is not None:
            monthly_agg[month_key]['daily'].append(r['daily_function_rating'])
        if r.get('social_function_rating') is not None:
            monthly_agg[month_key]['social'].append(r['social_function_rating'])

    function_trend = []
    for month in sorted(monthly_agg.keys()):
        function_trend.append({
            'date': month,
            'avg_daily': _avg(monthly_agg[month]['daily']),
            'avg_social': _avg(monthly_agg[month]['social']),
        })

    # QOLIE-31 distribution
    qolie_tiers = {'Poor': 0, 'Fair': 0, 'Good': 0, 'Excellent': 0}
    for v in qolie_vals:
        if v < 40:
            qolie_tiers['Poor'] += 1
        elif v < 60:
            qolie_tiers['Fair'] += 1
        elif v < 80:
            qolie_tiers['Good'] += 1
        else:
            qolie_tiers['Excellent'] += 1
    qolie_distribution = [{'tier': t, 'count': c} for t, c in qolie_tiers.items()]

    # Fatigue distribution
    fatigue_groups = {'Low': 0, 'Moderate': 0, 'High': 0}
    fatigue_vals = [r['fatigue_level'] for r in records if r.get('fatigue_level') is not None]
    for v in fatigue_vals:
        if v <= 3:
            fatigue_groups['Low'] += 1
        elif v <= 6:
            fatigue_groups['Moderate'] += 1
        else:
            fatigue_groups['High'] += 1
    fatigue_distribution = [{'level': l, 'count': c} for l, c in fatigue_groups.items()]

    conn.close()

    return {
        'available': True,
        'kpis': {
            'total_patients': total_patients,
            'total_assessments': total_assessments,
            'avg_daily_function': _avg(daily_vals),
            'avg_social_function': _avg(social_vals),
            'avg_qolie31': _avg(qolie_vals),
            'avg_wpai': _avg(wpai_vals),
            'patients_improving': improving,
            'patients_declining': declining,
        },
        'function_trend': function_trend,
        'qolie_distribution': qolie_distribution,
        'fatigue_distribution': fatigue_distribution,
        'recovery_summary': recovery_summary,
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed per-patient timelines, domain scores, comorbidity flags, volume."""
    conn = _conn()
    cur = conn.cursor()

    raw = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM pro_outcomes ORDER BY created_at")
    records = _parse_fields(raw)

    patient_records = defaultdict(list)
    for r in records:
        pid = r.get('patient_id', r.get('_patient_id'))
        patient_records[pid].append(r)

    # Patient timelines
    patient_timelines = []
    for pid in sorted(patient_records.keys()):
        recs = sorted(patient_records[pid], key=lambda x: x.get('assessment_date', x.get('_created_at', '')))
        assessments = []
        for r in recs:
            assessments.append({
                'date': r.get('assessment_date', r.get('_created_at', '')),
                'daily_function': r.get('daily_function_rating'),
                'social_function': r.get('social_function_rating'),
                'moca': r.get('moca_score'),
                'qolie31': r.get('qolie31_score'),
                'wpai': r.get('wpai_percent'),
                'fatigue': r.get('fatigue_level'),
                'mood': r.get('mood_rating'),
                'sleep_hours': r.get('sleep_hours'),
                'seizure_worry': r.get('seizure_worry_score'),
                'notes': r.get('notes', ''),
            })
        first_daily = recs[0].get('daily_function_rating')
        last_daily = recs[-1].get('daily_function_rating')
        if first_daily is not None and last_daily is not None:
            if last_daily > first_daily:
                trajectory = 'improving'
            elif last_daily < first_daily:
                trajectory = 'declining'
            else:
                trajectory = 'stable'
        else:
            trajectory = 'stable'

        patient_timelines.append({
            'patient_id': pid,
            'assessments': assessments,
            'assessment_count': len(assessments),
            'latest_daily': recs[-1].get('daily_function_rating'),
            'latest_social': recs[-1].get('social_function_rating'),
            'trajectory': trajectory,
        })

    # Domain scores
    daily_scores = [r.get('daily_function_rating') for r in records if r.get('daily_function_rating') is not None]
    social_scores = [r.get('social_function_rating') for r in records if r.get('social_function_rating') is not None]
    moca_scores = [r.get('moca_score') for r in records if r.get('moca_score') is not None]
    qolie_scores = [r.get('qolie31_score') for r in records if r.get('qolie31_score') is not None]
    wpai_scores = [r.get('wpai_percent') for r in records if r.get('wpai_percent') is not None]

    domain_scores = [
        {
            'domain': 'Daily Function',
            'avg_score': _avg(daily_scores),
            'patients_below_threshold': sum(1 for v in daily_scores if v < 5),
            'threshold': 5,
        },
        {
            'domain': 'Social Function',
            'avg_score': _avg(social_scores),
            'patients_below_threshold': sum(1 for v in social_scores if v < 5),
            'threshold': 5,
        },
        {
            'domain': 'Cognitive (MoCA)',
            'avg_score': _avg(moca_scores),
            'patients_below_threshold': sum(1 for v in moca_scores if v < 22),
            'threshold': 22,
        },
        {
            'domain': 'Quality of Life (QOLIE-31)',
            'avg_score': _avg(qolie_scores),
            'patients_below_threshold': sum(1 for v in qolie_scores if v < 50),
            'threshold': 50,
        },
        {
            'domain': 'Work Productivity (WPAI)',
            'avg_score': _avg(wpai_scores),
            'patients_below_threshold': sum(1 for v in wpai_scores if v > 50),
            'threshold': 50,
        },
    ]

    # Comorbidity flags — latest record per patient
    comorbidity_flags = []
    for pid in sorted(patient_records.keys()):
        recs = sorted(patient_records[pid], key=lambda x: x.get('assessment_date', x.get('_created_at', '')))
        latest = recs[-1]
        mem = bool(latest.get('memory_complaints'))
        conc = bool(latest.get('concentration_difficulty'))
        phq9 = latest.get('phq9_score')
        gad7 = latest.get('gad7_score')
        flag_count = sum([
            mem,
            conc,
            (phq9 or 0) >= 10,
            (gad7 or 0) >= 10,
        ])
        comorbidity_flags.append({
            'patient_id': pid,
            'has_memory_complaints': mem,
            'has_concentration_difficulty': conc,
            'latest_phq9': phq9,
            'latest_gad7': gad7,
            'flag_count': flag_count,
        })

    # Monthly volume
    monthly_counts = defaultdict(int)
    for r in records:
        date_str = r.get('assessment_date', r.get('_created_at', ''))
        if date_str:
            monthly_counts[date_str[:7]] += 1
    monthly_volume = [{'month': m, 'count': c} for m, c in sorted(monthly_counts.items())]

    conn.close()

    return {
        'patient_timelines': patient_timelines,
        'domain_scores': domain_scores,
        'comorbidity_flags': comorbidity_flags,
        'monthly_volume': monthly_volume,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Clinical definitions, quality metrics, thresholds, compliance references."""
    return {
        'concepts': [
            {
                'term': 'Daily Function Rating',
                'definition': 'Self-reported 1-10 scale measuring a patient\'s ability to perform activities of daily living (ADLs) including self-care, household tasks, and routine responsibilities. Higher scores indicate greater independence.',
            },
            {
                'term': 'Social Function Rating',
                'definition': 'Self-reported 1-10 scale assessing participation in social activities, interpersonal relationships, and community engagement. Higher scores indicate better social integration.',
            },
            {
                'term': 'QOLIE-31',
                'definition': 'Quality of Life in Epilepsy Inventory (31-item). A validated 0-100 scale instrument measuring health-related quality of life specific to epilepsy, covering seizure worry, overall QoL, emotional well-being, energy/fatigue, cognitive function, medication effects, and social function.',
            },
            {
                'term': 'WPAI',
                'definition': 'Work Productivity and Activity Impairment questionnaire. Measures the percentage of work productivity loss (0-100%) due to health problems. Higher percentages indicate greater impairment. Covers absenteeism and presenteeism.',
            },
            {
                'term': 'MoCA',
                'definition': 'Montreal Cognitive Assessment. A 0-30 point screening tool for mild cognitive impairment covering attention, concentration, executive function, memory, language, visuoconstructional skills, conceptual thinking, calculations, and orientation. Score >= 26 is normal; < 22 suggests significant impairment.',
            },
            {
                'term': 'Functional Recovery Trajectory',
                'definition': 'Longitudinal classification of a patient\'s functional recovery path based on comparing their first and last daily function ratings. Categories: improving (last > first), stable (last == first), declining (last < first).',
            },
            {
                'term': 'Fatigue Level',
                'definition': 'Self-reported 1-10 scale measuring subjective fatigue severity. Low (1-3): minimal impact on daily activities. Moderate (4-6): noticeable but manageable fatigue. High (7-10): significant fatigue impairing function.',
            },
            {
                'term': 'Seizure Worry Score',
                'definition': 'Self-reported 1-10 scale quantifying the degree of worry and anxiety about seizure occurrence. Higher scores reflect greater anticipatory anxiety, which can independently impair functional recovery and quality of life.',
            },
        ],
        'quality_metrics': [
            {
                'metric': 'Assessment Completion Rate',
                'target': '>= 90% of scheduled monthly assessments completed',
                'description': 'Percentage of patients who complete their monthly PRO assessment on time.',
            },
            {
                'metric': 'Functional Improvement Rate',
                'target': '>= 50% of patients showing improving trajectory',
                'description': 'Proportion of patients whose daily function rating improves over their assessment period.',
            },
            {
                'metric': 'QOLIE-31 Mean Score',
                'target': '>= 60 (Good quality of life threshold)',
                'description': 'Population-level mean quality of life score across all assessments.',
            },
            {
                'metric': 'WPAI Below 30%',
                'target': '>= 60% of assessments with WPAI < 30%',
                'description': 'Proportion of assessments where work productivity impairment is below the clinically significant threshold.',
            },
            {
                'metric': 'MoCA Screening Coverage',
                'target': '100% of patients screened at least once',
                'description': 'Percentage of patients with at least one MoCA cognitive screening recorded.',
            },
        ],
        'thresholds': [
            {
                'domain': 'Daily Function',
                'threshold': 5,
                'interpretation': 'Score below 5 indicates significant functional impairment requiring occupational therapy intervention.',
            },
            {
                'domain': 'Social Function',
                'threshold': 5,
                'interpretation': 'Score below 5 indicates social isolation risk; consider social skills training or support group referral.',
            },
            {
                'domain': 'Cognitive (MoCA)',
                'threshold': 22,
                'interpretation': 'Score below 22 suggests significant cognitive impairment; referral for neuropsychological evaluation recommended.',
            },
            {
                'domain': 'Quality of Life (QOLIE-31)',
                'threshold': 50,
                'interpretation': 'Score below 50 indicates poor quality of life; comprehensive treatment plan review warranted.',
            },
            {
                'domain': 'Work Productivity (WPAI)',
                'threshold': 50,
                'interpretation': 'WPAI above 50% indicates severe work impairment; vocational rehabilitation or workplace accommodation review needed.',
            },
        ],
        'compliance_references': [
            {
                'ref': 'ILAE PRO Task Force (2019)',
                'note': 'Recommends standardized PRO collection for epilepsy including QOLIE-31, PHQ-9, GAD-7, and functional ratings.',
            },
            {
                'ref': 'AAN Quality Measure: Epilepsy QoL Assessment',
                'note': 'Quality of life should be assessed at least annually using a validated instrument such as QOLIE-31.',
            },
            {
                'ref': 'CMS MIPS: Functional Outcome Assessment',
                'note': 'Functional status assessment using validated tools is a reportable quality measure for neurological conditions.',
            },
            {
                'ref': 'HIPAA Privacy Rule (45 CFR 164)',
                'note': 'All PRO data must be stored and transmitted in compliance with HIPAA requirements for protected health information.',
            },
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import pprint
    print('=== Functional Recovery Dashboard ===\n')
    print('--- overview() ---')
    pprint.pprint(overview())
    print('\n--- breakdown() ---')
    pprint.pprint(breakdown())
    print('\n--- definitions() ---')
    pprint.pprint(definitions())
