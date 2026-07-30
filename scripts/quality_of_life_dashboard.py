"""Quality of Life Dashboard — comprehensive QoL analytics from pro_outcomes in clinical.db.

Tracks QOLIE-31 (epilepsy-specific QoL), PHQ-9 (depression), GAD-7 (anxiety),
NDDI-E (depression in epilepsy), mood, fatigue, social/daily functioning,
seizure worry, and longitudinal trends per patient.

Sources:
- pro_outcomes table (fields_json containing multi-instrument PRO scores)
- patients table (for coverage calculations)
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


# --- Clinical interpretation helpers ---

def _qolie_category(score):
    if score is None:
        return 'Unknown'
    if score >= 70:
        return 'Good'
    if score >= 50:
        return 'Moderate'
    if score >= 30:
        return 'Poor'
    return 'Very Poor'


def _phq9_category(score):
    if score is None:
        return 'Unknown'
    if score <= 4:
        return 'Minimal'
    if score <= 9:
        return 'Mild'
    if score <= 14:
        return 'Moderate'
    if score <= 19:
        return 'Moderately Severe'
    return 'Severe'


def _gad7_category(score):
    if score is None:
        return 'Unknown'
    if score <= 4:
        return 'Minimal'
    if score <= 9:
        return 'Mild'
    if score <= 14:
        return 'Moderate'
    return 'Severe'


def _nddi_category(score):
    if score is None:
        return 'Unknown'
    if score <= 11:
        return 'No depression'
    if score <= 15:
        return 'Possible MDD'
    return 'Probable MDD'


def _seizure_worry_level(score):
    if score is None:
        return 'Unknown'
    if score <= 2:
        return 'Low'
    if score <= 5:
        return 'Moderate'
    return 'High'


def _extract_fields(row):
    """Extract fields from a pro_outcomes row (id, patient_id, fields_json, created_at)."""
    import json
    try:
        return json.loads(row[2]) if row[2] else {}
    except Exception:
        return {}


def overview():
    """QoL KPIs: total assessments, patients assessed, domain averages, severity distributions."""
    con = _conn()
    cur = con.cursor()

    total = _safe(cur, "SELECT COUNT(*) FROM pro_outcomes")
    patients_assessed = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM pro_outcomes")
    total_patients = _safe(cur, "SELECT COUNT(*) FROM patients")
    coverage_pct = round(patients_assessed / max(total_patients, 1) * 100, 1)

    # Aggregate domain scores
    rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM pro_outcomes")
    qolie_scores, phq9_scores, gad7_scores, nddi_scores = [], [], [], []
    mood_scores, fatigue_scores, social_scores, daily_scores, worry_scores = [], [], [], [], []

    for r in rows:
        f = _extract_fields(r)
        if f.get('qolie31_score') is not None:
            qolie_scores.append(f['qolie31_score'])
        if f.get('phq9_score') is not None:
            phq9_scores.append(f['phq9_score'])
        if f.get('gad7_score') is not None:
            gad7_scores.append(f['gad7_score'])
        if f.get('nddi_e_score') is not None:
            nddi_scores.append(f['nddi_e_score'])
        if f.get('mood_rating') is not None:
            mood_scores.append(f['mood_rating'])
        if f.get('fatigue_level') is not None:
            fatigue_scores.append(f['fatigue_level'])
        if f.get('social_function_rating') is not None:
            social_scores.append(f['social_function_rating'])
        if f.get('daily_function_rating') is not None:
            daily_scores.append(f['daily_function_rating'])
        if f.get('seizure_worry_score') is not None:
            worry_scores.append(f['seizure_worry_score'])

    def _avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else None

    def _dist(lst, fn):
        d = defaultdict(int)
        for v in lst:
            d[fn(v)] += 1
        return dict(d)

    domain_averages = {
        'qolie31': _avg(qolie_scores),
        'phq9': _avg(phq9_scores),
        'gad7': _avg(gad7_scores),
        'nddi_e': _avg(nddi_scores),
        'mood': _avg(mood_scores),
        'fatigue': _avg(fatigue_scores),
        'social_function': _avg(social_scores),
        'daily_function': _avg(daily_scores),
        'seizure_worry': _avg(worry_scores),
    }

    severity_distributions = {
        'qolie31': _dist(qolie_scores, _qolie_category),
        'phq9': _dist(phq9_scores, _phq9_category),
        'gad7': _dist(gad7_scores, _gad7_category),
        'nddi_e': _dist(nddi_scores, _nddi_category),
        'seizure_worry': _dist(worry_scores, _seizure_worry_level),
    }

    # Patients at risk (PHQ-9 >= 15 or GAD-7 >= 15 or QOLIE-31 < 30)
    at_risk = set()
    for r in rows:
        f = _extract_fields(r)
        if (f.get('phq9_score', 0) >= 15 or
                f.get('gad7_score', 0) >= 15 or
                (f.get('qolie31_score') is not None and f['qolie31_score'] < 30)):
            at_risk.add(r[1])

    con.close()
    return {
        'total_assessments': total,
        'patients_assessed': patients_assessed,
        'total_patients': total_patients,
        'coverage_pct': coverage_pct,
        'domains_tracked': 9,
        'domain_averages': domain_averages,
        'severity_distributions': severity_distributions,
        'patients_at_risk': len(at_risk),
    }


def breakdown():
    """Per-patient QoL profiles, longitudinal trends, and domain cross-tabs."""
    con = _conn()
    cur = con.cursor()

    rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM pro_outcomes ORDER BY patient_id")

    # Per-patient profiles
    patient_data = defaultdict(list)
    for r in rows:
        f = _extract_fields(r)
        f['_date'] = f.get('assessment_date', r[3])
        patient_data[r[1]].append(f)

    profiles = []
    for pid, assessments in sorted(patient_data.items()):
        latest = sorted(assessments, key=lambda x: x.get('_date', ''), reverse=True)[0]
        profiles.append({
            'patient_id': pid,
            'assessments': len(assessments),
            'latest_date': latest.get('_date', ''),
            'qolie31': latest.get('qolie31_score'),
            'qolie31_category': _qolie_category(latest.get('qolie31_score')),
            'phq9': latest.get('phq9_score'),
            'phq9_category': _phq9_category(latest.get('phq9_score')),
            'gad7': latest.get('gad7_score'),
            'gad7_category': _gad7_category(latest.get('gad7_score')),
            'nddi_e': latest.get('nddi_e_score'),
            'nddi_e_category': _nddi_category(latest.get('nddi_e_score')),
            'mood': latest.get('mood_rating'),
            'fatigue': latest.get('fatigue_level'),
            'social_function': latest.get('social_function_rating'),
            'daily_function': latest.get('daily_function_rating'),
            'seizure_worry': latest.get('seizure_worry_score'),
            'seizure_worry_level': _seizure_worry_level(latest.get('seizure_worry_score')),
        })

    # Longitudinal trends (per patient, ordered by date)
    trends = {}
    for pid, assessments in sorted(patient_data.items()):
        ordered = sorted(assessments, key=lambda x: x.get('_date', ''))
        trends[pid] = [{
            'date': a.get('_date', ''),
            'qolie31': a.get('qolie31_score'),
            'phq9': a.get('phq9_score'),
            'gad7': a.get('gad7_score'),
            'mood': a.get('mood_rating'),
            'fatigue': a.get('fatigue_level'),
        } for a in ordered]

    # Recent assessments
    all_sorted = []
    for r in rows:
        f = _extract_fields(r)
        all_sorted.append({
            'patient_id': r[1],
            'date': f.get('assessment_date', r[3]),
            'qolie31': f.get('qolie31_score'),
            'phq9': f.get('phq9_score'),
            'gad7': f.get('gad7_score'),
            'mood': f.get('mood_rating'),
            'notes': f.get('notes', ''),
        })
    all_sorted.sort(key=lambda x: x.get('date', ''), reverse=True)

    con.close()
    return {
        'profiles': profiles,
        'trends': trends,
        'recent_assessments': all_sorted[:20],
    }


def definitions():
    """QoL instrument definitions and clinical relevance."""
    return {
        'instruments': [
            {
                'name': 'QOLIE-31',
                'full_name': 'Quality of Life in Epilepsy Inventory-31',
                'range': '0–100',
                'interpretation': 'Higher = better quality of life',
                'categories': {'Good': '≥70', 'Moderate': '50–69', 'Poor': '30–49', 'Very Poor': '<30'},
                'clinical_use': 'Gold-standard epilepsy-specific QoL measure covering seizure worry, emotional well-being, energy/fatigue, cognitive function, medication effects, social function, and overall QoL.',
            },
            {
                'name': 'PHQ-9',
                'full_name': 'Patient Health Questionnaire-9',
                'range': '0–27',
                'interpretation': 'Higher = more severe depression',
                'categories': {'Minimal': '0–4', 'Mild': '5–9', 'Moderate': '10–14', 'Moderately Severe': '15–19', 'Severe': '20–27'},
                'clinical_use': 'Widely validated depression screening tool. Score ≥10 warrants clinical attention; ≥15 suggests active treatment needed.',
            },
            {
                'name': 'GAD-7',
                'full_name': 'Generalized Anxiety Disorder-7',
                'range': '0–21',
                'interpretation': 'Higher = more severe anxiety',
                'categories': {'Minimal': '0–4', 'Mild': '5–9', 'Moderate': '10–14', 'Severe': '15–21'},
                'clinical_use': 'Brief anxiety screening. Anxiety is a common comorbidity in epilepsy, affecting seizure control and QoL.',
            },
            {
                'name': 'NDDI-E',
                'full_name': 'Neurological Disorders Depression Inventory for Epilepsy',
                'range': '6–24',
                'interpretation': 'Higher = more depressive symptoms',
                'categories': {'No depression': '≤11', 'Possible MDD': '12–15', 'Probable MDD': '>15'},
                'clinical_use': 'Epilepsy-specific depression screener. Avoids confounding somatic symptoms common in epilepsy (fatigue, concentration) that inflate general depression scales.',
            },
            {
                'name': 'Seizure Worry',
                'full_name': 'Seizure Worry Score',
                'range': '0–10',
                'interpretation': 'Higher = greater worry',
                'categories': {'Low': '0–2', 'Moderate': '3–5', 'High': '6–10'},
                'clinical_use': 'Measures anticipatory anxiety about seizures, which is the single largest driver of reduced QoL in epilepsy.',
            },
        ],
        'functional_domains': [
            {'name': 'Mood Rating', 'range': '1–10', 'description': 'Self-reported daily mood (higher = better)'},
            {'name': 'Fatigue Level', 'range': '1–10', 'description': 'Self-reported fatigue severity (higher = worse)'},
            {'name': 'Social Function', 'range': '1–10', 'description': 'Self-rated social activity and participation (higher = better)'},
            {'name': 'Daily Function', 'range': '1–10', 'description': 'Self-rated ability to perform daily activities (higher = better)'},
        ],
        'glossary': {
            'PRO': 'Patient-Reported Outcome — data collected directly from patients about their health status',
            'MDD': 'Major Depressive Disorder',
            'QoL': 'Quality of Life — multidimensional concept encompassing physical, psychological, and social well-being',
            'At-Risk': 'Patients with PHQ-9 ≥15 or GAD-7 ≥15 or QOLIE-31 <30, warranting clinical review',
        },
    }
