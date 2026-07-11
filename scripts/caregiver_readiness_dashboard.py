"""Caregiver Readiness Dashboard — training completion, burnout risk, safety plan coverage

Tracks caregiver preparedness across epilepsy training, first aid certification,
rescue medication readiness, safety/action plan compliance, and burnout risk
indicators. Provides readiness matrices and training gap analysis.

Sources:
  data/clinical.db — caregivers, patients, emergency_sos_events tables
"""

import os
import json
import sqlite3
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# ── Readiness dimensions (binary columns used for composite score) ─────────
_READINESS_DIMS = [
    'epilepsy_training_completed',
    'first_aid_certified',
    'rescue_med_trained',
    'safety_plan_exists',
    'seizure_action_plan_exists',
]

# ── Standard training topics expected ──────────────────────────────────────
_EXPECTED_TOPICS = [
    'Seizure Recognition',
    'First Aid Response',
    'Rescue Medication Administration',
    'Seizure Diary Logging',
    'SUDEP Awareness',
    'Trigger Management',
    'Emergency Protocols',
    'Medication Management',
]


def _db_query(sql, params=()):
    """Execute a SQL query and return list of dicts."""
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _parse_training_topics(raw):
    """Parse training_topics JSON string into a list."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _readiness_score(row):
    """Count how many of the 5 readiness dimensions are met (0-5)."""
    total = 0
    for dim in _READINESS_DIMS:
        val = row.get(dim, 0)
        if val and int(val) == 1:
            total += 1
    return total


def _readiness_level(score):
    """Map a 0-5 readiness score to a named level."""
    if score == 5:
        return 'fully_ready'
    if score >= 3:
        return 'mostly_ready'
    if score >= 1:
        return 'partially_ready'
    return 'not_ready'


def _burnout_level(score):
    """Map a 0-100 burnout score to a named level."""
    if score is None:
        return 'unknown'
    score = float(score)
    if score <= 30:
        return 'Low'
    if score <= 60:
        return 'Moderate'
    if score <= 80:
        return 'High'
    return 'Critical'


def _burnout_color(score):
    """Return color for burnout score."""
    if score is None:
        return 'gray'
    score = float(score)
    if score > 60:
        return 'red'
    if score > 40:
        return 'yellow'
    return 'green'


# ── Public API ─────────────────────────────────────────────────────────────

def overview():
    """High-level KPIs, readiness distribution, burnout distribution, role and topic coverage."""
    caregivers = _db_query('SELECT * FROM caregivers')
    total = len(caregivers)

    if total == 0:
        return {
            'kpis': [],
            'readiness_distribution': [],
            'burnout_distribution': [],
            'role_distribution': [],
            'training_topic_coverage': [],
        }

    # ── KPI calculations ──────────────────────────────────────────────────
    training_count = sum(1 for c in caregivers if c.get('epilepsy_training_completed') == 1)
    rescue_count = sum(1 for c in caregivers if c.get('rescue_med_trained') == 1)
    safety_count = sum(1 for c in caregivers if c.get('safety_plan_exists') == 1)

    burnout_scores = [c.get('burnout_score') for c in caregivers if c.get('burnout_score') is not None]
    mean_burnout = round(sum(burnout_scores) / len(burnout_scores), 1) if burnout_scores else 0

    confidence_vals = [c.get('seizure_first_aid_confidence') for c in caregivers
                       if c.get('seizure_first_aid_confidence') is not None]
    mean_confidence = round((sum(confidence_vals) / len(confidence_vals)) * 10, 1) if confidence_vals else 0

    training_pct = round(training_count / total * 100, 1)
    rescue_pct = round(rescue_count / total * 100, 1)
    safety_pct = round(safety_count / total * 100, 1)

    kpis = [
        {'label': 'Total Caregivers', 'value': total, 'color': 'blue'},
        {'label': 'Training Completion Rate', 'value': str(training_pct) + '%', 'color': 'green' if training_pct >= 80 else 'yellow'},
        {'label': 'Rescue Med Trained', 'value': str(rescue_pct) + '%', 'color': 'green' if rescue_pct >= 80 else 'yellow'},
        {'label': 'Safety Plans Active', 'value': str(safety_pct) + '%', 'color': 'green' if safety_pct >= 80 else 'yellow'},
        {'label': 'Mean Burnout Score', 'value': mean_burnout, 'color': _burnout_color(mean_burnout)},
        {'label': 'Mean Confidence', 'value': str(mean_confidence) + '%', 'color': 'green' if mean_confidence >= 70 else 'yellow'},
    ]

    # ── Readiness distribution ────────────────────────────────────────────
    readiness_counts = Counter()
    for c in caregivers:
        score = _readiness_score(c)
        level = _readiness_level(score)
        readiness_counts[level] += 1

    level_colors = {
        'fully_ready': 'green',
        'mostly_ready': 'blue',
        'partially_ready': 'yellow',
        'not_ready': 'red',
    }
    readiness_distribution = []
    for level in ['fully_ready', 'mostly_ready', 'partially_ready', 'not_ready']:
        readiness_distribution.append({
            'level': level,
            'count': readiness_counts.get(level, 0),
            'color': level_colors[level],
        })

    # ── Burnout distribution ──────────────────────────────────────────────
    burnout_ranges = [
        ('Low (0-30)', 0, 30, 'green'),
        ('Moderate (31-60)', 31, 60, 'yellow'),
        ('High (61-80)', 61, 80, 'orange'),
        ('Critical (81-100)', 81, 100, 'red'),
    ]
    burnout_distribution = []
    for label, lo, hi, color in burnout_ranges:
        count = sum(1 for s in burnout_scores if lo <= s <= hi)
        burnout_distribution.append({'range': label, 'count': count, 'color': color})

    # ── Role distribution ─────────────────────────────────────────────────
    role_counts = Counter(c.get('role', 'Unknown') for c in caregivers)
    role_distribution = [{'role': role, 'count': cnt} for role, cnt in role_counts.most_common()]

    # ── Training topic coverage ───────────────────────────────────────────
    topic_counts = Counter()
    for c in caregivers:
        topics = _parse_training_topics(c.get('training_topics'))
        for t in topics:
            topic_counts[t] += 1
    training_topic_coverage = [{'topic': topic, 'count': cnt} for topic, cnt in topic_counts.most_common()]

    return {
        'kpis': kpis,
        'readiness_distribution': readiness_distribution,
        'burnout_distribution': burnout_distribution,
        'role_distribution': role_distribution,
        'training_topic_coverage': training_topic_coverage,
    }


def breakdown():
    """Per-caregiver profiles, readiness matrix, burnout risk alerts, training gaps."""
    caregivers = _db_query('SELECT * FROM caregivers')

    # Build patient lookup
    patients = _db_query('SELECT * FROM patients')
    patient_map = {}
    for p in patients:
        pid = p.get('patient_id') or p.get('id')
        patient_map[pid] = p

    # ── Caregiver profiles ────────────────────────────────────────────────
    caregiver_profiles = []
    readiness_matrix = []
    burnout_risk_alerts = []
    training_gaps = []

    for c in caregivers:
        patient_id = c.get('patient_id')
        patient = patient_map.get(patient_id, {})
        patient_name = patient.get('name', 'Unknown')
        patient_disease = patient.get('disease', 'Unknown')

        score = _readiness_score(c)
        level = _readiness_level(score)
        burnout = c.get('burnout_score')
        b_level = _burnout_level(burnout)
        caregiver_name = c.get('name', 'Unknown')
        stress = c.get('caregiver_stress')
        sleep_quality = c.get('caregiver_sleep_quality')

        profile = {
            'patient_id': patient_id,
            'caregiver_name': caregiver_name,
            'role': c.get('role'),
            'availability': c.get('availability'),
            'experience_years': c.get('experience_years'),
            'training_completed': c.get('epilepsy_training_completed', 0),
            'first_aid': c.get('first_aid_certified', 0),
            'rescue_med': c.get('rescue_med_trained', 0),
            'confidence': c.get('seizure_first_aid_confidence'),
            'stress': stress,
            'sleep_quality': sleep_quality,
            'work_impact': c.get('work_impact'),
            'burnout_score': burnout,
            'burnout_level': b_level,
            'safety_plan': c.get('safety_plan_exists', 0),
            'action_plan': c.get('seizure_action_plan_exists', 0),
            'readiness_level': level,
            'patient_name': patient_name,
            'patient_disease': patient_disease,
        }
        caregiver_profiles.append(profile)

        # Readiness matrix row
        training_val = c.get('epilepsy_training_completed', 0)
        first_aid_val = c.get('first_aid_certified', 0)
        rescue_med_val = c.get('rescue_med_trained', 0)
        safety_plan_val = c.get('safety_plan_exists', 0)
        action_plan_val = c.get('seizure_action_plan_exists', 0)
        readiness_matrix.append({
            'name': caregiver_name,
            'training': training_val,
            'first_aid': first_aid_val,
            'rescue_med': rescue_med_val,
            'safety_plan': safety_plan_val,
            'action_plan': action_plan_val,
            'overall_readiness': level,
        })

        # Burnout risk alerts
        risk_factors = []
        if burnout is not None and float(burnout) > 60:
            risk_factors.append('Burnout score above 60 (' + str(burnout) + ')')
        if stress is not None and float(stress) > 7:
            risk_factors.append('High stress level (' + str(stress) + '/10)')
        if sleep_quality is not None and float(sleep_quality) < 4:
            risk_factors.append('Poor sleep quality (' + str(sleep_quality) + '/10)')
        if risk_factors:
            burnout_risk_alerts.append({
                'caregiver_name': caregiver_name,
                'patient_id': patient_id,
                'burnout_score': burnout,
                'stress': stress,
                'sleep_quality': sleep_quality,
                'risk_factors': risk_factors,
            })

        # Training gaps
        topics = _parse_training_topics(c.get('training_topics'))
        missing = [t for t in _EXPECTED_TOPICS if t not in topics]
        if missing:
            training_gaps.append({
                'caregiver_name': caregiver_name,
                'missing_topics': missing,
            })

    return {
        'caregiver_profiles': caregiver_profiles,
        'readiness_matrix': readiness_matrix,
        'burnout_risk_alerts': burnout_risk_alerts,
        'training_gaps': training_gaps,
    }


def definitions():
    """Reference definitions for caregiver readiness concepts and compliance."""
    return {
        'concepts': [
            {'name': 'Caregiver Readiness',
             'description': 'Composite measure of a caregiver\'s preparedness to manage epilepsy emergencies, combining training completion, certification, and safety plan coverage.'},
            {'name': 'Burnout Score',
             'description': 'A 0-100 scale measuring caregiver burnout, where higher scores indicate greater emotional exhaustion, depersonalization, and reduced personal accomplishment.'},
            {'name': 'Seizure Action Plan',
             'description': 'A written, individualized plan specifying seizure first aid steps, rescue medication instructions, and when to call emergency services for a specific patient.'},
            {'name': 'Rescue Medication Training',
             'description': 'Formal training on administering emergency anti-seizure medications (e.g., intranasal midazolam, rectal diazepam, buccal midazolam) during prolonged seizures.'},
            {'name': 'SUDEP Awareness',
             'description': 'Knowledge of Sudden Unexpected Death in Epilepsy, including risk factors, prevention strategies (nocturnal supervision, seizure control), and monitoring practices.'},
            {'name': 'Respite Care',
             'description': 'Temporary relief for primary caregivers, providing planned breaks to reduce burnout and maintain long-term caregiving capacity.'},
            {'name': 'First Aid Certification',
             'description': 'Completion of accredited seizure first aid training covering recognition, timing, positioning, airway management, and post-ictal care.'},
            {'name': 'Safety Plan',
             'description': 'A comprehensive environmental and behavioral safety plan addressing seizure-related hazards at home, work, school, and during activities.'},
        ],
        'quality_metrics': [
            {'name': 'Readiness Score',
             'description': 'Sum of 5 binary dimensions (training, first aid, rescue med, safety plan, action plan). Fully ready = 5/5, not ready = 0/5.'},
            {'name': 'Burnout Risk Threshold',
             'description': 'Burnout score above 60, stress above 7/10, or sleep quality below 4/10 triggers a risk alert requiring clinical intervention.'},
            {'name': 'Training Coverage',
             'description': 'Percentage of expected training topics completed by each caregiver against the standard curriculum of 8 core topics.'},
            {'name': 'Confidence Level',
             'description': 'Self-reported seizure first aid confidence on a 1-10 scale, expressed as a percentage. Target is 70% or above.'},
        ],
        'compliance': [
            {'ref': 'ILAE Caregiver Guidelines',
             'note': 'International League Against Epilepsy recommendations for caregiver education, competency assessment, and ongoing support programs.'},
            {'ref': 'Epilepsy Foundation Standards',
             'note': 'Epilepsy Foundation seizure first aid training standards, including recognition, response, and rescue medication certification.'},
            {'ref': 'NICE Guidelines for Epilepsy',
             'note': 'National Institute for Health and Care Excellence guidelines (CG137/NG217) covering caregiver information, training, and psychological support.'},
        ],
        'remediation': [
            {'strategy': 'Burnout Intervention',
             'description': 'Refer caregivers with burnout scores above 60 to psychosocial support services, peer support groups, and scheduled respite care within 2 weeks.'},
            {'strategy': 'Training Gap Closure',
             'description': 'Enroll caregivers missing core training topics in the next available epilepsy education session. Target 100% coverage within 90 days.'},
            {'strategy': 'Safety Plan Review',
             'description': 'Schedule safety plan creation or review for caregivers without active plans. Include home hazard assessment and emergency contact verification.'},
            {'strategy': 'Respite Scheduling',
             'description': 'Ensure all primary caregivers have respite care scheduled at least monthly. Flag caregivers with no respite in the past 60 days.'},
        ],
    }


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN ===')
    pprint.pprint(breakdown())
    print('\n=== DEFINITIONS ===')
    pprint.pprint(definitions())
