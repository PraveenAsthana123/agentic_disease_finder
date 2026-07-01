"""Analytics AI Dashboard — clinical analytics overview and breakdowns
for an epilepsy EEG clinical decision support system.

Queries real data from data/clinical.db:
- patients (40 rows): demographics, disease, department
- analyses (21 rows): EEG analysis results with confidence and signal quality
- assessments (423 rows): neuropsychological instrument scores
- seizure_diary (25 rows): seizure events with severity and triggers
- medications (9 rows): prescribed anti-epileptic drugs
- appointments (120 rows): scheduling, status, provider workload
- uploads (21 rows): EEG file uploads
- transaction_log (633 rows): system activity audit trail
- mri_findings (40 rows): MRI scan results
- finops_costs (978 rows): operational cost tracking
"""

import sqlite3
import json
import math
import os
from datetime import datetime, timezone
from collections import defaultdict, Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    """Return ISO-8601 string from a datetime."""
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt) if dt else None


def _safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 2) if vals else 0


# ── Data loaders ──────────────────────────────────────────────────────────

def _load_patients(cur):
    rows = cur.execute(
        'SELECT patient_id, name, age, gender, disease, department, created_at '
        'FROM patients'
    ).fetchall()
    patients = []
    for r in rows:
        patients.append({
            'patient_id': r[0],
            'name': r[1] if r[1] else 'Unknown',
            'age': r[2],
            'gender': (r[3] or '').strip() or 'Unspecified',
            'disease': r[4] or 'epilepsy',
            'department': r[5] or 'General',
            'created_at': r[6],
        })
    return patients


def _load_analyses(cur):
    rows = cur.execute(
        'SELECT id, upload_id, patient_id, disease, predicted_label, confidence, '
        'signal_quality, created_at FROM analyses'
    ).fetchall()
    analyses = []
    for r in rows:
        analyses.append({
            'id': r[0], 'upload_id': r[1], 'patient_id': r[2],
            'disease': r[3], 'predicted_label': r[4],
            'confidence': r[5], 'signal_quality': r[6],
            'created_at': r[7],
        })
    return analyses


def _load_assessments(cur):
    rows = cur.execute(
        'SELECT id, patient_id, instrument, score, max_score, interpretation, '
        'level, created_at FROM assessments'
    ).fetchall()
    assessments = []
    for r in rows:
        assessments.append({
            'id': r[0], 'patient_id': r[1], 'instrument': r[2],
            'score': r[3], 'max_score': r[4],
            'interpretation': r[5], 'level': r[6],
            'created_at': r[7],
        })
    return assessments


def _load_seizures(cur):
    rows = cur.execute(
        'SELECT id, patient_id, event_date, duration_sec, severity, '
        'injury, er_visit, rescue_med, trigger, created_at '
        'FROM seizure_diary'
    ).fetchall()
    seizures = []
    for r in rows:
        seizures.append({
            'id': r[0], 'patient_id': r[1], 'event_date': r[2],
            'duration_sec': r[3], 'severity': r[4],
            'injury': r[5], 'er_visit': r[6], 'rescue_med': r[7],
            'trigger': r[8], 'created_at': r[9],
        })
    return seizures


def _load_medications(cur):
    rows = cur.execute(
        'SELECT id, patient_id, fields_json, created_at FROM medications'
    ).fetchall()
    medications = []
    for r in rows:
        try:
            fj = json.loads(r[2]) if r[2] else {}
        except (json.JSONDecodeError, TypeError):
            fj = {}
        fj['_id'] = r[0]
        fj['_patient_id'] = r[1]
        fj['_created_at'] = r[3]
        medications.append(fj)
    return medications


def _load_appointments(cur):
    rows = cur.execute(
        'SELECT id, patient_id, provider, department, appt_type, status, '
        'booked_at, scheduled_for, completed_at, duration_min, created_at '
        'FROM appointments'
    ).fetchall()
    appointments = []
    for r in rows:
        appointments.append({
            'id': r[0], 'patient_id': r[1], 'provider': r[2],
            'department': r[3], 'appt_type': r[4], 'status': r[5],
            'booked_at': r[6], 'scheduled_for': r[7],
            'completed_at': r[8], 'duration_min': r[9],
            'created_at': r[10],
        })
    return appointments


# ══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

def analytics_overview():
    """Return overview dict with KPIs, demographics, activity timeline,
    disease distribution, department workload, and signal quality."""
    try:
        conn = _conn()
        cur = conn.cursor()
        patients = _load_patients(cur)
        analyses = _load_analyses(cur)
        assessments = _load_assessments(cur)
        seizures = _load_seizures(cur)
        appointments = _load_appointments(cur)
        conn.close()
    except Exception as e:
        return {'error': f'Database query failed: {str(e)}'}

    total_patients = len(patients)
    total_analyses = len(analyses)
    total_assessments = len(assessments)
    total_seizures = len(seizures)

    # ── demographics ─────────────────────────────────────────────────
    ages = [p['age'] for p in patients if p['age'] is not None]
    avg_age = _safe_mean(ages)

    gender_counts = Counter(p['gender'] for p in patients)
    gender_distribution = [
        {'gender': g, 'count': c, 'pct': round(100 * c / total_patients, 1) if total_patients else 0}
        for g, c in gender_counts.most_common()
    ]

    disease_counts = Counter(p['disease'] for p in patients)
    disease_distribution = [
        {'disease': d, 'count': c, 'pct': round(100 * c / total_patients, 1) if total_patients else 0}
        for d, c in disease_counts.most_common()
    ]

    department_counts = Counter(p['department'] for p in patients)
    active_departments = len(department_counts)
    department_workload = [
        {'department': d, 'patient_count': c}
        for d, c in department_counts.most_common()
    ]

    # ── age histogram bins ───────────────────────────────────────────
    age_bins = {'0-17': 0, '18-30': 0, '31-45': 0, '46-60': 0, '61-75': 0, '76+': 0}
    for a in ages:
        if a < 18:
            age_bins['0-17'] += 1
        elif a <= 30:
            age_bins['18-30'] += 1
        elif a <= 45:
            age_bins['31-45'] += 1
        elif a <= 60:
            age_bins['46-60'] += 1
        elif a <= 75:
            age_bins['61-75'] += 1
        else:
            age_bins['76+'] += 1
    age_histogram = [{'bin': b, 'count': c} for b, c in age_bins.items()]

    # ── clinical activity timeline (by month) ────────────────────────
    monthly_activity = defaultdict(lambda: {'analyses': 0, 'assessments': 0})
    for a in analyses:
        if a.get('created_at'):
            month = a['created_at'][:7]  # YYYY-MM
            monthly_activity[month]['analyses'] += 1
    for a in assessments:
        if a.get('created_at'):
            month = a['created_at'][:7]
            monthly_activity[month]['assessments'] += 1
    activity_timeline = sorted([
        {'month': m, 'analyses': v['analyses'], 'assessments': v['assessments'],
         'total': v['analyses'] + v['assessments']}
        for m, v in monthly_activity.items()
    ], key=lambda x: x['month'])

    # ── disease distribution as pie chart data ───────────────────────
    disease_pie = [
        {'label': d, 'value': c}
        for d, c in disease_counts.most_common()
    ]

    # ── signal quality distribution ──────────────────────────────────
    sq_values = [a['signal_quality'] for a in analyses
                 if a.get('signal_quality') is not None]
    sq_counts = Counter()
    for sq in sq_values:
        if isinstance(sq, str):
            sq_counts[sq] += 1
        elif isinstance(sq, (int, float)):
            if sq >= 0.8:
                sq_counts['high'] += 1
            elif sq >= 0.5:
                sq_counts['medium'] += 1
            else:
                sq_counts['low'] += 1
    signal_quality_distribution = [
        {'quality': q, 'count': c, 'pct': round(100 * c / len(sq_values), 1) if sq_values else 0}
        for q, c in sq_counts.most_common()
    ]

    # ── KPI cards ────────────────────────────────────────────────────
    kpi_cards = [
        {'label': 'Total Patients', 'value': total_patients, 'unit': 'patients', 'icon': 'users'},
        {'label': 'Total Analyses', 'value': total_analyses, 'unit': 'analyses', 'icon': 'activity'},
        {'label': 'Total Assessments', 'value': total_assessments, 'unit': 'assessments', 'icon': 'clipboard'},
        {'label': 'Seizure Events', 'value': total_seizures, 'unit': 'events', 'icon': 'zap'},
        {'label': 'Avg Patient Age', 'value': avg_age, 'unit': 'years', 'icon': 'calendar'},
        {'label': 'Active Departments', 'value': active_departments, 'unit': 'depts', 'icon': 'briefcase'},
        {'label': 'Avg Confidence', 'value': _safe_mean([a['confidence'] for a in analyses if a.get('confidence')]),
         'unit': '%', 'icon': 'percent'},
        {'label': 'Total Appointments', 'value': len(appointments), 'unit': 'appts', 'icon': 'calendar'},
    ]

    # gender ratio string for KPI display
    m = sum(1 for p in patients if (p.get('gender') or '').lower() in ('m', 'male'))
    f = sum(1 for p in patients if (p.get('gender') or '').lower() in ('f', 'female'))
    gender_ratio = f'{m}M / {f}F' if (m + f) else 'N/A'

    # gender breakdown for pie chart
    gender_breakdown = [{'name': g['gender'], 'value': g['count']} for g in gender_distribution]

    # disease-by-gender cross-tab
    dxg = defaultdict(lambda: defaultdict(int))
    for p in patients:
        dxg[p.get('disease', 'unknown')][p.get('gender', 'Unknown')] += 1
    all_genders = sorted({p.get('gender', 'Unknown') for p in patients})
    disease_by_gender = [
        {**{'disease': d}, **{g: dxg[d].get(g, 0) for g in all_genders}}
        for d in sorted(dxg.keys())
    ]

    return {
        'generated_at': _iso(_now()),
        'total_patients': total_patients,
        'total_analyses': total_analyses,
        'total_assessments': total_assessments,
        'total_seizure_events': total_seizures,
        'seizure_events': total_seizures,
        'avg_patient_age': avg_age,
        'avg_age': avg_age,
        'gender_ratio': gender_ratio,
        'diseases_tracked': len(disease_counts),
        'active_departments': active_departments,
        'kpi_cards': kpi_cards,
        'gender_distribution': gender_distribution,
        'gender_breakdown': gender_breakdown,
        'disease_distribution': disease_distribution,
        'disease_pie_chart': disease_pie,
        'disease_by_gender': disease_by_gender,
        'age_histogram': age_histogram,
        'department_workload': department_workload,
        'activity_timeline': activity_timeline,
        'signal_quality_distribution': signal_quality_distribution,
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════

def analytics_breakdown():
    """Return detailed per-patient summaries, instrument distribution,
    seizure severity, appointment status, medication coverage, and
    monthly activity trends."""
    try:
        conn = _conn()
        cur = conn.cursor()
        patients = _load_patients(cur)
        analyses = _load_analyses(cur)
        assessments = _load_assessments(cur)
        seizures = _load_seizures(cur)
        medications = _load_medications(cur)
        appointments = _load_appointments(cur)
        conn.close()
    except Exception as e:
        return {'error': f'Database query failed: {str(e)}'}

    # Build lookup dicts keyed by patient_id
    analyses_by_pt = defaultdict(list)
    for a in analyses:
        analyses_by_pt[a['patient_id']].append(a)

    assessments_by_pt = defaultdict(list)
    for a in assessments:
        assessments_by_pt[a['patient_id']].append(a)

    seizures_by_pt = defaultdict(list)
    for s in seizures:
        seizures_by_pt[s['patient_id']].append(s)

    meds_by_pt = defaultdict(list)
    for m in medications:
        meds_by_pt[m['_patient_id']].append(m)

    appts_by_pt = defaultdict(list)
    for a in appointments:
        appts_by_pt[a['patient_id']].append(a)

    # ── per-patient summary ──────────────────────────────────────────
    per_patient_summary = []
    for p in patients:
        pid = p['patient_id']
        per_patient_summary.append({
            'patient_id': pid,
            'name': p['name'],
            'age': p['age'],
            'gender': p['gender'],
            'disease': p['disease'],
            'num_analyses': len(analyses_by_pt.get(pid, [])),
            'num_assessments': len(assessments_by_pt.get(pid, [])),
            'num_seizures': len(seizures_by_pt.get(pid, [])),
            'num_medications': len(meds_by_pt.get(pid, [])),
            'num_appointments': len(appts_by_pt.get(pid, [])),
        })
    per_patient_summary.sort(
        key=lambda x: x['num_analyses'] + x['num_assessments'] + x['num_seizures'],
        reverse=True,
    )

    # ── assessment instrument distribution ───────────────────────────
    instrument_counts = Counter(a['instrument'] for a in assessments if a.get('instrument'))
    instrument_distribution = [
        {'instrument': instr, 'count': c}
        for instr, c in instrument_counts.most_common()
    ]

    # ── top assessment instruments by usage ──────────────────────────
    top_instruments = [
        {'instrument': instr, 'count': c, 'pct': round(100 * c / len(assessments), 1) if assessments else 0}
        for instr, c in instrument_counts.most_common(10)
    ]

    # ── seizure severity distribution ────────────────────────────────
    severity_counts = Counter(s['severity'] for s in seizures if s.get('severity'))
    seizure_severity_distribution = [
        {'severity': sev, 'count': c, 'pct': round(100 * c / len(seizures), 1) if seizures else 0}
        for sev, c in severity_counts.most_common()
    ]

    # ── appointment status breakdown ─────────────────────────────────
    status_counts = Counter(a['status'] for a in appointments if a.get('status'))
    appointment_status_breakdown = [
        {'status': s, 'count': c, 'pct': round(100 * c / len(appointments), 1) if appointments else 0}
        for s, c in status_counts.most_common()
    ]

    # ── medication coverage ──────────────────────────────────────────
    patients_with_meds = len(set(m['_patient_id'] for m in medications))
    patients_without_meds = len(patients) - patients_with_meds
    medication_coverage = [
        {'name': 'With Medications', 'value': patients_with_meds},
        {'name': 'Without Medications', 'value': patients_without_meds},
    ]

    # ── monthly activity trend (combined) ────────────────────────────
    monthly = defaultdict(lambda: {'analyses': 0, 'assessments': 0, 'seizures': 0})
    for a in analyses:
        if a.get('created_at'):
            month = a['created_at'][:7]
            monthly[month]['analyses'] += 1
    for a in assessments:
        if a.get('created_at'):
            month = a['created_at'][:7]
            monthly[month]['assessments'] += 1
    for s in seizures:
        if s.get('created_at'):
            month = s['created_at'][:7]
            monthly[month]['seizures'] += 1
        elif s.get('event_date'):
            month = s['event_date'][:7]
            monthly[month]['seizures'] += 1

    monthly_activity_trend = sorted([
        {'month': m, 'analyses': v['analyses'], 'assessments': v['assessments'],
         'seizures': v['seizures'],
         'total': v['analyses'] + v['assessments'] + v['seizures']}
        for m, v in monthly.items()
    ], key=lambda x: x['month'])

    return {
        'generated_at': _iso(_now()),
        'per_patient_summary': per_patient_summary,
        'patient_summaries': per_patient_summary,
        'instrument_distribution': instrument_distribution,
        'top_instruments': top_instruments,
        'seizure_severity_distribution': seizure_severity_distribution,
        'seizure_severity': seizure_severity_distribution,
        'appointment_status_breakdown': appointment_status_breakdown,
        'appointment_status': appointment_status_breakdown,
        'medication_coverage': medication_coverage,
        'monthly_activity_trend': monthly_activity_trend,
        'monthly_trend': monthly_activity_trend,
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

def definitions():
    """Return Analytics AI definitions, clinical metrics, data quality
    indicators, regulatory references, and remediation strategies."""
    return {'sections': [
        {
            'title': 'Analytics Concepts',
            'items': [
                {
                    'term': 'Descriptive Analytics',
                    'definition': 'Statistical summarisation of historical clinical data '
                                  'to understand what has occurred. Includes counts, '
                                  'distributions, and trends across patients, analyses, '
                                  'assessments, and seizure events.',
                },
                {
                    'term': 'Key Performance Indicator (KPI)',
                    'definition': 'A quantifiable measure used to evaluate the clinical '
                                  'platform performance. Examples: total patients enrolled, '
                                  'average EEG analysis confidence, assessment completion '
                                  'rate, seizure event frequency.',
                },
                {
                    'term': 'Cohort Analysis',
                    'definition': 'Grouping patients by shared characteristics (disease, '
                                  'age range, gender, department) and comparing clinical '
                                  'outcomes and activity patterns across cohorts to identify '
                                  'disparities or trends.',
                },
                {
                    'term': 'Activity Timeline',
                    'definition': 'A chronological aggregation of clinical events (analyses, '
                                  'assessments, seizure entries) by month, enabling trend '
                                  'detection and workload forecasting.',
                },
                {
                    'term': 'Department Workload',
                    'definition': 'Distribution of patient volume and clinical activity '
                                  'across hospital departments, used for resource allocation '
                                  'and capacity planning.',
                },
            ],
        },
        {
            'title': 'Clinical Metrics',
            'items': [
                {
                    'term': 'Assessment Instrument',
                    'definition': 'A standardised clinical questionnaire or test used to '
                                  'evaluate a specific domain of patient health. Examples '
                                  'include PHQ-9 (depression), GAD-7 (anxiety), MoCA '
                                  '(cognitive screening), QOLIE-31 (epilepsy quality of '
                                  'life), NDDIE (epilepsy-specific depression), WAIS '
                                  '(intelligence), and Digit Span (working memory).',
                },
                {
                    'term': 'Seizure Severity Scale',
                    'definition': 'Classification of seizure events by clinical impact: '
                                  'Mild (brief, self-limiting, no injury), Moderate '
                                  '(prolonged or with post-ictal symptoms), Severe (injury, '
                                  'ER visit, rescue medication required). Informs treatment '
                                  'escalation and urgency of follow-up.',
                },
                {
                    'term': 'Signal Quality',
                    'definition': 'A measure of the fidelity and reliability of the EEG '
                                  'recording used for analysis. High signal quality indicates '
                                  'clean, artefact-free data suitable for automated '
                                  'classification. Low quality may indicate electrode issues, '
                                  'muscle artefact, or environmental noise.',
                },
                {
                    'term': 'Prediction Confidence',
                    'definition': 'The probability score (0-100%) assigned by the AI model '
                                  'to its predicted diagnosis label. Higher confidence '
                                  'indicates greater model certainty. Clinical review is '
                                  'recommended for predictions below 70% confidence.',
                },
                {
                    'term': 'Medication Coverage',
                    'definition': 'The proportion of patients with at least one recorded '
                                  'medication prescription. Low coverage may indicate '
                                  'documentation gaps rather than untreated patients, '
                                  'warranting a data completeness review.',
                },
            ],
        },
        {
            'title': 'Data Quality Indicators',
            'items': [
                {
                    'term': 'Completeness',
                    'definition': 'The proportion of expected data fields that are populated '
                                  'with valid values. Measured per table and per patient. '
                                  'Null or missing critical fields (age, disease, signal '
                                  'quality) reduce data completeness and may affect analytics '
                                  'accuracy.',
                },
                {
                    'term': 'Coverage',
                    'definition': 'The proportion of patients represented in each clinical '
                                  'data source (analyses, assessments, medications, seizure '
                                  'diary). Low coverage indicates underutilisation of a '
                                  'clinical tool or documentation workflow gaps.',
                },
                {
                    'term': 'Timeliness',
                    'definition': 'The recency of data entries relative to the current date. '
                                  'Stale data (no new entries for extended periods) may '
                                  'indicate workflow disruption, system downtime, or patient '
                                  'disengagement. Monitored via the activity timeline.',
                },
                {
                    'term': 'Consistency',
                    'definition': 'Agreement between related data fields across tables. '
                                  'Inconsistencies (e.g., patient disease in patients table '
                                  'differs from disease in analyses table) indicate data '
                                  'integrity issues requiring reconciliation.',
                },
                {
                    'term': 'Validity',
                    'definition': 'Data values conform to expected types, ranges, and '
                                  'formats. Examples: age within 0-120, confidence within '
                                  '0-100, severity in the allowed enum set. Invalid values '
                                  'are flagged for correction.',
                },
            ],
        },
        {
            'title': 'Clinical Relevance and Regulatory Standards',
            'items': [
                {
                    'standard': 'IEC 62304',
                    'relevance': 'Analytics AI is classified as a software item within the '
                                 'medical device software lifecycle. Data aggregation, '
                                 'statistical calculations, and KPI generation must comply '
                                 'with software development process requirements. Traceability '
                                 'from raw data to computed KPI is required (Class B minimum).',
                },
                {
                    'standard': 'FDA AI/ML PCCP',
                    'relevance': 'Predetermined Change Control Plan: analytics algorithms that '
                                 'compute KPIs, trends, or alerts from clinical data must '
                                 'document change boundaries. Updates to aggregation logic, '
                                 'bin definitions, or quality thresholds require validation '
                                 'before deployment.',
                },
                {
                    'standard': 'ILAE Classification',
                    'relevance': 'International League Against Epilepsy classification guides '
                                 'seizure categorisation and disease taxonomy used in the '
                                 'analytics module. Seizure severity, type, and syndrome '
                                 'classifications follow ILAE 2017 operational definitions.',
                },
                {
                    'standard': 'ISO 14971',
                    'relevance': 'Risk management for medical devices. Analytics errors '
                                 '(miscounted seizures, incorrect severity distribution, '
                                 'wrong confidence averages) are identified hazards. '
                                 'Mitigations include data validation, unit testing of '
                                 'aggregation functions, and audit trails.',
                },
                {
                    'standard': 'EU AI Act',
                    'relevance': 'Clinical analytics AI is a high-risk AI system under the '
                                 'EU AI Act. Requirements include transparency (methodology '
                                 'for KPI computation documented and accessible), data '
                                 'governance (clinical data quality monitoring), and human '
                                 'oversight (clinician review of analytics outputs before '
                                 'clinical decisions).',
                },
            ],
        },
        {
            'title': 'Remediation Strategies',
            'items': [
                {
                    'scenario': 'Data Gaps',
                    'strategy': 'Identify patients or time periods with missing clinical '
                                'records (no analyses, no assessments, no seizure diary '
                                'entries). Generate data completeness reports highlighting '
                                'gaps. Trigger clinical workflow reminders for incomplete '
                                'patient records. Escalate persistent gaps to department leads.',
                },
                {
                    'scenario': 'Quality Improvement',
                    'strategy': 'Monitor signal quality distribution over time. If the '
                                'proportion of low-quality EEG recordings increases, '
                                'investigate equipment calibration, electrode placement '
                                'protocols, and technician training. Set quality thresholds '
                                'and alert when breached.',
                },
                {
                    'scenario': 'Demographic Imbalance',
                    'strategy': 'Review gender and age distributions for bias. If certain '
                                'demographic groups are underrepresented, assess whether '
                                'this reflects true patient population or selection bias. '
                                'Adjust recruitment and outreach strategies. Document '
                                'demographic representation in regulatory submissions.',
                },
                {
                    'scenario': 'Activity Decline',
                    'strategy': 'Monitor monthly activity trends for sustained declines in '
                                'analyses, assessments, or seizure diary entries. Investigate '
                                'root causes: system downtime, workflow changes, staff '
                                'turnover, patient attrition. Implement corrective actions '
                                'and track recovery in subsequent months.',
                },
                {
                    'scenario': 'KPI Anomaly',
                    'strategy': 'Set control limits for critical KPIs (average confidence, '
                                'seizure frequency, assessment completion rate). When a KPI '
                                'deviates beyond 2 standard deviations from baseline, trigger '
                                'an investigation. Root cause analysis documented in CAPA '
                                '(Corrective and Preventive Action) system.',
                },
            ],
        },
    ]}


# ══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    funcs = {
        'overview': analytics_overview,
        'breakdown': analytics_breakdown,
        'definitions': definitions,
    }

    target = sys.argv[1] if len(sys.argv) > 1 else 'overview'
    if target not in funcs:
        print(f"Usage: python {sys.argv[0]} [overview|breakdown|definitions]")
        sys.exit(1)

    result = funcs[target]()
    print(json.dumps(result, indent=2, default=str))
