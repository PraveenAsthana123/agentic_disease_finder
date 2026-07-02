"""Research Coordinator Dashboard — study enrollment, protocol tracking,
cohort management, outcomes collection from clinical evaluations.

Maps clinical.db tables to research coordinator concepts:
- patients              -> enrolled subjects, demographics, disease distribution
- assessments           -> instruments administered per patient, completeness rate
- appointments          -> scheduled vs completed visits, compliance rate
- uploads + analyses    -> EEG submissions, analysis completion
- seizure_diary         -> seizure event frequency, severity distribution
- transaction_log       -> pipeline activity, data entry timeline
"""

import sqlite3
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Research Coordinator dashboard."""
    patients = _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients ORDER BY patient_id"
    )
    assessments = _db_query(
        "SELECT id, patient_id, instrument, score, max_score, interpretation, "
        "level, alert, examiner, created_at FROM assessments ORDER BY created_at"
    )
    appointments = _db_query(
        "SELECT id, patient_id, provider, department, appt_type, status, "
        "booked_at, scheduled_for, completed_at, duration_min, notes, created_at "
        "FROM appointments ORDER BY created_at"
    )
    seizure_events = _db_query(
        "SELECT id, patient_id, event_date, duration_sec, severity, location, "
        "trigger, created_at FROM seizure_diary ORDER BY event_date"
    )
    uploads = _db_query(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at"
    )
    analyses = _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, confidence, "
        "signal_quality, created_at FROM analyses ORDER BY created_at"
    )

    if not patients:
        return {
            'available': False,
            'message': 'No patient data available. Enroll subjects first.',
        }

    total_subjects = len(patients)
    total_assessments = len(assessments)
    total_visits = len(appointments)
    completed_visits = sum(1 for a in appointments if (a.get('status') or '').lower() == 'completed')
    visit_compliance_pct = round(completed_visits / total_visits * 100, 1) if total_visits else 0.0
    total_seizure_events = len(seizure_events)
    total_eeg_uploads = len(uploads)
    analyses_complete = len(analyses)

    # Distinct instruments used
    instruments_used = len(set(a['instrument'] for a in assessments if a.get('instrument')))

    # Date range across all tables
    all_dates = []
    for tbl in [patients, assessments, appointments, seizure_events, uploads, analyses]:
        for row in tbl:
            d = row.get('created_at') or row.get('event_date')
            if d:
                all_dates.append(str(d)[:10])
    date_range = {'earliest': min(all_dates), 'latest': max(all_dates)} if all_dates else {'earliest': None, 'latest': None}

    # Disease distribution
    disease_counts = Counter(p.get('disease', 'Unknown') for p in patients)
    disease_distribution = [
        {'disease': d, 'count': c}
        for d, c in disease_counts.most_common()
    ]

    # Enrollment by month
    enrollment_months = Counter()
    for p in patients:
        dt = (p.get('created_at') or '')[:7]
        if dt:
            enrollment_months[dt] += 1
    enrollment_by_month = [
        {'month': m, 'count': c}
        for m, c in sorted(enrollment_months.items())
    ]

    # Instrument coverage
    inst_patients = {}
    for a in assessments:
        inst = a.get('instrument')
        if inst:
            inst_patients.setdefault(inst, set()).add(a.get('patient_id'))
    inst_counts = Counter(a['instrument'] for a in assessments if a.get('instrument'))
    instrument_coverage = [
        {
            'instrument': inst,
            'count': inst_counts.get(inst, 0),
            'patients_assessed': len(inst_patients.get(inst, set())),
        }
        for inst in sorted(inst_counts.keys())
    ]

    # Visit status distribution
    status_counts = Counter((a.get('status') or 'unknown').lower() for a in appointments)
    visit_status_distribution = [
        {'status': s, 'count': c}
        for s, c in status_counts.most_common()
    ]

    return {
        'available': True,
        'total_subjects': total_subjects,
        'total_assessments': total_assessments,
        'total_visits': total_visits,
        'completed_visits': completed_visits,
        'visit_compliance_pct': visit_compliance_pct,
        'total_seizure_events': total_seizure_events,
        'total_eeg_uploads': total_eeg_uploads,
        'analyses_complete': analyses_complete,
        'instruments_used': instruments_used,
        'date_range': date_range,
        'disease_distribution': disease_distribution,
        'enrollment_by_month': enrollment_by_month,
        'instrument_coverage': instrument_coverage,
        'visit_status_distribution': visit_status_distribution,
        'kpis': [
            {'label': 'Enrolled Subjects', 'value': str(total_subjects)},
            {'label': 'Total Assessments', 'value': str(total_assessments)},
            {'label': 'Total Visits', 'value': str(total_visits)},
            {'label': 'Completed Visits', 'value': str(completed_visits)},
            {'label': 'Visit Compliance', 'value': f'{visit_compliance_pct}%',
             'color': '#10b981' if visit_compliance_pct >= 80 else '#f59e0b' if visit_compliance_pct >= 60 else '#ef4444'},
            {'label': 'Seizure Events', 'value': str(total_seizure_events),
             'color': '#ef4444' if total_seizure_events > 20 else '#f59e0b' if total_seizure_events > 5 else '#10b981'},
            {'label': 'EEG Uploads', 'value': str(total_eeg_uploads)},
            {'label': 'Analyses Complete', 'value': str(analyses_complete)},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed research coordinator breakdown — subject inventory, protocol
    matrix, visit log, seizure log, data submissions, daily activity, pipeline."""
    patients = _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients ORDER BY patient_id"
    )
    assessments = _db_query(
        "SELECT id, patient_id, instrument, score, max_score, interpretation, "
        "level, alert, examiner, created_at FROM assessments ORDER BY created_at"
    )
    appointments = _db_query(
        "SELECT id, patient_id, provider, department, appt_type, status, "
        "booked_at, scheduled_for, completed_at, duration_min, notes, created_at "
        "FROM appointments ORDER BY created_at"
    )
    seizure_events = _db_query(
        "SELECT id, patient_id, event_date, duration_sec, severity, location, "
        "trigger, created_at FROM seizure_diary ORDER BY event_date"
    )
    uploads = _db_query(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at"
    )
    analyses = _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, confidence, "
        "signal_quality, created_at FROM analyses ORDER BY created_at"
    )
    pipeline_raw = _db_query(
        "SELECT id, patient_id, component, action, actor, ref_id, detail, "
        "ts_utc, ts_local FROM transaction_log ORDER BY ts_utc"
    )

    if not patients:
        return {'available': False}

    # --- Subject inventory ---
    patient_assessments = Counter(a['patient_id'] for a in assessments)
    patient_visits = Counter(a['patient_id'] for a in appointments)
    patient_seizures = Counter(s['patient_id'] for s in seizure_events)
    patient_uploads = Counter(u['patient_id'] for u in uploads)

    subject_inventory = []
    for p in patients:
        pid = p['patient_id']
        subject_inventory.append({
            'patient_id': pid,
            'name': p.get('name', ''),
            'age': p.get('age'),
            'gender': p.get('gender'),
            'disease': p.get('disease'),
            'assessments_count': patient_assessments.get(pid, 0),
            'visits_count': patient_visits.get(pid, 0),
            'seizure_events': patient_seizures.get(pid, 0),
            'uploads': patient_uploads.get(pid, 0),
            'enrollment_date': (p.get('created_at') or '')[:10],
        })

    # --- Protocol matrix: per-patient per-instrument completion ---
    all_instruments = sorted(set(a['instrument'] for a in assessments if a.get('instrument')))
    patient_inst = {}
    for a in assessments:
        pid = a['patient_id']
        inst = a.get('instrument')
        if inst:
            patient_inst.setdefault(pid, Counter())[inst] += 1

    protocol_matrix = []
    for p in patients:
        pid = p['patient_id']
        row = {
            'patient_id': pid,
            'name': p.get('name', ''),
        }
        counts = patient_inst.get(pid, {})
        for inst in all_instruments:
            row[inst] = counts.get(inst, 0)
        protocol_matrix.append(row)

    # --- Visit log ---
    visit_log = [
        {
            'patient_id': a['patient_id'],
            'provider': a.get('provider'),
            'appt_type': a.get('appt_type'),
            'status': a.get('status'),
            'scheduled_for': a.get('scheduled_for'),
            'completed_at': a.get('completed_at'),
            'duration_min': a.get('duration_min'),
        }
        for a in appointments
    ]

    # --- Seizure log ---
    seizure_log = [
        {
            'patient_id': s['patient_id'],
            'event_date': s.get('event_date'),
            'duration_sec': s.get('duration_sec'),
            'severity': s.get('severity'),
            'location': s.get('location'),
            'trigger': s.get('trigger'),
        }
        for s in seizure_events
    ]

    # --- Data submissions: uploads + analyses join ---
    upload_map = {u['id']: u for u in uploads}
    data_submissions = []
    for an in analyses:
        uid = an.get('upload_id')
        u = upload_map.get(uid, {})
        data_submissions.append({
            'upload_id': uid,
            'patient_id': an.get('patient_id'),
            'file_name': u.get('file_name'),
            'disease': an.get('disease'),
            'predicted_label': an.get('predicted_label'),
            'confidence': an.get('confidence'),
            'signal_quality': an.get('signal_quality'),
        })
    # Include uploads without analyses
    analyzed_upload_ids = set(an.get('upload_id') for an in analyses)
    for u in uploads:
        if u['id'] not in analyzed_upload_ids:
            data_submissions.append({
                'upload_id': u['id'],
                'patient_id': u.get('patient_id'),
                'file_name': u.get('file_name'),
                'disease': u.get('disease'),
                'predicted_label': None,
                'confidence': None,
                'signal_quality': None,
            })

    # --- Daily activity ---
    daily_counts = Counter()
    for ev in pipeline_raw:
        day = (ev.get('ts_utc') or ev.get('ts_local') or '')[:10]
        if day:
            daily_counts[day] += 1
    daily_activity = [
        {'date': day, 'count': cnt}
        for day, cnt in sorted(daily_counts.items())
    ]

    # --- Pipeline events (last 50) ---
    pipeline_events = [
        {
            'id': ev['id'],
            'patient_id': ev.get('patient_id'),
            'component': ev.get('component'),
            'action': ev.get('action'),
            'actor': ev.get('actor'),
            'detail': ev.get('detail'),
            'ts_utc': ev.get('ts_utc'),
            'ts_local': ev.get('ts_local'),
        }
        for ev in pipeline_raw[-50:]
    ]

    return {
        'available': True,
        'subject_inventory': subject_inventory,
        'protocol_matrix': protocol_matrix,
        'visit_log': visit_log,
        'seizure_log': seizure_log,
        'data_submissions': data_submissions,
        'daily_activity': daily_activity,
        'pipeline_events': pipeline_events,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Research Coordinator dashboard."""
    return {
        'concepts': [
            {
                'name': 'Study Enrollment',
                'description': 'The process of formally registering eligible patients '
                               'as study subjects. Includes informed consent, eligibility '
                               'screening, demographic capture, and assignment to study '
                               'cohort. Enrollment rate is a key feasibility metric for '
                               'clinical epilepsy research.',
            },
            {
                'name': 'Protocol Compliance',
                'description': 'Adherence to the predefined study protocol including '
                               'required assessments, visit schedules, and data collection '
                               'procedures. Non-compliance may result in missing data, '
                               'protocol deviations, or subject exclusion from analysis.',
            },
            {
                'name': 'Cohort Management',
                'description': 'Organization and tracking of study subjects grouped by '
                               'disease type, treatment arm, demographics, or enrollment '
                               'period. Effective cohort management ensures balanced groups '
                               'and supports stratified analysis of outcomes.',
            },
            {
                'name': 'Outcomes Collection',
                'description': 'Systematic gathering of primary and secondary endpoint '
                               'data including seizure frequency, cognitive scores, quality '
                               'of life measures, and adverse events. Completeness and '
                               'accuracy of outcomes data directly impacts study validity.',
            },
            {
                'name': 'Data Completeness',
                'description': 'The proportion of expected data points that have been '
                               'successfully collected and recorded. Incomplete data can '
                               'introduce bias and reduce statistical power. Target '
                               'completeness for clinical trials is typically >95%.',
            },
            {
                'name': 'Visit Compliance',
                'description': 'The ratio of completed study visits to scheduled visits. '
                               'Low visit compliance indicates retention issues and may '
                               'result in protocol deviations. Compliance <80% triggers '
                               'subject retention interventions.',
            },
            {
                'name': 'Adverse Event Monitoring',
                'description': 'Continuous surveillance for adverse events (AEs) and '
                               'serious adverse events (SAEs) during the study period. '
                               'In epilepsy research, seizure events, medication side '
                               'effects, and cognitive decline are monitored as AEs.',
            },
            {
                'name': 'Data Quality Assurance',
                'description': 'Procedures to verify accuracy, consistency, and integrity '
                               'of collected study data. Includes source data verification, '
                               'query resolution, range checks, and audit trails. Essential '
                               'for regulatory submission readiness.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Data Completeness Rate',
                'description': 'Percentage of required data fields that have been captured '
                               'across all subjects and visits. Calculated as collected data '
                               'points divided by expected data points. Target >= 95% for '
                               'regulatory-grade studies.',
            },
            {
                'name': 'Visit Compliance Rate',
                'description': 'Percentage of scheduled study visits that were completed. '
                               'Completed visits divided by total scheduled visits. Rates '
                               'below 80% may compromise study integrity and require '
                               'corrective action plans.',
            },
            {
                'name': 'Instrument Coverage',
                'description': 'Number of distinct assessment instruments administered '
                               'per subject relative to the protocol-specified battery. '
                               'Full coverage ensures comprehensive neuropsychological '
                               'profiling for each enrolled subject.',
            },
            {
                'name': 'Follow-up Rate',
                'description': 'Percentage of enrolled subjects who remain active in the '
                               'study through the planned follow-up period. Attrition '
                               'above 20% may introduce selection bias and affect the '
                               'generalizability of study findings.',
            },
        ],
        'study_phases': [
            {
                'name': 'Screening',
                'description': 'Initial phase where potential subjects are evaluated '
                               'against inclusion/exclusion criteria. Includes medical '
                               'history review, baseline EEG, neurological examination, '
                               'and informed consent process.',
            },
            {
                'name': 'Baseline',
                'description': 'Pre-intervention data collection phase establishing '
                               'reference measurements. Comprehensive neuropsychological '
                               'battery (MoCA, MMSE, WAIS, PHQ-9), baseline seizure '
                               'frequency, and quality of life assessments.',
            },
            {
                'name': 'Treatment',
                'description': 'Active intervention phase with ongoing monitoring. '
                               'Subjects receive study treatment while seizure diaries, '
                               'cognitive assessments, and safety labs are collected at '
                               'protocol-defined intervals.',
            },
            {
                'name': 'Follow-up',
                'description': 'Post-treatment monitoring phase to assess durability '
                               'of outcomes. Repeated assessments at 3, 6, and 12 months. '
                               'Critical for detecting delayed treatment effects and '
                               'long-term seizure control.',
            },
            {
                'name': 'Close-out',
                'description': 'Final study phase including last visit, data lock, '
                               'query resolution, and database freeze. All outstanding '
                               'data queries must be resolved before statistical analysis '
                               'can begin.',
            },
        ],
        'compliance_refs': [
            {
                'name': 'ICH-GCP E6(R2)',
                'url': 'https://www.ich.org/page/efficacy-guidelines#6',
                'scope': 'International standard for the design, conduct, recording, '
                         'and reporting of clinical trials. Ensures data credibility '
                         'and subject protection. Requires documented procedures for '
                         'data handling, protocol deviations, and monitoring.',
            },
            {
                'name': 'FDA 21 CFR Part 11',
                'url': 'https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11',
                'scope': 'Regulations for electronic records and electronic signatures. '
                         'Requires audit trails, access controls, and validation of '
                         'computerized systems used in clinical data collection.',
            },
            {
                'name': 'HIPAA',
                'url': 'https://www.hhs.gov/hipaa/index.html',
                'scope': 'Health Insurance Portability and Accountability Act governing '
                         'protected health information. Research use requires IRB-approved '
                         'consent or HIPAA authorization. De-identification standards '
                         'apply to data sharing.',
            },
            {
                'name': 'IRB/Ethics',
                'url': 'https://www.hhs.gov/ohrp/regulations-and-policy/regulations/45-cfr-46/index.html',
                'scope': 'Institutional Review Board oversight ensuring ethical conduct '
                         'of research involving human subjects. Protocol amendments, '
                         'adverse events, and annual renewals require IRB review and '
                         'approval (45 CFR 46).',
            },
            {
                'name': 'CDISC/CDASH',
                'url': 'https://www.cdisc.org/standards/foundational/cdash',
                'scope': 'Clinical Data Acquisition Standards Harmonization for '
                         'consistent data collection across clinical trials. Defines '
                         'standard data collection fields for demographics, adverse '
                         'events, concomitant medications, and disposition.',
            },
        ],
        'remediation': [
            {
                'strategy': 'Missing Data Recovery',
                'description': 'Identify subjects with incomplete assessment batteries '
                               'and schedule targeted data collection visits. Prioritize '
                               'primary endpoint data and time-sensitive assessments '
                               'that cannot be retrospectively captured.',
            },
            {
                'strategy': 'Visit Compliance Intervention',
                'description': 'Implement automated appointment reminders, transportation '
                               'assistance, and flexible scheduling for subjects at risk '
                               'of non-compliance. Escalate to principal investigator '
                               'when compliance falls below 70%.',
            },
            {
                'strategy': 'Protocol Deviation Management',
                'description': 'Document all protocol deviations with root cause analysis. '
                               'Classify as major (affects subject safety or data integrity) '
                               'or minor (administrative). Implement corrective and '
                               'preventive actions (CAPA) for recurrent deviations.',
            },
            {
                'strategy': 'Data Quality Audit',
                'description': 'Conduct periodic source data verification comparing '
                               'electronic records against source documents. Resolve '
                               'data queries within 5 business days. Target query rate '
                               'below 2% of total data points entered.',
            },
        ],
    }
