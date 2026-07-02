"""Patient Reporting Dashboard — scheduled patient monitoring summaries,
report generation metrics, assessment coverage, appointment tracking,
seizure diary stats from clinical.db.

Maps clinical.db tables to patient reporting concepts:
- patients (40 rows)        -> patient demographics, report coverage
- assessments (423 rows)    -> instrument-based clinical assessments
- appointments (120 rows)   -> appointment schedule, status tracking
- seizure_diary (25 rows)   -> seizure event log for diary reports
- medications (9 rows)      -> medication records for med reports
- mri_findings (40 rows)    -> imaging findings for imaging reports
- analyses (21 rows)        -> EEG analysis results
- transaction_log (682 rows)-> pipeline events, audit trail
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


def _avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_patients():
    return _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients ORDER BY patient_id"
    )


def _load_assessments():
    return _db_query(
        "SELECT id, patient_id, instrument, score, max_score, "
        "interpretation, level, alert, examiner, created_at "
        "FROM assessments ORDER BY created_at"
    )


def _load_appointments():
    return _db_query(
        "SELECT id, patient_id, provider, department, appt_type, status, "
        "booked_at, scheduled_for, completed_at, duration_min, notes, created_at "
        "FROM appointments ORDER BY scheduled_for"
    )


def _load_seizure_diary():
    return _db_query(
        "SELECT * FROM seizure_diary ORDER BY event_date"
    )


def _load_medications():
    return _db_query(
        "SELECT id, patient_id, fields_json, created_at "
        "FROM medications ORDER BY created_at"
    )


def _load_mri_findings():
    return _db_query(
        "SELECT id, patient_id, fields_json, created_at "
        "FROM mri_findings ORDER BY created_at"
    )


def _load_analyses():
    return _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, confidence, "
        "signal_quality, report_path, created_at "
        "FROM analyses ORDER BY created_at"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, patient_id, component, action, actor, ref_id, detail, "
        "ts_utc FROM transaction_log ORDER BY ts_utc"
    )


SEVERITY_COLORS = {
    'normal': '#10b981',
    'mild': '#f59e0b',
    'moderate': '#f97316',
    'severe': '#ef4444',
}


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Patient Reporting dashboard."""
    patients = _load_patients()
    assessments = _load_assessments()
    appointments = _load_appointments()
    seizures = _load_seizure_diary()
    medications = _load_medications()
    mri = _load_mri_findings()
    analyses = _load_analyses()
    pipeline = _load_pipeline_events()

    if not patients:
        return {
            'available': False,
            'message': 'No patient data available. Import patient records first.',
        }

    total_patients = len(patients)
    total_assessments = len(assessments)
    total_appointments = len(appointments)
    total_seizures = len(seizures)
    total_medications = len(medications)
    total_mri = len(mri)
    total_analyses = len(analyses)
    pipeline_count = len(pipeline)

    # Report coverage: patients who have at least one assessment
    patients_with_assessments = len(set(a['patient_id'] for a in assessments if a.get('patient_id')))
    report_coverage = round(patients_with_assessments / total_patients, 4) if total_patients else 0

    # Total possible reports = patients * 5 report types (assessment, appointment, seizure, med, imaging)
    # Reports generated = count of patients who have data in each category
    patient_data = {}
    for p in patients:
        pid = p['patient_id']
        patient_data[pid] = {
            'assessments': 0, 'appointments': 0, 'seizures': 0,
            'medications': 0, 'mri': 0, 'analyses': 0,
        }
    for a in assessments:
        pid = a.get('patient_id')
        if pid in patient_data:
            patient_data[pid]['assessments'] += 1
    for a in appointments:
        pid = a.get('patient_id')
        if pid in patient_data:
            patient_data[pid]['appointments'] += 1
    for s in seizures:
        pid = s.get('patient_id')
        if pid in patient_data:
            patient_data[pid]['seizures'] += 1
    for m in medications:
        pid = m.get('patient_id')
        if pid in patient_data:
            patient_data[pid]['medications'] += 1
    for m in mri:
        pid = m.get('patient_id')
        if pid in patient_data:
            patient_data[pid]['mri'] += 1
    for an in analyses:
        pid = an.get('patient_id')
        if pid in patient_data:
            patient_data[pid]['analyses'] += 1

    # Reports generated: each patient-category pair with data = 1 report
    reports_generated = sum(
        sum(1 for v in d.values() if v > 0)
        for d in patient_data.values()
    )
    reports_possible = total_patients * 6  # 6 categories

    # Instrument distribution
    instrument_counts = Counter(a['instrument'] for a in assessments if a.get('instrument'))
    instrument_distribution = [
        {'instrument': inst, 'count': cnt}
        for inst, cnt in sorted(instrument_counts.items(), key=lambda x: -x[1])
    ]

    # Appointment status distribution
    status_counts = Counter(a.get('status', 'unknown') for a in appointments)
    appt_status_distribution = [
        {'status': st, 'count': cnt}
        for st, cnt in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    # Severity distribution across assessments
    level_counts = Counter(a.get('level', 'unknown') for a in assessments if a.get('level'))
    severity_order = ['normal', 'mild', 'moderate', 'severe']
    severity_distribution = [
        {
            'level': lvl,
            'count': level_counts.get(lvl, 0),
            'color': SEVERITY_COLORS.get(lvl, '#6b7280'),
        }
        for lvl in severity_order
        if level_counts.get(lvl, 0) > 0
    ]

    # Daily report activity (based on assessment creation dates)
    daily_counts = Counter()
    for a in assessments:
        day = (a.get('created_at') or '')[:10]
        if day:
            daily_counts[day] += 1
    for ap in appointments:
        day = (ap.get('created_at') or '')[:10]
        if day:
            daily_counts[day] += 1
    daily_activity = [
        {'date': day, 'count': cnt}
        for day, cnt in sorted(daily_counts.items())
    ]

    # Per-patient report readiness
    readiness = []
    for p in patients:
        pid = p['patient_id']
        d = patient_data.get(pid, {})
        categories_filled = sum(1 for v in d.values() if v > 0)
        completeness = round(categories_filled / 6 * 100, 1)
        readiness.append({
            'patient_id': pid,
            'name': p.get('name', ''),
            'completeness': completeness,
            'categories_filled': categories_filled,
        })
    readiness.sort(key=lambda x: -x['completeness'])

    return {
        'available': True,
        'total_patients': total_patients,
        'total_assessments': total_assessments,
        'total_appointments': total_appointments,
        'total_seizure_events': total_seizures,
        'total_medications': total_medications,
        'total_mri_findings': total_mri,
        'reports_generated': reports_generated,
        'reports_possible': reports_possible,
        'report_coverage': report_coverage,
        'pipeline_events': pipeline_count,
        'instrument_distribution': instrument_distribution,
        'appt_status_distribution': appt_status_distribution,
        'severity_distribution': severity_distribution,
        'daily_activity': daily_activity,
        'patient_readiness': readiness[:20],
        'kpis': [
            {'label': 'Total Patients', 'value': str(total_patients)},
            {'label': 'Reports Generated', 'value': str(reports_generated)},
            {'label': 'Assessments', 'value': str(total_assessments)},
            {'label': 'Appointments', 'value': str(total_appointments)},
            {'label': 'Seizure Events', 'value': str(total_seizures)},
            {'label': 'Medications', 'value': str(total_medications)},
            {'label': 'MRI Scans', 'value': str(total_mri)},
            {'label': 'Report Coverage', 'value': f'{report_coverage:.0%}',
             'color': '#10b981' if report_coverage >= 0.8 else '#f59e0b' if report_coverage >= 0.5 else '#ef4444'},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed patient reporting breakdown — per-patient report inventory,
    appointment schedule, recent assessments, seizure log, pipeline events."""
    patients = _load_patients()
    assessments = _load_assessments()
    appointments = _load_appointments()
    seizures = _load_seizure_diary()
    medications = _load_medications()
    mri = _load_mri_findings()
    analyses = _load_analyses()
    pipeline = _load_pipeline_events()

    if not patients:
        return {'available': False}

    # Build per-patient data counts
    patient_map = {p['patient_id']: p for p in patients}

    assess_by_pid = {}
    for a in assessments:
        assess_by_pid.setdefault(a.get('patient_id'), []).append(a)

    appt_by_pid = {}
    for a in appointments:
        appt_by_pid.setdefault(a.get('patient_id'), []).append(a)

    seizure_by_pid = {}
    for s in seizures:
        seizure_by_pid.setdefault(s.get('patient_id'), []).append(s)

    med_by_pid = {}
    for m in medications:
        med_by_pid.setdefault(m.get('patient_id'), []).append(m)

    mri_by_pid = {}
    for m in mri:
        mri_by_pid.setdefault(m.get('patient_id'), []).append(m)

    analysis_by_pid = {}
    for an in analyses:
        analysis_by_pid.setdefault(an.get('patient_id'), []).append(an)

    # --- Per-patient report inventory ---
    patient_reports = []
    for p in patients:
        pid = p['patient_id']
        pa = assess_by_pid.get(pid, [])
        pap = appt_by_pid.get(pid, [])
        ps = seizure_by_pid.get(pid, [])
        pm = med_by_pid.get(pid, [])
        pmri = mri_by_pid.get(pid, [])
        pan = analysis_by_pid.get(pid, [])

        categories = {
            'assessments': len(pa),
            'appointments': len(pap),
            'seizures': len(ps),
            'medications': len(pm),
            'mri_findings': len(pmri),
            'analyses': len(pan),
        }
        categories_filled = sum(1 for v in categories.values() if v > 0)
        completeness = round(categories_filled / 6 * 100, 1)

        # Latest assessment date
        latest_assess = ''
        if pa:
            dates = [a.get('created_at', '') for a in pa if a.get('created_at')]
            if dates:
                latest_assess = max(dates)[:10]

        # Instruments used
        instruments = sorted(set(a['instrument'] for a in pa if a.get('instrument')))

        patient_reports.append({
            'patient_id': pid,
            'name': p.get('name', ''),
            'age': p.get('age'),
            'gender': p.get('gender'),
            'disease': p.get('disease'),
            'department': p.get('department'),
            'assessments': categories['assessments'],
            'appointments': categories['appointments'],
            'seizures': categories['seizures'],
            'medications': categories['medications'],
            'mri_findings': categories['mri_findings'],
            'analyses': categories['analyses'],
            'completeness': completeness,
            'categories_filled': categories_filled,
            'latest_assessment': latest_assess,
            'instruments': instruments,
        })
    patient_reports.sort(key=lambda x: -x['completeness'])

    # --- Appointment schedule ---
    appointment_schedule = [
        {
            'id': a['id'],
            'patient_id': a['patient_id'],
            'provider': a.get('provider', ''),
            'department': a.get('department', ''),
            'appt_type': a.get('appt_type', ''),
            'status': a.get('status', ''),
            'scheduled_for': a.get('scheduled_for', ''),
            'duration_min': a.get('duration_min'),
            'notes': a.get('notes', ''),
        }
        for a in appointments
    ]

    # --- Recent assessments ---
    recent_assessments = []
    for a in assessments[-100:]:
        pct = None
        if a.get('score') is not None and a.get('max_score') and a['max_score'] > 0:
            pct = round(a['score'] / a['max_score'] * 100, 1)
        recent_assessments.append({
            'id': a['id'],
            'patient_id': a['patient_id'],
            'instrument': a['instrument'],
            'score': a['score'],
            'max_score': a['max_score'],
            'pct': pct,
            'interpretation': a.get('interpretation'),
            'level': a.get('level'),
            'alert': a.get('alert'),
            'created_at': a.get('created_at'),
        })
    recent_assessments.reverse()

    # --- Seizure event log ---
    seizure_log = []
    for s in seizures:
        seizure_log.append({
            'id': s.get('id'),
            'patient_id': s.get('patient_id'),
            'event_date': s.get('event_date', ''),
            'event_time': s.get('event_time', ''),
            'duration_sec': s.get('duration_sec'),
            'location': s.get('location', ''),
            'witnessed': s.get('witnessed'),
            'aura': s.get('aura'),
        })

    # --- Pipeline events ---
    pipeline_events = [
        {
            'id': ev['id'],
            'patient_id': ev.get('patient_id'),
            'component': ev.get('component'),
            'action': ev.get('action'),
            'actor': ev.get('actor'),
            'detail': ev.get('detail'),
            'ts_utc': ev.get('ts_utc'),
        }
        for ev in pipeline[-200:]
    ]

    return {
        'available': True,
        'patient_reports': patient_reports,
        'appointment_schedule': appointment_schedule,
        'recent_assessments': recent_assessments,
        'seizure_log': seizure_log,
        'pipeline_events': pipeline_events,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Patient Reporting dashboard."""
    return {
        'concepts': [
            {
                'name': 'Patient Summary Report',
                'description': 'Consolidated overview of a patient\'s clinical status '
                               'including demographics, active diagnoses, recent assessments, '
                               'current medications, and upcoming appointments. Provides '
                               'clinicians with a single-page snapshot for handoff or review.',
            },
            {
                'name': 'Monitoring Report',
                'description': 'Scheduled clinical status update tracking disease progression, '
                               'treatment response, and key metric trends over time. Generated '
                               'at configurable intervals (daily, weekly, monthly) based on '
                               'incoming assessment data and seizure diary entries.',
            },
            {
                'name': 'Assessment Report',
                'description': 'Standardized report generated from clinical instrument scores '
                               '(MoCA, MMSE, PHQ-9, GAD-7, WAIS, etc.). Includes raw scores, '
                               'normalized percentages, severity levels, clinical interpretations, '
                               'and comparison to population norms.',
            },
            {
                'name': 'Seizure Diary Report',
                'description': 'Chronological summary of seizure events from patient self-report '
                               'and caregiver observations. Includes event frequency, duration, '
                               'type classification, triggers, aura patterns, and seizure-free '
                               'interval calculations for treatment efficacy tracking.',
            },
            {
                'name': 'Medication Report',
                'description': 'Comprehensive medication record including current anti-epileptic '
                               'drug regimen, dosage history, titration schedules, adverse effects, '
                               'serum drug levels, and adherence metrics. Supports polytherapy '
                               'monitoring and drug interaction screening.',
            },
            {
                'name': 'Imaging Report',
                'description': 'Structured summary of neuroimaging findings from MRI, CT, and '
                               'functional imaging studies. Includes lesion characterization, '
                               'volumetric measurements, region-of-interest analysis, and '
                               'longitudinal comparison with prior studies.',
            },
            {
                'name': 'Longitudinal Report',
                'description': 'Multi-visit trend analysis tracking patient trajectory across '
                               'assessments, seizure frequency, medication changes, and imaging '
                               'findings. Uses statistical trend detection to identify improvement, '
                               'stability, or decline patterns over the treatment course.',
            },
            {
                'name': 'Discharge Summary',
                'description': 'Comprehensive end-of-care document synthesizing the entire '
                               'clinical episode including admission reason, diagnostic workup, '
                               'treatment course, final assessment scores, discharge medications, '
                               'follow-up plan, and patient education materials provided.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Report Completeness',
                'description': 'Percentage of required data fields populated in a patient report. '
                               'Measures across all six data categories (assessments, appointments, '
                               'seizure diary, medications, imaging, EEG analyses). Target: >80% '
                               'for clinical decision-support readiness.',
            },
            {
                'name': 'Data Recency',
                'description': 'Time elapsed since the most recent data entry for each patient. '
                               'Reports with data older than 30 days are flagged as stale. '
                               'Critical for ensuring monitoring reports reflect current clinical '
                               'status rather than historical snapshots.',
            },
            {
                'name': 'Assessment Coverage',
                'description': 'Ratio of administered instruments to recommended protocol. '
                               'Epilepsy monitoring protocols typically require seizure diary, '
                               'cognitive screening (MoCA/MMSE), mood assessment (PHQ-9/GAD-7), '
                               'and quality-of-life measure (QOLIE-31). Coverage <60% triggers '
                               'missing-assessment alerts.',
            },
            {
                'name': 'Clinical Correlation',
                'description': 'Degree to which report findings cross-reference related data '
                               'sources. For example, seizure diary entries correlated with EEG '
                               'analysis results and medication changes. Higher correlation '
                               'scores indicate more integrated, clinically useful reports.',
            },
        ],
        'report_types': [
            {
                'type': 'Assessment Summary',
                'description': 'Single-instrument or multi-instrument assessment results '
                               'with scores, interpretations, and clinical recommendations.',
            },
            {
                'type': 'Seizure Diary Summary',
                'description': 'Seizure event frequency, duration trends, and trigger '
                               'analysis from patient-reported diary entries.',
            },
            {
                'type': 'Medication Review',
                'description': 'Current medication regimen, adherence metrics, dosage '
                               'history, and adverse effect monitoring results.',
            },
            {
                'type': 'Imaging Summary',
                'description': 'Neuroimaging findings with lesion characterization, '
                               'volumetric data, and comparison to prior studies.',
            },
            {
                'type': 'Comprehensive Patient Report',
                'description': 'Integrated multi-source report combining assessments, '
                               'seizure data, medications, imaging, and EEG analyses '
                               'into a single longitudinal patient record.',
            },
        ],
        'compliance': [
            {
                'ref': 'FDA AI/ML Framework',
                'note': 'AI-generated patient reports used for clinical decision support '
                        'must follow the Predetermined Change Control Plan (PCCP). '
                        'Automated report generation algorithms require validation '
                        'against clinician-authored reports for accuracy and completeness.',
            },
            {
                'ref': 'EU AI Act Art. 6',
                'note': 'Automated patient reporting systems in medical contexts are '
                        'classified as high-risk AI. Require conformity assessment, '
                        'human oversight of generated reports, and transparency '
                        'obligations regarding AI involvement in report generation.',
            },
            {
                'ref': 'ISO 14971',
                'note': 'Risk management must address report generation failures, '
                        'including incomplete data aggregation, stale data inclusion, '
                        'incorrect severity classification, and missed critical alerts '
                        'in automated monitoring summaries.',
            },
            {
                'ref': 'HIPAA',
                'note': 'Patient reports contain protected health information (PHI). '
                        'Systems must encrypt reports at rest and in transit, enforce '
                        'role-based access controls, maintain audit trails for all '
                        'report generation and access events, and support patient '
                        'right-of-access requests.',
            },
            {
                'ref': 'Joint Commission',
                'note': 'Patient monitoring reports must meet Joint Commission '
                        'documentation standards including timely completion, '
                        'authenticated authorship, legibility, and inclusion of '
                        'all required clinical data elements per facility policy.',
            },
        ],
        'remediation': [
            {
                'strategy': 'Missing Data Alert Escalation',
                'description': 'When report completeness drops below 60%, automatically '
                               'escalate to care coordinator with specific missing data '
                               'categories identified. Generate pre-populated assessment '
                               'order sets to expedite data collection.',
            },
            {
                'strategy': 'Stale Report Refresh Protocol',
                'description': 'Flag patient reports with no new data entries for >14 days. '
                               'Trigger outreach workflow to schedule follow-up assessments '
                               'or seizure diary review. Update report timestamp only when '
                               'new clinical data is incorporated.',
            },
            {
                'strategy': 'Cross-Source Validation',
                'description': 'Validate report consistency by cross-referencing seizure '
                               'diary entries with EEG analysis results and medication '
                               'change logs. Flag discrepancies (e.g., reported seizure '
                               'without corresponding EEG or medication adjustment) for '
                               'clinician review.',
            },
            {
                'strategy': 'Report Version Control',
                'description': 'Maintain versioned history of all generated reports with '
                               'diff tracking between versions. Support report amendment '
                               'workflow with addendum capability rather than overwrite, '
                               'preserving the clinical record integrity.',
            },
        ],
    }
