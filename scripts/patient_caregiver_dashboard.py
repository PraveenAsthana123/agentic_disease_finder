"""Patient / Caregiver Dashboard — seizure diary review, mood & anxiety
tracking (PHQ-9, GAD-7), quality of life (QOLIE-31), medication overview,
appointment management, trigger analysis, and risk factor identification
from real clinical evaluations.

Maps clinical.db tables to patient/caregiver concepts:
- seizure_diary       -> daily seizure events with severity, triggers, aura, injury
- assessments (PHQ9)  -> depression screening (0-27)
- assessments (GAD7)  -> anxiety screening (0-21)
- assessments (QOLIE31) -> quality of life (0-100)
- assessments (BARTHEL)  -> ADL independence (0-100)
- medications         -> current anti-seizure medications
- appointments        -> upcoming and past appointments
- patients            -> demographics, disease
"""

import json
import sqlite3
import os
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


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


def _load_assessments(instrument):
    return _db_query(
        'SELECT patient_id, score, max_score, interpretation, level, answers_json, '
        'examiner, created_at FROM assessments WHERE instrument = ? ORDER BY created_at DESC',
        (instrument,)
    )


def _load_patients():
    return {r['patient_id']: r for r in _db_query('SELECT * FROM patients')}


def _load_medications():
    return _db_query('SELECT patient_id, fields_json, created_at FROM medications')


def _load_seizure_diary():
    return _db_query(
        'SELECT patient_id, event_date, event_time, duration_sec, location, '
        'witnessed, aura, awareness, motor_signs, injury, post_ictal, '
        'recovery_min, er_visit, rescue_med, severity, trigger, notes, created_at '
        'FROM seizure_diary ORDER BY event_date DESC'
    )


def _load_appointments():
    return _db_query(
        'SELECT patient_id, provider, department, appt_type, status, '
        'booked_at, scheduled_for, completed_at, duration_min, notes, created_at '
        'FROM appointments ORDER BY scheduled_for DESC'
    )


def overview():
    """Return KPI cards + chart data for the Patient/Caregiver overview tab."""
    patients = _load_patients()
    seizures = _load_seizure_diary()
    all_assessments = _db_query('SELECT COUNT(*) as cnt FROM assessments')
    total_assessments = all_assessments[0]['cnt'] if all_assessments else 0
    medications = _load_medications()
    appointments = _load_appointments()
    phq9 = _load_assessments('PHQ9')
    gad7 = _load_assessments('GAD7')
    qolie = _load_assessments('QOLIE31')

    # --- KPIs ---
    severity_values = []
    SEVERITY_MAP = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
    for s in seizures:
        sev = (s.get('severity') or '').capitalize()
        if sev in SEVERITY_MAP:
            severity_values.append(SEVERITY_MAP[sev])

    phq9_scores = [a['score'] for a in phq9 if a['score'] is not None]
    gad7_scores = [a['score'] for a in gad7 if a['score'] is not None]
    qolie_scores = [a['score'] for a in qolie if a['score'] is not None]

    kpis = {
        'total_patients': len(patients),
        'total_seizure_events': len(seizures),
        'total_assessments': total_assessments,
        'total_medications': len(medications),
        'total_appointments': len(appointments),
        'avg_seizure_severity': _avg(severity_values),
        'qol_average': _avg(qolie_scores),
        'mood_score_avg': _avg(phq9_scores),
    }

    # --- Seizure severity distribution ---
    sev_counter = Counter()
    for s in seizures:
        sev = (s.get('severity') or 'Unknown').capitalize()
        sev_counter[sev] += 1
    seizure_summary = [{'severity': sev, 'count': cnt} for sev, cnt in sev_counter.most_common()]

    # --- Trigger distribution ---
    trigger_counter = Counter()
    for s in seizures:
        t = s.get('trigger') or 'Unknown'
        trigger_counter[t] += 1
    trigger_distribution = [{'name': name, 'value': val} for name, val in trigger_counter.most_common()]

    # --- Mood overview ---
    phq9_by_pt = defaultdict(list)
    for a in phq9:
        phq9_by_pt[a['patient_id']].append(a)
    gad7_by_pt = defaultdict(list)
    for a in gad7:
        gad7_by_pt[a['patient_id']].append(a)

    phq9_level_counter = Counter()
    for a in phq9:
        lvl = a.get('level') or a.get('interpretation') or 'Unknown'
        phq9_level_counter[lvl] += 1
    gad7_level_counter = Counter()
    for a in gad7:
        lvl = a.get('level') or a.get('interpretation') or 'Unknown'
        gad7_level_counter[lvl] += 1

    mood_overview = {
        'phq9_avg': _avg(phq9_scores),
        'gad7_avg': _avg(gad7_scores),
        'phq9_levels': [{'level': lvl, 'count': cnt} for lvl, cnt in phq9_level_counter.most_common()],
        'gad7_levels': [{'level': lvl, 'count': cnt} for lvl, cnt in gad7_level_counter.most_common()],
    }

    # --- QoL distribution (binned) ---
    qol_bins = {'Low': 0, 'Moderate': 0, 'High': 0}
    for sc in qolie_scores:
        if sc <= 40:
            qol_bins['Low'] += 1
        elif sc <= 70:
            qol_bins['Moderate'] += 1
        else:
            qol_bins['High'] += 1
    qol_distribution = [{'range': r, 'count': c} for r, c in qol_bins.items()]

    # --- Appointment status distribution ---
    appt_counter = Counter()
    for a in appointments:
        appt_counter[a.get('status') or 'Unknown'] += 1
    appointment_status = [{'status': st, 'count': cnt} for st, cnt in appt_counter.most_common()]

    return {
        'kpis': kpis,
        'seizure_summary': seizure_summary,
        'trigger_distribution': trigger_distribution,
        'mood_overview': mood_overview,
        'qol_distribution': qol_distribution,
        'appointment_status': appointment_status,
    }


def breakdown():
    """Return seizure diary, patient profiles, medication list, appointments, and seizure timeline."""
    patients = _load_patients()
    seizures = _load_seizure_diary()
    medications = _load_medications()
    appointments = _load_appointments()
    phq9 = _load_assessments('PHQ9')
    gad7 = _load_assessments('GAD7')
    qolie = _load_assessments('QOLIE31')
    all_assessments_by_pt = defaultdict(int)
    for r in _db_query('SELECT patient_id, COUNT(*) as cnt FROM assessments GROUP BY patient_id'):
        all_assessments_by_pt[r['patient_id']] = r['cnt']

    # --- Seizure diary rows ---
    seizure_diary = []
    for s in seizures:
        seizure_diary.append({
            'patient_id': s['patient_id'],
            'event_date': s.get('event_date'),
            'duration_sec': s.get('duration_sec', 0),
            'severity': s.get('severity'),
            'trigger': s.get('trigger'),
            'aura': s.get('aura'),
            'injury': s.get('injury'),
            'er_visit': s.get('er_visit'),
            'rescue_med': s.get('rescue_med'),
        })

    # --- Index seizures by patient ---
    sz_by_pt = defaultdict(list)
    for s in seizures:
        sz_by_pt[s['patient_id']].append(s)

    # --- Index assessments by patient (latest per instrument) ---
    phq9_by_pt = {}
    for a in phq9:
        if a['patient_id'] not in phq9_by_pt:
            phq9_by_pt[a['patient_id']] = a['score']
    gad7_by_pt = {}
    for a in gad7:
        if a['patient_id'] not in gad7_by_pt:
            gad7_by_pt[a['patient_id']] = a['score']
    qolie_by_pt = {}
    for a in qolie:
        if a['patient_id'] not in qolie_by_pt:
            qolie_by_pt[a['patient_id']] = a['score']

    # --- Medications by patient ---
    meds_by_pt = defaultdict(list)
    for m in medications:
        data = _safe_json(m.get('fields_json'))
        drug = data.get('drug_name', data.get('medication', ''))
        if drug:
            meds_by_pt[m['patient_id']].append(drug)

    # --- Patient profiles ---
    patient_profiles = []
    for pid, pt in sorted(patients.items()):
        sz_list = sz_by_pt.get(pid, [])
        seizure_count = len(sz_list)
        last_seizure_date = sz_list[0].get('event_date') if sz_list else None
        pt_meds = meds_by_pt.get(pid, [])
        latest_phq9 = phq9_by_pt.get(pid)
        latest_gad7 = gad7_by_pt.get(pid)
        latest_qolie = qolie_by_pt.get(pid)

        # Risk factors
        risk_factors = []
        if latest_phq9 is not None and latest_phq9 >= 10:
            risk_factors.append('Depression')
        if latest_gad7 is not None and latest_gad7 >= 10:
            risk_factors.append('Anxiety')
        if seizure_count >= 3:
            risk_factors.append('Frequent seizures')
        has_er = any(s.get('er_visit', '').lower() == 'yes' for s in sz_list)
        if has_er:
            risk_factors.append('ER visits')

        patient_profiles.append({
            'patient_id': pid,
            'name': pt.get('name', pid),
            'age': pt.get('age'),
            'gender': pt.get('gender'),
            'disease': pt.get('disease'),
            'seizure_count': seizure_count,
            'last_seizure_date': last_seizure_date,
            'medications': pt_meds,
            'assessment_count': all_assessments_by_pt.get(pid, 0),
            'latest_phq9': latest_phq9,
            'latest_gad7': latest_gad7,
            'latest_qolie': latest_qolie,
            'risk_factors': risk_factors,
        })

    # --- Medication list ---
    medication_list = []
    for m in medications:
        data = _safe_json(m.get('fields_json'))
        drug = data.get('drug_name', data.get('medication', ''))
        if drug:
            medication_list.append({
                'patient_id': m['patient_id'],
                'drug_name': drug,
                'dose_mg': data.get('dose_mg'),
                'frequency': data.get('frequency'),
            })

    # --- Appointment list (recent 30) ---
    appointment_list = []
    for a in appointments[:30]:
        appointment_list.append({
            'patient_id': a['patient_id'],
            'provider': a.get('provider'),
            'department': a.get('department'),
            'appt_type': a.get('appt_type'),
            'status': a.get('status'),
            'scheduled_for': a.get('scheduled_for'),
        })

    # --- Seizure timeline (events per day) ---
    day_counter = Counter()
    for s in seizures:
        d = s.get('event_date')
        if d:
            day_counter[d] += 1
    seizure_timeline = [{'date': d, 'count': c} for d, c in sorted(day_counter.items())]

    return {
        'seizure_diary': seizure_diary,
        'patient_profiles': patient_profiles,
        'medication_list': medication_list,
        'appointment_list': appointment_list,
        'seizure_timeline': seizure_timeline,
    }


def definitions():
    """Return patient/caregiver-relevant definitions, quality metrics, and compliance refs."""
    return {
        'concepts': [
            {
                'name': 'Seizure Diary',
                'description': 'A daily patient-reported log of seizure events including date, time, duration, severity, triggers, aura, injury, ER visits, and rescue medication use. Essential for tracking seizure control and identifying patterns.',
            },
            {
                'name': 'PHQ-9',
                'description': 'Patient Health Questionnaire-9, a validated 9-item self-report tool for depression screening. Scores range 0-27: Minimal (0-4), Mild (5-9), Moderate (10-14), Moderately Severe (15-19), Severe (20-27). Depression is common in epilepsy.',
            },
            {
                'name': 'GAD-7',
                'description': 'Generalized Anxiety Disorder 7-item scale for anxiety screening. Scores range 0-21: Minimal (0-4), Mild (5-9), Moderate (10-14), Severe (15-21). Anxiety frequently co-occurs with epilepsy.',
            },
            {
                'name': 'QOLIE-31',
                'description': 'Quality of Life in Epilepsy 31-item instrument measuring 7 domains: seizure worry, overall QoL, emotional well-being, energy/fatigue, cognitive function, medication effects, and social function. Scores 0-100, higher is better.',
            },
            {
                'name': 'Seizure Triggers',
                'description': 'Patient-identified precipitants that may increase seizure likelihood, including sleep deprivation, stress, missed medications, alcohol, flickering lights, illness, and hormonal changes. Identifying triggers supports seizure avoidance strategies.',
            },
            {
                'name': 'Rescue Medication',
                'description': 'Emergency medication administered during or after prolonged or cluster seizures, typically benzodiazepines (e.g., diazepam rectal, midazolam nasal/buccal). Caregivers are trained in administration as part of the seizure action plan.',
            },
            {
                'name': 'Barthel Index',
                'description': 'A 10-item ordinal scale (0-100) measuring functional independence in activities of daily living (ADLs): feeding, bathing, grooming, dressing, bowel/bladder control, toilet use, transfers, mobility, and stairs. Higher scores indicate greater independence.',
            },
            {
                'name': 'Medication Adherence',
                'description': 'Consistency of taking prescribed anti-seizure medications (ASMs) as directed. Poor adherence is a leading cause of breakthrough seizures. Tools include pill counts, pharmacy refill records, and self-report scales (e.g., MMAS-8).',
            },
        ],
        'quality_metrics': [
            {'name': 'Diary Completion Rate', 'target': '>=80%'},
            {'name': 'Assessment Timeliness', 'target': '<=30 days'},
            {'name': 'Trigger Identification', 'target': '>=3 per patient'},
            {'name': 'Medication Documentation', 'target': '100%'},
        ],
        'compliance_references': [
            {
                'name': 'HIPAA Privacy',
                'description': 'Health Insurance Portability and Accountability Act — federal protection of patient health information, requiring consent for data sharing and secure storage of seizure diaries, assessments, and medication records.',
            },
            {
                'name': 'AAN Guidelines',
                'description': 'American Academy of Neurology evidence-based guidelines for seizure management, including when to initiate or change anti-seizure medications, monitoring protocols, and quality measures for epilepsy care.',
            },
            {
                'name': 'ILAE Classification',
                'description': 'International League Against Epilepsy classification of seizure types (focal, generalized, unknown onset) and epilepsy syndromes. Provides standardized terminology used in seizure diary documentation.',
            },
            {
                'name': 'FDA MedWatch',
                'description': 'U.S. Food and Drug Administration adverse event reporting system. Patients and caregivers can report suspected adverse drug reactions to anti-seizure medications directly.',
            },
        ],
        'remediation_strategies': [
            {
                'name': 'Seizure Action Plan',
                'description': 'A personalized emergency response protocol detailing steps for caregivers during and after a seizure, including when to administer rescue medication, when to call 911, and recovery positioning.',
            },
            {
                'name': 'Trigger Avoidance',
                'description': 'Lifestyle modifications to reduce seizure risk, including maintaining regular sleep schedules, stress management techniques, medication adherence reminders, and avoiding known personal triggers.',
            },
            {
                'name': 'Mood Management',
                'description': 'Cognitive behavioral therapy (CBT) and/or pharmacotherapy for comorbid depression and anxiety in epilepsy. Addresses the bidirectional relationship between mood disorders and seizure control.',
            },
            {
                'name': 'Self-Management Education',
                'description': 'Patient empowerment programs for daily epilepsy care, including seizure diary use, medication management, trigger tracking, safety planning, and knowing when to seek medical attention.',
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN (first 2 profiles) ===')
    bd = breakdown()
    for p in bd['patient_profiles'][:2]:
        pprint.pprint(p)
    print(f'\nSeizure diary entries: {len(bd["seizure_diary"])}')
    print(f'Medication records: {len(bd["medication_list"])}')
    print(f'Appointments: {len(bd["appointment_list"])}')
    print(f'Timeline days: {len(bd["seizure_timeline"])}')
    print('\n=== DEFINITIONS ===')
    pprint.pprint(definitions())
