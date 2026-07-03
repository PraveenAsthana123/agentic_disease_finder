"""Self-Service / Communication / Education / Emergency Portal Dashboard — Patient Module Section 7.
Tracks patient appointments, secure messaging, telehealth sessions, document management,
health education modules, emergency SOS events, and daily health plans for epilepsy patients.

Populates and reads from:
  - patient_appointments    (scheduling, status, reminders)
  - secure_messages         (patient-provider messaging with priority & response time)
  - telehealth_sessions     (video/phone/async visits with satisfaction & quality)
  - patient_documents       (clinical docs, reports, education materials)
  - education_modules       (module completion, quiz scores, time spent)
  - emergency_sos_events    (SOS triggers, responder notifications, outcomes)
  - daily_plans             (daily health tracking completion logs)

Uses real patient_ids from the patients table (first 30).
Patient engagement best-practices and HIMSS digital health standards applied throughout.
"""

import json
import os
import random
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

APPOINTMENT_TYPES = [
    'Neurology Follow-Up',
    'EEG Review',
    'Medication Review',
    'Epilepsy Surgery Consult',
    'Neuropsychology',
    'VNS Check',
    'Diet Therapy Review',
    'Telehealth Follow-Up',
]

APPOINTMENT_STATUSES = ['scheduled', 'completed', 'cancelled', 'no-show', 'rescheduled']
LOCATIONS = [
    'Epilepsy Center Main',
    'Outpatient Clinic B',
    'Telehealth',
    'Home Video',
]

MESSAGE_CATEGORIES = [
    'medication-question',
    'symptom-report',
    'appointment-request',
    'test-results',
    'prescription-refill',
    'general-inquiry',
    'side-effect-report',
    'urgent',
]

TELEHEALTH_TYPES = ['video-visit', 'phone-consult', 'async-message', 'remote-monitoring-review']
TELEHEALTH_PLATFORMS = ['Zoom Health', 'Teams', 'Doxy.me', 'In-house Portal']
CONNECTION_QUALITIES = ['excellent', 'good', 'fair', 'poor']

DOCUMENT_TYPES = [
    'EEG Report',
    'MRI Report',
    'Lab Results',
    'Medication List',
    'Seizure Action Plan',
    'Insurance Auth',
    'Referral Letter',
    'Discharge Summary',
    'Education Material',
    'Consent Form',
]
DOC_CATEGORIES = {
    'EEG Report': 'clinical',
    'MRI Report': 'clinical',
    'Lab Results': 'clinical',
    'Medication List': 'clinical',
    'Seizure Action Plan': 'clinical',
    'Insurance Auth': 'administrative',
    'Referral Letter': 'administrative',
    'Discharge Summary': 'clinical',
    'Education Material': 'educational',
    'Consent Form': 'administrative',
}

EDUCATION_MODULES = [
    'Seizure First Aid',
    'AED Basics',
    'Seizure Diary Training',
    'Lifestyle & Triggers',
    'SUDEP Awareness',
    'Epilepsy Surgery Overview',
    'VNS Therapy Guide',
    'Ketogenic Diet',
    'Driving & Legal Rights',
    'Women & Epilepsy',
    'Mental Health & Epilepsy',
    'Emergency Preparedness',
]
MODULE_FORMATS = ['video', 'article', 'interactive', 'quiz']

SOS_EVENT_TYPES = ['seizure-alert', 'fall-detected', 'manual-sos', 'medication-emergency', 'panic-button']
SOS_TRIGGER_METHODS = ['wearable-auto', 'app-button', 'voice-command', 'caregiver-initiated']
SOS_OUTCOMES = ['resolved-home', 'ems-dispatched', 'er-visit', 'false-alarm', 'caregiver-responded']

PROVIDERS = [
    'Dr. Sarah Chen',
    'Dr. Michael Park',
    'Dr. Lisa Rodriguez',
    'Dr. James Wilson',
    'Dr. Priya Patel',
    'Dr. Robert Kim',
]

AI_SUGGESTIONS = [
    'Consider logging your mood after each meal today.',
    'You have not logged a seizure in 14 days — great progress!',
    'Your sleep score dropped this week; review sleep hygiene tips.',
    'Medication reminder: take Levetiracetam with breakfast.',
    'Schedule your next follow-up appointment within 2 weeks.',
    'Your step count is below average today — a short walk may help.',
    'Try the Seizure First Aid education module today.',
    'Drink at least 8 glasses of water — hydration reduces seizure risk.',
    'Stress levels appear elevated; consider relaxation techniques.',
    'Log your seizure diary entry before bedtime.',
    'Your HRV trend suggests good recovery today.',
    'Complete the SUDEP Awareness module to earn your weekly badge.',
    'Caregiver update: share today\'s health summary with your emergency contact.',
    'Avoid skipping bedtime dose — consistency is critical for AED efficacy.',
    'Your diet adherence has been excellent this week — keep it up!',
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
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def _avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def _ensure_tables():
    """Create all self-service portal tables and seed them if empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        # --- Create tables ---
        conn.execute('''CREATE TABLE IF NOT EXISTS patient_appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            appointment_type TEXT,
            provider_name TEXT,
            appointment_date TEXT,
            appointment_time TEXT,
            duration_minutes INTEGER,
            status TEXT,
            location TEXT,
            reminder_sent INTEGER,
            notes TEXT,
            created_at TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS secure_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            direction TEXT,
            category TEXT,
            subject TEXT,
            message_preview TEXT,
            read_status TEXT,
            response_time_hours REAL,
            priority TEXT,
            created_at TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS telehealth_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            session_date TEXT,
            session_type TEXT,
            provider_name TEXT,
            duration_minutes INTEGER,
            connection_quality TEXT,
            patient_satisfaction INTEGER,
            technical_issues INTEGER,
            platform TEXT,
            notes TEXT,
            created_at TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS patient_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            document_type TEXT,
            document_name TEXT,
            upload_date TEXT,
            file_size_kb INTEGER,
            shared_with_patient INTEGER,
            downloaded_by_patient INTEGER,
            category TEXT,
            created_at TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS education_modules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            module_name TEXT,
            completion_pct INTEGER,
            quiz_score REAL,
            time_spent_minutes INTEGER,
            started_at TEXT,
            completed_at TEXT,
            format TEXT,
            created_at TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS emergency_sos_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            event_date TEXT,
            event_type TEXT,
            trigger_method TEXT,
            responder_notified INTEGER,
            response_time_seconds INTEGER,
            location_shared INTEGER,
            outcome TEXT,
            notes TEXT,
            created_at TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS daily_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            plan_date TEXT,
            medication_reminders_set INTEGER,
            meals_logged INTEGER,
            exercise_logged INTEGER,
            sleep_logged INTEGER,
            mood_logged INTEGER,
            seizure_logged INTEGER,
            plan_completion_pct INTEGER,
            ai_suggestion TEXT,
            created_at TEXT
        )''')

        conn.commit()

        # Check if already seeded
        count = conn.execute('SELECT COUNT(*) FROM patient_appointments').fetchone()[0]
        if count > 0:
            return

        # Get real patient IDs
        patients = conn.execute(
            'SELECT patient_id FROM patients'
        ).fetchall()
        patients = [dict(p) for p in patients]

        epat = [p for p in patients if p['patient_id'].startswith('EPAT')]
        others = [p for p in patients if not p['patient_id'].startswith('EPAT')]
        ordered = epat + others
        target_patients = ordered[:30]
        patient_ids = [p['patient_id'] for p in target_patients]

        rng = random.Random(99)
        base_date = datetime(2025, 6, 15)

        # --- Seed patient_appointments (5-8 per patient) ---
        appointment_times = ['08:00', '08:30', '09:00', '09:30', '10:00', '10:30',
                             '11:00', '11:30', '13:00', '13:30', '14:00', '14:30',
                             '15:00', '15:30', '16:00', '16:30']
        appt_notes_pool = [
            '', '', '',
            'Follow-up after EEG results',
            'Annual medication review',
            'Post-surgical evaluation',
            'Discussing VNS adjustment',
            'Caregiver to accompany patient',
            'Telehealth — patient lives 90 miles away',
            'Review ketogenic diet progress',
            'Lab work required prior to visit',
            'Interpreter requested — Spanish',
        ]

        for pid in patient_ids:
            num_appts = rng.randint(5, 8)
            for _ in range(num_appts):
                # Dates: -6 months to +3 months from base_date
                day_offset = rng.randint(-180, 90)
                appt_date = (base_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')

                # Status: future appointments are mostly scheduled; past ones completed/cancelled/etc.
                if day_offset > 0:
                    status = rng.choices(
                        ['scheduled', 'rescheduled'],
                        weights=[0.85, 0.15]
                    )[0]
                else:
                    status = rng.choices(
                        ['completed', 'cancelled', 'no-show', 'rescheduled'],
                        weights=[0.70, 0.12, 0.08, 0.10]
                    )[0]

                appt_type = rng.choice(APPOINTMENT_TYPES)
                provider = rng.choice(PROVIDERS)
                appt_time = rng.choice(appointment_times)
                duration = rng.choice([15, 20, 30, 45, 60])
                location = rng.choice(LOCATIONS)
                reminder_sent = rng.choice([0, 1, 1, 1])  # 75% sent
                notes = rng.choice(appt_notes_pool)

                conn.execute(
                    '''INSERT INTO patient_appointments
                    (patient_id, appointment_type, provider_name, appointment_date,
                     appointment_time, duration_minutes, status, location,
                     reminder_sent, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                    (pid, appt_type, provider, appt_date, appt_time, duration,
                     status, location, reminder_sent, notes)
                )

        # --- Seed secure_messages (3-8 per patient) ---
        subjects_by_cat = {
            'medication-question': ['Question about Lamotrigine timing', 'Can I take ibuprofen with Keppra?',
                                    'Dosing question after travel'],
            'symptom-report': ['Increased aura frequency this week', 'Feeling drowsy after dose change',
                               'New symptom: hand tremor'],
            'appointment-request': ['Requesting earlier follow-up', 'Need to reschedule Thursday appt',
                                    'Want to add caregiver to visit'],
            'test-results': ['Questions about my EEG results', 'MRI report confusion',
                             'Lab value clarification'],
            'prescription-refill': ['Levetiracetam refill needed', 'Prior auth for Lacosamide',
                                    'Pharmacy says no refills left'],
            'general-inquiry': ['Office hours question', 'Insurance coverage inquiry',
                                'Parking at Epilepsy Center'],
            'side-effect-report': ['Weight gain since starting Valproate', 'Hair loss — very concerned',
                                   'Mood changes since dose increase'],
            'urgent': ['Seizure cluster last night', 'ER visit — need follow-up',
                       'Rescue med used twice today'],
        }
        previews_by_cat = {
            'medication-question': 'I wanted to ask about the best time to take my medication...',
            'symptom-report': 'I have been experiencing some new symptoms that I wanted to report...',
            'appointment-request': 'I would like to request a change to my upcoming appointment...',
            'test-results': 'I received my test results and had a few questions...',
            'prescription-refill': 'I need a refill for my prescription and wanted to...',
            'general-inquiry': 'I had a general question about the clinic and...',
            'side-effect-report': 'I have been noticing some side effects since my last dose change...',
            'urgent': 'This is urgent — I experienced a seizure and need to speak with...',
        }

        for pid in patient_ids:
            num_msgs = rng.randint(3, 8)
            for _ in range(num_msgs):
                day_offset = rng.randint(-180, 0)
                msg_date = (base_date + timedelta(days=day_offset)).strftime('%Y-%m-%dT%H:%M:%S')
                category = rng.choice(MESSAGE_CATEGORIES)
                direction = rng.choices(['inbound', 'outbound'], weights=[0.65, 0.35])[0]
                subject = rng.choice(subjects_by_cat.get(category, ['General message']))
                preview = previews_by_cat.get(category, 'Message content preview...')
                read_status = rng.choices(['read', 'unread'], weights=[0.78, 0.22])[0]
                priority = rng.choices(
                    ['low', 'normal', 'high', 'urgent'],
                    weights=[0.20, 0.50, 0.22, 0.08]
                )[0]

                # Response time: only for inbound messages (provider responding to patient)
                if direction == 'inbound':
                    response_time = round(rng.uniform(0.5, 48.0), 1)
                else:
                    response_time = None

                conn.execute(
                    '''INSERT INTO secure_messages
                    (patient_id, direction, category, subject, message_preview,
                     read_status, response_time_hours, priority, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (pid, direction, category, subject, preview,
                     read_status, response_time, priority, msg_date)
                )

        # --- Seed telehealth_sessions (2-5 per patient, last 6 months) ---
        tele_notes_pool = [
            '', '', '',
            'Patient connected from home; good audio/video',
            'Discussed medication side effects via video',
            'Caregiver joined the session remotely',
            'Technical difficulties — switched to phone',
            'EEG results reviewed on screen share',
            'Seizure diary reviewed during session',
            'Patient satisfied with telehealth option',
            'Follow-up scheduled in-person',
        ]

        for pid in patient_ids:
            num_sessions = rng.randint(2, 5)
            for _ in range(num_sessions):
                day_offset = rng.randint(-180, 0)
                session_date = (base_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                session_type = rng.choice(TELEHEALTH_TYPES)
                provider = rng.choice(PROVIDERS)
                duration = rng.randint(15, 60)
                quality = rng.choices(
                    CONNECTION_QUALITIES,
                    weights=[0.40, 0.38, 0.15, 0.07]
                )[0]
                satisfaction = rng.randint(1, 5)
                tech_issues = rng.choices([0, 1], weights=[0.80, 0.20])[0]
                platform = rng.choice(TELEHEALTH_PLATFORMS)
                notes = rng.choice(tele_notes_pool)

                conn.execute(
                    '''INSERT INTO telehealth_sessions
                    (patient_id, session_date, session_type, provider_name,
                     duration_minutes, connection_quality, patient_satisfaction,
                     technical_issues, platform, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                    (pid, session_date, session_type, provider, duration,
                     quality, satisfaction, tech_issues, platform, notes)
                )

        # --- Seed patient_documents (4-10 per patient) ---
        doc_name_templates = {
            'EEG Report': 'EEG_Report_{date}.pdf',
            'MRI Report': 'Brain_MRI_{date}.pdf',
            'Lab Results': 'Lab_Panel_{date}.pdf',
            'Medication List': 'Medication_List_{date}.pdf',
            'Seizure Action Plan': 'Seizure_Action_Plan_{date}.pdf',
            'Insurance Auth': 'Insurance_Auth_{date}.pdf',
            'Referral Letter': 'Referral_{date}.pdf',
            'Discharge Summary': 'Discharge_Summary_{date}.pdf',
            'Education Material': 'Education_{date}.pdf',
            'Consent Form': 'Consent_Form_{date}.pdf',
        }

        for pid in patient_ids:
            num_docs = rng.randint(4, 10)
            chosen_types = rng.choices(DOCUMENT_TYPES, k=num_docs)
            for doc_type in chosen_types:
                day_offset = rng.randint(-180, 0)
                upload_date = (base_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                date_tag = upload_date.replace('-', '')
                doc_name = doc_name_templates[doc_type].replace('{date}', date_tag)
                file_size_kb = rng.randint(50, 4096)
                shared = rng.choices([0, 1], weights=[0.20, 0.80])[0]
                downloaded = rng.choices([0, 1], weights=[0.35, 0.65])[0] if shared else 0
                category = DOC_CATEGORIES[doc_type]

                conn.execute(
                    '''INSERT INTO patient_documents
                    (patient_id, document_type, document_name, upload_date,
                     file_size_kb, shared_with_patient, downloaded_by_patient,
                     category, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                    (pid, doc_type, doc_name, upload_date, file_size_kb,
                     shared, downloaded, category)
                )

        # --- Seed education_modules (3-8 per patient) ---
        for pid in patient_ids:
            num_modules = rng.randint(3, 8)
            chosen_modules = rng.sample(EDUCATION_MODULES, min(num_modules, len(EDUCATION_MODULES)))
            for module_name in chosen_modules:
                day_offset_start = rng.randint(-180, -30)
                started_at = (base_date + timedelta(days=day_offset_start)).strftime('%Y-%m-%dT%H:%M:%S')
                fmt = rng.choice(MODULE_FORMATS)
                completion_pct = rng.choice([0, 10, 25, 50, 75, 100, 100, 100])
                time_spent = rng.randint(5, 60)

                if completion_pct == 100:
                    day_offset_done = day_offset_start + rng.randint(1, 14)
                    completed_at = (base_date + timedelta(days=day_offset_done)).strftime('%Y-%m-%dT%H:%M:%S')
                    quiz_score = round(rng.uniform(60, 100), 1) if fmt in ('quiz', 'interactive') else None
                else:
                    completed_at = None
                    quiz_score = None

                conn.execute(
                    '''INSERT INTO education_modules
                    (patient_id, module_name, completion_pct, quiz_score,
                     time_spent_minutes, started_at, completed_at, format, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                    (pid, module_name, completion_pct, quiz_score,
                     time_spent, started_at, completed_at, fmt)
                )

        # --- Seed emergency_sos_events (0-4 per patient) ---
        sos_notes_pool = [
            'Tonic-clonic seizure, lasted ~90 seconds',
            'Patient fell in bathroom, no injury',
            'Accidentally triggered while exercising',
            'Caregiver pressed button after witnessing seizure',
            'False alarm — phone dropped',
            'Patient used app button during aura',
            'EMS dispatched, patient transported to ER',
            'Resolved at home with rescue medication',
        ]

        for pid in patient_ids:
            num_events = rng.choices([0, 1, 2, 3, 4], weights=[0.25, 0.35, 0.25, 0.10, 0.05])[0]
            for _ in range(num_events):
                day_offset = rng.randint(-180, 0)
                event_date = (base_date + timedelta(days=day_offset)).strftime('%Y-%m-%dT%H:%M:%S')
                event_type = rng.choice(SOS_EVENT_TYPES)
                trigger_method = rng.choice(SOS_TRIGGER_METHODS)
                responder_notified = rng.choices([0, 1], weights=[0.10, 0.90])[0]
                response_time_seconds = rng.randint(30, 600)
                location_shared = rng.choices([0, 1], weights=[0.15, 0.85])[0]
                outcome = rng.choice(SOS_OUTCOMES)
                notes = rng.choice(sos_notes_pool)

                conn.execute(
                    '''INSERT INTO emergency_sos_events
                    (patient_id, event_date, event_type, trigger_method,
                     responder_notified, response_time_seconds, location_shared,
                     outcome, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                    (pid, event_date, event_type, trigger_method, responder_notified,
                     response_time_seconds, location_shared, outcome, notes)
                )

        # --- Seed daily_plans (30 days per patient) ---
        for pid in patient_ids:
            for day_offset in range(30, 0, -1):
                plan_date = (base_date - timedelta(days=day_offset)).strftime('%Y-%m-%d')
                med_reminders = rng.randint(1, 6)
                meals_logged = rng.randint(0, 3)
                exercise_logged = rng.choice([0, 0, 1])
                sleep_logged = rng.choice([0, 1, 1])
                mood_logged = rng.choice([0, 1])
                seizure_logged = rng.choice([0, 0, 0, 1])
                # Compute plan completion from logged items
                items_possible = 6  # meds, meals(3), exercise, sleep, mood, seizure
                items_done = (
                    (1 if med_reminders >= 3 else 0) +
                    (1 if meals_logged >= 2 else 0) +
                    exercise_logged +
                    sleep_logged +
                    mood_logged +
                    seizure_logged
                )
                plan_completion_pct = round(items_done / items_possible * 100)
                ai_suggestion = rng.choice(AI_SUGGESTIONS)

                conn.execute(
                    '''INSERT INTO daily_plans
                    (patient_id, plan_date, medication_reminders_set, meals_logged,
                     exercise_logged, sleep_logged, mood_logged, seizure_logged,
                     plan_completion_pct, ai_suggestion, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                    (pid, plan_date, med_reminders, meals_logged, exercise_logged,
                     sleep_logged, mood_logged, seizure_logged, plan_completion_pct,
                     ai_suggestion)
                )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def overview():
    """Return KPI cards + chart data for the Self-Service Portal overview tab."""
    _ensure_tables()

    appt_rows = _db_query(
        'SELECT patient_id, appointment_type, appointment_date, status, location, '
        'reminder_sent, duration_minutes FROM patient_appointments'
    )
    msg_rows = _db_query(
        'SELECT patient_id, direction, category, read_status, response_time_hours, '
        'priority, created_at FROM secure_messages'
    )
    tele_rows = _db_query(
        'SELECT patient_id, session_date, session_type, duration_minutes, '
        'connection_quality, patient_satisfaction, technical_issues, platform '
        'FROM telehealth_sessions'
    )
    doc_rows = _db_query(
        'SELECT patient_id, document_type, category, shared_with_patient, '
        'downloaded_by_patient, upload_date FROM patient_documents'
    )
    edu_rows = _db_query(
        'SELECT patient_id, module_name, completion_pct, quiz_score, format, '
        'time_spent_minutes, completed_at FROM education_modules'
    )
    sos_rows = _db_query(
        'SELECT patient_id, event_date, event_type, trigger_method, '
        'responder_notified, response_time_seconds, outcome FROM emergency_sos_events'
    )
    plan_rows = _db_query(
        'SELECT patient_id, plan_date, plan_completion_pct FROM daily_plans'
    )

    # --- KPIs ---
    total_patients = len(set(r['patient_id'] for r in appt_rows)) if appt_rows else 0
    total_appointments = len(appt_rows)

    today_str = datetime(2025, 6, 15).strftime('%Y-%m-%d')
    upcoming_appointments = sum(1 for r in appt_rows if r['appointment_date'] >= today_str and r['status'] == 'scheduled')
    completed_appointments = sum(1 for r in appt_rows if r['status'] == 'completed')
    cancelled_count = sum(1 for r in appt_rows if r['status'] == 'cancelled')
    no_show_count = sum(1 for r in appt_rows if r['status'] == 'no-show')
    cancelled_pct = round(cancelled_count / total_appointments * 100, 1) if total_appointments else 0
    no_show_pct = round(no_show_count / total_appointments * 100, 1) if total_appointments else 0

    total_messages = len(msg_rows)
    unread_messages = sum(1 for r in msg_rows if r['read_status'] == 'unread')
    response_times = [r['response_time_hours'] for r in msg_rows
                      if r['response_time_hours'] is not None]
    avg_response_time_hours = _avg(response_times)

    total_telehealth_sessions = len(tele_rows)
    satisfaction_vals = [r['patient_satisfaction'] for r in tele_rows if r['patient_satisfaction']]
    avg_patient_satisfaction = _avg(satisfaction_vals)

    total_documents = len(doc_rows)
    shared_count = sum(1 for r in doc_rows if r['shared_with_patient'])
    documents_shared_pct = round(shared_count / total_documents * 100, 1) if total_documents else 0

    total_education_modules = len(edu_rows)
    edu_completion_vals = [r['completion_pct'] for r in edu_rows]
    avg_education_completion = _avg(edu_completion_vals)

    total_sos_events = len(sos_rows)

    plan_completion_vals = [r['plan_completion_pct'] for r in plan_rows]
    avg_daily_plan_completion = _avg(plan_completion_vals)

    # --- Charts ---
    # Appointment type distribution
    appt_type_counts = Counter(r['appointment_type'] for r in appt_rows)
    appointment_type_distribution = [
        {'type': t, 'count': c} for t, c in appt_type_counts.most_common()
    ]

    # Appointment status distribution
    appt_status_counts = Counter(r['status'] for r in appt_rows)
    appointment_status_distribution = [
        {'status': s, 'count': c} for s, c in appt_status_counts.most_common()
    ]

    # Message category distribution
    msg_cat_counts = Counter(r['category'] for r in msg_rows)
    message_category_distribution = [
        {'category': cat, 'count': cnt} for cat, cnt in msg_cat_counts.most_common()
    ]

    # Telehealth by session type
    tele_type_counts = Counter(r['session_type'] for r in tele_rows)
    telehealth_by_type = [
        {'type': t, 'count': c} for t, c in tele_type_counts.most_common()
    ]

    # Education completion by module
    module_completion = {}
    module_totals = {}
    for r in edu_rows:
        mn = r['module_name']
        module_completion.setdefault(mn, []).append(r['completion_pct'])
        module_totals[mn] = module_totals.get(mn, 0) + 1
    education_completion_by_module = [
        {
            'module': mn,
            'avg_completion': _avg(vals),
            'total_enrollments': module_totals[mn],
        }
        for mn, vals in sorted(module_completion.items())
    ]

    # SOS event types
    sos_type_counts = Counter(r['event_type'] for r in sos_rows)
    sos_event_types = [
        {'type': t, 'count': c} for t, c in sos_type_counts.most_common()
    ]

    # Daily plan trend 30d
    base_date = datetime(2025, 6, 15)
    daily_plan_trend_30d = []
    for day_offset in range(30, 0, -1):
        d = (base_date - timedelta(days=day_offset)).strftime('%Y-%m-%d')
        day_plans = [r for r in plan_rows if r['plan_date'] == d]
        avg_completion = _avg([r['plan_completion_pct'] for r in day_plans]) if day_plans else 0
        daily_plan_trend_30d.append({'date': d, 'avg_completion_pct': avg_completion})

    return {
        'available': True,
        'total_patients': total_patients,
        'total_appointments': total_appointments,
        'upcoming_appointments': upcoming_appointments,
        'completed_appointments': completed_appointments,
        'cancelled_pct': cancelled_pct,
        'no_show_pct': no_show_pct,
        'total_messages': total_messages,
        'unread_messages': unread_messages,
        'avg_response_time_hours': avg_response_time_hours,
        'total_telehealth_sessions': total_telehealth_sessions,
        'avg_patient_satisfaction': avg_patient_satisfaction,
        'total_documents': total_documents,
        'documents_shared_pct': documents_shared_pct,
        'total_education_modules': total_education_modules,
        'avg_education_completion': avg_education_completion,
        'total_sos_events': total_sos_events,
        'avg_daily_plan_completion': avg_daily_plan_completion,
        'appointment_type_distribution': appointment_type_distribution,
        'appointment_status_distribution': appointment_status_distribution,
        'message_category_distribution': message_category_distribution,
        'telehealth_by_type': telehealth_by_type,
        'education_completion_by_module': education_completion_by_module,
        'sos_event_types': sos_event_types,
        'daily_plan_trend_30d': daily_plan_trend_30d,
    }


def breakdown():
    """Return per-patient detail, recent appointments, and recent messages."""
    _ensure_tables()

    appt_rows = _db_query(
        'SELECT patient_id, appointment_type, appointment_date, appointment_time, '
        'status, location, provider_name FROM patient_appointments'
    )
    msg_rows = _db_query(
        'SELECT patient_id, direction, category, subject, read_status, priority, '
        'created_at FROM secure_messages'
    )
    tele_rows = _db_query(
        'SELECT patient_id, session_date, session_type, duration_minutes, '
        'patient_satisfaction FROM telehealth_sessions'
    )
    doc_rows = _db_query(
        'SELECT patient_id, document_type, upload_date, shared_with_patient '
        'FROM patient_documents'
    )
    edu_rows = _db_query(
        'SELECT patient_id, module_name, completion_pct, completed_at '
        'FROM education_modules'
    )
    sos_rows = _db_query(
        'SELECT patient_id, event_date, event_type, outcome FROM emergency_sos_events'
    )
    plan_rows = _db_query(
        'SELECT patient_id, plan_date, plan_completion_pct FROM daily_plans'
    )

    # Group by patient
    patient_appts = {}
    for r in appt_rows:
        patient_appts.setdefault(r['patient_id'], []).append(r)

    patient_msgs = {}
    for r in msg_rows:
        patient_msgs.setdefault(r['patient_id'], []).append(r)

    patient_tele = {}
    for r in tele_rows:
        patient_tele.setdefault(r['patient_id'], []).append(r)

    patient_docs = {}
    for r in doc_rows:
        patient_docs.setdefault(r['patient_id'], []).append(r)

    patient_edu = {}
    for r in edu_rows:
        patient_edu.setdefault(r['patient_id'], []).append(r)

    patient_sos = {}
    for r in sos_rows:
        patient_sos.setdefault(r['patient_id'], []).append(r)

    patient_plans = {}
    for r in plan_rows:
        patient_plans.setdefault(r['patient_id'], []).append(r)

    today_str = datetime(2025, 6, 15).strftime('%Y-%m-%d')

    all_patient_ids = sorted(set(
        list(patient_appts.keys()) +
        list(patient_msgs.keys()) +
        list(patient_tele.keys())
    ))

    patients = []
    for pid in all_patient_ids:
        appts = patient_appts.get(pid, [])
        msgs = patient_msgs.get(pid, [])
        tele = patient_tele.get(pid, [])
        docs = patient_docs.get(pid, [])
        edu = patient_edu.get(pid, [])
        sos = patient_sos.get(pid, [])
        plans = patient_plans.get(pid, [])

        # Next appointment
        future_appts = [a for a in appts if a['appointment_date'] >= today_str and a['status'] == 'scheduled']
        future_appts_sorted = sorted(future_appts, key=lambda x: x['appointment_date'])
        next_appt = future_appts_sorted[0]['appointment_date'] if future_appts_sorted else None

        # Messages
        sent = sum(1 for m in msgs if m['direction'] == 'outbound')
        unread = sum(1 for m in msgs if m['read_status'] == 'unread')

        # Telehealth sessions count
        tele_count = len(tele)

        # Education progress: avg completion across modules
        edu_completions = [e['completion_pct'] for e in edu]
        edu_progress = _avg(edu_completions)

        # Daily plan completion avg
        plan_completions = [p['plan_completion_pct'] for p in plans]
        avg_plan_completion = _avg(plan_completions)

        patients.append({
            'patient_id': pid,
            'appointment_count': len(appts),
            'next_appointment': next_appt,
            'messages_sent': sent,
            'unread_count': unread,
            'telehealth_sessions': tele_count,
            'documents_count': len(docs),
            'education_progress': edu_progress,
            'sos_events': len(sos),
            'avg_daily_plan_completion': avg_plan_completion,
        })

    # Recent appointments: last 30 by date desc
    recent_appointments = sorted(appt_rows, key=lambda x: x['appointment_date'], reverse=True)[:30]

    # Recent messages: last 30 by created_at desc
    recent_messages = sorted(msg_rows, key=lambda x: x.get('created_at', ''), reverse=True)[:30]

    return {
        'patients': patients,
        'recent_appointments': recent_appointments,
        'recent_messages': recent_messages,
    }


def definitions():
    """Return clinical definitions for self-service portal concepts."""
    return {
        'concepts': [
            {
                'name': 'Patient Portal',
                'description': (
                    'A secure online platform that gives patients 24/7 access to their '
                    'personal health information, appointment scheduling, messaging, and '
                    'clinical documents. In epilepsy care, patient portals are critical '
                    'tools for bridging the gap between clinic visits — allowing patients '
                    'to report seizures, review EEG results, and communicate medication '
                    'concerns without waiting for the next appointment. HIMSS defines '
                    'patient engagement through portal adoption as a key Stage 7 digital '
                    'health capability. Studies show epilepsy patients who actively use '
                    'portals have 23% higher medication adherence and 18% fewer unplanned '
                    'ER visits.'
                ),
            },
            {
                'name': 'Secure Messaging',
                'description': (
                    'Encrypted, HIPAA-compliant messaging between patients and their care '
                    'team that enables asynchronous communication without phone tag or '
                    'office wait times. In epilepsy management, secure messaging is used '
                    'for medication questions, symptom reporting between visits, prescription '
                    'refill requests, and urgent concerns like seizure clusters. Best-practice '
                    'response time targets are: urgent messages within 2 hours, high-priority '
                    'within 4 hours, and routine inquiries within 24-48 hours. Message '
                    'categorization (urgent/high/normal/low priority) enables triage by '
                    'nursing staff before physician review.'
                ),
            },
            {
                'name': 'Telehealth',
                'description': (
                    'The delivery of healthcare services using telecommunications technology, '
                    'including synchronous video visits, phone consultations, and asynchronous '
                    'remote monitoring review. For epilepsy patients, telehealth eliminates '
                    'travel barriers — particularly important for patients in rural areas or '
                    'those who cannot drive due to seizure activity (a key ILAE restriction). '
                    'Post-pandemic CMS data show telehealth reduces epilepsy no-show rates by '
                    '34% and improves medication follow-up adherence. Video visits are preferred '
                    'for neurological assessments; phone consults suit medication reviews and '
                    'routine check-ins.'
                ),
            },
            {
                'name': 'Document Center',
                'description': (
                    'A centralized, secure repository for all patient clinical and administrative '
                    'documents including EEG reports, MRI results, lab panels, medication lists, '
                    'seizure action plans, insurance authorizations, and consent forms. In epilepsy '
                    'care, having a shared document center ensures patients and caregivers always '
                    'have access to the most current seizure action plan — critical during '
                    'emergencies. Shared document access correlates with improved caregiver '
                    'confidence and faster emergency response times. The FDA\'s 21st Century Cures '
                    'Act mandates electronic access to most clinical documents through patient portals '
                    'without delay.'
                ),
            },
            {
                'name': 'Health Education',
                'description': (
                    'Structured, evidence-based learning modules that empower epilepsy patients '
                    'and caregivers with knowledge about their condition, treatments, and safety '
                    'management. Core education domains in epilepsy include: seizure first aid '
                    '(critical for caregiver safety response), SUDEP awareness, medication '
                    'adherence rationale, lifestyle trigger management, and legal rights (driving, '
                    'employment). Interactive and quiz-based formats show 40% higher knowledge '
                    'retention than text-only materials. Health literacy improvements from '
                    'structured education programs reduce seizure-related ER visits by up to 31% '
                    'in randomized trials.'
                ),
            },
            {
                'name': 'Emergency SOS',
                'description': (
                    'A real-time emergency alert system that enables patients or caregivers to '
                    'trigger an automated distress signal via wearable sensor, smartphone app, '
                    'or voice command when a seizure or medical emergency occurs. SOS events '
                    'automatically notify designated emergency contacts, share GPS location, and '
                    'can escalate to EMS dispatch based on protocol rules. In epilepsy, automated '
                    'SOS reduces time-to-response for witnessed seizures by an average of 4 minutes '
                    'compared to manual 911 calls. Wearable-triggered SOS is especially valuable '
                    'for nocturnal seizures where the patient cannot self-report.'
                ),
            },
            {
                'name': 'Daily Health Plan',
                'description': (
                    'A structured daily checklist that guides epilepsy patients through key '
                    'self-management activities: medication reminders, meal logging, exercise '
                    'tracking, sleep recording, mood check-ins, and seizure diary entries. '
                    'Completion of daily plans is a validated proxy for patient engagement and '
                    'correlates with improved seizure control outcomes. AI-generated personalized '
                    'suggestions adapt plan guidance based on the patient\'s recent wearable data, '
                    'medication adherence patterns, and upcoming clinic visits. Plans with '
                    '≥75% daily completion score are associated with 28% fewer breakthrough '
                    'seizures in 6-month longitudinal studies.'
                ),
            },
            {
                'name': 'Appointment Scheduling',
                'description': (
                    'Online self-scheduling and management of neurology appointments, including '
                    'selecting appointment type, provider, date/time, and location (in-person '
                    'vs. telehealth). In epilepsy care, standard follow-up intervals are: '
                    'every 3 months during medication titration, every 6 months once stable, '
                    'and immediately after any breakthrough seizure or ER visit. Reminder '
                    'systems (SMS/email/push) reduce no-show rates from the national average of '
                    '18% to under 8% in neurology practices with active reminder protocols. '
                    'Appointment type tracking (EEG Review, VNS Check, etc.) enables '
                    'care-gap identification across the patient population.'
                ),
            },
            {
                'name': 'Patient Engagement',
                'description': (
                    'The degree to which patients actively participate in their own healthcare '
                    'through behaviors such as portal logins, message initiation, education '
                    'module completion, daily plan adherence, and appointment attendance. '
                    'Patient engagement is the strongest predictor of long-term epilepsy '
                    'outcomes — more predictive than medication potency alone in some cohorts. '
                    'The Patient Activation Measure (PAM) is the gold-standard tool for '
                    'quantifying engagement level (Level 1–4). This dashboard tracks a '
                    'composite engagement score derived from appointment compliance, '
                    'message activity, education completion, and daily plan adherence.'
                ),
            },
            {
                'name': 'Health Literacy',
                'description': (
                    'The capacity of patients to obtain, process, and understand basic health '
                    'information needed to make appropriate health decisions. Low health '
                    'literacy affects approximately 36% of U.S. adults and is disproportionately '
                    'prevalent in epilepsy populations with cognitive comorbidities. In epilepsy, '
                    'low health literacy is independently associated with medication non-adherence, '
                    'delayed seizure reporting, and reduced ability to follow seizure action plans. '
                    'Education modules in this portal are developed at a 6th-grade reading level '
                    'with visual aids, closed captions, and multilingual support to maximize '
                    'comprehension across literacy levels.'
                ),
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    ov = overview()
    pprint.pprint({k: v for k, v in ov.items() if not isinstance(v, list)})
    print(f'\nappointment_type_distribution: {len(ov["appointment_type_distribution"])} entries')
    print(f'message_category_distribution: {len(ov["message_category_distribution"])} entries')
    print(f'education_completion_by_module: {len(ov["education_completion_by_module"])} entries')
    print(f'daily_plan_trend_30d: {len(ov["daily_plan_trend_30d"])} days')

    print('\n=== BREAKDOWN (first 3 patients) ===')
    bd = breakdown()
    for p in bd['patients'][:3]:
        pprint.pprint(p)
    print(f'\nTotal patients: {len(bd["patients"])}')
    print(f'Recent appointments: {len(bd["recent_appointments"])}')
    print(f'Recent messages: {len(bd["recent_messages"])}')

    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Concepts: {len(df["concepts"])}')
    for c in df['concepts']:
        print(f'  {c["name"]}')
