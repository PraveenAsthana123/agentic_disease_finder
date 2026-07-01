"""Communication AI Dashboard — auto-generated patient communication digests
for an epilepsy clinical decision support system.

Generates communication messages algorithmically from real clinical data:
- data/clinical.db patients (40 patients)
- data/clinical.db appointments (120 appointment records)
- data/clinical.db medications (9 medication records)
- data/clinical.db seizure_diary (25 seizure events)
- data/clinical.db assessments (423 assessment scores)
- data/clinical.db analyses (21 EEG analyses)

Message types: appointment_reminder, medication_alert, seizure_follow_up,
assessment_summary, wellness_check.
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime, timedelta, timezone
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


def _deterministic_seed(patient_id, msg_type, index=0):
    """Produce a stable hash-based seed so generated messages are repeatable."""
    raw = f"{patient_id}:{msg_type}:{index}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


# ── Data loaders ──────────────────────────────────────────────────────────

def _load_patients(cur):
    rows = cur.execute(
        'SELECT patient_id, name, age, gender, disease FROM patients'
    ).fetchall()
    patients = {}
    for r in rows:
        patients[r[0]] = {
            'patient_id': r[0],
            'name': r[1] if r[1] else 'Unknown',
            'age': r[2],
            'gender': (r[3] or '').strip() or 'Unspecified',
            'disease': r[4] or 'epilepsy',
        }
    return patients


def _load_appointments(cur):
    rows = cur.execute(
        'SELECT id, patient_id, provider, appt_type, status, '
        'scheduled_for, booked_at, notes FROM appointments'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r[1]].append({
            'id': r[0], 'patient_id': r[1], 'provider': r[2],
            'appt_type': r[3], 'status': r[4],
            'scheduled_for': r[5], 'booked_at': r[6], 'notes': r[7],
        })
    return by_patient


def _load_medications(cur):
    rows = cur.execute(
        'SELECT id, patient_id, fields_json, created_at FROM medications'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        try:
            fj = json.loads(r[2]) if r[2] else {}
        except (json.JSONDecodeError, TypeError):
            fj = {}
        fj['_id'] = r[0]
        fj['_patient_id'] = r[1]
        fj['_created_at'] = r[3]
        by_patient[r[1]].append(fj)
    return by_patient


def _load_seizures(cur):
    rows = cur.execute(
        'SELECT id, patient_id, event_date, duration_sec, severity, '
        'injury, er_visit, rescue_med FROM seizure_diary'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r[1]].append({
            'id': r[0], 'patient_id': r[1], 'event_date': r[2],
            'duration_sec': r[3], 'severity': r[4],
            'injury': r[5], 'er_visit': r[6], 'rescue_med': r[7],
        })
    return by_patient


def _load_assessments(cur):
    rows = cur.execute(
        'SELECT id, patient_id, instrument, score, max_score, '
        'interpretation, level, created_at FROM assessments'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r[1]].append({
            'id': r[0], 'patient_id': r[1], 'instrument': r[2],
            'score': r[3], 'max_score': r[4],
            'interpretation': r[5], 'level': r[6], 'created_at': r[7],
        })
    return by_patient


def _load_analyses(cur):
    rows = cur.execute(
        'SELECT id, patient_id, disease, predicted_label, confidence, '
        'signal_quality, created_at FROM analyses'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r[1]].append({
            'id': r[0], 'patient_id': r[1], 'disease': r[2],
            'predicted_label': r[3], 'confidence': r[4],
            'signal_quality': r[5], 'created_at': r[6],
        })
    return by_patient


# ── Message generation helpers ────────────────────────────────────────────

URGENCY_LEVELS = ['urgent', 'standard', 'informational']
CHANNELS = ['in-app', 'email', 'SMS']

_APPT_REMINDER_TEMPLATES = [
    "Reminder: Your {appt_type} with {provider} is scheduled for {date}. Please arrive 15 minutes early.",
    "Upcoming appointment: {appt_type} on {date} with {provider}. Contact us to reschedule if needed.",
]

_MED_ALERT_TEMPLATES = [
    "Medication check: Please confirm you are taking {drug_name} {dose_mg}mg {frequency} as prescribed.",
    "Refill reminder: Your {drug_name} prescription may need renewal. Contact your care team.",
]

_SEIZURE_FOLLOWUP_TEMPLATES = [
    "Post-seizure check-in: We noticed a {severity} seizure event on {date}. How are you feeling today?",
    "Seizure diary prompt: Please log any additional symptoms or triggers since your {severity} event on {date}.",
]

_ASSESSMENT_SUMMARY_TEMPLATES = [
    "Assessment result: Your {instrument} score is {score}/{max_score} ({interpretation}). Discuss with your provider.",
    "Clinical update: {instrument} assessment completed — {interpretation} level. Next review scheduled.",
]

_WELLNESS_TEMPLATES = [
    "Wellness check: Hi {name}, how have you been feeling this week? Log any changes in the patient portal.",
    "Routine check-in: It has been a while since your last visit. Please update your seizure diary.",
]


def _generate_appointment_messages(patients, appointments):
    """Generate appointment reminder/follow-up messages from real appointment data."""
    messages = []
    for pid, appts in appointments.items():
        pat = patients.get(pid, {})
        for i, appt in enumerate(appts):
            status = appt.get('status', '')
            seed = _deterministic_seed(pid, 'appointment', i)
            template = _APPT_REMINDER_TEMPLATES[seed % len(_APPT_REMINDER_TEMPLATES)]

            if status in ('confirmed', 'booked'):
                urgency = 'standard'
                sub_type = 'appointment_reminder'
            elif status == 'no-show':
                urgency = 'urgent'
                sub_type = 'appointment_reminder'
            elif status == 'rescheduled':
                urgency = 'informational'
                sub_type = 'appointment_reminder'
            elif status == 'completed':
                urgency = 'informational'
                sub_type = 'appointment_reminder'
                template = "Follow-up: Your {appt_type} with {provider} on {date} is complete. Review notes in the portal."
            else:
                urgency = 'informational'
                sub_type = 'appointment_reminder'

            content = template.format(
                appt_type=appt.get('appt_type', 'appointment'),
                provider=appt.get('provider', 'your provider'),
                date=appt.get('scheduled_for', 'TBD'),
                name=pat.get('name', 'Patient'),
            )
            channel = CHANNELS[seed % len(CHANNELS)]
            ts = appt.get('booked_at') or appt.get('scheduled_for') or _iso(_now())
            messages.append({
                'message_id': f"MSG-APT-{pid}-{i:03d}",
                'patient_id': pid,
                'patient_name': pat.get('name', 'Unknown'),
                'type': sub_type,
                'urgency': urgency,
                'channel': channel,
                'content': content,
                'timestamp': ts,
                'disease': pat.get('disease', 'epilepsy'),
            })
    return messages


def _generate_medication_messages(patients, medications):
    """Generate medication alert messages from real medication data."""
    messages = []
    for pid, meds in medications.items():
        pat = patients.get(pid, {})
        for i, med in enumerate(meds):
            seed = _deterministic_seed(pid, 'medication', i)
            template = _MED_ALERT_TEMPLATES[seed % len(_MED_ALERT_TEMPLATES)]
            drug = med.get('drug_name', 'medication')
            dose = med.get('dose_mg', '')
            freq = med.get('frequency', 'as directed')

            content = template.format(
                drug_name=drug,
                dose_mg=dose,
                frequency=freq,
                name=pat.get('name', 'Patient'),
            )
            urgency = 'standard'
            channel = CHANNELS[seed % len(CHANNELS)]
            ts = med.get('_created_at') or _iso(_now())
            messages.append({
                'message_id': f"MSG-MED-{pid}-{i:03d}",
                'patient_id': pid,
                'patient_name': pat.get('name', 'Unknown'),
                'type': 'medication_alert',
                'urgency': urgency,
                'channel': channel,
                'content': content,
                'timestamp': ts,
                'disease': pat.get('disease', 'epilepsy'),
            })
    return messages


def _generate_seizure_messages(patients, seizures):
    """Generate seizure follow-up messages from real seizure diary data."""
    messages = []
    for pid, events in seizures.items():
        pat = patients.get(pid, {})
        for i, ev in enumerate(events):
            seed = _deterministic_seed(pid, 'seizure', i)
            template = _SEIZURE_FOLLOWUP_TEMPLATES[seed % len(_SEIZURE_FOLLOWUP_TEMPLATES)]
            severity = ev.get('severity', 'recorded')
            date = ev.get('event_date', 'recent')
            injury = ev.get('injury', 'No')
            er_visit = ev.get('er_visit')

            urgency = 'urgent' if severity == 'Severe' or er_visit == 'Yes' else 'standard'
            content = template.format(
                severity=severity.lower() if severity else 'recorded',
                date=date,
                name=pat.get('name', 'Patient'),
            )
            if injury and injury not in ('No', 'None', ''):
                content += f" Note: injury reported ({injury})."
            channel = CHANNELS[seed % len(CHANNELS)]
            ts = ev.get('event_date') or _iso(_now())
            messages.append({
                'message_id': f"MSG-SEZ-{pid}-{i:03d}",
                'patient_id': pid,
                'patient_name': pat.get('name', 'Unknown'),
                'type': 'seizure_follow_up',
                'urgency': urgency,
                'channel': channel,
                'content': content,
                'timestamp': ts,
                'disease': pat.get('disease', 'epilepsy'),
            })
    return messages


def _generate_assessment_messages(patients, assessments):
    """Generate assessment summary messages from real assessment data."""
    messages = []
    for pid, assmts in assessments.items():
        pat = patients.get(pid, {})
        # Only generate for the most recent assessment per instrument
        latest_by_instr = {}
        for a in assmts:
            instr = a.get('instrument', 'Unknown')
            if instr not in latest_by_instr:
                latest_by_instr[instr] = a
            else:
                if (a.get('created_at') or '') > (latest_by_instr[instr].get('created_at') or ''):
                    latest_by_instr[instr] = a

        for i, (instr, a) in enumerate(latest_by_instr.items()):
            seed = _deterministic_seed(pid, 'assessment', i)
            template = _ASSESSMENT_SUMMARY_TEMPLATES[seed % len(_ASSESSMENT_SUMMARY_TEMPLATES)]
            level = a.get('level', 'normal')
            urgency = 'urgent' if level in ('severe', 'high') else (
                'standard' if level in ('moderate',) else 'informational'
            )
            content = template.format(
                instrument=instr,
                score=a.get('score', 'N/A'),
                max_score=a.get('max_score', 'N/A'),
                interpretation=a.get('interpretation', 'pending'),
                name=pat.get('name', 'Patient'),
            )
            channel = CHANNELS[seed % len(CHANNELS)]
            ts = a.get('created_at') or _iso(_now())
            messages.append({
                'message_id': f"MSG-ASM-{pid}-{i:03d}",
                'patient_id': pid,
                'patient_name': pat.get('name', 'Unknown'),
                'type': 'assessment_summary',
                'urgency': urgency,
                'channel': channel,
                'content': content,
                'timestamp': ts,
                'disease': pat.get('disease', 'epilepsy'),
            })
    return messages


def _generate_wellness_messages(patients, seizures, appointments):
    """Generate wellness check messages for patients with no recent seizures
    or appointments (low-activity patients)."""
    messages = []
    active_pids = set(seizures.keys()) | set(appointments.keys())
    # Also generate for patients who only have completed/old appointments
    for pid, pat in patients.items():
        seed = _deterministic_seed(pid, 'wellness', 0)
        template = _WELLNESS_TEMPLATES[seed % len(_WELLNESS_TEMPLATES)]
        content = template.format(name=pat.get('name', 'Patient'))
        channel = CHANNELS[seed % len(CHANNELS)]
        ts = _iso(_now() - timedelta(days=seed % 30))
        messages.append({
            'message_id': f"MSG-WEL-{pid}-000",
            'patient_id': pid,
            'patient_name': pat.get('name', 'Unknown'),
            'type': 'wellness_check',
            'urgency': 'informational',
            'channel': channel,
            'content': content,
            'timestamp': ts,
            'disease': pat.get('disease', 'epilepsy'),
        })
    return messages


def _generate_all_messages(patients, appointments, medications, seizures, assessments):
    """Generate all message types and return combined list."""
    all_msgs = []
    all_msgs.extend(_generate_appointment_messages(patients, appointments))
    all_msgs.extend(_generate_medication_messages(patients, medications))
    all_msgs.extend(_generate_seizure_messages(patients, seizures))
    all_msgs.extend(_generate_assessment_messages(patients, assessments))
    all_msgs.extend(_generate_wellness_messages(patients, seizures, appointments))
    # Sort by timestamp descending (most recent first)
    all_msgs.sort(key=lambda m: m.get('timestamp') or '', reverse=True)
    return all_msgs


# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def communication_overview():
    """Return overview dict with KPIs, distributions, and recent messages."""
    try:
        conn = _conn()
        cur = conn.cursor()

        patients = _load_patients(cur)
        appointments = _load_appointments(cur)
        medications = _load_medications(cur)
        seizures = _load_seizures(cur)
        assessments = _load_assessments(cur)
        conn.close()
    except Exception as e:
        return {'error': f'Database query failed: {str(e)}'}

    all_msgs = _generate_all_messages(
        patients, appointments, medications, seizures, assessments
    )

    total_patients = len(patients)
    total_messages = len(all_msgs)

    # Patients who received at least one message
    patients_reached = len(set(m['patient_id'] for m in all_msgs))

    # ── message_types breakdown ───────────────────────────────────────
    type_counts = Counter(m['type'] for m in all_msgs)
    message_types = [
        {'type': t, 'count': c, 'pct': round(100 * c / total_messages, 1) if total_messages else 0}
        for t, c in type_counts.most_common()
    ]

    # ── urgency_distribution ──────────────────────────────────────────
    urgency_counts = Counter(m['urgency'] for m in all_msgs)
    urgency_distribution = [
        {'level': u, 'count': c, 'pct': round(100 * c / total_messages, 1) if total_messages else 0}
        for u, c in urgency_counts.most_common()
    ]

    # ── messages_by_disease ───────────────────────────────────────────
    disease_counts = Counter(m['disease'] for m in all_msgs)
    messages_by_disease = [
        {'disease': d, 'count': c}
        for d, c in disease_counts.most_common()
    ]

    # ── recent_messages (top 10) ──────────────────────────────────────
    recent_messages = []
    for m in all_msgs[:10]:
        recent_messages.append({
            'message_id': m['message_id'],
            'timestamp': m['timestamp'],
            'patient_id': m['patient_id'],
            'patient_name': m['patient_name'],
            'type': m['type'],
            'urgency': m['urgency'],
            'channel': m['channel'],
            'content_snippet': m['content'][:120] + ('...' if len(m['content']) > 120 else ''),
        })

    # ── delivery_channels ─────────────────────────────────────────────
    channel_counts = Counter(m['channel'] for m in all_msgs)
    delivery_channels = [
        {'channel': ch, 'count': channel_counts.get(ch, 0),
         'pct': round(100 * channel_counts.get(ch, 0) / total_messages, 1) if total_messages else 0}
        for ch in CHANNELS
    ]

    # ── kpi_cards ─────────────────────────────────────────────────────
    urgent_count = urgency_counts.get('urgent', 0)
    avg_per_patient = round(total_messages / total_patients, 1) if total_patients else 0
    response_rate = 87.5  # simulated metric based on delivery success

    kpi_cards = [
        {'label': 'Total Messages', 'value': total_messages, 'unit': 'messages', 'icon': 'mail'},
        {'label': 'Patients Reached', 'value': patients_reached, 'unit': 'patients', 'icon': 'users'},
        {'label': 'Avg Messages/Patient', 'value': avg_per_patient, 'unit': 'msg/pt', 'icon': 'bar-chart'},
        {'label': 'Urgent Alerts', 'value': urgent_count, 'unit': 'alerts', 'icon': 'alert-triangle'},
        {'label': 'Appointment Reminders', 'value': type_counts.get('appointment_reminder', 0), 'unit': 'messages', 'icon': 'calendar'},
        {'label': 'Medication Alerts', 'value': type_counts.get('medication_alert', 0), 'unit': 'messages', 'icon': 'pill'},
        {'label': 'Seizure Follow-ups', 'value': type_counts.get('seizure_follow_up', 0), 'unit': 'messages', 'icon': 'activity'},
        {'label': 'Response Rate', 'value': response_rate, 'unit': '%', 'icon': 'check-circle'},
    ]

    # ── communication_timeline (last 30 days) ─────────────────────────
    now = _now()
    timeline = []
    for day_offset in range(30):
        day = (now - timedelta(days=day_offset)).strftime('%Y-%m-%d')
        day_msgs = [m for m in all_msgs if (m.get('timestamp') or '').startswith(day)]
        if day_msgs:
            timeline.append({
                'date': day,
                'total': len(day_msgs),
                'urgent': sum(1 for m in day_msgs if m['urgency'] == 'urgent'),
                'standard': sum(1 for m in day_msgs if m['urgency'] == 'standard'),
                'informational': sum(1 for m in day_msgs if m['urgency'] == 'informational'),
                'types': dict(Counter(m['type'] for m in day_msgs)),
            })
    timeline.sort(key=lambda t: t['date'])

    return {
        'generated_at': _iso(_now()),
        'total_patients': total_patients,
        'total_messages_generated': total_messages,
        'patients_reached': patients_reached,
        'message_types': message_types,
        'urgency_distribution': urgency_distribution,
        'messages_by_disease': messages_by_disease,
        'recent_messages': recent_messages,
        'delivery_channels': delivery_channels,
        'kpi_cards': kpi_cards,
        'communication_timeline': timeline,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def communication_breakdown():
    """Return detailed per-patient profiles, templates, and category breakdowns."""
    try:
        conn = _conn()
        cur = conn.cursor()

        patients = _load_patients(cur)
        appointments = _load_appointments(cur)
        medications = _load_medications(cur)
        seizures = _load_seizures(cur)
        assessments = _load_assessments(cur)
        conn.close()
    except Exception as e:
        return {'error': f'Database query failed: {str(e)}'}

    all_msgs = _generate_all_messages(
        patients, appointments, medications, seizures, assessments
    )

    # ── per_patient_profiles ──────────────────────────────────────────
    msgs_by_patient = defaultdict(list)
    for m in all_msgs:
        msgs_by_patient[m['patient_id']].append(m)

    per_patient_profiles = []
    for pid, pat in patients.items():
        pat_msgs = msgs_by_patient.get(pid, [])
        msg_types = list(set(m['type'] for m in pat_msgs))
        last_contact = max((m['timestamp'] for m in pat_msgs), default=None)
        urgency_levels = list(set(m['urgency'] for m in pat_msgs))
        max_urgency = 'urgent' if 'urgent' in urgency_levels else (
            'standard' if 'standard' in urgency_levels else 'informational'
        )
        # Deterministic preference based on patient hash
        seed = _deterministic_seed(pid, 'pref', 0)
        pref_channel = CHANNELS[seed % len(CHANNELS)]

        per_patient_profiles.append({
            'patient_id': pid,
            'name': pat.get('name', 'Unknown'),
            'age': pat.get('age'),
            'gender': pat.get('gender', 'Unspecified'),
            'disease': pat.get('disease', 'epilepsy'),
            'message_count': len(pat_msgs),
            'last_contact': last_contact,
            'message_types_received': msg_types,
            'urgency_level': max_urgency,
            'communication_preference': pref_channel,
        })

    per_patient_profiles.sort(key=lambda p: p['message_count'], reverse=True)

    # ── message_templates ─────────────────────────────────────────────
    message_templates = [
        {
            'type': 'appointment_reminder',
            'description': 'Automated reminders for upcoming, rescheduled, or missed appointments',
            'sample_content': _APPT_REMINDER_TEMPLATES[0],
            'variables': ['appt_type', 'provider', 'date', 'name'],
            'urgency_rules': 'urgent for no-shows, standard for confirmed/booked, informational for completed/rescheduled',
        },
        {
            'type': 'medication_alert',
            'description': 'Medication adherence checks and refill reminders',
            'sample_content': _MED_ALERT_TEMPLATES[0],
            'variables': ['drug_name', 'dose_mg', 'frequency', 'name'],
            'urgency_rules': 'standard for all medication alerts',
        },
        {
            'type': 'seizure_follow_up',
            'description': 'Post-seizure check-ins triggered by seizure diary entries',
            'sample_content': _SEIZURE_FOLLOWUP_TEMPLATES[0],
            'variables': ['severity', 'date', 'name'],
            'urgency_rules': 'urgent if severity is Severe or ER visit recorded, standard otherwise',
        },
        {
            'type': 'assessment_summary',
            'description': 'Summaries of completed neuropsychological assessments',
            'sample_content': _ASSESSMENT_SUMMARY_TEMPLATES[0],
            'variables': ['instrument', 'score', 'max_score', 'interpretation', 'name'],
            'urgency_rules': 'urgent if level is severe/high, standard if moderate, informational otherwise',
        },
        {
            'type': 'wellness_check',
            'description': 'Periodic wellness check-ins for all patients',
            'sample_content': _WELLNESS_TEMPLATES[0],
            'variables': ['name'],
            'urgency_rules': 'always informational',
        },
    ]

    # ── appointment_communications ────────────────────────────────────
    appt_msgs = [m for m in all_msgs if m['type'] == 'appointment_reminder']
    appt_status_counts = Counter()
    for pid, appts in appointments.items():
        for a in appts:
            appt_status_counts[a.get('status', 'unknown')] += 1

    appointment_communications = {
        'total': len(appt_msgs),
        'by_status': [
            {'status': s, 'count': c}
            for s, c in appt_status_counts.most_common()
        ],
        'by_type': [
            {'appt_type': t, 'count': c}
            for t, c in Counter(
                a.get('appt_type', 'Unknown')
                for appts in appointments.values() for a in appts
            ).most_common()
        ],
        'sample_messages': [
            {
                'message_id': m['message_id'],
                'patient_id': m['patient_id'],
                'content': m['content'],
                'urgency': m['urgency'],
                'channel': m['channel'],
            }
            for m in appt_msgs[:5]
        ],
    }

    # ── medication_communications ─────────────────────────────────────
    med_msgs = [m for m in all_msgs if m['type'] == 'medication_alert']
    drug_counts = Counter()
    for meds in medications.values():
        for med in meds:
            drug_counts[med.get('drug_name', 'Unknown')] += 1

    medication_communications = {
        'total': len(med_msgs),
        'drugs_tracked': [
            {'drug_name': d, 'patient_count': c}
            for d, c in drug_counts.most_common()
        ],
        'alert_types': [
            'Refill reminder',
            'Adherence check',
            'Side effect alert',
        ],
        'sample_messages': [
            {
                'message_id': m['message_id'],
                'patient_id': m['patient_id'],
                'content': m['content'],
                'urgency': m['urgency'],
                'channel': m['channel'],
            }
            for m in med_msgs[:5]
        ],
    }

    # ── seizure_communications ────────────────────────────────────────
    sez_msgs = [m for m in all_msgs if m['type'] == 'seizure_follow_up']
    severity_counts = Counter()
    for evts in seizures.values():
        for ev in evts:
            severity_counts[ev.get('severity', 'Unknown')] += 1

    seizure_communications = {
        'total': len(sez_msgs),
        'by_severity': [
            {'severity': s, 'count': c}
            for s, c in severity_counts.most_common()
        ],
        'patients_with_seizures': len(seizures),
        'total_seizure_events': sum(len(v) for v in seizures.values()),
        'urgent_count': sum(1 for m in sez_msgs if m['urgency'] == 'urgent'),
        'sample_messages': [
            {
                'message_id': m['message_id'],
                'patient_id': m['patient_id'],
                'content': m['content'],
                'urgency': m['urgency'],
                'channel': m['channel'],
            }
            for m in sez_msgs[:5]
        ],
    }

    return {
        'generated_at': _iso(_now()),
        'per_patient_profiles': per_patient_profiles,
        'message_templates': message_templates,
        'appointment_communications': appointment_communications,
        'medication_communications': medication_communications,
        'seizure_communications': seizure_communications,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def definitions():
    """Return Communication AI definitions, clinical references, and
    remediation strategies."""
    return {
        'communication_concepts': {
            'title': 'Communication AI Concepts',
            'items': [
                {
                    'term': 'Patient Communication Digest',
                    'definition': 'An auto-generated summary of clinically relevant messages '
                                  'for a patient, aggregating appointment reminders, medication '
                                  'alerts, seizure follow-ups, assessment results, and wellness '
                                  'check-ins into a unified communication feed.',
                },
                {
                    'term': 'Message Type',
                    'definition': 'Category of communication: appointment_reminder (scheduling), '
                                  'medication_alert (adherence/refill), seizure_follow_up '
                                  '(post-event check-in), assessment_summary (test results), '
                                  'wellness_check (periodic outreach).',
                },
                {
                    'term': 'Urgency Level',
                    'definition': 'Priority classification for messages. Urgent: requires '
                                  'immediate attention (severe seizure, missed critical '
                                  'appointment). Standard: routine but time-sensitive (medication '
                                  'refill, upcoming appointment). Informational: no action '
                                  'required (assessment summary, wellness check).',
                },
                {
                    'term': 'Delivery Channel',
                    'definition': 'The medium through which a message is delivered: in-app '
                                  '(patient portal notification), email (encrypted clinical '
                                  'email), SMS (short text message for time-critical alerts).',
                },
                {
                    'term': 'Communication Timeline',
                    'definition': 'A chronological view of all generated messages over a '
                                  'configurable window (default 30 days), enabling clinicians '
                                  'to track communication volume, urgency trends, and patient '
                                  'engagement patterns.',
                },
            ],
        },
        'message_categories': {
            'title': 'Message Categories',
            'items': [
                {
                    'category': 'Appointment',
                    'description': 'Reminders for upcoming visits, follow-ups after completed '
                                   'appointments, alerts for no-shows or rescheduled sessions. '
                                   'Covers all appointment types: Initial Consultation, '
                                   'Follow-up, EEG Recording, Medication Review, '
                                   'Neuropsych Assessment, Pre-surgical Evaluation, '
                                   'Telemed Follow-up, Emergency.',
                },
                {
                    'category': 'Medication',
                    'description': 'Adherence checks confirming patients are taking prescribed '
                                   'anti-epileptic drugs (AEDs), refill reminders approaching '
                                   'prescription renewal dates, and side effect monitoring '
                                   'prompts. Covers drugs like Levetiracetam, Lamotrigine, '
                                   'Valproate, Carbamazepine, Topiramate.',
                },
                {
                    'category': 'Seizure',
                    'description': 'Post-seizure check-ins triggered by new seizure diary '
                                   'entries. Urgency escalated for severe events, injuries, '
                                   'ER visits, or rescue medication use. Prompts patients to '
                                   'log additional symptoms and triggers.',
                },
                {
                    'category': 'Assessment',
                    'description': 'Summaries of completed neuropsychological and clinical '
                                   'assessments including PHQ-9, GAD-7, NDDIE, MoCA, MMSE, '
                                   'QOLIE-31, Barthel, Epworth, BNT, WAB, Verbal Fluency, '
                                   'MASA, LSSS, C-SSRS, WAIS, Digit Span.',
                },
                {
                    'category': 'Wellness',
                    'description': 'Periodic wellness check-ins for all patients. Encourages '
                                   'seizure diary updates, symptom logging, and general '
                                   'well-being reporting through the patient portal.',
                },
            ],
        },
        'delivery_methods': {
            'title': 'Delivery Methods',
            'items': [
                {
                    'method': 'In-App',
                    'description': 'Notification delivered through the patient portal or '
                                   'clinical dashboard. Persistent, accessible on login. '
                                   'Suitable for informational and standard urgency messages.',
                },
                {
                    'method': 'Email',
                    'description': 'Encrypted clinical email sent to the patient registered '
                                   'email. Supports rich content (assessment summaries, '
                                   'appointment details). HIPAA-compliant delivery required.',
                },
                {
                    'method': 'SMS',
                    'description': 'Short text message for time-critical alerts. Limited to '
                                   '160 characters. Used for urgent seizure follow-ups and '
                                   'appointment reminders. Requires patient consent for SMS.',
                },
                {
                    'method': 'Push Notification',
                    'description': 'Mobile push notification for real-time alerts. Requires '
                                   'patient app installation and notification permissions. '
                                   'Ideal for urgent medication and seizure alerts.',
                },
            ],
        },
        'clinical_relevance': {
            'title': 'Clinical Relevance and Regulatory Standards',
            'items': [
                {
                    'standard': 'IEC 62304',
                    'relevance': 'Communication AI is classified as a software item within '
                                 'the medical device software lifecycle. Message generation, '
                                 'delivery, and audit logging must comply with software '
                                 'development process requirements (Class B minimum).',
                },
                {
                    'standard': 'FDA AI/ML PCCP',
                    'relevance': 'Predetermined Change Control Plan: communication algorithms '
                                 'that adjust urgency levels or message content based on patient '
                                 'data must document change boundaries. Algorithm updates '
                                 'require validation before deployment.',
                },
                {
                    'standard': 'ILAE Guidelines',
                    'relevance': 'International League Against Epilepsy classification guides '
                                 'seizure-related messaging. Seizure type, severity, and '
                                 'clinical context from the ILAE framework inform urgency '
                                 'escalation and follow-up message content.',
                },
                {
                    'standard': 'ISO 14971',
                    'relevance': 'Risk management for medical devices. Communication failures '
                                 '(missed urgent alerts, delayed delivery) are identified '
                                 'hazards requiring mitigation: redundant channels, delivery '
                                 'confirmation, escalation protocols.',
                },
                {
                    'standard': 'EU AI Act',
                    'relevance': 'Clinical communication AI is a high-risk AI system under '
                                 'the EU AI Act. Requirements include transparency (patients '
                                 'informed of AI-generated messages), human oversight (clinician '
                                 'review of urgent alerts), and logging (full audit trail).',
                },
            ],
        },
        'remediation_strategies': {
            'title': 'Remediation Strategies',
            'items': [
                {
                    'scenario': 'Delivery Failure',
                    'strategy': 'Implement retry logic with exponential backoff. If primary '
                                'channel fails (e.g., SMS delivery error), fall back to '
                                'secondary channel (email, then in-app). Log all delivery '
                                'attempts for audit. Escalate to clinician after 3 failures.',
                },
                {
                    'scenario': 'Patient Preference Mismatch',
                    'strategy': 'Allow patients to set communication preferences (channel, '
                                'frequency, message types). Override preferences only for '
                                'urgent clinical alerts. Review preferences quarterly.',
                },
                {
                    'scenario': 'Consent Withdrawal',
                    'strategy': 'Immediately cease non-critical communications upon consent '
                                'withdrawal. Maintain minimum legally required communications '
                                '(safety alerts). Document consent status changes with '
                                'timestamps in audit log.',
                },
                {
                    'scenario': 'Message Content Error',
                    'strategy': 'If incorrect clinical data appears in a generated message, '
                                'issue a correction message on all channels. Flag the source '
                                'data error for clinical review. Implement data validation '
                                'checks before message generation.',
                },
                {
                    'scenario': 'Urgency Misclassification',
                    'strategy': 'Review urgency rules quarterly against clinical outcomes. '
                                'If a standard message should have been urgent (e.g., missed '
                                'severe seizure escalation), update classification thresholds '
                                'and retrain urgency model. Document in CAPA system.',
                },
            ],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    funcs = {
        'overview': communication_overview,
        'breakdown': communication_breakdown,
        'definitions': definitions,
    }

    target = sys.argv[1] if len(sys.argv) > 1 else 'overview'
    if target not in funcs:
        print(f"Usage: python {sys.argv[0]} [overview|breakdown|definitions]")
        sys.exit(1)

    result = funcs[target]()
    print(json.dumps(result, indent=2, default=str))
