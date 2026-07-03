"""Emergency Contact & Caregiver Dashboard — Patient Module Section 2.
Tracks emergency contacts, caregiver profiles, training status,
caregiver burden metrics, and safety/emergency plans for epilepsy patients.

Populates and reads from:
  - emergency_contacts  (one per patient: contact name, phone, email, relationship)
  - caregivers          (one per patient: profile, training, burden, safety plans)

Uses real patient_ids from the patients table (first 30).
ILAE caregiver education standards and Epilepsy Foundation guidelines applied throughout.
"""

import json
import os
import random
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

RELATIONSHIP_TYPES = [
    'spouse', 'parent', 'child', 'sibling', 'partner',
    'grandparent', 'friend', 'neighbor', 'professional_caregiver',
]

CAREGIVER_ROLES = ['parent', 'spouse', 'sibling', 'child', 'professional', 'friend']

AVAILABILITY_OPTIONS = ['full-time', 'part-time', 'on-call', 'weekends']

TRAINING_TOPICS = [
    'Seizure recognition',
    'Timing seizures',
    'Recovery position',
    'When to call 911',
    'Rescue medication administration',
    'Seizure first aid basics',
    'SUDEP awareness',
    'Medication management',
    'Epilepsy triggers',
    'Emotional support techniques',
    'Safety hazard assessment',
    'Emergency action plan review',
]

EMERGENCY_PROTOCOLS = [
    'Stay calm; time the seizure; protect from injury; do NOT restrain; turn on side after convulsions stop; call 911 if >5 min.',
    'Clear area of hazards; cushion head; do not put anything in mouth; note seizure type and duration; administer rescue med if prescribed and trained.',
    'Ensure airway is clear; place in recovery position; monitor breathing; do not leave unattended; call 911 if first seizure or >5 min.',
    'Move sharp objects away; loosen tight clothing; stay with patient until fully alert; record event details for neurologist.',
    'Activate seizure action plan; administer nasal/buccal rescue medication if trained; call emergency services if seizure clusters or prolonged.',
]

WHEN_TO_CALL_911 = [
    'Seizure lasts more than 5 minutes; repeated seizures without regaining consciousness; difficulty breathing after seizure; seizure occurs in water; first-time seizure; injury during seizure; patient is pregnant or diabetic.',
    'Seizure duration exceeds 5 minutes; patient does not return to baseline within 30 minutes; rescue medication fails to stop seizure; patient has no known seizure history; breathing difficulties post-ictal.',
    'Prolonged seizure (>5 min); status epilepticus suspected; injury sustained; seizure in water or dangerous environment; patient requests emergency services; unknown seizure history.',
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
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def _get_patient_ids():
    """Return first 30 patient_ids from the patients table."""
    if not os.path.exists(DB):
        return []
    conn = _db_conn()
    try:
        patients = conn.execute(
            'SELECT patient_id FROM patients'
        ).fetchall()
        patients = [dict(p) for p in patients]
        epat = [p for p in patients if p['patient_id'].startswith('EPAT')]
        others = [p for p in patients if not p['patient_id'].startswith('EPAT')]
        ordered = epat + others
        return [p['patient_id'] for p in ordered[:30]]
    finally:
        conn.close()


def _ensure_tables():
    """Create and populate emergency_contacts and caregivers tables if they don't exist or are empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        # --- emergency_contacts table ---
        conn.execute('''CREATE TABLE IF NOT EXISTS emergency_contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            contact_name TEXT,
            phone TEXT,
            email TEXT,
            relationship TEXT,
            is_primary INTEGER,
            notify_on_seizure INTEGER,
            last_verified TEXT,
            created_at TEXT
        )''')
        conn.commit()

        # --- caregivers table ---
        conn.execute('''CREATE TABLE IF NOT EXISTS caregivers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            name TEXT,
            role TEXT,
            availability TEXT,
            experience_years INTEGER,
            epilepsy_training_completed INTEGER,
            training_topics TEXT,
            first_aid_certified INTEGER,
            rescue_med_trained INTEGER,
            seizure_first_aid_confidence INTEGER,
            caregiver_stress INTEGER,
            caregiver_sleep_quality INTEGER,
            work_impact INTEGER,
            burnout_score INTEGER,
            last_respite_date TEXT,
            safety_plan_exists INTEGER,
            seizure_action_plan_exists INTEGER,
            emergency_protocol TEXT,
            when_to_call_911 TEXT,
            notes TEXT,
            created_at TEXT
        )''')
        conn.commit()

        # Check if already populated
        ec_count = conn.execute('SELECT COUNT(*) FROM emergency_contacts').fetchone()[0]
        cg_count = conn.execute('SELECT COUNT(*) FROM caregivers').fetchone()[0]
        if ec_count > 0 and cg_count > 0:
            return  # already populated

        patient_ids = _get_patient_ids()
        if not patient_ids:
            return

        rng = random.Random(99)

        # --- First name / last name pools ---
        first_names = [
            'Maria', 'James', 'Sarah', 'Robert', 'Linda', 'Michael', 'Jennifer',
            'William', 'Patricia', 'David', 'Elizabeth', 'Richard', 'Barbara',
            'Joseph', 'Susan', 'Thomas', 'Jessica', 'Charles', 'Karen', 'Daniel',
            'Nancy', 'Matthew', 'Lisa', 'Anthony', 'Betty', 'Mark', 'Margaret',
            'Donald', 'Sandra', 'Steven',
        ]
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
            'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
            'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
            'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
            'Ramirez', 'Lewis', 'Robinson',
        ]

        base_date = datetime(2025, 6, 15)

        # --- Populate emergency_contacts (one per patient) ---
        if ec_count == 0:
            for i, pid in enumerate(patient_ids):
                contact_first = rng.choice(first_names)
                contact_last = rng.choice(last_names)
                contact_name = f'{contact_first} {contact_last}'
                phone = f'+1-{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}'
                email = f'{contact_first.lower()}.{contact_last.lower()}{rng.randint(1,99)}@email.com'
                relationship = rng.choice(RELATIONSHIP_TYPES)
                is_primary = 1
                notify_on_seizure = 1 if rng.random() < 0.85 else 0
                days_ago = rng.randint(1, 180)
                last_verified = (base_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')

                conn.execute(
                    '''INSERT INTO emergency_contacts
                       (patient_id, contact_name, phone, email, relationship,
                        is_primary, notify_on_seizure, last_verified, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                    (pid, contact_name, phone, email, relationship,
                     is_primary, notify_on_seizure, last_verified)
                )

        # --- Populate caregivers (one per patient) ---
        if cg_count == 0:
            for i, pid in enumerate(patient_ids):
                cg_first = rng.choice(first_names)
                cg_last = rng.choice(last_names)
                cg_name = f'{cg_first} {cg_last}'
                role = rng.choice(CAREGIVER_ROLES)
                availability = rng.choice(AVAILABILITY_OPTIONS)
                experience_years = rng.randint(0, 20)

                # Training
                epilepsy_training_completed = 1 if rng.random() < 0.70 else 0
                if epilepsy_training_completed:
                    num_topics = rng.randint(3, len(TRAINING_TOPICS))
                    topics = rng.sample(TRAINING_TOPICS, num_topics)
                else:
                    topics = rng.sample(TRAINING_TOPICS, rng.randint(0, 2))
                training_topics_json = json.dumps(topics)

                first_aid_certified = 1 if rng.random() < 0.65 else 0
                rescue_med_trained = 1 if rng.random() < 0.55 else 0
                seizure_first_aid_confidence = rng.randint(1, 10)

                # Burden scores
                caregiver_stress = rng.randint(1, 10)
                caregiver_sleep_quality = rng.randint(1, 10)
                work_impact = rng.randint(1, 10)
                burnout_score = rng.randint(10, 95)

                # Respite
                respite_days_ago = rng.randint(7, 365)
                last_respite_date = (base_date - timedelta(days=respite_days_ago)).strftime('%Y-%m-%d')

                # Safety plans
                safety_plan_exists = 1 if rng.random() < 0.72 else 0
                seizure_action_plan_exists = 1 if rng.random() < 0.68 else 0
                emergency_protocol = rng.choice(EMERGENCY_PROTOCOLS)
                when_to_call_911 = rng.choice(WHEN_TO_CALL_911)

                # Notes
                notes_pool = [
                    '', '', '',
                    'Attended Epilepsy Foundation workshop last month.',
                    'Expressed concern about long-term caregiving sustainability.',
                    'Requested additional rescue medication training.',
                    'Works night shifts — availability limited on weekdays.',
                    'Lives with patient full-time.',
                    'Recently completed online seizure first aid certification.',
                    'Reports difficulty sleeping due to seizure monitoring.',
                    'Has previous nursing experience.',
                    'Requested respite care information.',
                    'Primary decision-maker for patient care.',
                ]
                notes = rng.choice(notes_pool)

                conn.execute(
                    '''INSERT INTO caregivers
                       (patient_id, name, role, availability, experience_years,
                        epilepsy_training_completed, training_topics,
                        first_aid_certified, rescue_med_trained,
                        seizure_first_aid_confidence,
                        caregiver_stress, caregiver_sleep_quality, work_impact,
                        burnout_score, last_respite_date,
                        safety_plan_exists, seizure_action_plan_exists,
                        emergency_protocol, when_to_call_911, notes, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                    (pid, cg_name, role, availability, experience_years,
                     epilepsy_training_completed, training_topics_json,
                     first_aid_certified, rescue_med_trained,
                     seizure_first_aid_confidence,
                     caregiver_stress, caregiver_sleep_quality, work_impact,
                     burnout_score, last_respite_date,
                     safety_plan_exists, seizure_action_plan_exists,
                     emergency_protocol, when_to_call_911, notes)
                )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def overview():
    """Return KPI cards + chart data for the Emergency Contact & Caregiver overview tab."""
    _ensure_tables()

    contacts = _db_query('SELECT * FROM emergency_contacts')
    caregivers = _db_query('SELECT * FROM caregivers')

    total_patients = len(set(
        [c['patient_id'] for c in contacts] + [g['patient_id'] for g in caregivers]
    ))
    total_emergency_contacts = len(contacts)
    total_caregivers = len(caregivers)

    # Safety plan & training rates
    pct_with_safety_plan = round(
        sum(1 for g in caregivers if g.get('safety_plan_exists')) / len(caregivers) * 100, 1
    ) if caregivers else 0
    pct_first_aid_certified = round(
        sum(1 for g in caregivers if g.get('first_aid_certified')) / len(caregivers) * 100, 1
    ) if caregivers else 0
    pct_rescue_med_trained = round(
        sum(1 for g in caregivers if g.get('rescue_med_trained')) / len(caregivers) * 100, 1
    ) if caregivers else 0

    # Burden averages
    avg_caregiver_stress = _avg([g.get('caregiver_stress', 5) for g in caregivers])
    avg_burnout_score = _avg([g.get('burnout_score', 50) for g in caregivers])
    avg_sleep_quality = _avg([g.get('caregiver_sleep_quality', 5) for g in caregivers])
    avg_work_impact = _avg([g.get('work_impact', 5) for g in caregivers])
    avg_first_aid_confidence = _avg([g.get('seizure_first_aid_confidence', 5) for g in caregivers])
    avg_experience_years = _avg([g.get('experience_years', 0) for g in caregivers])

    # Training completion rate
    pct_epilepsy_training = round(
        sum(1 for g in caregivers if g.get('epilepsy_training_completed')) / len(caregivers) * 100, 1
    ) if caregivers else 0

    # Seizure action plan rate
    pct_seizure_action_plan = round(
        sum(1 for g in caregivers if g.get('seizure_action_plan_exists')) / len(caregivers) * 100, 1
    ) if caregivers else 0

    # Notify on seizure rate
    pct_notify_on_seizure = round(
        sum(1 for c in contacts if c.get('notify_on_seizure')) / len(contacts) * 100, 1
    ) if contacts else 0

    # --- Chart data ---

    # Relationship distribution (pie)
    rel_counts = Counter(c.get('relationship', 'unknown') for c in contacts)
    relationship_distribution = [
        {'relationship': rel, 'count': cnt}
        for rel, cnt in rel_counts.most_common()
    ]

    # Role distribution (pie)
    role_counts = Counter(g.get('role', 'unknown') for g in caregivers)
    role_distribution = [
        {'role': r, 'count': cnt}
        for r, cnt in role_counts.most_common()
    ]

    # Availability breakdown
    avail_counts = Counter(g.get('availability', 'unknown') for g in caregivers)
    availability_breakdown = [
        {'availability': a, 'count': cnt}
        for a, cnt in avail_counts.most_common()
    ]

    # Burden distribution: stress buckets
    stress_buckets = {
        'Low (1-3)': (1, 4),
        'Moderate (4-6)': (4, 7),
        'High (7-8)': (7, 9),
        'Severe (9-10)': (9, 11),
    }
    burden_distribution = []
    for label, (lo, hi) in stress_buckets.items():
        cnt = sum(1 for g in caregivers if lo <= g.get('caregiver_stress', 5) < hi)
        burden_distribution.append({'stress_bucket': label, 'count': cnt})

    # Burnout distribution
    burnout_buckets = {
        'Low (10-30)': (10, 31),
        'Moderate (31-50)': (31, 51),
        'High (51-70)': (51, 71),
        'Critical (71-95)': (71, 96),
    }
    burnout_distribution = []
    for label, (lo, hi) in burnout_buckets.items():
        cnt = sum(1 for g in caregivers if lo <= g.get('burnout_score', 50) < hi)
        burnout_distribution.append({'burnout_bucket': label, 'count': cnt})

    # Training completion rate (topics covered per caregiver)
    training_topic_counts = Counter()
    for g in caregivers:
        topics = _safe_json(g.get('training_topics', '[]'))
        if isinstance(topics, list):
            for t in topics:
                training_topic_counts[t] += 1
    training_completion_rate = [
        {'topic': t, 'caregivers_trained': cnt}
        for t, cnt in training_topic_counts.most_common()
    ]

    return {
        'available': True,
        'total_patients': total_patients,
        'total_emergency_contacts': total_emergency_contacts,
        'total_caregivers': total_caregivers,
        'pct_with_safety_plan': pct_with_safety_plan,
        'pct_first_aid_certified': pct_first_aid_certified,
        'pct_rescue_med_trained': pct_rescue_med_trained,
        'pct_epilepsy_training': pct_epilepsy_training,
        'pct_seizure_action_plan': pct_seizure_action_plan,
        'pct_notify_on_seizure': pct_notify_on_seizure,
        'avg_caregiver_stress': avg_caregiver_stress,
        'avg_burnout_score': avg_burnout_score,
        'avg_sleep_quality': avg_sleep_quality,
        'avg_work_impact': avg_work_impact,
        'avg_first_aid_confidence': avg_first_aid_confidence,
        'avg_experience_years': avg_experience_years,
        'relationship_distribution': relationship_distribution,
        'role_distribution': role_distribution,
        'availability_breakdown': availability_breakdown,
        'burden_distribution': burden_distribution,
        'burnout_distribution': burnout_distribution,
        'training_completion_rate': training_completion_rate,
    }


def breakdown():
    """Return per-patient list with emergency contact + caregiver details."""
    _ensure_tables()

    contacts = _db_query('SELECT * FROM emergency_contacts')
    caregivers = _db_query('SELECT * FROM caregivers')

    # Index by patient_id
    contact_map = {c['patient_id']: c for c in contacts}
    caregiver_map = {g['patient_id']: g for g in caregivers}

    all_pids = sorted(set(list(contact_map.keys()) + list(caregiver_map.keys())))

    patients = []
    for pid in all_pids:
        ec = contact_map.get(pid, {})
        cg = caregiver_map.get(pid, {})

        # Parse training topics
        topics = _safe_json(cg.get('training_topics', '[]'))
        if not isinstance(topics, list):
            topics = []

        patients.append({
            'patient_id': pid,
            # Emergency contact details
            'emergency_contact': {
                'contact_name': ec.get('contact_name', ''),
                'phone': ec.get('phone', ''),
                'email': ec.get('email', ''),
                'relationship': ec.get('relationship', ''),
                'is_primary': bool(ec.get('is_primary', 0)),
                'notify_on_seizure': bool(ec.get('notify_on_seizure', 0)),
                'last_verified': ec.get('last_verified', ''),
            },
            # Caregiver profile
            'caregiver': {
                'name': cg.get('name', ''),
                'role': cg.get('role', ''),
                'availability': cg.get('availability', ''),
                'experience_years': cg.get('experience_years', 0),
            },
            # Training status
            'training_status': {
                'epilepsy_training_completed': bool(cg.get('epilepsy_training_completed', 0)),
                'training_topics': topics,
                'first_aid_certified': bool(cg.get('first_aid_certified', 0)),
                'rescue_med_trained': bool(cg.get('rescue_med_trained', 0)),
                'seizure_first_aid_confidence': cg.get('seizure_first_aid_confidence', 0),
            },
            # Burden scores
            'burden_scores': {
                'caregiver_stress': cg.get('caregiver_stress', 0),
                'caregiver_sleep_quality': cg.get('caregiver_sleep_quality', 0),
                'work_impact': cg.get('work_impact', 0),
                'burnout_score': cg.get('burnout_score', 0),
                'last_respite_date': cg.get('last_respite_date', ''),
            },
            # Safety plan status
            'safety_plan': {
                'safety_plan_exists': bool(cg.get('safety_plan_exists', 0)),
                'seizure_action_plan_exists': bool(cg.get('seizure_action_plan_exists', 0)),
                'emergency_protocol': cg.get('emergency_protocol', ''),
                'when_to_call_911': cg.get('when_to_call_911', ''),
            },
            'notes': cg.get('notes', ''),
        })

    return {
        'patients': patients,
    }


def definitions():
    """Return clinical concepts for the Emergency Contact & Caregiver dashboard."""
    return {
        'concepts': [
            {
                'name': 'Seizure First Aid',
                'description': 'The immediate actions taken during and after a seizure to ensure patient safety. Key principles: stay calm and time the seizure; clear the area of hazardous objects; cushion the head if convulsing; gently roll the person onto their side (recovery position) after convulsions stop to maintain airway patency; do NOT restrain the person or put anything in their mouth; stay with the person until they are fully alert and oriented. For focal aware seizures (previously "simple partial"), guide the person away from danger and provide verbal reassurance. The Epilepsy Foundation\'s Seizure First Aid certification covers all seizure types and is recommended for all caregivers and emergency contacts of people with epilepsy.',
            },
            {
                'name': 'Rescue Medication',
                'description': 'Emergency medications administered outside of hospital settings to terminate prolonged seizures or seizure clusters before emergency medical services arrive. FDA-approved rescue medications include: diazepam rectal gel (Diastat), midazolam nasal spray (Nayzilam), and diazepam nasal spray (Valtoco). Administration is indicated when a seizure exceeds 5 minutes or when cluster seizures occur per the individual seizure action plan. Caregivers must receive hands-on training in the specific rescue medication prescribed, including proper dosing, administration technique, timing, contraindications, and when to call 911 despite medication administration. Re-training annually is recommended. Documentation of each rescue medication use should be reported to the treating neurologist.',
            },
            {
                'name': 'Caregiver Burden',
                'description': 'The multidimensional strain experienced by informal caregivers of people with epilepsy, encompassing physical exhaustion (sleep disruption from nocturnal seizure monitoring), emotional distress (anxiety, helplessness, grief), social isolation (restricted activities, relationship strain), financial impact (reduced work hours, medical costs), and cognitive load (medication management, appointment coordination, trigger monitoring). Measured via standardized scales: Zarit Burden Interview (ZBI), Caregiver Strain Index (CSI), and epilepsy-specific tools like the Epilepsy Caregiver Quality of Life (ECQL). Burnout score (0-100) is a composite metric combining stress, sleep quality, work impact, and respite adequacy. Scores above 70 indicate critical burnout requiring intervention. Regular respite care, support groups, and counseling are evidence-based interventions.',
            },
            {
                'name': 'Seizure Action Plan',
                'description': 'A written, individualized document that outlines step-by-step instructions for responding to a seizure for a specific patient. Developed collaboratively by the neurologist, patient, and caregiver. Components include: patient identification and diagnosis; seizure type descriptions with visual recognition cues; timing protocol (when to start timing, duration thresholds); first aid steps specific to each seizure type; rescue medication instructions (drug, dose, route, maximum doses, interval); when to call 911 vs. manage at home; post-ictal care instructions; emergency contact list; neurologist contact information; current medication list; known triggers; allergies and contraindications. Updated annually or after any change in seizure pattern, medication, or care team. Schools, workplaces, and all caregivers should have current copies. The Epilepsy Foundation provides standardized seizure action plan templates.',
            },
            {
                'name': 'Emergency Protocol',
                'description': 'A comprehensive emergency response procedure for epilepsy-related events that extends beyond immediate seizure first aid. Covers: status epilepticus recognition (continuous seizure >5 minutes or repeated seizures without recovery between them — a medical emergency with mortality risk); seizure-related injuries (falls, burns, drowning prevention); post-ictal psychosis management; SUDEP risk factors and prevention (nocturnal supervision, seizure detection devices, prone position avoidance); emergency room information packet (current medications, seizure history, neurologist contact, DNR status if applicable); medical ID bracelet/necklace recommendations; home safety modifications (shower vs. bath, kitchen safety, stair gates, furniture padding). Emergency protocols should be reviewed with all household members and updated with each clinic visit.',
            },
            {
                'name': 'When to Call 911',
                'description': 'Evidence-based criteria for activating emergency medical services during an epilepsy-related event. Call 911 when: (1) the seizure lasts longer than 5 minutes (risk of status epilepticus); (2) the person does not regain consciousness or return to baseline between seizures (cluster seizures progressing to status); (3) it is the person\'s first known seizure; (4) the person is injured during the seizure; (5) the seizure occurs in water (drowning risk); (6) the person has difficulty breathing after the seizure; (7) the person is pregnant; (8) the person has diabetes (hypoglycemia-induced seizures require different treatment); (9) rescue medication was administered but the seizure continues; (10) there is any uncertainty about the appropriate response. These criteria should be prominently displayed in the patient\'s seizure action plan and known to all caregivers and emergency contacts.',
            },
            {
                'name': 'SUDEP Awareness',
                'description': 'Sudden Unexpected Death in Epilepsy (SUDEP) is the sudden, unexpected, non-traumatic, non-drowning death of a person with epilepsy, with or without evidence of a terminal seizure. SUDEP accounts for 8-17% of deaths in people with epilepsy and is the leading cause of epilepsy-related mortality. Risk factors include: uncontrolled generalized tonic-clonic seizures (highest risk factor), nocturnal seizures, young adult age, long duration of epilepsy, polytherapy, and prone sleeping position post-seizure. Caregiver education about SUDEP is recommended by the AAN (American Academy of Neurology) as part of routine epilepsy counseling. Risk reduction strategies include: optimizing seizure control, medication adherence, seizure detection devices (bed sensors, wearable monitors), nocturnal supervision, and avoiding prone sleeping position. The SUDEP discussion should be sensitive, age-appropriate, and documented in the medical record.',
            },
            {
                'name': 'Epilepsy Foundation Caregiver Resources',
                'description': 'The Epilepsy Foundation provides comprehensive support for caregivers including: (1) Seizure First Aid certification — online and in-person training with certification valid for 2 years; (2) Epilepsy & Seizures 24/7 Helpline (1-800-332-1000) — information, emotional support, and referrals; (3) Caregiver support groups — local and online peer support networks; (4) Epilepsy Learning Healthcare System (ELHS) — evidence-based self-management education; (5) My Seizure Diary — mobile app for tracking seizures, medications, and triggers; (6) Legal advocacy — navigating disability rights, FMLA, ADA accommodations; (7) Financial assistance — connecting caregivers with co-pay assistance and patient assistance programs; (8) Respite care referrals — local and national respite care directories; (9) Webinars and conferences — continuing education on epilepsy management. All resources available at epilepsy.com and through local Epilepsy Foundation affiliates.',
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN (first 3 patients) ===')
    bd = breakdown()
    for p in bd['patients'][:3]:
        pprint.pprint({k: v for k, v in p.items() if k != 'notes'})
    print(f'\nTotal patients: {len(bd["patients"])}')
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Concepts: {len(df["concepts"])}')
    for c in df['concepts']:
        print(f'  {c["name"]}')
