"""Demographics Dashboard — Patient Module Section 1.
Tracks patient demographic profiles including age, sex, ethnicity, race,
language, education, employment, insurance, and epilepsy classification.

Populates and reads from:
  - patient_demographics  (structured demographic data per patient)

Uses real patient_ids from the patients table (first 30).
ILAE Epilepsy Classification, SDOH frameworks applied throughout.
"""

import json
import math
import os
import random
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

SEX_OPTIONS = ['Male', 'Female', 'Other']
GENDER_IDENTITY_OPTIONS = ['Man', 'Woman', 'Non-binary', 'Transgender man', 'Transgender woman', 'Prefer not to say']
ETHNICITY_OPTIONS = ['Hispanic/Latino', 'Non-Hispanic/Latino', 'Unknown']
RACE_OPTIONS = ['White', 'Black/African American', 'Asian', 'Native American', 'Pacific Islander', 'Multiracial']
LANGUAGE_OPTIONS = ['English', 'Spanish', 'Mandarin', 'Hindi', 'Arabic', 'Other']
LANGUAGE_WEIGHTS = [0.70, 0.15, 0.05, 0.03, 0.03, 0.04]
EDUCATION_OPTIONS = ['High school', 'Some college', "Bachelor's", "Master's", 'Doctorate', 'GED', 'Trade school']
EMPLOYMENT_OPTIONS = ['Full-time', 'Part-time', 'Retired', 'Disabled', 'Student', 'Unemployed']
MARITAL_OPTIONS = ['Single', 'Married', 'Divorced', 'Widowed', 'Partnered']
INSURANCE_OPTIONS = ['Private', 'Medicare', 'Medicaid', 'Uninsured', 'Dual-eligible']
BLOOD_TYPE_OPTIONS = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
BLOOD_TYPE_WEIGHTS = [0.374, 0.066, 0.316, 0.063, 0.085, 0.015, 0.034, 0.006]
EPILEPSY_TYPES = ['Focal', 'Generalized', 'Combined', 'Unknown']
REFERRAL_SOURCES = ['Primary care', 'ER', 'Self-referral', 'Neurologist', 'Other specialist']

US_CITIES = [
    ('New York', 'NY', '10001'), ('Los Angeles', 'CA', '90001'),
    ('Chicago', 'IL', '60601'), ('Houston', 'TX', '77001'),
    ('Phoenix', 'AZ', '85001'), ('Philadelphia', 'PA', '19101'),
    ('San Antonio', 'TX', '78201'), ('San Diego', 'CA', '92101'),
    ('Dallas', 'TX', '75201'), ('San Jose', 'CA', '95101'),
    ('Austin', 'TX', '73301'), ('Jacksonville', 'FL', '32099'),
    ('Columbus', 'OH', '43085'), ('Charlotte', 'NC', '28201'),
    ('Indianapolis', 'IN', '46201'), ('Denver', 'CO', '80201'),
    ('Seattle', 'WA', '98101'), ('Boston', 'MA', '02101'),
    ('Nashville', 'TN', '37201'), ('Portland', 'OR', '97201'),
    ('Atlanta', 'GA', '30301'), ('Miami', 'FL', '33101'),
    ('Minneapolis', 'MN', '55401'), ('Cleveland', 'OH', '44101'),
    ('Detroit', 'MI', '48201'), ('St. Louis', 'MO', '63101'),
    ('Pittsburgh', 'PA', '15201'), ('Baltimore', 'MD', '21201'),
    ('Tampa', 'FL', '33601'), ('Sacramento', 'CA', '95801'),
]

NEUROLOGIST_NAMES = [
    'Dr. Sarah Chen', 'Dr. James Wilson', 'Dr. Maria Rodriguez',
    'Dr. Robert Kim', 'Dr. Emily Thompson', 'Dr. David Patel',
    'Dr. Lisa Chang', 'Dr. Michael Brown',
]

FIRST_NAMES_M = [
    'James', 'Robert', 'John', 'Michael', 'David', 'William', 'Richard',
    'Joseph', 'Thomas', 'Charles', 'Daniel', 'Matthew', 'Anthony', 'Mark',
    'Steven',
]
FIRST_NAMES_F = [
    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Barbara', 'Elizabeth',
    'Susan', 'Jessica', 'Sarah', 'Karen', 'Lisa', 'Nancy', 'Betty',
    'Margaret', 'Sandra',
]
LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
    'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
    'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
    'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
    'Ramirez', 'Lewis', 'Robinson',
]

EMERGENCY_CONTACTS = [
    'spouse', 'parent', 'sibling', 'child', 'friend',
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


def _median(values):
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return round((s[n // 2 - 1] + s[n // 2]) / 2, 1)


def _bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


def _generate_demographic(patient_id, patient_info, rng, idx):
    """Generate a single patient demographic record."""
    sex = rng.choices(SEX_OPTIONS, weights=[0.48, 0.48, 0.04])[0]

    if sex == 'Male':
        first_name = rng.choice(FIRST_NAMES_M)
        gender_identity = rng.choices(
            ['Man', 'Non-binary', 'Transgender woman', 'Prefer not to say'],
            weights=[0.93, 0.03, 0.02, 0.02]
        )[0]
    elif sex == 'Female':
        first_name = rng.choice(FIRST_NAMES_F)
        gender_identity = rng.choices(
            ['Woman', 'Non-binary', 'Transgender man', 'Prefer not to say'],
            weights=[0.93, 0.03, 0.02, 0.02]
        )[0]
    else:
        first_name = rng.choice(FIRST_NAMES_M + FIRST_NAMES_F)
        gender_identity = rng.choices(
            ['Non-binary', 'Prefer not to say', 'Man', 'Woman'],
            weights=[0.50, 0.20, 0.15, 0.15]
        )[0]

    last_name = LAST_NAMES[idx % len(LAST_NAMES)]
    full_name = f'{first_name} {last_name}'

    age = rng.choices(
        list(range(18, 79)),
        weights=[max(1, 10 - abs(x - 42) / 3) for x in range(18, 79)]
    )[0]
    birth_year = 2025 - age
    dob = f'{birth_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}'

    # Height and weight with sex-based distributions
    if sex == 'Male':
        height_cm = round(rng.gauss(175, 7), 1)
        weight_kg = round(rng.gauss(85, 15), 1)
    elif sex == 'Female':
        height_cm = round(rng.gauss(163, 6), 1)
        weight_kg = round(rng.gauss(72, 14), 1)
    else:
        height_cm = round(rng.gauss(170, 8), 1)
        weight_kg = round(rng.gauss(78, 15), 1)

    height_cm = max(150, min(200, height_cm))
    weight_kg = max(45, min(150, weight_kg))
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

    blood_type = rng.choices(BLOOD_TYPE_OPTIONS, weights=BLOOD_TYPE_WEIGHTS)[0]
    ethnicity = rng.choices(ETHNICITY_OPTIONS, weights=[0.18, 0.75, 0.07])[0]
    race = rng.choices(RACE_OPTIONS, weights=[0.58, 0.14, 0.06, 0.02, 0.01, 0.04])[0]

    primary_language = rng.choices(LANGUAGE_OPTIONS, weights=LANGUAGE_WEIGHTS)[0]
    interpreter_needed = 1 if primary_language not in ('English',) else 0

    education_level = rng.choices(
        EDUCATION_OPTIONS,
        weights=[0.25, 0.20, 0.25, 0.12, 0.03, 0.08, 0.07]
    )[0]
    occupation_pool = [
        'Teacher', 'Engineer', 'Nurse', 'Retail worker', 'Office manager',
        'Truck driver', 'Student', 'Homemaker', 'Retired', 'Accountant',
        'Social worker', 'Electrician', 'Chef', 'Pharmacist', 'Librarian',
        'Construction worker', 'Software developer', 'Mechanic', 'Artist',
        'Sales associate',
    ]
    occupation = rng.choice(occupation_pool)
    employment_status = rng.choices(
        EMPLOYMENT_OPTIONS,
        weights=[0.35, 0.15, 0.15, 0.15, 0.10, 0.10]
    )[0]
    marital_status = rng.choices(
        MARITAL_OPTIONS,
        weights=[0.30, 0.35, 0.15, 0.05, 0.15]
    )[0]
    insurance_type = rng.choices(
        INSURANCE_OPTIONS,
        weights=[0.40, 0.20, 0.20, 0.10, 0.10]
    )[0]

    # Emergency contact
    ec_relation = rng.choice(EMERGENCY_CONTACTS)
    ec_first = rng.choice(FIRST_NAMES_M + FIRST_NAMES_F)
    ec_last = rng.choice(LAST_NAMES)
    emergency_contact_name = f'{ec_first} {ec_last} ({ec_relation})'
    emergency_contact_phone = f'({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}'

    city, state, zipcode = US_CITIES[idx % len(US_CITIES)]

    referral_source = rng.choices(
        REFERRAL_SOURCES,
        weights=[0.35, 0.20, 0.15, 0.20, 0.10]
    )[0]

    epilepsy_type = rng.choices(EPILEPSY_TYPES, weights=[0.45, 0.30, 0.15, 0.10])[0]
    epilepsy_onset_age = rng.randint(1, min(65, age))
    years_with_epilepsy = age - epilepsy_onset_age

    primary_neurologist = rng.choice(NEUROLOGIST_NAMES)

    # Enrollment date within past 2 years
    enrollment_offset = rng.randint(0, 730)
    enrollment_date = (datetime(2025, 6, 15) - timedelta(days=enrollment_offset)).strftime('%Y-%m-%d')

    return {
        'patient_id': patient_id,
        'full_name': full_name,
        'date_of_birth': dob,
        'age': age,
        'sex': sex,
        'gender_identity': gender_identity,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'bmi': bmi,
        'blood_type': blood_type,
        'ethnicity': ethnicity,
        'race': race,
        'primary_language': primary_language,
        'interpreter_needed': interpreter_needed,
        'education_level': education_level,
        'occupation': occupation,
        'employment_status': employment_status,
        'marital_status': marital_status,
        'insurance_type': insurance_type,
        'emergency_contact_name': emergency_contact_name,
        'emergency_contact_phone': emergency_contact_phone,
        'address_city': city,
        'address_state': state,
        'address_zip': zipcode,
        'referral_source': referral_source,
        'epilepsy_type': epilepsy_type,
        'epilepsy_onset_age': epilepsy_onset_age,
        'years_with_epilepsy': years_with_epilepsy,
        'primary_neurologist': primary_neurologist,
        'enrollment_date': enrollment_date,
    }


def _populate_if_empty():
    """Populate patient_demographics table with realistic data if empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS patient_demographics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            full_name TEXT,
            date_of_birth TEXT,
            age INTEGER,
            sex TEXT,
            gender_identity TEXT,
            height_cm REAL,
            weight_kg REAL,
            bmi REAL,
            blood_type TEXT,
            ethnicity TEXT,
            race TEXT,
            primary_language TEXT,
            interpreter_needed INTEGER,
            education_level TEXT,
            occupation TEXT,
            employment_status TEXT,
            marital_status TEXT,
            insurance_type TEXT,
            emergency_contact_name TEXT,
            emergency_contact_phone TEXT,
            address_city TEXT,
            address_state TEXT,
            address_zip TEXT,
            referral_source TEXT,
            epilepsy_type TEXT,
            epilepsy_onset_age INTEGER,
            years_with_epilepsy INTEGER,
            primary_neurologist TEXT,
            enrollment_date TEXT,
            created_at TEXT
        )''')
        conn.commit()

        count = conn.execute('SELECT COUNT(*) FROM patient_demographics').fetchone()[0]
        if count > 0:
            return  # already populated

        # Get real patient IDs
        patients = conn.execute(
            'SELECT patient_id, name, age, gender FROM patients'
        ).fetchall()
        patients = [dict(p) for p in patients]

        epat = [p for p in patients if p['patient_id'].startswith('EPAT')]
        others = [p for p in patients if not p['patient_id'].startswith('EPAT')]
        ordered = epat + others
        target_patients = ordered[:30]
        patient_ids = [p['patient_id'] for p in target_patients]

        rng = random.Random(99)

        for idx, pid in enumerate(patient_ids):
            patient_info = target_patients[idx]
            record = _generate_demographic(pid, patient_info, rng, idx)

            conn.execute(
                '''INSERT INTO patient_demographics (
                    patient_id, full_name, date_of_birth, age, sex, gender_identity,
                    height_cm, weight_kg, bmi, blood_type, ethnicity, race,
                    primary_language, interpreter_needed, education_level, occupation,
                    employment_status, marital_status, insurance_type,
                    emergency_contact_name, emergency_contact_phone,
                    address_city, address_state, address_zip,
                    referral_source, epilepsy_type, epilepsy_onset_age,
                    years_with_epilepsy, primary_neurologist, enrollment_date,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                (
                    record['patient_id'], record['full_name'], record['date_of_birth'],
                    record['age'], record['sex'], record['gender_identity'],
                    record['height_cm'], record['weight_kg'], record['bmi'],
                    record['blood_type'], record['ethnicity'], record['race'],
                    record['primary_language'], record['interpreter_needed'],
                    record['education_level'], record['occupation'],
                    record['employment_status'], record['marital_status'],
                    record['insurance_type'], record['emergency_contact_name'],
                    record['emergency_contact_phone'], record['address_city'],
                    record['address_state'], record['address_zip'],
                    record['referral_source'], record['epilepsy_type'],
                    record['epilepsy_onset_age'], record['years_with_epilepsy'],
                    record['primary_neurologist'], record['enrollment_date'],
                )
            )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def _load_all_data():
    """Load all patient_demographics and return structured list."""
    _populate_if_empty()

    rows = _db_query(
        '''SELECT patient_id, full_name, date_of_birth, age, sex, gender_identity,
                  height_cm, weight_kg, bmi, blood_type, ethnicity, race,
                  primary_language, interpreter_needed, education_level, occupation,
                  employment_status, marital_status, insurance_type,
                  emergency_contact_name, emergency_contact_phone,
                  address_city, address_state, address_zip,
                  referral_source, epilepsy_type, epilepsy_onset_age,
                  years_with_epilepsy, primary_neurologist, enrollment_date
           FROM patient_demographics'''
    )
    return rows


def overview():
    """Return KPI cards + chart data for the Demographics overview tab."""
    data = _load_all_data()

    total_patients = len(data)
    if total_patients == 0:
        return {'available': False, 'total_patients': 0}

    # --- KPIs ---
    ages = [d['age'] for d in data]
    avg_age = _avg(ages)

    sex_counts = Counter(d['sex'] for d in data)
    male_pct = round(sex_counts.get('Male', 0) / total_patients * 100, 1)
    female_pct = round(sex_counts.get('Female', 0) / total_patients * 100, 1)

    bmis = [d['bmi'] for d in data if d['bmi']]
    avg_bmi = _avg(bmis)

    interpreter_count = sum(1 for d in data if d.get('interpreter_needed'))
    interpreter_needed_pct = round(interpreter_count / total_patients * 100, 1)

    years_epi = [d['years_with_epilepsy'] for d in data if d.get('years_with_epilepsy') is not None]
    avg_years_with_epilepsy = _avg(years_epi)

    epilepsy_counts = Counter(d['epilepsy_type'] for d in data)
    most_common_epilepsy_type = epilepsy_counts.most_common(1)[0][0] if epilepsy_counts else 'Unknown'

    # --- Charts ---

    # Age distribution (buckets)
    age_buckets = {'18-30': 0, '31-45': 0, '46-60': 0, '61+': 0}
    for a in ages:
        if a <= 30:
            age_buckets['18-30'] += 1
        elif a <= 45:
            age_buckets['31-45'] += 1
        elif a <= 60:
            age_buckets['46-60'] += 1
        else:
            age_buckets['61+'] += 1
    age_distribution = [{'bucket': k, 'count': v} for k, v in age_buckets.items()]

    # Sex distribution (pie)
    sex_distribution = [
        {'sex': s, 'count': sex_counts.get(s, 0)}
        for s in SEX_OPTIONS
    ]

    # Ethnicity distribution (bar)
    eth_counts = Counter(d['ethnicity'] for d in data)
    ethnicity_distribution = [
        {'ethnicity': e, 'count': eth_counts.get(e, 0)}
        for e in ETHNICITY_OPTIONS
    ]

    # Race distribution (bar)
    race_counts = Counter(d['race'] for d in data)
    race_distribution = [
        {'race': r, 'count': race_counts.get(r, 0)}
        for r in RACE_OPTIONS
    ]

    # Language distribution (pie)
    lang_counts = Counter(d['primary_language'] for d in data)
    language_distribution = [
        {'language': l, 'count': lang_counts.get(l, 0)}
        for l in LANGUAGE_OPTIONS
    ]

    # Education distribution (bar)
    edu_counts = Counter(d['education_level'] for d in data)
    education_distribution = [
        {'education': e, 'count': edu_counts.get(e, 0)}
        for e in EDUCATION_OPTIONS
    ]

    # Employment distribution (bar)
    emp_counts = Counter(d['employment_status'] for d in data)
    employment_distribution = [
        {'employment': e, 'count': emp_counts.get(e, 0)}
        for e in EMPLOYMENT_OPTIONS
    ]

    # Insurance distribution (pie)
    ins_counts = Counter(d['insurance_type'] for d in data)
    insurance_distribution = [
        {'insurance': i, 'count': ins_counts.get(i, 0)}
        for i in INSURANCE_OPTIONS
    ]

    # Epilepsy type distribution (bar)
    epilepsy_type_distribution = [
        {'epilepsy_type': e, 'count': epilepsy_counts.get(e, 0)}
        for e in EPILEPSY_TYPES
    ]

    # BMI categories
    bmi_cats = Counter(_bmi_category(d['bmi']) for d in data if d['bmi'])
    bmi_categories = [
        {'category': c, 'count': bmi_cats.get(c, 0)}
        for c in ['Underweight', 'Normal', 'Overweight', 'Obese']
    ]

    return {
        'available': True,
        'total_patients': total_patients,
        'avg_age': avg_age,
        'male_pct': male_pct,
        'female_pct': female_pct,
        'avg_bmi': avg_bmi,
        'interpreter_needed_pct': interpreter_needed_pct,
        'avg_years_with_epilepsy': avg_years_with_epilepsy,
        'most_common_epilepsy_type': most_common_epilepsy_type,
        'age_distribution': age_distribution,
        'sex_distribution': sex_distribution,
        'ethnicity_distribution': ethnicity_distribution,
        'race_distribution': race_distribution,
        'language_distribution': language_distribution,
        'education_distribution': education_distribution,
        'employment_distribution': employment_distribution,
        'insurance_distribution': insurance_distribution,
        'epilepsy_type_distribution': epilepsy_type_distribution,
        'bmi_categories': bmi_categories,
    }


def breakdown():
    """Return per-patient demographics, age stats, epilepsy onset stats, and referral sources."""
    data = _load_all_data()

    # Full patient list
    patients = sorted(data, key=lambda d: d.get('patient_id', ''))

    # Age stats
    ages = [d['age'] for d in data]
    age_stats = {
        'min': min(ages) if ages else 0,
        'max': max(ages) if ages else 0,
        'mean': _avg(ages),
        'median': _median(ages),
    }

    # Epilepsy onset stats
    onset_ages = [d['epilepsy_onset_age'] for d in data if d.get('epilepsy_onset_age') is not None]
    epilepsy_onset_stats = {
        'min': min(onset_ages) if onset_ages else 0,
        'max': max(onset_ages) if onset_ages else 0,
        'mean': _avg(onset_ages),
        'median': _median(onset_ages),
    }

    # Referral sources
    ref_counts = Counter(d['referral_source'] for d in data)
    referral_sources = [
        {'source': src, 'count': cnt}
        for src, cnt in ref_counts.most_common()
    ]

    return {
        'patients': patients,
        'age_stats': age_stats,
        'epilepsy_onset_stats': epilepsy_onset_stats,
        'referral_sources': referral_sources,
    }


def definitions():
    """Return demographic and clinical terminology definitions."""
    return {
        'concepts': [
            {
                'name': 'Demographics',
                'description': 'The statistical characteristics of a patient population, including age, sex, race, ethnicity, language, education level, employment status, marital status, and insurance coverage. Demographic data collection follows OMB (Office of Management and Budget) standards for race and ethnicity categories, and is essential for ensuring equitable care delivery, identifying health disparities, and meeting regulatory reporting requirements (CMS, Joint Commission). In epilepsy care, demographics inform treatment decisions — for example, age and sex influence AED selection due to teratogenicity concerns, bone density effects, and hormonal interactions.',
            },
            {
                'name': 'BMI Categories',
                'description': 'Body Mass Index (BMI) is calculated as weight (kg) divided by height (m) squared. WHO classification: Underweight (BMI < 18.5) — may indicate malnutrition or medication side effects (e.g., topiramate-induced weight loss); Normal (18.5-24.9) — healthy weight range; Overweight (25.0-29.9) — increased metabolic risk; Obese (30.0+) — associated with increased seizure frequency, sleep apnea (a seizure trigger), and altered AED pharmacokinetics (volume of distribution changes). Valproate and pregabalin are associated with weight gain, while topiramate and zonisamide may cause weight loss. BMI monitoring is standard of care in epilepsy management.',
            },
            {
                'name': 'Epilepsy Classification (ILAE)',
                'description': 'The International League Against Epilepsy (ILAE) 2017 classification framework defines epilepsy types as: Focal — seizures originate in networks limited to one hemisphere, may be aware or impaired awareness, motor or non-motor onset (most common type, ~60% of cases); Generalized — seizures originate in bilaterally distributed networks, includes absence, myoclonic, tonic, clonic, tonic-clonic, and atonic subtypes; Combined Generalized and Focal — patients exhibiting both focal and generalized seizure types (e.g., Dravet syndrome, Lennox-Gastaut syndrome); Unknown — insufficient information to classify. Classification guides AED selection: focal epilepsy responds to sodium channel blockers (carbamazepine, lacosamide), while generalized epilepsy requires broad-spectrum AEDs (valproate, levetiracetam, lamotrigine).',
            },
            {
                'name': 'Blood Types',
                'description': 'ABO blood group and Rh factor classification. Distribution in the US population: O+ (37.4%), A+ (31.6%), B+ (8.5%), O- (6.6%), A- (6.3%), AB+ (3.4%), B- (1.5%), AB- (0.6%). Blood type documentation is critical in epilepsy care for: surgical planning (resective epilepsy surgery, VNS implantation), emergency management of status epilepticus (may require transfusion if prolonged), and SUDEP autopsy protocols. Some research suggests associations between ABO blood type and cerebrovascular disease risk, which may intersect with post-stroke epilepsy.',
            },
            {
                'name': 'Interpreter Services',
                'description': 'Language access services required under Title VI of the Civil Rights Act and Section 1557 of the ACA for patients with limited English proficiency (LEP). In epilepsy care, interpreter services are critical for: accurate seizure semiology description (aura characteristics, post-ictal symptoms), medication counseling (complex AED titration schedules, drug interaction warnings), informed consent for procedures (EEG, epilepsy surgery evaluation), and emergency seizure action plan communication. Professional medical interpreters are preferred over family members to ensure accuracy of medical terminology and patient confidentiality. Approximately 8.2% of the US population has LEP status.',
            },
            {
                'name': 'Social Determinants of Health (SDOH)',
                'description': 'Non-medical factors that influence health outcomes, encompassing five domains (Healthy People 2030): Economic Stability (employment, income, insurance coverage, food security); Education Access and Quality (literacy, educational attainment, language); Healthcare Access and Quality (insurance type, provider availability, transportation); Neighborhood and Built Environment (housing quality, environmental exposures, crime/safety); Social and Community Context (social support, discrimination, incarceration history). In epilepsy, SDOH strongly predict outcomes: uninsured patients have 3x higher ER utilization for seizures; education level correlates with medication adherence; employment status affects access to consistent AED supply; and neighborhood factors influence exposure to seizure triggers (environmental pollution, sleep disruption). The demographics dashboard captures key SDOH indicators to enable disparity analysis and targeted interventions.',
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
        pprint.pprint(p)
    print(f'\nTotal patients: {len(bd["patients"])}')
    print(f'Age stats: {bd["age_stats"]}')
    print(f'Epilepsy onset stats: {bd["epilepsy_onset_stats"]}')
    print(f'Referral sources: {bd["referral_sources"]}')
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Concepts: {len(df["concepts"])}')
    for c in df['concepts']:
        print(f'  {c["name"]}')
