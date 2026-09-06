"""Seed the pnes_screening table with realistic differential data."""
import os
import random
import sqlite3

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

random.seed(42)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS pnes_screening (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    screening_date TEXT NOT NULL,
    referral_reason TEXT,
    eye_closure_score INTEGER DEFAULT 0,
    pelvic_thrusting_score INTEGER DEFAULT 0,
    side_to_side_head_score INTEGER DEFAULT 0,
    ictal_crying_score INTEGER DEFAULT 0,
    memory_recall_score INTEGER DEFAULT 0,
    gradual_onset_score INTEGER DEFAULT 0,
    duration_gt_2min INTEGER DEFAULT 0,
    eeg_ictal_normal INTEGER DEFAULT 0,
    eeg_interictal_normal INTEGER DEFAULT 0,
    trauma_history INTEGER DEFAULT 0,
    conversion_features INTEGER DEFAULT 0,
    psychiatric_comorbidity TEXT,
    pnes_probability REAL,
    epilepsy_probability REAL,
    classification TEXT,
    confidence REAL,
    video_eeg_recommended INTEGER DEFAULT 0,
    psychiatry_referral INTEGER DEFAULT 0,
    neuropsych_testing INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    reviewer_notes TEXT,
    reviewed_by TEXT,
    reviewed_date TEXT
);
"""

REFERRAL_REASONS = [
    'new onset events', 'treatment refractory', 'atypical semiology',
    'dual diagnosis concern', 'video-EEG referral'
]

COMORBIDITIES = [
    'anxiety', 'depression', 'PTSD', 'conversion disorder',
    'dissociative disorder', None
]

REVIEWERS = ['Dr. Patel', 'Dr. Singh', 'Dr. Kapoor', 'Dr. Sharma']

REVIEWER_NOTES_PNES = [
    'Semiology strongly suggestive of PNES. Recommend psychiatry follow-up.',
    'Normal ictal EEG with eye closure and asynchronous movements — PNES likely.',
    'Trauma history and conversion features support psychogenic origin.',
    'Video-EEG confirmed non-epileptic events. Psychiatric referral placed.',
]

REVIEWER_NOTES_EPILEPSY = [
    'Clear epileptiform discharges on EEG. Continue ASM.',
    'Ictal EEG abnormal — consistent with focal epilepsy.',
    'Interictal spikes present. No semiological PNES features.',
    'EEG and semiology consistent with generalized epilepsy.',
]

REVIEWER_NOTES_MIXED = [
    'Mixed picture — some events appear epileptic, others psychogenic.',
    'Concurrent PNES and epilepsy suspected. Video-EEG monitoring advised.',
]

REVIEWER_NOTES_INDET = [
    'Insufficient data for classification. Recommend prolonged video-EEG.',
    'Semiology inconclusive. Additional monitoring scheduled.',
]


def _random_date(start_year=2025, start_month=1, end_year=2026, end_month=6):
    y = random.randint(start_year, end_year)
    if y == start_year:
        m = random.randint(start_month, 12)
    elif y == end_year:
        m = random.randint(1, end_month)
    else:
        m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"


def _make_row(patient_id, classification):
    """Generate a single PNES screening row with realistic scores."""
    row = {'patient_id': patient_id, 'screening_date': _random_date()}
    row['referral_reason'] = random.choice(REFERRAL_REASONS)

    if classification == 'pnes_likely':
        # High semiological PNES scores, normal EEG, trauma history
        row['eye_closure_score'] = random.choices([2, 3], weights=[30, 70])[0]
        row['pelvic_thrusting_score'] = random.choices([1, 2, 3], weights=[20, 40, 40])[0]
        row['side_to_side_head_score'] = random.choices([1, 2, 3], weights=[20, 40, 40])[0]
        row['ictal_crying_score'] = random.choices([1, 2, 3], weights=[30, 40, 30])[0]
        row['memory_recall_score'] = random.choices([0, 1, 2], weights=[50, 30, 20])[0]
        row['gradual_onset_score'] = random.choices([2, 3], weights=[40, 60])[0]
        row['duration_gt_2min'] = random.choices([1, 0], weights=[80, 20])[0]
        row['eeg_ictal_normal'] = random.choices([1, 0], weights=[85, 15])[0]
        row['eeg_interictal_normal'] = random.choices([1, 0], weights=[70, 30])[0]
        row['trauma_history'] = random.choices([1, 0], weights=[75, 25])[0]
        row['conversion_features'] = random.choices([1, 0], weights=[60, 40])[0]
        row['psychiatric_comorbidity'] = random.choice(['PTSD', 'conversion disorder',
                                                         'dissociative disorder', 'anxiety', 'depression'])
        pnes_p = round(random.uniform(0.65, 0.95), 2)
        row['pnes_probability'] = pnes_p
        row['epilepsy_probability'] = round(1.0 - pnes_p + random.uniform(-0.05, 0.05), 2)
        row['confidence'] = round(random.uniform(0.70, 0.95), 2)
        row['video_eeg_recommended'] = random.choices([1, 0], weights=[70, 30])[0]
        row['psychiatry_referral'] = 1
        row['neuropsych_testing'] = random.choices([1, 0], weights=[50, 50])[0]

    elif classification == 'epilepsy_likely':
        # Low semiological PNES scores, abnormal EEG
        row['eye_closure_score'] = random.choices([0, 1], weights=[70, 30])[0]
        row['pelvic_thrusting_score'] = random.choices([0, 1], weights=[80, 20])[0]
        row['side_to_side_head_score'] = random.choices([0, 1], weights=[75, 25])[0]
        row['ictal_crying_score'] = random.choices([0, 1], weights=[85, 15])[0]
        row['memory_recall_score'] = random.choices([1, 2, 3], weights=[30, 40, 30])[0]
        row['gradual_onset_score'] = random.choices([0, 1], weights=[70, 30])[0]
        row['duration_gt_2min'] = random.choices([0, 1], weights=[70, 30])[0]
        row['eeg_ictal_normal'] = 0
        row['eeg_interictal_normal'] = random.choices([0, 1], weights=[80, 20])[0]
        row['trauma_history'] = random.choices([0, 1], weights=[85, 15])[0]
        row['conversion_features'] = 0
        row['psychiatric_comorbidity'] = random.choice(['anxiety', 'depression', None, None])
        pnes_p = round(random.uniform(0.05, 0.30), 2)
        row['pnes_probability'] = pnes_p
        row['epilepsy_probability'] = round(1.0 - pnes_p + random.uniform(-0.05, 0.05), 2)
        row['confidence'] = round(random.uniform(0.75, 0.95), 2)
        row['video_eeg_recommended'] = random.choices([0, 1], weights=[60, 40])[0]
        row['psychiatry_referral'] = 0
        row['neuropsych_testing'] = random.choices([0, 1], weights=[70, 30])[0]

    elif classification == 'mixed':
        # Moderate scores, some PNES features + some EEG abnormalities
        row['eye_closure_score'] = random.choices([1, 2], weights=[50, 50])[0]
        row['pelvic_thrusting_score'] = random.choices([0, 1, 2], weights=[30, 40, 30])[0]
        row['side_to_side_head_score'] = random.choices([0, 1, 2], weights=[30, 40, 30])[0]
        row['ictal_crying_score'] = random.choices([0, 1, 2], weights=[40, 40, 20])[0]
        row['memory_recall_score'] = random.choices([1, 2], weights=[50, 50])[0]
        row['gradual_onset_score'] = random.choices([1, 2], weights=[50, 50])[0]
        row['duration_gt_2min'] = random.choices([0, 1], weights=[50, 50])[0]
        row['eeg_ictal_normal'] = random.choices([0, 1], weights=[50, 50])[0]
        row['eeg_interictal_normal'] = random.choices([0, 1], weights=[50, 50])[0]
        row['trauma_history'] = random.choices([0, 1], weights=[50, 50])[0]
        row['conversion_features'] = random.choices([0, 1], weights=[60, 40])[0]
        row['psychiatric_comorbidity'] = random.choice(['anxiety', 'depression', 'PTSD', None])
        pnes_p = round(random.uniform(0.35, 0.60), 2)
        row['pnes_probability'] = pnes_p
        row['epilepsy_probability'] = round(1.0 - pnes_p + random.uniform(-0.10, 0.10), 2)
        row['confidence'] = round(random.uniform(0.40, 0.65), 2)
        row['video_eeg_recommended'] = 1
        row['psychiatry_referral'] = random.choices([1, 0], weights=[60, 40])[0]
        row['neuropsych_testing'] = random.choices([1, 0], weights=[50, 50])[0]

    else:  # indeterminate
        row['eye_closure_score'] = random.randint(0, 2)
        row['pelvic_thrusting_score'] = random.randint(0, 1)
        row['side_to_side_head_score'] = random.randint(0, 1)
        row['ictal_crying_score'] = random.randint(0, 1)
        row['memory_recall_score'] = random.randint(0, 2)
        row['gradual_onset_score'] = random.randint(0, 2)
        row['duration_gt_2min'] = random.randint(0, 1)
        row['eeg_ictal_normal'] = random.randint(0, 1)
        row['eeg_interictal_normal'] = random.randint(0, 1)
        row['trauma_history'] = random.randint(0, 1)
        row['conversion_features'] = random.randint(0, 1)
        row['psychiatric_comorbidity'] = random.choice(COMORBIDITIES)
        pnes_p = round(random.uniform(0.30, 0.55), 2)
        row['pnes_probability'] = pnes_p
        row['epilepsy_probability'] = round(random.uniform(0.30, 0.55), 2)
        row['confidence'] = round(random.uniform(0.25, 0.50), 2)
        row['video_eeg_recommended'] = 1
        row['psychiatry_referral'] = random.choices([0, 1], weights=[50, 50])[0]
        row['neuropsych_testing'] = 1

    row['classification'] = classification

    # Review status: ~60% reviewed
    if random.random() < 0.60:
        row['status'] = random.choice(['reviewed', 'confirmed_pnes', 'confirmed_epilepsy'])
        if classification == 'pnes_likely':
            row['status'] = random.choice(['reviewed', 'confirmed_pnes'])
            row['reviewer_notes'] = random.choice(REVIEWER_NOTES_PNES)
        elif classification == 'epilepsy_likely':
            row['status'] = random.choice(['reviewed', 'confirmed_epilepsy'])
            row['reviewer_notes'] = random.choice(REVIEWER_NOTES_EPILEPSY)
        elif classification == 'mixed':
            row['status'] = 'reviewed'
            row['reviewer_notes'] = random.choice(REVIEWER_NOTES_MIXED)
        else:
            row['status'] = 'reviewed'
            row['reviewer_notes'] = random.choice(REVIEWER_NOTES_INDET)
        row['reviewed_by'] = random.choice(REVIEWERS)
        row['reviewed_date'] = _random_date(2025, 3, 2026, 6)
    else:
        row['status'] = 'pending'
        row['reviewer_notes'] = None
        row['reviewed_by'] = None
        row['reviewed_date'] = None

    # Clamp epilepsy_probability
    row['epilepsy_probability'] = max(0.0, min(1.0, row['epilepsy_probability']))

    return row


def seed():
    conn = sqlite3.connect(DB)
    conn.execute(CREATE_SQL)
    # Clear existing data
    conn.execute("DELETE FROM pnes_screening")

    patients = [f"EPAT{i:03d}" for i in range(1, 31)]

    # Distribution: ~30% pnes, ~50% epilepsy, ~12% mixed, ~8% indeterminate
    # We'll generate 93 rows total
    classifications = (
        ['pnes_likely'] * 28 +
        ['epilepsy_likely'] * 46 +
        ['mixed'] * 12 +
        ['indeterminate'] * 7
    )
    random.shuffle(classifications)

    rows = []
    for i, cls in enumerate(classifications):
        pid = random.choice(patients)
        rows.append(_make_row(pid, cls))

    cols = [
        'patient_id', 'screening_date', 'referral_reason',
        'eye_closure_score', 'pelvic_thrusting_score', 'side_to_side_head_score',
        'ictal_crying_score', 'memory_recall_score', 'gradual_onset_score',
        'duration_gt_2min', 'eeg_ictal_normal', 'eeg_interictal_normal',
        'trauma_history', 'conversion_features', 'psychiatric_comorbidity',
        'pnes_probability', 'epilepsy_probability', 'classification', 'confidence',
        'video_eeg_recommended', 'psychiatry_referral', 'neuropsych_testing',
        'status', 'reviewer_notes', 'reviewed_by', 'reviewed_date'
    ]

    placeholders = ', '.join(['?'] * len(cols))
    col_names = ', '.join(cols)

    for r in rows:
        vals = [r[c] for c in cols]
        conn.execute(f"INSERT INTO pnes_screening ({col_names}) VALUES ({placeholders})", vals)

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM pnes_screening").fetchone()[0]
    conn.close()
    print(f"Seeded {count} rows into pnes_screening")
    return count


if __name__ == '__main__':
    seed()
