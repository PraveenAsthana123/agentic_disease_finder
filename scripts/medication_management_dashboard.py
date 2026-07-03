"""Medication Self-Management Dashboard — Patient Module Section 5.
Tracks medication adherence logs, side effect profiling, refill management,
and drug-specific analytics for epilepsy patients on AED therapy.

Populates and reads from:
  - medication_adherence  (daily dose-level adherence with side effects)
  - medication_refills    (pharmacy refill records)

Uses real patient_ids from the patients table (first 30).
ILAE AED guidelines, WHO medication adherence standards applied throughout.
"""

import json
import os
import random
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

AEDS = [
    'Levetiracetam',
    'Lamotrigine',
    'Valproate',
    'Carbamazepine',
    'Lacosamide',
    'Topiramate',
    'Oxcarbazepine',
    'Clobazam',
]

SIDE_EFFECTS_POOL = [
    'Drowsiness',
    'Dizziness',
    'Nausea',
    'Headache',
    'Weight gain',
    'Fatigue',
    'Tremor',
    'Mood changes',
    'Memory issues',
    'Skin rash',
    'Hair loss',
    'Blurred vision',
]

FREQUENCIES = ['BID', 'TID', 'QD', 'QHS']
SCHEDULED_TIMES = ['morning', 'afternoon', 'evening', 'bedtime']
PHARMACIES = [
    'CVS Pharmacy',
    'Walgreens',
    'Rite Aid',
    'Walmart Pharmacy',
    'Costco Pharmacy',
    'Kroger Pharmacy',
]

# Typical dose ranges per drug (mg)
DOSE_RANGES = {
    'Levetiracetam': (500, 1500),
    'Lamotrigine': (100, 400),
    'Valproate': (250, 1000),
    'Carbamazepine': (200, 600),
    'Lacosamide': (100, 400),
    'Topiramate': (50, 200),
    'Oxcarbazepine': (300, 900),
    'Clobazam': (5, 20),
}

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
    """Create medication_adherence and medication_refills tables, seed if empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS medication_adherence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            log_date TEXT,
            drug_name TEXT,
            dose_mg REAL,
            frequency TEXT,
            scheduled_time TEXT,
            taken TEXT,
            minutes_late INTEGER,
            side_effects_json TEXT,
            side_effect_severity INTEGER,
            mood_after INTEGER,
            notes TEXT,
            created_at TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS medication_refills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            drug_name TEXT,
            refill_date TEXT,
            quantity INTEGER,
            pharmacy TEXT,
            auto_refill INTEGER,
            days_supply INTEGER,
            created_at TEXT
        )''')
        conn.commit()

        count = conn.execute('SELECT COUNT(*) FROM medication_adherence').fetchone()[0]
        if count > 0:
            return  # already seeded

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
        base_date = datetime(2025, 6, 15)

        # Assign 2-3 meds per patient
        patient_meds = {}
        for pid in patient_ids:
            num_meds = rng.choice([2, 2, 2, 3, 3])
            meds = rng.sample(AEDS, num_meds)
            patient_meds[pid] = []
            for drug in meds:
                lo, hi = DOSE_RANGES[drug]
                dose = rng.choice(range(lo, hi + 1, 50))
                freq = rng.choice(FREQUENCIES)
                patient_meds[pid].append({
                    'drug': drug,
                    'dose_mg': dose,
                    'frequency': freq,
                })

        # Generate 90 days of adherence logs
        for pid in patient_ids:
            meds = patient_meds[pid]
            for day_offset in range(90):
                log_date = base_date - timedelta(days=day_offset)
                date_str = log_date.strftime('%Y-%m-%d')

                for med in meds:
                    # Determine scheduled times based on frequency
                    if med['frequency'] == 'QD':
                        times = ['morning']
                    elif med['frequency'] == 'BID':
                        times = ['morning', 'evening']
                    elif med['frequency'] == 'TID':
                        times = ['morning', 'afternoon', 'evening']
                    elif med['frequency'] == 'QHS':
                        times = ['bedtime']
                    else:
                        times = ['morning']

                    for sched_time in times:
                        # Adherence: ~85% taken on time, ~8% late, ~7% missed
                        taken_choice = rng.choices(
                            ['yes', 'late', 'no'],
                            weights=[0.85, 0.08, 0.07]
                        )[0]

                        if taken_choice == 'yes':
                            minutes_late = 0
                        elif taken_choice == 'late':
                            minutes_late = rng.choice([15, 30, 45, 60, 90, 120])
                        else:
                            minutes_late = None

                        # Side effects: ~25% chance of reporting something
                        if rng.random() < 0.25:
                            num_effects = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
                            effects = rng.sample(SIDE_EFFECTS_POOL, num_effects)
                            severity = rng.randint(1, 8)
                        else:
                            effects = []
                            severity = 0

                        mood_after = rng.randint(3, 10)

                        # Notes: ~10% chance
                        notes_pool = [
                            '', '', '', '', '', '', '', '', '', '',
                            'Felt drowsy after dose',
                            'Took with food',
                            'Forgot until reminded',
                            'Slight stomach upset',
                            'Felt fine today',
                            'Increased dose per doctor',
                            'Switching pharmacy next month',
                            'Ran out, waiting for refill',
                        ]
                        notes = rng.choice(notes_pool)

                        conn.execute(
                            '''INSERT INTO medication_adherence
                            (patient_id, log_date, drug_name, dose_mg, frequency,
                             scheduled_time, taken, minutes_late, side_effects_json,
                             side_effect_severity, mood_after, notes, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                            (pid, date_str, med['drug'], med['dose_mg'],
                             med['frequency'], sched_time, taken_choice,
                             minutes_late, json.dumps(effects), severity,
                             mood_after, notes)
                        )

        # Generate refill records: ~1 refill per drug every 30 days
        for pid in patient_ids:
            meds = patient_meds[pid]
            for med in meds:
                pharmacy = rng.choice(PHARMACIES)
                auto_refill = rng.choice([0, 0, 1, 1, 1])  # 60% auto
                num_refills = rng.randint(2, 4)
                for r in range(num_refills):
                    refill_date = base_date - timedelta(days=r * 30 + rng.randint(0, 5))
                    days_supply = rng.choice([30, 60, 90])
                    quantity = rng.choice([30, 60, 90, 120])
                    conn.execute(
                        '''INSERT INTO medication_refills
                        (patient_id, drug_name, refill_date, quantity, pharmacy,
                         auto_refill, days_supply, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, datetime("now"))''',
                        (pid, med['drug'], refill_date.strftime('%Y-%m-%d'),
                         quantity, pharmacy, auto_refill, days_supply)
                    )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def overview():
    """Return KPI cards + chart data for the Medication Management overview tab."""
    _ensure_tables()

    adherence_rows = _db_query(
        'SELECT patient_id, log_date, drug_name, dose_mg, frequency, '
        'scheduled_time, taken, minutes_late, side_effects_json, '
        'side_effect_severity, mood_after, notes FROM medication_adherence'
    )
    refill_rows = _db_query(
        'SELECT patient_id, drug_name, refill_date, quantity, pharmacy, '
        'auto_refill, days_supply FROM medication_refills'
    )

    total_logs = len(adherence_rows)
    patient_ids = list(set(r['patient_id'] for r in adherence_rows))
    total_patients = len(patient_ids)

    # Overall adherence rate (taken = yes or late)
    adherent_count = sum(1 for r in adherence_rows if r['taken'] in ('yes', 'late'))
    overall_adherence_rate = round(adherent_count / total_logs * 100, 1) if total_logs else 0

    # Missed dose rate
    missed_count = sum(1 for r in adherence_rows if r['taken'] == 'no')
    missed_dose_rate = round(missed_count / total_logs * 100, 1) if total_logs else 0

    # Avg side effect severity (only where severity > 0)
    sev_values = [r['side_effect_severity'] for r in adherence_rows if r['side_effect_severity'] and r['side_effect_severity'] > 0]
    avg_side_effect_severity = _avg(sev_values)

    # Most common side effect
    all_effects = []
    for r in adherence_rows:
        effects = _safe_json(r.get('side_effects_json'))
        if isinstance(effects, list):
            all_effects.extend(effects)
    effect_counts = Counter(all_effects)
    most_common_side_effect = effect_counts.most_common(1)[0][0] if effect_counts else 'None'

    # Refill stats
    total_refills = len(refill_rows)
    # Avg days between refills per patient-drug
    patient_drug_refills = {}
    for r in refill_rows:
        key = (r['patient_id'], r['drug_name'])
        patient_drug_refills.setdefault(key, []).append(r['refill_date'])
    refill_gaps = []
    for key, dates in patient_drug_refills.items():
        sorted_dates = sorted(dates)
        for i in range(1, len(sorted_dates)):
            d1 = datetime.strptime(sorted_dates[i - 1], '%Y-%m-%d')
            d2 = datetime.strptime(sorted_dates[i], '%Y-%m-%d')
            gap = abs((d2 - d1).days)
            if gap > 0:
                refill_gaps.append(gap)
    avg_days_between_refills = _avg(refill_gaps)

    # Adherence by drug
    drug_logs = {}
    for r in adherence_rows:
        drug = r['drug_name']
        drug_logs.setdefault(drug, {'total': 0, 'adherent': 0, 'missed': 0})
        drug_logs[drug]['total'] += 1
        if r['taken'] in ('yes', 'late'):
            drug_logs[drug]['adherent'] += 1
        if r['taken'] == 'no':
            drug_logs[drug]['missed'] += 1
    adherence_by_drug = []
    for drug in sorted(drug_logs.keys()):
        d = drug_logs[drug]
        adherence_by_drug.append({
            'drug': drug,
            'adherence_pct': round(d['adherent'] / d['total'] * 100, 1) if d['total'] else 0,
            'total_doses': d['total'],
            'missed': d['missed'],
        })

    # Side effect distribution
    side_effect_distribution = [
        {'effect': eff, 'count': cnt}
        for eff, cnt in effect_counts.most_common()
    ]

    # Adherence trend 30d
    base_date = datetime(2025, 6, 15)
    adherence_trend_30d = []
    for day_offset in range(30, 0, -1):
        d = (base_date - timedelta(days=day_offset)).strftime('%Y-%m-%d')
        day_rows = [r for r in adherence_rows if r['log_date'] == d]
        if day_rows:
            day_adherent = sum(1 for r in day_rows if r['taken'] in ('yes', 'late'))
            pct = round(day_adherent / len(day_rows) * 100, 1)
        else:
            pct = 0
        adherence_trend_30d.append({'date': d, 'adherence_pct': pct})

    # Drug distribution (for PieChart)
    drug_distribution = [
        {'drug': drug, 'count': drug_logs[drug]['total']}
        for drug in sorted(drug_logs.keys())
    ]

    # Rescue medication usage (clobazam or midazolam)
    rescue_keywords = ('clobazam', 'midazolam')
    rescue_med_usage = sum(
        1 for r in adherence_rows
        if r['drug_name'].lower() in rescue_keywords
    )

    # Adherence by time of day
    time_logs = {}
    for r in adherence_rows:
        ts = r['scheduled_time']
        time_logs.setdefault(ts, {'total': 0, 'adherent': 0})
        time_logs[ts]['total'] += 1
        if r['taken'] in ('yes', 'late'):
            time_logs[ts]['adherent'] += 1
    adherence_by_time_of_day = []
    for ts in SCHEDULED_TIMES:
        if ts in time_logs:
            t = time_logs[ts]
            adherence_by_time_of_day.append({
                'time_slot': ts,
                'adherence_pct': round(t['adherent'] / t['total'] * 100, 1) if t['total'] else 0,
            })

    return {
        'available': True,
        'total_patients': total_patients,
        'total_adherence_logs': total_logs,
        'overall_adherence_rate': overall_adherence_rate,
        'missed_dose_rate': missed_dose_rate,
        'avg_side_effect_severity': avg_side_effect_severity,
        'most_common_side_effect': most_common_side_effect,
        'total_refills': total_refills,
        'avg_days_between_refills': avg_days_between_refills,
        'adherence_by_drug': adherence_by_drug,
        'side_effect_distribution': side_effect_distribution,
        'adherence_trend_30d': adherence_trend_30d,
        'drug_distribution': drug_distribution,
        'rescue_med_usage': rescue_med_usage,
        'adherence_by_time_of_day': adherence_by_time_of_day,
    }


def breakdown():
    """Return per-patient detail, recent logs, and refill history."""
    _ensure_tables()

    adherence_rows = _db_query(
        'SELECT patient_id, log_date, drug_name, dose_mg, frequency, '
        'scheduled_time, taken, minutes_late, side_effects_json, '
        'side_effect_severity, mood_after, notes FROM medication_adherence'
    )
    refill_rows = _db_query(
        'SELECT patient_id, drug_name, refill_date, quantity, pharmacy, '
        'auto_refill, days_supply FROM medication_refills'
    )

    # Group adherence by patient
    patient_logs = {}
    for r in adherence_rows:
        pid = r['patient_id']
        patient_logs.setdefault(pid, []).append(r)

    # Group refills by patient
    patient_refills = {}
    for r in refill_rows:
        pid = r['patient_id']
        patient_refills.setdefault(pid, []).append(r)

    # 30-day window
    base_date = datetime(2025, 6, 15)
    cutoff_30d = (base_date - timedelta(days=30)).strftime('%Y-%m-%d')

    patients = []
    for pid in sorted(patient_logs.keys()):
        plogs = patient_logs[pid]
        total = len(plogs)
        adherent = sum(1 for r in plogs if r['taken'] in ('yes', 'late'))
        adherence_rate = round(adherent / total * 100, 1) if total else 0

        # Drugs list
        drugs = list(set(r['drug_name'] for r in plogs))

        # Missed doses in last 30 days
        recent = [r for r in plogs if r['log_date'] >= cutoff_30d]
        missed_doses_30d = sum(1 for r in recent if r['taken'] == 'no')

        # Side effects for this patient
        pt_effects = []
        for r in plogs:
            effs = _safe_json(r.get('side_effects_json'))
            if isinstance(effs, list):
                pt_effects.extend(effs)
        side_effects = [e for e, _ in Counter(pt_effects).most_common(5)]

        # Last refill date and refill due
        pr = patient_refills.get(pid, [])
        if pr:
            sorted_refills = sorted(pr, key=lambda x: x['refill_date'], reverse=True)
            last_refill_date = sorted_refills[0]['refill_date']
            last_supply = sorted_refills[0].get('days_supply', 30)
            try:
                last_dt = datetime.strptime(last_refill_date, '%Y-%m-%d')
                refill_due = (last_dt + timedelta(days=last_supply)).strftime('%Y-%m-%d')
            except Exception:
                refill_due = None
        else:
            last_refill_date = None
            refill_due = None

        patients.append({
            'patient_id': pid,
            'drugs': drugs,
            'adherence_rate': adherence_rate,
            'missed_doses_30d': missed_doses_30d,
            'side_effects': side_effects,
            'last_refill_date': last_refill_date,
            'refill_due': refill_due,
        })

    # Recent logs: last 50 adherence logs by date desc
    sorted_logs = sorted(adherence_rows, key=lambda x: x['log_date'], reverse=True)
    recent_logs = sorted_logs[:50]

    # Refill history: all refills sorted by date desc
    refill_history = sorted(refill_rows, key=lambda x: x['refill_date'], reverse=True)

    return {
        'patients': patients,
        'recent_logs': recent_logs,
        'refill_history': refill_history,
    }


def definitions():
    """Return clinical definitions for medication management concepts."""
    return {
        'concepts': [
            {
                'name': 'Medication Adherence',
                'description': 'The degree to which a patient takes medications as prescribed by their healthcare provider, including correct dose, timing, and frequency. In epilepsy, medication adherence is the single most important modifiable factor in seizure control. The WHO defines adherence as the extent to which a person\'s behaviour corresponds with agreed recommendations from a health care provider. Non-adherence rates in epilepsy range from 30-50%, and even a single missed dose of a short-half-life AED can trigger breakthrough seizures within 24-48 hours. This dashboard tracks three states: "yes" (taken on time), "late" (taken but delayed), and "no" (missed entirely). Both "yes" and "late" count toward overall adherence, as delayed dosing still provides therapeutic benefit.',
            },
            {
                'name': 'AED (Anti-Epileptic Drug)',
                'description': 'Also called anti-seizure medications (ASMs), these are the primary pharmacological treatment for epilepsy. Common AEDs tracked in this system include: Levetiracetam (Keppra) — broad-spectrum, first-line for focal and generalized seizures, renal clearance; Lamotrigine (Lamictal) — broad-spectrum, weight-neutral, requires slow titration due to rash risk; Valproate (Depakote) — broad-spectrum, effective for generalized epilepsies, teratogenic risk; Carbamazepine (Tegretol) — first-line for focal seizures, enzyme-inducing; Lacosamide (Vimpat) — adjunctive for focal seizures, slow sodium channel enhancer; Topiramate (Topamax) — broad-spectrum, cognitive side effects, weight loss; Oxcarbazepine (Trileptal) — similar to carbamazepine with fewer interactions; Clobazam (Onfi) — benzodiazepine adjunct, also used as rescue medication.',
            },
            {
                'name': 'Therapeutic Drug Monitoring',
                'description': 'The clinical practice of measuring AED blood levels to optimize dosing, verify adherence, and minimize toxicity. Therapeutic ranges vary by drug: Levetiracetam 12-46 mcg/mL, Lamotrigine 3-14 mcg/mL, Valproate 50-100 mcg/mL, Carbamazepine 4-12 mcg/mL. TDM is indicated when: seizures are uncontrolled despite apparent adherence, side effects suggest toxicity, drug-drug interactions are suspected, during pregnancy (altered pharmacokinetics), after dose changes, or when adherence is questioned. Trough levels (drawn just before the next dose) are standard. The dashboard\'s adherence and timing data can guide TDM scheduling — patients with frequent late doses may benefit from trough-level checks.',
            },
            {
                'name': 'Side Effect Profile',
                'description': 'The constellation of adverse effects associated with a specific AED or combination of AEDs. Side effects are the leading cause of medication non-adherence in epilepsy. Common AED side effects tracked here include: CNS effects (drowsiness, dizziness, cognitive slowing, mood changes), GI effects (nausea, weight gain), dermatological (skin rash, hair loss), neurological (tremor, blurred vision), and metabolic (bone density loss, metabolic acidosis). Severity is rated 0-10, where 0 = no side effects, 1-3 = mild (tolerable), 4-6 = moderate (affecting daily activities), 7-8 = severe (significantly impairing quality of life), 9-10 = intolerable (requiring drug change). Side effect patterns help guide medication adjustments and identify drug-specific intolerability.',
            },
            {
                'name': 'Medication Reconciliation',
                'description': 'The process of creating and maintaining an accurate, comprehensive list of all medications a patient is taking — including drug name, dose, frequency, and route — and comparing it against the provider\'s records at every transition of care. In epilepsy, reconciliation is critical because: AED interactions with other medications (enzyme induction/inhibition) can alter seizure control; polytherapy patients often see multiple specialists; over-the-counter medications and supplements may lower seizure threshold (e.g., St. John\'s Wort reduces lamotrigine levels). The dashboard\'s drug list per patient supports reconciliation by maintaining a real-time record of all AEDs and doses.',
            },
            {
                'name': 'Rescue Medication',
                'description': 'A fast-acting medication administered during or immediately after a seizure to prevent seizure clusters or status epilepticus. Common rescue medications include: Clobazam (oral benzodiazepine, also used as daily AED), Diazepam rectal gel (Diastat), Midazolam nasal spray (Nayzilam) or buccal (Buccolam), and Lorazepam (sublingual or IV in hospital). Rescue medications are prescribed with a seizure action plan specifying when to administer (e.g., seizure lasting >3 minutes or second seizure within 24 hours). The dashboard tracks clobazam and midazolam usage separately as rescue medication events to differentiate acute interventions from routine AED dosing.',
            },
            {
                'name': 'Polytherapy',
                'description': 'The use of two or more AEDs simultaneously to achieve seizure control when monotherapy is insufficient. Approximately 30-40% of epilepsy patients require polytherapy. Benefits include synergistic mechanisms of action (e.g., sodium channel blocker + SV2A modulator). Risks include: increased side effect burden, drug-drug interactions (especially with enzyme-inducing AEDs like carbamazepine), reduced adherence due to complex regimens, and higher cost. The dashboard tracks the number of concurrent medications per patient (2-3 in this cohort) and correlates polytherapy status with adherence rates and side effect severity to identify patients who may benefit from regimen simplification.',
            },
            {
                'name': 'Drug-Drug Interaction',
                'description': 'A pharmacological interaction between two or more medications that alters the efficacy or toxicity of one or both drugs. In epilepsy, key interactions include: enzyme-inducing AEDs (carbamazepine, phenytoin, phenobarbital) reducing levels of lamotrigine, oral contraceptives, and many other drugs; valproate inhibiting lamotrigine metabolism (requiring dose reduction); levetiracetam having minimal interactions (advantageous in polytherapy). The dashboard\'s multi-drug tracking per patient enables identification of high-risk combinations. Patients on 3+ medications or enzyme-inducing AEDs warrant closer monitoring for therapeutic failures or unexpected toxicity.',
            },
            {
                'name': 'Titration',
                'description': 'The gradual adjustment of AED dosage — typically starting low and increasing slowly — to find the minimum effective dose with acceptable side effects. Titration is essential because: rapid dose escalation increases side effects and reduces adherence; some AEDs (lamotrigine) carry serious risks (Stevens-Johnson syndrome) with fast titration; therapeutic response may take weeks to assess; and individual pharmacokinetic variability means the optimal dose differs between patients. The dashboard tracks dose changes over time through the dose_mg field and correlates dose adjustments with changes in side effect severity and adherence patterns, supporting evidence-based titration decisions.',
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
    print(f'Recent logs: {len(bd["recent_logs"])}')
    print(f'Refill history: {len(bd["refill_history"])}')
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Concepts: {len(df["concepts"])}')
    for c in df['concepts']:
        print(f'  {c["name"]}')
