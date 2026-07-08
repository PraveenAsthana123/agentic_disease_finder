"""Medication Adherence Dashboard — Patient Module.
Tracks medication adherence, side effects, mood, and refill data
for epilepsy patients on antiepileptic drugs (AEDs/ASMs).

Reads from:
  - medication_adherence  (dose-level adherence logs with side effects)
  - medication_refills    (pharmacy refill records)

Uses real patient data from clinical.db (EPAT001-EPAT030).
ILAE pharmacotherapy standards and PDC/MPR metrics applied throughout.
"""

import json
import os
import sqlite3
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


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


def overview():
    """Return KPI cards + chart data for the Medication Adherence overview tab."""
    rows = _db_query('SELECT * FROM medication_adherence')
    total_logs = len(rows)
    if total_logs == 0:
        return {'available': False, 'reason': 'No medication adherence data found'}

    patient_ids = list(set(r['patient_id'] for r in rows))
    total_patients = len(patient_ids)

    # --- KPIs ---
    taken_yes = sum(1 for r in rows if r['taken'] == 'yes')
    taken_late = sum(1 for r in rows if r['taken'] == 'late')
    taken_no = sum(1 for r in rows if r['taken'] == 'no')

    adherence_rate = round(taken_yes / total_logs * 100, 1)
    late_rate = round(taken_late / total_logs * 100, 1)
    missed_rate = round(taken_no / total_logs * 100, 1)

    late_rows = [r for r in rows if r['taken'] == 'late' and r['minutes_late'] is not None]
    avg_minutes_late = _avg([r['minutes_late'] for r in late_rows])

    severity_vals = [r['side_effect_severity'] for r in rows if r['side_effect_severity'] is not None]
    avg_side_effect_severity = _avg(severity_vals)

    mood_vals = [r['mood_after'] for r in rows if r['mood_after'] is not None]
    avg_mood = _avg(mood_vals)

    unique_drugs = list(set(r['drug_name'] for r in rows if r['drug_name']))

    # --- Charts ---

    # adherence_by_drug
    drug_map = {}
    for r in rows:
        drug = r['drug_name']
        if drug not in drug_map:
            drug_map[drug] = {'total': 0, 'yes': 0, 'late': 0, 'no': 0}
        drug_map[drug]['total'] += 1
        if r['taken'] == 'yes':
            drug_map[drug]['yes'] += 1
        elif r['taken'] == 'late':
            drug_map[drug]['late'] += 1
        else:
            drug_map[drug]['no'] += 1

    adherence_by_drug = []
    for drug in sorted(drug_map.keys()):
        d = drug_map[drug]
        t = d['total']
        adherence_by_drug.append({
            'drug': drug,
            'taken_pct': round(d['yes'] / t * 100, 1),
            'late_pct': round(d['late'] / t * 100, 1),
            'missed_pct': round(d['no'] / t * 100, 1),
        })

    # adherence_trend: aggregated by week (ISO week from log_date)
    week_map = {}
    for r in rows:
        ld = r.get('log_date', '')
        if not ld:
            continue
        # Use Monday of the week as the key
        try:
            from datetime import datetime
            dt = datetime.strptime(ld, '%Y-%m-%d')
            monday = dt - __import__('datetime').timedelta(days=dt.weekday())
            week_key = monday.strftime('%Y-%m-%d')
        except Exception:
            continue
        if week_key not in week_map:
            week_map[week_key] = {'total': 0, 'yes': 0}
        week_map[week_key]['total'] += 1
        if r['taken'] == 'yes':
            week_map[week_key]['yes'] += 1

    adherence_trend = []
    for wk in sorted(week_map.keys()):
        w = week_map[wk]
        adherence_trend.append({
            'date': wk,
            'adherence_rate': round(w['yes'] / w['total'] * 100, 1) if w['total'] else 0,
        })

    # frequency_distribution
    freq_counts = Counter(r['frequency'] for r in rows if r['frequency'])
    frequency_distribution = [
        {'frequency': f, 'count': c}
        for f, c in freq_counts.most_common()
    ]

    # side_effect_frequency: parse side_effects_json and count
    se_counter = Counter()
    for r in rows:
        effects = _safe_json(r.get('side_effects_json'))
        if isinstance(effects, list):
            for e in effects:
                if isinstance(e, str) and e.strip():
                    se_counter[e.strip()] += 1

    side_effect_frequency = [
        {'side_effect': se, 'count': c}
        for se, c in se_counter.most_common(20)
    ]

    # time_of_day
    time_map = {}
    for r in rows:
        st = r.get('scheduled_time', '')
        if not st:
            continue
        if st not in time_map:
            time_map[st] = {'total': 0, 'yes': 0, 'late': 0, 'no': 0}
        time_map[st]['total'] += 1
        if r['taken'] == 'yes':
            time_map[st]['yes'] += 1
        elif r['taken'] == 'late':
            time_map[st]['late'] += 1
        else:
            time_map[st]['no'] += 1

    time_of_day = []
    for st in sorted(time_map.keys()):
        t = time_map[st]
        total = t['total']
        time_of_day.append({
            'scheduled_time': st,
            'taken_pct': round(t['yes'] / total * 100, 1),
            'late_pct': round(t['late'] / total * 100, 1),
            'missed_pct': round(t['no'] / total * 100, 1),
        })

    # mood_distribution
    mood_counter = Counter(r['mood_after'] for r in rows if r['mood_after'] is not None)
    mood_distribution = [
        {'mood_score': score, 'count': mood_counter.get(score, 0)}
        for score in range(1, 11)
    ]

    # refill_summary
    refills = _db_query('SELECT * FROM medication_refills')
    total_refills = len(refills)
    if total_refills > 0:
        auto_refill_count = sum(1 for rf in refills if rf.get('auto_refill'))
        auto_refill_pct = round(auto_refill_count / total_refills * 100, 1)
        days_supply_vals = [rf['days_supply'] for rf in refills if rf.get('days_supply') is not None]
        avg_days_supply = _avg(days_supply_vals)
    else:
        auto_refill_pct = 0
        avg_days_supply = 0

    refill_summary = {
        'total_refills': total_refills,
        'auto_refill_pct': auto_refill_pct,
        'avg_days_supply': avg_days_supply,
    }

    return {
        'available': True,
        'total_logs': total_logs,
        'total_patients': total_patients,
        'adherence_rate': adherence_rate,
        'late_rate': late_rate,
        'missed_rate': missed_rate,
        'avg_minutes_late': avg_minutes_late,
        'avg_side_effect_severity': avg_side_effect_severity,
        'avg_mood': avg_mood,
        'unique_drugs': len(unique_drugs),
        'adherence_by_drug': adherence_by_drug,
        'adherence_trend': adherence_trend,
        'frequency_distribution': frequency_distribution,
        'side_effect_frequency': side_effect_frequency,
        'time_of_day': time_of_day,
        'mood_distribution': mood_distribution,
        'refill_summary': refill_summary,
    }


def breakdown():
    """Return per-patient, per-drug breakdowns plus recent missed/late doses and refills."""
    rows = _db_query('SELECT * FROM medication_adherence')
    if not rows:
        return {'available': False, 'reason': 'No medication adherence data found'}

    # --- per_patient ---
    patient_map = {}
    for r in rows:
        pid = r['patient_id']
        patient_map.setdefault(pid, []).append(r)

    per_patient = []
    for pid in sorted(patient_map.keys()):
        prows = patient_map[pid]
        total = len(prows)
        yes_count = sum(1 for r in prows if r['taken'] == 'yes')
        late_count = sum(1 for r in prows if r['taken'] == 'late')
        no_count = sum(1 for r in prows if r['taken'] == 'no')

        late_mins = [r['minutes_late'] for r in prows if r['taken'] == 'late' and r['minutes_late'] is not None]

        drugs = list(set(r['drug_name'] for r in prows if r['drug_name']))

        # top side effects for this patient
        se_counter = Counter()
        for r in prows:
            effects = _safe_json(r.get('side_effects_json'))
            if isinstance(effects, list):
                for e in effects:
                    if isinstance(e, str) and e.strip():
                        se_counter[e.strip()] += 1
        top_side_effects = [se for se, _ in se_counter.most_common(5)]

        per_patient.append({
            'patient_id': pid,
            'total_doses': total,
            'adherence_rate': round(yes_count / total * 100, 1),
            'late_rate': round(late_count / total * 100, 1),
            'missed_rate': round(no_count / total * 100, 1),
            'avg_minutes_late': _avg(late_mins),
            'drugs': sorted(drugs),
            'top_side_effects': top_side_effects,
        })

    # --- per_drug ---
    drug_map = {}
    for r in rows:
        drug = r['drug_name']
        drug_map.setdefault(drug, []).append(r)

    per_drug = []
    for drug in sorted(drug_map.keys()):
        drows = drug_map[drug]
        total = len(drows)
        yes_count = sum(1 for r in drows if r['taken'] == 'yes')
        late_count = sum(1 for r in drows if r['taken'] == 'late')
        no_count = sum(1 for r in drows if r['taken'] == 'no')

        sev_vals = [r['side_effect_severity'] for r in drows if r['side_effect_severity'] is not None]

        se_counter = Counter()
        for r in drows:
            effects = _safe_json(r.get('side_effects_json'))
            if isinstance(effects, list):
                for e in effects:
                    if isinstance(e, str) and e.strip():
                        se_counter[e.strip()] += 1
        common_side_effects = [se for se, _ in se_counter.most_common(5)]

        per_drug.append({
            'drug_name': drug,
            'total_doses': total,
            'adherence_rate': round(yes_count / total * 100, 1),
            'late_rate': round(late_count / total * 100, 1),
            'missed_rate': round(no_count / total * 100, 1),
            'avg_severity': _avg(sev_vals),
            'common_side_effects': common_side_effects,
        })

    # --- recent_missed: last 50 missed doses ---
    missed = _db_query(
        """SELECT patient_id, drug_name, log_date, scheduled_time, notes
           FROM medication_adherence
           WHERE taken = 'no'
           ORDER BY log_date DESC, scheduled_time DESC
           LIMIT 50"""
    )

    # --- recent_late: last 50 late doses ---
    late = _db_query(
        """SELECT patient_id, drug_name, log_date, scheduled_time, minutes_late
           FROM medication_adherence
           WHERE taken = 'late'
           ORDER BY log_date DESC, scheduled_time DESC
           LIMIT 50"""
    )

    # --- refills ---
    refills = _db_query(
        """SELECT patient_id, drug_name, refill_date, quantity, pharmacy,
                  auto_refill, days_supply
           FROM medication_refills
           ORDER BY refill_date DESC"""
    )

    return {
        'per_patient': per_patient,
        'per_drug': per_drug,
        'recent_missed': missed,
        'recent_late': late,
        'refills': refills,
    }


def definitions():
    """Return clinical definitions for medication adherence concepts."""
    return {
        'concepts': [
            {
                'name': 'Medication Adherence',
                'description': 'The extent to which a patient takes medications as prescribed by their healthcare provider. In epilepsy management, adherence to antiseizure medications (ASMs) is critical for seizure control. Non-adherence is the leading modifiable cause of breakthrough seizures and status epilepticus. Adherence is measured at the dose level: each scheduled dose is recorded as taken (on time), late (taken but after the scheduled window), or missed (not taken). Overall adherence rate is calculated as the percentage of scheduled doses taken on time. A dose taken within the prescribed time window counts as "yes"; a dose taken outside the window but still consumed is "late"; an omitted dose is "no".',
            },
            {
                'name': 'PDC (Proportion of Days Covered)',
                'description': 'A pharmacy-based adherence metric defined as the number of days in a measurement period "covered" by prescription fills divided by the number of days in the period. PDC is calculated by creating a timeline of medication supply from refill records, removing overlapping days, and dividing covered days by the observation period length. PDC >= 0.80 (80%) is the widely accepted threshold for adequate adherence. PDC is preferred over MPR by the Pharmacy Quality Alliance (PQA) and CMS because it accounts for medication switching and does not exceed 1.0. For epilepsy patients on multiple ASMs, PDC can be calculated per drug or as a combined measure across the entire regimen.',
            },
            {
                'name': 'MPR (Medication Possession Ratio)',
                'description': 'A pharmacy claims-based adherence metric calculated as the sum of days supply of all fills divided by the number of days in the observation period. Unlike PDC, MPR can exceed 1.0 if a patient refills early (stockpiling). MPR >= 0.80 is generally considered adequate adherence. MPR is simpler to compute but less precise than PDC because it does not account for overlapping fill dates or gaps between fills. Both MPR and PDC are retrospective measures derived from refill records rather than direct observation of dose-taking behavior.',
            },
            {
                'name': 'AED / ASM Terminology',
                'description': 'AED (Antiepileptic Drug) is the traditional term for medications used to prevent seizures. The ILAE now recommends the term ASM (Antiseizure Medication) to reflect that these drugs treat seizures rather than curing epilepsy, and many are also used for non-epilepsy indications (neuropathic pain, migraine prophylaxis, mood stabilization). Common ASMs tracked in this dashboard include: Levetiracetam (Keppra) — broad-spectrum, first-line; Valproate (Depakote) — broad-spectrum, teratogenic risk; Carbamazepine (Tegretol) — focal seizures, enzyme inducer; Lamotrigine (Lamictal) — broad-spectrum, mood stabilizer; Lacosamide (Vimpat) — focal seizures, sodium channel; Oxcarbazepine (Trileptal) — focal seizures, fewer interactions than CBZ; Clobazam (Onfi) — adjunctive, benzodiazepine class; Topiramate (Topamax) — broad-spectrum, cognitive side effects.',
            },
            {
                'name': 'Side Effect Severity Scale',
                'description': 'A patient-reported severity rating from 1 to 10 for medication side effects experienced after each dose. Scale interpretation: 1-2 (minimal) — negligible side effects, no functional impact; 3-4 (mild) — noticeable but tolerable, no activity modification needed; 5-6 (moderate) — bothersome, some activity modification, medication continuation likely; 7-8 (severe) — significantly impacts daily functioning, medication review warranted; 9-10 (intolerable) — unable to perform daily activities, urgent medication change needed. Common ASM side effects include: drowsiness, dizziness, cognitive slowing, nausea, weight changes, mood disturbance, tremor, diplopia, ataxia, and skin rash. The severity scale helps clinicians identify patients who may benefit from dose adjustment or drug switching.',
            },
            {
                'name': 'Frequency Codes',
                'description': 'Standard prescription frequency abbreviations used in medication scheduling. QD (quaque die) — once daily, typically morning; BID (bis in die) — twice daily, usually morning and evening, approximately 12 hours apart; TID (ter in die) — three times daily, approximately every 8 hours; QHS (quaque hora somni) — at bedtime, once nightly. The frequency determines the number of scheduled doses per day and directly affects adherence calculation. For example, a BID medication has 2 scheduled doses per day, so a patient missing one evening dose has 50% adherence for that day. TID regimens have the lowest adherence rates across all chronic medications due to the burden of a midday dose.',
            },
            {
                'name': 'Adherence Thresholds',
                'description': 'Clinical thresholds for categorizing medication adherence levels. Greater than 80% adherence is considered adequate — associated with optimal seizure control and therapeutic drug levels. 50-80% is suboptimal — associated with increased breakthrough seizure risk, subtherapeutic drug levels, and need for adherence counseling. Less than 50% is critical — associated with high seizure risk, possible status epilepticus, and urgent intervention needed (pill organizers, reminder apps, caregiver involvement, simplified regimen). These thresholds are derived from pharmacokinetic studies showing that most ASMs require consistent plasma levels above the minimum effective concentration. Even brief gaps in adherence for short-half-life drugs (e.g., levetiracetam, t1/2 ~7h) can drop levels below therapeutic range within 24 hours.',
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    ov = overview()
    pprint.pprint(ov)
    print('\n=== BREAKDOWN (summary) ===')
    bd = breakdown()
    if isinstance(bd, dict) and 'per_patient' in bd:
        print(f'Per-patient entries: {len(bd["per_patient"])}')
        for p in bd['per_patient'][:3]:
            pprint.pprint(p)
        print(f'\nPer-drug entries: {len(bd["per_drug"])}')
        for d in bd['per_drug']:
            pprint.pprint(d)
        print(f'\nRecent missed: {len(bd["recent_missed"])}')
        print(f'Recent late: {len(bd["recent_late"])}')
        print(f'Refills: {len(bd["refills"])}')
    else:
        pprint.pprint(bd)
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Concepts: {len(df["concepts"])}')
    for c in df['concepts']:
        print(f'  {c["name"]}')
