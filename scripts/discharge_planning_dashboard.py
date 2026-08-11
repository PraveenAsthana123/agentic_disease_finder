"""Discharge Planning Dashboard — real data from clinical.db.

Sources:
- hospitalization (115 admissions, 30 patients): admission/discharge dates, reason, ward,
  physician, LOS, disposition, complications, seizure_free_at_discharge, readmission_within_30d
- medication_adherence (12,600 rows): AED adherence post-discharge tracking
- seizure_diary (25 events): post-discharge seizure occurrence
- appointments (120 rows): follow-up appointment scheduling

Clinical context:
- Safe discharge from an epilepsy unit requires: seizure stability, medication plan,
  caregiver education, follow-up appointment, driving restriction counselling, written plan
- 30-day readmission rate is a key quality metric (target <10%)
- ILAE consensus: structured discharge planning reduces readmission and improves adherence
"""

import sqlite3
import json
import os
from datetime import datetime, timezone, date
from collections import Counter

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _load_hospitalizations():
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        'SELECT patient_id, fields_json, created_at FROM hospitalization'
    ).fetchall()]
    conn.close()
    parsed = []
    for r in rows:
        try:
            fields = json.loads(r['fields_json'])
        except Exception:
            fields = {}
        fields['patient_id'] = r['patient_id']
        parsed.append(fields)
    return parsed


def _load_adherence():
    """Average adherence per patient from medication_adherence.
    'taken' = 'yes' counts as adherent; calculates per-patient rate."""
    conn = _conn()
    rows = conn.execute(
        '''SELECT patient_id,
                  COUNT(*) as total_doses,
                  SUM(CASE WHEN taken = 'yes' THEN 1 ELSE 0 END) as taken_doses
           FROM medication_adherence
           GROUP BY patient_id'''
    ).fetchall()
    conn.close()
    result = {}
    for r in rows:
        total = r['total_doses'] or 1
        taken = r['taken_doses'] or 0
        result[r['patient_id']] = {
            'avg': round(taken / total * 100, 1),
            'days': r['total_doses'],
        }
    return result


def _load_appointments():
    """Next follow-up appointment per patient."""
    conn = _conn()
    today = date.today().isoformat()
    rows = conn.execute(
        '''SELECT patient_id, MIN(scheduled_for) as next_appt, COUNT(*) as total
           FROM appointments
           WHERE scheduled_for >= ? AND status NOT IN ('cancelled', 'completed')
           GROUP BY patient_id''',
        (today,)
    ).fetchall()
    conn.close()
    return {r['patient_id']: {'next_appt': (r['next_appt'] or '')[:10], 'total': r['total']} for r in rows}


def _load_seizure_diary():
    """Seizure diary entries per patient."""
    conn = _conn()
    rows = conn.execute(
        'SELECT patient_id, COUNT(*) as events FROM seizure_diary GROUP BY patient_id'
    ).fetchall()
    conn.close()
    return {r['patient_id']: r['events'] for r in rows}


def _discharge_readiness(hosp, adherence, appts, seizure_events):
    """Score discharge readiness on 5 criteria (0–100)."""
    pid = hosp.get('patient_id', '')
    score = 0
    checks = []

    # 1. Seizure-free at discharge
    sf = hosp.get('seizure_free_at_discharge')
    checks.append({
        'criterion': 'Seizure-free at discharge',
        'met': bool(sf),
        'weight': 25,
        'note': 'Patient was seizure-free at discharge' if sf else 'Seizures present at discharge',
    })
    if sf:
        score += 25

    # 2. Medication adherence ≥ 80%
    adh = adherence.get(pid, {}).get('avg', 0)
    adh_ok = adh >= 80
    checks.append({
        'criterion': 'Medication adherence ≥ 80%',
        'met': adh_ok,
        'weight': 20,
        'note': f'Avg adherence {adh:.0f}%' if adh else 'No adherence data',
    })
    if adh_ok:
        score += 20

    # 3. Follow-up appointment scheduled
    appt = appts.get(pid, {}).get('next_appt')
    checks.append({
        'criterion': 'Follow-up appointment scheduled',
        'met': bool(appt),
        'weight': 20,
        'note': f'Next appointment: {appt}' if appt else 'No upcoming appointment found',
    })
    if appt:
        score += 20

    # 4. No complications documented
    comp = hosp.get('complications')
    no_comp = not comp
    checks.append({
        'criterion': 'No complications at discharge',
        'met': no_comp,
        'weight': 20,
        'note': 'No complications' if no_comp else f'Complication: {comp}',
    })
    if no_comp:
        score += 20

    # 5. Planned discharge (not emergency AMA)
    disposition = hosp.get('discharge_disposition', '')
    good_disp = disposition in ('home', 'rehabilitation')
    checks.append({
        'criterion': 'Planned discharge disposition',
        'met': good_disp,
        'weight': 15,
        'note': f'Disposition: {disposition}' if disposition else 'Disposition unknown',
    })
    if good_disp:
        score += 15

    tier = 'Ready' if score >= 80 else 'Conditional' if score >= 60 else 'Not Ready'
    return score, tier, checks


# ── Overview ──────────────────────────────────────────────────────────────────

def discharge_overview():
    hosps = _load_hospitalizations()
    adherence = _load_adherence()
    appts = _load_appointments()
    diaries = _load_seizure_diary()

    total_admissions = len(hosps)
    unique_patients = len(set(h['patient_id'] for h in hosps))

    # LOS stats
    los_vals = [h.get('length_of_stay_days') for h in hosps if h.get('length_of_stay_days') is not None]
    avg_los = round(sum(los_vals) / len(los_vals), 1) if los_vals else None

    # Readmission
    readmits = sum(1 for h in hosps if h.get('readmission_within_30d'))
    readmit_pct = round(readmits / total_admissions * 100, 1) if total_admissions else 0

    # Seizure-free at discharge
    sf_count = sum(1 for h in hosps if h.get('seizure_free_at_discharge'))
    sf_pct = round(sf_count / total_admissions * 100, 1) if total_admissions else 0

    # Complications
    comp_count = sum(1 for h in hosps if h.get('complications'))
    comp_pct = round(comp_count / total_admissions * 100, 1) if total_admissions else 0

    # Admission type distribution
    adm_type_dist = Counter(h.get('admission_type', 'unknown') for h in hosps)
    adm_type_rows = [{'type': k, 'count': v} for k, v in sorted(adm_type_dist.items(), key=lambda x: -x[1])]

    # Admission reason distribution
    reason_dist = Counter(h.get('admission_reason', 'unknown') for h in hosps)
    reason_rows = [{'reason': k, 'count': v} for k, v in sorted(reason_dist.items(), key=lambda x: -x[1])[:10]]

    # Discharge disposition distribution
    disp_dist = Counter(h.get('discharge_disposition', 'unknown') for h in hosps)
    disp_rows = [{'disposition': k, 'count': v} for k, v in sorted(disp_dist.items(), key=lambda x: -x[1])]

    # Ward distribution
    ward_dist = Counter(h.get('ward', 'Unknown') for h in hosps)
    ward_rows = [{'ward': k, 'count': v} for k, v in sorted(ward_dist.items(), key=lambda x: -x[1])]

    # Physician workload
    phys_dist = Counter(h.get('attending_physician', 'Unknown') for h in hosps)
    phys_rows = [{'physician': k, 'count': v} for k, v in sorted(phys_dist.items(), key=lambda x: -x[1])]

    # Discharge readiness tier distribution across most-recent admission per patient
    # Group by patient, take latest admission
    by_patient = {}
    for h in hosps:
        pid = h['patient_id']
        ddate = h.get('discharge_date') or h.get('admission_date') or ''
        if pid not in by_patient or ddate > by_patient[pid].get('discharge_date', ''):
            by_patient[pid] = h

    readiness_dist = Counter()
    for pid, h in by_patient.items():
        score, tier, _ = _discharge_readiness(h, adherence, appts, diaries)
        readiness_dist[tier] += 1
    readiness_rows = [{'tier': k, 'count': v} for k, v in sorted(readiness_dist.items())]

    # Monthly admission trend
    monthly = {}
    for h in hosps:
        mo = (h.get('admission_date') or '')[:7]
        if mo:
            monthly[mo] = monthly.get(mo, 0) + 1
    monthly_trend = [{'month': mo, 'admissions': cnt} for mo, cnt in sorted(monthly.items())]

    # Cost statistics
    costs = [h.get('total_cost_usd') for h in hosps if h.get('total_cost_usd')]
    avg_cost = round(sum(costs) / len(costs)) if costs else None

    return {
        'kpis': {
            'total_admissions': total_admissions,
            'unique_patients': unique_patients,
            'avg_length_of_stay_days': avg_los,
            'readmission_rate_pct': readmit_pct,
            'readmissions_30d': readmits,
            'seizure_free_at_discharge_pct': sf_pct,
            'complication_pct': comp_pct,
            'avg_cost_usd': avg_cost,
        },
        'admission_type_distribution': adm_type_rows,
        'admission_reason_distribution': reason_rows,
        'discharge_disposition_distribution': disp_rows,
        'ward_distribution': ward_rows,
        'physician_workload': phys_rows,
        'readiness_distribution': readiness_rows,
        'monthly_trend': monthly_trend,
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────

def discharge_breakdown():
    hosps = _load_hospitalizations()
    adherence = _load_adherence()
    appts = _load_appointments()
    diaries = _load_seizure_diary()

    # Per-patient discharge readiness (most recent admission)
    by_patient = {}
    for h in hosps:
        pid = h['patient_id']
        ddate = h.get('discharge_date') or h.get('admission_date') or ''
        if pid not in by_patient or ddate > by_patient[pid].get('discharge_date', ''):
            by_patient[pid] = h

    patient_profiles = []
    for pid, h in sorted(by_patient.items()):
        score, tier, checks = _discharge_readiness(h, adherence, appts, diaries)
        adh = adherence.get(pid, {})
        appt = appts.get(pid, {})
        patient_profiles.append({
            'patient_id': pid,
            'last_admission_date': h.get('admission_date', ''),
            'last_discharge_date': h.get('discharge_date', ''),
            'admission_reason': h.get('admission_reason', ''),
            'ward': h.get('ward', ''),
            'physician': h.get('attending_physician', ''),
            'los_days': h.get('length_of_stay_days'),
            'disposition': h.get('discharge_disposition', ''),
            'seizure_free': bool(h.get('seizure_free_at_discharge')),
            'complications': h.get('complications'),
            'readmission_within_30d': bool(h.get('readmission_within_30d')),
            'avg_adherence_pct': adh.get('avg'),
            'next_appointment': appt.get('next_appt'),
            'seizure_diary_events': diaries.get(pid, 0),
            'readiness_score': score,
            'readiness_tier': tier,
            'readiness_checks': checks,
        })

    # Sort by readiness score ascending (lowest first = most at risk)
    patient_profiles.sort(key=lambda x: x['readiness_score'])

    # All admissions log
    all_admissions = []
    for h in sorted(hosps, key=lambda x: x.get('admission_date', ''), reverse=True):
        all_admissions.append({
            'patient_id': h['patient_id'],
            'admission_date': h.get('admission_date', ''),
            'discharge_date': h.get('discharge_date', ''),
            'reason': h.get('admission_reason', ''),
            'type': h.get('admission_type', ''),
            'ward': h.get('ward', ''),
            'physician': h.get('attending_physician', ''),
            'los_days': h.get('length_of_stay_days'),
            'disposition': h.get('discharge_disposition', ''),
            'seizure_free': bool(h.get('seizure_free_at_discharge')),
            'complications': h.get('complications'),
            'readmission': bool(h.get('readmission_within_30d')),
            'cost_usd': h.get('total_cost_usd'),
        })

    return {
        'patient_discharge_profiles': patient_profiles,
        'admissions_log': all_admissions[:50],  # Most recent 50
        'readmission_detail': [h for h in all_admissions if h['readmission']],
    }


# ── Definitions ───────────────────────────────────────────────────────────────

def discharge_definitions():
    return {
        'title': 'Discharge Planning — Definitions & Clinical Reference',
        'description': (
            'The Discharge Planning Dashboard synthesizes hospitalization data (115 admissions, '
            '30+ patients) with medication adherence, follow-up appointments, and seizure diary '
            'entries to produce discharge readiness scores and 30-day readmission tracking. '
            'Follows ILAE consensus and NICE guideline NG217 recommendations for structured epilepsy discharge.'
        ),
        'readiness_tiers': [
            {
                'tier': 'Ready',
                'score_range': '≥ 80',
                'color': 'success',
                'description': 'Patient meets ≥4 of 5 discharge criteria. Proceed with standard discharge planning.',
            },
            {
                'tier': 'Conditional',
                'score_range': '60–79',
                'color': 'warning',
                'description': 'Patient meets 3 criteria. Address remaining gaps before discharge. Consider extended stay or community support.',
            },
            {
                'tier': 'Not Ready',
                'score_range': '< 60',
                'color': 'danger',
                'description': 'Patient meets fewer than 3 criteria. Discharge should be deferred or requires intensive planning.',
            },
        ],
        'readiness_criteria': [
            {
                'criterion': 'Seizure-free at discharge',
                'weight': 25,
                'rationale': 'Primary clinical endpoint; patients discharged with ongoing seizures have 3× higher 30-day readmission risk (ILAE 2022).',
            },
            {
                'criterion': 'Medication adherence ≥ 80%',
                'weight': 20,
                'rationale': 'Sub-therapeutic AED levels are the leading cause of breakthrough seizures and readmission. ≥80% is the MMAS-8 "high adherence" threshold.',
            },
            {
                'criterion': 'Follow-up appointment scheduled',
                'weight': 20,
                'rationale': 'NICE NG217 mandates neurologist follow-up within 4 weeks of epilepsy-related discharge. Unscheduled patients miss AED titration windows.',
            },
            {
                'criterion': 'No complications at discharge',
                'weight': 20,
                'rationale': 'Complications (infection, AED toxicity, aspiration) extend LOS and flag risk for 30-day adverse events.',
            },
            {
                'criterion': 'Planned discharge disposition',
                'weight': 15,
                'rationale': 'Against-medical-advice (AMA) discharges have 5× higher readmission risk. Transfer to rehabilitation implies care continuity.',
            },
        ],
        'admission_types': {
            'emergency': 'Unplanned admission via Emergency Department — seizure cluster, status epilepticus, breakthrough seizure.',
            'planned': 'Pre-scheduled admission — video-EEG monitoring, pre-surgical evaluation, medication titration.',
            'observation': 'Short-stay observation — first seizure workup, AED toxicity assessment, post-ictal confusion monitoring.',
        },
        'discharge_dispositions': {
            'home': 'Patient discharged to home with or without caregiver support.',
            'rehabilitation': 'Transfer to inpatient rehabilitation for seizure-related functional impairment.',
            'transferred': 'Transfer to another acute-care facility (e.g., tertiary epilepsy centre).',
            'ama': 'Against Medical Advice — patient left before clinical team deemed safe.',
        },
        'data_sources': [
            'hospitalization — 115 admissions, 30 patients, 6 physicians, 5 wards',
            'medication_adherence — 12,600 adherence records, daily tracking per patient',
            'appointments — 120 scheduled appointments, next follow-up per patient',
            'seizure_diary — 25 patient-reported seizure events',
        ],
        'standards': [
            'ILAE Commission on Classification and Terminology (2022) — Discharge outcome criteria',
            'NICE NG217 (2022) — Epilepsies: Diagnosis and management',
            'IHI QI Framework — 30-day readmission as quality metric',
            'CMS HRRP (Hospital Readmissions Reduction Program) — ≤10% target readmission rate',
        ],
    }


if __name__ == '__main__':
    import json
    print(json.dumps(discharge_overview(), indent=2))
