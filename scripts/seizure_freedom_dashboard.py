"""Seizure Freedom Tracker — real data from clinical.db.

Sources:
- surgical_outcomes: Engel Class, ILAE outcome, seizure_free flag, surgery type
- seizure_trigger_logs: day-level seizure occurrence per patient
- seizure_metadata: drug responsiveness, syndrome, current seizure frequency
- seizure_diary: patient-reported seizure events

Clinical significance:
- Seizure freedom is the primary goal of epilepsy treatment
- Engel Class IA = completely seizure-free (gold standard)
- ILAE Outcome 1 = completely seizure-free + no AEDs
- Milestones: 6 months (return-to-work/activities), 12 months (driving eligibility in most jurisdictions)
"""

import sqlite3
import json
import os
from datetime import datetime, date, timedelta

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _load_surgical():
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM surgical_outcomes ORDER BY surgery_date"
    ).fetchall()]
    conn.close()
    return rows


def _load_trigger_logs():
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT patient_id, log_date, seizure_occurred, primary_trigger, medication_adherence, missed_doses "
        "FROM seizure_trigger_logs ORDER BY log_date"
    ).fetchall()]
    conn.close()
    return rows


def _load_seizure_metadata():
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT patient_id, fields_json FROM seizure_metadata"
    ).fetchall()]
    conn.close()
    out = {}
    for r in rows:
        fields = json.loads(r['fields_json']) if r['fields_json'] else {}
        out[r['patient_id']] = fields
    return out


def _engel_to_freedom(ec):
    """Return (seizure_free, outcome_label, color)."""
    if ec in ('IA', 'IB'):
        return True, 'Seizure-Free', 'success'
    if ec in ('IC', 'ID', 'IIA', 'IIB', 'IIC', 'IID'):
        return False, 'Worthwhile Improvement', 'warning'
    if ec and ec.startswith('III'):
        return False, 'Worthwhile Improvement (marginal)', 'orange'
    return False, 'No Improvement', 'danger'


def _ilae_label(v):
    labels = {
        1: 'Completely seizure-free',
        2: '≥90% reduction',
        3: '50–90% reduction',
        4: '<50% reduction',
        5: 'Worse',
        6: 'Unable to assess',
    }
    return labels.get(v, f'ILAE {v}')


# ── Overview ──────────────────────────────────────────────────────────────────

def freedom_overview():
    """Aggregate seizure freedom metrics across the cohort."""
    surgical = _load_surgical()
    trigger_logs = _load_trigger_logs()
    meta = _load_seizure_metadata()

    # ── Surgical cohort stats ──
    total_procedures = len(surgical)
    unique_patients_surgical = len(set(r['patient_id'] for r in surgical))
    seizure_free_count = sum(1 for r in surgical if r.get('seizure_free'))
    seizure_free_pct = round(seizure_free_count / total_procedures * 100, 1) if total_procedures else 0

    # Engel distribution
    engel_dist = {}
    for r in surgical:
        ec = r.get('engel_class') or 'Unknown'
        engel_dist[ec] = engel_dist.get(ec, 0) + 1
    engel_rows = [{'class': k, 'count': v} for k, v in sorted(engel_dist.items())]

    # ILAE distribution
    ilae_dist = {}
    for r in surgical:
        il = r.get('ilae_outcome')
        key = _ilae_label(il) if il else 'Unknown'
        ilae_dist[key] = ilae_dist.get(key, 0) + 1
    ilae_rows = [{'outcome': k, 'count': v} for k, v in sorted(ilae_dist.items())]

    # Surgery type distribution
    stype_dist = {}
    for r in surgical:
        st = r.get('surgery_type') or 'Unknown'
        stype_dist[st] = stype_dist.get(st, 0) + 1
    stype_rows = [{'type': k, 'count': v} for k, v in sorted(stype_dist.items())]

    # Average follow-up
    fu_vals = [r['follow_up_months'] for r in surgical if r.get('follow_up_months')]
    avg_followup = round(sum(fu_vals) / len(fu_vals), 1) if fu_vals else None

    # AED reduction
    aed_reduced = sum(1 for r in surgical if r.get('aed_reduction'))
    aed_reduction_pct = round(aed_reduced / total_procedures * 100, 1) if total_procedures else 0

    # Complications
    complications = sum(1 for r in surgical if r.get('complications'))
    complication_pct = round(complications / total_procedures * 100, 1) if total_procedures else 0

    # Pre/post frequency reduction
    reduction_rows = [
        {
            'patient_id': r['patient_id'],
            'surgery_type': r.get('surgery_type') or '',
            'pre_freq': r.get('pre_surgery_frequency'),
            'post_freq': r.get('post_surgery_frequency'),
            'reduction_pct': (
                round((1 - r['post_surgery_frequency'] / r['pre_surgery_frequency']) * 100, 1)
                if (r.get('pre_surgery_frequency') and r.get('post_surgery_frequency')
                    and r['pre_surgery_frequency'] > 0)
                else None
            ),
        }
        for r in surgical if r.get('pre_surgery_frequency')
    ]
    avg_reduction = (
        round(sum(x['reduction_pct'] for x in reduction_rows if x['reduction_pct'] is not None)
              / max(1, sum(1 for x in reduction_rows if x['reduction_pct'] is not None)), 1)
    )

    # ── Trigger-log cohort: seizure day rate per patient ──
    by_pt = {}
    for r in trigger_logs:
        pid = r['patient_id']
        if pid not in by_pt:
            by_pt[pid] = {'total': 0, 'seizure_days': 0, 'last': ''}
        by_pt[pid]['total'] += 1
        if r.get('seizure_occurred'):
            by_pt[pid]['seizure_days'] += 1
        if (r.get('log_date') or '') > by_pt[pid]['last']:
            by_pt[pid]['last'] = r.get('log_date') or ''

    tl_seizure_free = sum(1 for v in by_pt.values() if v['seizure_days'] == 0)
    tl_patients = len(by_pt)
    tl_rate_pct = round(tl_seizure_free / tl_patients * 100, 1) if tl_patients else 0

    # ── Drug responsiveness from seizure_metadata ──
    dr_dist = {}
    for pid, fields in meta.items():
        dr = fields.get('drug_responsiveness', 'Unknown')
        dr_dist[dr] = dr_dist.get(dr, 0) + 1
    dr_rows = [{'response': k, 'count': v} for k, v in sorted(dr_dist.items(), key=lambda x: -x[1])]

    # Monthly seizure freedom trend from trigger logs
    monthly = {}
    for r in trigger_logs:
        mo = (r.get('log_date') or '')[:7]
        if not mo:
            continue
        if mo not in monthly:
            monthly[mo] = {'total_days': 0, 'seizure_days': 0}
        monthly[mo]['total_days'] += 1
        if r.get('seizure_occurred'):
            monthly[mo]['seizure_days'] += 1

    monthly_trend = [
        {
            'month': mo,
            'seizure_free_pct': round((1 - v['seizure_days'] / v['total_days']) * 100, 1)
            if v['total_days'] else 0,
        }
        for mo, v in sorted(monthly.items())
    ]

    return {
        'kpis': {
            'total_procedures': total_procedures,
            'unique_surgical_patients': unique_patients_surgical,
            'seizure_free_count': seizure_free_count,
            'seizure_free_pct': seizure_free_pct,
            'avg_followup_months': avg_followup,
            'aed_reduction_pct': aed_reduction_pct,
            'complication_pct': complication_pct,
            'avg_seizure_frequency_reduction_pct': avg_reduction,
        },
        'trigger_log_cohort': {
            'total_patients': tl_patients,
            'seizure_free_patients': tl_seizure_free,
            'seizure_free_pct': tl_rate_pct,
        },
        'engel_distribution': engel_rows,
        'ilae_distribution': ilae_rows,
        'surgery_type_distribution': stype_rows,
        'drug_responsiveness_distribution': dr_rows,
        'monthly_seizure_free_trend': monthly_trend,
        'frequency_reduction_summary': {
            'total_with_pre_post': len(reduction_rows),
            'avg_reduction_pct': avg_reduction,
        },
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────

def freedom_breakdown():
    """Per-patient seizure freedom profiles, milestone tracker, and cohort register."""
    surgical = _load_surgical()
    trigger_logs = _load_trigger_logs()
    meta = _load_seizure_metadata()

    # ── Per-procedure table ──
    procedures = []
    for r in surgical:
        ec = r.get('engel_class') or 'Unknown'
        sf, label, color = _engel_to_freedom(ec)
        ilae = r.get('ilae_outcome')
        fu = r.get('follow_up_months') or 0

        # Milestone eligibility
        driving_eligible = sf and fu >= 12
        return_activities = sf and fu >= 6

        # Frequency reduction
        pre = r.get('pre_surgery_frequency')
        post = r.get('post_surgery_frequency')
        freq_reduction = None
        if pre and post is not None and pre > 0:
            freq_reduction = round((1 - post / pre) * 100, 1)

        procedures.append({
            'patient_id': r['patient_id'],
            'surgery_type': r.get('surgery_type') or '',
            'surgery_date': (r.get('surgery_date') or '')[:10],
            'pathology': r.get('pathology') or '',
            'hemisphere': r.get('hemisphere') or '',
            'engel_class': ec,
            'engel_label': label,
            'engel_color': color,
            'ilae_outcome': ilae,
            'ilae_label': _ilae_label(ilae) if ilae else '—',
            'follow_up_months': fu,
            'seizure_free': bool(sf),
            'aed_reduction': bool(r.get('aed_reduction')),
            'complications': r.get('complications') or None,
            'complication_severity': r.get('complication_severity') or None,
            'pre_frequency': pre,
            'post_frequency': post,
            'freq_reduction_pct': freq_reduction,
            'driving_eligible': driving_eligible,
            'return_to_activities': return_activities,
        })

    # ── Trigger-log patient profiles ──
    by_pt = {}
    for r in trigger_logs:
        pid = r['patient_id']
        if pid not in by_pt:
            by_pt[pid] = {'dates': [], 'seizure_days': 0, 'adherence_sum': 0, 'adherence_count': 0}
        by_pt[pid]['dates'].append(r.get('log_date') or '')
        if r.get('seizure_occurred'):
            by_pt[pid]['seizure_days'] += 1
        adh = r.get('medication_adherence')
        if adh is not None:
            by_pt[pid]['adherence_sum'] += adh
            by_pt[pid]['adherence_count'] += 1

    tl_profiles = []
    for pid, v in sorted(by_pt.items()):
        dates_sorted = sorted(d for d in v['dates'] if d)
        total_days = len(dates_sorted)
        sz_days = v['seizure_days']
        seizure_free_pct = round((1 - sz_days / total_days) * 100, 1) if total_days else 0
        adh_pct = round(v['adherence_sum'] / v['adherence_count'], 1) if v['adherence_count'] else None

        m = meta.get(pid, {})
        tl_profiles.append({
            'patient_id': pid,
            'total_log_days': total_days,
            'seizure_days': sz_days,
            'seizure_free_pct': seizure_free_pct,
            'seizure_free': sz_days == 0,
            'avg_medication_adherence': adh_pct,
            'drug_responsiveness': m.get('drug_responsiveness', '—'),
            'current_seizure_frequency': m.get('current_seizure_frequency', '—'),
            'first_log': dates_sorted[0] if dates_sorted else None,
            'last_log': dates_sorted[-1] if dates_sorted else None,
        })

    tl_profiles.sort(key=lambda x: x['seizure_free_pct'], reverse=True)

    # ── Milestone summary ──
    milestones = [
        {'milestone': '6-month seizure-free', 'description': 'Return-to-activities/work eligibility',
         'patients': sum(1 for p in procedures if p['seizure_free'] and (p['follow_up_months'] or 0) >= 6),
         'total': len(set(p['patient_id'] for p in procedures))},
        {'milestone': '12-month seizure-free', 'description': 'Driving eligibility (most jurisdictions)',
         'patients': sum(1 for p in procedures if p['seizure_free'] and (p['follow_up_months'] or 0) >= 12),
         'total': len(set(p['patient_id'] for p in procedures))},
        {'milestone': '24-month seizure-free', 'description': 'Reduced monitoring frequency',
         'patients': sum(1 for p in procedures if p['seizure_free'] and (p['follow_up_months'] or 0) >= 24),
         'total': len(set(p['patient_id'] for p in procedures))},
    ]

    return {
        'procedures': procedures,
        'trigger_log_profiles': tl_profiles,
        'milestones': milestones,
    }


# ── Definitions ───────────────────────────────────────────────────────────────

def freedom_definitions():
    return {
        'title': 'Seizure Freedom Tracker — Definitions & Clinical Reference',
        'description': (
            'Tracks seizure freedom outcomes across 28 surgical procedures (25 patients) using '
            'Engel Classification and ILAE Outcome Scale. Cross-references daily seizure logs '
            '(203 trigger-log days, 40 patients) and drug-responsiveness profiles '
            '(71 patients, seizure_metadata) to provide a cohort-wide freedom dashboard.'
        ),
        'engel_classes': [
            {'class': 'IA', 'label': 'Completely seizure-free; no auras', 'tier': 'Excellent', 'color': 'success'},
            {'class': 'IB', 'label': 'Only auras; no disabling seizures', 'tier': 'Excellent', 'color': 'success'},
            {'class': 'IC', 'label': 'Some disabling seizures after surgery; seizure-free ≥2 years', 'tier': 'Good', 'color': 'primary'},
            {'class': 'ID', 'label': 'Generalized convulsions with AED withdrawal only', 'tier': 'Good', 'color': 'primary'},
            {'class': 'IIA', 'label': 'Initially seizure-free but not ≥2 years', 'tier': 'Good', 'color': 'info'},
            {'class': 'IIB', 'label': 'Rare disabling seizures since surgery', 'tier': 'Good', 'color': 'info'},
            {'class': 'IIC', 'label': 'More than rare disabling seizures but marked improvement', 'tier': 'Fair', 'color': 'warning'},
            {'class': 'IID', 'label': 'Auras only (not seizure-free)', 'tier': 'Fair', 'color': 'warning'},
            {'class': 'IIIA', 'label': 'Worthwhile improvement', 'tier': 'Fair', 'color': 'warning'},
            {'class': 'IIIB', 'label': 'Prolonged auras only (not seizure-free)', 'tier': 'Fair', 'color': 'warning'},
            {'class': 'IVA', 'label': 'No worthwhile improvement', 'tier': 'Poor', 'color': 'danger'},
            {'class': 'IVB', 'label': 'No worthwhile improvement; worse', 'tier': 'Poor', 'color': 'danger'},
        ],
        'ilae_outcomes': [
            {'value': 1, 'label': 'Completely seizure-free; no AEDs', 'color': 'success'},
            {'value': 2, 'label': '≥90% reduction in seizure days', 'color': 'primary'},
            {'value': 3, 'label': '50–90% reduction in seizure days', 'color': 'info'},
            {'value': 4, 'label': '<50% reduction in seizure days', 'color': 'warning'},
            {'value': 5, 'label': 'No worthwhile change or worse', 'color': 'danger'},
            {'value': 6, 'label': 'Unable to assess (lost to follow-up)', 'color': 'secondary'},
        ],
        'milestones': [
            {'months': 6, 'milestone': 'Return to activities/work',
             'description': 'After 6 consecutive seizure-free months, many patients may resume restricted activities. Employer and insurer notification often required.'},
            {'months': 12, 'milestone': 'Driving eligibility (most jurisdictions)',
             'description': 'Canada, USA (most states), UK, and Australia require ≥12 months seizure-free before driving privileges may be restored. Jurisdictional variation applies.'},
            {'months': 24, 'milestone': 'Reduced monitoring frequency',
             'description': 'After 24 seizure-free months, annual (rather than bi-annual) neurology follow-up may be appropriate. AED tapering discussions may begin.'},
            {'months': 36, 'milestone': 'AED discontinuation consideration',
             'description': 'For selected patients (adults, normal MRI, single seizure type, Engel IA), AED discontinuation may be considered after ≥3 seizure-free years per ILAE guidelines.'},
        ],
        'drug_responsiveness': [
            {'label': 'Drug-responsive', 'description': 'Seizure-free on first or second AED. Best prognosis; ~65% of all newly diagnosed epilepsy.'},
            {'label': 'Partial response — reduced frequency', 'description': 'Meaningful reduction but not seizure-free. May benefit from add-on therapy, dosage adjustment, or alternative AEDs.'},
            {'label': 'Drug-resistant (failed ≥2 AEDs)', 'description': 'Failure of ≥2 appropriate AEDs at adequate doses. ILAE definition. Surgical evaluation strongly recommended.'},
            {'label': 'Newly diagnosed — response pending', 'description': 'Insufficient follow-up to classify drug responsiveness. Reassess at 6–12 months.'},
        ],
        'data_sources': [
            'surgical_outcomes — 28 procedures, 25 unique patients, follow-up 6–59 months',
            'seizure_trigger_logs — 203 log days, 40 patients',
            'seizure_metadata — 71 patients with syndrome + drug-responsiveness profiles',
        ],
        'clinical_reference': (
            'Engel Classification: Engel J et al., Neurology 1993. '
            'ILAE Outcome Scale: Wieser HG et al., Epilepsia 2001. '
            'Driving regulations per national epilepsy society guidelines (Canada: CMA 2017; '
            'USA: AAN 2007; UK: DVLA 2023).'
        ),
    }


if __name__ == '__main__':
    import json
    print(json.dumps(freedom_overview(), indent=2))
