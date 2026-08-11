"""Insurance Pre-Authorization Dashboard — prior-auth workflow analytics.

Models prior-authorization requests from billing_claims, mapping service
types and claim statuses into a pre-auth pipeline view for the Admin /
Compliance / Billing team.

Pre-auth classification logic:
  Service types requiring prior auth: pre_surgical_eval, neuropsych_assessment,
  eeg_recording, emergency (high-cost), consultation (specialist first visit).
  Status mapping from billing_claims:
    submitted          → pending_review
    approved / paid    → approved
    denied             → denied
    partially_approved → conditionally_approved
    appealed           → under_appeal
    write_off          → withdrawn

Sources:
  billing_claims  — 150 claims: amount, payer, service type, status, dates
  patients        — 41 patients: demographics, disease, department
"""

import sqlite3
import os
from datetime import datetime
from collections import Counter, defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

# Service types that require prior authorization in epilepsy neurology
PREAUTH_SERVICES = {
    'pre_surgical_eval':    {'label': 'Pre-Surgical Evaluation',      'priority': 'high'},
    'neuropsych_assessment': {'label': 'Neuropsychological Assessment', 'priority': 'high'},
    'eeg_recording':        {'label': 'EEG Recording',                 'priority': 'medium'},
    'emergency':            {'label': 'Emergency Services',            'priority': 'high'},
    'consultation':         {'label': 'Specialist Consultation',       'priority': 'medium'},
    'medication_review':    {'label': 'Complex Medication Review',     'priority': 'low'},
}

STATUS_MAP = {
    'submitted':          'pending_review',
    'approved':           'approved',
    'paid':               'approved',
    'denied':             'denied',
    'partially_approved': 'conditionally_approved',
    'appealed':           'under_appeal',
    'write_off':          'withdrawn',
}

STATUS_LABEL = {
    'pending_review':        'Pending Review',
    'approved':              'Approved',
    'denied':                'Denied',
    'conditionally_approved': 'Conditionally Approved',
    'under_appeal':          'Under Appeal',
    'withdrawn':             'Withdrawn',
}


def _query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _days_between(d1_str, d2_str):
    """Return integer days between two ISO datetime strings, or None."""
    try:
        fmt = '%Y-%m-%d %H:%M:%S'
        d1 = datetime.strptime(str(d1_str)[:19], fmt)
        d2 = datetime.strptime(str(d2_str)[:19], fmt)
        return max(0, (d2 - d1).days)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI summary for the Insurance Pre-Authorization dashboard."""
    claims = _query(
        "SELECT bc.*, p.name AS patient_name, p.disease, p.age, p.gender "
        "FROM billing_claims bc "
        "LEFT JOIN patients p ON bc.patient_id = p.patient_id "
        "ORDER BY bc.submitted_at"
    )
    if not claims:
        return {'available': False, 'message': 'No billing claims data found.'}

    # Filter to pre-auth-requiring services
    preauth_claims = [c for c in claims if c.get('service_type') in PREAUTH_SERVICES]
    total_preauth = len(preauth_claims)

    if total_preauth == 0:
        return {'available': False, 'message': 'No pre-authorization-requiring claims found.'}

    # Map to pre-auth statuses
    pa_statuses = [STATUS_MAP.get(c.get('status', ''), 'pending_review') for c in preauth_claims]
    status_counts = Counter(pa_statuses)

    approved_n = status_counts.get('approved', 0) + status_counts.get('conditionally_approved', 0)
    denied_n   = status_counts.get('denied', 0)
    pending_n  = status_counts.get('pending_review', 0)
    appeal_n   = status_counts.get('under_appeal', 0)

    approval_rate = round(approved_n / total_preauth * 100, 1) if total_preauth else 0.0
    denial_rate   = round(denied_n  / total_preauth * 100, 1) if total_preauth else 0.0

    # Days-to-decision for resolved claims
    decision_days = []
    for c in preauth_claims:
        if c.get('adjudicated_at') and c.get('submitted_at'):
            d = _days_between(c['submitted_at'], c['adjudicated_at'])
            if d is not None:
                decision_days.append(d)
    avg_decision_days = round(sum(decision_days) / len(decision_days), 1) if decision_days else None

    # Payer breakdown
    payer_counts = Counter(c.get('insurance_provider', 'Unknown') for c in preauth_claims)
    payer_approval = defaultdict(lambda: {'total': 0, 'approved': 0})
    for c in preauth_claims:
        payer = c.get('insurance_provider', 'Unknown')
        pa_s  = STATUS_MAP.get(c.get('status', ''), 'pending_review')
        payer_approval[payer]['total'] += 1
        if pa_s in ('approved', 'conditionally_approved'):
            payer_approval[payer]['approved'] += 1

    payer_summary = sorted([
        {
            'payer':         payer,
            'total':         v['total'],
            'approved':      v['approved'],
            'approval_rate': round(v['approved'] / v['total'] * 100, 1) if v['total'] else 0.0,
        }
        for payer, v in payer_approval.items()
    ], key=lambda x: -x['total'])

    # Service type breakdown
    service_counts = Counter(c.get('service_type') for c in preauth_claims)
    service_summary = [
        {
            'service_type': st,
            'label':        PREAUTH_SERVICES[st]['label'],
            'priority':     PREAUTH_SERVICES[st]['priority'],
            'count':        service_counts.get(st, 0),
        }
        for st in PREAUTH_SERVICES
        if service_counts.get(st, 0) > 0
    ]

    # Top denial reasons
    denial_claims = [c for c in preauth_claims if STATUS_MAP.get(c.get('status','')) == 'denied']
    denial_reason_counts = Counter(
        c.get('denial_reason') or 'Not specified' for c in denial_claims
    )
    top_denial_reasons = [
        {'reason': r, 'count': n}
        for r, n in denial_reason_counts.most_common(5)
    ]

    # Unique patients with pre-auth claims
    unique_patients = len(set(c.get('patient_id') for c in preauth_claims))

    # Financial impact
    total_billed   = sum(float(c.get('amount_billed') or 0)   for c in preauth_claims)
    total_approved_amt = sum(float(c.get('amount_approved') or 0) for c in preauth_claims)

    return {
        'available': True,
        'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        'summary': {
            'total_preauth_requests': total_preauth,
            'unique_patients':        unique_patients,
            'approval_rate_pct':      approval_rate,
            'denial_rate_pct':        denial_rate,
            'pending_count':          pending_n,
            'under_appeal_count':     appeal_n,
            'avg_decision_days':      avg_decision_days,
            'total_billed':           round(total_billed, 2),
            'total_approved_amt':     round(total_approved_amt, 2),
        },
        'status_distribution': [
            {'status': STATUS_LABEL.get(s, s), 'count': n}
            for s, n in status_counts.most_common()
        ],
        'payer_summary':     payer_summary,
        'service_summary':   service_summary,
        'top_denial_reasons': top_denial_reasons,
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed per-payer, per-service, per-patient pre-auth breakdown."""
    claims = _query(
        "SELECT bc.*, p.name AS patient_name, p.disease, p.age, p.gender "
        "FROM billing_claims bc "
        "LEFT JOIN patients p ON bc.patient_id = p.patient_id "
        "ORDER BY bc.submitted_at DESC"
    )
    if not claims:
        return {'available': False, 'message': 'No billing claims data found.'}

    preauth_claims = [c for c in claims if c.get('service_type') in PREAUTH_SERVICES]

    # Per-payer detail
    payer_detail = defaultdict(lambda: {
        'total': 0, 'approved': 0, 'denied': 0, 'pending': 0,
        'appeal': 0, 'total_billed': 0.0, 'total_approved_amt': 0.0,
        'decision_days': []
    })
    for c in preauth_claims:
        payer  = c.get('insurance_provider', 'Unknown')
        pa_s   = STATUS_MAP.get(c.get('status', ''), 'pending_review')
        payer_detail[payer]['total'] += 1
        payer_detail[payer]['total_billed'] += float(c.get('amount_billed') or 0)
        payer_detail[payer]['total_approved_amt'] += float(c.get('amount_approved') or 0)
        if pa_s in ('approved', 'conditionally_approved'):
            payer_detail[payer]['approved'] += 1
        elif pa_s == 'denied':
            payer_detail[payer]['denied'] += 1
        elif pa_s == 'pending_review':
            payer_detail[payer]['pending'] += 1
        elif pa_s == 'under_appeal':
            payer_detail[payer]['appeal'] += 1
        if c.get('adjudicated_at') and c.get('submitted_at'):
            d = _days_between(c['submitted_at'], c['adjudicated_at'])
            if d is not None:
                payer_detail[payer]['decision_days'].append(d)

    payer_rows = []
    for payer, v in sorted(payer_detail.items(), key=lambda x: -x[1]['total']):
        days = v['decision_days']
        payer_rows.append({
            'payer':           payer,
            'total':           v['total'],
            'approved':        v['approved'],
            'denied':          v['denied'],
            'pending':         v['pending'],
            'under_appeal':    v['appeal'],
            'approval_rate':   round(v['approved'] / v['total'] * 100, 1) if v['total'] else 0.0,
            'avg_decision_days': round(sum(days) / len(days), 1) if days else None,
            'total_billed':    round(v['total_billed'], 2),
            'total_approved_amt': round(v['total_approved_amt'], 2),
        })

    # Per-service type detail
    service_detail = defaultdict(lambda: {
        'total': 0, 'approved': 0, 'denied': 0, 'pending': 0
    })
    for c in preauth_claims:
        st  = c.get('service_type', 'unknown')
        pa_s = STATUS_MAP.get(c.get('status', ''), 'pending_review')
        service_detail[st]['total'] += 1
        if pa_s in ('approved', 'conditionally_approved'):
            service_detail[st]['approved'] += 1
        elif pa_s == 'denied':
            service_detail[st]['denied'] += 1
        elif pa_s == 'pending_review':
            service_detail[st]['pending'] += 1

    service_rows = [
        {
            'service_type':  st,
            'label':         PREAUTH_SERVICES.get(st, {}).get('label', st),
            'priority':      PREAUTH_SERVICES.get(st, {}).get('priority', 'medium'),
            'total':         v['total'],
            'approved':      v['approved'],
            'denied':        v['denied'],
            'pending':       v['pending'],
            'approval_rate': round(v['approved'] / v['total'] * 100, 1) if v['total'] else 0.0,
        }
        for st, v in sorted(service_detail.items(), key=lambda x: -x[1]['total'])
    ]

    # Per-patient recent pre-auth history (most recent 40 claims)
    patient_rows = []
    for c in preauth_claims[:40]:
        pa_s = STATUS_MAP.get(c.get('status', ''), 'pending_review')
        dd   = None
        if c.get('adjudicated_at') and c.get('submitted_at'):
            dd = _days_between(c['submitted_at'], c['adjudicated_at'])
        patient_rows.append({
            'claim_id':       c.get('claim_id'),
            'patient_id':     c.get('patient_id'),
            'patient_name':   c.get('patient_name') or c.get('patient_id'),
            'disease':        c.get('disease'),
            'service_type':   PREAUTH_SERVICES.get(c.get('service_type',''), {}).get('label', c.get('service_type','')),
            'payer':          c.get('insurance_provider', 'Unknown'),
            'preauth_status': STATUS_LABEL.get(pa_s, pa_s),
            'amount_billed':  round(float(c.get('amount_billed') or 0), 2),
            'decision_days':  dd,
            'submitted_at':   (c.get('submitted_at') or '')[:10],
            'adjudicated_at': (c.get('adjudicated_at') or '')[:10] or None,
            'denial_reason':  c.get('denial_reason'),
        })

    # Denial reason breakdown
    denial_claims = [c for c in preauth_claims if STATUS_MAP.get(c.get('status','')) == 'denied']
    denial_rows = sorted([
        {'reason': r, 'count': n, 'pct': round(n / len(denial_claims) * 100, 1)}
        for r, n in Counter(
            c.get('denial_reason') or 'Not specified' for c in denial_claims
        ).most_common()
    ], key=lambda x: -x['count'])

    return {
        'available':     True,
        'payer_detail':  payer_rows,
        'service_detail': service_rows,
        'recent_preauths': patient_rows,
        'denial_breakdown': denial_rows,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Glossary, CPT pre-auth requirements, payer policies, clinical standards."""
    return {
        'available': True,
        'title': 'Insurance Pre-Authorization — Definitions & Reference',
        'what_is_prior_auth': (
            'Prior authorization (PA) is a requirement by health insurers that '
            'clinicians obtain approval before delivering certain services or '
            'prescribing specific medications. In epilepsy neurology, PA applies '
            'to high-cost procedures, specialist evaluations, and long-term EEG monitoring.'
        ),
        'preauth_statuses': [
            {'status': 'Pending Review',          'description': 'Submitted; awaiting payer decision (typically 3–14 days).'},
            {'status': 'Approved',                'description': 'Payer granted authorization; service may proceed.'},
            {'status': 'Conditionally Approved',  'description': 'Approved with restrictions (e.g., limited sessions, specific facility).'},
            {'status': 'Denied',                  'description': 'Authorization refused; appeal rights apply within 60–180 days.'},
            {'status': 'Under Appeal',            'description': 'Denial contested; additional clinical documentation submitted.'},
            {'status': 'Withdrawn',               'description': 'Claim or PA request retracted (write-off or non-covered service).'},
        ],
        'services_requiring_preauth': [
            {
                'service':    'Pre-Surgical Evaluation',
                'cpt_codes':  ['95957', '99215'],
                'priority':   'High — always requires PA',
                'typical_tat': '5–14 business days',
                'notes':      'Comprehensive Phase I/II workup for epilepsy surgery candidacy.',
            },
            {
                'service':    'Neuropsychological Assessment',
                'cpt_codes':  ['96132', '96133'],
                'priority':   'High — PA required by most payers',
                'typical_tat': '5–10 business days',
                'notes':      'Pre-surgical baseline and post-operative cognitive tracking.',
            },
            {
                'service':    'EEG Recording (long-term / video-EEG)',
                'cpt_codes':  ['95950', '95953', '95956'],
                'priority':   'Medium — PA required for inpatient LTM',
                'typical_tat': '3–7 business days',
                'notes':      'Routine EEG (95816/95819) typically exempt; LTM requires PA.',
            },
            {
                'service':    'Emergency Services',
                'cpt_codes':  ['99285', '99291'],
                'priority':   'Retrospective — notify within 24–48 h',
                'typical_tat': 'Retroactive notification 24–48 h',
                'notes':      'Emergency admissions are exempted from pre-auth but require concurrent review.',
            },
            {
                'service':    'Specialist Consultation (first visit)',
                'cpt_codes':  ['99245', '99244'],
                'priority':   'Medium — varies by plan',
                'typical_tat': '3–5 business days',
                'notes':      'Some Medicare Advantage and commercial plans require PA for neurology referrals.',
            },
            {
                'service':    'Complex Medication Review (AED titration)',
                'cpt_codes':  ['99213', '99214'],
                'priority':   'Low — most plans exempt routine E/M',
                'typical_tat': 'N/A (usually no PA needed)',
                'notes':      'Specialized AED prescriptions (e.g., Epidiolex) may require separate drug PA.',
            },
        ],
        'payer_policies': [
            {'payer': 'Medicare',             'pa_required': 'Limited — LTM EEG, surgery eval', 'turnaround': '14 days standard / 3 days urgent'},
            {'payer': 'Medicaid',             'pa_required': 'Broad — most specialist visits',   'turnaround': 'Varies by state (5–14 days)'},
            {'payer': 'BlueCross BlueShield', 'pa_required': 'Moderate — specialist + imaging',  'turnaround': '5–7 business days'},
            {'payer': 'Aetna',                'pa_required': 'Moderate — neurology services',     'turnaround': '3–5 business days'},
            {'payer': 'UnitedHealthcare',     'pa_required': 'Extensive — LTM + neuropsych',     'turnaround': '3–7 business days'},
            {'payer': 'Cigna',                'pa_required': 'Moderate — surgical workup',       'turnaround': '5–10 business days'},
            {'payer': 'Self-Pay',             'pa_required': 'N/A',                               'turnaround': 'N/A'},
        ],
        'common_denial_reasons': [
            {'reason': 'Prior authorization not obtained',    'action': 'Retrospective PA request; provide clinical urgency justification.'},
            {'reason': 'Medical necessity not established',   'action': 'Submit EEG reports, seizure diary, ICD-10 specificity; attending letter.'},
            {'reason': 'Service not covered under plan',      'action': 'Review SBC/EOC; escalate to plan medical director if clinically indicated.'},
            {'reason': 'Incomplete documentation',            'action': 'Resubmit with clinical notes, imaging reports, and prior treatment history.'},
            {'reason': 'Out-of-network provider',             'action': 'Request single-case agreement or in-network referral pathway.'},
            {'reason': 'Coordination of benefits required',   'action': 'Verify primary/secondary payer order; resubmit to correct primary payer.'},
            {'reason': 'Patient eligibility expired',         'action': 'Confirm enrollment dates; check retroactive coverage reinstatement.'},
        ],
        'regulatory_references': [
            'CMS Medicare Benefit Policy Manual — Chapter 15 (Covered Medical Services)',
            'AMA CPT® Code Set (2026) — Neurology & EEG codes 95812–95957',
            'ILAE 2017 Classification of Seizures and Epilepsies',
            'CMS Prior Authorization Transparency Act (PATA) regulations',
            'HIPAA Administrative Simplification — ASC X12 278 PA transaction',
        ],
    }
