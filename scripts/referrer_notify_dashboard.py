"""Referrer Notification Dashboard — post-report referrer notification tracking.

After a neurologist signs off on an EEG report, the referring clinician
(GP, pediatrician, psychiatry, etc.) must be notified of findings.
This dashboard tracks the notification queue, delivery status, and
source-level performance from the referral_records table.

Notification status is derived from triage_status:
  completed  → notified     (report done, referrer informed)
  triaged    → queued       (triaged and ready — notification pending)
  scheduled  → in_progress  (active workup — will notify when complete)
  in_progress→ in_progress
  pending_triage → not_ready (too early to notify)
  cancelled  → cancelled

Sources:
- referral_records table (84 records, 41 patients, 7 sources, 4 urgency levels)
"""

import sqlite3
import os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

_STATUS_MAP = {
    'completed':      'notified',
    'triaged':        'queued',
    'scheduled':      'in_progress',
    'in_progress':    'in_progress',
    'pending_triage': 'not_ready',
    'cancelled':      'cancelled',
}

_STATUS_LABEL = {
    'notified':    'Notified',
    'queued':      'Queued',
    'in_progress': 'In Progress',
    'not_ready':   'Not Ready',
    'cancelled':   'Cancelled',
}

_STATUS_COLOR = {
    'notified':    'success',
    'queued':      'warning',
    'in_progress': 'info',
    'not_ready':   'secondary',
    'cancelled':   'danger',
}

_SOURCE_LABEL = {
    'emergency':        'Emergency',
    'neurology_clinic': 'Neurology Clinic',
    'primary_care':     'Primary Care / GP',
    'other_specialist': 'Other Specialist',
    'pediatrics':       'Pediatrics',
    'self_referral':    'Self-Referral',
    'psychiatry':       'Psychiatry',
}

_URGENCY_COLOR = {
    'emergent': 'danger',
    'urgent':   'warning',
    'routine':  'primary',
    'elective': 'secondary',
}


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
#  /api/referrer-notify/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """KPIs: notification queue depth, notified count, source breakdown,
    urgency-weighted queue, notification rate, monthly referral trend."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total = _safe(cur, "SELECT COUNT(*) FROM referral_records")
    patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM referral_records")

    # Count per triage_status → derive notify status
    status_rows = _safe_rows(cur,
        "SELECT triage_status, COUNT(*) FROM referral_records GROUP BY triage_status")
    status_counts = {r[0]: r[1] for r in status_rows}

    notified    = status_counts.get('completed', 0)
    queued      = status_counts.get('triaged', 0)
    in_progress = status_counts.get('scheduled', 0) + status_counts.get('in_progress', 0)
    not_ready   = status_counts.get('pending_triage', 0)
    cancelled   = status_counts.get('cancelled', 0)

    notify_rate = round(notified / total * 100, 1) if total else 0

    # Urgency × notification status
    urgency_rows = _safe_rows(cur,
        """SELECT urgency, triage_status, COUNT(*) FROM referral_records
           GROUP BY urgency, triage_status""")
    urgency_map = {}
    for urg, ts, cnt in urgency_rows:
        ns = _STATUS_MAP.get(ts, 'unknown')
        urgency_map.setdefault(urg, {}).setdefault(ns, 0)
        urgency_map[urg][ns] += cnt

    urgency_summary = []
    for urg in ['emergent', 'urgent', 'routine', 'elective']:
        m = urgency_map.get(urg, {})
        total_urg = sum(m.values())
        urgency_summary.append({
            'urgency':      urg,
            'total':        total_urg,
            'notified':     m.get('notified', 0),
            'queued':       m.get('queued', 0),
            'in_progress':  m.get('in_progress', 0),
            'not_ready':    m.get('not_ready', 0),
            'cancelled':    m.get('cancelled', 0),
            'color':        _URGENCY_COLOR.get(urg, 'info'),
        })

    # Monthly referral + notification trend
    monthly_rows = _safe_rows(cur,
        """SELECT strftime('%Y-%m', referral_date) as mo,
                  COUNT(*) as cnt,
                  SUM(CASE WHEN triage_status='completed' THEN 1 ELSE 0 END) as notif
           FROM referral_records GROUP BY mo ORDER BY mo""")
    monthly_trend = [
        {'month': r[0], 'referrals': r[1], 'notified': r[2]}
        for r in monthly_rows
    ]

    # Source breakdown (for pie)
    source_rows = _safe_rows(cur,
        "SELECT referral_source, COUNT(*) FROM referral_records GROUP BY referral_source ORDER BY 2 DESC")
    source_summary = [
        {'source': r[0], 'label': _SOURCE_LABEL.get(r[0], r[0].title()), 'count': r[1],
         'pct': round(r[1] / total * 100, 1) if total else 0}
        for r in source_rows
    ]

    conn.close()
    return {
        'kpis': {
            'total_referrals':  total,
            'total_patients':   patients,
            'notified':         notified,
            'queued':           queued,
            'in_progress':      in_progress,
            'not_ready':        not_ready,
            'cancelled':        cancelled,
            'notify_rate_pct':  notify_rate,
        },
        'urgency_summary': urgency_summary,
        'monthly_trend':   monthly_trend,
        'source_summary':  source_summary,
    }


# ──────────────────────────────────────────────────────────────
#  /api/referrer-notify/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-source notification stats, notification queue (queued/in_progress),
    per-patient summary, and recent completed (notified) referrals."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total = _safe(cur, "SELECT COUNT(*) FROM referral_records") or 1

    # Per-source: total, notified, queued, avg triage_score
    source_rows = _safe_rows(cur,
        """SELECT referral_source,
                  COUNT(*) as total,
                  SUM(CASE WHEN triage_status='completed' THEN 1 ELSE 0 END) as notified,
                  SUM(CASE WHEN triage_status='triaged'   THEN 1 ELSE 0 END) as queued,
                  ROUND(AVG(triage_score), 1) as avg_score
           FROM referral_records GROUP BY referral_source ORDER BY total DESC""")
    per_source = []
    for r in source_rows:
        src, tot, notif, queued, avg_score = r
        per_source.append({
            'source':         src,
            'label':          _SOURCE_LABEL.get(src, src.title()),
            'total':          tot,
            'notified':       notif or 0,
            'queued':         queued or 0,
            'notify_rate':    round((notif or 0) / tot * 100, 1) if tot else 0,
            'avg_triage_score': avg_score or 0,
        })

    # Notification queue (triaged = queued for notification)
    queue_rows = _safe_rows(cur,
        """SELECT id, patient_id, referral_source, referral_reason, urgency, triage_score,
                  assigned_to, referral_date, triage_date
           FROM referral_records
           WHERE triage_status IN ('triaged', 'in_progress', 'scheduled')
           ORDER BY
             CASE urgency WHEN 'emergent' THEN 0 WHEN 'urgent' THEN 1
                          WHEN 'routine'  THEN 2 ELSE 3 END,
             triage_score DESC
           LIMIT 20""")
    queue = [
        {
            'id':              r[0],
            'patient_id':      r[1],
            'source':          r[2],
            'source_label':    _SOURCE_LABEL.get(r[2], r[2].title()),
            'reason':          r[3],
            'urgency':         r[4],
            'urgency_color':   _URGENCY_COLOR.get(r[4], 'secondary'),
            'triage_score':    r[5],
            'assigned_to':     r[6] or 'Unassigned',
            'referral_date':   r[7],
            'triage_date':     r[8],
            'notify_status':   'queued' if r[9 - 4] else 'in_progress',  # placeholder
        }
        for r in queue_rows
    ]
    # Correct notify_status based on triage_status lookup
    ts_rows = _safe_rows(cur,
        """SELECT id, triage_status FROM referral_records
           WHERE triage_status IN ('triaged','in_progress','scheduled')""")
    ts_map = {r[0]: r[1] for r in ts_rows}
    for item in queue:
        item['notify_status'] = _STATUS_MAP.get(ts_map.get(item['id'], ''), 'in_progress')

    # Per-patient summary
    patient_rows = _safe_rows(cur,
        """SELECT patient_id,
                  COUNT(*) as total,
                  SUM(CASE WHEN triage_status='completed' THEN 1 ELSE 0 END) as notified,
                  SUM(CASE WHEN triage_status='triaged'   THEN 1 ELSE 0 END) as queued,
                  MAX(triage_score) as max_score
           FROM referral_records GROUP BY patient_id ORDER BY total DESC""")
    per_patient = [
        {
            'patient_id':   r[0],
            'total':        r[1],
            'notified':     r[2] or 0,
            'queued':       r[3] or 0,
            'pending':      r[1] - (r[2] or 0) - _safe(cur,
                "SELECT COUNT(*) FROM referral_records WHERE patient_id=? AND triage_status='cancelled'",
                (r[0],)),
            'max_score':    r[4],
        }
        for r in patient_rows
    ]

    # Recently notified (completed)
    recent_rows = _safe_rows(cur,
        """SELECT id, patient_id, referral_source, referral_reason, urgency,
                  assigned_to, triage_date, triage_score
           FROM referral_records WHERE triage_status='completed'
           ORDER BY triage_date DESC LIMIT 10""")
    recently_notified = [
        {
            'id':            r[0],
            'patient_id':    r[1],
            'source':        r[2],
            'source_label':  _SOURCE_LABEL.get(r[2], r[2].title()),
            'reason':        r[3],
            'urgency':       r[4],
            'assigned_to':   r[5] or 'Unassigned',
            'notified_date': r[6],
            'triage_score':  r[7],
        }
        for r in recent_rows
    ]

    conn.close()
    return {
        'per_source':       per_source,
        'queue':            queue,
        'per_patient':      per_patient,
        'recently_notified': recently_notified,
    }


# ──────────────────────────────────────────────────────────────
#  /api/referrer-notify/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Notification status definitions, workflow description, referral source glossary,
    urgency tiers, triage score explanation."""
    return {
        'title': 'Referrer Notification — Definitions & Workflow',
        'workflow': {
            'description': (
                'After an EEG report is signed off by the neurologist, '
                'the referring clinician must be notified of findings and '
                'recommended next steps. This dashboard tracks the notification '
                'queue, delivery history, and source-level performance.'
            ),
            'steps': [
                '1. Referral received → triage scored',
                '2. EEG acquisition & AI pre-read',
                '3. Neurologist review → sign-off',
                '4. Report generated (Structured Reporting Dashboard)',
                '5. ✉ Referrer notification dispatched (this dashboard)',
                '6. Confirmation logged → referral marked completed',
            ],
        },
        'notification_statuses': [
            {'status': 'notified',    'color': 'success',   'meaning': 'Report delivered to referring clinician'},
            {'status': 'queued',      'color': 'warning',   'meaning': 'Triage complete — notification pending dispatch'},
            {'status': 'in_progress', 'color': 'info',      'meaning': 'Workup active — notification will follow completion'},
            {'status': 'not_ready',   'color': 'secondary', 'meaning': 'Pending triage — too early to notify'},
            {'status': 'cancelled',   'color': 'danger',    'meaning': 'Referral cancelled — no notification needed'},
        ],
        'referral_sources': [
            {'id': 'emergency',        'label': 'Emergency',           'note': 'ER/urgent care acute seizure presentation'},
            {'id': 'neurology_clinic', 'label': 'Neurology Clinic',    'note': 'Internal neurology self-referral'},
            {'id': 'primary_care',     'label': 'Primary Care / GP',   'note': 'Family physician / general practitioner'},
            {'id': 'other_specialist', 'label': 'Other Specialist',    'note': 'Cardiology, endocrinology, sleep medicine, etc.'},
            {'id': 'pediatrics',       'label': 'Pediatrics',          'note': 'Pediatric neurology or general pediatrics'},
            {'id': 'self_referral',    'label': 'Self-Referral',       'note': 'Patient-initiated — notify GP as a courtesy'},
            {'id': 'psychiatry',       'label': 'Psychiatry',          'note': 'Psychiatric co-management referrals'},
        ],
        'urgency_tiers': [
            {'tier': 'emergent', 'sla': '< 2 h',   'color': 'danger',    'description': 'Active seizure / status epilepticus'},
            {'tier': 'urgent',   'sla': '< 24 h',  'color': 'warning',   'description': 'Recent unprovoked seizure, high risk'},
            {'tier': 'routine',  'sla': '< 5 days','color': 'primary',   'description': 'Established epilepsy follow-up'},
            {'tier': 'elective', 'sla': '< 14 days','color': 'secondary','description': 'Non-urgent screening / second opinion'},
        ],
        'triage_score': {
            'range': '0–100',
            'description': (
                'Composite clinical priority score: urgency, reason severity, '
                'patient age, and comorbidity burden. Higher score = higher priority.'
            ),
        },
        'channel_note': (
            'Notification channel (email/fax/portal) requires SMTP or FHIR credentials. '
            'This dashboard tracks queue and delivery status; actual dispatch is a Stage-2 integration.'
        ),
    }
