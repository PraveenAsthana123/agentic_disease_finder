"""Billing & Claims Dashboard — insurance billing analytics from clinical.db.

Tracks claim submissions, approval/denial rates, collection metrics,
payer mix, CPT/ICD-10 code distribution, and revenue cycle trends.

Sources:
- billing_claims table (claim_id, patient_id, service_type, amounts, status, codes)
- patients table (demographics for cross-referencing)
- appointments table (linked appointments for correlation)
- transaction_log (billing-related actions for audit trail)
"""

import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta
import random

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


def _ensure_table():
    """Create billing_claims table and seed realistic data if it doesn't exist."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='billing_claims'
    """)
    if cur.fetchone():
        conn.close()
        return

    cur.execute("""
        CREATE TABLE billing_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            claim_id TEXT UNIQUE NOT NULL,
            appointment_id INTEGER,
            provider TEXT NOT NULL,
            department TEXT NOT NULL,
            service_type TEXT NOT NULL,
            icd10_code TEXT,
            icd10_desc TEXT,
            cpt_code TEXT,
            cpt_desc TEXT,
            amount_billed REAL NOT NULL,
            amount_approved REAL,
            amount_paid REAL,
            amount_patient_owes REAL,
            insurance_provider TEXT,
            policy_number TEXT,
            status TEXT NOT NULL DEFAULT 'submitted',
            submitted_at TEXT NOT NULL,
            adjudicated_at TEXT,
            paid_at TEXT,
            denial_reason TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Seed realistic billing data from existing patients
    cur.execute("SELECT patient_id, name, disease, department FROM patients")
    patients = cur.fetchall()
    if not patients:
        conn.close()
        return

    # Try to get existing appointment IDs for linking
    appt_ids = []
    try:
        cur.execute("SELECT id FROM appointments")
        appt_ids = [r[0] for r in cur.fetchall()]
    except Exception:
        pass

    providers = [
        ('Dr. Sharma', 'Neurology'),
        ('Dr. Patel', 'Neurology'),
        ('Dr. Singh', 'Psychiatry'),
        ('Dr. Gupta', 'Neuropsychology'),
        ('Dr. Mehta', 'Neurology'),
        ('Dr. Kaur', 'EEG Lab'),
    ]

    service_types = [
        'eeg_recording', 'consultation', 'neuropsych_assessment',
        'medication_review', 'pre_surgical_eval', 'telemed',
        'emergency', 'follow_up'
    ]

    icd10_codes = [
        ('G40.001', 'Localization-related idiopathic epilepsy, not intractable, with status epilepticus'),
        ('G40.009', 'Localization-related idiopathic epilepsy, not intractable, without status epilepticus'),
        ('G40.101', 'Localization-related symptomatic epilepsy, not intractable, with status epilepticus'),
        ('G40.201', 'Absence epileptic syndrome, not intractable, with status epilepticus'),
        ('G40.301', 'Generalized idiopathic epilepsy, not intractable, with status epilepticus'),
        ('G40.409', 'Other generalized epilepsy, not intractable, without status epilepticus'),
        ('G40.501', 'Epileptic seizures related to external causes, not intractable, with status epilepticus'),
        ('G40.801', 'Other epilepsy, not intractable, with status epilepticus'),
        ('G40.911', 'Epilepsy, unspecified, intractable, with status epilepticus'),
        ('G40.919', 'Epilepsy, unspecified, not intractable'),
        ('F32.0', 'Major depressive disorder, single episode, mild'),
        ('F32.1', 'Major depressive disorder, single episode, moderate'),
        ('F32.2', 'Major depressive disorder, single episode, severe without psychotic features'),
        ('F32.9', 'Major depressive disorder, single episode, unspecified'),
        ('F20.0', 'Paranoid schizophrenia'),
        ('F20.1', 'Disorganized schizophrenia'),
        ('F20.9', 'Schizophrenia, unspecified'),
        ('R56.9', 'Unspecified convulsions / Seizures NOS'),
        ('G43.001', 'Migraine without aura, not intractable, with status migrainosus'),
        ('G43.109', 'Migraine with aura, not intractable, without status migrainosus'),
        ('G43.909', 'Migraine, unspecified, not intractable, without status migrainosus'),
    ]

    cpt_codes = [
        ('95816', 'EEG awake and drowsy'),
        ('95819', 'EEG awake and sleep'),
        ('95957', 'Digital EEG analysis'),
        ('99213', 'Office/outpatient visit, established, low complexity'),
        ('99214', 'Office/outpatient visit, established, moderate complexity'),
        ('99215', 'Office/outpatient visit, established, high complexity'),
        ('96132', 'Neuropsychological testing evaluation, first hour'),
        ('95950', 'Ambulatory continuous EEG monitoring'),
    ]

    # Map service types to typical CPT codes and amount ranges
    service_cpt_map = {
        'eeg_recording': [('95816', 'EEG awake and drowsy'), ('95819', 'EEG awake and sleep')],
        'consultation': [('99214', 'Office/outpatient visit, established, moderate complexity'),
                         ('99215', 'Office/outpatient visit, established, high complexity')],
        'neuropsych_assessment': [('96132', 'Neuropsychological testing evaluation, first hour')],
        'medication_review': [('99213', 'Office/outpatient visit, established, low complexity')],
        'pre_surgical_eval': [('95957', 'Digital EEG analysis'), ('99215', 'Office/outpatient visit, established, high complexity')],
        'telemed': [('99213', 'Office/outpatient visit, established, low complexity'),
                    ('99214', 'Office/outpatient visit, established, moderate complexity')],
        'emergency': [('99215', 'Office/outpatient visit, established, high complexity')],
        'follow_up': [('99213', 'Office/outpatient visit, established, low complexity'),
                      ('99214', 'Office/outpatient visit, established, moderate complexity')],
    }

    service_amount_ranges = {
        'eeg_recording': (800, 1500),
        'consultation': (200, 500),
        'neuropsych_assessment': (400, 800),
        'medication_review': (150, 350),
        'pre_surgical_eval': (1000, 2500),
        'telemed': (150, 300),
        'emergency': (500, 1200),
        'follow_up': (200, 400),
    }

    insurers = [
        'Medicare', 'Medicaid', 'BlueCross BlueShield', 'Aetna',
        'UnitedHealthcare', 'Cigna', 'Self-Pay'
    ]

    statuses_weighted = [
        ('paid', 45), ('approved', 15), ('submitted', 15),
        ('denied', 10), ('partially_approved', 8), ('appealed', 5),
        ('write_off', 2)
    ]
    status_pool = []
    for s, w in statuses_weighted:
        status_pool.extend([s] * w)

    denial_reasons = [
        'Prior authorization not obtained',
        'Service not covered under plan',
        'Duplicate claim submission',
        'Incorrect CPT code',
        'Patient eligibility expired',
        'Medical necessity not established',
        'Timely filing limit exceeded',
        'Incomplete documentation',
        'Out-of-network provider',
        'Coordination of benefits required',
    ]

    rng = random.Random(42)
    base = datetime(2026, 5, 1, 8, 0, 0)
    rows = []
    for i in range(150):
        pt = rng.choice(patients)
        prov = rng.choice(providers)
        svc = rng.choice(service_types)
        status = rng.choice(status_pool)
        icd = rng.choice(icd10_codes)
        cpt_options = service_cpt_map.get(svc, cpt_codes[:2])
        cpt = rng.choice(cpt_options)
        insurer = rng.choice(insurers)

        # Generate claim ID: CLM-YYYYMMDD-NNNN
        day_offset = rng.randint(0, 59)
        claim_date = base + timedelta(days=day_offset)
        claim_id = f"CLM-{claim_date.strftime('%Y%m%d')}-{i+1:04d}"

        # Link some claims to appointments
        appt_id = None
        if appt_ids and rng.random() < 0.4:
            appt_id = rng.choice(appt_ids)

        # Amount calculations
        lo, hi = service_amount_ranges.get(svc, (200, 500))
        amount_billed = round(rng.uniform(lo, hi), 2)

        # Calculate amounts based on status
        amount_approved = None
        amount_paid = None
        amount_patient_owes = None
        adjudicated_at = None
        paid_at = None
        denial_reason = None

        submitted_at = claim_date.strftime('%Y-%m-%d %H:%M:%S')

        if status == 'paid':
            adj_days = rng.randint(5, 30)
            adjudicated_at = (claim_date + timedelta(days=adj_days)).strftime('%Y-%m-%d %H:%M:%S')
            pay_days = rng.randint(3, 15)
            paid_at = (claim_date + timedelta(days=adj_days + pay_days)).strftime('%Y-%m-%d %H:%M:%S')
            if insurer == 'Self-Pay':
                amount_approved = amount_billed
                amount_paid = amount_billed
                amount_patient_owes = 0.0
            else:
                approval_pct = rng.uniform(0.70, 1.0)
                amount_approved = round(amount_billed * approval_pct, 2)
                insurance_pct = rng.uniform(0.75, 0.95)
                amount_paid = round(amount_approved * insurance_pct, 2)
                amount_patient_owes = round(amount_approved - amount_paid, 2)

        elif status == 'approved':
            adj_days = rng.randint(5, 25)
            adjudicated_at = (claim_date + timedelta(days=adj_days)).strftime('%Y-%m-%d %H:%M:%S')
            amount_approved = round(amount_billed * rng.uniform(0.75, 1.0), 2)
            amount_patient_owes = round(amount_approved * rng.uniform(0.05, 0.25), 2)

        elif status == 'partially_approved':
            adj_days = rng.randint(7, 30)
            adjudicated_at = (claim_date + timedelta(days=adj_days)).strftime('%Y-%m-%d %H:%M:%S')
            amount_approved = round(amount_billed * rng.uniform(0.40, 0.70), 2)
            amount_patient_owes = round(amount_billed - amount_approved, 2)

        elif status == 'denied':
            adj_days = rng.randint(10, 45)
            adjudicated_at = (claim_date + timedelta(days=adj_days)).strftime('%Y-%m-%d %H:%M:%S')
            amount_approved = 0.0
            amount_paid = 0.0
            amount_patient_owes = amount_billed
            denial_reason = rng.choice(denial_reasons)

        elif status == 'appealed':
            adj_days = rng.randint(10, 30)
            adjudicated_at = (claim_date + timedelta(days=adj_days)).strftime('%Y-%m-%d %H:%M:%S')
            amount_approved = 0.0
            denial_reason = rng.choice(denial_reasons)

        elif status == 'write_off':
            adj_days = rng.randint(30, 90)
            adjudicated_at = (claim_date + timedelta(days=adj_days)).strftime('%Y-%m-%d %H:%M:%S')
            amount_approved = 0.0
            amount_paid = 0.0
            amount_patient_owes = 0.0

        # Policy number
        policy_num = None
        if insurer != 'Self-Pay':
            policy_num = f"{insurer[:3].upper()}-{rng.randint(100000, 999999)}"

        rows.append((
            pt[0], claim_id, appt_id, prov[0], prov[1], svc,
            icd[0], icd[1], cpt[0], cpt[1],
            amount_billed, amount_approved, amount_paid, amount_patient_owes,
            insurer, policy_num, status, submitted_at,
            adjudicated_at, paid_at, denial_reason, ''
        ))

    cur.executemany("""
        INSERT INTO billing_claims
            (patient_id, claim_id, appointment_id, provider, department, service_type,
             icd10_code, icd10_desc, cpt_code, cpt_desc,
             amount_billed, amount_approved, amount_paid, amount_patient_owes,
             insurance_provider, policy_number, status, submitted_at,
             adjudicated_at, paid_at, denial_reason, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

    # Log a transaction for audit trail
    cur.execute("""
        INSERT INTO transaction_log (patient_id, component, action, actor, detail, ts_utc, ts_local)
        VALUES ('SYSTEM', 'billing_claims', 'create', 'system',
                'Seeded 150 billing claim records', datetime('now'), datetime('now','localtime'))
    """)
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────────────────────
#  /api/billing/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Aggregate billing health: total claims, collection rate,
    denial rate, outstanding balance, insurance breakdown, KPIs."""
    _ensure_table()
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total_claims = _safe(cur, "SELECT COUNT(*) FROM billing_claims")
    total_billed = _safe(cur, "SELECT COALESCE(SUM(amount_billed), 0) FROM billing_claims")
    total_collected = _safe(cur, "SELECT COALESCE(SUM(amount_paid), 0) FROM billing_claims WHERE amount_paid IS NOT NULL")
    collection_rate = round(total_collected / total_billed * 100, 1) if total_billed else 0

    denied_count = _safe(cur, "SELECT COUNT(*) FROM billing_claims WHERE status='denied'")
    denial_rate = round(denied_count / total_claims * 100, 1) if total_claims else 0

    avg_days = _safe(cur,
        """SELECT AVG(julianday(paid_at) - julianday(submitted_at))
           FROM billing_claims WHERE paid_at IS NOT NULL""")
    avg_days_to_payment = round(avg_days, 1) if avg_days else 0

    outstanding = _safe(cur,
        """SELECT COALESCE(SUM(amount_billed - COALESCE(amount_paid, 0)), 0)
           FROM billing_claims WHERE status NOT IN ('paid', 'write_off')""")

    unique_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM billing_claims")

    # Status distribution
    status_dist = []
    for row in _safe_rows(cur,
            "SELECT status, COUNT(*) FROM billing_claims GROUP BY status ORDER BY COUNT(*) DESC"):
        status_dist.append({
            "status": row[0], "count": row[1],
            "pct": round(row[1] / total_claims * 100, 1) if total_claims else 0
        })

    # Insurance breakdown
    insurance_bk = []
    for row in _safe_rows(cur,
            """SELECT insurance_provider, COUNT(*) as claims,
                      COALESCE(SUM(amount_billed), 0) as billed,
                      COALESCE(SUM(amount_paid), 0) as collected
               FROM billing_claims GROUP BY insurance_provider ORDER BY billed DESC"""):
        insurance_bk.append({
            "insurer": row[0], "claims": row[1],
            "billed": round(row[2], 2), "collected": round(row[3], 2)
        })

    # Monthly trend
    monthly_trend = []
    for row in _safe_rows(cur,
            """SELECT strftime('%Y-%m', submitted_at) as month, COUNT(*) as claims,
                      COALESCE(SUM(amount_billed), 0) as billed,
                      COALESCE(SUM(amount_paid), 0) as collected
               FROM billing_claims GROUP BY month ORDER BY month"""):
        monthly_trend.append({
            "month": row[0], "claims": row[1],
            "billed": round(row[2], 2), "collected": round(row[3], 2)
        })

    # Top services
    top_services = []
    for row in _safe_rows(cur,
            """SELECT service_type, COUNT(*) as claims,
                      ROUND(AVG(amount_billed), 2) as avg_billed
               FROM billing_claims GROUP BY service_type ORDER BY claims DESC"""):
        top_services.append({
            "service": row[0], "claims": row[1], "avg_billed": row[2]
        })

    conn.close()

    return {
        "available": True,
        "kpis": {
            "total_claims": total_claims,
            "total_billed": round(total_billed, 2),
            "total_collected": round(total_collected, 2),
            "collection_rate_pct": collection_rate,
            "collection_rate": collection_rate,
            "denial_rate_pct": denial_rate,
            "denial_rate": denial_rate,
            "avg_days_to_payment": avg_days_to_payment,
            "outstanding_balance": round(outstanding, 2),
            "unique_patients": unique_patients
        },
        "status_distribution": status_dist,
        "insurance_breakdown": insurance_bk,
        "monthly_trend": monthly_trend,
        "top_services": top_services
    }


# ──────────────────────────────────────────────────────────────
#  /api/billing/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient billing matrix, denied claims detail, aging buckets,
    recent claims, CPT analysis, payer mix."""
    _ensure_table()
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary (top 20)
    per_patient = []
    for row in _safe_rows(cur,
            """SELECT patient_id, COUNT(*) as claims,
                      COALESCE(SUM(amount_billed), 0) as total_billed,
                      COALESCE(SUM(amount_paid), 0) as total_paid,
                      COALESCE(SUM(amount_billed), 0) - COALESCE(SUM(amount_paid), 0) as balance
               FROM billing_claims GROUP BY patient_id
               ORDER BY total_billed DESC LIMIT 20"""):
        per_patient.append({
            "patient_id": row[0], "claims": row[1],
            "total_billed": round(row[2], 2), "total_paid": round(row[3], 2),
            "balance": round(row[4], 2)
        })

    # Denied claims detail
    denied_claims = []
    for row in _safe_rows(cur,
            """SELECT claim_id, patient_id, service_type, amount_billed,
                      denial_reason, submitted_at
               FROM billing_claims WHERE status IN ('denied', 'appealed')
               ORDER BY submitted_at DESC"""):
        denied_claims.append({
            "claim_id": row[0], "patient_id": row[1],
            "service_type": row[2], "amount_billed": row[3],
            "denial_reason": row[4], "submitted_at": row[5]
        })

    # Aging buckets
    aging_buckets = []
    bucket_defs = [
        ('0-30 days', 0, 30),
        ('31-60 days', 31, 60),
        ('61-90 days', 61, 90),
        ('90+ days', 91, 9999),
    ]
    for label, lo, hi in bucket_defs:
        row = _safe_rows(cur,
            """SELECT COUNT(*), COALESCE(SUM(amount_billed - COALESCE(amount_paid, 0)), 0)
               FROM billing_claims
               WHERE status NOT IN ('paid', 'write_off')
                 AND julianday('now') - julianday(submitted_at) BETWEEN ? AND ?""",
            (lo, hi))
        if row:
            aging_buckets.append({
                "bucket": label, "count": row[0][0],
                "amount": round(row[0][1], 2)
            })

    # Recent claims (last 15)
    recent_claims = []
    for row in _safe_rows(cur,
            """SELECT claim_id, patient_id, service_type, amount_billed,
                      status, submitted_at
               FROM billing_claims ORDER BY submitted_at DESC LIMIT 15"""):
        recent_claims.append({
            "claim_id": row[0], "patient_id": row[1],
            "service_type": row[2], "amount_billed": row[3],
            "status": row[4], "submitted_at": row[5]
        })

    # CPT analysis
    cpt_analysis = []
    for row in _safe_rows(cur,
            """SELECT cpt_code, cpt_desc, COUNT(*) as claims,
                      ROUND(AVG(amount_billed), 2) as avg_billed,
                      ROUND(AVG(CASE WHEN amount_paid IS NOT NULL THEN amount_paid END), 2) as avg_paid
               FROM billing_claims GROUP BY cpt_code ORDER BY claims DESC"""):
        cpt_analysis.append({
            "cpt_code": row[0], "cpt_desc": row[1], "claims": row[2],
            "avg_billed": row[3], "avg_paid": row[4]
        })

    # Payer mix
    payer_mix = []
    total_claims = _safe(cur, "SELECT COUNT(*) FROM billing_claims")
    for row in _safe_rows(cur,
            """SELECT insurance_provider, COUNT(*) as cnt
               FROM billing_claims GROUP BY insurance_provider ORDER BY cnt DESC"""):
        payer_mix.append({
            "insurer": row[0],
            "pct": round(row[1] / total_claims * 100, 1) if total_claims else 0
        })

    conn.close()

    return {
        "available": True,
        "per_patient": per_patient,
        "denied_claims": denied_claims,
        "aging_buckets": aging_buckets,
        "recent_claims": recent_claims,
        "cpt_analysis": cpt_analysis,
        "payer_mix": payer_mix
    }


# ──────────────────────────────────────────────────────────────
#  /api/billing/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Metric definitions for tooltip overlays."""
    return {
        "statuses": {
            "submitted": "Claim has been submitted to the insurance payer and is awaiting adjudication.",
            "approved": "Claim has been reviewed and approved for payment by the payer.",
            "partially_approved": "Claim approved for a reduced amount; the remainder may be patient responsibility.",
            "denied": "Claim was rejected by the payer. May be appealed with additional documentation.",
            "paid": "Payment has been received from the payer and/or patient.",
            "appealed": "A denied claim that has been resubmitted with additional justification.",
            "write_off": "Uncollectable balance that has been written off per policy."
        },
        "service_types": {
            "eeg_recording": "Standard or extended EEG recording session (awake, sleep, or ambulatory).",
            "consultation": "Initial or follow-up neurology/psychiatry office visit.",
            "neuropsych_assessment": "Neuropsychological testing and cognitive evaluation.",
            "medication_review": "Focused review of current medication regimen and adjustments.",
            "pre_surgical_eval": "Comprehensive evaluation prior to epilepsy surgery candidacy.",
            "telemed": "Telehealth/video consultation for remote patient follow-up.",
            "emergency": "Urgent or emergency department neurology consultation.",
            "follow_up": "Routine follow-up visit for ongoing treatment monitoring."
        },
        "aging_note": "Aging buckets measure days since claim submission for unpaid/unresolved claims. "
                      "Industry target: >90% resolved within 60 days.",
        "icd10_note": "ICD-10 codes classify diagnoses. Common neuro codes: G40.x (epilepsy), "
                      "F32.x (depression), F20.x (schizophrenia), R56.9 (seizures NOS), G43.x (migraine).",
        "cpt_note": "CPT codes identify services rendered. Key EEG codes: 95816 (awake), 95819 (sleep), "
                    "95957 (digital analysis), 95950 (ambulatory). E/M: 99213-99215.",
        "collection_rate_formula": "Collection Rate = (Total Paid / Total Billed) x 100. "
                                   "Industry benchmark for neurology practices: 92-96%."
    }


if __name__ == '__main__':
    import pprint
    print("=== Overview ===")
    pprint.pprint(overview())
    print("\n=== Breakdown ===")
    pprint.pprint(breakdown())
    print("\n=== Definitions ===")
    pprint.pprint(definitions())
