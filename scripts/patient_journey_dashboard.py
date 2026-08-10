"""Patient Care Journey Dashboard — real data from clinical.db.

Tracks the full care pathway: Referral → Appointment → Telehealth → Hospitalization.
Combines 4 tables to show care-touchpoint counts, funnel progression,
journey intensity per patient, and outcome metrics.

Sources:
- referral_records (84 rows) — urgency, source, triage status, triage score
- appointments (120 rows)    — department, type, status, duration
- telehealth_sessions (109 rows) — type, platform, quality, satisfaction
- hospitalization (115 rows) — admission type, LOS, discharge disposition,
                                readmission, seizure-free at discharge, cost
"""

import sqlite3
import os
import json

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


def _scalar(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def journey_overview():
    """Care journey overview — touchpoint funnel, patient counts, journey KPIs."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Touchpoint counts ──────────────────────────────────────────────────
    n_referrals    = _scalar(cur, "SELECT COUNT(*) FROM referral_records")
    n_appointments = _scalar(cur, "SELECT COUNT(*) FROM appointments")
    n_telehealth   = _scalar(cur, "SELECT COUNT(*) FROM telehealth_sessions")
    n_hosp         = _scalar(cur, "SELECT COUNT(*) FROM hospitalization")

    # Unique patients per touchpoint
    pts_ref  = _scalar(cur, "SELECT COUNT(DISTINCT patient_id) FROM referral_records")
    pts_appt = _scalar(cur, "SELECT COUNT(DISTINCT patient_id) FROM appointments")
    pts_tel  = _scalar(cur, "SELECT COUNT(DISTINCT patient_id) FROM telehealth_sessions")
    pts_hosp = _scalar(cur, "SELECT COUNT(DISTINCT patient_id) FROM hospitalization")

    # Total unique patients across all touchpoints
    rows = _safe(cur, """
        SELECT COUNT(DISTINCT p) FROM (
            SELECT patient_id AS p FROM referral_records
            UNION SELECT patient_id FROM appointments
            UNION SELECT patient_id FROM telehealth_sessions
            UNION SELECT patient_id FROM hospitalization
        )
    """)
    total_patients = rows[0][0] if rows else 0

    # ── 2. Referral KPIs ─────────────────────────────────────────────────────
    avg_triage_score = _scalar(cur, "SELECT AVG(triage_score) FROM referral_records")
    avg_triage_score = round(avg_triage_score or 0, 1)

    ref_urgency = _safe(cur,
        "SELECT urgency, COUNT(*) FROM referral_records GROUP BY urgency ORDER BY COUNT(*) DESC")
    ref_source  = _safe(cur,
        "SELECT referral_source, COUNT(*) FROM referral_records GROUP BY referral_source ORDER BY COUNT(*) DESC")
    ref_status  = _safe(cur,
        "SELECT triage_status, COUNT(*) FROM referral_records GROUP BY triage_status ORDER BY COUNT(*) DESC")
    ref_reason  = _safe(cur,
        "SELECT referral_reason, COUNT(*) FROM referral_records GROUP BY referral_reason ORDER BY COUNT(*) DESC LIMIT 8")

    # ── 3. Appointment KPIs ───────────────────────────────────────────────────
    appt_status = _safe(cur,
        "SELECT status, COUNT(*) FROM appointments GROUP BY status ORDER BY COUNT(*) DESC")
    appt_dept   = _safe(cur,
        "SELECT department, COUNT(*) FROM appointments GROUP BY department ORDER BY COUNT(*) DESC")
    appt_type   = _safe(cur,
        "SELECT appt_type, COUNT(*) FROM appointments GROUP BY appt_type ORDER BY COUNT(*) DESC")
    completed_appts = _scalar(cur,
        "SELECT COUNT(*) FROM appointments WHERE status = 'completed'")
    completion_rate = round(completed_appts / max(n_appointments, 1) * 100, 1)
    avg_duration = _scalar(cur,
        "SELECT AVG(duration_min) FROM appointments WHERE status = 'completed'")
    avg_duration = round(avg_duration or 0, 1)

    # ── 4. Telehealth KPIs ────────────────────────────────────────────────────
    tel_type    = _safe(cur,
        "SELECT session_type, COUNT(*) FROM telehealth_sessions GROUP BY session_type ORDER BY COUNT(*) DESC")
    tel_platform = _safe(cur,
        "SELECT platform, COUNT(*) FROM telehealth_sessions GROUP BY platform ORDER BY COUNT(*) DESC")
    tel_quality = _safe(cur,
        "SELECT connection_quality, COUNT(*) FROM telehealth_sessions GROUP BY connection_quality ORDER BY COUNT(*) DESC")
    avg_satisfaction = _scalar(cur,
        "SELECT AVG(patient_satisfaction) FROM telehealth_sessions")
    avg_satisfaction = round(avg_satisfaction or 0, 2)
    excellent_quality = _scalar(cur,
        "SELECT COUNT(*) FROM telehealth_sessions WHERE connection_quality = 'excellent'")
    excellent_pct = round(excellent_quality / max(n_telehealth, 1) * 100, 1)

    # ── 5. Hospitalization KPIs ───────────────────────────────────────────────
    hosp_rows = _safe(cur, "SELECT fields_json FROM hospitalization")
    adm_types = {}
    adm_reasons = {}
    wards = {}
    dispositions = {}
    los_list = []
    readmit_count = 0
    seizure_free_count = 0
    cost_list = []
    for r in hosp_rows:
        try:
            d = json.loads(r[0])
            at = d.get('admission_type') or 'unknown'
            adm_types[at] = adm_types.get(at, 0) + 1
            ar = d.get('admission_reason') or 'unknown'
            adm_reasons[ar] = adm_reasons.get(ar, 0) + 1
            w = d.get('ward') or 'unknown'
            wards[w] = wards.get(w, 0) + 1
            dd = d.get('discharge_disposition') or 'unknown'
            dispositions[dd] = dispositions.get(dd, 0) + 1
            los = d.get('length_of_stay_days')
            if los is not None:
                los_list.append(los)
            if d.get('readmission_within_30d'):
                readmit_count += 1
            if d.get('seizure_free_at_discharge'):
                seizure_free_count += 1
            cost = d.get('total_cost_usd')
            if cost:
                cost_list.append(cost)
        except Exception:
            pass

    avg_los = round(sum(los_list) / max(len(los_list), 1), 1)
    readmit_rate = round(readmit_count / max(n_hosp, 1) * 100, 1)
    seizure_free_rate = round(seizure_free_count / max(n_hosp, 1) * 100, 1)
    avg_cost = round(sum(cost_list) / max(len(cost_list), 1)) if cost_list else 0

    # ── 6. Care funnel (patients with each touchpoint) ───────────────────────
    funnel = [
        {"stage": "Referred",      "patients": pts_ref,  "events": n_referrals},
        {"stage": "Appointments",  "patients": pts_appt, "events": n_appointments},
        {"stage": "Telehealth",    "patients": pts_tel,  "events": n_telehealth},
        {"stage": "Hospitalised",  "patients": pts_hosp, "events": n_hosp},
    ]

    conn.close()
    return {
        "available": True,
        "total_patients": total_patients,
        "total_touchpoints": n_referrals + n_appointments + n_telehealth + n_hosp,
        # Referrals
        "n_referrals": n_referrals,
        "pts_referred": pts_ref,
        "avg_triage_score": avg_triage_score,
        "referral_urgency": [{"urgency": r[0], "count": r[1]} for r in ref_urgency],
        "referral_source":  [{"source": r[0],  "count": r[1]} for r in ref_source],
        "referral_status":  [{"status": r[0],  "count": r[1]} for r in ref_status],
        "referral_reason":  [{"reason": r[0],  "count": r[1]} for r in ref_reason],
        # Appointments
        "n_appointments": n_appointments,
        "pts_appointments": pts_appt,
        "appt_completion_rate": completion_rate,
        "avg_appt_duration_min": avg_duration,
        "appt_status":      [{"status": r[0],  "count": r[1]} for r in appt_status],
        "appt_department":  [{"dept": r[0],    "count": r[1]} for r in appt_dept],
        "appt_type":        [{"type": r[0],    "count": r[1]} for r in appt_type],
        # Telehealth
        "n_telehealth": n_telehealth,
        "pts_telehealth": pts_tel,
        "avg_patient_satisfaction": avg_satisfaction,
        "excellent_quality_pct": excellent_pct,
        "telehealth_type":     [{"type": r[0],     "count": r[1]} for r in tel_type],
        "telehealth_platform": [{"platform": r[0], "count": r[1]} for r in tel_platform],
        "telehealth_quality":  [{"quality": r[0],  "count": r[1]} for r in tel_quality],
        # Hospitalization
        "n_hospitalizations": n_hosp,
        "pts_hospitalised": pts_hosp,
        "avg_length_of_stay_days": avg_los,
        "readmission_rate_pct": readmit_rate,
        "seizure_free_at_discharge_pct": seizure_free_rate,
        "avg_cost_usd": avg_cost,
        "admission_types":  [{"type": k,   "count": v} for k, v in sorted(adm_types.items(), key=lambda x: -x[1])],
        "admission_reasons":[{"reason": k, "count": v} for k, v in sorted(adm_reasons.items(), key=lambda x: -x[1])[:8]],
        "wards":            [{"ward": k,   "count": v} for k, v in sorted(wards.items(), key=lambda x: -x[1])],
        "discharge_dispositions": [{"disposition": k, "count": v} for k, v in sorted(dispositions.items(), key=lambda x: -x[1])],
        # Funnel
        "care_funnel": funnel,
    }


def journey_breakdown():
    """Per-patient care journey breakdown — touchpoints, stages reached, intensity."""
    if not os.path.exists(DB):
        return {"available": False}

    conn = _conn()
    cur = conn.cursor()

    # Build per-patient journey map
    patients_ref  = {r[0]: r[1] for r in _safe(cur,
        "SELECT patient_id, COUNT(*) FROM referral_records GROUP BY patient_id")}
    patients_appt = {r[0]: r[1] for r in _safe(cur,
        "SELECT patient_id, COUNT(*) FROM appointments GROUP BY patient_id")}
    patients_tel  = {r[0]: r[1] for r in _safe(cur,
        "SELECT patient_id, COUNT(*) FROM telehealth_sessions GROUP BY patient_id")}
    patients_hosp = {r[0]: r[1] for r in _safe(cur,
        "SELECT patient_id, COUNT(*) FROM hospitalization GROUP BY patient_id")}

    all_patients = sorted(set(
        list(patients_ref.keys()) +
        list(patients_appt.keys()) +
        list(patients_tel.keys()) +
        list(patients_hosp.keys())
    ))

    per_patient = []
    for pid in all_patients:
        refs  = patients_ref.get(pid, 0)
        appts = patients_appt.get(pid, 0)
        tel   = patients_tel.get(pid, 0)
        hosp  = patients_hosp.get(pid, 0)
        total = refs + appts + tel + hosp
        stages = sum([refs > 0, appts > 0, tel > 0, hosp > 0])
        per_patient.append({
            "patient_id":     pid,
            "referrals":      refs,
            "appointments":   appts,
            "telehealth":     tel,
            "hospitalizations": hosp,
            "total_touchpoints": total,
            "stages_reached": stages,
            "intensity": "High" if total >= 15 else ("Medium" if total >= 7 else "Low"),
        })

    # Sort by total touchpoints desc
    per_patient.sort(key=lambda x: -x["total_touchpoints"])

    # Intensity distribution
    intensities = {"High": 0, "Medium": 0, "Low": 0}
    for p in per_patient:
        intensities[p["intensity"]] += 1

    # Stage coverage distribution (how many stages reached)
    stages_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in per_patient:
        s = p["stages_reached"]
        stages_dist[s] = stages_dist.get(s, 0) + 1

    conn.close()
    return {
        "available": True,
        "per_patient": per_patient,
        "intensity_distribution": [{"intensity": k, "count": v} for k, v in intensities.items()],
        "stages_reached_distribution": [
            {"stages": k, "count": v} for k, v in sorted(stages_dist.items())
        ],
        "avg_touchpoints_per_patient": round(
            sum(p["total_touchpoints"] for p in per_patient) / max(len(per_patient), 1), 1
        ),
    }


def journey_definitions():
    """Metric definitions for the Patient Care Journey dashboard."""
    return {
        "available": True,
        "dashboard": "Patient Care Journey — tracks the full care pathway across 4 touchpoint stages.",
        "sources": [
            {"table": "referral_records", "rows": 84, "description": "Inbound referrals with urgency, source, and triage status"},
            {"table": "appointments",     "rows": 120, "description": "Scheduled visits with department, type, and completion status"},
            {"table": "telehealth_sessions", "rows": 109, "description": "Remote care sessions with quality and satisfaction scores"},
            {"table": "hospitalization",  "rows": 115, "description": "Inpatient stays with LOS, cost, readmission, and outcomes"},
        ],
        "fields": {
            "care_funnel":      "Patients and events at each stage of the care journey",
            "avg_triage_score": "Mean priority score assigned during referral triage (0–100)",
            "appt_completion_rate": "% of appointments with status = completed",
            "avg_patient_satisfaction": "Mean patient-reported satisfaction (1–5 scale) for telehealth",
            "excellent_quality_pct": "% of telehealth sessions rated 'excellent' for connection quality",
            "avg_length_of_stay_days": "Mean inpatient days per hospitalization",
            "readmission_rate_pct": "% of hospitalizations with readmission within 30 days",
            "seizure_free_at_discharge_pct": "% of hospitalizations where patient was seizure-free at discharge",
            "intensity": "Care intensity tier: High (≥15 touchpoints), Medium (7–14), Low (<7)",
            "stages_reached": "Number of care stages (1–4) a patient has touchpoints in",
        },
        "urgency_levels": {
            "emergent": "Requires immediate attention (< 24 h)",
            "urgent":   "Requires attention within days",
            "routine":  "Standard scheduling timeframe",
            "elective": "Scheduled at patient/provider convenience",
        },
        "triage_statuses": {
            "pending_triage": "Referral received, not yet reviewed",
            "triaged":        "Priority assessed, awaiting scheduling",
            "scheduled":      "Appointment booked",
            "in_progress":    "Active workup underway",
            "completed":      "Referral episode closed",
            "cancelled":      "Referral withdrawn or not pursued",
        },
        "clinical_notes": [
            "49 unique patients appear across at least one care touchpoint.",
            "Patients with all 4 stages represent the most complex epilepsy cases.",
            "High readmission rate flags candidates for seizure-action-plan review.",
            "Telehealth satisfaction < 3 / 5 may indicate platform or access barriers.",
            "Avg LOS is tracked per ILAE quality indicator recommendations.",
        ],
    }
