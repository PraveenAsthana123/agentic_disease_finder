"""
Epilepsy Nurse Specialist (ENS) Module
=======================================
Real seizure-diary analytics from clinical.db seizure_diary table.

Endpoints:
  /api/nurse                  — full dashboard (summary + all sub-analyses)
  /api/nurse/diary-analysis   — seizure trend, severity distribution, trigger correlation
  /api/nurse/adherence        — medication adherence coaching (links meds + diary gaps)
  /api/nurse/safety           — SUDEP risk factors + safety counseling checklist
  /api/nurse/triage           — follow-up triage (risk-stratified patient list)
  /api/nurse/education        — patient/caregiver education gap assessment

All data from REAL seizure_diary + medications tables in data/clinical.db.
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _rows_as_dicts(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ─── 1. Seizure Diary Analysis ──────────────────────────────────────────

def seizure_diary_analysis(patient_id=None):
    """Monthly trend, severity distribution, trigger correlation, injury rate, ER visit rate."""
    c = _conn()
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()

    # All events
    cur = c.execute(f"SELECT * FROM seizure_diary {where} ORDER BY event_date", params)
    events = _rows_as_dicts(cur)
    n = len(events)
    if n == 0:
        c.close()
        return {"events": 0, "message": "No seizure diary entries found."}

    # Monthly trend: count per month
    monthly = defaultdict(int)
    for e in events:
        if e["event_date"]:
            month = e["event_date"][:7]  # YYYY-MM
            monthly[month] += 1
    monthly_trend = [{"month": k, "count": v} for k, v in sorted(monthly.items())]

    # Severity distribution
    sev_counts = defaultdict(int)
    for e in events:
        sev_counts[e["severity"] or "Unknown"] += 1
    severity_dist = [{"severity": k, "count": v, "pct": round(100 * v / n, 1)}
                     for k, v in sorted(sev_counts.items())]

    # Trigger correlation
    trig_counts = defaultdict(int)
    for e in events:
        trig_counts[e["trigger"] or "Not recorded"] += 1
    trigger_corr = [{"trigger": k, "count": v, "pct": round(100 * v / n, 1)}
                    for k, v in sorted(trig_counts.items(), key=lambda x: -x[1])]

    # Injury rate
    injury_count = sum(1 for e in events if e["injury"] and e["injury"] not in ("No", "None", None))
    injury_rate = round(100 * injury_count / n, 1) if n else 0

    # ER visit rate
    er_count = sum(1 for e in events if e["er_visit"] == "Yes")
    er_rate = round(100 * er_count / n, 1) if n else 0

    # Duration stats
    durs = [e["duration_sec"] for e in events if e["duration_sec"]]
    dur_stats = {
        "min_sec": min(durs) if durs else None,
        "avg_sec": round(sum(durs) / len(durs), 1) if durs else None,
        "max_sec": max(durs) if durs else None,
        "prolonged_5min_count": sum(1 for d in durs if d >= 300),
    }

    # Recovery stats
    recov = [e["recovery_min"] for e in events if e["recovery_min"]]
    recov_stats = {
        "min_min": min(recov) if recov else None,
        "avg_min": round(sum(recov) / len(recov), 1) if recov else None,
        "max_min": max(recov) if recov else None,
    }

    # Per-patient seizure frequency (events per patient)
    pat_counts = defaultdict(int)
    for e in events:
        pat_counts[e["patient_id"]] += 1
    high_freq_patients = [{"patient_id": k, "seizure_count": v}
                          for k, v in sorted(pat_counts.items(), key=lambda x: -x[1]) if v >= 2]

    c.close()
    return {
        "total_events": n,
        "distinct_patients": len(pat_counts),
        "date_range": {"first": events[0]["event_date"], "last": events[-1]["event_date"]},
        "monthly_trend": monthly_trend,
        "severity_distribution": severity_dist,
        "trigger_correlation": trigger_corr,
        "injury_rate_pct": injury_rate,
        "er_visit_rate_pct": er_rate,
        "duration_stats": dur_stats,
        "recovery_stats": recov_stats,
        "high_frequency_patients": high_freq_patients,
        "clinical_flags": {
            "prolonged_seizures": dur_stats["prolonged_5min_count"],
            "high_injury_rate": injury_rate > 20,
            "high_er_rate": er_rate > 15,
            "status_epilepticus_risk": any(d >= 300 for d in durs),
        },
    }


# ─── 2. Medication Adherence Coaching ───────────────────────────────────

def adherence_coaching(patient_id=None):
    """Link seizure events with medication data to identify adherence gaps.
    Uses real medications table (patient_id, fields_json with drug_name/dose/frequency)."""
    c = _conn()

    # Get medications
    if patient_id:
        meds_rows = c.execute("SELECT patient_id, fields_json FROM medications WHERE patient_id = ?",
                              (patient_id,)).fetchall()
    else:
        meds_rows = c.execute("SELECT patient_id, fields_json FROM medications").fetchall()

    med_map = {}
    for pid, fj in meds_rows:
        try:
            parsed = json.loads(fj) if fj else {}
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        drug = parsed.get("drug_name", "Unknown")
        dose = parsed.get("dose_mg", "?")
        freq = parsed.get("frequency", "?")
        aeds = parsed.get("aed", [drug])
        if pid not in med_map:
            med_map[pid] = []
        med_map[pid].append({"drug": drug, "dose_mg": dose, "frequency": freq, "all_aeds": aeds})

    # Get seizure counts per patient
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    seiz_rows = c.execute(
        f"SELECT patient_id, COUNT(*) as cnt FROM seizure_diary {where} GROUP BY patient_id", params
    ).fetchall()
    seiz_map = {r[0]: r[1] for r in seiz_rows}

    # Cross-reference: patients on meds who also have seizures (potential adherence concern)
    adherence_flags = []
    for pid, meds in med_map.items():
        seizure_count = seiz_map.get(pid, 0)
        drug_list = [m["drug"] for m in meds]
        flag = "concern" if seizure_count >= 2 else "adequate" if seizure_count <= 1 else "monitor"
        adherence_flags.append({
            "patient_id": pid,
            "medications": meds,
            "drug_count": len(meds),
            "seizure_count": seizure_count,
            "adherence_flag": flag,
            "coaching_note": _adherence_note(seizure_count, len(meds), drug_list),
        })

    # Patients with seizures but NO medication record (gap)
    all_seiz_pats = set(seiz_map.keys())
    med_pats = set(med_map.keys())
    no_med_record = all_seiz_pats - med_pats
    gaps = [{"patient_id": pid, "seizure_count": seiz_map[pid],
             "gap": "No medication record — may need prescribing review"}
            for pid in sorted(no_med_record)]

    c.close()
    return {
        "patients_with_meds": len(med_map),
        "patients_with_seizures": len(seiz_map),
        "adherence_assessments": sorted(adherence_flags, key=lambda x: -x["seizure_count"]),
        "medication_gaps": gaps,
        "summary": {
            "concern_count": sum(1 for a in adherence_flags if a["adherence_flag"] == "concern"),
            "adequate_count": sum(1 for a in adherence_flags if a["adherence_flag"] == "adequate"),
            "no_med_record_count": len(no_med_record),
        },
    }


def _adherence_note(seizure_count, med_count, drugs):
    if seizure_count == 0:
        return "No seizures recorded — continue current regimen."
    if seizure_count == 1:
        return "Single seizure event — assess triggers, reinforce adherence."
    if med_count == 0:
        return f"{seizure_count} seizures, no medications on file — urgent prescribing review."
    if seizure_count >= 3:
        return (f"{seizure_count} seizures despite {med_count} medication(s) ({', '.join(drugs)}) — "
                "consider dose adjustment, drug level check, or add-on therapy.")
    return (f"{seizure_count} seizures on {', '.join(drugs)} — "
            "review timing/dose compliance, check for missed doses.")


# ─── 3. SUDEP Risk & Safety Counseling ──────────────────────────────────

# Evidence-based SUDEP risk factors (from MORTEMUS, Harden 2017)
SUDEP_RISK_FACTORS = [
    {"factor": "Generalized tonic-clonic seizures (GTCS)", "weight": 3,
     "check": "severity == 'Severe' or motor_signs present"},
    {"factor": "High seizure frequency (≥3/month)", "weight": 2,
     "check": "seizure count per month"},
    {"factor": "Nocturnal seizures", "weight": 2,
     "check": "event_time suggests nighttime"},
    {"factor": "Duration > 5 min (status epilepticus risk)", "weight": 3,
     "check": "duration_sec >= 300"},
    {"factor": "Living alone / unsupervised", "weight": 2,
     "check": "witnessed == 'No' pattern"},
    {"factor": "Non-adherence to medication", "weight": 2,
     "check": "medication gaps or adherence concern"},
    {"factor": "Young adult male", "weight": 1,
     "check": "demographics"},
    {"factor": "Intellectual disability / comorbidities", "weight": 1,
     "check": "comorbidity record"},
]

SAFETY_CHECKLIST = [
    {"item": "Seizure action plan documented", "category": "planning"},
    {"item": "Rescue medication (midazolam/diazepam) prescribed and accessible", "category": "medication"},
    {"item": "Nocturnal monitoring device discussed", "category": "monitoring"},
    {"item": "Water safety (shower vs bath, swimming supervision)", "category": "environment"},
    {"item": "Driving restriction counseled per provincial/state law", "category": "driving"},
    {"item": "Workplace/school safety plan", "category": "environment"},
    {"item": "Fall prevention (padded furniture edges, helmet if needed)", "category": "environment"},
    {"item": "Kitchen safety (microwave vs stove, avoid deep frying alone)", "category": "environment"},
    {"item": "Emergency contact card / medical ID bracelet", "category": "identification"},
    {"item": "Caregiver trained in seizure first aid", "category": "caregiver"},
    {"item": "SUDEP risk discussed with patient/family", "category": "counseling"},
    {"item": "Follow-up appointment scheduled", "category": "follow-up"},
]


def safety_counseling(patient_id=None):
    """SUDEP risk factor assessment + safety counseling checklist."""
    c = _conn()
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()

    events = _rows_as_dicts(c.execute(f"SELECT * FROM seizure_diary {where}", params))
    n = len(events)

    # Compute risk indicators from real data
    risk_indicators = []
    severe_count = sum(1 for e in events if e["severity"] == "Severe")
    prolonged = sum(1 for e in events if (e["duration_sec"] or 0) >= 300)
    unwitnessed = sum(1 for e in events if e["witnessed"] == "No")
    injury_events = sum(1 for e in events if e["injury"] and e["injury"] not in ("No", None))

    if severe_count > 0:
        risk_indicators.append({"factor": "GTCS (severe seizures)", "present": True,
                                "count": severe_count, "risk_weight": 3})
    if n >= 3:
        risk_indicators.append({"factor": "High seizure frequency", "present": True,
                                "count": n, "risk_weight": 2})
    if prolonged > 0:
        risk_indicators.append({"factor": "Prolonged seizures (≥5 min)", "present": True,
                                "count": prolonged, "risk_weight": 3})
    if unwitnessed > 0:
        risk_indicators.append({"factor": "Unwitnessed seizures", "present": True,
                                "count": unwitnessed, "risk_weight": 2})

    total_risk_score = sum(r["risk_weight"] for r in risk_indicators)
    risk_level = "high" if total_risk_score >= 5 else "moderate" if total_risk_score >= 2 else "low"

    # Per-patient risk summary
    pat_risk = defaultdict(lambda: {"severe": 0, "prolonged": 0, "total": 0, "injuries": 0})
    for e in events:
        pid = e["patient_id"]
        pat_risk[pid]["total"] += 1
        if e["severity"] == "Severe":
            pat_risk[pid]["severe"] += 1
        if (e["duration_sec"] or 0) >= 300:
            pat_risk[pid]["prolonged"] += 1
        if e["injury"] and e["injury"] not in ("No", None):
            pat_risk[pid]["injuries"] += 1

    patient_risk_list = []
    for pid, d in sorted(pat_risk.items(), key=lambda x: -(x[1]["severe"] * 3 + x[1]["total"])):
        score = d["severe"] * 3 + d["prolonged"] * 3 + d["total"]
        level = "high" if score >= 5 else "moderate" if score >= 2 else "low"
        patient_risk_list.append({
            "patient_id": pid, "risk_level": level, "risk_score": score, **d,
        })

    c.close()
    return {
        "total_events_assessed": n,
        "risk_factors_reference": SUDEP_RISK_FACTORS,
        "detected_risk_indicators": risk_indicators,
        "overall_risk_score": total_risk_score,
        "overall_risk_level": risk_level,
        "patient_risk_summary": patient_risk_list,
        "safety_checklist": SAFETY_CHECKLIST,
        "high_risk_patient_count": sum(1 for p in patient_risk_list if p["risk_level"] == "high"),
    }


# ─── 4. Follow-up / Telephone Triage ───────────────────────────────────

def follow_up_triage(patient_id=None):
    """Risk-stratified patient list for follow-up calls. Urgency based on recent events."""
    c = _conn()
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()

    events = _rows_as_dicts(c.execute(
        f"SELECT * FROM seizure_diary {where} ORDER BY event_date DESC", params))

    # Group by patient, compute urgency
    pat_events = defaultdict(list)
    for e in events:
        pat_events[e["patient_id"]].append(e)

    triage_list = []
    for pid, evts in pat_events.items():
        latest = evts[0]
        total = len(evts)
        severe_count = sum(1 for e in evts if e["severity"] == "Severe")
        er_count = sum(1 for e in evts if e["er_visit"] == "Yes")
        injury_count = sum(1 for e in evts if e["injury"] and e["injury"] not in ("No", None))

        # Urgency scoring
        urgency_score = 0
        reasons = []
        if severe_count > 0:
            urgency_score += 3
            reasons.append(f"{severe_count} severe seizure(s)")
        if er_count > 0:
            urgency_score += 2
            reasons.append(f"{er_count} ER visit(s)")
        if injury_count > 0:
            urgency_score += 2
            reasons.append(f"{injury_count} injury event(s)")
        if total >= 3:
            urgency_score += 1
            reasons.append(f"High frequency ({total} events)")

        urgency = "urgent" if urgency_score >= 5 else "soon" if urgency_score >= 2 else "routine"

        triage_list.append({
            "patient_id": pid,
            "urgency": urgency,
            "urgency_score": urgency_score,
            "reasons": reasons,
            "total_seizures": total,
            "severe_count": severe_count,
            "er_visits": er_count,
            "injuries": injury_count,
            "latest_event_date": latest["event_date"],
            "latest_severity": latest["severity"],
            "recommended_action": _triage_action(urgency),
        })

    triage_list.sort(key=lambda x: -x["urgency_score"])

    c.close()
    return {
        "patients_assessed": len(triage_list),
        "triage_list": triage_list,
        "summary": {
            "urgent": sum(1 for t in triage_list if t["urgency"] == "urgent"),
            "soon": sum(1 for t in triage_list if t["urgency"] == "soon"),
            "routine": sum(1 for t in triage_list if t["urgency"] == "routine"),
        },
    }


def _triage_action(urgency):
    return {
        "urgent": "Call within 24h — assess for med change, ER follow-up, safety review.",
        "soon": "Call within 72h — review diary, reinforce adherence, schedule clinic.",
        "routine": "Next scheduled follow-up — review at regular appointment.",
    }.get(urgency, "Review at next visit.")


# ─── 5. Patient/Caregiver Education Assessment ─────────────────────────

EDUCATION_MODULES = [
    {"module": "Seizure First Aid", "category": "emergency",
     "target": "patient+caregiver", "key_points": [
         "Stay calm, time the seizure", "Clear area of hazards",
         "Do NOT restrain or put anything in mouth",
         "Turn on side (recovery position)", "Call 911 if >5 min or injury"]},
    {"module": "Medication Management", "category": "treatment",
     "target": "patient", "key_points": [
         "Take medications at same time daily", "Never skip doses",
         "Report side effects promptly", "Do not stop abruptly"]},
    {"module": "Trigger Avoidance", "category": "lifestyle",
     "target": "patient", "key_points": [
         "Prioritize sleep (7-9 hours)", "Manage stress",
         "Limit alcohol", "Track personal triggers in diary"]},
    {"module": "SUDEP Awareness", "category": "safety",
     "target": "patient+caregiver", "key_points": [
         "SUDEP risk exists but is rare (~1/1000/year)",
         "Best prevention: seizure control + medication adherence",
         "Nocturnal monitoring for high-risk patients",
         "Discuss openly — knowledge reduces anxiety"]},
    {"module": "Driving & Activity Safety", "category": "safety",
     "target": "patient", "key_points": [
         "Seizure-free period required before driving (varies by jurisdiction)",
         "Swimming only with buddy/supervision",
         "Heights and machinery precautions",
         "Carry medical ID at all times"]},
    {"module": "Emergency Planning", "category": "emergency",
     "target": "caregiver", "key_points": [
         "Know rescue medication administration",
         "Have seizure action plan posted at home/school/work",
         "Know when to call 911 vs manage at home",
         "Keep emergency contacts accessible"]},
]


def education_assessment(patient_id=None):
    """Assess knowledge gaps based on seizure diary data patterns."""
    c = _conn()
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()

    events = _rows_as_dicts(c.execute(f"SELECT * FROM seizure_diary {where}", params))
    n = len(events)

    # Recommend modules based on data patterns
    recommendations = []
    severe = sum(1 for e in events if e["severity"] == "Severe")
    injuries = sum(1 for e in events if e["injury"] and e["injury"] not in ("No", None))
    er_visits = sum(1 for e in events if e["er_visit"] == "Yes")
    sleep_trigger = sum(1 for e in events if e["trigger"] and "sleep" in (e["trigger"] or "").lower())

    for mod in EDUCATION_MODULES:
        priority = "standard"
        rationale = "Routine education module."
        if mod["module"] == "Seizure First Aid" and injuries > 0:
            priority = "high"
            rationale = f"{injuries} injury events — reinforce first aid training."
        elif mod["module"] == "SUDEP Awareness" and severe > 0:
            priority = "high"
            rationale = f"{severe} severe seizures — SUDEP discussion indicated."
        elif mod["module"] == "Trigger Avoidance" and sleep_trigger > 0:
            priority = "high"
            rationale = f"{sleep_trigger} sleep-deprivation triggers — lifestyle counseling needed."
        elif mod["module"] == "Emergency Planning" and er_visits > 0:
            priority = "high"
            rationale = f"{er_visits} ER visits — review emergency plan."
        elif mod["module"] == "Medication Management" and n >= 3:
            priority = "high"
            rationale = f"{n} seizures — adherence reinforcement critical."

        recommendations.append({**mod, "priority": priority, "rationale": rationale})

    recommendations.sort(key=lambda x: (0 if x["priority"] == "high" else 1))

    c.close()
    return {
        "total_modules": len(EDUCATION_MODULES),
        "high_priority_count": sum(1 for r in recommendations if r["priority"] == "high"),
        "recommendations": recommendations,
        "data_driven_context": {
            "total_seizures": n,
            "severe_seizures": severe,
            "injuries": injuries,
            "er_visits": er_visits,
            "sleep_trigger_events": sleep_trigger,
        },
    }


# ─── Full Dashboard ────────────────────────────────────────────────────

def full_dashboard(patient_id=None):
    """Complete ENS dashboard: diary analysis + adherence + safety + triage + education."""
    return {
        "role": "Epilepsy Nurse Specialist (ENS)",
        "icon": "🩹",
        "data_source": "REAL seizure_diary + medications tables (data/clinical.db)",
        "diary_analysis": seizure_diary_analysis(patient_id),
        "adherence_coaching": adherence_coaching(patient_id),
        "safety_counseling": safety_counseling(patient_id),
        "follow_up_triage": follow_up_triage(patient_id),
        "education_assessment": education_assessment(patient_id),
    }
