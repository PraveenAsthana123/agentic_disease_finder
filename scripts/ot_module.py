"""
Occupational Therapist (OT) Module
====================================
Real ADL/functional assessment from clinical.db BARTHEL, MOCA, MMSE,
seizure_diary, and medications tables.

Endpoints:
  /api/ot                       — full dashboard (all 4 sub-analyses)
  /api/ot/adl-assessment        — ADL/IADL (Barthel Index per-domain breakdown)
  /api/ot/fall-risk             — Home Safety / Fall Risk (seizure injuries + ADL mobility + med sedation)
  /api/ot/return-to-work        — Return-to-Work Planner (functional independence + seizure freq + cognition)
  /api/ot/cognitive-function    — Cognitive-Function OT (MOCA + MMSE with OT-specific recommendations)

All data from REAL assessments + seizure_diary + medications in data/clinical.db.
"""

import sqlite3
import os
import json
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")

# Barthel Index 10-item domain names (standard ordering)
BARTHEL_DOMAINS = [
    "Feeding",
    "Bathing",
    "Grooming",
    "Dressing",
    "Bowel control",
    "Bladder control",
    "Toilet use",
    "Transfers (bed↔chair)",
    "Mobility (level surface)",
    "Stairs",
]

# Max scores per Barthel domain (standard Mahoney & Barthel 1965)
BARTHEL_MAX = [10, 5, 5, 10, 10, 10, 10, 15, 15, 10]

# Barthel total score interpretation
BARTHEL_LEVELS = [
    (0, 20, "Total dependence", "Requires 24-hour care; full assistance for all ADLs"),
    (21, 60, "Severe dependence", "Requires substantial daily assistance; may need institutional care"),
    (61, 90, "Moderate dependence", "Needs some help with ADLs; community-living feasible with support"),
    (91, 99, "Slight dependence", "Mostly independent; minimal assistance for complex tasks"),
    (100, 100, "Independent", "Fully independent in all basic ADLs"),
]

# Medications with sedation risk (relevant to fall risk)
SEDATING_MEDS = {
    "phenobarbital": "high",
    "clonazepam": "high",
    "clobazam": "moderate",
    "carbamazepine": "moderate",
    "valproic acid": "moderate",
    "topiramate": "moderate",
    "pregabalin": "moderate",
    "gabapentin": "moderate",
    "zonisamide": "low",
    "levetiracetam": "low",
    "lamotrigine": "low",
    "lacosamide": "low",
    "oxcarbazepine": "moderate",
    "perampanel": "high",
    "brivaracetam": "low",
    "eslicarbazepine": "moderate",
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _rows_as_dicts(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _barthel_level(score):
    for lo, hi, label, desc in BARTHEL_LEVELS:
        if lo <= score <= hi:
            return {"level": label, "description": desc}
    return {"level": "Unknown", "description": ""}


# ─── 1. ADL/IADL Assessment (Barthel Index) ─────────────────────────────

def adl_assessment(patient_id=None):
    """Per-domain Barthel Index breakdown with functional profile and OT recommendations."""
    c = _conn()
    where = "WHERE instrument='BARTHEL'" + (" AND patient_id = ?" if patient_id else "")
    params = (patient_id,) if patient_id else ()

    cur = c.execute(
        f"SELECT patient_id, answers_json, score, max_score, interpretation, level, created_at "
        f"FROM assessments {where} ORDER BY created_at DESC", params
    )
    rows = _rows_as_dicts(cur)
    n = len(rows)
    if n == 0:
        c.close()
        return {"assessments": 0, "message": "No Barthel Index assessments found."}

    # Per-patient latest assessment
    latest_by_patient = {}
    for r in rows:
        pid = r["patient_id"]
        if pid not in latest_by_patient:
            latest_by_patient[pid] = r

    # Per-domain analysis across all patients
    domain_scores = defaultdict(list)
    patient_profiles = []

    for pid, r in latest_by_patient.items():
        answers = json.loads(r["answers_json"]) if r["answers_json"] else {}
        total = r["score"] or 0
        level_info = _barthel_level(total)

        # Extract per-domain scores
        domains = []
        impaired = []
        for i, domain_name in enumerate(BARTHEL_DOMAINS):
            key = f"item{i + 1}"
            raw = answers.get(key, 0)
            val = float(raw) if raw is not None else 0
            mx = BARTHEL_MAX[i]
            pct = round(100 * val / mx, 1) if mx > 0 else 0
            domain_scores[domain_name].append(val)
            status = "Independent" if val >= mx else ("Needs help" if val > 0 else "Dependent")
            domains.append({
                "domain": domain_name,
                "score": val,
                "max": mx,
                "pct": pct,
                "status": status,
            })
            if val < mx:
                impaired.append(domain_name)

        # OT recommendations based on impaired domains
        recs = _adl_recommendations(impaired, total)

        patient_profiles.append({
            "patient_id": pid,
            "total_score": total,
            "max_score": 100,
            "level": level_info["level"],
            "description": level_info["description"],
            "domains": domains,
            "impaired_domains": impaired,
            "ot_recommendations": recs,
            "assessed_at": r["created_at"],
        })

    # Cohort summary
    all_totals = [p["total_score"] for p in patient_profiles]
    dep_distribution = defaultdict(int)
    for p in patient_profiles:
        dep_distribution[p["level"]] += 1

    # Per-domain cohort averages
    domain_summary = []
    for i, name in enumerate(BARTHEL_DOMAINS):
        scores = domain_scores[name]
        mx = BARTHEL_MAX[i]
        avg = round(sum(scores) / len(scores), 1) if scores else 0
        impaired_n = sum(1 for s in scores if s < mx)
        domain_summary.append({
            "domain": name,
            "mean_score": avg,
            "max": mx,
            "mean_pct": round(100 * avg / mx, 1) if mx else 0,
            "impaired_count": impaired_n,
            "impaired_pct": round(100 * impaired_n / len(scores), 1) if scores else 0,
        })

    # Sort by highest impairment rate
    domain_summary.sort(key=lambda d: -d["impaired_pct"])

    c.close()
    return {
        "instrument": "Barthel Index (Mahoney & Barthel, 1965)",
        "total_assessments": n,
        "unique_patients": len(latest_by_patient),
        "cohort_summary": {
            "mean_total": round(sum(all_totals) / len(all_totals), 1) if all_totals else 0,
            "min_total": min(all_totals) if all_totals else 0,
            "max_total": max(all_totals) if all_totals else 0,
            "dependency_distribution": [
                {"level": k, "count": v, "pct": round(100 * v / len(all_totals), 1)}
                for k, v in sorted(dep_distribution.items())
            ],
        },
        "domain_analysis": domain_summary,
        "patient_profiles": sorted(patient_profiles, key=lambda p: p["total_score"]),
    }


def _adl_recommendations(impaired, total):
    """Generate OT-specific recommendations based on impaired ADL domains."""
    recs = []
    domain_recs = {
        "Feeding": "Adaptive utensils (weighted, angled), plate guards, non-slip mats; assess swallowing (refer SLP if needed)",
        "Bathing": "Shower chair/bench, grab bars, handheld showerhead, long-handled sponge; caregiver training for transfer assist",
        "Grooming": "Electric razor, pump-dispensed soap, suction-cup mirror, adapted toothbrush handle",
        "Dressing": "Velcro closures, button hooks, elastic shoelaces, dressing stick, seated dressing technique",
        "Bowel control": "Toileting schedule, fiber/hydration protocol; coordinate with nursing for bowel program",
        "Bladder control": "Timed voiding schedule, proximity to bathroom, night-time urinal/commode; coordinate with urology",
        "Toilet use": "Raised toilet seat, grab bars, commode chair; train caregiver for safe transfers",
        "Transfers (bed↔chair)": "Transfer board, bed rail, hospital bed height adjustment; seating assessment (cushion, wheelchair fit)",
        "Mobility (level surface)": "Gait aid assessment (walker, rollator), home walkway clearing, non-slip flooring; fall prevention program",
        "Stairs": "Stair rail assessment, stair-glide evaluation, single-floor living arrangement; energy conservation techniques",
    }
    for domain in impaired:
        if domain in domain_recs:
            recs.append({"domain": domain, "intervention": domain_recs[domain]})

    if total <= 60:
        recs.append({"domain": "General", "intervention": "Consider comprehensive home assessment + caregiver training program; explore community OT services"})
    elif total <= 90:
        recs.append({"domain": "General", "intervention": "Focus on targeted adaptive equipment for impaired domains; energy conservation strategies"})
    return recs


# ─── 2. Home Safety / Fall Risk ──────────────────────────────────────────

def fall_risk_assessment(patient_id=None):
    """Multi-factor fall risk: seizure injuries + BARTHEL mobility + medication sedation risk."""
    c = _conn()

    # 1. Seizure-related fall/injury data
    where_sz = "WHERE 1=1" + (" AND patient_id = ?" if patient_id else "")
    params = (patient_id,) if patient_id else ()
    cur = c.execute(f"SELECT * FROM seizure_diary {where_sz}", params)
    seizures = _rows_as_dicts(cur)
    total_seizures = len(seizures)
    fall_events = [s for s in seizures if s.get("injury") and s["injury"] not in ("No", "None", None)]
    fall_count = len(fall_events)
    er_from_falls = sum(1 for s in fall_events if s.get("er_visit") == "Yes")

    # 2. BARTHEL mobility scores (item8=Transfers, item9=Mobility, item10=Stairs)
    where_b = "WHERE instrument='BARTHEL'" + (" AND patient_id = ?" if patient_id else "")
    cur = c.execute(f"SELECT patient_id, answers_json, score FROM assessments {where_b}", params)
    barthel_rows = _rows_as_dicts(cur)

    mobility_risks = []
    for r in barthel_rows:
        answers = json.loads(r["answers_json"]) if r["answers_json"] else {}
        transfers = float(answers.get("item8", 0) or 0)
        mobility = float(answers.get("item9", 0) or 0)
        stairs = float(answers.get("item10", 0) or 0)
        total = r["score"] or 0
        # Flag if mobility domains are impaired
        risk_factors = []
        if transfers < 15:
            risk_factors.append("Impaired transfers")
        if mobility < 15:
            risk_factors.append("Impaired mobility")
        if stairs < 10:
            risk_factors.append("Cannot manage stairs independently")
        if total <= 60:
            risk_factors.append("Severe ADL dependence (Barthel ≤60)")

        if risk_factors:
            mobility_risks.append({
                "patient_id": r["patient_id"],
                "barthel_total": total,
                "transfers_score": f"{transfers}/15",
                "mobility_score": f"{mobility}/15",
                "stairs_score": f"{stairs}/10",
                "risk_factors": risk_factors,
            })

    # 3. Medication sedation risk (medications stores drug info in fields_json)
    where_m = "" if not patient_id else "WHERE patient_id = ?"
    cur = c.execute(f"SELECT patient_id, fields_json FROM medications {where_m}", params)
    meds_raw = _rows_as_dicts(cur)
    meds = []
    for mr in meds_raw:
        fields = json.loads(mr["fields_json"]) if mr.get("fields_json") else {}
        drug = fields.get("drug_name", "")
        meds.append({"patient_id": mr["patient_id"], "drug_name": drug,
                      "dose": fields.get("dose_mg", ""), "frequency": fields.get("frequency", "")})

    med_risks = []
    patients_with_sedating = defaultdict(list)
    for m in meds:
        drug_lower = (m["drug_name"] or "").lower()
        for sed_drug, sed_level in SEDATING_MEDS.items():
            if sed_drug in drug_lower:
                patients_with_sedating[m["patient_id"]].append({
                    "drug": m["drug_name"],
                    "dose": m.get("dose", ""),
                    "sedation_risk": sed_level,
                })
                break

    for pid, drugs in patients_with_sedating.items():
        high_count = sum(1 for d in drugs if d["sedation_risk"] == "high")
        mod_count = sum(1 for d in drugs if d["sedation_risk"] == "moderate")
        overall = "High" if high_count >= 1 else ("Moderate" if mod_count >= 1 else "Low")
        med_risks.append({
            "patient_id": pid,
            "sedating_medications": drugs,
            "combined_sedation_risk": overall,
            "polypharmacy_flag": len(drugs) >= 3,
        })

    # Composite risk per patient
    all_patients = set()
    for r in barthel_rows:
        all_patients.add(r["patient_id"])
    for s in seizures:
        all_patients.add(s["patient_id"])
    for m in meds:
        all_patients.add(m["patient_id"])

    # Home safety checklist (evidence-based)
    home_safety_checklist = [
        {"area": "Bathroom", "items": ["Grab bars at tub/shower and toilet", "Non-slip mat in tub/shower", "Raised toilet seat if needed", "Remove bath lock if seizure risk"]},
        {"area": "Bedroom", "items": ["Low bed or floor mattress for nocturnal seizures", "Padded bed rails if indicated", "Night-light for pathway to bathroom", "Remove sharp furniture edges near bed"]},
        {"area": "Kitchen", "items": ["Microwave over stove-top cooking", "Auto-shutoff appliances", "Avoid carrying hot liquids", "Timer/alarm reminders for cooking"]},
        {"area": "Living areas", "items": ["Remove loose rugs and cords", "Non-slip flooring", "Adequate lighting throughout", "Clear pathways (no clutter)"]},
        {"area": "Stairs", "items": ["Secure handrails both sides", "Non-slip stair treads", "Gate at top if fall risk high", "Consider single-floor living"]},
        {"area": "General", "items": ["Medical alert bracelet/pendant", "Seizure first-aid kit accessible", "Emergency contacts posted", "Supervised bathing if indicated"]},
    ]

    c.close()
    return {
        "seizure_fall_analysis": {
            "total_seizure_events": total_seizures,
            "events_with_injury": fall_count,
            "fall_rate_pct": round(100 * fall_count / total_seizures, 1) if total_seizures else 0,
            "er_visits_from_falls": er_from_falls,
            "injury_details": [
                {"patient_id": f["patient_id"], "date": f["event_date"],
                 "injury": f["injury"], "severity": f["severity"], "er_visit": f.get("er_visit")}
                for f in fall_events
            ],
        },
        "adl_mobility_risks": {
            "patients_with_mobility_impairment": len(mobility_risks),
            "details": sorted(mobility_risks, key=lambda r: r["barthel_total"]),
        },
        "medication_sedation_risks": {
            "patients_with_sedating_meds": len(med_risks),
            "details": med_risks,
        },
        "home_safety_checklist": home_safety_checklist,
        "clinical_flags": _fall_risk_flags(fall_count, total_seizures, mobility_risks, med_risks),
    }


def _fall_risk_flags(fall_count, seizure_count, mobility_risks, med_risks):
    flags = []
    if fall_count >= 2:
        flags.append({"flag": "Recurrent seizure-related falls", "severity": "high",
                       "action": "Urgent fall-prevention program + helmet assessment"})
    if seizure_count > 0 and fall_count / max(seizure_count, 1) > 0.3:
        flags.append({"flag": "High injury rate per seizure (>30%)", "severity": "high",
                       "action": "Review seizure type — consider protective equipment"})
    for mr in mobility_risks:
        if mr["barthel_total"] <= 60:
            flags.append({"flag": f"Patient {mr['patient_id']} severe ADL dependence",
                          "severity": "high", "action": "24-hour supervision + home modification assessment"})
    for mr in med_risks:
        if mr["combined_sedation_risk"] == "High":
            flags.append({"flag": f"Patient {mr['patient_id']} on high-sedation medication",
                          "severity": "moderate", "action": "Review timing of sedating meds; fall precautions"})
        if mr["polypharmacy_flag"]:
            flags.append({"flag": f"Patient {mr['patient_id']} polypharmacy (≥3 sedating meds)",
                          "severity": "high", "action": "Pharmacist review for sedation burden reduction"})
    if not flags:
        flags.append({"flag": "No high-priority fall risk flags", "severity": "low",
                       "action": "Continue routine fall prevention counseling"})
    return flags


# ─── 3. Return-to-Work Planner ──────────────────────────────────────────

def return_to_work(patient_id=None):
    """Vocational readiness: functional independence + seizure control + cognition composite."""
    c = _conn()
    params = (patient_id,) if patient_id else ()

    # BARTHEL for functional independence
    where_b = "WHERE instrument='BARTHEL'" + (" AND patient_id = ?" if patient_id else "")
    cur = c.execute(f"SELECT patient_id, score, created_at FROM assessments {where_b} ORDER BY created_at DESC", params)
    barthel = _rows_as_dicts(cur)
    barthel_latest = {}
    for r in barthel:
        if r["patient_id"] not in barthel_latest:
            barthel_latest[r["patient_id"]] = r

    # Seizure frequency per patient (from diary)
    where_sz = "" if not patient_id else "WHERE patient_id = ?"
    cur = c.execute(f"SELECT patient_id, event_date, severity FROM seizure_diary {where_sz}", params)
    seizures = _rows_as_dicts(cur)
    seizure_counts = defaultdict(int)
    recent_seizures = defaultdict(int)  # last 90 days proxy
    for s in seizures:
        seizure_counts[s["patient_id"]] += 1
        recent_seizures[s["patient_id"]] += 1  # all events treated as recent for this dataset

    # Cognitive scores (MOCA + MMSE)
    cognitive = {}
    for inst in ["MOCA", "MMSE"]:
        where_c = f"WHERE instrument='{inst}'" + (" AND patient_id = ?" if patient_id else "")
        cur = c.execute(f"SELECT patient_id, score, max_score FROM assessments {where_c} ORDER BY created_at DESC", params)
        for r in _rows_as_dicts(cur):
            pid = r["patient_id"]
            if pid not in cognitive:
                cognitive[pid] = {}
            if inst not in cognitive[pid]:
                cognitive[pid][inst] = r["score"]

    # Build per-patient RTW assessment
    all_pids = set(barthel_latest) | set(seizure_counts) | set(cognitive)
    rtw_profiles = []

    for pid in all_pids:
        b_score = barthel_latest.get(pid, {}).get("score", None)
        sz_count = seizure_counts.get(pid, 0)
        moca = cognitive.get(pid, {}).get("MOCA", None)
        mmse = cognitive.get(pid, {}).get("MMSE", None)

        # Functional readiness (0-100 scale)
        func_score = b_score if b_score is not None else None

        # Seizure control score (0-100): 0 seizures=100, each seizure reduces
        sz_control = max(0, 100 - sz_count * 20) if sz_count is not None else None

        # Cognitive readiness (0-100): MOCA ≥26 or MMSE ≥26 = full
        cog_score = None
        if moca is not None:
            cog_score = round(min(100, moca / 26 * 100), 1)
        elif mmse is not None:
            cog_score = round(min(100, mmse / 26 * 100), 1)

        # Composite RTW readiness (weighted: 40% functional + 30% seizure control + 30% cognitive)
        components = []
        if func_score is not None:
            components.append(("functional", func_score, 0.4))
        if sz_control is not None:
            components.append(("seizure_control", sz_control, 0.3))
        if cog_score is not None:
            components.append(("cognitive", cog_score, 0.3))

        if components:
            total_weight = sum(w for _, _, w in components)
            composite = round(sum(s * w for _, s, w in components) / total_weight, 1)
        else:
            composite = None

        # RTW category
        if composite is not None:
            if composite >= 80:
                rtw_category = "Ready for competitive employment"
            elif composite >= 60:
                rtw_category = "Supported/modified employment feasible"
            elif composite >= 40:
                rtw_category = "Vocational rehabilitation recommended"
            else:
                rtw_category = "Not currently work-ready; focus on rehabilitation"
        else:
            rtw_category = "Insufficient data for assessment"

        # Workplace accommodations
        accommodations = _rtw_accommodations(func_score, sz_count, cog_score)

        rtw_profiles.append({
            "patient_id": pid,
            "functional_score": func_score,
            "seizure_control_score": sz_control,
            "seizure_count_recent": sz_count,
            "cognitive_score": cog_score,
            "moca": moca,
            "mmse": mmse,
            "composite_rtw_score": composite,
            "rtw_category": rtw_category,
            "recommended_accommodations": accommodations,
        })

    rtw_profiles.sort(key=lambda p: p["composite_rtw_score"] or 0, reverse=True)

    # Cohort summary
    composites = [p["composite_rtw_score"] for p in rtw_profiles if p["composite_rtw_score"] is not None]
    cat_dist = defaultdict(int)
    for p in rtw_profiles:
        cat_dist[p["rtw_category"]] += 1

    c.close()
    return {
        "assessment": "Return-to-Work Readiness (OT composite: 40% functional + 30% seizure control + 30% cognitive)",
        "unique_patients": len(rtw_profiles),
        "cohort_summary": {
            "mean_rtw_score": round(sum(composites) / len(composites), 1) if composites else None,
            "category_distribution": [{"category": k, "count": v} for k, v in sorted(cat_dist.items())],
        },
        "patient_profiles": rtw_profiles,
    }


def _rtw_accommodations(func_score, sz_count, cog_score):
    """Evidence-based workplace accommodation recommendations."""
    accom = []
    if sz_count and sz_count > 0:
        accom.append("Seizure action plan on file with employer/HR")
        accom.append("Avoid heights, heavy machinery, driving (per seizure type/frequency)")
        if sz_count >= 3:
            accom.append("Consider work-from-home or flexible scheduling")
    if func_score is not None and func_score < 80:
        accom.append("Ergonomic workstation assessment")
        accom.append("Rest breaks for fatigue management")
    if func_score is not None and func_score < 60:
        accom.append("Physical accessibility (elevator, accessible restroom)")
    if cog_score is not None and cog_score < 80:
        accom.append("Written task instructions / checklists")
        accom.append("Reduced multitasking demands")
        accom.append("Extended time for complex tasks")
    if cog_score is not None and cog_score < 60:
        accom.append("Job coaching or supported employment services")
    if not accom:
        accom.append("Standard workplace; no special accommodations needed at this time")
    return accom


# ─── 4. Cognitive-Function OT ────────────────────────────────────────────

def cognitive_function_ot(patient_id=None):
    """MOCA + MMSE with OT-specific domain interpretation and intervention mapping."""
    c = _conn()
    params = (patient_id,) if patient_id else ()

    profiles = {}

    for inst, max_score, domains_map in [
        ("MOCA", 30, {
            "Visuospatial/Executive": {"items": ["item1"], "max": 5, "ot_relevance": "Driving, meal prep sequencing, financial management"},
            "Naming": {"items": ["item2"], "max": 3, "ot_relevance": "Communication during ADLs, safety instructions"},
            "Attention": {"items": ["item3"], "max": 6, "ot_relevance": "Medication management, cooking safety, work tasks"},
            "Language": {"items": ["item4"], "max": 3, "ot_relevance": "Following multi-step instructions for home/work tasks"},
            "Abstraction": {"items": ["item5"], "max": 2, "ot_relevance": "Problem-solving in novel situations, community navigation"},
            "Delayed Recall": {"items": ["item6"], "max": 5, "ot_relevance": "Remembering appointments, medication schedules, safety procedures"},
            "Orientation": {"items": ["item7"], "max": 6, "ot_relevance": "Community mobility, time management, scheduling"},
        }),
        ("MMSE", 30, {
            "Orientation (Time)": {"items": ["item1"], "max": 5, "ot_relevance": "Scheduling, appointment keeping, routine management"},
            "Orientation (Place)": {"items": ["item2"], "max": 5, "ot_relevance": "Community navigation, safety in unfamiliar environments"},
            "Registration": {"items": ["item3"], "max": 3, "ot_relevance": "Learning new tasks, adapting to new equipment"},
            "Attention/Calculation": {"items": ["item4"], "max": 5, "ot_relevance": "Financial management, meal planning, work calculations"},
            "Recall": {"items": ["item5"], "max": 3, "ot_relevance": "Medication management, daily routine adherence"},
            "Language/Praxis": {"items": ["item6", "item7", "item8", "item9"], "max": 9, "ot_relevance": "Following instructions, tool use, written communication"},
        }),
    ]:
        where = f"WHERE instrument='{inst}'" + (" AND patient_id = ?" if patient_id else "")
        cur = c.execute(
            f"SELECT patient_id, answers_json, score, max_score, created_at "
            f"FROM assessments {where} ORDER BY created_at DESC", params
        )
        rows = _rows_as_dicts(cur)
        for r in rows:
            pid = r["patient_id"]
            if pid not in profiles:
                profiles[pid] = {"patient_id": pid, "assessments": {}}
            if inst not in profiles[pid]["assessments"]:
                total = r["score"] or 0
                answers = json.loads(r["answers_json"]) if r["answers_json"] else {}

                # Cognitive impairment level
                if inst == "MOCA":
                    if total >= 26:
                        level = "Normal"
                    elif total >= 18:
                        level = "Mild Cognitive Impairment"
                    elif total >= 10:
                        level = "Moderate Cognitive Impairment"
                    else:
                        level = "Severe Cognitive Impairment"
                else:  # MMSE
                    if total >= 26:
                        level = "Normal"
                    elif total >= 20:
                        level = "Mild Cognitive Impairment"
                    elif total >= 10:
                        level = "Moderate Cognitive Impairment"
                    else:
                        level = "Severe Cognitive Impairment"

                # Per-domain breakdown with OT mapping
                domain_results = []
                impaired_domains = []
                for dom_name, dom_info in domains_map.items():
                    # Sum relevant items
                    dom_score = 0
                    for item_key in dom_info["items"]:
                        val = answers.get(item_key, 0)
                        dom_score += float(val) if val is not None else 0
                    dom_max = dom_info["max"]
                    pct = round(100 * dom_score / dom_max, 1) if dom_max else 0
                    impaired = dom_score < dom_max * 0.8  # <80% of max = impaired
                    if impaired:
                        impaired_domains.append(dom_name)
                    domain_results.append({
                        "domain": dom_name,
                        "score": dom_score,
                        "max": dom_max,
                        "pct": pct,
                        "impaired": impaired,
                        "ot_relevance": dom_info["ot_relevance"],
                    })

                # OT cognitive interventions
                interventions = _cognitive_interventions(impaired_domains, level, inst)

                profiles[pid]["assessments"][inst] = {
                    "total_score": total,
                    "max_score": max_score,
                    "level": level,
                    "domains": domain_results,
                    "impaired_domains": impaired_domains,
                    "ot_interventions": interventions,
                    "assessed_at": r["created_at"],
                }

    # Cohort summary
    patient_list = list(profiles.values())
    moca_scores = [p["assessments"]["MOCA"]["total_score"] for p in patient_list if "MOCA" in p["assessments"]]
    mmse_scores = [p["assessments"]["MMSE"]["total_score"] for p in patient_list if "MMSE" in p["assessments"]]

    c.close()
    return {
        "assessment": "Cognitive-Function OT (MOCA + MMSE with OT-specific domain mapping)",
        "unique_patients": len(patient_list),
        "cohort_summary": {
            "moca": {
                "n": len(moca_scores),
                "mean": round(sum(moca_scores) / len(moca_scores), 1) if moca_scores else None,
                "min": min(moca_scores) if moca_scores else None,
                "max": max(moca_scores) if moca_scores else None,
            },
            "mmse": {
                "n": len(mmse_scores),
                "mean": round(sum(mmse_scores) / len(mmse_scores), 1) if mmse_scores else None,
                "min": min(mmse_scores) if mmse_scores else None,
                "max": max(mmse_scores) if mmse_scores else None,
            },
        },
        "patient_profiles": patient_list,
    }


def _cognitive_interventions(impaired_domains, level, instrument):
    """OT-specific cognitive rehabilitation interventions."""
    interventions = []
    domain_interventions = {
        "Visuospatial/Executive": "Visual scanning exercises, route-planning practice, meal-prep sequencing drills",
        "Naming": "Word-finding strategies (circumlocution, categorization); coordinate with SLP",
        "Attention": "Graded attention tasks, distraction-free environment setup, timer-based task switching",
        "Language": "Simplified written instructions, step-by-step visual guides for ADLs",
        "Abstraction": "Structured problem-solving worksheets (D'Zurilla model), categorization exercises",
        "Delayed Recall": "External memory aids (smartphone alarms, pill organizers, wall calendars), spaced retrieval training",
        "Orientation": "Orientation board at home, GPS apps for community mobility, structured daily schedule",
        "Orientation (Time)": "Clock placement, daily schedule board, alarm-based routine cues",
        "Orientation (Place)": "Familiar route practice, landmark-based navigation, GPS aids",
        "Registration": "Repetition-based learning, multimodal encoding (see + say + do)",
        "Attention/Calculation": "Graded calculation tasks, bill-paying practice, budgeting worksheets",
        "Recall": "Spaced retrieval training, errorless learning for medication routine",
        "Language/Praxis": "Motor sequencing tasks, gesture rehearsal, written instruction boards",
    }
    for dom in impaired_domains:
        if dom in domain_interventions:
            interventions.append({"domain": dom, "intervention": domain_interventions[dom]})

    if level in ("Moderate Cognitive Impairment", "Severe Cognitive Impairment"):
        interventions.append({"domain": "General", "intervention": "Comprehensive cognitive rehabilitation program; consider supervised living arrangement"})
        interventions.append({"domain": "Safety", "intervention": "Home safety evaluation for unsupervised living; stove/appliance auto-shutoff"})
    elif level == "Mild Cognitive Impairment":
        interventions.append({"domain": "General", "intervention": "Compensatory strategy training; external memory aids; energy conservation"})

    return interventions


# ─── Full Dashboard ──────────────────────────────────────────────────────

def full_dashboard(patient_id=None):
    """Combined OT dashboard — all 4 modules for the patient(s)."""
    return {
        "role": "Occupational Therapist (OT)",
        "description": "Functional assessment, home safety, vocational planning, and cognitive rehabilitation for epilepsy patients",
        "modules": {
            "adl_assessment": adl_assessment(patient_id),
            "fall_risk": fall_risk_assessment(patient_id),
            "return_to_work": return_to_work(patient_id),
            "cognitive_function": cognitive_function_ot(patient_id),
        },
    }
