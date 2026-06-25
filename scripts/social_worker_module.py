"""
Medical Social Worker (MSW) Module — Epilepsy Psychosocial Support
===================================================================
Real analytics from patients, seizure_diary, medications, and assessments
tables in data/clinical.db.

Endpoints:
  /api/social-worker                      — full dashboard (all 4 sub-analyses)
  /api/social-worker/sdoh-screening       — Social Determinants of Health screening
  /api/social-worker/caregiver-burden     — ZBI/CSI proxy caregiver burden scoring
  /api/social-worker/benefits-vocational  — Benefits eligibility + vocational rehab
  /api/social-worker/treatment-barriers   — Treatment barrier detection from social factors

All data from REAL patients, medications, seizure_diary, and assessments
in data/clinical.db.

Evidence base:
  - Jacoby A, Baker GA (2008). Quality-of-life trajectories in epilepsy. Epilepsia.
  - Epilepsy Foundation (2023). Social Determinants of Health screening toolkit.
  - Zarit SH et al. (1980). Relatives of the impaired elderly: caregiver burden. Gerontologist.
  - Robinson BC (1983). Caregiver Strain Index. J Gerontol.
  - WHO (2019). Social determinants of health: the solid facts.
  - Cotterman-Hart S (2010). Employment issues in epilepsy. Neurol Clin.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# ── ASM side-effect / pregnancy knowledge (subset for social-work context) ──
ASM_SIDE_EFFECTS = {
    "Levetiracetam":  {"cognitive": "low",  "sedation": "moderate", "mood": "irritability/aggression", "pregnancy_cat": "C", "weight": "neutral"},
    "Lamotrigine":    {"cognitive": "low",  "sedation": "low",      "mood": "mood-stabilizing",        "pregnancy_cat": "C", "weight": "neutral"},
    "Valproate":      {"cognitive": "moderate", "sedation": "moderate", "mood": "mood-stabilizing",    "pregnancy_cat": "X", "weight": "gain"},
    "Carbamazepine":  {"cognitive": "moderate", "sedation": "moderate", "mood": "mood-stabilizing",    "pregnancy_cat": "D", "weight": "gain"},
    "Phenytoin":      {"cognitive": "high", "sedation": "moderate", "mood": "neutral",                 "pregnancy_cat": "D", "weight": "neutral"},
    "Oxcarbazepine":  {"cognitive": "low",  "sedation": "moderate", "mood": "neutral",                 "pregnancy_cat": "C", "weight": "neutral"},
    "Topiramate":     {"cognitive": "high", "sedation": "moderate", "mood": "depression risk",         "pregnancy_cat": "D", "weight": "loss"},
    "Zonisamide":     {"cognitive": "moderate", "sedation": "moderate", "mood": "neutral",             "pregnancy_cat": "C", "weight": "loss"},
    "Lacosamide":     {"cognitive": "low",  "sedation": "low",      "mood": "neutral",                 "pregnancy_cat": "C", "weight": "neutral"},
    "Clobazam":       {"cognitive": "moderate", "sedation": "high",  "mood": "aggression risk",        "pregnancy_cat": "C", "weight": "gain"},
    "Brivaracetam":   {"cognitive": "low",  "sedation": "moderate", "mood": "neutral",                 "pregnancy_cat": "C", "weight": "neutral"},
    "Perampanel":     {"cognitive": "moderate", "sedation": "moderate", "mood": "aggression risk",     "pregnancy_cat": "C", "weight": "gain"},
    "Phenobarbital":  {"cognitive": "high", "sedation": "high",     "mood": "depression risk",         "pregnancy_cat": "D", "weight": "gain"},
    "Ethosuximide":   {"cognitive": "low",  "sedation": "low",      "mood": "neutral",                 "pregnancy_cat": "C", "weight": "neutral"},
    "Pregabalin":     {"cognitive": "moderate", "sedation": "moderate", "mood": "neutral",             "pregnancy_cat": "C", "weight": "gain"},
    "Gabapentin":     {"cognitive": "low",  "sedation": "moderate", "mood": "neutral",                 "pregnancy_cat": "C", "weight": "gain"},
}

# ── SDOH domain scoring weights ──────────────────────────────────────────
SDOH_DOMAINS = [
    "Employment",
    "Housing",
    "Transportation",
    "Financial",
    "Social Support",
    "Education",
]

# ── Driving seizure-free period requirements (months) by jurisdiction ────
DRIVING_SEIZURE_FREE_MONTHS = {
    "most_US_states": 6,
    "strict_states": 12,
    "commercial_license": 24,
}


def _connect():
    """Return sqlite3 connection with Row factory, or None if DB missing."""
    if not DB.exists():
        return None
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_drug(name: str) -> str:
    """Normalize drug name to canonical form (case-insensitive)."""
    name = name.strip()
    for canonical in ASM_SIDE_EFFECTS:
        if name.lower() == canonical.lower():
            return canonical
    return name


def _get_patients(conn, patient_id=None):
    """Fetch patient rows as list of dicts."""
    if patient_id:
        rows = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM patients ORDER BY patient_id").fetchall()
    return [dict(r) for r in rows]


def _get_seizure_events(conn, patient_id=None):
    """Fetch seizure diary entries grouped by patient_id."""
    if patient_id:
        rows = conn.execute(
            "SELECT * FROM seizure_diary WHERE patient_id = ? ORDER BY event_date",
            (patient_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM seizure_diary ORDER BY patient_id, event_date"
        ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(dict(r))
    return by_patient


def _get_medications(conn, patient_id=None):
    """Fetch medication records grouped by patient_id, parsing fields_json."""
    if patient_id:
        rows = conn.execute(
            "SELECT * FROM medications WHERE patient_id = ? ORDER BY created_at",
            (patient_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM medications ORDER BY patient_id, created_at"
        ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
        drug_name = _normalize_drug(fields.get("drug_name", "Unknown"))
        drugs = [drug_name]
        for a in fields.get("aed", []):
            drugs.append(_normalize_drug(a))
        by_patient[r["patient_id"]].append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "drug_name": drug_name,
            "dose_mg": fields.get("dose_mg"),
            "frequency": fields.get("frequency", "Unknown"),
            "all_drugs": list(set(drugs)),
            "created_at": r["created_at"],
        })
    return by_patient


def _unique_drugs_for_patient(med_records):
    """Return deduplicated set of drug names from a patient's medication records."""
    drugs = set()
    for m in med_records:
        for d in m["all_drugs"]:
            drugs.add(d)
    return drugs


# ════════════════════════════════════════════════════════════════════════
# 1. Social Determinants of Health (SDOH) Screening
# ════════════════════════════════════════════════════════════════════════

def social_determinants_screening(patient_id: str = None):
    """
    Screen each patient across 6 SDOH domains using evidence-based risk
    factors derived from real demographic, seizure, and medication data.

    Scoring per domain (0-10 scale):
      Employment:      seizure frequency, cognitive side effects, driving inability
      Housing:         age vulnerability (<18 or >65), injury history, nocturnal seizures
      Transportation:  seizure frequency → driving restriction, rural/access proxy
      Financial:       polypharmacy cost, disability status proxy, medication count
      Social Support:  seizure severity, ER visits, caregiver indicators
      Education:       age <25 (educational disruption), cognitive drug load

    Composite vulnerability score = mean of all 6 domains scaled to 0-100.
    """
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found", "results": []}

    patients = _get_patients(conn, patient_id)
    seizures = _get_seizure_events(conn, patient_id)
    meds = _get_medications(conn, patient_id)
    conn.close()

    results = []
    for pt in patients:
        pid = pt["patient_id"]
        age = pt.get("age") or 30
        sz_events = seizures.get(pid, [])
        med_records = meds.get(pid, [])
        drugs = _unique_drugs_for_patient(med_records)
        sz_count = len(sz_events)

        # Injury & ER history from diary
        injury_count = sum(
            1 for e in sz_events
            if e.get("injury") and e["injury"] not in ("No", "None", None, "")
        )
        er_count = sum(1 for e in sz_events if e.get("er_visit") == "Yes")
        nocturnal = sum(
            1 for e in sz_events
            if e.get("event_time") and _is_nocturnal(e["event_time"])
        )
        severe_count = sum(
            1 for e in sz_events
            if e.get("severity") and e["severity"].lower() in ("severe", "high", "status")
        )

        # Cognitive drug burden
        cognitive_burden = sum(
            1 for d in drugs
            if ASM_SIDE_EFFECTS.get(d, {}).get("cognitive") in ("high", "moderate")
        )

        # ── Domain scoring ──
        # Employment (0-10)
        emp = 0
        if sz_count >= 4:
            emp += 4
        elif sz_count >= 2:
            emp += 2
        elif sz_count >= 1:
            emp += 1
        if cognitive_burden >= 2:
            emp += 3
        elif cognitive_burden >= 1:
            emp += 2
        if sz_count > 0:
            emp += 2  # driving restriction impacts employment
        if age < 18 or age > 65:
            emp += 1
        emp = min(emp, 10)

        # Housing (0-10)
        hou = 0
        if age < 18:
            hou += 2
        elif age > 65:
            hou += 3
        if injury_count >= 2:
            hou += 3
        elif injury_count >= 1:
            hou += 2
        if nocturnal >= 2:
            hou += 3
        elif nocturnal >= 1:
            hou += 2
        if severe_count >= 2:
            hou += 2
        hou = min(hou, 10)

        # Transportation (0-10)
        tra = 0
        if sz_count > 0:
            tra += 4  # cannot drive
        if sz_count >= 4:
            tra += 3  # heavy transport dependency
        elif sz_count >= 2:
            tra += 2
        if age < 18 or age > 70:
            tra += 2
        if nocturnal >= 1:
            tra += 1
        tra = min(tra, 10)

        # Financial (0-10)
        fin = 0
        drug_count = len(drugs)
        if drug_count >= 4:
            fin += 4
        elif drug_count >= 3:
            fin += 3
        elif drug_count >= 2:
            fin += 2
        elif drug_count >= 1:
            fin += 1
        if sz_count >= 4:
            fin += 2  # disability/lost wages proxy
        if er_count >= 2:
            fin += 2
        elif er_count >= 1:
            fin += 1
        if age > 65:
            fin += 1
        fin = min(fin, 10)

        # Social Support (0-10)
        soc = 0
        if severe_count >= 2:
            soc += 3
        elif severe_count >= 1:
            soc += 2
        if er_count >= 2:
            soc += 2
        elif er_count >= 1:
            soc += 1
        if injury_count >= 1:
            soc += 2
        if nocturnal >= 1:
            soc += 2
        if sz_count >= 4:
            soc += 1
        soc = min(soc, 10)

        # Education (0-10)
        edu = 0
        if age < 25:
            edu += 3  # educational disruption risk
        if cognitive_burden >= 2:
            edu += 3
        elif cognitive_burden >= 1:
            edu += 2
        if sz_count >= 4:
            edu += 2
        elif sz_count >= 2:
            edu += 1
        if severe_count >= 1:
            edu += 2
        edu = min(edu, 10)

        domain_scores = {
            "Employment": emp,
            "Housing": hou,
            "Transportation": tra,
            "Financial": fin,
            "Social Support": soc,
            "Education": edu,
        }
        composite = round(sum(domain_scores.values()) / 6 * 10, 1)

        if composite >= 60:
            priority = "high"
        elif composite >= 35:
            priority = "moderate"
        else:
            priority = "low"

        # Referral recommendations
        referrals = []
        if emp >= 6:
            referrals.append("Vocational rehabilitation referral")
        if hou >= 6:
            referrals.append("Independent living / assisted housing assessment")
        if tra >= 6:
            referrals.append("Transportation assistance program enrollment")
        if fin >= 6:
            referrals.append("Financial counseling + medication assistance program")
        if soc >= 6:
            referrals.append("Epilepsy support group + peer mentoring")
        if edu >= 6:
            referrals.append("Educational accommodation / IEP / 504 plan review")
        if composite >= 60:
            referrals.append("Comprehensive social work case management")

        results.append({
            "patient_id": pid,
            "name": pt.get("name", ""),
            "age": age,
            "gender": pt.get("gender", ""),
            "seizure_count": sz_count,
            "medication_count": drug_count,
            "domain_scores": domain_scores,
            "vulnerability_score": composite,
            "priority_level": priority,
            "referral_recommendations": referrals,
        })

    # Summary
    high_count = sum(1 for r in results if r["priority_level"] == "high")
    mod_count = sum(1 for r in results if r["priority_level"] == "moderate")
    avg_vuln = round(sum(r["vulnerability_score"] for r in results) / len(results), 1) if results else 0

    return {
        "analysis": "Social Determinants of Health (SDOH) Screening",
        "total_patients_screened": len(results),
        "summary": {
            "high_priority": high_count,
            "moderate_priority": mod_count,
            "low_priority": len(results) - high_count - mod_count,
            "mean_vulnerability_score": avg_vuln,
        },
        "results": results,
    }


def _is_nocturnal(time_str):
    """Check if a time string represents nocturnal hours (22:00-06:00)."""
    try:
        hour = int(time_str.split(":")[0])
        return hour >= 22 or hour < 6
    except (ValueError, IndexError, AttributeError):
        return False


# ════════════════════════════════════════════════════════════════════════
# 2. Caregiver Burden Assessment (ZBI / CSI proxy)
# ════════════════════════════════════════════════════════════════════════

def caregiver_burden(patient_id: str = None):
    """
    Approximate caregiver burden using the Zarit Burden Interview (ZBI, 0-88)
    and Caregiver Strain Index (CSI, 0-13) proxy scores derived from real
    clinical data: seizure frequency, severity, nocturnal events, injury
    history, medication complexity, and patient age.

    ZBI proxy scoring rationale (summed components, max 88):
      - Seizure frequency:      0-20 points (scaled by count)
      - Seizure severity:       0-15 points
      - Nocturnal seizures:     0-12 points (sleep disruption)
      - Injury history:         0-10 points
      - Medication complexity:  0-10 points (polypharmacy burden)
      - Patient age factor:     0-10 points (<18 or >65 = higher)
      - ER visit burden:        0-11 points

    CSI proxy (0-13, binary strain indicators):
      Each indicator scored 0 or 1 based on objective data thresholds.
    """
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found", "results": []}

    patients = _get_patients(conn, patient_id)
    seizures = _get_seizure_events(conn, patient_id)
    meds = _get_medications(conn, patient_id)
    conn.close()

    results = []
    for pt in patients:
        pid = pt["patient_id"]
        age = pt.get("age") or 30
        sz_events = seizures.get(pid, [])
        med_records = meds.get(pid, [])
        drugs = _unique_drugs_for_patient(med_records)
        sz_count = len(sz_events)

        injury_count = sum(
            1 for e in sz_events
            if e.get("injury") and e["injury"] not in ("No", "None", None, "")
        )
        er_count = sum(1 for e in sz_events if e.get("er_visit") == "Yes")
        nocturnal = sum(
            1 for e in sz_events
            if e.get("event_time") and _is_nocturnal(e["event_time"])
        )
        severe_count = sum(
            1 for e in sz_events
            if e.get("severity") and e["severity"].lower() in ("severe", "high", "status")
        )
        drug_count = len(drugs)

        # ── ZBI proxy (0-88) ──
        # Seizure frequency (0-20)
        zbi_freq = min(sz_count * 4, 20)
        # Seizure severity (0-15)
        zbi_sev = min(severe_count * 5, 15)
        # Nocturnal (0-12)
        zbi_noct = min(nocturnal * 4, 12)
        # Injury (0-10)
        zbi_inj = min(injury_count * 5, 10)
        # Medication complexity (0-10)
        zbi_med = min(drug_count * 3, 10)
        # Age factor (0-10)
        zbi_age = 0
        if age < 18:
            zbi_age = 8
        elif age < 10:
            zbi_age = 10
        elif age > 75:
            zbi_age = 8
        elif age > 65:
            zbi_age = 5
        # ER visit burden (0-11)
        zbi_er = min(er_count * 4, 11)

        zbi_total = min(
            zbi_freq + zbi_sev + zbi_noct + zbi_inj + zbi_med + zbi_age + zbi_er,
            88,
        )

        if zbi_total >= 61:
            zbi_level = "severe"
        elif zbi_total >= 41:
            zbi_level = "moderate-to-severe"
        elif zbi_total >= 21:
            zbi_level = "mild-to-moderate"
        else:
            zbi_level = "little-or-none"

        # ── CSI proxy (0-13) ──
        csi_items = {
            "sleep_disrupted": 1 if nocturnal >= 1 else 0,
            "inconvenience": 1 if sz_count >= 3 else 0,
            "physical_strain": 1 if injury_count >= 1 else 0,
            "confining": 1 if sz_count >= 4 else 0,
            "family_adjustments": 1 if sz_count >= 2 else 0,
            "change_personal_plans": 1 if er_count >= 1 else 0,
            "emotional_adjustments": 1 if severe_count >= 1 else 0,
            "upsetting_behavior": 1 if any(
                ASM_SIDE_EFFECTS.get(d, {}).get("mood", "").find("aggression") >= 0
                or ASM_SIDE_EFFECTS.get(d, {}).get("mood", "").find("irritability") >= 0
                for d in drugs
            ) else 0,
            "patient_changed": 1 if severe_count >= 2 else 0,
            "work_adjustments": 1 if sz_count >= 3 else 0,
            "financial_strain": 1 if drug_count >= 3 else 0,
            "overwhelmed": 1 if zbi_total >= 41 else 0,
            "constant_worry": 1 if sz_count >= 2 or severe_count >= 1 else 0,
        }
        csi_total = sum(csi_items.values())

        if csi_total >= 10:
            csi_level = "high-strain"
        elif csi_total >= 7:
            csi_level = "moderate-strain"
        else:
            csi_level = "low-strain"

        # Burnout risk
        if zbi_total >= 61 or csi_total >= 10:
            burnout_risk = "high"
        elif zbi_total >= 41 or csi_total >= 7:
            burnout_risk = "moderate"
        else:
            burnout_risk = "low"

        # Support recommendations
        recommendations = []
        if nocturnal >= 1:
            recommendations.append("Seizure monitoring device (bed sensor/wearable) for overnight safety")
        if csi_items["sleep_disrupted"]:
            recommendations.append("Caregiver sleep management counseling")
        if zbi_total >= 41:
            recommendations.append("Respite care referral — minimum 4 hours/week")
        if csi_total >= 7:
            recommendations.append("Caregiver support group enrollment")
        if burnout_risk == "high":
            recommendations.append("Caregiver mental health screening (PHQ-9 / GAD-7)")
            recommendations.append("Social work case management for family support")
        if csi_items["financial_strain"]:
            recommendations.append("Medication assistance program + insurance navigation")
        if injury_count >= 2:
            recommendations.append("Home safety assessment referral")

        respite_flag = zbi_total >= 41 or csi_total >= 7

        results.append({
            "patient_id": pid,
            "name": pt.get("name", ""),
            "age": age,
            "seizure_count": sz_count,
            "zbi_proxy_score": zbi_total,
            "zbi_level": zbi_level,
            "zbi_components": {
                "frequency": zbi_freq,
                "severity": zbi_sev,
                "nocturnal": zbi_noct,
                "injury": zbi_inj,
                "medication_complexity": zbi_med,
                "age_factor": zbi_age,
                "er_visits": zbi_er,
            },
            "csi_proxy_score": csi_total,
            "csi_level": csi_level,
            "csi_items": csi_items,
            "burnout_risk_level": burnout_risk,
            "support_recommendations": recommendations,
            "respite_referral_flag": respite_flag,
        })

    # Summary
    severe_burden = sum(1 for r in results if r["zbi_level"] == "severe")
    mod_severe = sum(1 for r in results if r["zbi_level"] == "moderate-to-severe")
    high_burnout = sum(1 for r in results if r["burnout_risk_level"] == "high")
    respite_needed = sum(1 for r in results if r["respite_referral_flag"])

    return {
        "analysis": "Caregiver Burden Assessment (ZBI/CSI proxy)",
        "total_patients_assessed": len(results),
        "summary": {
            "severe_burden": severe_burden,
            "moderate_to_severe_burden": mod_severe,
            "high_burnout_risk": high_burnout,
            "respite_referrals_needed": respite_needed,
            "mean_zbi": round(sum(r["zbi_proxy_score"] for r in results) / len(results), 1) if results else 0,
            "mean_csi": round(sum(r["csi_proxy_score"] for r in results) / len(results), 1) if results else 0,
        },
        "results": results,
    }


# ════════════════════════════════════════════════════════════════════════
# 3. Benefits & Vocational Assessment
# ════════════════════════════════════════════════════════════════════════

def benefits_vocational(patient_id: str = None):
    """
    Assess employment readiness, driving eligibility, disability benefit
    flags, and vocational rehabilitation needs for each patient.

    Evidence-based criteria:
      - Driving: most US states require 3-12 month seizure-free period
      - Disability (SSA): ≥1 seizure/month despite adherence, or significant
        functional limitation from seizures/medication side effects
      - Employment readiness: composite of seizure control + cognitive burden
        + side effect profile + age-appropriate expectations
    """
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found", "results": []}

    patients = _get_patients(conn, patient_id)
    seizures = _get_seizure_events(conn, patient_id)
    meds = _get_medications(conn, patient_id)
    conn.close()

    results = []
    for pt in patients:
        pid = pt["patient_id"]
        age = pt.get("age") or 30
        sz_events = seizures.get(pid, [])
        med_records = meds.get(pid, [])
        drugs = _unique_drugs_for_patient(med_records)
        sz_count = len(sz_events)
        drug_count = len(drugs)

        severe_count = sum(
            1 for e in sz_events
            if e.get("severity") and e["severity"].lower() in ("severe", "high", "status")
        )
        injury_count = sum(
            1 for e in sz_events
            if e.get("injury") and e["injury"] not in ("No", "None", None, "")
        )

        # Cognitive drug burden score (0-10)
        cognitive_load = 0
        sedation_load = 0
        pregnancy_risk_drugs = []
        mood_risk_drugs = []
        for d in drugs:
            info = ASM_SIDE_EFFECTS.get(d, {})
            cog = info.get("cognitive", "low")
            sed = info.get("sedation", "low")
            if cog == "high":
                cognitive_load += 3
            elif cog == "moderate":
                cognitive_load += 2
            if sed == "high":
                sedation_load += 3
            elif sed == "moderate":
                sedation_load += 1
            if info.get("pregnancy_cat") in ("X", "D"):
                pregnancy_risk_drugs.append(d)
            mood = info.get("mood", "neutral")
            if "risk" in mood or "irritability" in mood or "aggression" in mood:
                mood_risk_drugs.append(d)

        # ── Driving eligibility estimate ──
        # Check most recent seizure date to estimate seizure-free period
        last_seizure_date = None
        if sz_events:
            dates = [e["event_date"] for e in sz_events if e.get("event_date")]
            if dates:
                last_seizure_date = max(dates)

        driving_eligibility = {}
        if last_seizure_date:
            try:
                last_dt = datetime.strptime(last_seizure_date, "%Y-%m-%d")
                now = datetime.now()
                months_free = max(0, (now.year - last_dt.year) * 12 + (now.month - last_dt.month))
            except (ValueError, TypeError):
                months_free = 0

            driving_eligibility = {
                "last_seizure_date": last_seizure_date,
                "months_seizure_free": months_free,
                "standard_license_eligible": months_free >= DRIVING_SEIZURE_FREE_MONTHS["most_US_states"],
                "strict_state_eligible": months_free >= DRIVING_SEIZURE_FREE_MONTHS["strict_states"],
                "commercial_eligible": months_free >= DRIVING_SEIZURE_FREE_MONTHS["commercial_license"],
                "recommendation": (
                    "Eligible for standard license in most states"
                    if months_free >= 6
                    else f"Need {6 - months_free} more seizure-free months for standard license"
                ),
            }
        else:
            driving_eligibility = {
                "last_seizure_date": None,
                "months_seizure_free": None,
                "standard_license_eligible": sz_count == 0,
                "strict_state_eligible": sz_count == 0,
                "commercial_eligible": False,
                "recommendation": (
                    "No seizure diary entries — verify seizure history with patient"
                    if sz_count == 0
                    else "Unable to determine seizure-free period"
                ),
            }

        # ── Employment readiness (0-10 scale, 10 = fully ready) ──
        emp_readiness = 10
        if sz_count >= 4:
            emp_readiness -= 4
        elif sz_count >= 2:
            emp_readiness -= 2
        elif sz_count >= 1:
            emp_readiness -= 1
        emp_readiness -= min(cognitive_load, 3)
        emp_readiness -= min(sedation_load // 2, 2)
        if age < 16:
            emp_readiness -= 2
        emp_readiness = max(0, emp_readiness)

        # ── Disability benefit flags ──
        disability_flags = []
        if sz_count >= 4:
            disability_flags.append("Frequent seizures (≥4 recorded) — SSA listing 11.02 potential")
        if severe_count >= 2:
            disability_flags.append("Recurrent severe seizures — functional limitation evidence")
        if cognitive_load >= 4:
            disability_flags.append("High cognitive medication burden — vocational impact documented")
        if injury_count >= 2:
            disability_flags.append("Recurrent seizure-related injuries — safety-sensitive work restriction")
        if age > 65:
            disability_flags.append("Age >65 — Social Security retirement benefit evaluation")

        eligible_for_disability = len(disability_flags) >= 2

        # ── Vocational recommendations ──
        voc_recs = []
        if emp_readiness <= 4:
            voc_recs.append("Vocational rehabilitation program referral")
        if cognitive_load >= 3:
            voc_recs.append("Cognitive accommodation assessment (reduced multitasking, written instructions)")
        if sedation_load >= 3:
            voc_recs.append("Work schedule adjustment — avoid early morning / overnight shifts")
        if sz_count >= 2:
            voc_recs.append("ADA workplace accommodation request — seizure action plan for employer")
        if not driving_eligibility.get("standard_license_eligible", True):
            voc_recs.append("Transportation assistance for commute — public transit / rideshare program")
        if mood_risk_drugs:
            voc_recs.append(f"Behavioral health support — mood-altering medications: {', '.join(mood_risk_drugs)}")
        if pregnancy_risk_drugs and pt.get("gender", "").upper() in ("F", "FEMALE"):
            voc_recs.append(f"Reproductive health counseling — teratogenic medications: {', '.join(pregnancy_risk_drugs)}")
        if age >= 16 and age <= 22:
            voc_recs.append("Transition planning — school-to-work program / IEP transition services")

        results.append({
            "patient_id": pid,
            "name": pt.get("name", ""),
            "age": age,
            "gender": pt.get("gender", ""),
            "seizure_count": sz_count,
            "medication_count": drug_count,
            "driving_eligibility": driving_eligibility,
            "employment_readiness_score": emp_readiness,
            "employment_readiness_label": (
                "ready" if emp_readiness >= 7
                else "modified-duty" if emp_readiness >= 4
                else "significant-barriers"
            ),
            "cognitive_load_score": min(cognitive_load, 10),
            "sedation_load_score": min(sedation_load, 10),
            "disability_flags": disability_flags,
            "eligible_for_disability_review": eligible_for_disability,
            "pregnancy_risk_drugs": pregnancy_risk_drugs,
            "vocational_recommendations": voc_recs,
        })

    # Summary
    sig_barriers = sum(1 for r in results if r["employment_readiness_label"] == "significant-barriers")
    disability_count = sum(1 for r in results if r["eligible_for_disability_review"])
    driving_restricted = sum(
        1 for r in results
        if not r["driving_eligibility"].get("standard_license_eligible", True)
    )

    return {
        "analysis": "Benefits & Vocational Assessment",
        "total_patients_assessed": len(results),
        "summary": {
            "significant_employment_barriers": sig_barriers,
            "eligible_for_disability_review": disability_count,
            "driving_restricted": driving_restricted,
            "mean_employment_readiness": round(
                sum(r["employment_readiness_score"] for r in results) / len(results), 1
            ) if results else 0,
        },
        "results": results,
    }


# ════════════════════════════════════════════════════════════════════════
# 4. Treatment Barrier Detection
# ════════════════════════════════════════════════════════════════════════

def treatment_barrier_detection(patient_id: str = None):
    """
    Detect treatment adherence barriers arising from social/lifestyle factors.

    Barrier categories:
      - Financial: polypharmacy cost, brand-only drugs
      - Lifestyle: high-frequency dosing (TID+), complex regimen
      - Safety fear: pregnancy-risk drugs in women of childbearing age
      - Sleep/fatigue: nocturnal seizures disrupting med schedule
      - Cognitive: cognitive drug effects impairing self-management
      - Social stigma: seizure frequency + severity driving social withdrawal
      - Transportation: inability to access pharmacy / clinic
      - Medication gap: seizure events without corresponding medication records
    """
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found", "results": []}

    patients = _get_patients(conn, patient_id)
    seizures = _get_seizure_events(conn, patient_id)
    meds = _get_medications(conn, patient_id)
    conn.close()

    results = []
    for pt in patients:
        pid = pt["patient_id"]
        age = pt.get("age") or 30
        gender = pt.get("gender", "")
        sz_events = seizures.get(pid, [])
        med_records = meds.get(pid, [])
        drugs = _unique_drugs_for_patient(med_records)
        sz_count = len(sz_events)
        drug_count = len(drugs)

        nocturnal = sum(
            1 for e in sz_events
            if e.get("event_time") and _is_nocturnal(e["event_time"])
        )
        severe_count = sum(
            1 for e in sz_events
            if e.get("severity") and e["severity"].lower() in ("severe", "high", "status")
        )

        barriers = []
        barrier_score = 0  # 0-10

        # Financial barrier
        if drug_count >= 3:
            barriers.append({
                "category": "Financial",
                "detail": f"Polypharmacy ({drug_count} medications) — high out-of-pocket cost risk",
                "severity": "high" if drug_count >= 4 else "moderate",
            })
            barrier_score += 2 if drug_count >= 4 else 1

        # Lifestyle / dosing complexity
        high_freq_drugs = [
            m["drug_name"] for m in med_records
            if m.get("frequency", "").upper() in ("TID", "QID", "3X DAILY", "4X DAILY",
                                                    "THREE TIMES DAILY", "FOUR TIMES DAILY")
        ]
        if high_freq_drugs:
            barriers.append({
                "category": "Lifestyle",
                "detail": f"High-frequency dosing ({', '.join(set(high_freq_drugs))}) — adherence challenge",
                "severity": "moderate",
            })
            barrier_score += 1

        # Pregnancy fear
        pregnancy_risk = [
            d for d in drugs
            if ASM_SIDE_EFFECTS.get(d, {}).get("pregnancy_cat") in ("X", "D")
        ]
        is_childbearing = gender.upper() in ("F", "FEMALE") and 12 <= age <= 50
        if pregnancy_risk and is_childbearing:
            barriers.append({
                "category": "Safety fear",
                "detail": f"Teratogenic medications ({', '.join(pregnancy_risk)}) in woman of childbearing age — fear-driven non-adherence risk",
                "severity": "high",
            })
            barrier_score += 2

        # Sleep disruption
        if nocturnal >= 1:
            barriers.append({
                "category": "Sleep disruption",
                "detail": f"{nocturnal} nocturnal seizure(s) — disrupted sleep impacts morning medication timing",
                "severity": "moderate" if nocturnal < 3 else "high",
            })
            barrier_score += 1 if nocturnal < 3 else 2

        # Cognitive self-management
        cog_drugs = [
            d for d in drugs
            if ASM_SIDE_EFFECTS.get(d, {}).get("cognitive") in ("high", "moderate")
        ]
        if cog_drugs:
            barriers.append({
                "category": "Cognitive",
                "detail": f"Cognitive side effects from {', '.join(cog_drugs)} — impaired self-management capacity",
                "severity": "high" if any(
                    ASM_SIDE_EFFECTS.get(d, {}).get("cognitive") == "high" for d in cog_drugs
                ) else "moderate",
            })
            barrier_score += 1

        # Social stigma / withdrawal
        if sz_count >= 3 and severe_count >= 1:
            barriers.append({
                "category": "Social stigma",
                "detail": f"Frequent ({sz_count}) + severe seizures — social withdrawal and appointment avoidance risk",
                "severity": "moderate",
            })
            barrier_score += 1

        # Transportation to pharmacy/clinic
        if sz_count > 0 and age >= 16:
            barriers.append({
                "category": "Transportation",
                "detail": "Driving restriction due to seizure history — pharmacy/clinic access barrier",
                "severity": "moderate",
            })
            barrier_score += 1

        # Medication gap indicator: seizure events but no medications on record
        if sz_count > 0 and drug_count == 0:
            barriers.append({
                "category": "Medication gap",
                "detail": "Seizure events recorded but no medications on file — possible untreated or non-adherent",
                "severity": "high",
            })
            barrier_score += 2

        barrier_score = min(barrier_score, 10)

        if barrier_score >= 7:
            priority = "high"
        elif barrier_score >= 4:
            priority = "moderate"
        else:
            priority = "low"

        # Intervention recommendations
        interventions = []
        for b in barriers:
            if b["category"] == "Financial":
                interventions.append("Enroll in Patient Assistance Programs (PAPs) / 340B pharmacy")
            elif b["category"] == "Lifestyle":
                interventions.append("Pill organizer + smartphone medication reminder app setup")
            elif b["category"] == "Safety fear":
                interventions.append("Preconception counseling referral — medication switch discussion with neurologist")
            elif b["category"] == "Sleep disruption":
                interventions.append("Seizure safety plan — evening medication moved earlier; bed alarm referral")
            elif b["category"] == "Cognitive":
                interventions.append("Simplified medication regimen discussion with prescriber; caregiver-assisted management")
            elif b["category"] == "Social stigma":
                interventions.append("Individual counseling + Epilepsy Foundation peer support connection")
            elif b["category"] == "Transportation":
                interventions.append("Mail-order pharmacy enrollment; telemedicine follow-up option")
            elif b["category"] == "Medication gap":
                interventions.append("Urgent medication reconciliation with treating neurologist")
        # Deduplicate
        interventions = list(dict.fromkeys(interventions))

        results.append({
            "patient_id": pid,
            "name": pt.get("name", ""),
            "age": age,
            "gender": gender,
            "seizure_count": sz_count,
            "medication_count": drug_count,
            "detected_barriers": barriers,
            "barrier_count": len(barriers),
            "barrier_score": barrier_score,
            "priority": priority,
            "intervention_recommendations": interventions,
        })

    # Summary
    high_barrier = sum(1 for r in results if r["priority"] == "high")
    mod_barrier = sum(1 for r in results if r["priority"] == "moderate")
    mean_score = round(sum(r["barrier_score"] for r in results) / len(results), 1) if results else 0

    # Barrier category frequency
    cat_freq = defaultdict(int)
    for r in results:
        for b in r["detected_barriers"]:
            cat_freq[b["category"]] += 1
    barrier_frequency = [
        {"category": k, "patients_affected": v}
        for k, v in sorted(cat_freq.items(), key=lambda x: -x[1])
    ]

    return {
        "analysis": "Treatment Barrier Detection",
        "total_patients_assessed": len(results),
        "summary": {
            "high_barrier_priority": high_barrier,
            "moderate_barrier_priority": mod_barrier,
            "low_barrier_priority": len(results) - high_barrier - mod_barrier,
            "mean_barrier_score": mean_score,
            "barrier_category_frequency": barrier_frequency,
        },
        "results": results,
    }


# ════════════════════════════════════════════════════════════════════════
# 5. Full Dashboard
# ════════════════════════════════════════════════════════════════════════

def full_dashboard(patient_id: str = None):
    """Combined Medical Social Worker dashboard — all 4 modules."""
    sdoh = social_determinants_screening(patient_id)
    burden = caregiver_burden(patient_id)
    vocational = benefits_vocational(patient_id)
    barriers = treatment_barrier_detection(patient_id)

    total_patients = sdoh["total_patients_screened"]
    high_vuln = sdoh["summary"]["high_priority"]
    high_burden = burden["summary"]["high_burnout_risk"]
    sig_barriers_emp = vocational["summary"]["significant_employment_barriers"]
    high_barrier_tx = barriers["summary"]["high_barrier_priority"]
    driving_restricted = vocational["summary"]["driving_restricted"]
    respite_needed = burden["summary"]["respite_referrals_needed"]

    return {
        "module": "Medical Social Worker (Epilepsy)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_patients": total_patients,
            "high_sdoh_vulnerability": high_vuln,
            "high_caregiver_burnout": high_burden,
            "significant_employment_barriers": sig_barriers_emp,
            "high_treatment_barriers": high_barrier_tx,
            "driving_restricted": driving_restricted,
            "respite_referrals_needed": respite_needed,
            "mean_vulnerability_score": sdoh["summary"]["mean_vulnerability_score"],
            "mean_zbi_score": burden["summary"]["mean_zbi"],
            "mean_barrier_score": barriers["summary"]["mean_barrier_score"],
        },
        "sdoh_screening": sdoh,
        "caregiver_burden": burden,
        "benefits_vocational": vocational,
        "treatment_barriers": barriers,
    }
