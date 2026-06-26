"""
Epilepsy Specialist Nurse Module
==================================
Seizure diary analysis, AED adherence monitoring, SUDEP risk scoring,
seizure action plans, and patient/family education assessment — all from
REAL patient data in data/clinical.db.

Clinical evidence base:
  - Hesdorffer DC et al. (2011). Combined analysis of risk factors for
    SUDEP. Epilepsia 52:1150-1159.
  - Devinsky O et al. (2016). Sudden unexpected death in epilepsy:
    epidemiology, mechanisms, and prevention. Lancet Neurol 15:1075-1088.
  - Ryvlin P et al. (2013). Incidence and mechanisms of cardiorespiratory
    arrests in epilepsy monitoring units (MORTEMUS). Lancet Neurol 12:966-977.
  - Faught E et al. (2008). Nonadherence to antiepileptic drugs and increased
    mortality. Neurology 71:1572-1578.
  - Epilepsy Foundation (2022). Seizure First Aid guidelines.
  - NICE CG137 (2012, updated 2022). Epilepsies: diagnosis and management.
  - Shankar R et al. (2017). SUDEP and epilepsy-related mortality in
    pregnancy. Epilepsia 58:1235-1240.
  - Lhatoo SD et al. (2010). Mortality in epilepsy in the first 11-14 years
    after diagnosis. J Neurol Neurosurg Psychiatry 81:1114-1118.

Endpoints:
  /api/epilepsy-nurse                  — full dashboard (all sub-analyses)
  /api/epilepsy-nurse/seizure-diary    — seizure frequency, patterns, triggers
  /api/epilepsy-nurse/adherence        — AED adherence monitoring
  /api/epilepsy-nurse/sudep-risk       — SUDEP-7 risk assessment
  /api/epilepsy-nurse/action-plan      — per-patient seizure action plans
  /api/epilepsy-nurse/education        — patient/family education checklist
  /api/epilepsy-nurse/definitions      — metric definitions + references

All data from REAL patients, medications, seizure_diary, and assessments
in data/clinical.db.
"""

import sqlite3
import os
import json
from collections import defaultdict
from datetime import datetime, timedelta

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "clinical.db",
)

# ─── SUDEP risk factor weights (Hesdorffer et al. 2011, SUDEP-7 scoring) ──
# Simplified SUDEP risk model based on published epidemiological risk factors.
# Each factor contributes to a composite score 0-10.
SUDEP_RISK_FACTORS = {
    "gtcs_frequency": {
        "description": "Generalized tonic-clonic seizure (GTCS) frequency",
        "weight": 2.5,
        "reference": "Hesdorffer 2011: GTCS is strongest modifiable risk factor; OR 15.46 for >=3/year",
    },
    "nocturnal_seizures": {
        "description": "Nocturnal seizures (unsupervised)",
        "weight": 2.0,
        "reference": "MORTEMUS (Ryvlin 2013): 86% of monitored SUDEP cases occurred during sleep",
    },
    "aed_nonadherence": {
        "description": "Anti-epileptic drug non-adherence",
        "weight": 1.5,
        "reference": "Faught 2008: Non-adherence associated with 3x increased mortality",
    },
    "duration_of_epilepsy": {
        "description": "Duration of epilepsy > 15 years",
        "weight": 1.0,
        "reference": "Lhatoo 2010: Cumulative risk increases with duration",
    },
    "polytherapy_failure": {
        "description": "Failed >= 2 AED regimens (drug-resistant epilepsy)",
        "weight": 1.5,
        "reference": "Devinsky 2016: Drug-resistant epilepsy is major SUDEP risk factor",
    },
    "male_sex": {
        "description": "Male sex",
        "weight": 0.5,
        "reference": "Hesdorffer 2011: Males have slightly higher SUDEP incidence (OR ~1.4)",
    },
    "young_adult_age": {
        "description": "Young adult age (20-40 years)",
        "weight": 1.0,
        "reference": "Devinsky 2016: Peak SUDEP incidence in young adults 20-40 years",
    },
}

# ─── AED adherence complexity factors ──────────────────────────────────
AED_COMPLEXITY = {
    "levetiracetam": {"doses_per_day": 2, "titration_required": False, "monitoring": "none"},
    "lamotrigine": {"doses_per_day": 2, "titration_required": True, "monitoring": "rash_check"},
    "valproic acid": {"doses_per_day": 2, "titration_required": True, "monitoring": "LFT_levels"},
    "carbamazepine": {"doses_per_day": 3, "titration_required": True, "monitoring": "levels_CBC"},
    "phenytoin": {"doses_per_day": 1, "titration_required": True, "monitoring": "levels_mandatory"},
    "phenobarbital": {"doses_per_day": 1, "titration_required": True, "monitoring": "levels_mandatory"},
    "topiramate": {"doses_per_day": 2, "titration_required": True, "monitoring": "renal_bicarb"},
    "oxcarbazepine": {"doses_per_day": 2, "titration_required": True, "monitoring": "sodium"},
    "zonisamide": {"doses_per_day": 1, "titration_required": True, "monitoring": "renal_bicarb"},
    "lacosamide": {"doses_per_day": 2, "titration_required": True, "monitoring": "ECG"},
    "clobazam": {"doses_per_day": 2, "titration_required": True, "monitoring": "sedation"},
    "brivaracetam": {"doses_per_day": 2, "titration_required": False, "monitoring": "none"},
    "perampanel": {"doses_per_day": 1, "titration_required": True, "monitoring": "psychiatric"},
    "eslicarbazepine": {"doses_per_day": 1, "titration_required": True, "monitoring": "sodium"},
    "cannabidiol": {"doses_per_day": 2, "titration_required": True, "monitoring": "LFT_levels"},
    "cenobamate": {"doses_per_day": 1, "titration_required": True, "monitoring": "ECG_DRESS"},
}

# ─── Education checklist domains (NICE CG137, Epilepsy Foundation) ──────
EDUCATION_DOMAINS = [
    {
        "domain": "Driving restrictions",
        "description": "Seizure-free period required before driving (varies by jurisdiction: typically 3-12 months); must report to licensing authority",
        "reference": "NICE CG137 section 1.15; local DMV regulations",
        "risk_level": "high",
    },
    {
        "domain": "Water safety / Swimming",
        "description": "Never swim alone; inform lifeguard; shower preferred over bath; avoid deep water; use life jacket for boating",
        "reference": "Epilepsy Foundation water safety guidelines; drowning risk 15-19x general population",
        "risk_level": "high",
    },
    {
        "domain": "Pregnancy counseling",
        "description": "Pre-conception planning essential: folic acid 5mg/day, AED optimization (avoid valproate), teratogenicity risk, pregnancy registry enrollment",
        "reference": "NICE CG137 section 1.16; EURAP registry; FDA pregnancy category guidance",
        "risk_level": "high",
    },
    {
        "domain": "Alcohol and recreational drugs",
        "description": "Alcohol lowers seizure threshold; binge drinking especially dangerous; cannabis interactions with AEDs; recreational drugs may trigger seizures",
        "reference": "NICE CG137 section 1.15; Epilepsia review articles on alcohol-seizure interaction",
        "risk_level": "moderate",
    },
    {
        "domain": "Ketogenic diet awareness",
        "description": "Dietary therapy options for drug-resistant epilepsy: classic ketogenic, modified Atkins, MCT, LGIT; requires dietitian supervision",
        "reference": "NICE CG137 section 1.9; Cochrane review: ketogenic diets for epilepsy",
        "risk_level": "low",
    },
    {
        "domain": "Medication management",
        "description": "Take AEDs at consistent times; never abruptly stop; carry medication list; know side effects; brand vs generic switching risks",
        "reference": "NICE CG137 section 1.9; Faught 2008 adherence data",
        "risk_level": "high",
    },
    {
        "domain": "Sleep hygiene",
        "description": "Sleep deprivation is a common seizure trigger; maintain regular schedule; avoid shift work if possible; treat sleep apnea",
        "reference": "NICE CG137; Epilepsia reviews on sleep-seizure relationship",
        "risk_level": "moderate",
    },
    {
        "domain": "SUDEP awareness",
        "description": "Risk of sudden unexpected death in epilepsy; importance of seizure control, AED adherence, nocturnal supervision; when to discuss with family",
        "reference": "Devinsky 2016; NICE CG137 section 1.15.5; SUDEP Action charity resources",
        "risk_level": "high",
    },
    {
        "domain": "Seizure first aid (bystander education)",
        "description": "Recovery position, protect head, time the seizure, when to call emergency services (>5 min or first seizure), do NOT restrain or put anything in mouth",
        "reference": "Epilepsy Foundation first-aid guidelines; NICE CG137 section 1.13",
        "risk_level": "high",
    },
    {
        "domain": "Workplace / school safety",
        "description": "Disclosure rights, reasonable adjustments, avoiding heights/machinery/unsupervised hazards, Equality Act/ADA protections",
        "reference": "NICE CG137 section 1.15; Epilepsy Action employment guidance",
        "risk_level": "moderate",
    },
    {
        "domain": "Mental health awareness",
        "description": "Depression and anxiety common in epilepsy (30-50% prevalence); suicidality screening; AED mood effects; when to seek help",
        "reference": "NICE CG137 section 1.14; ILAE Commission on psychiatric aspects",
        "risk_level": "moderate",
    },
    {
        "domain": "Emergency ID / medical alert",
        "description": "Wear medical alert bracelet/necklace; carry seizure action plan card; ICE contacts programmed in phone",
        "reference": "Epilepsy Foundation; NICE CG137; best practice consensus",
        "risk_level": "moderate",
    },
]


# ─── Helper: get DB connection ─────────────────────────────────────────
def _get_conn():
    """Return a sqlite3 connection to clinical.db. Returns None if DB missing."""
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn, table_name):
    """Check if a table exists in the DB."""
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return c.fetchone() is not None


def _get_patients(conn, patient_id=None):
    """Fetch patient rows. Filter by patient_id if given."""
    c = conn.cursor()
    if patient_id:
        c.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,))
    else:
        c.execute("SELECT * FROM patients")
    return [dict(r) for r in c.fetchall()]


def _get_medications(conn, patient_id=None):
    """Fetch medications with parsed fields_json."""
    c = conn.cursor()
    if patient_id:
        c.execute("SELECT * FROM medications WHERE patient_id=?", (patient_id,))
    else:
        c.execute("SELECT * FROM medications")
    rows = []
    for r in c.fetchall():
        d = dict(r)
        try:
            d["fields"] = json.loads(d.get("fields_json", "{}"))
        except Exception:
            d["fields"] = {}
        rows.append(d)
    return rows


def _get_seizure_diary(conn, patient_id=None):
    """Fetch seizure diary entries."""
    if not _table_exists(conn, "seizure_diary"):
        return []
    c = conn.cursor()
    if patient_id:
        c.execute("SELECT * FROM seizure_diary WHERE patient_id=?", (patient_id,))
    else:
        c.execute("SELECT * FROM seizure_diary")
    return [dict(r) for r in c.fetchall()]


def _get_assessments(conn, patient_id=None, instrument=None):
    """Fetch assessments, optionally filtered by patient and/or instrument."""
    if not _table_exists(conn, "assessments"):
        return []
    c = conn.cursor()
    q = "SELECT * FROM assessments WHERE 1=1"
    params = []
    if patient_id:
        q += " AND patient_id=?"
        params.append(patient_id)
    if instrument:
        q += " AND instrument=?"
        params.append(instrument)
    c.execute(q, params)
    return [dict(r) for r in c.fetchall()]


# ═══════════════════════════════════════════════════════════════════════
#  1. SEIZURE DIARY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def seizure_diary_analysis(patient_id=None):
    """Analyze seizure diary: frequency, severity distribution, trigger patterns,
    temporal patterns, injury rates, ER visit rates."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        entries = _get_seizure_diary(conn, patient_id)
        patients = _get_patients(conn, patient_id)
        pid_set = {p["patient_id"] for p in patients}

        if not entries:
            return {
                "title": "Seizure Diary Analysis",
                "subtitle": "No seizure diary entries found" + (f" for {patient_id}" if patient_id else ""),
                "total_entries": 0,
                "patients_with_diary": 0,
                "patients": [],
                "aggregate": {},
            }

        # Group by patient
        by_patient = defaultdict(list)
        for e in entries:
            by_patient[e["patient_id"]].append(e)

        patient_results = []
        all_severities = []
        all_triggers = []
        all_injuries = []
        total_er = 0
        total_rescue = 0
        total_nocturnal = 0

        for pid, events in by_patient.items():
            n = len(events)
            # Severity distribution
            sev_dist = defaultdict(int)
            for e in events:
                sev = (e.get("severity") or "Unknown").capitalize()
                sev_dist[sev] += 1
                all_severities.append(sev)

            # Triggers
            triggers = defaultdict(int)
            for e in events:
                t = e.get("trigger")
                if t:
                    triggers[t] += 1
                    all_triggers.append(t)

            # Duration stats
            durations = [e.get("duration_sec") or 0 for e in events if e.get("duration_sec")]
            avg_dur = round(sum(durations) / len(durations), 1) if durations else 0
            max_dur = max(durations) if durations else 0

            # Recovery time
            recoveries = [e.get("recovery_min") or 0 for e in events if e.get("recovery_min")]
            avg_recovery = round(sum(recoveries) / len(recoveries), 1) if recoveries else 0

            # Injuries
            injuries = [e for e in events if e.get("injury") and e["injury"].lower() not in ("no", "none", "")]
            injury_types = defaultdict(int)
            for e in injuries:
                injury_types[e["injury"]] += 1
            all_injuries.extend([e["injury"] for e in injuries])

            # ER visits
            er_visits = sum(1 for e in events if e.get("er_visit") and e["er_visit"].lower() == "yes")
            total_er += er_visits

            # Rescue medication use
            rescue_used = sum(1 for e in events if e.get("rescue_med") and e["rescue_med"].lower() not in ("", "no", "none"))
            total_rescue += rescue_used

            # Nocturnal (estimate from event_time if available)
            nocturnal = 0
            for e in events:
                t = e.get("event_time")
                if t:
                    try:
                        hour = int(t.split(":")[0])
                        if hour >= 22 or hour < 6:
                            nocturnal += 1
                    except Exception:
                        pass
            total_nocturnal += nocturnal

            # Date range
            dates = [e.get("event_date") for e in events if e.get("event_date")]
            dates.sort()
            date_range = f"{dates[0]} to {dates[-1]}" if len(dates) >= 2 else (dates[0] if dates else "N/A")

            # Monthly frequency estimate
            if len(dates) >= 2:
                try:
                    d0 = datetime.strptime(dates[0], "%Y-%m-%d")
                    d1 = datetime.strptime(dates[-1], "%Y-%m-%d")
                    span_days = max((d1 - d0).days, 1)
                    monthly_freq = round(n * 30 / span_days, 1)
                except Exception:
                    monthly_freq = n
            else:
                monthly_freq = n

            # Seizure freedom check
            seizure_free_days = 0
            if dates:
                try:
                    last = datetime.strptime(dates[-1], "%Y-%m-%d")
                    seizure_free_days = (datetime.now() - last).days
                except Exception:
                    pass

            patient_results.append({
                "patient_id": pid,
                "total_seizures": n,
                "monthly_frequency": monthly_freq,
                "severity_distribution": dict(sev_dist),
                "triggers": dict(triggers),
                "average_duration_sec": avg_dur,
                "max_duration_sec": max_dur,
                "average_recovery_min": avg_recovery,
                "injuries": dict(injury_types),
                "injury_rate_pct": round(len(injuries) / n * 100, 1) if n else 0,
                "er_visits": er_visits,
                "rescue_medication_uses": rescue_used,
                "nocturnal_events": nocturnal,
                "date_range": date_range,
                "days_since_last_seizure": seizure_free_days,
                "seizure_free_90_days": seizure_free_days >= 90,
            })

        # Aggregate stats
        agg_sev = defaultdict(int)
        for s in all_severities:
            agg_sev[s] += 1
        agg_trig = defaultdict(int)
        for t in all_triggers:
            agg_trig[t] += 1
        agg_inj = defaultdict(int)
        for i in all_injuries:
            agg_inj[i] += 1

        return {
            "title": "Seizure Diary Analysis",
            "subtitle": "Seizure frequency, severity, triggers, injuries from real seizure diary entries in clinical.db",
            "total_entries": len(entries),
            "patients_with_diary": len(by_patient),
            "total_er_visits": total_er,
            "total_rescue_med_uses": total_rescue,
            "total_nocturnal_events": total_nocturnal,
            "aggregate_severity": dict(agg_sev),
            "aggregate_triggers": dict(agg_trig),
            "aggregate_injuries": dict(agg_inj),
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  2. AED ADHERENCE MONITORING
# ═══════════════════════════════════════════════════════════════════════

def adherence_monitoring(patient_id=None):
    """Assess AED adherence: polytherapy complexity, dosing frequency burden,
    titration requirements, monitoring demands, seizure-gap correlation."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        meds = _get_medications(conn, patient_id)
        patients = _get_patients(conn, patient_id)
        seizures = _get_seizure_diary(conn, patient_id)

        if not meds and not patients:
            return {
                "title": "AED Adherence Monitoring",
                "subtitle": "No medication or patient data found" + (f" for {patient_id}" if patient_id else ""),
                "patients": [],
            }

        # Group meds by patient
        meds_by_patient = defaultdict(list)
        for m in meds:
            meds_by_patient[m["patient_id"]].append(m)

        # Group seizures by patient
        seiz_by_patient = defaultdict(list)
        for s in seizures:
            seiz_by_patient[s["patient_id"]].append(s)

        # Build patient set from meds + patients table
        all_pids = set(m["patient_id"] for m in meds) | set(p["patient_id"] for p in patients)
        if patient_id:
            all_pids = {patient_id}

        patient_results = []
        for pid in sorted(all_pids):
            p_meds = meds_by_patient.get(pid, [])
            p_seizures = seiz_by_patient.get(pid, [])

            # Extract drug names
            drug_names = []
            for m in p_meds:
                f = m.get("fields", {})
                dn = f.get("drug_name", "")
                if dn:
                    drug_names.append(dn)
                # Check for AED list field
                aed_list = f.get("aed", [])
                if isinstance(aed_list, list):
                    for a in aed_list:
                        if a and a not in drug_names:
                            drug_names.append(a)

            n_drugs = len(set(drug_names))
            is_polytherapy = n_drugs >= 2

            # Calculate complexity score (0-10)
            complexity_score = 0
            total_daily_doses = 0
            titration_count = 0
            monitoring_needs = []

            for dn in set(drug_names):
                key = dn.lower().strip()
                info = AED_COMPLEXITY.get(key, {"doses_per_day": 2, "titration_required": False, "monitoring": "none"})
                total_daily_doses += info["doses_per_day"]
                if info["titration_required"]:
                    titration_count += 1
                if info["monitoring"] != "none":
                    monitoring_needs.append(f"{dn}: {info['monitoring']}")

            # Complexity scoring
            if n_drugs >= 3:
                complexity_score += 3
            elif n_drugs == 2:
                complexity_score += 2
            elif n_drugs == 1:
                complexity_score += 1

            if total_daily_doses >= 6:
                complexity_score += 2
            elif total_daily_doses >= 4:
                complexity_score += 1

            if titration_count >= 2:
                complexity_score += 2
            elif titration_count >= 1:
                complexity_score += 1

            if len(monitoring_needs) >= 2:
                complexity_score += 2
            elif len(monitoring_needs) >= 1:
                complexity_score += 1

            complexity_score = min(complexity_score, 10)

            # Adherence risk level
            if complexity_score >= 7:
                adherence_risk = "High"
                risk_color = "red"
            elif complexity_score >= 4:
                adherence_risk = "Moderate"
                risk_color = "amber"
            else:
                adherence_risk = "Low"
                risk_color = "green"

            # Seizure burden (as indirect adherence indicator)
            seizure_count = len(p_seizures)
            severe_seizures = sum(1 for s in p_seizures if (s.get("severity") or "").lower() == "severe")

            # Recommendations
            recommendations = []
            if is_polytherapy:
                recommendations.append("Review polytherapy regimen — consider simplification if seizure-free")
            if total_daily_doses >= 4:
                recommendations.append(f"High pill burden ({total_daily_doses} doses/day) — consider extended-release formulations or once-daily alternatives")
            if not drug_names:
                recommendations.append("No AED records found — verify medication history")
            if seizure_count > 0 and n_drugs > 0:
                recommendations.append("Active seizures with AED therapy — assess adherence with prescription refill data and patient interview")
            if monitoring_needs:
                recommendations.append(f"Monitoring required: {'; '.join(monitoring_needs)}")
            if complexity_score >= 7:
                recommendations.append("HIGH non-adherence risk: consider adherence aids (pill organizer, alarms, blister packs)")

            patient_results.append({
                "patient_id": pid,
                "aed_count": n_drugs,
                "aed_names": list(set(drug_names)),
                "is_polytherapy": is_polytherapy,
                "total_daily_doses": total_daily_doses,
                "titration_count": titration_count,
                "monitoring_needs": monitoring_needs,
                "complexity_score": complexity_score,
                "complexity_max": 10,
                "adherence_risk": adherence_risk,
                "risk_color": risk_color,
                "seizure_count": seizure_count,
                "severe_seizure_count": severe_seizures,
                "recommendations": recommendations,
            })

        return {
            "title": "AED Adherence Monitoring",
            "subtitle": "Polytherapy complexity, dosing burden, monitoring demands — Faught 2008 non-adherence mortality data",
            "total_patients": len(patient_results),
            "high_risk_count": sum(1 for p in patient_results if p["adherence_risk"] == "High"),
            "moderate_risk_count": sum(1 for p in patient_results if p["adherence_risk"] == "Moderate"),
            "low_risk_count": sum(1 for p in patient_results if p["adherence_risk"] == "Low"),
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  3. SUDEP RISK ASSESSMENT (SUDEP-7 scoring model)
# ═══════════════════════════════════════════════════════════════════════

def sudep_risk_assessment(patient_id=None):
    """SUDEP risk scoring per Hesdorffer 2011 + Devinsky 2016 + MORTEMUS.
    Composite 0-10 score from: GTCS frequency, nocturnal seizures,
    AED non-adherence, epilepsy duration, polytherapy failure, sex, age."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        patients = _get_patients(conn, patient_id)
        seizures = _get_seizure_diary(conn, patient_id)
        meds = _get_medications(conn, patient_id)

        if not patients:
            return {
                "title": "SUDEP Risk Assessment",
                "subtitle": "No patient data found" + (f" for {patient_id}" if patient_id else ""),
                "patients": [],
            }

        seiz_by_patient = defaultdict(list)
        for s in seizures:
            seiz_by_patient[s["patient_id"]].append(s)

        meds_by_patient = defaultdict(list)
        for m in meds:
            meds_by_patient[m["patient_id"]].append(m)

        patient_results = []
        for p in patients:
            pid = p["patient_id"]
            p_seizures = seiz_by_patient.get(pid, [])
            p_meds = meds_by_patient.get(pid, [])

            factors = {}
            total_score = 0

            # 1. GTCS frequency (proxy: severe seizures in diary)
            severe = sum(1 for s in p_seizures if (s.get("severity") or "").lower() == "severe")
            if severe >= 3:
                gtcs_score = SUDEP_RISK_FACTORS["gtcs_frequency"]["weight"]
            elif severe >= 1:
                gtcs_score = SUDEP_RISK_FACTORS["gtcs_frequency"]["weight"] * 0.5
            else:
                gtcs_score = 0
            factors["gtcs_frequency"] = {"value": severe, "score": round(gtcs_score, 2), "present": severe > 0}
            total_score += gtcs_score

            # 2. Nocturnal seizures
            nocturnal = 0
            for s in p_seizures:
                t = s.get("event_time")
                if t:
                    try:
                        hour = int(t.split(":")[0])
                        if hour >= 22 or hour < 6:
                            nocturnal += 1
                    except Exception:
                        pass
            # Also check if any seizure has no time (unknown supervision status)
            unknown_time = sum(1 for s in p_seizures if not s.get("event_time"))
            if nocturnal > 0:
                noct_score = SUDEP_RISK_FACTORS["nocturnal_seizures"]["weight"]
            elif unknown_time > 0 and len(p_seizures) > 0:
                noct_score = SUDEP_RISK_FACTORS["nocturnal_seizures"]["weight"] * 0.3  # partial risk: unknown supervision
            else:
                noct_score = 0
            factors["nocturnal_seizures"] = {"value": nocturnal, "unknown_time": unknown_time, "score": round(noct_score, 2), "present": nocturnal > 0}
            total_score += noct_score

            # 3. AED non-adherence (proxy: active seizures despite medications)
            n_drugs = 0
            drug_names = []
            for m in p_meds:
                f = m.get("fields", {})
                dn = f.get("drug_name", "")
                if dn:
                    drug_names.append(dn)
                aed_list = f.get("aed", [])
                if isinstance(aed_list, list):
                    drug_names.extend([a for a in aed_list if a])
            n_drugs = len(set(drug_names))

            # If on meds but still having seizures — possible non-adherence signal
            adh_score = 0
            if n_drugs > 0 and len(p_seizures) > 3:
                adh_score = SUDEP_RISK_FACTORS["aed_nonadherence"]["weight"]
            elif n_drugs > 0 and len(p_seizures) > 0:
                adh_score = SUDEP_RISK_FACTORS["aed_nonadherence"]["weight"] * 0.4
            elif n_drugs == 0 and p.get("disease", "").lower() == "epilepsy":
                adh_score = SUDEP_RISK_FACTORS["aed_nonadherence"]["weight"]  # no meds for epilepsy patient
            factors["aed_nonadherence"] = {"value": f"{len(p_seizures)} seizures on {n_drugs} AEDs", "score": round(adh_score, 2), "present": adh_score > 0}
            total_score += adh_score

            # 4. Duration of epilepsy (proxy: age, assuming onset in teens/twenties)
            age = p.get("age")
            dur_score = 0
            if age and age > 40:
                dur_score = SUDEP_RISK_FACTORS["duration_of_epilepsy"]["weight"]
            elif age and age > 30:
                dur_score = SUDEP_RISK_FACTORS["duration_of_epilepsy"]["weight"] * 0.5
            factors["duration_of_epilepsy"] = {"value": f"age {age}", "score": round(dur_score, 2), "present": dur_score > 0}
            total_score += dur_score

            # 5. Polytherapy failure (drug-resistant if >= 2 AEDs and still seizing)
            poly_score = 0
            if n_drugs >= 2 and len(p_seizures) > 0:
                poly_score = SUDEP_RISK_FACTORS["polytherapy_failure"]["weight"]
            elif n_drugs >= 2:
                poly_score = SUDEP_RISK_FACTORS["polytherapy_failure"]["weight"] * 0.3
            factors["polytherapy_failure"] = {"value": f"{n_drugs} AEDs", "score": round(poly_score, 2), "present": poly_score > 0}
            total_score += poly_score

            # 6. Male sex
            gender = (p.get("gender") or "").lower()
            male_score = SUDEP_RISK_FACTORS["male_sex"]["weight"] if gender == "male" else 0
            factors["male_sex"] = {"value": gender, "score": round(male_score, 2), "present": gender == "male"}
            total_score += male_score

            # 7. Young adult age (20-40)
            ya_score = 0
            if age and 20 <= age <= 40:
                ya_score = SUDEP_RISK_FACTORS["young_adult_age"]["weight"]
            factors["young_adult_age"] = {"value": age, "score": round(ya_score, 2), "present": ya_score > 0}
            total_score += ya_score

            total_score = round(min(total_score, 10), 2)

            # Risk level
            if total_score >= 6:
                risk_level = "High"
                risk_color = "red"
                action = "Urgent SUDEP risk counseling; optimize AED therapy; consider nocturnal supervision/monitoring devices; refer to comprehensive epilepsy center"
            elif total_score >= 3:
                risk_level = "Moderate"
                risk_color = "amber"
                action = "SUDEP risk discussion with patient/family; review AED adherence; seizure diary review; consider seizure detection devices"
            else:
                risk_level = "Low"
                risk_color = "green"
                action = "Routine SUDEP information provision per NICE CG137; continue standard care"

            patient_results.append({
                "patient_id": pid,
                "age": age,
                "gender": p.get("gender"),
                "sudep_score": total_score,
                "sudep_max": 10,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "recommended_action": action,
                "factors": factors,
                "seizure_count": len(p_seizures),
                "aed_count": n_drugs,
            })

        return {
            "title": "SUDEP Risk Assessment",
            "subtitle": "SUDEP-7 scoring model — Hesdorffer 2011, Devinsky 2016, MORTEMUS (Ryvlin 2013)",
            "total_patients": len(patient_results),
            "high_risk_count": sum(1 for p in patient_results if p["risk_level"] == "High"),
            "moderate_risk_count": sum(1 for p in patient_results if p["risk_level"] == "Moderate"),
            "low_risk_count": sum(1 for p in patient_results if p["risk_level"] == "Low"),
            "risk_factors_legend": SUDEP_RISK_FACTORS,
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  4. SEIZURE ACTION PLAN
# ═══════════════════════════════════════════════════════════════════════

def seizure_action_plan(patient_id=None):
    """Generate per-patient seizure first-aid plan: rescue medication,
    when to call emergency, recovery position, seizure type-specific actions."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        patients = _get_patients(conn, patient_id)
        seizures = _get_seizure_diary(conn, patient_id)
        meds = _get_medications(conn, patient_id)

        if not patients:
            return {
                "title": "Seizure Action Plans",
                "subtitle": "No patient data found",
                "patients": [],
            }

        seiz_by_patient = defaultdict(list)
        for s in seizures:
            seiz_by_patient[s["patient_id"]].append(s)

        meds_by_patient = defaultdict(list)
        for m in meds:
            meds_by_patient[m["patient_id"]].append(m)

        patient_results = []
        for p in patients:
            pid = p["patient_id"]
            p_seizures = seiz_by_patient.get(pid, [])
            p_meds = meds_by_patient.get(pid, [])

            # Determine typical seizure pattern from diary
            severity_counts = defaultdict(int)
            for s in p_seizures:
                sev = (s.get("severity") or "Unknown").capitalize()
                severity_counts[sev] += 1

            has_severe = severity_counts.get("Severe", 0) > 0
            has_injury_history = any(
                s.get("injury") and s["injury"].lower() not in ("no", "none", "")
                for s in p_seizures
            )
            has_er_history = any(
                s.get("er_visit") and s["er_visit"].lower() == "yes"
                for s in p_seizures
            )

            # Typical duration
            durations = [s.get("duration_sec") or 0 for s in p_seizures if s.get("duration_sec")]
            typical_duration = round(sum(durations) / len(durations), 0) if durations else None
            max_duration = max(durations) if durations else None
            status_epilepticus_risk = (max_duration or 0) >= 300  # >= 5 min

            # Drug names
            drug_names = []
            for m in p_meds:
                f = m.get("fields", {})
                dn = f.get("drug_name", "")
                if dn:
                    drug_names.append(dn)
                aed_list = f.get("aed", [])
                if isinstance(aed_list, list):
                    drug_names.extend([a for a in aed_list if a])
            drug_names = list(set(drug_names))

            # Build action plan
            plan = {
                "patient_id": pid,
                "patient_name": p.get("name", ""),
                "age": p.get("age"),
                "gender": p.get("gender"),
                "current_aeds": drug_names,
                "typical_seizure_duration_sec": typical_duration,
                "max_recorded_duration_sec": max_duration,
                "status_epilepticus_risk": status_epilepticus_risk,
                "has_injury_history": has_injury_history,
                "has_er_history": has_er_history,
                "seizure_severity_pattern": dict(severity_counts),
                "total_recorded_seizures": len(p_seizures),
            }

            # First aid steps (Epilepsy Foundation guidelines)
            plan["first_aid_steps"] = [
                "Stay calm and time the seizure",
                "Clear the area of hard or sharp objects",
                "Place something soft under the head",
                "Turn the person gently onto their side (recovery position)",
                "Do NOT restrain the person or hold them down",
                "Do NOT put anything in their mouth",
                "Do NOT give food or water until fully alert",
                "Stay with the person until fully recovered",
                "Note the time, duration, and any observations",
            ]

            # When to call emergency services
            emergency_criteria = [
                "Seizure lasts longer than 5 minutes",
                "Person does not regain consciousness",
                "A second seizure occurs shortly after the first",
                "Person is injured during the seizure",
                "This is the person's first known seizure",
                "Person has difficulty breathing after the seizure",
                "Seizure occurs in water",
                "Person is pregnant, has diabetes, or has heart disease",
            ]
            if status_epilepticus_risk:
                emergency_criteria.insert(0, "** HIGH PRIORITY: This patient has history of prolonged seizures (>5 min) — call 911 immediately if seizure begins **")
            plan["emergency_criteria"] = emergency_criteria

            # Rescue medication guidance
            rescue_meds = []
            if has_severe or status_epilepticus_risk:
                rescue_meds.append({
                    "medication": "Midazolam (buccal)",
                    "dose_guidance": "10mg buccal for adults (age-adjusted for children per BNF)",
                    "when_to_give": "If tonic-clonic seizure lasts > 5 minutes or as prescribed",
                    "reference": "NICE CG137; Epilepsy Foundation rescue medication guidelines",
                })
                rescue_meds.append({
                    "medication": "Diazepam (rectal)",
                    "dose_guidance": "10-20mg rectal for adults (alternative to buccal midazolam)",
                    "when_to_give": "If buccal midazolam unavailable and seizure > 5 minutes",
                    "reference": "NICE CG137 section 1.13",
                })
            plan["rescue_medications"] = rescue_meds

            # Recovery guidance
            plan["recovery_guidance"] = {
                "recovery_position": "Turn onto side (left lateral), support head, check airway is clear",
                "post_ictal_monitoring": "Stay with person; they may be confused, sleepy, or agitated for 15-60 minutes",
                "when_to_resume_normal_activity": "Only after fully alert, oriented, and able to walk safely",
                "documentation": "Record date, time, duration, type, any triggers, injuries, and medications given",
            }

            # Safety alerts
            safety_alerts = []
            if has_injury_history:
                safety_alerts.append("INJURY HISTORY: This patient has sustained seizure-related injuries — ensure protective environment")
            if status_epilepticus_risk:
                safety_alerts.append("STATUS EPILEPTICUS RISK: History of prolonged seizures — ensure rescue medication is readily available at all times")
            if not drug_names:
                safety_alerts.append("NO AED ON FILE: Verify medication status — untreated epilepsy has higher seizure risk")
            plan["safety_alerts"] = safety_alerts

            patient_results.append(plan)

        return {
            "title": "Seizure Action Plans",
            "subtitle": "Per-patient seizure first-aid plans — Epilepsy Foundation guidelines, NICE CG137",
            "total_patients": len(patient_results),
            "patients_with_status_risk": sum(1 for p in patient_results if p["status_epilepticus_risk"]),
            "patients_with_injury_history": sum(1 for p in patient_results if p["has_injury_history"]),
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  5. EDUCATION ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════

def education_assessment(patient_id=None):
    """Patient/family education checklist: driving, swimming, pregnancy,
    alcohol, ketogenic diet, medication management, SUDEP awareness, etc."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        patients = _get_patients(conn, patient_id)
        seizures = _get_seizure_diary(conn, patient_id)
        meds = _get_medications(conn, patient_id)

        if not patients:
            return {
                "title": "Patient Education Assessment",
                "subtitle": "No patient data found",
                "patients": [],
                "education_domains": EDUCATION_DOMAINS,
            }

        seiz_by_patient = defaultdict(list)
        for s in seizures:
            seiz_by_patient[s["patient_id"]].append(s)

        meds_by_patient = defaultdict(list)
        for m in meds:
            meds_by_patient[m["patient_id"]].append(m)

        patient_results = []
        for p in patients:
            pid = p["patient_id"]
            p_seizures = seiz_by_patient.get(pid, [])
            p_meds = meds_by_patient.get(pid, [])
            age = p.get("age")
            gender = (p.get("gender") or "").lower()

            # Determine which education domains are priority for this patient
            domain_priorities = []
            for dom in EDUCATION_DOMAINS:
                priority = "standard"
                notes = []

                if dom["domain"] == "Driving restrictions":
                    if len(p_seizures) > 0:
                        priority = "urgent"
                        notes.append(f"Active seizures ({len(p_seizures)} recorded) — driving likely restricted")

                elif dom["domain"] == "Pregnancy counseling":
                    if gender == "female" and age and 15 <= age <= 50:
                        priority = "urgent"
                        # Check for valproate
                        drug_names = []
                        for m in p_meds:
                            f = m.get("fields", {})
                            dn = (f.get("drug_name") or "").lower()
                            if dn:
                                drug_names.append(dn)
                        if any("valp" in d for d in drug_names):
                            notes.append("ON VALPROATE — teratogenicity risk; discuss alternatives + folic acid 5mg/day")
                        else:
                            notes.append("Childbearing age — routine pre-conception counseling recommended")

                elif dom["domain"] == "SUDEP awareness":
                    severe = sum(1 for s in p_seizures if (s.get("severity") or "").lower() == "severe")
                    if severe > 0:
                        priority = "urgent"
                        notes.append(f"{severe} severe seizures — SUDEP risk discussion essential")
                    elif len(p_seizures) > 0:
                        priority = "recommended"

                elif dom["domain"] == "Water safety / Swimming":
                    if len(p_seizures) > 0:
                        priority = "urgent"
                        notes.append("Active seizures — drowning risk counseling mandatory")

                elif dom["domain"] == "Medication management":
                    n_drugs = 0
                    for m in p_meds:
                        f = m.get("fields", {})
                        if f.get("drug_name"):
                            n_drugs += 1
                    if n_drugs >= 2:
                        priority = "urgent"
                        notes.append(f"Polytherapy ({n_drugs} AEDs) — adherence education critical")
                    elif n_drugs >= 1:
                        priority = "recommended"

                elif dom["domain"] == "Seizure first aid (bystander education)":
                    if len(p_seizures) > 0:
                        priority = "urgent"
                        notes.append("Active seizures — family/carer first-aid training required")

                elif dom["domain"] == "Mental health awareness":
                    # Check PHQ9/GAD7
                    phq = _get_assessments(conn, pid, "PHQ9")
                    gad = _get_assessments(conn, pid, "GAD7")
                    if phq:
                        latest = max(phq, key=lambda x: x.get("created_at", ""))
                        if (latest.get("score") or 0) >= 10:
                            priority = "urgent"
                            notes.append(f"PHQ-9 score {latest['score']} — moderate+ depression; active mental health support needed")
                    if gad:
                        latest = max(gad, key=lambda x: x.get("created_at", ""))
                        if (latest.get("score") or 0) >= 10:
                            priority = "urgent"
                            notes.append(f"GAD-7 score {latest['score']} — moderate+ anxiety")

                else:
                    if len(p_seizures) > 0:
                        priority = "recommended"

                domain_priorities.append({
                    "domain": dom["domain"],
                    "description": dom["description"],
                    "reference": dom["reference"],
                    "risk_level": dom["risk_level"],
                    "priority_for_patient": priority,
                    "patient_specific_notes": notes,
                })

            # Education completion score (proportion of domains addressed)
            urgent_count = sum(1 for d in domain_priorities if d["priority_for_patient"] == "urgent")
            recommended_count = sum(1 for d in domain_priorities if d["priority_for_patient"] == "recommended")

            patient_results.append({
                "patient_id": pid,
                "age": age,
                "gender": p.get("gender"),
                "seizure_count": len(p_seizures),
                "education_domains": domain_priorities,
                "total_domains": len(domain_priorities),
                "urgent_domains": urgent_count,
                "recommended_domains": recommended_count,
                "standard_domains": len(domain_priorities) - urgent_count - recommended_count,
            })

        return {
            "title": "Patient Education Assessment",
            "subtitle": "Education needs checklist — NICE CG137, Epilepsy Foundation guidelines",
            "total_patients": len(patient_results),
            "education_domains_catalog": EDUCATION_DOMAINS,
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  6. DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def definitions():
    """Return metric definitions, clinical references, and scoring explanations."""
    return {
        "title": "Epilepsy Specialist Nurse — Metric Definitions",
        "modules": {
            "seizure_diary_analysis": {
                "purpose": "Analyze seizure diary entries: frequency, severity, triggers, temporal patterns, injury and ER visit rates",
                "data_source": "seizure_diary table in clinical.db — real patient seizure logs",
                "key_metrics": {
                    "monthly_frequency": "Seizures per 30-day period, calculated from diary date range",
                    "severity_distribution": "Mild / Moderate / Severe counts per ILAE classification proxy",
                    "injury_rate": "Proportion of seizures resulting in injury (fall, laceration, burn)",
                    "nocturnal_events": "Seizures between 22:00 and 06:00 — higher SUDEP risk",
                    "seizure_free_90_days": "Boolean: no recorded seizures in past 90 days (driving eligibility proxy)",
                },
            },
            "adherence_monitoring": {
                "purpose": "Assess AED adherence risk from polytherapy complexity, dosing burden, and seizure control",
                "data_source": "medications + seizure_diary tables in clinical.db",
                "key_metrics": {
                    "complexity_score": "0-10 composite: AED count + daily doses + titration + monitoring needs",
                    "adherence_risk": "Low (<4) / Moderate (4-6) / High (>=7) based on complexity score",
                    "polytherapy": "Boolean: 2+ concurrent AEDs (risk factor for non-adherence, Faught 2008)",
                },
                "reference": "Faught E et al. Neurology 2008;71:1572-1578 — non-adherence and mortality",
            },
            "sudep_risk_assessment": {
                "purpose": "SUDEP-7 composite risk score from published risk factors",
                "data_source": "patients + seizure_diary + medications tables in clinical.db",
                "scoring": {
                    "range": "0-10",
                    "low": "<3 — routine SUDEP information provision",
                    "moderate": "3-5.99 — active risk discussion + AED review",
                    "high": ">=6 — urgent counseling + referral to epilepsy center",
                },
                "factors": SUDEP_RISK_FACTORS,
                "references": [
                    "Hesdorffer DC et al. Epilepsia 2011;52:1150-1159",
                    "Devinsky O et al. Lancet Neurol 2016;15:1075-1088",
                    "Ryvlin P et al. Lancet Neurol 2013;12:966-977 (MORTEMUS)",
                ],
            },
            "seizure_action_plan": {
                "purpose": "Per-patient seizure first-aid plan with rescue medication, emergency criteria, recovery guidance",
                "data_source": "patients + seizure_diary + medications tables",
                "components": [
                    "First aid steps (Epilepsy Foundation consensus)",
                    "Emergency service criteria (5-min rule, repeat seizures, first seizure)",
                    "Rescue medications (buccal midazolam, rectal diazepam) per NICE CG137",
                    "Recovery position and post-ictal monitoring",
                    "Patient-specific safety alerts",
                ],
            },
            "education_assessment": {
                "purpose": "12-domain patient/family education checklist with patient-specific prioritization",
                "data_source": "patients + seizure_diary + medications + assessments tables",
                "domains": [d["domain"] for d in EDUCATION_DOMAINS],
                "priority_levels": {
                    "urgent": "Active risk — must be addressed at this visit",
                    "recommended": "Relevant to this patient — address when possible",
                    "standard": "General epilepsy education — routine provision",
                },
                "references": [
                    "NICE CG137 (2012, updated 2022): Epilepsies: diagnosis and management",
                    "Epilepsy Foundation patient education resources",
                    "ILAE Commission on Education",
                ],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  FULL DASHBOARD (combines all sub-analyses)
# ═══════════════════════════════════════════════════════════════════════

def full_dashboard(patient_id=None):
    """Return combined dashboard with all sub-analyses + summary KPIs."""
    diary = seizure_diary_analysis(patient_id)
    adherence = adherence_monitoring(patient_id)
    sudep = sudep_risk_assessment(patient_id)
    action = seizure_action_plan(patient_id)
    education = education_assessment(patient_id)

    # Summary KPIs
    total_seizures = diary.get("total_entries", 0)
    total_patients = sudep.get("total_patients", 0)
    high_sudep = sudep.get("high_risk_count", 0)
    high_adherence_risk = adherence.get("high_risk_count", 0)
    status_risk = action.get("patients_with_status_risk", 0)

    return {
        "title": "Epilepsy Specialist Nurse Dashboard",
        "subtitle": "Seizure diary, AED adherence, SUDEP risk, action plans, education — all from real clinical.db data",
        "summary": {
            "total_patients": total_patients,
            "total_seizure_diary_entries": total_seizures,
            "patients_with_diary": diary.get("patients_with_diary", 0),
            "high_sudep_risk": high_sudep,
            "moderate_sudep_risk": sudep.get("moderate_risk_count", 0),
            "high_adherence_risk": high_adherence_risk,
            "patients_with_status_risk": status_risk,
            "patients_with_injury_history": action.get("patients_with_injury_history", 0),
            "total_er_visits": diary.get("total_er_visits", 0),
        },
        "seizure_diary": diary,
        "adherence": adherence,
        "sudep_risk": sudep,
        "action_plans": action,
        "education": education,
    }
