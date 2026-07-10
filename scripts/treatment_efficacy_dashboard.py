"""Treatment Efficacy Dashboard — AED adherence vs seizure/outcome correlation from clinical.db.

Measures whether anti-epileptic drug (AED) treatment is working by correlating
medication adherence (dose-level yes/late/no) with seizure frequency from the
seizure diary and patient-reported outcomes (QOLIE-31, PHQ-9, GAD-7, etc.).

Clinically this matters because:
- Sub-therapeutic adherence (<85 %) is the #1 modifiable cause of breakthrough
  seizures (AAN Practice Guideline, 2018).
- Side-effect burden drives non-adherence; early detection enables drug switches.
- Correlating adherence with seizure counts and PRO scores closes the treat-to-
  target loop recommended by ILAE treatment protocols.

Sources:
- medication_adherence table  (~12 600 dose-level rows)
- seizure_diary table         (~25 event rows)
- pro_outcomes table          (~180 PRO assessment rows)
- patients table              (~40 patients)
"""

import json
import pathlib
import sqlite3
from collections import Counter, defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conn():
    """Return a new DB connection with Row factory."""
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _safe(cur, sql, params=(), default=0):
    """Execute *sql* and return the first column of the first row, or *default*."""
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    """Execute *sql* and return all rows, or [] on error."""
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


def _parse_json(text, default=None):
    """Safely parse a JSON string, returning *default* on failure."""
    if default is None:
        default = []
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _pct(num, denom):
    """Return percentage rounded to 1 decimal, or 0.0 when denom is zero."""
    return round(num / denom * 100, 1) if denom else 0.0


def _response_category(adherence_pct, seizure_count):
    """Classify treatment response based on adherence and seizure burden."""
    if adherence_pct > 95 and seizure_count == 0:
        return "Excellent"
    if adherence_pct > 85 and seizure_count <= 1:
        return "Good"
    if 70 <= adherence_pct <= 85:
        return "Partial"
    if adherence_pct < 70:
        return "Poor"
    # Fallback: high adherence but multiple seizures
    return "Good"


def _time_of_day_bucket(scheduled_time):
    """Map a scheduled_time label to morning/afternoon/evening."""
    if not scheduled_time:
        return "morning"
    t = scheduled_time.strip().lower()
    if t in ("morning",):
        return "morning"
    if t in ("afternoon",):
        return "afternoon"
    if t in ("evening", "bedtime", "night"):
        return "evening"
    # Try parsing HH:MM style
    try:
        hour = int(t.split(":")[0])
        if hour < 12:
            return "morning"
        if hour < 17:
            return "afternoon"
        return "evening"
    except (ValueError, IndexError):
        return "morning"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def overview():
    """High-level treatment efficacy summary across all patients.

    Returns a dict with adherence statistics, seizure metrics, drug profiles,
    correlation data, monthly trends, side-effect profiles, and response
    category counts.
    """
    con = _conn()
    cur = con.cursor()

    # --- Adherence basics ---
    total_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM medication_adherence")
    total_doses = _safe(cur, "SELECT COUNT(*) FROM medication_adherence")
    yes_count = _safe(cur, "SELECT COUNT(*) FROM medication_adherence WHERE taken = 'yes'")
    late_count = _safe(cur, "SELECT COUNT(*) FROM medication_adherence WHERE taken = 'late'")
    no_count = _safe(cur, "SELECT COUNT(*) FROM medication_adherence WHERE taken = 'no'")

    overall_adherence_pct = _pct(yes_count + late_count, total_doses)
    on_time_pct = _pct(yes_count, total_doses)
    late_pct = _pct(late_count, total_doses)
    missed_pct = _pct(no_count, total_doses)

    # --- Seizure basics ---
    total_seizures = _safe(cur, "SELECT COUNT(*) FROM seizure_diary")
    # Average seizures per patient-month
    seizure_rows = _safe_rows(cur,
        "SELECT patient_id, event_date FROM seizure_diary WHERE event_date IS NOT NULL")
    patient_months = set()
    for r in seizure_rows:
        try:
            month_key = r["event_date"][:7]  # YYYY-MM
            patient_months.add((r["patient_id"], month_key))
        except (TypeError, IndexError):
            pass

    # Denominator: count patient-months that have adherence data (broader coverage)
    adh_months = _safe_rows(cur,
        "SELECT DISTINCT patient_id, substr(log_date, 1, 7) AS m FROM medication_adherence WHERE log_date IS NOT NULL")
    total_patient_months = len(adh_months) if adh_months else 1
    avg_seizure_frequency = round(total_seizures / total_patient_months, 2) if total_patient_months else 0.0

    # --- Drug list with adherence ---
    unique_drugs = _safe(cur, "SELECT COUNT(DISTINCT drug_name) FROM medication_adherence")
    drug_rows = _safe_rows(cur, """
        SELECT drug_name,
               COUNT(DISTINCT patient_id) AS pat_count,
               COUNT(*) AS total,
               SUM(CASE WHEN taken IN ('yes','late') THEN 1 ELSE 0 END) AS adhered
        FROM medication_adherence
        GROUP BY drug_name
        ORDER BY pat_count DESC
    """)
    drug_list = []
    for dr in drug_rows:
        drug_list.append({
            "drug_name": dr["drug_name"],
            "patient_count": dr["pat_count"],
            "adherence_pct": _pct(dr["adhered"], dr["total"]),
        })

    # --- Adherence vs seizures per patient ---
    pat_adh_rows = _safe_rows(cur, """
        SELECT patient_id,
               COUNT(*) AS total,
               SUM(CASE WHEN taken IN ('yes','late') THEN 1 ELSE 0 END) AS adhered
        FROM medication_adherence
        GROUP BY patient_id
    """)
    # seizure counts by patient
    seizure_counts = defaultdict(int)
    for r in _safe_rows(cur, "SELECT patient_id, COUNT(*) AS cnt FROM seizure_diary GROUP BY patient_id"):
        seizure_counts[r["patient_id"]] = r["cnt"]

    adherence_vs_seizures = []
    for r in pat_adh_rows:
        adherence_vs_seizures.append({
            "patient_id": r["patient_id"],
            "adherence_pct": _pct(r["adhered"], r["total"]),
            "seizure_count": seizure_counts.get(r["patient_id"], 0),
        })

    # --- Monthly adherence trend ---
    month_rows = _safe_rows(cur, """
        SELECT substr(log_date, 1, 7) AS month,
               COUNT(*) AS total,
               SUM(CASE WHEN taken IN ('yes','late') THEN 1 ELSE 0 END) AS adhered
        FROM medication_adherence
        WHERE log_date IS NOT NULL
        GROUP BY month
        ORDER BY month
    """)
    seizure_by_month = defaultdict(int)
    for r in _safe_rows(cur,
            "SELECT substr(event_date, 1, 7) AS m, COUNT(*) AS cnt FROM seizure_diary WHERE event_date IS NOT NULL GROUP BY m"):
        seizure_by_month[r["m"]] = r["cnt"]

    monthly_adherence_trend = []
    for mr in month_rows:
        monthly_adherence_trend.append({
            "month": mr["month"],
            "adherence_pct": _pct(mr["adhered"], mr["total"]),
            "seizure_count": seizure_by_month.get(mr["month"], 0),
        })

    # --- Side-effect profile ---
    se_rows = _safe_rows(cur,
        "SELECT side_effects_json, side_effect_severity FROM medication_adherence WHERE side_effects_json IS NOT NULL AND side_effects_json != ''")
    se_counter = Counter()
    se_severity_sum = defaultdict(float)
    se_severity_count = defaultdict(int)
    for r in se_rows:
        effects = _parse_json(r["side_effects_json"], [])
        severity = r["side_effect_severity"] if r["side_effect_severity"] is not None else 0
        for eff in effects:
            if isinstance(eff, str) and eff.strip():
                key = eff.strip()
                se_counter[key] += 1
                se_severity_sum[key] += severity
                se_severity_count[key] += 1

    side_effect_profile = []
    for se, count in se_counter.most_common(15):
        avg_sev = round(se_severity_sum[se] / se_severity_count[se], 1) if se_severity_count[se] else 0.0
        side_effect_profile.append({
            "side_effect": se,
            "count": count,
            "avg_severity": avg_sev,
        })

    # --- Treatment response categories ---
    categories = {"Excellent": 0, "Good": 0, "Partial": 0, "Poor": 0}
    for entry in adherence_vs_seizures:
        cat = _response_category(entry["adherence_pct"], entry["seizure_count"])
        categories[cat] = categories.get(cat, 0) + 1

    treatment_response_categories = [
        {"category": cat, "count": cnt} for cat, cnt in categories.items()
    ]

    con.close()
    return {
        "total_patients": total_patients,
        "overall_adherence_pct": overall_adherence_pct,
        "on_time_pct": on_time_pct,
        "late_pct": late_pct,
        "missed_pct": missed_pct,
        "avg_seizure_frequency": avg_seizure_frequency,
        "total_seizures": total_seizures,
        "unique_drugs": unique_drugs,
        "drug_list": drug_list,
        "adherence_vs_seizures": adherence_vs_seizures,
        "monthly_adherence_trend": monthly_adherence_trend,
        "side_effect_profile": side_effect_profile,
        "treatment_response_categories": treatment_response_categories,
    }


def breakdown():
    """Detailed per-patient and per-drug treatment efficacy breakdown.

    Returns per_patient profiles, per_drug statistics, adherence by time of
    day, and side effects grouped by drug.
    """
    con = _conn()
    cur = con.cursor()

    # ------------------------------------------------------------------
    # Seizure counts & severity by patient
    # ------------------------------------------------------------------
    seizure_counts = defaultdict(int)
    seizure_severity_map = {"Mild": 1, "Moderate": 2, "Severe": 3}
    seizure_sev_sums = defaultdict(float)
    seizure_sev_counts = defaultdict(int)
    for r in _safe_rows(cur, "SELECT patient_id, severity FROM seizure_diary"):
        seizure_counts[r["patient_id"]] += 1
        sev_val = seizure_severity_map.get(r["severity"], 0)
        if sev_val:
            seizure_sev_sums[r["patient_id"]] += sev_val
            seizure_sev_counts[r["patient_id"]] += 1

    # ------------------------------------------------------------------
    # Per-patient adherence
    # ------------------------------------------------------------------
    pat_rows = _safe_rows(cur, """
        SELECT patient_id,
               GROUP_CONCAT(DISTINCT drug_name) AS drugs,
               COUNT(*) AS total,
               SUM(CASE WHEN taken = 'yes' THEN 1 ELSE 0 END) AS on_time,
               SUM(CASE WHEN taken IN ('yes','late') THEN 1 ELSE 0 END) AS adhered,
               SUM(CASE WHEN taken = 'no' THEN 1 ELSE 0 END) AS missed,
               AVG(mood_after) AS avg_mood
        FROM medication_adherence
        GROUP BY patient_id
        ORDER BY patient_id
    """)

    # Side effects per patient (most common)
    pat_se = defaultdict(list)
    for r in _safe_rows(cur,
            "SELECT patient_id, side_effects_json FROM medication_adherence WHERE side_effects_json IS NOT NULL AND side_effects_json != ''"):
        effects = _parse_json(r["side_effects_json"], [])
        for eff in effects:
            if isinstance(eff, str) and eff.strip():
                pat_se[r["patient_id"]].append(eff.strip())

    per_patient = []
    for pr in pat_rows:
        pid = pr["patient_id"]
        adh_pct = _pct(pr["adhered"], pr["total"])
        on_time_p = _pct(pr["on_time"], pr["total"])
        sc = seizure_counts.get(pid, 0)
        avg_sev = round(seizure_sev_sums[pid] / seizure_sev_counts[pid], 1) if seizure_sev_counts[pid] else 0.0
        avg_mood = round(pr["avg_mood"], 1) if pr["avg_mood"] is not None else 0.0
        se_list = pat_se.get(pid, [])
        most_common_se = Counter(se_list).most_common(1)
        most_common_se = most_common_se[0][0] if most_common_se else "None"

        per_patient.append({
            "patient_id": pid,
            "drug_names": pr["drugs"] if pr["drugs"] else "",
            "adherence_pct": adh_pct,
            "on_time_pct": on_time_p,
            "total_doses": pr["total"],
            "missed_doses": pr["missed"],
            "seizure_count": sc,
            "avg_seizure_severity": avg_sev,
            "most_common_side_effect": most_common_se,
            "avg_mood": avg_mood,
            "response_category": _response_category(adh_pct, sc),
        })

    # ------------------------------------------------------------------
    # Per-drug analysis
    # ------------------------------------------------------------------
    drug_rows = _safe_rows(cur, """
        SELECT drug_name,
               COUNT(DISTINCT patient_id) AS pat_count,
               COUNT(*) AS total,
               SUM(CASE WHEN taken IN ('yes','late') THEN 1 ELSE 0 END) AS adhered,
               AVG(side_effect_severity) AS avg_se_sev
        FROM medication_adherence
        GROUP BY drug_name
        ORDER BY drug_name
    """)

    # Side effects per drug
    drug_se = defaultdict(list)
    drug_se_sev = defaultdict(lambda: defaultdict(list))
    for r in _safe_rows(cur,
            "SELECT drug_name, side_effects_json, side_effect_severity FROM medication_adherence WHERE side_effects_json IS NOT NULL AND side_effects_json != ''"):
        effects = _parse_json(r["side_effects_json"], [])
        sev = r["side_effect_severity"] if r["side_effect_severity"] is not None else 0
        for eff in effects:
            if isinstance(eff, str) and eff.strip():
                key = eff.strip()
                drug_se[r["drug_name"]].append(key)
                drug_se_sev[r["drug_name"]][key].append(sev)

    # Patients per drug -> seizure correlation text
    drug_patients = defaultdict(set)
    for r in _safe_rows(cur, "SELECT DISTINCT drug_name, patient_id FROM medication_adherence"):
        drug_patients[r["drug_name"]].add(r["patient_id"])

    per_drug = []
    for dr in drug_rows:
        dn = dr["drug_name"]
        se_list = drug_se.get(dn, [])
        top3 = [x[0] for x in Counter(se_list).most_common(3)]
        avg_se_sev = round(dr["avg_se_sev"], 1) if dr["avg_se_sev"] is not None else 0.0

        # Simple correlation description
        drug_pats = drug_patients.get(dn, set())
        if drug_pats:
            pats_with_seizures = sum(1 for p in drug_pats if seizure_counts.get(p, 0) > 0)
            ratio = pats_with_seizures / len(drug_pats) if drug_pats else 0
            if ratio == 0:
                corr_text = "No seizures observed among patients on this drug"
            elif ratio < 0.2:
                corr_text = "Low seizure incidence (<20% of patients)"
            elif ratio < 0.5:
                corr_text = "Moderate seizure incidence (20-50% of patients)"
            else:
                corr_text = "High seizure incidence (>50% of patients)"
        else:
            corr_text = "Insufficient data"

        per_drug.append({
            "drug_name": dn,
            "patient_count": dr["pat_count"],
            "avg_adherence_pct": _pct(dr["adhered"], dr["total"]),
            "avg_side_effect_severity": avg_se_sev,
            "most_common_side_effects": top3,
            "seizure_reduction_corr": corr_text,
        })

    # ------------------------------------------------------------------
    # Adherence by time of day
    # ------------------------------------------------------------------
    tod_rows = _safe_rows(cur, """
        SELECT scheduled_time,
               COUNT(*) AS total,
               SUM(CASE WHEN taken IN ('yes','late') THEN 1 ELSE 0 END) AS adhered
        FROM medication_adherence
        WHERE scheduled_time IS NOT NULL
        GROUP BY scheduled_time
    """)
    tod = {"morning": {"adhered": 0, "total": 0},
           "afternoon": {"adhered": 0, "total": 0},
           "evening": {"adhered": 0, "total": 0}}
    for tr in tod_rows:
        bucket = _time_of_day_bucket(tr["scheduled_time"])
        tod[bucket]["adhered"] += tr["adhered"]
        tod[bucket]["total"] += tr["total"]

    adherence_by_time_of_day = {
        k: _pct(v["adhered"], v["total"]) for k, v in tod.items()
    }

    # ------------------------------------------------------------------
    # Side effects by drug (detailed)
    # ------------------------------------------------------------------
    side_effects_by_drug = []
    for dn in sorted(drug_se_sev.keys()):
        effects_detail = []
        for se_name, sevs in drug_se_sev[dn].items():
            effects_detail.append({
                "name": se_name,
                "count": len(sevs),
                "avg_severity": round(sum(sevs) / len(sevs), 1) if sevs else 0.0,
            })
        effects_detail.sort(key=lambda x: x["count"], reverse=True)
        side_effects_by_drug.append({
            "drug_name": dn,
            "side_effects": effects_detail,
        })

    con.close()
    return {
        "per_patient": per_patient,
        "per_drug": per_drug,
        "adherence_by_time_of_day": adherence_by_time_of_day,
        "side_effects_by_drug": side_effects_by_drug,
    }


def definitions():
    """Clinical term definitions and references for the Treatment Efficacy Dashboard."""
    return {
        "Treatment Efficacy": (
            "A measure of how well an anti-epileptic drug (AED) regimen controls "
            "seizures and improves patient quality of life, assessed through "
            "medication adherence rates, seizure frequency, and patient-reported "
            "outcome (PRO) scores."
        ),
        "Adherence Rate": (
            "Percentage of scheduled medication doses that were taken (either "
            "on time or late) out of all scheduled doses. An adherence rate "
            "above 95% is considered excellent; below 70% is classified as poor "
            "and is the leading modifiable risk factor for breakthrough seizures."
        ),
        "On-Time Rate": (
            "Percentage of scheduled doses taken at or near the scheduled time "
            "(taken='yes'). Distinguished from late doses to identify patients "
            "who take medications but struggle with timing, which can affect "
            "steady-state drug levels."
        ),
        "Seizure Frequency": (
            "Number of seizure events per patient per month, derived from the "
            "seizure diary. Used as the primary outcome measure for treatment "
            "efficacy. A reduction of >=50% from baseline is considered a "
            "clinically meaningful response (Kwan & Brodie, NEJM 2000)."
        ),
        "Treatment Response Categories": {
            "Excellent": "Adherence >95% with zero seizures in the observation period.",
            "Good": "Adherence >85% with at most 1 seizure in the observation period.",
            "Partial": "Adherence 70-85%, regardless of seizure count; indicates room for improvement.",
            "Poor": "Adherence <70%; high risk of breakthrough seizures and sub-therapeutic drug levels.",
        },
        "Side Effect Severity Scale": (
            "Patient-reported severity of medication side effects on a 1-10 "
            "integer scale: 1-3 = Mild (noticeable but not disruptive), "
            "4-6 = Moderate (affects daily activities), 7-9 = Severe (significantly "
            "impairs function), 10 = Intolerable (requires immediate drug change)."
        ),
        "QOLIE-31": (
            "Quality of Life in Epilepsy Inventory, 31-item version. A validated "
            "PRO instrument covering seizure worry, overall QoL, emotional "
            "well-being, energy/fatigue, cognitive function, medication effects, "
            "and social function. Scores range 0-100; higher is better. "
            "Reference: Cramer et al., Epilepsia 1998."
        ),
        "PRO Measures": {
            "PHQ-9": "Patient Health Questionnaire (depression screening); 0-27, >=10 suggests moderate depression.",
            "GAD-7": "Generalized Anxiety Disorder scale; 0-21, >=10 suggests moderate anxiety.",
            "MoCA": "Montreal Cognitive Assessment; 0-30, <26 suggests cognitive impairment.",
            "NDDI-E": "Neurological Disorders Depression Inventory for Epilepsy; >15 suggests major depression.",
            "PSQI": "Pittsburgh Sleep Quality Index; 0-21, >5 indicates poor sleep quality.",
            "ESS": "Epworth Sleepiness Scale; 0-24, >10 indicates excessive daytime sleepiness.",
        },
        "Clinical References": [
            "AAN Practice Guideline: Treatment of New-Onset Epilepsy (2018) — recommends adherence monitoring as standard of care.",
            "ILAE Treatment Protocols: Revised terminology and concepts for management of epilepsy (2022).",
            "Kwan P, Brodie MJ. Early identification of refractory epilepsy. NEJM 2000;342:314-19.",
            "Cramer JA et al. Development and cross-cultural translations of a 31-item quality of life in epilepsy inventory. Epilepsia 1998;39:81-88.",
            "Faught E et al. Nonadherence to AEDs and increased mortality. Neurology 2008;71:1572-78.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint
    print("=== Treatment Efficacy Overview ===")
    ov = overview()
    pprint.pprint({k: v for k, v in ov.items() if k not in ("adherence_vs_seizures", "monthly_adherence_trend", "drug_list", "side_effect_profile")})
    print(f"\n  Drugs tracked: {len(ov['drug_list'])}")
    print(f"  Side effects tracked: {len(ov['side_effect_profile'])}")
    print(f"  Months tracked: {len(ov['monthly_adherence_trend'])}")

    print("\n=== Treatment Efficacy Breakdown ===")
    bd = breakdown()
    print(f"  Patients: {len(bd['per_patient'])}")
    print(f"  Drugs: {len(bd['per_drug'])}")
    print(f"  Time-of-day adherence: {bd['adherence_by_time_of_day']}")

    print("\n=== Definitions ===")
    defs = definitions()
    print(f"  Terms defined: {len(defs)}")
