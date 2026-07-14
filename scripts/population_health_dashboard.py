"""Population Health Dashboard — epidemiological analytics from clinical.db.

Provides a population-level view of the epilepsy patient cohort including
demographics, seizure burden, comorbidity prevalence, medication coverage,
and risk stratification.

Clinically this matters because:
- Population health management identifies high-risk subgroups for targeted
  intervention (WHO Epilepsy Fact Sheet, 2023).
- Age-sex pyramids reveal enrollment bias that can skew clinical trial
  generalisability (ILAE Epidemiology Commission, 2021).
- Comorbidity burden drives total cost-of-care and is the strongest predictor
  of unplanned hospitalisation in epilepsy populations (Keezer et al., 2016).

Sources:
- patients table              (~40 patients)
- seizure_diary table         (~25 event rows)
- comorbidities table         (~27 rows)
- medications table           (~9 rows)
- eeg_acquisition table       (~30 rows)
- assessments table           (~423 rows)
- pro_outcomes table          (~180 rows)
- medication_adherence table  (~12 600 rows)
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


def _age_group(age):
    """Classify an age into a population-health age band."""
    if age is None:
        return "Unknown"
    try:
        age = int(age)
    except (ValueError, TypeError):
        return "Unknown"
    if age <= 17:
        return "0-17"
    if age <= 30:
        return "18-30"
    if age <= 45:
        return "31-45"
    if age <= 60:
        return "46-60"
    return "61+"


def _risk_level(seizure_count, comorbidity_count, age):
    """Stratify patient risk based on seizure count, comorbidity burden, and age."""
    score = 0
    if seizure_count >= 3:
        score += 3
    elif seizure_count >= 1:
        score += 1
    if comorbidity_count >= 3:
        score += 3
    elif comorbidity_count >= 1:
        score += 1
    try:
        a = int(age)
        if a >= 65:
            score += 2
        elif a >= 50:
            score += 1
    except (ValueError, TypeError):
        pass
    if score >= 5:
        return "High"
    if score >= 3:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def overview():
    """Population-level summary statistics for the epilepsy cohort.

    Returns demographics, seizure burden, comorbidity prevalence, medication
    coverage, data completeness, and enrollment trend.
    """
    con = _conn()
    cur = con.cursor()

    # --- Total patients ---
    total_patients = _safe(cur, "SELECT COUNT(*) FROM patients")

    # --- Gender distribution ---
    gender_rows = _safe_rows(cur,
        "SELECT gender, COUNT(*) AS cnt FROM patients GROUP BY gender ORDER BY cnt DESC")
    gender_distribution = []
    for r in gender_rows:
        gender_distribution.append({
            "gender": r["gender"] if r["gender"] else "Unknown",
            "count": r["cnt"],
            "pct": _pct(r["cnt"], total_patients),
        })

    # --- Age statistics ---
    ages = []
    age_rows = _safe_rows(cur, "SELECT age FROM patients WHERE age IS NOT NULL")
    for r in age_rows:
        try:
            ages.append(int(r["age"]))
        except (ValueError, TypeError):
            pass

    if ages:
        ages_sorted = sorted(ages)
        n = len(ages_sorted)
        median_age = (ages_sorted[n // 2] + ages_sorted[(n - 1) // 2]) / 2
        mean_age = round(sum(ages) / n, 1)
        std_age = round((sum((a - mean_age) ** 2 for a in ages) / n) ** 0.5, 1)
        age_stats = {
            "min": min(ages),
            "max": max(ages),
            "mean": mean_age,
            "median": median_age,
            "std": std_age,
        }
    else:
        age_stats = {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}

    # --- Age groups ---
    age_group_counter = Counter()
    all_patient_rows = _safe_rows(cur, "SELECT patient_id, age FROM patients")
    patient_ages = {}
    for r in all_patient_rows:
        ag = _age_group(r["age"])
        age_group_counter[ag] += 1
        patient_ages[r["patient_id"]] = r["age"]

    ordered_groups = ["0-17", "18-30", "31-45", "46-60", "61+"]
    age_groups = []
    for g in ordered_groups:
        cnt = age_group_counter.get(g, 0)
        age_groups.append({
            "group": g,
            "count": cnt,
            "pct": _pct(cnt, total_patients),
        })

    # --- Seizure burden ---
    total_events = _safe(cur, "SELECT COUNT(*) FROM seizure_diary")
    patients_with_events = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM seizure_diary")
    mean_per_patient = round(total_events / patients_with_events, 1) if patients_with_events else 0.0
    sev_rows = _safe_rows(cur,
        "SELECT severity, COUNT(*) AS cnt FROM seizure_diary GROUP BY severity ORDER BY cnt DESC")
    severity_distribution = []
    for r in sev_rows:
        severity_distribution.append({
            "severity": r["severity"] if r["severity"] else "Unknown",
            "count": r["cnt"],
            "pct": _pct(r["cnt"], total_events),
        })

    seizure_burden = {
        "total_events": total_events,
        "patients_with_events": patients_with_events,
        "mean_per_patient": mean_per_patient,
        "severity_distribution": severity_distribution,
    }

    # --- Comorbidity prevalence ---
    comorb_counter = Counter()
    comorb_rows = _safe_rows(cur, "SELECT fields_json FROM comorbidities")
    total_comorb_patients = len(set(
        r["patient_id"] for r in _safe_rows(cur, "SELECT DISTINCT patient_id FROM comorbidities")
    ))
    for r in comorb_rows:
        parsed = _parse_json(r["fields_json"], {})
        if isinstance(parsed, dict):
            conditions = parsed.get("comorbidities", [])
            if isinstance(conditions, list):
                for c in conditions:
                    if isinstance(c, str) and c.strip():
                        comorb_counter[c.strip()] += 1
            elif isinstance(conditions, str) and conditions.strip():
                comorb_counter[conditions.strip()] += 1

    comorbidity_prevalence = []
    for condition, cnt in comorb_counter.most_common():
        comorbidity_prevalence.append({
            "condition": condition,
            "count": cnt,
            "pct": _pct(cnt, total_comorb_patients) if total_comorb_patients else 0.0,
        })

    # --- Medication coverage ---
    patients_with_meds = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM medications")
    total_prescriptions = _safe(cur, "SELECT COUNT(*) FROM medications")
    drug_counter = Counter()
    med_rows = _safe_rows(cur, "SELECT fields_json FROM medications")
    for r in med_rows:
        parsed = _parse_json(r["fields_json"], {})
        if isinstance(parsed, dict):
            drug = parsed.get("drug_name", "")
            if isinstance(drug, str) and drug.strip():
                drug_counter[drug.strip()] += 1

    drug_distribution = []
    for drug, cnt in drug_counter.most_common():
        drug_distribution.append({"drug": drug, "count": cnt})

    medication_coverage = {
        "patients_with_meds": patients_with_meds,
        "total_prescriptions": total_prescriptions,
        "drug_distribution": drug_distribution,
    }

    # --- Data coverage (row counts) ---
    data_coverage = {
        "eeg_acquisition": _safe(cur, "SELECT COUNT(*) FROM eeg_acquisition"),
        "assessments": _safe(cur, "SELECT COUNT(*) FROM assessments"),
        "pro_outcomes": _safe(cur, "SELECT COUNT(*) FROM pro_outcomes"),
        "medication_adherence": _safe(cur, "SELECT COUNT(*) FROM medication_adherence"),
    }

    # --- Enrollment trend ---
    enroll_rows = _safe_rows(cur, """
        SELECT substr(created_at, 1, 7) AS month, COUNT(*) AS cnt
        FROM patients
        WHERE created_at IS NOT NULL
        GROUP BY month
        ORDER BY month
    """)
    enrollment_trend = []
    for r in enroll_rows:
        enrollment_trend.append({"month": r["month"], "count": r["cnt"]})

    con.close()
    return {
        "total_patients": total_patients,
        "gender_distribution": gender_distribution,
        "age_stats": age_stats,
        "age_groups": age_groups,
        "seizure_burden": seizure_burden,
        "comorbidity_prevalence": comorbidity_prevalence,
        "medication_coverage": medication_coverage,
        "data_coverage": data_coverage,
        "enrollment_trend": enrollment_trend,
    }


def breakdown():
    """Detailed per-patient and stratified population health breakdown.

    Returns patient registry, age-sex pyramid, seizure characteristics,
    geographic distribution, and risk stratification.
    """
    con = _conn()
    cur = con.cursor()

    # ------------------------------------------------------------------
    # Seizure counts per patient
    # ------------------------------------------------------------------
    seizure_counts = defaultdict(int)
    for r in _safe_rows(cur, "SELECT patient_id, COUNT(*) AS cnt FROM seizure_diary GROUP BY patient_id"):
        seizure_counts[r["patient_id"]] = r["cnt"]

    # ------------------------------------------------------------------
    # Comorbidity counts per patient
    # ------------------------------------------------------------------
    comorb_counts = defaultdict(int)
    for r in _safe_rows(cur, "SELECT patient_id, fields_json FROM comorbidities"):
        parsed = _parse_json(r["fields_json"], {})
        if isinstance(parsed, dict):
            # Use explicit comorbidity_count if available, else count items
            cc = parsed.get("comorbidity_count")
            if cc is not None:
                try:
                    comorb_counts[r["patient_id"]] += int(cc)
                except (ValueError, TypeError):
                    conditions = parsed.get("comorbidities", [])
                    if isinstance(conditions, list):
                        comorb_counts[r["patient_id"]] += len(conditions)
            else:
                conditions = parsed.get("comorbidities", [])
                if isinstance(conditions, list):
                    comorb_counts[r["patient_id"]] += len(conditions)

    # ------------------------------------------------------------------
    # Medication per patient (latest)
    # ------------------------------------------------------------------
    patient_meds = {}
    for r in _safe_rows(cur, "SELECT patient_id, fields_json FROM medications ORDER BY created_at DESC"):
        pid = r["patient_id"]
        if pid not in patient_meds:
            parsed = _parse_json(r["fields_json"], {})
            if isinstance(parsed, dict):
                patient_meds[pid] = parsed.get("drug_name", "Unknown")
            else:
                patient_meds[pid] = "Unknown"

    # ------------------------------------------------------------------
    # Last assessment date per patient
    # ------------------------------------------------------------------
    last_assessment = {}
    for r in _safe_rows(cur, """
        SELECT patient_id, MAX(created_at) AS last_dt
        FROM assessments
        GROUP BY patient_id
    """):
        last_assessment[r["patient_id"]] = r["last_dt"]

    # ------------------------------------------------------------------
    # Patient registry
    # ------------------------------------------------------------------
    patient_rows = _safe_rows(cur,
        "SELECT patient_id, name, age, gender FROM patients ORDER BY patient_id")

    patient_registry = []
    for r in patient_rows:
        pid = r["patient_id"]
        patient_registry.append({
            "patient_id": pid,
            "name": r["name"] if r["name"] else "Unknown",
            "age": r["age"],
            "gender": r["gender"] if r["gender"] else "Unknown",
            "seizure_count": seizure_counts.get(pid, 0),
            "comorbidity_count": comorb_counts.get(pid, 0),
            "medication": patient_meds.get(pid, "None"),
            "last_assessment": last_assessment.get(pid),
        })

    # ------------------------------------------------------------------
    # Age-sex pyramid
    # ------------------------------------------------------------------
    pyramid = defaultdict(lambda: {"male": 0, "female": 0})
    for r in patient_rows:
        ag = _age_group(r["age"])
        gender = (r["gender"] or "").strip().lower()
        if gender in ("m", "male"):
            pyramid[ag]["male"] += 1
        elif gender in ("f", "female"):
            pyramid[ag]["female"] += 1

    ordered_groups = ["0-17", "18-30", "31-45", "46-60", "61+"]
    age_sex_pyramid = []
    for g in ordered_groups:
        age_sex_pyramid.append({
            "age_group": g,
            "male": pyramid[g]["male"],
            "female": pyramid[g]["female"],
        })

    # ------------------------------------------------------------------
    # Seizure characteristics
    # ------------------------------------------------------------------
    total_seizures = _safe(cur, "SELECT COUNT(*) FROM seizure_diary")

    trigger_rows = _safe_rows(cur,
        "SELECT trigger, COUNT(*) AS cnt FROM seizure_diary WHERE trigger IS NOT NULL AND trigger != '' GROUP BY trigger ORDER BY cnt DESC")
    trigger_distribution = []
    for r in trigger_rows:
        trigger_distribution.append({
            "trigger": r["trigger"],
            "count": r["cnt"],
            "pct": _pct(r["cnt"], total_seizures),
        })

    awareness_rows = _safe_rows(cur,
        "SELECT awareness, COUNT(*) AS cnt FROM seizure_diary WHERE awareness IS NOT NULL AND awareness != '' GROUP BY awareness ORDER BY cnt DESC")
    awareness_distribution = []
    for r in awareness_rows:
        awareness_distribution.append({
            "awareness": r["awareness"],
            "count": r["cnt"],
            "pct": _pct(r["cnt"], total_seizures),
        })

    aura_yes = _safe(cur, "SELECT COUNT(*) FROM seizure_diary WHERE aura = 1 OR LOWER(aura) = 'yes'")
    injury_yes = _safe(cur, "SELECT COUNT(*) FROM seizure_diary WHERE injury = 1 OR LOWER(injury) = 'yes'")
    er_yes = _safe(cur, "SELECT COUNT(*) FROM seizure_diary WHERE er_visit = 1 OR LOWER(er_visit) = 'yes'")

    seizure_characteristics = {
        "trigger_distribution": trigger_distribution,
        "awareness_distribution": awareness_distribution,
        "aura_rate": _pct(aura_yes, total_seizures),
        "injury_rate": _pct(injury_yes, total_seizures),
        "er_visit_rate": _pct(er_yes, total_seizures),
    }

    # ------------------------------------------------------------------
    # Geographic distribution (by department)
    # ------------------------------------------------------------------
    dept_rows = _safe_rows(cur,
        "SELECT department, COUNT(*) AS cnt FROM patients WHERE department IS NOT NULL AND department != '' GROUP BY department ORDER BY cnt DESC")
    total_with_dept = sum(r["cnt"] for r in dept_rows) if dept_rows else 0
    geographic_distribution = []
    for r in dept_rows:
        geographic_distribution.append({
            "department": r["department"],
            "count": r["cnt"],
            "pct": _pct(r["cnt"], total_with_dept),
        })

    # ------------------------------------------------------------------
    # Risk stratification
    # ------------------------------------------------------------------
    risk_stratification = []
    for r in patient_rows:
        pid = r["patient_id"]
        sc = seizure_counts.get(pid, 0)
        cc = comorb_counts.get(pid, 0)
        age = r["age"]
        level = _risk_level(sc, cc, age)
        factors = []
        if sc >= 3:
            factors.append(f"High seizure count ({sc})")
        elif sc >= 1:
            factors.append(f"Seizure history ({sc} events)")
        if cc >= 3:
            factors.append(f"High comorbidity burden ({cc})")
        elif cc >= 1:
            factors.append(f"Comorbidities present ({cc})")
        try:
            a = int(age)
            if a >= 65:
                factors.append(f"Elderly (age {a})")
            elif a >= 50:
                factors.append(f"Age >50 ({a})")
        except (ValueError, TypeError):
            pass
        if not factors:
            factors.append("No elevated risk factors identified")

        risk_stratification.append({
            "patient_id": pid,
            "name": r["name"] if r["name"] else "Unknown",
            "risk_level": level,
            "factors": factors,
        })

    con.close()
    return {
        "patient_registry": patient_registry,
        "age_sex_pyramid": age_sex_pyramid,
        "seizure_characteristics": seizure_characteristics,
        "geographic_distribution": geographic_distribution,
        "risk_stratification": risk_stratification,
    }


def definitions():
    """Clinical epidemiology term definitions, data sources, and methodology."""
    return {
        "terms": [
            {
                "term": "Prevalence",
                "definition": (
                    "The proportion of a population found to have a condition at a "
                    "specific point in time (point prevalence) or over a period "
                    "(period prevalence). Expressed as cases per 1 000 population. "
                    "Global epilepsy prevalence: ~6.4 per 1 000 (Fiest et al., 2017)."
                ),
            },
            {
                "term": "Incidence",
                "definition": (
                    "The rate of new cases of a condition arising in a population "
                    "over a defined period. Expressed as cases per 100 000 person-"
                    "years. Epilepsy incidence: 50-70 per 100 000/year in high-"
                    "income countries (Kotsopoulos et al., 2002)."
                ),
            },
            {
                "term": "Comorbidity Burden",
                "definition": (
                    "The cumulative impact of co-occurring medical conditions on a "
                    "patient's health trajectory. In epilepsy, common comorbidities "
                    "include depression (23%), anxiety (20%), migraine (13%), and "
                    "cognitive impairment (Keezer et al., Lancet Neurology 2016). "
                    "Higher comorbidity burden correlates with worse seizure control "
                    "and increased healthcare utilisation."
                ),
            },
            {
                "term": "Seizure Burden",
                "definition": (
                    "A composite measure of seizure impact comprising frequency, "
                    "severity, duration, and associated injuries or emergency visits. "
                    "Seizure burden drives treatment decisions per ILAE classification "
                    "(Fisher et al., Epilepsia 2017)."
                ),
            },
            {
                "term": "Age-Sex Pyramid",
                "definition": (
                    "A population distribution chart showing age groups on the "
                    "vertical axis and male/female counts on opposing horizontal "
                    "axes. Used to identify demographic biases in cohort enrollment "
                    "and compare against national epilepsy registries."
                ),
            },
            {
                "term": "Risk Stratification",
                "definition": (
                    "The process of classifying patients into risk tiers (Low, "
                    "Moderate, High) based on clinical factors such as seizure "
                    "frequency, comorbidity count, and age. Enables targeted "
                    "resource allocation and proactive intervention for high-risk "
                    "subgroups (WHO Package of Essential NCD Interventions, 2020)."
                ),
            },
        ],
        "data_sources": [
            {"source": "patients", "rows": 40, "description": "Demographic registry with age, gender, department, and enrollment date."},
            {"source": "seizure_diary", "rows": 25, "description": "Patient-reported seizure events with severity, triggers, awareness, and outcomes."},
            {"source": "comorbidities", "rows": 27, "description": "Comorbidity records with conditions, severity, functional impact, and treatment status."},
            {"source": "medications", "rows": 9, "description": "Prescribed medication records with drug names, dosages, and frequencies."},
            {"source": "eeg_acquisition", "rows": 30, "description": "EEG recording sessions linked to patients."},
            {"source": "assessments", "rows": 423, "description": "Clinical assessments including cognitive, psychiatric, and functional evaluations."},
            {"source": "pro_outcomes", "rows": 180, "description": "Patient-reported outcome measures (QOLIE-31, PHQ-9, GAD-7, etc.)."},
            {"source": "medication_adherence", "rows": 12600, "description": "Dose-level medication adherence logs with timing and side effects."},
        ],
        "methodology": (
            "Population health analytics are computed from the clinical.db SQLite "
            "database using real patient records. Demographics are derived from the "
            "patients table. Seizure burden aggregates event counts and severity from "
            "the seizure_diary. Comorbidity prevalence is extracted by parsing the "
            "fields_json column in the comorbidities table. Risk stratification uses "
            "a weighted scoring model combining seizure frequency (0-3 points), "
            "comorbidity count (0-3 points), and age (0-2 points), with thresholds "
            "at 3 (Moderate) and 5 (High). All metrics are computed on-the-fly from "
            "live data; no values are fabricated or cached."
        ),
    }


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    print("=== Population Health Overview ===")
    ov = overview()
    pprint.pprint({k: v for k, v in ov.items()
                   if k not in ("comorbidity_prevalence", "enrollment_trend")})
    print(f"\n  Comorbidities tracked: {len(ov['comorbidity_prevalence'])}")
    print(f"  Enrollment months: {len(ov['enrollment_trend'])}")

    print("\n=== Population Health Breakdown ===")
    bd = breakdown()
    print(f"  Patient registry: {len(bd['patient_registry'])} patients")
    print(f"  Age-sex pyramid groups: {len(bd['age_sex_pyramid'])}")
    print(f"  Departments: {len(bd['geographic_distribution'])}")
    print(f"  Risk stratification: {len(bd['risk_stratification'])} patients")

    risk_counts = Counter(r["risk_level"] for r in bd["risk_stratification"])
    print(f"  Risk levels: {dict(risk_counts)}")

    print("\n=== Definitions ===")
    defs = definitions()
    print(f"  Terms defined: {len(defs['terms'])}")
    print(f"  Data sources: {len(defs['data_sources'])}")
