"""Comorbidity Analysis Dashboard — Psychiatric comorbidity profiling for epilepsy patients.

All data derived from REAL clinical data in data/clinical.db (comorbidities, patients
tables).  No synthetic or fabricated data is used.

Psychiatric comorbidities are among the most clinically significant yet under-recognised
aspects of epilepsy care.  Between 30% and 50% of people with epilepsy (PWE) have at
least one co-occurring psychiatric condition, a prevalence two to five times that of
the general population.  These comorbidities are not merely incidental: they share
bidirectional pathophysiological links with epilepsy through overlapping neurotransmitter
dysregulation (serotonin, GABA, glutamate), limbic network disruption, HPA-axis
hyperactivity, neuroinflammation, and iatrogenic effects of antiseizure medications.

The most prevalent psychiatric comorbidities in epilepsy include:

  1. Depression — Affects 20-55% of PWE; peri-ictal dysphoria, inter-ictal depressive
     disorder, and major depressive disorder all occur.  The NDDI-E (Neurological
     Disorders Depression Inventory for Epilepsy) and PHQ-9 (Patient Health
     Questionnaire-9) are standard screening instruments.  SSRIs are first-line
     pharmacotherapy; bupropion is avoided due to dose-dependent seizure risk.

  2. Anxiety — 10-25% of PWE meet criteria for a diagnosable anxiety disorder.  The
     GAD-7 (Generalised Anxiety Disorder-7) captures worry severity.  Anticipatory
     anxiety about seizures compounds the burden, particularly in temporal lobe epilepsy.

  3. PTSD — Post-traumatic stress disorder prevalence is elevated in epilepsy
     (6-17%), especially after traumatic brain injury-related epilepsy or after
     experiencing ictal fear.  The PCL-5 (PTSD Checklist for DSM-5) is the standard
     screening tool.

  4. Psychosis — Postictal psychosis (PIP), inter-ictal psychosis (IIP), and forced
     normalisation (alternative psychosis) occur in 2-7% of PWE.  Temporal lobe
     epilepsy with bilateral hippocampal pathology carries the highest risk.

  5. ADHD — Attention-deficit/hyperactivity disorder co-occurs in 12-40% of children
     with epilepsy and persists into adulthood.  Methylphenidate is considered safe
     in controlled epilepsy; stimulant effects on seizure threshold are minimal at
     therapeutic doses.

  6. Sleep disorders — Insomnia, obstructive sleep apnoea, and restless legs syndrome
     are prevalent; fragmented sleep architecture worsens seizure control, creating
     a vicious cycle.

Clinical impact of unrecognised comorbidities:
  - Quality of life is more strongly predicted by depression severity than by seizure
    frequency in most cohorts (Gilliam et al., Neurology 2004).
  - Psychiatric comorbidities are independent risk factors for SUDEP (sudden unexpected
    death in epilepsy), treatment non-adherence, and surgical failure.
  - Systematic screening at every clinic visit is recommended by ILAE (International
    League Against Epilepsy) and AAN (American Academy of Neurology) practice guidelines.

Screening instruments implemented in this module:
  PHQ-9   — 9-item depression screen; scores 0-27; >=10 indicates moderate depression.
  GAD-7   — 7-item anxiety screen; scores 0-21; >=10 indicates moderate anxiety.
  C-SSRS  — Columbia Suicide Severity Rating Scale; categorical assessment of suicidal
             ideation and behaviour for safety screening.
  NDDI-E  — 6-item epilepsy-specific depression screen; score >15 suggests major
             depressive episode with 90% sensitivity.
  PCL-5   — 20-item PTSD screen; score >=33 is the provisional diagnostic threshold.
  MDQ     — Mood Disorder Questionnaire; screens for bipolar spectrum disorders.

Risk stratification model:
  The behavioral_risk_score (0-100) is a composite metric derived from screening
  instrument scores, comorbidity count, functional impact, and treatment status.
  - 0-24:  minimal risk (routine annual screening adequate)
  - 25-49: mild risk (enhanced monitoring, consider referral)
  - 50-74: moderate risk (psychiatric referral recommended)
  - 75-100: severe risk (urgent psychiatric evaluation, safety planning)

References:
  Kanner AM. Psychiatric comorbidities of epilepsy: towards identification of
    the key variables. Lancet Neurol 2016;15(7):787-797.
  Mula M, Sander JW. Psychosocial aspects of epilepsy: a wider approach.
    BJPsych Open 2016;2(4):270-274.
  Gilliam FG et al. Depression in epilepsy: ignoring clinical expression of neuronal
    network dysfunction? Epilepsia 2004;45(s2):28-33.
  Tellez-Zenteno JF et al. Psychiatric comorbidity in epilepsy: a population-based
    analysis. Epilepsia 2007;48(12):2336-2344.
  ILAE Psychiatric Comorbidities Task Force. Screening for psychiatric disorders
    in epilepsy. Epileptic Disord 2019;21(5):393-403.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _safe_div(a, b):
    return round(a / b, 4) if b else None


def _parse_fields(row):
    """Parse fields_json from a comorbidities row, returning a dict with safe defaults."""
    raw = row.get("fields_json", "{}")
    if not raw:
        raw = "{}"
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = {}
    return {
        "comorbidities": data.get("comorbidities", []) or [],
        "comorbidity_count": data.get("comorbidity_count", 0) or 0,
        "screening_instruments": data.get("screening_instruments", []) or [],
        "screening_date": data.get("screening_date"),
        "screened": bool(data.get("screened", False)),
        "behavioral_risk_score": data.get("behavioral_risk_score", 0.0) or 0.0,
        "risk_severity": data.get("risk_severity", "unknown"),
        "functional_impact": data.get("functional_impact", "unknown"),
        "treatment_status": data.get("treatment_status", "unknown"),
        "notes": data.get("notes", ""),
    }


def _age_group(age):
    """Bin age into clinical age groups."""
    if age is None:
        return "unknown"
    try:
        age = int(age)
    except (ValueError, TypeError):
        return "unknown"
    if age < 18:
        return "<18"
    elif age < 30:
        return "18-29"
    elif age < 45:
        return "30-44"
    elif age < 60:
        return "45-59"
    else:
        return "60+"


# ─────────────────────────────────────────────────────────────────────
# 1. overview()
# ─────────────────────────────────────────────────────────────────────

def overview():
    """KPIs and aggregate statistics for comorbidity screening programme.

    Returns a dict with:
      - total_patients_screened (int)
      - total_comorbidities_found (int)
      - avg_comorbidity_count (float)
      - risk_severity_distribution (dict)
      - top_comorbidities (list of {condition, count})
      - treatment_status_distribution (dict)
      - functional_impact_distribution (dict)
      - avg_behavioral_risk_score (float)
      - screening_instrument_usage (list of {instrument, count})
    """
    rows = _rows("SELECT * FROM comorbidities")
    if not rows:
        return {
            "available": False,
            "reason": "No comorbidity data found",
            "kpis": {"patients_screened": 0, "total_comorbidities": 0,
                     "avg_comorbidity_count": 0.0, "avg_risk_score": 0.0},
            "risk_severity_distribution": [],
            "top_comorbidities": [],
            "treatment_status_distribution": [],
        }

    parsed = [_parse_fields(r) for r in rows]

    # Total patients screened (unique patient_ids)
    patient_ids = set()
    for r in rows:
        pid = r.get("patient_id")
        if pid:
            patient_ids.add(pid)
    total_screened = len(patient_ids)

    # Comorbidity counts
    all_conditions = []
    total_comorbidity_count = 0
    for p in parsed:
        conditions = p["comorbidities"]
        if isinstance(conditions, list):
            all_conditions.extend(conditions)
        total_comorbidity_count += p["comorbidity_count"]

    total_comorbidities_found = len(all_conditions)
    avg_comorbidity_count = _safe_div(
        sum(p["comorbidity_count"] for p in parsed), len(parsed)
    )

    # Risk severity distribution
    severity_counter = Counter(p["risk_severity"] for p in parsed)
    risk_severity_dist = dict(severity_counter.most_common())

    # Top comorbidities by frequency
    condition_counter = Counter(all_conditions)
    top_comorbidities = [
        {"condition": cond, "count": cnt}
        for cond, cnt in condition_counter.most_common(20)
    ]

    # Treatment status distribution
    treatment_counter = Counter(p["treatment_status"] for p in parsed)
    treatment_status_dist = dict(treatment_counter.most_common())

    # Functional impact distribution
    impact_counter = Counter(p["functional_impact"] for p in parsed)
    functional_impact_dist = dict(impact_counter.most_common())

    # Average behavioral risk score
    risk_scores = [p["behavioral_risk_score"] for p in parsed if p["behavioral_risk_score"] is not None]
    avg_risk_score = _safe_div(sum(risk_scores), len(risk_scores)) if risk_scores else 0.0

    # Screening instrument usage frequency
    all_instruments = []
    for p in parsed:
        instruments = p["screening_instruments"]
        if isinstance(instruments, list):
            all_instruments.extend(instruments)
    instrument_counter = Counter(all_instruments)
    screening_instrument_usage = [
        {"instrument": inst, "count": cnt}
        for inst, cnt in instrument_counter.most_common()
    ]

    # Format for frontend consumption (recharts-compatible arrays)
    risk_severity_arr = [
        {"severity": sev, "value": cnt}
        for sev, cnt in severity_counter.most_common()
    ]
    treatment_status_arr = [
        {"status": st, "value": cnt}
        for st, cnt in treatment_counter.most_common()
    ]
    top_comorbidities_arr = [
        {"name": cond, "value": cnt}
        for cond, cnt in condition_counter.most_common(15)
    ]

    return {
        "available": True,
        "kpis": {
            "patients_screened": total_screened,
            "total_comorbidities": total_comorbidities_found,
            "avg_comorbidity_count": avg_comorbidity_count,
            "avg_risk_score": round(avg_risk_score, 1) if avg_risk_score else 0.0,
        },
        "risk_severity_distribution": risk_severity_arr,
        "top_comorbidities": top_comorbidities_arr,
        "treatment_status_distribution": treatment_status_arr,
        "functional_impact_distribution": functional_impact_dist,
        "avg_behavioral_risk_score": avg_risk_score,
        "screening_instrument_usage": screening_instrument_usage,
    }


# ─────────────────────────────────────────────────────────────────────
# 2. breakdown()
# ─────────────────────────────────────────────────────────────────────

def breakdown():
    """Detailed per-patient and per-condition comorbidity analytics.

    Returns a dict with:
      - patient_profiles (list of per-patient dicts)
      - co_occurrence_matrix (dict of {condition: {co_condition: count}})
      - risk_score_histogram (list of {bin_label, count})
      - severity_by_comorbidity_count (list of {comorbidity_count, avg_risk_score, n})
      - demographics_crosstab (dict with age_group and gender breakdowns)
    """
    como_rows = _rows("SELECT * FROM comorbidities")
    if not como_rows:
        return {
            "available": False,
            "reason": "No comorbidity data found",
            "per_patient": [],
            "co_occurrence_matrix": {},
            "risk_score_histogram": [],
            "risk_by_comorbidity_count": [],
            "functional_impact": [],
            "screening_instruments": [],
            "demographics_age": [],
            "demographics_gender": [],
        }

    # Build patient lookup from patients table
    patient_lookup = {}
    try:
        pat_rows = _rows("SELECT * FROM patients")
        for pr in pat_rows:
            pid = pr.get("id") or pr.get("patient_id")
            if pid:
                patient_lookup[str(pid)] = {
                    "age": pr.get("age"),
                    "gender": pr.get("gender"),
                    "diagnosis": pr.get("diagnosis"),
                }
    except Exception:
        pass  # patients table may not exist or have different schema

    parsed_rows = []
    for r in como_rows:
        p = _parse_fields(r)
        p["patient_id"] = r.get("patient_id", "")
        p["created_at"] = r.get("created_at", "")
        parsed_rows.append(p)

    # ── Per-patient profiles ──
    patient_profiles = []
    for p in parsed_rows:
        demo = patient_lookup.get(str(p["patient_id"]), {})
        patient_profiles.append({
            "patient_id": p["patient_id"],
            "comorbidity_count": p["comorbidity_count"],
            "risk_severity": p["risk_severity"],
            "behavioral_risk_score": p["behavioral_risk_score"],
            "conditions": p["comorbidities"],
            "treatment_status": p["treatment_status"],
            "functional_impact": p["functional_impact"],
            "screening_date": p["screening_date"],
            "age": demo.get("age"),
            "gender": demo.get("gender"),
            "diagnosis": demo.get("diagnosis"),
        })

    # ── Co-occurrence matrix ──
    # For each patient, count all pairs of conditions that appear together
    co_occurrence = defaultdict(lambda: defaultdict(int))
    for p in parsed_rows:
        conditions = p["comorbidities"]
        if not isinstance(conditions, list) or len(conditions) < 2:
            continue
        # Deduplicate within a single patient
        unique_conds = sorted(set(conditions))
        for i, c1 in enumerate(unique_conds):
            for c2 in unique_conds[i + 1:]:
                co_occurrence[c1][c2] += 1
                co_occurrence[c2][c1] += 1

    # Convert defaultdict to regular dict for serialisation
    co_occurrence_matrix = {k: dict(v) for k, v in co_occurrence.items()}

    # ── Risk score histogram ──
    bins = [
        (0, 10, "0-9"),
        (10, 20, "10-19"),
        (20, 30, "20-29"),
        (30, 40, "30-39"),
        (40, 50, "40-49"),
        (50, 60, "50-59"),
        (60, 70, "60-69"),
        (70, 80, "70-79"),
        (80, 90, "80-89"),
        (90, 101, "90-100"),
    ]
    bin_counts = {label: 0 for _, _, label in bins}
    for p in parsed_rows:
        score = p["behavioral_risk_score"]
        if score is None:
            continue
        for lo, hi, label in bins:
            if lo <= score < hi:
                bin_counts[label] += 1
                break
    risk_score_histogram = [
        {"bin_label": label, "count": bin_counts[label]}
        for _, _, label in bins
    ]

    # ── Severity by comorbidity count ──
    count_to_scores = defaultdict(list)
    for p in parsed_rows:
        cc = p["comorbidity_count"]
        score = p["behavioral_risk_score"]
        if score is not None:
            count_to_scores[cc].append(score)

    severity_by_count = sorted([
        {
            "comorbidity_count": cc,
            "avg_risk_score": _safe_div(sum(scores), len(scores)),
            "n": len(scores),
        }
        for cc, scores in count_to_scores.items()
    ], key=lambda x: x["comorbidity_count"])

    # ── Demographics cross-tab ──
    by_age_group = defaultdict(lambda: {"total": 0, "with_comorbidities": 0, "avg_risk_score": 0.0, "scores": []})
    by_gender = defaultdict(lambda: {"total": 0, "with_comorbidities": 0, "avg_risk_score": 0.0, "scores": []})

    for p in parsed_rows:
        demo = patient_lookup.get(str(p["patient_id"]), {})
        age = demo.get("age")
        gender = demo.get("gender", "unknown") or "unknown"
        ag = _age_group(age)

        by_age_group[ag]["total"] += 1
        if p["comorbidity_count"] > 0:
            by_age_group[ag]["with_comorbidities"] += 1
        if p["behavioral_risk_score"] is not None:
            by_age_group[ag]["scores"].append(p["behavioral_risk_score"])

        by_gender[gender]["total"] += 1
        if p["comorbidity_count"] > 0:
            by_gender[gender]["with_comorbidities"] += 1
        if p["behavioral_risk_score"] is not None:
            by_gender[gender]["scores"].append(p["behavioral_risk_score"])

    # Compute averages and drop raw scores
    def _finalize_group(group_dict):
        result = {}
        for key, data in group_dict.items():
            scores = data.pop("scores", [])
            data["avg_risk_score"] = _safe_div(sum(scores), len(scores)) if scores else 0.0
            data["comorbidity_rate"] = _safe_div(data["with_comorbidities"], data["total"])
            result[key] = data
        return result

    demographics_crosstab = {
        "by_age_group": _finalize_group(dict(by_age_group)),
        "by_gender": _finalize_group(dict(by_gender)),
    }

    # Format demographics as recharts-compatible arrays
    demographics_age = [
        {"age_group": ag, "comorbidity_rate": round((d["comorbidity_rate"] or 0) * 100, 1),
         "avg_risk_score": round(d["avg_risk_score"] or 0, 1), "n": d["total"]}
        for ag, d in sorted(demographics_crosstab["by_age_group"].items())
    ]
    demographics_gender = [
        {"gender": g, "comorbidity_rate": round((d["comorbidity_rate"] or 0) * 100, 1),
         "avg_risk_score": round(d["avg_risk_score"] or 0, 1), "n": d["total"]}
        for g, d in sorted(demographics_crosstab["by_gender"].items())
    ]

    # Build functional impact array from overview data
    como_parsed = [_parse_fields(r) for r in como_rows]
    impact_counter = Counter(p["functional_impact"] for p in como_parsed)
    functional_impact_arr = [
        {"impact": imp, "value": cnt, "fullMark": len(como_parsed)}
        for imp, cnt in impact_counter.most_common()
    ]

    # Build screening instruments array
    all_instruments = []
    for p in como_parsed:
        instruments = p["screening_instruments"]
        if isinstance(instruments, list):
            all_instruments.extend(instruments)
    instrument_counter = Counter(all_instruments)
    screening_instruments_arr = [
        {"name": inst, "value": cnt}
        for inst, cnt in instrument_counter.most_common()
    ]

    return {
        "available": True,
        "per_patient": patient_profiles,
        "co_occurrence_matrix": co_occurrence_matrix,
        "risk_score_histogram": risk_score_histogram,
        "risk_by_comorbidity_count": severity_by_count,
        "functional_impact": functional_impact_arr,
        "screening_instruments": screening_instruments_arr,
        "demographics_age": demographics_age,
        "demographics_gender": demographics_gender,
    }


# ─────────────────────────────────────────────────────────────────────
# 3. definitions()
# ─────────────────────────────────────────────────────────────────────

def definitions():
    """Clinical glossary of psychiatric comorbidity terms, screening instruments,
    severity scales, and common conditions in epilepsy.

    Returns a dict with categorised definition lists and a flat 'terms' array.
    """
    categories = {
        "core_concepts": [
            {
                "term": "Comorbidity",
                "definition": (
                    "The co-occurrence of two or more distinct medical conditions in the same "
                    "individual.  In epilepsy, comorbidities include psychiatric, cognitive, "
                    "and somatic conditions that coexist with the seizure disorder and may "
                    "share common pathophysiological mechanisms."
                ),
            },
            {
                "term": "Psychiatric comorbidity in epilepsy",
                "definition": (
                    "A psychiatric disorder that co-occurs with epilepsy at a rate higher than "
                    "expected by chance.  The relationship is bidirectional: epilepsy increases "
                    "the risk of psychiatric illness, and psychiatric illness (especially "
                    "depression) increases the risk of developing epilepsy.  Shared mechanisms "
                    "include serotonergic/GABAergic dysregulation, limbic network disruption, "
                    "HPA-axis hyperactivity, and neuroinflammation."
                ),
            },
        ],
        "screening_instruments": [
            {
                "term": "PHQ-9 (Patient Health Questionnaire-9)",
                "definition": (
                    "A 9-item self-report depression screening tool.  Each item is scored 0-3 "
                    "(not at all / several days / more than half the days / nearly every day).  "
                    "Total score range 0-27.  Cut-offs: 0-4 minimal, 5-9 mild, 10-14 moderate, "
                    "15-19 moderately severe, 20-27 severe depression.  Validated in epilepsy "
                    "populations with good sensitivity (88%) and specificity (78%) at >=10."
                ),
            },
            {
                "term": "GAD-7 (Generalised Anxiety Disorder-7)",
                "definition": (
                    "A 7-item anxiety screening tool scored 0-21.  Cut-offs: 0-4 minimal, "
                    "5-9 mild, 10-14 moderate, 15-21 severe anxiety.  Widely used in epilepsy "
                    "clinics to screen for generalised anxiety, panic disorder, and social "
                    "anxiety.  A score >=10 has 89% sensitivity and 82% specificity for GAD."
                ),
            },
            {
                "term": "C-SSRS (Columbia Suicide Severity Rating Scale)",
                "definition": (
                    "A structured interview that assesses suicidal ideation (type, intensity, "
                    "duration, controllability, deterrents) and behaviour (actual attempts, "
                    "aborted attempts, interrupted attempts, preparatory acts).  Required by "
                    "the FDA for all CNS clinical trials.  Epilepsy patients have a 3-5x "
                    "elevated suicide risk compared to the general population."
                ),
            },
            {
                "term": "NDDI-E (Neurological Disorders Depression Inventory for Epilepsy)",
                "definition": (
                    "A 6-item epilepsy-specific depression screening tool designed to minimise "
                    "confounding by antiseizure medication side effects (fatigue, concentration "
                    "difficulty).  Score range 6-24.  A cut-off >15 identifies major depressive "
                    "episodes with 90% sensitivity and 81% specificity.  Recommended as first-line "
                    "screening by ILAE Psychiatric Comorbidities Task Force."
                ),
            },
            {
                "term": "PCL-5 (PTSD Checklist for DSM-5)",
                "definition": (
                    "A 20-item self-report measure of PTSD symptoms aligned with DSM-5 criteria.  "
                    "Each item scored 0-4; total range 0-80.  A provisional diagnostic cut-off "
                    "of >=33 is recommended.  Relevant in epilepsy for patients with TBI-related "
                    "epilepsy, ictal fear auras, and psychogenic non-epileptic seizure (PNES) "
                    "differential diagnosis."
                ),
            },
            {
                "term": "MDQ (Mood Disorder Questionnaire)",
                "definition": (
                    "A screening instrument for bipolar spectrum disorders.  Contains 13 yes/no "
                    "items about manic/hypomanic symptoms, a clustering question, and a functional "
                    "impairment question.  A positive screen (>=7 yes items + clustering + "
                    "moderate-serious problems) has 73% sensitivity for bipolar I/II.  Important "
                    "in epilepsy because antidepressant monotherapy can destabilise bipolar patients."
                ),
            },
        ],
        "risk_severity_levels": [
            {
                "term": "Minimal (0-24)",
                "definition": (
                    "Low psychiatric risk.  Patient may have subclinical symptoms or no current "
                    "psychiatric comorbidity.  Routine annual screening is adequate.  No immediate "
                    "psychiatric referral required."
                ),
            },
            {
                "term": "Mild (25-49)",
                "definition": (
                    "Mild psychiatric risk.  Patient shows emerging symptoms or a single "
                    "comorbidity with limited functional impact.  Enhanced monitoring at each "
                    "epilepsy clinic visit recommended.  Consider referral if symptoms persist "
                    "beyond 3 months."
                ),
            },
            {
                "term": "Moderate (50-74)",
                "definition": (
                    "Moderate psychiatric risk.  Patient has one or more comorbidities with "
                    "measurable functional impact on daily activities, medication adherence, or "
                    "seizure control.  Psychiatric referral is recommended.  Collaborative care "
                    "between neurologist and psychiatrist is optimal."
                ),
            },
            {
                "term": "Severe (75-100)",
                "definition": (
                    "High psychiatric risk.  Patient has significant psychiatric burden including "
                    "possible suicidal ideation, treatment-resistant comorbidities, or severe "
                    "functional impairment.  Urgent psychiatric evaluation and safety planning "
                    "required.  Consider inpatient evaluation if C-SSRS is positive."
                ),
            },
        ],
        "functional_impact_scale": [
            {
                "term": "None",
                "definition": (
                    "No measurable impact on daily functioning.  Patient manages work, social, "
                    "and self-care activities without difficulty related to comorbidities."
                ),
            },
            {
                "term": "Mild",
                "definition": (
                    "Minor difficulties in one domain (e.g., occasional sleep disruption, "
                    "mild concentration difficulty).  Activities of daily living are preserved."
                ),
            },
            {
                "term": "Moderate",
                "definition": (
                    "Noticeable impact on two or more domains (work productivity, social "
                    "engagement, medication adherence).  Patient requires some support or "
                    "accommodations."
                ),
            },
            {
                "term": "Severe",
                "definition": (
                    "Substantial impairment across multiple domains.  Patient may be unable "
                    "to work, maintain relationships, or manage self-care independently.  "
                    "Comorbidities likely contributing to poor seizure control."
                ),
            },
        ],
        "treatment_status_categories": [
            {
                "term": "None",
                "definition": (
                    "No psychiatric treatment initiated.  Patient may be newly screened, "
                    "unaware of comorbidities, or has declined treatment."
                ),
            },
            {
                "term": "Untreated",
                "definition": (
                    "Comorbidities identified but not yet treated.  May be awaiting referral, "
                    "in a monitoring phase, or experiencing barriers to care (access, stigma, "
                    "cost)."
                ),
            },
            {
                "term": "Stable",
                "definition": (
                    "Patient is receiving treatment (pharmacotherapy, psychotherapy, or both) "
                    "and symptoms are adequately controlled.  Continued maintenance and "
                    "periodic reassessment recommended."
                ),
            },
            {
                "term": "Treatment-resistant",
                "definition": (
                    "Comorbidity persists despite adequate trials of two or more treatments.  "
                    "Requires specialist psychiatric input, combination therapy, or "
                    "consideration of neuromodulation (VNS, rTMS) for treatment-resistant "
                    "depression in epilepsy."
                ),
            },
        ],
        "common_comorbidities_in_epilepsy": [
            {
                "term": "Depression",
                "definition": (
                    "The most common psychiatric comorbidity in epilepsy (20-55% prevalence).  "
                    "Includes major depressive disorder, dysthymia, and epilepsy-specific "
                    "inter-ictal dysphoric disorder (IDD) characterised by irritability, "
                    "depressive moods, and anergia occurring in clusters."
                ),
            },
            {
                "term": "Anxiety",
                "definition": (
                    "Second most common psychiatric comorbidity (10-25%).  Includes generalised "
                    "anxiety disorder, panic disorder, social anxiety, and epilepsy-specific "
                    "anticipatory anxiety (fear of the next seizure).  Temporal lobe epilepsy "
                    "confers the highest risk due to amygdala involvement."
                ),
            },
            {
                "term": "PTSD (Post-Traumatic Stress Disorder)",
                "definition": (
                    "Elevated prevalence (6-17%) in epilepsy, particularly in TBI-related "
                    "epilepsy and patients with ictal fear.  Overlaps with psychogenic "
                    "non-epileptic seizures (PNES) in differential diagnosis."
                ),
            },
            {
                "term": "Psychosis",
                "definition": (
                    "Occurs in 2-7% of PWE.  Three types: postictal psychosis (PIP, onset "
                    "24-72h after seizure cluster, usually self-limiting), inter-ictal psychosis "
                    "(IIP, chronic schizophrenia-like illness), and forced normalisation "
                    "(psychosis emerging with seizure freedom after EEG normalisation)."
                ),
            },
            {
                "term": "ADHD (Attention-Deficit/Hyperactivity Disorder)",
                "definition": (
                    "Co-occurs in 12-40% of children with epilepsy and persists into adulthood.  "
                    "Inattention is more common than hyperactivity in epilepsy.  Methylphenidate "
                    "is considered safe at therapeutic doses; atomoxetine is an alternative."
                ),
            },
            {
                "term": "Sleep disorders",
                "definition": (
                    "Insomnia, obstructive sleep apnoea, and restless legs syndrome are highly "
                    "prevalent in epilepsy.  Sleep deprivation is a well-established seizure "
                    "trigger, creating a bidirectional cycle.  Treatment of sleep disorders "
                    "can improve seizure control."
                ),
            },
        ],
    }
    # Flatten all categories into a single 'terms' list for the frontend
    terms = []
    for cat_list in categories.values():
        terms.extend(cat_list)
    categories["terms"] = terms
    return categories
