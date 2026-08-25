"""
MODY2 — GCK-MODY (Maturity-Onset Diabetes of the Young Type 2)
===============================================================
Gene       : GCK (Glucokinase / Hexokinase IV)
Chromosome : 7p13
OMIM Gene  : *138079
OMIM Dis.  : #125851  (MODY2)
Inheritance: Autosomal Dominant (50% transmission per child)
Prevalence : ~1:1,000–1:4,000 (most underdiagnosed MODY; often labelled prediabetes or GDM)
MODY frac. : ~25–35% of all MODY (2nd most common in some series)

Mechanism
---------
GCK (Glucokinase, hexokinase IV) is the pancreatic beta-cell GLUCOSE SENSOR — it phosphorylates
glucose to glucose-6-phosphate and sets the threshold at which insulin secretion is triggered.
LOF variants raise this threshold (reset the "glucostat") by ~1–2 mmol/L: the beta-cell now
perceives a higher glucose as "normal" and secretes insulin at the new set-point.

CRITICAL distinction from all other MODY forms:
  * This is a GLUCOSE-SENSING DEFECT — not a secretory failure or transcription-factor LOF.
  * HbA1c is mildly elevated but STABLE and non-progressive over decades.
  * The body fully compensates at the new set-point; there is NO progressive beta-cell failure.
  * Complications (retinopathy, nephropathy, neuropathy) are rare/absent at these HbA1c levels.

Key Clinical Hallmarks
-----------------------
* Mild, stable fasting hyperglycaemia (5.4–8.3 mmol/L / 97–149 mg/dL) — detected incidentally
* Limited OGTT postprandial increment (<3.5 mmol/L from fasting to 2 h) — glucose "clamping"
* HbA1c: typically 5.6–7.6% — stable over years/decades (distinguishes from T2D/MODY3 progression)
* NO diabetic complications at typical MODY2 HbA1c levels
* Autoantibodies uniformly NEGATIVE (GADA, ZnT8, IA-2)
* Family history positive ~75–80% (AD, mild phenotype — often unrecognized in parents)
* Detected at any age: school screening, insurance medicals, pregnancy (GDM)
* NO renal glycosuria (GCK does not regulate SGLT2)
* MODY Probability Calculator: moderate pre-test probability when combined with family history

Treatment
---------
NO TREATMENT NEEDED in most cases:
  * Diet and lifestyle monitoring only — sulfonylure and insulin do NOT improve long-term outcome
  * Sulfonylure lowers glucose below the new set-point → hypoglycaemia without benefit
  * Insulin may cause hypoglycaemia and provides no advantage over diet-only management

PREGNANCY EXCEPTION (critical):
  * If fetus is GCK-NEGATIVE (unaffected): maternal hyperglycaemia → fetal macrosomia.
    Insulin therapy required for the MOTHER to normalize maternal glucose → protects fetus.
  * If fetus is GCK-POSITIVE (also carries variant): fetal glucostat also raised → fetus
    does NOT overgrow. Insulin treatment of mother → maternal hypoglycaemia, fetal growth
    restriction. NO treatment to mother if fetus is affected.
  * Fetal genotype determines treatment decision — maternal insulin driven by fetal GCK status.

Cohort: 40 patients, seed=307.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 307
_COHORT_SIZE = 40

# GCK variants — >600 known; most are unique missense; major classes:
_VARIANTS = [
    "V62M",          # common European
    "G46S",          # common mild missense
    "T228M",         # frequent
    "R186X",         # nonsense; common
    "R191W",         # missense
    "L451P",         # missense
    "G264S",         # missense
    "A456V",         # missense
    "Other_missense",
    "Other_splicing",
    "Other_nonsense",
]

_TREATMENTS = ["Diet only", "Monitoring only", "Insulin (pregnancy)", "Sulfonylurea (prior Rx)"]
_TREATMENT_WEIGHTS = [0.52, 0.22, 0.14, 0.12]

# In MODY2, sulfo "response" is: no benefit / hypoglycaemia / not started
_SULFO_RESPONSES = ["No_benefit", "Hypoglycaemia", "Not_started", "Partial_transient"]
_SULFO_RESPONSE_WEIGHTS = [0.12, 0.10, 0.68, 0.10]

_MISDIAGNOSES = ["Prediabetes", "GDM", "T2D", "None"]
_MISDIAGNOSIS_WEIGHTS = [0.38, 0.26, 0.16, 0.20]

_DETECT_MODES = ["Incidental_fasting", "Pregnancy_GDM_screen", "School_screening", "Family_cascade", "Insurance_medical"]
_DETECT_WEIGHTS = [0.30, 0.26, 0.12, 0.20, 0.12]

_COMPLICATIONS_POOL = ["none", "mild_retinopathy", "hypertension", "overweight"]


def _hba1c_percent_to_mmol(pct: float) -> float:
    """Convert HbA1c % (NGSP/DCCT) to mmol/mol (IFCC)."""
    return round((pct - 2.15) / 0.0915, 1)


def _weighted_choice(rng: random.Random, choices, weights):
    r = rng.random()
    cumulative = 0.0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if r < cumulative:
            return choice
    return choices[-1]


def _build_cohort() -> list:
    """Generate the 40-patient cohort deterministically (seed=307)."""
    rng = random.Random(_SEED)
    patients = []

    for i in range(1, _COHORT_SIZE + 1):
        age = rng.randint(10, 60)
        sex = rng.choice(["M", "F"])

        # Diagnosis age: any age, often earlier (school, pregnancy, incidental)
        age_at_dx = int(rng.triangular(8, min(age, 58), 28))
        age_at_dx = max(8, min(age_at_dx, age))
        duration = age - age_at_dx

        # GCK-MODY: HbA1c is MILDLY elevated and STABLE (5.6–7.6%)
        hba1c_pct = round(rng.uniform(5.6, 7.6), 1)
        hba1c_mmol = _hba1c_percent_to_mmol(hba1c_pct)

        # Fasting glucose: 5.4–8.3 mmol/L — stable set-point
        fasting_glucose_mmol = round(rng.uniform(5.4, 8.3), 1)
        # OGTT 2h increment: limited (<3.5 mmol/L) — key MODY2 pattern
        ogtt_increment_mmol = round(rng.uniform(1.2, 3.4), 1)
        ogtt_2h_mmol = round(fasting_glucose_mmol + ogtt_increment_mmol, 1)

        # C-peptide: NORMAL (not depleted — not a secretory defect)
        c_peptide = round(rng.uniform(0.6, 2.2), 2)

        family_hx = rng.random() < 0.77
        renal_glycosuria = False       # GCK does not regulate SGLT2
        neonatal_macrosomia = False    # GCK-MODY neonates: no macrosomia from mother if fetus GCK+

        variant = rng.choice(_VARIANTS)
        treatment = _weighted_choice(rng, _TREATMENTS, _TREATMENT_WEIGHTS)
        sulfo_response = _weighted_choice(rng, _SULFO_RESPONSES, _SULFO_RESPONSE_WEIGHTS)
        misdiagnosis = _weighted_choice(rng, _MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)
        detection_mode = _weighted_choice(rng, _DETECT_MODES, _DETECT_WEIGHTS)

        # Complications: rare at MODY2 HbA1c levels
        comp_pool = [c for c in _COMPLICATIONS_POOL if c != "none"]
        n_comp = rng.choices([0, 1], weights=[0.80, 0.20])[0]
        chosen_comps = rng.sample(comp_pool, min(n_comp, len(comp_pool)))
        complications = sorted(chosen_comps) if chosen_comps else ["none"]

        # Pregnancy detail for females of reproductive age
        pregnancy_insulin = False
        fetal_gck_tested = False
        if sex == "F" and 18 <= age <= 45:
            pregnancy_insulin = rng.random() < 0.40   # ~40% females had insulin in pregnancy
            fetal_gck_tested = rng.random() < 0.25

        patients.append({
            "patient_id": f"MODY2-{i:03d}",
            "age": int(age),
            "sex": sex,
            "age_at_diagnosis": int(age_at_dx),
            "duration_years": int(duration),
            "hba1c_percent": float(hba1c_pct),
            "hba1c_mmol": float(hba1c_mmol),
            "fasting_glucose_mmol": float(fasting_glucose_mmol),
            "ogtt_2h_mmol": float(ogtt_2h_mmol),
            "ogtt_increment_mmol": float(ogtt_increment_mmol),
            "c_peptide_nmol_L": float(c_peptide),
            "family_hx_positive": bool(family_hx),
            "renal_glycosuria": bool(renal_glycosuria),       # Always False
            "neonatal_macrosomia": bool(neonatal_macrosomia), # False (GCK+ fetus)
            "antibody_status": "NEGATIVE",
            "current_treatment": treatment,
            "sulfo_response": sulfo_response,
            "variant": variant,
            "complications": list(complications),
            "misdiagnosis_prior": misdiagnosis,
            "detection_mode": detection_mode,
            "pregnancy_insulin_used": bool(pregnancy_insulin) if sex == "F" and 18 <= age <= 45 else None,
            "fetal_gck_tested": bool(fetal_gck_tested) if sex == "F" and 18 <= age <= 45 else None,
        })

    return patients


# Build cohort once at import time
_COHORT: list = _build_cohort()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    """
    Return high-level overview of the MODY2 cohort including KPIs, patient
    list, key clinical facts, treatment summary, and diagnostic criteria.
    """
    patients = _COHORT

    hba1c_values = [p["hba1c_percent"] for p in patients]
    c_peptide_values = [p["c_peptide_nmol_L"] for p in patients]
    durations = [p["duration_years"] for p in patients]
    ages = [p["age"] for p in patients]
    dx_ages = [p["age_at_diagnosis"] for p in patients]
    fasting_values = [p["fasting_glucose_mmol"] for p in patients]
    ogtt_inc = [p["ogtt_increment_mmol"] for p in patients]

    n_diet_only = sum(1 for p in patients if "Diet" in p["current_treatment"] or "Monitoring" in p["current_treatment"])
    n_family_hx = sum(1 for p in patients if p["family_hx_positive"])
    n_misdiagnosed = sum(1 for p in patients if p["misdiagnosis_prior"] != "None")
    n_any_complication = sum(1 for p in patients if p["complications"] != ["none"])
    n_controlled_lt7 = sum(1 for p in patients if p["hba1c_percent"] < 7.0)
    n_sulfo_prior = sum(1 for p in patients if p["current_treatment"] == "Sulfonylurea (prior Rx)")

    kpis = {
        "cohort_size": int(_COHORT_SIZE),
        "mean_age_years": round(statistics.mean(ages), 1),
        "mean_age_at_diagnosis_years": round(statistics.mean(dx_ages), 1),
        "mean_duration_years": round(statistics.mean(durations), 1),
        "mean_hba1c_percent": round(statistics.mean(hba1c_values), 1),
        "mean_hba1c_mmol": round(statistics.mean([p["hba1c_mmol"] for p in patients]), 1),
        "mean_fasting_glucose_mmol": round(statistics.mean(fasting_values), 1),
        "mean_ogtt_increment_mmol": round(statistics.mean(ogtt_inc), 1),
        "mean_c_peptide_nmol_L": round(statistics.mean(c_peptide_values), 2),
        "pct_diet_only": round(100.0 * n_diet_only / _COHORT_SIZE, 1),
        "pct_family_hx_positive": round(100.0 * n_family_hx / _COHORT_SIZE, 1),
        "pct_prior_misdiagnosis": round(100.0 * n_misdiagnosed / _COHORT_SIZE, 1),
        "pct_hba1c_lt_7": round(100.0 * n_controlled_lt7 / _COHORT_SIZE, 1),
        "pct_any_complication": round(100.0 * n_any_complication / _COHORT_SIZE, 1),
        "pct_renal_glycosuria": 0.0,       # Always 0 for MODY2
        "pct_antibody_negative": 100.0,
        "pct_prior_sulfo_rx": round(100.0 * n_sulfo_prior / _COHORT_SIZE, 1),
        "male_count": int(sum(1 for p in patients if p["sex"] == "M")),
        "female_count": int(sum(1 for p in patients if p["sex"] == "F")),
    }

    key_facts = [
        "MODY2 is caused by autosomal dominant LOF variants in GCK (glucokinase / hexokinase IV) on chromosome 7p13.",
        "GCK is the pancreatic beta-cell GLUCOSE SENSOR: LOF raises the glucose threshold for insulin secretion by ~1–2 mmol/L — a 'glucostat reset'.",
        "MODY2-UNIQUE: mild, stable, non-progressive hyperglycaemia — HbA1c 5.6–7.6% that does NOT worsen over decades.",
        "Limited OGTT postprandial glucose increment (<3.5 mmol/L from fasting to 2 h) — the glucose 'clamping' effect of GCK LOF.",
        "NO diabetic complications at typical MODY2 HbA1c levels — vascular/microvascular damage threshold is not reached.",
        "NO TREATMENT NEEDED in most cases: diet and monitoring only — sulfonylurea and insulin cause hypoglycaemia without glycaemic benefit.",
        "Most common misdiagnoses: prediabetes (incidental finding), gestational diabetes (GDM in pregnancy), type 2 diabetes.",
        "PREGNANCY EXCEPTION: if fetus is GCK-negative (unaffected), maternal glucose drives fetal macrosomia — insulin required for mother.",
        "If fetus is GCK-positive (also carries variant), fetal glucostat is also raised — maternal insulin causes maternal hypo + fetal growth restriction; NO treatment.",
        "Autoantibodies (GADA, ZnT8, IA-2) are uniformly NEGATIVE — mandatory to rule out T1D before MODY testing.",
        "Family history positive in ~75–80%; AD inheritance with ~50% per-child transmission; mild phenotype often unrecognized in parents.",
        "Detected at any age: school fasting glucose screening, pregnancy GDM screen, insurance medical, or incidental blood test.",
        "Over 600 pathogenic GCK variants known; most are unique missense; molecular confirmation by NGS MODY panel.",
        "GCK Homozygous LOF (both alleles) → Permanent Neonatal Diabetes Mellitus (PNDM) — a very different, severe disorder.",
    ]

    treatment_summary = {
        "standard_management": "Diet + lifestyle monitoring ONLY — no pharmacotherapy required in most cases",
        "sulfonylure": "NOT recommended — glucose set-point is raised, not secretion failure; sulfonylure lowers below set-point → hypoglycaemia",
        "insulin": "NOT required except in pregnancy (if fetal GCK status warrants it)",
        "pregnancy_decision": "Depends on fetal GCK genotype: GCK-negative fetus → insulin for mother; GCK-positive fetus → no treatment",
        "fetal_gck_testing": "Amniocentesis or cell-free fetal DNA for GCK mutation; guides pregnancy insulin decision",
        "no_benefit_of_rx": "Population-level data show NO difference in HbA1c between treated and untreated MODY2 patients long-term",
        "monitoring": "Annual fasting glucose and HbA1c to confirm stability; no OGTT surveillance needed if diagnosis confirmed",
        "family_cascade": "Cascade testing of all first-degree relatives recommended — identify who needs surveillance vs reassurance",
        "misdiagnosis_action": "STOP unnecessary sulfonylurea or insulin once MODY2 confirmed; avoid GDM label in pregnancy without NGS confirmation",
    }

    diagnostic_criteria = {
        "molecular": "NGS MODY panel: GCK (with HNF1A, HNF4A, HNF1B, INS) — pathogenic GCK variant confirms diagnosis",
        "fasting_glucose_pattern": "Stable fasting glucose 5.4–8.3 mmol/L (97–149 mg/dL); does not worsen with time",
        "ogtt_pattern": "2-h OGTT increment <3.5 mmol/L from fasting — limited postprandial excursion",
        "hba1c_stability": "HbA1c 5.6–7.6%, stable over many years — most important functional differentiator from T2D/MODY3",
        "antibodies": "GADA, ZnT8, IA-2 all NEGATIVE — mandatory screen before molecular testing",
        "c_peptide": "NORMAL — not depleted (GCK LOF is sensing defect, not secretory failure or autoimmune destruction)",
        "no_complications": "Absence of retinopathy/nephropathy/neuropathy despite years of mild hyperglycaemia supports MODY2",
        "family_history": "Parent/sibling with mild stable hyperglycaemia (~75–80%); often undiagnosed or labelled prediabetes",
        "no_renal_glycosuria": "Urine dipstick glucose NEGATIVE — GCK does not regulate SGLT2; same as MODY1",
        "mody_calculator": "Exeter MODY Probability Calculator >25% in young, antibody-negative, family history positive → molecular test",
        "differential": "GCK homozygous LOF → PNDM (neonatal; both parents carriers; consanguinity); heterozygous → MODY2",
    }

    return {
        "disease": "MODY2 — GCK-MODY (Maturity-Onset Diabetes of the Young Type 2)",
        "gene": "GCK",
        "omim_gene": "*138079",
        "omim_disease": "#125851",
        "chromosome": "7p13",
        "inheritance": "Autosomal Dominant",
        "prevalence": "~1:1,000–1:4,000 (25–35% of all MODY; most underdiagnosed — often labelled prediabetes)",
        "cohort_size": int(_COHORT_SIZE),
        "seed": int(_SEED),
        "kpis": kpis,
        "patients": [dict(p) for p in patients],
        "key_facts": key_facts,
        "treatment_summary": treatment_summary,
        "diagnostic_criteria": diagnostic_criteria,
    }


def get_breakdown() -> dict:
    """
    Return stratified breakdown counts / distributions across key clinical
    and genetic dimensions of the MODY2 cohort.
    """
    patients = _COHORT

    # Variant distribution
    variant_dist: dict = {}
    for p in patients:
        v = p["variant"]
        variant_dist[v] = variant_dist.get(v, 0) + 1

    # Treatment distribution
    treatment_dist: dict = {}
    for p in patients:
        t = p["current_treatment"]
        treatment_dist[t] = treatment_dist.get(t, 0) + 1

    # Sulfo response / consequence distribution
    sulfo_dist: dict = {}
    for p in patients:
        r = p["sulfo_response"]
        sulfo_dist[r] = sulfo_dist.get(r, 0) + 1

    # Misdiagnosis distribution
    misdiag_dist: dict = {}
    for p in patients:
        m = p["misdiagnosis_prior"]
        misdiag_dist[m] = misdiag_dist.get(m, 0) + 1

    # Detection mode distribution
    detect_dist: dict = {}
    for p in patients:
        d = p["detection_mode"]
        detect_dist[d] = detect_dist.get(d, 0) + 1

    # Complication distribution
    comp_dist: dict = {c: 0 for c in _COMPLICATIONS_POOL}
    for p in patients:
        for c in p["complications"]:
            if c in comp_dist:
                comp_dist[c] += 1

    # Age groups
    age_groups: dict = {"<18": 0, "18-29": 0, "30-44": 0, "45+": 0}
    for p in patients:
        a = p["age"]
        if a < 18:
            age_groups["<18"] += 1
        elif a < 30:
            age_groups["18-29"] += 1
        elif a < 45:
            age_groups["30-44"] += 1
        else:
            age_groups["45+"] += 1

    # HbA1c tiers (shifted down for MODY2)
    hba1c_tiers: dict = {
        "Mild <6.5%": 0,
        "Moderate 6.5-7.0%": 0,
        "Elevated 7.0-7.6%": 0,
        "Above target >7.6%": 0,
    }
    for p in patients:
        h = p["hba1c_percent"]
        if h < 6.5:
            hba1c_tiers["Mild <6.5%"] += 1
        elif h < 7.0:
            hba1c_tiers["Moderate 6.5-7.0%"] += 1
        elif h <= 7.6:
            hba1c_tiers["Elevated 7.0-7.6%"] += 1
        else:
            hba1c_tiers["Above target >7.6%"] += 1

    # Fasting glucose tiers
    fasting_tiers: dict = {"5.4-6.0 mmol/L": 0, "6.1-7.0 mmol/L": 0, "7.1-8.3 mmol/L": 0}
    for p in patients:
        fg = p["fasting_glucose_mmol"]
        if fg <= 6.0:
            fasting_tiers["5.4-6.0 mmol/L"] += 1
        elif fg <= 7.0:
            fasting_tiers["6.1-7.0 mmol/L"] += 1
        else:
            fasting_tiers["7.1-8.3 mmol/L"] += 1

    # Duration tiers
    duration_tiers: dict = {"<5 years": 0, "5-10 years": 0, "10-20 years": 0, "20+ years": 0}
    for p in patients:
        d = p["duration_years"]
        if d < 5:
            duration_tiers["<5 years"] += 1
        elif d < 10:
            duration_tiers["5-10 years"] += 1
        elif d < 20:
            duration_tiers["10-20 years"] += 1
        else:
            duration_tiers["20+ years"] += 1

    # Age at diagnosis tiers
    dx_age_tiers: dict = {"<15": 0, "15-24": 0, "25-35": 0, "36+": 0}
    for p in patients:
        d = p["age_at_diagnosis"]
        if d < 15:
            dx_age_tiers["<15"] += 1
        elif d < 25:
            dx_age_tiers["15-24"] += 1
        elif d <= 35:
            dx_age_tiers["25-35"] += 1
        else:
            dx_age_tiers["36+"] += 1

    sex_dist = {
        "Male": int(sum(1 for p in patients if p["sex"] == "M")),
        "Female": int(sum(1 for p in patients if p["sex"] == "F")),
    }

    return {
        "variant_distribution": {k: int(v) for k, v in sorted(
            variant_dist.items(), key=lambda x: -x[1]
        )},
        "treatment_distribution": {k: int(v) for k, v in treatment_dist.items()},
        "sulfo_consequence_distribution": {k: int(v) for k, v in sulfo_dist.items()},
        "misdiagnosis_distribution": {k: int(v) for k, v in sorted(
            misdiag_dist.items(), key=lambda x: -x[1]
        )},
        "detection_mode_distribution": {k: int(v) for k, v in sorted(
            detect_dist.items(), key=lambda x: -x[1]
        )},
        "complication_distribution": {k: int(v) for k, v in comp_dist.items()},
        "age_groups": {k: int(v) for k, v in age_groups.items()},
        "hba1c_tiers": {k: int(v) for k, v in hba1c_tiers.items()},
        "fasting_glucose_tiers": {k: int(v) for k, v in fasting_tiers.items()},
        "sex_distribution": sex_dist,
        "duration_tiers": {k: int(v) for k, v in duration_tiers.items()},
        "dx_age_tiers": {k: int(v) for k, v in dx_age_tiers.items()},
        "family_hx_positive_count": int(sum(1 for p in patients if p["family_hx_positive"])),
        "renal_glycosuria_count": 0,   # Always 0 for MODY2
        "total_patients": int(_COHORT_SIZE),
    }


def get_definitions() -> dict:
    """
    Return clinical and molecular definitions for key MODY2 terms, suitable
    for a 'Definitions' panel in the frontend dashboard.
    """
    terms = [
        {
            "term": "MODY2",
            "definition": (
                "Maturity-Onset Diabetes of the Young Type 2 — caused by autosomal dominant "
                "loss-of-function (LOF) variants in GCK (glucokinase, hexokinase IV) on "
                "chromosome 7p13 (OMIM *138079; disease #125851). The mildest and most benign "
                "MODY form: mild, stable fasting hyperglycaemia (5.4–8.3 mmol/L), HbA1c "
                "5.6–7.6%, no progression, and no diabetic complications. Usually requires "
                "no pharmacological treatment. Accounts for ~25–35% of all MODY cases."
            ),
        },
        {
            "term": "GCK (Glucokinase / Hexokinase IV)",
            "definition": (
                "Glucokinase (GCK) is the glucose-sensing enzyme in pancreatic beta-cells (and "
                "hepatocytes) that phosphorylates glucose to glucose-6-phosphate (G6P). It acts "
                "as the pancreatic 'glucostat' — setting the threshold glucose concentration at "
                "which insulin secretion is triggered (~5 mmol/L in wild-type). LOF variants "
                "reduce GCK activity, raising this threshold by ~1–2 mmol/L; the beta-cell now "
                "treats a higher glucose as 'normal' and secretes insulin at the new set-point. "
                "Crucially this is a SENSING defect — not secretory failure."
            ),
        },
        {
            "term": "Glucostat reset",
            "definition": (
                "The core mechanism of MODY2: GCK LOF shifts the glucose set-point at which "
                "insulin secretion is triggered upward by ~1–2 mmol/L. The system is stable — "
                "the body fully compensates at the new set-point, secreting appropriate insulin "
                "at the elevated fasting glucose level. No progressive beta-cell failure occurs "
                "because the defect is in sensing, not secretion capacity. This explains why "
                "HbA1c remains stable over decades and why pharmacotherapy is ineffective."
            ),
        },
        {
            "term": "Limited OGTT increment",
            "definition": (
                "A hallmark of MODY2: during an oral glucose tolerance test (OGTT), the "
                "2-hour glucose rises by less than 3.5 mmol/L (63 mg/dL) from fasting. In "
                "contrast, T2D patients show much larger excursions. The GCK 'glucostat' "
                "limits postprandial glucose excursion because the enzyme buffers the glucose "
                "signal, clamping the response. Combined with stable fasting glucose and family "
                "history, an OGTT increment <3.5 mmol/L strongly suggests MODY2."
            ),
        },
        {
            "term": "No treatment required",
            "definition": (
                "In non-pregnant MODY2 patients, pharmacological treatment provides NO "
                "long-term benefit and may cause harm. Sulfonylureas lower blood glucose below "
                "the new (raised) set-point, causing hypoglycaemia without improving HbA1c "
                "long-term. Insulin is similarly unnecessary and risks hypoglycaemia. The "
                "elevated glucose is benign at MODY2 levels — microvascular complication "
                "thresholds are not reached. Diet and monitoring are all that is required."
            ),
        },
        {
            "term": "Pregnancy exception (GCK-MODY)",
            "definition": (
                "Pregnancy is the one setting where MODY2 treatment decisions are complex. "
                "If the fetus is GCK-negative (unaffected), maternal hyperglycaemia drives "
                "fetal pancreatic insulin secretion → fetal macrosomia. Maternal insulin "
                "therapy normalizes maternal glucose and prevents macrosomia. However, if the "
                "fetus is also GCK-positive (its own glucostat is raised), the fetus uses "
                "its set-point — maternal insulin lowers maternal glucose below both set-points "
                "→ maternal hypoglycaemia + fetal growth restriction. Treatment is withheld "
                "if fetus is GCK-positive. Fetal genotyping guides the decision."
            ),
        },
        {
            "term": "GCK homozygous LOF → PNDM",
            "definition": (
                "Homozygous or compound heterozygous GCK LOF (both alleles affected) → "
                "Permanent Neonatal Diabetes Mellitus (PNDM) — a severe, insulin-dependent "
                "disorder presenting within the first 6 months of life. This contrasts with "
                "heterozygous MODY2 (one wild-type allele sufficient for partial function). "
                "Parents of a PNDM child due to homozygous GCK LOF are typically both MODY2 "
                "carriers — consanguinity increases risk. Critical distinction: MODY2 does NOT "
                "progress to insulin dependency; GCK-PNDM does."
            ),
        },
        {
            "term": "Stable hyperglycaemia",
            "definition": (
                "The most clinically important feature of MODY2: fasting glucose and HbA1c "
                "remain constant over years and decades. In T2D and MODY3, HbA1c rises "
                "progressively as beta-cell function declines. In MODY2, there is no "
                "progressive beta-cell failure — the sensor is miscalibrated but function "
                "is preserved. Longitudinal stability of HbA1c (documented across multiple "
                "years) is the strongest functional evidence for MODY2 before molecular testing."
            ),
        },
        {
            "term": "Misdiagnosis as prediabetes or GDM",
            "definition": (
                "MODY2 is the most commonly misdiagnosed MODY because mild fasting "
                "hyperglycaemia is ubiquitous. Adults are labelled 'prediabetes' or 'T2D'; "
                "pregnant women receive 'GDM' diagnoses. Consequences: unnecessary lifestyle "
                "pressure, metformin, insulin, annual GDM recurrence labelling. The key clue "
                "is the stable HbA1c, family history of similar mild hyperglycaemia across "
                "multiple generations, and the diagnostic Exeter MODY Probability >25%. "
                "MODY2 pregnancy labelled GDM → unnecessarily treated with insulin → fetal "
                "growth restriction if fetus is also GCK-positive."
            ),
        },
        {
            "term": "Autosomal Dominant (AD) — MODY2",
            "definition": (
                "A single GCK LOF variant causes MODY2; the remaining wild-type allele "
                "provides partial glucokinase function, but is insufficient to maintain "
                "the normal set-point. Each child of a MODY2 carrier has 50% chance of "
                "inheriting the variant. Penetrance is very high — virtually all carriers "
                "have mildly elevated fasting glucose — but because the phenotype is mild, "
                "parents and grandparents are often told they have 'prediabetes' across "
                "generations without a unifying MODY2 diagnosis."
            ),
        },
        {
            "term": "NGS MODY Panel — GCK",
            "definition": (
                "Next-generation sequencing of the GCK gene (and HNF1A, HNF4A, HNF1B, INS) "
                "identifies the pathogenic variant. Over 600 GCK variants are known; most are "
                "unique missense, nonsense, or frameshift in specific families. A pathogenic GCK "
                "variant + the appropriate clinical phenotype (stable mild hyperglycaemia, family "
                "history, antibody-negative, C-peptide normal) confirms MODY2 and stops "
                "unnecessary treatment. Variants of uncertain significance (VUS) should prompt "
                "functional testing (GCK enzymatic activity assay or structural modelling)."
            ),
        },
        {
            "term": "C-peptide in MODY2",
            "definition": (
                "C-peptide (endogenous insulin secretion marker) is NORMAL in MODY2 — this is "
                "a sensing defect, not a secretory failure or autoimmune beta-cell destruction. "
                "Normal C-peptide distinguishes MODY2 from T1D (where C-peptide is absent) and "
                "from late-stage MODY3/MODY1 (where it declines with progressive failure). "
                "C-peptide also confirms that reducing pharmacotherapy (stopping sulfonylure) "
                "will not leave the patient insulin-deficient."
            ),
        },
        {
            "term": "MODY2 vs MODY3 vs prediabetes",
            "definition": (
                "MODY2: mild stable HbA1c 5.6–7.6%; no treatment; no progression; family hx; "
                "antibody negative; C-peptide normal; OGTT increment <3.5 mmol/L. "
                "MODY3: progressive HbA1c; sulfonylure-responsive; renal glycosuria 50%; "
                "family hx; antibody negative. Prediabetes: no family hx pattern; HbA1c may "
                "progress; OGTT increment larger; insulin resistance markers. "
                "Key single discriminator: HbA1c stability over ≥3 years distinguishes MODY2 "
                "from both prediabetes and MODY3 without molecular testing."
            ),
        },
    ]

    return {"terms": terms}


# ---------------------------------------------------------------------------
# Module self-test (run as script only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== MODY2 Dashboard Self-Test ===\n")

    ov = get_overview()
    print(f"Disease  : {ov['disease']}")
    print(f"Cohort   : {ov['cohort_size']} patients  |  Seed: {ov['seed']}")
    print("KPIs:")
    for k, v in ov["kpis"].items():
        print(f"  {k}: {v}")

    print("\n--- Breakdown ---")
    bk = get_breakdown()
    for k, v in bk.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Definitions (terms only) ---")
    defs = get_definitions()
    for d in defs["terms"]:
        print(f"  {d['term']}")

    print(f"\nTotal terms defined: {len(defs['terms'])}")
    print("\nAll 3 functions returned successfully.")
