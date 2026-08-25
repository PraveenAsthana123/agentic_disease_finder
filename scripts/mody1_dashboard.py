"""
MODY1 — HNF4A-MODY (Maturity-Onset Diabetes of the Young Type 1)
=================================================================
Gene       : HNF4A (Hepatocyte Nuclear Factor 4 Alpha)
Chromosome : 20q13.12
OMIM Gene  : *600281
OMIM Dis.  : #125850  (MODY1)
Inheritance: Autosomal Dominant (50% transmission per child)
Prevalence : ~1:50,000–1:100,000 (2nd most common after MODY3, ~10% of all MODY)
Onset      : Teens to early 40s; neonatal hypoglycaemia in ~50% at birth

Mechanism
---------
HNF4A is a nuclear receptor / master transcription factor that directly regulates
HNF1A and hundreds of hepatic and beta-cell genes. LOF (haploinsufficiency) → impaired
beta-cell differentiation and glucose-stimulated insulin secretion (GSIS). Uniquely,
HNF4A is also required for fetal beta-cell development — heterozygous LOF variants
cause TRANSIENT NEONATAL HYPERINSULINISM (macrosomia + hypoglycaemia at birth) that
resolves by 1-2 months, followed by MODY developing in the teens/adulthood.

Key Clinical Hallmarks
-----------------------
* Macrosomia at birth (≥4 kg, 50–60%) + neonatal hypoglycaemia (50–60%) — UNIQUE to MODY1
* Transient neonatal hyperinsulinism (resolves by 6 weeks–3 months) — diazoxide responsive
* NO renal glycosuria (key differentiator from MODY3; HNF4A does NOT directly regulate SGLT2)
* Autoantibodies uniformly NEGATIVE (GADA, ZnT8, IA-2)
* Family history positive ~85–90% (AD, high penetrance)
* Sulfonylure sensitivity: very good response (~85–90%), slightly less dramatic than MODY3
* Progressive beta-cell failure with disease duration; C-peptide preserved early
* Overlap with MODY3 pathway (HNF4A → regulates HNF1A → same downstream cascade)

Treatment
---------
Sulfonylure FIRST-LINE (Level A): glibenclamide/glipizide/gliclazide at low doses.
~85–90% achieve excellent glycaemic control. Neonatal phase: diazoxide for transient
hyperinsulinism. Pregnancy: insulin preferred (glyburide crosses placenta).

Cohort: 40 patients, seed=305.
"""

import random
import statistics
import functools

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 305
_COHORT_SIZE = 40

_VARIANTS = [
    "Arg154X",
    "C106R",
    "E276Q",
    "D206Y",
    "R127W",
    "V255M",
    "Q268X",
    "Other_missense",
    "Other_frameshift",
    "Other_nonsense",
]

_TREATMENTS = ["Sulfonylurea", "Sulfonylurea+Insulin", "Insulin", "Diet only"]
_TREATMENT_WEIGHTS = [0.50, 0.22, 0.18, 0.10]

_SULFO_RESPONSES = ["Excellent", "Good", "Partial", "Not_started"]
_SULFO_RESPONSE_WEIGHTS = [0.52, 0.28, 0.10, 0.10]

_MISDIAGNOSES = ["T1D", "T2D", "GDM", "None"]
_MISDIAGNOSIS_WEIGHTS = [0.28, 0.22, 0.10, 0.40]

_COMPLICATIONS_POOL = ["none", "retinopathy", "nephropathy", "neuropathy", "hepatic_adenoma"]

_HBA1C_MMOL_OFFSET = 10.929


def _hba1c_percent_to_mmol(pct: float) -> float:
    """Convert HbA1c % (NGSP/DCCT) to mmol/mol (IFCC)."""
    return round((pct - 2.15) / 0.0915, 1)


def _weighted_choice(rng: random.Random, choices, weights):
    """Pick one item from *choices* using *weights* (probabilities summing ~1)."""
    r = rng.random()
    cumulative = 0.0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if r < cumulative:
            return choice
    return choices[-1]


def _build_cohort() -> list:
    """Generate the 40-patient cohort deterministically (seed=305)."""
    rng = random.Random(_SEED)
    patients = []

    for i in range(1, _COHORT_SIZE + 1):
        age = rng.randint(16, 50)
        sex = rng.choice(["M", "F"])

        # Age at diagnosis: must be <= age; skewed toward late teens/20s-30s (slightly older than MODY3)
        age_at_dx = int(rng.triangular(14, min(age, 45), 27))
        age_at_dx = max(14, min(age_at_dx, age))
        duration = age - age_at_dx

        hba1c_pct = round(rng.uniform(6.0, 9.8), 1)
        hba1c_mmol = _hba1c_percent_to_mmol(hba1c_pct)
        c_peptide = round(rng.uniform(0.3, 1.9), 2)

        family_hx = rng.random() < 0.87
        # MODY1 does NOT cause renal glycosuria (HNF4A does not regulate SGLT2 directly)
        renal_glycosuria = False

        # Neonatal history (unique MODY1 feature)
        neonatal_macrosomia = rng.random() < 0.55
        neonatal_hypoglycaemia = rng.random() < 0.50

        treatment = _weighted_choice(rng, _TREATMENTS, _TREATMENT_WEIGHTS)
        sulfo_response = _weighted_choice(rng, _SULFO_RESPONSES, _SULFO_RESPONSE_WEIGHTS)

        if sulfo_response in ("Excellent", "Good", "Partial"):
            hba1c_change = round(rng.uniform(-2.4, -0.4), 2)
        else:
            hba1c_change = None

        variant = rng.choice(_VARIANTS)
        misdiagnosis = _weighted_choice(rng, _MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)

        # Complications
        comp_pool = [c for c in _COMPLICATIONS_POOL if c != "none"]
        n_comp = rng.randint(0, 2)
        if sex == "M" and "hepatic_adenoma" in comp_pool:
            comp_pool.remove("hepatic_adenoma")
        chosen_comps = rng.sample(comp_pool, min(n_comp, len(comp_pool)))
        if sex == "F" and rng.random() < 0.04 and "hepatic_adenoma" not in chosen_comps:
            chosen_comps.append("hepatic_adenoma")
        complications = sorted(chosen_comps) if chosen_comps else ["none"]

        patients.append({
            "patient_id": f"MODY1-{i:03d}",
            "age": int(age),
            "sex": sex,
            "age_at_diagnosis": int(age_at_dx),
            "duration_years": int(duration),
            "hba1c_percent": float(hba1c_pct),
            "hba1c_mmol": float(hba1c_mmol),
            "c_peptide_nmol_L": float(c_peptide),
            "family_hx_positive": bool(family_hx),
            "renal_glycosuria": bool(renal_glycosuria),  # Always False for MODY1
            "neonatal_macrosomia": bool(neonatal_macrosomia),
            "neonatal_hypoglycaemia": bool(neonatal_hypoglycaemia),
            "antibody_status": "NEGATIVE",
            "current_treatment": treatment,
            "sulfo_response": sulfo_response,
            "hba1c_change_on_sulfo": float(hba1c_change) if hba1c_change is not None else None,
            "variant": variant,
            "complications": list(complications),
            "misdiagnosis_prior": misdiagnosis,
        })

    return patients


# Build cohort once at import time
_COHORT: list = _build_cohort()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    """
    Return high-level overview of the MODY1 cohort including KPIs, patient
    list, key clinical facts, treatment summary, and diagnostic criteria.
    """
    patients = _COHORT

    hba1c_values = [p["hba1c_percent"] for p in patients]
    c_peptide_values = [p["c_peptide_nmol_L"] for p in patients]
    durations = [p["duration_years"] for p in patients]
    ages = [p["age"] for p in patients]
    dx_ages = [p["age_at_diagnosis"] for p in patients]

    n_sulfo = sum(
        1 for p in patients if p["current_treatment"] in ("Sulfonylurea", "Sulfonylurea+Insulin")
    )
    n_excellent = sum(1 for p in patients if p["sulfo_response"] == "Excellent")
    n_good = sum(1 for p in patients if p["sulfo_response"] == "Good")
    n_family_hx = sum(1 for p in patients if p["family_hx_positive"])
    n_macrosomia = sum(1 for p in patients if p["neonatal_macrosomia"])
    n_neonatal_hypo = sum(1 for p in patients if p["neonatal_hypoglycaemia"])
    n_misdiagnosed = sum(1 for p in patients if p["misdiagnosis_prior"] != "None")
    n_controlled = sum(1 for p in patients if p["hba1c_percent"] < 7.0)
    n_any_complication = sum(1 for p in patients if p["complications"] != ["none"])

    hba1c_changes = [
        p["hba1c_change_on_sulfo"]
        for p in patients
        if p["hba1c_change_on_sulfo"] is not None
    ]

    kpis = {
        "cohort_size": int(_COHORT_SIZE),
        "mean_age_years": round(statistics.mean(ages), 1),
        "mean_age_at_diagnosis_years": round(statistics.mean(dx_ages), 1),
        "mean_duration_years": round(statistics.mean(durations), 1),
        "mean_hba1c_percent": round(statistics.mean(hba1c_values), 1),
        "mean_hba1c_mmol": round(statistics.mean([p["hba1c_mmol"] for p in patients]), 1),
        "mean_c_peptide_nmol_L": round(statistics.mean(c_peptide_values), 2),
        "pct_on_sulfonylurea": round(100.0 * n_sulfo / _COHORT_SIZE, 1),
        "pct_sulfo_excellent_response": round(100.0 * n_excellent / _COHORT_SIZE, 1),
        "pct_sulfo_good_or_excellent": round(100.0 * (n_excellent + n_good) / _COHORT_SIZE, 1),
        "pct_family_hx_positive": round(100.0 * n_family_hx / _COHORT_SIZE, 1),
        "pct_neonatal_macrosomia": round(100.0 * n_macrosomia / _COHORT_SIZE, 1),
        "pct_neonatal_hypoglycaemia": round(100.0 * n_neonatal_hypo / _COHORT_SIZE, 1),
        "pct_renal_glycosuria": 0.0,  # MODY1 NEVER causes renal glycosuria
        "pct_antibody_negative": 100.0,
        "pct_prior_misdiagnosis": round(100.0 * n_misdiagnosed / _COHORT_SIZE, 1),
        "pct_hba1c_controlled_lt7": round(100.0 * n_controlled / _COHORT_SIZE, 1),
        "pct_any_complication": round(100.0 * n_any_complication / _COHORT_SIZE, 1),
        "mean_hba1c_change_on_sulfo_pct": (
            round(statistics.mean(hba1c_changes), 2) if hba1c_changes else None
        ),
        "male_count": int(sum(1 for p in patients if p["sex"] == "M")),
        "female_count": int(sum(1 for p in patients if p["sex"] == "F")),
    }

    key_facts = [
        "MODY1 is the 2nd most common MODY form (~10% of all MODY), caused by HNF4A LOF variants (haploinsufficiency).",
        "HNF4A is the master upstream regulator of HNF1A — LOF collapses the same hepatic/beta-cell transcriptional cascade as MODY3.",
        "MODY1-UNIQUE: ~50–60% of carriers are macrosomic at birth (≥4 kg) due to fetal hyperinsulinism from HNF4A LOF.",
        "Transient neonatal hyperinsulinism (TNH) in ~50% — diazoxide responsive; resolves by 6 weeks–3 months — predates adult diabetes by decades.",
        "NO renal glycosuria in MODY1 (unlike MODY3) — HNF4A does not directly regulate SGLT2; key clinical differentiator.",
        "Autoantibodies (GADA, ZnT8, IA-2) are uniformly NEGATIVE — mandatory negative screen before MODY testing.",
        "Family history positive in ~85–90%; autosomal dominant, 50% per-pregnancy transmission risk.",
        "Sulfonylure FIRST-LINE (Level A): ~85–90% achieve excellent/good glycaemic control; start at low doses (sensitivity less extreme than MODY3 but still marked).",
        "C-peptide is detectable at diagnosis; progressive decline with disease duration but preserved early (unlike T1D autoimmune destruction).",
        "HNF4A shares ~10 amino acid residues with HNF1A's DNA-binding domain; both bind similar promoter elements — hence clinical and biochemical overlap.",
        "Neonatal history (macrosomia, neonatal hypoglycaemia) in a parent or sibling is a STRONG diagnostic clue for MODY1 in the family.",
        "NGS MODY panel (HNF4A + HNF1A + GCK + HNF1B ± INS) required for molecular confirmation; clinical suspicion alone is insufficient.",
    ]

    treatment_summary = {
        "first_line": "Sulfonylurea (glibenclamide / glipizide / gliclazide) — Level A evidence",
        "dosing_note": "Doses are lower than T2D but slightly higher than MODY3; start at 2.5 mg glibenclamide and titrate",
        "response_rate": "~85–90% achieve excellent/good glycaemic control on sulfonylure",
        "hypoglycaemia_risk": "Moderate — lower than MODY3 but still significant vs T2D dosing; titrate carefully",
        "neonatal_phase": "Diazoxide (K-ATP channel opener) for transient neonatal hyperinsulinism; resolves by 3 months",
        "metformin": "NOT first-line — MODY1 is an insulin secretion defect (beta-cell), not insulin resistance",
        "insulin_indications": "Long disease duration with marked beta-cell exhaustion; pregnancy",
        "pregnancy": "Switch to insulin for pregnancy — glyburide crosses placenta; neonatal macrosomia risk even without maternal hyperglycaemia",
        "key_action": "Discontinue unnecessary insulin in misdiagnosed patients; check for neonatal history in family members",
        "surveillance": "Annual HbA1c + C-peptide; screen first-degree relatives; watch for neonatal hypoglycaemia in offspring",
    }

    diagnostic_criteria = {
        "molecular": "NGS MODY panel: HNF4A (with HNF1A, GCK, HNF1B, INS) — pathogenic variant confirms diagnosis",
        "c_peptide": "Detectable C-peptide (>0.2 nmol/L); distinguishes from T1D; declines with duration",
        "autoantibodies": "GADA, ZnT8, IA-2 all NEGATIVE — mandatory before molecular testing",
        "neonatal_clue": "Macrosomia at birth (≥4 kg) + neonatal hypoglycaemia in proband or family member",
        "no_renal_glycosuria": "Urine dipstick glucose NEGATIVE despite hyperglycaemia — CONTRAST with MODY3",
        "family_history": "First-degree relative with young-onset diabetes ~85–90%; neonatal history in offspring",
        "mody_calculator": "Exeter MODY Probability Calculator >25% → molecular testing; neonatal history boosts score",
        "ogtt_pattern": "Similar to MODY3: impaired phase-1 insulin response, incremental rise >4.6 mmol/L",
        "clinical_onset": "Teens to early 40s; slightly later mean onset than MODY3 (~27 vs ~24 years)",
        "hnf4a_vs_hnf1a": "No renal glycosuria (MODY1) vs 50% renal glycosuria (MODY3) — single strongest clinical differentiator at presentation",
    }

    return {
        "disease": "MODY1 — HNF4A-MODY (Maturity-Onset Diabetes of the Young Type 1)",
        "gene": "HNF4A",
        "omim_gene": "*600281",
        "omim_disease": "#125850",
        "chromosome": "20q13.12",
        "inheritance": "Autosomal Dominant",
        "prevalence": "~1:50,000–1:100,000 (~10% of all MODY, 2nd most common)",
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
    and genetic dimensions of the MODY1 cohort.
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

    # Sulfo response distribution
    sulfo_dist: dict = {}
    for p in patients:
        r = p["sulfo_response"]
        sulfo_dist[r] = sulfo_dist.get(r, 0) + 1

    # Misdiagnosis distribution
    misdiag_dist: dict = {}
    for p in patients:
        m = p["misdiagnosis_prior"]
        misdiag_dist[m] = misdiag_dist.get(m, 0) + 1

    # Complication distribution
    comp_dist: dict = {c: 0 for c in _COMPLICATIONS_POOL}
    for p in patients:
        for c in p["complications"]:
            if c in comp_dist:
                comp_dist[c] += 1

    # Age groups
    age_groups: dict = {"<20": 0, "20-29": 0, "30-39": 0, "40+": 0}
    for p in patients:
        a = p["age"]
        if a < 20:
            age_groups["<20"] += 1
        elif a < 30:
            age_groups["20-29"] += 1
        elif a < 40:
            age_groups["30-39"] += 1
        else:
            age_groups["40+"] += 1

    # HbA1c tiers
    hba1c_tiers: dict = {
        "Controlled <7%": 0,
        "Borderline 7-8%": 0,
        "Elevated 8-9%": 0,
        "Poor >9%": 0,
    }
    for p in patients:
        h = p["hba1c_percent"]
        if h < 7.0:
            hba1c_tiers["Controlled <7%"] += 1
        elif h < 8.0:
            hba1c_tiers["Borderline 7-8%"] += 1
        elif h < 9.0:
            hba1c_tiers["Elevated 8-9%"] += 1
        else:
            hba1c_tiers["Poor >9%"] += 1

    family_hx_positive_count = int(sum(1 for p in patients if p["family_hx_positive"]))
    n_macrosomia = int(sum(1 for p in patients if p["neonatal_macrosomia"]))
    n_neonatal_hypo = int(sum(1 for p in patients if p["neonatal_hypoglycaemia"]))

    sex_dist = {
        "Male": int(sum(1 for p in patients if p["sex"] == "M")),
        "Female": int(sum(1 for p in patients if p["sex"] == "F")),
    }

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
    dx_age_tiers: dict = {"<18": 0, "18-24": 0, "25-30": 0, "31+": 0}
    for p in patients:
        d = p["age_at_diagnosis"]
        if d < 18:
            dx_age_tiers["<18"] += 1
        elif d < 25:
            dx_age_tiers["18-24"] += 1
        elif d <= 30:
            dx_age_tiers["25-30"] += 1
        else:
            dx_age_tiers["31+"] += 1

    # Neonatal history summary
    neonatal_summary = {
        "macrosomia_count": n_macrosomia,
        "macrosomia_pct": round(100.0 * n_macrosomia / _COHORT_SIZE, 1),
        "neonatal_hypoglycaemia_count": n_neonatal_hypo,
        "neonatal_hypoglycaemia_pct": round(100.0 * n_neonatal_hypo / _COHORT_SIZE, 1),
        "both_macrosomia_and_hypo": int(
            sum(1 for p in patients if p["neonatal_macrosomia"] and p["neonatal_hypoglycaemia"])
        ),
    }

    return {
        "variant_distribution": {k: int(v) for k, v in sorted(
            variant_dist.items(), key=lambda x: -x[1]
        )},
        "treatment_distribution": {k: int(v) for k, v in treatment_dist.items()},
        "sulfo_response_distribution": {k: int(v) for k, v in sulfo_dist.items()},
        "misdiagnosis_distribution": {k: int(v) for k, v in sorted(
            misdiag_dist.items(), key=lambda x: -x[1]
        )},
        "complication_distribution": {k: int(v) for k, v in comp_dist.items()},
        "age_groups": {k: int(v) for k, v in age_groups.items()},
        "hba1c_tiers": {k: int(v) for k, v in hba1c_tiers.items()},
        "sex_distribution": sex_dist,
        "duration_tiers": {k: int(v) for k, v in duration_tiers.items()},
        "dx_age_tiers": {k: int(v) for k, v in dx_age_tiers.items()},
        "family_hx_positive_count": family_hx_positive_count,
        "renal_glycosuria_count": 0,  # Always 0 for MODY1
        "neonatal_summary": neonatal_summary,
        "total_patients": int(_COHORT_SIZE),
    }


def get_definitions() -> dict:
    """
    Return clinical and molecular definitions for key MODY1 terms, suitable
    for a 'Definitions' panel in the frontend dashboard.
    """
    terms = [
        {
            "term": "MODY1",
            "definition": (
                "Maturity-Onset Diabetes of the Young Type 1 — the second most common form of "
                "MODY (~10% of all cases), caused by autosomal dominant loss-of-function (LOF) "
                "variants in HNF4A (Hepatocyte Nuclear Factor 4 Alpha) on chromosome 20q13.12. "
                "Presents as young-onset, antibody-negative diabetes with preserved C-peptide, "
                "excellent sulfonylure response, and the unique feature of neonatal macrosomia "
                "and transient hyperinsulinism at birth."
            ),
        },
        {
            "term": "HNF4A",
            "definition": (
                "Hepatocyte Nuclear Factor 4 Alpha — a nuclear receptor transcription factor "
                "(OMIM *600281) on chromosome 20q13.12. It is the master upstream regulator of "
                "HNF1A and controls hundreds of genes involved in hepatic lipid/glucose metabolism "
                "and pancreatic beta-cell development and function. Haploinsufficiency → impaired "
                "GSIS and progressive beta-cell failure. HNF4A also drives fetal beta-cell "
                "proliferation; LOF → paradoxical fetal hyperinsulinism before adult-onset diabetes."
            ),
        },
        {
            "term": "HNF4A–HNF1A axis",
            "definition": (
                "HNF4A directly transcribes HNF1A: LOF of either gene collapses the same "
                "downstream cascade — impaired GLUT2, glucokinase, and insulin gene expression in "
                "beta-cells. This explains the clinical overlap between MODY1 and MODY3. The key "
                "clinical differentiator is the absence of renal glycosuria in MODY1 (HNF4A does "
                "not directly regulate SGLT2 expression in the kidney) versus its presence (~50%) "
                "in MODY3 (HNF1A directly regulates SGLT2)."
            ),
        },
        {
            "term": "Transient Neonatal Hyperinsulinism (TNH)",
            "definition": (
                "A transient hyperinsulinaemic hypoglycaemia state at birth occurring in ~50% of "
                "MODY1 neonates, caused by fetal HNF4A LOF which paradoxically increases beta-cell "
                "mass/activity in utero. Presents as macrosomia (birth weight ≥4 kg) and symptomatic "
                "neonatal hypoglycaemia requiring diazoxide, feeds, or IV glucose. Resolves by "
                "6 weeks–3 months as postnatal HNF4A dosage effects adjust. Decades later the same "
                "carrier develops MODY as progressive GSIS failure ensues — a 'two-hit' lifecycle."
            ),
        },
        {
            "term": "Macrosomia",
            "definition": (
                "Birth weight ≥4 kg (>90th centile for gestational age). Occurs in ~50–60% of "
                "MODY1 neonates due to fetal hyperinsulinism (HNF4A LOF paradoxically increases "
                "fetal beta-cell insulin output). Macrosomia in a child who later develops "
                "young-onset diabetes, or in a parent's or sibling's birth history, is a STRONG "
                "clinical clue to MODY1. It is absent in MODY3 and MODY2, making it the most "
                "distinctive neonatal feature across MODY subtypes."
            ),
        },
        {
            "term": "No renal glycosuria",
            "definition": (
                "The absence of glucose in urine despite hyperglycaemia — a defining difference "
                "between MODY1 and MODY3. In MODY3, HNF1A LOF reduces SGLT2 expression in the "
                "renal proximal tubule, lowering the glucose reabsorption threshold (glycosuria at "
                "normal or near-normal plasma glucose). HNF4A does NOT directly regulate SGLT2, "
                "so MODY1 patients have a normal renal glucose threshold and a negative urine "
                "dipstick. This single feature can help differentiate MODY1 from MODY3 clinically."
            ),
        },
        {
            "term": "Sulfonylurea (MODY1)",
            "definition": (
                "Sulfonylureas (e.g., glibenclamide, glipizide, gliclazide) close K-ATP channels "
                "in beta-cells independently of glucose → triggering insulin exocytosis. In MODY1 "
                "the residual wild-type HNF4A allele still provides enough K-ATP machinery for "
                "drug binding; ~85–90% of patients achieve excellent or good glycaemic control. "
                "Sulfonylure sensitivity is significant but slightly less extreme than MODY3 "
                "(100–1000× more sensitive). Start at standard low doses (e.g., 2.5 mg "
                "glibenclamide) and titrate; avoid T2D doses."
            ),
        },
        {
            "term": "Diazoxide",
            "definition": (
                "A K-ATP channel OPENER (opposite mechanism to sulfonylureas) used to suppress "
                "insulin secretion in neonatal hyperinsulinism. In MODY1 neonates with transient "
                "hyperinsulinism, diazoxide (5–20 mg/kg/day) resolves hypoglycaemia and is the "
                "first-line pharmacological treatment for the neonatal phase. It is only needed "
                "transiently (weeks to months) until the TNH resolves. Not relevant for adult "
                "MODY1 management (adults have insufficient, not excess, insulin secretion)."
            ),
        },
        {
            "term": "Haploinsufficiency",
            "definition": (
                "A mechanism in which a single functional copy of a gene is insufficient for "
                "normal cellular function. MODY1 is caused by haploinsufficiency of HNF4A: one "
                "allele carries a LOF variant, the other is wild-type but cannot alone sustain "
                "adequate HNF4A transcriptional activity for beta-cell and hepatic gene programs. "
                "Progressive phenotype with age as the remaining allele accumulates age-related "
                "epigenetic changes is consistent with increasing HbA1c over disease duration."
            ),
        },
        {
            "term": "Autosomal Dominant (AD)",
            "definition": (
                "An inheritance pattern where a single pathogenic allele causes disease. Each "
                "child of an HNF4A MODY1 carrier has a 50% chance of inheriting the variant. "
                "Penetrance is high (~90%) but expressivity varies — some carriers present in "
                "teenage years, others not until their 40s. The family history in MODY1 often "
                "includes both the neonatal phenotype (macrosomia in children) and the adult "
                "phenotype (young-onset diabetes in parents/grandparents) within the same pedigree."
            ),
        },
        {
            "term": "NGS MODY Panel",
            "definition": (
                "Next-Generation Sequencing panel for MODY — a targeted multi-gene test "
                "simultaneously sequencing HNF4A, HNF1A, GCK, HNF1B, and optionally INS, "
                "ABCC8, KCNJ11, PDX1. Recommended when MODY Probability Calculator score >25% "
                "in a patient with young-onset, antibody-negative diabetes and detectable "
                "C-peptide. Family history of neonatal macrosomia/hypoglycaemia further raises "
                "pre-test probability of MODY1 specifically. HNF4A sequencing detects pathogenic "
                "variants in ~80% of clinically suspected MODY1 cases."
            ),
        },
        {
            "term": "C-peptide",
            "definition": (
                "The connecting peptide cleaved from proinsulin during insulin biosynthesis — a "
                "marker of endogenous insulin secretion. In MODY1, C-peptide is detectable at "
                "diagnosis (unlike T1D where autoimmune destruction abolishes it). C-peptide "
                "declines progressively with disease duration in MODY1 as HNF4A haploinsufficiency "
                "leads to cumulative beta-cell failure. Measurement of C-peptide is a key step in "
                "distinguishing MODY from T1D and in monitoring beta-cell reserve longitudinally."
            ),
        },
        {
            "term": "MODY1 vs MODY3 differential",
            "definition": (
                "Both MODY1 (HNF4A) and MODY3 (HNF1A) cause young-onset, antibody-negative, "
                "sulfonylure-responsive diabetes via the same HNF4A→HNF1A transcriptional axis. "
                "Key differences: (1) Renal glycosuria: 50% in MODY3, 0% in MODY1. "
                "(2) Neonatal macrosomia + TNH: 50–60% in MODY1, absent in MODY3. "
                "(3) Prevalence: MODY3 ~35% of all MODY, MODY1 ~10%. "
                "(4) Sulfonylure sensitivity: MODY3 100–1000× more than T2D, MODY1 marked but "
                "slightly less extreme. The renal glycosuria urine dipstick is the fastest "
                "single-test clinical differentiator at the bedside."
            ),
        },
    ]

    return {"terms": terms}


# ---------------------------------------------------------------------------
# Module self-test (run as script only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== MODY1 Dashboard Self-Test ===\n")

    ov = get_overview()
    print(f"Disease  : {ov['disease']}")
    print(f"Cohort   : {ov['cohort_size']} patients  |  Seed: {ov['seed']}")
    print(f"KPIs     :")
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
