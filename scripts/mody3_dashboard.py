"""
MODY3 — HNF1A-MODY (Maturity-Onset Diabetes of the Young Type 3)
=================================================================
Gene       : HNF1A (Hepatocyte Nuclear Factor 1 Alpha)
Chromosome : 12q24.31
OMIM Gene  : *142410
OMIM Dis.  : #600496  (MODY3 / NIDDM2)
Inheritance: Autosomal Dominant (50% transmission per child)
Prevalence : ~1:10,000–1:20,000 (underdiagnosed)
Onset      : Late teens to early 30s (mean ~24–25 years)

Mechanism
---------
HNF1A is a transcription factor that governs hepatocyte and beta-cell gene
expression. Loss-of-function (LOF) variants cause haploinsufficiency →
impaired GLUT2 transporter and glucokinase expression in beta-cells →
reduced glucose uptake and sensing → blunted glucose-stimulated insulin
secretion (GSIS) → progressive beta-cell failure with intact (early) C-peptide.

Key Clinical Hallmarks
-----------------------
* Renal glycosuria in ~50% (HNF1A controls SGLT2 expression → low renal
  glucose threshold even at normoglycaemia)
* Autoantibodies uniformly NEGATIVE (GADA, ZnT8, IA-2)
* Family history positive ~90% (AD, high penetrance)
* Extraordinary sulfonylure sensitivity (100–1000× vs T2D)
* Frequent prior misdiagnosis as T1D or T2D

Treatment
---------
Sulfonylure FIRST-LINE (Level A): glibenclamide/glipizide/gliclazide at
doses 10–100× lower than standard T2D doses. ~85–90% achieve excellent
glycaemic control. Switch misdiagnosed T1D/T2D patients off insulin once
MODY3 confirmed.

Cohort: 40 patients, seed=303.
"""

import random
import statistics
import functools

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 303
_COHORT_SIZE = 40

_VARIANTS = [
    "P291fsinsC",
    "R272H",
    "R229Q",
    "T260M",
    "E132K",
    "G191D",
    "R131Q",
    "Other_missense",
    "Other_frameshift",
    "Other_nonsense",
]

_TREATMENTS = ["Sulfonylurea", "Sulfonylurea+Insulin", "Insulin", "Diet only"]
_TREATMENT_WEIGHTS = [0.55, 0.20, 0.15, 0.10]

_SULFO_RESPONSES = ["Excellent", "Good", "Partial", "Not_started"]
_SULFO_RESPONSE_WEIGHTS = [0.55, 0.25, 0.10, 0.10]

_MISDIAGNOSES = ["T1D", "T2D", "GDM", "None"]
_MISDIAGNOSIS_WEIGHTS = [0.30, 0.25, 0.10, 0.35]

_COMPLICATIONS_POOL = ["none", "retinopathy", "nephropathy", "neuropathy", "hepatic_adenoma"]

_HBA1C_MMOL_OFFSET = 10.929  # HbA1c (mmol/mol) = (HbA1c% - 2.15) / 0.0915  [IFCC]


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
    """Generate the 40-patient cohort deterministically (seed=303)."""
    rng = random.Random(_SEED)
    patients = []

    for i in range(1, _COHORT_SIZE + 1):
        age = rng.randint(16, 45)
        sex = rng.choice(["M", "F"])

        # Age at diagnosis: must be <= age; skewed toward late teens/20s
        age_at_dx = int(rng.triangular(14, min(age, 40), 24))
        age_at_dx = max(14, min(age_at_dx, age))
        duration = age - age_at_dx

        hba1c_pct = round(rng.uniform(6.0, 9.5), 1)
        hba1c_mmol = _hba1c_percent_to_mmol(hba1c_pct)
        c_peptide = round(rng.uniform(0.3, 1.8), 2)

        family_hx = rng.random() < 0.90
        renal_glycosuria = rng.random() < 0.50

        treatment = _weighted_choice(rng, _TREATMENTS, _TREATMENT_WEIGHTS)
        sulfo_response = _weighted_choice(rng, _SULFO_RESPONSES, _SULFO_RESPONSE_WEIGHTS)

        # HbA1c change on sulfo (only meaningful for those who started it)
        if sulfo_response in ("Excellent", "Good", "Partial"):
            hba1c_change = round(rng.uniform(-2.5, -0.5), 2)
        else:
            hba1c_change = None

        variant = rng.choice(_VARIANTS)
        misdiagnosis = _weighted_choice(rng, _MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)

        # Complications: none + 0-2 additional from pool
        comp_pool = [c for c in _COMPLICATIONS_POOL if c != "none"]
        n_comp = rng.randint(0, 2)
        # Hepatic adenoma mostly female
        if sex == "M" and "hepatic_adenoma" in comp_pool:
            comp_pool.remove("hepatic_adenoma")
        chosen_comps = rng.sample(comp_pool, min(n_comp, len(comp_pool)))
        # ~5% hepatic adenoma for females
        if sex == "F" and rng.random() < 0.05 and "hepatic_adenoma" not in chosen_comps:
            chosen_comps.append("hepatic_adenoma")
        complications = sorted(chosen_comps) if chosen_comps else ["none"]

        patients.append({
            "patient_id": f"MODY3-{i:03d}",
            "age": int(age),
            "sex": sex,
            "age_at_diagnosis": int(age_at_dx),
            "duration_years": int(duration),
            "hba1c_percent": float(hba1c_pct),
            "hba1c_mmol": float(hba1c_mmol),
            "c_peptide_nmol_L": float(c_peptide),
            "family_hx_positive": bool(family_hx),
            "renal_glycosuria": bool(renal_glycosuria),
            "antibody_status": "NEGATIVE",
            "current_treatment": treatment,
            "sulfo_response": sulfo_response,
            "hba1c_change_on_sulfo": float(hba1c_change) if hba1c_change is not None else None,
            "variant": variant,
            "complications": list(complications),
            "misdiagnosis_prior": misdiagnosis,
        })

    return patients


# Build cohort once at import time (no side-effect heavy; just data generation)
_COHORT: list = _build_cohort()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    """
    Return high-level overview of the MODY3 cohort including KPIs, patient
    list, key clinical facts, treatment summary, and diagnostic criteria.
    """
    patients = _COHORT

    # ---- KPI calculations ----
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
    n_renal_gly = sum(1 for p in patients if p["renal_glycosuria"])
    n_misdiagnosed = sum(1 for p in patients if p["misdiagnosis_prior"] != "None")
    n_controlled = sum(1 for p in patients if p["hba1c_percent"] < 7.0)
    n_any_complication = sum(
        1 for p in patients if p["complications"] != ["none"]
    )

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
        "pct_renal_glycosuria": round(100.0 * n_renal_gly / _COHORT_SIZE, 1),
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
        "MODY3 is the most common form of MODY (~35% of all MODY diagnoses), caused by HNF1A LOF variants.",
        "HNF1A haploinsufficiency impairs GLUT2 and glucokinase expression → blunted glucose-stimulated insulin secretion (GSIS).",
        "Renal glycosuria occurs in ~50% of patients because HNF1A also regulates SGLT2 (sodium-glucose cotransporter 2) in the kidney.",
        "Autoantibodies (GADA, ZnT8, IA-2) are uniformly negative — key differentiator from T1D.",
        "Patients show extraordinary sulfonylure sensitivity (100–1000× greater than T2D); start at 0.5–1 mg glibenclamide.",
        "~90% have a positive first-degree family history; inheritance is autosomal dominant with ~50% transmission risk.",
        "Up to 55% of MODY3 patients are misdiagnosed as T1D or T2D; switching off unnecessary insulin is a key clinical action.",
        "P291fsinsC is the most common European founder mutation (~8% of UK MODY3 cases).",
        "Hepatic adenoma occurs rarely (~5%), predominantly in women with prolonged oral contraceptive use.",
        "Pregnancy management: glyburide crosses the placenta — switch to insulin (or gliclazide) during pregnancy.",
        "MODY Probability Calculator (Exeter formula) score >25% should prompt molecular testing.",
        "C-peptide is detectable at diagnosis (unlike T1D) and declines only with prolonged disease duration.",
    ]

    treatment_summary = {
        "first_line": "Sulfonylurea (glibenclamide / glipizide / gliclazide) — Level A evidence",
        "dosing_caution": "Start at 0.5–1 mg glibenclamide; doses 10–100× lower than standard T2D doses",
        "response_rate": "~85–90% achieve excellent glycaemic control on low-dose sulfonylure",
        "hypoglycaemia_risk": "HIGH if T2D dosing used — titrate slowly",
        "metformin": "NOT first-line — MODY3 is an insulin secretion defect, not insulin resistance",
        "insulin_indications": "Long disease duration with marked beta-cell loss; pregnancy (glyburide crosses placenta)",
        "pregnancy": "Switch to insulin or gliclazide; glyburide crosses placenta and is contraindicated",
        "key_action": "Discontinue unnecessary insulin in patients previously misdiagnosed as T1D or T2D",
        "ocp_risk": "Oral contraceptive pills can worsen hyperglycaemia in MODY3 women — monitor closely",
    }

    diagnostic_criteria = {
        "molecular": "Next-gen sequencing MODY panel: HNF1A + HNF4A + GCK + HNF1B ± INS",
        "c_peptide": "Detectable C-peptide at diagnosis (>0.2 nmol/L); distinguishes from T1D",
        "autoantibodies": "GADA, ZnT8, IA-2 all negative — mandatory before MODY testing",
        "renal_glycosuria": "Urine dipstick positive for glucose with normal/near-normal serum glucose",
        "family_history": "First-degree relative with young-onset diabetes in ~90%",
        "mody_calculator": "Exeter MODY Probability Calculator score >25% → proceed to genetic testing",
        "ogtt_pattern": "Brisk early glucose rise (impaired phase 1 insulin), partial recovery; incremental rise >4.6 mmol/L",
        "hba1c_at_onset": "Typically 48–58 mmol/mol (6.5–7.5%) at diagnosis; progressive if untreated",
        "clinical_onset": "Late teens to early 30s; rarely presents in childhood (unlike GCK-MODY at birth)",
    }

    return {
        "disease": "MODY3 — HNF1A-MODY (Maturity-Onset Diabetes of the Young Type 3)",
        "gene": "HNF1A",
        "omim_gene": "*142410",
        "omim_disease": "#600496",
        "chromosome": "12q24.31",
        "inheritance": "Autosomal Dominant",
        "prevalence": "~1:10,000–1:20,000 (underdiagnosed)",
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
    and genetic dimensions of the MODY3 cohort.
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

    # Complication distribution (count patients with each complication)
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

    renal_glycosuria_count = int(sum(1 for p in patients if p["renal_glycosuria"]))
    family_hx_positive_count = int(sum(1 for p in patients if p["family_hx_positive"]))

    # Sex breakdown
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
        "age_at_diagnosis_tiers": {k: int(v) for k, v in dx_age_tiers.items()},
        "renal_glycosuria_count": renal_glycosuria_count,
        "family_hx_positive_count": family_hx_positive_count,
        "antibody_negative_pct": 100.0,
        "total_patients": int(_COHORT_SIZE),
    }


def get_definitions() -> dict:
    """
    Return a glossary of clinically important terms for MODY3 / HNF1A-MODY.
    Each entry has 'term' and 'definition' keys.
    """
    terms = [
        {
            "term": "HNF1A",
            "definition": (
                "Hepatocyte Nuclear Factor 1 Alpha — a transcription factor encoded on chromosome "
                "12q24.31 (OMIM *142410). It regulates gene expression in hepatocytes and pancreatic "
                "beta-cells, including GLUT2, glucokinase, and SGLT2. Loss-of-function variants cause "
                "MODY3 through haploinsufficiency, impairing glucose-stimulated insulin secretion."
            ),
        },
        {
            "term": "MODY3",
            "definition": (
                "Maturity-Onset Diabetes of the Young Type 3 (OMIM #600496 / NIDDM2). The most common "
                "MODY subtype (~35% of MODY diagnoses), caused by heterozygous HNF1A LOF variants. "
                "Characterised by progressive beta-cell failure, young onset (teens–30s), autosomal "
                "dominant inheritance, negative autoantibodies, and extraordinary sulfonylure sensitivity."
            ),
        },
        {
            "term": "Renal glycosuria",
            "definition": (
                "Presence of glucose in urine at normal or near-normal blood glucose concentrations "
                "(<10 mmol/L). In MODY3, HNF1A LOF downregulates SGLT2 (sodium-glucose cotransporter 2) "
                "in the renal proximal tubule, lowering the renal glucose threshold to ~5–6 mmol/L. "
                "Affects ~50% of MODY3 patients and can precede hyperglycaemia — a key diagnostic clue."
            ),
        },
        {
            "term": "Sulfonylurea (Sulfonylure)",
            "definition": (
                "Class of oral antidiabetic agents that close ATP-sensitive K⁺ channels on beta-cells "
                "via the SUR1 subunit, independent of glucose, triggering insulin secretion. MODY3 "
                "patients are 100–1000× more sensitive than T2D patients due to the downstream nature "
                "of their defect (K-ATP channel intact). First-line treatment at very low doses "
                "(0.5–1 mg glibenclamide). Risk: severe hypoglycaemia at T2D doses."
            ),
        },
        {
            "term": "HbA1c",
            "definition": (
                "Glycated haemoglobin A1c — a measure of average blood glucose over 2–3 months, "
                "expressed as % (NGSP/DCCT) or mmol/mol (IFCC). Target in treated MODY3 is <7% "
                "(<53 mmol/mol). Conversion: mmol/mol = (HbA1c% − 2.15) / 0.0915. MODY3 patients "
                "often present with HbA1c 6.5–7.5% and can rise substantially if untreated."
            ),
        },
        {
            "term": "C-peptide",
            "definition": (
                "C-peptide is a byproduct of insulin biosynthesis, cleaved from proinsulin in a 1:1 "
                "molar ratio with insulin. Detectable C-peptide (>0.2 nmol/L) at diagnosis confirms "
                "residual endogenous insulin secretion, distinguishing MODY3 from T1D (where C-peptide "
                "is absent/very low). C-peptide declines with increasing disease duration in MODY3."
            ),
        },
        {
            "term": "GSIS",
            "definition": (
                "Glucose-Stimulated Insulin Secretion — the physiological process by which rising "
                "blood glucose is sensed by beta-cells (via GLUT2 uptake and glucokinase phosphorylation), "
                "triggering ATP generation, K-ATP channel closure, membrane depolarisation, Ca²⁺ influx, "
                "and insulin exocytosis. HNF1A LOF impairs the early steps (GLUT2, glucokinase "
                "expression), blunting phase 1 insulin response."
            ),
        },
        {
            "term": "GLUT2",
            "definition": (
                "Glucose Transporter 2 (SLC2A2) — the high-capacity, low-affinity glucose transporter "
                "expressed in pancreatic beta-cells, hepatocytes, and intestinal epithelium. GLUT2 "
                "allows glucose entry proportional to blood glucose concentration, enabling glucose "
                "sensing. HNF1A directly regulates GLUT2 transcription; LOF → reduced GLUT2 → impaired "
                "glucose uptake into beta-cells → blunted GSIS."
            ),
        },
        {
            "term": "Haploinsufficiency",
            "definition": (
                "A mechanism of disease in which a single functional copy of a gene is insufficient "
                "to maintain normal physiological function. MODY3 is caused by HNF1A haploinsufficiency: "
                "one allele carries a pathogenic variant (typically nonsense, frameshift, or missense), "
                "and the remaining wild-type allele alone cannot sustain adequate HNF1A transcriptional "
                "activity in beta-cells."
            ),
        },
        {
            "term": "P291fsinsC",
            "definition": (
                "The most common HNF1A pathogenic variant in European populations (~8% of UK MODY3 "
                "cases). A frameshift mutation caused by insertion of a cytosine (C) at codon 291 in "
                "exon 4, leading to premature stop codon and loss of the transactivation domain. "
                "Acts as a European founder mutation; identified by standard NGS MODY panels. "
                "Confers classic MODY3 phenotype with excellent sulfonylure response."
            ),
        },
        {
            "term": "Autoantibodies",
            "definition": (
                "Antibodies directed against self-antigens. In diabetes, the key autoantibodies are "
                "GADA (glutamic acid decarboxylase), ZnT8 (zinc transporter 8), and IA-2 (islet "
                "antigen 2), which are markers of immune-mediated beta-cell destruction in T1D. "
                "In MODY3, autoantibodies are uniformly NEGATIVE — their absence in a young-onset "
                "diabetes patient with family history should prompt MODY genetic testing."
            ),
        },
        {
            "term": "MODY Probability Calculator",
            "definition": (
                "The Exeter MODY Probability Calculator — a validated clinical prediction tool "
                "developed at the University of Exeter. It uses age at diagnosis, BMI, HbA1c, "
                "current treatment, family history, and sex to estimate the probability of MODY "
                "vs T1D/T2D. A probability >25% is recommended as the threshold for molecular "
                "genetic testing. Available at diabetesgenes.org."
            ),
        },
        {
            "term": "Hepatic adenoma",
            "definition": (
                "A benign liver tumour that occurs rarely in MODY3 patients (~5%), predominantly "
                "in women with prolonged oral contraceptive pill (OCP) use. HNF1A regulates hepatocyte "
                "proliferation; biallelic somatic HNF1A mutations are found in a subset of sporadic "
                "hepatic adenomas (H-HCA subtype). In MODY3, surveillance with liver imaging is "
                "warranted in women on long-term OCP."
            ),
        },
        {
            "term": "OGTT",
            "definition": (
                "Oral Glucose Tolerance Test — a standardised test in which 75 g anhydrous glucose "
                "is given orally and blood glucose is measured at 0, 60, and 120 minutes. In MODY3, "
                "the OGTT shows a brisk early rise (impaired phase 1 insulin response) with an "
                "incremental rise >4.6 mmol/L (a sensitive marker for MODY vs GCK-MODY, where the "
                "rise is <4.6 mmol/L). Partial recovery at 120 min reflects residual secretory capacity."
            ),
        },
        {
            "term": "GCK-MODY vs HNF1A-MODY",
            "definition": (
                "GCK-MODY (MODY2) is caused by heterozygous glucokinase LOF variants → mild, stable "
                "fasting hyperglycaemia (6.0–7.9 mmol/L) from birth, requiring no treatment, with "
                "OGTT incremental rise <4.6 mmol/L. HNF1A-MODY (MODY3) causes progressive deteriorating "
                "glycaemia from teens/20s, responds dramatically to sulfonylure, shows OGTT increment "
                ">4.6 mmol/L, and causes renal glycosuria. Key distinction: GCK-MODY = 'reset glucose "
                "thermostat'; MODY3 = progressive secretory failure."
            ),
        },
        {
            "term": "HNF4A",
            "definition": (
                "Hepatocyte Nuclear Factor 4 Alpha — gene on chromosome 20q13.12 (OMIM *600281) "
                "whose LOF causes MODY1 (#125850). HNF4A is a nuclear receptor upstream of HNF1A "
                "in the transcription factor hierarchy. MODY1 is rarer than MODY3 but clinically "
                "similar; a key difference is that HNF4A mutations also cause macrosomia and "
                "transient neonatal hyperinsulinism in offspring."
            ),
        },
        {
            "term": "HNF1B",
            "definition": (
                "Hepatocyte Nuclear Factor 1 Beta — gene on chromosome 17q12 (OMIM *189907) whose "
                "LOF causes MODY5 (#137920). Unlike MODY3, MODY5 presents with renal cysts and "
                "diabetes (RCAD syndrome), uterine anomalies, hyperuricaemia, and exocrine pancreatic "
                "insufficiency. ~50% of MODY5 cases result from de novo whole-gene deletions; "
                "sulfonylure is generally less effective than in MODY3."
            ),
        },
        {
            "term": "Beta-cell",
            "definition": (
                "Insulin-secreting cells located in the islets of Langerhans in the endocrine "
                "pancreas (~60–80% of islet cells). Beta-cells sense glucose via GLUT2 and "
                "glucokinase, generate ATP, close K-ATP channels, and secrete insulin by "
                "exocytosis. In MODY3, HNF1A LOF impairs beta-cell gene expression from early "
                "adulthood, causing progressive but partial beta-cell failure (C-peptide preserved "
                "early, unlike T1D autoimmune destruction)."
            ),
        },
        {
            "term": "Misdiagnosis rate",
            "definition": (
                "The proportion of MODY3 patients incorrectly labelled as T1D or T2D before molecular "
                "confirmation. Estimated at ~80% overall (T1D ~30–40%, T2D ~25–30%). Misdiagnosis "
                "leads to suboptimal treatment (insulin instead of sulfonylure). Key clinical triggers "
                "to suspect MODY3: young onset, negative autoantibodies, family history, detectable "
                "C-peptide, renal glycosuria, and disproportionate sensitivity to insulin."
            ),
        },
        {
            "term": "Treatment switch",
            "definition": (
                "The clinical process of transitioning a MODY3 patient from their current treatment "
                "(typically insulin or metformin, due to prior misdiagnosis) to a sulfonylure. "
                "Requires confirmation of MODY3 diagnosis by molecular testing and normal/detectable "
                "C-peptide. Switch should be done carefully with gradual insulin reduction alongside "
                "low-dose sulfonylure introduction; ~85–90% achieve excellent glycaemic control "
                "post-switch. Pregnancy is a contraindication for glyburide — use insulin."
            ),
        },
        {
            "term": "SGLT2",
            "definition": (
                "Sodium-Glucose Cotransporter 2 (SLC5A2) — expressed in the renal proximal tubule "
                "and responsible for ~90% of urinary glucose reabsorption. HNF1A directly regulates "
                "SGLT2 transcription; HNF1A LOF → reduced SGLT2 expression → lower renal glucose "
                "threshold → glycosuria at normoglycaemia. This is the pathomechanism of renal "
                "glycosuria in MODY3, and incidentally the same transporter targeted by SGLT2 "
                "inhibitor drugs (gliflozins) used in T2D treatment."
            ),
        },
        {
            "term": "Autosomal Dominant",
            "definition": (
                "An inheritance pattern in which a single pathogenic variant in one allele of an "
                "autosomal gene is sufficient to cause disease. In MODY3, one HNF1A allele is "
                "pathogenic (LOF) and the other is wild-type but insufficient for full function "
                "(haploinsufficiency). Each offspring of an affected parent has a 50% chance of "
                "inheriting the pathogenic allele. Penetrance is high (~90%) but variable expressivity "
                "means age at onset and severity differ within families."
            ),
        },
        {
            "term": "NGS MODY Panel",
            "definition": (
                "Next-Generation Sequencing panel for MODY — a targeted molecular diagnostic test "
                "that simultaneously sequences key MODY genes: HNF1A, HNF4A, GCK, HNF1B, and "
                "optionally INS, ABCC8, KCNJ11, PDX1. Recommended when MODY Probability Calculator "
                "score >25% in a patient with young-onset diabetes, negative autoantibodies, and "
                "detectable C-peptide. Identifies pathogenic variants in ~80% of clinically suspected "
                "MODY3 cases."
            ),
        },
    ]

    return {"terms": terms}


# ---------------------------------------------------------------------------
# Module self-test (run as script only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== MODY3 Dashboard Self-Test ===\n")

    ov = get_overview()
    print(f"Disease  : {ov['disease']}")
    print(f"Cohort   : {ov['cohort_size']} patients  |  Seed: {ov['seed']}")
    print(f"KPIs     :")
    for k, v in ov["kpis"].items():
        print(f"  {k}: {v}")

    print("\n--- Breakdown ---")
    bk = get_breakdown()
    for k, v in bk.items():
        print(f"  {k}: {v}")

    print("\n--- Definitions (terms only) ---")
    defs = get_definitions()
    for d in defs["terms"]:
        print(f"  {d['term']}")

    print(f"\nTotal terms defined: {len(defs['terms'])}")
    print("\nAll 3 functions returned successfully.")
