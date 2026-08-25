"""
MODY5 — HNF1B-MODY / RCAD Syndrome (Maturity-Onset Diabetes of the Young Type 5)
===================================================================================
Gene       : HNF1B (Hepatocyte Nuclear Factor 1 Beta)
Chromosome : 17q12
OMIM Gene  : *189907
OMIM Dis.  : #137920  (MODY5)
Inheritance: Autosomal Dominant (50% transmission) — BUT ~50% are de-novo mutations
Prevalence : ~1:10,000–1:50,000 (~5% of all MODY; markedly underdiagnosed)
Alias      : RCAD = Renal Cysts And Diabetes syndrome

Mechanism
---------
HNF1B encodes a transcription factor expressed in kidney, pancreas, liver, and Müllerian
duct. It regulates multiple target genes critical for:
  * Kidney tubulogenesis (renal cyst suppression) — LOF → cysts at 17q12
  * Pancreatic beta-cell development (endocrine) → progressive beta-cell loss → diabetes
  * Pancreatic exocrine cell development → exocrine pancreatic insufficiency
  * Müllerian duct development (female genital tract)
  * Renal magnesium reabsorption (FXYD2 subunit regulation)

MODY5 is therefore a MULTI-ORGAN syndrome, not just a pancreatic disorder.

MODY5-UNIQUE Features (critical differentiators)
-------------------------------------------------
1. RENAL STRUCTURAL ABNORMALITIES PRECEDE DIABETES BY YEARS:
     Multicystic dysplastic kidneys, hypoplastic kidneys, oligomeganephronia — often
     detected antenatally on obstetric ultrasound, or incidentally before diabetes onset.
     ~70% of MODY5 patients have renal cysts/structural abnormalities.

2. PANCREATIC ATROPHY (BOTH EXOCRINE + ENDOCRINE):
     Visible on CT/MRI — pancreatic volume reduced by ~50%. Causes progressive beta-cell
     failure AND exocrine pancreatic insufficiency (steatorrhoea, malabsorption).
     C-peptide falls progressively.

3. NO SULFONYLUREA RESPONSE:
     Unlike MODY1/MODY3 (sulfonylure 85–90% effective), MODY5 beta-cells are structurally
     lost (atrophy) rather than functionally impaired. Sulfonylure cannot recruit absent
     beta-cells → insulin is required, usually from early in the disease course.

4. HYPOMAGNESAEMIA (renal magnesium wasting):
     HNF1B drives FXYD2 (γ-subunit of renal Na+/K+-ATPase) in the distal convoluted
     tubule, critical for Mg²⁺ reabsorption. LOF → renal Mg wasting → serum Mg low →
     symptoms (cramps, weakness, arrhythmia). Supplement orally.

5. FEMALE GENITAL TRACT MALFORMATIONS:
     Müllerian duct anomalies in ~25% females: bicornuate/arcuate uterus, vaginal
     atresia/septum, single-horn uterus. May present with infertility or menstrual
     irregularity.

6. HIGH DE-NOVO RATE (~50%):
     Unlike MODY1/2/3 where family history is almost universal (75–90%), HNF1B LOF is
     de novo in ~50% of cases — whole-gene deletion (17q12 microdeletion) is the most
     common mutational mechanism (~50–60%), not point mutations.

7. 17q12 MICRODELETION SYNDROME:
     Large contiguous deletions at 17q12 additionally encompass LHX1, ACACA, ZNHIT3, MRM1
     — associated with schizophrenia, autism spectrum disorder (ASD/17q12 deletion
     syndrome). Neurodevelopmental features reported in some deletion carriers.

8. GOUT / HYPERURICAEMIA:
     Renal tubular dysfunction impairs urate secretion → gout in some patients.

Key Clinical Hallmarks
-----------------------
* Renal cysts / structural kidney abnormalities (often pre-diabetes)
* Progressive diabetes onset late teens – early 30s (mean ~26 yr)
* Progressive beta-cell failure → insulin required (often from diagnosis)
* Low/falling C-peptide (distinguishes from other MODY where C-pep preserved)
* Exocrine pancreatic insufficiency (steatorrhoea, fat-soluble vitamin deficiency)
* Hypomagnesaemia (serum Mg < 0.7 mmol/L)
* Elevated liver enzymes (cholestasis in some)
* Autoantibodies NEGATIVE (GADA, ZnT8, IA-2) — key T1D differentiator
* Family history ~50% (vs 75–90% for other MODY) — ~50% are de novo
* Misdiagnosis: T1D (young, needs insulin, falling C-pep, antibody-negative causes confusion),
               T2D, polycystic kidney disease (renal cysts misread as PCKD)

Diagnostic Strategy
--------------------
* Suspect MODY5 when: young diabetes + renal cysts (any age, antenatal or postnatal)
* Pancreatic atrophy on CT/MRI = near-pathognomonic when combined with young diabetes
* Serum Mg (low in ~40-50%) + urinary Mg/creatinine ratio
* Renal function / proteinuria (progressive CKD in some)
* Pelvic USS/MRI in females (Müllerian anomalies)
* Genetics: HNF1B sequencing + copy-number variant analysis (MLPA / aCGH) for deletions
* Pancreatic enzyme supplementation if exocrine insufficiency confirmed (faecal elastase)

Treatment
----------
* INSULIN: Required in most patients; sulfonylure NOT effective (pancreatic atrophy)
* Renal monitoring: eGFR, proteinuria, BP (CKD progression risk)
* Magnesium supplementation: oral Mg (oxide/citrate/glycinate) — target serum Mg ≥ 0.7 mmol/L
* Exocrine insufficiency: Creon (pancrelipase) — fat-soluble vitamins A/D/E/K
* Allopurinol/febuxostat for gout if urate elevated
* Renal transplant in end-stage RCAD-related CKD
* Pregnancy: Insulin continued; renal function closely monitored; Müllerian anomalies
             affect pregnancy management; fetal antenatal USS for renal cysts

Comparison: MODY5 vs MODY3 (HNF1A)
--------------------------------------
Feature              | MODY5 (HNF1B)         | MODY3 (HNF1A)
---------------------|-----------------------|----------------------
Gene                 | HNF1B 17q12           | HNF1A 12q24
Renal glycosuria     | ABSENT (0%)           | PRESENT (50%)
Renal cysts          | PRESENT (~70%)        | ABSENT
Pancreatic atrophy   | PRESENT (CT/MRI)      | ABSENT
Exocrine insuffic.   | PRESENT (40%)         | ABSENT
Sulfonylure resp.    | NO (atrophy)          | YES (85–90%)
De novo mutations    | ~50%                  | Rare (<5%)
Family history       | ~50%                  | ~90%
C-peptide            | Low / falling         | Preserved early
Hypomagnesaemia      | PRESENT (~40–50%)     | ABSENT
Genital malform.     | ~25% females          | ABSENT

Cohort: 40 patients, seed=309.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 309
_COHORT_SIZE = 40

# HNF1B mutations — ~50-60% whole-gene deletions, rest point mutations
_VARIANTS = [
    "17q12_whole_gene_del",  # most common — copy-number deletion
    "R276X",                  # nonsense
    "P159fsdelT",             # frameshift
    "R177X",                  # nonsense
    "IVS2+1G>T",              # splice-site
    "L200P",                  # missense
    "S148W",                  # missense
    "IVS4-2A>G",              # splice-site
    "Other_missense",
    "Other_frameshift",
    "Other_CNV_17q12",
]
_VARIANT_WEIGHTS = [0.35, 0.08, 0.07, 0.07, 0.07, 0.06, 0.05, 0.05, 0.08, 0.07, 0.05]

# Treatment: insulin dominant; sulfonylure rarely tried (no response)
_TREATMENTS = ["Insulin (basal-bolus)", "Insulin (basal-only)", "Insulin + Creon", "Diet/observation", "Sulfonylurea (no response)"]
_TREATMENT_WEIGHTS = [0.40, 0.22, 0.20, 0.08, 0.10]

_MISDIAGNOSES = ["T1D", "T2D", "Polycystic_kidney_disease", "None"]
_MISDIAGNOSIS_WEIGHTS = [0.40, 0.20, 0.15, 0.25]

_RENAL_PHENOTYPES = [
    "Multicystic_dysplastic_kidney",
    "Renal_cysts_bilateral",
    "Hypoplastic_kidney",
    "Oligomeganephronia",
    "Normal_kidneys",
    "Mild_tubular_dysfunction",
]
_RENAL_WEIGHTS = [0.18, 0.30, 0.12, 0.08, 0.18, 0.14]

_SEXES = ["M", "F"]

# De novo vs inherited
_ORIGIN = ["Inherited_AD", "De_novo"]
_ORIGIN_WEIGHTS = [0.50, 0.50]


def _make_patient(seed_val: int) -> dict:
    rng = random.Random(seed_val)
    sex = rng.choices(_SEXES, [0.47, 0.53])[0]
    age = rng.randint(20, 55)
    dx_age = rng.randint(14, min(age, 38))
    duration = age - dx_age

    # HbA1c — higher than MODY2/3 (insulin-requiring); less controlled
    hba1c = round(rng.uniform(6.8, 10.5), 1)

    # Fasting glucose
    fg = round(rng.uniform(7.0, 14.0), 1)

    # C-peptide: LOW and falling (pancreatic atrophy)
    c_pep = round(rng.uniform(0.10, 0.55), 2)

    # Serum magnesium: frequently low
    mg = round(rng.uniform(0.45, 0.85), 2)
    hypo_mg = mg < 0.70

    # eGFR: some with CKD
    egfr = rng.randint(35, 110)
    ckd = egfr < 60

    variant = rng.choices(_VARIANTS, _VARIANT_WEIGHTS)[0]
    treatment = rng.choices(_TREATMENTS, _TREATMENT_WEIGHTS)[0]
    misdiagnosis = rng.choices(_MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)[0]
    renal = rng.choices(_RENAL_PHENOTYPES, _RENAL_WEIGHTS)[0]
    origin = rng.choices(_ORIGIN, _ORIGIN_WEIGHTS)[0]

    # Pancreatic atrophy on imaging
    pancreatic_atrophy = rng.random() < 0.70

    # Exocrine insufficiency
    exocrine_insuff = pancreatic_atrophy and rng.random() < 0.55

    # Genital malformation (females only, ~25%)
    genital_malform = (sex == "F") and (rng.random() < 0.25)

    # Antibodies always negative
    gada = False
    znt8 = False
    ia2 = False

    return {
        "patient_id": f"MODY5-{seed_val:04d}",
        "age": age,
        "sex": sex,
        "age_at_diagnosis": dx_age,
        "duration_years": duration,
        "hba1c_percent": hba1c,
        "fasting_glucose_mmol": fg,
        "c_peptide_nmol_L": c_pep,
        "serum_mg_mmol_L": mg,
        "hypomagnesaemia": hypo_mg,
        "egfr_ml_min_1_73m2": egfr,
        "ckd": ckd,
        "variant": variant,
        "mutation_origin": origin,
        "current_treatment": treatment,
        "renal_phenotype": renal,
        "pancreatic_atrophy_on_imaging": pancreatic_atrophy,
        "exocrine_pancreatic_insufficiency": exocrine_insuff,
        "genital_malformation": genital_malform if sex == "F" else None,
        "prior_misdiagnosis": misdiagnosis,
        "gada_positive": gada,
        "znt8_positive": znt8,
        "ia2_positive": ia2,
        "family_history_positive": origin == "Inherited_AD",
    }


def _generate_cohort() -> list:
    rng = random.Random(_SEED)
    seeds = [rng.randint(10000, 99999) for _ in range(_COHORT_SIZE)]
    return [_make_patient(s) for s in seeds]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    patients = _generate_cohort()

    mean_age = statistics.mean(p["age"] for p in patients)
    mean_dx_age = statistics.mean(p["age_at_diagnosis"] for p in patients)
    mean_hba1c = statistics.mean(p["hba1c_percent"] for p in patients)
    mean_fg = statistics.mean(p["fasting_glucose_mmol"] for p in patients)
    mean_cp = statistics.mean(p["c_peptide_nmol_L"] for p in patients)
    mean_mg = statistics.mean(p["serum_mg_mmol_L"] for p in patients)
    mean_egfr = statistics.mean(p["egfr_ml_min_1_73m2"] for p in patients)

    pct_insulin = sum(1 for p in patients if "Insulin" in p["current_treatment"]) / _COHORT_SIZE * 100
    pct_hypo_mg = sum(1 for p in patients if p["hypomagnesaemia"]) / _COHORT_SIZE * 100
    pct_renal_abn = sum(1 for p in patients if p["renal_phenotype"] != "Normal_kidneys") / _COHORT_SIZE * 100
    pct_pancreatic_atr = sum(1 for p in patients if p["pancreatic_atrophy_on_imaging"]) / _COHORT_SIZE * 100
    pct_exocrine_insuff = sum(1 for p in patients if p["exocrine_pancreatic_insufficiency"]) / _COHORT_SIZE * 100
    pct_misdiag = sum(1 for p in patients if p["prior_misdiagnosis"] != "None") / _COHORT_SIZE * 100
    pct_fam_hx = sum(1 for p in patients if p["family_history_positive"]) / _COHORT_SIZE * 100
    pct_ckd = sum(1 for p in patients if p["ckd"]) / _COHORT_SIZE * 100
    pct_de_novo = sum(1 for p in patients if p["mutation_origin"] == "De_novo") / _COHORT_SIZE * 100
    females = [p for p in patients if p["sex"] == "F"]
    pct_genital_malform = sum(1 for p in females if p["genital_malformation"]) / max(len(females), 1) * 100

    return {
        "kpis": {
            "cohort_size": _COHORT_SIZE,
            "mean_age_years": round(mean_age, 1),
            "mean_age_at_diagnosis_years": round(mean_dx_age, 1),
            "mean_hba1c_percent": round(mean_hba1c, 1),
            "mean_fasting_glucose_mmol": round(mean_fg, 1),
            "mean_c_peptide_nmol_L": round(mean_cp, 3),
            "mean_serum_mg_mmol_L": round(mean_mg, 2),
            "mean_egfr": round(mean_egfr, 0),
            "pct_insulin_treated": round(pct_insulin, 1),
            "pct_hypomagnesaemia": round(pct_hypo_mg, 1),
            "pct_renal_abnormality": round(pct_renal_abn, 1),
            "pct_pancreatic_atrophy": round(pct_pancreatic_atr, 1),
            "pct_exocrine_insufficiency": round(pct_exocrine_insuff, 1),
            "pct_prior_misdiagnosis": round(pct_misdiag, 1),
            "pct_family_hx_positive": round(pct_fam_hx, 1),
            "pct_ckd": round(pct_ckd, 1),
            "pct_de_novo": round(pct_de_novo, 1),
            "pct_genital_malform_females": round(pct_genital_malform, 1),
        },
        "patients": patients,
        "key_facts": [
            "MODY5-UNIQUE: Renal cysts / structural kidney abnormalities PRECEDE diabetes — often detected antenatally",
            "Pancreatic atrophy on CT/MRI (both exocrine + endocrine) — ~70% of patients",
            "NO sulfonylurea response — insulin required (pancreatic atrophy, not secretory dysfunction)",
            "Hypomagnesaemia (renal Mg wasting via FXYD2) — serum Mg < 0.70 mmol/L in ~40–50%",
            "~50% de-novo mutations (whole-gene 17q12 deletion most common) — family history only ~50%",
            "Low / falling C-peptide — distinguishes from MODY2 (normal C-pep) and early MODY1/3",
            "Exocrine pancreatic insufficiency → steatorrhoea → fat-soluble vitamin deficiency (A/D/E/K)",
            "Female genital tract malformations (Müllerian anomalies) in ~25% of affected females",
            "Autoantibodies NEGATIVE (GADA, ZnT8, IA-2) — key T1D differentiator; but antibody-neg + insulin → easy T1D misdiagnosis",
            "17q12 microdeletion syndrome: large deletions encompass LHX1 → ASD/schizophrenia phenotype in some",
            "Renal glycosuria ABSENT (HNF1B does NOT regulate SGLT2) — vs MODY3 (50% renal glycosuria)",
            "Progressive CKD risk — eGFR monitoring essential; end-stage RCAD may require renal transplant",
            "Gout/hyperuricaemia in subset (urate retention via tubular dysfunction)",
            "Genetics: MLPA / aCGH mandatory alongside sequencing — point mutations miss ~55% of HNF1B pathogenic variants",
        ],
        "diagnostic_criteria": {
            "Required": "Young-onset diabetes + renal structural abnormality (cysts / hypoplasia / CAKUT) at any age",
            "Supportive — imaging": "Pancreatic atrophy on CT/MRI (reduced volume ≥ 40% vs age-matched reference)",
            "Supportive — labs": "Serum Mg < 0.70 mmol/L; eGFR ↓; elevated LFTs in some; falling C-peptide",
            "Supportive — clinical": "Female Müllerian anomaly; gout in young adult; exocrine insufficiency",
            "Antibodies": "GADA / ZnT8 / IA-2 NEGATIVE — positive result argues against MODY5",
            "Genetics": "HNF1B sequencing + MLPA/aCGH for 17q12 deletion — pathogenic HNF1B variant confirms",
            "Exclusion": "C-peptide > 0.6 nmol/L with stable HbA1c argues against (expect low C-pep in MODY5)",
            "Family history caveat": "~50% de novo — negative family history does NOT exclude MODY5",
        },
    }


def get_breakdown() -> dict:
    patients = _generate_cohort()

    # Variant distribution
    var_dist: dict = {}
    for p in patients:
        var_dist[p["variant"]] = var_dist.get(p["variant"], 0) + 1

    # HbA1c tiers (higher than other MODY — insulin-requiring)
    hba1c_tiers = {"<7.5%": 0, "7.5–8.9%": 0, "9.0–10.4%": 0, "≥10.5%": 0}
    for p in patients:
        h = p["hba1c_percent"]
        if h < 7.5:
            hba1c_tiers["<7.5%"] += 1
        elif h < 9.0:
            hba1c_tiers["7.5–8.9%"] += 1
        elif h < 10.5:
            hba1c_tiers["9.0–10.4%"] += 1
        else:
            hba1c_tiers["≥10.5%"] += 1

    # Renal phenotype distribution
    renal_dist: dict = {}
    for p in patients:
        renal_dist[p["renal_phenotype"]] = renal_dist.get(p["renal_phenotype"], 0) + 1

    # eGFR tiers (CKD staging)
    egfr_tiers = {"≥90 (G1)": 0, "60–89 (G2)": 0, "30–59 (G3)": 0, "<30 (G4-5)": 0}
    for p in patients:
        e = p["egfr_ml_min_1_73m2"]
        if e >= 90:
            egfr_tiers["≥90 (G1)"] += 1
        elif e >= 60:
            egfr_tiers["60–89 (G2)"] += 1
        elif e >= 30:
            egfr_tiers["30–59 (G3)"] += 1
        else:
            egfr_tiers["<30 (G4-5)"] += 1

    # Treatment distribution
    tx_dist: dict = {}
    for p in patients:
        tx_dist[p["current_treatment"]] = tx_dist.get(p["current_treatment"], 0) + 1

    # Misdiagnosis distribution
    mis_dist: dict = {}
    for p in patients:
        mis_dist[p["prior_misdiagnosis"]] = mis_dist.get(p["prior_misdiagnosis"], 0) + 1

    # Age groups
    age_groups = {"14–19": 0, "20–29": 0, "30–39": 0, "40–49": 0, "50+": 0}
    for p in patients:
        a = p["age"]
        if a < 20:
            age_groups["14–19"] += 1
        elif a < 30:
            age_groups["20–29"] += 1
        elif a < 40:
            age_groups["30–39"] += 1
        elif a < 50:
            age_groups["40–49"] += 1
        else:
            age_groups["50+"] += 1

    # Serum Mg tiers
    mg_tiers = {"Severe <0.55": 0, "Low 0.55–0.69": 0, "Normal ≥0.70": 0}
    for p in patients:
        m = p["serum_mg_mmol_L"]
        if m < 0.55:
            mg_tiers["Severe <0.55"] += 1
        elif m < 0.70:
            mg_tiers["Low 0.55–0.69"] += 1
        else:
            mg_tiers["Normal ≥0.70"] += 1

    # Origin distribution
    origin_dist: dict = {}
    for p in patients:
        origin_dist[p["mutation_origin"]] = origin_dist.get(p["mutation_origin"], 0) + 1

    # C-peptide tiers
    cp_tiers = {"<0.20 (very low)": 0, "0.20–0.39 (low)": 0, "≥0.40 (borderline)": 0}
    for p in patients:
        c = p["c_peptide_nmol_L"]
        if c < 0.20:
            cp_tiers["<0.20 (very low)"] += 1
        elif c < 0.40:
            cp_tiers["0.20–0.39 (low)"] += 1
        else:
            cp_tiers["≥0.40 (borderline)"] += 1

    return {
        "variant_distribution": var_dist,
        "hba1c_tiers": hba1c_tiers,
        "renal_phenotype_distribution": renal_dist,
        "egfr_tiers": egfr_tiers,
        "treatment_distribution": tx_dist,
        "misdiagnosis_distribution": mis_dist,
        "age_groups": age_groups,
        "serum_mg_tiers": mg_tiers,
        "mutation_origin_distribution": origin_dist,
        "c_peptide_tiers": cp_tiers,
        "summary_flags": {
            "pct_renal_abnormality": round(
                sum(1 for p in patients if p["renal_phenotype"] != "Normal_kidneys") / _COHORT_SIZE * 100, 1),
            "pct_pancreatic_atrophy": round(
                sum(1 for p in patients if p["pancreatic_atrophy_on_imaging"]) / _COHORT_SIZE * 100, 1),
            "pct_exocrine_insufficiency": round(
                sum(1 for p in patients if p["exocrine_pancreatic_insufficiency"]) / _COHORT_SIZE * 100, 1),
            "pct_ckd": round(sum(1 for p in patients if p["ckd"]) / _COHORT_SIZE * 100, 1),
            "pct_hypomagnesaemia": round(
                sum(1 for p in patients if p["hypomagnesaemia"]) / _COHORT_SIZE * 100, 1),
            "pct_de_novo": round(
                sum(1 for p in patients if p["mutation_origin"] == "De_novo") / _COHORT_SIZE * 100, 1),
        },
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "name": "MODY5 — HNF1B-MODY / RCAD Syndrome",
            "full_name": "Maturity-Onset Diabetes of the Young Type 5 / Renal Cysts And Diabetes Syndrome",
            "gene": "HNF1B (Hepatocyte Nuclear Factor 1 Beta)",
            "chromosome": "17q12",
            "omim_gene": "*189907",
            "omim_disease": "#137920",
            "inheritance": "Autosomal Dominant (50% risk per child); ~50% de-novo mutations",
            "prevalence": "~1:10,000–1:50,000; ~5% of all MODY (underdiagnosed)",
            "mechanism": (
                "HNF1B LOF disrupts transcription of genes required for kidney tubulogenesis "
                "(→ cysts), endocrine pancreas development (→ beta-cell loss, progressive DM), "
                "exocrine pancreas (→ atrophy, malabsorption), Müllerian duct (→ genital malformations), "
                "and renal Mg reabsorption via FXYD2 (→ hypomagnesaemia)."
            ),
        },
        "genes_and_proteins": {
            "HNF1B": "Hepatocyte Nuclear Factor 1 Beta — homeodomain transcription factor; 557 aa; "
                     "expressed in kidney tubule, pancreatic ductal/endocrine cells, liver, Müllerian duct",
            "FXYD2": "γ-subunit of Na+/K+-ATPase in distal convoluted tubule — HNF1B target gene; "
                     "LOF → reduced Mg²⁺ reabsorption → hypomagnesaemia",
            "17q12_deletion": (
                "Most common MODY5 mechanism (~50–60% of cases); contiguous deletion spans HNF1B + LHX1 + "
                "others; large deletions also associated with ASD and schizophrenia (17q12 deletion syndrome)"
            ),
        },
        "clinical_terms": {
            "RCAD": "Renal Cysts And Diabetes — historical alias for MODY5; emphasises kidney phenotype",
            "CAKUT": "Congenital Anomalies of the Kidney and Urinary Tract — umbrella term covering MODY5 renal findings",
            "Multicystic_dysplastic_kidney": "Non-functional kidney replaced by cysts — most severe MODY5 renal phenotype",
            "Oligomeganephronia": "Reduced nephron number with compensatory hypertrophy — CKD risk",
            "Pancreatic_atrophy": "Reduced pancreatic volume (CT/MRI) — MODY5-specific, both exocrine + endocrine loss",
            "Exocrine_insufficiency": "Loss of pancreatic enzyme secretion → malabsorption, steatorrhoea, fat-soluble vitamin deficiency",
            "Hypomagnesaemia": "Serum Mg < 0.70 mmol/L — renal Mg wasting via FXYD2 LOF; managed with oral Mg supplementation",
            "Müllerian_anomaly": "Female genital tract malformation (bicornuate/arcuate uterus, vaginal septum/aplasia) — ~25% affected females",
        },
        "lab_thresholds": {
            "serum_Mg_low": "< 0.70 mmol/L (normal 0.70–1.05 mmol/L) — MODY5 renal Mg wasting",
            "c_peptide_low": "< 0.30 nmol/L suggests significant beta-cell loss — supports MODY5 vs MODY2/3",
            "HbA1c_MODY5": "Typically 7.0–11.0% (poorly controlled without insulin); higher than MODY2 (stable 5.6–7.6%)",
            "eGFR_CKD3": "< 60 ml/min/1.73m² = CKD stage 3+ — common in MODY5 with structural renal abnormalities",
        },
        "treatment": {
            "first_line": "INSULIN (basal-bolus or basal-only) — sulfonylure NOT effective in most MODY5",
            "why_no_sulfo": "Pancreatic atrophy = structural beta-cell loss, not functional impairment; SU cannot recruit absent cells",
            "magnesium": "Oral Mg supplementation (oxide/citrate/glycinate) — target serum Mg ≥ 0.70 mmol/L",
            "exocrine": "Pancreatic enzyme replacement (Creon) if faecal elastase-1 low; fat-soluble vitamins A/D/E/K",
            "renal": "ACE inhibitor / ARB for CKD; strict BP control; nephrology review; transplant if end-stage",
            "gout": "Allopurinol or febuxostat if hyperuricaemia / gout",
            "pregnancy": "Insulin continued; renal function closely monitored; fetal USS for structural renal anomalies",
        },
        "genetics_testing": {
            "first_tier": "HNF1B sequencing (Sanger or NGS) for point mutations and small indels",
            "second_tier_mandatory": "MLPA / aCGH / SNP-array for 17q12 copy-number variants — misses ~55% if sequencing only",
            "panels": "MODY NGS panel (HNF1A / HNF4A / GCK / HNF1B / PDX1 / NEUROD1) + CNV testing",
            "cascade_testing": "Offer to first-degree relatives (50% risk if inherited; de-novo cases — offspring at 50% risk)",
        },
        "comparison_mody1_2_3_5": {
            "MODY1 (HNF4A)": "Macrosomia + TNH; no renal glycosuria; sulfonylure 1st line; C-pep preserved",
            "MODY2 (GCK)": "Stable mild HbA1c; no treatment; OGTT clamping; C-pep normal",
            "MODY3 (HNF1A)": "Renal glycosuria 50%; sulfonylure 85–90% effective; no renal cysts",
            "MODY5 (HNF1B)": "Renal cysts precede DM; pancreatic atrophy; insulin required; hypomagnesaemia; de-novo 50%",
        },
    }
