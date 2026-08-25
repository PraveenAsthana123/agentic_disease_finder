"""
MODY10 — INS-MODY (Maturity-Onset Diabetes of the Young Type 10)
================================================================================
Gene       : INS (Insulin; Preproinsulin → Proinsulin → Insulin + C-peptide)
Chromosome : 11p15.5
OMIM Gene  : *176730
OMIM Dis.  : #613370  (MODY10)
Inheritance: Autosomal Dominant (heterozygous dominant-negative LOF → MODY10)
Prevalence : ~1% of MODY; possibly underdiagnosed; no major ethnic enrichment

Mechanism
---------
INS encodes preproinsulin (110 aa), which is processed to proinsulin and then
cleaved to yield insulin + C-peptide. Heterozygous mutations in the coding sequence
(especially in the disulfide-bonded A and B chains) produce MISFOLDED PROINSULIN
that cannot exit the ER normally.

MODY10 PATHOMECHANISM (Dominant-Negative ER-Stress Model):
1. Heterozygous INS missense mutation → abnormal cysteine pairing or hydrophobic core
   disruption → misfolded mutant proinsulin accumulates in ER
2. Misfolded proinsulin overwhelms ER protein folding quality control (BiP/GRP78, ERAD)
   → unfolded protein response (UPR): PERK, IRE1, ATF6 arms activated
3. Chronic UPR → oxidative stress → mitochondrial dysfunction → progressive beta-cell
   apoptosis (dominant-negative: the mutant copy actively damages the cell above what
   simple haploinsufficiency would cause)
4. Beta-cell mass falls progressively → C-peptide declines → insulin dependency
5. Unlike MODY9 (transcriptional, functional) or MODY2 (sensing, stable) — MODY10
   is a STRUCTURAL PROTEIN-QUALITY-CONTROL DISEASE

Key Founding Mutations
----------------------
* R46Q (c.136G>A) — A-chain; most common dominant-negative; abolishes B-chain/A-chain
  disulfide bridge → proinsulin misfolding; Molven 2008 Nat Genet (Norwegian family)
* R89C (c.265C>T) — B-chain; introduces free cysteine → aberrant disulfide; strong ER stress
* C96Y (c.287G>A) — disulfide loop; Cys96 normally forms a disulfide with Cys11(A-chain);
  Y substitution → no disulfide → severe misfolding → highest UPR activation
* H29D (c.85C>G) — signal peptide / A-chain boundary; processing defect
* L68M (c.202C>A) — B-chain hydrophobic core; Stoy 2007 Nat Genet (initial discovery paper)
* Y108C (c.323A>G) — C-peptide/A-chain junction; Stoy cohort

Distinguishing MODY10 from PNDM-INS
-------------------------------------
* PNDM-INS (de novo or homozygous/compound het): neonatal onset (<6 months), severe
  DKA, very low C-peptide; insulin mandatory from birth; not MODY
* MODY10 (heterozygous dominant-negative, familial): onset teens–40s; C-peptide
  preserved early then declines; family history 70–80%; autosomal dominant

Clinical Profile
----------------
* Onset: teens–early 40s (earlier than MODY7/KLF11; overlaps MODY3/HNF1A)
* C-peptide: Preserved early, falls progressively (structural apoptotic loss)
  UNLIKE MODY9 (functional, preserved) or MODY2 (normal/stable)
* HbA1c: Progressive — not stable (unlike MODY2 GCK)
* Treatment: INSULIN required in most (70–80%); SU adds marginal benefit early but does
  not arrest beta-cell loss; structural destruction = SU cannot restore lost beta-cell mass
* Autoantibodies: NEGATIVE (GADA, ZnT8, IA-2) — mandatory test to exclude T1D
* Misdiagnosis T1D: ~40% (misfolded proinsulin triggers suspicion; antibody-negative
  resolves it; C-peptide persistence differentiates from end-stage T1D)
* No exocrine involvement (vs MODY8/CEL)
* No renal cysts or Mullerian anomalies (vs MODY5/HNF1B)
* No renal glycosuria (vs MODY3/HNF1A)
* No macrosomia or TNH (vs MODY1/HNF4A)
* No KPD-remission pattern (vs MODY9/PAX4)
* Family Hx: 70–80% (de novo ~10–15%)

Diagnostic Strategy
-------------------
* Suspect MODY10: young-onset DM, antibody-negative, family history, progressive HbA1c,
  falling C-peptide (structural loss pattern)
* Test: INS gene sequencing; functional assay (ER stress reporter if novel variant)
* Expanded MODY NGS panel must include INS — not in oldest panels
* PNDM-INS overlap: neonatal onset → PNDM, not MODY

Cohort: 40 patients, seed=321.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 321
_COHORT_SIZE = 40

# INS variants — dominant-negative misfolded proinsulin
_VARIANTS = [
    "R46Q (c.136G>A)",         # A-chain; most common D-N; Molven 2008 Nat Genet
    "R89C (c.265C>T)",         # B-chain; free Cys → aberrant disulfide; strong ER stress
    "C96Y (c.287G>A)",         # disulfide loop; highest UPR; Stoy 2007 Nat Genet
    "L68M (c.202C>A)",         # B-chain hydrophobic core; Stoy 2007 discovery cohort
    "H29D (c.85C>G)",          # signal/A-chain boundary; processing defect
    "Y108C (c.323A>G)",        # C-peptide/A-chain junction; Stoy cohort
    "G32S (c.94G>A)",          # B-chain; conserved glycine; structural disruption
    "Novel_missense_INS",      # novel; ER stress assay pending
    "Splice_INS",              # splice site; altered processing; rare
]
_VARIANT_WEIGHTS = [0.30, 0.22, 0.18, 0.10, 0.07, 0.05, 0.04, 0.02, 0.02]

# Treatment: insulin dominant; SU marginal early benefit
_TREATMENTS = [
    "Insulin (basal-bolus)",
    "Insulin (basal only) + metformin",
    "Sulfonylurea (early, low C-peptide)",
    "Sulfonylurea + metformin (early phase only)",
    "Diet only (very early, mild, detected incidentally)",
]
_TREATMENT_WEIGHTS = [0.45, 0.28, 0.12, 0.10, 0.05]

_MISDIAGNOSES = ["T1D", "T2D", "None", "LADA"]
_MISDIAGNOSIS_WEIGHTS = [0.40, 0.22, 0.30, 0.08]

_SEXES = ["M", "F"]

# No major ethnic enrichment for MODY10 — worldwide distribution
_ETHNICITIES = [
    "European", "South Asian", "East Asian",
    "Middle Eastern", "African", "Latin American", "Other",
]
_ETHNICITY_WEIGHTS = [0.42, 0.20, 0.15, 0.09, 0.06, 0.05, 0.03]


def _make_patient(seed_val: int) -> dict:
    rng = random.Random(seed_val)
    sex = rng.choices(_SEXES, [0.50, 0.50])[0]
    age = rng.randint(22, 65)
    # MODY10 onset: teens–early 40s (mean ~26–32 yr); earlier than MODY7
    dx_age = rng.randint(14, min(age, 42))
    duration = age - dx_age

    # HbA1c: progressive rise with duration; moderate-severe range
    base_hba1c = rng.uniform(6.4, 9.8)
    duration_hba1c = min(duration * 0.06, 1.8)
    hba1c = round(base_hba1c + duration_hba1c, 1)

    # Fasting glucose: elevated at diagnosis
    fg = round(rng.uniform(6.2, 14.5), 1)

    # C-peptide: PRESERVED early then FALLS progressively (structural apoptotic loss)
    # Different from MODY9 (preserved) — here C-peptide tracks disease duration
    baseline_cp = round(rng.uniform(0.25, 1.40), 2)
    # Progressive structural loss — steeper decline than MODY9
    duration_penalty = min(duration * 0.020, 0.55)
    c_pep = max(round(baseline_cp - duration_penalty, 2), 0.04)

    variant = rng.choices(_VARIANTS, _VARIANT_WEIGHTS)[0]
    treatment = rng.choices(_TREATMENTS, _TREATMENT_WEIGHTS)[0]
    misdiagnosis = rng.choices(_MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)[0]
    ethnicity = rng.choices(_ETHNICITIES, _ETHNICITY_WEIGHTS)[0]

    # Autoantibodies always negative
    gada = False
    znt8 = False

    # Family history positive ~75%
    family_hx = rng.random() < 0.75

    # BMI: normal (18–30 kg/m²); not obese like T2D
    bmi = round(rng.uniform(18.5, 30.2), 1)

    # C96Y and R89C → strongest ER stress → earliest insulin requirement
    high_er_stress = variant in ("C96Y (c.287G>A)", "R89C (c.265C>T)")
    er_stress_level = "High (ER overload)" if high_er_stress else (
        "Moderate" if variant not in ("Novel_missense_INS", "Splice_INS") else "Unknown"
    )

    # Duration-based disease stage
    if duration <= 3:
        stage = "Early (C-peptide detectable)"
    elif duration <= 8:
        stage = "Intermediate (C-peptide declining)"
    else:
        stage = "Advanced (insulin-dependent)"

    return {
        "patient_id": f"MODY10-{seed_val:04d}",
        "sex": sex,
        "age": age,
        "age_at_diagnosis": dx_age,
        "disease_duration_years": duration,
        "hba1c_percent": hba1c,
        "fasting_glucose_mmol_L": fg,
        "c_peptide_nmol_L": c_pep,
        "bmi_kg_m2": bmi,
        "variant": variant,
        "treatment": treatment,
        "prior_misdiagnosis": misdiagnosis,
        "ethnicity": ethnicity,
        "gada_positive": gada,
        "znt8_positive": znt8,
        "family_history_positive": family_hx,
        "er_stress_level": er_stress_level,
        "disease_stage": stage,
    }


def _make_cohort() -> list:
    return [_make_patient(_SEED + i) for i in range(_COHORT_SIZE)]


# ---------------------------------------------------------------------------
# Endpoint helpers — called by api_backend.py
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    cohort = _make_cohort()

    ages = [p["age"] for p in cohort]
    dx_ages = [p["age_at_diagnosis"] for p in cohort]
    durations = [p["disease_duration_years"] for p in cohort]
    hba1cs = [p["hba1c_percent"] for p in cohort]
    c_peps = [p["c_peptide_nmol_L"] for p in cohort]

    kpis = {
        "cohort_size": _COHORT_SIZE,
        "mean_age_years": statistics.mean(ages),
        "mean_age_at_diagnosis_years": statistics.mean(dx_ages),
        "mean_duration_years": statistics.mean(durations),
        "mean_hba1c_percent": statistics.mean(hba1cs),
        "mean_c_peptide_nmol_L": statistics.mean(c_peps),
        "pct_insulin_required": round(
            sum(1 for p in cohort if "Insulin" in p["treatment"]) / _COHORT_SIZE * 100, 1
        ),
        "pct_prior_misdiagnosis": round(
            sum(1 for p in cohort if p["prior_misdiagnosis"] != "None") / _COHORT_SIZE * 100, 1
        ),
        "pct_misdiagnosed_t1d": round(
            sum(1 for p in cohort if p["prior_misdiagnosis"] == "T1D") / _COHORT_SIZE * 100, 1
        ),
        "pct_family_hx_positive": round(
            sum(1 for p in cohort if p["family_history_positive"]) / _COHORT_SIZE * 100, 1
        ),
        "pct_advanced_stage": round(
            sum(1 for p in cohort if "Advanced" in p["disease_stage"]) / _COHORT_SIZE * 100, 1
        ),
        "pct_r46q": round(
            sum(1 for p in cohort if "R46Q" in p["variant"]) / _COHORT_SIZE * 100, 1
        ),
    }

    key_facts = [
        "MODY10 — INS dominant-negative missense → misfolded proinsulin → ER stress → progressive beta-cell apoptosis",
        "Autoantibodies ALWAYS negative (GADA, ZnT8, IA-2) — mandatory test to exclude T1D",
        "C-peptide FALLS progressively — structural apoptotic loss (vs MODY9 functional preservation)",
        "Insulin required in 70–80%; SU cannot arrest structural beta-cell loss",
        "No exocrine insufficiency (vs MODY8), no renal cysts (vs MODY5), no KPD remission (vs MODY9)",
        "R46Q (c.136G>A) — most common dominant-negative; Molven 2008 Nat Genet (Norwegian)",
        "C96Y — strongest ER stress; earliest insulin dependency in cohort",
        "Onset teens–early 40s; family history 70–80%; ~10–15% de novo",
        "Most common misdiagnosis: T1D (~40%); antibody-negative DM + family history → test INS",
        "Expanded MODY NGS panel must include INS — not in oldest 4-gene panels",
        "PNDM-INS overlap: de novo or biallelic → neonatal onset; MODY10 = familial heterozygous AD",
        "ER stress biomarkers (BiP/GRP78) elevated in in-vitro models; not yet routine clinical test",
    ]

    alerts = {
        "do_not_label_T1D": (
            "Young antibody-negative DM + family history + progressive HbA1c + falling C-peptide "
            "→ test INS gene BEFORE labelling T1D. Antibody-negative DKA ≠ T1D."
        ),
        "SU_will_not_reverse_apoptosis": (
            "SU (sulfonylurea) cannot restore lost beta-cell mass in MODY10. Early-phase SU may "
            "improve residual GSIS transiently, but insulin is required as C-peptide declines. "
            "Unlike MODY3/1/6, SU is NOT first-line."
        ),
        "screen_family_mandatory": (
            "Autosomal dominant (50% risk); screen all first-degree relatives with INS sequencing "
            "+ C-peptide + HbA1c. De novo rate ~10–15%: absence of family history does NOT exclude MODY10."
        ),
        "distinguish_PNDM_vs_MODY10": (
            "INS mutations cause BOTH PNDM (neonatal, de novo/biallelic, severe) and MODY10 "
            "(familial, heterozygous AD, teens-40s onset). Onset age and family history separate them."
        ),
    }

    return {"kpis": kpis, "patients": cohort, "key_facts": key_facts, "alerts": alerts}


def get_breakdown() -> dict:
    cohort = _make_cohort()

    # Variant distribution
    from collections import Counter
    var_dist = dict(Counter(p["variant"] for p in cohort))

    # Ethnicity distribution
    eth_dist = dict(Counter(p["ethnicity"] for p in cohort))

    # HbA1c tiers
    hba1c_tiers: dict = {}
    for p in cohort:
        h = p["hba1c_percent"]
        if h < 7.0:
            key = "< 7.0%"
        elif h < 8.0:
            key = "7.0–7.9%"
        elif h < 9.0:
            key = "8.0–8.9%"
        elif h < 10.0:
            key = "9.0–9.9%"
        else:
            key = "≥ 10.0%"
        hba1c_tiers[key] = hba1c_tiers.get(key, 0) + 1

    # C-peptide tiers (falling — structural)
    cp_tiers: dict = {}
    for p in cohort:
        c = p["c_peptide_nmol_L"]
        if c < 0.10:
            key = "< 0.10 (very low)"
        elif c < 0.30:
            key = "0.10–0.29 (low)"
        elif c < 0.60:
            key = "0.30–0.59 (moderate)"
        elif c < 1.00:
            key = "0.60–0.99 (moderate-high)"
        else:
            key = "≥ 1.00 (preserved)"
        cp_tiers[key] = cp_tiers.get(key, 0) + 1

    # Age at diagnosis tiers
    dx_age_tiers: dict = {}
    for p in cohort:
        d = p["age_at_diagnosis"]
        if d < 18:
            key = "< 18 (adolescent)"
        elif d < 25:
            key = "18–24"
        elif d < 35:
            key = "25–34"
        elif d < 45:
            key = "35–44"
        else:
            key = "≥ 45"
        dx_age_tiers[key] = dx_age_tiers.get(key, 0) + 1

    # Disease stage
    stage_dist = dict(Counter(p["disease_stage"] for p in cohort))

    # ER stress level
    er_dist = dict(Counter(p["er_stress_level"] for p in cohort))

    # Treatment distribution
    tx_dist = dict(Counter(p["treatment"] for p in cohort))

    # Misdiagnosis distribution
    mis_dist = dict(Counter(p["prior_misdiagnosis"] for p in cohort))

    # BMI tiers
    bmi_tiers: dict = {}
    for p in cohort:
        b = p["bmi_kg_m2"]
        if b < 20:
            key = "< 20 (underweight)"
        elif b < 25:
            key = "20–24.9"
        elif b < 30:
            key = "25–29.9"
        else:
            key = "≥ 30"
        bmi_tiers[key] = bmi_tiers.get(key, 0) + 1

    # Duration tiers
    dur_tiers: dict = {}
    for p in cohort:
        dur = p["disease_duration_years"]
        if dur <= 3:
            key = "0–3 yr"
        elif dur <= 7:
            key = "4–7 yr"
        elif dur <= 12:
            key = "8–12 yr"
        else:
            key = "> 12 yr"
        dur_tiers[key] = dur_tiers.get(key, 0) + 1

    # Summary flags
    n = _COHORT_SIZE
    summary_flags = {
        "pct_insulin_required": round(
            sum(1 for p in cohort if "Insulin" in p["treatment"]) / n * 100, 1),
        "pct_misdiagnosed_T1D": round(
            sum(1 for p in cohort if p["prior_misdiagnosis"] == "T1D") / n * 100, 1),
        "pct_antibody_negative": 100.0,
        "pct_family_hx_positive": round(
            sum(1 for p in cohort if p["family_history_positive"]) / n * 100, 1),
        "pct_advanced_stage": round(
            sum(1 for p in cohort if "Advanced" in p["disease_stage"]) / n * 100, 1),
        "pct_c_pep_under_030": round(
            sum(1 for p in cohort if p["c_peptide_nmol_L"] < 0.30) / n * 100, 1),
    }

    return {
        "variant_distribution": var_dist,
        "ethnicity_distribution": eth_dist,
        "hba1c_tiers": hba1c_tiers,
        "c_peptide_tiers": cp_tiers,
        "age_at_diagnosis_tiers": dx_age_tiers,
        "disease_stage_distribution": stage_dist,
        "er_stress_distribution": er_dist,
        "treatment_distribution": tx_dist,
        "misdiagnosis_distribution": mis_dist,
        "bmi_tiers": bmi_tiers,
        "disease_duration_tiers": dur_tiers,
        "summary_flags": summary_flags,
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "full_name": "MODY10 — INS-MODY (Maturity-Onset Diabetes of the Young Type 10)",
            "gene": "INS — Insulin; preproinsulin precursor (110 aa); 11p15.5; OMIM *176730",
            "disease_omim": "#613370",
            "inheritance": "Autosomal Dominant — heterozygous dominant-negative missense; 50% transmission",
            "prevalence": "~1% of MODY; no major ethnic enrichment; global distribution",
            "mechanism": (
                "Dominant-negative: heterozygous INS missense → misfolded mutant proinsulin "
                "accumulates in ER → chronic UPR (PERK/IRE1/ATF6) → oxidative stress → "
                "progressive beta-cell apoptosis → falling C-peptide → insulin dependency"
            ),
            "protein_function": (
                "Preproinsulin (110 aa) → signal peptide cleavage → proinsulin (86 aa) → "
                "trypsin-like + carboxypeptidase cleavage → mature insulin (A+B chains, "
                "2 disulfide bonds) + C-peptide (31 aa). Disulfide bonds critical: "
                "A7-B7 and A20-B19 interchain; A6-A11 intrachain."
            ),
            "onset_age": "Teens to early 40s (mean ~26–32 yr); earlier than MODY7; overlaps MODY3",
            "c_peptide_pattern": (
                "Preserved early → falls progressively (structural apoptotic loss). "
                "Unlike MODY9 (functional, preserved). Unlike MODY2 (normal/stable)."
            ),
            "treatment": "Insulin (70–80%); SU marginal early — cannot arrest apoptosis",
            "autoantibodies": "NEGATIVE (GADA, ZnT8, IA-2) — always; test mandatory to exclude T1D",
            "family_history": "70–80% positive (50% AD); de novo ~10–15%",
            "misdiagnosis_rate": "T1D ~40%; LADA ~8%; no screening = lifelong insulin without genetic diagnosis",
        },

        "genes_and_proteins": {
            "INS (Insulin; *176730)": (
                "11p15.5. Preproinsulin: signal peptide (24aa) + B-chain (30aa) + C-peptide (31aa) "
                "+ A-chain (21aa). After ER signal cleavage: proinsulin. Folding requires 3 disulfide "
                "bonds (A6-A11; A7-B7; A20-B19). Mutation disrupts disulfide formation → misfolding."
            ),
            "BiP/GRP78 (HSPA5)": (
                "ER chaperone — the primary sensor of misfolded proinsulin. Normally bound to "
                "PERK/IRE1/ATF6 (keeping them inactive). Misfolded proinsulin sequesters BiP → "
                "releases PERK/IRE1/ATF6 → UPR activated."
            ),
            "PERK (EIF2AK3; MODY8)": (
                "ER kinase arm of UPR. PERK phosphorylates eIF2α → global translation attenuation "
                "(reduces new proinsulin load); also activates ATF4 → CHOP → apoptosis. Note: EIF2AK3 "
                "LOF biallelic = Wolcott-Rallison syndrome (PNDM + skeletal dysplasia)."
            ),
            "IRE1α / XBP1": (
                "IRE1α splices XBP1 mRNA → XBP1s → transcription of ERAD (ER-associated degradation) "
                "genes. ERAD attempts to clear misfolded proinsulin. Chronic IRE1 activation → RIDD "
                "(Regulated IRE1-Dependent Decay) → degrades insulin mRNA — secondary insulin "
                "deficiency on top of apoptotic structural loss."
            ),
        },

        "clinical_terms": {
            "MODY10": "Maturity-Onset Diabetes of the Young Type 10; INS gene; dominant-negative ER stress",
            "Dominant-negative": (
                "Mutant protein actively impairs the function of the wild-type protein or cell; "
                "more pathogenic than simple haploinsufficiency (loss of one copy)."
            ),
            "UPR (Unfolded Protein Response)": (
                "3-arm ER stress response: PERK → eIF2α phosphorylation; IRE1 → XBP1 splicing; "
                "ATF6 → ERAD transcription. Chronic UPR → CHOP-mediated apoptosis."
            ),
            "ERAD": (
                "ER-Associated Degradation. Retrotranslocates misfolded proinsulin to cytoplasm "
                "for proteasomal degradation. Overwhelmed in MODY10."
            ),
            "C-peptide": (
                "31-aa peptide cleaved from proinsulin; equimolar with insulin secretion; "
                "marker of residual beta-cell function. Falling C-peptide = progressive apoptosis."
            ),
            "PNDM-INS": (
                "Permanent Neonatal DM caused by de novo or biallelic INS mutations — severe "
                "misfolding; onset <6 months; requires lifelong insulin. NOT MODY10."
            ),
            "Disulfide bond": (
                "Covalent S-S bond between cysteine residues; critical for proinsulin 3D structure. "
                "Mutations disrupting disulfide bonds (C96Y, R89C) cause severe ER stress."
            ),
        },

        "lab_thresholds": {
            "C-peptide preserved (MODY10 early)": "≥ 0.60 nmol/L (fasting); detectable in stimulation test",
            "C-peptide low (MODY10 advanced)": "< 0.30 nmol/L; insulin dependency imminent",
            "C-peptide very low": "< 0.10 nmol/L; essentially no beta-cell reserve",
            "HbA1c target (MODY10 on insulin)": "< 7.0% (53 mmol/mol); CGM assists titration",
            "Autoantibodies (T1D exclusion)": "GADA < 5 IU/mL; ZnT8-Ab negative; IA-2 negative",
            "Fasting glucose at diagnosis": "Typically 6.2–14.5 mmol/L; progressive with duration",
            "INS NGS panel coverage": "Full exon + splice site coverage; INS exon 2-3 coding region",
        },

        "treatment": {
            "insulin_basal_bolus": (
                "First-line for moderate-to-severe C-peptide loss. Basal (glargine/detemir) "
                "+ rapid-acting (aspart/lispro/glulisine) at meals. Titrate to CGM targets."
            ),
            "sulfonylurea_early_phase": (
                "May provide marginal GSIS augmentation in early MODY10 (C-peptide > 0.30). "
                "Does NOT arrest apoptosis. Switch to insulin as C-peptide falls."
            ),
            "SGLT2_inhibitor_adjunct": (
                "Empagliflozin/dapagliflozin: glycemic + weight benefit; consider in BMI > 25. "
                "Caution: euDKA risk if C-peptide very low."
            ),
            "CGM_strongly_recommended": (
                "CGM (FreeStyle Libre, Dexcom G7): captures post-meal excursions + nocturnal "
                "hypoglycaemia; guides basal/bolus titration as beta-cell reserve declines."
            ),
            "genetic_counselling": (
                "50% AD transmission risk; all first-degree relatives: INS sequencing + HbA1c + "
                "C-peptide. Pre-conception counselling for mutation carriers."
            ),
        },

        "genetics_testing": {
            "INS_sequencing": (
                "Full coding sequence (exons 2–3 + intron-exon boundaries). Note: exon 1 is "
                "untranslated in most isoforms. Look for missense, splice, truncating."
            ),
            "functional_validation": (
                "For novel INS variants: in-vitro ER stress assay (CHOP-luciferase reporter; "
                "BiP induction; proinsulin-mCherry aggregation). Co-segregation in family."
            ),
            "MODY_panel_requirement": (
                "Expanded NGS MODY panel must include INS. Oldest panels (HNF1A/HNF4A/GCK/HNF1B) "
                "miss MODY10. Sanger sequencing INS if high pre-test probability."
            ),
            "PNDM_differentiation": (
                "De novo INS mutations → PNDM (neonatal; < 6 months). Familial heterozygous AD → "
                "MODY10 (teens–40s). Molecular genetics resolves overlap when onset ambiguous."
            ),
            "cascade_screening": (
                "All first-degree relatives of confirmed MODY10. 50% carry mutation. Early "
                "detection = insulin therapy before severe C-peptide loss."
            ),
        },

        "comparison_mody8_9_10": {
            "MODY8 (CEL)": {
                "gene": "CEL; 9q34.3; VNTR exon-11 deletion; misfolded CEL",
                "mechanism": "Misfolded CEL acinar apoptosis → pancreatic lipomatosis → exo+endocrine failure",
                "treatment": "Insulin MANDATORY; PERT + vit-ADEK; SU contraindicated",
                "c_peptide": "LOW at diagnosis (structural loss)",
                "exocrine": "YES — steatorrhoea; FEL1 < 200 µg/g",
                "ethnicity": "Norwegian/Scandinavian enriched",
            },
            "MODY9 (PAX4)": {
                "gene": "PAX4; 7q32.1; transcriptional LOF; ARX de-repression",
                "mechanism": "Haploinsufficiency → alpha-cell bias → functional GSIS impairment",
                "treatment": "SU first-line (75–80%); insulin for KPD until C-peptide recovery",
                "c_peptide": "PRESERVED (functional deficit, not structural); transiently dips in KPD",
                "exocrine": "NO",
                "unique": "KPD: DKA at onset → C-peptide recovery → SU/diet remission in 50–70%",
            },
            "MODY10 (INS)": {
                "gene": "INS; 11p15.5; dominant-negative missense; misfolded proinsulin",
                "mechanism": "Misfolded proinsulin → ER stress (UPR) → progressive beta-cell apoptosis",
                "treatment": "Insulin (70–80%); SU marginal early only; cannot arrest apoptosis",
                "c_peptide": "FALLS progressively (structural apoptotic loss); early = preserved",
                "exocrine": "NO",
                "unique": "Structural ER-stress disease; C-peptide trajectory separates from MODY9",
            },
        },
    }
