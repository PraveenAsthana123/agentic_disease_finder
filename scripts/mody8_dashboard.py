"""
MODY8 — CEL-MODY / BSSL-MODY (Maturity-Onset Diabetes of the Young Type 8)
=================================================================================
Gene       : CEL (Carboxyl Ester Lipase) — alias BSSL (Bile Salt-Stimulated Lipase),
             MELA (Mammary Gland Esterase A)
Chromosome : 9q34.3
OMIM Gene  : *114840
OMIM Dis.  : #609812  (MODY8)
Inheritance: Autosomal Dominant (heterozygous frameshift in VNTR → MODY8)
Prevalence : ~1–2% of MODY; rare; Norwegian/Scandinavian founder enrichment

Mechanism
---------
CEL encodes Carboxyl Ester Lipase — a pancreatic exocrine enzyme secreted into the
duodenum (bile salt-stimulated) to digest cholesterol esters, fat-soluble vitamins,
and triglycerides. The C-terminal tandem repeat (VNTR) region modulates secretion
and protein stability.

MODY8 UNIQUE PATHOMECHANISM:
1. Single-nucleotide deletion in a VNTR (tandem repeat) in CEL exon 11
   → frameshift → truncated/misfolded C-terminal protein domain
   → misfolded CEL aggregates in ER → toxic to pancreatic acinar cells
   → acinar cell apoptosis → progressive EXOCRINE failure (lipomatosis)
   → replacement of exocrine parenchyma with fat infiltration (pancreatic lipomatosis)
   → secondarily, beta-cells embedded in the damaged parenchyma also lost
   → COMBINED exocrine + endocrine pancreatic failure

2. This mechanism is FUNDAMENTALLY DIFFERENT from all other MODY types:
   - Other MODY types: transcription factor or enzyme haploinsufficiency
     directly impairing GSIS or insulin gene regulation
   - MODY8: misfolded EXOCRINE enzyme → structural parenchymal destruction
     → SECONDARY beta-cell loss (structural, not primary secretory defect)

MODY8-UNIQUE Features (critical differentiators)
-------------------------------------------------
1. EXOCRINE FAILURE PRECEDES (or co-presents with) DIABETES — UNIQUE:
   Only MODY type combining exocrine pancreatic insufficiency (EPI) with
   MODY diabetes. EPI presents as steatorrhoea, fat malabsorption, and
   fat-soluble vitamin (ADEK) deficiency — often the FIRST clinical sign.

2. PANCREATIC LIPOMATOSIS ON IMAGING — PATHOGNOMONIC:
   CT/MRI shows diffuse fat infiltration replacing exocrine parenchyma
   (pancreatic lipomatosis). This is distinct from the renal cysts + pancreatic
   atrophy of MODY5 (HNF1B). MODY8 = lipomatosis (fat replacement).
   MODY5 = atrophy (reduced volume, normal density). No cysts in MODY8.

3. NO SU RESPONSE — STRUCTURAL DAMAGE:
   Beta-cell loss is STRUCTURAL (embedded in destroyed exocrine tissue), not
   functional (no K-ATP or transcriptional deficit). Sulfonylurea cannot
   restore function of structurally absent beta-cells. INSULIN is required.

4. NORWEGIAN FOUNDER VARIANT (p.V698Lfs*5):
   Most known MODY8 families are from Norway/Scandinavia. Single nt deletion
   in VNTR repeat unit 16 → p.Val698Leufs*5 (V698Lfs*5) — the founding
   Johansson et al. 2011 (Nat Genet) variant. Other VNTR deletions have been
   reported in non-Scandinavian families.

5. PROGRESSIVE DUAL PANCREATIC FAILURE:
   Both exocrine and endocrine compartments fail progressively. Older patients
   have lower FEL-1, lower C-peptide, higher fat in stool, and higher insulin
   requirements. The exocrine failure may precede overt diabetes by years.

6. MISDIAGNOSIS SPECTRUM:
   - T1D (~35%): insulin-requiring, antibody check critical (MODY8 antibody-negative)
   - Chronic pancreatitis (~20%): EPI + abdominal history
   - CF-related diabetes / CFRD (~10%): steatorrhoea overlap
   - T2D (~15%): rare (insulin requirement usually prevents)
   - None (~20%): correctly identified (usually family screening)

7. NO RENAL FEATURES (differs from MODY5):
   No renal cysts, no hypomagnesaemia, no Mullerian anomalies, no renal glycosuria.
   Key differentiator from MODY5 (HNF1B) which has renal cysts and Mullerian anomalies.

8. AUTOANTIBODIES ALWAYS NEGATIVE (GADA, ZnT8, IA-2):
   Positive antibodies rule out MODY8 and suggest autoimmune T1D.

9. NOT IN OLDEST MODY PANELS — REQUIRES EXPANDED NGS + VNTR ANALYSIS:
   CEL exon 11 VNTR requires both sequencing AND copy number / repeat analysis
   (routine NGS may miss repeat-region deletions). Specific VNTR-aware testing
   methodology required.

10. TREATMENT — DUAL: ENZYME REPLACEMENT + INSULIN:
    (a) Pancreatic enzyme replacement therapy (PERT): Creon/Pancreaze with meals
    (b) Fat-soluble vitamin supplementation (ADEK)
    (c) Insulin (basal ± bolus) for diabetes — SU contraindicated (no functional beta-cells)

Key Clinical Hallmarks
-----------------------
* Exocrine pancreatic insufficiency (EPI): steatorrhoea + fat-soluble vitamin deficiency
* Pancreatic lipomatosis on CT/MRI (fat replacement — pathognomonic)
* Adult-onset MODY diabetes (typically late 20s–50s, mean ~35–45 yr)
* Strong family history (~80–85% first-degree relative affected)
* Autoantibodies NEGATIVE (GADA, ZnT8, IA-2)
* C-peptide LOW at diagnosis (structural beta-cell loss, not preserved like MODY3/6)
* INSULIN REQUIRED (no SU response — structural, not functional)
* NO renal cysts (vs MODY5), NO hypomagnesaemia (vs MODY5), NO Mullerian anomalies
* Norwegian/Scandinavian family origin is a strong clue for the founder variant
* FEL-1 (faecal elastase-1) low: < 200 µg/g stool (confirms EPI)
* CEL VNTR analysis mandatory alongside NGS sequencing

Diagnostic Strategy
--------------------
* Suspect MODY8: diabetes + exocrine insufficiency + pancreatic lipomatosis on imaging
* Key differentiators: lipomatosis (MODY8) vs atrophy+cysts (MODY5)
* FEL-1 < 200 µg/g stool → confirms EPI
* Antibodies negative → argues strongly against T1D
* C-peptide low → confirms substantial beta-cell loss (vs preserved in MODY3/6)
* Expanded MODY NGS panel with VNTR-aware CEL sequencing (exon 11 VNTR region)
* Family history + Norwegian/Scandinavian ancestry → founder variant screen
* Fat-soluble vitamins (A, D, E, K) to assess malabsorption severity

Cohort: 40 patients, seed=317.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 317
_COHORT_SIZE = 40

# CEL VNTR frameshift variants — all involve single nt deletion in VNTR repeats
_VARIANTS = [
    "p.V698Lfs*5",          # Norwegian founder — VNTR repeat unit 16 (Johansson 2011)
    "p.Q683Sfs*10",          # VNTR repeat unit 15 deletion
    "p.G717Rfs*3",           # VNTR repeat unit 17 deletion
    "p.L701Pfs*8",           # VNTR repeat unit 16 alt position
    "p.T710Kfs*2",           # VNTR repeat unit 17 alt
    "VNTR_del_other",        # other VNTR deletion — non-Scandinavian family
    "p.A694Vfs*14",          # VNTR repeat unit 15 alt
    "Splice_site_intron10",  # rare splice-site affecting VNTR region
    "CNV_CEL_partial_dup",   # partial duplication — alternate mechanism
]
_VARIANT_WEIGHTS = [0.38, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]

# Treatment: INSULIN mandatory (no SU); PERT with meals; vitamin supplementation
_TREATMENTS = [
    "Insulin (basal-bolus) + PERT",
    "Insulin (basal-only) + PERT",
    "Insulin (basal-bolus) + PERT + vitamin ADEK",
    "Insulin (basal-only) + PERT + vitamin ADEK",
    "Insulin (pump) + PERT",
]
_TREATMENT_WEIGHTS = [0.30, 0.25, 0.22, 0.15, 0.08]

_MISDIAGNOSES = ["T1D", "Chronic pancreatitis", "CF-related diabetes", "T2D", "None"]
_MISDIAGNOSIS_WEIGHTS = [0.35, 0.20, 0.10, 0.15, 0.20]

_SEXES = ["M", "F"]


def _make_patient(seed_val: int) -> dict:
    rng = random.Random(seed_val)
    sex = rng.choices(_SEXES, [0.50, 0.50])[0]
    age = rng.randint(25, 72)
    # MODY8 onset: late 20s to 50s (exocrine failure may precede by years)
    dx_age = rng.randint(20, min(age, 58))
    duration = age - dx_age

    # HbA1c: moderate-high (insulin-treated; structural loss)
    hba1c = round(rng.uniform(6.8, 11.5), 1)

    # Fasting glucose: moderate-elevated
    fg = round(rng.uniform(6.2, 14.5), 1)

    # C-peptide: LOW at diagnosis (structural beta-cell destruction; unlike MODY3/6)
    baseline_cp = round(rng.uniform(0.05, 0.55), 2)
    # Falls further with duration
    duration_penalty = min(duration * 0.015, 0.30)
    c_pep = max(round(baseline_cp - duration_penalty, 2), 0.02)

    variant = rng.choices(_VARIANTS, _VARIANT_WEIGHTS)[0]
    treatment = rng.choices(_TREATMENTS, _TREATMENT_WEIGHTS)[0]
    misdiagnosis = rng.choices(_MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)[0]

    # Autoantibodies always negative
    gada = False
    znt8 = False
    ia2 = False

    # Family history: ~82%
    fam_hx = rng.random() < 0.82

    # Exocrine markers
    # FEL-1 (faecal elastase-1): normal > 200 µg/g; EPI < 200; severe EPI < 100
    fel1 = round(rng.uniform(12, 185), 0)  # all have EPI by definition

    # Fat-soluble vitamin deficiency: vitamin D most common
    vit_d_def = rng.random() < 0.70
    vit_a_def = rng.random() < 0.45
    vit_k_def = rng.random() < 0.35
    vit_e_def = rng.random() < 0.30

    # Pancreatic fat fraction on MRI (%) — elevated in lipomatosis
    fat_fraction_pct = round(rng.uniform(38, 87), 1)

    # Body weight: often lower due to malabsorption
    bmi = round(rng.uniform(17.5, 27.5), 1)

    # Scandinavian ancestry (Norwegian/Swedish/Danish)
    scandinavian = rng.random() < 0.72

    return {
        "patient_id": f"MODY8-{seed_val:04d}",
        "age": age,
        "sex": sex,
        "age_at_diagnosis": dx_age,
        "duration_years": duration,
        "hba1c_percent": hba1c,
        "fasting_glucose_mmol": fg,
        "c_peptide_nmol_L": c_pep,
        "bmi_kg_m2": bmi,
        "variant": variant,
        "current_treatment": treatment,
        "prior_misdiagnosis": misdiagnosis,
        "family_history_positive": fam_hx,
        "gada_positive": gada,
        "znt8_positive": znt8,
        "ia2_positive": ia2,
        "exocrine_insufficiency": True,          # ALL MODY8 have EPI
        "fel1_ug_g": int(fel1),                  # faecal elastase-1 µg/g stool
        "pancreatic_fat_fraction_pct": fat_fraction_pct,
        "vitamin_d_deficient": vit_d_def,
        "vitamin_a_deficient": vit_a_def,
        "vitamin_k_deficient": vit_k_def,
        "vitamin_e_deficient": vit_e_def,
        "scandinavian_ancestry": scandinavian,
        "on_pert": True,                         # all on PERT (pancreatic enzyme replacement)
        "on_insulin": True,                      # all on insulin (no SU response)
        "su_response": False,                    # SU contraindicated — structural loss
        "renal_cysts": False,
        "pancreatic_atrophy_lipomatosis": True,  # pathognomonic
        "hypomagnesaemia": False,
        "renal_glycosuria": False,
        "mullerian_anomalies": False,
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
    mean_duration = statistics.mean(p["duration_years"] for p in patients)
    mean_bmi = statistics.mean(p["bmi_kg_m2"] for p in patients)
    mean_fel1 = statistics.mean(p["fel1_ug_g"] for p in patients)
    mean_fat_frac = statistics.mean(p["pancreatic_fat_fraction_pct"] for p in patients)

    pct_fam_hx = sum(1 for p in patients if p["family_history_positive"]) / _COHORT_SIZE * 100
    pct_misdiag = sum(1 for p in patients if p["prior_misdiagnosis"] != "None") / _COHORT_SIZE * 100
    pct_scandinavian = sum(1 for p in patients if p["scandinavian_ancestry"]) / _COHORT_SIZE * 100
    pct_vit_d_def = sum(1 for p in patients if p["vitamin_d_deficient"]) / _COHORT_SIZE * 100
    pct_severe_epi = sum(1 for p in patients if p["fel1_ug_g"] < 100) / _COHORT_SIZE * 100

    return {
        "kpis": {
            "cohort_size": _COHORT_SIZE,
            "mean_age_years": round(mean_age, 1),
            "mean_age_at_diagnosis_years": round(mean_dx_age, 1),
            "mean_duration_years": round(mean_duration, 1),
            "mean_hba1c_percent": round(mean_hba1c, 1),
            "mean_fasting_glucose_mmol": round(mean_fg, 1),
            "mean_c_peptide_nmol_L": round(mean_cp, 3),
            "mean_bmi_kg_m2": round(mean_bmi, 1),
            "mean_fel1_ug_g": round(mean_fel1, 0),
            "mean_pancreatic_fat_fraction_pct": round(mean_fat_frac, 1),
            "pct_family_hx_positive": round(pct_fam_hx, 1),
            "pct_prior_misdiagnosis": round(pct_misdiag, 1),
            "pct_scandinavian_ancestry": round(pct_scandinavian, 1),
            "pct_vitamin_d_deficient": round(pct_vit_d_def, 1),
            "pct_severe_epi_fel1_lt100": round(pct_severe_epi, 1),
            "pct_on_insulin": 100.0,
            "pct_on_pert": 100.0,
            "pct_su_response": 0.0,
        },
        "patients": patients,
        "key_facts": [
            "MODY8-CEL/BSSL: Only MODY type caused by a pancreatic exocrine enzyme gene — VNTR frameshift → misfolded CEL → acinar cell death → pancreatic lipomatosis → secondary beta-cell loss",
            "Dual pancreatic failure: exocrine (EPI, steatorrhoea, fat-soluble vitamin deficiency) + endocrine (MODY diabetes) — both progressive, often EPI precedes diabetes by years",
            "Pancreatic lipomatosis on CT/MRI (fat replacement) — pathognomonic; differs from MODY5 atrophy (no cysts, no renal features in MODY8)",
            "NO sulfonylurea response — structural beta-cell destruction; SU cannot restore absent beta-cells; insulin is MANDATORY",
            "Dual treatment: PERT (Creon/Pancreaze) for exocrine + insulin (basal ± bolus) for diabetes + fat-soluble vitamin ADEK supplementation",
            "Norwegian/Scandinavian founder variant p.V698Lfs*5 (single nt deletion in VNTR repeat unit 16) — Johansson et al. 2011 (Nat Genet)",
            "FEL-1 (faecal elastase-1) < 200 µg/g stool confirms EPI; < 100 = severe EPI — mandatory test in all suspected MODY8",
            "Autoantibodies always negative (GADA, ZnT8, IA-2) — distinguishes from T1D despite insulin requirement",
            "C-peptide LOW at diagnosis (structural loss, unlike preserved C-peptide in MODY3/6/7); falls further with duration",
            "NO renal cysts (vs MODY5), NO hypomagnesaemia (vs MODY5), NO Mullerian anomalies (vs MODY5) — key differentiators from HNF1B-MODY",
            "VNTR-aware CEL sequencing required — routine NGS may miss single nt deletion in repeat-rich VNTR region of exon 11",
            "High misdiagnosis: T1D (35%, insulin-requiring), chronic pancreatitis (20%), CFRD (10%) — antibody test + VNTR analysis resolves",
        ],
        "alerts": {
            "dual_failure_alert": "MODY8 is the only MODY type with combined exocrine + endocrine pancreatic failure — always check FEL-1 and pancreatic imaging in unexplained insulin-requiring diabetes + malabsorption",
            "lipomatosis_imaging": "Pancreatic lipomatosis (fat infiltration) on CT/MRI is pathognomonic — different from MODY5 atrophy; order pancreatic protocol imaging in all suspected MODY8",
            "no_su_ever": "Sulfonylurea is CONTRAINDICATED in MODY8 — structural beta-cell loss means no K-ATP functional pool to stimulate; insulin is the only option",
            "vntr_sequencing": "Standard NGS may miss MODY8 — single nt deletion in VNTR-rich CEL exon 11 requires VNTR-aware or long-read sequencing; Southern blot or repeat-specific PCR may be needed",
            "vitamin_monitoring": "Monitor fat-soluble vitamins (A, D, E, K) every 6–12 months — fat malabsorption causes deficiency; supplement routinely while on PERT",
        },
        "mody_registry": {
            "type": "MODY8",
            "gene": "CEL",
            "omim_gene": "*114840",
            "omim_disease": "#609812",
            "chromosome": "9q34.3",
            "inheritance": "Autosomal Dominant",
            "seed": _SEED,
            "cohort_size": _COHORT_SIZE,
        },
    }


def get_breakdown() -> dict:
    patients = _generate_cohort()

    # Variant distribution
    var_dist: dict = {}
    for p in patients:
        v = p["variant"]
        var_dist[v] = var_dist.get(v, 0) + 1

    # HbA1c tiers
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

    # C-peptide tiers — all expected low
    cp_tiers = {"<0.10": 0, "0.10–0.19": 0, "0.20–0.39": 0, "≥0.40": 0}
    for p in patients:
        cp = p["c_peptide_nmol_L"]
        if cp < 0.10:
            cp_tiers["<0.10"] += 1
        elif cp < 0.20:
            cp_tiers["0.10–0.19"] += 1
        elif cp < 0.40:
            cp_tiers["0.20–0.39"] += 1
        else:
            cp_tiers["≥0.40"] += 1

    # Age at diagnosis tiers
    dx_age_tiers = {"<25": 0, "25–34": 0, "35–44": 0, "45–54": 0, "≥55": 0}
    for p in patients:
        d = p["age_at_diagnosis"]
        if d < 25:
            dx_age_tiers["<25"] += 1
        elif d < 35:
            dx_age_tiers["25–34"] += 1
        elif d < 45:
            dx_age_tiers["35–44"] += 1
        elif d < 55:
            dx_age_tiers["45–54"] += 1
        else:
            dx_age_tiers["≥55"] += 1

    # FEL-1 tiers (faecal elastase-1)
    fel1_tiers = {"<100 (severe EPI)": 0, "100–149 (moderate EPI)": 0, "150–199 (mild EPI)": 0}
    for p in patients:
        f = p["fel1_ug_g"]
        if f < 100:
            fel1_tiers["<100 (severe EPI)"] += 1
        elif f < 150:
            fel1_tiers["100–149 (moderate EPI)"] += 1
        else:
            fel1_tiers["150–199 (mild EPI)"] += 1

    # Pancreatic fat fraction tiers
    fat_tiers = {"<50%": 0, "50–64%": 0, "65–74%": 0, "≥75%": 0}
    for p in patients:
        f = p["pancreatic_fat_fraction_pct"]
        if f < 50:
            fat_tiers["<50%"] += 1
        elif f < 65:
            fat_tiers["50–64%"] += 1
        elif f < 75:
            fat_tiers["65–74%"] += 1
        else:
            fat_tiers["≥75%"] += 1

    # Vitamin deficiency counts
    vit_def = {
        "Vitamin D": sum(1 for p in patients if p["vitamin_d_deficient"]),
        "Vitamin A": sum(1 for p in patients if p["vitamin_a_deficient"]),
        "Vitamin K": sum(1 for p in patients if p["vitamin_k_deficient"]),
        "Vitamin E": sum(1 for p in patients if p["vitamin_e_deficient"]),
    }

    # Misdiagnosis distribution
    mis_dist: dict = {}
    for p in patients:
        m = p["prior_misdiagnosis"]
        mis_dist[m] = mis_dist.get(m, 0) + 1

    # Treatment distribution
    tx_dist: dict = {}
    for p in patients:
        t = p["current_treatment"]
        tx_dist[t] = tx_dist.get(t, 0) + 1

    # Disease duration tiers
    dur_tiers = {"<5 yr": 0, "5–9 yr": 0, "10–19 yr": 0, "≥20 yr": 0}
    for p in patients:
        d = p["duration_years"]
        if d < 5:
            dur_tiers["<5 yr"] += 1
        elif d < 10:
            dur_tiers["5–9 yr"] += 1
        elif d < 20:
            dur_tiers["10–19 yr"] += 1
        else:
            dur_tiers["≥20 yr"] += 1

    # BMI tiers
    bmi_tiers = {"<18.5": 0, "18.5–22.9": 0, "23–27.4": 0, "≥27.5": 0}
    for p in patients:
        b = p["bmi_kg_m2"]
        if b < 18.5:
            bmi_tiers["<18.5"] += 1
        elif b < 23:
            bmi_tiers["18.5–22.9"] += 1
        elif b < 27.5:
            bmi_tiers["23–27.4"] += 1
        else:
            bmi_tiers["≥27.5"] += 1

    # Current age groups
    age_groups = {"20–29": 0, "30–39": 0, "40–49": 0, "50–59": 0, "60+": 0}
    for p in patients:
        a = p["age"]
        if a < 30:
            age_groups["20–29"] += 1
        elif a < 40:
            age_groups["30–39"] += 1
        elif a < 50:
            age_groups["40–49"] += 1
        elif a < 60:
            age_groups["50–59"] += 1
        else:
            age_groups["60+"] += 1

    return {
        "variant_distribution": var_dist,
        "hba1c_tiers": hba1c_tiers,
        "c_peptide_tiers": cp_tiers,
        "age_at_diagnosis_tiers": dx_age_tiers,
        "fel1_tiers": fel1_tiers,
        "pancreatic_fat_fraction_tiers": fat_tiers,
        "vitamin_deficiency_counts": vit_def,
        "misdiagnosis_distribution": mis_dist,
        "treatment_distribution": tx_dist,
        "disease_duration_tiers": dur_tiers,
        "bmi_tiers": bmi_tiers,
        "age_groups_current": age_groups,
        "summary_flags": {
            "pct_severe_epi_fel1_lt100": round(
                sum(1 for p in patients if p["fel1_ug_g"] < 100) / _COHORT_SIZE * 100, 1),
            "pct_vitamin_d_deficient": round(
                sum(1 for p in patients if p["vitamin_d_deficient"]) / _COHORT_SIZE * 100, 1),
            "pct_family_hx": round(
                sum(1 for p in patients if p["family_history_positive"]) / _COHORT_SIZE * 100, 1),
            "pct_misdiagnosed": round(
                sum(1 for p in patients if p["prior_misdiagnosis"] != "None") / _COHORT_SIZE * 100, 1),
            "pct_scandinavian": round(
                sum(1 for p in patients if p["scandinavian_ancestry"]) / _COHORT_SIZE * 100, 1),
            "pct_on_insulin": 100.0,
            "pct_su_response": 0.0,
            "pct_high_fat_fraction_gte75": round(
                sum(1 for p in patients if p["pancreatic_fat_fraction_pct"] >= 75) / _COHORT_SIZE * 100, 1),
        },
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "name": "MODY8 — CEL-MODY / BSSL-MODY",
            "full_name": "Maturity-Onset Diabetes of the Young Type 8",
            "gene": "CEL (Carboxyl Ester Lipase) — alias BSSL (Bile Salt-Stimulated Lipase), MELA (Mammary Gland Esterase A)",
            "chromosome": "9q34.3",
            "omim_gene": "*114840",
            "omim_disease": "#609812",
            "inheritance": "Autosomal Dominant (heterozygous VNTR frameshift in CEL exon 11)",
            "prevalence": "~1–2% of all MODY; rare; Norwegian/Scandinavian founder enrichment (p.V698Lfs*5)",
            "mechanism": (
                "Single-nucleotide deletion in CEL exon 11 VNTR (tandem repeat) → frameshift → misfolded/truncated "
                "C-terminal CEL protein → ER aggregation → toxic to pancreatic acinar cells → progressive exocrine "
                "failure (pancreatic lipomatosis) → secondary beta-cell destruction. "
                "Only MODY type caused by an exocrine pancreatic enzyme gene."
            ),
            "key_uniqueness": "Combined exocrine + endocrine pancreatic failure; pancreatic lipomatosis on imaging; NO SU response; insulin mandatory; only MODY with exocrine enzyme gene aetiology",
        },
        "genes_and_proteins": {
            "CEL/BSSL": (
                "Carboxyl Ester Lipase (CEL) / Bile Salt-Stimulated Lipase (BSSL) — "
                "pancreatic exocrine enzyme secreted into the duodenum to hydrolyze cholesterol esters, "
                "triglycerides, and fat-soluble vitamins (ADEK). Also secreted in breast milk. "
                "Gene: chromosome 9q34.3; *114840; 722 aa (with full VNTR repeats). "
                "C-terminal VNTR region (exon 11): 11-aa tandem repeats — mediates secretion and stability. "
                "Expressed exclusively in: pancreatic acini and lactating mammary gland."
            ),
            "VNTR mechanism": (
                "CEL exon 11 contains 11-bp tandem repeat units (VNTR). Single nt deletion within a repeat unit "
                "→ frameshift → truncated C-terminal tail → misfolded protein → ER aggregation → acinar cell "
                "ER stress + apoptosis → progressive exocrine parenchymal loss → fat infiltration (lipomatosis). "
                "Misfolded CEL also diffuses into islet micro-environment → secondary beta-cell toxicity. "
                "The VNTR region is repeat-rich → standard Sanger/NGS may miss the deletion without VNTR-aware PCR."
            ),
            "Pancreatic lipomatosis pathology": (
                "Progressive replacement of exocrine parenchyma with adipocytes (fat cells) — 'pancreatic lipomatosis'. "
                "Visible on CT as diffuse fat density throughout pancreas; on MRI as high T1 signal. "
                "Distinct from MODY5 pancreatic atrophy (volume loss, normal density, no fat). "
                "Pancreatic lipomatosis narrows ductal architecture, reduces enzyme secretion, and eventually "
                "encroaches on islets (beta-cells embedded in fat-replaced parenchyma)."
            ),
        },
        "clinical_terms": {
            "EPI (Exocrine Pancreatic Insufficiency)": "Inadequate pancreatic enzyme secretion → fat malabsorption (steatorrhoea) + fat-soluble vitamin (ADEK) deficiency; confirmed by FEL-1 < 200 µg/g stool",
            "FEL-1 (Faecal Elastase-1)": "Non-invasive stool test; measures pancreatic exocrine output; normal > 200 µg/g; mild EPI 100–200; severe EPI < 100; not affected by PERT timing",
            "PERT (Pancreatic Enzyme Replacement Therapy)": "Creon/Pancreaze with every meal — lipase + amylase + protease capsules; dose titrated to stool consistency and weight; taken with first bite of meal",
            "Pancreatic lipomatosis": "Diffuse fat infiltration (adipose replacement) of exocrine pancreas; pathognomonic for MODY8 on CT/MRI; progressive over decades; different from atrophy (MODY5)",
            "VNTR (Variable Number Tandem Repeat)": "Repetitive DNA sequence in CEL exon 11 (11-bp tandem repeat units); deletion of one nucleotide within a repeat unit causes frameshift → MODY8",
            "p.V698Lfs*5": "Val698Leufs*5 — Norwegian founder MODY8 variant; single nt deletion in VNTR repeat unit 16; causes misfolded C-terminal tail; described by Johansson et al. 2011 (Nat Genet)",
            "CEL-HYB (hybrid gene)": "CEL has a closely related paralogue CELP (pseudogene on chr 9q34); meiotic recombination can create CEL-CELP hybrid genes — some hybrids also cause MODY-like phenotype (controversial)",
            "CFRD (CF-Related Diabetes)": "Diabetes caused by progressive exocrine pancreatic destruction in cystic fibrosis; mimics MODY8 (EPI + insulin-requiring DM); differentiated by CFTR mutation analysis",
        },
        "lab_thresholds": {
            "FEL_1_normal": "> 200 µg/g stool — normal exocrine function",
            "FEL_1_mild_EPI": "100–199 µg/g stool — mild to moderate EPI (PERT indicated)",
            "FEL_1_severe_EPI": "< 100 µg/g stool — severe EPI (high-dose PERT essential)",
            "c_peptide_low_mody8": "< 0.40 nmol/L at diagnosis expected — structural beta-cell destruction (unlike MODY3/6/7 where C-peptide is preserved)",
            "HbA1c_mody8": "Elevated 6.8–11.5%; less stable than MODY7 due to complete absence of SU-responsive pool; insulin titration primary management",
            "pancreatic_fat_fraction": "> 40% fat on MRI pancreatic fat fraction quantification suggests lipomatosis; normal < 15%",
            "antibodies_negative": "GADA / ZnT8 / IA-2 all NEGATIVE — positive antibodies argue strongly against MODY8 and suggest T1D",
        },
        "treatment": {
            "insulin_mandatory": "INSULIN REQUIRED in all MODY8 patients — basal ± bolus; structural beta-cell loss means no SU-responsive pool",
            "su_contraindicated": "SULFONYLUREA CONTRAINDICATED — no functional beta-cells to stimulate; do not use in MODY8",
            "PERT": "Pancreatic enzyme replacement therapy (Creon/Pancreaze) with EVERY MEAL — lipase dose 25,000–50,000 IU per meal; titrate to stool consistency and weight",
            "vitamin_ADEK": "Fat-soluble vitamins A, D, E, K supplementation — mandatory due to fat malabsorption; monitor 25-OH vitamin D and INR (vitamin K proxy) every 6–12 months",
            "diet": "Low-fat diet reduces symptom burden; small frequent meals; snacks with enzyme replacement",
            "pregnancy": "Insulin + PERT continued; fetal monitoring; higher risk of malabsorption-related vitamin deficiency affecting fetal development (especially vitamin K for coagulation)",
            "cascade_testing": "First-degree relatives must undergo CEL VNTR testing — earlier detection enables PERT before overt EPI and insulin before significant hyperglycaemia",
            "monitoring": "Annual: HbA1c, C-peptide, FEL-1, fat-soluble vitamins (A,D,E,K), pancreatic imaging (assess progression of lipomatosis), body weight/BMI",
        },
        "genetics_testing": {
            "critical_vntr_note": "Standard NGS often MISSES the MODY8 VNTR deletion — single nt deletion in repetitive 11-bp tandem repeat region; requires VNTR-aware PCR, long-read sequencing (Oxford Nanopore), or Southern blot",
            "founder_screen": "Norwegian/Scandinavian ancestry: screen specifically for p.V698Lfs*5 (single nt deletion in repeat unit 16) as first-tier test",
            "mody_panel_inclusion": "CEL must be in expanded MODY panels alongside HNF1A, HNF4A, GCK, HNF1B, PDX1, NEUROD1, KLF11, PAX4, INS, BLK, ABCC8, KCNJ11",
            "cel_hyb_caveat": "CEL-CELP pseudogene homology can confound sequencing — ensure alignment tools distinguish CEL from CELP; duplicate read mapping artifacts can create false positive/negative calls",
            "cascade_testing": "All first-degree relatives should be tested when a proband is identified — penetrance approaches 100% in affected families",
        },
        "comparison_mody7_8_9": {
            "MODY7 (KLF11)": "Zinc finger repressor; MAO-A/ROS oxidative mechanism; SU ~75–80% response; no exocrine disease; oxidative beta-cell loss; Q62R+A347S founders; contested",
            "MODY8 (CEL)": "Exocrine enzyme gene; VNTR frameshift → lipomatosis → structural exocrine+endocrine failure; NO SU (structural); insulin mandatory; EPI+PERT; Norwegian founder p.V698Lfs*5",
            "MODY9 (PAX4)": "Paired box transcription factor; transcriptional repressor of glucagon; beta-cell differentiation defect; SU responsive; Asiatic founder R192H; no exocrine disease",
        },
    }
