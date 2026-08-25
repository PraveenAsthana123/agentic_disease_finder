"""
MODY7 — KLF11-MODY / TIEG2-MODY (Maturity-Onset Diabetes of the Young Type 7)
=================================================================================
Gene       : KLF11 (Krüppel-Like Factor 11) — alias TIEG2 (TGF-β-Inducible Early Gene 2)
Chromosome : 2p25.1
OMIM Gene  : *603301
OMIM Dis.  : #610508  (MODY7)
Inheritance: Autosomal Dominant (heterozygous LOF → MODY7)
Prevalence : ~1–2% of all MODY (extremely rare; historically underdiagnosed; controversial literature)

Mechanism
---------
KLF11 is a Krüppel-like factor (KLF) SP1-type zinc finger transcriptional repressor.
It contains three C-terminal C2H2 zinc finger motifs that bind GT-box/GC-box elements
(CACCC, CCGCCC) in gene promoters. KLF11 acts primarily via:

1. MAO-A repression — direct beta-cell mechanism:
   KLF11 recruits mSin3A co-repressor to repress MAO-A (monoamine oxidase A) promoter.
   LOF → excess MAO-A activity → elevated H₂O₂ (oxidative stress) → beta-cell apoptosis.
   This ROS-mediated mechanism is distinct from all other MODY types (all transcription-factor
   or enzyme haploinsufficiency) — KLF11 uniquely acts via oxidative beta-cell toxicity.

2. SHP repression — HNF axis link:
   KLF11 represses SHP (Small Heterodimer Partner, NR0B2). SHP normally represses HNF4A
   (MODY1) and HNF1A (MODY3) target genes. KLF11 LOF → elevated SHP → dampened HNF4A/HNF1A
   activity → indirect impairment of INS/GCK/GLUT2 expression. This positions MODY7 in the
   same HNF transcription factor regulatory axis as MODY1/MODY3.

3. PDX1 regulation (indirect):
   KLF11 binds SP1 elements in the PDX1 promoter, contributing to PDX1 expression.
   LOF → reduced PDX1 (MODY4) → secondary GSIS impairment.

MODY7-UNIQUE Features (critical differentiators)
-------------------------------------------------
1. OXIDATIVE STRESS MECHANISM — UNIQUE AMONG ALL MODY TYPES:
     Only MODY type with an ROS/oxidative stress mechanism. All other MODY types operate
     via haploinsufficiency of a transcription factor or enzyme. KLF11 LOF → MAO-A excess
     → H₂O₂ → beta-cell apoptosis is a functionally distinct pathomechanism. Research
     implication: antioxidant therapy (N-acetylcysteine) has been proposed (not yet
     standard of care).

2. MODY7 IS THE MOST CONTESTED MODY SUBTYPE:
     Originally described by Neve et al. 2005 (Nat Genet) in two large French families
     with Q62R + A347S founder mutations. Subsequent population studies have questioned
     penetrance and causality — some KLF11 variants appear in controls. Functional data
     (MAO-A + SHP repression assays) support pathogenicity of the original variants.
     Clinical practice: include KLF11 in expanded MODY panels; treat as MODY7 when
     clinical features are consistent and other causes excluded.

3. Q62R — THE FOUNDING MODY7 MUTATION (PCNLS/Sin3A-interaction domain):
     Gln62Arg missense in the N-terminal PCNLS/mSin3A-interaction domain — first
     reported by Neve et al. (2005). Q62R abrogates mSin3A co-repressor recruitment
     → loss of MAO-A repression → excess oxidative stress. A347S (zinc finger motif 1)
     is the second founding mutation from the same 2005 report.

4. SULFONYLUREA RESPONSIVE (if functional beta-cells remain):
     Early MODY7 patients retain functional beta-cells (oxidative loss is progressive).
     SU closes K-ATP channels → bypasses transcriptional/oxidative deficit → insulin
     exocytosis. ~75–80% response rate (slightly lower than MODY1/3/4 due to ongoing
     oxidative beta-cell loss reducing the responsive pool).

5. NO RENAL, PANCREATIC, OR EXOCRINE FEATURES:
     KLF11 is not expressed in kidney tubule or pancreatic acini.
     No renal cysts (vs MODY5), no pancreatic atrophy (vs MODY5), no exocrine
     insufficiency (vs MODY5), no renal glycosuria (vs MODY3).

6. NOT IN OLDEST MODY PANELS — REQUIRES EXPANDED NGS PANEL:
     Original MODY panels targeted HNF1A, HNF4A, GCK, HNF1B.
     KLF11 was identified in 2005; many clinical labs added it to expanded panels
     only after 2010. Many MODY7 cases labelled T2D (late adult onset, obese relatives)
     or antibody-negative T1D.

7. AUTOANTIBODIES ALWAYS NEGATIVE:
     GADA, ZnT8, IA-2 — all negative. Positive autoantibodies argue strongly against
     MODY7 and suggest autoimmune T1D.

8. PROGRESSIVE HbA1c (not stable like MODY2/GCK):
     Beta-cell loss is progressive (oxidative apoptosis is cumulative). HbA1c rises
     with duration. Onset slightly later than MODY3 (mean ~38–42 yr); overlap with T2D.
     Unlike MODY2, HbA1c is NOT stably mild.

9. C-PEPTIDE PRESERVED EARLY (falls with duration):
     At diagnosis C-peptide preserved → indicates functional beta-cells. Falls
     progressively as MAO-A-driven oxidative apoptosis accumulates over time.
     Long-duration MODY7 patients may become insulin-dependent.

10. ONSET ADULT (LATE 20S TO 50S) — SLIGHTLY LATER THAN MODY3/6:
     Mean onset ~35–45 yr; later than MODY3 (mean 24 yr) or MODY6 (35 yr).
     This later onset increases T2D misdiagnosis risk (especially in overweight patients).

Key Clinical Hallmarks
-----------------------
* Adult-onset diabetes (late 20s to 50s; mean ~38–42 yr)
* Strong family history (~70–80% first-degree relative affected)
* Autoantibodies NEGATIVE (GADA, ZnT8, IA-2) — T1D excluded
* C-peptide PRESERVED at diagnosis; falls progressively with oxidative beta-cell loss
* Sulfonylurea first-line: ~75–80% response rate
* NO renal cysts, NO pancreatic atrophy, NO renal glycosuria, NO macrosomia
* Progressive HbA1c — NOT stable (excludes GCK/MODY2 glucostat mechanism)
* Most contested MODY type — functional validation of variants is important
* KLF11 must be on EXPANDED MODY NGS panel (absent from oldest 4-gene panels)

Diagnostic Strategy
--------------------
* Suspect MODY7: adult-onset DM + family history + antibody-negative + C-pep preserved
* HbA1c progressive (not stable → excludes MODY2/GCK)
* No renal cysts/pancreatic atrophy → MODY5 excluded
* No renal glycosuria → MODY3 less likely (though only 50% MODY3 positive)
* Expanded NGS panel including KLF11 sequencing
* Functional assay: mSin3A/MAO-A reporter (confirms pathogenicity of missense)
* If strong family → cascade genetic testing for first-degree relatives
* MODY probability (Exeter calculator ≥ 25%) should trigger NGS

Comparison: MODY7 vs Other MODY Types
-----------------------------------------
Feature              | MODY7 (KLF11)         | MODY3 (HNF1A)         | MODY6 (NEUROD1)
---------------------|------------------------|------------------------|--------------------
Gene                 | KLF11 2p25.1          | HNF1A 12q24           | NEUROD1 2q31.3
Mechanism            | KLF (zinc finger)      | Homeodomain TF        | bHLH E-box TF
Special feature      | MAO-A/ROS oxidative    | SGLT2 target gene     | Cerebellar/hearing
Renal glycosuria     | ABSENT                | PRESENT (50%)         | ABSENT
Renal cysts          | ABSENT                | ABSENT                | ABSENT
Pancreatic atrophy   | ABSENT                | ABSENT                | ABSENT
SU response          | YES (~75–80%)         | YES (85–90%)          | YES (80–85%)
MODY frequency       | ~1–2% (contested)     | ~35%                  | ~1–2%
Founder variant      | Q62R (PCNLS domain)   | P291fsinsC (Euro)     | R111L (bHLH domain)
Controversy          | HIGH (causality debated)| LOW (well-validated) | LOW (well-validated)
Onset (mean yr)      | ~38–42                | ~24–25                | ~35
Misdiagnosis         | T2D (late onset)       | T1D (young)           | T1D/T2D

Cohort: 40 patients, seed=315.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 315
_COHORT_SIZE = 40

# KLF11 variants — PCNLS/Sin3A domain + zinc finger domain missense + frameshift
_VARIANTS = [
    "Q62R",                 # founding mutation — Neve 2005, PCNLS/mSin3A domain
    "A347S",                # founding mutation — Neve 2005, zinc finger 1
    "T220M",                # missense — repression domain 2
    "R399H",                # missense — zinc finger motif 2
    "E339K",                # missense — zinc finger linker
    "IVS7+3G>T",            # splice-site (intron 7)
    "P58L",                 # missense — PCNLS domain
    "G328E",                # missense — zinc finger domain
    "Other_frameshift",     # novel frameshift
    "Other_missense",       # novel missense — zinc finger region
    "Splice_other",         # other splice-site variant
]
_VARIANT_WEIGHTS = [0.22, 0.18, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.02]

# Treatment: SU-first; slightly lower response than MODY3 due to progressive oxidative loss
_TREATMENTS = [
    "Sulfonylurea (gliclazide)",
    "Sulfonylurea (glibenclamide)",
    "Diet/lifestyle only",
    "Metformin (adjunct)",
    "Insulin (basal-bolus)",
    "Insulin (basal-only)",
]
_TREATMENT_WEIGHTS = [0.30, 0.22, 0.16, 0.12, 0.11, 0.09]

_MISDIAGNOSES = ["T2D", "T1D", "Prediabetes", "None"]
_MISDIAGNOSIS_WEIGHTS = [0.32, 0.18, 0.12, 0.38]

_SEXES = ["M", "F"]


def _make_patient(seed_val: int) -> dict:
    rng = random.Random(seed_val)
    sex = rng.choices(_SEXES, [0.49, 0.51])[0]
    age = rng.randint(28, 68)
    # MODY7 onset: late 20s to 50s; mean ~38–42 yr (later than MODY3/6)
    dx_age = rng.randint(22, min(age, 56))
    duration = age - dx_age

    # HbA1c: progressive; moderate-high range
    hba1c = round(rng.uniform(6.1, 10.2), 1)

    # Fasting glucose: moderate-elevated
    fg = round(rng.uniform(5.8, 13.0), 1)

    # C-peptide: PRESERVED early; falls with oxidative beta-cell loss over duration
    baseline_cp = round(rng.uniform(0.45, 1.50), 2)
    # Oxidative beta-cell loss accumulates faster than transcriptional MODY types
    duration_penalty = min(duration * 0.025, 0.75)
    c_pep = max(round(baseline_cp - duration_penalty, 2), 0.06)

    variant = rng.choices(_VARIANTS, _VARIANT_WEIGHTS)[0]
    treatment = rng.choices(_TREATMENTS, _TREATMENT_WEIGHTS)[0]
    misdiagnosis = rng.choices(_MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)[0]

    # Autoantibodies always negative
    gada = False
    znt8 = False
    ia2 = False

    # Family history: ~75%
    fam_hx = rng.random() < 0.75

    # On SU: check hypoglycaemia episodes
    on_su = "Sulfonylurea" in treatment
    hypo_episodes = rng.randint(0, 3) if on_su else 0

    # SU response: ~77% on SU (slightly lower than MODY1/3/4 due to oxidative loss)
    su_responder = on_su and rng.random() < 0.77

    # BMI: slightly higher (T2D overlap population; later onset)
    bmi = round(rng.uniform(22.5, 34.0), 1)

    # Oxidative burden proxy: MDA (malondialdehyde) nmol/mL — research metric
    # Higher in MODY7 due to MAO-A-driven ROS
    mda_nmol_mL = round(rng.uniform(1.8, 5.2), 2)

    return {
        "patient_id": f"MODY7-{seed_val:04d}",
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
        "on_sulfonylurea": on_su,
        "su_hypoglycaemia_episodes_last_yr": hypo_episodes,
        "su_responder": su_responder,
        "mda_oxidative_stress_nmol_mL": mda_nmol_mL,
        "renal_cysts": False,
        "pancreatic_atrophy_on_imaging": False,
        "exocrine_insufficiency": False,
        "renal_glycosuria": False,
        "macrosomia_at_birth": False,
        "neonatal_hyperinsulinism_history": False,
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
    mean_mda = statistics.mean(p["mda_oxidative_stress_nmol_mL"] for p in patients)

    pct_su = sum(1 for p in patients if p["on_sulfonylurea"]) / _COHORT_SIZE * 100
    pct_su_resp = (
        sum(1 for p in patients if p["su_responder"]) /
        max(sum(1 for p in patients if p["on_sulfonylurea"]), 1) * 100
    )
    pct_fam_hx = sum(1 for p in patients if p["family_history_positive"]) / _COHORT_SIZE * 100
    pct_misdiag = sum(1 for p in patients if p["prior_misdiagnosis"] != "None") / _COHORT_SIZE * 100
    pct_diet = sum(1 for p in patients if "Diet" in p["current_treatment"]) / _COHORT_SIZE * 100
    pct_insulin = sum(1 for p in patients if "Insulin" in p["current_treatment"]) / _COHORT_SIZE * 100

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
            "mean_mda_oxidative_stress_nmol_mL": round(mean_mda, 2),
            "pct_on_sulfonylurea": round(pct_su, 1),
            "pct_su_responders_of_su_treated": round(pct_su_resp, 1),
            "pct_family_hx_positive": round(pct_fam_hx, 1),
            "pct_prior_misdiagnosis": round(pct_misdiag, 1),
            "pct_diet_only": round(pct_diet, 1),
            "pct_insulin_treated": round(pct_insulin, 1),
        },
        "patients": patients,
        "key_facts": [
            "MODY7-KLF11/TIEG2: Krüppel-like factor zinc finger transcriptional repressor — only MODY type with an oxidative stress (MAO-A/ROS) mechanism",
            "KLF11 LOF → excess MAO-A activity → H₂O₂ → progressive beta-cell apoptosis — distinct from all transcription-factor or enzyme haploinsufficiency MODY types",
            "KLF11 also represses SHP (NR0B2), placing MODY7 in the HNF4A–HNF1A regulatory axis: LOF → elevated SHP → dampened HNF4A/HNF1A targets",
            "Most contested MODY subtype — Neve et al. 2005 (Nat Genet) founding report; subsequent studies have questioned penetrance and causality in some cohorts",
            "Founding mutations: Q62R (PCNLS/mSin3A domain) + A347S (zinc finger 1) — both abrogating mSin3A co-repressor recruitment and MAO-A repression",
            "SU first-line (~75–80% response); beta-cells functional early but progressive oxidative loss reduces long-term SU response rate compared to MODY1/3/4",
            "Later onset than MODY3/6 (mean ~38–42 yr) → high T2D misdiagnosis risk; must include C-peptide and family history screening",
            "Autoantibodies always negative — positive GADA/ZnT8/IA-2 argues strongly against MODY7",
            "NOT in oldest MODY panels (4-gene: HNF1A/HNF4A/GCK/HNF1B) — requires expanded MODY NGS panel including KLF11",
            "No renal cysts, no pancreatic atrophy, no exocrine insufficiency, no renal glycosuria, no macrosomia",
        ],
        "alerts": {
            "controversy_alert": "MODY7 is the most contested MODY type — functional variant validation (mSin3A/MAO-A repression assay) is important before clinical labelling",
            "oxidative_mechanism": "Unique ROS/MAO-A mechanism: antioxidant therapy (N-acetylcysteine) is under investigation but NOT yet standard of care",
            "panel_gap": "KLF11 absent from oldest MODY panels — expanded NGS mandatory; many MODY7 families mislabelled T2D or T1D for decades",
            "su_response_lower": "SU response (~75–80%) slightly lower than MODY1/3/4 (~85–90%) due to progressive oxidative beta-cell loss",
            "later_onset_t2d_trap": "Later adult onset (mean ~38–42 yr) combined with family T2D history creates high T2D misdiagnosis rate (~32%)",
        },
        "mody_registry": {
            "type": "MODY7",
            "gene": "KLF11",
            "omim_gene": "*603301",
            "omim_disease": "#610508",
            "chromosome": "2p25.1",
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
    hba1c_tiers = {"<7%": 0, "7–8%": 0, "8–9%": 0, "9–10%": 0, "≥10%": 0}
    for p in patients:
        h = p["hba1c_percent"]
        if h < 7.0:
            hba1c_tiers["<7%"] += 1
        elif h < 8.0:
            hba1c_tiers["7–8%"] += 1
        elif h < 9.0:
            hba1c_tiers["8–9%"] += 1
        elif h < 10.0:
            hba1c_tiers["9–10%"] += 1
        else:
            hba1c_tiers["≥10%"] += 1

    # C-peptide tiers
    cp_tiers = {"<0.20": 0, "0.20–0.59": 0, "0.60–0.99": 0, "≥1.00": 0}
    for p in patients:
        cp = p["c_peptide_nmol_L"]
        if cp < 0.20:
            cp_tiers["<0.20"] += 1
        elif cp < 0.60:
            cp_tiers["0.20–0.59"] += 1
        elif cp < 1.00:
            cp_tiers["0.60–0.99"] += 1
        else:
            cp_tiers["≥1.00"] += 1

    # Age at diagnosis tiers (wider — later onset than MODY3)
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

    # Treatment distribution
    tx_dist: dict = {}
    for p in patients:
        t = p["current_treatment"]
        tx_dist[t] = tx_dist.get(t, 0) + 1

    # Misdiagnosis distribution
    mis_dist: dict = {}
    for p in patients:
        m = p["prior_misdiagnosis"]
        mis_dist[m] = mis_dist.get(m, 0) + 1

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

    # SU hypoglycaemia episodes
    hypo_tiers = {"0 episodes": 0, "1–2 episodes": 0, "3+ episodes": 0}
    on_su = [p for p in patients if p["on_sulfonylurea"]]
    for p in on_su:
        e = p["su_hypoglycaemia_episodes_last_yr"]
        if e == 0:
            hypo_tiers["0 episodes"] += 1
        elif e <= 2:
            hypo_tiers["1–2 episodes"] += 1
        else:
            hypo_tiers["3+ episodes"] += 1

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

    # MDA oxidative stress tiers
    mda_tiers = {"<2.5": 0, "2.5–3.4": 0, "3.5–4.4": 0, "≥4.5": 0}
    for p in patients:
        m = p["mda_oxidative_stress_nmol_mL"]
        if m < 2.5:
            mda_tiers["<2.5"] += 1
        elif m < 3.5:
            mda_tiers["2.5–3.4"] += 1
        elif m < 4.5:
            mda_tiers["3.5–4.4"] += 1
        else:
            mda_tiers["≥4.5"] += 1

    # BMI tiers
    bmi_tiers = {"<25": 0, "25–29.9": 0, "30–34.9": 0, "≥35": 0}
    for p in patients:
        b = p["bmi_kg_m2"]
        if b < 25:
            bmi_tiers["<25"] += 1
        elif b < 30:
            bmi_tiers["25–29.9"] += 1
        elif b < 35:
            bmi_tiers["30–34.9"] += 1
        else:
            bmi_tiers["≥35"] += 1

    return {
        "variant_distribution": var_dist,
        "hba1c_tiers": hba1c_tiers,
        "c_peptide_tiers": cp_tiers,
        "age_at_diagnosis_tiers": dx_age_tiers,
        "treatment_distribution": tx_dist,
        "misdiagnosis_distribution": mis_dist,
        "disease_duration_tiers": dur_tiers,
        "su_hypoglycaemia_tiers_on_su_patients": hypo_tiers,
        "age_groups_current": age_groups,
        "mda_oxidative_stress_tiers": mda_tiers,
        "bmi_tiers": bmi_tiers,
        "summary_flags": {
            "pct_on_su": round(sum(1 for p in patients if p["on_sulfonylurea"]) / _COHORT_SIZE * 100, 1),
            "pct_su_responders": round(
                sum(1 for p in patients if p["su_responder"]) /
                max(sum(1 for p in patients if p["on_sulfonylurea"]), 1) * 100, 1),
            "pct_family_hx": round(sum(1 for p in patients if p["family_history_positive"]) / _COHORT_SIZE * 100, 1),
            "pct_misdiagnosed": round(sum(1 for p in patients if p["prior_misdiagnosis"] != "None") / _COHORT_SIZE * 100, 1),
            "pct_insulin_required": round(sum(1 for p in patients if "Insulin" in p["current_treatment"]) / _COHORT_SIZE * 100, 1),
            "pct_diet_only": round(sum(1 for p in patients if "Diet" in p["current_treatment"]) / _COHORT_SIZE * 100, 1),
            "pct_high_mda": round(sum(1 for p in patients if p["mda_oxidative_stress_nmol_mL"] >= 3.5) / _COHORT_SIZE * 100, 1),
        },
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "name": "MODY7 — KLF11-MODY / TIEG2-MODY",
            "full_name": "Maturity-Onset Diabetes of the Young Type 7",
            "gene": "KLF11 (Krüppel-Like Factor 11) — alias TIEG2 (TGF-β-Inducible Early Gene 2)",
            "chromosome": "2p25.1",
            "omim_gene": "*603301",
            "omim_disease": "#610508",
            "inheritance": "Autosomal Dominant (heterozygous LOF → MODY7)",
            "prevalence": "~1–2% of all MODY; extremely rare; most contested MODY subtype in the literature",
            "mechanism": (
                "KLF11 is a Krüppel-like factor SP1-type zinc finger transcriptional repressor. "
                "LOF → reduced repression of MAO-A (monoamine oxidase A) → excess H₂O₂ (oxidative stress) "
                "→ progressive beta-cell apoptosis. Also represses SHP (NR0B2), linking MODY7 to the "
                "HNF4A/HNF1A axis. Additionally regulates PDX1 expression via SP1-box elements."
            ),
        },
        "genes_and_proteins": {
            "KLF11/TIEG2": (
                "Krüppel-Like Factor 11 / TGF-β-Inducible Early Gene 2 — "
                "SP1-type zinc finger transcriptional repressor; 513 aa; chromosome 2p25.1; *603301. "
                "Three C-terminal C2H2 zinc finger motifs bind GT-box/GC-box (CACCC, CCGCCC) elements. "
                "N-terminal PCNLS/mSin3A-interaction domain recruits mSin3A histone deacetylase co-repressor. "
                "Second repression domain (RD2) recruits Ski-interacting protein (SKIP). "
                "Expressed in: pancreatic islets, liver, testis, brain."
            ),
            "MAO-A pathway": (
                "MAO-A (Monoamine Oxidase A) oxidatively deaminates biogenic amines (serotonin, "
                "noradrenaline, dopamine) generating H₂O₂ as a by-product. KLF11 normally binds "
                "GT-box elements in the MAO-A promoter and recruits mSin3A to repress MAO-A. "
                "KLF11 LOF → excess MAO-A activity → excess H₂O₂ → oxidative beta-cell apoptosis. "
                "This ROS mechanism is unique among all MODY types."
            ),
            "SHP pathway": (
                "KLF11 represses SHP (Small Heterodimer Partner, NR0B2). SHP is a nuclear receptor "
                "co-repressor that dampens HNF4A (MODY1) and HNF1A (MODY3) target gene expression. "
                "KLF11 LOF → elevated SHP → secondary impairment of INS/GCK/GLUT2 transcription. "
                "This positions MODY7 as indirectly dysregulating the MODY1/MODY3 transcription axis."
            ),
            "Zinc finger structure": (
                "Three C2H2 zinc fingers (Cys-X2-Cys-X3-Phe-X5-Leu-X2-His-X3-5-His) bind CACCC/GT-box "
                "consensus elements. A347S disrupts zinc finger 1 → reduced DNA binding and MAO-A repression. "
                "Q62R is in the PCNLS/mSin3A recruitment domain → disrupts co-repressor recruitment without "
                "affecting DNA binding per se."
            ),
        },
        "clinical_terms": {
            "KLF (Krüppel-like factor)": "Family of SP1-type zinc finger TFs (27 members, KLF1–KLF27); named after Drosophila Krüppel segmentation gene; activate or repress target genes via CACCC/GT-box elements",
            "TIEG2": "Historical alias for KLF11 (TGF-β-Inducible Early Gene 2); first cloned as a TGF-β response gene; also known as TIEG3 in some older nomenclature",
            "mSin3A": "Co-repressor complex component (SIN3A histone deacetylase complex); recruited by KLF11 PCNLS domain to deacetylate histones at MAO-A promoter → gene silencing",
            "MAO-A": "Monoamine Oxidase A — flavoenzyme on outer mitochondrial membrane; deaminates serotonin/noradrenaline/dopamine; generates H₂O₂ as oxidative by-product; regulated by KLF11",
            "SHP (NR0B2)": "Small Heterodimer Partner — nuclear receptor without DNA-binding domain; represses HNF4A and HNF1A targets; normally repressed by KLF11; elevated in KLF11 LOF",
            "GT-box/GC-box": "CACCC / CCGCCC consensus binding elements for KLF11 zinc fingers; present in MAO-A, PDX1, and SHP promoters",
            "Q62R": "Gln62Arg — founding MODY7 mutation (Neve 2005); in PCNLS/mSin3A interaction domain; disrupts co-repressor recruitment without affecting zinc finger DNA-binding",
            "A347S": "Ala347Ser — second founding MODY7 mutation (Neve 2005); in zinc finger 1 domain; reduces DNA binding to MAO-A and SHP promoters",
            "GSIS": "Glucose-Stimulated Insulin Secretion — impaired in MODY7 via progressive oxidative beta-cell loss (MAO-A/H₂O₂) and secondary HNF axis impairment (SHP elevation)",
            "MDA": "Malondialdehyde — lipid peroxidation product; surrogate marker for oxidative stress; elevated in MODY7 (research marker, not routine clinical test)",
        },
        "lab_thresholds": {
            "c_peptide_preserved": "≥ 0.60 nmol/L at diagnosis expected in early MODY7; falls progressively with oxidative beta-cell loss",
            "HbA1c_MODY7": "Progressive 6.1–10.2%; rises with duration (oxidative cumulative loss); unlike MODY2 stable 5.6–7.6%",
            "HbA1c_SU_response": "≥ 1.5–2.0% HbA1c reduction within 3 months of SU = confirms functional residual beta-cell capacity",
            "antibodies_negative": "GADA / ZnT8 / IA-2 all NEGATIVE — positive antibodies argue against MODY7",
            "MDA_elevated_research": "MDA > 3.5 nmol/mL suggests elevated oxidative stress — research context only; not yet a clinical MODY7 biomarker",
        },
        "treatment": {
            "first_line": "SULFONYLUREA (gliclazide 40–80 mg/day or glibenclamide 2.5–5 mg/day) — ~75–80% response rate",
            "why_su_works": "Early MODY7: residual functional beta-cells; SU closes K-ATP → depolarization → Ca²⁺ → insulin exocytosis; bypasses oxidative deficit",
            "response_lower_than_mody3": "SU response (~75–80%) slightly lower than MODY1/3/4 (~85–90%) — progressive oxidative apoptosis reduces the SU-responsive beta-cell pool over time",
            "diet_early": "Diet/lifestyle adequate in mild early-onset cases (some patients diagnosed with HbA1c < 7% on diet alone)",
            "metformin_adjunct": "Metformin as adjunct to SU — reduces hepatic glucose production; does NOT address MAO-A mechanism",
            "insulin_late": "Insulin required for progressive beta-cell failure (duration > 10–15 yr in many; oxidative loss cumulative)",
            "pregnancy": "Switch to insulin (SU crosses placenta; monitor neonatal glucose; restart SU postpartum)",
            "antioxidant_research": "N-acetylcysteine (NAC) — proposed antioxidant intervention targeting MAO-A/ROS mechanism; NOT yet standard of care; under investigation",
            "cascade_testing": "Offer genetic testing to first-degree relatives; earlier detection = earlier SU treatment before significant oxidative loss",
        },
        "genetics_testing": {
            "critical_panel_note": "KLF11 is NOT in the oldest 4-gene MODY panels (HNF1A/HNF4A/GCK/HNF1B) — MUST use expanded MODY NGS panel",
            "controversy_note": "MODY7 is the most contested MODY type; functional validation (mSin3A/MAO-A repression assay) is recommended before clinical labelling of novel variants",
            "first_tier": "Expanded MODY NGS panel including KLF11 sequencing (coding regions + splice sites + CNV)",
            "variant_interpretation": "PCNLS domain (aa 1–100) and zinc finger motifs (aa 320–400) have highest pathogenicity prior; missense elsewhere requires functional validation",
            "functional_assay": "MAO-A reporter assay + mSin3A pull-down: confirms loss of repressor function for novel KLF11 missense variants",
            "panels": "MODY panel: HNF1A + HNF4A + GCK + HNF1B + PDX1 + NEUROD1 + KLF11 + CEL + PAX4 + INS + BLK + ABCC8 + KCNJ11",
        },
        "comparison_mody6_7_8": {
            "MODY6 (NEUROD1)": "bHLH E-box TF; cooperates with PDX1; SU ~80–85%; neurological spectrum (rare); R111L founder; well-validated",
            "MODY7 (KLF11)": "Zinc finger repressor; MAO-A/ROS oxidative mechanism (unique); SU ~75–80%; Q62R+A347S founders; most contested; later onset ~38–42 yr",
            "MODY8 (CEL)": "Carboxyl Ester Lipase — exocrine pancreatic enzyme; pancreatic lipomatosis (exocrine atrophy) → secondary beta-cell loss; NO SU; insulin required",
        },
    }
