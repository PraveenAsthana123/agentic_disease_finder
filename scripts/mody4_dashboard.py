"""
MODY4 — PDX1-MODY / IPF1-MODY (Maturity-Onset Diabetes of the Young Type 4)
==============================================================================
Gene       : PDX1 (Pancreatic and Duodenal Homeobox 1) — alias IPF1 (Insulin Promoter Factor 1)
Chromosome : 13q12.2
OMIM Gene  : *600733
OMIM Dis.  : #606392  (MODY4)
Inheritance: Autosomal Dominant (heterozygous LOF → MODY4); homozygous LOF → pancreatic agenesis / PNDM
Prevalence : ~1% of all MODY (rarest classical MODY; <200 families described worldwide)

Mechanism
---------
PDX1 (IPF1) is a homeodomain transcription factor that is the MASTER REGULATOR of:
  1. Beta-cell development (embryonic) — homozygous LOF → pancreatic agenesis (no pancreas)
  2. Beta-cell identity maintenance (postnatal) — PDX1 keeps mature beta-cells in their
     differentiated state; LOF → de-differentiation, beta-cell dysfunction
  3. Insulin gene transcription — PDX1 binds the A3/E1 elements of the INS promoter;
     single-copy LOF → ~50% reduced transcriptional drive → impaired insulin secretion
  4. Target gene regulation: GCK (glucokinase), GLUT2 (SLC2A2), PC1/PCSK1, Nkx6.1,
     MafA, Pax6, Pax4, Isl1 — all downstream of PDX1

Heterozygous LOF → MODY4: one PDX1 copy insufficient for full beta-cell mass and
function → progressive insulin secretory failure → diabetes.

MODY4-UNIQUE Features (critical differentiators)
-------------------------------------------------
1. RAREST CLASSICAL MODY (~1% of all MODY):
     Fewer than 200 families described worldwide in the early literature. Many diagnoses
     are missed because MODY4 is not included in some early gene panels. Prevalence
     is likely underestimated.

2. TWO-HIT DOSAGE EFFECT (Heterozygous vs Homozygous):
     Heterozygous PDX1 LOF → MODY4 (moderate, adult-onset diabetes).
     Compound heterozygous or homozygous PDX1 LOF → pancreatic agenesis or PNDM
     (neonatal-onset, insulin-requiring from birth, severe exocrine + endocrine loss).
     This two-hit model is diagnostic: finding a second pathogenic PDX1 variant changes
     the clinical picture entirely.

3. PDX1 IS THE MASTER BETA-CELL TRANSCRIPTION FACTOR:
     PDX1 directly drives the insulin promoter via A3/E1 box elements — haploinsufficiency
     reduces insulin gene expression. Downstream targets: GCK (glucose sensor),
     GLUT2 (glucose transporter), PC1/PCSK1 (proinsulin processing), MafA, Nkx6.1.
     PDX1 LOF is therefore a GSIS (glucose-stimulated insulin secretion) failure, not
     a glucose-sensing defect (cf. MODY2/GCK).

4. SULFONYLUREA RESPONSIVE (first-line, like MODY1/MODY3):
     Unlike MODY5 (structural atrophy, insulin required), MODY4 beta-cells are
     functionally impaired but PRESENT. Sulfonylure can close K-ATP channels and
     bypass the secretory deficit. 85–90% response rate reported (extrapolated from
     small series; same pathway as MODY1/MODY3).

5. NO PANCREATIC ATROPHY / NO EXOCRINE INSUFFICIENCY:
     Heterozygous PDX1 LOF does NOT cause pancreatic atrophy visible on CT/MRI.
     Exocrine function is preserved (unlike MODY5). This is a PURE beta-cell
     secretory defect — no radiological findings expected.

6. NO RENAL CYSTS (versus MODY5):
     PDX1 is not expressed in the kidney tubule. No renal phenotype. Renal structure
     and function are normal unless there is comorbid pathology.

7. NO RENAL GLYCOSURIA (versus MODY3):
     PDX1 does not regulate SGLT2 (renal glucose transporter). Renal glucose threshold
     is normal. The absence of glycosuria does not differentiate MODY4 from MODY1.

8. NO MACROSOMIA / NEONATAL HYPERINSULINISM (versus MODY1/HNF4A):
     Unlike MODY1 where HNF4A LOF paradoxically causes fetal hyperinsulinism (because
     HNF4A LOF → reduced Sur1/Kir6.2 → K-ATP channel under-expression → hyperinsulinism),
     PDX1 haploinsufficiency does NOT cause neonatal hyperinsulinism. Birthweight normal.

9. VARIABLE EXPRESSIVITY AND LATE ONSET:
     MODY4 has wider age-of-onset variability than MODY3 (mean ~24 yr) or MODY1 (~27 yr).
     Some families present in the 30s–50s. Penetrance is high (~80–90%) but variable
     expressivity makes it look like T2D in older relatives.

10. P63FSDELC — THE FOUNDING MODY4 MUTATION:
     Pro63 frameshift deletion (c.186delC / p.Pro63fs) in exon 1 was reported by
     Stoffers DA et al. (Nat Genet 1997) as the first human PDX1 mutation causing
     MODY-type diabetes. GC-rich region in exon 1 is a mutational hotspot.

Key Clinical Hallmarks
-----------------------
* Young–adult-onset diabetes (teens to 50s; variable onset)
* Strong family history (~80–90% first-degree relative affected)
* Autoantibodies NEGATIVE (GADA, ZnT8, IA-2) — key T1D differentiator
* C-peptide PRESERVED at diagnosis; falls progressively over years (secretory failure)
* Sulfonylurea first-line: excellent response (85–90%), as with MODY1/MODY3
* NO renal cysts, NO pancreatic atrophy, NO renal glycosuria, NO macrosomia
* Progressive beta-cell failure → eventual insulin requirement if diabetes duration >10 yr
* Misdiagnosis: T1D (antibody-negative, young) and T2D (older relatives, progressive)
* No extradiabetic features (pure beta-cell phenotype in heterozygous state)

Diagnostic Strategy
--------------------
* Suspect MODY4 when: young DM + family history + antibody-negative + C-pep preserved
* HbA1c progressive (not stable like MODY2)
* No renal, pancreatic, or genital features (excludes MODY5)
* No renal glycosuria (excludes MODY3 in 50% of cases)
* No neonatal hyperinsulinism history (excludes MODY1)
* MODY NGS panel: PDX1 + HNF1A + HNF4A + GCK + HNF1B
* Exon 1 sequencing quality check (GC-rich region — may need allele-specific PCR or long-read)
* If PDX1 variant found: screen all first-degree relatives; check for second hit (PNDM risk)
* PDX1 CNV testing if sequencing negative but clinical suspicion high

Treatment
----------
* Diet / lifestyle: adequate for early/mild disease in some families
* SULFONYLUREA FIRST-LINE (as for MODY1/MODY3): 85–90% excellent response rate
  - Glibenclamide 2.5–5 mg/day; titrate up slowly; hypoglycaemia risk (same as MODY3)
  - Gliclazide 40–80 mg/day (if glibenclamide not tolerated)
* Insulin: for advanced disease or pregnancy
* Pregnancy: switch to insulin; SU crosses placenta; monitor neonatal glucose
  (theoretical neonatal hypoglycaemia risk with SU in GCK-negative fetus analogy)
* GLP-1 receptor agonists (limited data) — may be considered in specific cases
* No Mg supplementation, no Creon, no renal monitoring beyond standard care

Comparison: MODY4 vs Other MODY Types
---------------------------------------
Feature              | MODY4 (PDX1)          | MODY3 (HNF1A)         | MODY5 (HNF1B)
---------------------|-----------------------|-----------------------|--------------------
Gene                 | PDX1 13q12.2          | HNF1A 12q24           | HNF1B 17q12
Renal glycosuria     | ABSENT                | PRESENT (50%)         | ABSENT
Renal cysts          | ABSENT                | ABSENT                | PRESENT (~70%)
Pancreatic atrophy   | ABSENT                | ABSENT                | PRESENT (CT/MRI)
Exocrine insuff.     | ABSENT                | ABSENT                | PRESENT (~40%)
Sulfonylure resp.    | YES (85–90%)          | YES (85–90%)          | NO (atrophy)
De-novo mutations    | Rare                  | Rare (<5%)            | ~50%
Family history       | ~80–90%               | ~90%                  | ~50%
C-peptide at Dx      | Preserved → falls     | Preserved → falls     | Low / falling
Homozygous → ?       | Pancreatic agenesis   | N/A (not known)       | Severe MODY/PNDM
Neonatal hyperinsul  | ABSENT                | ABSENT                | ABSENT
Macrosomia           | ABSENT                | ABSENT                | ABSENT
MODY frequency       | ~1%                   | ~35%                  | ~5%

Cohort: 40 patients, seed=311.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 311
_COHORT_SIZE = 40

# PDX1 variants — exon 1 GC-rich hotspot + other coding region mutations
_VARIANTS = [
    "P63fsdelC",           # founding mutation — Stoffers 1997
    "R197H",               # missense — homeodomain
    "Q59L",                # missense — activation domain
    "IVS2+1G>A",           # splice-site
    "L244P",               # missense — homeodomain
    "R208S",               # missense
    "G212R",               # missense — homeodomain
    "S210R",               # missense
    "Other_frameshift",    # novel frameshift
    "Other_missense",      # novel missense
    "Splice_other",        # other splice-site
]
_VARIANT_WEIGHTS = [0.20, 0.10, 0.10, 0.08, 0.08, 0.08, 0.07, 0.07, 0.10, 0.07, 0.05]

# Treatment: SU-first, similar to MODY1/MODY3
_TREATMENTS = [
    "Sulfonylurea (glibenclamide)",
    "Sulfonylurea (gliclazide)",
    "Diet/lifestyle only",
    "Insulin (basal-bolus)",
    "Insulin (basal-only)",
    "Metformin (adjunct)",
]
_TREATMENT_WEIGHTS = [0.35, 0.22, 0.15, 0.12, 0.08, 0.08]

_MISDIAGNOSES = ["T1D", "T2D", "Prediabetes", "None"]
_MISDIAGNOSIS_WEIGHTS = [0.25, 0.30, 0.15, 0.30]

_SEXES = ["M", "F"]


def _make_patient(seed_val: int) -> dict:
    rng = random.Random(seed_val)
    sex = rng.choices(_SEXES, [0.48, 0.52])[0]
    age = rng.randint(22, 60)
    # MODY4 has later/wider onset than MODY3 (mean ~35–40 yr in some series)
    dx_age = rng.randint(16, min(age, 52))
    duration = age - dx_age

    # HbA1c: moderate; SU-controlled like MODY3 if on therapy
    hba1c = round(rng.uniform(5.8, 9.5), 1)

    # Fasting glucose: moderate elevation
    fg = round(rng.uniform(5.6, 11.5), 1)

    # C-peptide: PRESERVED early but falls with duration
    # Simulate duration-dependent C-pep decline
    baseline_cp = round(rng.uniform(0.50, 1.60), 2)
    duration_penalty = min(duration * 0.02, 0.60)
    c_pep = max(round(baseline_cp - duration_penalty, 2), 0.08)

    variant = rng.choices(_VARIANTS, _VARIANT_WEIGHTS)[0]
    treatment = rng.choices(_TREATMENTS, _TREATMENT_WEIGHTS)[0]
    misdiagnosis = rng.choices(_MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)[0]

    # Autoantibodies always negative
    gada = False
    znt8 = False
    ia2 = False

    # Family history: high (~85%)
    fam_hx = rng.random() < 0.85

    # On SU: check hypoglycaemia episodes
    on_su = "Sulfonylurea" in treatment
    hypo_episodes = rng.randint(0, 4) if on_su else 0

    # Sulfonylure response: excellent in ~88% on SU
    su_responder = on_su and rng.random() < 0.88

    # Second hit screen (theoretical risk family member — screen flag)
    second_hit_screen = rng.random() < 0.40  # 40% have a family member screened

    return {
        "patient_id": f"MODY4-{seed_val:04d}",
        "age": age,
        "sex": sex,
        "age_at_diagnosis": dx_age,
        "duration_years": duration,
        "hba1c_percent": hba1c,
        "fasting_glucose_mmol": fg,
        "c_peptide_nmol_L": c_pep,
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
        "second_hit_family_screen": second_hit_screen,
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
            "pct_on_sulfonylurea": round(pct_su, 1),
            "pct_su_responders_of_su_treated": round(pct_su_resp, 1),
            "pct_family_hx_positive": round(pct_fam_hx, 1),
            "pct_prior_misdiagnosis": round(pct_misdiag, 1),
            "pct_diet_only": round(pct_diet, 1),
            "pct_insulin_treated": round(pct_insulin, 1),
        },
        "patients": patients,
        "key_facts": [
            "MODY4-RAREST: ~1% of all MODY — fewer than 200 families described worldwide; likely underdiagnosed due to absence from early gene panels",
            "PDX1 is the MASTER BETA-CELL TRANSCRIPTION FACTOR: drives insulin promoter (A3/E1 elements), GCK, GLUT2, PC1, MafA, Nkx6.1 directly",
            "TWO-HIT DOSAGE: Heterozygous LOF → MODY4 (moderate, adult onset); Homozygous/compound het → pancreatic agenesis or PNDM (neonatal onset)",
            "SULFONYLUREA FIRST-LINE (like MODY1/MODY3): 85–90% excellent response; SU closes K-ATP channels, bypasses secretory defect",
            "NO pancreatic atrophy on CT/MRI (vs MODY5): exocrine function preserved; beta-cell defect is functional not structural",
            "NO renal cysts (vs MODY5): PDX1 not expressed in kidney tubule; no renal phenotype in heterozygous state",
            "NO renal glycosuria (vs MODY3): PDX1 does not regulate SGLT2; renal glucose threshold normal",
            "NO macrosomia / neonatal hyperinsulinism (vs MODY1/HNF4A): birthweight normal; no neonatal pancreatic dysfunction",
            "C-peptide PRESERVED at diagnosis (vs MODY5 low C-pep): reflects functional beta-cell impairment, not structural loss",
            "P63fsdelC (Pro63 frameshift) — founding MODY4 mutation (Stoffers DA et al. Nat Genet 1997); GC-rich exon 1 hotspot",
            "Variable expressivity: onset from teens to 50s within same family; penetrance ~80–90%",
            "Autoantibodies NEGATIVE (GADA, ZnT8, IA-2) — mandatory to exclude T1D before diagnosing MODY4",
            "Progressive beta-cell failure: C-peptide falls over time; some patients require insulin >10 yr disease duration",
            "MODY NGS panel must include PDX1; exon 1 quality check required (GC-rich region, may need supplemental testing)",
        ],
        "diagnostic_criteria": {
            "Required": "Young–adult diabetes (onset teens–50s) + strong family history + antibody-negative + C-peptide preserved",
            "Supportive — genetics": "Pathogenic/likely-pathogenic PDX1 variant on MODY NGS panel (sequencing + CNV)",
            "Supportive — SU response": "Excellent sulfonylurea response (HbA1c reduction ≥ 1.5–2%) strongly suggests functional MODY",
            "Exclusion — MODY5": "No renal cysts, no pancreatic atrophy on imaging, no exocrine insufficiency",
            "Exclusion — MODY3": "Renal glycosuria absent does NOT exclude MODY3 (only 50% positive) — confirm by genetics",
            "Exclusion — MODY2": "HbA1c progressive (not stable at 5.6–7.6%); OGTT increment > 3.5 mmol/L",
            "Exclusion — MODY1": "No macrosomia history, no neonatal hyperinsulinism — confirms against MODY1",
            "Second hit": "If two PDX1 variants found in patient or family member → risk of PNDM/pancreatic agenesis — urgent specialist review",
            "Antibodies": "GADA / ZnT8 / IA-2 NEGATIVE — positive result argues against MODY4",
        },
    }


def get_breakdown() -> dict:
    patients = _generate_cohort()

    # Variant distribution
    var_dist: dict = {}
    for p in patients:
        var_dist[p["variant"]] = var_dist.get(p["variant"], 0) + 1

    # HbA1c tiers
    hba1c_tiers = {"<6.5%": 0, "6.5–7.4%": 0, "7.5–8.4%": 0, "≥8.5%": 0}
    for p in patients:
        h = p["hba1c_percent"]
        if h < 6.5:
            hba1c_tiers["<6.5%"] += 1
        elif h < 7.5:
            hba1c_tiers["6.5–7.4%"] += 1
        elif h < 8.5:
            hba1c_tiers["7.5–8.4%"] += 1
        else:
            hba1c_tiers["≥8.5%"] += 1

    # C-peptide tiers
    cp_tiers = {"<0.30 (low)": 0, "0.30–0.59 (moderate)": 0, "≥0.60 (preserved)": 0}
    for p in patients:
        c = p["c_peptide_nmol_L"]
        if c < 0.30:
            cp_tiers["<0.30 (low)"] += 1
        elif c < 0.60:
            cp_tiers["0.30–0.59 (moderate)"] += 1
        else:
            cp_tiers["≥0.60 (preserved)"] += 1

    # Age at diagnosis tiers
    dx_age_tiers = {"<20 yr": 0, "20–29 yr": 0, "30–39 yr": 0, "40–49 yr": 0, "≥50 yr": 0}
    for p in patients:
        a = p["age_at_diagnosis"]
        if a < 20:
            dx_age_tiers["<20 yr"] += 1
        elif a < 30:
            dx_age_tiers["20–29 yr"] += 1
        elif a < 40:
            dx_age_tiers["30–39 yr"] += 1
        elif a < 50:
            dx_age_tiers["40–49 yr"] += 1
        else:
            dx_age_tiers["≥50 yr"] += 1

    # Treatment distribution
    tx_dist: dict = {}
    for p in patients:
        tx_dist[p["current_treatment"]] = tx_dist.get(p["current_treatment"], 0) + 1

    # Misdiagnosis distribution
    mis_dist: dict = {}
    for p in patients:
        mis_dist[p["prior_misdiagnosis"]] = mis_dist.get(p["prior_misdiagnosis"], 0) + 1

    # Duration tiers
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
    hypo_tiers = {"0 episodes": 0, "1–2 episodes": 0, "3–4 episodes": 0}
    on_su = [p for p in patients if p["on_sulfonylurea"]]
    for p in on_su:
        e = p["su_hypoglycaemia_episodes_last_yr"]
        if e == 0:
            hypo_tiers["0 episodes"] += 1
        elif e <= 2:
            hypo_tiers["1–2 episodes"] += 1
        else:
            hypo_tiers["3–4 episodes"] += 1

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
        "treatment_distribution": tx_dist,
        "misdiagnosis_distribution": mis_dist,
        "disease_duration_tiers": dur_tiers,
        "su_hypoglycaemia_tiers_on_su_patients": hypo_tiers,
        "age_groups_current": age_groups,
        "summary_flags": {
            "pct_on_su": round(sum(1 for p in patients if p["on_sulfonylurea"]) / _COHORT_SIZE * 100, 1),
            "pct_su_responders": round(
                sum(1 for p in patients if p["su_responder"]) /
                max(sum(1 for p in patients if p["on_sulfonylurea"]), 1) * 100, 1),
            "pct_family_hx": round(sum(1 for p in patients if p["family_history_positive"]) / _COHORT_SIZE * 100, 1),
            "pct_misdiagnosed": round(sum(1 for p in patients if p["prior_misdiagnosis"] != "None") / _COHORT_SIZE * 100, 1),
            "pct_insulin_required": round(sum(1 for p in patients if "Insulin" in p["current_treatment"]) / _COHORT_SIZE * 100, 1),
            "pct_diet_only": round(sum(1 for p in patients if "Diet" in p["current_treatment"]) / _COHORT_SIZE * 100, 1),
        },
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "name": "MODY4 — PDX1-MODY / IPF1-MODY",
            "full_name": "Maturity-Onset Diabetes of the Young Type 4",
            "gene": "PDX1 (Pancreatic and Duodenal Homeobox 1) — alias IPF1 (Insulin Promoter Factor 1)",
            "chromosome": "13q12.2",
            "omim_gene": "*600733",
            "omim_disease": "#606392",
            "inheritance": "Autosomal Dominant (heterozygous LOF → MODY4); homozygous LOF → pancreatic agenesis",
            "prevalence": "~1% of all MODY; rarest classical MODY; likely underdiagnosed",
            "mechanism": (
                "PDX1/IPF1 is the master beta-cell transcription factor. Haploinsufficiency → "
                "reduced transcription of INS (insulin), GCK, GLUT2, PC1 → impaired GSIS "
                "(glucose-stimulated insulin secretion) → progressive diabetes. Homozygous LOF → "
                "no PDX1 → pancreatic agenesis or severe PNDM (neonatal onset)."
            ),
        },
        "genes_and_proteins": {
            "PDX1/IPF1": (
                "Pancreatic and Duodenal Homeobox 1 / Insulin Promoter Factor 1 — "
                "homeodomain transcription factor; 283 aa; chromosome 13q12.2; "
                "expressed in pancreatic beta-cells (high), delta-cells, duodenum (low). "
                "Binds TAAT core motifs via homeodomain (aa 206–267). "
                "Transactivates: INS (insulin gene A3/E1 elements), GCK, GLUT2, PC1/PCSK1, MafA, Nkx6.1."
            ),
            "INS promoter binding": (
                "PDX1 binds A3/E1 elements of the insulin promoter; cooperates with MafA and NeuroD1 "
                "to achieve full beta-cell-specific transcription. Haploinsufficiency → "
                "~50% reduction in INS drive → impaired insulin output."
            ),
            "Dosage effect": (
                "Two-hit model: 1 copy PDX1 → MODY4 (moderate, adult onset); "
                "0 copies PDX1 → pancreatic agenesis (no islets, no exocrine pancreas, PNDM at birth). "
                "Family screening must include second-hit risk assessment."
            ),
        },
        "clinical_terms": {
            "GSIS": "Glucose-Stimulated Insulin Secretion — impaired in MODY4 due to reduced INS/GCK/GLUT2 transcription",
            "Pancreatic agenesis": "Homozygous PDX1 LOF → no pancreas formed → neonatal diabetes + exocrine failure (vs MODY4 heterozygous = adult diabetes only)",
            "IPF1": "Insulin Promoter Factor 1 — historical alias for PDX1",
            "Haploinsufficiency": "Single functional copy of PDX1 is insufficient for normal beta-cell function → MODY4",
            "P63fsdelC": "First reported MODY4 variant (Stoffers et al. 1997); frameshift in GC-rich exon 1 hotspot",
            "GC_rich_exon_1": "PDX1 exon 1 contains a GC-rich region that can cause sequencing failure (polyC/polyG stretches); supplemental assays may be needed",
            "Variable expressivity": "Same pathogenic PDX1 variant causes different ages of onset within the same family; explains MODY4 looking like T2D in older members",
        },
        "lab_thresholds": {
            "c_peptide_preserved": "≥ 0.60 nmol/L at diagnosis expected in MODY4 (functional defect, not structural loss)",
            "HbA1c_MODY4": "Variable 5.8–9.5%; progressive with duration; higher than MODY2 (stable); responds to SU",
            "HbA1c_SU_response": "≥ 1.5–2.0% HbA1c reduction within 3 months of SU = excellent MODY-type response",
            "antibodies_negative": "GADA / ZnT8 / IA-2 all NEGATIVE — mandatory pre-requisite for MODY4 diagnosis",
        },
        "treatment": {
            "first_line": "SULFONYLUREA (glibenclamide 2.5–5 mg/day or gliclazide 40–80 mg/day) — 85–90% excellent response",
            "why_su_works": "PDX1 haploinsufficiency → functional beta-cells present; SU closes K-ATP channels → depolarization → Ca²⁺ influx → insulin exocytosis",
            "hypoglycaemia_risk": "Monitor for hypoglycaemia (same risk as MODY3 SU treatment); start low dose; educate patient",
            "diet_early": "Diet / lifestyle adequate for early/mild disease; HbA1c < 6.5% on diet alone in some",
            "insulin_late": "Insulin required if progressive beta-cell failure (duration > 10–15 yr in some); basal-bolus or basal-only",
            "pregnancy": "Switch to insulin (SU crosses placenta; neonatal hypoglycaemia risk); monitor postpartum for relapse",
            "no_creon": "No exocrine replacement needed (exocrine function normal in heterozygous PDX1 LOF)",
            "cascade_testing": "Offer genetic testing to all first-degree relatives; identify second-hit risk (compound-het family members may develop PNDM)",
        },
        "genetics_testing": {
            "first_tier": "MODY NGS panel including PDX1 sequencing (coding regions + splice sites); verify exon 1 quality",
            "exon_1_caveat": "GC-rich exon 1 may have reduced coverage on standard NGS — verify with Sanger or allele-specific PCR if clinical suspicion high",
            "cnv_testing": "PDX1 CNV (deletion/duplication) testing if sequencing negative but family history compelling",
            "second_hit": "If pathogenic PDX1 variant found, screen all relatives; if compound-het or homozygous found in child → URGENT referral",
            "panels": "MODY panel: HNF1A + HNF4A + GCK + HNF1B + PDX1 + NEUROD1 + KLF11 + CEL + PAX4 + INS",
        },
        "comparison_mody1_3_4_5": {
            "MODY1 (HNF4A)": "Macrosomia + TNH (50%); SU first-line; no renal glycosuria; HNF4A→HNF1A same axis as MODY3",
            "MODY2 (GCK)": "Stable mild HbA1c (glucostat reset); no treatment; OGTT increment < 3.5 mmol/L",
            "MODY3 (HNF1A)": "Renal glycosuria 50%; SU first-line 85–90%; no renal cysts; most common (35%)",
            "MODY4 (PDX1)": "Master TF for beta-cell identity; SU-responsive; NO renal glycosuria; NO cysts; NO atrophy; rarest (~1%); two-hit risk",
            "MODY5 (HNF1B)": "Renal cysts precede DM; pancreatic atrophy; insulin required; hypomagnesaemia; de-novo 50%",
        },
    }
