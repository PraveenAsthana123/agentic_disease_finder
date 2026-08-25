"""
MODY13 — KCNJ11-MODY (Maturity-Onset Diabetes of the Young Type 13)
================================================================================
Gene       : KCNJ11 (Potassium Inwardly Rectifying Channel Subfamily J Member 11;
             Kir6.2 — pore-forming subunit of the pancreatic K-ATP channel)
Chromosome : 11p15.1 (adjacent to ABCC8/SUR1; same genomic neighbourhood)
OMIM Gene  : *600937
OMIM Dis.  : referenced under *600937 phenotypic series (MODY13 / PNDM1 spectrum)
Inheritance: Autosomal Dominant (heterozygous activating / GOF missense → MODY13)
Prevalence : Rare; ~1–2% of MODY; often confused with MODY12 (same K-ATP channel);
             underdiagnosed; European-enriched discovery cohorts

Mechanism
---------
KCNJ11 encodes Kir6.2 (390 aa), the pore-forming subunit of the K-ATP channel.
The functional channel is a hetero-octamer: (Kir6.2)₄·(SUR1)₄.
Kir6.2 forms the actual K⁺ pore AND contains the primary ATP-binding site.
SUR1 (ABCC8) is the regulatory ABC-transporter subunit (MODY12).

Normal GSIS:
  Glucose↑ → glycolysis → ATP↑ → ATP binds Kir6.2 N-terminal (T224, R50 region) →
  K-ATP closes → membrane depolarises → Ca²⁺ influx → insulin exocytosis

MODY13 GOF mechanism:
  Activating KCNJ11 missense (often near ATP-binding site of Kir6.2 pore) →
  Kir6.2 has REDUCED affinity for ATP (or channel prefers open state) →
  At high glucose, ATP rise FAILS to close Kir6.2 pore →
  Reduced Ca²⁺ influx → blunted GSIS → hyperglycaemia

KEY: C-peptide is PRESERVED — beta-cell mass structurally intact; defect is in
K-ATP pore gating (ATP-sensing on Kir6.2), not beta-cell apoptosis or transcription.

Kir6.2 ATP-binding site:
  Primarily N-terminal residues (R50, T224 region) on Kir6.2 cytoplasmic domain.
  ATP coordinates to channel interior — each of 4 Kir6.2 subunits has one site.
  GOF reduces ATP IC₅₀ for channel closure → pore stays open.
  SU (sulfonylurea) closes K-ATP by binding SUR1 NBD2 directly (not Kir6.2) →
  SU effect is ATP-independent → works even when Kir6.2 ATP-binding is impaired.

K-ATP Spectrum (KCNJ11 GOF severity):
  Severe GOF → PNDM1 (neonatal; Kir6.2 pore barely ATP-responsive; onset < 6 months)
  Severe + neurological → DEND/iDEND (Developmental delay + Epilepsy + ND; V59M, Q52R)
  Moderate GOF → TNDM (transient neonatal DM; remits; recurs adulthood → MODY13)
  Mild GOF → MODY13 (teens–adult onset; standard SU; excellent response; no neurology)

DEND syndrome (Developmental delay, Epilepsy, Neonatal Diabetes):
  Caused by SEVERE KCNJ11 GOF variants (V59M, I296L, Q52R, etc.).
  Kir6.2 is expressed in brain, heart, skeletal muscle — severe GOF causes neurological
  and cardiac K-ATP dysfunction beyond pancreatic beta-cells.
  MODY13 (mild GOF: R201H, R201C): NO neurological features — pure diabetes only.
  This is the critical distinction for clinical reporting.

SULFONYLUREA FIRST-LINE:
  SU (glibenclamide/gliclazide) binds SUR1 (the ABCC8 partner, NOT Kir6.2) → closes K-ATP
  by allosteric mechanism via SUR1 NBD2 → bypasses Kir6.2 GOF ATP-binding defect.
  MODY13 SU response: ~80–85% excellent (slightly lower than MODY12's 85–90%,
  because SU works via SUR1, not directly on the mutated Kir6.2 subunit).
  Still the BEST response rate available; far superior to insulin.

Key Founding Mutations (KCNJ11 GOF, MODY13 / adult-onset phenotype)
----------------------------------------------------------------------
* R201H (c.602G>A) — ATP-binding vicinity; Sagen et al. 2004 (Norwegian family);
  most common adult-onset KCNJ11 GOF; pure diabetes without DEND; mild GOF
* R201C (c.601C>T) — same residue as R201H; similar phenotype; mild GOF
* E23K (c.67G>A) — common T2D GWAS variant; very mild GOF; population polymorphism;
  functionally borderline; not classic MODY13 but same GOF axis
* C42R (c.124T>C) — moderate GOF; PNDM/TNDM boundary; adult recurrence pattern
* H46Y (c.136C>T) — mild GOF; adult onset; Caucasian families
* I197F (c.589A>T) — pore-adjacent; moderate GOF; Caucasian
* Novel_KCNJ11_GOF — novel; patch-clamp ATP IC₅₀ assay mandatory
* V59M (c.175G>A) — NOT MODY13; PNDM/iDEND; severe GOF; included for clinical contrast

Differentiation from PNDM-KCNJ11:
  MODY13: heterozygous mild GOF → R201H/R201C → standard SU dose → channel closable →
           excellent response; no neonatal presentation; teens–adult onset; NO DEND
  PNDM1 : severe GOF (V59M) → channel minimally ATP-sensitive → very high SU dose;
           onset < 6 months; DEND neurological features in iDEND sub-spectrum

Comparison with MODY12 (ABCC8 — SUR1):
  MODY13 (KCNJ11/Kir6.2): pore subunit GOF; ATP binds here directly on Kir6.2
  MODY12 (ABCC8/SUR1): regulatory subunit GOF; ATP-sensing NBD domain on SUR1
  Both → K-ATP constitutively open → blunted GSIS; both respond to SU
  MODY13 unique: DEND/iDEND neurological spectrum with severe Kir6.2 variants
  MODY12 unique: SU binds mutated SUR1 directly (slightly higher SU response ~85–90%)
  Clinical presentation: near-identical for mild GOF; distinguish by gene panel

E23K (GWAS) vs MODY13 distinction:
  E23K (rs5219): population-level T2D risk variant; very mild GOF; HWE; not AD MODY
  MODY13 R201H/R201C: pedigree-segregating rare variant; AD transmission; formal MODY

Clinical Profile
----------------
* Onset: Variable; MODY-pattern: teens–adult (mean ~25–35 yr; later than MODY12)
* C-peptide: PRESERVED (K-ATP pore gating defect; no structural beta-cell loss)
* HbA1c: Progressive — severity correlates with degree of Kir6.2 GOF
* Treatment: SU FIRST-LINE — ~80–85% excellent response
* Autoantibodies: NEGATIVE (GADA, ZnT8, IA-2) — mandatory T1D/LADA exclusion
* Misdiagnosis: T1D most common (~40%); T2D ~22%
* DKA at presentation: ~15% (slightly less than MODY12's ~20%)
* No exocrine insufficiency (vs MODY8/CEL)
* No renal cysts (vs MODY5/HNF1B)
* No renal glycosuria (vs MODY3/HNF1A)
* No ER-stress or falling C-peptide (vs MODY10/INS)
* No DEND neurological features in MODY13 mild GOF (vs PNDM1-DEND severe GOF)

Diagnostic Strategy
-------------------
* Suspect MODY13: adult-onset DM, antibody-negative, family history, preserved C-peptide,
  excellent SU response, no neurological features (DEND excluded → mild GOF)
* Test: KCNJ11 gene sequencing; expanded MODY NGS panel (ABCC8 + KCNJ11 mandatory pair)
* Functional validation: patch-clamp (inside-out; ATP IC₅₀ shift confirms GOF);
  86Rb⁺ efflux assay (COSm6 cells; KCNJ11-variant + ABCC8-WT co-expression);
  pharmacological rescue by glibenclamide confirms SU correctability
* NOT in oldest MODY panels — KCNJ11 GOF as MODY13 requires expanded NGS panel
* Check ALL first-degree relatives; 50% AD transmission
* Always co-sequence ABCC8: MODY12 vs MODY13 clinically indistinguishable

Cohort: 40 patients, seed=327.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 327
_COHORT_SIZE = 40

# KCNJ11 GOF variants — pore-subunit Kir6.2 activating missense (MODY13 phenotype: mild GOF only)
_VARIANTS = [
    "R201H (c.602G>A)",          # Kir6.2 ATP-vicinity; Sagen 2004 (Norwegian); most common MODY13
    "R201C (c.601C>T)",          # Same residue; similar mild GOF; multiple families
    "E23K (c.67G>A)",            # T2D GWAS variant; very mild GOF; borderline MODY
    "C42R (c.124T>C)",           # Moderate GOF; PNDM/TNDM/MODY13 boundary
    "H46Y (c.136C>T)",           # Mild GOF; Caucasian adult-onset
    "I197F (c.589A>T)",          # Pore-adjacent; moderate GOF
    "Novel_KCNJ11_GOF",          # Novel; patch-clamp mandatory
    "Splice_KCNJ11",             # Splice site; partial GOF isoform; rare
]
_VARIANT_WEIGHTS = [0.32, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.01]

_ETHNICITIES = [
    "European-Norwegian/Scandinavian",
    "European-UK/Irish",
    "European-Other",
    "North American European",
    "Asian",
    "Other/Unknown",
]
_ETHNICITY_WEIGHTS = [0.28, 0.22, 0.20, 0.16, 0.08, 0.06]

_TREATMENTS = [
    "Sulfonylurea (monotherapy)",
    "Sulfonylurea + Metformin",
    "Insulin → switched to SU",
    "Sulfonylurea + Insulin (transitional)",
    "Metformin monotherapy",
    "Lifestyle / diet",
]
_TX_WEIGHTS = [0.46, 0.22, 0.20, 0.06, 0.04, 0.02]

_MISDIAGNOSES = [
    "T1D",
    "T2D",
    "PNDM / neonatal DM",
    "Prediabetes",
    "None (correctly diagnosed)",
]
_MISDIAG_WEIGHTS = [0.40, 0.22, 0.06, 0.10, 0.22]

_DISEASE_STAGES = [
    "Early (HbA1c 5.8–7.4%)",
    "Moderate (HbA1c 7.5–8.9%)",
    "Advanced (HbA1c ≥ 9.0%)",
]
_STAGE_WEIGHTS = [0.43, 0.35, 0.22]

# K-ATP pore (Kir6.2) GOF severity tiers — determines PNDM vs MODY13 boundary
_KATP_GOF_TIERS = [
    "Mild GOF (Kir6.2 ATP IC₅₀ moderately shifted; standard SU dose closes pore)",
    "Mild-Moderate GOF (responds well; slightly elevated SU dose needed)",
    "Moderate GOF (close to PNDM1 boundary; high SU dose; excellent response)",
    "Near-PNDM1 boundary (TNDM history or very early onset; no DEND features)",
]
_KATP_WEIGHTS = [0.42, 0.30, 0.18, 0.10]

# SU response categories — slightly lower than MODY12 (~80–85% vs 85–90%)
_SU_RESPONSE = [
    "Excellent (HbA1c < 7.0% on SU alone)",
    "Good (HbA1c 7.0–7.9% on SU alone)",
    "Partial (requires SU + Metformin)",
    "Insufficient (insulin required despite SU)",
]
_SU_RESPONSE_WEIGHTS = [0.54, 0.24, 0.13, 0.09]


def _build_cohort() -> list:
    rng = random.Random(_SEED)

    def wchoice(choices, weights):
        return rng.choices(choices, weights=weights, k=1)[0]

    cohort = []
    for i in range(_COHORT_SIZE):
        variant = wchoice(_VARIANTS, _VARIANT_WEIGHTS)
        ethnicity = wchoice(_ETHNICITIES, _ETHNICITY_WEIGHTS)
        treatment = wchoice(_TREATMENTS, _TX_WEIGHTS)
        prior_misdiag = wchoice(_MISDIAGNOSES, _MISDIAG_WEIGHTS)
        stage = wchoice(_DISEASE_STAGES, _STAGE_WEIGHTS)
        katp_tier = wchoice(_KATP_GOF_TIERS, _KATP_WEIGHTS)
        su_response = wchoice(_SU_RESPONSE, _SU_RESPONSE_WEIGHTS)

        # Age at diagnosis: MODY13 slightly later than MODY12 (mean ~25–35 yr)
        dx_age = round(rng.gauss(28, 10), 1)
        dx_age = max(0.5, min(60, dx_age))

        # HbA1c: correlates with stage
        if "Early" in stage:
            hba1c = round(rng.gauss(6.8, 0.5), 1)
            hba1c = max(5.8, min(7.4, hba1c))
        elif "Moderate" in stage:
            hba1c = round(rng.gauss(8.1, 0.5), 1)
            hba1c = max(7.5, min(8.9, hba1c))
        else:
            hba1c = round(rng.gauss(9.8, 0.9), 1)
            hba1c = max(9.0, min(13.0, hba1c))

        # Fasting glucose (mmol/L): correlated with stage
        fg_base = {"Early": 7.5, "Moderate": 10.5, "Advanced": 14.0}
        key = stage.split("(")[0].strip()
        fg = round(rng.gauss(fg_base.get(key, 10.0), 1.5), 1)
        fg = max(5.5, min(22.0, fg))

        # C-peptide: preserved throughout (K-ATP pore gating defect; no structural loss)
        c_pep = round(rng.gauss(0.82, 0.22), 2)
        c_pep = max(0.55, min(1.50, c_pep))

        # BMI: variable; MODY13 not strongly linked to obesity unlike T2D
        bmi = round(rng.gauss(23.5, 3.8), 1)
        bmi = max(16.5, min(38.0, bmi))

        # Family history: AD; 50% transmission but ~65–72% clinically positive
        family_hx = rng.random() < 0.68

        # Disease duration (years since dx)
        duration = round(rng.gauss(8.5, 6.0), 1)
        duration = max(0.1, min(35.0, duration))

        # DKA at first presentation (slightly less than MODY12 ~20%; MODY13 ~15%)
        dka_at_dx = rng.random() < 0.15

        # Neonatal DM history (personal or family — clue to PNDM/TNDM spectrum)
        neonatal_hx = rng.random() < 0.10

        cohort.append({
            "id": i + 1,
            "variant": variant,
            "ethnicity": ethnicity,
            "treatment": treatment,
            "prior_misdiagnosis": prior_misdiag,
            "stage": stage,
            "katp_tier": katp_tier,
            "su_response": su_response,
            "age_dx": dx_age,
            "hba1c": hba1c,
            "fasting_glucose": fg,
            "c_peptide_nmol_L": c_pep,
            "bmi": bmi,
            "family_history_positive": family_hx,
            "disease_duration_yr": duration,
            "dka_at_presentation": dka_at_dx,
            "neonatal_hx": neonatal_hx,
            "antibody_negative": True,          # always in MODY13
            "c_peptide_label": "Preserved",
        })
    return cohort


def get_overview() -> dict:
    cohort = _build_cohort()
    n = len(cohort)

    # KPIs
    mean_hba1c = round(statistics.mean(p["hba1c"] for p in cohort), 1)
    mean_dx_age = round(statistics.mean(p["age_dx"] for p in cohort), 1)
    mean_fg = round(statistics.mean(p["fasting_glucose"] for p in cohort), 1)
    pct_excellent_su = round(
        sum(1 for p in cohort if "Excellent" in p["su_response"]) / n * 100, 1)
    pct_t1d_misdiag = round(
        sum(1 for p in cohort if p["prior_misdiagnosis"] == "T1D") / n * 100, 1)
    pct_family_hx = round(
        sum(1 for p in cohort if p["family_history_positive"]) / n * 100, 1)
    pct_dka = round(
        sum(1 for p in cohort if p["dka_at_presentation"]) / n * 100, 1)
    pct_antibody_neg = 100.0
    pct_c_pep_preserved = round(
        sum(1 for p in cohort if p["c_peptide_nmol_L"] >= 0.60) / n * 100, 1)

    kpis = {
        "gene":                "KCNJ11",
        "chromosome":          "11p15.1",
        "mean_hba1c":          f"{mean_hba1c}%",
        "mean_dx_age":         f"{mean_dx_age} yr",
        "mean_fasting_glucose": f"{mean_fg} mmol/L",
        "pct_excellent_su":    f"{pct_excellent_su}%",
        "pct_t1d_misdiag":     f"{pct_t1d_misdiag}%",
        "pct_family_hx":       f"{pct_family_hx}%",
        "pct_dka_at_dx":       f"{pct_dka}%",
        "pct_antibody_neg":    f"{pct_antibody_neg}%",
        "pct_c_pep_preserved": f"{pct_c_pep_preserved}%",
        "omim_gene":           "*600937",
    }

    alerts = {
        "su_first_line":
            "SU (glibenclamide/gliclazide) is FIRST-LINE — ~80–85% excellent response. "
            "Works via SUR1 partner (not Kir6.2 directly) — bypasses GOF ATP-binding defect.",
        "dend_exclusion":
            "MODY13 (mild GOF: R201H, R201C): NO DEND/iDEND neurological features. "
            "DEND only with SEVERE GOF (V59M, I296L, Q52R) → NOT MODY13 phenotype → PNDM1-DEND.",
        "t1d_misdiagnosis":
            "T1D misdiagnosis ~40%. Mandatory: autoantibody panel (GADA, ZnT8, IA-2) + C-peptide. "
            "Antibody-negative + C-peptide preserved + family Hx + SU response → test KCNJ11.",
        "c_peptide_preserved":
            "C-peptide PRESERVED throughout — K-ATP pore gating defect only; beta-cell mass intact. "
            "Falling C-peptide would suggest MODY10 (INS/ER-stress) — not MODY13.",
        "co_panel_mandatory":
            "Always co-sequence ABCC8 (MODY12) when testing KCNJ11 (MODY13). "
            "Both on 11p15.1; clinically indistinguishable; expanded MODY NGS panel required.",
    }

    key_facts = [
        "KCNJ11 encodes Kir6.2 — pore-forming subunit of K-ATP channel; (Kir6.2)₄·(SUR1)₄ octamer",
        "Kir6.2 contains the primary ATP-binding site — MODY13 GOF reduces ATP affinity → pore stays open",
        "Mild GOF (R201H, R201C): MODY13 adult-onset; NO neurological features; excellent SU response",
        "Severe GOF (V59M, I296L): DEND/iDEND neurological spectrum; NOT MODY13 phenotype",
        "C-peptide PRESERVED — pore gating defect only; beta-cell mass structurally intact",
        "SU response ~80–85% (excellent); SU closes K-ATP via SUR1 partner — bypasses Kir6.2 GOF",
        "SU mechanism: binds SUR1 NBD2 → allosteric channel closure, ATP-independent",
        "T1D misdiagnosis ~40% — antibody-negative + C-peptide preserved distinguishes MODY13",
        "KCNJ11 (11p15.1) adjacent to ABCC8 (11p15.1) — co-sequence both in K-ATP MODY panels",
        "E23K (rs5219) T2D GWAS variant — same gene, very mild GOF, population level; not MODY13",
        "Autosomal Dominant; 50% transmission; all first-degree relatives need screening",
        "Oldest MODY panels (4–6 gene) do NOT include KCNJ11 — expanded NGS panel essential",
    ]

    patients_preview = [
        {
            "id": p["id"],
            "variant": p["variant"],
            "age_dx": p["age_dx"],
            "hba1c": p["hba1c"],
            "c_peptide": p["c_peptide_label"],
            "treatment": p["treatment"],
            "stage": p["stage"],
            "family_hx": p["family_history_positive"],
            "dka_at_dx": p["dka_at_presentation"],
            "neonatal_hx": p["neonatal_hx"],
        }
        for p in cohort[:12]
    ]

    return {
        "kpis": kpis,
        "alerts": alerts,
        "key_facts": key_facts,
        "patients": patients_preview,
        "cohort_size": n,
        "seed": _SEED,
    }


def get_breakdown() -> dict:
    cohort = _build_cohort()
    n = len(cohort)

    # Variant distribution
    var_dist: dict = {}
    for p in cohort:
        var_dist[p["variant"]] = var_dist.get(p["variant"], 0) + 1

    # Ethnicity distribution
    eth_dist: dict = {}
    for p in cohort:
        eth_dist[p["ethnicity"]] = eth_dist.get(p["ethnicity"], 0) + 1

    # HbA1c tiers
    hba1c_tiers = {"< 7.0%": 0, "7.0–7.9%": 0, "8.0–8.9%": 0, "≥ 9.0%": 0}
    for p in cohort:
        v = p["hba1c"]
        if v < 7.0:
            hba1c_tiers["< 7.0%"] += 1
        elif v < 8.0:
            hba1c_tiers["7.0–7.9%"] += 1
        elif v < 9.0:
            hba1c_tiers["8.0–8.9%"] += 1
        else:
            hba1c_tiers["≥ 9.0%"] += 1

    # C-peptide tiers (PRESERVED pattern)
    cp_tiers = {"≥ 1.0 nmol/L (High preserved)": 0,
                "0.70–0.99 nmol/L (Normal preserved)": 0,
                "0.50–0.69 nmol/L (Low-preserved)": 0,
                "< 0.50 nmol/L (Borderline)": 0}
    for p in cohort:
        v = p["c_peptide_nmol_L"]
        if v >= 1.0:
            cp_tiers["≥ 1.0 nmol/L (High preserved)"] += 1
        elif v >= 0.70:
            cp_tiers["0.70–0.99 nmol/L (Normal preserved)"] += 1
        elif v >= 0.50:
            cp_tiers["0.50–0.69 nmol/L (Low-preserved)"] += 1
        else:
            cp_tiers["< 0.50 nmol/L (Borderline)"] += 1

    # Age at diagnosis tiers — MODY13 shifts later than MODY12
    dx_age_tiers = {"< 16 yr (Paediatric)": 0, "16–25 yr (Young adult)": 0,
                    "26–40 yr (Adult)": 0, "> 40 yr (Older adult)": 0}
    for p in cohort:
        v = p["age_dx"]
        if v < 16:
            dx_age_tiers["< 16 yr (Paediatric)"] += 1
        elif v < 26:
            dx_age_tiers["16–25 yr (Young adult)"] += 1
        elif v <= 40:
            dx_age_tiers["26–40 yr (Adult)"] += 1
        else:
            dx_age_tiers["> 40 yr (Older adult)"] += 1

    # Kir6.2 GOF severity tiers
    katp_dist: dict = {}
    for p in cohort:
        katp_dist[p["katp_tier"]] = katp_dist.get(p["katp_tier"], 0) + 1

    # SU response distribution
    su_dist: dict = {}
    for p in cohort:
        su_dist[p["su_response"]] = su_dist.get(p["su_response"], 0) + 1

    # Disease stage distribution
    stage_dist: dict = {}
    for p in cohort:
        stage_dist[p["stage"]] = stage_dist.get(p["stage"], 0) + 1

    # Treatment distribution
    tx_dist: dict = {}
    for p in cohort:
        tx_dist[p["treatment"]] = tx_dist.get(p["treatment"], 0) + 1

    # Misdiagnosis distribution
    mis_dist: dict = {}
    for p in cohort:
        mis_dist[p["prior_misdiagnosis"]] = mis_dist.get(p["prior_misdiagnosis"], 0) + 1

    # BMI tiers
    bmi_tiers = {"< 20 (Underweight)": 0, "20–24.9 (Normal)": 0,
                 "25–29.9 (Overweight)": 0, "≥ 30 (Obese)": 0}
    for p in cohort:
        v = p["bmi"]
        if v < 20:
            bmi_tiers["< 20 (Underweight)"] += 1
        elif v < 25:
            bmi_tiers["20–24.9 (Normal)"] += 1
        elif v < 30:
            bmi_tiers["25–29.9 (Overweight)"] += 1
        else:
            bmi_tiers["≥ 30 (Obese)"] += 1

    # Fasting glucose tiers
    fg_tiers = {"5.5–7.0 mmol/L": 0, "7.1–10.0 mmol/L": 0,
                "10.1–14.0 mmol/L": 0, "> 14.0 mmol/L": 0}
    for p in cohort:
        v = p["fasting_glucose"]
        if v <= 7.0:
            fg_tiers["5.5–7.0 mmol/L"] += 1
        elif v <= 10.0:
            fg_tiers["7.1–10.0 mmol/L"] += 1
        elif v <= 14.0:
            fg_tiers["10.1–14.0 mmol/L"] += 1
        else:
            fg_tiers["> 14.0 mmol/L"] += 1

    # Disease duration tiers
    dur_tiers = {"< 2 yr": 0, "2–5 yr": 0, "6–15 yr": 0, "> 15 yr": 0}
    for p in cohort:
        v = p["disease_duration_yr"]
        if v < 2:
            dur_tiers["< 2 yr"] += 1
        elif v <= 5:
            dur_tiers["2–5 yr"] += 1
        elif v <= 15:
            dur_tiers["6–15 yr"] += 1
        else:
            dur_tiers["> 15 yr"] += 1

    summary_flags = {
        "pct_excellent_su": round(
            sum(1 for p in cohort if "Excellent" in p["su_response"]) / n * 100, 1),
        "pct_t1d_misdiagnosis": round(
            sum(1 for p in cohort if p["prior_misdiagnosis"] == "T1D") / n * 100, 1),
        "pct_antibody_negative": 100.0,
        "pct_family_hx_positive": round(
            sum(1 for p in cohort if p["family_history_positive"]) / n * 100, 1),
        "pct_c_pep_preserved": round(
            sum(1 for p in cohort if p["c_peptide_nmol_L"] >= 0.60) / n * 100, 1),
        "pct_dka_at_dx": round(
            sum(1 for p in cohort if p["dka_at_presentation"]) / n * 100, 1),
    }

    return {
        "variant_distribution":        var_dist,
        "ethnicity_distribution":       eth_dist,
        "hba1c_tiers":                  hba1c_tiers,
        "c_peptide_tiers":              cp_tiers,
        "age_at_diagnosis_tiers":       dx_age_tiers,
        "katp_gof_distribution":        katp_dist,
        "su_response_distribution":     su_dist,
        "disease_stage_distribution":   stage_dist,
        "treatment_distribution":       tx_dist,
        "misdiagnosis_distribution":    mis_dist,
        "bmi_tiers":                    bmi_tiers,
        "fasting_glucose_tiers":        fg_tiers,
        "disease_duration_tiers":       dur_tiers,
        "summary_flags":                summary_flags,
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "full_name": "MODY13 — KCNJ11-MODY (Maturity-Onset Diabetes of the Young Type 13)",
            "gene": "KCNJ11 — Kir6.2 (Inward-Rectifier K⁺ Channel 6.2); K-ATP pore subunit; 11p15.1; OMIM *600937",
            "disease_omim": "*600937 (phenotypic series; MODY13 / PNDM1 described under KCNJ11 locus)",
            "inheritance": "Autosomal Dominant — heterozygous activating (GOF) missense; 50% transmission",
            "prevalence": "~1–2% of MODY; underdiagnosed; European-enriched; clinically overlaps with MODY12",
            "mechanism": (
                "KCNJ11 GOF → Kir6.2 pore subunit mutation → reduced ATP affinity at pore ATP-binding site → "
                "K-ATP channel constitutively MORE open → at high glucose, ATP rise fails to close Kir6.2 pore → "
                "reduced Ca²⁺ influx → blunted GSIS → hyperglycaemia. Beta-cell mass intact → C-peptide PRESERVED."
            ),
            "protein_function": (
                "KCNJ11 (Kir6.2; 390 aa; ~43 kDa): inward-rectifier potassium channel; pore-forming subunit. "
                "Forms (Kir6.2)₄ tetrameric pore within K-ATP octamer. "
                "ATP binds directly to Kir6.2 N-terminal cytoplasmic domain (R50, T224 vicinity). "
                "GOF mutations reduce ATP IC₅₀ for channel closure. SU closes via SUR1 partner — NOT Kir6.2."
            ),
            "onset_age": "Variable; MODY-pattern: teens–adult (mean ~25–35 yr); neonatal if severe GOF (→ PNDM1)",
            "c_peptide_pattern": (
                "PRESERVED throughout clinical course. K-ATP pore gating defect only — no beta-cell structural loss. "
                "Unlike MODY10 (falling C-peptide from ER-stress apoptosis). Like MODY12 (ABCC8/SUR1 GOF)."
            ),
            "treatment": "SU first-line (~80–85% excellent response); insulin if SU insufficient; no DEND in mild GOF",
            "autoantibodies": "NEGATIVE (GADA, ZnT8, IA-2) — always; mandatory to exclude T1D/LADA",
            "family_history": "~65–72% positive (50% AD transmission); de novo 5–10%",
            "misdiagnosis_rate": (
                "T1D ~40% (most common — younger onset + DKA confounds). "
                "T2D ~22%; PNDM/neonatal DM ~6%"
            ),
        },

        "genes_and_proteins": {
            "KCNJ11 (*600937)": (
                "11p15.1. Kir6.2 (Inward-Rectifier K⁺ Channel 6.2). 390 aa, ~43 kDa. "
                "Pore-forming subunit of K-ATP octamer: (Kir6.2)₄·(SUR1)₄. "
                "Contains primary ATP-binding site (N-terminal cytoplasmic domain). "
                "GOF mutations reduce ATP affinity → pore stays open at high glucose. "
                "SU closes K-ATP via SUR1 NBD2 — NOT via Kir6.2 directly."
            ),
            "ABCC8 (*600509)": (
                "11p15.1 (adjacent to KCNJ11). SUR1 (Sulfonylurea Receptor 1). 1581 aa, ~177 kDa. "
                "Regulatory subunit of K-ATP octamer. MODY12: ABCC8 GOF → SUR1 NBD mutation. "
                "SU BINDING SITE: SUR1 NBD2 — SU closes K-ATP via this site (ATP-independently). "
                "This is why SU works for MODY13 even though Kir6.2 (not SUR1) is mutated."
            ),
            "K-ATP octamer (Kir6.2 + SUR1)": (
                "Hetero-octameric complex: 4 Kir6.2 pore subunits + 4 SUR1 regulatory subunits. "
                "MODY13 mutant Kir6.2 reduces ATP affinity → pore stays open. "
                "SU closes K-ATP by binding SUR1 NBD2 (allosteric closure) — bypasses Kir6.2 GOF. "
                "Diazoxide (K-ATP OPENER) would worsen MODY13 — CONTRAINDICATED."
            ),
        },

        "clinical_terms": {
            "MODY13": "Maturity-Onset Diabetes of the Young Type 13; KCNJ11 (Kir6.2) GOF; K-ATP pore subunit defect",
            "GOF (Gain-of-Function)": (
                "Activating mutation — channel MORE active (MORE open) than normal. In MODY13: Kir6.2 pore "
                "has reduced ATP affinity → stays open at high glucose → blunted GSIS. "
                "Opposite of LOF (loss-of-function in KCNJ11 = neonatal hyperinsulinism — rare)."
            ),
            "DEND / iDEND syndrome": (
                "Developmental delay, Epilepsy, Neonatal Diabetes (DEND) or intermediate DEND (iDEND). "
                "Caused ONLY by SEVERE KCNJ11 GOF (V59M, I296L, Q52R). Kir6.2 expressed in brain → "
                "severe neurological dysfunction. MODY13 MILD GOF (R201H, R201C): NO DEND. "
                "SU treatment can partially reverse neurological features in iDEND (high-dose glibenclamide)."
            ),
            "PNDM1-MODY13 spectrum": (
                "KCNJ11 GOF severity determines phenotype: severe GOF → PNDM1 (onset < 6 months; "
                "severe neurological features if DEND); moderate → TNDM (transient; remits; recurs adult MODY); "
                "mild GOF → MODY13 (teens–adult; standard SU; excellent response; no neurology). Same gene."
            ),
            "Kir6.2 ATP-binding defect": (
                "Kir6.2 N-terminal cytoplasmic domain binds ATP → channel closure. "
                "MODY13 GOF mutations near ATP-binding site reduce affinity (higher IC₅₀ for ATP). "
                "At high glucose: ATP rise insufficient to close mutant pore → GSIS blunted. "
                "SU overcomes by closing via SUR1 NBD2 (ATP-independent pathway)."
            ),
            "E23K (rs5219) GWAS vs MODY13": (
                "E23K in KCNJ11: common T2D GWAS variant; population frequency ~35%; very mild GOF; "
                "Hardy-Weinberg equilibrium. NOT autosomal dominant MODY. "
                "MODY13 variants (R201H, R201C): rare, pedigree-segregating, AD; genuine MODY phenotype. "
                "Clinical distinction: E23K = T2D risk modifier; R201H/C = MODY-causing GOF."
            ),
        },

        "lab_thresholds": {
            "C-peptide (MODY13)":        "≥ 0.60 nmol/L (fasting); preserved throughout course",
            "C-peptide (normal)":        "0.37–1.47 nmol/L; MODY13 within normal range",
            "Fasting glucose":           "Often 7–14 mmol/L in uncontrolled MODY13",
            "HbA1c target (SU therapy)": "< 7.0% (53 mmol/mol); monitor for hypoglycaemia at SU initiation",
            "Autoantibodies":            "GADA < 5 IU/mL; ZnT8-Ab negative; IA-2 negative (all MODY13)",
            "Kir6.2 ATP IC₅₀ (GOF)":    "GOF confirmed: ATP IC₅₀ > 200 µM (WT: ~10–50 µM) by patch-clamp",
            "86Rb⁺ efflux (GOF cutoff)": "> 130% WT activity (Kir6.2 + SUR1-WT cotransfection) = K-ATP GOF",
        },

        "treatment": {
            "sulfonylurea_first_line": (
                "SU (glibenclamide/gliclazide): binds SUR1 NBD2 → allosteric K-ATP closure → "
                "ATP-independent mechanism, fully bypasses Kir6.2 GOF ATP-binding defect. "
                "~80–85% achieve HbA1c < 7.0% — slightly lower than MODY12 (85–90%) because SU acts "
                "via SUR1 partner, not the mutated Kir6.2 subunit. Still EXCELLENT response."
            ),
            "insulin_to_su_switch": (
                "If diagnosed as T1D on insulin: confirm antibody-negative + C-peptide ≥ 0.6 nmol/L → "
                "KCNJ11 panel → if GOF confirmed (R201H, R201C), start low-dose glibenclamide, "
                "taper insulin over 2–4 weeks. Expect HbA1c normalisation; monitor for hypoglycaemia."
            ),
            "dka_management": (
                "DKA at first presentation (~15%): treat with standard IV insulin/fluids initially. "
                "Once metabolically stable: start SU — DKA in MODY13 is K-ATP GOF-mediated insulin "
                "insufficiency, NOT autoimmune beta-cell loss. SU resolves the underlying gating defect."
            ),
            "dose_titration": (
                "Start low: glibenclamide 0.5–1.25 mg bd (MODY13 patients SU-sensitive). "
                "Titrate to HbA1c < 7.0%. Monitor for hypoglycaemia, especially fasting. "
                "Gliclazide MR or glipizide: alternatives with lower hypoglycaemia risk in younger patients."
            ),
            "genetic_counselling": (
                "50% AD transmission; all first-degree relatives need KCNJ11 + ABCC8 sequencing + HbA1c. "
                "Ask for neonatal DM history in relatives — TNDM history + adult MODY = KCNJ11 GOF spectrum. "
                "Prenatal testing feasible if family planning; SU in utero (PNDM1 if severe GOF)."
            ),
        },

        "genetics_testing": {
            "KCNJ11_sequencing": (
                "Full coding sequence (exons 1–5 + splice sites); 11p15.1; 390 aa. "
                "Report activating (GOF) missense near ATP-binding site (N-terminal region; R50/T224 vicinity). "
                "GOF confirmed by patch-clamp ATP IC₅₀ assay — VUS not assumed GOF without functional data."
            ),
            "functional_validation": (
                "Patch-clamp (inside-out configuration): measure ATP IC₅₀ for K-ATP closure. "
                "GOF confirmed: IC₅₀ > 200 µM (normal < 50 µM). Glibenclamide IC₅₀ confirms SU correctability. "
                "86Rb⁺ efflux assay (Kir6.2-variant + SUR1-WT cotransfection) as secondary validation."
            ),
            "MODY_panel_requirement": (
                "KCNJ11 not in oldest 4–6-gene MODY panels. "
                "Request expanded MODY NGS panel including KCNJ11 + ABCC8 (K-ATP pair). "
                "Essential when younger onset + antibody-negative + DKA + C-peptide preserved."
            ),
            "cascade_screening": (
                "All first-degree relatives of confirmed MODY13. 50% carry GOF variant. "
                "Screen: KCNJ11 + ABCC8 sequencing + HbA1c + fasting glucose + C-peptide. "
                "Cascade benefit: pre-symptomatic SU initiation prevents DKA presentation."
            ),
            "abcc8_co_panel": (
                "Always co-sequence ABCC8 (MODY12) alongside KCNJ11. "
                "Both on 11p15.1; clinically indistinguishable (same K-ATP channel, different subunit). "
                "MODY12 and MODY13 require separate sequencing — co-mutation rare but reported."
            ),
        },

        "comparison_mody12_13": {
            "MODY12 (ABCC8/SUR1)": {
                "gene":      "ABCC8; 11p15.1; SUR1 regulatory subunit; 1581 aa",
                "mechanism": "SUR1 GOF → NBD mutation → K-ATP constitutively open → ↓ GSIS",
                "c_peptide": "PRESERVED — K-ATP gating defect only",
                "treatment": "SU 85–90%; SU binds SUR1 directly (mutated molecule = SU target)",
                "onset":     "Teens–adult (mean ~22–30 yr); neonatal if severe GOF (PNDM2)",
                "unique":    "PNDM2-TNDM-MODY12 severity spectrum; DKA at presentation ~20%",
            },
            "MODY13 (KCNJ11/Kir6.2)": {
                "gene":      "KCNJ11; 11p15.1; Kir6.2 pore subunit; 390 aa",
                "mechanism": "Kir6.2 GOF → pore ATP-binding reduced → K-ATP constitutively open",
                "c_peptide": "PRESERVED — pore gating defect only",
                "treatment": "SU 80–85%; SU closes via SUR1 partner (different subunit from mutation)",
                "onset":     "Adult (mean ~25–35 yr); R201H very mild; V59M neurological spectrum",
                "unique":    "DEND/iDEND neurological spectrum with severe variants (V59M, I296L, Q52R)",
            },
        },
    }
