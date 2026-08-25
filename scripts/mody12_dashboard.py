"""
MODY12 — ABCC8-MODY (Maturity-Onset Diabetes of the Young Type 12)
================================================================================
Gene       : ABCC8 (ATP-Binding Cassette Sub-family C Member 8; SUR1 — Sulfonylurea
             Receptor 1; regulatory subunit of the pancreatic K-ATP channel)
Chromosome : 11p15.1
OMIM Gene  : *600509
OMIM Dis.  : referenced under *600509 phenotypic series (MODY12)
Inheritance: Autosomal Dominant (heterozygous activating / GOF missense → MODY12)
Prevalence : Rare; ~1–2% of MODY; European-enriched early cohorts; underdiagnosed

Mechanism
---------
ABCC8 encodes SUR1 (Sulfonylurea Receptor 1), the regulatory (ABC-transporter)
subunit of the K-ATP channel octamer. The functional channel is a hetero-octamer:
(Kir6.2)₄ · (SUR1)₄ — Kir6.2 (KCNJ11, MODY13) forms the pore; SUR1 (ABCC8)
senses ATP/ADP and controls gating.

Normal GSIS:
  Glucose↑ → glycolysis → ATP↑ → SUR1 NBDs sense ATP/MgADP ratio →
  K-ATP closes → membrane depolarisation → Ca²⁺ influx → insulin exocytosis

MODY12 GOF mechanism:
  Activating ABCC8 missense (typically NBD1 or NBD2 domain) →
  SUR1 becomes constitutively MORE active (channel harder to close by ATP) →
  At high glucose, ATP rise FAILS to fully close K-ATP →
  Reduced Ca²⁺ influx → blunted GSIS → hyperglycaemia

KEY: C-peptide is PRESERVED — beta-cell mass structurally intact; the defect is
in K-ATP GATING (channel mechanics), not beta-cell apoptosis or transcription.

SUR1 NBD structure:
  NBD1 (Walker A/B motif): ATP binding; R1380 adjacent → founding mutation locus
  NBD2 (Walker A/B motif): MgADP sensing; critical for channel re-opening
  NBD1–NBD2 interface: dimerisation surface; mutations here impair ATP-driven closure

K-ATP Spectrum (ABCC8 GOF severity):
  Severe GOF → PNDM2 (neonatal; channel cannot close at birth; very high SU doses needed)
  Moderate GOF → TNDM (transient neonatal; resolves; diabetes recurs adulthood)
  Mild GOF → MODY12 (teens–adult onset; standard SU doses effective; EXCELLENT response)

SULFONYLUREA FIRST-LINE:
  SU (glibenclamide/gliclazide) binds SUR1 NBD2 DIRECTLY → closes K-ATP independently
  of ATP/ADP ratio — completely bypasses the GOF gating defect. This explains the
  exceptionally high SU response rate in MODY12 (~85–90%), the highest of all MODY types.

Key Founding Mutations (ABCC8 GOF, MODY12 phenotype)
------------------------------------------------------
* R1380L (c.4139G>T) — NBD2 Walker A adjacent; Babenko et al. 2006 NEJM (French family);
  first ABCC8 GOF → MODY-pattern; landmark mutation; mild GOF
* R1380C (c.4138C>T) — NBD2; similar position; moderate GOF
* K890T (c.2669A>C) — NBD1 Walker A; ATP-binding motif; Ellard 2007
* H1023R (c.3068A>G) — NBD1–NBD2 dimerisation interface; moderate GOF
* L1544P (c.4631T>C) — NBD2 C-terminal; rare; milder phenotype
* V187D (c.560T>A) — TMD1; affects SUR1 structural integrity; rare
* Novel_ABCC8_GOF — novel; functional K-ATP assay (86Rb+ efflux) mandatory

Differentiation from PNDM-ABCC8:
  MODY12: heterozygous mild GOF → standard SU dose → channel closable →
           EXCELLENT response; no neonatal presentation; teens–adult onset
  PNDM2 : heterozygous/biallelic severe GOF → channel barely closable →
           very high SU dose (0.4–0.8 mg/kg glibenclamide) needed; onset < 6 months

Comparison with MODY13 (KCNJ11 — Kir6.2):
  MODY12 (ABCC8/SUR1): regulatory subunit GOF; NBD ATP-sensing defect
  MODY13 (KCNJ11/Kir6.2): pore subunit GOF; channel ATP-binding site on pore
  Both → K-ATP stays open → reduced GSIS; both respond to SU by the same mechanism
  MODY12 unique: SU binds SUR1 directly (same molecule that is mutated)

Clinical Profile
----------------
* Onset: Variable; MODY-pattern teens–50s (mean 22–30 yr; earlier than MODY11)
* C-peptide: PRESERVED (K-ATP gating defect; no structural beta-cell loss)
* HbA1c: Progressive — severity correlates with degree of GOF
* Treatment: SU FIRST-LINE — exceptionally high response rate 85–90% (highest MODY)
* Autoantibodies: NEGATIVE (GADA, ZnT8, IA-2) — mandatory T1D/LADA exclusion
* Misdiagnosis: T1D most common (~45%) due to sometimes younger onset + DKA at
  presentation (K-ATP GOF can cause metabolic decompensation acutely)
* PNDM-MODY12 spectrum: Always ask for neonatal history and family history of
  transient neonatal hyperglycaemia — diagnostic clue bridging PNDM and MODY12
* No exocrine insufficiency (vs MODY8/CEL)
* No renal cysts (vs MODY5/HNF1B)
* No renal glycosuria (vs MODY3/HNF1A)
* No ER-stress or falling C-peptide (vs MODY10/INS)
* No pancreatic atrophy on imaging (vs MODY5/HNF1B)

Diagnostic Strategy
-------------------
* Suspect MODY12: young-to-mid onset DM, antibody-negative, family history,
  preserved C-peptide, EXCELLENT SU response, possibly DKA at presentation
* Test: ABCC8 gene sequencing; expanded MODY NGS panel (ABCC8 + KCNJ11 key)
* Functional validation: 86Rb+ efflux assay (K-ATP activity); patch-clamp; SUR1 NBD
  ATP-binding assay; COS-1/HEK293 co-transfection with Kir6.2 + SUR1-variant
* NOT in oldest MODY panels — ABCC8 described as MODY cause 2006; expanded panel essential
* Check ALL first-degree relatives; 50% AD transmission

Cohort: 40 patients, seed=325.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 325
_COHORT_SIZE = 40

# ABCC8 GOF variants — NBD1/NBD2/TMD domain activating missense
_VARIANTS = [
    "R1380L (c.4139G>T)",        # NBD2; Babenko 2006 NEJM; founding; mild GOF
    "R1380C (c.4138C>T)",        # NBD2; adjacent; moderate GOF
    "K890T (c.2669A>C)",         # NBD1 Walker A; Ellard 2007; ATP-binding
    "H1023R (c.3068A>G)",        # NBD1-NBD2 interface; moderate GOF
    "L1544P (c.4631T>C)",        # NBD2 C-terminal; rare; milder
    "V187D (c.560T>A)",          # TMD1; structural; rare
    "Novel_ABCC8_GOF",           # novel; 86Rb+ assay mandatory
    "Splice_ABCC8",              # splice site; partial GOF isoform; rare
]
_VARIANT_WEIGHTS = [0.30, 0.22, 0.18, 0.12, 0.07, 0.04, 0.05, 0.02]

_ETHNICITIES = [
    "European-French",
    "European-UK/Irish",
    "European-Other",
    "North American European",
    "Asian",
    "Other/Unknown",
]
_ETHNICITY_WEIGHTS = [0.30, 0.22, 0.18, 0.16, 0.08, 0.06]

_TREATMENTS = [
    "Sulfonylurea (monotherapy)",
    "Sulfonylurea + Metformin",
    "Insulin → switched to SU",
    "Sulfonylurea + Insulin (transitional)",
    "Metformin monotherapy",
    "Lifestyle / diet",
]
_TX_WEIGHTS = [0.48, 0.22, 0.18, 0.06, 0.04, 0.02]

_MISDIAGNOSES = [
    "T1D",
    "T2D",
    "PNDM / neonatal DM",
    "Prediabetes",
    "None (correctly diagnosed)",
]
_MISDIAG_WEIGHTS = [0.45, 0.20, 0.08, 0.08, 0.19]

_DISEASE_STAGES = [
    "Early (HbA1c 5.8–7.4%)",
    "Moderate (HbA1c 7.5–8.9%)",
    "Advanced (HbA1c ≥ 9.0%)",
]
_STAGE_WEIGHTS = [0.45, 0.35, 0.20]

# K-ATP channel GOF severity tiers (degree of GOF determines MODY12 vs PNDM boundary)
_KATP_GOF_TIERS = [
    "Mild GOF (K-ATP closes with standard SU dose)",
    "Mild-Moderate GOF (responds well, slightly higher SU dose needed)",
    "Moderate GOF (close to PNDM boundary; high SU dose; excellent response)",
    "Near-PNDM boundary (TNDM history or very early onset)",
]
_KATP_WEIGHTS = [0.40, 0.32, 0.18, 0.10]

# SU response categories
_SU_RESPONSE = [
    "Excellent (HbA1c < 7.0% on SU alone)",
    "Good (HbA1c 7.0–7.9% on SU alone)",
    "Partial (requires SU + Metformin)",
    "Insufficient (insulin required despite SU)",
]
_SU_RESPONSE_WEIGHTS = [0.58, 0.22, 0.12, 0.08]


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

        # Age at diagnosis: MODY12 earlier than MODY11 (mean ~22–30 yr)
        dx_age = round(rng.gauss(25, 9), 1)
        dx_age = max(0.5, min(55, dx_age))  # can include neonatal/transient cases

        # HbA1c: correlates with stage
        if "Early" in stage:
            hba1c = round(rng.gauss(6.7, 0.5), 1)
            hba1c = max(5.8, min(7.4, hba1c))
        elif "Moderate" in stage:
            hba1c = round(rng.gauss(8.1, 0.5), 1)
            hba1c = max(7.5, min(8.9, hba1c))
        else:
            hba1c = round(rng.gauss(9.8, 0.7), 1)
            hba1c = max(9.0, min(13.0, hba1c))

        # C-peptide: PRESERVED (K-ATP gating defect; no structural beta-cell loss)
        c_peptide = round(rng.gauss(0.82, 0.22), 2)
        c_peptide = max(0.45, min(1.55, c_peptide))

        # BMI: relatively normal (younger onset; less T2D overlap than MODY11)
        bmi = round(rng.gauss(24.5, 4.0), 1)
        bmi = max(16, min(40, bmi))

        family_hx = rng.random() < 0.72

        # Disease duration since diagnosis
        duration = round(rng.gauss(7.0, 5.5), 1)
        duration = max(0.3, min(28, duration))

        # Fasting glucose (blunted GSIS → moderately elevated fasting)
        fasting_glucose = round(rng.gauss(9.5, 2.8), 1)
        fasting_glucose = max(5.5, min(18.0, fasting_glucose))

        # DKA at presentation (K-ATP GOF can cause acute decompensation)
        dka_at_dx = rng.random() < 0.22

        # Neonatal history (spectrum: some MODY12 patients had transient neonatal DM)
        neonatal_hx = "Near-PNDM" in katp_tier and rng.random() < 0.55

        cohort.append({
            "id": i + 1,
            "variant": variant,
            "ethnicity": ethnicity,
            "treatment": treatment,
            "prior_misdiagnosis": prior_misdiag,
            "disease_stage": stage,
            "katp_gof_tier": katp_tier,
            "su_response": su_response,
            "age_at_diagnosis": dx_age,
            "hba1c_pct": hba1c,
            "c_peptide_nmol_L": c_peptide,
            "bmi": bmi,
            "family_history_positive": family_hx,
            "disease_duration_yr": duration,
            "fasting_glucose_mmol_L": fasting_glucose,
            "dka_at_presentation": dka_at_dx,
            "neonatal_dm_history": neonatal_hx,
            "autoantibodies": "Negative",
            "exocrine_insufficiency": False,
            "renal_cysts": False,
            "renal_glycosuria": False,
            "er_stress_marker": False,
            "pancreatic_atrophy": False,
        })
    return cohort


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    cohort = _build_cohort()
    n = _COHORT_SIZE

    mean_hba1c    = round(statistics.mean(p["hba1c_pct"] for p in cohort), 2)
    mean_cp       = round(statistics.mean(p["c_peptide_nmol_L"] for p in cohort), 3)
    mean_dx_age   = round(statistics.mean(p["age_at_diagnosis"] for p in cohort), 1)
    mean_bmi      = round(statistics.mean(p["bmi"] for p in cohort), 1)
    mean_fasting  = round(statistics.mean(p["fasting_glucose_mmol_L"] for p in cohort), 1)

    pct_su = round(
        sum(1 for p in cohort if "Sulfonylurea" in p["treatment"]) / n * 100, 1)
    pct_excellent_su = round(
        sum(1 for p in cohort if "Excellent" in p["su_response"]) / n * 100, 1)
    pct_insulin = round(
        sum(1 for p in cohort if "Insulin" in p["treatment"]) / n * 100, 1)
    pct_misdiag = round(
        sum(1 for p in cohort if p["prior_misdiagnosis"] != "None (correctly diagnosed)") / n * 100, 1)
    pct_fam_hx = round(
        sum(1 for p in cohort if p["family_history_positive"]) / n * 100, 1)
    pct_t1d_misdiag = round(
        sum(1 for p in cohort if p["prior_misdiagnosis"] == "T1D") / n * 100, 1)
    pct_dka = round(
        sum(1 for p in cohort if p["dka_at_presentation"]) / n * 100, 1)
    pct_neonatal_hx = round(
        sum(1 for p in cohort if p["neonatal_dm_history"]) / n * 100, 1)

    kpis = {
        "gene":                  "ABCC8 (SUR1)",
        "chromosome":            "11p15.1",
        "omim_gene":             "*600509",
        "omim_disease":          "*600509 (phenotypic series)",
        "cohort_size":           n,
        "mean_hba1c":            f"{mean_hba1c}%",
        "mean_c_peptide":        f"{mean_cp} nmol/L (preserved)",
        "mean_dx_age":           f"{mean_dx_age} yr",
        "mean_bmi":              f"{mean_bmi} kg/m²",
        "mean_fasting_glucose":  f"{mean_fasting} mmol/L",
        "pct_su_response":       f"{pct_su}%",
        "pct_excellent_su":      f"{pct_excellent_su}% (highest of all MODY types)",
        "pct_insulin_required":  f"{pct_insulin}%",
        "pct_misdiagnosed":      f"{pct_misdiag}%",
        "pct_t1d_misdiag":       f"{pct_t1d_misdiag}% (T1D most common error)",
        "pct_family_hx":         f"{pct_fam_hx}%",
        "pct_antibody_neg":      "100%",
        "pct_dka_at_dx":         f"{pct_dka}%",
        "pct_neonatal_hx":       f"{pct_neonatal_hx}%",
        "inheritance":           "Autosomal Dominant",
        "mechanism_class":       "K-ATP channel GOF (ABCC8/SUR1 activating missense)",
        "c_peptide_pattern":     "PRESERVED — K-ATP gating defect, not beta-cell loss",
        "prevalence":            "~1–2% MODY; underdiagnosed; European-enriched early cohorts",
        "treatment_first_line":  "Sulfonylurea (85–90% excellent response — highest of all MODY)",
    }

    key_facts = [
        "ABCC8 encodes SUR1 (Sulfonylurea Receptor 1) — regulatory subunit of K-ATP channel octamer (Kir6.2)₄·(SUR1)₄",
        "MODY12 GOF: SUR1 NBD mutations → K-ATP stays open at high glucose → reduced Ca²⁺ influx → blunted GSIS",
        "C-peptide PRESERVED — defect is in K-ATP gating (channel mechanics), not beta-cell structural loss",
        "SU binds SUR1 DIRECTLY at NBD2 → closes K-ATP bypassing ATP ratio → exceptionally high response rate (85–90%)",
        "Spectrum: severe GOF → PNDM2 (neonatal); moderate → TNDM (transient neonatal); mild → MODY12 (teens–adult)",
        "MODY12 vs MODY13 (KCNJ11): both K-ATP subunit GOF; ABCC8=regulatory SUR1, KCNJ11=pore Kir6.2; both SU-responsive",
        "T1D misdiagnosis ~45% — younger onset + occasional DKA at presentation triggers autoimmune workup",
        "DKA can occur at diagnosis in MODY12 (~20%) — K-ATP GOF causes acute insulin insufficiency; SU resolves it",
        "Family history 65–75%; neonatal DM history in some — check ALL first-degree relatives with ABCC8 sequencing",
        "NOT in oldest MODY panels (ABCC8 as MODY cause: Babenko 2006 NEJM); expanded NGS panel mandatory",
        "Functional validation: 86Rb+ efflux assay (K-ATP activity); patch-clamp; co-transfection with Kir6.2",
        "No exocrine, renal, neurological, or structural organ phenotype — differentiates from MODY5, 8",
    ]

    alerts = {
        "dka_trap": (
            "DKA at presentation in ~20% — do NOT assume T1D. Check: antibody-negative + family history + "
            "K-ATP gene panel IMMEDIATELY. SU resolves DKA; continuing insulin long-term is wrong management."
        ),
        "pndm_spectrum": (
            "PNDM2-MODY12 continuum: always ask for neonatal/infancy hyperglycaemia in patient or family. "
            "Transient neonatal DM that remitted → recurred as adult MODY = ABCC8 GOF diagnosis clue."
        ),
        "su_response_highest": (
            "SU response 85–90% — the HIGHEST of all MODY types. SU binds SUR1 directly. "
            "If patient on insulin: trial SU first (confirm antibody-negative + C-peptide preserved). "
            "Expected outcome: off insulin within weeks."
        ),
        "panel_gap": (
            "ABCC8 absent from oldest MODY panels (4-gene: HNF1A/HNF4A/GCK/HNF1B). "
            "Request expanded NGS panel including ABCC8 + KCNJ11 when K-ATP mechanism suspected."
        ),
        "functional_validation": (
            "Novel ABCC8 variants: 86Rb+ efflux assay in COSm6 cells co-transfected with Kir6.2+ABCC8-variant. "
            "GOF confirmed if channel activity > 130% WT. Patch-clamp: reduced ATP-sensitivity (higher IC₅₀)."
        ),
    }

    patients_preview = [
        {
            "id":          p["id"],
            "variant":     p["variant"],
            "age_dx":      p["age_at_diagnosis"],
            "hba1c":       p["hba1c_pct"],
            "c_peptide":   p["c_peptide_nmol_L"],
            "treatment":   p["treatment"],
            "stage":       p["disease_stage"],
            "family_hx":   p["family_history_positive"],
            "dka_at_dx":   p["dka_at_presentation"],
        }
        for p in cohort[:12]
    ]

    return {
        "dashboard":   "MODY12 — ABCC8-MODY (SUR1 K-ATP GOF)",
        "cohort_size": n,
        "kpis":        kpis,
        "key_facts":   key_facts,
        "alerts":      alerts,
        "patients":    patients_preview,
    }


def get_breakdown() -> dict:
    cohort = _build_cohort()
    n = _COHORT_SIZE

    # Variant distribution
    var_dist = {}
    for p in cohort:
        var_dist[p["variant"]] = var_dist.get(p["variant"], 0) + 1

    # Ethnicity distribution
    eth_dist = {}
    for p in cohort:
        eth_dist[p["ethnicity"]] = eth_dist.get(p["ethnicity"], 0) + 1

    # HbA1c tiers
    hba1c_tiers = {"< 7.0%": 0, "7.0–7.9%": 0, "8.0–8.9%": 0, "≥ 9.0%": 0}
    for p in cohort:
        v = p["hba1c_pct"]
        if v < 7.0:
            hba1c_tiers["< 7.0%"] += 1
        elif v < 8.0:
            hba1c_tiers["7.0–7.9%"] += 1
        elif v < 9.0:
            hba1c_tiers["8.0–8.9%"] += 1
        else:
            hba1c_tiers["≥ 9.0%"] += 1

    # C-peptide tiers (PRESERVED pattern)
    cp_tiers = {
        "< 0.40 nmol/L":     0,
        "0.40–0.59 nmol/L":  0,
        "0.60–0.99 nmol/L":  0,
        "≥ 1.00 nmol/L":     0,
    }
    for p in cohort:
        v = p["c_peptide_nmol_L"]
        if v < 0.40:
            cp_tiers["< 0.40 nmol/L"] += 1
        elif v < 0.60:
            cp_tiers["0.40–0.59 nmol/L"] += 1
        elif v < 1.00:
            cp_tiers["0.60–0.99 nmol/L"] += 1
        else:
            cp_tiers["≥ 1.00 nmol/L"] += 1

    # Age at diagnosis tiers (earlier than MODY11)
    dx_age_tiers = {"< 5 yr": 0, "5–14 yr": 0, "15–24 yr": 0, "25–34 yr": 0, "≥ 35 yr": 0}
    for p in cohort:
        a = p["age_at_diagnosis"]
        if a < 5:
            dx_age_tiers["< 5 yr"] += 1
        elif a < 15:
            dx_age_tiers["5–14 yr"] += 1
        elif a < 25:
            dx_age_tiers["15–24 yr"] += 1
        elif a < 35:
            dx_age_tiers["25–34 yr"] += 1
        else:
            dx_age_tiers["≥ 35 yr"] += 1

    # K-ATP GOF tier distribution
    katp_dist = {}
    for p in cohort:
        katp_dist[p["katp_gof_tier"]] = katp_dist.get(p["katp_gof_tier"], 0) + 1

    # SU response distribution
    su_dist = {}
    for p in cohort:
        su_dist[p["su_response"]] = su_dist.get(p["su_response"], 0) + 1

    # Disease stage distribution
    stage_dist = {}
    for p in cohort:
        stage_dist[p["disease_stage"]] = stage_dist.get(p["disease_stage"], 0) + 1

    # Treatment distribution
    tx_dist = {}
    for p in cohort:
        tx_dist[p["treatment"]] = tx_dist.get(p["treatment"], 0) + 1

    # Misdiagnosis distribution
    mis_dist = {}
    for p in cohort:
        mis_dist[p["prior_misdiagnosis"]] = mis_dist.get(p["prior_misdiagnosis"], 0) + 1

    # BMI tiers
    bmi_tiers = {"< 20": 0, "20–22.9": 0, "23–24.9": 0, "25–29.9": 0, "≥ 30": 0}
    for p in cohort:
        b = p["bmi"]
        if b < 20:
            bmi_tiers["< 20"] += 1
        elif b < 23:
            bmi_tiers["20–22.9"] += 1
        elif b < 25:
            bmi_tiers["23–24.9"] += 1
        elif b < 30:
            bmi_tiers["25–29.9"] += 1
        else:
            bmi_tiers["≥ 30"] += 1

    # Fasting glucose tiers
    fg_tiers = {
        "< 7.0 mmol/L":    0,
        "7.0–9.9 mmol/L":  0,
        "10–12.9 mmol/L":  0,
        "≥ 13 mmol/L":     0,
    }
    for p in cohort:
        g = p["fasting_glucose_mmol_L"]
        if g < 7.0:
            fg_tiers["< 7.0 mmol/L"] += 1
        elif g < 10.0:
            fg_tiers["7.0–9.9 mmol/L"] += 1
        elif g < 13.0:
            fg_tiers["10–12.9 mmol/L"] += 1
        else:
            fg_tiers["≥ 13 mmol/L"] += 1

    # Disease duration tiers
    dur_tiers = {}
    for p in cohort:
        d = p["disease_duration_yr"]
        if d < 3:
            key = "< 3 yr"
        elif d < 7:
            key = "3–6 yr"
        elif d < 12:
            key = "7–11 yr"
        else:
            key = "≥ 12 yr"
        dur_tiers[key] = dur_tiers.get(key, 0) + 1

    # Summary flags
    summary_flags = {
        "pct_su_response":      round(
            sum(1 for p in cohort if "Sulfonylurea" in p["treatment"]) / n * 100, 1),
        "pct_excellent_su":     round(
            sum(1 for p in cohort if "Excellent" in p["su_response"]) / n * 100, 1),
        "pct_insulin_required": round(
            sum(1 for p in cohort if "Insulin" in p["treatment"]) / n * 100, 1),
        "pct_t1d_misdiagnosis": round(
            sum(1 for p in cohort if p["prior_misdiagnosis"] == "T1D") / n * 100, 1),
        "pct_antibody_negative": 100.0,
        "pct_family_hx_positive": round(
            sum(1 for p in cohort if p["family_history_positive"]) / n * 100, 1),
        "pct_c_pep_preserved":  round(
            sum(1 for p in cohort if p["c_peptide_nmol_L"] >= 0.60) / n * 100, 1),
        "pct_dka_at_dx":        round(
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
            "full_name": "MODY12 — ABCC8-MODY (Maturity-Onset Diabetes of the Young Type 12)",
            "gene": "ABCC8 — SUR1 (Sulfonylurea Receptor 1); K-ATP regulatory subunit; 11p15.1; OMIM *600509",
            "disease_omim": "*600509 (phenotypic series; MODY12 described under ABCC8 locus)",
            "inheritance": "Autosomal Dominant — heterozygous activating (GOF) missense; 50% transmission",
            "prevalence": "~1–2% of MODY; underdiagnosed; European-enriched early cohorts; globally present",
            "mechanism": (
                "ABCC8 GOF → SUR1 NBD mutation → K-ATP channel constitutively MORE active (harder to close) → "
                "at high glucose, ATP rise fails to fully close K-ATP → reduced Ca²⁺ influx → blunted GSIS → "
                "hyperglycaemia. Beta-cell mass structurally intact → C-peptide PRESERVED."
            ),
            "protein_function": (
                "ABCC8 (SUR1; 1581 aa; ~177 kDa): ABC-transporter family; regulatory subunit of K-ATP octamer. "
                "Contains TMD0, TMD1, NBD1, TMD2, NBD2. NBD1 binds ATP (stimulatory); NBD2 senses MgADP "
                "(inhibitory — prevents K-ATP re-opening). GOF mutations in NBD1/NBD2 reduce ATP-sensitivity → "
                "channel stays open. SU (sulfonylurea) binds SUR1 NBD2 directly → closes K-ATP ATP-independently."
            ),
            "onset_age": "Variable; MODY-pattern: teens to adult (mean ~22–30 yr); neonatal if severe GOF (→ PNDM2)",
            "c_peptide_pattern": (
                "PRESERVED throughout clinical course. K-ATP gating defect — no beta-cell structural loss. "
                "Unlike MODY10 (falling C-peptide from ER-stress apoptosis). Like MODY13 (KCNJ11/Kir6.2 GOF)."
            ),
            "treatment": "SU first-line (85–90% excellent response — highest of all MODY types); insulin if SU insufficient",
            "autoantibodies": "NEGATIVE (GADA, ZnT8, IA-2) — always; mandatory to exclude T1D/LADA",
            "family_history": "65–75% positive (50% AD transmission); de novo 5–10%",
            "misdiagnosis_rate": (
                "T1D ~45% (most common — younger onset + DKA confounds). "
                "T2D ~20%; PNDM/neonatal DM ~8%"
            ),
        },

        "genes_and_proteins": {
            "ABCC8 (*600509)": (
                "11p15.1. SUR1 (Sulfonylurea Receptor 1). 1581 aa, ~177 kDa. ABC-transporter superfamily. "
                "Regulatory subunit of K-ATP octamer: (Kir6.2)₄·(SUR1)₄. "
                "NBD1 (Walker A: K890 region): ATP-binding, stimulatory. "
                "NBD2 (Walker A: R1380 region): MgADP sensing, inhibitory (prevents re-opening). "
                "SU BINDING SITE: NBD2 region — explains direct SU channel closure."
            ),
            "KCNJ11 (*600937)": (
                "11p15.1 (adjacent to ABCC8). Kir6.2 (inward-rectifier potassium channel 6.2). "
                "Pore-forming subunit of K-ATP. Forms (Kir6.2)₄ tetrameric pore. "
                "ATP binds directly to Kir6.2 N-terminus → induces channel closure. "
                "MODY13: KCNJ11 GOF (R201H, V59M mild); MODY12: ABCC8 GOF — same channel, different subunit."
            ),
            "K-ATP octamer (Kir6.2 + SUR1)": (
                "Hetero-octameric complex: 4 Kir6.2 pore subunits + 4 SUR1 regulatory subunits. "
                "MODY12 mutant SUR1 changes ATP-sensing → channel harder to close. "
                "SU closes by binding SUR1 directly (ATP-independent). "
                "Diazoxide (K-ATP OPENER) would worsen MODY12 — CONTRAINDICATED."
            ),
        },

        "clinical_terms": {
            "MODY12": "Maturity-Onset Diabetes of the Young Type 12; ABCC8 (SUR1) GOF; K-ATP regulatory subunit defect",
            "GOF (Gain-of-Function)": (
                "Activating mutation — channel MORE active than normal. In MODY12: K-ATP harder to close → "
                "reduced insulin secretion at high glucose. Opposite of LOF (loss-of-function in ABCC8 = HHF1 "
                "hyperinsulinism — channel can't open → continuous insulin secretion)."
            ),
            "PNDM2-MODY12 spectrum": (
                "ABCC8 GOF severity determines phenotype: severe GOF → PNDM2 (onset < 6 months; channel "
                "barely closable); moderate → TNDM (transient neonatal DM; remits; recurs adult MODY pattern); "
                "mild → MODY12 (teens–adult; standard SU doses effective). Same gene, severity gradient."
            ),
            "K-ATP gating defect": (
                "K-ATP channel gating = balance between ATP (closes) and MgADP (re-opens). "
                "MODY12: NBD mutation shifts equilibrium toward open state. SU overrides this by "
                "binding SUR1 directly — explains why SU is so effective (85–90% response rate)."
            ),
            "86Rb⁺ efflux assay": (
                "Standard functional assay for K-ATP GOF. Rub rubidium-86 (potassium surrogate) efflux "
                "measured from COSm6 cells co-transfected with Kir6.2 + ABCC8 variant. GOF: increased "
                "efflux at baseline; confirmed GOF if > 130% WT; inhibited by standard glibenclamide dose."
            ),
            "MODY12 vs MODY13": (
                "MODY12 (ABCC8/SUR1): regulatory subunit GOF; NBD mutations; R1380 most common. "
                "MODY13 (KCNJ11/Kir6.2): pore subunit GOF; R201H mild adult MODY. "
                "Both → K-ATP constitutively open; both respond to SU; clinically similar. "
                "Distinguish by gene panel; mechanism nearly identical."
            ),
        },

        "lab_thresholds": {
            "C-peptide (MODY12)":        "≥ 0.60 nmol/L (fasting); preserved throughout course",
            "C-peptide (normal)":        "0.37–1.47 nmol/L; MODY12 within normal range",
            "Fasting glucose":           "Often 7–14 mmol/L in uncontrolled MODY12",
            "HbA1c target (SU therapy)": "< 7.0% (53 mmol/mol); hypoglycaemia risk — low starting dose",
            "Autoantibodies":            "GADA < 5 IU/mL; ZnT8-Ab negative; IA-2 negative (all MODY12)",
            "86Rb+ efflux (GOF cutoff)": "> 130% WT activity = confirmed K-ATP GOF",
            "Glibenclamide IC₅₀ shift":  "MODY12 channel: higher IC₅₀ for ATP vs WT; pharmacological SU overcomes",
        },

        "treatment": {
            "sulfonylurea_first_line": (
                "SU (glibenclamide/gliclazide/glipizide): binds SUR1 NBD2 DIRECTLY → closes K-ATP "
                "without needing ATP rise. Completely bypasses the GOF gating defect. "
                "85–90% achieve HbA1c < 7.0% — highest response rate of all MODY types. "
                "Switch from insulin to SU expected to succeed in antibody-negative, C-peptide-preserved patients."
            ),
            "insulin_to_su_switch": (
                "If diagnosed as T1D on insulin: confirm antibody-negative + C-peptide ≥ 0.6 nmol/L → "
                "ABCC8 panel → if GOF confirmed, start low-dose glibenclamide, taper insulin over 2–4 weeks. "
                "Expect normalisation HbA1c; risk hypoglycaemia during transition (monitor closely)."
            ),
            "dka_management": (
                "DKA at first presentation: treat with standard IV insulin fluids initially. "
                "Once stable: start SU — this is CRITICAL; DKA in MODY12 is not from autoimmune beta-cell loss "
                "but from K-ATP GOF causing acute insulin insufficiency. SU resolves the underlying defect."
            ),
            "dose_titration": (
                "Start low: glibenclamide 0.5–1.25 mg bd (MODY12 patients are SU-sensitive). "
                "Titrate to HbA1c < 7.0%. Monitor for hypoglycaemia, especially fasting. "
                "Glipizide or gliclazide MR alternatives (lower hypoglycaemia risk in younger patients)."
            ),
            "genetic_counselling": (
                "50% AD transmission; all first-degree relatives need ABCC8 sequencing + HbA1c + fasting glucose. "
                "Ask for neonatal DM history in relatives — TNDM history + adult MODY = ABCC8 GOF spectrum. "
                "Prenatal testing feasible if family planning; SU in utero avoids PNDM if severe GOF."
            ),
        },

        "genetics_testing": {
            "ABCC8_sequencing": (
                "Full coding sequence (exons 1–39 + splice sites); 11p15.1; 1581 aa. "
                "Report activating (GOF) missense variants especially in NBD1/NBD2. "
                "GOF confirmed by 86Rb+ efflux assay — VUS not assumed GOF without functional data."
            ),
            "functional_validation": (
                "86Rb+ efflux assay (COSm6 + Kir6.2 + ABCC8-variant): GOF = > 130% WT efflux. "
                "Patch-clamp (inside-out): reduced ATP-sensitivity (higher IC₅₀ for channel closure). "
                "Glibenclamide IC₅₀ by patch-clamp confirms pharmacological correctability."
            ),
            "MODY_panel_requirement": (
                "ABCC8 absent from oldest 4-gene or 6-gene MODY panels. "
                "Request expanded MODY NGS panel including ABCC8 + KCNJ11. "
                "Especially important: K-ATP genes when younger onset + DKA + antibody-negative."
            ),
            "cascade_screening": (
                "All first-degree relatives of confirmed MODY12. 50% carry GOF variant. "
                "Screen: ABCC8 sequencing + HbA1c + fasting glucose + C-peptide. "
                "Cascade benefit: pre-symptomatic SU initiation prevents DKA presentation."
            ),
            "kcnj11_co_panel": (
                "Always sequence KCNJ11 alongside ABCC8 in suspected K-ATP MODY. "
                "Both subunits on 11p15.1 (adjacent genes). Same clinical presentation. "
                "MODY12 (ABCC8) and MODY13 (KCNJ11) require separate sequencing — not the same variant."
            ),
        },

        "comparison_mody12_13": {
            "MODY12 (ABCC8/SUR1)": {
                "gene":      "ABCC8; 11p15.1; SUR1 regulatory subunit",
                "mechanism": "SUR1 GOF → NBD mutation → K-ATP constitutively open → ↓ GSIS",
                "c_peptide": "PRESERVED — K-ATP gating defect only",
                "treatment": "SU 85–90%; SU binds SUR1 directly (same molecule mutated)",
                "onset":     "Teens–adult (mean ~22–30 yr); neonatal if severe GOF (PNDM2)",
                "unique":    "PNDM2-TNDM-MODY12 severity spectrum; DKA at presentation ~20%",
            },
            "MODY13 (KCNJ11/Kir6.2)": {
                "gene":      "KCNJ11; 11p15.1; Kir6.2 pore subunit",
                "mechanism": "Kir6.2 GOF → pore subunit ATP-binding reduced → K-ATP constitutively open",
                "c_peptide": "PRESERVED — pore gating defect only",
                "treatment": "SU 80–85%; SU closes via SUR1 partner (same channel, different subunit mutated)",
                "onset":     "Adult (mean ~25–35 yr); R201H very mild; V59M neurological spectrum",
                "unique":    "DEND/iDEND neurological spectrum with severe Kir6.2 variants (V59M, I296L)",
            },
        },
    }
