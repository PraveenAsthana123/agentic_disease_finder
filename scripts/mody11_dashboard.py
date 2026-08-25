"""
MODY11 — BLK-MODY (Maturity-Onset Diabetes of the Young Type 11)
================================================================================
Gene       : BLK (B Lymphocyte Kinase; B-Lymphoid Tyrosine Kinase)
Chromosome : 8p23.1
OMIM Gene  : *191305
OMIM Dis.  : #613375  (MODY11)
Inheritance: Autosomal Dominant (heterozygous hypomorphic LOF → MODY11)
Prevalence : Rare; ~1% of MODY; European/French-enriched; underdiagnosed globally

Mechanism
---------
BLK encodes a non-receptor Src-family tyrosine kinase primarily known for B-lymphocyte
signalling, but unexpectedly expressed in pancreatic beta cells. In beta cells BLK
plays a dual role:

1. PDX1 PHOSPHORYLATION: BLK phosphorylates the MODY4 master transcription factor
   PDX1 → enhanced PDX1 nuclear localisation → higher insulin gene transcription →
   more insulin biosynthesis. BLK haploinsufficiency → hypophosphorylated PDX1 →
   reduced insulin mRNA and protein content.

2. KATP-INDEPENDENT AMPLIFICATION: BLK participates in the cAMP/PKA amplifying arm
   of GSIS (glucose-stimulated insulin secretion) — the KATP-channel-independent
   pathway that accounts for ~30% of second-phase insulin release. BLK LOF blunts
   this amplification specifically at high glucose concentrations.

MODY11 PATHOMECHANISM:
  Heterozygous BLK hypomorphic missense → reduced (not absent) kinase activity →
  ↓ PDX1 phosphorylation → ↓ insulin synthesis + ↓ KATP-independent amplification →
  impaired second-phase GSIS → post-prandial hyperglycaemia → progressive HbA1c rise

KEY: MODY11 is a KINASE-SIGNALLING DEFECT (not a TF haploinsufficiency, not an
enzyme mutation, not a structural protein problem). C-peptide is PRESERVED because
beta-cell mass is structurally intact — the defect is in GSIS amplification.

Key Founding Mutations
----------------------
* A71T (c.211G>A) — signal-transduction subdomain; BLK Src-homology 2 domain;
  hypomorphic; Borowiec et al. 2009 PNAS (French cohort); reduces kinase activity ~50%
* P489L (c.1466C>T) — kinase catalytic domain; activation loop adjacent;
  Borowiec 2009 PNAS; reduces kinase activity ~60%; stronger phenotype than A71T
* K469N (c.1407G>C) — kinase domain; ATP-binding pocket adjacent; rare European
* E313K (c.937G>A) — SH2-kinase linker; French/UK families; mildest phenotype
* Novel_hypomorphic_BLK — reduced but not absent kinase activity; functional assay
  needed (BLK kinase assay + PDX1 phospho-immunoblot)

BLK and T2D GWAS
-----------------
The common BLK promoter variant rs922879 (minor allele frequency ~15%) is a confirmed
T2D GWAS signal, associated with modestly reduced BLK expression. MODY11 represents
the rare high-penetrance extreme of the same biological axis: rare coding hypomorphs
with strong beta-cell BLK LOF cause MODY11; common non-coding variants with mild
BLK reduction contribute to T2D risk. This convergence makes BLK unusual in bridging
the common-variant and rare-variant T2D genetics literature.

Clinical Profile
----------------
* Onset: Teens to 50s (mean 35–45 yr — the LATEST mean onset of described MODY types)
* C-peptide: PRESERVED (kinase-signalling defect, not structural; beta-cell mass intact)
* HbA1c: Progressive — not stable (unlike MODY2 GCK)
* Fasting glucose: often mildly elevated; post-prandial excursions more prominent
  (second-phase GSIS blunted selectively; first-phase partially preserved)
* Treatment: SU first-line (60–70%); closes K-ATP channels, amplifies first-phase
  release; may partially compensate for lost KATP-independent amplification. Some
  progress to insulin; metformin adjunct useful (BMI often higher, T2D overlap)
* Autoantibodies: NEGATIVE (GADA, ZnT8, IA-2) — mandatory T2D/T1D exclusion
* Misdiagnosis: T2D most common (~50%) due to late onset + BMI overlap + absence
  of distinctive organ features (no renal cysts, no exocrine disease, no glycosuria)
  LADA misdiagnosis ~8% (late onset + thin patients trigger autoimmune workup)
* Family history: 65–75%
* No exocrine insufficiency (vs MODY8/CEL)
* No renal cysts or Mullerian anomalies (vs MODY5/HNF1B)
* No renal glycosuria (vs MODY3/HNF1A)
* No macrosomia or TNH (vs MODY1/HNF4A)
* No KPD-DKA-remission pattern (vs MODY9/PAX4)
* No ER-stress or C-peptide fall (vs MODY10/INS)
* No MAO-A mechanism (vs MODY7/KLF11)
* Ethnicity: European/French enrichment; BLK common variants European-enriched;
  broader population data limited

Diagnostic Strategy
-------------------
* Suspect MODY11: young-to-middle-onset DM, antibody-negative, family history,
  progressive HbA1c, preserved C-peptide, prominent post-prandial excursions,
  NO distinctive organ features (hard to differentiate from T2D/MODY7 on clinical grounds)
* Test: BLK gene sequencing (full coding + splice sites); expanded MODY NGS panel
* Functional validation: in-vitro BLK kinase activity assay; PDX1 phospho-immunoblot;
  co-segregation analysis in family
* NOT in oldest MODY panels — BLK first described 2009; must request expanded panel

Cohort: 40 patients, seed=323.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 323
_COHORT_SIZE = 40

# BLK variants — hypomorphic kinase domain / SH2 domain missense
_VARIANTS = [
    "A71T (c.211G>A)",          # SH2 domain; Borowiec 2009 PNAS; most common; ~50% kinase loss
    "P489L (c.1466C>T)",        # kinase catalytic domain; Borowiec 2009 PNAS; ~60% kinase loss
    "K469N (c.1407G>C)",        # kinase ATP-binding; European
    "E313K (c.937G>A)",         # SH2-kinase linker; French/UK; mildest
    "L326P (c.977T>C)",         # SH3-SH2-kinase interface; moderate phenotype
    "G377R (c.1129G>A)",        # kinase domain N-lobe; rare
    "Novel_hypomorphic_BLK",    # novel; BLK kinase assay + PDX1-phospho pending
    "Splice_BLK",               # splice site; partial kinase isoform loss; rare
]
_VARIANT_WEIGHTS = [0.32, 0.25, 0.14, 0.10, 0.08, 0.05, 0.04, 0.02]

_ETHNICITIES = [
    "European-French",
    "European-UK/Irish",
    "European-Other",
    "North American European",
    "Asian",
    "Other/Unknown",
]
_ETHNICITY_WEIGHTS = [0.32, 0.20, 0.18, 0.16, 0.08, 0.06]

_TREATMENTS = [
    "Sulfonylurea (monotherapy)",
    "Sulfonylurea + Metformin",
    "Insulin (basal)",
    "Insulin (basal-bolus)",
    "Metformin monotherapy",
    "Lifestyle / diet",
]
_TX_WEIGHTS = [0.35, 0.28, 0.16, 0.10, 0.07, 0.04]

_MISDIAGNOSES = [
    "T2D",
    "LADA",
    "T1D",
    "Prediabetes",
    "None (index case correctly diagnosed)",
]
_MISDIAG_WEIGHTS = [0.50, 0.08, 0.06, 0.11, 0.25]

_DISEASE_STAGES = [
    "Early (HbA1c 6.0–7.4%)",
    "Moderate (HbA1c 7.5–8.9%)",
    "Advanced (HbA1c ≥ 9.0%)",
]
_STAGE_WEIGHTS = [0.42, 0.38, 0.20]

# BLK kinase activity tiers (% of wild-type kinase activity)
_KINASE_TIERS = [
    "30–45% WT (severe hypomorph)",
    "46–60% WT (moderate hypomorph)",
    "61–75% WT (mild hypomorph)",
    "> 75% WT (hypomorphic, near-normal)",
]
_KINASE_WEIGHTS = [0.22, 0.40, 0.26, 0.12]


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
        kinase_tier = wchoice(_KINASE_TIERS, _KINASE_WEIGHTS)

        # Age at diagnosis: MODY11 has the latest mean onset (~35-45 yr)
        dx_age = round(rng.gauss(38, 10), 1)
        dx_age = max(14, min(62, dx_age))

        # HbA1c: correlates with stage
        if "Early" in stage:
            hba1c = round(rng.gauss(6.8, 0.4), 1)
            hba1c = max(6.0, min(7.4, hba1c))
        elif "Moderate" in stage:
            hba1c = round(rng.gauss(8.1, 0.5), 1)
            hba1c = max(7.5, min(8.9, hba1c))
        else:
            hba1c = round(rng.gauss(9.6, 0.6), 1)
            hba1c = max(9.0, min(12.0, hba1c))

        # C-peptide: PRESERVED (kinase-signalling defect, not structural)
        # BLK haploinsufficiency does not destroy beta-cell mass
        c_peptide = round(rng.gauss(0.80, 0.20), 2)
        c_peptide = max(0.45, min(1.50, c_peptide))

        # BMI: higher than other MODYs — T2D phenotypic overlap
        bmi = round(rng.gauss(28.5, 4.5), 1)
        bmi = max(18, min(45, bmi))

        family_hx = rng.random() < 0.70

        # Disease duration since diagnosis
        duration = round(rng.gauss(8.5, 6.0), 1)
        duration = max(0.5, min(30, duration))

        # Post-prandial glucose: blunted second-phase GSIS → high post-prandial
        pp_glucose = round(rng.gauss(11.5, 2.5), 1)
        pp_glucose = max(7.0, min(18.0, pp_glucose))

        cohort.append({
            "id": i + 1,
            "variant": variant,
            "ethnicity": ethnicity,
            "treatment": treatment,
            "prior_misdiagnosis": prior_misdiag,
            "disease_stage": stage,
            "kinase_activity_tier": kinase_tier,
            "age_at_diagnosis": dx_age,
            "hba1c_pct": hba1c,
            "c_peptide_nmol_L": c_peptide,
            "bmi": bmi,
            "family_history_positive": family_hx,
            "disease_duration_yr": duration,
            "pp_glucose_2h_mmol_L": pp_glucose,
            "autoantibodies": "Negative",
            "exocrine_insufficiency": False,
            "renal_cysts": False,
            "renal_glycosuria": False,
            "kpd_pattern": False,
            "er_stress_marker": False,
        })
    return cohort


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    cohort = _build_cohort()
    n = _COHORT_SIZE

    mean_hba1c = round(statistics.mean(p["hba1c_pct"] for p in cohort), 2)
    mean_cp    = round(statistics.mean(p["c_peptide_nmol_L"] for p in cohort), 3)
    mean_dx_age = round(statistics.mean(p["age_at_diagnosis"] for p in cohort), 1)
    mean_bmi   = round(statistics.mean(p["bmi"] for p in cohort), 1)
    mean_pp_gluc = round(statistics.mean(p["pp_glucose_2h_mmol_L"] for p in cohort), 1)

    pct_su = round(
        sum(1 for p in cohort if "Sulfonylurea" in p["treatment"]) / n * 100, 1)
    pct_insulin = round(
        sum(1 for p in cohort if "Insulin" in p["treatment"]) / n * 100, 1)
    pct_misdiag = round(
        sum(1 for p in cohort if p["prior_misdiagnosis"] != "None (index case correctly diagnosed)") / n * 100, 1)
    pct_fam_hx = round(
        sum(1 for p in cohort if p["family_history_positive"]) / n * 100, 1)
    pct_t2d_misdiag = round(
        sum(1 for p in cohort if p["prior_misdiagnosis"] == "T2D") / n * 100, 1)

    kpis = {
        "gene":          "BLK",
        "chromosome":    "8p23.1",
        "omim_gene":     "*191305",
        "omim_disease":  "#613375",
        "cohort_size":   n,
        "mean_hba1c":    f"{mean_hba1c}%",
        "mean_c_peptide": f"{mean_cp} nmol/L (preserved)",
        "mean_dx_age":   f"{mean_dx_age} yr (latest mean of MODY types)",
        "mean_bmi":      f"{mean_bmi} kg/m²",
        "mean_2h_pp_glucose": f"{mean_pp_gluc} mmol/L",
        "pct_su_response": f"{pct_su}%",
        "pct_insulin_required": f"{pct_insulin}%",
        "pct_misdiagnosed": f"{pct_misdiag}%",
        "pct_t2d_misdiag": f"{pct_t2d_misdiag}% (highest of all MODY types)",
        "pct_family_hx": f"{pct_fam_hx}%",
        "pct_antibody_neg": "100%",
        "inheritance": "Autosomal Dominant",
        "mechanism_class": "Kinase-signalling defect (GSIS amplification)",
        "c_peptide_pattern": "PRESERVED — beta-cell mass structurally intact",
        "prevalence": "~1% MODY; very rare; European/French-enriched",
        "misdiagnosis_most_common": "T2D (~50%) — late onset + BMI overlap",
        "treatment_first_line": "Sulfonylurea (60–70% respond)",
        "gwas_link": "BLK rs922879 = T2D GWAS locus (common-variant / rare-variant convergence)",
    }

    key_facts = [
        "BLK is a Src-family non-receptor tyrosine kinase; beta-cell role was unexpected (known for B-lymphocytes)",
        "MODY11 mechanism: BLK haploinsufficiency → ↓ PDX1 phosphorylation + ↓ KATP-independent GSIS amplification",
        "Hypomorphic (not null) mutations: A71T, P489L reduce kinase activity 50–60% — partial loss only",
        "C-peptide PRESERVED throughout course — no beta-cell apoptosis (unlike MODY10/INS ER-stress)",
        "Second-phase GSIS blunted selectively → prominent post-prandial hyperglycaemia",
        "Latest mean onset (~35–45 yr) of all MODY types → highest T2D misdiagnosis rate (~50%)",
        "SU first-line: closes K-ATP channels, amplifies first-phase release, partially compensates",
        "BLK common variant rs922879 bridges rare MODY11 and common T2D genetics",
        "Autoantibodies ALWAYS negative; test mandatory to exclude T1D / LADA",
        "Expanded NGS MODY panel required — BLK absent from oldest MODY gene panels",
        "Functional validation: BLK kinase assay + PDX1 phospho-immunoblot for novel variants",
        "No renal, exocrine, neurological, or neonatal phenotype distinguishing features",
    ]

    alerts = {
        "diagnosis_trap": (
            "T2D misdiagnosis ~50%: late onset + BMI overlap + no distinctive organ signs. "
            "Clue: family history + antibody-negative + preserved C-peptide + SU over-response."
        ),
        "t2d_gwas_convergence": (
            "BLK rs922879 is a T2D GWAS locus. MODY11 = rare coding extreme of the same axis. "
            "T2D GRS tools may flag as T2D risk — molecular diagnosis essential."
        ),
        "su_response": (
            "SU works (60–70%) because beta-cell mass is intact. Amplifies first-phase GSIS. "
            "Does not fully rescue blunted second-phase; metformin adjunct for BMI > 27."
        ),
        "functional_validation": (
            "For any novel BLK variant: BLK kinase activity assay + PDX1 phospho-S269 immunoblot. "
            "Hypomorphic (partial activity) counts as MODY11-causing; null (zero activity) = lethal."
        ),
        "panel_gap": (
            "BLK not in oldest MODY panels (HNF1A/HNF4A/GCK/HNF1B). "
            "Request expanded NGS panel or targeted BLK sequencing when high clinical suspicion."
        ),
    }

    patients_preview = [
        {
            "id": p["id"],
            "variant": p["variant"],
            "age_dx": p["age_at_diagnosis"],
            "hba1c": p["hba1c_pct"],
            "c_peptide": p["c_peptide_nmol_L"],
            "treatment": p["treatment"],
            "stage": p["disease_stage"],
            "family_hx": p["family_history_positive"],
        }
        for p in cohort[:12]
    ]

    return {
        "dashboard": "MODY11 — BLK-MODY",
        "cohort_size": n,
        "kpis": kpis,
        "key_facts": key_facts,
        "alerts": alerts,
        "patients": patients_preview,
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

    # C-peptide tiers (PRESERVED pattern — all should be > 0.40)
    cp_tiers = {
        "< 0.40 nmol/L": 0,
        "0.40–0.59 nmol/L": 0,
        "0.60–0.99 nmol/L": 0,
        "≥ 1.00 nmol/L": 0,
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

    # Age at diagnosis tiers (late onset)
    dx_age_tiers = {"< 20 yr": 0, "20–29 yr": 0, "30–39 yr": 0, "40–49 yr": 0, "≥ 50 yr": 0}
    for p in cohort:
        a = p["age_at_diagnosis"]
        if a < 20:
            dx_age_tiers["< 20 yr"] += 1
        elif a < 30:
            dx_age_tiers["20–29 yr"] += 1
        elif a < 40:
            dx_age_tiers["30–39 yr"] += 1
        elif a < 50:
            dx_age_tiers["40–49 yr"] += 1
        else:
            dx_age_tiers["≥ 50 yr"] += 1

    # Disease stage distribution
    stage_dist = {}
    for p in cohort:
        stage_dist[p["disease_stage"]] = stage_dist.get(p["disease_stage"], 0) + 1

    # Kinase activity tier distribution
    kinase_dist = {}
    for p in cohort:
        kinase_dist[p["kinase_activity_tier"]] = kinase_dist.get(p["kinase_activity_tier"], 0) + 1

    # Treatment distribution
    tx_dist = {}
    for p in cohort:
        tx_dist[p["treatment"]] = tx_dist.get(p["treatment"], 0) + 1

    # Misdiagnosis distribution
    mis_dist = {}
    for p in cohort:
        mis_dist[p["prior_misdiagnosis"]] = mis_dist.get(p["prior_misdiagnosis"], 0) + 1

    # BMI tiers (higher than typical MODY — T2D phenotypic overlap)
    bmi_tiers = {"< 23": 0, "23–24.9": 0, "25–29.9": 0, "30–34.9": 0, "≥ 35": 0}
    for p in cohort:
        b = p["bmi"]
        if b < 23:
            bmi_tiers["< 23"] += 1
        elif b < 25:
            bmi_tiers["23–24.9"] += 1
        elif b < 30:
            bmi_tiers["25–29.9"] += 1
        elif b < 35:
            bmi_tiers["30–34.9"] += 1
        else:
            bmi_tiers["≥ 35"] += 1

    # Post-prandial glucose tiers
    pp_tiers = {"< 9 mmol/L": 0, "9–11.9 mmol/L": 0, "12–14.9 mmol/L": 0, "≥ 15 mmol/L": 0}
    for p in cohort:
        g = p["pp_glucose_2h_mmol_L"]
        if g < 9:
            pp_tiers["< 9 mmol/L"] += 1
        elif g < 12:
            pp_tiers["9–11.9 mmol/L"] += 1
        elif g < 15:
            pp_tiers["12–14.9 mmol/L"] += 1
        else:
            pp_tiers["≥ 15 mmol/L"] += 1

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
        "pct_su_response": round(
            sum(1 for p in cohort if "Sulfonylurea" in p["treatment"]) / n * 100, 1),
        "pct_insulin_required": round(
            sum(1 for p in cohort if "Insulin" in p["treatment"]) / n * 100, 1),
        "pct_t2d_misdiagnosis": round(
            sum(1 for p in cohort if p["prior_misdiagnosis"] == "T2D") / n * 100, 1),
        "pct_antibody_negative": 100.0,
        "pct_family_hx_positive": round(
            sum(1 for p in cohort if p["family_history_positive"]) / n * 100, 1),
        "pct_c_pep_preserved": round(
            sum(1 for p in cohort if p["c_peptide_nmol_L"] >= 0.60) / n * 100, 1),
        "pct_bmi_overweight": round(
            sum(1 for p in cohort if p["bmi"] >= 25) / n * 100, 1),
    }

    return {
        "variant_distribution": var_dist,
        "ethnicity_distribution": eth_dist,
        "hba1c_tiers": hba1c_tiers,
        "c_peptide_tiers": cp_tiers,
        "age_at_diagnosis_tiers": dx_age_tiers,
        "disease_stage_distribution": stage_dist,
        "kinase_activity_distribution": kinase_dist,
        "treatment_distribution": tx_dist,
        "misdiagnosis_distribution": mis_dist,
        "bmi_tiers": bmi_tiers,
        "pp_glucose_tiers": pp_tiers,
        "disease_duration_tiers": dur_tiers,
        "summary_flags": summary_flags,
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "full_name": "MODY11 — BLK-MODY (Maturity-Onset Diabetes of the Young Type 11)",
            "gene": "BLK — B Lymphocyte Kinase; Src-family non-receptor tyrosine kinase; 8p23.1; OMIM *191305",
            "disease_omim": "#613375",
            "inheritance": "Autosomal Dominant — heterozygous hypomorphic missense; 50% transmission",
            "prevalence": "~1% of MODY; rare; European/French-enriched; globally underdiagnosed",
            "mechanism": (
                "Hypomorphic BLK → ↓ PDX1 phosphorylation (reduced insulin biosynthesis) + "
                "↓ KATP-independent GSIS amplification (blunted second-phase insulin release) → "
                "post-prandial hyperglycaemia → progressive HbA1c rise. "
                "Beta-cell mass structurally intact → C-peptide PRESERVED."
            ),
            "protein_function": (
                "BLK (51 kDa; 505 aa): Src-family kinase with N-terminal SH3, SH2, and C-terminal "
                "kinase domains. In beta cells: (1) phosphorylates PDX1-Ser269 → PDX1 nuclear "
                "retention → insulin gene transcription; (2) activates cAMP/PKA amplifying arm "
                "of GSIS for ~30% of second-phase insulin release."
            ),
            "onset_age": "Teens to 50s (mean ~35–45 yr) — LATEST mean onset of MODY types described",
            "c_peptide_pattern": (
                "PRESERVED throughout clinical course. BLK haploinsufficiency impairs GSIS "
                "signalling, not beta-cell viability. Unlike MODY10 (falling C-peptide). "
                "Like MODY9 (preserved) but no KPD pattern."
            ),
            "treatment": "SU first-line (60–70%); metformin adjunct; some progress to insulin",
            "autoantibodies": "NEGATIVE (GADA, ZnT8, IA-2) — always; test mandatory to exclude T1D/LADA",
            "family_history": "65–75% positive (50% AD); de novo 10–15%",
            "misdiagnosis_rate": (
                "T2D ~50% (highest of MODY types) — late onset + BMI overlap. "
                "LADA ~8%; T1D ~6%"
            ),
        },

        "genes_and_proteins": {
            "BLK (*191305)": (
                "8p23.1. Src-family non-receptor tyrosine kinase. 505 aa, 51 kDa. "
                "Canonical: B-lymphocyte signalling (BCR signalling cascade). "
                "Beta-cell role: PDX1 phosphorylation + KATP-independent GSIS amplification. "
                "First MODY TK gene (not a TF, enzyme, or structural protein)."
            ),
            "PDX1 (MODY4 gene; *600733)": (
                "Master pancreatic transcription factor. BLK phosphorylates PDX1 at Ser269 → "
                "nuclear retention → drives INS, GCK, GLUT2 transcription. BLK LOF → "
                "hypophosphorylated PDX1 → partial insulin synthesis impairment. "
                "Note: PDX1 heterozygous LOF = MODY4; homozygous LOF = pancreatic agenesis."
            ),
            "KATP channel (Kir6.2/SUR1)": (
                "K-ATP channels close in response to ATP rise from glucose metabolism → "
                "membrane depolarisation → Ca²⁺ influx → first-phase insulin release. "
                "BLK amplifies the KATP-INDEPENDENT second-phase (cAMP/PKA/incretins). "
                "SU mimics K-ATP closure → partially restores BLK LOF deficit."
            ),
            "rs922879 (BLK promoter SNP)": (
                "Common T2D GWAS variant (MAF ~15%); modestly reduces BLK expression (~20%). "
                "T2D GWAS and MODY11 converge on the same gene — rare coding hypomorphs cause "
                "MODY11; common non-coding variant modulates T2D polygenic risk."
            ),
        },

        "clinical_terms": {
            "MODY11": "Maturity-Onset Diabetes of the Young Type 11; BLK gene; kinase-signalling GSIS defect",
            "Hypomorphic allele": (
                "Partial loss-of-function — reduces but does not abolish kinase activity. "
                "A71T (~50% kinase loss), P489L (~60% kinase loss). Null alleles likely lethal. "
                "MODY11 pathogenicity requires partial activity — makes functional validation essential."
            ),
            "GSIS (Glucose-Stimulated Insulin Secretion)": (
                "Two phases: (1) First-phase — rapid ATP-triggered K-ATP closure → Ca²⁺ → insulin; "
                "(2) Second-phase — sustained KATP-independent amplification via cAMP/PKA, incretins, "
                "lipid signals. BLK LOF blunts second-phase selectively."
            ),
            "KATP-independent amplification": (
                "~30% of second-phase GSIS. Involves GLP-1/GIP → adenylyl cyclase → cAMP → PKA → "
                "Epac2 → RIM2/RIM-BP → exocytosis amplification. BLK activates this pathway. "
                "Explains why DPP-4i/GLP-1 analogues may have added benefit in MODY11."
            ),
            "PDX1 phosphorylation": (
                "BLK-mediated phosphorylation of PDX1 Ser269 → PDX1 remains in nucleus → "
                "drives insulin gene transcription. Hypophosphorylated PDX1 is exported → "
                "partial insulin biosynthesis deficit on top of secretion impairment."
            ),
            "T2D GWAS convergence": (
                "BLK rs922879 is a confirmed T2D GWAS signal. MODY11 is the rare monogenic extreme "
                "of the same axis. Illustrates the continuum between common complex T2D risk and "
                "rare high-penetrance monogenic MODY."
            ),
        },

        "lab_thresholds": {
            "C-peptide (MODY11)": "≥ 0.60 nmol/L (fasting); consistent throughout course — preserved",
            "C-peptide (normal reference)": "0.37–1.47 nmol/L (fasting); MODY11 in normal range",
            "2-hour post-prandial glucose": "Often > 10 mmol/L in MODY11 — second-phase blunted",
            "HbA1c target (SU therapy)": "< 7.0% (53 mmol/mol); hypoglycaemia risk with SU — monitor",
            "Autoantibodies (T2D/T1D exclusion)": "GADA < 5 IU/mL; ZnT8-Ab negative; IA-2 negative",
            "BLK kinase activity (functional assay)": "< 70% WT activity = significant hypomorphic LOF",
            "PDX1 phospho-Ser269 (immunoblot)": "Reduced vs WT control = supports BLK LOF pathogenicity",
        },

        "treatment": {
            "sulfonylurea_first_line": (
                "SU (glibenclamide/glipizide): closes K-ATP channels → amplifies first-phase GSIS. "
                "60–70% achieve HbA1c < 7.0%. Hypoglycaemia risk — start low dose. "
                "Does not fully rescue blunted second-phase KATP-independent amplification."
            ),
            "metformin_adjunct": (
                "Particularly useful in BMI > 27 patients (T2D phenotypic overlap). "
                "Improves insulin sensitivity; no hypoglycaemia risk; supports SU co-therapy."
            ),
            "dpp4i_glp1_rationale": (
                "GLP-1 receptor agonists / DPP-4 inhibitors boost cAMP/PKA amplification → "
                "theoretically compensate for blunted BLK KATP-independent pathway. "
                "Evidence limited to case series; biologically plausible."
            ),
            "insulin_for_progressive_cases": (
                "Some MODY11 patients progress to insulin need with increasing HbA1c despite SU. "
                "C-peptide remains preserved — partial beta-cell function retained. "
                "Basal insulin + SU combination feasible."
            ),
            "genetic_counselling": (
                "50% AD transmission; all first-degree relatives need BLK sequencing + HbA1c + "
                "2-hour OGTT (post-prandial blunting may precede fasting hyperglycaemia). "
                "Early diagnosis = SU before insulin requirement."
            ),
        },

        "genetics_testing": {
            "BLK_sequencing": (
                "Full coding sequence (exons + splice sites); 8p23.1. "
                "Report partial LOF (hypomorphic) variants, not only null/truncating. "
                "Interpret in context of kinase activity assay."
            ),
            "functional_validation": (
                "BLK kinase activity assay (in-vitro phosphotransfer assay). "
                "PDX1 phospho-Ser269 immunoblot (co-transfection with BLK variant). "
                "Co-segregation in family: all affected = variant carriers."
            ),
            "MODY_panel_requirement": (
                "BLK absent from oldest 4-gene or 6-gene MODY panels. "
                "Expanded NGS MODY panel (10+ genes) required. "
                "Request specifically if: young/mid-onset T2D phenotype + family history + antibody-negative."
            ),
            "cascade_screening": (
                "All first-degree relatives of confirmed MODY11. 50% carry variant. "
                "Screen: BLK sequencing + HbA1c + 2-hour OGTT + C-peptide. "
                "Cascade benefits: switch from metformin/insulin to SU, avoid unnecessary T2D treatment."
            ),
            "T2D_panel_misclassification": (
                "BLK common variant rs922879 may appear on T2D polygenic risk tools as T2D risk. "
                "Do NOT misclassify MODY11 as high-T2D-risk — molecular diagnosis essential."
            ),
        },

        "comparison_mody10_11": {
            "MODY10 (INS)": {
                "gene": "INS; 11p15.5; dominant-negative misfolded proinsulin",
                "mechanism": "ER stress (UPR) → progressive beta-cell apoptosis",
                "c_peptide": "FALLS progressively — structural apoptotic loss",
                "treatment": "Insulin 70–80%; SU marginal early only",
                "onset": "Teens–early 40s (mean ~28–32 yr)",
                "unique": "Structural ER-stress disease; falling C-peptide distinguishes",
            },
            "MODY11 (BLK)": {
                "gene": "BLK; 8p23.1; hypomorphic kinase LOF",
                "mechanism": "↓ PDX1 phosphorylation + ↓ KATP-independent GSIS amplification",
                "c_peptide": "PRESERVED — no beta-cell structural loss",
                "treatment": "SU first-line 60–70%; metformin adjunct",
                "onset": "Teens–50s (mean ~35–45 yr — LATEST of MODY types)",
                "unique": "T2D GWAS convergence; kinase signalling defect; highest T2D misdiagnosis",
            },
        },
    }
