"""
MODY9 — PAX4-MODY (Maturity-Onset Diabetes of the Young Type 9)
================================================================================
Gene       : PAX4 (Paired Box 4)
Chromosome : 7q32.1
OMIM Gene  : *167413
OMIM Dis.  : #612225  (MODY9)
Inheritance: Autosomal Dominant (heterozygous LOF → MODY9)
Prevalence : ~1–2% of MODY; rare; East Asian enrichment (Thai, Japanese, Korean, Chinese)

Mechanism
---------
PAX4 encodes a paired-domain + homeodomain transcription factor expressed exclusively
in islet progenitor cells. Its primary function: transcriptional repression of ARX
(Aristaless Related Homeobox), the master alpha-cell fate determinant.

MODY9 PATHOMECHANISM:
1. Heterozygous PAX4 LOF variant → haploinsufficiency of PAX4 protein
   → insufficient repression of ARX in islet progenitors
   → alpha-cell fate programme not fully suppressed
   → impaired alpha-to-beta cell differentiation balance
   → reduced beta-cell mass and/or functional GSIS defect
   → progressive hyperglycaemia

2. The core deficit is transcriptional:
   - PAX4 binds Pax/ATTA elements in the ARX promoter → silences ARX
   - PAX4 also regulates GLUT2 and the insulin promoter directly (paired-domain binding)
   - Haploinsufficiency → partial de-repression of ARX + partial GSIS impairment

MODY9-UNIQUE Feature: Ketosis-Prone Diabetes (KPD)
---------------------------------------------------
The R121W (Arg121Trp) variant in the homeodomain of PAX4 — the most common East Asian
founder mutation — is associated with a distinct KPD (ketosis-prone diabetes) phenotype:

  • Patients present with acute diabetic ketoacidosis (DKA) without classic T1D
    autoimmunity (antibody-negative)
  • After initial insulin stabilisation, C-peptide function RECOVERS
  • Up to 50–70% of R121W carriers with KPD can discontinue insulin and be managed
    on sulfonylureas or diet alone (partial remission)
  • This phenotype is rare in other MODY types (no other MODY type routinely presents
    as DKA then remits)
  • Misdiagnosis as T1D is the most clinically dangerous error: insulin withdrawal is
    safe in KPD-MODY9 with monitoring, but unsafe if truly T1D

Key Founding Mutations (East Asian enrichment)
----------------------------------------------
* R121W (c.361C>T) — homeodomain, most prevalent; Thai, Korean, Japanese founders
  (Plengvidhya et al. 2007 — first PAX4 MODY paper; Thai cohort)
* A256V (c.767C>T) — homeodomain, Japanese families; Shimajiri et al. 2011
* IVS7-1G>A — splice site, Chinese families
* R37W (c.109C>T) — paired domain, European lineages
* Q59L, L56P — N-terminal domain, rare individual families

Disease Context
---------------
* PAX4 interacts with PDX1 (MODY4) and NEUROD1 (MODY6) in the same islet
  differentiation cascade — loss of any one causes MODY diabetes
* ARX repression by PAX4 is the critical branching point:
    - ARX ON (PAX4 LOF) → alpha-cell default
    - PAX4 ON (ARX repressed) → beta-cell differentiation
* Beta-cell mass is progressively reduced; SU compensates by maximising residual GSIS

Diagnostic Strategy
--------------------
* Suspect MODY9: young-onset DM + antibody-negative + family history + East Asian
* KPD presentation: DKA at onset, C-peptide recovery after initial stabilisation →
  PAX4 sequencing mandatory before lifelong T1D label
* No renal cysts (vs MODY5), no EPI (vs MODY8), no Mullerian anomalies (vs MODY5)
* Expanded MODY NGS panel must include PAX4 — not in oldest panels
* Sulfonylurea trial: 75–80% respond; non-response may reflect longer disease duration

Cohort: 40 patients, seed=319.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEED = 319
_COHORT_SIZE = 40

# PAX4 variants — R121W dominant in East Asian cohort
_VARIANTS = [
    "R121W (c.361C>T)",        # homeodomain; Thai/Korean/Japanese founder — Plengvidhya 2007
    "A256V (c.767C>T)",        # homeodomain; Japanese founder — Shimajiri 2011
    "IVS7-1G>A (splice)",      # splice site; Chinese families
    "R37W (c.109C>T)",         # paired domain; European lineages
    "Q59L (c.176A>T)",         # N-terminal domain
    "L56P (c.167T>C)",         # N-terminal domain; rare individual families
    "p.E155K (c.463G>A)",      # homeodomain; French families — Plengvidhya 2007
    "Novel_missense_PAX4",     # novel; functional validation pending
    "Truncating_NMD",          # frameshift or stop-gain → NMD → haploinsufficiency
]
_VARIANT_WEIGHTS = [0.38, 0.18, 0.12, 0.10, 0.07, 0.05, 0.05, 0.03, 0.02]

# Treatment: SU first-line; some KPD start on insulin then switch
_TREATMENTS = [
    "Sulfonylurea (glibenclamide/gliclazide)",
    "Sulfonylurea + metformin",
    "Insulin (KPD initial) → SU after remission",
    "Insulin (ongoing — longer duration / low C-peptide)",
    "Diet only (mild, detected screening)",
]
_TREATMENT_WEIGHTS = [0.38, 0.22, 0.20, 0.15, 0.05]

_MISDIAGNOSES = ["T1D", "T2D", "KPD (T1D label)", "None"]
_MISDIAGNOSIS_WEIGHTS = [0.25, 0.28, 0.15, 0.32]

_SEXES = ["M", "F"]

# East Asian ethnicities enriched for PAX4 MODY9
_ETHNICITIES = [
    "Thai", "Korean", "Japanese", "Chinese",
    "European", "South Asian", "Other",
]
_ETHNICITY_WEIGHTS = [0.24, 0.20, 0.16, 0.14, 0.12, 0.08, 0.06]


def _make_patient(seed_val: int) -> dict:
    rng = random.Random(seed_val)
    sex = rng.choices(_SEXES, [0.52, 0.48])[0]
    age = rng.randint(22, 68)
    # MODY9 onset: 20s–40s (mean ~30–35 yr); KPD often acute at presentation
    dx_age = rng.randint(18, min(age, 48))
    duration = age - dx_age

    # HbA1c: moderate (SU-treated); higher in insulin-dependent or long-duration
    hba1c = round(rng.uniform(5.8, 10.8), 1)

    # Fasting glucose: elevated at diagnosis; may normalise on SU
    fg = round(rng.uniform(5.8, 13.8), 1)

    # C-peptide: PRESERVED at diagnosis (functional deficit, not structural loss)
    # KPD patients may have transiently suppressed then recovered C-peptide
    baseline_cp = round(rng.uniform(0.35, 1.80), 2)
    # Slight fall with duration (progressive beta-cell reduction over time)
    duration_penalty = min(duration * 0.008, 0.25)
    c_pep = max(round(baseline_cp - duration_penalty, 2), 0.10)

    variant = rng.choices(_VARIANTS, _VARIANT_WEIGHTS)[0]
    treatment = rng.choices(_TREATMENTS, _TREATMENT_WEIGHTS)[0]
    misdiagnosis = rng.choices(_MISDIAGNOSES, _MISDIAGNOSIS_WEIGHTS)[0]
    ethnicity = rng.choices(_ETHNICITIES, _ETHNICITY_WEIGHTS)[0]

    # Autoantibodies always negative
    gada = False
    znt8 = False
    ia2 = False

    # Family history: ~68%
    fam_hx = rng.random() < 0.68

    # KPD presentation: DKA at onset (more common with R121W, ~30–35%)
    kdp_presentation = (variant == "R121W (c.361C>T)") and (rng.random() < 0.55)
    kdp_presentation = kdp_presentation or (rng.random() < 0.08)  # rare in others

    # SU response (75–80% overall; lower if long duration or on insulin already)
    su_response = rng.random() < 0.77

    # BMI: typically normal–slightly elevated (not obese like T2D)
    bmi = round(rng.uniform(20.5, 31.5), 1)

    return {
        "patient_id": f"MODY9-{seed_val:04d}",
        "age": age,
        "sex": sex,
        "ethnicity": ethnicity,
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
        "kdp_presentation": kdp_presentation,      # ketosis-prone DM at onset
        "su_response": su_response,
        "exocrine_insufficiency": False,            # NO EPI — differs from MODY8
        "renal_cysts": False,                       # NO renal cysts — differs from MODY5
        "pancreatic_atrophy": False,                # NO pancreatic atrophy
        "hypomagnesaemia": False,
        "renal_glycosuria": False,                  # NO renal glycosuria — differs from MODY3
        "mullerian_anomalies": False,
        "macrosomia_at_birth": False,               # NO macrosomia — differs from MODY1
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

    pct_fam_hx = sum(1 for p in patients if p["family_history_positive"]) / _COHORT_SIZE * 100
    pct_misdiag = sum(1 for p in patients if p["prior_misdiagnosis"] != "None") / _COHORT_SIZE * 100
    pct_kdp = sum(1 for p in patients if p["kdp_presentation"]) / _COHORT_SIZE * 100
    pct_su_resp = sum(1 for p in patients if p["su_response"]) / _COHORT_SIZE * 100
    pct_east_asian = sum(
        1 for p in patients if p["ethnicity"] in ("Thai", "Korean", "Japanese", "Chinese")
    ) / _COHORT_SIZE * 100

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
            "pct_family_hx_positive": round(pct_fam_hx, 1),
            "pct_prior_misdiagnosis": round(pct_misdiag, 1),
            "pct_kdp_presentation": round(pct_kdp, 1),
            "pct_su_response": round(pct_su_resp, 1),
            "pct_east_asian": round(pct_east_asian, 1),
            "pct_antibody_positive": 0.0,
            "pct_renal_cysts": 0.0,
            "pct_exocrine_insufficiency": 0.0,
        },
        "patients": patients,
        "key_facts": [
            "MODY9-PAX4: PAX4 is a paired-domain + homeodomain TF that represses ARX — LOF → insufficient ARX repression → alpha-cell fate bias → reduced beta-cell mass → GSIS failure",
            "UNIQUE KPD phenotype: R121W carriers (most common East Asian founder) may present with acute DKA — antibody-negative — C-peptide RECOVERS after insulin stabilisation → 50–70% achieve SU/diet remission",
            "East Asian enrichment: Thai, Korean, Japanese, Chinese founder mutations (R121W, A256V) — MODY9 is underdiagnosed in Asian populations misdiagnosed as T1D or T2D",
            "Sulfonylurea first-line: ~75–80% of MODY9 patients respond; SU closes K-ATP independent of transcriptional axis → bypasses PAX4 haploinsufficiency",
            "C-peptide PRESERVED at diagnosis (functional haploinsufficiency, not structural destruction) — key differentiator from MODY8 (structural) and T1D",
            "NO exocrine insufficiency (differs from MODY8/CEL), NO renal cysts (differs from MODY5/HNF1B), NO renal glycosuria (differs from MODY3/HNF1A), NO macrosomia (differs from MODY1/HNF4A)",
            "PAX4 regulates ARX, GLUT2, and the insulin promoter — shared pathway with PDX1 (MODY4) and NEUROD1 (MODY6) in the islet differentiation cascade",
            "Autoantibodies always negative (GADA, ZnT8, IA-2) — distinguishes from T1D even in KPD-presenting patients with DKA",
            "Expanded MODY NGS panel mandatory — PAX4 absent from oldest panels (HNF1A/HNF4A/GCK/HNF1B); request PAX4 explicitly in Asian patients with young-onset DM",
            "High misdiagnosis: T1D (25%), T2D (28%), KPD mislabelled T1D (15%) — antibody testing + PAX4 sequencing resolves in most cases",
            "Pregnancy: switch to insulin (glyburide crosses placenta); renal surveillance not required (unlike MODY5)",
            "Family screening is mandatory — 65–70% first-degree relatives affected; ~50% transmission (AD); KPD-MODY9 can masquerade as new-onset T1D in relatives",
        ],
        "alerts": {
            "kdp_alert": "KPD presentation (DKA at onset, antibody-negative) must trigger PAX4 sequencing before lifelong T1D insulin label — MODY9 may achieve SU remission after recovery",
            "east_asian_screen": "East Asian ancestry + young-onset DM + antibody-negative + family history → PAX4 sequencing priority; R121W and A256V are the most common founder variants",
            "su_trial": "Sulfonylurea trial is safe and effective in ~77% of MODY9 — initiate low dose (0.5 mg glibenclamide) with careful monitoring; C-peptide preservation predicts response",
            "no_exocrine": "MODY9 has NO exocrine pancreatic insufficiency — if EPI present, consider MODY8 (CEL) or MODY5 (HNF1B) instead",
            "panel_warning": "PAX4 (MODY9) is absent from many older MODY panels — specifically request PAX4 inclusion in the NGS panel, especially for East Asian patients",
        },
        "mody_registry": {
            "type": "MODY9",
            "gene": "PAX4",
            "omim_gene": "*167413",
            "omim_disease": "#612225",
            "chromosome": "7q32.1",
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
    hba1c_tiers = {"<7.0%": 0, "7.0–7.9%": 0, "8.0–9.4%": 0, "≥9.5%": 0}
    for p in patients:
        h = p["hba1c_percent"]
        if h < 7.0:
            hba1c_tiers["<7.0%"] += 1
        elif h < 8.0:
            hba1c_tiers["7.0–7.9%"] += 1
        elif h < 9.5:
            hba1c_tiers["8.0–9.4%"] += 1
        else:
            hba1c_tiers["≥9.5%"] += 1

    # C-peptide tiers — expected preserved
    cp_tiers = {
        "<0.20 nmol/L (low)": 0,
        "0.20–0.59 (borderline)": 0,
        "0.60–1.19 (preserved)": 0,
        "≥1.20 (normal)": 0,
    }
    for p in patients:
        cp = p["c_peptide_nmol_L"]
        if cp < 0.20:
            cp_tiers["<0.20 nmol/L (low)"] += 1
        elif cp < 0.60:
            cp_tiers["0.20–0.59 (borderline)"] += 1
        elif cp < 1.20:
            cp_tiers["0.60–1.19 (preserved)"] += 1
        else:
            cp_tiers["≥1.20 (normal)"] += 1

    # Misdiagnosis
    misdiag_dist: dict = {}
    for p in patients:
        m = p["prior_misdiagnosis"]
        misdiag_dist[m] = misdiag_dist.get(m, 0) + 1

    # Treatment
    tx_dist: dict = {}
    for p in patients:
        t = p["current_treatment"]
        tx_dist[t] = tx_dist.get(t, 0) + 1

    # Ethnicity breakdown
    eth_dist: dict = {}
    for p in patients:
        e = p["ethnicity"]
        eth_dist[e] = eth_dist.get(e, 0) + 1

    # Age at diagnosis brackets
    onset_brackets = {"<25yr": 0, "25–34yr": 0, "35–44yr": 0, "≥45yr": 0}
    for p in patients:
        a = p["age_at_diagnosis"]
        if a < 25:
            onset_brackets["<25yr"] += 1
        elif a < 35:
            onset_brackets["25–34yr"] += 1
        elif a < 45:
            onset_brackets["35–44yr"] += 1
        else:
            onset_brackets["≥45yr"] += 1

    # KPD vs non-KPD comparison
    kdp_pts = [p for p in patients if p["kdp_presentation"]]
    non_kdp_pts = [p for p in patients if not p["kdp_presentation"]]
    kdp_mean_cp = round(statistics.mean(p["c_peptide_nmol_L"] for p in kdp_pts), 3) if kdp_pts else 0
    non_kdp_mean_cp = round(statistics.mean(p["c_peptide_nmol_L"] for p in non_kdp_pts), 3) if non_kdp_pts else 0

    # Duration vs HbA1c progression (5-yr buckets)
    dur_hba1c: dict = {"0–4yr": [], "5–9yr": [], "10–14yr": [], "15+yr": []}
    for p in patients:
        dur = p["duration_years"]
        if dur < 5:
            dur_hba1c["0–4yr"].append(p["hba1c_percent"])
        elif dur < 10:
            dur_hba1c["5–9yr"].append(p["hba1c_percent"])
        elif dur < 15:
            dur_hba1c["10–14yr"].append(p["hba1c_percent"])
        else:
            dur_hba1c["15+yr"].append(p["hba1c_percent"])
    dur_hba1c_mean = {
        k: round(statistics.mean(v), 1) if v else None
        for k, v in dur_hba1c.items()
    }

    return {
        "variant_distribution": var_dist,
        "hba1c_tiers": hba1c_tiers,
        "c_peptide_tiers": cp_tiers,
        "misdiagnosis_distribution": misdiag_dist,
        "treatment_distribution": tx_dist,
        "ethnicity_distribution": eth_dist,
        "onset_age_brackets": onset_brackets,
        "kdp_vs_non_kdp": {
            "kdp_count": len(kdp_pts),
            "non_kdp_count": len(non_kdp_pts),
            "kdp_mean_c_peptide_nmol_L": kdp_mean_cp,
            "non_kdp_mean_c_peptide_nmol_L": non_kdp_mean_cp,
        },
        "duration_vs_hba1c_mean": dur_hba1c_mean,
        "su_response_by_variant": {
            v: {
                "n": sum(1 for p in patients if p["variant"] == v),
                "su_responders": sum(1 for p in patients if p["variant"] == v and p["su_response"]),
            }
            for v in _VARIANTS
        },
        "sex_breakdown": {
            "M": sum(1 for p in patients if p["sex"] == "M"),
            "F": sum(1 for p in patients if p["sex"] == "F"),
        },
        "patients": patients,
    }


def get_definitions() -> dict:
    return {
        "disease": "MODY9 (PAX4-MODY)",
        "full_name": "Maturity-Onset Diabetes of the Young Type 9",
        "gene": {
            "symbol": "PAX4",
            "full_name": "Paired Box 4",
            "omim": "*167413",
            "chromosome": "7q32.1",
            "protein_length": "349 amino acids",
            "domains": [
                "Paired domain (N-terminal) — DNA binding at Pax/ATTA elements",
                "Homeodomain (central) — DNA binding at Antennapedia-class sites; R121 in loop 1",
                "PST domain (C-terminal proline/serine/threonine) — transcriptional repression",
            ],
        },
        "disease_omim": "#612225",
        "inheritance": "Autosomal Dominant — heterozygous PAX4 LOF",
        "prevalence": "~1–2% of MODY; rare; East Asian enrichment",
        "mechanism": {
            "summary": "PAX4 haploinsufficiency → insufficient ARX repression → alpha-cell fate bias → reduced functional beta-cell mass → impaired GSIS",
            "arx_repression": "PAX4 binds ARX promoter → transcriptionally silences ARX (alpha-cell master TF) → enables beta-cell differentiation",
            "gsis_impairment": "PAX4 also activates GLUT2 and the insulin promoter directly; haploinsufficiency causes partial GSIS loss independent of ARX",
            "kdp_mechanism": "R121W disrupts homeodomain loop 1 → reduced ARX binding affinity → transient near-complete GSIS failure → DKA; residual PAX4 activity allows C-peptide recovery after acute phase",
        },
        "key_mutations": {
            "R121W": "c.361C>T; Arg121Trp; homeodomain loop 1; Thai/Korean/Japanese founder; KPD phenotype; Plengvidhya 2007 Nat Genet",
            "A256V": "c.767C>T; Ala256Val; homeodomain helix 3; Japanese founder; Shimajiri 2011 J Hum Genet",
            "IVS7-1G>A": "splice acceptor site intron 7; Chinese families; exon 8 skipping → truncation",
            "R37W": "c.109C>T; Arg37Trp; paired domain; European lineages; reduces Pax-element binding",
            "E155K": "c.463G>A; Glu155Lys; homeodomain; French families; Plengvidhya 2007",
        },
        "kdp_mody9": {
            "definition": "Ketosis-Prone Diabetes (KPD) subtype — DKA at onset, antibody-negative, C-peptide recovery, possible SU remission",
            "prevalence_in_mody9": "~30–40% of R121W carriers; ~8% other variants",
            "mechanism": "Acute severe haploinsufficiency → near-total GSIS loss → DKA; residual PAX4 function + insulin therapy → C-peptide recovery → partial remission possible",
            "remission_rate": "50–70% of KPD-MODY9 achieve SU or diet remission after initial insulin",
            "monitoring": "C-peptide recovery at 3 months guides insulin withdrawal trial; HbA1c and SMBG q3mo",
        },
        "clinical_hallmarks": [
            "Young-onset diabetes (teens–40s; mean 30–35 yr)",
            "Autoantibodies negative (GADA, ZnT8, IA-2) — always",
            "C-peptide preserved at diagnosis (functional, not structural deficit)",
            "Possible KPD at onset: DKA → recovery → remission (especially R121W)",
            "Family history positive ~68% (AD, 50% transmission)",
            "Sulfonylurea response 75–80% (excellent; low-dose start)",
            "East Asian ancestry (Thai, Korean, Japanese, Chinese) enriched",
            "No exocrine insufficiency (vs MODY8-CEL)",
            "No renal cysts (vs MODY5-HNF1B)",
            "No renal glycosuria (vs MODY3-HNF1A)",
            "No macrosomia (vs MODY1-HNF4A)",
            "No pancreatic atrophy (vs MODY5/MODY8)",
        ],
        "treatment": {
            "sulfonylurea": "First-line: glibenclamide 0.5–2.5 mg/day or gliclazide MR 30 mg/day; start low (MODY more sensitive than T2D); 75–80% achieve target HbA1c",
            "insulin_kdp": "Required acutely for KPD/DKA stabilisation; withdrawal trial at 3 months if C-peptide recovers (>0.20 nmol/L stimulated); replace with SU",
            "metformin": "Second-line adjunct if SU partial response; not insulin-sensitising mechanism; reasonable add-on",
            "pregnancy": "Switch to insulin pre-conception; glyburide crosses placenta; no renal monitoring required (unlike MODY5); standard obstetric care",
            "monitoring": "HbA1c q3–6mo; fasting glucose; C-peptide annually; no renal, exocrine, or vitamin monitoring required",
        },
        "differential_diagnosis": {
            "T1D_vs_MODY9": "Both may present with DKA (KPD-MODY9); antibodies negative in MODY9; C-peptide recovers in KPD-MODY9; family history more typical MODY pattern",
            "T2D_vs_MODY9": "T2D is polygenic, age >45, obese, no MODY family pattern; MODY9 younger, normal BMI, AD family history, SU hyper-sensitive",
            "MODY3_vs_MODY9": "MODY3 has renal glycosuria (50%) and exquisite SU sensitivity; MODY9 no renal glycosuria; SU response similar but less extreme",
            "MODY8_vs_MODY9": "MODY8 has EPI (steatorrhoea, FEL-1 low), pancreatic lipomatosis, insulin mandatory; MODY9 no exocrine disease, SU-responsive",
            "MODY5_vs_MODY9": "MODY5 has renal cysts, Mullerian anomalies, hypomagnesaemia, pancreatic atrophy; MODY9 has none of these",
        },
        "comparison_mody8_9_10": {
            "MODY8_CEL": {
                "mechanism": "Exocrine enzyme VNTR frameshift → acinar apoptosis → lipomatosis → structural beta-cell loss",
                "exocrine": "EPI always (100%)",
                "su_response": "None — structural",
                "insulin": "Mandatory",
                "c_peptide": "Low at diagnosis",
                "imaging": "Pancreatic lipomatosis",
                "ethnicity": "Norwegian/Scandinavian enriched",
            },
            "MODY9_PAX4": {
                "mechanism": "TF haploinsufficiency → ARX de-repression → alpha-cell bias → GSIS impairment",
                "exocrine": "None",
                "su_response": "75–80%",
                "insulin": "KPD phase only; SU long-term",
                "c_peptide": "Preserved (may transiently dip in KPD then recover)",
                "imaging": "Normal pancreas",
                "ethnicity": "East Asian enriched (Thai/Korean/Japanese/Chinese)",
            },
            "MODY10_INS": {
                "mechanism": "INS gene mutation → misfolded preproinsulin → ER stress → beta-cell apoptosis",
                "exocrine": "None",
                "su_response": "Variable — depends on ER stress severity",
                "insulin": "Often required",
                "c_peptide": "Falls with duration (structural ER apoptosis)",
                "imaging": "Normal",
                "ethnicity": "Pan-ethnic; European families described",
            },
        },
        "genetics": {
            "inheritance": "Autosomal Dominant — heterozygous LOF",
            "penetrance": "~70–80% — variable expressivity; some carriers asymptomatic to age 50+",
            "de_novo_rate": "~10–15% — lower than MODY5 (50% de novo)",
            "testing": "Expanded MODY NGS panel including PAX4; Sanger for known familial variant; functional validation (ARX reporter assay) for novel variants",
            "family_screening": "All first-degree relatives — 50% transmission; KPD-onset relatives may present as T1D → antibody test + PAX4 sequencing",
        },
        "references": [
            "Plengvidhya N et al. (2007) PAX4 mutations in Thais with maturity onset diabetes of the young. J Clin Endocrinol Metab 92(7):2821-6",
            "Shimajiri Y et al. (2011) Comprehensive expression analysis of PAX4 variants in Japanese MODY patients. J Hum Genet 56(7):533-8",
            "Brun T et al. (2008) MODY9 (PAX4) — cellular mechanisms of impaired glucose-stimulated insulin secretion. Endocrinology 149:3697",
            "Malecki MT et al. (1999) Mutations in NEUROD1 (MODY6) — original NEUROD1 MODY paper (same islet axis as PAX4). Nat Genet 23:323-8",
        ],
    }
