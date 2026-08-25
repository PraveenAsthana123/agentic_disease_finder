"""
Wolfram Syndrome 1 (WFS1 — DIDMOAD)
================================================================================
Gene       : WFS1 (Wolframin ER Transmembrane Glycoprotein)
Chromosome : 4p16.1
OMIM Gene  : *606201
OMIM Dis.  : #222300 (Wolfram Syndrome 1; DIDMOAD)
Inheritance: Autosomal Recessive — biallelic loss-of-function (LOF); compound
             heterozygous common; ~1/770,000 prevalence worldwide; rare but
             severe multisystem condition
Acronym    : DIDMOAD — Diabetes Insipidus, Diabetes Mellitus, Optic Atrophy,
             Deafness (Wolfram's original 1938 description did not include DI/D)

Mechanism
---------
WFS1 encodes wolframin — an 890-aa nine-transmembrane ER glycoprotein that
localises exclusively to the endoplasmic reticulum membrane.

Wolframin function:
  • Maintains ER Ca²⁺ homeostasis and ER membrane integrity
  • Regulates ER stress / unfolded protein response (UPR) signalling
  • Critical for beta-cell ER Ca²⁺ buffering and SERCA pump regulation
  • Stabilises IP3R-SERCA Ca²⁺ micro-domains essential for GSIS
  • Modulates IRE1α ubiquitination and PERK-eIF2α pathway activation

WFS1 biallelic LOF → wolframin absent/non-functional →
  • ER Ca²⁺ depletion in pancreatic beta-cells →
  • Unresolved ER stress → chronic UPR activation →
  • PERK-eIF2α-ATF4-CHOP axis → beta-cell apoptosis →
  • Progressive beta-cell loss → C-peptide FALLS → absolute insulin dependence
  • Same ER-stress pathway in retinal ganglion cells (optic atrophy),
    cochlear hair cells (sensorineural deafness), hypothalamic neurons (DI),
    brainstem/cerebellum (neurodegeneration), renal collecting duct (DI)

This makes Wolfram unique: the SAME ER-stress mechanism simultaneously damages
MULTIPLE cell types → temporal cascade of organ manifestations.

Temporal Cascade (approximate mean ages at onset)
--------------------------------------------------
1. Diabetes Mellitus (DM): ~6 yr (range 1–20 yr); juvenile insulin-dependent
2. Optic Atrophy (OA): ~11 yr (range 2–24 yr); bilateral progressive; blindness
3. Diabetes Insipidus (DI): ~14 yr (range 5–38 yr); central; ~70% of patients
4. Sensorineural Deafness (D): ~16 yr (range 4–58 yr); high-frequency; ~65%
5. Neurological: ~20–30 yr; cerebellar ataxia, autonomic neuropathy, brainstem
6. Psychiatric: ~20 yr; depression, suicidality (25%), psychosis (10%)
7. Renal tract abnormalities: dilated upper urinary tract / atonic bladder (50%)

Key Clinical Features
---------------------
• DM: juvenile-onset (~6 yr); C-peptide FALLS progressively (ER-stress apoptosis)
  — distinguishes from MODY10/INS (same mechanism, but LATER onset)
  — distinguishes from ALL other MODY types (which have preserved C-peptide)
  — 100% insulin-dependent from diagnosis; NOT T1D (antibody-negative)
• OA: bilateral progressive optic neuropathy; corticoretinal atrophy; OCT/VEP
• DI: central (hypothalamic ADH/AVP deficiency); polydipsia + polyuria NOT from DM
• Deafness: sensorineural; high-frequency; audiometric monitoring
• Neurological (late): dysarthria, dysphagia, ataxia, nystagmus; brainstem atrophy
• Psychiatric: depression/suicidality in 25%; psychosis in ~10%; anxiety common
• Renal: hydronephrosis, hydroureter, atonic bladder (reflex nephropathy risk)
• Death: median ~39–40 yr (before modern supportive care); brainstem atrophy

Key Mutations (WFS1 biallelic LOF)
------------------------------------
* p.Leu432Pro (c.1295T>C): Lebanese founder; ER insertion domain; most common Lebanese
* c.1236_1239del (c.1236delATCA / p.His412fs): common European/pan-ethnic frameshift
* p.Arg558His (c.1673G>A): TM domain; severe; early onset; European
* p.Arg821Cys (c.2461C>T): central European founder; TM8 domain
* p.Arg456His (c.1367G>A): Turkish founder; TM domain; frequent Turkey/Middle East
* c.2051dupC (p.Gln684fs): frameshift; British; loss of last 3 TM domains
* p.Val779Met (c.2335G>A): missense; TM8; hypomorphic; later onset possible
* Splice_WFS1 (intron splice site variants): partial retention; severe in compound het
* Compound_heterozygous_WFS1: most common European genotype; one null + one missense

Diagnostics
-----------
• WFS1 biallelic sequencing: full coding region (8 exons); CNV array for del/dup
• MRI brain: brainstem atrophy, cerebellar volume loss, pontine signal change
• Visual: visual acuity, colour vision, VEP (prolonged latency), OCT (RNFL loss)
• Audiology: pure-tone audiometry (high-frequency SNHL)
• Endocrine: fasting glucose, HbA1c, C-peptide, GAD-Ab (negative), insulin
• DI testing: paired plasma/urine osmolality, water deprivation test, ADH levels
• Ophthalmology: visual fields, fundoscopy (pale optic disc, cupping)
• Renal: renal USS / urodynamics (dilated upper tract, atonic bladder)
• Biomarkers: plasma neutrophil gelatinase-associated lipocalin (NGAL) — elevated

Management (no disease-modifying Rx as of 2026)
------------------------------------------------
• DM: insulin (basal-bolus or CSII/pump); CGM; C-peptide monitoring
• DI: DDAVP (desmopressin) intranasal or oral; strict fluid balance
• OA: no proven neuroprotective Rx; low-vision aids; mobility aids
• Deafness: hearing aids; cochlear implant contraindicated (MRI surveillance)
• Neurological: physiotherapy, speech therapy, dysphagia diet modification
• Psychiatric: antidepressants; psychotherapy; suicidality risk monitoring
• Emerging: Sodium valproate ER-stress reduction (pilot; limited evidence);
  GLP-1RA (wolframin-independent beta-cell ER-stress reduction — speculative)
• Registry: Wolfram International Registry (critical for longitudinal data)

Wolfram Syndrome 2 (WFS2): CISD2 gene; 4q24; OMIM #604928 — NOT WFS1 (this file).

Cohort: 40 patients, seed=329.
"""

import random
import statistics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEED = 329
_COHORT_SIZE = 40

# WFS1 biallelic mutations — LOF (most common European + global)
_MUTATIONS = [
    "p.Leu432Pro (c.1295T>C) / p.Leu432Pro",    # Lebanese founder; homozygous
    "c.1236_1239del (p.His412fs) / missense",     # European frameshift + missense (compound het)
    "p.Arg558His (c.1673G>A) / null",             # TM domain; severe; compound het
    "p.Arg821Cys (c.2461C>T) / p.Arg821Cys",     # Central European founder homozygous
    "p.Arg456His (c.1367G>A) / null",             # Turkish founder; compound het
    "c.2051dupC (p.Gln684fs) / missense",         # British compound het
    "p.Val779Met (c.2335G>A) / splice",           # Hypomorphic; later onset; compound het
    "Splice_WFS1 / frameshift",                   # Splice site compound het
    "Novel_WFS1 / c.1236_1239del",                # Novel + known frameshift (compound)
]
_MUTATION_WEIGHTS = [0.18, 0.20, 0.15, 0.10, 0.12, 0.09, 0.06, 0.06, 0.04]

_ETHNICITIES = [
    "European (Central/Eastern)",
    "European (UK/Irish)",
    "Middle Eastern / Lebanese",
    "Turkish",
    "European (Other)",
    "South Asian",
    "North American European",
    "Other/Mixed",
]
_ETHNICITY_WEIGHTS = [0.20, 0.18, 0.15, 0.12, 0.12, 0.08, 0.09, 0.06]

# Main DIDMOAD features present in patient
_FEATURES = [
    "DM+OA",
    "DM+OA+DI",
    "DM+OA+D",
    "DM+OA+DI+D",      # Full DIDMOAD
    "DM+OA+DI+D+Neuro", # Full + neurological
    "DM+OA+D+Neuro",
]
_FEATURE_WEIGHTS = [0.12, 0.22, 0.10, 0.28, 0.18, 0.10]

_PSYCH = [
    "Depression (mild–moderate)",
    "Depression (severe with suicidality)",
    "Anxiety disorder",
    "Psychosis",
    "No psychiatric comorbidity",
]
_PSYCH_WEIGHTS = [0.28, 0.08, 0.15, 0.08, 0.41]

_NEURO_STATUS = [
    "None",
    "Cerebellar ataxia only",
    "Brainstem atrophy (MRI) + ataxia",
    "Dysarthria + dysphagia + ataxia",
    "Autonomic neuropathy (GI/bladder)",
]
_NEURO_WEIGHTS = [0.35, 0.22, 0.20, 0.12, 0.11]

_RENAL = [
    "Normal",
    "Mild hydronephrosis",
    "Hydronephrosis + hydroureter",
    "Atonic bladder + retention",
]
_RENAL_WEIGHTS = [0.50, 0.22, 0.16, 0.12]

_INSULIN_DELIVERY = [
    "Basal-bolus (MDI)",
    "CSII (insulin pump)",
    "Basal-bolus + CGM",
    "CSII + CGM",
]
_INSULIN_WEIGHTS = [0.38, 0.18, 0.28, 0.16]

_OA_STAGE = [
    "Early (VA ≥ 0.5; OCT RNFL loss)",
    "Moderate (VA 0.2–0.49; visual field loss)",
    "Severe (VA 0.05–0.19; marked field defect)",
    "Profound (VA < 0.05; functional blindness)",
]
_OA_WEIGHTS = [0.25, 0.30, 0.28, 0.17]

_MISDIAGNOSES = [
    "T1D (antibody-negative not checked)",
    "Alström Syndrome (misdiagnosis at diagnosis)",
    "None (correctly diagnosed Wolfram)",
    "T2D (rare; older teenager)",
    "Isolated optic neuritis / MS (OA onset first)",
]
_MISDIAG_WEIGHTS = [0.45, 0.05, 0.30, 0.08, 0.12]

_HEARING_STATUS = [
    "Normal",
    "Mild SNHL (25–40 dB HL)",
    "Moderate SNHL (41–70 dB HL)",
    "Severe SNHL (71–90 dB HL)",
]
_HEARING_WEIGHTS = [0.35, 0.30, 0.22, 0.13]


# ---------------------------------------------------------------------------
# Cohort builder
# ---------------------------------------------------------------------------

def _build_cohort() -> list:
    rng = random.Random(_SEED)

    def wchoice(choices, weights):
        return rng.choices(choices, weights=weights, k=1)[0]

    cohort = []
    for i in range(_COHORT_SIZE):
        mutation = wchoice(_MUTATIONS, _MUTATION_WEIGHTS)
        ethnicity = wchoice(_ETHNICITIES, _ETHNICITY_WEIGHTS)
        features = wchoice(_FEATURES, _FEATURE_WEIGHTS)
        psych = wchoice(_PSYCH, _PSYCH_WEIGHTS)
        neuro = wchoice(_NEURO_STATUS, _NEURO_WEIGHTS)
        renal = wchoice(_RENAL, _RENAL_WEIGHTS)
        insulin_mode = wchoice(_INSULIN_DELIVERY, _INSULIN_WEIGHTS)
        oa_stage = wchoice(_OA_STAGE, _OA_WEIGHTS)
        prior_misdiag = wchoice(_MISDIAGNOSES, _MISDIAG_WEIGHTS)
        hearing = wchoice(_HEARING_STATUS, _HEARING_WEIGHTS)

        # DM onset: mean ~6 yr; range 1–20
        dm_onset = round(rng.gauss(6.2, 3.2), 1)
        dm_onset = max(1.0, min(20.0, dm_onset))

        # OA onset: mean ~11 yr
        oa_onset = round(rng.gauss(10.8, 3.5), 1)
        oa_onset = max(2.0, min(24.0, max(dm_onset + 0.5, oa_onset)))

        # DI onset (if present): mean ~14 yr
        di_present = "DI" in features
        di_onset = None
        if di_present:
            di_onset = round(rng.gauss(14.0, 4.0), 1)
            di_onset = max(5.0, min(38.0, di_onset))

        # HbA1c: juvenile onset insulin-dependent; varied control
        hba1c = round(rng.gauss(8.4, 1.4), 1)
        hba1c = max(6.0, min(14.0, hba1c))

        # C-peptide: FALLS progressively — ER-stress apoptosis
        # Lower values = longer disease duration / more advanced
        disease_dur = round(rng.gauss(9.0, 5.0), 1)
        disease_dur = max(0.5, min(28.0, disease_dur))

        # C-peptide falls as disease progresses; most are insulin-dependent
        c_pep_base = max(0.01, 0.35 - (disease_dur * 0.018) + rng.gauss(0, 0.06))
        c_pep = round(c_pep_base, 3)
        c_pep = max(0.01, min(0.80, c_pep))

        # Current age: DM onset + disease duration (some dx as adults if late detection)
        current_age = round(dm_onset + disease_dur + rng.gauss(0, 1.5), 1)
        current_age = max(dm_onset + 0.5, min(50.0, current_age))

        # BMI: often low-normal (juvenile DM; malabsorption; DI fluid issues)
        bmi = round(rng.gauss(20.5, 3.0), 1)
        bmi = max(14.0, min(32.0, bmi))

        # Consanguinity (relevant for AR)
        consanguinity = rng.random() < 0.22

        # Family history (sibling or relative with Wolfram)
        family_hx = rng.random() < 0.28

        # On DDAVP (desmopressin) for DI
        on_ddavp = di_present and rng.random() < 0.92

        # DKA at DM diagnosis (can occur — not autoimmune but still DKA)
        dka_at_dx = rng.random() < 0.25

        cohort.append({
            "id": i + 1,
            "mutation": mutation,
            "ethnicity": ethnicity,
            "features": features,
            "psych": psych,
            "neuro": neuro,
            "renal": renal,
            "insulin_mode": insulin_mode,
            "oa_stage": oa_stage,
            "prior_misdiagnosis": prior_misdiag,
            "hearing_status": hearing,
            "dm_onset_yr": dm_onset,
            "oa_onset_yr": oa_onset,
            "di_onset_yr": di_onset,
            "di_present": di_present,
            "hba1c": hba1c,
            "c_peptide_nmol_L": c_pep,
            "c_peptide_label": "Falling (ER-stress apoptosis)",
            "disease_duration_yr": disease_dur,
            "current_age_yr": current_age,
            "bmi": bmi,
            "consanguinity": consanguinity,
            "family_hx": family_hx,
            "on_ddavp": on_ddavp,
            "dka_at_dx": dka_at_dx,
            "antibody_negative": True,  # always — not autoimmune
        })
    return cohort


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    cohort = _build_cohort()
    n = len(cohort)

    mean_dm_onset = round(statistics.mean(p["dm_onset_yr"] for p in cohort), 1)
    mean_hba1c = round(statistics.mean(p["hba1c"] for p in cohort), 1)
    pct_di = round(sum(1 for p in cohort if p["di_present"]) / n * 100, 1)
    pct_neuro = round(sum(1 for p in cohort if p["neuro"] != "None") / n * 100, 1)
    pct_psych = round(sum(1 for p in cohort if p["psych"] != "No psychiatric comorbidity") / n * 100, 1)
    pct_full_didmoad = round(sum(1 for p in cohort if "DI+D" in p["features"]) / n * 100, 1)
    pct_consang = round(sum(1 for p in cohort if p["consanguinity"]) / n * 100, 1)
    pct_t1d_misdiag = round(
        sum(1 for p in cohort if "T1D" in p["prior_misdiagnosis"]) / n * 100, 1)
    mean_c_pep = round(statistics.mean(p["c_peptide_nmol_L"] for p in cohort), 3)
    pct_dka = round(sum(1 for p in cohort if p["dka_at_dx"]) / n * 100, 1)

    kpis = {
        "gene":                    "WFS1",
        "chromosome":              "4p16.1",
        "syndrome":                "DIDMOAD",
        "inheritance":             "Autosomal Recessive",
        "mean_dm_onset":           f"{mean_dm_onset} yr",
        "mean_hba1c":              f"{mean_hba1c}%",
        "pct_di":                  f"{pct_di}%",
        "pct_neuro":               f"{pct_neuro}%",
        "pct_psych":               f"{pct_psych}%",
        "pct_full_didmoad":        f"{pct_full_didmoad}%",
        "pct_t1d_misdiag":         f"{pct_t1d_misdiag}%",
        "pct_consanguinity":       f"{pct_consang}%",
        "mean_c_peptide":          f"{mean_c_pep} nmol/L (falling)",
        "pct_dka_at_dx":           f"{pct_dka}%",
        "omim_gene":               "*606201",
        "omim_disease":            "#222300",
    }

    alerts = {
        "c_peptide_falls":
            "C-peptide FALLS progressively — ER-stress-driven beta-cell apoptosis (PERK-CHOP axis). "
            "Absolute insulin dependence from diagnosis. NOT preserved (unlike MODY1-13 and PNDM/TNDM).",
        "t1d_misdiagnosis":
            f"T1D misdiagnosis ~{pct_t1d_misdiag}%. Wolfram DM is antibody-negative (GADA, ZnT8, IA-2). "
            "KEY: antibody-negative juvenile-onset insulin-dependent DM + optic atrophy = Wolfram until proven otherwise.",
        "ophthalmology_urgent":
            "Optic atrophy (mean onset ~11 yr) progresses to functional blindness. Annual VEP + OCT + visual field. "
            "Wolfram OA irreversible — no neuroprotective Rx proven. Low-vision aids early.",
        "psychiatric_monitoring":
            "Depression with suicidality (~25%) is under-recognised in Wolfram. Annual psychiatric review. "
            "Psychosis ~10%. Multi-disciplinary team: psychiatry + neurology + endocrine.",
        "di_distinction":
            "Diabetes Insipidus (central, ADH-deficient) occurs in ~70% — polyuria is NOT from hyperglycaemia alone. "
            "Paired plasma/urine osmolality + water deprivation test distinguishes DI from diabetic osmotic diuresis. "
            "DDAVP (desmopressin) is effective treatment for central DI.",
        "no_su_relevant":
            "SU (sulfonylurea) is NOT indicated. Wolfram DM requires insulin from diagnosis (beta-cell apoptosis). "
            "No K-ATP channel defect — SU mechanism does not apply. Never substitute insulin with SU.",
    }

    key_facts = [
        "WFS1 encodes wolframin — 890-aa ER transmembrane glycoprotein; 9 TM domains; Chr 4p16.1; OMIM *606201",
        "Biallelic WFS1 LOF → wolframin absent → ER Ca²⁺ homeostasis lost → chronic ER stress → UPR → multi-organ apoptosis",
        "DIDMOAD temporal cascade: DM (~6 yr) → Optic Atrophy (~11 yr) → Diabetes Insipidus (~14 yr) → Deafness (~16 yr)",
        "C-peptide FALLS progressively — beta-cell ER-stress apoptosis (CHOP); unlike MODY1-13 where C-pep is preserved",
        "T1D misdiagnosis ~45% — antibody-negative juvenile-onset insulin-dependent DM + OA = Wolfram first",
        "Optic atrophy (OA): bilateral progressive optic neuropathy; OCT + VEP + visual field; no proven neuroprotectant",
        "Diabetes Insipidus: central (ADH deficiency); ~70%; polyuria + polydipsia independent of DM; treat with DDAVP",
        "Sensorineural deafness: high-frequency SNHL; ~65%; audiometry monitoring annually",
        "Neurological: cerebellar ataxia, brainstem atrophy, dysarthria, dysphagia (onset ~20–30 yr)",
        "Psychiatric: depression/suicidality (25%), psychosis (10%) — annual psychiatric review mandatory",
        "Renal: upper tract dilatation, atonic bladder (~50%) — renal USS + urodynamics",
        "Autosomal Recessive — both alleles LOF required; consanguinity increases risk; siblings at 25% risk",
    ]

    patients_preview = [
        {
            "id": p["id"],
            "mutation": p["mutation"].split("/")[0].strip(),
            "features": p["features"],
            "dm_onset": p["dm_onset_yr"],
            "hba1c": p["hba1c"],
            "c_peptide": round(p["c_peptide_nmol_L"], 2),
            "oa_stage": p["oa_stage"].split("(")[0].strip(),
            "hearing": p["hearing_status"].split("(")[0].strip(),
            "neuro": p["neuro"].split("(")[0].strip() if p["neuro"] != "None" else "—",
            "dka_at_dx": p["dka_at_dx"],
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

    # Feature distribution (DIDMOAD combinations)
    feat_dist: dict = {}
    for p in cohort:
        feat_dist[p["features"]] = feat_dist.get(p["features"], 0) + 1

    # Mutation distribution
    mut_dist: dict = {}
    for p in cohort:
        short = p["mutation"].split("/")[0].strip()
        mut_dist[short] = mut_dist.get(short, 0) + 1

    # Ethnicity
    eth_dist: dict = {}
    for p in cohort:
        eth_dist[p["ethnicity"]] = eth_dist.get(p["ethnicity"], 0) + 1

    # OA stage
    oa_dist: dict = {}
    for p in cohort:
        oa_dist[p["oa_stage"]] = oa_dist.get(p["oa_stage"], 0) + 1

    # Neurological status
    neuro_dist: dict = {}
    for p in cohort:
        neuro_dist[p["neuro"]] = neuro_dist.get(p["neuro"], 0) + 1

    # Psychiatric
    psych_dist: dict = {}
    for p in cohort:
        psych_dist[p["psych"]] = psych_dist.get(p["psych"], 0) + 1

    # HbA1c tiers
    hba1c_tiers = {"< 7.5%": 0, "7.5–8.9%": 0, "9.0–10.9%": 0, "≥ 11.0%": 0}
    for p in cohort:
        v = p["hba1c"]
        if v < 7.5:
            hba1c_tiers["< 7.5%"] += 1
        elif v < 9.0:
            hba1c_tiers["7.5–8.9%"] += 1
        elif v < 11.0:
            hba1c_tiers["9.0–10.9%"] += 1
        else:
            hba1c_tiers["≥ 11.0%"] += 1

    # C-peptide tiers (FALLING pattern)
    cp_tiers = {
        "< 0.10 nmol/L (Absent)": 0,
        "0.10–0.19 nmol/L (Minimal)": 0,
        "0.20–0.34 nmol/L (Low-falling)": 0,
        "0.35–0.59 nmol/L (Reduced)": 0,
        "≥ 0.60 nmol/L (Partially preserved)": 0,
    }
    for p in cohort:
        v = p["c_peptide_nmol_L"]
        if v < 0.10:
            cp_tiers["< 0.10 nmol/L (Absent)"] += 1
        elif v < 0.20:
            cp_tiers["0.10–0.19 nmol/L (Minimal)"] += 1
        elif v < 0.35:
            cp_tiers["0.20–0.34 nmol/L (Low-falling)"] += 1
        elif v < 0.60:
            cp_tiers["0.35–0.59 nmol/L (Reduced)"] += 1
        else:
            cp_tiers["≥ 0.60 nmol/L (Partially preserved)"] += 1

    # DM onset tiers
    dm_onset_tiers = {
        "< 4 yr (Neonatal/infant)": 0,
        "4–7 yr (Early childhood)": 0,
        "8–13 yr (Late childhood)": 0,
        "> 13 yr (Adolescent+)": 0,
    }
    for p in cohort:
        v = p["dm_onset_yr"]
        if v < 4:
            dm_onset_tiers["< 4 yr (Neonatal/infant)"] += 1
        elif v < 8:
            dm_onset_tiers["4–7 yr (Early childhood)"] += 1
        elif v <= 13:
            dm_onset_tiers["8–13 yr (Late childhood)"] += 1
        else:
            dm_onset_tiers["> 13 yr (Adolescent+)"] += 1

    # Hearing status
    hearing_dist: dict = {}
    for p in cohort:
        hearing_dist[p["hearing_status"]] = hearing_dist.get(p["hearing_status"], 0) + 1

    # Renal status
    renal_dist: dict = {}
    for p in cohort:
        renal_dist[p["renal"]] = renal_dist.get(p["renal"], 0) + 1

    # Insulin delivery
    insulin_dist: dict = {}
    for p in cohort:
        insulin_dist[p["insulin_mode"]] = insulin_dist.get(p["insulin_mode"], 0) + 1

    # Misdiagnosis
    mis_dist: dict = {}
    for p in cohort:
        mis_dist[p["prior_misdiagnosis"]] = mis_dist.get(p["prior_misdiagnosis"], 0) + 1

    # Disease duration tiers
    dur_tiers = {"< 3 yr": 0, "3–7 yr": 0, "8–15 yr": 0, "> 15 yr": 0}
    for p in cohort:
        v = p["disease_duration_yr"]
        if v < 3:
            dur_tiers["< 3 yr"] += 1
        elif v <= 7:
            dur_tiers["3–7 yr"] += 1
        elif v <= 15:
            dur_tiers["8–15 yr"] += 1
        else:
            dur_tiers["> 15 yr"] += 1

    summary_flags = {
        "pct_di": round(sum(1 for p in cohort if p["di_present"]) / n * 100, 1),
        "pct_neuro": round(sum(1 for p in cohort if p["neuro"] != "None") / n * 100, 1),
        "pct_psych": round(sum(1 for p in cohort if p["psych"] != "No psychiatric comorbidity") / n * 100, 1),
        "pct_hearing_loss": round(sum(1 for p in cohort if p["hearing_status"] != "Normal") / n * 100, 1),
        "pct_renal": round(sum(1 for p in cohort if p["renal"] != "Normal") / n * 100, 1),
        "pct_t1d_misdiagnosis": round(
            sum(1 for p in cohort if "T1D" in p["prior_misdiagnosis"]) / n * 100, 1),
        "pct_consanguinity": round(sum(1 for p in cohort if p["consanguinity"]) / n * 100, 1),
        "pct_antibody_negative": 100.0,
        "pct_dka_at_dx": round(sum(1 for p in cohort if p["dka_at_dx"]) / n * 100, 1),
    }

    return {
        "feature_distribution":       feat_dist,
        "mutation_distribution":      mut_dist,
        "ethnicity_distribution":     eth_dist,
        "oa_stage_distribution":      oa_dist,
        "neurological_distribution":  neuro_dist,
        "psychiatric_distribution":   psych_dist,
        "hba1c_tiers":                hba1c_tiers,
        "c_peptide_tiers":            cp_tiers,
        "dm_onset_tiers":             dm_onset_tiers,
        "hearing_distribution":       hearing_dist,
        "renal_distribution":         renal_dist,
        "insulin_delivery":           insulin_dist,
        "misdiagnosis_distribution":  mis_dist,
        "disease_duration_tiers":     dur_tiers,
        "summary_flags":              summary_flags,
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "full_name": "Wolfram Syndrome 1 (WFS1) — DIDMOAD",
            "acronym": "DIDMOAD: Diabetes Insipidus, Diabetes Mellitus, Optic Atrophy, Deafness",
            "gene": "WFS1 (Wolframin ER Transmembrane Glycoprotein); 890 aa; ~100 kDa; 4p16.1; OMIM *606201",
            "disease_omim": "#222300 (Wolfram Syndrome 1 / DIDMOAD)",
            "inheritance": "Autosomal Recessive — biallelic LOF; compound heterozygous common; 50% sibling risk",
            "prevalence": "~1/770,000 worldwide; 1/100,000 some populations; rare but panethnic",
            "mechanism": (
                "WFS1 biallelic LOF → wolframin absent → ER Ca²⁺ homeostasis disrupted → "
                "chronic unresolved ER stress → sustained UPR activation → "
                "PERK-eIF2α-ATF4-CHOP axis → progressive apoptosis of: "
                "pancreatic beta-cells (DM), retinal ganglion cells (OA), hypothalamic neurons (DI), "
                "cochlear hair cells (D), brainstem/cerebellar neurons (ataxia). "
                "Same ER-stress mechanism; organ-specific vulnerability explains temporal cascade."
            ),
            "protein_function": (
                "WFS1/Wolframin: 890 aa, ~100 kDa. ER membrane glycoprotein; 9 transmembrane domains. "
                "Functions: ER Ca²⁺ homeostasis (SERCA pump regulation); "
                "IRE1α ubiquitination and UPR modulation; "
                "ER membrane integrity; IP3R-SERCA Ca²⁺ microdomain maintenance. "
                "LOF disrupts ER Ca²⁺ buffering → all ER-stressed cell types progressively die."
            ),
            "temporal_cascade": (
                "DM: ~6 yr (range 1–20 yr) → "
                "OA: ~11 yr (range 2–24 yr) → "
                "DI: ~14 yr (~70% of patients) → "
                "Deafness: ~16 yr (~65% of patients) → "
                "Neurological: ~20–30 yr → "
                "Psychiatric: ~20 yr → "
                "Death: median ~39–40 yr (brainstem atrophy, respiratory failure)"
            ),
            "c_peptide_pattern": (
                "C-peptide FALLS progressively. Beta-cell apoptosis driven by ER stress (CHOP). "
                "Absolute insulin dependence from DM diagnosis. "
                "Contrasts sharply with ALL MODY types (C-pep preserved) and PNDM/TNDM (variable). "
                "Very similar mechanism to MODY10/INS ER-stress (but earlier onset and multi-organ)."
            ),
            "treatment": (
                "No disease-modifying therapy as of 2026. Multisystem supportive care: "
                "insulin (basal-bolus/CSII + CGM); DDAVP for DI; low-vision aids; hearing aids; "
                "physiotherapy/speech therapy for neurological decline; "
                "psychiatric monitoring (suicidality); renal USS + urodynamics. "
                "Wolfram International Registry: critical for longitudinal evidence."
            ),
            "autoantibodies": "NEGATIVE (GADA, ZnT8, IA-2) — always; Wolfram DM is not autoimmune",
            "family_history": "~28% (sibling or relative with Wolfram); consanguinity ~22%",
            "misdiagnosis_rate": (
                "T1D ~45% (most common — juvenile-onset antibody-negative insulin-dependent DM). "
                "Optic neuritis/MS ~12% (OA onset before DM diagnosis). "
                "Alström syndrome ~5% (ALMS1 gene; overlapping multisystem features; distinguished by NGS)."
            ),
        },

        "genes_and_proteins": {
            "WFS1 (*606201)": (
                "4p16.1. Wolframin ER Transmembrane Glycoprotein. 890 aa, ~100 kDa. "
                "9 transmembrane domains; N-terminus cytoplasmic; C-terminus ER lumen. "
                "ER-resident; regulates Ca²⁺ homeostasis and UPR. "
                "LOF → chronic ER stress → PERK-CHOP mediated apoptosis in multiple cell types. "
                "8 coding exons; hotspot exon 8 (encodes TM domains 5–9); most mutations here."
            ),
            "WFS2 (CISD2, *611507)": (
                "4q24; CDGSH Iron-Sulfur Domain Protein 2 (Miner1). OMIM Disease #604928. "
                "Distinct from WFS1. AR. Features: Wolfram-like DM + optic atrophy but NO deafness; "
                "peptic ulcer; bleeding tendency (platelet dysfunction). "
                "ER inter-membrane space protein; Ca²⁺ regulation / mitochondria-ER crosstalk. "
                "Rarer than WFS1; distinct genotype-phenotype. NOT covered here (WFS1 file only)."
            ),
            "Wolframin in ER stress (UPR arms)": (
                "Wolframin modulates all 3 UPR sensor arms: "
                "PERK (protein kinase R-like ER kinase): wolframin-LOF → sustained PERK → eIF2α phosphorylation → ATF4 → CHOP → apoptosis. "
                "IRE1α: wolframin ubiquitinates/degrades IRE1α to limit JNK/apoptosis — LOF → IRE1α accumulates → JNK activation. "
                "ATF6: wolframin-LOF → ATF6 cleavage altered → pro-apoptotic gene expression. "
                "Net result: unresolvable ER stress → cell-type-specific apoptosis cascade."
            ),
        },

        "clinical_terms": {
            "DIDMOAD": (
                "Acronym for 4 cardinal features of Wolfram Syndrome 1: "
                "Diabetes Insipidus (central; hypothalamic; AVP/ADH deficiency; ~70%), "
                "Diabetes Mellitus (insulin-requiring; juvenile onset; ~6 yr; antibody-negative), "
                "Optic Atrophy (bilateral progressive; retinal ganglion cell loss; ~11 yr), "
                "Deafness (sensorineural; high-frequency SNHL; ~65% of patients). "
                "Not all patients develop all 4 features; DM + OA sufficient for diagnosis."
            ),
            "Central Diabetes Insipidus (DI)": (
                "AVP/ADH deficiency from hypothalamic neuron apoptosis (ER stress). "
                "Polyuria + polydipsia independent from diabetic glycosuria. "
                "Distinguish from nephrogenic DI (renal insensitivity) — Wolfram is CENTRAL. "
                "Diagnosis: paired plasma/urine osmolality; water deprivation test; ADH levels. "
                "Treatment: DDAVP (desmopressin) oral/intranasal; strict fluid intake monitoring."
            ),
            "Optic Atrophy (OA) in Wolfram": (
                "Progressive bilateral optic neuropathy from retinal ganglion cell ER-stress apoptosis. "
                "Mean onset ~11 yr. OCT: retinal nerve fibre layer (RNFL) thinning. "
                "VEP: prolonged P100 latency, reduced amplitude. Visual field: arcuate/central scotoma. "
                "Progresses to functional blindness in severe cases (~17% in cohort). "
                "No proven neuroprotective therapy. Low-vision aids, orientation/mobility training."
            ),
            "ER Stress / UPR": (
                "Unfolded Protein Response: cellular response to ER protein folding overload. "
                "Adaptive UPR (acute): reduces translation, upregulates chaperones (BiP/GRP78), clears misfolded proteins. "
                "Maladaptive UPR (chronic): PERK-eIF2α-ATF4-CHOP axis → transcription of pro-apoptotic genes. "
                "Wolfram: wolframin-LOF → chronic unresolvable ER stress → maladaptive UPR → cell death. "
                "Same mechanism as MODY10 (INS ER-stress from misfolded proinsulin) but multi-organ."
            ),
            "Wolfram vs Alström Syndrome": (
                "Both AR multi-system syndromes with DM + sensorineural deafness + visual impairment. "
                "Alström (ALMS1; Chr 2p13.1): SNHL + progressive cone-rod dystrophy (not classic OA) + "
                "dilated cardiomyopathy (childhood onset) + truncal obesity + T2D-like (insulin resistance). "
                "Wolfram (WFS1): progressive OA + central DI + progressive neurodegeneration; "
                "no cardiomyopathy; no obesity; insulin-dependent DM (beta-cell apoptosis). "
                "Distinguish by: WFS1 + ALMS1 gene panels; cardiomyopathy; obesity; DI presence."
            ),
        },

        "lab_thresholds": {
            "C-peptide (Wolfram DM)":     "Falling progressively; typically < 0.35 nmol/L (L > 5 yr duration); absolute insulin dependence",
            "HbA1c target":               "< 7.5% (57 mmol/mol); CGM preferred for time-in-range optimisation",
            "GADA / ZnT8-Ab / IA-2":      "ALL NEGATIVE — Wolfram DM is not autoimmune",
            "Plasma osmolality (DI)":     "> 295 mOsm/kg with urine osmolality < 300 mOsm/kg → central DI suspected",
            "DDAVP response (DI)":        "Urine osmolality rises ≥ 50% after DDAVP → central DI confirmed",
            "OCT RNFL (OA)":             "< 70 µm average RNFL = significant thinning; < 50 µm = severe atrophy",
            "Audiometry (SNHL)":          "25–40 dB HL mild; 41–70 dB moderate; > 70 dB severe/profound SNHL",
            "MRI brain (brainstem)":      "Pontine/cerebellar atrophy; T2 signal change in dorsal brainstem",
            "Renal USS":                  "Dilated pelvis / hydroureter; atonic bladder; post-void residual > 100 mL",
        },

        "treatment": {
            "diabetes_insulin": (
                "Insulin is ALWAYS required in Wolfram DM. Beta-cell apoptosis = absolute insulin deficiency. "
                "SU not indicated (no K-ATP defect). GLP-1RA: may reduce ER stress (in vitro/preclinical) but "
                "not proven clinically. Basal-bolus MDI or CSII (pump) + CGM standard of care. "
                "C-peptide monitoring annually tracks progression of beta-cell loss."
            ),
            "di_treatment": (
                "DDAVP (desmopressin) oral tablets or intranasal spray: first-line for central DI. "
                "Start low: DDAVP 0.1 mg oral bd; titrate to symptom control. Monitor serum sodium (hyponatraemia risk). "
                "Fluid restriction + DDAVP: critical to avoid dilutional hyponatraemia. "
                "Discontinue DDAVP if hospitalised without adequate fluid monitoring."
            ),
            "optic_atrophy_management": (
                "No disease-modifying neuroprotective agent proven in clinical trials (2026). "
                "Sodium valproate: ER-stress modulation (pilot UK trial; limited benefit). "
                "Ophthalmology: annual VEP + OCT + visual fields. Low-vision rehabilitation early. "
                "Avoid retinal toxins (hydroxychloroquine, amiodarone, tobacco, high-dose supplements)."
            ),
            "neurological_support": (
                "Physiotherapy: cerebellar ataxia balance + gait training. "
                "Speech and language therapy: dysarthria + dysphagia assessment; modified diet textures. "
                "PEG feeding consideration for severe dysphagia. "
                "Autonomic neuropathy: postural hypotension management; GI prokinetics. "
                "Occupational therapy: assistive devices for ataxia + visual impairment."
            ),
            "psychiatric_management": (
                "Annual psychiatric assessment from teenage years. "
                "Suicidality risk (25%): PHQ-9 + specialist review; antidepressants (SSRI first-line). "
                "Psychosis (~10%): low-dose antipsychotic; avoid olanzapine (glucose effects). "
                "Coping strategies for multi-system disease burden (chronic grief + disability adjustment). "
                "Family therapy and peer support groups (Wolfram Syndrome Association)."
            ),
            "genetic_counselling": (
                "AR inheritance: both parents are obligate carriers (25% sibling risk). "
                "Cascade testing: WFS1 biallelic sequencing for siblings. "
                "Pre-natal diagnosis available (CVS/amniocentesis) if family mutations known. "
                "Carrier testing for partners of affected individuals when family planning. "
                "Register with Wolfram International Registry for longitudinal follow-up."
            ),
        },

        "diagnostics": {
            "WFS1_sequencing": (
                "Full coding sequence (exons 1–8) + splice site analysis; Chr 4p16.1. "
                "Exon 8 hotspot: encodes TM domains 5–9; most pathogenic missense + frameshift. "
                "CNV array or MLPA: for large deletions/duplications (missed by Sanger). "
                "Functional annotation: variant in silico + patient fibroblast ER-stress assay. "
                "Mandatory: biallelic variants required for Wolfram 1 diagnosis (AR)."
            ),
            "wolfram2_distinction": (
                "If WFS1 biallelic negative but Wolfram phenotype: test CISD2 (Wolfram 2). "
                "WFS2 (CISD2): peptic ulcer + bleeding tendency distinguish it from WFS1. "
                "Next-gen panel: WFS1 + CISD2 + differential (ALMS1, BBS, NDM panels as appropriate)."
            ),
            "mri_protocol": (
                "MRI brain + brainstem: thin slice T2/FLAIR axial + sagittal midline. "
                "Look for: pontine atrophy, cerebellar volume loss, pontine T2 signal change. "
                "Baseline at diagnosis; repeat every 2–3 yr or when neurological symptoms develop. "
                "MRI contraindicated if cochlear implant (check WFS1 patient for implant before imaging)."
            ),
            "ophthalmological_workup": (
                "Annual: best-corrected visual acuity (BCVA), colour vision (Ishihara), contrast sensitivity. "
                "OCT: RNFL + macular ganglion cell analysis. VEP: pattern + flash. "
                "Automated visual fields (Humphrey 24-2). Fundoscopy (optic disc pallor). "
                "Baseline at DM diagnosis (OA may be subclinical initially)."
            ),
        },

        "comparison_mody10_wolfram": {
            "MODY10 (INS)": {
                "gene":        "INS (Insulin); 11p15.5; preproinsulin",
                "mechanism":   "Dominant-negative misfolded proinsulin → ER stress → CHOP apoptosis",
                "onset":       "Teens–adult (AD; mean ~25 yr); NOT neonatal in MODY10",
                "c_peptide":   "FALLS progressively (ER-stress apoptosis — same mechanism)",
                "inheritance": "Autosomal Dominant (heterozygous dominant-negative)",
                "features":    "Diabetes only (no OA, DI, deafness, neuro)",
                "treatment":   "Insulin required (70–80%); SU ineffective",
            },
            "Wolfram 1 (WFS1)": {
                "gene":        "WFS1 (Wolframin); 4p16.1; ER membrane glycoprotein",
                "mechanism":   "Biallelic LOF → wolframin absent → ER Ca²⁺ loss → multi-organ ER stress",
                "onset":       "Childhood (~6 yr for DM); multi-organ cascade over decades",
                "c_peptide":   "FALLS progressively (same ER-stress / CHOP apoptosis mechanism)",
                "inheritance": "Autosomal Recessive (biallelic; compound-het common)",
                "features":    "DIDMOAD + neuro + psychiatric + renal (multi-system)",
                "treatment":   "Insulin always; DDAVP for DI; multidisciplinary supportive",
            },
        },
    }
