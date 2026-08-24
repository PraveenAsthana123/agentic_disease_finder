#!/usr/bin/env python3
"""SLC6A8 Creatine Transporter Deficiency (CCDS1) Epilepsy Dashboard.

SLC6A8 encodes a Na+/Cl−-dependent creatine transporter (635 aa), a member of the
SLC6 neurotransmitter transporter superfamily expressed ubiquitously (brain, muscle,
kidney).  CCDS1 is X-linked (Xq28), making hemizygous males the most severely affected,
while heterozygous females show highly variable penetrance via X-inactivation (Lyon
mosaicism).

CREATINE BIOSYNTHESIS & TRANSPORT PATHWAY (three-step):

  Step 1 — AGAT (kidney/pancreas):
    L-Arginine + Glycine → L-Ornithine + Guanidinoacetate (GAA)

  Step 2 — GAMT (liver/pancreas):
    SAM + GAA → SAH + Creatine

  Step 3 — SLC6A8 (ubiquitous):     [← SLC6A8 DEFICIENCY BLOCKS HERE]
    Plasma Creatine ──× CANNOT ENTER CELLS → creatine accumulates in blood / urine

SLC6A8 LOF — TRANSPORT FAILURE (distinct from biosynthesis failure in AGAT/GAMT):

  CREATINE IS MADE BUT CANNOT ENTER BRAIN / MUSCLE:
    Plasma creatine: ELEVATED or NORMAL-HIGH (accumulates because can't enter cells)
    Urine creatine: MARKEDLY ELEVATED (spills into urine — PATHOGNOMONIC)
    Urine creatine/creatinine ratio: CRITICALLY ELEVATED >1.0 (normal <0.15 in males)
    Urine creatinine: LOW-NORMAL (creatinine is creatine catabolite; formed INSIDE cells
                                   → when creatine can't enter → creatinine REDUCED)
    Brain H-MRS Creatine peak (3.0 ppm): ABSENT — shared PATHOGNOMONIC with GAMT & AGAT

  GAA: NORMAL (AGAT is intact → GAA biosynthesis normal)
  Methionine, tHcy, SAM, MMA: ALL NORMAL (biosynthesis pathways intact)

TREATMENT FAILURE — THE KEY CLINICAL DISTINCTION vs AGAT & GAMT:

  Creatine monohydrate (standard oral creatine): LARGELY INEFFECTIVE IN MALES
    → The transporter that would normally import creatine into brain/muscle is absent
    → Plasma creatine rises further (already elevated), no brain benefit
    → Females with mild-moderate disease may show partial response (mosaic normal cells)

  Alternative experimental strategies:
    - Guanidinoacetate (GAA) supplementation (enters via SLC6A6/BGT-1, bypasses SLC6A8)
    - Cyclocreatine (creatine analogue using alternative transporters, experimental)
    - Creatine ethyl ester: INEFFECTIVE (still requires SLC6A8 or cell membrane integration)
    - Arginine + glycine supplementation: may increase GAA/creatine synthesis but delivery fails

PREVALENCE & GENETICS:
  ~300–500 cases worldwide (2026); most common of 3 CCDS (CCDS1)
  X-linked (Xq28); hemizygous males severely affected; carrier females 50–70% symptomatic
  de novo mutations ~30–40%
  OMIM Gene: 300036 (SLC6A8); Disease: 300352 (CCDS1-CTDS)
"""

import random

# ---------------------------------------------------------------------------
# Deterministic cohort — seed 127
# ---------------------------------------------------------------------------
_rng = random.Random(127)

PHENOTYPE_CLASSES = [
    "Severe-Hemizygous-Male",      # 60% — classic X-linked, profound IDD, drug-resistant epilepsy
    "Moderate-Carrier-Female",     # 30% — variable X-inactivation, learning difficulties to IDD
    "Mild-Attenuated",             # 10% — partial function, mild learning difficulties
]
PHENO_WEIGHTS = [0.60, 0.30, 0.10]

# Key X-linked variants (SLC6A8, 635 aa, Xq28)
VARIANTS = [
    "p.Gly132Val",      # catalytic domain — most common ~20%; active site, severe transport failure
    "p.Asp126Asn",      # substrate-binding pocket — severe ~18%
    "p.Arg365Trp",      # TM helix 6/7 junction — severe ~16%
    "p.Tyr389Cys",      # Na+-binding site — moderate-severe ~14%
    "p.Pro544Leu",      # C-terminal cytoplasmic tail — moderate ~12%
    "c.IVS5+1G>A",      # splice-site null — severe ~10%
    "p.Arg513Gly",      # intracellular loop — moderate ~8%
    "Large Xq28 del",   # chromosomal deletion — severe/complex phenotype ~7%
    "p.Thr411Ile",      # TM helix — mild ~5%: partial transport activity retained
    "p.Ala478Thr",      # mild — partial function ~5%
]
VARIANT_WEIGHTS = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.05, 0.05]

# Adjust weights to sum to 1.0 (they already sum to 1.00)

SEIZURE_TYPES = [
    "Focal seizures (frontal/temporal — creatine energy deficit in localised cortex)",
    "Generalized tonic-clonic (diffuse creatine energy failure)",
    "Infantile spasms / West syndrome (early severe brain creatine depletion)",
    "Absence / myoclonic (partial energy deficit, cortical hyperexcitability)",
    "Febrile / stress-induced (metabolic vulnerability during catabolic states)",
    "Epileptic encephalopathy (continuous spike-wave / CSWS pattern)",
]


def _choose_pheno():
    r = _rng.random()
    cum = 0.0
    for ph, w in zip(PHENOTYPE_CLASSES, PHENO_WEIGHTS):
        cum += w
        if r < cum:
            return ph
    return PHENOTYPE_CLASSES[-1]


def _choose_variant(pheno):
    is_mild   = "Mild" in pheno
    is_female = "Female" in pheno
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            if is_mild and v in ("p.Gly132Val", "p.Asp126Asn", "c.IVS5+1G>A", "Large Xq28 del"):
                return _rng.choice(["p.Thr411Ile", "p.Ala478Thr", "p.Arg513Gly"])
            return v
    return VARIANTS[0]


def _make_patient(idx):
    pheno      = _choose_pheno()
    is_severe  = "Severe" in pheno     # hemizygous male
    is_female  = "Female" in pheno     # carrier female, moderate
    is_mild    = "Mild"   in pheno     # attenuated

    sex = "F" if is_female else "M"

    # ---------------------------------------------------------------------------
    # KEY BIOMARKER SIGNATURE — TRANSPORT FAILURE (opposite of biosynthesis failure)
    # ---------------------------------------------------------------------------
    # Plasma creatine: ELEVATED (can't enter cells, accumulates in blood)
    if is_severe:
        creatine_plasma = round(_rng.uniform(90,  200), 1)   # ELEVATED; normal male 20–80 µmol/L
        urine_creatine  = round(_rng.uniform(800, 3500), 0)  # MARKEDLY ELEVATED (µmol/mmol crCr)
        urine_cr_ratio  = round(_rng.uniform(1.2, 5.0), 2)   # CRITICALLY ELEVATED; normal <0.15
        urine_creatinine = round(_rng.uniform(5,  20), 1)    # LOW (can't produce inside cells)
        ck_u_l          = round(_rng.uniform(80, 300), 0)    # mildly elevated (muscle creatine deficient)
    elif is_female:
        creatine_plasma = round(_rng.uniform(50, 120), 1)    # mildly elevated or normal-high
        urine_creatine  = round(_rng.uniform(200, 900), 0)
        urine_cr_ratio  = round(_rng.uniform(0.3, 1.5), 2)
        urine_creatinine = round(_rng.uniform(15,  50), 1)
        ck_u_l          = round(_rng.uniform(45, 150), 0)
    else:  # mild
        creatine_plasma = round(_rng.uniform(30,  80), 1)    # borderline
        urine_creatine  = round(_rng.uniform(80,  300), 0)
        urine_cr_ratio  = round(_rng.uniform(0.2, 0.8), 2)
        urine_creatinine = round(_rng.uniform(25,  70), 1)
        ck_u_l          = round(_rng.uniform(30, 100), 0)

    # GAA: NORMAL (AGAT biosynthesis intact)
    gaa_umol = round(_rng.uniform(0.6, 2.8), 2)   # normal 0.5–3.0 µmol/L
    # Methionine: NORMAL (SAM pathway not involved)
    methionine   = round(_rng.uniform(18, 42), 1)
    # tHcy: NORMAL
    hcy_umol     = round(_rng.uniform(5,  14), 1)
    # SAM: NORMAL
    sam_umol     = round(_rng.uniform(60, 125), 1)
    # MMA: NORMAL
    mma_normal   = True

    # ---------------------------------------------------------------------------
    # Clinical features
    # ---------------------------------------------------------------------------
    if is_severe:
        sz_prob  = 0.88
        dr_prob  = 0.58   # drug-resistant — intermediate between GAMT (60-80%) and AGAT (25-35%)
        idd_prob = 0.99
        speech_abs = 0.75
        autism  = 0.75
        behav   = 0.85   # hyperactivity/ADHD-like
        creatine_tx_response = False  # standard creatine largely ineffective in males
    elif is_female:
        sz_prob  = 0.38
        dr_prob  = 0.25
        idd_prob = 0.60
        speech_abs = 0.35
        autism  = 0.40
        behav   = 0.50
        creatine_tx_response = _rng.random() < 0.35  # partial mosaic benefit
    else:  # mild
        sz_prob  = 0.15
        dr_prob  = 0.08
        idd_prob = 0.30
        speech_abs = 0.12
        autism  = 0.18
        behav   = 0.30
        creatine_tx_response = _rng.random() < 0.55

    seizures       = _rng.random() < sz_prob
    drug_resistant = seizures and _rng.random() < dr_prob
    idd            = _rng.random() < idd_prob
    speech_absent  = _rng.random() < speech_abs
    autism_like    = _rng.random() < autism
    behavioral     = _rng.random() < behav
    nbs_detected   = _rng.random() < (0.45 if is_severe else (0.35 if is_female else 0.20))
    gaa_tx_tried   = _rng.random() < (0.30 if is_severe else 0.10)  # GAA supplementation experimental

    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    age_onset_mo = (
        _rng.randint(3,  18) if is_severe  else
        _rng.randint(12, 48) if is_female  else
        _rng.randint(24, 96)
    )

    variant = _choose_variant(pheno)

    return {
        "id":                         f"SLC6A8-{idx:03d}",
        "phenotype":                  pheno,
        "sex":                        sex,
        "variant":                    variant,
        "age_onset_months":           age_onset_mo,
        # KEY BIOMARKERS — TRANSPORT FAILURE
        "creatine_plasma_umol_l":     creatine_plasma,     # ELEVATED — accumulates (can't enter cells)
        "urine_creatine_umol_mmol":   urine_creatine,      # MARKEDLY ELEVATED — pathognomonic
        "urine_cr_creatinine_ratio":  urine_cr_ratio,      # CRITICALLY ELEVATED — primary diagnostic
        "urine_creatinine_umol_l":    urine_creatinine,    # LOW-NORMAL (catabolite absent)
        "gaa_umol_l":                 gaa_umol,            # NORMAL — KEY NEGATIVE (biosynthesis intact)
        "methionine_umol_l":          methionine,          # NORMAL
        "homocysteine_umol_l":        hcy_umol,            # NORMAL
        "sam_umol_l":                 sam_umol,            # NORMAL
        "ck_u_l":                     ck_u_l,
        "mma_normal":                 mma_normal,
        "mecbl_normal":               True,
        "adocbl_normal":              True,
        # Clinical
        "seizures":                   seizures,
        "seizure_type":               seizure_type,
        "drug_resistant_sz":          drug_resistant,
        "idd":                        idd,
        "speech_absent":              speech_absent,
        "autism_like":                autism_like,
        "behavioral_hyperactivity":   behavioral,
        "nbs_detected":               nbs_detected,
        "creatine_tx_response":       creatine_tx_response,  # mostly False in severe males
        "gaa_tx_tried":               gaa_tx_tried,
    }


_PATIENTS = [_make_patient(i + 1) for i in range(40)]


# ---------------------------------------------------------------------------
# API response functions
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    n = len(_PATIENTS)

    def avg(key):
        vals = [p[key] for p in _PATIENTS if isinstance(p.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else 0

    def pct(pred):
        return round(100 * sum(1 for p in _PATIENTS if pred(p)) / n, 1)

    males   = [p for p in _PATIENTS if p["sex"] == "M"]
    females = [p for p in _PATIENTS if p["sex"] == "F"]
    severe  = [p for p in _PATIENTS if "Severe" in p["phenotype"]]

    avg_cr_ratio_males = round(
        sum(p["urine_cr_creatinine_ratio"] for p in males) / len(males), 2
    ) if males else 0

    return {
        "title": "SLC6A8 Creatine Transporter Deficiency — CCDS1 Dashboard",
        "subtitle": (
            "Cerebral Creatine Deficiency Syndrome 1 (CCDS1) · X-linked Xq28 · "
            "Creatine transport failure · Most common of 3 CCDS · ~300–500 cases worldwide (2026)"
        ),
        "gene_card": {
            "gene":         "SLC6A8",
            "protein":      "Creatine Transporter 1 (CrT1)",
            "size_aa":      635,
            "locus":        "Xq28",
            "inheritance":  "X-linked (hemizygous males severe; heterozygous females variable)",
            "family":       "SLC6 neurotransmitter transporter superfamily (Na+/Cl− co-transporter)",
            "function":     "Imports creatine from bloodstream into cells (brain, muscle, kidney)",
            "omim_gene":    "300036",
            "omim_disease": "300352",
            "prevalence":   "~300–500 cases worldwide (2026) — most common of 3 CCDS",
            "ccds":         "CCDS1 (Cerebral Creatine Deficiency Syndrome 1)",
        },
        "kpis": {
            "n_patients":            n,
            "n_males":               len(males),
            "n_females":             len(females),
            "avg_creatine_plasma":   avg("creatine_plasma_umol_l"),
            "avg_cr_ratio_males":    avg_cr_ratio_males,
            "avg_gaa":               avg("gaa_umol_l"),
            "pct_seizures":          pct(lambda p: p["seizures"]),
            "pct_drug_resistant":    pct(lambda p: p["drug_resistant_sz"]),
            "pct_idd":               pct(lambda p: p["idd"]),
            "pct_creatine_tx_fail_male": round(
                100 * sum(1 for p in males if not p["creatine_tx_response"]) / len(males), 1
            ) if males else 0,
        },
        "phenotype_distribution": [
            {"name": "Severe-Hemizygous-Male",  "pct": 60},
            {"name": "Moderate-Carrier-Female", "pct": 30},
            {"name": "Mild-Attenuated",         "pct": 10},
        ],
        "pathway_diagram": {
            "step1": "AGAT (kidney): Arg + Gly → GAA   ✅ INTACT",
            "step2": "GAMT (liver):  SAM + GAA → Creatine   ✅ INTACT",
            "step3": "SLC6A8 (ubiquitous): Creatine → [CELLS]   ✗ BLOCKED — transport failure",
            "consequence": "Creatine accumulates in plasma & urine; brain/muscle remain depleted",
            "h_mrs":       "Brain 3.0 ppm Creatine peak ABSENT — shared with GAMT & AGAT",
        },
        "biomarker_signature": {
            "urine_creatine_creatinine_ratio": {
                "value":         "CRITICALLY ELEVATED >1.0 (males)",
                "normal":        "<0.15 adult males",
                "significance":  "PRIMARY DIAGNOSTIC TEST — creatine spills into urine",
                "direction":     "↑↑↑ PATHOGNOMONIC",
            },
            "plasma_creatine": {
                "value":        "ELEVATED or NORMAL-HIGH",
                "normal":       "20–80 µmol/L",
                "significance": "Builds up in blood (transport block); OPPOSITE of GAMT/AGAT where it is absent",
                "direction":    "↑ (contrast: GAMT/AGAT ↓↓)",
            },
            "brain_h_mrs_creatine": {
                "value":       "ABSENT (3.0 ppm peak absent)",
                "significance":"All 3 CCDS share this — shared pathognomonic finding",
                "direction":   "↓↓↓ ABSENT",
            },
            "gaa": {
                "value":       "NORMAL (0.5–3.0 µmol/L)",
                "significance":"AGAT intact — KEY NEGATIVE vs GAMT (↑↑↑) and AGAT (↓↓)",
                "direction":   "→ NORMAL",
            },
            "methionine":   {"value": "NORMAL", "direction": "→ KEY NEGATIVE"},
            "thcy":         {"value": "NORMAL <15 µmol/L", "direction": "→ KEY NEGATIVE"},
            "sam":          {"value": "NORMAL", "direction": "→ KEY NEGATIVE"},
            "mma":          {"value": "NORMAL", "direction": "→ KEY NEGATIVE"},
        },
        "key_clinical_distinctions": [
            "CREATINE TX FAILS IN MALES — standard creatine monohydrate largely ineffective (transporter absent)",
            "Urine creatine/creatinine ratio ELEVATED (vs GAMT/AGAT where urine creatinine is LOW)",
            "Plasma creatine ELEVATED (vs GAMT/AGAT where plasma creatine is absent/very low)",
            "GAA NORMAL — biosynthesis intact; no GAA neurotoxicity unlike GAMT",
            "X-linked — males severe (hemizygous); females variably affected (mosaic Lyon)",
            "Brain H-MRS creatine absent — shared with GAMT & AGAT (all 3 CCDS)",
            "Most common of 3 CCDS (~300–500 cases vs ~200–250 GAMT, ~150 AGAT)",
        ],
        "ccds_triad_comparison": [
            {
                "disease":         "CCDS1 — SLC6A8 (THIS)",
                "step":            "Step 3 — transport into cells",
                "plasma_creatine": "↑ ELEVATED",
                "urine_creatine":  "↑↑↑ CRITICALLY HIGH",
                "gaa":             "→ NORMAL",
                "tx_creatine":     "FAILS in males (transporter absent)",
                "inheritance":     "X-linked",
            },
            {
                "disease":         "CCDS2 — GAMT",
                "step":            "Step 2 — GAA → creatine (final synthesis)",
                "plasma_creatine": "↓↓↓ ABSENT",
                "urine_creatine":  "↓ LOW",
                "gaa":             "↑↑↑ ELEVATED (50–300 µmol/L) — neurotoxic",
                "tx_creatine":     "EFFECTIVE + ornithine needed (SLC6A8 intact)",
                "inheritance":     "AR",
            },
            {
                "disease":         "CCDS3 — AGAT",
                "step":            "Step 1 — Arg+Gly → GAA (first synthesis)",
                "plasma_creatine": "↓↓↓ ABSENT",
                "urine_creatine":  "↓ LOW",
                "gaa":             "↓↓ VERY LOW / ABSENT (<0.5 µmol/L) — opposite of GAMT",
                "tx_creatine":     "EFFECTIVE alone (SLC6A8 intact, no GAA toxicity)",
                "inheritance":     "AR",
            },
        ],
        "nbs_status": {
            "primary_marker":   "Urine creatine/creatinine ratio (requires urine NBS; not standard blood spot)",
            "detection_rate":   "40–50% via targeted metabolic screening; MISSED by standard NBS",
            "note":             "Standard NBS blood-spot amino acid panel will miss SLC6A8 (creatine not routinely measured)",
            "second_tier":      "SLC6A8 gene sequencing if clinical suspicion (X-linked IDD + elevated urine Cr/CrCr)",
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(100 * sum(1 for p in _PATIENTS if pred(p)) / n, 1)

    males = [p for p in _PATIENTS if p["sex"] == "M"]

    seizure_type_counts: dict[str, int] = {}
    for p in _PATIENTS:
        if p.get("seizure_type"):
            k = p["seizure_type"].split(" (")[0]
            seizure_type_counts[k] = seizure_type_counts.get(k, 0) + 1

    variant_counts: dict[str, int] = {}
    for p in _PATIENTS:
        v = p.get("variant", "?")
        variant_counts[v] = variant_counts.get(v, 0) + 1

    return {
        "clinical_features": {
            "seizures_pct":            pct(lambda p: p["seizures"]),
            "drug_resistant_sz_pct":   pct(lambda p: p["drug_resistant_sz"]),
            "idd_pct":                 pct(lambda p: p["idd"]),
            "speech_absent_pct":       pct(lambda p: p["speech_absent"]),
            "autism_like_pct":         pct(lambda p: p["autism_like"]),
            "behavioral_pct":          pct(lambda p: p["behavioral_hyperactivity"]),
            "nbs_detected_pct":        pct(lambda p: p["nbs_detected"]),
            "creatine_tx_fail_male_pct": round(
                100 * sum(1 for p in males if not p["creatine_tx_response"]) / len(males), 1
            ) if males else 0,
            "gaa_tx_tried_pct":        pct(lambda p: p["gaa_tx_tried"]),
        },
        "kpi_pcts": {
            "seizures":         pct(lambda p: p["seizures"]),
            "drug_resistant":   pct(lambda p: p["drug_resistant_sz"]),
            "idd":              pct(lambda p: p["idd"]),
            "speech_absent":    pct(lambda p: p["speech_absent"]),
            "autism_like":      pct(lambda p: p["autism_like"]),
            "behavioral":       pct(lambda p: p["behavioral_hyperactivity"]),
        },
        "seizure_types": [
            {"type": k, "n": v} for k, v in sorted(seizure_type_counts.items(), key=lambda x: -x[1])
        ],
        "metabolic_triggers": [
            {
                "trigger":     "Acute illness / febrile state (catabolic stress)",
                "pct":         88,
                "mechanism":   "Increased energy demand in creatine-depleted neurons — acute decompensation",
            },
            {
                "trigger":     "Missed therapy / dietary variation",
                "pct":         72,
                "mechanism":   "No tx reservoir in brain (transporter absent); no buffer against depletion",
            },
            {
                "trigger":     "Sleep deprivation / physiological stress",
                "pct":         60,
                "mechanism":   "Creatine-phosphate energy buffer unavailable in neurons",
            },
            {
                "trigger":     "Metformin / biguanide use",
                "pct":         45,
                "mechanism":   "MODERATE RISK: further impairs SLC6A8-dependent creatine uptake",
            },
            {
                "trigger":     "Oral creatine loading (paradoxical)",
                "pct":         30,
                "mechanism":   "Raises plasma creatine further without brain benefit; osmotic / renal load risk",
            },
            {
                "trigger":     "Growth spurts (increased demand)",
                "pct":         25,
                "mechanism":   "Muscle/brain energy demand rises; creatine-phosphate buffer unavailable",
            },
        ],
        "treatments": [
            {
                "treatment":   "Guanidinoacetate (GAA) supplementation",
                "level":       "Level B (emerging)",
                "mechanism":   "GAA enters via SLC6A6/BGT-1 (taurine/GABA transporter) — bypasses absent SLC6A8; phosphorylated to phospho-GAA which provides partial creatine-phosphate equivalent",
                "note":        "Experimental; limited cases; monitor for GAA neurotoxicity (avoid excess — GAMT lesson)",
            },
            {
                "treatment":   "Creatine monohydrate",
                "level":       "NOT RECOMMENDED for hemizygous males",
                "mechanism":   "Transporter absent — creatine cannot enter brain; raises plasma creatine further without benefit; may worsen renal creatine load",
                "note":        "May have PARTIAL benefit in carrier females (mosaic SLC6A8-normal cells)",
            },
            {
                "treatment":   "Cyclocreatine (synthetic creatine analog)",
                "level":       "Experimental / investigational",
                "mechanism":   "Lipid-soluble creatine analog; enters cells via non-SLC6A8 mechanisms; provides phosphorylatable energy substrate",
                "note":        "Mouse models: positive; human data very limited; not yet approved",
            },
            {
                "treatment":   "Levetiracetam (LEV)",
                "level":       "Level A — First-line AED",
                "mechanism":   "Broad-spectrum; SV2A modulation; well-tolerated; minimal metabolic interactions",
                "note":        "First-choice for seizure management in all CCDS1 phenotypes",
            },
            {
                "treatment":   "Lamotrigine",
                "level":       "Level B",
                "mechanism":   "Sodium-channel; effective for focal + generalized seizures",
                "note":        "Second-line if LEV inadequate or in females with focal features",
            },
            {
                "treatment":   "L-Arginine + Glycine supplementation",
                "level":       "Level C (unclear benefit)",
                "mechanism":   "Upregulates AGAT/GAMT synthesis of creatine; but transport failure limits delivery",
                "note":        "Theoretically futile in hemizygous males; may provide small benefit in females",
            },
        ],
        "drug_risks": [
            {
                "drug":     "Metformin",
                "risk":     "MODERATE RISK",
                "reason":   "Inhibits SLC6A8-dependent creatine transport; directly worsens disease mechanism",
                "action":   "Avoid; use alternative glucose-lowering agents if needed",
            },
            {
                "drug":     "Creatine monohydrate (in males)",
                "risk":     "NOT RECOMMENDED",
                "reason":   "No brain entry (transporter absent); raises plasma creatine; osmotic/renal load",
                "action":   "Do not prescribe standard creatine in hemizygous males",
            },
            {
                "drug":     "Vigabatrin",
                "risk":     "LOW-MODERATE RISK",
                "reason":   "GABA-T inhibitor; generally acceptable but monitor visual fields",
                "action":   "Can use for infantile spasms with appropriate monitoring",
            },
            {
                "drug":     "VPA (valproate)",
                "risk":     "MODERATE RISK",
                "reason":   "Carnitine depletion + hepatotoxicity; competes with creatine energy metabolsim",
                "action":   "Use with caution; monitor liver, carnitine, ammonia",
            },
            {
                "drug":     "Guanidinoacetate (GAA) excess",
                "risk":     "MODERATE RISK (if overdosed)",
                "reason":   "High-dose GAA can be neurotoxic (GABA-A inhibition, NMDA activation — same as GAMT excess GAA)",
                "action":   "Titrate carefully; monitor plasma GAA levels during supplementation",
            },
        ],
        "variant_distribution": [
            {"variant": v, "n": c} for v, c in sorted(variant_counts.items(), key=lambda x: -x[1])
        ],
        "patients": [
            {
                "id":                   p["id"],
                "phenotype":            p["phenotype"],
                "sex":                  p["sex"],
                "variant":              p["variant"],
                "age_onset_months":     p["age_onset_months"],
                "creatine_plasma":      p["creatine_plasma_umol_l"],
                "urine_cr_ratio":       p["urine_cr_creatinine_ratio"],
                "gaa":                  p["gaa_umol_l"],
                "methionine":           p["methionine_umol_l"],
                "homocysteine":         p["homocysteine_umol_l"],
                "ck":                   p["ck_u_l"],
                "seizures":             p["seizures"],
                "seizure_type":         p["seizure_type"],
                "drug_resistant":       p["drug_resistant_sz"],
                "idd":                  p["idd"],
                "speech_absent":        p["speech_absent"],
                "autism_like":          p["autism_like"],
                "behavioral":           p["behavioral_hyperactivity"],
                "nbs_detected":         p["nbs_detected"],
                "creatine_response":    p["creatine_tx_response"],
            }
            for p in _PATIENTS
        ],
    }


def get_definitions() -> dict:
    return {
        "title": "SLC6A8 / CCDS1 — Definitions & Differential Diagnosis",
        "gene_definition": {
            "gene":       "SLC6A8",
            "full_name":  "Solute Carrier Family 6 Member 8 (Creatine Transporter 1, CrT1)",
            "size_aa":    635,
            "locus":      "Xq28",
            "family":     "SLC6 — Na+/Cl− neurotransmitter transporter superfamily (12 TM helices)",
            "function":   (
                "Transports creatine across plasma membranes using the electrochemical gradient of "
                "Na+ and Cl− (Na+:Cl−:creatine = 2:1:1 stoichiometry). Required for creatine uptake "
                "in virtually all tissues including brain (neurons + astrocytes), muscle, and kidney."
            ),
            "omim":       "Gene 300036 / Disease 300352",
        },
        "key_concepts": [
            {
                "concept":    "Why creatine transport fails (not synthesis)",
                "detail": (
                    "AGAT and GAMT (steps 1 and 2) are intact in SLC6A8 deficiency. Creatine is made "
                    "normally in kidney → liver. The failure is at the IMPORT STEP: without SLC6A8, "
                    "creatine cannot enter brain cells, muscle cells, or other SLC6A8-dependent tissues. "
                    "Creatine accumulates in plasma and spills massively into urine instead."
                ),
            },
            {
                "concept":    "Why urine creatine/creatinine ratio is the best diagnostic test",
                "detail": (
                    "Urine Cr/CrCr ratio captures the bidirectional abnormality: creatine is ELEVATED "
                    "(spillage into urine from blocked cellular import), while creatinine is LOW-NORMAL "
                    "(creatinine is the non-enzymatic catabolite of phosphocreatine, which requires "
                    "intracellular creatine — absent in muscle/brain → less creatinine produced). "
                    "Normal ratio <0.15 in adult males; CCDS1 typically >1.0–5.0."
                ),
            },
            {
                "concept":    "Why standard creatine monohydrate fails in hemizygous males",
                "detail": (
                    "The very transporter required to import creatine into brain/muscle is absent. "
                    "Oral creatine raises plasma creatine further but provides no brain benefit — "
                    "plasma creatine is already elevated and the bottleneck is transport. This is the "
                    "KEY DISTINCTION from GAMT and AGAT where creatine treatment IS effective "
                    "(SLC6A8 is intact in those disorders). Carrier females may show partial response "
                    "via X-inactivation mosaicism (some cells with normal SLC6A8)."
                ),
            },
            {
                "concept":    "X-linked — male vs female severity",
                "detail": (
                    "Hemizygous males (single X chromosome with mutant SLC6A8): severe phenotype — "
                    "profound IDD, 80–90% seizures, absent speech. "
                    "Heterozygous females: variable — 50–70% have some symptoms (learning difficulties, "
                    "mild IDD, behavioural problems). Severity depends on X-inactivation ratio (Lyon "
                    "mosaicism): females with skewed inactivation toward the mutant allele may be more "
                    "severely affected, approaching male severity."
                ),
            },
            {
                "concept":    "Brain H-MRS — shared pathognomonic finding of ALL 3 CCDS",
                "detail": (
                    "The 1H-MRS creatine peak at 3.0 ppm (methylene protons of creatine + "
                    "phosphocreatine) is ABSENT in CCDS1, CCDS2 (GAMT), and CCDS3 (AGAT). "
                    "This is the shared pathognomonic neuroimaging/spectroscopy finding. "
                    "In SLC6A8, the absent peak reflects failure of creatine IMPORT into brain "
                    "(not biosynthesis failure as in GAMT/AGAT). It is the most sensitive "
                    "single diagnostic test when clinical suspicion exists."
                ),
            },
            {
                "concept":    "GAA supplementation strategy",
                "detail": (
                    "Guanidinoacetate (GAA) enters cells via SLC6A6 (BGT-1) and possibly SLC6A14, "
                    "bypassing the absent SLC6A8. Once inside neurons, GAA can be phosphorylated by "
                    "creatine kinase to phospho-GAA, which provides a partial energy reservoir. "
                    "Emerging evidence suggests seizure reduction and modest developmental benefit "
                    "in some CCDS1 patients. Must be carefully dosed: excess GAA is neurotoxic "
                    "(same mechanism as GAMT where GAA accumulates to 50–300 µmol/L and causes "
                    "GABA-A inhibition + NMDA activation)."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease":          "CCDS2 — GAMT Deficiency",
                "key_distinction":  "GAA MASSIVELY ELEVATED (50–300 µmol/L) vs NORMAL in SLC6A8",
                "plasma_creatine":  "ABSENT (vs ELEVATED in SLC6A8)",
                "urine_creatine":   "LOW (vs MARKEDLY HIGH in SLC6A8)",
                "creatine_tx":      "EFFECTIVE + ornithine (SLC6A8 intact) vs INEFFECTIVE (transport absent)",
                "gaa":              "↑↑↑ neurotoxic vs → NORMAL",
                "inheritance":      "AR vs X-linked",
            },
            {
                "disease":          "CCDS3 — AGAT Deficiency",
                "key_distinction":  "GAA VERY LOW / ABSENT (<0.5 µmol/L) vs NORMAL in SLC6A8",
                "plasma_creatine":  "ABSENT vs ELEVATED",
                "urine_creatine":   "LOW vs MARKEDLY HIGH",
                "creatine_tx":      "EFFECTIVE alone (no GAA toxicity) vs INEFFECTIVE",
                "gaa":              "↓↓ nearly absent vs → NORMAL",
                "inheritance":      "AR vs X-linked",
            },
            {
                "disease":          "Non-Ketotic Hyperglycinaemia (NKH / GCS deficiency)",
                "key_distinction":  "Glycine MARKEDLY ELEVATED (CSF/plasma); creatine normal; H-MRS creatine PRESENT",
                "urine_creatine":   "Normal",
                "creatine_tx":      "Not applicable",
            },
            {
                "disease":          "GLUT1 Deficiency (SLC2A1)",
                "key_distinction":  "Glucose transport failure (not creatine); low CSF/plasma glucose ratio; different H-MRS",
                "urine_creatine":   "Normal",
            },
            {
                "disease":          "X-linked IDD (fragile X, ARX, PQBP1, etc.)",
                "key_distinction":  "Urine creatine/creatinine ratio NORMAL; H-MRS creatine PRESENT",
                "note":             "SLC6A8 should be screened in all males with unexplained IDD",
            },
        ],
        "treatment_summary": {
            "do": [
                "Guanidinoacetate (GAA) supplementation — Level B, emerging therapy; bypasses absent transporter",
                "LEV (levetiracetam) — Level A first-line AED for seizures",
                "Cyclocreatine — investigational; monitor for efficacy in individual patients",
                "Carrier female creatine monohydrate — may have partial benefit (mosaic normal cells)",
                "Urine Cr/CrCr ratio monitoring to assess biochemical response to any therapy",
            ],
            "avoid": [
                "Creatine monohydrate in hemizygous males — transport absent, no brain benefit",
                "Metformin — directly impairs residual creatine transport",
                "Excess GAA supplementation — risk of GAA neurotoxicity (GAMT lesson)",
                "Creatine ethyl ester — still requires membrane transport mechanism",
            ],
        },
        "variant_summary": [
            {
                "variant":    "p.Gly132Val",
                "domain":     "Catalytic / substrate-binding domain",
                "severity":   "Severe",
                "frequency":  "~20% — most common CCDS1 variant",
            },
            {
                "variant":    "p.Asp126Asn",
                "domain":     "Substrate-binding pocket",
                "severity":   "Severe",
                "frequency":  "~18%",
            },
            {
                "variant":    "p.Arg365Trp",
                "domain":     "TM helix 6/7 junction",
                "severity":   "Severe",
                "frequency":  "~16%",
            },
            {
                "variant":    "p.Tyr389Cys",
                "domain":     "Na+-binding coordination site",
                "severity":   "Moderate-severe",
                "frequency":  "~14%",
            },
            {
                "variant":    "p.Pro544Leu",
                "domain":     "C-terminal cytoplasmic tail",
                "severity":   "Moderate",
                "frequency":  "~12%",
            },
            {
                "variant":    "c.IVS5+1G>A",
                "domain":     "Splice-site (intron 5)",
                "severity":   "Severe (null allele)",
                "frequency":  "~10%",
            },
            {
                "variant":    "p.Arg513Gly",
                "domain":     "Intracellular loop 4",
                "severity":   "Moderate",
                "frequency":  "~8%",
            },
            {
                "variant":    "Large Xq28 del",
                "domain":     "Chromosomal deletion (may include adjacent genes)",
                "severity":   "Severe/complex",
                "frequency":  "~7%",
            },
            {
                "variant":    "p.Thr411Ile",
                "domain":     "TM helix 8",
                "severity":   "Mild (partial transport retained)",
                "frequency":  "~5%",
            },
            {
                "variant":    "p.Ala478Thr",
                "domain":     "TM helix 9",
                "severity":   "Mild",
                "frequency":  "~5%",
            },
        ],
    }
