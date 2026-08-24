#!/usr/bin/env python3
"""MAT1A (Methionine Adenosyltransferase I/III Deficiency) Epilepsy Dashboard.

MAT1A encodes the alpha-1 subunit of hepatic Methionine Adenosyltransferase (MAT),
a Mg2+/K+-dependent enzyme that catalyzes the FIRST and ONLY step of SAM biosynthesis:

  L-Methionine + ATP → S-Adenosylmethionine (SAM) + PPi + Pi

This is the OBLIGATORY reaction to produce SAM — the universal methyl donor for
ALL biological methylation reactions (DNA, RNA, histones, neurotransmitters,
creatine, phospholipids, myelin). Without MAT1A, liver CANNOT produce SAM from methionine.

MAT ISOFORM SYSTEM — MAT1A IS LIVER-SPECIFIC:
  MAT I   — homotetramer (alpha1/alpha1/alpha1/alpha1); encoded by MAT1A; hepatic; Km ~1 mM for methionine
  MAT III — homodimer (alpha1/alpha1); encoded by MAT1A; hepatic; Km ~8 mM for methionine (high-methionine sensor)
  MAT II  — heterotetramer (alpha2/alpha2/beta/beta); encoded by MAT2A + MAT2B; UBIQUITOUS (brain, all tissues)

MAT II (MAT2A) is expressed in ALL non-hepatic tissues including brain.
  → Brain SAM synthesis is PRESERVED in MAT1A deficiency (MAT2A compensates in extrahepatic tissues)
  → Neurological features arise from SECONDARY effects: hypermethioninemia toxicity +
    reduced hepatic SAM → impaired systemic methylation (myelination, phospholipids)

INHERITANCE:
  Autosomal Recessive (AR): biallelic LOF → MAT I deficiency (severe; very high methionine)
    Both alpha1 alleles nonfunctional → no MAT I or MAT III activity → only MAT II in liver
    → Severe hypermethioninemia, demyelination, liver disease, seizures
  Autosomal Dominant (AD): heterozygous dominant-negative (p.Arg264His = most common)
    → MAT III homodimer disrupted (alpha1-wt cannot dimerize with alpha1-His264 properly)
    → MAT I tetramer partially functional
    → Mild-moderate hypermethioninemia; OFTEN CLINICALLY BENIGN (incidental on NBS)

MAT1A LOF RESULT — SAM CANNOT BE MADE (INVERSE OF AHCY):

  L-Methionine → ← MAT1A BLOCKED (LOF) → SAM CANNOT BE SYNTHESIZED
              ↑↑↑ ACCUMULATES               SAM VERY LOW (product missing)
                                            ALL methyltransferases starved of methyl donor
                                            Myelin basic protein methylation ↓ → demyelination
                                            Phosphatidylcholine synthesis ↓ → liver disease
                                            Creatine synthesis ↓ → secondary creatine deficiency
                                            SAH DECREASES (less SAM → less methyltransferase activity → less SAH)
                                            tHcy LOW/NORMAL (less SAH via AHCY → less Hcy produced)

CRITICAL BIOMARKER FINGERPRINT (MAT1A vs AHCY vs CBS — all hypermethioninemias):

  Feature              MAT1A              AHCY               CBS                GNMT
  Methionine           ↑↑↑ EXTREME        ↑↑ HIGH (200-600)  ↑↑ HIGH (60-500)   ↑ HIGH
                       (200-2000+)                                               (>100)
  SAM                  ↓↓ VERY LOW        ↑↑ ELEVATED        NORMAL             ↑↑ ELEVATED
                       (<50 µmol/L)       (synthesis intact)  (normal)           (cannot be used)
  SAH                  LOW/NORMAL         ↑↑↑ PATHOGNOMONIC  NORMAL             LOW/NORMAL
                       (less SAM → less)   HIGH               NORMAL
  SAM/SAH ratio        NORMAL or HIGH     ↓↓↓ SEVERELY LOW   NORMAL             HIGH
                       (SAM low, SAH low)  (<0.5)             (>4)
  tHcy                 NORMAL (<30)       40-150 (MODERATE)  100-500 (HIGHEST)  NORMAL
  MMA                  NORMAL             NORMAL             NORMAL             NORMAL
  MeCbl/AdoCbl         NORMAL             NORMAL             NORMAL             NORMAL
  Myopathy             ABSENT             ↑↑ 85-90% HALLMARK ABSENT             ABSENT
  Cardiomyopathy       ABSENT             60-70%             ABSENT             ABSENT
  Liver disease        40-50% (SAM ↓)     70-75% (PEMT ↓)    ABSENT             PROMINENT
  Ectopia lentis       ABSENT             ABSENT             90% PATHOGN.       ABSENT
  White matter disease 40-50% (SAM ↓)    PRESENT (methylation↓) 10-15%         ABSENT
  Breath odor          PRESENT (DMSP)     ABSENT             ABSENT             ABSENT
  NBS detection        ~90% (Met ↑↑↑)    ~70% (Met ↑↑)      ~60% (Met ↑)       ~50%
  Treatment SAM        Level A (corrects)  ABSOLUTE CI        Level B            ABSOLUTE CI
  Treatment betaine    ABSOLUTE CI         CAUTION            Level A            ABSOLUTE CI

KEY DISTINCTION FROM AHCY (BOTH CAUSE HYPERMETHIONINEMIA):
  MAT1A: SAM LOW (cannot make SAM from methionine) → SAH LOW → tHcy NORMAL → NO myopathy
  AHCY:  SAM HIGH (makes SAM but cannot clear SAH) → SAH HIGH → tHcy MODERATE → SEVERE MYOPATHY
  SAM supplementation: CORRECT treatment in MAT1A; ABSOLUTELY CONTRAINDICATED in AHCY
  Betaine: CONTRAINDICATED in MAT1A (BHMT raises methionine further); Level B in CBS

BREATH ODOR (PATHOGNOMONIC):
  Excess methionine → transamination → dimethylsulfide (DMS) / methane thiol / DMSP
  Patients have a characteristic sweet/garlic/cabbage breath odor detectable clinically
  One of the rare IEMs with a recognizable bedside diagnosis clue
  NOT seen in CBS, AHCY, MTHFR, cblE/cblG

OMIM:
  Gene: *610550 (MAT1A)
  Disease: #250850 (Hypermethioninemia due to MAT1A deficiency / Methionine adenosyltransferase I/III deficiency)
"""

import random
import math

_SEED = 101
_rng = random.Random(_SEED)

# ---------------------------------------------------------------------------
# Phenotypic classes
# ---------------------------------------------------------------------------
PHENOTYPE_CLASSES = [
    "Benign-Incidental (AD-p.Arg264His or mild-AR / NBS-detected / asymptomatic / no-treatment-needed)",
    "Classic-Mild-Moderate (AR-partial-LOF / white-matter-disease / liver / SAM-restriction-needed / MODAL)",
    "Severe-MAT-I-Deficiency (AR-complete-LOF / extreme-methionine / demyelination / seizures / liver-failure)",
]

PHENO_WEIGHTS = [0.50, 0.30, 0.20]  # 50% benign / 30% classic / 20% severe

VARIANTS = [
    "p.Arg264His",    # AD dominant-negative; most common; ~40% of all MAT1A alleles; BENIGN
    "p.Thr222Met",    # AR; moderate-severe; catalytic domain; ~15% of AR alleles
    "p.Ala259Val",    # AR; moderate; catalytic domain; ~13%
    "p.Arg356Trp",    # AR; severe; oligomerization domain; ~12%
    "p.Glu57Lys",     # AR; severe; ATP-binding pocket; ~10%
    "p.Gly378Ser",    # AR; moderate; C-terminal domain; ~8%
    "p.Tyr21Cys",     # AR; severe neonatal; N-terminal; ~5% (rare but catastrophic)
]

VARIANT_WEIGHTS = [0.40, 0.15, 0.13, 0.12, 0.10, 0.08, 0.02]

SEIZURE_TYPES = [
    "Focal (frontal/parietal — demyelination-related white matter disease)",
    "Generalized tonic-clonic (severe SAM deficiency / metabolic encephalopathy)",
    "Infantile spasms (severe MAT I neonatal class — extreme methioninemia)",
    "Absence (mild hypermethioninemia-related cortical hyperexcitability)",
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
    is_benign  = "Benign" in pheno
    is_severe  = "Severe" in pheno
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            # Enforce phenotype–variant consistency
            if is_benign and v not in ("p.Arg264His", "p.Gly378Ser"):
                return _rng.choice(["p.Arg264His", "p.Gly378Ser"])
            if is_severe and v in ("p.Arg264His", "p.Gly378Ser"):
                return _rng.choice(["p.Arg356Trp", "p.Glu57Lys", "p.Tyr21Cys"])
            return v
    return VARIANTS[0]


def _make_patient(idx):
    pheno       = _choose_pheno()
    is_benign   = "Benign" in pheno
    is_classic  = "Classic" in pheno
    is_severe   = "Severe" in pheno

    # Methionine: MAT1A hallmark — VERY HIGH (much higher than AHCY/CBS in severe)
    if is_severe:
        methion  = round(_rng.uniform(900, 2100),  1)
        sam_umol = round(_rng.uniform(5,   35),    1)   # SAM VERY LOW
        sah_umol = round(_rng.uniform(5,   25),    1)   # SAH LOW/NORMAL
        hcy_umol = round(_rng.uniform(12,  32),    1)   # tHcy NORMAL-LOW
        liver_enz= round(_rng.uniform(120, 500),   0)   # AST U/L
    elif is_classic:
        methion  = round(_rng.uniform(250, 900),   1)
        sam_umol = round(_rng.uniform(15,  60),    1)
        sah_umol = round(_rng.uniform(8,   30),    1)
        hcy_umol = round(_rng.uniform(8,   22),    1)
        liver_enz= round(_rng.uniform(40,  200),   0)
    else:  # benign/incidental
        methion  = round(_rng.uniform(100, 400),   1)
        sam_umol = round(_rng.uniform(30,  80),    1)   # SAM low-normal (partial activity)
        sah_umol = round(_rng.uniform(10,  35),    1)
        hcy_umol = round(_rng.uniform(6,   18),    1)   # NORMAL tHcy
        liver_enz= round(_rng.uniform(20,  60),    0)   # AST often normal

    mma_normal    = True   # ALL patients — propionate arm intact
    mecbl_normal  = True
    adocbl_normal = True

    # Clinical features by phenotype
    white_matter    = _rng.random() < (0.75 if is_severe else (0.45 if is_classic else 0.05))
    liver_disease   = _rng.random() < (0.80 if is_severe else (0.45 if is_classic else 0.10))
    idd             = _rng.random() < (0.70 if is_severe else (0.40 if is_classic else 0.05))
    seizures        = _rng.random() < (0.30 if is_severe else (0.18 if is_classic else 0.02))
    nbs_detected    = _rng.random() < (0.97 if is_severe else (0.90 if is_classic else 0.80))
    breath_odor     = _rng.random() < (0.85 if is_severe else (0.60 if is_classic else 0.30))
    psychiatric     = _rng.random() < (0.20 if is_severe else (0.15 if is_classic else 0.05))
    on_sam_suppl    = _rng.random() < (0.0  if is_benign else (0.65 if is_classic else 0.80))
    on_met_restrict = _rng.random() < (0.0  if is_benign else (0.80 if is_classic else 0.95))

    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    age_onset_mo = (
        _rng.randint(0, 6)   if is_severe  else
        _rng.randint(6, 36)  if is_classic else
        _rng.randint(0, 0)   # benign: NBS day 2-5, but we store 0 for "NBS-only"
    )

    variant = _choose_variant(pheno)

    return {
        "id":               f"MAT1A-{idx:03d}",
        "phenotype":        pheno,
        "variant":          variant,
        "age_onset_months": age_onset_mo,
        # Biomarkers
        "methionine_umol_l":   methion,
        "sam_umol_l":          sam_umol,    # VERY LOW — pathognomonic for MAT1A
        "sah_umol_l":          sah_umol,    # LOW/NORMAL
        "homocysteine_umol_l": hcy_umol,    # NORMAL in most
        "ast_u_l":             liver_enz,
        "mma_normal":          mma_normal,
        "mecbl_normal":        mecbl_normal,
        "adocbl_normal":       adocbl_normal,
        # Clinical
        "white_matter_disease":   white_matter,
        "liver_disease":          liver_disease,
        "idd":                    idd,
        "seizures":               seizures,
        "seizure_type":           seizure_type,
        "nbs_detected":           nbs_detected,
        "breath_odor":            breath_odor,
        "psychiatric":            psychiatric,
        "myopathy":               False,         # ABSENT — distinguishes from AHCY
        "cardiomyopathy":         False,         # ABSENT — distinguishes from AHCY
        "ectopia_lentis":         False,         # ABSENT — distinguishes from CBS
        "on_sam_suppl":           on_sam_suppl,
        "on_met_restriction":     on_met_restrict,
    }


_PATIENTS = [_make_patient(i + 1) for i in range(40)]


# ---------------------------------------------------------------------------
# API response functions
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    n = len(_PATIENTS)

    def avg(key):
        vals = [p[key] for p in _PATIENTS if isinstance(p.get(key), (int, float))]
        return round(sum(vals) / len(vals), 1) if vals else 0

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    pheno_dist = {}
    for p in _PATIENTS:
        ph = p["phenotype"].split(" (")[0]
        pheno_dist[ph] = pheno_dist.get(ph, 0) + 1

    return {
        "dashboard_id": "mat1a",
        "title": "MAT1A Epilepsy Dashboard",
        "subtitle": (
            "Methionine Adenosyltransferase I/III Deficiency — "
            "Hypermethioninemia with SAM Deficiency / "
            "MAT1A-395aa-Mg2+-K+-Hepatic-10q22.3-AR/AD"
        ),
        "gene": "MAT1A",
        "disease_name": (
            "Methionine Adenosyltransferase I/III Deficiency "
            "(Hypermethioninemia due to hepatic MAT deficiency)"
        ),
        "chromosome": "10q22.3",
        "inheritance": "Autosomal Recessive (MAT I — severe) / Autosomal Dominant (MAT III — p.Arg264His, benign)",
        "omim_gene": "OMIM *610550",
        "omim_disease": "OMIM #250850",
        "protein_size": (
            "395 aa; Mg2+/K+-dependent; liver-specific; "
            "forms MAT I (homotetramer, Km ~1 mM) and MAT III (homodimer, Km ~8 mM); "
            "MAT II (MAT2A/MAT2B) is the ubiquitous isoform expressed in brain (preserved in MAT1A deficiency)"
        ),
        "prevalence": "~1 in 50,000–100,000 births; most common isolated hypermethioninemia cause in NBS populations",
        "cohort_n": n,
        "phenotype_distribution": pheno_dist,
        "function": (
            "MAT1A encodes the alpha-1 subunit of hepatic methionine adenosyltransferase. "
            "It catalyzes the ONLY route to synthesize SAM in the liver: "
            "L-methionine + ATP → SAM + PPi + Pi. "
            "SAM is the universal methyl donor for ALL biological methylation reactions "
            "(DNA/RNA/histone, neurotransmitters, creatine, phospholipids, myelin). "
            "MAT1A forms two liver-specific isoforms: MAT I (homotetramer, low Km, basal SAM) "
            "and MAT III (homodimer, high Km, high-methionine sensor). "
            "Brain and all extrahepatic tissues use MAT II (MAT2A/MAT2B), which is NOT affected "
            "in MAT1A deficiency — neurological features arise from secondary mechanisms."
        ),
        "mechanism": (
            "MAT1A LOF → liver cannot convert methionine to SAM → methionine MASSIVELY ELEVATED "
            "(200–2000+ µmol/L; highest of all isolated hypermethioninemias). "
            "SAM VERY LOW in liver (<50 µmol/L, often undetectable) — pathognomonic deficit. "
            "DOWNSTREAM CONSEQUENCES OF LOW HEPATIC SAM: "
            "(1) Methyltransferases deprived of methyl donor → phospholipid synthesis impaired "
            "    (PEMT) → fatty liver / hepatopathy; "
            "(2) Myelin basic protein methylation impaired → cerebral demyelination / white matter; "
            "(3) Creatine synthesis (GAMT) partially impaired → secondary creatine deficit; "
            "(4) Less SAM → less methyltransferase activity → LESS SAH produced → "
            "    SAH is LOW or normal (OPPOSITE of AHCY where SAH is MASSIVELY elevated); "
            "(5) Less SAH → less AHCY substrate → less Hcy produced → tHcy NORMAL or only "
            "    mildly elevated (distinguishes MAT1A from CBS where tHcy is 100-500). "
            "BRAIN IS PARTIALLY PROTECTED: MAT II (MAT2A/MAT2B) is intact in brain → "
            "local SAM synthesis preserved; neurological features from hypermethioninemia toxicity "
            "and systemic SAM deficit (hepatic export reduced). "
            "BREATH ODOR: excess methionine → transamination + bacterial metabolism → "
            "dimethylsulfide / methanethiol / DMSP → characteristic sweet/garlic/cabbage breath "
            "(pathognomonic bedside finding; unique to MAT1A among HHcy disorders)."
        ),
        "key_positive_features": (
            "KEY POSITIVES (unique to MAT1A among metabolic epilepsy disorders): "
            "(1) Methionine EXTREMELY HIGH (200–2000+ µmol/L) — highest absolute methionine of all inherited HHcy. "
            "(2) SAM VERY LOW (pathognomonic — <50 µmol/L; OPPOSITE of AHCY where SAM is elevated). "
            "(3) SAH LOW or NORMAL (less SAM → less methyltransferase activity → less SAH produced). "
            "(4) tHcy NORMAL or mildly elevated (<30 µmol/L) — KEY DISTINCTION from CBS (tHcy 100-500). "
            "(5) Characteristic BREATH ODOR (dimethylsulfide / garlic/cabbage) — pathognomonic. "
            "(6) NBS detected ~90% via extreme methionine elevation. "
            "KEY NEGATIVES (absent in MAT1A): "
            "(1) Myopathy ABSENT (unlike AHCY 85-90%). "
            "(2) Cardiomyopathy ABSENT (unlike AHCY 60-70%). "
            "(3) Ectopia lentis ABSENT (unlike CBS 90%). "
            "(4) MMA NORMAL (propionate arm intact). "
            "(5) MeCbl/AdoCbl NORMAL (cobalamin system intact). "
            "(6) Megaloblastic anemia ABSENT (no methylfolate trap). "
            "(7) SAH NOT elevated (unlike AHCY — key differential)."
        ),
        "nbs_primary": (
            "Amino acid panel — EXTREME methionine elevation (200-2000+ µmol/L; normal <60). "
            "~90% of severe/classic MAT1A cases detected on NBS. "
            "Highest methionine of all inherited HHcy disorders — cannot be missed. "
            "Benign AD forms (p.Arg264His) detected at ~80% — methionine 100-400 µmol/L. "
            "Distinguishable from CBS/AHCY by HIGHER methionine + NORMAL tHcy + LOW SAM."
        ),
        "nbs_secondary": (
            "SAM/SAH by HPLC: SAM VERY LOW (pathognomonic) + SAH LOW/NORMAL (opposite of AHCY). "
            "Plasma total homocysteine: NORMAL or mildly elevated (<30 µmol/L). "
            "Urine organic acids: MMA NORMAL. "
            "Serum B12/folate: NORMAL (cobalamin/folate system intact). "
            "Liver function tests: elevated transaminases (40-50% in symptomatic). "
            "Brain MRI: periventricular/subcortical white matter lesions in demyelinating forms. "
            "Breath odor assessment: characteristic dimethylsulfide/garlic odor. "
            "MAT1A enzyme activity in liver biopsy / erythrocytes: reduced or absent. "
            "Genetic confirmation by MAT1A sequencing."
        ),
        "kpis": {
            "avg_methionine_umol_l":   avg("methionine_umol_l"),
            "avg_sam_umol_l":          avg("sam_umol_l"),
            "avg_sah_umol_l":          avg("sah_umol_l"),
            "avg_homocysteine_umol_l": avg("homocysteine_umol_l"),
            "avg_ast_u_l":             avg("ast_u_l"),
            "pct_white_matter":        pct(lambda p: p["white_matter_disease"]),
            "pct_liver_disease":       pct(lambda p: p["liver_disease"]),
            "pct_idd":                 pct(lambda p: p["idd"]),
            "pct_seizures":            pct(lambda p: p["seizures"]),
            "pct_nbs_detected":        pct(lambda p: p["nbs_detected"]),
            "pct_breath_odor":         pct(lambda p: p["breath_odor"]),
            "pct_myopathy":            0,
            "pct_cardiomyopathy":      0,
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Focal (frontal/parietal — demyelination-related white matter disease)", "pct": 48},
        {"type": "Generalized tonic-clonic (severe SAM deficiency / metabolic encephalopathy)", "pct": 33},
        {"type": "Infantile spasms (severe MAT I neonatal — extreme methioninemia)", "pct": 13},
        {"type": "Absence (mild hypermethioninemia-related cortical hyperexcitability)", "pct": 6},
    ]

    metabolic_triggers = [
        {
            "trigger": "High-methionine diet (meat, dairy, eggs, legumes without restriction)",
            "pct": 95,
            "mechanism": (
                "Dietary methionine absorbed → cannot be converted to SAM (MAT1A absent) → "
                "methionine MASSIVELY elevates further. Even maintenance-level protein intake "
                "can cause dangerous methionine spikes. Methionine restriction is cornerstone "
                "therapy in symptomatic MAT1A patients."
            ),
        },
        {
            "trigger": "Betaine (TMG) supplementation — ABSOLUTELY CONTRAINDICATED",
            "pct": 85,
            "mechanism": (
                "BHMT: Hcy + betaine → methionine. In MAT1A where methionine is already "
                "massively elevated, betaine drives FURTHER methionine accumulation → catastrophic "
                "worsening. Betaine is ABSOLUTELY CONTRAINDICATED in MAT1A (opposite of CBS where "
                "betaine is Level A). This is the critical contraindication trap for clinicians."
            ),
        },
        {
            "trigger": "Protein catabolism / fasting / catabolic illness",
            "pct": 65,
            "mechanism": (
                "Protein breakdown releases methionine → feeds the blocked cycle → methionine "
                "elevation worsens. Emergency protocol: IV dextrose + lipids to suppress catabolism; "
                "methionine-free amino acid formula. High-protein TPN without Met restriction "
                "is a common iatrogenic error."
            ),
        },
        {
            "trigger": "Valproate (VPA) — HIGH RISK with pre-existing liver disease",
            "pct": 45,
            "mechanism": (
                "MAT1A patients with severe AR forms have hepatomegaly + liver disease. "
                "VPA is hepatotoxic and depletes carnitine. Combined with pre-existing "
                "hepatopathy in severe MAT1A, VPA can precipitate liver failure. "
                "LEV is preferred first-line AED."
            ),
        },
        {
            "trigger": "SAM inhibitors / adenosine analogs",
            "pct": 30,
            "mechanism": (
                "While SAM supplementation is the TREATMENT for MAT1A, drugs that inhibit SAM "
                "synthesis (adenosine analogs, SAICAR pathway inhibitors) can further reduce the "
                "already critically low SAM. Avoid drugs that compete with SAM biosynthesis."
            ),
        },
    ]

    treatments = [
        {
            "treatment": "SAM (S-Adenosylmethionine) supplementation — oral SAMe",
            "level": "Level A (symptomatic MAT1A — corrects the product deficiency directly)",
            "mechanism": (
                "MAT1A cannot make SAM from methionine → exogenous SAM bypasses the blocked step. "
                "Oral SAMe (S-adenosylmethionine, 400-1600 mg/day) is absorbed and crosses into "
                "tissues → partially restores methylation capacity for myelin, phospholipids, "
                "neurotransmitters, creatine. Brain: SAMe penetrates BBB; MAT2A (brain) can "
                "synthesize its own SAM but hepatic export of SAM is reduced. "
                "KEY DISTINCTION from AHCY: SAM is the TREATMENT in MAT1A; "
                "SAM is ABSOLUTELY CONTRAINDICATED in AHCY. This is the critical clinical trap."
            ),
        },
        {
            "treatment": "Methionine restriction (low-Met diet + methionine-free amino acid formula)",
            "level": "Level A (symptomatic MAT1A — reduces methionine overload and toxicity)",
            "mechanism": (
                "Reducing dietary methionine decreases the methionine overload → reduces "
                "dimethylsulfide (breath odor) and neurotoxicity from hypermethioninemia. "
                "Target plasma methionine <800 µmol/L in severe; <400 in classic. "
                "Methionine-free amino acid formula provides essential amino acids. "
                "NOTE: Benign AD forms (p.Arg264His) often need NO dietary restriction — "
                "asymptomatic patients may not benefit from treatment."
            ),
        },
        {
            "treatment": "Choline supplementation — 0.5-1 g/day",
            "level": "Level B (supports phospholipid synthesis via PEMT pathway)",
            "mechanism": (
                "SAM deficiency impairs PEMT (phosphatidylethanolamine N-methyltransferase), "
                "the enzyme that methylates PE → PC (phosphatidylcholine). Choline "
                "supplementation provides an alternative phospholipid precursor → "
                "partially corrects hepatic PC deficiency and fatty liver."
            ),
        },
        {
            "treatment": "Creatine monohydrate — 0.1-0.3 g/kg/day",
            "level": "Level B (secondary creatine deficiency from GAMT-SAM deprivation)",
            "mechanism": (
                "GAMT (guanidinoacetate methyltransferase) uses SAM to synthesize creatine. "
                "In MAT1A deficiency, hepatic SAM is very low → creatine synthesis partially "
                "impaired → secondary creatine deficiency contributes to fatigue and possibly "
                "myalgia (less severe than AHCY but present in severe MAT I)."
            ),
        },
        {
            "treatment": "Levetiracetam (LEV) — first-line AED for seizures",
            "level": "Level B (first-line — safe with pre-existing liver disease)",
            "mechanism": (
                "LEV has no impact on methionine metabolism or SAM/SAH pathway. "
                "No hepatotoxicity at standard doses (dose-adjust for renal impairment). "
                "No carnitine depletion. SV2A mechanism unrelated to metabolic pathways. "
                "Preferred over VPA (hepatotoxic) in MAT1A with liver disease."
            ),
        },
        {
            "treatment": "Liver function monitoring + hepatoprotective measures",
            "level": "Level A (liver — hepatopathy in symptomatic MAT1A)",
            "mechanism": (
                "SAM deficiency impairs PEMT → phosphatidylcholine deficiency → fatty liver "
                "→ non-alcoholic steatohepatitis (NASH)-like hepatopathy. "
                "Liver function tests every 3-6 months. Avoid hepatotoxic drugs (VPA, "
                "acetaminophen excess, alcohol). SAMe supplementation also provides "
                "hepatoprotective effects. Liver transplant considered in end-stage cirrhosis "
                "in severe AR forms."
            ),
        },
        {
            "treatment": "Brain MRI surveillance — white matter monitoring",
            "level": "Level A (demyelination — white matter disease in classic/severe MAT1A)",
            "mechanism": (
                "SAM deficiency impairs MBP (myelin basic protein) methylation → "
                "progressive cerebral demyelination / leukoencephalopathy. "
                "MRI every 12-24 months in symptomatic patients. SAMe supplementation "
                "± methionine restriction may stabilize or improve white matter lesions. "
                "Demyelination is the major neurological substrate for cognitive impairment "
                "and seizures in MAT1A."
            ),
        },
    ]

    drug_risks = [
        {
            "agent": "Betaine (TMG) — ALL FORMS including cosmetics/supplements",
            "risk": "ABSOLUTE CONTRAINDICATION — catastrophically worsens hypermethioninemia",
            "mechanism": (
                "BHMT: betaine + Hcy → methionine + dimethylglycine. "
                "In MAT1A, methionine is already 200-2000+ µmol/L. Betaine drives "
                "FURTHER methionine accumulation → neurotoxicity, demyelination worsening. "
                "Betaine is the primary treatment for CBS and HHcy of other causes; "
                "in MAT1A it is the OPPOSITE — ABSOLUTELY CONTRAINDICATED. "
                "Many well-meaning physicians prescribe betaine for 'elevated homocysteine' "
                "but MAT1A patients have NORMAL Hcy — betaine would only worsen methionine."
            ),
        },
        {
            "agent": "High-protein TPN/enteral feeds without methionine restriction",
            "risk": "HIGH RISK — methionine loading in hospitalized patients",
            "mechanism": (
                "Standard TPN solutions contain methionine. In MAT1A patients who cannot "
                "convert methionine → SAM, any methionine load causes dangerous accumulation. "
                "Always use methionine-restricted amino acid formula for MAT1A patients "
                "requiring nutritional support. Alert nutrition team before ordering feeds."
            ),
        },
        {
            "agent": "Valproate (VPA)",
            "risk": "HIGH RISK — hepatotoxic in pre-existing liver disease",
            "mechanism": (
                "Severe/classic MAT1A patients have hepatomegaly and liver disease. "
                "VPA is intrinsically hepatotoxic (CoA sequestration, mitochondrial toxicity, "
                "ammonia accumulation). Combined with pre-existing MAT1A hepatopathy → "
                "risk of VPA-induced liver failure. Also depletes carnitine (secondary "
                "mitochondrial burden). Use LEV first-line."
            ),
        },
        {
            "agent": "Methionine-containing supplements (Met, cysteine, taurine high-dose)",
            "risk": "HIGH RISK — direct methionine loading",
            "mechanism": (
                "Methionine-containing amino acid supplements or sports nutrition products "
                "load the already-saturated pathway → acute methionine spike. "
                "Even cysteine at high doses can be converted (via CBS reverse) and "
                "contribute to methionine load. Taurine from methionine → indirect risk. "
                "Screen ALL supplements for methionine content."
            ),
        },
        {
            "agent": "Antifolates (methotrexate, trimethoprim)",
            "risk": "MODERATE RISK — impairs remethylation in a background of already low SAM",
            "mechanism": (
                "Methotrexate/trimethoprim block DHFR → reduce 5-methylTHF → impair MTR "
                "→ Hcy rises (which is normally already low in MAT1A, so this is less "
                "catastrophic than in cblE/cblG). However, combined with SAM deficiency, "
                "antifolates can further reduce methylation capacity. Avoid when possible."
            ),
        },
        {
            "agent": "Nitrous oxide (N2O)",
            "risk": "LOW RISK (MTR/MeCbl system intact; tHcy normal in MAT1A)",
            "mechanism": (
                "N2O inactivates MTR cobalamin. In MAT1A where tHcy is NORMAL, the "
                "effect of N2O is much less severe than in cblE/cblG or CBS. "
                "HOWEVER: if methionine is very high, any additional Hcy rise from MTR "
                "inactivation could compound methionine-cycle dysfunction. "
                "Inform anesthesiologist; not absolute CI but disclose history."
            ),
        },
    ]

    variants = [
        {
            "variant": "p.Arg264His",
            "domain": "Dimer interface / oligomerization",
            "prevalence": "Most common; AD dominant-negative; ~40% of ALL MAT1A alleles in Western populations",
            "severity": "BENIGN in heterozygotes (MAT III homodimer disrupted; MAT I tetramers partially functional); mild NBS hypermethioninemia; NO treatment needed",
        },
        {
            "variant": "p.Thr222Met",
            "domain": "Catalytic domain",
            "prevalence": "~15% of AR alleles; biallelic causes moderate-severe MAT I deficiency",
            "severity": "Moderate-severe (AR biallelic); white matter, liver, cognitive; methionine 300-900 µmol/L",
        },
        {
            "variant": "p.Ala259Val",
            "domain": "Catalytic domain",
            "prevalence": "~13% of AR alleles",
            "severity": "Moderate; partial enzyme activity retained; manageable with methionine restriction + SAMe",
        },
        {
            "variant": "p.Arg356Trp",
            "domain": "Oligomerization / subunit interface domain",
            "prevalence": "~12% of AR alleles",
            "severity": "Severe; tetramer formation completely disrupted; extreme methioninemia 800-2000; demyelination, seizures",
        },
        {
            "variant": "p.Glu57Lys",
            "domain": "ATP-binding pocket",
            "prevalence": "~10% of AR alleles",
            "severity": "Severe; ATP cannot bind → no SAM synthesis; extreme methioninemia; liver failure in some",
        },
        {
            "variant": "p.Gly378Ser",
            "domain": "C-terminal regulatory domain",
            "prevalence": "~8% of AR alleles",
            "severity": "Moderate; partial SAM synthesis retained; clinical spectrum wide",
        },
        {
            "variant": "p.Tyr21Cys",
            "domain": "N-terminal domain",
            "prevalence": "~2% of AR alleles; rare but severe",
            "severity": "Severe neonatal; complete LOF; methionine >2000 µmol/L; liver failure; seizures; high mortality without early treatment",
        },
    ]

    return {
        "patient_sample": _PATIENTS[:12],
        "seizure_types": seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "treatments": treatments,
        "drug_risks": drug_risks,
        "variant_breakdown": variants,
        "biomarker_ranges": {
            "methionine_umol_l": {
                "severe_MAT_I": "900–2100+ µmol/L (EXTREME; highest of all inherited HHcy disorders)",
                "classic_mild": "250–900 µmol/L",
                "benign_incidental": "100–400 µmol/L",
                "note": "KEY NBS SIGNAL — methionine elevation is the diagnostic trigger; tHcy is NORMAL distinguishing MAT1A from CBS/AHCY",
            },
            "sam_umol_l": {
                "severe": "5–35 µmol/L (VERY LOW — pathognomonic; SAM synthesis blocked)",
                "classic": "15–60 µmol/L (low)",
                "benign": "30–80 µmol/L (low-normal; partial MAT III activity retained)",
                "note": "SAM VERY LOW is PATHOGNOMONIC for MAT1A — OPPOSITE of AHCY where SAM is ELEVATED",
            },
            "sah_umol_l": {
                "all": "5–35 µmol/L (LOW or NORMAL)",
                "note": "SAH LOW/NORMAL because less SAM → less methyltransferase activity → less SAH produced. "
                        "This is the KEY DISTINCTION from AHCY (SAH massively elevated) — "
                        "SAH normal completely rules out AHCY",
            },
            "homocysteine_umol_l": {
                "all": "6–32 µmol/L (NORMAL or ONLY MILDLY ELEVATED)",
                "note": "tHcy NORMAL is a KEY NEGATIVE — distinguishes MAT1A from CBS (tHcy 100-500) and AHCY (tHcy 40-150). "
                        "Normal tHcy in the setting of very high methionine = MAT1A until proven otherwise",
            },
            "mma": "NORMAL (<5 mmol/mol Cr) — ALL patients; propionate arm completely intact",
            "mecbl_adocbl": "BOTH NORMAL — cobalamin system entirely intact; MAT1A does not involve cobalamin",
            "serum_folate": "NORMAL — no methylfolate trap; MTHFR/MTR system intact",
            "megaloblastic_anemia": "ABSENT — no folate trap; not a feature of MAT1A",
            "ck_myopathy": "NORMAL — NO myopathy in MAT1A; CK elevation absent (unlike AHCY CK 500-10,000)",
        },
        "kpi_pcts": {
            "white_matter_disease": pct(lambda p: p["white_matter_disease"]),
            "liver_disease":        pct(lambda p: p["liver_disease"]),
            "idd":                  pct(lambda p: p["idd"]),
            "seizures":             pct(lambda p: p["seizures"]),
            "nbs_detected":         pct(lambda p: p["nbs_detected"]),
            "breath_odor":          pct(lambda p: p["breath_odor"]),
            "myopathy":             0,
            "cardiomyopathy":       0,
        },
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "gene":          "MAT1A (Methionine Adenosyltransferase 1A; also MAT-I/III alpha subunit)",
            "gene_omim":     "*610550",
            "disease_omim":  "#250850 (Hypermethioninemia due to MAT1A deficiency / MAT I/III deficiency)",
            "protein":       "395 amino acids; Mg2+/K+-dependent; liver-specific expression; "
                             "MAT I = homotetramer (Km ~1 mM); MAT III = homodimer (Km ~8 mM, methionine sensor)",
            "cofactor":      "Mg2+ (catalytic) + K+ (activating); ATP substrate",
            "locus":         "10q22.3",
            "inheritance":   "AR (biallelic LOF → MAT I deficiency, severe) / AD (heterozygous p.Arg264His → MAT III disruption, benign)",
            "reaction":      "L-Methionine + ATP → SAM + PPi + Pi (ONLY route for hepatic SAM synthesis)",
            "pathway":       "Methionine cycle — SAM biosynthesis step (FIRST step, upstream of all methyltransferases and AHCY)",
            "regulation":    "SAM product-inhibits MAT I (homotetramer); MAT III (homodimer) is activated by high methionine "
                             "— acts as a methionine safety valve in normal liver; "
                             "MAT1A expression is liver-specific; MAT2A (brain/ubiquitous) is unaffected in MAT1A deficiency",
            "unique_feature": "MAT1A deficiency = SAM DEFICIENCY despite METHIONINE OVERLOAD — "
                              "the opposite biochemical profile to AHCY (where SAM is elevated). "
                              "Brain is partially protected by MAT2A. "
                              "p.Arg264His is the most common human hypermethioninemia allele; BENIGN in heterozygotes.",
        },
        "key_concepts": [
            {
                "concept": "MAT1A vs AHCY — Same Symptom (Hypermethioninemia), OPPOSITE Biochemistry",
                "explanation": (
                    "Both MAT1A and AHCY cause elevated plasma methionine detected on NBS. "
                    "But their biochemistry is DIRECTLY OPPOSITE: "
                    "MAT1A: methionine HIGH + SAM VERY LOW (cannot make SAM) + SAH LOW/NORMAL + tHcy NORMAL. "
                    "AHCY: methionine HIGH + SAM ELEVATED (makes SAM but cannot clear SAH) + SAH MASSIVELY HIGH + tHcy MODERATE. "
                    "CRITICAL TREATMENT IMPLICATION: "
                    "SAM supplementation = TREATMENT for MAT1A (corrects product deficiency). "
                    "SAM supplementation = ABSOLUTELY CONTRAINDICATED in AHCY (catastrophic SAH accumulation). "
                    "Betaine = ABSOLUTELY CONTRAINDICATED in MAT1A (worsens methionine via BHMT). "
                    "Betaine = Level B (cautious) in AHCY. "
                    "Getting this distinction wrong is potentially fatal — always confirm SAM levels before prescribing."
                ),
            },
            {
                "concept": "Benign Incidental Hypermethioninemia (AD p.Arg264His) — No Treatment Needed",
                "explanation": (
                    "p.Arg264His is the MOST COMMON MAT1A variant — dominant-negative. "
                    "Heterozygous p.Arg264His disrupts MAT III homodimer but leaves MAT I tetramer partially functional. "
                    "Result: mild-moderate hypermethioninemia (100-400 µmol/L) with NO clinical sequelae. "
                    "~50% of MAT1A NBS positives are this benign form. "
                    "NO treatment is required in p.Arg264His heterozygotes. "
                    "Excessive dietary restriction in these children causes growth faltering with no benefit. "
                    "Genetic testing to identify p.Arg264His heterozygosity should be done BEFORE dietary restriction."
                ),
            },
            {
                "concept": "Breath Odor — Pathognomonic Dimethylsulfide / Garlic/Cabbage Smell",
                "explanation": (
                    "Excess plasma methionine (200-2000+ µmol/L) undergoes transamination to α-ketobutyrate "
                    "and then bacterial/enzymatic conversion to methanethiol, dimethylsulfide (DMS), and DMSP. "
                    "DMS/methanethiol is exhaled → characteristic sweet, garlic, or cabbage-like breath odor. "
                    "This is PATHOGNOMONIC for severe hypermethioninemia and clinically unique to MAT1A deficiency "
                    "among inherited HHcy disorders. "
                    "Not seen in CBS, AHCY, MTHFR, cblE/cblG, or any combined MMA+HHcy disorder. "
                    "A bedside clue for diagnosis — should prompt immediate methionine measurement."
                ),
            },
            {
                "concept": "SAM Supplementation — Corrects MAT1A but DESTROYS AHCY",
                "explanation": (
                    "SAM (S-adenosylmethionine / SAMe, a dietary supplement) is used in two very different ways: "
                    "MAT1A: exogenous SAM bypasses the enzymatic block → restores methylation capacity → "
                    "improves white matter, liver, and neurodevelopment. Level A for symptomatic MAT1A. "
                    "AHCY: exogenous SAM enters the stalled pathway → methyltransferases produce SAH → "
                    "AHCY cannot hydrolyze SAH → catastrophic SAH accumulation → global methylation crisis. "
                    "ABSOLUTELY CONTRAINDICATED in AHCY. "
                    "Clinical implication: ALWAYS measure SAM AND SAH before prescribing SAMe to a patient "
                    "with hypermethioninemia. Never assume the direction of SAM without lab confirmation."
                ),
            },
            {
                "concept": "MAT Isoforms — MAT1A (Liver) vs MAT2A (Ubiquitous/Brain)",
                "explanation": (
                    "MAT SYSTEM: "
                    "MAT I (homotetramer, alpha1/alpha1/alpha1/alpha1): encoded by MAT1A; liver-specific; Km ~1 mM. "
                    "MAT III (homodimer, alpha1/alpha1): encoded by MAT1A; liver-specific; Km ~8 mM. "
                    "MAT II (heterotetramer, alpha2/alpha2/beta/beta): encoded by MAT2A + MAT2B; UBIQUITOUS (brain, all tissues). "
                    "In MAT1A deficiency: MAT I and III lost → hepatic SAM synthesis severely impaired. "
                    "BUT: MAT II (MAT2A/MAT2B) is INTACT → brain synthesizes its OWN SAM locally → partial neurological protection. "
                    "This is why MAT1A patients can be relatively neurologically preserved compared to AHCY "
                    "(where methyltransferases are blocked GLOBALLY by SAH, not just in liver). "
                    "Neurological features in MAT1A arise from: hypermethioninemia neurotoxicity + "
                    "reduced hepatic SAM export + myelin impairment (MBP methylation)."
                ),
            },
            {
                "concept": "MMA NORMAL + tHcy NORMAL + SAH NORMAL = MAT1A Fingerprint",
                "explanation": (
                    "The MAT1A biochemical fingerprint is unique among hypermethioninemias: "
                    "Methionine VERY HIGH (200-2000+) — screened on NBS. "
                    "SAM VERY LOW (<50 µmol/L) — pathognomonic product deficiency. "
                    "SAH LOW or NORMAL (<30 µmol/L) — less SAM → less methyltransferase activity → less SAH. "
                    "tHcy NORMAL (<30 µmol/L) — Hcy not elevated because less SAH → less AHCY substrate → less Hcy released. "
                    "MMA NORMAL — propionate arm completely intact. "
                    "MeCbl/AdoCbl NORMAL — cobalamin system intact. "
                    "No megaloblastic anemia — folate trap absent. "
                    "This pattern (very high methionine + normal tHcy + low SAM + low SAH) cannot be confused "
                    "with any other inherited metabolic disorder when all biomarkers are measured."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease": "AHCY deficiency (Adenosylhomocysteinase Deficiency) — 20q11.22",
                "distinguishing": (
                    "AHCY: SAM ELEVATED (synthesis intact, downstream clearance blocked by SAH accumulation). "
                    "MAT1A: SAM VERY LOW (synthesis itself blocked). "
                    "AHCY: SAH MASSIVELY HIGH (pathognomonic); MAT1A: SAH LOW/NORMAL. "
                    "AHCY: tHcy MODERATE (40-150 µmol/L); MAT1A: tHcy NORMAL (<30). "
                    "AHCY: SEVERE MYOPATHY 85-90% (hallmark); MAT1A: ABSENT. "
                    "AHCY: cardiomyopathy 60-70%; MAT1A: ABSENT. "
                    "Treatment: SAM is CONTRAINDICATED in AHCY; SAM is THERAPEUTIC in MAT1A."
                ),
            },
            {
                "disease": "CBS deficiency (Classical Homocystinuria) — 21q22.3",
                "distinguishing": (
                    "CBS: tHcy VERY HIGH (100-500 µmol/L, HIGHEST of all HHcy); MAT1A: tHcy NORMAL. "
                    "CBS: methionine elevated (60-500) but LOWER than MAT1A severe (900-2000+). "
                    "CBS: ectopia lentis PATHOGNOMONIC 90%; MAT1A: ABSENT. "
                    "CBS: marfanoid habitus 80%; MAT1A: ABSENT. "
                    "CBS: B6 responsiveness 50% (PLP-dependent); MAT1A: NOT B6-responsive. "
                    "CBS: thromboembolism HIGH RISK (50-60%); MAT1A: LOW (tHcy normal). "
                    "CBS: betaine Level A; MAT1A: betaine ABSOLUTELY CONTRAINDICATED."
                ),
            },
            {
                "disease": "GNMT deficiency (Glycine N-Methyltransferase Deficiency) — 6p12.1",
                "distinguishing": (
                    "GNMT: methionine HIGH + SAM ELEVATED (GNMT cannot consume SAM via glycine methylation). "
                    "MAT1A: SAM VERY LOW (cannot synthesize SAM). "
                    "GNMT: SAH LOW/NORMAL (same as MAT1A — less methyltransfer activity); SAM elevated. "
                    "GNMT: liver disease prominent (similar to MAT1A); no CNS white matter. "
                    "GNMT: tHcy NORMAL (same as MAT1A). "
                    "Key: GNMT has SAM elevated; MAT1A has SAM severely depressed."
                ),
            },
            {
                "disease": "Tyrosinemia type I / other causes of isolated hypermethioninemia",
                "distinguishing": (
                    "Tyrosinemia type I (FAH deficiency) can cause secondary hypermethioninemia via liver damage. "
                    "Succinylacetone is the diagnostic marker for Tyrosinemia I (not present in MAT1A). "
                    "Methionine elevation in Tyrosinemia I is secondary; tyrosine and tyrosine catabolites elevated. "
                    "MAT1A: SAM very low (primary), no succinylacetone, no elevated tyrosine catabolites. "
                    "Also: prematurity, transient neonatal hypermethioninemia, liver disease of any cause "
                    "can cause secondary methionine elevation — these resolve with liver recovery."
                ),
            },
            {
                "disease": "cblE (MTRR) / cblG (MTR) — Isolated HHcy with LOW methionine",
                "distinguishing": (
                    "cblE/cblG: methionine LOW (remethylation impaired → methionine not produced from Hcy). "
                    "MAT1A: methionine VERY HIGH. "
                    "cblE/cblG: tHcy 40-200 µmol/L (ELEVATED); MAT1A: tHcy NORMAL. "
                    "cblE/cblG: MeCbl absent (fibroblasts); MAT1A: MeCbl NORMAL. "
                    "cblE/cblG: megaloblastic anemia 80-90% (methylfolate trap); MAT1A: ABSENT. "
                    "No overlap — opposite biochemical direction for methionine."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    ov = get_overview()
    print(f"Overview keys: {list(ov.keys())}")
    print(f"Cohort n={ov['cohort_n']}, pheno dist={ov['phenotype_distribution']}")
    print(f"KPIs: {json.dumps(ov['kpis'], indent=2)}")
    br = get_breakdown()
    print(f"Breakdown — treatments: {len(br['treatments'])}, drug_risks: {len(br['drug_risks'])}, variants: {len(br['variant_breakdown'])}")
    df = get_definitions()
    print(f"Definitions — concepts: {len(df['key_concepts'])}, differential: {len(df['differential_diagnosis'])}")
    print("MAT1A dashboard OK")
