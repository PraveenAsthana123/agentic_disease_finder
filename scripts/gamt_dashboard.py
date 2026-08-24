#!/usr/bin/env python3
"""GAMT (Guanidinoacetate N-Methyltransferase Deficiency) Epilepsy Dashboard.

GAMT encodes Guanidinoacetate N-methyltransferase, a SAM-dependent methyltransferase
expressed primarily in the liver and pancreas that catalyses the SECOND and FINAL step
of creatine biosynthesis.

CREATINE BIOSYNTHESIS PATHWAY (two-step):

  Step 1 — AGAT (kidney/pancreas):
    L-Arginine + Glycine → Ornithine + Guanidinoacetate (GAA)

  Step 2 — GAMT (liver/pancreas):
    SAM + Guanidinoacetate (GAA) → SAH + Creatine

GAMT IS THE FINAL COMMITTED STEP OF CREATINE SYNTHESIS:
  - GAMT is one of the major SAM-consuming methyltransferases in the liver
  - The reaction is irreversible — creatine cannot be catabolised back to GAA
  - Creatine is transported by SLC6A8 (creatine transporter) into muscle/brain
  - In muscle: Creatine + ATP ⇌ Phosphocreatine + ADP (creatine kinase)
  - Phosphocreatine is the cellular energy buffer (millisecond ATP regeneration)
  - In brain: Creatine/phosphocreatine buffer is ESSENTIAL for synaptic transmission

GAMT LOF — DUAL PATHOLOGY (unique among SAM-methyltransferase defects):

  1. CREATINE CANNOT BE MADE:
     SAM + GAA → × GAMT BLOCKED → Creatine = ZERO (or near-zero)
     ↓
     • Muscle: ATP regeneration impaired (mild myopathy, creatine-deficient)
     • Brain: synaptic energy buffer absent → neuronal hyperexcitability → SEIZURES
     • MRS: creatine peak at 3.0 ppm ABSENT — pathognomonic on H-MRS

  2. GUANIDINOACETATE (GAA) ACCUMULATES MASSIVELY — DIRECTLY NEUROTOXIC:
     AGAT still active → GAA continues to be produced → GAMT cannot clear it
     GAA plasma: 10–100× normal (normal <1–3 µmol/L → GAMT: 50–300 µmol/L)
     GAA urine: 20–200× normal
     GAA CSF: markedly elevated

     GAA NEUROTOXICITY MECHANISMS:
     (a) GAA inhibits GABA-A receptors → reduces inhibitory tone → hyperexcitability
     (b) GAA activates NMDA glutamate receptors → excitotoxicity
     (c) GAA generates reactive oxygen species → oxidative neuronal damage
     (d) GAA depletes GABA in cortical interneurons
     → GAA is a potent endogenous convulsant — more epileptogenic than creatine deficiency alone

METABOLIC SIGNATURE OF GAMT DEFICIENCY:
  Guanidinoacetate (GAA):   MASSIVELY ELEVATED (50–300 µmol/L; normal <3) — PATHOGNOMONIC
  Creatine (plasma):        VERY LOW or ABSENT (<5 µmol/L; normal 20–80)
  Creatinine (plasma/urine): LOW (creatinine is creatine catabolite — source absent)
  SAM:                      LOW-NORMAL (GAMT consumes SAM; without GAMT, SAM slightly elevated)
  Methionine:               NORMAL (unlike MAT1A/GNMT/AHCY — major KEY NEGATIVE)
  SAH:                      NORMAL-LOW (GAMT normally generates SAH from GAA methylation; blocked)
  tHcy:                     NORMAL (CBS/MTR/MTRR intact)
  MMA:                      NORMAL (propionate arm intact)
  MeCbl/AdoCbl:             NORMAL (cobalamin system intact)
  Sarcosine:                NORMAL (GNMT intact — major KEY NEGATIVE distinguishing from GNMT)
  Brain H-MRS:              Creatine peak (3.0 ppm) ABSENT — PATHOGNOMONIC

CLINICAL HALLMARKS (unique combination — no other disorder):
  (1) Guanidinoacetate MASSIVELY ELEVATED — pathognomonic, measured on urine/plasma amino acids
  (2) Creatine ABSENT on brain H-MRS (creatine/phosphocreatine 3.0 ppm peak absent)
  (3) Drug-resistant epilepsy (60–80%) — GAA is a direct convulsant; often refractory to AEDs
  (4) Profound IDD (90%+) — both creatine deficiency AND GAA neurotoxicity
  (5) Low/absent urine creatinine — distinctive; urine creatinine critically low
  (6) Methionine NORMAL — KEY NEGATIVE vs all other SAM-cycle disorders

GAMT vs GNMT (BOTH SAM-CONSUMING METHYLTRANSFERASES — VERY DIFFERENT):
  GNMT: SAM + Glycine → SAH + Sarcosine (safety valve; ~50-75% hepatic SAM consumed)
    LOF: SAM ↑↑↑, methionine ↑↑, sarcosine absent, tHcy normal, liver disease dominant
  GAMT: SAM + GAA → SAH + Creatine (creatine biosynthesis; ~5-15% hepatic SAM consumed)
    LOF: GAA ↑↑↑, creatine absent, methionine NORMAL, drug-resistant epilepsy dominant
  KEY DISTINCTION:
    Methionine: GNMT → HIGH; GAMT → NORMAL
    SAM:        GNMT → VERY HIGH; GAMT → NORMAL-LOW (small fraction of SAM consumed)
    Dominant feature: GNMT → liver disease; GAMT → epilepsy + IDD

GAMT vs AGAT (THE OTHER CREATINE SYNTHESIS STEP):
  AGAT deficiency: GAA = VERY LOW (AGAT cannot make GAA → no GAA accumulation)
  GAMT deficiency: GAA = MASSIVELY HIGH (AGAT makes GAA but GAMT cannot clear it)
  Both: Creatine absent, IDD, seizures, respond to creatine supplementation
  MAJOR DIFFERENCE: GAA high in GAMT, GAA low in AGAT — opposite GAA direction

INHERITANCE AND GENE:
  Gene: GAMT, 6p21.3, 236 amino acids
  Autosomal Recessive — biallelic LOF required
  ~200–250 patients reported worldwide (2026) — rare but more common than GNMT/MAT1A/AHCY
  Heterozygotes: slightly elevated GAA; usually asymptomatic

OMIM:
  Gene:    *601240 (GAMT)
  Disease: #612736 (Guanidinoacetate methyltransferase deficiency / GAMT deficiency /
            Cerebral creatine deficiency syndrome 2, CCDS2)
"""

import random
import math

_SEED = 113
_rng = random.Random(_SEED)

# ---------------------------------------------------------------------------
# Phenotypic classes
# ---------------------------------------------------------------------------
PHENOTYPE_CLASSES = [
    "Classic-Severe (profound IDD + drug-resistant epilepsy + movement disorder / GAA massively elevated / MODAL)",
    "Moderate (moderate IDD + partially controlled seizures / GAA elevated / creatine very low)",
    "Mild (mild IDD + infrequent seizures / GAA mildly elevated / late-presenting)",
]

PHENO_WEIGHTS = [0.65, 0.25, 0.10]  # 65% classic-severe / 25% moderate / 10% mild

VARIANTS = [
    "p.Ile124Val",     # catalytic site; most common worldwide; ~25%; moderate/severe
    "p.Asp204Asn",     # catalytic; severe; ~20%
    "p.Ser99Asn",      # substrate-binding adjacent; moderate; ~15%
    "p.Arg286Cys",     # GAA substrate binding; severe; ~12%
    "p.Ala223Val",     # mild-moderate; ~10%
    "p.Arg216Gln",     # catalytic; moderate; ~10%
    "c.IVS4+1G>A",    # splice-null; severe; ~8%
]

VARIANT_WEIGHTS = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]

SEIZURE_TYPES = [
    "Focal seizures (frontal/occipital — GAA-mediated GABA-A inhibition, NMDA activation)",
    "Generalized tonic-clonic (severe GAA toxicity / creatine depletion)",
    "Infantile spasms / West syndrome (neonatal-onset severe form)",
    "Myoclonic (GAA-mediated cortical hyperexcitability / creatine-deficient)",
    "Lennox-Gastaut pattern (refractory multi-type — severe IDD + drug-resistant)",
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
    is_mild   = "Mild"     in pheno
    is_severe = "Classic"  in pheno
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            if is_mild and v in ("p.Asp204Asn", "p.Arg286Cys", "c.IVS4+1G>A"):
                return _rng.choice(["p.Ile124Val", "p.Ser99Asn", "p.Ala223Val"])
            if is_severe and v == "p.Ala223Val":
                return _rng.choice(["p.Asp204Asn", "p.Arg286Cys", "c.IVS4+1G>A"])
            return v
    return VARIANTS[0]


def _make_patient(idx):
    pheno      = _choose_pheno()
    is_mild    = "Mild"    in pheno
    is_moderate = "Moderate" in pheno
    is_severe  = "Classic" in pheno   # Classic-Severe

    # GAA — MASSIVELY ELEVATED in all GAMT; highest in severe
    if is_severe:
        gaa_umol    = round(_rng.uniform(120, 300), 1)   # plasma GAA; normal <3 µmol/L
        creatine_pl = round(_rng.uniform(0.5, 5.0), 1)   # plasma creatine; normal 20-80
        creatinine  = round(_rng.uniform(2,  15),   1)   # urine creatinine very low
        ck_u_l      = round(_rng.uniform(80, 400),  0)   # CK mildly elevated (creatine deficiency myopathy)
    elif is_moderate:
        gaa_umol    = round(_rng.uniform(50, 150),  1)
        creatine_pl = round(_rng.uniform(3,  12),   1)
        creatinine  = round(_rng.uniform(10, 35),   1)
        ck_u_l      = round(_rng.uniform(50, 200),  0)
    else:  # mild
        gaa_umol    = round(_rng.uniform(15,  60),  1)
        creatine_pl = round(_rng.uniform(8,   25),  1)
        creatinine  = round(_rng.uniform(25,  60),  1)
        ck_u_l      = round(_rng.uniform(30,  100), 0)

    # Methionine: NORMAL (major KEY NEGATIVE)
    methionine  = round(_rng.uniform(18, 42), 1)   # normal range ~10-45 µmol/L
    # tHcy: NORMAL
    hcy_umol    = round(_rng.uniform(5, 14), 1)
    # SAM: slightly low-normal (GAMT was consuming SAM; blocked → SAM no lower)
    # Actually in GAMT deficiency SAM is slightly LOW or NORMAL (not elevated like GNMT)
    sam_umol    = round(_rng.uniform(50, 120), 1)   # normal range 60-120; slightly low-normal
    sah_umol    = round(_rng.uniform(10, 30), 1)    # NORMAL-LOW

    # Clinical features
    seizures       = _rng.random() < (0.82 if is_severe else (0.60 if is_moderate else 0.30))
    drug_resistant = seizures and _rng.random() < (0.75 if is_severe else (0.45 if is_moderate else 0.15))
    idd            = _rng.random() < (0.98 if is_severe else (0.85 if is_moderate else 0.55))
    speech_absent  = _rng.random() < (0.75 if is_severe else (0.40 if is_moderate else 0.10))
    movement_dis   = _rng.random() < (0.50 if is_severe else (0.25 if is_moderate else 0.05))
    autism_like    = _rng.random() < (0.55 if is_severe else (0.35 if is_moderate else 0.15))
    hypotonia      = _rng.random() < (0.50 if is_severe else (0.30 if is_moderate else 0.10))
    myopathy       = _rng.random() < (0.35 if is_severe else (0.18 if is_moderate else 0.05))   # mild creatine-deficiency myopathy
    nbs_detected   = _rng.random() < (0.75 if is_severe else (0.55 if is_moderate else 0.30))
    on_creatine    = _rng.random() < (0.0  if is_mild   else (0.80 if is_moderate else 0.90))
    on_ornithine   = _rng.random() < (0.0  if is_mild   else (0.70 if is_moderate else 0.85))

    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    age_onset_mo = (
        _rng.randint(2,  18) if is_severe  else
        _rng.randint(6,  36) if is_moderate else
        _rng.randint(12, 60)  # mild: later-presenting
    )

    variant = _choose_variant(pheno)

    return {
        "id":                  f"GAMT-{idx:03d}",
        "phenotype":           pheno,
        "variant":             variant,
        "age_onset_months":    age_onset_mo,
        # Biomarkers
        "gaa_umol_l":          gaa_umol,         # MASSIVELY ELEVATED — pathognomonic
        "creatine_umol_l":     creatine_pl,       # VERY LOW / ABSENT
        "creatinine_umol_l":   creatinine,        # LOW (creatinine = creatine catabolite)
        "methionine_umol_l":   methionine,        # NORMAL — KEY NEGATIVE vs GNMT/MAT1A/AHCY
        "homocysteine_umol_l": hcy_umol,          # NORMAL
        "sam_umol_l":          sam_umol,          # LOW-NORMAL
        "sah_umol_l":          sah_umol,          # NORMAL-LOW
        "ck_u_l":              ck_u_l,
        "mma_normal":          True,
        "mecbl_normal":        True,
        "adocbl_normal":       True,
        "sarcosine_normal":    True,   # GNMT intact — sarcosine PRESENT (KEY NEGATIVE vs GNMT)
        # Clinical
        "seizures":            seizures,
        "seizure_type":        seizure_type,
        "drug_resistant_sz":   drug_resistant,
        "idd":                 idd,
        "speech_absent":       speech_absent,
        "movement_disorder":   movement_dis,
        "autism_like":         autism_like,
        "hypotonia":           hypotonia,
        "myopathy":            myopathy,          # mild creatine-deficiency myopathy (unlike GNMT/MAT1A)
        "nbs_detected":        nbs_detected,
        "on_creatine_tx":      on_creatine,
        "on_ornithine_tx":     on_ornithine,
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
        "dashboard_id": "gamt",
        "title": "GAMT Epilepsy Dashboard",
        "subtitle": (
            "Guanidinoacetate N-Methyltransferase Deficiency — "
            "Cerebral Creatine Deficiency Syndrome 2 (CCDS2) / "
            "GAMT-236aa-SAM-Dependent-6p21.3-AR"
        ),
        "gene": "GAMT",
        "disease_name": (
            "Guanidinoacetate N-Methyltransferase Deficiency "
            "(GAMT Deficiency / Cerebral Creatine Deficiency Syndrome 2 / CCDS2)"
        ),
        "chromosome": "6p21.3",
        "inheritance": "Autosomal Recessive — biallelic LOF",
        "omim_gene":    "OMIM *601240",
        "omim_disease": "OMIM #612736",
        "protein_size": (
            "236 aa; SAM-dependent methyltransferase; catalyses final step of creatine biosynthesis; "
            "SAM + Guanidinoacetate → SAH + Creatine; "
            "expressed predominantly in liver and pancreas; "
            "belongs to the class I SAM-dependent methyltransferase superfamily (seven-stranded β-sheet); "
            "homodimeric active form; "
            "substrate specificity: guanidinoacetate (GAA) — does NOT methylate glycine (GNMT substrate)"
        ),
        "prevalence": (
            "~200–250 patients reported worldwide (2026); "
            "most common of the three cerebral creatine deficiency syndromes (GAMT > AGAT > SLC6A8 female); "
            "more common than GNMT (<30 cases) and MAT1A; "
            "estimated prevalence 1:250,000–1:1,000,000; "
            "likely underdiagnosed due to non-specific phenotype (IDD + epilepsy)"
        ),
        "cohort_n": n,
        "phenotype_distribution": pheno_dist,
        "function": (
            "GAMT (Guanidinoacetate N-methyltransferase) catalyses: SAM + Guanidinoacetate (GAA) → SAH + Creatine. "
            "It is the SECOND and FINAL committed step in creatine biosynthesis. "
            "Step 1 (AGAT, kidney/pancreas): Arginine + Glycine → Ornithine + Guanidinoacetate. "
            "Step 2 (GAMT, liver/pancreas): SAM + GAA → SAH + Creatine. "
            "Creatine is then transported by SLC6A8 (creatine transporter) into muscle and brain. "
            "In muscle: Creatine kinase phosphorylates creatine → phosphocreatine (PCr), "
            "the millisecond ATP buffer for muscle contraction. "
            "In brain: PCr/creatine buffer is ESSENTIAL for synaptic ATP regeneration — "
            "without creatine, neurons cannot sustain rapid firing during synaptic activity."
        ),
        "mechanism": (
            "GAMT LOF creates a DUAL PATHOLOGICAL STATE: "
            "(1) CREATINE ABSENT: The final methylation step cannot proceed. "
            "Plasma creatine approaches zero (<5 µmol/L; normal 20–80). "
            "Brain H-MRS: creatine peak at 3.0 ppm is ABSENT — pathognomonic. "
            "Muscle and brain are creatine-starved → no phosphocreatine buffer → "
            "neuronal hyperexcitability because ATP cannot be rapidly regenerated during burst firing. "
            "(2) GUANIDINOACETATE (GAA) MASSIVELY ACCUMULATES: "
            "AGAT continues to produce GAA (kidney/pancreas still active), "
            "but GAMT cannot clear it → GAA accumulates to 50–300 µmol/L (normal <3). "
            "GAA is DIRECTLY EPILEPTOGENIC via multiple mechanisms: "
            "(a) GAA inhibits GABA-A receptors → reduces cortical inhibition; "
            "(b) GAA activates NMDA receptors → excitotoxicity; "
            "(c) GAA depletes GABA in cortical interneurons; "
            "(d) GAA generates reactive oxygen species → oxidative neuronal damage. "
            "The combination of absent creatine + massive GAA makes seizures in GAMT "
            "the most drug-resistant among ALL creatine synthesis disorders — "
            "AEDs alone rarely achieve seizure control without treating the underlying metabolic defect."
        ),
        "key_positive_features": (
            "KEY POSITIVES (unique to GAMT among inherited epilepsies): "
            "(1) Guanidinoacetate (GAA) MASSIVELY ELEVATED (50–300 µmol/L; normal <3) — pathognomonic. "
            "    Detectable on plasma/urine amino acid panel or metabolomics. "
            "(2) Creatine ABSENT or very low (<5 µmol/L; normal 20–80). "
            "(3) Brain H-MRS: creatine/phosphocreatine peak (3.0 ppm) ABSENT — pathognomonic. "
            "(4) Urine creatinine VERY LOW (creatinine = creatine catabolite — source absent). "
            "(5) Drug-resistant epilepsy (60–80%) — AEDs alone fail; metabolic treatment required. "
            "(6) Profound IDD (90%+) with severely impaired/absent speech. "
            "KEY NEGATIVES (absent in GAMT — major differentiators): "
            "(1) Methionine NORMAL — KEY NEGATIVE vs MAT1A/GNMT/AHCY/CBS. "
            "(2) tHcy NORMAL (<15 µmol/L) — KEY NEGATIVE vs CBS/MTHFR/cblE/cblG. "
            "(3) MMA NORMAL — KEY NEGATIVE vs cblA/cblB/cblC/cblD/MMUT. "
            "(4) SAM NORMAL-LOW (not elevated) — KEY NEGATIVE vs GNMT (SAM very high). "
            "(5) Sarcosine PRESENT/NORMAL — KEY NEGATIVE vs GNMT (sarcosine absent). "
            "(6) Megaloblastic anemia ABSENT (folate/cobalamin system intact). "
            "(7) Myopathy: MILD creatine-deficiency myopathy POSSIBLE (not hallmark). "
            "(8) Ectopia lentis ABSENT. Liver disease ABSENT (unlike GNMT liver-dominant)."
        ),
        "nbs_primary": (
            "Expanded NBS — urine or plasma guanidinoacetate (GAA) elevation (50–300 µmol/L; normal <3). "
            "~70–75% of GAMT cases detected on expanded NBS programs including GAA. "
            "Standard MS/MS NBS (acylcarnitine + amino acids): may MISS GAMT — "
            "creatine is not a standard NBS analyte; GAA is not routinely measured on basic MS/MS. "
            "Low urine creatinine is an indirect NBS flag — but not specific. "
            "Second-tier: plasma/urine GAA by specific amino acid chromatography or LC-MS/MS. "
            "GAMT must be ADDED to expanded NBS panel for reliable detection — "
            "without it, most cases present late with severe IDD + epilepsy."
        ),
        "nbs_secondary": (
            "Plasma guanidinoacetate (GAA): MASSIVELY ELEVATED (50–300 µmol/L; normal <3 µmol/L). "
            "Plasma creatine: VERY LOW or absent (<5 µmol/L; normal 20–80). "
            "Urine creatinine: CRITICALLY LOW (creatinine is creatine catabolite — no creatine source). "
            "Urine GAA: 20–200× normal. "
            "Brain H-MRS (diagnostic standard): creatine/phosphocreatine peak at 3.0 ppm ABSENT. "
            "Plasma methionine: NORMAL (no hypermethioninemia — distinguishes from MAT1A/GNMT). "
            "Plasma total homocysteine: NORMAL (<15 µmol/L). "
            "Plasma SAM: NORMAL-LOW (no SAM accumulation — distinguishes from GNMT). "
            "Urine organic acids: MMA NORMAL; no characteristic organic acid elevation in GAMT. "
            "Serum B12/folate: NORMAL. CK: mildly elevated (30–400 U/L) in ~30–40% of cases. "
            "EEG: epileptiform discharges (85%); hypsarrhythmia in infantile spasm cases. "
            "Brain MRI: white matter changes, cortical atrophy; may be relatively non-specific. "
            "GAMT enzyme activity (liver biopsy or fibroblasts): absent or severely reduced. "
            "Genetic confirmation: GAMT sequencing."
        ),
        "kpis": {
            "avg_gaa_umol_l":          avg("gaa_umol_l"),
            "avg_creatine_umol_l":     avg("creatine_umol_l"),
            "avg_creatinine_umol_l":   avg("creatinine_umol_l"),
            "avg_methionine_umol_l":   avg("methionine_umol_l"),
            "avg_homocysteine_umol_l": avg("homocysteine_umol_l"),
            "avg_ck_u_l":              avg("ck_u_l"),
            "pct_seizures":            pct(lambda p: p["seizures"]),
            "pct_drug_resistant":      pct(lambda p: p["drug_resistant_sz"]),
            "pct_idd":                 pct(lambda p: p["idd"]),
            "pct_speech_absent":       pct(lambda p: p["speech_absent"]),
            "pct_movement_disorder":   pct(lambda p: p["movement_disorder"]),
            "pct_autism_like":         pct(lambda p: p["autism_like"]),
            "pct_nbs_detected":        pct(lambda p: p["nbs_detected"]),
            "pct_myopathy":            pct(lambda p: p["myopathy"]),
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Focal seizures (frontal/occipital — GAA GABA-A inhibition + NMDA activation)", "pct": 38},
        {"type": "Generalized tonic-clonic (severe GAA toxicity + creatine depletion)", "pct": 28},
        {"type": "Infantile spasms / West syndrome (neonatal-onset severe GAMT)", "pct": 20},
        {"type": "Myoclonic (GAA-mediated cortical hyperexcitability)", "pct": 8},
        {"type": "Lennox-Gastaut pattern (refractory multi-type epilepsy)", "pct": 6},
    ]

    metabolic_triggers = [
        {
            "trigger": "High-arginine diet (meat, fish, dairy — AGAT substrate supply)",
            "pct": 78,
            "mechanism": (
                "AGAT (kidney): Arginine + Glycine → Ornithine + GAA. "
                "High dietary arginine → more AGAT activity → more GAA produced → "
                "GAMT cannot clear it → GAA accumulates further → worse epilepsy. "
                "Arginine restriction reduces AGAT activity and GAA production. "
                "Low-arginine diet (Level B treatment) — combined with creatine + ornithine."
            ),
        },
        {
            "trigger": "Intercurrent illness / catabolic state / fasting",
            "pct": 65,
            "mechanism": (
                "Catabolism releases arginine from muscle protein → AGAT substrate increases → "
                "more GAA produced → exacerbation of seizures. "
                "IV glucose to suppress catabolism during illness. "
                "Maintain oral creatine and ornithine supplementation during illness."
            ),
        },
        {
            "trigger": "Valproate (VPA) — MODERATE RISK",
            "pct": 55,
            "mechanism": (
                "VPA inhibits creatine kinase in some reports, potentially worsening the "
                "energy-depletion phenotype of GAMT deficiency. "
                "VPA also depletes carnitine (may compound creatine-deficient myopathy). "
                "LEV is preferred first-line AED; VPA should be avoided where possible. "
                "Not an absolute contraindication in refractory cases, but use with caution."
            ),
        },
        {
            "trigger": "Glycine-rich supplements / collagen / gelatin",
            "pct": 48,
            "mechanism": (
                "AGAT: Arginine + Glycine → GAA. "
                "Excess dietary glycine can drive AGAT toward more GAA production "
                "(glycine is the second substrate of AGAT). "
                "Avoid glycine supplements; collagen/gelatin rich in glycine — use minimally."
            ),
        },
        {
            "trigger": "Creatine deficiency (untreated / non-adherent) — chronic worsening",
            "pct": 90,
            "mechanism": (
                "Without exogenous creatine supplementation, brain and muscle remain creatine-depleted. "
                "Creatine monohydrate (300–400 mg/kg/day) is Level A — the primary treatment. "
                "Stopping creatine leads to rapid recurrence of severe epilepsy. "
                "Creatine supplementation partially normalizes brain H-MRS creatine peak."
            ),
        },
        {
            "trigger": "Metformin — MODERATE RISK",
            "pct": 38,
            "mechanism": (
                "Metformin inhibits mitochondrial complex I and competes with creatine transporter (SLC6A8). "
                "In creatine-deficient GAMT, any further impairment of creatine uptake "
                "can worsen the energy deficit. Avoid metformin in GAMT patients."
            ),
        },
    ]

    treatments = [
        {
            "name": "Creatine Monohydrate",
            "level": "Level A — First-Line Primary Treatment",
            "note": (
                "300–400 mg/kg/day oral creatine monohydrate. "
                "Replaces the absent creatine — corrects the phosphocreatine energy deficit in brain/muscle. "
                "Brain H-MRS creatine peak partially normalises within weeks-months of treatment. "
                "Seizure frequency decreases in ~70% on creatine alone (but drug-resistant cases need ornithine). "
                "Must be continued indefinitely — stopping causes rapid relapse. "
                "Best outcomes when started early (presymptomatic NBS detection). "
                "Does NOT normalise GAA — ornithine supplementation required to reduce GAA."
            ),
        },
        {
            "name": "Ornithine Supplementation",
            "level": "Level A — First-Line (Combined with Creatine)",
            "note": (
                "100–150 mg/kg/day L-ornithine. "
                "Mechanism: Ornithine competes with glycine for the AGAT active site — "
                "AGAT: Arginine + Glycine → Ornithine + GAA; excess ornithine shifts equilibrium → "
                "less GAA produced → GAA plasma levels fall. "
                "Reduces GAA by 60–90% in responders. "
                "Combined creatine + ornithine is more effective than either alone. "
                "GAA reduction → reduced epileptogenicity → improved seizure control. "
                "Ornithine also reduces IDD progression if started early."
            ),
        },
        {
            "name": "Arginine Restriction",
            "level": "Level B — Adjunct",
            "note": (
                "Low-arginine diet (protein moderation, especially arginine-rich sources: "
                "meat, fish, dairy). "
                "Reduces AGAT substrate → less GAA synthesis. "
                "Less powerful than ornithine supplementation for GAA reduction. "
                "Combined arginine restriction + ornithine supplementation maximises GAA reduction. "
                "Requires dietitian monitoring to avoid protein deficiency."
            ),
        },
        {
            "name": "LEV (Levetiracetam)",
            "level": "Level A — First-Line AED",
            "note": (
                "No interaction with creatine metabolism. "
                "No hepatotoxicity (important — avoid VPA). "
                "Does not worsen GAA accumulation. "
                "First-line AED in GAMT epilepsy across all seizure types. "
                "Seizure control is often INCOMPLETE without metabolic treatment (creatine + ornithine). "
                "Combined LEV + creatine + ornithine achieves best seizure outcomes."
            ),
        },
        {
            "name": "ACTH / Vigabatrin",
            "level": "Level A — Infantile Spasms",
            "note": (
                "Standard IS/West syndrome protocol for GAMT presenting with infantile spasms. "
                "ACTH preferred; vigabatrin second-line. "
                "Must be combined with metabolic treatment (creatine + ornithine) — "
                "ACTH/vigabatrin alone will not prevent relapse without metabolic correction."
            ),
        },
        {
            "name": "Glycine Restriction",
            "level": "Level C — Experimental",
            "note": (
                "Reduces AGAT substrate (glycine is second substrate of AGAT: Arg + Gly → Orn + GAA). "
                "Less practical than arginine restriction. "
                "Used in some centres as adjunct when GAA remains elevated despite creatine + ornithine. "
                "No randomised trial data."
            ),
        },
    ]

    drug_risks = [
        {
            "drug": "Valproate (VPA)",
            "risk": "MODERATE RISK — Prefer LEV",
            "mechanism": (
                "VPA may inhibit creatine kinase and creatine uptake. "
                "Carnitine depletion from VPA compounds creatine-deficiency myopathy. "
                "Not absolute CI, but LEV is strongly preferred. "
                "If VPA unavoidable, supplement carnitine and monitor CK."
            ),
        },
        {
            "drug": "Metformin",
            "risk": "MODERATE RISK — Avoid",
            "mechanism": (
                "Inhibits mitochondrial complex I; may compete with SLC6A8 creatine transporter. "
                "In GAMT, any further creatine uptake impairment worsens the energy deficit. "
                "Avoid in GAMT patients with concurrent diabetes/metabolic syndrome."
            ),
        },
        {
            "drug": "Arginine supplements / arginine-rich formulas",
            "risk": "HIGH RISK — Feeds GAA Synthesis",
            "mechanism": (
                "Arginine is the primary AGAT substrate: Arginine + Glycine → GAA. "
                "Arginine supplements directly increase GAA production → worsens seizure burden. "
                "Avoid arginine-enriched sports/medical nutrition in GAMT patients."
            ),
        },
        {
            "drug": "Glycine supplements / collagen / gelatin (excess)",
            "risk": "MODERATE RISK",
            "mechanism": (
                "Glycine is the second AGAT substrate. Excess dietary glycine can increase AGAT-driven "
                "GAA production. Avoid glycine supplements; limit collagen-based products."
            ),
        },
        {
            "drug": "N2O (Nitrous oxide)",
            "risk": "LOW RISK (cobalamin system intact)",
            "mechanism": (
                "GAMT deficiency does NOT affect the cobalamin/methylation system. "
                "MTR (cobalamin-dependent) is intact. "
                "N2O is NOT an absolute CI in GAMT — unlike cblE/cblG/MTHFR. "
                "Use standard anaesthetic precautions."
            ),
        },
        {
            "drug": "SAMe / SAM supplements",
            "risk": "NOT INDICATED (no SAM deficit)",
            "mechanism": (
                "GAMT deficiency does NOT cause SAM deficiency (unlike MAT1A). "
                "SAM supplements have no role in GAMT — do not confuse with MAT1A treatment. "
                "The therapeutic target is GAA reduction and creatine replacement."
            ),
        },
    ]

    # Variant breakdown
    variant_counts: dict[str, int] = {}
    for p in _PATIENTS:
        v = p["variant"]
        variant_counts[v] = variant_counts.get(v, 0) + 1

    variants_list = [
        {
            "variant": v,
            "count": variant_counts.get(v, 0),
            "pct": round(variant_counts.get(v, 0) / n * 100),
        }
        for v in VARIANTS
    ]

    # Patient sample (first 12)
    sample = [
        {
            "id":              p["id"],
            "phenotype":       p["phenotype"].split(" (")[0],
            "variant":         p["variant"],
            "gaa":             p["gaa_umol_l"],
            "creatine":        p["creatine_umol_l"],
            "creatinine":      p["creatinine_umol_l"],
            "methionine":      p["methionine_umol_l"],
            "homocysteine":    p["homocysteine_umol_l"],
            "ck":              p["ck_u_l"],
            "seizures":        p["seizures"],
            "drug_resistant":  p["drug_resistant_sz"],
            "nbs":             p["nbs_detected"],
        }
        for p in _PATIENTS[:12]
    ]

    return {
        "dashboard_id": "gamt",
        "cohort_n": n,
        "kpi_pcts": {
            "pct_seizures":          pct(lambda p: p["seizures"]),
            "pct_drug_resistant":    pct(lambda p: p["drug_resistant_sz"]),
            "pct_idd":               pct(lambda p: p["idd"]),
            "pct_speech_absent":     pct(lambda p: p["speech_absent"]),
            "pct_movement_disorder": pct(lambda p: p["movement_disorder"]),
            "pct_autism_like":       pct(lambda p: p["autism_like"]),
            "pct_hypotonia":         pct(lambda p: p["hypotonia"]),
            "pct_myopathy":          pct(lambda p: p["myopathy"]),
            "pct_nbs":               pct(lambda p: p["nbs_detected"]),
        },
        "phenotype_dist": [
            {"label": "Classic-Severe", "pct": 65},
            {"label": "Moderate",       "pct": 25},
            {"label": "Mild",           "pct": 10},
        ],
        "seizure_types":      seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "treatments":         treatments,
        "drug_risks":         drug_risks,
        "variants":           variants_list,
        "patient_sample":     sample,
        "biomarker_ranges": {
            "gaa_mild_umol_l":         "15–60  (normal <3)",
            "gaa_moderate_umol_l":     "50–150 (normal <3)",
            "gaa_severe_umol_l":       "120–300 (normal <3) — PATHOGNOMONIC",
            "creatine_umol_l":         "<5–25 (normal 20–80) — VERY LOW",
            "creatinine_umol_l":       "2–60 (normal 40–120) — CRITICALLY LOW",
            "methionine_umol_l":       "18–42 (NORMAL — KEY NEGATIVE)",
            "homocysteine_umol_l":     "5–14 (NORMAL — KEY NEGATIVE)",
            "sam_umol_l":              "50–120 (NORMAL-LOW — not elevated, KEY NEGATIVE vs GNMT)",
            "sarcosine":               "PRESENT/NORMAL (GNMT intact — KEY NEGATIVE vs GNMT)",
            "mma":                     "NORMAL (KEY NEGATIVE)",
            "ck_u_l_mild":             "30–100",
            "ck_u_l_severe":           "80–400 (mild creatine-deficiency myopathy)",
            "brain_mrs":               "Creatine peak 3.0 ppm ABSENT — PATHOGNOMONIC",
        },
    }


def get_definitions() -> dict:
    return {
        "dashboard_id": "gamt",
        "gene_card": {
            "gene":       "GAMT",
            "full_name":  "Guanidinoacetate N-methyltransferase",
            "chromosome": "6p21.3",
            "size_aa":    236,
            "structure":  (
                "Homodimeric; class I SAM-dependent methyltransferase (seven-stranded β-sheet fold). "
                "SAM-binding pocket (Rossmann fold) + guanidinoacetate (GAA) binding site. "
                "Substrate specificity: GAA — does NOT methylate glycine (GNMT) or other small molecules. "
                "Expressed predominantly in liver and pancreas. "
                "Liver is the primary site of creatine biosynthesis step 2."
            ),
            "omim_gene":    "*601240",
            "omim_disease": "#612736",
            "inheritance":  "Autosomal Recessive",
        },
        "key_concepts": [
            {
                "term": "Dual Pathology — Absent Creatine + Massive GAA Accumulation",
                "definition": (
                    "GAMT deficiency has two simultaneous, independently epileptogenic defects: "
                    "(1) Creatine cannot be synthesised → brain phosphocreatine = zero → "
                    "ATP cannot be rapidly regenerated during synaptic burst firing → neuronal hyperexcitability. "
                    "(2) GAA accumulates to 50–300 µmol/L (normal <3) → GAA is a potent convulsant via "
                    "GABA-A inhibition + NMDA activation. "
                    "This dual mechanism explains why seizures in GAMT are MORE drug-resistant than in "
                    "any other creatine synthesis disorder — and why both creatine AND ornithine treatment "
                    "are required to achieve seizure control."
                ),
            },
            {
                "term": "GAA as Endogenous Epileptogenic Molecule",
                "definition": (
                    "Guanidinoacetate (GAA) is not merely a substrate that accumulates — "
                    "it is a potent endogenous convulsant: "
                    "(a) Inhibits GABA-A receptors → loss of inhibitory tone → cortical hyperexcitability. "
                    "(b) Activates NMDA glutamate receptors → excitotoxicity. "
                    "(c) Depletes GABA from cortical interneurons. "
                    "(d) Generates reactive oxygen species → oxidative neuronal damage. "
                    "This makes GAMT epilepsy fundamentally different from other metabolic epilepsies — "
                    "conventional AEDs cannot overcome GAA's direct proconvulsant action. "
                    "Ornithine supplementation reduces GAA → reduces epileptogenicity → "
                    "improves AED responsiveness."
                ),
            },
            {
                "term": "Creatine Biosynthesis Pathway — AGAT → GAMT → SLC6A8",
                "definition": (
                    "Step 1 — AGAT (L-arginine:glycine amidinotransferase, kidney/pancreas): "
                    "L-Arginine + Glycine → L-Ornithine + Guanidinoacetate (GAA). "
                    "Step 2 — GAMT (liver/pancreas): SAM + GAA → SAH + Creatine. "
                    "Step 3 — SLC6A8 (creatine transporter, ubiquitous): "
                    "Creatine transported into muscle and brain via Na+/Cl- dependent transporter. "
                    "All three genes (AGAT, GAMT, SLC6A8) cause cerebral creatine deficiency syndromes (CCDS): "
                    "CCDS1 = SLC6A8 (X-linked, males severe / females may have phenotype); "
                    "CCDS2 = GAMT (AR, most common, GAA HIGH — pathognomonic); "
                    "CCDS3 = AGAT (AR, rarest, GAA LOW — opposite sign from GAMT)."
                ),
            },
            {
                "term": "GAMT vs GNMT — Two SAM-Consuming Methyltransferases, Opposite Clinical Profiles",
                "definition": (
                    "GAMT and GNMT both use SAM to methylate a small molecule substrate. "
                    "GNMT: SAM + Glycine → SAH + Sarcosine (hepatic SAM overflow valve; ~50-75% hepatic SAM). "
                    "GAMT: SAM + GAA → SAH + Creatine (creatine biosynthesis; ~5-15% hepatic SAM). "
                    "KEY DIFFERENCES: "
                    "Methionine: GNMT LOF → HIGH (200–500 µmol/L); GAMT LOF → NORMAL. "
                    "SAM: GNMT → MASSIVELY HIGH (800–2200); GAMT → NORMAL-LOW. "
                    "Sarcosine: GNMT → ABSENT; GAMT → PRESENT (GNMT is intact in GAMT deficiency). "
                    "GAA: GNMT → NORMAL; GAMT → MASSIVELY ELEVATED (50–300 µmol/L). "
                    "Dominant feature: GNMT → liver disease (SAM toxicity); GAMT → epilepsy + IDD (GAA + creatine deficit). "
                    "NBS methionine: GNMT → elevated; GAMT → NORMAL."
                ),
            },
            {
                "term": "Treatment Rationale — Creatine (Replace) + Ornithine (Reduce GAA)",
                "definition": (
                    "Treatment targets both pathological components of GAMT deficiency: "
                    "(1) Creatine monohydrate (300–400 mg/kg/day, Level A): "
                    "provides exogenous creatine to bypass the enzymatic block → "
                    "brain/muscle creatine partially restored → phosphocreatine buffer partially recovered → "
                    "neuronal energy deficiency improved → seizure threshold raised. "
                    "(2) Ornithine supplementation (100–150 mg/kg/day, Level A): "
                    "ornithine competes with glycine for the AGAT active site → "
                    "less GAA produced by AGAT → GAA accumulation reduced by 60–90% → "
                    "GAA epileptogenicity reduced → AED responsiveness improved. "
                    "Combined creatine + ornithine is synergistic — both targets must be addressed. "
                    "(3) Arginine restriction (Level B): reduces AGAT substrate → less GAA."
                ),
            },
            {
                "term": "Brain H-MRS — Creatine Peak Absence (3.0 ppm)",
                "definition": (
                    "H-MRS (proton magnetic resonance spectroscopy) is the gold standard non-invasive "
                    "diagnostic tool for all three CCDS: "
                    "Normal: creatine/phosphocreatine resonance peak at 3.0 ppm. "
                    "GAMT: creatine peak ABSENT (no creatine in brain — level approaches zero). "
                    "AGAT: same finding (creatine absent, GAA absent). "
                    "SLC6A8: same finding (creatine cannot enter brain). "
                    "After creatine treatment in GAMT: peak partially recovers but rarely normalises "
                    "(brain creatine uptake continues via SLC6A8 which is intact in GAMT). "
                    "Creatine peak recovery on MRS correlates with clinical improvement."
                ),
            },
            {
                "term": "Why Methionine is NORMAL in GAMT (Unlike all Hypermethioninemia Disorders)",
                "definition": (
                    "GAMT uses SAM as a methyl donor — but GAMT is responsible for only ~5–15% "
                    "of hepatic SAM consumption (compared to GNMT's 50–75%). "
                    "When GAMT is absent, other methyltransferases (DNMT, COMT, PEMT, GNMT) "
                    "compensate for the lost SAM consumption. "
                    "Net result: SAM is NORMAL or only mildly low — NOT elevated. "
                    "Methionine remains NORMAL because there is no SAM surplus. "
                    "This is the single most important KEY NEGATIVE that distinguishes GAMT from "
                    "MAT1A, GNMT, AHCY, CBS, MTHFR — none of which cause elevated GAA or absent creatine."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disorder": "AGAT Deficiency (CCDS3 — Arginine:glycine amidinotransferase)",
                "key_distinction": (
                    "AGAT: GAA = VERY LOW (AGAT cannot make GAA — source deficient). "
                    "GAMT: GAA = MASSIVELY HIGH (AGAT makes GAA but GAMT cannot clear it). "
                    "Both: Creatine absent (same downstream consequence). "
                    "Both: Methionine normal, tHcy normal, MMA normal. "
                    "Brain H-MRS: creatine absent in both — identical MRS. "
                    "GAA direction is the single discriminating test: HIGH = GAMT; LOW = AGAT. "
                    "AGAT is rarer, typically milder (less epileptic — no GAA neurotoxicity). "
                    "Treatment: both respond to creatine; ornithine NOT needed in AGAT (no GAA)."
                ),
            },
            {
                "disorder": "SLC6A8 Deficiency (CCDS1 — Creatine Transporter Defect)",
                "key_distinction": (
                    "SLC6A8 (X-linked): creatine cannot ENTER brain/muscle (transporter absent). "
                    "GAA: NORMAL (synthesis intact — AGAT + GAMT both working). "
                    "Plasma/urine creatine: NORMAL or HIGH (creatine cannot be transported → accumulates). "
                    "Urine creatine/creatinine ratio: ELEVATED (opposite of GAMT where creatinine very low). "
                    "Brain H-MRS: creatine absent (same as GAMT — but different reason). "
                    "X-linked: affects males severely; carrier females may have mild-moderate phenotype. "
                    "Treatment: creatine supplementation is largely INEFFECTIVE (transporter absent). "
                    "Creatinine-based NBS: urine creatine HIGH in SLC6A8 vs LOW in GAMT."
                ),
            },
            {
                "disorder": "GNMT Deficiency",
                "key_distinction": (
                    "GNMT: SAM MASSIVELY HIGH (800–2200 µmol/L); methionine HIGH (40–500); "
                    "sarcosine ABSENT; GAA NORMAL; liver disease dominant (60–80%). "
                    "GAMT: SAM NORMAL-LOW; methionine NORMAL; sarcosine PRESENT; "
                    "GAA MASSIVELY HIGH (50–300); liver disease ABSENT; epilepsy dominant (60–80%). "
                    "Both use SAM as methyl donor — but completely different clinical/biochemical profiles. "
                    "The GAA/methionine pair is the fastest discriminating biochemical test."
                ),
            },
            {
                "disorder": "MAT1A / AHCY Deficiency",
                "key_distinction": (
                    "MAT1A: methionine MASSIVELY HIGH (200–2000+ µmol/L); SAM VERY LOW; breath odor. "
                    "AHCY: SAH MASSIVELY HIGH; SAM/SAH ratio <0.5; myopathy 85–90%; cardiomyopathy. "
                    "GAMT: methionine NORMAL; tHcy NORMAL; GAA MASSIVELY HIGH; creatine absent. "
                    "None of MAT1A/AHCY/CBS/MTHFR have elevated GAA or absent creatine — "
                    "GAA measurement immediately excludes the entire HHcy/hypermethioninemia spectrum."
                ),
            },
            {
                "disorder": "Methylmalonic Acidemias (cblC/MMACHC, MMUT, cblA-cblB)",
                "key_distinction": (
                    "MMA disorders: MMA MASSIVELY ELEVATED (urine 200–10,000+ mmol/mol creatinine). "
                    "GAMT: MMA NORMAL (propionate/cobalamin arm intact). "
                    "cblC: MMA + homocysteine BOTH elevated; GAA normal; NBS visible (C3 elevated). "
                    "GAMT: MMA normal; homocysteine normal; GAA elevated; NBS often missed on standard MS/MS. "
                    "MMA + elevated C3 on NBS excludes GAMT; GAA not elevated in any MMA disorder."
                ),
            },
        ],
        "treatment_summary": {
            "first_line": (
                "Creatine monohydrate 300–400 mg/kg/day (Level A) + "
                "Ornithine 100–150 mg/kg/day (Level A) + "
                "LEV (first-line AED, Level A)"
            ),
            "absolute_ci": [],   # No absolute CI in GAMT (unlike GNMT/MAT1A/AHCY)
            "high_risk": [
                "Arginine supplements — feeds AGAT → more GAA → worse epilepsy",
                "Valproate — may impair creatine kinase + carnitine depletion; prefer LEV",
                "Metformin — may impair creatine transporter (SLC6A8)",
                "Glycine supplements / excess collagen — AGAT substrate, drives GAA",
            ],
            "not_indicated": [
                "SAMe/SAM supplements — no SAM deficiency in GAMT (unlike MAT1A)",
                "Betaine — no homocysteine elevation (tHcy normal; unlike CBS/cblE/cblG)",
                "Folinic acid — no methylfolate trap (unlike MTHFR/cblE/cblG)",
            ],
        },
    }
