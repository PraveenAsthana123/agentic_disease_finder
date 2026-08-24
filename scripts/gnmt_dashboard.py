#!/usr/bin/env python3
"""GNMT (Glycine N-Methyltransferase Deficiency) Epilepsy Dashboard.

GNMT encodes Glycine N-methyltransferase, a PLP-independent, SAM-dependent
methyltransferase expressed primarily in the liver and pancreas.

GNMT ENZYMATIC FUNCTION — THE HEPATIC SAM SAFETY VALVE:

  SAM + Glycine → SAH + Sarcosine (N-methylglycine)

GNMT is the DOMINANT SAM-consuming enzyme in the liver — responsible for
consuming ~50–75% of all hepatic SAM under basal conditions. Its primary
role is NOT to produce sarcosine (sarcosine is further metabolised by
SARDH → glycine). Its REAL physiological function is to ACT AS A BUFFER:
it prevents SAM from rising to toxic concentrations in the liver.

GNMT IS UNIQUE IN METHYLTRANSFERASE BIOLOGY:
  1. Product (SAH) does NOT inhibit GNMT — unlike ALL other SAM-dependent
     methyltransferases where SAH is a potent competitive inhibitor.
     This makes GNMT the ideal SAM "overflow valve."
  2. GNMT IS DIRECTLY INHIBITED BY 5-methylTHF (the active folate):
     when folate is replete, GNMT is suppressed → SAM conserved for
     methylation. When folate is deficient, GNMT is de-repressed → SAM
     consumed more rapidly → hypomethylation.
  3. GNMT is the primary link between folate status and SAM homeostasis.

GNMT LOF — SAM CANNOT BE CONSUMED VIA THE SAFETY VALVE:

  SAM → × GNMT BLOCKED → Sarcosine ABSENT
  ↑↑↑ SAM MASSIVELY ACCUMULATES IN LIVER
  Methionine → MAT1A/MAT3 → SAM (synthesis still intact and uninhibited)
  SAM accumulation → feedback elevates plasma methionine (export + enzyme inhibition)
  SAH: LOW or NORMAL (less SAH produced because the GNMT reaction is blocked)
  SAM/SAH ratio: MARKEDLY ELEVATED (>10–25; pathognomonic; OPPOSITE of AHCY <0.5)
  tHcy: NORMAL (CBS intact; MTR/MTRR intact; transsulfuration preserved)
  Sarcosine (N-methylglycine): ABSENT / UNDETECTABLE (GNMT reaction product)
  Glycine: NORMAL or mildly elevated (GNMT is major glycine consumer; mild accumulation)
  MMA: NORMAL (propionate/CoA arm intact)
  MeCbl/AdoCbl: NORMAL (cobalamin system intact)

EXCESS SAM TOXICITY — WHY LIVER DISEASE IS PROMINENT:
  SAM at very high concentrations is directly hepatotoxic:
  (1) Non-enzymatic transmethylation of cellular proteins → adducts
  (2) Inhibition of PEMT at paradoxically high SAM (paradoxical phospholipid effect)
  (3) Reactive oxygen species from aberrant SAM oxidation
  (4) Hepatic steatosis → NASH → cirrhosis
  → Liver disease is the DOMINANT clinical feature (60–80%)

GNMT vs MAT1A — OPPOSITE SAM PHENOTYPES (BOTH CAUSE HYPERMETHIONINEMIA):
  MAT1A: SAM VERY LOW (cannot MAKE SAM) → liver disease from methylation DEFICIT
  GNMT:  SAM MARKEDLY HIGH (cannot CONSUME SAM) → liver disease from SAM EXCESS
  Both elevate methionine but via opposite mechanisms:
    MAT1A: methionine high because it cannot be converted to SAM
    GNMT:  methionine high because SAM accumulates → feed-back export + inhibits MAT1A auto-feedback
  SAM/SAH ratio: LOW in MAT1A (proportional low SAH + low SAM but ratio can be normal)
                 MARKEDLY HIGH in GNMT (SAM very high, SAH stays low → ratio >10–25)
  Betaine/SAM supplements: LEVEL A (MAT1A); ABSOLUTE CI (GNMT — already elevated)

GNMT vs AHCY (BOTH CAUSE ELEVATED SAM/METHIONINE + LIVER DISEASE):
  AHCY:  SAH MASSIVELY ELEVATED + SAM/SAH ratio VERY LOW (<0.5) + MYOPATHY 85-90%
  GNMT:  SAH LOW/NORMAL + SAM/SAH ratio MARKEDLY HIGH (>10) + NO myopathy
  AHCY has the SAM-SAH ratio inverted vs GNMT:
    AHCY: SAM high, SAH even higher → ratio crushed to <0.5
    GNMT: SAM very high, SAH normal → ratio elevated to >10–25
  Both cause liver disease, but for different reasons (PEMT impaired vs SAM toxicity)

FOLATE INTERACTION (UNIQUE TO GNMT):
  5-methylTHF normally inhibits GNMT — when GNMT is absent, this regulation is moot.
  However, low folate WORSENS GNMT deficiency phenotype because:
    folate-deficient state de-represses GNMT in heterozygotes → accelerated SAM consumption
    in homozygous GNMT-null: folate supplementation does not help (no functional enzyme)
  High-dose folinic acid NOT routinely recommended (no methylfolate trap; MTHFR intact)

INHERITANCE AND GENE:
  Gene: GNMT, 9q22.2, 296 amino acids
  Autosomal Recessive — biallelic LOF required for classic phenotype
  Fewer than 30 patients reported worldwide (2026) — ultrarare
  Heterozygotes: slightly elevated methionine/SAM on NBS; asymptomatic

OMIM:
  Gene: *606460 (GNMT)
  Disease: #606464 (Hypermethioninemia due to GNMT deficiency /
           Glycine N-methyltransferase deficiency)
"""

import random
import math

_SEED = 107
_rng = random.Random(_SEED)

# ---------------------------------------------------------------------------
# Phenotypic classes
# ---------------------------------------------------------------------------
PHENOTYPE_CLASSES = [
    "Mild-Hepatic (isolated liver disease / mildly elevated methionine + SAM / normal development / MODAL)",
    "Classic-Moderate (liver disease + IDD + moderate hypermethioninemia / SAM markedly elevated)",
    "Severe-Hepatic (liver failure / cirrhosis / marked hypermethioninemia / seizures / IDD)",
]

PHENO_WEIGHTS = [0.40, 0.45, 0.15]  # 40% mild / 45% classic / 15% severe

VARIANTS = [
    "p.Arg49Cys",     # catalytic domain; most common; ~35%; moderate phenotype
    "p.Pro249Leu",    # dimer interface disruption; severe; ~20%
    "p.Trp98Arg",     # tetramer interface; severe; ~15%
    "p.Val161Met",    # near FAD-proximate pocket; moderate; ~15%
    "p.Ala23Val",     # N-terminal; mild; ~10%
    "c.IVS3+1G>A",   # splice-null; severe; ~5%
]

VARIANT_WEIGHTS = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]

SEIZURE_TYPES = [
    "Focal (frontal lobe — SAM-mediated neuronal hyperexcitability)",
    "Generalized tonic-clonic (severe SAM toxicity / metabolic encephalopathy)",
    "Infantile spasms (severe neonatal form — extreme SAM elevation)",
    "Absence (mild hypermethioninemia-related cortical perturbation)",
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
    is_mild   = "Mild"   in pheno
    is_severe = "Severe" in pheno
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            if is_mild and v in ("p.Pro249Leu", "p.Trp98Arg", "c.IVS3+1G>A"):
                return _rng.choice(["p.Arg49Cys", "p.Val161Met", "p.Ala23Val"])
            if is_severe and v in ("p.Ala23Val",):
                return _rng.choice(["p.Pro249Leu", "p.Trp98Arg", "c.IVS3+1G>A"])
            return v
    return VARIANTS[0]


def _make_patient(idx):
    pheno       = _choose_pheno()
    is_mild     = "Mild"    in pheno
    is_classic  = "Classic" in pheno
    is_severe   = "Severe"  in pheno

    # Methionine elevation — GNMT causes hypermethioninemia (less extreme than MAT1A)
    if is_severe:
        methion    = round(_rng.uniform(180, 500),  1)
        sam_umol   = round(_rng.uniform(800, 2200), 1)   # MARKEDLY ELEVATED — pathognomonic
        sah_umol   = round(_rng.uniform(8,   30),   1)   # SAH LOW/NORMAL
        hcy_umol   = round(_rng.uniform(6,   18),   1)   # tHcy NORMAL
        ast_u_l    = round(_rng.uniform(180, 700),  0)
    elif is_classic:
        methion    = round(_rng.uniform(80,  250),  1)
        sam_umol   = round(_rng.uniform(300, 900),  1)
        sah_umol   = round(_rng.uniform(6,   25),   1)
        hcy_umol   = round(_rng.uniform(5,   16),   1)
        ast_u_l    = round(_rng.uniform(60,  250),  0)
    else:  # mild
        methion    = round(_rng.uniform(40,  120),  1)
        sam_umol   = round(_rng.uniform(120, 400),  1)
        sah_umol   = round(_rng.uniform(5,   20),   1)
        hcy_umol   = round(_rng.uniform(5,   13),   1)
        ast_u_l    = round(_rng.uniform(30,  100),  0)

    # SAM/SAH ratio — MARKEDLY ELEVATED (pathognomonic)
    sam_sah_ratio = round(sam_umol / max(sah_umol, 1.0), 1)

    # Sarcosine: ABSENT (GNMT cannot make it)
    sarcosine_absent = True
    mma_normal       = True
    mecbl_normal     = True
    adocbl_normal    = True

    # Clinical features
    liver_disease    = _rng.random() < (0.90 if is_severe else (0.75 if is_classic else 0.55))
    hepatomegaly     = _rng.random() < (0.85 if is_severe else (0.70 if is_classic else 0.45))
    idd              = _rng.random() < (0.75 if is_severe else (0.50 if is_classic else 0.10))
    seizures         = _rng.random() < (0.45 if is_severe else (0.25 if is_classic else 0.05))
    white_matter     = _rng.random() < (0.40 if is_severe else (0.22 if is_classic else 0.05))
    nbs_detected     = _rng.random() < (0.80 if is_severe else (0.60 if is_classic else 0.40))
    psychiatric      = _rng.random() < (0.25 if is_severe else (0.18 if is_classic else 0.05))
    on_met_restrict  = _rng.random() < (0.0  if is_mild   else (0.70 if is_classic else 0.90))

    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    age_onset_mo = (
        _rng.randint(1,  12) if is_severe  else
        _rng.randint(6,  48) if is_classic else
        _rng.randint(0,   0)  # mild: NBS-only or incidental
    )

    variant = _choose_variant(pheno)

    return {
        "id":                 f"GNMT-{idx:03d}",
        "phenotype":          pheno,
        "variant":            variant,
        "age_onset_months":   age_onset_mo,
        # Biomarkers
        "methionine_umol_l":  methion,
        "sam_umol_l":         sam_umol,       # MARKEDLY ELEVATED — hallmark
        "sah_umol_l":         sah_umol,       # LOW / NORMAL
        "sam_sah_ratio":      sam_sah_ratio,  # >10–25 — PATHOGNOMONIC
        "homocysteine_umol_l": hcy_umol,      # NORMAL
        "ast_u_l":            ast_u_l,
        "sarcosine_absent":   sarcosine_absent,  # ALWAYS absent — GNMT makes sarcosine
        "mma_normal":         mma_normal,
        "mecbl_normal":       mecbl_normal,
        "adocbl_normal":      adocbl_normal,
        # Clinical
        "liver_disease":      liver_disease,
        "hepatomegaly":       hepatomegaly,
        "idd":                idd,
        "seizures":           seizures,
        "seizure_type":       seizure_type,
        "white_matter":       white_matter,
        "nbs_detected":       nbs_detected,
        "psychiatric":        psychiatric,
        "myopathy":           False,   # ABSENT — distinguishes from AHCY 85-90%
        "cardiomyopathy":     False,   # ABSENT — distinguishes from AHCY 60-70%
        "ectopia_lentis":     False,   # ABSENT — distinguishes from CBS 90%
        "breath_odor":        False,   # ABSENT — distinguishes from MAT1A (DMS)
        "on_met_restriction": on_met_restrict,
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
        "dashboard_id": "gnmt",
        "title": "GNMT Epilepsy Dashboard",
        "subtitle": (
            "Glycine N-Methyltransferase Deficiency — "
            "Hypermethioninemia with SAM Accumulation / "
            "GNMT-296aa-PLP-Independent-SAM-Dependent-9q22.2-AR"
        ),
        "gene": "GNMT",
        "disease_name": (
            "Glycine N-Methyltransferase Deficiency "
            "(Hypermethioninemia due to GNMT deficiency)"
        ),
        "chromosome": "9q22.2",
        "inheritance": "Autosomal Recessive — biallelic LOF; heterozygotes asymptomatic",
        "omim_gene":    "OMIM *606460",
        "omim_disease": "OMIM #606464",
        "protein_size": (
            "296 aa; SAM-dependent methyltransferase; PLP-independent; "
            "homotetrameric (active) and homodimeric (less active); "
            "liver/pancreas predominant; "
            "major hepatic SAM 'safety valve' — consumes 50–75% of hepatic SAM; "
            "uniquely inhibited by 5-methylTHF (folate status sensor); "
            "SAH does NOT inhibit GNMT (unlike all other SAM-MT enzymes)"
        ),
        "prevalence": "Fewer than 30 patients reported worldwide (2026); ultrarare",
        "cohort_n": n,
        "phenotype_distribution": pheno_dist,
        "function": (
            "GNMT (Glycine N-methyltransferase) catalyses: SAM + Glycine → SAH + Sarcosine. "
            "Its physiological role is NOT primarily to methylate glycine — it is the hepatic SAM buffer. "
            "GNMT consumes ~50–75% of all hepatic SAM under basal conditions, "
            "preventing SAM from reaching hepatotoxic concentrations. "
            "GNMT is unique: its product SAH does NOT inhibit it (unlike ALL other SAM-MTs), "
            "making it the ideal overflow valve. "
            "It is directly inhibited by 5-methylTHF (the active folate cofactor), "
            "linking hepatic SAM homeostasis to folate status. "
            "When folate is high, GNMT is inhibited → SAM conserved. "
            "When folate is low, GNMT runs faster → SAM consumed rapidly. "
            "GNMT deficiency removes this safety valve: SAM accumulates to toxic concentrations."
        ),
        "mechanism": (
            "GNMT LOF → hepatic SAM cannot be consumed via the safety-valve reaction → "
            "SAM MASSIVELY ACCUMULATES (800–2000+ µmol/L plasma; normal <100). "
            "Methionine is elevated (60–500 µmol/L) because: "
            "(1) excess SAM exports into plasma, (2) feedback on MAT1/III activity, "
            "(3) hepatocellular dysfunction impairs methionine clearance. "
            "SAH stays LOW or NORMAL — because GNMT normally produces SAH; with GNMT absent, "
            "less SAH is generated from this reaction → SAH is paradoxically low. "
            "SAM/SAH ratio is MARKEDLY ELEVATED (>10–25; normal 2–4; pathognomonic for GNMT). "
            "This is the OPPOSITE of AHCY (where SAH is massively elevated → ratio <0.5). "
            "tHcy is NORMAL (CBS, MTR, MTRR all intact; transsulfuration pathway unaffected). "
            "Sarcosine is ABSENT or undetectable (the GNMT reaction product is not made). "
            "LIVER TOXICITY: excess SAM at high concentrations: "
            "(1) directly hepatotoxic (non-enzymatic protein transmethylation); "
            "(2) reactive oxygen species from aberrant SAM oxidation; "
            "(3) steatosis → NASH → cirrhosis. "
            "Liver disease is the DOMINANT clinical feature (60–80%), unlike CBS/MTHFR "
            "where liver disease is absent."
        ),
        "key_positive_features": (
            "KEY POSITIVES (unique to GNMT among HHcy/hypermethioninemia disorders): "
            "(1) SAM MARKEDLY ELEVATED (800–2000+ µmol/L) — pathognomonic; "
            "    highest SAM of ALL inherited metabolic disorders. "
            "(2) SAM/SAH ratio MARKEDLY ELEVATED (>10–25) — "
            "    OPPOSITE of AHCY where ratio is crushed to <0.5. "
            "(3) Sarcosine (N-methylglycine) ABSENT — GNMT reaction product not made. "
            "(4) tHcy NORMAL (<20 µmol/L) — CBS/MTR/MTRR all intact. "
            "(5) Liver disease PROMINENT (60–80%) due to SAM toxicity. "
            "(6) SAH LOW or NORMAL (less SAH produced because GNMT reaction blocked). "
            "KEY NEGATIVES (absent in GNMT): "
            "(1) Myopathy ABSENT (unlike AHCY 85–90%). "
            "(2) Cardiomyopathy ABSENT (unlike AHCY 60–70%). "
            "(3) Ectopia lentis ABSENT (unlike CBS 90%). "
            "(4) Breath odor ABSENT (unlike MAT1A — no DMS/garlic odor). "
            "(5) MMA NORMAL (propionate arm intact). "
            "(6) Megaloblastic anemia ABSENT (folate/cobalamin system intact). "
            "(7) Methylfolate trap ABSENT (MTHFR/MTR intact; serum folate normal)."
        ),
        "nbs_primary": (
            "Amino acid panel — methionine elevation (40–500 µmol/L; normal <60). "
            "~50–60% of GNMT cases detected on NBS. "
            "Methionine elevation is LESS EXTREME than MAT1A (200–2000+ µmol/L). "
            "NBS-invisible in ~40–50% of mild cases (methionine borderline). "
            "GNMT vs MAT1A on NBS: both cause methionine elevation, but "
            "GNMT has lower methionine + SAM elevated (opposite of MAT1A SAM LOW). "
            "Second-tier: SAM/SAH ratio by HPLC — ratio >10 distinguishes GNMT from MAT1A."
        ),
        "nbs_secondary": (
            "SAM/SAH ratio by HPLC: MARKEDLY ELEVATED (>10–25; pathognomonic). "
            "SAM: markedly elevated (800–2000+ µmol/L; normal <100). "
            "SAH: LOW or NORMAL (5–30 µmol/L; paradoxically low despite SAM elevation). "
            "Plasma sarcosine (N-methylglycine): ABSENT or undetectable. "
            "Plasma total homocysteine: NORMAL (<20 µmol/L). "
            "Urine organic acids: MMA NORMAL; sarcosine may appear if residual GNMT activity. "
            "Glycine: normal or mildly elevated. "
            "Serum B12/folate: NORMAL (cobalamin/folate system intact). "
            "Liver function: elevated AST/ALT/GGT (60–80%). "
            "Liver biopsy: steatosis, periportal inflammation, progression to NASH/cirrhosis. "
            "Brain MRI: white matter changes in ~25% (SAM-mediated). "
            "GNMT enzyme activity in liver: absent or severely reduced. "
            "Genetic confirmation: GNMT sequencing."
        ),
        "kpis": {
            "avg_methionine_umol_l":   avg("methionine_umol_l"),
            "avg_sam_umol_l":          avg("sam_umol_l"),
            "avg_sah_umol_l":          avg("sah_umol_l"),
            "avg_sam_sah_ratio":       avg("sam_sah_ratio"),
            "avg_homocysteine_umol_l": avg("homocysteine_umol_l"),
            "avg_ast_u_l":             avg("ast_u_l"),
            "pct_liver_disease":       pct(lambda p: p["liver_disease"]),
            "pct_hepatomegaly":        pct(lambda p: p["hepatomegaly"]),
            "pct_idd":                 pct(lambda p: p["idd"]),
            "pct_seizures":            pct(lambda p: p["seizures"]),
            "pct_nbs_detected":        pct(lambda p: p["nbs_detected"]),
            "pct_myopathy":            0,
            "pct_cardiomyopathy":      0,
            "pct_sarcosine_absent":    100,   # ALL patients — GNMT cannot make sarcosine
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Focal (frontal lobe — SAM-mediated neuronal hyperexcitability)", "pct": 47},
        {"type": "Generalized tonic-clonic (severe SAM toxicity / metabolic encephalopathy)", "pct": 31},
        {"type": "Infantile spasms (severe neonatal form — extreme SAM elevation)", "pct": 15},
        {"type": "Absence (mild hypermethioninemia-related cortical perturbation)", "pct": 7},
    ]

    metabolic_triggers = [
        {
            "trigger": "High-methionine diet (meat, dairy, eggs without protein restriction)",
            "pct": 92,
            "mechanism": (
                "Dietary methionine absorbed → MAT1A converts to SAM → SAM CANNOT be consumed "
                "by GNMT (absent) → SAM accumulates further. Even normal protein intake "
                "can cause dangerous SAM spikes. Methionine restriction reduces SAM burden."
            ),
        },
        {
            "trigger": "SAM supplements (SAMe) — ABSOLUTELY CONTRAINDICATED",
            "pct": 90,
            "mechanism": (
                "Exogenous SAM directly increases the already-elevated hepatic SAM pool. "
                "In GNMT deficiency, there is no GNMT safety valve to consume the excess. "
                "SAM supplementation → catastrophic worsening of SAM toxicity → acute liver crisis. "
                "ABSOLUTE CONTRAINDICATION — opposite of MAT1A where SAM is Level A treatment."
            ),
        },
        {
            "trigger": "Betaine (TMG) supplementation — ABSOLUTELY CONTRAINDICATED",
            "pct": 88,
            "mechanism": (
                "BHMT: Hcy + Betaine → Methionine. Even though tHcy is normal in GNMT, "
                "betaine-driven methionine synthesis feeds methionine → SAM via MAT1A → "
                "further SAM accumulation in a system already lacking the SAM buffer. "
                "ABSOLUTE CONTRAINDICATION — same as SAM supplements."
            ),
        },
        {
            "trigger": "Protein catabolism / catabolic illness / fasting",
            "pct": 68,
            "mechanism": (
                "Protein breakdown releases methionine → converted to SAM → cannot be "
                "buffered → SAM spike. Emergency: IV dextrose to suppress catabolism; "
                "avoid fasting; maintain caloric intake during illness."
            ),
        },
        {
            "trigger": "Methionine-containing parenteral nutrition",
            "pct": 55,
            "mechanism": (
                "Standard TPN/PN solutions contain methionine → uncontrolled SAM load. "
                "Low-methionine PN formulas required for GNMT patients."
            ),
        },
        {
            "trigger": "Hepatotoxic medications (VPA, statins, methotrexate)",
            "pct": 48,
            "mechanism": (
                "Pre-existing liver disease (SAM toxicity) dramatically lowers threshold "
                "for drug-induced hepatotoxicity. VPA is HIGH RISK — pre-existing NASH/cirrhosis "
                "plus VPA's own mitochondrial/carnitine-depletion mechanism = catastrophic. "
                "Methotrexate ABSOLUTE CI (hepatic folate antagonism + SAM perturbation)."
            ),
        },
    ]

    treatments = [
        {
            "name": "Methionine Restriction",
            "level": "Level A",
            "note": "Primary treatment — reduces dietary methionine input to lower SAM burden. "
                    "Low-methionine formula + natural protein limit. Cornerstone in Classic/Severe. "
                    "Mild patients: moderate protein moderation may suffice.",
        },
        {
            "name": "LEV (Levetiracetam)",
            "level": "Level A — First-Line AED",
            "note": "No hepatotoxicity; no interaction with SAM metabolism; no folate depletion. "
                    "First-line for all seizure types in GNMT deficiency.",
        },
        {
            "name": "Liver Transplant",
            "level": "Level B — Severe",
            "note": "Orthotopic liver transplant corrects GNMT deficiency (GNMT is liver-expressed). "
                    "Normalizes SAM/methionine post-transplant. Considered in progressive cirrhosis "
                    "or hepatic failure not responding to methionine restriction.",
        },
        {
            "name": "Adenosine supplementation",
            "level": "Level C — Experimental",
            "note": "Provides adenosine as SAH precursor to modulate SAM/SAH ratio; "
                    "theoretical — no clinical trial data; case report basis only.",
        },
        {
            "name": "Choline (for PEMT support)",
            "level": "Level B",
            "note": "PEMT (phosphatidylethanolamine N-methyltransferase) uses SAM; "
                    "paradoxically, in GNMT deficiency excess SAM may impair membrane homeostasis. "
                    "Choline supplementation supports phosphatidylcholine from the PEMT-independent route.",
        },
        {
            "name": "ACTH / Vigabatrin",
            "level": "Level A — Infantile Spasms",
            "note": "Standard IS protocol for GNMT with neonatal/infantile spasms (West syndrome). "
                    "ACTH preferred; vigabatrin second-line (monitor liver).",
        },
    ]

    drug_risks = [
        {
            "drug": "SAMe (S-Adenosylmethionine)",
            "risk": "ABSOLUTE CONTRAINDICATION",
            "mechanism": "Directly elevates already-toxic SAM — no GNMT safety valve → acute liver failure.",
        },
        {
            "drug": "Betaine (TMG)",
            "risk": "ABSOLUTE CONTRAINDICATION",
            "mechanism": "BHMT → more methionine → more SAM → SAM crisis. No benefit (tHcy already normal).",
        },
        {
            "drug": "Valproate (VPA)",
            "risk": "HIGH RISK — Pre-existing Liver Disease",
            "mechanism": "VPA hepatotoxicity + carnitine depletion on top of existing GNMT liver disease "
                         "(steatosis/NASH/cirrhosis) → catastrophic. LEV is first-line alternative.",
        },
        {
            "drug": "Methotrexate",
            "risk": "HIGH RISK — Hepatic + SAM Perturbation",
            "mechanism": "Hepatotoxic in pre-existing liver disease; DHFR block depletes THF, "
                         "perturbing SAM/folate interplay uniquely in GNMT deficiency.",
        },
        {
            "drug": "Methionine-containing supplements / high-protein products",
            "risk": "HIGH RISK",
            "mechanism": "Direct methionine load → SAM surge. Avoid methionine-enriched sports "
                         "nutrition, whey, casein supplements. Use low-methionine formula.",
        },
        {
            "drug": "Statins (high-dose)",
            "risk": "MODERATE RISK",
            "mechanism": "Statin hepatotoxicity risk magnified in pre-existing GNMT liver disease. "
                         "Use lowest effective dose with close LFT monitoring.",
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
            "methionine":      p["methionine_umol_l"],
            "sam":             p["sam_umol_l"],
            "sam_sah_ratio":   p["sam_sah_ratio"],
            "sah":             p["sah_umol_l"],
            "homocysteine":    p["homocysteine_umol_l"],
            "ast":             p["ast_u_l"],
            "liver_disease":   p["liver_disease"],
            "seizures":        p["seizures"],
            "nbs":             p["nbs_detected"],
        }
        for p in _PATIENTS[:12]
    ]

    return {
        "dashboard_id": "gnmt",
        "cohort_n": n,
        "kpi_pcts": {
            "pct_liver_disease":  pct(lambda p: p["liver_disease"]),
            "pct_hepatomegaly":   pct(lambda p: p["hepatomegaly"]),
            "pct_idd":            pct(lambda p: p["idd"]),
            "pct_seizures":       pct(lambda p: p["seizures"]),
            "pct_white_matter":   pct(lambda p: p["white_matter"]),
            "pct_nbs":            pct(lambda p: p["nbs_detected"]),
            "pct_psychiatric":    pct(lambda p: p["psychiatric"]),
            "pct_myopathy":       0,
            "pct_cardiomyopathy": 0,
        },
        "phenotype_dist": [
            {"label": "Mild-Hepatic", "pct": 40},
            {"label": "Classic-Moderate", "pct": 45},
            {"label": "Severe-Hepatic", "pct": 15},
        ],
        "seizure_types":     seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "treatments":        treatments,
        "drug_risks":        drug_risks,
        "variants":          variants_list,
        "patient_sample":    sample,
        "biomarker_ranges": {
            "methionine_mild_umol_l":   "40–120",
            "methionine_classic_umol_l": "80–250",
            "methionine_severe_umol_l": "180–500",
            "sam_mild_umol_l":          "120–400",
            "sam_classic_umol_l":       "300–900",
            "sam_severe_umol_l":        "800–2200",
            "sah_umol_l":               "5–30 (LOW; normal ~15–40)",
            "sam_sah_ratio":            ">10–25 (pathognomonic; normal 2–4)",
            "homocysteine_umol_l":      "5–18 (NORMAL; distinguishes from CBS/cblE/cblG)",
            "sarcosine":                "ABSENT (undetectable — GNMT product not made)",
            "mma":                      "NORMAL (propionate arm intact)",
            "ast_u_l_mild":             "30–100",
            "ast_u_l_classic":          "60–250",
            "ast_u_l_severe":           "180–700",
        },
    }


def get_definitions() -> dict:
    return {
        "dashboard_id": "gnmt",
        "gene_card": {
            "gene":       "GNMT",
            "full_name":  "Glycine N-methyltransferase",
            "chromosome": "9q22.2",
            "size_aa":    296,
            "structure":  (
                "Homotetrameric (active form) and homodimeric. "
                "Each subunit has a SAM-binding pocket and a glycine-binding site. "
                "Unique: SAH does NOT inhibit GNMT (unlike all other SAM-MTs). "
                "Inhibited by 5-methylTHF (folate status sensor). "
                "Predominant expression: liver >> pancreas."
            ),
            "omim_gene":    "*606460",
            "omim_disease": "#606464",
            "inheritance":  "Autosomal Recessive",
        },
        "key_concepts": [
            {
                "term": "GNMT as Hepatic SAM Safety Valve",
                "definition": (
                    "GNMT consumes 50–75% of all hepatic SAM under normal conditions. "
                    "Unlike other methyltransferases, its product SAH does not inhibit it — "
                    "making it the ideal buffer enzyme. Its REAL function is not to produce "
                    "sarcosine (which is recycled back to glycine by SARDH), but to prevent "
                    "SAM from accumulating to hepatotoxic concentrations."
                ),
            },
            {
                "term": "SAM/SAH Ratio — Inverse Signature vs AHCY",
                "definition": (
                    "AHCY deficiency: SAH massive ↑ → SAM/SAH ratio severely LOW (<0.5). "
                    "GNMT deficiency: SAM massive ↑, SAH low → SAM/SAH ratio MARKEDLY HIGH (>10–25). "
                    "These are mirror-image disorders with opposite SAM/SAH ratio signatures — "
                    "the ratio is the single most discriminating test between them."
                ),
            },
            {
                "term": "Sarcosine Absence — Pathognomonic Metabolomic Marker",
                "definition": (
                    "GNMT catalyses: SAM + Glycine → SAH + Sarcosine. "
                    "In GNMT deficiency, sarcosine (N-methylglycine) is absent from plasma and urine. "
                    "Sarcosine is normally present in small amounts from GNMT activity. "
                    "Its absence, combined with elevated SAM, is pathognomonic for GNMT deficiency. "
                    "(Note: sarcosine elevation occurs in sarcosine dehydrogenase deficiency — opposite sign.)"
                ),
            },
            {
                "term": "5-methylTHF as GNMT Inhibitor — Folate-SAM Homeostasis Link",
                "definition": (
                    "GNMT is directly inhibited by 5-methylTHF (the active form of folate). "
                    "This creates a feedback loop: "
                    "high folate → high 5-methylTHF → GNMT inhibited → SAM conserved → more methylation. "
                    "Low folate → GNMT de-repressed → SAM consumed faster → less methylation. "
                    "In GNMT deficiency, this regulatory axis is broken, "
                    "disconnecting folate status from SAM homeostasis."
                ),
            },
            {
                "term": "SAM Toxicity Mechanism (vs MAT1A SAM Deficiency)",
                "definition": (
                    "MAT1A deficiency: SAM LOW → methylation starved → liver disease from DEFICIT. "
                    "GNMT deficiency: SAM HIGH → methylation overwhelmed → liver disease from EXCESS. "
                    "High SAM is hepatotoxic via: "
                    "(1) non-enzymatic transmethylation of cellular proteins (adducts), "
                    "(2) aberrant SAM oxidation → reactive oxygen species, "
                    "(3) steatosis → NASH → fibrosis → cirrhosis. "
                    "Treatment: reduce SAM load (methionine restriction); "
                    "avoid SAM supplements (absolutely contraindicated — opposite of MAT1A)."
                ),
            },
            {
                "term": "Why tHcy is NORMAL in GNMT (Unlike CBS/cblE/cblG/MTHFR)",
                "definition": (
                    "GNMT deficiency does NOT affect the homocysteine remethylation pathway: "
                    "MTR (methionine synthase) is intact → Hcy → Methionine preserved. "
                    "MTRR is intact → MTR reactivation preserved. "
                    "CBS is intact → Hcy transsulfuration preserved. "
                    "Result: tHcy is NORMAL (<20 µmol/L) in GNMT deficiency. "
                    "This is a KEY NEGATIVE distinguishing GNMT from CBS (tHcy 100–500), "
                    "cblE/cblG (tHcy 40–200), MTHFR (tHcy 50–300)."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disorder": "MAT1A Deficiency",
                "key_distinction": (
                    "MAT1A: SAM VERY LOW (<50 µmol/L) — cannot MAKE SAM. "
                    "GNMT: SAM VERY HIGH (800–2000+) — cannot CONSUME SAM. "
                    "Both elevate methionine but via OPPOSITE mechanisms. "
                    "Breath odor (DMS/garlic): PRESENT in MAT1A, ABSENT in GNMT. "
                    "SAM supplementation: LEVEL A in MAT1A; ABSOLUTE CI in GNMT. "
                    "Betaine: ABSOLUTE CI in both."
                ),
            },
            {
                "disorder": "AHCY Deficiency",
                "key_distinction": (
                    "AHCY: SAH MASSIVELY ELEVATED + SAM/SAH ratio VERY LOW (<0.5) + MYOPATHY 85–90%. "
                    "GNMT: SAH LOW + SAM/SAH ratio VERY HIGH (>10–25) + NO myopathy. "
                    "Mirror-image SAM/SAH ratio is the single best discriminating test. "
                    "AHCY: cardiomyopathy 60–70%; GNMT: cardiomyopathy absent. "
                    "Both cause liver disease but from different mechanisms."
                ),
            },
            {
                "disorder": "CBS Deficiency (Classical Homocystinuria)",
                "key_distinction": (
                    "CBS: tHcy 100–500 (HIGHEST) + ectopia lentis 90% + marfanoid habitus + thromboembolism. "
                    "GNMT: tHcy NORMAL (<20) + no ectopia lentis + no thromboembolism. "
                    "CBS: methionine elevated via blocked transsulfuration. "
                    "GNMT: methionine elevated via SAM accumulation. "
                    "SAM: normal in CBS; massively elevated in GNMT."
                ),
            },
            {
                "disorder": "Isolated hypermethioninemia (BHMT, citrin, fumarylacetoacetate hydrolase)",
                "key_distinction": (
                    "BHMT deficiency (rare): elevated methionine + SAM but different enzyme defect. "
                    "Citrin deficiency: hypermethioninemia + neonatal intrahepatic cholestasis (NICCD); "
                    "citrulline elevated; GNMT normal. "
                    "FAH (tyrosinemia type I): succinylacetone elevated; GNMT normal. "
                    "GNMT confirmed by: sarcosine absent + SAM markedly elevated + "
                    "SAM/SAH ratio >10 + genetic GNMT sequencing."
                ),
            },
            {
                "disorder": "Sarcosine dehydrogenase (SARDH) deficiency — Sarcosinemia",
                "key_distinction": (
                    "SARDH deficiency causes ELEVATED sarcosine (cannot clear it). "
                    "GNMT deficiency causes ABSENT sarcosine (cannot make it). "
                    "Opposite sarcosine findings on metabolomics. "
                    "SARDH: methionine/SAM normal; GNMT: methionine/SAM elevated."
                ),
            },
        ],
        "treatment_summary": {
            "first_line": "Methionine restriction (Level A) + LEV (AED Level A)",
            "absolute_ci": [
                "SAMe/SAM supplements — directly worsens SAM toxicity",
                "Betaine (TMG) — raises methionine → more SAM",
            ],
            "high_risk": [
                "Valproate — hepatotoxicity on pre-existing liver disease",
                "Methotrexate — hepatotoxic + SAM/folate perturbation",
                "High-protein / methionine-enriched supplements",
            ],
            "liver_transplant": (
                "Indicated for progressive cirrhosis / hepatic failure. "
                "Normalizes SAM metabolism post-transplant (GNMT is liver-specific)."
            ),
        },
    }
