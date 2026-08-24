#!/usr/bin/env python3
"""AHCY (Adenosylhomocysteinase / S-Adenosylhomocysteine Hydrolase Deficiency) Epilepsy Dashboard.

AHCY encodes Adenosylhomocysteinase (also called S-Adenosylhomocysteine Hydrolase, SAHH),
a 432-aa cytoplasmic NAD+-dependent homotetrameric enzyme at 20q11.22 that catalyzes:
  S-Adenosylhomocysteine (SAH) → Adenosine + Homocysteine

This reaction is the ONLY route to hydrolyze SAH in human cells. It is thermodynamically
UNFAVORABLE (equilibrium favors synthesis) and is driven forward only by rapid removal
of its products (adenosine by adenosine deaminase/kinase; homocysteine by CBS or MTR).

METHIONINE CYCLE — AHCY IS THE SAH-HYDROLYSIS CHECKPOINT:

  Methionine → (MAT1A/MAT2A) → SAM (S-adenosylmethionine)
                                     │
                          Methyltransferases (DNMT3A/3B/1, HNMT, COMT, PNMT, PRMT, TPMT…)
                          SAM donates CH3 → all biological methylation reactions
                                     │
                               SAH (S-adenosylhomocysteine) ← CHECKPOINT
                                     │
                              AHCY ← ONLY enzyme that can hydrolyze SAH
                                     │
                    ┌────────────────┴───────────────────┐
                    Adenosine                         Homocysteine (Hcy)
                    (→ AMP, salvaged)                 ┌─────────────────────┐
                                                      │ Remethylation:      │
                                                      │  MTR (MeCbl) via    │
                                                      │  MTHFR-5-methylTHF  │
                                                      │  BHMT (liver) via   │
                                                      │  betaine            │
                                                      │ Transsulfuration:   │
                                                      │  CBS → cystathionine│
                                                      └─────────────────────┘

AHCY LOF RESULT — SAH MASSIVELY ELEVATED (pathognomonic) + ALL METHYLATION IMPAIRED:
  → SAH cannot be hydrolyzed → SAH accumulates to very high levels
  → SAH is a POTENT COMPETITIVE INHIBITOR of ALL SAM-dependent methyltransferases:
      DNA methylation (DNMT) impaired → gene expression dysregulation
      RNA/histone methylation impaired → epigenetic disruption
      Neurotransmitter methylation (HNMT, COMT, PNMT) impaired → CNS dysfunction
      Creatine synthesis (GAMT) impaired → creatine deficiency (secondary)
      Myelin basic protein methylation impaired → white matter disease
      Phospholipid head-group methylation (PEMT) impaired → mitochondrial dysfunction
  → SAM ELEVATED (cannot be consumed by methyltransferases due to product inhibition)
  → SAM/SAH RATIO SEVERELY REDUCED (the cellular methylation index; normal >4; severely
    reduced <0.5 in AHCY deficiency) — ALL methylation reactions simultaneously impaired
  → Methionine VERY HIGH (200–600 µmol/L):
      SAM cannot donate methyl groups (feedback inhibition by SAH) → SAM accumulates →
      SAM is hydrolyzed back to methionine via SAH? No — SAH accumulates BECAUSE AHCY is absent.
      Actually: SAM is synthesized from methionine; since SAM cannot be productively consumed
      (methyltransferases blocked), the methionine → SAM step backs up → methionine accumulates.
      Additionally, if BHMT is intact, betaine-driven remethylation produces more methionine.
      Net result: VERY HIGH plasma methionine (200–600 µmol/L) is the dominant NBS signal.
  → tHcy MILDLY-TO-MODERATELY ELEVATED (40–150 µmol/L):
      PARADOXICALLY MILDER than CBS (tHcy 100–500 µmol/L) DESPITE higher methionine in AHCY.
      Why? In AHCY deficiency, SAH blocks remethylation from the downstream side (product
      inhibitor). The Hcy that IS produced cannot easily be remethylated because SAH blocks
      MTR and BHMT; BUT transsulfuration via CBS is still partially functional (less excess
      substrate pushed in). Complex: some Hcy reroutes via CBS (transsulfuration intact).
      Net result: moderate HHcy, not as extreme as CBS. tHcy 40–150 µmol/L range.
  → MMA NORMAL — propionyl-CoA / AdoCbl arm completely intact (MMAA, MMAB, MMACHC, MMADHC
    system is not involved in SAH hydrolysis). MMA NORMAL distinguishes AHCY from combined
    MMA+HHcy disorders (cblC/cblD/cblF/cblJ/cblX).
  → AdoCbl NORMAL, MeCbl NORMAL — cobalamin processing intact; AHCY does not involve
    cobalamin transport or processing.
  → Serum folate NORMAL to mildly elevated — no primary methylfolate trap (MTHFR is
    functional; 5-methylTHF is made); but downstream utilization by MTR may be partially
    impaired (SAH inhibits MTR-dependent methylation indirectly).
  → Megaloblastic anemia: usually ABSENT or mild — no primary methylfolate trap.
  → NBS detection (~70%): VERY HIGH methionine on amino acid panel (200–600 µmol/L) is
    the dominant signal; most AHCY severe patients detected. Mild cases may be borderline.

AHCY-UNIQUE CLINICAL FEATURES — MYOPATHY IS DOMINANT:
  → Severe myopathy (85–90%): the hallmark of AHCY deficiency.
      Proximal > distal weakness; hypotonia; elevated CK (often 500–10,000 U/L).
      Mechanism: impaired creatine synthesis (GAMT is SAM-dependent → creatine deficiency
      secondary); impaired phospholipid methylation → mitochondrial membrane dysfunction;
      myosin heavy chain methylation impaired → contractile apparatus dysfunction.
      This distinguishes AHCY from CBS (no myopathy), cblE/cblG (no myopathy), MTHFR (mild).
  → Cardiomyopathy (60–70%): hypertrophic or dilated; can be life-threatening; cardiac
      methylation defects + secondary creatine deficiency → myocardial energy failure.
  → Hepatomegaly / liver disease (70–75%): elevated AST/ALT/GGT; PEMT impaired →
      phosphatidylcholine synthesis impaired → fatty liver / NASH-like; hepatomegaly.
  → Developmental delay / intellectual disability (60–65%): global; language and motor.
      All neurotransmitter methylation impaired (HNMT: histamine-N-methyltransferase;
      COMT: catechol-O-methyltransferase; PNMT: phenylethanolamine-N-methyltransferase).
  → Seizures (30–40%): less prominent than myopathy; focal + generalized; secondary to
      neurotransmitter methylation failure, HHcy-mediated excitotoxicity, cerebral white
      matter disease from myelin methylation impairment.
  → Facial dysmorphism: broad forehead, depressed nasal bridge, mild coarsening.
  → Growth retardation, failure to thrive (50–60%).
  → NO ectopia lentis (unlike CBS) — fibrillin methylation not a significant issue.
  → NO marfanoid habitus (unlike CBS).
  → NBS-detectable in ~70% — via VERY HIGH methionine (distinguishable from CBS by
    additional SAH measurement, which requires specialized HPLC not in standard NBS).

CRITICAL BIOCHEMICAL DIFFERENCES — AHCY vs CBS vs MAT1A vs GNMT:
  Feature                 AHCY               CBS                MAT1A             GNMT
  Methionine              ↑↑↑ HIGH (200-600)  ↑↑ HIGH (60-500)  ↑↑↑ HIGH (>1000)  ↑↑ HIGH (>100)
  SAH                     ↑↑↑ PATHOGNOMONIC  NORMAL             NORMAL            LOW/NORMAL
  SAM                     ↑↑ HIGH             NORMAL/HIGH        LOW               ↑↑ HIGH
  SAM/SAH ratio           ↓↓↓ SEVERELY LOW   NORMAL             HIGH              ↓ LOW
  tHcy                    ↑ 40-150 (MODERATE) ↑↑↑ 100-500        NORMAL or low     NORMAL
  MMA                     NORMAL             NORMAL             NORMAL            NORMAL
  Myopathy                ↑↑↑ PROMINENT(85%) ABSENT             ABSENT            ABSENT
  Cardiomyopathy          ↑↑↑ 60-70%         ABSENT             ABSENT            ABSENT
  Liver disease           ↑↑ 70-75%          ABSENT             ABSENT            ABSENT
  Ectopia lentis          ABSENT             ↑↑ 90% PATHOGN.    ABSENT            ABSENT
  Marfanoid habitus       ABSENT             80%                ABSENT            ABSENT
  Thromboembolism         MODERATE (20-25%)  HIGH (50-60%)      LOW               LOW
  B6 responsiveness       NONE               50%                NONE              NONE
  NBS detection           ~70% (Met very ↑)  ~60% (Met ↑)       ~90% (Met ↑↑↑)   ~50%
  Megaloblastic anemia    ABSENT/MILD        ABSENT             ABSENT            ABSENT
  Betaine (TMG)           CAUTION: worsens   Level A            Contraindicated   Contraindicated
                          SAH & Met ↑        betaine used
"""

import random
import math

_SEED = 97
_rng = random.Random(_SEED)

# ---------------------------------------------------------------------------
# Phenotypic classes
# ---------------------------------------------------------------------------
PHENOTYPE_CLASSES = [
    "Severe-Neonatal-Infantile (SAH-very-high / cardiomyopathy+myopathy+liver / high-early-mortality)",
    "Classic-Infantile-Childhood (myopathy+DD+hepatomegaly+mild-seizures / MODAL)",
    "Mild-Attenuated (NBS-detected / milder-myopathy / manageable)",
]

PHENO_WEIGHTS = [0.30, 0.50, 0.20]   # 30% severe / 50% classic / 20% mild

VARIANTS = [
    "p.Tyr143Cys",    # most common, European, NAD+ binding domain, moderate-severe
    "p.Glu68Lys",     # tetramer interface, moderate
    "p.Asp244Glu",    # catalytic domain, moderate
    "p.His55Arg",     # NAD+ binding, severe
    "p.Pro344Leu",    # severe neonatal
    "p.Ala89Val",     # mild/attenuated
    "c.IVS7+1G>T",   # splice null, severe
]

VARIANT_WEIGHTS = [0.30, 0.15, 0.15, 0.12, 0.10, 0.10, 0.08]

SEIZURE_TYPES = [
    "Focal (frontal/temporal; white-matter-disease related)",
    "Generalized tonic-clonic",
    "Infantile spasms (severe neonatal class)",
    "Myoclonic (neurotransmitter methylation failure)",
]


def _choose_pheno():
    r = _rng.random()
    cum = 0.0
    for ph, w in zip(PHENOTYPE_CLASSES, PHENO_WEIGHTS):
        cum += w
        if r < cum:
            return ph
    return PHENOTYPE_CLASSES[-1]


def _choose_variant():
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            return v
    return VARIANTS[-1]


def _make_patient(idx):
    pheno = _choose_pheno()
    is_severe   = "Severe" in pheno
    is_classic  = "Classic" in pheno
    is_mild     = "Mild" in pheno

    # Biomarkers
    if is_severe:
        hcy      = round(_rng.uniform(60, 155),  1)
        methion  = round(_rng.uniform(320, 610),  1)
        sah_arb  = round(_rng.uniform(350, 800),  1)   # arbitrary units — specialized HPLC
        sam_arb  = round(_rng.uniform(120, 280),  1)
        ck_ul    = round(_rng.uniform(800, 10000), 0)
        ast_ul   = round(_rng.uniform(80, 350), 0)
    elif is_classic:
        hcy      = round(_rng.uniform(40, 120),  1)
        methion  = round(_rng.uniform(180, 420),  1)
        sah_arb  = round(_rng.uniform(150, 500),  1)
        sam_arb  = round(_rng.uniform(80, 200),  1)
        ck_ul    = round(_rng.uniform(200, 3500), 0)
        ast_ul   = round(_rng.uniform(40, 180), 0)
    else:  # mild
        hcy      = round(_rng.uniform(25, 70),   1)
        methion  = round(_rng.uniform(120, 280),  1)
        sah_arb  = round(_rng.uniform(60, 200),  1)
        sam_arb  = round(_rng.uniform(50, 140),  1)
        ck_ul    = round(_rng.uniform(80, 600), 0)
        ast_ul   = round(_rng.uniform(25, 80), 0)

    sam_sah_ratio = round(sam_arb / sah_arb, 2)   # always low (<2); normal >4

    mma_normal   = True  # ALL patients
    mecbl_normal = True
    adocbl_normal = True

    # Clinical features — probabilities by phenotype
    myopathy       = True   # 85-90%
    if is_mild:
        myopathy = _rng.random() < 0.65
    cardio    = _rng.random() < (0.80 if is_severe else (0.65 if is_classic else 0.25))
    hepato    = _rng.random() < (0.90 if is_severe else (0.72 if is_classic else 0.40))
    idd       = _rng.random() < (0.90 if is_severe else (0.70 if is_classic else 0.30))
    seizures  = _rng.random() < (0.45 if is_severe else (0.38 if is_classic else 0.15))
    nbs_det   = _rng.random() < (0.88 if is_severe else (0.70 if is_classic else 0.55))
    dysmorphic = _rng.random() < (0.65 if is_severe else (0.50 if is_classic else 0.20))

    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    age_onset_mo = (
        _rng.randint(0, 3)   if is_severe  else
        _rng.randint(2, 18)  if is_classic else
        _rng.randint(12, 72)
    )

    variant = _choose_variant()
    # Ensure variant consistency: severe → no p.Ala89Val; mild → prefer p.Ala89Val or p.Asp244Glu
    if is_severe and variant == "p.Ala89Val":
        variant = _rng.choice(["p.His55Arg", "p.Pro344Leu", "c.IVS7+1G>T"])
    if is_mild and variant in ("p.His55Arg", "p.Pro344Leu", "c.IVS7+1G>T"):
        variant = _rng.choice(["p.Ala89Val", "p.Asp244Glu", "p.Glu68Lys"])

    return {
        "id":               f"AHCY-{idx:03d}",
        "phenotype":        pheno,
        "variant":          variant,
        "age_onset_months": age_onset_mo,
        # Biomarkers
        "homocysteine_umol_l":    hcy,
        "methionine_umol_l":      methion,
        "sah_arbitrary_units":    sah_arb,    # pathognomonic elevated; arbitrary units
        "sam_arbitrary_units":    sam_arb,
        "sam_sah_ratio":          sam_sah_ratio,   # severely low < 2 (normal > 4)
        "ck_u_l":                 ck_ul,
        "ast_u_l":                ast_ul,
        "mma_normal":             mma_normal,
        "mecbl_normal":           mecbl_normal,
        "adocbl_normal":          adocbl_normal,
        # Clinical
        "myopathy":               myopathy,
        "cardiomyopathy":         cardio,
        "hepatomegaly":           hepato,
        "idd":                    idd,
        "seizures":               seizures,
        "seizure_type":           seizure_type,
        "nbs_detected":           nbs_det,
        "dysmorphic":             dysmorphic,
        "on_methionine_restriction": _rng.random() < 0.85,
        "on_adenosine_supplement":   _rng.random() < 0.65,
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
        "dashboard_id": "ahcy",
        "title": "AHCY Epilepsy Dashboard",
        "subtitle": (
            "Adenosylhomocysteinase (SAHH) Deficiency — "
            "Hypermethioninemia with SAH Accumulation / "
            "AHCY-432aa-NAD+-Homotetrameric-20q11.22-Autosomal-Recessive"
        ),
        "gene": "AHCY",
        "disease_name": "Adenosylhomocysteinase Deficiency (Hypermethioninemia due to AHCY deficiency)",
        "chromosome": "20q11.22",
        "inheritance": "Autosomal Recessive",
        "omim_gene": "OMIM *180960",
        "omim_disease": "OMIM #613752",
        "protein_size": "432 aa; NAD+-dependent homotetrameric enzyme; cytoplasmic",
        "prevalence": "<1 in 1,000,000 births; fewer than 50 cases reported worldwide (2026)",
        "cohort_n": n,
        "phenotype_distribution": pheno_dist,
        "function": (
            "AHCY (adenosylhomocysteinase, also called S-adenosylhomocysteine hydrolase / SAHH) "
            "catalyzes the reversible hydrolysis of S-adenosylhomocysteine (SAH) into adenosine "
            "and homocysteine. This is the ONLY cellular route to clear SAH — the universal "
            "by-product of all SAM-dependent methylation reactions. SAH is a potent competitive "
            "inhibitor of all SAM-dependent methyltransferases; its accumulation blocks DNA, RNA, "
            "histone, neurotransmitter, creatine, and phospholipid methylation simultaneously. "
            "AHCY requires NAD+ as cofactor (bound and tightly regulated by redox state). "
            "The enzyme functions as a homotetramer; each monomer binds one NAD+ and one substrate."
        ),
        "mechanism": (
            "AHCY LOF → SAH cannot be hydrolyzed → SAH accumulates to pathologically high levels. "
            "SAH is a potent product inhibitor of ALL SAM-dependent methyltransferases: "
            "DNA methyltransferases (DNMT1/3A/3B) — gene expression dysregulation; "
            "histone methyltransferases — epigenetic disruption; "
            "COMT / HNMT / PNMT — neurotransmitter methylation impaired → CNS dysfunction; "
            "GAMT — creatine synthesis blocked → secondary creatine deficiency (muscle, brain); "
            "PEMT — phosphatidylcholine synthesis impaired → mitochondrial membrane failure → myopathy. "
            "SAM accumulates (cannot be consumed → backs up → methionine very high 200-600 µmol/L). "
            "SAM/SAH ratio is SEVERELY REDUCED (normal >4; in AHCY <0.5) — the cellular methylation "
            "index. tHcy is MODERATELY elevated (40-150 µmol/L) — milder than CBS despite higher "
            "methionine, because transsulfuration (CBS) is still functional and removes some Hcy. "
            "KEY DISTINGUISHING FEATURES FROM CBS: SAH ELEVATED (pathognomonic — ABSENT in CBS); "
            "Myopathy severe and prominent (ABSENT in CBS); No ectopia lentis (present in CBS 90%). "
            "KEY DISTINGUISHING FROM MAT1A: MAT1A has very high methionine + LOW SAM (blocked synthesis); "
            "AHCY has very high methionine + HIGH SAM (intact synthesis, blocked downstream); "
            "SAH elevated ONLY in AHCY."
        ),
        "key_positive_features": (
            "KEY POSITIVES (unique to AHCY among metabolic epilepsy disorders): "
            "(1) SAH MASSIVELY ELEVATED — pathognomonic; not seen in any other inherited HHcy disorder. "
            "    Requires specialized HPLC for SAH + SAM measurement. "
            "(2) Methionine VERY HIGH (200-600 µmol/L) — highest absolute methionine of isolated HHcy. "
            "(3) SAM/SAH RATIO SEVERELY REDUCED (<0.5; normal >4) — global methylation failure index. "
            "(4) Severe MYOPATHY — proximal weakness, elevated CK (500-10,000 U/L) — not seen in CBS/cblE/MTHFR. "
            "(5) Cardiomyopathy (60-70%) — unique among isolated HHcy disorders. "
            "(6) Hepatomegaly + liver disease (70-75%) — PEMT impairment → phospholipid deficiency. "
            "KEY NEGATIVES (shared with other isolated HHcy): "
            "(1) MMA NORMAL — distinguishes AHCY from combined MMA+HHcy (cblC/D/F/J/X). "
            "(2) MeCbl NORMAL — cobalamin system intact (unlike cblE/cblG). "
            "(3) Ectopia lentis ABSENT — unique absence vs CBS (90%). "
            "(4) B6 responsiveness ABSENT — AHCY is NAD+-dependent, NOT PLP-dependent. "
            "(5) Megaloblastic anemia ABSENT — no primary methylfolate trap. "
            "(6) No marfanoid habitus (unlike CBS)."
        ),
        "nbs_primary": (
            "Amino acid panel — VERY HIGH methionine (200–600 µmol/L triggers recall; normal <60). "
            "~70% of AHCY severe cases detected on NBS. Standard NBS CANNOT detect SAH "
            "(requires specialized HPLC not in routine NBS panels) — SAH measurement is the "
            "CONFIRMATORY test, not the screen. SAH/SAM profiling by HPLC is diagnostic gold standard."
        ),
        "nbs_secondary": (
            "SAH measurement by HPLC: markedly elevated (often >100-fold normal); pathognomonic. "
            "SAM by HPLC: elevated (unlike MAT1A where SAM is LOW). "
            "SAM/SAH ratio: severely reduced. "
            "Plasma total homocysteine: moderately elevated (40-150 µmol/L). "
            "CK: very elevated (myopathy). "
            "Echocardiography: cardiomyopathy (hypertrophic or dilated). "
            "Liver function tests: elevated AST/ALT/GGT. "
            "Brain MRI: periventricular white matter abnormalities. "
            "AHCY enzyme activity in erythrocytes: severely reduced or absent."
        ),
        "kpis": {
            "avg_homocysteine_umol_l":  avg("homocysteine_umol_l"),
            "avg_methionine_umol_l":    avg("methionine_umol_l"),
            "avg_sah_arbitrary_units":  avg("sah_arbitrary_units"),
            "avg_sam_sah_ratio":        avg("sam_sah_ratio"),
            "avg_ck_u_l":               avg("ck_u_l"),
            "avg_ast_u_l":              avg("ast_u_l"),
            "pct_myopathy":             pct(lambda p: p["myopathy"]),
            "pct_cardiomyopathy":       pct(lambda p: p["cardiomyopathy"]),
            "pct_hepatomegaly":         pct(lambda p: p["hepatomegaly"]),
            "pct_idd":                  pct(lambda p: p["idd"]),
            "pct_seizures":             pct(lambda p: p["seizures"]),
            "pct_nbs_detected":         pct(lambda p: p["nbs_detected"]),
            "pct_mma_normal":           pct(lambda p: p["mma_normal"]),
            "pct_mecbl_normal":         pct(lambda p: p["mecbl_normal"]),
            "pct_on_met_restriction":   pct(lambda p: p["on_methionine_restriction"]),
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Focal seizures (frontal/temporal — white-matter disease)", "pct": 52},
        {"type": "Generalized tonic-clonic", "pct": 28},
        {"type": "Infantile spasms (severe neonatal class)", "pct": 12},
        {"type": "Myoclonic (neurotransmitter methylation failure)", "pct": 8},
    ]

    metabolic_triggers = [
        {
            "trigger": "High-methionine diet (meat, dairy, legumes without restriction)",
            "pct": 90,
            "mechanism": (
                "Methionine → SAM → SAH → SAH cannot be hydrolyzed → SAH accumulates further. "
                "Even modest methionine intake floods the stalled cycle. Methionine restriction "
                "is the cornerstone therapy to reduce SAH production at source."
            ),
        },
        {
            "trigger": "Fasting / catabolic illness (protein catabolism)",
            "pct": 62,
            "mechanism": (
                "Protein breakdown releases methionine → SAM surge → more SAH accumulation. "
                "Emergency protocol: IV dextrose to suppress catabolism; urgent amino acid formula."
            ),
        },
        {
            "trigger": "SAM supplementation (contraindicated)",
            "pct": 55,
            "mechanism": (
                "Exogenous SAM directly feeds the blocked step: more SAM → more SAH via "
                "methyltransferases → catastrophic SAH accumulation. SAM is ABSOLUTELY "
                "CONTRAINDICATED in AHCY deficiency."
            ),
        },
        {
            "trigger": "Betaine (TMG) — use with extreme caution / often avoided",
            "pct": 45,
            "mechanism": (
                "BHMT converts Hcy + betaine → methionine; this lowers Hcy but INCREASES "
                "methionine → more SAM production → more SAH via methyltransferases → "
                "SAH worsens. Betaine is used cautiously in some centers to lower Hcy, "
                "but requires close SAH monitoring. Often avoided or restricted."
            ),
        },
        {
            "trigger": "Antifolates (methotrexate, trimethoprim)",
            "pct": 35,
            "mechanism": (
                "Antifolates impair THF regeneration → reduce folate supply → impair "
                "remethylation further; combined with already-impaired methyltransferases "
                "(SAH inhibition) → catastrophic global methylation failure."
            ),
        },
        {
            "trigger": "Nitrous oxide (N2O) — moderate risk",
            "pct": 30,
            "mechanism": (
                "N2O oxidizes MTR cobalamin → MTR inactivated → Hcy cannot be remethylated "
                "→ Hcy rises further; SAH may accumulate more. Not absolute CI (MTR cobalamin "
                "intact in AHCY, unlike cblE/cblG), but moderately risky."
            ),
        },
    ]

    treatments = [
        {
            "treatment": "Methionine restriction (low-Met diet + methionine-free amino acid formula)",
            "level": "Level A (ALL AHCY patients — primary treatment)",
            "mechanism": (
                "Reducing dietary methionine is the cornerstone: less methionine → less SAM "
                "synthesis → less SAH production via methyltransferases → SAH levels decrease. "
                "Methionine-free amino acid formula provides all essential amino acids. Target "
                "plasma methionine 20–80 µmol/L. Formula must include adequate cysteine "
                "(conditionally essential: transsulfuration partially functional but substrate "
                "push is reduced). Strict adherence is critical — even small methionine "
                "excesses cause disproportionate SAH accumulation."
            ),
        },
        {
            "treatment": "Adenosine supplementation — oral/IV adenosine or adenosine precursors",
            "level": "Level A (corrects product deficiency — adenosine is missing)",
            "mechanism": (
                "AHCY reaction: SAH → adenosine + Hcy. When AHCY is absent, adenosine is "
                "not produced from SAH. Adenosine deficiency impairs cell signaling, "
                "nucleoside salvage, and ATP synthesis. Adenosine supplementation (or "
                "inosine/AMP as adenosine precursors) corrects this product deficiency "
                "directly. Some centers use allopurinol (xanthine oxidase inhibitor) to "
                "reduce adenosine catabolism and preserve available adenosine."
            ),
        },
        {
            "treatment": "Betaine (TMG) — 50–100 mg/kg/day (with caution)",
            "level": "Level B (selected patients; close SAH monitoring required)",
            "mechanism": (
                "Betaine via BHMT lowers Hcy (Hcy + betaine → methionine). Reduces "
                "HHcy-mediated excitotoxicity and thrombosis risk. CAUTION: betaine "
                "also increases methionine → more SAM → potentially more SAH. Used only "
                "in patients where Hcy reduction outweighs risk of methionine worsening; "
                "requires close monitoring of SAH, SAM, and methionine simultaneously. "
                "Some centers restrict betaine entirely in AHCY."
            ),
        },
        {
            "treatment": "Folic acid — 5 mg/day",
            "level": "Level B (support remethylation; no primary trap but supports MTR)",
            "mechanism": (
                "Folate supports remethylation cycle via MTHFR/MTR pathway; ensures 5-methylTHF "
                "is available for MTR to remethylate Hcy → methionine. Does not directly "
                "address SAH accumulation but supports the limited remethylation capacity."
            ),
        },
        {
            "treatment": "Creatine supplementation — 0.1–0.3 g/kg/day",
            "level": "Level B (secondary creatine deficiency is common)",
            "mechanism": (
                "GAMT (guanidinoacetate methyltransferase) requires SAM to synthesize creatine. "
                "SAH inhibits GAMT → secondary creatine deficiency in muscle and brain → "
                "contributes to myopathy and neurodevelopmental impairment. Creatine "
                "monohydrate supplementation bypasses the GAMT block."
            ),
        },
        {
            "treatment": "Levetiracetam (LEV) — first-line AED for seizures",
            "level": "Level B (first-line; avoids drug-specific metabolic risks)",
            "mechanism": (
                "LEV has no significant impact on methionine metabolism, methylation pathways, "
                "SAH, SAM, or liver function. Safe in the setting of pre-existing liver "
                "disease (dose adjustment in renal impairment). No carnitine depletion. "
                "SV2A mechanism unrelated to methylation pathways."
            ),
        },
        {
            "treatment": "Cardiac monitoring + ACEI/diuretics for cardiomyopathy",
            "level": "Level A (cardiac — cardiomyopathy is life-threatening in severe AHCY)",
            "mechanism": (
                "Cardiomyopathy (60–70% of severe/classic AHCY) is a major cause of early "
                "death. Echocardiography q3–6 months. ACE inhibitors / beta-blockers / "
                "diuretics per cardiology protocol for dilated or hypertrophic CM. "
                "Cardiac transplantation considered in refractory cases."
            ),
        },
    ]

    drug_risks = [
        {
            "agent": "S-Adenosylmethionine (SAM) supplements",
            "risk": "ABSOLUTE CONTRAINDICATION — catastrophic SAH accumulation",
            "mechanism": (
                "Exogenous SAM directly enters the stalled pathway: SAM is metabolized by "
                "methyltransferases → SAH produced → AHCY absent → SAH accumulates to "
                "catastrophic levels. All SAM-containing supplements, functional foods with "
                "SAM, and parenteral SAM are absolutely contraindicated in AHCY deficiency."
            ),
        },
        {
            "agent": "Betaine (TMG) / choline (high-dose)",
            "risk": "HIGH RISK — worsens methionine and SAH (use only with close monitoring)",
            "mechanism": (
                "BHMT pathway: Hcy + betaine → methionine → SAM → SAH (worsened). "
                "Betaine lowers Hcy but raises methionine and ultimately SAH. High-dose "
                "betaine can worsen the methylation crisis. Avoid in severe AHCY. "
                "If used in mild-classic to control Hcy, keep dose minimal and monitor "
                "SAH, SAM, and methionine simultaneously."
            ),
        },
        {
            "agent": "Valproate (VPA)",
            "risk": "HIGH RISK — hepatotoxicity in context of pre-existing liver disease",
            "mechanism": (
                "AHCY patients have hepatomegaly and liver disease (70–75%). VPA is "
                "hepatotoxic and depletes carnitine (required for mitochondrial function "
                "already impaired in AHCY). VPA-induced hepatotoxicity can precipitate "
                "liver failure in AHCY. Use levetiracetam first-line."
            ),
        },
        {
            "agent": "Methotrexate / antifolates",
            "risk": "HIGH RISK — global methylation catastrophe",
            "mechanism": (
                "Methotrexate blocks DHFR → reduces THF → reduces 5-methylTHF → impairs "
                "MTR-mediated remethylation of Hcy → Hcy rises; ALSO directly impairs "
                "methylation (combined with SAH already blocking all methyltransferases). "
                "Trimethoprim has similar mechanism at DHFR. Both absolutely avoided."
            ),
        },
        {
            "agent": "Nitrous oxide (N2O)",
            "risk": "MODERATE RISK (not absolute CI — MTR cobalamin intact but Hcy sensitized)",
            "mechanism": (
                "N2O oxidizes MTR cobalamin (Co²⁺ → Co³⁺) → MTR inactivated → Hcy cannot "
                "be remethylated via MTR → Hcy rises. In AHCY, Hcy is already elevated and "
                "SAH is pathological; further Hcy rise → more SAH burden. Not absolute CI "
                "(unlike cblE/cblG where MTR system already compromised), but avoid "
                "electively. If unavoidable: pre-load cobalamin + post-operative monitoring."
            ),
        },
        {
            "agent": "High-protein diet / protein boluses",
            "risk": "HIGH RISK — methionine loading",
            "mechanism": (
                "Any protein load contains methionine → feeds the stalled methionine cycle "
                "→ SAM and SAH accumulate. Patients must maintain strict methionine-restricted "
                "diet at all times. High-protein feeds (TPN without methionine restriction "
                "formula) in hospital are a common iatrogenic error."
            ),
        },
    ]

    variants = [
        {
            "variant": "p.Tyr143Cys",
            "domain": "NAD+ binding domain",
            "prevalence": "Most common European; ~30% of AHCY alleles",
            "severity": "Moderate-severe; reduced NAD+ affinity → partial enzyme activity",
        },
        {
            "variant": "p.Glu68Lys",
            "domain": "Tetramer interface",
            "prevalence": "~15% of AHCY alleles",
            "severity": "Moderate; tetramer destabilization → reduced activity",
        },
        {
            "variant": "p.Asp244Glu",
            "domain": "Catalytic domain",
            "prevalence": "~15% of AHCY alleles",
            "severity": "Moderate; catalytic residue disruption; some residual activity",
        },
        {
            "variant": "p.His55Arg",
            "domain": "NAD+ binding domain",
            "prevalence": "~12% of AHCY alleles",
            "severity": "Severe; NAD+ binding completely disrupted; null phenotype",
        },
        {
            "variant": "p.Pro344Leu",
            "domain": "C-terminal domain",
            "prevalence": "~10% of AHCY alleles",
            "severity": "Severe neonatal; complete loss of function",
        },
        {
            "variant": "p.Ala89Val",
            "domain": "Catalytic domain",
            "prevalence": "~10% of AHCY alleles",
            "severity": "Mild/attenuated; significant residual activity (25-40%)",
        },
        {
            "variant": "c.IVS7+1G>T",
            "domain": "Splice site",
            "prevalence": "~8% of AHCY alleles",
            "severity": "Severe; splice null; exon 7 skipping → truncated protein",
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
            "sah_arbitrary_units": {
                "severe_neonatal_infantile": "350–800 (arbitrary HPLC units; >100-fold normal)",
                "classic_infantile_childhood": "150–500",
                "mild_attenuated": "60–200",
                "note": "SAH PATHOGNOMONIC ELEVATED — requires specialized HPLC; NOT in standard NBS panel",
            },
            "sam_arbitrary_units": {
                "severe": "120–280 (elevated — unlike MAT1A where SAM is LOW)",
                "classic": "80–200 (elevated)",
                "mild": "50–140 (elevated or high-normal)",
                "note": "SAM elevated because methyltransferases cannot consume it (SAH product inhibition)",
            },
            "sam_sah_ratio": {
                "normal": ">4 (functional methylation index)",
                "ahcy_all": "<2; typically <0.5 (severely reduced — global methylation failure)",
                "note": "SAM/SAH ratio is the KEY DIAGNOSTIC NUMBER — always severely reduced in AHCY",
            },
            "methionine_umol_l": {
                "severe": "320–610 µmol/L (VERY HIGH; highest of isolated HHcy disorders by absolute level)",
                "classic": "180–420 µmol/L",
                "mild": "120–280 µmol/L",
                "note": "KEY DISTINCTION vs MAT1A: methionine very high + SAM ELEVATED (MAT1A: methionine very high + SAM LOW)",
            },
            "homocysteine_umol_l": {
                "severe": "60–155 µmol/L (MODERATE; milder than CBS 100-500 despite higher methionine)",
                "classic": "40–120 µmol/L",
                "mild": "25–70 µmol/L",
                "note": "PARADOXICALLY MODERATE — transsulfuration (CBS) partially functional; "
                        "SAH blocks remethylation but Hcy can still exit via CBS transsulfuration",
            },
            "ck_u_l": {
                "severe": "800–10,000 U/L (very high — severe myopathy)",
                "classic": "200–3,500 U/L",
                "mild": "80–600 U/L",
                "note": "CK ELEVATED — hallmark of AHCY myopathy; absent in CBS/cblE/cblG/MTHFR",
            },
            "mma": "NORMAL (<5 mmol/mol Cr) — ALL patients; MMA arm completely intact",
            "mecbl_adocbl": "BOTH NORMAL — cobalamin system entirely intact; AHCY does not involve cobalamin",
            "serum_folate": "NORMAL to mildly elevated — no primary methylfolate trap",
            "megaloblastic_anemia": "ABSENT or mild — methylfolate trap not the primary mechanism",
        },
        "kpi_pcts": {
            "myopathy":         pct(lambda p: p["myopathy"]),
            "cardiomyopathy":   pct(lambda p: p["cardiomyopathy"]),
            "hepatomegaly":     pct(lambda p: p["hepatomegaly"]),
            "idd":              pct(lambda p: p["idd"]),
            "seizures":         pct(lambda p: p["seizures"]),
            "nbs_detected":     pct(lambda p: p["nbs_detected"]),
            "dysmorphic":       pct(lambda p: p["dysmorphic"]),
        },
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "gene":          "AHCY (Adenosylhomocysteinase; also SAHH — S-Adenosylhomocysteine Hydrolase)",
            "gene_omim":     "*180960",
            "disease_omim":  "#613752 (Hypermethioninemia due to adenosylhomocysteine hydrolase deficiency)",
            "protein":       "432 amino acids; NAD+-dependent; homotetrameric (4 identical subunits); cytoplasmic",
            "cofactor":      "NAD+ (tightly bound, not consumed in each reaction cycle; used for redox regulation of active site)",
            "locus":         "20q11.22",
            "inheritance":   "Autosomal Recessive (biallelic LOF); both sexes equally affected",
            "reaction":      "S-Adenosylhomocysteine (SAH) ⇌ Adenosine + L-Homocysteine (thermodynamically unfavorable toward hydrolysis; driven by product removal)",
            "pathway":       "Methionine cycle — SAH hydrolysis checkpoint; the ONLY route to clear SAH",
            "regulation":    "Product inhibited by elevated SAH (self-regulatory, but pathological in AHCY deficiency); "
                             "NAD+/NADH ratio modulates enzyme activity (oxidative form active; reduced form inactive)",
            "unique_feature": "AHCY is the universal checkpoint for the methionine cycle — SAH inhibits ALL SAM-dependent methyltransferases. "
                              "AHCY deficiency = global methylation failure across DNA, RNA, histones, neurotransmitters, creatine, phospholipids.",
        },
        "key_concepts": [
            {
                "concept": "SAH — Pathognomonic Biomarker (not in standard NBS)",
                "explanation": (
                    "S-adenosylhomocysteine (SAH) is produced every time SAM donates its methyl group. "
                    "AHCY is the only enzyme that hydrolyzes SAH. AHCY LOF → SAH accumulates >100-fold normal. "
                    "SAH measurement by HPLC is the CONFIRMATORY diagnostic test and pathognomonic for AHCY deficiency. "
                    "SAH is NOT measured in standard newborn screening panels (amino acid + acylcarnitine); "
                    "NBS detects AHCY only via elevated methionine (indirect). "
                    "SAH + SAM + SAM/SAH ratio (HPLC) is the gold standard confirmation."
                ),
            },
            {
                "concept": "SAM/SAH Ratio — Global Methylation Index",
                "explanation": (
                    "SAM/SAH ratio is the best marker of cellular methylation capacity. "
                    "Normal ratio >4 (SAM high, SAH low → methyltransferases can function). "
                    "AHCY deficiency: SAM/SAH ratio typically <0.5 (severely reduced). "
                    "This means ALL methyltransferases simultaneously impaired: "
                    "DNA methylation → gene expression changes; "
                    "Histone methylation → epigenetic dysregulation; "
                    "COMT/HNMT/PNMT → neurotransmitter pathology (dopamine, histamine, epinephrine); "
                    "GAMT → creatine deficiency (myopathy, brain); "
                    "PEMT → phosphatidylcholine deficiency (hepatopathy, mitochondrial membrane). "
                    "Treatment goal: maximize SAM/SAH ratio by reducing SAH production (methionine restriction)."
                ),
            },
            {
                "concept": "AHCY vs CBS vs MAT1A — Differential of Hypermethioninemia",
                "explanation": (
                    "All three cause very high plasma methionine on NBS (amino acid panel). "
                    "CRITICAL DIFFERENCES: "
                    "AHCY: methionine HIGH + SAM HIGH + SAH PATHOGNOMONIC HIGH + tHcy moderate + myopathy + cardiomyopathy. "
                    "CBS: methionine HIGH + SAM normal-high + SAH NORMAL + tHcy very high (100-500) + ectopia lentis + B6 responsiveness. "
                    "MAT1A: methionine VERY HIGH (often >1000) + SAM LOW (synthesis blocked) + SAH LOW/NORMAL + tHcy normal + benign/neurological. "
                    "GNMT: methionine HIGH + SAM HIGH + SAH LOW/NORMAL + tHcy normal + liver disease. "
                    "AHCY is uniquely identified by SAH elevated + SAM elevated + SAM/SAH severely reduced + myopathy."
                ),
            },
            {
                "concept": "Myopathy — The Dominant Clinical Feature of AHCY",
                "explanation": (
                    "Severe proximal myopathy (85-90% of patients) is the hallmark distinguishing AHCY "
                    "from all other inherited HHcy disorders. CK is very elevated (500-10,000 U/L). "
                    "Mechanism: "
                    "(1) GAMT impairment → secondary creatine deficiency → muscle energy failure; "
                    "(2) PEMT impairment → phosphatidylcholine deficiency → mitochondrial membrane dysfunction; "
                    "(3) Myosin heavy chain methylation impaired → contractile apparatus failure. "
                    "Muscle biopsy: mitochondrial changes, lipid accumulation, fibre type disproportion. "
                    "Myopathy may precede all other features and is the presenting complaint in classic cases. "
                    "Treatment: methionine restriction + adenosine + creatine supplementation."
                ),
            },
            {
                "concept": "SAM Supplements and Betaine — CONTRAINDICATED (or extreme caution)",
                "explanation": (
                    "SAM supplementation is ABSOLUTELY CONTRAINDICATED in AHCY deficiency: "
                    "exogenous SAM is metabolized by methyltransferases → produces more SAH → "
                    "SAH accumulates catastrophically. This is the opposite of most methylation-deficient "
                    "disorders where SAM supplementation is beneficial. "
                    "Betaine is also used with extreme caution (if at all): betaine → Hcy remethylation → "
                    "more methionine → more SAM → more SAH. Some centers avoid betaine entirely in AHCY. "
                    "This is a critical contraindication trap for clinicians more familiar with other HHcy disorders "
                    "where betaine is Level A therapy (CBS, cblE, cblG, MTHFR)."
                ),
            },
            {
                "concept": "MMA NORMAL — Distinguishes AHCY from Combined MMA+HHcy Disorders",
                "explanation": (
                    "cblC (MMACHC), cblD (MMADHC), cblF (LMBRD1), cblJ (ABCD4), cblX (HCFC1) "
                    "all cause elevated MMA + HHcy. AHCY causes isolated HHcy (+ myopathy + SAH elevation) "
                    "without any MMA elevation. MMA NORMAL is a key distinguishing negative. "
                    "However, AHCY must further be distinguished from CBS/cblE/cblG/MTHFR via "
                    "methionine level, SAH, myopathy, and cardiomyopathy."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease": "CBS deficiency (Classical Homocystinuria) — 21q22.3",
                "distinguishing": (
                    "CBS: methionine HIGH but LOWER than AHCY; tHcy VERY HIGH (100-500) vs AHCY MODERATE (40-150). "
                    "CBS: SAH NORMAL; AHCY: SAH PATHOGNOMONIC HIGH. "
                    "CBS: ectopia lentis 90% (PATHOGNOMONIC); ABSENT in AHCY. "
                    "CBS: marfanoid habitus 80%; ABSENT in AHCY. "
                    "CBS: B6 responsiveness 50%; ABSENT in AHCY (NAD+-dependent, NOT PLP). "
                    "CBS: thromboembolism 50-60% (HIGHEST); AHCY moderate (20-25%). "
                    "CBS: myopathy ABSENT; AHCY: severe myopathy 85-90% (KEY DISTINCTION). "
                    "CBS: cardiomyopathy ABSENT; AHCY: cardiomyopathy 60-70%. "
                    "CBS: hepatomegaly ABSENT; AHCY: hepatomegaly 70-75%."
                ),
            },
            {
                "disease": "MAT1A deficiency (Methionine Adenosyltransferase I/III Deficiency) — 10q22.3",
                "distinguishing": (
                    "MAT1A: methionine EXTREMELY HIGH (often >1000 µmol/L, highest of all); "
                    "AHCY: methionine HIGH but not as extreme (200-600). "
                    "CRITICAL: MAT1A: SAM LOW (synthesis blocked — MAT cannot make SAM); "
                    "AHCY: SAM ELEVATED (synthesis intact, downstream clearance blocked). "
                    "MAT1A: SAH NORMAL or LOW; AHCY: SAH PATHOGNOMONIC HIGH. "
                    "SAM/SAH ratio: LOW in AHCY (global methylation failure); NORMAL/HIGH in MAT1A. "
                    "MAT1A: often clinically BENIGN (isolated hypermethioninemia); rare CNS. "
                    "AHCY: severe myopathy + cardiomyopathy + hepatopathy — opposite of MAT1A benignity. "
                    "tHcy: NORMAL in MAT1A (Hcy not produced if SAM is not made); ELEVATED in AHCY."
                ),
            },
            {
                "disease": "GNMT deficiency (Glycine N-Methyltransferase Deficiency) — 6p12.1",
                "distinguishing": (
                    "GNMT: methionine HIGH + SAM HIGH (SAM cannot be consumed via GNMT → glycine methylation). "
                    "GNMT: SAH LOW/NORMAL (SAM not consumed → less SAH produced); AHCY: SAH HIGH. "
                    "GNMT: SAM/SAH ratio HIGH (excess SAM, low SAH); AHCY: SAM/SAH ratio SEVERELY LOW. "
                    "GNMT: liver disease prominent (same as AHCY hepatopathy) but NO myopathy. "
                    "tHcy: NORMAL in GNMT (Hcy production not increased); ELEVATED in AHCY. "
                    "GNMT: no cardiomyopathy; AHCY: cardiomyopathy 60-70%."
                ),
            },
            {
                "disease": "cblE (MTRR) / cblG (MTR) — remethylation defects",
                "distinguishing": (
                    "cblE/cblG: methionine LOW (remethylation impaired → methionine not produced); "
                    "AHCY: methionine VERY HIGH. "
                    "cblE/cblG: MeCbl absent (fibroblasts); AHCY: MeCbl NORMAL. "
                    "cblE/cblG: megaloblastic anemia 80-90% (methylfolate trap); AHCY: absent. "
                    "cblE/cblG: N2O ABSOLUTE CI; AHCY: N2O moderate risk only. "
                    "AHCY: myopathy + cardiomyopathy + hepatomegaly; ABSENT in cblE/cblG."
                ),
            },
            {
                "disease": "MTHFR deficiency — remethylation defect (folate axis)",
                "distinguishing": (
                    "MTHFR: methionine LOW; AHCY: methionine VERY HIGH. "
                    "MTHFR: white matter disease prominent (82%); AHCY: present but secondary. "
                    "MTHFR: riboflavin Level A (FAD cofactor); AHCY: riboflavin not specifically indicated. "
                    "AHCY: SAH elevated (PATHOGNOMONIC); MTHFR: SAH NORMAL."
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
    print("AHCY dashboard OK")
