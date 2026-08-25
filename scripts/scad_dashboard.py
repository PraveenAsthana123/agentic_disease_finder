#!/usr/bin/env python3
"""SCAD (Short-Chain Acyl-CoA Dehydrogenase Deficiency) Dashboard.

ACADS gene encodes SHORT-CHAIN ACYL-CoA DEHYDROGENASE (SCAD):
  - 412 aa (precursor); ~43 kDa subunit; mitochondrial matrix; FAD-dependent; homotetrameric
  - Catalyses: Short-chain acyl-CoA (C4–C6; primarily butyryl-CoA) + FAD → trans-2-enoyl-acyl-CoA + FADH2
  - Step 1 of mitochondrial beta-oxidation spiral for C4–C6 fatty acids
  - Part of the classical FAO triad: VLCAD (C14–C20) → MCAD (C6–C12) → SCAD (C4–C6)

SCAD LOF → short-chain acyl-CoA species CANNOT be dehydrogenated:
  Butyryl-CoA (C4)      ACCUMULATES → butyrylcarnitine (C4) ↑ [PRIMARY NBS MARKER, highly nonspecific]
  Hexanoyl-CoA (C6)     secondarily accumulates → C6 ↑ (minor)
  → Beta-oxidation spiral ARRESTED at short-chain stage
  → Urine: ethylmalonic acid (EMA) ↑ (variable); methylsuccinic acid (MSA) ↑; butyrylglycine ↑

CLINICAL CONTROVERSY — SCAD IS UNIQUE AMONG FAO DISORDERS:
  - MOST patients identified by NBS are ASYMPTOMATIC — major NBS panel debate
  - Common "susceptibility variants" (625G>A = p.Gly209Ser, 511C>T = p.Arg171Trp) found in
    7% and 14% of the GENERAL POPULATION respectively — NOT clearly pathogenic alone
  - Only biallelic NULL alleles (frameshift, splice-site, catalytic missense) → true symptomatic disease
  - Selective inclusion of SCAD in NBS panels is debated internationally (UK removed from panel 2012)
  - NBS SCAD positives require genotyping to distinguish true deficiency from common variant carriers

KEY FACTS (HIGHEST YIELD):
  1. C4 (butyrylcarnitine) — PRIMARY NBS MARKER; >0.5–1.0 µmol/L BUT HIGHLY NONSPECIFIC
  2. C4/C2 ratio — elevated; helps distinguish SCAD from benign causes of C4 elevation
  3. C4/C3 ratio — sometimes used; elevated in SCAD (C3 normal)
  4. Ethylmalonic acid (EMA) — urine; characteristic but VARIABLE (may be normal between episodes)
  5. Methylsuccinic acid (MSA) — urine; co-elevated with EMA in SCAD
  6. Butyrylglycine — urine glycine conjugate; elevated when symptomatic
  7. C4-OH (3-hydroxybutyrylcarnitine) — may be mildly elevated in some SCAD patients
  8. C8 NORMAL — KEY NEGATIVE vs MCAD (MCAD elevates C8; SCAD does NOT)
  9. C14 NORMAL — KEY NEGATIVE vs VLCAD (VLCAD elevates C14:1; SCAD does NOT)
 10. 625G>A (p.Gly209Ser) — common variant; 7% carrier frequency; NOT pathogenic in isolation
 11. 511C>T (p.Arg171Trp) — common variant; 14% carrier frequency; NOT pathogenic in isolation
 12. Hypotonia, developmental delay, seizures — when TRULY symptomatic (rare)
 13. Hypoketotic tendency — milder than MCAD; short-chain FAO blockade less energy-critical
 14. Riboflavin (B2) — may help some patients (FAD cofactor supplementation)
 15. BENIGN in MAJORITY — major NBS controversy; selective panels omit SCAD

OMIM Disease: #201470 (SCAD deficiency)
OMIM Gene:    *606885 (ACADS)
Chromosome:   12q24.31
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Protein:      ACADS = 412 aa (precursor); mitochondrial matrix; FAD-dependent; homotetrameric
              Active site: Glu368 (catalytic base); FAD cofactor
Prevalence:   ~1:50,000 (true symptomatic deficiency); highly variable depending on genotype threshold

FATTY ACID BETA-OXIDATION — WHERE SCAD FITS:
  Very-long-chain FA (C14–C20) → [VLCAD] → C12–C14 range
  Medium-chain FA (C6–C12)     → [MCAD] → C4–C6 range
  Short-chain FA (C4–C6)       → [SCAD + FAD BLOCKED ⚠] → trans-2-butenoyl-CoA (crotonyl-CoA)
  → [ECHS1/EHHADH/HADH/HADHA] → 3-hydroxybutyryl-CoA → acetoacetyl-CoA → [ACAT1] → 2 × acetyl-CoA

  SCAD block → butyryl-CoA CANNOT enter spiral → accumulates as butyrylcarnitine (C4)
  SCAD block → FADH2 not generated from C4–C6 → minor effect on ketogenesis (C4 is near end of spiral)
  SCAD block → LESS energy-critical than MCAD/VLCAD because C4 contributes small fraction of total FAO

VLCAD vs MCAD vs SCAD:
  VLCAD (ACADVL, C14–C20): C14:1 dominant NBS marker; cardiomyopathy + rhabdomyolysis; severe
  MCAD  (ACADM, C6–C12):   C8 dominant NBS marker; hepatopathy + hypoketotic hypoglycaemia; serious
  SCAD  (ACADS, C4–C6):    C4 (butyrylcarnitine) elevated; usually BENIGN; NBS controversy

BIOMARKER PATTERN — SCAD DEFICIENCY:
  C4 (Butyrylcarnitine)           >0.5 µmol/L   — PRIMARY NBS MARKER ↑ (but nonspecific)
  C4/C2 ratio                     >0.20          — elevated; modestly discriminating
  C4/C3 ratio                     >0.50          — elevated (C3 normal helps rule out PA)
  C4-OH (3-OH-butyrylcarnitine)   MILDLY ↑       — minor secondary accumulation
  C6 (Hexanoylcarnitine)          NORMAL/mildly ↑ — C6 intact with MCAD; not diagnostic
  C8 (Octanoylcarnitine)          NORMAL          — KEY NEGATIVE vs MCAD
  C14:1 (Tetradecenoylcarnitine)  NORMAL          — KEY NEGATIVE vs VLCAD
  C0 (Free carnitine)             NORMAL or low   — less depletion than MCAD
  Ethylmalonic acid (EMA)         ELEVATED (urine) — variable; may be >50 mmol/mol Cr
  Methylsuccinic acid (MSA)       ELEVATED (urine) — co-elevated with EMA
  Butyrylglycine                  ELEVATED (urine) — glycine conjugate of butyryl-CoA
  Ketones (β-OHB)                 Near normal/mildly low — less dramatic than MCAD
  Blood glucose                   Mildly low during crisis — less severe than MCAD

WHY C4 IS NONSPECIFIC:
  - C4 includes BOTH butyrylcarnitine AND isobutyrylcarnitine (indistinguishable on standard MS/MS)
  - Isobutyrylcarnitine elevated in ISOBUTYRYL-CoA CARBOXYLASE DEFICIENCY (IBD) — separate disease
  - Multiple benign conditions and dietary factors elevate C4
  - 625G>A (p.Gly209Ser) and 511C>T (p.Arg171Trp) are very common and cause mild C4 elevation without disease
  - NBS C4 positive → require confirmatory urine OA (EMA/MSA) + ACADS sequencing

TREATMENT SUMMARY:
  Avoid fasting (Level B — less critical than MCAD but prudent)
  L-carnitine: if C0 depleted (Level B)
  Riboflavin (B2): 50–200 mg/day — trial for symptomatic patients (FAD cofactor supplementation)
  No dietary fat restriction needed
  VPA: some risk (worsens FAO, avoid in symptomatic patients)
  KD: relative CI (not absolute, but avoid in symptomatic patients)
  Emergency glucose: for rare hypoglycaemic crisis (less acute than MCAD)

PHENOTYPES:
  Asymptomatic NBS-detected (common variant):  ~70% — most common; often 625G>A and/or 511C>T
  Biochemical SCAD (mildly symptomatic):        ~20% — hypotonia, feeding difficulties, variable
  Classic SCAD (biallelic null alleles):        ~10% — hypotonia + developmental delay + seizures

COMMON VARIANTS (ACADS):
  625G>A  (p.Gly209Ser):     7% allele frequency in general population; susceptibility variant; NOT clearly pathogenic alone
  511C>T  (p.Arg171Trp):    14% allele frequency in general population; susceptibility variant; NOT clearly pathogenic alone
  These two variants ALONE do NOT cause classic SCAD — require biallelic null alleles for true disease
  True pathogenic:
  c.1195C>T (p.Arg399Cys): ~10% of true SCAD alleles; catalytic domain; LOF
  c.529G>A  (p.Val177Met):  ~8% of true SCAD alleles; FAD-binding region; partial LOF
  c.736C>T  (p.Arg246Cys):  ~6% of true SCAD alleles; subunit interface; LOF
  c.347A>G  (p.Asn116Ser):  ~5% of true SCAD alleles; mitochondrial import; reduced protein
  Frame/splice null alleles:  various; ~25% of true SCAD alleles
"""

import random

SEED = 275          # deterministic cohort (MCAD=269, VLCAD=271, LCHAD=273, SCAD=275)
N_PATIENTS = 40

random.seed(SEED)

PHENOTYPES = [
    ("Asymptomatic NBS-detected (common variant)", 0.70),
    ("Biochemical SCAD (mildly symptomatic)",      0.20),
    ("Classic SCAD (biallelic null — symptomatic)", 0.10),
]

VARIANTS = [
    {"variant": "625G>A (p.Gly209Ser)",   "freq": 40, "domain": "FAD-binding subdomain",    "phenotype": "NBS / Asymptomatic",  "note": "Susceptibility variant; 7% general population; NOT pathogenic alone; major NBS controversy cause"},
    {"variant": "511C>T (p.Arg171Trp)",   "freq": 30, "domain": "Substrate-binding pocket",  "phenotype": "NBS / Asymptomatic",  "note": "Susceptibility variant; 14% general population; NOT pathogenic alone"},
    {"variant": "c.1195C>T (p.Arg399Cys)","freq": 10, "domain": "Catalytic domain",          "phenotype": "Classic / Biochemical","note": "True pathogenic; LOF; catalytic Arg; classic SCAD phenotype"},
    {"variant": "c.529G>A (p.Val177Met)", "freq":  8, "domain": "FAD-binding region",        "phenotype": "Biochemical / Mild",   "note": "True pathogenic; partial LOF; FAD affinity reduced"},
    {"variant": "c.736C>T (p.Arg246Cys)", "freq":  6, "domain": "Subunit interface",          "phenotype": "Classic",             "note": "True pathogenic; disrupts homotetramer assembly; severe LOF"},
    {"variant": "c.347A>G (p.Asn116Ser)", "freq":  5, "domain": "Mitochondrial targeting",    "phenotype": "Biochemical",         "note": "True pathogenic; reduced import; low protein level"},
    {"variant": "Frameshift / Splice null","freq":  1, "domain": "Various (null alleles)",     "phenotype": "Classic / Severe",    "note": "Null allele; complete LOF; compound heterozygotes with above"},
]


def _weighted_choice(options):
    r = random.random()
    cum = 0.0
    for val, prob in options:
        cum += prob
        if r < cum:
            return val
    return options[-1][0]


def _make_cohort():
    cohort = []
    for i in range(N_PATIENTS):
        ph = _weighted_choice(PHENOTYPES)
        is_asym    = ph == "Asymptomatic NBS-detected (common variant)"
        is_biochem = ph == "Biochemical SCAD (mildly symptomatic)"
        is_classic = ph == "Classic SCAD (biallelic null — symptomatic)"

        onset_mo = (
            0                         if is_asym    else
            random.randint(1, 18)     if is_biochem else
            random.randint(2, 24)     # classic
        )

        # C4 (butyrylcarnitine) — PRIMARY NBS MARKER
        c4 = round(
            random.uniform(0.5, 1.5)  if is_asym    else
            random.uniform(0.8, 2.5)  if is_biochem else
            random.uniform(1.2, 4.0),   # classic
            2
        )

        # C4/C2 ratio
        c2 = round(random.uniform(15, 30), 1)
        c4_c2_ratio = round(c4 / c2, 3)

        # C4-OH (3-hydroxybutyrylcarnitine) — mildly elevated
        c4_oh = round(
            random.uniform(0.05, 0.15) if is_asym    else
            random.uniform(0.10, 0.30) if is_biochem else
            random.uniform(0.15, 0.50),
            2
        )

        # C6 (hexanoylcarnitine) — largely normal
        c6 = round(
            random.uniform(0.02, 0.08) if is_asym    else
            random.uniform(0.04, 0.12) if is_biochem else
            random.uniform(0.05, 0.18),
            2
        )

        # C8 (octanoylcarnitine) — NORMAL (KEY NEGATIVE vs MCAD)
        c8 = round(random.uniform(0.02, 0.12), 2)

        # C14:1 (tetradecenoylcarnitine) — NORMAL (KEY NEGATIVE vs VLCAD)
        c14_1 = round(random.uniform(0.02, 0.10), 2)

        # Free carnitine C0 — normal to mildly low
        c0 = round(
            random.uniform(20, 50)  if is_asym    else
            random.uniform(15, 40)  if is_biochem else
            random.uniform(10, 35),
            1
        )

        # Ethylmalonic acid (EMA, urine) — characteristic but variable
        ema = round(
            random.uniform(5, 40)    if is_asym    else
            random.uniform(20, 120)  if is_biochem else
            random.uniform(40, 200),
            1
        )

        # Methylsuccinic acid (MSA, urine) — co-elevated with EMA
        msa = round(
            random.uniform(3, 20)   if is_asym    else
            random.uniform(10, 60)  if is_biochem else
            random.uniform(20, 100),
            1
        )

        # Butyrylglycine (urine glycine conjugate)
        butyrylgly = round(
            random.uniform(0, 5)    if is_asym    else
            random.uniform(3, 20)   if is_biochem else
            random.uniform(10, 50),
            1
        )

        # Blood glucose — near normal (SCAD rarely causes severe hypoglycaemia)
        glucose = (
            round(random.uniform(3.5, 6.0), 1)  if is_asym    else
            round(random.uniform(2.8, 5.5), 1)  if is_biochem else
            round(random.uniform(2.0, 4.5), 1)
        )

        # Beta-OHB — mildly reduced during illness (less dramatic than MCAD)
        bohb = round(
            random.uniform(0.3, 1.5)  if is_asym    else
            random.uniform(0.2, 1.0)  if is_biochem else
            random.uniform(0.1, 0.8),
            2
        )

        # ALT — usually normal; mildly elevated in classic
        alt = (
            random.randint(10, 40)   if is_asym    else
            random.randint(20, 80)   if is_biochem else
            random.randint(30, 150)
        )

        # Variant assignment
        v_roll = random.random()
        if v_roll < 0.40:
            variant = VARIANTS[0]["variant"]  # 625G>A
        elif v_roll < 0.70:
            variant = VARIANTS[1]["variant"]  # 511C>T
        elif v_roll < 0.80:
            variant = VARIANTS[2]["variant"]  # p.Arg399Cys
        elif v_roll < 0.88:
            variant = VARIANTS[3]["variant"]  # p.Val177Met
        elif v_roll < 0.94:
            variant = VARIANTS[4]["variant"]  # p.Arg246Cys
        elif v_roll < 0.99:
            variant = VARIANTS[5]["variant"]  # p.Asn116Ser
        else:
            variant = VARIANTS[6]["variant"]  # Frameshift

        # Hypotonia
        hypotonia = (
            False                     if is_asym    else
            random.random() < 0.50    if is_biochem else
            random.random() < 0.80
        )

        # Developmental delay
        dev_delay = (
            False                     if is_asym    else
            random.random() < 0.30    if is_biochem else
            random.random() < 0.60
        )

        # Seizures
        seizures = (
            False                     if is_asym    else
            random.random() < 0.15    if is_biochem else
            random.random() < 0.40
        )

        # Riboflavin response
        riboflavin_tried = not is_asym
        riboflavin_resp  = (random.random() < 0.40) if riboflavin_tried else False

        # Treatments
        tx = []
        if not is_asym:
            tx.append("Avoid Fasting (Level B)")
        if c0 < 20:
            tx.append("L-Carnitine (C0 depleted)")
        if not is_asym:
            tx.append("Riboflavin (B2) — trial")
        if is_classic:
            tx.append("Metabolic diet review")

        cohort.append({
            "id":             f"SCAD-{i+1:03d}",
            "phenotype":      ph,
            "onset_mo":       onset_mo,
            "c4":             c4,
            "c4_c2_ratio":    c4_c2_ratio,
            "c4_oh":          c4_oh,
            "c6":             c6,
            "c8":             c8,
            "c14_1":          c14_1,
            "c0":             c0,
            "ema":            ema,
            "msa":            msa,
            "butyrylgly":     butyrylgly,
            "glucose":        glucose,
            "bohb":           bohb,
            "alt":            alt,
            "variant":        variant,
            "hypotonia":      hypotonia,
            "dev_delay":      dev_delay,
            "seizures":       seizures,
            "riboflavin_tried": riboflavin_tried,
            "riboflavin_resp":  riboflavin_resp,
            "treatments":     tx,
        })
    return cohort


_COHORT = None


def _get_cohort():
    global _COHORT
    if _COHORT is None:
        _COHORT = _make_cohort()
    return _COHORT


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_overview():
    cohort = _get_cohort()

    ph_counts = {}
    for p in cohort:
        ph_counts[p["phenotype"]] = ph_counts.get(p["phenotype"], 0) + 1

    variant_counts = {}
    for p in cohort:
        variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1

    asym_n      = ph_counts.get("Asymptomatic NBS-detected (common variant)", 0)
    biochem_n   = ph_counts.get("Biochemical SCAD (mildly symptomatic)", 0)
    classic_n   = ph_counts.get("Classic SCAD (biallelic null — symptomatic)", 0)
    seizure_n   = sum(1 for p in cohort if p["seizures"])
    hypotonia_n = sum(1 for p in cohort if p["hypotonia"])
    dev_delay_n = sum(1 for p in cohort if p["dev_delay"])
    ribo_resp_n = sum(1 for p in cohort if p["riboflavin_resp"])

    avg_c4   = round(sum(p["c4"] for p in cohort) / N_PATIENTS, 2)
    avg_ema  = round(sum(p["ema"] for p in cohort) / N_PATIENTS, 1)
    avg_c0   = round(sum(p["c0"] for p in cohort) / N_PATIENTS, 1)

    top_variant = max(variant_counts, key=variant_counts.get)
    top_v_pct   = round(100 * variant_counts[top_variant] / N_PATIENTS)

    return {
        "disease": "SCAD Deficiency (Short-Chain Acyl-CoA Dehydrogenase Deficiency)",
        "gene":    "ACADS (SCAD)",
        "locus":   "12q24.31",
        "omim_gene":    "606885",
        "omim_disease": "201470",
        "inheritance":  "Autosomal Recessive (AR)",
        "protein":      "ACADS — 412 aa precursor; mitochondrial matrix; FAD-dependent; homotetrameric",
        "prevalence":   "~1:50,000 (true symptomatic); highly genotype-dependent; NBS controversy",
        "n_patients":   N_PATIENTS,
        "seed":         SEED,
        "kpis": {
            "total_patients":   N_PATIENTS,
            "asymptomatic_nbs": asym_n,
            "biochemical_n":    biochem_n,
            "classic_n":        classic_n,
            "seizures_n":       seizure_n,
            "hypotonia_n":      hypotonia_n,
            "dev_delay_n":      dev_delay_n,
            "riboflavin_resp_n": ribo_resp_n,
            "avg_c4_umol":      avg_c4,
            "avg_ema_mmol_cr":  avg_ema,
            "avg_c0_umol":      avg_c0,
        },
        "phenotype_distribution": ph_counts,
        "top_variant":      top_variant,
        "top_variant_pct":  top_v_pct,
        "primary_nbs_marker":  "C4 (Butyrylcarnitine) >0.5 µmol/L — NONSPECIFIC; requires confirmation",
        "nbs_controversy":     "625G>A (p.Gly209Ser) and 511C>T (p.Arg171Trp) are common variants in general population — NOT pathogenic alone; UK removed SCAD from panel 2012",
        "characteristic_urine": ["Ethylmalonic acid (EMA)", "Methylsuccinic acid (MSA)", "Butyrylglycine"],
        "key_negatives":    ["C8 NORMAL vs MCAD", "C14:1 NORMAL vs VLCAD", "C3 NORMAL vs PA", "No HG/SG/PPG vs MCAD"],
        "absolute_ci":      [],  # SCAD has NO absolute CI (unlike MCAD/VLCAD)
        "high_risk_drugs":  ["VPA (avoid in symptomatic patients)"],
        "first_line_treatment": "Watchful waiting for asymptomatic; Riboflavin (B2) trial for symptomatic; avoid fasting",
        "clinical_summary": (
            "SCAD deficiency (ACADS, 12q24.31) is the MOST CONTROVERSIAL fatty acid oxidation (FAO) disorder. "
            "ACADS encodes short-chain acyl-CoA dehydrogenase (412 aa; FAD-dependent) — Step 1 of beta-oxidation "
            "for C4–C6 fatty acids. SCAD LOF → C4 (butyrylcarnitine) accumulates → PRIMARY NBS marker, BUT highly nonspecific. "
            "The majority of NBS-detected patients carry common susceptibility variants (625G>A, 511C>T) "
            "found in 7–14% of the general population and are ASYMPTOMATIC. "
            "True symptomatic SCAD (biallelic null alleles) causes hypotonia, developmental delay, and rarely seizures. "
            "C8 and C14:1 NORMAL — KEY NEGATIVES vs MCAD and VLCAD. "
            "Riboflavin (B2) trial for symptomatic patients (FAD cofactor supplementation may help). "
            "No absolute dietary CI (unlike MCAD/VLCAD). UK removed SCAD from NBS panel in 2012 due to controversy."
        ),
    }


def get_breakdown():
    cohort = _get_cohort()
    patients = cohort[:10]  # representative subset for detailed view

    biomarkers = {
        "c4": {
            "label":     "C4 (Butyrylcarnitine)",
            "normal":    "<0.4 µmol/L",
            "status":    "ELEVATED >0.5–1.0 µmol/L — PRIMARY NBS MARKER (highly nonspecific)",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "C4 (butyrylcarnitine) is the primary tandem MS/MS NBS marker for SCAD deficiency. "
                "Butyryl-CoA (C4) accumulates when SCAD is absent → conjugated to carnitine → C4 released into blood. "
                "NBS cut-off: >0.5–1.0 µmol/L (lab-specific). "
                "CRITICAL: C4 is HIGHLY NONSPECIFIC — includes isobutyrylcarnitine (cannot distinguish on standard MS/MS). "
                "Causes of C4 elevation: SCAD deficiency, isobutyryl-CoA carboxylase deficiency, common SCAD variants (625G>A, 511C>T), "
                "dietary artefacts, maternal conditions. "
                "Confirmatory testing (urine OA + ACADS sequencing) MANDATORY for all NBS C4 positives."
            ),
        },
        "c4_c2_ratio": {
            "label":     "C4/C2 Ratio (Butyryl / Acetyl)",
            "normal":    "<0.15",
            "status":    "ELEVATED >0.20 — modestly discriminating",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "C4/C2 ratio is used as a second-tier discriminator for SCAD NBS positives. "
                "C2 (acetylcarnitine) reflects overall acetyl-CoA availability; ratio elevation suggests selective C4 accumulation. "
                "More useful than C4 alone, but still limited specificity. "
                "Low sensitivity for mild/variant SCAD; normal C4/C2 does not rule out true deficiency."
            ),
        },
        "c4_oh": {
            "label":     "C4-OH (3-Hydroxybutyrylcarnitine)",
            "normal":    "<0.10 µmol/L",
            "status":    "MILDLY ELEVATED — minor secondary accumulation",
            "direction": "↑ (mild)",
            "color":     "warning",
            "rationale": (
                "3-Hydroxybutyryl-CoA (the next intermediate after crotonyl-CoA in short-chain FAO) "
                "may mildly accumulate when SCAD is blocked. "
                "C4-OH elevation is non-specific; also seen in ACAT1 deficiency and ketosis. "
                "In SCAD: C4-OH elevation is minor; C4 remains the dominant signal."
            ),
        },
        "c8": {
            "label":     "C8 (Octanoylcarnitine)",
            "normal":    "<0.15 µmol/L",
            "status":    "NORMAL — KEY NEGATIVE vs MCAD",
            "direction": "→ NORMAL",
            "color":     "success",
            "rationale": (
                "C8 (octanoylcarnitine) is the PRIMARY NBS marker for MCAD deficiency (medium-chain). "
                "In SCAD deficiency, C8 is NORMAL because medium-chain FAO (MCAD, acting on C6–C12) is INTACT. "
                "SCAD acts only on C4–C6 substrates. "
                "C8 NORMAL = KEY NEGATIVE vs MCAD. "
                "C8 >0.3 µmol/L would strongly suggest MCAD (or combined deficiency) rather than SCAD."
            ),
        },
        "c14_1": {
            "label":     "C14:1 (Tetradecenoylcarnitine)",
            "normal":    "<0.25 µmol/L",
            "status":    "NORMAL — KEY NEGATIVE vs VLCAD",
            "direction": "→ NORMAL",
            "color":     "success",
            "rationale": (
                "C14:1 (tetradecenoylcarnitine) is the PRIMARY NBS marker for VLCAD deficiency (very-long-chain). "
                "In SCAD deficiency, C14:1 is NORMAL because long-chain FAO (VLCAD, acting on C14–C20) is INTACT. "
                "C14:1 NORMAL = KEY NEGATIVE vs VLCAD. "
                "SCAD acylcarnitine profile = C4 ↑ with C8 NORMAL and C14:1 NORMAL is the diagnostic fingerprint."
            ),
        },
        "ema": {
            "label":     "Ethylmalonic Acid (EMA) — Urine OA",
            "normal":    "<20 mmol/mol Cr",
            "status":    "ELEVATED >30–200 mmol/mol Cr (when symptomatic) — VARIABLE; may normalise",
            "direction": "↑ (variable)",
            "color":     "warning",
            "rationale": (
                "Ethylmalonic acid (EMA) is formed from butyryl-CoA via alternative metabolism — "
                "butyryl-CoA undergoes carboxylation to ethylmalonyl-CoA → EMA excreted in urine. "
                "EMA is the most characteristic urine OA in SCAD deficiency, but VARIABLE: "
                "may be elevated during metabolic stress, near-normal between episodes. "
                "EMA elevation also occurs in GLUTARIC ACIDURIA TYPE 2 (MADD/GA2) — important differential. "
                "EMA + methylsuccinic acid (MSA) co-elevated = characteristic SCAD pattern (less EMA elevation than GA2)."
            ),
        },
        "msa": {
            "label":     "Methylsuccinic Acid (MSA) — Urine OA",
            "normal":    "<10 mmol/mol Cr",
            "status":    "ELEVATED >15–100 mmol/mol Cr — co-elevated with EMA",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "Methylsuccinic acid (MSA) is formed from methylmalonyl-CoA via metabolism of accumulated short-chain species. "
                "MSA co-elevation with EMA is the characteristic urine OA pattern in SCAD deficiency. "
                "MSA/EMA ratio can help distinguish SCAD from GA2 (MADD): "
                "GA2 has much higher EMA elevation; SCAD has more balanced EMA/MSA elevation. "
                "MSA is also elevated in some methylmalonic acidaemia variants — C3 should be checked to exclude."
            ),
        },
        "butyrylglycine": {
            "label":     "Butyrylglycine — Urine OA",
            "normal":    "<3 mmol/mol Cr",
            "status":    "ELEVATED >5–50 mmol/mol Cr — glycine conjugate; elevated when symptomatic",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "Butyrylglycine is the glycine conjugate of butyryl-CoA (glycine N-acyltransferase). "
                "Analogous to hexanoylglycine (HG) in MCAD, but for the short-chain substrate. "
                "Less pathognomonic than HG for MCAD — butyrylglycine also produced from dietary sources and gut bacteria. "
                "More elevated in symptomatic / biallelic null SCAD than in common-variant carriers. "
                "Useful as supporting evidence when EMA and MSA are also elevated."
            ),
        },
        "glucose": {
            "label":     "Blood Glucose",
            "normal":    "3.5–6.0 mmol/L",
            "status":    "NEAR NORMAL — mild hypoglycaemia possible in crisis; less severe than MCAD/VLCAD",
            "direction": "→ Near normal (↓ mild in crisis)",
            "color":     "success",
            "rationale": (
                "SCAD deficiency causes mild hypoketotic tendency during fasting stress, but glucose levels "
                "are usually near normal because: "
                "(1) C4–C6 fatty acids contribute only a small fraction of total liver FAO; "
                "(2) MCAD and VLCAD are intact → most FAO-derived acetyl-CoA generation continues; "
                "(3) Ketogenesis is only mildly impaired (short-chain FAO blockade is less energy-critical). "
                "Severe hypoglycaemia is RARE in SCAD — in contrast to MCAD where hypoketotic hypoglycaemia is the hallmark."
            ),
        },
        "bohb": {
            "label":     "β-Hydroxybutyrate (β-OHB) / Ketones",
            "normal":    "<0.5 mmol/L (fed); up to 2.0 mmol/L (normal fasting ketosis)",
            "status":    "MILDLY LOW — mild hypoketotic tendency; less severe than MCAD/VLCAD",
            "direction": "↓ (mild)",
            "color":     "warning",
            "rationale": (
                "SCAD blockade causes only mild impairment of ketogenesis because "
                "C4–C6 fatty acids are a minor substrate for hepatic ketone production. "
                "MCAD and VLCAD (which remain intact in SCAD deficiency) provide most of the acetyl-CoA for ketogenesis. "
                "β-OHB is mildly reduced during fasting/illness in SCAD — NOT the dramatic hypoketosis of MCAD crisis. "
                "CONTRAST: MCAD/VLCAD → HYPOketotic hypoglycaemia (dramatic, life-threatening); "
                "SCAD → mild hypoketotic tendency (less clinically significant)."
            ),
        },
        "c0": {
            "label":     "C0 (Free Carnitine)",
            "normal":    "25–50 µmol/L",
            "status":    "NORMAL or mildly LOW — less depletion than MCAD/VLCAD",
            "direction": "→ Normal / ↓ mild",
            "color":     "success",
            "rationale": (
                "Free carnitine (C0) depletion is LESS PROMINENT in SCAD than in MCAD/VLCAD. "
                "Butyryl-CoA (C4) conjugation to carnitine depletes C0 mildly. "
                "L-carnitine supplementation: only if C0 is clearly depleted (<10 µmol/L); "
                "not universally recommended for SCAD (unlike some MCAD protocols)."
            ),
        },
        "alt": {
            "label":     "ALT (Alanine Aminotransferase)",
            "normal":    "<40 U/L",
            "status":    "NORMAL or MILDLY ELEVATED — hepatopathy is rare in SCAD",
            "direction": "→ Normal / ↑ mild (classic)",
            "color":     "success",
            "rationale": (
                "Hepatopathy is NOT a hallmark of SCAD deficiency (contrast with MCAD where Reye-like hepatopathy is prominent). "
                "Short-chain fatty acids are less directly hepatotoxic than medium-chain fatty acids (octanoate in MCAD). "
                "ALT mildly elevated in classic SCAD with metabolic crises; typically normal in asymptomatic NBS cases. "
                "Significant hepatopathy in a patient with C4 elevation should prompt differential for MCAD or other FAO disorder."
            ),
        },
    }

    phenotype_biomarker_patterns = [
        {
            "phenotype":  "Asymptomatic NBS-detected (common variant)",
            "prevalence": "70%",
            "c4":         "0.5–1.5 µmol/L (mildly elevated)",
            "ema":        "5–40 mmol/mol Cr (variable)",
            "glucose":    "Normal (3.5–6.0 mmol/L)",
            "bohb":       "Normal",
            "variant":    "625G>A and/or 511C>T (susceptibility variants; NOT pathogenic alone)",
            "prognosis":  "Excellent; likely no clinical disease; NBS detection of uncertain significance",
        },
        {
            "phenotype":  "Biochemical SCAD (mildly symptomatic)",
            "prevalence": "20%",
            "c4":         "0.8–2.5 µmol/L",
            "ema":        "20–120 mmol/mol Cr",
            "glucose":    "Near normal (2.8–5.5 mmol/L)",
            "bohb":       "Mildly low (0.2–1.0 mmol/L)",
            "variant":    "Mix of common variants + one true pathogenic allele",
            "prognosis":  "Generally mild; hypotonia and feeding difficulties may improve with age; riboflavin trial",
        },
        {
            "phenotype":  "Classic SCAD (biallelic null alleles — symptomatic)",
            "prevalence": "10%",
            "c4":         "1.2–4.0 µmol/L",
            "ema":        "40–200 mmol/mol Cr",
            "glucose":    "Mildly low (2.0–4.5 mmol/L)",
            "bohb":       "Low (0.1–0.8 mmol/L)",
            "variant":    "Biallelic null alleles (catalytic domain / splice / frameshift)",
            "prognosis":  "Hypotonia + developmental delay + seizures possible; monitor; riboflavin + supportive",
        },
    ]

    treatment_table = [
        {
            "intervention":  "Watchful waiting (asymptomatic NBS)",
            "level":         "Standard — no specific treatment needed for common-variant carriers",
            "rationale":     "Majority of NBS-detected SCAD patients carry 625G>A and/or 511C>T — susceptibility variants; likely no clinical risk",
            "contraindication": None,
        },
        {
            "intervention":  "Riboflavin (Vitamin B2) — 50–200 mg/day",
            "level":         "Level B — trial for symptomatic patients",
            "rationale":     "FAD cofactor supplementation may augment residual SCAD activity (FAD affinity partially restored); some patients show EMA/MSA normalisation",
            "contraindication": None,
        },
        {
            "intervention":  "Avoid prolonged fasting (symptomatic patients)",
            "level":         "Level B — prudent but less critical than in MCAD/VLCAD",
            "rationale":     "SCAD block is less energy-critical than MCAD/VLCAD; hypoglycaemia is rare but prolonged fasting may exacerbate symptoms",
            "contraindication": None,
        },
        {
            "intervention":  "L-Carnitine supplementation",
            "level":         "Level B — only if C0 <10 µmol/L or symptomatic depletion",
            "rationale":     "Replaces mild secondary carnitine depletion; less benefit than in MCAD; not universally recommended",
            "contraindication": None,
        },
        {
            "intervention":  "ACADS gene sequencing",
            "level":         "MANDATORY for all NBS C4 positives",
            "rationale":     "Required to distinguish true pathogenic biallelic LOF from common susceptibility variants (625G>A, 511C>T); guides prognosis and management",
            "contraindication": None,
        },
        {
            "intervention":  "VPA (Valproic acid)",
            "level":         "AVOID in symptomatic SCAD patients",
            "rationale":     "VPA inhibits mitochondrial FAO; worsens short-chain block; depletes carnitine; hepatotoxic risk in FAO disorders",
            "contraindication": "HIGH RISK (symptomatic patients)",
        },
        {
            "intervention":  "Ketogenic Diet",
            "level":         "Relative CI — use with caution in symptomatic SCAD",
            "rationale":     "KD generates short-chain ketone bodies (β-OHB, AcAc); may worsen C4 accumulation; not an absolute CI unlike MCAD/VLCAD",
            "contraindication": "Relative CI (classic/symptomatic SCAD)",
        },
        {
            "intervention":  "NBS panel inclusion",
            "level":         "Controversial — NOT universally recommended",
            "rationale":     "UK removed SCAD from NBS 2012; majority of positives are asymptomatic common-variant carriers; harm from parental anxiety + overtreatment",
            "contraindication": "NBS policy debate — country-specific",
        },
    ]

    return {
        "biomarkers":               biomarkers,
        "phenotype_patterns":       phenotype_biomarker_patterns,
        "treatment_table":          treatment_table,
        "patient_sample":           patients,
        "variant_table":            VARIANTS,
        "key_differentials": {
            "SCAD_vs_MCAD":     "C4 (SCAD) vs C8 (MCAD); MCAD causes serious Reye-like crisis; SCAD usually benign; C8 NORMAL in SCAD",
            "SCAD_vs_VLCAD":    "C4 (SCAD) vs C14:1 (VLCAD); VLCAD → cardiomyopathy + rhabdomyolysis; SCAD rarely serious; C14:1 NORMAL in SCAD",
            "SCAD_vs_GA2":      "EMA elevated in BOTH; GA2 has MUCH higher EMA + C4-C12 acylcarnitine profile; riboflavin response common in GA2 (riboflavin-responsive MADD)",
            "SCAD_vs_IBD":      "C4 elevated in BOTH SCAD and isobutyryl-CoA carboxylase deficiency (IBD); cannot distinguish on standard MS/MS; urine OA + gene panel needed",
            "Common_variants":  "625G>A (7% population) + 511C>T (14% population) = susceptibility variants; NOT pathogenic alone; biallelic NULL required for symptomatic disease",
        },
        "exam_pearls": [
            "C4 (butyrylcarnitine) = PRIMARY NBS MARKER — HIGHLY NONSPECIFIC; many common-variant carriers",
            "625G>A (p.Gly209Ser) = 7% general population; NOT pathogenic alone — NBS controversy cause",
            "511C>T (p.Arg171Trp) = 14% general population; NOT pathogenic alone",
            "C8 NORMAL = KEY NEGATIVE vs MCAD (MCAD elevates C8; SCAD does NOT)",
            "C14:1 NORMAL = KEY NEGATIVE vs VLCAD (VLCAD elevates C14:1; SCAD does NOT)",
            "EMA + MSA + butyrylglycine — characteristic urine OA (variable; may normalise between episodes)",
            "MAJORITY ASYMPTOMATIC — SCAD is the most controversial NBS disorder; UK removed 2012",
            "Riboflavin (B2) trial for symptomatic patients — FAD cofactor may restore residual activity",
            "NO ABSOLUTE CI (unlike MCAD/VLCAD where fasting + KD are absolute CI)",
            "VPA: AVOID in symptomatic patients — worsens FAO",
            "EMA also elevated in GA2 (MADD) — much higher in GA2; riboflavin response common in GA2",
            "ACADS sequencing MANDATORY for all NBS C4 positives — guides prognosis",
            "Hypotonia + developmental delay + seizures in TRUE (biallelic null) SCAD only (~10%)",
            "FAO triad: VLCAD (C14–C20) → MCAD (C6–C12) → SCAD (C4–C6) — completing the chain-length spectrum",
        ],
    }


def get_definitions():
    return {
        "disease_name":   "SCAD Deficiency (Short-Chain Acyl-CoA Dehydrogenase Deficiency)",
        "gene":           "ACADS (also called SCAD)",
        "locus":          "12q24.31",
        "omim_gene":      "*606885",
        "omim_disease":   "#201470",
        "inheritance":    "Autosomal Recessive (AR) — biallelic LOF required for true clinical disease",
        "protein":        "ACADS — 412 aa precursor; mitochondrial matrix; FAD-dependent; homotetrameric",
        "enzymatic_function": (
            "Catalyses Step 1 of mitochondrial beta-oxidation for C4–C6 (short-chain) acyl-CoA species: "
            "Short-chain acyl-CoA (primarily butyryl-CoA/C4) + FAD → trans-2-enoyl-acyl-CoA + FADH2. "
            "FADH2 is transferred to ETF (electron transfer flavoprotein) → ETFDH (ubiquinone oxidoreductase) → "
            "enters respiratory chain at Complex I/III level. "
            "Completes the FAO triad: VLCAD (C14–C20) → MCAD (C6–C12) → SCAD (C4–C6)."
        ),
        "pathway":       "Mitochondrial fatty acid beta-oxidation (Step 1, C4–C6 chain length specificity)",
        "metabolic_block": (
            "ACADS LOF → Short-chain acyl-CoAs (C4–C6) CANNOT be dehydrogenated → "
            "accumulate as free acids and acylcarnitine conjugates (C4 = butyrylcarnitine) → "
            "beta-oxidation spiral arrested at short-chain stage → "
            "minor effect on ketogenesis (C4 contributes small fraction of total FAO); "
            "LESS ENERGY-CRITICAL than MCAD/VLCAD block — medium/long-chain FAO intact."
        ),
        "nbs_marker":    "C4 (butyrylcarnitine) >0.5–1.0 µmol/L on tandem MS/MS NBS — HIGHLY NONSPECIFIC",
        "nbs_controversy": (
            "SCAD is the most controversial disorder on NBS panels. "
            "Common variants 625G>A (p.Gly209Ser; 7% population) and 511C>T (p.Arg171Trp; 14% population) "
            "cause C4 elevation without clinical disease. "
            "UK removed SCAD from NBS panel in 2012. "
            "Other countries maintain SCAD on panels but with strict genotype-based management protocols."
        ),
        "confirmatory_biomarkers": {
            "plasma_acylcarnitines": "C4 ↑; C4/C2 ratio ↑; C4-OH mildly ↑; C8 NORMAL; C14:1 NORMAL",
            "urine_organic_acids":   "Ethylmalonic acid (EMA) ↑; Methylsuccinic acid (MSA) ↑; Butyrylglycine ↑ (variable; may normalise)",
            "enzyme_activity":       "SCAD enzyme activity in lymphocytes/fibroblasts; <10% for true deficiency",
            "molecular":             "ACADS sequencing MANDATORY; 625G>A and 511C>T are common susceptibility variants (not pathogenic alone); true pathogenic = catalytic/splice/null alleles",
        },
        "clinical_features": {
            "asymptomatic":    "Majority: NBS-detected; no clinical disease; common susceptibility variants only",
            "symptomatic":     "Hypotonia, feeding difficulties, developmental delay — in biallelic null allele patients",
            "seizures":        "Epilepsy in ~40% of classic SCAD; less common than in MCAD crisis",
            "metabolic_crisis": "Rare; mild hypoketotic tendency during illness; much less severe than MCAD/VLCAD",
            "hallmark":        "ABSENCE of dramatic clinical phenotype — SCAD deficiency is usually benign",
        },
        "treatments": {
            "asymptomatic":     "Watchful waiting; no specific treatment required for common-variant carriers",
            "riboflavin":       "Level B trial for symptomatic: 50–200 mg/day; FAD cofactor may augment residual SCAD activity",
            "fasting_avoidance":"Level B — prudent for symptomatic; NOT absolute CI unlike MCAD/VLCAD",
            "l_carnitine":      "Level B — only if C0 <10 µmol/L",
            "contraindications": ["VPA (HIGH RISK in symptomatic patients)", "KD (relative CI in classic SCAD)", "NOT ABSOLUTE CI for fasting unlike MCAD/VLCAD"],
        },
        "prevalence": "~1:50,000 (true symptomatic biallelic null deficiency); much higher NBS positive rate due to common susceptibility variants",
        "common_variants": {
            "625G>A (p.Gly209Ser)": "7% allele frequency in general population; susceptibility variant; NOT pathogenic alone; major NBS false-positive cause",
            "511C>T (p.Arg171Trp)": "14% allele frequency in general population; susceptibility variant; NOT pathogenic alone",
        },
        "key_exam_facts": [
            "C4 (butyrylcarnitine) = PRIMARY NBS MARKER — HIGHLY NONSPECIFIC",
            "625G>A (7% population) + 511C>T (14% population) = common susceptibility variants, NOT pathogenic alone",
            "MOST CONTROVERSIAL NBS disorder — UK removed from panel 2012",
            "Majority of NBS C4 positives are ASYMPTOMATIC common-variant carriers",
            "C8 NORMAL = KEY NEGATIVE vs MCAD",
            "C14:1 NORMAL = KEY NEGATIVE vs VLCAD",
            "EMA + MSA + butyrylglycine = characteristic urine OA (variable)",
            "Riboflavin (B2) trial = first-line for symptomatic patients",
            "NO ABSOLUTE CONTRAINDICATIONS (unlike MCAD/VLCAD where fasting + KD are absolute CI)",
            "TRUE symptomatic SCAD (biallelic null) → hypotonia + developmental delay + seizures",
            "ACADS sequencing MANDATORY for all NBS positives — guides management",
            "FAO TRIAD complete: VLCAD (C14–C20) → MCAD (C6–C12) → SCAD (C4–C6)",
        ],
        "glossary": {
            "ACADS":   "Acyl-CoA Dehydrogenase Short-chain gene (12q24.31)",
            "SCAD":    "Short-Chain Acyl-CoA Dehydrogenase enzyme product of ACADS",
            "FAO":     "Fatty Acid Oxidation (mitochondrial beta-oxidation pathway)",
            "C4":      "Butyrylcarnitine — acylcarnitine of 4-carbon (short-chain) fatty acid; PRIMARY NBS marker (nonspecific)",
            "EMA":     "Ethylmalonic acid — characteristic urine OA in SCAD; also elevated in GA2",
            "MSA":     "Methylsuccinic acid — co-elevated with EMA in SCAD urine",
            "ETF":     "Electron Transfer Flavoprotein — receives FADH2 from ACADS in beta-oxidation",
            "ETFDH":   "ETF:Ubiquinone Oxidoreductase — transfers electrons to respiratory chain",
            "NBS":     "Newborn Screening — expanded tandem MS/MS detects C4 elevation",
            "625G>A":  "p.Gly209Ser — common ACADS susceptibility variant; 7% general population; NOT pathogenic alone",
            "511C>T":  "p.Arg171Trp — common ACADS susceptibility variant; 14% general population; NOT pathogenic alone",
            "GA2":     "Glutaric Aciduria Type 2 (MADD) — also elevates EMA; important differential; riboflavin-responsive",
            "IBD":     "Isobutyryl-CoA Carboxylase Deficiency — also elevates C4; cannot distinguish SCAD vs IBD on standard NBS MS/MS",
            "FAD":     "Flavin Adenine Dinucleotide — cofactor of SCAD; riboflavin (B2) is FAD precursor",
            "HYPOketosis": "Mildly inappropriately low ketones — less dramatic than in MCAD crisis; SCAD rarely causes severe HYPOketotic hypoglycaemia",
        },
        "references": [
            "van Maldegem BT et al. (2010). Clinical, biochemical, and genetic heterogeneity in SCAD deficiency. J Inherit Metab Dis 33:1-8.",
            "Jethva R, Ficicioglu C (2008). SCAD deficiency. Mol Genet Metab 95(4):169-173.",
            "Nochi Z et al. (2017). SCAD deficiency: From genomics to clinical symptoms. J Inherit Metab Dis 40(3):371-378.",
            "Andresen BS et al. (2000). Clear relationship between genotype and phenotype in SCAD. AJHG 67(6):1162-1676.",
            "Pedersen CB et al. (2008). SCAD deficiency — an underappreted cause of muscle hypotonia in infancy. Gene 423(1):101-106.",
            "OMIM #201470 — SCAD Deficiency (ACADS gene *606885). omim.org.",
        ],
    }
