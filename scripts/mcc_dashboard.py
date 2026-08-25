#!/usr/bin/env python3
"""MCC (3-Methylcrotonyl-CoA Carboxylase) Deficiency — 3-MCC Deficiency Dashboard.

3-MCC is a biotin-dependent mitochondrial enzyme (heterodimer of MCCC1 + MCCC2 subunits)
catalysing step 3 of leucine catabolism:
    3-Methylcrotonyl-CoA + CO₂ + ATP  →  [MCC]  →  3-Methylglutaconyl-CoA + ADP + Pi

MCC LOF → 3-methylcrotonyl-CoA CANNOT be carboxylated → accumulates:
  → 3-Methylcrotonylglycine (3-MCG): ELEVATED in urine — PRIMARY PATHOGNOMONIC MARKER
  → 3-Hydroxy-isovalerate (3-HIV): ELEVATED in urine (side-product of CoA hydrolysis → oxidation)
  → 3-Hydroxy-isovalerylcarnitine (C5-OH): ELEVATED — NBS PRIMARY MARKER
  → 3-Methylcrotonate: accumulates in plasma (volatile, minor)
  → Free carnitine: LOW (secondary — conjugation to C5-OH depletes pool)

KEY FACTS (EXAM HIGHEST-YIELD):
  1. MOST COMMONLY DETECTED organic acidemia on NBS in many countries
  2. MAJORITY are ASYMPTOMATIC (benign variant) — penetrance is low
  3. MATERNAL 3-MCC DEFICIENCY can flag on infant NBS (metabolites cross placenta)
  4. NOT BIOTIN-RESPONSIVE (KEY distinction from HLCS and biotinidase deficiency)
  5. NBS C5-OH differential: 3-MCC vs HMGCL vs HLCS vs BTD vs MCD
  6. 3-MCG on urine OA = PATHOGNOMONIC — distinguishes from all C5-OH causes except MCCC mutations
  7. Leucine restriction + L-carnitine if symptomatic; majority need no treatment
  8. VPA: AVOID (carnitine depletion + CoA sequestration worsens metabolic stress)

OMIM Disease: #210200 (3-Methylcrotonylglycinuria I — MCCC1) / #210210 (type II — MCCC2)
OMIM Genes:  *609010 (MCCC1) / *609014 (MCCC2)
Chromosomes: 3q27.1 (MCCC1 / alpha / biotin-binding subunit) · 5q13.2 (MCCC2 / beta subunit)
Inheritance: Autosomal Recessive (AR), biallelic LOF in MCCC1 or MCCC2
Protein:     MCCC1 = 725 aa (alpha: biotin-carboxylase domain, BCCP biotin-binding domain)
             MCCC2 = 563 aa (beta: carboxyl-transferase domain)
             Native enzyme: α₆β₆ dodecamer; biotin covalently attached to MCCC1 Lys-688
Prevalence:  ~1:30,000–50,000 (NBS era); highest NBS detection rate of any OA in Portugal/Spain

LEUCINE CATABOLISM CONTEXT (where MCC fits):
  L-Leucine  → BCAT (Step 1a) → KIC (α-ketoisocaproate)
  KIC        → BCKDH (Step 1b) → Isovaleryl-CoA
  Isovaleryl-CoA → IVD (Step 2) → 3-Methylcrotonyl-CoA   [IVD deficiency upstream]
  3-MC-CoA   → MCC (Step 3) → 3-Methylglutaconyl-CoA    [THIS ENZYME]
  3-MG-CoA   → AUH (Step 4) → HMG-CoA
  HMG-CoA    → HMGCL (Step 5) → Acetoacetate + Acetyl-CoA

BIOMARKER PATTERN — 3-MCC DEFICIENCY:
  C5-OH (3-OH-isovalerylcarnitine)  ELEVATED (>0.4 µmol/L) — NBS TIER-1 PRIMARY MARKER
  3-MCG (3-methylcrotonylglycine)   ELEVATED (>50 mmol/mol Cr) — PATHOGNOMONIC in urine OA
  3-HIV (3-hydroxy-isovalerate)     ELEVATED (>30 mmol/mol Cr) — secondary marker
  Free carnitine                    LOW (secondary depletion via C5-OH conjugation)
  Plasma leucine                    NORMAL or mildly elevated (catabolism backed up)
  NH3, lactate, glucose             NORMAL (key negatives vs UCDs, PDH deficiency)
  C3 (propionylcarnitine)           NORMAL — KEY NEGATIVE vs HLCS (where C3 ↑ too)
  C6-DC (glutarylcarnitine)         ABSENT — KEY NEGATIVE vs GCDH
  MMA                               ABSENT/NORMAL — KEY NEGATIVE vs MMUT/PA
  Biotinidase activity              NORMAL — KEY NEGATIVE vs BTD
  Total homocysteine                NORMAL — KEY NEGATIVE vs MMACHC/cblC

NBS C5-OH DIFFERENTIAL (CRITICAL):
  3-MCC (MCCC1/2): C5-OH ↑ + 3-MCG ↑ + 3-HIV ↑; C3 NORMAL; Biotin-unresponsive → most common
  HMGCL:           C5-OH ↑ + C6-DC (HMG-CoA-derived) ↑ + hypoglycaemia; no 3-MCG
  HLCS:            C5-OH ↑ + C3 ↑ + C5 (3-HIA?) + MULTIPLE carboxylase defs; BIOTIN-RESPONSIVE
  BTD:             C5-OH ↑ + skin rash + hearing loss + optic atrophy; BIOTIN-RESPONSIVE
  MCD:             Rare; C5-OH ↑ + C3 ↑; BIOTIN-RESPONSIVE (HCS defect in biotin attachment)

PATHOGENESIS — DUAL ACCUMULATION:
  Block at MCC → 3-methylcrotonyl-CoA accumulates → two fates:
  (1) CoA hydrolysis → 3-methylcrotonate → hydroxylation → 3-hydroxy-isovalerate (3-HIV)
  (2) Glycine conjugation → 3-methylcrotonylglycine (3-MCG) — predominant detox route
  C5-OH (3-OH-isovalerylcarnitine) formed via carnitine conjugation of 3-HIV-CoA → depletes carnitine
  Unlike IVD (where toxic isovaleric acid causes acute odour crises), the accumulating
  metabolites in 3-MCC are LESS ACUTELY TOXIC → explains high proportion of asymptomatic cases

TREATMENT:
  Leucine restriction: 1st line if symptomatic (reduce substrate flux)
  L-carnitine:        if secondary depletion documented (100–200 mg/kg/day)
  Biotin:             DOES NOT HELP — MCC is not biotin-responsive (contrast HLCS/BTD)
  Glycine:            Not indicated (contrast IVD where glycine is cornerstone)
  VPA:                AVOID — carnitine depletion + CoA sequestration
  Fasting:            AVOID — increases leucine catabolism flux → metabolic decompensation
  Most NBS-detected cases: OBSERVE ONLY; no dietary restriction needed

PHENOTYPES:
  Asymptomatic / benign:    50–60% (NBS-detected, mother-detected; no clinical disease)
  Classic symptomatic:      25–30% — acute metabolic crises (hypoglycaemia, encephalopathy)
  Attenuated / intermediate: 10–15% — developmental delay, hypotonia, occasional crises

SEIZURES IN 3-MCC:
  ~25–30% in symptomatic phenotype; usually secondary to metabolic decompensation
  No specific epilepsy syndrome; focal/multifocal > generalised
  Resolve with metabolic correction; drug-resistant epilepsy (DRE) uncommon (~10%)
  EEG: non-specific slowing during crises; usually normalises inter-ictally

COMMON PATHOGENIC VARIANTS — MCCC1 (alpha subunit, biotin-binding):
  p.Arg385Cys (c.1153C>T): 18% alleles — BC domain; severe; most common worldwide
  p.Arg288* (c.862C>T):    14% — null; severe; pan-ethnic
  p.Arg385His (c.1154G>A): 12% — BC domain; moderate; often asymptomatic
  p.Ser346Ile (c.1037G>T): 10% — BCCP domain; moderate; European
  p.Ile361Thr (c.1082T>C):  8% — near biotin-attachment; moderate

COMMON PATHOGENIC VARIANTS — MCCC2 (beta subunit, carboxyl-transferase):
  p.Val428Met (c.1282G>A): 20% — CT domain; classic; Portuguese founder
  p.Leu157His (c.470T>A):  16% — CT domain; severe; pan-ethnic
  p.Arg174Gln (c.521G>A):  12% — CT domain; moderate; Korean
  p.Thr354Met (c.1061C>T): 10% — CT domain; attenuated; Japanese

MATERNAL 3-MCC DEFICIENCY (UNIQUE FEATURE):
  Asymptomatic mother with 3-MCC deficiency → metabolites (C5-OH, 3-MCG) cross placenta
  → elevate in infant's blood → infant's NBS flags as 3-MCC deficiency
  → infant is heterozygous carrier, NOT affected
  → Distinguish: test MOTHER; infant levels normalise by 2–4 weeks; infant enzyme/gene normal
  → ~30% of 3-MCC NBS detections in some programmes are MATERNAL (secondary)
"""

import random

SEED       = 259      # next after IVD (seed 253), continuing series
N_PATIENTS = 40

# ── Phenotype classes ─────────────────────────────────────────────────────────
PHENOTYPE_CLASSES = [
    {
        "class": "Asymptomatic / Benign (NBS-detected or maternal)",
        "pct": 55,
        "age_onset_months_range": (0, 1),      # detected at NBS; no clinical onset
        "c5oh_range": (0.5, 2.5),              # µmol/L
        "mcg_urine_range": (20, 100),           # mmol/mol Cr (may be borderline)
        "hiv_urine_range": (15, 60),
        "free_carn_range": (25, 45),
        "leucine_range": (90, 160),
        "nh3_range": (20, 45),
        "seizures_prob": 0.02,
        "crises_range": (0, 0),
        "dd_prob": 0.05,
        "note": "NBS or maternal detection; no symptoms; often observe-only",
    },
    {
        "class": "Classic Symptomatic",
        "pct": 28,
        "age_onset_months_range": (2, 18),
        "c5oh_range": (2.5, 12.0),
        "mcg_urine_range": (80, 500),
        "hiv_urine_range": (50, 300),
        "free_carn_range": (8, 22),
        "leucine_range": (160, 400),
        "nh3_range": (20, 90),
        "seizures_prob": 0.35,
        "crises_range": (1, 6),
        "dd_prob": 0.55,
        "note": "Acute crises (hypoglycaemia, encephalopathy, hypotonia); febrile/fasting trigger",
    },
    {
        "class": "Attenuated / Intermediate",
        "pct": 17,
        "age_onset_months_range": (6, 36),
        "c5oh_range": (0.8, 3.5),
        "mcg_urine_range": (30, 180),
        "hiv_urine_range": (20, 120),
        "free_carn_range": (15, 30),
        "leucine_range": (120, 280),
        "nh3_range": (20, 60),
        "seizures_prob": 0.18,
        "crises_range": (0, 2),
        "dd_prob": 0.30,
        "note": "Developmental delay, hypotonia; occasional mild crises; incomplete penetrance",
    },
]


def _gen_cohort():
    """Generate synthetic 40-patient cohort with MCC-realistic biomarker profiles."""
    random.seed(SEED)
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append(PHENOTYPE_CLASSES[0])
    phenotype_dist = phenotype_dist[:N_PATIENTS]

    patients = []
    for i in range(N_PATIENTS):
        cls = phenotype_dist[i]
        pid = f"MCC-{i+1:03d}"

        # Gene/subunit
        gene = random.choice(["MCCC1", "MCCC1", "MCCC2"])   # MCCC1 slightly more common

        # Variant selection
        if gene == "MCCC1":
            variants = [
                "p.Arg385Cys", "p.Arg288*", "p.Arg385His",
                "p.Ser346Ile", "p.Ile361Thr",
            ]
        else:
            variants = [
                "p.Val428Met", "p.Leu157His", "p.Arg174Gln",
                "p.Thr354Met", "p.Leu434Pro",
            ]
        variant = random.choice(variants)

        onset_mo = round(random.uniform(*cls["age_onset_months_range"]), 1)
        c5oh     = round(random.uniform(*cls["c5oh_range"]), 2)
        mcg      = round(random.uniform(*cls["mcg_urine_range"]), 1)
        hiv      = round(random.uniform(*cls["hiv_urine_range"]), 1)
        carn     = round(random.uniform(*cls["free_carn_range"]), 1)
        leu      = round(random.uniform(*cls["leucine_range"]), 1)
        nh3      = round(random.uniform(*cls["nh3_range"]), 1)
        seiz     = random.random() < cls["seizures_prob"]
        n_cris   = random.randint(*cls["crises_range"])
        dd       = random.random() < cls["dd_prob"]
        biotin_tried = (cls["class"] == "Classic Symptomatic") and random.random() < 0.4
        maternal = (cls["class"] == "Asymptomatic / Benign (NBS-detected or maternal)") and random.random() < 0.28

        patients.append({
            "id": pid,
            "gene": gene,
            "phenotype": cls["class"],
            "variant": variant,
            "maternal_detection": maternal,
            "age_onset_months": onset_mo,
            "c5oh_umol_l": c5oh,
            "mcg_urine_mmol_mol_cr": mcg,
            "hiv_urine_mmol_mol_cr": hiv,
            "free_carnitine_umol_l": carn,
            "plasma_leucine_umol_l": leu,
            "nh3_umol_l": nh3,
            "seizures": seiz,
            "crisis_count": n_cris,
            "developmental_delay": dd,
            "biotin_tried_no_response": biotin_tried,
        })
    return patients


_COHORT = _gen_cohort()


def get_overview():
    pheno_counts = {}
    for p in _COHORT:
        k = p["phenotype"]
        pheno_counts[k] = pheno_counts.get(k, 0) + 1

    pheno_dist = [
        {"class": k, "n": v, "pct": round(v / N_PATIENTS * 100)}
        for k, v in pheno_counts.items()
    ]

    n_seiz  = sum(1 for p in _COHORT if p["seizures"])
    n_dd    = sum(1 for p in _COHORT if p["developmental_delay"])
    n_cris  = sum(1 for p in _COHORT if p["crisis_count"] > 0)
    n_mat   = sum(1 for p in _COHORT if p["maternal_detection"])
    n_low_c = sum(1 for p in _COHORT if p["free_carnitine_umol_l"] < 20)
    n_bio   = sum(1 for p in _COHORT if p["biotin_tried_no_response"])
    avg_c5  = round(sum(p["c5oh_umol_l"] for p in _COHORT) / N_PATIENTS, 2)
    avg_mcg = round(sum(p["mcg_urine_mmol_mol_cr"] for p in _COHORT) / N_PATIENTS, 1)

    return {
        "disease": "3-MCC Deficiency (3-Methylcrotonyl-CoA Carboxylase Deficiency / 3-Methylcrotonylglycinuria)",
        "gene": "MCCC1 (alpha) / MCCC2 (beta)",
        "omim_gene": "609010 (MCCC1) / 609014 (MCCC2)",
        "omim_disease": "210200 (type I) / 210210 (type II)",
        "locus": "3q27.1 (MCCC1) / 5q13.2 (MCCC2)",
        "inheritance": "Autosomal Recessive",
        "prevalence": "~1:30,000–50,000 (NBS era); most common OA on NBS in some countries",
        "pathway_step": "Leucine catabolism step 3 (after IVD) — 3-methylcrotonyl-CoA → 3-methylglutaconyl-CoA",
        "n_patients": N_PATIENTS,
        "seed": SEED,
        "kpi": {
            "n_patients":    {"label": "Cohort", "value": str(N_PATIENTS), "color": "#1b5e20"},
            "seizures":      {"label": "Seizures", "value": f"{n_seiz} ({round(n_seiz/N_PATIENTS*100)}%)", "color": "#b71c1c"},
            "crises":        {"label": "Metabolic Crises", "value": f"{n_cris} ({round(n_cris/N_PATIENTS*100)}%)", "color": "#e65100"},
            "maternal":      {"label": "Maternal Detection", "value": f"{n_mat} ({round(n_mat/N_PATIENTS*100)}%)", "color": "#6a1b9a"},
            "low_carnitine": {"label": "Low Carnitine", "value": f"{n_low_c} ({round(n_low_c/N_PATIENTS*100)}%)", "color": "#0d47a1"},
            "dev_delay":     {"label": "Dev Delay", "value": f"{n_dd} ({round(n_dd/N_PATIENTS*100)}%)", "color": "#827717"},
            "avg_c5oh":      {"label": "Avg C5-OH (µmol/L)", "value": str(avg_c5), "color": "#0d47a1"},
            "avg_mcg":       {"label": "Avg 3-MCG (mmol/mol Cr)", "value": str(avg_mcg), "color": "#4a148c"},
        },
        "phenotype_dist": pheno_dist,
        "no_biotin_response_note": (
            "3-MCC Deficiency is NOT biotin-responsive. Unlike HLCS (holocarboxylase synthetase) and "
            "biotinidase deficiency (BTD), where supplemental biotin restores enzyme activity, MCC apoenzyme "
            "itself is structurally defective. Biotin supplementation has NO effect on C5-OH, 3-MCG, or "
            "clinical outcomes in MCCC1/2 deficiency. This is the single most important differential point "
            f"vs HLCS/BTD on NBS. In this cohort, {n_bio} symptomatic patients had biotin trialled — none responded."
        ),
        "maternal_detection_note": (
            "Maternal 3-MCC deficiency is a major source of false-positive NBS: an asymptomatic heterozygous "
            "or homozygous mother with undiagnosed 3-MCC deficiency excretes C5-OH and 3-MCG into breast milk "
            "and/or transplacentally elevates infant blood C5-OH at birth. The infant is typically an obligate "
            "heterozygous carrier — NOT affected. Resolution: test mother (plasma amino acids + urine OA + gene panel); "
            "infant levels normalise by 3–4 weeks if maternal origin confirmed."
        ),
        "hallmark_biomarker": (
            "C5-OH (3-hydroxy-isovalerylcarnitine) ELEVATED on NBS (PRIMARY MARKER, >0.4 µmol/L). "
            "3-Methylcrotonylglycine (3-MCG) ELEVATED in urine OA (PATHOGNOMONIC, >50 mmol/mol Cr). "
            "3-Hydroxy-isovalerate (3-HIV) ELEVATED. Free carnitine LOW (secondary depletion). "
            "C3 NORMAL (KEY NEGATIVE vs HLCS). Biotinidase activity NORMAL (KEY NEGATIVE vs BTD). "
            "MMA ABSENT (KEY NEGATIVE vs MMUT). NH3 NORMAL (KEY NEGATIVE vs UCDs)."
        ),
        "nbs_c5oh_note": (
            "C5-OH on NBS triggers an expanded differential: 3-MCC (most common by far), HMGCL, "
            "HLCS, BTD, MCD. Urine organic acids (3-MCG, 3-HIV) and C3/C6-DC acylcarnitine profile "
            "narrow the diagnosis. Biotinidase assay differentiates BTD. Maternal blood spot or "
            "plasma/urine rules out maternal 3-MCC deficiency as cause of infant's positive screen."
        ),
    }


def get_breakdown():
    biomarkers = {
        "c5oh": {
            "label": "C5-OH (3-OH-isovalerylcarnitine)",
            "normal": "<0.4 µmol/L",
            "status": "ELEVATED (NBS PRIMARY MARKER) — >0.5–12 µmol/L",
            "direction": "↑↑",
            "color": "danger",
            "rationale": (
                "3-Methylcrotonyl-CoA CANNOT be carboxylated → accumulates → CoA hydrolysis → "
                "3-hydroxy-isovalerate → esterified with carnitine → C5-OH (3-OH-isovalerylcarnitine). "
                "Renal excretion depletes free carnitine. C5-OH is the NBS Tier-1 primary marker for "
                "3-MCC. Threshold: >0.4–0.6 µmol/L on Tier-1 screen (varies by programme)."
            ),
        },
        "mcg": {
            "label": "3-Methylcrotonylglycine (3-MCG)",
            "normal": "<5 mmol/mol Cr",
            "status": "ELEVATED PATHOGNOMONIC — >50–500 mmol/mol Cr in urine OA",
            "direction": "↑↑ PATHOGNOMONIC",
            "color": "danger",
            "rationale": (
                "Glycine conjugation of accumulated 3-methylcrotonyl-CoA → 3-MCG. "
                "This is the most specific urinary marker for 3-MCC deficiency — no other IEM produces "
                "3-MCG as the predominant glycine conjugate. Essential for differentiating 3-MCC from "
                "all other C5-OH-elevating conditions. Quantify by urine organic acid GC-MS."
            ),
        },
        "hiv": {
            "label": "3-Hydroxy-isovalerate (3-HIV)",
            "normal": "<10 mmol/mol Cr",
            "status": "ELEVATED — >30–300 mmol/mol Cr in urine OA",
            "direction": "↑",
            "color": "warning",
            "rationale": (
                "Secondary metabolite: 3-methylcrotonyl-CoA → hydrolysis → 3-methylcrotonate → "
                "omega-hydroxylation → 3-hydroxy-isovalerate. Less specific than 3-MCG (also seen in "
                "HMGCL, HLCS, BTD) but quantitatively elevated in 3-MCC. Supports diagnosis alongside 3-MCG."
            ),
        },
        "free_carnitine": {
            "label": "Free Carnitine (C0)",
            "normal": "25–55 µmol/L",
            "status": "LOW — secondary depletion via C5-OH conjugation",
            "direction": "↓",
            "color": "warning",
            "rationale": (
                "Carnitine conjugates 3-OH-isovaleryl-CoA → C5-OH → renally excreted, depleting free "
                "carnitine pool. Secondary carnitine deficiency is common in symptomatic 3-MCC. "
                "L-carnitine supplementation (100–200 mg/kg/day) replenishes pool + enhances C5-OH excretion."
            ),
        },
        "c3": {
            "label": "C3 (Propionylcarnitine)",
            "normal": "<5 µmol/L",
            "status": "NORMAL — KEY NEGATIVE vs HLCS",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "Propionyl-CoA carboxylase (PCC) requires biotin. In HLCS (holocarboxylase synthetase "
                "deficiency) ALL biotin-dependent carboxylases fail → C3 (PCC defect) + C5-OH (MCC defect) "
                "BOTH elevated. In isolated 3-MCC, only MCC is defective → C3 NORMAL. C3 NORMAL on NBS "
                "acylcarnitine profile is a key discriminator: 3-MCC vs HLCS."
            ),
        },
        "nh3": {
            "label": "Ammonia (NH3)",
            "normal": "<50 µmol/L",
            "status": "NORMAL (may be mildly elevated in crisis) — KEY NEGATIVE vs UCDs",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "MCC block does not involve the urea cycle. NH3 is NORMAL or only transiently mildly "
                "elevated during severe metabolic decompensation. Persistent hyperammonaemia suggests a "
                "urea cycle disorder, not 3-MCC deficiency."
            ),
        },
        "mma": {
            "label": "Methylmalonic Acid (MMA)",
            "normal": "<5 µmol/L plasma / <3 mmol/mol Cr urine",
            "status": "ABSENT / NORMAL — KEY NEGATIVE vs MMUT/MMACHC",
            "direction": "→ ABSENT",
            "color": "success",
            "rationale": (
                "MMA is NOT produced in the leucine catabolism pathway. MMA absent in 3-MCC — "
                "key distinguisher from methylmalonyl-CoA mutase (MMUT) deficiency and combined "
                "MMA + homocystinuria (MMACHC cblC type). Urine organic acids showing 3-MCG without "
                "MMA confirm 3-MCC deficiency."
            ),
        },
        "biotinidase": {
            "label": "Biotinidase Activity",
            "normal": ">30 nmol/min/mL",
            "status": "NORMAL — KEY NEGATIVE vs BTD",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "Biotinidase deficiency (BTD) causes secondary MCC deficiency (biotin recycling fails → "
                "MCC apoenzyme cannot be biotinylated) with C5-OH ↑ and rash/hearing loss/optic atrophy. "
                "Biotinidase is NORMAL in primary MCCC1/2 deficiency. Biotinidase assay differentiates "
                "BTD from primary 3-MCC on NBS."
            ),
        },
    }

    enzyme_mechanism = {
        "function": (
            "3-Methylcrotonyl-CoA Carboxylase (MCC) is a biotin-dependent mitochondrial enzyme "
            "forming an α₆β₆ dodecamer (MCCC1 = alpha/biotin-carboxylase+BCCP subunit; MCCC2 = beta/carboxyl-transferase subunit). "
            "Biotin is covalently attached to MCCC1 Lys-688 by holocarboxylase synthetase (HLCS) — "
            "this biotinylation step is normal in primary 3-MCC deficiency."
        ),
        "reaction": (
            "3-Methylcrotonyl-CoA + HCO₃⁻ + ATP  →  [MCC/MCCC1+MCCC2]  →  "
            "3-Methylglutaconyl-CoA + ADP + Pi    (Step 3, leucine catabolism)"
        ),
        "block": (
            "MCCC1 or MCCC2 LOF → MCC holoenzyme non-functional (even with normal HLCS biotinylation). "
            "3-Methylcrotonyl-CoA accumulates → cannot be carboxylated to 3-methylglutaconyl-CoA. "
            "Dual overflow: (1) glycine conjugation → 3-MCG (predominant detox); "
            "(2) CoA hydrolysis + omega-hydroxylation → 3-HIV → carnitine ester → C5-OH."
        ),
        "not_biotin_responsive": (
            "Unlike HLCS (where HLCS fails to attach biotin to MCC apoenzyme, "
            "so biotin supplementation restores full activity via mass action) or BTD (where biotin recycling fails), "
            "in MCCC1/MCCC2 deficiency the apoenzyme is structurally mutant. "
            "Adding more biotin does NOT restore catalytic activity. Biotin DOES NOT HELP. "
            "This is the single most testable differential point in clinical exams."
        ),
        "leucine_path": (
            "L-Leucine → BCAT → KIC → BCKDH → Isovaleryl-CoA → [IVD, step 2] → "
            "3-Methylcrotonyl-CoA → [MCC BLOCKED, step 3] → "
            "↛ 3-Methylglutaconyl-CoA → ↛ HMG-CoA → ↛ Acetoacetate + Acetyl-CoA"
        ),
    }

    variants = [
        # MCCC1 variants
        {"gene": "MCCC1", "variant": "p.Arg385Cys", "freq": 18, "domain": "BC domain (biotin-carboxylase)", "phenotype": "Severe; most common worldwide; often classic symptomatic", "note": "Active-site adjacent; impairs carboxylation chemistry"},
        {"gene": "MCCC1", "variant": "p.Arg288*",   "freq": 14, "domain": "Null (premature stop)", "phenotype": "Severe; pan-ethnic; classic symptomatic", "note": "Complete loss; no residual MCC activity"},
        {"gene": "MCCC1", "variant": "p.Arg385His",  "freq": 12, "domain": "BC domain", "phenotype": "Moderate; often asymptomatic on NBS", "note": "Partial activity retained; lower C5-OH"},
        {"gene": "MCCC1", "variant": "p.Ser346Ile",  "freq": 10, "domain": "BCCP domain (biotin-binding)", "phenotype": "Moderate; European; variable penetrance", "note": "Reduces biotin accessibility to biotin-carboxylase domain"},
        # MCCC2 variants
        {"gene": "MCCC2", "variant": "p.Val428Met",  "freq": 20, "domain": "CT domain (carboxyl-transferase)", "phenotype": "Classic; Portuguese/Iberian founder; symptomatic", "note": "Highest frequency MCCC2 variant; CT active site"},
        {"gene": "MCCC2", "variant": "p.Leu157His",  "freq": 16, "domain": "CT domain", "phenotype": "Severe; pan-ethnic; classic", "note": "Disrupts hydrophobic core of beta subunit"},
        {"gene": "MCCC2", "variant": "p.Arg174Gln",  "freq": 12, "domain": "CT domain", "phenotype": "Moderate; Korean founder; attenuated", "note": "Partial CT activity retained"},
    ]

    seizure_types = [
        {"type": "Metabolic encephalopathy / seizure cluster (crisis)", "pct": 70, "note": "During decompensation; hypoglycaemia-driven; resolve with glucose + carnitine"},
        {"type": "Focal cortical seizures (inter-ictal)", "pct": 20, "note": "Secondary to chronic metabolic stress; LEV first-line"},
        {"type": "Infantile spasms / hypsarrhythmia (rare)", "pct": 5,  "note": "Rare; early-onset severe cases; ACTH/vigabatrin if confirmed IS"},
        {"type": "Drug-resistant epilepsy (DRE)", "pct": 10, "note": "Uncommon; usually resolves with metabolic control + optimised diet"},
    ]

    treatments = [
        {
            "therapy": "Leucine-restricted diet + MCC-free amino-acid formula",
            "level": "A",
            "dose": "Natural protein 0.8–1.5 g/kg/day; leucine <100–150 mg/kg/day",
            "rationale": "Reduces 3-methylcrotonyl-CoA substrate flux → less 3-MCG + C5-OH accumulation. Most effective in symptomatic phenotypes.",
        },
        {
            "therapy": "L-Carnitine",
            "level": "A",
            "dose": "100–200 mg/kg/day oral; IV during crises (50 mg/kg loading)",
            "rationale": "Secondary carnitine depletion from C5-OH conjugation. Replenishes free carnitine → protects energy metabolism + drives C5-OH excretion.",
        },
        {
            "therapy": "IV Glucose (anti-catabolic emergency)",
            "level": "A",
            "dose": "10–15 mg/kg/min GIR during acute crisis; nil-by-mouth if vomiting",
            "rationale": "Suppresses endogenous leucine release from muscle proteolysis → reduces substrate flux through blocked MCC pathway. First-line in metabolic decompensation.",
        },
        {
            "therapy": "Levetiracetam (LEV) — seizure management",
            "level": "B",
            "dose": "20–40 mg/kg/day in 2–3 divided doses",
            "rationale": "First-line AED for 3-MCC-associated seizures. No interaction with biotin or carnitine metabolism. Fewer metabolic adverse effects than VPA.",
        },
        {
            "therapy": "Biotin supplementation",
            "level": "NOT EFFECTIVE",
            "dose": "N/A — do not use",
            "rationale": "3-MCC deficiency is NOT biotin-responsive. MCCC1/2 apoenzyme is structurally defective. Biotin does not restore MCC activity. Test only to EXCLUDE BTD/HLCS as the cause.",
        },
        {
            "therapy": "Glycine supplementation",
            "level": "NOT INDICATED",
            "dose": "N/A",
            "rationale": "Unlike IVD (isovaleric acidemia) where glycine is the CORNERSTONE of treatment, glycine supplementation has no established benefit in 3-MCC deficiency. 3-MCG conjugation is sufficient; glycine does NOT deplete meaningfully.",
        },
        {
            "therapy": "Valproic Acid (VPA)",
            "level": "AVOID",
            "dose": "Contraindicated in symptomatic cases",
            "rationale": "VPA depletes carnitine (via valproyl-carnitine excretion) → worsens secondary carnitine deficiency. VPA also inhibits mitochondrial beta-oxidation + CoA sequestration → metabolic decompensation risk. Use LEV instead.",
        },
        {
            "therapy": "Fasting avoidance",
            "level": "A",
            "dose": "Max 4–6 h (infant), 8–10 h (child); glucose polymer at home for illness",
            "rationale": "Fasting → endogenous leucine release from muscle protein → 3-methylcrotonyl-CoA surge → crisis. All symptomatic families must have written emergency glucose protocol.",
        },
        {
            "therapy": "Observe only (asymptomatic NBS-detected)",
            "level": "A",
            "dose": "No dietary or pharmacological intervention; monitor annually",
            "rationale": "~55% of NBS-detected 3-MCC patients (especially p.Arg385His + maternal detections) are clinically silent. Unnecessary leucine restriction in asymptomatic infants risks growth failure. Shared decision-making; follow metabolic team guidance.",
        },
    ]

    systemic_features = [
        {"feature": "Hypotonia (generalised)",        "pct": 45, "note": "Common in symptomatic phenotype; resolves with metabolic control in most"},
        {"feature": "Developmental delay",             "pct": 30, "note": "Mild-moderate; mostly in classic/intermediate phenotypes"},
        {"feature": "Metabolic encephalopathy (crisis)","pct": 28, "note": "Acute: lethargy, vomiting, hypoglycaemia during febrile/catabolic stress"},
        {"feature": "Hypoglycaemia",                   "pct": 25, "note": "Non-ketotic (unlike HMGCL); complicates crisis management; IV glucose essential"},
        {"feature": "Failure to thrive",               "pct": 20, "note": "Early infancy in classic phenotype; catches up with treatment"},
        {"feature": "Seizures",                        "pct": 18, "note": "Secondary to metabolic crises; usually controlled with metabolic management + LEV"},
        {"feature": "Feeding difficulties",            "pct": 18, "note": "Vomiting during crisis; protein aversion in some"},
        {"feature": "Hepatomegaly (transient)",        "pct": 12, "note": "During acute crisis; normalises with treatment; not chronic"},
        {"feature": "Rash or hair loss",               "pct": 2,  "note": "NOT expected — seen in BTD/HLCS; ABSENCE of rash supports primary 3-MCC over BTD"},
    ]

    cohort_preview = _COHORT[:10]

    return {
        "biomarkers": biomarkers,
        "enzyme_mechanism": enzyme_mechanism,
        "variants": variants,
        "seizure_types": seizure_types,
        "treatments": treatments,
        "systemic_features": systemic_features,
        "cohort_preview": cohort_preview,
    }


def get_definitions():
    return {
        "3-MCC / MCC": (
            "3-Methylcrotonyl-CoA Carboxylase. Biotin-dependent mitochondrial enzyme catalysing "
            "step 3 of leucine catabolism: 3-methylcrotonyl-CoA + CO₂ + ATP → 3-methylglutaconyl-CoA + ADP + Pi. "
            "Composed of MCCC1 (alpha/BC+BCCP subunit, 3q27.1) and MCCC2 (beta/CT subunit, 5q13.2). "
            "Native holoenzyme = α₆β₆ dodecamer."
        ),
        "3-MCC Deficiency": (
            "Autosomal recessive IEM caused by biallelic LOF in MCCC1 or MCCC2. "
            "OMIM: 210200 (type I, MCCC1) / 210210 (type II, MCCC2). Most common OA detected by NBS "
            "in several countries. Penetrance is LOW — majority of NBS cases are asymptomatic. "
            "NOT biotin-responsive (contrast HLCS/BTD). Leucine catabolism step 3 block."
        ),
        "3-MCG (3-Methylcrotonylglycine)": (
            "Pathognomonic urinary biomarker for 3-MCC deficiency. Formed by glycine conjugation of "
            "accumulated 3-methylcrotonyl-CoA. Detected on urine organic acid GC-MS. "
            "Reference: <5 mmol/mol Cr. In 3-MCC: >50–500 mmol/mol Cr. "
            "No other IEM produces 3-MCG as the predominant glycine conjugate. "
            "Essential for differentiating 3-MCC from all other causes of C5-OH elevation on NBS."
        ),
        "C5-OH (3-Hydroxy-isovalerylcarnitine)": (
            "Primary NBS marker for 3-MCC deficiency (and HMGCL, HLCS, BTD). "
            "Formed by carnitine conjugation of 3-hydroxy-isovaleryl-CoA (derived from 3-methylcrotonyl-CoA). "
            "Threshold on dried blood spot: >0.4–0.6 µmol/L. Elevates free carnitine, depleting it. "
            "C5-OH is NOT specific for 3-MCC — urine OA (3-MCG) is required to confirm."
        ),
        "3-HIV (3-Hydroxy-isovalerate)": (
            "Secondary urinary marker in 3-MCC deficiency. "
            "Formed by omega-hydroxylation of 3-methylcrotonate (which arises from hydrolysis of "
            "3-methylcrotonyl-CoA). Less specific than 3-MCG (also elevated in HMGCL, HLCS, BTD). "
            "Reference: <10 mmol/mol Cr. In 3-MCC: >30–300 mmol/mol Cr."
        ),
        "NOT Biotin-Responsive (KEY DISTINCTION)": (
            "Biotin supplementation DOES NOT correct 3-MCC deficiency. The MCC holoenzyme itself is "
            "structurally defective (MCCC1 or MCCC2 mutation). Biotin's role is to be attached to "
            "MCCC1 by HLCS — this step is NORMAL in primary 3-MCC deficiency. "
            "Contrast: HLCS deficiency (biotin attachment fails → biotin restores activity) and "
            "BTD deficiency (biotin recycling fails → biotin supplementation curative). "
            "Clinically: no improvement in C5-OH, 3-MCG, or symptoms after biotin trial in MCCC1/2 patients."
        ),
        "Maternal 3-MCC Deficiency": (
            "An asymptomatic or mildly affected mother with 3-MCC deficiency transfers elevated C5-OH "
            "and 3-MCG transplacentally or via breast milk → infant's NBS screen flags as 3-MCC positive. "
            "The infant is an obligate heterozygous carrier (not affected). Distinguish by: "
            "(1) test mother — she will have elevated C5-OH/3-MCG/MCCC1 or MCCC2 mutation; "
            "(2) infant's levels normalise by 3–4 weeks; "
            "(3) infant gene panel: heterozygous single variant only. "
            "Prevalence: ~25–30% of 3-MCC NBS-positive cases in some programmes are maternal secondary detections."
        ),
        "C5-OH NBS Differential": (
            "When C5-OH is elevated on NBS, consider:\n"
            "1. 3-MCC deficiency (MCCC1/2) — most common; 3-MCG ↑↑; C3 NORMAL; biotin-unresponsive\n"
            "2. HMGCL deficiency — HMG-CoA lyase; C5-OH ↑ + C6-DC ↑ (HMG-related); hypoglycaemia ↑\n"
            "3. HLCS (holocarboxylase synthetase) — biotin-responsive; C5-OH ↑ + C3 ↑ (all carboxylases affected)\n"
            "4. BTD (biotinidase deficiency) — biotin-responsive; biotinidase assay LOW; rash + hearing loss\n"
            "5. Maternal 3-MCC — check mother; infant normalises; infant heterozygous"
        ),
        "Leucine Catabolism Pathway (Step 3 context)": (
            "L-Leucine (essential BCAA) catabolism:\n"
            "Step 1a: L-Leucine → KIC (α-ketoisocaproate) via BCAT (mitochondrial/cytosolic)\n"
            "Step 1b: KIC → Isovaleryl-CoA + CO₂ via BCKDH complex (TPP-dependent)\n"
            "Step 2:  Isovaleryl-CoA → 3-Methylcrotonyl-CoA via IVD (FAD-dependent ACAD) — IVD deficiency upstream\n"
            "Step 3:  3-Methylcrotonyl-CoA → 3-Methylglutaconyl-CoA via MCC [BLOCKED HERE in 3-MCC deficiency]\n"
            "Step 4:  3-Methylglutaconyl-CoA → HMG-CoA via AUH (enoyl-CoA hydratase)\n"
            "Step 5:  HMG-CoA → Acetoacetate + Acetyl-CoA via HMGCL — HMGCL deficiency is downstream"
        ),
        "Inheritance & Epidemiology": (
            "Autosomal recessive (AR); biallelic LOF in MCCC1 OR MCCC2. "
            "Prevalence: ~1:30,000–50,000 (NBS era). In Portugal, highest detected frequency of any OA. "
            "Penetrance is low — many homozygous/compound-het individuals remain asymptomatic lifelong. "
            "MCCC1 and MCCC2 mutations are equally distributed across populations; "
            "certain founder mutations (MCCC2 p.Val428Met in Iberian; MCCC2 p.Arg174Gln in Korean) cluster regionally."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN keys ===", list(get_breakdown().keys()))
    print("\n=== DEFINITIONS keys ===", list(get_definitions().keys()))
    print(f"\n✓ MCC dashboard: {N_PATIENTS} patients, seed={SEED}")
