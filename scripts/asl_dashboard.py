#!/usr/bin/env python3
"""ASL (Argininosuccinate Lyase) Deficiency — Argininosuccinic Aciduria Dashboard.

ASL encodes Argininosuccinate Lyase, a cytoplasmic homotetrameric enzyme:
  L-Argininosuccinate  →  L-Arginine + Fumarate
  UREA CYCLE STEP 4 OF 5 — cleavage step; produces arginine AND fumarate (TCA link)

ASL DISEASE: Argininosuccinic Aciduria (ASA) / Argininosuccinate Lyase Deficiency
  OMIM Disease: #207900   Gene: ASL, OMIM *608310
  Chromosome: 7q11.21
  Inheritance: Autosomal Recessive (AR)
  Protein: 464 aa; cytoplasmic homotetrameric enzyme; ubiquitous expression
  Prevalence: ~1:70,000 (third most common UCD after OTC and ASS1)

MECHANISM — LOSS-OF-FUNCTION (step 4 block → argininosuccinate CANNOT be cleaved):
  Normal ASL: Argininosuccinate → Arginine + Fumarate
  ASL LOF: Argininosuccinate CANNOT be cleaved → ACCUMULATES MASSIVELY in all body fluids
  Argininosuccinate: VERY HIGH → CRITICALLY HIGH (plasma, urine, CSF) — PATHOGNOMONIC
  Plasma citrulline: ELEVATED (100-300 µmol/L; normal 15-35) — but NOT as high as ASS1 (>500)
  Plasma ammonia: ELEVATED but often milder than neonatal CPS1/OTC (step 4 partial NH3 production)
  Plasma arginine: LOW — not produced downstream (block at step 4; liver ARG1 bypass minor)
  Urine orotic acid: MILDLY ELEVATED (5-15 µmol/mol Cr) — less than OTC, more than CPS1/NAGS
  Fumarate NOT produced — TCA cycle link disrupted; NO energy substrate from urea cycle
  Trichorrhexis nodosa (brittle/fragile hair): 50-70% — UNIQUE among all UCDs; hallmark of ASL
  Hepatomegaly/liver disease: 40-60%; elevated LFTs; portal fibrosis in some
  Systemic hypertension: 40-50% — UNIQUE cardiovascular complication; low nitric oxide (NO) production

POSITION IN UREA CYCLE — ASL AS STEP 4 CLEAVAGE ENZYME:
  NAGS: Glutamate + Acetyl-CoA → NAG [cofactor generator]
  Step 1: NH₃ + CO₂ + 2ATP → [CPS1, requires NAG] → Carbamoyl-P       (mitochondrial)
  Step 2: Carbamoyl-P + Ornithine → [OTC] → Citrulline + Pi             (mitochondrial)
  Step 3: Citrulline + Aspartate + ATP → [ASS1] → Argininosuccinate     (cytoplasmic)
  Step 4: Argininosuccinate → [ASL] → Arginine + Fumarate               (cytoplasmic, BLOCKED)
  Step 5: Arginine → [ARG1] → Ornithine + Urea                         (cytoplasmic)

  ASL BLOCK CONSEQUENCES:
    Argininosuccinate ACCUMULATES MASSIVELY — plasma, urine, CSF; not seen in other UCDs
    Arginine NOT produced from urea cycle → conditionally essential
    Fumarate NOT produced → TCA anaplerosis disrupted; NO production from arginine impaired
    Ornithine cycle INTERRUPTED — ornithine from ARG1 cannot regenerate → cycle arrest
    Nitrogen STUCK as argininosuccinate/argininosuccinylpeptides — hyperammonemia
    NO (nitric oxide) deficiency — systemic hypertension, endothelial dysfunction

ASL BIOCHEMISTRY (LOF → argininosuccinate critically high → step 4 block):
  Plasma argininosuccinate: VERY HIGH — CRITICALLY HIGH (>100 µmol/L; normal undetectable)
                             THE PATHOGNOMONIC BIOMARKER — unique to ASL
  Plasma citrulline:        ELEVATED (100-300 µmol/L; normal 15-35)
                             Lower than ASS1 (>500), higher than CPS1/NAGS (<5)
  Plasma ammonia:           ELEVATED (>150 µmol/L crises; >500 neonatal null; normal <50)
                             Often milder than ASS1/CPS1 due to partial cycle function
  Plasma arginine:          LOW (<30 µmol/L; normal 60-120) — not produced at step 4
                             Less critically low than ASS1 (<10) — some liver bypass
  Urine orotic acid:        MILDLY ELEVATED (5-15 µmol/mol Cr; less than OTC >20)
                             Intermediate between OTC (markedly high) and CPS1/NAGS (normal)
  Urine argininosuccinate:  VERY HIGH — massive urinary excretion; detectable on urine amino acids
  Plasma fumarate:          LOW-NORMAL — NOT produced at step 4 (TCA disruption)
  Plasma aspartate:         MILDLY ELEVATED — substrate feeds ASS1 → argininosuccinate accumulates
  PLP:                      NORMAL — KEY NEGATIVE (not PLP-dependent; vs ALDH7A1/PDE)
  alpha-AASA:               NORMAL — KEY NEGATIVE (vs ALDH7A1-PDE)
  tHcy:                     NORMAL — KEY NEGATIVE (vs CBS/MTHFR)
  MMA:                      NORMAL — KEY NEGATIVE
  GABA:                     NORMAL — KEY NEGATIVE (vs ABAT)
  GHB:                      NORMAL — KEY NEGATIVE (vs SSADH)
  GAA:                      NORMAL — KEY NEGATIVE (vs GAMT)

ASL KEY DISTINCTIONS FROM OTHER UCDs:
  vs ASS1: argininosuccinate VERY HIGH (ASL) vs ABSENT (ASS1) — single metabolite distinguishes
           citrulline elevated but lower in ASL than ASS1
  vs OTC:  argininosuccinate HIGH in ASL, undetectable OTC; orotic MILDLY elevated ASL, MARKEDLY OTC
  vs CPS1: argininosuccinate HIGH in ASL, undetectable CPS1; citrulline elevated ASL, critically low CPS1
  vs NAGS: identical to CPS1 distinction; NCG trial negative in ASL
  Trichorrhexis nodosa: UNIQUE to ASL among all UCDs — brittle hair at nodes
  Hypertension: cardiovascular complication UNIQUE to ASL — NO deficiency
  Liver disease: hepatomegaly/fibrosis 40-60% — more prominent than other UCDs

EPILEPSY AND NEUROLOGICAL FEATURES IN ASL:
  Seizures: 40-50% overall; both from hyperammonemia AND direct argininosuccinate neurotoxicity
  GTCS (generalized tonic-clonic): 35% — MODAL seizure type
  Focal seizures: 25% — temporal/frontal
  Absence: 15% — particularly in attenuated forms with normal ammonia
  Status epilepticus: 15-20% (lower than CPS1/NAGS due to often milder hyperammonemia)
  Drug-resistant epilepsy: 15-20%
  IDD (intellectual/developmental disability): 50-65% — even with good ammonia control
  Neurocognitive impairment even in treated, well-controlled patients — argininosuccinate direct toxicity
  Brain MRI: periventricular leukoencephalopathy, cortical atrophy, basal ganglia signal change
  EEG: diffuse slowing (ammonia), focal temporal slowing (direct neurotoxicity)

TREATMENTS (evidence-graded):
  Level A (established):
    Arginine (high-dose oral): 400-700 mg/kg/day — PRIMARY; replaces conditionally essential arginine;
                               provides fumarate equivalent; supports ornithine cycling via ARG1
    Sodium Benzoate + Phenylacetate/RAVICTI: nitrogen scavenging
    Low-protein diet: reduce nitrogen load
    IV Dextrose 10%: anti-catabolic in acute crises
    CRRT/HD: Level A for NH3 >500 µmol/L — emergent
    Liver transplant: Level A — CURATIVE for hyperammonemia + ammonia/argininosuccinate normalises;
                     BUT trichorrhexis nodosa, hypertension, neurocognitive issues may persist
                     (systemic/endothelial disease not fully reversed by liver transplant — UNIQUE)
  Level B (probable):
    LEV (levetiracetam): first-line AED; no hepatotoxicity; preferred
    Sildenafil/NO augmentation: Level B experimental; targets NO deficiency → hypertension
    Citrulline supplementation: Level B; alternative to arginine in some protocols
  Absolute Contraindications:
    Valproate (VPA): ABSOLUTE CI — inhibits NAGS/CPS1; hyperammonemia risk; hepatotoxic (multiple UCDs)
    High-protein diet: ABSOLUTE CI — nitrogen overload
    L-Asparaginase: ABSOLUTE CI — hyperammonemia via asparagine/glutamine catabolism
    Prolonged fasting: HIGH RISK — catabolism

PHENOTYPIC CLASSES:
  Classic Neonatal (40%): null/severe variants; NH3 >500 µmol/L day 1-3; CRRT mandatory
  Late-Onset/Episodic (35%): residual 5-25% activity; episodic crises; trichorrhexis nodosa prominent
  Mild/NBS-Detected (25%): attenuated; detected by elevated citrulline/argininosuccinate on NBS

TOP PATHOGENIC VARIANTS (ASL gene):
  p.Arg95His       (active site, ~15%, severe neonatal — most common worldwide)
  p.Arg385Cys      (catalytic domain, ~12%, moderate-severe)
  p.Leu209Ser      (~10%, severe neonatal)
  p.Arg193Trp      (~9%, moderate)
  c.IVS11+1G>A    (splice null, ~8%, severe)
  p.Ala398Thr      (attenuated, ~7%, NBS-detected)
  p.Gln286Arg      (~6%, moderate)
  p.Val335Met      (mild hypomorphic, ~5%, attenuated)

UNIQUE ASL SYSTEMIC FEATURES (beyond ammonia):
  Trichorrhexis nodosa: fragile hair at nodes; shaft splits; 50-70%; responds partially to arginine
  Systemic hypertension: 40-50%; from impaired NO synthesis (ASL is also a component of NO cycle)
  Portal hypertension/liver fibrosis: 20-30% long-term; not typically seen in OTC/CPS1/NAGS
  Cognitive impairment despite NH3 control: argininosuccinate neurotoxic beyond ammonia effects
"""

import random

SEED = 217
random.seed(SEED)

N_PATIENTS = 40

# Phenotypic classes
PHENOTYPE_CLASSES = [
    {"name": "Classic Neonatal",    "pct": 40, "residual_activity": "0-5%",   "severity": "severe"},
    {"name": "Late-Onset/Episodic", "pct": 35, "residual_activity": "5-25%",  "severity": "moderate"},
    {"name": "Mild/NBS-Detected",   "pct": 25, "residual_activity": "25-50%", "severity": "mild"},
]

# Biomarker reference ranges and disease values
BIOMARKERS = {
    "plasma_argininosuccinate": {
        "label": "Plasma Argininosuccinate",
        "normal": "Undetectable (<1 µmol/L)",
        "disease": "VERY HIGH — CRITICALLY HIGH (>100 µmol/L; often >500 neonatal)",
        "direction": "↑↑↑ PATHOGNOMONIC",
        "status": "CRITICALLY ELEVATED",
        "color": "danger",
    },
    "plasma_citrulline": {
        "label": "Plasma Citrulline",
        "normal": "15-35 µmol/L",
        "disease": "ELEVATED (100-300 µmol/L; lower than ASS1 >500, higher than CPS1/NAGS <5)",
        "direction": "↑↑ ELEVATED",
        "status": "HIGH",
        "color": "warning",
    },
    "plasma_ammonia": {
        "label": "Plasma Ammonia",
        "normal": "<50 µmol/L",
        "disease": "ELEVATED — >500 neonatal null; >150 crisis; often milder than ASS1/CPS1",
        "direction": "↑↑ ELEVATED",
        "status": "HIGH (often milder than other UCDs)",
        "color": "warning",
    },
    "plasma_arginine": {
        "label": "Plasma Arginine",
        "normal": "60-120 µmol/L",
        "disease": "LOW (<30 µmol/L; not produced at step 4; conditionally essential)",
        "direction": "↓ LOW",
        "status": "LOW — CONDITIONALLY ESSENTIAL",
        "color": "info",
    },
    "urine_orotic_acid": {
        "label": "Urine Orotic Acid",
        "normal": "<6 µmol/mol Cr",
        "disease": "MILDLY ELEVATED (5-15 µmol/mol Cr; less than OTC >20, more than CPS1/NAGS normal)",
        "direction": "↑ MILDLY ELEVATED",
        "status": "INTERMEDIATE",
        "color": "warning",
    },
    "urine_argininosuccinate": {
        "label": "Urine Argininosuccinate",
        "normal": "Undetectable",
        "disease": "VERY HIGH — massive urinary excretion (pathognomonic on urine amino acids)",
        "direction": "↑↑↑ PATHOGNOMONIC",
        "status": "CRITICALLY ELEVATED",
        "color": "danger",
    },
    "plp": {
        "label": "PLP (Pyridoxal-5-Phosphate)",
        "normal": "20-100 nmol/L",
        "disease": "NORMAL — KEY NEGATIVE (ASL is not PLP-dependent; vs ALDH7A1/PDE)",
        "direction": "→ NORMAL",
        "status": "NORMAL — KEY NEGATIVE",
        "color": "success",
    },
    "alpha_aasa": {
        "label": "Alpha-AASA",
        "normal": "<1 µmol/mol Cr",
        "disease": "NORMAL — KEY NEGATIVE (vs ALDH7A1-PDE)",
        "direction": "→ NORMAL",
        "status": "NORMAL — KEY NEGATIVE",
        "color": "success",
    },
    "thcy": {
        "label": "Total Homocysteine (tHcy)",
        "normal": "<15 µmol/L",
        "disease": "NORMAL — KEY NEGATIVE (vs CBS/MTHFR)",
        "direction": "→ NORMAL",
        "status": "NORMAL — KEY NEGATIVE",
        "color": "success",
    },
    "mma": {
        "label": "Methylmalonic Acid (MMA)",
        "normal": "<0.4 µmol/L",
        "disease": "NORMAL — KEY NEGATIVE",
        "direction": "→ NORMAL",
        "status": "NORMAL — KEY NEGATIVE",
        "color": "success",
    },
    "gaba": {
        "label": "GABA (CSF/plasma)",
        "normal": "Normal range",
        "disease": "NORMAL — KEY NEGATIVE (vs ABAT where GABA HIGH)",
        "direction": "→ NORMAL",
        "status": "NORMAL — KEY NEGATIVE",
        "color": "success",
    },
    "ghb": {
        "label": "Gamma-Hydroxybutyrate (GHB)",
        "normal": "<5 µmol/L",
        "disease": "NORMAL — KEY NEGATIVE (vs SSADH where GHB VERY HIGH)",
        "direction": "→ NORMAL",
        "status": "NORMAL — KEY NEGATIVE",
        "color": "success",
    },
}

# Pathogenic variants
VARIANTS = [
    {"variant": "p.Arg95His",   "domain": "Active site",        "freq": 15, "phenotype": "Classic severe neonatal", "note": "Most common worldwide; abolishes catalysis"},
    {"variant": "p.Arg385Cys",  "domain": "Catalytic domain",   "freq": 12, "phenotype": "Moderate-severe",          "note": "Second most common; partial residual activity"},
    {"variant": "p.Leu209Ser",  "domain": "Tetramer interface",  "freq": 10, "phenotype": "Severe neonatal",          "note": "Disrupts tetramer assembly"},
    {"variant": "p.Arg193Trp",  "domain": "Substrate binding",  "freq":  9, "phenotype": "Moderate",                 "note": "Argininosuccinate binding pocket"},
    {"variant": "c.IVS11+1G>A", "domain": "Splice site (null)", "freq":  8, "phenotype": "Severe neonatal",          "note": "Null allele; complete loss of protein"},
    {"variant": "p.Ala398Thr",  "domain": "C-terminal",         "freq":  7, "phenotype": "Attenuated/NBS-detected",  "note": "Residual ~25% activity; NBS cohort"},
    {"variant": "p.Gln286Arg",  "domain": "Core beta-sheet",    "freq":  6, "phenotype": "Moderate",                 "note": "Structural destabilisation"},
    {"variant": "p.Val335Met",  "domain": "Hypomorphic",        "freq":  5, "phenotype": "Mild attenuated",          "note": "Minimal enzyme impairment; often NBS-detected"},
]

# Treatment table
TREATMENTS = [
    {
        "therapy": "Arginine (high-dose oral)",
        "level": "A",
        "dose": "400-700 mg/kg/day",
        "rationale": "PRIMARY therapy; replaces conditionally essential arginine; provides fumarate equivalent; ornithine cycling via ARG1",
        "class": "Amino acid supplementation",
    },
    {
        "therapy": "Sodium Benzoate + Phenylacetate (RAVICTI)",
        "level": "A",
        "dose": "Standard nitrogen-scavenger dosing",
        "rationale": "Alternative nitrogen disposal; reduces ammonia load",
        "class": "Nitrogen scavenger",
    },
    {
        "therapy": "Low-protein diet",
        "level": "A",
        "dose": "Disease-specific protein prescription",
        "rationale": "Reduce nitrogen load; prevent hyperammonemia episodes",
        "class": "Dietary management",
    },
    {
        "therapy": "IV Dextrose 10% (anti-catabolic)",
        "level": "A",
        "dose": "GIR 6-8 mg/kg/min acute",
        "rationale": "Prevents catabolism in crisis; suppresses endogenous nitrogen release",
        "class": "Metabolic stabilisation",
    },
    {
        "therapy": "CRRT / Haemodialysis",
        "level": "A",
        "dose": "NH3 >500 µmol/L",
        "rationale": "Emergency ammonia removal; faster than nitrogen scavengers alone",
        "class": "Extracorporeal",
    },
    {
        "therapy": "Liver Transplant",
        "level": "A",
        "dose": "Curative timing (before severe brain injury)",
        "rationale": "CURATIVE for hyperammonemia; normalises argininosuccinate + citrulline; "
                     "NOTE: systemic features (trichorrhexis, hypertension, neurocognition) may persist",
        "class": "Definitive therapy",
    },
    {
        "therapy": "LEV (Levetiracetam)",
        "level": "B",
        "dose": "20-60 mg/kg/day",
        "rationale": "First-line AED; no hepatotoxicity; preferred in all UCDs",
        "class": "Antiseizure medication",
    },
    {
        "therapy": "Sildenafil / NO augmentation",
        "level": "B",
        "dose": "Experimental; case series",
        "rationale": "Targets NO deficiency → systemic hypertension; ASL is part of NO cycle",
        "class": "Vascular / experimental",
    },
    {
        "therapy": "Valproate (VPA)",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "Inhibits NAGS → no NAG → CPS1 off; catastrophic hyperammonemia; hepatotoxic in UCDs",
        "class": "Contraindicated AED",
    },
    {
        "therapy": "High-protein diet",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "Nitrogen overload → acute decompensation",
        "class": "Dietary contraindication",
    },
    {
        "therapy": "L-Asparaginase",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "Asparagine/glutamine catabolism → acute hyperammonemia",
        "class": "Contraindicated drug",
    },
]

# Seizure types in ASL
SEIZURE_TYPES = [
    {"type": "GTCS (generalised tonic-clonic)", "pct": 35, "note": "MODAL; ammonia encephalopathy + direct toxicity"},
    {"type": "Focal seizures",                  "pct": 25, "note": "Temporal/frontal; direct argininosuccinate neurotoxicity"},
    {"type": "Absence seizures",                "pct": 15, "note": "Particularly attenuated forms with normal ammonia"},
    {"type": "Myoclonic encephalopathy",         "pct": 12, "note": "Neonatal onset; burst-suppression EEG"},
    {"type": "Status epilepticus",               "pct": 18, "note": "Often first presentation; crisis; NH3 >300"},
    {"type": "Drug-resistant epilepsy",          "pct": 17, "note": "Both ammonia-mediated AND argininosuccinate direct toxicity"},
]

# Unique systemic features
SYSTEMIC_FEATURES = [
    {"feature": "Trichorrhexis nodosa (brittle/fragile hair)", "pct": 60, "note": "UNIQUE to ASL among all UCDs; hair shaft nodes; responds partially to arginine"},
    {"feature": "Systemic hypertension",                       "pct": 45, "note": "UNIQUE cardiovascular complication; impaired NO synthesis from arginine deficiency"},
    {"feature": "Hepatomegaly / liver disease",                "pct": 50, "note": "Elevated LFTs; portal fibrosis 20-30%; more than other UCDs"},
    {"feature": "Intellectual/developmental disability (IDD)", "pct": 57, "note": "Even with good ammonia control; argininosuccinate directly neurotoxic"},
    {"feature": "Periventricular leukoencephalopathy",         "pct": 35, "note": "Brain MRI; white matter signal change"},
    {"feature": "Protein aversion",                           "pct": 50, "note": "Behavioural clue to underlying hyperammonemia; late-onset phenotype"},
]


def _generate_cohort():
    random.seed(SEED)
    patients = []
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls["name"]] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append("Late-Onset/Episodic")
    phenotype_dist = phenotype_dist[:N_PATIENTS]
    random.shuffle(phenotype_dist)

    for i in range(N_PATIENTS):
        phenotype = phenotype_dist[i]
        if phenotype == "Classic Neonatal":
            age_onset = round(random.uniform(0, 5), 1)    # days 0-5
            nh3_peak  = random.randint(500, 1200)
            as_plasma = random.randint(400, 900)
            citrulline = random.randint(150, 350)
        elif phenotype == "Late-Onset/Episodic":
            age_onset = round(random.uniform(2, 36), 1)   # months 2-36
            nh3_peak  = random.randint(150, 400)
            as_plasma = random.randint(100, 400)
            citrulline = random.randint(80, 200)
        else:  # Mild/NBS
            age_onset = round(random.uniform(0, 30), 1)
            nh3_peak  = random.randint(50, 150)
            as_plasma = random.randint(40, 120)
            citrulline = random.randint(60, 150)

        arginine_level = random.randint(8, 30)
        orotic = round(random.uniform(3, 16), 1)
        hair_nodosa = phenotype != "Classic Neonatal" or random.random() < 0.4
        seizures = random.random() < 0.45
        htn = random.random() < 0.45
        liver = random.random() < 0.50
        idd = random.random() < 0.57

        v = random.choice(VARIANTS)
        patients.append({
            "id": f"ASL-{i+1:03d}",
            "phenotype": phenotype,
            "age_onset_months": age_onset,
            "nh3_peak_umol_l": nh3_peak,
            "argininosuccinate_plasma": as_plasma,
            "citrulline_umol_l": citrulline,
            "arginine_umol_l": arginine_level,
            "orotic_acid_urine": orotic,
            "trichorrhexis_nodosa": hair_nodosa,
            "seizures": seizures,
            "hypertension": htn,
            "liver_disease": liver,
            "idd": idd,
            "variant": v["variant"],
        })
    return patients


COHORT = _generate_cohort()


def get_overview():
    n = len(COHORT)
    n_seizures    = sum(1 for p in COHORT if p["seizures"])
    n_htn         = sum(1 for p in COHORT if p["hypertension"])
    n_hair        = sum(1 for p in COHORT if p["trichorrhexis_nodosa"])
    n_liver       = sum(1 for p in COHORT if p["liver_disease"])
    n_idd         = sum(1 for p in COHORT if p["idd"])
    n_neonatal    = sum(1 for p in COHORT if p["phenotype"] == "Classic Neonatal")
    n_late        = sum(1 for p in COHORT if p["phenotype"] == "Late-Onset/Episodic")
    n_mild        = sum(1 for p in COHORT if p["phenotype"] == "Mild/NBS-Detected")
    avg_nh3       = round(sum(p["nh3_peak_umol_l"] for p in COHORT) / n)
    avg_as        = round(sum(p["argininosuccinate_plasma"] for p in COHORT) / n)
    avg_citr      = round(sum(p["citrulline_umol_l"] for p in COHORT) / n)
    avg_arg       = round(sum(p["arginine_umol_l"] for p in COHORT) / n)

    return {
        "disease": "Argininosuccinic Aciduria (ASA) — ASL Deficiency",
        "omim_gene": "608310",
        "omim_disease": "207900",
        "gene": "ASL",
        "chromosome": "7q11.21",
        "inheritance": "Autosomal Recessive (AR)",
        "protein": "464 aa; cytoplasmic homotetrameric enzyme",
        "prevalence": "~1:70,000 (third most common UCD after OTC and ASS1)",
        "urea_cycle_step": "Step 4 of 5 — Argininosuccinate → Arginine + Fumarate (cleavage)",
        "n_patients": n,
        "kpi": {
            "n_patients":         {"value": n,                       "label": "Cohort size",           "color": "#7b1fa2"},
            "seizures_pct":       {"value": f"{round(n_seizures/n*100)}%",  "label": "Seizures (%)",  "color": "#b71c1c"},
            "trichorrhexis_pct":  {"value": f"{round(n_hair/n*100)}%",      "label": "Trichorrhexis nodosa (UNIQUE)", "color": "#e65100"},
            "htn_pct":            {"value": f"{round(n_htn/n*100)}%",       "label": "Hypertension (UNIQUE)", "color": "#880e4f"},
            "liver_pct":          {"value": f"{round(n_liver/n*100)}%",     "label": "Liver disease (%)","color": "#1a237e"},
            "idd_pct":            {"value": f"{round(n_idd/n*100)}%",       "label": "IDD (%)",          "color": "#006064"},
            "avg_as":             {"value": f"{avg_as} µmol/L",             "label": "Mean Argininosuccinate", "color": "#c62828"},
            "avg_nh3":            {"value": f"{avg_nh3} µmol/L",            "label": "Mean Peak NH3",    "color": "#d84315"},
            "avg_citr":           {"value": f"{avg_citr} µmol/L",           "label": "Mean Citrulline",  "color": "#4a148c"},
            "avg_arg":            {"value": f"{avg_arg} µmol/L",            "label": "Mean Arginine",    "color": "#2e7d32"},
            "neonatal_pct":       {"value": f"{round(n_neonatal/n*100)}%",  "label": "Classic Neonatal (%)","color": "#37474f"},
        },
        "phenotype_dist": [
            {"class": "Classic Neonatal",    "n": n_neonatal, "pct": round(n_neonatal/n*100)},
            {"class": "Late-Onset/Episodic", "n": n_late,     "pct": round(n_late/n*100)},
            {"class": "Mild/NBS-Detected",   "n": n_mild,     "pct": round(n_mild/n*100)},
        ],
        "hallmark_biomarker": "Plasma AND urine argininosuccinate VERY HIGH — PATHOGNOMONIC (undetectable in health + all other UCDs)",
        "key_distinction_ass1": "Argininosuccinate VERY HIGH in ASL vs ABSENT in ASS1 — single metabolite differentiates",
        "unique_systemic_features": "Trichorrhexis nodosa (50-70%) + Systemic hypertension (40-50%) UNIQUE to ASL among all UCDs",
        "liver_transplant_caveat": "CURATIVE for hyperammonemia but systemic features (trichorrhexis, hypertension, neurocognition) may persist — UNIQUE to ASL",
        "vpa_contraindication": "VPA ABSOLUTE CI — inhibits NAGS/CPS1; catastrophic hyperammonemia in ALL UCDs",
        "seizure_mechanism": "DUAL: hyperammonemia-mediated + direct argininosuccinate neurotoxicity (unique vs OTC/CPS1/NAGS)",
    }


def get_breakdown():
    return {
        "biomarkers": BIOMARKERS,
        "variants": VARIANTS,
        "treatments": TREATMENTS,
        "seizure_types": SEIZURE_TYPES,
        "systemic_features": SYSTEMIC_FEATURES,
        "phenotype_classes": PHENOTYPE_CLASSES,
        "cohort_preview": COHORT[:10],
        "urea_cycle_context": {
            "step": 4,
            "total_steps": 5,
            "enzyme": "Argininosuccinate Lyase",
            "reaction": "Argininosuccinate → Arginine + Fumarate",
            "location": "Cytoplasmic (like ASS1)",
            "cofactor": "None (not PLP-dependent)",
            "upstream_blocked": "Argininosuccinate accumulates (from ASS1 step 3)",
            "downstream_deficit": "Arginine NOT produced; Fumarate NOT produced (TCA link broken)",
            "no_cycle_disruption": "ASL also part of NO synthesis cycle → NO deficiency → hypertension",
        },
        "differential_diagnosis": {
            "vs_ass1": {
                "key_diff": "Argininosuccinate: VERY HIGH in ASL, ABSENT in ASS1",
                "citrulline": "Elevated (100-300) in ASL; CRITICALLY HIGH (>500) in ASS1",
                "orotic": "Mildly elevated in ASL; normal/mild in ASS1",
            },
            "vs_otc": {
                "key_diff": "Argininosuccinate present in ASL, absent in OTC; orotic MARKEDLY HIGH OTC vs MILDLY elevated ASL",
                "citrulline": "Elevated ASL; critically LOW OTC",
            },
            "vs_cps1_nags": {
                "key_diff": "Argininosuccinate present in ASL, absent in CPS1/NAGS; citrulline elevated ASL vs critically low CPS1/NAGS",
                "ncg_trial": "No response to NCG in ASL (vs complete response NAGS, partial CPS1)",
            },
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "ASL (Argininosuccinate Lyase, OMIM *608310) encodes a 464-amino-acid cytoplasmic "
            "homotetrameric enzyme that catalyses step 4 of the urea cycle: "
            "L-Argininosuccinate → L-Arginine + Fumarate. "
            "ASL is ALSO a component of the arginine/NO synthesis cycle — impairment reduces "
            "nitric oxide (NO) production → systemic hypertension and endothelial dysfunction. "
            "Inheritance: autosomal recessive; chromosome 7q11.21."
        ),
        "pathomechanism": (
            "ASL LOF → argininosuccinate CANNOT be cleaved → accumulates massively in plasma, "
            "urine, CSF, and tissues. Dual consequence: (1) hyperammonemia from urea cycle block; "
            "(2) direct argininosuccinate neurotoxicity (independent of NH3 level). "
            "Arginine NOT produced downstream → conditionally essential; high-dose arginine is PRIMARY therapy. "
            "Fumarate NOT produced → TCA anaplerosis disrupted. "
            "NO synthesis impaired (ASL is also in arginine/NO cycle) → systemic hypertension. "
            "Trichorrhexis nodosa: mechanism involves low NO in hair follicle; responds partially to arginine."
        ),
        "biomarker_pattern": (
            "PATHOGNOMONIC: Plasma argininosuccinate VERY HIGH (>100 µmol/L; undetectable in health and ALL other UCDs). "
            "Urine argininosuccinate VERY HIGH. "
            "Citrulline ELEVATED (100-300 µmol/L) — intermediate between ASS1 (>500) and CPS1/NAGS (<5). "
            "NH3 elevated but often milder than CPS1/OTC. "
            "Arginine LOW (<30 µmol/L). "
            "Orotic acid MILDLY elevated (5-15 µmol/mol Cr). "
            "KEY NEGATIVES: PLP normal, alpha-AASA normal, tHcy normal, MMA normal, GABA normal, GHB normal."
        ),
        "key_distinction_from_ass1": (
            "ASL vs ASS1 — single metabolite distinguishes: "
            "ASL: argininosuccinate VERY HIGH (substrate ACCUMULATES at step 4 block). "
            "ASS1: argininosuccinate ABSENT (cannot be MADE at step 3 block). "
            "Citrulline in ASL (100-300) is lower than ASS1 (>500). "
            "Both are cytoplasmic, autosomal recessive — the metabolite pattern is decisive."
        ),
        "unique_systemic_features": (
            "ASL is UNIQUE among all UCDs for two systemic features not seen in OTC/CPS1/NAGS/ASS1: "
            "1. TRICHORRHEXIS NODOSA (50-70%): brittle hair with nodes; hair shaft splits; "
            "   may improve with arginine therapy (NO restoration in follicles). "
            "2. SYSTEMIC HYPERTENSION (40-50%): from NO deficiency; ASL is part of arginine/NO cycle; "
            "   vascular endothelial dysfunction; may require antihypertensive therapy. "
            "Liver disease (40-60%) also more prominent than other UCDs; portal fibrosis long-term."
        ),
        "liver_transplant_note": (
            "Liver transplant is Level A curative for hyperammonemia component; normalises "
            "argininosuccinate, citrulline, and ammonia. HOWEVER, unlike OTC/CPS1/NAGS, "
            "trichorrhexis nodosa, systemic hypertension, and pre-existing neurocognitive impairment "
            "may NOT fully resolve post-transplant — because ASL also functions outside the liver "
            "(endothelial, renal, CNS). This persistence of systemic features is UNIQUE to ASL."
        ),
        "seizure_management": (
            "Seizures in ASL have DUAL mechanism: (1) hyperammonemia-mediated (shared with all UCDs); "
            "(2) DIRECT argininosuccinate neurotoxicity (unique to ASL — independent of NH3 level). "
            "This means seizures may persist even with good ammonia control. "
            "EEG: diffuse slowing (ammonia), focal temporal slowing (direct toxicity), "
            "burst-suppression in neonatal crisis. "
            "AED: LEV first-line (no hepatotoxicity). NEVER VPA (catastrophic hyperammonemia). "
            "Treat ammonia FIRST; add AED for persistent/breakthrough seizures."
        ),
        "ar_inheritance_note": (
            "AR inheritance: both parents obligate heterozygous carriers; 25% recurrence risk per pregnancy. "
            "Males and females equally affected — contrast with OTC (X-linked). "
            "Heterozygous ASL carriers: typically asymptomatic. "
            "Cascade carrier testing after index diagnosis. "
            "Newborn screening (NBS): elevated citrulline AND argininosuccinate on tandem MS → urgent evaluation. "
            "~1:70,000 worldwide; third most common UCD after OTC (~1:14,000-50,000) and ASS1 (~1:57,000)."
        ),
        "unique_features_vs_other_ucd": (
            "1. ARGININOSUCCINATE VERY HIGH — the pathognomonic hallmark (absent in ALL other UCDs). "
            "2. DUAL SEIZURE MECHANISM — hyperammonemia + direct argininosuccinate neurotoxicity. "
            "3. TRICHORRHEXIS NODOSA — UNIQUE among UCDs; hair fragility; responsive to arginine. "
            "4. SYSTEMIC HYPERTENSION — UNIQUE among UCDs; NO cycle disruption; cardiovascular risk. "
            "5. LIVER TRANSPLANT CAVEAT — systemic features may persist; unique among UCDs. "
            "6. CYTOPLASMIC enzyme (like ASS1) — contrasts with mitochondrial CPS1/OTC/NAGS. "
            "7. FUMARATE deficit — TCA anaplerosis disrupted; energy metabolism impact. "
            "8. THIRD MOST COMMON UCD — ~1:70,000; clinically distinct from other UCDs by hair/BP."
        ),
    }
