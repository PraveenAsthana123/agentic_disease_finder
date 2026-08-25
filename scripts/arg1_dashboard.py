#!/usr/bin/env python3
"""ARG1 (Arginase-1) Deficiency — Hyperargininemia Dashboard.

ARG1 encodes Arginase-1, a cytoplasmic homotrimeric enzyme:
  L-Arginine  →  L-Ornithine + Urea
  UREA CYCLE STEP 5 OF 5 — the FINAL step; releases urea; regenerates ornithine

ARG1 DISEASE: Hyperargininemia / Arginase-1 Deficiency
  OMIM Disease: #207800   Gene: ARG1, OMIM *608313
  Chromosome: 6q23.2
  Inheritance: Autosomal Recessive (AR)
  Protein: 322 aa; cytoplasmic homotrimeric enzyme; binds two Mn²⁺ ions per monomer
  Prevalence: ~1:300,000–1,000,000 (rarest classical UCD among known disorders)

MECHANISM — LOSS-OF-FUNCTION (step 5 block → arginine CANNOT be hydrolysed):
  Normal ARG1: Arginine → Ornithine + Urea  (releases urea nitrogen; regenerates ornithine)
  ARG1 LOF: Arginine CANNOT be hydrolysed → ACCUMULATES MASSIVELY in plasma
  Ornithine NOT regenerated → urea cycle slows from substrate depletion at step 2
  Urea NOT produced → nitrogen STUCK as arginine pool
  Steps 1-4 (CPS1, OTC, ASS1, ASL) remain INTACT → arginine still flows in from cycle + diet

POSITION IN UREA CYCLE — ARG1 AS STEP 5 FINAL ENZYME:
  NAGS: Glutamate + Acetyl-CoA → NAG [cofactor generator]
  Step 1: NH₃ + CO₂ + 2ATP → [CPS1, requires NAG] → Carbamoyl-P       (mitochondrial)
  Step 2: Carbamoyl-P + Ornithine → [OTC] → Citrulline + Pi             (mitochondrial)
  Step 3: Citrulline + Aspartate + ATP → [ASS1] → Argininosuccinate     (cytoplasmic)
  Step 4: Argininosuccinate → [ASL] → Arginine + Fumarate               (cytoplasmic)
  Step 5: Arginine → [ARG1] → Ornithine + Urea                          (cytoplasmic, BLOCKED)

  ARG1 BLOCK CONSEQUENCES:
    Arginine ACCUMULATES MASSIVELY — plasma, urine, CSF; PATHOGNOMONIC
    Ornithine NOT regenerated — urea cycle slowed from step 2 (OTC requires ornithine)
    Urea NOT produced from cycle — nitrogen disposal impaired
    Hyperammonemia MILD vs other UCDs — steps 1-4 intact; arginine pool buffers NH3
    Guanidino compounds ELEVATED — alternative arginine catabolism → neurotoxic
    Progressive spastic paraplegia — UNIQUE among ALL UCDs; arginine/guanidino neurotoxicity

ARG1 BIOCHEMISTRY (LOF → arginine critically high → step 5 block):
  Plasma arginine:        VERY HIGH → CRITICALLY HIGH (>200-500 µmol/L; normal 15-115)
                           THE PATHOGNOMONIC BIOMARKER — ONLY UCD with arginine THIS HIGH
  Plasma ammonia:         ONLY MILDLY ELEVATED (50-150 µmol/L; normal <50) or NORMAL
                           CRITICAL DISTINCTION FROM ALL OTHER UCDs — hyperammonemia NOT dominant
  Plasma citrulline:      NORMAL or mildly elevated (15-80 µmol/L; normal 15-35)
                           Upstream steps intact; slight elevation from ornithine depletion
  Plasma ornithine:       LOW-NORMAL (<50 µmol/L; normal 30-100) — cannot be regenerated
  Urine orotic acid:      MILDLY ELEVATED (10-30 µmol/mol Cr; normal <6)
                           Carbamoyl-P still produced (steps 1-2 intact); small overflow
  Guanidino compounds:    ELEVATED — homoarginine, guanidinoacetate (alternative arginine paths)
                           Neurotoxic; contribute to spasticity + IDD
  Plasma glutamine:       MILDLY ELEVATED (400-700 µmol/L) — alternative NH3 detox
  Argininosuccinate:      NORMAL — KEY NEGATIVE (contrast ASL where VERY HIGH)
  PLP:                    NORMAL — KEY NEGATIVE (not PLP-dependent; vs ALDH7A1/PDE)
  alpha-AASA:             NORMAL — KEY NEGATIVE (vs ALDH7A1-PDE)
  tHcy:                   NORMAL — KEY NEGATIVE (vs CBS/MTHFR)
  MMA:                    NORMAL — KEY NEGATIVE
  GABA:                   NORMAL — KEY NEGATIVE (vs ABAT)
  GHB:                    NORMAL — KEY NEGATIVE (vs SSADH)
  GAA:                    NORMAL — KEY NEGATIVE (vs GAMT)

ARG1 KEY DISTINCTIONS FROM OTHER UCDs:
  vs ALL proximal UCDs (CPS1/OTC/NAGS/ASS1/ASL):
    Arginine VERY HIGH in ARG1; arginine LOW in ALL others (not produced downstream of block)
    NH3 MILDLY elevated in ARG1; NH3 CRITICALLY elevated in proximal UCDs
    Progressive SPASTIC PARAPLEGIA in ARG1; acute hyperammonemic crises in proximal UCDs
    Citrulline NORMAL in ARG1; citrulline LOW (CPS1/OTC/NAGS) or HIGH (ASS1) or moderate (ASL)
  vs OAT (gyrate atrophy):
    OAT: ornithine VERY HIGH (400-1500); ARG1: ornithine LOW-NORMAL
    OAT: gyrate atrophy of retina PATHOGNOMONIC; ARG1: spastic paraplegia PATHOGNOMONIC
    Both AR, both amino acid metabolic disorders — but completely different substrates
  UNIQUE to ARG1: Arginine VERY HIGH is the ONLY UCD where arginine is elevated — all others LOW

EPILEPSY AND NEUROLOGICAL FEATURES IN ARG1:
  Progressive spastic diplegia/paraplegia: 80-90% — UNIQUE hallmark; NOT seen in other UCDs
    Gradual onset age 1-3 years; progressive over years; loss of ambulation 5-20 years if untreated
  Seizures: 60-70% overall; from guanidino compound neurotoxicity + mild hyperammonemia
  GTCS (generalised tonic-clonic): 40% — MODAL seizure type
  Focal seizures: 30% — temporal/frontal; cortical hyperexcitability
  Myoclonic encephalopathy: 20% — guanidino compound-mediated excitotoxicity
  Absence: 15% — attenuated phenotypes; subclinical
  Status epilepticus: 15% — less common than proximal UCDs (NH3 milder)
  Drug-resistant epilepsy: 20-30%
  IDD (intellectual/developmental disability): 80-90% — MOST SEVERE cognitive outcome among UCDs
  Microcephaly: 40-50% — progressive; correlated with arginine level
  Tremor/ataxia: 40-50% — cerebellar involvement; guanidino neurotoxicity
  Growth retardation: 50-60% — failure to thrive; anabolic deficit
  EEG: diffuse slowing, focal cortical abnormalities, photoparoxysmal response
  MRI: white matter signal change; periventricular; cortical atrophy progressive

TREATMENTS (evidence-graded):
  Level A (established):
    Low-arginine diet: PRIMARY dietary therapy; restrict dietary arginine to minimum requirements;
                       protein prescription based on essential AA formula
    Arginine-free essential amino acid formula: replaces dietary protein without adding arginine
    Sodium Benzoate + Phenylacetate (RAVICTI): nitrogen scavenging; less dominant need than proximal UCDs
    Physiotherapy / occupational therapy: spasticity management; mobility preservation
    IV Dextrose 10% (anti-catabolic): in metabolic crises (rare); prevent catabolism
    CRRT / HD: Level A for NH3 >300 µmol/L (rare acute decompensation)
  Level B (probable):
    LEV (levetiracetam): first-line AED; no hepatotoxicity; preferred in UCDs
    Baclofen: oral/intrathecal for spasticity; Level B; symptom management
    Liver transplant: Level B only — corrects hepatic ARG1; reduces arginine; limited evidence
                      neurological damage may NOT reverse post-transplant (extrahepatic ARG1 role)
    Arginine monitoring + diet titration: ongoing target arginine <200 µmol/L
  Absolute Contraindications:
    Valproate (VPA): ABSOLUTE CI — inhibits NAGS/CPS1; hyperammonemia; hepatotoxic in ALL UCDs
    High-protein diet / excess arginine: ABSOLUTE CI — worsens arginine accumulation
    L-Asparaginase: ABSOLUTE CI — asparagine/glutamine catabolism → hyperammonemia
    Arginine supplementation: ABSOLUTE CI — directly worsens toxicity (contrast with other UCDs!)
    High-dose protein supplements: ABSOLUTE CI

PHENOTYPIC CLASSES:
  Classic Spastic (70%): progressive spastic diplegia/paraplegia; IDD 80-90%; seizures; onset age 1-3y
  Mild Attenuated (20%): NBS-detected; mild or no spasticity; better cognitive outcome; low arginine
  Neonatal Acute (10%): very rare; severe hyperammonemia crisis (null alleles); day 1-5 onset

TOP PATHOGENIC VARIANTS (ARG1 gene):
  p.Arg21Cys       (Mn²⁺-binding site, ~20%; most common worldwide; disrupts metal coordination)
  p.Trp122Stop     (null, ~15%; severe; truncation → complete LOF)
  p.Arg291Ser      (catalytic domain, ~12%; moderate-severe)
  p.Gln158Stop     (null, ~10%; severe; premature stop)
  p.Arg108Gln      (active site, ~9%; classic spastic)
  p.His141Leu      (Mn²⁺-binding, ~8%; manganese coordination disrupted)
  p.Asp145Gly      (catalytic, ~7%; moderate spastic)
  p.Ala26Thr       (N-terminal, ~6%; attenuated; residual ~20% activity)

UNIQUE ARG1 FEATURES (beyond hyperammonemia):
  1. PROGRESSIVE SPASTIC PARAPLEGIA: UNIQUE hallmark — corticospinal tract degeneration
     Guanidino compounds (homoarginine, guanidinoacetate) directly toxic to upper motor neurons
     Treatable but NOT reversible once established → early diet critical
  2. MILD HYPERAMMONEMIA: UNIQUE among UCDs — proximal steps intact; arginine buffers NH3
     NH3 rarely >200 µmol/L except neonatal null alleles
  3. ARGININE VERY HIGH: the ONLY UCD where arginine is the elevated substrate (all others LOW)
  4. LIVER TRANSPLANT CAVEAT: neurological disease may persist post-transplant
     ARG1 expressed extrahepnatically (red blood cells, brain, kidney) — hepatic correction insufficient
  5. ARGININE SUPPLEMENTATION CONTRAINDICATED: unique — in all other UCDs arginine is PRIMARY therapy
"""

import random

SEED = 223
random.seed(SEED)

N_PATIENTS = 40

# Phenotypic classes
PHENOTYPE_CLASSES = [
    {"name": "Classic Spastic",     "pct": 70, "residual_activity": "0-10%",  "severity": "severe"},
    {"name": "Mild Attenuated",     "pct": 20, "residual_activity": "10-30%", "severity": "mild"},
    {"name": "Neonatal Acute",      "pct": 10, "residual_activity": "0-3%",   "severity": "critical"},
]

# Biomarker reference ranges and disease values
BIOMARKERS = {
    "plasma_arginine": {
        "label": "Plasma Arginine",
        "normal": "15–115 µmol/L",
        "disease": "VERY HIGH — CRITICALLY HIGH (>200–500 µmol/L; normal 15–115) — PATHOGNOMONIC HALLMARK",
        "direction": "↑↑↑ PATHOGNOMONIC",
        "status": "CRITICALLY ELEVATED",
        "color": "danger",
    },
    "plasma_ammonia": {
        "label": "Plasma Ammonia",
        "normal": "<50 µmol/L",
        "disease": "MILDLY ELEVATED or NORMAL (50–150 µmol/L) — CRITICAL DISTINCTION from proximal UCDs",
        "direction": "↑ MILD (NH3 NOT dominant)",
        "status": "MILDLY ELEVATED — UNIQUE vs other UCDs",
        "color": "warning",
    },
    "plasma_citrulline": {
        "label": "Plasma Citrulline",
        "normal": "15–35 µmol/L",
        "disease": "NORMAL or mildly elevated (15–80 µmol/L) — upstream steps intact",
        "direction": "→ NORMAL/mild ↑",
        "status": "NORMAL — KEY DISTINCTION (not LOW like CPS1/OTC/NAGS)",
        "color": "info",
    },
    "plasma_ornithine": {
        "label": "Plasma Ornithine",
        "normal": "30–100 µmol/L",
        "disease": "LOW-NORMAL (<50 µmol/L) — not regenerated from arginine; cycle slows at step 2",
        "direction": "↓ LOW-NORMAL",
        "status": "LOW-NORMAL — cycle substrate depletion",
        "color": "info",
    },
    "urine_orotic_acid": {
        "label": "Urine Orotic Acid",
        "normal": "<6 µmol/mol Cr",
        "disease": "MILDLY ELEVATED (10–30 µmol/mol Cr) — carbamoyl-P overflow from ornithine depletion",
        "direction": "↑ MILD",
        "status": "MILDLY ELEVATED — intermediate",
        "color": "warning",
    },
    "guanidino_compounds": {
        "label": "Guanidino Compounds (plasma/CSF)",
        "normal": "Low (homoarginine <5 µmol/L)",
        "disease": "ELEVATED — homoarginine, guanidinoacetate, argininic acid elevated; directly neurotoxic",
        "direction": "↑ ELEVATED — NEUROTOXIC",
        "status": "ELEVATED — spasticity + IDD mechanism",
        "color": "warning",
    },
    "argininosuccinate": {
        "label": "Argininosuccinate (plasma/urine)",
        "normal": "Undetectable",
        "disease": "NORMAL — KEY NEGATIVE (contrast ASL where VERY HIGH; step 4 intact in ARG1)",
        "direction": "→ NORMAL",
        "status": "NORMAL — KEY NEGATIVE",
        "color": "success",
    },
    "plp": {
        "label": "PLP (Pyridoxal-5-Phosphate)",
        "normal": "20–100 nmol/L",
        "disease": "NORMAL — KEY NEGATIVE (ARG1 not PLP-dependent; vs ALDH7A1/PDE)",
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
    {"variant": "p.Arg21Cys",    "domain": "Mn²⁺-binding site",  "freq": 20, "phenotype": "Classic spastic severe",    "note": "Most common worldwide; disrupts manganese coordination"},
    {"variant": "p.Trp122Stop",  "domain": "Null — truncation",  "freq": 15, "phenotype": "Severe/neonatal acute",      "note": "Complete LOF; severe neonatal in homozygotes"},
    {"variant": "p.Arg291Ser",   "domain": "Catalytic domain",    "freq": 12, "phenotype": "Classic spastic moderate",  "note": "Catalytic pocket disruption; partial residual"},
    {"variant": "p.Gln158Stop",  "domain": "Null — premature stop","freq": 10, "phenotype": "Severe neonatal",           "note": "Null allele; complete protein loss"},
    {"variant": "p.Arg108Gln",   "domain": "Active site",          "freq":  9, "phenotype": "Classic spastic",           "note": "Active site arginine critical for catalysis"},
    {"variant": "p.His141Leu",   "domain": "Mn²⁺-binding",        "freq":  8, "phenotype": "Classic spastic moderate",  "note": "Second Mn-binding histidine; metal coordination lost"},
    {"variant": "p.Asp145Gly",   "domain": "Catalytic",            "freq":  7, "phenotype": "Classic spastic",           "note": "Asp145 coordinates Mn²⁺ cluster"},
    {"variant": "p.Ala26Thr",    "domain": "N-terminal",           "freq":  6, "phenotype": "Mild attenuated",           "note": "Residual ~20% activity; attenuated spasticity"},
]

# Treatment table
TREATMENTS = [
    {
        "therapy": "Low-arginine diet",
        "level": "A",
        "dose": "Disease-specific AA prescription; target arginine <200 µmol/L",
        "rationale": "PRIMARY therapy; restricts substrate accumulation; halts spasticity progression",
        "class": "Dietary management",
    },
    {
        "therapy": "Arginine-free essential amino acid formula",
        "level": "A",
        "dose": "Full protein equivalent without arginine",
        "rationale": "Provides essential amino acids; maintains anabolic state without arginine load",
        "class": "Dietary management",
    },
    {
        "therapy": "Sodium Benzoate + Phenylacetate (RAVICTI)",
        "level": "A",
        "dose": "Standard nitrogen-scavenger dosing",
        "rationale": "Alternative nitrogen disposal; reduces NH3 (less critical than proximal UCDs)",
        "class": "Nitrogen scavenger",
    },
    {
        "therapy": "Physiotherapy / Occupational therapy",
        "level": "A",
        "dose": "Regular programme; early intensive",
        "rationale": "Spasticity management; mobility preservation; delay ambulation loss",
        "class": "Rehabilitation",
    },
    {
        "therapy": "IV Dextrose 10% (anti-catabolic)",
        "level": "A",
        "dose": "GIR 6–8 mg/kg/min in crisis",
        "rationale": "Prevents catabolism-driven arginine release; rare acute decompensation",
        "class": "Metabolic stabilisation",
    },
    {
        "therapy": "CRRT / Haemodialysis",
        "level": "A",
        "dose": "NH3 >300 µmol/L (rare; neonatal null only)",
        "rationale": "Emergency arginine + ammonia removal; reserved for neonatal acute phenotype",
        "class": "Extracorporeal",
    },
    {
        "therapy": "LEV (Levetiracetam)",
        "level": "B",
        "dose": "20–60 mg/kg/day",
        "rationale": "First-line AED in UCDs; no hepatotoxicity; preferred",
        "class": "Antiseizure medication",
    },
    {
        "therapy": "Baclofen (oral / intrathecal)",
        "level": "B",
        "dose": "Oral: 5–20 mg TID; intrathecal: continuous infusion",
        "rationale": "GABA-B agonist; reduces spasticity; UNIQUE symptom target in ARG1 vs other UCDs",
        "class": "Spasticity management",
    },
    {
        "therapy": "Liver Transplant",
        "level": "B",
        "dose": "Timing: before severe neurological injury",
        "rationale": "Corrects hepatic ARG1; reduces arginine; limited evidence; "
                     "neurological damage may NOT reverse (extrahepatic ARG1 role in red cells/brain)",
        "class": "Definitive (limited evidence)",
    },
    {
        "therapy": "Valproate (VPA)",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "Inhibits NAGS/CPS1; catastrophic hyperammonemia; hepatotoxic in ALL UCDs",
        "class": "Contraindicated AED",
    },
    {
        "therapy": "High-protein diet / Arginine supplementation",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "Directly worsens arginine accumulation; accelerates spasticity + IDD — UNIQUE CI in ARG1 (arginine is PRIMARY therapy in ALL other UCDs; in ARG1 it is CONTRAINDICATED)",
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

# Seizure types in ARG1
SEIZURE_TYPES = [
    {"type": "GTCS (generalised tonic-clonic)", "pct": 40, "note": "MODAL; guanidino cortical hyperexcitability + mild NH3"},
    {"type": "Focal seizures",                  "pct": 30, "note": "Temporal/frontal; cortical hyperexcitability"},
    {"type": "Myoclonic encephalopathy",         "pct": 20, "note": "Guanidino compound excitotoxicity; progressive"},
    {"type": "Absence seizures",                 "pct": 15, "note": "Attenuated phenotypes; subclinical EEG activity"},
    {"type": "Status epilepticus",               "pct": 15, "note": "Less common than proximal UCDs; NH3 milder"},
    {"type": "Drug-resistant epilepsy",          "pct": 25, "note": "Ongoing guanidino neurotoxicity despite AED"},
]

# Unique systemic features
SYSTEMIC_FEATURES = [
    {"feature": "Progressive spastic diplegia/paraplegia",  "pct": 85, "note": "UNIQUE to ARG1 among ALL UCDs; corticospinal tract degeneration; gradual onset 1-3 years"},
    {"feature": "Intellectual/developmental disability",    "pct": 85, "note": "MOST SEVERE among UCDs; guanidino compound toxicity; partially reversible with early diet"},
    {"feature": "Growth retardation",                       "pct": 55, "note": "Failure to thrive; anabolic deficit; protein restriction caveat"},
    {"feature": "Microcephaly",                             "pct": 45, "note": "Progressive; correlated with arginine level; treatable"},
    {"feature": "Tremor / cerebellar ataxia",               "pct": 45, "note": "Cerebellar involvement; guanidino neurotoxicity; overlaps with spasticity"},
    {"feature": "Protein aversion",                         "pct": 40, "note": "Dietary self-regulation; unlike other UCDs where protein aversion = NH3 warning sign"},
]


def _generate_cohort():
    random.seed(SEED)
    patients = []
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls["name"]] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append("Classic Spastic")
    phenotype_dist = phenotype_dist[:N_PATIENTS]
    random.shuffle(phenotype_dist)

    for i in range(N_PATIENTS):
        phenotype = phenotype_dist[i]
        if phenotype == "Neonatal Acute":
            age_onset_months = round(random.uniform(0, 1), 1)
            nh3_peak = random.randint(300, 800)
            arginine = random.randint(400, 700)
            citrulline = random.randint(30, 80)
            ornithine = random.randint(15, 40)
        elif phenotype == "Classic Spastic":
            age_onset_months = round(random.uniform(12, 36), 1)
            nh3_peak = random.randint(60, 150)
            arginine = random.randint(200, 500)
            citrulline = random.randint(15, 60)
            ornithine = random.randint(20, 50)
        else:  # Mild Attenuated
            age_onset_months = round(random.uniform(0, 60), 1)
            nh3_peak = random.randint(50, 100)
            arginine = random.randint(150, 280)
            citrulline = random.randint(15, 45)
            ornithine = random.randint(25, 60)

        orotic = round(random.uniform(8, 30), 1)
        spasticity = phenotype != "Mild Attenuated" or random.random() < 0.3
        seizures = random.random() < 0.65
        idd = random.random() < 0.85
        microcephaly = random.random() < 0.45
        growth_retard = random.random() < 0.55

        v = random.choice(VARIANTS)
        patients.append({
            "id": f"ARG1-{i+1:03d}",
            "phenotype": phenotype,
            "age_onset_months": age_onset_months,
            "nh3_peak_umol_l": nh3_peak,
            "arginine_plasma": arginine,
            "citrulline_umol_l": citrulline,
            "ornithine_umol_l": ornithine,
            "orotic_acid_urine": orotic,
            "spastic_paraplegia": spasticity,
            "seizures": seizures,
            "idd": idd,
            "microcephaly": microcephaly,
            "growth_retardation": growth_retard,
            "variant": v["variant"],
        })
    return patients


COHORT = _generate_cohort()


def get_overview():
    n = len(COHORT)
    n_spastic     = sum(1 for p in COHORT if p["spastic_paraplegia"])
    n_seizures    = sum(1 for p in COHORT if p["seizures"])
    n_idd         = sum(1 for p in COHORT if p["idd"])
    n_micro       = sum(1 for p in COHORT if p["microcephaly"])
    n_growth      = sum(1 for p in COHORT if p["growth_retardation"])
    n_classic     = sum(1 for p in COHORT if p["phenotype"] == "Classic Spastic")
    n_mild        = sum(1 for p in COHORT if p["phenotype"] == "Mild Attenuated")
    n_neonatal    = sum(1 for p in COHORT if p["phenotype"] == "Neonatal Acute")
    avg_arg       = round(sum(p["arginine_plasma"] for p in COHORT) / n)
    avg_nh3       = round(sum(p["nh3_peak_umol_l"] for p in COHORT) / n)
    avg_citr      = round(sum(p["citrulline_umol_l"] for p in COHORT) / n)
    avg_orn       = round(sum(p["ornithine_umol_l"] for p in COHORT) / n)

    return {
        "disease": "Hyperargininemia — ARG1 (Arginase-1) Deficiency",
        "omim_gene": "608313",
        "omim_disease": "207800",
        "gene": "ARG1",
        "chromosome": "6q23.2",
        "inheritance": "Autosomal Recessive (AR)",
        "protein": "322 aa; cytoplasmic homotrimeric enzyme; binds two Mn²⁺ ions",
        "prevalence": "~1:300,000–1,000,000 (rarest classical UCD)",
        "urea_cycle_step": "Step 5 of 5 — Arginine → Ornithine + Urea (FINAL STEP; regenerates ornithine)",
        "n_patients": n,
        "kpi": {
            "n_patients":       {"value": n,                         "label": "Cohort size",                   "color": "#4a148c"},
            "arginine_avg":     {"value": f"{avg_arg} µmol/L",       "label": "Mean Plasma Arginine (PATHOGNOMONIC)", "color": "#b71c1c"},
            "spastic_pct":      {"value": f"{round(n_spastic/n*100)}%", "label": "Spastic Paraplegia (UNIQUE)",  "color": "#880e4f"},
            "idd_pct":          {"value": f"{round(n_idd/n*100)}%",  "label": "IDD (most severe UCD)",          "color": "#4e342e"},
            "seizures_pct":     {"value": f"{round(n_seizures/n*100)}%", "label": "Seizures (%)",               "color": "#c62828"},
            "nh3_avg":          {"value": f"{avg_nh3} µmol/L",       "label": "Mean Peak NH3 (MILD vs UCDs)",   "color": "#e65100"},
            "micro_pct":        {"value": f"{round(n_micro/n*100)}%", "label": "Microcephaly (%)",              "color": "#1a237e"},
            "growth_pct":       {"value": f"{round(n_growth/n*100)}%","label": "Growth retardation (%)",        "color": "#006064"},
            "avg_citr":         {"value": f"{avg_citr} µmol/L",      "label": "Mean Citrulline (NORMAL range)", "color": "#2e7d32"},
            "avg_orn":          {"value": f"{avg_orn} µmol/L",       "label": "Mean Ornithine (LOW-NORMAL)",    "color": "#37474f"},
            "classic_pct":      {"value": f"{round(n_classic/n*100)}%","label": "Classic Spastic (%)",         "color": "#6a1b9a"},
        },
        "phenotype_dist": [
            {"class": "Classic Spastic",  "n": n_classic,  "pct": round(n_classic/n*100)},
            {"class": "Mild Attenuated",  "n": n_mild,     "pct": round(n_mild/n*100)},
            {"class": "Neonatal Acute",   "n": n_neonatal, "pct": round(n_neonatal/n*100)},
        ],
        "hallmark_biomarker": "Plasma Arginine VERY HIGH (>200–500 µmol/L) — PATHOGNOMONIC; ONLY UCD with arginine THIS HIGH (all other UCDs: arginine LOW)",
        "hallmark_clinical": "Progressive Spastic Diplegia/Paraplegia (80-90%) — UNIQUE to ARG1 among ALL UCDs; gradual onset NOT acute crises",
        "key_distinction_all_ucds": "Arginine HIGH (ARG1) vs arginine LOW (CPS1/OTC/NAGS/ASS1/ASL) — INVERSE biomarker pattern; NH3 MILD (ARG1) vs CRITICAL (proximal UCDs)",
        "arginine_ci_note": "Arginine supplementation ABSOLUTELY CONTRAINDICATED in ARG1 — unique reversal: arginine is PRIMARY therapy in ALL other UCDs; in ARG1 it is the TOXIC substrate",
        "liver_transplant_caveat": "Level B only — corrects hepatic ARG1 but neurological damage may persist; ARG1 expressed extrahepnatically (erythrocytes, brain, kidney)",
        "vpa_contraindication": "VPA ABSOLUTE CI — inhibits NAGS/CPS1; catastrophic hyperammonemia in ALL UCDs",
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
            "step": 5,
            "total_steps": 5,
            "enzyme": "Arginase-1",
            "reaction": "Arginine → Ornithine + Urea",
            "location": "Cytoplasmic (like ASS1 and ASL)",
            "cofactor": "Two Mn²⁺ ions per monomer (binuclear manganese cluster)",
            "upstream_intact": "Steps 1-4 (CPS1, OTC, ASS1, ASL) INTACT — arginine produced normally from cycle + diet",
            "downstream_deficit": "Ornithine NOT regenerated → cycle slows at step 2 (OTC needs ornithine); Urea NOT produced",
            "unique_consequence": "Arginine accumulates rather than being depleted; guanidino compounds form; progressive spasticity",
        },
        "differential_diagnosis": {
            "vs_proximal_ucds": {
                "key_diff": "Arginine VERY HIGH in ARG1; arginine LOW in ALL proximal UCDs (CPS1/OTC/NAGS/ASS1/ASL)",
                "nh3": "MILD in ARG1 (steps 1-4 intact); CRITICALLY HIGH in CPS1/OTC/NAGS/ASS1/ASL",
                "clinical": "Spastic paraplegia in ARG1; hyperammonemic crises in proximal UCDs",
                "citrulline": "Normal in ARG1; LOW in CPS1/OTC/NAGS; VERY HIGH in ASS1; elevated in ASL",
            },
            "vs_oat": {
                "key_diff": "ARG1: arginine VERY HIGH, ornithine LOW-NORMAL; OAT: ornithine VERY HIGH (400-1500), arginine normal",
                "retina": "OAT: gyrate atrophy of retina PATHOGNOMONIC; ARG1: spastic paraplegia PATHOGNOMONIC",
                "cycle_block": "ARG1 step 5 of urea cycle; OAT independent of urea cycle (mitochondrial ornithine catabolism)",
            },
            "vs_oat_clinical": {
                "key_diff": "Both have ornithine metabolism involvement but OPPOSITE substrate accumulation",
                "hair": "OAT: trichorrhexis nodosa ABSENT (vs ASL); ARG1: no hair abnormality",
            },
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "ARG1 (Arginase-1, OMIM *608313) encodes a 322-amino-acid cytoplasmic homotrimeric enzyme "
            "that catalyses the FINAL step of the urea cycle: L-Arginine → L-Ornithine + Urea. "
            "Each monomer binds a binuclear Mn²⁺ cluster essential for catalytic hydrolysis. "
            "ARG1 regenerates ornithine for re-entry into step 2 (OTC) and releases urea — "
            "the terminal nitrogen disposal product. Chromosome 6q23.2; autosomal recessive."
        ),
        "pathomechanism": (
            "ARG1 LOF → arginine CANNOT be hydrolysed → accumulates massively in plasma, urine, CSF. "
            "Ornithine NOT regenerated → urea cycle slows from ornithine depletion at step 2. "
            "Steps 1-4 (CPS1, OTC, ASS1, ASL) remain INTACT, continuing to produce arginine from diet and cycle. "
            "Consequence (1): arginine VERY HIGH — directly and via guanidino compound formation (homoarginine, "
            "guanidinoacetate) → neuronal excitotoxicity → progressive corticospinal tract degeneration → spastic paraplegia. "
            "Consequence (2): hyperammonemia MILD (not crisis-dominant) — steps 1-4 still buffer some NH3. "
            "Consequence (3): ornithine depletion → carbamoyl-P slight overflow → mild orotic aciduria."
        ),
        "biomarker_pattern": (
            "PATHOGNOMONIC: Plasma arginine VERY HIGH (>200–500 µmol/L; normal 15–115). "
            "THE ONLY UCD where arginine is the elevated substrate — all other UCDs show arginine LOW. "
            "NH3 only MILDLY elevated (50–150 µmol/L) or even NORMAL — critical distinction from proximal UCDs. "
            "Citrulline: NORMAL or mildly elevated (15–80 µmol/L) — upstream steps intact. "
            "Ornithine: LOW-NORMAL (<50 µmol/L) — not regenerated from arginine. "
            "Urine orotic: MILDLY elevated (10–30 µmol/mol Cr) — carbamoyl-P overflow from ornithine shortage. "
            "Guanidino compounds (homoarginine, guanidinoacetate): ELEVATED — neurotoxic. "
            "KEY NEGATIVES: Argininosuccinate normal (contrast ASL), PLP normal, alpha-AASA normal, "
            "tHcy normal, MMA normal, GABA normal, GHB normal."
        ),
        "key_distinction_from_other_ucds": (
            "ARG1 vs ALL proximal UCDs — completely INVERTED arginine pattern: "
            "ARG1: arginine VERY HIGH (substrate ACCUMULATES at step 5 block). "
            "CPS1/OTC/NAGS/ASS1/ASL: arginine LOW (not produced downstream of earlier blocks). "
            "Clinical inversion: ARG1 = progressive spastic paraplegia (NOT acute NH3 crises); "
            "proximal UCDs = acute hyperammonemic crises (NOT spastic paraplegia). "
            "NH3 inversion: ARG1 mild; CPS1/OTC/NAGS/ASS1/ASL critically high. "
            "Treatment inversion: ARG1 = arginine RESTRICTION (ABSOLUTE CI); "
            "all other UCDs = arginine SUPPLEMENTATION (PRIMARY therapy)."
        ),
        "unique_systemic_features": (
            "ARG1 has THREE unique features not seen in any other UCD: "
            "1. PROGRESSIVE SPASTIC DIPLEGIA/PARAPLEGIA (80-90%): corticospinal tract degeneration; "
            "   gradual onset age 1-3 years; loss of ambulation 5-20 years if untreated; "
            "   mechanism: guanidino compound neurotoxicity (homoarginine, guanidinoacetate). "
            "2. MILD/ABSENT HYPERAMMONEMIA: NH3 50-150 µmol/L only — UNIQUE; proximal UCDs >200–1000. "
            "   Steps 1-4 intact; arginine pool buffers nitrogen; fewer hyperammonemic crises. "
            "3. ARGININE CONTRAINDICATED: in ALL other UCDs, arginine is the PRIMARY therapy; "
            "   in ARG1, arginine supplementation WORSENS the disease — absolute contraindication. "
            "IDD: 80-90% — most severe cognitive outcome among ALL UCDs; partially reversible with early diet."
        ),
        "liver_transplant_note": (
            "Liver transplant in ARG1 deficiency: Level B evidence only — much weaker than in proximal UCDs. "
            "Corrects hepatic ARG1 deficiency; reduces plasma arginine and guanidino compounds. "
            "HOWEVER, ARG1 is expressed extrahepnatically (red blood cells, brain, kidney, gut). "
            "Neurological damage that is already established may NOT reverse post-transplant — "
            "because extrahepatic ARG1 contributes to guanidino metabolism in CNS. "
            "Timing is critical: transplant BEFORE severe neurological injury for best outcome. "
            "Contrast: liver transplant is Level A CURATIVE in CPS1/OTC/NAGS/ASS1/ASL for NH3."
        ),
        "seizure_management": (
            "Seizures in ARG1 arise from guanidino compound neurotoxicity + mild hyperammonemia. "
            "Guanidino compounds (homoarginine, guanidinoacetate) are excitotoxic → cortical hyperexcitability. "
            "Primary seizure prevention: low-arginine diet reduces guanidino compound accumulation. "
            "EEG: diffuse slowing, focal cortical abnormalities, photoparoxysmal response in some. "
            "AED: LEV first-line (no hepatotoxicity). Baclofen for spasticity (not seizures). "
            "NEVER VPA (catastrophic hyperammonemia in all UCDs). "
            "Status epilepticus less common than proximal UCDs — NH3 milder."
        ),
        "ar_inheritance_note": (
            "AR inheritance: both parents obligate heterozygous carriers; 25% recurrence risk per pregnancy. "
            "Males and females equally affected — contrast with OTC (X-linked). "
            "Heterozygous carriers: typically asymptomatic. "
            "Newborn screening (NBS): elevated arginine on tandem MS → urgent evaluation. "
            "~1:300,000–1,000,000; rarest classical UCD; comparable rarity to NAGS/CPS1. "
            "Late-presenting phenotype means NBS is CRITICAL — without NBS, diagnosis often delayed "
            "until progressive spasticity at age 2-5 years is investigated."
        ),
        "unique_features_vs_other_ucd": (
            "1. ARGININE VERY HIGH — ONLY UCD with elevated arginine (all others: arginine LOW). "
            "2. SPASTIC PARAPLEGIA — UNIQUE among ALL UCDs; gradual progressive; NOT acute crisis. "
            "3. MILD HYPERAMMONEMIA — UNIQUE; steps 1-4 intact; NH3 NOT the dominant presentation. "
            "4. ARGININE ABSOLUTELY CONTRAINDICATED — reversal of all other UCDs where arginine = primary Rx. "
            "5. LIVER TRANSPLANT LEVEL B ONLY — extrahepatic ARG1 limits curative potential; unique. "
            "6. GUANIDINO COMPOUND NEUROTOXICITY — specific mechanism beyond ammonia; unique pathway. "
            "7. STEP 5 = FINAL STEP — ornithine NOT regenerated; cycle slows from substrate depletion. "
            "8. RAREST CLASSICAL UCD — ~1:300,000-1,000,000; comparable to NAGS; rarest of 5 steps."
        ),
    }
