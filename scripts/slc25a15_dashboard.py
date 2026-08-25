#!/usr/bin/env python3
"""SLC25A15 (ORC1/ORNT1) Deficiency — HHH Syndrome Dashboard.

SLC25A15 encodes the mitochondrial ornithine transporter 1 (ORC1/ORNT1):
  Cytoplasmic Ornithine  →  [SLC25A15]  →  Mitochondrial Matrix Ornithine
  Simultaneously:          Mitochondrial Citrulline  →  Cytoplasm

  HHH SYNDROME: Hyperornithinemia–Hyperammonemia–Homocitrullinuria
  OMIM Disease: #238970   Gene: SLC25A15, OMIM *603861
  Chromosome: 13q14.11
  Inheritance: Autosomal Recessive (AR)
  Protein: 301 aa; inner mitochondrial membrane carrier; 6 transmembrane domains;
           mediates ornithine/citrulline antiport (ornithine IN, citrulline OUT)
  Prevalence: ~1:2,000,000 (ultra-rare; ~60-70 cases worldwide 2026)

MECHANISM — TRANSPORT FAILURE (ornithine CANNOT enter mitochondrial matrix):
  Normal SLC25A15: cytoplasmic ornithine imported → available for OTC (step 2 of urea cycle)
  SLC25A15 LOF: ornithine transport BLOCKED → ornithine ACCUMULATES in cytoplasm
                OTC has NO ornithine substrate despite CPS1 making carbamoyl-P normally
                Carbamoyl-P transferred to LYSINE instead of ornithine → HOMOCITRULLINE
                Carbamoyl-P also overflows into pyrimidine synthesis → orotic acid elevated
                NH₃ cannot be disposed → hyperammonemia (episodic crises)

POSITION IN UREA CYCLE — SLC25A15 AS ORNITHINE TRANSPORT (STEP 2a):
  NAGS: Glutamate + Acetyl-CoA → NAG [cofactor generator]
  Step 1: NH₃ + CO₂ + 2ATP → [CPS1, requires NAG] → Carbamoyl-P        (mitochondrial)
  Step 2a: [SLC25A15] imports ornithine into matrix (TRANSPORT — BLOCKED IN HHH)
  Step 2: Carbamoyl-P + Ornithine → [OTC] → Citrulline + Pi              (mitochondrial)
  Step 2b: [SLC25A15] exports citrulline out to cytoplasm
  Step 3: Citrulline + Aspartate + ATP → [ASS1] → Argininosuccinate      (cytoplasmic)
  Step 4: Argininosuccinate → [ASL] → Arginine + Fumarate                (cytoplasmic)
  Step 5: Arginine → [ARG1] → Ornithine + Urea                           (cytoplasmic)
  Then: Ornithine needs SLC25A15 to re-enter mitochondria → CYCLE BLOCKED at transport

HHH TRIAD (ALL THREE PATHOGNOMONIC):
  1. HyperOrnithinemia: ornithine VERY HIGH (400–1500 µmol/L; normal 30–100)
     — ornithine CANNOT enter mitochondria → pools in cytoplasm
  2. HyperAmmonemia: NH₃ ELEVATED episodically (150–400 µmol/L; crisis >200)
     — OTC has no ornithine substrate; NH₃ cannot be disposed via urea cycle
  3. Homocitrullinuria: urine homocitrulline ELEVATED (PATHOGNOMONIC)
     — carbamoyl-P carbamylates LYSINE (instead of ornithine) → homocitrulline excreted
     — NO other UCD produces homocitrulline; unique to SLC25A15 block

SLC25A15 BIOCHEMISTRY (LOF → ornithine transport BLOCKED):
  Plasma ornithine:       VERY HIGH (400–1500 µmol/L; normal 30–100) — PATHOGNOMONIC
                          Accumulates in cytoplasm; cannot cross mitochondrial membrane
  Plasma ammonia:         ELEVATED episodically (150–400 µmol/L during crisis; normal <50)
                          OTC substrate (ornithine) absent in mitochondria; NH₃ cannot enter cycle
  Urine homocitrulline:   ELEVATED — PATHOGNOMONIC; alternative carbamylation of lysine
                          N^ε-carbamyllysine formed when carbamoyl-P has no ornithine acceptor
  Urine orotic acid:      ELEVATED (5–50 µmol/mol Cr) — carbamoyl-P overflow to pyrimidines
  Plasma citrulline:      LOW-NORMAL (<30 µmol/L) — OTC cannot function without ornithine
  Plasma arginine:        LOW-NORMAL (<60 µmol/L) — reduced downstream production
  Plasma glutamine:       ELEVATED (500–900 µmol/L) — alternative NH₃ detox pathway
  OAT enzyme:             NORMAL — KEY DISTINCTION from OAT deficiency (gyrate atrophy)
  GAA:                    NORMAL — KEY NEGATIVE (vs GAMT/AGAT)
  tHcy:                   NORMAL — KEY NEGATIVE (vs CBS/MTHFR)
  MMA:                    NORMAL — KEY NEGATIVE
  GABA:                   NORMAL — KEY NEGATIVE (vs ABAT)
  GHB:                    NORMAL — KEY NEGATIVE (vs SSADH)
  PLP:                    NORMAL — KEY NEGATIVE (vs ALDH7A1-PDE)
  alpha-AASA:             NORMAL — KEY NEGATIVE (vs ALDH7A1)

KEY DISTINCTIONS FROM OTHER UCDs AND OAT:
  vs OAT (ornithine aminotransferase deficiency):
    OAT: ornithine VERY HIGH + gyrate atrophy (retinal) + NO hyperammonemia + NO homocitrullinuria
    SLC25A15: ornithine VERY HIGH + hyperammonemia + homocitrullinuria + NO gyrate atrophy
    SAME ornithine level, COMPLETELY DIFFERENT consequences — transport vs catabolism defect
  vs ARG1:
    ARG1: ARGININE very high, ornithine LOW-NORMAL, NO homocitrullinuria
    SLC25A15: ORNITHINE very high, arginine low, homocitrullinuria PATHOGNOMONIC
  vs CPS1/OTC/NAGS (proximal UCDs):
    All have citrulline VERY LOW + NH₃ CRITICALLY HIGH (>500 neonatal)
    SLC25A15: ornithine VERY HIGH (unique) + homocitrullinuria (unique)
    CPS1/OTC/NAGS: ornithine normal/low; no homocitrullinuria
  vs ASS1:
    ASS1: citrulline VERY HIGH (>500 µmol/L); SLC25A15: citrulline LOW-NORMAL
  vs ASL:
    ASL: argininosuccinate VERY HIGH; SLC25A15: argininosuccinate NORMAL
  UNIQUE TO SLC25A15:
    ONLY UCD/organic acid disorder with HOMOCITRULLINURIA — pathognomonic for transport block
    Coagulation defects (40%) — ornithine inhibits thrombin/fibrinogen; unique among UCDs
    French-Canadian founder (p.Phe188del) — specific genetic cluster; ORNT1 founder effect
"""

import random

SEED      = 229      # next in UCD series (223→229)
N_PATIENTS = 40

# Phenotypic classes
PHENOTYPE_CLASSES = [
    {"name": "Classic Episodic",    "pct": 50, "note": "Recurrent NH₃ crises + IDD; most common; later spastic paraplegia"},
    {"name": "Severe Neonatal",     "pct": 20, "note": "NH₃ crisis day 1–7; very high ornithine; rare null/null genotype"},
    {"name": "Mild Attenuated",     "pct": 30, "note": "Hyperornithinemia with minimal NH₃; NBS-detected or family screening"},
]

# Key pathogenic variants
VARIANTS = [
    {"variant": "p.Arg179X",    "freq": 25, "domain": "Transmembrane helix 4 — truncating", "phenotype": "Severe; null", "note": "Most common worldwide; French-Canadian AND other ethnic groups; nonsense"},
    {"variant": "p.Phe188del",  "freq": 22, "domain": "TM4/loop — 3 bp deletion", "phenotype": "Severe; French-Canadian", "note": "c.562_564delTTC; FOUNDER in Quebec/French-Canada ~50-60% of alleles in this population"},
    {"variant": "p.Leu72Pro",   "freq": 15, "domain": "TM2 — transmembrane helix 2", "phenotype": "Severe; Middle East", "note": "Mediterranean/Middle Eastern; disrupts TM helix packing; loss of transport"},
    {"variant": "p.Tyr107Asp",  "freq": 12, "domain": "Matrix loop — substrate contact", "phenotype": "Moderate", "note": "Reduces ornithine binding affinity; residual transport ~10-15% activity"},
    {"variant": "c.813+1G>A",   "freq": 10, "domain": "Splice — IVS6+1 null", "phenotype": "Severe; null", "note": "Splice donor null; complete loss of SLC25A15 mRNA; neonatal/severe episodic"},
    {"variant": "p.Pro229Leu",  "freq": 8,  "domain": "TM5 — transmembrane helix 5", "phenotype": "Moderate", "note": "Partial transport activity; episodic phenotype; responds to diet"},
    {"variant": "p.Ala390Val",  "freq": 7,  "domain": "C-terminal domain", "phenotype": "Mild-moderate", "note": "Retains ~25% transport; late-onset; NBS or family screening common"},
    {"variant": "p.Gly113Arg",  "freq": 6,  "domain": "TM3 — transmembrane helix 3", "phenotype": "Severe", "note": "Disrupts helix charge; abolishes ornithine antiport; neonatal/classic phenotype"},
]

# Biomarker panel
BIOMARKERS = {
    "orn":       {"label": "Plasma Ornithine",       "normal": "30–100 µmol/L",   "status": "VERY HIGH (400–1500)",     "direction": "↑↑↑ CRITICAL",  "color": "danger"},
    "nh3":       {"label": "Plasma Ammonia (peak)",  "normal": "<50 µmol/L",      "status": "ELEVATED episodic 150–400","direction": "↑↑ HIGH",        "color": "danger"},
    "homocit":   {"label": "Urine Homocitrulline",   "normal": "Absent (<0.2)",    "status": "ELEVATED PATHOGNOMONIC",   "direction": "↑↑ UNIQUE",      "color": "danger"},
    "orotic":    {"label": "Urine Orotic Acid",      "normal": "<6 µmol/mol Cr",  "status": "ELEVATED (5–50)",          "direction": "↑ ELEVATED",     "color": "warning"},
    "citr":      {"label": "Plasma Citrulline",      "normal": "15–35 µmol/L",    "status": "LOW-NORMAL (<30)",         "direction": "↓ LOW",          "color": "warning"},
    "arg":       {"label": "Plasma Arginine",        "normal": "15–115 µmol/L",   "status": "LOW-NORMAL (<60)",         "direction": "↓ LOW-NORMAL",   "color": "warning"},
    "gln":       {"label": "Plasma Glutamine",       "normal": "400–700 µmol/L",  "status": "ELEVATED (500–900)",       "direction": "↑ ALT NH3 DETOX","color": "warning"},
    "oat":       {"label": "OAT enzyme activity",    "normal": "Normal",           "status": "NORMAL — KEY NEGATIVE",    "direction": "→ NORMAL",       "color": "success"},
    "gaa":       {"label": "GAA (guanidinoacetate)", "normal": "<3 µmol/L",       "status": "NORMAL — KEY NEGATIVE",    "direction": "→ NORMAL",       "color": "success"},
    "tHcy":      {"label": "Total Homocysteine",     "normal": "<15 µmol/L",      "status": "NORMAL — KEY NEGATIVE",    "direction": "→ NORMAL",       "color": "success"},
    "mma":       {"label": "MMA",                    "normal": "<0.4 µmol/L",     "status": "NORMAL — KEY NEGATIVE",    "direction": "→ NORMAL",       "color": "success"},
    "plp":       {"label": "PLP (B6)",               "normal": "Normal",           "status": "NORMAL — KEY NEGATIVE",    "direction": "→ NORMAL",       "color": "success"},
    "alpha_aasa":{"label": "alpha-AASA",             "normal": "Absent",           "status": "NORMAL — KEY NEGATIVE",    "direction": "→ NORMAL",       "color": "success"},
    "argsucc":   {"label": "Argininosuccinate",      "normal": "Absent",           "status": "NORMAL — KEY NEGATIVE",    "direction": "→ NORMAL",       "color": "success"},
}

# Treatments
TREATMENTS = [
    {
        "therapy": "Low-protein diet + EAA formula",
        "level": "A",
        "dose": "0.8–1.5 g/kg/day protein; essential AA formula supplement",
        "rationale": "Reduces nitrogen load; prevents NH₃ crises; limits ornithine production from dietary arginine",
        "class": "Dietary (primary)",
    },
    {
        "therapy": "Citrulline supplementation",
        "level": "A",
        "dose": "100–200 mg/kg/day; adjust to maintain citrulline 15–60 µmol/L",
        "rationale": "Bypasses OTC ornithine-import block: provides citrulline as product, allowing step 3 (ASS1) onward; maintains partial urea cycle flux without requiring ornithine import",
        "class": "Amino acid bypass (primary)",
    },
    {
        "therapy": "Sodium benzoate + phenylbutyrate/RAVICTI",
        "level": "A",
        "dose": "Benzoate: 200–300 mg/kg/day; phenylbutyrate: 250–500 mg/kg/day OR RAVICTI 4.5–11.2 mL/m²/day",
        "rationale": "Nitrogen scavenging via glycine-hippurate and glutamine-phenylacetylglutamine; reduces NH₃ during crises; maintenance chronic control",
        "class": "Nitrogen scavenging",
    },
    {
        "therapy": "IV glucose + anti-catabolics (acute crisis)",
        "level": "A",
        "dose": "GIR 8–12 mg/kg/min glucose; insulin 0.05–0.1 U/kg/h; BCAA-restricted if >3 days",
        "rationale": "Anti-catabolic: halts protein catabolism (primary NH₃ source in crises); glucose oxidises to prevent muscle breakdown",
        "class": "Acute crisis (IV)",
    },
    {
        "therapy": "CRRT / Haemodialysis",
        "level": "A",
        "dose": "If NH₃ >400 µmol/L or rising despite medical management",
        "rationale": "Rapidly clears ammonia in acute hyperammonemic crises; life-saving bridge; faster than medical scavenging alone",
        "class": "Acute crisis (dialysis)",
    },
    {
        "therapy": "LEV (levetiracetam)",
        "level": "B",
        "dose": "10–40 mg/kg/day oral/IV; first-line AED",
        "rationale": "No hepatotoxicity; no NAGS/CPS1 inhibition; safest AED in all UCDs; SV2A mechanism avoids ornithine interaction",
        "class": "AED — first-line",
    },
    {
        "therapy": "Liver Transplant",
        "level": "B",
        "dose": "Pre-symptomatic if NH₃ uncontrollable; limited evidence",
        "rationale": "Corrects hepatic SLC25A15; reduces NH₃ and ornithine partially; LIMITED — SLC25A15 expressed ubiquitously; neurological benefit variable; less evidence than CPS1/OTC/ASS1/ASL",
        "class": "Definitive (limited evidence)",
    },
    {
        "therapy": "Valproate (VPA)",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "Inhibits NAGS → no NAG → CPS1 completely off → catastrophic hyperammonemia in ALL UCDs; in HHH the already-impaired urea cycle collapses entirely; multiple fatalities reported",
        "class": "Contraindicated AED",
    },
    {
        "therapy": "Ornithine supplementation",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "Ornithine CANNOT cross mitochondrial membrane (transport block); supplementation raises cytoplasmic ornithine further; worsens hyperornithinemia without improving urea cycle flux; UNIQUE CI in SLC25A15 (ornithine is supplemented in OAT for unrelated reason)",
        "class": "Contraindicated supplement",
    },
    {
        "therapy": "High-protein diet / L-Asparaginase",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "High protein → massive NH₃ load → hyperammonemic crisis; L-asparaginase → asparagine/glutamine catabolism → acute NH₃",
        "class": "Dietary/Drug contraindication",
    },
]

# Seizure types
SEIZURE_TYPES = [
    {"type": "GTCS (generalised tonic-clonic)", "pct": 30, "note": "Secondary to hyperammonemia; MODAL; acute crisis-triggered"},
    {"type": "Myoclonic encephalopathy",         "pct": 25, "note": "Chronic ornithine neurotoxicity + NH₃; sub-acute onset"},
    {"type": "Focal seizures",                   "pct": 20, "note": "Frontal/temporal; hyperammonemic focal cortical spread"},
    {"type": "Status epilepticus",               "pct": 20, "note": "Hyperammonemic crises; life-threatening; NH₃ >300 µmol/L"},
    {"type": "Absence seizures",                 "pct": 12, "note": "Attenuated phenotypes; hyperornithinemia without acute NH₃"},
    {"type": "Drug-resistant epilepsy (DRE)",    "pct": 25, "note": "Ongoing ornithine-mediated neurotoxicity; structural epilepsy from prior crises"},
]

# Systemic features
SYSTEMIC_FEATURES = [
    {"feature": "Hyperammonemic crises (episodic)", "pct": 85, "note": "Primary acute manifestation; triggered by protein load, fever, illness, catabolism"},
    {"feature": "Intellectual/developmental disability", "pct": 65, "note": "Moderate-severe; correlates with NH₃ peak exposure; partially preventable with early treatment"},
    {"feature": "Spastic paraplegia",              "pct": 35, "note": "Present but MILDER than ARG1; shared ornithine mechanism but cytoplasmic vs. urea cycle deficit"},
    {"feature": "Coagulation defects",             "pct": 40, "note": "UNIQUE to HHH: ornithine inhibits thrombin formation and fibrinogen polymerisation; NOT seen in other UCDs"},
    {"feature": "Protein aversion",                "pct": 65, "note": "Behavioural NH₃ self-regulation; strong diet avoidance; diagnostic behavioural clue"},
    {"feature": "Hepatomegaly / liver dysfunction","pct": 30, "note": "Ornithine hepatotoxicity; elevated transaminases; more prominent than in proximal UCDs"},
]


def _generate_cohort():
    random.seed(SEED)
    patients = []
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls["name"]] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append("Classic Episodic")
    phenotype_dist = phenotype_dist[:N_PATIENTS]
    random.shuffle(phenotype_dist)

    for i in range(N_PATIENTS):
        phenotype = phenotype_dist[i]
        if phenotype == "Severe Neonatal":
            age_onset_months = round(random.uniform(0, 0.5), 2)
            nh3_peak   = random.randint(350, 700)
            ornithine  = random.randint(900, 1500)
            citrulline = random.randint(5, 20)
        elif phenotype == "Classic Episodic":
            age_onset_months = round(random.uniform(1, 24), 1)
            nh3_peak   = random.randint(150, 400)
            ornithine  = random.randint(400, 900)
            citrulline = random.randint(10, 30)
        else:  # Mild Attenuated
            age_onset_months = round(random.uniform(6, 120), 1)
            nh3_peak   = random.randint(60, 180)
            ornithine  = random.randint(250, 500)
            citrulline = random.randint(15, 35)

        orotic        = round(random.uniform(5, 50), 1)
        homocitrulline = round(random.uniform(8, 80), 1)  # always elevated
        spasticity    = phenotype != "Mild Attenuated" and random.random() < 0.45
        seizures      = random.random() < 0.45
        idd           = phenotype != "Mild Attenuated" and random.random() < 0.75
        coagulopathy  = random.random() < 0.40

        v = random.choice(VARIANTS)
        patients.append({
            "id": f"SLC25A15-{i+1:03d}",
            "phenotype": phenotype,
            "age_onset_months": age_onset_months,
            "nh3_peak_umol_l": nh3_peak,
            "ornithine_plasma": ornithine,
            "citrulline_umol_l": citrulline,
            "orotic_urine": orotic,
            "homocitrulline_urine": homocitrulline,
            "spastic_paraplegia": spasticity,
            "seizures": seizures,
            "idd": idd,
            "coagulopathy": coagulopathy,
            "variant": v["variant"],
        })
    return patients


COHORT = _generate_cohort()


def get_overview():
    n           = len(COHORT)
    n_seizures  = sum(1 for p in COHORT if p["seizures"])
    n_idd       = sum(1 for p in COHORT if p["idd"])
    n_spastic   = sum(1 for p in COHORT if p["spastic_paraplegia"])
    n_coag      = sum(1 for p in COHORT if p["coagulopathy"])
    n_classic   = sum(1 for p in COHORT if p["phenotype"] == "Classic Episodic")
    n_neonatal  = sum(1 for p in COHORT if p["phenotype"] == "Severe Neonatal")
    n_mild      = sum(1 for p in COHORT if p["phenotype"] == "Mild Attenuated")
    avg_orn     = round(sum(p["ornithine_plasma"] for p in COHORT) / n)
    avg_nh3     = round(sum(p["nh3_peak_umol_l"] for p in COHORT) / n)
    avg_citr    = round(sum(p["citrulline_umol_l"] for p in COHORT) / n)
    avg_homocit = round(sum(p["homocitrulline_urine"] for p in COHORT) / n, 1)

    return {
        "disease": "HHH Syndrome — SLC25A15 (Ornithine Transporter 1) Deficiency",
        "omim_gene": "603861",
        "omim_disease": "238970",
        "gene": "SLC25A15",
        "alias": "ORC1 / ORNT1",
        "chromosome": "13q14.11",
        "inheritance": "Autosomal Recessive (AR)",
        "protein": "301 aa; inner mitochondrial membrane carrier; 6 TM domains; ornithine/citrulline antiport",
        "prevalence": "~1:2,000,000 (ultra-rare; ~60-70 cases worldwide 2026)",
        "mechanism": "Ornithine transport BLOCKED → ornithine accumulates in cytoplasm; OTC has no substrate; homocitrulline via alternative carbamylation of lysine",
        "n_patients": n,
        "kpi": {
            "n_patients":     {"value": n,                              "label": "Cohort size",                      "color": "#1a237e"},
            "ornithine_avg":  {"value": f"{avg_orn} µmol/L",           "label": "Mean Plasma Ornithine (PATHOGNOMONIC)", "color": "#b71c1c"},
            "nh3_avg":        {"value": f"{avg_nh3} µmol/L",           "label": "Mean Peak NH3 (crisis threshold)",  "color": "#e65100"},
            "homocit_avg":    {"value": f"{avg_homocit} µmol/mol Cr",  "label": "Mean Urine Homocitrulline (UNIQUE)","color": "#880e4f"},
            "citr_avg":       {"value": f"{avg_citr} µmol/L",          "label": "Mean Citrulline (LOW — OTC blocked)", "color": "#37474f"},
            "seizures_pct":   {"value": f"{round(n_seizures/n*100)}%", "label": "Seizures (%)",                      "color": "#c62828"},
            "idd_pct":        {"value": f"{round(n_idd/n*100)}%",      "label": "IDD (%)",                           "color": "#4e342e"},
            "spastic_pct":    {"value": f"{round(n_spastic/n*100)}%",  "label": "Spastic Paraplegia (MILDER than ARG1)","color": "#4a148c"},
            "coag_pct":       {"value": f"{round(n_coag/n*100)}%",     "label": "Coagulopathy (UNIQUE to HHH)",      "color": "#006064"},
            "classic_pct":    {"value": f"{round(n_classic/n*100)}%",  "label": "Classic Episodic (%)",              "color": "#1b5e20"},
        },
        "phenotype_dist": [
            {"class": "Classic Episodic", "n": n_classic,  "pct": round(n_classic/n*100)},
            {"class": "Mild Attenuated",  "n": n_mild,     "pct": round(n_mild/n*100)},
            {"class": "Severe Neonatal",  "n": n_neonatal, "pct": round(n_neonatal/n*100)},
        ],
        "hallmark_biomarker": (
            "HHH TRIAD (ALL THREE PATHOGNOMONIC): "
            "(1) Plasma Ornithine VERY HIGH (400–1500 µmol/L) — transport block; "
            "(2) Episodic Hyperammonemia (150–400 µmol/L) — OTC no substrate; "
            "(3) Urine Homocitrulline ELEVATED — alternative lysine carbamylation, UNIQUE to SLC25A15"
        ),
        "hallmark_clinical": "Episodic Hyperammonemic Crises (85%) + Intellectual Disability (65%) — protein-triggered crises; early diet prevents IDD",
        "unique_coagulopathy": "Coagulation defects (40%) UNIQUE among ALL UCDs — ornithine inhibits thrombin/fibrinogen; monitor PT/INR/fibrinogen",
        "key_distinction_oat": "SLC25A15 vs OAT: BOTH hyperornithinemia; KEY DIFFERENCE: SLC25A15 = hyperammonemia + homocitrullinuria (NO gyrate atrophy); OAT = gyrate atrophy (retinal) + NO hyperammonemia + NO homocitrullinuria",
        "citrulline_bypass": "Citrulline supplementation (Level A) BYPASSES the SLC25A15 transport block — provides citrulline as the product of the OTC reaction, allowing ASS1→ASL→ARG1 to proceed WITHOUT requiring ornithine import",
        "ornithine_ci_note": "Ornithine supplementation ABSOLUTELY CONTRAINDICATED — ornithine CANNOT enter mitochondria (the transport block is the disease); raising cytoplasmic ornithine worsens hyperornithinemia; does NOT help urea cycle",
        "vpa_contraindication": "VPA ABSOLUTE CI — inhibits NAGS/CPS1; in HHH syndrome, the already-impaired urea cycle collapses entirely; catastrophic hyperammonemia; multiple fatalities",
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
        "transport_mechanism": {
            "gene":      "SLC25A15 (ORC1/ORNT1)",
            "function":  "Mitochondrial inner membrane ornithine/citrulline antiporter",
            "reaction":  "Cytoplasmic Ornithine IN ↔ Mitochondrial Citrulline OUT (1:1 antiport)",
            "block":     "SLC25A15 LOF → ornithine CANNOT enter mitochondrial matrix → OTC has no substrate",
            "upstream_intact": "CPS1 + NAGS INTACT — carbamoyl-P made normally but has no ornithine acceptor",
            "alternative_path": "Carbamoyl-P + Lysine → Homocitrulline (N-ε-carbamyllysine) — PATHOGNOMONIC byproduct",
            "overflow_path": "Excess carbamoyl-P → pyrimidine synthesis → orotic acid elevated (like OTC, unlike CPS1/NAGS)",
            "unique_consequence": "Triple pathology: hyperornithinemia + hyperammonemia + homocitrullinuria — NO other UCD has all three",
        },
        "differential_diagnosis": {
            "vs_oat": {
                "key_diff": "CRITICAL: BOTH hyperornithinemia but completely different diseases",
                "hhh_features": "SLC25A15: NH₃ HIGH + homocitrullinuria + NO gyrate atrophy (retinal normal)",
                "oat_features": "OAT: gyrate atrophy 100% (retinal) + NO hyperammonemia + NO homocitrullinuria",
                "shared": "Plasma ornithine VERY HIGH in BOTH (400-1500); NOT distinguishable by ornithine level alone",
                "separator": "OROTIC ACID: elevated in SLC25A15 (OTC bypass); normal in OAT; HOMOCITRULLINE: elevated SLC25A15; absent OAT",
            },
            "vs_arg1": {
                "key_diff": "INVERTED amino acid pattern",
                "arg1": "ARG1: arginine VERY HIGH + ornithine LOW-NORMAL; spastic paraplegia SEVERE; NH₃ MILD",
                "slc25a15": "SLC25A15: ornithine VERY HIGH + arginine LOW-NORMAL; spastic paraplegia MILDER; NH₃ EPISODIC HIGH",
                "shared": "Both have some spastic paraplegia; both are ornithine-related disorders",
            },
            "vs_otc": {
                "key_diff": "OTC step 2 enzyme vs SLC25A15 transport (step 2a)",
                "otc": "OTC: ornithine NORMAL-LOW; homocitrullinuria ABSENT; orotic MARKEDLY HIGH; X-linked",
                "slc25a15": "SLC25A15: ornithine VERY HIGH; homocitrullinuria PRESENT; orotic elevated (moderate); AR",
                "shared": "Both block at step 2; both have elevated orotic acid; both have NH₃ crisis",
            },
            "vs_cps1": {
                "key_diff": "CPS1: ornithine normal; orotic NORMAL; citrulline LOW (no carbamoyl-P made)",
                "slc25a15": "SLC25A15: ornithine VERY HIGH; orotic ELEVATED; homocitrullinuria UNIQUE",
            },
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "SLC25A15 (ORC1/ORNT1, OMIM *603861) encodes a 301-amino-acid mitochondrial inner membrane "
            "carrier protein with 6 transmembrane domains. It mediates electroneutral ornithine/citrulline "
            "antiport: imports one ornithine molecule INTO the mitochondrial matrix in exchange for "
            "exporting one citrulline molecule OUT. This transport is ESSENTIAL for the continuity of "
            "the urea cycle: ornithine produced in the cytoplasm by ARG1 (step 5) must re-enter "
            "mitochondria to serve as substrate for OTC (step 2). Chromosome 13q14.11; autosomal recessive."
        ),
        "pathomechanism": (
            "SLC25A15 LOF → ornithine CANNOT cross the inner mitochondrial membrane → "
            "ornithine ACCUMULATES in cytoplasm (hyperornithinemia, 400–1500 µmol/L). "
            "OTC (step 2) in the mitochondrial matrix has NO ornithine substrate despite CPS1 "
            "(step 1) continuing to make carbamoyl-P. "
            "Three consequences: "
            "(1) Carbamoyl-P transferred to LYSINE instead of ornithine → "
            "    N-ε-carbamyllysine = HOMOCITRULLINE → excreted in urine (PATHOGNOMONIC). "
            "(2) Excess carbamoyl-P overflows into pyrimidine synthesis → orotic acid elevated. "
            "(3) Ammonia CANNOT be disposed via urea cycle → accumulates → hyperammonemia. "
            "This generates the UNIQUE HHH triad: hyperornithinemia + hyperammonemia + homocitrullinuria."
        ),
        "hhh_triad_explanation": (
            "HHH SYNDROME — THREE PATHOGNOMONIC FINDINGS, ALL ARISING FROM ONE TRANSPORT BLOCK: "
            "H1 — HYPERORNITHINEMIA: ornithine pools in cytoplasm (400–1500 µmol/L). "
            "    Normal recycling via SLC25A15 is impossible; ARG1 still produces ornithine from arginine. "
            "H2 — HYPERAMMONEMIA: OTC in mitochondria lacks ornithine substrate; NH₃ via diet and protein "
            "    catabolism cannot be captured by the urea cycle; episodic crises (150–400 µmol/L). "
            "H3 — HOMOCITRULLINURIA: carbamoyl-P (still made by intact CPS1) is transferred to the "
            "    ε-amino group of lysine (the next available acceptor) → homocitrulline (N-ε-carbamyllysine). "
            "    Homocitrulline is excreted in urine. NO other UCD produces homocitrulline — it requires "
            "    both (a) intact CPS1/NAGS and (b) absent OTC substrate → unique to SLC25A15 block."
        ),
        "citrulline_bypass_mechanism": (
            "Citrulline supplementation (Level A) in SLC25A15 deficiency works by BYPASSING the transport block: "
            "Normally: OTC (step 2) produces citrulline inside mitochondria → SLC25A15 exports it out. "
            "In HHH: no ornithine enters → OTC cannot run → no citrulline produced endogenously. "
            "Citrulline supplementation: provides exogenous citrulline DIRECTLY in cytoplasm → "
            "ASS1 (step 3) can use it immediately → argininosuccinate → arginine → ornithine via ARG1 → "
            "partial urea cycle maintained WITHOUT requiring SLC25A15-mediated ornithine import. "
            "This is different from how citrulline works in OTC/CPS1/NAGS deficiency."
        ),
        "ornithine_ci_rationale": (
            "Ornithine supplementation ABSOLUTELY CONTRAINDICATED in SLC25A15 deficiency — "
            "counterintuitive but mechanistically clear: "
            "The problem is NOT insufficient ornithine — it is the inability to TRANSPORT it into mitochondria. "
            "Supplementing ornithine raises CYTOPLASMIC ornithine further, which: "
            "(1) Worsens hyperornithinemia (already 400–1500 µmol/L). "
            "(2) Does NOT help OTC (ornithine still cannot cross the membrane). "
            "(3) May worsen neurological toxicity — elevated cytoplasmic ornithine is itself neurotoxic. "
            "Compare: ornithine is THERAPEUTIC in OAT deficiency (where ornithine is high but the PROBLEM "
            "is impaired catabolism, and arginine restriction reduces ornithine production — opposite strategy)."
        ),
        "vs_oat_distinction": (
            "SLC25A15 vs OAT — the ornithine twins: SAME elevated ornithine, COMPLETELY DIFFERENT diseases. "
            "OAT (ornithine aminotransferase deficiency): "
            "  Ornithine VERY HIGH (similar range), but from impaired catabolism (OAT converts Orn → P5C). "
            "  Gyrate atrophy of choroid and retina (100%) — specific retinal pattern. "
            "  NO hyperammonemia — urea cycle steps 1-5 all INTACT; only catabolism impaired. "
            "  NO homocitrullinuria — carbamoyl-P pathway untouched. "
            "SLC25A15 (HHH): "
            "  Ornithine VERY HIGH, from transport failure (ornithine cannot re-enter mitochondria). "
            "  NO gyrate atrophy — retina NORMAL; ophthalmic exam NORMAL. "
            "  Hyperammonemia PRESENT — OTC substrate absent; crises common. "
            "  Homocitrullinuria PRESENT — PATHOGNOMONIC; absent in OAT. "
            "DIAGNOSTIC KEY: fundus examination + urine homocitrulline separates instantly."
        ),
        "coagulopathy_mechanism": (
            "HHH syndrome-specific coagulopathy (40% of patients): "
            "Ornithine directly inhibits thrombin formation and fibrinogen polymerisation at high concentrations. "
            "Elevated cytoplasmic ornithine (400–1500 µmol/L vs normal <100) inhibits: "
            "  - Thrombin-fibrinogen interaction (competitive inhibition). "
            "  - Fibrin polymerisation (electrostatic disruption). "
            "Clinical: prolonged PT/INR; reduced fibrinogen; bleeding risk during crises. "
            "UNIQUE to HHH syndrome among ALL urea cycle disorders — no other UCD produces hyperornithinemia "
            "sufficient to cause coagulopathy (other UCDs have normal ornithine). "
            "Management: monitor PT/INR/fibrinogen regularly; FFP during crises if abnormal."
        ),
        "french_canadian_founder": (
            "French-Canadian founder allele: p.Phe188del (c.562_564delTTC). "
            "3-base-pair in-frame deletion removing phenylalanine 188 in the TM4/matrix loop junction. "
            "Accounts for ~50-60% of alleles in Quebec/French-Canadian patients with HHH syndrome. "
            "p.Phe188del patients: classic episodic phenotype; responds to diet + citrulline; "
            "IDD moderate when diagnosed early; spastic paraplegia less common than null/null. "
            "HHH syndrome was first described by Shih et al. (1969) in a French-Canadian family. "
            "Total HHH cases worldwide ~60-70 (2026); French-Canadian cluster is the largest single ethnic group."
        ),
        "seizure_management": (
            "Seizures in HHH syndrome arise from: (1) acute hyperammonemia (NH₃ >150 µmol/L triggers cortical "
            "hyperexcitability via glutamate/NMDA pathway); (2) chronic ornithine neurotoxicity; "
            "(3) structural damage from prior untreated crises. "
            "Primary prevention: protein restriction + citrulline + nitrogen scavenging to prevent NH₃ elevation. "
            "AED: LEV first-line (no hepatotoxicity, no NAGS/CPS1 inhibition). "
            "Acute crisis seizures: correct NH₃ first (IV glucose, benzoate/phenylbutyrate, CRRT if severe). "
            "NEVER VPA — inhibits NAGS, collapses residual urea cycle flux, fatal hyperammonemia."
        ),
        "ar_inheritance_note": (
            "AR inheritance: 25% recurrence risk per pregnancy; equal sex ratio. "
            "Compare OTC (X-linked): SLC25A15 affects both sexes equally. "
            "Heterozygous carriers: typically asymptomatic (50% transport activity sufficient). "
            "NBS: plasma ornithine and homocitrulline on expanded NBS (MS/MS); some programmes flag "
            "elevated ornithine on amino acid profile. "
            "~1:2,000,000; ultra-rare; ~60–70 cases worldwide (2026). "
            "Most common in French-Canadian (Quebec) and some Middle Eastern/Mediterranean populations."
        ),
    }
