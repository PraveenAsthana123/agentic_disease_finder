#!/usr/bin/env python3
"""SLC25A13 (Citrin / AGC2) Deficiency — Citrin Deficiency / CTLN2 Dashboard.

SLC25A13 encodes citrin, the mitochondrial aspartate-glutamate carrier 2 (AGC2):
  Cytoplasmic Glutamate + H⁺  →  [SLC25A13]  →  Mitochondrial Glutamate
  Simultaneously:               Mitochondrial Aspartate  →  Cytoplasm

  THREE AGE-RELATED PHENOTYPES:
  1. NICCD  — Neonatal Intrahepatic Cholestasis caused by Citrin Deficiency (0–1 yr)
  2. FTTDCD — Failure to Thrive and Dyslipidemia caused by Citrin Deficiency (1–11 yr)
  3. CTLN2  — Citrullinemia Type 2 (adult-onset; most clinically severe)

  OMIM Disease (NICCD): #605814   (FTTDCD): #222700   (CTLN2): #603471
  OMIM Gene: *603859
  Chromosome: 7q21.3
  Inheritance: Autosomal Recessive (AR)
  Protein: 675 aa; inner mitochondrial membrane; AGC family; EF-hand Ca²⁺-sensing domains
  Prevalence: ~1:17,000–20,000 East Asia (Japan/China/Korea); ~1:100,000 worldwide
  Most common UCD in East Asian adults

MECHANISM — ASPARTATE EXPORT FAILURE:
  Normal SLC25A13: imports cytoplasmic glutamate INTO mitochondria; exports aspartate OUT
  → cytoplasmic aspartate available for ASS1 (urea cycle step 3: Cit + Asp → Argininosuccinate)
  → also drives the malate-aspartate shuttle (transfers reducing equivalents across membrane)
  SLC25A13 LOF: aspartate CANNOT exit mitochondria → cytoplasmic aspartate DEPLETED
  → ASS1 stalls (no aspartate substrate despite citrulline being present)
  → citrulline ACCUMULATES (same biochemical profile as ASS1 deficiency but different cause)
  → NH₃ ELEVATED (but often milder episodically; partial compensation via other routes)
  → Malate-aspartate shuttle impaired → NADH/NAD⁺ ratio altered → gluconeogenesis defective
     → carbohydrate tolerance impaired (CARBOHYDRATE AVERSION — PATHOGNOMONIC behavior)
     → galactose intolerance in NICCD (galactosemia-like phenotype)
     → dyslipidemia (impaired lipid metabolism; CTLN2 hallmark)

POSITION IN UREA CYCLE — SLC25A13 PROVIDES ASPARTATE FOR STEP 3:
  Step 1: NH₃ + CO₂ + 2ATP → [CPS1+NAGS] → Carbamoyl-P                  (mitochondrial)
  Step 2: Carbamoyl-P + Orn → [OTC] → Citrulline                          (mitochondrial)
  Step 2b: Citrulline exits mitochondria → cytoplasm [via SLC25A15/ORC1]
  Step 3: Citrulline + Aspartate + ATP → [ASS1] → Argininosuccinate       (cytoplasmic)
         ↑ SLC25A13 PROVIDES ASPARTATE HERE — BLOCKED IN CITRIN DEFICIENCY
  Step 4: Argininosuccinate → [ASL] → Arginine + Fumarate                 (cytoplasmic)
  Step 5: Arginine → [ARG1] → Ornithine + Urea                            (cytoplasmic)
  ASS1 stalls (no aspartate) → citrulline ACCUMULATES in plasma (300–1000 µmol/L CTLN2)

KEY BIOMARKERS FOR CTLN2 (adult):
  Plasma citrulline:      VERY HIGH (300–1000 µmol/L; normal 15–35) — ASS1 stalled (no Asp)
  Plasma ammonia:         ELEVATED episodically (100–400 µmol/L; crisis can be severe)
  Plasma threonine:       ELEVATED (malate-aspartate shuttle defect; NADH/NAD⁺ shifted)
  Plasma methionine:      ELEVATED (liver dysfunction + transulfuration impaired)
  Plasma tyrosine:        ELEVATED (liver disease; catabolism impaired)
  Plasma arginine:        LOW-NORMAL (<60 µmol/L) — downstream production impaired
  Argininosuccinate:      ABSENT — KEY DISTINCTION from ASL (where argininosuccinate VERY HIGH)
                          — ASS1 CANNOT make argininosuccinate without aspartate substrate
  Orotic acid (urine):    ELEVATED (5–30 µmol/mol Cr) — carbamoyl-P overflow (like OTC/SLC25A15)
  Lactate:                ELEVATED — malate-aspartate shuttle impaired; cytoplasmic NADH excess
  Galactose (NICCD):      ELEVATED — galactosemia-like; lactose/galactose intolerance
  AFP (NICCD):            ELEVATED — liver injury biomarker in neonatal phase
  Lipids (CTLN2):         Triglycerides HIGH; VLDL HIGH; HDL LOW — dyslipidemia hallmark
  Carbohydrate AVERSION:  PATHOGNOMONIC behavioral finding — patients self-restrict carbs/sugars
                          (carbs worsen NADH/NAD⁺ imbalance → hyperammonemia)

KEY DISTINCTIONS FROM OTHER DISORDERS:
  vs ASS1 (Citrullinemia Type 1 / CTLN1):
    ASS1: enzyme defect → citrulline VERY HIGH + argininosuccinate ABSENT + neonatal-predominant
    SLC25A13: SAME elevated citrulline profile but ADULT-ONSET predominantly + carb aversion
    ASS1: no malate-aspartate shuttle defect → no lactate/NADH pattern
    SLC25A13: NICCD in neonates (not neonatal hyperammonemia crisis; cholestasis predominates)
    CTLN2 vs CTLN1: gene panel distinguishes; similar biochemistry but age/phenotype differs
  vs OTC:
    OTC: citrulline VERY LOW (block upstream; citrulline cannot be made)
    SLC25A13: citrulline VERY HIGH (citrulline made but cannot proceed to step 3)
    OTC: orotic MARKEDLY HIGH; SLC25A13: orotic MILDLY elevated (intermediate)
  vs SLC25A15 (HHH syndrome):
    SLC25A15: ornithine VERY HIGH + homocitrullinuria + citrulline LOW
    SLC25A13: citrulline VERY HIGH + NO homocitrullinuria + ornithine NORMAL
    Both: SLC25 family transporters; both AR; both impair urea cycle at different points
  vs CPS1/NAGS:
    CPS1/NAGS: citrulline VERY LOW (cannot make citrulline); SLC25A13: citrulline VERY HIGH
"""

import random

SEED       = 235      # next in carrier series after SLC25A15 (seed 229)
N_PATIENTS = 40

# Phenotypic classes — three age-related phenotypes
PHENOTYPE_CLASSES = [
    {"name": "CTLN2 (Adult)",   "pct": 55, "note": "Episodic hyperammonemia in adults; personality changes; carb aversion; liver transplant curative"},
    {"name": "NICCD (Neonatal)","pct": 30, "note": "Neonatal cholestasis; galactosemia-like; often self-resolves by 1 year; may progress"},
    {"name": "FTTDCD (Child)",  "pct": 15, "note": "Failure to thrive; hyperlipidemia; hypoglycemia; subclinical; bridges NICCD → CTLN2"},
]

# Top pathogenic variants (predominantly East Asian)
VARIANTS = [
    {"variant": "c.IVS16+1G>A (IVS16ins3kb)", "freq": 55, "domain": "Intron 16 — 3kb insertion (most common Japan/Korea/China)", "phenotype": "Severe; CTLN2 + NICCD", "note": "1.7kb Alu + 1.3kb insert; founder allele East Asia; 50-60% Japanese alleles; introduces premature stop"},
    {"variant": "c.1638_1660dup23",            "freq": 15, "domain": "Exon 17 — 23bp duplication frameshift",                    "phenotype": "Severe; CTLN2",         "note": "Second most common Japanese/Korean; frameshift → null; classic adult CTLN2"},
    {"variant": "p.Arg605Gln (c.1814G>A)",     "freq": 12, "domain": "TM domain — EF-hand adjacent",                            "phenotype": "Moderate; CTLN2",       "note": "Most common Southern Chinese, Vietnamese, Malaysian; partial function; late CTLN2"},
    {"variant": "p.Trp498Stop (c.1493G>A)",    "freq": 8,  "domain": "EF-hand — nonsense",                                       "phenotype": "Severe",                "note": "Japan + Korea; null allele; NICCD severe; CTLN2 early"},
    {"variant": "c.IVS13−14A>G",               "freq": 6,  "domain": "Splice — IVS13",                                           "phenotype": "Moderate-severe",       "note": "Splice site; partial skipping; residual protein"},
    {"variant": "c.851_854delGTAT",            "freq": 5,  "domain": "Exon 9 — deletion frameshift",                             "phenotype": "Severe",                "note": "Deletion → premature stop; Japan; null allele"},
    {"variant": "p.Met1_Phe16del",             "freq": 4,  "domain": "N-terminal signal/mitochondrial targeting sequence",       "phenotype": "Moderate",              "note": "N-terminal deletion; mistargeting or instability; milder CTLN2"},
    {"variant": "p.Glu601stop",                "freq": 3,  "domain": "C-terminal EF-hand — nonsense",                            "phenotype": "Moderate-severe",       "note": "EF-hand disruption; reduced Ca²⁺ sensing; CTLN2 + NICCD"},
]

# Biomarker panel
BIOMARKERS = {
    "citr":     {"label": "Plasma Citrulline",       "normal": "15–35 µmol/L",    "status": "VERY HIGH (300–1000 CTLN2)",   "direction": "↑↑↑ CRITICAL",  "color": "danger"},
    "nh3":      {"label": "Plasma Ammonia (crisis)",  "normal": "<50 µmol/L",      "status": "ELEVATED episodic 100–400",    "direction": "↑↑ HIGH",        "color": "danger"},
    "thr":      {"label": "Plasma Threonine",         "normal": "70–170 µmol/L",   "status": "ELEVATED (200–500)",           "direction": "↑ MAS defect",   "color": "warning"},
    "met":      {"label": "Plasma Methionine",        "normal": "15–40 µmol/L",    "status": "ELEVATED (50–200 NICCD/CTLN2)","direction": "↑ liver disease","color": "warning"},
    "tyr":      {"label": "Plasma Tyrosine",          "normal": "30–120 µmol/L",   "status": "ELEVATED (liver dysfunction)", "direction": "↑ liver",        "color": "warning"},
    "orotic":   {"label": "Urine Orotic Acid",        "normal": "<6 µmol/mol Cr",  "status": "ELEVATED (5–30)",              "direction": "↑ ELEVATED",     "color": "warning"},
    "arg":      {"label": "Plasma Arginine",          "normal": "15–115 µmol/L",   "status": "LOW-NORMAL (<60)",             "direction": "↓ LOW-NORMAL",   "color": "warning"},
    "lactate":  {"label": "Plasma Lactate",           "normal": "<2.0 mmol/L",     "status": "ELEVATED (2–5) MAS defect",    "direction": "↑ NADH excess",  "color": "warning"},
    "argsucc":  {"label": "Argininosuccinate",        "normal": "Absent",           "status": "ABSENT — KEY NEGATIVE",        "direction": "→ ABSENT (Asp↓)","color": "success"},
    "orn":      {"label": "Plasma Ornithine",         "normal": "30–100 µmol/L",   "status": "NORMAL — KEY NEGATIVE",        "direction": "→ NORMAL",       "color": "success"},
    "homocit":  {"label": "Urine Homocitrulline",     "normal": "Absent",           "status": "ABSENT — KEY NEGATIVE vs HHH", "direction": "→ ABSENT",       "color": "success"},
    "tHcy":     {"label": "Total Homocysteine",       "normal": "<15 µmol/L",      "status": "NORMAL — KEY NEGATIVE",        "direction": "→ NORMAL",       "color": "success"},
    "mma":      {"label": "MMA",                      "normal": "<0.4 µmol/L",     "status": "NORMAL — KEY NEGATIVE",        "direction": "→ NORMAL",       "color": "success"},
    "gaa":      {"label": "GAA (guanidinoacetate)",   "normal": "<3 µmol/L",       "status": "NORMAL — KEY NEGATIVE",        "direction": "→ NORMAL",       "color": "success"},
    "vlcfa":    {"label": "VLCFA",                    "normal": "Normal",           "status": "NORMAL — KEY NEGATIVE",        "direction": "→ NORMAL",       "color": "success"},
}

# Treatments (evidence-graded)
TREATMENTS = [
    {
        "therapy": "High-fat, low-carbohydrate diet",
        "level": "A",
        "dose": "Fat 40–60% kcal; carb <30% kcal; protein 1–2 g/kg/day; avoid simple sugars",
        "rationale": "Carbohydrate oxidation requires malate-aspartate shuttle for NADH transfer; blocked in SLC25A13 LOF → excess cytoplasmic NADH → pyruvate → lactate accumulates; fat bypasses this dependence; patients self-select high-fat diet (carbohydrate aversion is behavioural adaptation)",
        "class": "Dietary (primary — adults)",
    },
    {
        "therapy": "Arginine supplementation",
        "level": "A",
        "dose": "200–500 mg/kg/day; adjust to maintain arginine 80–150 µmol/L",
        "rationale": "Provides arginine downstream of the ASS1 block → maintains ornithine cycle flux via ARG1; reduces citrulline accumulation (partial); supports urea cycle completion; standard in CTLN2 management",
        "class": "Amino acid (primary — adults)",
    },
    {
        "therapy": "MCT-enriched lactose-free formula (NICCD)",
        "level": "A",
        "dose": "MCT formula; lactose-free; supplemental fat-soluble vitamins (A, D, E, K)",
        "rationale": "MCT directly enters mitochondria (bypasses carnitine shuttle); provides energy without glucose/galactose; lactose-free critical because SLC25A13 LOF impairs galactose metabolism (galactosemia-like); normalises liver function in NICCD",
        "class": "Dietary (primary — NICCD)",
    },
    {
        "therapy": "Liver transplant",
        "level": "A",
        "dose": "Orthotopic LT; pre-emptive before severe encephalopathy preferred",
        "rationale": "SLC25A13 is predominantly expressed in liver; LT replaces citrin-deficient hepatocytes with donor citrin-expressing liver; CURATIVE for the metabolic defect; biochemical normalisation; neurological recovery excellent if transplanted before extensive damage; strongly preferred over long-term dietary management in CTLN2",
        "class": "Definitive (curative in liver-expressed disease)",
    },
    {
        "therapy": "Sodium benzoate + phenylbutyrate/RAVICTI",
        "level": "A",
        "dose": "Benzoate 200–300 mg/kg/day; RAVICTI 4.5–11.2 mL/m²/day; acute crises",
        "rationale": "Nitrogen scavenging: conjugates glycine (benzoate→hippurate) and glutamine (phenylbutyrate→phenylacetylglutamine) → NH₃ clearance via renal excretion; bridge to LT or dietary control; maintenance in non-LT candidates",
        "class": "Nitrogen scavenging",
    },
    {
        "therapy": "LEV (levetiracetam)",
        "level": "B",
        "dose": "10–40 mg/kg/day oral/IV; first-line AED",
        "rationale": "No hepatotoxicity; no NAGS/CPS1 inhibition; safest AED in all metabolic liver diseases; SV2A mechanism does not interact with aspartate transport",
        "class": "AED — first-line",
    },
    {
        "therapy": "Valproate (VPA)",
        "level": "ABSOLUTE CI",
        "dose": "NEVER",
        "rationale": "Triple CI in SLC25A13: (1) inhibits NAGS → CPS1 off → catastrophic NH₃; (2) hepatotoxicity in pre-existing liver disease (NICCD/CTLN2); (3) worsens mitochondrial dysfunction. Multiple fatalities. NEVER use in any citrin deficiency phenotype.",
        "class": "Contraindicated AED",
    },
    {
        "therapy": "IV Glucose / high-carbohydrate loads",
        "level": "HIGH RISK",
        "dose": "AVOID in CTLN2; if essential (hypoglycemia), use minimal with close monitoring",
        "rationale": "Carbohydrates require malate-aspartate shuttle for NADH oxidation; BLOCKED in SLC25A13 → cytoplasmic NADH excess → lactate → hyperammonemia worsens. CTLN2 patients instinctively avoid carbs. IV glucose in acute crises may paradoxically worsen NH₃. Use fat-based caloric supplementation instead.",
        "class": "Dietary/IV contraindication",
    },
    {
        "therapy": "Lactose / galactose / alcohol",
        "level": "HIGH RISK",
        "dose": "AVOID",
        "rationale": "Galactose worsens NICCD (galactosemia-like phenotype); alcohol activates CTLN2 crises (impairs hepatic aspartate utilisation); lactose must be restricted in NICCD. All increase hepatic metabolic load beyond SLC25A13 capacity.",
        "class": "Dietary contraindication",
    },
]

# Seizure types
SEIZURE_TYPES = [
    {"type": "Focal seizures (hyperammonemic)",  "pct": 30, "note": "CTLN2: cortical hyperexcitability from NH₃ >100 µmol/L; frontal/temporal"},
    {"type": "GTCS (generalised tonic-clonic)",  "pct": 25, "note": "Acute hyperammonemic crisis; hepatic encephalopathy-related"},
    {"type": "Status epilepticus",               "pct": 20, "note": "CTLN2 acute decompensation; NH₃ >300; psychiatric features + seizures"},
    {"type": "Drug-resistant epilepsy (DRE)",    "pct": 20, "note": "Chronic liver disease; ongoing metabolic dysfunction; structural epilepsy"},
    {"type": "Absence / absence-like",           "pct": 15, "note": "FTTDCD phase; mild episodic hyperammonemia; metabolic contribution"},
    {"type": "Neonatal seizures (NICCD)",        "pct": 10, "note": "Acute liver failure in severe NICCD; hypoglycaemia-triggered; less common"},
]

# Systemic features
SYSTEMIC_FEATURES = [
    {"feature": "Cholestasis / neonatal jaundice (NICCD)",  "pct": 95, "note": "NICCD hallmark; direct hyperbilirubinaemia; often presents as prolonged jaundice"},
    {"feature": "Episodic hyperammonemia (CTLN2)",          "pct": 90, "note": "Adult crisis: confusion, coma, disorientation, nocturnal delirium; protein/alcohol/carb triggered"},
    {"feature": "Carbohydrate aversion (CTLN2/FTTDCD)",     "pct": 80, "note": "PATHOGNOMONIC behaviour: patients instinctively avoid sweets, carbs, soft drinks; prefer protein/fat"},
    {"feature": "Dyslipidemia / hyperlipidaemia",           "pct": 75, "note": "CTLN2 hallmark: elevated TG, VLDL, LDL; reduced HDL; impaired lipid metabolism"},
    {"feature": "Psychiatric / personality changes",         "pct": 70, "note": "CTLN2: nocturnal abnormal behaviour, aggression, hallucinations; often misdiagnosed as psychiatric"},
    {"feature": "Hepatomegaly / liver dysfunction",          "pct": 65, "note": "Elevated transaminases; steatohepatitis; cirrhosis in untreated CTLN2"},
    {"feature": "Failure to thrive (FTTDCD)",               "pct": 55, "note": "Childhood bridge phase: hypoglycaemia, fatigue, poor growth; often subclinical"},
    {"feature": "Intellectual disability (NICCD severe)",   "pct": 20, "note": "Only in severe NICCD with liver failure and repeated NH₃ crises; rare; LT prevents"},
]


def _generate_cohort():
    random.seed(SEED)
    patients = []
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls["name"]] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append("CTLN2 (Adult)")
    phenotype_dist = phenotype_dist[:N_PATIENTS]
    random.shuffle(phenotype_dist)

    for i in range(N_PATIENTS):
        phenotype = phenotype_dist[i]
        if phenotype == "NICCD (Neonatal)":
            age_onset_months  = round(random.uniform(0, 2), 2)
            nh3_peak          = random.randint(80, 250)
            citrulline        = random.randint(100, 400)
            threonine         = random.randint(150, 350)
            methionine        = random.randint(50, 200)
            tg_level          = round(random.uniform(1.5, 4.0), 1)
        elif phenotype == "FTTDCD (Child)":
            age_onset_months  = round(random.uniform(12, 132), 1)
            nh3_peak          = random.randint(60, 150)
            citrulline        = random.randint(80, 300)
            threonine         = random.randint(130, 280)
            methionine        = random.randint(35, 120)
            tg_level          = round(random.uniform(2.0, 5.0), 1)
        else:  # CTLN2 (Adult)
            age_onset_months  = round(random.uniform(180, 600), 0)
            nh3_peak          = random.randint(100, 400)
            citrulline        = random.randint(300, 1000)
            threonine         = random.randint(200, 500)
            methionine        = random.randint(50, 200)
            tg_level          = round(random.uniform(2.5, 8.0), 1)

        orotic_urine      = round(random.uniform(5, 30), 1)
        carb_aversion     = phenotype == "CTLN2 (Adult)" and random.random() < 0.85
        seizures          = random.random() < 0.40
        dyslipidemia      = random.random() < 0.75
        liver_disease     = random.random() < 0.65
        psychiatric       = phenotype == "CTLN2 (Adult)" and random.random() < 0.70
        v                 = random.choice(VARIANTS)

        patients.append({
            "id":               f"SLC25A13-{i+1:03d}",
            "phenotype":        phenotype,
            "age_onset_months": age_onset_months,
            "nh3_peak_umol_l":  nh3_peak,
            "citrulline_umol_l":citrulline,
            "threonine_umol_l": threonine,
            "methionine_umol_l":methionine,
            "tg_mmol_l":        tg_level,
            "orotic_urine":     orotic_urine,
            "carb_aversion":    carb_aversion,
            "seizures":         seizures,
            "dyslipidemia":     dyslipidemia,
            "liver_disease":    liver_disease,
            "psychiatric":      psychiatric,
            "variant":          v["variant"],
        })
    return patients


COHORT = _generate_cohort()


def get_overview():
    n           = len(COHORT)
    n_seizures  = sum(1 for p in COHORT if p["seizures"])
    n_dyslip    = sum(1 for p in COHORT if p["dyslipidemia"])
    n_liver     = sum(1 for p in COHORT if p["liver_disease"])
    n_psych     = sum(1 for p in COHORT if p["psychiatric"])
    n_carb_av   = sum(1 for p in COHORT if p["carb_aversion"])
    n_ctln2     = sum(1 for p in COHORT if p["phenotype"] == "CTLN2 (Adult)")
    n_niccd     = sum(1 for p in COHORT if p["phenotype"] == "NICCD (Neonatal)")
    n_fttdcd    = sum(1 for p in COHORT if p["phenotype"] == "FTTDCD (Child)")
    avg_citr    = round(sum(p["citrulline_umol_l"] for p in COHORT) / n)
    avg_nh3     = round(sum(p["nh3_peak_umol_l"] for p in COHORT) / n)
    avg_thr     = round(sum(p["threonine_umol_l"] for p in COHORT) / n)

    return {
        "disease": "Citrin Deficiency — SLC25A13 (Aspartate-Glutamate Carrier 2) Deficiency",
        "omim_gene": "603859",
        "omim_disease_ctln2": "603471",
        "omim_disease_niccd": "605814",
        "omim_disease_fttdcd": "222700",
        "gene": "SLC25A13",
        "alias": "Citrin / AGC2 / Aralar2",
        "chromosome": "7q21.3",
        "inheritance": "Autosomal Recessive (AR)",
        "protein": "675 aa; inner mitochondrial membrane; aspartate-glutamate carrier 2; EF-hand Ca²⁺-sensing",
        "prevalence": "~1:17,000–20,000 East Asia; most common UCD in East Asian adults; ~1:100,000 worldwide",
        "mechanism": "Aspartate CANNOT exit mitochondria → ASS1 (step 3) stalled (no aspartate substrate) → citrulline ACCUMULATES; malate-aspartate shuttle impaired → carbohydrate intolerance",
        "n_patients": n,
        "kpi": {
            "n_patients":    {"value": n,                             "label": "Cohort size",                             "color": "#0d47a1"},
            "citrulline_avg":{"value": f"{avg_citr} µmol/L",          "label": "Mean Plasma Citrulline (VERY HIGH CTLN2)", "color": "#b71c1c"},
            "nh3_avg":       {"value": f"{avg_nh3} µmol/L",           "label": "Mean Peak NH3 (episodic crises)",         "color": "#e65100"},
            "thr_avg":       {"value": f"{avg_thr} µmol/L",           "label": "Mean Threonine ↑ (MAS defect)",           "color": "#6a1b9a"},
            "seizures_pct":  {"value": f"{round(n_seizures/n*100)}%", "label": "Seizures (%)",                            "color": "#c62828"},
            "dyslip_pct":    {"value": f"{round(n_dyslip/n*100)}%",   "label": "Dyslipidemia (CTLN2 hallmark)",           "color": "#1565c0"},
            "liver_pct":     {"value": f"{round(n_liver/n*100)}%",    "label": "Liver Disease (%)",                       "color": "#4e342e"},
            "psych_pct":     {"value": f"{round(n_psych/n*100)}%",    "label": "Psychiatric Features (CTLN2)",            "color": "#37474f"},
            "carb_aversion_pct": {"value": f"{round(n_carb_av/n*100)}%", "label": "Carbohydrate Aversion (PATHOGNOMONIC)", "color": "#2e7d32"},
            "ctln2_pct":     {"value": f"{round(n_ctln2/n*100)}%",   "label": "CTLN2 Adult (%)",                         "color": "#0d47a1"},
        },
        "phenotype_dist": [
            {"class": "CTLN2 (Adult)",    "n": n_ctln2,  "pct": round(n_ctln2/n*100)},
            {"class": "NICCD (Neonatal)", "n": n_niccd,  "pct": round(n_niccd/n*100)},
            {"class": "FTTDCD (Child)",   "n": n_fttdcd, "pct": round(n_fttdcd/n*100)},
        ],
        "hallmark_biomarker": (
            "Plasma Citrulline VERY HIGH (300–1000 µmol/L in CTLN2) — ASS1 stalled by absent aspartate substrate; "
            "Threonine ELEVATED (MAS defect); Argininosuccinate ABSENT (key: ASS1 cannot run without aspartate)"
        ),
        "hallmark_clinical": "Carbohydrate Aversion (PATHOGNOMONIC) + Episodic Hyperammonemia + Dyslipidemia in adults (CTLN2); Cholestasis in neonates (NICCD)",
        "carb_aversion_note": "PATHOGNOMONIC behavior: CTLN2 patients instinctively avoid carbohydrates, sugars, soft drinks — they self-select high-fat, high-protein diet without knowing why. This behavioral finding is unique and diagnostically valuable before biochemistry returns.",
        "vs_ass1_distinction": "SLC25A13 vs ASS1 (CTLN1): SAME elevated citrulline; KEY DIFFERENCES: CTLN2=adult-onset + carb aversion + dyslipidemia + liver disease prominent; ASS1=neonatal-predominant + no malate-aspartate shuttle defect + no carb aversion. Gene panel distinguishes.",
        "liver_transplant_note": "Liver transplant is CURATIVE for citrin deficiency (SLC25A13 predominantly liver-expressed). LT normalises citrulline, ammonia, and metabolic profile. Should be offered before severe encephalopathy or cirrhosis.",
        "carb_risk_note": "IV GLUCOSE HIGH RISK in CTLN2: carbohydrate loads worsen hyperammonemia (malate-aspartate shuttle blocked → cytoplasmic NADH excess → lactate → metabolic crisis). Prefer fat-based caloric support.",
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
            "gene":           "SLC25A13 (Citrin / AGC2)",
            "function":       "Mitochondrial inner membrane aspartate-glutamate carrier 2 (AGC2)",
            "reaction":       "Cytoplasmic Glutamate + H⁺ IN ↔ Mitochondrial Aspartate OUT (electrogenic)",
            "block":          "SLC25A13 LOF → aspartate CANNOT exit mitochondria → cytoplasmic aspartate DEPLETED",
            "urea_cycle_impact": "ASS1 (step 3) stalls — needs cytoplasmic aspartate as co-substrate for: Citrulline + Aspartate → Argininosuccinate",
            "shuttle_impact": "Malate-aspartate shuttle impaired → cytoplasmic NADH/NAD⁺ ratio elevated → gluconeogenesis impaired → carbohydrate intolerance",
            "citrulline_accumulation": "Citrulline VERY HIGH — ASS1 substrate backed up; same biochemical as ASS1 enzyme defect but mechanism is SUBSTRATE DEPLETION, not enzyme defect",
            "unique_feature": "CARBOHYDRATE AVERSION — patients self-restrict carbs; fat preferred; carb worsens NADH/NAD⁺ imbalance → hyperammonemia",
        },
        "differential_diagnosis": {
            "vs_ass1_ctln1": {
                "key_diff": "Same citrulline elevation, COMPLETELY different mechanisms and age",
                "ctln2_features": "SLC25A13: adult-onset MODAL + carb aversion + dyslipidemia + liver transplant curative + MAS defect → threonine, lactate elevated",
                "ctln1_features": "ASS1: neonatal-onset MODAL + no carb aversion + no MAS defect + arginine supplementation primary",
                "shared": "Plasma citrulline VERY HIGH in both; argininosuccinate ABSENT in both (ASS1 cannot make it — either enzyme LOF or substrate absent)",
                "gene_panel": "Gene sequencing distinguishes: SLC25A13 vs ASS1 variants",
            },
            "vs_otc": {
                "key_diff": "Citrulline direction OPPOSITE",
                "otc": "OTC: citrulline VERY LOW (<5 µmol/L); block upstream of citrulline production",
                "slc25a13": "SLC25A13: citrulline VERY HIGH (300–1000 µmol/L); citrulline produced but downstream step blocked",
                "shared": "Both impair urea cycle; both have orotic acid elevated",
            },
            "vs_slc25a15_hhh": {
                "key_diff": "Different SLC25 transporters; INVERTED citrulline levels",
                "slc25a15": "SLC25A15 (HHH): ornithine VERY HIGH + citrulline LOW + homocitrullinuria PRESENT",
                "slc25a13": "SLC25A13 (CTLN2): citrulline VERY HIGH + ornithine NORMAL + homocitrullinuria ABSENT",
                "shared": "Both SLC25 family; both AR; both impair urea cycle; both have elevated orotic acid",
            },
            "vs_cps1_nags": {
                "key_diff": "CPS1/NAGS: citrulline VERY LOW (cannot make citrulline); SLC25A13: citrulline VERY HIGH",
                "cps1_nags": "CPS1/NAGS: block at step 1 — carbamoyl-P not made — citrulline not synthesised",
                "slc25a13": "SLC25A13: steps 1-2 intact — citrulline produced normally — stalls at step 3 (no aspartate)",
            },
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "SLC25A13 (citrin, AGC2, OMIM *603859) encodes a 675-amino-acid mitochondrial inner membrane "
            "protein belonging to the SLC25 family of metabolite carriers. Citrin is the aspartate-glutamate "
            "carrier isoform 2 (AGC2), predominantly expressed in liver, kidney, and intestine. "
            "It transports aspartate OUT of and glutamate + H⁺ INTO the mitochondrial matrix in a "
            "1:1 electrogenic exchange. This transport is essential for: "
            "(1) The malate-aspartate shuttle (MAS) — transfers NADH reducing equivalents from the cytoplasm "
            "into mitochondria; (2) The urea cycle — provides cytoplasmic aspartate for ASS1 (step 3). "
            "Chromosome 7q21.3; autosomal recessive; EF-hand Ca²⁺-sensing N-terminal domain regulates activity."
        ),
        "pathomechanism_ctln2": (
            "SLC25A13 LOF → aspartate CANNOT exit mitochondria → cytoplasmic aspartate DEPLETED. "
            "Three cascading consequences: "
            "(1) Urea cycle block at ASS1 (step 3): ASS1 catalyses Citrulline + Aspartate + ATP → Argininosuccinate. "
            "    Without cytoplasmic aspartate, ASS1 stalls → citrulline ACCUMULATES (300–1000 µmol/L in CTLN2). "
            "    NH₃ accumulates episodically (100–400 µmol/L); MILDER than complete ASS1 enzyme deficiency "
            "    because partial compensation occurs via alternative metabolic routes in adults. "
            "(2) MAS impairment: cytoplasmic NADH cannot be transferred into mitochondria → cytoplasmic "
            "    NADH/NAD⁺ ratio ELEVATED → gluconeogenesis from lactate/amino acids impaired → "
            "    carbohydrate oxidation generates excess cytoplasmic NADH → metabolic crisis. "
            "(3) Three age phenotypes: NICCD (neonatal cholestasis; galactose metabolism impaired; "
            "    self-resolves in most) → FTTDCD (failure to thrive; subclinical) → CTLN2 (adult; "
            "    episodic hyperammonemia; psychiatric; liver disease)."
        ),
        "carbohydrate_aversion_mechanism": (
            "Carbohydrate aversion in CTLN2 is a PATHOGNOMONIC behavioral finding explained by: "
            "Carbohydrates (glucose, fructose, sucrose) are oxidised via glycolysis → NADH produced in cytoplasm. "
            "Normally: cytoplasmic NADH is transferred into mitochondria via the malate-aspartate shuttle (MAS). "
            "In SLC25A13 LOF: MAS is impaired → cytoplasmic NADH CANNOT be efficiently re-oxidised → "
            "pyruvate converted to LACTATE (via LDH) instead of entering TCA → lactic acidosis + energy deficit. "
            "Additionally: glucose metabolism increases demand for aspartate via gluconeogenesis and anaplerosis, "
            "but cytoplasmic aspartate is already depleted → double burden. "
            "Result: carbohydrate ingestion worsens the metabolic state (lactate, NH₃) → patients develop "
            "aversion as a survival behaviour. CTLN2 patients prefer protein + fat (bypasses MAS dependence). "
            "This behaviour can appear years before CTLN2 diagnosis — ASK about it in work-up."
        ),
        "niccd_mechanism": (
            "NICCD (Neonatal Intrahepatic Cholestasis caused by Citrin Deficiency): "
            "In neonates, dietary lactose provides galactose which requires galactose-1-P-uridyltransferase + "
            "galactose metabolism. SLC25A13 LOF impairs the hepatic NADH/NAD⁺ balance critical for galactose "
            "metabolism → galactosemia-like picture (elevated galactose, AFP). "
            "Cholestasis: impaired bile acid export (secondary to MAS dysfunction and hepatocellular energy deficit). "
            "Clinical: prolonged neonatal jaundice (direct hyperbilirubinaemia), elevated transaminases, "
            "galactosuria, AFP elevated, hypoproteinaemia. "
            "NICCD management: MCT-based lactose-free formula; fat-soluble vitamin supplementation. "
            "NICCD often self-resolves by 12 months as diet shifts away from lactose/galactose — "
            "BUT the child then enters the FTTDCD phase and may later develop CTLN2."
        ),
        "vs_ass1_distinction": (
            "SLC25A13 (CTLN2) vs ASS1 (CTLN1) — the citrullinemia twins: "
            "SAME biochemical hallmark (citrulline VERY HIGH; argininosuccinate ABSENT). "
            "OPPOSITE MECHANISM: "
            "  CTLN1 (ASS1): enzyme deficiency → citrulline present but ASS1 ENZYME cannot catalyse step 3. "
            "  CTLN2 (SLC25A13): enzyme intact → ASS1 enzyme present but SUBSTRATE (aspartate) absent. "
            "DIFFERENT AGE OF ONSET: "
            "  CTLN1: predominantly neonatal hyperammonemia crisis; citrulline >500 µmol/L neonatal. "
            "  CTLN2: predominantly adult onset (mean 30–50 years); NICCD in neonates (different phenotype). "
            "DIFFERENT FEATURES: "
            "  CTLN2: carb aversion (PATHOGNOMONIC); dyslipidemia; psychiatric features; liver transplant curative. "
            "  CTLN1: NO carb aversion; arginine supplementation primary; liver transplant less central. "
            "ETHNIC DISTRIBUTION: "
            "  CTLN2: predominantly East Asian (Japan, China, Korea); SLC25A13 founder alleles. "
            "  CTLN1: pan-ethnic; no ethnic clustering."
        ),
        "liver_transplant_rationale": (
            "Liver transplant is CURATIVE in citrin deficiency and strongly recommended for CTLN2. "
            "Rationale: SLC25A13 is PREDOMINANTLY liver-expressed (highest expression: liver, kidney, intestine). "
            "LT replaces citrin-deficient hepatocytes with normal donor hepatocytes → "
            "aspartate-glutamate transport restored → ASS1 (step 3) can function → "
            "citrulline normalises → NH₃ normalises → malate-aspartate shuttle restored. "
            "Outcomes: excellent biochemical normalisation; neurological recovery if transplanted before "
            "severe encephalopathy or cirrhosis. Recurrence: NOT expected (liver-expressed disease corrected). "
            "Timing: offer BEFORE advanced liver disease, severe psychiatric episodes, or repeated coma. "
            "Note: NICCD typically does NOT require LT; FTTDCD rarely requires LT; CTLN2 = main indication."
        ),
        "east_asian_founder_variants": (
            "SLC25A13 has major founder variants predominantly in East Asian populations: "
            "c.IVS16+1G>A (IVS16ins3kb): 3-kilobase insertion (1.7kb Alu + 1.3kb sequence) in intron 16 → "
            "  cryptic splice activation → premature stop codon. Accounts for ~50-60% of alleles in Japan "
            "  and significant proportions in Korea, China. Most common SLC25A13 mutation worldwide. "
            "c.1638_1660dup23: 23bp duplication in exon 17 → frameshift → premature stop. "
            "  Second most common Japanese/Korean. ~15% Japanese alleles. "
            "p.Arg605Gln: missense in TM/EF-hand junction. Most common Southern Chinese, Vietnamese, Malaysian. "
            "Combined carrier frequency in Japan: ~1:65–70. Newborn screening expanding. "
            "Diagnosis in East Asian patients with elevated citrulline: test SLC25A13 before ASS1."
        ),
        "seizure_management": (
            "Seizures in citrin deficiency arise from: "
            "(1) Acute hyperammonemia (NH₃ >100 µmol/L → cortical hyperexcitability; NMDA dysregulation). "
            "(2) Hepatic encephalopathy (CTLN2 liver disease → toxin accumulation). "
            "(3) Hypoglycemia (NICCD/FTTDCD: fasting hypoglycemia → excitotoxic seizures). "
            "Primary: control NH₃ via diet (high-fat, low-carb) + arginine + nitrogen scavenging. "
            "AED first-line: LEV (no hepatotoxicity, no urea cycle interference). "
            "ABSOLUTE CI: VPA (hepatotoxic + NH₃ worsening — triple CI in citrin deficiency). "
            "AVOID high-carbohydrate loading (worsens hyperammonemia via MAS pathway). "
            "Liver transplant definitively prevents further encephalopathic and hyperammonemic seizures in CTLN2."
        ),
    }
