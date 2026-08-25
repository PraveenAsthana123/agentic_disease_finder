#!/usr/bin/env python3
"""IVD (Isovaleryl-CoA Dehydrogenase) Deficiency — Isovaleric Acidemia (IVA) Dashboard.

IVD encodes isovaleryl-CoA dehydrogenase, a mitochondrial FAD-dependent ACAD-family enzyme:
  Isovaleryl-CoA + FAD  →  [IVD]  →  3-Methylcrotonyl-CoA + FADH₂
  (step 2 in leucine catabolism — after branched-chain aminotransferase removes the amino group)

IVD LOF → Isovaleryl-CoA CANNOT be dehydrogenated → accumulates:
  → Isovaleric acid (IVA): ELEVATED (FREE ACID — "sweaty feet" odor PATHOGNOMONIC)
  → Isovalerylglycine (IVG): ELEVATED in urine (more stable, PATHOGNOMONIC primary biomarker)
  → Isovalerylcarnitine (C5): ELEVATED (newborn screening PRIMARY MARKER)
  → 3-Hydroxy-isovalerate: mildly elevated (oxidation side-product)
  → Secondary glycine ELEVATED (substrate for IVG conjugation → depleted)
  → Secondary free carnitine LOW (conjugation to C5 → depleted)

OMIM Disease: #243500 (Isovaleric acidemia)
OMIM Gene:   *607036 (IVD)
Chromosome: 15q15.1
Inheritance: Autosomal Recessive (AR)
Protein: 415 aa; mitochondrial matrix; homotetrameric; FAD-dependent acyl-CoA dehydrogenase (ACAD family)
Prevalence: ~1:100,000–250,000 (NBS era); may be commoner due to mild NBS-detected variants

MECHANISM — ISOVALERYL-CoA ACCUMULATION + CONJUGATION DETOXIFICATION:
  Normal IVD: Isovaleryl-CoA (from leucine catabolism) is dehydrogenated to 3-methylcrotonyl-CoA
  IVD LOF: isovaleryl-CoA accumulates → spontaneous hydrolysis → free isovaleric acid (IVA)
  IVA is volatile → "sweaty feet" odor; crosses BBB → neurological toxicity
  Two conjugation pathways provide detoxification:
    (1) Glycine conjugation: isovaleryl-CoA + glycine → isovalerylglycine (IVG) — MOST IMPORTANT
        IVG is water-soluble, renal-excreted, non-toxic → MAJOR DETOX ROUTE
        IVG MORE STABLE than IVA → better urinary biomarker
    (2) Carnitine conjugation: isovaleryl-CoA + carnitine → C5 (isovalerylcarnitine) → RENAL EXCRETION
        Both glycine + carnitine supplementation ACCELERATE these pathways

LEUCINE CATABOLISM CONTEXT:
  Leucine → α-ketoisocaproate (KIC) [BCAT → Step 1] → isovaleryl-CoA [BCKDH → Step 1b]
  → [IVD → Step 2, BLOCKED] → 3-methylcrotonyl-CoA → 3-methylglutaconyl-CoA → HMG-CoA
  → acetyl-CoA + acetoacetate (ketogenesis — ketogenic amino acid)
  IVD LOF: Leucine catabolism stalls at step 2 → isovaleryl-CoA accumulates
"""

import random

SEED       = 253      # next after MMACHC (seed 247), continuing series
N_PATIENTS = 40

# Phenotypic classes
PHENOTYPE_CLASSES = [
    {"name": "Classic neonatal acute",   "pct": 35, "note": "Days 1-5 of life: vomiting, encephalopathy, sweaty-feet odor, metabolic acidosis, mild NH3 elevation; life-threatening without treatment"},
    {"name": "Chronic intermittent",     "pct": 40, "note": "Episodic metabolic crises triggered by febrile illness, fasting, or high-protein intake; sweaty-feet odor during crisis; normal interictal"},
    {"name": "NBS-detected asymptomatic","pct": 25, "note": "Found by newborn screening (C5 elevation); often p.Leu13Pro; may remain asymptomatic with dietary management; mild or no clinical disease"},
]

# Top pathogenic variants
VARIANTS = [
    {"variant": "p.Leu13Pro (c.38T>C)",      "freq": 32, "domain": "Mitochondrial targeting sequence / N-terminal fold", "phenotype": "Mild; NBS-detected; protein mis-folds but retains partial activity", "note": "MOST COMMON worldwide (30-35% allele freq); predominantly detected by NBS with C5 elevation but mild or absent phenotype; 'NBS allele'; protein instability but partial leucine catabolism retained; homozygous → very mild / asymptomatic course"},
    {"variant": "p.Ala282Val (c.845C>T)",    "freq": 17, "domain": "FAD-binding domain",                                  "phenotype": "Classic severe; broad ethnic distribution",        "note": "Second most common; FAD binding impaired → complete loss of dehydrogenase activity; classic neonatal or intermittent severe phenotype; pan-ethnic; IVG markedly elevated"},
    {"variant": "c.IVS9+1G>A",               "freq": 10, "domain": "Splice donor — intron 9 disrupted",                  "phenotype": "Null allele; classic severe neonatal",             "note": "Splice donor destruction → exon 9 skipping or intron retention → premature stop → null; severe neonatal onset; marked IVG + C5 elevation; no residual activity"},
    {"variant": "p.Arg363Cys (c.1087C>T)",   "freq": 8,  "domain": "Active site loop — substrate binding",               "phenotype": "Classic severe; pan-ethnic",                       "note": "Active site disruption → complete loss; classic neonatal; high IVG; carnitine depletion; no cofactor response"},
    {"variant": "p.Pro330Ser (c.988C>T)",    "freq": 7,  "domain": "FAD-binding β-strand",                              "phenotype": "Intermediate; Mennonite + Dutch founder",          "note": "FAD-binding domain; partial residual activity; intermediate-to-severe phenotype; founder effect in Old Order Mennonite and Dutch populations"},
    {"variant": "p.Gly289Asp (c.866G>A)",    "freq": 6,  "domain": "Substrate channel entry",                            "phenotype": "Classic severe; European",                         "note": "Substrate channel disruption; full loss of activity; classic intermittent or neonatal; broad European distribution"},
    {"variant": "p.Thr278Ile (c.833C>T)",    "freq": 5,  "domain": "FAD-binding / substrate interface",                  "phenotype": "Moderate; variable onset",                         "note": "FAD binding partially impaired; moderate phenotype; C5 and IVG elevated but below classic severe; episodic crises with febrile illness"},
    {"variant": "c.IVS11-1G>C",              "freq": 4,  "domain": "Splice acceptor — intron 11",                        "phenotype": "Severe null allele; neonatal",                     "note": "Splice acceptor loss → exon 12 skipping → frameshift → null; neonatal severe; high IVG; not ethnic-specific"},
]

# Biomarker panel
BIOMARKERS = {
    "c5":        {"label": "Plasma C5 (Isovalerylcarnitine)",  "normal": "<0.5 µmol/L",        "status": "ELEVATED — PRIMARY NBS MARKER (classic >1.0; severe >5)",  "direction": "↑↑ NBS marker",  "color": "danger"},
    "ivg_urine": {"label": "Urine Isovalerylglycine (IVG)",    "normal": "<5 mmol/mol Cr",     "status": "PATHOGNOMONIC (>100 classic; >200 severe crisis)",          "direction": "↑↑↑ PATHOGNOMONIC", "color": "danger"},
    "iva":       {"label": "Plasma Isovaleric Acid (IVA)",     "normal": "<10 µmol/L",         "status": "ELEVATED — SWEATY FEET odor (volatile biomarker)",         "direction": "↑↑ KEY POSITIVE",  "color": "danger"},
    "3oh_iva":   {"label": "Urine 3-OH-Isovalerate",           "normal": "<30 mmol/mol Cr",    "status": "ELEVATED (50–300; secondary oxidation product)",            "direction": "↑ ELEVATED",        "color": "warning"},
    "carnitine": {"label": "Free Carnitine (plasma)",          "normal": "25–60 µmol/L",       "status": "LOW (secondary depletion — C5 conjugation)",               "direction": "↓ LOW",             "color": "warning"},
    "glycine":   {"label": "Plasma Glycine",                   "normal": "150–350 µmol/L",     "status": "VARIABLE (may be high if not supplemented; conjugation substrate)", "direction": "↑/→ VARIABLE", "color": "warning"},
    "leucine":   {"label": "Plasma Leucine",                   "normal": "60–170 µmol/L",      "status": "MILDLY ELEVATED (substrate before block; classic >200)",   "direction": "↑ MILDLY",         "color": "warning"},
    "nh3":       {"label": "Plasma Ammonia",                   "normal": "<50 µmol/L",         "status": "NORMAL to mildly elevated (acute crisis: 80–200; KEY NEGATIVE vs UCDs)", "direction": "→/↑ MILD", "color": "success"},
    "mma":       {"label": "Methylmalonic Acid (MMA)",         "normal": "<0.4 µmol/L",        "status": "NORMAL — KEY NEGATIVE vs MMUT/cblA/PCCA/PCCB",            "direction": "→ NORMAL",          "color": "success"},
    "methylcit": {"label": "Urine Methylcitrate",              "normal": "<5 mmol/mol Cr",     "status": "ABSENT — KEY NEGATIVE vs Propionic Acidemia (PCCA/PCCB)",  "direction": "→ ABSENT",          "color": "success"},
    "c5oh":      {"label": "Plasma C5-OH (3-Methylcrotonyl)",  "normal": "<0.3 µmol/L",        "status": "NORMAL — KEY NEGATIVE vs MCC (MCCC1/MCCC2) and HLCS/BTD",  "direction": "→ NORMAL",          "color": "success"},
    "thcy":      {"label": "Total Homocysteine",               "normal": "<15 µmol/L",         "status": "NORMAL — KEY NEGATIVE vs CBS/cblC",                        "direction": "→ NORMAL",          "color": "success"},
}

# Treatments (evidence-graded)
TREATMENTS = [
    {
        "therapy": "Glycine supplementation",
        "level": "A",
        "dose": "150–300 mg/kg/day in 3 divided doses; titrate to normalise IVG excretion (target IVG < 200 mmol/mol Cr interictal); increase to 400 mg/kg/day in acute crisis",
        "rationale": "GLYCINE IS THE CORNERSTONE DETOX ROUTE in IVD. Isovaleryl-CoA + glycine → isovalerylglycine (IVG) (catalysed by glycine N-acyltransferase in liver/kidney). IVG is water-soluble, non-toxic, renally excreted → removes isovaleryl-CoA from the metabolic pool. Glycine supplementation: (1) replenishes the glycine substrate pool depleted by high IVG conjugation; (2) drives the conjugation equilibrium towards IVG formation; (3) accelerates isovaleryl-CoA clearance. UNIQUE to IVD — glycine is not a standard treatment in other organic acidemias. During acute crisis, glycine is the FIRST line with IV glucose.",
        "class": "Conjugation accelerator (primary — UNIQUE to IVD)",
    },
    {
        "therapy": "Leucine-restricted diet + IVA-free formula",
        "level": "A",
        "dose": "Natural protein 0.8–1.5 g/kg/day; leucine intake < 100–150 mg/kg/day (ISVA-free formula for protein target); target plasma leucine 100–200 µmol/L",
        "rationale": "Leucine is the sole substrate for the IVD pathway. Restricting leucine reduces flux through BCKDH → isovaleryl-CoA formation → less IVA/IVG accumulation. Leucine-free (isovaleric acid-free) amino acid formula provides protein adequacy without contributing substrate. Critical: avoid over-restriction → leucine is essential; growth failure from leucine deficiency occurs if formula is not adequately supplemented.",
        "class": "Dietary (primary — substrate restriction)",
    },
    {
        "therapy": "L-Carnitine supplementation",
        "level": "A",
        "dose": "100–200 mg/kg/day oral; IV during acute crises (100 mg/kg over 30 min then infusion); titrate to normalise free carnitine (> 25 µmol/L)",
        "rationale": "IVD LOF → isovaleryl-CoA + carnitine → C5 (isovalerylcarnitine) → renally excreted. This secondary conjugation pathway depletes free carnitine → secondary carnitine deficiency. Carnitine supplementation: (1) replenishes free carnitine pool for energy metabolism (fatty acid β-oxidation); (2) enhances C5 excretion (detoxification synergistic with glycine); (3) protects cardiac and skeletal muscle. Particularly important in acute crises when isovaleryl-CoA flux is maximal.",
        "class": "Carnitine supplementation (secondary detox + deficiency correction)",
    },
    {
        "therapy": "IV glucose (anti-catabolic) during crises",
        "level": "A",
        "dose": "10% dextrose IV at 8–12 mg/kg/min (GIR); +/- insulin 0.05–0.1 U/kg/hr to prevent hyperglycaemia; continue until leucine levels normalising and clinical improvement",
        "rationale": "Febrile illness / fasting → catabolic state → increased protein turnover → increased leucine catabolism → MORE isovaleryl-CoA → MORE IVA/IVG → metabolic crisis. HIGH-DOSE GLUCOSE: (1) anti-catabolic — suppresses endogenous protein breakdown and leucine release; (2) provides energy substrate bypassing the IVD block (glucose-derived acetyl-CoA bypasses ketogenic leucine pathway); (3) insulin co-administration enhances anabolism. Combine with glycine + carnitine for complete crisis management.",
        "class": "Emergency anti-catabolic (crisis management)",
    },
    {
        "therapy": "LEV (levetiracetam)",
        "level": "B",
        "dose": "10–40 mg/kg/day; first-line AED if seizures occur",
        "rationale": "Seizures occur in 25–35% of IVD patients with basal ganglia involvement or global cortical toxicity from IVA. LEV: no hepatotoxicity, no leucine catabolism interference, no carnitine depletion. SV2A mechanism is safe in the IVD metabolic milieu. Preferred over enzyme-inducing AEDs which increase metabolic demand.",
        "class": "AED — first-line",
    },
    {
        "therapy": "Riboflavin (FAD cofactor therapy)",
        "level": "C",
        "dose": "100–300 mg/day pharmacological dose; trial for 3 months; assess IVG/C5 reduction > 50%",
        "rationale": "IVD is FAD-dependent. Some missense variants (particularly FAD-binding domain mutations like p.Ala282Val, p.Pro330Ser) retain apoenzyme that may be stabilised by pharmacological FAD supplementation (riboflavin → FMN → FAD). Limited clinical evidence (case reports); no RCT; biochemical response predicts potential clinical benefit. Add as adjunct to dietary + glycine/carnitine therapy if FAD-binding domain variant confirmed.",
        "class": "Cofactor (FAD-binding variant subset — experimental)",
    },
    {
        "therapy": "Valproate (VPA)",
        "level": "ABSOLUTE CI",
        "dose": "CONTRAINDICATED — do not use; replace with LEV",
        "rationale": "VPA in IVD is ABSOLUTELY CONTRAINDICATED via triple mechanism: (1) Competitive substrate inhibition: valproyl-CoA (VPA metabolite) is an isovaleryl-CoA structural analogue → directly COMPETES with isovaleryl-CoA at the IVD active site → WORSENS the primary enzyme block (valproyl-CoA inhibits ALREADY DEFICIENT IVD); (2) Carnitine depletion: VPA → valproyl-carnitine → depletes free carnitine → abolishes the carnitine conjugation detox pathway; (3) Hepatotoxicity: VPA fatty acid oxidation inhibition + carnitine depletion → mitochondrial failure in the setting of ongoing isovaleryl-CoA accumulation. FATAL without glycine + carnitine resuscitation. Use LEV (levetiracetam) ALWAYS.",
        "class": "ABSOLUTE CI — FATAL (competitive IVD inhibition + carnitine depletion)",
    },
    {
        "therapy": "Fasting / high-leucine protein loads",
        "level": "EXTREME HAZARD",
        "dose": "NEVER fast > 4–6h; avoid high-leucine foods (whey, dairy, meat in excess); written emergency plan for illness",
        "rationale": "Fasting triggers endogenous protein catabolism → massive leucine release → isovaleryl-CoA surge → acute metabolic crisis. High-leucine protein (whey protein, BCAA supplements, high-meat diet) provides direct substrate for IVD pathway. MOST CRISES ARE TRIGGERED by intercurrent illness + fasting. Emergency glucopolymer/glucose drinks for home management; IV glucose for hospital admission.",
        "class": "Dietary hazard (crisis trigger)",
    },
]

# Seizure types
SEIZURE_TYPES = [
    {"type": "Focal / complex partial seizures",                 "pct": 25, "note": "Basal ganglia + cortical involvement from IVA toxicity; frontal-temporal predominance; LEV first-line"},
    {"type": "GTCS (acute metabolic crisis)",                    "pct": 20, "note": "During acute neonatal or intermittent crisis; IVA neurotoxicity + metabolic encephalopathy; resolves with metabolic correction"},
    {"type": "Infantile spasms (acute neonatal phase)",          "pct": 12, "note": "Classic neonatal phenotype; encephalopathic period; West syndrome-like; metabolic correction + ACTH trial"},
    {"type": "Myoclonic seizures",                               "pct": 15, "note": "IVA disrupts GABAergic and glutamatergic transmission; myoclonus + absence variants; LEV, clonazepam"},
    {"type": "Drug-resistant epilepsy (structural substrate)",   "pct": 10, "note": "Repeat crises → basal ganglia + cortical injury → refractory focal epilepsy; multi-AED; avoid VPA absolutely"},
    {"type": "Febrile seizures (crisis-triggered)",              "pct": 30, "note": "Most common seizure presentation in intermittent IVA; fever → crisis → seizure; metabolic management + AED"},
]

# Systemic features
SYSTEMIC_FEATURES = [
    {"feature": '"Sweaty feet" / cheesy odor',                    "pct": 90, "note": "PATHOGNOMONIC: isovaleric acid (volatile, odorous) smells like sweaty feet or ripe cheese; strongest during acute crisis; present interictal at high IVA levels; recognisable at bedside"},
    {"feature": "Acute metabolic encephalopathy (crisis)",        "pct": 70, "note": "Neonatal or episodic: vomiting, lethargy, coma; triggered by febrile illness/fasting; IVA crosses BBB → neurotoxicity; reverses with glycine + carnitine + glucose"},
    {"feature": "Metabolic acidosis (high anion gap)",            "pct": 75, "note": "IVA dissociates at physiological pH → isovalerate anion → high anion gap (25–40 mEq/L during crisis); bicarbonate/Bicarb supplementation during crisis"},
    {"feature": "Bone marrow suppression (neutropenia/thrombocytopenia)", "pct": 30, "note": "Isovaleryl-CoA accumulation inhibits mitochondrial fatty acid oxidation in rapidly dividing marrow cells → pancytopenia during decompensation; reverses with metabolic control"},
    {"feature": "Pancreatitis",                                   "pct": 15, "note": "Acute pancreatitis reported in intermittent decompensation; IVA may be directly toxic to pancreatic acinar cells; manage with nil-by-mouth + IV glucose"},
    {"feature": "Intellectual disability (post-crisis)",          "pct": 25, "note": "Severity proportional to number/severity of crises; NBS-detected + treated → normal cognition in majority; untreated classic → moderate-severe IDD"},
    {"feature": "Basal ganglia changes (MRI)",                    "pct": 20, "note": "Bilateral putaminal + caudate signal changes on T2 MRI after severe crisis; similar to other organic acidemias; related to IVA neurotoxicity + energy failure"},
    {"feature": "Cardiomyopathy",                                 "pct": 10, "note": "Less common than in PA; isovaleryl-CoA inhibits mitochondrial respiratory chain (Complex I) in cardiomyocytes; dilated cardiomyopathy if severe/recurrent decompensation; echocardiography monitoring"},
]


def _generate_cohort():
    random.seed(SEED)
    patients = []
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls["name"]] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append("Chronic intermittent")
    phenotype_dist = phenotype_dist[:N_PATIENTS]
    random.shuffle(phenotype_dist)

    for i in range(N_PATIENTS):
        phenotype = phenotype_dist[i]

        if phenotype == "NBS-detected asymptomatic":
            age_onset_months    = round(random.uniform(0, 0.5), 2)   # detected at NBS
            c5_umol             = round(random.uniform(0.6, 2.5), 2) # mild elevation
            ivg_urine           = random.randint(20, 120)
            free_carnitine      = random.randint(18, 35)
            nh3_umol            = random.randint(30, 60)
            crisis_count        = 0
            sweaty_feet         = random.random() < 0.35             # mild/intermittent
            idd                 = False
            bg_mri              = False
            cardiomyopathy      = False
        elif phenotype == "Classic neonatal acute":
            age_onset_months    = round(random.uniform(0.03, 0.15), 3)  # days 1-5
            c5_umol             = round(random.uniform(3.0, 15.0), 1)
            ivg_urine           = random.randint(300, 1200)
            free_carnitine      = random.randint(3, 15)
            nh3_umol            = random.randint(80, 280)
            crisis_count        = random.randint(1, 3)
            sweaty_feet         = True
            idd                 = random.random() < 0.45
            bg_mri              = random.random() < 0.30
            cardiomyopathy      = random.random() < 0.12
        else:  # Chronic intermittent
            age_onset_months    = round(random.uniform(3, 60), 1)
            c5_umol             = round(random.uniform(1.0, 6.0), 2)
            ivg_urine           = random.randint(80, 500)
            free_carnitine      = random.randint(8, 28)
            nh3_umol            = random.randint(40, 180)
            crisis_count        = random.randint(1, 8)
            sweaty_feet         = random.random() < 0.85
            idd                 = random.random() < 0.20
            bg_mri              = random.random() < 0.15
            cardiomyopathy      = random.random() < 0.08

        seizures    = random.random() < 0.30
        dre         = seizures and random.random() < 0.15
        bms         = random.random() < 0.28
        pancreatitis= random.random() < 0.12
        v           = random.choice(VARIANTS)

        patients.append({
            "id":                    f"IVD-{i+1:03d}",
            "phenotype":             phenotype,
            "age_onset_months":      age_onset_months,
            "c5_umol_l":             c5_umol,
            "ivg_urine_mmol_mol_cr": ivg_urine,
            "free_carnitine_umol_l": free_carnitine,
            "nh3_umol_l":            nh3_umol,
            "crisis_count":          crisis_count,
            "sweaty_feet_odor":      sweaty_feet,
            "seizures":              seizures,
            "dre":                   dre,
            "idd":                   idd,
            "bg_mri_changes":        bg_mri,
            "cardiomyopathy":        cardiomyopathy,
            "bone_marrow_suppression": bms,
            "pancreatitis":          pancreatitis,
            "variant":               v["variant"],
        })
    return patients


COHORT = _generate_cohort()


def get_overview():
    n               = len(COHORT)
    n_sweaty        = sum(1 for p in COHORT if p["sweaty_feet_odor"])
    n_seizures      = sum(1 for p in COHORT if p["seizures"])
    n_dre           = sum(1 for p in COHORT if p["dre"])
    n_idd           = sum(1 for p in COHORT if p["idd"])
    n_bg            = sum(1 for p in COHORT if p["bg_mri_changes"])
    n_cardi         = sum(1 for p in COHORT if p["cardiomyopathy"])
    n_neonatal      = sum(1 for p in COHORT if p["phenotype"] == "Classic neonatal acute")
    n_intermit      = sum(1 for p in COHORT if p["phenotype"] == "Chronic intermittent")
    n_nbs           = sum(1 for p in COHORT if p["phenotype"] == "NBS-detected asymptomatic")
    avg_c5          = round(sum(p["c5_umol_l"] for p in COHORT) / n, 2)
    avg_ivg         = round(sum(p["ivg_urine_mmol_mol_cr"] for p in COHORT) / n)
    avg_carnitine   = round(sum(p["free_carnitine_umol_l"] for p in COHORT) / n, 1)
    avg_nh3         = round(sum(p["nh3_umol_l"] for p in COHORT) / n)

    return {
        "disease": "Isovaleric Acidemia (IVA) — IVD Deficiency",
        "omim_gene": "607036",
        "omim_disease": "243500",
        "gene": "IVD",
        "alias": "Isovaleryl-CoA Dehydrogenase / IVA / Isovalericacidemia",
        "chromosome": "15q15.1",
        "inheritance": "Autosomal Recessive (AR)",
        "protein": "415 aa; mitochondrial matrix homotetrameric enzyme; FAD-dependent acyl-CoA dehydrogenase (ACAD family)",
        "prevalence": "~1:100,000–250,000; may be commoner in NBS era due to mild p.Leu13Pro variant detection",
        "mechanism": (
            "Isovaleryl-CoA (from leucine catabolism) CANNOT be dehydrogenated → "
            "isovaleric acid + IVG accumulate; "
            "IVA is volatile and neurotoxic → 'sweaty feet' odor + encephalopathy; "
            "C5 (isovalerylcarnitine) elevated on NBS; "
            "glycine conjugation → IVG is the PRIMARY detox and MOST SPECIFIC urinary biomarker"
        ),
        "n_patients": n,
        "kpi": {
            "n_patients":    {"value": n,                                "label": "Cohort size",                               "color": "#1a237e"},
            "c5_avg":        {"value": f"{avg_c5} µmol/L",               "label": "Mean C5 Isovalerylcarnitine (NBS marker)",  "color": "#b71c1c"},
            "ivg_avg":       {"value": f"{avg_ivg} mmol/mol Cr",         "label": "Mean Urine IVG (PATHOGNOMONIC)",            "color": "#c62828"},
            "carnitine_avg": {"value": f"{avg_carnitine} µmol/L",        "label": "Mean Free Carnitine (LOW)",                 "color": "#6a1b9a"},
            "nh3_avg":       {"value": f"{avg_nh3} µmol/L",              "label": "Mean NH3 (mildly elevated in crisis)",      "color": "#e65100"},
            "sweaty_pct":    {"value": f"{round(n_sweaty/n*100)}%",      "label": "'Sweaty feet' odor (PATHOGNOMONIC)",        "color": "#0d47a1"},
            "seizures_pct":  {"value": f"{round(n_seizures/n*100)}%",    "label": "Seizures (%)",                              "color": "#b71c1c"},
            "dre_pct":       {"value": f"{round(n_dre/n*100)}%",         "label": "Drug-Resistant Epilepsy (%)",               "color": "#bf360c"},
            "idd_pct":       {"value": f"{round(n_idd/n*100)}%",         "label": "Intellectual Disability (%)",               "color": "#37474f"},
            "bg_mri_pct":    {"value": f"{round(n_bg/n*100)}%",          "label": "Basal Ganglia MRI Changes (%)",             "color": "#880e4f"},
        },
        "phenotype_dist": [
            {"class": "Classic neonatal acute",    "n": n_neonatal, "pct": round(n_neonatal/n*100)},
            {"class": "Chronic intermittent",      "n": n_intermit, "pct": round(n_intermit/n*100)},
            {"class": "NBS-detected asymptomatic", "n": n_nbs,      "pct": round(n_nbs/n*100)},
        ],
        "hallmark_biomarker": (
            "C5 (isovalerylcarnitine) ELEVATED on NBS (> 0.5 µmol/L; PRIMARY SCREENING MARKER); "
            "Urine Isovalerylglycine (IVG) PATHOGNOMONIC (most specific biomarker; > 100 mmol/mol Cr classic); "
            "Isovaleric acid ELEVATED (volatile — 'sweaty feet' odor); "
            "Free Carnitine LOW (secondary depletion); "
            "MMA ABSENT, Methylcitrate ABSENT (KEY NEGATIVES vs PA/MMA)"
        ),
        "hallmark_clinical": (
            "'Sweaty feet / cheesy' odor (PATHOGNOMONIC from isovaleric acid) + "
            "acute metabolic encephalopathy triggered by febrile illness or fasting"
        ),
        "glycine_unique_note": (
            "GLYCINE SUPPLEMENTATION IS PATHOGNOMONIC TREATMENT — unique to IVD among organic acidemias. "
            "Glycine drives the isovaleryl-CoA → IVG conjugation route → IVG is non-toxic + water-soluble + renally excreted. "
            "IVG is also the most specific urinary diagnostic biomarker."
        ),
        "vpa_absolute_ci_note": (
            "VALPROATE ABSOLUTELY CONTRAINDICATED: valproyl-CoA structurally mimics isovaleryl-CoA → "
            "competes at IVD active site → worsens the PRIMARY enzyme defect. "
            "Plus carnitine depletion → abolishes the detox pathway. FATAL."
        ),
        "nbs_note": (
            "C5 (isovalerylcarnitine) detected by tandem MS/MS on NBS filter card. "
            "IMPORTANT: C5 can also be elevated in 2-methylbutyryl-CoA dehydrogenase (SBCAD) deficiency — "
            "distinguish by urine IVG (elevated in IVD; absent in SBCAD) and gene panel. "
            "p.Leu13Pro allele may give borderline C5 — urine IVG is more sensitive."
        ),
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
        "enzyme_mechanism": {
            "gene":       "IVD (Isovaleryl-CoA Dehydrogenase)",
            "function":   "FAD-dependent mitochondrial enzyme (ACAD family); catalyses dehydrogenation of isovaleryl-CoA in leucine catabolism",
            "reaction":   "Isovaleryl-CoA + FAD → 3-Methylcrotonyl-CoA + FADH₂ (step 2 of leucine catabolism)",
            "block":      "IVD LOF → isovaleryl-CoA CANNOT be oxidised → spontaneous hydrolysis → isovaleric acid (IVA) + IVG via glycine conjugation",
            "detox_glycine": "Glycine + isovaleryl-CoA → IVG (catalysed by glycine N-acyltransferase) → IVG renally excreted (NON-TOXIC) — MOST IMPORTANT DETOX ROUTE",
            "detox_carnitine": "Carnitine + isovaleryl-CoA → C5 (isovalerylcarnitine) → C5 renally excreted — secondary detox",
            "leucine_path": "L-Leucine → BCAT (Step 1a) → α-ketoisocaproate (KIC) → BCKDH (Step 1b) → Isovaleryl-CoA → [IVD BLOCKED] → accumulates → IVA + IVG",
            "etf_link":   "IVD transfers electrons via ETF (ETFA-ETFB) → ETFDH → Complex III of the respiratory chain (same ETF-linked pathway as MCAD/VLCAD/GCDH)",
        },
        "differential_diagnosis": {
            "vs_sbcad": {
                "key_diff": "SBCAD (short/branched-chain acyl-CoA dehydrogenase; 2-methylbutyryl-CoA dehydrogenase): C5 elevated BUT urine shows 2-methylbutyrylglycine, NOT IVG",
                "ivd": "IVD: C5 (isovalerylcarnitine) + IVG ELEVATED; 2-methylbutyrylglycine ABSENT",
                "sbcad": "SBCAD: C5 (2-methylbutyrylcarnitine) + 2-methylbutyrylglycine elevated; IVG ABSENT; often benign NBS finding",
            },
            "vs_pa": {
                "key_diff": "Propionic Acidemia (PCCA/PCCB): C3 (not C5) elevated on NBS; methylcitrate PATHOGNOMONIC (not IVG); no sweaty-feet odor",
                "ivd": "IVD: C5 elevated; methylcitrate ABSENT; IVG PATHOGNOMONIC; leucine substrate",
                "pa": "PA: C3 elevated; methylcitrate PATHOGNOMONIC; NH3 500+ severely; Ile/Val/Met/Thr precursors; no IVG; cardiomyopathy common",
            },
            "vs_mcc": {
                "key_diff": "MCC deficiency (MCCC1/MCCC2 — 3-Methylcrotonyl-CoA Carboxylase): C5-OH (3-methylcrotonylcarnitine) elevated — NOT C5; IVG absent",
                "ivd": "IVD: C5 (not C5-OH); IVG elevated; leucine → isovaleryl-CoA block BEFORE 3-methylcrotonyl-CoA",
                "mcc": "MCC: C5-OH (3-methylcrotonylcarnitine + 3-methylcrotonylglycine) elevated; C5 NORMAL; IVG ABSENT; usually benign NBS finding in majority",
            },
            "vs_mma": {
                "key_diff": "MMA (MMUT/cblA/cblB): C3 elevated; MMA urine PATHOGNOMONIC (>200 mmol/mol Cr); no C5; no IVG; no sweaty-feet odor",
                "ivd": "IVD: C5 elevated; MMA ABSENT; IVG PATHOGNOMONIC; no CKD progression",
                "mma": "MMA: C3 not C5; MMA very high; CKD progressive UNIQUE; no sweaty-feet; OHCbl response in cblA/cblB",
            },
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "IVD (Isovaleryl-CoA Dehydrogenase, OMIM *607036) encodes a 415-amino-acid mitochondrial "
            "matrix enzyme belonging to the acyl-CoA dehydrogenase (ACAD) superfamily. IVD forms a "
            "homotetramer (α4) and is FAD-dependent. It catalyses the second step of leucine catabolism: "
            "  Isovaleryl-CoA + FAD → 3-Methylcrotonyl-CoA + FADH₂ "
            "IVD transfers reducing equivalents from FADH₂ via the electron-transfer flavoprotein (ETF: "
            "ETFA-ETFB heterodimer) → ETFDH → Complex III → oxidative phosphorylation. "
            "IVD is thus ETF-linked, like MCAD, VLCAD, GCDH, and SBCAD in the ACAD family. "
            "Chromosome 15q15.1; autosomal recessive; > 80 pathogenic variants known."
        ),
        "pathomechanism": (
            "IVD LOF → isovaleryl-CoA (derived from leucine catabolism) CANNOT be dehydrogenated. "
            "Isovaleryl-CoA accumulates → spontaneous hydrolysis → free isovaleric acid (IVA). "
            "IVA is volatile (accounts for the 'sweaty feet / cheesy' odor) and crosses the blood-brain "
            "barrier → neurological toxicity (disrupts mitochondrial energy metabolism, GABAergic "
            "transmission, and protein synthesis). "
            "Two conjugation pathways provide detoxification: "
            "(1) Glycine conjugation (PRIMARY): glycine N-acyltransferase conjugates isovaleryl-CoA + "
            "    glycine → isovalerylglycine (IVG) → water-soluble, non-toxic, renally excreted. "
            "    IVG is the most SPECIFIC urinary biomarker (more stable than IVA; not volatile). "
            "(2) Carnitine conjugation: isovaleryl-CoA + carnitine → C5 (isovalerylcarnitine) → "
            "    renally excreted. C5 is the NBS biomarker. "
            "Both detox pathways are SATURABLE in severe crisis → supplementation with glycine + "
            "carnitine accelerates flux and prevents accumulation."
        ),
        "sweaty_feet_odor_mechanism": (
            "The pathognomonic 'sweaty feet / cheesy ripe cheese' odor of IVA arises from isovaleric acid "
            "(3-methylbutanoic acid) — a 5-carbon branched-chain fatty acid. IVA is volatile at body "
            "temperature and is excreted via sweat, urine, and breath. It is detected at concentrations "
            "as low as 0.1 µM (below the threshold for other diagnostic tests), making bedside odor "
            "recognition clinically significant. The odor is: "
            "• PRESENT during acute metabolic crisis (IVA very high) "
            "• MAY BE ABSENT in NBS-detected mild phenotype (low IVA) "
            "• Also found in urine, saliva, cerumen (earwax), and on the skin. "
            "IMPORTANT: sweat-foot odor ≠ sweat gland disorder — it is a metabolic odor from circulating "
            "isovaleric acid. Families often describe it as 'sweaty feet', 'blue cheese', or 'strong cheese'."
        ),
        "glycine_supplementation_rationale": (
            "Glycine supplementation is UNIQUE to IVD management among the organic acidemias. The "
            "mechanism: glycine N-acyltransferase (GLYAT, liver + kidney) catalyses: "
            "  Isovaleryl-CoA + Glycine → Isovalerylglycine (IVG) + CoASH "
            "By providing excess glycine substrate, the reaction is driven towards IVG formation: "
            "(1) IVG is non-toxic (unlike IVA) "
            "(2) IVG is water-soluble → renally excreted without reabsorption "
            "(3) IVG represents the DIRECT elimination route for the toxic isovaleryl group "
            "(4) Without supplementation: glycine is consumed by ongoing conjugation → glycine depletion "
            "    → conjugation capacity falls → IVA rises → crisis. "
            "Target: urine IVG > 100 mmol/mol Cr interictal (confirming effective conjugation); "
            "increase glycine dose to 400 mg/kg/day during crisis. "
            "IVG elevation on urine organic acids is thus BOTH a diagnostic marker AND a treatment "
            "efficacy marker — the only organic acidemia with this dual use."
        ),
        "vpa_absolute_contraindication": (
            "Valproate (VPA, valproic acid) is ABSOLUTELY CONTRAINDICATED in IVD deficiency. "
            "Three distinct mechanisms converge to make VPA lethal in IVD: "
            "(1) COMPETITIVE SUBSTRATE INHIBITION (PRIMARY): VPA is metabolised to valproyl-CoA (3-"
            "    propylpentanoyl-CoA), a structural analogue of isovaleryl-CoA. Valproyl-CoA directly "
            "    COMPETES with isovaleryl-CoA at the IVD active site → product inhibition of an already "
            "    deficient enzyme → WORSENS the primary enzyme block → isovaleryl-CoA accumulates further. "
            "(2) CARNITINE DEPLETION: VPA → valproyl-carnitine → depletes the free carnitine pool → "
            "    abolishes the C5 carnitine conjugation detox pathway → IVA rises unchecked. "
            "(3) HEPATOTOXICITY: VPA inhibits mitochondrial fatty acid β-oxidation + depletes carnitine "
            "    → mitochondrial failure in an already metabolically stressed liver (managing ongoing "
            "    isovaleryl-CoA flux). "
            "MANAGEMENT: REPLACE VPA with LEV (levetiracetam) — no hepatotoxicity, no leucine "
            "catabolism interference, no carnitine depletion, no IVD inhibition."
        ),
        "nbs_c5_differential": (
            "C5 (isovalerylcarnitine) is the newborn screening (NBS) marker for IVD detected by tandem "
            "mass spectrometry (MS/MS). However, C5 elevation is NOT specific to IVD. "
            "DIFFERENTIAL for elevated C5 on NBS: "
            "(1) IVD deficiency (Isovaleric acidemia): CONFIRMED by urine IVG + gene panel. "
            "(2) SBCAD deficiency (short/branched-chain acyl-CoA dehydrogenase; 2-methylbutyryl-CoA "
            "    dehydrogenase): C5 represents 2-methylbutyrylcarnitine — urine shows 2-methylbutyrylglycine "
            "    NOT IVG. Usually benign (most SBCAD-detected patients remain asymptomatic). "
            "(3) Pivaloyl-carnitine ingestion (pivalate-conjugated antibiotics): iatrogenic C5 elevation — "
            "    resolves after stopping medication; no organic acid abnormalities. "
            "RULE OF THUMB: "
            "• C5 + IVG in urine → IVD (isovaleric acidemia) "
            "• C5 + 2-methylbutyrylglycine → SBCAD (usually benign) "
            "• C5 alone, no glycine conjugate → pivalate exposure or SBCAD "
            "Gene panel (IVD + ACADSB genes) confirms."
        ),
        "treatment_rationale": (
            "IVD treatment strategy targets three goals: (1) reduce isovaleryl-CoA flux, "
            "(2) accelerate conjugation detoxification, (3) prevent catabolic crises. "
            "(1) Leucine restriction (Level A): Lys is the sole substrate for IVD. "
            "    Natural protein restriction to 0.8–1.5 g/kg/day + IVA-free amino acid supplement. "
            "    Target plasma leucine 100–200 µmol/L. Avoid over-restriction → growth failure. "
            "(2) Glycine supplementation (Level A — UNIQUE to IVD): 150–300 mg/kg/day drives "
            "    isovaleryl-CoA → IVG conjugation → renal excretion. PATHOGNOMONIC TREATMENT. "
            "(3) L-Carnitine (Level A): replenishes secondary deficiency; drives C5 detox; "
            "    protects muscle and cardiac function; 100–200 mg/kg/day oral. "
            "(4) Acute crisis: HIGH-DOSE GLUCOSE (10% dextrose, 8–12 mg/kg/min GIR) to suppress "
            "    catabolism → stop leucine release → isovaleryl-CoA flux falls; "
            "    IV glycine 400 mg/kg/day; IV carnitine 100 mg/kg. Written emergency plans for all. "
            "(5) AED: LEV first-line. VPA ABSOLUTELY CONTRAINDICATED (triple mechanism — see VPA note)."
        ),
    }
