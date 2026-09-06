#!/usr/bin/env python3
"""MCCC1 (3-Methylcrotonyl-CoA Carboxylase 1 Alpha Subunit) Deficiency — 3-MCC1 / 3-Methylcrotonylglycinuria Dashboard.

MCCC1 encodes the biotin-dependent alpha subunit of 3-methylcrotonyl-CoA carboxylase (MCC):
  3-Methylcrotonyl-CoA + HCO₃⁻ + ATP  →  [MCC]  →  3-Methylglutaconyl-CoA + ADP + Pᵢ
  (step 3 in leucine catabolism — after IVD; biotin-dependent carboxylation)

MCCC1 LOF → 3-Methylcrotonyl-CoA CANNOT be carboxylated → accumulates:
  → 3-Methylcrotonylglycine (3-MCG): ELEVATED in urine (most specific urinary biomarker)
  → 3-Hydroxyisovalerate (3-HIVA): ELEVATED in urine (secondary hydration product)
  → 3-Hydroxyisovalerylcarnitine (C5-OH): ELEVATED in plasma (NBS PRIMARY MARKER)
  → 3-Methylcrotonyl-CoA itself: not directly measured
  → Secondary free carnitine: LOW (C5-OH conjugation depletes carnitine)
  → Biotin levels: NORMAL (unlike BTD/HLCS — biotin metabolism intact)

OMIM Disease: #210200 (3-Methylcrotonyl-CoA carboxylase 1 deficiency / 3-MCC1 deficiency)
OMIM Gene:   *603696 (MCCC1)
Chromosome: 3q27.1
Inheritance: Autosomal Recessive (AR) — isolated MCCC1; also heterodimeric enzyme with MCCC2
Protein: 725 aa; mitochondrial matrix; biotin-dependent carboxylase alpha subunit (BCCA);
         forms heterodimer (αβ) with MCCC2 (OMIM *609010, 563 aa beta subunit)
Prevalence: ~1:36,000 (most common organic aciduria by NBS in many populations)

UNIQUE CLINICAL FEATURE — PREDOMINANTLY BENIGN:
  MCCC1 deficiency is UNIQUE among the organic acidemias because the MAJORITY of NBS-detected
  patients are ASYMPTOMATIC or have a BENIGN course:
  - Most NBS-detected infants remain healthy without dietary treatment
  - MATERNAL 3-MCC DEFICIENCY: the most common cause of C5-OH elevation detected via newborn
    screening is an UNAFFECTED MOTHER who is homozygous or compound heterozygous for MCCC1/2 variants;
    the infant's NBS reflects maternal metabolites transferred in utero/via breast milk
  - Penetrance is VARIABLE — most homozygous individuals are asymptomatic
  This makes MCCC1 one of the most important "expanded NBS" diseases to understand for
  counseling: a C5-OH elevation does NOT automatically mean a sick child

LEUCINE CATABOLISM POSITION:
  L-Leucine → BCAT (Step 1a) → α-ketoisocaproate (KIC)
     → BCKDH (Step 1b) → Isovaleryl-CoA
     → [IVD → Step 2] → 3-Methylcrotonyl-CoA
     → [MCC / MCCC1+MCCC2 → Step 3, BLOCKED] → 3-Methylglutaconyl-CoA
     → AUH (Step 4) → 3-Hydroxy-3-methylglutaryl-CoA (HMG-CoA)
     → HMGCL (Step 5) → Acetyl-CoA + Acetoacetate (ketogenesis)

BIOTIN CONTEXT:
  MCC requires biotin cofactor attached to the alpha subunit (MCCC1).
  IMPORTANT: In MCCC1 deficiency, biotin metabolism is INTACT (unlike BTD and HLCS where biotin
  recycling/attachment to ALL 4 carboxylases is defective). MCCC1 deficiency is an isolated
  carboxylase defect — only MCC is affected, NOT PC, PCC, or ACC.
"""

import random

SEED       = 259      # next after IVD (seed 253), continuing series
N_PATIENTS = 40

# Phenotypic classes
PHENOTYPE_CLASSES = [
    {"name": "Asymptomatic (NBS-detected / maternal MCCD)",
     "pct": 60,
     "note": "NBS C5-OH elevation only; no clinical symptoms; metabolic studies normal or mildly abnormal; "
             "frequently due to maternal 3-MCC deficiency (metabolites cross placenta/breast milk); "
             "majority remain healthy without any dietary intervention; repeat NBS on mother if infant NBS positive"},
    {"name": "Symptomatic (metabolic crisis)",
     "pct": 25,
     "note": "Episodic metabolic crises triggered by catabolic stress (febrile illness, fasting, surgery); "
             "3-MCG + C5-OH markedly elevated; metabolic acidosis, hypoglycaemia, vomiting, encephalopathy; "
             "responds to glucose + carnitine; minority develop chronic neurological features"},
    {"name": "Mild/late-onset",
     "pct": 15,
     "note": "Variable presentation in childhood or adulthood; mild IDD or learning difficulties; "
             "hypotonia; rare cardiomyopathy; often identified by cascade family testing; "
             "leucine restriction + carnitine control biomarkers but minimal clinical benefit proven"},
]

# Top pathogenic variants (MCCC1 alpha subunit)
VARIANTS = [
    {"variant": "p.Arg385Ser (c.1153C>T)",    "freq": 18, "domain": "Biotin-carboxylase (BC) domain — active site", "phenotype": "Classic symptomatic; pan-ethnic",
     "note": "Most common pathogenic MCCC1 variant; BC domain disrupts biotin carboxylation; full loss of MCC activity; "
             "3-MCG markedly elevated; symptomatic phenotype in biallelic state; NBS C5-OH very high (>2.0 µmol/L)"},
    {"variant": "p.Thr428Ile (c.1283C>T)",    "freq": 14, "domain": "Biotin-carboxylase (BC) domain", "phenotype": "Moderate; European",
     "note": "BC domain; partial residual activity; intermediate phenotype; moderate C5-OH elevation; "
             "mostly NBS-detected; variable clinical expression; homozygous → often asymptomatic"},
    {"variant": "p.Leu282Pro (c.845T>C)",     "freq": 11, "domain": "Carboxyl-transferase (CT) domain", "phenotype": "Symptomatic-moderate; pan-ethnic",
     "note": "CT domain disruption; complete loss; symptomatic in neonatal or infantile period; high 3-MCG; "
             "leucine restriction required; carnitine supplementation improves free carnitine"},
    {"variant": "p.Ala196Val (c.587C>T)",     "freq": 9,  "domain": "BC domain — biotin-binding loop", "phenotype": "Mild-moderate; often asymptomatic",
     "note": "Biotin-binding loop; reduced biotin attachment efficiency; partial activity; "
             "biotin trial may show modest biochemical improvement (not consistently); often NBS-detected; "
             "many remain asymptomatic even if biallelic"},
    {"variant": "c.IVS14+1G>A",              "freq": 8,  "domain": "Splice donor intron 14", "phenotype": "Null; neonatal symptomatic",
     "note": "Splice donor destruction → exon 14 skipping or intron retention → null allele → "
             "no MCC alpha subunit → complete enzyme absence; severe symptomatic neonatal phenotype; "
             "highest C5-OH values; marked 3-MCG excretion"},
    {"variant": "p.Gln404Arg (c.1211A>G)",   "freq": 7,  "domain": "Carboxyl-transferase domain", "phenotype": "Symptomatic-moderate",
     "note": "CT domain; disrupts substrate binding; loss of carboxylation; symptomatic with febrile triggers; "
             "3-MCG elevated 50-200 mmol/mol Cr; responds to carnitine supplementation"},
    {"variant": "p.Gly294Arg (c.880G>A)",    "freq": 6,  "domain": "BC-CT interdomain linker", "phenotype": "Variable (NBS to symptomatic)",
     "note": "Interdomain linker; disrupts BC-CT domain interface and quaternary structure with beta subunit; "
             "variable MCC activity; heterozygous + second null → symptomatic; two missense → often NBS-detected asymptomatic"},
    {"variant": "p.Tyr543Cys (c.1628A>G)",   "freq": 5,  "domain": "CT domain — substrate channel", "phenotype": "Moderate; variable",
     "note": "CT domain substrate channel; partial loss; moderate 3-MCG; variable phenotype; often detected via "
             "sibling cascade testing after index case diagnosed by NBS"},
]

# Biomarker panel
BIOMARKERS = {
    "c5oh_plasma":   {"label": "Plasma C5-OH (3-OH-Isovalerylcarnitine)",   "normal": "<0.3 µmol/L",       "status": "ELEVATED — PRIMARY NBS MARKER (classic >0.6; severe >2.0; maternal 0.3–0.8)",    "direction": "↑↑ NBS primary",    "color": "danger"},
    "3mcg_urine":    {"label": "Urine 3-Methylcrotonylglycine (3-MCG)",     "normal": "<5 mmol/mol Cr",     "status": "ELEVATED — MOST SPECIFIC URINARY BIOMARKER (symptomatic >50; severe >200)",    "direction": "↑↑↑ PATHOGNOMONIC", "color": "danger"},
    "3hiva_urine":   {"label": "Urine 3-Hydroxyisovalerate (3-HIVA)",       "normal": "<15 mmol/mol Cr",    "status": "ELEVATED — secondary hydration product (50–500 symptomatic; >100 classic)",       "direction": "↑↑ ELEVATED",        "color": "danger"},
    "carnitine":     {"label": "Free Carnitine (plasma)",                   "normal": "25–60 µmol/L",       "status": "LOW — secondary depletion from C5-OH conjugation",                              "direction": "↓ LOW",              "color": "warning"},
    "leucine":       {"label": "Plasma Leucine",                            "normal": "60–170 µmol/L",      "status": "NORMAL to mildly elevated (block at Step 3 — upstream leucine may accumulate in severe)", "direction": "→/↑ VARIABLE",    "color": "warning"},
    "biotin":        {"label": "Plasma Biotin",                             "normal": ">200 nmol/L",        "status": "NORMAL — KEY NEGATIVE vs BTD (biotinidase deficiency) and HLCS",              "direction": "→ NORMAL",           "color": "success"},
    "nh3":           {"label": "Plasma Ammonia",                            "normal": "<50 µmol/L",         "status": "NORMAL — KEY NEGATIVE vs UCDs (CPS1/OTC/ASS1/ASL)",                          "direction": "→ NORMAL",           "color": "success"},
    "mma":           {"label": "Methylmalonic Acid (MMA)",                  "normal": "<0.4 µmol/L",        "status": "NORMAL — KEY NEGATIVE vs MMUT/cblA/PCCA/PCCB",                              "direction": "→ NORMAL",           "color": "success"},
    "ivg_urine":     {"label": "Urine Isovalerylglycine (IVG)",             "normal": "<5 mmol/mol Cr",     "status": "ABSENT — KEY NEGATIVE vs IVD (Isovaleric Acidemia — IVD block is Step 2 BEFORE MCC)", "direction": "→ ABSENT",     "color": "success"},
    "c5_plasma":     {"label": "Plasma C5 (Isovalerylcarnitine)",           "normal": "<0.5 µmol/L",        "status": "NORMAL — KEY NEGATIVE vs IVD (C5 and C5-OH are different acylcarnitines)",    "direction": "→ NORMAL",           "color": "success"},
    "methylcit":     {"label": "Urine Methylcitrate",                       "normal": "<5 mmol/mol Cr",     "status": "ABSENT — KEY NEGATIVE vs Propionic Acidemia (PCCA/PCCB)",                    "direction": "→ ABSENT",           "color": "success"},
    "pc_pcc":        {"label": "PC / PCC / ACC activities",                 "normal": "Normal range",        "status": "NORMAL — isolated MCC defect only; ALL other biotin-dependent carboxylases INTACT (KEY NEGATIVE vs HLCS/BTD)", "direction": "→ NORMAL", "color": "success"},
}

# Treatments (evidence-graded)
TREATMENTS = [
    {
        "therapy": "L-Carnitine supplementation",
        "level": "A",
        "dose": "100–200 mg/kg/day oral; IV 100 mg/kg during acute crisis; titrate to normalise free carnitine (target > 25 µmol/L); titrate to reduce C5-OH",
        "rationale": "MCCC1 LOF → 3-methylcrotonyl-CoA + carnitine → C5-OH (3-hydroxyisovalerylcarnitine) → renally excreted. "
                     "This secondary conjugation depletes free carnitine → secondary carnitine deficiency. "
                     "Carnitine supplementation: (1) replenishes free carnitine pool for energy metabolism; "
                     "(2) enhances C5-OH excretion (detoxification); (3) protects cardiac and skeletal muscle. "
                     "Most consistently beneficial intervention across all phenotypes. "
                     "Monitor free carnitine and C5-OH monthly initially; stable disease quarterly.",
        "class": "Carnitine supplementation (primary — secondary deficiency + detox)",
    },
    {
        "therapy": "IV glucose (anti-catabolic) during crisis",
        "level": "A",
        "dose": "10% dextrose IV at 8–12 mg/kg/min (GIR); +/- insulin 0.05–0.1 U/kg/hr; continue until leucine normalising and clinical recovery; add IV carnitine 100 mg/kg",
        "rationale": "Febrile illness / fasting → catabolic state → increased protein catabolism → increased leucine flux → "
                     "more 3-methylcrotonyl-CoA accumulation → metabolic crisis. "
                     "HIGH-DOSE GLUCOSE: (1) anti-catabolic — suppresses endogenous protein breakdown + leucine release; "
                     "(2) provides energy substrate bypassing the MCC block (glucose-derived acetyl-CoA bypasses leucine pathway); "
                     "(3) insulin co-administration enhances anabolism. "
                     "Combine with IV carnitine. Written emergency letters for all symptomatic patients.",
        "class": "Emergency anti-catabolic (crisis management)",
    },
    {
        "therapy": "Leucine-restricted diet (symptomatic patients only)",
        "level": "B",
        "dose": "Natural protein 1.0–1.5 g/kg/day; leucine intake < 150–200 mg/kg/day; "
                "3-MCC-free amino acid formula for protein adequacy; plasma leucine target 100–200 µmol/L; "
                "NOT required in asymptomatic NBS-detected patients",
        "rationale": "Leucine is the sole substrate for the MCC pathway. Reducing leucine intake lowers 3-methylcrotonyl-CoA "
                     "flux → less 3-MCG and C5-OH accumulation. IMPORTANT CAVEAT: many MCCC1-deficient patients are "
                     "asymptomatic and do NOT benefit from dietary restriction — avoiding over-treatment is critical. "
                     "Dietary restriction is indicated only in patients with: documented symptomatic crises OR "
                     "persistently very high biomarkers (3-MCG > 200 mmol/mol Cr) AND a clinical phenotype consistent with disease. "
                     "Growth monitoring essential — leucine-free formula with adequate nutrition.",
        "class": "Dietary (secondary — symptomatic patients only; NOT for asymptomatic NBS-detected)",
    },
    {
        "therapy": "Biotin supplementation (trial)",
        "level": "C",
        "dose": "10–20 mg/day pharmacological biotin; trial for 6–12 weeks; "
                "assess biochemical response (C5-OH + 3-MCG reduction > 50%); "
                "discontinue if no biochemical response",
        "rationale": "MCC is biotin-dependent (biotin attached to MCCC1 via holocarboxylase synthetase). "
                     "IMPORTANT: MCCC1 deficiency is NOT biotin-responsive in most patients — biotin metabolism is intact; "
                     "the defect is in MCCC1 apoenzyme structure, not biotin availability. "
                     "A small subset of patients (estimated < 15%) with specific missense variants in the biotin-binding "
                     "domain may show partial biochemical improvement with pharmacological biotin doses. "
                     "Contrast with BTD/HLCS: those respond dramatically and consistently to biotin. "
                     "Trial is low-risk but should not be continued if no biochemical response.",
        "class": "Cofactor trial (biotin — usually non-responsive; trial warranted if biotin-domain variant)",
    },
    {
        "therapy": "LEV (levetiracetam)",
        "level": "B",
        "dose": "10–40 mg/kg/day; first-line AED for seizures; "
                "monitor leucine catabolism — LEV does not impair MCC pathway",
        "rationale": "Seizures occur in 20–30% of symptomatic MCCC1 patients (metabolic encephalopathy + "
                     "cortical toxicity from 3-HIVA accumulation). LEV: no hepatotoxicity, no carnitine depletion, "
                     "no leucine catabolism interference. SV2A mechanism safe in metabolic milieu. "
                     "Preferred over enzyme-inducing AEDs which increase metabolic demand and carnitine depletion.",
        "class": "AED — first-line",
    },
    {
        "therapy": "Maternal MCCD management (infant NBS scenario)",
        "level": "A — Diagnostic protocol",
        "dose": "Step 1: repeat NBS or confirmatory plasma acylcarnitines on infant; "
                "Step 2: test MATERNAL plasma C5-OH + urine 3-MCG; "
                "Step 3: if mother confirmed MCCD → infant is NOT affected (maternal metabolite transfer); "
                "Step 4: STOP dietary treatment in infant; counsel mother",
        "rationale": "The MOST IMPORTANT clinical action in MCCC1/MCCD NBS: ALWAYS rule out maternal 3-MCC deficiency "
                     "before diagnosing the infant. Maternal MCCC1 or MCCC2 deficiency → 3-MCG and C5-OH in maternal "
                     "plasma → transferred to fetus in utero (cord blood) and to neonate via breast milk → false NBS "
                     "positive in infant. The MOTHER is affected (often asymptomatically); the INFANT is a heterozygous "
                     "carrier at most. Maternal testing prevents: (1) unnecessary infant dietary restriction + formula; "
                     "(2) parental anxiety; (3) misdiagnosis. Estimate: 30–40% of NBS C5-OH cases in MCCD programs "
                     "are actually maternal MCCD. ALWAYS test the mother FIRST.",
        "class": "CRITICAL diagnostic protocol — rule out maternal MCCD before infant treatment",
    },
    {
        "therapy": "Valproate (VPA)",
        "level": "HIGH RISK — AVOID",
        "dose": "AVOID if possible; if essential, mandatory L-carnitine co-supplementation 100 mg/kg/day; "
                "monitor free carnitine weekly; replace with LEV as first-line AED",
        "rationale": "VPA in MCCC1: HIGH RISK (not absolute CI unlike IVD). Mechanism: "
                     "(1) Carnitine depletion: VPA → valproyl-carnitine → depletes free carnitine pool → "
                     "worsens secondary carnitine deficiency ALREADY present from C5-OH conjugation; "
                     "(2) Mitochondrial toxicity: VPA inhibits mitochondrial fatty acid β-oxidation — "
                     "additive with MCCC1 metabolic burden. "
                     "VPA does NOT directly inhibit MCC enzyme (unlike VPA→valproyl-CoA competing with IVD). "
                     "Risk is primarily carnitine depletion + metabolic decompensation. "
                     "Replace with LEV; if VPA must be used → mandatory high-dose carnitine + monitoring.",
        "class": "HIGH RISK (carnitine depletion) — not absolute CI unlike IVD",
    },
    {
        "therapy": "Fasting avoidance / illness protocol",
        "level": "A (symptomatic patients) / B (asymptomatic NBS-detected)",
        "dose": "Symptomatic: no fast > 4–6h (infant), > 8–10h (child/adult); "
                "glucopolymer/glucose drink at home during illness; "
                "written emergency plan; IV glucose if vomiting. "
                "Asymptomatic NBS-detected: standard fasting tolerance; no special protocol needed",
        "rationale": "Fasting + febrile illness → catabolic state → leucine surge → 3-methylcrotonyl-CoA surge → crisis. "
                     "IMPORTANT distinction from IVD: MOST NBS-detected MCCC1 patients are asymptomatic and do NOT "
                     "require fasting protocols — over-medicalisation harms quality of life. "
                     "Fasting protocols are reserved for patients with documented metabolic crises.",
        "class": "Dietary hazard prevention (symptomatic patients only)",
    },
]

# Seizure types in MCCC1
SEIZURE_TYPES = [
    {"type": "Febrile seizures (crisis-triggered)",                "pct": 18, "note": "Most common; fever → catabolic crisis → 3-HIVA accumulation → cortical irritability; metabolic correction resolves; LEV first-line"},
    {"type": "Focal / complex partial seizures",                   "pct": 12, "note": "Cortical toxicity from 3-HIVA; temporal-frontal predominance; LEV; seizures less prominent than in IVD/PA/MMA"},
    {"type": "GTCS (acute metabolic encephalopathy)",              "pct": 10, "note": "During severe metabolic crisis; hypoglycaemia-driven or 3-HIVA neurotoxicity; reverses with metabolic correction"},
    {"type": "Infantile spasms (rare, severe phenotype)",          "pct": 5,  "note": "Very rare; severe neonatal MCCC1 with marked 3-MCG; West syndrome-like; ACTH + metabolic management"},
    {"type": "Absence seizures",                                   "pct": 8,  "note": "Mild cortical hyperexcitability; usually mild and controllable; LEV or ethosuximide"},
    {"type": "Drug-resistant epilepsy (structural)",               "pct": 6,  "note": "Post-crisis cortical injury in severe uncontrolled symptomatic MCCC1; multi-AED; avoid VPA (carnitine depletion)"},
    {"type": "No seizures (asymptomatic majority)",                "pct": 60, "note": "MAJORITY of NBS-detected MCCC1 patients NEVER develop seizures — benign/asymptomatic disease course"},
]

# Systemic features
SYSTEMIC_FEATURES = [
    {"feature": "Asymptomatic (NBS-detected only)",                    "pct": 60, "note": "MAJORITY — C5-OH elevated on NBS; no clinical symptoms; normal development; most do not require treatment; biggest teaching point in 3-MCC1"},
    {"feature": "Acute metabolic crisis (catabolic trigger)",          "pct": 30, "note": "Symptomatic subset: vomiting, lethargy, metabolic acidosis, hypoglycaemia during febrile illness or fasting; responds to IV glucose + carnitine"},
    {"feature": "Hypotonia",                                           "pct": 20, "note": "Mild-moderate; proximal; related to carnitine deficiency + MCC block in muscle energy metabolism; improves with carnitine supplementation"},
    {"feature": "Intellectual disability (post-crisis)",               "pct": 15, "note": "Rare in NBS-detected + treated patients; seen in late-diagnosed or recurrent-crisis patients; mild-to-moderate when present"},
    {"feature": "Metabolic acidosis (high anion gap)",                 "pct": 25, "note": "During acute crisis only; 3-HIVA and other organic acids → anion gap 20–35 mEq/L; bicarbonate + IV glucose reverses"},
    {"feature": "Cardiomyopathy",                                      "pct": 8,  "note": "Rare; dilated cardiomyopathy in severe secondary carnitine deficiency; echocardiography + aggressive carnitine supplementation; improves with treatment"},
    {"feature": "Maternal MCCD (maternal 3-MCC deficiency)",           "pct": 35, "note": "PATHOGNOMONIC scenario: C5-OH elevation in infant NBS reflects maternal MCCC1/2 deficiency; mother is affected (often asymptomatically); infant is a carrier or unaffected; ALWAYS test the mother first"},
    {"feature": "Lactic acidosis (secondary)",                         "pct": 12, "note": "During severe crisis; secondary to energy failure from MCC block; not primary lactic acidosis; resolves with metabolic stabilisation"},
]


def _generate_cohort():
    random.seed(SEED)
    patients = []
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls["name"]] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append("Asymptomatic (NBS-detected / maternal MCCD)")
    phenotype_dist = phenotype_dist[:N_PATIENTS]
    random.shuffle(phenotype_dist)

    for i in range(N_PATIENTS):
        phenotype = phenotype_dist[i]

        if "Asymptomatic" in phenotype:
            age_onset_months  = round(random.uniform(0, 0.3), 2)    # NBS at birth
            c5oh_umol         = round(random.uniform(0.35, 1.2), 2) # mildly elevated
            mcg_urine         = random.randint(5, 60)                # mild 3-MCG
            hiva_urine        = random.randint(20, 120)
            free_carnitine    = random.randint(18, 40)
            crisis_count      = 0
            idd               = False
            cardiomyopathy    = False
            hypotonia         = random.random() < 0.10
        elif "Symptomatic" in phenotype:
            age_onset_months  = round(random.uniform(1, 36), 1)
            c5oh_umol         = round(random.uniform(1.0, 5.5), 1)
            mcg_urine         = random.randint(60, 450)
            hiva_urine        = random.randint(100, 600)
            free_carnitine    = random.randint(4, 18)
            crisis_count      = random.randint(1, 6)
            idd               = random.random() < 0.30
            cardiomyopathy    = random.random() < 0.10
            hypotonia         = random.random() < 0.35
        else:  # Mild / late-onset
            age_onset_months  = round(random.uniform(12, 96), 1)
            c5oh_umol         = round(random.uniform(0.5, 2.5), 2)
            mcg_urine         = random.randint(20, 150)
            hiva_urine        = random.randint(40, 250)
            free_carnitine    = random.randint(12, 30)
            crisis_count      = random.randint(0, 2)
            idd               = random.random() < 0.20
            cardiomyopathy    = random.random() < 0.05
            hypotonia         = random.random() < 0.25

        seizures     = crisis_count > 0 and random.random() < 0.35
        dre          = seizures and random.random() < 0.12
        maternal_mccd= "Asymptomatic" in phenotype and random.random() < 0.55
        v            = random.choice(VARIANTS)

        patients.append({
            "id":                     f"MCC1-{i+1:03d}",
            "phenotype":              phenotype,
            "age_onset_months":       age_onset_months,
            "c5oh_umol_l":            c5oh_umol,
            "mcg_urine_mmol_mol_cr":  mcg_urine,
            "hiva_urine_mmol_mol_cr": hiva_urine,
            "free_carnitine_umol_l":  free_carnitine,
            "crisis_count":           crisis_count,
            "seizures":               seizures,
            "dre":                    dre,
            "idd":                    idd,
            "hypotonia":              hypotonia,
            "cardiomyopathy":         cardiomyopathy,
            "maternal_mccd":          maternal_mccd,
            "variant":                v["variant"],
        })
    return patients


COHORT = _generate_cohort()


def get_overview():
    n               = len(COHORT)
    n_asymp         = sum(1 for p in COHORT if "Asymptomatic" in p["phenotype"])
    n_sympt         = sum(1 for p in COHORT if "Symptomatic" in p["phenotype"])
    n_late          = sum(1 for p in COHORT if "late" in p["phenotype"].lower())
    n_seizures      = sum(1 for p in COHORT if p["seizures"])
    n_dre           = sum(1 for p in COHORT if p["dre"])
    n_idd           = sum(1 for p in COHORT if p["idd"])
    n_maternal      = sum(1 for p in COHORT if p["maternal_mccd"])
    n_cardi         = sum(1 for p in COHORT if p["cardiomyopathy"])
    avg_c5oh        = round(sum(p["c5oh_umol_l"] for p in COHORT) / n, 2)
    avg_mcg         = round(sum(p["mcg_urine_mmol_mol_cr"] for p in COHORT) / n)
    avg_carnitine   = round(sum(p["free_carnitine_umol_l"] for p in COHORT) / n, 1)

    return {
        "disease": "3-Methylcrotonyl-CoA Carboxylase 1 Deficiency (3-MCC1 / MCCC1 / 3-Methylcrotonylglycinuria)",
        "omim_gene": "603696",
        "omim_disease": "210200",
        "gene": "MCCC1",
        "alias": "3-MCC1 / 3-Methylcrotonyl-CoA Carboxylase Alpha Subunit / 3-Methylcrotonylglycinuria Type I / BCCA",
        "chromosome": "3q27.1",
        "inheritance": "Autosomal Recessive (AR)",
        "protein": "725 aa; mitochondrial matrix; biotin-dependent carboxylase alpha subunit (BCCA); "
                   "forms αβ heterodimer with MCCC2 (OMIM *609010, beta subunit, 4q34.2)",
        "prevalence": "~1:36,000 (most common organic aciduria detected by expanded NBS in many populations); "
                      "true symptomatic prevalence much lower — majority of NBS-detected cases are asymptomatic or maternal",
        "mechanism": (
            "3-Methylcrotonyl-CoA (from leucine catabolism, step 3) CANNOT be carboxylated → "
            "3-MCG (most specific urinary biomarker) + 3-HIVA accumulate; "
            "C5-OH (3-hydroxyisovalerylcarnitine) elevated on NBS; "
            "biotin levels NORMAL (unlike BTD/HLCS — only MCC isolated); "
            "MAJORITY of NBS-detected cases are asymptomatic or reflect maternal MCCD"
        ),
        "n_patients": n,
        "kpi": {
            "n_patients":    {"value": n,                                 "label": "Cohort size",                                    "color": "#1a237e"},
            "c5oh_avg":      {"value": f"{avg_c5oh} µmol/L",             "label": "Mean C5-OH Isovalerylcarnitine (NBS marker)",    "color": "#b71c1c"},
            "mcg_avg":       {"value": f"{avg_mcg} mmol/mol Cr",          "label": "Mean Urine 3-MCG (most specific biomarker)",     "color": "#880e4f"},
            "carnitine_avg": {"value": f"{avg_carnitine} µmol/L",         "label": "Mean Free Carnitine (LOW)",                      "color": "#6a1b9a"},
            "asymp_pct":     {"value": f"{round(n_asymp/n*100)}%",        "label": "Asymptomatic (majority — unique feature)",       "color": "#2e7d32"},
            "maternal_pct":  {"value": f"{round(n_maternal/n*100)}%",     "label": "Maternal MCCD (test mother first!)",             "color": "#0d47a1"},
            "seizures_pct":  {"value": f"{round(n_seizures/n*100)}%",     "label": "Seizures (%)",                                   "color": "#c62828"},
            "dre_pct":       {"value": f"{round(n_dre/n*100)}%",          "label": "Drug-Resistant Epilepsy (%)",                    "color": "#bf360c"},
            "idd_pct":       {"value": f"{round(n_idd/n*100)}%",          "label": "Intellectual Disability (%)",                    "color": "#37474f"},
            "cardi_pct":     {"value": f"{round(n_cardi/n*100)}%",        "label": "Cardiomyopathy (%)",                             "color": "#e65100"},
        },
        "phenotype_dist": [
            {"class": "Asymptomatic (NBS-detected / maternal MCCD)", "n": n_asymp, "pct": round(n_asymp/n*100)},
            {"class": "Symptomatic (metabolic crisis)",               "n": n_sympt, "pct": round(n_sympt/n*100)},
            {"class": "Mild / late-onset",                            "n": n_late,  "pct": round(n_late/n*100)},
        ],
        "hallmark_biomarker": (
            "C5-OH (3-hydroxyisovalerylcarnitine) ELEVATED on NBS (> 0.3 µmol/L; PRIMARY SCREENING MARKER); "
            "Urine 3-MCG (3-methylcrotonylglycine) MOST SPECIFIC urinary biomarker (> 50 mmol/mol Cr symptomatic); "
            "Urine 3-HIVA ELEVATED; Biotin NORMAL (KEY NEGATIVE vs BTD/HLCS); "
            "IVG ABSENT (KEY NEGATIVE vs IVD); C5 NORMAL (KEY NEGATIVE vs IVD); "
            "MMA ABSENT (KEY NEGATIVE vs PA/MMA)"
        ),
        "hallmark_clinical": (
            "PREDOMINANTLY BENIGN / ASYMPTOMATIC — most NBS-detected MCCC1 patients are healthy; "
            "maternal MCCD is the single most common cause of C5-OH elevation in NBS programs; "
            "ALWAYS test the mother before treating the infant"
        ),
        "maternal_mccd_note": (
            "MATERNAL 3-MCC DEFICIENCY: The MOST IMPORTANT clinical action in MCCC1 NBS is to TEST THE MOTHER. "
            "An affected mother (homozygous or compound heterozygous MCCC1/MCCC2) transfers 3-MCG and C5-OH "
            "to the infant in utero and via breast milk → the infant's NBS is positive, but the INFANT is a "
            "heterozygous CARRIER (not affected). "
            "Testing: maternal plasma C5-OH + urine 3-MCG. "
            "If maternal MCCD confirmed → stop infant treatment + counsel mother."
        ),
        "benign_majority_note": (
            "UNIQUE CLINICAL FEATURE: MCCC1 deficiency is one of very few expanded NBS conditions where the "
            "MAJORITY of detected patients are asymptomatic and DO NOT require dietary treatment. "
            "Avoid over-medicalisation: unnecessary leucine restriction can cause growth failure and nutritional "
            "deficiency in a child who would otherwise have been healthy. "
            "Carnitine supplementation is the safest and most consistently beneficial intervention."
        ),
        "nbs_c5oh_differential_note": (
            "C5-OH on NBS is NOT specific to MCCC1: also elevated in MCCC2, BTD (biotinidase), "
            "HLCS (holocarboxylase synthetase), and β-methylcrotonylglycinuria. "
            "Distinguish by: (1) urine 3-MCG (specific for MCCC1/MCCC2); "
            "(2) serum biotinidase activity (BTD deficiency); "
            "(3) all 4 carboxylase activities (HLCS — all low); "
            "(4) maternal testing; (5) gene panel."
        ),
        "vpa_note": (
            "VALPROATE HIGH RISK (not absolute CI unlike IVD): VPA → carnitine depletion → worsens "
            "secondary carnitine deficiency ALREADY present from C5-OH conjugation. "
            "VPA does NOT directly inhibit MCC enzyme (unlike valproyl-CoA competing at IVD active site). "
            "Replace with LEV; if VPA essential → mandatory high-dose carnitine supplementation."
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
            "gene":         "MCCC1 (3-Methylcrotonyl-CoA Carboxylase Alpha Subunit, BCCA)",
            "function":     "Biotin-dependent mitochondrial enzyme; alpha subunit (725 aa) of MCC heterodimer (αβ); "
                            "catalyses step 3 of leucine catabolism: carboxylation of 3-methylcrotonyl-CoA",
            "reaction":     "3-Methylcrotonyl-CoA + HCO₃⁻ + ATP → 3-Methylglutaconyl-CoA + ADP + Pᵢ (step 3 of leucine catabolism)",
            "block":        "MCCC1 LOF → 3-methylcrotonyl-CoA CANNOT be carboxylated → accumulates → "
                            "3-MCG (glycine conjugate, urinary biomarker) + 3-HIVA (hydration product) + C5-OH (carnitine conjugate, NBS marker)",
            "biotin_link":  "Biotin is covalently attached to MCCC1 Lys738 by holocarboxylase synthetase (HLCS). "
                            "Isolated MCCC1 deficiency: biotin attachment is intact; only MCC activity is lost. "
                            "Contrast: HLCS deficiency → all 4 carboxylases (PC, PCC, MCC, ACC) defective; "
                            "BTD deficiency → biotin recycling failed → all 4 carboxylases progressively deficient.",
            "leucine_path": "L-Leucine → BCAT → KIC → BCKDH → Isovaleryl-CoA → [IVD Step 2] → 3-Methylcrotonyl-CoA → "
                            "[MCCC1+MCCC2 BLOCKED Step 3] → accumulates → 3-MCG + 3-HIVA + C5-OH",
            "downstream":   "Normal: 3-Methylcrotonyl-CoA → [MCC] → 3-Methylglutaconyl-CoA → [AUH] → HMG-CoA → [HMGCL] → "
                            "Acetyl-CoA + Acetoacetate → energy (ketogenesis). Block prevents ketogenic end-product production from leucine.",
        },
        "differential_diagnosis": {
            "vs_mccc2": {
                "key_diff": "MCCC2 (beta subunit deficiency, OMIM *609010, 4q34.2): biochemically IDENTICAL to MCCC1 — "
                            "same C5-OH + 3-MCG elevation; gene panel distinguishes MCCC1 vs MCCC2",
                "mccc1":    "MCCC1: alpha subunit (725aa); 3q27.1; same phenotype + biomarker profile as MCCC2",
                "mccc2":    "MCCC2: beta subunit (563aa); 4q34.2; same phenotype + biomarker profile as MCCC1; "
                            "distinguish by gene panel only",
            },
            "vs_btd": {
                "key_diff": "Biotinidase deficiency (BTD): C5-OH elevated; HOWEVER biotin VERY LOW (<100 nmol/L); "
                            "ALL 4 carboxylases low (PC, PCC, MCC, ACC); serum biotinidase activity ABSENT; responds DRAMATICALLY to biotin",
                "mccc1":    "MCCC1: biotin NORMAL; only MCC deficient (PC, PCC, ACC intact); biotinidase normal; biotin trial usually not responsive",
                "btd":      "BTD: biotin LOW; all 4 carboxylases low; biotinidase activity absent; MUST treat with biotin 5–20 mg/day — fully responsive",
            },
            "vs_hlcs": {
                "key_diff": "Holocarboxylase synthetase (HLCS) deficiency: C5-OH + 3-MCG elevated (like MCCC1); "
                            "BUT also MMA elevated (PCC deficient), lactate elevated (PC deficient), dermatitis + alopecia (biotin-like features); "
                            "ALL 4 carboxylases low; biotin responsive",
                "mccc1":    "MCCC1: MMA NORMAL (PCC intact); PC intact; no alopecia/dermatitis; only 3-MCG + C5-OH",
                "hlcs":     "HLCS: MMA elevated + lactic acidosis + alopecia + dermatitis; ALL 4 carboxylases deficient; biotin dramatically responsive",
            },
            "vs_ivd": {
                "key_diff": "IVD (Isovaleric Acidemia): C5 elevated (NOT C5-OH); IVG elevated (NOT 3-MCG); "
                            "'sweaty feet' odor PATHOGNOMONIC; IVD is Step 2, MCC is Step 3 of leucine catabolism",
                "mccc1":    "MCCC1: C5-OH elevated (not C5); 3-MCG elevated (not IVG); no odor; biotin normal",
                "ivd":      "IVD: C5 elevated; IVG PATHOGNOMONIC; 'sweaty feet' odor; C5-OH NORMAL; glycine supplement is cornerstone",
            },
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "MCCC1 (3-Methylcrotonyl-CoA Carboxylase Alpha Subunit, OMIM *603696) encodes a 725-amino-acid "
            "mitochondrial matrix enzyme. MCCC1 is the alpha (biotin-containing) subunit of "
            "3-methylcrotonyl-CoA carboxylase (MCC), a biotin-dependent enzyme that forms an α₆β₆ "
            "dodecamer with MCCC2 (beta subunit, OMIM *609010). MCC catalyses step 3 of leucine catabolism: "
            "  3-Methylcrotonyl-CoA + HCO₃⁻ + ATP → 3-Methylglutaconyl-CoA + ADP + Pᵢ "
            "The reaction requires biotin attached to Lys738 of MCCC1 via holocarboxylase synthetase (HLCS). "
            "Chromosome 3q27.1; autosomal recessive; > 100 pathogenic variants known in MCCC1."
        ),
        "pathomechanism": (
            "MCCC1 LOF → 3-methylcrotonyl-CoA (generated by IVD in leucine catabolism) CANNOT be carboxylated. "
            "3-Methylcrotonyl-CoA accumulates → two conjugation/hydration pathways: "
            "(1) Glycine conjugation: 3-methylcrotonyl-CoA + glycine → 3-methylcrotonylglycine (3-MCG) → "
            "    water-soluble, urinary excretion (MOST SPECIFIC BIOMARKER). "
            "(2) Hydration: 3-methylcrotonyl-CoA → 3-methylcrotonyl-CoA hydratase → 3-hydroxyisovaleryl-CoA "
            "    → 3-HIVA (secondary urinary biomarker) + C5-OH (carnitine conjugate → NBS marker). "
            "DOWNSTREAM BLOCK: leucine catabolism cannot proceed beyond step 3 → "
            "3-methylglutaconyl-CoA, HMG-CoA, and ketone bodies CANNOT be generated from leucine. "
            "UNIQUE: Most MCCC1-deficient individuals are asymptomatic or mildly affected — "
            "likely because the 3-MCG conjugation pathway has very high capacity, preventing toxic accumulation "
            "in the majority."
        ),
        "benign_predominantly_asymptomatic": (
            "MCCC1 deficiency is one of the most important 'expanded NBS' conditions to understand from a "
            "counseling perspective because the MAJORITY of detected patients are ASYMPTOMATIC. "
            "Key data points: "
            "(1) In large NBS programs, 50–70% of biallelic MCCC1 compound heterozygotes / homozygotes remain "
            "    clinically unaffected without any dietary restriction throughout life. "
            "(2) The benign phenotype may relate to: (a) high capacity of the 3-MCG conjugation route (preventing "
            "    toxic accumulation); (b) alternative minor leucine catabolism routes; (c) some residual activity. "
            "(3) OVER-TREATMENT is the primary harm: unnecessary leucine restriction causes growth failure, "
            "    nutritional deficiency, and parental burden in children who would remain healthy. "
            "(4) CARNITINE supplementation is the safest universal treatment (corrects secondary deficiency "
            "    without dietary risk). "
            "PRACTICAL GUIDANCE: Asymptomatic NBS-detected MCCC1 → confirm diagnosis (urine 3-MCG, gene panel, "
            "maternal testing) → start carnitine → observe + metabolic monitoring → no leucine restriction unless "
            "symptomatic crises documented."
        ),
        "maternal_mccd_clinical_protocol": (
            "MATERNAL 3-MCC DEFICIENCY is the SINGLE MOST IMPORTANT differential in a C5-OH-positive NBS result. "
            "Mechanism: "
            "(1) Mother is homozygous or compound heterozygous for MCCC1 or MCCC2 variants → "
            "    maternal 3-MCG and C5-OH in maternal plasma. "
            "(2) During pregnancy: maternal 3-MCG and 3-HIVA cross the placenta → fetal blood accumulates these "
            "    metabolites → neonate's blood spot (NBS filter card) picks up maternal metabolites. "
            "(3) After birth: breast milk contains maternal 3-MCG and C5-OH → nursing infant's acylcarnitine profile "
            "    reflects maternal disease. "
            "PROTOCOL when infant NBS C5-OH elevated: "
            "STEP 1 — Test mother IMMEDIATELY: plasma acylcarnitines + urine organic acids (3-MCG). "
            "STEP 2 — If maternal C5-OH elevated + 3-MCG elevated → maternal MCCD confirmed. "
            "STEP 3 — Infant is at most a heterozygous carrier → STOP infant treatment + dietary restriction. "
            "STEP 4 — Counsel mother: she has isolated MCCC1/MCCC2 deficiency, usually asymptomatic, "
            "          may require carnitine supplementation. "
            "STEP 5 — Confirm infant genotype by gene panel if uncertain. "
            "Historical note: In several national NBS programs, 30–50% of initial MCCC1/MCCC2 NBS positives "
            "were MATERNAL cases — missed because the mother was not tested first."
        ),
        "biotin_dependency_vs_btd_hlcs": (
            "CRITICAL DISTINCTION — MCCC1 is NOT biotin-responsive in most patients. "
            "Reason: the defect is in the MCCC1 apoenzyme structure (alpha subunit), not in biotin availability "
            "or attachment. Biotin metabolism (holocarboxylase synthetase function + biotinidase recycling) is INTACT. "
            "Compare to: "
            "(1) BTD (Biotinidase Deficiency): biotin RECYCLING failed → biotin levels fall → all 4 carboxylases "
            "    (MCC, PC, PCC, ACC) progressively deficient → responds DRAMATICALLY to 5–20 mg/day biotin; "
            "    biotin levels very low (< 100 nmol/L) — opposite of MCCC1 (biotin normal). "
            "(2) HLCS (Holocarboxylase Synthetase Deficiency): biotin ATTACHMENT to all 4 carboxylases failed → "
            "    all 4 carboxylases (MCC, PC, PCC, ACC) deficient simultaneously → responds to high-dose biotin "
            "    (10–40 mg/day pharmacological doses); features MMA (PCC) + lactic acidosis (PC) + 3-MCG (MCC). "
            "MCCC1: biotin NORMAL, only MCC deficient, biotinidase normal, HLCS normal → no benefit from biotin "
            "supplementation in most cases. A small subset with biotin-binding domain variants may show modest "
            "biochemical improvement — trial is safe and low-risk but rarely achieves complete normalisation."
        ),
        "c5oh_nbs_differential_guide": (
            "C5-OH (3-hydroxyisovalerylcarnitine) elevated on NBS blood spot — differential diagnosis: "
            "(1) MCCC1 deficiency (3-MCC1): urine 3-MCG elevated; biotin normal; PC/PCC/ACC intact; gene: MCCC1; "
            "    most cases asymptomatic; maternal MCCD must be excluded. "
            "(2) MCCC2 deficiency (3-MCC2): biochemically identical to MCCC1; gene: MCCC2; same clinical spectrum. "
            "(3) BTD (Biotinidase Deficiency): biotin VERY LOW; biotinidase activity absent; ALL 4 carboxylases "
            "    low; alopecia + dermatitis + sensorineural deafness + optic atrophy; dramatically biotin-responsive. "
            "(4) HLCS (Holocarboxylase Synthetase Deficiency): ALL 4 carboxylases low; MMA elevated (PCC deficit); "
            "    lactic acidosis (PC deficit); biotin-responsive; neonatal acute + skin features. "
            "(5) β-Methylcrotonylglycinuria (isolated — unknown gene): rare; 3-MCG + C5-OH elevated; "
            "    MCC activity normal; mechanism unclear; usually benign. "
            "(6) Maternal MCCD: most common cause of C5-OH elevation in many NBS programs; test mother first. "
            "DIAGNOSTIC ALGORITHM: "
            "C5-OH↑ → urine 3-MCG (if high → MCC deficiency; if absent → BTD/HLCS first) → "
            "serum biotinidase activity (if absent → BTD) → carboxylase activities (if all low → HLCS) → "
            "maternal testing → gene panel (MCCC1 vs MCCC2)."
        ),
        "treatment_rationale": (
            "MCCC1 treatment strategy reflects the predominantly benign phenotype: "
            "(1) L-CARNITINE (Level A — ALL patients): corrects secondary carnitine deficiency from C5-OH "
            "    conjugation. Safe, universally beneficial, avoids the risks of dietary restriction. "
            "    100–200 mg/kg/day oral; monitor free carnitine (target > 25 µmol/L). "
            "(2) LEUCINE RESTRICTION (Level B — symptomatic patients only): NOT recommended for asymptomatic "
            "    NBS-detected patients. Reserve for patients with documented metabolic crises + markedly high "
            "    biomarkers (3-MCG > 200 mmol/mol Cr). Avoid over-restriction → growth failure. "
            "(3) BIOTIN TRIAL (Level C): low-risk but rarely effective. Trial 10–20 mg/day for 8–12 weeks; "
            "    stop if no biochemical response. "
            "(4) ANTI-CATABOLIC CRISIS MANAGEMENT (Level A): IV glucose (8–12 mg/kg/min) + IV carnitine "
            "    during acute decompensation. Written emergency plan for symptomatic patients. "
            "(5) MATERNAL MCCD PROTOCOL (Level A — diagnostic): test mother before treating infant. "
            "    Stop infant treatment if maternal MCCD confirmed. "
            "AED: LEV first-line. VPA HIGH RISK (carnitine depletion) — avoid; if used → mandatory carnitine."
        ),
    }
