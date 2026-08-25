#!/usr/bin/env python3
"""LCHAD / MTP (Long-Chain 3-Hydroxyacyl-CoA Dehydrogenase / Mitochondrial Trifunctional Protein) Dashboard.

HADHA gene encodes MTP ALPHA SUBUNIT (Mitochondrial Trifunctional Protein):
  - 763 aa precursor (83 kDa mature); mitochondrial inner membrane
  - Contains TWO catalytic activities:
      (1) 2-enoyl-CoA hydratase (MTP Step 2): trans-2-enoyl-acyl-CoA + H2O → L-3-hydroxyacyl-CoA
      (2) L-3-hydroxyacyl-CoA dehydrogenase [LCHAD] (MTP Step 3): L-3-OH-acyl-CoA + NAD+ → 3-ketoacyl-CoA + NADH

HADHB gene encodes MTP BETA SUBUNIT:
  - 474 aa; mitochondrial inner membrane
  - Contains ONE catalytic activity:
      (3) Long-chain 3-ketoacyl-CoA thiolase (MTP Step 4): 3-ketoacyl-CoA + CoA → acyl-CoA(n-2) + acetyl-CoA

MTP COMPLEX = α2β2 HETEROTETRAMER anchored to inner mitochondrial membrane
  → Catalyses Steps 2, 3, and 4 of beta-oxidation spiral for LONG-CHAIN (C12–C18) fatty acids
  → Steps 2-3-4 for C12-C18 substrates; shorter chains handed off to soluble matrix enzymes

LCHAD DEFICIENCY (isolated — p.Glu474Gln HADHA): Step 3 blocked → 3-hydroxy-acyl-CoAs ACCUMULATE
  → C16-OH (3-hydroxypalmitoylcarnitine) is PRIMARY NBS MARKER >0.08 µmol/L
  → Retinal pigmentary degeneration + peripheral neuropathy = UNIQUE among all FAO disorders
  → Maternal AFLP (Acute Fatty Liver of Pregnancy) or HELLP when carrying LCHAD-deficient fetus

TFP DEFICIENCY (complete — HADHB LOF or non-p.Glu474Gln HADHA): Steps 2+3+4 all blocked
  → More severe neonatal phenotype; cardiomyopathy + liver failure; same 3-OH-acylcarnitine profile

KEY FACTS (HIGHEST YIELD):
  1.  C16-OH (3-hydroxypalmitoylcarnitine) — PRIMARY NBS MARKER; >0.08 µmol/L tandem MS/MS
  2.  C16-OH/C16 ratio — >0.07 highly specific for LCHAD/TFP; discriminates from false positives
  3.  C14-OH, C18:1-OH also elevated — characteristic 3-OH-acylcarnitine profile
  4.  C8 NORMAL — KEY NEGATIVE (same as VLCAD; C8 is for MCAD)
  5.  C14:1 NORMAL — KEY NEGATIVE vs VLCAD (VLCAD elevates C14:1; LCHAD does NOT)
  6.  RETINAL PIGMENTARY DEGENERATION — UNIQUE among ALL FAO disorders; progressive; irreversible
  7.  PERIPHERAL NEUROPATHY — axonal; UNIQUE among FAO disorders; progressive; EMG/NCS evidence
  8.  MATERNAL AFLP / HELLP — mother of LCHAD-deficient fetus develops AFLP in 3rd trimester
  9.  HYPOketotic HYPOGLYCAEMIA — fasting/illness trigger; same mechanism as VLCAD/MCAD
 10.  CARDIOMYOPATHY — dilated > hypertrophic; particularly in TFP deficiency
 11.  RHABDOMYOLYSIS — exercise/fasting triggered; CK 10,000–100,000 U/L; myoglobinuria
 12.  FASTING: ABSOLUTE CI — same as VLCAD/MCAD
 13.  KD: ABSOLUTE CI — floods long-chain fat onto blocked MTP complex
 14.  MCT oil THERAPEUTIC — bypasses LCHAD block (C8–C10 processed by SCAD/MCAD, not LCHAD)
 15.  DHA SUPPLEMENTATION — Level B; DHA (C22:6) has shortened chain after LCHAD; replenishes retinal/neuronal DHA
 16.  VPA: HIGH RISK — inhibits FAO globally; worsens LCHAD crisis
 17.  p.Glu474Gln (c.1528G>A) HADHA — FOUNDER; 50–90% Northern European; isolated LCHAD (Step 3 only)
 18.  ~1:75,000–1:100,000 combined LCHAD+TFP

OMIM Disease (LCHAD isolated): #609016 (HADHA p.Glu474Gln)
OMIM Disease (TFP):             #609015 (complete, HADHB or other HADHA)
OMIM Gene HADHA:                *600890
OMIM Gene HADHB:                *143450
Chromosome:                     2p23.3 (both HADHA and HADHB adjacent)
Inheritance:                    Autosomal Recessive (AR), biallelic LOF
Protein:                        MTP = HADHA(α2)HADHB(β2) heterotetramer; inner mitochondrial membrane
Prevalence:                     ~1:75,000–100,000 (LCHAD+TFP combined); LCHAD alone ~1:100,000–250,000

FATTY ACID BETA-OXIDATION — WHERE LCHAD/MTP FITS:
  Very long-chain acyl-CoA (C14–C20) → [VLCAD + FAD] → trans-2-enoyl-acyl-CoA (C14–C20)    ← VLCAD block
  Continuing (for C12–C18 after VLCAD):
    trans-2-enoyl-acyl-CoA + H2O → [MTP/HADHA hydratase = Step 2] → L-3-OH-acyl-CoA
    L-3-OH-acyl-CoA + NAD+ → [MTP/HADHA LCHAD = Step 3  BLOCKED ⚠] → 3-keto-acyl-CoA + NADH
    3-keto-acyl-CoA + CoA  → [MTP/HADHB thiolase = Step 4] → acyl-CoA(n-2) + acetyl-CoA
  LCHAD block → L-3-OH-acylcarnitines ACCUMULATE → C16-OH, C14-OH, C18:1-OH elevated in blood
  LCHAD block → NADH NOT generated from step 3 → energy deficit
  Shorter chains (C4–C10) handled by soluble enzymes SCAD/MCAD — NOT by MTP

LCHAD vs VLCAD vs MCAD vs SCAD:
  LCHAD (HADHA/MTP, C12–C18): C16-OH primary NBS; retinopathy + neuropathy + maternal AFLP; MCT helpful; 2p23.3
  VLCAD (ACADVL, C14–C20):    C14:1 primary NBS; cardiomyopathy + rhabdomyolysis; MCT helpful; 17p13.1
  MCAD  (ACADM, C6–C12):      C8 primary NBS; Reye-like; HG/SG/PPG pathognomonic; fasting CI; 1p31.1
  SCAD  (ACADS, C4–C6):       C4 (butyrylcarnitine) NBS; mostly benign/asymptomatic; 12q24.31

MATERNAL AFLP / HELLP — UNIQUE LCHAD FEATURE (KEY EXAM CONCEPT):
  Mechanism: LCHAD-deficient fetus cannot oxidize long-chain 3-OH-fatty acids (C16-OH, C18:1-OH)
             → these 3-OH-fatty acids cross placenta and accumulate in maternal liver
             → maternal hepatocellular damage → AFLP (Acute Fatty Liver of Pregnancy)
             → also triggers HELLP (Hemolysis, Elevated Liver enzymes, Low Platelets)
  Timing:    3rd trimester (usually 30–38 weeks gestation)
  Maternal risk: jaundice, coagulopathy, hepatic failure, coma, maternal death if untreated
  Clinical pearl: a mother with unexplained AFLP/HELLP → investigate newborn for LCHAD deficiency
  EXCLUSIVE to LCHAD/TFP among FAO disorders (not VLCAD, not MCAD, not SCAD)

RETINAL PIGMENTARY DEGENERATION:
  Mechanism: DHA (C22:6, docosahexaenoic acid) synthesis requires LCHAD activity at C18:1-OH stage
             → LCHAD LOF → DHA cannot be elongated → retinal photoreceptor deprivation
             → also: 3-OH-acylcarnitine accumulation is directly toxic to retinal pigment epithelium
  Onset:     Usually first 2–5 years of life in classic LCHAD; progressive
  Features:  Night blindness first; peripheral field loss; macular atrophy in late stage
  Test:      ERG (electroretinogram) + fundoscopy; annual ophthalmology referral
  Prevention: DHA supplementation (Level B) may slow progression but does not halt it
  UNIQUE:    No other FAO disorder causes retinal degeneration — key LCHAD identifier

PERIPHERAL NEUROPATHY:
  Mechanism: 3-OH-acylcarnitines may be directly toxic to Schwann cells; DHA depletion in myelin
  Type:      Axonal; length-dependent; sensorimotor
  Test:      EMG/NCS; sural nerve biopsy shows axonal loss without demyelination
  Onset:     Usually childhood or adolescence; progressive with age and crises
  UNIQUE:    No other FAO disorder causes progressive peripheral neuropathy

BIOMARKER PATTERN — LCHAD DEFICIENCY:
  C16-OH (3-hydroxypalmitoylcarnitine)   >0.08 µmol/L  — PRIMARY NBS MARKER ↑↑↑
  C16-OH/C16 ratio                       >0.07          — highly specific discriminator ↑↑
  C14-OH (3-hydroxymyristoylcarnitine)   ELEVATED ↑     — supportive; C14-chain length
  C18:1-OH (3-hydroxyoleoylcarnitine)    ELEVATED ↑     — supportive; C18:1-chain length
  C16 (palmitoylcarnitine)               ELEVATED ↑     — non-specific; less than VLCAD
  C8 (octanoylcarnitine)                 NORMAL          — KEY NEGATIVE vs MCAD (C8 is MCAD marker)
  C14:1 (tetradecenoylcarnitine)         NORMAL          — KEY NEGATIVE vs VLCAD (C14:1 is VLCAD marker)
  Urine 3-OH dicarboxylic acids          ELEVATED ↑     — 3-OH-C6DC, 3-OH-C8DC, 3-OH-C12DC; PATHOGNOMONIC
  Blood glucose                          LOW during crisis — hypoketotic hypoglycaemia
  Ketones (β-OHB)                        LOW/ABSENT — hypoketosis despite hypoglycaemia
  CK (creatine kinase)                   ELEVATED — rhabdomyolysis (10,000–100,000 U/L)
  ALT/AST                                ELEVATED during crisis — hepatopathy
  Lactate                                ELEVATED — secondary lactic acidosis (NADH accumulation blocks pyruvate dehydrogenase)
  Troponin I/T                           ELEVATED in cardiomyopathy (especially TFP)
"""
import random

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
random.seed(273)

N_PATIENTS = 40

PHENOTYPES = [
    ("Classic LCHAD (Infantile/Childhood)", 0.55),   # isolated LCHAD — p.Glu474Gln homozygous
    ("Severe TFP (Neonatal)",              0.25),    # complete TFP deficiency — HADHB or compound het
    ("Adult/Adolescent (Myopathic)",       0.20),    # mild; exercise rhabdo; developing retinopathy
]

VARIANTS = [
    {
        "variant":      "p.Glu474Gln (c.1528G>A) — HADHA",
        "frequency":    "50–90% (Northern European) — FOUNDER",
        "gene":         "HADHA",
        "effect":       "Isolated LCHAD deficiency (Step 3 only); hydratase + thiolase preserved",
        "mechanism":    "Glu474 is the catalytic base of LCHAD active site; Gln substitution abolishes NAD+ binding → LCHAD inactive",
        "phenotype":    "Classic LCHAD — infantile/childhood episodic; retinopathy ± neuropathy prominent",
        "exam_pearl":   "FOUNDER mutation; when HOMOZYGOUS = isolated LCHAD (Step 3 only blocked); compound het with LOF = TFP phenotype",
    },
    {
        "variant":      "p.Arg452Cys (c.1354C>T) — HADHA",
        "frequency":    "~10%",
        "gene":         "HADHA",
        "effect":       "More complete MTP/TFP involvement; hydratase activity also reduced",
        "mechanism":    "Arg452 in LCHAD domain; Cys substitution disrupts domain folding → α subunit misfolded → TFP destabilized",
        "phenotype":    "TFP phenotype — severe; cardiomyopathy + liver involvement; less retinopathy",
        "exam_pearl":   "Distinguished from p.Glu474Gln by more complete TFP enzyme loss",
    },
    {
        "variant":      "p.Leu440Pro (c.1319T>C) — HADHA",
        "frequency":    "~8%",
        "gene":         "HADHA",
        "effect":       "Severe; disrupts LCHAD catalytic domain helix",
        "mechanism":    "Proline introduction into helix destroys secondary structure → LCHAD domain unfolded",
        "phenotype":    "Severe LCHAD/TFP; neonatal onset common",
        "exam_pearl":   "Structural class variant; Pro breaks helices",
    },
    {
        "variant":      "Large exonic deletion — HADHA/HADHB",
        "frequency":    "~7%",
        "gene":         "HADHA or HADHB",
        "effect":       "Null; complete loss of MTP subunit",
        "mechanism":    "Exon deletion → frameshift/truncation → no protein → complete TFP deficiency",
        "phenotype":    "TFP deficiency — severe; neonatal; cardiomyopathy",
        "exam_pearl":   "Detected by MLPA not Sanger; always complete TFP (null allele)",
    },
    {
        "variant":      "p.Gly846Ser (c.2536G>A) — HADHB",
        "frequency":    "~5%",
        "gene":         "HADHB",
        "effect":       "HADHB thiolase (Step 4) deficiency",
        "mechanism":    "Gly846 at β-subunit active site; Ser substitution reduces 3-ketoacyl-CoA thiolase activity",
        "phenotype":    "TFP deficiency via β subunit; severe infantile",
        "exam_pearl":   "HADHB variant → TFP (not isolated LCHAD); Step 4 (thiolase) primarily affected",
    },
    {
        "variant":      "Compound heterozygous (p.Glu474Gln + null)",
        "frequency":    "~10%",
        "gene":         "HADHA",
        "effect":       "More severe than homozygous p.Glu474Gln; TFP phenotype",
        "mechanism":    "One allele: isolated LCHAD defect; other allele: no protein → net TFP deficiency",
        "phenotype":    "Classic LCHAD to TFP depending on null allele; intermediate severity",
        "exam_pearl":   "Genotype-phenotype correlation important: homozygous p.Glu474Gln = isolated LCHAD; compound het with null = TFP",
    },
]


def _weighted_choice(options):
    r = random.random()
    cumulative = 0
    for label, prob in options:
        cumulative += prob
        if r < cumulative:
            return label
    return options[-1][0]


def _make_cohort():
    cohort = []
    for i in range(N_PATIENTS):
        ph = _weighted_choice(PHENOTYPES)
        is_classic  = ph == "Classic LCHAD (Infantile/Childhood)"
        is_severe   = ph == "Severe TFP (Neonatal)"
        is_adult    = ph == "Adult/Adolescent (Myopathic)"

        onset_mo = (
            random.randint(0, 2)     if is_severe  else
            random.randint(3, 60)    if is_classic else
            random.randint(120, 360)  # 10–30 years
        )

        # C16-OH (3-hydroxypalmitoylcarnitine) — PRIMARY NBS MARKER
        c16_oh = round(
            random.uniform(0.8, 5.0)   if is_severe  else
            random.uniform(0.2, 2.5)   if is_classic else
            random.uniform(0.08, 0.8),
            3
        )

        # C16-OH / C16 ratio — highly specific
        c16 = round(random.uniform(1.5, 8.0), 2)
        c16_oh_c16_ratio = round(c16_oh / c16, 3)

        # C14-OH (3-hydroxymyristoylcarnitine)
        c14_oh = round(
            random.uniform(0.15, 0.80)  if is_severe  else
            random.uniform(0.05, 0.35)  if is_classic else
            random.uniform(0.02, 0.12),
            3
        )

        # C18:1-OH (3-hydroxyoleoylcarnitine)
        c18_1_oh = round(
            random.uniform(0.20, 1.20)  if is_severe  else
            random.uniform(0.08, 0.50)  if is_classic else
            random.uniform(0.02, 0.15),
            3
        )

        # C8 (octanoylcarnitine) — NORMAL (KEY NEGATIVE vs MCAD)
        c8 = round(random.uniform(0.03, 0.12), 3)

        # C14:1 (tetradecenoylcarnitine) — NORMAL (KEY NEGATIVE vs VLCAD)
        c14_1 = round(random.uniform(0.02, 0.10), 3)

        # C0 (free carnitine) — depleted
        c0 = round(
            random.uniform(4, 14)   if is_severe  else
            random.uniform(10, 22)  if is_classic else
            random.uniform(18, 38),
            1
        )

        # Blood glucose
        glucose = (
            round(random.uniform(0.5, 2.5), 1)  if is_severe  else
            round(random.uniform(1.2, 3.2), 1)  if is_classic else
            round(random.uniform(3.5, 5.5), 1)
        )

        # Beta-OHB — LOW (hypoketosis)
        bohb = round(
            random.uniform(0.03, 0.25)  if is_severe  else
            random.uniform(0.05, 0.35)  if is_classic else
            random.uniform(0.4, 2.0),
            2
        )

        # CK — rhabdomyolysis
        ck = (
            random.randint(200, 3000)     if is_severe  else
            random.randint(500, 15000)    if is_classic else
            random.randint(5000, 100000)
        )

        # Retinopathy (UNIQUE to LCHAD)
        retinopathy = (
            random.random() < 0.15  if is_severe  else   # TFP: less specific (dies early)
            random.random() < 0.75  if is_classic else   # Classic LCHAD: hallmark
            random.random() < 0.60                        # Adult: developing
        )

        # Peripheral neuropathy (UNIQUE)
        neuropathy = (
            random.random() < 0.10  if is_severe  else
            random.random() < 0.55  if is_classic else
            random.random() < 0.70
        )

        # Cardiomyopathy (especially TFP)
        cardiomyopathy = (
            random.random() < 0.75  if is_severe  else
            random.random() < 0.20  if is_classic else
            random.random() < 0.08
        )

        # Maternal AFLP (retrospective — maternal history)
        maternal_aflp = (
            random.random() < 0.40  if is_severe  else
            random.random() < 0.30  if is_classic else
            random.random() < 0.15
        )

        # Lactate — secondary lactic acidosis
        lactate = round(
            random.uniform(4.0, 15.0)  if is_severe  else
            random.uniform(2.0, 8.0)   if is_classic else
            random.uniform(1.0, 3.5),
            1
        )

        # ALT
        alt = (
            random.randint(200, 800)  if is_severe  else
            random.randint(60, 300)   if is_classic else
            random.randint(20, 80)
        )

        # Variant assignment
        v_roll = random.random()
        if v_roll < 0.60:
            variant = VARIANTS[0]["variant"]  # p.Glu474Gln — FOUNDER
        elif v_roll < 0.70:
            variant = VARIANTS[1]["variant"]  # p.Arg452Cys
        elif v_roll < 0.78:
            variant = VARIANTS[2]["variant"]  # p.Leu440Pro
        elif v_roll < 0.85:
            variant = VARIANTS[3]["variant"]  # Large deletion
        elif v_roll < 0.90:
            variant = VARIANTS[4]["variant"]  # p.Gly846Ser (HADHB)
        else:
            variant = VARIANTS[5]["variant"]  # Compound het

        # Seizures — secondary to hypoglycaemia/metabolic crises
        seizures = (
            random.random() < 0.65  if is_severe  else
            random.random() < 0.40  if is_classic else
            random.random() < 0.08
        )

        # Treatments
        tx = ["MCT Oil (long-term dietary)", "Avoid Fasting Protocol", "Long-Chain Fat Restriction"]
        if is_severe or is_classic:
            tx.append("IV Glucose 10%")
            tx.append("Emergency Protocol Card")
        if retinopathy:
            tx.append("DHA Supplementation (retinal/neural)")
            tx.append("Annual Ophthalmology (ERG + fundoscopy)")
        if neuropathy:
            tx.append("Physiotherapy / Neuropathy management")
        if cardiomyopathy:
            tx.append("Cardiac Monitoring (Echo/ECG)")
        if is_adult:
            tx.append("Triheptanoin C7 Oil (anaplerosis)")
            tx.append("Exercise Warm-Up Protocol")
        if c0 < 12:
            tx.append("L-Carnitine (if C0 depleted)")
        if ck > 10000:
            tx.append("IV Fluids (rhabdomyolysis — prevent AKI)")

        cohort.append({
            "id":             f"LCHAD-{i+1:03d}",
            "phenotype":      ph,
            "onset_mo":       onset_mo,
            "c16_oh":         c16_oh,
            "c16_oh_c16":     c16_oh_c16_ratio,
            "c16":            c16,
            "c14_oh":         c14_oh,
            "c18_1_oh":       c18_1_oh,
            "c8":             c8,
            "c14_1":          c14_1,
            "c0":             c0,
            "glucose":        glucose,
            "bohb":           bohb,
            "ck":             ck,
            "lactate":        lactate,
            "alt":            alt,
            "retinopathy":    retinopathy,
            "neuropathy":     neuropathy,
            "cardiomyopathy": cardiomyopathy,
            "maternal_aflp":  maternal_aflp,
            "variant":        variant,
            "seizures":       seizures,
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

    retinopathy_n   = sum(1 for p in cohort if p["retinopathy"])
    neuropathy_n    = sum(1 for p in cohort if p["neuropathy"])
    cardiac_n       = sum(1 for p in cohort if p["cardiomyopathy"])
    maternal_aflp_n = sum(1 for p in cohort if p["maternal_aflp"])
    seizure_n       = sum(1 for p in cohort if p["seizures"])
    rhabdo_n        = sum(1 for p in cohort if p["ck"] >= 5000)

    return {
        "n_patients":      N_PATIENTS,
        "disease":         "LCHAD / MTP Deficiency (Long-Chain 3-Hydroxyacyl-CoA Dehydrogenase / Mitochondrial Trifunctional Protein)",
        "gene_primary":    "HADHA (MTP alpha subunit — enoyl-CoA hydratase + LCHAD activities)",
        "gene_secondary":  "HADHB (MTP beta subunit — 3-ketoacyl-CoA thiolase activity)",
        "locus":           "2p23.3 (both HADHA and HADHB adjacent)",
        "inheritance":     "Autosomal Recessive (AR); biallelic LOF; MTP = α2β2 inner mitochondrial membrane complex",
        "prevalence":      "~1:75,000–100,000 (LCHAD+TFP combined); LCHAD alone ~1:100,000–250,000",
        "nbs_marker":      "C16-OH (3-hydroxypalmitoylcarnitine) >0.08 µmol/L — PRIMARY NBS",
        "pathway_role":    "MTP = Steps 2+3+4 of beta-oxidation for long-chain (C12–C18) fatty acids; LCHAD = Step 3 specifically",
        "phenotype_distribution": ph_counts,
        "variant_distribution":   variant_counts,
        "clinical_summary": {
            "retinopathy_n":   retinopathy_n,
            "retinopathy_pct": round(retinopathy_n / N_PATIENTS * 100, 1),
            "neuropathy_n":    neuropathy_n,
            "neuropathy_pct":  round(neuropathy_n / N_PATIENTS * 100, 1),
            "cardiomyopathy_n": cardiac_n,
            "cardiomyopathy_pct": round(cardiac_n / N_PATIENTS * 100, 1),
            "maternal_aflp_n": maternal_aflp_n,
            "maternal_aflp_pct": round(maternal_aflp_n / N_PATIENTS * 100, 1),
            "seizures_n":      seizure_n,
            "seizures_pct":    round(seizure_n / N_PATIENTS * 100, 1),
            "rhabdo_n":        rhabdo_n,
            "rhabdo_pct":      round(rhabdo_n / N_PATIENTS * 100, 1),
        },
        "mean_c16_oh":   round(sum(p["c16_oh"] for p in cohort) / N_PATIENTS, 3),
        "mean_ck":       round(sum(p["ck"] for p in cohort) / N_PATIENTS),
        "mean_glucose":  round(sum(p["glucose"] for p in cohort) / N_PATIENTS, 2),
        "mean_lactate":  round(sum(p["lactate"] for p in cohort) / N_PATIENTS, 2),
        "unique_features": [
            "Retinal pigmentary degeneration — UNIQUE among ALL FAO disorders",
            "Peripheral neuropathy (axonal) — UNIQUE among FAO disorders",
            "Maternal AFLP/HELLP (fetal LCHAD deficiency → maternal hepatotoxicity) — UNIQUE",
            "3-OH-acylcarnitines (C16-OH, C14-OH, C18:1-OH) elevated — not plain acylcarnitines",
        ],
        "key_negatives": [
            "C8 NORMAL — KEY NEGATIVE vs MCAD (MCAD elevates C8; LCHAD does NOT)",
            "C14:1 NORMAL — KEY NEGATIVE vs VLCAD (VLCAD elevates C14:1; LCHAD does NOT)",
            "No HG/SG/PPG (glycine conjugates) — KEY NEGATIVE vs MCAD",
        ],
        "founder_mutation": "p.Glu474Gln (c.1528G>A) HADHA — 50–90% Northern European; isolated LCHAD Step 3 deficiency",
    }


def get_breakdown():
    cohort = _get_cohort()
    patients = cohort[:25]  # representative subset for detailed breakdown

    biomarkers = {
        "c16_oh": {
            "label":     "C16-OH (3-Hydroxypalmitoylcarnitine)",
            "normal":    "<0.08 µmol/L",
            "status":    "ELEVATED >0.08 µmol/L — PRIMARY NBS MARKER",
            "direction": "↑↑↑",
            "color":     "danger",
            "rationale": (
                "3-Hydroxypalmitoylcarnitine (C16-OH) is the ester of 3-hydroxypalmitic acid with carnitine. "
                "It accumulates because LCHAD cannot oxidize L-3-hydroxypalmitoyl-CoA (Step 3 blocked). "
                "C16-OH is the most sensitive and specific NBS marker for LCHAD/TFP deficiency on tandem MS/MS. "
                "Threshold >0.08 µmol/L; values >0.5 µmol/L in severe phenotype. "
                "The '3-OH' prefix is critical: this is NOT plain palmitoylcarnitine (C16) elevated in VLCAD."
            ),
        },
        "c16_oh_c16_ratio": {
            "label":     "C16-OH / C16 Ratio (3-OH-Palmitate:Palmitate)",
            "normal":    "<0.07",
            "status":    "ELEVATED >0.07 — highly specific for LCHAD/TFP",
            "direction": "↑↑",
            "color":     "danger",
            "rationale": (
                "The C16-OH/C16 ratio corrects for non-specific C16 elevation (e.g. heterozygous mothers, non-fasting states). "
                "A ratio >0.07 is highly specific for LCHAD/TFP deficiency. "
                "Helps distinguish true LCHAD from benign C16 elevation. "
                "Best secondary discriminator after C16-OH absolute value."
            ),
        },
        "c14_oh": {
            "label":     "C14-OH (3-Hydroxymyristoylcarnitine)",
            "normal":    "<0.03 µmol/L",
            "status":    "ELEVATED — supportive marker",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "L-3-Hydroxymyristoyl-CoA (C14-OH-CoA) also accumulates in LCHAD deficiency "
                "because LCHAD processes C14-chain length substrates too. "
                "C14-OH elevated alongside C16-OH confirms the 3-hydroxy-acylcarnitine pattern of LCHAD. "
                "Less diagnostic than C16-OH alone but confirms chain-length spectrum."
            ),
        },
        "c18_1_oh": {
            "label":     "C18:1-OH (3-Hydroxyoleoylcarnitine)",
            "normal":    "<0.05 µmol/L",
            "status":    "ELEVATED — supportive; confirms long-chain 3-OH profile",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "3-Hydroxyoleoyl-CoA accumulates when LCHAD cannot process this C18:1-chain 3-OH intermediate. "
                "C18:1-OH is particularly relevant because oleic acid (C18:1) is the dominant dietary monounsaturated fat. "
                "The triad C16-OH + C14-OH + C18:1-OH defines the complete LCHAD 3-OH-acylcarnitine profile. "
                "Critical note: C18:1 (plain oleoylcarnitine, no OH prefix) is elevated in VLCAD — the 3-OH prefix distinguishes LCHAD."
            ),
        },
        "c8": {
            "label":     "C8 (Octanoylcarnitine)",
            "normal":    "<0.15 µmol/L",
            "status":    "NORMAL — KEY NEGATIVE vs MCAD",
            "direction": "→ NORMAL",
            "color":     "success",
            "rationale": (
                "C8 (octanoylcarnitine) is the PRIMARY NBS marker for MCAD deficiency, not LCHAD. "
                "In LCHAD/MTP deficiency, C8 is NORMAL because MCAD (which processes C6–C12) is intact. "
                "C8 NORMAL + C16-OH ELEVATED = LCHAD pattern. "
                "If both C8 AND C16-OH are elevated: consider MADD (ETFA/ETFB/ETFDH) which elevates all chain lengths."
            ),
        },
        "c14_1": {
            "label":     "C14:1 (Tetradecenoylcarnitine)",
            "normal":    "<0.12 µmol/L",
            "status":    "NORMAL — KEY NEGATIVE vs VLCAD",
            "direction": "→ NORMAL",
            "color":     "success",
            "rationale": (
                "C14:1 (tetradecenoylcarnitine) is the PRIMARY NBS marker for VLCAD deficiency, not LCHAD. "
                "In LCHAD/MTP deficiency, C14:1 is NORMAL because VLCAD (which catalyses Step 1 for C14–C20) is intact. "
                "LCHAD blocks Step 3 for C12–C18; VLCAD blocks Step 1 for C14–C20. "
                "C14:1 NORMAL + C16-OH ELEVATED = distinguishes LCHAD from VLCAD."
            ),
        },
        "glucose": {
            "label":     "Blood Glucose",
            "normal":    "3.5–6.0 mmol/L",
            "status":    "LOW during crisis — hypoketotic hypoglycaemia",
            "direction": "↓ (crisis)",
            "color":     "danger",
            "rationale": (
                "Same mechanism as VLCAD/MCAD: fasting depletes glucose → long-chain FAO attempt → LCHAD blocked → "
                "no acetyl-CoA from C12–C18 → no ketones → brain deprived of both glucose and ketones. "
                "Emergency: IV glucose 10% at 8–10 mg/kg/min immediately. "
                "Hypoketotic hypoglycaemia most severe in classic infantile and neonatal phenotypes."
            ),
        },
        "bohb": {
            "label":     "β-Hydroxybutyrate (Ketones)",
            "normal":    "0.1–0.5 mmol/L (fasting)",
            "status":    "LOW/ABSENT during crisis — inappropriately low for degree of hypoglycaemia",
            "direction": "↓ (hypoketosis)",
            "color":     "warning",
            "rationale": (
                "The hallmark 'hypoketosis' in LCHAD: blood glucose is LOW but ketones are inappropriately LOW. "
                "Normally, hypoglycaemia triggers fat mobilization → ketogenesis → ketones as brain fuel. "
                "In LCHAD, long-chain fatty acids cannot progress to acetyl-CoA (LCHAD blocked) → no ketones. "
                "Low ketones despite hypoglycaemia = LCHAD, VLCAD, MCAD, or other FAO disorder (not starvation)."
            ),
        },
        "ck": {
            "label":     "CK (Creatine Kinase) — Rhabdomyolysis Marker",
            "normal":    "<200 U/L",
            "status":    "ELEVATED during episodes — rhabdomyolysis",
            "direction": "↑↑ (episodic)",
            "color":     "danger",
            "rationale": (
                "Rhabdomyolysis occurs when skeletal muscle relies on long-chain FAO during exercise/fasting → "
                "LCHAD block → energy deficit → myocyte necrosis → CK release. "
                "Most prominent in adult/adolescent myopathic phenotype. "
                "CK >10,000 U/L → AKI risk from myoglobin. "
                "IV fluids immediately to flush myoglobin from tubules."
            ),
        },
        "lactate": {
            "label":     "Lactate",
            "normal":    "<2.0 mmol/L",
            "status":    "ELEVATED — secondary lactic acidosis",
            "direction": "↑",
            "color":     "warning",
            "rationale": (
                "LCHAD block → NADH NOT generated from Step 3 → BUT also NADH accumulates from incomplete beta-oxidation "
                "and from 3-OH-acyl-CoA accumulation. "
                "Elevated NADH/NAD+ ratio → pyruvate → lactate (Cori cycle backup). "
                "Lactic acidosis more prominent than in MCAD; distinguishes LCHAD from MCAD in acute crisis. "
                "Secondary lactate (not primary mtDNA disorder): DNA sequencing normal."
            ),
        },
    }

    treatments = {
        "mct_oil": {
            "label": "MCT Oil (Medium-Chain Triglycerides C8/C10)",
            "level": "Level A — Primary long-term dietary fat source",
            "rationale": (
                "MCT oil provides C8 (caprylic) and C10 (capric) acids — these bypass LCHAD/MTP completely. "
                "C8–C10 do not require MTP Step 2/3/4 for their chain-length range (those are handled by SCAD/MCAD in the matrix). "
                "MCT provides usable fat-derived energy to all tissues without needing MTP. "
                "Dose: 1.5–3 g/kg/day; given with meals. "
                "Critical: MCT-based formula is the standard dietary management for infants with LCHAD."
            ),
        },
        "avoid_fasting": {
            "label": "Avoid Fasting (Absolute — Level A)",
            "level": "Level A — Most preventable trigger",
            "rationale": (
                "Fasting forces long-chain FAO → LCHAD blocked → no C12–C18 derived acetyl-CoA → crisis. "
                "Protocol: never fast >4h (infants <6 mo), >6h (infants 6–12 mo), >8h (children), >10h (adults). "
                "During illness/surgery: IV glucose 10% at 8–10 mg/kg/min mandatory. "
                "Emergency protocol card for all patients."
            ),
        },
        "lc_fat_restriction": {
            "label": "Long-Chain Fat Restriction (Level A)",
            "level": "Level A — Reduces substrate load on blocked MTP",
            "rationale": (
                "Restricting dietary long-chain saturated + monounsaturated fats (C12–C18) reduces substrate for blocked MTP. "
                "Target: long-chain fat <10–15% of total energy. "
                "Replaced by MCT oil as primary fat source. "
                "Same principle as VLCAD management (both blocked at long-chain oxidation steps)."
            ),
        },
        "dha_supplement": {
            "label": "DHA (Docosahexaenoic Acid) Supplementation (Level B)",
            "level": "Level B — Retinal and neural protection",
            "rationale": (
                "DHA (C22:6, omega-3) is the critical retinal and neural membrane fatty acid. "
                "In LCHAD deficiency, DHA synthesis from alpha-linolenic acid (ALA) requires LCHAD at the C18:1-OH stage. "
                "LCHAD block → DHA cannot be elongated from precursors → retinal and neural DHA depletion. "
                "DHA supplementation (100–200 mg/day) may slow retinal degeneration and peripheral neuropathy. "
                "UNIQUE to LCHAD among FAO disorders — no other FAO disorder needs DHA supplement for retinal disease."
            ),
        },
        "triheptanoin": {
            "label": "Triheptanoin (C7 Anaplerotic Oil — Level B)",
            "level": "Level B — Anaplerotic TCA support",
            "rationale": (
                "Triheptanoin is a triglyceride of heptanoic acid (C7 odd-chain). "
                "Metabolized to propionyl-CoA + acetyl-CoA; propionyl-CoA enters TCA as succinyl-CoA (anaplerosis). "
                "Replenishes TCA intermediates; particularly useful in myopathic adult phenotype. "
                "Same rationale as in VLCAD management."
            ),
        },
        "iv_glucose": {
            "label": "IV Glucose 10% (Level A — Emergency)",
            "level": "Level A — Emergency anti-catabolic therapy",
            "rationale": (
                "During LCHAD crisis: IV glucose 10% at 8–10 mg/kg/min provides glucose → suppresses lipolysis → "
                "stops the demand on blocked long-chain FAO. "
                "Immediately reverses hypoketotic hypoglycaemia. "
                "First-line emergency measure alongside IV fluids (for rhabdomyolysis prevention)."
            ),
        },
        "vpa_avoid": {
            "label": "VPA (Valproic Acid) — HIGH RISK; Avoid",
            "level": "HIGH RISK — Avoid in LCHAD deficiency",
            "rationale": (
                "VPA inhibits mitochondrial FAO globally: depletes carnitine + inhibits electron chain. "
                "In LCHAD deficiency, VPA worsens the already-impaired long-chain FAO → acute energy crisis. "
                "3-OH-acylcarnitine accumulation may be exacerbated by VPA-induced FAO suppression. "
                "Substitute: levetiracetam, lamotrigine, oxcarbazepine preferred AEDs."
            ),
        },
        "kd_absolute_ci": {
            "label": "Ketogenic Diet — ABSOLUTE CONTRAINDICATION",
            "level": "ABSOLUTE CI — Floods long-chain fat on blocked MTP",
            "rationale": (
                "KD uses long-chain fat (C12–C18, cream/butter) as primary substrate → floods blocked MTP → "
                "massively increases 3-OH-acylcarnitines → retinal/neurological toxicity + rhabdomyolysis + hypoglycaemia. "
                "Absolute CI in LCHAD (same as VLCAD). "
                "MCT-based KD also CI (MCT is safe for energy but KD formulations include long-chain fat)."
            ),
        },
    }

    return {
        "n_patients":   N_PATIENTS,
        "patients":     [
            {
                "id":             p["id"],
                "phenotype":      p["phenotype"],
                "onset_mo":       p["onset_mo"],
                "c16_oh":         p["c16_oh"],
                "c16_oh_c16":     p["c16_oh_c16"],
                "c14_oh":         p["c14_oh"],
                "c18_1_oh":       p["c18_1_oh"],
                "c8":             p["c8"],
                "c14_1":          p["c14_1"],
                "c0":             p["c0"],
                "glucose":        p["glucose"],
                "ck":             p["ck"],
                "lactate":        p["lactate"],
                "retinopathy":    p["retinopathy"],
                "neuropathy":     p["neuropathy"],
                "cardiomyopathy": p["cardiomyopathy"],
                "maternal_aflp":  p["maternal_aflp"],
                "variant":        p["variant"],
                "seizures":       p["seizures"],
                "treatments":     p["treatments"],
            }
            for p in patients
        ],
        "biomarkers":   biomarkers,
        "treatments":   treatments,
        "variants":     VARIANTS,
        "phenotype_profiles": {
            "Classic LCHAD (Infantile/Childhood)": {
                "prevalence": "~55% (most common)",
                "genotype":   "Homozygous p.Glu474Gln (FOUNDER) — isolated LCHAD (Step 3 only)",
                "onset":      "3 months to 5 years",
                "features":   "Episodic hypoketotic hypoglycaemia; progressive retinopathy; neuropathy developing; hepatomegaly; ± mild cardiomyopathy",
                "prognosis":  "Good quality of life with dietary management; retinopathy progressive despite treatment",
                "key_point":  "Retinopathy + neuropathy are UNIQUE; follow annually with ERG + EMG/NCS; DHA supplementation started early",
            },
            "Severe TFP (Neonatal)": {
                "prevalence": "~25%",
                "genotype":   "HADHB LOF or compound het HADHA — complete MTP (Steps 2+3+4 all lost)",
                "onset":      "Neonatal (first 2 months)",
                "features":   "Severe cardiomyopathy (DCM/HCM); hypoglycaemia; liver failure; lactic acidosis; sudden cardiac death risk",
                "prognosis":  "High neonatal mortality without NBS; cardiomyopathy partially reversible with MCT + fasting avoidance",
                "key_point":  "Complete TFP = cardiomyopathy prominent (all 3 MTP steps lost); more severe than isolated LCHAD",
            },
            "Adult/Adolescent (Myopathic)": {
                "prevalence": "~20%",
                "genotype":   "Mild compound het or homozygous hypomorphic",
                "onset":      "10–30 years (adolescence or adulthood)",
                "features":   "Exercise-induced rhabdomyolysis; CK >10,000 U/L; developing retinopathy; peripheral neuropathy",
                "prognosis":  "Good with avoidance of triggers; retinopathy and neuropathy progressive",
                "key_point":  "May present as unexplained exercise rhabdomyolysis; retinopathy at examination helps diagnose; DHA started",
            },
        },
    }


def get_definitions():
    return {
        "disease":         "LCHAD / MTP Deficiency (Long-Chain 3-Hydroxyacyl-CoA Dehydrogenase / Mitochondrial Trifunctional Protein Deficiency)",
        "gene_primary":    "HADHA — encodes MTP alpha subunit (763 aa); enoyl-CoA hydratase (Step 2) + LCHAD (Step 3) activities",
        "gene_secondary":  "HADHB — encodes MTP beta subunit (474 aa); 3-ketoacyl-CoA thiolase (Step 4) activity",
        "locus":           "2p23.3 (HADHA and HADHB are adjacent genes at same locus)",
        "omim_gene_hadha": "*600890 (HADHA)",
        "omim_gene_hadhb": "*143450 (HADHB)",
        "omim_disease_lchad": "#609016 (isolated LCHAD deficiency — HADHA p.Glu474Gln)",
        "omim_disease_tfp":   "#609015 (complete TFP deficiency — HADHB or other HADHA)",
        "inheritance":     "Autosomal Recessive (AR); biallelic LOF required",
        "prevalence":      "~1:75,000–100,000 combined LCHAD+TFP; isolated LCHAD ~1:100,000–250,000",
        "protein_structure": (
            "MTP = HADHA(α)2 · HADHB(β)2 heterotetramer anchored to inner mitochondrial membrane. "
            "HADHA α-subunit (763 aa): N-terminal 2-enoyl-CoA hydratase domain (Step 2) + C-terminal LCHAD domain (Step 3). "
            "HADHB β-subunit (474 aa): 3-ketoacyl-CoA thiolase domain (Step 4). "
            "LCHAD (p.Glu474Gln): isolated Step 3 deficiency only; Steps 2+4 preserved. "
            "TFP: all 3 steps lost (HADHB null or severe HADHA mutation destabilizes whole complex)."
        ),
        "enzyme_function": (
            "MTP catalyses Steps 2, 3, and 4 of mitochondrial beta-oxidation for LONG-CHAIN (C12–C18) fatty acids: "
            "Step 2 (HADHA hydratase): trans-2-enoyl-acyl-CoA + H2O → L-3-hydroxyacyl-CoA. "
            "Step 3 (HADHA LCHAD): L-3-hydroxyacyl-CoA + NAD+ → 3-ketoacyl-CoA + NADH. [BLOCKED IN LCHAD] "
            "Step 4 (HADHB thiolase): 3-ketoacyl-CoA + CoA → acyl-CoA(n-2) + acetyl-CoA. "
            "VLCAD catalyses Step 1 (C14–C20 dehydrogenation) upstream of MTP. "
            "SCAD/MCAD handle shorter chains (C4–C12) — these are NOT substrates for MTP."
        ),
        "pathomechanism": (
            "LCHAD (Step 3) deficiency → L-3-hydroxyacyl-CoAs CANNOT be dehydrogenated → accumulate → "
            "form 3-OH-acylcarnitines in blood (C16-OH, C14-OH, C18:1-OH). "
            "NADH not generated at Step 3 → energy deficit. "
            "3-OH-acylcarnitines are directly toxic to retinal pigment epithelium → pigmentary retinopathy. "
            "DHA synthesis impaired (LCHAD needed at C18:1-OH step of DHA elongation) → retinal/neural DHA depletion. "
            "Maternal hepatotoxicity: fetal 3-OH-fatty acids cross placenta → accumulate in maternal liver → AFLP/HELLP."
        ),
        "nbs_markers": {
            "primary":       "C16-OH (3-hydroxypalmitoylcarnitine) >0.08 µmol/L — tandem MS/MS",
            "secondary":     ["C14-OH", "C18:1-OH"],
            "best_ratio":    "C16-OH/C16 >0.07 — most specific discriminator",
            "key_negatives": [
                "C8 NORMAL (MCAD elevates C8 — LCHAD does NOT)",
                "C14:1 NORMAL (VLCAD elevates C14:1 — LCHAD does NOT)",
                "No HG/SG/PPG (MCAD-specific glycine conjugates — absent in LCHAD)",
            ],
        },
        "unique_clinical_triad": [
            "Retinal pigmentary degeneration (UNIQUE among ALL FAO disorders — no other FAO causes retinopathy)",
            "Peripheral neuropathy — axonal (UNIQUE among FAO disorders)",
            "Maternal AFLP/HELLP (fetal LCHAD → maternal hepatotoxicity — UNIQUE FAO maternal-fetal interaction)",
        ],
        "also_present": ["Hypoketotic hypoglycaemia", "Cardiomyopathy (esp. TFP)", "Rhabdomyolysis", "Lactic acidosis (secondary)"],
        "key_differentials": {
            "vs_VLCAD": (
                "C16-OH elevated (LCHAD) vs C14:1 elevated (VLCAD). "
                "C14:1 NORMAL in LCHAD (KEY NEGATIVE). "
                "Retinopathy + neuropathy present in LCHAD (absent in VLCAD). "
                "Maternal AFLP in LCHAD (absent in VLCAD). "
                "Cardiomyopathy: hallmark in VLCAD infantile; more in TFP not isolated LCHAD."
            ),
            "vs_MCAD": (
                "C8 NORMAL in LCHAD (KEY NEGATIVE — C8 is MCAD primary marker). "
                "3-OH-acylcarnitines elevated in LCHAD; plain acylcarnitines in MCAD. "
                "No HG/SG/PPG in LCHAD (MCAD pathognomonic). "
                "Retinopathy/neuropathy in LCHAD (absent in MCAD). "
                "MCAD: 1p31.1; LCHAD: 2p23.3."
            ),
            "vs_MADD": (
                "MADD (ETFA/ETFB/ETFDH deficiency) elevates C4-C18 ALL chain lengths. "
                "LCHAD: only C16-OH, C14-OH, C18:1-OH (long-chain 3-OH pattern). "
                "MADD: C8 elevated; LCHAD: C8 NORMAL. "
                "MADD: riboflavin-responsive; LCHAD: not riboflavin-responsive."
            ),
        },
        "treatment_overview": {
            "first_line":    "MCT oil + fasting avoidance + long-chain fat restriction",
            "emergency":     "IV glucose 10% 8–10 mg/kg/min; IV fluids for rhabdomyolysis; MCT orally",
            "absolute_ci":   ["Fasting", "Ketogenic diet (long-chain fat floods blocked MTP — catastrophic)"],
            "high_risk":     ["VPA (inhibits FAO; depletes carnitine)"],
            "unique_therapy": "DHA supplementation (retinal/neural protection — UNIQUE to LCHAD among FAO disorders)",
            "monitoring":    "Annual ERG + fundoscopy (retinopathy); EMG/NCS (neuropathy); Echo/ECG; acylcarnitine profile quarterly; ophthalmology",
        },
        "key_exam_pearls": [
            "C16-OH (3-OH prefix CRITICAL) — not plain C16 — is the PRIMARY NBS marker for LCHAD; C16 elevated in VLCAD",
            "Retinal pigmentary degeneration = LCHAD; the ONLY FAO disorder causing retinopathy",
            "Peripheral neuropathy = LCHAD; the ONLY FAO disorder causing progressive axonal neuropathy",
            "Maternal AFLP/HELLP = LCHAD — a mother with unexplained AFLP → test newborn for LCHAD",
            "p.Glu474Gln (c.1528G>A) HADHA = FOUNDER; 50–90% Northern European; ISOLATED LCHAD Step 3 deficiency",
            "C14:1 NORMAL in LCHAD — KEY NEGATIVE vs VLCAD (VLCAD elevates C14:1; LCHAD does NOT)",
            "C8 NORMAL in LCHAD — KEY NEGATIVE vs MCAD (MCAD elevates C8; LCHAD does NOT)",
            "DHA supplementation is UNIQUE to LCHAD management (no other FAO disorder needs DHA for retinal protection)",
            "KD = ABSOLUTE CI in LCHAD (same as VLCAD — long-chain fat floods blocked enzyme)",
            "TFP deficiency (HADHB or non-p.Glu474Gln HADHA) = more severe; cardiomyopathy + Steps 2+3+4 all blocked",
            "LCHAD is on inner mitochondrial membrane (HADHA/HADHB MTP complex) — exam: C16-OH not C16",
            "3-hydroxy dicarboxylic acids in urine (3-OH-C6DC, 3-OH-C8DC, 3-OH-C12DC) = PATHOGNOMONIC for LCHAD",
        ],
        "variants": VARIANTS,
        "phenotype_summary": {
            "classic_lchad":  "55% — homozygous p.Glu474Gln; infantile/childhood; retinopathy + neuropathy; good prognosis with diet",
            "severe_tfp":     "25% — neonatal; complete MTP deficiency; cardiomyopathy; high mortality without NBS",
            "adult_myopathic": "20% — exercise rhabdomyolysis; developing retinopathy; mild/hypomorphic alleles",
        },
        "chromosome": "2p23.3",
        "protein_class": "Mitochondrial Trifunctional Protein (MTP); α2β2 heterotetramer; inner mitochondrial membrane; NAD+-dependent (LCHAD) + FAD-independent hydratase + CoA-thiolase",
        "related_disorders": [
            "VLCAD deficiency (ACADVL, C14–C20, C14:1 primary NBS, cardiomyopathy hallmark)",
            "MCAD deficiency (ACADM, C6–C12, C8 primary NBS, Reye-like, HG/SG/PPG pathognomonic)",
            "SCAD deficiency (ACADS, C4, butyrylcarnitine, usually benign)",
            "MADD/GA2 (ETFA/ETFB/ETFDH, all C4-C18 elevated, riboflavin-responsive)",
            "ACAT1/T2 deficiency (mitochondrial thiolase for ketolysis/isoleucine — distinct from HADHB thiolase)",
        ],
    }
