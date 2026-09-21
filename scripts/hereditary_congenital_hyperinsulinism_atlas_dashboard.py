#!/usr/bin/env python3
"""Hereditary-Congenital-Hyperinsulinism-Atlas — Complete 8-Gene CHI Atlas
ABCC8   (sulfonylurea receptor 1 / SUR1; 1582 aa; 11p15.1; AR/AD;
         most common K-ATP CHI (~65%); diazoxide-UNRESPONSIVE;
         18F-DOPA PET distinguishes focal (resectable) from diffuse;
         near-total pancreatectomy for diffuse diazoxide-unresponsive;
         seed SEED_BASE+0) ·
KCNJ11  (Kir6.2 / inward-rectifier K+ channel; 390 aa; 11p15.1; AR/AD;
         K-ATP CHI — pore-forming subunit; diazoxide-UNRESPONSIVE;
         same 18F-DOPA PET algorithm as ABCC8 for focal/diffuse;
         AD de novo: transient/mild; AR biallelic: severe diffuse;
         seed SEED_BASE+1) ·
GLUD1   (glutamate dehydrogenase 1; 558 aa; 10q23.3; AD GOF;
         HI/HA syndrome — hyperinsulinism-hyperammonaemia;
         protein-induced hypoglycaemia (leucine stimulus);
         plasma NH3 mildly elevated (60-200 µmol/L) PATHOGNOMONIC;
         diazoxide-RESPONSIVE; low-leucine diet; seed SEED_BASE+2) ·
GCK     (glucokinase; 465 aa; 7p13; AD GOF;
         glucosensor shift: lower glucose threshold for insulin secretion;
         variable severity (mild to severe); diazoxide-RESPONSIVE usually;
         mosaic: highly variable phenotype; seed SEED_BASE+3) ·
HADH    (3-hydroxyacyl-CoA dehydrogenase / SCHAD; 314 aa; 4q25; AR;
         protein-induced CHI — SCHAD inhibits GDH → LOF disinhibits GDH;
         urine 3-OH-glutaric acid elevated PATHOGNOMONIC;
         diazoxide-RESPONSIVE; good prognosis; seed SEED_BASE+4) ·
HNF4A   (hepatocyte nuclear factor 4 alpha; 474 aa; 20q13.12; AD LOF;
         macrosomia + neonatal CHI → MODY1 hyperglycaemia in adulthood;
         diazoxide-RESPONSIVE; fetal hyperinsulinism resolves;
         same gene causes MODY1 in adults — biphasic phenotype;
         seed SEED_BASE+5) ·
UCP2    (uncoupling protein 2; 309 aa; 11q13.4; AD GOF;
         mild CHI; mechanism: UCP2 GOF → mitochondrial uncoupling →
         reduces ATP/ADP ratio → K-ATP channel opens → K+ outflow → insulin decreases;
         GOF variant INCREASES uncoupling → paradox overcomes normal K-ATP regulation;
         spontaneous resolution by 3-6 years; diazoxide-RESPONSIVE;
         seed SEED_BASE+6) ·
SLC16A1 (monocarboxylate transporter 1 / MCT1; 478 aa; 17q24.2; AD GOF;
         exercise-induced hypoglycaemia (EIHI) — UNIQUE mechanism;
         promoter gain-of-function: pyruvate enters beta-cell → closes K-ATP → insulin surge;
         PATHOGNOMONIC: hypoglycaemia only with exercise (pyruvate released from muscle);
         diazoxide-UNHELPFUL (not K-ATP);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2910–2917)
"""
import random

SEED_BASE = 2910

ATLAS_GENES = [
    {
        "gene": "ABCC8",
        "protein": (
            "ABCC8 -- 11p15.1 AR/AD -- 1582aa -- Sulfonylurea-Receptor-1-SUR1-"
            "177kDa-K-ATP-Regulatory-Subunit-"
            "CHI-Most-Common-65pct-Diazoxide-Unresponsive-18F-DOPA-PET-Focal-Diffuse-"
            "OMIM-Gene-600509-Disease-CHI-256450"
        ),
        "locus": "11p15.1",
        "protein_size": (
            "1582 aa / 177 kDa (SUR1 — sulfonylurea receptor 1; "
            "ABC transporter family C; 3 domains: TMD0, TMD1-NBD1, TMD2-NBD2; "
            "ABCC8 + KCNJ11 form K-ATP octamer: (SUR1·Kir6.2)₄; "
            "SUR1 role: regulatory subunit — senses ADP/ATP ratio; "
            "glucose ↑ → ATP ↑/ADP ↓ → SUR1 ADP dissociates → K+ channel closes → "
            "membrane depolarises → Ca²⁺ influx → insulin secretion; "
            "ABCC8 LOF (AR): K-ATP channels constitutively CLOSED → "
            "  continuous membrane depolarisation → unregulated insulin release; "
            "DIAZOXIDE: opens K-ATP channels — UNRESPONSIVE in K-ATP CHI (channel absent/non-functional); "
            "ABCC8 AD: often transient TNDM-like neonatal CHI (milder); "
            "encoded 11p15.1 — same locus as KCNJ11 (adjacent genes); "
            "sulfonylureas (glibenclamide): bind SUR1 → FOR KCNJ11 neonatal diabetes (opposite direction — "
            "  closes K-ATP in diabetes, already closed in CHI); "
            "DISTINCTION: CHI = LOF (K-ATP closed) vs neonatal diabetes = GOF (K-ATP stuck open)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — most common pattern — LOSS-OF-FUNCTION — ABCC8 CHI: "
            "  DIFFUSE CHI: biallelic AR mutations → all beta-cells affected; "
            "    diazoxide-UNRESPONSIVE; near-total pancreatectomy often needed; "
            "  PATERNALLY INHERITED + LOH: focal CHI — single paternal ABCC8 mutation + "
            "    somatic loss of maternal 11p15 → focal adenomatous hyperplasia of beta-cells; "
            "    18F-DOPA PET localises focal lesion → laparoscopic partial pancreatectomy CURATIVE; "
            "AUTOSOMAL DOMINANT (AD): de novo or inherited; "
            "  often milder, transient neonatal CHI; diazoxide-responsive subgroup; "
            "PREVALENCE: ~1:25,000–50,000 births (most common cause of persistent neonatal hypoglycaemia)"
        ),
        "disease_category": (
            "K-ATP CHI — MOST COMMON CAUSE (~65% of all CHI): "
            "  PRESENTATION: "
            "    Neonatal hypoglycaemia — severe, within hours-days of birth; "
            "    Macrosomia (in-utero hyperinsulinism → fetal growth); "
            "    Blood glucose <2.6 mmol/L despite normal feeding; "
            "    Seizures from hypoglycaemia; "
            "  BIOCHEMISTRY (CRITICAL CLUE): "
            "    Hypoglycaemia + inappropriately elevated insulin (>2 mU/L when BG <3.0 mmol/L); "
            "    Suppressed ketones (hyperinsulinism = ANTI-KETOGENIC); "
            "    Low FFAs (insulin suppresses lipolysis); "
            "    High glucose infusion rate (GIR) required (>8 mg/kg/min); "
            "  DIAGNOSTIC STEPS: "
            "    1. Critical sample: glucose, insulin, ketones, FFAs, C-peptide; "
            "    2. Glucagon stimulation test: BG rise >1.5 mmol/L → confirms hyperinsulinism; "
            "    3. Diazoxide trial: UNRESPONSIVE (confirms K-ATP CHI); "
            "    4. 18F-DOPA PET-CT: distinguish focal vs diffuse (only PET/genetic defines this); "
            "    5. Gene panel: ABCC8 + KCNJ11"
        ),
        "disease_pathway": (
            "ABCC8 LOF → K-ATP CHANNEL CONSTITUTIVELY CLOSED → UNREGULATED INSULIN SECRETION: "
            "  Normal: glucose ↑ → glycolysis → ATP/ADP ↑ → K-ATP closes → depolarise → "
            "    Ca²⁺ VGCCs open → [Ca²⁺]i ↑ → insulin exocytosis; "
            "  ABCC8 LOF: SUR1 absent/non-functional → K-ATP cannot open (even in hypoglycaemia); "
            "    Membrane CHRONICALLY depolarised → Ca²⁺ chronically elevated → "
            "    insulin secreted even when BG is dangerously low; "
            "  FOCAL vs DIFFUSE: "
            "    Focal: paternal LOF + somatic maternal 11p15.1 deletion in a focal beta-cell clone → "
            "      ALL K-ATP lost only in that cluster → focal unregulated insulin from focal region; "
            "    Diffuse: biallelic → all beta-cells affected → uniform CHI; "
            "  DIAZOXIDE: opens K-ATP by binding MgADP site on SUR1 NBD2 — "
            "    K-ATP absent/dysfunctional → diazoxide has nothing to open → UNRESPONSIVE"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC PATTERN: "
            "Macrosomia + neonatal hypoglycaemia + high GIR (>8 mg/kg/min) + "
            "critical sample: elevated insulin + suppressed ketones + suppressed FFAs. "
            "18F-DOPA PET-CT: ESSENTIAL — determines focal (curable with surgery) vs diffuse. "
            "DIAZOXIDE UNRESPONSIVE confirms K-ATP CHI (ABCC8 or KCNJ11). "
            "Gene panel ABCC8 + KCNJ11 confirms genetic subtype."
        ),
        "treatment": (
            "TREATMENT — ABCC8/K-ATP CHI: "
            "INITIAL STABILISATION: "
            "  IV glucose (10-15%): titrate to maintain BG >3.5 mmol/L; "
            "  Nasogastric feeds: continuous; "
            "  High GIR (8-20 mg/kg/min): via central line; "
            "MEDICAL (if diazoxide trial fails): "
            "  Octreotide SC/IV: somatostatin analogue; inhibits insulin via SST2 receptors; "
            "    4-20 µg/kg/day; tachyphylaxis develops; "
            "    RISK: necrotising enterocolitis in neonates (controversial); "
            "  Glucagon infusion: 10 µg/kg/h IV; bridging only; "
            "  Nifedipine: Ca²⁺ channel blocker — some benefit in diffuse; "
            "  mTOR inhibitors (sirolimus/everolimus): emerging for refractory diffuse; "
            "SURGICAL: "
            "  FOCAL: 18F-DOPA PET → laparoscopic partial pancreatectomy → CURATIVE (~98%); "
            "  DIFFUSE diazoxide-unresponsive: near-total (98%) pancreatectomy; "
            "    Post-op: DIABETES MELLITUS (insulin-dependent) + exocrine insufficiency (PERT) — "
            "    trading CHI for insulin-dependent DM; "
            "LONG-TERM: neurological follow-up (hypoglycaemic brain injury risk)"
        ),
        "seed": 2910,
    },
    {
        "gene": "KCNJ11",
        "protein": (
            "KCNJ11 -- 11p15.1 AR/AD -- 390aa -- Kir6.2-Inward-Rectifier-K-Channel-"
            "43kDa-K-ATP-Pore-Forming-Subunit-"
            "CHI-Diazoxide-Unresponsive-18F-DOPA-PET-Focal-Diffuse-"
            "OMIM-Gene-600937-Disease-CHI-256450"
        ),
        "locus": "11p15.1",
        "protein_size": (
            "390 aa / 43 kDa (Kir6.2 — inward rectifier potassium channel; "
            "2 transmembrane domains (M1, M2) + intracellular N + C terminus; "
            "PORE-FORMING subunit of K-ATP channel — Kir6.2 forms the K+ pore; "
            "(SUR1·Kir6.2)₄: heterooctamer — 4 Kir6.2 + 4 SUR1; "
            "ATP binds DIRECTLY to Kir6.2 intracellular domain → channel closes; "
            "KCNJ11 LOF (AR): Kir6.2 absent → K-ATP channel absent → constitutive closure → CHI; "
            "KCNJ11 GOF (AD de novo): K-ATP stays OPEN → NEONATAL DIABETES (opposite phenotype); "
            "SAME GENE: CHI = LOF (biallelic AR) vs Neonatal Diabetes = GOF (AD de novo); "
            "encoded 11p15.1 — immediately adjacent to ABCC8 on same chromosome; "
            "DEND syndrome: extreme GOF → neonatal diabetes + developmental delay + epilepsy (KCNJ11 GOF, not CHI)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — KCNJ11 CHI: "
            "  DIFFUSE CHI: biallelic AR mutations → all beta-cells → diazoxide-UNRESPONSIVE; "
            "  FOCAL CHI: paternal KCNJ11 LOF + 11p15.1 LOH → focal; 18F-DOPA PET → surgery curative; "
            "AUTOSOMAL DOMINANT (AD): "
            "  Heterozygous de novo or inherited: transient neonatal CHI, often diazoxide-responsive; "
            "  Mild hypoglycaemia; spontaneous resolution in some; "
            "PREVALENCE: 2nd most common K-ATP CHI after ABCC8; "
            "CRITICAL DISTINCTION FROM GOF: "
            "  LOF (biallelic) → CHI (hypoglycaemia) — THIS ATLAS; "
            "  GOF (de novo AD) → neonatal diabetes + DEND syndrome (Hereditary Neonatal Diabetes Atlas)"
        ),
        "disease_category": (
            "K-ATP CHI — KCNJ11 LOF: "
            "  CLINICALLY IDENTICAL TO ABCC8 CHI: "
            "    Neonatal macrosomia + hypoglycaemia + high GIR + diazoxide-UNRESPONSIVE; "
            "    Critical sample: elevated insulin + suppressed ketones + low FFAs; "
            "  FOCAL vs DIFFUSE: same 18F-DOPA PET algorithm as ABCC8; "
            "  DIFFERENTIAL (KCNJ11 vs ABCC8): "
            "    Clinical features IDENTICAL; distinguished ONLY by gene panel; "
            "    Both 11p15.1 genes on same panel; "
            "  KEY FACT: SAME gene locus as K-ATP neonatal diabetes (KCNJ11 GOF); "
            "    CHI panel tests LOF; neonatal diabetes panel tests GOF; "
            "    PHENOTYPE DICTATES DIRECTION OF TEST (hypo vs hyper glycaemia)"
        ),
        "disease_pathway": (
            "KCNJ11 LOF → K-ATP CHANNEL ABSENT → CONSTITUTIVE MEMBRANE DEPOLARISATION → CHI: "
            "  Normal K-ATP function: Kir6.2 pore opens when ATP falls (fasting/hypoglycaemia) → "
            "    K+ efflux → membrane hyperpolarises → Ca²⁺ VGCCs close → insulin secretion halted; "
            "  KCNJ11 LOF: Kir6.2 absent → no K-ATP channel → K+ cannot exit via K-ATP → "
            "    membrane cannot hyperpolarise → Ca²⁺ VGCCs cannot close → "
            "    insulin secreted CONTINUOUSLY regardless of BG; "
            "  FOCAL: as ABCC8 focal — paternal LOF + 11p15.1 LOH in focal clone; "
            "  DIAZOXIDE: acts on SUR1 (ABCC8) — Kir6.2 absent → no channel for diazoxide to open; "
            "    diazoxide INEFFECTIVE in KCNJ11 CHI (Kir6.2 absent)"
        ),
        "pathognomonic": (
            "Clinically identical to ABCC8 CHI. "
            "DISTINGUISHING TESTS: "
            "1. Critical sample: insulin elevated + ketones suppressed + FFAs suppressed. "
            "2. Diazoxide trial: UNRESPONSIVE (confirms K-ATP CHI). "
            "3. 18F-DOPA PET: focal vs diffuse. "
            "4. Gene panel (ABCC8 + KCNJ11 together): sequence both simultaneously. "
            "KEY LEARNING: same gene, OPPOSITE phenotype depending on GOF vs LOF."
        ),
        "treatment": (
            "TREATMENT: Identical to ABCC8 K-ATP CHI. "
            "IV dextrose + NG feeds → octreotide → 18F-DOPA PET. "
            "FOCAL: partial pancreatectomy → CURATIVE. "
            "DIFFUSE: near-total pancreatectomy → insulin-dependent DM + PERT. "
            "DIAZOXIDE: UNHELPFUL (do not persist with diazoxide in K-ATP CHI); "
            "  early 18F-DOPA PET + surgical referral for non-responders. "
            "NEUROLOGICAL: urgent treatment of hypoglycaemia critical — "
            "  prolonged neonatal hypoglycaemia → basal ganglia injury (kernicterus-equivalent)."
        ),
        "seed": 2911,
    },
    {
        "gene": "GLUD1",
        "protein": (
            "GLUD1 -- 10q23.3 AD-GOF -- 558aa -- Glutamate-Dehydrogenase-1-"
            "61kDa-Mitochondrial-GDH-NADP-NAD-"
            "HI-HA-Hyperinsulinism-Hyperammonaemia-Syndrome-Protein-Induced-"
            "OMIM-Gene-138130-Disease-HIHA-606762"
        ),
        "locus": "10q23.3",
        "protein_size": (
            "558 aa / 61 kDa (GDH — glutamate dehydrogenase 1; "
            "mitochondrial matrix enzyme; homohexamer (6 × 61 kDa = ~366 kDa native); "
            "REACTION: glutamate + NAD(P)⁺ → α-ketoglutarate + NH₃ + NAD(P)H; "
            "REGULATION: allosteric — INHIBITED by GTP and ATP; ACTIVATED by ADP and leucine; "
            "HADH (SCHAD) provides another inhibitory control via protein-protein interaction; "
            "FUNCTION in beta-cell: glutamate (from protein/aminoacids) → GDH → α-KG → "
            "  enters TCA → ↑ ATP/ADP → K-ATP closes → insulin; "
            "GLUD1 GOF MUTATIONS (AD): de novo in ~90%; "
            "  allosteric inhibition by GTP/ATP REDUCED → GDH constitutively MORE active → "
            "  glutamate/leucine → EXCESS ATP → K-ATP closes → insulin secreted; "
            "HYPERAMMONAEMIA: glutamate → α-KG + NH₃; excess GDH activity → more NH₃ produced → "
            "  mild-moderate plasma NH₃ elevation (not severe enough for encephalopathy usually); "
            "encoded 10q23.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION — GLUD1 HI/HA SYNDROME: "
            "  ~90% de novo (new mutation in proband); "
            "  ~10% inherited from affected parent (mild/unrecognised in parent); "
            "  PENETRANCE: high; EXPRESSIVITY: variable; "
            "  PROTEIN-INDUCED: leucine ACTIVATES GDH → protein meal or leucine load → "
            "    exaggerated insulin → post-prandial hypoglycaemia (after high-protein meal); "
            "  FASTING: also hypoglycaemia (basal GDH overactive); "
            "  HYPERAMMONAEMIA: "
            "    Plasma NH₃ mildly elevated 60–200 µmol/L (normal <50); "
            "    NOT acidotic; NOT hepatopathy; NO hyperammonaemia crisis (different from UCDs); "
            "    NH₃ does NOT need emergency treatment at these levels"
        ),
        "disease_category": (
            "HI/HA SYNDROME — HYPERINSULINISM-HYPERAMMONAEMIA: "
            "  ONSET: "
            "    Neonatal (rarely): may present in newborn period; "
            "    More commonly: infancy to childhood (post-weaning, high-protein feeds); "
            "  CLINICAL FEATURES: "
            "    Post-prandial hypoglycaemia — ESPECIALLY AFTER HIGH-PROTEIN MEALS; "
            "    Fasting hypoglycaemia; "
            "    Mild-moderate hyperammonaemia (NH₃ 60–200 µmol/L) — NO SYMPTOMS; "
            "    Normal liver function (NH₃ elevated without liver disease); "
            "    Seizures from hypoglycaemia (NOT from hyperammonaemia); "
            "  DIAGNOSTIC CLUE: "
            "    CHI + elevated NH₃ without liver disease → GLUD1 until proven otherwise; "
            "    Leucine provocation test (not recommended routinely — risk); "
            "    GLUD1 gene sequencing confirms"
        ),
        "disease_pathway": (
            "GLUD1 GOF → EXCESSIVE GDH ACTIVITY → EXCESS ATP FROM GLUTAMATE OXIDATION → CHI: "
            "  Normal: GDH regulated by GTP (allosteric inhibitor) and HADH (protein interaction); "
            "  GLUD1 GOF: mutant GDH insensitive to GTP/ATP inhibition → runs continuously → "
            "    glutamate → α-KG → TCA → EXCESS ATP even at baseline; "
            "  Beta-cell: excess ATP → K-ATP closes → membrane depolarises → Ca²⁺ → insulin; "
            "  PROTEIN/LEUCINE: leucine activates GDH directly → GOF amplified → insulin surge; "
            "  HYPERAMMONAEMIA: glutamate → α-KG + NH₃ (excess reaction products) → "
            "    NH₃ accumulates modestly; "
            "    Liver NORMALLY handles NH₃ via urea cycle — borderline overwhelmed → mild NH₃ elevation; "
            "  HADH: protein inhibitor of GDH → HADH LOF (separate gene) → same net effect"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC COMBINATION: "
            "CHI (hypoglycaemia + elevated insulin + suppressed ketones) + "
            "ELEVATED PLASMA AMMONIA (NH3 60-200 µmol/L) WITHOUT LIVER DISEASE. "
            "Protein-induced hypoglycaemia (post-high-protein meal). "
            "NH3 elevation without acidosis, without liver failure, without urea cycle defect. "
            "DIAZOXIDE-RESPONSIVE: effective in HI/HA (GDH pathway, not K-ATP blocked)."
        ),
        "treatment": (
            "TREATMENT — GLUD1 HI/HA: "
            "DIAZOXIDE: effective (K-ATP functional — not a K-ATP CHI); "
            "  dose 5–15 mg/kg/day PO in 3 divided doses; monitor for oedema, hypertrichosis; "
            "  often required long-term; "
            "DIETARY: "
            "  Low-leucine diet: reduces GDH stimulation; practical but restrictive; "
            "  Avoid high-protein meals without concurrent carbohydrate; "
            "  Continuous NG feed: for neonatal/severe cases; "
            "OCTREOTIDE: as adjunct if diazoxide insufficient; "
            "HYPERAMMONAEMIA: "
            "  NH3 60-200 µmol/L: does NOT require emergency urea cycle treatment; "
            "  DO NOT confuse with urea cycle defects (completely different mechanism); "
            "  Monitor NH3 — ensure does not exceed ~300 µmol/L; "
            "PROGNOSIS: generally good — CHI often improves with age; "
            "  learning difficulties in ~25% (due to recurrent/prolonged hypoglycaemia, not NH3)"
        ),
        "seed": 2912,
    },
    {
        "gene": "GCK",
        "protein": (
            "GCK -- 7p13 AD-GOF -- 465aa -- Glucokinase-"
            "52kDa-Hexokinase-IV-Glucose-Sensor-"
            "CHI-Variable-Severity-Diazoxide-Responsive-"
            "OMIM-Gene-138079-Disease-GCKHI-602485"
        ),
        "locus": "7p13",
        "protein_size": (
            "465 aa / 52 kDa (glucokinase / hexokinase IV; "
            "phosphorylates glucose → glucose-6-phosphate; "
            "UNIQUE KINETICS: sigmoidal curve, S0.5 ~8 mmol/L (vs hexokinase I S0.5 ~0.1 mmol/L); "
            "LOW AFFINITY = beta-cell glucose sensor: activity proportional to plasma glucose; "
            "GCK acts as threshold sensor — insulin secretion begins only when BG exceeds ~5 mmol/L; "
            "GLUCOSE SET-POINT: the BG level at which insulin secretion is half-maximal; "
            "GCK GOF (AD): left-shifts glucose set-point → insulin secretion occurs at LOWER BG → "
            "  hypoglycaemia (glucose set-point shifted down to 2-3 mmol/L in severe variants); "
            "GCK LOF (AD): right-shifts set-point → mild fasting hyperglycaemia → MODY2; "
            "MOSAIC GCK GOF: highly variable phenotype — depends on clone size; "
            "encoded 7p13; expressed in beta-cells AND liver (2 promoters)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION — GCK CHI: "
            "  Heterozygous GOF sufficient for CHI; "
            "  Variable severity: depends on degree of set-point shift; "
            "  De novo: ~50% of GCK CHI; "
            "  MOSAIC GCK GOF: somatic mosaic — parent may appear unaffected; "
            "    Variable phenotype depending on fraction of affected cells; "
            "    Deep sequencing of blood + tissues needed for diagnosis; "
            "  HOMOZYGOUS GOF (very rare): severe CHI, persistent; "
            "  EXPRESSIVITY: wide — from mild asymptomatic to severe persistent CHI; "
            "  DISTINCTION FROM MODY2: "
            "    GCK GOF → CHI (hypoglycaemia); GCK LOF → MODY2 (mild hyperglycaemia); "
            "    Same gene — OPPOSITE clinical phenotype"
        ),
        "disease_category": (
            "GCK-CHI — GLUCOKINASE GAIN-OF-FUNCTION: "
            "  ONSET: neonatal to childhood; "
            "  SEVERITY: VARIABLE — from mild asymptomatic to severe persistent neonatal CHI; "
            "  CLINICAL FEATURES: "
            "    Hypoglycaemia (fasting and post-prandial); "
            "    BG set-point shifted DOWN: patient asymptomatic at lower BG than normal; "
            "    Macrosomia (if severe, in-utero hyperinsulinism); "
            "    Often MILDER than K-ATP CHI; "
            "  BIOCHEMISTRY: "
            "    Hypoglycaemia + inappropriately elevated insulin + suppressed ketones; "
            "    BUT insulin elevation MAY BE SUBTLE (just above threshold for patient's low set-point); "
            "  DIAZOXIDE: usually RESPONSIVE; "
            "    Partial pancreatectomy rarely needed (unlike K-ATP diffuse CHI); "
            "  DIAGNOSIS: critical sample + GCK gene sequencing"
        ),
        "disease_pathway": (
            "GCK GOF → LOWERED GLUCOSE SET-POINT → INSULIN AT SUBNORMAL BG: "
            "  Normal GCK: glucose sensor — low activity at BG <5 mmol/L → "
            "    K-ATP open (low ATP/ADP) → K+ efflux → hyperpolarised → no insulin; "
            "  GCK GOF: higher enzyme activity at same BG → more glucose-6-phosphate → "
            "    more glycolysis → more ATP → K-ATP closes at LOWER BG → insulin secreted; "
            "  NET EFFECT: 'normal' regulation preserved but operating at a shifted threshold; "
            "  SET-POINT SHIFT: BG at half-maximal insulin secretion drops from ~5 to 2-3 mmol/L; "
            "  DIAZOXIDE RESPONSIVE: "
            "    K-ATP channel is PRESENT and FUNCTIONAL (not K-ATP CHI) → "
            "    diazoxide can open K-ATP → hyperpolarise → suppress insulin; "
            "  MOSAIC: only fraction of beta-cells have GOF → partial shift → variable phenotype"
        ),
        "pathognomonic": (
            "NO single pathognomonic finding. "
            "PATTERN: hypoglycaemia + elevated insulin (may be subtle) + suppressed ketones + "
            "DIAZOXIDE-RESPONSIVE + stable (mild) hypoglycaemia in many. "
            "GCK gene sequencing confirms — look for GOF variants. "
            "MOSAIC: may need deep sequencing if heterozygous not detected on standard panel. "
            "DISTINGUISH FROM MODY2 (GCK LOF): clinical direction (hypo vs hyper) is key."
        ),
        "treatment": (
            "TREATMENT — GCK CHI: "
            "MILD CASES: "
            "  Frequent feeds; avoid prolonged fasting; "
            "  BG monitoring; "
            "  Often manageable without medication; "
            "MODERATE-SEVERE: "
            "  Diazoxide (5-15 mg/kg/day) — usually RESPONSIVE; "
            "  Octreotide if diazoxide insufficient; "
            "  Continuous NG feeds in neonatal period; "
            "SURGICAL: rarely needed (unlike K-ATP CHI); "
            "  Partial pancreatectomy only if truly refractory; "
            "  18F-DOPA PET NOT required (GCK CHI is diffuse by mechanism — no focal lesion); "
            "PROGNOSIS: variable — "
            "  Mild variants: improve with age or managed with diet; "
            "  Severe GOF: persistent CHI; "
            "LONG-TERM: monitor for MODY2-like hyperglycaemia in adulthood "
            "  (rare — set-point shift may benefit from no treatment as glucose rises with age)"
        ),
        "seed": 2913,
    },
    {
        "gene": "HADH",
        "protein": (
            "HADH -- 4q25 AR -- 314aa -- 3-Hydroxyacyl-CoA-Dehydrogenase-Short-Chain-SCHAD-"
            "34kDa-Mitochondrial-FAD-NADH-"
            "Protein-Induced-CHI-3-OH-Glutaric-Aciduria-PATHOGNOMONIC-Diazoxide-Responsive-"
            "OMIM-Gene-601609-Disease-HADH-CHI-609975"
        ),
        "locus": "4q25",
        "protein_size": (
            "314 aa / 34 kDa (SCHAD — short-chain L-3-hydroxyacyl-CoA dehydrogenase; "
            "mitochondrial matrix enzyme; homotetramer; "
            "ROLE IN BETA-OXIDATION: oxidises 3-hydroxyacyl-CoA → 3-ketoacyl-CoA (short chain); "
            "UNIQUE BETA-CELL FUNCTION: SCHAD directly INHIBITS GDH (GLUD1) via protein-protein interaction; "
            "SCHAD LOF → GDH DISINHIBITED → same net effect as GLUD1 GOF; "
            "BIOMARKER: HADH LOF → 3-OH-glutaric acid (3-OHG) accumulates in urine; "
            "  3-OHG elevated in urine = PATHOGNOMONIC for HADH CHI; "
            "  Also: 3-OH-glutaryl-carnitine (C5-OH) on acylcarnitine profile — elevated; "
            "encoded 4q25; consanguineous families common (AR)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — HADH/SCHAD CHI: "
            "  Both alleles required; 25% sibling recurrence; "
            "  Consanguineous families prevalent in reported cases; "
            "  FOUNDER: no major founder variant described; "
            "  PHENOTYPE: protein-induced hypoglycaemia + urine 3-OH-glutaric acid elevated; "
            "  GOOD PROGNOSIS: diazoxide-responsive; "
            "    Some achieve diazoxide-free remission by childhood; "
            "  PREVALENCE: uncommon — but important because diazoxide-responsive and surgically avoidable"
        ),
        "disease_category": (
            "HADH/SCHAD CHI — PROTEIN-INDUCED HYPERINSULINISM: "
            "  ONSET: neonatal to early infancy; "
            "  CLINICAL: "
            "    Protein-induced hypoglycaemia (amino acid feeds → leucine → GDH disinhibited → ATP → insulin); "
            "    Fasting hypoglycaemia also present; "
            "    Less severe than K-ATP CHI; "
            "    RESPONDS TO DIAZOXIDE; "
            "  BIOCHEMICAL CLUE: "
            "    Urine organic acids: 3-OH-GLUTARIC ACID elevated (PATHOGNOMONIC); "
            "    Plasma acylcarnitine: C5-OH (3-OH-glutaryl-carnitine) elevated; "
            "    Ammonia: can be mildly elevated (GDH disinhibited → same NH3 mechanism as GLUD1 GOF); "
            "  DIFFERENTIAL FROM GLUD1 GOF: "
            "    Both: protein-induced CHI + mild hyperammonaemia; "
            "    HADH: urine 3-OHG elevated (absent in GLUD1 GOF); "
            "    Gene sequencing distinguishes"
        ),
        "disease_pathway": (
            "HADH LOF → GDH DISINHIBITED → EXCESS NH3 + ATP FROM GLUTAMATE → CHI: "
            "  Normal: SCHAD protein binds GDH → allosteric inhibition → GDH activity controlled; "
            "  HADH LOF: SCHAD absent → GDH unbound → constitutively MORE active; "
            "  Mechanism then IDENTICAL to GLUD1 GOF: "
            "    glutamate + leucine → GDH overactive → α-KG + NH3; "
            "    α-KG → TCA → excess ATP → K-ATP closes → insulin; "
            "  BIOMARKER GENERATION: "
            "    HADH LOF → short-chain beta-oxidation impaired → "
            "    3-hydroxy-fatty acids accumulate → metabolic conversion → 3-OH-glutaric acid; "
            "    3-OHG excreted in urine → diagnostic biomarker; "
            "  DIAZOXIDE RESPONSIVE: K-ATP channel INTACT (not K-ATP CHI)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: Urine 3-OH-GLUTARIC ACID (3-OHG) elevated in a patient with CHI. "
            "Also: plasma C5-OH acylcarnitine elevated on NBS acylcarnitine profile. "
            "Confirm with HADH gene sequencing. "
            "PROTEIN-INDUCED HYPOGLYCAEMIA: timing post high-protein feed is a clue. "
            "DIAZOXIDE-RESPONSIVE: distinguishes from K-ATP CHI — do not pancreatectomise HADH CHI patients."
        ),
        "treatment": (
            "TREATMENT — HADH CHI: "
            "DIAZOXIDE: effective; 5-15 mg/kg/day; "
            "  Monitor oedema, hypertrichosis; chlorothiazide co-prescription to reduce fluid retention; "
            "DIETARY: "
            "  Avoid high-protein bolus feeds; balance protein with carbohydrate; "
            "  Continuous NG feeds for severe neonatal cases; "
            "OCTREOTIDE: adjunct if diazoxide insufficient; "
            "SURGICAL: NOT indicated — diazoxide-responsive; "
            "  18F-DOPA PET NOT required; "
            "PROGNOSIS: EXCELLENT — "
            "  Most achieve remission (diazoxide-free) by 5-10 years; "
            "  Minimal long-term metabolic consequences; "
            "BIOMARKER MONITORING: urine 3-OHG normalises on treatment — useful response marker. "
            "FAMILY CASCADE: AR — test siblings + carrier testing for parents."
        ),
        "seed": 2914,
    },
    {
        "gene": "HNF4A",
        "protein": (
            "HNF4A -- 20q13.12 AD-LOF -- 474aa -- Hepatocyte-Nuclear-Factor-4-Alpha-"
            "53kDa-Nuclear-Receptor-RXR-Dimerisation-"
            "Macrosomia-Neonatal-CHI-Transitions-MODY1-Adulthood-Diazoxide-Responsive-"
            "OMIM-Gene-600281-Disease-MODY1-125850"
        ),
        "locus": "20q13.12",
        "protein_size": (
            "474 aa / 53 kDa (HNF4A — hepatocyte nuclear factor 4 alpha; "
            "nuclear receptor superfamily (NR2A1); Zn-finger DBD + ligand-binding domain; "
            "LIGANDS: fatty acids (HNF4A is constitutively active — fatty acids stabilise); "
            "DIMERISATION: homodimer; "
            "FUNCTION IN PANCREAS: master transcription factor for beta-cell gene expression — "
            "  drives expression of ABCC8 (SUR1), KCNJ11 (Kir6.2), insulin, glucokinase; "
            "HNF4A LOF → beta-cell transcription impaired → PARADOX: "
            "  Neonatal: K-ATP subunit expression REDUCED → K-ATP channels insufficient → "
            "    membrane PARTIALLY depolarised → CHI (same mechanism as K-ATP LOF); "
            "  Later childhood/adult: beta-cell function decreases → MODY1 (hyperglycaemia); "
            "BIPHASIC PHENOTYPE: CHI in neonate → MODY1 in adult — SAME gene; "
            "MACROSOMIA: in-utero HNF4A haploinsufficiency → beta-cell dysregulation in fetus; "
            "encoded 20q13.12"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — LOSS-OF-FUNCTION — HNF4A: "
            "  Heterozygous sufficient; "
            "  PENETRANCE: high for MODY1 in adults; moderate for neonatal CHI; "
            "  EXPRESSIVITY: variable — some HNF4A heterozygotes have overt neonatal CHI, "
            "    others only macrosomia, others only later MODY1; "
            "  MACROSOMIA: ~56% of HNF4A LOF neonates are LGA (large for gestational age); "
            "  NEONATAL CHI: ~15-20% have overt hypoglycaemia requiring treatment; "
            "  MODY1 (adult): beta-cell function declines → hyperglycaemia typically 20s-40s; "
            "  FAMILY HISTORY: parent with MODY1 diabetes + new infant with macrosomia/hypoglycaemia → "
            "    HNF4A until proven otherwise; "
            "  DISTINCTION FROM HNF1A (MODY3): different gene, same beta-cell TF family"
        ),
        "disease_category": (
            "BIPHASIC HNF4A DISEASE — NEONATAL CHI → ADULT MODY1: "
            "  NEONATAL PHASE: "
            "    Macrosomia (LGA; birth weight often >4 kg); "
            "    Neonatal hypoglycaemia (CHI): diazoxide-RESPONSIVE; "
            "    High GIR in first days; "
            "    Usually RESOLVES within weeks to months; "
            "  PAEDIATRIC/ADOLESCENT: "
            "    Normal glucose tolerance; asymptomatic interval; "
            "  ADULT PHASE (MODY1): "
            "    Progressive beta-cell failure → fasting hyperglycaemia → T2DM-like; "
            "    Responds to sulfonylureas (SU treatment) — INSULIN NOT USUALLY FIRST-LINE; "
            "    Elevated alanine aminotransferase (HNF4A expressed in liver — hepatocyte function); "
            "  KEY DIAGNOSTIC CLUE: "
            "    Family with MODY1 diabetes + macrosomic neonate → check HNF4A; "
            "    Or: neonatal CHI in otherwise healthy full-term macrosomic infant → HNF4A"
        ),
        "disease_pathway": (
            "HNF4A LOF → REDUCED K-ATP SUBUNIT EXPRESSION → PARADOXICAL CHI IN NEONATAL PERIOD: "
            "  HNF4A normally: transcribes ABCC8 (SUR1), KCNJ11 (Kir6.2), and other beta-cell genes; "
            "  HNF4A LOF (1 allele): ABCC8/KCNJ11 reduced → fewer K-ATP channels per beta-cell; "
            "  NEONATAL: reduced K-ATP → less K+ efflux at hypoglycaemia → "
            "    membrane less hyperpolarised → some Ca²⁺ channels remain open → "
            "    inappropriate insulin at low BG → CHI; "
            "  LATER ADULT: HNF4A also drives overall beta-cell differentiation/survival → "
            "    haploinsufficiency → progressive beta-cell loss → MODY1; "
            "  BIPHASIC MECHANISM: same LOF → paradoxically CHI in neonate, then DM in adult; "
            "  LIVER: HNF4A expressed in hepatocytes → alanine aminotransferase (ALT) mildly elevated"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC COMBINATION: "
            "Macrosomia (LGA) + neonatal CHI (diazoxide-responsive) + FAMILY HISTORY OF MODY1 DIABETES. "
            "OR: macrosomic neonate + neonatal hypoglycaemia + ALT mildly elevated. "
            "HNF4A gene sequencing confirms. "
            "MODY1-CHI duality: the same HNF4A variant that causes MODY1 in adults causes CHI in newborns."
        ),
        "treatment": (
            "TREATMENT — HNF4A CHI: "
            "NEONATAL CHI: "
            "  Diazoxide (5-15 mg/kg/day): RESPONSIVE — key discriminating response; "
            "  Continue until spontaneous resolution (weeks to months); "
            "  Wean diazoxide under BG monitoring; "
            "  NG feeds + IV dextrose for severe cases; "
            "  SURGICAL NOT REQUIRED; "
            "ADULT MODY1 TRANSITION: "
            "  Monitor BG annually from adolescence; "
            "  Sulfonylurea (gliclazide/glipizide) — effective; "
            "  Insulin: if SU fails; "
            "GENETIC COUNSELLING: "
            "  AD — 50% transmission; "
            "  Family members with apparent T2DM → test for MODY1 (HNF4A); "
            "  Macrosomic neonates in HNF4A families → prepare for CHI. "
            "HEPATIC: ALT monitoring; generally mild and self-limiting."
        ),
        "seed": 2915,
    },
    {
        "gene": "UCP2",
        "protein": (
            "UCP2 -- 11q13.4 AD-GOF -- 309aa -- Uncoupling-Protein-2-"
            "34kDa-Inner-Mitochondrial-Membrane-"
            "Mild-CHI-Spontaneous-Resolution-Diazoxide-Responsive-"
            "OMIM-Gene-601693-Disease-CHI-UCP2-612101"
        ),
        "locus": "11q13.4",
        "protein_size": (
            "309 aa / 34 kDa (UCP2 — uncoupling protein 2; "
            "inner mitochondrial membrane; 3 tandem repeats of ~100 aa; "
            "FAMILY: UCP1 (thermogenesis in brown adipose tissue) → UCP2 (widely expressed); "
            "FUNCTION: proton leak across IMM — UNCOUPLES electron transport from ATP synthesis → "
            "  dissipates proton gradient as heat instead of ATP; "
            "NORMAL BETA-CELL ROLE: UCP2 reduces ATP:ADP ratio → keeps K-ATP partially open → "
            "  dampens insulin secretion (negative regulator); "
            "UCP2 GOF (AD): increased proton leak → MORE ATP dissipated → LESS ATP produced from glucose → "
            "  BUT PARADOX: at low glucose, still enough 'excess' uncoupling relative to K-ATP threshold → "
            "    mild CHI; "
            "ACTUALLY: UCP2 overexpression in beta-cells INHIBITS insulin (GOF → diabetes in mice); "
            "HUMAN UCP2 GOF MUTATIONS: rare; the P228L and related variants reduce stability → "
            "  net effect: INCREASED UCP2 activity in some contexts → reduced ATP → CHI??; "
            "MECHANISM DEBATED: likely GOF mutations affect protein stability/feedback → "
            "  net REDUCED ATP suppression of K-ATP → K-ATP stays open → CHI; "
            "encoded 11q13.4; SPONTANEOUS RESOLUTION typical"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION — UCP2 CHI: "
            "  Very rare; few pedigrees described; "
            "  Heterozygous GOF variants; "
            "  De novo and familial both reported; "
            "  PENETRANCE: incomplete — some family members unaffected; "
            "  MILD PHENOTYPE: hypoglycaemia often mild and transient; "
            "  SPONTANEOUS RESOLUTION: by 3-6 years in most; "
            "  DIAZOXIDE-RESPONSIVE: K-ATP present and functional; "
            "  PREVALENCE: rare — but clinically important as surgical CHI avoided; "
            "  UCP2 GOF may be underdiagnosed (mild CHI, resolves, gene sequencing not done)"
        ),
        "disease_category": (
            "UCP2 GOF CHI — MILD TRANSIENT HYPERINSULINISM: "
            "  ONSET: neonatal to early infancy; "
            "  CLINICAL: "
            "    Mild hypoglycaemia (BG rarely <1.5 mmol/L; often 2.0-3.0 mmol/L); "
            "    No macrosomia typically; "
            "    Seizures: uncommon (mild hypoglycaemia); "
            "    Feeding-responsive hypoglycaemia; "
            "  BIOCHEMISTRY: "
            "    Mildly elevated insulin; mildly suppressed ketones; "
            "    GIR requirement modest (4-8 mg/kg/min); "
            "  COURSE: "
            "    SPONTANEOUS RESOLUTION: majority by 3-6 years of age; "
            "  MANAGEMENT: "
            "    Diazoxide-responsive; "
            "    Often managed with frequent feeds alone in mild cases; "
            "  DIAGNOSIS: "
            "    Gene sequencing (UCP2) — after K-ATP CHI excluded"
        ),
        "disease_pathway": (
            "UCP2 GOF → ALTERED MITOCHONDRIAL ATP PRODUCTION → MILD INSULIN DYSREGULATION: "
            "  Normal UCP2: proton leak dampens beta-cell ATP → K-ATP partially open → "
            "    raises threshold for insulin secretion (negative regulator of CHI); "
            "  UCP2 GOF: gain of uncoupling activity → "
            "    EXPECTED: more ATP dissipated → LESS insulin → diabetes; "
            "  PARADOX IN HUMANS: reported GOF variants appear to cause CHI by: "
            "    altered protein-protein interaction with mitofilin/IMM components → "
            "    disrupts feedback regulation → net K-ATP channel closure tendency at lower BG; "
            "  MECHANISM INCOMPLETELY UNDERSTOOD: "
            "    Some authors propose GOF variants actually reduce UCP2 stability at the IMM → "
            "    LESS uncoupling → MORE ATP → K-ATP closes → CHI; "
            "  NET EFFECT: K-ATP closes at mild hypoglycaemia → insulin secreted → maintains cycle"
        ),
        "pathognomonic": (
            "NO single pathognomonic finding. "
            "PATTERN: mild neonatal/infantile CHI + diazoxide-RESPONSIVE + spontaneous resolution. "
            "K-ATP CHI excluded (ABCC8/KCNJ11 negative). "
            "UCP2 gene sequencing confirms. "
            "CLINICAL CLUE: milder than K-ATP CHI, resolves by school age without surgery."
        ),
        "treatment": (
            "TREATMENT — UCP2 CHI: "
            "MILD CASES: "
            "  Frequent feeds (2-3 hourly); avoid prolonged fasting; "
            "  Monitor BG; no medication needed if feeds maintain BG >3.5 mmol/L; "
            "MODERATE: "
            "  Diazoxide (5-10 mg/kg/day): effective; "
            "  Wean gradually as hypoglycaemia improves with age; "
            "  Target: discontinue by 3-6 years; "
            "OCTREOTIDE: rarely needed; "
            "SURGICAL: NOT INDICATED; "
            "PROGNOSIS: EXCELLENT — "
            "  Spontaneous resolution in majority; "
            "  No long-term metabolic consequences after resolution; "
            "  Normal glucose tolerance in childhood/adulthood; "
            "FAMILY COUNSELLING: AD — test first-degree relatives; "
            "  unaffected parents with UCP2 variant → low penetrance."
        ),
        "seed": 2916,
    },
    {
        "gene": "SLC16A1",
        "protein": (
            "SLC16A1 -- 17q24.2 AD-GOF -- 478aa -- Monocarboxylate-Transporter-1-MCT1-"
            "52kDa-12-TM-Proton-Symporter-"
            "Exercise-Induced-Hypoglycaemia-EIHI-Pyruvate-Beta-Cell-Promoter-GOF-"
            "OMIM-Gene-600682-Disease-EIHI-606370"
        ),
        "locus": "17q24.2",
        "protein_size": (
            "478 aa / 52 kDa (MCT1 — monocarboxylate transporter 1; "
            "solute carrier family 16A; 12 transmembrane domains; "
            "TRANSPORT: lactate, pyruvate, short-chain fatty acids (COTRANSPORTED with H+); "
            "NORMAL BETA-CELL: MCT1 ABSENT from beta-cells (UNIQUE); "
            "  WHY: if pyruvate entered beta-cells, exercise would always cause insulin → hypoglycaemia; "
            "  Pyruvate/lactate from working muscle → systemic circulation → "
            "    normally EXCLUDED from beta-cells → NO exercise-related insulin; "
            "SLC16A1 PROMOTER GOF MUTATION: "
            "  SLC16A1 promoter gains a new transcription factor binding site → "
            "  MCT1 ECTOPICALLY EXPRESSED in beta-cells; "
            "  Exercise → muscle releases pyruvate/lactate → enters beta-cell via MCT1 → "
            "    mitochondrial oxidation → ATP ↑ → K-ATP closes → insulin → hypoglycaemia; "
            "EXON-SEQUENCING MISSES PROMOTER: must sequence promoter region specifically; "
            "encoded 17q24.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION (PROMOTER) — SLC16A1 EIHI: "
            "  Heterozygous promoter GOF mutations; "
            "  Standard exon-sequencing MISSES these — promoter sequencing required; "
            "  De novo and familial; "
            "  EXPRESSIVITY: variable — severity of EIHI varies; "
            "  PENETRANCE: high for EIHI phenotype; "
            "  DISTINCT FROM OTHER CHI: hypoglycaemia ONLY with exercise; "
            "    fasting hypoglycaemia ABSENT or minimal; "
            "  PREVALENCE: rare but likely underdiagnosed (promoter mutations missed on standard panels); "
            "  CLINICAL CLUE: athlete or active child with post-exercise seizures → EIHI"
        ),
        "disease_category": (
            "EIHI — EXERCISE-INDUCED HYPERINSULINISM — UNIQUE MECHANISM: "
            "  ONSET: childhood to adolescence (when exercise becomes more vigorous); "
            "  CLINICAL FEATURES: "
            "    Hypoglycaemia ONLY during/immediately after anaerobic exercise; "
            "    FASTING: glucose NORMAL (MCT1 not relevant to fasting); "
            "    POST-EXERCISE (5-30 min after): hypoglycaemia → dizziness, confusion, seizure; "
            "    Anaerobic exercise (sprinting, resistance training): highest pyruvate release; "
            "    Aerobic (low intensity): less risk; "
            "  BIOCHEMISTRY (during exercise): "
            "    Insulin elevated (post-exercise, inappropriate); "
            "    Ketones: suppressed; "
            "  CRITICAL DISTINCTION: "
            "    ALL OTHER CHI: fasting ± post-prandial hypoglycaemia; "
            "    EIHI: exercise-specific — fasting screen NORMAL; "
            "  DIAGNOSTIC PITFALL: fasting study (standard CHI workup) NEGATIVE → "
            "    EIHI missed unless exercise provocation performed"
        ),
        "disease_pathway": (
            "SLC16A1 PROMOTER GOF → ECTOPIC MCT1 IN BETA-CELLS → EXERCISE PYRUVATE → INSULIN: "
            "  Normal beta-cells: MCT1 absent → pyruvate CANNOT enter beta-cell → "
            "    exercise does not trigger insulin → no exercise hypoglycaemia; "
            "  SLC16A1 PROMOTER GOF: new transcription factor binding site (e.g. Sp1/ETS) in SLC16A1 5'UTR → "
            "    MCT1 ectopically expressed in beta-cells; "
            "  EXERCISE: anaerobic muscle → glycolysis → pyruvate release into blood → "
            "    pyruvate + H+ enter beta-cell via MCT1 → mitochondrial oxidation → acetyl-CoA → "
            "    TCA → ATP ↑ → K-ATP closes → membrane depolarises → Ca²⁺ → insulin surge; "
            "  FASTING: no pyruvate in blood → MCT1 in beta-cell has no substrate → "
            "    K-ATP unaffected → NO fasting hypoglycaemia; "
            "  EXON SEQUENCING: completely NORMAL — mutation is in PROMOTER (untranslated)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC PATTERN: "
            "Exercise-induced hypoglycaemia (post-anaerobic exercise 5-30 min) + "
            "NORMAL fasting glucose + NORMAL fasting study. "
            "Diazoxide: NOT helpful (pyruvate-driven, not K-ATP channel defect per se). "
            "DIAGNOSIS: PROMOTER SEQUENCING of SLC16A1 (NOT exon sequencing). "
            "Provocation: supervised anaerobic exercise test with BG monitoring confirms diagnosis. "
            "KEY CLINICAL CLUE: athlete with post-exercise seizures — ALWAYS consider EIHI."
        ),
        "treatment": (
            "TREATMENT — SLC16A1 EIHI: "
            "NON-PHARMACOLOGICAL (PREFERRED): "
            "  Avoid anaerobic exercise; switch to low-intensity aerobic exercise; "
            "  Pre-exercise carbohydrate loading (30-60 g complex carbohydrate before exercise); "
            "  Rapid-acting glucose (glucose gel/sports drink) immediately post-exercise; "
            "  Medical alert — particularly for competitive athletes; "
            "PHARMACOLOGICAL: "
            "  Diazoxide: UNHELPFUL (mechanism not K-ATP); "
            "  Octreotide: may reduce insulin post-exercise (some benefit); "
            "  D-lactic acid supplementation: competes with pyruvate for MCT1 (experimental); "
            "SURGICAL: NOT indicated; "
            "CAREER/ACTIVITY COUNSELLING: "
            "  Elite competitive sports (anaerobic): HIGH RISK — discuss with patient/family; "
            "  Swimming/diving alone: CONTRAINDICATED (unconsciousness in water = fatal); "
            "  Medical alert bracelet mandatory; "
            "  GLUCAGON KIT: always carry (for post-exercise hypoglycaemia emergency). "
            "GENETIC COUNSELLING: AD promoter mutation — 50% inheritance."
        ),
        "seed": 2917,
    },
]


def _rng(seed):
    return random.Random(seed)


def _generate_patients(entry):
    rng = _rng(entry["seed"])
    gene = entry["gene"]
    n = 40
    patients = []
    for i in range(n):
        # Age at diagnosis in days (neonatal) or months
        if gene in ("ABCC8", "KCNJ11"):
            age_days = rng.randint(0, 7)  # first week of life
        elif gene in ("GLUD1", "HADH"):
            age_days = rng.randint(0, 90)  # neonatal to early infancy
        elif gene in ("GCK", "HNF4A"):
            age_days = rng.randint(0, 30)  # neonatal, macrosomia
        elif gene == "UCP2":
            age_days = rng.randint(0, 14)  # neonatal
        else:  # SLC16A1
            age_days = rng.randint(365 * 5, 365 * 15)  # school age to adolescence (days)
        sex = rng.choice(["M", "F"])
        bw_g = round(rng.uniform(2500, 5000)) if gene in ("ABCC8", "KCNJ11", "HNF4A", "GCK") else round(rng.uniform(2500, 4200))
        macrosomia = bw_g > 4000
        bgl_nadir = round(rng.uniform(0.8, 2.8), 1)  # mmol/L — hypoglycaemia
        insulin_mu = round(rng.uniform(3, 25), 1)  # inappropriately elevated
        gir_mgkgmin = round(rng.uniform(4, 20), 1)  # glucose infusion rate
        diazoxide_responsive = gene not in ("ABCC8", "KCNJ11", "SLC16A1")
        focal_lesion = gene in ("ABCC8", "KCNJ11") and rng.random() < 0.4  # ~40% focal
        ammonia_elevated = gene in ("GLUD1", "HADH") and rng.random() < 0.8
        exercise_induced = gene == "SLC16A1"
        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "age_diagnosis_days": age_days,
            "sex": sex,
            "birth_weight_g": bw_g,
            "macrosomia": macrosomia,
            "bgl_nadir_mmol_l": bgl_nadir,
            "insulin_mu_l": insulin_mu,
            "gir_mg_kg_min": gir_mgkgmin,
            "diazoxide_responsive": diazoxide_responsive,
            "focal_lesion": focal_lesion,
            "ammonia_elevated": ammonia_elevated,
            "exercise_induced": exercise_induced,
        })
    return patients


def generate_overview():
    genes = [g["gene"] for g in ATLAS_GENES]
    total_patients = 0
    gene_rows = []
    for entry in ATLAS_GENES:
        pts = _generate_patients(entry)
        total_patients += len(pts)
        avg_bgl = round(sum(p["bgl_nadir_mmol_l"] for p in pts) / len(pts), 2)
        avg_gir = round(sum(p["gir_mg_kg_min"] for p in pts) / len(pts), 1)
        diaz_n = sum(1 for p in pts if p["diazoxide_responsive"])
        macr_n = sum(1 for p in pts if p["macrosomia"])
        gene_rows.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_summary": entry["protein"],
            "patients": len(pts),
            "avg_bgl_nadir_mmol_l": avg_bgl,
            "avg_gir_mg_kg_min": avg_gir,
            "diazoxide_responsive_pct": round(100 * diaz_n / len(pts)),
            "macrosomia_pct": round(100 * macr_n / len(pts)),
        })
    return {
        "atlas": "Hereditary Congenital Hyperinsulinism Atlas",
        "subtitle": "8-Gene Reference: ABCC8-KCNJ11-GLUD1-GCK-HADH-HNF4A-UCP2-SLC16A1",
        "description": (
            "Comprehensive atlas of hereditary congenital hyperinsulinism (CHI), "
            "covering the eight major genetic causes: "
            "ABCC8 (SUR1 — K-ATP CHI, most common ~65%, 18F-DOPA PET focal/diffuse), "
            "KCNJ11 (Kir6.2 — K-ATP CHI, pore subunit, same locus as neonatal diabetes GOF), "
            "GLUD1 (GDH — HI/HA syndrome, protein-induced, NH3 elevated PATHOGNOMONIC), "
            "GCK (glucokinase GOF — lowered glucose set-point, variable, diazoxide-responsive), "
            "HADH/SCHAD (3-OH-glutaric aciduria PATHOGNOMONIC, protein-induced, diazoxide-responsive), "
            "HNF4A (macrosomia + neonatal CHI → MODY1 adulthood — biphasic), "
            "UCP2 (mild CHI, spontaneous resolution), "
            "SLC16A1/MCT1 (exercise-induced EIHI — PROMOTER mutation, fasting normal). "
            "320 patients (8 × 40), seeds 2910-2917."
        ),
        "total_patients": total_patients,
        "total_genes": len(genes),
        "genes": genes,
        "gene_rows": gene_rows,
        "categories": {
            "K-ATP channel (diazoxide-unresponsive)": ["ABCC8", "KCNJ11"],
            "GDH pathway (protein-induced, diazoxide-responsive)": ["GLUD1", "HADH"],
            "Glucose sensor / transcription factor": ["GCK", "HNF4A"],
            "Mitochondrial uncoupling (mild/transient)": ["UCP2"],
            "Exercise-induced (promoter GOF, fasting normal)": ["SLC16A1"],
        },
        "key_facts": [
            "ABCC8/KCNJ11 K-ATP CHI: diazoxide-UNRESPONSIVE — 18F-DOPA PET mandatory to distinguish focal (curable surgery) from diffuse",
            "GLUD1 HI/HA: CHI + mildly elevated NH3 (60-200 µmol/L) + protein-induced = GLUD1 until proven otherwise",
            "HADH: urine 3-OH-GLUTARIC ACID elevated PATHOGNOMONIC — diazoxide-responsive, no surgery needed",
            "HNF4A: macrosomia + neonatal CHI (diazoxide-responsive) → MODY1 diabetes in adulthood — biphasic",
            "UCP2: mild CHI, spontaneous resolution by 3-6 years — avoid unnecessary pancreatectomy",
            "SLC16A1 EIHI: post-exercise hypoglycaemia, NORMAL fasting — promoter mutation (exon sequencing MISSES IT)",
            "Critical sample: insulin elevated + ketones suppressed + FFAs suppressed = hyperinsulinism confirmed",
            "K-ATP CHI focal: 18F-DOPA PET → partial pancreatectomy CURATIVE (~98% success)",
        ],
        "diagnostic_algorithm": (
            "CHI workup: "
            "1. Critical sample: glucose, insulin, ketones, FFAs, C-peptide, NH3; "
            "   Confirm hyperinsulinism: insulin >2 mU/L + BG <3.0 mmol/L + suppressed ketones; "
            "2. Diazoxide trial (5-15 mg/kg/day for 5 days): "
            "   RESPONSIVE → GLUD1, GCK, HADH, HNF4A, UCP2 subtypes likely; "
            "   UNRESPONSIVE → K-ATP CHI (ABCC8, KCNJ11) → proceed to 18F-DOPA PET; "
            "3. If diazoxide-responsive: "
            "   NH3 elevated → GLUD1 or HADH (urine 3-OHG distinguishes); "
            "   Macrosomia + family MODY1 → HNF4A; "
            "   Mild/transient → UCP2 or GCK; "
            "4. If exercise-induced only (fasting normal): "
            "   EIHI → SLC16A1 PROMOTER sequencing (exon panel misses it); "
            "5. 18F-DOPA PET (K-ATP CHI, diazoxide-unresponsive): "
            "   Focal → partial pancreatectomy (curative); "
            "   Diffuse → near-total pancreatectomy (last resort); "
            "6. Gene panel: ABCC8 + KCNJ11 + GLUD1 + GCK + HADH + HNF4A + UCP2 + SLC16A1 (with promoter)"
        ),
    }


def generate_breakdown():
    result = []
    for entry in ATLAS_GENES:
        pts = _generate_patients(entry)
        result.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein": entry["protein"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "patient_count": len(pts),
            "seed": entry["seed"],
        })
    return {"genes": result, "count": len(result)}


def generate_definitions():
    return {
        "definitions": [
            {
                "term": "Congenital Hyperinsulinism (CHI) — Critical Sample",
                "definition": (
                    "CHI: inappropriately elevated insulin during hypoglycaemia → "
                    "failure of normal counter-regulatory insulin suppression. "
                    "CRITICAL SAMPLE (drawn when BG <3.0 mmol/L): "
                    "  Insulin: >2 mU/L (some use >1 µU/mL) confirms hyperinsulinism; "
                    "  Ketones: SUPPRESSED (hyperinsulinism is anti-ketogenic — key distinguisher); "
                    "  FFAs: SUPPRESSED (insulin suppresses lipolysis); "
                    "  C-peptide: elevated (confirms endogenous insulin excess); "
                    "  NH3: elevated only in GLUD1 and HADH CHI. "
                    "GLUCAGON STIMULATION: 0.3 mg IM → BG rise >1.5 mmol/L confirms hyperinsulinism "
                    "(glycogen mobilised by glucagon despite insulin-driven hypoglycaemia). "
                    "GIR >8 mg/kg/min required to maintain BG >3.5 mmol/L = hyperinsulinism until proven otherwise."
                ),
            },
            {
                "term": "K-ATP Channel CHI (ABCC8/KCNJ11) — 18F-DOPA PET and Surgical Decision",
                "definition": (
                    "K-ATP CHI (ABCC8/KCNJ11 LOF): diazoxide-UNRESPONSIVE. "
                    "18F-DOPA PET-CT: "
                    "  FOCAL: increased 18F-DOPA uptake in one region → focal adenomatous hyperplasia; "
                    "    Mechanism: paternal K-ATP LOF + somatic maternal 11p15.1 LOH in focal clone; "
                    "    Treatment: laparoscopic partial pancreatectomy → CURATIVE ~98%; "
                    "    No long-term diabetes after focal resection (90-95% preservation). "
                    "  DIFFUSE: uniform 18F-DOPA uptake → all beta-cells affected; "
                    "    Treatment: near-total (98%) pancreatectomy → insulin-dependent DM + PERT; "
                    "    ACCEPT insulin-dependent DM to prevent brain injury from recurrent hypoglycaemia. "
                    "18F-DOPA PET MUST be performed BEFORE surgery — clinical/genetic features alone "
                    "cannot distinguish focal from diffuse."
                ),
            },
            {
                "term": "HI/HA Syndrome (GLUD1) — Differentiating from Urea Cycle Defects",
                "definition": (
                    "GLUD1 GOF → GDH hyperactive → excess glutamate oxidation → NH3 + ATP → CHI + mild hyperammonaemia. "
                    "NH3 60-200 µmol/L: elevated BUT: "
                    "  NOT a urea cycle defect (no citrulline/argininosuccinate elevation on amino acids); "
                    "  NOT liver disease (normal liver function); "
                    "  NOT organic acidaemia (normal urine organic acids — no 3-OHG); "
                    "  NOT hyperammonaemia crisis (NH3 rarely exceeds 200 µmol/L); "
                    "PROTEIN-INDUCED: high-protein meal → leucine → activates GDH further → insulin surge. "
                    "TREATMENT: diazoxide-RESPONSIVE + low-leucine diet. "
                    "DO NOT treat NH3 with sodium benzoate/phenylbutyrate (urea cycle protocol) — wrong diagnosis."
                ),
            },
            {
                "term": "HADH/SCHAD CHI — Urine 3-OH-Glutaric Acid Biomarker",
                "definition": (
                    "HADH (SCHAD) LOF → GDH disinhibited (SCHAD normally inhibits GDH by protein interaction) → "
                    "same pathway as GLUD1 GOF. "
                    "UNIQUE BIOMARKER: urine 3-OH-glutaric acid (3-OHG) elevated — "
                    "  secondary to short-chain beta-oxidation impairment. "
                    "  Also: C5-OH acylcarnitine elevated on NBS newborn screening. "
                    "DISTINGUISHES HADH FROM GLUD1: both have protein-induced CHI + mild NH3 + "
                    "  diazoxide-responsive; HADH additionally has 3-OHG in urine. "
                    "MANAGEMENT: diazoxide + avoid high-protein bolus; prognosis EXCELLENT; "
                    "  surgical pancreatectomy NOT needed. "
                    "TEST: urine organic acids + acylcarnitine profile on every CHI workup."
                ),
            },
            {
                "term": "HNF4A — Biphasic Phenotype: Neonatal CHI then Adult MODY1",
                "definition": (
                    "HNF4A LOF (AD): SAME variant → neonatal CHI + adult MODY1. "
                    "NEONATAL: HNF4A drives ABCC8/KCNJ11 expression; haploinsufficiency → fewer K-ATP → "
                    "  mild K-ATP-like CHI; macrosomia (in-utero hyperinsulinism); "
                    "  diazoxide-RESPONSIVE; resolves weeks-months. "
                    "ADULTHOOD (MODY1): HNF4A drives beta-cell survival/differentiation; "
                    "  haploinsufficiency → progressive beta-cell loss → fasting hyperglycaemia → T2DM-like. "
                    "SULFONYLUREAS effective for MODY1. "
                    "CLINICAL CLUE: parent with apparent T2DM (actually MODY1) + macrosomic neonate → "
                    "  HNF4A gene sequencing. "
                    "GENETIC COUNSELLING: AD — offspring have 50% risk of same biphasic phenotype."
                ),
            },
            {
                "term": "SLC16A1/MCT1 EIHI — Exercise-Induced Hyperinsulinism (Promoter Mutation)",
                "definition": (
                    "SLC16A1 PROMOTER GOF: MCT1 ectopically expressed in beta-cells (normally absent). "
                    "EXERCISE → muscle pyruvate → enters beta-cell via MCT1 → ATP → K-ATP closes → insulin → hypoglycaemia. "
                    "FASTING: normal glucose (no pyruvate in blood at rest). "
                    "CLINICAL CLUE: post-anaerobic exercise hypoglycaemia + NORMAL fasting study. "
                    "DIAGNOSTIC TRAP: "
                    "  Standard fasting CHI study: NORMAL → EIHI missed; "
                    "  Exon gene panel: NORMAL → promoter mutation missed; "
                    "  MUST request: supervised anaerobic exercise test + SLC16A1 PROMOTER sequencing. "
                    "MANAGEMENT: avoid anaerobic exercise; pre-exercise carbohydrate; "
                    "  diazoxide UNHELPFUL; always carry glucose gel + glucagon kit."
                ),
            },
            {
                "term": "Diazoxide — Mechanism and Response Classification",
                "definition": (
                    "DIAZOXIDE: K-ATP channel opener. "
                    "MECHANISM: binds NBD2 of SUR1 (ABCC8) → stabilises MgADP-bound open state → "
                    "  K-ATP opens → K+ efflux → membrane hyperpolarises → Ca²⁺ VGCCs close → "
                    "  insulin secretion halted. "
                    "RESPONSIVE (K-ATP channel intact): GLUD1, GCK, HADH, HNF4A, UCP2 CHI; "
                    "UNRESPONSIVE (K-ATP absent/dysfunctional): ABCC8, KCNJ11 K-ATP CHI; "
                    "UNHELPFUL (different mechanism): SLC16A1 EIHI. "
                    "SIDE EFFECTS: fluid retention (chlorothiazide co-prescription recommended); "
                    "  hypertrichosis (cosmetic; reversible); "
                    "  rarely pulmonary hypertension in neonates (monitor at initiation). "
                    "DOSE: 5-15 mg/kg/day in 3 divided doses; assess after 5 days at maximum dose."
                ),
            },
            {
                "term": "Hypoglycaemia Brain Injury — Prevention in CHI",
                "definition": (
                    "PROLONGED NEONATAL HYPOGLYCAEMIA (BG <2.6 mmol/L for >30 min): "
                    "  Basal ganglia injury (MRI: T2 hyperintensity); "
                    "  Cerebral cortex watershed damage; "
                    "  Hippocampal injury → memory/learning deficits. "
                    "TARGET BG: >3.5 mmol/L in CHI (higher than general neonatal target); "
                    "GIR target: whatever achieves BG >3.5 mmol/L (may need 15-20 mg/kg/min via central line). "
                    "MONITORING: continuous glucose monitor (CGM) in hospital; "
                    "  check capillary BG 2-4 hourly minimum. "
                    "SAFE DISCHARGE: only when GIR <4 mg/kg/min + oral feeds maintaining BG >3.5 mmol/L "
                    "  through a 5-6 hour fast. "
                    "NEUROLOGICAL FOLLOW-UP: all CHI patients — developmental assessment annually."
                ),
            },
        ]
    }
