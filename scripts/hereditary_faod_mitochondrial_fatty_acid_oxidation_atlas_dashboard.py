#!/usr/bin/env python3
"""Hereditary-FAOD-Mitochondrial-Fatty-Acid-Oxidation-Atlas -- Complete 8-Gene Reference
Hereditary Fatty Acid Oxidation Disorders / Mitochondrial Beta-Oxidation:
Medium-Chain Deficiency (MCAD), Very-Long-Chain Deficiency (VLCAD),
Long-Chain Hydroxyacyl-CoA Dehydrogenase / MTP (LCHAD/MTP alpha),
Carnitine Palmitoyltransferase 1A (CPT1A), Carnitine Palmitoyltransferase 2 (CPT2),
Carnitine Acylcarnitine Translocase (CACT/SLC25A20),
Mitochondrial Trifunctional Protein beta-subunit (HADHB/MTP-beta),
Short-Chain 3-Hydroxyacyl-CoA Dehydrogenase (SCHAD/HADH):

ACADM   (Medium-Chain Acyl-CoA Dehydrogenase; 421aa; 1p31.1;
           AR LOF — MCAD deficiency; MOST COMMON fatty acid oxidation disorder in Caucasians
           (1/10,000–15,000 live births; NBS: ~1/8,000–14,000 in Europe/N.America);
           MECHANISM: ACADM encodes MCAD (homotetrameric mitochondrial flavoprotein);
             beta-oxidation of C6-C12 acyl-CoA species; MCAD catalyses second step of beta-ox
             (oxidation of medium-chain acyl-CoA → enoyl-CoA by FAD-reduction);
             MCAD LOF → medium-chain acyl-CoAs ACCUMULATE (C6-C12) → octanoyl-CoA (C8) + C10
             esterified with carnitine → ELEVATED PLASMA C8, C8:C10, C10:C12 acylcarnitines
             (hallmark on newborn screen tandem-MS); blocked beta-ox → REDUCED acetyl-CoA →
             reduced ketogenesis → HYPOKETOTIC HYPOGLYCEMIA during fasting/illness;
           CLINICAL: typically ASYMPTOMATIC between episodes; triggered by prolonged fasting,
             febrile illness, physiologic stress; acute presentation: vomiting, lethargy,
             hypoketotic hypoglycemia (glucose <2.2 mmol/L with NO ketones = pathognomonic)
             → rapid progression to seizures, encephalopathy, Reye-like syndrome, cardiac arrest;
             sudden unexpected death in infancy (SUDI) — first presentation in 20-25%;
             SURVIVOR rates excellent with NBS + management;
           PATHOGNOMONIC: C8 acylcarnitine markedly elevated (>0.35 µmol/L on dried blood spot);
             C8:C10 ratio elevated; HYPOKETOTIC hypoglycemia (glucose low + urine ketones absent
             or minimal); urine organic acids: hexanoylglycine, suberylglycine, phenylpropionylglycine
             (glycine conjugates of C6-C8 fatty acids — formed when glycine N-acyltransferase
             conjugates accumulating acyl-CoAs → PATHOGNOMONIC on urine OA);
           K304E mutation: most common ACADM mutation in Northern Europeans (>50% alleles);
             c.985A>G (p.Lys304Glu); moderate MCAD residual activity (~0%);
           DIAGNOSTIC: NBS (tandem-MS): C8 elevated = screen positive; plasma acylcarnitines
             (C8, C10:1, C8:C10 ratio); urine organic acids (hexanoyl-, suberyl-,
             phenylpropionylglycine); ACADM gene sequencing; enzymatic assay (fibroblasts/lymphocytes);
           TREATMENT: AVOID prolonged fasting (max 8-12h infants, 12-16h older children, 16-24h adults);
             emergency regimen: 10% dextrose IV at illness onset (2-2.5× maintenance); cornstarch
             (uncooked; slow-release glucose) before prolonged fast; NO dietary fat restriction needed;
             L-carnitine CONTROVERSIAL (may worsen by driving acyl-CoA to cytotoxic acylcarnitine
             accumulation); excellent prognosis with NBS):

ACADVL  (Very-Long-Chain Acyl-CoA Dehydrogenase; 655aa; 17p13.1;
           AR LOF — VLCAD deficiency; second most common LCFAO disorder (1/30,000–100,000);
           MECHANISM: ACADVL encodes VLCAD (homodimer; inner mitochondrial membrane; unique as only
             LCFAO enzyme that does NOT require trifunctional protein; FAD-linked oxidation of
             C14-C20 acyl-CoA species; uses CoQ directly from inner IMM);
             VLCAD LOF → VERY-LONG-CHAIN acyl-CoAs ACCUMULATE → C14:1, C14:1-OH, C14 acylcarnitines
             (hallmark on NBS); uncoupled oxidative phosphorylation → cardiomyopathy;
           CLINICAL: THREE FORMS based on residual VLCAD activity:
             NEONATAL SEVERE (<5% residual): cardiomyopathy (hypertrophic, dilated, arrhythmia),
               hypotonia, liver failure, hypoketotic hypoglycemia → fatal without treatment;
             INFANTILE HEPATIC (<10% residual): hypoketotic hypoglycemia, hepatomegaly, no HCM;
             ADULT MYOPATHIC (>10% residual): rhabdomyolysis triggered by exercise/fasting/fever,
               CK markedly elevated (>10,000 IU/L), myoglobinuria → AKI; peripheral myopathy,
               muscle pain; usually diagnosed late (20-30s) unless NBS;
           PATHOGNOMONIC: C14:1 acylcarnitine elevated (primary marker; >0.2 µmol/L dried blood spot);
             C14, C14:2, C14-OH also elevated; C14:1 ratio to C2/C3 used for NBS specificity;
             CARDIOMYOPATHY + hypoketotic hypoglycemia + C14:1 elevated = VLCAD until proven otherwise;
           DIAGNOSTIC: NBS tandem-MS (C14:1 elevated); plasma acylcarnitines; echocardiogram (cardiac);
             CK, LFTs, lactate; ACADVL gene sequencing; VLCAD enzyme assay (fibroblasts);
           TREATMENT: avoid fasting (max 10-12h); MCT oil supplements (medium-chain triglycerides;
             bypasses VLCAD block; MCT metabolized by MCAD → ketones available for brain/heart);
             low-fat/high-carbohydrate diet; emergency IV dextrose for illness; exercise precautions;
             cardiac monitoring (echo annually); avoid prolonged aerobic exercise without carbohydrate
             preloading; riboflavin (100 mg/day) — cofactor for FAD-linked dehydrogenases, minor benefit):

HADHA   (Hydroxyacyl-CoA Dehydrogenase Trifunctional Protein Subunit Alpha; 763aa; 3p25.3;
           AR LOF — LCHAD deficiency (specific) OR MTP deficiency (combined);
           MECHANISM: HADHA encodes MTP alpha-subunit (1218aa mitochondrial inner membrane
             hetero-octamer with HADHB; 4 alpha + 4 beta); MTP alpha contains TWO ENZYMATIC ACTIVITIES:
             (1) long-chain 2-enoyl-CoA hydratase (second step of beta-ox); and (2) long-chain
             3-HYDROXYACYL-COA DEHYDROGENASE (LCHAD; third step; NAD+-linked; C12-C16 substrates);
             HADHA LOF → BOTH activities lost (MTP deficiency) or SELECTIVE LCHAD LOSS if common
             mutation c.1528G>A (p.Glu510Gln) — most common; selectively destroys LCHAD activity
             while partially sparing other MTP activities;
             ACCUMULATION: long-chain 3-hydroxyacylcarnitines (LCHAC; C16-OH, C18-OH, C18:1-OH)
             → these are TOXIC to retina, peripheral nerves, and liver mitochondria;
           CLINICAL: SEVERE AND UNIQUE features:
             HYPOGLYCEMIA: hypoketotic (like all LCFAO);
             CARDIOMYOPATHY: dilated or hypertrophic; arrhythmias;
             PERIPHERAL NEUROPATHY: progressive axonal; sensorimotor; onset in early childhood
               (PATHOGNOMONIC for LCHAD/MTP among FAOD);
             RETINOPATHY: chorioretinal degeneration, pigmentary retinopathy, progressive visual loss;
               PATHOGNOMONIC for LCHAD/MTP — NO OTHER FAOD causes retinopathy;
             HEPATOPATHY: acute fatty liver, liver failure in acute episodes;
             MATERNAL EFFECT (CRITICAL): LCHAD-deficient fetus → LCHAC accumulate and pass to
               MOTHER (carrier or affected) → ACUTE FATTY LIVER OF PREGNANCY (AFLP) or HELLP
               syndrome in heterozygous CARRIER MOTHERS of LCHAD-deficient fetuses;
               AFLP/HELLP in pregnancy → TEST NEWBORN FOR LCHAD/MTP DEFICIENCY;
           PATHOGNOMONIC: C16-OH, C18-OH, C18:1-OH acylcarnitines elevated (3-hydroxy species);
             TRIAD of peripheral neuropathy + retinopathy + cardiomyopathy + AFLP in maternal history;
             c.1528G>A homozygous = LCHAD-specific deficiency (isolated LCHAD);
           DIAGNOSTIC: NBS tandem-MS (C16-OH, C18-OH elevated); plasma acylcarnitines (3-OH species);
             urine organic acids (3-hydroxy dicarboxylic acids); HADHA sequencing (c.1528G>A screen);
             ophthalmologic evaluation (ERG + fundus); nerve conduction study; echo;
           TREATMENT: low-fat/high-carbohydrate diet + MCT oil (MCT bypasses LCHAD block);
             avoid fasting; DHA supplementation (docosahexaenoic acid; 20-65 mg/kg/day)
             SLOWS retinopathy progression (LCHAD-specific benefit — DHA depleted by blocked
             peroxisomal-mitochondrial interplay); annual ophthalmologic follow-up; cardiac monitoring;
             MATERNAL: carrier mothers of LCHAD fetuses need monitoring for AFLP in subsequent
             pregnancies; neonatal screening of ALL infants born after AFLP in mother):

CPT1A   (Carnitine Palmitoyltransferase 1A; Liver Isoform; 773aa; 11q13.3;
           AR LOF — CPT1A deficiency; affects LIVER ISOFORM of CPT1 (CPT1A);
           MECHANISM: CPT1A is the outer mitochondrial membrane enzyme that INITIATES carnitine
             transport of long-chain acyl-CoAs into the mitochondrial matrix;
             RATE-LIMITING STEP of long-chain fatty acid beta-oxidation;
             malonyl-CoA (from fed-state fatty acid synthesis) INHIBITS CPT1A → physiologic
             suppression of beta-ox during anabolism; CPT1A LOF → LONG-CHAIN acyl-CoAs CANNOT
             ENTER mitochondria → beta-oxidation BLOCKED AT TRANSPORT STEP → no long-chain
             ketone production from fatty acid → HYPOKETOTIC HYPOGLYCEMIA + hepatic fat accumulation;
             UNIQUE ACYLCARNITINE: low C16/C18 acylcarnitines (unlike other LCFAO where they
             ACCUMULATE; here they cannot enter mitochondria so stay as free CoA esters,
             NOT transferred to carnitine) + ELEVATED FREE CARNITINE (C0) and LOW total carnitine
             (because acylcarnitines not formed); C0/(C16+C18) ratio is ELEVATED (>100)
             = PATHOGNOMONIC for CPT1A (unlike all other FAOD);
           ARCTIC VARIANT: c.1436C>T (p.P479L) — VERY HIGH frequency in Indigenous Arctic
             populations (Inuit, Alaska Native, First Nations): allele frequency 0.3-0.8!
             Most homozygous P479L individuals are ASYMPTOMATIC (milder residual activity);
             thought to represent a COLD-ADAPTIVE VARIANT (altered malonyl-CoA sensitivity
             allows more fatty acid beta-oxidation in cold conditions); creates NBS false positives
             in Arctic/Indigenous populations → important to know population background;
           CLINICAL: neonatal/infant onset; severe hypoketotic hypoglycemia during fasting/illness;
             hepatomegaly; elevated aminotransferases; USUALLY NO cardiac involvement
             (CPT1B is the cardiac/muscle isoform; CPT1A = liver only);
           PATHOGNOMONIC: elevated C0 + VERY LOW C16/C18 + C0/(C16+C18) RATIO >100 (unique);
             hypoketotic hypoglycemia; hepatomegaly; absence of cardiac disease;
           DIAGNOSTIC: NBS (C0/(C16+C18) ratio); plasma acylcarnitines; free carnitine;
             CPT1A gene sequencing; CPT1A enzyme assay (lymphocytes, fibroblasts);
             liver biopsy (microvesicular steatosis; not routinely done);
           TREATMENT: avoid fasting (max 8-12h infants); high-carbohydrate diet; cornstarch;
             DO NOT give L-carnitine (already elevated C0; carnitine supplementation unhelpful
             and may be harmful — drives more acylcarnitine cycling); IV dextrose for illness;
             NO fat restriction beyond fasting avoidance; generally good prognosis):

CPT2    (Carnitine Palmitoyltransferase 2; 658aa; 1p32.3;
           AR LOF — CPT2 deficiency; MOST COMMON disorder of LONG-CHAIN FATTY ACID TRANSPORT;
           MECHANISM: CPT2 is the INNER mitochondrial membrane enzyme that COMPLETES carnitine
             transport cycle: removes carnitine from acylcarnitine (transferred by CACT/SLC25A20
             across IMM) and re-esterifies to acyl-CoA → released into matrix for beta-oxidation;
             CPT2 LOF → long-chain acylcarnitines ACCUMULATE in inner membrane space (cannot
             be converted back to acyl-CoA) → C16, C18, C18:1 acylcarnitines elevated;
             CARDIAC AND SKELETAL MUSCLE (CPT2 expressed in muscle, liver, heart, brain);
           CLINICAL: THREE FORMS based on residual CPT2 activity:
             NEONATAL LETHAL (<5% residual activity): neonatal multiorgan failure, seizures,
               liver failure, brain dysgenesis (neuronal migration defects = PATHOGNOMONIC
               for severe CPT2), cardiac arrhythmias; essentially uniformly fatal; cystic
               renal dysplasia at autopsy;
             SEVERE INFANTILE (<10%): neonatal/infant hepatic failure + cardiomyopathy +
               hypoketotic hypoglycemia; high mortality without transplantation;
             ADULT MYOPATHIC FORM (>15% residual — MOST COMMON): episodic rhabdomyolysis
               triggered by prolonged exercise, fasting, cold, fever, general anesthesia;
               CK markedly elevated during attacks (>10,000 IU/L), often >100,000 IU/L;
               myoglobinuria → acute kidney injury; muscle weakness during attacks; normal
               BETWEEN ATTACKS (unlike VLCAD myopathic which has progressive weakness);
               S113L mutation: c.338C>T (p.Ser113Leu) = most common allele in adult CPT2
               deficiency (compound heterozygous frequently); P50H, R631C also common;
           PATHOGNOMONIC: C16 + C18 + C18:1 acylcarnitines elevated (same as VLCAD);
             C16:C2 ratio used to distinguish; rhabdomyolysis + myoglobinuria after exercise
             in adult = CPT2 or VLCAD; ANESTHESIA TRIGGER (malignant hyperthermia-LIKE reactions
             — volatile agents + succinylcholine CONTRAINDICATED in severe neonatal CPT2);
           DIAGNOSTIC: NBS may miss adult myopathic (normal C16 between attacks);
             plasma acylcarnitines during/after rhabdomyolysis attack; CK, LFTs; CPT2 gene
             sequencing (check S113L); CPT2 enzyme assay (lymphocytes);
           TREATMENT: avoid triggers: prolonged exercise without carbohydrate loading, fasting,
             cold exposure, general anesthesia (use total IV anesthesia; AVOID VOLATILE + succinylcholine);
             high-carbohydrate diet; MCT oil (some benefit — partial bypass); L-carnitine
             (may help clear acylcarnitines); bezafibrate (PPAR-alpha agonist; increases LCFAO enzyme
             expression; 400-600 mg/day; REDUCES frequency of rhabdomyolysis attacks in adult form);
             IV dextrose for acute rhabdomyolysis; aggressive IV hydration for AKI prevention;
             furosemide for myoglobinuria + oliguria):

SLC25A20 (Carnitine Acylcarnitine Translocase; CACT; 301aa; 3p21.31;
           AR LOF — CACT deficiency; RARE but clinically severe;
           MECHANISM: SLC25A20 encodes CACT (Carnitine Acylcarnitine Translocase), a mitochondrial
             inner membrane antiporter that exchanges:
             (1) long-chain ACYLCARNITINE (from CPT1 reaction) entering matrix (import)
             (2) FREE CARNITINE leaving matrix (export)
             CPT1A (outer MMM) adds carnitine → acylcarnitine; CACT transports acylcarnitine
             across IMM; CPT2 (inner MMM) removes carnitine → acyl-CoA for beta-oxidation;
             CACT LOF → acylcarnitines CANNOT CROSS IMM → accumulate in intermembrane space →
             beta-oxidation COMPLETELY BLOCKED; ALSO: free carnitine cannot exit matrix →
             carnitine trap → SYSTEMIC CARNITINE DEPLETION (C16, C18 acylcarnitines elevated
             in blood as substrate for CACT never transported + carnitine depleted);
             HYPERAMMONEMIA: blocked FAO → reduced acetyl-CoA → reduced ATP → urea cycle
             impaired (CPS1 requires N-acetylglutamate from acetyl-CoA) → SECONDARY
             HYPERAMMONEMIA (NH3 elevated; distracts from diagnosis unless checked);
           CLINICAL: NEONATAL ONSET SEVERE (most cases):
             neonatal onset hours-days of life: cardiomegaly/cardiomyopathy, arrhythmia,
             hypoglycemia, hyperammonemia, elevated aminotransferases, hypoketotic hypoglycemia;
             hepatomegaly; hypotonia; SUDDEN DEATH frequent if unrecognized; very high neonatal
             mortality; survivors may have cardiac damage;
             LATE-ONSET FORMS (rare): milder, presenting in childhood with acute metabolic decompensation
             triggered by illness or fasting;
           PATHOGNOMONIC: MARKEDLY ELEVATED C16 + C18 + C18:1 acylcarnitines on NBS (often highest
             values of all LCFAO disorders) + LOW FREE CARNITINE (C0 depleted) + HYPERAMMONEMIA
             (secondary) + NEONATAL ONSET CARDIOMYOPATHY; ALL long-chain acylcarnitines elevated;
           DIAGNOSTIC: NBS tandem-MS (C16 + C18 markedly elevated); plasma acylcarnitines; free
             carnitine (low); ammonia; liver enzymes; echo; SLC25A20 gene sequencing;
             CACT enzyme assay (lymphocytes, fibroblasts — technically demanding);
           TREATMENT: EMERGENCY: IV 10% dextrose + insulin (to suppress lipolysis); L-carnitine
             IV supplementation (replace depleted carnitine; 100-400 mg/kg/day IV in acute phase);
             sodium benzoate/phenylacetate for hyperammonemia if NH3 >200 µmol/L; riboflavin;
             cardiac monitoring + anti-arrhythmic if needed; low-fat/MCT formula; avoid fasting
             ABSOLUTELY; prognosis guarded in severe neonatal form; heart transplant considered
             if refractory cardiomyopathy):

HADHB   (Hydroxyacyl-CoA Dehydrogenase Trifunctional Protein Subunit Beta; MTP-beta; 474aa; 2p23.3;
           AR LOF — MTP deficiency (complete); DISTINCT from LCHAD deficiency (HADHA c.1528G>A);
           MECHANISM: HADHB encodes MTP beta-subunit (474aa); HADHB contains the LONG-CHAIN
             3-KETOACYL-COA THIOLASE activity (fourth and final step of beta-oxidation cycle;
             releases acetyl-CoA); the complete MTP (4×alpha/4×beta hetero-octamer) has THREE
             activities: (1) HADHA: enoyl-CoA hydratase; (2) HADHA: LCHAD; (3) HADHB: thiolase;
             HADHB LOF → COMPLETE MTP DYSFUNCTION (all three MTP activities lost);
             CLINICAL CONSEQUENCE: similar to HADHA (LCHAD) deficiency BUT without selective
             LCHAD loss; MTP-deficient patients due to HADHB mutations tend to have a phenotype
             OVERLAPPING WITH LCHAD but potentially MORE SEVERE (all three activities lost);
             3-OH species accumulate (same as LCHAD); however RETINOPATHY may be MILDER than
             ISOLATED LCHAD deficiency (mechanism unclear; possibly less specific LCHAC toxicity
             to retinal pigment epithelium when all three activities are uniformly absent);
             maternal AFLP still occurs (heterozygous carrier mothers of HADHB-deficient fetuses
             have AFLP risk — same principle as HADHA);
           CLINICAL: neonatal/early-infantile presentation: hypoketotic hypoglycemia, liver disease,
             cardiomyopathy, hypotonia; PERIPHERAL NEUROPATHY develops (similar to LCHAD);
             RETINOPATHY: may develop but often milder than LCHAD; DISTINGUISH FROM HADHA:
             measure individual MTP activities in fibroblasts (HADHA c.1528G>A = isolated LCHAD
             only; HADHB = combined MTP all three activities lost); genetic sequencing;
           PATHOGNOMONIC: C16-OH, C18-OH, C18:1-OH acylcarnitines elevated (3-hydroxy species;
             same as HADHA); but also elevated 3-ketospecies if assay available;
           DIAGNOSTIC: NBS (C16-OH, C18-OH); plasma acylcarnitines (3-OH long-chain species);
             HADHA c.1528G>A NOT present (distinguishes from isolated LCHAD); HADHB sequencing;
             individual MTP enzyme activities in fibroblasts;
           TREATMENT: same principles as HADHA/LCHAD: low fat, MCT oil, avoid fasting, DHA
             supplementation, ophthalmologic monitoring, cardiac monitoring; L-carnitine;
             prognosis similar to MTP deficiency, depends on severity of residual activity):

HADH    (Short-Chain 3-Hydroxyacyl-CoA Dehydrogenase; SCHAD; 261aa; 4q25;
           AR LOF — SCHAD/HADH deficiency; UNIQUE FAOD PHENOTYPE: CONGENITAL HYPERINSULINISM;
           MECHANISM: HADH encodes SCHAD (homotetrameric mitochondrial enzyme; NAD+-linked;
             C4-C6 3-hydroxyacyl-CoA substrates; shortest chain LCFAO enzyme);
             SCHAD LOF → SHORT-CHAIN 3-HYDROXYACYL-CoA ACCUMULATION (3-OH-butyryl-CoA etc.);
             HYPERINSULINISM MECHANISM: SCHAD normally INHIBITS GDH (glutamate dehydrogenase;
             mitochondrial; catalyzes glutamate → α-ketoglutarate → enters TCA cycle → ATP);
             SCHAD directly binds and INHIBITS GDH via protein-protein interaction;
             SCHAD LOF → GDH UNRESTRAINED → excessive glutamate → excessive TCA flux →
             excess ATP production → KATP CHANNEL CLOSES (pancreatic beta cell Kir6.2 KATP
             closes when ATP rises) → membrane depolarization → Ca2+ entry → INSULIN EXOCYTOSIS
             WITHOUT GLUCOSE STIMULUS → HYPERINSULINISM;
             NOTE: this KATP-INDEPENDENT mechanism (similar to GDH-activating gain-of-function
             mutations in GDH = HYPERINSULINISM-HYPERAMMONEMIA HI/HA syndrome) but DISTINCT:
             HADH-HI has NO hyperammonemia (GDH drives leucine-stimulated insulin secretion
             but without excess NH3 because HADH-mediated inhibition is specific to insulin,
             not the NH3-generating GDH reaction in liver);
           CLINICAL: CONGENITAL HYPERINSULINISM (CHI):
             neonatal or early-infantile persistent hyperinsulinemic hypoglycemia;
             hypoglycemia unresponsive to glucagon; positive insulin and C-peptide during
             hypoglycemia; ELEVATED plasma 3-OH-butyrylcarnitine (C4-OH) on acylcarnitines;
             urine 3-hydroxyglutaric acid elevated;
             KEY DISTINCTION from other HI: LEUCINE-SENSITIVE HYPERINSULINISM (GDH/HADH-HI)
             — protein meal triggers hypoglycemia; diazoxide-RESPONSIVE (unlike ABCC8/KCNJ11 HI
             which may be diazoxide-unresponsive);
             NO cardiac or skeletal muscle disease (unlike other FAOD);
             NO hypoketotic hypoglycemia pattern (insulin suppresses ketones = HYPOKETOTIC
             but mechanism is INSULIN-MEDIATED not FAO block);
           PATHOGNOMONIC: HYPERINSULINISM + elevated C4-OH acylcarnitine + urine
             3-OH-glutaric aciduria + diazoxide-responsive + leucine sensitivity +
             NO hyperammonemia (distinguishes from GDH HI/HA syndrome) +
             NO retinopathy/neuropathy (distinguishes from LCHAD);
           DIAGNOSTIC: plasma acylcarnitines (C4-OH elevated; quantitative assay); urine OA
             (3-hydroxy-glutaric acid, 3-hydroxy-butyric acid elevated); insulin during hypoglycemia
             (elevated inappropriately); C-peptide; HADH gene sequencing;
             fasting study (if safe): insulin >2 µIU/mL with glucose <2.2 mmol/L = hyperinsulinism;
           TREATMENT: DIAZOXIDE (5-15 mg/kg/day divided TID; opens KATP channels to suppress
             insulin secretion; first-line; usually responsive in HADH-HI); chlorothiazide adjunct;
             LOW-PROTEIN diet (reduces leucine load → reduces GDH-mediated insulin drive);
             avoid prolonged fasting (also triggers episodes); AVOID HIGH-PROTEIN MEALS;
             octreotide (somatostatin analog) if diazoxide-refractory; rare pancreatectomy;
             prognosis excellent with diazoxide).
"""
import random
import statistics

ATLAS = "Hereditary-FAOD-Mitochondrial-Fatty-Acid-Oxidation-Atlas"
GENES = ["ACADM", "ACADVL", "HADHA", "CPT1A", "CPT2", "SLC25A20", "HADHB", "HADH"]
SEEDS = {
    "ACADM":    4118,
    "ACADVL":   4119,
    "HADHA":    4120,
    "CPT1A":    4121,
    "CPT2":     4122,
    "SLC25A20": 4123,
    "HADHB":    4124,
    "HADH":     4125,
}

PROFILES = {
    "ACADM": {
        "syndrome":          "MCAD-Medium-Chain-Acyl-CoA-Dehydrogenase-Deficiency-Most-Common-FAOD-NBS",
        "locus":             "1p31.1",
        "size_aa":           421,
        "inheritance":       "AR LOF",
        "nbs_marker":        "C8 acylcarnitine",
        "hypoketotic_hypo":  True,
        "cardiomyopathy":    False,
        "rhabdomyolysis":    False,
        "retinopathy":       False,
        "neuropathy":        False,
        "hyperinsulinism":   False,
        "hyperammonemia":    False,
        "maternal_aflp":     False,
        "neonatal_lethal":   False,
        "exercise_trigger":  False,
        "diazoxide_rx":      False,
        "mcfad_fasting_max_h": 12,
        "mean_age_onset_yr": 0.3,
        "key_mutation":      "K304E (c.985A>G; >50% European alleles)",
    },
    "ACADVL": {
        "syndrome":          "VLCAD-Very-Long-Chain-Acyl-CoA-Dehydrogenase-Deficiency-Cardiac-Myopathic",
        "locus":             "17p13.1",
        "size_aa":           655,
        "inheritance":       "AR LOF",
        "nbs_marker":        "C14:1 acylcarnitine",
        "hypoketotic_hypo":  True,
        "cardiomyopathy":    True,
        "rhabdomyolysis":    True,
        "retinopathy":       False,
        "neuropathy":        False,
        "hyperinsulinism":   False,
        "hyperammonemia":    False,
        "maternal_aflp":     False,
        "neonatal_lethal":   False,
        "exercise_trigger":  True,
        "diazoxide_rx":      False,
        "mcfad_fasting_max_h": 10,
        "mean_age_onset_yr": 0.1,
        "key_mutation":      "Multiple; >300 mutations; genotype-phenotype correlation",
    },
    "HADHA": {
        "syndrome":          "LCHAD-Long-Chain-Hydroxyacyl-CoA-Dehydrogenase-Deficiency-Retinopathy-AFLP",
        "locus":             "3p25.3",
        "size_aa":           763,
        "inheritance":       "AR LOF",
        "nbs_marker":        "C16-OH acylcarnitine",
        "hypoketotic_hypo":  True,
        "cardiomyopathy":    True,
        "rhabdomyolysis":    True,
        "retinopathy":       True,
        "neuropathy":        True,
        "hyperinsulinism":   False,
        "hyperammonemia":    False,
        "maternal_aflp":     True,
        "neonatal_lethal":   False,
        "exercise_trigger":  True,
        "diazoxide_rx":      False,
        "mcfad_fasting_max_h": 6,
        "mean_age_onset_yr": 0.05,
        "key_mutation":      "c.1528G>A (p.E510Q); selective LCHAD loss; >80% European alleles",
    },
    "CPT1A": {
        "syndrome":          "CPT1A-Carnitine-Palmitoyltransferase-1A-Deficiency-Liver-Hypoketotic-Hypoglycemia",
        "locus":             "11q13.3",
        "size_aa":           773,
        "inheritance":       "AR LOF",
        "nbs_marker":        "C0/(C16+C18) ratio elevated (unique)",
        "hypoketotic_hypo":  True,
        "cardiomyopathy":    False,
        "rhabdomyolysis":    False,
        "retinopathy":       False,
        "neuropathy":        False,
        "hyperinsulinism":   False,
        "hyperammonemia":    False,
        "maternal_aflp":     False,
        "neonatal_lethal":   False,
        "exercise_trigger":  False,
        "diazoxide_rx":      False,
        "mcfad_fasting_max_h": 8,
        "mean_age_onset_yr": 0.2,
        "key_mutation":      "P479L (c.1436C>T); Arctic/Inuit high-frequency adaptive variant",
    },
    "CPT2": {
        "syndrome":          "CPT2-Carnitine-Palmitoyltransferase-2-Deficiency-Adult-Rhabdomyolysis-Neonatal-Lethal",
        "locus":             "1p32.3",
        "size_aa":           658,
        "inheritance":       "AR LOF",
        "nbs_marker":        "C16/C18 acylcarnitines",
        "hypoketotic_hypo":  True,
        "cardiomyopathy":    True,
        "rhabdomyolysis":    True,
        "retinopathy":       False,
        "neuropathy":        False,
        "hyperinsulinism":   False,
        "hyperammonemia":    False,
        "maternal_aflp":     False,
        "neonatal_lethal":   True,
        "exercise_trigger":  True,
        "diazoxide_rx":      False,
        "mcfad_fasting_max_h": 8,
        "mean_age_onset_yr": 0.02,
        "key_mutation":      "S113L (c.338C>T); adult myopathic form; most common allele",
    },
    "SLC25A20": {
        "syndrome":          "CACT-Carnitine-Acylcarnitine-Translocase-Deficiency-Neonatal-Hyperammonemia-Cardiomyopathy",
        "locus":             "3p21.31",
        "size_aa":           301,
        "inheritance":       "AR LOF",
        "nbs_marker":        "C16+C18 markedly elevated; C0 low",
        "hypoketotic_hypo":  True,
        "cardiomyopathy":    True,
        "rhabdomyolysis":    False,
        "retinopathy":       False,
        "neuropathy":        False,
        "hyperinsulinism":   False,
        "hyperammonemia":    True,
        "maternal_aflp":     False,
        "neonatal_lethal":   True,
        "exercise_trigger":  False,
        "diazoxide_rx":      False,
        "mcfad_fasting_max_h": 4,
        "mean_age_onset_yr": 0.002,
        "key_mutation":      "Multiple; M134V (Italian founder); several Asian founder variants",
    },
    "HADHB": {
        "syndrome":          "MTP-beta-Mitochondrial-Trifunctional-Protein-Beta-Subunit-Complete-MTP-Deficiency",
        "locus":             "2p23.3",
        "size_aa":           474,
        "inheritance":       "AR LOF",
        "nbs_marker":        "C16-OH, C18-OH (same as HADHA)",
        "hypoketotic_hypo":  True,
        "cardiomyopathy":    True,
        "rhabdomyolysis":    True,
        "retinopathy":       True,
        "neuropathy":        True,
        "hyperinsulinism":   False,
        "hyperammonemia":    False,
        "maternal_aflp":     True,
        "neonatal_lethal":   False,
        "exercise_trigger":  True,
        "diazoxide_rx":      False,
        "mcfad_fasting_max_h": 6,
        "mean_age_onset_yr": 0.05,
        "key_mutation":      "Multiple; no common single allele; gene panel required",
    },
    "HADH": {
        "syndrome":          "SCHAD-Short-Chain-3-Hydroxyacyl-CoA-Dehydrogenase-Deficiency-Congenital-Hyperinsulinism",
        "locus":             "4q25",
        "size_aa":           261,
        "inheritance":       "AR LOF",
        "nbs_marker":        "C4-OH acylcarnitine",
        "hypoketotic_hypo":  False,
        "cardiomyopathy":    False,
        "rhabdomyolysis":    False,
        "retinopathy":       False,
        "neuropathy":        False,
        "hyperinsulinism":   True,
        "hyperammonemia":    False,
        "maternal_aflp":     False,
        "neonatal_lethal":   False,
        "exercise_trigger":  False,
        "diazoxide_rx":      True,
        "mcfad_fasting_max_h": 12,
        "mean_age_onset_yr": 0.05,
        "key_mutation":      "c.636C>G (p.I212M) most common in Pakistani families; multiple AR alleles",
    },
}


def _make_patients(gene: str) -> list[dict]:
    """Deterministic synthetic cohort of 40 patients per gene."""
    rng = random.Random(SEEDS[gene])
    prof = PROFILES[gene]
    patients = []
    for i in range(40):
        # age at first presentation (months; log-normal around mean)
        base_age_yr = prof["mean_age_onset_yr"]
        age_onset_m = max(0.01, rng.gauss(base_age_yr * 12, base_age_yr * 6))
        # NBS detected?
        nbs_detected = rng.random() < 0.75  # most countries with NBS
        # Features
        hypo_episode = prof["hypoketotic_hypo"] and rng.random() < 0.85
        cardiac = prof["cardiomyopathy"] and rng.random() < 0.55
        rhabdo = prof["rhabdomyolysis"] and rng.random() < 0.50
        retinopathy = prof["retinopathy"] and rng.random() < 0.65
        neuropathy = prof["neuropathy"] and rng.random() < 0.60
        hyperinsulinism = prof["hyperinsulinism"] and rng.random() < 0.95
        hyperammonemia = prof["hyperammonemia"] and rng.random() < 0.80
        maternal_aflp = prof["maternal_aflp"] and rng.random() < 0.20  # in mother (20% of maternal pregnancies)
        neonatal_onset = age_onset_m < 1.0
        # Treatment
        on_mcf_diet = (gene != "HADH") and rng.random() < 0.80
        on_mct = gene in ("ACADVL", "HADHA", "CPT2", "HADHB", "SLC25A20") and rng.random() < 0.70
        on_diazoxide = prof["diazoxide_rx"] and rng.random() < 0.80
        on_bezafibrate = (gene == "CPT2") and rng.random() < 0.40
        on_carnitine = gene in ("SLC25A20", "CPT2") and rng.random() < 0.55
        dha_supplement = (gene in ("HADHA", "HADHB")) and rng.random() < 0.75
        # Lab values (simulated)
        min_glucose_mmol = round(rng.gauss(1.6, 0.5), 1) if hypo_episode else round(rng.gauss(3.2, 0.4), 1)
        min_glucose_mmol = max(0.5, min_glucose_mmol)
        patients.append({
            "id":              f"{gene}-{i+1:03d}",
            "gene":            gene,
            "seed":            SEEDS[gene],
            "age_onset_m":     round(age_onset_m, 2),
            "nbs_detected":    nbs_detected,
            "hypo_episode":    hypo_episode,
            "cardiac":         cardiac,
            "rhabdomyolysis":  rhabdo,
            "retinopathy":     retinopathy,
            "neuropathy":      neuropathy,
            "hyperinsulinism": hyperinsulinism,
            "hyperammonemia":  hyperammonemia,
            "maternal_aflp":   maternal_aflp,
            "neonatal_onset":  neonatal_onset,
            "on_mcf_diet":     on_mcf_diet,
            "on_mct":          on_mct,
            "on_diazoxide":    on_diazoxide,
            "on_bezafibrate":  on_bezafibrate,
            "on_carnitine":    on_carnitine,
            "dha_supplement":  dha_supplement,
            "min_glucose":     min_glucose_mmol,
        })
    return patients


def _cohort_stats(patients: list[dict]) -> dict:
    n = len(patients)
    return {
        "n":                n,
        "nbs_n":            sum(1 for p in patients if p["nbs_detected"]),
        "hypo_n":           sum(1 for p in patients if p["hypo_episode"]),
        "cardiac_n":        sum(1 for p in patients if p["cardiac"]),
        "rhabdo_n":         sum(1 for p in patients if p["rhabdomyolysis"]),
        "retinopathy_n":    sum(1 for p in patients if p["retinopathy"]),
        "neuropathy_n":     sum(1 for p in patients if p["neuropathy"]),
        "hyperinsulinism_n":sum(1 for p in patients if p["hyperinsulinism"]),
        "hyperammonemia_n": sum(1 for p in patients if p["hyperammonemia"]),
        "maternal_aflp_n":  sum(1 for p in patients if p["maternal_aflp"]),
        "neonatal_n":       sum(1 for p in patients if p["neonatal_onset"]),
        "diazoxide_n":      sum(1 for p in patients if p["on_diazoxide"]),
        "bezafibrate_n":    sum(1 for p in patients if p["on_bezafibrate"]),
        "carnitine_n":      sum(1 for p in patients if p["on_carnitine"]),
        "dha_n":            sum(1 for p in patients if p["dha_supplement"]),
        "mean_age_onset_m": round(statistics.mean(p["age_onset_m"] for p in patients), 2),
        "mean_min_glucose": round(statistics.mean(p["min_glucose"] for p in patients), 2),
    }


def generate_overview():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    return {
        "atlas":          ATLAS,
        "subtitle":       "Hereditary Fatty Acid Oxidation Disorders: 8-Gene Complete Reference (ACADM/ACADVL/HADHA/CPT1A/CPT2/SLC25A20/HADHB/HADH)",
        "total_patients": len(all_patients),
        "genes":          GENES,
        "seeds":          SEEDS,
        "cohort_summary": {
            g: {
                "syndrome":          PROFILES[g]["syndrome"],
                "inheritance":       PROFILES[g]["inheritance"],
                "locus":             PROFILES[g]["locus"],
                "size_aa":           PROFILES[g]["size_aa"],
                "nbs_marker":        PROFILES[g]["nbs_marker"],
                "n":                 cohort_stats[g]["n"],
                "nbs_pct":           round(cohort_stats[g]["nbs_n"] / 40 * 100, 1),
                "hypo_pct":          round(cohort_stats[g]["hypo_n"] / 40 * 100, 1),
                "cardiac_pct":       round(cohort_stats[g]["cardiac_n"] / 40 * 100, 1),
                "rhabdo_pct":        round(cohort_stats[g]["rhabdo_n"] / 40 * 100, 1),
                "retinopathy_pct":   round(cohort_stats[g]["retinopathy_n"] / 40 * 100, 1),
                "neuropathy_pct":    round(cohort_stats[g]["neuropathy_n"] / 40 * 100, 1),
                "hyperinsulinism_pct":round(cohort_stats[g]["hyperinsulinism_n"] / 40 * 100, 1),
                "hyperammonemia_pct":round(cohort_stats[g]["hyperammonemia_n"] / 40 * 100, 1),
                "maternal_aflp_pct": round(cohort_stats[g]["maternal_aflp_n"] / 40 * 100, 1),
                "cardiomyopathy":    PROFILES[g]["cardiomyopathy"],
                "retinopathy_flag":  PROFILES[g]["retinopathy"],
                "neuropathy_flag":   PROFILES[g]["neuropathy"],
                "hyperinsulinism_flag": PROFILES[g]["hyperinsulinism"],
                "neonatal_lethal":   PROFILES[g]["neonatal_lethal"],
                "exercise_trigger":  PROFILES[g]["exercise_trigger"],
                "maternal_aflp_flag":PROFILES[g]["maternal_aflp"],
                "diazoxide_rx":      PROFILES[g]["diazoxide_rx"],
                "key_mutation":      PROFILES[g]["key_mutation"],
                "mean_age_onset_m":  cohort_stats[g]["mean_age_onset_m"],
            } for g in GENES
        },
        "key_facts": {
            "mcad_most_common_faod":
                "ACADM (MCAD deficiency) is the MOST COMMON fatty acid oxidation disorder in Caucasian populations (1/10,000–15,000; NBS frequency ~1/10,000). K304E mutation (c.985A>G) accounts for >50% of European alleles. Presentation: hypoketotic hypoglycemia during fasting/illness. Urine organic acids: hexanoylglycine, suberylglycine, phenylpropionylglycine = PATHOGNOMONIC. C8 acylcarnitine on NBS. Excellent prognosis with NBS + fasting avoidance. SUDI (Sudden Unexpected Death in Infancy) in 20-25% of undiagnosed cases.",
            "lchad_retinopathy_neuropathy_aflp":
                "HADHA (LCHAD deficiency; c.1528G>A) is the only FAOD causing RETINOPATHY (chorioretinal degeneration) + PERIPHERAL NEUROPATHY + CARDIOMYOPATHY. Maternal ACUTE FATTY LIVER OF PREGNANCY (AFLP) or HELLP in carrier mothers of LCHAD-deficient fetuses: long-chain 3-hydroxyacylcarnitines from fetus cross placenta → maternal hepatotoxicity. Any AFLP/HELLP → TEST NEWBORN for LCHAD. DHA supplementation slows retinopathy. C16-OH acylcarnitine on NBS. c.1528G>A p.E510Q is the specific LCHAD mutation (>80% European alleles).",
            "cpt2_adult_rhabdomyolysis_anesthesia":
                "CPT2 (Carnitine Palmitoyltransferase 2) deficiency is the MOST COMMON long-chain FAOD transport disorder. Adult myopathic form (S113L allele): episodic rhabdomyolysis from prolonged exercise, fasting, cold, fever. CK >10,000 IU/L (often >100,000). General anesthesia TRIGGER: volatile anesthetics + succinylcholine CONTRAINDICATED in severe neonatal CPT2 (malignant hyperthermia-like reaction). Use TOTAL IV ANESTHESIA only. Bezafibrate (PPAR-alpha agonist) reduces attack frequency in adult form. Three forms: neonatal lethal (brain dysgenesis), severe infantile, adult myopathic.",
            "hadh_hyperinsulinism_schad":
                "HADH (SCHAD deficiency) is the ONLY FAOD causing HYPERINSULINISM instead of hypoketotic hypoglycemia. Mechanism: SCHAD normally inhibits GDH (glutamate dehydrogenase); SCHAD LOF → unrestrained GDH → KATP-channel-independent insulin secretion. C4-OH acylcarnitine elevated + urine 3-OH-glutaric acid. NO hyperammonemia (unlike GDH gain-of-function HI/HA syndrome). DIAZOXIDE-RESPONSIVE. Leucine-sensitive. No retinopathy, neuropathy, or cardiomyopathy. Excellent prognosis with diazoxide + low-protein diet.",
            "cact_neonatal_hyperammonemia":
                "SLC25A20 (CACT deficiency) presents in NEONATES with cardiomegaly + hyperammonemia + hypoketotic hypoglycemia + markedly elevated C16+C18 acylcarnitines + LOW free carnitine. Secondary hyperammonemia (from reduced acetyl-CoA → reduced N-acetylglutamate → reduced CPS1 activity) is a clinical TRAP — may be confused with urea cycle disorders. Emergency treatment: IV dextrose + IV carnitine + sodium benzoate/phenylacetate for NH3. High neonatal mortality without recognition.",
            "cpt1a_arctic_variant_c0_c16_ratio":
                "CPT1A deficiency has a UNIQUE acylcarnitine profile: ELEVATED free carnitine (C0) + LOW C16/C18 acylcarnitines (unlike all other LCFAO where long-chain species accumulate) → C0/(C16+C18) RATIO >100 = PATHOGNOMONIC for CPT1A. ARCTIC VARIANT P479L (c.1436C>T): very high allele frequency in Inuit/Alaska Native/First Nations populations (0.3–0.8); most P479L homozygotes are ASYMPTOMATIC — thought to be a cold-adaptive variant. Creates NBS false positives in Arctic populations.",
            "nbs_faod_tandem_ms":
                "NEWBORN SCREENING (NBS) with tandem-mass spectrometry detects most FAOD: MCAD (C8↑), VLCAD (C14:1↑), LCHAD/MTP (C16-OH↑, C18-OH↑), CPT1A (C0/(C16+C18)↑), CPT2/CACT (C16+C18↑), SCHAD (C4-OH↑). NBS has dramatically reduced FAOD mortality. CRITICAL: NBS does not prevent ALL presentations — false negatives occur; maintain clinical suspicion in HYPOKETOTIC HYPOGLYCEMIA. During acute metabolic crisis: DO NOT WAIT for NBS results.",
            "hypoketotic_hypoglycemia_differential":
                "HYPOKETOTIC HYPOGLYCEMIA (low glucose + NO ketones on urinalysis) = PATHOGNOMONIC for FAOD or HYPERINSULINISM. Normal response to hypoglycemia is KETOSIS. Absent ketones during hypoglycemia → BLOCKED FAO (FAOD) or HYPERINSULINISM (insulin suppresses lipolysis + FAO). DISTINGUISH: FAOD = elevated acylcarnitines on plasma panel; HYPERINSULINISM = elevated insulin + C-peptide during episode; HADH-HI = elevated C4-OH + hyperinsulinism (both present). EMERGENCY management: IV 10% dextrose (2-2.5× maintenance) regardless of cause.",
        },
    }


def generate_breakdown():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    metabolic_groups = {
        "medium_chain_fao_defect": {
            "genes":       ["ACADM"],
            "description": "ACADM (MCAD): medium-chain acyl-CoA dehydrogenase; C6-C12 acyl-CoA oxidation; most common FAOD in Caucasians. ACCUMULATES: C8, C10 acylcarnitines + glycine conjugates (hexanoylglycine, suberylglycine, phenylpropionylglycine on urine OA = diagnostic triad). BLOCKED: C8→C6 acetyl-CoA production → reduced ketogenesis. PRESENTATION: hypoketotic hypoglycemia during fasting/fever; SUDI; encephalopathy. TREATMENT: avoid fasting (max 12h), emergency IV dextrose; no dietary fat restriction. K304E allele. NBS detected pre-symptomatically.",
            "key_test":    "NBS tandem-MS: C8 acylcarnitine elevated (primary marker; >0.35 µmol/L dried blood spot); C8:C10 ratio elevated. Urine OA: hexanoylglycine + suberylglycine + phenylpropionylglycine (PATHOGNOMONIC triad). ACADM sequencing (K304E screen).",
        },
        "very_long_chain_fao_defect": {
            "genes":       ["ACADVL"],
            "description": "ACADVL (VLCAD): very-long-chain acyl-CoA dehydrogenase; C14-C20 acyl-CoA oxidation; homodimer on inner mitochondrial membrane. THREE CLINICAL FORMS based on residual activity: (1) neonatal severe: HCM + hypoketotic hypoglycemia + liver failure; (2) infantile hepatic: hypoglycemia + hepatomegaly; (3) adult myopathic: exercise-triggered rhabdomyolysis (most common presentation; CK >10,000). ACCUMULATES: C14:1 (primary marker), C14, C14:2, C14-OH acylcarnitines. TREATMENT: MCT oil (bypasses VLCAD block), avoid fasting + prolonged exercise, echo annually.",
            "key_test":    "NBS tandem-MS: C14:1 acylcarnitine elevated (primary marker; >0.2 µmol/L). Echocardiogram (cardiomyopathy in neonatal/infantile forms). CK (rhabdomyolysis in adult form). ACADVL sequencing. VLCAD enzyme activity assay (fibroblasts).",
        },
        "long_chain_hydroxyacyl_defect_maternal_aflp": {
            "genes":       ["HADHA", "HADHB"],
            "description": "HADHA (LCHAD/MTP-alpha) and HADHB (MTP-beta): components of mitochondrial trifunctional protein (MTP; inner MMM hetero-octamer). HADHA c.1528G>A = isolated LCHAD deficiency (selective LCHAD loss). HADHB = complete MTP (all three activities absent). BOTH accumulate 3-hydroxy long-chain acylcarnitines (C16-OH, C18-OH, C18:1-OH = toxic to retina and nerves). UNIQUE TO LCHAD/HADHB: retinopathy + peripheral neuropathy + maternal AFLP/HELLP (carrier mothers). HADHA c.1528G>A = LCHAD-specific (>80% European); HADHB has multiple alleles. DHA supplementation slows retinopathy (LCHAD). MCT oil. Ophthalmologic monitoring MANDATORY.",
            "key_test":    "NBS tandem-MS: C16-OH, C18-OH, C18:1-OH acylcarnitines elevated. HADHA sequencing: c.1528G>A (LCHAD) vs HADHB (complete MTP). MTP individual enzyme activities (fibroblasts): distinguish LCHAD-specific (HADHA) from complete MTP (HADHB). Ophthalmologic evaluation (ERG + fundus photography — MANDATORY at diagnosis). Nerve conduction study (peripheral neuropathy). Maternal history: AFLP/HELLP in prior pregnancy.",
        },
        "carnitine_transport_defect": {
            "genes":       ["CPT1A", "CPT2", "SLC25A20"],
            "description": "Carnitine transport cycle disorders. CPT1A (outer MMM, liver isoform): UNIQUE C0/(C16+C18) ratio elevation; no cardiac disease; Arctic P479L variant (asymptomatic). CPT2 (inner MMM; muscle/heart/liver): three forms (neonatal lethal brain dysgenesis, infantile HCM, adult rhabdomyolysis S113L); VOLATILE ANESTHETICS CONTRAINDICATED in neonatal/infantile; bezafibrate for adult form. SLC25A20/CACT (inner MMM antiporter): severe neonatal onset; hyperammonemia secondary (CPS1 reduced); MARKEDLY elevated C16+C18 + low C0; carnitine supplementation IV. CPT1A: no carnitine supplement (C0 elevated). DISTINGUISH from each other: C0/acylcarnitine ratios + clinical form.",
            "key_test":    "CPT1A: C0/(C16+C18) ratio (unique elevation). CPT2/CACT: C16+C18+C18:1 elevation; free carnitine (low in CACT, normal/variable CPT2). Ammonia (CACT: elevated secondary; CPT2: normal). Echo (CACT, CPT2 neonatal: cardiomegaly). Brain MRI (CPT2 neonatal: neuronal migration defect). Enzyme assay distinguishes CPT1A vs CPT2 vs CACT.",
        },
        "short_chain_hyperinsulinism_defect": {
            "genes":       ["HADH"],
            "description": "HADH (SCHAD): only FAOD causing HYPERINSULINISM (not hypoketotic hypoglycemia from FAO block alone). MECHANISM: SCHAD directly inhibits GDH; HADH LOF → unrestrained GDH → KATP-independent insulin secretion. ACCUMULATES: C4-OH acylcarnitine + urine 3-OH-glutaric acid. DIAZOXIDE-RESPONSIVE (unlike ABCC8/KCNJ11 mutations which may be diazoxide-unresponsive). NO hyperammonemia (unlike GDH gain-of-function HI/HA syndrome: GLUD1-GOF). Leucine-sensitive (protein meal triggers hypoglycemia). NO retinopathy, neuropathy, cardiomyopathy. TREATMENT: diazoxide + low-protein diet. Excellent prognosis.",
            "key_test":    "Plasma acylcarnitines: C4-OH elevated (quantitative; not always marked elevation). Urine OA: 3-hydroxyglutaric acid + 3-hydroxybutyric acid. Fasting study: insulin elevated during hypoglycemia (>2 µIU/mL at glucose <2.2 mmol/L). C-peptide elevated. Ammonia: NORMAL (distinguishes from GLUD1 HI/HA syndrome). HADH sequencing. Diazoxide trial (expected response in HADH-HI).",
        },
    }

    return {
        "atlas":          ATLAS,
        "total_patients": len(all_patients),
        "cohort_stats":   cohort_stats,
        "metabolic_groups": metabolic_groups,
        "gene_summary": {
            g: {
                "gene":              g,
                "syndrome":          PROFILES[g]["syndrome"],
                "n":                 cohort_stats[g]["n"],
                "seed":              SEEDS[g],
                "hypo_pct":          round(cohort_stats[g]["hypo_n"] / 40 * 100, 1),
                "cardiac_pct":       round(cohort_stats[g]["cardiac_n"] / 40 * 100, 1),
                "rhabdo_pct":        round(cohort_stats[g]["rhabdo_n"] / 40 * 100, 1),
                "retinopathy_pct":   round(cohort_stats[g]["retinopathy_n"] / 40 * 100, 1),
                "neuropathy_pct":    round(cohort_stats[g]["neuropathy_n"] / 40 * 100, 1),
                "hyperinsulinism_pct":round(cohort_stats[g]["hyperinsulinism_n"] / 40 * 100, 1),
                "hyperammonemia_pct":round(cohort_stats[g]["hyperammonemia_n"] / 40 * 100, 1),
                "neonatal_pct":      round(cohort_stats[g]["neonatal_n"] / 40 * 100, 1),
                "mean_age_onset_m":  cohort_stats[g]["mean_age_onset_m"],
                "nbs_marker":        PROFILES[g]["nbs_marker"],
                "key_mutation":      PROFILES[g]["key_mutation"],
                "diazoxide_rx":      PROFILES[g]["diazoxide_rx"],
                "exercise_trigger":  PROFILES[g]["exercise_trigger"],
                "maternal_aflp":     PROFILES[g]["maternal_aflp"],
            } for g in GENES
        },
    }


def generate_definitions():
    return {
        "definitions": [
            {
                "term": "Fatty Acid Oxidation Cycle — Mitochondrial Beta-Oxidation: The 4-Step Cycle, Carnitine Transport, Chain-Length Specificity",
                "definition": "Mitochondrial beta-oxidation (FAO) is the primary energy source during fasting, exercise, and illness when glucose is limited. The cycle releases acetyl-CoA (enters TCA cycle → ATP, ketone bodies) and reduces NAD+ and FAD (feed electron transport chain). THE CARNITINE TRANSPORT CYCLE: long-chain fatty acids (>C12) CANNOT directly cross the inner mitochondrial membrane (IMM); they require the carnitine shuttle: (1) Fatty acid + CoA → acyl-CoA (outer mitochondrial membrane; ACSL enzymes); (2) CPT1 (outer MMM) transfers acyl-CoA to carnitine → ACYLCARNITINE + CoA; (3) SLC25A20/CACT transports acylcarnitine INTO matrix (antiporter with free carnitine); (4) CPT2 (inner MMM, matrix face) removes carnitine → acyl-CoA + free carnitine; (5) ACSL enzymes in matrix reactivate as acyl-CoA. THE 4-STEP BETA-OXIDATION CYCLE (per iteration, releases 2 carbons as acetyl-CoA): STEP 1: acyl-CoA dehydrogenase (VLCAD for C14-C20; LCAD for C12-C16; MCAD for C6-C12; SCAD for C4-C6) → enoyl-CoA + FADH2; STEP 2: enoyl-CoA hydratase (MTP alpha-subunit for LCHAD; ECHS1 for short-chain) → 3-hydroxyacyl-CoA; STEP 3: 3-hydroxyacyl-CoA dehydrogenase (LCHAD/HADHA for long-chain; SCHAD/HADH for short-chain) → 3-ketoacyl-CoA + NADH; STEP 4: 3-ketoacyl-CoA thiolase (MTP beta-subunit/HADHB for long-chain; ACAT1 for short-chain) → acetyl-CoA + shorter acyl-CoA (enters next cycle). CHAIN LENGTH SPECIFICITY: ACADM handles C6-C12 acyl-CoAs; ACADVL handles C14-C20; MTP (HADHA+HADHB) handles C12-C16; SCHAD (HADH) handles C4-C6. FAO FAILURE → HYPOKETOTIC HYPOGLYCEMIA: reduced acetyl-CoA → reduced ketogenesis; absence of urinary ketones during hypoglycemia = FAO block until proven otherwise. Toxic ACCUMULATION: species-specific acylcarnitines and organic acids (used for diagnosis); medium-chain: glycine conjugates (hexanoylglycine); long-chain: 3-hydroxy species (retinal/nerve toxicity in LCHAD)."
            },
            {
                "term": "MCAD Deficiency — Most Common FAOD: Newborn Screen, K304E Allele, Hypoketotic Hypoglycemia Emergency, Fasting Protocol",
                "definition": "ACADM (MCAD) deficiency is the most common inherited fatty acid oxidation disorder in populations of European ancestry (1/10,000–15,000 live births). MCAD is a FAD-linked homotetrameric enzyme oxidizing medium-chain acyl-CoA (C6-C12) in the first step of beta-oxidation. MCAD LOF → octanoyl-CoA (C8-CoA) + decanoyl-CoA (C10-CoA) accumulate → plasma C8 acylcarnitine elevated (the newborn screen marker: >0.35 µmol/L dried blood spot on tandem-MS). ORGANIC ACID SIGNATURE: hexanoylglycine, suberylglycine, phenylpropionylglycine in urine (formed by glycine N-acyltransferase conjugating C6/C8/phenylpropionyl-CoA from blocked beta-ox) — this triad is PATHOGNOMONIC for MCAD deficiency. K304E MUTATION: c.985A>G encoding p.Lys304Glu, a medium MCAD protein with ~0% residual activity; >50% of European MCAD alleles; identified 1980s-1990s; carrier frequency ~1/50 in Northern Europeans. CLINICAL COURSE: completely asymptomatic between episodes → acute HYPOKETOTIC HYPOGLYCEMIA (glucose <2.2 mmol/L, urine ketones absent or +1 maximum) triggered by: fasting >12 hours (infants), febrile illness, poor oral intake, stress; rapid progression to vomiting → lethargy → seizures → coma → SUDI (Sudden Unexpected Death in Infancy); 20-25% of undiagnosed MCAD patients present with SUDI as their first event; this drove universal NBS adoption in many countries. MANAGEMENT: (1) AVOID prolonged fasting: infants 0-12 months: max 8-hour fast; toddlers: max 12 hours; children: max 16 hours; adults: max 24 hours; (2) EMERGENCY PROTOCOL at illness/fever: 10% dextrose IV at 2-2.5× maintenance rate; or dextrose polymer drink if tolerating orally; do NOT wait for hypoglycemia to develop; (3) UNCOOKED CORNSTARCH (1-2 g/kg at bedtime) for overnight fasting protection in children >2 years; (4) NO dietary fat restriction (fat can be metabolized by long-chain pathways); (5) L-CARNITINE CONTROVERSY: routine supplementation NOT recommended (may worsen toxic acylcarnitine accumulation by driving C8-carnitine conjugation); only supplement if carnitine-depleted; PROGNOSIS: EXCELLENT with NBS detection and fasting avoidance; intellectual outcome NORMAL; no cumulative organ damage between episodes; survival and development indistinguishable from general population."
            },
            {
                "term": "LCHAD Deficiency (HADHA c.1528G>A) — Retinopathy, Peripheral Neuropathy, Maternal AFLP: The Unique FAOD Triad",
                "definition": "LCHAD deficiency (HADHA c.1528G>A) is the only fatty acid oxidation disorder causing CHORIORETINAL DEGENERATION (retinopathy) and PERIPHERAL NEUROPATHY in addition to the standard FAOD features of hypoketotic hypoglycemia and cardiomyopathy. This triad is PATHOGNOMONIC for LCHAD/MTP among inherited metabolic diseases. MECHANISM OF RETINOPATHY: long-chain 3-hydroxyacylcarnitines (C16-OH, C18-OH, C18:1-OH) are cytotoxic to retinal pigment epithelium (RPE) cells via mitochondrial dysfunction and apoptosis; progressive chorioretinal degeneration → reduced ERG amplitude → visual field loss → legal blindness by adolescence/young adulthood in untreated patients; DHA (docosahexaenoic acid) supplementation (20-65 mg/kg/day) slows but does not halt retinopathy; fundus photography + ERG at diagnosis and annually. PERIPHERAL NEUROPATHY: axonal sensorimotor neuropathy; same toxic 3-hydroxy species affect peripheral nerves; rhabdomyolysis episodes from impaired LCFAO in muscle. MATERNAL ACUTE FATTY LIVER OF PREGNANCY (AFLP) AND HELLP: CRITICAL CLINICAL OBSERVATION — heterozygous carrier mothers gestating a LCHAD-deficient fetus accumulate 3-hydroxyacylcarnitines from the fetus (pass through placenta → reach maternal liver) → maternal hepatotoxicity → AFLP (nausea, vomiting, jaundice, elevated LFTs, coagulopathy in 3rd trimester) or HELLP syndrome (hemolysis, elevated liver enzymes, low platelets); DIAGNOSTIC RULE: ANY pregnant woman with AFLP or HELLP → MANDATORY testing of newborn for LCHAD/MTP deficiency (umbilical cord blood or heel stick); if LCHAD confirmed in newborn → test mother for CPT1 deficiency as DDx; repeat in FUTURE pregnancies. HADHA c.1528G>A: selectively destroys LCHAD activity while MTP enoyl-CoA hydratase activity is partially preserved; c.1528G>A p.Glu510Gln is >80% of European LCHAD alleles; homozygous c.1528G>A = isolated LCHAD; compound heterozygous with other HADHA alleles = variable phenotype. NBS MARKER: C16-OH, C18-OH, C18:1-OH acylcarnitines (3-hydroxy species). TREATMENT: low-fat diet (<30% energy from fat) + MCT oil supplementation (MCT bypasses LCHAD block → ketones via MCAD/SCAD); DHA 20-65 mg/kg/day; avoid fasting (max 6h infants); cardiac monitoring (echo annually); ophthalmologic monitoring MANDATORY (ERG + fundus annually); nerve conduction study periodically."
            },
            {
                "term": "CPT2 Deficiency — Three Forms, Adult Rhabdomyolysis, Anesthesia Contraindication, Bezafibrate",
                "definition": "CPT2 deficiency presents in three distinct clinical forms depending on residual CPT2 enzyme activity: NEONATAL LETHAL (<5% residual): presents within hours of birth with multiorgan failure (cardiac arrhythmia, liver failure, hypoglycemia), seizures from metabolic crisis, and — PATHOGNOMONIC FOR SEVERE CPT2 — NEURONAL MIGRATION DEFECTS (brain dysgenesis: lissencephaly-like, polymicrogyria, agenesis of corpus callosum); renal cystic dysplasia at autopsy; uniformly fatal without intensive support; volatile anesthetics + succinylcholine ABSOLUTELY CONTRAINDICATED (malignant hyperthermia-like reaction); use total IV anesthesia ONLY; SEVERE INFANTILE (<10%): hepatic failure + cardiomyopathy + hypoketotic hypoglycemia in early infancy; high mortality; ADULT MYOPATHIC FORM (>15% residual — MOST COMMON FORM overall): episodic rhabdomyolysis from specific triggers: prolonged unaccustomed exercise (marathon, military training), prolonged fasting, cold exposure, fever, general anesthesia, stress, viral illness (influenza); CK during attack: >10,000 IU/L (often 100,000-1,000,000 IU/L); myoglobinuria → dark urine → AKI; between attacks: COMPLETELY NORMAL muscle function, normal CK, normal EMG (unlike metabolic myopathies with progressive weakness); S113L (c.338C>T, p.Ser113Leu) allele is the most common CPT2 adult form allele; compound heterozygosity S113L/other allele common. ANESTHESIA MANAGEMENT: preoperative evaluation mandatory; use PROPOFOL + REMIFENTANIL/FENTANYL (total IV); avoid sevoflurane, desflurane, isoflurane (all volatile agents), succinylcholine (depolarizing neuromuscular blocker); non-depolarizing agents safe (rocuronium, vecuronium). TREATMENT OF RHABDOMYOLYSIS: aggressive IV hydration (saline 200-400 mL/hr); forced diuresis; sodium bicarbonate (alkalinize urine, prevent myoglobin precipitation in renal tubules); dialysis if AKI. PREVENTION: high-carbohydrate loading before exercise (50-100g glucose 1-2h before); MCT oil (partial bypass); avoid overnight fasting; L-carnitine (may help clear C16 acylcarnitines); BEZAFIBRATE 400-600 mg/day (PPAR-alpha agonist increases transcription of LCFAO enzymes including residual CPT2; reduces frequency and severity of rhabdomyolysis attacks in adult form by ~50% in clinical studies; NOT approved for CPT2 but used off-label)."
            },
            {
                "term": "SCHAD (HADH) Deficiency — Congenital Hyperinsulinism via GDH Disinhibition: Mechanism, Diazoxide Response, DDx from HI/HA",
                "definition": "SCHAD deficiency (HADH LOF) is the only fatty acid oxidation disorder presenting as CONGENITAL HYPERINSULINISM rather than hypoketotic hypoglycemia from blocked FAO. MECHANISM OF HYPERINSULINISM: SCHAD normally directly BINDS and INHIBITS GDH (glutamate dehydrogenase; mitochondrial enzyme; EC 1.4.1.3) via protein-protein interaction (SCHAD binds GDH allosteric regulatory domain); GDH catalyzes glutamate → α-ketoglutarate (TCA substrate) + NH3; GDH activation in beta cells → increased TCA → elevated ATP/ADP ratio → KATP channels close → membrane depolarization → voltage-gated Ca2+ entry → insulin granule exocytosis (this is the normal nutrient-coupling mechanism). SCHAD ABSENT → GDH unrestrained → constitutive GDH activity in beta cells → constitutive high ATP → KATP channel permanently closing tendency → HYPERINSULINISM despite normal or low glucose; this is KATP-CHANNEL-INDEPENDENT mechanism (same pathway as GDH gain-of-function GLUD1 mutations). CRITICAL DISTINCTION FROM GLUD1 HI/HA SYNDROME: GLUD1 mutations cause HYPERINSULINISM + HYPERAMMONEMIA (GDH also produces NH3 in liver; GDH-GOF liver = excess NH3) → HADH-HI has NORMAL ammonia (GDH inhibited only in beta cells by SCHAD; liver GDH is not over-activated because SCHAD is not expressed in liver or expression is low); this diagnostic distinction is key. ACYLCARNITINE MARKER: C4-OH (3-hydroxybutyrylcarnitine) is elevated in SCHAD deficiency (short-chain 3-hydroxyacyl-CoA accumulates); quantitative C4-OH >0.1 µmol/L plasma; urine 3-hydroxyglutaric acid and 3-OH-butyric acid elevated. LEUCINE SENSITIVITY: high-protein meal triggers hypoglycemia (leucine is an allosteric GDH activator; in SCHAD-HI, unrestrained GDH amplifies leucine-triggered secretion). DIAZOXIDE RESPONSE: SCHAD-HI is DIAZOXIDE-RESPONSIVE (diazoxide activates KATP channels → counteracts GDH-driven membrane depolarization); most SCHAD-HI patients respond well to diazoxide 5-15 mg/kg/day (compare: ABCC8 LOF mutations causing diffuse HI are often diazoxide-unresponsive). TREATMENT: diazoxide + chlorothiazide (adjunct); low-protein diet (reduce leucine load); avoid prolonged fasting; octreotide if diazoxide-insufficient; frequent feeds. PROGNOSIS: excellent with medical management; no progressive organ involvement."
            },
            {
                "term": "CACT (SLC25A20) Deficiency — Neonatal Emergency: Secondary Hyperammonemia, IV Carnitine, CPT2-CACT Distinction",
                "definition": "CACT deficiency (SLC25A20 AR LOF) is a severe neonatal metabolic emergency. MECHANISM: CACT/SLC25A20 is a mitochondrial inner membrane antiporter that transports ACYLCARNITINES into the matrix in exchange for FREE CARNITINE; without CACT, acylcarnitines cannot enter the matrix → COMPLETE BLOCKADE of long-chain fatty acid beta-oxidation (no transport, no oxidation); simultaneously, free carnitine is TRAPPED in the matrix (cannot exit without exchange transport) → systemic carnitine depletion (low free carnitine; acylcarnitines accumulate in plasma as substrate for CACT is 'stranded'). SECONDARY HYPERAMMONEMIA: blocked FAO → no acetyl-CoA → no N-acetylglutamate synthesis (NAG; cofactor for CPS1, the first enzyme of urea cycle) → CPS1 activity reduced → urea cycle impaired → NH3 accumulates (usually NH3 100-400 µmol/L); this secondary hyperammonemia MIMICS primary urea cycle disorder → clinical TRAP (may initiate UCD workup while FAO disorder is missed if acylcarnitines not checked); always check plasma acylcarnitines in any neonatal hyperammonemia. ACYLCARNITINE PROFILE: C16 + C18 + C18:1 ALL markedly elevated (highest of all FAOD); C0 (free carnitine) LOW (carnitine trap); distinction from CPT2 neonatal (also C16+C18 elevated but free carnitine less depleted). CLINICAL: hours to days of life onset; cardiomegaly + arrhythmia (from C16/C18 acylcarnitine toxicity to cardiac mitochondria + Ca2+ handling); liver failure; hypotonia; seizures; HIGH NEONATAL MORTALITY. EMERGENCY TREATMENT: (1) STOP feeds; (2) IV 10% dextrose + insulin to suppress lipolysis; (3) L-CARNITINE 100-400 mg/kg/day IV (replete depleted carnitine; drives acylcarnitine excretion; critical in CACT); (4) ammonia detoxification: sodium benzoate + phenylacetate (Ucyclyd; 250 mg/kg each IV if NH3 >200 µmol/L); (5) cardiac monitoring; (6) riboflavin (minor benefit); LONG-TERM: MCT formula; L-carnitine supplementation ongoing; avoid fasting absolutely; heart transplant for refractory cardiomyopathy in survivors; DISTINCTION FROM CPT2: CACT has lower C0 (carnitine trapping), more severe hyperammonemia, earlier neonatal onset; CPT2 may have normal C0; enzyme assay or gene sequencing required to distinguish definitively."
            },
            {
                "term": "CPT1A Deficiency — Liver-Specific FAOD: Arctic P479L Variant, C0/(C16+C18) Ratio, No Cardiac Disease",
                "definition": "CPT1A deficiency affects the LIVER ISOFORM of Carnitine Palmitoyltransferase 1 (CPT1A), leaving the cardiac/muscle isoform (CPT1B) intact. This explains the unique clinical phenotype: HYPOKETOTIC HYPOGLYCEMIA + HEPATOMEGALY but ABSENT CARDIAC DISEASE (unlike CPT2, VLCAD, CACT which affect cardiac isoforms or cardiac-expressed enzymes). MECHANISM: CPT1A transfers acyl group from long-chain acyl-CoA to carnitine on the outer mitochondrial membrane (rate-limiting step of LCFAO); CPT1A LOF → long-chain acyl-CoAs CANNOT be esterified to acylcarnitine for transport into mitochondria → beta-oxidation blocked at entry; UNIQUE ACYLCARNITINE PROFILE: long-chain acylcarnitines (C16, C18) are LOW or normal (unlike all other LCFAO where they ACCUMULATE; here the block is BEFORE acylcarnitine formation) + FREE CARNITINE (C0) is ELEVATED (not consumed because acyl-CoA → acylcarnitine reaction is blocked) → C0/(C16+C18) RATIO IS MARKEDLY ELEVATED (>100; normal <10) = PATHOGNOMONIC ratio for CPT1A deficiency; this inverted pattern is unique among FAOD. ARCTIC/INUIT P479L VARIANT: c.1436C>T encoding p.Pro479Leu; allele frequency 0.3-0.8 in Inuit (Alaska, Canada, Greenland), Alaskan Native, and some First Nations populations; MOST HOMOZYGOUS P479L INDIVIDUALS ARE CLINICALLY ASYMPTOMATIC or have only minor metabolic perturbation; COLD-ADAPTIVE HYPOTHESIS: P479L alters malonyl-CoA sensitivity of CPT1A (malonyl-CoA normally inhibits CPT1A to prevent futile FAO during anabolism) → P479L may allow greater FAO even in anabolic state → improved fatty acid fuel utilization in cold Arctic environment (cold exposure increases LCFAO demand for thermogenesis); P479L creates NBS false positives in Arctic Indigenous populations → critical to understand population genetics context before managing. TREATMENT: avoid prolonged fasting (max 8-12h infants); high-carbohydrate diet; uncooked cornstarch; IV dextrose for illness; DO NOT give L-carnitine (free carnitine already elevated; supplementation unhelpful and potentially drives alternate toxic pathways); NO dietary fat restriction; PROGNOSIS: generally good; episodes reduce with age; no cumulative cardiac or neuromuscular damage (liver isoform only)."
            },
            {
                "term": "Cascade Diagnostic Algorithm — FAOD Workup: NBS Positive → Plasma Acylcarnitines → Urine OA → Enzyme → Gene",
                "definition": "STEP 1 — CLINICAL SUSPICION: hypoketotic hypoglycemia (ANY age; fasting, fever, illness) → check urine ketones immediately; if glucose low + urine ketones absent → FAOD or hyperinsulinism panel; rhabdomyolysis from exercise/fasting (especially adult) → CPT2 or VLCAD; neonatal cardiomyopathy + C16/C18 elevated → VLCAD or CACT or CPT2 neonatal; neonatal hyperammonemia + C16/C18 + low C0 → CACT; retinopathy + neuropathy + maternal AFLP → LCHAD/MTP; hyperinsulinism + C4-OH elevated → HADH (SCHAD); STEP 2 — NEWBORN SCREEN INTERPRETATION: C8↑ → MCAD; C14:1↑ → VLCAD; C16-OH+C18-OH↑ → LCHAD/MTP (HADHA or HADHB); C16+C18↑ (no hydroxyl) → CPT2 or CACT (check free carnitine: low = CACT; normal = CPT2); C0/(C16+C18)↑ → CPT1A; C4-OH↑ → SCHAD/HADH (check insulin); STEP 3 — CONFIRMATORY PLASMA ACYLCARNITINES: precise quantification beyond NBS; C8:C10 ratio (MCAD); 3-OH species pattern (LCHAD vs MTP); free carnitine (CACT: low; CPT1A: high; CPT2: variable); STEP 4 — URINE ORGANIC ACIDS: hexanoylglycine + suberylglycine + phenylpropionylglycine → MCAD; 3-OH dicarboxylic acids → VLCAD/LCHAD; 3-hydroxyglutaric acid → SCHAD; STEP 5 — DISEASE-SPECIFIC TESTS: insulin + C-peptide during hypoglycemia (if hyperinsulinism suspected); echo + EKG (cardiac FAOD: VLCAD, CACT, CPT2); liver function + ammonia (CACT, CPT1A); ophthalmologic evaluation (LCHAD/HADHB); nerve conduction (LCHAD/HADHB); CPT enzyme assays (distinguish CPT1 vs CPT2); MTP individual activities (distinguish LCHAD/HADHA vs MTP-HADHB); STEP 6 — GENETIC SEQUENCING: gene panel (ACADM, ACADVL, HADHA, HADHB, CPT1A, CPT2, SLC25A20, HADH) for confirmation and family counseling; STEP 7 — FAMILY CASCADE: parents are obligate carriers; siblings 25% risk; prenatal diagnosis available; EMERGENCY PROTOCOL during any acute episode: DO NOT delay treatment for diagnostic confirmation — IV 10% dextrose FIRST (removes fatty acid mobilization drive), then investigate."
            },
        ],
        "key_distinguishers": {
            "mcad_most_common_glycine_conjugates":   "MCAD (ACADM): MOST COMMON FAOD Caucasians. C8↑ NBS. Urine: hexanoylglycine+suberylglycine+phenylpropionylglycine (PATHOGNOMONIC). K304E >50% European alleles. SUDI in 20-25% undiagnosed. Avoid fasting. NO fat restriction. Excellent prognosis.",
            "lchad_retinopathy_neuropathy_aflp":     "LCHAD (HADHA c.1528G>A): ONLY FAOD with retinopathy + neuropathy. Maternal AFLP/HELLP in carrier mothers. C16-OH/C18-OH↑. DHA supplementation. MCT oil. Ophthalmologic monitoring mandatory. c.1528G>A p.E510Q = selective LCHAD (vs HADHB = complete MTP).",
            "hadhb_complete_mtp_milder_retinopathy": "MTP-beta (HADHB): complete MTP all 3 activities lost. Retinopathy MILDER than LCHAD. Maternal AFLP still occurs. C16-OH/C18-OH↑ (same as HADHA). Distinguish by MTP individual enzyme activities: HADHB = thiolase absent also; HADHA = thiolase partial.",
            "cpt2_adult_rhabdo_anesthesia_bezafibrate":"CPT2: MOST COMMON LCFAO transport disorder. Adult myopathic (S113L): rhabdomyolysis from exercise/fever/cold. VOLATILE ANESTHETICS + succinylcholine CONTRAINDICATED. Total IV anesthesia. Bezafibrate 400mg/day reduces attacks. 3 forms: neonatal lethal (brain dysgenesis), infantile, adult.",
            "cpt1a_inverted_c0_ratio_arctic_p479l":  "CPT1A: UNIQUE inverted profile C0↑ + C16/C18↓ → C0/(C16+C18) >100 PATHOGNOMONIC. NO cardiac disease (liver isoform only). Arctic P479L: asymptomatic in most Inuit/Indigenous → NBS false positive. No carnitine supplement.",
            "cact_neonatal_hyperammonemia_low_c0":   "CACT (SLC25A20): C16+C18 MARKEDLY↑ + C0 LOW (carnitine trap) + HYPERAMMONEMIA secondary + neonatal cardiomyopathy. EMERGENCY: IV dextrose + IV carnitine + NH3 scavengers. Distinguish from CPT2: free carnitine (CACT=low; CPT2=variable).",
            "hadh_hyperinsulinism_no_hyperammonemia":"SCHAD (HADH): ONLY FAOD causing HYPERINSULINISM (not hypoglycemia from FAO block). C4-OH↑ + 3-OH-glutaric aciduria. NO hyperammonemia (distinguishes from GLUD1 HI/HA). DIAZOXIDE-RESPONSIVE. Leucine-sensitive. No cardiac/retinal/neurologic involvement.",
            "vlcad_c14_1_3_forms_mct":               "VLCAD (ACADVL): C14:1↑ NBS. 3 forms: neonatal HCM (severe), infantile hepatic, adult rhabdomyolysis (most common on NBS era). MCT oil supplementation (bypasses VLCAD block). Riboflavin minor benefit. Genotype-phenotype: residual activity determines form.",
        },
        "treatment_matrix": {
            "fasting_avoidance_all_faod":         "ALL FAOD require fasting avoidance: Infants (<12m): max 8-12h fast. Toddlers (1-3y): max 12h. Children (3-12y): max 14-16h. Adults: max 18-24h. EXCEPTION: SCHAD/HADH-HI has different fasting rules (avoid but mechanism is hyperinsulinism, not direct FAO block). EMERGENCY at any illness: IV 10% dextrose at 2-2.5× maintenance OR dextrose polymer drink. Do NOT manage at home in infants with fever/vomiting. Hospital protocol card should be worn by patient at all times.",
            "mct_oil_supplement_lcfao":           "MCT oil (medium-chain triglycerides C6-C12): recommended for VLCAD, HADHA, HADHB, CPT2, SLC25A20. Dose: 0.5-2g/kg/day in divided meals. Provides ketogenic fuel that bypasses the LCFAO block (MCT → MCAD pathway → acetyl-CoA → ketones). Does NOT work for MCAD (C8-C12 accumulated further). Monitor for GI intolerance (start low, titrate).",
            "diazoxide_schad_hi":                 "Diazoxide: FIRST-LINE for SCHAD (HADH) hyperinsulinism. Dose: 5-15 mg/kg/day divided TID. Mechanism: opens KATP channels → counteracts GDH-driven insulin secretion. Add chlorothiazide (7-10 mg/kg/day) to prevent fluid retention from diazoxide. Titrate to glucose >3.5 mmol/L. Monitor: hypertrichosis, fluid retention, pulmonary hypertension (rare). Target resolution of hypoglycemia without need for frequent feeds.",
            "bezafibrate_cpt2_adult":             "Bezafibrate 400mg/day (PPAR-alpha agonist): OFF-LABEL for adult myopathic CPT2. Mechanism: increases transcription of CPTI, CPTII, VLCAD, LCAD enzymes → higher residual CPT2 activity from upregulation. Reduces rhabdomyolysis frequency ~50% in observational studies. Safe profile. Monitor: CPK, renal function (fibrates + statins = myopathy risk; avoid combination). Not approved but widely used in metabolic centers.",
            "dha_supplementation_lchad":          "DHA (docosahexaenoic acid): 20-65 mg/kg/day for HADHA (LCHAD) and HADHB (MTP) deficiency. Evidence: randomized trial showed reduced rate of retinal degeneration (ERG amplitude preservation). Mechanism unclear — may replete DHA in retinal phospholipids depleted by impaired peroxisomal-mitochondrial DHA synthesis. Start at diagnosis, continue lifelong. Brand: algae-derived DHA supplements. Monitor: annual ERG + fundus. Also consider in HADHB (complete MTP).",
            "carnitine_supplementation_selective": "L-Carnitine supplementation — SELECTIVE use: GIVE in: SLC25A20/CACT (IV carnitine 100-400 mg/kg/day acute; 50-100 mg/kg/day maintenance — depleted by trap); CPT2 (may help clear acylcarnitines); some VLCAD. DO NOT GIVE in: MCAD (may worsen C8-carnitine accumulation toxicity); CPT1A (free carnitine already elevated). Check plasma free carnitine before supplementing. Target free carnitine 30-60 µmol/L.",
            "anesthesia_protocol_cpt2":           "CPT2 ANESTHESIA PROTOCOL: (1) Preoperative metabolic clearance MANDATORY; (2) Use TOTAL IV ANESTHESIA: propofol induction + maintenance, remifentanil/fentanyl for analgesia; (3) AVOID ALL VOLATILE AGENTS (sevoflurane, desflurane, isoflurane, halothane); (4) AVOID succinylcholine (depolarizing NMBA); use rocuronium/vecuronium (non-depolarizing — safe); (5) Maintain IV dextrose perioperatively; (6) Monitor CK 24h post-op; (7) Alert anesthesiologist BEFORE scheduling surgery.",
            "ophthalmologic_monitoring_lchad":    "Ophthalmologic monitoring (LCHAD/HADHB): ERG (electroretinogram) at diagnosis and annually; fundus photography + OCT annually; visual field testing from age 5-6. Warn family: retinal degeneration is progressive even with optimal treatment; early detection allows protective measures (low-light environments, UV protection); low vision rehabilitation when needed. DHA may slow progression — start at diagnosis.",
        },
    }
