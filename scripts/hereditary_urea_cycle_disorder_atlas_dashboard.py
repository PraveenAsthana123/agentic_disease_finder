#!/usr/bin/env python3
"""Hereditary-Urea-Cycle-Disorder-Atlas -- Complete 8-Gene Reference
Urea cycle enzyme/transporter deficiencies with hyperammonemia phenotypes
(OTC/X-linked-most-common, ASS1/CTLN1-citrullinemia, ASL/argininosuccinicaciduria,
 ARG1/hyperargininemia, CPS1/CPS1-deficiency, NAGS/carglumic-acid-CURATIVE,
 SLC25A13/citrin-CTLN2-Japanese, SLC25A15/HHH-syndrome):

OTC    (Ornithine transcarbamylase; 354aa; Xp11.4; X-linked;
         Most common UCD worldwide (1:14,000);
         X-linked: males severe neonatal crisis; females mosaic/variable;
         Citrulline LOW + orotic acid HIGH PATHOGNOMONIC;
         Nitrogen scavengers (sodium benzoate/phenylbutyrate);
         Liver transplant CURATIVE; avoid fasting/catabolism;
         seed SEED_BASE+0).
ASS1   (Argininosuccinate synthetase 1; 412aa; 9q34.11; AR;
         Citrullinemia type 1 (CTLN1);
         Plasma citrulline >1000 μmol/L PATHOGNOMONIC;
         Arginine supplementation mandatory;
         Liver transplant curative;
         seed SEED_BASE+1).
ASL    (Argininosuccinate lyase; 464aa; 7q11.21; AR;
         Argininosuccinic aciduria (ASA);
         Argininosuccinic acid in plasma/urine PATHOGNOMONIC;
         Liver-brain dissociation: liver LTx corrects UCD but NOT neurological;
         Systemic hypertension in 75% even treated;
         seed SEED_BASE+2).
ARG1   (Arginase 1; 322aa; 6q23.2; AR;
         Hyperargininemia (argininemia);
         MILD hyperammonemia — spastic diplegia + cognitive decline PATHOGNOMONIC;
         Plasma arginine >200 μmol/L diagnostic;
         Low-arginine diet prevents neurological progression;
         seed SEED_BASE+3).
CPS1   (Carbamoyl phosphate synthetase 1; 1500aa; 2q34; AR;
         CPS1 deficiency; most severe neonatal UCD;
         Very low citrulline (identical to OTC/NAGS); orotic acid NORMAL (unlike OTC);
         Carglumic acid NOT effective — distinct from NAGS;
         seed SEED_BASE+4).
NAGS   (N-acetylglutamate synthase; 534aa; 17q21.31; AR;
         NAGS deficiency; ONLY UCD where carglumic acid (Carbaglu FDA2010) is CURATIVE;
         Low citrulline + normal orotic acid (same as CPS1); biochemistry identical;
         Carglumic acid mimics NAG, bypasses NAGS defect;
         seed SEED_BASE+5).
SLC25A13 (Citrin / mitochondrial aspartate-glutamate carrier 2; 675aa; 7q21.3; AR;
           Citrin deficiency / Citrullinemia type II (CTLN2);
           Japanese founder (exon16 ins3kb); adult onset;
           Alcohol + high-carbohydrate diet ABSOLUTELY TRIGGER acute liver failure;
           Low-carb high-fat/protein diet; liver transplant CURATIVE;
           seed SEED_BASE+6).
SLC25A15 (Ornithine carrier 1 / ORNT1; 301aa; 13q14.11; AR;
           HHH syndrome (hyperornithinemia-hyperammonemia-homocitrullinuria);
           Homocitrulline in urine PATHOGNOMONIC;
           Progressive spastic paraparesis + cognitive decline;
           Ornithine restriction + citrulline supplement;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3806-3813)
"""
import random

SEED_BASE = 3806

ATLAS_GENES = [
    {
        "gene": "OTC",
        "protein": (
            "OTC -- Xp11.4 X-linked LOF -- 354aa -- "
            "Ornithine-transcarbamylase-mitochondrial-matrix-enzyme-"
            "most-common-UCD-1in14000-males-severe-neonatal-females-variable-mosaic-"
            "citrulline-low-orotic-acid-high-PATHOGNOMONIC-"
            "nitrogen-scavengers-sodium-benzoate-phenylbutyrate-liver-transplant-CURATIVE"
        ),
        "locus": "Xp11.4",
        "protein_size": (
            "354 aa / ~40 kDa / Xp11.4 OTC ornithine transcarbamylase molecular context: "
            "STRUCTURE: "
            "  354 aa; mitochondrial matrix enzyme; catalyzes carbamoyl phosphate + ornithine -> citrulline; "
            "  Second step in urea cycle; X-linked inheritance (gene on X chromosome); "
            "  Males: hemizygous LOF = severe neonatal hyperammonemia (most severe form); "
            "  Females: carrier mosaic (variable expression depending on X-inactivation ratio); "
            "  Female carriers: 20-30% become symptomatic (migraine, protein aversion, encephalopathy); "
            "  >600 pathogenic OTC variants identified; "
            "MALE PRESENTATION (NEONATAL): "
            "  Hours-to-days: lethargy, poor feeding, vomiting, hypotonia; "
            "  Progressive encephalopathy, coma, herniation if untreated; "
            "  Plasma ammonia: >1000 μmol/L (normal <50); "
            "  WITHOUT DIAGNOSIS/TREATMENT: 100% fatal or severe neurological damage; "
            "FEMALE CARRIER PRESENTATION: "
            "  Ranges: asymptomatic → migraine + protein aversion → late-onset hyperammonemia; "
            "  Often triggered by catabolism (illness, surgery, fasting, high protein); "
            "  DIAGNOSIS IN FEMALES: enzyme activity unreliable (mosaic); gene sequencing mandatory; "
            "BIOCHEMISTRY PATHOGNOMONIC: "
            "  Plasma citrulline: LOW/ABSENT (<5 μmol/L) — OTC produces citrulline; "
            "  Urine orotic acid: ELEVATED — carbamoyl phosphate shunted to pyrimidine synthesis; "
            "  Plasma ammonia: severely elevated during crisis; "
            "  CITRULLINE LOW + OROTIC ACID HIGH = OTC pattern (vs CPS1/NAGS: BOTH low); "
            "  Plasma glutamine elevated; arginine low; "
            "TREATMENT: "
            "  Acute hyperammonemic crisis: hemodialysis for NH3 >500 μmol/L + IV dextrose (stop catabolism); "
            "  Nitrogen scavengers: sodium benzoate 250 mg/kg/day + sodium phenylbutyrate/phenylacetate; "
            "    (Buphenyl, RAVICTI/glycerol phenylbutyrate FDA2013); "
            "  Citrulline supplementation 200 mg/kg/day (replenishes depleted citrulline); "
            "  Low-protein diet with essential amino acid supplement; "
            "  FASTING/CATABOLISM ABSOLUTELY AVOID: triggers hyperammonemia crisis; "
            "  Liver transplant: CURATIVE for urea cycle defect; NO ammonia crises post-LTx; "
            "    NOTE: LTx does NOT reverse pre-existing neurological damage"
        ),
        "syndrome": (
            "OTC-deficiency / X-linked-most-common-UCD / "
            "males-severe-neonatal-hyperammonemia / females-carrier-variable / "
            "citrulline-LOW + orotic-acid-HIGH-PATHOGNOMONIC / "
            "nitrogen-scavengers-RAVICTI / liver-transplant-CURATIVE / "
            "fasting-catabolism-ABSOLUTELY-AVOID"
        ),
        "inheritance": "X-linked (Xp11.4) — males hemizygous severe; females carrier mosaic variable",
        "key_therapy": "Nitrogen scavengers (RAVICTI/glycerol phenylbutyrate FDA2013) — citrulline supplementation — LTx CURATIVE",
        "pathognomonic": (
            "Plasma citrulline LOW (<5 μmol/L) + urine orotic acid ELEVATED PATHOGNOMONIC OTC pattern; "
            "Distinguishes OTC from CPS1/NAGS (orotic acid NORMAL in CPS1/NAGS); "
            "Plasma ammonia >1000 μmol/L neonatal crisis; "
            "Protein aversion (child refuses protein foods) behavioral marker"
        ),
        "key_drugs_ci": [
            "Sodium-benzoate-250mgkgday-nitrogen-scavenger",
            "RAVICTI-glycerol-phenylbutyrate-FDA2013-nitrogen-scavenger",
            "Citrulline-supplementation-200mgkgday",
            "Hemodialysis-NH3-gt500-micromolL",
            "Liver-transplant-CURATIVE-no-crises-post-LTx",
            "FASTING-CATABOLISM-ABSOLUTELY-AVOID",
        ],
    },
    {
        "gene": "ASS1",
        "protein": (
            "ASS1 -- 9q34.11 AR LOF -- 412aa -- "
            "Argininosuccinate-synthetase-1-cytosolic-enzyme-"
            "citrullinemia-type-1-CTLN1-"
            "plasma-citrulline-gt1000-micromolL-PATHOGNOMONIC-"
            "arginine-supplementation-mandatory-liver-transplant-curative"
        ),
        "locus": "9q34.11",
        "protein_size": (
            "412 aa / ~47 kDa / 9q34.11 ASS1 argininosuccinate synthetase molecular context: "
            "STRUCTURE: "
            "  412 aa; cytosolic; catalyzes citrulline + aspartate -> argininosuccinate (step 3 of urea cycle); "
            "  Deficiency -> citrulline massively accumulates (cannot be processed); "
            "  AR inheritance; biallelic ASS1 mutations; "
            "  Two phenotypes: neonatal-onset severe (classic) + mild adult citrullinemia; "
            "CITRULLINEMIA TYPE 1 (CTLN1) -- CLASSIC NEONATAL: "
            "  Onset: 24-72 hr of life (similar to OTC); "
            "  Severe hyperammonemia (>1000 μmol/L); encephalopathy; "
            "  Progressive lethargy, coma, cerebral edema; "
            "  HIGH-RISK: neonatal death without diagnosis; "
            "BIOCHEMISTRY PATHOGNOMONIC: "
            "  Plasma citrulline: MASSIVELY ELEVATED >1000 μmol/L (diagnostic threshold); "
            "    Normal <40 μmol/L; CTLN1 >1000 = PATHOGNOMONIC; "
            "  Urine orotic acid: NORMAL or mildly elevated (no carbamoyl phosphate shunting); "
            "  Arginine: low (downstream of block); "
            "  ASS enzyme activity in fibroblasts: absent/very low; "
            "NBS: citrulline >150 μmol/L on Guthrie card triggers ASS1/ASL investigation; "
            "ARGININE SUPPLEMENTATION: "
            "  Essential: arginine is conditionally essential in UCD (cannot be synthesized past block); "
            "  Supplementation prevents arginine deficiency; "
            "  Citrulline competes with arginine — net effect: scavenges nitrogen (1 mol N per mol citrulline); "
            "TREATMENT: "
            "  Acute: hemodialysis + dextrose + nitrogen scavengers (benzoate/phenylbutyrate); "
            "  Chronic: low-protein diet + arginine 300-600 mg/kg/day; "
            "  Nitrogen scavengers: sodium benzoate + RAVICTI; "
            "  Liver transplant: CURATIVE (expresses ASS1 in liver); "
            "    NBS: LTx often performed in 1st year before brain damage"
        ),
        "syndrome": (
            "Citrullinemia-type-1-CTLN1 / AR-ASS1-deficiency / "
            "plasma-citrulline-gt1000-micromolL-PATHOGNOMONIC / "
            "neonatal-severe-hyperammonemia / arginine-supplementation-mandatory / "
            "nitrogen-scavengers / liver-transplant-curative"
        ),
        "inheritance": "AR (biallelic ASS1 mutations)",
        "key_therapy": "Arginine 300-600 mg/kg/day + nitrogen scavengers — LTx CURATIVE — low-protein diet",
        "pathognomonic": (
            "Plasma citrulline >1000 μmol/L PATHOGNOMONIC for ASS1 (CTLN1) or ASL; "
            "Urine orotic acid NORMAL (distinguishes from OTC); "
            "ASS1 vs ASL: argininosuccinic acid ABSENT in ASS1 (present in ASL); "
            "Newborn screen citrulline >150 μmol/L = tier-2 testing mandatory"
        ),
        "key_drugs_ci": [
            "Arginine-300-600mgkgday-supplement-MANDATORY",
            "Sodium-benzoate-nitrogen-scavenger",
            "RAVICTI-glycerol-phenylbutyrate-FDA2013",
            "Hemodialysis-acute-NH3-gt500",
            "Liver-transplant-CURATIVE-first-year-before-damage",
            "Low-protein-essential-AA-supplement",
        ],
    },
    {
        "gene": "ASL",
        "protein": (
            "ASL -- 7q11.21 AR LOF -- 464aa -- "
            "Argininosuccinate-lyase-cytosolic-enzyme-"
            "argininosuccinic-aciduria-ASA-"
            "argininosuccinic-acid-plasma-urine-PATHOGNOMONIC-"
            "liver-brain-dissociation-LTx-corrects-UCD-NOT-neurological-"
            "systemic-hypertension-75pct-even-treated"
        ),
        "locus": "7q11.21",
        "protein_size": (
            "464 aa / ~52 kDa / 7q11.21 ASL argininosuccinate lyase molecular context: "
            "STRUCTURE: "
            "  464 aa; cytosolic homotetramer; cleaves argininosuccinate -> arginine + fumarate (step 4 urea cycle); "
            "  Deficiency -> argininosuccinate accumulates in blood + urine; arginine cannot be synthesized; "
            "  AR inheritance; biallelic ASL mutations; "
            "  Most common: c.1153C>T (p.Arg385Cys) founder variant; "
            "ARGININOSUCCINIC ACIDURIA (ASA): "
            "  Onset: neonatal severe OR late-onset attenuated forms; "
            "  Neonatal: hyperammonemia crisis (similar to other UCDs); "
            "  Late-onset: cognitive delay, behavioral, hepatomegaly, trichorrhexis nodosa (brittle hair); "
            "TRICHORRHEXIS NODOSA: "
            "  Brittle, breaking hair (nodes visible on microscopy); "
            "  PATHOGNOMONIC for ASL deficiency in UCD context; "
            "  Caused by arginine deficiency affecting hair protein synthesis; "
            "BIOCHEMISTRY PATHOGNOMONIC: "
            "  Argininosuccinic acid: ELEVATED in plasma AND urine = PATHOGNOMONIC; "
            "  Plasma citrulline: elevated (moderate, <1000 μmol/L — less than ASS1); "
            "  Arginine: LOW (downstream of block); orotic acid: normal-mildly elevated; "
            "LIVER-BRAIN DISSOCIATION (CRITICAL CONCEPT): "
            "  Liver transplant: corrects hepatic urea cycle → no more ammonia crises; "
            "  BUT: neurological disease persists/progresses post-LTx; "
            "  ASL is expressed in neurons/brain; LTx does NOT normalize CNS arginine or fumarate; "
            "  SYSTEMIC HYPERTENSION: 75% of treated ASL patients regardless of ammonia control; "
            "  Mechanism: nitric oxide synthase requires arginine → NO deficiency → vascular stiffness; "
            "  ARGININE SUPPLEMENTATION: corrects NO deficiency; improves hypertension; "
            "TREATMENT: "
            "  Arginine: 400-700 mg/kg/day CRITICAL (both UCD support + NO synthesis); "
            "  Nitrogen scavengers: sodium benzoate + RAVICTI; "
            "  Low-protein diet with EAA supplement; "
            "  Anti-hypertensives if arginine supplementation alone insufficient; "
            "  Liver transplant: reduces ammonia crises but NOT neuroprotective — does NOT cure"
        ),
        "syndrome": (
            "Argininosuccinic-aciduria-ASA / AR-ASL-deficiency / "
            "argininosuccinic-acid-plasma-urine-PATHOGNOMONIC / "
            "trichorrhexis-nodosa-PATHOGNOMONIC / "
            "liver-brain-dissociation-LTx-corrects-ammonia-NOT-neuro / "
            "systemic-hypertension-75pct / arginine-supplement-mandatory"
        ),
        "inheritance": "AR (biallelic ASL mutations)",
        "key_therapy": "Arginine 400-700 mg/kg/day (CRITICAL for NO synthesis) — nitrogen scavengers — anti-hypertensives",
        "pathognomonic": (
            "Argininosuccinic acid elevated in plasma AND urine PATHOGNOMONIC; "
            "Trichorrhexis nodosa (brittle hair, nodes on microscopy) PATHOGNOMONIC; "
            "Systemic hypertension 75% despite good ammonia control; "
            "Liver-brain dissociation: LTx corrects UCD but neurological disease persists"
        ),
        "key_drugs_ci": [
            "Arginine-400-700mgkgday-CRITICAL-NO-synthesis",
            "Sodium-benzoate-phenylbutyrate-nitrogen-scavengers",
            "Anti-hypertensives-arginine-first-line",
            "LTx-reduces-crises-NOT-neuroprotective",
            "Low-protein-EAA-supplement",
            "FASTING-ABSOLUTELY-AVOID",
        ],
    },
    {
        "gene": "ARG1",
        "protein": (
            "ARG1 -- 6q23.2 AR LOF -- 322aa -- "
            "Arginase-1-cytosolic-liver-hepatocytes-final-step-urea-cycle-"
            "hyperargininemia-argininemia-"
            "MILD-hyperammonemia-progressive-spastic-diplegia-PATHOGNOMONIC-"
            "plasma-arginine-gt200-micromolL-diagnostic-"
            "low-arginine-diet-prevents-neurological-progression"
        ),
        "locus": "6q23.2",
        "protein_size": (
            "322 aa / ~35 kDa / 6q23.2 ARG1 arginase 1 molecular context: "
            "STRUCTURE: "
            "  322 aa; cytosolic homotrimer; final step of urea cycle: arginine -> ornithine + urea; "
            "  Expressed predominantly in hepatocytes (ARG1 = liver arginase); "
            "  Deficiency -> arginine accumulates; ammonia elevation is MILD vs other UCDs; "
            "  AR inheritance; biallelic ARG1 mutations; "
            "HYPERARGININEMIA — UNIQUE PRESENTATION: "
            "  DOES NOT cause severe neonatal hyperammonemia (KEY DIFFERENCE from other UCDs); "
            "  Ammonia mildly elevated; arginine is directly toxic to CNS (not primarily ammonia); "
            "  Onset: 1-3 years; initially appear normal in neonatal period; "
            "NEUROLOGICAL PRESENTATION (PATHOGNOMONIC): "
            "  Progressive SPASTIC DIPLEGIA (spasticity > ataxia); "
            "  Intellectual disability progressing over years; "
            "  Seizures (often generalized); "
            "  Loss of milestones if untreated; "
            "  Spastic diplegia in toddler with prior normal neonatal screen = think ARG1; "
            "BIOCHEMISTRY: "
            "  Plasma arginine: >200 μmol/L (diagnostic; normal <120 μmol/L); "
            "    Often 400-800 μmol/L in symptomatic patients; "
            "  Guanidino compounds (homoarginine, alpha-keto-delta-guanidinovaleric acid) elevated; "
            "  Plasma ammonia: mildly elevated (NOT the severe crises of OTC/ASS1/ASL); "
            "  Orotic acid: normal or mildly elevated; citrulline: normal; "
            "  ARG1 enzyme activity in RBCs: absent/very low (ARG1 expressed in RBCs); "
            "TREATMENT: "
            "  Low-arginine diet: PRIMARY TREATMENT; restricts arginine intake; "
            "    Essential amino acid mix WITHOUT arginine; "
            "    Prevents or halts neurological progression if started early; "
            "  Nitrogen scavengers: less critical than other UCDs (mild hyperammonemia); "
            "  Liver transplant: CONTROVERSIAL; arginine still elevated post-LTx; "
            "    LTx corrects liver arginase deficiency but brain ARG2 (mitochondrial) "
            "    may contribute to residual neurological toxicity; "
            "    LTx does NOT reliably reverse neurological damage"
        ),
        "syndrome": (
            "Hyperargininemia-argininemia / AR-ARG1-deficiency / "
            "MILD-hyperammonemia-NOT-severe-neonatal-KEY-DISTINCTION / "
            "progressive-spastic-diplegia + intellectual-disability-PATHOGNOMONIC / "
            "plasma-arginine-gt200-micromolL-diagnostic / "
            "low-arginine-diet-prevents-progression"
        ),
        "inheritance": "AR (biallelic ARG1 mutations)",
        "key_therapy": "Low-arginine diet (EAA mix without arginine) — prevents neurological progression — LTx controversial",
        "pathognomonic": (
            "Progressive spastic diplegia + intellectual disability PATHOGNOMONIC in UCD context; "
            "Plasma arginine >200 μmol/L diagnostic (vs normal <120 μmol/L); "
            "MILD ammonia elevation (not severe neonatal crisis) KEY DISTINCTION from OTC/ASS1; "
            "RBC arginase 1 activity absent/low (ARG1 expressed in RBCs)"
        ),
        "key_drugs_ci": [
            "Low-arginine-EAA-mix-WITHOUT-arginine-PRIMARY-treatment",
            "Nitrogen-scavengers-less-critical-mild-NH3",
            "Sodium-benzoate-phenylbutyrate-if-hyperammonemia",
            "LTx-controversial-neurological-may-persist",
            "Guanidino-compounds-toxic-arginine-DIRECT-CNS-toxicity",
            "PROTEIN-EXCESS-ABSOLUTELY-AVOID",
        ],
    },
    {
        "gene": "CPS1",
        "protein": (
            "CPS1 -- 2q34 AR LOF -- 1500aa -- "
            "Carbamoyl-phosphate-synthetase-1-mitochondrial-first-step-urea-cycle-"
            "most-severe-neonatal-UCD-"
            "citrulline-VERY-LOW-orotic-acid-NORMAL-PATHOGNOMONIC-"
            "N-acetylglutamate-NAG-activates-CPS1-"
            "carglumic-acid-NOT-effective-DISTINGUISH-from-NAGS"
        ),
        "locus": "2q34",
        "protein_size": (
            "1500 aa / ~165 kDa / 2q34 CPS1 carbamoyl phosphate synthetase 1 molecular context: "
            "STRUCTURE: "
            "  1500 aa (largest urea cycle enzyme); mitochondrial matrix; "
            "  Catalyzes: NH3 + CO2 + 2ATP -> carbamoyl phosphate (first step of urea cycle); "
            "  Requires N-acetylglutamate (NAG) as obligate ALLOSTERIC ACTIVATOR; "
            "  NAG produced by NAGS enzyme; "
            "  AR inheritance; biallelic CPS1 mutations; "
            "  Very large gene (>250 exons); many private variants; "
            "CPS1 DEFICIENCY — SEVERE NEONATAL: "
            "  Among the most severe of UCDs; "
            "  Onset: first 24-48 hr; catastrophic hyperammonemia; "
            "  NH3 >2000 μmol/L common; "
            "  HIGH mortality without immediate recognition + treatment; "
            "BIOCHEMISTRY: "
            "  Plasma citrulline: VERY LOW (<5 μmol/L); citrulline production blocked at step 1; "
            "  Urine orotic acid: NORMAL (no carbamoyl phosphate surplus to shunt); "
            "  Plasma ammonia: severely elevated; "
            "  CITRULLINE LOW + OROTIC ACID NORMAL = CPS1 or NAGS pattern; "
            "  DISTINGUISHING CPS1 vs NAGS: carglumic acid response test; "
            "    NAGS: responds dramatically to carglumic acid (see NAGS entry); "
            "    CPS1: NO RESPONSE to carglumic acid; "
            "CPS1 vs OTC DISTINCTION: "
            "  OTC: citrulline low + orotic acid HIGH; "
            "  CPS1: citrulline low + orotic acid NORMAL; "
            "TREATMENT: "
            "  Acute: hemodialysis + IV dextrose (10%) + nitrogen scavengers; "
            "  Nitrogen scavengers: sodium benzoate + phenylbutyrate (RAVICTI); "
            "  Citrulline supplementation 200 mg/kg/day; "
            "  Low-protein diet with EAA; "
            "  Carglumic acid: NOT EFFECTIVE (CPS1 is absent; activator analogue cannot substitute); "
            "  Liver transplant: CURATIVE (restores hepatic urea cycle); "
            "    Due to severity, early LTx (6-12 months) often planned"
        ),
        "syndrome": (
            "CPS1-deficiency / AR-LOF / most-severe-neonatal-UCD / "
            "citrulline-VERY-LOW + orotic-acid-NORMAL-PATHOGNOMONIC / "
            "carglumic-acid-NOT-effective-DISTINGUISH-from-NAGS / "
            "N-acetylglutamate-activates-CPS1 / liver-transplant-CURATIVE"
        ),
        "inheritance": "AR (biallelic CPS1 mutations)",
        "key_therapy": "Hemodialysis acute + nitrogen scavengers + citrulline — LTx CURATIVE — NO carglumic acid benefit",
        "pathognomonic": (
            "Citrulline VERY LOW + orotic acid NORMAL PATHOGNOMONIC pattern (CPS1/NAGS); "
            "Carglumic acid NO RESPONSE distinguishes CPS1 from NAGS; "
            "Plasma ammonia >2000 μmol/L neonatal crisis; "
            "Largest urea cycle enzyme (1500 aa); many private variants; gene panel mandatory"
        ),
        "key_drugs_ci": [
            "Hemodialysis-NH3-gt500-urgent",
            "Sodium-benzoate-phenylbutyrate-RAVICTI-nitrogen-scavengers",
            "Citrulline-200mgkgday",
            "Carglumic-acid-NOT-EFFECTIVE-CPS1-DISTINGUISH-from-NAGS",
            "Liver-transplant-CURATIVE-early-6-12mo",
            "Low-protein-EAA-supplement",
        ],
    },
    {
        "gene": "NAGS",
        "protein": (
            "NAGS -- 17q21.31 AR LOF -- 534aa -- "
            "N-acetylglutamate-synthase-mitochondrial-matrix-enzyme-"
            "activates-CPS1-via-NAG-allosteric-"
            "ONLY-UCD-carglumic-acid-Carbaglu-FDA2010-CURATIVE-"
            "citrulline-low-orotic-acid-normal-biochemically-identical-CPS1-"
            "NAGS-deficiency-diagnosis-carglumic-acid-RESPONSE-TEST-PATHOGNOMONIC"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "534 aa / ~58 kDa / 17q21.31 NAGS N-acetylglutamate synthase molecular context: "
            "STRUCTURE: "
            "  534 aa; mitochondrial matrix; "
            "  Catalyzes: acetyl-CoA + glutamate -> N-acetylglutamate (NAG); "
            "  NAG is the obligate allosteric activator of CPS1 (first urea cycle step); "
            "  NAGS deficiency -> no NAG -> CPS1 inactive -> no carbamoyl phosphate -> hyperammonemia; "
            "  AR inheritance; biallelic NAGS mutations; "
            "  Rare: <200 reported patients worldwide; "
            "NAGS DEFICIENCY: "
            "  Neonatal to adult onset (variable); "
            "  Hyperammonemia; acute encephalopathy; "
            "  BIOCHEMICALLY IDENTICAL TO CPS1: citrulline low + orotic acid normal; "
            "  ONLY WAY TO DISTINGUISH FROM CPS1: carglumic acid response test; "
            "CARGLUMIC ACID (N-CARBAMOYLGLUTAMATE / CARBAGLU) — CURATIVE FOR NAGS ONLY: "
            "  Carbaglu (carglumic acid): FDA2010 approved for NAGS deficiency; "
            "  Structural analogue of NAG; activates CPS1 DIRECTLY bypassing NAGS defect; "
            "  In NAGS deficiency: carglumic acid restores CPS1 activity -> urea cycle normalizes; "
            "  DRAMATIC RESPONSE: ammonia normalizes within 24-48 hr on carglumic acid; "
            "  LONG-TERM: carglumic acid monotherapy sufficient in many patients (no protein restriction needed); "
            "  IN CPS1 DEFICIENCY: NO RESPONSE (CPS1 absent; NAG analogue cannot substitute); "
            "  IN OTC DEFICIENCY: PARTIAL RESPONSE (not curative; OTC block downstream); "
            "DIAGNOSIS: "
            "  Citrulline low + orotic acid normal (identical to CPS1); "
            "  NAGS enzyme activity (in fibroblasts) or NAGS gene sequencing; "
            "  Carglumic acid therapeutic trial: positive response PATHOGNOMONIC for NAGS; "
            "TREATMENT: "
            "  Acute: carglumic acid 100-250 mg/kg/day (IV or oral) -> often rapid response; "
            "  Chronic: carglumic acid maintenance (dose titrated to ammonia levels); "
            "  Nitrogen scavengers: rarely needed long-term if carglumic acid effective; "
            "  Low-protein diet: may be liberalized when carglumic acid controls NH3; "
            "  Liver transplant: usually AVOIDED (carglumic acid is curative); "
            "  UNIQUE: NAGS is the ONLY UCD where a specific pharmacological CURE exists (carglumic acid)"
        ),
        "syndrome": (
            "NAGS-deficiency / AR-LOF / "
            "citrulline-low + orotic-acid-normal-SAME-AS-CPS1 / "
            "ONLY-UCD-carglumic-acid-Carbaglu-FDA2010-CURATIVE / "
            "NAG-analogue-activates-CPS1-directly / "
            "carglumic-acid-response-PATHOGNOMONIC-for-NAGS"
        ),
        "inheritance": "AR (biallelic NAGS mutations)",
        "key_therapy": "Carglumic acid (Carbaglu FDA2010) — CURATIVE for NAGS ONLY — monotherapy sufficient long-term",
        "pathognomonic": (
            "Carglumic acid dramatic response (NH3 normalizes 24-48 hr) PATHOGNOMONIC for NAGS; "
            "Citrulline LOW + orotic acid NORMAL (biochemically identical to CPS1 — only carglumic acid distinguishes); "
            "ONLY UCD with pharmacological cure (carglumic acid bypasses NAGS defect entirely); "
            "No response to carglumic acid = CPS1 (not NAGS)"
        ),
        "key_drugs_ci": [
            "Carglumic-acid-Carbaglu-FDA2010-100-250mgkgday-CURATIVE",
            "ONLY-NAGS-responds-to-carglumic-acid-KEY-DISTINCTION",
            "Nitrogen-scavengers-rarely-needed-if-carglumic-acid-effective",
            "Liver-transplant-usually-AVOIDED-pharmacological-cure",
            "Low-protein-can-be-liberalized-on-carglumic-acid",
            "CPS1-NO-RESPONSE-OTC-partial-response",
        ],
    },
    {
        "gene": "SLC25A13",
        "protein": (
            "SLC25A13 -- 7q21.3 AR LOF -- 675aa -- "
            "Citrin-mitochondrial-aspartate-glutamate-carrier-2-AGC2-"
            "citrin-deficiency-citrullinemia-type-II-CTLN2-"
            "Japanese-founder-exon16-ins3kb-East-Asian-"
            "adult-onset-acute-liver-failure-hyperammonemia-"
            "alcohol-high-carbohydrate-ABSOLUTELY-TRIGGER-PATHOGNOMONIC-"
            "low-carb-high-fat-protein-diet-liver-transplant-CURATIVE"
        ),
        "locus": "7q21.3",
        "protein_size": (
            "675 aa / ~74 kDa / 7q21.3 SLC25A13 citrin molecular context: "
            "STRUCTURE: "
            "  675 aa; inner mitochondrial membrane aspartate-glutamate carrier 2 (AGC2); "
            "  Shuttles aspartate out and glutamate into mitochondria; "
            "  Essential for malate-aspartate shuttle (NADH regeneration) + urea cycle (aspartate provision); "
            "  Deficiency: aspartate depleted in cytosol; urea cycle impaired; "
            "  AR inheritance; biallelic SLC25A13 mutations; "
            "JAPANESE FOUNDER MUTATIONS: "
            "  c.851_854del (IVS16ins3kb, p.Thr290Leufs): most common, ~85% Japanese alleles; "
            "  Also common: c.1177+1G>A, c.615+5G>A; "
            "  HIGH PREVALENCE in East Asian populations (Japan, China, Korea, Taiwan); "
            "  Carrier frequency ~1/65 in Japan; "
            "  RARE in non-East Asian populations; "
            "DISEASE PRESENTATIONS — THREE DISTINCT PHASES: "
            "  1) NEONATAL INTRAHEPATIC CHOLESTASIS (NICCD, neonates): "
            "    Cholestasis, elevated citrulline, galactosemia-like; usually resolves by 1 year; "
            "    Majority improve with diet (MCT formula, lactose restriction); "
            "  2) FAILURE TO THRIVE + FATTY LIVER (ages 1-10): "
            "    Hepatomegaly, dyslipidemia, fatigue; "
            "  3) CITRULLINEMIA TYPE II (CTLN2, adults): "
            "    ACUTE EPISODES: hyperammonemia + hepatic encephalopathy; "
            "    TRIGGERED: ALCOHOL consumption (ABSOLUTELY); high-carbohydrate diet; fasting; surgery; "
            "    Behavioral changes, disorientation, coma; "
            "    SOME ADULT PATIENTS: CRAVE protein/fat-rich foods (instinct to avoid carbohydrates); "
            "PATHOGNOMONIC TRIGGERS: "
            "  ALCOHOL: hepatic citrullinemia type II MOST RELIABLY triggered by alcohol ingestion; "
            "    Even moderate alcohol -> acute hyperammonemic encephalopathy; "
            "  HIGH-CARBOHYDRATE DIET: glucose/fructose excess worsens citrin-dependent shuttle defect; "
            "  NIGHT SHIFT WORK + SLEEP DEPRIVATION: reported triggers; "
            "  IV dextrose infusions (hospital) can precipitate crisis — ALERT anesthesiology/ICU; "
            "DIAGNOSIS: "
            "  Plasma citrulline elevated (adult-onset context); "
            "  Ammonia elevated during crisis; thrombocytopenia; coagulopathy; "
            "  SLC25A13 gene sequencing; "
            "  Liver biopsy: steatosis, focal necrosis, glycogenosis; "
            "TREATMENT: "
            "  DIET: low-carbohydrate high-fat high-protein (MCT-enriched); "
            "    Medium-chain triglycerides provide energy without worsening malate-aspartate shuttle; "
            "  Sodium pyruvate: investigational (malate-aspartate shuttle bypass); "
            "  Arginine supplementation: useful in acute crisis; "
            "  AVOID: glucose infusions, alcohol, high-carbohydrate meals; "
            "  Liver transplant: CURATIVE (corrects citrin deficiency in liver); "
            "    Most definitive treatment for recurrent CTLN2 adults; "
            "  LIVER DONOR SCREENING: citrin deficiency carriers should be excluded as donors"
        ),
        "syndrome": (
            "Citrullinemia-type-II-CTLN2-citrin-deficiency / AR-SLC25A13-LOF / "
            "Japanese-founder-exon16-ins3kb-East-Asian-prevalence / "
            "adult-onset-hyperammonemia-hepatic-encephalopathy / "
            "alcohol-high-carbohydrate-ABSOLUTELY-TRIGGER-PATHOGNOMONIC / "
            "low-carb-MCT-diet / liver-transplant-CURATIVE"
        ),
        "inheritance": "AR (biallelic SLC25A13 mutations) — Japanese/East Asian founder",
        "key_therapy": "Low-carbohydrate MCT-enriched diet — AVOID alcohol + glucose infusions — LTx CURATIVE",
        "pathognomonic": (
            "Acute hyperammonemia triggered by ALCOHOL or high-carbohydrate diet PATHOGNOMONIC for CTLN2; "
            "Protein/fat food craving (avoidance of carbohydrates) behavioral marker; "
            "Japanese/East Asian ancestry (founder c.851_854del ~85% alleles); "
            "IV dextrose in hospital can precipitate crisis — alert ICU/anesthesia"
        ),
        "key_drugs_ci": [
            "Low-carbohydrate-MCT-high-fat-protein-diet-PRIMARY",
            "Arginine-supplementation-acute-crisis",
            "ALCOHOL-ABSOLUTELY-CI-triggers-acute-crisis",
            "IV-DEXTROSE-AVOID-precipitates-crisis",
            "High-carbohydrate-diet-ABSOLUTELY-AVOID",
            "Liver-transplant-CURATIVE-recurrent-CTLN2",
        ],
    },
    {
        "gene": "SLC25A15",
        "protein": (
            "SLC25A15 -- 13q14.11 AR LOF -- 301aa -- "
            "Ornithine-carrier-1-ORNT1-mitochondrial-inner-membrane-"
            "HHH-syndrome-hyperornithinemia-hyperammonemia-homocitrullinuria-"
            "homocitrulline-urine-PATHOGNOMONIC-"
            "progressive-spastic-paraparesis-cognitive-decline-"
            "ornithine-restriction-citrulline-supplement"
        ),
        "locus": "13q14.11",
        "protein_size": (
            "301 aa / ~33 kDa / 13q14.11 SLC25A15 ornithine carrier 1 (ORNT1) molecular context: "
            "STRUCTURE: "
            "  301 aa; mitochondrial inner membrane carrier; "
            "  Transports ornithine into mitochondria in exchange for citrulline (out); "
            "  Deficiency: ornithine cannot enter mitochondria; accumulates in cytosol (hyperornithinemia); "
            "    Without ornithine in mitochondria: OTC substrate absent -> carbamoyl phosphate shunted; "
            "    -> partial urea cycle defect -> hyperammonemia; "
            "    -> homocitrulline: carbamoyl phosphate + lysine (alternative) -> homocitrulline in urine; "
            "  AR inheritance; biallelic SLC25A15 mutations; "
            "  p.Phe188del: common founder mutation in Italian + Franco-Canadian populations; "
            "HHH SYNDROME — TRIAD: "
            "  1) HYPERORNITHINEMIA: plasma ornithine markedly elevated (>200 μmol/L; normal 40-70); "
            "  2) HYPERAMMONEMIA: episodic; less severe than proximal UCDs; "
            "  3) HOMOCITRULLINURIA: homocitrulline in urine = PATHOGNOMONIC; "
            "    Homocitrulline is NOT present in OTC deficiency — key distinction; "
            "CLINICAL: "
            "  Onset: neonatal to adult (variable); "
            "  PROGRESSIVE SPASTIC PARAPARESIS: hallmark neurological feature; "
            "    Upper motor neuron: spasticity, hyperreflexia, Babinski positive; "
            "  Intellectual disability (variable); cognitive decline; "
            "  Behavioral abnormalities; psychiatric symptoms; "
            "  Seizures (less common than spasticity); "
            "  Hepatopathy in some patients; "
            "  Thromboembolic complications: coagulopathy + portal hypertension reported; "
            "LYSINURIC PROTEIN INTOLERANCE (LPI) DDX: "
            "  LPI (SLC7A7): also has hyperammonemia + orotic acid elevated; "
            "  LPI: LOW urine lysine/arginine/ornithine (dibasic aminoaciduria); "
            "  HHH: HIGH urine ornithine + homocitrulline; "
            "DIAGNOSIS: "
            "  Plasma ornithine elevated; urine homocitrulline present (PATHOGNOMONIC); "
            "  Plasma ammonia elevated (episodic); urine orotic acid elevated; "
            "  SLC25A15 gene sequencing; "
            "TREATMENT: "
            "  Low-protein / ornithine-restricted diet: primary; limits ornithine precursor intake; "
            "  Citrulline supplementation: provides intramitochondrial substrate alternative; "
            "  Pyridoxine (B6): improves mitochondrial ornithine metabolism; "
            "  Nitrogen scavengers: sodium benzoate if hyperammonemia; "
            "  Liver transplant: controversial; some neurological improvement reported but inconsistent"
        ),
        "syndrome": (
            "HHH-syndrome-hyperornithinemia-hyperammonemia-homocitrullinuria / AR-SLC25A15-LOF / "
            "homocitrulline-urine-PATHOGNOMONIC / "
            "progressive-spastic-paraparesis + cognitive-decline / "
            "ornithine-restriction + citrulline-supplement / "
            "Italian-Franco-Canadian-p.Phe188del-founder"
        ),
        "inheritance": "AR (biallelic SLC25A15 mutations) — Italian/Franco-Canadian founder p.Phe188del",
        "key_therapy": "Ornithine-restricted diet + citrulline supplementation + pyridoxine — nitrogen scavengers for hyperammonemia",
        "pathognomonic": (
            "Homocitrulline in urine PATHOGNOMONIC for HHH syndrome; "
            "Progressive spastic paraparesis (UMN) in UCD context distinguishes HHH; "
            "Plasma ornithine >200 μmol/L + hyperammonemia + homocitrullinuria = diagnostic TRIAD; "
            "Distinguish from LPI (SLC7A7): HHH has HIGH urine ornithine; LPI has LOW dibasic AA"
        ),
        "key_drugs_ci": [
            "Ornithine-restricted-low-protein-diet-PRIMARY",
            "Citrulline-supplementation-intramitochondrial-bypass",
            "Pyridoxine-B6-supplement",
            "Sodium-benzoate-nitrogen-scavenger-if-hyperammonemia",
            "LTx-controversial-inconsistent-neurological-benefit",
            "PROTEIN-EXCESS-ABSOLUTELY-AVOID",
        ],
    },
]


def _make_cohort(gene_entry: dict, seed: int) -> dict:
    rng = random.Random(seed)
    gene = gene_entry["gene"]

    n = 40
    inh = gene_entry["inheritance"]

    # Age at diagnosis varies by UCD type
    if gene in ("OTC", "ASS1", "CPS1", "NAGS"):
        # Severe neonatal UCDs
        mean_age = round(rng.uniform(0.0, 0.2), 2)   # days to weeks
        sd_age   = round(rng.uniform(0.0, 0.1), 1)
    elif gene == "ASL":
        mean_age = round(rng.uniform(0.0, 1.5), 2)    # neonatal to late
        sd_age   = round(rng.uniform(0.2, 0.8), 1)
    elif gene == "ARG1":
        mean_age = round(rng.uniform(1.5, 3.5), 1)    # toddler/childhood
        sd_age   = round(rng.uniform(0.5, 1.5), 1)
    elif gene == "SLC25A13":
        mean_age = round(rng.uniform(20.0, 45.0), 1)  # adult onset CTLN2
        sd_age   = round(rng.uniform(5.0, 15.0), 1)
    elif gene == "SLC25A15":
        mean_age = round(rng.uniform(2.0, 15.0), 1)   # variable onset
        sd_age   = round(rng.uniform(2.0, 8.0), 1)
    else:
        mean_age = round(rng.uniform(0.1, 5.0), 1)
        sd_age   = round(rng.uniform(0.5, 2.0), 1)

    # Hyperammonemia crisis
    if gene in ("OTC", "ASS1", "CPS1", "NAGS", "ASL"):
        hap_n = rng.randint(32, 40)
    elif gene == "SLC25A13":
        hap_n = rng.randint(25, 38)
    elif gene == "ARG1":
        hap_n = rng.randint(10, 22)  # mild in ARG1
    else:
        hap_n = rng.randint(18, 32)

    # Neonatal presentation
    if gene in ("OTC", "ASS1", "CPS1", "NAGS"):
        neonatal_n = rng.randint(28, 38)
    elif gene in ("ASL",):
        neonatal_n = rng.randint(15, 30)
    elif gene in ("ARG1", "SLC25A13", "SLC25A15"):
        neonatal_n = rng.randint(0, 8)
    else:
        neonatal_n = rng.randint(5, 20)

    # Neurological impairment
    if gene in ("OTC", "CPS1", "NAGS"):
        neuro_n = rng.randint(20, 36)
    elif gene in ("ASS1",):
        neuro_n = rng.randint(15, 30)
    elif gene in ("ASL",):
        neuro_n = rng.randint(22, 38)  # liver-brain dissociation
    elif gene == "ARG1":
        neuro_n = rng.randint(28, 38)  # progressive spastic diplegia
    elif gene == "SLC25A13":
        neuro_n = rng.randint(20, 35)
    elif gene == "SLC25A15":
        neuro_n = rng.randint(25, 38)  # spastic paraparesis
    else:
        neuro_n = rng.randint(15, 30)

    # Liver transplant performed
    if gene in ("OTC", "ASS1", "CPS1"):
        ltx_n = rng.randint(20, 35)
    elif gene in ("NAGS",):
        ltx_n = rng.randint(2, 8)  # rarely needed (carglumic acid)
    elif gene in ("ASL",):
        ltx_n = rng.randint(12, 25)
    elif gene == "SLC25A13":
        ltx_n = rng.randint(15, 30)
    elif gene in ("ARG1",):
        ltx_n = rng.randint(5, 18)
    elif gene == "SLC25A15":
        ltx_n = rng.randint(3, 12)
    else:
        ltx_n = rng.randint(8, 22)

    # Nitrogen scavenger treatment
    if gene in ("OTC", "ASS1", "ASL", "CPS1"):
        nsv_n = rng.randint(30, 40)
    elif gene == "NAGS":
        nsv_n = rng.randint(5, 15)  # carglumic acid preferred
    elif gene == "ARG1":
        nsv_n = rng.randint(10, 22)
    else:
        nsv_n = rng.randint(18, 35)

    # Carglumic acid treatment (NAGS specific)
    carglumic_n = rng.randint(28, 38) if gene == "NAGS" else rng.randint(0, 3)

    # Spastic paraparesis/diplegia
    if gene == "ARG1":
        spastic_n = rng.randint(28, 38)
    elif gene == "SLC25A15":
        spastic_n = rng.randint(22, 35)
    elif gene == "ASL":
        spastic_n = rng.randint(10, 20)
    else:
        spastic_n = rng.randint(2, 12)

    # Pathognomonic biochemical marker
    if gene == "OTC":
        patho_n = rng.randint(35, 40)  # low citrulline + high orotic acid
    elif gene == "ASS1":
        patho_n = rng.randint(36, 40)  # citrulline >1000
    elif gene == "ASL":
        patho_n = rng.randint(35, 40)  # argininosuccinic acid elevated
    elif gene == "ARG1":
        patho_n = rng.randint(33, 40)  # arginine >200
    elif gene == "CPS1":
        patho_n = rng.randint(30, 38)  # citrulline low + orotic normal
    elif gene == "NAGS":
        patho_n = rng.randint(28, 36)  # carglumic acid response
    elif gene == "SLC25A13":
        patho_n = rng.randint(25, 35)  # alcohol/carb trigger
    elif gene == "SLC25A15":
        patho_n = rng.randint(30, 38)  # homocitrullinuria
    else:
        patho_n = rng.randint(20, 35)

    # Female carriers symptomatic (OTC X-linked)
    female_symptomatic_n = rng.randint(8, 20) if gene == "OTC" else 0

    # Alcohol trigger (CTLN2)
    alcohol_trigger_n = rng.randint(28, 38) if gene == "SLC25A13" else rng.randint(0, 2)

    # Hypertension (ASL specific)
    hypertension_n = rng.randint(28, 35) if gene == "ASL" else rng.randint(2, 8)

    return {
        "gene": gene,
        "n": n,
        "seed": seed,
        "mean_age_dx": mean_age,
        "sd_age_dx": sd_age,
        "hyperammonemia_crisis_n": hap_n,
        "neonatal_presentation_n": neonatal_n,
        "neurological_n": neuro_n,
        "liver_transplant_n": ltx_n,
        "nitrogen_scavenger_n": nsv_n,
        "carglumic_acid_n": carglumic_n,
        "spastic_paraparesis_n": spastic_n,
        "pathognomonic_marker_n": patho_n,
        "female_symptomatic_n": female_symptomatic_n,
        "alcohol_trigger_n": alcohol_trigger_n,
        "hypertension_n": hypertension_n,
        "inheritance": inh,
        "key_therapy": gene_entry["key_therapy"],
        "pathognomonic": gene_entry["pathognomonic"],
        "syndrome": gene_entry["syndrome"],
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    total = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Urea-Cycle-Disorder-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "description": (
            "Complete 8-gene Hereditary Urea Cycle Disorder reference atlas. "
            "OTC (X-linked most common UCD; citrulline low + orotic acid high PATHOGNOMONIC; "
            "nitrogen scavengers; LTx curative; fasting ABSOLUTELY AVOID). "
            "ASS1 (CTLN1; citrulline >1000 μmol/L PATHOGNOMONIC; arginine mandatory). "
            "ASL (argininosuccinic aciduria; argininosuccinic acid PATHOGNOMONIC; "
            "liver-brain dissociation; hypertension 75%; arginine CRITICAL). "
            "ARG1 (hyperargininemia; MILD ammonia; spastic diplegia PATHOGNOMONIC; "
            "arginine >200 μmol/L; low-arginine diet KEY). "
            "CPS1 (most severe neonatal; citrulline low + orotic acid NORMAL; "
            "carglumic acid NOT effective — DISTINGUISH from NAGS). "
            "NAGS (ONLY UCD where carglumic acid/Carbaglu FDA2010 is CURATIVE; "
            "biochemically identical to CPS1; carglumic acid response PATHOGNOMONIC). "
            "SLC25A13 (citrin CTLN2; Japanese founder; adult onset; "
            "alcohol + high-carbohydrate ABSOLUTELY TRIGGER; MCT diet; LTx curative). "
            "SLC25A15 (HHH syndrome; homocitrulline urine PATHOGNOMONIC; "
            "spastic paraparesis; ornithine restriction + citrulline supplement). "
            "320-patient aggregate, 8 x 40 patients, seeds 3806-3813."
        ),
        "gene_summaries": [
            {
                "gene": g["gene"],
                "syndrome": g["syndrome"],
                "key_therapy": g["key_therapy"],
                "pathognomonic": g["pathognomonic"],
                "inheritance": g["inheritance"],
                "locus": g["locus"],
                "protein_size": g["protein_size"].split("molecular context")[0].strip(),
            }
            for g in ATLAS_GENES
        ],
        "cohort_summary": {
            c["gene"]: {
                "n": c["n"],
                "mean_age_dx_yr": c["mean_age_dx"],
                "hyperammonemia_crisis_n": c["hyperammonemia_crisis_n"],
                "liver_transplant_n": c["liver_transplant_n"],
                "neurological_n": c["neurological_n"],
            }
            for c in cohorts
        },
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    return {
        "atlas": "Hereditary-Urea-Cycle-Disorder-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_data": {c["gene"]: c for c in cohorts},
        "atlas_genes": ATLAS_GENES,
        "total_patients": sum(c["n"] for c in cohorts),
    }


def generate_definitions():
    return {
        "atlas": "Hereditary-Urea-Cycle-Disorder-Atlas",
        "definitions": {
            "Urea cycle": (
                "The urea cycle converts toxic ammonia (NH3) to urea for excretion; "
                "5 hepatic enzymes + 2 mitochondrial transporters: "
                "CPS1 (step 1) → OTC (step 2) → ASS1 (step 3) → ASL (step 4) → ARG1 (step 5); "
                "NAGS produces NAG (obligate activator of CPS1); "
                "SLC25A13 (citrin): aspartate-glutamate carrier for urea cycle substrate supply; "
                "SLC25A15 (ORNT1): ornithine transporter into mitochondria; "
                "Combined incidence: ~1:35,000 births"
            ),
            "OTC deficiency": (
                "X-linked most common UCD; hemizygous males: catastrophic neonatal hyperammonemia; "
                "BIOCHEMISTRY: citrulline LOW + urine orotic acid ELEVATED PATHOGNOMONIC; "
                "Female carriers: 20-30% symptomatic (migraine, protein aversion, episodic encephalopathy); "
                "Nitrogen scavengers (RAVICTI/glycerol phenylbutyrate FDA2013) + citrulline supplement; "
                "Liver transplant CURATIVE; FASTING + CATABOLISM ABSOLUTELY AVOID"
            ),
            "Citrullinemia type 1 (CTLN1 / ASS1)": (
                "AR ASS1 deficiency; third step of urea cycle; citrulline accumulates massively; "
                "Plasma citrulline >1000 μmol/L PATHOGNOMONIC (normal <40 μmol/L); "
                "Urine orotic acid NORMAL (distinguishes from OTC); "
                "Arginine supplementation 300-600 mg/kg/day MANDATORY; "
                "Liver transplant CURATIVE; NBS: citrulline >150 μmol/L triggers tier-2"
            ),
            "Argininosuccinic aciduria (ASA / ASL)": (
                "AR ASL deficiency; step 4 of urea cycle; argininosuccinate accumulates; "
                "Argininosuccinic acid in plasma + urine = PATHOGNOMONIC; "
                "LIVER-BRAIN DISSOCIATION: liver transplant corrects ammonia crises but NOT neurological disease; "
                "Systemic hypertension in 75% even with good ammonia control (nitric oxide deficiency); "
                "Arginine supplementation CRITICAL for NO synthesis; "
                "Trichorrhexis nodosa (brittle hair) PATHOGNOMONIC for ASL"
            ),
            "Hyperargininemia (ARG1)": (
                "AR ARG1 deficiency; final step of urea cycle; arginine accumulates; "
                "UNIQUE: MILD hyperammonemia (unlike severe neonatal of other UCDs); "
                "Clinical: progressive spastic diplegia + intellectual disability PATHOGNOMONIC; "
                "Plasma arginine >200 μmol/L diagnostic; "
                "Low-arginine diet PRIMARY treatment — prevents neurological progression; "
                "Liver transplant controversial (neurological may persist)"
            ),
            "CPS1 deficiency": (
                "AR; first step of urea cycle; most severe neonatal onset; "
                "NH3 often >2000 μmol/L at presentation; "
                "Citrulline VERY LOW + orotic acid NORMAL (identical to NAGS); "
                "Carglumic acid NOT EFFECTIVE (CPS1 enzyme absent; NAG analogue cannot substitute); "
                "CRITICAL DISTINCTION FROM NAGS: no carglumic acid response; "
                "Liver transplant CURATIVE; gene panel: >250 exons, many private variants"
            ),
            "NAGS deficiency + carglumic acid": (
                "AR; produces NAG, obligate CPS1 activator; "
                "Biochemically IDENTICAL to CPS1 (citrulline low + orotic acid normal); "
                "ONLY UCD with pharmacological cure: carglumic acid (N-carbamoylglutamate, Carbaglu FDA2010); "
                "Carglumic acid activates CPS1 DIRECTLY bypassing NAGS defect; "
                "DRAMATIC RESPONSE: ammonia normalizes 24-48 hr; "
                "Carglumic acid POSITIVE RESPONSE = PATHOGNOMONIC for NAGS deficiency; "
                "Liver transplant usually UNNECESSARY"
            ),
            "Citrin deficiency / CTLN2 (SLC25A13)": (
                "AR mitochondrial aspartate-glutamate carrier deficiency; "
                "Japanese/East Asian founder (SLC25A13 exon16 ins3kb); carrier freq ~1/65 Japan; "
                "Adult-onset citrullinemia type II: episodic hyperammonemia + hepatic encephalopathy; "
                "ALCOHOL ABSOLUTELY TRIGGERS acute crisis; high-carbohydrate diet TRIGGERS; "
                "IV dextrose (hospital) precipitates crisis — ALERT ICU/anesthesia; "
                "Low-carbohydrate MCT-enriched high-protein diet; liver transplant CURATIVE"
            ),
            "HHH syndrome (SLC25A15)": (
                "AR ornithine carrier deficiency (ORNT1); Italian/Franco-Canadian p.Phe188del founder; "
                "TRIAD: Hyperornithinemia + Hyperammonemia + Homocitrullinuria; "
                "Homocitrulline in urine = PATHOGNOMONIC (distinguishes from OTC where absent); "
                "Progressive spastic paraparesis (UMN signs) hallmark neurological feature; "
                "Ornithine-restricted diet + citrulline supplement + pyridoxine; "
                "DDx LPI (SLC7A7): HHH has HIGH urine ornithine; LPI has LOW dibasic amino acids"
            ),
            "Nitrogen scavengers in UCD": (
                "Sodium benzoate: conjugates glycine → hippurate (excretes 1 N per mole); "
                "Sodium phenylbutyrate: prodrug → phenylacetate → conjugates glutamine → phenylacetylglutamine (2 N); "
                "Glycerol phenylbutyrate (RAVICTI) FDA2013: improved palatability, sustained release; "
                "Alternative nitrogen excretion pathway bypassing blocked urea cycle; "
                "Indicated in OTC, ASS1, ASL, CPS1, ARG1 (less critical), SLC25A15; "
                "NAGS: usually not needed long-term if carglumic acid effective"
            ),
            "Liver-brain dissociation in ASL": (
                "Liver transplant corrects hepatic ASL deficiency → no more ammonia crises; "
                "But: ASL is expressed in neurons — brain receives no benefit from liver transplant; "
                "Neurological disease (intellectual disability, behavior) persists/progresses post-LTx; "
                "Systemic hypertension (75%): NO deficiency in vascular endothelium → persists post-LTx; "
                "Arginine supplementation benefits NO synthesis in both hepatic and extra-hepatic tissues; "
                "This is a fundamental concept: UCD liver transplant ≠ neurological cure in ASL"
            ),
        },
        "critical_clinical_rules": {
            "OTC-CITRULLINE-LOW-OROTIC-HIGH-PATHOGNOMONIC": (
                "OTC deficiency: plasma citrulline LOW (<5 μmol/L) + urine orotic acid ELEVATED = PATHOGNOMONIC; "
                "Distinguishes OTC from CPS1/NAGS (both have citrulline low but orotic NORMAL in CPS1/NAGS); "
                "X-linked: males hemizygous severe; females carrier test unreliable — gene sequencing; "
                "FASTING/CATABOLISM ABSOLUTELY AVOID — triggers crisis in males AND symptomatic females; "
                "RAVICTI (glycerol phenylbutyrate) FDA2013: nitrogen scavenger; citrulline supplement mandatory"
            ),
            "ASS1-CITRULLINE-GT1000-PATHOGNOMONIC": (
                "ASS1 (CTLN1): plasma citrulline >1000 μmol/L PATHOGNOMONIC; "
                "Orotic acid NORMAL — distinguishes from OTC; "
                "Arginine supplementation 300-600 mg/kg/day MANDATORY — cannot be synthesized past block; "
                "Liver transplant curative — often first year of life in NBS-detected cases; "
                "NBS: citrulline >150 μmol/L triggers ASS1/ASL investigation (tier-2)"
            ),
            "ASL-LIVER-BRAIN-DISSOCIATION-HYPERTENSION": (
                "ASL deficiency: argininosuccinic acid elevated PATHOGNOMONIC; "
                "LIVER-BRAIN DISSOCIATION: LTx corrects ammonia crises but NOT neurological disease; "
                "Systemic hypertension in 75% even with excellent ammonia control — check BP every visit; "
                "Arginine supplementation CRITICAL for NO synthesis — reduces hypertension; "
                "Trichorrhexis nodosa (brittle hair) PATHOGNOMONIC in UCD context"
            ),
            "ARG1-MILD-NH3-SPASTIC-DIPLEGIA-PATHOGNOMONIC": (
                "ARG1 (hyperargininemia): MILD or absent neonatal hyperammonemia — KEY DISTINCTION from other UCDs; "
                "Progressive spastic diplegia + intellectual disability PATHOGNOMONIC for ARG1 in toddler; "
                "Plasma arginine >200 μmol/L diagnostic; "
                "Low-arginine diet (EAA mix WITHOUT arginine) PRIMARY treatment — prevents neurological progression; "
                "Liver transplant controversial — neurological may persist due to brain ARG2"
            ),
            "CPS1-vs-NAGS-CARGLUMIC-ACID-CRITICAL-DISTINCTION": (
                "CPS1 + NAGS: BOTH have citrulline LOW + orotic acid NORMAL — biochemically IDENTICAL; "
                "ONLY DISTINCTION: carglumic acid response; "
                "  NAGS: carglumic acid DRAMATIC response (NH3 normalizes 24-48 hr) = PATHOGNOMONIC; "
                "  CPS1: NO response to carglumic acid; "
                "NAGS deficiency ONLY UCD treatable with pharmacological cure (Carbaglu FDA2010); "
                "CPS1: liver transplant curative; NAGS: liver transplant usually avoidable"
            ),
            "SLC25A13-CTLN2-ALCOHOL-CARB-TRIGGER-PATHOGNOMONIC": (
                "Citrin deficiency (CTLN2): alcohol + high-carbohydrate diet ABSOLUTELY TRIGGER acute crisis; "
                "IV dextrose infusion (hospital) PRECIPITATES crisis — MUST alert ICU and anesthesia; "
                "Japanese/East Asian ancestry: test SLC25A13 first; "
                "Low-carbohydrate MCT-enriched high-protein diet PRIMARY management; "
                "Liver transplant CURATIVE for recurrent adult CTLN2; "
                "Donor selection: SLC25A13 carrier donors must be excluded"
            ),
            "SLC25A15-HHH-HOMOCITRULLINE-PATHOGNOMONIC": (
                "HHH syndrome (SLC25A15): homocitrulline in urine PATHOGNOMONIC; "
                "Homocitrulline ABSENT in OTC deficiency — key DDx; "
                "Progressive spastic paraparesis (UMN) hallmark — check for spasticity in all HHH; "
                "Ornithine-restricted diet + citrulline supplement + pyridoxine; "
                "DDx LPI (SLC7A7): HHH has HIGH urine ornithine; LPI has LOW dibasic amino acids; "
                "Italian/Franco-Canadian p.Phe188del founder mutation"
            ),
            "UCD-FASTING-CATABOLISM-ABSOLUTELY-AVOID": (
                "ALL UCDs: fasting/catabolism triggers hyperammonemia crisis; "
                "Perioperative: IV dextrose mandatory (exception: SLC25A13 — use lipid-based instead); "
                "Intercurrent illness: increase caloric support, temporarily reduce protein; "
                "Surgery, anesthesia, trauma, prolonged fasting: ALL are crisis triggers; "
                "24/7 emergency protocol letter: every UCD patient should carry one; "
                "ICU/ER: NEVER withhold IV calories while awaiting ammonia result"
            ),
            "UCD-NEWBORN-SCREENING-CITRULLINE-THRESHOLD": (
                "OTC: citrulline LOW on NBS (sensitive only for severe neonatal male); "
                "ASS1/ASL: citrulline ELEVATED on NBS — tier-2 to distinguish; "
                "ARG1: arginine elevated on NBS; "
                "CPS1/NAGS: citrulline LOW — often missed on NBS without specific metabolite; "
                "CTLN2 (SLC25A13): citrulline may be normal in asymptomatic neonates — late presentation; "
                "HHH (SLC25A15): ornithine elevated on some NBS panels; "
                "Negative NBS does NOT rule out OTC/CPS1/NAGS — symptomatic infant with NH3 crisis: full gene panel"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Urea Cycle Disorder Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(
            f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, "
            f"hap_crisis={d['hyperammonemia_crisis_n']}/40, "
            f"ltx={d['liver_transplant_n']}/40, "
            f"neuro={d['neurological_n']}/40"
        )
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
