#!/usr/bin/env python3
"""Hereditary-Metabolic-Myopathy-Atlas — Complete 8-Gene Metabolic Myopathy Spectrum Atlas
(PYGM · CPT2 · ACADVL · ETFDH · HADHA · PFKM · AMPD1 · SLC22A5).

PYGM    (Muscle Glycogen Phosphorylase; 842 aa; 11q13.1; AR;
         McArdle Disease — GSD Type V;
         SECOND WIND PHENOMENON PATHOGNOMONIC;
         Forearm ischemic test: LACTATE FAILS TO RISE, AMMONIA RISES NORMALLY;
         p.Arg50Stop 90% European allele; CK 1000-50000× in crisis;
         seed SEED_BASE+0).
CPT2    (Carnitine Palmitoyltransferase 2; 658 aa; 1p32.3; AR;
         CPT2 Deficiency — long-chain FAO;
         THERMOGENIC TRIGGERS PATHOGNOMONIC: fever/fasting/cold/exercise → rhabdomyolysis;
         p.Ser113Leu common mild allele; triheptanoin medium-chain fat;
         seed SEED_BASE+1).
ACADVL  (Very-Long-Chain Acyl-CoA Dehydrogenase; 655 aa; 17p13.1; AR;
         VLCAD Deficiency — NBS detected (acylcarnitine C14:1 elevated);
         cardiomyopathy severe neonatal form; rhabdomyolysis mild adult form;
         MCT supplementation; triheptanoin available;
         seed SEED_BASE+2).
ETFDH   (Electron Transfer Flavoprotein Dehydrogenase; 617 aa; 4q32.1; AR;
         MADD/GA2 — Multiple Acyl-CoA Dehydrogenase Deficiency;
         RIBOFLAVIN 100-300mg/day ESSENTIALLY CURATIVE in riboflavin-responsive MADD;
         ETF/ETFDH variant → FAD cofactor loss → riboflavin restores;
         seed SEED_BASE+3).
HADHA   (Mitochondrial Trifunctional Protein Alpha Subunit; 763 aa; 2p23.3; AR;
         LCHAD/TFP Deficiency;
         PERIPHERAL RETINOPATHY + PERIPHERAL NEUROPATHY PATHOGNOMONIC combination;
         AFLP maternal risk; G1528C p.Glu510Gln common LCHAD mutation;
         seed SEED_BASE+4).
PFKM    (Phosphofructokinase Muscle Isoform; 780 aa; 12q13.3; AR;
         Tarui Disease — GSD Type VII;
         HEMOLYTIC ANEMIA + EXERCISE MYOPATHY UNIQUE COMBINATION;
         HIGH-CARB PARADOX: IV glucose CONTRAINDICATED in acute crisis (blocks FFAs);
         p.Arg232His/p.Trp249Stop Ashkenazi founder;
         seed SEED_BASE+5).
AMPD1   (AMP Deaminase 1; 747 aa; 1p13.3; AR;
         Myoadenylate Deaminase Deficiency;
         Forearm test: AMMONIA FAILS TO RISE, LACTATE RISES NORMALLY — opposite of McArdle;
         p.Gln12Stop (34CT) 2% European carrier; generally benign; no specific treatment;
         seed SEED_BASE+6).
SLC22A5 (Solute Carrier Family 22 Member 5 / OCTN2; 557 aa; 5q31.1; AR;
         Primary Carnitine Deficiency — CDSP;
         CARNITINE SUPPLEMENTATION ESSENTIALLY CURATIVE (L-carnitine 100mg/kg/day PO);
         cardiomyopathy crisis FATAL WITHOUT treatment; NBS detected;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2230-2237).
"""

import random

SEED_BASE = 2230

METABOLIC_GENES = [
    # -- PYGM — McArdle Disease, GSD V -----------------------------------------------
    {
        "gene": "PYGM",
        "alt_name": (
            "PYGM (PYGM-842aa-11q13.1 / AR — McArdle-Disease-GSD-V — "
            "SECOND-WIND-PHENOMENON-PATHOGNOMONIC — "
            "Forearm-Ischemic-Test-LACTATE-FAILS-TO-RISE-AMMONIA-RISES — "
            "p.Arg50Stop-90pct-European — "
            "CK-1000-50000x-In-Crisis)"
        ),
        "protein": (
            "PYGM -- 11q13.1 AR -- PYGM-842aa -- "
            "Muscle-Glycogen-Phosphorylase-97kDa-Glycogenolysis-First-Step-Enzyme -- "
            "McArdle-Disease-GSD-V-OMIM-232600 -- "
            "SECOND-WIND-PHENOMENON-PATHOGNOMONIC-Fatigue-Then-Recovery-11-15min-Exercise -- "
            "FOREARM-ISCHEMIC-TEST-Lactate-FAILS-TO-RISE-Ammonia-RISES-Normal -- "
            "p.Arg50Stop-NM_005609.6-c.148C>T-90pct-European-Allele -- "
            "CK-1000-50000x-ULN-During-Crisis-Rhabdomyolysis -- "
            "GLUCOSE-SUCROSE-TRICK-10-15g-Before-Exercise-Preventive -- "
            "Avoid-Static-High-Intensity-Exercise-Isometric-Contractions -- "
            "OMIM-Gene-PYGM-608455-Disease-GSD5-232600"
        ),
        "locus": "11q13.1",
        "protein_size": "842 aa / 97 kDa",
        "inheritance": (
            "AR (biallelic loss-of-function); "
            "Muscle glycogen phosphorylase — catalyzes first step of glycogenolysis; "
            "p.Arg50Stop (c.148C>T): ~90% of European McArdle alleles — founder effect; "
            "Second Wind Phenomenon: PATHOGNOMONIC — exercise fatigue/cramps at 8-10 min then "
            "recovery at 11-15 min as FFA + hepatic glycogenolysis compensate; "
            "Forearm ischemic test: lactate FAILS TO RISE (no muscle glycogenolysis), "
            "ammonia RISES NORMALLY (purine nucleotide cycle intact); "
            "CK: 1000-50000× ULN during crisis (rhabdomyolysis); baseline mildly elevated; "
            "Onset: adolescence-adulthood (mean 15-25 yr); "
            "Myoglobinuria (port-wine urine) in ~25% — renal failure risk; "
            "Avoid static/isometric high-intensity exercise; aerobic exercise beneficial"
        ),
        "key_features": [
            "SECOND WIND PHENOMENON — PATHOGNOMONIC: cramps/fatigue at 8-10 min exercise, spontaneous "
            "recovery at 11-15 min as alternative fuels (FFAs, ketones) mobilised",
            "FOREARM ISCHEMIC TEST: venous lactate FAILS TO RISE (blocked glycogenolysis); "
            "venous ammonia RISES NORMALLY (purine nucleotide cycle intact) — opposite of AMPD1",
            "p.Arg50Stop — ~90% of European McArdle alleles; simple PCR genotyping available "
            "as first-line molecular test before full sequencing",
            "CK 1000-50000× ULN during rhabdomyolysis crisis; baseline CK persistently elevated "
            "(300-2000 IU/L) — distinguishes from other metabolic myopathies at rest",
            "GLUCOSE/SUCROSE TRICK — 10-15g oral glucose 5-15 min before exercise prevents second "
            "wind delay; IV glucose beneficial in acute crisis (opposite of PFKM — critical DDx)",
            "Myoglobinuria (port-wine urine) in ~25% of patients; acute kidney injury risk — "
            "IV fluids mandatory in rhabdomyolysis crisis; creatinine + urinalysis monitoring",
            "Avoid static high-intensity isometric exercise (weight lifting, sustained grips); "
            "aerobic low-intensity exercise is BENEFICIAL — improves exercise tolerance",
            "Late-onset fixed proximal weakness in ~25% over age 40 — distinct from acute crisis phase",
        ],
        "treatment": (
            "Acute crisis: IV normal saline hydration (maintain urine output >200 mL/hr); "
            "monitor creatinine + potassium; stop trigger activity. "
            "Prevention: GLUCOSE/SUCROSE TRICK — 10-15g oral glucose/sucrose 5-15 min before exercise. "
            "Exercise training: supervised aerobic exercise programme (walking/cycling) — BENEFICIAL; "
            "progressive low-intensity training improves VO2max. "
            "Diet: normal carbohydrate diet; avoid prolonged fasting. "
            "Avoid: static isometric/weight-lifting exercise; succinylcholine (malignant hyperthermia risk). "
            "Vitamin B6 (pyridoxine): anecdotal benefit for cramps; Level C. "
            "Ramipril: open-label trial (PREXIMO) — equivocal results. "
            "Gene therapy: preclinical research phase 2026."
        ),
        "monitoring": [
            "CK: baseline + quarterly (crisis frequency indicator)",
            "Renal: creatinine + urinalysis at each crisis; annual baseline",
            "Urine: myoglobinuria screen (dipstick ± spectrophotometry) during crisis",
            "Exercise: cardiopulmonary exercise testing (CPET) every 2-3 years",
            "Fixed weakness: annual manual muscle testing (late-onset fixed proximal weakness)",
            "Cardiac: ECG baseline (rarely cardiac PYGM expression but screen)",
            "Genetic: p.Arg50Stop PCR screen in European patients; cascade sibling testing",
            "Anaesthesia: alert card — succinylcholine-related malignant hyperthermia risk",
        ],
    },
    # -- CPT2 — CPT2 Deficiency -------------------------------------------------------
    {
        "gene": "CPT2",
        "alt_name": (
            "CPT2 (CPT2-658aa-1p32.3 / AR — CPT2-Deficiency-Long-Chain-FAO — "
            "THERMOGENIC-TRIGGERS-PATHOGNOMONIC-Fever-Fasting-Cold-Exercise-Rhabdomyolysis — "
            "p.Ser113Leu-Common-Mild-Allele — "
            "Triheptanoin-Medium-Chain-Fat-Supplement)"
        ),
        "protein": (
            "CPT2 -- 1p32.3 AR -- CPT2-658aa -- "
            "Carnitine-Palmitoyltransferase-2-74kDa-Inner-Mitochondrial-Membrane-Long-Chain-FAO -- "
            "CPT2-Deficiency-OMIM-255110 -- "
            "THERMOGENIC-TRIGGERS-PATHOGNOMONIC-Fever-Fasting-Cold-Prolonged-Exercise-All-Trigger-Rhabdomyolysis -- "
            "p.Ser113Leu-c.338C>T-Common-Mild-Myopathic-Allele -- "
            "TRIHEPTANOIN-C7-Medium-Odd-Chain-FA-Substrate-Anaplerosis -- "
            "HIGH-CARB-DIET-BENEFICIAL-Opposite-of-PFKM -- "
            "L-Carnitine-Benefit-Limited-Controversial -- "
            "Rhabdomyolysis-AKI-Risk-Myoglobinuria -- "
            "OMIM-Gene-CPT2-600650-Disease-CPT2-Def-255110"
        ),
        "locus": "1p32.3",
        "protein_size": "658 aa / 74 kDa",
        "inheritance": (
            "AR (biallelic); "
            "CPT2 = inner mitochondrial membrane enzyme; transfers acyl group from acylcarnitine to CoA "
            "for beta-oxidation of long-chain fatty acids; "
            "Myopathic (mild) form: p.Ser113Leu most common — exercise + fever/fasting/cold trigger rhabdomyolysis; "
            "Severe neonatal/infantile form: homozygous null — hypoketotic hypoglycemia + cardiomyopathy; "
            "THERMOGENIC TRIGGERS PATHOGNOMONIC — combination of fever/fasting/cold/exercise unique to CPT2; "
            "CK: normal between episodes; 5000-500000 IU/L in crisis; "
            "L-carnitine: plasma free carnitine low — supplementation used but benefit limited; "
            "Triheptanoin: odd-chain MCT → anaplerosis + bypasses long-chain block; "
            "HIGH-CARB diet beneficial (OPPOSITE of PFKM paradox)"
        ),
        "key_features": [
            "THERMOGENIC TRIGGERS PATHOGNOMONIC — any combination of fever + fasting + cold exposure + "
            "prolonged exercise precipitates rhabdomyolysis; no second wind (unlike PYGM)",
            "p.Ser113Leu (c.338C>T) — most common mild myopathic allele (~60-70% European alleles); "
            "residual enzyme activity ~25%; later onset; exercise-induced rhabdomyolysis",
            "Severe neonatal form: biallelic null mutations → hypoketotic hypoglycemia, cardiomyopathy, "
            "hepatomegaly — distinct from myopathic form; ICU presentation",
            "CK normal between episodes (distinguishes from PYGM which has persistently elevated baseline CK); "
            "CK 5000-500000 IU/L in acute rhabdomyolysis crisis",
            "HIGH-CARB DIET BENEFICIAL — increases glucose availability → spares long-chain FA utilisation; "
            "OPPOSITE of PFKM where glucose IV is contraindicated in crisis",
            "L-carnitine supplementation: plasma carnitine low; supplementation used but evidence limited — "
            "does not prevent attacks; triheptanoin more promising",
            "Triheptanoin (C7 MCT): odd-chain medium fatty acid → bypasses CPT2 block; "
            "anaplerosis of TCA cycle; FDA-approved for LC-FAOD (Dojolvi 2020)",
            "Myoglobinuria risk: ~50% patients experience port-wine urine; AKI — monitor creatinine; "
            "IV saline mandatory in crisis",
        ],
        "treatment": (
            "Acute crisis: IV normal saline (urine output >200 mL/hr); monitor renal function; "
            "avoid further fasting/cold. "
            "Preventive: HIGH-CARB diet; glucose loading before prolonged exercise; "
            "avoid fasting >4-6 hours; avoid cold exposure. "
            "Triheptanoin (Dojolvi): 1-2g/kg/day (≤35g/day) — odd-chain MCT; FDA/EMA approved LC-FAOD. "
            "L-carnitine: 50-100mg/kg/day PO — plasma levels guide; benefit limited but used. "
            "Exercise: aerobic low-intensity; warm up slowly; carry carbohydrate snacks. "
            "Fever management: aggressive antipyretics + increased carbohydrate intake at fever onset. "
            "Alert card: surgery/anaesthesia — glucose infusion perioperatively."
        ),
        "monitoring": [
            "CK: baseline + at any febrile illness/exercise crisis",
            "Renal: creatinine + urinalysis at each crisis; annual baseline",
            "Plasma carnitine: free + total; every 6 months (guide supplementation)",
            "Acylcarnitine profile: C16, C18:1 elevated — confirm diagnosis + response",
            "Cardiac: echocardiogram in infantile/severe forms; annual if cardiac form",
            "Liver: LFTs in severe neonatal form; hepatomegaly surveillance",
            "Nutrition: dietitian review every 6 months; fat intake diary",
            "Genetic: cascade testing; NBS for severe forms",
        ],
    },
    # -- ACADVL — VLCAD Deficiency ----------------------------------------------------
    {
        "gene": "ACADVL",
        "alt_name": (
            "ACADVL (ACADVL-655aa-17p13.1 / AR — VLCAD-Deficiency — "
            "NBS-Detected-Acylcarnitine-C14:1-Elevated — "
            "Cardiomyopathy-Severe-Neonatal-Form — "
            "Rhabdomyolysis-Mild-Adult-Form — "
            "MCT-Supplementation-Triheptanoin)"
        ),
        "protein": (
            "ACADVL -- 17p13.1 AR -- ACADVL-655aa -- "
            "Very-Long-Chain-Acyl-CoA-Dehydrogenase-70kDa-Inner-Mitochondrial-Membrane -- "
            "VLCAD-Deficiency-OMIM-201475 -- "
            "NBS-DETECTED-Acylcarnitine-C14:1-Tetradecenoylcarnitine-Elevated-Pathognomonic-NBS -- "
            "SEVERE-NEONATAL-FORM-Cardiomyopathy-Hypoketotic-Hypoglycemia-ICU -- "
            "MILD-ADULT-FORM-Exercise-Induced-Rhabdomyolysis-Myalgia -- "
            "MCT-Supplementation-Bypasses-VLCAD-Block -- "
            "Triheptanoin-C7-Anaplerosis-FDA-Approved-LC-FAOD -- "
            "Avoid-Fasting-Critical -- "
            "OMIM-Gene-ACADVL-609575-Disease-VLCAD-201475"
        ),
        "locus": "17p13.1",
        "protein_size": "655 aa / 70 kDa",
        "inheritance": (
            "AR (biallelic); "
            "ACADVL = first enzyme in long-chain beta-oxidation within inner mitochondrial membrane; "
            "Newborn screening: C14:1 (tetradecenoylcarnitine) ELEVATED — pathognomonic on NBS; "
            "3 phenotypes: (1) Severe neonatal — cardiomyopathy + hypoketotic hypoglycemia (null alleles); "
            "(2) Childhood hepatic — hypoglycemia predominant; (3) Mild adult myopathic — exercise rhabdomyolysis; "
            "CK: 5000-100000+ in crisis; normal between episodes in mild form; "
            "Cardiac involvement: predominantly severe neonatal form (40%); rare in mild adult; "
            "MCT: medium-chain fats bypass VLCAD block; triheptanoin provides anaplerosis"
        ),
        "key_features": [
            "NBS DETECTED — C14:1 (tetradecenoylcarnitine) elevated on acylcarnitine profile; "
            "PATHOGNOMONIC biomarker; NBS has dramatically improved outcomes",
            "SEVERE NEONATAL FORM — biallelic null/severe mutations; hypertrophic cardiomyopathy + "
            "hypoketotic hypoglycemia; ICU presentation first days of life; ~40% cardiac involvement",
            "MILD ADULT MYOPATHIC FORM — at least one missense with residual activity; exercise-induced "
            "rhabdomyolysis + myalgia; fasting precipitates attacks; similar to CPT2 clinically",
            "CK 5000-100000+ IU/L in acute rhabdomyolysis; normal baseline in mild adult form; "
            "myoglobinuria risk — AKI in severe crisis",
            "MCT SUPPLEMENTATION — medium-chain fats (C8-C12) bypass VLCAD block; cornerstone of diet "
            "management; combined with avoidance of long-chain fats",
            "Triheptanoin (Dojolvi): C7 odd-chain MCT; anaplerosis of TCA cycle; FDA-approved for LC-FAOD "
            "including VLCAD; 1-2g/kg/day (max 35g/day)",
            "Avoid fasting >4-6 hours (young children <2 hours); glucose loading during illness; "
            "emergency letter mandatory for hospital admissions",
            "L-carnitine: used if plasma carnitine low; secondary carnitine deficiency common",
        ],
        "treatment": (
            "Acute crisis: IV 10% dextrose (high rate, avoid fasting); IV saline for myoglobinuria; "
            "monitor renal + cardiac function. "
            "Diet: low long-chain fat; MCT supplementation (C8-C10); avoid fasting. "
            "Triheptanoin (Dojolvi): 1-2g/kg/day — preferred MCT for anaplerosis. "
            "L-carnitine: 50-100mg/kg/day if deficient. "
            "Cardiac: ECG + echocardiogram in neonatal/severe form; ACE inhibitor + beta-blocker if DCM. "
            "NBS follow-up: early initiation of diet before symptoms in NBS-detected. "
            "Emergency protocol: 'sick day rules' — increase carbohydrate, reduce fat, contact metabolic team."
        ),
        "monitoring": [
            "Acylcarnitine profile: C14:1 — response to treatment; every 3-6 months",
            "Carnitine: free + total plasma; every 3-6 months",
            "CK: at each illness/crisis; quarterly in active disease",
            "Cardiac: echocardiogram every 6 months in severe form; annually in mild",
            "Renal: creatinine + urinalysis at each crisis",
            "Glucose: fasting glucose + ketones diary; continuous glucose monitoring if hypoglycaemia",
            "Liver: LFTs, liver USS in hepatic phenotype",
            "Nutrition: dietitian every 6 months; LCFA intake diary",
        ],
    },
    # -- ETFDH — MADD / GA2, riboflavin-responsive ------------------------------------
    {
        "gene": "ETFDH",
        "alt_name": (
            "ETFDH (ETFDH-617aa-4q32.1 / AR — MADD-GA2-Multiple-Acyl-CoA-Dehydrogenase-Deficiency — "
            "RIBOFLAVIN-100-300mg/day-ESSENTIALLY-CURATIVE-Riboflavin-Responsive-MADD — "
            "ETF-ETFDH-Variant-FAD-Cofactor-Loss-Riboflavin-Restores — "
            "Must-Try-Before-Calling-Refractory)"
        ),
        "protein": (
            "ETFDH -- 4q32.1 AR -- ETHDH-617aa -- "
            "Electron-Transfer-Flavoprotein-Dehydrogenase-68kDa-Mitochondrial-FAD-Dependent -- "
            "MADD-GA2-OMIM-231680 -- "
            "RIBOFLAVIN-RESPONSIVE-MADD-100-300mg-Vitamin-B2-ESSENTIALLY-CURATIVE -- "
            "FAD-Cofactor-Loss-ETF-ETFDH-Variant-Riboflavin-Restores-Enzyme-Activity -- "
            "Multiple-Acyl-CoA-Dehydrogenases-Impaired-All-FAD-Dependent -- "
            "IVA-Like-Crisis-Isovaleric-Acid-Odor-Sweat-Urine -- "
            "Avoid-Fasting-Critical -- "
            "Riboflavin-80pct-Response-Rate-ETFDH-Missense -- "
            "OMIM-Gene-ETFDH-231675-Disease-MADD-231680"
        ),
        "locus": "4q32.1",
        "protein_size": "617 aa / 68 kDa",
        "inheritance": (
            "AR (biallelic); "
            "ETFDH encodes electron transfer flavoprotein dehydrogenase — FAD-dependent; "
            "also ETFA/ETFB mutations cause MADD; "
            "MADD = Multiple Acyl-CoA Dehydrogenase Deficiency (GA2 = glutaric aciduria type II); "
            "Riboflavin-responsive MADD (RR-MADD): ETFDH missense → partial FAD binding loss → "
            "riboflavin (vitamin B2) supplements restore FAD → enzyme activity restored; "
            "Riboflavin 100-300mg/day ESSENTIALLY CURATIVE in RR-MADD — must trial before labelling refractory; "
            "CK: 500-5000 IU/L in myopathic form; elevated lipid storage myopathy; "
            "IVA-like crisis: isovaleric acid odour (sweaty feet odour); organic acids elevated; "
            "Avoid fasting; CoQ10 supplementation adjunctive"
        ),
        "key_features": [
            "RIBOFLAVIN 100-300mg/day ESSENTIALLY CURATIVE in riboflavin-responsive MADD (RR-MADD) — "
            "ETFDH missense alleles lose FAD binding; riboflavin restores cofactor → enzyme activity; "
            "~80% response rate in ETFDH missense cases; dramatic clinical improvement within weeks",
            "MUST TRIAL RIBOFLAVIN before calling MADD refractory — most ETFDH missense patients "
            "respond dramatically; failure to trial = inadequate management",
            "Multiple Acyl-CoA Dehydrogenase Deficiency: all FAD-dependent acyl-CoA dehydrogenases impaired "
            "(SCAD, MCAD, VLCAD, LCAD, glutaryl-CoA DH) → complex acylcarnitine profile elevation",
            "IVA-like crisis: isovaleric acid + other organic acids elevated; 'sweaty feet' odour in urine/sweat; "
            "Organic acid urine screen: glutaric acid + multiple acylcarnitines elevated",
            "Lipid storage myopathy on biopsy — oil red O stain shows excess lipid vacuoles in muscle fibres; "
            "characteristic for ETFDH/MADD (and HADHA) — different from glycogen storage (PAS stain)",
            "Myopathic form: proximal weakness + exercise intolerance; episodic crisis with fasting; "
            "CK 500-5000 IU/L; distinct from severe neonatal/infantile forms (more severe, metabolic acidosis)",
            "CoQ10 supplementation (200-300mg/day) adjunctive — some patients have secondary CoQ10 deficiency; "
            "riboflavin + CoQ10 combination used",
            "Avoid fasting; low-fat diet; L-carnitine supplementation if carnitine deficient",
        ],
        "treatment": (
            "RIBOFLAVIN (Vitamin B2): 100-300mg/day in 2-3 divided doses — FIRST LINE; "
            "trial MANDATORY before labelling refractory; response within 2-8 weeks (strength, CK). "
            "CoQ10: 200-300mg/day in 2-3 doses — adjunctive; secondary CoQ10 deficiency common. "
            "L-carnitine: 50-100mg/kg/day if carnitine deficient. "
            "Diet: low long-chain fat; avoid prolonged fasting; complex carbohydrates. "
            "Acute crisis: IV glucose + IV carnitine; avoid fasting; treat intercurrent illness. "
            "Physiotherapy: strength training tolerated once riboflavin initiated. "
            "No disease-modifying therapy beyond riboflavin/CoQ10 for RR-MADD 2026."
        ),
        "monitoring": [
            "CK: monthly until stable on riboflavin; then quarterly (response marker)",
            "Acylcarnitine profile: C4-C10 pattern — response to riboflavin; every 3-6 months",
            "Organic acids urine: glutaric acid + acylcarnitines — normalisation on treatment",
            "Carnitine: free + total; every 6 months",
            "CoQ10: plasma CoQ10 level — guide supplementation; every 6-12 months",
            "Riboflavin dosing: assess at 4-8 weeks for clinical + biochemical response",
            "LFTs: baseline; hepatomegaly surveillance (infantile form)",
            "Physiotherapy: functional strength assessment every 6 months",
        ],
    },
    # -- HADHA — LCHAD / TFP Deficiency -----------------------------------------------
    {
        "gene": "HADHA",
        "alt_name": (
            "HADHA (HADHA-763aa-2p23.3 / AR — LCHAD-TFP-Deficiency — "
            "PERIPHERAL-RETINOPATHY-PERIPHERAL-NEUROPATHY-PATHOGNOMONIC-Combination — "
            "AFLP-Maternal-Heterozygote-Acute-Fatty-Liver-Pregnancy — "
            "G1528C-p.Glu510Gln-Common-LCHAD-Mutation — "
            "MCT-Avoid-Long-Chain-Fat)"
        ),
        "protein": (
            "HADHA -- 2p23.3 AR -- HADHA-763aa -- "
            "Mitochondrial-Trifunctional-Protein-Alpha-Subunit-79kDa-LCHAD-LCEH-LCAT-Activities -- "
            "LCHAD-TFP-Deficiency-OMIM-609015 -- "
            "PERIPHERAL-RETINOPATHY-Pigmentary-Retinal-Degeneration-PATHOGNOMONIC-Extra-Muscular -- "
            "PERIPHERAL-NEUROPATHY-Sensorimotor-PATHOGNOMONIC-Combination-Retinopathy-Plus-Neuropathy -- "
            "AFLP-Acute-Fatty-Liver-Pregnancy-Maternal-Heterozygote-Risk -- "
            "G1528C-p.Glu510Gln-c.1528G>C-Common-LCHAD-Point-Mutation-LCHAD-Specific -- "
            "MCT-Medium-Chain-Fat-Supplement-Mainstay -- "
            "Avoid-Long-Chain-Fat-LCFA -- "
            "Ophthalmology-6-Monthly-Mandatory -- "
            "OMIM-Gene-HADHA-600890-Disease-LCHAD-609015"
        ),
        "locus": "2p23.3",
        "protein_size": "763 aa / 79 kDa",
        "inheritance": (
            "AR (biallelic HADHA or HADHB mutations); "
            "HADHA = alpha subunit of mitochondrial trifunctional protein (MTP); "
            "MTP carries 3 activities: LCHAD + LCEH + LCAT — long-chain beta-oxidation trifunctional; "
            "LCHAD-specific mutations (e.g. G1528C) vs TFP mutations (affect all 3 activities); "
            "G1528C (p.Glu510Gln) — most common LCHAD-specific mutation (~90% LCHAD alleles); "
            "PERIPHERAL RETINOPATHY + PERIPHERAL NEUROPATHY: PATHOGNOMONIC combination — "
            "extra-muscular involvement unique to LCHAD/TFP (not seen in VLCAD or CPT2); "
            "AFLP (Acute Fatty Liver of Pregnancy): maternal heterozygote carrying LCHAD fetus — "
            "long-chain acylcarnitines from affected fetus overwhelm maternal liver; "
            "CK: 500-10000 IU/L; hypoketotic hypoglycemia in neonatal form"
        ),
        "key_features": [
            "PERIPHERAL RETINOPATHY + PERIPHERAL NEUROPATHY — PATHOGNOMONIC combination unique to LCHAD/TFP; "
            "pigmentary retinal degeneration (similar to RP); progressive; ophthalmology 6-monthly mandatory",
            "AFLP (Acute Fatty Liver of Pregnancy) — maternal heterozygote risk when carrying LCHAD-affected fetus; "
            "long-chain acylcarnitines from fetus accumulate in mother; potentially fatal; "
            "early delivery mandatory on diagnosis; screen ALL AFLP mothers for LCHAD child",
            "G1528C (p.Glu510Gln, c.1528G>C) — most common LCHAD-specific point mutation (~90% European alleles); "
            "PCR genotyping available; affects LCHAD catalytic site specifically",
            "MCT SUPPLEMENTATION mainstay — medium-chain fats (C8-C10) bypass LCHAD block; "
            "combined with avoidance of long-chain fatty acids; cornflour/uncooked starch for fasting prevention",
            "CK 500-10000 IU/L; hypoketotic hypoglycemia in neonatal/infantile form; "
            "rhabdomyolysis on fasting/illness; acylcarnitine C16-OH, C18:1-OH elevated — diagnostic",
            "Neonatal NBS: C16-OH acylcarnitine elevated (3-hydroxy acylcarnitines); early diagnosis "
            "before symptomatic allows diet to prevent retinopathy progression",
            "Avoid long-chain fat (LCT) foods; avoid fasting >4-6 hours; sick day protocol mandatory; "
            "avoid medium-to-long-chain fat loading (unlike PFKM where fat is preferred fuel in crisis)",
            "Progressive peripheral neuropathy: sensorimotor; EMG/NCS monitoring; physiotherapy; orthotics",
        ],
        "treatment": (
            "Diet: MCT-based diet — low LCT, high MCT supplementation; "
            "MCT oil/powder; avoidance of prolonged fasting. "
            "Triheptanoin (C7): adjunctive anaplerosis; FDA approved LC-FAOD. "
            "Ophthalmology: 6-monthly retinal examination (OCT + ERG); dark adaptation testing; "
            "vitamin A supplementation anecdotal. "
            "Peripheral neuropathy: physiotherapy; AFO; pain management (gabapentin). "
            "AFLP: emergency delivery + IV glucose; screen neonate for LCHAD. "
            "Acute crisis: IV 10% dextrose + IV carnitine (if deficient); avoid fat loading. "
            "L-carnitine: secondary deficiency; supplement if low. "
            "Emergency letter: perioperative glucose infusion mandatory."
        ),
        "monitoring": [
            "Ophthalmology: retinal examination + OCT + ERG every 6 months — MANDATORY",
            "Dark adaptation: annual dark adaptation testing; vitamin A level",
            "Neurology: NCS + EMG every 12 months (peripheral neuropathy progression)",
            "Acylcarnitine: C16-OH, C18:1-OH every 3-6 months (treatment response)",
            "CK: quarterly baseline; at each crisis",
            "Glucose: fasting glucose + ketones; continuous glucose monitoring if hypoglycaemic",
            "Cardiac: echocardiogram annually (cardiac TFP involvement possible)",
            "Obstetric: AFLP surveillance in all female LCHAD heterozygote carriers who are pregnant",
        ],
    },
    # -- PFKM — Tarui Disease, GSD VII -------------------------------------------------
    {
        "gene": "PFKM",
        "alt_name": (
            "PFKM (PFKM-780aa-12q13.3 / AR — Tarui-Disease-GSD-VII — "
            "HEMOLYTIC-ANEMIA-EXERCISE-MYOPATHY-UNIQUE-COMBINATION — "
            "HIGH-CARB-PARADOX-IV-GLUCOSE-ABSOLUTELY-CONTRAINDICATED-Acute-Crisis — "
            "p.Arg232His-p.Trp249Stop-Ashkenazi-Founder — "
            "Give-Fat-Protein-Not-Glucose-In-Crisis)"
        ),
        "protein": (
            "PFKM -- 12q13.3 AR -- PFKM-780aa -- "
            "Phosphofructokinase-Muscle-Isoform-85kDa-Glycolysis-Rate-Limiting-Step-PFK1 -- "
            "Tarui-Disease-GSD-VII-OMIM-232800 -- "
            "HEMOLYTIC-ANEMIA-EXERCISE-MYOPATHY-UNIQUE-COMBINATION-Two-Systems-One-Gene -- "
            "HIGH-CARB-PARADOX-Glucose-IV-Blocks-FFAs-Worsens-Crisis-CONTRAINDICATED -- "
            "p.Arg232His-Ashkenazi-Jewish-Founder-Most-Common -- "
            "p.Trp249Stop-Second-Ashkenazi-Allele -- "
            "Give-Fat-Protein-Not-Glucose-In-Acute-Crisis -- "
            "No-Second-Wind-Unlike-PYGM -- "
            "Reticulocytosis-Elevated-Bilirubin-Uric-Acid -- "
            "OMIM-Gene-PFKM-610681-Disease-GSD7-232800"
        ),
        "locus": "12q13.3",
        "protein_size": "780 aa / 85 kDa",
        "inheritance": (
            "AR (biallelic); "
            "PFKM = muscle isoform of phosphofructokinase (PFK-M) — rate-limiting enzyme of glycolysis; "
            "p.Arg232His: Ashkenazi Jewish founder (~70% of Ashkenazi alleles); "
            "p.Trp249Stop: second Ashkenazi Jewish allele; "
            "HEMOLYTIC ANEMIA + EXERCISE MYOPATHY: UNIQUE COMBINATION — PFK-M also partially expressed "
            "in RBCs (hybrid PFK isoforms in RBCs use M subunits); partial PFK loss in erythrocytes → "
            "haemolytic anaemia; "
            "HIGH-CARB PARADOX: glucose loading blocks free fatty acid (FFA) release → worsens energy crisis; "
            "IV glucose ABSOLUTELY CONTRAINDICATED in acute crisis; "
            "CK: 500-10000 IU/L in crisis; NO SECOND WIND (unlike PYGM); "
            "Forearm ischemic test: lactate FAILS TO RISE (glycolytic block); ammonia RISES"
        ),
        "key_features": [
            "HEMOLYTIC ANEMIA + EXERCISE MYOPATHY — UNIQUE COMBINATION in metabolic myopathies; "
            "PFK-M isoform expressed in both muscle AND red blood cells; partial loss → dual manifestation; "
            "reticulocytosis, elevated bilirubin, elevated LDH — screen for haemolysis",
            "HIGH-CARB PARADOX — PATHOGNOMONIC and DANGEROUS: oral/IV carbohydrate loading "
            "suppresses FFA release → the only available fuel (FFAs) is blocked → worsens crisis; "
            "IV glucose ABSOLUTELY CONTRAINDICATED in acute myopathy crisis (opposite of McArdle/GSD III)",
            "In acute crisis: GIVE FAT + PROTEIN, NOT GLUCOSE — ketogenic substrate; "
            "IV lipid emulsion + amino acids if severe crisis; avoid dextrose IV",
            "p.Arg232His + p.Trp249Stop: Ashkenazi Jewish founder alleles; PCR genotyping available; "
            "founder effect in this population — targeted testing cost-effective",
            "NO SECOND WIND (unlike PYGM/McArdle) — no alternative fuel rescue mechanism; "
            "exercise capacity consistently impaired; patients learn to self-limit",
            "Forearm ischemic test: lactate FAILS TO RISE (glycolytic block at PFK step) + "
            "ammonia RISES normally — same pattern as PYGM but different enzyme block",
            "Uric acid elevated — hyperuricaemia common; gout risk; raised by strenuous exercise; "
            "allopurinol if symptomatic gout",
            "CK 500-10000 IU/L in crisis; myoglobinuria risk; AKI in severe rhabdomyolysis",
        ],
        "treatment": (
            "ACUTE CRISIS: GIVE FAT + PROTEIN, NO GLUCOSE — IV lipid emulsion (Intralipid) + amino acids; "
            "AVOID dextrose/glucose infusion (HIGH-CARB PARADOX). "
            "Chronic management: avoid high-carbohydrate meals before exercise; "
            "ketogenic or moderate-fat diet; small frequent meals. "
            "Exercise: avoid sustained high-intensity; brief rest stops reduce lactic acid accumulation; "
            "aerobic low-intensity tolerated; patients self-limit. "
            "Haemolysis: monitor FBC, bilirubin, reticulocytes; folate supplementation if haemolytic; "
            "avoid oxidative haemolytic triggers. "
            "Gout: allopurinol if symptomatic hyperuricaemia; adequate hydration. "
            "IV fluids: normal saline for myoglobinuria; monitor renal function. "
            "Alert card: GLUCOSE IV CONTRAINDICATED in crisis."
        ),
        "monitoring": [
            "FBC: haemoglobin, reticulocytes, bilirubin — haemolytic anaemia screen; quarterly",
            "CK: baseline + at crisis; quarterly",
            "Uric acid: 6-monthly (hyperuricaemia/gout monitoring)",
            "LFTs: bilirubin elevation from haemolysis; every 6 months",
            "Renal: creatinine + urinalysis at each crisis (myoglobinuria/AKI)",
            "Exercise: self-reported activity diary; CPET every 2-3 years",
            "Genetic: Ashkenazi Jewish population — p.Arg232His PCR screen cost-effective",
            "Folate: supplementation if haemolytic; folate level annually",
        ],
    },
    # -- AMPD1 — Myoadenylate Deaminase Deficiency ------------------------------------
    {
        "gene": "AMPD1",
        "alt_name": (
            "AMPD1 (AMPD1-747aa-1p13.3 / AR — Myoadenylate-Deaminase-Deficiency — "
            "Forearm-Test-AMMONIA-FAILS-TO-RISE-LACTATE-RISES-NORMALLY-Opposite-of-McArdle — "
            "p.Gln12Stop-34CT-2pct-European-Carrier — "
            "Generally-Benign-No-Specific-Treatment — "
            "Exclude-McArdle-If-CK-High)"
        ),
        "protein": (
            "AMPD1 -- 1p13.3 AR -- AMPD1-747aa -- "
            "AMP-Deaminase-1-Muscle-Isoform-88kDa-Purine-Nucleotide-Cycle-AMP-to-IMP -- "
            "Myoadenylate-Deaminase-Deficiency-OMIM-615511 -- "
            "FOREARM-TEST-AMMONIA-FAILS-TO-RISE-Purine-Cycle-Blocked-LACTATE-RISES-NORMALLY -- "
            "OPPOSITE-OF-McArdle-PYGM-Critical-DDx -- "
            "p.Gln12Stop-c.34C>T-Most-Common-AMPD1-Mutation-2pct-European-Carrier -- "
            "Generally-BENIGN-Post-Exercise-Myalgia-Fatigue-Only -- "
            "No-Specific-Treatment-Available -- "
            "Exclude-McArdle-If-CK-Elevated-in-Crisis -- "
            "OMIM-Gene-AMPD1-102770-Disease-MAMD-615511"
        ),
        "locus": "1p13.3",
        "protein_size": "747 aa / 88 kDa",
        "inheritance": (
            "AR (biallelic); also mild heterozygote phenotype described; "
            "AMPD1 = AMP deaminase muscle isoform; catalyses AMP → IMP + NH3 (purine nucleotide cycle); "
            "p.Gln12Stop (c.34C>T): most common AMPD1 mutation; ~2% European carrier frequency; "
            "Forearm ischemic test: AMMONIA FAILS TO RISE (purine nucleotide cycle blocked); "
            "LACTATE RISES NORMALLY (glycolysis intact) — OPPOSITE of McArdle (PYGM); "
            "Generally BENIGN — post-exercise myalgia + fatigue; no rhabdomyolysis crisis typically; "
            "CK: mild-moderate elevation 200-2000 IU/L; baseline often normal; "
            "No specific treatment; exclude secondary AMPD1 deficiency (most common form); "
            "Primary vs secondary distinction important"
        ),
        "key_features": [
            "FOREARM ISCHEMIC TEST: AMMONIA FAILS TO RISE (purine nucleotide cycle blocked — AMPD1 absent); "
            "LACTATE RISES NORMALLY (glycolysis fully intact) — EXACT OPPOSITE of McArdle (PYGM); "
            "Critical DDx: know which test distinguishes AMPD1 vs PYGM",
            "GENERALLY BENIGN — post-exercise myalgia, cramps, fatigue; no acute rhabdomyolysis crisis "
            "in typical primary AMPD1 deficiency; ambulant essentially 100%; no NIV; no cardiac",
            "p.Gln12Stop (c.34C>T) — most common mutation; ~2% European carrier frequency (high for an AR disease); "
            "homozygous individuals often discovered incidentally or with minimal symptoms",
            "No specific treatment — no drug, supplement, or diet proven to alter course; "
            "anecdotal reports of riboflavin, D-ribose — not evidence-based",
            "Exclude McArdle if CK markedly elevated during crisis — AMPD1 CK mild (200-2000 IU/L); "
            "marked CK elevation (>5000) should trigger forearm test to distinguish",
            "Secondary AMPD1 deficiency — more common than primary; seen in inflammatory myopathies, "
            "muscular dystrophies — treat underlying disease, not AMPD1",
            "Exercise tolerance: mildly reduced; aerobic exercise generally well-tolerated; "
            "avoid exhaustive exercise; pacing strategies helpful",
            "Genetic counseling: AR pattern; siblings 25% risk; however clinical significance often minimal",
        ],
        "treatment": (
            "No specific disease-modifying treatment proven 2026. "
            "Pacing strategies: avoid sustained maximal exertion; self-pacing during exercise. "
            "Aerobic exercise: low-to-moderate intensity exercise generally tolerated; pacing programme. "
            "Symptom management: analgesics/NSAIDs for post-exercise myalgia (short-term). "
            "D-ribose: anecdotal reports of reduced post-exercise fatigue; not evidence-based; "
            "Level C if trialled. "
            "Exclude secondary AMPD1: screen for inflammatory myopathy, muscular dystrophy — "
            "treat underlying condition if found. "
            "Genetic counseling: AR; sibling cascade; clinical significance often mild."
        ),
        "monitoring": [
            "CK: annual baseline; at any rhabdomyolysis-like crisis (to exclude PYGM/other)",
            "Exercise: self-reported myalgia diary; CPET if clinically indicated",
            "Forearm ischemic test: at diagnosis — ammonia FAILS TO RISE, lactate RISES normally",
            "Exclude secondary: CPK, aldolase, muscle biopsy if clinical features atypical",
            "Annual review: symptom progression assessment; functional capacity",
            "Genetic: sibling cascade; prenatal counseling if family requests",
        ],
    },
    # -- SLC22A5 — Primary Carnitine Deficiency ----------------------------------------
    {
        "gene": "SLC22A5",
        "alt_name": (
            "SLC22A5 (SLC22A5-557aa-5q31.1 / AR — Primary-Carnitine-Deficiency-CDSP — "
            "CARNITINE-SUPPLEMENTATION-ESSENTIALLY-CURATIVE-L-Carnitine-100mg-kg-PO — "
            "Cardiomyopathy-Crisis-FATAL-WITHOUT-Treatment — "
            "OCTN2-Transporter-Low-Plasma-Free-Carnitine — "
            "NBS-Detected)"
        ),
        "protein": (
            "SLC22A5 -- 5q31.1 AR -- SLC22A5-557aa -- "
            "OCTN2-Organic-Cation-Carnitine-Transporter-63kDa-Renal-Cardiac-Skeletal-Muscle -- "
            "Primary-Carnitine-Deficiency-CDSP-OMIM-212140 -- "
            "CARNITINE-SUPPLEMENTATION-ESSENTIALLY-CURATIVE-L-Carnitine-100-200mg-kg-PO -- "
            "Cardiomyopathy-HYPERTROPHIC-Dilated-FATAL-WITHOUT-Treatment -- "
            "Low-Plasma-Free-Carnitine-Below-5-umol-L-Pathognomonic -- "
            "NBS-DETECTED-Acylcarnitine-Low-C0-Free-Carnitine -- "
            "Cardiomyopathy-Reversible-With-Treatment -- "
            "Renal-Carnitine-Loss-Increased-Urine-Carnitine -- "
            "OMIM-Gene-SLC22A5-603377-Disease-CDSP-212140"
        ),
        "locus": "5q31.1",
        "protein_size": "557 aa / 63 kDa",
        "inheritance": (
            "AR (biallelic); "
            "SLC22A5 = OCTN2 carnitine transporter — expressed in kidney, heart, skeletal muscle, intestine; "
            "LOF → failure to reabsorb carnitine in kidney → renal wasting → plasma carnitine critically low; "
            "Primary Carnitine Deficiency (CDSP): plasma free carnitine <5 μmol/L (normal 25-50 μmol/L); "
            "NBS: low C0 (free carnitine) on acylcarnitine profile — early detection before symptoms; "
            "Cardiomyopathy: hypertrophic → dilated; FATAL WITHOUT treatment — cardiomyopathy can be rapidly fatal; "
            "L-carnitine 100-200mg/kg/day PO ESSENTIALLY CURATIVE — plasma carnitine normalises; "
            "cardiomyopathy REVERSIBLE with treatment in most cases; "
            "Hypoglycaemia: hypoketotic hypoglycaemia in childhood; muscle weakness if untreated"
        ),
        "key_features": [
            "L-CARNITINE 100-200mg/kg/day PO ESSENTIALLY CURATIVE — most complete treatment response "
            "of any metabolic myopathy; plasma carnitine normalises; cardiomyopathy reverses; "
            "continuation mandatory lifelong (stopping → relapse)",
            "Cardiomyopathy FATAL WITHOUT TREATMENT — hypertrophic cardiomyopathy → dilated; "
            "sudden cardiac death possible; L-carnitine starts cardiac recovery within weeks; "
            "echocardiogram mandatory at diagnosis",
            "Plasma free carnitine <5 μmol/L (often <2 μmol/L) — PATHOGNOMONIC; "
            "normal 25-50 μmol/L; urine carnitine ELEVATED (renal wasting — distinguishes from "
            "secondary carnitine deficiency where urine carnitine is normal/low)",
            "NBS DETECTED — low C0 (free carnitine) on dried blood spot acylcarnitine profile; "
            "early NBS detection before cardiac symptoms dramatically improves outcomes; "
            "~90% of cases NBS-identifiable",
            "Cardiomyopathy REVERSIBLE with treatment — echocardiogram normalisation in months "
            "after L-carnitine started; one of few reversible cardiomyopathies in metabolic disease",
            "IV carnitine in crisis — acute cardiomyopathy/hypoglycaemic crisis: IV L-carnitine "
            "100mg/kg bolus + infusion; cardiac monitoring mandatory during IV administration",
            "Lifelong L-carnitine mandatory — stopping → carnitine drops → myopathy/cardiomyopathy "
            "relapses; compliance monitoring critical especially in adolescents",
            "Hypoglycaemia: hypoketotic hypoglycaemia in childhood; avoid fasting; "
            "IV dextrose in hypoglycaemic crisis (unlike PFKM where glucose is contraindicated)",
        ],
        "treatment": (
            "L-carnitine: 100-200mg/kg/day PO in 3-4 divided doses — FIRST LINE; LIFELONG; "
            "plasma carnitine target 25-50 μmol/L (guide dosing). "
            "IV L-carnitine: 100mg/kg IV bolus in acute cardiomyopathy crisis + IV infusion. "
            "Cardiac: ACE inhibitor + beta-blocker if dilated cardiomyopathy before carnitine normalisation. "
            "Monitor cardiac function: echocardiogram monthly until stable on treatment. "
            "Hypoglycaemia: emergency IV 10% dextrose; avoid fasting; emergency protocol. "
            "NBS follow-up: early treatment before symptoms; compliance programme. "
            "Compliance: adolescent compliance critical — stopping = fatal relapse; "
            "adherence support essential."
        ),
        "monitoring": [
            "Plasma carnitine: free + total; monthly until stable, then quarterly (target 25-50 μmol/L)",
            "Cardiac: echocardiogram monthly until cardiac function normalises; then 6-monthly",
            "ECG: Holter if palpitations; QTc monitoring",
            "CK: baseline + quarterly (skeletal muscle response)",
            "Glucose: fasting glucose + ketones diary; hypoglycaemia frequency",
            "Renal: urine carnitine excretion (distinguish primary vs secondary deficiency)",
            "Compliance: plasma carnitine level — proxy compliance marker; adolescent monitoring",
            "Genetic: NBS follow-up; sibling testing; maternal carnitine check (heterozygote mothers "
            "may need supplementation in pregnancy)",
        ],
    },
]

# ──────────────────────────────────────────────────────────────────
# Patient simulation
# ──────────────────────────────────────────────────────────────────
def _make_cohort(gene_entry: dict, seed: int) -> list[dict]:
    random.seed(seed)
    gene = gene_entry["gene"]
    n = 40
    patients = []

    # Gene-specific parameter tables
    onset_ranges = {
        "PYGM":    [15, 15, 16, 17, 18, 18, 20, 20, 22, 25, 25, 28, 30, 35],
        "CPT2":    [10, 12, 15, 15, 18, 20, 20, 22, 25, 25, 28, 30, 35, 40],
        "ACADVL":  [0, 0, 0, 1, 2, 5, 8, 12, 15, 20, 25, 30, 35, 40],
        "ETFDH":   [5, 8, 10, 12, 15, 18, 20, 20, 25, 28, 30, 35, 40, 45],
        "HADHA":   [0, 0, 1, 2, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35],
        "PFKM":    [10, 12, 14, 15, 15, 16, 18, 18, 20, 22, 25, 30, 35, 40],
        "AMPD1":   [20, 22, 25, 25, 28, 30, 30, 32, 35, 35, 38, 40, 42, 45],
        "SLC22A5": [0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15],
    }

    ck_ranges = {
        "PYGM":    (300, 15000),
        "CPT2":    (80, 20000),
        "ACADVL":  (100, 25000),
        "ETFDH":   (500, 5000),
        "HADHA":   (500, 10000),
        "PFKM":    (500, 10000),
        "AMPD1":   (200, 2000),
        "SLC22A5": (150, 3000),
    }

    ambulant_prob = {
        "PYGM": 0.95, "CPT2": 0.90, "ACADVL": 0.75,
        "ETFDH": 0.80, "HADHA": 0.85, "PFKM": 0.95,
        "AMPD1": 0.98, "SLC22A5": 0.85,
    }

    ventilator_prob = {
        "PYGM": 0.05, "CPT2": 0.05, "ACADVL": 0.12,
        "ETFDH": 0.15, "HADHA": 0.10, "PFKM": 0.05,
        "AMPD1": 0.02, "SLC22A5": 0.08,
    }

    cardiac_prob = {
        "PYGM": 0.10, "CPT2": 0.15, "ACADVL": 0.40,
        "ETFDH": 0.20, "HADHA": 0.25, "PFKM": 0.10,
        "AMPD1": 0.03, "SLC22A5": 0.60,
    }

    triggers = {
        "PYGM":    ["sustained_exercise", "static_isometric_exercise"],
        "CPT2":    ["prolonged_fasting", "fever", "cold_exposure", "prolonged_exercise"],
        "ACADVL":  ["fasting", "illness_fever", "prolonged_exercise"],
        "ETFDH":   ["fasting", "illness", "high_fat_meal"],
        "HADHA":   ["fasting", "fat_loading", "illness_fever"],
        "PFKM":    ["high_carb_meal", "exercise_after_carb_load", "sustained_exercise"],
        "AMPD1":   ["exhaustive_exercise", "sustained_exercise"],
        "SLC22A5": ["fasting", "illness_fever", "none_if_treated"],
    }

    for i in range(n):
        pid = f"{gene}-{seed}-{i+1:03d}"
        onset_age = random.choice(onset_ranges.get(gene, [15, 20, 25, 30]))
        current_age = onset_age + random.randint(3, 35)
        ck_lo, ck_hi = ck_ranges.get(gene, (200, 5000))
        ck_peak = random.randint(ck_lo, ck_hi)
        ambulant = random.random() < ambulant_prob.get(gene, 0.80)
        ventilator = random.random() < ventilator_prob.get(gene, 0.10)
        cardiac = random.random() < cardiac_prob.get(gene, 0.15)
        rhabdo_episodes = random.randint(0, 12) if gene in ("PYGM", "CPT2", "ACADVL", "PFKM") else random.randint(0, 4)
        key_trigger = random.choice(triggers.get(gene, ["exercise", "fasting"]))

        # Gene-specific extras
        second_wind = (gene == "PYGM") and random.random() < 0.88
        riboflavin_responsive = (gene == "ETFDH") and random.random() < 0.80
        retinopathy = (gene == "HADHA") and random.random() < 0.70
        neuropathy = (gene == "HADHA") and random.random() < 0.65
        hemolytic_anemia = (gene == "PFKM") and random.random() < 0.85
        nbs_detected = (gene in ("ACADVL", "SLC22A5")) and random.random() < 0.90
        carnitine_responsive = (gene == "SLC22A5") and random.random() < 0.95

        patients.append({
            "id": pid,
            "gene": gene,
            "onset_age": onset_age,
            "current_age": current_age,
            "ck_peak_iul": ck_peak,
            "ambulant": ambulant,
            "ventilator": ventilator,
            "cardiac_involvement": cardiac,
            "rhabdo_episodes": rhabdo_episodes,
            "key_trigger": key_trigger,
            "second_wind": second_wind,
            "riboflavin_responsive": riboflavin_responsive,
            "retinopathy": retinopathy,
            "peripheral_neuropathy": neuropathy,
            "hemolytic_anemia": hemolytic_anemia,
            "nbs_detected": nbs_detected,
            "carnitine_responsive": carnitine_responsive,
        })
    return patients


def _aggregate_cohort():
    all_patients = []
    for idx, entry in enumerate(METABOLIC_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))
    return all_patients


# ──────────────────────────────────────────────────────────────────
# API response builders
# ──────────────────────────────────────────────────────────────────
def overview() -> dict:
    cohort = _aggregate_cohort()
    total = len(cohort)
    ambulant = sum(1 for p in cohort if p["ambulant"])
    ventilator = sum(1 for p in cohort if p["ventilator"])
    cardiac = sum(1 for p in cohort if p["cardiac_involvement"])
    second_wind = sum(1 for p in cohort if p["second_wind"])
    riboflavin_resp = sum(1 for p in cohort if p["riboflavin_responsive"])
    retinopathy = sum(1 for p in cohort if p["retinopathy"])
    hemolytic = sum(1 for p in cohort if p["hemolytic_anemia"])
    nbs = sum(1 for p in cohort if p["nbs_detected"])
    carnitine_resp = sum(1 for p in cohort if p["carnitine_responsive"])
    avg_onset = round(sum(p["onset_age"] for p in cohort) / total, 1)
    avg_ck = round(sum(p["ck_peak_iul"] for p in cohort) / total, 0)

    gene_counts = {}
    for g in METABOLIC_GENES:
        gn = g["gene"]
        subset = [p for p in cohort if p["gene"] == gn]
        gene_counts[gn] = {
            "n": len(subset),
            "ambulant_pct": round(100 * sum(1 for p in subset if p["ambulant"]) / len(subset)),
            "ventilator_pct": round(100 * sum(1 for p in subset if p["ventilator"]) / len(subset)),
            "cardiac_pct": round(100 * sum(1 for p in subset if p["cardiac_involvement"]) / len(subset)),
            "avg_onset": round(sum(p["onset_age"] for p in subset) / len(subset), 1),
            "avg_ck_peak": round(sum(p["ck_peak_iul"] for p in subset) / len(subset), 0),
            "avg_rhabdo_episodes": round(
                sum(p["rhabdo_episodes"] for p in subset) / len(subset), 1
            ),
        }

    return {
        "atlas": "Hereditary-Metabolic-Myopathy-Atlas",
        "subtitle": "Complete 8-Gene Metabolic Myopathy Spectrum Atlas",
        "genes": [g["gene"] for g in METABOLIC_GENES],
        "gene_count": len(METABOLIC_GENES),
        "total_patients": total,
        "seeds": list(range(SEED_BASE, SEED_BASE + len(METABOLIC_GENES))),
        "kpis": {
            "total_patients": total,
            "ambulant_pct": round(100 * ambulant / total),
            "ventilator_pct": round(100 * ventilator / total),
            "cardiac_involvement_pct": round(100 * cardiac / total),
            "second_wind_pct": round(100 * second_wind / total),
            "riboflavin_responsive_pct": round(100 * riboflavin_resp / total),
            "retinopathy_pct": round(100 * retinopathy / total),
            "hemolytic_anemia_pct": round(100 * hemolytic / total),
            "nbs_detected_pct": round(100 * nbs / total),
            "carnitine_responsive_pct": round(100 * carnitine_resp / total),
            "avg_onset_years": avg_onset,
            "avg_ck_peak_iul": int(avg_ck),
        },
        "gene_summary": gene_counts,
        "pathognomonic_features": {
            "PYGM":    "SECOND WIND PHENOMENON — exercise fatigue at 8-10 min, recovery at 11-15 min",
            "CPT2":    "THERMOGENIC TRIGGERS — fever+fasting+cold+exercise → rhabdomyolysis",
            "ACADVL":  "C14:1 ELEVATED on NBS acylcarnitine; cardiomyopathy in severe neonatal form",
            "ETFDH":   "RIBOFLAVIN 100-300mg/day ESSENTIALLY CURATIVE — riboflavin-responsive MADD",
            "HADHA":   "PERIPHERAL RETINOPATHY + PERIPHERAL NEUROPATHY — unique extra-muscular combination; AFLP maternal risk",
            "PFKM":    "HEMOLYTIC ANEMIA + EXERCISE MYOPATHY; HIGH-CARB PARADOX — IV glucose CONTRAINDICATED",
            "AMPD1":   "AMMONIA FAILS TO RISE on forearm test (opposite of McArdle); generally benign",
            "SLC22A5": "L-CARNITINE ESSENTIALLY CURATIVE; cardiomyopathy REVERSIBLE with treatment; NBS detected",
        },
        "inheritance_map": {
            g["gene"]: "AR" for g in METABOLIC_GENES
        },
        "protein_sizes": {g["gene"]: g["protein_size"] for g in METABOLIC_GENES},
        "loci": {g["gene"]: g["locus"] for g in METABOLIC_GENES},
        "key_pharmacological_distinctions": [
            "PYGM: GLUCOSE/SUCROSE TRICK (10-15g before exercise); glucose IV BENEFICIAL in crisis; aerobic exercise beneficial",
            "CPT2: HIGH-CARB DIET beneficial; L-carnitine controversial; triheptanoin (C7) anaplerosis; avoid prolonged fasting",
            "ACADVL: MCT supplementation + triheptanoin; avoid LCFA; avoid fasting; IV dextrose in crisis",
            "ETFDH: RIBOFLAVIN 100-300mg/day ESSENTIALLY CURATIVE — must trial before calling refractory; CoQ10 adjunctive",
            "HADHA: MCT supplements; avoid long-chain fat; ophthalmology 6-monthly; screen mother for AFLP",
            "PFKM: HIGH-CARB PARADOX — IV GLUCOSE ABSOLUTELY CONTRAINDICATED in crisis; give IV fat+protein instead",
            "AMPD1: No specific treatment; generally benign; exclude McArdle (PYGM) if CK high",
            "SLC22A5: L-carnitine 100-200mg/kg/day PO LIFELONG curative; IV carnitine in crisis; cardiomyopathy reversible",
        ],
        "critical_treatment_alerts": {
            "PYGM": "GLUCOSE HELPFUL (opposite of PFKM); sucrose trick pre-exercise; avoid succinylcholine",
            "CPT2": "Triheptanoin FDA-approved; high-carb diet; aggressive fever management with carb loading",
            "ACADVL": "NBS early treatment prevents retinopathy-like complications; triheptanoin preferred MCT",
            "ETFDH": "RIBOFLAVIN trial MANDATORY — 100-300mg/day; CoQ10 200-300mg/day adjunctive",
            "HADHA": "Ophthalmology 6-monthly; AFLP obstetric emergency; avoid LCT loading",
            "PFKM": "GLUCOSE IV ABSOLUTELY CONTRAINDICATED in crisis — give IV lipid + amino acids instead",
            "AMPD1": "Reassurance; no evidence-based treatment; exclude secondary AMPD1; pacing strategies",
            "SLC22A5": "L-carnitine lifelong; stopping = FATAL relapse; IV carnitine in cardiac crisis",
        },
    }


def breakdown() -> dict:
    cohort = _aggregate_cohort()
    patients_out = []
    for p in cohort:
        entry = next(g for g in METABOLIC_GENES if g["gene"] == p["gene"])
        patients_out.append({
            **p,
            "protein": entry["protein"],
            "alt_name": entry["alt_name"],
            "inheritance": entry["inheritance"],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:300],
        })
    return {
        "atlas": "Hereditary-Metabolic-Myopathy-Atlas",
        "total": len(patients_out),
        "patients": patients_out,
        "gene_profiles": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "key_features": g["key_features"],
                "treatment": g["treatment"],
                "monitoring": g["monitoring"],
            }
            for g in METABOLIC_GENES
        ],
    }


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Metabolic-Myopathy-Atlas",
        "glossary": {
            "Metabolic Myopathy": (
                "Hereditary muscle disease caused by enzyme defects in energy metabolism pathways "
                "(glycogenolysis, glycolysis, fatty acid oxidation, purine nucleotide cycle, carnitine transport). "
                "Clinical hallmark: exercise intolerance with or without rhabdomyolysis; CK elevated in crisis. "
                "Key distinction from structural myopathies: biopsy may show glycogen/lipid storage "
                "or be near-normal between episodes."
            ),
            "Second Wind Phenomenon (PYGM/McArdle)": (
                "PATHOGNOMONIC for McArdle Disease (PYGM). "
                "Phase 1: fatigue + cramps at 8-10 min exercise (muscle glycogenolysis blocked, no fuel). "
                "Phase 2: spontaneous recovery at 11-15 min as hepatic glycogenolysis + FFA mobilisation "
                "delivers alternative substrate to muscle. "
                "No second wind in CPT2, PFKM, ACADVL."
            ),
            "Forearm Ischemic Test": (
                "Diagnostic test for metabolic myopathy: forearm exercise under ischemia (cuff inflated); "
                "measure venous lactate + ammonia at 0, 1, 3, 5, 10 min post-exercise. "
                "PYGM (McArdle): lactate FAILS TO RISE, ammonia RISES normally. "
                "AMPD1: lactate RISES normally, ammonia FAILS TO RISE (opposite). "
                "Safety note: ischemic test can cause rhabdomyolysis — non-ischemic forearm test preferred."
            ),
            "Thermogenic Triggers (CPT2)": (
                "PATHOGNOMONIC for CPT2 deficiency: any combination of fever + prolonged fasting + "
                "cold exposure + sustained exercise precipitates rhabdomyolysis. "
                "Mechanism: long-chain FA oxidation blocked at CPT2 → energy crisis in sustained exercise "
                "when glucose depleted. "
                "CPT1 (liver) normal → hepatic beta-oxidation intact but muscle CPT2 absent."
            ),
            "High-Carbohydrate Paradox (PFKM/Tarui)": (
                "PATHOGNOMONIC clinical danger in Tarui Disease (PFKM). "
                "Mechanism: IV/oral glucose → insulin → suppresses FFA release → "
                "the only alternative fuel (FFAs) is blocked → worsens energy crisis. "
                "Treatment: GIVE FAT + PROTEIN, NOT GLUCOSE in acute PFKM crisis. "
                "IV lipid emulsion (Intralipid) + amino acids preferred. "
                "OPPOSITE of McArdle/GSD III where glucose IV is BENEFICIAL."
            ),
            "Riboflavin-Responsive MADD (ETFDH)": (
                "Most clinically important treatable metabolic myopathy. "
                "ETFDH missense mutations → reduced FAD binding affinity → low ETF dehydrogenase activity. "
                "Riboflavin (Vitamin B2) supplementation increases FAD availability → restores enzyme activity. "
                "Riboflavin 100-300mg/day: ~80% response rate in ETFDH missense — strength improves, "
                "CK normalises within weeks. MUST TRIAL before labelling MADD refractory."
            ),
            "AFLP (Acute Fatty Liver of Pregnancy) — HADHA": (
                "Obstetric emergency caused by maternal heterozygote carrying LCHAD-affected fetus. "
                "Long-chain 3-hydroxy acylcarnitines from affected fetus accumulate in maternal liver. "
                "Maternal hepatotoxicity → AFLP — potentially fatal to mother. "
                "All women with AFLP: test infant for LCHAD (HADHA/G1528C). "
                "Early delivery mandatory once diagnosis confirmed."
            ),
            "Primary Carnitine Deficiency (SLC22A5/OCTN2)": (
                "OCTN2 carnitine transporter loss → failure to reabsorb carnitine in kidney → "
                "plasma free carnitine <5 μmol/L (normal 25-50 μmol/L). "
                "Clinical: cardiomyopathy (hypertrophic → dilated), hypoglycaemia, skeletal muscle weakness. "
                "L-carnitine 100-200mg/kg/day PO ESSENTIALLY CURATIVE — cardiomyopathy reverses. "
                "NBS: low C0 detected early; treatment before symptoms = normal life expectancy."
            ),
            "Acylcarnitine Profile": (
                "Dried blood spot or plasma tandem mass spectrometry: measures acylcarnitine species C0-C26. "
                "ACADVL: C14:1 (tetradecenoylcarnitine) elevated — pathognomonic. "
                "CPT2: C16, C18:1 elevated. "
                "HADHA: C16-OH, C18:1-OH (3-hydroxy species) elevated. "
                "SLC22A5: C0 (free carnitine) critically low. "
                "ETFDH/MADD: multiple species elevated (C4-C10 + glutarylcarnitine)."
            ),
            "McArdle Disease (GSD V)": (
                "Muscle glycogen phosphorylase (PYGM) deficiency — glycogenolysis blocked in skeletal muscle. "
                "p.Arg50Stop: ~90% European alleles — PCR screen first-line. "
                "CK persistently elevated at rest (300-2000 IU/L); 1000-50000× in crisis. "
                "SECOND WIND pathognomonic. Forearm test: lactate fails to rise. "
                "Glucose/sucrose trick prevents second wind delay."
            ),
            "Tarui Disease (GSD VII)": (
                "Muscle phosphofructokinase (PFKM) deficiency — glycolysis blocked at PFK step. "
                "DUAL MANIFESTATION: exercise myopathy + haemolytic anaemia (PFK-M in RBCs). "
                "HIGH-CARB PARADOX: glucose IV contraindicated in crisis. "
                "p.Arg232His + p.Trp249Stop: Ashkenazi Jewish founders. "
                "Uric acid elevated; gout risk; no second wind."
            ),
            "VLCAD Deficiency": (
                "Very-long-chain acyl-CoA dehydrogenase (ACADVL) deficiency — first FAO enzyme. "
                "3 forms: severe neonatal (cardiomyopathy), childhood hepatic, mild adult (rhabdomyolysis). "
                "NBS: C14:1 elevated on acylcarnitine. "
                "Treatment: MCT/triheptanoin; avoid fasting; IV dextrose in crisis."
            ),
            "Triheptanoin (C7 MCT)": (
                "Odd-chain medium-chain triglyceride — C7 fatty acid. "
                "Bypasses long-chain FAO block (CPT2, VLCAD, LCHAD). "
                "Provides propionyl-CoA for TCA cycle anaplerosis (unlike even-chain MCTs). "
                "FDA-approved (Dojolvi) for LC-FAO disorders including VLCAD and CPT2. "
                "Dose: 1-2g/kg/day (max 35g/day) in divided doses."
            ),
            "Lipid Storage Myopathy": (
                "Excess lipid vacuoles in muscle fibres on oil red O stain. "
                "Seen in: ETFDH/MADD, HADHA, CPT2, ACADVL (when untreated). "
                "Mechanism: impaired long-chain FAO → lipid accumulates in sarcoplasm. "
                "Distinct from glycogen storage (PAS stain in PYGM/PFKM). "
                "Lipid myopathy + riboflavin response = ETFDH until proven otherwise."
            ),
            "Myoadenylate Deaminase Deficiency (AMPD1)": (
                "AMP deaminase muscle isoform deficiency — purine nucleotide cycle blocked. "
                "AMMONIA fails to rise on forearm test (opposite of McArdle). "
                "p.Gln12Stop: 2% European carrier frequency — most common AMPD1 mutation. "
                "Generally benign — post-exercise myalgia only; no rhabdomyolysis crisis. "
                "No specific treatment; exclude secondary AMPD1."
            ),
            "Newborn Screening (NBS) for Metabolic Myopathies": (
                "Tandem mass spectrometry acylcarnitine profile on dried blood spot at 24-48h of life. "
                "Detects: ACADVL (C14:1 elevated), SLC22A5 (C0 low), HADHA (C16-OH elevated), "
                "CPT2 (C16/C18:1 elevated), ETFDH/MADD (multiple acylcarnitines). "
                "NBS dramatically improves outcomes — ACADVL cardiomyopathy prevented by early MCT diet."
            ),
        },
        "diagnostic_algorithm": [
            "1. Exercise intolerance + cramps + rhabdomyolysis → metabolic myopathy screen",
            "2. Forearm ischemic/non-ischemic test: "
            "   lactate fails to rise + ammonia rises → GLYCOLYTIC BLOCK (PYGM, PFKM); "
            "   ammonia fails to rise + lactate rises → PURINE CYCLE BLOCK (AMPD1)",
            "3. Acylcarnitine profile (plasma/DBS): C14:1↑→ACADVL; C16/C18:1↑→CPT2; "
            "   C16-OH↑→HADHA; C0↓→SLC22A5; multiple↑→ETFDH/MADD",
            "4. Urine organic acids: glutaric acid + multiple acylcarnitines → ETFDH/MADD; "
            "   organic acid screen mandatory in all metabolic myopathy workup",
            "5. Plasma free carnitine: <5 μmol/L → primary carnitine deficiency (SLC22A5); "
            "   low with urine carnitine elevated = primary; low with urine carnitine normal = secondary",
            "6. Trigger analysis: thermogenic triggers (fever/fasting/cold) → CPT2; "
            "   high-carb meal + exercise → PFKM; sustained static exercise → PYGM",
            "7. CK pattern: persistently elevated at rest → PYGM (300-2000 baseline); "
            "   normal between episodes → CPT2/ACADVL; mild 200-2000 → AMPD1",
            "8. Extra-muscular features: retinopathy + neuropathy → HADHA; "
            "   haemolytic anaemia → PFKM; cardiomyopathy → SLC22A5/ACADVL",
            "9. Gene panel: PYGM (include p.Arg50Stop PCR), CPT2, ACADVL, ETFDH, HADHA, PFKM, AMPD1, SLC22A5",
            "10. Riboflavin trial (ETFDH/MADD confirmed): 100-300mg/day × 8 weeks — "
            "    response confirms riboflavin-responsive MADD",
            "11. AFLP history in mother → test infant for HADHA G1528C urgently",
            "12. NBS positive for C14:1/C0/C16-OH → initiate diet immediately; confirm with sequencing",
        ],
        "references": [
            "Quinlivan R et al. McArdle disease: a clinical review. J Neurol Neurosurg Psychiatry. 2010.",
            "Ørngreen MC et al. CPT2 deficiency: update on diagnosis and management. JIMD Rep. 2016.",
            "Vockley J et al. VLCAD deficiency: outcomes of newborn screening. Genet Med. 2016.",
            "Liang WC et al. Riboflavin-responsive MADD and ETFDH mutations. Brain. 2009.",
            "Tyni T et al. LCHAD deficiency and maternal AFLP. J Inherit Metab Dis. 1998.",
            "Tarui S et al. Phosphofructokinase deficiency in skeletal muscle. Biochem Biophys Res Commun. 1965.",
        ],
        "standards": [
            "ACMG/AMP Variant Interpretation Framework (Richards et al. 2015)",
            "SSIEM (Society for the Study of Inborn Errors of Metabolism) Metabolic Myopathy Guidelines",
            "RCPCH Newborn Bloodspot Screening Programme — LC-FAOD panel",
            "FDA Approval: Dojolvi (triheptanoin) for LC-FAOD including VLCAD and CPT2 (2020)",
            "IEM Emergency Protocol — BIMDG (British Inherited Metabolic Disease Group) emergency guidelines",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== DEFINITIONS ===")
    defs = definitions()
    print(json.dumps({"atlas": defs["atlas"], "glossary_keys": list(defs["glossary"].keys())}, indent=2))
    print("\n=== BREAKDOWN (first patient) ===")
    bk = breakdown()
    print(json.dumps({"total": bk["total"], "first_patient": bk["patients"][0]}, indent=2))
