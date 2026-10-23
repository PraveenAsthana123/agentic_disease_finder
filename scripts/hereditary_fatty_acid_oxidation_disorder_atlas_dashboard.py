#!/usr/bin/env python3
"""Hereditary-Fatty-Acid-Oxidation-Disorder-Atlas -- Complete 8-Gene Reference
Fatty Acid Oxidation (FAO) Disorders: ACADVL-ACADM-HADHA-ACADS-CPT1A-CPT2-SLC25A20-HADH
(ACADVL/VLCAD-Deficiency-C14:1-ELEVATED-PATHOGNOMONIC-NBS-DETECTABLE-Cardiac+Hypoglycemia-MCT-Avoid-Fasting,
 ACADM/MCAD-Deficiency-MOST-COMMON-FAOD-C8-C6-C10-NBS-DETECTABLE-Fasting-Hypoglycemia-Sudden-Death-Pre-NBS,
 HADHA/LCHAD-TFP-Deficiency-3-OH-C16-ELEVATED-PATHOGNOMONIC-RHABDOMYOLYSIS-Retinopathy-PATHOGNOMONIC-Maternal-AFLP,
 ACADS/SCAD-Deficiency-C4-Elevated-NBS-Often-BENIGN-EthylmaloniCAcid-Urine-Riboflavin,
 CPT1A/CPT1A-Deficiency-C0-HIGH-PATHOGNOMONIC-Long-Chain-AC-LOW-Arctic-Founder-Hypoketotic-Hypoglycemia,
 CPT2/CPT2-Deficiency-Most-Common-CPT-Deficiency-Myopathic-Adolescent-Rhabdomyolysis-Exercise,
 SLC25A20/CACT-Deficiency-Severe-Neonatal-FAOD-Long-Chain-AC-Elevated-Cardiac-Arrhythmia,
 HADH/SCHAD-Deficiency-Congenital-Hyperinsulinism-DDx-Hypoglycemia-3-OH-Butyrylcarnitine-Elevated):

ACADVL (Very long-chain acyl-CoA dehydrogenase; 655aa; 17p13.1; AR-LOF;
     ACADVL encodes very long-chain acyl-CoA dehydrogenase (VLCAD), the outer mitochondrial membrane enzyme
     catalyzing the first step of long-chain fatty acid beta-oxidation (C14-C24);
     VLCAD is the primary enzyme for very long-chain fatty acids in heart and skeletal muscle;
     VLCAD LOF -> accumulation of C14:1-acylcarnitine (tetradecenoylcarnitine) -> NBS-detectable;
     VLCAD DEFICIENCY -- OMIM 201475;
     THREE PHENOTYPES by residual activity:
       SEVERE CARDIAC (neonatal): pericardial effusion, hypertrophic cardiomyopathy, arrhythmias, hypoglycemia;
       HEPATIC (childhood): fasting-induced hypoglycemia, hepatomegaly, without primary cardiac disease;
       MYOPATHIC (adult): rhabdomyolysis triggered by prolonged exercise, fasting, illness;
     KEY BIOMARKER: C14:1-acylcarnitine ELEVATED on NBS/plasma acylcarnitine profile -> PATHOGNOMONIC;
       C14:1 is the signature acylcarnitine of VLCAD deficiency; C14:0 and C14:2 also elevated;
     NBS: DETECTED by expanded NBS (C14:1 elevation on tandem MS) in most countries;
     CRITICAL DISTINCTION from LCHAD/TFP: VLCAD lacks retinopathy and peripheral neuropathy (LCHAD features);
       VLCAD: C14:1 dominant; LCHAD: 3-OH-C16 dominant + retinopathy;
     TREATMENT: avoid fasting (critical) -- feed every 4-6h; MCT supplementation controversial for cardiac/myopathic;
       for severe neonatal cardiac: IV glucose + MCT formula; cornstarch for older children (slow-release glucose);
       AVOID prolonged fasting and excessive medium-chain fat (accumulation of C8-C12 may be toxic);
     MONITORING: echocardiogram, plasma acylcarnitines, CK (myopathic), ophthalmology (unlike LCHAD, retinopathy rare);
     seed SEED_BASE+0).

ACADM (Medium-chain acyl-CoA dehydrogenase; 421aa; 1p31.1; AR-LOF;
     ACADM encodes medium-chain acyl-CoA dehydrogenase (MCAD), the mitochondrial matrix enzyme catalyzing
     the first step of medium-chain fatty acid beta-oxidation (C4-C12);
     MCAD DEFICIENCY -- OMIM 201450;
     MOST COMMON FATTY ACID OXIDATION DISORDER (1:10,000-1:15,000 Northern European); carrier frequency 1:70;
     PATHOPHYSIOLOGY: fasting/illness -> impaired ketogenesis (MCAD needed for acetyl-CoA from medium-chain FA)
       -> hypoketotic hypoglycemia -> fatal encephalopathy if untreated; NOT ketotic (absence of ketones is pathognomonic);
     KEY BIOMARKER: C8-acylcarnitine (octanoylcarnitine) MARKEDLY ELEVATED -> PATHOGNOMONIC;
       C6 and C10:1 also elevated; hexanoylglycine and phenylpropionylglycine in urine;
     FOUNDER MUTATION: p.K329E (985A>G) -- >80% of Northern European alleles; single mutation;
     NBS: DETECTED by expanded NBS (C8 elevation) -- BEFORE NBS implementation, MCAD was leading cause of
       sudden unexplained infant death (Reye syndrome phenocopy); NBS has dramatically reduced mortality;
     CLINICAL: often asymptomatic at birth; first crisis triggered by fasting + intercurrent illness (URTI);
       hypoglycemia WITHOUT ketones (hypoketotic hypoglycemia) PATHOGNOMONIC; Reye-like presentation;
       SUDDEN DEATH pre-NBS era in 25% of undiagnosed first presentations;
     TREATMENT: AVOID FASTING -- most important intervention; feed every 4-6h; increase frequency during illness;
       emergency protocol: 10% dextrose IV (2 mg/kg/min glucose infusion) during intercurrent illness;
       no specific diet restriction (medium-chain fats can continue -- different from VLCAD);
       carnitine supplementation: controversial (may worsen by driving acylcarnitines INTO mitochondria);
     PROGNOSIS: EXCELLENT with NBS diagnosis; virtually all NBS-detected patients asymptomatic with feeding protocol;
     seed SEED_BASE+1).

HADHA (Mitochondrial trifunctional protein alpha subunit; 763aa; 2p23.3; AR-LOF;
     HADHA encodes the alpha subunit of mitochondrial trifunctional protein (MTP/TFP);
     TFP is a hetero-octamer (4 alpha + 4 beta) catalyzing THREE long-chain FAO steps:
       (1) LCHAD: 3-hydroxyacyl-CoA dehydrogenase (long-chain); (2) LH: 3-hydroxyacyl-CoA hydratase (long-chain);
       (3) LCKAT: long-chain thiolase (encoded by HADHB -- beta subunit);
     HADHA mutations -> isolated LCHAD deficiency OR complete TFP deficiency depending on mutation location;
     LCHAD DEFICIENCY -- OMIM 609016; TFP DEFICIENCY -- OMIM 609015;
     KEY BIOMARKER: 3-OH-C16-acylcarnitine (3-hydroxypalmitoylcarnitine) ELEVATED -> PATHOGNOMONIC;
       3-OH-C16, 3-OH-C18, 3-OH-C18:1 elevated on plasma acylcarnitines; LCHAD-specific;
     PATHOGNOMONIC FEATURE 1: RETINOPATHY (pigmentary retinopathy) -- PATHOGNOMONIC for LCHAD/TFP;
       progressive retinal degeneration leading to visual impairment; NOT seen in VLCAD, MCAD, SCAD;
       ophthalmology baseline at diagnosis; annual follow-up mandatory;
     PATHOGNOMONIC FEATURE 2: PERIPHERAL NEUROPATHY -- sensorimotor, progressive; muscles weakness + sensory loss;
     PATHOGNOMONIC FEATURE 3: RHABDOMYOLYSIS -- exercise or illness-triggered; CK markedly elevated;
       rhabdomyolysis + retinopathy in child/adolescent = LCHAD/TFP until proven otherwise;
     MATERNAL FATTY LIVER: mothers of LCHAD-affected fetuses develop AFLP (acute fatty liver of pregnancy)
       or HELLP syndrome in 20-25% -- PATHOGNOMONIC maternal-fetal relationship;
       mechanism: fetal LCHAD-deficient placenta accumulates 3-hydroxy fatty acids -> maternal hepatotoxicity;
       RETROGRADE DIAGNOSIS: any woman with AFLP/HELLP should trigger newborn LCHAD screening;
     NBS: DETECTED (3-OH-C16 on tandem MS);
     TREATMENT: LOW-LONG-CHAIN FAT diet with MCT supplementation (MCT bypasses LCHAD/TFP requirement);
       avoid fasting; ophthalmology monitoring; DHA (docosahexaenoic acid) supplementation (reduces retinopathy);
     seed SEED_BASE+2).

ACADS (Short-chain acyl-CoA dehydrogenase; 412aa; 12q24.31; AR-LOF;
     ACADS encodes short-chain acyl-CoA dehydrogenase (SCAD), the mitochondrial matrix enzyme catalyzing
     the first step of short-chain fatty acid beta-oxidation (C4-C6, primarily butyryl-CoA = C4);
     SCAD DEFICIENCY -- OMIM 201470;
     OFTEN CLINICALLY BENIGN / BIOCHEMICAL PHENOTYPE ONLY:
       most SCAD-deficient patients identified by NBS are asymptomatic throughout life;
       some have non-specific symptoms: hypotonia, developmental delay, seizures (causal relationship debated);
       metabolic decompensation is RARE (contrast with MCAD/VLCAD where crises are common);
     KEY BIOMARKER: C4-acylcarnitine (butyrylcarnitine) ELEVATED on NBS/plasma;
       ethylmalonic acid (EMA) in urine -- ELEVATED in SCAD deficiency;
     IMPORTANT DDx: EMA is also elevated in MADD (Multiple Acyl-CoA Dehydrogenase Deficiency / GA2);
       C4 elevation alone does not distinguish SCAD from MADD; MADD has MULTIPLE acylcarnitines elevated;
     COMMON VARIANTS: p.511C>T (c.319C>T) and p.625G>A (c.164C>T) -- disease-modifying polymorphisms;
       these variants reduce enzyme activity but do NOT cause disease alone; interact with other factors;
     NBS: DETECTED (C4 elevation) but clinical significance uncertain -- many programs do NOT treat unless symptomatic;
     TREATMENT: riboflavin (cofactor) -- some SCAD-deficient patients respond; low-fat diet generally not required;
       carnitine supplementation; avoid fasting as general precaution; management largely symptomatic;
     seed SEED_BASE+3).

CPT1A (Carnitine palmitoyltransferase 1A, liver isoform; 773aa; 11q13.3; AR-LOF;
     CPT1A encodes carnitine palmitoyltransferase 1A (liver isoform), the outer mitochondrial membrane enzyme
     that catalyzes the RATE-LIMITING step of long-chain fatty acid transport into the mitochondrial matrix:
     palmitoyl-CoA + L-carnitine -> palmitoylcarnitine + CoA-SH (step 1 of carnitine shuttle);
     CPT1A DEFICIENCY -- OMIM 255120;
     UNIQUE BIOMARKER: C0 (free carnitine) HIGH + long-chain acylcarnitines (C12-C18) LOW -> PATHOGNOMONIC;
       the acylcarnitine profile is opposite to CPT2/CACT: C0 very high (carnitine NOT consumed), LC-AC low;
       this distinguishes CPT1A from all other FAODs where LC-acylcarnitines are elevated;
     ARCTIC FOUNDER MUTATION: p.P479L (c.1436C>T) -- EXTREMELY COMMON in Inuit/First Nations/Alaska Native;
       allele frequency 0.58-0.70 in some Arctic populations (homozygous rate ~25%!);
       THIS VARIANT MAY BE ADAPTIVE in Arctic populations (promotes thermogenesis; different risk/benefit);
       NBS programs in Arctic regions must carefully interpret CPT1A results;
     CLINICAL: hypoketotic hypoglycemia (most common presentation); hepatomegaly; hepatocellular dysfunction;
       RENAL TUBULAR ACIDOSIS -- uncommon feature distinguishing CPT1A from other hepatic FAODs;
     TREATMENT: AVOID FASTING; cornstarch (uncooked) for sustained glucose release; carnitine may worsen
       (theoretical -- could drive C0 INTO defective pathway but rarely clinically relevant);
       MCT diet RELATIVELY CONTRAINDICATED -- CPT1A is needed for C8+ fatty acids; MCT bypasses CPT1A
       BUT some MCT still requires CPT1A for entry; low-long-chain fat + continuous feeds;
     seed SEED_BASE+4).

CPT2 (Carnitine palmitoyltransferase 2; 658aa; 1p32.3; AR-LOF;
     CPT2 encodes carnitine palmitoyltransferase 2 (CPT2), the inner mitochondrial membrane enzyme
     catalyzing the second step of the carnitine shuttle:
     palmitoylcarnitine -> palmitoyl-CoA + L-carnitine (regenerates long-chain acyl-CoA inside matrix);
     CPT2 DEFICIENCY -- OMIM 600650;
     MOST COMMON DISORDER OF LONG-CHAIN FATTY ACID TRANSPORT (combined with CACT and CPT1A);
     THREE CLINICAL PHENOTYPES:
       LETHAL NEONATAL (rare): cardiac malformations, arrhythmias, respiratory failure, brain migration defects;
       SEVERE INFANTILE HEPATOCARDIOMUSCULAR (rare): like CACT -- hepatomegaly, cardiomyopathy, hypoglycemia;
       MYOPATHIC (most common, ~75%): adolescent/adult; exercise + cold + illness-triggered RHABDOMYOLYSIS;
         muscle pain and weakness; CK markedly elevated during attacks; renal failure (myoglobinuria);
     KEY BIOMARKER: C16:0, C18:0, C18:1-acylcarnitines ELEVATED; C0 LOW (free carnitine depleted);
       long-chain acylcarnitine elevation pattern (similar to CACT but milder in myopathic form);
     FOUNDER MUTATION in myopathic form: p.S113L (c.338C>T) -- ~60-70% of myopathic CPT2 alleles;
     NBS: DETECTED in neonatal forms (C16/C18 elevated); myopathic form often NOT detected (normal NBS at birth);
     TREATMENT: AVOID TRIGGERS (prolonged exercise, cold exposure, fasting, fever);
       carnitine supplementation; high-carbohydrate diet (spares FA oxidation);
       bezafibrate (experimental) -- reported to increase CPT2 residual activity in myopathic;
       acute rhabdomyolysis: IV hydration (prevent myoglobinuric renal failure); bicarbonate for urine alkalinization;
     seed SEED_BASE+5).

SLC25A20 (Carnitine-acylcarnitine translocase; 301aa; 3p21.31; AR-LOF;
     SLC25A20 encodes carnitine-acylcarnitine translocase (CACT), the inner mitochondrial membrane transporter
     that exchanges matrix free carnitine for cytoplasmic long-chain acylcarnitine (step 2 of carnitine shuttle);
     CACT DEFICIENCY -- OMIM 212138;
     SEVERE NEONATAL PRESENTATION -- one of most severe FAODs:
       neonatal onset within 24-48h of birth: cardiac arrhythmias (often fatal), cardiomyopathy,
       hypoglycemia, hyperammonemia, elevated transaminases, coma;
       CARDIAC ARRHYTHMIAS are the primary cause of death (different from VLCAD where cardiomyopathy dominates);
     KEY BIOMARKER: LONG-CHAIN ACYLCARNITINES MARKEDLY ELEVATED (C16, C18, C18:1);
       FREE CARNITINE (C0) LOW (depleted by accumulating long-chain acylcarnitines);
       hyperammonemia common (secondary to liver dysfunction -- different from UCD primary hyperammonemia);
     NBS: DETECTED (C16/C18 elevated); acylcarnitine profile overlaps with CPT2 neonatal form;
     DISTINGUISHING CPT2 vs CACT: molecular testing required; phenotypically similar neonatal presentation;
       CACT has slightly more severe cardiac arrhythmia; both have long-chain AC elevation;
     TREATMENT: emergency IV glucose (2-4 mg/kg/min); anti-arrhythmic management; carnitine IV;
       long-term: LOW LONG-CHAIN FAT diet with MCT formula; frequent feeds; avoid fasting;
       carnitine supplementation: essential (replenish depleted C0);
       PROGNOSIS: poor in severe neonatal form -- mortality high even with treatment; survivors may have
       neurodevelopmental sequelae from hypoglycemic episodes;
     seed SEED_BASE+6).

HADH (Short-chain 3-hydroxyacyl-CoA dehydrogenase / SCHAD; 314aa; 4q25; AR-LOF;
     HADH encodes short-chain/medium-chain 3-hydroxyacyl-CoA dehydrogenase (SCHAD),
     the mitochondrial enzyme catalyzing the third step of short-chain beta-oxidation:
     L-3-hydroxyacyl-CoA + NAD+ -> 3-ketoacyl-CoA + NADH;
     HADH DEFICIENCY -- OMIM 609975; also called 3-hydroxyacyl-CoA dehydrogenase deficiency;
     UNIQUE PRESENTATION -- CONGENITAL HYPERINSULINISM (not classical FAOD hypoglycemia):
       HADH-deficient infants present with PROTEIN-SENSITIVE HYPERINSULINEMIC HYPOGLYCEMIA, not fasting-induced;
       MECHANISM: HADH normally inhibits glutamate dehydrogenase (GDH) in beta cells via protein-protein interaction;
         HADH LOF -> GDH uninhibited -> excess leucine (dietary protein) stimulates GDH -> excess insulin;
       THIS IS THE ONLY FAOD THAT CAUSES HYPERINSULINISM (opposite physiology from other FAODs);
     KEY BIOMARKER: 3-hydroxybutyrylcarnitine (C4-OH) ELEVATED in plasma;
       3-OH-glutaric acid may be elevated in urine; distinct from LCHAD (3-OH-C16 vs 3-OH-C4);
     DISTINGUISHING from other hyperinsulinism: protein-sensitive (leucine challenge triggers hypoglycemia);
       GDH-HI (GLUD1) also protein-sensitive but has hyperammonemia; HADH has NORMAL ammonia;
       normal ammonia + protein-sensitive hyperinsulinism + C4-OH elevated -> HADH until proven otherwise;
     NBS: may detect C4-OH elevation (not universally part of NBS panels);
     TREATMENT: diazoxide (KATP channel opener) -- RESPONDS to diazoxide (unlike KATP-channel HI from ABCC8/KCNJ11);
       avoid protein-rich meals; frequent feeds; leucine-restricted diet in severe cases;
       medical management usually adequate -- pancreatectomy rarely required (unlike KATP-HI);
     seed SEED_BASE+7).

SEED_BASE = 4222
"""

import json
import random
from typing import Any


ATLAS_ID   = "hereditary-fatty-acid-oxidation-disorder-atlas"
ATLAS_NAME = "Hereditary-Fatty-Acid-Oxidation-Disorder-Atlas"
SEED_BASE  = 4222
N_PER_GENE = 40

GENES = ["ACADVL", "ACADM", "HADHA", "ACADS", "CPT1A", "CPT2", "SLC25A20", "HADH"]

GENE_META = {
    "ACADVL":   {"aa": 655,  "locus": "17p13.1", "kda": 70,  "inh": "AR", "disease": "VLCAD Deficiency (Very Long-Chain Acyl-CoA Dehydrogenase Deficiency)",       "omim_gene": 609575, "omim_disease": 201475},
    "ACADM":    {"aa": 421,  "locus": "1p31.1",  "kda": 45,  "inh": "AR", "disease": "MCAD Deficiency (Medium-Chain Acyl-CoA Dehydrogenase Deficiency)",          "omim_gene": 607008, "omim_disease": 201450},
    "HADHA":    {"aa": 763,  "locus": "2p23.3",  "kda": 83,  "inh": "AR", "disease": "LCHAD / Mitochondrial Trifunctional Protein (TFP) Deficiency",              "omim_gene": 600890, "omim_disease": 609016},
    "ACADS":    {"aa": 412,  "locus": "12q24.31","kda": 44,  "inh": "AR", "disease": "SCAD Deficiency (Short-Chain Acyl-CoA Dehydrogenase Deficiency)",           "omim_gene": 606885, "omim_disease": 201470},
    "CPT1A":    {"aa": 773,  "locus": "11q13.3", "kda": 88,  "inh": "AR", "disease": "CPT1A Deficiency / Carnitine Palmitoyltransferase 1A Deficiency",           "omim_gene": 600528, "omim_disease": 255120},
    "CPT2":     {"aa": 658,  "locus": "1p32.3",  "kda": 74,  "inh": "AR", "disease": "CPT2 Deficiency / Carnitine Palmitoyltransferase 2 Deficiency",             "omim_gene": 600650, "omim_disease": 600650},
    "SLC25A20": {"aa": 301,  "locus": "3p21.31", "kda": 33,  "inh": "AR", "disease": "CACT Deficiency / Carnitine-Acylcarnitine Translocase Deficiency",          "omim_gene": 613698, "omim_disease": 212138},
    "HADH":     {"aa": 314,  "locus": "4q25",    "kda": 35,  "inh": "AR", "disease": "SCHAD Deficiency / Congenital Hyperinsulinism-Hypoglycemia (HAD type)",     "omim_gene": 601609, "omim_disease": 609975},
}

GENE_PATHOGNOMONIC = {
    "ACADVL":   "C14:1-acylcarnitine ELEVATED PATHOGNOMONIC (most specific marker); THREE phenotypes (cardiac/hepatic/myopathic) by residual activity; NBS-detectable; MCT oil controversial; AVOID fasting absolutely; retinopathy ABSENT (distinguishes from LCHAD)",
    "ACADM":    "C8-acylcarnitine (octanoylcarnitine) MARKEDLY ELEVATED PATHOGNOMONIC; MOST COMMON FAOD (1:10,000-15,000 N. European); hypoketotic hypoglycemia (NO ketones) PATHOGNOMONIC; NBS-detected; p.K329E founder >80% N.European alleles; AVOID FASTING primary treatment",
    "HADHA":    "3-OH-C16-acylcarnitine ELEVATED PATHOGNOMONIC; RETINOPATHY PATHOGNOMONIC (unique among FAODs); peripheral neuropathy; rhabdomyolysis; MATERNAL AFLP/HELLP in 20-25% of mothers (fetal LCHAD accumulates 3-OH FA -> maternal hepatotoxicity); DHA supplementation reduces retinopathy",
    "ACADS":    "C4-acylcarnitine (butyrylcarnitine) ELEVATED; ethylmalonic acid urine ELEVATED; OFTEN CLINICALLY BENIGN (most NBS-detected asymptomatic); SCAD polymorphisms p.511C>T and p.625G>A reduce activity but are NOT disease-causing alone; riboflavin cofactor trial",
    "CPT1A":    "C0 (free carnitine) MARKEDLY HIGH + long-chain acylcarnitines LOW PATHOGNOMONIC (OPPOSITE of other FAODs); ARCTIC FOUNDER p.P479L extremely common Inuit/First Nations (carrier frequency up to 70%); hypoketotic hypoglycemia + hepatomegaly; renal tubular acidosis uncommon",
    "CPT2":     "C16/C18/C18:1-acylcarnitines ELEVATED; C0 LOW; MOST COMMON CPT deficiency; MYOPATHIC form (>75%): adolescent rhabdomyolysis triggered by exercise/cold/fasting; p.S113L founder 60-70% myopathic alleles; bezafibrate experimental (increases residual activity)",
    "SLC25A20": "LONG-CHAIN acylcarnitines MARKEDLY ELEVATED + C0 VERY LOW; SEVERE NEONATAL cardiac ARRHYTHMIAS primary cause of death (distinguishes CACT from VLCAD cardiomyopathy); hyperammonemia secondary; molecular testing distinguishes CACT from CPT2 (similar acylcarnitine profile)",
    "HADH":     "PROTEIN-SENSITIVE HYPERINSULINISM (NOT fasting hypoglycemia) PATHOGNOMONIC; ONLY FAOD causing hyperinsulinism; HADH normally inhibits GDH in beta cells; HADH LOF -> GDH uninhibited -> leucine-stimulated excess insulin; NORMAL AMMONIA (DDx GLUD1-HI which has hyperammonemia); diazoxide RESPONSIVE",
}

GENE_GROUPS = {
    "ACADVL":   "long-chain-fao",
    "ACADM":    "medium-chain-fao",
    "HADHA":    "long-chain-fao",
    "ACADS":    "short-chain-fao",
    "CPT1A":    "carnitine-shuttle",
    "CPT2":     "carnitine-shuttle",
    "SLC25A20": "carnitine-shuttle",
    "HADH":     "short-chain-fao",
}

LONG_CHAIN_GROUP    = ["ACADVL", "HADHA"]
MEDIUM_CHAIN_GROUP  = ["ACADM"]
SHORT_CHAIN_GROUP   = ["ACADS", "HADH"]
CARNITINE_SHUTTLE   = ["CPT1A", "CPT2", "SLC25A20"]


def _make_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    meta = GENE_META[gene]

    nbs_detectable        = gene in ("ACADVL", "ACADM", "HADHA", "ACADS", "CPT2", "SLC25A20")
    has_retinopathy       = gene == "HADHA"
    has_rhabdomyolysis    = gene in ("ACADVL", "HADHA", "CPT2")
    has_cardiomyopathy    = gene in ("ACADVL", "SLC25A20")
    has_cardiac_arrhythmia= gene == "SLC25A20"
    has_hyperinsulinism   = gene == "HADH"
    is_benign_often       = gene == "ACADS"
    arctic_founder        = gene == "CPT1A"
    is_neonatal_severe    = gene in ("SLC25A20",)
    is_myopathic_adolescent = gene == "CPT2"
    c0_high               = gene == "CPT1A"
    c0_low                = gene in ("CPT2", "SLC25A20")
    maternal_aflp         = gene == "HADHA"

    # Key acylcarnitine markers
    key_acylcarnitine = {
        "ACADVL":   "C14:1",
        "ACADM":    "C8",
        "HADHA":    "3-OH-C16",
        "ACADS":    "C4",
        "CPT1A":    "C0 high, C16/C18 low",
        "CPT2":     "C16, C18, C18:1",
        "SLC25A20": "C16, C18, C18:1",
        "HADH":     "C4-OH (3-hydroxybutyrylcarnitine)",
    }

    # Generate patient ages (years)
    if is_neonatal_severe:
        ages = [rng.uniform(0, 0.1) for _ in range(N_PER_GENE)]
    elif is_myopathic_adolescent:
        ages = [rng.uniform(8, 40) for _ in range(N_PER_GENE)]
    elif is_benign_often:
        ages = [rng.uniform(0, 30) for _ in range(N_PER_GENE)]
    elif has_hyperinsulinism:
        ages = [rng.uniform(0, 2) for _ in range(N_PER_GENE)]
    else:
        ages = [rng.uniform(0, 10) for _ in range(N_PER_GENE)]

    # Glucose (mg/dL) at presentation
    glucose_vals = []
    for _ in range(N_PER_GENE):
        if has_hyperinsulinism:
            glucose_vals.append(rng.uniform(20, 55))
        elif gene in ("ACADM", "ACADVL", "CPT1A", "SLC25A20", "HADHA"):
            glucose_vals.append(rng.uniform(15, 50))
        elif gene == "CPT2":
            glucose_vals.append(rng.uniform(40, 90))  # myopathic, often normal glucose
        else:
            glucose_vals.append(rng.uniform(30, 80))

    # CK (U/L) for rhabdomyolysis genes
    ck_vals = []
    for _ in range(N_PER_GENE):
        if gene == "CPT2" and is_myopathic_adolescent:
            ck_vals.append(rng.uniform(5000, 100000))
        elif gene in ("ACADVL", "HADHA"):
            ck_vals.append(rng.uniform(1000, 50000) if rng.random() < 0.6 else rng.uniform(50, 500))
        else:
            ck_vals.append(rng.uniform(50, 500))

    outcomes = []
    for i, g_val in enumerate(glucose_vals):
        if g_val < 30:
            outcomes.append(rng.choice(["severe", "severe", "moderate"]))
        elif g_val < 50:
            outcomes.append(rng.choice(["moderate", "moderate", "mild", "severe"]))
        else:
            outcomes.append(rng.choice(["mild", "mild", "moderate"]))

    if is_benign_often:
        outcomes = [rng.choice(["asymptomatic", "asymptomatic", "mild"]) for _ in range(N_PER_GENE)]

    treatments = {
        "ACADVL":   ["avoid_fasting_feeds_q4-6h", "IV_glucose_emergency", "cornstarch", "MCT_limited_hepatic_form", "L_carnitine"],
        "ACADM":    ["avoid_fasting_feeds_q4-6h", "IV_dextrose_10pct_illness", "cornstarch", "emergency_protocol_card"],
        "HADHA":    ["low_LC_fat_diet_MCT", "avoid_fasting", "DHA_supplementation", "ophthalmology_monitoring", "cornstarch"],
        "ACADS":    ["riboflavin_cofactor", "avoid_fasting_precaution", "carnitine", "low_fat_diet_if_symptomatic"],
        "CPT1A":    ["avoid_fasting_cornstarch", "IV_glucose_emergency", "low_LC_fat_diet", "frequent_feeds"],
        "CPT2":     ["avoid_triggers_exercise_cold", "carnitine", "high_carbohydrate_diet", "IV_hydration_rhabdo", "bezafibrate_experimental"],
        "SLC25A20": ["IV_glucose_emergency", "antiarrhythmic", "carnitine_IV", "low_LC_fat_MCT", "frequent_feeds"],
        "HADH":     ["diazoxide", "avoid_protein_excess", "frequent_feeds", "leucine_restricted_diet"],
    }

    patients = []
    for i in range(N_PER_GENE):
        p = {
            "id":                f"{gene}-{seed}-{i:03d}",
            "seed":              seed,
            "gene":              gene,
            "gender":            rng.choice(["M", "F"]),
            "age_dx":            round(ages[i], 2),
            "glucose_min":       round(glucose_vals[i], 1),
            "ck_peak":           round(ck_vals[i], 0) if ck_vals else None,
            "outcome":           outcomes[i],
            "treatment":         rng.choice(treatments[gene]),
            "key_acylcarnitine": key_acylcarnitine[gene],
            "nbs_detected":      nbs_detectable and rng.random() < 0.88,
            "has_retinopathy":   has_retinopathy and ages[i] > 2.0 and rng.random() < 0.55,
            "has_rhabdomyolysis": has_rhabdomyolysis and rng.random() < (0.7 if gene == "CPT2" else 0.4),
            "has_cardiomyopathy": has_cardiomyopathy and rng.random() < 0.5,
            "has_hyperinsulinism": has_hyperinsulinism,
            "is_benign":         is_benign_often and outcomes[i] == "asymptomatic",
            "c0_high":           c0_high,
            "faod_group":        GENE_GROUPS[gene],
        }
        patients.append(p)

    return {
        "gene":               gene,
        "seed":               seed,
        "n":                  N_PER_GENE,
        "disease":            meta["disease"],
        "pathognomonic":      GENE_PATHOGNOMONIC[gene],
        "faod_group":         GENE_GROUPS[gene],
        "omim_gene":          meta["omim_gene"],
        "omim_disease":       meta["omim_disease"],
        "aa":                 meta["aa"],
        "kda":                meta["kda"],
        "locus":              meta["locus"],
        "inh":                meta["inh"],
        "nbs_detectable":     nbs_detectable,
        "has_retinopathy":    has_retinopathy,
        "has_rhabdomyolysis": has_rhabdomyolysis,
        "has_cardiomyopathy": has_cardiomyopathy,
        "has_cardiac_arrhythmia": has_cardiac_arrhythmia,
        "has_hyperinsulinism": has_hyperinsulinism,
        "is_often_benign":    is_benign_often,
        "arctic_founder":     arctic_founder,
        "is_neonatal_severe": is_neonatal_severe,
        "c0_high":            c0_high,
        "c0_low":             c0_low,
        "maternal_aflp":      maternal_aflp,
        "key_acylcarnitine":  key_acylcarnitine[gene],
        "mean_glucose":       round(sum(glucose_vals) / len(glucose_vals), 1),
        "pct_severe":         round(sum(1 for o in outcomes if o == "severe") / len(outcomes), 3),
        "patients":           patients,
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]

    gene_table = []
    for c in cohorts:
        gene_table.append({
            "gene":               c["gene"],
            "disease":            c["disease"],
            "locus":              c["locus"],
            "aa":                 c["aa"],
            "kda":                c["kda"],
            "inheritance":        c["inh"],
            "faod_group":         c["faod_group"],
            "key_acylcarnitine":  c["key_acylcarnitine"],
            "nbs_detected":       c["nbs_detectable"],
            "has_retinopathy":    c["has_retinopathy"],
            "has_rhabdomyolysis": c["has_rhabdomyolysis"],
            "has_cardiomyopathy": c["has_cardiomyopathy"],
            "has_hyperinsulinism":c["has_hyperinsulinism"],
            "is_often_benign":    c["is_often_benign"],
            "arctic_founder":     c["arctic_founder"],
            "mean_glucose":       c["mean_glucose"],
            "pct_severe":         c["pct_severe"],
            "pathognomonic":      c["pathognomonic"],
            "omim_disease":       c["omim_disease"],
        })

    return {
        "atlas_id":              ATLAS_ID,
        "total_patients":        N_PER_GENE * len(GENES),
        "genes":                 GENES,
        "seeds":                 f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "nbs_detectable_genes":  ["ACADVL", "ACADM", "HADHA", "ACADS", "CPT2", "SLC25A20"],
        "retinopathy_gene":      ["HADHA"],
        "rhabdomyolysis_genes":  ["ACADVL", "HADHA", "CPT2"],
        "cardiomyopathy_genes":  ["ACADVL", "SLC25A20"],
        "hyperinsulinism_gene":  ["HADH"],
        "arctic_founder_gene":   ["CPT1A"],
        "long_chain_group":      LONG_CHAIN_GROUP,
        "medium_chain_group":    MEDIUM_CHAIN_GROUP,
        "short_chain_group":     SHORT_CHAIN_GROUP,
        "carnitine_shuttle":     CARNITINE_SHUTTLE,
        "primary_cascade_split": (
            "STEP 1: Plasma acylcarnitine profile + free carnitine (C0). "
            "STEP 2: C14:1 elevated -> VLCAD (ACADVL). "
            "STEP 3: C8 elevated (most prominent) -> MCAD (ACADM) — most common FAOD, check p.K329E. "
            "STEP 4: 3-OH-C16 elevated -> LCHAD/TFP (HADHA) — check maternal AFLP history; ophthalmology. "
            "STEP 5: C4 elevated alone -> SCAD (ACADS) — usually benign; check EMA urine. "
            "STEP 6: C0 VERY HIGH + LC-AC LOW -> CPT1A (OPPOSITE pattern; check Arctic ancestry). "
            "STEP 7: C16/C18/C18:1 elevated + C0 LOW -> CPT2 or CACT (SLC25A20) — neonatal severe = CACT; "
            "adolescent exercise-rhabdomyolysis = CPT2 myopathic (p.S113L). "
            "STEP 8: Protein-sensitive hypoglycemia + C4-OH elevated + NORMAL ammonia -> HADH (hyperinsulinism; diazoxide responsive)."
        ),
        "gene_table":            gene_table,
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]

    return {
        "atlas_id": ATLAS_ID,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "faod_groups": {
            "long_chain_fao": {
                "genes": LONG_CHAIN_GROUP,
                "description": "ACADVL (VLCAD) + HADHA (LCHAD/TFP): long-chain FA oxidation defects; most clinically severe; NBS-detectable; LCHAD has unique retinopathy + maternal AFLP",
                "key_acylcarnitines": {"ACADVL": "C14:1", "HADHA": "3-OH-C16"},
            },
            "medium_chain_fao": {
                "genes": MEDIUM_CHAIN_GROUP,
                "description": "ACADM (MCAD): most common FAOD; C8 elevated; hypoketotic hypoglycemia; NBS has virtually eliminated mortality; p.K329E founder in N. Europeans",
                "key_acylcarnitines": {"ACADM": "C8"},
            },
            "short_chain_fao": {
                "genes": SHORT_CHAIN_GROUP,
                "description": "ACADS (SCAD, often benign, C4 elevated) + HADH (SCHAD, unique hyperinsulinism via GDH disinhibition, C4-OH elevated)",
                "key_acylcarnitines": {"ACADS": "C4", "HADH": "C4-OH"},
            },
            "carnitine_shuttle": {
                "genes": CARNITINE_SHUTTLE,
                "description": "CPT1A (C0 high, LC-AC low, Arctic founder) + CPT2 (C16/C18 high, myopathic adolescent, p.S113L) + SLC25A20/CACT (C16/C18 high, severe neonatal arrhythmias)",
                "c0_pattern": "CPT1A: C0 HIGH (opposite!); CPT2/CACT: C0 LOW",
            },
        },
        "acylcarnitine_algorithm": {
            "C14:1 dominant": {
                "gene": "ACADVL",
                "action": "VLCAD deficiency; check cardiac (echo), liver; avoid fasting; 3 phenotypes by residual activity",
            },
            "C8 dominant": {
                "gene": "ACADM",
                "action": "MCAD deficiency (most common FAOD); check p.K329E founder; strict no-fasting protocol; excellent prognosis NBS-detected",
            },
            "3-OH-C16 dominant": {
                "gene": "HADHA",
                "action": "LCHAD/TFP deficiency; ophthalmology mandatory; ask about maternal AFLP; DHA supplementation; low-LC-fat + MCT diet",
            },
            "C4 alone elevated": {
                "gene": "ACADS",
                "action": "SCAD deficiency; often benign; check EMA urine; riboflavin trial; avoid over-treatment of asymptomatic patients",
            },
            "C0 HIGH + LC-AC LOW": {
                "gene": "CPT1A",
                "action": "CPT1A deficiency (OPPOSITE acylcarnitine pattern); check Arctic ancestry; avoid fasting; MCT controversial",
            },
            "C16/C18/C18:1 elevated + C0 LOW, neonatal": {
                "gene": "SLC25A20",
                "action": "CACT deficiency; emergency — cardiac arrhythmias; IV glucose + antiarrhythmics; urgent molecular confirmation",
            },
            "C16/C18/C18:1 elevated + C0 LOW, adolescent rhabdomyolysis": {
                "gene": "CPT2",
                "action": "CPT2 myopathic (most common); check p.S113L; avoid exercise/cold/fasting triggers; bezafibrate experimental",
            },
            "C4-OH elevated + protein-sensitive hypoglycemia + normal ammonia": {
                "gene": "HADH",
                "action": "SCHAD/HADH deficiency (hyperinsulinism); diazoxide-responsive; avoid high-protein meals; normal ammonia excludes GLUD1-HI",
            },
        },
        "distinguishing_features": [
            {"gene": "ACADVL",   "feature": "C14:1 elevated; cardiac/hepatic/myopathic 3 phenotypes",     "action": "Cardiac echo; avoid fasting; MCT limited"},
            {"gene": "ACADM",    "feature": "C8 dominant; most common FAOD; hypoketotic hypoglycemia",    "action": "No fasting; 10% dextrose IV illness; excellent prognosis"},
            {"gene": "HADHA",    "feature": "3-OH-C16; retinopathy UNIQUE; maternal AFLP; neuropathy",    "action": "Low-LC fat + MCT; DHA; annual ophthalmology; test mother"},
            {"gene": "ACADS",    "feature": "C4 + EMA urine; often benign; polymorphisms common",         "action": "Riboflavin trial; avoid fasting; don't over-treat asymptomatic"},
            {"gene": "CPT1A",    "feature": "C0 HIGH (opposite!); LC-AC LOW; Arctic founder p.P479L",     "action": "Avoid fasting; low-LC fat; cornstarch; Arctic population counseling"},
            {"gene": "CPT2",     "feature": "C16/C18 elevated; myopathic adolescent; p.S113L founder",    "action": "Avoid triggers; high-carb diet; bezafibrate experimental; IV hydration rhabdo"},
            {"gene": "SLC25A20", "feature": "C16/C18 elevated; severe neonatal arrhythmias",              "action": "Emergency IV glucose + antiarrhythmic; carnitine IV; low-LC fat"},
            {"gene": "HADH",     "feature": "Protein-sensitive hyperinsulinism; NORMAL ammonia; C4-OH",   "action": "Diazoxide; restrict protein excess; DDx GLUD1-HI (has NH3)"},
        ],
        "cohort_summary": [
            {"gene": c["gene"], "mean_glucose": c["mean_glucose"], "pct_severe": c["pct_severe"],
             "nbs_detectable": c["nbs_detectable"], "faod_group": c["faod_group"]}
            for c in cohorts
        ],
    }


def generate_definitions():
    key_distinguishers = [
        {
            "name":        "MCAD deficiency: the most common FAOD — hypoketotic hypoglycemia and NBS success story",
            "description": (
                "ACADM/MCAD deficiency (1:10,000-15,000) was the leading cause of sudden infant death and Reye syndrome phenocopy before NBS. "
                "The pathophysiology is specific: fasting or illness triggers lipolysis -> medium-chain FA released -> MCAD cannot oxidize them -> "
                "acetyl-CoA production impaired -> impaired ketogenesis -> NO ketones (hypoketotic) -> "
                "HYPOKETOTIC HYPOGLYCEMIA = pathognomonic (contrast with ketotic hypoglycemia from GSD or ketone utilization defects). "
                "The absence of ketones during hypoglycemia is the bedside clue. "
                "Key biomarker: C8-acylcarnitine (octanoylcarnitine) MARKEDLY elevated on acylcarnitine profile; hexanoylglycine + phenylpropionylglycine in urine. "
                "Northern European founder mutation p.K329E (985A>G): >80% of alleles; single mutation testing feasible. "
                "NBS has transformed outcome: virtually all NBS-detected patients remain asymptomatic with feeding protocol (every 4-6h; increase frequency during illness; "
                "10% dextrose 2 mg/kg/min IV during intercurrent illness). "
                "Carnitine supplementation: controversial — may worsen by driving acylcarnitines into mitochondria; not routinely recommended. "
                "No specific dietary fat restriction (medium-chain fat is not the culprit — the issue is impaired oxidation, not intake)."
            ),
        },
        {
            "name":        "LCHAD/TFP deficiency: retinopathy and maternal AFLP — pathognomonic features unique among FAODs",
            "description": (
                "HADHA/LCHAD deficiency has two PATHOGNOMONIC features not seen in any other FAOD: "
                "(1) RETINOPATHY: pigmentary retinal degeneration progressing to visual impairment; "
                "occurs in long-term survivors; ophthalmology examination at diagnosis, then annually; "
                "DHA (docosahexaenoic acid 65 mg/day) supplementation may slow retinal degeneration. "
                "(2) MATERNAL ACUTE FATTY LIVER OF PREGNANCY (AFLP) / HELLP syndrome: "
                "mothers who carry LCHAD-affected fetuses develop AFLP or HELLP in ~20-25% of pregnancies; "
                "mechanism: LCHAD-deficient fetus cannot oxidize placental long-chain 3-hydroxy fatty acids -> "
                "these accumulate and cross to mother -> hepatotoxic to maternal liver (maternal CPT1A is normal but "
                "3-OH FA overload exceeds capacity); "
                "RETROGRADE DIAGNOSIS: any woman with AFLP or HELLP should prompt neonatal LCHAD/TFP screening "
                "of the newborn; if LCHAD confirmed in baby -> screen mother for heterozygous HADHA mutation. "
                "Key acylcarnitine: 3-OH-C16 (and 3-OH-C18, 3-OH-C18:1) PATHOGNOMONIC; "
                "treatment: low-long-chain fat diet with MCT supplementation (MCT bypasses LCHAD requirement)."
            ),
        },
        {
            "name":        "CPT1A deficiency: C0 high / LC-AC low — the opposite acylcarnitine pattern and Arctic founder",
            "description": (
                "CPT1A deficiency produces an acylcarnitine profile that is THE OPPOSITE of all other long-chain FAODs: "
                "FREE CARNITINE (C0) IS VERY HIGH and long-chain acylcarnitines (C16, C18) ARE LOW. "
                "This is because CPT1A catalyzes the FIRST STEP of the carnitine shuttle: "
                "palmitoyl-CoA + L-carnitine -> palmitoylcarnitine; CPT1A block -> carnitine NOT consumed -> C0 accumulates; "
                "palmitoylcarnitine NOT formed -> LC-AC stay low. "
                "CPT2 and CACT deficiencies show the OPPOSITE: C0 low, LC-AC elevated (block at later steps). "
                "ARCTIC FOUNDER: p.P479L (c.1436C>T) is found at extraordinary frequency in Inuit/First Nations/Alaska Native populations "
                "(carrier frequency up to 70%; homozygous rate ~25% in some communities). "
                "This may represent an ADAPTIVE variant in Arctic populations (promotes thermogenesis via inhibition of CPT1A -> "
                "carnitine stays available for other purposes; different metabolic phenotype in cold climate). "
                "NBS programs in Arctic regions MUST carefully interpret C0 elevation — most positive screens are homozygous p.P479L "
                "with uncertain clinical significance; cultural context required in genetic counseling."
            ),
        },
        {
            "name":        "HADH/SCHAD deficiency: the only FAOD causing hyperinsulinism — GDH disinhibition mechanism",
            "description": (
                "HADH (SCHAD) deficiency causes PROTEIN-SENSITIVE HYPERINSULINEMIC HYPOGLYCEMIA — not fasting-induced hypoglycemia like other FAODs. "
                "The mechanism is unique: HADH (short-chain 3-hydroxyacyl-CoA dehydrogenase) in pancreatic beta cells "
                "normally INHIBITS glutamate dehydrogenase (GDH) via protein-protein interaction (HADH-GDH complex). "
                "HADH LOF -> GDH uninhibited -> leucine (from dietary protein) over-stimulates GDH -> "
                "excess alpha-ketoglutarate -> excess ATP in beta cells -> KATP channels close -> insulin hypersecretion. "
                "KEY DIAGNOSTIC DISTINCTIONS: "
                "(1) PROTEIN-SENSITIVE (not fasting-triggered): hypoglycemia after protein meals, not overnight fasting; "
                "(2) NORMAL AMMONIA: this distinguishes HADH from GLUD1-HI (hyperinsulinism-hyperammonemia syndrome), "
                "which has IDENTICAL leucine sensitivity but ELEVATED ammonia (GDH also drives glutamate -> alpha-KG in liver); "
                "(3) C4-OH (3-hydroxybutyrylcarnitine) elevated on acylcarnitine profile. "
                "TREATMENT: diazoxide RESPONSIVE (opens KATP channels; different from ABCC8/KCNJ11 where diazoxide often fails); "
                "avoid excess protein; pancreatectomy rarely required (unlike focal KATP-HI)."
            ),
        },
        {
            "name":        "CACT deficiency vs CPT2 neonatal: cardiac arrhythmias vs cardiomyopathy — distinguishing the neonatal FAODs",
            "description": (
                "SLC25A20 (CACT) and CPT2 neonatal form both present with elevated C16/C18/C18:1 acylcarnitines and low C0. "
                "Clinical distinction: "
                "(1) CACT: CARDIAC ARRHYTHMIAS are the primary cause of death (ventricular fibrillation, AV block); "
                "hyperammonemia is common (secondary to liver failure; NOT a primary UCD); "
                "onset within 24-48h of birth; mortality high even with treatment; "
                "(2) CPT2 neonatal: similar presentation but cardiomyopathy more prominent than arrhythmias; "
                "also may have brain migration defects (periventricular heterotopia) — unique to CPT2 neonatal form; "
                "CACT (CPT2) neonatal: cardiac malformations in 20% of CPT2 neonatal (CACT has none). "
                "Molecular testing (SLC25A20 vs CPT2 sequencing) required for definitive distinction. "
                "CACT: inner mitochondrial membrane EXCHANGER (matrix free carnitine out / cytoplasmic acylcarnitine in); "
                "CPT2: inner mitochondrial membrane ENZYME (reconverts acylcarnitine to acyl-CoA). "
                "Both: emergency IV glucose + carnitine; anti-arrhythmic; low-LC-fat + MCT long-term."
            ),
        },
        {
            "name":        "VLCAD deficiency phenotype spectrum: three presentations by residual enzyme activity",
            "description": (
                "ACADVL/VLCAD deficiency has THREE clinical phenotypes determined by residual enzyme activity: "
                "(1) SEVERE CARDIAC (neonatal, <10% residual): hypertrophic cardiomyopathy + pericardial effusion + arrhythmias "
                "+ hypoglycemia within days of birth; neonatal crisis; high mortality without treatment; "
                "(2) HEPATIC (childhood, 10-20% residual): fasting-induced hypoglycemia + hepatomegaly; NO primary cardiac disease; "
                "presents with illness-triggered metabolic decompensation (like MCAD but with C14:1 instead of C8); "
                "(3) MYOPATHIC (adult/adolescent, >20% residual): exercise/fasting-triggered rhabdomyolysis + myalgias; "
                "CK markedly elevated during attacks; NO cardiomyopathy or hypoglycemia at rest. "
                "CRITICAL DISTINCTION from LCHAD: VLCAD LACKS retinopathy and peripheral neuropathy; "
                "VLCAD key marker is C14:1 (not 3-OH-C16 of LCHAD). "
                "MCT controversy: MCT oil provides C8-C12 FA that bypasses VLCAD; may help hepatic/myopathic forms; "
                "but accumulation of medium-chain intermediates may be toxic in cardiac form — use cautiously."
            ),
        },
        {
            "name":        "SCAD deficiency: biochemical phenotype vs clinical disease — avoid over-treatment",
            "description": (
                "ACADS/SCAD deficiency is frequently identified on NBS (C4/butyrylcarnitine elevation) but is often "
                "CLINICALLY BENIGN — the majority of NBS-detected patients remain entirely asymptomatic throughout life. "
                "COMMON VARIANTS: p.511C>T (c.319C>T, p.Arg171Trp) and p.625G>A (c.164C>T, p.Gly55Ser) are "
                "DISEASE-MODIFYING POLYMORPHISMS — they reduce enzyme activity but are NOT sufficient to cause disease alone; "
                "these variants interact with other metabolic stressors; found in >7% of general population. "
                "CLINICAL SYMPTOMS (when present): hypotonia, developmental delay, seizures — BUT causal relationship debated; "
                "symptoms may be coincidental in many reported cases. "
                "DDx for C4 elevation: Multiple Acyl-CoA Dehydrogenase Deficiency (MADD/GA2) — has MULTIPLE acylcarnitines elevated "
                "(C4, C5, C8, C14, C16, etc.); ethylmalonic encephalopathy (ETHE1) — more severe with EMA + lactate + MRI changes. "
                "TREATMENT: riboflavin (FAD cofactor for SCAD) — some patients respond; carnitine if secondary depletion; "
                "avoid fasting as general precaution; do NOT impose restrictive fat diet in asymptomatic patients."
            ),
        },
        {
            "name":        "Fasting protocols and emergency cards: the universal FAOD management principle",
            "description": (
                "ALL fatty acid oxidation disorders (except HADH hyperinsulinism and SCAD when asymptomatic) share ONE critical management principle: "
                "AVOID FASTING. Fatty acid oxidation is the backup fuel source during fasting/illness; "
                "when FAO is impaired, the patient CANNOT generate energy from fat stores -> "
                "progressive energy failure -> hypoglycemia, organ dysfunction, death. "
                "FEEDING PROTOCOLS by age: "
                "Neonates (<3 months): max fast 4h; "
                "Infants (3-12 months): max fast 6h; "
                "Children (1-5 years): max fast 8h; "
                "Children (>5 years): max fast 10h; "
                "Adults: max fast 12h (varies by gene and severity). "
                "EMERGENCY PROTOCOL (intercurrent illness/pre-surgical): "
                "IV dextrose 10% at 2-4 mg/kg/min glucose infusion rate; "
                "do NOT allow fasting for ANY operation or illness; "
                "emergency letter/card at all times for anesthesia team. "
                "Exception: HADH (hyperinsulinism) — restrict PROTEIN not fast avoidance; "
                "CPT2 myopathic — avoid exercise triggers, not primarily fasting."
            ),
        },
    ]

    treatment_matrix = [
        {
            "intervention":    "Avoid fasting / frequent feeds",
            "indications":     "ALL FAODs except HADH hyperinsulinism when protein is the trigger; most critical intervention in MCAD, VLCAD, LCHAD, CPT1A, CACT",
            "mechanism":       "FAO is backup fuel during fasting; FAO-deficient patients cannot mobilize energy from fat -> hypoglycemia, organ failure during any fast",
            "dose":            "Age-appropriate fasting limits: neonates ≤4h, infants ≤6h, toddlers ≤8h, children ≤10-12h; cornstarch (1-2 g/kg) for overnight extension",
            "contraindicated": "Prolonged fasting in any FAOD is absolutely contraindicated; NPO for surgery without IV glucose is dangerous",
        },
        {
            "intervention":    "IV dextrose 10% emergency infusion",
            "indications":     "Acute metabolic decompensation in MCAD, VLCAD, CPT1A, HADHA, CACT; intercurrent illness with vomiting/inability to feed",
            "mechanism":       "Provides glucose as alternative fuel; suppresses lipolysis (insulin response) -> reduces free FA flux -> reduces acylcarnitine accumulation",
            "dose":            "2-4 mg/kg/min glucose infusion rate (pediatric); maintenance 10% dextrose in 0.45% NaCl; target blood glucose 90-120 mg/dL",
            "contraindicated": "NOT for HADH hyperinsulinism (glucose may worsen hyperinsulinism acutely); use dextrose carefully in HADH",
        },
        {
            "intervention":    "MCT (medium-chain triglyceride) oil/formula",
            "indications":     "LCHAD/TFP (HADHA): MCT bypasses long-chain FAO; CACT (SLC25A20): MCT partially bypasses; VLCAD hepatic form: MCT may help",
            "mechanism":       "MCT (C8-C12) enters mitochondria via CPT1-independent route -> MCAD (not LCHAD/VLCAD) oxidizes them -> generates ketones and energy without requiring defective enzyme",
            "dose":            "LCHAD: MCT 30-50% of total fat calories; supplement with essential long-chain FA (linoleic, alpha-linolenic, DHA) to prevent deficiency",
            "contraindicated": "CPT2 myopathic: MCT requires CPT2 for entry at higher chain lengths (partial); not primary treatment; ACADS: not indicated",
        },
        {
            "intervention":    "L-carnitine supplementation",
            "indications":     "CACT (essential — replenish depleted C0); CPT2 (may help clear acylcarnitines); useful when secondary carnitine depletion confirmed by low plasma C0",
            "mechanism":       "Replenishes free carnitine pool; facilitates export of acylcarnitines from mitochondria (detoxification of accumulated acyl-CoA); required cofactor for FA entry",
            "dose":            "50-100 mg/kg/day oral; 50 mg/kg IV bolus in acute crisis (CACT); monitor plasma C0 to guide dosing",
            "contraindicated": "MCAD: controversial — may drive more acylcarnitines INTO mitochondria; not routinely recommended; avoid in MCAD without secondary C0 depletion",
        },
        {
            "intervention":    "DHA (docosahexaenoic acid) supplementation",
            "indications":     "LCHAD/TFP (HADHA): specifically indicated to prevent/slow retinal degeneration",
            "mechanism":       "Long-chain PUFA synthesis is impaired in LCHAD due to defective beta-oxidation of very-long-chain FA; DHA is critical for photoreceptor membrane integrity; supplementation replaces the deficit",
            "dose":            "DHA 65 mg/day (infants); adjust by age; long-term; check DHA levels (RBC membrane)",
            "contraindicated": "No contraindications; should be given to ALL LCHAD patients regardless of retinal status",
        },
        {
            "intervention":    "Diazoxide",
            "indications":     "HADH (SCHAD) deficiency hyperinsulinism; opens KATP channels in beta cells -> reduces insulin secretion",
            "mechanism":       "KATP channel opener (binds SUR1/Kir6.2 complex); hyperpolarizes beta cell -> reduces Ca2+ influx -> reduces insulin secretion",
            "dose":            "5-15 mg/kg/day in 3 divided doses; titrate to normoglycemia",
            "contraindicated": "Not indicated for other FAODs (no hyperinsulinism mechanism); avoid if KATP channel mutation present (diazoxide-unresponsive ABCC8/KCNJ11)",
        },
    ]

    return {
        "atlas_id":          ATLAS_ID,
        "key_distinguishers":key_distinguishers,
        "treatment_matrix":  treatment_matrix,
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "primary_cascade_split": (
            "STEP 1: C14:1 elevated -> VLCAD. "
            "STEP 2: C8 dominant -> MCAD (most common; check p.K329E). "
            "STEP 3: 3-OH-C16 elevated -> LCHAD/TFP (check maternal AFLP; ophthalmology). "
            "STEP 4: C4 alone + EMA urine -> SCAD (often benign; riboflavin). "
            "STEP 5: C0 VERY HIGH + LC-AC LOW -> CPT1A (opposite pattern; Arctic ancestry). "
            "STEP 6: C16/C18 elevated + C0 LOW + neonatal arrhythmias -> CACT (SLC25A20). "
            "STEP 7: C16/C18 elevated + C0 LOW + adolescent rhabdomyolysis -> CPT2 (p.S113L). "
            "STEP 8: Protein-sensitive hypoglycemia + C4-OH + normal ammonia -> HADH (hyperinsulinism; diazoxide)."
        ),
        "faod_groups": {
            "long_chain":       {"genes": LONG_CHAIN_GROUP,  "primary_markers": "C14:1 (ACADVL); 3-OH-C16 (HADHA)"},
            "medium_chain":     {"genes": MEDIUM_CHAIN_GROUP, "primary_markers": "C8 (ACADM)"},
            "short_chain":      {"genes": SHORT_CHAIN_GROUP,  "primary_markers": "C4 (ACADS); C4-OH (HADH)"},
            "carnitine_shuttle":{"genes": CARNITINE_SHUTTLE,  "primary_markers": "C0 HIGH (CPT1A); C16/C18+C0 LOW (CPT2/CACT)"},
        },
    }
