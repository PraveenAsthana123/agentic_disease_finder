#!/usr/bin/env python3
"""Hereditary-Lipodystrophy-Atlas — Complete 8-Gene Lipodystrophy Atlas
AGPAT2  (1-acylglycerol-3-phosphate O-acyltransferase 2; 278 aa; 9q34.3; AR;
         Congenital Generalized Lipodystrophy type 1 (CGL1/Berardinelli-Seip);
         AGPAT2 deficiency -> severely reduced phosphatidic acid and DAG -> failure to
         form fat droplets -> near-complete absence of metabolically active adipose tissue;
         severely elevated triglycerides (500-3000 mg/dL), insulin resistance (HOMA-IR 10-80),
         diabetes, hepatic steatosis, acanthosis nigricans;
         recombinant leptin (metreleptin) = SPECIFIC treatment; seed SEED_BASE+0) *
BSCL2   (Seipin / BSCL2; 462 aa; 11q12.3; AR;
         CGL2 - MOST COMMON and MOST SEVERE CGL; Seipin is ER membrane protein essential
         for lipid droplet biogenesis; biallelic LOF -> CGL2 with intellectual disability
         30-60% (unlike CGL1); severe hypertriglyceridemia; acanthosis nigricans;
         metreleptin effective; seed SEED_BASE+1) *
CAV1    (Caveolin-1; 178 aa; 7q31.2; AR;
         CGL3; Caveolin-1 deficiency; phenotype similar CGL1/CGL2 but milder;
         caveolae absent on electron microscopy (PATHOGNOMONIC);
         associated with pulmonary arterial hypertension; seed SEED_BASE+2) *
CAVIN1  (Cavin-1 / PTRF; 392 aa; 17q21.2; AR;
         CGL4; ONLY CGL with skeletal muscle involvement (myopathy + elevated CK)
         + cardiac arrhythmia; caveolae absent; muscular dystrophy phenotype
         DISTINGUISHES from CGL1/2/3; seed SEED_BASE+3) *
LMNA    (Lamin A/C; 664 aa; 1q22; AD;
         FPLD2 - Dunnigan syndrome - MOST COMMON familial partial lipodystrophy;
         p.Arg482 hotspot mutations (Arg482Trp/Gln/Leu) -> loss of fat from
         extremities/gluteal + accumulation face/neck/abdomen starting puberty;
         females more severely affected; cardiomyopathy + laminopathy overlap;
         premature death cardiac/arrhythmia; seed SEED_BASE+4) *
PPARG   (PPAR-gamma; 477 aa; 3p25.2; AD;
         FPLD3; PPAR-gamma haploinsufficiency -> partial loss of peripheral fat;
         hypertension + dyslipidemia + DM;
         thiazolidinediones (TZDs) = SPECIFIC treatment (PPARG agonist replaces
         the lost function); seed SEED_BASE+5) *
AKT2    (AKT serine/threonine kinase 2; 481 aa; 19q13.2; AD;
         FPLD6; AKT2 LOF mutations; partial lipodystrophy + SEVERE insulin resistance;
         unlike other FPLDs, AD with incomplete penetrance;
         somatic activating mutations -> hypoglycemia (opposite phenotype); seed SEED_BASE+6) *
PLIN1   (Perilipin-1; 522 aa; 15q26.1; AD;
         FPLD4; Perilipin-1 deficiency; partial lipodystrophy + severe
         hypertriglyceridemia + pancreatitis risk;
         PLIN1 is the dominant lipid droplet coat protein;
         mutations reduce lipolysis regulation;
         frameshift mutations via heterozygous loss; seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 2998-3005)
"""
import random

SEED_BASE = 2998

ATLAS_GENES = [
    {
        "gene": "AGPAT2",
        "protein": (
            "AGPAT2 -- 9q34.3 AR -- 278aa -- 1-Acylglycerol-3-Phosphate-O-Acyltransferase-2-"
            "31kDa-ER-Membrane-Lysophosphatidic-Acid-Acyltransferase-"
            "CGL1-Berardinelli-Seip-Near-Complete-Fat-Absence-Metreleptin-SPECIFIC-OMIM-603100"
        ),
        "locus": "9q34.3",
        "protein_size": (
            "278 aa / 31 kDa (AGPAT2 -- 1-acylglycerol-3-phosphate O-acyltransferase 2; "
            "FUNCTION: enzyme in the de novo glycerophospholipid synthesis pathway (Kennedy pathway); "
            "  Catalyses: lysophosphatidic acid (LPA) + acyl-CoA -> phosphatidic acid (PA); "
            "  PA is the central metabolic precursor for: "
            "    (1) Diacylglycerol (DAG) -> triacylglycerol (TAG) storage -> lipid droplet formation; "
            "    (2) Phosphatidylcholine, phosphatidylethanolamine (membrane phospholipids); "
            "  ER membrane localisation; highest expression in adipose tissue; "
            "  Essential for de novo adipogenesis and fat droplet biogenesis; "
            "AGPAT2 LOF -> CGL1 MECHANISM: "
            "  Without functional AGPAT2: PA severely reduced -> DAG pool collapses; "
            "  Lipid droplets cannot form in pre-adipocytes -> adipogenesis arrested; "
            "  Near-complete absence of metabolically active adipose tissue from birth; "
            "  Mechanical/structural adipose (palms, scalp, periarticular, orbits) partially spared; "
            "  Consequence of absent fat: "
            "    No leptin production -> leptin level extremely low (<1 ng/mL); "
            "    Ectopic fat deposition: liver (severe hepatic steatosis -> cirrhosis), muscle, heart; "
            "    Severely elevated triglycerides (500-3000 mg/dL -> pancreatitis risk); "
            "    Extreme insulin resistance (HOMA-IR 10-80+); DM in 70-90%; "
            "    Acanthosis nigricans (severe insulin resistance marker); "
            "  DISTINCT from BSCL2/CGL2: mechanical fat spared in CGL1; no intellectual disability; "
            "encoded 9q34.3; OMIM gene 603100, disease CGL1 #608594"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE -- AGPAT2 / CONGENITAL GENERALIZED LIPODYSTROPHY TYPE 1 (CGL1): "
            "  ONSET: congenital; near-complete absence of subcutaneous fat from birth; "
            "  CLINICAL FEATURES: "
            "    Muscular appearance (paradox: no fat -> visible musculature); "
            "    Prominent superficial veins (absence of subcutaneous fat); "
            "    Acromegaloid features (prominent jaw, large hands/feet -- GH excess from low leptin feedback); "
            "    Hepatomegaly: severe hepatic steatosis -> cirrhosis risk (3rd-5th decade); "
            "    Metabolic: severe hypertriglyceridemia (eruptive xanthomata + pancreatitis); "
            "    Diabetes mellitus: 70-90%; insulin-resistant (type A extreme insulin resistance); "
            "    Acanthosis nigricans: axillae, neck -- severe insulin resistance marker; "
            "    Polycystic ovaries (PCOS) in females (hyperinsulinaemia); "
            "  LEPTIN: extremely low (<1 ng/mL) -- no adipose -> no leptin source; "
            "  MECHANICAL FAT SPARED (CGL1): palms, soles, scalp, periarticular, orbits partially preserved; "
            "    CGL2 (BSCL2): even mechanical fat absent; more severe; "
            "  CARDIAC: hypertrophic cardiomyopathy (ectopic myocardial fat + metabolic cardiomyopathy); "
            "  INTELLECTUAL DISABILITY: NOT in CGL1 (distinguishes from CGL2/BSCL2); "
            "  DIAGNOSIS: "
            "    Clinical: congenital absence of subcutaneous fat + muscular appearance; "
            "    Low leptin + severe hypertriglyceridemia + extreme HOMA-IR; "
            "    Genetic: AGPAT2 sequencing; biallelic LOF mutations; "
            "  TREATMENT: "
            "    Metreleptin (recombinant leptin) = SPECIFIC TREATMENT: "
            "      Replaces absent leptin -> restores leptin signalling -> reduced hyperphagia; "
            "      Dramatically lowers triglycerides + HbA1c; improves hepatic steatosis; "
            "      FDA/EMA approved for CGL; titrated by body weight; "
            "    Low fat diet: reduce triglycerides; "
            "    Insulin (diabetes) + fibrates (hypertriglyceridemia); "
            "    Liver monitoring: annual USS + LFTs; cirrhosis screening; "
            "  PROGNOSIS: without metreleptin: early cirrhosis + pancreatitis + cardiomyopathy; "
            "    With metreleptin: markedly improved metabolic control"
        ),
        "disease_category": (
            "CGL1-AGPAT2-METRELEPTIN-SPECIFIC-NEAR-COMPLETE-FAT-ABSENT: "
            "  KEY RULE: congenital near-complete fat absence + low leptin + severe HTG -> CGL1/CGL2; "
            "  METRELEPTIN SPECIFIC: replaces absent leptin -> corrects metabolic syndrome; FDA approved; "
            "  CGL1 vs CGL2: CGL1 mechanical fat SPARED; CGL2 even mechanical fat absent; CGL1 NO intellectual disability; "
            "  TRIGLYCERIDES 500-3000: pancreatitis risk; fibrates + low-fat diet + metreleptin; "
            "  HEPATIC STEATOSIS: universal; progression to cirrhosis without treatment; "
            "  INSULIN RESISTANCE EXTREME: HOMA-IR 10-80; acanthosis nigricans; DM 70-90%; "
            "  GENETIC TESTING: AGPAT2 sequencing + MLPA; biallelic mutations confirm CGL1"
        ),
    },
    {
        "gene": "BSCL2",
        "protein": (
            "BSCL2 -- 11q12.3 AR -- 462aa -- Seipin-"
            "ER-Integral-Membrane-Protein-52kDa-Lipid-Droplet-Biogenesis-ER-Tubular-ER-Junctions-"
            "CGL2-MOST-COMMON-MOST-SEVERE-Intellectual-Disability-30-60pct-Metreleptin-EFFECTIVE-OMIM-606158"
        ),
        "locus": "11q12.3",
        "protein_size": (
            "462 aa / 52 kDa (BSCL2 / Seipin -- integral ER membrane protein; "
            "FUNCTION: oligomeric ring structure at ER-lipid droplet contact sites; "
            "  Seipin forms 11-mer ring at ER tubular junctions; "
            "  Essential for lipid droplet (LD) biogenesis: "
            "    Nucleates LD formation: concentrates TAG/DAG at ER sites where LDs bud; "
            "    Without Seipin: LD biogenesis severely impaired or aberrant (tiny LDs, mislocalised); "
            "  Regulates ER-LD contact: maintains protein/lipid flux between ER and growing LD; "
            "  Expressed ubiquitously but highest in adipose + brain (explains neurological features); "
            "  Absence -> adipose tissue cannot form and/or maintain lipid droplets -> lipodystrophy; "
            "BSCL2 LOF -> CGL2 (MOST SEVERE AND MOST COMMON CGL): "
            "  Virtually complete absence of ALL adipose (metabolic AND mechanical fat both absent); "
            "    Mechanically active fat (palms, orbits) ALSO absent -- distinguishes from CGL1; "
            "  Very low to absent leptin (<0.5 ng/mL); "
            "  Severe ectopic fat: liver, muscle, heart -> cirrhosis, myopathy, cardiomyopathy; "
            "  More severe metabolic phenotype than CGL1; "
            "  INTELLECTUAL DISABILITY: 30-60% of CGL2 patients (Seipin expressed in brain); "
            "    Not present in CGL1 (AGPAT2) -- KEY DISTINGUISHER; "
            "  Hypertrophic cardiomyopathy: more frequent and earlier than CGL1; "
            "  Founder mutations: p.Asn88Ser (Lebanese/Middle Eastern); "
            "    p.Glu189Lys and others in other populations; "
            "encoded 11q12.3; OMIM gene 606158, disease CGL2 #269700"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE -- BSCL2 / CONGENITAL GENERALIZED LIPODYSTROPHY TYPE 2 (CGL2): "
            "  ONSET: congenital; virtually complete absence of all adipose tissue; "
            "  KEY DIFFERENCES FROM CGL1 (AGPAT2): "
            "    1. MECHANICAL FAT ABSENT: even orbital/palmar/plantar fat gone -- more severe; "
            "    2. INTELLECTUAL DISABILITY: 30-60% (vs 0% in CGL1) -- Seipin expressed in brain; "
            "    3. SLIGHTLY MORE SEVERE METABOLIC: earlier/worse hepatic disease; "
            "    4. HYPERTROPHIC CARDIOMYOPATHY: more common; earlier onset; "
            "  CLINICAL: "
            "    Near-complete lipoatrophy from birth; muscular appearance; prominent veins; "
            "    Acromegaloid features; "
            "    Hepatomegaly + severe steatohepatitis -> cirrhosis (often 2nd-3rd decade); "
            "    Extreme hypertriglyceridemia + insulin resistance + DM in >80%; "
            "    Pancreatitis events (eruptive xanthomata); "
            "    PCOS in females; "
            "    Intellectual disability: mild-moderate (IQ 50-70 range in affected cases); "
            "  LEPTIN: near-absent (<0.5 ng/mL); "
            "  DIAGNOSIS: "
            "    Same as CGL1 clinically + genetic confirmation BSCL2 biallelic LOF; "
            "    Liver biopsy if cirrhosis staging needed; "
            "  TREATMENT: "
            "    Metreleptin: SPECIFIC and EFFECTIVE (same as CGL1); FDA/EMA approved; "
            "    Low-fat diet; fibrates; insulin; "
            "    Neurological support if intellectual disability; "
            "    Cardiac surveillance: echo annually (hypertrophic cardiomyopathy); "
            "    Liver surveillance: annual USS + LFTs; hepatology referral; "
            "  PROGNOSIS: more complications than CGL1; intellectual disability subset; "
            "    Metreleptin substantially improves metabolic outcomes"
        ),
        "disease_category": (
            "CGL2-BSCL2-MOST-COMMON-MOST-SEVERE-CGL-INTELLECTUAL-DISABILITY: "
            "  KEY RULE: CGL2 = MOST COMMON CGL globally; MOST SEVERE; ALL fat absent (including mechanical); "
            "  INTELLECTUAL DISABILITY 30-60%: distinguishes CGL2 from CGL1/3/4; Seipin in brain; "
            "  METRELEPTIN: SPECIFIC and EFFECTIVE; same as CGL1; replace absent leptin; "
            "  CARDIAC: hypertrophic cardiomyopathy more frequent + earlier than CGL1; echo annually; "
            "  LIVER: severe steatohepatitis -> cirrhosis often 2nd-3rd decade without treatment; "
            "  GENETIC TESTING: BSCL2 sequencing; biallelic LOF confirms CGL2"
        ),
    },
    {
        "gene": "CAV1",
        "protein": (
            "CAV1 -- 7q31.2 AR -- 178aa -- Caveolin-1-"
            "21kDa-Integral-Membrane-Protein-Caveolae-Scaffold-Cholesterol-Sphingolipid-Raft-"
            "CGL3-Milder-CGL-Caveolae-Absent-EM-PATHOGNOMONIC-Pulmonary-Arterial-Hypertension-OMIM-601047"
        ),
        "locus": "7q31.2",
        "protein_size": (
            "178 aa / 21 kDa (CAV1 -- Caveolin-1; integral plasma membrane protein; "
            "FUNCTION: principal structural protein of caveolae; "
            "  Caveolae: flask-shaped plasma membrane invaginations (50-100 nm); "
            "  Caveolin-1 inserts into inner membrane leaflet via hairpin hydrophobic domain; "
            "  Oligomerises into 7-14-mer complexes -> scaffolds caveolae; "
            "  CAVEOLAE FUNCTIONS: "
            "    Lipid regulation: cholesterol + sphingolipid trafficking; "
            "    Lipid droplet formation and dynamics; "
            "    Signal transduction: concentrates and regulates receptor tyrosine kinases, eNOS, G-proteins; "
            "    Endocytosis: caveolae-dependent endocytosis (distinct from clathrin); "
            "    Mechanosensing: plasma membrane tension sensing; "
            "  ADIPOCYTE: caveolae particularly abundant in adipocytes; "
            "    CAV1 essential for adipocyte caveolae -> lipid droplet regulation; "
            "  PULMONARY ENDOTHELIUM: CAV1 regulates eNOS -> NO production -> pulmonary vascular tone; "
            "    CAV1 deficiency -> eNOS dysregulation -> pulmonary arterial hypertension (PAH); "
            "CAV1 LOF -> CGL3: "
            "  Milder than CGL1/CGL2: partial lipoatrophy (metabolically active fat mainly affected); "
            "  Caveolae absent on electron microscopy -- PATHOGNOMONIC for CAV1 and CAVIN1 mutations; "
            "  PAH: 10-20% of CGL3 patients -- unique complication not in CGL1/CGL2; "
            "encoded 7q31.2; OMIM gene 601047, disease CGL3 #612526"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE -- CAV1 / CONGENITAL GENERALIZED LIPODYSTROPHY TYPE 3 (CGL3): "
            "  ONSET: congenital but milder than CGL1/CGL2; "
            "  PHENOTYPE: "
            "    Generalised lipoatrophy (mainly metabolically active fat; some mechanical fat spared); "
            "    Similar to CGL1 but metabolic features generally milder; "
            "    Hypertriglyceridemia + insulin resistance + DM (as in CGL1/CGL2 but less severe); "
            "    Hepatic steatosis present; "
            "    No intellectual disability; "
            "  PATHOGNOMONIC FINDING: "
            "    CAVEOLAE ABSENT on electron microscopy (skin biopsy, muscle biopsy); "
            "    No flask-shaped membrane invaginations visible -> confirms CAV1 (or CAVIN1) mutation; "
            "    Normal caveolae requires BOTH CAV1 and CAVIN1; either absent -> caveolae absent; "
            "  PULMONARY ARTERIAL HYPERTENSION (PAH): "
            "    Unique to CGL3 among CGL types; "
            "    CAV1 regulates eNOS in pulmonary endothelium -> CAV1 LOF -> eNOS dysregulation -> PAH; "
            "    Screen with ECHO for TR jet velocity; RHC if elevated; "
            "  LEPTIN: low (but generally higher than CGL1/CGL2 as some fat preserved); "
            "  DIAGNOSIS: "
            "    Clinical lipoatrophy + absent caveolae on EM + genetic confirmation; "
            "    CAV1 sequencing; biallelic LOF; "
            "    Pulmonary evaluation: echo at diagnosis, annually; "
            "  TREATMENT: "
            "    Metreleptin: effective (less studied than CGL1/2 but used); "
            "    Low-fat diet; fibrates for HTG; "
            "    PAH management if confirmed: sildenafil, bosentan (as for PAH); "
            "  PROGNOSIS: better than CGL1/2; PAH major complication to monitor"
        ),
        "disease_category": (
            "CGL3-CAV1-CAVEOLAE-ABSENT-EM-PATHOGNOMONIC-PAH: "
            "  KEY RULE: CGL3 milder than CGL1/CGL2; CAVEOLAE ABSENT on EM = pathognomonic (also in CAVIN1/CGL4); "
            "  PAH: unique to CGL3 among CGL subtypes; screen with ECHO at diagnosis; treat if confirmed; "
            "  NO INTELLECTUAL DISABILITY: distinguishes from CGL2 (BSCL2); "
            "  ELECTRON MICROSCOPY: absence of flask-shaped caveolae in skin/muscle biopsy confirms CAV1; "
            "  METRELEPTIN: effective; same approach as CGL1/2; "
            "  GENETIC TESTING: CAV1 sequencing; biallelic LOF confirms CGL3"
        ),
    },
    {
        "gene": "CAVIN1",
        "protein": (
            "CAVIN1 -- 17q21.2 AR -- 392aa -- Cavin-1-PTRF-"
            "Polymerase-I-and-Transcript-Release-Factor-44kDa-Caveolae-Coat-Protein-"
            "CGL4-ONLY-CGL-Myopathy-CK-Elevated-Cardiac-Arrhythmia-Muscular-Dystrophy-Distinguishes-OMIM-603198"
        ),
        "locus": "17q21.2",
        "protein_size": (
            "392 aa / 44 kDa (CAVIN1 / PTRF -- polymerase-I and transcript release factor; "
            "FUNCTION: coat protein required for caveolae biogenesis and stability; "
            "  Cavin1 is the founding member of the cavin family (cavin1-4); "
            "  Co-localises with Caveolin-1 at caveolae; "
            "  Mechanism: cavin1 oligomers coat the cytoplasmic face of caveolae; "
            "    Without cavin1: caveolin-1 cannot form stable caveolae -> caveolae absent; "
            "    Cavin1 KO mice: caveolae absent, same as CAV1 KO -- confirming essential role; "
            "  MUSCLE: caveolae particularly abundant in skeletal and cardiac muscle; "
            "    Caveolae function in muscle: membrane repair, T-tubule organisation, mechanosensing; "
            "    CAVIN1 LOF -> caveolae absent in muscle -> MYOPATHY; "
            "  ADIPOSE: same as CAV1 -- caveolae absent -> lipodystrophy; "
            "CGL4 DISTINGUISHING FEATURES (unique among all CGL types): "
            "  1. SKELETAL MUSCLE INVOLVEMENT: myopathy (muscle weakness + wasting); "
            "     CK markedly elevated (500-5000 IU/L); "
            "     Muscular dystrophy phenotype (limb-girdle distribution); "
            "     NOT present in CGL1/2/3; "
            "  2. CARDIAC ARRHYTHMIA: conduction abnormalities; sudden cardiac death risk; "
            "     Likely: caveolae absent in cardiomyocytes -> ion channel dysregulation -> arrhythmia; "
            "  3. CAVEOLAE ABSENT on EM (same as CAV1/CGL3); "
            "encoded 17q21.2; OMIM gene 603198, disease CGL4 #613327"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE -- CAVIN1 / CONGENITAL GENERALIZED LIPODYSTROPHY TYPE 4 (CGL4): "
            "  ONSET: congenital; "
            "  LIPODYSTROPHY FEATURES: "
            "    Generalised lipoatrophy (metabolically active fat absent); similar to CGL3 in severity; "
            "    Hypertriglyceridemia + insulin resistance + DM (moderate-severe); "
            "    Hepatic steatosis; "
            "    Low leptin; "
            "  UNIQUE CGL4 FEATURES (not in CGL1/2/3): "
            "    1. SKELETAL MUSCLE MYOPATHY: "
            "       Muscle weakness (limb-girdle distribution) + wasting; "
            "       CK markedly elevated (500-5000 IU/L) -- KEY BIOMARKER; "
            "       Muscular dystrophy phenotype; muscle biopsy: dystrophic changes; "
            "       Diaphragm involvement: respiratory insufficiency possible; "
            "    2. CARDIAC ARRHYTHMIA: "
            "       Conduction abnormalities on ECG (PR prolongation, BBB, QT prolongation); "
            "       Sudden cardiac death reported -- require cardiac monitoring; "
            "       Annual Holter/ECG + echocardiogram; "
            "    3. CAVEOLAE ABSENT on EM: shared with CGL3 (CAV1); "
            "  DISTINGUISHING FROM CGL3: "
            "    CGL4 (CAVIN1): myopathy + elevated CK + cardiac arrhythmia; "
            "    CGL3 (CAV1): PAH but NO myopathy, NO elevated CK; "
            "    Both: caveolae absent on EM; "
            "  DIAGNOSIS: "
            "    Clinical: lipoatrophy + myopathy + elevated CK; "
            "    Caveolae absent on EM (muscle or skin biopsy); "
            "    CAVIN1 sequencing; biallelic LOF; "
            "  TREATMENT: "
            "    Metreleptin: effective for metabolic features; "
            "    Low-fat diet; fibrates; insulin; "
            "    Cardiac surveillance: ECG + Holter + echo annually; ICD if significant arrhythmia; "
            "    Respiratory monitoring: spirometry if diaphragm involvement; "
            "    Physiotherapy for myopathy; "
            "  PROGNOSIS: cardiac arrhythmia = main mortality risk"
        ),
        "disease_category": (
            "CGL4-CAVIN1-ONLY-CGL-WITH-MYOPATHY-CK-ELEVATED-CARDIAC-ARRHYTHMIA: "
            "  PATHOGNOMONIC: CGL + myopathy + elevated CK = CGL4 (CAVIN1) -- not in CGL1/2/3; "
            "  CARDIAC ARRHYTHMIA: sudden death risk; ECG + Holter + echo annual surveillance mandatory; "
            "  CK 500-5000 IU/L: key biomarker distinguishing CGL4 from other CGLs; "
            "  CAVEOLAE ABSENT on EM: shared with CGL3 (CAV1); both cavin1 + cav1 needed; "
            "  DISTINGUISHES from CGL3: CGL4 = myopathy + arrhythmia; CGL3 = PAH (no myopathy); "
            "  GENETIC TESTING: CAVIN1 sequencing; biallelic LOF confirms CGL4"
        ),
    },
    {
        "gene": "LMNA",
        "protein": (
            "LMNA -- 1q22 AD -- 664aa -- Lamin-A-C-"
            "74kDa-Nuclear-Lamina-Type-V-Intermediate-Filament-"
            "FPLD2-Dunnigan-Syndrome-MOST-COMMON-FPLD-Arg482-Hotspot-Females-More-Severe-Cardiomyopathy-OMIM-150330"
        ),
        "locus": "1q22",
        "protein_size": (
            "664 aa / 74 kDa (LMNA -- Lamin A/C; type V intermediate filament protein; "
            "FUNCTION: structural component of the nuclear lamina (inner nuclear membrane scaffold); "
            "  Lamin A and Lamin C: two major isoforms from LMNA (alternative splicing); "
            "    Lamin A: 664 aa; includes CaaX motif (farnesylation + proteolytic processing); "
            "    Lamin C: 572 aa; shorter isoform; not farnesylated; "
            "  Nuclear lamina: meshwork underlying inner nuclear membrane; "
            "    Functions: nuclear shape + mechanical stability; heterochromatin organisation; "
            "    Gene expression regulation: lamin-associated domains (LADs) = transcriptionally silent; "
            "    DNA repair, replication, cell cycle; "
            "  Adipocyte lamin A/C: regulates adipogenic gene expression (PPARgamma, C/EBPalpha LADs); "
            "FPLD2 (DUNNIGAN SYNDROME) -- Arg482 HOTSPOT MUTATIONS: "
            "  p.Arg482Trp, p.Arg482Gln, p.Arg482Leu: >90% of FPLD2 mutations cluster at codon 482; "
            "  Mutation changes surface charge of Ig-fold domain -> altered interactions with HP1/BAF; "
            "  Leads to redistribution of heterochromatin -> altered gene expression in mature adipocytes; "
            "  Adipogenesis proceeds but mature adipocyte maintenance fails at PUBERTY; "
            "  ONSET AT PUBERTY: fat lost from extremities + gluteal starting puberty (key feature); "
            "  Fat accumulates face/neck/abdomen (compensatory or altered distribution); "
            "  Females more severely affected metabolically (more visible clinical phenotype); "
            "  LAMINOPATHY OVERLAP: LMNA mutations also cause EDMD, LGMD1B, DCM, FPLD2, Hutchinson-Gilford progeria; "
            "encoded 1q22; OMIM gene 150330, disease FPLD2 #151660"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT -- LMNA / FAMILIAL PARTIAL LIPODYSTROPHY TYPE 2 (FPLD2 -- DUNNIGAN SYNDROME): "
            "  MOST COMMON hereditary partial lipodystrophy; "
            "  ONSET: puberty (distinguishes from CGL -- not congenital); "
            "  FAT DISTRIBUTION: "
            "    LOSS: extremities (arms + legs) + gluteal region -> limbs thin/muscular appearance; "
            "    ACCUMULATION: face + neck + abdomen (moon-face appearance, abdomen prominent); "
            "    Neck fat often mistaken for Cushing syndrome; "
            "    Females: more severe loss of extremity fat; more pronounced metabolic syndrome; "
            "    Males: milder and sometimes missed; "
            "  METABOLIC: "
            "    Hypertriglyceridemia (200-800 mg/dL); "
            "    Insulin resistance + DM (40-70%); "
            "    Hypertension; "
            "    Low HDL; "
            "    Leptin: low-normal (partial fat preserved in face/neck; some leptin production); "
            "  LAMINOPATHY COMPLICATIONS: "
            "    CARDIOMYOPATHY (dilated or hypertrophic): major complication; "
            "    Conduction system disease: AV block, sudden cardiac death -- LMNA well-known cause; "
            "    Muscular dystrophy overlap (some FPLD2 families: proximal weakness); "
            "    Premature death: cardiac/arrhythmia (main cause of premature mortality in FPLD2); "
            "  DIAGNOSIS: "
            "    Clinical: partial fat redistribution (loss from limbs, gain face/neck) starting puberty; "
            "    Exclude Cushing (neck fat accumulation); "
            "    Genetic: LMNA p.Arg482 codon (>90% of FPLD2) -- if suspicious, sequence LMNA; "
            "  TREATMENT: "
            "    Metreleptin: EFFECTIVE (leptin partially low -- replacement improves metabolic); "
            "    Fibrates + statins + antihypertensives; "
            "    Insulin / GLP-1 agonists / SGLT2i for diabetes; "
            "    TZDs (PPARG agonists): may help but limited by fluid retention; "
            "    Cardiac surveillance: ECG + Holter + echo annually; ICD if significant arrhythmia; "
            "    Genetic counselling: AD; 50% offspring risk"
        ),
        "disease_category": (
            "FPLD2-LMNA-DUNNIGAN-MOST-COMMON-FPLD-PUBERTY-ONSET-ARG482-CARDIOMYOPATHY: "
            "  KEY RULE: partial lipodystrophy starting PUBERTY = FPLD2 (LMNA) most likely; "
            "  ARG482 HOTSPOT: >90% of FPLD2; test p.Arg482Trp/Gln/Leu first; "
            "  FAT PATTERN: loss from extremities/gluteal + gain face/neck/abdomen (CUSHING MIMIC); "
            "  FEMALES MORE SEVERE: more pronounced metabolic + phenotypic expression; "
            "  CARDIOMYOPATHY + ARRHYTHMIA: LMNA = major cause of sudden cardiac death; annual cardiac surveillance; "
            "  LAMINOPATHY OVERLAP: LMNA -> EDMD / LGMD1B / DCM / progeria -- screen for overlap; "
            "  GENETIC TESTING: LMNA sequencing; Arg482 first; heterozygous AD"
        ),
    },
    {
        "gene": "PPARG",
        "protein": (
            "PPARG -- 3p25.2 AD -- 477aa -- Peroxisome-Proliferator-Activated-Receptor-Gamma-"
            "57kDa-Nuclear-Receptor-Ligand-Activated-TF-Master-Adipogenesis-Regulator-"
            "FPLD3-Haploinsufficiency-TZD-SPECIFIC-Treatment-PPARG-Agonist-OMIM-601487"
        ),
        "locus": "3p25.2",
        "protein_size": (
            "477 aa / 57 kDa (PPARG -- peroxisome proliferator-activated receptor gamma; "
            "FUNCTION: nuclear receptor; ligand-activated transcription factor; "
            "  MASTER REGULATOR OF ADIPOGENESIS: "
            "    PPARG1 (477 aa): ubiquitous; PPARG2 (505 aa): adipose-specific (28 extra N-terminal aa); "
            "    Heterodimer with RXR -> binds PPRE (PPAR response element) in target gene promoters; "
            "    Activates: adipogenic gene programme (FABP4, LPL, GLUT4, perilipin, adiponectin, leptin); "
            "    Without PPARG: adipogenesis cannot proceed; "
            "  INSULIN SENSITISER: PPARG activation -> increases fatty acid uptake into adipocytes; "
            "    Reduces ectopic fat; improves peripheral insulin sensitivity; "
            "  LIGANDS: endogenous = fatty acid derivatives; pharmacological = TZDs (rosiglitazone, pioglitazone); "
            "FPLD3 -- PPARG HAPLOINSUFFICIENCY: "
            "  Heterozygous LOF: reduced PPARG activity -> partial failure of adipogenesis; "
            "  PARTIAL lipodystrophy: less severe than CGL; fat lost from extremities but face/neck preserved; "
            "  Metabolic syndrome: hypertension + dyslipidemia + insulin resistance + DM; "
            "  THIAZOLIDINEDIONES (TZDs) = SPECIFIC TREATMENT: "
            "    TZDs are PPARG agonists -> pharmacologically supplement the lost PPARG function; "
            "    Rosiglitazone/pioglitazone: improve insulin sensitivity, reduce HTG, improve fat distribution; "
            "    MECHANISM: TZDs activate remaining wild-type PPARG allele -> partial functional recovery; "
            "    Note: fluid retention side effect; cardiac risk with rosiglitazone (monitor); "
            "PPARG GOF mutations: "
            "  Rare; cause severe obesity + insulin hypersensitivity (opposite direction); "
            "encoded 3p25.2; OMIM gene 601487, disease FPLD3 #604367"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT -- PPARG / FAMILIAL PARTIAL LIPODYSTROPHY TYPE 3 (FPLD3): "
            "  ONSET: typically adult (post-pubertal; less clear onset than FPLD2); "
            "  PHENOTYPE: "
            "    Partial lipoatrophy: mainly lower limbs + gluteal; less severe than FPLD2; "
            "    Face/neck fat relatively preserved (unlike FPLD2 where face gains fat); "
            "    Some patients have face fat loss as well; "
            "    Metabolic: hypertension + hypertriglyceridemia + low HDL + insulin resistance + DM (40-70%); "
            "    PCOS in females; "
            "  LEPTIN: low-normal (partial fat preserved); "
            "  METABOLIC SYNDROME PROMINENT: "
            "    Hypertension often severe; resistant hypertension; "
            "    Dyslipidemia (hypertriglyceridemia + low HDL); "
            "    Type 2 DM (insulin resistant); "
            "  TZD SPECIFIC TREATMENT: "
            "    Pioglitazone or rosiglitazone: PPARG agonist -> activates remaining WT PPARG -> "
            "      improves peripheral fat storage -> reduces ectopic fat -> improves insulin sensitivity; "
            "    Clinically: reduces HbA1c + triglycerides + improves fat distribution; "
            "    Contraindications: heart failure (TZD fluid retention); pioglitazone preferred; "
            "  DIAGNOSIS: "
            "    Clinical: partial lipodystrophy + metabolic syndrome; "
            "    Genetic: PPARG sequencing; heterozygous LOF mutation; "
            "    Functional: luciferase reporter assay for novel variants; "
            "  TREATMENT: "
            "    TZDs (SPECIFIC); metformin; GLP-1 agonists; fibrates; antihypertensives; "
            "    Metreleptin: less studied than CGL; may help in severe cases; "
            "  PROGNOSIS: metabolic syndrome complications (CV disease); less laminopathy overlap than FPLD2"
        ),
        "disease_category": (
            "FPLD3-PPARG-HAPLOINSUFFICIENCY-TZD-SPECIFIC-TREATMENT: "
            "  KEY RULE: FPLD3 specific treatment = TZD (PPARG agonist replaces haploinsufficient function); "
            "  TZD MECHANISM: pharmacological PPARG agonism supplements reduced PPARG activity; "
            "  METABOLIC SYNDROME PROMINENT: hypertension + dyslipidemia + DM 40-70%; "
            "  PPARG = MASTER ADIPOGENESIS: LOF -> partial adipogenesis failure -> partial lipodystrophy; "
            "  FPLD3 vs FPLD2: FPLD3 less severe fat redistribution; no cardiomyopathy/laminopathy overlap; "
            "  GENETIC TESTING: PPARG sequencing; heterozygous LOF; AD inheritance"
        ),
    },
    {
        "gene": "AKT2",
        "protein": (
            "AKT2 -- 19q13.2 AD -- 481aa -- AKT-Serine-Threonine-Kinase-2-"
            "56kDa-PI3K-Downstream-Effector-PH-Kinase-Regulatory-Domain-"
            "FPLD6-Partial-Lipodystrophy-SEVERE-Insulin-Resistance-AD-Incomplete-Penetrance-Somatic-GOF-Hypoglycemia-OMIM-164731"
        ),
        "locus": "19q13.2",
        "protein_size": (
            "481 aa / 56 kDa (AKT2 -- AKT serine/threonine kinase 2; Protein Kinase B beta; "
            "FUNCTION: central effector kinase downstream of insulin receptor / PI3K; "
            "  Insulin -> InsR -> IRS1/2 -> PI3K -> PIP3 -> AKT2 activation (by PDK1 + mTORC2); "
            "  AKT2 substrates in adipocytes: "
            "    AS160 (TBC1D4): phosphorylation -> GLUT4 vesicle translocation to membrane -> glucose uptake; "
            "    FOXO1: phosphorylation -> nuclear exclusion -> suppress gluconeogenesis; "
            "    GSK3: phosphorylation -> glycogen synthesis; "
            "    TSC2: phosphorylation -> mTORC1 -> protein synthesis; "
            "  AKT2 is the predominant AKT isoform in adipose tissue + liver for metabolic functions; "
            "AKT2 LOF -> FPLD6 (GERMLINE): "
            "  Heterozygous LOF: partial lipodystrophy (mild to moderate lipoatrophy); "
            "  SEVERE insulin resistance (despite partial fat loss -- insulin signalling intrinsically impaired); "
            "  AD with INCOMPLETE PENETRANCE: not all carriers develop full phenotype; "
            "  Triglycerides elevated (200-600 mg/dL); "
            "AKT2 SOMATIC GOF MUTATIONS (OPPOSITE PHENOTYPE): "
            "  Somatic gain-of-function AKT2 mutations in pancreatic islets -> constitutive AKT2 -> "
            "    Hyperinsulinaemia -> hypoglycaemia (insulinoma-like without structural tumour); "
            "  NOT inherited germline -- somatic mosaicism; "
            "  OPPOSITE: germline LOF = insulin resistance; somatic GOF = hypoglycaemia; "
            "encoded 19q13.2; OMIM gene 164731, disease FPLD6 #615980"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (INCOMPLETE PENETRANCE) -- AKT2 / FPLD TYPE 6 (FPLD6): "
            "  ONSET: variable; often adult onset; "
            "  PHENOTYPE: "
            "    Partial lipodystrophy (variable severity of lipoatrophy; generally milder fat loss than FPLD2); "
            "    SEVERE insulin resistance: disproportionately severe relative to fat loss; "
            "      HOMA-IR markedly elevated; extreme post-prandial hyperinsulinaemia; "
            "    Diabetes mellitus (40-60%); "
            "    Hypertriglyceridemia (200-600 mg/dL); "
            "    Acanthosis nigricans; "
            "    Polycystic ovaries in females; "
            "  INCOMPLETE PENETRANCE: "
            "    Not all heterozygous carriers manifest full phenotype; "
            "    Variable expressivity within families; "
            "    Can be missed in family cascade testing; "
            "  SOMATIC GOF -- DIFFERENT DISEASE: "
            "    Somatic AKT2 gain-of-function mutations -> constitutive kinase activity in islets; "
            "    -> Hypoglycaemia (hyperinsulinaemic) in isolated islets; "
            "    -> Treated with diazoxide or pancreatectomy (different from FPLD6); "
            "    CLINICAL PEARL: AKT2 mutations can cause OPPOSITE phenotypes depending on germline vs somatic + LOF vs GOF; "
            "  DIAGNOSIS: "
            "    Clinical: partial lipodystrophy + severe insulin resistance; "
            "    Genetic: AKT2 sequencing; heterozygous LOF germline; "
            "    If hypoglycaemia: test for somatic GOF (requires tissue from affected area); "
            "  TREATMENT: "
            "    Metformin + insulin sensitisers; GLP-1 agonists; SGLT2 inhibitors; "
            "    Insulin if DM; "
            "    Metreleptin: data limited; may help in severe cases; "
            "    Fibrates for HTG; "
            "  PROGNOSIS: severe insulin resistance; CV risk from metabolic syndrome"
        ),
        "disease_category": (
            "FPLD6-AKT2-SEVERE-INSULIN-RESISTANCE-INCOMPLETE-PENETRANCE-SOMATIC-GOF-HYPOGLYCEMIA: "
            "  KEY RULE: AKT2 germline LOF = FPLD + SEVERE insulin resistance; somatic GOF = hypoglycaemia (OPPOSITE); "
            "  INCOMPLETE PENETRANCE: AD but variable; do not dismiss carriers as unaffected without metabolic testing; "
            "  SEVERE INSULIN RESISTANCE: disproportionate to degree of fat loss -- intrinsic signalling defect; "
            "  AKT2 = central insulin signalling effector in adipose/liver; LOF -> downstream cascade fails; "
            "  SOMATIC GOF DISTINCT: requires tissue testing; different treatment (diazoxide/pancreatectomy); "
            "  GENETIC TESTING: AKT2 germline sequencing; heterozygous LOF confirms FPLD6"
        ),
    },
    {
        "gene": "PLIN1",
        "protein": (
            "PLIN1 -- 15q26.1 AD -- 522aa -- Perilipin-1-"
            "56kDa-Lipid-Droplet-Coat-Protein-PAT-Domain-ABHD5-Interaction-Lipolysis-Regulation-"
            "FPLD4-Partial-Lipodystrophy-Severe-HTG-Pancreatitis-Frameshift-Heterozygous-LOF-OMIM-170290"
        ),
        "locus": "15q26.1",
        "protein_size": (
            "522 aa / 56 kDa (PLIN1 -- Perilipin-1; founding member of the PAT (perilipin-adipophilin-TIP47) family; "
            "FUNCTION: dominant coat protein of lipid droplets in adipocytes; "
            "  Localisation: cytoplasmic surface of lipid droplets in white and brown adipocytes; "
            "  LIPOLYSIS REGULATION (dual gatekeeper): "
            "    BASAL STATE (unstimulated): "
            "      PLIN1 coats LD surface -> physical barrier against lipases; "
            "      Sequesters ABHD5 (ATGL co-activator, also called CGI-58) -> ATGL inactive; "
            "      ATGL (adipose triglyceride lipase) = rate-limiting triglyceride hydrolase; "
            "      Net: basal lipolysis suppressed; fat stored efficiently; "
            "    STIMULATED (catecholamines -> PKA): "
            "      PKA phosphorylates PLIN1 at Ser81/Ser522 -> PLIN1 conformation change; "
            "      ABHD5 released from PLIN1 -> binds and activates ATGL; "
            "      Phosphorylated PLIN1 also recruits HSL (hormone-sensitive lipase) -> TAG -> DAG; "
            "      Net: hormonally stimulated lipolysis proceeds; "
            "  PLIN1 LOF -> unregulated basal lipolysis: "
            "    ABHD5 not sequestered -> ATGL constitutively active -> unregulated lipolysis; "
            "    Excess FFA release -> ectopic fat accumulation + severe hypertriglyceridemia; "
            "    Reduced fat storage capacity -> partial lipodystrophy; "
            "FPLD4 -- HETEROZYGOUS FRAMESHIFT/LOF: "
            "  Frameshift mutations cause haploinsufficiency -> partial PLIN1 loss -> partial lipolysis dysregulation; "
            "encoded 15q26.1; OMIM gene 170290, disease FPLD4 #613877"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT -- PLIN1 / FAMILIAL PARTIAL LIPODYSTROPHY TYPE 4 (FPLD4): "
            "  ONSET: variable; often becomes apparent in 2nd-4th decade; "
            "  PHENOTYPE: "
            "    Partial lipodystrophy: loss of subcutaneous fat from extremities; "
            "    Fat preserved face/neck (pattern similar to FPLD2 but milder); "
            "    SEVERE HYPERTRIGLYCERIDEMIA: 400-2000+ mg/dL; often most prominent feature; "
            "      Mechanism: unregulated lipolysis -> excess FFA -> hepatic VLDL overproduction; "
            "    PANCREATITIS RISK: severe HTG -> recurrent pancreatitis (chylomicronaemia); "
            "      Pancreatitis = major morbidity in FPLD4; "
            "    Insulin resistance + DM (40-60%); "
            "    Hepatic steatosis (ectopic fat from unregulated FFA flux); "
            "    Leptin: low-normal (partial fat preserved); "
            "  LIPOLYSIS DYSREGULATION -- CORE MECHANISM: "
            "    PLIN1 haploinsufficiency -> impaired ABHD5/ATGL gating -> basal lipolysis elevated; "
            "    Elevated basal FFA -> hypertriglyceridemia -> VLDL overproduction; "
            "    Even moderate dietary fat intake -> extreme HTG; "
            "  PANCREATITIS: "
            "    Recurrent acute pancreatitis; "
            "    Mechanism: TG >1000 mg/dL -> pancreatic lipase -> FFA in pancreatic capillaries -> injury; "
            "    Complications: necrosis, pseudocyst, chronic pancreatitis, exocrine insufficiency; "
            "  DIAGNOSIS: "
            "    Clinical: partial lipodystrophy + severe HTG + pancreatitis history; "
            "    Genetic: PLIN1 sequencing; heterozygous frameshift/LOF; "
            "  TREATMENT: "
            "    Severe HTG management: fibrates (first-line) + omega-3 fatty acids + strict low-fat diet; "
            "    Volanesorsen/evinacumab for refractory severe HTG; "
            "    Insulin for DM; "
            "    Metreleptin: limited data; may help; "
            "    Pancreatitis: acute management (NPO, IV fluids, analgesia); prophylactic HTG control; "
            "  PROGNOSIS: pancreatitis events = main acute morbidity; chronic DM + CV risk"
        ),
        "disease_category": (
            "FPLD4-PLIN1-SEVERE-HTG-PANCREATITIS-RISK-UNREGULATED-LIPOLYSIS: "
            "  KEY RULE: FPLD4 = severe HTG + pancreatitis risk = PLIN1 unregulated lipolysis; "
            "  PANCREATITIS: recurrent when TG >1000 mg/dL; aggressive HTG control mandatory; "
            "  PLIN1 = LIPID DROPLET GATEKEEPER: LOF -> ABHD5/ATGL dysregulated -> basal lipolysis unrestrained; "
            "  SEVERE HTG MECHANISM: unregulated FFA -> hepatic VLDL overproduction; "
            "  TREATMENT: fibrates + omega-3 + strict low-fat diet; volanesorsen for refractory; "
            "  GENETIC TESTING: PLIN1 sequencing; heterozygous frameshift confirms FPLD4"
        ),
    },
]


def _make_patients(seed: int, gene: str) -> list:
    """Generate 40 synthetic lipodystrophy-spectrum patients per gene."""
    rng = random.Random(seed)

    # CGL genes: near-absent leptin, very high TG, severe IR, high DM rate
    # FPLD genes: partial fat loss, low-normal leptin, moderate-high TG, variable DM
    gene_profiles = {
        "AGPAT2": dict(
            tg_min=500,  tg_max=3000,
            leptin_min=0.1, leptin_max=0.8,
            homa_ir_min=15, homa_ir_max=80,
            dm_pct=0.85,
            metreleptin_eligible_pct=0.95,
            pancreatitis_pct=0.45,
            is_cgl=True,
        ),
        "BSCL2": dict(
            tg_min=600,  tg_max=3500,
            leptin_min=0.05, leptin_max=0.5,
            homa_ir_min=20, homa_ir_max=90,
            dm_pct=0.90,
            metreleptin_eligible_pct=0.95,
            pancreatitis_pct=0.50,
            is_cgl=True,
        ),
        "CAV1": dict(
            tg_min=400,  tg_max=2000,
            leptin_min=0.3, leptin_max=1.5,
            homa_ir_min=10, homa_ir_max=60,
            dm_pct=0.70,
            metreleptin_eligible_pct=0.80,
            pancreatitis_pct=0.35,
            is_cgl=True,
        ),
        "CAVIN1": dict(
            tg_min=400,  tg_max=2200,
            leptin_min=0.3, leptin_max=1.8,
            homa_ir_min=12, homa_ir_max=65,
            dm_pct=0.72,
            metreleptin_eligible_pct=0.80,
            pancreatitis_pct=0.38,
            is_cgl=True,
        ),
        "LMNA": dict(
            tg_min=200,  tg_max=800,
            leptin_min=2.0, leptin_max=8.0,
            homa_ir_min=4,  homa_ir_max=30,
            dm_pct=0.60,
            metreleptin_eligible_pct=0.65,
            pancreatitis_pct=0.12,
            is_cgl=False,
        ),
        "PPARG": dict(
            tg_min=180,  tg_max=700,
            leptin_min=2.5, leptin_max=9.0,
            homa_ir_min=3,  homa_ir_max=25,
            dm_pct=0.55,
            metreleptin_eligible_pct=0.40,
            pancreatitis_pct=0.10,
            is_cgl=False,
        ),
        "AKT2": dict(
            tg_min=200,  tg_max=600,
            leptin_min=3.0, leptin_max=10.0,
            homa_ir_min=8,  homa_ir_max=50,
            dm_pct=0.55,
            metreleptin_eligible_pct=0.35,
            pancreatitis_pct=0.08,
            is_cgl=False,
        ),
        "PLIN1": dict(
            tg_min=400,  tg_max=2000,
            leptin_min=2.0, leptin_max=8.0,
            homa_ir_min=5,  homa_ir_max=35,
            dm_pct=0.50,
            metreleptin_eligible_pct=0.50,
            pancreatitis_pct=0.55,
            is_cgl=False,
        ),
    }
    p = gene_profiles.get(gene, gene_profiles["AGPAT2"])

    associated_features_map = {
        "AGPAT2": [
            ["near-complete-lipoatrophy", "acanthosis-nigricans", "hepatomegaly"],
            ["severe-hypertriglyceridemia", "eruptive-xanthomata"],
            ["near-complete-lipoatrophy", "muscular-appearance", "prominent-veins"],
            ["DM-extreme-insulin-resistance", "pancreatitis", "hepatic-steatosis"],
            ["acromegaloid-features", "near-complete-lipoatrophy", "severe-HTG"],
        ],
        "BSCL2": [
            ["complete-lipoatrophy-all-fat-absent", "intellectual-disability", "hepatomegaly"],
            ["hypertrophic-cardiomyopathy", "severe-HTG", "DM"],
            ["mechanical-fat-absent", "severe-lipoatrophy", "acanthosis-nigricans"],
            ["intellectual-disability", "pancreatitis", "hepatic-cirrhosis"],
            ["complete-lipoatrophy", "BSCL2-most-severe-CGL", "near-absent-leptin"],
        ],
        "CAV1": [
            ["generalised-lipoatrophy-milder", "pulmonary-arterial-hypertension"],
            ["caveolae-absent-EM-pathognomonic", "hepatic-steatosis"],
            ["partial-mechanical-fat-spared", "PAH", "moderate-HTG"],
            ["lipoatrophy", "insulin-resistance", "PAH-screening-required"],
        ],
        "CAVIN1": [
            ["generalised-lipoatrophy", "skeletal-muscle-myopathy", "elevated-CK"],
            ["cardiac-arrhythmia", "myopathy", "caveolae-absent-EM"],
            ["limb-girdle-myopathy", "elevated-CK-500-5000", "HTG"],
            ["muscular-dystrophy-phenotype", "arrhythmia", "lipoatrophy"],
            ["elevated-CK", "respiratory-insufficiency-risk", "cardiac-conduction-abnormality"],
        ],
        "LMNA": [
            ["partial-lipodystrophy-puberty-onset", "fat-loss-extremities", "fat-gain-face-neck"],
            ["cardiomyopathy", "arrhythmia", "conduction-disease"],
            ["Dunnigan-phenotype", "partial-lipoatrophy-limbs", "metabolic-syndrome"],
            ["Arg482-mutation", "females-more-severe", "hypertriglyceridemia"],
            ["laminopathy-overlap", "cardiac-sudden-death-risk", "fat-redistribution"],
        ],
        "PPARG": [
            ["partial-lipoatrophy-lower-limbs", "hypertension", "dyslipidemia"],
            ["FPLD3-TZD-responsive", "insulin-resistance", "DM"],
            ["hypertension-resistant", "low-HDL", "hypertriglyceridemia"],
            ["partial-lipodystrophy", "PCOS-females", "metabolic-syndrome"],
        ],
        "AKT2": [
            ["partial-lipodystrophy", "severe-insulin-resistance-disproportionate"],
            ["acanthosis-nigricans", "severe-HOMA-IR", "incomplete-penetrance"],
            ["partial-lipoatrophy", "DM-insulin-resistant", "HTG"],
            ["AKT2-LOF", "severe-post-prandial-hyperinsulinemia", "PCOS"],
        ],
        "PLIN1": [
            ["partial-lipodystrophy", "severe-HTG-400-2000", "pancreatitis-recurrent"],
            ["chylomicronaemia", "eruptive-xanthomata", "pancreatitis"],
            ["unregulated-lipolysis", "hepatic-steatosis", "severe-HTG"],
            ["FPLD4-PLIN1-frameshift", "lipoatrophy-extremities", "severe-HTG"],
        ],
    }
    features_options = associated_features_map.get(gene, [["lipodystrophy"]])

    treatment_map = {
        "AGPAT2": [
            "metreleptin+low-fat-diet+fibrates",
            "metreleptin+insulin+low-fat-diet",
            "metreleptin+fibrates+omega3",
        ],
        "BSCL2": [
            "metreleptin+insulin+low-fat-diet",
            "metreleptin+fibrates+cardiac-surveillance",
            "metreleptin+low-fat-diet+fibrates+hepatology",
        ],
        "CAV1": [
            "metreleptin+low-fat-diet+PAH-screening",
            "metreleptin+fibrates+sildenafil-for-PAH",
            "low-fat-diet+fibrates+insulin+echo-annually",
        ],
        "CAVIN1": [
            "metreleptin+low-fat-diet+cardiac-surveillance+ICD-if-arrhythmia",
            "metreleptin+fibrates+ECG-Holter-annually+physiotherapy",
            "low-fat-diet+insulin+cardiac-monitoring+spirometry",
        ],
        "LMNA": [
            "metreleptin+fibrates+statins+cardiac-surveillance",
            "insulin+fibrates+ACE-inhibitor+ECG-Holter-annually",
            "GLP1-agonist+fibrates+cardiomyopathy-management",
        ],
        "PPARG": [
            "pioglitazone-TZD-SPECIFIC+fibrates+antihypertensives",
            "rosiglitazone+metformin+fibrates",
            "pioglitazone+GLP1-agonist+SGLT2i+fibrates",
        ],
        "AKT2": [
            "metformin+GLP1-agonist+fibrates",
            "insulin+SGLT2i+fibrates",
            "metformin+insulin+fibrates+PCOS-management",
        ],
        "PLIN1": [
            "fibrates+omega3+strict-low-fat-diet+pancreatitis-protocol",
            "volanesorsen+fibrates+low-fat-diet",
            "insulin+fibrates+omega3+pancreatitis-prevention",
        ],
    }
    treatments = treatment_map.get(gene, ["low-fat-diet+fibrates"])

    mutation_map = {
        "AGPAT2": ["p.Asn94Ser", "p.Glu260Lys", "p.Gly188Arg", "del_exon3", "p.Arg218Ter"],
        "BSCL2":  ["p.Asn88Ser", "p.Glu189Lys", "p.Phe286Leu", "p.Cys295Arg", "p.Trp341Ter"],
        "CAV1":   ["p.Pro132Leu", "p.Phe160Ter", "del_exon2-3", "p.Arg169Trp", "p.Leu141Arg"],
        "CAVIN1": ["p.Glu264Ter", "p.Gln258Ter", "del_exon5", "p.Arg244Ter", "p.Ser186Phe"],
        "LMNA":   ["p.Arg482Trp", "p.Arg482Gln", "p.Arg482Leu", "p.Asn466Asp", "p.Lys486Asn"],
        "PPARG":  ["p.Pro467Leu", "p.Phe388Leu", "p.Arg280Cys", "p.Val290Met", "p.Cys114Arg"],
        "AKT2":   ["p.Arg274His", "p.Arg208Cys", "p.Arg274Cys", "p.Val270Ala", "p.Asp219Asn"],
        "PLIN1":  ["c.1210_1211insC", "c.1546_1547insTG", "c.1210delC", "p.Leu404Ter", "c.1209_1213del"],
    }
    mutations = mutation_map.get(gene, ["unknown"])

    patients = []
    for i in range(40):
        tg = round(rng.uniform(p["tg_min"], p["tg_max"]), 1)
        leptin = round(rng.uniform(p["leptin_min"], p["leptin_max"]), 2)
        homa_ir = round(rng.uniform(p["homa_ir_min"], p["homa_ir_max"]), 1)
        has_dm = rng.random() < p["dm_pct"]
        has_pancreatitis = rng.random() < p["pancreatitis_pct"]
        metreleptin_used = rng.random() < p["metreleptin_eligible_pct"]

        # HbA1c: DM patients elevated; non-DM lower
        if has_dm:
            hba1c = round(rng.uniform(7.5, 12.0), 1)
        else:
            hba1c = round(rng.uniform(5.2, 6.9), 1)

        # Gene-specific additional features
        ck_elevated = False
        ck_value = rng.randint(30, 180)
        has_pah = False
        has_intellectual_disability = False
        has_cardiac_arrhythmia = False

        if gene == "CAVIN1":
            ck_elevated = rng.random() < 0.82  # >80% CAVIN1 have elevated CK
            ck_value = rng.randint(500, 5000) if ck_elevated else rng.randint(50, 200)
            has_cardiac_arrhythmia = rng.random() < 0.55
        if gene == "CAV1":
            has_pah = rng.random() < 0.18
        if gene == "BSCL2":
            has_intellectual_disability = rng.random() < 0.45  # 30-60% range
        if gene == "LMNA":
            has_cardiac_arrhythmia = rng.random() < 0.40

        # Onset: CGL = congenital; FPLD = puberty/adult
        if p["is_cgl"]:
            age_dx = rng.randint(0, 5)
        else:
            age_dx = rng.randint(14, 55)

        lipodystrophy_type = "CGL" if p["is_cgl"] else "FPLD"

        treatment = rng.choice(treatments)
        mutation = rng.choice(mutations)
        features = rng.choice(features_options)

        patients.append({
            "id":                          f"{gene}-{seed}-{i+1:03d}",
            "gene":                        gene,
            "lipodystrophy_type":          lipodystrophy_type,
            "age_at_diagnosis":            age_dx,
            "triglycerides_mg_dL":         tg,
            "leptin_ng_mL":                leptin,
            "homa_ir":                     homa_ir,
            "diabetes_mellitus":           has_dm,
            "hba1c_pct":                   hba1c,
            "pancreatitis_episode":        has_pancreatitis,
            "metreleptin_treatment":       metreleptin_used,
            "ck_IU_L":                     ck_value,
            "ck_elevated":                 ck_elevated,
            "pulmonary_arterial_hypert":   has_pah,
            "intellectual_disability":     has_intellectual_disability,
            "cardiac_arrhythmia":          has_cardiac_arrhythmia,
            "associated_features":         features,
            "treatment":                   treatment,
            "mutation":                    mutation,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Lipodystrophy-Atlas."""
    return {
        "atlas":          "Hereditary-Lipodystrophy-Atlas",
        "subtitle":       (
            "Complete 8-Gene Lipodystrophy Reference Atlas "
            "(AGPAT2-BSCL2-CAV1-CAVIN1-LMNA-PPARG-AKT2-PLIN1)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "AGPAT2": "AR LOF (AGPAT2 deficiency; CGL1; near-complete fat absence from birth; metreleptin SPECIFIC)",
            "BSCL2":  "AR LOF (Seipin; CGL2 MOST COMMON + MOST SEVERE; all fat absent; intellectual disability 30-60%)",
            "CAV1":   "AR LOF (Caveolin-1; CGL3; caveolae absent EM PATHOGNOMONIC; PAH unique complication)",
            "CAVIN1": "AR LOF (Cavin1/PTRF; CGL4; ONLY CGL with myopathy + elevated CK + cardiac arrhythmia)",
            "LMNA":   "AD LOF (Lamin A/C; FPLD2 Dunnigan MOST COMMON FPLD; Arg482 hotspot; puberty onset; cardiomyopathy)",
            "PPARG":  "AD LOF (PPAR-gamma haploinsufficiency; FPLD3; TZDs = SPECIFIC treatment)",
            "AKT2":   "AD LOF (AKT2 germline LOF; FPLD6; severe insulin resistance; incomplete penetrance)",
            "PLIN1":  "AD LOF (Perilipin-1 frameshift; FPLD4; severe HTG + pancreatitis; unregulated lipolysis)",
        },
        "key_clinical_rules": [
            "CGL-vs-FPLD: CGL (AGPAT2/BSCL2/CAV1/CAVIN1) = congenital near-complete fat absence + near-absent leptin (<1 ng/mL) + TG 500-3000 mg/dL + HOMA-IR 10-80; FPLD (LMNA/PPARG/AKT2/PLIN1) = puberty/adult onset partial fat redistribution + partial leptin loss + TG 200-800 mg/dL",
            "METRELEPTIN-SPECIFIC: recombinant leptin is SPECIFIC treatment for CGL (all 4 types) and beneficial in FPLD; replaces absent/low leptin -> reduces hyperphagia, lowers TG + HbA1c, improves hepatic steatosis; FDA/EMA approved for CGL",
            "CGL1-vs-CGL2: AGPAT2 (CGL1) = mechanical fat spared (palms/orbits); NO intellectual disability; BSCL2 (CGL2) = ALL fat absent including mechanical; intellectual disability 30-60%; more severe cardiomyopathy",
            "CGL3-vs-CGL4-CAVEOLAE: both CAV1 (CGL3) and CAVIN1 (CGL4) = caveolae absent on EM (PATHOGNOMONIC); CGL3 = pulmonary arterial hypertension (unique); CGL4 = myopathy + elevated CK (500-5000 IU/L) + cardiac arrhythmia (UNIQUE -- not in any other CGL)",
            "LMNA-ARG482-DUNNIGAN: >90% FPLD2 mutations at Arg482 codon (Trp/Gln/Leu); puberty onset fat loss from extremities/gluteal + gain face/neck/abdomen; females more severely affected; cardiomyopathy + arrhythmia = LMNA laminopathy; ECG + Holter annually",
            "PPARG-TZD-SPECIFIC: FPLD3 = PPAR-gamma haploinsufficiency -> TZDs (pioglitazone/rosiglitazone) pharmacologically supplement lost function -> SPECIFIC treatment; TZDs reduce insulin resistance + TG + improve fat distribution",
            "AKT2-GERMLINE-vs-SOMATIC: AKT2 germline LOF = FPLD6 + severe insulin resistance; AKT2 somatic GOF = hypoglycaemia (OPPOSITE PHENOTYPE); incomplete penetrance in FPLD6 families -- check all carriers metabolically",
            "PLIN1-PANCREATITIS: FPLD4 = severe HTG (400-2000 mg/dL) + pancreatitis risk; PLIN1 is lipid droplet gatekeeper; LOF -> unregulated ATGL/ABHD5 -> basal lipolysis -> FFA excess -> HTG; fibrates + omega-3 + low-fat diet mandatory; volanesorsen for refractory",
            "LEPTIN-BIOMARKER: leptin level as lipodystrophy severity marker -- CGL leptin <1 ng/mL (confirms near-complete fat absence); FPLD leptin low-normal (2-10 ng/mL); leptin deficiency = metabolic driver across all types",
            "8-GENE-DIFFERENTIAL: CGL (congenital, leptin <1, TG >500) vs FPLD (puberty/adult, leptin partial, TG 200-800); within CGL: CGL2 most severe + intellectual disability; within FPLD: FPLD2 most common (LMNA Arg482); TZD specific for PPARG; pancreatitis prominent in PLIN1",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 hereditary lipodystrophy-spectrum genes."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        treatments = {}
        for pt in pts:
            treatments[pt["treatment"]] = treatments.get(pt["treatment"], 0) + 1
        mutations_seen = {}
        for pt in pts:
            mutations_seen[pt["mutation"]] = mutations_seen.get(pt["mutation"], 0) + 1

        mean_tg = round(sum(pt["triglycerides_mg_dL"] for pt in pts) / len(pts), 1)
        mean_leptin = round(sum(pt["leptin_ng_mL"] for pt in pts) / len(pts), 2)
        mean_homa_ir = round(sum(pt["homa_ir"] for pt in pts) / len(pts), 1)
        dm_pct = round(100 * sum(1 for pt in pts if pt["diabetes_mellitus"]) / len(pts), 1)
        pancreatitis_pct = round(100 * sum(1 for pt in pts if pt["pancreatitis_episode"]) / len(pts), 1)
        metreleptin_pct = round(100 * sum(1 for pt in pts if pt["metreleptin_treatment"]) / len(pts), 1)
        ck_elevated_pct = round(100 * sum(1 for pt in pts if pt["ck_elevated"]) / len(pts), 1)
        pah_pct = round(100 * sum(1 for pt in pts if pt["pulmonary_arterial_hypert"]) / len(pts), 1)
        id_pct = round(100 * sum(1 for pt in pts if pt["intellectual_disability"]) / len(pts), 1)
        arrhythmia_pct = round(100 * sum(1 for pt in pts if pt["cardiac_arrhythmia"]) / len(pts), 1)
        mean_age_dx = round(sum(pt["age_at_diagnosis"] for pt in pts) / len(pts), 1)
        mean_hba1c = round(sum(pt["hba1c_pct"] for pt in pts) / len(pts), 1)

        genes_data.append({
            "gene":                       g["gene"],
            "locus":                      g["locus"],
            "protein":                    g["protein"],
            "protein_size":               g["protein_size"],
            "inheritance":                g["inheritance"],
            "disease_category":           g["disease_category"],
            "n_patients":                 len(pts),
            "mean_triglycerides_mg_dL":   mean_tg,
            "mean_leptin_ng_mL":          mean_leptin,
            "mean_homa_ir":               mean_homa_ir,
            "dm_prevalence_pct":          dm_pct,
            "pancreatitis_pct":           pancreatitis_pct,
            "metreleptin_treatment_pct":  metreleptin_pct,
            "ck_elevated_pct":            ck_elevated_pct,
            "pah_pct":                    pah_pct,
            "intellectual_disability_pct": id_pct,
            "cardiac_arrhythmia_pct":     arrhythmia_pct,
            "mean_age_dx":                mean_age_dx,
            "mean_hba1c_pct":             mean_hba1c,
            "treatment_breakdown":        treatments,
            "mutation_breakdown":         mutations_seen,
            "patients":                   pts,
        })
    return {
        "atlas": "Hereditary-Lipodystrophy-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Key clinical definitions for Hereditary-Lipodystrophy-Atlas."""
    definitions = [
        {
            "term": "CGL vs FPLD -- Congenital Generalised vs Familial Partial Lipodystrophy Differential",
            "genes": ["AGPAT2", "BSCL2", "CAV1", "CAVIN1", "LMNA", "PPARG", "AKT2", "PLIN1"],
            "definition": (
                "CGL vs FPLD -- DIFFERENTIAL DIAGNOSIS: "
                "CONGENITAL GENERALISED LIPODYSTROPHY (CGL): "
                "  ONSET: congenital (birth or first year); "
                "  FAT DISTRIBUTION: near-complete or complete absence of subcutaneous fat; "
                "    Muscular appearance (no subcutaneous fat -> visible musculature); "
                "    Prominent superficial veins; acromegaloid features; "
                "  LEPTIN: near-absent (<1 ng/mL for CGL1/3/4; <0.5 ng/mL for CGL2); "
                "  TRIGLYCERIDES: very high (500-3000+ mg/dL); eruptive xanthomata; pancreatitis; "
                "  HOMA-IR: severely elevated (10-80+); "
                "  DM prevalence: 70-90%; "
                "  METRELEPTIN: SPECIFIC treatment; FDA/EMA approved for CGL; "
                "  Subtypes: "
                "    CGL1 (AGPAT2): mechanical fat spared; no intellectual disability; "
                "    CGL2 (BSCL2): all fat absent; intellectual disability 30-60%; MOST SEVERE; "
                "    CGL3 (CAV1): milder; caveolae absent EM; PAH; "
                "    CGL4 (CAVIN1): myopathy + CK elevated + arrhythmia; caveolae absent EM; "
                "FAMILIAL PARTIAL LIPODYSTROPHY (FPLD): "
                "  ONSET: puberty or adult (NOT congenital -- key distinguishing feature); "
                "  FAT DISTRIBUTION: PARTIAL -- loss from specific depots (extremities/gluteal); "
                "    fat may ACCUMULATE elsewhere (face/neck in FPLD2); "
                "  LEPTIN: low-normal (partial fat preserved -> partial leptin production); "
                "  TRIGLYCERIDES: moderate-high (200-800 mg/dL); "
                "  HOMA-IR: elevated (3-50); "
                "  DM prevalence: 40-70%; "
                "  Subtypes: "
                "    FPLD2 (LMNA Arg482): MOST COMMON; puberty onset; cardiomyopathy; "
                "    FPLD3 (PPARG): haploinsufficiency; TZDs SPECIFIC treatment; "
                "    FPLD4 (PLIN1): severe HTG + pancreatitis predominant; "
                "    FPLD6 (AKT2): severe insulin resistance; incomplete penetrance; "
                "DIAGNOSIS CLUE: "
                "  Congenital + muscular appearance + leptin <1 ng/mL -> CGL panel (AGPAT2, BSCL2, CAV1, CAVIN1); "
                "  Puberty onset + partial fat redistribution -> FPLD panel (LMNA, PPARG, AKT2, PLIN1)."
            ),
        },
        {
            "term": "Metreleptin -- Recombinant Leptin Therapy for Lipodystrophy",
            "genes": ["AGPAT2", "BSCL2", "CAV1", "CAVIN1", "LMNA"],
            "definition": (
                "METRELEPTIN -- RECOMBINANT LEPTIN THERAPY: "
                "RATIONALE: all lipodystrophies share leptin deficiency (CGL: near-absent; FPLD: low); "
                "  Leptin normally produced by adipocytes: signals satiety + regulates energy homeostasis; "
                "  Absent fat -> absent leptin -> hyperphagia + neuroendocrine dysregulation; "
                "  Leptin also: regulates hepatic fat oxidation; suppresses VLDL production; sensitises insulin; "
                "MECHANISM OF ACTION: "
                "  Metreleptin = recombinant human leptin (Met-leptin); SC injection daily; "
                "  Acts on hypothalamic leptin receptor (LEPR / Ob-R); "
                "  -> POMC activation -> satiety; -> NPY/AgRP suppression -> reduced appetite; "
                "  -> Hepatic: reduces hepatic lipogenesis + increases fat oxidation -> reduces hepatic steatosis; "
                "  -> Triglycerides: markedly reduced (mechanisms: FFA flux reduction + hepatic VLDL suppression); "
                "  -> HbA1c: reduced (insulin sensitisation via central + peripheral mechanisms); "
                "CLINICAL OUTCOMES (randomised + observational data): "
                "  TG reduction: 50-80% from baseline in CGL patients; "
                "  HbA1c reduction: 2-4 percentage points in treated CGL; "
                "  Hepatic steatosis: reduced on MRI; fibrosis may stabilise; "
                "  Pancreatitis: risk reduced when TG lowered below 500 mg/dL threshold; "
                "DOSING: "
                "  CGL: 0.04-0.08 mg/kg/day SC (males lower dose -- lower fat); "
                "    Females (higher adipose target): 0.06-0.12 mg/kg/day; "
                "  Titrate by TG + HbA1c response; "
                "  Monitor: CBC (lymphoma risk signal -- neutralising antibodies reported); "
                "APPROVAL STATUS: "
                "  FDA (Myalept): approved for CGL + severe FPLD; "
                "  EMA (Myalepta): approved for lipodystrophy + leptin deficiency evidence; "
                "SIDE EFFECTS: "
                "  Hypoglycaemia: if insulin not titrated down as sensitivity improves; "
                "  Lymphoma signal: T-cell lymphoma cases in FPLD patients (REMS program in USA); "
                "  Antibody formation: neutralising anti-leptin antibodies; rare but monitor; "
                "FPLD: "
                "  Less dramatic than CGL but beneficial; TG and HbA1c reduction; "
                "  Approved in severe cases with metabolic complications."
            ),
        },
        {
            "term": "Caveolae Absent on Electron Microscopy -- Pathognomonic for CGL3 (CAV1) and CGL4 (CAVIN1)",
            "genes": ["CAV1", "CAVIN1"],
            "definition": (
                "CAVEOLAE ABSENT ON EM -- PATHOGNOMONIC FINDING (CGL3/CGL4): "
                "NORMAL CAVEOLAE: "
                "  Flask-shaped plasma membrane invaginations (50-100 nm diameter); "
                "  Electron microscopy (EM): visible as omega/flask-shaped structures at plasma membrane; "
                "  Caveolin-1 (CAV1) + Cavin-1 (CAVIN1) both required: "
                "    CAV1 scaffolds the caveolae membrane; "
                "    CAVIN1 coats the cytoplasmic face; stabilises the complex; "
                "    Without EITHER: stable caveolae cannot form -> absent on EM; "
                "  Adipocytes: particularly enriched in caveolae (30% of plasma membrane); "
                "  Skeletal muscle: abundant caveolae (membrane repair + mechanosensing); "
                "PATHOGNOMONIC IN CGL3 AND CGL4: "
                "  CGL3 (CAV1 LOF): caveolae absent on EM of skin or adipose biopsy; "
                "  CGL4 (CAVIN1 LOF): caveolae absent on EM of skeletal muscle or skin biopsy; "
                "  Both CAV1 and CAVIN1 absent -> absent caveolae; "
                "  NOT absent in CGL1 (AGPAT2) or CGL2 (BSCL2) -- different mechanism; "
                "DISTINGUISHING CGL3 vs CGL4 (BOTH have absent caveolae): "
                "  CGL3 (CAV1): "
                "    NO myopathy; normal CK; "
                "    PAH (pulmonary arterial hypertension) -- unique to CGL3; "
                "    Milder metabolic phenotype; "
                "  CGL4 (CAVIN1): "
                "    MYOPATHY: limb-girdle weakness; muscular dystrophy on biopsy; "
                "    CK markedly elevated (500-5000 IU/L) -- distinguishing biomarker; "
                "    CARDIAC ARRHYTHMIA + conduction disease; "
                "    NO PAH; "
                "  Summary: absent caveolae = CAV1 or CAVIN1; myopathy + high CK = CAVIN1; PAH = CAV1; "
                "BIOPSY PROTOCOL: "
                "  Skin biopsy (punch 4 mm) or muscle biopsy; "
                "  EM processing: glutaraldehyde fix -> osmium stain -> thin sections; "
                "  Report: presence/absence of flask-shaped caveolae at plasma membrane; "
                "GENETIC CONFIRMATION: CAV1 or CAVIN1 sequencing; biallelic LOF."
            ),
        },
        {
            "term": "CGL4 (CAVIN1) -- Myopathy and Cardiac Arrhythmia Distinguish from All Other CGL Types",
            "genes": ["CAVIN1"],
            "definition": (
                "CGL4 (CAVIN1) -- UNIQUE FEATURES: MYOPATHY + CARDIAC ARRHYTHMIA: "
                "WHY MUSCLE IN CAVIN1: "
                "  Caveolae are most abundant in skeletal and cardiac muscle (membrane repair + T-tubule organisation); "
                "  CAVIN1 is essential for muscle caveolae (not just adipose); "
                "  Loss of caveolae in muscle -> membrane fragility (like muscular dystrophies) -> myopathy; "
                "  Cardiomyocytes: caveolae regulate ion channel distribution (Na/K channels, L-type Ca); "
                "    Loss -> arrhythmogenic substrate; "
                "SKELETAL MUSCLE INVOLVEMENT: "
                "  Proximal muscle weakness (limb-girdle distribution); "
                "  Muscle wasting (on exam and MRI musculoskeletal); "
                "  CK markedly elevated (500-5000 IU/L) -- present in >80% of CAVIN1 patients; "
                "  Muscle biopsy: dystrophic changes (variation in fibre size, central nuclei, fibrosis); "
                "    + EM: caveolae absent; "
                "  Diaphragm: may be involved -> respiratory insufficiency; spirometry required; "
                "CARDIAC: "
                "  Conduction abnormalities: PR prolongation, bundle branch block, QT prolongation; "
                "  Arrhythmias: ventricular arrhythmia; sudden cardiac death reported; "
                "  Dilated cardiomyopathy component in some patients; "
                "  MANAGEMENT: annual ECG + Holter + echocardiogram; ICD if significant VT/VF; "
                "  Genetic cardiomyopathy team co-management; "
                "CK AS DIAGNOSTIC CLUE: "
                "  CGL + CK 500-5000 IU/L = CGL4 (CAVIN1) until proven otherwise; "
                "  Check CK in all CGL patients at diagnosis; "
                "  No other CGL type (AGPAT2, BSCL2, CAV1) has significantly elevated CK; "
                "TREATMENT: "
                "  Metabolic: metreleptin + low-fat diet + fibrates; "
                "  Cardiac: ECG monitoring; consider ICD prophylactically in those with significant arrhythmia; "
                "  Muscular: physiotherapy; respiratory support if diaphragm involved; "
                "  Avoid aggressive exercise in severe myopathy."
            ),
        },
        {
            "term": "FPLD2 (LMNA Arg482) -- Dunnigan Syndrome Puberty Onset, Cardiomyopathy, Laminopathy",
            "genes": ["LMNA"],
            "definition": (
                "FPLD2 / DUNNIGAN SYNDROME (LMNA Arg482) -- MOST COMMON FPLD: "
                "ARG482 HOTSPOT: "
                "  >90% of FPLD2 mutations: p.Arg482Trp, p.Arg482Gln, p.Arg482Leu; "
                "  Arg482 in immunoglobulin-fold domain of Lamin A/C; "
                "  Mutation changes surface charge -> altered binding to HP1alpha, BAF, emerin; "
                "  Alters LAD (lamin-associated domain) organisation -> adipogenic gene expression disrupted; "
                "  Adipogenesis proceeds at puberty but fat MAINTENANCE fails -> progressive fat loss; "
                "PUBERTY ONSET (KEY FEATURE): "
                "  Fat loss from extremities + gluteal BEGINS AT PUBERTY (not congenital); "
                "  Females: more severe loss; more pronounced metabolic syndrome; "
                "  Males: milder; may be subtle/missed; "
                "FAT REDISTRIBUTION: "
                "  Lost from: arms, legs, gluteal region; "
                "  Gained at: face (round face), neck (fat cushion), abdomen (prominent); "
                "  Neck fat: mimics Cushing syndrome (exclude with 24h UFC + overnight dexamethasone suppression); "
                "  DIFFERENTIAL: Cushing (hypercortisolism), HIV lipodystrophy (ARV use), multiple symmetric lipomatosis; "
                "METABOLIC: "
                "  Hypertriglyceridemia (200-800 mg/dL); "
                "  Low HDL; insulin resistance; DM (60%); hypertension; "
                "  Premature atherosclerosis; "
                "  Acanthosis nigricans (insulin resistance); PCOS in females; "
                "LAMINOPATHY CARDIAC COMPLICATIONS: "
                "  Dilated cardiomyopathy OR hypertrophic cardiomyopathy; "
                "  Conduction disease: AV block, LBBB, atrial fibrillation; "
                "  SUDDEN CARDIAC DEATH: LMNA = one of most common single-gene causes of sudden cardiac death; "
                "  Risk factors for SCD: LMNA + NSVT + LVEF <45% + male sex; "
                "  ICD indications: LVEF <45% + NSVT/syncope; "
                "  Cardiac surveillance: ECG + Holter annually; echo every 1-3 years; "
                "TREATMENT: "
                "  Metabolic: metreleptin; fibrates + statins; ACE-I/ARB; insulin/GLP-1 agonist; "
                "  TZDs: potentially beneficial (PPARG activation) but fluid retention risk; "
                "  Cardiac: evidence-based heart failure therapy (ACE-I/beta-blocker/MRA if DCM); ICD; "
                "  Genetic counselling: AD; 50% offspring risk; screen family."
            ),
        },
        {
            "term": "TZD Specific Treatment for FPLD3 (PPARG Haploinsufficiency) -- Mechanism and Evidence",
            "genes": ["PPARG"],
            "definition": (
                "THIAZOLIDINEDIONES (TZDs) -- SPECIFIC TREATMENT FOR FPLD3 (PPARG): "
                "WHY TZDs ARE SPECIFIC FOR FPLD3: "
                "  FPLD3 = PPARG haploinsufficiency -> 50% reduction in PPARG transcriptional activity; "
                "  TZDs = high-affinity synthetic PPARG ligands (agonists): "
                "    Pioglitazone (preferred): full PPARG agonist; "
                "    Rosiglitazone: full PPARG agonist (CV risk concerns -- restricted); "
                "  TZD activates remaining WT PPARG allele -> partial functional restoration; "
                "  Net effect: PPARG-driven adipogenic programme re-activated -> "
                "    improved peripheral fat storage -> ectopic fat redistribution -> reduced HTG + insulin resistance; "
                "MECHANISM OF BENEFIT: "
                "  Enhanced GLUT4 expression in adipocytes -> better glucose uptake; "
                "  Increased adiponectin secretion -> hepatic + peripheral insulin sensitisation; "
                "  Reduced ectopic fat (liver, muscle) -> improves hepatic insulin sensitivity; "
                "  Triglyceride lowering: LPL activation + adipocyte FFA uptake; "
                "CLINICAL OUTCOMES IN FPLD3: "
                "  TG reduction: 30-60% from baseline; "
                "  HbA1c: 1-2 point reduction; "
                "  Fat redistribution: some restoration of peripheral fat (slow; months to years); "
                "  Blood pressure: modest reduction; "
                "  Insulin dose reduction in DM; "
                "CAUTIONS: "
                "  Fluid retention: oedema + heart failure risk (avoid in NYHA III/IV); "
                "  Weight gain: increased adiposity (desired effect for lipodystrophy!); "
                "  Bone: fracture risk in women with long-term TZD use; "
                "  Pioglitazone preferred over rosiglitazone (CV safety); "
                "TZD USE IN OTHER FPLD TYPES: "
                "  FPLD2 (LMNA): potentially beneficial (PPARG-responsive pathway); "
                "    Limited data; fluid retention risk with cardiomyopathy -- caution; "
                "  FPLD4 (PLIN1): not specifically studied; "
                "  CGL types: TZDs less effective (PPARG-independent mechanisms); "
                "PPARG MASTER REGULATOR: "
                "  PPARG2 (adipose-specific): master transcription factor for adipogenesis; "
                "  TZD ligand -> PPARG-RXR heterodimer -> PPRE -> adipogenic gene expression; "
                "  This is why PPARG LOF -> lipodystrophy AND why TZD agonism can partially restore function."
            ),
        },
        {
            "term": "PLIN1 Unregulated Lipolysis -- Severe Hypertriglyceridemia and Pancreatitis Risk",
            "genes": ["PLIN1"],
            "definition": (
                "PLIN1 (PERILIPIN-1) -- LIPID DROPLET GATEKEEPER AND UNREGULATED LIPOLYSIS: "
                "NORMAL PLIN1 FUNCTION: "
                "  Perilipin-1 coats lipid droplet surface in adipocytes; "
                "  BASAL STATE: PLIN1 sequesters ABHD5 (CGI-58, ATGL co-activator) -> ATGL inactive -> "
                "    Basal lipolysis suppressed; fat stored; "
                "  STIMULATED (PKA activation -> catecholamines): "
                "    PKA phosphorylates PLIN1 -> conformation change; "
                "    ABHD5 released -> activates ATGL -> TAG -> DAG + FFA (first step lipolysis); "
                "    HSL recruited to phospho-PLIN1 -> DAG -> MAG + FFA (second step); "
                "    Regulated FFA release; "
                "PLIN1 LOF (FPLD4 MECHANISM): "
                "  Heterozygous frameshift -> haploinsufficiency -> partial PLIN1 reduction; "
                "  ABHD5 not sequestered -> constitutive ATGL activation -> UNREGULATED BASAL LIPOLYSIS; "
                "  Excess FFA release at all times -> "
                "    Hepatic: FFA -> VLDL overproduction -> severe hypertriglyceridemia; "
                "    Peripheral: FFA toxicity -> insulin resistance; "
                "    Muscle/liver: ectopic fat deposition; "
                "SEVERE HYPERTRIGLYCERIDEMIA: "
                "  400-2000+ mg/dL (vs normal <150 mg/dL); "
                "  Chylomicronaemia syndrome when TG >1000 mg/dL: "
                "    Eruptive xanthomata (orange-red skin eruptions); "
                "    Lipaemia retinalis (cream-white retinal vessels on fundoscopy); "
                "    Abdominal pain (even before acute pancreatitis); "
                "    TG >1000 mg/dL = impending pancreatitis; "
                "PANCREATITIS (MAJOR COMPLICATION): "
                "  Recurrent acute pancreatitis; "
                "  Mechanism: TG >1000 mg/dL -> pancreatic lipase -> FFA in pancreatic capillaries -> injury; "
                "  Complications: necrosis, pseudocyst, chronic pancreatitis, exocrine insufficiency; "
                "TG MANAGEMENT: "
                "  Fibrates (fenofibrate): first-line; reduce TG 30-50%; "
                "  Omega-3 fatty acids (4 g/day): additional 20-30% TG reduction; "
                "  Strict low-fat diet (<20 g fat/day in acute phase; <50 g/day long-term); "
                "  Volanesorsen (antisense ApoC3 inhibitor): 70-80% TG reduction; "
                "  Evinacumab (ANGPTL3 inhibitor): alternative for refractory HTG; "
                "  Acute pancreatitis: NPO + IV fluids; insulin infusion (lowers TG acutely); plasmapheresis if severe."
            ),
        },
        {
            "term": "8-Gene Hereditary Lipodystrophy Differential Guide -- CGL vs FPLD by Gene",
            "genes": ["AGPAT2", "BSCL2", "CAV1", "CAVIN1", "LMNA", "PPARG", "AKT2", "PLIN1"],
            "definition": (
                "8-GENE HEREDITARY LIPODYSTROPHY DIFFERENTIAL: "
                "BY ONSET: "
                "  CONGENITAL (birth): AGPAT2 (CGL1), BSCL2 (CGL2), CAV1 (CGL3), CAVIN1 (CGL4); "
                "  PUBERTY / ADULT: LMNA (FPLD2), PPARG (FPLD3), AKT2 (FPLD6), PLIN1 (FPLD4); "
                "BY LEPTIN: "
                "  Near-absent (<1 ng/mL): AGPAT2, BSCL2, CAV1, CAVIN1 (all CGL); "
                "  Low-normal (2-10 ng/mL): LMNA, PPARG, AKT2, PLIN1 (all FPLD); "
                "BY INHERITANCE: "
                "  AR: AGPAT2, BSCL2, CAV1, CAVIN1; "
                "  AD: LMNA, PPARG, AKT2, PLIN1; "
                "BY TRIGLYCERIDES: "
                "  Very high (>500 mg/dL): AGPAT2, BSCL2, CAV1, CAVIN1, PLIN1; "
                "  Moderate-high (200-800 mg/dL): LMNA, PPARG, AKT2; "
                "BY SPECIFIC DISTINGUISHING FEATURE: "
                "  Intellectual disability 30-60%: BSCL2 (CGL2) ONLY; "
                "  Caveolae absent on EM + PAH: CAV1 (CGL3); "
                "  Caveolae absent on EM + myopathy + elevated CK + arrhythmia: CAVIN1 (CGL4); "
                "  Puberty onset fat redistribution + cardiomyopathy + Arg482: LMNA (FPLD2); "
                "  TZD specific treatment (PPARG agonist replaces lost function): PPARG (FPLD3); "
                "  Severe insulin resistance disproportionate + incomplete penetrance + somatic GOF opposite: AKT2 (FPLD6); "
                "  Severe HTG + pancreatitis + unregulated lipolysis: PLIN1 (FPLD4); "
                "  Mechanical fat spared + NO intellectual disability: AGPAT2 (CGL1); "
                "BY TREATMENT: "
                "  Metreleptin SPECIFIC: AGPAT2 (CGL1), BSCL2 (CGL2), CAV1 (CGL3), CAVIN1 (CGL4); "
                "  TZD SPECIFIC: PPARG (FPLD3); "
                "  Fibrates + volanesorsen (HTG + pancreatitis): PLIN1 (FPLD4); "
                "  Annual cardiac surveillance mandatory: LMNA (arrhythmia + cardiomyopathy); "
                "    Also: CAVIN1 (arrhythmia); "
                "  PAH monitoring: CAV1 (echo annually); "
                "BY BIOMARKER: "
                "  Leptin <1 ng/mL: CGL (any); "
                "  CK >500 IU/L: CAVIN1 (CGL4); "
                "  Echo TR jet velocity elevated: CAV1 (CGL3); "
                "PRACTICAL FIRST STEP: "
                "  Congenital + leptin <1 + TG >500 -> CGL panel; then CK to distinguish CAVIN1 (CGL4); "
                "    EM for caveolae to distinguish CAV1/CAVIN1 from AGPAT2/BSCL2; "
                "    Intellectual disability -> BSCL2 first; mechanical fat spared -> AGPAT2 first; "
                "  Puberty onset + partial fat redistribution -> FPLD panel; "
                "    Arg482 sequencing first for LMNA; PPARG if metabolic syndrome prominent + TZD candidate; "
                "    Pancreatitis + severe HTG -> PLIN1."
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Lipodystrophy-Atlas",
        "count":       len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:1000])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    for g in bd["genes"]:
        print(
            f"  {g['gene']:8s}: n={g['n_patients']}, "
            f"mean_TG={g['mean_triglycerides_mg_dL']} mg/dL, "
            f"mean_leptin={g['mean_leptin_ng_mL']} ng/mL, "
            f"HOMA-IR={g['mean_homa_ir']}, "
            f"DM={g['dm_prevalence_pct']}%, "
            f"pancreatitis={g['pancreatitis_pct']}%, "
            f"metreleptin={g['metreleptin_treatment_pct']}%, "
            f"mean_age_dx={g['mean_age_dx']}"
        )
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
    for d in df["definitions"]:
        print(f"  - {d['term'][:80]}")
