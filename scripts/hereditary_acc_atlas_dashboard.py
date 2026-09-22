#!/usr/bin/env python3
"""Hereditary-Adrenocortical-Carcinoma-Predisposition-Atlas — Complete 8-Gene Reference
TP53   (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome;
         50-70% PEDIATRIC ACC — HIGHEST SINGLE-GENE RISK PATHOGNOMONIC;
         R337H Brazilian founder (1:300 Southern Brazil) — mandatory population screening;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+0) ·
NF1    (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis Type 1;
         Adrenocortical tumours 3-5% lifetime; adrenal pheochromocytoma 5%;
         Café-au-lait macules ≥6 + Lisch nodules PATHOGNOMONIC;
         seed SEED_BASE+1) ·
MEN1   (Menin; 610aa; 11q13.1; AD LOF;
         Multiple Endocrine Neoplasia Type 1;
         Adrenocortical tumours 30-75% (adenoma > carcinoma); malignant ~5%;
         Parathyroid 95% + pituitary 40% + pancreatic NET 70% TRIAD;
         seed SEED_BASE+2) ·
ARMC5  (Armadillo Repeat Containing 5; 1032aa; 16p11.2; AD LOF;
         Primary Bilateral Macronodular Adrenocortical Hyperplasia — PBMAH;
         MOST COMMON genetic cause of bilateral adrenal disease;
         Subclinical Cushing syndrome PATHOGNOMONIC bilateral nodules;
         seed SEED_BASE+3) ·
PRKAR1A (Protein Kinase cAMP-Dependent Regulatory Type 1 Alpha; 381aa; 17q24.2; AD LOF;
         Carney Complex;
         PPNAD — Primary Pigmented Nodular Adrenocortical Disease — bilateral micro-nodular;
         Paradoxical cortisol rise on low-dose dexamethasone PATHOGNOMONIC;
         Cardiac myxoma 30-40% — ANNUAL ECHOCARDIOGRAM MANDATORY;
         seed SEED_BASE+4) ·
CDKN1C (Cyclin-Dependent Kinase Inhibitor 1C; 316aa; 11p15.4; AD LOF/GOF;
         Beckwith-Wiedemann Syndrome (BWS);
         Pediatric ACC #2 predisposition after TP53; Wilms tumour 4-7%; hepatoblastoma;
         Macrosomia + Omphalocele + Macroglossia TRIAD PATHOGNOMONIC;
         seed SEED_BASE+5) ·
APC    (Adenomatous Polyposis Coli; 2843aa; 5q22.2; AD LOF;
         Familial Adenomatous Polyposis / Gardner Syndrome;
         Adrenocortical adenoma ~10% FAP patients; rarely malignant;
         CHRPE (Congenital Hypertrophy of Retinal Pigment Epithelium) PATHOGNOMONIC;
         seed SEED_BASE+6) ·
DICER1 (DICER1 Ribonuclease III; 1922aa; 14q32.13; AD LOF;
         DICER1 Syndrome;
         Adrenocortical nodules/tumours in paediatric series; hotspot RNase IIIb domain;
         PPB Type I PATHOGNOMONIC in infancy; SLCT ovary; ERMS cervix;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3174-3181)
"""
import random

SEED_BASE = 3174

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-Tetramer-"
            "Li-Fraumeni-Pediatric-ACC-50-70pct-HIGHEST-PATHOGNOMONIC-"
            "R337H-Brazilian-Founder-AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 17p13.1 TP53 encodes Tumour Protein p53 (p53): "
            "STRUCTURE: "
            "  Tetrameric transcription factor (4x43 kDa); N-terminal transactivation domain; "
            "  Proline-rich PXXP region; central DNA-binding domain (most variants here); "
            "  Tetramerisation domain; C-terminal regulatory domain; "
            "  LOF variants → impaired G1/S checkpoint → unrepaired DSBs proliferate → carcinogenesis; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  PEDIATRIC ACC: 50-70% of children <5yr with ACC carry germline TP53 — HIGHEST single-gene risk; "
            "  WBMRI Toronto Protocol — annual whole-body MRI: cornerstone surveillance; "
            "  CPC (Choroid Plexus Carcinoma) age <5 = PATHOGNOMONIC LFS — test TP53 immediately; "
            "  Annual breast MRI from age 20 (women); annual colonoscopy from 25; brain MRI annually; "
            "R337H BRAZILIAN FOUNDER: "
            "  TP53 p.Arg337His — prevalent in Southern Brazil (1 in 300 individuals); "
            "  Highest-frequency TP53 germline variant in any population; "
            "  ACC in children <5 yr in LFS families from this region = SCREEN ALL; "
            "  Population-level TP53 newborn screening piloted in Paraná state (Brazil); "
            "AVOID RADIATION ABSOLUTELY: "
            "  TP53 LOF → impaired G1/S checkpoint → ionising radiation → accelerated second primaries; "
            "  Sarcomas in RT field within 5-10 yr documented; "
            "  Adrenal surgery preferred over ablative radiation for ACC; "
            "  If RT unavoidable: lower dose, conformal; document decision; "
            "ADRENAL SPECIFICS: "
            "  ACC in LFS: predominantly pediatric (under 5); adult ACC with TP53 germline ~5%; "
            "  R337H carriers: ACC risk ~5% lifetime but variable by penetrance modifiers; "
            "  Adrenal ultrasound annually from birth in R337H carriers (Brazil protocol); "
            "  Complete surgical resection (R0) is the only curative option — adrenal laparoscopy CONTROVERSIAL in ACC"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo ~30%; heterozygous sufficient; biallelic = embryonic lethal)",
        "cancer_risk": "Pediatric ACC 50-70% (HIGHEST); Adrenal 5% adult LFS; Sarcoma 30%; Breast 30% (women); CRC 5%; CNS 10%",
        "pathognomonic": "Pediatric ACC under 5yr + TP53 germline = Li-Fraumeni PATHOGNOMONIC; R337H = Brazilian founder screen",
        "surveillance_key": "WBMRI Toronto annually; Annual adrenal USS from birth (R337H); AVOID RADIATION ABSOLUTELY; Annual brain MRI; Breast MRI from 20",
        "key_distinctions": [
            "PEDIATRIC-ACC-50-70PCT-HIGHEST-SINGLE-GENE",
            "R337H-BRAZILIAN-FOUNDER-1-IN-300",
            "AVOID-RADIATION-ABSOLUTELY",
            "WBMRI-TORONTO-ANNUALLY",
            "CPC-AGE-5-PATHOGNOMONIC-LFS",
            "ADRENAL-USS-ANNUALLY-FROM-BIRTH-R337H",
        ],
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RasGAP-"
            "Neurofibromatosis-Type1-Adrenocortical-Tumours-3-5pct-Pheo-5pct-"
            "Cafe-au-Lait-Lisch-PATHOGNOMONIC-OMIM-613113"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 17q11.2 NF1 encodes Neurofibromin (NF1): "
            "STRUCTURE: "
            "  319 kDa protein; central GRD (GTPase-activating Related Domain): accelerates RAS-GTP hydrolysis; "
            "  NF1 LOF → reduced RAS-GAP activity → elevated RAS-GTP → activated RAS/MAPK → tumour growth; "
            "  Largest known tumour suppressor gene (350 kb genomic span); high spontaneous mutation rate; "
            "NEUROFIBROMATOSIS TYPE 1: "
            "  NIH Diagnostic Criteria: ≥2 of: ≥6 café-au-lait macules (>5mm prepubertal/>15mm postpubertal); "
            "  ≥2 neurofibromas OR 1 plexiform neurofibroma; axillary/inguinal freckling; optic glioma; "
            "  ≥2 Lisch nodules (iris hamartomas); distinctive bone lesion (sphenoid dysplasia/tibial pseudoarthrosis); "
            "  First-degree relative with NF1; "
            "ADRENOCORTICAL TUMOURS IN NF1: "
            "  Adrenocortical adenomas: 3-5% of NF1 patients; typically non-functional; "
            "  Adrenocortical carcinoma: rare (<1%) but documented; "
            "  Pheochromocytoma (adrenal medulla): ~5% of NF1 — distinguish from adrenocortical; "
            "  Annual blood pressure monitoring; if hypertension → pheo biochemistry (24h urinary catecholamines); "
            "  CT/MRI adrenal incidentaloma: size >4 cm or growing → resect; "
            "NF1 ONCOLOGICAL BURDEN: "
            "  Malignant Peripheral Nerve Sheath Tumour (MPNST): 8-13% lifetime — dominant malignancy; "
            "  Optic glioma (pilocytic astrocytoma) in childhood; leukaemia (JMML); breast cancer 5x; "
            "  Adrenal pathology secondary to dominant NF1 risks but clinically significant; "
            "SURVEILLANCE: "
            "  Annual clinical examination; neuroimaging if new neurological symptoms; "
            "  Blood pressure annually; urine catecholamines if hypertension; "
            "  Adrenal imaging not routine unless symptoms or incidentaloma"
        ),
        "inheritance": "Autosomal Dominant LOF (50% de novo; high penetrance; highly variable expression)",
        "cancer_risk": "MPNST 8-13% lifetime; Adrenocortical tumour 3-5%; Pheo 5%; Optic glioma 15% (children); JMML; Breast 5x",
        "pathognomonic": "≥6 café-au-lait macules + Lisch nodules + axillary freckling = NF1 PATHOGNOMONIC (NIH criteria ≥2 features)",
        "surveillance_key": "Annual BP monitoring; urine catecholamines if hypertension; adrenal imaging for incidentaloma >4cm; MPNST surveillance",
        "key_distinctions": [
            "CAFE-AU-LAIT-LISCH-FRECKLING-PATHOGNOMONIC",
            "MPNST-8-13PCT-DOMINANT-MALIGNANCY",
            "ADRENAL-CORTEX-3-5PCT-ADENOMA",
            "PHEO-ADRENAL-MEDULLA-5PCT",
            "ANNUAL-BP-URINE-CATECHOLAMINES",
            "LARGEST-TUMOUR-SUPPRESSOR-GENE-350KB",
        ],
    },
    {
        "gene": "MEN1",
        "protein": (
            "MEN1 -- 11q13.1 Autosomal-Dominant-LOF -- 610aa -- "
            "Menin-68kDa-Histone-Methyltransferase-Scaffold-"
            "MEN1-Parathyroid-95pct-Pituitary-40pct-PancreaticNET-70pct-"
            "Adrenocortical-Adenoma-30-75pct-Carcinoma-Rare-OMIM-131100"
        ),
        "locus": "11q13.1",
        "protein_size": (
            "610 aa / 11q13.1 MEN1 encodes Menin (MEN1): "
            "STRUCTURE: "
            "  68 kDa nuclear scaffold protein; no intrinsic enzymatic activity; "
            "  Interacts with MLL histone H3K4 methyltransferases (MLL1/2-WBP complex); "
            "  Regulates CDK inhibitor expression (CDKN1B/p27, CDKN2C/p18) — MEN1 LOF → CDK inhibitor loss; "
            "  Winged-helix JunD inhibitor; scaffold for SWI/SNF chromatin remodelling; "
            "MULTIPLE ENDOCRINE NEOPLASIA TYPE 1 (MEN1 SYNDROME): "
            "  Classic Triad: Parathyroid 95% (EARLIEST; hypercalcemia) + Pituitary 40% + Pancreatic/duodenal NET 70%; "
            "  Most common hereditary endocrine tumour syndrome; "
            "  Germline MEN1 testing: any patient with ≥2 MEN1 tumours OR 1 tumour + FH MEN1; "
            "ADRENOCORTICAL TUMOURS IN MEN1: "
            "  Adrenocortical tumours: 30-75% of MEN1 carriers (autopsy series); "
            "  Predominantly non-functional adenomas; usually bilateral nodules; "
            "  Adrenocortical carcinoma in MEN1: ~5% of adrenal tumours — rare but documented; "
            "  Functional tumours: primary hyperaldosteronism (Conn adenoma), cortisol-secreting adenoma; "
            "  Annual adrenal imaging (CT/MRI) recommended in MEN1 surveillance; "
            "  Large >3 cm or growing lesion → resect (size criterion lower in MEN1 due to bilateral risk); "
            "MEDICAL TREATMENT MEN1-NET: "
            "  Pancreatic NET <2 cm: watch-and-wait; ≥2 cm: surgery; "
            "  Somatostatin analogues (octreotide, lanreotide) for functional NETs; "
            "  Everolimus (mTOR inhibitor) + sunitinib: for advanced pancreatic NET; "
            "  Parathyroid surgery: 3.5 gland resection + cryopreservation; "
            "  Proton pump inhibitor for Zollinger-Ellison gastrinoma"
        ),
        "inheritance": "Autosomal Dominant LOF (heterozygous; 10% de novo; high penetrance by age 50)",
        "cancer_risk": "Parathyroid 95%; Pancreatic/Duodenal NET 70%; Pituitary 40%; Adrenocortical 30-75% (adenoma); ACC <5%",
        "pathognomonic": "Multiglandular parathyroid hyperplasia + pancreatic NET + pituitary adenoma = MEN1 TRIAD PATHOGNOMONIC",
        "surveillance_key": "Annual Ca2+/PTH from age 8; annual adrenal CT/MRI; 3-yearly pituitary MRI; annual fasting gastrin/glucagon/PP",
        "key_distinctions": [
            "PARATHYROID-95PCT-EARLIEST-MEN1",
            "ADRENOCORTICAL-ADENOMA-30-75PCT",
            "ACC-MALIGNANT-RARE-5PCT-ADRENAL",
            "EVEROLIMUS-SUNITINIB-PANCREATIC-NET",
            "ANNUAL-ADRENAL-IMAGING-MEN1",
            "MULTIGLANDULAR-PARATHYROID-NOT-SINGLE-GLAND",
        ],
    },
    {
        "gene": "ARMC5",
        "protein": (
            "ARMC5 -- 16p11.2 Autosomal-Dominant-LOF -- 1032aa -- "
            "Armadillo-Repeat-ARM-WD40-Scaffold-"
            "PBMAH-Primary-Bilateral-Macronodular-Adrenocortical-Hyperplasia-"
            "MOST-COMMON-PBMAH-Gene-Subclinical-Cushing-PATHOGNOMONIC-OMIM-615954"
        ),
        "locus": "16p11.2",
        "protein_size": (
            "1032 aa / 16p11.2 ARMC5 encodes Armadillo Repeat Containing 5 (ARMC5): "
            "STRUCTURE: "
            "  ARM (Armadillo) repeat scaffold protein; mediates protein-protein interactions; "
            "  Regulates adrenocortical cell proliferation and apoptosis; "
            "  LOF → bilateral adrenocortical hyperplasia with macronodule formation; "
            "  Most frequently mutated gene in primary bilateral macronodular adrenocortical hyperplasia (PBMAH); "
            "PRIMARY BILATERAL MACRONODULAR ADRENOCORTICAL HYPERPLASIA (PBMAH): "
            "  MOST COMMON genetic cause of bilateral adrenal nodular disease; "
            "  ARMC5 pathogenic variants in ~25-50% of all PBMAH cases; "
            "  Both adrenal glands enlarged with multiple macronodules (>1 cm); "
            "  Typically subclinical Cushing syndrome: midnight salivary cortisol elevated; "
            "  1 mg overnight dexamethasone suppression test: suppression ABSENT; "
            "  PATHOGNOMONIC: bilateral macronodular adrenals + subclinical autonomous cortisol secretion; "
            "CLINICAL FEATURES: "
            "  Mean age at diagnosis 50-65 yr; slowly progressive autonomous cortisol secretion; "
            "  Subclinical Cushing: hypertension, impaired glucose tolerance, osteoporosis, dyslipidaemia; "
            "  Weight gain minimal (vs classic Cushing's); frank Cushing rare; "
            "  Meningioma risk elevated in ARMC5 carriers; "
            "MANAGEMENT: "
            "  Bilateral adrenalectomy: curative; requires lifelong steroid replacement (Addisonian); "
            "  Unilateral adrenalectomy: partial; may recur in contralateral adrenal; "
            "  Annual imaging if conservative management; annual biochemistry (midnight cortisol, 1mg DST); "
            "GENETICS: "
            "  Germline + somatic second hit (Knudson model); "
            "  Penetrance age-dependent; family cascade testing recommended; "
            "  ARMC5 variants account for >50% of familial PBMAH"
        ),
        "inheritance": "Autosomal Dominant LOF (germline + somatic second hit Knudson; age-dependent penetrance)",
        "cancer_risk": "PBMAH (bilateral macronodular hyperplasia) near 100% in carriers; ACC risk low; Meningioma risk elevated",
        "pathognomonic": "Bilateral macronodular adrenals (>1cm nodes both sides) + subclinical Cushing + ARMC5 variant = PBMAH PATHOGNOMONIC",
        "surveillance_key": "Annual midnight salivary cortisol + 1mg DST; Annual adrenal CT/MRI; BP/glucose/lipid annually; Meningioma MRI if symptoms",
        "key_distinctions": [
            "PBMAH-MOST-COMMON-GENETIC-CAUSE-BILATERAL",
            "SUBCLINICAL-CUSHING-BILATERAL-MACRONODULES-PATHOGNOMONIC",
            "ARMC5-25-50PCT-ALL-PBMAH",
            "BILATERAL-ADRENALECTOMY-CURATIVE-ADDISONIAN",
            "MENINGIOMA-RISK-ELEVATED",
            "KNUDSON-GERMLINE-PLUS-SOMATIC-SECOND-HIT",
        ],
    },
    {
        "gene": "PRKAR1A",
        "protein": (
            "PRKAR1A -- 17q24.2 Autosomal-Dominant-LOF -- 381aa -- "
            "PKA-R1alpha-43kDa-cAMP-Regulatory-Subunit-"
            "Carney-Complex-PPNAD-Bilateral-Micro-Nodular-"
            "Paradoxical-Cortisol-Dexamethasone-PATHOGNOMONIC-"
            "Cardiac-Myxoma-30pct-ANNUAL-ECHO-MANDATORY-Spotty-Pigmentation-PATHOGNOMONIC-OMIM-188830"
        ),
        "locus": "17q24.2",
        "protein_size": (
            "381 aa / 17q24.2 PRKAR1A encodes Protein Kinase cAMP-Dependent Regulatory Type I Alpha (PKA R1α): "
            "STRUCTURE: "
            "  43 kDa regulatory subunit of Protein Kinase A (PKA); inhibits PKA catalytic subunit; "
            "  PRKAR1A LOF → disinhibited PKA → constitutive cAMP/PKA activation → autonomous cortisol; "
            "  Two cAMP-binding domains (CBD-A, CBD-B); dimerisation domain; "
            "CARNEY COMPLEX (CNC): "
            "  PPNAD (Primary Pigmented Nodular Adrenocortical Disease): bilateral micro-nodular disease; "
            "  Multiple small (<1 cm) pigmented nodules — nodules NOT bilateral macronodules (cf. ARMC5/PBMAH); "
            "  PARADOXICAL CORTISOL RISE on Liddle test (low-dose dexamethasone) = PATHOGNOMONIC; "
            "  Normal pituitary MRI; ACTH undetectable; urinary 17-OHCS rises >50% on 2-day dex = DIAGNOSTIC; "
            "SPOTTY SKIN PIGMENTATION PATHOGNOMONIC: "
            "  Lentigines (perioral, periocular, genital mucosal, scleral) = PATHOGNOMONIC Carney Complex; "
            "  Blue nevi; myxomatous skin lesions; labial mucosal pigmentation; "
            "CARDIAC MYXOMA — LIFE-THREATENING: "
            "  30-40% of Carney Complex; may be bilateral (left AND right), valvular; "
            "  Embolic stroke risk if undetected; "
            "  ANNUAL ECHOCARDIOGRAM MANDATORY from diagnosis — cannot be omitted; "
            "  Surgical removal required; can recur in any chamber; "
            "OTHER CARNEY COMPLEX FEATURES: "
            "  Testicular large-cell calcifying Sertoli cell tumour (LCCSCT) 30-40% males; "
            "  Thyroid follicular adenoma 75%; GH-secreting pituitary adenoma 10%; osteochondromyxoma; "
            "  Psammomatous melanotic schwannoma; breast ductal adenoma; "
            "MANAGEMENT: "
            "  Bilateral adrenalectomy for Cushing in PPNAD; "
            "  Annual echocardiogram (cardiac myxoma); "
            "  Annual thyroid ultrasound; testicular ultrasound annually (males)"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo ~30%; high penetrance; heterozygous sufficient for Carney Complex)",
        "cancer_risk": "PPNAD (bilateral) near universal; Cardiac myxoma 30-40% (LIFE-THREATENING); Thyroid adenoma 75%; Pituitary GH-secreting 10%",
        "pathognomonic": "Spotty perioral/genital lentigines + PPNAD + cardiac myxoma = Carney Complex PATHOGNOMONIC; paradoxical Liddle test diagnostic",
        "surveillance_key": "ANNUAL ECHOCARDIOGRAM MANDATORY; Annual thyroid USS; Midnight cortisol/Liddle test; Annual testicular USS (males); Spotty pigmentation clinical exam",
        "key_distinctions": [
            "PARADOXICAL-CORTISOL-RISE-LIDDLE-PATHOGNOMONIC",
            "PPNAD-BILATERAL-MICRO-NODULAR-NOT-MACRO",
            "CARDIAC-MYXOMA-ANNUAL-ECHO-MANDATORY",
            "SPOTTY-PIGMENTATION-PERIORAL-PATHOGNOMONIC",
            "BILATERAL-ADRENALECTOMY-PPNAD",
            "LCCSCT-TESTICULAR-30-40PCT-MALES",
        ],
    },
    {
        "gene": "CDKN1C",
        "protein": (
            "CDKN1C -- 11p15.4 Autosomal-Dominant-LOF/GOF -- 316aa -- "
            "p57KIP2-36kDa-CDK-Inhibitor-Imprinted-"
            "Beckwith-Wiedemann-Syndrome-BWS-"
            "Pediatric-ACC-2nd-Most-Common-Wilms-4-7pct-Hepatoblastoma-"
            "Macrosomia-Omphalocele-Macroglossia-TRIAD-PATHOGNOMONIC-OMIM-130650"
        ),
        "locus": "11p15.4",
        "protein_size": (
            "316 aa / 11p15.4 CDKN1C encodes p57KIP2, cyclin-dependent kinase inhibitor 1C: "
            "STRUCTURE: "
            "  36 kDa CDK inhibitor; inhibits CDK2/cyclin E (G1-S transition); "
            "  Maternally expressed; paternally imprinted (imprinted gene — Beckwith-Wiedemann locus 11p15); "
            "  LOF on maternal allele → BWS (loss of CDK inhibition → overgrowth + tumour predisposition); "
            "  GOF variants on paternal allele → IMAGe syndrome (opposite: intrauterine growth restriction); "
            "BECKWITH-WIEDEMANN SYNDROME (BWS): "
            "  MOLECULAR CAUSES: IC2 (CDKN1C region) methylation loss, paternal UPD 11p15, IC1 gain of methylation, CDKN1C LOF variants; "
            "  TRIAD: Macrosomia + Omphalocele/Umbilical hernia + Macroglossia = PATHOGNOMONIC; "
            "  Other features: Hemihypertrophy (body asymmetry), neonatal hypoglycemia, ear pits/creases, Dandy-Walker; "
            "PAEDIATRIC TUMOUR RISK IN BWS: "
            "  Overall tumour risk 7-10% (HIGHEST RISK in first 7 years); "
            "  Wilms tumour: 4-7%; "
            "  Hepatoblastoma: 2-3%; "
            "  Adrenocortical carcinoma (ACC): 1-2% — 2nd most common ACC predisposition gene (after TP53); "
            "  Neuroblastoma; rhabdomyosarcoma (rare); "
            "  Risk stratified by molecular subtype: IC1 gain of methylation → HIGHEST Wilms risk; "
            "SURVEILLANCE (Tumour Watch): "
            "  3-monthly abdominal ultrasound from birth to age 7 (Wilms/hepatoblastoma); "
            "  AFP every 3 months (hepatoblastoma); "
            "  Adrenal imaging annually; "
            "  MRI brain/spine if neurological symptoms (neuroblastoma); "
            "  After age 7: reduce frequency; adrenal surveillance continues to age 18"
        ),
        "inheritance": "Autosomal Dominant LOF (maternal allele); imprinted gene — also caused by UPD11p15 and methylation defects",
        "cancer_risk": "Overall 7-10% <age 7; Wilms 4-7%; Hepatoblastoma 2-3%; ACC 1-2% (2nd after TP53); Neuroblastoma rare",
        "pathognomonic": "Macrosomia + Omphalocele + Macroglossia = BWS TRIAD PATHOGNOMONIC; ear creases + hemihypertrophy support",
        "surveillance_key": "3-monthly abdominal USS from birth to age 7; AFP 3-monthly; Annual adrenal imaging; brain MRI if neurological",
        "key_distinctions": [
            "PEDIATRIC-ACC-2ND-MOST-COMMON-AFTER-TP53",
            "BWS-TRIAD-MACROSOMIA-OMPHALOCELE-MACROGLOSSIA",
            "IMPRINTED-GENE-MATERNAL-ALLELE",
            "WILMS-4-7PCT-HIGHEST-RISK",
            "3-MONTHLY-ABDOMINAL-USS-BIRTH-TO-7",
            "IC1-GAIN-METHYLATION-HIGHEST-WILMS-RISK",
        ],
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "APC-310kDa-WNT-Gatekeeper-beta-catenin-Destruction-Complex-"
            "FAP-Gardner-Adrenocortical-Adenoma-10pct-"
            "CHRPE-PATHOGNOMONIC-Prophylactic-Colectomy-OMIM-175100"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 5q22.2 APC encodes Adenomatous Polyposis Coli (APC): "
            "STRUCTURE: "
            "  310 kDa scaffolding protein; β-catenin destruction complex (APC + Axin + GSK3β + CK1); "
            "  APC LOF → β-catenin escapes destruction → nuclear WNT signalling → proliferation; "
            "  Mutation cluster region (MCR): codons 1250-1464 — dense pathogenic variant hotspot; "
            "  CHRPE (Congenital Hypertrophy Retinal Pigment Epithelium): linked to variants 5-prime to codon 1444; "
            "FAMILIAL ADENOMATOUS POLYPOSIS (FAP) / GARDNER SYNDROME: "
            "  FAP: >100 colorectal polyps from age 10-30; 100% CRC risk by age 40 untreated; "
            "  Gardner Syndrome: FAP + desmoid tumours + osteomas + epidermoid cysts; "
            "  Prophylactic colectomy mandatory (before polyps become malignant); "
            "  Upper GI surveillance: duodenal/periampullary polyps (Spigelman classification); "
            "CHRPE — PATHOGNOMONIC: "
            "  Bilateral multifocal CHRPE (≥4 lesions) = APC germline PATHOGNOMONIC; "
            "  Present in >90% FAP; congenital (present from birth); asymptomatic; "
            "  Ophthalmoscopy detects before polyps develop — useful pre-symptomatic marker; "
            "ADRENOCORTICAL TUMOURS IN FAP: "
            "  Adrenocortical adenoma: ~10% of FAP patients on cross-sectional imaging; "
            "  Typically non-functional; rarely adrenocortical carcinoma; "
            "  Annual CT/MRI (for desmoid monitoring) coincidentally surveys adrenals; "
            "  Adrenal incidentaloma >4 cm or growing → functional work-up + resection; "
            "OTHER FAP CANCERS: "
            "  Small bowel / duodenal / periampullary: 5-8% lifetime; "
            "  Thyroid (cribriform-morular PTC — PATHOGNOMONIC FAP in women); "
            "  Hepatoblastoma in children of FAP families (AFP surveillance); "
            "  Medulloblastoma (Turcot variant) — DESMOPLASTIC SHH subtype; "
            "  Gastric: fundic gland polyps (low malignant risk)"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo ~25%; heterozygous; biallelic = embryonic lethal; MUTYH-associated polyposis = AR DDx)",
        "cancer_risk": "CRC 100% untreated; Duodenal/periampullary 5-8%; Thyroid cribriform PTC; Adrenocortical adenoma 10% (carcinoma rare)",
        "pathognomonic": "≥4 CHRPE lesions bilateral = APC PATHOGNOMONIC; ≥100 colorectal polyps = FAP; cribriform-morular PTC in women = FAP",
        "surveillance_key": "Annual colonoscopy from age 10-12; prophylactic colectomy before malignancy; annual upper GI from 25; CHRPE fundoscopy",
        "key_distinctions": [
            "CHRPE-BILATERAL-MULTIFOCAL-PATHOGNOMONIC",
            "100PCT-CRC-UNTREATED-PROPHYLACTIC-COLECTOMY",
            "ADRENOCORTICAL-ADENOMA-10PCT-FAP",
            "CRIBRIFORM-MORULAR-PTC-PATHOGNOMONIC-FAP",
            "DESMOID-TUMOURS-GARDNER-MESENTERIC",
            "MUTYH-AR-POLYPOSIS-DDX",
        ],
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-218kDa-RNase-III-miRNA-Processor-"
            "DICER1-Syndrome-Adrenocortical-Nodule-Cortical-Neoplasm-"
            "PPB-Type-I-PATHOGNOMONIC-Infancy-SLCT-ERMS-Cervix-MNG-75pct-"
            "RNase-IIIb-Hotspot-R950-E1813-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 14q32.13 DICER1 encodes DICER1, an endoribonuclease (RNase III): "
            "STRUCTURE: "
            "  218 kDa enzyme; processes pre-miRNA → mature miRNA; "
            "  Two RNase III domains: RNase IIIa and RNase IIIb; PAZ and helicase domains; "
            "  RNase IIIb domain hotspot variants (R950, R944, E1813, D1810) — gain-of-function (GOF) component; "
            "  Germline LOF + somatic RNase IIIb hotspot = biallelic mechanism unique to DICER1; "
            "DICER1 SYNDROME (DICER1-PLEUROPULMONARY BLASTOMA FAMILIAL TUMOUR & DYSPLASIA SYNDROME): "
            "  PPB Type I: PATHOGNOMONIC in infancy (<2yr); cystic lung lesion; chest CT 3-monthly first 3yr; "
            "  If PPB Type I untreated → Type II (cystic-solid) → Type III (solid) — progression preventable; "
            "  Embryonal rhabdomyosarcoma (ERMS) of cervix: premenopausal women — PATHOGNOMONIC DICER1; "
            "  Sertoli-Leydig cell tumour (SLCT) ovary: young women PATHOGNOMONIC; "
            "ADRENOCORTICAL TUMOURS IN DICER1 SYNDROME: "
            "  Adrenocortical nodules documented in paediatric and young adult DICER1 carriers; "
            "  Adrenocortical carcinoma: rare but documented in DICER1 syndrome (hotspot mutations confirmed); "
            "  Annual adrenal ultrasound in paediatric carriers recommended by DICER1 consortium guidelines; "
            "MULTINODULAR GOITRE (MNG): "
            "  75% of female DICER1 carriers develop MNG by age 30-40; "
            "  Annual thyroid ultrasound from age 8; "
            "  Differentiated thyroid cancer: ~1% — low risk but requires surveillance; "
            "OTHER DICER1 SYNDROME TUMOURS: "
            "  Ciliary body medulloepithelioma (eye); pituitary blastoma; pinealoblastoma; "
            "  Nasal chondromesenchymal hamartoma; Wilms tumour (rare); "
            "CASCADE TESTING: "
            "  Family cascade: siblings and offspring of DICER1 variant carriers; "
            "  Chest CT: 3-monthly in first 3 yr if PPB concern; "
            "  Annual pelvic USS from puberty; annual thyroid USS from age 8"
        ),
        "inheritance": "Autosomal Dominant LOF (germline) + somatic RNase IIIb hotspot (biallelic mechanism); de novo ~20%; 70% familial",
        "cancer_risk": "PPB ~5%; SLCT ovary 3-5%; ERMS cervix 2%; MNG 75% females; Adrenocortical nodule/tumour; Thyroid 1%",
        "pathognomonic": "PPB Type I in infant = DICER1 germline PATHOGNOMONIC; SLCT ovary young woman = DICER1; ERMS cervix = DICER1",
        "surveillance_key": "Chest CT 3-monthly first 3yr (PPB); Annual thyroid USS from age 8; Annual pelvic USS from puberty; Annual adrenal USS",
        "key_distinctions": [
            "PPB-TYPE-I-INFANT-PATHOGNOMONIC",
            "SLCT-OVARY-YOUNG-WOMAN-PATHOGNOMONIC",
            "ERMS-CERVIX-PATHOGNOMONIC",
            "MNG-75PCT-FEMALE-CARRIERS",
            "RNASE-IIIB-HOTSPOT-R950-E1813-SOMATIC",
            "ADRENOCORTICAL-NODULE-ANNUAL-USS-PAEDIATRIC",
        ],
    },
]


def _make_patients(gene_info, n=40):
    rng = random.Random(gene_info["seed"])
    gene = gene_info["gene"]

    base_age_map = {
        "TP53":    (4,  12),   # predominantly pediatric ACC
        "NF1":     (35, 55),
        "MEN1":    (30, 55),
        "ARMC5":   (45, 65),
        "PRKAR1A": (20, 45),
        "CDKN1C":  (2,  8),   # Beckwith-Wiedemann — neonatal/early childhood
        "APC":     (35, 60),
        "DICER1":  (5,  30),
    }
    lo, hi = base_age_map.get(gene, (30, 60))

    acc_pct_map = {
        "TP53":    (55, 75),
        "NF1":     (10, 20),
        "MEN1":    (5,  12),
        "ARMC5":   (2,  8),
        "PRKAR1A": (3,  8),
        "CDKN1C":  (15, 25),
        "APC":     (3,  8),
        "DICER1":  (4,  10),
    }
    acc_lo, acc_hi = acc_pct_map.get(gene, (5, 15))

    pts = []
    for i in range(n):
        age = rng.randint(lo, hi)
        has_acc = rng.random() < rng.uniform(acc_lo / 100, acc_hi / 100)
        cortisol_excess = gene in ("ARMC5", "PRKAR1A") or (gene == "TP53" and has_acc)
        bilateral = gene in ("ARMC5", "PRKAR1A")
        pts.append({
            "id": f"{gene}-{SEED_BASE + list(g['gene'] for g in ATLAS_GENES).index(gene)}-{i+1:03d}",
            "gene": gene,
            "age_onset": age,
            "has_acc": has_acc,
            "cortisol_excess": cortisol_excess,
            "bilateral_adrenal": bilateral,
            "functional_adrenal": cortisol_excess or (gene == "MEN1" and rng.random() < 0.15),
        })
    return pts


def generate_overview():
    all_patients = []
    for idx, g in enumerate(ATLAS_GENES):
        g["seed"] = SEED_BASE + idx
        all_patients.extend(_make_patients(g))

    total = len(all_patients)
    acc_n = sum(1 for p in all_patients if p["has_acc"])
    cortisol_n = sum(1 for p in all_patients if p["cortisol_excess"])
    bilateral_n = sum(1 for p in all_patients if p["bilateral_adrenal"])
    functional_n = sum(1 for p in all_patients if p["functional_adrenal"])
    mean_age = round(sum(p["age_onset"] for p in all_patients) / total, 1)

    gene_summary = []
    for g in ATLAS_GENES:
        pts = [p for p in all_patients if p["gene"] == g["gene"]]
        acc_count = sum(1 for p in pts if p["has_acc"])
        gene_summary.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "acc_n":            acc_count,
            "acc_pct":          round(acc_count / len(pts) * 100, 1),
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "cortisol_pct":     round(sum(1 for p in pts if p["cortisol_excess"]) / len(pts) * 100, 1),
            "bilateral_pct":    round(sum(1 for p in pts if p["bilateral_adrenal"]) / len(pts) * 100, 1),
            "inheritance":      g["inheritance"].split("(")[0].strip(),
            "key_distinctions": g["key_distinctions"],
        })

    return {
        "atlas":           "Hereditary-Adrenocortical-Carcinoma-Predisposition-Atlas",
        "seed_range":      f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":  total,
        "genes_n":         len(ATLAS_GENES),
        "acc_total_n":     acc_n,
        "acc_total_pct":   round(acc_n / total * 100, 1),
        "mean_age_onset":  mean_age,
        "cortisol_excess_n":     cortisol_n,
        "cortisol_excess_pct":   round(cortisol_n / total * 100, 1),
        "bilateral_adrenal_n":   bilateral_n,
        "bilateral_adrenal_pct": round(bilateral_n / total * 100, 1),
        "functional_adrenal_n":  functional_n,
        "functional_adrenal_pct": round(functional_n / total * 100, 1),
        "gene_summary":    gene_summary,
        "genes_detail": [
            {
                "gene":             g["gene"],
                "protein":          g["protein"],
                "locus":            g["locus"],
                "protein_size":     g["protein_size"],
                "inheritance":      g["inheritance"],
                "cancer_risk":      g["cancer_risk"],
                "pathognomonic":    g["pathognomonic"],
                "surveillance_key": g["surveillance_key"],
                "key_distinctions": g["key_distinctions"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_breakdown():
    rows = []
    for idx, g in enumerate(ATLAS_GENES):
        g["seed"] = SEED_BASE + idx
        pts = _make_patients(g)
        acc_n = sum(1 for p in pts if p["has_acc"])
        bilateral_n = sum(1 for p in pts if p["bilateral_adrenal"])
        cortisol_n = sum(1 for p in pts if p["cortisol_excess"])
        rows.append({
            "gene":               g["gene"],
            "locus":              g["locus"],
            "n":                  len(pts),
            "acc_n":              acc_n,
            "acc_pct":            round(acc_n / len(pts) * 100, 1),
            "bilateral_n":        bilateral_n,
            "bilateral_pct":      round(bilateral_n / len(pts) * 100, 1),
            "cortisol_n":         cortisol_n,
            "cortisol_pct":       round(cortisol_n / len(pts) * 100, 1),
            "mean_age_onset":     round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "pathognomonic":      g["pathognomonic"],
            "surveillance_key":   g["surveillance_key"],
            "inheritance":        g["inheritance"].split("(")[0].strip(),
            "key_distinctions":   g["key_distinctions"],
        })
    return {
        "atlas":     "Hereditary-Adrenocortical-Carcinoma-Predisposition-Atlas",
        "breakdown": rows,
    }


def generate_definitions():
    defs = [
        {
            "term": "TP53 / LFS / PEDIATRIC ACC 50-70% / R337H BRAZILIAN FOUNDER",
            "definition": (
                "TP53 (Tumour Protein p53) — 393aa / 43 kDa / 17p13.1 / AD LOF\n"
                "Li-Fraumeni Syndrome — #1 genetic cause of pediatric ACC.\n\n"
                "PEDIATRIC ACC (HIGHEST SINGLE-GENE RISK):\n"
                "  50-70% of children <5yr with ACC carry germline TP53.\n"
                "  ACC presenting in child <5yr = TP53 germline testing MANDATORY.\n"
                "  Complete R0 surgical resection is the only curative option.\n\n"
                "R337H BRAZILIAN FOUNDER:\n"
                "  1 in 300 individuals in Southern Brazil (Paraná state) carry R337H.\n"
                "  Highest-frequency germline TP53 variant in any population worldwide.\n"
                "  ACC risk in R337H: ~5% lifetime (lower penetrance than classic LFS).\n"
                "  Population-level newborn TP53 screening piloted in Brazil.\n\n"
                "AVOID RADIATION ABSOLUTELY:\n"
                "  Impaired G1/S checkpoint → radiation-induced second primaries within field.\n"
                "  Adrenal surgery always preferred over ablative RT for ACC in LFS.\n\n"
                "WBMRI TORONTO PROTOCOL:\n"
                "  Annual whole-body MRI: detects ACC, sarcoma, CNS tumours pre-symptomatic.\n"
                "  Annual adrenal ultrasound from birth in R337H carriers (Brazil protocol).\n"
                "  Annual breast MRI from age 20 (women); annual colonoscopy from 25."
            ),
        },
        {
            "term": "NF1 / NEUROFIBROMATOSIS / ADRENOCORTICAL 3-5% / CAFÉ-AU-LAIT PATHOGNOMONIC",
            "definition": (
                "NF1 (Neurofibromin 1) — 2839aa / 319 kDa / 17q11.2 / AD LOF\n"
                "Neurofibromatosis Type 1 — adrenocortical tumours 3-5%.\n\n"
                "NIH DIAGNOSTIC CRITERIA (≥2 features):\n"
                "  ≥6 café-au-lait macules (>5mm prepubertal; >15mm postpubertal).\n"
                "  ≥2 neurofibromas OR 1 plexiform neurofibroma.\n"
                "  Axillary/inguinal freckling (Crowe sign).\n"
                "  ≥2 Lisch nodules (iris hamartomas).\n"
                "  Optic glioma; bone dysplasia; FDR with NF1.\n\n"
                "ADRENAL PATHOLOGY:\n"
                "  Adrenocortical adenoma: 3-5% (usually non-functional, incidental).\n"
                "  Pheochromocytoma (adrenal medulla): 5% — DISTINGUISH from cortical.\n"
                "  Annual BP; urine metanephrines/catecholamines if hypertensive.\n\n"
                "DOMINANT MALIGNANCY — MPNST:\n"
                "  Malignant Peripheral Nerve Sheath Tumour 8-13% lifetime.\n"
                "  Arising in plexiform neurofibroma; rapid growth = URGENT resection."
            ),
        },
        {
            "term": "MEN1 / MULTIPLE ENDOCRINE NEOPLASIA / ADRENAL ADENOMA 30-75% / ANNUAL IMAGING",
            "definition": (
                "MEN1 (Menin) — 610aa / 68 kDa / 11q13.1 / AD LOF\n"
                "MEN1 Syndrome — adrenocortical tumours 30-75%.\n\n"
                "MEN1 TRIAD:\n"
                "  Parathyroid 95% (EARLIEST — hypercalcaemia).\n"
                "  Pancreatic/duodenal NET 70% (gastrinoma → ZES; insulinoma; VIPoma).\n"
                "  Pituitary 40% (prolactinoma most common).\n\n"
                "ADRENAL IN MEN1:\n"
                "  30-75% carry adrenocortical tumours (mostly adenomas, autopsy series).\n"
                "  Usually bilateral nodular; non-functional or mildly autonomous.\n"
                "  ACC: rare (~5% of MEN1 adrenal lesions) — but clinically significant.\n"
                "  Annual CT/MRI adrenal as part of MEN1 surveillance.\n"
                "  >3 cm or growing → resect (lower size threshold bilateral).\n\n"
                "TREATMENT MEN1:\n"
                "  Pancreatic NET ≥2 cm → surgery; somatostatin analogues for functional.\n"
                "  Everolimus + sunitinib for advanced pNET.\n"
                "  3.5-gland parathyroidectomy + cryopreservation."
            ),
        },
        {
            "term": "ARMC5 / PBMAH / SUBCLINICAL CUSHING BILATERAL PATHOGNOMONIC / BILATERAL ADRENALECTOMY",
            "definition": (
                "ARMC5 (Armadillo Repeat Containing 5) — 1032aa / 16p11.2 / AD LOF\n"
                "PBMAH — most common genetic bilateral adrenal disease.\n\n"
                "PBMAH DIAGNOSTIC FEATURES:\n"
                "  Both adrenal glands enlarged with multiple macronodules (>1 cm).\n"
                "  Subclinical autonomous cortisol secretion: 1mg DST non-suppression.\n"
                "  Midnight salivary cortisol elevated; urinary cortisol often normal initially.\n"
                "  PATHOGNOMONIC: bilateral macronodular adrenals + autonomous cortisol.\n\n"
                "ARMC5 GENETICS:\n"
                "  25-50% of all PBMAH; germline + somatic second hit (Knudson model).\n"
                "  Family cascade testing — siblings at 50% risk.\n"
                "  Meningioma risk elevated in ARMC5 carriers.\n\n"
                "MANAGEMENT:\n"
                "  Bilateral adrenalectomy: curative; LIFELONG steroid replacement (Addison).\n"
                "  Unilateral adrenalectomy: partial control; monitor contralateral.\n"
                "  Annual biochemistry (1mg DST, midnight cortisol, BP, glucose, lipids)."
            ),
        },
        {
            "term": "PRKAR1A / CARNEY COMPLEX / PPNAD / PARADOXICAL LIDDLE PATHOGNOMONIC / CARDIAC MYXOMA ANNUAL ECHO",
            "definition": (
                "PRKAR1A (PKA Regulatory Subunit R1α) — 381aa / 43 kDa / 17q24.2 / AD LOF\n"
                "Carney Complex — PPNAD + cardiac myxoma + spotty pigmentation.\n\n"
                "PPNAD — PARADOXICAL LIDDLE TEST PATHOGNOMONIC:\n"
                "  Urinary 17-OHCS RISES >50% on 2-day low-dose dexamethasone = DIAGNOSTIC.\n"
                "  Normal pituitary MRI; ACTH undetectable; small bilateral micronodules.\n"
                "  Contrast: ARMC5/PBMAH = macronodules; PRKAR1A/PPNAD = micronodules.\n\n"
                "SPOTTY PIGMENTATION:\n"
                "  Perioral, periocular, genital mucosal lentigines = Carney Complex PATHOGNOMONIC.\n"
                "  Any patient with adrenal Cushing + lentigines → PRKAR1A testing.\n\n"
                "CARDIAC MYXOMA — LIFE-THREATENING:\n"
                "  30-40% carriers; can be bilateral (any chamber) + valvular.\n"
                "  Embolic stroke if undetected → ANNUAL ECHOCARDIOGRAM MANDATORY.\n"
                "  Surgical removal required; recurrence in any chamber.\n\n"
                "OTHER CARNEY FEATURES:\n"
                "  LCCSCT testicular 30-40% males; thyroid adenoma 75%; pituitary GH 10%.\n"
                "  Bilateral adrenalectomy for Cushing in PPNAD."
            ),
        },
        {
            "term": "CDKN1C / BECKWITH-WIEDEMANN / PEDIATRIC ACC / MACROSOMIA-OMPHALOCELE-MACROGLOSSIA PATHOGNOMONIC",
            "definition": (
                "CDKN1C (p57KIP2) — 316aa / 36 kDa / 11p15.4 / AD LOF (imprinted)\n"
                "Beckwith-Wiedemann Syndrome — pediatric ACC 2nd most common predisposition.\n\n"
                "BWS TRIAD PATHOGNOMONIC:\n"
                "  Macrosomia + Omphalocele/umbilical hernia + Macroglossia = BWS.\n"
                "  Ear creases/pits, hemihypertrophy, neonatal hypoglycaemia support diagnosis.\n\n"
                "TUMOUR SURVEILLANCE:\n"
                "  3-monthly abdominal USS birth to age 7 (Wilms, hepatoblastoma).\n"
                "  AFP 3-monthly to age 4 (hepatoblastoma).\n"
                "  Annual adrenal imaging (ACC 1-2%).\n"
                "  Overall 7-10% tumour risk in first 7 years.\n\n"
                "MOLECULAR SUBTYPE AND WILMS RISK:\n"
                "  IC1 gain of methylation → HIGHEST Wilms tumour risk (~25%).\n"
                "  IC2 loss of methylation (most common BWS) → lower Wilms.\n"
                "  CDKN1C LOF → intermediate risk.\n\n"
                "IMPRINTED GENE:\n"
                "  Maternal allele expressed; paternal imprinted.\n"
                "  LOF on maternal allele → BWS; GOF → IMAGe syndrome (opposite phenotype)."
            ),
        },
        {
            "term": "APC / FAP / ADRENOCORTICAL ADENOMA 10% / CHRPE PATHOGNOMONIC / PROPHYLACTIC COLECTOMY",
            "definition": (
                "APC (Adenomatous Polyposis Coli) — 2843aa / 310 kDa / 5q22.2 / AD LOF\n"
                "FAP/Gardner Syndrome — adrenocortical adenoma ~10%; CHRPE PATHOGNOMONIC.\n\n"
                "CHRPE — CONGENITAL HYPERTROPHY RETINAL PIGMENT EPITHELIUM:\n"
                "  Bilateral multifocal (≥4) CHRPE = APC PATHOGNOMONIC.\n"
                "  Congenital; asymptomatic; fundoscopy identifies before polyps develop.\n"
                "  Present in >90% FAP; correlates with 5-prime variants (before codon 1444).\n\n"
                "ADRENAL IN FAP:\n"
                "  ~10% of FAP patients on imaging: adrenocortical adenoma (usually incidental).\n"
                "  ACC rare in FAP — but standard incidentaloma criteria apply (>4 cm → evaluate).\n"
                "  Surveillance CT (desmoids, upper GI) coincidentally surveys adrenals.\n\n"
                "FAP COLORECTAL:\n"
                "  >100 polyps from age 10-30; 100% CRC risk untreated.\n"
                "  Prophylactic colectomy mandatory before malignancy.\n"
                "  NSAID sulindac/celecoxib: reduces polyp burden (NOT curative).\n\n"
                "GARDNER SYNDROME:\n"
                "  FAP + desmoid tumours (mesenteric; can be life-threatening) + osteomas + epidermoid cysts.\n"
                "  Desmoid risk: variants codon 1310-2011; sulindac + imatinib for desmoids."
            ),
        },
        {
            "term": "DICER1 / DICER1 SYNDROME / ADRENOCORTICAL NODULE / PPB TYPE I PATHOGNOMONIC",
            "definition": (
                "DICER1 — 1922aa / 218 kDa / 14q32.13 / AD LOF\n"
                "DICER1 Syndrome — pleuropulmonary blastoma + adrenocortical nodule.\n\n"
                "PPB TYPE I — PATHOGNOMONIC:\n"
                "  Cystic lung lesion in infant <2yr = PPB Type I = DICER1 germline PATHOGNOMONIC.\n"
                "  Chest CT 3-monthly in first 3yr if family history or known carrier.\n"
                "  PPB I → II (cystic-solid) → III (solid): progression = worse prognosis.\n\n"
                "ADRENOCORTICAL NODULE IN DICER1 SYNDROME:\n"
                "  Adrenocortical nodules and tumours documented in paediatric DICER1 series.\n"
                "  Annual adrenal ultrasound from diagnosis in paediatric carriers (DICER1 consortium).\n"
                "  Hotspot variants R950, E1813 in RNase IIIb domain (somatic second hit).\n\n"
                "DICER1 SYNDROME SPECTRUM:\n"
                "  SLCT (ovary): young women — PATHOGNOMONIC.\n"
                "  ERMS cervix: premenopausal — PATHOGNOMONIC.\n"
                "  MNG: 75% female carriers (annual thyroid USS from age 8).\n"
                "  Ciliary body medulloepithelioma; pituitary blastoma; pinealoblastoma.\n\n"
                "GENETICS:\n"
                "  Germline LOF + somatic RNase IIIb hotspot = biallelic mechanism.\n"
                "  70% familial; 20% de novo.\n"
                "  Family cascade: screen siblings + offspring."
            ),
        },
        {
            "term": "CASCADE TESTING — Hereditary Adrenocortical Carcinoma",
            "definition": (
                "CASCADE TESTING PRIORITIES for Hereditary ACC Predisposition:\n\n"
                "TIER 1 — HIGHEST YIELD (definitive ACC predisposition syndromes):\n"
                "  TP53 LFS: ALL first-degree relatives; R337H population screen in Southern Brazil.\n"
                "    → Adrenal USS annually from birth; WBMRI Toronto annually; AVOID RADIATION.\n"
                "  CDKN1C BWS: siblings/offspring; 3-monthly abdominal USS birth-age 7.\n"
                "    → ACC 1-2%; Wilms 4-7%; Hepatoblastoma 2-3%; AFP 3-monthly.\n\n"
                "TIER 2 — MODERATE YIELD (bilateral adrenal disease syndromes):\n"
                "  ARMC5 PBMAH: all FDRs; annual 1mg DST + adrenal imaging.\n"
                "    → Bilateral adrenalectomy if progressive Cushing; lifelong steroid replacement.\n"
                "  PRKAR1A Carney: all FDRs; ANNUAL ECHOCARDIOGRAM MANDATORY.\n"
                "    → Paradoxical Liddle test; bilateral adrenalectomy for PPNAD Cushing.\n\n"
                "TIER 3 — SYNDROME-SPECIFIC (adrenal secondary to dominant other risk):\n"
                "  MEN1: adrenal imaging annual; focus on parathyroid/pNET/pituitary primary.\n"
                "  NF1: annual BP; urine catecholamines if hypertension; MPNST primary concern.\n"
                "  APC FAP: adrenal on CT done for desmoids; prophylactic colectomy primary.\n"
                "  DICER1: annual adrenal USS (paediatric); PPB chest CT primary; SLCT/ERMS.\n\n"
                "PATHOGNOMONIC POINTERS:\n"
                "  Pediatric ACC <5yr → TP53 germline immediately (regardless of family history).\n"
                "  Bilateral macronodular adrenals + subclinical Cushing → ARMC5 first.\n"
                "  Lentigines + Cushing + cardiac myxoma → PRKAR1A Carney Complex.\n"
                "  ≥6 café-au-lait + neurofibromas → NF1 (pheo vs cortical distinguish).\n"
                "  PPB infant lung + adrenal nodule → DICER1.\n"
                "  Macro-omphalocele-macroglossia neonatal → BWS CDKN1C/IC testing.\n"
            ),
        },
    ]

    return {
        "atlas":       "Hereditary-Adrenocortical-Carcinoma-Predisposition-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": defs,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("genes_detail",)}, indent=2))
    print("\n=== BREAKDOWN summary ===")
    br = generate_breakdown()
    for row in br["breakdown"]:
        print(f"  {row['gene']:8s} n={row['n']} mean_age={row['mean_age_onset']} "
              f"acc_n={row['acc_n']} ({row['acc_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term']}")
