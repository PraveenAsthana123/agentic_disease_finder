#!/usr/bin/env python3
"""Hereditary-Thyroid-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
RET     (REarranged during Transfection; 1114aa; 10q11.21; AD GOF;
         MEN2A / FMTC / MEN2B;
         medullary thyroid carcinoma 95% penetrance;
         C634R hotspot MEN2A PATHOGNOMONIC; M918T hotspot MEN2B PATHOGNOMONIC;
         vandetanib / cabozantinib FDA approved MTC;
         prophylactic thyroidectomy age 0-6 months MEN2B;
         seed SEED_BASE+0) .
PTEN    (Phosphatase and tensin homologue; 403aa; 10q23.31; AD LOF;
         Cowden syndrome / PHTS;
         follicular and papillary thyroid cancer 35-67%;
         macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC;
         everolimus mTOR inhibitor;
         seed SEED_BASE+1) .
APC     (Adenomatous polyposis coli; 2843aa; 5q22.2; AD LOF;
         Familial adenomatous polyposis (FAP) / Gardner;
         cribriform-morular variant PTC PATHOGNOMONIC young women;
         CHRPE bilateral multifocal PATHOGNOMONIC;
         prophylactic colectomy mandatory;
         seed SEED_BASE+2) .
DICER1  (DICER1 RNase III; 1922aa; 14q32.13; AD LOF;
         DICER1 syndrome;
         multinodular goiter 75% female carriers PATHOGNOMONIC in context;
         differentiated thyroid carcinoma 1%;
         PPB PATHOGNOMONIC; CT chest siblings <8yr MANDATORY;
         seed SEED_BASE+3) .
MEN1    (Menin; 610aa; 11q13.1; AD LOF;
         Multiple endocrine neoplasia type 1 (MEN1);
         follicular thyroid adenoma 30-75% carriers (autopsy);
         parathyroid 95% EARLIEST manifestation;
         everolimus / sunitinib pancreatic NET;
         seed SEED_BASE+4) .
PRKAR1A (Protein kinase cAMP-dependent regulatory I alpha; 381aa; 17q24.2; AD LOF;
         Carney complex (CNC);
         thyroid follicular adenoma near universal;
         cardiac myxoma 30-40% PATHOGNOMONIC; annual echo MANDATORY;
         spotty perioral pigmentation PATHOGNOMONIC;
         seed SEED_BASE+5) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         anaplastic thyroid carcinoma in LFS PATHOGNOMONIC;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+6) .
NF1     (Neurofibromin; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis type 1;
         follicular thyroid carcinoma association;
         cafe-au-lait macules PATHOGNOMONIC;
         MPNST 8-13% lifetime; selumetinib FDA 2020;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3326-3333)
"""
import random

SEED_BASE = 3326

ATLAS_GENES = [
    {
        "gene": "RET",
        "protein": (
            "RET -- 10q11.21 Autosomal-Dominant-GOF -- 1114aa -- "
            "REarranged-during-Transfection-124kDa-RTK-MEN2A-FMTC-MEN2B-"
            "C634R-MEN2A-PATHOGNOMONIC-M918T-MEN2B-PATHOGNOMONIC-"
            "Vandetanib-Cabozantinib-FDA-Prophylactic-Thyroidectomy-MEN2B-Age-0-6mo-OMIM-164761"
        ),
        "locus": "10q11.21",
        "protein_size": (
            "1114 aa / 124 kDa / 10q11.21 RET encodes REarranged during Transfection receptor tyrosine kinase: "
            "STRUCTURE: "
            "  1114 aa / 124 kDa; transmembrane receptor tyrosine kinase; "
            "  Extracellular domain: cadherin-like repeats + cysteine-rich domain; "
            "  Transmembrane domain (aa 636-657); "
            "  Intracellular kinase domain (aa 724-1013): activation loop Y905/Y1062; "
            "  GOF variants → constitutive dimerisation or kinase activation → RAS/MAPK and PI3K/AKT; "
            "  Ligands: GDNF family (GDNF, NTN, ARTN, PSPN) via GFRα co-receptors; "
            "MEN2A (Multiple Endocrine Neoplasia Type 2A): "
            "  Medullary thyroid carcinoma (MTC): 95% lifetime — HIGHEST penetrance; "
            "  Phaeochromocytoma: 50% lifetime; "
            "  Primary hyperparathyroidism: 20-30%; "
            "  C634R (exon 11) MOST COMMON MEN2A hotspot — PATHOGNOMONIC; "
            "  Hirschsprung disease risk (LOF alleles co-existing); "
            "MEN2B (Multiple Endocrine Neoplasia Type 2B): "
            "  M918T (exon 16) hotspot — PATHOGNOMONIC MEN2B; 95% de novo; "
            "  EARLIEST MTC onset: infancy (0-1yr); prophylactic thyroidectomy 0-6 MONTHS; "
            "  Mucosal neuromas lips/tongue PATHOGNOMONIC MEN2B; "
            "  Marfanoid habitus; intestinal ganglioneuromas; "
            "  NO hyperparathyroidism (DDx from MEN2A); "
            "FMTC (Familial Medullary Thyroid Carcinoma): "
            "  MTC only; no pheo/HPT; milder C609/C611/C618/C620 exon 10 variants; "
            "  Annual calcitonin from age 6 months (MEN2B) / age 5yr (MEN2A); "
            "TREATMENT RET/MTC: "
            "  Vandetanib: FDA 2011 progressive/symptomatic MTC (RET kinase inhibitor); "
            "  Cabozantinib: FDA 2012 progressive MTC (RET + MET + VEGFR2); "
            "  Pralsetinib: FDA 2020 RET-mutant MTC (highly selective); "
            "  Selpercatinib: FDA 2020 RET-mutant MTC (highly selective); "
            "  Thyroid surgery: total thyroidectomy + central node dissection; "
            "  Prophylactic thyroidectomy: MEN2B by age 6 months; MEN2A by age 5 yr; "
            "SURVEILLANCE: "
            "  Annual calcitonin + CEA; annual neck USS; annual DOPA-PET (MTC monitoring); "
            "  Annual metanephrines (pheochromocytoma screen); annual Ca2+/PTH (MEN2A);"
        ),
        "inheritance": "AD GOF",
        "syndrome": "MEN2A / FMTC / MEN2B",
        "tc_risk": "MTC 95% lifetime — HIGHEST penetrance germline RET GOF",
        "pathognomonic": "C634R hotspot MEN2A; M918T hotspot MEN2B; mucosal neuromas MEN2B; MTC infant/child",
        "key_avoid": "DELAY THYROIDECTOMY — prophylactic surgery must be age-stratified: MEN2B 0-6 months",
        "surveillance": "Annual calcitonin + CEA + neck USS; annual metanephrines; annual Ca2+/PTH (MEN2A)",
        "targeted_rx": "Selpercatinib / pralsetinib FDA2020; vandetanib / cabozantinib FDA older",
        "key_rule": "PROPHYLACTIC THYROIDECTOMY MANDATORY — MEN2B by 6 months; MEN2A by age 5yr; calcitonin drives timing",
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "PTEN-47kDa-PI3K-AKT-Phosphatase-Cowden-PHTS-"
            "Follicular-Papillary-Thyroid-35-67pct-"
            "Macrocephaly-PATHOGNOMONIC-Lhermitte-Duclos-PATHOGNOMONIC-"
            "Everolimus-mTOR-OMIM-601728"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 47 kDa / 10q23.31 PTEN encodes phosphatase and tensin homologue: "
            "STRUCTURE: "
            "  403 aa / 47 kDa; dual-specificity phosphatase (lipid + protein); "
            "  N-terminal PBD (phosphatase binding domain); "
            "  Phosphatase domain (aa 7-185): active site C124 — dephosphorylates PIP3 → PIP2; "
            "  C2 domain (aa 186-351): membrane recruitment; "
            "  C-terminal tail (aa 352-403): regulatory phosphorylation sites; "
            "  PTEN LOF → PIP3 accumulation → constitutive PI3K/AKT/mTOR → proliferation, survival; "
            "COWDEN SYNDROME / PHTS (PTEN Hamartoma Tumour Syndrome): "
            "  Thyroid cancer: 35-67% lifetime (follicular > papillary; anaplastic rare); "
            "  Macrocephaly ≥97th percentile PATHOGNOMONIC — screen children presenting with large head; "
            "  Lhermitte-Duclos disease (cerebellar gangliocytoma dysplastic) PATHOGNOMONIC adult; "
            "  Breast cancer: 85% lifetime (female); "
            "  Endometrial cancer: 28-44% lifetime; "
            "  Colorectal cancer: 9-16% lifetime; "
            "  Renal cell carcinoma: 34% lifetime; "
            "  Facial trichilemmomas PATHOGNOMONIC (pathognomonic when multiple); "
            "  Papillomatous papules perioral PATHOGNOMONIC; acral keratoses; "
            "THYROID SPECIFICS IN PTEN: "
            "  Annual thyroid USS from age 7-18yr (paediatric onset); "
            "  Multinodular goiter common precursor; "
            "  Follicular thyroid carcinoma: most common malignant histology; "
            "  Anaplastic TC: rare but documented in PTEN germline context; "
            "  Total thyroidectomy for multinodular goiter management; "
            "TREATMENT PTEN: "
            "  Everolimus (mTOR inhibitor): FDA approved for PTEN-mutant advanced TC; "
            "  Lenvatinib / sorafenib for advanced differentiated TC; "
            "  Brain MRI for Lhermitte-Duclos surveillance;"
        ),
        "inheritance": "AD LOF",
        "syndrome": "Cowden Syndrome / PTEN Hamartoma Tumour Syndrome (PHTS)",
        "tc_risk": "Follicular/papillary thyroid cancer 35-67% lifetime",
        "pathognomonic": "Macrocephaly ≥97th pct PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC adult; trichilemmomas",
        "key_avoid": "MISS MACROCEPHALY — macrocephaly in child mandates PTEN testing; thyroid USS from age 7",
        "surveillance": "Annual thyroid USS from age 7; annual breast MRI from 30; annual endometrial sampling from 30-35",
        "targeted_rx": "Everolimus FDA for PTEN-mutant TC; lenvatinib/sorafenib advanced differentiated TC",
        "key_rule": "MACROCEPHALY ≥97th PERCENTILE — paediatric macrocephaly + any benign tumour → test PTEN immediately",
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "APC-310kDa-WNT-Destruction-Complex-Scaffold-FAP-Gardner-"
            "Cribriform-Morular-PTC-PATHOGNOMONIC-CHRPE-PATHOGNOMONIC-"
            "Prophylactic-Colectomy-Mandatory-OMIM-175100"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 310 kDa / 5q22.2 APC encodes adenomatous polyposis coli: "
            "STRUCTURE: "
            "  2843 aa / 310 kDa; scaffolding protein; "
            "  N-terminal homodimerisation domain; "
            "  ARM repeats: protein interactions; "
            "  20-aa repeats (aa 1020-1169): axin binding; β-catenin destruction; "
            "  SAMP repeats (aa 1500-2075): axin binding; "
            "  C-terminal EB1/HDLG binding domain; "
            "  APC LOF → β-catenin destruction complex fails → nuclear WNT → proliferation; "
            "  MCR (mutation cluster region): codons 1250-1464 hotspot; "
            "FAP / GARDNER SYNDROME: "
            "  100% CRC risk by age 40 untreated; prophylactic colectomy MANDATORY; "
            "  CHRPE (congenital hypertrophy retinal pigment epithelium): bilateral multifocal PATHOGNOMONIC; "
            "  Desmoid tumours: 15% (Gardner variant); mesenteric desmoids life-threatening; "
            "  Hepatoblastoma: children of FAP families (<5yr); AFP surveillance; "
            "  Medulloblastoma: SHH subtype (Turcot variant); "
            "THYROID IN FAP: "
            "  Cribriform-morular variant of papillary thyroid carcinoma (CMVPTC): PATHOGNOMONIC APC; "
            "  CMVPTC: young women (<30yr); multifocal; unique cribriform architecture; "
            "  Thyroid cancer: ~2-5% of FAP; predominantly CMVPTC histology; "
            "  Annual thyroid USS from age 15 in FAP (or sooner if goiter); "
            "  Total thyroidectomy: treatment for CMVPTC given multifocal bilateral nature; "
            "  Standard PTC prognosis if detected early; "
            "DESMOID MANAGEMENT: "
            "  NSAID (sulindac) + tamoxifen: first-line desmoid FAP; "
            "  Nirogacestat (gamma-secretase inhibitor): FDA 2023 desmoid tumours; "
            "  Sunitinib / sorafenib: advanced desmoid FAP; "
            "SURVEILLANCE: "
            "  Annual colonoscopy from age 10-12; prophylactic colectomy before polyps malignant; "
            "  Annual thyroid USS from age 15; upper GI surveillance from 25;"
        ),
        "inheritance": "AD LOF",
        "syndrome": "Familial Adenomatous Polyposis (FAP) / Gardner Syndrome",
        "tc_risk": "Cribriform-morular PTC 2-5% FAP; PATHOGNOMONIC histological variant",
        "pathognomonic": "CHRPE bilateral multifocal PATHOGNOMONIC APC; cribriform-morular PTC young woman PATHOGNOMONIC",
        "key_avoid": "STANDARD PTC PROTOCOLS for CMVPTC — multifocal bilateral requires total thyroidectomy not hemi",
        "surveillance": "Annual thyroid USS from age 15; annual colonoscopy from 10-12; prophylactic colectomy",
        "targeted_rx": "Nirogacestat FDA2023 desmoid; lenvatinib/sorafenib advanced differentiated TC",
        "key_rule": "CRIBRIFORM-MORULAR PTC — PATHOGNOMONIC APC germline; total thyroidectomy (multifocal bilateral); APC test mandatory",
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-219kDa-RNase-III-miRNA-Processor-"
            "Multinodular-Goiter-75pct-Females-PATHOGNOMONIC-in-Context-"
            "Differentiated-TC-1pct-PPB-PATHOGNOMONIC-CT-Chest-Siblings-LT-8yr-MANDATORY-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 219 kDa / 14q32.13 DICER1 encodes DICER1 endoribonuclease: "
            "STRUCTURE: "
            "  1922 aa / 219 kDa; multidomain RNase III enzyme; "
            "  N-terminal DExH helicase domain; "
            "  PAZ domain (aa 848-988): 3' overhang recognition; "
            "  RNase IIIa domain (aa 1326-1448): cleavage one strand; "
            "  RNase IIIb domain (aa 1647-1844): cleavage other strand — hotspot mutations here; "
            "  Hotspot codons: E1705/D1709 (metal binding); E1813/D1810 (product release); "
            "  DICER1 germline LOF + somatic RNase IIIb hotspot = biallelic tumorigenesis; "
            "DICER1 SYNDROME THYROID FEATURES: "
            "  Multinodular goiter (MNG): 75% female DICER1 carriers by age 30-40; "
            "  MNG as FIRST clinical manifestation in many carriers; thyroid the most common organ; "
            "  Annual thyroid USS from age 8 in all DICER1 carriers (male and female); "
            "  Differentiated thyroid carcinoma (DTC): ~1% of carriers; "
            "  Thyroid carcinoma in DICER1: both papillary and follicular histology documented; "
            "DICER1 SYNDROME OTHER FEATURES: "
            "  PPB Type I (pleuropulmonary blastoma): PATHOGNOMONIC; cystic lung infancy; "
            "  CT CHEST ALL SIBLINGS <8yr MANDATORY — PPB is lethal if missed; "
            "  Sertoli-Leydig cell tumour (SLCT) ovary: PATHOGNOMONIC young women; "
            "  Cervical embryonal rhabdomyosarcoma (ERMS): PATHOGNOMONIC; "
            "  Cystic nephroma PATHOGNOMONIC; pineoblastoma; ciliary body medulloepithelioma; "
            "AVOID RADIATION CHILDREN — DICER1 syndrome; "
            "TREATMENT: "
            "  MNG: thyroid USS annual, levothyroxine only if hypothyroid; "
            "  DTC: lobectomy for low-risk, total thyroidectomy for multifocal; "
            "  RAI: standard for DTC if total thyroidectomy performed; "
            "  PPB surgery: DICER1 PPB resection + chemotherapy (VAC or VDC); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "DICER1 Syndrome",
        "tc_risk": "MNG 75% females; DTC ~1% carriers",
        "pathognomonic": "PPB Type I infant PATHOGNOMONIC DICER1; SLCT ovary young woman; cervical ERMS; MNG young",
        "key_avoid": "MISS PPB — CT chest ALL siblings <8yr MANDATORY; thyroid USS from age 8",
        "surveillance": "Annual thyroid USS from age 8; CT chest siblings <8yr (PPB); annual pelvic USS puberty",
        "targeted_rx": "RAI for DTC after total thyroidectomy; lenvatinib/sorafenib advanced DTC",
        "key_rule": "CT CHEST ALL SIBLINGS <8yr MANDATORY — PPB Type I is lethal if missed; thyroid is the sentinel organ",
    },
    {
        "gene": "MEN1",
        "protein": (
            "MEN1 -- 11q13.1 Autosomal-Dominant-LOF -- 610aa -- "
            "Menin-68kDa-H3K4-Methyltransferase-Scaffold-MEN1-Triad-"
            "Parathyroid-95pct-Earliest-Pituitary-40pct-PancNET-70pct-"
            "Thyroid-Adenoma-30-75pct-Everolimus-Sunitinib-OMIM-131100"
        ),
        "locus": "11q13.1",
        "protein_size": (
            "610 aa / 68 kDa / 11q13.1 MEN1 encodes menin: "
            "STRUCTURE: "
            "  610 aa / 68 kDa; nuclear scaffold protein; no intrinsic enzymatic activity; "
            "  Interacts with MLL1/MLL2 histone H3K4 methyltransferase complex (KAT3A/KAT3B); "
            "  Regulates CDK inhibitors: CDKN1B (p27), CDKN2C (p18) — MEN1 LOF → CDK inhibitor loss; "
            "  JunD binding domain (aa 1-40): JunD transcriptional repressor interaction; "
            "  C-terminal nuclear export signal; "
            "  MEN1 LOF → reduced H3K4 methylation at tumour suppressor loci → proliferation; "
            "MULTIPLE ENDOCRINE NEOPLASIA TYPE 1 (MEN1): "
            "  Classic Triad: Parathyroid 95% (EARLIEST) + Pituitary 40% + Pancreatic/duodenal NET 70%; "
            "  Most common hereditary endocrine tumour syndrome (1 in 30,000); "
            "  Annual calcium/PTH from age 5-8 (first manifestation hyperparathyroidism); "
            "  Gastrinoma / Zollinger-Ellison: 25-40%; ulcers; PPI lifelong; "
            "THYROID IN MEN1: "
            "  Follicular thyroid adenoma: 30-75% (autopsy data, cross-sectional imaging series); "
            "  Predominantly non-functional; bilateral multinodular; "
            "  Thyroid carcinoma in MEN1: rare (<5% of thyroid lesions); "
            "  Annual thyroid USS as part of MEN1 surveillance from age 10; "
            "  Size >3 cm or growing → resection (lower threshold in MEN1 bilateral risk); "
            "  Total thyroidectomy if carcinoma or multiple growing nodules; "
            "PANCREATIC NET TREATMENT: "
            "  <2 cm: watch + annual MRI; ≥2 cm: surgery (Whipple or distal pancreatectomy); "
            "  Everolimus (mTOR inhibitor): FDA approved advanced pancreatic NET; "
            "  Sunitinib: FDA approved advanced pancreatic NET; "
            "  Somatostatin analogues (octreotide/lanreotide): functional tumour symptom control; "
            "  Lutetium-177 DOTATATE (PRRT): FDA 2018 advanced somatostatin-positive NETs; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Multiple Endocrine Neoplasia Type 1 (MEN1)",
        "tc_risk": "Follicular thyroid adenoma 30-75%; carcinoma rare (<5% of thyroid lesions)",
        "pathognomonic": "Multiglandular parathyroid hyperplasia + pancreatic NET + pituitary adenoma = MEN1 TRIAD PATHOGNOMONIC",
        "key_avoid": "SINGLE GLAND PARATHYROID SURGERY — MEN1 requires 3.5-gland resection; single adenomectomy will recur",
        "surveillance": "Annual Ca2+/PTH from age 8; annual thyroid USS; 3-yearly pituitary MRI; annual fasting gastrin/glucagon",
        "targeted_rx": "Everolimus / sunitinib / lutetium-177 DOTATATE FDA approved pancreatic NET",
        "key_rule": "3.5-GLAND PARATHYROID RESECTION — MEN1 hyperparathyroidism is multiglandular; single-gland surgery fails",
    },
    {
        "gene": "PRKAR1A",
        "protein": (
            "PRKAR1A -- 17q24.2 Autosomal-Dominant-LOF -- 381aa -- "
            "PKA-R1alpha-43kDa-cAMP-Regulatory-Subunit-Carney-Complex-"
            "Thyroid-Follicular-Adenoma-Near-Universal-"
            "Cardiac-Myxoma-30-40pct-ANNUAL-ECHO-MANDATORY-"
            "Spotty-Perioral-Pigmentation-PATHOGNOMONIC-OMIM-188830"
        ),
        "locus": "17q24.2",
        "protein_size": (
            "381 aa / 43 kDa / 17q24.2 PRKAR1A encodes protein kinase cAMP-dependent regulatory I alpha: "
            "STRUCTURE: "
            "  381 aa / 43 kDa; regulatory subunit of Protein Kinase A (PKA); "
            "  Dimerisation/docking domain (aa 1-45): AKAP anchoring; "
            "  Inhibitory domain (aa 94-100): pseudosubstrate; "
            "  cAMP-binding domain A (aa 149-252): first cAMP binding pocket; "
            "  cAMP-binding domain B (aa 253-376): second cAMP binding pocket; "
            "  PRKAR1A LOF → disinhibited PKA catalytic subunit → constitutive cAMP/PKA signalling; "
            "CARNEY COMPLEX (CNC): "
            "  PRKAR1A LOF (75-80%); some due to PRKAR1A large deletions (MLPA); "
            "  Spotty skin pigmentation: lentigines (perioral, periocular, conjunctival, genital mucosal) PATHOGNOMONIC; "
            "  Blue nevi; myxomatous skin lesions; labial mucosal spots; "
            "CARDIAC MYXOMA — LIFE-THREATENING: "
            "  30-40% of Carney Complex patients; can be bilateral (both ventricles) and valvular; "
            "  Embolic stroke risk if undetected; sudden death documented; "
            "  ANNUAL ECHOCARDIOGRAM MANDATORY from diagnosis — cannot be omitted; "
            "  Can recur in any cardiac chamber after resection; "
            "THYROID IN CARNEY COMPLEX: "
            "  Thyroid follicular adenoma: near universal (>70% on USS); "
            "  Multinodular goiter; predominantly benign; "
            "  Follicular thyroid carcinoma: rare but documented (<5%); "
            "  Annual thyroid USS from diagnosis / age 10; "
            "  Total thyroidectomy if carcinoma or multinodular with suspicious FNA; "
            "OTHER CARNEY COMPLEX FEATURES: "
            "  PPNAD (Primary Pigmented Nodular Adrenocortical Disease): bilateral micro-nodular; "
            "  Paradoxical cortisol rise on Liddle test PATHOGNOMONIC; "
            "  GH-secreting pituitary adenoma: 10% (acromegaly); "
            "  LCCSCT (large-cell calcifying Sertoli cell tumour): 30-40% males PATHOGNOMONIC; "
            "  Psammomatous melanotic schwannoma; breast ductal adenoma; "
            "SURVEILLANCE: "
            "  Annual echocardiogram MANDATORY; annual thyroid USS; "
            "  Annual midnight cortisol / Liddle test; annual testosterone/DHEAS (males); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Carney Complex (CNC)",
        "tc_risk": "Thyroid follicular adenoma near universal (>70%); carcinoma rare (<5%)",
        "pathognomonic": "Spotty perioral/genital lentigines PATHOGNOMONIC; cardiac myxoma PATHOGNOMONIC; LCCSCT males PATHOGNOMONIC",
        "key_avoid": "MISS CARDIAC MYXOMA — annual echo MANDATORY; myxoma causes fatal embolism if undetected",
        "surveillance": "ANNUAL ECHOCARDIOGRAM MANDATORY; annual thyroid USS; annual midnight cortisol; Liddle test",
        "targeted_rx": "Surgical resection cardiac myxoma; bilateral adrenalectomy PPNAD Cushing; thyroidectomy if carcinoma",
        "key_rule": "ANNUAL ECHOCARDIOGRAM MANDATORY — cardiac myxoma causes fatal stroke/sudden death if missed",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-"
            "Anaplastic-Thyroid-Carcinoma-PATHOGNOMONIC-LFS-Context-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; homotetrameric transcription factor; genome guardian; "
            "  N-terminal transactivation domain (aa 1-42): MDM2 binding (E3 ubiquitin ligase); "
            "  Proline-rich region (aa 40-90): apoptosis regulation; "
            "  DNA-binding domain (aa 94-292): most GOF hotspot mutations (R175H, G245S, R248W, R273H); "
            "  Tetramerisation domain (aa 323-356): functional tetramer assembly; "
            "  C-terminal regulatory (aa 356-393): post-translational modification; "
            "  TP53 activates CDKN1A (p21) → G1/S arrest; PUMA/NOXA → apoptosis; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; AD LOF; ~20% de novo; "
            "  Near 100% penetrance by age 70; "
            "  Sarcoma: 50-60% (dominant LFS cancer); "
            "  Breast cancer <45yr: 30%; "
            "  Brain tumour (glioma, CPC): 15%; "
            "  Adrenocortical carcinoma child: PATHOGNOMONIC; "
            "  Thyroid carcinoma in LFS: predominantly anaplastic histology; "
            "TP53 AND ANAPLASTIC THYROID CARCINOMA (ATC): "
            "  ATC: somatic TP53 LOF in 60-80% sporadic ATC (most aggressive thyroid malignancy); "
            "  Germline TP53 in ATC: documented in LFS families; PATHOGNOMONIC LFS context; "
            "  ATC prognosis: median survival 3-5 months; surgery + chemoRT standard; "
            "  AVOID RADIATION ABSOLUTELY in germline TP53 — secondary malignancy acceleration; "
            "  Dabrafenib + trametinib: FDA for BRAF V600E mutant ATC (somatic; check germline separately); "
            "  In LFS patient with ATC: prefer surgery over RT consolidation; "
            "SURVEILLANCE LFS: "
            "  WBMRI: annually (Toronto protocol); brain MRI annually; "
            "  Abdominopelvic US every 3-4 months <18yr; "
            "  Breast MRI annually from age 20-25yr; "
            "  Annual thyroid USS (thyroid cancer risk elevated); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Li-Fraumeni Syndrome (LFS)",
        "tc_risk": "Anaplastic thyroid carcinoma in LFS context PATHOGNOMONIC; overall TC risk ~2-3x baseline",
        "pathognomonic": "ACC child PATHOGNOMONIC LFS; CPC child PATHOGNOMONIC; anaplastic TC in LFS context",
        "key_avoid": "RADIATION — AVOID RADIATION ABSOLUTELY in germline TP53; surgery preferred over RT for ATC in LFS",
        "surveillance": "WBMRI annually + brain MRI + abdominopelvic US; annual thyroid USS",
        "targeted_rx": "Dabrafenib + trametinib FDA for BRAF V600E ATC (somatic; check germline separately)",
        "key_rule": "AVOID RADIATION ABSOLUTELY — TP53 germline + radiation → secondary sarcoma/carcinoma in RT field",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-NF1-"
            "Follicular-Thyroid-Carcinoma-Association-"
            "Cafe-au-Lait-PATHOGNOMONIC-MPNST-8-13pct-"
            "Selumetinib-FDA2020-Paediatric-Plexiform-OMIM-613113"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes neurofibromin: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; GTPase activating protein (GAP) for RAS; "
            "  Central GRD (GTPase-activating related domain, aa 1175-1551): RAS-GTP hydrolysis; "
            "  NF1 LOF → reduced RAS-GAP activity → elevated RAS-GTP → activated RAS/MAPK; "
            "  SEC14 domain: lipid binding; "
            "  CSRD and CTD domains: scaffold functions; "
            "  Largest known tumour suppressor gene (350 kb genomic span); "
            "  Highest spontaneous germline mutation rate of any known tumour suppressor; "
            "NF1 SYNDROME — NIH DIAGNOSTIC CRITERIA (≥2 of 8): "
            "  ≥6 café-au-lait macules (>5mm prepubertal / >15mm postpubertal) PATHOGNOMONIC; "
            "  ≥2 neurofibromas OR ≥1 plexiform neurofibroma; "
            "  Axillary OR inguinal freckling PATHOGNOMONIC (Crowe sign); "
            "  Optic pathway glioma; "
            "  ≥2 Lisch nodules (iris hamartomas); "
            "  Distinctive osseous lesion (sphenoid dysplasia; tibial pseudoarthrosis); "
            "  First-degree relative with NF1; "
            "NF1 THYROID ASSOCIATION: "
            "  Follicular thyroid adenoma and carcinoma: association documented in NF1 series; "
            "  Probable 2-3x elevated risk; predominantly follicular histology; "
            "  Annual thyroid USS recommended in NF1 from age 20 (no paediatric guidance yet); "
            "  MPNST (malignant peripheral nerve sheath tumour): dominant cancer 8-13%; "
            "NF1 ONCOLOGICAL BURDEN: "
            "  Optic pathway glioma: 15-20% children; selumetinib FDA 2020 paediatric symptomatic; "
            "  JMML: 200x elevated in NF1 children; "
            "  Breast cancer: ~5x elevated; "
            "  GI stromal tumour (GIST): elevated; "
            "  Phaeochromocytoma: ~5%; adrenocortical adenoma: 3-5%; "
            "TREATMENT NF1: "
            "  Selumetinib (MEK inhibitor): FDA 2020 paediatric inoperable plexiform neurofibromas; "
            "  MPNST: complete resection (R0) first-line; poor response to chemotherapy; "
            "  Optic glioma: selumetinib first-line (replaces carboplatin/vincristine); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Neurofibromatosis Type 1 (NF1)",
        "tc_risk": "Follicular thyroid carcinoma 2-3x elevated association; dominant risk is MPNST 8-13%",
        "pathognomonic": "≥6 cafe-au-lait macules PATHOGNOMONIC; axillary freckling PATHOGNOMONIC; Lisch nodules PATHOGNOMONIC",
        "key_avoid": "AVOID RADIATION — radiation risk in NF1 similar to TP53; MPNST acceleration documented",
        "surveillance": "Annual NF specialist exam; annual thyroid USS from 20; selumetinib for symptomatic plexiform; BP annually",
        "targeted_rx": "Selumetinib FDA2020 paediatric plexiform NF; consider for MPNST (clinical trial)",
        "key_rule": "CAFE-AU-LAIT ≥6 MACULES PATHOGNOMONIC — NF1 diagnosis mandates thyroid USS + annual NF surveillance",
    },
]

# Tumour types per gene
TUMOUR_TYPES_BY_GENE: dict = {
    "RET":     ["Medullary Thyroid Carcinoma (MTC)", "MTC Bilateral", "MTC + Phaeochromocytoma (MEN2A)", "MTC + Mucosal Neuromas (MEN2B)", "MTC Infant MEN2B", "FMTC (MTC only family)"],
    "PTEN":    ["Follicular Thyroid Carcinoma", "Papillary Thyroid Carcinoma (PTC)", "Follicular Thyroid Adenoma (benign)", "Multinodular Goiter + Carcinoma", "Anaplastic TC rare"],
    "APC":     ["Cribriform-Morular PTC (CMVPTC)", "Papillary Thyroid Carcinoma", "Follicular Thyroid Adenoma", "CMVPTC Multifocal Bilateral"],
    "DICER1":  ["Multinodular Goiter", "Papillary Thyroid Carcinoma", "Follicular Thyroid Carcinoma", "PPB Type I (lung sentinel)", "SLCT Ovary (sentinel)"],
    "MEN1":    ["Follicular Thyroid Adenoma (benign)", "Follicular Thyroid Carcinoma", "Multinodular Goiter", "Thyroid Adenoma Bilateral"],
    "PRKAR1A": ["Thyroid Follicular Adenoma", "Follicular Thyroid Carcinoma", "Multinodular Goiter + Adenoma", "Thyroid Adenoma + Cardiac Myxoma"],
    "TP53":    ["Anaplastic Thyroid Carcinoma (ATC)", "Papillary Thyroid Carcinoma (LFS)", "Poorly Differentiated TC", "Secondary TC Post-RT Field"],
    "NF1":     ["Follicular Thyroid Carcinoma", "Follicular Thyroid Adenoma", "Papillary Thyroid Carcinoma (NF1)", "Multinodular Goiter NF1"],
}

# Pathogenic variants per gene
VARIANTS_BY_GENE: dict = {
    "RET":     ["p.C634R (exon 11, MEN2A PATHOGNOMONIC)", "p.M918T (exon 16, MEN2B PATHOGNOMONIC)", "p.C618S (exon 10, FMTC)", "p.C620R (exon 10, MEN2A milder)", "p.C609Y (exon 10, FMTC)", "p.C634W (exon 11, MEN2A)", "Large exon deletion (MLPA)"],
    "PTEN":    ["p.R130Q (phosphatase dead)", "p.R130G (phosphatase dead)", "p.C124S (active site ablation)", "p.H93R (DBD)", "p.Y178C (C2 domain)", "Exon 3 deletion", "Promoter variant (reporter assay)"],
    "APC":     ["p.E1309del (del codon 1309, MCR hotspot)", "p.Q1338Ter (MCR)", "p.R876Ter (5' codons, attenuated)", "p.Y486Ter (Gardner desmoid)", "5q22 deletion (MLPA)", "Splicing variant IVS9"],
    "DICER1":  ["p.D1709N (RNase IIIb hotspot)", "p.E1705K (RNase IIIb hotspot)", "p.G1809R (RNase IIIb hotspot)", "p.R1748W (RNase IIIb)", "Exon 24 deletion", "Frameshift c.4662delT"],
    "MEN1":    ["p.R460Ter (frameshift region)", "p.W341Ter (common)", "p.V184E (missense)", "p.L22R (JunD binding)", "11q13 deletion (MLPA)", "IVS2+1G>A splice"],
    "PRKAR1A": ["p.R74Ter (common CNC)", "p.L206R (cAMP-BD A)", "p.S9Ter (N-terminal)", "17q24.2 large deletion (MLPA 30%)", "c.708+1G>A splice", "p.E143Val"],
    "TP53":    ["p.R175H (DNA-binding domain hotspot GOF)", "p.R248W (hotspot)", "p.G245S (hotspot)", "p.R337H (Brazilian founder)", "p.C176F (ZnF)", "Splice donor IVS4+1"],
    "NF1":     ["p.R1947Ter (common)", "p.Q1966Ter", "Exon 22 skipping (splice)", "17q11.2 microdeletion (MLPA 5%)", "p.R1276Q (GRD missense)", "Large segmental deletion"],
}

# Treatment protocols per gene
TREATMENT_PROTOCOLS_BY_GENE: dict = {
    "RET":     ["Total thyroidectomy + central neck dissection", "Selpercatinib (RET-selective FDA 2020) advanced MTC", "Pralsetinib (RET-selective FDA 2020) advanced MTC", "Cabozantinib (RET+MET+VEGFR2) progressive MTC", "Vandetanib (RET kinase) progressive MTC", "Prophylactic thyroidectomy (MEN2B age 0-6mo; MEN2A age 5yr)", "Annual calcitonin + CEA monitoring"],
    "PTEN":    ["Total thyroidectomy (multinodular + carcinoma risk)", "Everolimus (mTOR) PTEN-mutant advanced TC", "Lenvatinib / sorafenib advanced DTC", "RAI post-total thyroidectomy DTC", "Breast MRI annual surveillance", "Annual thyroid USS from age 7"],
    "APC":     ["Total thyroidectomy (CMVPTC multifocal bilateral)", "RAI post-thyroidectomy CMVPTC", "Prophylactic colectomy (before polyps malignant)", "Sulindac + tamoxifen desmoid", "Nirogacestat (gamma-secretase) FDA 2023 desmoid", "Annual thyroid USS from age 15"],
    "DICER1":  ["Annual thyroid USS from age 8", "Total/hemithyroidectomy DTC depending on risk", "RAI if total thyroidectomy DTC", "PPB: resection + VAC/VDC chemotherapy", "SLCT: ovarian surgery + FIGO staging", "CT chest siblings <8yr (PPB surveillance)"],
    "MEN1":    ["3.5-gland parathyroid resection (multiglandular MEN1)", "Pancreatic NET surgery if ≥2 cm", "Everolimus (mTOR) advanced pancreatic NET", "Sunitinib advanced pancreatic NET", "Lutetium-177 DOTATATE (PRRT) advanced NET", "Annual thyroid USS + resection if growing/carcinoma"],
    "PRKAR1A": ["ANNUAL ECHOCARDIOGRAM — cardiac myxoma resection", "Bilateral adrenalectomy PPNAD Cushing", "Annual thyroid USS + thyroidectomy if carcinoma/growing", "Steroid replacement post-adrenalectomy (Addisonian)", "Annual testicular USS males (LCCSCT)", "Annual pituitary MRI (GH adenoma)"],
    "TP53":    ["Surgery preferred over RT (AVOID RADIATION ABSOLUTELY)", "Doxorubicin-based for ATC (non-LFS protocol modified)", "Dabrafenib + trametinib BRAF V600E ATC (check germline separately)", "WBMRI Toronto annual surveillance", "Annual thyroid USS", "MDM2 inhibitors investigational"],
    "NF1":     ["Total thyroidectomy for follicular thyroid carcinoma NF1", "Selumetinib (MEK) FDA 2020 paediatric plexiform NF", "MPNST: surgery R0 first-line", "RAI post-thyroidectomy NF1 DTC", "Annual thyroid USS NF1 from age 20", "Annual BP + urine catecholamines (pheo screen)"],
}

# Surveillance protocols per gene
SURVEILLANCE_BY_GENE: dict = {
    "RET":     ["Annual calcitonin + CEA from age 6mo (MEN2B) / 5yr (MEN2A)", "Annual neck USS", "Annual plasma/urine metanephrines (pheo screen)", "Annual serum Ca2+/PTH (MEN2A hyperparathyroidism)", "Annual 24h urine catecholamines", "DOPA-PET if calcitonin rising MTC recurrence"],
    "PTEN":    ["Annual thyroid USS from age 7-18yr", "Annual breast MRI from age 25-30yr (women)", "Annual endometrial sampling from age 30-35yr", "Annual renal USS/MRI from age 40yr", "Annual dermatology exam (trichilemmomas)", "Brain MRI if neurological symptoms (Lhermitte-Duclos)"],
    "APC":     ["Annual colonoscopy from age 10-12yr", "Annual thyroid USS from age 15yr", "Annual upper GI endoscopy from age 25yr (duodenal)", "Annual fundoscopy (CHRPE bilateral)", "Annual abdominal MRI (desmoid post-FAP diagnosis)", "AFP 3-monthly until age 7yr (hepatoblastoma)"],
    "DICER1":  ["Annual thyroid USS from age 8yr", "CT chest baseline + 3-monthly first 3yr (PPB)", "Annual pelvic USS from puberty (SLCT/cervical ERMS)", "CT chest all siblings <8yr MANDATORY", "Annual breast USS/MRI (cystic nephroma)", "Annual renal USS (cystic nephroma)"],
    "MEN1":    ["Annual Ca2+/PTH from age 5-8yr (parathyroid)", "Annual fasting gastrin + glucagon + PP (pancreatic NET)", "Annual thyroid USS from age 10yr", "3-yearly pituitary MRI (pituitary adenoma)", "Annual adrenal CT/MRI (adrenocortical adenoma)", "Annual prolactin + IGF-1 (pituitary)"],
    "PRKAR1A": ["ANNUAL ECHOCARDIOGRAM MANDATORY (cardiac myxoma)", "Annual thyroid USS (follicular adenoma)", "Annual midnight salivary cortisol + Liddle test (PPNAD)", "Annual testicular USS from puberty males (LCCSCT)", "Annual pituitary MRI + IGF-1 (GH adenoma)", "Annual dermatology (lentigines, blue nevi)"],
    "TP53":    ["WBMRI Toronto annually", "Annual brain MRI", "Annual thyroid USS", "Abdominopelvic USS every 3-4 months <18yr", "Annual breast MRI from age 20-25yr (women)", "Annual CBC (leukemia surveillance)"],
    "NF1":     ["Annual clinical NF specialist exam", "Annual thyroid USS from age 20yr", "Annual BP + urine catecholamines (pheo/adrenal)", "MRI spine/CNS if new neurological symptoms", "Selumetinib for symptomatic paediatric plexiform NF", "Annual ophthalmology (optic pathway glioma)"],
}

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]


def _make_patient(gene_idx: int, patient_idx: int) -> dict:
    seed = SEED_BASE + gene_idx + patient_idx * len(ATLAS_GENES)
    rng = random.Random(seed)
    gene = _GENE_LIST[gene_idx]
    gene_info = ATLAS_GENES[gene_idx]
    tumour_types = TUMOUR_TYPES_BY_GENE[gene]
    variants = VARIANTS_BY_GENE[gene]
    treatments = TREATMENT_PROTOCOLS_BY_GENE[gene]

    age_at_dx = rng.randint(4, 68)
    tumour_type = rng.choice(tumour_types)
    variant = rng.choice(variants)
    treatment = rng.choice(treatments)
    cr = rng.random() < 0.78
    radiation = rng.random() < (0.05 if gene in ("TP53", "NF1") else 0.22)
    relapse = rng.random() < 0.14 if cr else rng.random() < 0.42

    return {
        "patient_id": f"HTC-{gene}-{patient_idx:03d}",
        "gene": gene,
        "syndrome": gene_info["syndrome"],
        "age_at_dx": age_at_dx,
        "tumour_type": tumour_type,
        "variant": variant,
        "treatment": treatment,
        "cr": cr,
        "radiation": radiation,
        "relapse": relapse,
    }


def _generate_cohort() -> list:
    patients = []
    for gi in range(len(ATLAS_GENES)):
        for pi in range(40):
            patients.append(_make_patient(gi, pi))
    return patients


def generate_overview() -> dict:
    cohort = _generate_cohort()
    n = len(cohort)
    cr_n = sum(1 for p in cohort if p["cr"])
    rad_n = sum(1 for p in cohort if p["radiation"])
    relapse_n = sum(1 for p in cohort if p["relapse"])
    mean_age = round(sum(p["age_at_dx"] for p in cohort) / n, 1)

    gene_summary = {}
    for g in _GENE_LIST:
        pts = [p for p in cohort if p["gene"] == g]
        gene_summary[g] = {
            "n": len(pts),
            "cr_pct": round(100 * sum(1 for p in pts if p["cr"]) / len(pts), 1),
            "radiation_pct": round(100 * sum(1 for p in pts if p["radiation"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
        }

    return {
        "atlas": "Hereditary-Thyroid-Cancer-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Thyroid Cancer Predisposition Reference — RET-PTEN-APC-DICER1-MEN1-PRKAR1A-TP53-NF1",
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "total_patients": n,
        "cr_pct": round(100 * cr_n / n, 1),
        "radiation_pct": round(100 * rad_n / n, 1),
        "relapse_pct": round(100 * relapse_n / n, 1),
        "mean_age_at_dx": mean_age,
        "genes": _GENE_LIST,
        "gene_summary": gene_summary,
        "key_rules": [
            "PROPHYLACTIC THYROIDECTOMY MANDATORY in RET — MEN2B by age 6 months; MEN2A by age 5yr; calcitonin drives timing",
            "ANNUAL ECHOCARDIOGRAM MANDATORY in PRKAR1A — cardiac myxoma causes fatal embolism/sudden death if missed",
            "CRIBRIFORM-MORULAR PTC PATHOGNOMONIC APC — total thyroidectomy (multifocal bilateral); test APC mandatory",
            "AVOID RADIATION ABSOLUTELY in TP53 — secondary malignancy acceleration; surgery preferred over RT",
            "CT CHEST ALL SIBLINGS <8yr MANDATORY in DICER1 — PPB Type I is lethal if missed",
            "3.5-GLAND PARATHYROID RESECTION in MEN1 — multiglandular hyperplasia; single adenomectomy recurs",
            "MACROCEPHALY ≥97th PERCENTILE — paediatric macrocephaly mandates PTEN testing; thyroid USS from age 7",
            "CAFE-AU-LAIT MACULES ≥6 PATHOGNOMONIC NF1 — annual thyroid USS from age 20",
        ],
    }


def generate_breakdown() -> dict:
    cohort = _generate_cohort()
    breakdown = {}
    for gi, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        pts = [p for p in cohort if p["gene"] == gene]
        tumour_counts: dict = {}
        for p in pts:
            tumour_counts[p["tumour_type"]] = tumour_counts.get(p["tumour_type"], 0) + 1
        top_tumours = sorted(tumour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        variant_counts: dict = {}
        for p in pts:
            variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1
        top_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        breakdown[gene] = {
            "gene": gene,
            "protein": gene_info["protein"],
            "locus": gene_info["locus"],
            "syndrome": gene_info["syndrome"],
            "inheritance": gene_info["inheritance"],
            "tc_risk": gene_info["tc_risk"],
            "pathognomonic": gene_info["pathognomonic"],
            "key_avoid": gene_info["key_avoid"],
            "key_rule": gene_info["key_rule"],
            "surveillance": gene_info["surveillance"],
            "targeted_rx": gene_info["targeted_rx"],
            "n_patients": len(pts),
            "cr_pct": round(100 * sum(1 for p in pts if p["cr"]) / len(pts), 1),
            "radiation_pct": round(100 * sum(1 for p in pts if p["radiation"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "top_tumour_types": [{"type": t, "count": c} for t, c in top_tumours],
            "top_variants": [{"variant": v, "count": c} for v, c in top_variants],
            "treatment_protocols": TREATMENT_PROTOCOLS_BY_GENE[gene],
            "surveillance_protocols": SURVEILLANCE_BY_GENE[gene],
        }
    return {"breakdown": breakdown, "genes": _GENE_LIST}


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Thyroid-Cancer-Predisposition-Atlas",
        "definitions": {
            "men2_ret": (
                "MEN2 (Multiple Endocrine Neoplasia Type 2): RET germline GOF; "
                "MEN2A: MTC 95% + Pheo 50% + HPT 20%; C634R most common; "
                "MEN2B: MTC infancy + mucosal neuromas + marfanoid; M918T PATHOGNOMONIC; "
                "FMTC: MTC only; milder exon 10 variants; "
                "Prophylactic thyroidectomy: MEN2B by 6 months; MEN2A by age 5yr."
            ),
            "cowden_pten": (
                "Cowden Syndrome / PHTS (PTEN Hamartoma Tumour Syndrome): PTEN germline LOF; "
                "Thyroid 35-67% (follicular > papillary); breast 85% (women); endometrial 28-44%; "
                "Macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC adult; "
                "Annual thyroid USS from age 7; everolimus for PTEN-mutant advanced TC."
            ),
            "fap_apc": (
                "FAP (Familial Adenomatous Polyposis): APC germline LOF; "
                "100% CRC risk untreated; prophylactic colectomy mandatory; "
                "Cribriform-morular PTC PATHOGNOMONIC in young FAP women; "
                "CHRPE bilateral multifocal PATHOGNOMONIC; "
                "Annual thyroid USS from age 15."
            ),
            "dicer1_syndrome": (
                "DICER1 Syndrome: DICER1 germline LOF; thyroid is the most commonly affected organ; "
                "MNG 75% female carriers; DTC ~1%; "
                "PPB PATHOGNOMONIC; CT chest ALL siblings <8yr MANDATORY; "
                "Annual thyroid USS from age 8."
            ),
            "men1": (
                "MEN1: MEN1 germline LOF; parathyroid 95% (EARLIEST); pituitary 40%; pancreatic NET 70%; "
                "Thyroid follicular adenoma 30-75%; "
                "3.5-gland parathyroid resection mandatory (multiglandular); "
                "Everolimus/sunitinib pancreatic NET; lutetium-177 DOTATATE PRRT."
            ),
            "carney_complex": (
                "Carney Complex (CNC): PRKAR1A germline LOF; "
                "Cardiac myxoma 30-40% PATHOGNOMONIC — annual echo MANDATORY; "
                "Thyroid follicular adenoma near universal; "
                "Spotty perioral/genital lentigines PATHOGNOMONIC; "
                "PPNAD (bilateral micro-nodular adrenocortical hyperplasia); paradoxical Liddle test PATHOGNOMONIC."
            ),
            "lfs_tp53": (
                "Li-Fraumeni Syndrome (LFS): TP53 germline LOF; "
                "Anaplastic TC in LFS context PATHOGNOMONIC; "
                "AVOID RADIATION ABSOLUTELY; "
                "WBMRI Toronto annually; ACC child PATHOGNOMONIC."
            ),
            "nf1_thyroid": (
                "NF1 (Neurofibromatosis Type 1): NF1 germline LOF; "
                "Follicular thyroid carcinoma 2-3x elevated; "
                "≥6 cafe-au-lait macules PATHOGNOMONIC; MPNST 8-13% lifetime dominant risk; "
                "Annual thyroid USS from age 20; selumetinib FDA 2020."
            ),
            "cascade_testing": (
                "CASCADE TESTING — Hereditary Thyroid Cancer: "
                "1. RET: annual calcitonin; prophylactic thyroidectomy age-stratified; pheo screen; "
                "2. PTEN: annual thyroid USS from age 7; macrocephaly screen children; "
                "3. APC: annual colonoscopy from age 10; thyroid USS from 15; CHRPE fundoscopy; "
                "4. DICER1: annual thyroid USS from 8; CT chest siblings <8yr MANDATORY; "
                "5. MEN1: annual Ca2+/PTH; thyroid USS; 3.5-gland parathyroid resection; "
                "6. PRKAR1A: ANNUAL ECHO MANDATORY; thyroid USS; midnight cortisol; "
                "7. TP53: WBMRI annually; thyroid USS; AVOID radiation; "
                "8. NF1: annual thyroid USS from 20; annual BP + catecholamines; selumetinib if plexiform."
            ),
        },
        "key_clinical_rules": [
            {
                "rule": "PROPHYLACTIC THYROIDECTOMY MANDATORY (RET)",
                "gene": "RET",
                "rationale": "MTC in RET GOF has near 100% penetrance; calcitonin elevation signals progression; MEN2B MTC occurs in infancy",
                "consequence": "Delayed thyroidectomy in MEN2B → MTC by age 2yr; metastatic disease by age 5yr",
            },
            {
                "rule": "ANNUAL ECHOCARDIOGRAM MANDATORY (PRKAR1A)",
                "gene": "PRKAR1A",
                "rationale": "Cardiac myxoma in Carney Complex causes fatal embolic stroke or sudden death if undetected",
                "consequence": "Undetected myxoma fragment embolises → fatal stroke; bilateral/valvular myxoma risk",
            },
            {
                "rule": "CRIBRIFORM-MORULAR PTC → TOTAL THYROIDECTOMY (APC)",
                "gene": "APC",
                "rationale": "CMVPTC is multifocal bilateral; hemithyroidectomy leaves disease; bilateral total resection required",
                "consequence": "Hemithyroidectomy leaves multifocal bilateral disease; recurrence in remnant lobe",
            },
            {
                "rule": "AVOID RADIATION ABSOLUTELY (TP53)",
                "gene": "TP53",
                "rationale": "TP53 LOF impairs G1/S checkpoint; ionising radiation causes secondary sarcoma/carcinoma in field",
                "consequence": "Secondary malignancy in RT field within 5-10yr; accelerated carcinogenesis",
            },
            {
                "rule": "CT CHEST ALL SIBLINGS <8yr MANDATORY (DICER1)",
                "gene": "DICER1",
                "rationale": "PPB Type I (cystic lung) progresses to solid Type III with 50% mortality; CT is only reliable detection",
                "consequence": "Missed PPB Type I → Type III solid PPB → 50% fatal; preventable with early CT",
            },
            {
                "rule": "3.5-GLAND PARATHYROID RESECTION (MEN1)",
                "gene": "MEN1",
                "rationale": "MEN1 hyperparathyroidism is multiglandular hyperplasia; single adenomectomy inevitably recurs",
                "consequence": "Single adenomectomy recurrence >80% at 10yr; multiglandular resection is curative",
            },
            {
                "rule": "MACROCEPHALY → TEST PTEN (PTEN)",
                "gene": "PTEN",
                "rationale": "Macrocephaly ≥97th percentile + any benign tumour = PHTS until disproven; thyroid USS mandatory",
                "consequence": "Missed PTEN → no thyroid/breast/endometrial surveillance; preventable malignancy",
            },
            {
                "rule": "CAFE-AU-LAIT ≥6 → NF1 THYROID SURVEILLANCE (NF1)",
                "gene": "NF1",
                "rationale": "NF1 follicular TC association 2-3x elevated; annual thyroid USS prevents late-stage diagnosis",
                "consequence": "Advanced follicular TC at diagnosis without surveillance; MPNST dominant risk also elevated",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}")
    print(f"Genes: {ov['genes']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"CR%: {ov['cr_pct']}")
    print(f"Mean age dx: {ov['mean_age_at_dx']}")
    print("\n=== BREAKDOWN KEYS ===")
    bd = generate_breakdown()
    for g in bd["genes"]:
        print(f"  {g}: n={bd['breakdown'][g]['n_patients']}, CR={bd['breakdown'][g]['cr_pct']}%")
    print("\n=== DEFINITIONS ===")
    df = generate_definitions()
    for k in list(df["definitions"].keys())[:3]:
        print(f"  {k}: {df['definitions'][k][:60]}...")
