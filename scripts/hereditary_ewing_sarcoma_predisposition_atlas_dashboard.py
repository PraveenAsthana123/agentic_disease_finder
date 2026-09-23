#!/usr/bin/env python3
"""Hereditary-Ewing-Sarcoma-Predisposition-Atlas -- Complete 8-Gene Reference
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         Ewing sarcoma 5-8% in LFS families; highest childhood sarcoma risk;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+0) .
BRCA2   (Breast cancer type 2 susceptibility protein; 3418aa; 13q12.3; AD LOF / biallelic AR FA-D1;
         Fanconi anaemia complementation group D1 (biallelic);
         ES-like round cell sarcomas in FA-D1; AVOID alkylating agents ABSOLUTELY;
         sibling donor exclusion MANDATORY;
         seed SEED_BASE+1) .
NF1     (Neurofibromin; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis 1;
         cafe-au-lait macules PATHOGNOMONIC;
         MPNST 8-13%; Ewing sarcoma 2-3x elevated; selumetinib FDA 2020 pediatric plexiform;
         seed SEED_BASE+2) .
RB1     (Retinoblastoma protein; 928aa; 13q14.2; AD LOF;
         Hereditary retinoblastoma;
         bilateral RB PATHOGNOMONIC;
         secondary Ewing sarcoma 10-15x post-RT;
         CDK4/6i INACTIVE in RB1-null; AVOID high-dose RT at sarcoma site;
         seed SEED_BASE+3) .
CDKN2A  (Cyclin-dependent kinase inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMMM / Familial melanoma;
         p16 IHC loss PATHOGNOMONIC;
         ES somatic CDKN2A deletion 20-25% — germline predisposition;
         CDK4/6i palbociclib; pancreatic adenocarcinoma 20x concurrent risk;
         seed SEED_BASE+4) .
DICER1  (DICER1 RNase III; 1922aa; 14q32.13; AD LOF;
         DICER1 syndrome;
         PPB PATHOGNOMONIC; small cell undifferentiated sarcomas 2-4x elevated;
         CT chest siblings <8yr; AVOID radiation children;
         seed SEED_BASE+5) .
SMARCB1 (SWI/SNF-related matrix-associated actin-dependent regulator subfamily B member 1; 385aa; 22q11.23; AD LOF;
         ATRT / malignant rhabdoid tumour;
         ATRT under 3yr PATHOGNOMONIC;
         INI1-IHC nuclear loss PATHOGNOMONIC;
         CIC-rearranged undifferentiated round cell sarcoma overlap;
         tazemetostat EZH2i FDA 2020;
         seed SEED_BASE+6) .
EXT1    (Exostosin glycosyltransferase 1; 746aa; 8q24.11; AD LOF;
         Hereditary multiple exostoses type 1;
         multiple osteochondromas PATHOGNOMONIC;
         secondary chondrosarcoma 1-5% lifetime;
         secondary Ewing-like sarcoma at exostosis sites rare but documented;
         MLPA mandatory large deletions 10-15%;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3318-3325)
"""
import random

SEED_BASE = 3318

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-ES-5-8pct-LFS-Families-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; nuclear transcription factor; genome guardian; "
            "  N-terminal transactivation domain (aa 1-42): MDM2 binding site; "
            "  Proline-rich region (aa 40-90): apoptosis regulation; "
            "  DNA-binding domain (aa 94-292): most hotspot mutations (R175H, G245S, R248W, R248Q, R249S, R273H, R282W); "
            "  Tetramerisation domain (aa 323-356): functional tetramer formation; "
            "  C-terminal regulatory domain (aa 356-393): post-translational modification; "
            "  TP53 activates p21 (CDKN1A) → G1/S arrest; activates PUMA/NOXA → apoptosis; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; AD LOF; ~20% de novo; virtually 100% penetrance by age 70; "
            "  Sarcoma: 50-60% of LFS cancers; ES among the dominant sarcoma types; "
            "  Ewing sarcoma in LFS: 5-8% of LFS malignancies; often aggressive anaplastic histology; "
            "  Choroid plexus carcinoma PATHOGNOMONIC LFS diagnosis; "
            "  Adrenocortical carcinoma child PATHOGNOMONIC; "
            "EWING SARCOMA IN TP53: "
            "  ES in LFS: typically young onset (median age 12-14 yr); "
            "  Anaplastic Ewing histology in TP53 germline: PATHOGNOMONIC LFS overlap; "
            "  Radiation-induced ES: post-RT field, TP53 germline ACCELERATES secondary malignancy; "
            "  AVOID RADIATION ABSOLUTELY: radiation in TP53 germline dramatically increases secondary sarcoma; "
            "  R337H Brazilian founder: high penetrance ACC + soft tissue sarcoma population; "
            "SURVEILLANCE LFS: "
            "  WBMRI: annually from diagnosis / age 18yr (Toronto protocol); "
            "  Brain MRI: annually; "
            "  Abdominopelvic US: every 3-4 months age <18yr; "
            "  Breast MRI: annually from age 20-25yr; mammogram alternate 6-monthly; "
            "TREATMENT TP53/ES: "
            "  Standard ES: VDC/IE (vincristine-doxorubicin-cyclophosphamide / ifosfamide-etoposide); "
            "  TP53 germline: AVOID radiation consolidation; consider surgery over RT; "
            "  MDM2 inhibitors (idasanutlin): under investigation TP53-wild-type tumours (not applicable germline LOF); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Li-Fraumeni Syndrome (LFS)",
        "es_risk": "5-8% of LFS cancers",
        "pathognomonic": "Choroid plexus carcinoma child, adrenocortical carcinoma child",
        "key_avoid": "RADIATION ABSOLUTELY — secondary sarcoma acceleration",
        "surveillance": "WBMRI annually + brain MRI + abdominopelvic US",
        "targeted_rx": "MDM2i investigational (idasanutlin)",
        "key_rule": "AVOID RADIATION ABSOLUTELY — germline TP53 dramatically multiplies secondary malignancy risk",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF / biallelic-AR-FA-D1 -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FA-D1-ES-Like-Round-Cell-Sarcomas-"
            "AVOID-Alkylating-ABSOLUTELY-Sibling-Exclusion-MANDATORY-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes breast cancer susceptibility protein 2: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; nuclear DNA repair scaffold protein; "
            "  N-terminal domain (aa 1-40): PALB2 binding; "
            "  BRCA2-PALB2 interaction: essential for HR pathway entry; "
            "  BRC repeats (aa 1002-2085): 8 repeats; each binds RAD51 monomer; "
            "  DNA-binding domain (aa 2402-3190): ssDNA/dsDNA junction recognition; "
            "  C-terminal RAD51-binding domain (aa 3265-3330): nuclear localisation; "
            "  BRCA2 loads RAD51 onto ssDNA resection tails → homologous recombination; "
            "FANCONI ANAEMIA COMPLEMENTATION GROUP D1 (FA-D1): "
            "  OMIM 605724; biallelic AR; most severe FA phenotype; "
            "  Biallelic BRCA2: universal Fanconi anaemia with early cancer predisposition; "
            "  ES-like round cell sarcomas: documented in FA-D1 context; "
            "  Solid tumours appear first (Wilms, brain tumours, AML, ES-like sarcoma) — before BMF; "
            "  Median solid tumour onset: 2.4 years (earlier than other FA groups); "
            "EWING SARCOMA IN BRCA2: "
            "  Monoallelic BRCA2: HBOC — ES 2-3x elevated vs population; "
            "  Biallelic BRCA2 (FA-D1): ES-like round cell sarcomas PATHOGNOMONIC FA-D1 context; "
            "  DNA crosslink repair failure → genomic instability → round cell sarcoma genesis; "
            "  HRD score elevated: platinum sensitivity; olaparib maintenance; "
            "TREATMENT BRCA2/ES: "
            "  AVOID alkylating agents ABSOLUTELY in biallelic FA-D1 (ifosfamide/cyclophosphamide myelotoxicity); "
            "  Reduced-intensity conditioning for HSCT; "
            "  Monoallelic BRCA2/ES: standard VDC/IE ± olaparib for HRD-positive; "
            "  Sibling HSCT donor exclusion: screen sibling donors for BRCA2 biallelic first; "
            "SURVEILLANCE BRCA2: "
            "  Monoallelic: annual breast MRI from 25yr; pancreatic EUS from 50yr; "
            "  Biallelic FA-D1: oncology surveillance from birth; CBC every 3 months; "
        ),
        "inheritance": "AD LOF / biallelic AR FA-D1",
        "syndrome": "Fanconi Anaemia Group D1 (biallelic) / HBOC (monoallelic)",
        "es_risk": "FA-D1: ES-like round cell sarcomas; monoallelic: ES 2-3x",
        "pathognomonic": "FA-D1 biallelic: earliest-onset solid tumours among FA groups",
        "key_avoid": "ALKYLATING AGENTS ABSOLUTELY in biallelic (ifosfamide/cyclophosphamide lethal myelotoxicity)",
        "surveillance": "Biallelic: CBC every 3 months from birth; monoallelic: breast MRI annually 25yr",
        "targeted_rx": "Olaparib (HRD-positive monoallelic ES); platinum sensitivity",
        "key_rule": "SIBLING DONOR EXCLUSION MANDATORY — test sibling before bone marrow harvest in FA-D1",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-Cafe-au-Lait-PATHOGNOMONIC-MPNST-8-13pct-ES-2-3x-"
            "Selumetinib-FDA2020-Plexiform-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes neurofibromin: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; cytoplasmic RAS-GAP (GTPase-activating protein); "
            "  N-terminal domain (aa 1-1186): regulatory scaffold; "
            "  GRD/GAP-related domain (aa 1187-1530): RAS-GTP hydrolysis catalytic domain; "
            "    NF1-GRD accelerates RAS intrinsic GTPase 1000x: RAS-GTP → RAS-GDP; "
            "  SEC14-PH domain (aa 1559-1816): lipid sensing; "
            "  C-terminal domain (aa 1816-2839): SPRED1 interaction; "
            "  NF1 LOF: sustained RAS-GTP → hyperactivation of RAF/MEK/ERK and PI3K/AKT; "
            "NEUROFIBROMATOSIS TYPE 1: "
            "  OMIM 162200; AD LOF; ~50% de novo; 1:3000 birth prevalence; "
            "  Cafe-au-lait macules: >6 CALMs diameter >5mm pre-pubertal / >15mm post-pubertal PATHOGNOMONIC; "
            "  Neurofibromas: dermal cutaneous (benign) vs plexiform (10-15% lifetime malignant transform); "
            "  Lisch nodules (iris hamartomas): >2 PATHOGNOMONIC; "
            "  Optic pathway glioma: 15% NF1 children; selumetinib FDA 2020 (Phase II SPRINT); "
            "EWING SARCOMA IN NF1: "
            "  ES risk in NF1: 2-3x elevated vs general population; "
            "  MPNST is the dominant sarcoma concern in NF1 (8-13% lifetime); "
            "  NF1-associated ES: often arising adjacent to plexiform neurofibroma; "
            "  MEK pathway activation in NF1-ES: binimetinib investigational; "
            "  AVOID conventional radiotherapy alone: MPNST and ES in NF1 field increase risk of new sarcoma; "
            "SURVEILLANCE NF1: "
            "  Annual clinical exam with NF specialist; "
            "  Annual MRI spine/paraspinal: if any new neurological symptoms; "
            "  Selumetinib FDA 2020: pediatric inoperable plexiform neurofibromas (Phase II SPRINT 66% response); "
            "  MEK inhibitor binimetinib: MPNST Phase III (NF-ClinSeq); "
            "TREATMENT NF1/ES: "
            "  Standard ES VDC/IE backbone; "
            "  NF1-associated MPNST-ES distinction critical: MPNST = resection first-line; "
            "  Selumetinib: plexiform NF1 (not ES); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Neurofibromatosis Type 1",
        "es_risk": "2-3x elevated; MPNST dominant sarcoma concern 8-13%",
        "pathognomonic": "≥6 cafe-au-lait macules, Lisch nodules, axillary/inguinal freckling",
        "key_avoid": "Conventional RT alone in NF1 field (MPNST secondary risk)",
        "surveillance": "Annual NF specialist; annual spine MRI if symptomatic",
        "targeted_rx": "Selumetinib FDA 2020 (plexiform NF); binimetinib MEKi MPNST investigational",
        "key_rule": "CAFE-AU-LAIT MACULES PATHOGNOMONIC — ≥6 CALMs mandates NF1 germline testing before sarcoma treatment",
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "pRb-105kDa-E2F-Regulator-Bilateral-RB-PATHOGNOMONIC-Secondary-ES-10-15x-Post-RT-"
            "CDK4-6i-INACTIVE-RB1-Null-AVOID-High-Dose-RT-OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 105 kDa / 13q14.2 RB1 encodes retinoblastoma protein (pRb): "
            "STRUCTURE: "
            "  928 aa / 105 kDa; nuclear tumour suppressor; master cell cycle regulator; "
            "  N-terminal domain (aa 1-378): structural scaffold; "
            "  Pocket A domain (aa 379-572): E2F binding surface A; "
            "  Spacer region (aa 573-645): caspase cleavage site D886; "
            "  Pocket B domain (aa 646-772): E2F binding surface B (A+B pocket = E2F interaction); "
            "    CDK4/CDK6-cyclinD phosphorylates pRb at S780/S795 → releases E2F transcription factors; "
            "  C-terminal domain (aa 773-928): LXCXE motif binding (viral oncoproteins); "
            "  Hypophosphorylated pRb: sequesters E2F → represses S-phase entry; "
            "HEREDITARY RETINOBLASTOMA: "
            "  OMIM 180200; AD LOF; ~40% de novo; "
            "  Bilateral retinoblastoma: presents mean age 15 months — PATHOGNOMONIC germline; "
            "  Unilateral: 25% carry germline RB1; "
            "  Second cancers post-retinoblastoma: "
            "    Osteosarcoma: 40% lifetime (irradiated survivors); "
            "    Ewing sarcoma: 10-15x elevated vs population; documented in RT field and outside RT field; "
            "    Soft tissue sarcoma: 15-20x; "
            "EWING SARCOMA IN RB1: "
            "  Radiation-induced ES: predominantly within RT field (orbit, skull, adjacent spine); "
            "  ES at RB1-haploinsufficient locus 13q14.2: del13q14.2 somatic common in sporadic ES; "
            "  CDK4/CDK6 inhibitors: INACTIVE in RB1-null ES (pRb absent → target lost); "
            "  Non-RT field ES also elevated: RB1 haploinsufficiency genomic instability predisposition; "
            "TREATMENT RB1/ES: "
            "  Standard ES VDC/IE backbone; "
            "  AVOID high-dose RT at ES site: secondary osteosarcoma/sarcoma in RT field risk; "
            "  CDK4/6i (palbociclib, ribociclib): INACTIVE when RB1-null; do NOT use; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Hereditary Retinoblastoma",
        "es_risk": "10-15x elevated; RT-field secondary ES after retinoblastoma",
        "pathognomonic": "Bilateral retinoblastoma (≤3yr) PATHOGNOMONIC germline RB1",
        "key_avoid": "High-dose RT at ES site (secondary sarcoma); CDK4/6i (INACTIVE RB1-null)",
        "surveillance": "Annual MRI spine/pelvis post-retinoblastoma; bone scan; CBC",
        "targeted_rx": "CDK4/6i INACTIVE in RB1-null (do not use); standard MAP/VDC-IE",
        "key_rule": "CDK4/6i INACTIVE in RB1-NULL ES — pRb absent means no CDK4/6i target; select alternative cell cycle approach",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-17kDa-CDK4-6-Inhibitor-p16-IHC-Loss-PATHOGNOMONIC-"
            "ES-Somatic-9p21-Deletion-20-25pct-CDK4-6i-Palbociclib-Pancreatic-20x-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 17 kDa / 9p21.3 CDKN2A encodes p16-INK4A (and p14-ARF alternate reading frame): "
            "STRUCTURE: "
            "  156 aa / 17 kDa (p16-INK4A); 132 aa / 14 kDa (p14-ARF, alternate reading frame); "
            "  p16-INK4A: 4 ankyrin repeats (aa 1-156); binds CDK4 and CDK6 active site; "
            "    p16 binding to CDK4/CDK6 → prevents cyclinD assembly → pRb remains hypophosphorylated → E2F repression; "
            "  p14-ARF: alternative reading frame product; binds MDM2 → stabilises TP53; "
            "  CDKN2A locus: frequent somatic deletion in >20 cancer types; "
            "FAMMM / FAMILIAL ATYPICAL MULTIPLE MOLE MELANOMA: "
            "  OMIM 600160; AD LOF; CDKN2A germline 25-40% familial melanoma; "
            "  Multiple atypical melanocytic nevi (dysplastic nevi): >50 total nevi, ≥1 clinically atypical; "
            "  p16 IHC nuclear loss in tumour: PATHOGNOMONIC functional CDKN2A inactivation; "
            "EWING SARCOMA IN CDKN2A: "
            "  Somatic CDKN2A deletion (9p21.3): 20-25% of sporadic Ewing sarcoma; "
            "  Germline CDKN2A LOF: predisposition to ES in context of post-irradiated sites; "
            "  CDK4/CDK6 amplification in ES subset: CDK4/6i palbociclib investigational (CDKN2A-deleted CDK4/6-amplified); "
            "  Concurrent pancreatic cancer risk 20x: critical DDx for CDKN2A families; "
            "SURVEILLANCE CDKN2A: "
            "  Annual full-body skin exam by dermoscopically trained dermatologist; "
            "  Annual pancreatic EUS/MRCP from age 40yr (or 10yr before earliest FHx pancreatic cancer); "
            "  Annual brain MRI: CDKN2A germline associated with glioblastoma; "
            "TREATMENT CDKN2A/ES: "
            "  Standard ES VDC/IE backbone; "
            "  CDK4/6i palbociclib: investigational in CDKN2A-deleted CDK4/CDK6-amplified ES; "
            "  AVOID CDK4/6i when RB1 co-deleted (INACTIVE — no pRb target); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "FAMMM / Familial Atypical Multiple Mole Melanoma",
        "es_risk": "Somatic 9p21 deletion 20-25% sporadic ES; germline predisposition post-irradiated",
        "pathognomonic": "p16-IHC nuclear loss in tumour PATHOGNOMONIC CDKN2A inactivation",
        "key_avoid": "CDK4/6i when RB1 co-deleted (INACTIVE — dual inactivation loses target)",
        "surveillance": "Annual skin exam + annual pancreatic EUS from 40yr",
        "targeted_rx": "CDK4/6i palbociclib investigational (CDK4/6-amplified/CDKN2A-deleted subset)",
        "key_rule": "PANCREATIC CANCER 20x CONCURRENT — CDKN2A germline families need annual pancreatic surveillance EUS/MRCP from age 40",
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-219kDa-RNase-III-PPB-PATHOGNOMONIC-Small-Cell-Undifferentiated-Sarcomas-2-4x-"
            "CT-Chest-Siblings-LT-8yr-AVOID-Radiation-Children-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 219 kDa / 14q32.13 DICER1 encodes DICER1 RNase III endonuclease: "
            "STRUCTURE: "
            "  1922 aa / 219 kDa; cytoplasmic RNA processing endonuclease; "
            "  PAZ domain (aa 1-1294 N-terminal region): dsRNA binding; "
            "  RNase IIIa domain (aa 1295-1664): 5-prime strand cleavage (miRNA processing); "
            "  RNase IIIb domain (aa 1700-1851): 3-prime strand cleavage (miRNA processing); "
            "    Metal ion E1705 and E1813: catalytic residues in RNase IIIb domain (hotspot somatic mutations); "
            "  dsRBD (aa 1852-1912): dsRNA binding domain; "
            "  DICER1 cleaves pre-miRNA → mature miRNA → RISC loading → target mRNA silencing; "
            "DICER1 SYNDROME: "
            "  OMIM 606241; AD LOF; ~50% de novo; "
            "  Pleuropulmonary blastoma (PPB): PATHOGNOMONIC DICER1 syndrome; 60-70% PPB carry germline DICER1; "
            "  PPB type I (cystic): < 2yr; type II (cystic-solid): 2-3yr; type III (solid): 3-4yr; "
            "  PPB progression: type I → II → III (solid) if untreated; "
            "EWING SARCOMA / SMALL CELL SARCOMA IN DICER1: "
            "  Small cell undifferentiated sarcomas in DICER1 syndrome: 2-4x elevated vs population; "
            "  Round cell sarcomas of thoracic/abdominal sites documented in DICER1 context; "
            "  miRNA pathway disruption → mesenchymal tumour predisposition; "
            "  AVOID radiation in DICER1 children: radiation-associated secondary sarcoma increased; "
            "SURVEILLANCE DICER1: "
            "  CT chest: annually until age 8yr in germline carriers and first-degree relatives <8yr; "
            "  CT chest siblings <8yr: MANDATORY — PPB lethal if missed; "
            "  Renal ultrasound: every 2-3yr (DICER1-associated Wilms tumour 2-4x); "
            "  Pelvic US: annually from age 8yr (ovarian SLCT); "
            "  Thyroid US: annually (DICER1 thyroid nodules/carcinoma); "
            "TREATMENT DICER1/ES: "
            "  Standard ES VDC/IE backbone; "
            "  AVOID radiation in children: select surgery consolidation over RT; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "DICER1 Syndrome",
        "es_risk": "Small cell undifferentiated sarcomas 2-4x; round cell thoracic/abdominal",
        "pathognomonic": "Pleuropulmonary blastoma (PPB) PATHOGNOMONIC DICER1 germline",
        "key_avoid": "Radiation in children (DICER1 secondary sarcoma risk)",
        "surveillance": "CT chest annually until 8yr; siblings <8yr CT chest MANDATORY",
        "targeted_rx": "No targeted therapy; surgery consolidation preferred over RT",
        "key_rule": "CT CHEST SIBLINGS <8yr MANDATORY — PPB is lethal if missed and CT is the surveillance tool",
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 Autosomal-Dominant-LOF -- 385aa -- "
            "INI1-44kDa-SWI-SNF-Chromatin-Remodelling-ATRT-Under-3yr-PATHOGNOMONIC-"
            "INI1-IHC-Nuclear-Loss-PATHOGNOMONIC-CIC-Sarcoma-Overlap-Tazemetostat-EZH2i-FDA2020-OMIM-601607"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / 44 kDa / 22q11.23 SMARCB1 encodes SWI/SNF-related regulator subunit (INI1/hSNF5): "
            "STRUCTURE: "
            "  385 aa / 44 kDa; nuclear chromatin remodelling complex subunit; "
            "  N-terminal domain (aa 1-80): actin-related protein binding; "
            "  RPT1 (aa 81-181) and RPT2 (aa 182-295): repeat domains; SWI/SNF complex scaffolding; "
            "  C-terminal coiled-coil domain (aa 296-385): protein-protein interaction; "
            "  SMARCB1 part of BAF (BRG1/BRM-associated factor) complex: "
            "    BAF complex: remodels nucleosomes → gene activation at tumour suppressor loci; "
            "    SMARCB1 LOF: EZH2 (PRC2 complex) unopposed → H3K27me3 silencing of tumour suppressors; "
            "    EZH2 inhibition with tazemetostat restores tumour suppressor expression; "
            "ATRT / MALIGNANT RHABDOID TUMOUR: "
            "  OMIM 601428; AD LOF; ~35% germline; "
            "  ATRT (atypical teratoid/rhabdoid tumour): CNS; under 3yr PATHOGNOMONIC germline SMARCB1; "
            "  MRT (malignant rhabdoid tumour): extracranial (kidney, soft tissue); "
            "  INI1 IHC: nuclear staining retained in normal cells; nuclear LOSS in SMARCB1-deficient tumours PATHOGNOMONIC; "
            "EWING SARCOMA / CIC-REARRANGED SARCOMA IN SMARCB1: "
            "  CIC-DUX4 rearranged undifferentiated round cell sarcoma: histologic Ewing-like; "
            "  SMARCB1 loss documented in rare CIC-rearranged sarcomas; "
            "  Round cell sarcoma spectrum: EWSR1-FLI1 (classic ES) → EWSR1-ERG → CIC-DUX4 → BCOR-CCNB3; "
            "  SMARCB1-deficient round cell sarcoma: distinct from classic ES but treated similarly; "
            "  EZH2i tazemetostat FDA 2020: epithelioid sarcoma (SMARCB1-LOF); "
            "  HSCT consolidation: malignant rhabdoid tumour (not typically ES); "
            "SURVEILLANCE SMARCB1: "
            "  Germline SMARCB1 screening: first-degree relatives of index ATRT/MRT patients; "
            "  Brain MRI every 3-6 months first 3yr life; "
            "  Renal US annually until age 5yr; "
            "TREATMENT SMARCB1/ES-LIKE: "
            "  Standard ES VDC/IE if classic ES morphology; "
            "  Tazemetostat (EZH2i): FDA 2020 epithelioid sarcoma; investigational SMARCB1-deficient ES-like; "
            "  HSCT consolidation: ATRT/MRT high-risk (not standard for ES); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Rhabdoid Tumour Predisposition Syndrome 1 (RTPS1)",
        "es_risk": "CIC-rearranged Ewing-like sarcoma; SMARCB1-deficient round cell sarcoma",
        "pathognomonic": "INI1-IHC nuclear loss PATHOGNOMONIC SMARCB1 inactivation",
        "key_avoid": "Misclassifying SMARCB1-deficient round cell sarcoma as classic ES (different prognosis/treatment)",
        "surveillance": "Brain MRI every 3-6 months first 3yr life; renal US annually <5yr",
        "targeted_rx": "Tazemetostat EZH2i FDA 2020 (epithelioid sarcoma / SMARCB1-deficient)",
        "key_rule": "INI1 IHC NUCLEAR LOSS PATHOGNOMONIC — order INI1 IHC on all paediatric round cell sarcomas to exclude SMARCB1-deficient subtype",
    },
    {
        "gene": "EXT1",
        "protein": (
            "EXT1 -- 8q24.11 Autosomal-Dominant-LOF -- 746aa -- "
            "EXT1-90kDa-Heparan-Sulfate-Polymerase-HME-Type1-Multiple-Osteochondromas-PATHOGNOMONIC-"
            "Secondary-Chondrosarcoma-1-5pct-Secondary-ES-Exostosis-Sites-MLPA-Mandatory-Large-Deletions-OMIM-133700"
        ),
        "locus": "8q24.11",
        "protein_size": (
            "746 aa / 90 kDa / 8q24.11 EXT1 encodes exostosin glycosyltransferase 1: "
            "STRUCTURE: "
            "  746 aa / 90 kDa; endoplasmic reticulum membrane glycosyltransferase; "
            "  N-terminal signal anchor (aa 1-20): ER membrane retention; "
            "  Glycosyltransferase domain (aa 21-230): GlcNAc transferase activity (EXT1 alone partial); "
            "  Coiled-coil interaction domain (aa 231-400): EXT2 heterodimerisation interface; "
            "  C-terminal domain (aa 401-746): EXT2 complementation → full polymerase activity; "
            "  EXT1-EXT2 heterocomplex: functionally complete heparan sulfate polymerase; "
            "  Heparan sulfate chains: co-receptors for FGF, BMP, Wnt, Hedgehog signalling; "
            "  EXT1 LOF: HS chain truncation → dysregulated growth factor gradients → perichondrial proliferation → osteochondromas; "
            "HEREDITARY MULTIPLE EXOSTOSES TYPE 1 (HME1): "
            "  OMIM 133700; AD LOF; ~65% of all HME; "
            "  Multiple osteochondromas (cartilage-capped bony exostoses): PATHOGNOMONIC HME; "
            "  Osteochondromas: metaphyses of long bones, ribs, pelvis, spine; "
            "  Malignant transformation: predominantly chondrosarcoma 1-5% lifetime; "
            "  Secondary Ewing sarcoma: rare but documented at exostosis sites (dedifferentiated sarcoma arising within/adjacent to exostosis); "
            "  EXT1 (8q24.11) vs EXT2 (11p12-p11): "
            "    EXT1: more severe phenotype; higher chondrosarcoma rate; "
            "    EXT2: milder phenotype; "
            "EWING SARCOMA IN EXT1: "
            "  ES at exostosis sites: documented; arises from underlying bone/cartilage; "
            "  Dedifferentiated sarcoma classification: exostosis → high-grade sarcoma (OS/ES/MFH); "
            "  MLPA mandatory: 10-15% EXT1 germline mutations are large deletions/duplications; Sanger misses; "
            "  EXT1 large deletions: more severe phenotype; higher sarcoma risk; "
            "SURVEILLANCE EXT1: "
            "  Annual radiograph survey of major exostoses; "
            "  MRI of any exostosis showing rapid growth, new pain, or cartilage cap >2cm; "
            "  CT chest: annual for spine/rib exostoses (thoracic outlet, spinal cord compression risk); "
            "TREATMENT EXT1/ES: "
            "  Surgical resection of malignant exostosis (ES/chondrosarcoma); "
            "  Standard ES VDC/IE if ES histology confirmed; "
            "  Chondrosarcoma: chemotherapy-RESISTANT — surgery is the definitive treatment; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Hereditary Multiple Exostoses Type 1 (HME1)",
        "es_risk": "Secondary ES at exostosis sites (rare, dedifferentiated sarcoma pathway)",
        "pathognomonic": "Multiple osteochondromas PATHOGNOMONIC HME; ≥2 exostoses in child = germline",
        "key_avoid": "Diagnosing chondrosarcoma from exostosis as ES (chondrosarcoma is chemo-resistant)",
        "surveillance": "Annual radiograph survey; MRI any exostosis rapid growth or cap >2cm",
        "targeted_rx": "Surgery definitive for chondrosarcoma; standard VDC/IE if ES histology confirmed",
        "key_rule": "MLPA MANDATORY — 10-15% EXT1 germline mutations are large deletions; Sanger sequencing alone MISSES them",
    },
]

TUMOUR_TYPES_BY_GENE = {
    "TP53":    ["Conventional Ewing sarcoma", "Anaplastic Ewing sarcoma", "Secondary post-RT ES", "Soft tissue ES", "Pleomorphic high-grade sarcoma"],
    "BRCA2":   ["FA-D1 round cell sarcoma", "ES-like sarcoma", "Conventional ES (monoallelic)", "HRD-positive ES", "Soft tissue round cell tumour"],
    "NF1":     ["Conventional Ewing sarcoma", "MPNST with ES-like histology", "Peripheral nerve sheath ES", "Soft tissue ES", "Para-spinal ES"],
    "RB1":     ["Radiation-induced ES (in-field)", "Conventional ES out-of-field", "Secondary sarcoma post-retinoblastoma RT", "Diaphyseal ES", "Round cell sarcoma"],
    "CDKN2A":  ["Conventional Ewing sarcoma", "Post-irradiated site ES", "CDK4/6-amplified ES", "9p21-deleted ES", "Extraskeletal ES"],
    "DICER1":  ["Small cell undifferentiated sarcoma", "Thoracic round cell sarcoma", "Abdominal ES-like sarcoma", "Pleuropulmonary blastoma (type III)", "Soft tissue round cell tumour"],
    "SMARCB1": ["CIC-rearranged round cell sarcoma", "SMARCB1-deficient sarcoma", "Epithelioid sarcoma", "MRT/ATRT-associated sarcoma", "Undifferentiated round cell sarcoma"],
    "EXT1":    ["Dedifferentiated exostosis sarcoma (ES-type)", "Secondary ES at exostosis site", "Peripheral chondrosarcoma", "Dedifferentiated chondrosarcoma", "High-grade sarcoma ex-exostosis"],
}

VARIANTS_BY_GENE = {
    "TP53":    ["p.Arg248Trp (c.742C>T)", "p.Arg273His (c.818G>A)", "p.Gly245Ser (c.733G>A)", "p.Arg175His (c.524G>A)", "p.Arg282Trp (c.844C>T)"],
    "BRCA2":   ["p.Glu1308Ter (c.3922G>T)", "p.Trp31Ter (c.92G>A)", "Large deletion exons 1-2", "p.Ile2490Val (c.7468A>G benign)", "p.Lys3326Ter (c.9976A>T)"],
    "NF1":     ["p.Arg1947Ter (c.5839C>T)", "Large deletion chromosome 17q11.2", "p.Arg440Ter (c.1318C>T)", "p.Glu1200Lys (c.3598G>A)", "Exon 17 splice variant IVS17-2A>G"],
    "RB1":     ["p.Arg787Ter (c.2359C>T)", "p.Arg556Ter (c.1666C>T)", "Large deletion exons 14-17", "p.Gln775Ter (c.2323C>T)", "Promoter methylation germline epimutation"],
    "CDKN2A":  ["p.Arg24Ter (c.70C>T)", "p.Ala148Thr (c.442G>A)", "Large deletion 9p21.3 (including CDKN2B)", "p.Pro114Leu (c.341C>T)", "Intron 2 splice variant"],
    "DICER1":  ["p.Glu1705Lys (c.5113G>A RNase IIIb hotspot)", "p.Glu1813Lys (c.5437G>A RNase IIIb hotspot)", "Frameshift exon 9", "Large deletion exon 1", "p.Arg944Gln (c.2831G>A)"],
    "SMARCB1": ["p.Arg374Ter (c.1120C>T)", "Large deletion 22q11.23", "p.Gln318Ter (c.952C>T)", "p.Arg377His (c.1130G>A)", "Exon 7 splice variant"],
    "EXT1":    ["p.Arg340Cys (c.1018C>T)", "Large deletion exons 1-3", "p.Glu246Ter (c.736G>T)", "p.Gln462Ter (c.1384C>T)", "Splice variant IVS5-1G>T"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "TP53":    ["VDC/IE (surgery consolidation; AVOID RT)", "Ifosfamide-etoposide (avoid RT)", "MDM2i investigational", "WBMRI-guided surveillance", "Surgery ± limb salvage"],
    "BRCA2":   ["Reduced alkylator VDC/IE (FA-D1)", "Olaparib maintenance (HRD+)", "Platinum-based regimen (HRD sensitivity)", "HSCT conditioning dose-reduced (biallelic)", "Standard VDC/IE (monoallelic)"],
    "NF1":     ["Standard VDC/IE", "Selumetinib (plexiform NF, not ES)", "Binimetinib MEKi investigational", "Surgery ± limb salvage", "MPNST: resection first-line"],
    "RB1":     ["Standard VDC/IE (avoid high-dose RT)", "Surgery consolidation preferred", "Doxorubicin-cisplatin (no CDK4/6i)", "Bone scintigraphy surveillance", "Limb salvage surgery"],
    "CDKN2A":  ["Standard VDC/IE", "Palbociclib investigational (CDK4/6-amplified subset)", "Pancreatic surveillance EUS annually", "Dermatology surveillance annual", "Surgery ± radiation (if no RT contraindication)"],
    "DICER1":  ["Standard VDC/IE (surgery consolidation; AVOID RT children)", "Ifosfamide-etoposide", "Vincristine-actinomycin-cyclophosphamide", "CT chest surveillance annually <8yr", "Thyroid surveillance"],
    "SMARCB1": ["Standard VDC/IE (ES morphology)", "Tazemetostat EZH2i FDA 2020 (epithelioid/SMARCB1-deficient)", "HSCT consolidation (ATRT/MRT high-risk)", "Brain MRI surveillance 3-6mo <3yr", "INI1-IHC guided diagnosis"],
    "EXT1":    ["Standard VDC/IE (ES histology confirmed)", "Surgical excision (exostosis with malignant transformation)", "Chemo-RESISTANT if chondrosarcoma component", "MLPA mutation analysis", "Annual radiograph survey"],
}

SURVEILLANCE_BY_GENE = {
    "TP53":    ["WBMRI annually from diagnosis", "Brain MRI annually", "Abdominopelvic US every 3-4 months <18yr", "Breast MRI annually from 20-25yr", "Annual clinical exam NCI/LFS specialist"],
    "BRCA2":   ["CBC every 3 months from birth (biallelic FA-D1)", "Annual breast MRI from 25yr (monoallelic)", "Annual pancreatic EUS from 50yr", "Annual ovarian US (monoallelic)", "Bone marrow aspirate annually (FA-D1)"],
    "NF1":     ["Annual NF specialist clinical exam", "Annual MRI spine if symptomatic", "Ophthalmology annually <7yr (OPG)", "Annual BP monitoring", "Annual cognitive/learning assessment child"],
    "RB1":     ["Annual MRI spine/pelvis post-retinoblastoma", "Bone scan annually post-RT", "CBC annually", "Annual ophthalmology (fellow eye)", "Annual clinical exam oncology"],
    "CDKN2A":  ["Annual full-body skin exam dermoscopy", "Annual pancreatic EUS/MRCP from 40yr", "Annual brain MRI (glioblastoma risk)", "Annual ophthalmology (uveal melanoma risk)", "Annual clinical genetics review"],
    "DICER1":  ["CT chest annually until age 8yr", "Siblings <8yr CT chest MANDATORY", "Renal US every 2-3yr", "Pelvic US annually from 8yr (SLCT)", "Thyroid US annually"],
    "SMARCB1": ["Brain MRI every 3-6 months first 3yr life", "Renal US annually until age 5yr", "Annual spinal MRI germline carriers", "INI1-IHC on all round cell sarcoma specimens", "Annual clinical genetics review"],
    "EXT1":    ["Annual radiograph survey major exostoses", "MRI any exostosis rapid growth / cap >2cm", "CT chest annually (thoracic exostoses)", "Annual orthopaedic review", "MLPA analysis all first-degree relatives"],
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

    age_at_dx = rng.randint(3, 28)
    tumour_type = rng.choice(tumour_types)
    variant = rng.choice(variants)
    treatment = rng.choice(treatments)
    cr = rng.random() < 0.62
    radiation = rng.random() < (0.12 if gene in ("TP53", "BRCA2", "DICER1") else 0.35)
    relapse = rng.random() < 0.28 if cr else rng.random() < 0.55

    return {
        "patient_id": f"HES-{gene}-{patient_idx:03d}",
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
        "atlas": "Hereditary-Ewing-Sarcoma-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Ewing Sarcoma Predisposition Reference",
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "total_patients": n,
        "cr_pct": round(100 * cr_n / n, 1),
        "radiation_pct": round(100 * rad_n / n, 1),
        "relapse_pct": round(100 * relapse_n / n, 1),
        "mean_age_at_dx": mean_age,
        "genes": _GENE_LIST,
        "gene_summary": gene_summary,
        "key_rules": [
            "AVOID RADIATION ABSOLUTELY in TP53 germline — secondary sarcoma acceleration",
            "AVOID ALKYLATING AGENTS ABSOLUTELY in BRCA2 biallelic FA-D1 (ifosfamide/cyclophosphamide lethal myelotoxicity)",
            "SIBLING DONOR EXCLUSION MANDATORY in BRCA2 FA-D1 — test sibling before harvest",
            "CDK4/6i INACTIVE in RB1-null ES — select alternative cell cycle target",
            "MLPA MANDATORY for EXT1 — 10-15% large deletions missed by Sanger",
            "CT CHEST SIBLINGS <8yr MANDATORY in DICER1 — PPB is lethal if missed",
            "INI1 IHC on ALL paediatric round cell sarcomas — exclude SMARCB1-deficient subtype",
            "CAFE-AU-LAIT MACULES PATHOGNOMONIC — ≥6 CALMs mandates NF1 testing before sarcoma treatment",
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
            "es_risk": gene_info["es_risk"],
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
        "atlas": "Hereditary-Ewing-Sarcoma-Predisposition-Atlas",
        "definitions": {
            "ewing_sarcoma": (
                "Ewing sarcoma (ES): malignant round cell bone/soft tissue tumour; "
                "EWSR1-FLI1 fusion in 85% sporadic cases (somatic); "
                "median age 15yr; most common primary bone malignancy in adolescents; "
                "diaphysis of long bones most common site; VDRC pattern characteristic MRI; "
                "histology: monomorphic small blue round cells; CD99 positive (membrane); "
                "EWSR1 break-apart FISH: diagnostic confirmation; "
                "5-year OS 70% localised / 30% metastatic."
            ),
            "lfs": (
                "Li-Fraumeni Syndrome (LFS): TP53 germline LOF; AD; virtually 100% penetrance by age 70; "
                "classic triad: sarcoma + breast cancer + brain tumour; "
                "ES 5-8% of LFS malignancies; AVOID radiation ABSOLUTELY; "
                "WBMRI Toronto protocol annually."
            ),
            "fa_d1": (
                "Fanconi Anaemia Group D1 (FA-D1): biallelic BRCA2 LOF; AR; most severe FA phenotype; "
                "solid tumours before BMF; Wilms, medulloblastoma, ES-like sarcoma, AML; "
                "AVOID alkylating agents; sibling donor exclusion mandatory before HSCT harvest."
            ),
            "nf1": (
                "Neurofibromatosis Type 1 (NF1): NF1 germline LOF; AD; 1:3000 birth prevalence; "
                "cafe-au-lait macules >6 PATHOGNOMONIC; MPNST 8-13% lifetime; ES 2-3x elevated; "
                "selumetinib FDA 2020 plexiform neurofibromas pediatric."
            ),
            "hme1": (
                "Hereditary Multiple Exostoses Type 1 (HME1): EXT1 germline LOF; AD; 65% of all HME; "
                "multiple osteochondromas PATHOGNOMONIC; chondrosarcoma 1-5% lifetime; "
                "secondary ES at exostosis sites documented; MLPA mandatory (10-15% large deletions)."
            ),
            "dicer1_syndrome": (
                "DICER1 Syndrome: DICER1 germline LOF; AD; ~50% de novo; "
                "PPB PATHOGNOMONIC; small cell undifferentiated sarcomas 2-4x elevated; "
                "CT chest siblings <8yr MANDATORY; AVOID radiation children."
            ),
            "rtps1": (
                "Rhabdoid Tumour Predisposition Syndrome 1 (RTPS1): SMARCB1 germline LOF; AD; ~35% germline; "
                "ATRT under 3yr PATHOGNOMONIC; INI1-IHC nuclear loss PATHOGNOMONIC; "
                "CIC-rearranged sarcoma overlap; tazemetostat EZH2i FDA 2020."
            ),
            "ini1_ihc": (
                "INI1 IHC (immunohistochemistry): nuclear staining in normal cells; "
                "NUCLEAR LOSS in SMARCB1-deficient tumours PATHOGNOMONIC for SMARCB1 inactivation; "
                "order INI1 IHC on all paediatric round cell sarcomas; "
                "SMARCB1-deficient ES-like sarcoma: different prognosis/treatment from classic ES."
            ),
            "cascade_testing": (
                "CASCADE TESTING — Hereditary Ewing Sarcoma: "
                "1. TP53: all first-degree relatives of LFS index case; WBMRI annual; AVOID RT; "
                "2. BRCA2 biallelic (FA-D1): test sibling donors BEFORE harvest; AVOID alkylating; "
                "3. NF1: clinical exam; annual NF specialist; selumetinib if plexiform; "
                "4. RB1: post-retinoblastoma annual bone imaging; CDK4/6i INACTIVE RB1-null; "
                "5. CDKN2A: annual skin exam; annual pancreatic EUS from 40yr; "
                "6. DICER1: CT chest siblings <8yr MANDATORY; AVOID RT children; "
                "7. SMARCB1: INI1-IHC all paediatric round cell sarcoma; brain MRI 3-6mo <3yr; "
                "8. EXT1: MLPA mandatory; annual radiograph survey exostoses; MRI cap >2cm."
            ),
        },
        "key_clinical_rules": [
            {
                "rule": "AVOID RADIATION ABSOLUTELY",
                "gene": "TP53",
                "rationale": "TP53 germline radiation dramatically accelerates secondary sarcoma; surgery consolidation preferred",
                "consequence": "Secondary sarcoma within RT field; accelerated carcinogenesis",
            },
            {
                "rule": "AVOID ALKYLATING AGENTS ABSOLUTELY (biallelic FA-D1)",
                "gene": "BRCA2",
                "rationale": "Ifosfamide/cyclophosphamide cause lethal myelotoxicity in FA-D1 biallelic BRCA2",
                "consequence": "Fatal bone marrow aplasia; reduced-intensity regimen mandatory",
            },
            {
                "rule": "SIBLING DONOR EXCLUSION MANDATORY",
                "gene": "BRCA2",
                "rationale": "Sibling may also carry biallelic BRCA2 (FA-D1); harvesting their marrow risks donor sarcoma",
                "consequence": "Donor develops FA-D1 cancer; use unrelated donor or haploidentical with reduced conditioning",
            },
            {
                "rule": "CDK4/6i INACTIVE in RB1-null ES",
                "gene": "RB1",
                "rationale": "pRb is the CDK4/6i target; RB1-null tumours lack pRb → CDK4/6i has no target",
                "consequence": "Wasted therapy; select alternative cell cycle target (CDK2i, AURKA)",
            },
            {
                "rule": "MLPA MANDATORY",
                "gene": "EXT1",
                "rationale": "10-15% EXT1 germline mutations are large deletions/duplications; Sanger sequencing misses them",
                "consequence": "Missed diagnosis; undetected at-risk relatives; MLPA required for complete testing",
            },
            {
                "rule": "CT CHEST SIBLINGS <8yr MANDATORY",
                "gene": "DICER1",
                "rationale": "PPB (pleuropulmonary blastoma) is lethal if missed; CT is the only reliable detection method",
                "consequence": "Type I PPB → type III solid PPB with 50% mortality if surveillance missed",
            },
            {
                "rule": "INI1 IHC on ALL paediatric round cell sarcomas",
                "gene": "SMARCB1",
                "rationale": "SMARCB1-deficient round cell sarcoma is Ewing-like but has different prognosis and treatment",
                "consequence": "Misclassified as classic ES; different targeted therapy (tazemetostat) missed",
            },
            {
                "rule": "CAFE-AU-LAIT MACULES PATHOGNOMONIC",
                "gene": "NF1",
                "rationale": "≥6 CALMs mandate NF1 testing; ES treatment plan differs if NF1 is co-diagnosis",
                "consequence": "MPNST vs ES distinction critical (MPNST = surgery first-line, ES = chemotherapy first-line)",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}")
    print(f"Genes: {ov['genes']}")
    print(f"CR%: {ov['cr_pct']}")
    print(f"Mean age dx: {ov['mean_age_at_dx']}")
    print("\n=== BREAKDOWN KEYS ===")
    bd = generate_breakdown()
    for g in bd["genes"]:
        print(f"  {g}: n={bd['breakdown'][g]['n_patients']}, CR={bd['breakdown'][g]['cr_pct']}%")
    print("\n=== DEFINITIONS ===")
    df = generate_definitions()
    for k in list(df["definitions"].keys())[:3]:
        print(f"  {k}: {df['definitions'][k][:80]}...")
