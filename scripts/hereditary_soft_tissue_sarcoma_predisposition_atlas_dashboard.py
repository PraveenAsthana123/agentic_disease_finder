#!/usr/bin/env python3
"""Hereditary-Soft-Tissue-Sarcoma-Predisposition-Atlas -- Complete 8-Gene Reference
TP53   (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome — sarcoma #1 MOST COMMON cancer in LFS pediatric;
         rhabdomyosarcoma/UPS/leiomyosarcoma;
         AVOID RADIATION ABSOLUTELY; WB-MRI Toronto protocol;
         seed SEED_BASE+0) .
NF1    (Neurofibromin RAS-GAP; 2839aa; 17q11.2; AD LOF;
         MPNST 8-13% HIGHEST hereditary STS; plexiform neurofibroma → MPNST;
         selumetinib FDA2020 inoperable plexiform NF;
         MPNST = sarcoma doxorubicin/ifosfamide NOT glioma regimen;
         seed SEED_BASE+1) .
RB1    (pRb E2F regulator; 928aa; 13q14.2; AD LOF;
         secondary STS at radiation field leiomyosarcoma 40x post-RT;
         AVOID high-dose radiation RB1 germline;
         CDK4-6i INACTIVE in RB1-null STS;
         bilateral retinoblastoma PATHOGNOMONIC;
         seed SEED_BASE+2) .
SMARCB1 (INI1/BAF47 SWI/SNF subunit; 385aa; 22q11.23; AD LOF;
          Rhabdoid Tumour Predisposition Syndrome type 1 (RTPS1);
          MRT/ATRT infants PATHOGNOMONIC;
          epithelioid sarcoma INI1 loss IHC PATHOGNOMONIC;
          Tazemetostat FDA2020 EZH2 inhibitor epithelioid sarcoma;
          seed SEED_BASE+3) .
DICER1 (RNase III endoribonuclease; 1922aa; 14q32.13; AD LOF;
         PPB type I/II/III PATHOGNOMONIC;
         embryonal rhabdomyosarcoma cervix/uterus PATHOGNOMONIC;
         annual low-dose chest CT birth to 8yr MANDATORY;
         seed SEED_BASE+4) .
BRCA2  (HR scaffold/FANCD1; 3418aa; 13q12.3; AD LOF;
         biallelic FA-D1 embryonal RMS PATHOGNOMONIC infancy;
         AVOID alkylating agents ABSOLUTELY in FA-D1;
         olaparib FDA2020 HRD-positive STS monoallelic;
         seed SEED_BASE+5) .
EXT1   (Exostosin glycosyltransferase 1; 746aa; 8q24.11; AD LOF;
         Hereditary Multiple Exostoses type 1 osteochondromas PATHOGNOMONIC;
         secondary chondrosarcoma 1-5% lifetime;
         cap >2cm MRI = malignant transformation URGENT excision;
         MLPA mandatory large deletions 30%;
         seed SEED_BASE+6) .
SMARCA4 (BRG1 SWI/SNF ATPase catalytic; 1647aa; 19p13.2; AD LOF;
          SCCOHT SMARCA4/BRG1 PATHOGNOMONIC;
          MRT/ATRT SMARCA4-deficient;
          BRG1 IHC loss = SMARCA4 deficient;
          tazemetostat EZH2 inhibitor clinical trials;
          seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3438-3445)
"""
import random

SEED_BASE = 3438

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-Guardian-of-Genome-"
            "LFS-Li-Fraumeni-Syndrome-Sarcoma-Number1-Pediatric-RMS-UPS-LMS-"
            "AVOID-RADIATION-ABSOLUTELY-WB-MRI-Toronto-Protocol-"
            "OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 soft tissue sarcoma molecular context: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; N-terminal transactivation domains (TAD1 aa 1-40, TAD2 aa 40-67); "
            "  Proline-rich domain (aa 67-98); DNA-binding domain (DBD aa 94-292) — hotspot mutations; "
            "  Tetramerisation domain (aa 325-356); C-terminal regulatory domain; "
            "  p53 activates CDKN1A/p21, MDM2, PUMA, NOXA on genotoxic stress — senescence/apoptosis; "
            "  TP53 LOF → cell cycle arrest abolished → genome instability → sarcoma transformation; "
            "CANCER RISKS (SARCOMA FOCUS): "
            "  Sarcoma: #1 MOST COMMON cancer in LFS pediatric patients — 30% lifetime risk; "
            "  Rhabdomyosarcoma (RMS): embryonal and alveolar subtypes in LFS children; "
            "  Undifferentiated pleomorphic sarcoma (UPS): adults with TP53 germline; "
            "  Leiomyosarcoma (LMS): uterine and retroperitoneal LMS elevated LFS adults; "
            "  R248W / R273H dominant-negative hotspots: 50% pediatric LFS sarcoma — highest frequency; "
            "  LFS spectrum lifetime: sarcoma 30%, breast 54%, brain 20-26%, ACC 5-10%; "
            "KEY MANAGEMENT: "
            "  AVOID RADIATION ABSOLUTELY — radiation in LFS → catastrophic secondary sarcoma in field; "
            "  WB-MRI Toronto protocol: annual whole-body MRI + annual brain MRI MANDATORY; "
            "  Proton therapy if radiation unavoidable — multidisciplinary decision (rarely justified); "
            "  Surgery first-line for all TP53-associated STS — avoid neoadjuvant radiation; "
            "  Cascade germline TP53 testing all first-degree relatives of TP53/LFS sarcoma patient"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) — sarcoma #1 pediatric; breast, brain, ACC, leukemia multi-cancer",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo ~25% LFS; dominant negative R248W/R273H",
        "sarcoma_risk": "Sarcoma 30% lifetime #1 pediatric LFS; RMS, UPS, LMS; R248W/R273H 50% pediatric LFS sarcoma",
        "pathognomonic": "LFS multi-cancer cluster PATHOGNOMONIC; pediatric sarcoma + family sarcoma/breast/brain = LFS; radiation sensitivity",
        "key_avoid": "AVOID RADIATION ABSOLUTELY — radiation in LFS = catastrophic secondary sarcoma; WB-MRI Toronto protocol MANDATORY annual; surgery first-line for all LFS sarcoma",
        "key_rule": "TP53/LFS: AVOID RADIATION ABSOLUTELY. Annual WB-MRI Toronto protocol. Sarcoma #1 pediatric cancer. R248W/R273H dominant-negative hotspots. Cascade testing first-degree relatives.",
        "surveillance": "Annual whole-body MRI Toronto protocol from diagnosis MANDATORY; annual brain MRI; annual breast MRI women from age 20yr; annual abdominal US + blood count; cascade TP53 germline first-degree relatives; pre-surgery TP53 germline testing all pediatric sarcoma patients",
        "targeted_rx": "Surgery first-line all TP53/LFS sarcoma — avoid neoadjuvant radiation; doxorubicin+ifosfamide adult STS (UPS/LMS) first-line; vincristine+actinomycin D+cyclophosphamide (VAC) embryonal RMS children; pembrolizumab hypermutant TP53-associated STS high TMB; trabectedin retroperitoneal LMS second-line; proton therapy if radiation unavoidable (multidisciplinary); cascade TP53 germline all first-degree relatives",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-Ras-GTPase-Accelerating-Protein-"
            "MPNST-8-13pct-HIGHEST-Hereditary-STS-Plexiform-Neurofibroma-"
            "Selumetinib-FDA2020-Inoperable-Plexiform-NF-"
            "OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 soft tissue sarcoma molecular context: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; GRD domain (RAS-GAP, aa 1198-1530) — accelerates RAS GTP hydrolysis; "
            "  PH domain; SEC14 domain; ARM repeats; C-terminal PDZ-binding; "
            "  NF1 LOF → RAS-GTP accumulation → MAPK/ERK + PI3K/AKT hyperactivation; "
            "  Plexiform neurofibroma: benign NF1 tumour — MRI surveillance for MPNST transformation; "
            "CANCER RISKS (SARCOMA FOCUS): "
            "  MPNST (Malignant Peripheral Nerve Sheath Tumour): 8-13% lifetime HIGHEST hereditary STS risk; "
            "  MPNST = SARCOMA — doxorubicin/ifosfamide chemotherapy, NOT immunotherapy, NOT glioma regimen; "
            "  Plexiform neurofibroma → MPNST transformation: MRI surveillance MANDATORY; "
            "  MPNST 5-year survival <50% — aggressive histology, surgery with wide margins first-line; "
            "  NF1 sarcoma: internal plexiform NF = highest MPNST risk; size >3cm + growth = alert; "
            "KEY MANAGEMENT: "
            "  Annual whole-body MRI (MPNST surveillance) MANDATORY; "
            "  Selumetinib FDA2020 MEK inhibitor — inoperable plexiform neurofibromas (NOT MPNST treatment); "
            "  MPNST: wide surgical resection + doxorubicin+ifosfamide (DO NOT use selumetinib for MPNST); "
            "  Annual blood pressure check; dermatology café-au-lait annual; "
            "  Café-au-lait 6+ macules PATHOGNOMONIC NF1; Lisch nodules PATHOGNOMONIC NF1"
        ),
        "syndrome": "Neurofibromatosis type 1 (NF1) — MPNST 8-13% HIGHEST hereditary STS; plexiform neurofibroma; optic glioma",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo 50% cases; haploinsufficiency",
        "sarcoma_risk": "MPNST 8-13% HIGHEST hereditary STS; plexiform neurofibroma → MPNST; doxorubicin/ifosfamide MPNST sarcoma",
        "pathognomonic": "Café-au-lait macules 6+ PATHOGNOMONIC; Lisch nodules PATHOGNOMONIC; MPNST from plexiform NF; NF1 MPNST = sarcoma NOT glioma",
        "key_avoid": "Do NOT treat MPNST as glioma — MPNST = sarcoma (doxorubicin+ifosfamide). Do NOT use selumetinib for MPNST. Do NOT miss annual whole-body MRI (MPNST risk from plexiform NF)",
        "key_rule": "NF1: annual whole-body MRI MANDATORY (MPNST 8-13%). MPNST = SARCOMA doxorubicin/ifosfamide NOT glioma treatment. Selumetinib FDA2020 plexiform NF NOT MPNST. Café-au-lait 6+ PATHOGNOMONIC. Wide surgical margins MPNST.",
        "surveillance": "Annual whole-body MRI (plexiform NF + MPNST surveillance); annual ophthalmology from age 1yr (optic glioma); annual blood pressure check; selumetinib eligibility if plexiform NF symptomatic/inoperable (not MPNST); annual dermatology café-au-lait + neurofibromas; cascade NF1 germline first-degree relatives",
        "targeted_rx": "Selumetinib FDA2020 MEK inhibitor inoperable plexiform neurofibromas (NOT MPNST); doxorubicin+ifosfamide MPNST first-line sarcoma regimen; trabectedin MPNST second-line; pazopanib MPNST third-line; wide surgical excision MPNST (negative margins critical — R0 resection); radiation MPNST: post-operative RT if R1/R2 (not as primary — multidisciplinary); gemcitabine+docetaxel MPNST refractory",
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "pRb-105kDa-E2F-Regulator-G1-S-Checkpoint-"
            "Secondary-STS-Radiation-Field-Leiomyosarcoma-40x-Post-RT-"
            "AVOID-High-Dose-Radiation-RB1-Germline-"
            "Bilateral-Retinoblastoma-PATHOGNOMONIC-CDK4-6i-INACTIVE-RB1-Null-"
            "OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 105 kDa / 13q14.2 RB1 soft tissue sarcoma molecular context: "
            "STRUCTURE: "
            "  928 aa / 105 kDa; pocket domain A (aa 379-572) + B (aa 646-771) — E2F binding; "
            "  N-terminal domain; spacer region; C-terminal domain; "
            "  pRb binds and represses E2F transcription factors — arrests G1→S; "
            "  pRb phosphorylation (CDK4/6-cyclinD → CDK2-cyclinE) → E2F release → S-phase entry; "
            "  RB1 LOF → E2F constitutively active → uncontrolled proliferation; "
            "  CDK4/6 inhibitors INACTIVE in RB1-null STS (palbociclib/ribociclib/abemaciclib useless); "
            "CANCER RISKS (SARCOMA FOCUS): "
            "  Secondary STS at radiation field: leiomyosarcoma 40x lifetime risk post-RT; "
            "  Secondary STS is highest secondary cancer risk of any hereditary syndrome post-radiation; "
            "  Secondary STS sites: orbit/head-neck (post-retinoblastoma RT), pelvis, extremity; "
            "  Osteosarcoma: 30% lifetime risk (primary bone sarcoma — not soft tissue but same RB1 path); "
            "  Bilateral retinoblastoma PATHOGNOMONIC: germline RB1 — lifetime sarcoma surveillance mandatory; "
            "KEY MANAGEMENT: "
            "  AVOID high-dose radiation at any site if RB1 germline — secondary sarcoma LETHAL; "
            "  Annual MRI surveillance of prior radiation fields MANDATORY (secondary STS detection); "
            "  CDK4-6 inhibitors (palbociclib, ribociclib) INACTIVE in RB1-null STS — do not use; "
            "  Bilateral retinoblastoma → RB1 germline test → annual STS surveillance for life; "
            "  Systemic therapy STS: doxorubicin+ifosfamide first-line (RB1-null has no targeted therapy)"
        ),
        "syndrome": "Hereditary retinoblastoma (RB1 germline) — bilateral RB PATHOGNOMONIC; secondary STS post-RT; osteosarcoma",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo ~40%; haploinsufficiency; somatic LOH 13q14",
        "sarcoma_risk": "Secondary STS post-radiation 40x; leiomyosarcoma at radiation field; osteosarcoma 30%; CDK4-6i INACTIVE RB1-null",
        "pathognomonic": "Bilateral retinoblastoma PATHOGNOMONIC RB1 germline; secondary sarcoma in radiation field PATHOGNOMONIC; LMS post-orbit RT",
        "key_avoid": "AVOID high-dose radiation RB1 germline — secondary sarcoma LETHAL. CDK4-6i (palbociclib/ribociclib/abemaciclib) INACTIVE in RB1-null STS — do NOT prescribe. Annual MRI radiation field surveillance MANDATORY",
        "key_rule": "RB1: bilateral retinoblastoma PATHOGNOMONIC → germline test. AVOID radiation RB1 germline (secondary STS 40x). CDK4-6i INACTIVE RB1-null. Annual MRI radiation field surveillance. Doxorubicin+ifosfamide LMS first-line.",
        "surveillance": "Annual MRI prior radiation fields from diagnosis MANDATORY; annual ophthalmology (bilateral RB survivors); annual whole-body MRI RB1 germline adults (secondary STS + osteosarcoma); annual chest X-ray (pulmonary metastases sarcoma); cascade RB1 germline first-degree relatives; ophthalmology newborn relatives RB1",
        "targeted_rx": "Doxorubicin+ifosfamide LMS/secondary STS RB1-null first-line; gemcitabine+docetaxel LMS second-line; trabectedin LPS/LMS third-line; AVOID CDK4-6i (palbociclib/ribociclib/abemaciclib — inactive RB1-null; prescribing is a harm); enucleation retinoblastoma (intraocular salvage: intra-arterial chemotherapy carboplatin/melphalan); systemic carboplatin+vincristine+etoposide metastatic retinoblastoma; AVOID external beam radiation orbit (secondary STS 40x)",
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 Autosomal-Dominant-LOF -- 385aa -- "
            "INI1-BAF47-47kDa-SWI-SNF-Core-Subunit-"
            "RTPS1-Rhabdoid-Tumour-Predisposition-Syndrome-Type1-"
            "MRT-ATRT-Infants-Under-3yr-PATHOGNOMONIC-"
            "Epithelioid-Sarcoma-INI1-Loss-IHC-PATHOGNOMONIC-"
            "Tazemetostat-FDA2020-EZH2-Inhibitor-OMIM-601607"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / 47 kDa / 22q11.23 SMARCB1 soft tissue sarcoma molecular context: "
            "STRUCTURE: "
            "  385 aa / 47 kDa; N-terminal domain (aa 1-50); RPT1 repeat (aa 93-163); "
            "  RPT2 repeat (aa 170-239); C-terminal coiled-coil (aa 340-385); "
            "  SMARCB1/INI1 is core structural subunit of SWI/SNF chromatin remodelling complex; "
            "  SMARCB1 LOF → SWI/SNF complex destabilised → EZH2 (PRC2) unopposed → H3K27me3 increase; "
            "  EZH2 inhibitor tazemetostat FDA2020: restores chromatin balance in SMARCB1-deficient tumours; "
            "CANCER RISKS (SARCOMA FOCUS): "
            "  Malignant rhabdoid tumour (MRT): infants under 3yr PATHOGNOMONIC — RTPS1; retroperitoneum/kidney; "
            "  ATRT (atypical teratoid rhabdoid tumour): SMARCB1-deficient brain tumour infants — high fatality; "
            "  Epithelioid sarcoma (proximal/axial type): INI1/SMARCB1 nuclear loss IHC PATHOGNOMONIC; "
            "  Epithelioid sarcoma: young adults extremity (distal) or proximal/axial (trunk); "
            "  INI1 IHC nuclear loss: universal diagnostic marker — loss = SMARCB1 deficient (ES/MRT/ATRT); "
            "KEY MANAGEMENT: "
            "  Tazemetostat FDA2020: EZH2 inhibitor — epithelioid sarcoma INI1-loss APPROVED indication; "
            "  AVOID radiation infants with rhabdoid (MRT/ATRT = rapidly fatal before RT can help); "
            "  INI1 IHC: universal screening in ES/ATRT/MRT — loss = diagnostic, predicts tazemetostat eligibility; "
            "  Germline SMARCB1 testing: any MRT/ATRT infant under 3yr — 33-40% have germline mutation; "
            "  Sibling surveillance MRI brain+spine+abdomen: any germline SMARCB1 family"
        ),
        "syndrome": "Rhabdoid Tumour Predisposition Syndrome type 1 (RTPS1) — MRT/ATRT infants; epithelioid sarcoma INI1-loss",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo ~60% RTPS1; haploinsufficiency; somatic LOH",
        "sarcoma_risk": "MRT infants under 3yr PATHOGNOMONIC RTPS1; epithelioid sarcoma INI1 loss PATHOGNOMONIC; tazemetostat FDA2020 ES",
        "pathognomonic": "MRT infants PATHOGNOMONIC RTPS1; INI1/SMARCB1 nuclear IHC loss PATHOGNOMONIC epithelioid sarcoma; ATRT infants PATHOGNOMONIC",
        "key_avoid": "AVOID radiation infants MRT/ATRT — tumour rapidly fatal; radiation adds toxicity without benefit. Do NOT miss INI1 IHC — universal screen in ES/ATRT/MRT. Germline SMARCB1 test all infant MRT/ATRT",
        "key_rule": "SMARCB1: INI1 IHC loss PATHOGNOMONIC ES/MRT/ATRT. Tazemetostat FDA2020 epithelioid sarcoma INI1-loss. MRT infants <3yr PATHOGNOMONIC RTPS1 → germline test. AVOID radiation infants rhabdoid. Sibling surveillance MRI.",
        "surveillance": "MRI brain+spine+abdomen infants germline SMARCB1 from birth; INI1 IHC all epithelioid sarcoma/MRT/ATRT; germline SMARCB1 test all MRT/ATRT infant index cases; cascade germline first-degree relatives; annual abdominal MRI RTPS1 germline carriers childhood; tazemetostat eligibility if epithelioid sarcoma INI1 loss confirmed",
        "targeted_rx": "Tazemetostat FDA2020 (EZH2 inhibitor) epithelioid sarcoma INI1-loss (disease control rate ~26%); doxorubicin+ifosfamide epithelioid sarcoma/MRT first-line; high-dose chemotherapy + autologous SCT infant MRT (experimental baby-brain protocol); AVOID radiation infants rhabdoid; complete surgical resection epithelioid sarcoma (wide margins critical); carboplatin+etoposide+vincristine ATRT infants; pembrolizumab INI1-deficient STS (emerging data)",
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-218kDa-RNase-III-Endoribonuclease-miRNA-Processing-"
            "PPB-Pleuropulmonary-Blastoma-Type-I-II-III-PATHOGNOMONIC-"
            "Embryonal-RMS-Cervix-Uterus-Vagina-PATHOGNOMONIC-"
            "Annual-Low-Dose-CT-Birth-to-8yr-MANDATORY-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 218 kDa / 14q32.13 DICER1 soft tissue sarcoma molecular context: "
            "STRUCTURE: "
            "  1922 aa / 218 kDa; N-terminal helicase domain; PAZ domain (aa 860-1008); "
            "  RNase IIIa domain (aa 1285-1450); RNase IIIb domain (aa 1454-1846) — dices pre-miRNA; "
            "  dsRBD (double-strand RNA binding domain, C-terminal); "
            "  DICER1 LOF → miRNA biogenesis impaired → oncogenic mRNAs de-repressed; "
            "  RNase IIIb hotspot mutations (E1705K, D1709N): somatic hits in DICER1 syndrome tumours; "
            "CANCER RISKS (SARCOMA FOCUS): "
            "  PPB (pleuropulmonary blastoma): TYPE I/II/III PATHOGNOMONIC — #1 presenting tumour DICER1; "
            "  PPB type I (cystic, <2yr): lowest mortality; type II/III (solid component): higher mortality; "
            "  Embryonal rhabdomyosarcoma (eRMS): cervix/uterus/vagina/salpinx = DICER1 PATHOGNOMONIC; "
            "  eRMS DICER1 uterine-cervical: young women — botryoid morphology DICER1 marker; "
            "  Wilms tumour 3x; cystic nephroma PATHOGNOMONIC; nasal chondromesenchymal hamartoma; "
            "KEY MANAGEMENT: "
            "  Annual low-dose chest CT from BIRTH to 8yr MANDATORY (PPB detection siblings and index); "
            "  AVOID radiation where possible PPB (young patients already vulnerable); "
            "  Gynaecological surveillance: annual pelvic exam from puberty (eRMS cervix/uterus); "
            "  Renal ultrasound annually from birth (Wilms tumour + cystic nephroma); "
            "  DICER1 germline testing: PPB type I/II/III, eRMS cervix/uterus, cystic nephroma any age"
        ),
        "syndrome": "DICER1 syndrome — PPB type I/II/III PATHOGNOMONIC; eRMS cervix/uterus PATHOGNOMONIC; Wilms; cystic nephroma",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); germline + somatic RNase IIIb second hit in tumour",
        "sarcoma_risk": "PPB type I/II/III PATHOGNOMONIC; embryonal RMS cervix/uterus/vagina PATHOGNOMONIC; eRMS DICER1 = highest cervical eRMS",
        "pathognomonic": "PPB PATHOGNOMONIC DICER1; eRMS cervix/uterus PATHOGNOMONIC; cystic nephroma any age PATHOGNOMONIC; nasal chondromesenchymal hamartoma",
        "key_avoid": "AVOID radiation PPB where possible (young patients). Do NOT miss annual chest CT birth to 8yr (PPB surveillance). Germline DICER1 test all PPB type I/II/III and eRMS cervix/uterus",
        "key_rule": "DICER1: annual low-dose chest CT birth to 8yr MANDATORY (PPB). eRMS cervix/uterus PATHOGNOMONIC → DICER1 germline test. Gynaecological surveillance from puberty. Renal US Wilms. Cascade first-degree relatives.",
        "surveillance": "Annual low-dose chest CT from birth to 8yr MANDATORY; annual renal ultrasound from birth (Wilms+cystic nephroma); annual pelvic exam from puberty; annual thyroid ultrasound from age 8yr; nasal exam annually children; cascade DICER1 germline first-degree relatives; DICER1 germline test all PPB/eRMS cervix/cystic nephroma",
        "targeted_rx": "PPB type I: resection (lobectomy) — survival >90%; PPB type II/III: surgery + chemotherapy (vinorelbine+cyclophosphamide+cisplatin); eRMS cervix/uterus: VAC (vincristine+actinomycin D+cyclophosphamide) + surgery fertility-sparing if possible; AVOID radiation PPB (use alternative to radiation in young children); bevacizumab anti-VEGF PPB recurrent (emerging); pembrolizumab DICER1-driven STS (experimental); Wilms standard (actinomycin D+vincristine ± doxorubicin per stage)",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FANCD1-"
            "Biallelic-FA-D1-Embryonal-RMS-PATHOGNOMONIC-Infancy-"
            "AVOID-Alkylating-Agents-ABSOLUTELY-FA-D1-"
            "Olaparib-FDA2020-HRD-Positive-STS-Monoallelic-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 soft tissue sarcoma molecular context: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; N-terminal PALB2-binding (aa 10-40); "
            "  BRC repeats 1-8 (aa 1002-2085) — RAD51 binding for HR repair; "
            "  OB folds (aa 2396-3186) — ssDNA binding; C-terminal nuclear localisation; "
            "  BRCA2 loads RAD51 onto ssDNA at DSBs — homologous recombination repair; "
            "  BRCA2/FANCD1 LOF → HR deficiency → replication fork collapse → NHEJ errors; "
            "  Biallelic BRCA2 = FA complementation group D1 (FA-D1) — most severe Fanconi subtype; "
            "CANCER RISKS (SARCOMA FOCUS): "
            "  FA-D1 biallelic: embryonal rhabdomyosarcoma PATHOGNOMONIC in infancy — median onset age 2yr; "
            "  FA-D1 = most severe Fanconi anaemia — median onset STS/ALL/brain tumour at age 2yr; "
            "  eRMS + FA-D1: AVOID alkylating agents (cyclophosphamide/ifosfamide) ABSOLUTELY — fatal toxicity; "
            "  FA-D1 eRMS: non-alkylating chemotherapy regimen (vinorelbine + actinomycin D — NOT VAC); "
            "  Monoallelic BRCA2 (AD): HRD — olaparib FDA2020 BRCA2 STS if HRD positive (emerging data); "
            "KEY MANAGEMENT: "
            "  FA-D1 diagnosis: AVOID ALL alkylating agents — cyclophosphamide, ifosfamide, melphalan fatal; "
            "  Sibling exclusion from SCT donor MANDATORY (carrier sibling = donor failure risk); "
            "  Bone marrow failure monitoring: CBC + FISH bone marrow if FA-D1 confirmed; "
            "  Monoallelic BRCA2 STS: test for HRD (genomic scar/LOH 13q) — olaparib eligibility; "
            "  FA Fanconi workup: chromosome fragility test (DEB/MMC test) all siblings"
        ),
        "syndrome": "BRCA2/FANCD1 — biallelic FA-D1 eRMS infancy PATHOGNOMONIC; monoallelic HRD STS olaparib; HBOC",
        "inheritance": "AD LOF monoallelic (HBOC); biallelic AR (FA-D1 most severe Fanconi) — both parents carriers",
        "sarcoma_risk": "FA-D1 biallelic eRMS PATHOGNOMONIC infancy; AVOID alkylating agents ABSOLUTELY FA-D1; olaparib HRD STS monoallelic",
        "pathognomonic": "FA-D1 biallelic eRMS infancy PATHOGNOMONIC; alkylator hypersensitivity PATHOGNOMONIC FA-D1; bone marrow failure FA",
        "key_avoid": "AVOID alkylating agents (cyclophosphamide/ifosfamide/melphalan) ABSOLUTELY in FA-D1 — fatal toxicity. Sibling exclusion from SCT donor MANDATORY. Do NOT use VAC protocol FA-D1 eRMS",
        "key_rule": "BRCA2/FA-D1: AVOID alkylating agents ABSOLUTELY (fatal). eRMS infancy = FA-D1 → chromosome fragility test. Sibling exclusion SCT donor MANDATORY. Monoallelic BRCA2 STS: HRD test → olaparib FDA2020. CBC bone marrow monitoring.",
        "surveillance": "FA-D1: CBC + FISH bone marrow annually; chromosome fragility DEB/MMC test siblings; annual whole-body MRI (multiple primary tumour risk FA-D1); cascade BRCA2 germline first-degree relatives; monoallelic BRCA2: annual breast MRI from 25yr; annual pelvic US; olaparib eligibility STS HRD positive; sibling SCT donor exclusion MANDATORY FA-D1",
        "targeted_rx": "FA-D1 eRMS: vinorelbine+actinomycin D (non-alkylating modified VAC — AVOID cyclophosphamide); haematopoietic SCT FA-D1 (sibling excluded as donor); olaparib FDA2020 BRCA2 STS monoallelic HRD-positive (emerging); cisplatin BRCA2 STS HRD-positive (HR-deficient = platinum sensitive); pembrolizumab HRD+MSI-H BRCA2 STS; AVOID ifosfamide FA-D1 (fatal); niraprib/rucaparib BRCA2 monoallelic STS (experimental)",
    },
    {
        "gene": "EXT1",
        "protein": (
            "EXT1 -- 8q24.11 Autosomal-Dominant-LOF -- 746aa -- "
            "EXT1-90kDa-Exostosin-Glycosyltransferase-Heparan-Sulfate-Chain-Elongation-"
            "HME1-Hereditary-Multiple-Exostoses-Type1-Osteochondromas-PATHOGNOMONIC-"
            "Secondary-Chondrosarcoma-1-5pct-Peripheral-"
            "MLPA-Mandatory-Large-Deletions-30pct-OMIM-133700"
        ),
        "locus": "8q24.11",
        "protein_size": (
            "746 aa / 90 kDa / 8q24.11 EXT1 soft tissue sarcoma molecular context: "
            "STRUCTURE: "
            "  746 aa / 90 kDa; N-terminal cytoplasmic domain; single transmembrane domain; "
            "  C-terminal glycosyltransferase domain — exostosin transferase activity; "
            "  EXT1 forms heterodimer with EXT2 — heparan sulfate chain elongation enzyme; "
            "  EXT1 LOF → heparan sulfate proteoglycan deficiency → SHH/BMP/FGF signalling dysregulation; "
            "  Osteochondroma formation: loss of heparan sulfate regulation → ectopic bone + cartilage growth; "
            "CANCER RISKS (SARCOMA FOCUS): "
            "  Hereditary Multiple Exostoses type 1 (HME1): osteochondromas PATHOGNOMONIC multiple sites; "
            "  Secondary chondrosarcoma from osteochondroma: 1-5% lifetime risk (peripheral chondrosarcoma); "
            "  Secondary Ewing-like round cell sarcoma at exostosis sites (rare — low-frequency risk); "
            "  Cap >2cm on MRI = malignant transformation alert — URGENT surgical excision required; "
            "  Cartilage cap thickness on MRI: growing cap + bone destruction = chondrosarcoma diagnosis; "
            "  EXT1 > EXT2 for malignant transformation risk (EXT1 genotype = higher sarcoma conversion); "
            "KEY MANAGEMENT: "
            "  MLPA MANDATORY — large deletions 30% EXT1 not captured by standard exon sequencing; "
            "  MRI of symptomatic exostoses: cartilage cap thickness monitoring (>2cm = URGENT excision); "
            "  Surgical resection symptomatic/growing exostoses — no targeted therapy approved; "
            "  Annual radiological survey children (growth plate exostoses) — plain X-ray; "
            "  Chondrosarcoma arising EXT1: wide surgical excision (chemotherapy-resistant, RT-resistant)"
        ),
        "syndrome": "Hereditary Multiple Exostoses type 1 (HME1/EXT1) — osteochondromas PATHOGNOMONIC; secondary chondrosarcoma 1-5%",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); haploinsufficiency; somatic LOH second hit in tumour",
        "sarcoma_risk": "Secondary chondrosarcoma 1-5% from osteochondroma; cap >2cm MRI = urgent excision; EXT1 > EXT2 malignant risk",
        "pathognomonic": "Multiple osteochondromas PATHOGNOMONIC HME1; cartilage cap >2cm = malignant transformation; peripheral chondrosarcoma EXT1",
        "key_avoid": "Do NOT dismiss cap growth on MRI — cap >2cm = malignant transformation URGENT excision. MLPA mandatory (30% large deletions missed). Do NOT delay surgery growing osteochondroma (chondrosarcoma chemotherapy-resistant)",
        "key_rule": "EXT1/HME1: multiple osteochondromas PATHOGNOMONIC. MLPA MANDATORY (30% large deletions). MRI cap >2cm = URGENT excision. Secondary chondrosarcoma 1-5% lifetime. Chondrosarcoma = surgery only (chemotherapy/RT resistant). EXT1 > EXT2 malignant risk.",
        "surveillance": "Annual plain X-ray skeleton childhood (osteochondroma growth); MRI symptomatic exostoses (cap thickness measurement); MRI surveillance EXT1 cap >1cm annually; whole-body skeletal survey at diagnosis; chondrosarcoma surveillance annually from age 20yr MRI of largest exostoses; cascade EXT1 germline first-degree relatives; MLPA EXT1 all index cases",
        "targeted_rx": "Wide surgical excision chondrosarcoma (peripheral from EXT1) — only curative treatment; surgery is first-line for symptomatic growing osteochondromas (excision cap + stalk); conventional chondrosarcoma: chemotherapy-resistant + radiation-resistant — surgery only; dedifferentiated chondrosarcoma: doxorubicin+ifosfamide (dedifferentiated component sarcoma); no approved targeted therapy osteochondroma/chondrosarcoma; IDH1/2 inhibitors (ivosidenib/enasidenib) if IDH-mutant chondrosarcoma co-mutation (rare EXT1)",
    },
    {
        "gene": "SMARCA4",
        "protein": (
            "SMARCA4 -- 19p13.2 Autosomal-Dominant-LOF -- 1647aa -- "
            "BRG1-185kDa-SWI-SNF-ATPase-Catalytic-Subunit-"
            "SCCOHT-Small-Cell-Carcinoma-Ovary-Hypercalcemic-Type-PATHOGNOMONIC-"
            "MRT-ATRT-SMARCA4-Deficient-"
            "BRG1-IHC-Loss-PATHOGNOMONIC-"
            "Tazemetostat-EZH2-Inhibitor-Clinical-Trials-OMIM-603254"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "1647 aa / 185 kDa / 19p13.2 SMARCA4 soft tissue sarcoma molecular context: "
            "STRUCTURE: "
            "  1647 aa / 185 kDa; N-terminal domain; HSA domain (aa 324-385); "
            "  DBINO domain; SnAC domain; DEXDc ATPase domain (aa 751-1003); "
            "  HELICc domain (aa 1015-1200); bromodomain (aa 1353-1469) — acetyl-lysine binding; "
            "  BRG1 is ATPase catalytic subunit of SWI/SNF complex — chromatin remodelling; "
            "  SMARCA4/BRG1 LOF → SWI/SNF activity lost → EZH2 (PRC2) unopposed → H3K27me3; "
            "  EZH2 inhibitor sensitivity: SWI/SNF-deficient cancers (tazemetostat clinical trials); "
            "CANCER RISKS (SARCOMA FOCUS): "
            "  SCCOHT (small cell carcinoma of ovary, hypercalcemic type): SMARCA4/BRG1 PATHOGNOMONIC; "
            "  SCCOHT: young women (mean age 24yr), highly aggressive, hypercalcaemia PATHOGNOMONIC; "
            "  Hypercalcaemia SCCOHT: PTHrP secretion — PATHOGNOMONIC clinical feature SCCOHT; "
            "  BRG1 IHC loss: SMARCA4 deficient — diagnostic marker universal in SCCOHT/MRT/ATRT; "
            "  MRT/ATRT SMARCA4-deficient: infants — RTPS2 (rhabdoid tumour predisposition type 2); "
            "KEY MANAGEMENT: "
            "  BRG1 IHC: universal screening in SCCOHT/suspected MRT/ATRT — loss = SMARCA4 deficient; "
            "  Germline SMARCA4 testing: all SCCOHT, all MRT/ATRT infants (germline in ~43% SCCOHT); "
            "  Tazemetostat EZH2 inhibitor: clinical trials SMARCA4-deficient sarcomas/SCCOHT; "
            "  Ovarian surveillance: no proven effective surveillance SCCOHT (tumour rapid onset); "
            "  Hypercalcaemia management: IV bisphosphonate/denosumab if PTHrP-mediated in SCCOHT"
        ),
        "syndrome": "SMARCA4/RTPS2 — SCCOHT PATHOGNOMONIC young women; MRT/ATRT SMARCA4-deficient infants; BRG1 IHC loss",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); germline SMARCA4 ~43% SCCOHT; de novo possible",
        "sarcoma_risk": "SCCOHT PATHOGNOMONIC young women mean 24yr; MRT SMARCA4-deficient RTPS2; hypercalcaemia SCCOHT PATHOGNOMONIC PTHrP",
        "pathognomonic": "SCCOHT SMARCA4/BRG1 loss PATHOGNOMONIC; hypercalcaemia SCCOHT PATHOGNOMONIC PTHrP; BRG1 IHC loss = SMARCA4 deficient",
        "key_avoid": "Do NOT miss BRG1 IHC in SCCOHT/ATRT/MRT — loss = SMARCA4 deficient (diagnostic + treatment eligibility tazemetostat). Do NOT delay germline SMARCA4 testing SCCOHT young woman (family surveillance). Hypercalcaemia SCCOHT = PTHrP not primary hyperparathyroidism",
        "key_rule": "SMARCA4: BRG1 IHC loss PATHOGNOMONIC SCCOHT/MRT/ATRT. SCCOHT mean age 24yr + hypercalcaemia PATHOGNOMONIC. Germline test all SCCOHT. Tazemetostat EZH2 inhibitor clinical trials. Cascade first-degree relatives.",
        "surveillance": "BRG1 IHC all suspected SCCOHT/ATRT/MRT; germline SMARCA4 all SCCOHT + MRT/ATRT infants; cascade germline first-degree relatives; ovarian surveillance: pelvic MRI annually germline SMARCA4 women (no proven screening, offer risk-reducing oophorectomy after childbearing); serum calcium monitoring germline SMARCA4 women; MRI brain+spine infants germline SMARCA4 (ATRT risk)",
        "targeted_rx": "Tazemetostat EZH2 inhibitor clinical trials SMARCA4-deficient SCCOHT/MRT; doxorubicin+carboplatin+cyclophosphamide SCCOHT first-line (aggressive multiagent); high-dose chemotherapy + autologous SCT SCCOHT (MSKCC intensive protocol); denosumab/IV bisphosphonate hypercalcaemia SCCOHT (PTHrP-mediated); pembrolizumab PD-L1+ SMARCA4-deficient STS (emerging); risk-reducing bilateral salpingo-oophorectomy germline SMARCA4 women after childbearing; surgery first-line SCCOHT (cytoreductive) before systemic",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

TREATMENT_PROTOCOLS_BY_GENE = {
    "TP53": [
        "Surgery first-line all TP53/LFS sarcoma — AVOID neoadjuvant radiation",
        "Doxorubicin+ifosfamide adult STS (UPS/LMS) first-line chemotherapy",
        "VAC (vincristine+actinomycin D+cyclophosphamide) embryonal RMS children LFS",
        "Pembrolizumab hypermutant TP53-associated STS high TMB (>10 mut/Mb)",
        "Trabectedin retroperitoneal LMS second-line (TP53 non-predictive)",
        "AVOID radiation in LFS — proton therapy if radiation unavoidable (multidisciplinary)",
        "Annual WB-MRI Toronto protocol surveillance MANDATORY",
        "Cascade germline TP53 testing all first-degree relatives",
    ],
    "NF1": [
        "Selumetinib FDA2020 MEK inhibitor inoperable plexiform neurofibromas (NOT MPNST)",
        "Doxorubicin+ifosfamide MPNST first-line sarcoma regimen (NOT glioma treatment)",
        "Trabectedin MPNST second-line (NF1-MPNST)",
        "Pazopanib MPNST third-line antiangiogenic",
        "Wide surgical excision MPNST (R0 negative margins critical)",
        "Gemcitabine+docetaxel MPNST refractory second-line",
        "Annual whole-body MRI surveillance (MPNST from plexiform NF)",
        "Cascade NF1 germline first-degree relatives",
    ],
    "RB1": [
        "Doxorubicin+ifosfamide LMS/secondary STS RB1-null first-line",
        "Gemcitabine+docetaxel LMS second-line",
        "Trabectedin LPS/LMS retroperitoneal third-line",
        "AVOID CDK4-6 inhibitors (palbociclib/ribociclib/abemaciclib — inactive RB1-null; harm risk)",
        "Enucleation retinoblastoma intraocular (intra-arterial carboplatin/melphalan salvage)",
        "AVOID external beam radiation orbit (secondary STS 40x lifetime risk)",
        "Annual MRI prior radiation fields surveillance MANDATORY",
        "Cascade RB1 germline first-degree relatives + newborn ophthalmology",
    ],
    "SMARCB1": [
        "Tazemetostat FDA2020 EZH2 inhibitor epithelioid sarcoma INI1-loss (disease control ~26%)",
        "Doxorubicin+ifosfamide epithelioid sarcoma/MRT first-line systemic",
        "High-dose chemotherapy + autologous SCT infant MRT (baby-brain protocol)",
        "AVOID radiation infants rhabdoid MRT/ATRT (rapidly fatal, RT adds toxicity without benefit)",
        "Complete surgical resection epithelioid sarcoma (wide margins critical)",
        "Carboplatin+etoposide+vincristine ATRT infants (non-radiation-based)",
        "Pembrolizumab INI1-deficient STS (emerging checkpoint immunotherapy data)",
        "Germline SMARCB1 test all MRT/ATRT infants; cascade first-degree relatives",
    ],
    "DICER1": [
        "PPB type I: lobectomy — survival >90% (cystic PPB early resection)",
        "PPB type II/III: surgery + vinorelbine+cyclophosphamide+cisplatin chemotherapy",
        "eRMS cervix/uterus: VAC (vincristine+actinomycin D+cyclophosphamide) + fertility-sparing surgery",
        "AVOID radiation PPB in young children (alternative chemotherapy protocols preferred)",
        "Bevacizumab anti-VEGF PPB recurrent (emerging data)",
        "Wilms tumour: actinomycin D+vincristine ± doxorubicin per COG staging",
        "Annual low-dose chest CT birth to 8yr surveillance MANDATORY",
        "Cascade DICER1 germline first-degree relatives",
    ],
    "BRCA2": [
        "FA-D1 eRMS: vinorelbine+actinomycin D (non-alkylating modified regimen — AVOID cyclophosphamide)",
        "AVOID ALL alkylating agents (cyclophosphamide/ifosfamide/melphalan) FA-D1 — FATAL TOXICITY",
        "Haematopoietic SCT FA-D1 (matched unrelated donor — sibling EXCLUDED as donor)",
        "Olaparib FDA2020 BRCA2 STS monoallelic HRD-positive (emerging indication)",
        "Cisplatin BRCA2 STS HRD-positive (HR-deficient = platinum sensitive)",
        "Pembrolizumab HRD+MSI-H BRCA2 STS (immunotherapy for HRD + MMR-d STS)",
        "DEB/MMC chromosome fragility test siblings FA-D1 MANDATORY",
        "Cascade BRCA2 germline first-degree relatives",
    ],
    "EXT1": [
        "Wide surgical excision chondrosarcoma peripheral EXT1 — only curative treatment",
        "Surgical excision symptomatic/growing osteochondromas (cap + stalk removal)",
        "URGENT excision osteochondroma cap >2cm on MRI (malignant transformation)",
        "Conventional chondrosarcoma: chemotherapy-resistant + radiation-resistant — surgery only",
        "Dedifferentiated chondrosarcoma: doxorubicin+ifosfamide (dedifferentiated sarcoma component)",
        "No approved targeted therapy osteochondroma/primary chondrosarcoma EXT1",
        "MLPA EXT1 all index cases (30% large deletions missed by standard sequencing)",
        "Annual MRI largest exostoses from age 20yr (cap thickness malignant transformation)",
    ],
    "SMARCA4": [
        "Tazemetostat EZH2 inhibitor clinical trials SMARCA4-deficient SCCOHT/MRT",
        "Doxorubicin+carboplatin+cyclophosphamide SCCOHT first-line multiagent",
        "High-dose chemotherapy + autologous SCT SCCOHT (MSKCC intensive protocol)",
        "Surgery first-line SCCOHT (cytoreductive) before systemic therapy",
        "Denosumab/IV bisphosphonate hypercalcaemia SCCOHT (PTHrP-mediated)",
        "Pembrolizumab PD-L1+ SMARCA4-deficient STS (emerging immunotherapy data)",
        "Risk-reducing bilateral salpingo-oophorectomy germline SMARCA4 women after childbearing",
        "Cascade SMARCA4 germline first-degree relatives",
    ],
}

SURVEILLANCE_BY_GENE = {
    "TP53": [
        "Annual whole-body MRI Toronto protocol MANDATORY from diagnosis",
        "Annual brain MRI (brain tumor surveillance LFS)",
        "Annual breast MRI women from age 20yr MANDATORY",
        "Annual abdominal US + blood count",
        "AVOID radiation ALL TP53/LFS carriers — proton sparing if essential",
        "Cascade TP53 germline first-degree relatives",
        "Pre-surgery germline TP53 testing all pediatric sarcoma patients",
    ],
    "NF1": [
        "Annual whole-body MRI (plexiform NF → MPNST surveillance) MANDATORY",
        "Annual ophthalmology + VEP from age 1yr (optic glioma 15-20%)",
        "Annual blood pressure check",
        "Selumetinib eligibility if plexiform NF symptomatic/inoperable (not MPNST)",
        "Annual dermatology (café-au-lait, neurofibromas)",
        "Cascade NF1 germline first-degree relatives",
    ],
    "RB1": [
        "Annual MRI prior radiation fields from diagnosis MANDATORY",
        "Annual ophthalmology bilateral retinoblastoma survivors",
        "Annual whole-body MRI RB1 germline adults (secondary STS + osteosarcoma)",
        "Annual chest X-ray (pulmonary metastases sarcoma)",
        "Ophthalmology newborn relatives RB1 germline",
        "Cascade RB1 germline first-degree relatives",
    ],
    "SMARCB1": [
        "MRI brain+spine+abdomen infants germline SMARCB1 from birth",
        "INI1 IHC all epithelioid sarcoma/MRT/ATRT (universal diagnostic screen)",
        "Germline SMARCB1 test all MRT/ATRT infant index cases",
        "Annual abdominal MRI RTPS1 germline carriers childhood",
        "Cascade germline first-degree relatives",
        "Tazemetostat eligibility: INI1 IHC loss confirmed epithelioid sarcoma",
    ],
    "DICER1": [
        "Annual low-dose chest CT from birth to 8yr MANDATORY (PPB surveillance)",
        "Annual renal ultrasound from birth (Wilms + cystic nephroma)",
        "Annual pelvic exam from puberty (eRMS cervix/uterus)",
        "Annual thyroid ultrasound from age 8yr",
        "Nasal exam annually children (nasal chondromesenchymal hamartoma)",
        "Cascade DICER1 germline first-degree relatives",
    ],
    "BRCA2": [
        "FA-D1: CBC + FISH bone marrow annually",
        "Chromosome fragility DEB/MMC test siblings MANDATORY",
        "Annual whole-body MRI FA-D1 (multiple primary tumour risk)",
        "Monoallelic BRCA2: annual breast MRI from age 25yr",
        "Annual pelvic US monoallelic BRCA2",
        "Sibling SCT donor exclusion MANDATORY FA-D1",
        "Cascade BRCA2 germline first-degree relatives",
    ],
    "EXT1": [
        "Annual plain X-ray skeleton childhood (osteochondroma growth)",
        "MRI symptomatic exostoses (cap thickness measurement)",
        "MRI largest exostoses annually from age 20yr",
        "Whole-body skeletal survey at diagnosis",
        "MLPA EXT1 all index cases (large deletion detection)",
        "Cascade EXT1 germline first-degree relatives",
    ],
    "SMARCA4": [
        "BRG1 IHC all suspected SCCOHT/ATRT/MRT (universal screen)",
        "Germline SMARCA4 all SCCOHT + MRT/ATRT infants",
        "Pelvic MRI annually germline SMARCA4 women",
        "Serum calcium monitoring germline SMARCA4 women (hypercalcaemia)",
        "MRI brain+spine infants germline SMARCA4 (ATRT risk)",
        "Cascade germline first-degree relatives",
    ],
}


def _make_patients(gene: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n = 40

    sarcoma_subtypes = {
        "TP53": [
            ("Undifferentiated pleomorphic sarcoma (UPS)", 0.28),
            ("Embryonal rhabdomyosarcoma (eRMS)", 0.27),
            ("Leiomyosarcoma (LMS)", 0.25),
            ("Osteosarcoma (LFS-associated)", 0.12),
            ("Adrenocortical carcinoma (ACC)", 0.08),
        ],
        "NF1": [
            ("MPNST (malignant peripheral nerve sheath tumour)", 0.50),
            ("Plexiform neurofibroma (symptomatic/growing)", 0.25),
            ("Low-grade glioma (NF1-associated LGG)", 0.15),
            ("Ewing-like round cell sarcoma (NF1-associated)", 0.10),
        ],
        "RB1": [
            ("Leiomyosarcoma (secondary post-RT)", 0.40),
            ("Osteosarcoma (RB1-germline)", 0.30),
            ("Spindle cell sarcoma (radiation field)", 0.20),
            ("Retinoblastoma (bilateral)", 0.10),
        ],
        "SMARCB1": [
            ("Epithelioid sarcoma (INI1-loss, proximal/axial)", 0.40),
            ("Malignant rhabdoid tumour (MRT)", 0.30),
            ("ATRT (atypical teratoid rhabdoid tumour)", 0.20),
            ("Epithelioid sarcoma (distal extremity)", 0.10),
        ],
        "DICER1": [
            ("PPB type II/III (pleuropulmonary blastoma)", 0.40),
            ("Embryonal rhabdomyosarcoma (cervix/uterus)", 0.30),
            ("PPB type I (cystic)", 0.15),
            ("Wilms tumour (DICER1-associated)", 0.15),
        ],
        "BRCA2": [
            ("Embryonal rhabdomyosarcoma FA-D1 (infancy)", 0.40),
            ("Leiomyosarcoma (BRCA2 monoallelic HRD)", 0.25),
            ("ALL/AML (FA-D1 haematological)", 0.20),
            ("Brain tumour (FA-D1 medulloblastoma)", 0.15),
        ],
        "EXT1": [
            ("Chondrosarcoma (peripheral from osteochondroma)", 0.50),
            ("Multiple osteochondromas (symptomatic)", 0.30),
            ("Dedifferentiated chondrosarcoma", 0.12),
            ("Ewing-like round cell sarcoma (exostosis site)", 0.08),
        ],
        "SMARCA4": [
            ("SCCOHT (small cell carcinoma ovary, hypercalcemic type)", 0.45),
            ("Malignant rhabdoid tumour SMARCA4-deficient", 0.25),
            ("ATRT SMARCA4-deficient (infant)", 0.20),
            ("SWI/SNF-deficient undifferentiated carcinoma", 0.10),
        ],
    }

    variants_by_gene = {
        "TP53": [
            "p.Arg248Trp (R248W — dominant negative DBD hotspot; 50% pediatric LFS sarcoma)",
            "p.Arg273His (R273H — DNA contact hotspot dominant negative)",
            "p.Gly245Ser (G245S — structural hotspot DBD)",
            "p.Arg175His (R175H — structural hotspot conformational change)",
            "p.Arg337His (R337H — Brazilian founder tetramerisation domain)",
            "p.Arg248Gln (R248Q — DBD hotspot LFS sarcoma)",
            "c.IVS6+1G>A (splice donor intron 6 TP53)",
        ],
        "NF1": [
            "p.Arg1534Ter (GRD truncation MPNST high risk)",
            "p.Glu1200Lys (GRD RAS-GAP contact residue)",
            "c.IVS14+1G>A (splice intron 14 NF1)",
            "del exons 4-7 (large deletion NF1 — MLPA required)",
            "p.Tyr489Ter (early truncation NF1)",
            "p.Leu847Pro (ARM repeat hydrophobic NF1)",
        ],
        "RB1": [
            "p.Arg455Ter (R455X — ISCN pocket domain truncation)",
            "p.Glu137Ter (early truncation pocket domain A)",
            "c.IVS7+1G>A (splice donor exon 7 RB1)",
            "del exon 13-14 (large deletion RB1 — MLPA required)",
            "p.Trp563Ter (pocket domain B truncation)",
            "p.Leu690Pro (pocket domain B structural hydrophobic)",
        ],
        "SMARCB1": [
            "p.Arg374Ter (coiled-coil domain truncation RTPS1)",
            "p.Gln318Ter (RPT2 domain truncation SMARCB1)",
            "c.IVS6+1G>A (splice donor exon 6 SMARCB1)",
            "del exons 1-9 (homozygous deletion MRT biallelic)",
            "p.Arg40His (N-terminal domain structural SMARCB1)",
            "p.Tyr129Ter (RPT1 domain truncation early)",
        ],
        "DICER1": [
            "p.Glu1705Lys (E1705K — RNase IIIb metal ion binding somatic hot)",
            "p.Asp1709Asn (D1709N — RNase IIIb somatic hotspot PPB)",
            "p.Gln1727Ter (RNase IIIb truncation germline DICER1)",
            "c.IVS24+1G>A (splice donor exon 24 DICER1)",
            "p.Arg1810Ter (dsRBD truncation germline)",
            "del exons 20-24 (large deletion DICER1 — MLPA)",
        ],
        "BRCA2": [
            "p.Lys3326Ter (K3326X — C-terminal truncation high frequency FA-D1)",
            "p.Glu1308Ter (BRC repeat 5 truncation germline)",
            "c.IVS11+1G>A (splice donor intron 11 BRCA2 FA-D1)",
            "p.Trp2626Ter (OB fold truncation BRCA2)",
            "del exons 15-26 (large deletion BRCA2 FA-D1)",
            "p.Arg2336His (OB fold structural BRCA2 sarcoma)",
        ],
        "EXT1": [
            "p.Arg340Ter (glycosyltransferase domain truncation EXT1)",
            "p.Glu475Lys (glycosyltransferase active site EXT1)",
            "c.IVS5+1G>A (splice donor exon 5 EXT1)",
            "del exons 1-3 (large deletion EXT1 — MLPA 30%)",
            "p.Leu224Pro (transmembrane domain hydrophobic EXT1)",
            "p.Gln175Ter (early truncation EXT1 extracellular)",
        ],
        "SMARCA4": [
            "p.Glu1525Lys (bromodomain interface SMARCA4 SCCOHT)",
            "p.Arg1192Ter (HELICc domain truncation BRG1)",
            "c.IVS26+1G>A (splice donor exon 26 SMARCA4)",
            "del exons 16-20 (large deletion SMARCA4 SCCOHT)",
            "p.Gly1232Glu (DEXDc ATPase catalytic SMARCA4)",
            "p.Arg885Ter (SnAC domain truncation SMARCA4)",
        ],
    }

    age_ranges = {
        "TP53": (3, 50),
        "NF1":  (10, 55),
        "RB1":  (1, 40),
        "SMARCB1": (0, 35),
        "DICER1": (0, 30),
        "BRCA2": (0, 45),
        "EXT1": (15, 60),
        "SMARCA4": (15, 40),
    }

    subtypes = sarcoma_subtypes[gene]
    variants = variants_by_gene[gene]
    age_lo, age_hi = age_ranges[gene]

    patients = []
    for i in range(n):
        age = rng.randint(age_lo, age_hi)
        r = rng.random()
        cumulative = 0.0
        tumour = subtypes[-1][0]
        for t, p in subtypes:
            cumulative += p
            if r < cumulative:
                tumour = t
                break
        variant = rng.choice(variants)
        sex = rng.choice(["M", "F"])
        stage = rng.choices(["I", "II", "III", "IV"], weights=[0.30, 0.28, 0.22, 0.20])[0]
        radiation_contraindicated = gene in ("TP53", "RB1", "DICER1", "BRCA2") and rng.random() < 0.80
        targeted_therapy = rng.random() < 0.55
        mpnst_risk = (gene == "NF1") and (rng.random() < 0.12)
        ini1_loss = (gene == "SMARCB1") and (rng.random() < 0.85)
        ppb_dicer1 = (gene == "DICER1") and (rng.random() < 0.42)
        fa_d1_alkylator_contraindicated = (gene == "BRCA2") and (rng.random() < 0.35)
        malignant_transformation = (gene == "EXT1") and (rng.random() < 0.04)
        sccoht = (gene == "SMARCA4") and (rng.random() < 0.48)
        relapse = (stage in ("III", "IV")) and (rng.random() < 0.50)
        hypercalcaemia = (gene == "SMARCA4") and sccoht and (rng.random() < 0.65)

        patients.append({
            "patient_id": f"{gene[:5].upper()}-{seed:04d}-{i+1:02d}",
            "gene": gene,
            "age_at_dx": age,
            "sex": sex,
            "tumour_type": tumour,
            "stage": stage,
            "variant": variant,
            "radiation_contraindicated": radiation_contraindicated,
            "targeted_therapy": targeted_therapy,
            "mpnst_risk": mpnst_risk,
            "ini1_loss": ini1_loss,
            "ppb_dicer1": ppb_dicer1,
            "fa_d1_alkylator_contraindicated": fa_d1_alkylator_contraindicated,
            "malignant_transformation": malignant_transformation,
            "sccoht": sccoht,
            "hypercalcaemia": hypercalcaemia,
            "relapse": relapse,
        })
    return patients


def generate_overview() -> dict:
    from collections import Counter
    cohorts = {}
    for i, g in enumerate(ATLAS_GENES):
        cohorts[g["gene"]] = _make_patients(g["gene"], SEED_BASE + i)
    total = sum(len(pts) for pts in cohorts.values())
    gene_counts = {gene: len(pts) for gene, pts in cohorts.items()}
    radiation_ci_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["radiation_contraindicated"]) / total, 1
    )
    targeted_therapy_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["targeted_therapy"]) / total, 1
    )
    mpnst_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["mpnst_risk"]) / total, 1
    )
    ini1_loss_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["ini1_loss"]) / total, 1
    )
    ppb_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["ppb_dicer1"]) / total, 1
    )
    sccoht_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["sccoht"]) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )

    return {
        "atlas": "Hereditary-Soft-Tissue-Sarcoma-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "radiation_contraindicated_rate_pct": radiation_ci_rate,
        "targeted_therapy_rate_pct": targeted_therapy_rate,
        "mpnst_risk_rate_pct": mpnst_rate,
        "ini1_loss_rate_pct": ini1_loss_rate,
        "ppb_dicer1_rate_pct": ppb_rate,
        "sccoht_rate_pct": sccoht_rate,
        "mean_age_at_dx": mean_age,
        "key_facts": [
            "TP53/LFS: sarcoma #1 MOST COMMON pediatric LFS cancer (30% lifetime); RMS/UPS/LMS; R248W/R273H dominant-negative 50% pediatric LFS sarcoma; AVOID RADIATION ABSOLUTELY; WB-MRI Toronto protocol MANDATORY",
            "NF1: MPNST 8-13% HIGHEST hereditary STS risk; plexiform neurofibroma → MPNST transformation; MPNST = SARCOMA (doxorubicin NOT glioma regimen); selumetinib FDA2020 plexiform NF NOT MPNST; annual whole-body MRI MANDATORY",
            "RB1: secondary STS post-radiation 40x HIGHEST secondary cancer risk hereditary syndrome; CDK4-6i INACTIVE RB1-null STS; AVOID high-dose radiation germline RB1; bilateral retinoblastoma PATHOGNOMONIC",
            "SMARCB1: MRT/ATRT infants under 3yr PATHOGNOMONIC RTPS1; epithelioid sarcoma INI1/SMARCB1 nuclear loss IHC PATHOGNOMONIC; tazemetostat FDA2020 EZH2 inhibitor epithelioid sarcoma INI1-loss APPROVED",
            "DICER1: PPB type I/II/III PATHOGNOMONIC; eRMS cervix/uterus/vagina PATHOGNOMONIC; annual low-dose chest CT birth to 8yr MANDATORY; AVOID radiation PPB young children",
            "BRCA2: FA-D1 biallelic eRMS PATHOGNOMONIC infancy median age 2yr; AVOID alkylating agents ABSOLUTELY FA-D1 (cyclophosphamide/ifosfamide FATAL); sibling exclusion SCT donor MANDATORY; olaparib FDA2020 HRD-positive STS monoallelic",
            "EXT1/HME1: multiple osteochondromas PATHOGNOMONIC; secondary chondrosarcoma 1-5% from osteochondroma; cap >2cm MRI = malignant transformation URGENT excision; chondrosarcoma surgery-only (chemotherapy-resistant); MLPA MANDATORY 30% large deletions",
            "SMARCA4: SCCOHT PATHOGNOMONIC young women mean age 24yr + hypercalcaemia PATHOGNOMONIC PTHrP; BRG1 IHC loss PATHOGNOMONIC SMARCA4-deficient; tazemetostat EZH2 clinical trials; germline test all SCCOHT (43% germline)",
        ],
    }


def generate_breakdown() -> dict:
    from collections import Counter
    breakdown = {}
    for i, g in enumerate(ATLAS_GENES):
        gene = g["gene"]
        pts = _make_patients(gene, SEED_BASE + i)
        tumour_counts = Counter(p["tumour_type"] for p in pts)
        variant_counts = Counter(p["variant"] for p in pts)
        top_tumours = tumour_counts.most_common(3)
        top_variants = variant_counts.most_common(3)
        breakdown[gene] = {
            "gene_info": {
                "gene": gene,
                "protein": g["protein"],
                "locus": g["locus"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "sarcoma_risk": g["sarcoma_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "radiation_contraindicated_pct": round(100 * sum(1 for p in pts if p["radiation_contraindicated"]) / len(pts), 1),
            "targeted_therapy_pct": round(100 * sum(1 for p in pts if p["targeted_therapy"]) / len(pts), 1),
            "mpnst_risk_pct": round(100 * sum(1 for p in pts if p["mpnst_risk"]) / len(pts), 1),
            "ini1_loss_pct": round(100 * sum(1 for p in pts if p["ini1_loss"]) / len(pts), 1),
            "ppb_dicer1_pct": round(100 * sum(1 for p in pts if p["ppb_dicer1"]) / len(pts), 1),
            "fa_d1_alkylator_ci_pct": round(100 * sum(1 for p in pts if p["fa_d1_alkylator_contraindicated"]) / len(pts), 1),
            "malignant_transformation_pct": round(100 * sum(1 for p in pts if p["malignant_transformation"]) / len(pts), 1),
            "sccoht_pct": round(100 * sum(1 for p in pts if p["sccoht"]) / len(pts), 1),
            "hypercalcaemia_pct": round(100 * sum(1 for p in pts if p["hypercalcaemia"]) / len(pts), 1),
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
        "atlas": "Hereditary-Soft-Tissue-Sarcoma-Predisposition-Atlas",
        "definitions": {
            "tp53_lfs_sarcoma_avoid_radiation_wb_mri": (
                "TP53/LFS: 393aa / 43kDa; 17p13.1; guardian of the genome tumour suppressor; "
                "SARCOMA #1 PEDIATRIC LFS CANCER: 30% lifetime risk — #1 most common cancer in LFS pediatric cohort; "
                "RMS / UPS / LMS: embryonal rhabdomyosarcoma + undifferentiated pleomorphic sarcoma + leiomyosarcoma dominant subtypes LFS STS; "
                "R248W / R273H DOMINANT NEGATIVE HOTSPOTS: 50% pediatric LFS sarcoma — highest-frequency variants sarcoma cluster; "
                "AVOID RADIATION ABSOLUTELY: radiation in LFS carriers → catastrophic secondary sarcoma along radiation field; "
                "WB-MRI TORONTO PROTOCOL: annual whole-body MRI + annual brain MRI MANDATORY from diagnosis; "
                "SURGERY FIRST-LINE: all TP53/LFS STS — avoid neoadjuvant/adjuvant radiation; proton only if multidisciplinary team confirms unavoidable; "
                "De novo TP53 ~25% LFS: test both parents before cascade; pre-surgery germline testing MANDATORY all pediatric STS."
            ),
            "nf1_mpnst_highest_sts_selumetinib": (
                "NF1: 2839aa / 319kDa; 17q11.2; RAS-GAP neurofibromin; NF1 LOF → RAS-GTP → MAPK/ERK hyperactivation; "
                "MPNST 8-13% HIGHEST HEREDITARY STS RISK: malignant peripheral nerve sheath tumour from plexiform neurofibroma — annual MRI MANDATORY; "
                "MPNST IS SARCOMA: doxorubicin+ifosfamide first-line — NOT glioma treatment, NOT immunotherapy, NOT selumetinib; "
                "SELUMETINIB FDA2020: MEK inhibitor APPROVED for inoperable plexiform neurofibromas — NOT MPNST treatment; "
                "PLEXIFORM NEUROFIBROMA → MPNST: internal plexiform >3cm + rapid growth = MPNST alert — MRI whole-body annually; "
                "CAFÉ-AU-LAIT 6+ MACULES PATHOGNOMONIC: >5mm prepubertal / >15mm postpubertal — NF1 diagnostic criterion; "
                "WIDE SURGICAL RESECTION MPNST: R0 negative margins critical — 5-year survival <50% without R0 resection."
            ),
            "rb1_secondary_sts_radiation_avoid_cdk4_6i": (
                "RB1: 928aa / 105kDa; 13q14.2; pRb E2F G1-S checkpoint regulator; pRb LOF → E2F constitutive → uncontrolled S-phase; "
                "SECONDARY STS POST-RADIATION 40x: leiomyosarcoma at radiation field — HIGHEST secondary cancer risk of any hereditary syndrome; "
                "AVOID HIGH-DOSE RADIATION RB1 GERMLINE: secondary sarcoma LETHAL — avoid radiation orbit/pelvis/extremity RB1 germline carriers; "
                "CDK4-6i INACTIVE RB1-NULL: palbociclib / ribociclib / abemaciclib are INACTIVE in RB1-null STS (pRb is the CDK4/6 target — absent = drug useless + harm risk); "
                "BILATERAL RETINOBLASTOMA PATHOGNOMONIC: germline RB1 → lifetime STS surveillance mandatory; "
                "OSTEOSARCOMA 30%: primary bone sarcoma elevated in RB1 germline (same pathway); "
                "ANNUAL MRI RADIATION FIELD SURVEILLANCE: MANDATORY from diagnosis for all bilateral RB survivors (secondary STS detection)."
            ),
            "smarcb1_ini1_loss_pathognomonic_tazemetostat": (
                "SMARCB1/INI1: 385aa / 47kDa; 22q11.23; SWI/SNF core structural subunit; SMARCB1 LOF → EZH2 (PRC2) unopposed → H3K27me3 silencing; "
                "MRT INFANTS PATHOGNOMONIC RTPS1: malignant rhabdoid tumour under 3yr — 33-40% have germline SMARCB1; "
                "INI1/SMARCB1 IHC NUCLEAR LOSS PATHOGNOMONIC: universal diagnostic marker epithelioid sarcoma + MRT + ATRT — loss = deficient; "
                "EPITHELIOID SARCOMA: proximal/axial type INI1 loss PATHOGNOMONIC; young adults extremity or trunk; "
                "TAZEMETOSTAT FDA2020: EZH2 inhibitor APPROVED epithelioid sarcoma INI1-loss (disease control ~26%); "
                "AVOID RADIATION INFANTS RHABDOID: MRT/ATRT rapidly fatal — radiation adds toxicity without survival benefit; "
                "GERMLINE SMARCB1 ALL INFANT MRT/ATRT: 33-40% germline positive → family surveillance cascade obligatory."
            ),
            "dicer1_ppb_pathognomonic_erms_cervix_annual_ct": (
                "DICER1: 1922aa / 218kDa; 14q32.13; RNase III endoribonuclease; DICER1 LOF → miRNA biogenesis impaired → oncogenic mRNA de-repression; "
                "PPB TYPE I/II/III PATHOGNOMONIC: pleuropulmonary blastoma — #1 presenting tumour DICER1 syndrome; "
                "PPB type I (cystic, <2yr): >90% survival lobectomy alone; type II/III (solid): higher mortality + chemotherapy; "
                "eRMS CERVIX/UTERUS PATHOGNOMONIC: embryonal rhabdomyosarcoma lower genital tract young women = DICER1 germline test MANDATORY; "
                "ANNUAL LOW-DOSE CHEST CT BIRTH TO 8yr MANDATORY: PPB surveillance siblings + index — earliest PPB detection = best outcome; "
                "AVOID RADIATION PPB: young patients already vulnerable — use chemotherapy-based protocols; "
                "CYSTIC NEPHROMA PATHOGNOMONIC: any age cystic nephroma = DICER1 syndrome test; nasal chondromesenchymal hamartoma PATHOGNOMONIC."
            ),
            "brca2_fa_d1_erms_avoid_alkylating_olaparib": (
                "BRCA2/FANCD1: 3418aa / 384kDa; 13q12.3; HR scaffold RAD51 loader; BRCA2 LOF → HR deficiency → replication fork collapse; "
                "FA-D1 BIALLELIC: most severe Fanconi anaemia complementation group — embryonal RMS PATHOGNOMONIC infancy median age 2yr; "
                "AVOID ALKYLATING AGENTS ABSOLUTELY FA-D1: cyclophosphamide + ifosfamide + melphalan = FATAL TOXICITY in FA-D1; "
                "NON-ALKYLATING eRMS PROTOCOL FA-D1: vinorelbine + actinomycin D (modified regimen — NOT VAC which uses cyclophosphamide); "
                "SIBLING SCT DONOR EXCLUSION MANDATORY: carrier sibling as SCT donor = donor failure risk — must exclude by testing; "
                "OLAPARIB FDA2020 MONOALLELIC BRCA2 STS: HRD-positive STS (emerging indication — HR-deficient = PARP inhibitor sensitive); "
                "DEB/MMC CHROMOSOME FRAGILITY TEST: MANDATORY siblings FA-D1 index case — diagnostic gold standard Fanconi."
            ),
            "ext1_hme1_osteochondroma_pathognomonic_chondrosarcoma": (
                "EXT1: 746aa / 90kDa; 8q24.11; exostosin glycosyltransferase; EXT1-EXT2 heterodimer — heparan sulfate chain elongation; "
                "HEREDITARY MULTIPLE EXOSTOSES HME1: multiple osteochondromas PATHOGNOMONIC — appendicular skeleton dominant; "
                "SECONDARY CHONDROSARCOMA 1-5% LIFETIME: from osteochondroma malignant transformation — peripheral chondrosarcoma; "
                "CAP >2cm MRI = MALIGNANT TRANSFORMATION: URGENT surgical excision — cartilage cap + bone destruction = chondrosarcoma; "
                "CHONDROSARCOMA SURGERY ONLY: conventional chondrosarcoma = chemotherapy-resistant + radiation-resistant — surgery is ONLY curative treatment; "
                "EXT1 > EXT2 MALIGNANT RISK: EXT1 genotype = higher chondrosarcoma conversion rate than EXT2; "
                "MLPA MANDATORY: 30% EXT1 large deletions not captured by standard exon sequencing — MLPA required for complete diagnosis."
            ),
            "smarca4_sccoht_pathognomonic_brg1_ihc_tazemetostat": (
                "SMARCA4/BRG1: 1647aa / 185kDa; 19p13.2; SWI/SNF ATPase catalytic subunit; BRG1 LOF → chromatin remodelling lost → EZH2 unopposed; "
                "SCCOHT PATHOGNOMONIC SMARCA4/BRG1: small cell carcinoma of ovary, hypercalcemic type — young women mean age 24yr; "
                "HYPERCALCAEMIA SCCOHT PATHOGNOMONIC: PTHrP secretion — NOT primary hyperparathyroidism (PTH suppressed); "
                "BRG1 IHC LOSS PATHOGNOMONIC: SMARCA4-deficient diagnostic marker — universal screen in SCCOHT/ATRT/MRT; "
                "GERMLINE SMARCA4 ~43% SCCOHT: test all SCCOHT for germline — family surveillance mandatory if positive; "
                "TAZEMETOSTAT CLINICAL TRIALS: EZH2 inhibitor — SMARCA4-deficient tumours (SWI/SNF deficient → EZH2 sensitivity); "
                "RTPS2 MRT/ATRT: SMARCA4-deficient rhabdoid infants — less established than SMARCB1-RTPS1 but surveillance mandatory germline."
            ),
        },
        "key_clinical_distinctions": [
            "TP53/LFS: AVOID RADIATION ABSOLUTELY (secondary sarcoma). Sarcoma #1 pediatric LFS cancer. WB-MRI Toronto protocol MANDATORY. Surgery first-line. R248W/R273H dominant-negative 50% pediatric LFS sarcoma",
            "NF1: MPNST 8-13% HIGHEST hereditary STS. MPNST = SARCOMA (doxorubicin NOT glioma). Selumetinib FDA2020 plexiform NF NOT MPNST. Annual whole-body MRI mandatory. R0 surgical margin critical MPNST",
            "RB1: secondary STS post-RT 40x HIGHEST hereditary syndrome. CDK4-6i INACTIVE RB1-null (palbociclib useless + harmful). AVOID radiation RB1 germline. Bilateral retinoblastoma PATHOGNOMONIC → lifetime STS surveillance",
            "SMARCB1: INI1/SMARCB1 IHC nuclear loss PATHOGNOMONIC epithelioid sarcoma. Tazemetostat FDA2020 INI1-loss ES APPROVED. MRT infants PATHOGNOMONIC RTPS1 → germline test. AVOID radiation infants rhabdoid",
            "DICER1: PPB PATHOGNOMONIC. eRMS cervix/uterus PATHOGNOMONIC → DICER1 germline test. Annual chest CT birth to 8yr MANDATORY. Cystic nephroma any age PATHOGNOMONIC DICER1. AVOID radiation PPB young children",
            "BRCA2/FA-D1: AVOID alkylating agents ABSOLUTELY (cyclophosphamide/ifosfamide FATAL FA-D1). eRMS infancy FA-D1 PATHOGNOMONIC. Sibling SCT donor exclusion MANDATORY. Olaparib FDA2020 HRD-positive monoallelic BRCA2 STS",
            "EXT1/HME1: multiple osteochondromas PATHOGNOMONIC. Cap >2cm MRI = malignant transformation URGENT excision. Chondrosarcoma surgery-only (chemo-resistant + RT-resistant). MLPA MANDATORY 30% large deletions. EXT1 > EXT2 malignant risk",
            "SMARCA4: SCCOHT PATHOGNOMONIC young women + hypercalcaemia PATHOGNOMONIC PTHrP. BRG1 IHC loss = SMARCA4-deficient universal marker. Germline in 43% SCCOHT → cascade. Tazemetostat EZH2 inhibitor clinical trials",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
