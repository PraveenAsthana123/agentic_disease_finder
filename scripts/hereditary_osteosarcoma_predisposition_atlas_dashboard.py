#!/usr/bin/env python3
"""Hereditary-Osteosarcoma-Predisposition-Atlas -- Complete 8-Gene Reference
RB1     (Retinoblastoma protein; 928aa; 13q14.2; AD LOF;
         Hereditary retinoblastoma;
         Osteosarcoma 40% lifetime HIGHEST hereditary OS risk;
         bilateral RB PATHOGNOMONIC;
         secondary OS post-RT CDK4/6i INACTIVE in RB1-null tumours;
         AVOID high-dose RT at OS site;
         seed SEED_BASE+0) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         OS #1 cancer in LFS children; anaplastic/pleomorphic OS PATHOGNOMONIC LFS;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+1) .
RECQL4  (RecQ-like helicase 4; 1208aa; 8q24.12; AR LOF;
         Rothmund-Thomson type 2 (RTS2);
         osteosarcoma 30-50%;
         poikiloderma onset 3-6 months PATHOGNOMONIC;
         Rapadilino / FIRES syndromes allelic;
         seed SEED_BASE+2) .
BRCA2   (Breast cancer type 2 susceptibility protein; 3418aa; 13q12.3; AD LOF / biallelic AR FA-D1;
         Fanconi anaemia complementation group D1 (biallelic);
         OS/bone tumours FA-D1;
         AVOID alkylating agents ABSOLUTELY;
         sibling donor exclusion MANDATORY;
         seed SEED_BASE+3) .
NF1     (Neurofibromin; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis 1;
         cafe-au-lait macules PATHOGNOMONIC;
         MPNST 8-13%; osteosarcoma 2-3x elevated vs general population;
         selumetinib FDA 2020 pediatric plexiform;
         seed SEED_BASE+4) .
CDKN2A  (Cyclin-dependent kinase inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMMM / Familial melanoma;
         osteosarcoma at post-irradiated sites;
         p16 IHC loss PATHOGNOMONIC;
         CDK4/6i palbociclib in CDK4/CDK6-amplified OS;
         pancreatic adenocarcinoma 20x concurrent risk;
         seed SEED_BASE+5) .
DICER1  (DICER1 RNase III; 1922aa; 14q32.13; AD LOF;
         DICER1 syndrome;
         PPB PATHOGNOMONIC; mesenchymal tumours including OS 2-4x elevated;
         CT chest siblings <8yr; AVOID radiation children;
         seed SEED_BASE+6) .
WRN     (Werner syndrome helicase; 1432aa; 8p12; AR LOF;
         Werner syndrome;
         adult-onset bilateral cataracts PATHOGNOMONIC;
         mesenchymal cancer predominance (osteosarcoma, soft tissue sarcoma);
         type 2 DM + scleroderma PATHOGNOMONIC;
         vemurafenib for melanoma component;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3310-3317)
"""
import random

SEED_BASE = 3310

ATLAS_GENES = [
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "pRb-105kDa-E2F-Regulator-Hereditary-Retinoblastoma-OS-40pct-Lifetime-HIGHEST-"
            "Bilateral-RB-PATHOGNOMONIC-CDK4-6i-INACTIVE-RB1-Null-AVOID-High-Dose-RT-OS-Site-OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 105 kDa / 13q14.2 RB1 encodes retinoblastoma protein (pRb): "
            "STRUCTURE: "
            "  928 aa / 105 kDa; nuclear tumour suppressor; master cell cycle regulator; "
            "  N-terminal domain (aa 1-378): structural scaffold; "
            "  Pocket A domain (aa 379-572): E2F binding surface A; "
            "  Spacer region (aa 573-645): caspase cleavage site (D886); "
            "  Pocket B domain (aa 646-772): E2F binding surface B (A+B pocket = E2F interaction); "
            "    CDK4/CDK6-cyclinD phosphorylates pRb at S780/S795 → releases E2F transcription factors; "
            "    Hypophosphorylated pRb: sequestors E2F → represses S-phase entry; "
            "  C-terminal domain (aa 773-928): LXCXE motif binding (viral oncoproteins); "
            "  RB1 LOF: unrestrained E2F activity → unchecked G1/S transition; "
            "HEREDITARY RETINOBLASTOMA: "
            "  OMIM 180200; AD LOF; 40% de novo; bilateral RB PATHOGNOMONIC germline; "
            "  Bilateral retinoblastoma: presents mean age 15 months; "
            "  Unilateral: 25% carry germline RB1 (vs 75% somatic); "
            "  Second cancers after retinoblastoma: "
            "    Osteosarcoma: HIGHEST risk 40% lifetime in irradiated survivors; "
            "    Pinealoblastoma (trilateral RB): PATHOGNOMONIC germline; "
            "    Soft tissue sarcoma: 15-20x; "
            "    Melanoma: 5-10x elevated; "
            "OSTEOSARCOMA IN RB1: "
            "  OS site: typically bone (distal femur, proximal tibia, proximal humerus); "
            "  OS after RT field: radiation-induced OS predominantly within RT field; "
            "  CDK4/CDK6 inhibitors (palbociclib, ribociclib): "
            "    INACTIVE in RB1-null tumours (loss of pRb → CDK4/6i target absent); "
            "    RB1-null OS: CDK4/6i are ineffective; alternative cell cycle targets needed; "
            "  Bone scintigraphy + MRI surveillance: annual from age 6yr in germline RB1; "
            "TREATMENT RB1/OS: "
            "  MAP (methotrexate-adriamycin-cisplatin) OS standard backbone; "
            "  AVOID re-irradiation at OS site (secondary OS risk in RT field); "
            "  CDK4/6i: INACTIVE in RB1-null OS; do NOT use as monotherapy; "
            "  Li-Fraumeni-like overlap: RB1 + TP53 germline rarely co-occur but increases risk; "
        ),
        "inheritance": "Autosomal dominant LOF; 40% de novo; bilateral RB PATHOGNOMONIC germline; two-hit model (germline first hit + somatic second hit); OS risk correlates with prior RT exposure; CDK4/6i inactive in RB1-null tumours",
        "cancer_risk": "Osteosarcoma 40% lifetime HIGHEST hereditary OS risk (especially post-RT); retinoblastoma (sentinel); soft tissue sarcoma 15-20x; melanoma 5-10x; pinealoblastoma (trilateral RB PATHOGNOMONIC); second primaries in RT field predominate",
        "pathognomonic": "Bilateral retinoblastoma in infant PATHOGNOMONIC germline RB1; trilateral RB (pinealoblastoma) PATHOGNOMONIC RB1; OS arising in prior RT field = RB1 germline mandatory; CDK4/6i INACTIVE in RB1-null OS tumours (no pRb substrate)",
        "surveillance_key": "Annual bone scintigraphy/MRI from age 6yr; AVOID re-irradiation at OS site; CDK4/6i INACTIVE in RB1-null tumours; MAP (methotrexate-doxorubicin-cisplatin) backbone; bilateral fundoscopy until age 5yr; cascade 50% first-degree",
        "key_distinctions": [
            "OS-40PCT-LIFETIME-HIGHEST-HEREDITARY-OS-RISK",
            "BILATERAL-RB-PATHOGNOMONIC-GERMLINE-RB1",
            "CDK4-6I-INACTIVE-RB1-NULL-TUMOURS",
            "AVOID-HIGH-DOSE-RT-OS-SITE",
            "TRILATERAL-RB-PINEALOBLASTOMA-PATHOGNOMONIC",
            "ANNUAL-BONE-SCINTIGRAPHY-MRI-FROM-AGE-6YR",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-OS-Number1-Cancer-LFS-Children-Anaplastic-Pleomorphic-"
            "PATHOGNOMONIC-AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; sequence-specific transcription factor; guardian of the genome; "
            "  N-terminal transactivation domain (aa 1-67): MDM2 binding; "
            "  Proline-rich domain (aa 68-98): apoptosis/cell cycle arrest switch; "
            "  DNA-binding domain (DBD; aa 102-292): most common mutation hotspot (R175/G245/R248/R249/R273/R282); "
            "  Tetramerisation domain (aa 323-356): p53 tetrameric active form; "
            "  C-terminal regulatory domain (aa 363-393): allosteric regulation; "
            "  TP53 LOF: loss of G1/S and G2/M checkpoints; apoptosis failure; "
            "    Dominant-negative GOF mutations (e.g. R248W): suppress wild-type p53 in heterozygous cells; "
            "LI-FRAUMENI SYNDROME (LFS) / TP53 OSTEOSARCOMA: "
            "  OMIM 151623; AD LOF; 7-20% de novo; "
            "  OS in LFS: #1 cancer type in LFS children; "
            "    OS PATHOGNOMONIC in LFS when anaplastic/pleomorphic histology; "
            "    OS median age in LFS: 10-15yr (earlier than sporadic OS at 15-20yr); "
            "    OS site: axial skeleton more common than sporadic OS; "
            "  LFS tumour spectrum: OS + soft tissue sarcoma (25-30% each); "
            "    Breast cancer (premenopausal women 25-30%); "
            "    Brain tumours (SHH-MB, DIPG, CPC, LGG 10-15%); "
            "    Adrenocortical carcinoma (ACC; infant sentinel tumour); "
            "    Leukaemia/lymphoma 5-10%; "
            "RADIATION IN LFS/TP53: "
            "  AVOID RADIATION ABSOLUTELY: germline TP53 → radiation hypersensitivity; "
            "    Historical LFS children treated with RT → second sarcomas in field; "
            "    Each Gy dramatically increases second primary rate in LFS; "
            "  WBMRI Toronto protocol: annually (brain + chest + abdomen + pelvis); "
            "    NO CT/PET scanning — AVOID ionising surveillance; "
            "  Proton beam: preferred if OS RT is unavoidable; "
            "TREATMENT TP53/LFS-OS: "
            "  MAP backbone (methotrexate-doxorubicin-cisplatin); "
            "  AVOID radiation fields as adjuvant/consolidation; "
            "  Limb-sparing surgery preferred over amputation (avoids RT); "
            "  APR-246 (eprenetapopt): TP53 reactivation; investigational; "
        ),
        "inheritance": "Autosomal dominant LOF; 7-20% de novo; near-complete penetrance by age 60yr (70-80%); females penetrance higher (breast) and earlier onset; R337H Brazilian founder (adrenocortical carcinoma HIGHEST risk); dominant-negative GOF mutations more severe",
        "cancer_risk": "Osteosarcoma 25-30% lifetime (#1 LFS childhood cancer); soft tissue sarcoma 25-30%; breast cancer 25-30% (premenopausal); brain tumours 10-15%; ACC 10-15% (R337H 70%); leukaemia 5-10%",
        "pathognomonic": "Anaplastic/pleomorphic OS in child/young adult PATHOGNOMONIC LFS TP53 germline; OS + second primary sarcoma = LFS PATHOGNOMONIC pattern; ACC in infant PATHOGNOMONIC LFS sentinel; R337H Brazilian founder 1/300 Southern Brazil",
        "surveillance_key": "WBMRI Toronto annually (brain+chest+abdomen+pelvis); AVOID RADIATION ABSOLUTELY; AVOID CT/PET; proton beam if RT unavoidable; breast MRI from age 20yr (females); cascade all first-degree 50% risk; APR-246 investigational; MAP backbone OS",
        "key_distinctions": [
            "OS-NUMBER1-CANCER-LFS-CHILDREN",
            "ANAPLASTIC-PLEOMORPHIC-OS-PATHOGNOMONIC-LFS",
            "AVOID-RADIATION-ABSOLUTELY-TP53-HYPERSENSITIVITY",
            "WBMRI-TORONTO-ANNUALLY-NOT-CT-PET",
            "R337H-BRAZILIAN-FOUNDER-1-IN-300-SOUTHERN-BRAZIL",
            "ACC-INFANT-SENTINEL-TUMOUR-LFS",
        ],
    },
    {
        "gene": "RECQL4",
        "protein": (
            "RECQL4 -- 8q24.12 Autosomal-Recessive-LOF -- 1208aa -- "
            "RECQL4-133kDa-RecQ-Helicase-RTS2-Rothmund-Thomson-Type2-OS-30-50pct-"
            "Poikiloderma-3-6months-PATHOGNOMONIC-Rapadilino-FIRES-Allelic-OMIM-603780"
        ),
        "locus": "8q24.12",
        "protein_size": (
            "1208 aa / 133 kDa / 8q24.12 RECQL4 encodes RecQ-like helicase 4: "
            "STRUCTURE: "
            "  1208 aa / 133 kDa; RecQ family 3'-5' DNA helicase; nuclear and mitochondrial; "
            "  N-terminal Sld2-like domain (aa 1-364): origin firing, TopBP1 interaction; "
            "  HRDC domain (aa 365-491): helicase/RNaseD C-terminal; nucleic acid binding; "
            "  Helicase core (aa 492-900): ATP hydrolysis + DNA unwinding; "
            "    Walker A motif (aa 540-547): ATP binding; "
            "    Walker B motif (aa 608-613): ATP hydrolysis; "
            "  RQC domain (aa 900-990): winged-helix + zinc-binding; "
            "  C-terminal domain (aa 990-1208): nuclear localisation; "
            "  RECQL4 function: replication initiation + Holliday junction resolution + BER coordination; "
            "ROTHMUND-THOMSON SYNDROME TYPE 2 (RTS2): "
            "  OMIM 268400; AR LOF; biallelic RECQL4 mutations; "
            "  PATHOGNOMONIC feature: "
            "    Poikiloderma: onset 3-6 months of age, face first then extremities; "
            "      Poikiloderma = erythema + telangiectasia + hypo/hyperpigmentation + atrophy; "
            "      PATHOGNOMONIC for RTS2 in infant; "
            "    Short stature; "
            "    Sparse/absent hair (alopecia, sparse eyebrows/eyelashes); "
            "    Radial ray defects (radius/thumb hypoplasia): 25-30% RTS2; "
            "    Cataracts (juvenile bilateral): 10-15%; "
            "  Osteosarcoma in RTS2: "
            "    OS risk: 30-50% lifetime (HIGHEST among DNA-repair syndromes); "
            "    OS age: predominantly 5-20yr; peak 10-15yr; "
            "    OS site: distal femur, proximal tibia (same sites as sporadic OS); "
            "  ALLELIC DISORDERS (RECQL4): "
            "    Rapadilino syndrome: radial/patella hypoplasia + no poikiloderma + OS risk; "
            "    FIRES syndrome (FANCL + RECQL4-related): rare; "
            "TREATMENT RTS2/RECQL4-OS: "
            "  MAP (methotrexate-adriamycin-cisplatin) standard OS backbone; "
            "  Skin photoprotection: MANDATORY lifelong (UV sensitivity in poikiloderma); "
            "  Cataract surveillance: annual slit-lamp from age 1yr; "
            "  AVOID unnecessary alkylating agents (potential FA-like sensitivity in some RECQL4 patients); "
        ),
        "inheritance": "Autosomal recessive LOF; biallelic mutations required for RTS2; heterozygous carriers unaffected; genotype-phenotype correlations: truncating mutations → higher OS risk; missense mutations with residual helicase activity → milder RTS2",
        "cancer_risk": "Osteosarcoma 30-50% lifetime (HIGHEST DNA-repair-syndrome OS risk); skin cancer (SCC, BCC) in poikiloderma field; lymphoma rare; RTS2 cancer risk predominantly OS in childhood-adolescence",
        "pathognomonic": "Poikiloderma onset 3-6 months of age PATHOGNOMONIC RTS2 (erythema → telangiectasia → hypo/hyperpigmentation → atrophy on face first); OS in child with poikiloderma = RECQL4 sequencing MANDATORY; radial ray defect + OS = Rapadilino/RECQL4 allelic",
        "surveillance_key": "Annual limb MRI/bone scintigraphy from age 5yr; skin photoprotection MANDATORY lifelong; annual slit-lamp for cataracts; RECQL4 sequencing MANDATORY poikiloderma + OS; MAP backbone OS treatment; genetic counselling both parents carriers; cascade testing siblings 25% risk",
        "key_distinctions": [
            "POIKILODERMA-3-6-MONTHS-PATHOGNOMONIC-RTS2",
            "OS-30-50PCT-LIFETIME-RECQL4",
            "RAPADILINO-FIRES-SYNDROMES-ALLELIC-RECQL4",
            "SKIN-PHOTOPROTECTION-MANDATORY-LIFELONG",
            "RADIAL-RAY-DEFECTS-25-30PCT-RTS2",
            "RECQL4-3PRIME-5PRIME-HELICASE-REPLICATION-INITIATION",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF-Monoallelic-AD-Biallelic-AR-FA-D1 -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FA-D1-OS-Bone-Tumours-PATHOGNOMONIC-"
            "AVOID-Alkylating-Sibling-Exclusion-MANDATORY-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes breast cancer type 2 susceptibility protein: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; large nuclear scaffold protein; homologous recombination (HR) repair; "
            "  N-terminal PALB2-binding domain (aa 10-40): PALB2 tethers BRCA2 to nuclear scaffold; "
            "  BRCA2-specific repeats BRC1-8 (aa 1002-2667): RAD51 binding x8; "
            "    Each BRC repeat recruits one RAD51 monomer for HR; "
            "  DBD/OB-fold domain (aa 2402-3190): ssDNA binding; "
            "  C-terminal DSS1/RAD51 binding (aa 3196-3418): DSS1 chaperone; "
            "  BRCA2 function: loads RAD51 onto RPA-coated ssDNA at DSB ends → HR repair; "
            "BIALLELIC BRCA2 (FANCONI ANAEMIA FA-D1) / OSTEOSARCOMA: "
            "  Biallelic BRCA2 → FA-D1 Fanconi anaemia; MOST SEVERE Fanconi group; "
            "  FA-D1 tumour spectrum (childhood onset): "
            "    Osteosarcoma/bone tumours: PATHOGNOMONIC FA-D1 skeletal malignancy; "
            "    Medulloblastoma (desmoplastic): PATHOGNOMONIC FA-D1; "
            "    Hepatoblastoma: PATHOGNOMONIC FA-D1; "
            "    Wilms tumour: PATHOGNOMONIC FA-D1; "
            "    AML (myeloid): PATHOGNOMONIC FA-D1; "
            "    Rhabdomyosarcoma: PATHOGNOMONIC FA-D1; "
            "  DEB (diepoxybutane) + MMC (mitomycin C) chromosomal fragility test PATHOGNOMONIC FA; "
            "  OS age in FA-D1: predominantly <10yr; "
            "TREATMENT FA-D1/BRCA2-OS: "
            "  AVOID alkylating agents (cyclophosphamide, melphalan, ifosfamide): lethal FA toxicity; "
            "  AVOID stem cell transplant from biologically related siblings: "
            "    Sibling donor exclusion: 25% chance sibling is also FA-D1 biallelic; "
            "    FA complementation testing of all potential sibling donors MANDATORY; "
            "  Cisplatin preferred over alkylating agents in OS backbone; "
            "  DEB/MMC test FIRST before any chemotherapy; "
        ),
        "inheritance": "Monoallelic BRCA2: AD LOF; HBOC (breast/ovarian); biallelic BRCA2: AR FA-D1; 25% sibling risk for FA if both parents carriers; FA-D1 OS predominantly early childhood compound heterozygous",
        "cancer_risk": "Monoallelic: breast (70-80% by age 80yr), ovarian (15-25%), pancreatic (5-7%), prostate; biallelic FA-D1: OS PATHOGNOMONIC + medulloblastoma + hepatoblastoma + Wilms + AML + RMS; FA-D1 cancer risk essentially 100% by age 10yr without treatment",
        "pathognomonic": "OS/bone tumours in child <10yr = FA-D1 BRCA2 testing MANDATORY; DEB/MMC chromosomal fragility PATHOGNOMONIC biallelic FA-D1; multiple childhood solid tumours (OS+MB or OS+Wilms) = FA-D1 PATHOGNOMONIC pattern; SIBLING DONOR EXCLUSION MANDATORY",
        "surveillance_key": "DEB/MMC chromosomal fragility test FIRST before chemotherapy; AVOID alkylating agents absolutely; sibling donor exclusion mandatory (FA complementation testing); cisplatin preferred over cyclophosphamide/ifosfamide; bone MRI annually FA-D1 known; brain MRI 3-monthly first 3yr",
        "key_distinctions": [
            "OS-BONE-TUMOURS-PATHOGNOMONIC-FA-D1-BRCA2",
            "AVOID-ALKYLATING-AGENTS-ABSOLUTELY-FA-D1",
            "SIBLING-DONOR-EXCLUSION-MANDATORY-FA-TESTING",
            "DEB-MMC-CHROMOSOMAL-FRAGILITY-PATHOGNOMONIC-FA",
            "CISPLATIN-PREFERRED-OVER-ALKYLATING-FA-D1-OS",
            "FA-D1-MULTI-TUMOUR-OS-MB-HBL-WILMS-AML-RMS",
        ],
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-NF1-Cafe-au-Lait-PATHOGNOMONIC-"
            "MPNST-8-13pct-OS-2-3x-Selumetinib-FDA2020-Pediatric-Plexiform-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes neurofibromin: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; RAS GTPase-activating protein (RAS-GAP); "
            "  N-terminal CRAL-TRIO domain (aa 1-256): lipid binding; "
            "  SEC14 domain (aa 1-256): intracellular lipid transfer; "
            "  PH domain (aa 1-256): membrane association; "
            "  GRD/RAS-GAP domain (aa 1198-1530): RAS-GAP catalytic core; "
            "    Accelerates RAS GTPase activity: RAS-GTP → RAS-GDP (inactivation); "
            "    NF1 LOF: RAS-GTP constitutively active → ERK/MAPK + PI3K/AKT hyperactivation; "
            "  CSRD (aa 1580-1780): cysteine/serine-rich domain; "
            "  Pre-GRD (aa 1161-1198): secondary RAS-GAP interaction; "
            "  C-terminal HEAT repeat domain (aa 2700-2839): protein-protein interaction; "
            "NEUROFIBROMATOSIS TYPE 1 (NF1): "
            "  OMIM 162200; AD LOF; 50% de novo; near-complete penetrance; "
            "  PATHOGNOMONIC diagnostic criteria (2 required for diagnosis): "
            "    Cafe-au-lait macules ≥6 (≥5mm prepubertal, ≥15mm postpubertal): PATHOGNOMONIC NF1; "
            "    Axillary/inguinal freckling: PATHOGNOMONIC NF1 (Crowe sign); "
            "    Lisch nodules (iris hamartomas ≥2): PATHOGNOMONIC NF1; "
            "    Optic pathway glioma (OPG); "
            "    Neurofibromas (≥2 cutaneous/subcutaneous OR ≥1 plexiform); "
            "    Osseous lesion (sphenoid wing dysplasia, tibial pseudarthrosis); "
            "    First-degree relative with NF1; "
            "OSTEOSARCOMA IN NF1: "
            "  OS risk: 2-3x elevated vs general population; "
            "  OS mechanism: NF1 LOF → RAS hyperactivation → enhanced osteoblast proliferation; "
            "  MPNST in NF1: 8-13% lifetime (much higher than OS); "
            "  MPNST arising from plexiform neurofibroma: most common NF1 high-grade malignancy; "
            "TREATMENT NF1: "
            "  Selumetinib (MEK1/2 inhibitor): FDA 2020 for pediatric plexiform neurofibromas; "
            "    Reduces plexiform volume in NF1 (MEK inhibition downstream RAS); "
            "  OS treatment: MAP backbone (standard); "
            "  AVOID whole-body radiation (secondary malignancy risk in NF1); "
        ),
        "inheritance": "Autosomal dominant LOF; 50% de novo; near-complete penetrance but variable expressivity; haploinsufficiency of neurofibromin → RAS-GAP loss → RAS hyperactivation; second-hit somatic NF1 mutation in tumours",
        "cancer_risk": "MPNST 8-13% lifetime; osteosarcoma 2-3x elevated; glioma (OPG, LGG) 15-20%; JMML 200x in children; leukaemia 5x; pheochromocytoma rare; carcinoid rare; gastrointestinal stromal tumour (GIST) rare",
        "pathognomonic": "Cafe-au-lait macules 6+ PATHOGNOMONIC NF1; axillary/inguinal freckling PATHOGNOMONIC NF1 (Crowe sign); Lisch nodules 2+ PATHOGNOMONIC NF1; plexiform neurofibroma any age PATHOGNOMONIC NF1; MPNST arising in NF1 plexiform = PATHOGNOMONIC malignant transformation",
        "surveillance_key": "Annual clinical exam + MRI plexiform monitoring; selumetinib FDA 2020 pediatric plexiform; MAP backbone for OS; AVOID whole-body radiation; OPG surveillance MRI to age 7yr in NF1 children; cascade 50% first-degree relatives",
        "key_distinctions": [
            "NF1-CAFE-AU-LAIT-MACULES-6-PLUS-PATHOGNOMONIC",
            "MPNST-8-13PCT-LIFETIME-NF1",
            "OS-2-3X-ELEVATED-NF1",
            "SELUMETINIB-MEK-INHIBITOR-FDA2020-PEDIATRIC-PLEXIFORM",
            "AXILLARY-FRECKLING-CROWE-SIGN-PATHOGNOMONIC",
            "AVOID-WHOLE-BODY-RADIATION-NF1",
        ],
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-ARF-CDK4-6i-Inactive-FAMMM-Familial-Melanoma-OS-Post-Irradiated-Sites-"
            "p16-IHC-Loss-PATHOGNOMONIC-Palbociclib-CDK4-Amp-OS-Pancreatic-20x-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 16 kDa / 9p21.3 CDKN2A encodes p16-INK4A (alternate reading frame also encodes p14-ARF): "
            "STRUCTURE: "
            "  p16-INK4A: 156 aa / 16 kDa; CDK4/CDK6 inhibitor; "
            "  4 ankyrin repeat domains (aa 1-156): CDK4/CDK6 binding surface; "
            "    p16 binds CDK4/CDK6 → prevents cyclin D binding → pRb remains hypophosphorylated; "
            "    Hypophosphorylated pRb sequesters E2F → G1 arrest; "
            "    CDKN2A LOF: CDK4/6 uninhibited → pRb hyperphosphorylated → E2F free → S phase; "
            "  p14-ARF: 132 aa alternate reading frame; MDM2 binding → p53 stabilisation; "
            "    CDKN2A locus LOF disrupts BOTH p16-INK4A (CDK4/6 pathway) AND p14-ARF (p53 pathway); "
            "  Combined CDK4/6 + p53 pathway loss from single locus deletion; "
            "FAMMM (FAMILIAL ATYPICAL MULTIPLE MOLE MELANOMA) / CDKN2A: "
            "  OMIM 600160; AD LOF; FAMMM diagnosis: ≥3 affected first/second-degree relatives; "
            "  Melanoma: 25-36x elevated; familial melanoma 10-15% carry germline CDKN2A; "
            "  Osteosarcoma in CDKN2A: "
            "    OS predominantly at post-irradiated sites (radiation-induced OS in prior RT field); "
            "    p16 IHC loss PATHOGNOMONIC in OS tumour (absent nuclear p16 staining); "
            "    CDKN2A deletion/LOF is common somatic event in sporadic OS (~25%); "
            "  Pancreatic adenocarcinoma: 20x elevated concurrent risk with melanoma; "
            "CDK4/6 INHIBITORS IN CDKN2A-LOF OS: "
            "  CDK4/CDK6 amplification: present in ~5-10% of sporadic OS; "
            "  CDK4/6i (palbociclib/ribociclib/abemaciclib): investigational in CDK4-amplified OS; "
            "    NB: in CDKN2A-LOF/RB1-proficient OS → CDK4/6i may be active (pRb still present); "
            "    DIFFERENT from RB1-null where CDK4/6i is inactive; "
        ),
        "inheritance": "Autosomal dominant LOF; 25-40% de novo; incomplete penetrance (melanoma penetrance ~58% by age 80yr European); CDKN2A deletion disrupts both p16-INK4A and p14-ARF reading frames; modifier locus MC1R increases melanoma penetrance",
        "cancer_risk": "Melanoma 25-36x elevated; pancreatic adenocarcinoma 20x elevated; osteosarcoma at post-irradiated sites; lung cancer 2-3x; CDKN2A somatic deletion in 25% of sporadic OS",
        "pathognomonic": "p16 IHC nuclear loss PATHOGNOMONIC CDKN2A LOF in OS tumour; OS arising at prior RT field + FAMMM family history = CDKN2A germline testing; melanoma + pancreatic cancer in same patient/family = CDKN2A testing MANDATORY",
        "surveillance_key": "Annual skin dermatology from age 18yr (melanoma); EUS/MRI pancreas from age 40yr or 10yr before index (pancreatic cancer 20x); p16 IHC on OS biopsy; CDK4/6i (palbociclib) investigational CDK4-amplified OS; cascade 50% first-degree relatives",
        "key_distinctions": [
            "P16-IHC-LOSS-PATHOGNOMONIC-CDKN2A-LOF",
            "OS-POST-IRRADIATED-SITES-CDKN2A",
            "PANCREATIC-ADENOCARCINOMA-20X-CONCURRENT-RISK",
            "CDK4-6I-PALBOCICLIB-INVESTIGATIONAL-CDK4-AMP-OS",
            "FAMMM-FAMILIAL-MELANOMA-CDKN2A",
            "CDKN2A-DISRUPTS-BOTH-P16-INK4A-AND-P14-ARF",
        ],
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-219kDa-RNase-III-DICER1-Syndrome-PPB-PATHOGNOMONIC-Mesenchymal-OS-2-4x-"
            "CT-Chest-Siblings-LT-8yr-AVOID-Radiation-Children-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 219 kDa / 14q32.13 DICER1 encodes DICER1 RNase III endonuclease: "
            "STRUCTURE: "
            "  1922 aa / 219 kDa; cytoplasmic RNase III; miRNA/siRNA processing; "
            "  DEAD-box helicase domain (aa 1-607): dsRNA recognition; "
            "  DUF283 domain (aa 608-784): partner RNA-protein interaction; "
            "  PAZ domain (aa 860-987): 3' end recognition of pre-miRNA; "
            "  RNase IIIa domain (aa 1287-1508): 5' strand cleavage of pre-miRNA; "
            "  RNase IIIb domain (aa 1534-1848): 3' strand cleavage of pre-miRNA; "
            "    Metal-binding residues (E1705, D1709, E1813, E1817): catalytic RNase activity; "
            "    Hotspot somatic mutations: E1705/D1709 metal-binding (recurrent in PPB/others); "
            "  dsRBD (aa 1849-1922): dsRNA binding; "
            "  DICER1 function: processes pre-miRNA → mature miRNA; processes dsRNA → siRNA; "
            "    DICER1 LOF: global miRNA processing defect → tumour-suppressor miRNA loss; "
            "DICER1 SYNDROME / OSTEOSARCOMA: "
            "  OMIM 606241; AD LOF; germline LOF (most exon truncating) + somatic missense RNase IIIb; "
            "  PATHOGNOMONIC tumours in DICER1 syndrome: "
            "    Pleuropulmonary blastoma (PPB): PATHOGNOMONIC DICER1 syndrome type I/Ir/II/III; "
            "    Cystic nephroma: PATHOGNOMONIC DICER1 syndrome; "
            "    SLCT (Sertoli-Leydig cell tumour): PATHOGNOMONIC DICER1 in young females; "
            "    Cervical ERMS: PATHOGNOMONIC DICER1; "
            "  Mesenchymal tumours including OS: 2-4x elevated risk; "
            "    Mechanism: miRNA loss → upregulation of growth-promoting pathways in mesenchyme; "
            "  CT chest in siblings <8yr: PPB surveillance MANDATORY; "
            "  AVOID radiation in children with DICER1 syndrome; "
        ),
        "inheritance": "Autosomal dominant LOF; 15-25% de novo; germline LOF mutation + somatic RNase IIIb hotspot = two-hit model in DICER1 tumours; haploinsufficiency contributes to tumour predisposition; CT chest surveillance MANDATORY siblings <8yr",
        "cancer_risk": "PPB type I PATHOGNOMONIC; cystic nephroma PATHOGNOMONIC; SLCT PATHOGNOMONIC young females; cervical ERMS PATHOGNOMONIC; mesenchymal tumours including OS 2-4x elevated; thyroid nodules/differentiated thyroid cancer; nasal chondromesenchymal hamartoma",
        "pathognomonic": "PPB type I in child <8yr PATHOGNOMONIC DICER1 syndrome; SLCT in female <40yr PATHOGNOMONIC DICER1; cystic nephroma in child PATHOGNOMONIC DICER1; CT chest of siblings <8yr MANDATORY if index case PPB",
        "surveillance_key": "CT chest siblings <8yr (PPB surveillance); annual pelvic US females from puberty (SLCT); thyroid US annually from age 8yr; AVOID radiation children with DICER1; DICER1 sequencing mandatory PPB/SLCT/cystic nephroma; MAP backbone OS; cascade 50% risk",
        "key_distinctions": [
            "PPB-TYPE-I-PATHOGNOMONIC-DICER1-SYNDROME",
            "CT-CHEST-SIBLINGS-LT-8YR-MANDATORY",
            "MESENCHYMAL-OS-2-4X-ELEVATED-DICER1",
            "SLCT-YOUNG-FEMALE-PATHOGNOMONIC-DICER1",
            "CYSTIC-NEPHROMA-PATHOGNOMONIC-DICER1",
            "AVOID-RADIATION-CHILDREN-DICER1",
        ],
    },
    {
        "gene": "WRN",
        "protein": (
            "WRN -- 8p12 Autosomal-Recessive-LOF -- 1432aa -- "
            "WRN-162kDa-Werner-Helicase-Exonuclease-Werner-Syndrome-Adult-Onset-Bilateral-Cataracts-PATHOGNOMONIC-"
            "Mesenchymal-Cancer-OS-Type2-DM-Scleroderma-PATHOGNOMONIC-Vemurafenib-Melanoma-OMIM-277700"
        ),
        "locus": "8p12",
        "protein_size": (
            "1432 aa / 162 kDa / 8p12 WRN encodes Werner syndrome helicase: "
            "STRUCTURE: "
            "  1432 aa / 162 kDa; RecQ family helicase + unique 3'-5' exonuclease domain; "
            "  Exonuclease domain (aa 1-236): 3'-5' exonuclease; "
            "    Only RecQ helicase with exonuclease activity; "
            "    Degrades mismatched DNA, processes Okazaki fragments; "
            "  HRDC domain (aa 237-330): auxiliary DNA binding; "
            "  RecQ helicase core (aa 500-900): ATP-dependent 3'-5' DNA unwinding; "
            "    Walker A/B motifs: ATP binding/hydrolysis; "
            "  RQC domain (aa 950-1080): winged-helix + zinc-binding; "
            "    RecQ C-terminal: protein-protein interactions (p53, PARP1, FEN1, PCNA); "
            "  HRDC2 (aa 1072-1150): auxiliary interaction domain; "
            "  NLS (aa 1370-1432): nuclear localisation signal; "
            "  WRN function: HR repair + replication fork restart + telomere maintenance; "
            "    WRN LOF: telomere shortening; premature replication fork collapse; genomic instability; "
            "WERNER SYNDROME: "
            "  OMIM 277700; AR LOF; biallelic WRN mutations; "
            "  PATHOGNOMONIC features (Werner tetrad): "
            "    (1) Adult-onset bilateral cataracts: onset 20s-30s PATHOGNOMONIC Werner; "
            "    (2) Type 2 diabetes mellitus: onset 30s-40s; insulin resistance; PATHOGNOMONIC Werner triad; "
            "    (3) Scleroderma-like skin changes: leg ulcers, calcinosis cutis: PATHOGNOMONIC Werner triad; "
            "    (4) Short stature: absence of pubertal growth spurt; "
            "  Additional Werner features: premature greying, premature hair loss, hoarse voice; "
            "  Progeria of adulthood: accelerated ageing phenotype from age 20s; "
            "OSTEOSARCOMA IN WERNER: "
            "  OS risk: dramatically elevated; predominantly adult OS (age 20s-40s); "
            "  Mesenchymal cancer predominance: OS, soft tissue sarcoma, leiomyosarcoma, myxoid liposarcoma; "
            "  Werner cancer spectrum: ~66% of Werner cancers are mesenchymal (vs ~1% in general population); "
            "  Melanoma: elevated; vemurafenib for BRAF V600E-positive melanoma component; "
        ),
        "inheritance": "Autosomal recessive LOF; biallelic WRN mutations; heterozygous carriers have slightly elevated cancer risk (inconclusive data); Japanese founder mutations (c.3139-1G>A splice + 1 exonic) account for 50% of Japanese Werner cases; consanguineous families overrepresented",
        "cancer_risk": "Osteosarcoma (adult, mesenchymal predominance); soft tissue sarcoma 40-50x elevated; melanoma elevated; thyroid cancer; meningioma; leukemia; Werner mesenchymal cancer spectrum = ~66% of all Werner cancers",
        "pathognomonic": "Adult-onset bilateral cataracts onset 20s-30s PATHOGNOMONIC Werner syndrome; type 2 DM + scleroderma + cataracts PATHOGNOMONIC Werner triad; premature ageing + mesenchymal sarcoma in adult = WRN sequencing MANDATORY; Werner OS age 20-40yr (vs sporadic OS age 15-25yr)",
        "surveillance_key": "Annual slit-lamp from age 20yr (cataracts); annual HbA1c + fasting glucose from age 25yr (DM); wound care + dermatology (leg ulcers); lipid screen + cardiovascular annually; WRN sequencing mandatory premature ageing + sarcoma; MAP backbone OS; vemurafenib BRAF V600E melanoma; cascade 25% siblings",
        "key_distinctions": [
            "ADULT-ONSET-BILATERAL-CATARACTS-PATHOGNOMONIC-WERNER",
            "TYPE2-DM-SCLERODERMA-PATHOGNOMONIC-WERNER-TRIAD",
            "MESENCHYMAL-CANCER-PREDOMINANCE-66PCT-WERNER",
            "OS-ADULT-ONSET-AGE-20-40YR-WERNER",
            "VEMURAFENIB-BRAF-V600E-MELANOMA-WERNER",
            "WRN-UNIQUE-EXONUCLEASE-DOMAIN-RECQ-FAMILY",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Simulated tumour types per gene (osteosarcoma context)
# ─────────────────────────────────────────────────────────────────────────────
TUMOUR_TYPES_BY_GENE = {
    "RB1":    ["OS-post-RT-field-RB1", "OS-conventional-RB1", "OS-distal-femur-RB1", "OS-proximal-tibia-RB1", "STS-secondary-RB1"],
    "TP53":   ["OS-anaplastic-LFS", "OS-pleomorphic-LFS", "OS-axial-LFS", "OS-multifocal-LFS", "OS-SHH-associated-LFS"],
    "RECQL4": ["OS-distal-femur-RTS2", "OS-proximal-tibia-RTS2", "OS-conventional-RECQL4", "OS-poikiloderma-context", "OS-Rapadilino"],
    "BRCA2":  ["OS-FA-D1-bone", "OS-FA-D1-long-bone", "Bone-tumour-FA-D1", "OS-biallelic-BRCA2", "OS-childhood-FA-D1"],
    "NF1":    ["OS-NF1-long-bone", "OS-NF1-conventional", "MPNST-NF1", "OS-NF1-extremity", "OS-NF1-post-RT"],
    "CDKN2A": ["OS-post-irradiated-CDKN2A", "OS-p16-null", "OS-CDK4-amplified", "OS-radiation-induced", "OS-CDKN2A-del"],
    "DICER1": ["OS-mesenchymal-DICER1", "OS-DICER1-syndrome", "PPB-type-I-DICER1", "OS-extremity-DICER1", "Mesenchymal-DICER1"],
    "WRN":    ["OS-adult-Werner", "OS-mesenchymal-Werner", "STS-leiomyosarcoma-Werner", "OS-Werner-age30", "OS-Werner-age25"],
}

VARIANTS_BY_GENE = {
    "RB1":    ["p.Arg320Ter (nonsense exon 10)", "p.Tyr180His (splice-site DBF)", "del exon 1-6 (large deletion)", "p.Arg661Ter (nonsense C-term)", "p.Asp401Gly (pocket A missense)"],
    "TP53":   ["p.Arg248Trp (hotspot GOF)", "p.Arg273His (hotspot GOF)", "p.Arg175His (dominant-negative)", "R337H (Brazilian founder)", "del 17p13.1 (gross deletion LFS)"],
    "RECQL4": ["p.Ala459Val (helicase core)", "p.Arg892Ter (helicase truncation)", "del exon 9-14 (helicase partial)", "p.Gln1186Ter (C-terminal truncation)", "p.Arg255Ter (HRDC truncation)"],
    "BRCA2":  ["p.Leu1455Ter (BRC repeat)", "p.Trp2626Ter (DBD truncation)", "del exon 15-24 (partial deletion)", "p.Arg2830Ter (DBD truncation)", "p.Cys3080Arg (OB-fold missense)"],
    "NF1":    ["p.Arg1241Ter (GRD domain)", "del exon 14-22 (GRD partial del)", "p.Tyr2264Ter (C-terminal)", "p.Glu1350Lys (GRD surface)", "c.4110+1G>A (GRD splice)"],
    "CDKN2A": ["p.Arg58Ter (ankyrin 1)", "p.Gly101Trp (ankyrin 2 missense)", "del 9p21.3 (homozygous del)", "p.Ala148Thr (C-term ankyrin)", "p.Arg80Ter (ankyrin 2 truncation)"],
    "DICER1": ["p.Glu1705Lys (RNase IIIb hotspot)", "p.Asp1709Glu (RNase IIIb hotspot)", "p.Gln1802Ter (RNase IIIb truncation)", "del exon 21-25 (RNase IIIb del)", "p.Glu1813Gln (metal binding)"],
    "WRN":    ["c.3139-1G>A (splice Japanese founder)", "p.Arg369Ter (HRDC truncation)", "p.Cys1367Ter (NLS truncation)", "del exon 7-11 (helicase partial)", "p.Leu1074Pro (RQC helicase-breaking)"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "RB1":    ["MAP (methotrexate-doxorubicin-cisplatin): standard OS backbone", "CDK4/6i INACTIVE in RB1-null OS — do not use as monotherapy", "AVOID re-irradiation at OS site (secondary OS risk)", "Annual bone scintigraphy/MRI from age 6yr in germline RB1 survivors"],
    "TP53":   ["MAP backbone; AVOID radiation adjuvant/consolidation ABSOLUTELY", "Limb-sparing surgery preferred (avoids RT requirement)", "WBMRI Toronto annually; AVOID CT/PET", "APR-246 (eprenetapopt): TP53 reactivation investigational"],
    "RECQL4": ["MAP backbone (methotrexate-adriamycin-cisplatin) for OS", "Skin photoprotection MANDATORY lifelong (UV sensitivity poikiloderma)", "Annual slit-lamp for cataracts; bone MRI annually from age 5yr", "RECQL4 sequencing MANDATORY poikiloderma + OS family"],
    "BRCA2":  ["DEB/MMC chromosomal fragility test FIRST before any chemotherapy", "AVOID alkylating agents ABSOLUTELY (cyclophosphamide lethal FA toxicity)", "Cisplatin preferred over ifosfamide/cyclophosphamide in OS backbone", "Sibling donor exclusion mandatory (FA complementation testing)"],
    "NF1":    ["MAP backbone for OS; selumetinib FDA 2020 for pediatric plexiform neurofibromas", "AVOID whole-body radiation (secondary malignancy NF1)", "Annual MRI plexiform monitoring; OPG surveillance MRI to age 7yr", "MEK inhibitor (selumetinib/trametinib) for inoperable plexiform"],
    "CDKN2A": ["MAP backbone for OS; CDK4/6i palbociclib investigational CDK4-amplified OS", "Annual skin dermatology from age 18yr (melanoma surveillance)", "EUS/MRI pancreas from age 40yr (pancreatic 20x risk)", "p16 IHC on OS biopsy (p16-null confirms CDKN2A LOF)"],
    "DICER1": ["MAP backbone for OS; CT chest siblings <8yr (PPB surveillance)", "AVOID radiation in children with DICER1 syndrome", "Annual pelvic US females from puberty (SLCT surveillance)", "Thyroid US annually from age 8yr (thyroid nodules/DTC in DICER1)"],
    "WRN":    ["MAP backbone for OS; vemurafenib for BRAF V600E melanoma", "Annual slit-lamp from age 20yr; annual HbA1c + fasting glucose", "Wound care + dermatology (leg ulcers + scleroderma); cardiovascular surveillance", "WRN sequencing mandatory premature ageing + sarcoma in adult"],
}

SURVEILLANCE_BY_GENE = {
    "RB1": [
        "Annual bone scintigraphy/MRI from age 6yr (OS surveillance germline RB1)",
        "Annual whole-body MRI or bone scan post-OS diagnosis",
        "AVOID re-irradiation at OS site in RB1 survivors",
        "CDK4/6i: not effective in RB1-null OS — do not apply",
        "CASCADE 50% offspring risk; bilateral fundoscopy until age 5yr",
    ],
    "TP53": [
        "WBMRI Toronto annually: brain + chest + abdomen + pelvis",
        "Rapid MRI 6-monthly first 5yr; annual thereafter",
        "Breast MRI from age 20yr (females; or 10yr before index case)",
        "Adrenal/abdominal US q3-6m infancy (ACC sentinel LFS)",
        "AVOID CT/PET; AVOID radiation absolutely; proton beam if unavoidable",
    ],
    "RECQL4": [
        "Annual limb MRI/bone scintigraphy from age 5yr (OS surveillance RTS2)",
        "Annual slit-lamp examination from age 1yr (cataract surveillance)",
        "Skin photoprotection: UVA/UVB, regular dermatology from infancy",
        "RECQL4 sequencing MANDATORY poikiloderma onset 3-6 months + OS family history",
        "CASCADE: 25% sibling risk (AR); both parents carriers; prenatal testing option",
    ],
    "BRCA2": [
        "DEB/MMC chromosomal fragility test MANDATORY before any chemotherapy",
        "Bone MRI annually: birth to age 10yr (FA-D1 OS surveillance)",
        "Brain MRI every 3 months: birth to age 3yr (medulloblastoma FA-D1)",
        "AFP + liver US q3-6m birth to 7yr (hepatoblastoma FA-D1)",
        "SIBLING DONOR EXCLUSION: FA complementation testing of all potential sibling donors",
    ],
    "NF1": [
        "Annual clinical exam + full-body skin exam (cafe-au-lait + neurofibroma)",
        "MRI plexiform neurofibroma: baseline + when clinically indicated for growth",
        "OPG surveillance MRI annually age 2-7yr (optic pathway glioma NF1)",
        "Selumetinib: pediatric progressive/symptomatic inoperable plexiform",
        "CASCADE: 50% offspring risk; annual annual assessment age 0-18yr",
    ],
    "CDKN2A": [
        "Annual full-body skin exam dermatology from age 18yr (melanoma surveillance)",
        "EUS or MRI pancreas annually from age 40yr or 10yr before index case (pancreatic 20x)",
        "p16 IHC on all OS biopsies (absent nuclear p16 confirms CDKN2A LOF)",
        "CDK4 FISH/CGH: CDK4-amplified OS → palbociclib investigational trial eligibility",
        "CASCADE: 50% first-degree relatives; CDKN2A sequencing mandatory FAMMM",
    ],
    "DICER1": [
        "CT chest: all siblings <8yr of PPB index case (PPB surveillance MANDATORY)",
        "Annual pelvic US from puberty: SLCT surveillance in females",
        "Annual thyroid US from age 8yr (thyroid nodules/differentiated thyroid cancer)",
        "Renal US annually: cystic nephroma surveillance birth to age 8yr",
        "CASCADE: 50% offspring risk; DICER1 sequencing mandatory PPB/SLCT/cystic nephroma",
    ],
    "WRN": [
        "Annual slit-lamp examination from age 20yr (bilateral cataract surveillance)",
        "Annual fasting glucose + HbA1c from age 25yr (type 2 DM surveillance)",
        "Annual lipid panel + cardiovascular assessment (premature atherosclerosis Werner)",
        "Annual dermatology: leg ulcers + scleroderma-like changes; wound care",
        "CASCADE: 25% sibling risk (AR); WRN sequencing mandatory premature ageing + sarcoma",
    ],
}


def _make_patient(gene_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_index]["gene"]
    tumour = rng.choice(TUMOUR_TYPES_BY_GENE[gene])
    variant = rng.choice(VARIANTS_BY_GENE[gene])

    # Age at diagnosis (osteosarcoma context — gene-appropriate)
    if gene == "RB1":
        age = rng.randint(8, 16)    # secondary OS after retinoblastoma RT, age 8-16yr
    elif gene == "TP53":
        age = rng.randint(5, 15)    # LFS OS #1 childhood cancer, mean ~10yr
    elif gene == "RECQL4":
        age = rng.randint(5, 18)    # RTS2 OS peak 10-15yr
    elif gene == "BRCA2":
        age = rng.randint(3, 12)    # FA-D1 very early onset OS <10yr
    elif gene == "NF1":
        age = rng.randint(10, 20)   # NF1 OS slightly older than LFS/RB1
    elif gene == "CDKN2A":
        age = rng.randint(14, 24)   # post-irradiation sites, adults
    elif gene == "DICER1":
        age = rng.randint(8, 18)    # DICER1 syndrome mesenchymal
    elif gene == "WRN":
        age = rng.randint(22, 35)   # Werner adult onset OS age 20-40yr
    else:
        age = rng.randint(10, 20)

    # CR rates (gene-appropriate — OS generally poor prognosis)
    cr_rates = {
        "RB1":    0.55,   # poor — secondary OS post-RT, CDK4/6i inactive
        "TP53":   0.45,   # worst — anaplastic/pleomorphic OS LFS
        "RECQL4": 0.60,   # moderate
        "BRCA2":  0.50,   # limited by FA chemotherapy constraints
        "NF1":    0.62,   # moderate
        "CDKN2A": 0.58,   # moderate — adult OS
        "DICER1": 0.65,   # relatively better mesenchymal
        "WRN":    0.52,   # adult Werner OS
    }
    cr = rng.random() < cr_rates.get(gene, 0.55)

    # Radiation rates (gene-appropriate — strict avoidance in many)
    rad_rates = {
        "RB1":    0.05,   # AVOID high-dose RT at OS site
        "TP53":   0.00,   # AVOID RADIATION ABSOLUTELY
        "RECQL4": 0.20,   # some limited RT but cautious
        "BRCA2":  0.10,   # caution FA radiosensitivity
        "NF1":    0.25,   # selective
        "CDKN2A": 0.30,   # post-irradiation context but limited further RT
        "DICER1": 0.20,   # AVOID radiation children
        "WRN":    0.15,   # genomic instability — limited RT
    }
    radiation = rng.random() < rad_rates.get(gene, 0.15)

    # Chemotherapy used (nearly all OS patients get chemo)
    chemo_rates = {
        "RB1":    0.92, "TP53":   0.95, "RECQL4": 0.90, "BRCA2":  0.88,
        "NF1":    0.90, "CDKN2A": 0.88, "DICER1": 0.90, "WRN":    0.85,
    }
    targeted = rng.random() < chemo_rates.get(gene, 0.90)

    # HSCT rates (rarely used in solid OS; higher for FA-D1 BRCA2)
    hsct_rates = {
        "RB1":    0.05, "TP53":   0.08, "RECQL4": 0.06, "BRCA2":  0.25,
        "NF1":    0.04, "CDKN2A": 0.04, "DICER1": 0.05, "WRN":    0.06,
    }
    transplant = rng.random() < hsct_rates.get(gene, 0.06)

    relapse = rng.random() < (0.20 if cr else 0.65)

    return {
        "gene": gene,
        "seed": seed,
        "age_dx": age,
        "tumour_type": tumour,
        "variant": variant,
        "cr": cr,
        "radiation": radiation,
        "targeted": targeted,
        "transplant": transplant,
        "relapse": relapse,
    }


def _generate_cohort() -> list:
    patients = []
    for gi in range(8):
        for offset in range(40):
            patients.append(_make_patient(gi, SEED_BASE + gi * 40 + offset))
    return patients


def generate_overview() -> dict:
    cohort = _generate_cohort()
    n = len(cohort)
    cr_pct = round(100 * sum(p["cr"] for p in cohort) / n)
    radiation_pct = round(100 * sum(p["radiation"] for p in cohort) / n)
    chemo_pct = round(100 * sum(p["targeted"] for p in cohort) / n)
    hsct_pct = round(100 * sum(p["transplant"] for p in cohort) / n)
    relapse_pct = round(100 * sum(p["relapse"] for p in cohort) / n)
    mean_age = round(sum(p["age_dx"] for p in cohort) / n, 1)

    from collections import Counter
    tumour_counter = Counter(p["tumour_type"] for p in cohort)
    top_tumours = dict(tumour_counter.most_common(8))

    gene_summaries = []
    for gi, ginfo in enumerate(ATLAS_GENES):
        gp = [p for p in cohort if p["gene"] == ginfo["gene"]]
        gn = len(gp)
        gene_summaries.append({
            "gene": ginfo["gene"],
            "locus": ginfo["locus"],
            "n_patients": gn,
            "mean_age_dx": round(sum(p["age_dx"] for p in gp) / gn, 1),
            "cr_pct": round(100 * sum(p["cr"] for p in gp) / gn),
            "radiation_pct": round(100 * sum(p["radiation"] for p in gp) / gn),
            "chemo_pct": round(100 * sum(p["targeted"] for p in gp) / gn),
            "hsct_pct": round(100 * sum(p["transplant"] for p in gp) / gn),
            "relapse_pct": round(100 * sum(p["relapse"] for p in gp) / gn),
            "cancer_risk": ginfo["cancer_risk"],
        })

    return {
        "atlas": "Hereditary-Osteosarcoma-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference: RB1 · TP53 · RECQL4 · BRCA2 · NF1 · CDKN2A · DICER1 · WRN",
        "seeds": "3310-3317",
        "total_patients": n,
        "cr_pct": cr_pct,
        "radiation_pct": radiation_pct,
        "chemo_pct": chemo_pct,
        "hsct_pct": hsct_pct,
        "relapse_pct": relapse_pct,
        "mean_age_dx": mean_age,
        "top_tumour_types": top_tumours,
        "gene_summaries": gene_summaries,
        "atlas_note": (
            "Hereditary Osteosarcoma Predisposition Atlas: "
            "RB1 germline 40% OS lifetime HIGHEST hereditary risk (bilateral RB PATHOGNOMONIC; CDK4/6i INACTIVE RB1-null; AVOID RT at OS site), "
            "TP53 LFS OS #1 LFS childhood cancer (anaplastic/pleomorphic PATHOGNOMONIC; AVOID RADIATION ABSOLUTELY; WBMRI Toronto), "
            "RECQL4 RTS2 OS 30-50% (poikiloderma 3-6 months PATHOGNOMONIC; AVOID alkylating high-dose), "
            "BRCA2 FA-D1 OS bone tumours PATHOGNOMONIC (AVOID alkylating ABSOLUTELY; sibling donor exclusion MANDATORY), "
            "NF1 OS 2-3x (cafe-au-lait PATHOGNOMONIC; MPNST 8-13%; selumetinib FDA 2020), "
            "CDKN2A p16 IHC loss PATHOGNOMONIC (OS post-irradiated sites; pancreatic 20x; CDK4/6i palbociclib CDK4-amp), "
            "DICER1 PPB PATHOGNOMONIC (mesenchymal/OS 2-4x; CT chest siblings <8yr; AVOID RT children), "
            "WRN adult bilateral cataracts PATHOGNOMONIC (mesenchymal predominance; DM + scleroderma PATHOGNOMONIC; vemurafenib). "
            "320-patient aggregate 8x40, seeds 3310-3317."
        ),
    }


def generate_breakdown() -> dict:
    cohort = _generate_cohort()
    from collections import Counter
    breakdown = []
    for gi, ginfo in enumerate(ATLAS_GENES):
        gp = [p for p in cohort if p["gene"] == ginfo["gene"]]
        gn = len(gp)
        tumour_counter = Counter(p["tumour_type"] for p in gp)
        variant_counter = Counter(p["variant"] for p in gp)
        breakdown.append({
            "gene": ginfo["gene"],
            "locus": ginfo["locus"],
            "n_patients": gn,
            "mean_age_dx": round(sum(p["age_dx"] for p in gp) / gn, 1),
            "cr_pct": round(100 * sum(p["cr"] for p in gp) / gn),
            "radiation_pct": round(100 * sum(p["radiation"] for p in gp) / gn),
            "chemo_pct": round(100 * sum(p["targeted"] for p in gp) / gn),
            "hsct_pct": round(100 * sum(p["transplant"] for p in gp) / gn),
            "relapse_pct": round(100 * sum(p["relapse"] for p in gp) / gn),
            "inheritance": ginfo["inheritance"],
            "cancer_risk": ginfo["cancer_risk"],
            "pathognomonic": ginfo["pathognomonic"],
            "top_tumor_types": dict(tumour_counter.most_common(5)),
            "top_variants": dict(variant_counter.most_common(4)),
            "treatment_protocols": TREATMENT_PROTOCOLS_BY_GENE[ginfo["gene"]],
            "surveillance_protocols": SURVEILLANCE_BY_GENE[ginfo["gene"]],
            "key_distinctions": ginfo["key_distinctions"],
        })
    return {"atlas": "Hereditary-Osteosarcoma-Predisposition-Atlas", "breakdown": breakdown}


def generate_definitions() -> dict:
    definitions = {}
    for ginfo in ATLAS_GENES:
        definitions[ginfo["gene"]] = {
            "locus": ginfo["locus"],
            "protein": ginfo["protein"],
            "protein_size": ginfo["protein_size"],
            "inheritance": ginfo["inheritance"],
            "cancer_risk": ginfo["cancer_risk"],
            "pathognomonic": ginfo["pathognomonic"],
            "surveillance_key": ginfo["surveillance_key"],
            "key_distinctions": ginfo["key_distinctions"],
        }

    return {
        "atlas": "Hereditary-Osteosarcoma-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "RB1_OS_HIGHEST_CDK46I_INACTIVE": (
                "RB1 germline: OS 40% lifetime HIGHEST hereditary OS risk. Bilateral RB PATHOGNOMONIC germline. "
                "CDK4/6i (palbociclib/ribociclib) INACTIVE in RB1-null OS — pRb is the drug's substrate; absence of pRb = no target. "
                "AVOID high-dose re-irradiation at OS site. Annual bone scintigraphy/MRI from age 6yr survivors."
            ),
            "TP53_LFS_OS_AVOID_RADIATION": (
                "TP53 LFS: OS #1 cancer in LFS children. Anaplastic/pleomorphic OS PATHOGNOMONIC LFS germline. "
                "AVOID RADIATION ABSOLUTELY — germline TP53 hypersensitivity → radiation-induced second sarcomas. "
                "WBMRI Toronto annually (NOT CT/PET). MAP backbone OS. Limb-sparing surgery to avoid RT requirement."
            ),
            "RECQL4_RTS2_POIKILODERMA_OS": (
                "RECQL4 RTS2: Poikiloderma onset 3-6 months PATHOGNOMONIC (erythema → telangiectasia → pigmentation → atrophy, face first). "
                "OS 30-50% lifetime HIGHEST among DNA-repair syndromes for OS. Rapadilino/FIRES syndromes allelic. "
                "Skin photoprotection MANDATORY lifelong. Annual bone MRI from age 5yr. MAP backbone OS."
            ),
            "BRCA2_FA_D1_AVOID_ALKYLATING": (
                "BRCA2 FA-D1: OS/bone tumours PATHOGNOMONIC FA-D1 (with MB, HBL, Wilms, AML, RMS). "
                "DEB/MMC chromosomal fragility PATHOGNOMONIC FA — perform FIRST before any chemotherapy. "
                "AVOID alkylating agents absolutely (cyclophosphamide/ifosfamide lethal FA toxicity). "
                "SIBLING DONOR EXCLUSION MANDATORY — FA complementation testing of all potential sibling donors."
            ),
            "NF1_OS_MPNST_SELUMETINIB": (
                "NF1: Cafe-au-lait macules 6+ PATHOGNOMONIC NF1. MPNST 8-13% lifetime. OS 2-3x elevated. "
                "Selumetinib (MEK1/2 inhibitor) FDA 2020 for pediatric progressive inoperable plexiform neurofibromas. "
                "AVOID whole-body radiation (secondary malignancy in NF1). MAP backbone for OS."
            ),
            "CDKN2A_P16_IHC_CDK46I_PANCREATIC": (
                "CDKN2A: p16 IHC nuclear loss PATHOGNOMONIC CDKN2A LOF in OS tumour. OS at post-irradiated sites. "
                "Pancreatic adenocarcinoma 20x concurrent risk. CDK4/6i palbociclib investigational in CDK4-amplified OS. "
                "Annual skin dermatology (melanoma). EUS/MRI pancreas from age 40yr."
            ),
            "DICER1_PPB_CT_CHEST_SIBLINGS": (
                "DICER1 syndrome: PPB type I PATHOGNOMONIC DICER1 (all PPB types PATHOGNOMONIC). "
                "CT chest in ALL siblings <8yr of PPB index case — PPB surveillance MANDATORY. "
                "Mesenchymal tumours including OS 2-4x elevated. AVOID radiation children with DICER1. "
                "SLCT in female <40yr and cystic nephroma in child PATHOGNOMONIC DICER1."
            ),
            "WRN_CATARACTS_SCLERODERMA_MESENCHYMAL": (
                "Werner syndrome: Adult-onset bilateral cataracts onset 20s-30s PATHOGNOMONIC Werner. "
                "Type 2 DM + scleroderma-like skin changes PATHOGNOMONIC Werner triad. "
                "Mesenchymal cancer predominance (~66% Werner cancers are mesenchymal — OS + STS). "
                "Vemurafenib for BRAF V600E-positive melanoma component. Annual slit-lamp from age 20yr."
            ),
        },
        "cascade_testing_rule": (
            "Hereditary Osteosarcoma Predisposition Atlas — Cascade Testing: "
            "RB1: bilateral RB = germline testing; OS survivors annual bone scintigraphy/MRI from age 6yr; AVOID RT OS site; CDK4/6i INACTIVE RB1-null; cascade 50%. "
            "TP53 LFS: OS anaplastic PATHOGNOMONIC; WBMRI annually; AVOID RADIATION ABSOLUTELY; cascade 50% first-degree. "
            "RECQL4 RTS2: poikiloderma 3-6 months PATHOGNOMONIC; AR — both parents carriers; sibling 25% risk; annual bone MRI age 5yr+. "
            "BRCA2 FA-D1: DEB/MMC FIRST; AVOID alkylating; sibling donor exclusion MANDATORY; cascade 50% monoallelic. "
            "NF1: cafe-au-lait 6+ PATHOGNOMONIC; selumetinib pediatric plexiform; AVOID whole-body RT; cascade 50%. "
            "CDKN2A: p16 IHC on all OS; melanoma + pancreatic = CDKN2A mandatory; palbociclib CDK4-amp OS investigational; cascade 50%. "
            "DICER1: PPB = DICER1 sequencing mandatory; CT chest ALL siblings <8yr; SLCT surveillance females; cascade 50%. "
            "WRN: AR — 25% sibling risk; annual slit-lamp from age 20yr; DM surveillance age 25yr+; wound care scleroderma."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["breakdown"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first gene) ===")
    df = generate_definitions()
    print(json.dumps(df["definitions"]["RB1"], indent=2)[:1500])
