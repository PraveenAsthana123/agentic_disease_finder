#!/usr/bin/env python3
"""Hereditary-Medulloblastoma-Predisposition-Atlas -- Complete 8-Gene Reference
PTCH1   (Patched-1 Receptor; 1447aa; 9q22.32; AD LOF;
         Gorlin syndrome / Naevoid Basal Cell Carcinoma Syndrome (NBCCS);
         SHH-medulloblastoma germline 25-30% of SHH subgroup;
         desmoplastic/nodular MB PATHOGNOMONIC PTCH1-Gorlin;
         AVOID RADIATION ABSOLUTELY — each Gray induces ~10 BCCs;
         calcified falx cerebri + jaw keratocysts PATHOGNOMONIC;
         seed SEED_BASE+0) .
SUFU    (Suppressor of Fused homologue; 484aa; 10q24.32; AD LOF;
         Gorlin-like SHH pathway distal;
         SHH-MB germline HIGHEST 50-60% lifetime risk;
         desmoplastic/nodular MB PATHOGNOMONIC;
         AVOID vismodegib in growing skeleton — growth plate fusion;
         AVOID radiation children — secondary BCC;
         seed SEED_BASE+1) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         SHH-MB + anaplastic histology PATHOGNOMONIC LFS;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         ONC201/dordaviprone FDA2022 H3K27M+;
         seed SEED_BASE+2) .
APC     (Adenomatous polyposis coli; 2843aa; 5q22.2; AD LOF;
         FAP / Turcot Type 2 (APC-associated brain tumours);
         WNT-medulloblastoma 0.5-1% FAP lifetime; monosomy 6 excellent prognosis;
         CAPP5 MB surveillance; colonoscopy from age 10-12yr;
         seed SEED_BASE+3) .
CREBBP  (CREB-binding protein; 2441aa; 16p13.3; AD LOF;
         Rubinstein-Taybi Syndrome type 1;
         WNT-MB; broad thumbs + big toes PATHOGNOMONIC;
         intellectual disability 100%; CDDP-based chemotherapy standard;
         seed SEED_BASE+4) .
EP300   (E1A-binding protein p300; 2414aa; 22q13.2; AD LOF;
         Rubinstein-Taybi Syndrome type 2;
         WNT-MB same pathway as CREBBP; phenotypically milder than RTS1;
         CREBBP vs EP300 requires gene sequencing; broad thumbs PATHOGNOMONIC;
         seed SEED_BASE+5) .
BRCA2   (Breast cancer type 2 susceptibility protein; 3418aa; 13q12.3; AD LOF / biallelic AR FA-D1;
         Fanconi anaemia complementation group D1 (biallelic);
         desmoplastic MB PATHOGNOMONIC FA-D1 (alongside HBL/AML/Wilms/RMS);
         AVOID alkylating agents; SIBLING DONOR EXCLUSION MANDATORY;
         seed SEED_BASE+6) .
PALB2   (Partner and localiser of BRCA2; 1186aa; 16p12.2; AD LOF / biallelic AR FA-N;
         Fanconi anaemia complementation group N (biallelic); HBOC2 (monoallelic);
         Brain tumours including MB in FA-N biallelic Fanconi;
         AVOID alkylating agents; AVOID radiation;
         DEB/MMC chromosomal fragility test;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3302-3309)
"""
import random

SEED_BASE = 3302

ATLAS_GENES = [
    {
        "gene": "PTCH1",
        "protein": (
            "PTCH1 -- 9q22.32 Autosomal-Dominant-LOF -- 1447aa -- "
            "Patched1-161kDa-SHH-Receptor-Gorlin-NBCCS-SHH-MB-25-30pct-Germline-"
            "BCC-Desmoplastic-Nodular-AVOID-RADIATION-ABSOLUTELY-OMIM-601309"
        ),
        "locus": "9q22.32",
        "protein_size": (
            "1447 aa / 161 kDa / 9q22.32 PTCH1 encodes Patched-1, the SHH receptor: "
            "STRUCTURE: "
            "  1447 aa / 161 kDa; 12-pass transmembrane receptor; SHH signalling pathway; "
            "  Two large extracellular loops (ECL1 aa 50-450; ECL2 aa 850-1200): SHH binding; "
            "  Sterol-sensing domain (SSD; aa 420-700): 5-TM sterol-sensing motif; "
            "    Binds oxysterols that regulate SMO activity indirectly; "
            "  C-terminal cytoplasmic tail (aa 1200-1447): ubiquitination + endocytosis; "
            "  PTCH1 function: represses Smoothened (SMO) in absence of SHH; "
            "    SHH binds PTCH1 → PTCH1 inhibition relieved → SMO active → GLI TF activation; "
            "    LOF PTCH1: constitutive SMO/GLI activation → SHH-driven tumour; "
            "GORLIN SYNDROME (PTCH1): "
            "  OMIM 109400; AD; 30-50% de novo; classic Gorlin triad: "
            "    (1) Multiple basal cell carcinomas (BCCs): onset teens-20s; "
            "    (2) Jaw keratocysts (odontogenic): PATHOGNOMONIC Gorlin, onset childhood; "
            "    (3) Calcified falx cerebri: PATHOGNOMONIC Gorlin, age >20yr; "
            "  Additional PATHOGNOMONIC features: "
            "    Bifid rib / fused rib / extra rib: 40-60% Gorlin; "
            "    Macrocephaly + frontal bossing + coarse facial features; "
            "    Lamellar calcification of the falx: earliest calcification sign in childhood; "
            "  Basal cell carcinoma risk: 90% affected by age 40yr in Caucasians; "
            "    UV-exposed areas + non-UV areas (trunk, scalp, face); "
            "    Number of BCCs: tens to thousands in Gorlin; "
            "    RADIATION absolutely contraindicated: each Gray induces ~10 new BCCs; "
            "SHH-MEDULLOBLASTOMA IN PTCH1: "
            "  Medulloblastoma: 5% of Gorlin patients develop MB (CBTRUS 2019); "
            "  Age at MB: mainly <5yr (peak 1-3yr); "
            "  Histology: desmoplastic/nodular MB PATHOGNOMONIC Gorlin/PTCH1 germline; "
            "  SHH-MB subgroup: 25-30% of SHH-MB carry germline PTCH1 mutations; "
            "  AVOID RADIATION: radiation triggers hundreds of BCCs; proton beam preferred if unavoidable; "
            "  Vismodegib (hedgehog inhibitor): adult BCC; AVOID in children/adolescents; "
            "SURVEILLANCE (PTCH1/GORLIN): "
            "  Annual brain MRI from birth to age 5yr (MB surveillance); "
            "  Jaw OPG annually from age 5yr to 21yr (keratocysts); "
            "  Full skin dermatology annually from puberty; "
            "  Echocardiogram at birth (cardiac fibroma 2%); "
            "  Spine/rib X-ray: baseline at diagnosis; "
        ),
        "inheritance": "Autosomal dominant LOF; 30-50% de novo; near-complete penetrance (>95%) for jaw keratocysts and calcified falx; BCC penetrance 90% by age 40yr in Caucasians; MB penetrance ~5% lifetime; some population/ethnicity effects on BCC load (low in Asian Gorlin)",
        "cancer_risk": "BCC 90% Caucasian by age 40yr; medulloblastoma (SHH) 5% lifetime (HIGHEST in infancy-early childhood); jaw keratocysts 60-90%; ovarian fibroma 17%; cardiac fibroma 2%; meningioma rare; rhabdomyosarcoma rare",
        "pathognomonic": "Desmoplastic/nodular MB in child <5yr PATHOGNOMONIC Gorlin; calcified falx cerebri in child PATHOGNOMONIC Gorlin; jaw keratocyst in child PATHOGNOMONIC Gorlin; multiple BCCs before age 20yr PATHOGNOMONIC Gorlin; bifid rib PATHOGNOMONIC Gorlin",
        "surveillance_key": "Brain MRI annually birth to 5yr (MB); jaw OPG annually 5-21yr (keratocysts); skin dermatology annually puberty onwards; AVOID RADIATION ABSOLUTELY; echocardiogram at birth (cardiac fibroma); vismodegib AVOID in children; cascade 50% first-degree relatives",
        "key_distinctions": [
            "SHH-MB-25-30PCT-GERMLINE-PTCH1-IN-SHH-SUBGROUP",
            "DESMOPLASTIC-NODULAR-MB-PATHOGNOMONIC-GORLIN",
            "AVOID-RADIATION-ABSOLUTELY-10-BCC-PER-GRAY",
            "JAW-KERATOCYST-PATHOGNOMONIC-GORLIN",
            "CALCIFIED-FALX-CEREBRI-PATHOGNOMONIC-GORLIN",
            "VISMODEGIB-AVOID-CHILDREN-GROWTH-PLATE",
        ],
    },
    {
        "gene": "SUFU",
        "protein": (
            "SUFU -- 10q24.32 Autosomal-Dominant-LOF -- 484aa -- "
            "SUFU-55kDa-SHH-Pathway-Distal-GLI-Regulator-Gorlin-like-SHH-MB-HIGHEST-"
            "50-60pct-Lifetime-AVOID-Radiation-AVOID-Vismodegib-OMIM-607035"
        ),
        "locus": "10q24.32",
        "protein_size": (
            "484 aa / 55 kDa / 10q24.32 SUFU encodes Suppressor of Fused homologue: "
            "STRUCTURE: "
            "  484 aa / 55 kDa; cytoplasmic/nuclear scaffold protein; no transmembrane domain; "
            "  N-terminal domain (aa 1-270): interacts with GLI1/GLI2/GLI3 zinc finger regions; "
            "  C-terminal domain (aa 270-484): dimerisation + microtubule association; "
            "  SUFU function: negative regulator of GLI transcription factors; "
            "    Sequesters GLI proteins in cytoplasm → prevents target gene activation; "
            "    In active SHH signalling: SUFU dissociates from GLI2/GLI3 → nuclear translocation; "
            "    LOF SUFU: constitutive GLI nuclear activity → SHH-driven proliferation; "
            "  SUFU vs PTCH1: SUFU is DISTAL to PTCH1/SMO in the SHH cascade; "
            "    SUFU-LOF cannot be rescued by vismodegib (SMO inhibitor): "
            "      Vismodegib acts upstream of SUFU; SUFU-mutant tumours are vismodegib-RESISTANT; "
            "SUFU-ASSOCIATED SHH-MB: "
            "  Medulloblastoma: SUFU germline = HIGHEST lifetime MB risk (~50-60%); "
            "    Higher than PTCH1 (5%) because SUFU is distal to PTCH1 — SUFU-LOF causes more constitutive GLI activation; "
            "  Age at MB: predominantly <2yr (infantile desmoplastic/nodular); "
            "  Histology: desmoplastic/nodular (extensive) MB PATHOGNOMONIC; "
            "  SHH-MB germline: SUFU + PTCH1 together account for ~40% of all germline SHH-MB; "
            "  Key clinical difference from PTCH1-Gorlin: "
            "    SUFU: lower BCC frequency than PTCH1 (some SUFU families have few/no BCCs); "
            "    SUFU: jaw keratocysts and calcified falx less common than PTCH1; "
            "    SUFU: HIGHER MB penetrance (~50-60% vs ~5% PTCH1); "
            "  Brain MRI: 3-monthly in first 5yr of life; "
            "TREATMENT SUFU-MB: "
            "  Chemotherapy-only approach (<3yr): avoid radiation in Gorlin-like patients; "
            "  Carboplatin + vincristine: standard infant MB backbone; "
            "  AVOID vismodegib (SMO inhibitor): SUFU-LOF is downstream of SMO → vismodegib ineffective; "
            "  AVOID radiation: secondary malignancy risk (BCC field); "
            "  GLI inhibitors (GANT61, ATO/arsenic trioxide): downstream GLI inhibition; "
        ),
        "inheritance": "Autosomal dominant LOF; 25-40% de novo; much higher MB penetrance than PTCH1; BCC penetrance variable (lower than Gorlin PTCH1 in some families); jaw keratocysts and calcified falx less common than PTCH1; MB age-dependent penetrance",
        "cancer_risk": "Medulloblastoma (SHH desmoplastic) HIGHEST ~50-60% lifetime; BCC lower frequency than PTCH1 (variable 10-30%); jaw keratocysts less common; meningioma rare; ovarian fibroma rare",
        "pathognomonic": "Desmoplastic/nodular (extensive) MB in infant <2yr PATHOGNOMONIC SUFU; SHH-MB in child without classic Gorlin features = SUFU sequencing MANDATORY; vismodegib-resistant SHH-MB = SUFU mutation likely; SUFU-LOF medulloblastoma = highest germline MB risk of all SHH-pathway genes",
        "surveillance_key": "Brain MRI every 3 months from birth to age 2yr then 6-monthly to age 5yr; AVOID vismodegib (resistant - SUFU distal to SMO); AVOID radiation children; carboplatin-vincristine infant backbone; GLI inhibitors investigational; cascade 50% risk",
        "key_distinctions": [
            "SHH-MB-HIGHEST-50-60PCT-LIFETIME-SUFU",
            "VISMODEGIB-RESISTANT-SUFU-DISTAL-TO-SMO",
            "DESMOPLASTIC-NODULAR-EXTENSIVE-MB-PATHOGNOMONIC",
            "MB-UNDER-2YR-SUFU-TYPICAL-AGE",
            "LOWER-BCC-THAN-PTCH1-BUT-HIGHER-MB-PENETRANCE",
            "GLI-INHIBITORS-DOWNSTREAM-THERAPEUTIC-TARGET",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-SHH-MB-Anaplastic-PATHOGNOMONIC-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; sequence-specific transcription factor and guardian of the genome; "
            "  N-terminal transactivation domain (aa 1-67): MDM2 binding; "
            "  Proline-rich domain (aa 68-98): apoptosis/cell cycle arrest switch; "
            "  DNA-binding domain (aa 102-292): most common mutation hotspot (R175/G245/R248/R249/R273/R282); "
            "  Tetramerisation domain (aa 323-356): p53 tetrameric active form; "
            "  C-terminal regulatory domain (aa 363-393): allosteric regulation; "
            "  TP53 LOF: loss of G1/S checkpoint, apoptosis; gain-of-function (GOF) mutations drive oncogenesis; "
            "TP53 / LI-FRAUMENI SYNDROME: "
            "  OMIM 151623; AD; 7-20% de novo; penetrance 50% by age 40yr women; 41% men; "
            "  LFS tumour spectrum: sarcomas (osteosarcoma, STS), breast cancer (premenopausal), "
            "    brain tumours (SHH-MB, DIPG, choroid plexus carcinoma), ACC, leukaemia; "
            "  SHH-MB in LFS: 4-6% lifetime; PATHOGNOMONIC in children when MB histology = anaplastic; "
            "  TP53 + SHH-MB: anaplastic histology + chromothripsis = PATHOGNOMONIC LFS; "
            "    WNT-MB in LFS: less common but also described; "
            "  Choroid plexus carcinoma (CPC): ~50% of CPCs in children carry germline TP53; "
            "  R337H founder mutation: Brazilian population (1:300 Southern Brazil); "
            "RADIATION IN LFS/TP53: "
            "  AVOID RADIATION ABSOLUTELY: "
            "    Germline TP53 → radiation hypersensitivity → radiation-induced second primaries; "
            "    Historical: children with LFS treated with radiotherapy → second sarcomas in field; "
            "  MRI-based surveillance: WBMRI (whole-body MRI) Toronto protocol annually; "
            "    Components: brain MRI + chest MRI + abdominal MRI + liver US + breast MRI (females ≥18yr); "
            "TREATMENT TP53/LFS-MB: "
            "  Chemotherapy-only backbone; AVOID radiation absolutely; "
            "  Carboplatin/cisplatin + vincristine + cyclophosphamide (standard MB protocols); "
            "  ONC201 (dordaviprone): H3K27M+ diffuse midline glioma FDA-approved 2022; "
            "    Relevant for DIPG in LFS (not specifically for MB but same patient population); "
            "  APR-246 (eprenetapopt/magrolimab): TP53-targeting; investigational; "
        ),
        "inheritance": "Autosomal dominant LOF; 7-20% de novo; near-complete penetrance by age 60yr (70-80%); females penetrance higher (breast cancer) and earlier onset; R337H Brazilian founder variant (adrenocortical carcinoma HIGHEST in this founder); genotype-phenotype correlations emerging (dominant-negative GOF mutations more severe)",
        "cancer_risk": "Sarcoma (STS, osteosarcoma) 25-30% lifetime; breast cancer 25-30% (premenopausal women); brain tumours SHH-MB/DIPG/CPC/LGG 10-15%; ACC 10-15% (R337H 70%); leukaemia/lymphoma 5-10%; colorectal 5-10%; adrenal 5-10% (infant sentinel tumour)",
        "pathognomonic": "SHH-MB + anaplastic histology + chromothripsis = TP53 germline PATHOGNOMONIC (SHH-MB/TP53); choroid plexus carcinoma in child PATHOGNOMONIC LFS testing; ACC in infant PATHOGNOMONIC LFS sentinel tumour; osteosarcoma + second primary = PATHOGNOMONIC LFS pattern",
        "surveillance_key": "WBMRI Toronto annually (brain+chest+abdomen+liver+breast); rapid brain MRI 6-monthly first 5yr; AVOID RADIATION ABSOLUTELY; cascade all first-degree 50% risk; ONC201 H3K27M+ DIPG; APR-246 investigational; breast MRI from age 20yr or 10yr pre-index diagnosis",
        "key_distinctions": [
            "SHH-MB-ANAPLASTIC-CHROMOTHRIPSIS-PATHOGNOMONIC-LFS",
            "AVOID-RADIATION-ABSOLUTELY-TP53-HYPERSENSITIVITY",
            "WBMRI-TORONTO-ANNUALLY-NOT-CT-PET",
            "R337H-BRAZILIAN-FOUNDER-1-IN-300-SOUTHERN-BRAZIL",
            "CHOROID-PLEXUS-CARCINOMA-50PCT-CARRY-GERMLINE-TP53",
            "ONC201-FDA2022-H3K27M-DIFFUSE-MIDLINE-GLIOMA",
        ],
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "APC-309kDa-WNT-BetaCatenin-Scaffold-FAP-Turcot-Type2-WNT-MB-Monosomy6-"
            "Excellent-Prognosis-CAPP5-Colonoscopy-10yr-OMIM-175100"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 309 kDa / 5q22.2 APC encodes adenomatous polyposis coli protein: "
            "STRUCTURE: "
            "  2843 aa / 309 kDa; scaffold protein; WNT/beta-catenin pathway regulator; "
            "  N-terminal dimerisation domain (aa 1-57); "
            "  Armadillo repeats (aa 163-736): protein-protein interaction; "
            "  SAMP repeats (aa 1020-2100): axin binding; "
            "  Mutation cluster region (MCR, aa 1286-1513): most colorectal adenoma mutations; "
            "  C-terminal EB1 binding (aa 2130-2843): microtubule interaction; "
            "  APC function: beta-catenin destruction complex (with Axin, GSK3B, CK1); "
            "    APC LOF → beta-catenin nuclear accumulation → WNT target gene overactivation; "
            "TURCOT SYNDROME TYPE 2 (APC-ASSOCIATED): "
            "  APC germline mutation + CNS tumours = Turcot Type 2 (MB-polyposis association); "
            "  MB type: WNT-medulloblastoma; "
            "    WNT-MB constitutive beta-catenin activation; monosomy 6 PATHOGNOMONIC WNT-MB; "
            "    WNT-MB: BEST prognosis of all MB subgroups (5yr survival >90%); "
            "    Beta-catenin IHC nuclear staining PATHOGNOMONIC WNT-MB in tumour; "
            "  MB risk in FAP/APC: 0.5-1% lifetime; peak age 5-15yr; "
            "  Absolute MB risk lower than PTCH1 or SUFU but WNT-MB excellent response; "
            "  Other CNS tumours in FAP: rarely ependymoma, astrocytoma; "
            "COLORECTAL CANCER IN APC/FAP: "
            "  Colorectal adenomas: by teens in FAP; cancer 100% untreated by age 40yr; "
            "  CAPP5 trial: colonoscopy + aspirin surveillance in FAP; "
            "  Prophylactic colectomy: timing depends on adenoma burden; "
            "  Hepatoblastoma (separate portal): AFP + liver US q3-6m birth-7yr for FAP families; "
            "APC/FAP DESMOID: "
            "  3' APC mutations: desmoid highest risk; "
            "  Nirogacestat gamma-secretase inhibitor FDA2023 for progressive desmoid; "
        ),
        "inheritance": "Autosomal dominant LOF; 20-30% de novo; near-complete penetrance for polyposis by age 40yr; WNT-MB penetrance 0.5-1% lifetime; genotype-phenotype correlation: 5' mutations hepatoblastoma/desmoid risk, 3' mutations desmoid highest, MCR (1286-1513) dense polyposis",
        "cancer_risk": "Colorectal cancer 100% untreated by age 40yr; WNT-medulloblastoma 0.5-1%; hepatoblastoma 0.5-2% (infancy); desmoid 10-20% (Gardner); gastric/periampullary/thyroid/brain rare; duodenal/periampullary polyps 90% by age 70yr",
        "pathognomonic": "WNT-MB in FAP patient PATHOGNOMONIC Turcot Type 2; nuclear beta-catenin IHC in MB PATHOGNOMONIC WNT-MB; monosomy 6 PATHOGNOMONIC WNT-MB excellent prognosis; colorectal adenomas in teen + MB = APC germline mandatory; WNT-MB excellent prognosis (>90% 5yr)",
        "surveillance_key": "Brain MRI annually from age 5yr; colonoscopy from age 10-12yr; AFP + liver US q3-6m birth-7yr (hepatoblastoma FAP); CAPP5 aspirin surveillance; prophylactic colectomy when adenoma burden high; nirogacestat for progressive desmoid; cascade 50% risk",
        "key_distinctions": [
            "WNT-MB-MONOSOMY6-PATHOGNOMONIC-EXCELLENT-PROGNOSIS",
            "BETA-CATENIN-NUCLEAR-IHC-PATHOGNOMONIC-WNT-MB",
            "FAP-TURCOT-TYPE2-APC-NOT-MMR",
            "COLONOSCOPY-FROM-AGE-10-12YR",
            "WNT-MB-BEST-PROGNOSIS-5YR-SURVIVAL-GT90PCT",
            "NIROGACESTAT-FDA2023-PROGRESSIVE-DESMOID",
        ],
    },
    {
        "gene": "CREBBP",
        "protein": (
            "CREBBP -- 16p13.3 Autosomal-Dominant-LOF -- 2441aa -- "
            "CBP-265kDa-HAT-KAT3A-WNT-MB-RTS1-BroadThumbs-BigToes-PATHOGNOMONIC-"
            "IntellectualDisability-100pct-CDDP-Sensitive-OMIM-600140"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "2441 aa / 265 kDa / 16p13.3 CREBBP encodes CREB-binding protein (CBP/KAT3A): "
            "STRUCTURE: "
            "  2441 aa / 265 kDa; transcriptional co-activator and histone acetyltransferase (HAT); "
            "  RID (receptor interaction domain, aa 1-111): nuclear receptor binding; "
            "  KIX domain (aa 586-672): CREB and MYB transcription factor binding; "
            "  PHD finger (aa 1235-1288): chromatin binding; "
            "  Bromodomain (aa 1087-1197): acetyl-histone binding; "
            "  HAT domain (aa 1288-1758): histone acetyltransferase catalytic core; "
            "    Acetylates H3K18, H3K27, H3K56; "
            "  TAZ2/CH3 domain (aa 1764-1849): p53 binding; "
            "  NCBD (aa 2057-2117): MYB/STAT interaction; "
            "  CREBBP function: bridge between transcription factors (CREB, p53, MYB) and basal machinery; "
            "    Histone acetylation → chromatin opening → transcriptional activation; "
            "RUBINSTEIN-TAYBI SYNDROME TYPE 1 (CREBBP): "
            "  OMIM 180849; AD; majority de novo; "
            "  PATHOGNOMONIC features: "
            "    Broad/spatulate thumbs and big toes: PATHOGNOMONIC (90-95% of RTS); "
            "    Short stature (height <-2 SD); "
            "    Intellectual disability: 100% (mild-moderate), IQ 35-75; "
            "    Characteristic facies: arched eyebrows + long eyelashes + beaked nose + grimacing smile; "
            "  Medulloblastoma (WNT) in RTS: "
            "    WNT-MB: CREBBP role in WNT pathway regulation; "
            "    CREBBP acts as co-activator for beta-catenin/TCF complex in WNT-MB; "
            "    RTS1 medulloblastoma: small number of cases reported (RTS cancer risk ~3% overall); "
            "    WNT-MB in RTS: excellent prognosis subgroup; nuclear beta-catenin IHC; "
            "  Other cancers in RTS1: leukaemia, lymphoma, retinoblastoma (rare); "
            "TREATMENT CREBBP/RTS-MB: "
            "  WNT-MB standard: cisplatin + vincristine + cyclophosphamide; "
            "  RTS intellectual disability: multidisciplinary educational support; "
            "  Vorinostat (HDAC inhibitor): investigational in RTS1 cognition trials; "
            "    Rationale: restore histone acetylation downstream of CREBBP deficiency; "
        ),
        "inheritance": "Autosomal dominant LOF; majority de novo; haploinsufficiency mechanism; CREBBP and EP300 together account for >90% of genetically confirmed RTS; CREBBP more common (60-70% of RTS) vs EP300 (30-40%); CREBBP more severe phenotype",
        "cancer_risk": "Overall cancer risk ~3% (haematological malignancy, lymphoma, MB, retinoblastoma rare); WNT-MB excellent prognosis subgroup; CREBBP somatic mutations common in lymphomas (diffuse large B-cell lymphoma follicular lymphoma); germline CREBBP = constitutional risk via haploinsufficiency",
        "pathognomonic": "Broad spatulate thumbs + big toes PATHOGNOMONIC RTS; WNT-MB in RTS1 patient; intellectual disability + characteristic facies + broad thumbs + short stature = CREBBP/EP300 sequencing MANDATORY; vorinostat HDAC inhibition investigational for RTS cognition",
        "surveillance_key": "Annual neuro-oncology review; WNT-MB standard cisplatin-vincristine-cyclophosphamide; baseline brain MRI at diagnosis; haematological malignancy surveillance; vorinostat investigational cognition trials; multidisciplinary educational support; cascade 50% if familial (most de novo)",
        "key_distinctions": [
            "BROAD-THUMBS-BIG-TOES-PATHOGNOMONIC-RTS1",
            "WNT-MB-EXCELLENT-PROGNOSIS-CREBBP",
            "INTELLECTUAL-DISABILITY-100PCT-RTS",
            "VORINOSTAT-HDAC-INVESTIGATIONAL-RTS-COGNITION",
            "CREBBP-MORE-SEVERE-THAN-EP300",
            "CREBBP-SOMATIC-COMMON-IN-DLBCL-FOLLICULAR-LYMPHOMA",
        ],
    },
    {
        "gene": "EP300",
        "protein": (
            "EP300 -- 22q13.2 Autosomal-Dominant-LOF -- 2414aa -- "
            "p300-264kDa-HAT-KAT3B-WNT-MB-RTS2-BroadThumbs-Phenotypically-Milder-"
            "CREBBP-EP300-Sequencing-Required-OMIM-602700"
        ),
        "locus": "22q13.2",
        "protein_size": (
            "2414 aa / 264 kDa / 22q13.2 EP300 encodes E1A-binding protein p300 (KAT3B): "
            "STRUCTURE: "
            "  2414 aa / 264 kDa; paralogue of CREBBP (63% homology in functional domains); "
            "  RID (aa 1-111): nuclear receptor binding; "
            "  KIX domain (aa 565-662): CREB and MYB binding; "
            "  PHD finger (aa 1195-1256): chromatin binding; "
            "  Bromodomain (aa 1057-1165): acetyl-histone binding; "
            "  HAT domain (aa 1256-1720): histone acetyltransferase; "
            "    Same H3K18/H3K27/H3K56 acetylation substrates as CREBBP; "
            "  TAZ2/CH3 domain (aa 1727-1812): p53 binding; "
            "  E1A-binding region (aa 1-2414): viral protein interaction; "
            "  EP300 vs CREBBP: identical functional domains; different somatic mutation profiles; "
            "    CREBBP: more frequent somatic mutations in B-cell lymphomas; "
            "    EP300: more frequent somatic mutations in solid tumours (colorectal, bladder); "
            "RUBINSTEIN-TAYBI SYNDROME TYPE 2 (EP300): "
            "  OMIM 613684; AD; majority de novo; "
            "  PHENOTYPICALLY MILDER than RTS1 (CREBBP): "
            "    Intellectual disability: milder (IQ often 50-80 range vs 35-75 CREBBP); "
            "    Broad thumbs: present but less pronounced than CREBBP; "
            "    Facies: similar but less distinctive; "
            "  Medulloblastoma (WNT) in RTS2: "
            "    WNT-MB association: same mechanism as CREBBP (co-activator in WNT pathway); "
            "    Fewer cases reported than CREBBP-associated MB; "
            "    Same excellent prognosis WNT-MB subgroup expected; "
            "  Critical clinical point: CREBBP vs EP300 CANNOT be distinguished on phenotype alone; "
            "    Gene sequencing of BOTH required in RTS-like phenotype; "
            "    IQ ≥ 50 + milder features: more likely EP300; "
            "    IQ < 50 + classic facies + broad thumbs: more likely CREBBP; "
            "EP300 SOMATIC IN CANCER: "
            "  Common somatic mutation: colorectal cancer, bladder, head-neck, lung cancers; "
            "  EP300 amplification: rare variant in solid tumours; "
        ),
        "inheritance": "Autosomal dominant LOF; majority de novo; haploinsufficiency; EP300 accounts for 30-40% of RTS vs 60-70% for CREBBP; milder phenotype than CREBBP; CREBBP and EP300 share near-identical functional domains → same pathway consequence of LOF",
        "cancer_risk": "Cancer risk modestly elevated (similar to CREBBP but less well characterised); WNT-MB same mechanism as CREBBP; haematological malignancy described; EP300 somatic: colorectal, bladder, head-neck cancers (somatic ≠ germline risk)",
        "pathognomonic": "Broad thumbs in child with intellectual disability + RTS phenotype → CREBBP and EP300 sequencing MANDATORY (cannot distinguish on clinical grounds alone); WNT-MB in RTS2 patient; milder phenotype than RTS1 = EP300 more likely but not definitive",
        "surveillance_key": "WNT-MB standard cisplatin-vincristine-cyclophosphamide; annual brain MRI at diagnosis; multidisciplinary educational support; genetic counselling (de novo in most, empirical 1% recurrence); IQ monitoring; CREBBP vs EP300 gene sequencing required for definitive diagnosis; vorinostat HDAC investigational",
        "key_distinctions": [
            "RTS2-MILDER-THAN-RTS1-CREBBP",
            "CREBBP-EP300-CANNOT-DISTINGUISH-CLINICALLY",
            "BOTH-CREBBP-AND-EP300-MUST-BE-SEQUENCED-IN-RTS",
            "WNT-MB-EXCELLENT-PROGNOSIS-EP300",
            "EP300-SOMATIC-COLORECTAL-BLADDER-CANCER",
            "INTELLECTUAL-DISABILITY-MILDER-IQ-50-80-EP300",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF-Monoallelic-AD-Biallelic-AR-FA-D1 -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FA-D1-Desmoplastic-MB-PATHOGNOMONIC-"
            "AVOID-Alkylating-Sibling-Exclusion-MANDATORY-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes breast cancer type 2 susceptibility protein: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; large nuclear scaffold protein; homologous recombination (HR) repair; "
            "  N-terminal PALB2-binding domain (aa 10-40): PALB2 tethers BRCA2 to nuclear scaffold; "
            "  BRCA2-specific repeats BRC1-8 (aa 1002-2667): RAD51 binding x8; "
            "    Each BRC repeat recruits one RAD51 monomer; "
            "  DBD / OB-fold domain (aa 2402-3190): ssDNA binding; "
            "    OB1 (aa 2402-2667): PALB2 C-terminal binding; "
            "    OB2/OB3 (aa 2667-3026): single-stranded DNA binding; "
            "    Tower domain (aa 2856-2961): DNA binding; "
            "  C-terminal DSS1/RAD51 binding (aa 3196-3418): DSS1 chaperone; "
            "  BRCA2 function: loads RAD51 onto RPA-coated ssDNA at DSB ends → HR repair; "
            "BIALLELIC BRCA2 (FANCONI ANAEMIA FA-D1): "
            "  Biallelic BRCA2 mutations → FA-D1 Fanconi anaemia; "
            "  MOST SEVERE Fanconi complementation group; "
            "  FA-D1 tumour spectrum (childhood onset): "
            "    Medulloblastoma (desmoplastic): PATHOGNOMONIC FA-D1; "
            "    Hepatoblastoma: PATHOGNOMONIC FA-D1; "
            "    Wilms tumour: PATHOGNOMONIC FA-D1; "
            "    AML (myeloid): PATHOGNOMONIC FA-D1; "
            "    Rhabdomyosarcoma: PATHOGNOMONIC FA-D1; "
            "  Chromosomal fragility test: DEB (diepoxybutane) + MMC (mitomycin C) PATHOGNOMONIC FA; "
            "  MB age in FA-D1: predominantly <3yr; "
            "TREATMENT FA-D1/BRCA2: "
            "  AVOID alkylating agents (cyclophosphamide, melphalan): lethal FA toxicity; "
            "  AVOID stem cell transplant from biologically related siblings: "
            "    Sibling donor exclusion: 25% chance sibling is also FA-D1 biallelic; "
            "    FA complementation testing of all potential sibling donors MANDATORY; "
            "  Carboplatin preferred over cisplatin (less nephrotoxicity in FA); "
            "  Doxorubicin: dose-reduced in FA (cardiac + DNA damage sensitivity); "
            "  Proton beam: preferred if radiation unavoidable (reduce scatter dose); "
        ),
        "inheritance": "Monoallelic BRCA2: AD LOF; HBOC (breast/ovarian); biallelic BRCA2: AR FA-D1 (Fanconi anaemia group D1); 25% sibling risk for FA if both parents carriers; FA-D1 MB predominantly de novo compound heterozygous",
        "cancer_risk": "Monoallelic: breast (70-80% by age 80yr), ovarian (15-25%), pancreatic (5-7%), prostate; biallelic FA-D1: MB PATHOGNOMONIC + hepatoblastoma + Wilms + AML + RMS + ALL; FA-D1 cancer risk essentially 100% by age 10yr without transplant",
        "pathognomonic": "Desmoplastic MB <3yr = FA-D1 BRCA2 testing MANDATORY; DEB/MMC chromosomal fragility PATHOGNOMONIC biallelic FA-D1; multiple childhood solid tumours (MB+HBL or MB+Wilms) = FA-D1 PATHOGNOMONIC pattern; SIBLING DONOR EXCLUSION MANDATORY for any haematopoietic transplant in FA-D1",
        "surveillance_key": "DEB/MMC chromosomal fragility test FIRST before chemotherapy; AVOID alkylating agents absolutely; sibling donor exclusion mandatory (FA complementation testing); carboplatin preferred; doxorubicin dose-reduced; proton beam preferred; brain MRI 3-monthly first 3yr in known FA-D1",
        "key_distinctions": [
            "DESMOPLASTIC-MB-PATHOGNOMONIC-FA-D1-BRCA2",
            "AVOID-ALKYLATING-AGENTS-ABSOLUTELY-FA-D1",
            "SIBLING-DONOR-EXCLUSION-MANDATORY-FA-TESTING",
            "DEB-MMC-CHROMOSOMAL-FRAGILITY-PATHOGNOMONIC-FA",
            "CARBOPLATIN-PREFERRED-OVER-CISPLATIN-FA",
            "FA-D1-MULTI-TUMOUR-MB-HBL-WILMS-AML-RMS-PATHOGNOMONIC",
        ],
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF-Monoallelic-Biallelic-AR-FA-N -- 1186aa -- "
            "PALB2-131kDa-BRCA2-Anchor-FA-N-Brain-Tumours-MB-AVOID-Alkylating-"
            "DEB-MMC-PATHOGNOMONIC-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 encodes partner and localiser of BRCA2: "
            "STRUCTURE: "
            "  1186 aa / 131 kDa; nuclear scaffold adapter protein; "
            "  N-terminal coiled-coil domain (aa 1-100): BRCA1 C-terminus binding; "
            "    BRCA1-PALB2-BRCA2 ternary complex formation at DSBs; "
            "  WD40 repeat domain (aa 853-1186): BRCA2 N-terminus binding; "
            "    WD40 repeats form beta-propeller → chromatin association; "
            "    WD40 domain: RAD51C interaction; "
            "  Coiled-coil middle domain (aa 395-445): PALB2-PALB2 oligomerisation; "
            "  PALB2 function: nuclear scaffold anchor for BRCA2; "
            "    Without PALB2: BRCA2 cannot localise to nuclear chromatin; "
            "    PALB2 LOF → BRCA2 mislocalised → HR deficient → DNA damage accumulation; "
            "BIALLELIC PALB2 (FANCONI ANAEMIA FA-N): "
            "  FA-N: rarest Fanconi complementation group; "
            "  FA-N tumour spectrum (severe): "
            "    Brain tumours including medulloblastoma; "
            "    Wilms tumour; "
            "    Haematological malignancy (AML, ALL); "
            "    Breast cancer (young onset, FA-N carrier mothers); "
            "  Key difference from FA-D1 (BRCA2): "
            "    FA-N phenotypically similar to FA-D1 but somewhat distinct tumour spectrum; "
            "    Less hepatoblastoma in FA-N compared to FA-D1; "
            "    MB in FA-N: described in case series; desmoplastic histology; "
            "  DEB/MMC chromosomal fragility PATHOGNOMONIC for any FA subtype; "
            "  Pancytopenia + chromosomal fragility: classic FA presentation; "
            "MONOALLELIC PALB2 (HBOC2): "
            "  Monoallelic PALB2: 2nd most common HR gene after BRCA2; "
            "  Breast cancer (monoallelic): 53% lifetime risk (BOADICEA); "
            "  Pancreatic cancer: 2-3x elevated (monoallelic); "
            "  Ovarian cancer: 2-3x elevated (monoallelic); "
            "  Olaparib PARP inhibitor: FDA-approved for monoallelic PALB2 breast/ovarian; "
            "TREATMENT PALB2/FA-N MB: "
            "  Same principles as FA-D1 (BRCA2): "
            "  AVOID alkylating agents; carboplatin-based; sibling donor exclusion; "
            "  Olaparib: FDA-approved in PALB2 monoallelic breast/ovarian; "
            "  AVOID high-dose radiation in FA-N (radiosensitivity); "
        ),
        "inheritance": "Monoallelic PALB2: AD LOF; HBOC2 (2nd most common after BRCA2); biallelic PALB2: AR FA-N (Fanconi anaemia N); FA-N parents each monoallelic carrier (25% child risk for FA-N); FA-N rarest but most severe brain-tumour subset among Fanconi groups",
        "cancer_risk": "Monoallelic: breast 53% lifetime, pancreatic 2-3x, ovarian 2-3x; biallelic FA-N: brain tumours including MB + Wilms + AML + ALL; FA-N cancer risk 100% without treatment; second most severe FA subtype after FA-D1 (BRCA2)",
        "pathognomonic": "DEB/MMC chromosomal fragility PATHOGNOMONIC any FA including FA-N; MB + pancytopenia in infant = FA-N testing; brain tumours in young PALB2 monoallelic carrier family = FA-N compound heterozygous in index case; olaparib FDA for monoallelic PALB2 breast/ovarian",
        "surveillance_key": "DEB/MMC fragility test before chemotherapy; AVOID alkylating agents; sibling donor exclusion mandatory; carboplatin-based regimen; monoallelic: breast MRI from age 30yr; annual MRI pancreas from age 40yr (monoallelic PALB2); olaparib maintenance (monoallelic); brain MRI 3-monthly FA-N first 3yr",
        "key_distinctions": [
            "FA-N-RAREST-FA-SUBTYPE-BRAIN-TUMOURS-PALB2",
            "PALB2-ANCHOR-FOR-BRCA2-NUCLEAR-LOCALISATION",
            "AVOID-ALKYLATING-AGENTS-FA-N",
            "OLAPARIB-FDA-MONOALLELIC-PALB2-BREAST-OVARIAN",
            "BREAST-53PCT-LIFETIME-MONOALLELIC-PALB2",
            "DEB-MMC-PATHOGNOMONIC-ANY-FA-INCLUDING-FA-N",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Simulated tumour types per gene (medulloblastoma context)
# ─────────────────────────────────────────────────────────────────────────────
TUMOUR_TYPES_BY_GENE = {
    "PTCH1":  ["MB-SHH-desmoplastic-nodular", "MB-SHH-classic", "BCC-radiation-induced", "Jaw-keratocyst", "MB-SHH-anaplastic"],
    "SUFU":   ["MB-SHH-desmoplastic-nodular-extensive", "MB-SHH-desmoplastic-infant", "MB-SHH-classic", "BCC-SUFU", "MB-SHH-LCCA"],
    "TP53":   ["MB-SHH-anaplastic-TP53", "MB-SHH-chromothripsis", "MB-WNT-LFS", "CPC-LFS", "MB-SHH-classic-LFS"],
    "APC":    ["MB-WNT-monosomy6", "MB-WNT-APC", "Colorectal-adenoma-FAP", "MB-WNT-classic", "MB-WNT-beta-catenin-nuclear"],
    "CREBBP": ["MB-WNT-RTS1", "MB-WNT-CREBBP", "Lymphoma-RTS1", "MB-WNT-classic-CREBBP", "Retinoblastoma-RTS1"],
    "EP300":  ["MB-WNT-RTS2", "MB-WNT-EP300", "Lymphoma-RTS2", "MB-WNT-classic-EP300", "MB-WNT-beta-catenin"],
    "BRCA2":  ["MB-SHH-desmoplastic-FA-D1", "Hepatoblastoma-FA-D1", "Wilms-FA-D1", "AML-FA-D1", "RMS-embryonal-FA-D1"],
    "PALB2":  ["MB-FA-N-brain-tumour", "Wilms-FA-N", "AML-FA-N", "ALL-FA-N", "MB-SHH-FA-N"],
}

VARIANTS_BY_GENE = {
    "PTCH1":  ["p.Glu476Ter (nonsense)", "p.Arg852Ter (nonsense)", "del exon 1-3 (partial deletion)", "p.Ala823Val (missense SSD)", "p.Ser1436Ter (C-terminal truncation)"],
    "SUFU":   ["p.Arg418Ter (nonsense)", "p.Leu412Pro (helix-breaking)", "p.Asp268Asn (GLI-binding surface)", "del exon 4-6 (partial deletion)", "p.Gln444Ter (near-end truncation)"],
    "TP53":   ["p.Arg248Trp (hotspot GOF)", "p.Arg273His (hotspot GOF)", "p.Arg175His (hotspot dominant-negative)", "R337H (Brazilian founder)", "del 17p13.1 (gross deletion LFS)"],
    "APC":    ["p.Gln1338Ter (MCR)", "c.3183del (frameshift MCR)", "p.Arg1450Ter (MCR)", "del 5q22.2 (partial deletion)", "p.Trp1327Ter (5' MCR WNT-MB)"],
    "CREBBP": ["p.Arg1446Ter (HAT domain)", "p.Leu1268Pro (helix-breaking HAT)", "del 16p13.3 (partial deletion)", "p.Glu1449Lys (HAT surface)", "p.Trp1512Ter (TAZ2 truncation)"],
    "EP300":  ["p.Arg1445Ter (HAT domain)", "p.Leu1267Pro (helix-breaking HAT)", "del 22q13.2 (partial deletion)", "p.Glu1448Lys (HAT surface)", "p.Trp1510Ter (TAZ2 truncation)"],
    "BRCA2":  ["p.Leu1455Ter (BRC repeat)", "p.Trp2626Ter (DBD truncation)", "del exon 15-24 (partial deletion)", "p.Arg2830Ter (DBD truncation)", "p.Cys3080Arg (OB-fold missense)"],
    "PALB2":  ["p.Gln775Ter (WD40)", "p.Leu939Pro (helix-breaking WD40)", "del exon 4-7 (partial deletion)", "p.Arg414Ter (coiled-coil)", "p.Trp1038Ter (WD40 truncation)"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "PTCH1":  ["Chemotherapy-only (avoid radiation); carboplatin/vincristine infant backbone", "Vismodegib: adult BCC only; AVOID in children/adolescents", "AVOID radiotherapy; proton beam if unavoidable", "Brain MRI every 3-months first 5yr of life"],
    "SUFU":   ["Chemotherapy-only (infant MB); avoid radiation and vismodegib", "Carboplatin + vincristine (infant backbone)", "GLI inhibitors (GANT61/ATO): investigational", "Brain MRI every 3-months first 2yr"],
    "TP53":   ["Cisplatin/carboplatin + vincristine + cyclophosphamide; AVOID radiation", "ONC201/dordaviprone: H3K27M+ diffuse midline glioma (DIPG in LFS)", "APR-246 (eprenetapopt): investigational TP53 reactivation", "WBMRI Toronto annually: whole-body MRI surveillance"],
    "APC":    ["WNT-MB: cisplatin + vincristine + cyclophosphamide (excellent prognosis)", "Consider reduced-intensity regimen (WNT-MB best prognosis)", "Colonoscopy annually from 10yr; colectomy when polyposis advanced", "Nirogacestat FDA2023: progressive desmoid tumour"],
    "CREBBP": ["WNT-MB: cisplatin + vincristine + cyclophosphamide", "Multidisciplinary educational support; IQ monitoring", "Vorinostat HDAC inhibitor: investigational for cognition in RTS1", "Annual haematological review (lymphoma surveillance)"],
    "EP300":  ["WNT-MB: cisplatin + vincristine + cyclophosphamide", "Multidisciplinary educational support", "Gene sequencing: CREBBP AND EP300 required (cannot distinguish clinically)", "Annual haematological review"],
    "BRCA2":  ["AVOID alkylating agents ABSOLUTELY (cyclophosphamide lethal FA toxicity)", "Carboplatin preferred over cisplatin; doxorubicin dose-reduced FA", "DEB/MMC chromosomal fragility test FIRST before any chemotherapy", "Proton beam preferred if radiation unavoidable"],
    "PALB2":  ["AVOID alkylating agents; carboplatin-based chemotherapy", "DEB/MMC chromosomal fragility test before chemotherapy", "Olaparib: FDA-approved monoallelic PALB2 breast/ovarian", "Sibling donor exclusion mandatory (FA complementation testing)"],
}

SURVEILLANCE_BY_GENE = {
    "PTCH1": [
        "Brain MRI every 3 months: birth to age 5yr (MB surveillance)",
        "Annual jaw OPG X-ray from age 5yr to 21yr (keratocysts)",
        "Full skin dermatology annually from puberty (BCC surveillance)",
        "Echocardiogram at birth (cardiac fibroma 2%)",
        "AVOID radiation: refer to proton centre; CASCADE 50% risk",
    ],
    "SUFU": [
        "Brain MRI every 3 months: birth to age 2yr; then 6-monthly to age 5yr",
        "Annual dermatology review from puberty (BCC less common than PTCH1)",
        "AVOID vismodegib in children; AVOID radiation",
        "GLI inhibitor trials: enroll if SHH-MB confirmed SUFU germline",
        "CASCADE 50% offspring risk; preconception/prenatal testing option",
    ],
    "TP53": [
        "WBMRI Toronto annually: brain + chest + abdomen + pelvis",
        "Rapid brain MRI 6-monthly first 5yr; annual thereafter",
        "Breast MRI from age 20yr (females; or 10yr before index case)",
        "Adrenal/abdominal US q3-6m infancy (ACC sentinel LFS)",
        "AVOID CT/PET surveillance; AVOID radiation absolutely",
    ],
    "APC": [
        "Brain MRI annually from age 5yr to 15yr (WNT-MB surveillance)",
        "Colonoscopy annually from age 10-12yr (FAP)",
        "AFP + liver US q3-6m from birth to age 7yr (hepatoblastoma FAP)",
        "EGD from age 20yr or onset of GI symptoms",
        "CASCADE: 50% risk; CAPP5 aspirin regimen from age 10yr",
    ],
    "CREBBP": [
        "Annual neuro-oncology review post-MB treatment",
        "Haematological malignancy surveillance: full blood count annually",
        "IQ and developmental monitoring: annual SALT/psychology",
        "Baseline brain MRI at RTS diagnosis",
        "CASCADE: majority de novo; empirical recurrence risk ~1%; vorinostat investigational",
    ],
    "EP300": [
        "Annual neuro-oncology review post-MB treatment",
        "Haematological malignancy surveillance: full blood count annually",
        "IQ and developmental monitoring: annual SALT/psychology",
        "Gene sequencing of BOTH CREBBP and EP300 (phenotype alone insufficient)",
        "CASCADE: majority de novo; empirical recurrence risk ~1%",
    ],
    "BRCA2": [
        "DEB/MMC chromosomal fragility test: MANDATORY before any chemotherapy",
        "Brain MRI every 3 months: birth to age 3yr (FA-D1 MB surveillance)",
        "AFP + liver US q3-6m birth to 7yr (hepatoblastoma FA-D1 surveillance)",
        "Renal US annually birth to 8yr (Wilms tumour FA-D1 surveillance)",
        "SIBLING DONOR EXCLUSION: FA complementation testing of all potential sibling donors",
    ],
    "PALB2": [
        "DEB/MMC chromosomal fragility test: MANDATORY before any chemotherapy",
        "Brain MRI every 3 months: birth to age 3yr (FA-N brain tumour surveillance)",
        "Annual full blood count (haematological malignancy FA-N)",
        "Monoallelic carrier: breast MRI from age 30yr; annual MRI pancreas from age 40yr",
        "Sibling donor exclusion: FA complementation testing mandatory; olaparib maintenance monoallelic",
    ],
}


def _make_patient(gene_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_index]["gene"]
    tumour = rng.choice(TUMOUR_TYPES_BY_GENE[gene])
    variant = rng.choice(VARIANTS_BY_GENE[gene])

    # Age at diagnosis (medulloblastoma context)
    if gene in ("PTCH1",):
        age = rng.randint(0, 5)
    elif gene in ("SUFU", "BRCA2", "PALB2"):
        age = rng.randint(0, 3)
    elif gene in ("TP53",):
        age = rng.randint(0, 10)
    elif gene in ("APC",):
        age = rng.randint(5, 15)
    elif gene in ("CREBBP", "EP300"):
        age = rng.randint(2, 14)
    else:
        age = rng.randint(1, 10)

    # CR rates (medulloblastoma context - generally good for WNT, fair for SHH)
    cr_rates = {
        "PTCH1": 0.78, "SUFU": 0.75, "TP53": 0.58, "APC": 0.92,
        "CREBBP": 0.88, "EP300": 0.87, "BRCA2": 0.65, "PALB2": 0.62,
    }
    cr = rng.random() < cr_rates.get(gene, 0.75)

    # Radiation rates (low - mostly avoiding radiation in hereditary MB)
    rad_rates = {
        "PTCH1": 0.05, "SUFU": 0.04, "TP53": 0.00, "APC": 0.35,
        "CREBBP": 0.30, "EP300": 0.30, "BRCA2": 0.08, "PALB2": 0.06,
    }
    radiation = rng.random() < rad_rates.get(gene, 0.15)

    # Chemotherapy used (almost all MB patients get chemo)
    chemo_rates = {
        "PTCH1": 0.92, "SUFU": 0.95, "TP53": 0.98, "APC": 0.90,
        "CREBBP": 0.88, "EP300": 0.88, "BRCA2": 0.95, "PALB2": 0.94,
    }
    targeted = rng.random() < chemo_rates.get(gene, 0.92)  # 'targeted' = chemo administered

    # HSCT rates (used in some infant MB + FA protocols)
    hsct_rates = {
        "PTCH1": 0.08, "SUFU": 0.15, "TP53": 0.10, "APC": 0.05,
        "CREBBP": 0.06, "EP300": 0.06, "BRCA2": 0.30, "PALB2": 0.28,
    }
    transplant = rng.random() < hsct_rates.get(gene, 0.10)

    relapse = rng.random() < (0.15 if cr else 0.60)

    return {
        "gene": gene,
        "seed": seed,
        "age_dx": age,
        "tumour_type": tumour,
        "variant": variant,
        "cr": cr,
        "radiation": radiation,
        "targeted": targeted,      # chemotherapy administered
        "transplant": transplant,  # HSCT
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
        "atlas": "Hereditary-Medulloblastoma-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference: PTCH1 · SUFU · TP53 · APC · CREBBP · EP300 · BRCA2 · PALB2",
        "seeds": "3302-3309",
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
            "Hereditary Medulloblastoma Predisposition Atlas: "
            "SHH subgroup (PTCH1-Gorlin 25-30%, SUFU-Gorlin-like HIGHEST 50-60%, TP53-LFS anaplastic, BRCA2/PALB2-FA desmoplastic), "
            "WNT subgroup (APC-Turcot2 excellent prognosis monosomy6, CREBBP-RTS1, EP300-RTS2). "
            "Key rules: PTCH1/SUFU AVOID RADIATION ABSOLUTELY (BCC induction); TP53 AVOID RADIATION ABSOLUTELY (hypersensitivity); "
            "BRCA2/PALB2 AVOID ALKYLATING AGENTS (Fanconi toxicity); APC WNT-MB best prognosis (>90% 5yr); "
            "SUFU vismodegib-RESISTANT (distal to SMO). 320-patient aggregate 8×40, seeds 3302-3309."
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
    return {"atlas": "Hereditary-Medulloblastoma-Predisposition-Atlas", "breakdown": breakdown}


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
        "atlas": "Hereditary-Medulloblastoma-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "PTCH1_GORLIN_AVOID_RADIATION": (
                "PTCH1 Gorlin: AVOID RADIATION ABSOLUTELY. Each Gray induces ~10 new BCCs. "
                "SHH-MB 25-30% germline PTCH1 in SHH subgroup. Desmoplastic/nodular MB PATHOGNOMONIC. "
                "Brain MRI q3m birth-5yr. Jaw OPG annually 5-21yr. Calcified falx + jaw keratocysts PATHOGNOMONIC."
            ),
            "SUFU_HIGHEST_MB_VISMODEGIB_RESISTANT": (
                "SUFU: HIGHEST lifetime MB risk 50-60%. Desmoplastic/nodular (extensive) MB <2yr PATHOGNOMONIC. "
                "VISMODEGIB RESISTANT (SUFU is distal to SMO; SMO inhibitor cannot rescue SUFU-LOF). "
                "Brain MRI q3m first 2yr then q6m to 5yr. AVOID radiation children. GLI inhibitors investigational."
            ),
            "TP53_LFS_SHH_ANAPLASTIC_AVOID_RADIATION": (
                "TP53 LFS: SHH-MB + anaplastic histology + chromothripsis PATHOGNOMONIC LFS. "
                "AVOID RADIATION ABSOLUTELY. WBMRI Toronto annually (NOT CT/PET). "
                "ONC201/dordaviprone FDA2022 for H3K27M+ diffuse midline glioma. R337H Brazilian founder 1/300."
            ),
            "APC_TURCOT2_WNT_MB_EXCELLENT": (
                "APC Turcot Type 2: WNT-MB monosomy 6 PATHOGNOMONIC excellent prognosis (>90% 5yr). "
                "Nuclear beta-catenin IHC PATHOGNOMONIC WNT-MB. Colonoscopy from age 10-12yr. "
                "Reduced-intensity chemotherapy WNT-MB safe. AFP + liver US q3-6m birth-7yr (HBL FAP)."
            ),
            "CREBBP_EP300_RTS_WNT_MB": (
                "CREBBP (RTS1) and EP300 (RTS2): WNT-MB excellent prognosis; broad thumbs/big toes PATHOGNOMONIC RTS. "
                "CANNOT distinguish CREBBP vs EP300 on clinical grounds alone - sequence BOTH. "
                "CREBBP more severe (IQ 35-75) vs EP300 milder (IQ 50-80). Vorinostat HDAC investigational."
            ),
            "BRCA2_FA_D1_AVOID_ALKYLATING": (
                "BRCA2 FA-D1: desmoplastic MB PATHOGNOMONIC FA-D1 (with HBL/AML/Wilms/RMS). "
                "DEB/MMC chromosomal fragility PATHOGNOMONIC FA - perform FIRST before any chemotherapy. "
                "AVOID alkylating agents absolutely (cyclophosphamide lethal FA toxicity). "
                "SIBLING DONOR EXCLUSION MANDATORY - FA complementation testing of all potential sibling donors."
            ),
            "PALB2_FA_N_DEB_MMC": (
                "PALB2 FA-N: brain tumours including MB in biallelic FA-N. "
                "DEB/MMC chromosomal fragility PATHOGNOMONIC any FA including FA-N. "
                "AVOID alkylating agents; carboplatin-based. Sibling donor exclusion mandatory. "
                "Monoallelic PALB2: breast 53% lifetime; olaparib FDA-approved; MRI breast from age 30yr."
            ),
        },
        "cascade_testing_rule": (
            "Hereditary Medulloblastoma Predisposition Atlas — Cascade Testing: "
            "PTCH1: brain MRI q3m birth-5yr; jaw OPG annually 5-21yr; AVOID radiation absolutely; cascade 50%. "
            "SUFU: brain MRI q3m birth-2yr then q6m; AVOID vismodegib and radiation; cascade 50%. "
            "TP53 LFS: WBMRI annually; AVOID radiation; cascade 50% first-degree relatives. "
            "APC: colonoscopy from 10yr; AFP + liver US birth-7yr (HBL); brain MRI 5-15yr; cascade 50%. "
            "CREBBP: RTS1 - sequence BOTH CREBBP and EP300; multidisciplinary support; most de novo. "
            "EP300: RTS2 - same as CREBBP; milder phenotype; most de novo. "
            "BRCA2 FA-D1: DEB/MMC FIRST; AVOID alkylating; sibling donor exclusion; cascade 50%. "
            "PALB2 FA-N: DEB/MMC FIRST; AVOID alkylating; monoallelic carrier MRI breast from age 30yr."
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
    print(json.dumps(df["definitions"]["PTCH1"], indent=2)[:1500])
