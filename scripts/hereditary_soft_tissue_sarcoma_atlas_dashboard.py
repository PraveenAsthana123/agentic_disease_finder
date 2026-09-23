#!/usr/bin/env python3
"""Hereditary-Soft-Tissue-Sarcoma-Desmoid-Predisposition-Atlas -- Complete 8-Gene Reference
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         Soft tissue sarcoma 30-50% lifetime DOMINANT PATHOGNOMONIC;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annual;
         UPS/LMS/RMS/angiosarcoma; doxorubicin first-line;
         seed SEED_BASE+0) .
NF1     (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis type 1;
         MPNST 8-13% HIGHEST sarcoma risk PATHOGNOMONIC;
         plexiform neurofibroma -> MPNST malignant transformation;
         selumetinib MEKi FDA2020; AVOID radiation secondary MPNST;
         seed SEED_BASE+1) .
APC     (Adenomatous polyposis coli; 2843aa; 5q22.2; AD LOF;
         Gardner syndrome / FAP;
         desmoid fibromatosis 10-30% PATHOGNOMONIC mesenteric;
         surgery AVOID mesenteric desmoid (paradoxical growth);
         sulindac/celecoxib; sorafenib; imatinib off-label;
         seed SEED_BASE+2) .
DICER1  (DICER1 ribonuclease III; 1922aa; 14q32.13; AD LOF;
         DICER1 syndrome;
         pleuropulmonary blastoma PATHOGNOMONIC type-I/II/III;
         cervical embryonal RMS PATHOGNOMONIC;
         AVOID radiation children; multi-nodular thyroid goitre;
         seed SEED_BASE+3) .
SMARCB1 (SWI/SNF related matrix associated actin dependent regulator B1; 385aa; 22q11.23; AD LOF;
         Rhabdoid tumor predisposition syndrome 2 (RTPS2);
         malignant rhabdoid tumor/ATRT PATHOGNOMONIC;
         epithelioid sarcoma INI1-IHC-loss PATHOGNOMONIC;
         tazemetostat EZH2i FDA2020; HSCT consolidation;
         seed SEED_BASE+4) .
BRCA2   (Breast cancer gene 2; 3418aa; 13q12.3; AD LOF;
         Hereditary breast & ovarian cancer (HBOC);
         leiomyosarcoma 3-4x elevated; FA-D1 biallelic embryonal RMS;
         cisplatin/olaparib HRD sensitivity;
         seed SEED_BASE+5) .
RB1     (Retinoblastoma gene 1; 928aa; 13q14.2; AD LOF;
         Hereditary retinoblastoma;
         secondary STS 15-20x post-RT PATHOGNOMONIC at radiation field;
         leiomyosarcoma dominant secondary; AVOID high-dose RT;
         CDK4/6i RESISTANT RB1-null;
         seed SEED_BASE+6) .
FH      (Fumarate hydratase; 510aa; 1q43; AD LOF;
         Hereditary leiomyomatosis and renal cell carcinoma (HLRCC);
         uterine leiomyomas PATHOGNOMONIC multiple young-onset;
         cutaneous leiomyomas PATHOGNOMONIC;
         FH-IHC nuclear loss PATHOGNOMONIC; 2SC IHC PATHOGNOMONIC;
         bevacizumab+erlotinib type-2-papillary-RCC;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3254-3261)
"""
import random

SEED_BASE = 3254

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-STS-30-50pct-PATHOGNOMONIC-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes p53 (tumour protein p53; master transcriptional regulator): "
            "STRUCTURE: "
            "  393 aa / 43 kDa; transcription factor with tetramerisation; "
            "  N-terminal transactivation domain 1+2 (aa 1-67); MDM2-binding; "
            "  Proline-rich region (aa 68-98): apoptosis signalling; "
            "  DNA-binding domain (aa 102-292): 80% of cancer hotspot mutations; "
            "  Nuclear localisation signal + tetramerisation domain (aa 293-356); "
            "  C-terminal regulatory domain (aa 357-393): acetylation, ubiquitination; "
            "  p53 hotspots R175H, R248W, R248Q, R273H, R273C, R249S: 'contact' or 'structural' GOF; "
            "  p53 activates: CDKN1A (p21), MDM2, PUMA, NOXA, BAX (apoptosis); "
            "  LFS TP53 germline LOF -> unopposed cell proliferation + genomic instability; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; AD LOF; 1/3,000-1/5,000 births; ~20% de novo; "
            "  Classic criteria: sarcoma proband <45yr + first-degree relative cancer <45yr + first-/second-degree relative cancer <45yr or sarcoma any age; "
            "  Chompret 2015: any sarcoma/brain/breast/adrenocortical <46yr + 1+ relative with LFS tumour; "
            "SOFT TISSUE SARCOMA (TP53 / LFS): "
            "  Soft tissue sarcoma 30-50% lifetime DOMINANT -- PATHOGNOMONIC of LFS when <35yr; "
            "  Tumour types: undifferentiated pleomorphic sarcoma (UPS) 35%; leiomyosarcoma 22%; rhabdomyosarcoma (paediatric) 14%; angiosarcoma 8%; other STS 21%; "
            "  Osteosarcoma: 15-25% LFS (primary bone STS -- peak paediatric); "
            "  Adrenocortical carcinoma: 3-7% LFS children (R337H); "
            "  Brain tumours (DIPG, GBM, choroid plexus): 15%; "
            "  Premenopausal breast: 30% women LFS (high-risk); "
            "RADIATION ABSOLUTE CONTRAINDICATION (TP53 / LFS): "
            "  AVOID RADIATION ABSOLUTELY -- radiation-induced secondary sarcoma/GBM at field; "
            "  DNA damage repair failure -> radiation-field secondary malignancies; "
            "  Post-RT STS in LFS: latency 5-20yr; lethality very high (second primary); "
            "  WBMRI Toronto Protocol: full body MRI annually (NOT CT, NOT PET/CT -- ionising); "
            "  Breast MRI annual from age 20yr; "
            "  US abdomen 3-6 monthly for adrenal lesions; "
            "TREATMENT (TP53 STS): "
            "  First-line: doxorubicin 75mg/m2 IV + ifosfamide 9g/m2 (AI regimen); "
            "  Gemcitabine + docetaxel: second-line leiomyosarcoma; "
            "  Trabectedin: LMS/MLPS second-line; "
            "  MDM2 inhibitors (APG-115, idasanutlin): TP53-wildtype STS not LFS; "
            "  Olaparib: limited data germline TP53 STS (clinical trials); "
            "  R337H Brazilian founder: 1/300 carrier frequency southern Brazil (common population screen); "
            "SURVEILLANCE (LFS): "
            "  WBMRI annually (Toronto: chest/abd/pelvis/limbs); "
            "  Annual brain MRI; "
            "  Annual breast MRI from age 20yr; "
            "  US abdomen q6M (adrenal); "
            "  Colonoscopy from age 25yr q2yr; "
            "  Cascade: all first-degree relatives 50% risk; "
        ),
        "inheritance": "AD LOF; OMIM 151623; 1/3,000-1/5,000 births; ~20% de novo; full penetrance; LFS classic + Chompret criteria; p53 hot-spot mutations (R248W, R175H, R273H, G245S) dominant most pathogenic; germline deletions/truncations LOF",
        "cancer_risk": "Soft tissue sarcoma 30-50% lifetime DOMINANT PATHOGNOMONIC; osteosarcoma 15-25%; premenopausal breast 30%; brain tumour 15% (DIPG, GBM); adrenocortical carcinoma 3-7% children (R337H); colorectal 3-5%",
        "pathognomonic": "Soft tissue sarcoma under 35yr with family history PATHOGNOMONIC; adrenocortical carcinoma in child PATHOGNOMONIC LFS especially R337H; multiple primary tumours including sarcoma + brain + breast; WBMRI full-body surveillance standard",
        "surveillance_key": "AVOID RADIATION ABSOLUTELY; WBMRI Toronto annual NOT CT/PET; annual breast MRI from 20yr; US abdomen q6M adrenal; R337H southern Brazil 1/300 population; radiation-field secondary STS latency 5-20yr lethal",
        "key_distinctions": [
            "STS-30-50PCT-LIFETIME-DOMINANT-PATHOGNOMONIC-LFS",
            "AVOID-RADIATION-ABSOLUTELY-LFS",
            "WBMRI-TORONTO-ANNUAL-NOT-CT",
            "R337H-BRAZILIAN-FOUNDER-1IN300",
            "ADRENOCORTICAL-CA-CHILDREN-PATHOGNOMONIC-R337H",
            "SECONDARY-SARCOMA-POST-RT-LETHAL-TP53",
        ],
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-MPNST-8-13pct-HIGHEST-"
            "Plexiform-NF-Selumetinib-MEKi-FDA2020-AVOID-Radiation-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes Neurofibromin (RAS-GAP tumour suppressor): "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; cytoplasmic RAS GTPase-activating protein (GAP); "
            "  GRD (GTPase-related domain, aa 1175-1530): catalytic RAS-GAP core; "
            "  Accelerates RAS-GTP -> RAS-GDP hydrolysis (400x vs intrinsic rate); "
            "  NF1 LOF -> RAS-GTP accumulates -> RAF-MEK-ERK hyperactivation -> proliferation; "
            "  Second somatic hit required for tumour (Knudson two-hit) in most NF1 tumours; "
            "NF1 MPNST (MALIGNANT PERIPHERAL NERVE SHEATH TUMOUR): "
            "  MPNST 8-13% lifetime HIGHEST sarcoma risk in NF1 PATHOGNOMONIC; "
            "  MPNST arises from plexiform neurofibroma (pNF): rapid painful enlargement = malignant signal; "
            "  NF1-MPNST 5-year survival: 20-50% (worse vs sporadic MPNST 50-60%); "
            "  MPNST surgical margin: wide resection >1cm margin; marginal excision = high recurrence; "
            "  MPNST chemotherapy (ifosfamide + doxorubicin): modest response 25%; "
            "  PRC2 (EED/SUZ12) co-loss in NF1-MPNST: aggressive phenotype; "
            "  AVOID radiation in NF1 tumours: secondary MPNST risk in radiation field; "
            "  Deep plexiform NF: selumetinib prior to surgery (volume reduction); "
            "SELUMETINIB (MEKi FDA2020): "
            "  Selumetinib (AZD6244/Koselugo): MEK1/2 inhibitor; first NF1-specific approved drug; "
            "  FDA2020: symptomatic inoperable plexiform NF children ≥2yr; "
            "  SPRINT trial: 71% plexiform NF volume reduction ≥20%; "
            "  Adult MPNST: selumetinib trials ongoing (SARC006); "
            "  NF1-optic glioma: selumetinib active off-label; "
            "  NF1 DFSP: imatinib (PDGFRB) off-label; "
            "  NF1 breast cancer 3-5x elevated (similar BRCA1): annual breast MRI from 30yr; "
            "SURVEILLANCE (NF1 STS-focused): "
            "  Annual full-body skin examination (cutaneous/subcutaneous NF, DFSP); "
            "  Annual palpation/imaging plexiform NF (rapid growth = MPNST alert); "
            "  Annual MRI target plexiform NF (baseline + growth monitoring); "
            "  PET-FDG: plexiform NF SUV > 3.5 = MPNST transformation signal; "
            "  Annual breast MRI from 30yr; "
            "  AVOID radiation; "
        ),
        "inheritance": "AD LOF; OMIM 162200; 1/3,000 births = most common dominant cancer predisposition; 50% de novo; full penetrance; expressivity variable; NF1 second allele somatic LOF for tumour (Knudson two-hit)",
        "cancer_risk": "MPNST 8-13% PATHOGNOMONIC (highest sarcoma risk NF1); plexiform neurofibroma 25-30% (precursor); DFSP dermatofibrosarcoma protuberans elevated; breast 3-5x; optic glioma 15-20%; GI stromal tumour (GIST) 1-2%",
        "pathognomonic": "Plexiform neurofibroma PATHOGNOMONIC NF1; café-au-lait macules ≥6 PATHOGNOMONIC; Lisch nodules iris hamartomas >90% adults PATHOGNOMONIC; rapid painful plexiform NF enlargement = MPNST transformation PATHOGNOMONIC signal",
        "surveillance_key": "Annual plexiform NF exam + targeted MRI; PET-FDG SUV>3.5 MPNST transformation; selumetinib FDA2020 symptomatic inoperable pNF; AVOID radiation secondary MPNST; annual breast MRI from 30yr; wide surgical margin MPNST",
        "key_distinctions": [
            "MPNST-8-13PCT-HIGHEST-SARCOMA-NF1-PATHOGNOMONIC",
            "PLEXIFORM-NF-MPNST-TRANSFORMATION-PATHOGNOMONIC-SIGNAL",
            "SELUMETINIB-FDA2020-FIRST-NF1-DRUG",
            "AVOID-RADIATION-SECONDARY-MPNST-NF1",
            "PET-FDG-SUV-3-5-MPNST-TRANSFORMATION",
            "NF1-MPNST-5YR-SURVIVAL-20-50PCT-POOR",
        ],
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "APC-309kDa-WNT-Beta-Catenin-Tumour-Suppressor-Gardner-Desmoid-"
            "PATHOGNOMONIC-Sulindac-Sorafenib-Surgery-AVOID-Mesenteric-OMIM-175100"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 309 kDa / 5q22.2 APC encodes adenomatous polyposis coli (WNT pathway scaffold): "
            "STRUCTURE: "
            "  2843 aa / 309 kDa; scaffold protein in WNT signalling; "
            "  Armadillo repeats (aa 334-767): binding domain; "
            "  SAMP repeats (aa 1020, 1296, 1526): axin binding -> destruction complex; "
            "  15-aa repeats (aa 1020-1169): beta-catenin binding (3 of 7 sites); "
            "  MCR (mutation cluster region, aa 1250-1500): most codon 1309/1310/1311; "
            "  APC LOF -> destruction complex (APC/Axin/CK1/GSK3) impaired; "
            "  Beta-catenin not phosphorylated -> not ubiquitinated -> nuclear accumulation; "
            "  Nuclear beta-catenin + TCF/LEF: transcriptional activation -> MYC, cyclin D1; "
            "DESMOID FIBROMATOSIS (APC / GARDNER / FAP): "
            "  Desmoid tumour 10-30% FAP/Gardner PATHOGNOMONIC; "
            "  Mesenteric desmoid: most clinically dangerous (bowel obstruction, ureteric compression); "
            "  Mesenteric desmoid prevalence: 8-12% FAP; post-colectomy trigger (surgery stimulus); "
            "  Abdominal wall desmoid: 12-18% FAP; less dangerous; "
            "  Extra-abdominal desmoid: shoulder/limb girdle; least dangerous; "
            "  Desmoid genotype-phenotype: MCR beyond codon 1444 = HIGHEST desmoid risk; "
            "  I1307K (Ashkenazi Jewish founder, 5q22.2 region): 1% population-level FAP risk; "
            "DESMOID MANAGEMENT (APC / GARDNER): "
            "  Surgery AVOID for mesenteric desmoid: paradoxical growth post-resection (documented); "
            "  Sulindac (NSAID): first-line anti-proliferative desmoid; polyp regression + desmoid; "
            "  Celecoxib (COX2i): alternative NSAID; superior tolerability; "
            "  Sorafenib: POSITIVE RCT (SORAFENIB vs placebo 2019) -- PFS benefit mesenteric desmoid; "
            "  Imatinib (PDGFRB inhibitor): off-label; response 10-15%; "
            "  Nirogacestat (gamma-secretase inhibitor Notch): FDA2023 desmoid (sporadic/germline); "
            "  Tamoxifen/hormonal therapy: historical; modest benefit; anti-estrogenic; "
            "  Pazopanib: off-label TKI; response 35% PALETTE trial (NOT desmoid specific); "
            "SURVEILLANCE (APC DESMOID): "
            "  Colonoscopy annually post-colectomy (surveillance of rectal remnant); "
            "  Abdominopelvic MRI at diagnosis (desmoid staging); "
            "  Annual abdominopelvic MRI (desmoid surveillance); "
            "  Upper GI endoscopy q1-3yr (duodenal adenoma surveillance); "
            "  Cascade: first-degree relatives 50% FAP risk; molecular testing or colonoscopy; "
        ),
        "inheritance": "AD LOF; OMIM 175100; 1/7,000-1/10,000 births; ~25% de novo; FAP penetrance near 100% for polyps; desmoid 10-30%; codon 1444+ highest desmoid risk genotype-phenotype; attenuated FAP (AFAP): 5' or 3' or exon 9 mutations",
        "cancer_risk": "Desmoid fibromatosis 10-30% PATHOGNOMONIC (mesenteric most dangerous); colorectal cancer near 100% by 40yr without prophylactic colectomy; duodenal/periampullary cancer 4-12%; thyroid cancer (cribriform-morular variant) 1-2%; medulloblastoma 1%",
        "pathognomonic": "Desmoid fibromatosis mesenteric PATHOGNOMONIC Gardner syndrome; >100 colorectal adenomatous polyps PATHOGNOMONIC FAP; CHRPE (congenital hypertrophy of retinal pigment epithelium) bilateral multifocal PATHOGNOMONIC FAP; osteomas jaw PATHOGNOMONIC Gardner",
        "surveillance_key": "Surgery AVOID mesenteric desmoid (paradoxical growth); sulindac/celecoxib first-line; nirogacestat FDA2023 desmoid; sorafenib RCT positive; annual abdominal MRI; codon 1444+ = HIGHEST desmoid risk; prophylactic colectomy before 25yr",
        "key_distinctions": [
            "DESMOID-MESENTERIC-PATHOGNOMONIC-GARDNER-FAP-10-30PCT",
            "SURGERY-AVOID-MESENTERIC-DESMOID-PARADOXICAL-GROWTH",
            "NIROGACESTAT-FDA2023-DESMOID-GAMMA-SECRETASE",
            "SORAFENIB-RCT-POSITIVE-MESENTERIC-DESMOID",
            "CODON-1444-PLUS-HIGHEST-DESMOID-RISK",
            "SULINDAC-CELECOXIB-FIRST-LINE-DESMOID",
        ],
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-219kDa-RNase-III-miRNA-PPB-PATHOGNOMONIC-"
            "Cervical-ERMS-PATHOGNOMONIC-AVOID-Radiation-Children-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 219 kDa / 14q32.13 DICER1 encodes DICER1 (ribonuclease III; miRNA processor): "
            "STRUCTURE: "
            "  1922 aa / 219 kDa; nuclear-cytoplasmic RNase III enzyme; "
            "  PAZ domain (aa 860-972): dsRNA 3' end recognition; "
            "  Connector helix (aa 973-1038); "
            "  RNase IIIa domain (aa 1244-1472): catalytic -- cuts guide strand; "
            "  RNase IIIb domain (aa 1473-1702): catalytic -- cuts passenger strand; "
            "  KEY: somatic second-hit hotspot mutations cluster in RNase IIIb metal-ion binding residues (E1705, D1709, E1813, D1810, G1809); "
            "  DICER1 germline LOF (first hit) + somatic RNase IIIb hotspot (second hit) -> biallelic inactivation -> miRNA processing failure -> oncogenesis; "
            "  miRNA dysregulation -> cell cycle, apoptosis, differentiation pathway disruption; "
            "DICER1 SYNDROME: "
            "  OMIM 606241; AD LOF; 1/10,000-1/30,000 estimated incidence; 30% de novo; "
            "  DICER1 syndrome spectrum: PPB, cervical ERMS, SLCT, cystic nephroma, multi-nodular thyroid, others; "
            "  Somatic hotspot = pathognomonic second hit -- confirms biallelic DICER1 inactivation; "
            "PLEUROPULMONARY BLASTOMA (PPB): "
            "  PPB: PATHOGNOMONIC DICER1 syndrome children; "
            "  PPB type I (cystic, <2yr): cysts only; best prognosis; "
            "  PPB type II (cystic-solid, 2-4yr): mixed; intermediate; "
            "  PPB type III (solid, >4yr): most aggressive; 5yr OS 40%; "
            "  AVOID radiation in PPB (young children, developing lungs + radiation sensitivity); "
            "  PPB treatment: surgery + VIP (etoposide/ifosfamide/cisplatin) + consolidation; "
            "  PPB: DICER1 germline testing of ENTIRE FAMILY -- siblings screened CT chest; "
            "CERVICAL EMBRYONAL RMS (ERMS): "
            "  Cervical ERMS: PATHOGNOMONIC DICER1 -- rare sarcoma botryoides of cervix; "
            "  Cervical ERMS median age: 12yr (adolescent/young adult); "
            "  Fertility-sparing surgery if feasible (young patient, local disease); "
            "  VAC (vincristine/actinomycin/cyclophosphamide) chemotherapy standard; "
            "  Avoid pelvic radiation in young DICER1 females; "
            "SERTOLI-LEYDIG CELL TUMOUR (SLCT): "
            "  SLCT: DICER1 somatic hot-spot in >60% of moderately/poorly differentiated SLCT; "
            "  Germline DICER1 in ~15% SLCT; androgen production (virilisation); "
            "  Bilateral sequential SLCT = DICER1 germline highly likely; "
            "OTHER DICER1 TUMOURS: "
            "  Multi-nodular thyroid goitre: most common DICER1 feature; "
            "  Cystic nephroma / anaplastic Wilms overlap; "
            "  Nasal chondromesenchymal hamartoma (NCMH); "
            "  Pituitary blastoma (PATO phenotype, <24mo); "
            "  Pinealoblastoma; embryonal tumour with multilayered rosettes (ETMR); "
            "SURVEILLANCE (DICER1): "
            "  CT chest q6M to 2yr, then annually to age 6yr (PPB); "
            "  Annual thyroid ultrasound from diagnosis (multi-nodular goitre); "
            "  Annual abdominal US (renal cystic nephroma); "
            "  Annual gynaecological exam from puberty (SLCT, cervical ERMS); "
            "  Cascade: first-degree relatives 50% -- CT chest siblings <8yr; "
        ),
        "inheritance": "AD LOF (germline first hit); somatic RNase IIIb hotspot = second hit; OMIM 606241; 1/10,000-1/30,000; 30% de novo; two-hit model; somatic hotspot in RNase IIIb domain distinguishes DICER1 syndrome from sporadic tumour",
        "cancer_risk": "PPB (pleuropulmonary blastoma) PATHOGNOMONIC -- type I/II/III; cervical ERMS PATHOGNOMONIC adolescent; SLCT Sertoli-Leydig >60% somatic hotspot; multi-nodular thyroid goitre most common feature; cystic nephroma; pituitary blastoma rare",
        "pathognomonic": "Pleuropulmonary blastoma type I/II/III PATHOGNOMONIC DICER1; cervical embryonal RMS PATHOGNOMONIC DICER1; bilateral multi-nodular thyroid goitre in child/young adult + PPB family history PATHOGNOMONIC DICER1 syndrome",
        "surveillance_key": "CT chest q6M to 2yr then annually to age 6yr (PPB screening); AVOID radiation children; annual thyroid US; annual abdominal US; PPB type III VIP + surgery; germline test ALL family including siblings; cervical ERMS fertility-sparing",
        "key_distinctions": [
            "PPB-PATHOGNOMONIC-DICER1-TYPE-I-II-III",
            "CERVICAL-ERMS-PATHOGNOMONIC-DICER1-ADOLESCENT",
            "AVOID-RADIATION-CHILDREN-DICER1",
            "SOMATIC-RNASE-IIIb-HOTSPOT-SECOND-HIT-PATHOGNOMONIC",
            "SLCT-DICER1-60PCT-SOMATIC-HOTSPOT",
            "CT-CHEST-SCREENING-SIBLINGS-UNDER-8YR",
        ],
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 Autosomal-Dominant-LOF -- 385aa -- "
            "INI1-44kDa-SWI-SNF-MRT-PATHOGNOMONIC-ATRT-PATHOGNOMONIC-"
            "Epithelioid-Sarcoma-INI1-IHC-Loss-Tazemetostat-EZH2i-FDA2020-OMIM-609322"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / 44 kDa / 22q11.23 SMARCB1 encodes INI1 / BAF47 (SWI/SNF chromatin remodelling core subunit): "
            "STRUCTURE: "
            "  385 aa / 44 kDa; core structural subunit of BAF/PBAF SWI/SNF complex; "
            "  RVT (repeat-like) motif (aa 5-66, 150-212): DNA-binding; "
            "  SNH domain (aa 282-383): nuclear localisation + protein-protein interaction; "
            "  SMARCB1 LOF -> SWI/SNF complex destabilisation -> loss of BAF/PBAF function; "
            "  SMARCB1 LOF -> PRC2 (EZH2) activity unopposed -> H3K27me3 accumulation; "
            "  H3K27me3 -> global transcriptional repression -> developmental arrest -> tumour; "
            "  Tazemetostat (EZH2 inhibitor) works: SMARCB1 LOF = EZH2 'dependency' -- synthetic lethality; "
            "  SMARCB1 IHC: nuclear expression = normal; nuclear LOSS = SMARCB1 inactivation PATHOGNOMONIC; "
            "MALIGNANT RHABDOID TUMOUR (MRT): "
            "  MRT: PATHOGNOMONIC SMARCB1 biallelic LOF; extrarenal, renal, CNS (ATRT); "
            "  MRT median age: <2yr (early infancy); rapid lethal course; "
            "  INI1 IHC nuclear loss PATHOGNOMONIC MRT; EMA+, Vimentin+, CK+ (variable); "
            "  MRT treatment: ICE (ifosfamide/carboplatin/etoposide) induction; HSCT consolidation; "
            "  MRT 5yr OS: <30% (very poor prognosis); bilateral/multifocal = germline; "
            "  MRT family: RTPS2 (Rhabdoid Tumour Predisposition Syndrome type 2); "
            "ATRT (ATYPICAL TERATOID/RHABDOID TUMOUR): "
            "  ATRT: CNS MRT; under 3yr PATHOGNOMONIC SMARCB1 germline; "
            "  ATRT germline SMARCB1: 30-50% of ATRT cases -- de novo ~50%; "
            "  ATRT subtypes: ATRT-TYR (tyrosinase), ATRT-SHH, ATRT-MYC; "
            "  ATRT treatment: surgery + ICE + intrathecal chemo + HSCT; "
            "EPITHELIOID SARCOMA: "
            "  Epithelioid sarcoma (ES): SMARCB1/INI1 loss in 90% PATHOGNOMONIC; "
            "  Proximal type ES: young adults (20-40yr); pelvis/perineum/proximal extremities; aggressive; "
            "  Classic (distal) ES: finger/forearm distal; more indolent; "
            "  Tazemetostat (EZH2i): FDA2020 locally advanced/metastatic ES with SMARCB1 loss; "
            "  ES treatment: tazemetostat 800mg BID; doxorubicin/ifosfamide; "
            "  ES SMARCB1: nuclear IHC loss in 90% -- essential diagnostic criterion; "
            "SURVEILLANCE (SMARCB1): "
            "  MRI brain + full spine at diagnosis and q6M to age 5yr; "
            "  US abdomen q6M to age 5yr (renal MRT); "
            "  Annual MRI from age 5yr (surveillance); "
            "  Cascade: first-degree relatives molecular testing (50% risk); "
            "  De novo ~50%: both parents molecular testing; "
        ),
        "inheritance": "AD LOF; OMIM 609322; rare (<1/200,000 births); ~50% de novo; MRT/ATRT germline vs somatic: somatic only = no cascade risk; germline = 50% first-degree risk; RTPS2 (rhabdoid tumour predisposition syndrome 2); SMARCB1 vs SMARCA4 (RTPS1): different gene same syndrome",
        "cancer_risk": "MRT (extrarenal + renal) PATHOGNOMONIC biallelic SMARCB1; ATRT under 3yr PATHOGNOMONIC germline SMARCB1 30-50%; epithelioid sarcoma INI1-loss 90% PATHOGNOMONIC; schwannomatosis elevated; EMC extraskeletal myxoid chondrosarcoma rare",
        "pathognomonic": "INI1 IHC nuclear loss PATHOGNOMONIC SMARCB1 inactivation (MRT, ATRT, epithelioid sarcoma); MRT under 2yr PATHOGNOMONIC; bilateral/multifocal ATRT = germline SMARCB1 near-certain; tazemetostat synthetic lethality via EZH2 dependency",
        "surveillance_key": "MRI brain + full spine q6M to age 5yr; US abdomen q6M (renal MRT); tazemetostat EZH2i FDA2020 epithelioid sarcoma; ICE + HSCT consolidation MRT/ATRT; INI1 IHC essential for ES diagnosis; both parents molecular test (50% de novo)",
        "key_distinctions": [
            "MRT-ATRT-PATHOGNOMONIC-SMARCB1-BIALLELIC",
            "INI1-IHC-NUCLEAR-LOSS-PATHOGNOMONIC",
            "TAZEMETOSTAT-EZH2I-FDA2020-EPITHELIOID-SARCOMA",
            "HSCT-CONSOLIDATION-MRT-ATRT",
            "EPITHELIOID-SARCOMA-PROXIMAL-TYPE-YOUNG-ADULTS",
            "DE-NOVO-50PCT-BOTH-PARENTS-MOLECULAR-TEST",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-HBOC-LMS-3-4x-FA-D1-ERMS-"
            "Cisplatin-HRD-Olaparib-PARP-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes BRCA2 (homologous recombination DNA repair scaffold): "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; nuclear scaffold for homologous recombination (HR) repair; "
            "  OB folds (aa 2402-3190): ssDNA binding (4 OB folds); "
            "  BRC repeats (aa 1002-2082): RAD51 binding (8 BRC repeats); "
            "  RAD51-binding C-terminal (aa 3190-3266): second RAD51 interaction; "
            "  PALB2-binding N-terminal (aa 10-40): nuclear localisation partner; "
            "  BRCA2 + RAD51 -> ssDNA loading at DSB -> strand invasion -> HR repair; "
            "  BRCA2 LOF -> HR impaired -> NHEJ/SSA error-prone repair -> genomic instability; "
            "  HRD (homologous recombination deficiency) tumour signature: LOH + LST + TAI; "
            "BRCA2 SOFT TISSUE SARCOMA RISK: "
            "  Uterine leiomyosarcoma: BRCA2 germline 3-4x elevated risk; "
            "  Retroperitoneal LMS: BRCA2 germline elevated; "
            "  Embryonal RMS (ERMS): biallelic BRCA2 (Fanconi Anemia complementation D1, FA-D1) -> RMS PATHOGNOMONIC; "
            "  FA-D1 biallelic: Wilms tumour 50%, RMS 30%, medulloblastoma SHH, AML -- most severe FA subtype; "
            "  BRCA2 monoallelic: 2-4x STS risk (NCI prospective data); "
            "CISPLATIN / HRD SENSITIVITY (BRCA2 STS): "
            "  HRD STS (BRCA2, BRCA1, PALB2): enhanced cisplatin sensitivity; "
            "  Cisplatin/carboplatin-based regimens preferred in HRD STS; "
            "  Olaparib (PARP inhibitor): FDA-approved BRCA1/2 germline HER2-negative breast/ovarian/pancreatic/prostate; "
            "  Olaparib in STS: off-label; clinical trials SARC045 ongoing (germline BRCA2 STS); "
            "  Gemcitabine/docetaxel: LMS second-line regardless of BRCA2 status; "
            "  Niraparib + pembrolizumab: HRD STS trial ongoing; "
            "SURVEILLANCE (BRCA2 STS): "
            "  Annual breast MRI + mammogram from 25-30yr; "
            "  Annual transvaginal US + CA-125 from 30yr (ovarian); "
            "  Prophylactic bilateral salpingo-oophorectomy (PBSO) by 40-45yr; "
            "  Annual abdominal/pelvic MRI (uterine LMS, retroperitoneal); "
            "  Prostate PSA + DRE annual from 40yr (male BRCA2); "
            "  Cascade: all first-degree relatives 50% risk; "
        ),
        "inheritance": "AD LOF; OMIM 600185; 1/400-1/1,000 births (HBOC pathogenic variants); 0.1-0.5% general population; BRCA2 biallelic = FA-D1 (very rare, severe paediatric phenotype); second allele somatic LOH/mutation for tumorigenesis (Knudson two-hit)",
        "cancer_risk": "Breast 47-69% (lifetime); ovarian 11-17%; pancreatic 3-5% (HIGHEST germline risk); prostate 15-20% (aggressive); uterine LMS/retroperitoneal LMS 3-4x elevated; FA-D1 biallelic: Wilms 50% + RMS 30% + medulloblastoma",
        "pathognomonic": "FA-D1 biallelic BRCA2 = Wilms tumour + RMS + medulloblastoma PATHOGNOMONIC most severe FA; uterine LMS in BRCA2 carrier HRD signature; olaparib PARP synthetic lethality in BRCA2-null tumour; cisplatin sensitivity HRD",
        "surveillance_key": "Annual breast MRI + mammogram from 25-30yr; PBSO by 40-45yr; cisplatin/olaparib HRD sensitivity LMS; annual abdominopelvic MRI; FA-D1 biallelic severe paediatric cancer syndrome; SARC045 olaparib STS trial",
        "key_distinctions": [
            "LMS-3-4X-ELEVATED-BRCA2-UTERINE-RETROPERITONEAL",
            "FA-D1-BIALLELIC-RMS-WILMS-MEDULLOBLASTOMA-PATHOGNOMONIC",
            "CISPLATIN-HRD-SENSITIVITY-BRCA2-STS",
            "OLAPARIB-PARP-SARC045-TRIAL",
            "PBSO-BY-40-45YR-BRCA2-FEMALE",
            "HRD-SIGNATURE-LOH-LST-TAI",
        ],
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "pRb-105kDa-E2F-Cell-Cycle-Retinoblastoma-Secondary-STS-15-20x-Post-RT-"
            "PATHOGNOMONIC-CDK4-6i-RESISTANT-AVOID-High-Dose-RT-OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 105 kDa / 13q14.2 RB1 encodes pRb (retinoblastoma protein; E2F transcription factor regulator): "
            "STRUCTURE: "
            "  928 aa / 105 kDa; pocket protein family (RB1, RBL1/p107, RBL2/p130); "
            "  N-terminal domain (aa 1-355); "
            "  Pocket domain A + B (aa 356-868): E2F binding (A-B cleft); "
            "  C-terminal domain (aa 869-928): CDK phosphorylation sites; "
            "  RB1 hypophosphorylated (active): binds E2F1/2/3 -> represses S-phase genes; "
            "  CDK4/6-cyclin D phosphorylates RB1 -> releases E2F -> S-phase entry; "
            "  RB1 LOF -> E2F constitutively active -> uncontrolled cell cycle -> tumour; "
            "  CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib) require intact RB1: "
            "  CDK4/6i RESISTANT in RB1-null tumours (no substrate to phosphorylate -- target absent); "
            "HEREDITARY RETINOBLASTOMA (RB1): "
            "  OMIM 180200; AD LOF; 1/15,000-1/20,000 live births; "
            "  Hereditary Rb: bilateral/multifocal = GERMLINE until proven otherwise; "
            "  40% Rb is hereditary (germline RB1); 60% sporadic (somatic biallelic); "
            "  Bilateral Rb: >95% hereditary; unilateral Rb: ~15% hereditary; "
            "  Germline RB1: 50% risk to offspring; genetic counselling MANDATORY; "
            "  Treatment: enucleation + chemotherapy (carboplatin/etoposide/vincristine); "
            "  AVOID high-dose external beam RT in germline RB1 (secondary sarcoma risk); "
            "  Intra-arterial chemotherapy: melphalan (first-line eye salvage); "
            "SECONDARY SOFT TISSUE SARCOMA (RB1 POST-RT): "
            "  Secondary STS at radiation field: 15-20x elevated risk in germline RB1; "
            "  Radiation-induced osteosarcoma in orbit (post-RT Rb): PATHOGNOMONIC secondary; "
            "  Leiomyosarcoma: most common radiation-induced secondary STS in RB1; "
            "  Secondary STS latency: 5-30yr post-RT; mean 10-15yr; "
            "  Treatment secondary STS: doxorubicin/ifosfamide; no repeat RT; surgery if feasible; "
            "  CDK4/6 inhibitors NOT ACTIVE in RB1-null STS (mechanism of resistance); "
            "  Amplified CDK4 sarcoma (DDL/WDL liposarcoma): CDK4/6i ACTIVE (RB1 intact); "
            "SURVEILLANCE (RB1): "
            "  Retinoblastoma: EUA (examination under anaesthesia) q3-4W until age 3yr, then q3-6M to 5yr; "
            "  Annual MRI orbits/brain (pinealoblastoma screening in hereditary Rb); "
            "  Secondary STS surveillance: annual MRI pelvis/limbs (prior RT fields); "
            "  Avoid repeated CT imaging (radiation); "
            "  Cascade: offspring testing on day 1 of life; "
        ),
        "inheritance": "AD LOF; OMIM 180200; 1/15,000-1/20,000 births; 40% Rb hereditary; bilateral/multifocal Rb = germline until proven; de novo ~7% new germline; full penetrance for Rb (>95% bilateral); variable penetrance low-penetrance RB1 variants",
        "cancer_risk": "Retinoblastoma (bilateral near-100% germline); secondary STS 15-20x at radiation field PATHOGNOMONIC; osteosarcoma (secondary post-RT 30%; independent RB1 STS risk 4-7%); leiomyosarcoma secondary; melanoma elevated; lung cancer elevated",
        "pathognomonic": "Bilateral/multifocal retinoblastoma PATHOGNOMONIC hereditary RB1; radiation-field secondary osteosarcoma/LMS in prior Rb patient = RB1 germline until proven PATHOGNOMONIC; CDK4/6i RESISTANT in confirmed RB1-null sarcoma (key prescribing rule)",
        "surveillance_key": "EUA q3-4W to age 3yr; annual MRI orbits/brain (pinealoblastoma); secondary STS surveillance annual MRI prior RT fields; AVOID high-dose external beam RT; CDK4/6i RESISTANT RB1-null; offspring testing day 1 of life",
        "key_distinctions": [
            "SECONDARY-STS-15-20X-POST-RT-PATHOGNOMONIC-RB1",
            "BILATERAL-MULTIFOCAL-RB-PATHOGNOMONIC-GERMLINE",
            "CDK4-6I-RESISTANT-RB1-NULL-STS",
            "AVOID-HIGH-DOSE-EXTERNAL-BEAM-RT-RB1",
            "RADIATION-INDUCED-LMS-DOMINANT-SECONDARY-STS",
            "OFFSPRING-TESTING-DAY-1-LIFE",
        ],
    },
    {
        "gene": "FH",
        "protein": (
            "FH -- 1q43 Autosomal-Dominant-LOF -- 510aa -- "
            "Fumarate-Hydratase-55kDa-TCA-Cycle-HLRCC-Uterine-Leiomyoma-"
            "PATHOGNOMONIC-Cutaneous-Leiomyoma-PATHOGNOMONIC-FH-IHC-2SC-PATHOGNOMONIC-"
            "Bevacizumab-Erlotinib-Type2-Papillary-RCC-OMIM-150800"
        ),
        "locus": "1q43",
        "protein_size": (
            "510 aa / 55 kDa / 1q43 FH encodes fumarate hydratase (TCA cycle enzyme; tumour suppressor): "
            "STRUCTURE: "
            "  510 aa / 55 kDa (mature monomer); mitochondrial and cytoplasmic isoforms; "
            "  FH catalyses fumarate -> malate (TCA cycle; step 7 of 8); "
            "  FH tetramer: active functional form; Class II lyase fold; "
            "  FH LOF -> fumarate accumulates (>1,000-fold intracellular accumulation); "
            "  Fumarate: oncometabolite; competitive inhibitor of alpha-ketoglutarate-dependent dioxygenases; "
            "  Fumarate inhibits: PHD (prolyl hydroxylase) -> HIF-1alpha accumulates (pseudo-hypoxia); "
            "  Fumarate inhibits: TET enzymes -> genome-wide DNA hypermethylation; "
            "  Fumarate inhibits: KDM histone demethylases -> H3K9me3 chromatin remodelling; "
            "  Fumarate + cysteine -> 2-succinocysteine (2SC): PATHOGNOMONIC IHC marker for FH LOF; "
            "  2SC IHC (S-(2-succino)-cysteine): positive cytoplasmic staining in FH-deficient tissue = PATHOGNOMONIC; "
            "  FH IHC: nuclear + cytoplasmic loss = FH deficiency (normal: nuclear + cytoplasmic positive); "
            "HEREDITARY LEIOMYOMATOSIS AND RCC (HLRCC): "
            "  OMIM 150800; AD LOF; 1/50,000-1/100,000 (likely underdiagnosed); "
            "  HLRCC: uterine leiomyoma + cutaneous leiomyoma + type 2 papillary RCC; "
            "  Fumarate accumulation: molecular driver oncogenesis HLRCC; "
            "UTERINE LEIOMYOMA (HLRCC): "
            "  Uterine leiomyoma: PATHOGNOMONIC HLRCC -- multiple, large, early onset (mean 30yr); "
            "  FH-associated fibroids: earlier, larger, more numerous than sporadic; "
            "  Histology: large nuclei, prominent orangeophilic nucleoli = PATHOGNOMONIC FH-associated; "
            "  Hysterectomy often required 30-35yr (multiple large fibroids); "
            "  MyomyoFH (ESMO 2023): FH mutations in fibroids -- therapeutic targets; "
            "CUTANEOUS LEIOMYOMA (HLRCC): "
            "  Cutaneous leiomyomas (multiple): PATHOGNOMONIC HLRCC; "
            "  Distribution: trunk, extremities; symptomatic (cold/touch triggered pain); "
            "  Solitary cutaneous leiomyoma in young adult (<35yr) = FH testing; "
            "  Multiple cutaneous leiomyomas = FH germline until proven (85%+ of multi-cutaneous-LM); "
            "TYPE 2 PAPILLARY RCC (HLRCC): "
            "  FH-deficient type 2 papillary RCC PATHOGNOMONIC HLRCC; "
            "  Type 2 pRCC aggressive: high-grade, early metastasis, poor prognosis vs type 1 pRCC; "
            "  FH IHC + 2SC IHC on renal mass biopsy: PATHOGNOMONIC FH-deficient RCC; "
            "  Bevacizumab + erlotinib: best combination FH-deficient pRCC (anti-VEGF + anti-EGFR); "
            "  Bevacizumab alone: modest benefit (30-40% response, 14mo PFS); "
            "  Erlotinib + bevacizumab: ORR 64% (Srinivasan 2014 JCO); "
            "  Cabozantinib (MET+VEGFR2): emerging option FH-deficient RCC; "
            "  mTOR inhibitors (everolimus): modest benefit FH-deficient (PI3K/mTOR pathway activated); "
            "SURVEILLANCE (FH): "
            "  Annual abdominal/pelvic MRI from age 10yr (renal surveillance -- AGGRESSIVE screening); "
            "  Annual gynaecological US + MRI (uterine fibroids); "
            "  FH IHC on any renal biopsy (screen for FH-deficiency); "
            "  2SC IHC: confirmatory FH deficiency on tissue; "
            "  Cascade: first-degree relatives 50% risk; "
        ),
        "inheritance": "AD LOF; OMIM 150800; 1/50,000-1/100,000 (underdiagnosed); full penetrance for uterine fibroids; variable penetrance for RCC; FH biallelic = fumarase deficiency (severe neurometabolic, encephalopathy) -- DIFFERENT from HLRCC monoallelic; Knudson two-hit for RCC",
        "cancer_risk": "Uterine leiomyoma PATHOGNOMONIC multiple early onset; cutaneous leiomyoma PATHOGNOMONIC multiple symptomatic; type 2 papillary RCC PATHOGNOMONIC FH-deficient aggressive; collecting duct carcinoma rare; adrenal cortical carcinoma rare (fumarate inhibits SDH epigenetically)",
        "pathognomonic": "Multiple cutaneous leiomyomas young adult PATHOGNOMONIC HLRCC; uterine leiomyoma <35yr multiple/large PATHOGNOMONIC HLRCC; FH IHC nuclear loss PATHOGNOMONIC; 2SC IHC positive (succino-cysteine) PATHOGNOMONIC FH deficiency; type 2 pRCC + FH-deficiency = HLRCC",
        "surveillance_key": "Annual abdominal/pelvic MRI from age 10yr (aggressive RCC screening); FH IHC + 2SC IHC on renal biopsy; bevacizumab + erlotinib best combination type 2 pRCC; hysterectomy often 30-35yr (multiple fibroids); 2SC IHC diagnostic on any tissue",
        "key_distinctions": [
            "UTERINE-LEIOMYOMA-MULTIPLE-YOUNG-ONSET-PATHOGNOMONIC-HLRCC",
            "CUTANEOUS-LEIOMYOMA-MULTIPLE-SYMPTOMATIC-PATHOGNOMONIC",
            "FH-IHC-NUCLEAR-LOSS-PATHOGNOMONIC",
            "2SC-IHC-POSITIVE-SUCCINO-CYSTEINE-PATHOGNOMONIC-FH",
            "BEVACIZUMAB-ERLOTINIB-TYPE2-PAPILLARY-RCC-BEST-COMBINATION",
            "ANNUAL-MRI-FROM-AGE-10YR-AGGRESSIVE-RCC-SCREENING",
        ],
    },
]

# -------------------------------------------------------------------
# Per-gene patient simulation data
# -------------------------------------------------------------------
TUMOR_TYPES = {
    "TP53":    ["undiff-pleomorphic-sarcoma", "leiomyosarcoma", "rhabdomyosarcoma", "angiosarcoma", "malignant-fibrous-histiocytoma", "osteosarcoma-secondary"],
    "NF1":     ["MPNST", "plexiform-NF-malignant", "DFSP-dermatofibrosarcoma", "spindle-cell-sarcoma", "high-grade-NF1-sarcoma"],
    "APC":     ["desmoid-mesenteric", "desmoid-abdominal-wall", "desmoid-extra-abdominal", "intra-abdominal-lipoma", "Gardner-associated-sarcoma-rare"],
    "DICER1":  ["pleuropulmonary-blastoma-type-III", "pleuropulmonary-blastoma-type-I", "pleuropulmonary-blastoma-type-II", "cervical-ERMS", "SLCT-sertoli-leydig"],
    "SMARCB1": ["malignant-rhabdoid-MRT", "ATRT-CNS", "epithelioid-sarcoma-proximal", "epithelioid-sarcoma-distal", "extraskeletal-myxoid-chondrosarcoma"],
    "BRCA2":   ["uterine-leiomyosarcoma", "retroperitoneal-LMS", "embryonal-RMS-FA-D1", "high-grade-STS-HRD", "soft-tissue-LMS"],
    "RB1":     ["retinoblastoma-primary", "secondary-LMS-post-RT", "secondary-osteosarcoma-post-RT", "secondary-angiosarcoma-post-RT", "secondary-sarcoma-post-RT-NOS"],
    "FH":      ["uterine-leiomyoma-multiple", "cutaneous-leiomyoma", "type2-papillary-RCC", "collecting-duct-carcinoma", "FH-deficient-sarcoma-rare"],
}

PATHOGENIC_VARIANTS = {
    "TP53":    {"p.R248W": 9, "p.R175H": 7, "p.R273H": 6, "p.G245S": 5, "p.R337H": 5, "c.375+1G>A": 4, "large-del": 4},
    "NF1":     {"NF1-frameshift-c.2033_2034del": 8, "NF1-splice-c.2041-1G>A": 7, "p.R681X": 6, "NF1-del-exon1-3": 6, "p.R1306X": 5, "NF1-large-del": 5, "NF1-missense-GRD": 3},
    "APC":     {"c.3927_3931del": 9, "p.E1309D": 7, "p.I1307K": 7, "c.1309del": 6, "c.2644del": 5, "p.E1317Q": 4, "p.R1450X": 2},
    "DICER1":  {"p.E1705K-RNase-IIIb": 10, "p.D1709N-RNase-IIIb": 8, "p.E1813K-RNase-IIIb": 7, "DICER1-frameshift-c.5428del": 6, "DICER1-splice-c.5117-1G>T": 5, "p.G1809R": 4},
    "SMARCB1": {"p.R40X": 9, "SMARCB1-del-exon3-4": 8, "p.E110del": 7, "SMARCB1-large-del": 7, "c.1141-2A>G": 5, "p.R241H": 4},
    "BRCA2":   {"c.5946delT": 8, "c.6275_6276del": 7, "p.W31X": 6, "p.K3326X": 6, "p.I2490T": 5, "c.517-1G>T": 5, "BRCA2-large-del": 3},
    "RB1":     {"p.R445X": 8, "p.R661W": 7, "p.R455X": 6, "RB1-splice-c.1981-1G>T": 6, "RB1-large-del": 6, "p.Y79X": 4, "RB1-exon-skipping": 3},
    "FH":      {"p.H153Y": 8, "p.R233H": 7, "p.R58P": 6, "FH-splice-c.714+1G>A": 6, "FH-large-del": 5, "p.N289S": 5, "p.P387S": 3},
}

TREATMENT_PROTOCOLS = {
    "TP53":    ["doxorubicin-75mgm2-ifosfamide-9gm2-AI", "gemcitabine-docetaxel-LMS-2L", "trabectedin-LMS-MLPS-2L", "surgery-wide-margin", "WBMRI-annual-Toronto", "NO-radiation-ABSOLUTE-CI"],
    "NF1":     ["selumetinib-MEKi-FDA2020-plexiform-NF", "doxorubicin-ifosfamide-MPNST", "surgery-wide-margin-MPNST", "everolimus-mTOR-plexiform-refractory", "NO-radiation-secondary-MPNST-risk"],
    "APC":     ["sulindac-NSAID-first-line", "celecoxib-COX2i-alternative", "nirogacestat-FDA2023-gamma-secretase", "sorafenib-RCT-positive", "imatinib-PDGFRB-off-label", "surgery-AVOID-mesenteric", "tamoxifen-historical"],
    "DICER1":  ["surgery-PPB", "VIP-etoposide-ifosfamide-cisplatin-PPB-III", "VAC-ERMS-cervical", "NO-radiation-children-developing-lungs", "observation-PPB-type-I"],
    "SMARCB1": ["tazemetostat-EZH2i-FDA2020-ES", "ICE-ifosfamide-carboplatin-etoposide-MRT", "HSCT-autologous-consolidation-ATRT-MRT", "intrathecal-chemotherapy-ATRT", "multiagent-MRT-protocol"],
    "BRCA2":   ["doxorubicin-ifosfamide-LMS-1L", "gemcitabine-docetaxel-LMS-2L", "cisplatin-HRD-sensitivity", "olaparib-PARP-off-label-SARC045-trial", "PBSO-prophylactic-salpingo-oophorectomy"],
    "RB1":     ["carboplatin-etoposide-vincristine-Rb", "intra-arterial-melphalan-eye-salvage", "enucleation-Rb-primary", "doxorubicin-ifosfamide-secondary-STS", "NO-CDK4-6i-RB1-null-resistant", "AVOID-high-dose-external-beam-RT"],
    "FH":      ["bevacizumab-erlotinib-type2-pRCC-BEST-combination", "cabozantinib-MET-VEGFR2-emerging", "lapatinib-bevacizumab-trials", "hysterectomy-uterine-fibroids-30-35yr", "surgery-curative-localised-RCC"],
}

SURVEILLANCE_PROTOCOLS = {
    "TP53":    ["WBMRI-annually-Toronto-chest-abd-pelvis-limbs", "annual-brain-MRI", "annual-breast-MRI-from-20yr", "US-abdomen-q6M-adrenal", "colonoscopy-from-25yr-q2yr", "cascade-FDR-50pct"],
    "NF1":     ["annual-plexiform-NF-exam-targeted-MRI", "PET-FDG-MPNST-SUV-3-5-threshold", "annual-breast-MRI-from-30yr", "annual-ophthalmology-to-8yr-OPG", "AVOID-radiation", "cascade-FDR-50pct"],
    "APC":     ["annual-colonoscopy-post-colectomy", "annual-abdominopelvic-MRI-desmoid", "upper-GI-endoscopy-q1-3yr", "prophylactic-colectomy-before-25yr", "cascade-FDR-50pct"],
    "DICER1":  ["CT-chest-q6M-to-2yr-then-annual-to-6yr-PPB", "annual-thyroid-US", "annual-abdominal-US-cystic-nephroma", "gynaecological-exam-from-puberty", "cascade-FDR-siblings-CT-chest-under-8yr"],
    "SMARCB1": ["MRI-brain-full-spine-q6M-to-5yr", "US-abdomen-q6M-to-5yr-renal-MRT", "annual-MRI-from-5yr", "both-parents-molecular-test", "cascade-FDR-50pct"],
    "BRCA2":   ["annual-breast-MRI-mammogram-from-25yr", "transvaginal-US-CA-125-from-30yr", "PBSO-by-40-45yr", "annual-abdominopelvic-MRI", "PSA-DRE-annual-from-40yr-male", "cascade-FDR-50pct"],
    "RB1":     ["EUA-q3-4W-to-3yr-q3-6M-to-5yr", "annual-MRI-orbits-brain-pinealoblastoma", "annual-MRI-prior-RT-fields-secondary-STS", "AVOID-CT-ionising-radiation", "offspring-testing-day-1-life"],
    "FH":      ["annual-abdominopelvic-MRI-from-10yr-aggressive", "annual-gynaecological-US-MRI", "FH-IHC-2SC-IHC-renal-biopsy", "cascade-FDR-50pct"],
}

# -------------------------------------------------------------------
# Age distributions and clinical parameters per gene
# -------------------------------------------------------------------
GENE_PARAMS = {
    "TP53":    {"ages": (8, 45),  "cr_base": 48, "gtr_base": 62, "targeted_base": 38, "radiation_base": 2,  "relapse_base": 42},
    "NF1":     {"ages": (22, 55), "cr_base": 32, "gtr_base": 55, "targeted_base": 52, "radiation_base": 4,  "relapse_base": 55},
    "APC":     {"ages": (28, 55), "cr_base": 60, "gtr_base": 45, "targeted_base": 55, "radiation_base": 18, "relapse_base": 35},
    "DICER1":  {"ages": (1, 18),  "cr_base": 55, "gtr_base": 70, "targeted_base": 42, "radiation_base": 5,  "relapse_base": 30},
    "SMARCB1": {"ages": (1, 40),  "cr_base": 22, "gtr_base": 58, "targeted_base": 65, "radiation_base": 28, "relapse_base": 65},
    "BRCA2":   {"ages": (30, 65), "cr_base": 42, "gtr_base": 52, "targeted_base": 45, "radiation_base": 22, "relapse_base": 48},
    "RB1":     {"ages": (1, 40),  "cr_base": 55, "gtr_base": 68, "targeted_base": 35, "radiation_base": 15, "relapse_base": 38},
    "FH":      {"ages": (28, 60), "cr_base": 45, "gtr_base": 72, "targeted_base": 48, "radiation_base": 25, "relapse_base": 32},
}

RESECTION_OPTIONS = {
    "TP53":    ["wide-margin", "marginal", "intralesional", "unresectable"],
    "NF1":     ["wide-margin", "marginal", "incomplete-plexiform", "unresectable"],
    "APC":     ["watchful-waiting", "NSAIDs-only", "partial-excision", "wide-excision", "multi-visceral"],
    "DICER1":  ["lobectomy-pneumonectomy", "fertility-sparing-cervical", "salpingo-oophorectomy", "complete-resection"],
    "SMARCB1": ["gross-total-resection", "subtotal-resection", "biopsy-only", "enucleation-MRT-renal"],
    "BRCA2":   ["total-hysterectomy-BSO", "wide-excision", "PBSO-prophylactic", "unresectable"],
    "RB1":     ["enucleation-unilateral", "bilateral-enucleation", "globe-salvage", "secondary-STS-wide-excision"],
    "FH":      ["total-hysterectomy-fibroids", "partial-nephrectomy", "radical-nephrectomy", "metastasectomy"],
}


def _weighted_choice(rng, choices, weights=None):
    if weights is None:
        return rng.choice(choices)
    total = sum(weights)
    r = rng.random() * total
    upto = 0
    for c, w in zip(choices, weights):
        upto += w
        if r < upto:
            return c
    return choices[-1]


def _json_safe(obj):
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 2)
    return obj


def _generate_patients(gene_def: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = gene_def["gene"]
    params = GENE_PARAMS[gene]
    tumors = TUMOR_TYPES[gene]
    variants = list(PATHOGENIC_VARIANTS[gene].keys())
    variant_weights = list(PATHOGENIC_VARIANTS[gene].values())
    treatments = TREATMENT_PROTOCOLS[gene]
    resections = RESECTION_OPTIONS[gene]

    patients = []
    for i in range(n):
        age_lo, age_hi = params["ages"]
        age_dx = rng.randint(age_lo, age_hi)
        tumor = rng.choice(tumors)
        variant = _weighted_choice(rng, variants, variant_weights)
        resection = rng.choice(resections)
        treatment = rng.choice(treatments)

        cr_prob = (params["cr_base"] + rng.randint(-10, 10)) / 100.0
        gtr_prob = (params["gtr_base"] + rng.randint(-8, 8)) / 100.0
        targeted_prob = (params["targeted_base"] + rng.randint(-8, 8)) / 100.0
        radiation_prob = (params["radiation_base"] + rng.randint(-5, 5)) / 100.0
        relapse_prob = (params["relapse_base"] + rng.randint(-10, 10)) / 100.0

        # TP53: NO radiation ever (absolute CI)
        if gene == "TP53":
            radiation_prob = 0.0

        response_roll = rng.random()
        if response_roll < cr_prob:
            response = "CR"
        elif response_roll < cr_prob + 0.22:
            response = "PR"
        elif response_roll < cr_prob + 0.22 + 0.18:
            response = "SD"
        else:
            response = "PD"

        patients.append({
            "gene": gene,
            "patient_id": f"{gene}-{seed + i}",
            "age_dx": age_dx,
            "tumor_type": tumor,
            "resection": resection,
            "treatment": treatment,
            "response": response,
            "cr": response == "CR",
            "gtr": rng.random() < gtr_prob,
            "targeted_therapy": rng.random() < targeted_prob,
            "radiation_received": rng.random() < radiation_prob,
            "relapse": rng.random() < relapse_prob,
            "variant": variant,
        })
    return patients


def generate_overview() -> dict:
    all_genes = []
    gene_summaries = []
    total_tumor_counts: dict = {}
    total_cr = 0
    total_gtr = 0
    total_targeted = 0
    total_radiation = 0
    total_relapse = 0
    all_pts = []

    for i, gdef in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        pts = _generate_patients(gdef, seed)
        all_pts.extend(pts)
        cr_n = sum(1 for p in pts if p["cr"])
        gtr_n = sum(1 for p in pts if p.get("gtr"))
        targeted_n = sum(1 for p in pts if p["targeted_therapy"])
        radiation_n = sum(1 for p in pts if p["radiation_received"])
        relapse_n = sum(1 for p in pts if p["relapse"])

        for p in pts:
            tt = p["tumor_type"]
            total_tumor_counts[tt] = total_tumor_counts.get(tt, 0) + 1

        gene_summaries.append({
            "gene": gdef["gene"],
            "n_patients": len(pts),
            "cr_pct": round(cr_n / len(pts) * 100, 1),
            "gtr_pct": round(gtr_n / len(pts) * 100, 1),
            "targeted_pct": round(targeted_n / len(pts) * 100, 1),
            "radiation_pct": round(radiation_n / len(pts) * 100, 1),
            "relapse_pct": round(relapse_n / len(pts) * 100, 1),
        })
        all_genes.append(gdef["gene"])
        total_cr += cr_n
        total_gtr += gtr_n
        total_targeted += targeted_n
        total_radiation += radiation_n
        total_relapse += relapse_n

    total = len(all_pts)
    mean_age = round(sum(p["age_dx"] for p in all_pts) / total, 1) if total else 0

    return _json_safe({
        "atlas": "Hereditary-Soft-Tissue-Sarcoma-Desmoid-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference -- TP53 · NF1 · APC · DICER1 · SMARCB1 · BRCA2 · RB1 · FH",
        "genes": all_genes,
        "total_patients": total,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_age_dx": mean_age,
        "cr_pct": round(total_cr / total * 100, 1) if total else 0,
        "gtr_pct": round(total_gtr / total * 100, 1) if total else 0,
        "targeted_pct": round(total_targeted / total * 100, 1) if total else 0,
        "radiation_pct": round(total_radiation / total * 100, 1) if total else 0,
        "relapse_pct": round(total_relapse / total * 100, 1) if total else 0,
        "gene_summaries": gene_summaries,
        "top_tumor_types": dict(sorted(total_tumor_counts.items(), key=lambda x: -x[1])[:10]),
        "clinical_pearls": [
            "TP53 (LFS): AVOID RADIATION ABSOLUTELY — radiation-field secondary STS lethal; WBMRI Toronto annual, NOT CT/PET.",
            "NF1: MPNST arises from plexiform NF — rapid painful enlargement is MALIGNANT TRANSFORMATION SIGNAL; PET-FDG SUV>3.5 threshold.",
            "APC/Gardner desmoid: NEVER perform upfront surgery for mesenteric desmoid — paradoxical growth documented post-resection; nirogacestat FDA2023.",
            "DICER1: PPB type I (cystic lung) in child <2yr = PATHOGNOMONIC — test ALL siblings CT chest; AVOID radiation children.",
            "SMARCB1: INI1 IHC nuclear loss on any sarcoma in child/young adult = SMARCB1 germline testing MANDATORY; tazemetostat EZH2i FDA2020 epithelioid sarcoma.",
            "BRCA2: Leiomyosarcoma in BRCA2 carrier = HRD — cisplatin sensitivity; biallelic FA-D1 = RMS/Wilms/medulloblastoma PATHOGNOMONIC most severe FA.",
            "RB1: CDK4/6 inhibitors (palbociclib) are RESISTANT in RB1-null STS — prescribing trap in multi-specialty oncology.",
            "FH (HLRCC): FH IHC + 2SC IHC on ANY renal mass in young patient with uterine fibroids/cutaneous leiomyomas = PATHOGNOMONIC HLRCC; bevacizumab+erlotinib best for type 2 pRCC.",
        ],
        "key_management_rules": [
            "TP53 LFS: RADIATION ABSOLUTE CONTRAINDICATION — secondary field sarcoma risk; WBMRI annually.",
            "NF1: AVOID radiation — secondary MPNST risk at field; selumetinib MEKi FDA2020 for symptomatic inoperable plexiform NF.",
            "APC desmoid: SURGERY AVOID mesenteric desmoid — paradoxical growth; nirogacestat FDA2023 + sorafenib RCT-proven.",
            "DICER1: AVOID radiation children — developing organ sensitivity; CT chest screening siblings <8yr (PPB family risk).",
            "SMARCB1: INI1 IHC loss = tazemetostat eligibility (epithelioid sarcoma); ICE + HSCT for MRT/ATRT.",
            "BRCA2: Cisplatin/olaparib HRD sensitivity in LMS; biallelic FA-D1 = most severe paediatric cancer syndrome.",
            "RB1: CDK4/6i RESISTANT RB1-null; AVOID high-dose external beam RT (secondary LMS/osteosarcoma); offspring testing day 1 of life.",
            "FH: Annual abdominal/pelvic MRI from AGE 10yr — type 2 papillary RCC aggressive early metastasis; bevacizumab+erlotinib.",
        ],
    })


def generate_breakdown() -> dict:
    breakdown = []

    for i, gdef in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        pts = _generate_patients(gdef, seed)
        gene = gdef["gene"]

        cr_n = sum(1 for p in pts if p["cr"])
        gtr_n = sum(1 for p in pts if p.get("gtr"))
        targeted_n = sum(1 for p in pts if p["targeted_therapy"])
        radiation_n = sum(1 for p in pts if p["radiation_received"])
        relapse_n = sum(1 for p in pts if p["relapse"])

        tumor_counts: dict = {}
        variant_counts: dict = {}
        tx_counts: dict = {}
        resection_counts: dict = {}

        for p in pts:
            tumor_counts[p["tumor_type"]] = tumor_counts.get(p["tumor_type"], 0) + 1
            variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1
            tx_counts[p["treatment"]] = tx_counts.get(p["treatment"], 0) + 1
            resection_counts[p["resection"]] = resection_counts.get(p["resection"], 0) + 1

        breakdown.append({
            "gene": gene,
            "locus": gdef["locus"],
            "n_patients": len(pts),
            "mean_age_dx": round(sum(p["age_dx"] for p in pts) / len(pts), 1) if pts else 0,
            "cr_pct": round(cr_n / len(pts) * 100, 1) if pts else 0,
            "cr_n": cr_n,
            "gtr_pct": round(gtr_n / len(pts) * 100, 1) if pts else 0,
            "gtr_n": gtr_n,
            "targeted_pct": round(targeted_n / len(pts) * 100, 1) if pts else 0,
            "targeted_n": targeted_n,
            "radiation_pct": round(radiation_n / len(pts) * 100, 1) if pts else 0,
            "radiation_n": radiation_n,
            "relapse_pct": round(relapse_n / len(pts) * 100, 1) if pts else 0,
            "relapse_n": relapse_n,
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "key_distinctions": gdef["key_distinctions"],
            "top_tumor_types": dict(sorted(tumor_counts.items(), key=lambda x: -x[1])[:5]),
            "top_variants": dict(sorted(variant_counts.items(), key=lambda x: -x[1])[:4]),
            "top_treatments": dict(sorted(tx_counts.items(), key=lambda x: -x[1])[:4]),
            "resection_distribution": resection_counts,
            "patients": pts,
        })

    return _json_safe({
        "atlas": "Hereditary-Soft-Tissue-Sarcoma-Desmoid-Predisposition-Atlas",
        "total_genes": len(breakdown),
        "breakdown": breakdown,
    })


def generate_definitions() -> dict:
    definitions = {}
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        definitions[g] = {
            "gene": g,
            "locus": gdef["locus"],
            "full_protein_detail": gdef["protein_size"],
            "inheritance": gdef["inheritance"],
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
            "key_distinctions": gdef["key_distinctions"],
            "pathogenic_variants": PATHOGENIC_VARIANTS[g],
            "tumor_types": TUMOR_TYPES[g],
            "treatment_protocols": TREATMENT_PROTOCOLS[g],
            "surveillance_protocols": SURVEILLANCE_PROTOCOLS[g],
        }

    return _json_safe({
        "atlas": "Hereditary-Soft-Tissue-Sarcoma-Desmoid-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "TP53_AVOID_RADIATION_STS": (
                "AVOID radiation ABSOLUTELY in LFS (TP53 germline) -- radiation-field secondary STS lethal; "
                "WBMRI Toronto Protocol annually (NOT CT, NOT PET/CT -- ionising radiation contraindicated); "
                "R337H founder Brazil: 1/300 carrier frequency southern Brazil; "
                "secondary sarcoma post-RT in LFS: latency 5-20yr; prognosis very poor"
            ),
            "NF1_MPNST_SELUMETINIB": (
                "NF1-MPNST 8-13% HIGHEST sarcoma risk PATHOGNOMONIC -- arises from plexiform NF; "
                "rapid painful plexiform NF growth = malignant transformation ALERT; "
                "PET-FDG SUV >3.5 = MPNST transformation signal; wide surgical margin >1cm required; "
                "AVOID radiation NF1 secondary MPNST risk; selumetinib MEKi FDA2020 symptomatic inoperable pNF"
            ),
            "APC_DESMOID_SURGERY_AVOID": (
                "SURGERY AVOID upfront mesenteric desmoid -- paradoxical growth post-resection documented; "
                "nirogacestat (gamma-secretase inhibitor, Notch) FDA2023 POSITIVE RCT desmoid; "
                "sorafenib RCT positive 2019 mesenteric desmoid; sulindac/celecoxib first-line; "
                "codon 1444+ MCR APC = HIGHEST desmoid risk genotype"
            ),
            "DICER1_PPB_AVOID_RT_CHILDREN": (
                "PPB type I (cystic lung, <2yr) PATHOGNOMONIC DICER1 -- test ALL siblings CT chest (PPB family); "
                "AVOID radiation in DICER1 children (developing organs + radiation sensitivity); "
                "DICER1 somatic RNase IIIb hotspot (E1705/D1709/E1813) = pathognomonic second hit; "
                "cervical ERMS PATHOGNOMONIC DICER1 -- fertility-sparing surgery if feasible adolescent"
            ),
            "SMARCB1_INI1_IHC_TAZEMETOSTAT": (
                "INI1 IHC nuclear LOSS = SMARCB1 inactivation PATHOGNOMONIC (MRT/ATRT/epithelioid sarcoma); "
                "tazemetostat 800mg BID FDA2020 locally advanced/metastatic epithelioid sarcoma + SMARCB1 loss; "
                "MRT/ATRT under 3yr with INI1 loss = germline testing MANDATORY; "
                "HSCT autologous consolidation after ICE induction MRT/ATRT"
            ),
            "RB1_CDK46I_RESISTANT": (
                "CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib) RESISTANT in RB1-null STS -- target absent; "
                "CDK4-amplified STS (DDL/WDL liposarcoma) has INTACT RB1 -- CDK4/6i ACTIVE (key distinction); "
                "AVOID high-dose external beam RT hereditary RB1 -- secondary LMS/osteosarcoma risk; "
                "offspring testing day 1 of life (bilateral Rb 95% hereditary)"
            ),
            "FH_HLRCC_SURVEILLANCE_AGE10": (
                "Annual abdominal/pelvic MRI from AGE 10yr in FH germline -- type 2 pRCC early metastasis aggressive; "
                "2SC IHC (succino-cysteine) PATHOGNOMONIC FH-deficiency on any tissue; "
                "FH IHC nuclear loss PATHOGNOMONIC; "
                "bevacizumab + erlotinib: best combination FH-deficient type 2 papillary RCC (ORR 64% Srinivasan 2014)"
            ),
            "BRCA2_HRD_LMS": (
                "BRCA2 germline leiomyosarcoma HRD-sensitive: cisplatin/carboplatin preferred first-line regimen; "
                "olaparib PARP inhibitor: off-label BRCA2 STS; SARC045 trial ongoing; "
                "FA-D1 biallelic BRCA2: Wilms 50% + RMS 30% + medulloblastoma = MOST SEVERE Fanconi; "
                "PBSO by 40-45yr female BRCA2 (breast 47-69%, ovarian 11-17% lifetime risk)"
            ),
        },
        "cascade_testing_rule": (
            "TP53/NF1/APC/DICER1/SMARCB1/BRCA2/RB1/FH: all AD -- cascade to all first-degree relatives (50% risk per child). "
            "SMARCB1 MRT/ATRT: ~50% de novo -- test both parents before assuming inherited. "
            "RB1: offspring testing on day 1 of life (bilateral Rb = hereditary 95%+). "
            "DICER1: ALL siblings <8yr require CT chest (PPB risk in family). "
            "FH: biallelic FH = fumarase deficiency (severe neurometabolic -- DIFFERENT from HLRCC monoallelic). "
            "APC I1307K: Ashkenazi Jewish founder allele -- population-level cascade in AJ families. "
            "TP53 R337H: southern Brazil population screening (1/300 carrier)."
        ),
    })


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2)[:3000])
    print("\n--- breakdown (first gene) ---")
    bd = generate_breakdown()
    print(json.dumps(bd["breakdown"][0], indent=2)[:2000])
