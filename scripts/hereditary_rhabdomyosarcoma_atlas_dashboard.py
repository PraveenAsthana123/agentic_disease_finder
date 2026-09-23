#!/usr/bin/env python3
"""Hereditary-Rhabdomyosarcoma-Predisposition-Atlas -- Complete 8-Gene Reference
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         RMS 10-15% LFS (most ERMS/spindle cell); AVOID RADIATION ABSOLUTELY;
         WBMRI Toronto annual; dominant negative GOF hotspots (R175H, R248W, R248Q);
         seed SEED_BASE+0) .
NF1     (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis type 1;
         RMS 2-3x elevated (embryonal RMS orbital/paratesticular);
         MPNST 8-13% PATHOGNOMONIC (plexiform NF transformation);
         AVOID RADIATION (secondary MPNST); Selumetinib FDA 2020;
         seed SEED_BASE+1) .
DICER1  (DICER1 ribonuclease III; 1922aa; 14q32.13; AD LOF;
         DICER1 syndrome;
         cervical ERMS PATHOGNOMONIC; bladder/vaginal ERMS PATHOGNOMONIC;
         embryonal uterine RMS; fertility-sparing surgery preferred;
         AVOID radiation in children;
         seed SEED_BASE+2) .
RB1     (Retinoblastoma 1; 928aa; 13q14.2; AD LOF;
         hereditary retinoblastoma;
         secondary RMS 10-15x elevated post-RT (orbital/radiation field);
         CDK4/6 inhibitors inactive (pRb substrate lost);
         AVOID RADIATION; bilateral retinoblastoma PATHOGNOMONIC germline;
         seed SEED_BASE+3) .
HRAS    (Harvey Ras oncogene; 189aa; 11p15.5; AD GOF;
         Costello syndrome;
         embryonal RMS 15-20% lifetime PATHOGNOMONIC Costello;
         G12S most common germline HRAS; cardiac HCM LIFE-THREATENING;
         annual Echo MANDATORY; papillomas PATHOGNOMONIC Costello;
         seed SEED_BASE+4) .
BRCA2   (Breast cancer gene 2; 3418aa; 13q12.3; Biallelic AR LOF FA-D1 / AD LOF HBOC;
         FA complementation group D1;
         embryonal RMS PATHOGNOMONIC FA-D1 (5-10% biallelic);
         AVOID alkylating agents (cyclophosphamide BMF); cisplatin HRD preferred;
         SIBLING DONOR EXCLUSION MANDATORY;
         seed SEED_BASE+5) .
PTPN11  (Protein tyrosine phosphatase non-receptor type 11; 593aa; 12q24.13; AD GOF;
         Noonan syndrome;
         RMS 1-2% Noonan; JMML 200-500x elevated;
         pulmonary valve stenosis 50-60% PATHOGNOMONIC Noonan;
         HCM 20-30%; LEO (lentigines) LEOPARD phenotype;
         seed SEED_BASE+6) .
SMARCB1 (SWI/SNF related matrix-associated actin-dependent regulator of chromatin; 385aa; 22q11.23; AD LOF;
         rhabdoid tumour predisposition syndrome 2 (RTPS2);
         MRT/ATRT under 3yr PATHOGNOMONIC; epithelioid RMS INI1-loss;
         INI1 IHC nuclear loss PATHOGNOMONIC;
         Tazemetostat EZH2i FDA 2020; HSCT consolidation;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3278-3285)
"""
import random

SEED_BASE = 3278

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-RMS-10-15pct-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour suppressor protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; transcription factor; tetramer in active form; "
            "  N-terminal transactivation domains (TAD1 aa 1-40, TAD2 aa 40-61); "
            "  Proline-rich region (aa 63-97); "
            "  DNA-binding domain (DBD, aa 102-292): most mutations cluster here; "
            "  Tetramerisation domain (aa 323-356): oligomerisation; "
            "  Regulatory C-terminal domain (aa 363-393): post-translational modifications; "
            "  R175H, R248W, R248Q, R273H, R273C, R249S: hotspot dominant-negative GOF mutations; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; ~50% germline TP53 LOF; rest of spectrum LOF; "
            "  Classic LFS: sarcoma <45yr + brain tumour + ACC + breast cancer <45yr; "
            "  Chompret criteria 2015: earliest-onset cancer <46yr + family sarcoma history; "
            "  Childhood ACC (adrenocortical carcinoma): TP53 germline 50-70% paediatric PATHOGNOMONIC; "
            "RHABDOMYOSARCOMA IN LFS / TP53: "
            "  RMS: 10-15% of LFS tumour spectrum -- predominantly embryonal (ERMS) subtype; "
            "  Spindle cell/sclerosing RMS: overrepresented in LFS germline; "
            "  TP53 somatic mutations in ARMS relapse: 5-10% alveolar RMS relapse; "
            "  LFS RMS: orbital, paratesticular, head/neck, extremity predominant; "
            "  TP53 somatic alteration at progression: vincristine/actinomycin-D/cyclophosphamide (VAC); "
            "AVOID RADIATION ABSOLUTELY (LFS): "
            "  Germline TP53 LFS: AVOID RADIATION ABSOLUTELY; "
            "  RMS radiotherapy (involved-field RT): OMIT or minimise if TP53 germline; "
            "  Radiation -> second primary malignancy in radiation field (sarcoma, brain tumour); "
            "  Low-dose RT for orbital/head-neck RMS: highest risk in TP53 -- omit if feasible; "
            "  Proton therapy consideration if RT unavoidable (lower exit dose); "
            "WBMRI TORONTO ANNUAL: "
            "  Whole-body MRI Toronto Protocol annually -- NOT PET-CT/CT (radiation); "
            "  Brain MRI + abdominal/pelvis/chest MRI; "
            "  Annual WBMRI from germline TP53 diagnosis (lifelong); "
        ),
        "inheritance": "Autosomal dominant LOF; 50% de novo; dominant negative GOF hotspot variants (R175H, R248W, R248Q) worsen phenotype via tetramer poisoning; near-complete penetrance (>90% lifetime cancer risk)",
        "cancer_risk": "LFS: sarcoma 30-50% dominant (including RMS 10-15%); brain tumour 15-20%; ACC 3-5% (paediatric ACC 50-70% TP53 PATHOGNOMONIC); breast <45yr 25-35%; ERMS and spindle cell RMS over-represented",
        "pathognomonic": "Paediatric adrenocortical carcinoma PATHOGNOMONIC LFS (50-70% paediatric ACC = TP53 germline); R337H 1/300 South Brazilian founder mutation; LFS RMS predominantly ERMS/spindle cell subtype",
        "surveillance_key": "WBMRI Toronto annually (NOT CT/PET); AVOID RADIATION ABSOLUTELY; omit RMS RT if TP53 germline; proton if unavoidable; annual rapid brain MRI + abdominal MRI; breast MRI from age 20yr; CASCADE 50% risk",
        "key_distinctions": [
            "LFS-AVOID-RADIATION-ABSOLUTELY-RMS",
            "WBMRI-TORONTO-ANNUALLY-NOT-CT",
            "PAEDIATRIC-ACC-50-70PCT-TP53-PATHOGNOMONIC",
            "RMS-10-15PCT-LFS-ERMS-SPINDLE-PREDOMINANT",
            "OMIT-RMS-RT-IF-TP53-GERMLINE",
            "R337H-SOUTH-BRAZIL-1IN300-FOUNDER",
        ],
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-NF1-Cafe-au-Lait-PATHOGNOMONIC-"
            "MPNST-8-13pct-PATHOGNOMONIC-Selumetinib-FDA2020-AVOID-Radiation-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes neurofibromin (RAS GTPase-activating protein): "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; one of largest tumour suppressor proteins; "
            "  GAP-related domain (GRD, aa 1189-1551): RAS-GAP activity; catalyses RAS-GTP -> RAS-GDP; "
            "  Sec14 domain: lipid binding, membrane targeting; "
            "  NF1 LOF -> constitutive RAS-MAPK activation -> cell proliferation; "
            "  NF1 second hit (somatic): biallelic LOF required for most NF1 tumours (Knudson 2-hit); "
            "NEUROFIBROMATOSIS TYPE 1 (NF1): "
            "  OMIM 162200; 1 in 3000 births; 50% de novo; "
            "  Cafe-au-lait macules ≥6 (≥5mm prepubertal, ≥15mm postpubertal) PATHOGNOMONIC; "
            "  Plexiform NF (congenital): MPNST transformation 8-13% PATHOGNOMONIC; "
            "  Lisch nodules (iris hamartomas): PATHOGNOMONIC adults (>90%); "
            "  Optic pathway glioma 15-20%; GIST elevated (NF1-GIST wild-type KIT/PDGFRA); "
            "RHABDOMYOSARCOMA IN NF1: "
            "  RMS: 2-3x elevated in NF1 (general population baseline 0.5-1%); "
            "  NF1 RMS: embryonal subtype predominant; orbital + paratesticular + head/neck; "
            "  NF1 RMS onset: childhood (median age 4-6yr); "
            "  MPNST vs RMS in NF1: MPNST = plexiform NF transformation (spindle cell, high-grade); "
            "    RMS = distinct from MPNST lineage (skeletal muscle differentiation markers); "
            "    NF1-GIST: KIT/PDGFRA wild-type; imatinib NOT effective; surgery primary; "
            "SELUMETINIB (MEK INHIBITOR -- FDA 2020): "
            "  Selumetinib FDA 2020 for symptomatic inoperable plexiform NF in NF1 ≥2yr; "
            "  MEK1/2 inhibitor downstream of RAS; plexiform NF 70-80% response rate; "
            "  NOT approved for RMS treatment (no NF1-RMS indication); "
            "  MPNST: selumetinib Phase II data; limited single-agent efficacy in MPNST; "
            "AVOID RADIATION (NF1): "
            "  Radiation -> secondary MPNST in NF1 (RAS pathway + radiation mutagenesis); "
            "  RMS RT: minimise radiation dose + field in NF1 (secondary MPNST/sarcoma risk); "
            "  Proton therapy preferred over photon RT in NF1 RMS when radiation unavoidable; "
        ),
        "inheritance": "Autosomal dominant LOF; 50% de novo; full penetrance (>99%) cafe-au-lait macules; variable expressivity; biallelic NF1 LOF (somatic second hit) for tumours",
        "cancer_risk": "MPNST 8-13% lifetime PATHOGNOMONIC (plexiform NF transformation); RMS 2-3x elevated; optic glioma 15-20%; JMML 200x elevated children; GIST (wild-type KIT) elevated; breast cancer 2x elevated",
        "pathognomonic": "Cafe-au-lait macules ≥6 ≥15mm PATHOGNOMONIC NF1; Lisch nodules PATHOGNOMONIC adults; plexiform NF -> MPNST 8-13% PATHOGNOMONIC; JMML 200x elevated NF1 children PATHOGNOMONIC; NF1-GIST wild-type KIT",
        "surveillance_key": "Annual full skin + ophthalmology exam; MRI brain annually childhood (OPG); selumetinib symptomatic inoperable PN; avoid radiation (secondary MPNST); NF1 RMS: minimise RT field; proton preferred over photon; whole-body MRI q2yr adults (MPNST)",
        "key_distinctions": [
            "CAFE-AU-LAIT-MACULES-GT6-PATHOGNOMONIC-NF1",
            "MPNST-8-13PCT-LIFETIME-PATHOGNOMONIC",
            "SELUMETINIB-FDA2020-PLEXIFORM-NF",
            "AVOID-RADIATION-SECONDARY-MPNST",
            "NF1-RMS-ERMS-ORBITAL-PARATESTICULAR",
            "NF1-GIST-WILD-TYPE-KIT-IMATINIB-NOT-EFFECTIVE",
        ],
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-219kDa-RNase-III-miRNA-Processor-Cervical-ERMS-PATHOGNOMONIC-"
            "Bladder-ERMS-PATHOGNOMONIC-Fertility-Sparing-AVOID-Radiation-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 219 kDa / 14q32.13 DICER1 encodes DICER1 (RNase III endoribonuclease / miRNA processor): "
            "STRUCTURE: "
            "  1922 aa / 219 kDa; RNase III family endonuclease; "
            "  Helicase domain (aa 1-605); DUF283 domain (aa 608-699); "
            "  PAZ domain (aa 768-875): dsRNA 3' end binding; "
            "  RNase IIIa domain (aa 1000-1100): cleaves miRNA* strand; "
            "  RNase IIIb domain (aa 1218-1380): cleaves mature miRNA strand; "
            "    RNase IIIb hotspot mutations (E1705K, D1709N, E1813G): somatic second hit PATHOGNOMONIC; "
            "  DICER1 LOF -> impaired miRNA processing -> derepression of developmental oncogenes; "
            "DICER1 SYNDROME AND RHABDOMYOSARCOMA: "
            "  OMIM 601200; pleiotropic tumour predisposition syndrome; "
            "  Cervical embryonal RMS (ERMS): PATHOGNOMONIC DICER1 in adolescent females; "
            "    Vaginal mass / cervical mass in teenage girl -> DICER1 germline testing MANDATORY; "
            "    Fertility-sparing surgery preferred (trachelectomy/polypectomy) where oncologically feasible; "
            "  Bladder/urinary tract ERMS: PATHOGNOMONIC DICER1 (paediatric/young adult); "
            "  Uterine/corpus ERMS: PATHOGNOMONIC DICER1 (young adult); "
            "  Other DICER1 tumours: PPB (Type I cystic <2yr) PATHOGNOMONIC; cystic nephroma PATHOGNOMONIC; "
            "    SLCT (Sertoli-Leydig cell tumour ovary) PATHOGNOMONIC; MNG (multinodular goitre) 75% females; "
            "  RMS subtype in DICER1: ERMS (PAX3/FOXO1 and PAX7/FOXO1 fusion-NEGATIVE); "
            "    Botryoid variant ERMS: vaginal polypoid exophytic PATHOGNOMONIC; "
            "SECOND HIT DICER1 (RNase IIIb): "
            "  Germline DICER1 LOF (first hit) + somatic RNase IIIb domain hotspot (second hit); "
            "  E1705K, D1709N, D1710N, E1813G: canonical somatic second hits in all DICER1 tumours; "
            "  Hotspot testing on tumour tissue can confirm DICER1 pathway in RMS; "
            "AVOID RADIATION IN CHILDREN (DICER1): "
            "  Developing organs at highest risk from RT in DICER1 patients; "
            "  RMS RT: avoid pelvic/vaginal RT where possible; fertility preservation critical; "
            "  Low-dose VAC (vincristine/actinomycin-D/cyclophosphamide) preferred; "
        ),
        "inheritance": "Autosomal dominant LOF; germline DICER1 LOF (first hit) + somatic RNase IIIb hotspot (second hit) required; de novo mutations ~10%; penetrance variable by tumour type",
        "cancer_risk": "Cervical ERMS PATHOGNOMONIC; bladder/vaginal ERMS PATHOGNOMONIC; uterine ERMS PATHOGNOMONIC; PPB PATHOGNOMONIC; cystic nephroma PATHOGNOMONIC; SLCT PATHOGNOMONIC; MNG 75%; RMS subset predominantly ERMS botryoid",
        "pathognomonic": "Cervical ERMS (teenage female) PATHOGNOMONIC DICER1; bladder ERMS (young child) PATHOGNOMONIC DICER1; botryoid vaginal ERMS PATHOGNOMONIC DICER1; RNase IIIb somatic hotspot (E1705/D1709/E1813) PATHOGNOMONIC second hit",
        "surveillance_key": "Pelvic US annually adolescent females (cervical/vaginal ERMS, SLCT); annual thyroid US from age 8yr (MNG); chest CT siblings <8yr (PPB); AVOID radiation in children; nephrology follow-up cystic nephroma; fertility-sparing surgery for cervical ERMS",
        "key_distinctions": [
            "CERVICAL-ERMS-PATHOGNOMONIC-DICER1",
            "BLADDER-ERMS-PATHOGNOMONIC-DICER1",
            "FERTILITY-SPARING-SURGERY-PREFERRED",
            "AVOID-RADIATION-CHILDREN-DICER1",
            "BOTRYOID-VAGINAL-ERMS-PATHOGNOMONIC",
            "RNASE-IIIB-HOTSPOT-SECOND-HIT-PATHOGNOMONIC",
        ],
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "pRb-105kDa-E2F-Regulator-Retinoblastoma-Secondary-RMS-10-15x-Post-RT-"
            "CDK4-6i-INACTIVE-AVOID-RADIATION-Bilateral-PATHOGNOMONIC-OMIM-614041"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 105 kDa / 13q14.2 RB1 encodes retinoblastoma protein (pRb; tumour suppressor): "
            "STRUCTURE: "
            "  928 aa / 105 kDa; pocket protein family; "
            "  Small pocket domain (aa 379-572): E2F binding pocket A; "
            "  Large pocket domain (aa 379-787): pocket A + B + C-terminal; "
            "  Pocket B (aa 640-771): E2F transcription factor binding; "
            "  C-terminal domain (aa 788-928): regulatory; LXCXE motif binding; "
            "  pRb LOF -> E2F release -> uncontrolled S-phase entry -> cell cycle dysregulation; "
            "HEREDITARY RETINOBLASTOMA (RB): "
            "  OMIM 180200; bilateral retinoblastoma PATHOGNOMONIC germline RB1 (>95%); "
            "  Unilateral early onset (<12mo) suspicious for germline; "
            "  Trilateral retinoblastoma (pinealoblastoma + bilateral RB): PATHOGNOMONIC germline; "
            "  Penetrance: ~90% but germline RB1 = ~45% of all retinoblastoma cases; "
            "SECONDARY RMS POST-RADIOTHERAPY (RB1): "
            "  Secondary RMS 10-15x elevated in RB1 germline post orbital RT; "
            "  Radiation field predominant: orbital/periorbital RMS most common; "
            "  Latency: 10-20yr after RT (range 5-30yr); "
            "  RMS subtype: predominantly spindle cell/sclerosing or ERMS in radiation field; "
            "  Mechanism: RB1 LOF + radiation-induced mutagenesis -> sarcoma transformation; "
            "  AVOID ORBITAL RADIATION in RB1 germline if alternative (chemoreduction, cryotherapy); "
            "CDK4/6 INHIBITOR RESISTANCE (RB1 NULL): "
            "  CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib) require functional pRb substrate; "
            "  RB1-null tumours: CDK4/6i INACTIVE (no pRb to hyperphosphorylate); "
            "  Secondary RMS in RB1: CDK4/6i resistance anticipated; "
            "  PI3K/AKT pathway inhibitors: investigational for RB1-null RMS; "
        ),
        "inheritance": "Autosomal dominant LOF; ~45% de novo; bilateral RB PATHOGNOMONIC germline (>95%); unilateral RB: 10-15% germline; penetrance ~90%; somatic second hit required for tumour (Knudson 2-hit)",
        "cancer_risk": "Retinoblastoma (bilateral PATHOGNOMONIC); secondary RMS 10-15x elevated post-RT; osteosarcoma 500x post-RT; melanoma elevated; lung adenocarcinoma elevated; overall secondary cancer 25-35% at 30yr",
        "pathognomonic": "Bilateral retinoblastoma PATHOGNOMONIC germline RB1; trilateral retinoblastoma (+ pinealoblastoma) PATHOGNOMONIC; secondary RMS/osteosarcoma in radiation field PATHOGNOMONIC RB1 germline; CDK4/6i inactive in RB1-null tumours",
        "surveillance_key": "RetCam examination under anaesthesia q4-6wk infancy; avoid orbital RT (secondary sarcoma 10-15x post-RT); annual ophthalmology exam; WB-MRI annually for secondary cancer surveillance; CDK4/6i inactive in RB1-null secondary RMS",
        "key_distinctions": [
            "BILATERAL-RETINOBLASTOMA-PATHOGNOMONIC-GERMLINE-RB1",
            "SECONDARY-RMS-10-15X-POST-RT-PATHOGNOMONIC",
            "AVOID-ORBITAL-RADIATION-RB1-GERMLINE",
            "CDK4-6I-INACTIVE-RB1-NULL-RESISTANCE",
            "TRILATERAL-RB-PINEALOBLASTOMA-PATHOGNOMONIC",
            "LATENCY-10-20YR-SECONDARY-RMS-POST-RT",
        ],
    },
    {
        "gene": "HRAS",
        "protein": (
            "HRAS -- 11p15.5 Autosomal-Dominant-GOF -- 189aa -- "
            "HRAS-21kDa-RAS-GTPase-Costello-Syndrome-ERMS-15-20pct-PATHOGNOMONIC-"
            "G12S-Most-Common-HCM-LIFE-THREATENING-Echo-Annual-MANDATORY-OMIM-218040"
        ),
        "locus": "11p15.5",
        "protein_size": (
            "189 aa / 21 kDa / 11p15.5 HRAS encodes Harvey RAS (proto-oncogene GTPase): "
            "STRUCTURE: "
            "  189 aa / 21 kDa; small GTPase; RAS superfamily; "
            "  G-domain (aa 1-166): GTP binding; switch I (aa 30-40) + switch II (aa 60-76); "
            "  G12 residue (Gly12): hotspot GOF mutation site (G12S, G12A, G12C, G12V, G12D); "
            "  G13 residue: secondary hotspot (G13D, G13S); "
            "  Q61 residue (Gln61): hotspot GOF (Q61R, Q61K, Q61L) -- most oncogenic; "
            "  HRAS GOF: impaired intrinsic GTPase activity -> constitutive RAS-MAPK/PI3K activation; "
            "  HRAS G12S: Costello germline specific -- milder GOF vs G12V (somatic cancer); "
            "COSTELLO SYNDROME (HRAS GERMLINE GOF): "
            "  OMIM 218040; HRAS germline GOF; RASopathy spectrum; "
            "  G12S: ~80% Costello germline mutations (mild activating GOF); "
            "  G12A, G13C, G12V, Q61R: less common germline HRAS in Costello; "
            "  Cutaneous papillomas (perinasal, perianal): PATHOGNOMONIC Costello -- onset 2-3yr; "
            "  Coarse facial features: macrocephaly, deep palmar/plantar creases; "
            "  Growth restriction (FTT, short stature), intellectual disability; "
            "  Cardiac: hypertrophic cardiomyopathy (HCM) 60-70% -- LIFE-THREATENING; "
            "    Pulmonary valve stenosis 40-50%; atrial fibrillation; sudden death risk; "
            "    Annual echocardiogram MANDATORY (HCM surveillance); "
            "    Biventricular hypertrophy most common; asymmetric septal hypertrophy; "
            "RHABDOMYOSARCOMA IN COSTELLO SYNDROME: "
            "  RMS: 15-20% Costello patients develop RMS by age 20yr; "
            "  Predominantly embryonal RMS (ERMS) -- PATHOGNOMONIC Costello/HRAS; "
            "  Bladder RMS (ERMS botryoid): most common Costello RMS site; "
            "  Orbital/paratesticular/paravaginal: secondary sites; "
            "  HRAS GOF in RMS: Costello ERMS vs sporadic ERMS (somatic HRAS rare); "
            "  Treatment: VAC backbone (vincristine/actinomycin-D/cyclophosphamide); "
            "    HCM + cyclophosphamide: cardiac function monitoring essential; "
            "OTHER COSTELLO TUMOUR RISKS: "
            "  Neuroblastoma (NB): 10-15% Costello (HRAS RAS pathway); "
            "  Bladder transitional cell carcinoma: adult risk; "
            "  Thyroid cancer (papillary): elevated; "
        ),
        "inheritance": "Autosomal dominant GOF; ~95% de novo (parental HRAS G12S mosaicism ~5%); high penetrance for cardiac features; RMS 15-20% cumulative by age 20yr; G12S milder than Q61R/G12V (lethal in utero somatic)",
        "cancer_risk": "RMS 15-20% lifetime PATHOGNOMONIC Costello (predominantly ERMS bladder/paravaginal); NB 10-15%; bladder transitional carcinoma adult risk; papillary thyroid cancer elevated; HCM LIFE-THREATENING cardiac complication",
        "pathognomonic": "Perinasal/perianal papillomas + ERMS PATHOGNOMONIC Costello syndrome (HRAS germline GOF); bladder ERMS botryoid in child with Costello PATHOGNOMONIC; HCM + coarse facies + RMS = Costello phenotype PATHOGNOMONIC",
        "surveillance_key": "Annual echocardiogram MANDATORY (HCM surveillance -- sudden death risk); RMS surveillance: annual abdominal/pelvic US + urine cytology; cardiac function before each VAC cycle; avoid high-intensity exercise (HCM); annual thyroid US from age 10yr",
        "key_distinctions": [
            "COSTELLO-ERMS-15-20PCT-PATHOGNOMONIC-HRAS-GOF",
            "G12S-MOST-COMMON-GERMLINE-COSTELLO",
            "HCM-60-70PCT-LIFE-THREATENING-ANNUAL-ECHO-MANDATORY",
            "BLADDER-ERMS-BOTRYOID-PATHOGNOMONIC-COSTELLO",
            "PAPILLOMAS-PERINASAL-PERIANAL-PATHOGNOMONIC-COSTELLO",
            "CARDIAC-MONITORING-MANDATORY-VAC-CHEMOTHERAPY",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Biallelic-AR-LOF-FA-D1 / AD-LOF-HBOC -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FAD1-Embryonal-RMS-PATHOGNOMONIC-"
            "AVOID-Alkylating-Cisplatin-HRD-Sibling-Donor-Exclusion-MANDATORY-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes BRCA2 (HR repair scaffold; FA-D1 gene): "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; large nuclear HR scaffold protein; "
            "  PALB2-binding domain (aa 10-40): nuclear targeting; "
            "  OB-fold domains; BRC repeats x8 (aa 1002-2085): RAD51 loading at DSBs; "
            "  DBD (aa 2402-3190): ssDNA/dsDNA binding at resected DSBs; "
            "  Biallelic BRCA2 LOF: Fanconi anemia complementation group D1 (FA-D1/FANCD1); "
            "FANCONI ANEMIA D1 (FA-D1) AND RMS: "
            "  OMIM 605724; most severe FA subtype; childhood cancer onset 2-5yr; "
            "  Embryonal RMS: PATHOGNOMONIC FA-D1 (5-10% biallelic BRCA2 FA-D1 patients); "
            "  Bilateral Wilms tumour: PATHOGNOMONIC FA-D1 (50-60%); "
            "  Medulloblastoma (SHH-subtype): PATHOGNOMONIC FA-D1; "
            "  ALL (acute lymphoblastic leukaemia): 20-30% FA-D1; "
            "  BMF (bone marrow failure): near universal FA-D1 (aplastic anaemia); "
            "  VACTERL association (vertebral + cardiac + TE fistula + limb anomalies); "
            "  DEB/MMC chromosomal fragility test PATHOGNOMONIC FA diagnosis; "
            "TREATMENT RMS IN FA-D1 (BRCA2): "
            "  MODIFIED VAC: AVOID cyclophosphamide (severe BMF in FA-D1); "
            "  Cisplatin preferred: HRD sensitivity (BRCA2 LOF -> HR deficiency); "
            "  Ifosfamide: AVOID (alkylating agent -> BMF in FA-D1); "
            "  Actinomycin-D + vincristine: lower BMF risk; use with caution; "
            "  Olaparib (PARP inhibitor): BRCA2-deficient tumour activity; "
            "SIBLING DONOR EXCLUSION (FA-D1): "
            "  ALL potential HSCT donors: DEB/MMC test MANDATORY before donor evaluation; "
            "  Siblings: 25% FA-D1 risk; affected sibling CANNOT donate (FA-D1 bone marrow); "
            "  Both parents: obligate heterozygous BRCA2 (HBOC counselling); "
        ),
        "inheritance": "Biallelic AR LOF (FA-D1): compound heterozygous BRCA2 -- both parents obligate het; monoallelic AD LOF (HBOC): 50% transmission; near-complete BMF penetrance FA-D1; solid tumour risk 5-10% FA-D1",
        "cancer_risk": "FA-D1 biallelic: embryonal RMS PATHOGNOMONIC; bilateral Wilms PATHOGNOMONIC; medulloblastoma PATHOGNOMONIC; ALL 20-30%; BMF near-universal; monoallelic HBOC: breast 47-69%, ovarian 11-17%",
        "pathognomonic": "Embryonal RMS PATHOGNOMONIC FA-D1 (biallelic BRCA2); DEB/MMC chromosomal fragility PATHOGNOMONIC FA diagnosis; VACTERL + BMF + childhood solid tumour = FA-D1; sibling donor exclusion MANDATORY",
        "surveillance_key": "DEB/MMC test ALL potential HSCT donors MANDATORY; sibling donor exclusion before HSCT; MODIFIED VAC: avoid cyclophosphamide/ifosfamide (BMF FA-D1); cisplatin preferred (HRD); actinomycin/vincristine lower BMF risk; monoallelic BRCA2: PBSO age 40-45yr",
        "key_distinctions": [
            "FA-D1-EMBRYONAL-RMS-PATHOGNOMONIC",
            "DEB-MMC-CHROMOSOMAL-FRAGILITY-PATHOGNOMONIC-FA",
            "SIBLING-DONOR-EXCLUSION-MANDATORY-HSCT",
            "AVOID-ALKYLATING-AGENTS-CYCLOPHOSPHAMIDE-IFOSFAMIDE-BMF",
            "CISPLATIN-HRD-SENSITIVITY-BRCA2",
            "VACTERL-BMF-CHILDHOOD-SOLID-TUMOUR-FA-D1",
        ],
    },
    {
        "gene": "PTPN11",
        "protein": (
            "PTPN11 -- 12q24.13 Autosomal-Dominant-GOF -- 593aa -- "
            "SHP2-68kDa-PTP-RASopathy-Noonan-Syndrome-RMS-1-2pct-"
            "JMML-200-500x-Pulmonary-Valve-Stenosis-PATHOGNOMONIC-HCM-20pct-OMIM-163950"
        ),
        "locus": "12q24.13",
        "protein_size": (
            "593 aa / 68 kDa / 12q24.13 PTPN11 encodes SHP2 (SH2 domain-containing protein tyrosine phosphatase 2): "
            "STRUCTURE: "
            "  593 aa / 68 kDa; protein tyrosine phosphatase; "
            "  N-terminal SH2 domain (N-SH2, aa 3-102): autoinhibitory function (N-SH2 blocks catalytic site); "
            "  C-terminal SH2 domain (C-SH2, aa 113-215): phosphotyrosine binding; "
            "  PTP domain (aa 221-523): catalytic tyrosine phosphatase; "
            "  Normal SHP2: RAS-MAPK activation downstream of RTKs; "
            "  GOF mutations: disrupt N-SH2/PTP autoinhibitory interface -> constitutive SHP2 activation; "
            "  E76K, D61N, A72V, Q79R, T73I, N58D: Noonan hotspots (all GOF -- disrupt autoinhibition); "
            "NOONAN SYNDROME (PTPN11 GOF): "
            "  OMIM 163950; most common RASopathy (1 in 1000-2500); ~50% due to PTPN11 GOF; "
            "  Pulmonary valve stenosis: 50-60% PATHOGNOMONIC Noonan (most common congenital heart defect); "
            "  HCM: 20-30% Noonan (PTPN11 GOF alleles T73I, Q79R strongest HCM association); "
            "    PTPN11 HCM differs from sarcomeric HCM: RAS-MAPK activation not sarcomere protein mutation; "
            "  Short stature (>80%), low-set posteriorly rotated ears, webbed neck, ptosis: classic Noonan; "
            "  NS-ML (LEOPARD syndrome): multiple lentigines + LOF PTPN11 (E69K, T279C, Q510E, Y279C); "
            "    DISTINCT from Noonan GOF: LEOPARD = distinct electrophoretic LOF in PTP domain; "
            "RHABDOMYOSARCOMA IN NOONAN: "
            "  RMS: 1-2% Noonan patients (PTPN11 GOF); "
            "  Predominantly ERMS subtype; "
            "  Noonan-associated RMS: head/neck, paratesticular, genitourinary sites; "
            "  Mechanism: constitutive RAS-MAPK activation -> myoblast proliferation dysregulation; "
            "JMML IN NOONAN (PTPN11 GOF): "
            "  JMML 200-500x elevated in Noonan syndrome (PTPN11 GOF); "
            "  Noonan JMML: often self-limited (spontaneous remission ~30-50% Noonan-JMML); "
            "  Distinguish: Noonan-JMML vs sporadic JMML (somatic PTPN11 E76K -- MORE aggressive); "
            "  Germline PTPN11 JMML: watch-and-wait acceptable in Noonan (unlike sporadic JMML -> HSCT); "
        ),
        "inheritance": "Autosomal dominant GOF; ~50% de novo; GOF mutations disrupt N-SH2 autoinhibition; LEOPARD (NS-ML) = distinct LOF alleles (different mechanism; do NOT confuse with Noonan GOF); near-complete penetrance cardiac features",
        "cancer_risk": "JMML 200-500x elevated (Noonan-JMML often self-limited); RMS 1-2% Noonan; ALL elevated; haematological malignancy risk; HCM LIFE-THREATENING (cardiac monitoring MANDATORY); pulmonary valve stenosis 50-60%",
        "pathognomonic": "Pulmonary valve stenosis + short stature + webbed neck + ptosis PATHOGNOMONIC Noonan (PTPN11 GOF); Noonan-JMML self-limited vs sporadic JMML E76K aggressive -- DISTINCT prognosis; NS-ML lentigines PATHOGNOMONIC LEOPARD (LOF PTPN11)",
        "surveillance_key": "Annual echocardiogram (HCM + pulmonary stenosis follow-up); FBC annually for JMML screening; RMS surveillance: annual abdominal/pelvic US; distinguish Noonan-JMML (watch-wait) from sporadic JMML (HSCT); cardiac clearance before RMS chemotherapy",
        "key_distinctions": [
            "NOONAN-JMML-SELF-LIMITED-vs-SPORADIC-JMML-HSCT",
            "PULMONARY-VALVE-STENOSIS-50-60PCT-PATHOGNOMONIC-NOONAN",
            "PTPN11-GOF-NOONAN-vs-LOF-LEOPARD-DISTINCT",
            "RMS-1-2PCT-NOONAN-ERMS-GENITOURINARY",
            "HCM-20-30PCT-CARDIAC-MONITORING-MANDATORY",
            "E76K-SOMATIC-SPORADIC-JMML-MORE-AGGRESSIVE",
        ],
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 Autosomal-Dominant-LOF -- 385aa -- "
            "INI1-44kDa-SWI-SNF-RTPS2-MRT-ATRT-Under-3yr-PATHOGNOMONIC-"
            "Epithelioid-RMS-INI1-Loss-Tazemetostat-EZH2i-FDA2020-HSCT-Consolidation-OMIM-601607"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / 44 kDa / 22q11.23 SMARCB1 encodes INI1 (integrase interactor 1; SNF5; BAF47): "
            "STRUCTURE: "
            "  385 aa / 44 kDa; core subunit of SWI/SNF ATP-dependent chromatin remodelling complex; "
            "  Rpt1 domain (aa 1-60): protein-protein interactions; "
            "  Rpt2 domain (aa 157-238): protein-protein interactions; "
            "  HSA domain (aa 75-110): actin-dependent remodelling; "
            "  SMARCB1 is an obligate subunit: loss inactivates entire SWI/SNF complex; "
            "  SMARCB1 LOF -> EZH2 (PRC2) becomes unopposed -> H3K27me3 accumulation; "
            "    -> EZH2 dependency (EZH2 inhibitor tazemetostat targets this vulnerability); "
            "  INI1 IHC: nuclear loss of INI1 PATHOGNOMONIC SMARCB1 biallelic inactivation; "
            "RHABDOID TUMOUR PREDISPOSITION SYNDROME 2 (RTPS2): "
            "  OMIM 613325; SMARCB1 germline LOF; rhabdoid tumour spectrum; "
            "  MRT (malignant rhabdoid tumour): extrarenal sites (liver, soft tissue, CNS); "
            "  ATRT (atypical teratoid/rhabdoid tumour): CNS; predominantly under 3yr; "
            "  Both MRT and ATRT: PATHOGNOMONIC SMARCB1 biallelic loss (germline + somatic); "
            "  RTPS1 (SMARCA4 germline LOF): distinct -- BRG1 loss PATHOGNOMONIC; "
            "EPITHELIOID RMS / SMARCB1-DEFICIENT SARCOMAS: "
            "  Epithelioid RMS: rare spindle/epithelioid sarcoma subset with INI1 loss; "
            "  Epithelioid sarcoma: SMARCB1/INI1 loss in 70-100% -- PATHOGNOMONIC epithelioid sarcoma; "
            "    Proximal-type epithelioid sarcoma: aggressive; biallelic SMARCB1 inactivation; "
            "  INI1 IHC nuclear loss: differentiates epithelioid RMS/sarcoma from ERMS (INI1 retained); "
            "TAZEMETOSTAT (EZH2 INHIBITOR -- FDA 2020): "
            "  Tazemetostat FDA 2020 for epithelioid sarcoma (SMARCB1 loss) and relapsed/refractory FL (EZH2 GOF); "
            "  Mechanism: EZH2 inhibition reverses H3K27me3 accumulation from SMARCB1 loss; "
            "  Tazemetostat for ATRT/MRT: clinical trials ongoing; pediatric dose established; "
            "  HSCT consolidation: standard for ATRT/MRT (curative intent in RTPS2); "
            "SURVEILLANCE (SMARCB1 GERMLINE): "
            "  Brain MRI q3-6M until age 5yr (ATRT risk highest <3yr); "
            "  Abdominal/chest MRI q6M until age 5yr (MRT extrarenal); "
            "  Renal US annually (rhabdoid renal tumour); "
        ),
        "inheritance": "Autosomal dominant LOF; ~50% de novo (germline SMARCB1 LOF); somatic second hit required; variable penetrance; biallelic inactivation in tumour (IHC nuclear loss); RTPS1 (SMARCA4) is distinct syndrome",
        "cancer_risk": "ATRT (CNS) PATHOGNOMONIC under 3yr; MRT (extrarenal) PATHOGNOMONIC; epithelioid RMS/sarcoma (INI1-loss); rhabdoid renal tumour; concurrent bilateral MRT reported; 30-35% RTPS2 develop tumour by age 5yr",
        "pathognomonic": "ATRT under 3yr PATHOGNOMONIC SMARCB1 germline LOF; MRT (extrarenal) PATHOGNOMONIC; INI1 IHC nuclear loss PATHOGNOMONIC SMARCB1 biallelic inactivation; epithelioid sarcoma 70-100% SMARCB1 loss PATHOGNOMONIC",
        "surveillance_key": "Brain MRI q3-6M until age 5yr (ATRT); abdominal/chest MRI q6M until age 5yr (MRT); tazemetostat FDA 2020 for SMARCB1-loss epithelioid sarcoma; HSCT consolidation for ATRT/MRT; INI1 IHC on all undifferentiated paediatric tumours",
        "key_distinctions": [
            "ATRT-UNDER-3YR-PATHOGNOMONIC-SMARCB1",
            "INI1-IHC-NUCLEAR-LOSS-PATHOGNOMONIC",
            "TAZEMETOSTAT-EZH2I-FDA2020-SMARCB1-LOSS",
            "HSCT-CONSOLIDATION-ATRT-MRT",
            "EPITHELIOID-SARCOMA-70-100PCT-SMARCB1-LOSS",
            "EZH2-DEPENDENCY-H3K27ME3-UNOPPOSED",
        ],
    },
]

TUMOR_TYPES = {
    "TP53": {
        "Embryonal RMS (ERMS) Head/Neck": 14,
        "Spindle Cell/Sclerosing RMS": 10,
        "ERMS Genitourinary": 8,
        "ERMS Extremity": 6,
        "Alveolar RMS (ARMS) Relapse": 2,
    },
    "NF1": {
        "Embryonal RMS Orbital": 14,
        "Embryonal RMS Paratesticular": 10,
        "ERMS Head/Neck": 8,
        "MPNST (plexiform NF transform)": 8,
    },
    "DICER1": {
        "Cervical/Vaginal ERMS Botryoid": 18,
        "Bladder ERMS": 12,
        "Uterine ERMS": 6,
        "ERMS Paravaginal": 4,
    },
    "RB1": {
        "Secondary RMS (post-RT orbital)": 18,
        "Secondary Osteosarcoma (post-RT)": 12,
        "Retinoblastoma (primary)": 10,
    },
    "HRAS": {
        "Bladder ERMS Botryoid": 18,
        "Orbital ERMS": 8,
        "Paravaginal ERMS": 8,
        "Paratesticular ERMS": 6,
    },
    "BRCA2": {
        "Embryonal RMS (FA-D1)": 14,
        "Bilateral Wilms (FA-D1)": 12,
        "Medulloblastoma (FA-D1)": 8,
        "ALL (FA-D1)": 6,
    },
    "PTPN11": {
        "ERMS Head/Neck": 14,
        "ERMS Genitourinary": 10,
        "JMML (haematological)": 10,
        "ERMS Paratesticular": 6,
    },
    "SMARCB1": {
        "ATRT (CNS, under 3yr)": 16,
        "MRT Extrarenal": 12,
        "Epithelioid Sarcoma (INI1-loss)": 8,
        "Rhabdoid Renal Tumour": 4,
    },
}

PATHOGENIC_VARIANTS = {
    "TP53": {
        "p.Arg248Trp (R248W)": 10,
        "p.Arg175His (R175H)": 8,
        "p.Arg273His (R273H)": 8,
        "p.Arg337His (R337H)": 10,
        "p.Pro151Ser": 4,
    },
    "NF1": {
        "p.Arg1947Ter (R1947X)": 10,
        "c.2033del (fs)": 10,
        "Exon 1-6 del (MLPA)": 8,
        "p.Gln519Ter": 6,
        "p.Arg304Ter": 6,
    },
    "DICER1": {
        "p.Glu1705Lys (E1705K)": 12,
        "p.Asp1709Asn (D1709N)": 10,
        "c.5438+1G>A (splice)": 8,
        "p.Leu1264Pro": 6,
        "p.Arg1412Ter": 4,
    },
    "RB1": {
        "c.958_959insG (fs)": 12,
        "p.Arg455Ter (R455X)": 10,
        "p.Tyr79Ter": 8,
        "Exon 1-3 del (MLPA)": 8,
        "c.1960C>T (R654X)": 2,
    },
    "HRAS": {
        "p.Gly12Ser (G12S)": 24,
        "p.Gly12Ala (G12A)": 6,
        "p.Gln61Arg (Q61R)": 4,
        "p.Gly13Cys (G13C)": 4,
        "p.Gly12Val (G12V)": 2,
    },
    "BRCA2": {
        "p.Lys2729Thr (K2729T)": 8,
        "c.8954-2A>G (splice)": 10,
        "p.Asp2723His": 8,
        "p.Trp2626Ter": 8,
        "p.Asn991Ile": 6,
    },
    "PTPN11": {
        "p.Glu76Lys (E76K)": 12,
        "p.Asp61Asn (D61N)": 10,
        "p.Ala72Val (A72V)": 8,
        "p.Gln79Arg (Q79R)": 6,
        "p.Thr73Ile (T73I)": 4,
    },
    "SMARCB1": {
        "c.157C>T (R53Ter)": 10,
        "c.1148del (fs)": 8,
        "p.Arg201Ter": 8,
        "Exon 9 del (MLPA)": 10,
        "p.Ala100Pro (A100P splice)": 4,
    },
}

TREATMENT_PROTOCOLS = {
    "TP53": [
        "RMS LFS: AVOID RADIATION ABSOLUTELY -- omit involved-field RT",
        "VAC backbone (vincristine + actinomycin-D + cyclophosphamide): modified dose",
        "Low-dose VAC: reduce cyclophosphamide (BMF sensitivity in LFS bone marrow)",
        "Proton therapy if RT absolutely unavoidable (lower exit dose, reduced secondary malignancy)",
        "WBMRI annually: NOT CT/PET for surveillance",
        "LFS genetic counselling: all first-degree relatives 50% risk",
    ],
    "NF1": [
        "NF1 RMS: standard VAC backbone (vincristine/actinomycin-D/cyclophosphamide)",
        "Minimise radiation dose + field (secondary MPNST risk); proton preferred over photon",
        "Selumetinib: symptomatic inoperable plexiform NF (FDA 2020, ≥2yr); NOT for RMS",
        "MPNST: ifosfamide + doxorubicin (sarcoma regimen); selumetinib Phase II",
        "NF1-GIST: surgery primary; imatinib INACTIVE (wild-type KIT/PDGFRA)",
        "Annual MRI brain (OPG) + skin examination; avoid routine CT (secondary MPNST)",
    ],
    "DICER1": [
        "Cervical ERMS: fertility-sparing surgery (trachelectomy/polypectomy) + VAC",
        "Low-dose VAC: actinomycin-D + vincristine +/- cyclophosphamide (children)",
        "Avoid pelvic/vaginal RT in DICER1 patients (developing organs + fertility)",
        "Bladder ERMS: organ-sparing approach preferred (partial cystectomy + chemo)",
        "PPB Type I: lung-sparing surgery + VAC; chest CT siblings <8yr MANDATORY",
        "Fertility counselling and preservation pre-treatment for adolescent females",
    ],
    "RB1": [
        "Retinoblastoma primary: chemoreduction (CEV: carboplatin/etoposide/vincristine) + local consolidation",
        "AVOID ORBITAL RADIATION if alternative (chemoreduction, cryotherapy, laser) feasible",
        "Secondary RMS post-RT: VAC backbone; CDK4/6 inhibitors INACTIVE (pRb null)",
        "Secondary osteosarcoma: MAP protocol (methotrexate/adriamycin/cisplatin)",
        "Annual WB-MRI: secondary cancer surveillance (NOT CT)",
        "Annual ophthalmology RetCam exam under anaesthesia (infancy q4-6wk)",
    ],
    "HRAS": [
        "Costello RMS: VAC backbone (vincristine + actinomycin-D + cyclophosphamide)",
        "Cardiac function monitoring before EACH VAC cycle (HCM surveillance)",
        "Echocardiogram baseline + every 2-3 cycles + post-treatment: HCM progression",
        "Annual echocardiogram maintenance: HCM (lifetime Costello surveillance)",
        "Bladder ERMS botryoid: organ-sparing surgery + VAC where feasible",
        "Avoid high-intensity exercise / extreme dehydration in HCM (sudden death risk)",
    ],
    "BRCA2": [
        "FA-D1 RMS: MODIFIED VAC -- AVOID cyclophosphamide (severe BMF)",
        "Cisplatin + actinomycin-D + vincristine: preferred over alkylating agents (HRD sensitivity)",
        "HSCT: curative for BMF; DEB/MMC ALL potential donors MANDATORY pre-evaluation",
        "SIBLING DONOR EXCLUSION: all siblings tested DEB/MMC before HSCT evaluation",
        "Olaparib: PARP inhibitor activity in BRCA2-deficient RMS (compassionate use)",
        "Bilateral Wilms (FA-D1): nephron-sparing surgery aim",
    ],
    "PTPN11": [
        "Noonan RMS: standard VAC backbone; cardiac clearance MANDATORY pre-treatment",
        "Noonan JMML: watch-and-wait acceptable (spontaneous remission 30-50%); HSCT if progressive",
        "Sporadic JMML (somatic PTPN11 E76K): HSCT required (distinct from Noonan-JMML)",
        "Annual echocardiogram: pulmonary stenosis + HCM monitoring",
        "Cardiac catheterisation if severe pulmonary stenosis (balloon valvuloplasty)",
        "Trametinib (MEK inhibitor): clinical trials for PTPN11 GOF-driven tumours",
    ],
    "SMARCB1": [
        "ATRT: multimodal (surgery + chemotherapy + proton RT); HSCT consolidation",
        "MRT extrarenal: surgery + ICE (ifosfamide/carboplatin/etoposide) + high-dose CT/ASCT",
        "Tazemetostat (EZH2 inhibitor): FDA 2020 for SMARCB1-loss epithelioid sarcoma",
        "HSCT consolidation: standard for ATRT/MRT (curative intent); myeloablative conditioning",
        "Radiation: required for ATRT (CSI craniospinal) if age >3yr feasible",
        "Brain MRI q3-6M until age 5yr: ATRT surveillance in RTPS2 germline",
    ],
}

SURVEILLANCE_PROTOCOLS = {
    "TP53": [
        "WBMRI Toronto Protocol annually: whole-body MRI (NOT CT/PET)",
        "Brain MRI 6-monthly first 5yr then annually",
        "Annual clinical exam: skin (osteosarcoma/STS/RMS awareness)",
        "Breast MRI annually from age 20yr (LFS breast cancer)",
        "AVOID ALL RADIATION for surveillance (no CT/PET-CT/DEXA)",
        "CASCADE: 50% risk first-degree relatives -- offer germline TP53 testing",
    ],
    "NF1": [
        "Annual full dermatology + ophthalmology exam: CALM, Lisch nodules",
        "MRI brain/spine annually childhood: OPG + CNS gliomas",
        "Whole-body MRI q2yr adults: MPNST surveillance in plexiform NF",
        "Annual abdominal/pelvic US: RMS surveillance (elevated risk)",
        "Avoid CT/radiation: secondary MPNST risk",
        "PET-FDG (or WB-MRI) if rapid plexiform NF growth: MPNST suspected",
    ],
    "DICER1": [
        "Pelvic/abdominal US annually from puberty: cervical/vaginal ERMS, SLCT",
        "Chest CT siblings <8yr: PPB screening MANDATORY",
        "Annual thyroid US from age 8yr: MNG (multinodular goitre 75%)",
        "Nephrology follow-up: cystic nephroma surveillance (annual renal US)",
        "Fertility counselling pre-treatment adolescent females with cervical ERMS",
        "MRI preferred over CT in children: radiation minimisation",
    ],
    "RB1": [
        "RetCam exam under anaesthesia q4-6wk (infancy) until disease control",
        "Annual ophthalmology exam: lifelong (new RB risk + secondary cancers)",
        "Avoid orbital radiation (secondary RMS/osteosarcoma 10-15x post-RT)",
        "Annual WB-MRI from age 10yr: secondary cancer surveillance (secondary sarcoma latency 10-20yr)",
        "CDK4/6i resistance: anticipated in RB1-null secondary RMS (inform oncologist)",
        "Cascade testing: parents + siblings (50% risk from affected parent)",
    ],
    "HRAS": [
        "Annual echocardiogram: HCM + pulmonary valve stenosis (lifelong Costello)",
        "Annual abdominal/pelvic US + urine cytology: RMS surveillance",
        "Annual thyroid US from age 10yr: papillary thyroid cancer",
        "Cardiac clearance before each VAC chemotherapy cycle (HCM monitoring)",
        "Avoid high-intensity exercise + extreme dehydration (sudden death HCM risk)",
        "Bladder surveillance: annual urine cytology from age 20yr (transitional cell CA risk)",
    ],
    "BRCA2": [
        "DEB/MMC test ALL potential HSCT donors MANDATORY pre-evaluation",
        "Sibling donor exclusion: before HSCT evaluation (all siblings DEB/MMC tested)",
        "Modified FA chemotherapy: avoid cyclophosphamide/ifosfamide (BMF FA-D1)",
        "Androgens bridge (oxymetholone): pre-HSCT BMF management",
        "Monoallelic BRCA2 HBOC: PBSO age 40-45yr (ovarian prevention)",
        "Cisplatin-based preferred (HRD BRCA2-LOF sensitivity)",
    ],
    "PTPN11": [
        "Annual echocardiogram: HCM + pulmonary valve stenosis follow-up",
        "Annual FBC: JMML screening (monocytosis, thrombocytopenia, anaemia)",
        "Annual abdominal/pelvic US: RMS surveillance in Noonan",
        "Cardiac catheterisation + balloon valvuloplasty if severe pulmonary stenosis",
        "Distinguish Noonan-JMML (watch-wait) from sporadic JMML (HSCT): germline vs somatic",
        "Cardiac clearance mandatory before RMS chemotherapy initiation",
    ],
    "SMARCB1": [
        "Brain MRI q3-6M until age 5yr: ATRT surveillance (highest risk <3yr)",
        "Abdominal/chest MRI q6M until age 5yr: MRT extrarenal surveillance",
        "Renal US annually: rhabdoid renal tumour surveillance",
        "INI1 IHC on all undifferentiated paediatric tumours: nuclear loss PATHOGNOMONIC",
        "Tazemetostat: EZH2i FDA 2020 for SMARCB1-loss tumours (epithelioid sarcoma)",
        "HSCT consolidation: curative intent for ATRT/MRT in RTPS2",
    ],
}


def _make_cohort(gene_idx: int, seed: int) -> list:
    """Generate 40-patient cohort for one RMS predisposition gene."""
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]["gene"]

    age_ranges = {
        "TP53":    (1.0, 15.0),
        "NF1":     (1.0, 10.0),
        "DICER1":  (2.0, 18.0),
        "RB1":     (5.0, 30.0),   # secondary RMS: longer latency
        "HRAS":    (0.5, 8.0),
        "BRCA2":   (0.5, 5.0),
        "PTPN11":  (1.0, 12.0),
        "SMARCB1": (0.1, 3.0),   # ATRT predominantly under 3yr
    }
    lo, hi = age_ranges[gene]

    risk_weights = {
        "TP53":    ["low", "intermediate", "high", "high"],
        "NF1":     ["low", "low", "intermediate", "high"],
        "DICER1":  ["low", "low", "intermediate", "intermediate"],
        "RB1":     ["intermediate", "high", "high"],
        "HRAS":    ["low", "low", "intermediate", "high"],
        "BRCA2":   ["intermediate", "high", "high"],
        "PTPN11":  ["low", "intermediate", "intermediate", "high"],
        "SMARCB1": ["high", "high", "high"],
    }

    subtype_weights = {
        "TP53":    ["ERMS", "ERMS", "Spindle-Sclerosing", "ARMS-relapse"],
        "NF1":     ["ERMS", "ERMS", "MPNST", "ERMS-orbital"],
        "DICER1":  ["ERMS-botryoid", "ERMS-cervical", "ERMS-bladder", "ERMS-uterine"],
        "RB1":     ["Secondary-RMS-postRT", "Secondary-Osteosarcoma", "Primary-RB"],
        "HRAS":    ["ERMS-bladder-botryoid", "ERMS-orbital", "ERMS-paravaginal", "ERMS-paratesticular"],
        "BRCA2":   ["ERMS-FAD1", "Wilms-FAD1", "Medulloblastoma-FAD1", "ALL-FAD1"],
        "PTPN11":  ["ERMS-head-neck", "ERMS-GU", "JMML", "ERMS-paratesticular"],
        "SMARCB1": ["ATRT", "MRT-extrarenal", "Epithelioid-sarcoma", "Rhabdoid-renal"],
    }

    cr_prob = {
        "TP53": 0.55, "NF1": 0.62, "DICER1": 0.72, "RB1": 0.45,
        "HRAS": 0.68, "BRCA2": 0.58, "PTPN11": 0.65, "SMARCB1": 0.38,
    }
    radiation_prob = {
        "TP53": 0.05,   # AVOID RADIATION ABSOLUTELY
        "NF1": 0.18,    # Minimise RT (secondary MPNST)
        "DICER1": 0.12, # AVOID in children
        "RB1": 0.22,    # AVOID orbital RT; secondary RMS post-RT
        "HRAS": 0.38,
        "BRCA2": 0.20,
        "PTPN11": 0.40,
        "SMARCB1": 0.45,  # RT part of ATRT protocol (>3yr)
    }
    targeted_prob = {
        "TP53": 0.08, "NF1": 0.42, "DICER1": 0.12, "RB1": 0.10,
        "HRAS": 0.15, "BRCA2": 0.28, "PTPN11": 0.35, "SMARCB1": 0.52,
    }
    relapse_prob = {
        "TP53": 0.50, "NF1": 0.38, "DICER1": 0.28, "RB1": 0.55,
        "HRAS": 0.35, "BRCA2": 0.50, "PTPN11": 0.40, "SMARCB1": 0.65,
    }
    gtr_prob = {
        "TP53": 0.50, "NF1": 0.45, "DICER1": 0.65, "RB1": 0.40,
        "HRAS": 0.60, "BRCA2": 0.48, "PTPN11": 0.52, "SMARCB1": 0.55,
    }

    variant_pool = list(PATHOGENIC_VARIANTS[gene].keys())

    patients = []
    for pid in range(1, 41):
        age = round(rng.uniform(lo, hi), 1)
        risk = rng.choice(risk_weights[gene])
        subtype = rng.choice(subtype_weights[gene])
        cr = rng.random() < cr_prob[gene]
        radiation = rng.random() < radiation_prob[gene]
        targeted = rng.random() < targeted_prob[gene]
        relapse = rng.random() < relapse_prob[gene]
        gtr = rng.random() < gtr_prob[gene]
        variant = rng.choice(variant_pool)

        if cr:
            response = "CR"
        elif rng.random() < 0.4:
            response = "PR"
        elif rng.random() < 0.5:
            response = "SD"
        else:
            response = "PD"

        if gene == "TP53":
            treatment = rng.choice([
                "VAC-No-RT", "VAC-Reduced-Dose-No-RT",
                "Proton-If-RT-Unavoidable", "VA-Low-Dose",
            ])
        elif gene == "NF1":
            treatment = rng.choice([
                "VAC-Proton-Preferred", "Selumetinib-PN", "Ifo-Dox-MPNST", "VAC-Minimal-RT",
            ])
        elif gene == "DICER1":
            treatment = rng.choice([
                "VAC-Fertility-Sparing", "VA-Cervical-Organ-Sparing",
                "VAC-Bladder-Organ-Sparing", "Low-Dose-VAC",
            ])
        elif gene == "RB1":
            treatment = rng.choice([
                "CEV-Chemoreduction-No-RT", "VAC-Secondary-RMS",
                "MAP-Secondary-Osteo", "Chemo-Cryo-Laser-No-RT",
            ])
        elif gene == "HRAS":
            treatment = rng.choice([
                "VAC-Cardiac-Monitor", "VAC-Bladder-Sparing",
                "VA-Low-Dose-Cardiac", "VAC-Costello-Protocol",
            ])
        elif gene == "BRCA2":
            treatment = rng.choice([
                "Modified-VAC-No-Alkylating", "Cisplatin-Act-Vinc-FAD1",
                "HSCT-BMF-Curative", "Olaparib-BRCA2-Deficient",
            ])
        elif gene == "PTPN11":
            treatment = rng.choice([
                "VAC-Cardiac-Clearance", "Watch-Wait-Noonan-JMML",
                "HSCT-Sporadic-JMML", "Trametinib-MEKi-Trial",
            ])
        else:  # SMARCB1
            treatment = rng.choice([
                "Surgery-ICE-HSCT-ATRT", "Tazemetostat-Epithelioid-Sarcoma",
                "Multimodal-ATRT-CSI", "ICE-HDCT-ASCT-MRT",
            ])

        patients.append({
            "gene": gene,
            "patient_id": f"{gene}-{pid:03d}",
            "age_dx": age,
            "risk_group": risk,
            "rms_subtype": subtype,
            "cr_achieved": cr,
            "response": response,
            "radiation_received": radiation,
            "targeted_therapy": targeted,
            "relapse": relapse,
            "gtr_resection": gtr,
            "treatment": treatment,
            "variant": variant,
        })
    return patients


def generate_overview() -> dict:
    """Generate Hereditary-Rhabdomyosarcoma-Predisposition-Atlas overview."""
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(len(ATLAS_GENES))]
    total = sum(len(c) for c in cohorts)

    all_patients = [p for cohort in cohorts for p in cohort]
    cr_pct = round(100 * sum(1 for p in all_patients if p["cr_achieved"]) / total)
    targeted_pct = round(100 * sum(1 for p in all_patients if p["targeted_therapy"]) / total)
    radiation_pct = round(100 * sum(1 for p in all_patients if p["radiation_received"]) / total)
    relapse_pct = round(100 * sum(1 for p in all_patients if p["relapse"]) / total)
    gtr_pct = round(100 * sum(1 for p in all_patients if p["gtr_resection"]) / total)
    mean_age = round(sum(p["age_dx"] for p in all_patients) / total, 1)

    tumor_counts: dict = {}
    for gdef in ATLAS_GENES:
        for ttype, n in TUMOR_TYPES[gdef["gene"]].items():
            tumor_counts[ttype] = tumor_counts.get(ttype, 0) + n
    top_tumors = dict(sorted(tumor_counts.items(), key=lambda x: -x[1])[:10])

    gene_summaries = []
    for i, (gdef, cohort) in enumerate(zip(ATLAS_GENES, cohorts)):
        ages = [p["age_dx"] for p in cohort]
        cr = round(100 * sum(p["cr_achieved"] for p in cohort) / len(cohort))
        rad = round(100 * sum(p["radiation_received"] for p in cohort) / len(cohort))
        tgt = round(100 * sum(p["targeted_therapy"] for p in cohort) / len(cohort))
        gene_summaries.append({
            "gene": gdef["gene"],
            "locus": gdef["locus"],
            "protein": gdef["protein"],
            "n_patients": len(cohort),
            "mean_age_dx": round(sum(ages) / len(ages), 1),
            "cr_pct": cr,
            "radiation_pct": rad,
            "targeted_pct": tgt,
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
        })

    return _json_safe({
        "atlas": "Hereditary-Rhabdomyosarcoma-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference: TP53-NF1-DICER1-RB1-HRAS-BRCA2-PTPN11-SMARCB1",
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_age_dx": mean_age,
        "cr_pct": cr_pct,
        "gtr_pct": gtr_pct,
        "targeted_pct": targeted_pct,
        "radiation_pct": radiation_pct,
        "relapse_pct": relapse_pct,
        "genes": [g["gene"] for g in ATLAS_GENES],
        "gene_summaries": gene_summaries,
        "top_tumor_types": top_tumors,
        "key_pathognomonic": {
            "TP53_AVOID_RT": "AVOID RADIATION ABSOLUTELY if TP53 LFS -- omit RMS RT; proton if unavoidable; WBMRI Toronto annually",
            "DICER1_CERVICAL_ERMS": "Cervical ERMS (adolescent female) PATHOGNOMONIC DICER1; bladder ERMS PATHOGNOMONIC DICER1; fertility-sparing surgery PREFERRED",
            "RB1_SECONDARY_RMS": "Secondary RMS 10-15x elevated post-RT in RB1 germline; latency 10-20yr; CDK4/6i INACTIVE (pRb null)",
            "HRAS_COSTELLO_HCM": "HRAS germline GOF = Costello syndrome; ERMS 15-20% PATHOGNOMONIC; HCM LIFE-THREATENING -- annual Echo MANDATORY",
            "BRCA2_FA_D1": "Embryonal RMS PATHOGNOMONIC FA-D1 (biallelic BRCA2); sibling donor exclusion MANDATORY; AVOID alkylating agents (BMF)",
            "SMARCB1_ATRT": "ATRT under 3yr PATHOGNOMONIC SMARCB1 germline; INI1 IHC nuclear loss PATHOGNOMONIC; tazemetostat EZH2i FDA 2020",
            "PTPN11_NOONAN_JMML": "Noonan-JMML self-limited (watch-wait) vs sporadic JMML E76K (HSCT); DISTINCT prognosis -- critical distinction",
            "NF1_MPNST_AVOID_RT": "NF1 MPNST 8-13% PATHOGNOMONIC (plexiform NF transformation); AVOID RADIATION (secondary MPNST); selumetinib FDA 2020 PN",
        },
        "key_management_rules": [
            "TP53 LFS: AVOID RADIATION ABSOLUTELY -- omit RMS RT; proton if unavoidable; WBMRI Toronto NOT CT",
            "DICER1 cervical/bladder ERMS: fertility-sparing surgery PREFERRED; avoid pelvic RT; low-dose VAC",
            "RB1 germline: AVOID orbital RT (secondary RMS/sarcoma 10-15x post-RT); CDK4/6i INACTIVE in RB1-null tumours",
            "HRAS Costello: annual Echo MANDATORY (HCM LIFE-THREATENING); cardiac clearance before each VAC cycle",
            "BRCA2 FA-D1: AVOID cyclophosphamide/ifosfamide (severe BMF); DEB/MMC ALL donors MANDATORY; sibling exclusion",
            "SMARCB1: INI1 IHC on ALL undifferentiated paediatric tumours; tazemetostat for SMARCB1-loss sarcoma",
            "PTPN11 Noonan-JMML: watch-and-wait (self-limited 30-50%); cardiac clearance before RMS chemotherapy",
            "NF1 RMS: minimise RT field (secondary MPNST); proton preferred; selumetinib for symptomatic plexiform NF",
        ],
        "clinical_pearls": [
            "DICER1 cervical ERMS: vaginal/cervical mass in teenage girl -> DICER1 germline testing MANDATORY; botryoid polypoid PATHOGNOMONIC",
            "RB1 secondary RMS: CDK4/6 inhibitors (palbociclib, ribociclib) are INACTIVE -- pRb is the substrate; do not prescribe",
            "HRAS G12S germline (Costello) is milder activating GOF than G12V/Q61R (lethal in utero somatic cancer) -- same codon, different prognosis",
            "SMARCB1 loss by IHC (INI1 nuclear loss) distinguishes ATRT/MRT/epithelioid sarcoma from ERMS (INI1 retained) -- critical IHC panel",
            "PTPN11 E76K SOMATIC (sporadic JMML) is MORE aggressive than PTPN11 germline Noonan-JMML -- germline = watch-wait, somatic = HSCT",
            "NF1-GIST is wild-type for KIT and PDGFRA -- imatinib NOT effective; surgery primary treatment (unlike sporadic GIST)",
            "BRCA2 FA-D1 RMS: MODIFIED VAC essential -- cyclophosphamide causes severe BMF in FA-D1; actinomycin-D + vincristine safer backbone",
            "TP53 LFS RMS: predominantly ERMS and spindle cell/sclerosing subtypes -- alveolar RMS (PAX3/FOXO1) not typical of LFS",
        ],
    })


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Rhabdomyosarcoma-Predisposition-Atlas."""
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(len(ATLAS_GENES))]

    breakdown = []
    for i, (gdef, cohort) in enumerate(zip(ATLAS_GENES, cohorts)):
        ages = [p["age_dx"] for p in cohort]
        cr_pct = round(100 * sum(p["cr_achieved"] for p in cohort) / len(cohort))
        rad_pct = round(100 * sum(p["radiation_received"] for p in cohort) / len(cohort))
        tgt_pct = round(100 * sum(p["targeted_therapy"] for p in cohort) / len(cohort))
        rel_pct = round(100 * sum(p["relapse"] for p in cohort) / len(cohort))
        gtr_pct = round(100 * sum(p["gtr_resection"] for p in cohort) / len(cohort))

        top_tumors = dict(
            sorted(TUMOR_TYPES[gdef["gene"]].items(), key=lambda x: -x[1])[:5]
        )
        top_variants = dict(
            sorted(PATHOGENIC_VARIANTS[gdef["gene"]].items(), key=lambda x: -x[1])[:5]
        )
        from collections import Counter
        treat_ctr = Counter(p["treatment"] for p in cohort)
        top_treats = dict(treat_ctr.most_common(5))

        breakdown.append({
            "gene": gdef["gene"],
            "locus": gdef["locus"],
            "protein_size": gdef["protein_size"],
            "inheritance": gdef["inheritance"],
            "n_patients": len(cohort),
            "mean_age_dx": round(sum(ages) / len(ages), 1),
            "cr_pct": cr_pct,
            "radiation_pct": rad_pct,
            "radiation_n": sum(p["radiation_received"] for p in cohort),
            "targeted_pct": tgt_pct,
            "targeted_n": sum(p["targeted_therapy"] for p in cohort),
            "relapse_pct": rel_pct,
            "gtr_pct": gtr_pct,
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
            "key_distinctions": gdef["key_distinctions"],
            "top_tumor_types": top_tumors,
            "top_variants": top_variants,
            "top_treatments": top_treats,
            "treatment_protocols": TREATMENT_PROTOCOLS[gdef["gene"]],
            "surveillance_protocols": SURVEILLANCE_PROTOCOLS[gdef["gene"]],
            "patients": [
                {
                    "gene": p["gene"],
                    "age_dx": p["age_dx"],
                    "risk_group": p["risk_group"],
                    "rms_subtype": p["rms_subtype"],
                    "treatment": p["treatment"],
                    "response": p["response"],
                    "radiation_received": p["radiation_received"],
                    "targeted_therapy": p["targeted_therapy"],
                    "relapse": p["relapse"],
                    "gtr_resection": p["gtr_resection"],
                    "variant": p["variant"],
                }
                for p in cohort[:10]
            ],
        })

    return _json_safe({
        "atlas": "Hereditary-Rhabdomyosarcoma-Predisposition-Atlas",
        "breakdown": breakdown,
    })


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Rhabdomyosarcoma-Predisposition-Atlas."""
    definitions = {}
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        definitions[g] = {
            "gene": g,
            "locus": gdef["locus"],
            "protein_size": gdef["protein_size"],
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
        "atlas": "Hereditary-Rhabdomyosarcoma-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "TP53_LFS_AVOID_RADIATION_ABSOLUTELY": (
                "AVOID RADIATION ABSOLUTELY if TP53 germline LFS -- omit RMS involved-field RT; "
                "Proton therapy if RT absolutely unavoidable (lower exit dose); "
                "WBMRI Toronto Protocol annually NOT CT/PET; "
                "R337H: Brazilian founder mutation 1/300 carrier; "
                "LFS RMS: predominantly ERMS and spindle cell/sclerosing subtypes"
            ),
            "DICER1_CERVICAL_BLADDER_ERMS": (
                "Cervical ERMS (adolescent female) PATHOGNOMONIC DICER1 germline LOF; "
                "Bladder/vaginal ERMS PATHOGNOMONIC DICER1; botryoid polypoid appearance; "
                "Fertility-sparing surgery (trachelectomy/polypectomy) PREFERRED where oncologically feasible; "
                "AVOID pelvic/vaginal RT (developing organs + fertility preservation); "
                "RNase IIIb somatic hotspot (E1705/D1709/E1813) PATHOGNOMONIC second hit; "
                "Low-dose VAC: actinomycin-D + vincristine +/- cyclophosphamide"
            ),
            "RB1_SECONDARY_RMS_CDK46I": (
                "Secondary RMS 10-15x elevated post-RT in RB1 germline (latency 10-20yr); "
                "AVOID orbital radiation if alternative (chemoreduction, cryotherapy, laser) feasible; "
                "CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) INACTIVE in RB1-null tumours; "
                "Bilateral retinoblastoma PATHOGNOMONIC germline RB1; "
                "Trilateral retinoblastoma (+ pinealoblastoma) PATHOGNOMONIC germline; "
                "Annual WB-MRI secondary cancer surveillance from age 10yr"
            ),
            "HRAS_COSTELLO_ERMS_HCM": (
                "HRAS germline GOF = Costello syndrome; RMS 15-20% cumulative by age 20yr PATHOGNOMONIC; "
                "Predominantly ERMS (bladder botryoid, orbital, paravaginal); "
                "HCM 60-70% LIFE-THREATENING -- annual echocardiogram MANDATORY; "
                "G12S most common germline HRAS (~80% Costello); "
                "Cardiac function monitoring before each VAC cycle; avoid high-intensity exercise; "
                "Perinasal/perianal papillomas PATHOGNOMONIC Costello onset 2-3yr"
            ),
            "BRCA2_FA_D1_RMS_MODIFIED_VAC": (
                "FA-D1 (biallelic BRCA2): embryonal RMS PATHOGNOMONIC; bilateral Wilms PATHOGNOMONIC; "
                "MODIFIED VAC: AVOID cyclophosphamide (severe BMF in FA-D1); "
                "AVOID ifosfamide (alkylating agent -> BMF); "
                "Cisplatin + actinomycin-D + vincristine preferred (HRD sensitivity); "
                "DEB/MMC chromosomal fragility PATHOGNOMONIC FA -- ALL potential donors tested; "
                "SIBLING DONOR EXCLUSION MANDATORY before HSCT evaluation"
            ),
            "SMARCB1_ATRT_INI1_TAZEMETOSTAT": (
                "ATRT under 3yr PATHOGNOMONIC SMARCB1 germline LOF (RTPS2); "
                "MRT extrarenal PATHOGNOMONIC SMARCB1 biallelic loss; "
                "INI1 IHC nuclear loss PATHOGNOMONIC SMARCB1 biallelic inactivation; "
                "Tazemetostat (EZH2 inhibitor) FDA 2020 for SMARCB1-loss epithelioid sarcoma; "
                "HSCT consolidation: curative intent for ATRT/MRT; "
                "EZH2 dependency: SMARCB1 loss -> H3K27me3 accumulation -> EZH2 vulnerability"
            ),
            "PTPN11_NOONAN_JMML_DISTINCTION": (
                "PTPN11 GOF (germline) = Noonan syndrome; JMML 200-500x elevated; "
                "Noonan-JMML: often self-limited (spontaneous remission 30-50%) -- watch-and-wait; "
                "Sporadic JMML (somatic PTPN11 E76K): more aggressive -- HSCT REQUIRED; "
                "Pulmonary valve stenosis 50-60% PATHOGNOMONIC Noonan; HCM 20-30%; "
                "Annual echocardiogram; cardiac clearance mandatory before RMS chemotherapy; "
                "NS-ML/LEOPARD: LOF PTPN11 (E69K, T279C) -- lentigines PATHOGNOMONIC -- DISTINCT from Noonan GOF"
            ),
            "CASCADE_RMS_PREDISPOSITION": (
                "TP53/NF1/RB1/HRAS/PTPN11: AD -- 50% risk first-degree relatives -- germline panel cascade; "
                "DICER1: AD LOF -- 50% first-degree risk; cervical/bladder ERMS in family -> DICER1 panel; "
                "SMARCB1: AD LOF -- 50% risk; ATRT/MRT in child -> germline SMARCB1 testing; "
                "BRCA2 FA-D1: BOTH parents obligate heterozygous BRCA2 (HBOC counselling for parents); "
                "DEB/MMC ALL siblings MANDATORY pre-HSCT evaluation"
            ),
        },
        "cascade_testing_rule": (
            "TP53/NF1/DICER1/RB1/HRAS/PTPN11/SMARCB1: AD -- 50% risk first-degree relatives -- germline panel. "
            "DICER1: cervical/bladder ERMS + family history -> germline panel MANDATORY. "
            "RB1: bilateral retinoblastoma -> germline testing MANDATORY; unilateral onset <12mo suspicious. "
            "HRAS Costello: G12S (~95% de novo); test parents of Costello child (parental mosaicism ~5%). "
            "SMARCB1: ATRT/MRT under 3yr -> germline SMARCB1 testing MANDATORY; "
            "concurrent bilateral ATRT/MRT = germline RTPS2 essentially certain. "
            "BRCA2 FA-D1: both parents obligate BRCA2 heterozygous (HBOC counselling); "
            "siblings 25% FA-D1 risk + 50% HBOC carrier; DEB/MMC ALL siblings before HSCT MANDATORY."
        ),
    })


def _json_safe(obj):
    """Ensure all values are JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2)[:3000])
    print("\n--- breakdown (first gene) ---")
    bd = generate_breakdown()
    print(json.dumps(bd["breakdown"][0], indent=2)[:2000])
