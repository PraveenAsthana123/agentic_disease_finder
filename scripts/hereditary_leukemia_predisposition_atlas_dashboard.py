#!/usr/bin/env python3
"""Hereditary-Leukemia-Predisposition-Atlas -- Complete 8-Gene Reference
RUNX1   (RUNX family transcription factor 1; 453aa; 21q22.12; AD LOF;
         Familial platelet disorder with predisposition to myeloid malignancy (FPD/AML);
         AML/MDS 35-44% lifetime; PATHOGNOMONIC dense granule defect + thrombocytopenia;
         NO FAMILY HSCT DONOR without germline exclusion; seed SEED_BASE+0) .
GATA2   (GATA binding protein 2; 480aa; 3q21.3; AD LOF;
         GATA2 deficiency / Emberger syndrome / MonoMAC syndrome;
         MDS/AML 80% lifetime; monosomy 7 PATHOGNOMONIC GATA2;
         lymphedema + immunodeficiency; HSCT curative; seed SEED_BASE+1) .
CEBPA   (CCAAT/enhancer binding protein alpha; 358aa; 19q13.11; AD biallelic;
         Biallelic CEBPA AML; germline N-terminal bZIP + somatic C-terminal bZIP;
         AML 90-100% biallelic; GOOD PROGNOSIS with 7+3; HSCT debated;
         seed SEED_BASE+2) .
DDX41   (DEAD-box helicase 41; 622aa; 5q35.3; AD LOF;
         DDX41 AML/MDS predisposition; D140G/Y259C most common germline;
         LATE-ONSET 60-70yr AML/MDS; R2-splicing domain mutations;
         SIBLING DONOR EXCLUSION mandatory; seed SEED_BASE+3) .
ETV6    (ETS variant transcription factor 6; 452aa; 12p13.2; AD LOF;
         Hereditary thrombocytopenia-2 (THC5) / childhood ALL predisposition;
         ALL 20-30x elevated PATHOGNOMONIC ETV6 germline; thrombocytopenia;
         ETV6-RUNX1 somatic fusion independent of germline ETV6 LOF;
         seed SEED_BASE+4) .
SAMD9L  (Sterile alpha motif domain-containing 9-like; 1589aa; 7q21.2; AD LOF;
         Ataxia-pancytopenia syndrome (ATXPC) / MIRAGE-like;
         monosomy 7 PATHOGNOMONIC adaptive reversion to disomy 7;
         bone marrow failure + cerebellar ataxia; HSCT for MDS;
         seed SEED_BASE+5) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         t-AML/MDS post-alkylator/RT; ALL in LFS 5-10%;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annual;
         seed SEED_BASE+6) .
BRCA2   (Breast cancer gene 2; 3418aa; 13q12.3; Biallelic AR LOF FA-D1 / AD LOF HBOC;
         FA complementation group D1;
         AML PATHOGNOMONIC FA-D1 (biallelic); AVOID alkylating agents;
         SIBLING DONOR EXCLUSION MANDATORY; seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3286-3293)
"""
import random

SEED_BASE = 3286

ATLAS_GENES = [
    {
        "gene": "RUNX1",
        "protein": (
            "RUNX1 -- 21q22.12 Autosomal-Dominant-LOF -- 453aa -- "
            "CBFalpha2-51kDa-Runt-Domain-TF-FPD-AML-35-44pct-"
            "NO-FAMILY-HSCT-WITHOUT-GERMLINE-EXCLUSION-OMIM-601399"
        ),
        "locus": "21q22.12",
        "protein_size": (
            "453 aa / 51 kDa / 21q22.12 RUNX1 encodes RUNX family transcription factor 1 (CBFalpha2): "
            "STRUCTURE: "
            "  453 aa / 51 kDa; master haematopoietic transcription factor; "
            "  Runt homology domain (RHD, aa 50-178): DNA binding + CBFbeta heterodimerisation; "
            "  Transactivation domain (TAD, aa 291-371): HDAC/HAT recruitment; "
            "  RUNX1 LOF -> disrupted CBF (core binding factor) complex -> haematopoietic maturation block; "
            "  Somatic RUNX1 second hit: frameshift/nonsense (N-terminal) or missense/dominant-negative (RHD); "
            "  RUNX1-RUNX1T1 (t(8;21)): somatic AML-defining; independent of germline RUNX1 LOF; "
            "FAMILIAL PLATELET DISORDER WITH PREDISPOSITION TO MYELOID MALIGNANCY (FPD/AML): "
            "  OMIM 601399; AD; first described 1999 (Song et al.); "
            "  Thrombocytopenia: 50-90% of FPD/AML (mild-moderate, 50-150 x10^9/L); "
            "  Platelet dysfunction: dense granule defect (absence/reduction); aggregation impaired; "
            "    Platelet dense granule defect: PATHOGNOMONIC FPD/AML (electron microscopy); "
            "  AML/MDS: 35-44% lifetime cumulative risk; "
            "    AML subtypes: M0, M1, M2 (AML-NOS); occasionally M4/M5; "
            "    MDS subtypes: RA, RARS, RCMD, RAEB; "
            "  ALL: <5% FPD/AML (rare); lymphoid predisposition distinct from RUNX1 AML; "
            "  Clonal haematopoiesis: RUNX1 germline + somatic RUNX1 second hit detectable years before AML; "
            "HSCT DONOR EXCLUSION (RUNX1): "
            "  NO FAMILY HSCT DONOR without germline RUNX1 exclusion testing; "
            "  Sibling donor screening: exclude RUNX1 germline before sibling allograft; "
            "  Matched unrelated donor (MUD) preferred if sibling testing not feasible; "
            "  Post-allograft: monitor donor chimaerism for donor-derived AML (rare but described); "
            "SURVEILLANCE FPD/AML: "
            "  Annual full blood count + peripheral blood morphology; "
            "  Bone marrow biopsy if cytopenia worsens or clonal CH detected; "
            "  Flow cytometry panel: CD34, CD117, CD33, CD13 for MDS immunophenotype; "
            "  Yearly CBC from diagnosis; BM bx if unexplained cytopenia or CH variant allele fraction >5%; "
        ),
        "inheritance": "Autosomal dominant LOF; ~50% de novo; variable penetrance for thrombocytopenia/AML; families with same variant show variable expressivity; clonal haematopoiesis precedes AML by years",
        "cancer_risk": "AML/MDS 35-44% lifetime; thrombocytopenia 50-90%; ALL <5%; clonal haematopoiesis detectable decades before AML; somatic acquisition of second RUNX1 hit or TET2/DNMT3A drives progression",
        "pathognomonic": "Dense platelet granule defect PATHOGNOMONIC FPD/AML (electron microscopy); mild-moderate thrombocytopenia PATHOGNOMONIC FPD/AML; RUNX1-RUNX1T1 somatic fusion independent of germline",
        "surveillance_key": "Annual CBC + morphology; BM biopsy if unexplained cytopenia or CH variant >5%; NO family HSCT donor without RUNX1 germline exclusion; monitor donor chimaerism post-allograft; cascade testing 50% risk",
        "key_distinctions": [
            "FPD-AML-35-44PCT-LIFETIME-HIGHEST-RISK",
            "DENSE-GRANULE-DEFECT-PATHOGNOMONIC",
            "NO-FAMILY-HSCT-WITHOUT-RUNX1-GERMLINE-EXCLUSION",
            "THROMBOCYTOPENIA-50-90PCT",
            "CLONAL-CH-PRECEDES-AML-BY-YEARS",
            "SOMATIC-RUNX1-SECOND-HIT-DRIVES-PROGRESSION",
        ],
    },
    {
        "gene": "GATA2",
        "protein": (
            "GATA2 -- 3q21.3 Autosomal-Dominant-LOF -- 480aa -- "
            "GATA2-50kDa-ZnFinger-TF-Emberger-MonoMAC-MDS-AML-80pct-"
            "Monosomy7-PATHOGNOMONIC-Lymphedema-HSCT-Curative-OMIM-137295"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "480 aa / 50 kDa / 3q21.3 GATA2 encodes GATA binding protein 2 (zinc finger transcription factor): "
            "STRUCTURE: "
            "  480 aa / 50 kDa; zinc finger transcription factor; "
            "  N-terminal zinc finger (NZF, aa 217-240): DNA binding assistance; "
            "  C-terminal zinc finger (CZF, aa 318-341): primary DNA (GATA motif) binding; "
            "  Transactivation domain (aa 1-74): cofactor recruitment; "
            "  GATA2 LOF -> haematopoietic stem cell (HSC) exhaustion -> aplasia/MDS/AML; "
            "  GATA2 T354M (most common European); R361H (most common Asian); 5' enhancer deletions; "
            "  GATA2 zinc finger 1 (ZF1) mutations: predominantly Emberger/lymphedema phenotype; "
            "  GATA2 zinc finger 2 (ZF2) mutations: predominantly MonoMAC/MDS/AML phenotype; "
            "GATA2 DEFICIENCY -- CLINICAL SPECTRUM: "
            "  OMIM 137295; Emberger syndrome (lymphedema + MDS/AML); MonoMAC syndrome; "
            "  Familial MDS/AML (GATA2-related); "
            "  Monosomy 7 PATHOGNOMONIC in GATA2 MDS (70-80% of GATA2 MDS = monosomy 7); "
            "  Lymphedema: peripheral (lower limb) 70% GATA2; congenital lymphedema deformans; "
            "  MonoMAC syndrome: monocytopenia + NK cell deficiency + B-cell lymphopenia + MAC infection; "
            "    Mycobacterial avium complex (MAC) infection PATHOGNOMONIC MonoMAC; "
            "    HPV-related warts + viral skin lesions in MonoMAC; "
            "    PCP (Pneumocystis jirovecii pneumonia) risk; "
            "MDS/AML IN GATA2: "
            "  MDS 80% cumulative by age 40yr; AML transition 50% of MDS cases; "
            "  Hypocellular MDS: PATHOGNOMONIC GATA2 (BM cellularity <10-25%); "
            "  Progression: aplasia -> hypocellular MDS -> monosomy 7 MDS -> AML; "
            "  Refractory anaemia with excess blasts (RAEB): highest AML transition risk; "
            "HSCT CURATIVE (GATA2): "
            "  HSCT: only curative treatment for GATA2 haematological disease; "
            "  Timing: before AML transformation; at MDS/aplasia with >5% blasts or symptomatic; "
            "  Conditioning: RIC (reduced-intensity) preferred (pulmonary/renal complications); "
            "  Outcome: OS 70-80% at 2yr post-allograft in MDS/aplasia phase; "
            "  Lymphedema: does NOT resolve post-HSCT (non-haematological manifestation); "
        ),
        "inheritance": "Autosomal dominant LOF; ~50% de novo; near-complete penetrance for haematological disease by age 40yr; variable expressivity (lymphedema vs MonoMAC vs MDS/AML); GATA2 ZF2 mutations: predominantly myeloid",
        "cancer_risk": "MDS 80% cumulative by age 40yr; AML 40-50% (transition from MDS); monosomy 7 PATHOGNOMONIC; lymphoma rare; solid tumours (HPV-related cervical/anogenital) in MonoMAC",
        "pathognomonic": "Monosomy 7 PATHOGNOMONIC GATA2 MDS (70-80%); hypocellular MDS PATHOGNOMONIC GATA2; monocytopenia + NK deficiency PATHOGNOMONIC MonoMAC; MAC infection in immunocompetent young person PATHOGNOMONIC",
        "surveillance_key": "Annual CBC + BM biopsy from diagnosis; HSCT at MDS/aplasia (before AML); RIC conditioning; lymphedema does NOT resolve post-HSCT; MAC prophylaxis if MonoMAC; HPV vaccination MANDATORY; cascade 50% risk",
        "key_distinctions": [
            "GATA2-MDS-80PCT-CUMULATIVE-BY-40YR",
            "MONOSOMY-7-PATHOGNOMONIC-GATA2-MDS",
            "HYPOCELLULAR-MDS-PATHOGNOMONIC-GATA2",
            "MONOPENIA-NK-DEFICIENCY-MONOMAC-PATHOGNOMONIC",
            "HSCT-CURATIVE-LYMPHEDEMA-DOES-NOT-RESOLVE",
            "MAC-INFECTION-IMMUNOCOMPETENT-YOUNG-PATHOGNOMONIC",
        ],
    },
    {
        "gene": "CEBPA",
        "protein": (
            "CEBPA -- 19q13.11 Autosomal-Dominant-biallelic -- 358aa -- "
            "CEBPalpha-42kDa-bZIP-TF-Biallelic-AML-90-100pct-GOOD-PROGNOSIS-"
            "Germline-Nterminal-Somatic-Cterminal-OMIM-116897"
        ),
        "locus": "19q13.11",
        "protein_size": (
            "358 aa / 42 kDa / 19q13.11 CEBPA encodes CCAAT/enhancer-binding protein alpha (C/EBPalpha): "
            "STRUCTURE: "
            "  358 aa / 42 kDa; basic leucine zipper (bZIP) transcription factor; "
            "  Transactivation domain 1 (TAD1, aa 1-58); "
            "  Transactivation domain 2 (TAD2, aa 107-135); "
            "  Basic DNA-binding region (aa 279-321): CAAT motif binding; "
            "  Leucine zipper (aa 321-358): dimerisation domain; "
            "  N-terminal frame-shift mutations: truncate p30 isoform (dominant negative); "
            "  C-terminal in-frame mutations: disrupt bZIP DNA binding/dimerisation; "
            "BIALLELIC CEBPA AML: "
            "  OMIM 116897; germline CEBPA AML predisposition: "
            "  Germline N-terminal frameshift (first hit) + somatic C-terminal in-frame mutation (second hit); "
            "  PATHOGNOMONIC biallelic CEBPA pattern: germline N-term + somatic C-term; "
            "  AML LIFETIME RISK: 90-100% biallelic CEBPA germline; near-certain AML; "
            "  AML SUBTYPE: FAB M1 or M2 (FAB M4/M5 rare); CD34+CD117+CD33+ immunophenotype; "
            "  AML onset: 20-40yr (later than other germline AML syndromes); "
            "  De novo AML: biallelic CEBPA = 9% of CN-AML (cytogenetically normal); "
            "GOOD PROGNOSIS (BIALLELIC CEBPA): "
            "  5yr OS: 60-80% with standard 7+3 induction (daunorubicin + cytarabine); "
            "  CR rate: 85-95% first induction; "
            "  Relapse: second AML often new CEBPA C-terminal variant (clonally unrelated relapses); "
            "    New C-terminal mutation at relapse: PATHOGNOMONIC CEBPA AML recurrence behaviour; "
            "  HSCT in CR1: debated (good prognosis with chemotherapy alone in biallelic CEBPA); "
            "    Consensus 2023: HSCT NOT routinely indicated CR1 biallelic CEBPA; consolidate 3-4x HiDAC; "
            "    HSCT recommended: CR2 or refractory biallelic CEBPA AML; "
            "MONOALLELIC CEBPA AML: "
            "  C-terminal monoallelic CEBPA: sporadic AML (no germline); "
            "    Monoallelic does NOT confer germline predisposition; "
            "    Germline testing required ONLY if N-terminal frameshift (family screening); "
        ),
        "inheritance": "Autosomal dominant LOF (germline N-terminal frameshift) with somatic second-hit (C-terminal); biallelic CEBPA AML: near-complete penetrance; monoallelic CEBPA N-terminal germline: 50% offspring at risk",
        "cancer_risk": "AML 90-100% lifetime (biallelic CEBPA); predominantly CN-AML FAB M1/M2; second de novo AML (new C-terminal) at relapse; non-haematological tumours rare",
        "pathognomonic": "Biallelic CEBPA pattern PATHOGNOMONIC germline (N-terminal germline + somatic C-terminal second hit); new C-terminal variant at relapse PATHOGNOMONIC CEBPA AML clonal behaviour; GOOD PROGNOSIS AML",
        "surveillance_key": "Annual CBC from germline CEBPA N-terminal diagnosis; BM biopsy if cytopenia; 7+3 standard; HSCT NOT CR1 (biallelic good prognosis); HSCT at CR2; germline N-terminal: test family; cascade 50% risk",
        "key_distinctions": [
            "BIALLELIC-CEBPA-AML-90-100PCT-NEAR-CERTAIN",
            "GERMLINE-N-TERMINAL-SOMATIC-C-TERMINAL-PATHOGNOMONIC",
            "GOOD-PROGNOSIS-60-80PCT-5YR-OS",
            "HSCT-NOT-CR1-BIALLELIC-CEBPA",
            "NEW-C-TERMINAL-AT-RELAPSE-CLONALLY-UNRELATED",
            "MONOALLELIC-C-TERMINAL-NOT-GERMLINE",
        ],
    },
    {
        "gene": "DDX41",
        "protein": (
            "DDX41 -- 5q35.3 Autosomal-Dominant-LOF -- 622aa -- "
            "DDX41-68kDa-DEAD-Box-Helicase-InnateSensing-LATE-ONSET-AML-60-70yr-"
            "D140G-Y259C-SIBLING-EXCLUSION-MANDATORY-OMIM-608280"
        ),
        "locus": "5q35.3",
        "protein_size": (
            "622 aa / 68 kDa / 5q35.3 DDX41 encodes DEAD-box helicase 41 (ATP-dependent RNA helicase): "
            "STRUCTURE: "
            "  622 aa / 68 kDa; DEAD-box helicase family; "
            "  N-terminal domain (aa 1-173): R2 splicing complex interaction; "
            "  DEAD-box helicase domain (aa 174-489): DEADc + HELICc motifs; "
            "    DEADc D140 residue: D140G most common germline variant (European/Japanese); "
            "    ATP-binding P-loop (aa 174-182): ATPase activity; "
            "    RNA helicase motif II DEAD: D-E-A-D sequence; "
            "  C-terminal ZNF_MYND zinc finger (aa 490-529): transcription regulation; "
            "  Y259C: most common germline variant (frameshift behaviour in RNA helicase domain); "
            "  DDX41 function: innate immune sensing of cytosolic DNA/RNA (cGAS-STING); "
            "    R2 spliceosome assembly; pre-mRNA splicing; "
            "    DDX41 LOF -> impaired innate immune + splicing -> haematopoietic transformation; "
            "LATE-ONSET AML/MDS (DDX41): "
            "  DDX41 AML: predominantly LATE-ONSET (median 65-70yr); "
            "    Pediatric DDX41 AML very rare (unlike RUNX1/GATA2/CEBPA/ETV6); "
            "  AML risk: 40-50% lifetime cumulative (lower penetrance than RUNX1/GATA2/CEBPA); "
            "  MDS risk: 30-40% cumulative; "
            "  AML subtypes: AML-NOS; occasional MDS/AML-MRC (with somatic SF3B1/SRSF2 co-mutations); "
            "  Somatic DDX41 second hit: R525H (most common somatic second hit in DDX41 AML); "
            "    Biallelic DDX41 (germline D140G/Y259C + somatic R525H): 80-90% of DDX41 AML cases; "
            "SIBLING DONOR EXCLUSION (DDX41): "
            "  NO SIBLING HSCT DONOR without DDX41 germline exclusion testing; "
            "  D140G prevalence: ~1/1000 European population (carrier); "
            "  Sibling carrier risk: 50% -> donor-derived AML risk post-allograft; "
            "TREATMENT DDX41 AML: "
            "  Standard 7+3 induction; venetoclax + azacitidine (HMA-ven) promising in unfit patients; "
            "  Allograft: recommended if fit (MUD preferred; sibling only after testing); "
            "  Prognosis: intermediate (OS 2-3yr median in elderly); younger patients better outcome; "
        ),
        "inheritance": "Autosomal dominant LOF; D140G/Y259C most common germline; autosomal dominant with incomplete penetrance (40-50%); paternal imprinting NOT established; late-onset disease distinguishes DDX41 from other germline AML syndromes",
        "cancer_risk": "AML 40-50% lifetime (predominantly 60-70yr); MDS 30-40%; NHL rare; solid tumours not established",
        "pathognomonic": "Late-onset AML/MDS 60-70yr in germline DDX41 PATHOGNOMONIC; R525H somatic second hit PATHOGNOMONIC biallelic DDX41 AML; D140G/Y259C germline most common European/Japanese",
        "surveillance_key": "Annual CBC + BM biopsy from age 50yr; DDX41 AML: standard 7+3; HMA-ven unfit; allograft if fit; NO sibling HSCT donor without DDX41 germline exclusion; cascade testing all first-degree relatives",
        "key_distinctions": [
            "LATE-ONSET-AML-60-70YR-DISTINCT-DDX41",
            "D140G-Y259C-MOST-COMMON-GERMLINE",
            "R525H-SOMATIC-SECOND-HIT-80-90PCT",
            "SIBLING-DONOR-EXCLUSION-MANDATORY-DDX41",
            "AML-40-50PCT-LIFETIME-INCOMPLETE-PENETRANCE",
            "HMA-VENETOCLAX-PROMISING-UNFIT",
        ],
    },
    {
        "gene": "ETV6",
        "protein": (
            "ETV6 -- 12p13.2 Autosomal-Dominant-LOF -- 452aa -- "
            "ETV6-57kDa-ETS-TF-Hereditary-Thrombocytopenia-2-ALL-20-30x-PATHOGNOMONIC-"
            "ETV6-RUNX1-Somatic-Independent-OMIM-600618"
        ),
        "locus": "12p13.2",
        "protein_size": (
            "452 aa / 57 kDa / 12p13.2 ETV6 encodes ETS variant transcription factor 6 (TEL): "
            "STRUCTURE: "
            "  452 aa / 57 kDa; ETS family transcription factor; "
            "  PNT (pointed) domain (aa 1-133): oligomerisation / protein-protein interaction; "
            "    Germline P214L / R399C / P406A: common pathogenic germline variants; "
            "  ETS DNA-binding domain (aa 337-418): GGAA/T motif binding; "
            "  ETV6 LOF -> impaired megakaryocyte maturation (thrombocytopenia) + lymphoid predisposition; "
            "  ETV6-RUNX1 somatic translocation t(12;21)(p13;q22): "
            "    most common childhood ALL translocation (~25% of B-ALL); "
            "    PATHOGNOMONIC somatic; independent of germline ETV6 LOF; "
            "    Do NOT confuse somatic ETV6-RUNX1 fusion with germline ETV6 predisposition; "
            "HEREDITARY THROMBOCYTOPENIA-2 (THC5 / ETV6-THC): "
            "  OMIM 616216; germline ETV6 LOF; AD; "
            "  Thrombocytopenia: mild-moderate (50-150 x10^9/L); "
            "  Macrocytosis: red cell macrocytosis in ETV6 THC (PATHOGNOMONIC); "
            "  Bleeding: mild; rarely requires platelet transfusion; "
            "  No dysmorphic features; no solid tumour predisposition established; "
            "ALL PREDISPOSITION (ETV6): "
            "  ALL 20-30x elevated vs general population (PATHOGNOMONIC ETV6 germline); "
            "  B-ALL: predominantly B-lineage (CD19+CD10+ precursor B); "
            "  ETV6-RUNX1-like gene expression: germline ETV6 ALL may show ETV6-RUNX1-like profile; "
            "  Childhood onset: median 3-6yr ALL in ETV6 germline (similar to sporadic ALL peak); "
            "  Treatment: standard risk ALL protocols (AALL0434 / COG-based); "
            "    Outcome: ETV6 germline ALL: good prognosis (ETV6-RUNX1-like profile favorable); "
            "  Somatic co-mutations: JAK2 V617F, SH2B3 LOF in ETV6 ALL progression; "
        ),
        "inheritance": "Autosomal dominant LOF; P214L/R399C/P406A most common pathogenic germline; ~50% de novo; incomplete penetrance for ALL; near-complete penetrance for thrombocytopenia",
        "cancer_risk": "ALL 20-30x elevated (PATHOGNOMONIC ETV6 germline); B-ALL predominantly; thrombocytopenia >90%; macrocytosis; T-ALL rare; solid tumours not established",
        "pathognomonic": "ALL 20-30x elevated PATHOGNOMONIC ETV6 germline; red cell macrocytosis PATHOGNOMONIC ETV6-THC; ETV6-RUNX1 somatic t(12;21) independent of germline ETV6 LOF; do NOT conflate",
        "surveillance_key": "Annual CBC from diagnosis; ALL: standard risk COG-ALL protocol; germline ETV6 ALL: good prognosis; red cell macrocytosis monitor; cascade 50% risk; somatic ETV6-RUNX1 ALL: no germline testing indicated",
        "key_distinctions": [
            "ALL-20-30X-ELEVATED-PATHOGNOMONIC-ETV6",
            "RED-CELL-MACROCYTOSIS-PATHOGNOMONIC-ETV6-THC",
            "ETV6-RUNX1-SOMATIC-INDEPENDENT-OF-GERMLINE-ETV6",
            "MILD-THROMBOCYTOPENIA-GT90PCT",
            "GOOD-PROGNOSIS-B-ALL-ETV6-GERMLINE",
            "P214L-R399C-P406A-MOST-COMMON-PATHOGENIC",
        ],
    },
    {
        "gene": "SAMD9L",
        "protein": (
            "SAMD9L -- 7q21.2 Autosomal-Dominant-LOF -- 1589aa -- "
            "SAMD9L-173kDa-SAM-Domain-Interferon-Induced-ATXPC-Monosomy7-PATHOGNOMONIC-"
            "Adaptive-Reversion-Disomy7-BM-Failure-Cerebellar-Ataxia-HSCT-OMIM-610456"
        ),
        "locus": "7q21.2",
        "protein_size": (
            "1589 aa / 173 kDa / 7q21.2 SAMD9L encodes sterile alpha motif domain-containing 9-like protein: "
            "STRUCTURE: "
            "  1589 aa / 173 kDa; SAM domain protein; "
            "  SAM (sterile alpha motif) domain (aa 1-67): protein-protein interaction; "
            "  SAMD9L: interferon-stimulated gene; innate immune function; "
            "  SAMD9L promotes translational repression; antiproliferative effect; "
            "  SAMD9L LOF: unrestrained cell proliferation + impaired haematopoiesis; "
            "  SAMD9L GOF (gain-of-function): rare; associated with distinct phenotype; "
            "ATAXIA-PANCYTOPENIA SYNDROME (ATXPC): "
            "  OMIM 159550; SAMD9L GOF or LOF; AD; "
            "  Clinical triad: cerebellar ataxia + pancytopenia + monosomy 7 MDS; "
            "  ATXPC PATHOGNOMONIC clinical triad; "
            "MONOSOMY 7 IN SAMD9L: "
            "  Monosomy 7: somatic loss of chromosome 7 containing the mutant SAMD9L allele; "
            "  ADAPTIVE REVERSION: monosomy 7 removes pathogenic SAMD9L allele -> 'escape'; "
            "  Monosomy 7 is PATHOGNOMONIC adaptive mechanism in SAMD9L (selective advantage); "
            "  SAMD9L monosomy 7 MDS: PATHOGNOMONIC in young patient with pancytopenia + ataxia; "
            "  Uniparental isodisomy 7 (UPD7): rare alternative reversion mechanism; "
            "  Monosomy 7 = HIGH-RISK MDS -> AML transformation; "
            "BONE MARROW FAILURE (SAMD9L): "
            "  Aplasia/hypoplasia: 60-70% of SAMD9L; "
            "  MDS: 50-60% cumulative (predominantly monosomy 7 subtype); "
            "  AML: 20-30% from MDS transition; "
            "HSCT INDICATION (SAMD9L): "
            "  HSCT: indicated for MDS/aplasia in SAMD9L; "
            "  Timing: at MDS diagnosis or progressive aplasia; "
            "  Outcome: HSCT OS 60-70% at 5yr (limited data); "
            "  Cerebellar ataxia: does NOT improve post-HSCT (neurological non-haematological); "
            "  Conditioning: RIC preferred (CNS toxicity risk); "
        ),
        "inheritance": "Autosomal dominant LOF (or GOF for ataxia-pancytopenia); de novo or familial; variable penetrance; monosomy 7 adaptive reversion leads to somatic mosaicism",
        "cancer_risk": "MDS 50-60% (predominantly monosomy 7); AML 20-30% from MDS; aplasia 60-70%; lymphoma rare; cerebellar ataxia non-haematological",
        "pathognomonic": "Monosomy 7 adaptive reversion PATHOGNOMONIC SAMD9L (removes mutant allele); clinical triad cerebellar ataxia + pancytopenia + monosomy 7 PATHOGNOMONIC ATXPC; monosomy 7 in young patient with ataxia",
        "surveillance_key": "Annual CBC + BM biopsy; HSCT at MDS/aplasia; cerebellar ataxia does NOT resolve post-HSCT; RIC conditioning; monitor somatic reversion clones; annual neurological assessment; cascade 50% risk",
        "key_distinctions": [
            "MONOSOMY-7-ADAPTIVE-REVERSION-PATHOGNOMONIC-SAMD9L",
            "ATXPC-TRIAD-CEREBELLAR-ATAXIA-PANCYTOPENIA-MONO7",
            "CEREBELLAR-ATAXIA-DOES-NOT-RESOLVE-POST-HSCT",
            "MDS-50-60PCT-CUMULATIVE",
            "HSCT-INDICATED-AML-TRANSFORMATION-RISK",
            "RIC-CONDITIONING-CNS-TOXICITY-RISK",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-t-AML-MDS-ALL-5-10pct-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour suppressor protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; transcription factor; tetramer in active form; "
            "  N-terminal transactivation domains (TAD1 aa 1-40, TAD2 aa 40-61); "
            "  DNA-binding domain (DBD, aa 102-292): most mutations cluster here; "
            "  Tetramerisation domain (aa 323-356): oligomerisation; "
            "  R175H, R248W, R248Q, R273H, R273C: hotspot dominant-negative GOF mutations; "
            "LI-FRAUMENI SYNDROME (LFS) AND LEUKAEMIA: "
            "  OMIM 151623; LFS: sarcoma + brain tumour + ACC + breast <45yr; "
            "  ALL in LFS: 5-10% of LFS tumour spectrum; childhood B-ALL or T-ALL; "
            "  t-AML/MDS in LFS: therapy-related leukaemia post-alkylator/RT treatment; "
            "  t-AML: TP53 germline + prior alkylating agent or RT exposure -> t-AML; "
            "    Complex karyotype t-AML (TP53-mutant AML): PATHOGNOMONIC TP53 germline prior therapy; "
            "    TP53-mutant AML: very poor prognosis (median OS 4-8 months standard therapy); "
            "  APL (acute promyelocytic leukaemia): NOT associated with germline TP53; "
            "AVOID RADIATION ABSOLUTELY (LFS): "
            "  Germline TP53 LFS: AVOID RADIATION ABSOLUTELY; "
            "  Paediatric ALL cranial RT: OMIT if TP53 germline (radiation-related CNS tumour risk); "
            "  RT for mediastinal lymphoma: reassess if TP53 germline (secondary sarcoma); "
            "  WBMRI Toronto protocol annually (NOT CT/PET); "
            "TP53-MUTANT AML TREATMENT: "
            "  Magrolimab (anti-CD47) + azacitidine: Phase III ENHANCE trial; "
            "  Eprenetapopt (APR-246, p53 reactivator): Phase II/III data; "
            "  Venetoclax + azacitidine: limited benefit TP53-mutant AML; "
            "  Allograft: considered if MRD-negative CR achieved; complex karyotype persists; "
        ),
        "inheritance": "Autosomal dominant LOF; 50% de novo; dominant negative GOF hotspot variants worsen phenotype; near-complete penetrance (>90% lifetime cancer risk); t-AML requires prior alkylator/RT exposure as co-factor",
        "cancer_risk": "LFS: sarcoma 30-50% dominant; brain tumour 15-20%; breast <45yr 25-35%; ALL 5-10%; t-AML/MDS post-therapy; ACC paediatric 50-70% PATHOGNOMONIC; TP53-mutant AML: very poor prognosis",
        "pathognomonic": "Paediatric ACC 50-70% TP53 germline PATHOGNOMONIC; complex karyotype t-AML + TP53 mutation + prior therapy PATHOGNOMONIC; R337H 1/300 South Brazilian founder; WBMRI annually NOT CT/PET",
        "surveillance_key": "WBMRI Toronto annually (NOT CT/PET); AVOID RADIATION ABSOLUTELY; omit cranial RT in ALL if TP53 germline; APR-246 or magrolimab for TP53-mutant AML; annual rapid brain MRI; breast MRI from 20yr; CASCADE 50%",
        "key_distinctions": [
            "LFS-ALL-5-10PCT-B-ALL-T-ALL",
            "T-AML-POST-ALKYLATOR-RT-TP53-GERMLINE",
            "AVOID-RADIATION-ABSOLUTELY-OMIT-CRANIAL-RT",
            "WBMRI-TORONTO-ANNUALLY-NOT-CT-PET",
            "TP53-MUTANT-AML-VERY-POOR-PROGNOSIS",
            "APR-246-MAGROLIMAB-NOVEL-AGENTS",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Biallelic-AR-LOF-FA-D1-AD-LOF-HBOC -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FA-D1-AML-PATHOGNOMONIC-Biallelic-"
            "AVOID-Alkylating-Sibling-Donor-Exclusion-MANDATORY-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes breast cancer susceptibility protein 2: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; one of largest human proteins; genome stability scaffold; "
            "  N-terminal PALB2-binding domain (aa 1-40); "
            "  BRCA2-RAD51 interaction: BRC repeats (aa 1002-2085); "
            "  C-terminal OB folds (aa 2402-3190): ssDNA binding; "
            "  BRCA2 function: HR repair scaffold; loads RAD51 onto ssDNA at DSBs; "
            "  Biallelic BRCA2 LOF: Fanconi anaemia complementation group D1 (FA-D1); "
            "FANCONI ANAEMIA D1 (FA-D1) AND LEUKAEMIA: "
            "  OMIM 605724; biallelic BRCA2 LOF; rarest FA type but most severe malignancy risk; "
            "  AML PATHOGNOMONIC FA-D1 (biallelic BRCA2): 25-35% cumulative by age 10yr; "
            "  ALL: 5-10% FA-D1; B-ALL and T-ALL described; "
            "  Wilms tumour: PATHOGNOMONIC FA-D1; medulloblastoma PATHOGNOMONIC FA-D1; "
            "  RMS embryonal: PATHOGNOMONIC FA-D1 (biallelic BRCA2); "
            "  DEB test (diepoxybutane): chromosomal fragility PATHOGNOMONIC FA; "
            "  MMC (mitomycin C) test: alternative chromosomal fragility confirmation; "
            "AVOID ALKYLATING AGENTS (FA-D1 / BRCA2): "
            "  Cyclophosphamide: AVOID in FA-D1 AML (DNA ICL repair defect -> extreme toxicity); "
            "  Alkylating agent-free AML induction: fludarabine + cytarabine combinations; "
            "  Cisplatin: HRD-sensitive (BRCA2 LOF) but ICL mechanism -> risk in FA-D1; "
            "  HSCT: only curative treatment for FA-D1 AML; RIC conditioning (AVOID cyclophosphamide); "
            "SIBLING DONOR EXCLUSION (BRCA2/FA-D1): "
            "  Sibling HSCT donor: EXCLUDE biallelic BRCA2 FA-D1 status MANDATORY; "
            "  Sibling may carry same biallelic FA-D1 or monoallelic HBOC risk; "
            "  MUD (matched unrelated donor) preferred if sibling testing inconclusive; "
        ),
        "inheritance": "Biallelic AR LOF (FA-D1) + AD LOF HBOC (monoallelic); FA-D1 biallelic BRCA2: extreme rarity (~1% of FA); monoallelic HBOC: 1/400 population; biallelic FA-D1: AML + Wilms + medulloblastoma + RMS",
        "cancer_risk": "FA-D1 biallelic: AML 25-35% by age 10yr PATHOGNOMONIC; Wilms PATHOGNOMONIC; medulloblastoma PATHOGNOMONIC; RMS embryonal PATHOGNOMONIC; HBOC monoallelic: breast 50-85%, ovarian 15-30%",
        "pathognomonic": "AML PATHOGNOMONIC FA-D1 biallelic BRCA2 (25-35% by age 10yr); DEB/MMC chromosomal fragility PATHOGNOMONIC FA; Wilms + medulloblastoma + RMS cluster PATHOGNOMONIC FA-D1; AVOID alkylating agents",
        "surveillance_key": "DEB/MMC test diagnosis; HSCT curative (RIC, AVOID cyclophosphamide); sibling donor exclusion MANDATORY; alkylating-agent-free induction; cisplatin cautiously; annual CBC + BM; HBOC monoallelic: breast/OC surveillance",
        "key_distinctions": [
            "FA-D1-AML-25-35PCT-BY-AGE-10YR-PATHOGNOMONIC",
            "DEB-MMC-CHROMOSOMAL-FRAGILITY-PATHOGNOMONIC-FA",
            "AVOID-ALKYLATING-AGENTS-CYCLOPHOSPHAMIDE",
            "SIBLING-DONOR-EXCLUSION-MANDATORY",
            "WILMS-MEDULLOBLASTOMA-RMS-CLUSTER-FA-D1-PATHOGNOMONIC",
            "HSCT-CURATIVE-RIC-ALKYLATING-FREE",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Simulated tumour types per gene (leukemia context)
# ─────────────────────────────────────────────────────────────────────────────
TUMOUR_TYPES_BY_GENE = {
    "RUNX1":  ["AML-NOS", "MDS-RAEB", "AML-M1", "MDS-RA", "AML-M2"],
    "GATA2":  ["MDS-Monosomy7", "AML", "Aplasia", "MonoMAC", "RAEB-2"],
    "CEBPA":  ["AML-M1-biallelic", "AML-M2-biallelic", "AML-NOS", "AML-refractory", "AML-relapse"],
    "DDX41":  ["AML-NOS", "MDS-MRC", "AML-M2", "MDS-RAEB", "AML-NK"],
    "ETV6":   ["B-ALL", "B-ALL-ETV6RUNX1like", "B-ALL-relapse", "MDS", "AML"],
    "SAMD9L": ["MDS-Monosomy7", "Aplasia", "AML-Monosomy7", "MDS-RAEB", "MDS-hypocellular"],
    "TP53":   ["t-AML-complex-karyotype", "ALL-B", "ALL-T", "t-MDS", "t-AML-therapy-related"],
    "BRCA2":  ["AML-FA-D1", "ALL-FA-D1", "Wilms", "Medulloblastoma", "RMS-embryonal"],
}

VARIANTS_BY_GENE = {
    "RUNX1":  ["p.Arg204Gln", "p.Arg162Gln", "p.Tyr287*", "p.Pro206_Glu235del", "p.Arg201Gln"],
    "GATA2":  ["p.Thr354Met", "p.Arg361His", "p.Leu359Val", "5'enhancer-del", "p.Arg330Gln"],
    "CEBPA":  ["N-term-frameshift+somatic-Cterm-inframe", "p.Gln311*", "p.Lys313Asnfs*4", "p.Arg297Trp", "p.Ile339Val-somatic"],
    "DDX41":  ["p.Asp140Gly", "p.Tyr259Cys", "p.Arg525His-somatic", "p.Glu7*", "p.Val152Gly"],
    "ETV6":   ["p.Pro214Leu", "p.Arg399Cys", "p.Pro406Ala", "p.Thr273Ile", "p.Leu394Phe"],
    "SAMD9L": ["p.Arg986Cys", "p.Ile1736Asn", "p.Gln1433Arg", "p.Ala1448Pro", "p.Thr1764Ile"],
    "TP53":   ["p.Arg175His", "p.Arg248Trp", "p.Arg248Gln", "p.Arg273His", "p.Arg273Cys"],
    "BRCA2":  ["p.Trp31*", "p.Lys3326*", "p.Glu1308*", "IVS7+2T>G", "p.Asn991Ile"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "RUNX1":  [
        "Standard 7+3 induction (daunorubicin 60mg/m2 d1-3 + cytarabine 100mg/m2 d1-7)",
        "Post-remission: allograft (MUD preferred; NO family donor without RUNX1 exclusion)",
        "Thrombocytopenia management: eltrombopag if severe; avoid antiplatelet agents",
        "Clonal haematopoiesis surveillance: annual CBC + BM biopsy if CH variant >5%",
        "Annual platelet function test: ADP/collagen aggregation + dense granule electron microscopy",
    ],
    "GATA2":  [
        "HSCT curative (RIC; timing at MDS/aplasia before AML transformation)",
        "MAC prophylaxis: azithromycin + rifabutin (MonoMAC/monocytopenia)",
        "PCP prophylaxis: TMP-SMX (NK cell deficiency)",
        "HPV vaccination MANDATORY (anogenital malignancy risk in MonoMAC)",
        "G-CSF: limited role (haematopoietic stem cell exhaustion); bridge to HSCT",
    ],
    "CEBPA":  [
        "Standard 7+3 induction (CR 85-95%; good prognosis biallelic CEBPA)",
        "Consolidation: 3-4 cycles HiDAC (cytarabine 3g/m2 q12h d1,3,5)",
        "HSCT NOT indicated in CR1 (biallelic CEBPA good prognosis with chemo alone)",
        "HSCT recommended CR2 or refractory biallelic CEBPA AML",
        "Relapse: new C-terminal CEBPA mutation often new clonal AML; re-induction + allograft",
    ],
    "DDX41":  [
        "Standard 7+3 induction for fit patients",
        "HMA-venetoclax (azacitidine + venetoclax) for unfit or elderly (promising Phase II data)",
        "Allograft: recommended if fit; sibling donor excluded without DDX41 germline testing",
        "MUD preferred to sibling donor in DDX41 germline",
        "Annual CBC + BM biopsy from age 50yr in DDX41 germline carriers (pre-AML surveillance)",
    ],
    "ETV6":   [
        "Standard risk ALL protocol (COG AALL0434 or equivalent; ETV6-RUNX1-like prognosis)",
        "B-ALL: dexamethasone/vincristine/asparaginase/MTX induction (standard risk 4-drug)",
        "Omit cranial RT (standard practice; prophylactic CNS RT not routinely used)",
        "ETV6 ALL relapse: blinatumomab or inotuzumab ozogamicin + salvage",
        "Platelet support: transfuse if thrombocytopenia symptomatic (platelet <20 x10^9/L)",
    ],
    "SAMD9L": [
        "HSCT indicated for MDS/aplasia (timing: at diagnosis or progressive aplasia)",
        "RIC conditioning preferred (AVOID TBI; CNS toxicity risk in cerebellar ataxia)",
        "Fludarabine-based conditioning (FluBu or FluTreo): standard RIC in SAMD9L",
        "Post-HSCT: cerebellar ataxia does NOT improve (neurological non-haematological)",
        "Annual neurological assessment + CBC + BM biopsy pre-HSCT",
    ],
    "TP53":   [
        "ALL: standard risk protocol; OMIT cranial RT (radiation secondary malignancy risk TP53)",
        "t-AML: APR-246 (eprenetapopt) + azacitidine (Phase III ENHANCE); OR magrolimab + aza",
        "Venetoclax: limited benefit TP53-mutant AML (BCL2-independent apoptosis)",
        "Allograft if MRD-negative CR; complex karyotype may persist post-induction",
        "WBMRI Toronto annually (NOT CT/PET); AVOID RADIATION absolutely",
    ],
    "BRCA2":  [
        "FA-D1 AML: alkylating-agent-FREE induction (fludarabine + cytarabine combinations)",
        "HSCT curative (RIC; AVOID cyclophosphamide in conditioning; busulfan-based preferred)",
        "SIBLING DONOR EXCLUSION: exclude biallelic BRCA2 FA-D1 status MANDATORY pre-allograft",
        "DEB / MMC chromosomal fragility test at AML diagnosis to confirm FA-D1",
        "Supportive: G-CSF for neutropenia; EPO for anaemia; transfuse per FA guidelines",
    ],
}

SURVEILLANCE_BY_GENE = {
    "RUNX1":  [
        "Annual full blood count + peripheral blood morphology",
        "BM biopsy if cytopenia worsens or clonal CH variant allele fraction >5%",
        "Platelet function test annually (dense granule electron microscopy; aggregation)",
        "NO family HSCT donor without RUNX1 germline exclusion testing",
        "CASCADE: all first-degree relatives; 50% risk per offspring",
    ],
    "GATA2":  [
        "Annual CBC + BM biopsy from diagnosis",
        "HSCT at MDS/aplasia (before AML); RIC conditioning; timing critical",
        "Lymphoedema monitoring: compression stockings; physiotherapy",
        "MAC + PCP prophylaxis; HPV vaccination; annual dermatology (warts)",
        "Ophthalmology annually; annual pulmonary function (lymphatic lung involvement)",
    ],
    "CEBPA":  [
        "Annual CBC from germline CEBPA N-terminal diagnosis",
        "BM biopsy if unexplained cytopenia",
        "Relapse BM: molecular testing for new C-terminal CEBPA variant (new clone)",
        "Cascade testing: N-terminal frameshift germline -> 50% offspring risk",
        "HSCT NOT CR1; consolidate 3-4x HiDAC; HSCT at CR2",
    ],
    "DDX41":  [
        "Annual CBC + BM biopsy from age 50yr",
        "Somatic variant testing if cytopenia (R525H second-hit detection)",
        "No SIBLING HSCT donor without DDX41 germline exclusion",
        "Cascade testing: D140G/Y259C pathogenic -> 50% offspring risk",
        "Annual haematology review from age 45yr",
    ],
    "ETV6":   [
        "Annual CBC (thrombocytopenia monitoring)",
        "ALL: COG-based surveillance post-therapy (relapse monitoring 3yr post-CR)",
        "Cascade: ETV6 P214L/R399C/P406A germline -> 50% offspring risk",
        "ETV6 ALL-treated patients: annual late-effects assessment",
        "Annual haematology review from birth (if germline known)",
    ],
    "SAMD9L": [
        "Annual CBC + BM biopsy (MDS monitoring)",
        "Annual cerebellar/neurological assessment (ataxia severity scoring)",
        "HSCT referral at MDS or progressive aplasia",
        "Post-HSCT: neurological surveillance (ataxia non-haematological)",
        "Cascade: 50% offspring risk; CNS MRI baseline",
    ],
    "TP53":   [
        "WBMRI Toronto annually (NOT CT/PET)",
        "Annual rapid brain MRI + abdominal MRI",
        "Breast MRI from age 20yr (annual)",
        "OMIT cranial RT in ALL if TP53 germline",
        "CASCADE: all first-degree relatives; 50% risk",
    ],
    "BRCA2":  [
        "DEB/MMC test at AML diagnosis (FA-D1 confirmation)",
        "Annual CBC + BM biopsy from birth (FA-D1 surveillance)",
        "SIBLING DONOR EXCLUSION: biallelic BRCA2 FA-D1 exclusion before allograft",
        "Monoallelic HBOC: breast MRI from age 25yr; ovarian risk-reduction BSO by 35-40yr",
        "Alkylating-agent-free regimen; HSCT RIC (busulfan-based, AVOID cyclophosphamide)",
    ],
}


def _make_patient(gene_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_index]["gene"]
    tumour = rng.choice(TUMOUR_TYPES_BY_GENE[gene])
    variant = rng.choice(VARIANTS_BY_GENE[gene])

    # Age at diagnosis (leukemia context - varies by gene)
    if gene == "DDX41":
        age = rng.randint(58, 75)
    elif gene == "ETV6":
        age = rng.randint(2, 14)
    elif gene == "SAMD9L":
        age = rng.randint(3, 20)
    elif gene == "BRCA2":
        age = rng.randint(1, 10)
    elif gene == "TP53":
        age = rng.randint(4, 35)
    elif gene == "CEBPA":
        age = rng.randint(18, 50)
    elif gene == "GATA2":
        age = rng.randint(10, 38)
    else:  # RUNX1
        age = rng.randint(18, 55)

    # Gene-specific CR rates
    cr_rates = {"RUNX1": 0.78, "GATA2": 0.65, "CEBPA": 0.90, "DDX41": 0.72,
                "ETV6": 0.88, "SAMD9L": 0.62, "TP53": 0.38, "BRCA2": 0.55}
    cr = rng.random() < cr_rates.get(gene, 0.70)

    # Radiation rates (low in leukemia)
    rad_rates = {"RUNX1": 0.05, "GATA2": 0.04, "CEBPA": 0.06, "DDX41": 0.04,
                 "ETV6": 0.10, "SAMD9L": 0.03, "TP53": 0.08, "BRCA2": 0.05}
    radiation = rng.random() < rad_rates.get(gene, 0.05)

    # Targeted therapy rates
    targeted_rates = {"RUNX1": 0.20, "GATA2": 0.30, "CEBPA": 0.10, "DDX41": 0.45,
                      "ETV6": 0.40, "SAMD9L": 0.25, "TP53": 0.60, "BRCA2": 0.50}
    targeted = rng.random() < targeted_rates.get(gene, 0.30)

    # Transplant rates
    tx_rates = {"RUNX1": 0.55, "GATA2": 0.80, "CEBPA": 0.30, "DDX41": 0.55,
                "ETV6": 0.15, "SAMD9L": 0.70, "TP53": 0.45, "BRCA2": 0.65}
    transplant = rng.random() < tx_rates.get(gene, 0.50)

    relapse = rng.random() < (0.25 if cr else 0.70)

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


def _generate_cohort() -> list[dict]:
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
    targeted_pct = round(100 * sum(p["targeted"] for p in cohort) / n)
    transplant_pct = round(100 * sum(p["transplant"] for p in cohort) / n)
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
            "targeted_pct": round(100 * sum(p["targeted"] for p in gp) / gn),
            "transplant_pct": round(100 * sum(p["transplant"] for p in gp) / gn),
            "relapse_pct": round(100 * sum(p["relapse"] for p in gp) / gn),
            "cancer_risk": ginfo["cancer_risk"],
        })

    return {
        "atlas": "Hereditary-Leukemia-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference: RUNX1 · GATA2 · CEBPA · DDX41 · ETV6 · SAMD9L · TP53 · BRCA2",
        "seeds": "3286-3293",
        "total_patients": n,
        "cr_pct": cr_pct,
        "radiation_pct": radiation_pct,
        "targeted_pct": targeted_pct,
        "transplant_pct": transplant_pct,
        "relapse_pct": relapse_pct,
        "mean_age_dx": mean_age,
        "gene_summaries": gene_summaries,
        "top_tumor_types": top_tumours,
        "key_management_rules": [
            "RUNX1 FPD/AML: NO family HSCT donor without RUNX1 germline exclusion testing MANDATORY",
            "GATA2: HSCT curative — timing critical (before AML transformation); lymphedema does NOT resolve",
            "CEBPA biallelic: GOOD PROGNOSIS — HSCT NOT indicated in CR1; consolidate 3-4x HiDAC",
            "DDX41: SIBLING DONOR EXCLUSION mandatory; late-onset AML 60-70yr; R525H somatic second hit",
            "ETV6: ALL 20-30x elevated PATHOGNOMONIC; ETV6-RUNX1 somatic t(12;21) independent of germline",
            "SAMD9L: monosomy 7 PATHOGNOMONIC adaptive reversion; cerebellar ataxia does NOT resolve post-HSCT",
            "TP53: AVOID RADIATION ABSOLUTELY; t-AML complex karyotype very poor prognosis",
            "BRCA2 FA-D1: AML PATHOGNOMONIC biallelic; AVOID alkylating agents; SIBLING EXCLUSION MANDATORY",
        ],
        "clinical_pearls": [
            "RUNX1 germline: dense platelet granule defect + mild thrombocytopenia = FPD/AML (electron microscopy to confirm)",
            "GATA2 germline: monosomy 7 in young patient = GATA2 deficiency until proven otherwise",
            "CEBPA biallelic AML: new C-terminal mutation at relapse = clonally unrelated new AML (normal behaviour)",
            "DDX41 germline: D140G/Y259C most common; R525H somatic second hit confirms biallelic inactivation",
            "ETV6 germline ALL: do NOT conflate with somatic ETV6-RUNX1 t(12;21) fusion — completely independent",
            "SAMD9L: monosomy 7 in this syndrome is PROTECTIVE (removes mutant allele), not primary oncogenic event",
            "TP53 ALL: omit cranial RT absolutely; standard-risk ALL protocol with modified RT-free CNS prophylaxis",
            "BRCA2 FA-D1: DEB/MMC chromosomal fragility test FIRST before AML induction — confirms FA diagnosis",
        ],
    }


def generate_breakdown() -> dict:
    cohort = _generate_cohort()
    from collections import Counter
    breakdown = []
    for ginfo in ATLAS_GENES:
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
            "targeted_pct": round(100 * sum(p["targeted"] for p in gp) / gn),
            "transplant_pct": round(100 * sum(p["transplant"] for p in gp) / gn),
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
    return {"atlas": "Hereditary-Leukemia-Predisposition-Atlas", "breakdown": breakdown}


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
        "atlas": "Hereditary-Leukemia-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "RUNX1_FPD_AML": (
                "FPD/AML: NO family HSCT donor without RUNX1 germline exclusion testing. "
                "Dense platelet granule defect PATHOGNOMONIC. AML 35-44% lifetime. "
                "Annual CBC + BM biopsy if CH variant >5%."
            ),
            "GATA2_HSCT_CURATIVE": (
                "GATA2: HSCT curative — timing at MDS/aplasia before AML transformation. "
                "Lymphedema does NOT resolve post-HSCT. Monosomy 7 PATHOGNOMONIC GATA2 MDS. "
                "MonoMAC: MAC + PCP prophylaxis. HPV vaccination MANDATORY."
            ),
            "CEBPA_GOOD_PROGNOSIS": (
                "Biallelic CEBPA AML: GOOD PROGNOSIS. HSCT NOT indicated in CR1. "
                "Consolidate 3-4x HiDAC. New C-terminal at relapse = clonally unrelated new AML. "
                "Germline testing only if N-terminal frameshift detected."
            ),
            "DDX41_SIBLING_EXCLUSION": (
                "DDX41: SIBLING DONOR EXCLUSION MANDATORY. Late-onset AML 60-70yr. "
                "R525H somatic second hit PATHOGNOMONIC biallelic DDX41 AML. "
                "Annual BM biopsy from age 50yr."
            ),
            "ETV6_ALL_NOT_t1221": (
                "ETV6 germline: ALL 20-30x elevated PATHOGNOMONIC. "
                "ETV6-RUNX1 somatic t(12;21) is INDEPENDENT of germline ETV6 LOF. "
                "Do NOT order germline testing for somatic ETV6-RUNX1 fusion ALL."
            ),
            "SAMD9L_MONOSOMY7_REVERSION": (
                "SAMD9L: monosomy 7 is ADAPTIVE REVERSION (removes mutant SAMD9L allele). "
                "Cerebellar ataxia does NOT resolve post-HSCT. "
                "HSCT at MDS/aplasia; RIC conditioning (avoid TBI)."
            ),
            "TP53_AVOID_RADIATION": (
                "LFS TP53: AVOID RADIATION ABSOLUTELY. Omit cranial RT in ALL. "
                "t-AML complex karyotype: APR-246 + azacitidine or magrolimab + azacitidine. "
                "WBMRI Toronto annually (NOT CT/PET)."
            ),
            "BRCA2_FA_D1_AVOID_ALKYLATING": (
                "BRCA2 FA-D1: AML PATHOGNOMONIC biallelic (25-35% by age 10yr). "
                "AVOID alkylating agents (cyclophosphamide). Alkylating-agent-FREE induction. "
                "DEB/MMC test first. SIBLING DONOR EXCLUSION MANDATORY."
            ),
        },
        "cascade_testing_rule": (
            "Hereditary Leukemia Predisposition Atlas — Cascade Testing: "
            "All 8 genes follow autosomal dominant (or AR for BRCA2 FA-D1) inheritance. "
            "First-degree relatives carry 50% risk (AD genes) or 25% risk (AR FA-D1). "
            "RUNX1/DDX41: NO family HSCT donor before germline exclusion testing. "
            "GATA2: monosomy 7 in young patient with pancytopenia = GATA2 germline testing FIRST. "
            "CEBPA: germline testing indicated if N-terminal frameshift detected in AML. "
            "ETV6: germline testing indicated if ETV6-LOF (NOT t(12;21) ETV6-RUNX1 somatic fusion). "
            "SAMD9L: germline testing if pancytopenia + cerebellar ataxia + monosomy 7. "
            "TP53: WBMRI Toronto annually; omit cranial RT; alkylating agents restricted. "
            "BRCA2: DEB/MMC confirms FA-D1; sibling exclusion MANDATORY before allograft."
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
    print(json.dumps(df["definitions"]["RUNX1"], indent=2)[:1500])
