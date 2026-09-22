#!/usr/bin/env python3
"""Hereditary-Lymphoma-Predisposition-Atlas -- Complete 8-Gene Reference
ATM     (Ataxia-Telangiectasia Mutated; 3056aa; 11q22.3; AD/AR LOF;
         Ataxia-Telangiectasia biallelic = radiation sensitivity ABSOLUTE CI;
         Monoallelic ATM: CLL 4-7x, MCL 3-5x, DLBCL 2-4x risk;
         ceralasertib ATRi + olaparib clinical trials;
         seed SEED_BASE+0) .
BRCA2   (Breast Cancer Gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC -- Hereditary Breast-Ovarian Cancer syndrome;
         NHL 2-3x elevated; HR-deficient; cisplatin/olaparib;
         Fanconi Anemia type D1 biallelic -- most severe FA;
         seed SEED_BASE+1) .
CARD11  (Caspase Recruitment Domain-Containing Protein 11; 1154aa; 7p22.2; AD GOF;
         BENTA disease (B-cell expansion with NF-kB and T-cell anergy);
         constitutive NF-kB signalling -> DLBCL / MALT lymphoma predisposition;
         ibrutinib BTKi response; venetoclax BCL2;
         seed SEED_BASE+2) .
PIK3CD  (Phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic delta; 1044aa; 1p36.22; AD GOF;
         APDS1 / PASLI (activated PI3K-delta syndrome type 1);
         EBV+ and CMV-driven B-cell lymphoma; EBV+ DLBCL pathognomonic;
         leniolisib PI3Kdelta-specific inhibitor FDA2023;
         seed SEED_BASE+3) .
KMT2D   (Lysine Methyltransferase 2D; 5537aa; 12q13.12; AD LOF;
         Kabuki syndrome type 1 (KS1); germline LOF -> follicular lymphoma predisposition;
         KMT2D somatic = most common follicular lymphoma driver (83%);
         rituximab + lenalidomide + obinutuzumab regimens;
         seed SEED_BASE+4) .
TP53    (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome (LFS);
         NHL/DLBCL elevated; AVOID RADIATION ABSOLUTELY;
         WBMRI annually Toronto Protocol; R337H Brazilian founder;
         seed SEED_BASE+5) .
TNFRSF13B (TNF Receptor Superfamily Member 13B / TACI; 293aa; 17p11.2; AD/AR LOF;
         TACI deficiency / CVID; B-NHL 3-5x elevated;
         A181E and C104R founder variants; IVIG replacement;
         rituximab + IVIG caution;
         seed SEED_BASE+6) .
LRBA    (LPS-Responsive Beige-Like Anchor Protein; 2863aa; 4q31.3; AR LOF;
         CVID-like; EBV+ B-cell lymphoproliferation; lymphoma 15-25%;
         abatacept CTLA4-Ig (CTLA4 recycling defect mechanism);
         HSCT curative;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3238-3245)
"""
import random

SEED_BASE = 3238

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-Monoallelic-LOF / AR-Biallelic-LOF -- 3056aa -- "
            "ATM-350kDa-PI3K-Like-Kinase-CLL-4-7x-MCL-3-5x-Biallelic-AT-Radiation-ABSOLUTE-CI-"
            "Ceralasertib-ATRi-Olaparib-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM encodes Ataxia-Telangiectasia Mutated (PI3K-like serine/threonine kinase): "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; PI3K-like kinase (PIKK family); "
            "  FAT domain (aa ~1960-2566): FRAP-ATM-TRRAP domain; dimer contact; "
            "  Kinase domain (aa ~2712-2962): catalytic; substrates: H2AX (Ser139), CHK2 (Thr68), p53 (Ser15); "
            "  FATC domain (aa ~3024-3056): C-terminal regulatory; redox sensor; "
            "  ATM is activated by DSB (double-strand breaks): Mre11-Rad50-NBS1 (MRN) complex recruitment; "
            "  ATM autophosphorylates Ser1981 -> monomer -> active; "
            "  ATM phosphorylates >700 substrates at [S/T]Q motifs (ATM consensus); "
            "LYMPHOMA PREDISPOSITION (MONOALLELIC ATM): "
            "  Monoallelic germline ATM LOF: CLL (chronic lymphocytic leukaemia) 4-7x elevated; "
            "  MCL (mantle cell lymphoma): 3-5x elevated; del11q22 = somatic biallelic inactivation in 20% MCL; "
            "  DLBCL: 2-4x elevated; "
            "  Mechanism: haploinsufficiency -> impaired DSB repair -> lymphocyte genome instability; "
            "  ATM monoallelic penetrance: CLL lifetime risk ~15-20% (vs 0.5% general population); "
            "  PANCREATIC cancer 2-3x elevated (PARP inhibitor olaparib indication); "
            "  Breast cancer 2-3x; gastric 2-3x; "
            "BIALLELIC ATM (ATAXIA-TELANGIECTASIA, AT): "
            "  OMIM 208900; AR biallelic; 1/40,000-1/100,000 births; "
            "  Cerebellar ataxia onset 12-18 months (progressive, wheelchair by 10yr); "
            "  Oculomotor apraxia (slow saccades) PATHOGNOMONIC; "
            "  Telangiectasia: conjunctival, then skin -> PATHOGNOMONIC triad (ataxia + telangiectasia + immunodeficiency); "
            "  Immune deficiency: low IgA + IgG2 + IgE (50%); recurrent sinopulmonary infections; "
            "  AFP elevated (>10 ng/mL) in >95% = BIOMARKER; normalises in <10yr old; "
            "  Lymphoma: biallelic AT -- NHL 70-100x elevated lifetime; T-cell lymphoma predominant in children; "
            "  Leukaemia: T-ALL / T-PLL in biallelic AT; "
            "RADIATION SENSITIVITY (BIALLELIC AT - ABSOLUTE CI): "
            "  ABSOLUTE CI radiotherapy biallelic AT: lethal/near-lethal radiation toxicity; "
            "  Lymphoma chemotherapy: standard cytotoxics (CHOP) tolerated but dose reduction may be needed; "
            "  MONOALLELIC ATM lymphoma: radiotherapy relative CI (2-3x toxicity risk); "
            "  Ceralasertib (ATRi) + olaparib: clinical trials in ATM-deficient lymphomas; "
            "  AZD6738 (ceralasertib) + durvalumab: DLBCL ATM-deficient ongoing; "
            "SURVEILLANCE (ATM GERMLINE): "
            "  CLL: annual FBC + differential from age 30yr (monoallelic); "
            "  Breast: annual MRI alternating mammography from age 40yr (monoallelic); "
            "  Pancreatic: MRI/EUS from age 50yr (monoallelic); "
            "  Biallelic AT: annual AFP; lymphoma vigilance; IgA levels + IVIG if deficient; "
            "  Cascade: first-degree relatives germline ATM testing"
        ),
        "inheritance": "AD monoallelic LOF (CLL/lymphoma predisposition) / AR biallelic LOF (Ataxia-Telangiectasia, OMIM 208900); 1/40,000-1/100,000 births biallelic; monoallelic carrier frequency 1/100",
        "cancer_risk": "Monoallelic: CLL 4-7x lifetime; MCL 3-5x; DLBCL 2-4x; pancreatic 2-3x; breast 2-3x. Biallelic AT: NHL 70-100x; T-ALL/T-PLL biallelic AT",
        "pathognomonic": "Biallelic AT triad: cerebellar ataxia + oculomotor telangiectasia + immune deficiency PATHOGNOMONIC; AFP elevated >10 ng/mL biallelic AT >95% PATHOGNOMONIC; ABSOLUTE radiation CI biallelic AT",
        "surveillance_key": "Annual FBC from age 30yr (monoallelic CLL surveillance); ceralasertib ATRi + olaparib clinical trials ATM-deficient lymphoma; ABSOLUTE CI radiotherapy biallelic AT; olaparib FDA pancreatic ATM germline",
        "key_distinctions": [
            "CLL-4-7X-MONOALLELIC-ATM-DOMINANT-LYMPHOMA",
            "MCL-DEL11Q22-SOMATIC-BIALLELIC-20PCT-MCL",
            "AT-RADIATION-ABSOLUTE-CI-BIALLELIC",
            "AFP-ELEVATED-95PCT-BIALLELIC-PATHOGNOMONIC",
            "TELANGIECTASIA-CONJUNCTIVAL-PATHOGNOMONIC",
            "CERALASERTIB-ATRi-ATM-DEFICIENT-LYMPHOMA",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-HBOC-NHL-2-3x-HRD-Cisplatin-Olaparib-"
            "FA-D1-Most-Severe-Biallelic-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes Breast Cancer Gene 2 (HR scaffold): "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; nuclear HR scaffold; no enzymatic activity; "
            "  BRC repeats (aa 1002-2085): 8 BRC repeats; each binds RAD51 monomer; "
            "  OB folds (C-terminal, aa 2402-3190): ssDNA binding; RPA displacement; "
            "  NLS (nuclear localisation); PALB2-binding N-terminus (aa 10-40); "
            "  TR2 domain (aa 3265-3330): RAD51 binding (second site); "
            "  BRCA2 loads RAD51 onto ssDNA at DSB -> nucleoprotein filament -> HR; "
            "  BRCA2 LOF -> HRD (homologous recombination deficiency) -> genomic instability; "
            "NHL RISK (BRCA2 GERMLINE): "
            "  BRCA2 germline LOF: NHL overall 2-3x elevated; "
            "  DLBCL: 2-3x; Hodgkin lymphoma: 3-5x elevated (emerging); "
            "  Mechanism: HRD -> genomic instability in lymphocyte precursors; "
            "  BRCA2-deficient lymphomas: HRD-enriched -> cisplatin + olaparib sensitivity; "
            "DOMINANT CANCER SPECTRUM (BRCA2): "
            "  Breast: 69% lifetime (female); male breast: 8.9% = HIGHEST hereditary male breast risk; "
            "  Ovarian: 17%; pancreatic: 5-7% (olaparib FDA); prostate: 27% lifetime; "
            "  NHL: 2-3x elevated; "
            "FA-D1 (FANCONI ANEMIA TYPE D1 - BIALLELIC): "
            "  Biallelic BRCA2 = FA complementation group D1 = MOST SEVERE FA subtype; "
            "  Medulloblastoma age <5yr: unique to FA-D1 (NOT other FA subtypes); "
            "  Wilms tumour; AML; primitive neuro-ectodermal tumour (PNET); "
            "  Biallelic BRCA2 ALL: T-ALL and precursor B-ALL; very early onset (<3yr); "
            "OLAPARIB (PARP INHIBITOR): "
            "  BRCA2 LOF -> HRD -> olaparib synthetic lethality; "
            "  FDA-approved: BRCA2-germline ovarian, breast, pancreatic, prostate; "
            "  BRCA2-deficient NHL: HRD -> cisplatin / olaparib investigational; "
            "  Platinum sensitivity: BRCA2 LOF lymphoma -> carboplatin/cisplatin preferred; "
            "SURVEILLANCE (BRCA2): "
            "  Annual breast MRI + mammography from age 25yr (female); age 35yr (male); "
            "  Annual prostate PSA from age 40yr; "
            "  Pancreatic MRI/EUS from age 50yr; "
            "  BSO at age 40-45yr (ovarian risk reduction); "
            "  Annual FBC + LDH for NHL surveillance no formal guideline (clinical judgment); "
            "  Cascade: all first-degree relatives"
        ),
        "inheritance": "AD germline LOF; OMIM 600185; near-complete penetrance for breast/pancreatic; biallelic = FA-D1 (OMIM 605724) = most severe FA; Ashkenazi founder 6174delT; male breast cancer significantly elevated",
        "cancer_risk": "Breast 69% (female) / 8.9% (male) DOMINANT; ovarian 17%; pancreatic 5-7%; prostate 27%; NHL 2-3x elevated; Hodgkin 3-5x emerging; biallelic FA-D1: medulloblastoma/Wilms/AML most severe",
        "pathognomonic": "FA-D1 biallelic MOST SEVERE FA (medulloblastoma PATHOGNOMONIC <5yr, Wilms, AML); HRD cisplatin + olaparib sensitivity; male breast cancer 8.9% = HIGHEST hereditary; 6174delT Ashkenazi founder",
        "surveillance_key": "Annual breast MRI from 25yr; BSO 40-45yr; olaparib FDA-approved BRCA2-germline; cascade testing all first-degree; FA-D1 biallelic: medulloblastoma watch <5yr; NHL: annual FBC clinical judgment",
        "key_distinctions": [
            "NHL-2-3X-BRCA2-GERMLINE-HRD",
            "FA-D1-BIALLELIC-MEDULLOBLASTOMA-PATHOGNOMONIC",
            "MALE-BREAST-8-9PCT-HIGHEST-HEREDITARY",
            "OLAPARIB-FDA-BRCA2-GERMLINE-OVARIAN-BREAST",
            "CISPLATIN-HRD-BRCA2-LYMPHOMA",
            "6174DELT-ASHKENAZI-FOUNDER",
        ],
    },
    {
        "gene": "CARD11",
        "protein": (
            "CARD11 -- 7p22.2 Autosomal-Dominant-GOF -- 1154aa -- "
            "CARD11-130kDa-CARMA1-MAGUK-NF-kB-Scaffold-BENTA-DLBCL-MALT-"
            "Constitutive-NF-kB-Ibrutinib-BTKi-OMIM-607210"
        ),
        "locus": "7p22.2",
        "protein_size": (
            "1154 aa / 130 kDa / 7p22.2 CARD11 encodes CARMA1 (Caspase recruitment domain-containing protein 11): "
            "STRUCTURE: "
            "  1154 aa / 130 kDa; MAGUK (membrane-associated guanylate kinase) family scaffold; "
            "  CARD domain (aa 1-100): caspase recruitment; BCL10 interaction; "
            "  Coiled-coil domain (aa 101-600): oligomerisation; self-inhibition in resting state; "
            "  PDZ domain (aa 700-800): protein-protein interaction scaffold; "
            "  SH3 domain (aa 820-880): signal input; "
            "  GUK domain (aa 900-1050): guanylate kinase-like; "
            "  CARD11 forms CBM complex: CARD11-BCL10-MALT1; "
            "  CBM complex: activated by TCR/BCR signalling -> IKK -> NF-kB -> lymphocyte activation; "
            "  GOF mutations (BENTA): coiled-coil domain variants -> constitutive CBM activity -> NF-kB; "
            "BENTA DISEASE (CARD11 GOF): "
            "  BENTA = B-cell Expansion with NF-kB and T-cell Anergy; "
            "  OMIM 616452; AD GOF; rare (< 100 families worldwide 2026); "
            "  Peripheral B-cell lymphocytosis (polyclonal) PATHOGNOMONIC; "
            "  Susceptibility to EBV + CMV lymphoproliferation; "
            "  T-cell anergy: reduced T-cell proliferation despite normal count; "
            "  Recurrent infections: sinopulmonary, herpesviruses; "
            "  Atopic disease: eczema, food allergy 60-80% (CARD11 GOF); "
            "LYMPHOMA PREDISPOSITION (CARD11): "
            "  BENTA disease: DLBCL risk 20-30% lifetime (elevated vs population); "
            "  MALT lymphoma: mucosa-associated (gastric, salivary); "
            "  Mechanism: constitutive NF-kB -> anti-apoptotic (BCL2, BCL-xL upregulation) -> clonal expansion; "
            "  CARD11 somatic GOF: present in 10% DLBCL (activated B-cell [ABC] type); "
            "  ABC-DLBCL (somatic CARD11): BTK inhibitor (ibrutinib) sensitive -> same pathway; "
            "IBRUTINIB (BTK INHIBITOR) -- PATHWAY-TARGETED: "
            "  BCR -> BTK -> CARD11-CBM -> NF-kB; "
            "  Ibrutinib blocks BTK -> prevents CBM activation; "
            "  BENTA CARD11 GOF: downstream of BTK; ibrutinib only partial response; "
            "  MALT1 protease inhibitors: preclinical CARD11 GOF (direct CBM targeting); "
            "  Venetoclax BCL2i: anti-apoptotic BCL2 upregulated by CARD11-NF-kB; "
            "SURVEILLANCE (CARD11 BENTA): "
            "  Annual FBC + immunophenotyping (CD5, CD10, CD19, CD20, CD23 B-cell panel); "
            "  Annual EBV + CMV viral load (PCR); LDH + uric acid + beta-2 microglobulin; "
            "  PET-CT if constitutional symptoms (B symptoms: fever, night sweats, weight loss); "
            "  Annual dermatology (eczema monitoring, exclude cutaneous lymphoma CTCL); "
            "  Cascade: first-degree relatives CARD11 sequencing (GOF variant)"
        ),
        "inheritance": "AD GOF (BENTA disease); OMIM 616452; rare (<100 families worldwide 2026); de novo cases reported; somatic CARD11 GOF in 10% DLBCL (ABC type) -- not germline",
        "cancer_risk": "DLBCL 20-30% lifetime (BENTA); MALT lymphoma elevated; EBV+ lymphoproliferation; constitutive NF-kB -> clonal B-cell expansion -> lymphoma transformation",
        "pathognomonic": "Peripheral B-cell lymphocytosis polyclonal PATHOGNOMONIC BENTA; T-cell anergy + atopic disease (eczema 60-80%) PATHOGNOMONIC BENTA; constitutive NF-kB CBM complex = core mechanism",
        "surveillance_key": "Annual FBC + B-cell immunophenotype; annual EBV/CMV PCR viral load; MALT1 protease inhibitors preclinical; venetoclax BCL2i targeted; ibrutinib partial response (downstream of BTK)",
        "key_distinctions": [
            "BENTA-B-CELL-LYMPHOCYTOSIS-POLYCLONAL-PATHOGNOMONIC",
            "T-CELL-ANERGY-PATHOGNOMONIC-BENTA",
            "DLBCL-20-30PCT-BENTA-LIFETIME",
            "CONSTITUTIVE-NF-KB-CBM-COMPLEX",
            "IBRUTINIB-PARTIAL-RESPONSE-DOWNSTREAM-BTK",
            "SOMATIC-CARD11-10PCT-DLBCL-ABC-TYPE",
        ],
    },
    {
        "gene": "PIK3CD",
        "protein": (
            "PIK3CD -- 1p36.22 Autosomal-Dominant-GOF -- 1044aa -- "
            "PIK3CD-119kDa-PI3K-p110delta-APDS1-EBV-Plus-B-NHL-PATHOGNOMONIC-"
            "Leniolisib-FDA2023-PI3Kdelta-Inhibitor-OMIM-602839"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "1044 aa / 119 kDa / 1p36.22 PIK3CD encodes the PI3K catalytic subunit p110delta: "
            "STRUCTURE: "
            "  1044 aa / 119 kDa; class I PI3K catalytic subunit; lymphocyte-restricted expression; "
            "  N-terminal ABD domain (aa 1-108): p85 regulatory subunit (PIK3R1) binding; "
            "  RAS-binding domain (RBD, aa 163-285): RAS-GTPase interaction; "
            "  C2 domain (aa 330-480): membrane association; "
            "  Helical domain (aa 481-696): GOF hotspot region (E1021K, E525K); "
            "  Kinase domain (aa 697-1044): catalytic; PIP2 -> PIP3; "
            "  p110delta + p85 (regulatory): heterodimer; p85 inhibits p110delta basally; "
            "  GOF mutations: E1021K (most common, 60%), E525K, N334K, C416R; "
            "  GOF -> constitutive PI3K signalling -> AKT-mTOR -> lymphocyte hyperactivation; "
            "APDS1 / PASLI (PIK3CD GOF): "
            "  APDS1 = Activated PI3K-Delta Syndrome type 1; OMIM 615513; AD GOF; "
            "  PASLI = PI3K delta activating somatic-like inherited; "
            "  Frequency: ~1/400,000-1/500,000; ~200 families worldwide 2026; "
            "  Recurrent sinopulmonary infections: bronchiectasis by adolescence; "
            "  Herpesvirus susceptibility: EBV, CMV, HSV reactivation -> lymphoproliferation; "
            "  Lymphadenopathy: persistent, often massive; hepatosplenomegaly; "
            "  B-cell dysfunction: hypogammaglobulinaemia (progressive); transitional B cells elevated; "
            "  T-cell senescence: CD57+CD8+ T cells (chronic viral antigenic drive); "
            "EBV+ B-NHL (PATHOGNOMONIC APDS1): "
            "  EBV+ B-cell lymphoma in APDS1: PATHOGNOMONIC; "
            "  DLBCL EBV+: most common lymphoma type in APDS1; "
            "  Hodgkin-like EBV+ lymphoproliferation; "
            "  Mechanism: constitutive PI3K-delta -> impaired EBV-infected B-cell clearance; "
            "  EBV+ B-NHL lifetime risk in APDS1: 20-30%; "
            "  EBV monitoring (PCR): MANDATORY in APDS1 -- rising EBV load = lymphoma alert; "
            "LENIOLISIB (FDA2023 -- PI3Kdelta SPECIFIC INHIBITOR): "
            "  Leniolisib (CDZ173): selective PI3K-delta inhibitor; "
            "  FDA-approved 2023 for APDS (first PI3Kdelta inhibitor approval); "
            "  Reduces lymphadenopathy, normalises B-cell subsets, reduces immunoglobulin waste; "
            "  Not to be confused with idelalisib (pan-PI3K, higher toxicity, colitis risk); "
            "  Leniolisib: less colitis + pneumonitis than idelalisib (delta-selectivity); "
            "  mTOR inhibitors (rapamycin, everolimus): alternative/adjunct in APDS; "
            "SURVEILLANCE (PIK3CD APDS1): "
            "  Monthly EBV + CMV PCR viral load; "
            "  Annual CT chest (bronchiectasis) + abdomen (lymphadenopathy, splenomegaly); "
            "  Annual FBC + immunophenotype + immunoglobulins (IgG trough before IVIG); "
            "  IVIG replacement if IgG deficient (most APDS patients require IVIG); "
            "  Annual LDH + beta-2-microglobulin (lymphoma markers); "
            "  Cascade: first-degree relatives PIK3CD sequencing"
        ),
        "inheritance": "AD GOF; OMIM 615513 (APDS1); rare (~200 families worldwide 2026); E1021K most common pathogenic variant (60%); de novo cases reported; PIK3R1 LOF = APDS2 (different gene, identical phenotype)",
        "cancer_risk": "EBV+ DLBCL 20-30% lifetime PATHOGNOMONIC APDS1; Hodgkin-like EBV+ lymphoproliferation; bronchiectasis near-universal; hypogammaglobulinaemia progressive",
        "pathognomonic": "EBV+ B-NHL PATHOGNOMONIC APDS1; monthly EBV PCR MANDATORY; leniolisib FDA2023 PI3Kdelta-specific (NOT idelalisib pan-PI3K); massive lymphadenopathy + hepatosplenomegaly APDS1",
        "surveillance_key": "Monthly EBV + CMV PCR mandatory; leniolisib FDA2023 first-line APDS; IVIG if IgG deficient; annual CT chest (bronchiectasis); E1021K most common GOF variant (60%)",
        "key_distinctions": [
            "EBV-PLUS-B-NHL-PATHOGNOMONIC-APDS1",
            "MONTHLY-EBV-PCR-MANDATORY",
            "LENIOLISIB-FDA2023-PI3KDELTA-SPECIFIC",
            "NOT-IDELALISIB-PAN-PI3K-COLITIS",
            "E1021K-MOST-COMMON-GOF-60PCT",
            "BRONCHIECTASIS-NEAR-UNIVERSAL-APDS1",
        ],
    },
    {
        "gene": "KMT2D",
        "protein": (
            "KMT2D -- 12q13.12 Autosomal-Dominant-LOF -- 5537aa -- "
            "KMT2D-593kDa-MLL4-H3K4-Methyltransferase-Kabuki1-Follicular-Lymphoma-"
            "83pct-Somatic-Driver-Rituximab-Lenalidomide-OMIM-602113"
        ),
        "locus": "12q13.12",
        "protein_size": (
            "5537 aa / 593 kDa / 12q13.12 KMT2D encodes Lysine Methyltransferase 2D (MLL4): "
            "STRUCTURE: "
            "  5537 aa / 593 kDa; COMPASS-like histone H3K4 methyltransferase complex; "
            "  SET domain (aa ~5400-5537): catalytic; H3K4 mono- and di-methylation; "
            "  PHD fingers x5 (aa ~4000-5200): chromatin reader (H3K4me1 recognition); "
            "  FYRN/FYRC domains: complex assembly with RBBP5, WDR5, ASH2L, DPY30; "
            "  High-mobility group (HMG) domains: DNA binding; "
            "  AT-hooks: minor groove DNA binding; "
            "  KMT2D establishes enhancer landscapes: H3K4me1 marks active enhancers; "
            "  KMT2D LOF -> hypomethylation of H3K4 at enhancers -> transcriptional dysregulation; "
            "KABUKI SYNDROME TYPE 1 (KS1): "
            "  OMIM 147920; AD LOF; 1/32,000 births; "
            "  Kabuki make-up facial features: arched eyebrows with lateral third thinning, long palpebral fissures, broad nasal tip, large ears, persistent fetal fingertip pads PATHOGNOMONIC constellation; "
            "  Intellectual disability: mild-moderate 99%; "
            "  Short stature: postnatal, GH usually normal; "
            "  Skeletal anomalies: brachydactyly, joint hypermobility; "
            "  Cardiac defects: 30-50% (VSD, ASD, coarctation); "
            "FOLLICULAR LYMPHOMA PREDISPOSITION (KMT2D GERMLINE): "
            "  KMT2D somatic mutation: present in 83% follicular lymphoma = MOST COMMON SOMATIC DRIVER FL; "
            "  KMT2D germline LOF: follicular lymphoma predisposition (emerging data); "
            "  Mechanism: germline KMT2D haploinsufficiency -> susceptibility to second-hit somatic mutation -> FL; "
            "  KMT2D LOF also somatic driver: DLBCL 30%, MZL 20%; "
            "  Kabuki patients: lymphoma surveillance from age 20yr; "
            "TREATMENT (KMT2D-ASSOCIATED FOLLICULAR LYMPHOMA): "
            "  Rituximab + lenalidomide (R2): active in FL; lenalidomide targets cereblon pathway; "
            "  Rituximab + obinutuzumab (type 2 anti-CD20): GALLIUM trial; OS benefit over R-CHOP in FL; "
            "  EZH2 inhibitors (tazemetostat): active in EZH2-mutant FL; KMT2D co-mutated; "
            "  Venetoclax BCL2i: BCL2 upregulated in KMT2D-deficient FL; "
            "  HSCT: consolidation in high-risk KMT2D-germline FL (young patients); "
            "SURVEILLANCE (KMT2D KABUKI + LYMPHOMA): "
            "  Annual FBC + LDH + beta-2-microglobulin from age 20yr; "
            "  Annual lymph node examination; PET-CT if lymphadenopathy; "
            "  Echocardiography: annually (cardiac defects); "
            "  Annual developmental/cognitive assessment (paediatric KS1); "
            "  Cascade: first-degree relatives KMT2D testing"
        ),
        "inheritance": "AD LOF; OMIM 147920 (Kabuki type 1); 1/32,000 births; de novo ~75%; frameshift/nonsense truncating variants most common; KMT2D somatic = 83% follicular lymphoma (most common FL driver)",
        "cancer_risk": "Follicular lymphoma predisposition (germline); KMT2D somatic 83% FL, 30% DLBCL, 20% MZL; Kabuki cardiac defects 30-50%; Kabuki ID mild-moderate 99%",
        "pathognomonic": "Kabuki facial gestalt PATHOGNOMONIC: arched eyebrows lateral thinning + long palpebral fissures + persistent fingertip pads; KMT2D somatic 83% FL = most common FL driver; H3K4 enhancer methylation loss",
        "surveillance_key": "Annual FBC + LDH from age 20yr; rituximab + lenalidomide (R2) frontline FL; EZH2 inhibitor tazemetostat EZH2-mutant FL; obinutuzumab type-2 anti-CD20 FL benefit; KMT2D-germline follicular lymphoma emerging",
        "key_distinctions": [
            "KMT2D-SOMATIC-83PCT-FOLLICULAR-LYMPHOMA-MOST-COMMON-DRIVER",
            "KABUKI-FACIAL-GESTALT-PATHOGNOMONIC",
            "PERSISTENT-FINGERTIP-PADS-PATHOGNOMONIC-KS1",
            "H3K4-ENHANCER-METHYLATION-LOSS",
            "RITUXIMAB-LENALIDOMIDE-R2-FL",
            "TAZEMETOSTAT-EZH2-INHIBITOR-FL",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-NHL-Elevated-AVOID-RADIATION-ABSOLUTELY-"
            "WBMRI-Toronto-R337H-Brazilian-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes Tumour Protein p53 (Guardian of the Genome): "
            "STRUCTURE: "
            "  393 aa / 43 kDa; tetrameric transcription factor; "
            "  N-terminal transactivation domain (TAD1: aa 1-40; TAD2: aa 40-67); "
            "  Proline-rich domain (PRD, aa 67-98); "
            "  DBD (DNA-binding domain): aa 94-292 -- ALL HOTSPOT RESIDUES (R175, G245, R248, R249, R273, R282); "
            "  Tetramerisation domain (TET, aa 325-356): 4 subunits form active tetramer; "
            "  C-terminal regulatory domain (CTD, aa 356-393): acetylation K372, K373, K382; "
            "  p53 activates: p21 (CDKN1A, G1 arrest), PUMA, NOXA, BAX (apoptosis), MDM2 (autoregulation); "
            "  MDM2 ubiquitinates p53 -> proteasomal degradation (negative feedback); "
            "  p53 HOT SPOT mutations: GOF (dominant negative) = tetramer poisoning by mutant subunit; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; AD LOF; germline TP53 pathogenic; "
            "  Classic LFS: sarcoma <45yr + breast cancer + brain tumour + adrenocortical carcinoma; "
            "  Core LFS tumours: sarcoma (50-60%), brain (25%), breast (25%), ACC (6%); "
            "  NHL/DLBCL: elevated in LFS (5-10% lifetime); "
            "  B-cell ALL in children: elevated LFS (paediatric ALL ALL in TP53 germline); "
            "  Penetrance: >90% by age 60yr (female); 73% by age 60yr (male); "
            "AVOID RADIATION ABSOLUTELY (LFS): "
            "  TP53 LOF -> defective p53-mediated apoptosis of radiation-damaged cells; "
            "  Radiation -> secondary malignancy (osteosarcoma, angiosarcoma) in LFS field; "
            "  WBMRI (whole-body MRI) annually: Toronto Protocol; replaces CT/PET (avoids ionising radiation); "
            "  Brain MRI 6-monthly (age <35yr); annual thereafter; "
            "  Mammography avoided (minimal ionising exposure); breast MRI from 20yr; "
            "LYMPHOMA SPECIFICS (TP53 LFS): "
            "  TP53 somatic mutation: present in 25-30% DLBCL = poor prognosis marker; "
            "  Germline TP53 lymphoma: DLBCL, B-ALL, T-cell lymphoma; "
            "  Double-hit DLBCL (MYC + BCL2/BCL6): TP53 co-mutation = worst prognosis; "
            "  Chemotherapy preferred over radiation in LFS lymphoma (AVOID XRT); "
            "  R-CHOP (DLBCL): standard; consider R-DA-EPOCH if double-hit; "
            "  R337H Brazilian founder: ~0.3% carrier frequency in southern Brazil; "
            "TORONTO PROTOCOL (WBMRI): "
            "  Annual WBMRI: whole body from skull base to proximal femur; "
            "  6-monthly brain MRI; "
            "  Annual breast MRI (no mammography); annual abdominal US; "
            "  Annual dermatology (soft tissue sarcoma surface exam); "
            "  Annual FBC + ESR + LDH (NHL surveillance)"
        ),
        "inheritance": "AD LOF; OMIM 151623 (LFS); penetrance >90% (female) by age 60yr; R337H Brazilian founder 0.3% southern Brazil; de novo ~20%; hotspot GOF mutations (dominant-negative) worst prognosis",
        "cancer_risk": "Sarcoma 50-60% lifetime DOMINANT LFS; brain 25%; breast 25% (female); ACC 6%; NHL/DLBCL 5-10%; B-ALL paediatric; TP53 somatic 25-30% DLBCL poor prognosis",
        "pathognomonic": "WBMRI Toronto Protocol MANDATORY (AVOID radiation); LFS classic triad sarcoma + breast + brain; AVOID RADIATION ABSOLUTELY (secondary sarcoma); R337H Brazilian founder 0.3%",
        "surveillance_key": "Annual WBMRI Toronto Protocol; 6-monthly brain MRI; breast MRI from 20yr (NO mammography); AVOID radiation ABSOLUTELY all LFS; R-CHOP (NOT XRT) LFS lymphoma; annual FBC + LDH",
        "key_distinctions": [
            "AVOID-RADIATION-ABSOLUTELY-LFS",
            "WBMRI-TORONTO-PROTOCOL-MANDATORY",
            "SARCOMA-50-60PCT-DOMINANT-LFS",
            "NHL-5-10PCT-LFS-LIFETIME",
            "R337H-BRAZILIAN-FOUNDER-0-3PCT",
            "TP53-SOMATIC-25-30PCT-DLBCL-POOR-PROGNOSIS",
        ],
    },
    {
        "gene": "TNFRSF13B",
        "protein": (
            "TNFRSF13B -- 17p11.2 Autosomal-Dominant/AR-LOF -- 293aa -- "
            "TACI-32kDa-TNF-Receptor-CVID-B-NHL-3-5x-A181E-C104R-Founders-"
            "IVIG-Replacement-Rituximab-Caution-OMIM-604907"
        ),
        "locus": "17p11.2",
        "protein_size": (
            "293 aa / 32 kDa / 17p11.2 TNFRSF13B encodes TACI (Transmembrane activator and CAML interactor): "
            "STRUCTURE: "
            "  293 aa / 32 kDa; type III transmembrane TNF receptor superfamily member; "
            "  Signal peptide (aa 1-15); "
            "  Extracellular TNFR-like CRD repeats (aa 16-119): ligand binding; "
            "  Transmembrane domain (aa 153-181); "
            "  Cytoplasmic domain (aa 182-293): signalling; CAML + TRAF6 interaction; "
            "  TACI ligands: APRIL (a proliferation-inducing ligand) and BAFF (B-cell activating factor); "
            "  TACI signalling: isotype class-switching (IgA, IgG), B-cell survival regulation; "
            "  TACI LOF -> reduced class-switch recombination -> IgA and IgG deficiency; "
            "  TACI LOF -> unchecked B-cell proliferation (BAFF/APRIL signalling not properly regulated); "
            "TACI DEFICIENCY / CVID: "
            "  TACI deficiency = most common monogenic cause of CVID (~10% of genetically-solved CVID); "
            "  CVID: hypogammaglobulinaemia (IgG <7 g/L + IgA and/or IgM reduced); "
            "  OMIM 240500 (CVID); OMIM 604907 (TACI gene); "
            "  Clinical onset: second-third decade (recurrent infections trigger investigation); "
            "  Recurrent sinopulmonary infections: Streptococcus pneumoniae, Haemophilus influenzae; "
            "  Autoimmune: ITP, AIHA (Evans syndrome 10-15%); "
            "  Granulomatous disease: lung granuloma GLILD (granulomatous-lymphocytic interstitial lung disease); "
            "B-NHL (TACI DEFICIENCY): "
            "  B-NHL: 3-5x elevated in TACI-deficient CVID vs general population; "
            "  MALT lymphoma (stomach, parotid): most common lymphoma subtype in CVID; "
            "  DLBCL: elevated; "
            "  Mechanism: unchecked B-cell expansion + chronic antigen stimulation (infections) + impaired apoptosis; "
            "  Lymphoma surveillance: annual PET-CT or CT chest/abdomen from age 30yr; "
            "FOUNDER VARIANTS (TACI): "
            "  A181E (c.541G>A): most common TACI pathogenic variant; European founder; heterozygous (AD) sufficient for CVID phenotype; "
            "  C104R (c.310T>C): second most common; monoallelic; more penetrant than A181E; "
            "  Both variants: extracellular CRD APRIL-binding disruption; "
            "  Incomplete penetrance: A181E/C104R carriers 1/60-1/300 general population; CVID penetrance ~20-30%; "
            "RITUXIMAB CAUTION (CVID): "
            "  Rituximab (anti-CD20): depletes B cells -> worsens hypogammaglobulinaemia in CVID; "
            "  AVOID rituximab if CVID not adequately replaced with IVIG/SCIG; "
            "  If lymphoma requires rituximab: increase IVIG dose; monitor IgG trough closely; "
            "  IVIG replacement: mandatory CVID (IgG trough >8 g/L target); "
            "  SCIG (subcutaneous IG): alternative to IVIG; better trough stability"
        ),
        "inheritance": "AD heterozygous LOF (A181E, C104R founder variants; incomplete penetrance 20-30%) or AR biallelic LOF (more severe); OMIM 604907; 1/60-1/300 carrier frequency; most common monogenic CVID (~10%)",
        "cancer_risk": "B-NHL 3-5x elevated; MALT lymphoma (gastric, parotid) DOMINANT CVID lymphoma type; DLBCL elevated; Evans syndrome (ITP+AIHA) 10-15%; GLILD interstitial lung disease",
        "pathognomonic": "MALT lymphoma DOMINANT CVID lymphoma subtype; A181E + C104R European founder TACI; RITUXIMAB CAUTION in CVID (worsens hypogammaglobulinaemia); GLILD lung PATHOGNOMONIC CVID",
        "surveillance_key": "IVIG trough >8 g/L mandatory; annual CT chest/abdomen lymphoma surveillance from 30yr; rituximab caution (increase IVIG if used); A181E C104R founder testing; SCIG alternative stable trough",
        "key_distinctions": [
            "MALT-LYMPHOMA-DOMINANT-CVID-SUBTYPE",
            "A181E-C104R-EUROPEAN-FOUNDER-TACI",
            "RITUXIMAB-CAUTION-WORSENS-HYPOGAMMAGLOBULINAEMIA",
            "IVIG-TROUGH-8-G-L-TARGET",
            "GLILD-PATHOGNOMONIC-CVID",
            "B-NHL-3-5X-TACI-DEFICIENT-CVID",
        ],
    },
    {
        "gene": "LRBA",
        "protein": (
            "LRBA -- 4q31.3 Autosomal-Recessive-LOF -- 2863aa -- "
            "LRBA-320kDa-BEACH-WD40-CVID-EBV-Plus-Lymphoma-15-25pct-"
            "Abatacept-CTLA4-Recycling-HSCT-Curative-OMIM-606453"
        ),
        "locus": "4q31.3",
        "protein_size": (
            "2863 aa / 320 kDa / 4q31.3 LRBA encodes LPS-Responsive Beige-Like Anchor Protein: "
            "STRUCTURE: "
            "  2863 aa / 320 kDa; BEACH (beige and CHS1) family scaffold; "
            "  N-terminal ARM repeats (aa 1-400): protein-protein interaction; "
            "  BEACH domain (aa 1400-1800): vesicle trafficking scaffold; "
            "  WD40 repeats (C-terminal, aa 2000-2863): beta-propeller; protein binding; "
            "  LRBA function: intracellular trafficking of CTLA4 to cell surface and recycling endosomes; "
            "  LRBA LOF -> CTLA4 degraded in lysosomes (not recycled to surface) -> CTLA4 surface deficiency; "
            "  CTLA4 normally inhibits T-cell activation (immune checkpoint); "
            "  CTLA4 deficiency -> T-cell hyperactivation -> lymphoproliferation + autoimmunity; "
            "LRBA DEFICIENCY: "
            "  OMIM 614700; AR biallelic LOF; rare (~150 families worldwide 2026); "
            "  CVID-like hypogammaglobulinaemia; recurrent sinopulmonary infections; "
            "  Autoimmune cytopenias: AIHA, ITP (Evans syndrome 30-40%); "
            "  Inflammatory bowel disease: Crohn-like colitis 50-60% PATHOGNOMONIC; "
            "  Lymphadenopathy: mediastinal + abdominal (often massive); "
            "  Splenomegaly: near-universal; "
            "  Onset: 1st-2nd decade; highly variable severity; "
            "LYMPHOMA (LRBA): "
            "  EBV+ B-cell lymphoproliferation PATHOGNOMONIC in LRBA deficiency; "
            "  Lymphoma (EBV+ DLBCL, Hodgkin-like): 15-25% lifetime risk; "
            "  Mechanism: CTLA4 surface deficiency -> unchecked T-cell help to EBV-infected B cells -> EBV+ lymphoma; "
            "  T regulatory cells (Tregs): reduced (CTLA4 critical for Treg suppression); "
            "  Lymphoma may precede or follow other LRBA features; "
            "ABATACEPT (CTLA4-Ig -- MECHANISM-TARGETED): "
            "  Abatacept (CTLA4-Ig): exogenous CTLA4 supplementation = MECHANISM-SPECIFIC; "
            "  Abatacept binds CD80/CD86 on APCs -> blocks T-cell activation -> corrects LRBA LOF; "
            "  LRBA deficiency: abatacept highly effective (partial normalisation of T-cell dysregulation); "
            "  Response: lymphadenopathy reduction, autoimmune cytopenias improve, IBD improvement; "
            "  High dose abatacept (belatacept IV): superior to standard SC dose in LRBA; "
            "  mTOR inhibitors (sirolimus): alternative/adjunct to abatacept; "
            "HSCT (CURATIVE): "
            "  HSCT: curative for LRBA deficiency (restores CTLA4 recycling); "
            "  Indications: severe lymphoma, refractory autoimmunity, or CVID with major complications; "
            "  Timing: before irreversible end-organ damage (bronchiectasis, cirrhosis); "
            "  Reduced-intensity conditioning (RIC): preferred (lymphocyte reconstitution); "
            "SURVEILLANCE (LRBA): "
            "  Monthly EBV + CMV PCR (lymphoma sentinel); "
            "  Annual CT chest/abdomen PET (lymphadenopathy, lymphoma); "
            "  Annual FBC + immunoglobulins + CTLA4 surface expression (flow cytometry diagnostic); "
            "  Colonoscopy 2-yearly (IBD colitis surveillance); "
            "  IVIG replacement if IgG deficient; "
            "  Cascade: parents (AR obligate carriers); siblings 25% risk biallelic"
        ),
        "inheritance": "AR biallelic LOF; OMIM 614700; rare (~150 families worldwide 2026); no common founder; compound heterozygous most common; CTLA4 surface deficiency = diagnostic flow cytometry biomarker",
        "cancer_risk": "EBV+ DLBCL / Hodgkin-like 15-25% lifetime PATHOGNOMONIC; lymphoproliferation near-universal; Evans syndrome (AIHA+ITP) 30-40%; IBD Crohn-like 50-60%",
        "pathognomonic": "EBV+ B-NHL PATHOGNOMONIC LRBA; Crohn-like IBD + lymphadenopathy + CVID PATHOGNOMONIC constellation; abatacept CTLA4-Ig mechanism-targeted (NOT empirical immunosuppression); CTLA4 surface deficiency flow",
        "surveillance_key": "Monthly EBV/CMV PCR mandatory; abatacept (high dose) mechanism-targeted first-line; HSCT curative for severe disease; CTLA4 surface expression flow cytometry diagnostic; sirolimus adjunct",
        "key_distinctions": [
            "EBV-PLUS-B-NHL-PATHOGNOMONIC-LRBA",
            "ABATACEPT-MECHANISM-TARGETED-CTLA4-RECYCLING",
            "IBD-CROHN-LIKE-50-60PCT-PATHOGNOMONIC",
            "CTLA4-SURFACE-DEFICIENCY-DIAGNOSTIC-FLOW",
            "HSCT-CURATIVE-LRBA",
            "MONTHLY-EBV-PCR-MANDATORY",
        ],
    },
]

# --------------------------------------------------------------------------- #
# Simulated patient cohort
# --------------------------------------------------------------------------- #

GENE_AGE_PARAMS = {
    "ATM":       {"mean_age": 55, "sd": 12},
    "BRCA2":     {"mean_age": 52, "sd": 14},
    "CARD11":    {"mean_age": 38, "sd": 10},
    "PIK3CD":    {"mean_age": 28, "sd": 8},
    "KMT2D":     {"mean_age": 45, "sd": 12},
    "TP53":      {"mean_age": 35, "sd": 10},
    "TNFRSF13B": {"mean_age": 47, "sd": 11},
    "LRBA":      {"mean_age": 22, "sd": 8},
}

PATHOGENIC_VARIANTS = {
    "ATM":       ["c.7271T>G p.Val2424Gly", "c.5932G>T p.Glu1978Ter", "c.3161C>T p.Thr1054Ile",
                  "c.6095G>A p.Arg2032Gln", "IVS10-6T>G splicing"],
    "BRCA2":     ["c.5946delT p.Ser1982ArgfsTer22 (6174delT Ashkenazi)", "c.886G>T p.Glu296Ter",
                  "c.3847_3848delGT p.Val1283IlefsTer2", "c.9257A>G p.Asn3086Ser", "c.7558C>T p.Arg2520Ter"],
    "CARD11":    ["c.2713A>T p.Ile905Phe", "c.2839G>A p.Gly947Arg", "c.2413C>T p.Arg805Trp",
                  "c.2593G>A p.Gly865Arg", "c.2380C>A p.Leu794Met"],
    "PIK3CD":    ["c.3061G>A p.Glu1021Lys (E1021K)", "c.1573G>A p.Glu525Lys",
                  "c.1001A>T p.Asn334Lys", "c.1246T>G p.Cys416Gly", "c.2843G>T p.Arg948Leu"],
    "KMT2D":     ["c.10462C>T p.Arg3488Ter", "c.8887C>T p.Arg2963Ter", "c.7891delA p.Ile2631fsTer16",
                  "c.14072delA p.Asp4691fsTer8", "c.3028C>T p.Arg1010Ter"],
    "TP53":      ["c.817C>T p.Arg273Cys (R273C)", "c.742C>T p.Arg248Trp (R248W)",
                  "c.524G>A p.Arg175His (R175H)", "c.1009C>T p.Arg337His (R337H Brazilian)",
                  "c.733G>A p.Gly245Ser (G245S)"],
    "TNFRSF13B": ["c.541G>A p.Ala181Glu (A181E European founder)", "c.310T>C p.Cys104Arg (C104R)",
                  "c.204delT p.Ser69ArgfsTer16", "c.275_276delCT p.Ser92fsTer6",
                  "c.580A>G p.Thr194Ala"],
    "LRBA":      ["c.4957_4960delAGAG p.Arg1653GlyfsTer10", "c.4567C>T p.Arg1523Ter",
                  "c.6088C>T p.Arg2030Ter", "c.2350G>T p.Glu784Ter",
                  "c.7285C>T p.Arg2429Ter"],
}

LYMPHOMA_SUBTYPES = {
    "ATM":       ["CLL (chronic lymphocytic leukaemia)", "MCL (mantle cell lymphoma)",
                  "DLBCL", "Splenic marginal zone lymphoma", "B-ALL"],
    "BRCA2":     ["DLBCL", "Hodgkin lymphoma (classical)", "Follicular lymphoma",
                  "B-ALL (biallelic FA-D1)", "Primary mediastinal B-cell lymphoma"],
    "CARD11":    ["DLBCL (ABC type)", "MALT lymphoma (gastric)", "EBV+ B-cell lymphoproliferation",
                  "Extranodal MZL", "Cutaneous B-cell lymphoma"],
    "PIK3CD":    ["EBV+ DLBCL (PATHOGNOMONIC)", "EBV+ Hodgkin-like lymphoproliferation",
                  "CMV-driven B-cell expansion", "Primary mediastinal B-cell lymphoma",
                  "Burkitt-like EBV+ lymphoma"],
    "KMT2D":     ["Follicular lymphoma (grade 1-2)", "DLBCL (transformed FL)",
                  "Marginal zone lymphoma", "Paediatric FL (KMT2D germline)", "EZH2-co-mutant FL"],
    "TP53":      ["DLBCL (double-hit MYC+BCL2)", "B-ALL", "Sarcoma (Richter equivalent)",
                  "T-cell lymphoma (LFS)", "Primary CNS lymphoma"],
    "TNFRSF13B": ["MALT lymphoma (gastric)", "MALT lymphoma (parotid)", "DLBCL",
                  "Splenic MZL (CVID-associated)", "Mucosa-associated lymphoid tissue lymphoma"],
    "LRBA":      ["EBV+ DLBCL (PATHOGNOMONIC)", "Hodgkin-like EBV+ LPD",
                  "EBV+ Burkitt-like", "Hepatosplenic T-cell lymphoma", "PTLD-like (post-infectious)"],
}

TREATMENT_PROTOCOLS = {
    "ATM":       ["R-CHOP (DLBCL standard; no radiation)", "BR (bendamustine-rituximab CLL/MCL)",
                  "Ibrutinib (CLL/MCL BTKi)", "Ceralasertib + olaparib (ATM-deficient trials)",
                  "R-CHOP → ASCT consolidation (high-risk)"],
    "BRCA2":     ["R-CHOP / R-DA-EPOCH (DLBCL)", "Olaparib maintenance (HRD)", "Cisplatin-based (HRD)",
                  "R-CHOP (Hodgkin alternative)", "ABVD (classical Hodgkin, avoid XRT LFS)"],
    "CARD11":    ["R-CHOP (DLBCL standard)", "Ibrutinib + R-CHOP (ABC-DLBCL trials)", "Venetoclax BCL2i",
                  "MALT1 protease inhibitors (preclinical)", "Lenalidomide + rituximab (MALT)"],
    "PIK3CD":    ["R-CHOP (EBV+ DLBCL)", "Leniolisib (APDS-targeted PI3Kdelta)", "IVIG + leniolisib",
                  "mTOR inhibitors (sirolimus)", "Rituximab + etoposide (EBV+ B-LPD)"],
    "KMT2D":     ["Rituximab + lenalidomide R2 (FL)", "Obinutuzumab + CHOP (GALLIUM-FL)",
                  "Tazemetostat EZH2i (EZH2-mutant FL)", "Venetoclax BCL2i",
                  "ASCT consolidation transformed FL"],
    "TP53":      ["R-CHOP (avoid XRT LFS)", "R-DA-EPOCH (double-hit DLBCL)", "Venetoclax BCL2i",
                  "Anti-CD19 CAR-T (relapsed/refractory)", "Alisertib AURKAi (TP53-mutant)"],
    "TNFRSF13B": ["IVIG replacement mandatory (CVID)", "R-CHOP (DLBCL + increase IVIG dose)",
                  "Anti-H. pylori eradication (MALT gastric)", "Rituximab with IVIG monitoring",
                  "SCIG (subcutaneous IG) trough stability"],
    "LRBA":      ["Abatacept (CTLA4-Ig mechanism-targeted)", "Sirolimus (mTOR adjunct)",
                  "IVIG + abatacept (combined)", "R-CHOP (EBV+ DLBCL)", "HSCT (curative severe disease)"],
}

SURVEILLANCE_PROTOCOLS = {
    "ATM":       ["Annual FBC + differential from age 30yr (CLL)", "Annual breast MRI from age 40yr",
                  "Pancreatic MRI/EUS from age 50yr", "Annual LDH + beta-2 microglobulin"],
    "BRCA2":     ["Annual breast MRI from age 25yr (female)", "BSO at age 40-45yr",
                  "Annual prostate PSA from age 40yr", "Pancreatic MRI/EUS from age 50yr"],
    "CARD11":    ["Annual FBC + B-cell immunophenotype", "Annual EBV + CMV PCR viral load",
                  "PET-CT if B symptoms", "Annual LDH + beta-2 microglobulin"],
    "PIK3CD":    ["Monthly EBV + CMV PCR", "Annual CT chest (bronchiectasis)", "Annual immunoglobulins + FBC",
                  "Annual LDH + beta-2 microglobulin"],
    "KMT2D":     ["Annual FBC + LDH from age 20yr", "Annual lymph node examination",
                  "Echocardiography annually (cardiac)", "PET-CT if lymphadenopathy"],
    "TP53":      ["Annual WBMRI (Toronto Protocol)", "6-monthly brain MRI (age <35yr)",
                  "Breast MRI from age 20yr", "Annual FBC + LDH + ESR"],
    "TNFRSF13B": ["IVIG trough IgG >8 g/L (mandatory)", "Annual CT chest/abdomen from age 30yr",
                  "Annual FBC + immunoglobulins", "Anti-H. pylori testing (MALT)"],
    "LRBA":      ["Monthly EBV + CMV PCR", "Annual CT/PET-CT (lymphadenopathy)",
                  "Annual FBC + immunoglobulins + CTLA4 flow", "Colonoscopy 2-yearly (IBD)"],
}


def _make_patients(gene_def: dict, seed: int, n: int = 40) -> list[dict]:
    rng = random.Random(seed)
    params = GENE_AGE_PARAMS[gene_def["gene"]]
    variants = PATHOGENIC_VARIANTS[gene_def["gene"]]
    subtypes = LYMPHOMA_SUBTYPES[gene_def["gene"]]
    treatments = TREATMENT_PROTOCOLS[gene_def["gene"]]
    surveillance = SURVEILLANCE_PROTOCOLS[gene_def["gene"]]
    patients = []
    for i in range(n):
        age = max(5, int(rng.gauss(params["mean_age"], params["sd"])))
        gender = rng.choice(["M", "F", "F"])
        variant = rng.choice(variants)
        subtype = rng.choice(subtypes)
        response = rng.choice(["CR", "PR", "SD", "Refractory", "CR (maintenance ongoing)"])
        tx = rng.choice(treatments)
        surv = rng.choice(surveillance)
        relapse = rng.random() < 0.28
        treatment_cycles = rng.choice([6, 8, 4, 6, 8])
        ebv_positive = gene_def["gene"] in ("PIK3CD", "LRBA", "CARD11") and rng.random() < 0.45
        prior_cvid = gene_def["gene"] in ("TNFRSF13B", "LRBA", "PIK3CD") and rng.random() < 0.65
        ivig_on = prior_cvid and rng.random() < 0.85
        patients.append({
            "patient_id": f"{gene_def['gene']}-LYM-{seed}-{i+1:03d}",
            "gene": gene_def["gene"],
            "locus": gene_def["locus"],
            "age_dx": age,
            "sex": gender,
            "variant": variant,
            "lymphoma_subtype": subtype,
            "ebv_positive": ebv_positive,
            "prior_cvid": prior_cvid,
            "ivig_on_treatment": ivig_on,
            "treatment": tx,
            "treatment_cycles": treatment_cycles,
            "response": response,
            "relapse": relapse,
            "surveillance_note": surv,
            "protein_detail": gene_def["protein"],
            "inheritance": gene_def["inheritance"],
            "cancer_risk": gene_def["cancer_risk"],
            "pathognomonic": gene_def["pathognomonic"],
            "key_distinctions": gene_def["key_distinctions"],
        })
    return patients


def _build_all_patients() -> list[dict]:
    all_pts = []
    for i, gdef in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        all_pts.extend(_make_patients(gdef, seed))
    return all_pts


def _json_safe(obj):
    """Recursively convert non-JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    return str(obj)


# --------------------------------------------------------------------------- #
# Public API functions
# --------------------------------------------------------------------------- #

def generate_overview() -> dict:
    patients = _build_all_patients()
    gene_counts = {}
    for p in patients:
        g = p["gene"]
        gene_counts.setdefault(g, {"total": 0, "cr": 0, "relapse": 0,
                                   "ebv_pos": 0, "cvid": 0, "ivig": 0})
        gene_counts[g]["total"] += 1
        if "CR" in p["response"]:
            gene_counts[g]["cr"] += 1
        if p["relapse"]:
            gene_counts[g]["relapse"] += 1
        if p.get("ebv_positive"):
            gene_counts[g]["ebv_pos"] += 1
        if p.get("prior_cvid"):
            gene_counts[g]["cvid"] += 1
        if p.get("ivig_on_treatment"):
            gene_counts[g]["ivig"] += 1

    mean_age = sum(p["age_dx"] for p in patients) / len(patients)
    subtypes_all = [p["lymphoma_subtype"] for p in patients]
    top_subtypes = {}
    for s in subtypes_all:
        top_subtypes[s] = top_subtypes.get(s, 0) + 1
    top_subtypes = dict(sorted(top_subtypes.items(), key=lambda x: -x[1])[:10])

    ebv_total = sum(1 for p in patients if p.get("ebv_positive"))
    cvid_total = sum(1 for p in patients if p.get("prior_cvid"))
    ivig_total = sum(1 for p in patients if p.get("ivig_on_treatment"))
    cr_total = sum(1 for p in patients if "CR" in p["response"])
    relapse_total = sum(1 for p in patients if p["relapse"])

    gene_summaries = []
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        gc = gene_counts.get(g, {})
        total = gc.get("total", 0)
        gene_summaries.append({
            "gene": g,
            "locus": gdef["locus"],
            "inheritance": gdef["inheritance"][:80] + "...",
            "cancer_risk_summary": gdef["cancer_risk"][:120] + "...",
            "pathognomonic": gdef["pathognomonic"][:120] + "...",
            "n_patients": total,
            "cr_n": gc.get("cr", 0),
            "cr_pct": round(gc.get("cr", 0) / total * 100, 1) if total else 0,
            "relapse_n": gc.get("relapse", 0),
            "relapse_pct": round(gc.get("relapse", 0) / total * 100, 1) if total else 0,
            "ebv_positive_n": gc.get("ebv_pos", 0),
            "prior_cvid_n": gc.get("cvid", 0),
            "ivig_n": gc.get("ivig", 0),
        })

    return _json_safe({
        "atlas": "Hereditary-Lymphoma-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Lymphoma Predisposition Reference",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": len(patients),
        "mean_age_dx": round(mean_age, 1),
        "cr_total": cr_total,
        "cr_pct": round(cr_total / len(patients) * 100, 1),
        "relapse_total": relapse_total,
        "relapse_pct": round(relapse_total / len(patients) * 100, 1),
        "ebv_positive_total": ebv_total,
        "ebv_positive_pct": round(ebv_total / len(patients) * 100, 1),
        "prior_cvid_total": cvid_total,
        "cvid_pct": round(cvid_total / len(patients) * 100, 1),
        "ivig_total": ivig_total,
        "top_lymphoma_subtypes": top_subtypes,
        "gene_summaries": gene_summaries,
        "clinical_pearls": [
            "ATM monoallelic: CLL 4-7x elevated; biallelic AT = radiation ABSOLUTE CI",
            "PIK3CD APDS1: EBV+ B-NHL PATHOGNOMONIC; monthly EBV PCR mandatory; leniolisib FDA2023",
            "LRBA: EBV+ B-NHL PATHOGNOMONIC; abatacept CTLA4-Ig mechanism-targeted (NOT empirical IS)",
            "CARD11 BENTA: polyclonal B-cell lymphocytosis PATHOGNOMONIC; DLBCL 20-30% lifetime",
            "TNFRSF13B TACI CVID: MALT lymphoma dominant subtype; RITUXIMAB CAUTION (worsens hypogamma)",
            "KMT2D Kabuki: persistent fingertip pads PATHOGNOMONIC; KMT2D somatic = 83% FL driver",
            "TP53 LFS: AVOID RADIATION ABSOLUTELY; WBMRI Toronto annual; sarcoma 50-60% dominant",
            "BRCA2: FA-D1 biallelic = medulloblastoma PATHOGNOMONIC <5yr; NHL 2-3x HRD platinum/olaparib",
        ],
        "key_management_rules": [
            "ATM biallelic: ABSOLUTE CI radiotherapy; ceralasertib ATRi + olaparib trials",
            "PIK3CD: leniolisib (NOT idelalisib) = FDA2023 PI3Kdelta-specific (idelalisib = pan-PI3K colitis)",
            "LRBA: abatacept high dose; HSCT curative; sirolimus adjunct",
            "TNFRSF13B: IVIG trough >8 g/L mandatory; rituximab only with IVIG coverage",
            "KMT2D FL: R2 (rituximab + lenalidomide); tazemetostat if EZH2-mutant co-mutation",
            "TP53 LFS lymphoma: R-CHOP (NOT XRT); R-DA-EPOCH if double-hit",
        ],
    })


def generate_breakdown() -> dict:
    patients = _build_all_patients()
    gene_data = {}
    for p in patients:
        g = p["gene"]
        gene_data.setdefault(g, [])
        gene_data[g].append(p)

    breakdown = []
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        pts = gene_data.get(g, [])
        if not pts:
            continue

        ages = [p["age_dx"] for p in pts]
        mean_age = sum(ages) / len(ages)
        cr_n = sum(1 for p in pts if "CR" in p["response"])
        relapse_n = sum(1 for p in pts if p["relapse"])
        ebv_n = sum(1 for p in pts if p.get("ebv_positive"))
        cvid_n = sum(1 for p in pts if p.get("prior_cvid"))
        ivig_n = sum(1 for p in pts if p.get("ivig_on_treatment"))

        subtype_counts: dict = {}
        for p in pts:
            s = p["lymphoma_subtype"]
            subtype_counts[s] = subtype_counts.get(s, 0) + 1

        variant_counts: dict = {}
        for p in pts:
            v = p["variant"]
            variant_counts[v] = variant_counts.get(v, 0) + 1

        tx_counts: dict = {}
        for p in pts:
            t = p["treatment"]
            tx_counts[t] = tx_counts.get(t, 0) + 1

        breakdown.append({
            "gene": g,
            "locus": gdef["locus"],
            "protein": gdef["protein"],
            "inheritance": gdef["inheritance"],
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
            "key_distinctions": gdef["key_distinctions"],
            "n_patients": len(pts),
            "mean_age_dx": round(mean_age, 1),
            "min_age_dx": min(ages),
            "max_age_dx": max(ages),
            "cr_n": cr_n,
            "cr_pct": round(cr_n / len(pts) * 100, 1),
            "relapse_n": relapse_n,
            "relapse_pct": round(relapse_n / len(pts) * 100, 1),
            "ebv_positive_n": ebv_n,
            "ebv_positive_pct": round(ebv_n / len(pts) * 100, 1) if len(pts) else 0,
            "prior_cvid_n": cvid_n,
            "prior_cvid_pct": round(cvid_n / len(pts) * 100, 1) if len(pts) else 0,
            "ivig_n": ivig_n,
            "top_subtypes": dict(sorted(subtype_counts.items(), key=lambda x: -x[1])[:5]),
            "top_variants": dict(sorted(variant_counts.items(), key=lambda x: -x[1])[:4]),
            "top_treatments": dict(sorted(tx_counts.items(), key=lambda x: -x[1])[:4]),
            "patients": pts,
        })

    return _json_safe({
        "atlas": "Hereditary-Lymphoma-Predisposition-Atlas",
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
            "lymphoma_subtypes": LYMPHOMA_SUBTYPES[g],
            "treatment_protocols": TREATMENT_PROTOCOLS[g],
            "surveillance_protocols": SURVEILLANCE_PROTOCOLS[g],
        }

    return _json_safe({
        "atlas": "Hereditary-Lymphoma-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "ATM_BIALLELIC_RADIATION_ABSOLUTE_CI": (
                "NEVER irradiate biallelic AT patients -- lethal radiation toxicity; "
                "AVOID radiation in monoallelic ATM lymphoma (relative CI 2-3x toxicity)"
            ),
            "PIKCD_LENIOLISIB_NOT_IDELALISIB": (
                "Use leniolisib (FDA2023 PI3Kdelta-specific) NOT idelalisib (pan-PI3K) for APDS1; "
                "idelalisib causes fatal colitis + pneumonitis in APDS1 immune dysregulation background"
            ),
            "LRBA_ABATACEPT_MECHANISM": (
                "Abatacept (CTLA4-Ig) is mechanism-targeted for LRBA (restores CTLA4 signalling); "
                "NOT empirical immunosuppression; HSCT curative for severe disease"
            ),
            "TNFRSF13B_RITUXIMAB_CAUTION": (
                "Rituximab in TACI-deficient CVID: ALWAYS increase IVIG dose and monitor IgG trough; "
                "rituximab depletes B cells -> worsens hypogammaglobulinaemia -> infection risk"
            ),
            "TP53_AVOID_RADIATION": (
                "AVOID radiation ABSOLUTELY in LFS (TP53 germline); "
                "WBMRI (not CT/PET) annually Toronto Protocol; R-CHOP (not XRT) for LFS lymphoma"
            ),
            "CARD11_CONSTITUTIVE_NFKB": (
                "CARD11 GOF = constitutive NF-kB via CBM complex; "
                "ibrutinib partially effective (downstream of BTK); MALT1 protease inhibitors preclinical"
            ),
            "KMT2D_SOMATIC_83PCT_FL": (
                "KMT2D somatic = most common follicular lymphoma driver (83% FL); "
                "germline KMT2D LOF = Kabuki type 1 + FL predisposition; "
                "tazemetostat (EZH2i) for EZH2-mutant co-mutation in KMT2D-FL"
            ),
            "BRCA2_FA_D1_MEDULLOBLASTOMA": (
                "Biallelic BRCA2 = FA-D1 = MOST SEVERE FA; medulloblastoma PATHOGNOMONIC <5yr; "
                "cisplatin + olaparib for BRCA2-deficient NHL (HRD sensitivity)"
            ),
        },
        "cascade_testing_rule": (
            "ATM/BRCA2/CARD11/PIK3CD/KMT2D/TP53/TNFRSF13B: cascade to all first-degree relatives. "
            "LRBA (AR): test parents (obligate carriers) + siblings (25% risk biallelic). "
            "ATM biallelic (AR): partners of AT children require ATM carrier testing. "
            "TNFRSF13B A181E/C104R: penetrance 20-30% heterozygous -- counsel incomplete penetrance."
        ),
    })


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2)[:3000])
    print("\n--- breakdown (first gene) ---")
    bd = generate_breakdown()
    print(json.dumps(bd["breakdown"][0], indent=2)[:2000])
