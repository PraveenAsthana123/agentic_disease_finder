#!/usr/bin/env python3
"""Hereditary-Hepatoblastoma-Predisposition-Atlas -- Complete 8-Gene Reference
APC     (adenomatous polyposis coli; 2843aa; 5q22.2; AD LOF;
         Familial adenomatous polyposis / Gardner syndrome;
         Hepatoblastoma 0.5-2% of FAP; 50-100x general population;
         5' mutation codons 200-1600 centromeric; US liver q3-6m birth-7yr;
         seed SEED_BASE+0) .
CTNNB1  (catenin beta-1 / beta-catenin; 781aa; 3p22.1; AD GOF germline rare;
         Familial / de novo Wnt activating GOF;
         Hepatoblastoma 90% carry somatic CTNNB1 activating mutations;
         Germline CTNNB1 GOF (exon 3 del/missense): very rare familial HBL;
         seed SEED_BASE+1) .
BRCA2   (breast cancer gene 2; 3418aa; 13q12.3; Biallelic AR LOF FA-D1 / AD HBOC;
         Fanconi anaemia complementation group D1;
         Hepatoblastoma PATHOGNOMONIC FA-D1 (alongside AML/Wilms/medulloblastoma);
         AVOID alkylating agents; SIBLING DONOR EXCLUSION MANDATORY;
         seed SEED_BASE+2) .
TP53    (tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         Hepatoblastoma in LFS childhood tumour spectrum (1-3%);
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annual;
         seed SEED_BASE+3) .
GPC3    (glypican-3; 580aa; Xq26.2; X-linked recessive;
         Simpson-Golabi-Behmel syndrome (SGBS);
         Hepatoblastoma predisposition 5-8% SGBS; GPC3 overexpressed in HBL;
         Overgrowth syndrome; visceral anomalies PATHOGNOMONIC;
         seed SEED_BASE+4) .
NSD1    (nuclear receptor binding SET domain protein 1; 2696aa; 5q35.3; AD LOF/GOF;
         Sotos syndrome / Beckwith-Wiedemann-like overgrowth;
         Hepatoblastoma 5-10x elevated in Sotos; overgrowth PATHOGNOMONIC;
         Macrocephaly + tall stature PATHOGNOMONIC Sotos; seed SEED_BASE+5) .
DICER1  (DICER1 ribonuclease III; 1922aa; 14q32.13; AD LOF;
         DICER1 tumour predisposition syndrome;
         Hepatoblastoma component of DICER1 tumour spectrum (2-4x elevated);
         PPB PATHOGNOMONIC DICER1; AVOID radiation; CT chest siblings <8yr;
         seed SEED_BASE+6) .
NFE2L2  (nuclear factor erythroid 2 like 2 / NRF2; 605aa; 2q31.2; AD GOF germline;
         Hereditary NRF2 gain-of-function / Autosomal dominant polycystic liver-like;
         Paediatric HCC/hepatoblastoma; early-onset HCC (teens-20s);
         Very rare (Kerins 2018 cohort); sulforaphane pathway;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3294-3301)
"""
import random

SEED_BASE = 3294

ATLAS_GENES = [
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "APC-309kDa-WNT-BetaCatenin-Scaffolding-FAP-Gardner-"
            "Hepatoblastoma-50-100x-5prime-Codons-200-1600-OMIM-175100"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 309 kDa / 5q22.2 APC encodes adenomatous polyposis coli protein: "
            "STRUCTURE: "
            "  2843 aa / 309 kDa; scaffold protein; WNT/beta-catenin pathway regulator; "
            "  N-terminal dimerisation domain (aa 1-57); "
            "  Armadillo repeats (aa 163-736): protein-protein interaction; "
            "  SAMP repeats (aa 1020-2100): axin binding; "
            "  mutation cluster region (MCR, aa 1286-1513): most colorectal adenoma-causing mutations; "
            "  C-terminal EB1 binding (aa 2130-2843): microtubule interaction; "
            "  APC truncating mutations -> cannot degrade beta-catenin -> WNT target gene overactivation; "
            "HEPATOBLASTOMA IN FAP (APC): "
            "  OMIM 175100; FAP hepatoblastoma: 50-100x general population risk; "
            "  Absolute risk: 0.5-2% of FAP patients develop hepatoblastoma; "
            "  Age: predominantly <5yr (peak 12-18 months); "
            "  Pathognomonic association: hepatoblastoma before polyposis symptoms in FAP family; "
            "  APC mutation location (5' end) correlated with hepatoblastoma: "
            "    Exon 4-15 (codons 200-1600, centromeric): hepatoblastoma risk highest; "
            "    Codons 1309 area (MCR): colorectal dominant; "
            "    5' mutations (exon 1-10): hepatoblastoma + desmoid + polyp risk simultaneously; "
            "  Somatic CTNNB1 mutations (90% of sporadic HBL): independent WNT activation pathway; "
            "SURVEILLANCE HEPATOBLASTOMA (APC/FAP): "
            "  AFP + liver US every 3-6 months from birth to age 7yr; "
            "  AFP >100 ng/mL in first 6 months of life + rising: investigate urgently; "
            "  CT abdomen: if US suspicious; "
            "  Post-age-7: AFP annually until 18yr (late hepatoblastoma rare but described); "
            "  Colorectal: annual colonoscopy from age 10-12yr (FAP standard); "
            "  EGD: from age 20yr or at polyposis onset; "
            "FAP DESMOID + HEPATOBLASTOMA: "
            "  Gardner syndrome (FAP + desmoid + osteomas + epidermoid cysts): "
            "    Desmoid: mesenteric PATHOGNOMONIC FAP post-surgery; "
            "    Nirogacestat (gamma-secretase inhibitor) FDA 2023 for progressive desmoid; "
            "    Surgery: AVOID mesenteric desmoid (paradoxical growth post-op); "
        ),
        "inheritance": "Autosomal dominant LOF; 20-30% de novo; near-complete penetrance for polyposis by age 40yr; hepatoblastoma penetrance 0.5-2%; genotype-phenotype correlation (5' mutations favour hepatoblastoma/desmoid)",
        "cancer_risk": "Colorectal cancer 100% by age 40yr untreated; hepatoblastoma 0.5-2% (50-100x elevated); desmoid 10-20% (Gardner); gastric/periampullary/thyroid/brain (medulloblastoma Turcot variant) rare",
        "pathognomonic": "Hepatoblastoma <5yr in FAP family PATHOGNOMONIC; 100+ colorectal adenomas PATHOGNOMONIC FAP; desmoid post-abdominal surgery PATHOGNOMONIC Gardner; hepatoblastoma + polyposis family = APC germline testing MANDATORY",
        "surveillance_key": "AFP + liver US q3-6m from birth to age 7yr; colonoscopy from age 10-12yr; nirogacestat for progressive desmoid; AVOID surgery for mesenteric desmoid (paradoxical growth); cascade 50% risk per offspring",
        "key_distinctions": [
            "HEPATOBLASTOMA-50-100X-ELEVATED-FAP-APC",
            "AFP-LIVER-US-Q3-6M-BIRTH-TO-AGE-7YR",
            "5PRIME-APC-MUTATIONS-HEPATOBLASTOMA-RISK",
            "DESMOID-MESENTERIC-AVOID-SURGERY-PARADOXICAL",
            "NIROGACESTAT-FDA2023-PROGRESSIVE-DESMOID",
            "COLORECTAL-100PCT-BY-40YR-UNTREATED",
        ],
    },
    {
        "gene": "CTNNB1",
        "protein": (
            "CTNNB1 -- 3p22.1 Autosomal-Dominant-GOF-germline-rare -- 781aa -- "
            "BetaCatenin-85kDa-WNT-Armadillo-Scaffold-Somatic-GOF-90pct-HBL-"
            "Germline-Exon3-Deletion-Missense-Familial-OMIM-116806"
        ),
        "locus": "3p22.1",
        "protein_size": (
            "781 aa / 85 kDa / 3p22.1 CTNNB1 encodes catenin beta-1 (beta-catenin): "
            "STRUCTURE: "
            "  781 aa / 85 kDa; armadillo repeat scaffold protein; "
            "  N-terminal regulatory domain (aa 1-130): phosphorylation targets; "
            "    GSK3B/CK1 phosphorylation sites: S33/S37/T41/S45 (exon 3 key region); "
            "    Exon 3 mutations: most common somatic + germline activating mutations; "
            "  Armadillo repeat domain (aa 130-665): 12 armadillo repeats; E-cadherin + TCF binding; "
            "  C-terminal transactivation (aa 695-781): CBP/p300 co-activator recruitment; "
            "  CTNNB1 function: WNT signalling + cell-cell adhesion (E-cadherin complex); "
            "  LOF mutations (N-terminal): intellectual disability, corpus callosum agenesis; "
            "  GOF mutations (exon 3 del/missense): cancer (HBL, endometrial, melanoma); "
            "CTNNB1 SOMATIC GOF IN HEPATOBLASTOMA: "
            "  Somatic activating CTNNB1 mutations: present in ~90% of hepatoblastomas; "
            "  Most common somatic HBL mutation: exon 3 in-frame deletions / missense S33/T41; "
            "  Somatic CTNNB1 does NOT indicate germline predisposition in index case; "
            "  Large somatic deletions (>50bp exon 3): associated with aggressive HBL; "
            "  AFP response to platinum: beta-catenin pathway drives AFP secretion; "
            "CTNNB1 GERMLINE GOF (RARE FAMILIAL): "
            "  Very rare (<50 families reported); exon 3 in-frame deletion or missense; "
            "  Familial hepatoblastoma: 2-3 affected siblings/parent-child; "
            "  De novo GOF: intellectual disability + macrocephaly + HBL reported; "
            "  CTNNB1 germline GOF ≠ FAP (APC LOF pathway distinct); "
            "  No polyposis; Wnt hyperactivation in liver specifically; "
            "TREATMENT CTNNB1-MUTANT HBL: "
            "  Standard: cisplatin + doxorubicin (PLADO protocol: SIOPEL-6); "
            "  Cisplatin monotherapy: SIOPEL-6 low-risk (PRETEXT I-II, AFP >100); "
            "  PRETEXT IV/metastatic: PLADO + SIOPEL-4 protocol; "
            "  AFP monitoring: major response marker (normalisation CR threshold); "
            "  Liver transplant: unresectable but chemotherapy-responsive HBL; "
        ),
        "inheritance": "Somatic GOF dominant (90% of sporadic HBL); germline GOF (exon 3 deletion/missense): very rare AD; de novo in most germline cases; familial HBL without FAP = CTNNB1 germline testing second-line after APC excluded",
        "cancer_risk": "Somatic: 90% of hepatoblastomas (not germline); germline GOF: hepatoblastoma (families described); endometrial cancer; desmoid-like; intellectual disability de novo GOF; macrocephaly",
        "pathognomonic": "90% of hepatoblastomas carry somatic CTNNB1 exon 3 mutations PATHOGNOMONIC HBL tumour biology; germline CTNNB1 GOF in familial HBL without FAP/polyposis PATHOGNOMONIC rare germline WNT; de novo CTNNB1 GOF + HBL + intellectual disability",
        "surveillance_key": "Somatic CTNNB1: standard SIOPEL-6 PLADO protocol; AFP q3m; germline CTNNB1 GOF: AFP + liver US q3-6m from birth to 7yr; liver transplant if unresectable chemotherapy-responsive; cascade testing if familial",
        "key_distinctions": [
            "SOMATIC-CTNNB1-90PCT-OF-SPORADIC-HBL",
            "EXON-3-INFRAME-DELETIONS-MISSENSE-GOF",
            "GERMLINE-CTNNB1-GOF-FAMILIAL-HBL-RARE",
            "AFP-PRIMARY-RESPONSE-MARKER-NORMALISATION-CR",
            "PLADO-SIOPEL6-CISPLATIN-DOXORUBICIN-STANDARD",
            "LIVER-TRANSPLANT-UNRESECTABLE-CHEMO-RESPONSIVE",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Biallelic-AR-LOF-FA-D1-AD-LOF-HBOC -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FA-D1-Hepatoblastoma-PATHOGNOMONIC-Biallelic-"
            "AVOID-Alkylating-Sibling-Donor-Exclusion-MANDATORY-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes breast cancer susceptibility protein 2: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; genome stability scaffold; "
            "  N-terminal PALB2-binding domain (aa 1-40); "
            "  BRC repeats (aa 1002-2085): RAD51 interaction (8 repeats); "
            "  C-terminal OB folds (aa 2402-3190): ssDNA binding; "
            "  BRCA2 function: HR repair; loads RAD51 onto resected DSBs; "
            "  Biallelic BRCA2 LOF: FA complementation group D1 (FA-D1); "
            "HEPATOBLASTOMA IN FA-D1 (BRCA2): "
            "  OMIM 605724; biallelic BRCA2 FA-D1: most severe childhood tumour spectrum; "
            "  Hepatoblastoma PATHOGNOMONIC FA-D1 (biallelic BRCA2); "
            "  Childhood cancer cluster: AML + Wilms + hepatoblastoma + medulloblastoma + RMS (all PATHOGNOMONIC); "
            "  DEB test / MMC test: chromosomal fragility PATHOGNOMONIC FA (all subtypes); "
            "  Hepatoblastoma in FA-D1: reported 5-10% cumulative (highest FA subtype for liver tumour); "
            "AVOID ALKYLATING AGENTS (FA-D1): "
            "  Cyclophosphamide: ABSOLUTELY CONTRAINDICATED in FA (ICL repair defect); "
            "  Cisplatin-based HBL protocols: ICL mechanism -> risk in FA; reduced dose mandatory; "
            "  SIOPEL-6 low-risk cisplatin monotherapy: dose-reduced in FA-D1; "
            "  Hepatoblastoma resection: mainstay (surgery + reduced-dose chemotherapy); "
            "  Liver transplant: option if unresectable + cisplatin dose reduction; "
            "SIBLING DONOR EXCLUSION (BRCA2 / FA-D1): "
            "  Sibling HSCT donor for AML/MDS: EXCLUDE FA-D1 biallelic BRCA2 MANDATORY; "
            "  Sibling may carry same biallelic FA-D1 alleles; MUD preferred; "
        ),
        "inheritance": "Biallelic AR LOF (FA-D1); monoallelic AD LOF (HBOC, breast/ovarian); FA-D1 biallelic: extreme rarity; hepatoblastoma part of FA-D1 multi-tumour cluster; sibling exclusion MANDATORY for HSCT donors",
        "cancer_risk": "FA-D1 biallelic: hepatoblastoma PATHOGNOMONIC (5-10%); AML 25-35% by age 10yr; Wilms PATHOGNOMONIC; medulloblastoma PATHOGNOMONIC; RMS embryonal PATHOGNOMONIC; HBOC monoallelic: breast 50-85%, ovarian 15-30%",
        "pathognomonic": "Hepatoblastoma PATHOGNOMONIC FA-D1 biallelic BRCA2; DEB/MMC chromosomal fragility PATHOGNOMONIC FA; multi-tumour cluster (HBL+AML+Wilms+medulloblastoma) PATHOGNOMONIC FA-D1; AVOID alkylating agents",
        "surveillance_key": "DEB/MMC at diagnosis; AFP + liver US from birth (FA-D1 surveillance); AVOID cyclophosphamide/alkylating agents; cisplatin dose-reduced; SIBLING DONOR EXCLUSION; cascade testing biallelic FA-D1 siblings",
        "key_distinctions": [
            "HEPATOBLASTOMA-PATHOGNOMONIC-FA-D1-BIALLELIC-BRCA2",
            "DEB-MMC-CHROMOSOMAL-FRAGILITY-PATHOGNOMONIC-FA",
            "AVOID-ALKYLATING-CYCLOPHOSPHAMIDE-ABSOLUTELY",
            "SIBLING-DONOR-EXCLUSION-MANDATORY",
            "MULTI-TUMOUR-CLUSTER-HBL-AML-WILMS-MEDULLO",
            "CISPLATIN-DOSE-REDUCED-FA-D1",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-Hepatoblastoma-1-3pct-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour suppressor protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; tetramer in active form; "
            "  N-terminal transactivation domains (TAD1 aa 1-40, TAD2 aa 40-61); "
            "  Proline-rich region (aa 61-94): apoptosis regulation; "
            "  DNA-binding domain (DBD, aa 102-292): most mutations cluster here; "
            "  Tetramerisation domain (aa 323-356); "
            "  R175H, R248W, R248Q, R273H, R273C: hotspot dominant-negative GOF mutations; "
            "HEPATOBLASTOMA IN LFS (TP53): "
            "  OMIM 151623; LFS tumour spectrum: sarcoma + brain + breast + ACC + leukaemia + adrenal; "
            "  Hepatoblastoma: component of LFS (1-3% of LFS); childhood onset <7yr; "
            "  Hepatoblastoma in LFS: less common than ACC (50-70%) or sarcoma (30-50%); "
            "  BUT hepatoblastoma can be sentinel LFS diagnosis in infant before other LFS features; "
            "  TP53 R337H (Brazilian founder): 1/300 South Brazilians; ACC PATHOGNOMONIC; HBL reported; "
            "  Anaplastic hepatoblastoma: TP53 somatic mutation in aggressive HBL (PRETEXT IV); "
            "AVOID RADIATION ABSOLUTELY (LFS): "
            "  Germline TP53 LFS: AVOID RADIATION ABSOLUTELY (secondary tumour risk); "
            "  HBL resection surgery: standard (surgery-first after chemotherapy downstaging); "
            "  Chemotherapy: PLADO preferred; avoid radiotherapy entirely if TP53 germline; "
            "  WBMRI Toronto annually (NOT CT-abdomen surveillance); "
            "  Paediatric ACC 50-70% TP53 germline PATHOGNOMONIC; "
        ),
        "inheritance": "Autosomal dominant LOF; 50% de novo; near-complete penetrance >90% lifetime cancer risk; dominant-negative GOF hotspot variants worsen phenotype; R337H Brazilian founder 1/300 population",
        "cancer_risk": "LFS: sarcoma 30-50% dominant; ACC 50-70% PATHOGNOMONIC; brain tumour 15-20%; breast <45yr 25-35%; hepatoblastoma 1-3% (childhood); leukaemia ALL 5-10%; t-AML post-therapy; HBL sentinel LFS diagnosis",
        "pathognomonic": "Paediatric ACC 50-70% TP53 germline PATHOGNOMONIC; hepatoblastoma as sentinel LFS tumour in infant PATHOGNOMONIC if TP53 family history; WBMRI annually NOT CT; AVOID RADIATION ABSOLUTELY; R337H South Brazilian PATHOGNOMONIC",
        "surveillance_key": "WBMRI Toronto annually (NOT CT/PET); AVOID RADIATION ABSOLUTELY; HBL: PLADO chemotherapy + surgery; avoid radiotherapy; ACC surveillance from birth (q3-6m US + AFP); annual brain MRI; cascade 50% risk",
        "key_distinctions": [
            "LFS-HEPATOBLASTOMA-1-3PCT-CHILDHOOD",
            "AVOID-RADIATION-ABSOLUTELY",
            "WBMRI-TORONTO-ANNUALLY-NOT-CT",
            "ACC-50-70PCT-PATHOGNOMONIC-LFS",
            "R337H-SOUTH-BRAZILIAN-1IN300-FOUNDER",
            "HBL-SENTINEL-LFS-DIAGNOSIS-INFANT",
        ],
    },
    {
        "gene": "GPC3",
        "protein": (
            "GPC3 -- Xq26.2 X-linked-Recessive -- 580aa -- "
            "Glypican3-66kDa-HSPG-GPI-Anchored-SGBS-Overgrowth-Hepatoblastoma-"
            "5-8pct-Visceral-Anomalies-PATHOGNOMONIC-OMIM-300037"
        ),
        "locus": "Xq26.2",
        "protein_size": (
            "580 aa / 66 kDa / Xq26.2 GPC3 encodes glypican-3 (heparan sulfate proteoglycan 3): "
            "STRUCTURE: "
            "  580 aa / 66 kDa; heparan sulfate proteoglycan (HSPG); GPI-anchored to cell surface; "
            "  Signal peptide (aa 1-24); "
            "  GPC3 core protein (aa 25-556): 14 cysteine residues forming disulfide bonds; "
            "  GPI anchor attachment site (aa 557-580); "
            "  Heparan sulfate attachment sites (Ser495, Ser509); "
            "  GPC3 function: co-receptor for Wnt, FGF, IGF2, Hedgehog signalling; "
            "  GPC3 LOF -> overgrowth (normally suppresses IGF2 proliferation signal); "
            "  GPC3 overexpressed in hepatoblastoma/HCC -> diagnostic serum marker; "
            "SIMPSON-GOLABI-BEHMEL SYNDROME (SGBS): "
            "  OMIM 312870; X-linked recessive (males severely affected); "
            "  Cardinal features: macrosomia + macrocephaly + coarse facies + polydactyly; "
            "  Supernumerary nipples PATHOGNOMONIC SGBS; "
            "  Visceral anomalies: cardiac (50%), renal (40%), intestinal malrotation (25%); "
            "  Intellectual disability: mild-moderate in most males; "
            "HEPATOBLASTOMA IN SGBS: "
            "  Hepatoblastoma: 5-8% of SGBS males; "
            "  Embryonal + fetal histology predominant; "
            "  Wilms tumour: also elevated in SGBS (hepatoblastoma + Wilms = embryonal tumour cluster); "
            "  AFP surveillance: AFP + liver US q3-6m from birth to 7yr; "
            "  GPC3 serum level elevated in HBL: diagnostic adjunct (not surveillance primary); "
            "GPC3 AS HEPATOBLASTOMA BIOMARKER: "
            "  GPC3 overexpressed in >90% of HBL: immunohistochemistry staining PATHOGNOMONIC; "
            "  Serum GPC3 (sGPC3) elevated in HBL: sensitivity ~70%; less useful than AFP; "
            "  GPC3 CAR-T therapy in HBL: preclinical trials (paediatric liver cancer potential); "
        ),
        "inheritance": "X-linked recessive; males severely affected; heterozygous females: mild macrosomia/supernumerary nipples; GPC3 deletion/frameshift/missense; 50% of carrier females' sons affected; prenatal diagnosis possible",
        "cancer_risk": "Hepatoblastoma 5-8% (SGBS males); Wilms tumour elevated; adrenocortical tumours (rare); neuroblastoma (case reports); solid tumour cluster embryonal histology; no adult cancer predisposition established",
        "pathognomonic": "Supernumerary nipples PATHOGNOMONIC SGBS; hepatoblastoma in overgrowth syndrome male (macrosomia/macrocephaly) PATHOGNOMONIC SGBS GPC3; GPC3 IHC staining of HBL tumour PATHOGNOMONIC (>90% of HBL); visceral anomalies + overgrowth + HBL",
        "surveillance_key": "AFP + liver US q3-6m from birth to 7yr; echocardiography (cardiac anomalies 50%); renal US from birth; GPC3 IHC diagnostic for HBL pathology; serum GPC3 adjunct; PLADO chemotherapy + resection standard",
        "key_distinctions": [
            "SGBS-HEPATOBLASTOMA-5-8PCT-MALES",
            "SUPERNUMERARY-NIPPLES-PATHOGNOMONIC-SGBS",
            "GPC3-IHC-POSITIVE-GT90PCT-HBL-PATHOGNOMONIC",
            "X-LINKED-RECESSIVE-MALES-SEVERELY-AFFECTED",
            "VISCERAL-ANOMALIES-CARDIAC-RENAL-MALROTATION",
            "SERUM-GPC3-ELEVATED-HBL-ADJUNCT-NOT-PRIMARY",
        ],
    },
    {
        "gene": "NSD1",
        "protein": (
            "NSD1 -- 5q35.3 Autosomal-Dominant-LOF-Haploinsufficiency -- 2696aa -- "
            "NSD1-304kDa-Histone-H3K36-Methyltransferase-Sotos-Overgrowth-"
            "Macrocephaly-PATHOGNOMONIC-Hepatoblastoma-5-10x-OMIM-117550"
        ),
        "locus": "5q35.3",
        "protein_size": (
            "2696 aa / 304 kDa / 5q35.3 NSD1 encodes nuclear receptor binding SET domain protein 1: "
            "STRUCTURE: "
            "  2696 aa / 304 kDa; histone methyltransferase; "
            "  N-terminal NSD1 NR-binding domains (NSD1_N, aa 1-500); "
            "  PHD fingers (6 total: aa 850-2300): histone recognition; "
            "  SET domain (aa 2087-2228): H3K36me2 methyltransferase catalytic; "
            "  PWWP domain (aa 2300-2400): chromatin targeting; "
            "  NSD1 function: H3K36 dimethylation -> gene activation vs repression (context-dependent); "
            "  NSD1 LOF (haploinsufficiency): overgrowth + intellectual disability; "
            "  NSD1 GOF/overexpression: AML (NSD1 fusion in t-AML), paediatric HCC; "
            "SOTOS SYNDROME: "
            "  OMIM 117550; AD; NSD1 haploinsufficiency; "
            "  Cardinal triad: overgrowth + macrocephaly + intellectual disability PATHOGNOMONIC; "
            "  Advanced bone age (>2 SD): PATHOGNOMONIC Sotos; "
            "  Characteristic facies: prominent forehead + hypertelorism + pointed chin (metopic ridge); "
            "  Downslanted palpebral fissures: PATHOGNOMONIC Sotos facies; "
            "HEPATOBLASTOMA IN SOTOS: "
            "  Hepatoblastoma: 5-10x elevated in Sotos syndrome; "
            "  Absolute risk: ~1-3% of Sotos patients; "
            "  Age at HBL: usually <5yr (same as sporadic HBL); "
            "  NSD1 LOF -> epigenetic deregulation -> hepatocyte proliferation; "
            "  5q35 microdeletion (common cause of Sotos): encompasses NSD1; "
            "  Wilms tumour: also reported in Sotos (embryonal tumour cluster); "
            "SURVEILLANCE SOTOS: "
            "  AFP + liver US q3-6m from birth to 5-7yr; "
            "  Renal US: Wilms surveillance from birth to 7yr; "
            "  Cardiac assessment: congenital heart disease 5-10% Sotos; "
        ),
        "inheritance": "Autosomal dominant haploinsufficiency; ~60% de novo; NSD1 intragenic mutations or 5q35 microdeletion (MLPA mandatory); variable intellectual disability; overgrowth penetrant; HBL/Wilms penetrance 1-3%",
        "cancer_risk": "Hepatoblastoma 5-10x elevated (1-3% Sotos); Wilms tumour (embryonal cluster); AML (NSD1 fusion; somatic, distinct from germline LOF); haematological malignancies rare with germline NSD1 LOF alone",
        "pathognomonic": "Cardinal Sotos triad (overgrowth + macrocephaly + intellectual disability) PATHOGNOMONIC; downslanted palpebral fissures + prominent forehead PATHOGNOMONIC facies; hepatoblastoma in Sotos child PATHOGNOMONIC NSD1 germline; advanced bone age PATHOGNOMONIC",
        "surveillance_key": "AFP + liver US q3-6m birth to 7yr; renal US from birth (Wilms); MLPA for 5q35 deletion if NSD1 sequencing negative; cardiac assessment; PLADO standard HBL treatment; cascade 50% risk; MLPA diagnoses ~50% Sotos",
        "key_distinctions": [
            "SOTOS-TRIAD-OVERGROWTH-MACROCEPHALY-ID-PATHOGNOMONIC",
            "HEPATOBLASTOMA-5-10X-ELEVATED-SOTOS",
            "5Q35-MICRODELETION-MLPA-MANDATORY",
            "ADVANCED-BONE-AGE-GT2SD-PATHOGNOMONIC",
            "DOWNSLANTED-PALPEBRAL-FISSURES-PATHOGNOMONIC",
            "WILMS-TUMOUR-EMBRYONAL-CLUSTER-NSD1",
        ],
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-219kDa-RNase-III-miRNA-Processor-PPB-PATHOGNOMONIC-"
            "Hepatoblastoma-2-4x-AVOID-Radiation-CT-Chest-Siblings-LT-8yr-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 219 kDa / 14q32.13 DICER1 encodes DICER1 ribonuclease III: "
            "STRUCTURE: "
            "  1922 aa / 219 kDa; RNase III endonuclease; "
            "  DEAD-box helicase domain (aa 1-610); "
            "  PAZ domain (aa 820-971): dsRNA 3'-end binding; "
            "  Platform domain + connector helix (aa 972-1226); "
            "  RNase IIIa (aa 1258-1367): cleaves sense strand of pre-miRNA; "
            "  RNase IIIb (aa 1371-1487): cleaves antisense strand of pre-miRNA; "
            "    RNase IIIb hotspot mutations: E1705/D1709/D1810/G1809/E1813 (somatic second hit); "
            "    Most common somatic second hit in DICER1 tumours; "
            "  dsRBD domain (aa 1760-1822): dsRNA binding auxiliary; "
            "  DICER1 function: cleaves pre-miRNA hairpin -> mature miRNA duplex; "
            "DICER1 TUMOUR PREDISPOSITION SYNDROME: "
            "  OMIM 606241; AD; >50 tumour types associated; "
            "  PPB (pleuropulmonary blastoma): PATHOGNOMONIC DICER1 (type I/II/III); "
            "  Cervical ERMS (embryonal rhabdomyosarcoma of cervix): PATHOGNOMONIC; "
            "  Cystic nephroma: PATHOGNOMONIC DICER1; "
            "  Thyroid multinodular goitre + thyroid cancer (follicular): 3-4x elevated; "
            "  Ovarian Sertoli-Leydig cell tumour (SLCT): PATHOGNOMONIC DICER1; "
            "  Bladder ERMS: PATHOGNOMONIC; "
            "HEPATOBLASTOMA IN DICER1: "
            "  Hepatoblastoma: 2-4x elevated in DICER1 tumour spectrum; "
            "  Part of broader hepatic tumour spectrum (HCC also reported); "
            "  AFP + liver US from birth: q3-6m surveillance in DICER1; "
            "  AVOID radiation in children with DICER1 (radiation-induced secondary tumours); "
            "  CT chest siblings <8yr: PPB screening (lung cysts early stage I); "
        ),
        "inheritance": "Autosomal dominant LOF; 50% de novo; incomplete penetrance (~50%); second hit required for tumour (somatic RNase IIIb hotspot mutation); heterozygous germline LOF + somatic second hit biallelic inactivation in tumour",
        "cancer_risk": "PPB 10-15% cumulative; SLCT 10-15% (ovarian); thyroid 3-4x; cervical ERMS PATHOGNOMONIC; cystic nephroma PATHOGNOMONIC; hepatoblastoma 2-4x; bladder ERMS PATHOGNOMONIC; pineoblastoma rare; peritoneal sarcoma",
        "pathognomonic": "PPB PATHOGNOMONIC DICER1 (type I lung cysts in infant); cervical ERMS PATHOGNOMONIC; cystic nephroma PATHOGNOMONIC; ovarian SLCT PATHOGNOMONIC; hepatoblastoma in DICER1 syndrome; CT chest siblings <8yr MANDATORY screening",
        "surveillance_key": "AFP + liver US q3-6m from birth; CT chest siblings <8yr (PPB lung cysts); avoid radiation; thyroid US annually from age 8yr; ovarian US female from age 8yr; PLADO standard HBL; cascade 50% risk",
        "key_distinctions": [
            "PPB-PATHOGNOMONIC-DICER1",
            "HEPATOBLASTOMA-2-4X-ELEVATED",
            "AVOID-RADIATION-CHILDREN-DICER1",
            "CT-CHEST-SIBLINGS-LT-8YR-MANDATORY",
            "RNASE-IIIB-HOTSPOT-SOMATIC-SECOND-HIT",
            "CERVICAL-ERMS-PATHOGNOMONIC-DICER1",
        ],
    },
    {
        "gene": "NFE2L2",
        "protein": (
            "NFE2L2 -- 2q31.2 Autosomal-Dominant-GOF-germline -- 605aa -- "
            "NRF2-68kDa-bZIP-CNC-TF-Oxidative-Stress-Response-Germline-GOF-"
            "Paediatric-HCC-Early-Onset-20s-KEAP1-Independent-Very-Rare-OMIM-600492"
        ),
        "locus": "2q31.2",
        "protein_size": (
            "605 aa / 68 kDa / 2q31.2 NFE2L2 encodes nuclear factor erythroid 2-related factor 2 (NRF2): "
            "STRUCTURE: "
            "  605 aa / 68 kDa; basic leucine zipper (bZIP) transcription factor; CNC superfamily; "
            "  Neh2 domain (aa 17-96): KEAP1 binding (DLG + ETGE degrons); "
            "    ETGE motif (aa 79-82): primary KEAP1 binding; most common GOF mutation site; "
            "    DLG motif (aa 29-32): secondary KEAP1 binding; "
            "  Neh4-Neh5 (aa 98-242): CBP/p300 transcriptional activation; "
            "  Neh6 (aa 327-399): beta-TrCP degradation domain; "
            "  Neh1 (aa 435-561): bZIP DNA binding; heterodimerises with small Maf proteins; "
            "  Neh3 (aa 561-605): C-terminal transactivation; "
            "  NRF2 function: master antioxidant transcription factor; ARE (antioxidant response element); "
            "  Somatic NRF2 GOF (ETGE/DLG mutations): common in lung, liver, oesophageal cancers; "
            "  KEAP1 LOF (somatic): equivalent NRF2 activation via protein stabilisation; "
            "GERMLINE NFE2L2 GOF (HEREDITARY): "
            "  Very rare (<100 families reported in literature); "
            "  Germline GOF mutations: primarily ETGE domain (Asp77Glu/Glu78Gln/Leu80Pro/Ile81Ser); "
            "  Constitutional NRF2 hyperactivation: cytoprotection of pre-neoplastic hepatocytes; "
            "  Paediatric HCC/hepatoblastoma: early-onset (teens-20s; earlier than sporadic HCC); "
            "  Kerins et al. 2018 (first germline NFE2L2 GOF cohort); "
            "  Chronic liver disease NOT required: NRF2 GOF drives transformation independent of cirrhosis; "
            "NRF2 PATHWAY TARGETING: "
            "  Sulforaphane (broccoli-derived NRF2 activator): paradoxical - therapeutic in deficiency; "
            "  NRF2 inhibitors: ML385, brusatol (research; no clinical approval 2025); "
            "  Sorafenib/lenvatinib: standard HCC systemic therapy regardless of NRF2 status; "
            "  TACE (transarterial chemoembolisation): unresectable HCC standard bridging; "
            "  Liver transplant: HCC within Milan criteria (single <5cm or 3x <3cm); "
        ),
        "inheritance": "Autosomal dominant GOF; predominantly de novo (very rare germline); ETGE domain mutations most pathogenic; NFE2L2 constitutive activation -> KEAP1-independent antioxidant programme; very rare syndrome (<100 published cases)",
        "cancer_risk": "Paediatric/early-onset HCC (teens-30s); hepatoblastoma component reported; early-onset without cirrhosis PATHOGNOMONIC germline NFE2L2 GOF; lung cancer elevated (somatic NRF2 most common); no established extra-hepatic predisposition in germline GOF families",
        "pathognomonic": "Early-onset HCC (teens-20s) without chronic liver disease or cirrhosis PATHOGNOMONIC germline NFE2L2 GOF; ETGE domain mutation in HCC from young patient = germline NFE2L2 GOF testing; NRF2 overexpression on IHC in absence of KEAP1 mutation",
        "surveillance_key": "AFP + liver US q3-6m from age 10yr; MRI liver from age 16yr (better sensitivity than US for HCC); sorafenib/lenvatinib systemic; TACE bridging; transplant Milan criteria; NRF2 inhibitors investigational; cascade 50% risk",
        "key_distinctions": [
            "EARLY-ONSET-HCC-TEENS-20S-WITHOUT-CIRRHOSIS-PATHOGNOMONIC",
            "ETGE-DOMAIN-GOF-MUTATIONS-PATHOGNOMONIC",
            "GERMLINE-NFE2L2-VERY-RARE-LT100-FAMILIES",
            "KEAP1-INDEPENDENT-NRF2-HYPERACTIVATION",
            "MRI-LIVER-FROM-AGE-16YR-SURVEILLANCE",
            "NRF2-INHIBITORS-ML385-BRUSATOL-INVESTIGATIONAL",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Simulated tumour types per gene (hepatoblastoma context)
# ─────────────────────────────────────────────────────────────────────────────
TUMOUR_TYPES_BY_GENE = {
    "APC":    ["Hepatoblastoma-epithelial", "Hepatoblastoma-mixed", "Hepatoblastoma-fetal", "Colorectal-adenoma", "Desmoid"],
    "CTNNB1": ["Hepatoblastoma-mixed-CTNNB1", "Hepatoblastoma-fetal-CTNNB1", "Hepatoblastoma-embryonal", "Hepatoblastoma-PRETEXT-IV", "HCC-HBL-transition"],
    "BRCA2":  ["Hepatoblastoma-FA-D1", "AML-FA-D1", "Wilms-FA-D1", "Medulloblastoma-FA-D1", "RMS-embryonal-FA-D1"],
    "TP53":   ["Hepatoblastoma-LFS", "Sarcoma-LFS", "ACC-LFS", "Brain-LFS", "ALL-LFS"],
    "GPC3":   ["Hepatoblastoma-SGBS", "Wilms-SGBS", "Hepatoblastoma-embryonal-SGBS", "Hepatoblastoma-fetal-SGBS", "Hepatoblastoma-mixed-SGBS"],
    "NSD1":   ["Hepatoblastoma-Sotos", "Wilms-Sotos", "Hepatoblastoma-fetal-Sotos", "Hepatoblastoma-mixed-Sotos", "Hepatoblastoma-embryonal-Sotos"],
    "DICER1": ["Hepatoblastoma-DICER1", "PPB-type-I-DICER1", "PPB-type-III-DICER1", "Cystic-nephroma-DICER1", "SLCT-DICER1"],
    "NFE2L2": ["HCC-early-onset", "Hepatoblastoma-NFE2L2", "HCC-NRF2-GOF", "HCC-PRETEXT-III", "HCC-Milan-eligible"],
}

VARIANTS_BY_GENE = {
    "APC":    ["p.Gln1338*", "p.Arg1450*", "p.Glu1309Asp", "p.Gln545*", "p.Arg876*"],
    "CTNNB1": ["exon3-inframe-del-somatic", "p.Ser45Phe-GOF", "p.Thr41Ala-GOF", "p.Ser33Cys-GOF", "exon3-del-germline-GOF"],
    "BRCA2":  ["p.Trp31*", "p.Lys3326*", "p.Glu1308*", "IVS7+2T>G", "p.Asn991Ile"],
    "TP53":   ["p.Arg175His", "p.Arg248Trp", "p.Arg248Gln", "p.Arg273His", "p.Arg337His"],
    "GPC3":   ["exon3-del-XLR", "p.Arg132*", "p.Trp396*", "exon1-del-XLR", "p.Gly423Ser"],
    "NSD1":   ["5q35-microdeletion", "p.Arg1984*", "p.Arg1922Gln", "p.Gln2127*", "p.Leu1944Pro"],
    "DICER1": ["p.Arg1905*", "p.Asp1810Gly-somatic-RNaseIIIb", "p.Glu1813Gly-somatic", "p.Glu1705*", "p.Val1711Glu"],
    "NFE2L2": ["p.Asp77Glu-GOF-ETGE", "p.Glu78Gln-GOF-ETGE", "p.Leu80Pro-GOF-ETGE", "p.Ile81Ser-GOF-ETGE", "p.Arg34Gly-GOF-DLG"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "APC": [
        "PLADO protocol: cisplatin + doxorubicin (SIOPEL-6) for hepatoblastoma",
        "Surgical resection: primary goal (hepatectomy/partial hepatectomy after downstaging)",
        "Colonoscopy from age 10-12yr (FAP standard): annual surveillance",
        "AFP + liver US q3-6m from birth to 7yr (hepatoblastoma early detection)",
        "Nirogacestat (gamma-secretase inhibitor) for progressive unresectable desmoid (FDA 2023)",
    ],
    "CTNNB1": [
        "SIOPEL-6: cisplatin monotherapy (low-risk PRETEXT I-II, AFP >100 ng/mL, no metastases)",
        "PLADO (cisplatin + doxorubicin): high-risk or PRETEXT III-IV hepatoblastoma",
        "AFP monitoring every cycle: normalisation at end of treatment = complete response criterion",
        "Liver transplant: unresectable hepatoblastoma responsive to chemotherapy (Milan criteria not applicable)",
        "Somatic CTNNB1 analysis on tumour: prognostic (large exon 3 deletion = aggressive)",
    ],
    "BRCA2": [
        "FA-D1 hepatoblastoma: AVOID cyclophosphamide; cisplatin dose-reduced (50-75% of standard)",
        "SIOPEL-6 cisplatin modified (reduced dose FA-D1): response-adapted escalation",
        "Surgical resection: mainstay if feasible after chemotherapy downstaging",
        "Liver transplant: option if unresectable with good chemo response (dose-reduced bridging)",
        "Sibling donor exclusion MANDATORY before any haematopoietic transplant in FA-D1",
    ],
    "TP53": [
        "PLADO hepatoblastoma chemotherapy: cisplatin + doxorubicin (LFS does NOT restrict cisplatin)",
        "AVOID radiotherapy absolutely (secondary tumour risk with germline TP53)",
        "Surgical resection: primary endpoint; achieve R0 if possible",
        "WBMRI Toronto annually: whole-body MRI surveillance for LFS multi-organ monitoring",
        "Annual rapid brain MRI; breast MRI from age 20yr; ACC surveillance q3-6m abdominal MRI",
    ],
    "GPC3": [
        "PLADO protocol (cisplatin + doxorubicin): standard hepatoblastoma treatment",
        "AFP monitoring: primary response marker (same as sporadic HBL)",
        "Renal US from birth: Wilms surveillance q3-6m to age 7yr",
        "Echocardiography: congenital heart disease screening at birth (50% SGBS)",
        "AFP + liver US q3-6m from birth to 7yr (hepatoblastoma surveillance SGBS)",
    ],
    "NSD1": [
        "PLADO protocol: standard hepatoblastoma treatment (NSD1 does not alter chemosensitivity)",
        "Liver US + AFP q3-6m from birth to 7yr (Sotos hepatoblastoma surveillance)",
        "Renal US from birth to 7yr: Wilms tumour surveillance in Sotos",
        "MLPA 5q35 deletion testing: if NSD1 sequencing negative (50% Sotos = deletion)",
        "Cardiac assessment at birth (5-10% Sotos have congenital heart disease)",
    ],
    "DICER1": [
        "PLADO chemotherapy for hepatoblastoma; standard SIOPEL protocols",
        "CT chest siblings <8yr: PPB type I (lung cysts) screening MANDATORY",
        "AVOID radiation in children with DICER1 (increased radiation-induced secondary tumour risk)",
        "Thyroid US annually from age 8yr; ovarian US annually female from age 8yr",
        "AFP + liver US q3-6m from birth to 7yr; resection after chemotherapy downstaging",
    ],
    "NFE2L2": [
        "Sorafenib / lenvatinib: first-line systemic for unresectable HCC (NRF2 GOF does not predict resistance)",
        "TACE (transarterial chemoembolisation): bridging to transplant for unresectable HCC",
        "Liver transplant: HCC within Milan criteria (single <5cm or 3x <3cm); best OS",
        "MRI liver from age 16yr: better HCC detection than US in young patients",
        "AFP monitoring q6m from age 10yr; MRI annually from age 16yr",
    ],
}

SURVEILLANCE_BY_GENE = {
    "APC": [
        "AFP + liver US every 3-6 months from birth to age 7yr",
        "Colonoscopy annually from age 10-12yr (polyposis surveillance)",
        "EGD from age 20yr or at polyposis onset (gastric/duodenal polyps)",
        "No family HSCT donor without APC germline exclusion if allograft needed",
        "CASCADE: 50% offspring risk; prenatal diagnosis available",
    ],
    "CTNNB1": [
        "AFP + liver US q3-6m from birth (germline CTNNB1 GOF if familial HBL)",
        "Somatic CTNNB1 testing on tumour tissue (prognostic, not hereditary surveillance)",
        "PLADO response monitoring: AFP normalisation = complete response",
        "Cascade testing: germline CTNNB1 GOF -> 50% offspring risk; de novo in most",
        "No polyposis surveillance (CTNNB1 GOF ≠ FAP/APC pathway)",
    ],
    "BRCA2": [
        "DEB/MMC chromosomal fragility test at diagnosis (FA-D1 confirmation)",
        "AFP + liver US from birth (FA-D1 hepatoblastoma surveillance)",
        "Annual CBC + BM biopsy (AML/MDS surveillance FA-D1)",
        "SIBLING DONOR EXCLUSION: exclude biallelic FA-D1 before allograft",
        "Monoallelic HBOC: breast MRI from age 25yr; ovarian BSO by 35-40yr",
    ],
    "TP53": [
        "WBMRI Toronto annually (NOT CT/PET); whole-body MRI multi-organ LFS surveillance",
        "Annual rapid brain MRI + abdominal MRI + AFP",
        "Breast MRI from age 20yr; adrenal US q3-6m from birth to age 10yr",
        "AVOID radiation absolutely; no surveillance PET/CT",
        "CASCADE: all first-degree relatives; 50% risk",
    ],
    "GPC3": [
        "AFP + liver US q3-6m from birth to 7yr (hepatoblastoma surveillance SGBS)",
        "Renal US from birth to 7yr (Wilms surveillance)",
        "Echocardiography at birth (congenital heart disease 50% SGBS)",
        "Karyotype + chromosome microarray: characterise deletion extent",
        "Cascade: 50% of carrier females' sons affected (X-linked)",
    ],
    "NSD1": [
        "AFP + liver US q3-6m from birth to 7yr (Sotos hepatoblastoma surveillance)",
        "Renal US from birth to 7yr (Wilms tumour surveillance)",
        "MLPA 5q35 if NSD1 sequencing negative (detects ~50% deletions)",
        "Cardiac assessment at birth; annual growth monitoring",
        "CASCADE: 50% offspring risk; predominantly de novo",
    ],
    "DICER1": [
        "AFP + liver US q3-6m from birth to 7yr",
        "CT chest siblings <8yr (PPB lung cysts type I - MANDATORY)",
        "Thyroid US annually from age 8yr",
        "Ovarian US annually from age 8yr (female); SLCT screening",
        "AVOID radiation in children; cascade 50% risk",
    ],
    "NFE2L2": [
        "AFP q6m from age 10yr; liver US q6m from age 10yr",
        "MRI liver annually from age 16yr",
        "Liver function tests annually from diagnosis",
        "TACE bridging; transplant assessment if within Milan criteria",
        "CASCADE: 50% offspring risk (de novo in most cases)",
    ],
}


def _make_patient(gene_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_index]["gene"]
    tumour = rng.choice(TUMOUR_TYPES_BY_GENE[gene])
    variant = rng.choice(VARIANTS_BY_GENE[gene])

    # Age at diagnosis (hepatoblastoma context - varies by gene)
    if gene in ("APC", "CTNNB1"):
        age = rng.randint(0, 5)
    elif gene in ("GPC3", "NSD1", "DICER1"):
        age = rng.randint(0, 6)
    elif gene == "BRCA2":
        age = rng.randint(0, 7)
    elif gene == "TP53":
        age = rng.randint(0, 8)
    elif gene == "NFE2L2":
        age = rng.randint(12, 28)
    else:
        age = rng.randint(0, 5)

    # Gene-specific CR rates (hepatoblastoma context)
    cr_rates = {
        "APC": 0.82, "CTNNB1": 0.79, "BRCA2": 0.60, "TP53": 0.72,
        "GPC3": 0.75, "NSD1": 0.80, "DICER1": 0.77, "NFE2L2": 0.50,
    }
    cr = rng.random() < cr_rates.get(gene, 0.75)

    # Radiation rates (low in hepatoblastoma - surgery-based disease)
    rad_rates = {
        "APC": 0.03, "CTNNB1": 0.04, "BRCA2": 0.02, "TP53": 0.00,
        "GPC3": 0.03, "NSD1": 0.04, "DICER1": 0.01, "NFE2L2": 0.12,
    }
    radiation = rng.random() < rad_rates.get(gene, 0.03)

    # Resection rates (surgery primary endpoint)
    resection_rates = {
        "APC": 0.85, "CTNNB1": 0.80, "BRCA2": 0.72, "TP53": 0.78,
        "GPC3": 0.80, "NSD1": 0.83, "DICER1": 0.79, "NFE2L2": 0.55,
    }
    targeted = rng.random() < resection_rates.get(gene, 0.80)  # 'targeted' = resection achieved

    # Transplant rates (liver transplant in unresectable HBL)
    tx_rates = {
        "APC": 0.12, "CTNNB1": 0.18, "BRCA2": 0.20, "TP53": 0.15,
        "GPC3": 0.14, "NSD1": 0.12, "DICER1": 0.13, "NFE2L2": 0.30,
    }
    transplant = rng.random() < tx_rates.get(gene, 0.15)

    relapse = rng.random() < (0.18 if cr else 0.65)

    return {
        "gene": gene,
        "seed": seed,
        "age_dx": age,
        "tumour_type": tumour,
        "variant": variant,
        "cr": cr,
        "radiation": radiation,
        "targeted": targeted,      # resection achieved
        "transplant": transplant,  # liver transplant
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
    resection_pct = round(100 * sum(p["targeted"] for p in cohort) / n)
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
            "resection_pct": round(100 * sum(p["targeted"] for p in gp) / gn),
            "transplant_pct": round(100 * sum(p["transplant"] for p in gp) / gn),
            "relapse_pct": round(100 * sum(p["relapse"] for p in gp) / gn),
            "cancer_risk": ginfo["cancer_risk"],
        })

    return {
        "atlas": "Hereditary-Hepatoblastoma-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference: APC · CTNNB1 · BRCA2 · TP53 · GPC3 · NSD1 · DICER1 · NFE2L2",
        "seeds": "3294-3301",
        "total_patients": n,
        "cr_pct": cr_pct,
        "radiation_pct": radiation_pct,
        "resection_pct": resection_pct,
        "transplant_pct": transplant_pct,
        "relapse_pct": relapse_pct,
        "mean_age_dx": mean_age,
        "gene_summaries": gene_summaries,
        "top_tumor_types": top_tumours,
        "key_management_rules": [
            "APC FAP: AFP + liver US q3-6m from BIRTH to age 7yr — hepatoblastoma 50-100x; colonoscopy from age 10-12yr",
            "CTNNB1 somatic: 90% of all sporadic hepatoblastomas — PLADO/SIOPEL-6; AFP normalisation = CR criterion",
            "BRCA2 FA-D1: hepatoblastoma PATHOGNOMONIC biallelic — AVOID alkylating agents; cisplatin dose-reduced; SIBLING EXCLUSION MANDATORY",
            "TP53 LFS: hepatoblastoma sentinel tumour in infant — AVOID RADIATION ABSOLUTELY; PLADO (not radiation); WBMRI Toronto annual",
            "GPC3 SGBS: AFP + liver US q3-6m from birth — hepatoblastoma 5-8% SGBS males; supernumerary nipples PATHOGNOMONIC",
            "NSD1 Sotos: AFP + liver US q3-6m from birth — hepatoblastoma 5-10x elevated; Sotos triad macrocephaly+overgrowth+ID PATHOGNOMONIC",
            "DICER1: AFP + liver US q3-6m from birth — CT chest siblings <8yr MANDATORY (PPB); AVOID radiation children",
            "NFE2L2 GOF: early-onset HCC teens-20s without cirrhosis PATHOGNOMONIC — MRI liver from age 16yr; transplant Milan criteria",
        ],
        "clinical_pearls": [
            "APC FAP: hepatoblastoma before polyposis — in infant with hepatoblastoma, ask family history for colorectal polyps/cancer; APC testing FIRST",
            "CTNNB1: somatic CTNNB1 mutations in 90% of HBL — do NOT order germline CTNNB1 unless ≥2 affected family members without APC",
            "BRCA2 FA-D1: in infant with hepatoblastoma — check DEB/MMC chromosomal fragility BEFORE starting cisplatin; reduce dose if FA confirmed",
            "TP53 LFS: hepatoblastoma + family history (ACC/sarcoma/brain/breast) = LFS — WBMRI annually; avoid any radiation surveillance (no CT/PET)",
            "GPC3 SGBS: macrosomia + supernumerary nipples + hepatoblastoma = SGBS until proven otherwise — GPC3 IHC on tumour always positive",
            "NSD1 Sotos: tall stature + macrocephaly + hepatoblastoma in toddler = Sotos — MLPA 5q35 mandatory (sequencing alone misses 50%)",
            "DICER1: hepatoblastoma + lung cysts in siblings = DICER1 syndrome — CT chest siblings <8yr MANDATORY; PPB type I resectable = excellent outcome",
            "NFE2L2: young adult HCC without viral hepatitis/cirrhosis/alcohol = NFE2L2 germline GOF testing; ETGE domain mutation pathognomonic",
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
            "resection_pct": round(100 * sum(p["targeted"] for p in gp) / gn),
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
    return {"atlas": "Hereditary-Hepatoblastoma-Predisposition-Atlas", "breakdown": breakdown}


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
        "atlas": "Hereditary-Hepatoblastoma-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "APC_FAP_HEPATOBLASTOMA": (
                "APC FAP: hepatoblastoma 50-100x elevated. AFP + liver US q3-6m from BIRTH to age 7yr. "
                "5' APC mutations (codons 200-1600) correlate with hepatoblastoma risk. "
                "Annual colonoscopy from age 10-12yr. Nirogacestat for progressive desmoid (FDA 2023)."
            ),
            "CTNNB1_SOMATIC_NOT_GERMLINE": (
                "CTNNB1: 90% of sporadic HBL carry somatic exon 3 GOF mutations. "
                "Somatic CTNNB1 does NOT indicate germline predisposition. "
                "Germline CTNNB1 GOF: very rare familial HBL without polyposis. AFP normalisation = CR."
            ),
            "BRCA2_FA_D1_AVOID_ALKYLATING": (
                "BRCA2 FA-D1: hepatoblastoma PATHOGNOMONIC biallelic. DEB/MMC FIRST before cisplatin. "
                "AVOID cyclophosphamide. Cisplatin dose-reduced in FA-D1. "
                "SIBLING DONOR EXCLUSION MANDATORY for any haematopoietic transplant."
            ),
            "TP53_LFS_AVOID_RADIATION": (
                "TP53 LFS: AVOID RADIATION ABSOLUTELY. Hepatoblastoma 1-3% LFS sentinel tumour in infants. "
                "PLADO chemotherapy + surgery (no radiotherapy). WBMRI Toronto annually (NOT CT/PET). "
                "R337H South Brazilian founder 1/300 population."
            ),
            "GPC3_SGBS": (
                "GPC3 SGBS: hepatoblastoma 5-8% of SGBS males. X-linked recessive. "
                "Supernumerary nipples PATHOGNOMONIC SGBS. GPC3 IHC positive in >90% of HBL. "
                "AFP + liver US q3-6m from birth; renal US (Wilms also elevated in SGBS)."
            ),
            "NSD1_SOTOS": (
                "NSD1 Sotos: hepatoblastoma 5-10x elevated. Sotos triad (overgrowth+macrocephaly+ID) PATHOGNOMONIC. "
                "MLPA 5q35 mandatory (detects ~50% Sotos = microdeletion; sequencing alone insufficient). "
                "AFP + liver US q3-6m from birth; renal US (Wilms also elevated)."
            ),
            "DICER1_PPB_HEPATOBLASTOMA": (
                "DICER1: hepatoblastoma 2-4x elevated. PPB PATHOGNOMONIC DICER1. "
                "CT chest siblings <8yr MANDATORY (PPB type I lung cysts = early resectable). "
                "AVOID radiation in children. AFP + liver US q3-6m from birth."
            ),
            "NFE2L2_EARLY_HCC": (
                "NFE2L2 germline GOF: early-onset HCC (teens-20s) WITHOUT cirrhosis PATHOGNOMONIC. "
                "Very rare (<100 families). ETGE domain mutations (Asp77/Glu78/Leu80/Ile81). "
                "MRI liver from age 16yr; sorafenib/lenvatinib systemic; transplant Milan criteria."
            ),
        },
        "cascade_testing_rule": (
            "Hereditary Hepatoblastoma Predisposition Atlas — Cascade Testing: "
            "APC: colonoscopy from age 10-12yr; AFP + liver US q3-6m from birth to 7yr in FAP families. "
            "CTNNB1 germline GOF: AFP + liver US if familial HBL without polyposis (exclude APC first). "
            "BRCA2 FA-D1: DEB/MMC test + sibling exclusion MANDATORY before any allograft. "
            "TP53 LFS: WBMRI annually; AVOID radiation; cascade first-degree relatives 50% risk. "
            "GPC3 SGBS: X-linked; 50% of carrier females' sons affected; echocardiography at birth. "
            "NSD1 Sotos: predominantly de novo; MLPA 5q35 mandatory; AFP + liver US birth to 7yr. "
            "DICER1: CT chest siblings <8yr (PPB); AFP + liver US from birth; avoid radiation. "
            "NFE2L2 GOF: MRI liver from age 16yr; very rare de novo; cascade 50% if familial."
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
    print(json.dumps(df["definitions"]["APC"], indent=2)[:1500])
