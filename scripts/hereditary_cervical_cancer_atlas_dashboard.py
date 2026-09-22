#!/usr/bin/env python3
"""Hereditary-Cervical-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
STK11   (LKB1 / Peutz-Jeghers Syndrome; 433aa; 19p13.3; AD LOF;
         OMIM 175200 PJS; perioral/buccal mucocutaneous melanin pigmentation PATHOGNOMONIC;
         SCTAT (sex cord tumour with annular tubules) PATHOGNOMONIC in females;
         Adenoma malignum / minimal deviation adenocarcinoma of cervix PATHOGNOMONIC;
         Cervical 10-13% lifetime; breast 45%; pancreatic 30-40%; HIGHEST hereditary cervical risk;
         seed SEED_BASE+0) .
BRCA1   (Breast Cancer Gene 1; 1863aa; 17q21.31; AD LOF;
         HBOC -- Hereditary Breast-Ovarian Cancer syndrome;
         Cervical adenocarcinoma 2-3x elevated; HR-deficient; olaparib PARP;
         BSO recommended age 35-40yr for ovarian risk reduction;
         Breast cancer 72%; ovarian 44%;
         seed SEED_BASE+1) .
TP53    (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome (LFS);
         Cervical carcinosarcoma/sarcoma rare elevated; if cervical cancer: SURGERY over radiotherapy;
         AVOID RADIATION ABSOLUTELY -- radiation-induced secondary sarcoma lethal;
         WBMRI annually Toronto Protocol; R337H Brazilian founder;
         seed SEED_BASE+2) .
MSH2    (MutS Homolog 2; 934aa; 2p21; AD LOF;
         Lynch Syndrome type 2 (HNPCC);
         Cervical adenocarcinoma 5-10% (endocervical glandular; NOT squamous);
         Endometrial 40-60% DOMINANT Lynch2;
         Muir-Torre sebaceous neoplasms PATHOGNOMONIC; EPCAM 3-prime deletion;
         seed SEED_BASE+3) .
FANCA   (Fanconi Anemia Complementation Group A; 1455aa; 16q24.3; AR LOF;
         Fanconi Anemia type A -- most common FA 60%;
         Cervical SCC ~150-200x elevated; HPV vaccination CRITICAL and MANDATORY;
         AVOID aldehyde/alcohol ABSOLUTELY; DEB test PATHOGNOMONIC;
         BMF + radial ray anomalies; HSCT curative for BMF (solid tumour risk persists);
         seed SEED_BASE+4) .
PTEN    (Phosphatase and Tensin Homolog; 403aa; 10q23.31; AD LOF;
         Cowden Syndrome / PHTS (PTEN Hamartoma Tumour Syndrome);
         Macrocephaly (HC >=97th centile) PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC;
         Cervical adenocarcinoma elevated; endometrial 28-44% DOMINANT;
         mTOR/Everolimus targeted; trichilemmoma PATHOGNOMONIC;
         seed SEED_BASE+5) .
ATM     (Ataxia-Telangiectasia Mutated; 3056aa; 11q22.3; AD/AR LOF;
         Ataxia-Telangiectasia biallelic;
         RADIOSENSITIVITY ABSOLUTE biallelic -- cervical chemoradiation = ABSOLUTE CI biallelic A-T;
         Monoallelic ATM: cervical 2-3x elevated;
         Ceralasertib ATRi + olaparib clinical trials;
         seed SEED_BASE+6) .
BRCA2   (Breast Cancer Gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC -- Hereditary Breast-Ovarian Cancer syndrome;
         Cervical adenocarcinoma 2-3x elevated; HR-deficient; cisplatin/olaparib;
         Fanconi Anemia type D1 (biallelic) -- medulloblastoma/Wilms/AML MOST SEVERE FA;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3222-3229)
"""
import random

SEED_BASE = 3222

ATLAS_GENES = [
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-48kDa-AMPK-Master-Kinase-PJS-Cervical-SCTAT-PATHOGNOMONIC-"
            "Adenoma-Malignum-PATHOGNOMONIC-Cervical-10-13pct-Breast-45pct-Pancreatic-30-40pct-OMIM-175200"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 encodes LKB1 (Liver Kinase B1 / Serine-Threonine Kinase 11): "
            "STRUCTURE: "
            "  433 aa / 48 kDa; serine/threonine kinase; "
            "  N-terminal nuclear localisation signal (NLS); "
            "  Kinase domain (aa 49-309): catalytic core; ATP-binding K78; activation loop T363; "
            "  C-terminal regulatory domain (aa 310-433): STRAD-alpha binding; "
            "  STK11 forms heterotrimeric complex: STK11-STRAD-alpha-MO25-alpha; "
            "  STRAD-alpha: pseudokinase activator -- requires STRAD for full kinase activity; "
            "  STK11 phosphorylates and activates AMPK (AMP-activated protein kinase) on T172; "
            "  AMPK activation -> mTOR inhibition -> metabolic checkpoint; "
            "  STK11 LOF -> mTOR hyperactivation -> hamartomatous polyp proliferation; "
            "  STK11 also regulates: cell polarity, tight junction formation, cell migration; "
            "PEUTZ-JEGHERS SYNDROME (PJS): "
            "  OMIM 175200; AD LOF; near-complete penetrance; de novo ~25%; "
            "  GI hamartomatous polyps: throughout GI tract (small bowel predominant); "
            "  Mucocutaneous melanin pigmentation: perioral/buccal/labial spots = PATHOGNOMONIC; "
            "  Pigmentation onset: infancy/childhood; fades in adulthood (INTERNAL FEATURES PERSIST); "
            "  Polyps: small bowel 95%; colon 60%; stomach 50%; intussusception risk; "
            "CERVICAL CANCER IN PJS (STK11): "
            "  Cervical cancer: 10-13% lifetime (3rd most common PJS cancer after breast and CRC); "
            "  SCTAT (sex cord tumour with annular tubules): PATHOGNOMONIC for PJS in females; "
            "  Adenoma malignum (minimal deviation adenocarcinoma, MDC): PATHOGNOMONIC for PJS; "
            "  SCTAT: benign in ~15% (PJS-associated); malignant 25% PJS-associated (vs sporadic worse); "
            "  Adenoma malignum: deceptively well-differentiated; deep invasion; late presentation; "
            "SCTAT -- PATHOGNOMONIC: "
            "  SCTAT = sex cord tumour with annular tubules; PJS-associated = mostly benign; "
            "  Annular tubular architecture on histology = PATHOGNOMONIC; "
            "  Oestrogen-secreting (can cause precocious puberty, menstrual irregularity); "
            "  Annual pelvic MRI + US from age 18yr in STK11 females; "
            "ADENOMA MALIGNUM (MDC) -- PATHOGNOMONIC: "
            "  Minimal deviation adenocarcinoma of cervix = PATHOGNOMONIC PJS feature; "
            "  HIGHLY DECEPTIVE: abundant mucin-secreting glands, minimal atypia; "
            "  Often missed on standard smear/HPV test -> requires MRI + cone biopsy; "
            "  Deep infiltration despite well-differentiated appearance; STK11 LOF driver; "
            "FULL PJS CANCER SPECTRUM: "
            "  Breast: 45% lifetime = DOMINANT PJS cancer in females; "
            "  Pancreatic: 30-40% = HIGHEST RR in PJS (~130x vs general population); "
            "  CRC: 39%; small bowel: 13%; gastric: 29%; cervical: 10-13%; ovarian: 21%; "
            "  Annual surveillance from multiple ages: breast MRI from 25yr; CAPS pancreatic from 30yr; "
            "  Colonoscopy every 2-3yr from age 18yr; small bowel MRI/CE from age 8yr; "
            "TREATMENT (STK11 PATHWAY): "
            "  mTOR inhibitors (everolimus): pre-clinical STK11-null tumours; "
            "  OCP: may reduce endometrial/cervical risk but NOT contraindication-free in PJS; "
            "  Adenoma malignum: radical hysterectomy + bilateral salpingo-oophorectomy; "
            "  SCTAT (benign): surveillance; malignant SCTAT: standard ovarian cancer treatment"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 175200 (PJS); near-complete penetrance; ~25% de novo; family cascade mandatory; SCTAT/adenoma malignum PATHOGNOMONIC in females",
        "cancer_risk": "Cervical: 10-13% lifetime (SCTAT PATHOGNOMONIC + adenoma malignum PATHOGNOMONIC); breast 45% DOMINANT; pancreatic 30-40% (~130x RR HIGHEST PJS RR); CRC 39%; gastric 29%; ovarian 21%; small bowel 13%",
        "pathognomonic": "Perioral/buccal mucocutaneous melanin pigmentation (STK11 PJS) PATHOGNOMONIC; SCTAT (sex cord tumour annular tubules) PATHOGNOMONIC in PJS females; adenoma malignum (minimal deviation adenocarcinoma cervix) PATHOGNOMONIC PJS",
        "surveillance_key": "Annual pelvic MRI + US from age 18yr (SCTAT/adenoma malignum); breast MRI from 25yr; CAPS pancreatic MRI from 30yr; small bowel MRI from age 8yr; colonoscopy every 2-3yr from 18yr; STK11 germline all adenoma malignum cervix",
        "key_distinctions": [
            "SCTAT-PATHOGNOMONIC-PJS-FEMALES",
            "ADENOMA-MALIGNUM-PATHOGNOMONIC-PJS-DECEPTIVE",
            "CERVICAL-10-13PCT-3RD-MOST-COMMON-PJS",
            "PERIORAL-BUCCAL-PIGMENTATION-PATHOGNOMONIC",
            "BREAST-45PCT-DOMINANT-PJS-FEMALE",
            "PANCREATIC-30-40PCT-130X-HIGHEST-PJS-RR",
        ],
    },
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "BRCA1-213kDa-RING-E3-BARD1-Cervical-2-3x-BSO-Age35-40-"
            "Olaparib-PARP-Breast-72pct-Ovarian-44pct-HBOC-OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 213 kDa / 17q21.31 BRCA1 encodes Breast Cancer Gene 1 (RING E3 ubiquitin ligase scaffold): "
            "STRUCTURE: "
            "  1863 aa / 213 kDa; multidomain scaffold; "
            "  RING domain (aa 1-109): heterodimerises with BARD1 RING -> E3 ubiquitin ligase; "
            "  BRCT repeats (aa 1646-1863): 2 tandem BRCT repeats; pSer-binding (phosphoprotein docking); "
            "  Coiled-coil domain (aa 1002-1064): PALB2 interaction -> BRCA2-RAD51 axis; "
            "  NLS (nuclear localisation signals): multiple; "
            "  BRCA1 functions: HR (homologous recombination), cell cycle checkpoint, transcription; "
            "  BRCA1 responds to DSB (double-strand break): ATM phospho-BRCA1 -> HR pathway; "
            "  BRCA1-BARD1 E3: histone H2A ubiquitination -> DSB focus; "
            "  BRCT mutations (BRCT hotspots): Y1853fs, C61G, 185delAG, 5382insC; "
            "HBOC AND CERVICAL RISK: "
            "  BRCA1 germline LOF: cervical adenocarcinoma 2-3x elevated; "
            "  Cervical cancer in BRCA1: predominantly adenocarcinoma (glandular cell) NOT squamous; "
            "  HR deficiency -> genomic instability in cervical glandular epithelium; "
            "  Standard cervical screening (HPV smear) sufficient; no additional cervical surveillance; "
            "BREAST AND OVARIAN (DOMINANT): "
            "  Breast cancer (female): 72% lifetime = DOMINANT BRCA1 risk; "
            "  Ovarian cancer: 44% lifetime; HIGHEST hereditary ovarian risk in BRCA1; "
            "  Male breast cancer: 1.2% (vs 0.1% general population); "
            "  Pancreatic cancer: 2-3x elevated; "
            "BSO -- BILATERAL SALPINGO-OOPHORECTOMY: "
            "  BSO recommended age 35-40yr (after childbearing complete) = STANDARD OF CARE; "
            "  BSO reduces ovarian cancer risk >95% and breast cancer risk ~50% (via oestrogen reduction); "
            "  BSO + HRT (non-oestrogen-dependent cancers): HRT post-BSO does NOT reverse breast benefit; "
            "  BSO timing: BRCA1 35-40yr (earlier than BRCA2 40-45yr due to earlier ovarian onset); "
            "OLAPARIB (PARP INHIBITOR): "
            "  BRCA1 LOF -> HRD -> PARP inhibitor synthetic lethality; "
            "  Olaparib FDA-approved: BRCA1-germline ovarian, breast, pancreatic, prostate; "
            "  BRCA1-mutant cervical adenocarcinoma: HRD -> platinum/olaparib benefit; "
            "SURVEILLANCE (BRCA1): "
            "  Annual breast MRI (alternating with mammography) from age 25yr; "
            "  BSO at age 35-40yr; CA-125 + TVUS until BSO; "
            "  Annual dermatology (BRCA1 subtle skin risk); pancreatic MRI from age 50yr; "
            "  Cascade testing all first-degree relatives (BRCA1 AD)"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 113705/604370; near-complete penetrance for breast/ovarian; ~10-15% de novo; family cascade mandatory; Ashkenazi founder mutations 185delAG/5382insC",
        "cancer_risk": "Breast: 72% lifetime DOMINANT; ovarian: 44% = HIGHEST hereditary ovarian risk; cervical adenocarcinoma: 2-3x elevated; pancreatic 2-3x; male breast 1.2%; contralateral breast 30% at 10yr",
        "pathognomonic": "BSO age 35-40yr = STANDARD OF CARE BRCA1 (ovarian prevention); BRCA1 cervical = adenocarcinoma NOT squamous; HRD (HR-deficient) -> cisplatin/olaparib sensitivity; 185delAG + 5382insC Ashkenazi founder mutations",
        "surveillance_key": "Annual breast MRI from age 25yr; BSO at 35-40yr; olaparib FDA-approved BRCA1-germline; cascade testing all first-degree relatives; cervical: standard HPV-smear programme sufficient; no additional cervical surveillance beyond guideline",
        "key_distinctions": [
            "BSO-AGE-35-40-STANDARD-CARE-BRCA1",
            "BREAST-72PCT-DOMINANT-BRCA1",
            "OVARIAN-44PCT-HIGHEST-HEREDITARY",
            "CERVICAL-2-3X-ADENOCARCINOMA-NOT-SQUAMOUS",
            "OLAPARIB-FDA-BRCA1-GERMLINE-OVARIAN-BREAST",
            "185DELAG-5382INSC-ASHKENAZI-FOUNDERS",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-AVOID-RADIATION-ABSOLUTELY-"
            "WBMRI-Toronto-R337H-Brazilian-Cervical-Surgery-NOT-RT-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes Tumour Protein p53 (Guardian of the Genome): "
            "STRUCTURE: "
            "  393 aa / 43 kDa; tetrameric transcription factor; "
            "  N-terminal transactivation domain (TAD1: aa 1-40; TAD2: aa 40-67); "
            "  Proline-rich domain (aa 67-98); "
            "  DBD (DNA-binding domain): aa 94-292 -- ALL HOTSPOT RESIDUES (R175, G245, R248, R249, R273, R282); "
            "  Tetramerisation domain (aa 325-356); "
            "  C-terminal regulatory domain (aa 356-393): acetylation, ubiquitination; "
            "  p53 activates: p21 (CDKN1A -- G1 arrest), BAX (apoptosis), PUMA, NOXA, MDM2; "
            "  p53 responds to: DNA damage, oncogene activation, hypoxia, oxidative stress; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; AD LOF; de novo ~25%; highly penetrant LFS (~80-100% lifetime cancer); "
            "  Sarcoma (STS + osteosarcoma): 50-60% = PRIMARY LFS cancer type; "
            "  Breast: 30-40% (early-onset <40yr); brain tumour: 10-15%; ACC: 10-15% children; "
            "CERVICAL CANCER IN LFS (TP53): "
            "  Cervical carcinosarcoma (malignant mixed Müllerian tumour): elevated in TP53 LOF carriers; "
            "  Cervical adenosquamous carcinoma: uncommon but elevated with germline TP53; "
            "  p53-null or p53-diffuse overexpression IHC in cervical tumour -> germline TP53 workup; "
            "  PRIMARY LFS cancers (sarcoma/breast/brain/ACC) far exceed cervical as site of first cancer; "
            "AVOID RADIATION -- ABSOLUTE RULE: "
            "  LFS: AVOID THERAPEUTIC RADIATION ABSOLUTELY -- RULE 1; "
            "  Radiation-induced secondary sarcomas in TP53 LOF: extreme latency risk; "
            "  Cervical cancer standard treatment = chemoradiation -> CONTRAINDICATED in LFS; "
            "  If cervical cancer in LFS: SURGERY (radical hysterectomy) = PREFERRED over chemoradiation; "
            "  Replace CT surveillance with MRI (non-ionising) in ALL LFS patients; "
            "TORONTO WBMRI PROTOCOL: "
            "  WBMRI: annually -- detects sarcoma, breast, ACC, brain, CRC in one examination; "
            "  No ionising radiation; brain MRI annual (with gadolinium); "
            "R337H BRAZILIAN FOUNDER: "
            "  TP53 R337H: frequency ~1/300 southern Brazil; predominantly paediatric ACC; "
            "  Lower penetrance than classic LFS; specific surveillance programme; "
            "p53 IHC ABERRANT PATTERNS: "
            "  p53-null (complete loss): LOF mutation PATHOGNOMONIC; "
            "  p53-overexpression (diffuse strong): GOF missense PATHOGNOMONIC"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 191170/151623; ~25% de novo; highly penetrant LFS (~80-100% lifetime cancer); GOF hotspot mutations possible; family cascade mandatory",
        "cancer_risk": "Sarcoma (STS + osteosarcoma): 50-60% PRIMARY LFS; breast 30-40%; brain tumour 10-15%; ACC 10-15% children; cervical carcinosarcoma elevated; chemoradiation CONTRAINDICATED in LFS -- surgery preferred for cervical cancer",
        "pathognomonic": "AVOID RADIATION ABSOLUTELY (LFS Rule 1); cervical cancer in LFS -> SURGERY not chemoradiation; R337H Brazilian founder ~1/300 south Brazil paediatric ACC; p53-null or p53-overexpression IHC = aberrant PATHOGNOMONIC in cervical tumours",
        "surveillance_key": "WBMRI annually Toronto Protocol (no radiation); brain MRI annual; abdominal US 6-monthly children; AVOID radiation absolutely; if cervical cancer: radical hysterectomy preferred (NOT chemoradiation); TP53 germline all early-onset carcinosarcoma/cervical sarcoma",
        "key_distinctions": [
            "AVOID-RADIATION-ABSOLUTELY-LFS-RULE-1",
            "CERVICAL-SURGERY-NOT-CHEMORADIATION-LFS",
            "WBMRI-TORONTO-ANNUALLY-NO-CT",
            "SARCOMA-50-60PCT-PRIMARY-LFS",
            "R337H-BRAZILIAN-FOUNDER-1-IN-300",
            "P53-ABERRANT-IHC-CERVICAL-WORKUP",
        ],
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutS-Homolog2-105kDa-Lynch2-Cervical-5-10pct-Adenocarcinoma-NOT-Squamous-"
            "Endometrial-40-60pct-DOMINANT-Muir-Torre-PATHOGNOMONIC-EPCAM-3prime-OMIM-120435"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 105 kDa / 2p21 MSH2 encodes MutS Homolog 2 (MMR mismatch recognition): "
            "STRUCTURE: "
            "  934 aa / 105 kDa; "
            "  MSH2 forms two heterodimers: MutS-alpha (MSH2+MSH6: mismatches) and MutS-beta (MSH2+MSH3: IDLs); "
            "  MSH2 is the shared/core subunit of BOTH heterodimers; "
            "  MSH2 ATPase domain: conformational changes on mismatch recognition; "
            "  MSH2-MSH6 (MutS-alpha): recognises single base-base mismatches + 1-4 nucleotide IDLs; "
            "  MSH2-MSH3 (MutS-beta): recognises 2-12 nucleotide IDLs; "
            "  MSH2 LOF -> both MutS-alpha AND MutS-beta deficient -> broad MMR deficiency; "
            "LYNCH SYNDROME TYPE 2 (MSH2): "
            "  OMIM 609309; AD LOF; Lynch Type 2 -- broadest extracolonic spectrum; "
            "  CRC: 40-60% lifetime; endometrial: 40-60% DOMINANT in females; "
            "  Urinary tract (ureter/renal pelvis): 25-28% = HIGHEST urinary tract risk in Lynch; "
            "  Ovarian: 10-12%; gastric: 8-12%; sebaceous neoplasms MUIR-TORRE; "
            "CERVICAL CANCER IN LYNCH (MSH2): "
            "  Cervical adenocarcinoma: 5-10% lifetime = Lynch2-associated cervical risk; "
            "  Lynch-associated cervical cancer: ALMOST ALWAYS ADENOCARCINOMA (endocervical glandular); "
            "  NOT squamous cell carcinoma (Lynch does not elevate SCC risk significantly); "
            "  dMMR IHC in cervical adenocarcinoma -> Lynch workup mandatory; "
            "  MSH2-loss IHC: MSH2+MSH6 both lost (MSH6 destabilises without MSH2 partner); "
            "  MSI testing of cervical adenocarcinoma: MSI-H in Lynch cervical; "
            "MUIR-TORRE SYNDROME (MSH2 PATHOGNOMONIC): "
            "  Sebaceous neoplasms (adenoma, carcinoma, epithelioma) + visceral malignancy = PATHOGNOMONIC; "
            "  Sebaceous neoplasms on face/trunk: sebaceous adenoma PATHOGNOMONIC MSH2/MSH6; "
            "  Keratoacanthoma: may occur (sebaceous differentiation); "
            "  Any sebaceous neoplasm -> reflex IHC MSH2/MSH6/MLH1/PMS2 MANDATORY; "
            "EPCAM 3-PRIME DELETION (MSH2 SILENCING): "
            "  3' EPCAM deletion (del EPCAM D13S1830): EPCAM read-through -> MSH2 promoter methylation; "
            "  IHC: MSH2+MSH6 LOST; EPCAM NORMAL on IHC; MLPA 3'-specific = diagnostic; "
            "  Standard NGS/WES misses EPCAM deletion -> MLPA mandatory if IHC suggests MSH2 Lynch; "
            "PEMBROLIZUMAB (MSI-H ALL HISTOLOGIES): "
            "  Pembrolizumab FDA-approved: MSI-H/dMMR solid tumours ALL histologies; "
            "  MSH2-mutant cervical adenocarcinoma (dMMR/MSI-H): pembrolizumab first-line option; "
            "ASPIRIN CAPP2 -- LEVEL A: "
            "  Aspirin 600mg/day: 50% CRC risk reduction in Lynch = LEVEL A evidence; "
            "SURVEILLANCE (MSH2 LYNCH2): "
            "  Colonoscopy 1-2yr from age 25yr; "
            "  Annual gynaecological (endometrial/ovarian) surveillance from age 30yr; "
            "  Annual urine cytology; cystoscopy every 2yr from age 25yr (urinary tract); "
            "  Cervical adenocarcinoma surveillance: additional annual USS + smear"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 120435/609309; Lynch Syndrome Type 2; near-complete penetrance for CRC/endometrial; EPCAM 3' deletion mimics MSH2 LOF; Muir-Torre = PATHOGNOMONIC MSH2/MSH6; family cascade mandatory",
        "cancer_risk": "Endometrial: 40-60% DOMINANT females Lynch2; CRC: 40-60%; urinary tract: 25-28% HIGHEST Lynch urinary; cervical adenocarcinoma: 5-10%; ovarian: 10-12%; gastric 8-12%; sebaceous Muir-Torre PATHOGNOMONIC",
        "pathognomonic": "Muir-Torre syndrome (sebaceous neoplasms + visceral) PATHOGNOMONIC MSH2/MSH6; Lynch cervical = adenocarcinoma NOT squamous; MSH2+MSH6 both lost IHC; EPCAM 3' deletion silences MSH2 (MLPA mandatory); pembrolizumab MSI-H all histologies",
        "surveillance_key": "Annual gynaecological surveillance (endometrial/cervical) from age 30yr; colonoscopy 1-2yr from 25yr; urinary tract every 2yr; sebaceous neoplasm -> reflex IHC MSH2 MANDATORY; MSH2-cervical dMMR -> pembrolizumab; EPCAM MLPA if MSH2-IHC Lynch suspected",
        "key_distinctions": [
            "CERVICAL-ADENOCARCINOMA-NOT-SQUAMOUS-LYNCH",
            "ENDOMETRIAL-40-60PCT-DOMINANT-LYNCH2",
            "MUIR-TORRE-SEBACEOUS-PATHOGNOMONIC",
            "EPCAM-3PRIME-DELETION-MSH2-PROMOTER",
            "URINARY-TRACT-25-28PCT-HIGHEST-LYNCH",
            "PEMBROLIZUMAB-MSI-H-ALL-HISTOLOGIES",
        ],
    },
    {
        "gene": "FANCA",
        "protein": (
            "FANCA -- 16q24.3 Autosomal-Recessive-LOF -- 1455aa -- "
            "FANCA-163kDa-FA-Core-Complex-Scaffold-FA-Type-A-60pct-"
            "Cervical-SCC-150-200x-HPV-Vaccination-CRITICAL-AVOID-ALDEHYDE-ABSOLUTELY-DEB-PATHOGNOMONIC-OMIM-607139"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "1455 aa / 163 kDa / 16q24.3 FANCA encodes Fanconi Anemia Complementation Group A scaffold: "
            "STRUCTURE: "
            "  1455 aa / 163 kDa; largest FA core complex component; "
            "  N-terminal HEAT repeats (aa 1-~200): protein-protein interactions; "
            "  Nuclear localisation signals (NLS): multiple; "
            "  FANCG-binding domain: essential for FA core complex nuclear import; "
            "  No catalytic domain (scaffold function); "
            "  FA Core Complex: FANCA-FANCB-FANCC-FANCE-FANCF-FANCG-FANCL-FANCM-8 proteins; "
            "  FANCA required for nuclear import of FA core complex (escorts FANCG); "
            "  FA core complex activates FANCD2 monoubiquitination (FANCD2-Ub = DNA damage activated); "
            "  FANCD2-Ub: recruits BRCA2 (FANCD1) + other HR proteins to ICL sites; "
            "  FANCA LOF -> FA core complex nuclear import FAILED -> FANCD2 NOT monoubiquitinated; "
            "FA TYPE A -- MOST COMMON FA: "
            "  FANCA: ~60-70% of all FA cases = MOST COMMON FA complementation group; "
            "  Classic FA phenotype: bone marrow failure (BMF) + physical anomalies + cancer; "
            "  BMF onset: age 5-10yr; progressive pancytopenia; "
            "  Radial ray anomalies: absent/hypoplastic thumbs (50-75%); "
            "  Radius: absent/hypoplastic (50%); VACTERL-H association; "
            "CERVICAL SCC IN FA (FANCA): "
            "  Cervical SCC: ~150-200x elevated risk vs general population; "
            "  FA cervical SCC: onset median age ~25-30yr (vs ~50yr general); "
            "  FA cancer risk order: AML > ESCC > Head-neck SCC > Cervical SCC > Other solid tumours; "
            "  Cervical SCC in FA = HPV-driven SCC in HPV-hypersensitive FA cells; "
            "  FA pathway required for aldehyde-induced interstrand crosslink repair; "
            "  FA cells: hypersensitive to HPV-induced mutagenesis + aldehyde crosslinks; "
            "HPV VACCINATION -- CRITICAL AND MANDATORY: "
            "  FA patients: HPV vaccination = CRITICAL and MANDATORY (both sexes); "
            "  HPV9 vaccine (Gardasil9): HPV 6/11/16/18/31/33/45/52/58 -> reduces HPV-driven SCC risk; "
            "  FA cells: HPV-related DNA damage amplified (ICL repair deficient); "
            "  Vaccination: ideal BEFORE sexual debut; post-HSCT revaccination required; "
            "  Annual cervical examination + HPV testing from age 16yr; "
            "AVOID ALDEHYDE/ALCOHOL ABSOLUTELY: "
            "  FANCA: AVOID alcohol ABSOLUTELY (ethanol -> acetaldehyde via ADH); "
            "  Formaldehyde: occupational exposure AVOID ABSOLUTELY; "
            "  Acetaldehyde/aldehydes: ICL induction in FA cells = hypersensitive; "
            "  ALSO AVOID: mitomycin C, cross-linking chemotherapy agents; "
            "DEB TEST -- PATHOGNOMONIC: "
            "  Diepoxybutane (DEB) challenge: chromosomal fragility = PATHOGNOMONIC FA; "
            "  DEB -> ICLs -> FA pathway required -> FA cells show increased chromosomal breaks; "
            "HSCT AND SOLID TUMOUR RISK: "
            "  RIC-HSCT (reduced-intensity conditioning): curative for BMF; "
            "  Post-HSCT: solid tumour risk (cervical SCC, ESCC, head-neck SCC) PERSISTS; "
            "  Conditioning regimen: NO cyclophosphamide full-dose (RIC MANDATORY for FA); "
            "  Radiotherapy: ABSOLUTE CI in FA (FANCA LOF -> radiation hypersensitivity)"
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF; OMIM 607139; most common FA (~60-70%); BMF + physical anomalies + cancer predisposition; HPV vaccination MANDATORY; AVOID radiation ABSOLUTELY; DEB test diagnostic PATHOGNOMONIC",
        "cancer_risk": "Cervical SCC: ~150-200x elevated; AML/MDS 25-30x; ESCC 400x RR HIGHEST solid tumour; head/neck SCC highly elevated; solid tumour risk PERSISTS after HSCT; cancer onset very early (age 20-35yr)",
        "pathognomonic": "DEB (diepoxybutane) chromosomal fragility test PATHOGNOMONIC for FA diagnosis; HPV vaccination CRITICAL and MANDATORY in FA; AVOID ALDEHYDE ABSOLUTELY (alcohol/formaldehyde); AVOID RADIATION ABSOLUTELY (RT for cervical cancer CONTRAINDICATED in FA)",
        "surveillance_key": "HPV vaccination CRITICAL and MANDATORY (Gardasil9); annual cervical smear + HPV testing from age 16yr; AVOID alcohol/formaldehyde/radiation ABSOLUTELY; RIC-HSCT for BMF (NOT full myeloablative); post-HSCT cervical surveillance continues; DEB test for diagnosis",
        "key_distinctions": [
            "CERVICAL-SCC-150-200X-HPV-HYPERSENSITIVE",
            "HPV-VACCINATION-CRITICAL-MANDATORY-FA",
            "AVOID-ALDEHYDE-ABSOLUTELY-ICL-PATHWAY",
            "DEB-TEST-PATHOGNOMONIC-FA-DIAGNOSIS",
            "AVOID-RADIATION-ABSOLUTELY-CERVICAL-RT-CI",
            "SOLID-TUMOUR-RISK-PERSISTS-POST-HSCT",
        ],
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "PTEN-47kDa-Phosphatase-PI3K-Antagonist-Cowden-PHTS-"
            "Macrocephaly-PATHOGNOMONIC-Lhermitte-Duclos-PATHOGNOMONIC-"
            "Trichilemmoma-PATHOGNOMONIC-Endometrial-28-44pct-Cervical-Elevated-Everolimus-OMIM-601728"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 47 kDa / 10q23.31 PTEN encodes Phosphatase and Tensin Homolog: "
            "STRUCTURE: "
            "  403 aa / 47 kDa; dual-specificity phosphatase; "
            "  N-terminal CBR3 loop (aa 1-13): PTEN binding to plasma membrane; "
            "  Phosphatase domain (aa 14-185): contains active site C124 (essential for catalysis); "
            "  C2 domain (aa 186-351): calcium-independent membrane-binding; "
            "  PDZ-binding motif (aa 400-403): MAST kinase interaction; "
            "  PTEN primary lipid substrate: PIP3 (phosphatidylinositol-3,4,5-trisphosphate); "
            "  PTEN converts PIP3 -> PIP2 -> antagonises PI3K; "
            "  PTEN LOF -> PIP3 accumulation -> AKT hyperactivation -> mTOR hyperactivation; "
            "  mTOR: downstream: S6K1, 4EBP1 -> uncontrolled translation/proliferation; "
            "COWDEN SYNDROME / PHTS: "
            "  OMIM 158350 (Cowden); AD LOF; PHTS = PTEN Hamartoma Tumour Syndrome (umbrella); "
            "  Cowden = classic adult PHTS; Bannayan-Riley-Ruvalcaba (BRR) = childhood PHTS; "
            "MUCOCUTANEOUS FEATURES (PATHOGNOMONIC): "
            "  Macrocephaly (HC >=97th centile or OFC >=58cm female/60cm male): PATHOGNOMONIC; "
            "  Macrocephaly: most sensitive indicator of PHTS; "
            "  Lhermitte-Duclos disease (dysplastic gangliocytoma of cerebellum): PATHOGNOMONIC; "
            "  Trichilemmoma (benign hair follicle tumour, facial): PATHOGNOMONIC; "
            "  Papillomatous papules (mucocutaneous); cobblestone gingival papillomatosis; "
            "  Acral keratoses; penile freckling (in males); "
            "CERVICAL AND ENDOMETRIAL IN PHTS: "
            "  Endometrial cancer: 28-44% lifetime = DOMINANT gynaecological PTEN risk; "
            "  Cervical adenocarcinoma: elevated (endocervical glandular -- PTEN expression high cervix); "
            "  PTEN null IHC: loss of PTEN staining in endometrial/cervical adenocarcinoma = LOF; "
            "  Annual pelvic USS from age 30yr; BSO after childbearing for endometrial risk reduction; "
            "FULL PHTS CANCER SPECTRUM: "
            "  Breast: 85% lifetime (highest PHTS risk); thyroid 35% (predominantly follicular); "
            "  Endometrial: 28-44%; colorectal: 9%; renal (RCC): 34% (all-cause); "
            "EVEROLIMUS (mTOR INHIBITOR): "
            "  PTEN LOF -> mTOR hyperactivation -> everolimus mechanistic rationale; "
            "  Everolimus FDA-approved: TSC/SEGA (tuberous sclerosis); some AML/LAM in TSC; "
            "  PTEN-null endometrial/cervical: everolimus-based protocols (clinical trials); "
            "  Lenvatinib + everolimus: FDA-approved advanced endometrial cancer (second-line); "
            "PTEN NULL IHC -- PATHOGNOMONIC: "
            "  PTEN-null IHC (complete loss of staining in tumour, retained in stroma): PATHOGNOMONIC LOF; "
            "  Any cervical/endometrial glandular tumour with PTEN-null IHC -> germline PTEN workup; "
            "SURVEILLANCE (PHTS): "
            "  Annual breast MRI from age 30yr; annual thyroid US from 18yr; "
            "  Annual pelvic USS (endometrial/cervical) from age 30yr; "
            "  Colonoscopy every 5yr from age 35yr; "
            "  Brain MRI if neurological symptoms (Lhermitte-Duclos)"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 601728/158350; near-complete penetrance for mucocutaneous features; PHTS (Cowden/BRR/Proteus-like); macrocephaly most sensitive indicator; family cascade mandatory; de novo ~10-20%",
        "cancer_risk": "Breast: 85% lifetime DOMINANT; endometrial: 28-44% = dominant gynaecological PTEN; thyroid 35% (follicular); renal 34%; cervical adenocarcinoma elevated; colorectal 9%; PTEN-null IHC any gynaecological tumour = germline workup",
        "pathognomonic": "Macrocephaly (OFC >=97th centile) PATHOGNOMONIC PHTS; Lhermitte-Duclos disease (dysplastic gangliocytoma cerebellum) PATHOGNOMONIC PHTS; trichilemmoma (facial) PATHOGNOMONIC; PTEN-null IHC cervical/endometrial = LOF PATHOGNOMONIC; cobblestone gingival papillomatosis",
        "surveillance_key": "Annual breast MRI from age 30yr; annual pelvic USS from 30yr (endometrial/cervical); annual thyroid US from 18yr; BSO after childbearing for endometrial reduction; everolimus mTORi for PTEN-null tumours; any gynaecological adenocarcinoma + macrocephaly -> germline PTEN",
        "key_distinctions": [
            "MACROCEPHALY-OFC-97TH-CENTILE-PATHOGNOMONIC",
            "LHERMITTE-DUCLOS-PATHOGNOMONIC",
            "TRICHILEMMOMA-FACIAL-PATHOGNOMONIC",
            "ENDOMETRIAL-28-44PCT-DOMINANT-PTEN",
            "PTEN-NULL-IHC-PATHOGNOMONIC",
            "EVEROLIMUS-MTOR-RATIONALE",
        ],
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant/Recessive-LOF -- 3056aa -- "
            "ATM-350kDa-PI3K-Like-Kinase-A-T-Biallelic-"
            "RADIOSENSITIVITY-ABSOLUTE-Cervical-Chemoradiation-CI-Biallelic-"
            "Monoallelic-Cervical-2-3x-Ceralasertib-ATRi-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM encodes Ataxia-Telangiectasia Mutated (PI3K-like kinase): "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; PI3K-like serine/threonine kinase; "
            "  N-terminal HEAT repeats (aa 1-~1400): regulatory scaffold; "
            "  FAT domain (aa ~1900-2500): FRAP-ATM-TRRAP domain; "
            "  PI3K kinase domain (aa ~2713-2962): catalytic core; "
            "  FATC domain (aa 2963-3056): C-terminal regulatory; "
            "  ATM homodimer (inactive) -> DSB signal -> autophosphorylation S1981 -> monomer (ACTIVE); "
            "  ATM primary substrate: H2AX (gammaH2AX) -- DSB marker; "
            "  ATM substrates: BRCA1 (S1423), CHK2, MDM2, p53, FANCD2; "
            "  ATM LOF -> DSB repair failure -> chromosomal instability -> cancer; "
            "ATAXIA-TELANGIECTASIA (BIALLELIC A-T): "
            "  OMIM 208900; AR biallelic; complete/near-complete ATM loss; "
            "  Cerebellar ataxia: progressive from age 1-2yr; wheelchair-bound age 10yr; "
            "  Ocular telangiectasia (bulbar conjunctiva): age 2-8yr; "
            "  Immunodeficiency: IgA/IgG/IgE; sinopulmonary infections; "
            "  A-T leukaemia/lymphoma: 80-100x elevated (lymphoid malignancy PRIMARY in A-T); "
            "RADIOSENSITIVITY ABSOLUTE -- BIALLELIC A-T: "
            "  A-T: AVOID ALL THERAPEUTIC RADIATION -- LETHAL; "
            "  ATM LOF -> failed DSB repair after radiation -> chromosomal catastrophe; "
            "  Cervical cancer standard treatment = chemoradiation = ABSOLUTE CONTRAINDICATION in A-T; "
            "  Cervical cancer in biallelic A-T: SURGERY MANDATORY (radical hysterectomy); "
            "  RT sensitivity: >2x more sensitive than normal; even diagnostic X-rays minimise; "
            "MONOALLELIC ATM -- CERVICAL AND OTHER: "
            "  Monoallelic ATM: cervical 2-3x elevated; breast 15-25% lifetime; "
            "  Pancreatic 5-10x; prostate 2-4x; "
            "  Monoallelic ATM cervical: modest but real elevated risk; standard screening sufficient; "
            "CERALASERTIB (ATRi) + OLAPARIB: "
            "  ATM LOF -> HRD-like -> PARP inhibitor synthetic lethality (olaparib); "
            "  Ceralasertib (ATR inhibitor) + olaparib: clinical trials ATM-deficient solid tumours; "
            "  ATM LOF -> increased ATR reliance -> ATRi synthetic lethal; "
            "SURVEILLANCE (ATM): "
            "  Annual breast MRI from age 25yr (monoallelic); "
            "  Annual pelvic exam; standard cervical screening programme; "
            "  Avoid radiation for all imaging: prefer MRI/US"
        ),
        "inheritance": "Autosomal Dominant (monoallelic LOF; AD); Autosomal Recessive (biallelic A-T; AR); OMIM 607585/208900; biallelic = full A-T; monoallelic = intermediate cancer risk; radiosensitivity ABSOLUTE biallelic; family cascade mandatory",
        "cancer_risk": "Biallelic A-T: lymphoid malignancy 80-100x PRIMARY; cervical SCC/adenocarcinoma elevated; monoallelic ATM: breast 15-25%; cervical 2-3x; pancreatic 5-10x; prostate 2-4x; chemoradiation for cervical ABSOLUTE CI in biallelic A-T",
        "pathognomonic": "RADIOSENSITIVITY ABSOLUTE in biallelic A-T; cervical chemoradiation = ABSOLUTE CONTRAINDICATION biallelic A-T (surgery mandatory); cerebellar ataxia + telangiectasia + immunodeficiency = A-T TRIAD PATHOGNOMONIC; ATM-mutant cervical -> ceralasertib/olaparib trials",
        "surveillance_key": "AVOID ALL RADIATION ABSOLUTELY biallelic A-T; cervical cancer = radical hysterectomy NOT chemoradiation in A-T; ceralasertib ATRi + olaparib for ATM-deficient cervical; annual breast MRI from 25yr monoallelic; standard cervical screening monoallelic",
        "key_distinctions": [
            "RADIOSENSITIVITY-ABSOLUTE-BIALLELIC-AT",
            "CERVICAL-CHEMORADIATION-ABSOLUTE-CI-AT",
            "SURGERY-MANDATORY-CERVICAL-BIALLELIC-AT",
            "CERVICAL-2-3X-MONOALLELIC-ATM",
            "CERALASERTIB-ATRi-OLAPARIB-ATM-DEFICIENT",
            "ATAXIA-TELANGIECTASIA-TRIAD-PATHOGNOMONIC",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-RAD51-Loader-HBOC-Cervical-2-3x-"
            "Olaparib-PARP-Cisplatin-Sensitive-FA-D1-Biallelic-MOST-SEVERE-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes Breast Cancer Gene 2 (HR scaffold / RAD51 loader): "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; largest common cancer predisposition protein; "
            "  N-terminal transactivation domain (NTD, aa 1-40): PALB2 interaction; "
            "  8 BRC repeats (aa 1002-2085): RAD51 monomer binding; "
            "  OB folds (aa 2402-3190): ssDNA binding and RAD51 filament loading; "
            "  TR2/C-terminal domain (aa 3190-3418): DSS1, DNA binding; "
            "  BRCA2 function: scaffold for HR -- loads RAD51 onto ssDNA at DSB resection ends; "
            "  BRCA2-RAD51 filament: strand invasion into homologous template = HR template switch; "
            "  BRCA2 requires PALB2 (N-terminal bridge) for nuclear localisation with BRCA1; "
            "HBOC AND CERVICAL RISK: "
            "  BRCA2 germline LOF: cervical adenocarcinoma 2-3x elevated; "
            "  Cervical cancer in BRCA2: predominantly adenocarcinoma (glandular cell) NOT squamous; "
            "  HRD -> genomic instability; cervical cancer in BRCA2 carriers: platinum/olaparib benefit; "
            "  Standard cervical screening (HPV smear) sufficient; no additional cervical imaging; "
            "BREAST AND OVARIAN (DOMINANT): "
            "  Breast cancer (female): 69% lifetime; "
            "  Ovarian cancer: 17% lifetime (lower than BRCA1 44% but still highly elevated); "
            "  Male breast cancer: 6-8% = HIGHEST hereditary male breast cancer risk; "
            "  Pancreatic: 3-5x; prostate: 8x (HIGHEST hereditary prostate risk); "
            "BSO -- BILATERAL SALPINGO-OOPHORECTOMY: "
            "  BSO recommended age 40-45yr (later than BRCA1 due to later ovarian onset); "
            "OLAPARIB (PARP INHIBITOR): "
            "  BRCA2 LOF -> HRD -> PARP inhibitor synthetic lethality; "
            "  Olaparib FDA-approved: BRCA2-germline ovarian, breast, pancreatic, prostate; "
            "  Cisplatin/carboplatin: BRCA2 HR-deficient tumours sensitive to platinum crosslinks; "
            "  BRCA2-mutant cervical adenocarcinoma: cisplatin/olaparib benefit expected; "
            "FA-D1 BIALLELIC -- MOST SEVERE FA: "
            "  BRCA2 = FANCD1; biallelic = Fanconi Anemia type D1 = MOST SEVERE FA; "
            "  FA-D1: medulloblastoma + Wilms tumour + AML in infancy/early childhood; "
            "  FA-D1 children: AVOID all alkylating/cross-linking agents (mitomycin, cyclophosphamide); "
            "  FA-D1 = RAREST FA (biallelic BRCA2 lethal in development -- severe FA-D1 births rare); "
            "SURVEILLANCE (BRCA2): "
            "  Annual breast MRI (alternating with mammography) from age 25yr; "
            "  BSO at age 40-45yr; PSA from 40yr; "
            "  Pancreatic MRI/MRCP from age 50yr (or 10yr before earliest familial case); "
            "  Cascade testing all first-degree relatives (BRCA2 AD)"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 600185/612555; near-complete penetrance for breast; ~5-10% de novo; biallelic = FA-D1 MOST SEVERE; family cascade mandatory; Ashkenazi founder mutation 6174delT",
        "cancer_risk": "Breast: 69% lifetime; ovarian: 17%; male breast 6-8% HIGHEST hereditary male breast; prostate: 8x HIGHEST hereditary prostate; cervical adenocarcinoma: 2-3x elevated; pancreatic 3-5x; FA-D1 biallelic medulloblastoma/Wilms/AML infancy",
        "pathognomonic": "BSO age 40-45yr = STANDARD OF CARE BRCA2 (ovarian prevention); FA-D1 biallelic = MOST SEVERE FA (medulloblastoma + Wilms + AML infancy); BRCA2 cervical = adenocarcinoma NOT squamous; HRD -> cisplatin/olaparib sensitivity; 6174delT Ashkenazi founder",
        "surveillance_key": "Annual breast MRI from age 25yr; BSO at 40-45yr; PSA from 40yr; olaparib FDA-approved BRCA2-germline; cascade testing all first-degree relatives; cervical: standard HPV-smear programme sufficient; BRCA2 cervical dMMR -> consider HRD therapy",
        "key_distinctions": [
            "BSO-AGE-40-45-STANDARD-CARE-BRCA2",
            "BREAST-69PCT-DOMINANT-BRCA2",
            "FA-D1-BIALLELIC-MOST-SEVERE-FA",
            "CERVICAL-2-3X-ADENOCARCINOMA-NOT-SQUAMOUS",
            "PROSTATE-8X-HIGHEST-HEREDITARY-PROSTATE",
            "6174DELT-ASHKENAZI-FOUNDER",
        ],
    },
]


def _make_patients(gene_entry: dict) -> list:
    gene = gene_entry["gene"]
    seed = SEED_BASE + ATLAS_GENES.index(gene_entry)
    rng  = random.Random(seed)

    age_params = {
        "STK11":  (32, 12),
        "BRCA1":  (42, 14),
        "TP53":   (35, 13),
        "MSH2":   (47, 14),
        "FANCA":  (27, 8),
        "PTEN":   (44, 13),
        "ATM":    (48, 14),
        "BRCA2":  (43, 14),
    }
    mu, sigma = age_params.get(gene, (40, 13))

    severe_rates = {
        "STK11":  0.62,
        "BRCA1":  0.54,
        "TP53":   0.58,
        "MSH2":   0.55,
        "FANCA":  0.72,
        "PTEN":   0.50,
        "ATM":    0.48,
        "BRCA2":  0.52,
    }
    sev_rate = severe_rates.get(gene, 0.5)

    patients = []
    for i in range(40):
        age       = max(5, round(rng.gauss(mu, sigma), 1))
        sev_event = rng.random() < sev_rate
        patients.append({
            "id":        f"{gene}-{i+1:02d}",
            "age_onset": age,
            "severe":    sev_event,
            "seed":      seed,
        })
    return patients


def generate_overview():
    rows = []
    for g in ATLAS_GENES:
        pts   = _make_patients(g)
        sev_n = sum(1 for p in pts if p["severe"])
        rows.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
            "inheritance":      g["inheritance"],
            "cancer_risk":      g["cancer_risk"],
            "protein":          g["protein"],
        })

    total_pts  = sum(r["n"]        for r in rows)
    total_sev  = sum(r["severe_n"] for r in rows)
    highest    = max(rows, key=lambda r: r["severe_pct"])

    return {
        "atlas":              "Hereditary-Cervical-Cancer-Predisposition-Atlas",
        "seed_range":         f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes_n":            len(ATLAS_GENES),
        "total_patients":     total_pts,
        "severe_total_n":     total_sev,
        "severe_total_pct":   round(total_sev / total_pts * 100, 1),
        "highest_risk_gene":  highest["gene"],
        "highest_risk_pct":   highest["severe_pct"],
        "gene_summary":       rows,
        "genes_detail": [
            {
                "gene":             g["gene"],
                "inheritance":      g["inheritance"],
                "cancer_risk":      g["cancer_risk"],
                "pathognomonic":    g["pathognomonic"],
                "surveillance_key": g["surveillance_key"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_breakdown():
    breakdown = []
    for g in ATLAS_GENES:
        pts      = _make_patients(g)
        seed_idx = ATLAS_GENES.index(g)
        sev_n    = sum(1 for p in pts if p["severe"])
        breakdown.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "seed":             SEED_BASE + seed_idx,
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
        })
    return {"atlas": "Hereditary-Cervical-Cancer-Predisposition-Atlas", "breakdown": breakdown}


def generate_definitions():
    defs = [
        {
            "term": "STK11 / PJS / SCTAT-PATHOGNOMONIC / ADENOMA-MALIGNUM-PATHOGNOMONIC / PERIORAL-PIGMENTATION-PATHOGNOMONIC",
            "definition": (
                "STK11 -- 433aa / 48 kDa / 19p13.3 / AD LOF\n"
                "Peutz-Jeghers Syndrome; SCTAT PATHOGNOMONIC; adenoma malignum PATHOGNOMONIC; perioral pigmentation PATHOGNOMONIC.\n\n"
                "PJS CERVICAL CANCER SPECTRUM:\n"
                "  Cervical: 10-13% lifetime = 3rd most common PJS cancer.\n"
                "  SCTAT (sex cord tumour with annular tubules): PATHOGNOMONIC for PJS in females.\n"
                "  Adenoma malignum (minimal deviation adenocarcinoma): PATHOGNOMONIC for PJS.\n\n"
                "SCTAT -- PATHOGNOMONIC:\n"
                "  PJS-associated SCTAT: mostly benign (~85%); malignant 25% PJS-associated.\n"
                "  Oestrogen-secreting -> precocious puberty / menstrual irregularity.\n"
                "  Annual pelvic MRI + US from age 18yr in STK11 females.\n\n"
                "ADENOMA MALIGNUM (MDC) -- PATHOGNOMONIC:\n"
                "  Minimal deviation adenocarcinoma = highly deceptive: minimal atypia, deep invasion.\n"
                "  Often missed on standard smear -> requires MRI + cone biopsy.\n"
                "  STK11 germline in all adenoma malignum/MDC cervix diagnoses = MANDATORY test.\n\n"
                "PERIORAL PIGMENTATION -- PATHOGNOMONIC:\n"
                "  Melanin spots: perioral, buccal, labial = PATHOGNOMONIC PJS.\n"
                "  Onset infancy/childhood; fades adulthood (internal features persist).\n\n"
                "FULL PJS CANCER SPECTRUM:\n"
                "  Breast: 45% DOMINANT; pancreatic: 30-40% (~130x RR); CRC: 39%.\n"
                "  Small bowel: 13%; gastric: 29%; ovarian: 21%; cervical: 10-13%.\n\n"
                "SURVEILLANCE:\n"
                "  Annual pelvic MRI + US from age 18yr (SCTAT/adenoma malignum detection).\n"
                "  Breast MRI from 25yr; CAPS pancreatic MRI from 30yr; small bowel MRI from age 8yr.\n"
                "  Colonoscopy every 2-3yr from 18yr."
            ),
        },
        {
            "term": "BRCA1 / HBOC / CERVICAL-2-3X-ADENOCARCINOMA / BSO-AGE-35-40 / OLAPARIB-PARP",
            "definition": (
                "BRCA1 -- 1863aa / 213 kDa / 17q21.31 / AD LOF\n"
                "HBOC; cervical adenocarcinoma 2-3x; BSO age 35-40yr; olaparib PARP inhibitor FDA-approved.\n\n"
                "HBOC CERVICAL RISK:\n"
                "  Cervical: 2-3x elevated; predominantly ADENOCARCINOMA (not squamous).\n"
                "  HR deficiency -> genomic instability in cervical glandular epithelium.\n"
                "  Standard cervical screening (HPV smear) sufficient.\n\n"
                "BREAST AND OVARIAN (DOMINANT):\n"
                "  Breast: 72% lifetime = DOMINANT BRCA1 risk.\n"
                "  Ovarian: 44% lifetime = HIGHEST hereditary ovarian risk.\n\n"
                "BSO -- BILATERAL SALPINGO-OOPHORECTOMY:\n"
                "  BSO recommended age 35-40yr = STANDARD OF CARE BRCA1.\n"
                "  Reduces ovarian risk >95% and breast risk ~50%.\n"
                "  Earlier than BRCA2 (40-45yr) due to earlier ovarian onset.\n\n"
                "OLAPARIB:\n"
                "  BRCA1 LOF -> HRD -> PARP inhibitor synthetic lethality.\n"
                "  Olaparib FDA-approved: BRCA1-germline ovarian, breast, pancreatic, prostate.\n"
                "  BRCA1 cervical adenocarcinoma: HRD -> platinum/olaparib benefit.\n\n"
                "ASHKENAZI FOUNDERS:\n"
                "  185delAG (c.68_69delAG): most common BRCA1 founder.\n"
                "  5382insC (c.5266dupC): 2nd Ashkenazi founder.\n\n"
                "SURVEILLANCE:\n"
                "  Annual breast MRI from age 25yr; BSO at 35-40yr.\n"
                "  Cascade testing all first-degree relatives."
            ),
        },
        {
            "term": "TP53 / LFS / AVOID-RADIATION-ABSOLUTELY / CERVICAL-SURGERY-NOT-CHEMORADIATION / WBMRI-TORONTO",
            "definition": (
                "TP53 -- 393aa / 43 kDa / 17p13.1 / AD LOF\n"
                "Li-Fraumeni Syndrome; AVOID RADIATION ABSOLUTELY; cervical cancer = SURGERY NOT chemoradiation; WBMRI Toronto.\n\n"
                "LFS CERVICAL CANCER:\n"
                "  Cervical carcinosarcoma/sarcoma: elevated in TP53 LOF carriers.\n"
                "  PRIMARY LFS cancers: sarcoma 50-60%, breast 30-40%, brain 10-15%, ACC 10-15%.\n"
                "  Cervical cancer: when detected in LFS -> SURGERY is preferred treatment.\n\n"
                "AVOID RADIATION -- ABSOLUTE RULE 1:\n"
                "  LFS: NO THERAPEUTIC RADIATION -- radiation-induced secondary sarcoma = lethal.\n"
                "  Cervical cancer STANDARD TREATMENT = chemoradiation -> CONTRAINDICATED in LFS.\n"
                "  Radical hysterectomy + lymphadenectomy = preferred approach in LFS cervical.\n"
                "  Replace CT surveillance with MRI in ALL LFS patients.\n\n"
                "TORONTO WBMRI PROTOCOL:\n"
                "  WBMRI: annually -- detects sarcoma, breast, ACC, brain, CRC.\n"
                "  No ionising radiation; brain MRI annual.\n\n"
                "R337H BRAZILIAN FOUNDER:\n"
                "  TP53 R337H: ~1/300 southern Brazil; predominantly paediatric ACC.\n\n"
                "p53 IHC (CERVICAL):\n"
                "  p53-null: LOF mutation PATHOGNOMONIC.\n"
                "  p53-overexpression: GOF missense PATHOGNOMONIC.\n"
                "  Any cervical carcinosarcoma + p53 aberrant IHC -> germline TP53 workup mandatory."
            ),
        },
        {
            "term": "MSH2 / LYNCH2 / CERVICAL-ADENOCARCINOMA-NOT-SQUAMOUS / MUIR-TORRE-PATHOGNOMONIC / EPCAM-3PRIME",
            "definition": (
                "MSH2 -- 934aa / 105 kDa / 2p21 / AD LOF\n"
                "Lynch Syndrome Type 2; cervical adenocarcinoma 5-10% (NOT squamous); Muir-Torre PATHOGNOMONIC; EPCAM 3' deletion.\n\n"
                "LYNCH CERVICAL CANCER:\n"
                "  Cervical adenocarcinoma: 5-10% lifetime Lynch2.\n"
                "  ALMOST ALWAYS ADENOCARCINOMA (endocervical glandular) -- NOT squamous.\n"
                "  dMMR IHC in cervical adenocarcinoma -> Lynch workup MANDATORY.\n"
                "  MSH2 IHC loss: MSH2+MSH6 both lost (MSH6 destabilises).\n\n"
                "DOMINANT LYNCH2 RISKS:\n"
                "  Endometrial: 40-60% DOMINANT in females.\n"
                "  CRC: 40-60%; urinary tract: 25-28% HIGHEST Lynch.\n\n"
                "MUIR-TORRE -- PATHOGNOMONIC:\n"
                "  Sebaceous neoplasms (adenoma/carcinoma) + visceral malignancy = PATHOGNOMONIC MSH2/MSH6.\n"
                "  Any sebaceous neoplasm -> reflex IHC MMR MANDATORY.\n\n"
                "EPCAM 3' DELETION:\n"
                "  3' EPCAM deletion -> read-through transcript -> MSH2 promoter methylation.\n"
                "  IHC: MSH2+MSH6 lost; EPCAM normal on IHC.\n"
                "  MLPA 3'-specific diagnostic; standard NGS misses EPCAM deletion.\n\n"
                "PEMBROLIZUMAB (MSI-H):\n"
                "  FDA-approved: MSI-H/dMMR solid tumours all histologies.\n"
                "  MSH2-mutant cervical dMMR: pembrolizumab first-line option.\n\n"
                "ASPIRIN CAPP2 -- LEVEL A:\n"
                "  Aspirin 600mg/day: 50% CRC risk reduction = LEVEL A evidence Lynch.\n\n"
                "SURVEILLANCE:\n"
                "  Annual gynaecological (endometrial + cervical) from age 30yr.\n"
                "  Urinary tract cystoscopy every 2yr from 25yr.\n"
                "  Colonoscopy 1-2yr from 25yr."
            ),
        },
        {
            "term": "FANCA / FA-TYPE-A / CERVICAL-SCC-150-200X / HPV-VACCINATION-CRITICAL-MANDATORY / AVOID-ALDEHYDE-ABSOLUTELY",
            "definition": (
                "FANCA -- 1455aa / 163 kDa / 16q24.3 / AR LOF\n"
                "FA type A most common 60%; cervical SCC ~150-200x elevated; HPV vaccination CRITICAL and MANDATORY; AVOID ALDEHYDE ABSOLUTELY.\n\n"
                "FA CERVICAL SCC RISK:\n"
                "  Cervical SCC: ~150-200x elevated in Fanconi Anemia.\n"
                "  FA cervical onset: median age ~25-30yr (vs ~50yr general population).\n"
                "  FA cells: HPV-induced mutagenesis amplified (ICL repair deficient).\n"
                "  Cancer risk order: AML > ESCC > head-neck SCC > cervical SCC.\n\n"
                "HPV VACCINATION -- CRITICAL AND MANDATORY:\n"
                "  Gardasil9 (HPV9): MANDATORY for all FA patients (both sexes).\n"
                "  Vaccinate before sexual debut; revaccinate post-HSCT.\n"
                "  FA cells: HPV DNA damage amplified; vaccination = primary prevention.\n\n"
                "AVOID ALDEHYDE ABSOLUTELY:\n"
                "  Alcohol: AVOID ABSOLUTELY (ethanol -> acetaldehyde via ADH -> ICL induction).\n"
                "  Formaldehyde: occupational exposure AVOID ABSOLUTELY.\n"
                "  FA cells: hypersensitive to aldehyde-induced DNA interstrand crosslinks.\n\n"
                "AVOID RADIATION ABSOLUTELY (CERVICAL CONTEXT):\n"
                "  Standard cervical cancer treatment = chemoradiation -> ABSOLUTE CI in FA.\n"
                "  FA: radiation hypersensitivity (ATM pathway also impaired in FA).\n"
                "  Cervical cancer in FA: SURGERY mandatory (radical hysterectomy).\n\n"
                "DEB TEST -- PATHOGNOMONIC:\n"
                "  Diepoxybutane chromosomal fragility = PATHOGNOMONIC FA diagnosis.\n\n"
                "POST-HSCT SOLID TUMOUR RISK:\n"
                "  HSCT cures BMF; solid tumour risk (cervical SCC) PERSISTS post-HSCT.\n"
                "  Annual cervical examination from age 16yr continues post-HSCT.\n\n"
                "SURVEILLANCE:\n"
                "  Annual cervical smear + HPV testing from age 16yr.\n"
                "  HPV vaccination MANDATORY before debut.\n"
                "  AVOID alcohol, formaldehyde, radiation, cross-linking chemotherapy."
            ),
        },
        {
            "term": "PTEN / COWDEN-PHTS / MACROCEPHALY-PATHOGNOMONIC / LHERMITTE-DUCLOS-PATHOGNOMONIC / ENDOMETRIAL-28-44PCT",
            "definition": (
                "PTEN -- 403aa / 47 kDa / 10q23.31 / AD LOF\n"
                "Cowden/PHTS; macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC; trichilemmoma PATHOGNOMONIC; endometrial 28-44% DOMINANT.\n\n"
                "PATHOGNOMONIC FEATURES:\n"
                "  Macrocephaly (OFC >=97th centile): PATHOGNOMONIC PHTS -- most sensitive indicator.\n"
                "  Lhermitte-Duclos disease (dysplastic gangliocytoma cerebellum): PATHOGNOMONIC.\n"
                "  Trichilemmoma (facial): PATHOGNOMONIC hair follicle hamartoma.\n"
                "  Cobblestone gingival papillomatosis; papillomatous papules.\n\n"
                "CERVICAL AND ENDOMETRIAL:\n"
                "  Endometrial: 28-44% DOMINANT gynaecological PTEN risk.\n"
                "  Cervical adenocarcinoma: elevated (PTEN expression high in endocervical glands).\n"
                "  PTEN-null IHC: complete loss in tumour = LOF PATHOGNOMONIC.\n"
                "  Any cervical/endometrial glandular tumour + macrocephaly -> germline PTEN mandatory.\n\n"
                "FULL PHTS SPECTRUM:\n"
                "  Breast: 85% DOMINANT; thyroid: 35% (follicular); renal: 34%; colorectal: 9%.\n\n"
                "EVEROLIMUS (mTOR INHIBITOR):\n"
                "  PTEN LOF -> mTOR hyperactivation -> everolimus rationale.\n"
                "  Lenvatinib + everolimus: FDA-approved advanced endometrial cancer.\n"
                "  PTEN-null cervical adenocarcinoma: everolimus-based trials.\n\n"
                "SURVEILLANCE:\n"
                "  Annual breast MRI from age 30yr.\n"
                "  Annual pelvic USS (endometrial/cervical) from age 30yr.\n"
                "  Annual thyroid US from age 18yr.\n"
                "  Brain MRI if neurological symptoms (Lhermitte-Duclos)."
            ),
        },
        {
            "term": "ATM / A-T-BIALLELIC / RADIOSENSITIVITY-ABSOLUTE / CERVICAL-CHEMORADIATION-CI-AT / CERALASERTIB-ATRi",
            "definition": (
                "ATM -- 3056aa / 350 kDa / 11q22.3 / AD LOF (monoallelic) / AR biallelic A-T\n"
                "A-T biallelic; radiosensitivity ABSOLUTE; cervical chemoradiation = ABSOLUTE CI biallelic A-T; monoallelic cervical 2-3x.\n\n"
                "ATAXIA-TELANGIECTASIA (BIALLELIC A-T):\n"
                "  Cerebellar ataxia: progressive from age 1-2yr; wheelchair-bound age 10yr.\n"
                "  Telangiectasia (bulbar conjunctival): age 2-8yr.\n"
                "  Immunodeficiency: IgA/IgG; sinopulmonary infections.\n"
                "  Lymphoid malignancy: 80-100x elevated PRIMARY A-T cancer.\n\n"
                "RADIOSENSITIVITY ABSOLUTE -- BIALLELIC A-T:\n"
                "  A-T: AVOID ALL THERAPEUTIC RADIATION -- LETHAL.\n"
                "  Standard cervical cancer treatment = chemoradiation = ABSOLUTE CI in biallelic A-T.\n"
                "  Cervical cancer in A-T: radical hysterectomy = SURGERY MANDATORY.\n"
                "  RT sensitivity >2x vs normal; even diagnostic X-rays minimise.\n\n"
                "MONOALLELIC ATM -- CERVICAL:\n"
                "  Monoallelic ATM: cervical 2-3x elevated (modest); breast 15-25%.\n"
                "  Pancreatic 5-10x; prostate 2-4x.\n"
                "  Standard cervical screening sufficient for monoallelic ATM.\n\n"
                "CERALASERTIB (ATRi) + OLAPARIB:\n"
                "  ATM LOF -> HRD-like -> PARP inhibitor synthetic lethality.\n"
                "  Ceralasertib (ATR inhibitor) + olaparib: clinical trials ATM-deficient solid tumours.\n\n"
                "SURVEILLANCE:\n"
                "  AVOID radiation for all imaging (MRI/US preferred).\n"
                "  Annual breast MRI from age 25yr (monoallelic).\n"
                "  Standard cervical screening programme for monoallelic ATM."
            ),
        },
        {
            "term": "BRCA2 / HBOC / CERVICAL-2-3X / FA-D1-BIALLELIC-MOST-SEVERE / OLAPARIB-PARP-CISPLATIN",
            "definition": (
                "BRCA2 -- 3418aa / 384 kDa / 13q12.3 / AD LOF\n"
                "HBOC; cervical adenocarcinoma 2-3x; FA-D1 biallelic MOST SEVERE FA; olaparib PARP; cisplatin sensitive.\n\n"
                "HBOC CERVICAL RISK:\n"
                "  Cervical adenocarcinoma: 2-3x elevated (predominantly adenocarcinoma NOT squamous).\n"
                "  HRD -> genomic instability; platinum/olaparib benefit.\n"
                "  Standard cervical screening sufficient; no additional cervical imaging.\n\n"
                "BREAST AND OVARIAN (DOMINANT):\n"
                "  Breast: 69% lifetime; ovarian: 17%; male breast: 6-8% HIGHEST hereditary.\n"
                "  Prostate: 8x HIGHEST hereditary prostate risk; pancreatic 3-5x.\n\n"
                "BSO AND OLAPARIB:\n"
                "  BSO age 40-45yr (BRCA2 later than BRCA1 due to later ovarian onset).\n"
                "  Olaparib FDA-approved: BRCA2-germline ovarian, breast, pancreatic, prostate.\n"
                "  Cisplatin/carboplatin: HRD tumours platinum-sensitive.\n\n"
                "FA-D1 BIALLELIC -- MOST SEVERE FA:\n"
                "  BRCA2 = FANCD1; biallelic = Fanconi Anemia type D1 = MOST SEVERE FA.\n"
                "  FA-D1: medulloblastoma + Wilms + AML in infancy/early childhood.\n"
                "  AVOID alkylating/cross-linking agents in FA-D1 children.\n\n"
                "CASCADE TESTING -- HEREDITARY CERVICAL CANCER PANEL:\n"
                "  STK11: adenoma malignum/SCTAT -> STK11 sequencing MANDATORY; pelvic MRI annual.\n"
                "  FANCA: early-onset cervical SCC + BMF/radial ray -> FA panel; HPV vaccination CRITICAL.\n"
                "  TP53: cervical carcinosarcoma + p53 aberrant IHC -> germline TP53; NO chemoradiation.\n"
                "  MSH2: cervical adenocarcinoma + dMMR IHC -> Lynch workup; EPCAM MLPA if MSH2 suspected.\n"
                "  ATM/BRCA1/BRCA2/PTEN: HRD/glandular adenocarcinoma -> HRD panel; olaparib benefit.\n\n"
                "TIER 1 -- MOST ACTIONABLE (CERVICAL-SPECIFIC):\n"
                "  STK11: SCTAT/adenoma malignum surveillance; pelvic MRI annual from 18yr.\n"
                "  FANCA: HPV vaccination MANDATORY; AVOID aldehyde/radiation; annual cervical from 16yr.\n"
                "  TP53: SURGERY not chemoradiation for cervical cancer; WBMRI Toronto annually.\n\n"
                "TIER 2 -- DNA REPAIR (HRD):\n"
                "  BRCA1/BRCA2/ATM: cisplatin/olaparib/ceralasertib sensitivity.\n"
                "  PTEN: mTOR/everolimus; lenvatinib + everolimus FDA-approved.\n\n"
                "TIER 3 -- IMMUNE:\n"
                "  MSH2 dMMR: pembrolizumab MSI-H all histologies.\n"
                "  Aspirin CAPP2 50% CRC reduction LEVEL A Lynch."
            ),
        },
    ]

    return {
        "atlas":       "Hereditary-Cervical-Cancer-Predisposition-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": defs,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("genes_detail",)}, indent=2))
    print("\n=== BREAKDOWN summary ===")
    br = generate_breakdown()
    for row in br["breakdown"]:
        print(f"  {row['gene']:10s} n={row['n']} mean_age={row['mean_age_onset']} "
              f"severe_n={row['severe_n']} ({row['severe_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term']}")
