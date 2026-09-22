#!/usr/bin/env python3
"""Hereditary-Gastric-Cancer-Atlas — Complete 8-Gene Gastric Cancer Predisposition Atlas
CDH1   (E-cadherin; 784 aa; 16q22.1; AD LOF;
         Hereditary Diffuse Gastric Cancer (HDGC);
         Gastric lifetime risk 40-83% (penetrance varies by family); DIFFUSE/SIGNET-RING CELL histology PATHOGNOMONIC;
         Lobular breast cancer 42-55% lifetime; prophylactic TOTAL GASTRECTOMY mandatory (positive occult SRC in 80-100%);
         seed SEED_BASE+0) ·
CTNNA1 (alpha-E-catenin; 906 aa; 5q31.3; AD LOF;
         HDGC without CDH1; familial diffuse gastric cancer;
         Alpha-catenin loss -> loss of intercellular adhesion complex -> diffuse SRC histology same as CDH1;
         Fewer data than CDH1; prophylactic gastrectomy same protocol;
         seed SEED_BASE+1) ·
BRCA2  (BRCA2/FANCD1; 3418 aa; 13q12.3; AD LOF;
         HBOC; gastric cancer 3-7x RR (particularly intestinal-type);
         Breast 45-65%; ovarian 15-25%; pancreatic 5-7%; prostate elevated;
         BIALLELIC = Fanconi anaemia FANCD1 (childhood cancer: medulloblastoma, Wilms, AML);
         PARP inhibitors (olaparib/rucaparib/niraparib) FDA-approved;
         seed SEED_BASE+2) ·
PALB2  (PALB2/FANCN; 1186 aa; 16p12.2; AD LOF;
         Partner and localiser of BRCA2; FANCN (biallelic Fanconi);
         Breast 35-60% lifetime; pancreatic 2-5x RR; gastric 2-3x RR;
         PARP inhibitor eligibility (olaparib FDA2022 for PALB2 early breast);
         seed SEED_BASE+3) ·
ATM    (ATM serine/threonine kinase; 3056 aa; 11q22.3; AD LOF;
         Ataxia-telangiectasia heterozygote = hereditary cancer predisposition;
         Gastric ~2-4x RR; breast 15-25% (moderate risk); pancreatic; prostate elevated;
         ATM = DNA damage response kinase -- homologous recombination deficiency;
         Olaparib response (ATM-LOF = HRD); avoid excess radiation;
         seed SEED_BASE+4) ·
TP53   (Tumour protein p53; 393 aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome (LFS); GASTRIC is a classical LFS tumour type;
         Whole-body MRI surveillance annual MANDATORY; breast 25-50%; sarcoma 15-25%;
         AVOID RADIATION (radiation-induced secondary cancers in TP53 carriers);
         Brain tumours 10-15%; adrenocortical carcinoma (ACC) 15-20%;
         seed SEED_BASE+5) ·
RNF43  (RING finger protein 43; 783 aa; 17q22; AD LOF;
         Wnt-pathway E3 ubiquitin ligase (negative regulator); serrated gastric polyposis;
         Gastric sessile serrated adenomas -> gastric adenocarcinoma;
         RNF43 LOF = Wnt ligand hypersensitivity; RSPO3 amplification tumours also RNF43-LOF;
         seed SEED_BASE+6) ·
POLE   (DNA polymerase epsilon catalytic subunit; 2286 aa; 12q24.33; AD GOF exonuclease hotspot;
         Germline POLE exonuclease domain mutations (P286R, V411L, L424V, S459F) -> ULTRA-HYPERMUTATED phenotype;
         Gastric + endometrial + colorectal; TMB >100 mut/Mb typical;
         EXCEPTIONAL immunotherapy response (pembrolizumab FDA-approved all TMB-high solid tumours);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3094-3101)
"""
import random

SEED_BASE = 3094

ATLAS_GENES = [
    {
        "gene": "CDH1",
        "protein": (
            "CDH1 -- 16q22.1 Autosomal-Dominant-LOF -- 784aa -- E-Cadherin-Cell-Adhesion-Molecule-"
            "Hereditary-Diffuse-Gastric-Cancer-HDGC-Signet-Ring-Cell-Carcinoma-PATHOGNOMONIC-"
            "Prophylactic-Total-Gastrectomy-Mandatory-Lobular-Breast-Cancer-OMIM-192090"
        ),
        "locus": "16q22.1",
        "protein_size": (
            "784 aa / 87 kDa (CDH1; E-cadherin; type 1 classical cadherin; "
            "STRUCTURE: extracellular domain (EC1-EC5) -- calcium-binding repeats (3 Ca2+ ions per domain interface); "
            "transmembrane domain; intracellular tail binds beta-catenin (CTNNB1) and alpha-catenin (CTNNA1); "
            "FUNCTION: "
            "  Homophilic trans-interactions: EC1-EC1 binding between adjacent cells -> epithelial adhesion; "
            "  E-cadherin/catenin complex anchors actin cytoskeleton via alpha-catenin; "
            "  Tumour suppression: E-cadherin sequesters beta-catenin at cell membrane -- "
            "    CDH1-LOF -> beta-catenin nuclear translocation -> WNT target gene activation; "
            "  Diffuse gastric carcinoma histogenesis: loss of CDH1 -> non-cohesive single cells infiltrate stroma "
            "    (signet-ring cell carcinoma -- mucin displaces nucleus to periphery); "
            "HDGC (Hereditary Diffuse Gastric Cancer): "
            "  Clinical criteria (IGCLC 2020): "
            "    - >= 2 diffuse gastric cancers in FDR/SDR, 1 diagnosed <50yr; "
            "    - >= 3 diffuse gastric cancers in FDR/SDR; "
            "    - Diffuse GC < 40yr; "
            "    - Personal or FDR history DGC + lobular breast cancer (1 <70yr); "
            "  CDH1 pathogenic variants in ~25-40% of clinical HDGC families; "
            "  Penetrance: gastric cancer 40-83% (male) / 44-83% (female) lifetime (family-dependent); "
            "  Occult signet-ring cell foci found in 80-100% of prophylactic gastrectomy specimens; "
            "LOBULAR BREAST CANCER: "
            "  Lifetime risk 42-55% in female CDH1 carriers; "
            "  E-cadherin LOSS is the defining molecular feature of invasive lobular carcinoma (ILC); "
            "  Enhanced breast MRI annually from age 30 (not US/mammography alone); "
            "PROPHYLACTIC TOTAL GASTRECTOMY: "
            "  STRONGLY RECOMMENDED for all pathogenic CDH1 carriers after age 20-25 (when family complete); "
            "  Risk-benefit: morbidity of total gastrectomy vs near-certain cancer risk; "
            "  Pre-gastrectomy: mandatory upper GI endoscopy (targeted biopsies) -- Cambridge protocol; "
            "    BUT negative endoscopy does NOT obviate gastrectomy (occult SRC below endoscopic resolution); "
            "  Surgical technique: R0 total gastrectomy with Roux-en-Y reconstruction; "
            "    Preserve spleen/pancreas if possible; "
            "  Nutritional sequelae: B12 injection lifelong; iron supplementation; Ca/Vit D; dumping syndrome management."
        ),
        "inheritance": (
            "AD LOF 16q22.1 -- Hereditary Diffuse Gastric Cancer (HDGC). Penetrance 40-83% gastric lifetime. "
            "Lobular breast cancer 42-55% (female). Prophylactic total gastrectomy strongly recommended age 20-25. "
            "Large genomic deletions in CDH1 account for ~5% of CDH1 pathogenic variants -- MLPA essential alongside sequencing. "
            "Founder mutations: Newfoundland p.A634V; Maori c.1137+1G>A; Dutch c.2398delC."
        ),
        "disease_category": "Hereditary Diffuse Gastric Cancer / Lobular Breast Cancer",
        "patient_generator_params": {
            "gastric_cancer_risk": 0.70, "lobular_breast_risk": 0.48, "colorectal_risk": 0.04,
            "prophylactic_gastrectomy_done": 0.45, "occult_src_on_path": 0.85,
            "signet_ring_histology": 0.90, "intestinal_histology": 0.05,
            "endoscopy_negative_pre_sx": 0.75,
            "age_range": (22, 70), "mean_age_dx": 38,
            "severity_dist": {"severe": 0.35, "moderate": 0.45, "mild": 0.20},
        },
    },
    {
        "gene": "CTNNA1",
        "protein": (
            "CTNNA1 -- 5q31.3 Autosomal-Dominant-LOF -- 906aa -- Alpha-E-Catenin-Cadherin-Associated-"
            "Protein-HDGC-Without-CDH1-Familial-Diffuse-Gastric-Cancer-Adherens-Junction-"
            "Same-Gastrectomy-Protocol-as-CDH1-OMIM-116805"
        ),
        "locus": "5q31.3",
        "protein_size": (
            "906 aa / 100 kDa (CTNNA1; alpha-E-catenin; vinculin homologue; adherens junction component; "
            "STRUCTURE: N-terminal domain binds CTNNB1 (beta-catenin) and VCL (vinculin); "
            "central domain: homodimerisation; C-terminal: actin-binding module; "
            "Mechanosensing: alpha-catenin undergoes force-dependent conformational change, "
            "  exposing cryptic vinculin-binding site -> strengthening actin connection under tension; "
            "FUNCTION: "
            "  Links CDH1/CTNNB1 complex to actin cytoskeleton -- alpha-catenin is the crucial cytoskeletal anchor; "
            "  CTNNA1-LOF: E-cadherin complex destabilised -> cell-cell adhesion loss -> diffuse gastric histology; "
            "  Acts independently of beta-catenin nuclear signalling (unlike CDH1-LOF): "
            "    CTNNA1-LOF does NOT directly activate Wnt target genes -- distinct oncogenic mechanism; "
            "HDGC without CDH1: "
            "  CTNNA1 pathogenic variants explain a proportion of CDH1-negative HDGC families; "
            "  Penetrance less well-defined than CDH1 (smaller case series); "
            "  Diffuse gastric cancer + signet-ring cell histology clinically identical to CDH1-HDGC; "
            "  Lobular breast cancer risk: data limited but biologically plausible (E-cadherin complex integrity); "
            "MANAGEMENT IDENTICAL TO CDH1-HDGC: "
            "  Prophylactic total gastrectomy strongly recommended when pathogenic variant confirmed; "
            "  Cambridge protocol endoscopy pre-gastrectomy (targeted biopsies); "
            "  Female carriers: enhanced breast MRI from age 30 (lobular risk extrapolated from CDH1 data); "
            "  Cascade testing first-degree relatives; "
            "  IMPORTANT: Do NOT exclude CTNNA1 in CDH1-negative HDGC families -- dedicated CTNNA1 sequencing + MLPA required."
        ),
        "inheritance": (
            "AD LOF 5q31.3 -- HDGC without CDH1. Penetrance less well-defined; assumed similar to CDH1 from case series. "
            "Prophylactic total gastrectomy recommended same as CDH1. Lobular breast cancer risk presumed elevated. "
            "CTNNA1 large deletions reported -- MLPA/CMA required alongside sequencing."
        ),
        "disease_category": "Hereditary Diffuse Gastric Cancer (CDH1-negative) / Alpha-Catenin Deficiency",
        "patient_generator_params": {
            "gastric_cancer_risk": 0.58, "lobular_breast_risk": 0.38, "colorectal_risk": 0.03,
            "prophylactic_gastrectomy_done": 0.40, "occult_src_on_path": 0.78,
            "signet_ring_histology": 0.88, "intestinal_histology": 0.06,
            "endoscopy_negative_pre_sx": 0.72,
            "age_range": (25, 72), "mean_age_dx": 44,
            "severity_dist": {"severe": 0.30, "moderate": 0.48, "mild": 0.22},
        },
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- BRCA2-FANCD1-DNA-Repair-"
            "HBOC-Gastric-3-7x-RR-Breast-45-65pct-Ovarian-15-25pct-Pancreatic-5-7pct-"
            "PARP-Inhibitor-Olaparib-FDA-Approved-Biallelic-Fanconi-FANCD1-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa (BRCA2; FANCD1; breast cancer type 2 susceptibility protein; "
            "STRUCTURE: 8 BRC repeats (aa 1002-2085) -- binds RAD51 monomers; "
            "  OB folds (DBD: DNA-binding domain): ssDNA binding; "
            "  C-terminal domain: binds PALB2 N-terminus (interacting scaffold); "
            "  BRCA1-BRCA2-PALB2: three-gene complex essential for nuclear DSB repair; "
            "FUNCTION: "
            "  Homologous recombination (HR) repair of DNA double-strand breaks (DSBs): "
            "    BRCA2 delivers RAD51 nucleoprotein filaments to ssDNA at DSB; "
            "    RAD51 polymerises on ssDNA -> invades homologous template -> error-free repair; "
            "  Without BRCA2: cell relies on NHEJ (error-prone) -> chromosomal instability; "
            "  Tumour suppression: biallelic LOF required for tumour initiation (second hit); "
            "  FANCD1 function: BRCA2 is the biallelic FANCONI gene FANCD1; "
            "    Biallelic BRCA2: Fanconi anaemia complementation group D1 -- childhood cancer (brain, AML, Wilms); "
            "GASTRIC CANCER: "
            "  3-7x relative risk vs general population; "
            "  Predominantly INTESTINAL type (unlike CDH1 diffuse histology); "
            "  Cumulative risk: ~3-10% lifetime (depends on family history, sex, ethnicity); "
            "  H. pylori eradication MANDATORY for all BRCA2 carriers (H. pylori + BRCA2 = additive risk); "
            "PARP INHIBITOR THERAPY: "
            "  Olaparib (Lynparza) FDA-approved: "
            "    HER2-negative advanced breast (adjuvant + metastatic); "
            "    Germline BRCA1/2 pancreatic cancer maintenance (POLO trial); "
            "    Germline BRCA1/2 ovarian cancer (first-line + recurrent); "
            "    Germline BRCA1/2 prostate cancer (PROfound); "
            "  Rucaparib / niraparib: similar indications; "
            "  MECHANISM: PARP trapping + replication fork collapse -> synthetic lethality in HR-deficient cells; "
            "  Gastric cancer (germline BRCA2): PARP inhibitor sensitivity -- ongoing trials (JAVELIN Gastric 100, MONOCLE)."
        ),
        "inheritance": (
            "AD LOF 13q12.3 -- HBOC with gastric component. Gastric 3-7x RR; breast 45-65% lifetime; ovarian 15-25%; pancreatic 5-7%. "
            "Biallelic BRCA2 = FANCD1 Fanconi anaemia (childhood cancer). PARP inhibitors (olaparib/rucaparib/niraparib) FDA-approved. "
            "Founder mutations: Ashkenazi 6174delT; Polish c.9067_9068delTT; Icelandic 999del5."
        ),
        "disease_category": "HBOC / Gastric Cancer / Fanconi Anaemia FANCD1",
        "patient_generator_params": {
            "gastric_cancer_risk": 0.08, "breast_cancer_risk": 0.55, "ovarian_cancer_risk": 0.20,
            "pancreatic_cancer_risk": 0.06, "prostate_cancer_risk": 0.15,
            "parp_inhibitor_eligible": 0.60, "hpylori_eradication_done": 0.65,
            "intestinal_histology": 0.80, "signet_ring_histology": 0.05,
            "age_range": (30, 75), "mean_age_dx": 53,
            "severity_dist": {"severe": 0.25, "moderate": 0.50, "mild": 0.25},
        },
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- PALB2-FANCN-Partner-Localiser-BRCA2-"
            "Breast-35-60pct-Pancreatic-Elevated-Gastric-2-3x-RR-PARP-Inhibitor-Olaparib-FDA2022-"
            "Biallelic-FANCN-Fanconi-Anaemia-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa (PALB2; partner and localiser of BRCA2; FANCN; "
            "STRUCTURE: N-terminal coiled-coil domain: binds BRCA1 BRCT domain; "
            "  central WD40 repeat domain: protein interaction scaffold; "
            "  C-terminal: binds BRCA2 N-terminus -> forms ternary BRCA1-PALB2-BRCA2 complex; "
            "FUNCTION: "
            "  Nuclear anchor: PALB2 recruits BRCA2 to sites of DNA damage (DSBs) via BRCA1; "
            "  Without PALB2: BRCA2 fails to localise to nuclear foci -> HR deficiency; "
            "  BRCA1-PALB2-BRCA2 axis: essential three-protein HR complex; "
            "  PALB2 is not redundant -- PALB2-LOF phenocopies BRCA2-LOF in terms of HR defect; "
            "CANCER RISKS: "
            "  Breast: 35-60% lifetime risk (moderate-to-high; similar to BRCA2 depending on variant); "
            "  Pancreatic: 2-5x RR (elevated moderate risk); "
            "  Gastric: 2-3x RR (emerging data from PALB2 registry studies); "
            "  Ovarian: slightly elevated vs general population (less than BRCA1/2); "
            "  Male breast cancer: elevated (PALB2 contributes to male breast cancer panel); "
            "PARP INHIBITOR THERAPY: "
            "  Olaparib FDA2022: germline PALB2 early-stage breast cancer (OlympiA trial showed DFS benefit); "
            "  PALB2 HR deficiency: similar PARP inhibitor sensitivity to BRCA1/2 carriers; "
            "  PALB2 pancreatic cancer: platinum + gemcitabine + PARP inhibitor maintenance trials ongoing; "
            "BIALLELIC PALB2 = FANCN: "
            "  Rare Fanconi anaemia complementation group N; "
            "  Very severe childhood cancer predisposition: medulloblastoma, Wilms, AML; "
            "  Similar but more severe phenotype to biallelic BRCA2/FANCD1."
        ),
        "inheritance": (
            "AD LOF 16p12.2 -- breast/pancreatic/gastric predisposition. Breast 35-60%; pancreatic 2-5x; gastric 2-3x. "
            "Biallelic PALB2 = FANCN Fanconi anaemia. Olaparib FDA2022 approved for germline PALB2 early breast. "
            "Founder mutations: Finnish c.1592delT; Polish c.509_510delGA; Finnish c.3323dupA."
        ),
        "disease_category": "PALB2 Hereditary Breast/Pancreatic/Gastric / FANCN",
        "patient_generator_params": {
            "gastric_cancer_risk": 0.06, "breast_cancer_risk": 0.45, "ovarian_cancer_risk": 0.06,
            "pancreatic_cancer_risk": 0.08, "prostate_cancer_risk": 0.08,
            "parp_inhibitor_eligible": 0.50, "hpylori_eradication_done": 0.60,
            "intestinal_histology": 0.75, "signet_ring_histology": 0.04,
            "age_range": (32, 75), "mean_age_dx": 52,
            "severity_dist": {"severe": 0.22, "moderate": 0.52, "mild": 0.26},
        },
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-LOF -- 3056aa -- ATM-Serine-Threonine-Kinase-"
            "Ataxia-Telangiectasia-Heterozygote-Gastric-2-4x-RR-Breast-15-25pct-Moderate-"
            "Avoid-Radiation-HRD-Olaparib-Sensitivity-Pancreatic-Elevated-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa (ATM; ataxia-telangiectasia mutated; PIKK family kinase; "
            "STRUCTURE: HEAT repeats (aa 1-1300): scaffolding/substrate recognition; "
            "  FAT domain; kinase domain (PI3K-like); FATC domain (C-terminus): regulatory; "
            "  ATM normally exists as inactive dimer -- DNA DSB-sensing via MRN complex (MRE11-RAD50-NBS1) activates ATM; "
            "FUNCTION (master DSB sensor): "
            "  Activated by DNA double-strand breaks (radiation, replication stress): "
            "  ATM trans-autophosphorylates (pS1981) -> monomerises -> active kinase; "
            "  Key substrates: H2AX (gamma-H2AX: DSB marker), BRCA1 (pS1387), CHK2 (pT68), p53 (pS15); "
            "  Cell cycle checkpoints: S-phase (via CHK2-CDC25A), G2/M (via CHK2-CDC25C); "
            "  Homologous recombination: ATM phosphorylates BRCA1 -> facilitates BRCA2-RAD51 loading; "
            "  Without ATM: inefficient HR + persistent unrepaired DSBs -> chromosomal instability; "
            "ATAXIA-TELANGIECTASIA (BIALLELIC): "
            "  Cerebellar ataxia onset childhood; oculomotor telangiectasia; combined immunodeficiency; "
            "  Extreme radiation sensitivity (no radiotherapy possible); lymphoma/leukaemia risk >>; "
            "  IgA deficiency; AFP elevated (diagnostic marker); "
            "HETEROZYGOTE CARRIERS (HEREDITARY CANCER): "
            "  ~1% of general population are ATM heterozygotes (carrier frequency); "
            "  Cancer risks (heterozygotes, not biallelic A-T): "
            "    Breast: 15-25% lifetime (moderate risk -- NOT BRCA1/2-level); "
            "    Gastric: 2-4x relative risk (meta-analysis data); "
            "    Pancreatic: 4-5x relative risk; "
            "    Prostate (aggressive): 2-4x RR; "
            "    CLL/B-cell lymphoma: slightly elevated; "
            "  HRD PHENOTYPE: ATM carriers have partial HR deficiency -- PARP inhibitor sensitivity; "
            "  Olaparib: PROFOUND trial (prostate) showed efficacy in ATM-mutant cohort; gastric ATM-LOF trials ongoing; "
            "  RADIATION CAUTION: heterozygous carriers may have intermediate radiation sensitivity "
            "    (not as severe as biallelic A-T but avoid excess diagnostic radiation where possible)."
        ),
        "inheritance": (
            "AD LOF 11q22.3 (heterozygous predisposition). Gastric 2-4x RR; breast 15-25% moderate risk; pancreatic 4-5x. "
            "Biallelic ATM = ataxia-telangiectasia (severe; radiation contraindicated). "
            "Avoid excess radiation in heterozygotes. ATM = HRD -- olaparib PROfound trial data for prostate. "
            "Founder mutations: Ashkenazi c.7271T>G; UK Irish founder; Polish c.1066-6T>G."
        ),
        "disease_category": "ATM-Heterozygote Hereditary Cancer / Gastric-Breast-Pancreatic Risk",
        "patient_generator_params": {
            "gastric_cancer_risk": 0.07, "breast_cancer_risk": 0.20, "ovarian_cancer_risk": 0.04,
            "pancreatic_cancer_risk": 0.07, "prostate_cancer_risk": 0.12,
            "parp_inhibitor_eligible": 0.40, "hpylori_eradication_done": 0.58,
            "intestinal_histology": 0.82, "signet_ring_histology": 0.03,
            "radiation_caution": 0.95,
            "age_range": (35, 80), "mean_age_dx": 58,
            "severity_dist": {"severe": 0.20, "moderate": 0.50, "mild": 0.30},
        },
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- Tumour-Protein-p53-Guardian-Genome-"
            "Li-Fraumeni-Syndrome-LFS-Gastric-Classical-LFS-Tumour-AVOID-RADIATION-Whole-Body-MRI-Mandatory-"
            "Breast-Sarcoma-Brain-ACC-Adrenocortical-Carcinoma-Germline-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa (TP53; p53; tumour protein p53; transcription factor; "
            "STRUCTURE: N-terminal transactivation domain (TAD1 aa1-40, TAD2 aa40-67); "
            "  proline-rich region (aa67-98); DNA-binding domain (DBD aa94-292) -- contains most hotspot mutations; "
            "  tetramerisation domain (aa325-356): p53 functions as a dimer of dimers; "
            "  C-terminal regulatory domain: acetylation/ubiquitination; "
            "FUNCTION (guardian of the genome): "
            "  Transcription factor activated by DNA damage, oncogene activation, hypoxia; "
            "  Activated by ATM (pS15), CHK1/CHK2 (pS20) -> MDM2 inhibition relieved -> p53 stabilised; "
            "  MDM2-p53 axis: MDM2 is the primary E3 ubiquitin ligase targeting p53 for degradation; "
            "  Target genes: "
            "    Cell cycle arrest: CDKN1A (p21 WAF1/CIP1 -> G1 arrest), GADD45A; "
            "    Apoptosis: BAX, PUMA, NOXA, APAF1, FAS; "
            "    Senescence: p21 + RB1 pathway; "
            "    DNA repair: DDB2, XPC, FANCC (Fanconi pathway); "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  Classic LFS tumours: sarcoma, breast cancer, brain tumours (glioblastoma, DIPG), "
            "    adrenocortical carcinoma (ACC), leukaemia; "
            "  Extended LFS/LFL (Li-Fraumeni-Like): gastric, colorectal, pancreatic, ovarian, pheo/PGL; "
            "  Gastric cancer: recognised LFS tumour -- ~5-15% of LFS kindreds have affected member; "
            "  Brazilian founder: p.R337H (prevalent Brazil due to founder effect, ~0.3% carrier frequency); "
            "  Clat2/Eeles criteria for testing; "
            "AVOID RADIATION (CRITICAL): "
            "  TP53 carriers: normal cells cannot properly G1-arrest after radiation -> unchecked radiation damage; "
            "  Radiation-induced secondary cancers documented in TP53 carriers receiving radiotherapy; "
            "  CONTRAINDICATION: avoid radiotherapy whenever alternative chemotherapy/surgery feasible; "
            "  Diagnostic imaging: prefer MRI/USS over CT/X-ray -- minimise cumulative radiation; "
            "SURVEILLANCE: "
            "  WHOLE-BODY MRI (WBMRI) annually: "
            "    Detects brain, breast, soft tissue, visceral, abdominal tumours; "
            "    Toronto protocol (2016): WBMRI + annual breast MRI (from age 20-25) + annual dermatology; "
            "  Breast MRI annually from age 20 (earlier if family breast cancer <30yr); "
            "  Colonoscopy every 2-5yr from 25; OGD 2-5yr; "
            "  Brain MRI annually (if family history brain); "
            "  Adrenal imaging: annual USS/CT if family history ACC."
        ),
        "inheritance": (
            "AD LOF 17p13.1 -- Li-Fraumeni Syndrome (LFS). Gastric is a classical LFS tumour. "
            "AVOID RADIATION -- use MRI/USS instead of CT/X-ray where possible. "
            "Whole-body MRI annually mandatory surveillance. Breast cancer 25-50%; sarcoma 15-25%; brain 10-15%; ACC 15-20%. "
            "Brazilian founder p.R337H (0.3% Brazil carrier rate). MDM2 inhibitors (clinical trials)."
        ),
        "disease_category": "Li-Fraumeni Syndrome / Gastric-Sarcoma-Breast-Brain-ACC",
        "patient_generator_params": {
            "gastric_cancer_risk": 0.12, "breast_cancer_risk": 0.40, "sarcoma_risk": 0.20,
            "brain_tumour_risk": 0.12, "acc_risk": 0.18, "colorectal_risk": 0.06,
            "avoid_radiation": 0.98, "wbmri_surveillance": 0.70,
            "intestinal_histology": 0.65, "signet_ring_histology": 0.10,
            "hpylori_eradication_done": 0.55,
            "age_range": (18, 65), "mean_age_dx": 32,
            "severity_dist": {"severe": 0.40, "moderate": 0.42, "mild": 0.18},
        },
    },
    {
        "gene": "RNF43",
        "protein": (
            "RNF43 -- 17q22 Autosomal-Dominant-LOF -- 783aa -- RING-Finger-Protein-43-Wnt-E3-Ligase-"
            "Gastric-Serrated-Polyposis-Sessile-Serrated-Adenoma-WNT-Pathway-RSPO3-Amplification-"
            "Gastric-Adenocarcinoma-Risk-OMIM-612482"
        ),
        "locus": "17q22",
        "protein_size": (
            "783 aa / 88 kDa (RNF43; RING finger protein 43; E3 ubiquitin-protein ligase; "
            "STRUCTURE: extracellular domain: binds LGR4/5 receptor complex; "
            "  single-pass transmembrane domain; "
            "  intracellular RING finger domain: E3 ubiquitin ligase activity; "
            "  PA domain (protease-associated); "
            "FUNCTION (Wnt pathway negative regulator): "
            "  In absence of Wnt ligand: RNF43 ubiquitinates Frizzled receptors (FZD) -> proteasomal degradation; "
            "  LGR4/5-RSPO3 binding: R-spondins bind LGR4/5, inhibiting RNF43 E3 ligase activity -> FZD stabilised; "
            "  RNF43-LOF: Frizzled receptors accumulate on cell surface -> hypersensitive to Wnt ligands; "
            "  Oncogenic consequence: elevated Wnt/beta-catenin signalling -> cell proliferation/stem-cell expansion; "
            "  RNF43 is a p53 target gene (forms regulatory axis); "
            "GASTRIC CANCER PREDISPOSITION: "
            "  RNF43 germline LOF variants: gastric serrated polyposis + gastric adenocarcinoma; "
            "  Serrated gastric lesions -> risk of progression to gastric adenocarcinoma; "
            "  Sessile serrated adenomas (SSAs) gastric + colonic in RNF43 families; "
            "  RSPO3 amplification (somatic): common in RNF43-mutant gastric cancers (RSPO bypasses E3 inhibition); "
            "  Porcupine inhibitors (WNT974, LGK-974): Wnt ligand secretion inhibitor in RNF43-LOF cancers (trials); "
            "CLINICAL FEATURES: "
            "  Gastric polyposis: fundic gland polyps + sessile serrated polyps; "
            "  Colonoscopy: serrated adenomas colon + cecum; "
            "  Pancreatic: KRAS/BRAF-driven tumours in RNF43-LOF context; "
            "  Surveillance: annual OGD from 30-35 for gastric polyposis monitoring; "
            "  Colonic SSA surveillance: colonoscopy every 1-3yr depending on polyp burden; "
            "  Porcupine inhibitor trials for RNF43-LOF gastric cancer: biomarker stratification required."
        ),
        "inheritance": (
            "AD LOF 17q22 -- gastric serrated polyposis + gastric cancer predisposition. "
            "Wnt-pathway E3 ligase; RSPO3 amplification in RNF43-LOF tumours. "
            "Annual OGD from 30-35 for gastric polyposis surveillance. Porcupine inhibitors (clinical trials for Wnt-addicted RNF43-LOF gastric Ca)."
        ),
        "disease_category": "Gastric Serrated Polyposis / Wnt-Pathway Gastric Cancer",
        "patient_generator_params": {
            "gastric_cancer_risk": 0.22, "gastric_polyps_risk": 0.75, "colorectal_ssa_risk": 0.45,
            "pancreatic_cancer_risk": 0.05, "breast_cancer_risk": 0.08,
            "hpylori_eradication_done": 0.62, "endoscopy_surveillance_current": 0.70,
            "intestinal_histology": 0.70, "signet_ring_histology": 0.06,
            "wnt_pathway_enriched": 0.80,
            "age_range": (30, 72), "mean_age_dx": 50,
            "severity_dist": {"severe": 0.20, "moderate": 0.55, "mild": 0.25},
        },
    },
    {
        "gene": "POLE",
        "protein": (
            "POLE -- 12q24.33 Autosomal-Dominant-GOF-Exonuclease-Hotspot -- 2286aa -- "
            "DNA-Polymerase-Epsilon-Catalytic-Subunit-Germline-Exonuclease-Domain-ULTRA-HYPERMUTATED-"
            "TMB-gt100-Gastric-Endometrial-Colorectal-EXCEPTIONAL-Immunotherapy-P286R-V411L-"
            "Pembrolizumab-TMB-High-FDA2020-OMIM-174762"
        ),
        "locus": "12q24.33",
        "protein_size": (
            "2286 aa / 261 kDa (POLE; DNA polymerase epsilon catalytic subunit; B-family DNA polymerase; "
            "STRUCTURE: N-terminal palm-fingers-thumb domain: polymerase active site; "
            "  Exonuclease domain (aa 268-476): 3'->5' proofreading; "
            "  CTD (C-terminal domain): processivity, interaction with POLE2 (accessory subunit) and RFC; "
            "FUNCTION: "
            "  Leading strand replicative polymerase (with PCNA processivity clamp); "
            "  3'->5' exonuclease (proofreading): corrects misincorporated nucleotides in real-time; "
            "  POLE exonuclease catalytic residues: D275, E277, D368 -- hotspot mutations disable proofreading; "
            "GERMLINE EXONUCLEASE DOMAIN MUTATIONS: "
            "  Hotspot variants: P286R (c.857C>G), V411L (c.1231G>T), L424V, S459F, M444K; "
            "  These mutations INACTIVATE proofreading -> 10-1000x increased mutation rate; "
            "  ULTRA-HYPERMUTATION: TMB typically 100-1000 mut/Mb (vs normal CRC 1-10 mut/Mb); "
            "CANCER PREDISPOSITION: "
            "  POLE germline: predisposition to colorectal polyps + CRC; gastric; endometrial; brain tumours; "
            "  Colorectal polyposis: 10-100 polyps (oligo/attenuated phenotype); "
            "  Gastric cancer: intestinal type with ultra-high TMB; "
            "  Endometrial: very high TMB endometrial cancer (excellent prognosis post-PD-1); "
            "  Brain: glioma risk (particularly pediatric gliomas POLE-associated); "
            "IMMUNOTHERAPY (CRITICAL CLINICAL PEARL): "
            "  TMB-high (>10 mut/Mb) FDA2020: pembrolizumab tissue-agnostic; "
            "  POLE ultra-hypermutated tumours: EXCEPTIONAL responders to PD-1 blockade; "
            "  MSS/pMMR POLE tumours: respond to pembrolizumab DESPITE intact MMR (TMB drives response, not MMR); "
            "  POLE gastric: test TMB in all advanced/metastatic gastric cancer -- "
            "    POLE-driven ultra-hypermutation predicts immunotherapy benefit even in MSS tumours; "
            "  Ipilimumab + nivolumab: CheckMate 649 gastric -- POLE subset analysis ongoing; "
            "DISTINCTION: "
            "  Somatic POLE mutations: common in endometrial (ultramutated subtype TCGA); "
            "  Germline POLE: rarer; syndromic polyposis/cancer predisposition; "
            "  MMR-proficient POLE tumours: MSI testing will be MSS -- must test TMB or POLE sequencing directly."
        ),
        "inheritance": (
            "AD GOF exonuclease domain 12q24.33 -- ultra-hypermutated cancer predisposition. "
            "Hotspots: P286R, V411L, L424V, S459F. TMB >100 mut/Mb typical; MSS on standard MSI testing (not MSI-H). "
            "EXCEPTIONAL immunotherapy responders (pembrolizumab FDA2020 TMB-high). "
            "Gastric+endometrial+colorectal predisposition. Test TMB directly -- do NOT rely on MSI status alone."
        ),
        "disease_category": "POLE Ultra-Hypermutated Gastric/Colorectal/Endometrial / TMB-High Immunotherapy",
        "patient_generator_params": {
            "gastric_cancer_risk": 0.18, "colorectal_cancer_risk": 0.30, "endometrial_cancer_risk": 0.22,
            "brain_tumour_risk": 0.05, "gastric_polyps_risk": 0.35, "colorectal_polyps_risk": 0.65,
            "tmb_high_pct": 0.95, "immunotherapy_response_excellent": 0.80,
            "mss_despite_tmb_high": 0.90,
            "hpylori_eradication_done": 0.60,
            "intestinal_histology": 0.85, "signet_ring_histology": 0.02,
            "age_range": (25, 68), "mean_age_dx": 46,
            "severity_dist": {"severe": 0.22, "moderate": 0.50, "mild": 0.28},
        },
    },
]


def _generate_patients_for_gene(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene_info = next(g for g in ATLAS_GENES if g["gene"] == gene)
    params = gene_info["patient_generator_params"]

    # Mutation sets per gene
    mutation_sets = {
        "CDH1":  ["p.A634V", "c.2398delC", "c.1137+1G>A", "p.E757K", "p.R732Q", "c.1901C>T", "IVS6+1G>A", "exon_deletion"],
        "CTNNA1":["p.R54C", "p.A897_L898del", "c.1392+1G>A", "p.E174K", "large_del_5q31", "p.R611Q", "c.2317C>T", "p.V408M"],
        "BRCA2": ["p.V1736A", "6174delT", "c.9067_9068delTT", "999del5", "p.R2336H", "c.5946delT", "p.L3049X", "IVS7+2T>A"],
        "PALB2": ["c.1592delT", "c.3323dupA", "c.509_510delGA", "p.Y551X", "p.L939W", "c.2257C>T", "p.W1038X", "c.3116_3117insA"],
        "ATM":   ["c.7271T>G", "c.1066-6T>G", "p.R3047X", "c.5932G>C", "c.4258C>T", "p.S707X", "c.8525T>A", "c.7788delT"],
        "TP53":  ["p.R175H", "p.R248W", "p.R248Q", "p.R273H", "p.G245S", "p.R337H", "p.R282W", "p.V143A"],
        "RNF43": ["p.G659Vfs", "p.R117X", "c.1460dupA", "p.E183X", "p.L489X", "c.2047C>T", "p.R117fs", "p.W480X"],
        "POLE":  ["p.P286R", "p.V411L", "p.L424V", "p.S459F", "p.M444K", "p.D275N", "p.Y458D", "p.L424P"],
    }

    severity_opts = (
        ["severe"] * int(params["severity_dist"]["severe"] * 100) +
        ["moderate"] * int(params["severity_dist"]["moderate"] * 100) +
        ["mild"] * int(params["severity_dist"]["mild"] * 100)
    )
    mutations = mutation_sets.get(gene, ["unknown_variant"])
    age_range = params.get("age_range", (30, 70))

    patients = []
    for i in range(n):
        age_dx = rng.randint(*age_range)
        if age_dx < params.get("mean_age_dx", 50) - 5:
            age_dx = rng.randint(age_range[0], params.get("mean_age_dx", 50) + 10)
        sev = rng.choice(severity_opts)

        # Core cancer outcomes
        gastric_ca = rng.random() < params.get("gastric_cancer_risk", 0.10)
        breast_ca = rng.random() < params.get("breast_cancer_risk", 0.0)
        ovarian_ca = rng.random() < params.get("ovarian_cancer_risk", 0.0)
        pancreatic_ca = rng.random() < params.get("pancreatic_cancer_risk", 0.0)

        # Histology
        signet_ring = rng.random() < params.get("signet_ring_histology", 0.0)
        intestinal = rng.random() < params.get("intestinal_histology", 0.0)

        # Gastric-specific
        gastric_polyps = rng.random() < params.get("gastric_polyps_risk", 0.0)
        prophylactic_gastrectomy = rng.random() < params.get("prophylactic_gastrectomy_done", 0.0)
        occult_src = rng.random() < params.get("occult_src_on_path", 0.0)
        endoscopy_neg = rng.random() < params.get("endoscopy_negative_pre_sx", 0.0)

        # Treatment eligibility
        parp_eligible = rng.random() < params.get("parp_inhibitor_eligible", 0.0)
        tmb_high = rng.random() < params.get("tmb_high_pct", 0.0)
        immuno_response = rng.random() < params.get("immunotherapy_response_excellent", 0.0)
        mss_tmb_high = rng.random() < params.get("mss_despite_tmb_high", 0.0)

        # Surveillance
        hpylori_done = rng.random() < params.get("hpylori_eradication_done", 0.0)
        wbmri_done = rng.random() < params.get("wbmri_surveillance", 0.0)
        endo_current = rng.random() < params.get("endoscopy_surveillance_current", 0.0)

        # Other cancers
        sarcoma = rng.random() < params.get("sarcoma_risk", 0.0)
        brain = rng.random() < params.get("brain_tumour_risk", 0.0)
        acc = rng.random() < params.get("acc_risk", 0.0)
        prostate = rng.random() < params.get("prostate_cancer_risk", 0.0)
        colorectal = rng.random() < params.get("colorectal_cancer_risk", 0.0)
        colorectal_polyps = rng.random() < params.get("colorectal_polyps_risk", 0.0)
        colorectal_ssa = rng.random() < params.get("colorectal_ssa_risk", 0.0)
        avoid_radiation = rng.random() < params.get("avoid_radiation", 0.0)
        lobular_breast = rng.random() < params.get("lobular_breast_risk", 0.0)
        wnt_enriched = rng.random() < params.get("wnt_pathway_enriched", 0.0)

        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "age_at_diagnosis_yrs": age_dx,
            "mutation": rng.choice(mutations),
            "severity": sev,
            "gastric_cancer": gastric_ca,
            "lobular_breast_cancer": lobular_breast,
            "breast_cancer": breast_ca,
            "ovarian_cancer": ovarian_ca,
            "pancreatic_cancer": pancreatic_ca,
            "prostate_cancer": prostate,
            "colorectal_cancer": colorectal,
            "sarcoma": sarcoma,
            "brain_tumour": brain,
            "acc": acc,
            "signet_ring_histology": signet_ring,
            "intestinal_histology": intestinal,
            "gastric_polyps": gastric_polyps,
            "colorectal_polyps": colorectal_polyps,
            "colorectal_ssa": colorectal_ssa,
            "prophylactic_gastrectomy_done": prophylactic_gastrectomy,
            "occult_src_on_gastrectomy": occult_src,
            "endoscopy_negative_pre_surgery": endoscopy_neg,
            "parp_inhibitor_eligible": parp_eligible,
            "tmb_high": tmb_high,
            "immunotherapy_response_excellent": immuno_response,
            "mss_despite_tmb_high": mss_tmb_high,
            "hpylori_eradication_done": hpylori_done,
            "wbmri_surveillance": wbmri_done,
            "endoscopy_surveillance_current": endo_current,
            "avoid_radiation_flag": avoid_radiation,
            "wnt_pathway_enriched": wnt_enriched,
        })
    return patients


# ─── API generators ────────────────────────────────────────────────────────────

def generate_overview() -> dict:
    """Overview data for Hereditary-Gastric-Cancer-Atlas."""
    return {
        "atlas":          "Hereditary-Gastric-Cancer-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Gastric Cancer Predisposition Atlas "
            "(CDH1-CTNNA1-BRCA2-PALB2-ATM-TP53-RNF43-POLE)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "CDH1": (
                "AD LOF 16q22.1 (E-cadherin; 784aa; Hereditary Diffuse Gastric Cancer HDGC; "
                "DIFFUSE/SIGNET-RING-CELL histology PATHOGNOMONIC; gastric 40-83% lifetime; "
                "lobular breast 42-55%; PROPHYLACTIC TOTAL GASTRECTOMY MANDATORY -- "
                "80-100% occult SRC foci even with negative endoscopy)"
            ),
            "CTNNA1": (
                "AD LOF 5q31.3 (alpha-E-catenin; 906aa; HDGC without CDH1; "
                "Alpha-catenin links E-cadherin complex to actin -- LOF = same diffuse histology as CDH1; "
                "SAME GASTRECTOMY PROTOCOL as CDH1; CDH1-negative HDGC families require dedicated CTNNA1 testing)"
            ),
            "BRCA2": (
                "AD LOF 13q12.3 (BRCA2/FANCD1; 3418aa; HBOC; gastric 3-7x RR; "
                "breast 45-65%; ovarian 15-25%; pancreatic 5-7%; "
                "PARP inhibitors FDA-approved; biallelic = FANCD1 Fanconi anaemia; "
                "H. PYLORI ERADICATION MANDATORY for all BRCA2 carriers)"
            ),
            "PALB2": (
                "AD LOF 16p12.2 (PALB2/FANCN; 1186aa; BRCA2 partner-localiser; "
                "breast 35-60%; pancreatic 2-5x; gastric 2-3x; "
                "Olaparib FDA2022 early breast; biallelic = FANCN Fanconi anaemia; "
                "BRCA1-PALB2-BRCA2 ternary HR complex)"
            ),
            "ATM": (
                "AD LOF 11q22.3 (ATM kinase; 3056aa; heterozygote carrier predisposition; "
                "gastric 2-4x RR; breast 15-25% moderate; pancreatic 4-5x; "
                "AVOID EXCESS RADIATION; HRD phenotype -- olaparib sensitivity (PROfound); "
                "1% carrier frequency general population)"
            ),
            "TP53": (
                "AD LOF 17p13.1 (p53; 393aa; Li-Fraumeni Syndrome; gastric is CLASSICAL LFS tumour; "
                "AVOID RADIATION ABSOLUTELY -- radiation-induced secondary cancers documented; "
                "WHOLE-BODY MRI ANNUALLY MANDATORY; breast 25-50%; sarcoma 15-25%; ACC 15-20%)"
            ),
            "RNF43": (
                "AD LOF 17q22 (Wnt E3 ligase; 783aa; gastric serrated polyposis; "
                "Frizzled receptor ubiquitination -- LOF = Wnt hypersensitivity; "
                "Gastric sessile serrated adenomas -> gastric adenocarcinoma; "
                "RSPO3 amplification in RNF43-LOF tumours; porcupine inhibitors trials)"
            ),
            "POLE": (
                "AD GOF exonuclease domain 12q24.33 (DNA polymerase epsilon; 2286aa; "
                "P286R/V411L/L424V hotspots -- PROOFREADING DISABLED; TMB >100 mut/Mb; "
                "gastric+endometrial+colorectal+brain; MSS despite TMB-high; "
                "EXCEPTIONAL immunotherapy response; pembrolizumab FDA2020 TMB-high approval)"
            ),
        },
        "key_clinical_rules": [
            "CDH1: PROPHYLACTIC TOTAL GASTRECTOMY mandatory -- negative endoscopy does NOT exclude cancer (80-100% occult SRC foci on gastrectomy specimen)",
            "CTNNA1: CDH1-negative HDGC families require dedicated CTNNA1 sequencing + MLPA -- same gastrectomy recommendation as CDH1",
            "BRCA2: H. pylori eradication MANDATORY for all BRCA2 carriers (H. pylori + BRCA2 = additive gastric risk) -- check and treat",
            "ATM: AVOID EXCESS RADIATION -- heterozygous ATM carriers have intermediate radiation sensitivity; prefer MRI over CT for surveillance",
            "TP53/LFS: AVOID RADIATION ABSOLUTELY -- documented radiation-induced secondary cancers; whole-body MRI annually from diagnosis",
            "TP53: AVOID radiation therapy whenever chemotherapy/surgery alternative exists -- contraindicated, not just cautioned",
            "POLE: TEST TMB DIRECTLY for POLE-suspected tumours -- standard MSI testing will show MSS (POLE-driven ultra-hypermutation is MSS-TMB-high, not MSI-H)",
            "POLE: TMB >100 mut/Mb -> pembrolizumab -- EXCEPTIONAL responders even in MSS gastric cancer (do not miss POLE by relying on MSI alone)",
            "RNF43: Annual OGD surveillance from age 30-35 for gastric serrated polyposis -- manage like sessile serrated adenoma protocol",
            "CDH1/CTNNA1: Enhanced breast MRI annually from age 30 (lobular breast cancer risk -- mammography/USS insufficient for lobular histology)",
            "All genes: PANEL TESTING recommended for hereditary gastric cancer families -- multi-gene panel (CDH1+CTNNA1+BRCA1+BRCA2+PALB2+ATM+TP53+RNF43+POLE)",
        ],
        "gene_panel_note": (
            "Hereditary gastric cancer gene panel (2024 IGCLC guidance): "
            "MINIMUM: CDH1, CTNNA1; "
            "EXTENDED (hereditary gastric risk): BRCA2, PALB2, ATM, TP53, RNF43, POLE, BRCA1, MLH1, MSH2, STK11, APC; "
            "KEY CLINICAL DISTINCTIONS: "
            "  CDH1/CTNNA1: DIFFUSE/signet-ring cell histology -- prophylactic total gastrectomy; "
            "  BRCA2/PALB2/ATM: INTESTINAL type gastric predominance -- endoscopic surveillance + H. pylori eradication; "
            "  TP53/LFS: gastric is ONE of multiple cancer types -- whole-body MRI protocol; "
            "  RNF43: gastric SERRATED POLYPOSIS -- OGD surveillance for polyp progression; "
            "  POLE: ultra-hypermutated INTESTINAL type -- immunotherapy response outstanding; "
            "Surveillance summary: "
            "  CDH1/CTNNA1: prophylactic total gastrectomy (preferred) OR annual endoscopy (Cambridge protocol) if deferred; "
            "  BRCA2/PALB2/ATM: OGD every 3-5yr from 40 (personalise by family Hx); H. pylori test and treat; "
            "  TP53: whole-body MRI annually (Toronto LFS protocol); OGD in surveillance bundle; "
            "  RNF43: annual OGD from 30-35; colonoscopy every 1-3yr (SSA burden); "
            "  POLE: colonoscopy from 25 (polyposis risk); OGD every 3-5yr; TMB-guide immunotherapy; "
            "H. pylori eradication: recommended for ALL hereditary gastric cancer gene carriers "
            "  (RR reduction for intestinal-type gastric cancer with H. pylori eradication documented); "
            "Cascade testing: first-degree relatives of all CDH1/CTNNA1 pathogenic variant carriers mandatory"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Gastric-Cancer-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)

        gastric_n         = sum(1 for p in patients if p["gastric_cancer"])
        lobular_breast_n  = sum(1 for p in patients if p["lobular_breast_cancer"])
        breast_n          = sum(1 for p in patients if p["breast_cancer"])
        ovarian_n         = sum(1 for p in patients if p["ovarian_cancer"])
        pancreatic_n      = sum(1 for p in patients if p["pancreatic_cancer"])
        prostate_n        = sum(1 for p in patients if p["prostate_cancer"])
        colorectal_n      = sum(1 for p in patients if p["colorectal_cancer"])
        sarcoma_n         = sum(1 for p in patients if p["sarcoma"])
        brain_n           = sum(1 for p in patients if p["brain_tumour"])
        acc_n             = sum(1 for p in patients if p["acc"])
        signet_ring_n     = sum(1 for p in patients if p["signet_ring_histology"])
        intestinal_n      = sum(1 for p in patients if p["intestinal_histology"])
        gastric_polyps_n  = sum(1 for p in patients if p["gastric_polyps"])
        prophylactic_n    = sum(1 for p in patients if p["prophylactic_gastrectomy_done"])
        occult_src_n      = sum(1 for p in patients if p["occult_src_on_gastrectomy"])
        parp_n            = sum(1 for p in patients if p["parp_inhibitor_eligible"])
        tmb_high_n        = sum(1 for p in patients if p["tmb_high"])
        immuno_n          = sum(1 for p in patients if p["immunotherapy_response_excellent"])
        hpylori_n         = sum(1 for p in patients if p["hpylori_eradication_done"])
        wbmri_n           = sum(1 for p in patients if p["wbmri_surveillance"])
        severe_n          = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n        = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n            = sum(1 for p in patients if p["severity"] == "mild")
        mean_diag_age     = round(sum(p["age_at_diagnosis_yrs"] for p in patients) / n, 1)
        mutations_seen    = list({p["mutation"] for p in patients})

        clinical_notes = {
            "CDH1":  "PROPHYLACTIC TOTAL GASTRECTOMY mandatory -- negative endoscopy does NOT exclude risk (80-100% occult SRC). Lobular breast cancer 42-55% -- enhanced MRI from age 30. AVOID biopsy-based surveillance as sole management after age 25.",
            "CTNNA1":"HDGC without CDH1 -- same diffuse gastric histology and same gastrectomy protocol. CDH1-negative HDGC families require dedicated CTNNA1 testing. Fewer data than CDH1 but biologically equivalent predisposition.",
            "BRCA2": "Gastric 3-7x RR (INTESTINAL type -- not signet ring). H. pylori eradication MANDATORY for all BRCA2 carriers. PARP inhibitors (olaparib/rucaparib) FDA-approved for breast/ovarian/pancreatic/prostate. Biallelic FANCD1.",
            "PALB2": "Breast 35-60%; pancreatic 2-5x; gastric 2-3x. BRCA1-PALB2-BRCA2 HR complex -- PARP inhibitor sensitivity. Olaparib FDA2022 for early breast. H. pylori eradication recommended. Biallelic FANCN Fanconi.",
            "ATM":   "AVOID EXCESS RADIATION -- heterozygotes have partial radiosensitivity. Gastric 2-4x RR; breast 15-25% moderate; pancreatic 4-5x. HRD phenotype -- olaparib PROfound data for prostate; gastric trials ongoing. ~1% population carrier rate.",
            "TP53":  "AVOID RADIATION ABSOLUTELY -- LFS carriers develop radiation-induced secondary cancers. Whole-body MRI annually MANDATORY (Toronto protocol). Gastric is classical LFS tumour. Breast 25-50%; sarcoma 15-25%; ACC 15-20%.",
            "RNF43": "Gastric serrated polyposis -- Wnt pathway hypersensitivity (Frizzled receptor accumulation). Annual OGD from 30-35 for polyp surveillance. SSA gastric + colonic. Porcupine inhibitors for RNF43-LOF gastric cancer (trials). RSPO3 amplification co-occurs.",
            "POLE":  "ULTRA-HYPERMUTATED: TMB >100 mut/Mb but MSS on standard MSI testing. Hotspots P286R, V411L, L424V, S459F -- proofreading disabled. EXCEPTIONAL immunotherapy response. Test TMB directly (NOT MSI alone). Pembrolizumab FDA2020 TMB-high.",
        }

        genes_data.append({
            "gene":                      gene,
            "locus":                     gene_info["locus"],
            "n":                         n,
            "n_patients":                n,
            "severe_pct":                round(severe_n / n * 100, 1),
            "moderate_pct":              round(moderate_n / n * 100, 1),
            "mild_pct":                  round(mild_n / n * 100, 1),
            "gastric_cancer_pct":        round(gastric_n / n * 100, 1),
            "lobular_breast_pct":        round(lobular_breast_n / n * 100, 1),
            "breast_cancer_pct":         round(breast_n / n * 100, 1),
            "ovarian_cancer_pct":        round(ovarian_n / n * 100, 1),
            "pancreatic_cancer_pct":     round(pancreatic_n / n * 100, 1),
            "prostate_cancer_pct":       round(prostate_n / n * 100, 1),
            "colorectal_cancer_pct":     round(colorectal_n / n * 100, 1),
            "sarcoma_pct":               round(sarcoma_n / n * 100, 1),
            "brain_tumour_pct":          round(brain_n / n * 100, 1),
            "acc_pct":                   round(acc_n / n * 100, 1),
            "signet_ring_pct":           round(signet_ring_n / n * 100, 1),
            "intestinal_histology_pct":  round(intestinal_n / n * 100, 1),
            "gastric_polyps_pct":        round(gastric_polyps_n / n * 100, 1),
            "prophylactic_gastrectomy_pct": round(prophylactic_n / n * 100, 1),
            "occult_src_pct":            round(occult_src_n / n * 100, 1),
            "parp_eligible_pct":         round(parp_n / n * 100, 1),
            "tmb_high_pct":              round(tmb_high_n / n * 100, 1),
            "immunotherapy_response_pct":round(immuno_n / n * 100, 1),
            "hpylori_eradication_pct":   round(hpylori_n / n * 100, 1),
            "wbmri_surveillance_pct":    round(wbmri_n / n * 100, 1),
            "mean_age_dx_yrs":           mean_diag_age,
            "sample_mutations":          mutations_seen[:4],
            "protein":                   gene_info["protein"],
            "inheritance":               gene_info["inheritance"][:250],
            "disease_category":          gene_info["disease_category"],
            "clinical_note":             clinical_notes.get(gene, ""),
        })

    return {
        "atlas": "Hereditary-Gastric-Cancer-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Gastric-Cancer-Atlas."""
    definitions = [
        {
            "term": "CDH1-HDGC-Prophylactic-Gastrectomy-Decision",
            "definition": (
                "Hereditary Diffuse Gastric Cancer (HDGC) -- CDH1 pathogenic variant -- prophylactic gastrectomy decision: "
                "BACKGROUND: "
                "  CDH1 carriers: 40-83% lifetime gastric cancer risk; occult signet-ring cell (SRC) foci in 80-100% of prophylactic gastrectomy specimens; "
                "  Standard endoscopy (even with Cambridge protocol biopsies) CANNOT reliably detect these foci: "
                "    SRC cells lie in lamina propria, submucosa -- below endoscopic resolution; "
                "    Negative endoscopy does NOT reduce cancer risk -- it is a surveillance tool, NOT a safe harbour; "
                "RECOMMENDATION (IGCLC 2020 / HEREDITARY GI CANCER CONSORTIUM): "
                "  PROPHYLACTIC TOTAL GASTRECTOMY STRONGLY RECOMMENDED for all pathogenic CDH1/CTNNA1 carriers; "
                "  Age: typically offered 20-25 years (after reproductive counselling completion); "
                "  Timing: defer if planning pregnancy; earlier if strong family history of young-onset gastric cancer; "
                "PRE-GASTRECTOMY PROTOCOL (Cambridge Protocol): "
                "  Mandatory upper GI endoscopy 6-8 weeks pre-surgery: "
                "    Systematic biopsies: 6 regions (cardia, fundus, body, antrum x2 quadrants, pylorus); "
                "    Purpose: rule out advanced/metastatic cancer that changes surgical management; "
                "    Purpose: NOT to reassure or delay surgery; "
                "  CT chest/abdomen/pelvis: staging before surgery; "
                "  Nutritional assessment: pre-operative; "
                "SURGICAL TECHNIQUE: "
                "  R0 total gastrectomy with Roux-en-Y esophagojejunostomy; "
                "  Preserve spleen and pancreas where technically possible; "
                "  D1+ lymph node dissection (no extended D2 necessary for prophylactic intent); "
                "  Frozen section of proximal and distal margins; "
                "POST-GASTRECTOMY MANAGEMENT: "
                "  Vitamin B12 injections lifelong (intrinsic factor absent); "
                "  Oral iron supplementation + monitoring (non-haem iron absorption impaired); "
                "  Calcium and Vitamin D (post-gastrectomy bone disease); "
                "  Dumping syndrome: dietary management (small frequent meals, avoid simple sugars); "
                "  Nutritional review: every 6-12 months lifelong; "
                "  Weight loss: 10-15% typical; monitored; "
                "PATIENTS DECLINING GASTRECTOMY: "
                "  Cambridge protocol endoscopy annually: 30-100+ biopsies from all gastric regions; "
                "  ACCEPT: negative endoscopy is NOT reassurance -- continued cancer risk; "
                "  Intensive surveillance is a bridge to gastrectomy, not an alternative; "
                "  Psychosocial support and patient decision aids important."
            ),
        },
        {
            "term": "HDGC-Lobular-Breast-Cancer-CDH1-Surveillance",
            "definition": (
                "CDH1 carrier lobular breast cancer (LBC) surveillance: "
                "RISK: female CDH1 carriers have 42-55% lifetime risk of lobular breast cancer (ILC); "
                "PATHOGENESIS: E-cadherin (CDH1) is the defining molecular marker of ILC; "
                "  CDH1 LOF -> absence of E-cadherin -> classic ILC histology (single-file infiltration, discohesive cells); "
                "  CDH1 germline carriers develop lobular ILC rather than ductal (IDC); "
                "WHY ENHANCED IMAGING IS REQUIRED: "
                "  ILC is characteristically missed on mammography (linear growth pattern, equal density to parenchyma); "
                "  ILC has low sensitivity on USS; "
                "  ENHANCED MRI is the modality of choice for ILC surveillance; "
                "SURVEILLANCE RECOMMENDATIONS: "
                "  Annual breast MRI: from age 30 (or 5-10 years before youngest affected relative); "
                "  Annual clinical breast examination; "
                "  Consider mammography + MRI annually from age 40; "
                "  Do NOT rely on mammography alone in CDH1 carriers (insufficient for ILC detection); "
                "RISK-REDUCING SURGERY: "
                "  Prophylactic bilateral mastectomy: discussed after gastrectomy decision; "
                "  Reduces breast cancer risk >95%; "
                "  Reconstruction options discussed with plastic surgery; "
                "  Not routinely recommended before 25-30 but individual risk-benefit discussion; "
                "MALE CDH1 CARRIERS: "
                "  Male breast cancer risk elevated (smaller absolute risk); "
                "  Annual breast examination; mammography if glandular breast tissue present; "
                "CTNNA1 FEMALE CARRIERS: "
                "  Breast risk extrapolated from CDH1 data; "
                "  Same surveillance protocol as CDH1 carriers (pending better CTNNA1-specific data)."
            ),
        },
        {
            "term": "BRCA2-Gastric-Cancer-Helicobacter-Pylori-Protocol",
            "definition": (
                "BRCA2 carriers -- gastric cancer risk and H. pylori protocol: "
                "GASTRIC CANCER RISK IN BRCA2 CARRIERS: "
                "  Meta-analyses: BRCA2 carriers have approximately 3-7x relative risk (RR) of gastric cancer; "
                "  Absolute lifetime risk: ~3-8% (vs ~0.5-1% general population Western countries); "
                "  Higher absolute risk in East Asian BRCA2 carriers (higher background gastric cancer incidence); "
                "  Predominantly INTESTINAL TYPE gastric adenocarcinoma (not diffuse/signet ring like CDH1); "
                "  BRCA1 also associated with elevated gastric risk (though lesser data than BRCA2); "
                "HELICOBACTER PYLORI INTERACTION: "
                "  H. pylori infection: the major modifiable risk factor for intestinal-type gastric cancer; "
                "  H. pylori + BRCA2-LOF: additive (possibly synergistic) oncogenic risk; "
                "  Mechanism: H. pylori-induced chronic gastritis -> atrophy -> intestinal metaplasia -> dysplasia -> adenocarcinoma; "
                "  BRCA2-LOF cells in atrophic gastric epithelium: impaired HR -> accumulate somatic mutations accelerated; "
                "PROTOCOL (BRCA2 / PALB2 / ATM hereditary cancer carriers): "
                "  TEST ALL CARRIERS: urea breath test (UBT) or stool antigen test at baseline (pre-OGD); "
                "  TREAT H. PYLORI: standard triple therapy (PPI + clarithromycin + amoxicillin x7-14 days); "
                "    Test for eradication 4+ weeks after treatment completion; "
                "    Re-treat failure with quadruple therapy (bismuth + PPI + metronidazole + tetracycline); "
                "  POST-ERADICATION: confirm negative (UBT or stool Ag); "
                "  GASTRIC SURVEILLANCE: OGD with systematic biopsies every 3-5 years from age 40 (or 5 yr earlier than youngest family case); "
                "    Assess for intestinal metaplasia (IM) / dysplasia; "
                "    High-risk ethnicity (East Asian, South American, Eastern European): earlier surveillance from 35; "
                "PARP INHIBITOR INTERACTION: "
                "  Olaparib in BRCA2 gastric cancer: sensitivity established (BRCA2-LOF = HRD); "
                "  TOPAZ-1: not BRCA2-specific but BRCA2-LOF shows trend benefit; "
                "  Ongoing: BGB-290 (pamiparib) + chemotherapy BRCA2 gastric cancer trials."
            ),
        },
        {
            "term": "TP53-Li-Fraumeni-Radiation-Avoidance-WBMRI",
            "definition": (
                "Li-Fraumeni Syndrome (TP53) -- radiation avoidance and whole-body MRI protocol: "
                "RADIATION SENSITIVITY IN TP53 CARRIERS: "
                "  p53 function: critical G1/S checkpoint after ionising radiation (IR); "
                "  TP53-LOF carriers: G1 checkpoint absent -> irradiated cells proceed to replicate damaged DNA -> "
                "    chromosomal instability -> secondary radiation-induced cancers; "
                "  DOCUMENTED OUTCOME: TP53 carriers receiving radiation therapy (breast, bone, brain) have "
                "    substantially elevated risk of SECONDARY RADIATION-INDUCED MALIGNANCIES; "
                "  Angiosarcoma within radiation field reported; second primary sarcoma documented; "
                "CLINICAL IMPLICATIONS: "
                "  AVOID ALL IONISING RADIATION WHERE POSSIBLE: "
                "    No chest X-ray unless acutely required; "
                "    No CT scanning for surveillance (use MRI); "
                "    No PET-CT unless no MRI-based alternative; "
                "    No bone scintigraphy (nuclear medicine); "
                "  TREATMENT DECISIONS: "
                "    Breast cancer: avoid radiotherapy -- mastectomy (not lumpectomy + radiotherapy); "
                "    Brain tumours: radiosurgery only if absolutely no surgical alternative; "
                "    Sarcoma: surgery first; radiotherapy high-risk decision with oncology multidisciplinary; "
                "WHOLE-BODY MRI (WBMRI) -- TORONTO LFS PROTOCOL (Kim et al. 2011; Villani et al. 2016): "
                "  ANNUAL WBMRI from age of LFS diagnosis: "
                "    Scan from vertex of skull to mid-thigh; "
                "    Sequences: STIR (whole body fat-suppressed) + DWI (diffusion-weighted) for solid tumour detection; "
                "    Add IV gadolinium for brain + abdominal sequences; "
                "  ADDITIONAL ANNUAL SURVEILLANCE: "
                "    Breast MRI (annual from age 20-25 or 5yr before youngest affected relative); "
                "    Colonoscopy from age 25 (every 2-5yr); "
                "    OGD from age 25-30 (gastric LFS component); "
                "    Abdominal/pelvic USS: annual (adrenal, renal); "
                "    Clinical examination: dermatology + full systemic review; "
                "  WBMRI EVIDENCE: "
                "    Tumour detection rate: 7-10% annual yield in LFS adults (Villani 2016); "
                "    Stage shift: WBMRI detects earlier stage tumours vs symptom-based; "
                "GASTRIC CANCER IN LFS: "
                "  Gastric cancer is a recognised LFS/LFL (Li-Fraumeni-Like) tumour type; "
                "  OGD included in Toronto surveillance bundle (every 2-5yr); "
                "  H. pylori eradication: recommended for all LFS carriers."
            ),
        },
        {
            "term": "POLE-Ultra-Hypermutation-Immunotherapy-Protocol",
            "definition": (
                "POLE exonuclease domain germline mutations -- ultra-hypermutation and immunotherapy: "
                "MOLECULAR MECHANISM: "
                "  POLE exonuclease catalytic residues D275-E277 (motif DxE) + D368: essential for proofreading; "
                "  Germline hotspot mutations (P286R, V411L, L424V, S459F): INACTIVATE 3'->5' exonuclease; "
                "  Without proofreading: polymerase misincorporation rate increases 10-1000x -> "
                "    ULTRA-HYPERMUTATED phenotype (TMB typically 100-1000 mut/Mb); "
                "  C:A>T:A transversion signature + C>A substitutions characteristic; "
                "  Tumour neoantigens: ultra-high neoantigen load -> enhanced immunogenicity; "
                "CANCER SPECTRUM: "
                "  Colorectal cancer: polyposis (10-100 adenomas, oligo/attenuated FAP-like) + CRC; "
                "  Gastric cancer: intestinal type; ultra-high TMB; mean age 40-50yr; "
                "  Endometrial cancer: POLE ultramutated subtype (TCGA subtype 1) -- best prognosis; "
                "  Brain: glioma (pediatric and adult); "
                "  Rare: ovarian, pancreatic; "
                "MSI vs POLE DISTINCTION (CRITICAL): "
                "  POLE ultra-hypermutated tumours: MICROSATELLITE STABLE (MSS) on standard MSI testing; "
                "    MSI-PCR: microsatellite loci stable (POLE error = base substitutions, not insertion/deletions in microsatellites); "
                "    IHC MMR: all 4 proteins retained (proficient MMR); "
                "  Standard Lynch/MSI workflow WILL MISS POLE tumours; "
                "  TEST TMB DIRECTLY or sequence POLE exonuclease domain on all advanced gastric cancers; "
                "IMMUNOTHERAPY RESPONSE: "
                "  POLE ultra-hypermutated tumours: EXCEPTIONAL PD-1 responders; "
                "    High TMB -> high neoantigen load -> robust T-cell response; "
                "    Complete responses documented in POLE-driven tumours treated with pembrolizumab; "
                "  FDA2020 pembrolizumab approval for TMB-high (>10 mut/Mb) solid tumours (KEYNOTE-158); "
                "  POLE gastric cancer: pembrolizumab typically produces COMPLETE REMISSION in ultra-hypermutated subset; "
                "  Do NOT withhold immunotherapy because MSI testing is negative -- TEST TMB; "
                "GERMLINE TESTING ALGORITHM: "
                "  Somatic POLE mutation found: test germline for same hotspot variant; "
                "  Polyposis phenotype (10-100 adenomas) with negative APC/MUTYH: test POLE + POLD1; "
                "  Young-onset colorectal/gastric/endometrial with ultra-high TMB and MSS: POLE germline testing; "
                "POLD1: similar phenotype, POLD1 exonuclease domain mutations -- same concept, less data."
            ),
        },
        {
            "term": "Hereditary-Gastric-Cancer-Panel-Testing-Algorithm",
            "definition": (
                "Hereditary gastric cancer -- multi-gene panel testing algorithm (2024): "
                "INDICATIONS FOR PANEL TESTING: "
                "  Diffuse gastric cancer (DGC) or signet-ring cell carcinoma < 50yr; "
                "  DGC at any age + first/second degree relative with DGC; "
                "  DGC + personal/family history lobular breast cancer; "
                "  Intestinal-type gastric cancer < 40yr; "
                "  >= 2 first-degree relatives gastric cancer (any type) with 1 < 50yr; "
                "  Gastric polyposis (>=3 fundic gland or sessile serrated polyps); "
                "  Pathogenic variant in cancer gene where gastric is in the spectrum (BRCA2, ATM, TP53, STK11, Lynch); "
                "MINIMUM PANEL: CDH1 + CTNNA1 (diffuse histology); "
                "EXTENDED PANEL (recommended): CDH1, CTNNA1, BRCA1, BRCA2, PALB2, ATM, TP53, RNF43, POLE, POLD1, "
                "  MLH1, MSH2, MSH6, PMS2 (Lynch), STK11 (PJS), APC (GAPPS variant), SMAD4; "
                "INTERPRETATION CASCADE: "
                "  Pathogenic CDH1/CTNNA1: prophylactic gastrectomy discussion; "
                "  Pathogenic BRCA2/PALB2/ATM: PARP inhibitor eligibility; gastric surveillance + H. pylori eradication; "
                "  Pathogenic TP53: LFS protocol -- WBMRI annually + radiation avoidance; "
                "  Pathogenic RNF43: serrated polyposis protocol -- OGD + colonoscopy from 30-35; "
                "  Pathogenic POLE: TMB testing on tumour + immunotherapy evaluation; "
                "  Lynch syndrome: colonoscopy + endometrial surveillance protocol; "
                "VARIANTS OF UNCERTAIN SIGNIFICANCE (VUS): "
                "  Do NOT act on VUS for prophylactic surgery; "
                "  Multi-gene panel testing increases VUS yield -- pre-test genetic counselling essential; "
                "  VUS follow-up: ClinVar/InSiGHT/LOVD reclassification updates; "
                "FAMILY HISTORY APPROACH: "
                "  Unaffected relatives tested only after proband pathogenic variant confirmed; "
                "  Predictive testing: age 18+ for adult-onset syndromes; "
                "  Children: defer CDH1/CTNNA1 testing until teens (gastrectomy not offered before 18-20yr); "
                "GENETIC COUNSELLING: "
                "  Pre-test: implications for prophylactic surgery, insurance, family; "
                "  Post-test positive: management pathway, cascade testing, psychological support; "
                "  Psychological impact of prophylactic gastrectomy recommendation: profound -- multidisciplinary support."
            ),
        },
        {
            "term": "RNF43-Gastric-Serrated-Polyposis-Wnt-Protocol",
            "definition": (
                "RNF43 hereditary gastric serrated polyposis and Wnt pathway targeting: "
                "PATHOGENESIS: "
                "  RNF43 ubiquitinates Frizzled (FZD) receptors -> proteasomal degradation (Wnt off-state); "
                "  R-spondin (RSPO3) + LGR4/5: inhibit RNF43 E3 ligase -> FZD stabilised -> Wnt activated (normal on-state); "
                "  RNF43 germline LOF: constitutive FZD accumulation -> Wnt hypersensitivity even without ligand; "
                "  RSPO3 gene amplification (somatic): seen in ~5-10% RNF43-LOF gastric tumours (bypass mechanism); "
                "CLINICAL PHENOTYPE: "
                "  Gastric serrated polyposis: predominantly fundic and body gastric region; "
                "  Sessile serrated adenomas (SSA) in colon + gastric: similar to colonic serrated polyposis syndrome; "
                "  Gastric adenocarcinoma: intestinal subtype; may present late (arising from longstanding polyposis); "
                "  MSS gastric cancer (not dMMR); "
                "SURVEILLANCE PROTOCOL: "
                "  Upper GI endoscopy (OGD): annually from age 30-35; "
                "    Systematic examination of fundus, body, antrum, pylorus; "
                "    Targeted biopsies of sessile serrated-appearing lesions; "
                "    Remove SSAs where technically feasible endoscopically; "
                "    High-grade dysplasia or large (>1cm) lesions: consider surgery; "
                "  Colonoscopy: every 1-3 years (SSA burden determines interval); "
                "    Sessile serrated adenomas: thorough proximal colon examination; "
                "    SSA with dysplasia: 1yr; SSA no dysplasia: 1-3yr; no SSA: 3yr; "
                "WNTPATHWAY TARGETED THERAPY: "
                "  Porcupine inhibitors: block Wnt ligand acylation (required for Wnt secretion); "
                "    WNT974 (LGK-974): preclinical + Phase I for RNF43-LOF solid tumours; "
                "    LGX818: another porcupine inhibitor (BRAF-Wnt combination); "
                "  RSPO3-amplified RNF43-LOF tumours: most likely to respond to porcupine inhibition; "
                "  Tankyrase inhibitors (block AXIN degradation): Wnt downstream inhibition; "
                "  Biomarker strategy: RNF43 LOF + RSPO3 amplification = best biomarker-positive population; "
                "H. PYLORI: "
                "  Eradicate in all RNF43 carriers (modifiable risk factor for gastric cancer in polyposis context)."
            ),
        },
    ]
    return {
        "atlas": "Hereditary-Gastric-Cancer-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }
